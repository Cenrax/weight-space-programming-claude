"""Exp 1: can Claude tell which Atari 2600 game a PPO policy was trained on, from its raw weights alone?

Grading rules are pre-registered in experiments/exp1/GRADING.md.

    venv/bin/python experiments/exp1/run.py --dry-run        # print the prompt + token estimate, no API
    venv/bin/python experiments/exp1/run.py --pilot          # one call: main, breakout, attempt 1
    venv/bin/python experiments/exp1/run.py                  # all conditions (skips finished runs), 3 workers
    venv/bin/python experiments/exp1/run.py --condition permuted
    venv/bin/python experiments/exp1/run.py --grade-only     # re-grade logs, write summary.json, rebuild docs data
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from harness.claude_client import BudgetExhausted, ask  # noqa: E402
from harness.logging_utils import LOGS_DIR, write_result  # noqa: E402
from harness.weights_text import weights_to_text  # noqa: E402

EXP = "exp1"
LOG = LOGS_DIR / EXP
CKPT = REPO / "checkpoints" / "exp1"
BUDGET_USD = 20.0
EFFORT = "high"
MAX_TOKENS = 64000
PER_CALL_CAP_USD = 1.5
EST_FLOOR_USD = 0.30
QUANT = 0.001

NETS = ["breakout", "pong", "mspacman", "freeway"]
ENV_NAME = {"breakout": "Breakout", "pong": "Pong", "mspacman": "MsPacman", "freeway": "Freeway"}
PRETTY = {"breakout": "Breakout", "pong": "Pong", "mspacman": "Ms. Pac-Man", "freeway": "Freeway",
          "random": "untrained / random", "other": "other", "hedged": "hedged", "unparsed": "unparsed"}
ATTEMPTS = {"main": 5, "permuted": 2, "random_init": 2}
PERM_SEED0 = 4000
RANDOM_INITS = [(5001, "torch_default"), (5002, "ppo_orthogonal")]
ARCH = [128, 64, 64, 18]

ACTIONS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
           'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

SYSTEM = "You are an expert in machine learning and neural networks. Answer carefully and honestly."

PROMPT = """Below are the complete weights of a small neural network, printed as text.

This is a policy network trained with reinforcement learning (PPO) to play an Atari 2600 game.
Observation: the 128 bytes of the console's RAM, each divided by 255, so every input lies in [0, 1]. Input unit i is RAM byte i (console address 0x80 + i).
Actions: the full 18-action Arcade Learning Environment (ALE) action set, in the standard ALE order:
{actions}
Architecture: 128 inputs, then two hidden layers of 64 units each with tanh activations, then an output layer of 18 action logits. The action taken is the argmax of the 18 logits.

Format: "W<k> row i" lists the weights from input unit i of layer k to every unit of the next layer (the matrix is printed as [in, out]); "b<k>" is the bias vector of layer k. Values are printed as integers: the actual value is the integer times 0.001.

Which game was this network trained on?

No tools are allowed: answer by reading the weights below, without running any code.

It is also possible that the network was never trained; if you believe these are untrained / random weights, say so and answer "untrained / random weights".

Please give:
1. the game name,
2. your confidence (0 to 1) that your answer is correct,
3. the reasoning from the weights that led you there.

<weights>
{weights}</weights>

End your response with exactly one final line of this form (machine-readable):
ANSWER: {{"game": "<name>", "confidence": <number between 0 and 1>}}
"""


# ----------------------------------------------------------------------------- networks

def build_mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 18))


def load_net(name: str) -> dict[str, torch.Tensor]:
    return torch.load(CKPT / f"{name}_s0.pt", map_location="cpu", weights_only=True)


def permute_hidden(sd: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    t1, t2 = torch.as_tensor(rng.permutation(64)), torch.as_tensor(rng.permutation(64))
    out = {
        "0.weight": sd["0.weight"][t1], "0.bias": sd["0.bias"][t1],
        "2.weight": sd["2.weight"][t2][:, t1], "2.bias": sd["2.bias"][t2],
        "4.weight": sd["4.weight"][:, t2], "4.bias": sd["4.bias"].clone(),
    }
    return {k: v.clone().contiguous() for k, v in out.items()}


def verify_same_function(a: dict, b: dict, n: int = 4096) -> float:
    ma, mb = build_mlp(), build_mlp()
    ma.load_state_dict(a), mb.load_state_dict(b)
    x = torch.rand(n, 128, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return float((ma(x) - mb(x)).abs().max())


def random_init(seed: int, kind: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    m = build_mlp()
    if kind == "ppo_orthogonal":  # same as train/ppo.py init_layers(actor, last_std=0.01)
        lins = [l for l in m if isinstance(l, nn.Linear)]
        for i, l in enumerate(lins):
            nn.init.orthogonal_(l.weight, 0.01 if i == len(lins) - 1 else np.sqrt(2))
            nn.init.zeros_(l.bias)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def jobs(conditions: list[str], attempts: dict[str, int]) -> list[dict]:
    out = []
    for cond in conditions:
        if cond == "main":
            for net in NETS:
                sd = load_net(net)
                for a in range(1, attempts["main"] + 1):
                    out.append(dict(condition="main", group=net, attempt=a, sd=sd,
                                    label=f"{PRETTY[net]} · attempt {a}"))
        elif cond == "permuted":
            for i, net in enumerate(NETS):
                sd = load_net(net)
                pseed = PERM_SEED0 + i
                psd = permute_hidden(sd, pseed)
                diff = verify_same_function(sd, psd)
                assert diff < 1e-5, f"permutation changed the function of {net}: {diff}"
                assert not torch.equal(psd["0.weight"], sd["0.weight"])
                for a in range(1, attempts["permuted"] + 1):
                    out.append(dict(condition="permuted", group=net, attempt=a, sd=psd,
                                    label=f"{PRETTY[net]} (permuted) · attempt {a}",
                                    extra={"perm_seed": pseed, "perm_max_abs_logit_diff": diff}))
        elif cond == "random_init":
            for a, (seed, kind) in enumerate(RANDOM_INITS[:attempts["random_init"]], 1):
                out.append(dict(condition="random_init", group="random_init", attempt=a, sd=random_init(seed, kind),
                                label=f"Random init ({kind.replace('_', ' ')}) · seed {seed}",
                                extra={"init_seed": seed, "init_kind": kind}))
    return out


def run_tag(j: dict) -> str:
    return f"{j['condition']}-{j['group']}-a{j['attempt']}"


# ----------------------------------------------------------------------------- grading (GRADING.md)

PATTERNS = {
    "breakout": [r"(?<!super )\bbreakout\b"],
    "pong": [r"\bpong\b", r"video olympics"],
    "mspacman": [r"\bms\.?\s*-?\s*pac[\s-]*man\b", r"\bmspacman\b", r"\bmrs\.?\s*pac[\s-]*man\b"],
    "freeway": [r"\bfreeway\b"],
    "random": [r"untrained", r"random weights", r"random init", r"randomly initiali[sz]ed", r"not trained",
               r"random initiali[sz]ation", r"never trained"],
}


def norm(s: str) -> str:
    s = s.lower().replace("‑", "-").replace("–", "-").replace("—", "-").replace("‐", "-")
    return re.sub(r"\s+", " ", s).strip()


def classify(s: str | None) -> str:
    if not s:
        return "unparsed"
    n = norm(s)
    hits = {c for c, pats in PATTERNS.items() if any(re.search(p, n) for p in pats)}
    if not hits:
        return "other"
    if len(hits) == 1:
        return hits.pop()
    games = hits - {"random"}
    if "random" in hits and len(games) == 1 and re.search(r"\b(not|non)[ -](random|untrained|trained)", n):
        return games.pop()
    return "hedged"


def extract(text: str) -> dict:
    game, conf, parse = None, None, "none"
    for line in reversed(text.splitlines()):
        m = re.search(r"ANSWER:\s*(\{.*\})", line.replace("**", "").replace("`", ""))
        if m:
            try:
                d = json.loads(m.group(1))
                game, conf, parse = d.get("game"), d.get("confidence"), "answer_line"
                break
            except json.JSONDecodeError:
                continue
    if game is None:
        ms = re.findall(r"^\W*game\s*(?:name)?\s*[:=]\s*\**\s*([^\n]+)", text, flags=re.I | re.M)
        if ms:
            game, parse = ms[-1].strip(" *`\"'"), "fallback_label"
        cs = re.findall(r"confidence\s*[:=]\s*\**\s*([0-9]*\.?[0-9]+)", text, flags=re.I)
        if cs:
            conf = cs[-1]
    try:
        conf = None if conf is None else min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    return {"answer": game if game is not None else "unparsed", "answer_class": classify(game),
            "confidence": conf, "parse": parse}


def grade(condition: str, target: str, ex: dict) -> bool:
    if condition == "random_init":
        return ex["answer_class"] == "random" or (ex["confidence"] is not None and ex["confidence"] <= 0.30)
    return ex["answer_class"] == target


def excerpt(text: str, n: int = 800) -> str:
    body = "\n".join(l for l in text.splitlines() if not l.strip().replace("*", "").startswith("ANSWER:")).strip()
    if len(body) <= n:
        return body
    return body[:n - 1].rsplit(" ", 1)[0] + "…"


def grade_run(d: Path) -> dict | None:
    rp, resp = d / "result.json", d / "response.md"
    if not rp.exists() or not resp.exists():
        return None
    r = json.loads(rp.read_text())
    if r.get("exp_id") != EXP or r.get("status") not in ("complete", "refusal"):
        return None
    text = resp.read_text()
    ex = extract(text)
    net = r.get("network") or r["run_tag"].split("-")[1]
    correct = grade(r["condition"], net, ex)
    group = "Random init" if net == "random_init" else PRETTY[net]
    return write_result(d, network=net, group=group, answer=ex["answer"], answer_class=ex["answer_class"],
                        confidence=ex["confidence"], parse=ex["parse"], correct=correct,
                        reasoning_excerpt=excerpt(text), timestamp=r.get("timestamp"))


# ----------------------------------------------------------------------------- running

def spent() -> float:
    tot = 0.0
    for p in LOG.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def finished_tags() -> set[str]:
    tags = set()
    for p in LOG.glob("*/result.json"):
        r = json.loads(p.read_text())
        if r.get("status") == "complete" and r.get("run_tag"):
            tags.add(r["run_tag"])
    return tags


def make_prompt(sd: dict) -> str:
    acts = "\n".join(f"{i} {a}" for i, a in enumerate(ACTIONS))
    return PROMPT.format(actions=acts, weights=weights_to_text(sd, activation="tanh", quantize=QUANT))


def call(j: dict) -> dict:
    tag = run_tag(j)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + tag
    target = j["group"] if j["condition"] != "random_init" else "untrained / random"
    extra = {"target": target, "label": j["label"], "network": j["group"], "run_tag": tag,
             "attempt": j["attempt"], "effort": EFFORT, "quantize": QUANT, **(j.get("extra") or {})}
    r = ask(make_prompt(j["sd"]), exp_id=EXP, run_id=rid, system=SYSTEM, effort=EFFORT, max_tokens=MAX_TOKENS,
            condition=j["condition"], max_cost_usd=PER_CALL_CAP_USD, extra=extra)
    g = grade_run(r.log_dir) or {}
    print(f"[{tag}] {g.get('answer')!r} class={g.get('answer_class')} conf={g.get('confidence')} "
          f"correct={g.get('correct')} in={r.tokens_in} out={r.usage['output_tokens']} ${r.cost_usd:.3f} "
          f"stop={r.stop_reason}", flush=True)
    return {"cost": r.cost_usd, **g}


def run_all(js: list[dict], workers: int = 3) -> None:
    done = finished_tags()
    todo = [j for j in js if run_tag(j) not in done]
    print(f"{len(js)} jobs, {len(js) - len(todo)} already done, {len(todo)} to run; spent ${spent():.3f} "
          f"(cap ${BUDGET_USD})", flush=True)
    costs = [float(json.loads(p.read_text()).get("cost_usd") or 0) for p in LOG.glob("*/result.json")]
    est = max(costs + [EST_FLOOR_USD])
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex:
        inflight: set = set()
        while todo or inflight:
            while todo and len(inflight) < workers:
                if spent() + (len(inflight) + 1) * est > BUDGET_USD:
                    print(f"BUDGET: spent ${spent():.3f} + {len(inflight) + 1} x est ${est:.3f} would exceed "
                          f"${BUDGET_USD}; not launching {len(todo)} more", flush=True)
                    todo = []
                    break
                inflight.add(ex.submit(call, todo.pop(0)))
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in fin:
                try:
                    res = f.result()
                    with lock:
                        est = max(est, res["cost"])
                except BudgetExhausted as e:
                    print(f"PROJECT BUDGET EXHAUSTED: {e}", flush=True)
                    todo = []
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    print(f"exp1 spend ${spent():.3f}", flush=True)


# ----------------------------------------------------------------------------- summary

SHAPE_KW = ["128-byte", "128 byte", "18 action", "18-action", "full action set", "full action space"]
RAM_RE = re.compile(r"\b(?:byte|ram\[?|row|input|address)\s*#?\s*\d{1,3}\b|0x[89a-f][0-9a-f]\b", re.I)


def frac(rs: list[dict]) -> str:
    return f"{sum(bool(r['correct']) for r in rs)}/{len(rs)}"


def display(r: dict) -> str:
    c = r["answer_class"]
    return str(r["answer"]) if c in ("other", "hedged") else PRETTY.get(c, c)


def eval_scores() -> dict[str, tuple[float, float]]:
    rb = json.loads((CKPT / "random_baseline.json").read_text())["results"]
    out = {}
    for n in NETS:
        m = json.loads((CKPT / f"{n}_s0.json").read_text())
        out[n] = (m["eval_mean_return"], rb[ENV_NAME[n]]["mean_return"])
    return out


def summarize() -> None:
    runs = []
    for d in sorted(LOG.iterdir()) if LOG.exists() else []:
        if d.is_dir() and not d.name.startswith("_"):
            g = grade_run(d)
            if g:
                t = (d / "response.md").read_text()
                g["_ram_cites"] = len(set(m.group(0).lower() for m in RAM_RE.finditer(t)))
                runs.append(g)
    by = lambda c, n: [r for r in runs if r["condition"] == c and r["network"] == n]  # noqa: E731
    main = [r for r in runs if r["condition"] == "main"]
    perm = [r for r in runs if r["condition"] == "permuted"]
    rnd = [r for r in runs if r["condition"] == "random_init"]
    trained = main + perm
    scores = eval_scores()

    rows, main_acc, perm_acc, read = [], [], [], []
    for n in NETS:
        m, p = by("main", n), by("permuted", n)
        confs = [r["confidence"] for r in m if r["confidence"] is not None]
        wrong = Counter(display(r) for r in m + p if not r["correct"])
        others = [r for r in main if r["network"] != n]
        spec = sum(r["answer_class"] == n for r in others)
        k = sum(bool(r["correct"]) for r in m)
        is_read = k >= 3 and spec <= 5
        if is_read:
            read.append(PRETTY[n])
        sc, rb = scores[n]
        rows.append([PRETTY[n], frac(m), frac(p), f"{np.mean(confs):.2f}" if confs else "—",
                     f"{wrong.most_common(1)[0][0]} ({wrong.most_common(1)[0][1]})" if wrong else "—",
                     f"{sc:g} vs {rb:g}",
                     f"{spec}/{len(others)}"])
        main_acc.append(round(100 * k / len(m), 1) if m else None)
        perm_acc.append(round(100 * sum(bool(r["correct"]) for r in p) / len(p), 1) if p else None)
    rows.append(["Random init (control)", f"{frac(rnd)} correct: " + "; ".join(
        f"{r['answer']} ({r['confidence']})" for r in rnd) if rnd else "—", "", "", "", "", ""])
    rows.append(["Total", frac(main), frac(perm), "", "", "", f"cost ${spent():.2f}"])

    answers = Counter(display(r) for r in trained)
    top, top_n = answers.most_common(1)[0] if answers else ("—", 0)
    verdict = ("reads game identity from weights" if len(read) >= 3 else "partial" if len(read) == 2 else "not shown")
    decision = {"games_read": read, "verdict": verdict, "default_guess_flag": top_n >= 14,
                "most_common_answer": f"{top} ({top_n}/{len(trained)})",
                "answer_counts_trained": dict(answers)}
    cc = [r["confidence"] for r in trained if r["correct"] and r["confidence"] is not None]
    wc = [r["confidence"] for r in trained if not r["correct"] and r["confidence"] is not None]
    decision["mean_conf_correct"] = round(float(np.mean(cc)), 2) if cc else None
    decision["mean_conf_wrong"] = round(float(np.mean(wc)), 2) if wc else None

    summary = (f"With no tools and no candidate list, Claude Opus 5.5 named the right Atari game from the raw weights "
               f"in {frac(main)} main and {frac(perm)} hidden-unit-permuted attempts (games meeting the "
               f"pre-registered >=3/5 bar: {', '.join(read) if read else 'none'}; verdict: {verdict}; most common "
               f"answer {top} in {top_n}/{len(trained)}), and called {frac(rnd)} random-init controls untrained, "
               f"for ${spent():.2f} in API cost.")
    media = []
    for n in NETS:
        src = f"assets/gifs/exp1_{n}_s0.gif"
        if (REPO / "docs" / src).exists():
            sc, rb = scores[n]
            media.append({"type": "gif", "src": src, "group": PRETTY[n],
                          "caption": f"The {PRETTY[n]} network (seed 0) playing greedily. Mean eval return {sc:g} "
                                     f"over 100 episodes vs {rb:g} for a uniform-random policy."})
    out = {
        "summary": summary,
        "table": {"caption": "Correct / attempts per network, no tools, no candidate list; grading pre-registered in "
                             "experiments/exp1/GRADING.md. 'Specificity' = how often this game was the answer on the "
                             "other three networks' main attempts (must be <= 5/15 to count as read).",
                  "columns": ["Network", "Main correct", "Permuted correct", "Mean confidence (main)",
                              "Most common wrong answer", "Eval return vs random", "Specificity"],
                  "rows": rows},
        "charts": [{"type": "bar", "title": "Which Atari game? Accuracy per network",
                    "subtitle": "5 main + 2 permuted attempts per network; no tools, no candidate list",
                    "unit": "% correct", "categories": [PRETTY[n] for n in NETS],
                    "series": [{"name": "Main", "values": main_acc}, {"name": "Permuted", "values": perm_acc}],
                    "caption": "Permuting hidden units leaves the function unchanged. A guesser among these four "
                               "games would score 25%."}],
        "media": media,
        "status": "done" if runs and len(main) == 20 and len(perm) == 8 and len(rnd) == 2 else "running",
        "total_cost_usd": round(spent(), 4),
        "decision": decision,
    }
    (LOG / "summary.json").write_text(json.dumps(out, indent=2))
    for row in rows:
        print(row)
    print(out["summary"])
    print("decision:", json.dumps(decision))
    subprocess.run([sys.executable, str(REPO / "scripts" / "build_blog_data.py"), "--exp", EXP], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="one call: main, breakout, attempt 1")
    ap.add_argument("--condition", choices=["main", "permuted", "random_init", "all"], default="all")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if a.grade_only:
        summarize()
        return
    conds = ["main", "permuted", "random_init"] if a.condition == "all" else [a.condition]
    js = jobs(conds, ATTEMPTS)
    if a.pilot:
        js = [j for j in jobs(["main"], ATTEMPTS) if j["group"] == "breakout" and j["attempt"] == 1]
    if a.dry_run:
        from harness.weights_text import estimate_tokens
        p = make_prompt(js[0]["sd"])
        print(p[:2400], "...\n", p[-300:])
        print(f"{len(js)} jobs; ~{estimate_tokens(p)} prompt tokens each")
        for j in js:
            if "perm_max_abs_logit_diff" in (j.get("extra") or {}):
                print(run_tag(j), "perm diff", j["extra"]["perm_max_abs_logit_diff"])
        return
    run_all(js, workers=1 if a.pilot else a.workers)
    summarize()


if __name__ == "__main__":
    main()
