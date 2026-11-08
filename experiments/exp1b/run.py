"""Exp 1b: can Claude tell which RL benchmark a policy network was trained on, from its raw weights alone?

Grading rules are pre-registered in experiments/exp1b/GRADING.md.

    venv/bin/python experiments/exp1b/run.py --pilot                 # one call: main, cartpole, attempt 1
    venv/bin/python experiments/exp1b/run.py                         # all conditions (skips finished runs)
    venv/bin/python experiments/exp1b/run.py --condition permuted --attempts 2
    venv/bin/python experiments/exp1b/run.py --grade-only            # re-grade logs, write summary, no API calls
    venv/bin/python experiments/exp1b/run.py --dry-run               # print the prompt + token estimate
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

EXP = "exp1b"
LOG = LOGS_DIR / EXP
# Per-version settings. v1: 8->32->32->4 (pad8x4). v2: 10->32->32->5 (pad10x5), see GRADING.md "v2".
VERSIONS = {
    "v1": dict(ckpt=REPO / "checkpoints" / "exp1b", n_in=8, n_out=4, suffix="", group_suffix="", budget=10.0,
               perm_seed0=2000, random_seeds=[1001, 1002, 1003]),
    "v2": dict(ckpt=REPO / "checkpoints" / "exp1b_v2", n_in=10, n_out=5, suffix="_v2", group_suffix=" (v2)",
               budget=5.0, perm_seed0=3000, random_seeds=[2001, 2002, 2003]),
}
CFG = VERSIONS["v1"]
EFFORT = "high"
MAX_TOKENS = 64000
PER_CALL_CAP_USD = 1.5

NETS = ["cartpole", "acrobot", "mountaincar", "lunarlander"]
PRETTY = {"cartpole": "CartPole", "acrobot": "Acrobot", "mountaincar": "MountainCar", "lunarlander": "LunarLander",
          "random": "untrained / random", "other": "other", "unparsed": "unparsed"}
ATTEMPTS = {"main": 5, "permuted": 2, "random_init": 3}

SYSTEM = "You are an expert in machine learning and neural networks. Answer carefully and honestly."

PROMPT = """Below are the complete weights of a small neural network, printed as text.

This is a policy network trained with reinforcement learning on a widely used RL benchmark environment.
Architecture: an input vector of dimension {n_in}, then two hidden layers of 32 units each with tanh activations, then an output layer of {n_out} action logits. The action taken is the argmax of the {n_out} logits.

Format: "W<k> row i" lists the weights from input unit i of layer k to every unit of the next layer (the matrix is printed as [in, out]); "b<k>" is the bias vector of layer k. Values are rounded to 3 decimals.

Which environment was this network trained on?

No tools are allowed: answer by reading the weights below, without running any code.

It is also possible that the network was never trained; if you believe these are untrained / random weights, say so and answer "untrained / random weights".

Please give:
1. the environment name,
2. your confidence (0 to 1) that your answer is correct,
3. the reasoning from the weights that led you there.

<weights>
{weights}</weights>

End your response with exactly one final line of this form (machine-readable):
ANSWER: {{"environment": "<name>", "confidence": <number between 0 and 1>}}
"""


# ----------------------------------------------------------------------------- networks

def build_mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(CFG["n_in"], 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, CFG["n_out"]))


def load_net(name: str) -> dict[str, torch.Tensor]:
    return torch.load(CFG["ckpt"] / f"{name}.pt", map_location="cpu")


def permute_hidden(sd: dict[str, torch.Tensor], seed: int) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    p1, p2 = rng.permutation(32), rng.permutation(32)
    t1, t2 = torch.as_tensor(p1), torch.as_tensor(p2)
    out = {
        "0.weight": sd["0.weight"][t1], "0.bias": sd["0.bias"][t1],
        "2.weight": sd["2.weight"][t2][:, t1], "2.bias": sd["2.bias"][t2],
        "4.weight": sd["4.weight"][:, t2], "4.bias": sd["4.bias"].clone(),
    }
    return {k: v.clone().contiguous() for k, v in out.items()}, p1.tolist(), p2.tolist()


def verify_same_function(a: dict, b: dict, n: int = 4096) -> float:
    ma, mb = build_mlp(), build_mlp()
    ma.load_state_dict(a), mb.load_state_dict(b)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, CFG["n_in"], generator=g) * 2.0
    with torch.no_grad():
        return float((ma(x) - mb(x)).abs().max())


def random_init(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_mlp().state_dict().items()}


def jobs(conditions: list[str], attempts: dict[str, int]) -> list[dict]:
    """conditions are base names (main/permuted/random_init); the version suffix is added to the logged condition."""
    out = []
    sfx, gs = CFG["suffix"], CFG["group_suffix"]
    for cond in conditions:
        if cond == "main":
            for i, net in enumerate(NETS):
                sd = load_net(net)
                for a in range(1, attempts["main"] + 1):
                    out.append(dict(condition="main" + sfx, group=net, attempt=a, sd=sd,
                                    label=f"{PRETTY[net]}{gs} · attempt {a}"))
        elif cond == "permuted":
            for i, net in enumerate(NETS):
                sd = load_net(net)
                pseed = CFG["perm_seed0"] + i
                psd, p1, p2 = permute_hidden(sd, pseed)
                diff = verify_same_function(sd, psd)
                assert diff < 1e-5, f"permutation changed the function of {net}: {diff}"
                assert not torch.equal(psd["0.weight"], sd["0.weight"])
                for a in range(1, attempts["permuted"] + 1):
                    out.append(dict(condition="permuted" + sfx, group=net, attempt=a, sd=psd,
                                    label=f"{PRETTY[net]}{gs} (permuted) · attempt {a}",
                                    extra={"perm_seed": pseed, "perm_max_abs_logit_diff": diff}))
        elif cond == "random_init":
            for a, seed in enumerate(CFG["random_seeds"][:attempts["random_init"]], 1):
                out.append(dict(condition="random_init" + sfx, group="random_init", attempt=a, sd=random_init(seed),
                                label=f"Random init{gs} · seed {seed}", extra={"init_seed": seed}))
    return out


def run_tag(j: dict) -> str:
    return f"{j['condition']}-{j['group']}-a{j['attempt']}"


# ----------------------------------------------------------------------------- grading

ALIASES = {
    "cartpole": ["cartpole", "cart pole", "inverted pendulum on a cart", "pole balancing", "pole on a cart",
                 "balancing a pole", "invertedpendulum"],
    "acrobot": ["acrobot", "two link pendulum swing", "double pendulum swing", "two link robot arm swing"],
    "mountaincar": ["mountaincar", "mountain car"],
    "lunarlander": ["lunarlander", "lunar lander", "lunar landing", "moon lander"],
    "random": ["untrained", "random weights", "random init", "randomly initialized", "randomly initialised",
               "not trained", "random initialization", "random initialisation"],
}
# Matching is also done with all spaces removed, so "inverted pendulum" hits "invertedpendulum" (MuJoCo
# InvertedPendulum = cart-pole, pre-registered). Plain "Pendulum-v1" is excluded explicitly in classify().


def norm(s: str) -> str:
    s = s.lower().replace("‑", "-").replace("–", "-")
    return re.sub(r"[\s_\-/]+", " ", s).strip()


def classify(s: str | None, last: bool = False) -> str:
    if not s:
        return "unparsed"
    n = norm(s)
    if not last and re.search(r"\bpendulum v\d", n) and "inverted" not in n:
        return "other"
    hits = []
    for cls, al in ALIASES.items():
        for a in al:
            for hay, needle in ((n, a), (n.replace(" ", ""), a.replace(" ", ""))):
                idx = hay.rfind(needle) if last else hay.find(needle)
                if idx >= 0:
                    # positions in the space-free string are not comparable; scale roughly
                    pos = idx if hay is n else idx * len(n) / max(1, len(hay))
                    hits.append((pos, cls))
    if not hits:
        return "other"
    hits.sort()
    return hits[-1][1] if last else hits[0][1]


def extract(text: str) -> dict:
    env, conf, parse = None, None, "none"
    for line in reversed(text.splitlines()):
        m = re.search(r"ANSWER:\s*(\{.*\})", line.replace("**", ""))
        if m:
            try:
                d = json.loads(m.group(1))
                env, conf, parse = d.get("environment"), d.get("confidence"), "answer_line"
                break
            except json.JSONDecodeError:
                continue
    if env is None:
        ms = re.findall(r"environment\s*[:=]\s*\**\s*([^\n]+)", text, flags=re.I)
        if ms:
            env, parse = ms[-1].strip(" *`\"'"), "fallback_label"
        cs = re.findall(r"confidence\s*[:=]\s*\**\s*([0-9]*\.?[0-9]+)", text, flags=re.I)
        if cs:
            conf = cs[-1]
    try:
        conf = None if conf is None else min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    cls = classify(env) if env is not None else "unparsed"
    if env is None:
        c2 = classify(text[-1500:], last=True)
        if c2 not in ("other", "unparsed"):
            cls, env, parse = c2, PRETTY[c2], "fallback_text"
    return {"answer": env if env is not None else "unparsed", "answer_class": cls, "confidence": conf, "parse": parse}


def grade(condition: str, target: str, ex: dict) -> bool:
    if condition.startswith("random_init"):
        return ex["answer_class"] == "random" or (ex["confidence"] is not None and ex["confidence"] <= 0.30)
    return ex["answer_class"] == target


def excerpt(text: str, n: int = 800) -> str:
    body = "\n".join(l for l in text.splitlines() if not l.strip().replace("*", "").startswith("ANSWER:")).strip()
    if len(body) <= n:
        return body
    cut = body[:n - 1].rsplit(" ", 1)[0]
    return cut + "…"


def grade_run(d: Path) -> dict | None:
    rp = d / "result.json"
    resp = d / "response.md"
    if not rp.exists() or not resp.exists():
        return None
    r = json.loads(rp.read_text())
    if r.get("exp_id") != EXP or r.get("status") not in ("complete", "refusal"):
        return None
    text = resp.read_text()
    ex = extract(text)
    tgt = r.get("target")
    correct = grade(r["condition"], tgt, ex)
    net = r.get("network") or r["run_tag"].split("-")[1]  # run_tag = <condition>-<network>-a<n>
    gs = " (v2)" if r["condition"].endswith("_v2") else ""
    group = ("Random init" if net == "random_init" else PRETTY[net]) + gs
    return write_result(d, network=net, group=group, answer=ex["answer"], answer_class=ex["answer_class"], confidence=ex["confidence"],
                        parse=ex["parse"], correct=correct, reasoning_excerpt=excerpt(text),
                        timestamp=r.get("timestamp"))


# ----------------------------------------------------------------------------- running

def version_of(cond: str | None) -> str:
    return "v2" if (cond or "").endswith("_v2") else "v1"


def spent(version: str | None = None) -> float:
    """Summed cost of exp1b runs (all versions, or only `version`)."""
    tot = 0.0
    for p in LOG.glob("*/result.json"):
        try:
            r = json.loads(p.read_text())
            if version and version_of(r.get("condition")) != version:
                continue
            tot += float(r.get("cost_usd") or 0)
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
    return PROMPT.format(weights=weights_to_text(sd, activation="tanh"), n_in=CFG["n_in"], n_out=CFG["n_out"])


def call(j: dict) -> dict:
    tag = run_tag(j)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + tag
    prompt = make_prompt(j["sd"])
    target = j["group"] if not j["condition"].startswith("random_init") else "untrained / random"
    extra = {"target": target, "label": j["label"], "network": j["group"], "run_tag": tag,
             "attempt": j["attempt"], "effort": EFFORT, "version": version_of(j["condition"]),
             **(j.get("extra") or {})}
    r = ask(prompt, exp_id=EXP, run_id=rid, system=SYSTEM, effort=EFFORT, max_tokens=MAX_TOKENS,
            condition=j["condition"], max_cost_usd=PER_CALL_CAP_USD, extra=extra)
    g = grade_run(r.log_dir) or {}
    print(f"[{tag}] {g.get('answer')!r} conf={g.get('confidence')} correct={g.get('correct')} "
          f"in={r.tokens_in} out={r.usage['output_tokens']} ${r.cost_usd:.3f} stop={r.stop_reason}", flush=True)
    return {"cost": r.cost_usd, **g}


def run_all(js: list[dict], workers: int = 4) -> None:
    done = finished_tags()
    todo = [j for j in js if run_tag(j) not in done]
    ver, cap = version_of(js[0]["condition"]) if js else "v1", CFG["budget"]
    print(f"{len(js)} jobs, {len(js) - len(todo)} already done, {len(todo)} to run; {ver} spent so far "
          f"${spent(ver):.3f} (cap ${cap})")
    worst = max([float(json.loads(p.read_text()).get("cost_usd") or 0) for p in LOG.glob("*/result.json")] + [0.2])
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex:
        inflight = set()
        while todo or inflight:
            while todo and len(inflight) < workers:
                if spent(ver) + (len(inflight) + 1) * worst > cap:
                    print(f"BUDGET: {ver} spent ${spent(ver):.3f} + {len(inflight) + 1} x worst ${worst:.3f} would "
                          f"exceed ${cap}; not launching more", flush=True)
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
                        worst = max(worst, res["cost"])
                except BudgetExhausted as e:
                    print(f"PROJECT BUDGET EXHAUSTED: {e}", flush=True)
                    todo = []
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    print(f"{ver} spend ${spent(ver):.3f}; total exp1b spend ${spent():.3f}")


# ----------------------------------------------------------------------------- summary

PAD_KEYWORDS = ["pad", "always zero", "always-zero", "unused input", "never used", "constant zero", "zero-padded"]


def mentions_padding(d: Path) -> bool:
    t = (d / "response.md").read_text().lower() if (d / "response.md").exists() else ""
    return any(k in t for k in PAD_KEYWORDS)


def display(r: dict) -> str:
    """Display label for an answer (reporting only; grading uses answer_class)."""
    if r["answer_class"] == "other":
        n = norm(str(r["answer"]))
        return "MPE simple_adversary" if "simple adversary" in n or "simpleadversary" in n else str(r["answer"])
    return PRETTY.get(r["answer_class"], r["answer_class"])


def frac(rs: list[dict]) -> str:
    return f"{sum(bool(r['correct']) for r in rs)}/{len(rs)}"


def summarize() -> None:
    runs = []
    for d in sorted(LOG.iterdir()) if LOG.exists() else []:
        if d.is_dir() and not d.name.startswith("_"):
            g = grade_run(d)
            if g:
                g["_pad"] = mentions_padding(d)
                runs.append(g)
    by = lambda c, n: [r for r in runs if r["condition"] == c and r["network"] == n]  # noqa: E731

    def stats(ver: str) -> dict:
        sfx = VERSIONS[ver]["suffix"]
        st = {"main_acc": [], "perm_acc": [], "per_net": {}}
        for n in NETS:
            m, p = by("main" + sfx, n), by("permuted" + sfx, n)
            confs = [r["confidence"] for r in m if r["confidence"] is not None]
            wrong = Counter(display(r) for r in m + p if not r["correct"])
            st["per_net"][n] = dict(main=m, perm=p, conf=f"{np.mean(confs):.2f}" if confs else "—",
                                    wrong=f"{wrong.most_common(1)[0][0]} ({wrong.most_common(1)[0][1]})" if wrong else "—")
            st["main_acc"].append(round(100 * sum(r["correct"] for r in m) / len(m), 1) if m else None)
            st["perm_acc"].append(round(100 * sum(r["correct"] for r in p) / len(p), 1) if p else None)
        st["main"] = [r for r in runs if r["condition"] == "main" + sfx]
        st["perm"] = [r for r in runs if r["condition"] == "permuted" + sfx]
        st["rnd"] = [r for r in runs if r["condition"] == "random_init" + sfx]
        st["trained"] = st["main"] + st["perm"]
        st["answers"] = Counter(r["answer_class"] for r in st["trained"])
        st["answers_display"] = Counter(display(r) for r in st["trained"])
        st["pad"] = sum(r["_pad"] for r in st["trained"])
        st["cost"] = spent(ver)
        return st

    v1, v2 = stats("v1"), stats("v2")
    has_v2 = bool(v2["main"])
    rows = []
    for n in NETS:
        a, b = v1["per_net"][n], v2["per_net"][n]
        rows.append([PRETTY[n], frac(a["main"]), frac(a["perm"]), a["wrong"],
                     frac(b["main"]) if b["main"] else "—", frac(b["perm"]) if b["perm"] else "—", b["conf"], b["wrong"]])
    def rnd_cell(st):
        return (f"{frac(st['rnd'])} said untrained/low-conf: " + ", ".join(f"{r['answer']} ({r['confidence']})"
                for r in st["rnd"])) if st["rnd"] else "—"
    rows.append(["Random init (control)", rnd_cell(v1), "", "", rnd_cell(v2), "", "", ""])
    rows.append(["Total", frac(v1["main"]), frac(v1["perm"]), f"cost ${v1['cost']:.2f}",
                 frac(v2["main"]) if has_v2 else "—", frac(v2["perm"]) if has_v2 else "—", "",
                 f"cost ${v2['cost']:.2f}"])

    # v2 pre-registered decision rule (GRADING.md, v2 section)
    decision = None
    if has_v2:
        tot = sum(r["correct"] for r in v2["main"])
        non_ll = [n for n in NETS if n != "lunarlander"
                  if sum(r["correct"] for r in v2["per_net"][n]["main"]) >= 3]
        top, top_n = v2["answers_display"].most_common(1)[0]
        tr = v2["trained"]
        pad_c = sum(r["_pad"] and r["correct"] for r in tr)
        nopad_c = sum((not r["_pad"]) and r["correct"] for r in tr)
        n_pad = sum(r["_pad"] for r in tr)
        cc = [r["confidence"] for r in tr if r["correct"] and r["confidence"] is not None]
        wc = [r["confidence"] for r in tr if not r["correct"] and r["confidence"] is not None]
        passed = tot >= 10 and len(non_ll) >= 2
        decision = {"main_v2_correct": tot, "non_lunarlander_nets_ge_3of5": [PRETTY[n] for n in non_ll],
                    "evidence_beyond_shape": passed,
                    "default_guess_failure": top_n >= 15, "most_common_answer": f"{top} ({top_n}/{len(tr)})",
                    "padding_mentioned": f"{n_pad}/{len(tr)}",
                    "correct_when_padding_mentioned": f"{pad_c}/{n_pad}",
                    "correct_when_not_mentioned": f"{nopad_c}/{len(tr) - n_pad}",
                    "mean_conf_correct": round(float(np.mean(cc)), 2) if cc else None,
                    "mean_conf_wrong": round(float(np.mean(wc)), 2) if wc else None}

    s1 = (f"v1 (all nets padded to 8 inputs/4 actions, LunarLander's native shape): Claude Opus 5.5 named the right "
          f"environment from the raw weights in {frac(v1['main'])} main and {frac(v1['perm'])} permuted attempts, "
          f"answering LunarLander in {v1['answers']['lunarlander']}/{len(v1['trained'])}.")
    if has_v2:
        s2 = (f" v2 (padded to 10 inputs/5 actions, chosen to avoid any classic-control shape, but it turned out to "
              f"match PettingZoo MPE simple_adversary's good agent): {frac(v2['main'])} main and {frac(v2['perm'])} "
              f"permuted correct, most common answer {decision['most_common_answer']} at low confidence "
              f"(mean {decision['mean_conf_wrong']} on wrong answers vs {decision['mean_conf_correct']} on right ones). "
              f"Every response that inferred padding was right ({decision['correct_when_padding_mentioned']}) and "
              f"every one that did not was wrong ({decision['correct_when_not_mentioned']}). The pre-registered "
              f"bar for reading the weights beyond the shape (>= 10/20 and >= 3/5 on two non-LunarLander nets) was "
              f"{'met' if decision['evidence_beyond_shape'] else 'not met'}.")
    else:
        s2 = ""
    s3 = (f" Random-init controls were called untrained in {frac(v1['rnd'])} (v1)"
          + (f" and {frac(v2['rnd'])} (v2)" if v2["rnd"] else "") + f". Total API cost ${spent():.2f}.")
    out = {
        "summary": s1 + s2 + s3,
        "table": {"caption": "Correct / attempts per network, no tools, no candidate list. Grading pre-registered in "
                             "experiments/exp1b/GRADING.md (v1 and v2 sections).",
                  "columns": ["Network", "v1 main", "v1 permuted", "v1 most common wrong answer",
                              "v2 main", "v2 permuted", "v2 mean confidence (main)", "v2 most common wrong answer"],
                  "rows": rows},
        "charts": [{"type": "bar", "title": "Main accuracy per network: v1 (8→4 padding) vs v2 (10→5 padding)",
                    "subtitle": "5 attempts per network per version; no tools, no candidate list",
                    "unit": "% correct", "categories": [PRETTY[n] for n in NETS],
                    "series": [{"name": "v1 (8 in / 4 out)", "values": v1["main_acc"]},
                               {"name": "v2 (10 in / 5 out)", "values": v2["main_acc"] if has_v2 else [None] * 4}],
                    "caption": "In v1 the padded shape equals LunarLander's native interface. v2 removes that "
                               "shortcut, but 10 in / 5 out matches PettingZoo MPE simple_adversary, which became the new default guess."},
                   {"type": "bar", "title": "Original vs hidden-unit-permuted weights (v2)" if has_v2 else
                    "Original vs hidden-unit-permuted weights (v1)",
                    "subtitle": "Permutation leaves the function unchanged", "unit": "% correct",
                    "categories": [PRETTY[n] for n in NETS],
                    "series": [{"name": "Main", "values": (v2 if has_v2 else v1)["main_acc"]},
                               {"name": "Permuted", "values": (v2 if has_v2 else v1)["perm_acc"]}]}],
        "status": "done" if runs and all(r.get("correct") is not None for r in runs) else "running",
        "total_cost_usd": round(spent(), 4), "cost_v1_usd": round(v1["cost"], 4), "cost_v2_usd": round(v2["cost"], 4),
        "v2_decision": decision,
    }
    (LOG / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["table"]["rows"], indent=0))
    print(out["summary"])
    print("v2 decision:", decision)
    subprocess.run([sys.executable, str(REPO / "scripts" / "build_blog_data.py"), "--exp", EXP], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="one call: main, cartpole, attempt 1")
    ap.add_argument("--condition", choices=["main", "permuted", "random_init", "all"], default="all")
    ap.add_argument("--attempts", type=int, default=None, help="override attempts per network for the condition(s)")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--version", choices=["v1", "v2"], default="v1",
                    help="v1: 8->32->32->4 nets (checkpoints/exp1b); v2: 10->32->32->5 (checkpoints/exp1b_v2)")
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    global CFG
    CFG = VERSIONS[a.version]
    if a.grade_only:
        summarize()
        return
    conds = ["main", "permuted", "random_init"] if a.condition == "all" else [a.condition]
    att = dict(ATTEMPTS)
    if a.attempts:
        for c in conds:
            att[c] = a.attempts
    js = jobs(conds, att)
    if a.pilot:
        js = [j for j in jobs(["main"], att) if j["group"] == "cartpole" and j["attempt"] == 1]
    if a.dry_run:
        from harness.weights_text import estimate_tokens
        p = make_prompt(js[0]["sd"])
        print(p[:1500], "...\n", p[-400:])
        print(f"{len(js)} jobs; ~{estimate_tokens(p)} prompt tokens each")
        return
    run_all(js, workers=1 if a.pilot else a.workers)
    summarize()


if __name__ == "__main__":
    main()
