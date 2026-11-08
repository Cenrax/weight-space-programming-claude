"""Experiment 4: can Claude fix a failing RL policy by editing its weights, in one shot, with no tools?

Run from the repo root (keep CPU use moderate):

    E="taskpolicy -a env VECLIB_MAXIMUM_THREADS=1 venv/bin/python experiments/exp4/run.py"
    $E --baselines                       # unedited net + hand-coded heuristic (no API calls)
    $E --pilot                           # ONE MountainCar call (counts as mountaincar attempt 1 if settings unchanged)
    $E --case mountaincar --attempts 5   # runs attempts until 5 exist for the case (skips existing ones)
    $E --case lunarlander --attempts 5
    $E --grade-only                      # re-parse + re-evaluate every saved response in logs/exp4/
    $E --summarize                       # GIFs, logs/exp4/summary.json, docs/data/exp4.json
    $E --show-prompt mountaincar         # print the prompt (no API call)

Grading rules are fixed in experiments/exp4/GRADING.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from envs.wrappers import make_env  # noqa: E402
from eval.rollout import evaluate, load_checkpoint, policy_from_state_dict  # noqa: E402
from harness import weights_text as wt  # noqa: E402
from harness.logging_utils import new_run_id, read_result, write_result  # noqa: E402

EXP = "exp4"
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "logs" / EXP
GIFS = ROOT / "docs" / "assets" / "gifs"
BUDGET_USD = 12.0
N_EVAL, EVAL_SEED, TRACE_SEED = 1024, 10_000, 0
SETTINGS = {"effort": "high", "max_tokens": 64000, "thinking": True}  # see GRADING.md (pre-run changes)

CASES = {
    "mountaincar": {
        "ckpt": ROOT / "checkpoints/exp4/mountaincar_fail.pt",
        "group": "MountainCar",
        "env_doc": HERE / "prompts/env_mountaincar.md",
        "stride": 5,
        "obs_names": ["position", "velocity"],
        "actions": "0 = accelerate left, 1 = don't accelerate, 2 = accelerate right",
        "success": "the car reaches the flag (position >= 0.5) before the 200-step time limit",
        "goal": "reaches the flag",
        "before_gif": "assets/gifs/exp4_mountaincar_fail.gif",
    },
    "lunarlander": {
        "ckpt": ROOT / "checkpoints/exp4/lunarlander_hover.pt",
        "group": "LunarLander",
        "env_doc": HERE / "prompts/env_lunarlander.md",
        "stride": 10,
        "obs_names": ["x", "y", "vx", "vy", "angle", "ang_vel", "leg_L", "leg_R"],
        "actions": "0 = do nothing, 1 = fire left orientation engine, 2 = fire main engine, "
                   "3 = fire right orientation engine",
        "success": "the lander comes to rest on the ground (the episode terminates because the lander is no longer "
                   "awake, without the body touching the ground and without leaving the screen) with its final x "
                   "observation inside the landing pad between the two flags, |x| <= 0.2 (the pad spans world x in "
                   "[8, 12], i.e. observation x in [-0.2, 0.2]), before the 1000-step time limit",
        "goal": "lands between the flags",
        "before_gif": "assets/gifs/exp4_lunarlander_hover.gif",
    },
}


# ----------------------------------------------------------------------------- episodes

def net_policy(sd, meta):
    net = policy_from_state_dict(sd, meta)

    @torch.no_grad()
    def act(obs: np.ndarray) -> list[int]:
        return net(torch.as_tensor(obs, dtype=torch.float32)).argmax(-1).tolist()
    return act


def mc_heuristic(obs: np.ndarray) -> list[int]:
    """Push in the direction of velocity: right if v >= 0 else left."""
    return [2 if o[1] >= 0 else 0 for o in obs]


class _Disc:  # gymnasium's heuristic() only reads env.unwrapped.continuous
    unwrapped = type("U", (), {"continuous": False})()


def ll_heuristic(obs: np.ndarray) -> list[int]:
    from gymnasium.envs.box2d.lunar_lander import heuristic
    return [int(heuristic(_Disc, o)) for o in obs]


def run_episodes(act_fn, meta, n=N_EVAL, seed=EVAL_SEED, n_par=16):
    """Same seeding/lockstep scheme as eval.rollout.evaluate, but also records each episode's final observation
    (needed for 'landed between the flags'). Returns per-episode lists."""
    envs = [make_env(meta["env_id"], "raw") for _ in range(min(n_par, n))]
    obs, slot, ret = [None] * len(envs), [-1] * len(envs), np.zeros(len(envs))
    R, OUT, FIN = [None] * n, [None] * n, [None] * n
    nxt = 0
    for i, e in enumerate(envs):
        obs[i], _ = e.reset(seed=seed + nxt)
        e.action_space.seed(seed + nxt)
        slot[i], nxt = nxt, nxt + 1
    active = list(range(len(envs)))
    while active:
        acts = act_fn(np.stack([obs[i] for i in active]))
        still = []
        for a, i in zip(acts, active):
            o, r, te, tr, info = envs[i].step(int(a))
            ret[i] += r
            obs[i] = o
            if te or tr:
                ep = slot[i]
                R[ep], OUT[ep], FIN[ep] = float(ret[i]), info.get("outcome"), o.copy()
                ret[i] = 0.0
                if nxt < n:
                    obs[i], _ = envs[i].reset(seed=seed + nxt)
                    envs[i].action_space.seed(seed + nxt)
                    slot[i], nxt = nxt, nxt + 1
                    still.append(i)
            else:
                still.append(i)
        active = still
    for e in envs:
        e.close()
    return R, OUT, FIN


def score_policy(case: str, sd=None, meta=None, policy=None) -> dict:
    """GRADING.md metrics on the 1024 eval seeds. sd/meta for a net, or policy=callable (heuristic)."""
    meta = meta or load_checkpoint(str(CASES[case]["ckpt"]))[1]
    st = evaluate(sd, meta, n_episodes=N_EVAL, seed=EVAL_SEED, greedy=True, policy=policy)
    out = {"mean_return": st["mean_return"], "std_return": st["std_return"], "mean_length": st["mean_length"],
           "outcomes": st["outcomes"]}
    if case == "mountaincar":
        out["success_rate"] = st["success_rate"]
    else:
        act = policy if policy is not None else net_policy(sd, meta)
        R, OUT, FIN = run_episodes(act, meta)
        between = [o == "landed" and abs(float(f[0])) <= 0.2 for o, f in zip(OUT, FIN)]
        out["success_rate"] = float(np.mean(between))
        out["landing_rate"] = st["landing_rate"]
        out["success_rate_200"] = st["success_rate_200"]
        out["crosscheck_outcomes_match"] = dict(Counter(OUT)) == st["outcomes"]
        out["crosscheck_mean_return_diff"] = abs(float(np.mean(R)) - st["mean_return"])
        if not out["crosscheck_outcomes_match"]:
            print(f"WARNING: between-flags loop outcomes {dict(Counter(OUT))} != evaluate {st['outcomes']}")
    return out


def record_trace(case: str, sd, meta, seed=TRACE_SEED) -> dict:
    act = net_policy(sd, meta)
    env = make_env(meta["env_id"], "raw")
    o, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    steps, total, t = [], 0.0, 0
    while True:
        a = act(o[None])[0]
        o2, r, te, tr, info = env.step(a)
        steps.append({"t": t, "obs": [float(v) for v in o], "action": int(a), "reward": float(r)})
        total += r
        t += 1
        o = o2
        if te or tr:
            break
    env.close()
    return {"steps": steps, "return": total, "length": t, "outcome": info.get("outcome"),
            "final_obs": [float(v) for v in o], "action_counts": dict(Counter(s["action"] for s in steps))}


# ----------------------------------------------------------------------------- prompt

def trace_text(case: str, tr: dict) -> str:
    c = CASES[case]
    k = c["stride"]
    keep = [s for s in tr["steps"] if s["t"] % k == 0]
    if keep[-1]["t"] != tr["steps"][-1]["t"]:
        keep.append(tr["steps"][-1])
    if case == "mountaincar":
        fmt = lambda s: f"{s['t']:>4}  {s['obs'][0]:+.4f}  {s['obs'][1]:+.5f}  {s['action']}  {s['reward']:+.1f}"
        head = "   t  position  velocity  action  reward"
    else:
        fmt = lambda s: (f"{s['t']:>4}  " + "  ".join(f"{v:+.3f}" for v in s["obs"][:6]) +
                         f"  {s['obs'][6]:.0f} {s['obs'][7]:.0f}  {s['action']}  {s['reward']:+.3f}")
        head = "   t  " + "  ".join(f"{n:>6}" for n in c["obs_names"][:6]) + "  legs  action  reward"
    counts = ", ".join(f"action {a}: {n}" for a, n in sorted(tr["action_counts"].items()))
    fin = "  ".join(f"{n}={v:+.4f}" for n, v in zip(c["obs_names"], tr["final_obs"]))
    return (f"Env seed {TRACE_SEED}. Every {k}th step is shown (plus the last step). Each row is the observation the "
            f"network saw at step t, the greedy action it chose, and the reward received for that step.\n\n"
            f"```\n{head}\n" + "\n".join(fmt(s) for s in keep) + "\n```\n\n"
            f"Episode result: {tr['length']} steps, total return {tr['return']:.1f}, ended by "
            f"{'time limit (truncation)' if tr['outcome'] == 'timeout' else tr['outcome']}. "
            f"Final observation: {fin}.\nAction counts over all {tr['length']} steps: {counts}.\n")


def build_prompt(case: str) -> str:
    c = CASES[case]
    sd, meta = load_checkpoint(str(c["ckpt"]))
    arch = meta["arch"]
    n_w = sum(v.numel() for v in sd.values())
    tr = record_trace(case, sd, meta)
    weights = wt.weights_to_text(sd, header=False)
    in_desc = ", ".join(f"{i} = {n}" for i, n in enumerate(c["obs_names"]))
    edit_hint = f"W2 row 7  <{arch[1]} numbers>"
    return f"""I have a small neural-network control policy that fails at its task. I want you to (1) explain why it \
fails and (2) fix it by editing its weights directly.

You have no tools: you cannot run code, run the network, or run the environment, and you cannot test your edit. \
Work it out by reasoning, and give your answer in this one reply. The edited network will then be evaluated on many \
episodes.

{c['env_doc'].read_text().strip()}

# The policy network

- Architecture: MLP {' -> '.join(map(str, arch))} with tanh hidden activations ({n_w} weights). The architecture is \
fixed.
- Input: the raw observation as float32, no scaling or normalisation ({in_desc}).
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index). Actions: {c['actions']}.
- Weights are printed below with 3 decimals. `W<k> row i` lists the weights from input unit i of layer k to every \
output unit j = 0, 1, ...; that is, row i of W<k> in the [in, out] orientation used in the forward pass above, so \
W<k>[i, j] connects input unit i to output unit j. `b<k>` is the bias of layer k.

## Full weights

```
{weights.strip()}
```

# One recorded episode of the current policy

{trace_text(case, tr)}
# Your task

1. Explain why this network fails. Point to the specific weights, units and parts of the trace that show it.
2. Change the weights so that the greedy policy succeeds as often as possible. Success means: {c['success']}. It \
will be evaluated over many episodes with random starting conditions as described in the environment.
3. Give your change as edits to this network. Any weight you don't mention keeps its current value. You may change \
as many or as few weights as you like, and you may repurpose hidden units, but the architecture \
({' -> '.join(map(str, arch))}, tanh) and the input/output meaning are fixed.

# Answer format

- First a line `DIAGNOSIS: <why it fails, at most 300 characters>`.
- Then your explanation of the failure and of how your edit works.
- End your reply with exactly one machine-readable block of edits, delimited like this:

EDITS:
W1[0,5] = 12.0
b1[5] = -3.5
{edit_hint}
b3[2] = -1.0
END_EDITS

Rules for the EDITS block: one edit per line, no comments or prose inside the block; indices are 0-based; \
`W<k>[i,j] = v` sets the weight from input unit i to output unit j of layer k (same orientation as the printed \
rows); `b<k>[j] = v` sets one bias; `W<k> row i <values>` replaces a whole row (it must have exactly as many \
values as the printed row); `b<k> <values>` replaces a whole bias vector. Only the last EDITS block in your reply is \
used.
"""


# ----------------------------------------------------------------------------- parsing + grading

_LEAD = re.compile(r"^\s*(?:[-*>]+\s+)?([Wb]\d+)\b")


def _clean_line(ln: str) -> str:
    ln = ln.replace("−", "-").replace("–", "-").replace("**", "").replace("`", "")
    ln = re.split(r"\s(?:#|//)|^(?:#|//)", ln, maxsplit=1)[0] if not ln.lstrip().startswith("#") else ""
    return ln.rstrip()


def _line_ok(ln: str) -> bool:
    s = wt._BULLET_RE.sub("", ln.strip())
    if wt._ELEM_RE.match(s):
        return True
    m = wt._ROW_RE.match(s)
    if m and (m.group(2) or m.group(3)) and wt._NUMS_RE.fullmatch(m.group(4)):
        return True
    m = wt._BIAS_RE.match(s)
    return bool(m and wt._NUMS_RE.fullmatch(m.group(2)))


def extract_edits(text: str) -> tuple[str | None, list[str]]:
    """Block after the last 'EDITS:' up to 'END_EDITS' (cleaned), plus the W/b lines that are not valid edits."""
    idx = [m.end() for m in re.finditer(r"EDITS:", text)]
    if not idx:
        return None, []
    body = text[idx[-1]:]
    end = body.find("END_EDITS")
    body = body if end < 0 else body[:end]
    lines, bad = [], []
    for raw in body.splitlines():
        if raw.strip().startswith("```"):
            continue
        ln = _clean_line(raw)
        if not ln.strip():
            continue
        if _LEAD.match(ln) and not _line_ok(ln):
            bad.append(raw.strip())
        lines.append(ln)
    return "\n".join(lines), bad


def grade_run(d: Path, case: str) -> dict:
    sd0, meta = load_checkpoint(str(CASES[case]["ckpt"]))
    base = wt.params_from_state_dict(sd0)
    text = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    res = read_result(d)
    block, bad = extract_edits(text)
    g = {"case": case, "has_edits_block": block is not None, "unparsed_edit_lines": bad, "parse_error": None,
         "stop_reason": res.get("stop_reason")}
    edited = None
    if block is None:
        g["parse_error"] = "no EDITS: block"
    elif bad:
        g["parse_error"] = f"{len(bad)} unrecognised W/b line(s), e.g. {bad[0][:120]!r}"
    else:
        try:
            edited = wt.parse_weights(block, base=base, strict=True)
            if not all(np.isfinite(v).all() for v in edited.values()):
                g["parse_error"], edited = "NaN/inf in edited weights", None
        except ValueError as e:
            g["parse_error"] = f"strict parse failed: {e}"
    if edited is not None:
        g["n_edit_lines"] = len([ln for ln in block.splitlines() if _LEAD.match(ln)])
        sd = wt.params_to_state_dict(edited, sd0)
        torch.save({k: torch.as_tensor(v) for k, v in sd.items()}, d / "edited.pt")
        (d / "edited.json").write_text(json.dumps({**meta, "notes": f"exp4 Claude edit of {CASES[case]['ckpt'].name}"},
                                                  indent=1))
        (d / "edits.txt").write_text(block + "\n")
        diff = wt.diff_params(base, edited)
        st = score_policy(case, sd, meta)
        g.update(diff=diff, stats=st)
        score, mean_ret, frac = st["success_rate"], st["mean_return"], diff["frac"]
    else:  # failure; record the partial parse (never the headline)
        src = block if block is not None else text
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                part = wt.parse_weights(src, base=base, strict=False)
                if not all(np.isfinite(v).all() for v in part.values()):
                    raise ValueError("NaN/inf")
            except Exception as e:  # noqa: BLE001
                part, g["partial_parse_error"] = None, str(e)
        g["partial_parse_warnings"] = [str(x.message) for x in w][:20]
        if part is not None:
            psd = wt.params_to_state_dict(part, sd0)
            torch.save({k: torch.as_tensor(v) for k, v in psd.items()}, d / "partial_edited.pt")
            pst = score_policy(case, psd, meta)
            g.update(partial_parse_diff=wt.diff_params(base, part), partial_parse_stats=pst)
        unedited = json.loads((HERE / "baselines.json").read_text())[case]["unedited"]
        score, mean_ret, frac = 0.0, unedited["mean_return"], 0.0
    (d / "grade.json").write_text(json.dumps(g, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    m = re.search(r"DIAGNOSIS:\s*(.+)", text)
    diag = (m.group(1).strip() if m else text.strip().split("\n")[0])[:300]
    expl = text[: text.rfind("EDITS:")] if "EDITS:" in text else text
    write_result(d, target=CASES[case]["goal"], answer=diag, correct=bool(score >= 0.9), score=round(score, 4),
                 group=CASES[case]["group"], pct_weights_changed=round(100 * frac, 3), mean_return=round(mean_ret, 2),
                 parse_ok=edited is not None, parse_error=g["parse_error"],
                 partial_parse_score=(g.get("partial_parse_stats") or {}).get("success_rate"),
                 landing_rate=(g.get("stats") or {}).get("landing_rate"),
                 success_rate_200=(g.get("stats") or {}).get("success_rate_200"),
                 reasoning_excerpt=expl.strip()[:1500])
    print(f"[{d.name}] parse_ok={edited is not None} success={score:.4f} return={mean_ret:.1f} "
          f"changed={100 * frac:.2f}% {g['parse_error'] or ''}", flush=True)
    return g


# ----------------------------------------------------------------------------- runs

def exp_spend() -> float:
    tot = 0.0
    for p in LOGS.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def existing(case: str) -> list[Path]:
    return sorted(d for d in LOGS.glob(f"*-{case}-a*") if (d / "result.json").exists()
                  and read_result(d).get("status") == "complete")


_LOCK = threading.Lock()
_RESERVED = [0.0]  # cost estimates of calls in flight (their result.json may not exist yet)


def run_attempt(case: str, n: int, est_cost: float) -> Path | None:
    from harness.claude_client import BudgetExhausted, ask
    with _LOCK:
        spent = exp_spend()
        if spent + _RESERVED[0] + est_cost > BUDGET_USD:
            print(f"[{case} a{n}] skipped: exp4 spend ${spent:.2f} + in flight ${_RESERVED[0]:.2f} + est "
                  f"${est_cost:.2f} > ${BUDGET_USD}", flush=True)
            return None
        _RESERVED[0] += est_cost
    try:
        return _run_attempt(case, n, spent)
    except BudgetExhausted as e:
        print(f"[{case} a{n}] project budget exhausted: {e}", flush=True)
        return None
    finally:
        with _LOCK:
            _RESERVED[0] -= est_cost


def _run_attempt(case: str, n: int, spent: float) -> Path:
    from harness.claude_client import ask
    prompt = build_prompt(case)
    rid = new_run_id(f"{case}-a{n}")
    print(f"[{rid}] calling claude ({SETTINGS})", flush=True)
    r = ask(prompt, exp_id=EXP, run_id=rid, condition="no-tools one-shot edit", max_cost_usd=BUDGET_USD - spent,
            effort=SETTINGS["effort"], max_tokens=SETTINGS["max_tokens"], thinking=SETTINGS["thinking"],
            extra={"label": f"{CASES[case]['group']} · attempt {n}", "group": CASES[case]["group"],
                   "target": CASES[case]["goal"], "attempt": n, "effort": SETTINGS["effort"]})
    print(f"[{rid}] done: in={r.tokens_in} out={r.usage['output_tokens']} cost=${r.cost_usd:.3f} "
          f"stop={r.stop_reason}", flush=True)
    grade_run(r.log_dir, case)
    return r.log_dir


def do_baselines() -> None:
    out = {}
    for case in CASES:
        sd, meta = load_checkpoint(str(CASES[case]["ckpt"]))
        heur = mc_heuristic if case == "mountaincar" else ll_heuristic
        out[case] = {"unedited": score_policy(case, sd, meta),
                     "heuristic": score_policy(case, None, meta, policy=heur),
                     "heuristic_desc": "push in the direction of velocity (2 if v >= 0 else 0)" if case == "mountaincar"
                     else "gymnasium.envs.box2d.lunar_lander.heuristic (discrete)"}
        rb = json.loads((ROOT / "checkpoints/exp4/random_baseline.json").read_text())["results"][meta["env_id"]]
        out[case]["random"] = rb
        print(case, json.dumps({k: v.get("success_rate") if isinstance(v, dict) else v for k, v in out[case].items()}))
    (HERE / "baselines.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- summary

def summarize() -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    from eval.render_gif import collect_frames, save_gif
    base = json.loads((HERE / "baselines.json").read_text())
    rows, charts, media = [], [], []
    per_case = {}
    for case, c in CASES.items():
        runs = existing(case)
        res = sorted((read_result(d) | {"_dir": d} for d in runs), key=lambda r: r.get("attempt") or 0)
        if not res:
            continue
        scores = [r["score"] or 0.0 for r in res]
        best = max(res, key=lambda r: ((r["score"] or 0), r.get("mean_return") or -1e9))
        per_case[case] = (res, best)
        b = base[case]
        rows.append([c["group"], f"{100 * b['unedited']['success_rate']:.1f}%",
                     f"{100 * best['score']:.1f}%", f"{100 * float(np.median(scores)):.1f}%",
                     f"{sum(bool(r['correct']) for r in res)}/{len(res)}",
                     f"{best.get('pct_weights_changed', 0):.1f}%",
                     f"{float(np.median([r.get('pct_weights_changed') or 0 for r in res])):.1f}%",
                     f"{b['unedited']['mean_return']:.1f} → {best.get('mean_return'):.1f}",
                     f"{100 * b['heuristic']['success_rate']:.1f}%"])
        charts.append({"type": "bar", "title": f"{c['group']}: success rate per attempt",
                       "subtitle": f"1024 seeded greedy episodes; success = {c['goal']}", "unit": "% success",
                       "categories": [f"attempt {r.get('attempt') or i + 1}" for i, r in enumerate(res)],
                       "series": [{"name": "Opus 5.5 edit", "values": [round(100 * (r['score'] or 0), 1) for r in res]}],
                       "caption": f"Unedited network: {100 * b['unedited']['success_rate']:.1f}%. Hand-coded heuristic "
                                  f"({b['heuristic_desc']}): {100 * b['heuristic']['success_rate']:.1f}%."})
        bd = best["_dir"]
        if (bd / "edited.pt").exists():
            sd = torch.load(bd / "edited.pt", weights_only=True)
            meta = json.loads((bd / "edited.json").read_text())
            frames, ret, steps = collect_frames(sd, meta, seed=EVAL_SEED)
            out = GIFS / f"exp4_claude_{case}.gif"
            mb, nfr, ev = save_gif(frames, str(out))
            print(f"gif {out.name}: {mb:.2f} MB, return {ret:.1f}, {steps} steps")
            media += [{"type": "gif", "src": c["before_gif"], "group": c["group"],
                       "caption": f"Before: the unedited policy ({c['group']}, seed {EVAL_SEED}), "
                                  f"{100 * b['unedited']['success_rate']:.0f}% success."},
                      {"type": "gif", "src": f"assets/gifs/exp4_claude_{case}.gif", "group": c["group"],
                       "caption": f"After: Opus 5.5's best edit ({best.get('label')}), seed {EVAL_SEED}, episode "
                                  f"return {ret:.1f}; {100 * best['score']:.1f}% success over 1024 episodes, "
                                  f"{best.get('pct_weights_changed'):.1f}% of weights changed."}]
    total = exp_spend()
    parts = []
    for case, (res, best) in per_case.items():
        parts.append(f"{CASES[case]['group']} {100 * base[case]['unedited']['success_rate']:.0f}% → best "
                     f"{100 * best['score']:.1f}% (median {100 * float(np.median([r['score'] or 0 for r in res])):.1f}%"
                     f", {sum(bool(r['correct']) for r in res)}/{len(res)} attempts ≥ 90%)")
    summary = ("With one reply, no tools and no rollouts, Opus 5.5's weight edits moved success from "
               + "; ".join(parts) + " over 1024 seeded episodes; every attempt did this by wiring repurposed hidden units "
               "into a known textbook controller (push in the direction of velocity; Gymnasium's own LunarLander "
               f"heuristic, recalled from memory) that overrides the learned policy (total API cost ${total:.2f}).")
    js = {"summary": summary,
          "table": {"caption": "Success over 1024 seeded greedy episodes (seeds 10000–11023), 5 one-shot attempts per case",
                    "columns": ["Case", "Before (unedited)", "After: best", "After: median", "Attempts ≥ 90%",
                                "% weights changed (best)", "% weights changed (median)", "Mean return before → best",
                                "Hand-coded heuristic"],
                    "rows": rows},
          "media": media, "charts": charts, "total_cost_usd": round(total, 4)}
    (LOGS / "summary.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    print(json.dumps(js["table"], indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(ROOT / "scripts/build_blog_data.py"), "--exp", EXP], check=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--case", choices=list(CASES))
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--show-prompt", choices=list(CASES))
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--est-cost", type=float, default=1.2, help="per-call cost estimate used for the budget check")
    a = ap.parse_args()
    if a.show_prompt:
        print(build_prompt(a.show_prompt))
        return
    if a.baselines:
        do_baselines()
        return
    if a.grade_only:
        for case in ([a.case] if a.case else CASES):
            for d in existing(case):
                grade_run(d, case)
        return
    if a.summarize:
        summarize()
        return
    if a.pilot:
        run_attempt("mountaincar", 1, a.est_cost)
        return
    cases = [a.case] if a.case else list(CASES)
    jobs = []
    for case in cases:
        have = {read_result(d).get("attempt") for d in existing(case)}
        jobs += [(case, n) for n in range(1, a.attempts + 1) if n not in have]
    print(f"jobs: {jobs}; exp4 spend so far ${exp_spend():.2f}")
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(run_attempt, c, n, a.est_cost) for c, n in jobs]
        for f in futs:
            f.result()
    print(f"exp4 spend ${exp_spend():.2f}")


if __name__ == "__main__":
    main()
