"""Experiment 5b: write policy weights from scratch WITH a budget of real-environment rollouts (blog Exp. 6 protocol).

Claude gets a sandbox (numpy, the saved weights starting at all-zero, save_weights, and a budgeted rollout()).
Budget 0 = the exp5 zero-shot result (reused). Grading is fixed in experiments/exp5b/GRADING.md.

    E="taskpolicy -a env VECLIB_MAXIMUM_THREADS=1 venv/bin/python experiments/exp5b/run.py"
    $E --show-prompt acrobot --budget 5      # print prompt + tool description (no API call)
    $E --test-rollout acrobot                # one local rollout() of the zero net (no API call)
    $E --pilot                               # Acrobot budget 5
    $E --parallel 2                          # remaining (env, budget) cells; skips finished ones
    $E --extra                               # pre-registered 2nd budget-25 Breakout attempt (if money allows)
    $E --grade-run logs/exp5b/<run>          # (re)grade one run from its saved weights
    $E --summarize                           # table, charts, GIFs, logs/exp5b/summary.json, docs/data/exp5b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import threading
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
from eval.rollout import policy_from_state_dict  # noqa: E402
from harness.logging_utils import new_run_id, read_result, run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp5_run", ROOT / "experiments" / "exp5" / "run.py")
X5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X5)

EXP = "exp5b"
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "logs" / EXP
GIFS = ROOT / "docs" / "assets" / "gifs"
BUDGET_USD = 12.0
EFFORT, MAX_TOKENS = "high", 32000
CAPS = {5: {"turns": 30, "usd": 1.20}, 25: {"turns": 70, "usd": 2.20}}
ENVS = ["acrobot", "lunarlander", "breakout"]
BUDGETS = [5, 25]
EXTRA_ENV = "breakout"           # pre-registered (GRADING.md): largest exp5 spread / (PPO-final - random)
ROLLOUT_N_MAX = 32
ROLLOUT_SEED0 = 100_000          # call k uses seeds 100000 + 1000*k + i (disjoint from final eval 10000-10099)
BREAKOUT_ROLLOUT_MAX_STEPS = 5000
TRACE_ROWS = 30


# ----------------------------------------------------------------------------- rollout (runs in the parent)

def params_to_sd(p):
    return X5.params_to_sd({k: np.asarray(v, np.float32) for k, v in p.items()})


@torch.no_grad()
def run_rollout(env: str, weights, k: int, n: int = 32, trace_obs=None) -> dict:
    """Greedy rollouts of `weights` on n episodes with seeds ROLLOUT_SEED0 + 1000*k + i (same lockstep as evaluate)."""
    n = int(n)
    if not 1 <= n <= ROLLOUT_N_MAX:
        raise ValueError(f"n must be between 1 and {ROLLOUT_N_MAX}")
    meta = X5.meta_of(env)
    atari = meta["env_kind"] == "atari_ram"
    if trace_obs is not None:
        trace_obs = [int(i) for i in trace_obs]
        if any(not 0 <= i < meta["arch"][0] for i in trace_obs) or len(trace_obs) > 16:
            raise ValueError(f"trace_obs: up to 16 indices in 0..{meta['arch'][0] - 1}")
    net = policy_from_state_dict(params_to_sd(weights), meta)
    seed = ROLLOUT_SEED0 + 1000 * k
    max_steps = BREAKOUT_ROLLOUT_MAX_STEPS if atari else None
    n_par = min(16, n)
    envs = [make_env(meta["env_id"], meta["env_kind"]) for _ in range(n_par)]
    nxt, slot, obs = 0, [-1] * n_par, [None] * n_par
    ret, ln = np.zeros(n_par), np.zeros(n_par, int)
    R, L, O = [None] * n, [None] * n, [None] * n
    traces = {e: [] for e in range(n)}
    acts_all = Counter()

    def fmt_obs(o):
        if atari:
            idx = trace_obs or []
            return {str(i): int(round(float(o[i]) * 255)) for i in idx}
        return [round(float(v), 3) for v in o]

    for i in range(n_par):
        obs[i], _ = envs[i].reset(seed=seed + nxt)
        envs[i].action_space.seed(seed + nxt)
        slot[i], nxt = nxt, nxt + 1
    active = list(range(n_par))
    while active:
        x = torch.as_tensor(np.stack([obs[i] for i in active]), dtype=torch.float32)
        acts = net(x).argmax(-1).tolist()
        still = []
        for a, i in zip(acts, active):
            ep = slot[i]
            acts_all[a] += 1
            traces[ep].append((int(ln[i]), fmt_obs(obs[i]), int(a), round(float(ret[i]), 2)))
            o, r, te, tr, info = envs[i].step(a)
            ret[i] += info.get("true_reward", r)
            ln[i] += 1
            obs[i] = o
            if max_steps is not None and ln[i] >= max_steps:
                tr = True
            if te or tr:
                R[ep], L[ep] = float(ret[i]), int(ln[i])
                O[ep] = info.get("outcome", "terminated" if te else "truncated")
                traces[ep].append((int(ln[i]), fmt_obs(o), None, round(float(ret[i]), 2)))
                ret[i], ln[i] = 0.0, 0
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
    Ra = np.array(R)

    def sub(ep):
        t = traces[ep]
        every = max(1, math.ceil((len(t) - 1) / (TRACE_ROWS - 1)))
        rows = t[:-1:every] + [t[-1]]
        return {"episode": ep, "return": round(R[ep], 2), "length": L[ep], "outcome": O[ep], "every_k_steps": every,
                "rows (step, obs, action, return_so_far)": rows}

    order = np.argsort(Ra)
    out = {"episodes": n, "mean_return": round(float(Ra.mean()), 3), "std_return": round(float(Ra.std()), 3),
           "min_return": round(float(Ra.min()), 2), "max_return": round(float(Ra.max()), 2),
           "returns": [round(v, 1) for v in R], "outcomes": dict(Counter(O)),
           "mean_length": round(float(np.mean(L)), 1), "min_length": int(min(L)), "max_length": int(max(L))}
    if env == "acrobot":
        out["success_rate"] = O.count("swung_up") / n
    elif env == "lunarlander":
        out["success_rate_return_ge_200"] = float((Ra >= 200).mean())
    if atari:
        out["note"] = f"rollout episodes are truncated after {BREAKOUT_ROLLOUT_MAX_STEPS} agent steps"
    tot = sum(acts_all.values())
    out["action_frequencies"] = {str(a): round(c / tot, 3) for a, c in sorted(acts_all.items())}
    out["traces"] = {"worst": sub(int(order[0])), "best": sub(int(order[-1]))}
    return out


# ----------------------------------------------------------------------------- prompt + tool text

def obs_desc(env):
    c = X5.ENVS[env]
    if env == "breakout":
        return ("the 128 RAM bytes divided by 255: x[i] = RAM[i] / 255 for i = 0..127, so every input is in [0, 1]. "
                "You are not told what any RAM byte means")
    return "the raw observation as float32, no scaling or normalisation (" + \
        ", ".join(f"x[{i}] = {n}" for i, n in enumerate(c["obs_names"])) + ")"


def build_prompt(env: str, budget: int) -> str:
    c = X5.ENVS[env]
    meta = X5.meta_of(env)
    arch = meta["arch"]
    n_w = sum(arch[k] * arch[k + 1] + arch[k + 1] for k in range(len(arch) - 1))
    doc = (ROOT / "experiments" / "exp5" / "prompts" / f"env_{env}.md").read_text().strip()
    game = "Breakout" if env == "breakout" else c["env_id"]
    shapes = ", ".join(f"W{k + 1} [{arch[k]} x {arch[k + 1]}], b{k + 1} [{arch[k + 1]}]" for k in range(len(arch) - 1))
    turns = CAPS[budget]["turns"]
    trace_note = ("Breakout traces include only the RAM bytes you list in `trace_obs` (as integer byte values 0-255, "
                  "up to 16 indices); by default a trace shows step, action and score so far. Rollout episodes are "
                  f"truncated after {BREAKOUT_ROLLOUT_MAX_STEPS} agent steps (the final evaluation uses the normal "
                  "27,000-step limit)." if env == "breakout" else
                  "Traces show the full observation (3 decimals).")
    return f"""Write, from scratch, the weights of a neural-network policy that plays {game} as well as possible.

There is no training: you choose every number yourself. You have a Python sandbox (the `python` tool) holding the \
network's weights, which start at **all zero**, and a small budget of test runs in the real environment: you can \
call `rollout()` at most **{budget} times** in total. Interleave edits and rollouts however you like. When you \
finish, your **final saved weights** are evaluated, unchanged, on 100 new episodes.

{doc}

# The policy network

- Architecture: MLP {' -> '.join(map(str, arch))} with tanh hidden activations ({n_w} weights). The architecture is \
fixed. Shapes: {shapes}.
- Input x: {obs_desc(env)}.
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index), at every step. Actions: {c['actions']}.
- Weights use the [in, out] orientation: W<k>[i, j] connects input unit i of layer k to output unit j.
- Goal: {c['objective']}. The final score is the mean return over 100 episodes with random starting conditions as \
described in the environment (different seeds from your rollouts).

# Your tools

- `python`: a persistent numpy session. The saved weights are preloaded as `W1, b1, W2, b2, W3, b3` (float32 numpy \
arrays, [in, out]). Save with `save_weights(W1=..., b1=..., ...)` (only the given tensors are replaced) or with the \
`save_weights` text tool. Saving is free and unlimited.
- `rollout(n=32, trace_obs=None)` (callable inside `python`): runs the **currently saved** weights greedily on n ≤ 32 \
fresh episodes of the real environment and returns a dict: mean/std/min/max return, every episode's return, the \
outcome counts, success rate, episode lengths, action frequencies, and short subsampled traces (every k steps) of \
the best and the worst episode. {trace_note} Each call uses one of your {budget} rollouts, whatever n is; \
`budget_status()` shows how many are left (free).
- You can make at most {turns} tool calls in total; after that you will be asked for your final answer.

Rules: you may analyse and construct the weights in any way with numpy, but you may **not** run the network (or any \
part of it) on inputs inside the sandbox, and you may **not** re-implement or simulate the environment's dynamics. \
All information about how the network behaves must come from `rollout()`. The weights that count are the ones saved \
when you finish (not the best ones you tried), so make sure your final saved weights are the ones you want.

# Final answer

When you are done, reply with a line `CONTROLLER: <the controller your final saved weights implement, at most 300 \
characters>`, followed by a short explanation of the network and of what you changed in response to the rollouts.
"""


def tool_extra(env: str, budget: int) -> str:
    return (f"Budget: rollout() may be called at most {budget} times in this session. Rollouts, final evaluation and "
            "budget counters live outside this sandbox; gymnasium and the environment cannot be imported here.")


# ----------------------------------------------------------------------------- run one cell

def exp_spend() -> float:
    tot = 0.0
    for p in LOGS.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def finished(env: str, budget: int) -> list[Path]:
    return sorted(d for d in LOGS.glob(f"*-{env}-b{budget}-a*") if read_result(d).get("status") == "complete")


_LOCK = threading.Lock()
_RESERVED = [0.0]


def run_cell(env: str, budget: int, attempt: int = 1) -> Path | None:
    from harness.claude_client import BudgetExhausted, ask
    from harness.sandbox import Sandbox
    cap = CAPS[budget]["usd"]
    with _LOCK:
        spent = exp_spend()
        if spent + _RESERVED[0] + cap > BUDGET_USD:
            print(f"[{env} b{budget}] skipped: spend ${spent:.2f} + in flight ${_RESERVED[0]:.2f} + cap ${cap:.2f} "
                  f"> ${BUDGET_USD}", flush=True)
            return None
        _RESERVED[0] += cap
    tag = f"{env}-b{budget}-a{attempt}"
    rid = new_run_id(tag)
    d = run_dir(EXP, rid)
    curve: list[float] = []
    sb = Sandbox(X5.zero_params(env), log_dir=d, timeout=120, max_output_chars=20000, threads=1, isolate=True)

    def rollout(n: int = 32, trace_obs=None):
        k = sb.fns["rollout"].used          # already incremented by the sandbox before this runs
        out = run_rollout(env, sb.weights, k, n=n, trace_obs=trace_obs)
        out = {"rollout_call": k, "rollouts_left": budget - k, **out}
        curve.append(out["mean_return"])
        write_result(d, score_curve=list(curve), rollouts_used=k)
        return out

    sb.register("rollout", rollout, budget=budget,
                doc=("rollout(n=32, trace_obs=None) -> dict. Runs the SAVED weights greedily on n<=32 fresh episodes "
                     "of the real environment; returns returns/outcomes/success/lengths/action frequencies and "
                     "subsampled traces of the best and worst episode."))
    sb.start()
    try:
        print(f"[{rid}] calling claude (budget {budget}, caps {CAPS[budget]})", flush=True)
        extra = {"label": f"{X5.ENVS[env]['group']} · {budget} rollouts", "group": X5.ENVS[env]["group"],
                 "env": env, "budget": budget, "attempt": attempt,
                 "target": X5.ENVS[env]["env_id"], "effort": EFFORT, "max_turns": CAPS[budget]["turns"],
                 "max_cost_usd": cap, "run_tag": tag}
        r = ask(build_prompt(env, budget), exp_id=EXP, run_id=rid, effort=EFFORT, max_tokens=MAX_TOKENS,
                tools=sb.tools(tool_extra(env, budget)), tool_handlers=sb.handlers(),
                max_turns=CAPS[budget]["turns"], condition=f"budget_{budget}", max_cost_usd=cap, extra=extra)
        print(f"[{rid}] done: turns={r.n_turns} in={r.tokens_in} out={r.usage['output_tokens']} "
              f"cost=${r.cost_usd:.3f} stop={r.stop_reason} rollouts={sb.fns['rollout'].used}", flush=True)
    except BudgetExhausted as e:
        print(f"[{tag}] project budget exhausted: {e}", flush=True)
        r = None
    finally:
        np.savez(d / "final_weights.npz", **sb.weights)
        write_result(d, rollouts_used=sb.fns["rollout"].used, score_curve=list(curve), n_saves=sb.n_saves)
        sb.close()
        with _LOCK:
            _RESERVED[0] -= cap
    subprocess.run([sys.executable, __file__, "--grade-run", str(d)], check=False)
    return d


# ----------------------------------------------------------------------------- grading + audit

FWD = re.compile(r"np\.tanh\s*\(.*(@|\bdot\s*\(|matmul)", re.S)
SIM = re.compile(r"\brk4\b|\bdsdt\b|_dsdt|def\s+step\s*\(|odeint|solve_ivp|\bdt\s*=\s*0\.2\b|gravity|box2d", re.I)


def audit(d: Path) -> dict:
    flags = []
    tp = d / "transcript.jsonl"
    for ln in tp.read_text().splitlines() if tp.exists() else []:
        ev = json.loads(ln)
        if ev.get("event") != "message" or ev.get("role") != "assistant":
            continue
        for b in ev.get("content", []):
            if b.get("type") == "tool_use" and b.get("name") == "python":
                code = b["input"].get("code", "")
                kinds = [k for k, rx in (("forward_pass", FWD), ("env_sim", SIM)) if rx.search(code)]
                if kinds:
                    flags.append({"turn": ev["turn"], "kinds": kinds, "code_head": code[:400]})
    return {"n_flagged_calls": len(flags), "flagged": flags}


def grade_run(d: Path) -> dict:
    res = read_result(d)
    env = res["env"]
    meta = X5.meta_of(env)
    fw = d / "final_weights.npz"
    params = {k: np.load(fw)[k] for k in X5.zero_params(env)} if fw.exists() else X5.zero_params(env)
    sd = params_to_sd(params)
    torch.save(sd, d / "edited.pt")
    (d / "edited.json").write_text(json.dumps({**meta, "algo": f"claude-rollout-budget-{res['budget']}",
                                               "notes": f"exp5b final saved weights ({d.name})"}, indent=1))
    st = X5.score_net(env, sd, meta)
    agree = X5.heuristic_agreement(env, sd, meta)
    nz = int(sum(int((np.asarray(v) != 0).sum()) for v in params.values()))
    ppo100k = X5.baselines()[env]["ppo_100k"]["mean_return"]
    text = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    m = re.search(r"CONTROLLER:\s*(.+)", text)
    ans = (m.group(1).strip() if m else (text.strip().split("\n")[0] if text.strip() else "(no final reply)"))[:300]
    au = audit(d)
    g = {"stats": st, "nonzero_weights": nz, "heuristic_agreement": agree, "audit": au}
    if env == "breakout":
        g["ram_usage"] = X5.breakout_ram_usage(params)
    (d / "grade.json").write_text(json.dumps(g, indent=1, default=str))
    score = float(st["mean_return"])
    write_result(d, answer=ans, score=round(score, 3), correct=bool(score > ppo100k),
                 success_rate=st["success_rate"], std_return=round(st["std_return"], 3),
                 max_return=st["max_return"], outcomes=st["outcomes"], landing_rate=st.get("landing_rate"),
                 heuristic_agreement=None if agree is None else round(agree, 4), n_nonzero=nz,
                 ram_bytes_used=(g.get("ram_usage") or {}).get("ram_bytes_used"),
                 audit_n_flagged_calls=au["n_flagged_calls"], reasoning_excerpt=text.strip()[:1500])
    print(f"[{d.name}] final score {score:.2f} ± {st['std_return']:.1f} success={st['success_rate']} "
          f"(PPO-100k {ppo100k:.2f}) curve={res.get('score_curve')} flags={au['n_flagged_calls']}", flush=True)
    return g


# ----------------------------------------------------------------------------- summary

SUMMARY_NOTES = (  # written after reading the transcripts; facts only, headline numbers are untouched
    "\n\nBreakout at budget 5 scores 0: after its last rollout Claude saved an untested 'fix' whose FIRE bias "
    "(−2.5) loses the argmax to NOOP, so the ball is never served (all 100 episodes hit the 27k-step cap). "
    "Post hoc, its best tested save would have scored 31.7 on the same seeds (not the pre-registered score). "
    "Budget 25 on Breakout (184.7 and 92.1, pre-registered extra attempt) came from decoding RAM bytes from the "
    "traces (ball direction 103/105, raw paddle input 70) and tuning a dead zone; Claude was never given a RAM map. "
    "Audit: no environment simulation in any run; one minor breach (Breakout b5 computed layer-1 pre-activations "
    "on two hand-made inputs) and one grey-area check (LunarLander b25 evaluated its own |x| tanh formula on "
    "scalars, not via the weights). Excluding the breach leaves D1 at 2/3; also excluding the grey-area run would "
    "make it 1/3.")


def summarize() -> None:
    from eval.render_gif import collect_frames, save_gif
    base = X5.baselines()
    fmt = lambda v: "—" if v is None else (f"{v:.1f}" if abs(v) < 1000 else f"{v:.0f}")
    pct = lambda v: "" if v is None else f" ({100 * v:.0f}%)"
    rows, charts, media, dec, cells = [], [], [], {}, {}
    for env in ENVS:
        c = X5.ENVS[env]
        b = base[env]
        fin = c["final"]
        z = [X5.read_result(p) for p in sorted((ROOT / "logs/exp5").glob(f"*-{env}-a*"))
             if (p / "result.json").exists() and X5.read_result(p).get("status") == "complete"]
        zs = [r["score"] for r in z]
        z_med, z_best = float(np.median(zs)), max(zs)
        cell = {}
        for B in BUDGETS:
            rs = [read_result(d) | {"_dir": d} for d in finished(env, B) if read_result(d).get("score") is not None]
            if rs:
                cell[B] = {"runs": rs, "score": float(np.mean([r["score"] for r in rs]))}
        cells[env] = cell
        sc = lambda B: cell.get(B, {}).get("score")
        runs_txt = lambda B: "—" if B not in cell else " / ".join(
            f"{r['score']:.1f}{pct(r.get('success_rate'))}" for r in cell[B]["runs"]) + \
            (f" (mean {cell[B]['score']:.1f})" if len(cell[B]["runs"]) > 1 else "")
        rows.append([c["group"], fmt(b["random"]["mean_return"]), fmt(b["ppo_100k"]["mean_return"]),
                     fmt(b["ppo_1M"]["mean_return"]), fmt(b[f"ppo_{fin}"]["mean_return"]) + f" [{fin}]",
                     fmt(z_med), fmt(z_best), runs_txt(5), runs_txt(25)])
        s5, s25 = sc(5), sc(25)
        inc = s5 is not None and s25 is not None and s25 > s5 > z_med
        ppo_fin = b[f"ppo_{fin}"]["mean_return"]
        dec[env] = {"zero_shot_median": z_med, "zero_shot_best": z_best, "budget_5": s5, "budget_25": s25,
                    "increasing": inc, "b25_gt_b0_median": s25 is not None and s25 > z_med,
                    "ge_ppo_final": [B for B in BUDGETS if sc(B) is not None and sc(B) >= ppo_fin],
                    "ppo_final": ppo_fin}
        refs = [{"y": round(b["ppo_100k"]["mean_return"], 1), "label": "PPO 100k"},
                {"y": round(ppo_fin, 1), "label": f"PPO {fin}"}]
        if env != "lunarlander":
            refs.append({"y": round(b["random"]["mean_return"], 1), "label": "random"})
        charts.append({"type": "line", "title": f"{c['group']}: final score vs rollout budget",
                       "subtitle": "mean return, 100 greedy episodes (seeds 10000–10099)", "x_label": "rollout budget",
                       "y_label": "mean return", "x": [0, 5, 25],
                       "series": [{"name": "Opus 5.5", "values": [round(z_med, 1), None if s5 is None else round(s5, 1),
                                                                 None if s25 is None else round(s25, 1)]},
                                  {"name": "zero-shot best", "values": [round(z_best, 1), None, None]}],
                       "reference_lines": refs if env != "lunarlander" else refs[1:],
                       "caption": "Budget 0 is the exp5 zero-shot median (3–5 attempts; best shown separately). "
                                  "Budgets 5 and 25: one run each, scored on the final saved weights."
                                  + (" PPO-100k (−1495) is off the chart." if env == "lunarlander" else "")})
        # per-rollout curve chart for the budget-25 run(s)
        if 25 in cell:
            r0 = cell[25]["runs"][0]
            cv = r0.get("score_curve") or []
            r5 = (cell.get(5, {}).get("runs") or [{}])[0].get("score_curve") or []
            if cv:
                charts.append({"type": "line", "title": f"{c['group']}: what Claude saw, rollout by rollout",
                               "subtitle": "mean return of each 32-episode rollout batch (fresh seeds)",
                               "x_label": "rollout call", "y_label": "batch mean return",
                               "x": list(range(1, len(cv) + 1)),
                               "series": [{"name": "budget 25", "values": [round(v, 1) for v in cv]},
                                          {"name": "budget 5", "values": [round(v, 1) for v in r5] +
                                           [None] * (len(cv) - len(r5))}],
                               "reference_lines": [{"y": round(ppo_fin, 1), "label": f"PPO {fin}"}],
                               "caption": "Each point evaluates the weights saved at that moment; the final score "
                                          "uses the last saved weights on held-out seeds."})
        allr = [r for B in cell for r in cell[B]["runs"]]
        if allr:
            best = max(allr, key=lambda r: r["score"])
            bd = best["_dir"]
            sd = torch.load(bd / "edited.pt", weights_only=True)
            meta = json.loads((bd / "edited.json").read_text())
            atari = meta["env_kind"] == "atari_ram"
            frames, ret, steps = collect_frames(sd, meta, seed=X5.EVAL_SEED, max_steps=300 if atari else 27_000)
            out = GIFS / f"exp5b_{env}.gif"
            mb, _, _ = save_gif(frames, str(out), every=1 if atari else None)
            print(f"gif {out.name}: {mb:.2f} MB, return {ret:.1f}, {steps} steps")
            media.append({"type": "gif", "src": f"assets/gifs/exp5b_{env}.gif", "group": c["group"],
                          "caption": f"{c['group']}: Opus 5.5's best network with rollouts ({best.get('label')}), "
                                     f"seed {X5.EVAL_SEED}" + (f", first {steps} agent steps (return so far {ret:.0f})"
                                                              if atari else f", episode return {ret:.1f}") +
                                     f"; mean return {best['score']:.1f} over 100 episodes."})
    inc = [e for e in dec if dec[e]["increasing"]]
    weak = [e for e in dec if dec[e]["b25_gt_b0_median"]]
    gef = [e for e in dec if dec[e]["ge_ppo_final"]]
    total = exp_spend()
    G = lambda L: ", ".join(X5.ENVS[e]["group"] for e in L) or "none"
    summary = (f"Starting from all-zero weights with a sandbox and a budget of real-environment rollouts, score rose "
               f"monotonically with budget (25 > 5 > zero-shot median) on {len(inc)}/3 environments ({G(inc)}; D1 needs "
               f"≥ 2), the budget-25 network beat the zero-shot median on {len(weak)}/3 ({G(weak)}), and a "
               f"rollout-budget network matched or beat the final PPO checkpoint on {len(gef)}/3 ({G(gef)}). "
               f"Total API cost ${total:.2f}." + SUMMARY_NOTES)
    js = {"summary": summary, "status": "done",
          "table": {"caption": "Mean return over 100 seeded greedy episodes (seeds 10000–10099); success rate in "
                               "parentheses (Acrobot goal, LunarLander return ≥ 200). Zero-shot = exp5 (no tools, "
                               "budget 0). Budget columns: final saved weights of one run each.",
                    "columns": ["Env", "Random", "PPO 100k", "PPO 1M", "PPO final", "Zero-shot median",
                                "Zero-shot best", "Budget 5", "Budget 25"], "rows": rows},
          "decision": {"D1_increasing": {"envs": inc, "count": len(inc), "pass": len(inc) >= 2},
                       "D1_weak_b25_gt_b0_median": {"envs": weak, "count": len(weak)},
                       "D2_ge_ppo_final": {e: dec[e]["ge_ppo_final"] for e in gef}, "per_env": dec},
          "media": media, "charts": charts, "total_cost_usd": round(total, 4)}
    (LOGS / "summary.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    print(json.dumps({k: js[k] for k in ("summary", "table", "decision")}, indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(ROOT / "scripts/build_blog_data.py"), "--exp", EXP], check=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-prompt", choices=ENVS)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--test-rollout", choices=ENVS)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--grade-run")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--env", choices=ENVS, action="append")
    a = ap.parse_args()
    if a.show_prompt:
        print(build_prompt(a.show_prompt, a.budget))
        print("---- extra tool text ----\n" + tool_extra(a.show_prompt, a.budget))
    elif a.test_rollout:
        import time
        t = time.time()
        out = run_rollout(a.test_rollout, X5.zero_params(a.test_rollout), 1,
                          trace_obs=[72, 99, 101] if a.test_rollout == "breakout" else None)
        s = json.dumps(out)
        print(s[:3000], f"\n... {len(s)} chars, {time.time() - t:.1f}s")
    elif a.grade_run:
        grade_run(Path(a.grade_run))
    elif a.summarize:
        summarize()
    elif a.pilot:
        run_cell("acrobot", 5)
    elif a.extra:
        if BUDGET_USD - exp_spend() >= CAPS[25]["usd"]:
            run_cell(EXTRA_ENV, 25, attempt=2)
        else:
            print(f"extra attempt not run: remaining ${BUDGET_USD - exp_spend():.2f} < ${CAPS[25]['usd']}")
    else:
        jobs = [(e, B) for B in BUDGETS for e in (a.env or ENVS) if not finished(e, B)]
        print(f"jobs: {jobs}; exp5b spend so far ${exp_spend():.2f}", flush=True)
        with ThreadPoolExecutor(max_workers=min(2, a.parallel)) as ex:
            for f in [ex.submit(run_cell, e, B) for e, B in jobs]:
                f.result()
        print(f"exp5b spend ${exp_spend():.2f}")


if __name__ == "__main__":
    main()
