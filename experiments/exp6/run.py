"""Experiment 6 (Go): write a 243->512x6->82 ReLU Go policy from scratch with 0/10/25/50 forward-pass budgets.

Claude gets a sandbox (numpy + torch on CPU, the all-zero weights W1..W7/b1..b7 [in,out], save_weights, and for
budget > 0 a budgeted play_games(opponent, n_records) = one batch of 128 games of the SAVED weights). Grading is fixed
in experiments/exp6/GRADING.md.

    E() { taskpolicy -a env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python experiments/exp6/run.py "$@"; }
    E --show-prompt 25          # print prompt + tool descriptions (no API call)
    E --baselines               # zero net + heuristic vs the 3 opponents on EVAL_SEED -> experiments/exp6/baselines.json
    E --test-tool               # one local play_games() of the zero net (no API call)
    E --run 0                   # one cell (budget 0 = pilot)
    E --grade-run logs/exp6/<run>
    E --audit-run logs/exp6/<run>
    E --summarize               # table, figure, charts, logs/exp6/summary.json, docs/data/exp6.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from envs.go.features import FEATURE_DOC  # noqa: E402
from eval.go_eval import (CLAUDE_ARCH, EVAL_SEED, PPO_OPPONENTS, HeuristicPolicy, NetPolicy,  # noqa: E402
                          evaluate_claude, game_record, load_policy, make_forward_pass_tool, net_from_params,
                          play_match, public)
from harness.logging_utils import new_run_id, read_result, run_dir, write_result  # noqa: E402

EXP = "exp6"
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "logs" / EXP
FIG = ROOT / "docs" / "assets" / "figures"
BUDGET_USD = 40.0
EFFORT, MAX_TOKENS = "high", 64000
CAPS = {0: {"turns": 30, "usd": 5.0}, 10: {"turns": 50, "usd": 7.0}, 25: {"turns": 80, "usd": 10.0},
        50: {"turns": 130, "usd": 14.0}}
BUDGETS = [0, 10, 25, 50]
N_PARAMS = sum(CLAUDE_ARCH[k] * CLAUDE_ARCH[k + 1] + CLAUDE_ARCH[k + 1] for k in range(len(CLAUDE_ARCH) - 1))


def zero_params() -> dict[str, np.ndarray]:
    p = {}
    for k in range(1, len(CLAUDE_ARCH)):
        p[f"W{k}"] = np.zeros((CLAUDE_ARCH[k - 1], CLAUDE_ARCH[k]), np.float32)
        p[f"b{k}"] = np.zeros(CLAUDE_ARCH[k], np.float32)
    return p


# ----------------------------------------------------------------------------- prompt

RULES = """\
- 9×9 board, Black moves first, players alternate; passing is always legal.
- A chain with no liberties is removed; you capture opponent chains left without liberties by your move. Suicide
  (a move that leaves your own chain without liberties and captures nothing) is illegal.
- Simple ko: after a move that captures exactly one stone and was played on a point whose neighbours were all
  enemy stones, the captured point may not be played on the very next move.
- The game ends after two consecutive passes or after 200 moves in total (passes count as moves); it is then scored
  as it stands, with no removal of dead stones.
- Area scoring: your stones on the board plus the empty points in regions that touch only your stones. Komi 7.0
  is added to White. Higher score wins; an equal score is a draw (half a win)."""


def build_prompt(budget: int) -> str:
    turns = CAPS[budget]["turns"]
    shapes = ", ".join(f"W{k} [{CLAUDE_ARCH[k - 1]} x {CLAUDE_ARCH[k]}], b{k} [{CLAUDE_ARCH[k]}]"
                       for k in range(1, len(CLAUDE_ARCH)))
    if budget > 0:
        sandbox = (f"To see what the network actually does, you have to spend one of your forward passes: "
                   f"`play_games(opponent, n_records)` plays **one batch of 128 games** of your **currently saved** "
                   f"weights against an opponent, and you can call it at most **{budget} times** in total. Interleave "
                   "edits and forward passes however you like.")
        tools = (f"- `play_games(opponent='small_s0', n_records=3)` (callable inside `python`): one forward pass, i.e. "
                 "128 games of your saved weights against `opponent` (one of `small_s0`, `small_s2`, `large`). It "
                 "returns win/loss/draw counts and win rate (overall, as Black, as White), score margins, game "
                 "lengths and how games ended, per-move statistics for you and the opponent (passes, captures "
                 "available/taken, self-ataris, ataris escaped, own eyes filled; for you also how often your raw "
                 "argmax pointed at an illegal point, entropy, non-finite logits), and up to 8 full game records "
                 f"(worst losses first) with move lists and final boards. Each call uses one of your {budget} forward "
                 "passes whatever its arguments; `budget_status()` (free) shows how many are left.\n")
        info = "All information about how the network plays must come from `play_games()`."
        last = ("An edit saved after your last forward pass is scored untested, so make sure the weights saved when "
                "you finish are the ones you want.")
        expl = " and of what you changed after each forward pass"
    else:
        sandbox = ("In this condition you have **0 forward passes**: there is no way to run the network or watch it "
                   "play, so the weights you save are evaluated exactly as they are.")
        tools = ""
        info = "There is no way to observe the network playing in this condition."
        last = "Make sure the weights saved when you finish are the ones you want."
        expl = ""
    return f"""Write high-performing weights for a neural network that plays 9×9 Go, from scratch, without training \
or web access. The architecture is fixed: an MLP 243 → 512 × 6 → 82 ({N_PARAMS:,} weights).

You have a Python sandbox (the `python` tool), a persistent session like a notebook, where you can read and save the \
network's weight tensors (all zero at the start) and analyse them with numpy or PyTorch on a CPU, but you cannot run \
the network there. {sandbox}

# The game (rules as implemented)

{RULES}

# Network input, output and move selection

{FEATURE_DOC.strip()}

# The network

- Architecture: 243 → 512 → 512 → 512 → 512 → 512 → 512 → 82, ReLU after each of the six hidden layers, no \
activation on the output. Shapes: {shapes}.
- Forward pass: h1 = relu(x @ W1 + b1); h2 = relu(h1 @ W2 + b2); ...; h6 = relu(h5 @ W6 + b6); logits = h6 @ W7 + b7.
- Weights use the [in, out] orientation: W<k>[i, j] connects unit i of layer k's input to unit j of its output.
- Move selection: illegal moves are masked out and the legal move with the highest logit is played (ties broken \
uniformly at random); you never pick moves yourself.

# Your tools

- `python`: the saved weights are preloaded as float32 numpy arrays `W1, b1, ..., W7, b7` (and in the dict `weights`). \
Save with `save_weights(W1=..., b1=..., ...)` (only the given tensors are replaced; saving is free and unlimited) or \
with the `save_weights` text tool for small edits. `load_saved_weights()` returns what is currently saved.
{tools}- You can make at most {turns} tool calls in total; after that you will be asked for your final answer.

# How you are scored

Your final network plays 128 games against each of three opponents, `small_s0`, `small_s2` and `large` (384 games), \
with colours alternating and each pair of games starting from 2 shared random opening moves, on held-out opening \
seeds that no forward pass uses. The score is your win rate against each opponent (a draw counts half). The weights \
that count are the **last weights you save**, not the best ones you tried: whatever is saved when the session ends \
is what is scored. {last}

# Rules

- No training or gradient methods: no backpropagation, optimisers, fitting or regressing weights to targets, or any \
other optimisation of the weights against data. Every weight must be written by your own reasoning and by code that \
sets it directly.
- Do not run the network, any of its layers, or any part of it on inputs in the sandbox (no board encodings, real or \
made up, pushed through the weights, by code or by hand), and do not simulate Go games (no game engine that plays \
out moves, self-play, or tests the policy). You may write Go-logic code that helps you construct the weights (for \
example neighbour tables, or which input patterns mean a chain has one liberty), and you may analyse the weight \
arrays themselves freely. {info}
- No web access.

# Final answer

When you are done, reply with a line `NETWORK: <what your final saved network computes, at most 300 characters>`, \
followed by a short explanation of how the layers work{expl}.
"""


def tool_extra(budget: int) -> str:
    if budget > 0:
        return (f"Budget: play_games() may be called at most {budget} times in this session. The Go engine, the "
                "opponents and the final evaluation live outside this sandbox and cannot be imported here.")
    return ("Budget: 0 forward passes; there is no play_games() in this condition. The Go engine, the opponents and "
            "the final evaluation live outside this sandbox and cannot be imported here.")


# ----------------------------------------------------------------------------- spend guard

def exp_spend() -> float:
    tot = 0.0
    for p in LOGS.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def in_flight_reserve() -> float:
    """Remaining cap of runs (other processes) still marked running."""
    res = 0.0
    for p in LOGS.glob("*/result.json"):
        try:
            r = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if r.get("status") == "running" and r.get("max_cost_usd"):
            res += max(0.0, float(r["max_cost_usd"]) - float(r.get("cost_usd") or 0))
    return res


def finished(budget: int) -> list[Path]:
    return sorted(d for d in LOGS.glob(f"*-go-b{budget}-a*") if read_result(d).get("status") in ("complete", "crashed", "budget_exhausted"))


# ----------------------------------------------------------------------------- run one cell

def run_cell(budget: int, attempt: int = 1) -> Path | None:
    from harness.claude_client import BudgetExhausted, ask
    from harness.sandbox import Sandbox
    cap = CAPS[budget]["usd"]
    spent, res = exp_spend(), in_flight_reserve()
    if spent + res + cap > BUDGET_USD:
        print(f"[b{budget}] not run: spend ${spent:.2f} + in flight ${res:.2f} + cap ${cap:.2f} > ${BUDGET_USD}",
              flush=True)
        return None
    tag = f"go-b{budget}-a{attempt}"
    rid = new_run_id(tag)
    d = run_dir(EXP, rid)
    curve: list[dict] = []
    played_saves: set[int] = set()
    sb = Sandbox(zero_params(), log_dir=d, timeout=300, max_output_chars=20000, threads=2, preload=("torch",),
                 isolate=True)
    if budget > 0:
        tool = make_forward_pass_tool(sb, allowed_opponents=PPO_OPPONENTS)

        def play_games(opponent: str = "small_s0", n_records: int = 3):
            out = tool(opponent, n_records)
            k = sb.fns["play_games"].used
            played_saves.add(sb.n_saves)
            curve.append({"call": k, "opponent": opponent, "win_rate": round(out["win_rate"], 4),
                          "mean_margin": round(out["mean_margin"], 2), "save": sb.n_saves})
            write_result(d, practice_curve=list(curve), passes_used=k)
            return {"forward_pass": k, "forward_passes_left": budget - k, **out}

        sb.register("play_games", play_games, budget=budget, doc=tool.__doc__)
    sb.start()
    extra = {"label": f"Go · {budget} forward passes" if budget else "Go · 0 forward passes (one shot)",
             "group": "Go", "budget": budget, "attempt": attempt, "target": "Go 9×9", "effort": EFFORT,
             "max_turns": CAPS[budget]["turns"], "max_cost_usd": cap, "run_tag": tag}
    write_result(d, status="running", **extra)
    try:
        print(f"[{rid}] calling claude (budget {budget}, caps {CAPS[budget]})", flush=True)
        r = ask(build_prompt(budget), exp_id=EXP, run_id=rid, effort=EFFORT, max_tokens=MAX_TOKENS,
                tools=sb.tools(tool_extra(budget)), tool_handlers=sb.handlers(), max_turns=CAPS[budget]["turns"],
                condition=f"budget_{budget}", max_cost_usd=cap, extra=extra)
        print(f"[{rid}] done: turns={r.n_turns} in={r.tokens_in} out={r.usage['output_tokens']} "
              f"cost=${r.cost_usd:.3f} stop={r.stop_reason} passes={budget and sb.fns['play_games'].used}", flush=True)
    except BudgetExhausted as e:
        print(f"[{tag}] project budget exhausted: {e}", flush=True)
        write_result(d, status="budget_exhausted")
    except Exception as e:  # scored on whatever was saved (GRADING.md)
        print(f"[{tag}] crashed: {type(e).__name__}: {e}", flush=True)
        write_result(d, status="crashed", error=f"{type(e).__name__}: {e}")
    finally:
        np.savez_compressed(d / "final_weights.npz", **sb.weights)
        write_result(d, passes_used=sb.fns["play_games"].used if budget else 0, practice_curve=list(curve),
                     n_saves=sb.n_saves, played_saves=sorted(played_saves))
        sb.close()
        prune_saves(d, played_saves, sb.n_saves)
    grade_run(d)
    audit_run(d)
    return d


def prune_saves(d: Path, keep: set[int], last: int) -> None:
    """Uncompressed saves are ~6 MB each: keep (compressed) only the saves a forward pass played + the last."""
    wd = d / "weights"
    if not wd.exists():
        return
    keep = set(keep) | {last}
    for p in sorted(wd.glob("save_*.npz")):
        n = int(p.stem.split("_")[1])
        if n in keep:
            z = dict(np.load(p))
            p.unlink()
            np.savez_compressed(p, **z)
        else:
            p.unlink()
    lp = wd / "latest.npz"
    if lp.exists():
        lp.unlink()


# ----------------------------------------------------------------------------- grading

def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 4), round(c + h, 4)


def load_final(d: Path) -> dict[str, np.ndarray]:
    fw = d / "final_weights.npz"
    if not fw.exists():
        return zero_params()
    z = np.load(fw)
    return {k: z[k] for k in zero_params()}


def grade_run(d: Path) -> dict:
    res = read_result(d)
    params = load_final(d)
    t = time.time()
    ev = evaluate_claude(params, opponents=PPO_OPPONENTS + ("heuristic",))
    nz = int(sum(int((v != 0).sum()) for v in params.values()))
    win = {o: round(ev[o]["win_rate"], 4) for o in PPO_OPPONENTS}
    text = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    m = re.search(r"NETWORK:\s*(.+)", text)
    ans = (m.group(1).strip() if m else (text.strip().split("\n")[0] if text.strip() else "(no final reply)"))[:300]
    g = {"eval": ev, "winrates": win, "wilson95": {o: wilson(win[o], 128) for o in PPO_OPPONENTS},
         "vs_heuristic": round(ev["heuristic"]["win_rate"], 4), "nonzero_weights": nz,
         "pct_nonzero_weights": round(100 * nz / N_PARAMS, 3), "eval_seconds": round(time.time() - t, 1)}
    (d / "grade.json").write_text(json.dumps(g, indent=1, default=str))
    score = float(np.mean(list(win.values())))
    write_result(d, answer=ans, score=round(score, 4), correct=bool(win["small_s0"] > 0.5 and win["small_s2"] > 0.5),
                 winrates=win, wilson95=g["wilson95"], vs_heuristic=g["vs_heuristic"],
                 pct_nonzero_weights=g["pct_nonzero_weights"], n_nonzero=nz,
                 reasoning_excerpt=text.strip()[:1500])
    print(f"[{d.name}] final: {win} mean {score:.3f} vs heuristic {g['vs_heuristic']} nz {g['pct_nonzero_weights']}% "
          f"curve={[c['win_rate'] for c in res.get('practice_curve') or []]}", flush=True)
    return g


# ----------------------------------------------------------------------------- audit (regex flags; verdict by hand)

AUDIT_RX = {
    "matmul": re.compile(r"(?<![@\w])@(?!\w*\s*\()|\.dot\(|np\.dot|matmul|einsum|torch\.mm|F\.linear|nn\.Linear"),
    "activation": re.compile(r"relu|np\.maximum\s*\(|\.clamp\(|clip\(\s*[^,]+,\s*0"),
    "forward_words": re.compile(r"forward|activations?\b|logits?\s*=|evaluate|predict", re.I),
    "board_input": re.compile(r"np\.zeros\(\s*243|x\s*=\s*np\.zeros|encode\(|make_board|board\s*=\s*np\.", re.I),
    "game_sim": re.compile(r"def\s+(play|step|make_move|apply_move|remove_dead|simulate)\w*|self_?play|"
                           r"legal_moves|while\s+not\s+done|capture_stones|rollout", re.I),
    "training": re.compile(r"\.backward\(|optim\.|requires_grad|autograd|lstsq|minimize|\.fit\(|gradient", re.I),
}


def audit_run(d: Path) -> dict:
    flags = []
    tp = d / "transcript.jsonl"
    n_cells = 0
    for ln in tp.read_text().splitlines() if tp.exists() else []:
        ev = json.loads(ln)
        if ev.get("event") != "message" or ev.get("role") != "assistant":
            continue
        for b in ev.get("content", []):
            if b.get("type") == "tool_use" and b.get("name") == "python":
                n_cells += 1
                code = b["input"].get("code", "")
                hits = {}
                for k, rx in AUDIT_RX.items():
                    lines = [f"{i + 1}: {l.strip()[:160]}" for i, l in enumerate(code.splitlines()) if rx.search(l)]
                    if lines:
                        hits[k] = lines[:12]
                if hits:
                    flags.append({"turn": ev.get("turn"), "kinds": sorted(hits), "lines": hits})
    out = {"n_python_cells": n_cells, "n_flagged_cells": len(flags), "flagged": flags}
    (d / "audit_flags.json").write_text(json.dumps(out, indent=1))
    write_result(d, audit_n_flagged_cells=len(flags), audit_n_python_cells=n_cells)
    print(f"[{d.name}] audit: {len(flags)}/{n_cells} python cells flagged", flush=True)
    return out


# ----------------------------------------------------------------------------- baselines (no API)

def baselines() -> dict:
    out = {"seed": EVAL_SEED, "n_games": 128}
    z = evaluate_claude(zero_params(), opponents=PPO_OPPONENTS + ("heuristic",))
    out["zero_net"] = {o: round(z[o]["win_rate"], 4) for o in PPO_OPPONENTS + ("heuristic",)}
    h = HeuristicPolicy()
    out["heuristic"] = {o: round(public(play_match(h, load_policy(o), n_games=128, seed=EVAL_SEED))["win_rate"], 4)
                        for o in PPO_OPPONENTS}
    for k in ("zero_net", "heuristic"):
        out[k]["mean_vs_ppo"] = round(float(np.mean([out[k][o] for o in PPO_OPPONENTS])), 4)
    (HERE / "baselines.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- summary

COLORS = {"small_s0": "#2f62b8", "small_s2": "#008a7a", "large": "#8256b8"}
NAMES = {"small_s0": "small PPO, seed 0", "small_s2": "small PPO, seed 2", "large": "PPO 512×6 (same size)"}
SUMMARY_NOTES = (  # written after reading the transcripts; facts only, headline numbers are untouched
    "\n\nThe one-shot network (0 passes, 4 turns, $0.70) already beat all three opponents by a wide margin, so the "
    "forward passes added little: mean win rate 88.8% → 89.5% → 95.7% → 90.2% for 0/10/25/50 passes (2 of 3 steps "
    "up; the 25→50 drop of 5.5 pp is within single-run noise, and the b50 run's own re-test of its final weights fell "
    "from 91.8% to 83.2% vs large between two batches). Every run built the network as a hand-written Go program: "
    "board planes copied through the hidden layers, an offset 'score plane' accumulating move bonuses layer by layer, "
    "ReLU pattern units that detect chains in atari (fixed 1–3-stone shapes in b0/b10; liberty marks propagated "
    "along chains, one step per layer, in b25/b50), plus capture/save/self-atari/eye-fill/contact terms and a line "
    "prior; only 1.1–2.0% of the 1.48M weights are non-zero. With passes, Claude read loss records (mostly "
    "premature passing and dead groups as Black under komi 7) and tuned scalar weights; it found most later tweaks "
    "were within noise. Caveat: our equal-size PPO net trained on 0.71M games (the original's on 20M) and is weak — "
    "the random-move heuristic bot already wins 42% against it — so 'beats the equal-size net' is a much lower bar "
    "here than in the original. Audit: 0 violations, 0 minor breaches in 4 runs.")


def best_win_board(d: Path, opp: str = "large") -> dict | None:
    net = net_from_params(load_final(d))
    r = play_match(NetPolicy(net, name="claude"), load_policy(opp), n_games=128, seed=EVAL_SEED)
    mg = r["_margin"]
    if mg.max() <= 0:
        return None
    i = int(np.argmax(mg))
    return game_record(r["_game"], i, bool(r["_a_black"][i]), float(mg[i]), you="claude")


def make_figure(xs, cells, base, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    fig.patch.set_facecolor("white")
    for o in PPO_OPPONENTS:
        ys = [cells[b]["winrates"][o] * 100 if b in cells else np.nan for b in xs]
        ax.plot(range(len(xs)), ys, "-o", color=COLORS[o], lw=2, ms=5, label=NAMES[o])
    ax.axhline(50, color="#9a9a9a", lw=1, ls="--")
    ax.text(len(xs) - 1 + 0.08, 51, "even (50%)", va="bottom", ha="left", color="#7a7a7a", fontsize=8)
    ax.axhline(base["heuristic"]["mean_vs_ppo"] * 100, color="#b5b5b5", lw=1, ls=":")
    ax.text(len(xs) - 1 + 0.08, base["heuristic"]["mean_vs_ppo"] * 100 - 1.5, "heuristic bot\n(mean)", va="top",
            ha="left", color="#7a7a7a", fontsize=8)
    ax.axhline(base["zero_net"]["mean_vs_ppo"] * 100, color="#cfcfcf", lw=1, ls=":")
    ax.text(len(xs) - 1 + 0.08, base["zero_net"]["mean_vs_ppo"] * 100 + 1.5, "all-zero start", va="bottom",
            ha="left", color="#7a7a7a", fontsize=8)
    ax.set_xticks(range(len(xs)), [("0 (one shot)" if b == 0 else str(b)) for b in xs])
    ax.set_xlim(-0.3, len(xs) - 1 + 0.9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("forward-pass budget (batches of 128 games)")
    ax.set_ylabel("win rate of final weights (%)")
    ax.set_title("Opus 5.5's Go network: wins vs each opponent", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.33))
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", dpi=130)
    plt.close(fig)


def summarize() -> None:
    base = json.loads((HERE / "baselines.json").read_text())
    audit = json.loads((HERE / "audit.json").read_text()) if (HERE / "audit.json").exists() else {}
    cells = {}
    for b in BUDGETS:
        ds = finished(b)
        ds = [x for x in ds if read_result(x).get("score") is not None]
        if ds:
            r = read_result(ds[-1])
            cells[b] = r | {"_dir": ds[-1]}
    pc = lambda v: f"{100 * v:.1f}%"
    rows = []
    for b in BUDGETS:
        if b not in cells:
            rows.append([f"{b}", "not run", "", "", "", "", "", ""])
            continue
        r = cells[b]
        w = r["winrates"]
        rows.append([("0 (one shot)" if b == 0 else str(b)), pc(w["small_s0"]), pc(w["small_s2"]), pc(w["large"]),
                     pc(r["score"]), pc(r.get("vs_heuristic") or 0), str(r.get("passes_used", 0)),
                     f"${r.get('cost_usd', 0):.2f}"])
    for k, lab in (("zero_net", "all-zero start (reference)"), ("heuristic", "heuristic bot (reference)")):
        z = base[k]
        rows.append([lab, pc(z["small_s0"]), pc(z["small_s2"]), pc(z["large"]), pc(z["mean_vs_ppo"]),
                     pc(z["heuristic"]) if "heuristic" in z else "—", "—", "—"])
    viol = {k for k, v in audit.items() if isinstance(v, dict) and v.get("verdict") == "violation"}
    def ok(b):
        return b in cells and cells[b]["_dir"].name not in viol
    ra = 0 in cells and cells[0]["winrates"]["small_s0"] > 0.5 and cells[0]["winrates"]["small_s2"] > 0.5
    rb = 50 in cells and cells[50]["winrates"]["large"] > 0.5
    lb = lambda b, o: cells[b]["wilson95"][o][0] if b in cells else None
    means = [cells[b]["score"] if b in cells else None for b in BUDGETS]
    steps = [(BUDGETS[i], BUDGETS[i + 1], means[i + 1] - means[i]) for i in range(3)
             if means[i] is not None and means[i + 1] is not None]
    dec = {"R_A_zero_pass_beats_both_small": bool(ra),
           "R_A_wilson_lower_bounds": {o: lb(0, o) for o in ("small_s0", "small_s2")} if 0 in cells else None,
           "R_B_fifty_pass_beats_large": bool(rb), "R_B_wilson_lower_bound": lb(50, "large") if 50 in cells else None,
           "R_C_steps_mean_winrate": [{"from": a, "to": c, "delta": round(dl, 4)} for a, c, dl in steps],
           "R_C_increasing_steps": sum(dl > 0 for _, _, dl in steps),
           "R_D_all_beat_zero_net": all(cells[b]["score"] > base["zero_net"]["mean_vs_ppo"] for b in cells),
           "replicates_original": bool(ra and rb),
           "excluding_violations": {"R_A": bool(ra and ok(0)), "R_B": bool(rb and ok(50))},
           "violation_runs": sorted(viol)}
    FIG.mkdir(parents=True, exist_ok=True)
    make_figure(BUDGETS, cells, base, FIG / "exp6_winrate_vs_budget.png")
    media = [{"type": "png", "src": "assets/figures/exp6_winrate_vs_budget.png",
              "caption": "Win rate of Opus 5.5's final saved Go network vs each opponent, by forward-pass budget "
                         "(128 games per point, held-out opening seed). Grey: 50% line, the heuristic bot's mean, "
                         "and the all-zero starting network."}]
    charts = [{"type": "line", "title": "Win rate vs forward-pass budget",
               "subtitle": "final saved weights, 128 games per opponent, held-out seed",
               "x_label": "forward passes", "y_label": "win rate (%)", "x": BUDGETS,
               "series": [{"name": NAMES[o], "values": [round(100 * cells[b]["winrates"][o], 1) if b in cells else None
                                                       for b in BUDGETS]} for o in PPO_OPPONENTS],
               "reference_lines": [{"y": 50, "label": "even"},
                                   {"y": round(100 * base["heuristic"]["mean_vs_ppo"], 1),
                                    "label": "heuristic bot (mean)"}],
               "caption": "One run per budget. Budget 0 = one shot, no way to run the network."}]
    for b in (50, 25, 10):
        cv = (cells.get(b) or {}).get("practice_curve") or []
        if cv:
            charts.append({"type": "line", "title": f"What Claude saw: budget {b}, forward pass by forward pass",
                           "subtitle": "win rate of each 128-game practice batch (saved weights at that moment)",
                           "x_label": "forward pass", "y_label": "win rate (%)", "x": [c["call"] for c in cv],
                           "series": [{"name": NAMES[o], "values": [round(100 * c["win_rate"], 1)
                                                                   if c["opponent"] == o else None for c in cv]}
                                      for o in PPO_OPPONENTS if any(c["opponent"] == o for c in cv)],
                           "reference_lines": [{"y": 50, "label": "even"}],
                           "caption": "Each point is one play_games() call against the named opponent."})
            break
    board = None
    best = max(cells.values(), key=lambda r: r["winrates"]["large"]) if cells else None
    if best is not None:
        rec = best_win_board(best["_dir"])
        if rec:
            board = {"run": best["_dir"].name, "label": best.get("label"), **rec}
    total = exp_spend()
    w = lambda b: cells[b]["winrates"] if b in cells else None
    summary = (f"Starting from all-zero weights in a sandbox where the network could only be run through a budget of "
               f"128-game forward passes, Opus 5.5 wrote 243→512×6→82 Go networks whose final weights won "
               + "; ".join(f"{b} passes: {', '.join(f'{100 * w(b)[o]:.0f}%' for o in PPO_OPPONENTS)}"
                           for b in BUDGETS if b in cells)
               + " (vs small_s0, small_s2, large). "
               f"Original headline A (0 passes > 50% vs both small nets): {'yes' if ra else 'no'}. "
               f"Headline B (50 passes beats the equal-size net): {'yes' if rb else 'no'}. "
               f"Total API cost ${total:.2f}." + SUMMARY_NOTES)
    js = {"summary": summary, "status": "done" if len(cells) == len(BUDGETS) else "running",
          "table": {"caption": "Win rate of the final saved weights, 128 games per opponent (opening seed "
                               f"{EVAL_SEED}, colours alternate, draw = half). Mean = mean over the three PPO "
                               "opponents. Heuristic column = context only.",
                    "columns": ["Forward passes", "vs small_s0", "vs small_s2", "vs large (512×6)", "Mean",
                                "vs heuristic bot", "Passes used", "Cost"], "rows": rows},
          "decision": dec, "media": media, "charts": charts, "final_board_vs_large": board,
          "total_cost_usd": round(total, 4), "baselines": base}
    (LOGS / "summary.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: js[k] for k in ("summary", "table", "decision")}, indent=1, ensure_ascii=False))
    if board:
        print(board["final_board"], "\n", board["result"], board["you_are"])
    subprocess.run([sys.executable, str(ROOT / "scripts/build_blog_data.py"), "--exp", EXP], check=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-prompt", type=int)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--test-tool", action="store_true")
    ap.add_argument("--run", type=int, choices=BUDGETS)
    ap.add_argument("--grade-run")
    ap.add_argument("--audit-run")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.show_prompt is not None:
        from harness.sandbox import Sandbox
        print(build_prompt(a.show_prompt))
        print("---- python tool description ----")
        sb = Sandbox(zero_params(), log_dir=ROOT / "runs" / "_exp6_showprompt", timeout=300, preload=("torch",))
        if a.show_prompt:
            tool = make_forward_pass_tool(sb, allowed_opponents=PPO_OPPONENTS)
            sb.register("play_games", tool, budget=a.show_prompt, doc=tool.__doc__)
        print(sb.tools(tool_extra(a.show_prompt))[0]["description"])
    elif a.baselines:
        baselines()
    elif a.test_tool:
        class _S:
            weights = zero_params()
        t = time.time()
        out = make_forward_pass_tool(_S(), allowed_opponents=PPO_OPPONENTS)("large", 3)
        s = json.dumps(out)
        print(s[:2500], f"\n... {len(s)} chars, {time.time() - t:.1f}s")
    elif a.run is not None:
        if finished(a.run):
            print(f"budget {a.run} already finished: {finished(a.run)}")
            return
        run_cell(a.run)
        print(f"exp6 spend ${exp_spend():.2f}")
    elif a.grade_run:
        grade_run(Path(a.grade_run))
    elif a.audit_run:
        audit_run(Path(a.audit_run))
    elif a.summarize:
        summarize()


if __name__ == "__main__":
    main()
