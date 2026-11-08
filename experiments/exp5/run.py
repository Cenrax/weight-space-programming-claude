"""Experiment 5: can Claude write a policy network's weights from scratch, zero-shot (one reply, no tools, no rollouts)?

Run from the repo root (keep CPU use moderate: <= 3 eval processes):

    E="taskpolicy -a env VECLIB_MAXIMUM_THREADS=1 venv/bin/python experiments/exp5/run.py"
    venv/bin/python experiments/exp5/build_env_docs.py
    $E --show-prompt cartpole            # print the prompt (no API call)
    $E --baselines [--env breakout]      # zero net, PPO recheck, heuristics (no API calls) -> baselines.json
    $E --pilot                           # ONE CartPole call (= cartpole attempt 1 if settings unchanged)
    $E --parallel 3                      # all remaining attempts (3/3/3/5/5); skips existing ones
    $E --grade-only [--env X]            # re-parse + re-evaluate saved responses
    $E --grade-run logs/exp5/<run>       # grade one run (used as a subprocess by the runner)
    $E --summarize                       # GIFs, logs/exp5/summary.json, docs/data/exp5.json

Grading rules are fixed in experiments/exp5/GRADING.md.
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

EXP = "exp5"
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "logs" / EXP
CK = ROOT / "checkpoints" / EXP
GIFS = ROOT / "docs" / "assets" / "gifs"
BUDGET_USD = 18.0
N_EVAL, EVAL_SEED = 100, 10_000
SETTINGS = {"effort": "high", "max_tokens": 64000, "thinking": True}  # see GRADING.md (pre-run changes)
CONDITION = "zero-shot write from scratch, no tools, no rollouts"

ENVS = {
    "cartpole": {"group": "CartPole", "env_id": "CartPole-v1", "final": "5M", "attempts": 3,
                 "obs_names": ["cart position", "cart velocity", "pole angle (rad)", "pole angular velocity"],
                 "actions": "0 = push cart to the left, 1 = push cart to the right",
                 "objective": "keep the pole up for the full 500 steps in every episode (return = steps survived)",
                 "est": 1.35},
    "acrobot": {"group": "Acrobot", "env_id": "Acrobot-v1", "final": "5M", "attempts": 3,
                "obs_names": ["cos(theta1)", "sin(theta1)", "cos(theta2)", "sin(theta2)", "dtheta1", "dtheta2"],
                "actions": "0 = apply -1 torque, 1 = apply 0 torque, 2 = apply +1 torque (to the actuated joint)",
                "objective": "swing the free end above the target height in as few steps as possible "
                             "(return = -steps until success, -500 if it never succeeds)",
                "est": 1.35},
    "mountaincar": {"group": "MountainCar", "env_id": "MountainCar-v0", "final": "5M", "attempts": 3,
                    "obs_names": ["position", "velocity"],
                    "actions": "0 = accelerate left, 1 = don't accelerate, 2 = accelerate right",
                    "objective": "reach the flag (position >= 0.5) in as few steps as possible "
                                 "(return = -steps, -200 if it never reaches the flag)",
                    "est": 1.35},
    "lunarlander": {"group": "LunarLander", "env_id": "LunarLander-v3", "final": "10M", "attempts": 5,
                    "obs_names": ["x", "y", "vx", "vy", "angle", "angular velocity", "left leg contact",
                                  "right leg contact"],
                    "actions": "0 = do nothing, 1 = fire left orientation engine, 2 = fire main engine, "
                               "3 = fire right orientation engine",
                    "objective": "land safely on the landing pad and come to rest, maximising the episode return "
                                 "(an episode scoring >= 200 is considered a solution)",
                    "est": 1.35},
    "breakout": {"group": "Breakout", "env_id": "Breakout", "final": "10M", "attempts": 5,
                 "obs_names": None,
                 "actions": "the 18 actions in the table above (0 = NOOP, 1 = FIRE, 3 = RIGHT, 4 = LEFT, ...)",
                 "objective": "score as many points as possible per episode (game = 5 lives)",
                 "est": 1.45},
}


def meta_of(env: str) -> dict:
    m = json.loads((CK / f"{env}_100k.json").read_text())
    return {k: m[k] for k in ("env_id", "env_kind", "arch", "activation", "obs_preprocess")}


def zero_params(env: str):
    a = meta_of(env)["arch"]
    from collections import OrderedDict
    p = OrderedDict()
    for k in range(len(a) - 1):
        p[f"W{k + 1}"] = np.zeros((a[k], a[k + 1]), np.float32)
        p[f"b{k + 1}"] = np.zeros(a[k + 1], np.float32)
    return p


def params_to_sd(p) -> dict:
    sd = {}
    for k in range(len(p) // 2):
        sd[f"{2 * k}.weight"] = torch.as_tensor(np.ascontiguousarray(p[f"W{k + 1}"].T))  # torch is [out, in]
        sd[f"{2 * k}.bias"] = torch.as_tensor(p[f"b{k + 1}"])
    return sd


# ----------------------------------------------------------------------------- heuristics (memorisation flag)

class _Disc:  # gymnasium's heuristic() only reads env.unwrapped.continuous
    unwrapped = type("U", (), {"continuous": False})()


def h_lunarlander(obs):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    return [int(heuristic(_Disc, o)) for o in obs]


HEURISTICS = {
    "mountaincar": (lambda obs: [2 if o[1] >= 0 else 0 for o in obs], "push in the direction of velocity (2 if v >= 0 else 0)", True),
    "lunarlander": (h_lunarlander, "gymnasium.envs.box2d.lunar_lander.heuristic (discrete)", True),
    "cartpole": (lambda obs: [1 if o[2] + o[3] > 0 else 0 for o in obs], "1 if theta + theta_dot > 0 else 0", False),
    "acrobot": (lambda obs: [2 if o[4] > 0 else 0 for o in obs], "2 if dtheta1 > 0 else 0", False),
}


def net_act(sd, meta):
    net = policy_from_state_dict(sd, meta)

    @torch.no_grad()
    def act(obs):
        return net(torch.as_tensor(np.asarray(obs), dtype=torch.float32)).argmax(-1).tolist()
    return act


def visited_states(act_fn, meta, n=N_EVAL, seed=EVAL_SEED, n_par=16):
    """Same seeding/lockstep scheme as eval.rollout.evaluate; returns every observation the policy acted on."""
    envs = [make_env(meta["env_id"], meta["env_kind"]) for _ in range(min(n_par, n))]
    obs, nxt, seen = [None] * len(envs), 0, []
    for i, e in enumerate(envs):
        obs[i], _ = e.reset(seed=seed + nxt)
        e.action_space.seed(seed + nxt)
        nxt += 1
    active = list(range(len(envs)))
    while active:
        X = np.stack([obs[i] for i in active])
        seen.append(X)
        acts = act_fn(X)
        still = []
        for a, i in zip(acts, active):
            o, r, te, tr, _ = envs[i].step(int(a))
            obs[i] = o
            if te or tr:
                if nxt < n:
                    obs[i], _ = envs[i].reset(seed=seed + nxt)
                    envs[i].action_space.seed(seed + nxt)
                    nxt += 1
                    still.append(i)
            else:
                still.append(i)
        active = still
    for e in envs:
        e.close()
    return np.concatenate(seen)


def heuristic_agreement(env, sd, meta):
    if env not in HEURISTICS:
        return None
    act = net_act(sd, meta)
    S = visited_states(act, meta)
    a, h = np.array(act(S)), np.array(HEURISTICS[env][0](S))
    return float((a == h).mean())


def success_of(env: str, st: dict):
    o, n = st["outcomes"], st["episodes"]
    return {"cartpole": lambda: o.get("timeout", 0) / n, "acrobot": lambda: o.get("swung_up", 0) / n,
            "mountaincar": lambda: o.get("flag", 0) / n, "lunarlander": lambda: st["success_rate_200"],
            "breakout": lambda: None}[env]()


def score_net(env: str, sd=None, meta=None, policy=None) -> dict:
    meta = meta or meta_of(env)
    st = evaluate(sd, meta, n_episodes=N_EVAL, seed=EVAL_SEED, greedy=True, policy=policy)
    st["success_rate"] = success_of(env, st)
    return st


# ----------------------------------------------------------------------------- prompt

def zero_template(env: str) -> str:
    p = zero_params(env)
    lines = []
    for name, a in p.items():
        if a.ndim == 2:
            lines += [f"{name} row {i}  " + " ".join("0" for _ in range(a.shape[1])) for i in range(a.shape[0])]
        else:
            lines.append(f"{name}  " + " ".join("0" for _ in range(a.shape[0])))
    return "\n".join(lines)


def build_prompt(env: str) -> str:
    c = ENVS[env]
    meta = meta_of(env)
    arch = meta["arch"]
    n_w = sum(arch[k] * arch[k + 1] + arch[k + 1] for k in range(len(arch) - 1))
    doc = (HERE / "prompts" / f"env_{env}.md").read_text().strip()
    if meta["env_kind"] == "atari_ram":
        in_desc = ("the 128 RAM bytes divided by 255: x[i] = RAM[i] / 255 for i = 0..127, so every input is in "
                   "[0, 1]. You are not told what any RAM byte means")
        game = "Breakout"
    else:
        in_desc = "the raw observation as float32, no scaling or normalisation (" + \
            ", ".join(f"x[{i}] = {n}" for i, n in enumerate(c["obs_names"])) + ")"
        game = c["env_id"]
    shapes = ", ".join(f"W{k + 1} [{arch[k]} x {arch[k + 1]}], b{k + 1} [{arch[k + 1]}]" for k in range(len(arch) - 1))
    return f"""Write, from scratch, the weights of a neural-network policy that plays {game} as well as possible.

There is no training: you choose every number yourself. You have no tools: you cannot run code, run the network or \
run the environment, and you get no data, losses, gradients or test episodes. Work it out by reasoning and give the \
complete answer in this one reply. The network you write will then be evaluated, unchanged, on many episodes.

{doc}

# The policy network

- Architecture: MLP {' -> '.join(map(str, arch))} with tanh hidden activations ({n_w} weights). The architecture is \
fixed. Shapes: {shapes}.
- Input x: {in_desc}.
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index), at every step. Actions: {c['actions']}.
- Weights use the [in, out] orientation of the forward pass above: W<k>[i, j] connects input unit i of layer k to \
output unit j. `W<k> row i` lists W<k>[i, 0], W<k>[i, 1], ... . `b<k>` is the bias of layer k.
- Goal: {c['objective']}. It is evaluated by the mean return over many episodes with random starting conditions as \
described in the environment.

## The network with every weight at 0 (the template you fill in)

```
{zero_template(env)}
```

# Answer format

- First a line `CONTROLLER: <the controller your weights implement, at most 300 characters>`.
- Then your explanation of how the network works.
- End your reply with exactly one machine-readable block of weights, delimited like this:

WEIGHTS:
W1[2,0] = 12.0
b1[0] = -3.5
W2 row 0  <{arch[2]} numbers>
b3 <{arch[3]} numbers>
END_WEIGHTS

Rules for the WEIGHTS block: one entry per line, no comments or prose inside the block; indices are 0-based; \
`W<k>[i,j] = v` sets the weight from input unit i to output unit j of layer k (same orientation as the template \
rows); `b<k>[j] = v` sets one bias; `W<k> row i <values>` sets a whole row (exactly as many values as the template \
row); `b<k> <values>` sets a whole bias vector. **Every weight you do not specify stays 0.** Only the last WEIGHTS \
block in your reply is used.
"""


# ----------------------------------------------------------------------------- parsing + grading

_LEAD = re.compile(r"^\s*(?:[-*>]+\s+)?([Wb]\d+)\b")
_START = re.compile(r"(?<!END_)WEIGHTS:")


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


def extract_block(text: str):
    idx = [m.end() for m in _START.finditer(text)]
    if not idx:
        return None, []
    body = text[idx[-1]:]
    end = body.find("END_WEIGHTS")
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


def breakout_ram_usage(p) -> dict:
    W1 = np.asarray(p["W1"])
    rows = [int(i) for i in np.where(np.abs(W1).sum(1) > 0)[0]]
    known = {72: "player_x", 99: "ball_x", 101: "ball_y", 57: "lives", 84: "score", 77: "blocks_hit_count"}
    return {"ram_bytes_used": rows, "n_ram_bytes_used": len(rows),
            "annotated": {str(k): v for k, v in known.items() if k in rows},
            "per_byte_l1": {str(i): round(float(np.abs(W1[i]).sum()), 3) for i in rows}}


def baselines() -> dict:
    p = HERE / "baselines.json"
    return json.loads(p.read_text()) if p.exists() else {}


def grade_run(d: Path) -> dict:
    res = read_result(d)
    env = res["env"]
    meta = meta_of(env)
    base = zero_params(env)
    text = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    block, bad = extract_block(text)
    g = {"env": env, "has_weights_block": block is not None, "unparsed_lines": bad, "parse_error": None,
         "stop_reason": res.get("stop_reason")}
    params = None
    if block is None:
        g["parse_error"] = "no WEIGHTS: block"
    elif bad:
        g["parse_error"] = f"{len(bad)} unrecognised W/b line(s), e.g. {bad[0][:120]!r}"
    else:
        try:
            params = wt.parse_weights(block, base=base, strict=True)
            if not all(np.isfinite(v).all() for v in params.values()):
                g["parse_error"], params = "NaN/inf in weights", None
        except ValueError as e:
            g["parse_error"] = f"strict parse failed: {e}"
    agree = None
    if params is not None:
        g["n_weight_lines"] = len([ln for ln in block.splitlines() if _LEAD.match(ln)])
        sd = params_to_sd(params)
        torch.save(sd, d / "edited.pt")
        (d / "edited.json").write_text(json.dumps({**meta, "algo": "claude-zero-shot", "notes":
                                                   f"exp5 weights written by Claude ({d.name})"}, indent=1))
        (d / "weights.txt").write_text(block + "\n")
        g["nonzero"] = wt.diff_params(base, params)
        st = score_net(env, sd, meta)
        agree = heuristic_agreement(env, sd, meta)
        if env == "breakout":
            g["ram_usage"] = breakout_ram_usage(params)
        g["stats"] = st
        score, succ = st["mean_return"], st["success_rate"]
    else:
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
            psd = params_to_sd(part)
            torch.save(psd, d / "partial_edited.pt")
            g["partial_parse_stats"] = score_net(env, psd, meta)
        z = baselines()[env]["zero"]
        score, succ = z["mean_return"], z["success_rate"]
    ppo100k = json.loads((CK / f"{env}_100k.json").read_text())["eval_mean_return"]
    (d / "grade.json").write_text(json.dumps(g, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    m = re.search(r"CONTROLLER:\s*(.+)", text)
    ans = (m.group(1).strip() if m else text.strip().split("\n")[0])[:300]
    cut = _START.search(text)
    expl = text[: cut.start()] if cut else text
    write_result(d, target=ENVS[env]["env_id"], answer=ans, score=round(float(score), 3),
                 correct=bool(params is not None and score > ppo100k), success_rate=succ,
                 heuristic_agreement=None if agree is None else round(agree, 4),
                 parse_ok=params is not None, parse_error=g["parse_error"],
                 partial_parse_score=(g.get("partial_parse_stats") or {}).get("mean_return"),
                 max_return=(g.get("stats") or {}).get("max_return"),
                 landing_rate=(g.get("stats") or {}).get("landing_rate"),
                 outcomes=(g.get("stats") or {}).get("outcomes"),
                 n_weight_lines=g.get("n_weight_lines"), n_nonzero=(g.get("nonzero") or {}).get("changed"),
                 ram_bytes_used=(g.get("ram_usage") or {}).get("ram_bytes_used"),
                 reasoning_excerpt=expl.strip()[:1500])
    print(f"[{d.name}] parse_ok={params is not None} return={score:.2f} success={succ} agree={agree} "
          f"(PPO-100k {ppo100k:.2f}) {g['parse_error'] or ''}", flush=True)
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


def existing(env: str) -> list[Path]:
    return sorted(d for d in LOGS.glob(f"*-{env}-a*") if (d / "result.json").exists()
                  and read_result(d).get("status") == "complete")


_LOCK = threading.Lock()
_RESERVED = [0.0]


def run_attempt(env: str, n: int) -> Path | None:
    from harness.claude_client import BudgetExhausted, ask
    est = ENVS[env]["est"]
    with _LOCK:
        spent = exp_spend()
        if spent + _RESERVED[0] + est > BUDGET_USD:
            print(f"[{env} a{n}] skipped: spend ${spent:.2f} + in flight ${_RESERVED[0]:.2f} + est ${est:.2f} "
                  f"> ${BUDGET_USD}", flush=True)
            return None
        _RESERVED[0] += est
    try:
        rid = new_run_id(f"{env}-a{n}")
        print(f"[{rid}] calling claude ({SETTINGS})", flush=True)
        r = ask(build_prompt(env), exp_id=EXP, run_id=rid, condition=CONDITION, max_cost_usd=BUDGET_USD - spent,
                effort=SETTINGS["effort"], max_tokens=SETTINGS["max_tokens"], thinking=SETTINGS["thinking"],
                extra={"label": f"{ENVS[env]['group']} · attempt {n}", "group": ENVS[env]["group"], "env": env,
                       "attempt": n, "effort": SETTINGS["effort"]})
        print(f"[{rid}] done: in={r.tokens_in} out={r.usage['output_tokens']} cost=${r.cost_usd:.3f} "
              f"stop={r.stop_reason}", flush=True)
    except BudgetExhausted as e:
        print(f"[{env} a{n}] project budget exhausted: {e}", flush=True)
        return None
    finally:
        with _LOCK:
            _RESERVED[0] -= est
    subprocess.run([sys.executable, __file__, "--grade-run", str(r.log_dir)], check=False)
    return r.log_dir


def do_baselines(envs) -> None:
    out = baselines()
    rb = json.loads((CK / "random_baseline.json").read_text())["results"]
    for env in envs:
        meta = meta_of(env)
        b = out.get(env, {})
        zsd = params_to_sd(zero_params(env))
        b["zero"] = score_net(env, zsd, meta)
        b["random"] = rb[meta["env_id"]] | {"success_rate": success_of(env, rb[meta["env_id"]] | {
            "success_rate_200": rb[meta["env_id"]].get("success_rate_200")})}
        for tag in ("100k", "1M", ENVS[env]["final"]):
            sd, m = load_checkpoint(str(CK / f"{env}_{tag}.pt"))
            st = score_net(env, sd, m)
            st["stored_mean_return"] = m["eval_mean_return"]
            st["heuristic_agreement"] = heuristic_agreement(env, sd, m)
            b[f"ppo_{tag}"] = st
        if env in HEURISTICS:
            fn, desc, primary = HEURISTICS[env]
            b["heuristic"] = score_net(env, None, meta, policy=fn) | {"desc": desc, "primary_flag": primary}
        out[env] = b
        print(env, json.dumps({k: (v.get("mean_return"), v.get("success_rate")) for k, v in b.items()
                               if isinstance(v, dict)}), flush=True)
        (HERE / "baselines.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- summary

def summarize() -> None:
    from eval.render_gif import collect_frames, save_gif
    base = baselines()
    rows, charts, media, per = [], [], [], {}
    fmt = lambda v: "—" if v is None else (f"{v:.1f}" if abs(v) < 1000 else f"{v:.0f}")
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    for env, c in ENVS.items():
        res = sorted((read_result(d) | {"_dir": d} for d in existing(env)), key=lambda r: r.get("attempt") or 0)
        if not res:
            continue
        scores = [r["score"] for r in res]
        best = max(res, key=lambda r: (r["score"], r.get("success_rate") or 0, -(r.get("attempt") or 0)))
        med = float(np.median(scores))
        b = base[env]
        fin = ENVS[env]["final"]
        agrees = [r.get("heuristic_agreement") for r in res if r.get("heuristic_agreement") is not None]
        per[env] = {"res": res, "best": best, "median": med}
        sr = lambda st: "" if st.get("success_rate") is None else f" ({pct(st['success_rate'])})"
        agree_cell = "—" if not agrees else (f"best {pct(best.get('heuristic_agreement'))}, median "
                                             f"{pct(float(np.median(agrees)))}" + ("" if HEURISTICS[env][2] else " (secondary)"))
        rows.append([c["group"], fmt(b["random"]["mean_return"]) + sr(b["random"]),
                     fmt(b["ppo_100k"]["mean_return"]) + sr(b["ppo_100k"]),
                     fmt(b["ppo_1M"]["mean_return"]) + sr(b["ppo_1M"]),
                     fmt(b[f"ppo_{fin}"]["mean_return"]) + sr(b[f"ppo_{fin}"]) + f" [{fin}]",
                     fmt(best["score"]) + ("" if best.get("success_rate") is None else f" ({pct(best['success_rate'])})"),
                     fmt(med), agree_cell])
        charts.append({"type": "bar", "title": f"{c['group']}: mean return per attempt vs PPO",
                       "subtitle": "100 seeded greedy episodes (seeds 10000–10099)", "unit": "mean return",
                       "categories": [f"attempt {r.get('attempt')}" for r in res] + ["PPO 100k", "PPO 1M", f"PPO {fin}",
                                                                                   "random", "zero net"],
                       "series": [{"name": "mean return", "values": [round(r["score"], 1) for r in res] +
                                   [round(b[k]["mean_return"], 1) for k in ("ppo_100k", "ppo_1M", f"ppo_{fin}",
                                                                            "random", "zero")]}],
                       "caption": f"Opus 5.5 one-shot attempts (no tools, no rollouts) next to PPO checkpoints."})
        bd = best["_dir"]
        if (bd / "edited.pt").exists():
            sd = torch.load(bd / "edited.pt", weights_only=True)
            meta = json.loads((bd / "edited.json").read_text())
            atari = meta["env_kind"] == "atari_ram"
            frames, ret, steps = collect_frames(sd, meta, seed=EVAL_SEED, max_steps=300 if atari else 27_000)
            out = GIFS / f"exp5_claude_{env}.gif"
            mb, nfr, ev = save_gif(frames, str(out), every=1 if atari else None)
            print(f"gif {out.name}: {mb:.2f} MB, return {ret:.1f}, {steps} steps")
            media.append({"type": "gif", "src": f"assets/gifs/exp5_claude_{env}.gif", "group": c["group"],
                          "caption": f"{c['group']}: Opus 5.5's best zero-shot network ({best.get('label')}), seed "
                                     f"{EVAL_SEED}" + (f", first {steps} agent steps (return so far {ret:.0f})" if atari
                                                       else f", episode return {ret:.1f}") +
                                     f"; mean return {best['score']:.1f} over 100 episodes."})
    beat100k = [e for e in per if per[e]["median"] > base[e]["ppo_100k"]["mean_return"]]
    best_ge_fin = [e for e in per if per[e]["best"]["score"] >= base[e][f"ppo_{ENVS[e]['final']}"]["mean_return"]]
    med_ge_fin = [e for e in per if per[e]["median"] >= base[e][f"ppo_{ENVS[e]['final']}"]["mean_return"]]
    total = exp_spend()
    g = lambda L: ", ".join(ENVS[e]["group"] for e in L) or "none"
    summary = (f"Writing every weight in one reply with no tools or rollouts, Opus 5.5's median network beat PPO-100k "
               f"on {len(beat100k)}/{len(per)} environments ({g(beat100k)}; H1 needs ≥ 4) and its best network "
               f"matched or beat the final PPO checkpoint on {len(best_ge_fin)}/{len(per)} ({g(best_ge_fin)}; median: "
               f"{len(med_ge_fin)}/{len(per)}). Total API cost ${total:.2f}.")
    js = {"summary": summary, "status": "done",
          "table": {"caption": "Mean return over 100 seeded greedy episodes (success rate in parentheses: CartPole 500 "
                               "steps, Acrobot goal, MountainCar flag, LunarLander return ≥ 200). Heuristic agreement = "
                               "share of visited steps where the network's action equals a hand-coded controller's "
                               "(MountainCar: push with velocity; LunarLander: Gymnasium heuristic(); CartPole/Acrobot: "
                               "secondary reference rules).",
                    "columns": ["Env", "Random", "PPO 100k", "PPO 1M", "PPO final", "Claude best", "Claude median",
                                "Heuristic agreement"], "rows": rows},
          "decision": {"H1_median_beats_ppo100k": {"envs": beat100k, "count": len(beat100k), "pass": len(beat100k) >= 4},
                       "H2_best_ge_ppo_final": {"envs": best_ge_fin, "count": len(best_ge_fin)},
                       "median_ge_ppo_final": {"envs": med_ge_fin, "count": len(med_ge_fin)}},
          "media": media, "charts": charts, "total_cost_usd": round(total, 4)}
    (LOGS / "summary.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    print(json.dumps({k: js[k] for k in ("summary", "table", "decision")}, indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(ROOT / "scripts/build_blog_data.py"), "--exp", EXP], check=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--env", choices=list(ENVS), action="append")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--grade-run")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--show-prompt", choices=list(ENVS))
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    envs = a.env or list(ENVS)
    if a.show_prompt:
        print(build_prompt(a.show_prompt))
    elif a.baselines:
        do_baselines(envs)
    elif a.grade_run:
        grade_run(Path(a.grade_run))
    elif a.grade_only:
        for env in envs:
            for d in existing(env):
                grade_run(d)
    elif a.summarize:
        summarize()
    elif a.pilot:
        run_attempt("cartpole", 1)
    else:
        jobs = []
        for env in envs:
            have = {read_result(d).get("attempt") for d in existing(env)}
            jobs += [(env, n) for n in range(1, ENVS[env]["attempts"] + 1) if n not in have]
        print(f"jobs: {jobs}; exp5 spend so far ${exp_spend():.2f}", flush=True)
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            for f in [ex.submit(run_attempt, e, n) for e, n in jobs]:
                f.result()
        print(f"exp5 spend ${exp_spend():.2f}")


if __name__ == "__main__":
    main()
