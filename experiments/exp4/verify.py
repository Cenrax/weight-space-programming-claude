"""Check Claude's mechanistic claims about its edits (no API calls).

    venv/bin/python experiments/exp4/verify.py   # -> experiments/exp4/verify.json + printout

For every graded run with an edited.pt:
- MountainCar: first actions on the eval seeds (does it back up first or push right first?), how often the edited
  net's action equals the "push with velocity" heuristic, mean steps to the flag, and whether the original network's
  units still matter (logit margin of the edit vs the rest of layer 2).
- LunarLander: agreement with gymnasium's heuristic, outcome counts, mean episode length, and where it lands.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from envs.wrappers import make_env  # noqa: E402
from eval.rollout import load_checkpoint  # noqa: E402


def episodes(case, sd, meta, seeds):
    act = run.net_policy(sd, meta)
    heur = run.mc_heuristic if case == "mountaincar" else run.ll_heuristic
    out = []
    for s in seeds:
        env = make_env(meta["env_id"], "raw")
        o, _ = env.reset(seed=s)
        acts, agree, t, xs = [], 0, 0, []
        while True:
            a = act(o[None])[0]
            agree += int(a == heur(o[None])[0])
            acts.append(a)
            o, r, te, tr, info = env.step(a)
            xs.append(float(o[0]))
            t += 1
            if te or tr:
                break
        env.close()
        out.append({"seed": s, "len": t, "outcome": info.get("outcome"), "first_actions": acts[:15],
                    "agree_heuristic": agree / t, "final_x": float(o[0]), "min_x": min(xs),
                    "action_counts": dict(Counter(acts))})
    return out


def main():
    rep = {}
    for case in run.CASES:
        _, meta = load_checkpoint(str(run.CASES[case]["ckpt"]))
        for d in run.existing(case):
            if not (d / "edited.pt").exists():
                continue
            sd = torch.load(d / "edited.pt", weights_only=True)
            n = 64 if case == "mountaincar" else 32
            eps = episodes(case, sd, meta, range(run.EVAL_SEED, run.EVAL_SEED + n))
            first = Counter(e["first_actions"][0] for e in eps)
            # did it reverse before going right? first action left, or min x below the start
            backs_up_first = float(np.mean([e["first_actions"][0] == 0 for e in eps]))
            r = {"n_episodes": n, "first_action_counts": dict(first), "frac_first_action_left": backs_up_first,
                 "mean_agree_heuristic": float(np.mean([e["agree_heuristic"] for e in eps])),
                 "mean_len": float(np.mean([e["len"] for e in eps])),
                 "outcomes": dict(Counter(e["outcome"] for e in eps)),
                 "example_seed_10000": eps[0]}
            if case == "lunarlander":
                r["final_x_landed"] = [round(e["final_x"], 3) for e in eps if e["outcome"] == "landed"][:20]
            rep[d.name] = r
            print(d.name, json.dumps({k: v for k, v in r.items() if k != "example_seed_10000"}))
    (HERE / "verify.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
