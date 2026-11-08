"""Uniform-random-policy baselines (same seeded episodes as checkpoint evals: env seed 10_000 + i).

  venv/bin/python train/random_baselines.py   -> checkpoints/<exp>/random_baseline.json for exp1, exp1b, exp4, exp5
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from eval.rollout import evaluate  # noqa: E402

SETS = {
    "exp1": [("Breakout", "atari_ram"), ("Pong", "atari_ram"), ("MsPacman", "atari_ram"), ("Freeway", "atari_ram")],
    "exp1b": [(e, "pad8x4") for e in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3"]],
    "exp4": [("MountainCar-v0", "raw", 1024), ("LunarLander-v3", "raw", 1024)],
    "exp5": [(e, "raw") for e in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3"]] + [("Breakout", "atari_ram")],
}


def main():
    only = sys.argv[1:] or list(SETS)
    for exp in only:
        res = {}
        for item in SETS[exp]:
            env_id, kind = item[:2]
            n = item[2] if len(item) > 2 else 100
            s = evaluate(None, {"env_id": env_id, "env_kind": kind}, n_episodes=n, seed=10_000, policy="random")
            res[env_id] = s
            print(exp, env_id, f"{s['mean_return']:.2f}±{s['std_return']:.2f}", s["outcomes"], flush=True)
        os.makedirs(os.path.join(ROOT, "checkpoints", exp), exist_ok=True)
        with open(os.path.join(ROOT, "checkpoints", exp, "random_baseline.json"), "w") as f:
            json.dump({"policy": "uniform random", "eval_seed": 10_000, "results": res}, f, indent=1)


if __name__ == "__main__":
    main()
