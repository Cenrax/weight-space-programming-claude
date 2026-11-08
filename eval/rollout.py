"""Evaluate an MLP policy (checkpoint or arbitrary state_dict) on its environment.

Usage:
  venv/bin/python eval/rollout.py checkpoints/exp1/breakout_s0.pt --episodes 100
  venv/bin/python eval/rollout.py checkpoints/exp4/mountaincar_fail.pt --episodes 1024 --sampled
  venv/bin/python eval/rollout.py --random --env-id CartPole-v1 --env-kind raw --episodes 100

Python:
  from eval.rollout import evaluate, load_checkpoint, build_mlp
  sd, meta = load_checkpoint("checkpoints/exp5/cartpole_1M.pt")
  stats = evaluate(sd, meta, n_episodes=100, seed=10_000, greedy=True)
  # weights written by Claude: any dict with keys 0.weight,0.bias,2.weight,... (torch tensors or nested lists/np arrays)

Episode i always uses env seed `seed + i`, so results are reproducible and comparable across policies.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.wrappers import make_env  # noqa: E402

ACTS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_mlp(arch, activation="tanh") -> nn.Sequential:
    layers = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(ACTS[activation]())
    return nn.Sequential(*layers)


def load_checkpoint(pt_path):
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    js = os.path.splitext(pt_path)[0] + ".json"
    with open(js) as f:
        meta = json.load(f)
    return sd, meta


def policy_from_state_dict(state_dict, meta) -> nn.Sequential:
    net = build_mlp(meta["arch"], meta.get("activation", "tanh"))
    sd = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in state_dict.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def evaluate(state_dict, meta, n_episodes=100, seed=10_000, greedy=True, n_parallel=16, policy=None,
             max_steps=None, record_trace=False):
    """Run n_episodes; returns dict of stats. policy: optional callable obs[B,D]->actions[B] overriding the net
    (use policy='random' for a uniform-random baseline). meta needs env_id, env_kind (+arch/activation for nets)."""
    env_id, env_kind = meta["env_id"], meta.get("env_kind", "raw")
    net = None
    if policy is None:
        net = policy_from_state_dict(state_dict, meta)
    n_par = min(n_parallel, n_episodes)
    envs = [make_env(env_id, env_kind) for _ in range(n_par)]
    gen = torch.Generator().manual_seed(seed)
    next_ep = 0
    slot_ep = [-1] * n_par
    obs = [None] * n_par
    ep_ret = np.zeros(n_par)
    ep_len = np.zeros(n_par, dtype=int)
    returns = [None] * n_episodes
    lengths = [None] * n_episodes
    outcomes = [None] * n_episodes
    traces = {}
    for i in range(n_par):
        obs[i], _ = envs[i].reset(seed=seed + next_ep)
        envs[i].action_space.seed(seed + next_ep)
        slot_ep[i] = next_ep
        next_ep += 1
    active = list(range(n_par))
    while active:
        x = torch.as_tensor(np.stack([obs[i] for i in active]), dtype=torch.float32)
        if policy == "random":
            acts = [int(envs[i].action_space.sample()) for i in active]
        elif policy is not None:
            acts = [int(a) for a in policy(x.numpy())]
        else:
            logits = net(x)
            if greedy:
                acts = logits.argmax(-1).tolist()
            else:
                acts = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1).tolist()
        still = []
        for j, i in enumerate(active):
            ep = slot_ep[i]
            if record_trace and ep == 0:
                traces.setdefault(0, []).append({"obs": [round(float(v), 4) for v in obs[i]], "action": acts[j]})
            o, r, te, tr, info = envs[i].step(acts[j])
            r = info.get("true_reward", r)
            ep_ret[i] += r
            ep_len[i] += 1
            obs[i] = o
            if max_steps is not None and ep_len[i] >= max_steps:
                tr = True
            if te or tr:
                returns[ep], lengths[ep] = float(ep_ret[i]), int(ep_len[i])
                outcomes[ep] = info.get("outcome", "terminated" if te else "truncated")
                ep_ret[i], ep_len[i] = 0.0, 0
                if next_ep < n_episodes:
                    obs[i], _ = envs[i].reset(seed=seed + next_ep)
                    envs[i].action_space.seed(seed + next_ep)
                    slot_ep[i] = next_ep
                    next_ep += 1
                    still.append(i)
            else:
                still.append(i)
        active = still
    for e in envs:
        e.close()
    R = np.array(returns)
    out = {
        "episodes": n_episodes,
        "seed": seed,
        "greedy": bool(greedy) if policy is None else None,
        "mean_return": float(R.mean()),
        "std_return": float(R.std()),
        "min_return": float(R.min()),
        "max_return": float(R.max()),
        "mean_length": float(np.mean(lengths)),
        "outcomes": dict(Counter(outcomes)),
    }
    if env_id.startswith("MountainCar"):
        out["success_rate"] = out["outcomes"].get("flag", 0) / n_episodes
    if env_id.startswith("LunarLander"):
        out["landing_rate"] = out["outcomes"].get("landed", 0) / n_episodes
        out["success_rate_200"] = float((R >= 200).mean())
    if record_trace:
        out["trace_episode0"] = traces.get(0, [])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pt", nargs="?", help="checkpoint .pt (json alongside)")
    p.add_argument("--meta", help="json metadata to use with an arbitrary state_dict .pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--sampled", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--random", action="store_true", help="uniform random policy baseline")
    p.add_argument("--env-id")
    p.add_argument("--env-kind", default="raw")
    p.add_argument("--parallel", type=int, default=16)
    a = p.parse_args()
    if a.random:
        meta = {"env_id": a.env_id, "env_kind": a.env_kind}
        stats = evaluate(None, meta, a.episodes, a.seed, policy="random", n_parallel=a.parallel)
    else:
        sd = torch.load(a.pt, map_location="cpu", weights_only=True)
        meta = json.load(open(a.meta or os.path.splitext(a.pt)[0] + ".json"))
        stats = evaluate(sd, meta, a.episodes, a.seed, greedy=not a.sampled, n_parallel=a.parallel)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
