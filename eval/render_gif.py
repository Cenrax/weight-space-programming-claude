"""Render one greedy episode of a checkpoint to a small GIF in docs/assets/gifs/.

  venv/bin/python eval/render_gif.py checkpoints/exp1/breakout_s0.pt
  venv/bin/python eval/render_gif.py checkpoints/exp4/mountaincar_fail.pt --out docs/assets/gifs/exp4_mc_fail.gif
  venv/bin/python eval/render_gif.py --random --env-id Pong --env-kind atari_ram --out docs/assets/gifs/pong_random.gif

Defaults: 15 fps, <= 20 s (300 frames), frame_skip chosen so the clip spans the episode (classic control renders every
env step, so long episodes are subsampled), width <= 240 px; if the file exceeds --max-mb it is re-encoded smaller.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.wrappers import make_env  # noqa: E402
from eval.rollout import policy_from_state_dict  # noqa: E402


@torch.no_grad()
def collect_frames(sd, meta, seed=10_000, max_steps=27_000, random_policy=False):
    env = make_env(meta["env_id"], meta.get("env_kind", "raw"), render_mode="rgb_array")
    net = None if random_policy else policy_from_state_dict(sd, meta)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    frames, ret, done, t = [env.render()], 0.0, False, 0
    while not done and t < max_steps:
        a = env.action_space.sample() if random_policy else int(net(torch.as_tensor(obs)[None]).argmax())
        obs, r, te, tr, info = env.step(a)
        ret += r
        frames.append(env.render())
        done = te or tr
        t += 1
    env.close()
    return frames, ret, t


def save_gif(frames, out, fps=15, max_seconds=20, width=240, max_mb=2.0, every=None):
    max_frames = fps * max_seconds
    if every is None:
        every = max(1, int(np.ceil(len(frames) / max_frames)))
    sel = frames[::every][:max_frames]
    while True:
        h, w = sel[0].shape[:2]
        s = min(1.0, width / w)
        ims = [Image.fromarray(f).resize((int(w * s), int(h * s)), Image.NEAREST if s >= 1 else Image.BILINEAR)
               .convert("P", palette=Image.ADAPTIVE, colors=64) for f in sel]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
        mb = os.path.getsize(out) / 1e6
        if mb <= max_mb or width <= 96:
            return mb, len(sel), every
        width = int(width * 0.8)
        if len(sel) > 150:
            sel = sel[: int(len(sel) * 0.8)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pt", nargs="?")
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-seconds", type=int, default=20)
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--every", type=int, default=None, help="keep every k-th env frame (default: fit episode)")
    p.add_argument("--max-mb", type=float, default=2.0)
    p.add_argument("--random", action="store_true")
    p.add_argument("--env-id")
    p.add_argument("--env-kind", default="raw")
    a = p.parse_args()
    if a.random:
        sd, meta = None, {"env_id": a.env_id, "env_kind": a.env_kind}
        name = f"{a.env_id.lower()}_random"
    else:
        sd = torch.load(a.pt, map_location="cpu", weights_only=True)
        meta = json.load(open(os.path.splitext(a.pt)[0] + ".json"))
        exp = os.path.basename(os.path.dirname(os.path.abspath(a.pt)))
        name = f"{exp}_{os.path.splitext(os.path.basename(a.pt))[0]}"
    out = a.out or os.path.join(ROOT, "docs", "assets", "gifs", name + ".gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames, ret, steps = collect_frames(sd, meta, a.seed, random_policy=a.random)
    # Atari (frameskip 4) at 15 fps is real time; classic control: fit the whole episode into the clip
    every = a.every or (1 if meta.get("env_kind") == "atari_ram" else None)
    mb, n, ev = save_gif(frames, out, a.fps, a.max_seconds, a.width, a.max_mb, every)
    print(json.dumps({"out": os.path.relpath(out, ROOT), "episode_return": ret, "episode_steps": steps,
                      "gif_frames": n, "every": ev, "mb": round(mb, 2)}))


if __name__ == "__main__":
    main()
