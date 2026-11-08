"""Single-file CleanRL-style PPO (discrete actions) with a plain MLP actor and a SEPARATE critic.

Only the actor (an nn.Sequential: Linear, Act, Linear, Act, ..., Linear -> logits) is saved, so checkpoints contain
exactly the policy weights shown to Claude. Contract (PLAN.md): checkpoints/<exp_id>/<name>.pt (state_dict, keys
0.weight, 0.bias, 2.weight, ...) + <name>.json metadata incl. greedy eval over >=100 seeded episodes (eval/rollout.py).

Examples:
  # smoke test
  venv/bin/python train/ppo.py --env-id CartPole-v1 --env-kind raw --arch 4 64 64 2 --total-steps 50000 \
      --exp-id smoke --name cartpole
  # exp1 Atari RAM
  venv/bin/python train/ppo.py --env-id Breakout --env-kind atari_ram --arch 128 64 64 18 --total-steps 10000000 \
      --exp-id exp1 --name breakout_s0 --seed 0
  # exp5 with intermediate checkpoints (saved as <name>_100k, <name>_1M; final as --final-name)
  venv/bin/python train/ppo.py --env-id CartPole-v1 --env-kind raw --arch 4 64 64 2 --total-steps 10000000 \
      --exp-id exp5 --name cartpole --ckpt-steps 100000 1000000 --final-name cartpole_10M
  # exp1b padded + shaped MountainCar
  venv/bin/python train/ppo.py --env-id MountainCar-v0 --env-kind pad8x4 --shaping mc_energy --arch 8 32 32 4 ...

Hyperparameter presets are chosen from the env (atari / lunar / classic); any flag overrides the preset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# macOS Accelerate/vecLib spawns its own GCD worker threads per process (not governed by torch.set_num_threads);
# with ~10 parallel trainings this oversubscribed the machine (load avg ~400). Pin everything to 1 thread.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.wrappers import OBS_PREPROCESS, make_env  # noqa: E402
from eval.rollout import build_mlp, evaluate  # noqa: E402

PRESETS = {
    "atari": dict(num_envs=8, num_steps=128, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, num_minibatches=4,
                  update_epochs=4, clip_coef=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, critic_hidden=256),
    "lunar": dict(num_envs=16, num_steps=256, lr=3e-4, gamma=0.999, gae_lambda=0.98, num_minibatches=8,
                  update_epochs=4, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, critic_hidden=64),
    "classic": dict(num_envs=8, num_steps=256, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, num_minibatches=4,
                    update_epochs=4, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, critic_hidden=64),
}


def step_tag(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


class VecEnv:
    """Minimal synchronous vector env with same-step autoreset; keeps the true (unshaped) reward."""

    def __init__(self, fns, seed):
        self.envs = [f() for f in fns]
        self.n = len(self.envs)
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.single_obs_dim = self.obs.shape[1]
        self.n_actions = self.envs[0].action_space.n

    def step(self, actions):
        obs = np.empty_like(self.obs)
        rew = np.zeros(self.n, np.float32)
        true_rew = np.zeros(self.n, np.float32)
        term = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        final_obs = {}
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr, info = e.step(int(a))
            rew[i], term[i], trunc[i] = r, te, tr
            true_rew[i] = info.get("true_reward", r)
            if te or tr:
                final_obs[i] = o
                o, _ = e.reset()
            obs[i] = o
        self.obs = obs
        return obs, rew, true_rew, term, trunc, final_obs


def init_layers(seq: nn.Sequential, last_std: float):
    lins = [m for m in seq if isinstance(m, nn.Linear)]
    for i, m in enumerate(lins):
        nn.init.orthogonal_(m.weight, last_std if i == len(lins) - 1 else np.sqrt(2))
        nn.init.zeros_(m.bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--env-kind", default="raw", choices=["raw", "atari_ram", "pad8x4", "pad10x5"])
    p.add_argument("--shaping", default=None, help="training-only reward shaping: mc_energy | mc_rightward | ll_hover")
    p.add_argument("--arch", type=int, nargs="+", required=True, help="actor layer sizes in..out")
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--final-name", default=None)
    p.add_argument("--ckpt-steps", type=int, nargs="*", default=[])
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--no-sampled-eval", action="store_true")
    p.add_argument("--preset", default=None, choices=list(PRESETS))
    for k, v in PRESETS["classic"].items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--log-every", type=int, default=20, help="iterations")
    p.add_argument("--obs-shift", type=float, nargs="*", default=None,
                   help="fixed affine obs normalisation used during training: x'=(x-shift)*scale; folded into the first "
                        "actor Linear at save time so the checkpoint consumes the documented raw/padded obs")
    p.add_argument("--obs-scale", type=float, nargs="*", default=None)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    preset = args.preset or ("atari" if args.env_kind == "atari_ram" else
                             "lunar" if args.env_id.startswith("LunarLander") else "classic")
    hp = dict(PRESETS[preset])
    for k in hp:
        v = getattr(args, k)
        if v is not None:
            hp[k] = v
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.join(ROOT, "checkpoints", args.exp_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    curve_path = os.path.join(ROOT, "runs", f"{args.exp_id}_{args.final_name or args.name}_curve.csv")
    curve_f = open(curve_path, "w")
    curve_f.write("global_step,sps,mean_return_last50,episodes\n")

    venv = VecEnv([lambda: make_env(args.env_id, args.env_kind, args.shaping, gamma=hp["gamma"])] * hp["num_envs"],
                  seed=args.seed * 1000)
    obs_dim, n_act = venv.single_obs_dim, venv.n_actions
    assert args.arch[0] == obs_dim and args.arch[-1] == n_act, (args.arch, obs_dim, n_act)

    shift = torch.zeros(obs_dim)
    scale = torch.ones(obs_dim)
    if args.obs_shift:
        shift[: len(args.obs_shift)] = torch.tensor(args.obs_shift)
    if args.obs_scale:
        scale[: len(args.obs_scale)] = torch.tensor(args.obs_scale)

    def norm(o):
        return (o - shift) * scale

    actor = build_mlp(args.arch, args.activation)
    init_layers(actor, 0.01)
    ch = hp["critic_hidden"]
    critic = nn.Sequential(nn.Linear(obs_dim, ch), nn.Tanh(), nn.Linear(ch, ch), nn.Tanh(), nn.Linear(ch, 1))
    init_layers(critic, 1.0)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=hp["lr"], eps=1e-5)

    N, T = hp["num_envs"], hp["num_steps"]
    batch = N * T
    mb = batch // hp["num_minibatches"]
    n_iters = max(1, args.total_steps // batch)
    obs_buf = torch.zeros(T, N, obs_dim)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    logp_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)

    ep_ret = np.zeros(N)
    recent = deque(maxlen=50)
    n_eps = 0
    global_step = 0
    pending = sorted(s for s in args.ckpt_steps if s < args.total_steps)
    t0 = time.time()
    meta_base = {
        "env_id": args.env_id, "env_kind": args.env_kind, "arch": args.arch, "activation": args.activation,
        "obs_preprocess": OBS_PREPROCESS[args.env_kind], "algo": "ppo", "seed": args.seed,
        "shaping_train_only": args.shaping, "hyperparams": hp, "notes": args.notes,
        "train_obs_norm_folded": ({"shift": shift.tolist(), "scale": scale.tolist()}
                                  if (args.obs_shift or args.obs_scale) else None),
    }

    def save(name, steps):
        sd = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        # fold fixed obs normalisation into layer 0: W(x-s)*c + b = (W*c)x + (b - (W*c)s)
        w = sd["0.weight"] * scale[None, :]
        sd["0.bias"] = sd["0.bias"] - w @ shift
        sd["0.weight"] = w
        torch.save(sd, os.path.join(out_dir, name + ".pt"))
        te = time.time()
        g = evaluate(sd, meta_base, args.eval_episodes, args.eval_seed, greedy=True)
        meta = dict(meta_base, train_steps=int(steps), eval_mean_return=g["mean_return"],
                    eval_std_return=g["std_return"], eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
                    eval_greedy=g, wallclock_train_s=round(te - t0, 1))
        if not args.no_sampled_eval:
            s = evaluate(sd, meta_base, args.eval_episodes, args.eval_seed, greedy=False)
            meta.update(eval_sampled_mean_return=s["mean_return"], eval_sampled_std_return=s["std_return"],
                        eval_sampled=s)
        meta["train_recent_mean_return"] = float(np.mean(recent)) if recent else None
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[ckpt] {name} steps={steps} greedy={g['mean_return']:.2f}±{g['std_return']:.2f} "
              f"outcomes={g['outcomes']} sampled={meta.get('eval_sampled_mean_return')} (eval {time.time()-te:.0f}s)",
              flush=True)

    next_obs = torch.as_tensor(venv.obs, dtype=torch.float32)
    for it in range(1, n_iters + 1):
        if not args.no_anneal_lr:
            opt.param_groups[0]["lr"] = hp["lr"] * (1.0 - (it - 1) / n_iters)
        for t in range(T):
            global_step += N
            obs_buf[t] = next_obs
            with torch.no_grad():
                logits = actor(norm(next_obs))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp_buf[t] = dist.log_prob(a)
                val_buf[t] = critic(norm(next_obs)).squeeze(-1)
            act_buf[t] = a
            o, r, tr_r, term, trunc, final_obs = venv.step(a.numpy())
            if args.env_kind == "atari_ram":
                r = np.sign(r)
            r = r.astype(np.float32)
            # bootstrap truncated episodes with V(final_obs)
            for i, fo in final_obs.items():
                if trunc[i] and not term[i]:
                    with torch.no_grad():
                        r[i] += hp["gamma"] * critic(norm(torch.as_tensor(fo, dtype=torch.float32))).item()
            rew_buf[t] = torch.as_tensor(r)
            done_buf[t] = torch.as_tensor((term | trunc).astype(np.float32))
            ep_ret += tr_r
            for i in np.nonzero(term | trunc)[0]:
                recent.append(ep_ret[i])
                ep_ret[i] = 0
                n_eps += 1
            next_obs = torch.as_tensor(o, dtype=torch.float32)

        with torch.no_grad():
            next_val = critic(norm(next_obs)).squeeze(-1)
            adv = torch.zeros(T, N)
            last = torch.zeros(N)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + hp["gamma"] * nv * nonterm - val_buf[t]
                last = delta + hp["gamma"] * hp["gae_lambda"] * nonterm * last
                adv[t] = last
            ret = adv + val_buf

        b_obs, b_act, b_logp = norm(obs_buf.reshape(-1, obs_dim)), act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        stop = False
        for _ in range(hp["update_epochs"]):
            perm = torch.randperm(batch)
            for s in range(0, batch, mb):
                idx = perm[s:s + mb]
                logits = actor(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                newlogp = dist.log_prob(b_act[idx])
                ratio = (newlogp - b_logp[idx]).exp()
                madv = b_adv[idx]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - hp["clip_coef"], 1 + hp["clip_coef"])).mean()
                v = critic(b_obs[idx]).squeeze(-1)
                vloss = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg - hp["ent_coef"] * ent + hp["vf_coef"] * vloss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (newlogp - b_logp[idx])).mean().item()
            if args.target_kl is not None and kl > args.target_kl:
                break

        sps = global_step / (time.time() - t0)
        mr = float(np.mean(recent)) if recent else float("nan")
        if it % args.log_every == 0 or it == n_iters:
            print(f"step={global_step} it={it}/{n_iters} sps={sps:.0f} ret50={mr:.2f} eps={n_eps} "
                  f"ent={ent.item():.3f} kl={kl:.4f} vloss={vloss.item():.3f} eta_min={(args.total_steps-global_step)/max(sps,1)/60:.1f}",
                  flush=True)
            curve_f.write(f"{global_step},{sps:.0f},{mr:.3f},{n_eps}\n")
            curve_f.flush()
        while pending and global_step >= pending[0]:
            save(f"{args.name}_{step_tag(pending.pop(0))}", global_step)

    save(args.final_name or args.name, global_step)
    curve_f.close()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
