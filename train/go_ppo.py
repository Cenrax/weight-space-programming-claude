"""PPO self-play for 9x9 Go (exp6): plain nn.Sequential MLP actor (243 -> ... -> 82, masked logits) + separate critic.

Each iteration plays `--games-per-iter` complete games with the batched numpy engine (envs/go). The learner plays
one colour (alternating by game); the opponent of each block of `--opp-block` games is drawn from:
  random (uniform legal board move)  p = --p-random
  the current learner (frozen copy)  p = --p-latest
  a uniformly chosen past snapshot   otherwise (pool of the last --pool-size snapshots, one every --pool-every iters)
Both sides sample from their softmax over legal moves. Reward: +1 win / -1 loss / 0 draw at game end, from the
learner's perspective (area score, komi 7.0, 200-ply cap); gamma = 1, GAE over the learner's own decisions.

Checkpoint contract (PLAN.md): checkpoints/exp6/<name>.pt (actor state_dict, keys 0.weight, 0.bias, ...) +
<name>.json. Full training state for --resume: runs/exp6_<name>_state.pt. Curve: runs/exp6_<name>_curve.csv.

  # smoke
  venv/bin/python train/go_ppo.py --name smoke --arch 243 32 32 82 --max-games 3000 --exp-id _exp6_smoke
  # the three exp6 nets
  venv/bin/python train/go_ppo.py --name small_s0 --arch 243 32 32 82 --seed 0 --hours 3
  venv/bin/python train/go_ppo.py --name small_s2 --arch 243 32 32 82 --seed 2 --hours 3
  venv/bin/python train/go_ppo.py --name large --arch 243 512 512 512 512 512 512 82 --seed 0 --hours 3 \
      --lr 1e-4 --threads 6 --critic-hidden 512 512
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.go.engine import BLACK, NN, PASS, GoBatch  # noqa: E402
from envs.go.features import encode  # noqa: E402
from eval.go_eval import HeuristicPolicy, NetPolicy, RandomPolicy, build_mlp, play_match, public  # noqa: E402

ENV_ID = "go9x9 (envs/go: komi 7.0, area scoring, simple ko, no suicide, 200-ply cap)"
OBS_PREPROCESS = ("243 = 3 planes x 81 points (p=row*9+col, row 0 top), side-to-move perspective: "
                  "x[p]=own stone, x[81+p]=opponent stone, x[162+p]=empty; output 82 = 81 points + pass (index 81); "
                  "illegal moves masked before argmax/softmax")


def layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def make_actor(arch, activation):
    net = build_mlp(arch, activation)
    lins = [m for m in net if isinstance(m, nn.Linear)]
    for m in lins[:-1]:
        layer_init(m, np.sqrt(2))
    layer_init(lins[-1], 0.01)
    return net


def make_critic(hidden, activation):
    net = build_mlp([243] + list(hidden) + [1], activation)
    lins = [m for m in net if isinstance(m, nn.Linear)]
    for m in lins[:-1]:
        layer_init(m, np.sqrt(2))
    layer_init(lins[-1], 1.0)
    return net


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


@torch.no_grad()
def sample(net, x: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    lg = masked_logits(net(torch.from_numpy(x)), torch.from_numpy(mask))
    logp_all = torch.log_softmax(lg, -1).numpy().astype(np.float64)
    p = np.exp(logp_all) * mask
    p /= p.sum(1, keepdims=True)
    a = np.minimum((p.cumsum(1) < rng.random((len(x), 1))).sum(1), NN)
    a = np.where(mask[np.arange(len(x)), a], a, np.argmax(p, 1))       # guard against float edge cases
    return a, logp_all[np.arange(len(x)), a]


def random_actions(mask: np.ndarray, rng) -> np.ndarray:
    m = mask.copy()
    m[:, PASS] = False
    m[~m.any(1), PASS] = True
    return (rng.random(m.shape) * m).argmax(1)


def rollout(actor, critic, opp_nets, opp_of_game, n_games, rng, max_moves):
    """Play n_games to completion. opp_of_game[g] = index into opp_nets (None entry = random policy)."""
    g = GoBatch(n_games, max_moves=max_moves, record=False)
    learner_black = (np.arange(n_games) % 2) == 0
    waves = []
    passes = decisions = 0
    while not g.done.all():
        live = ~g.done
        mask = g.legal_mask()
        acts = np.full(n_games, PASS)
        lt = live & ((g.to_play == BLACK) == learner_black)
        ix = np.nonzero(lt)[0]
        if ix.size:
            x = encode(g, ix)
            m = mask[ix]
            a, lp = sample(actor, x, m, rng)
            with torch.no_grad():
                v = critic(torch.from_numpy(x)).squeeze(-1).numpy()
            acts[ix] = a
            waves.append((ix, x.astype(np.uint8), m.copy(), a, lp, v))
            passes += int((a == PASS).sum())
            decisions += ix.size
        ot = live & ~lt
        for oi in np.unique(opp_of_game[ot]):
            jx = np.nonzero(ot & (opp_of_game == oi))[0]
            net = opp_nets[oi]
            if net is None:
                acts[jx] = random_actions(mask[jx], rng)
            else:
                acts[jx], _ = sample(net, encode(g, jx), mask[jx], rng)
        g.step(acts)
    margin = g.score()
    margin = np.where(learner_black, margin, -margin)
    R = np.sign(margin)
    return waves, R, margin, g, dict(passes=passes, decisions=decisions)


def gae(waves, R, n_games, gamma, lam):
    next_v = np.zeros(n_games)
    acc = np.zeros(n_games)
    seen = np.zeros(n_games, bool)
    advs = [None] * len(waves)
    for w in range(len(waves) - 1, -1, -1):
        ix, _, _, _, _, v = waves[w]
        r = np.where(~seen[ix], R[ix], 0.0)
        delta = r + gamma * next_v[ix] * seen[ix] - v
        acc[ix] = delta + gamma * lam * acc[ix] * seen[ix]
        advs[w] = acc[ix].copy()
        next_v[ix] = v
        seen[ix] = True
    adv = np.concatenate(advs)
    val = np.concatenate([w[5] for w in waves])
    return adv, adv + val


def save_checkpoint(args, actor, games, steps, iters, wall, extra_eval, path_base):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    torch.save(actor.state_dict(), path_base + ".pt.tmp")
    os.replace(path_base + ".pt.tmp", path_base + ".pt")
    meta = {
        "env_id": ENV_ID, "env_kind": "go9x9", "arch": args.arch, "activation": args.activation,
        "obs_preprocess": OBS_PREPROCESS, "algo": "PPO self-play vs pool of past selves + random (train/go_ppo.py)",
        "train_steps": int(steps), "games_trained": int(games), "iterations": int(iters), "seed": args.seed,
        "eval_mean_return": extra_eval.get("random", {}).get("win_rate"), "eval_std_return": None,
        "eval_episodes": extra_eval.get("n_games"), "eval_metric": "greedy win rate vs random (draw = 0.5)",
        "eval_during_training": extra_eval, "wallclock_s": round(wall, 1),
        "hyperparams": {k: v for k, v in vars(args).items() if k not in ("name",)},
    }
    old = path_base + ".json"
    if os.path.exists(old):
        try:
            with open(old) as f:
                prev = json.load(f)
            if "eval_table" in prev:
                meta["eval_table_stale"] = prev["eval_table"]
        except Exception:
            pass
    with open(path_base + ".json", "w") as f:
        json.dump(meta, f, indent=1)


def quick_eval(actor, n_games, seed):
    me = NetPolicy(copy.deepcopy(actor), name="learner")
    out = {"n_games": n_games}
    for name, opp in (("random", RandomPolicy()), ("heuristic", HeuristicPolicy())):
        r = public(play_match(me, opp, n_games=n_games, seed=seed))
        out[name] = {k: r[k] for k in ("win_rate", "mean_margin", "mean_game_length")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--exp-id", default="exp6")
    ap.add_argument("--arch", type=int, nargs="+", default=[243, 32, 32, 82])
    ap.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--max-games", type=int, default=10**9)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--opp-block", type=int, default=32)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--p-random", type=float, default=0.1)
    ap.add_argument("--p-latest", type=float, default=0.4)
    ap.add_argument("--pool-every", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    assert args.arch[0] == 243 and args.arch[-1] == 82

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tag = args.name if args.exp_id == "exp6" else f"{args.exp_id}_{args.name}"
    ckpt_base = os.path.join(ROOT, "checkpoints", args.exp_id, args.name)
    state_path = os.path.join(ROOT, "runs", f"exp6_{tag}_state.pt")
    curve_path = os.path.join(ROOT, "runs", f"exp6_{tag}_curve.csv")

    actor = make_actor(args.arch, args.activation)
    critic = make_critic(args.critic_hidden, args.activation)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    pool: list[dict] = [copy.deepcopy(actor.state_dict())]
    it = games = steps = 0
    wall0 = 0.0
    last_eval: dict = {}
    if args.resume and os.path.exists(state_path):
        S = torch.load(state_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(S["actor"]); critic.load_state_dict(S["critic"]); opt.load_state_dict(S["opt"])
        pool, it, games, steps, wall0, last_eval = S["pool"], S["iter"], S["games"], S["steps"], S["wall"], S["eval"]
        rng = np.random.default_rng(args.seed + 7919 * it)
        print(f"resumed at iter {it}, games {games}", flush=True)
    fields = ["iter", "games", "steps", "wall_s", "win_rate_all", "win_vs_random", "win_vs_latest", "win_vs_pool",
              "mean_len", "pass_rate", "entropy", "approx_kl", "clipfrac", "v_loss", "eval_random", "eval_heuristic"]
    new_curve = not (args.resume and os.path.exists(curve_path))
    cf = open(curve_path, "w" if new_curve else "a", newline="")
    cw = csv.DictWriter(cf, fieldnames=fields)
    if new_curve:
        cw.writeheader()

    t0 = time.time() - wall0
    last_save = time.time()
    opp_net_cache = [build_mlp(args.arch, args.activation) for _ in range(args.games_per_iter // args.opp_block + 1)]
    print(f"[{args.name}] arch {args.arch} act {args.activation} params "
          f"{sum(p.numel() for p in actor.parameters())} critic {args.critic_hidden} threads {args.threads}", flush=True)
    while games < args.max_games and (time.time() - t0) < args.hours * 3600:
        it += 1
        ti = time.time()
        # ---- opponents for this iteration
        n_blocks = int(np.ceil(args.games_per_iter / args.opp_block))
        opp_nets, kinds = [], []
        for b in range(n_blocks):
            u = rng.random()
            if u < args.p_random:
                opp_nets.append(None); kinds.append("random")
            else:
                sd = actor.state_dict() if u < args.p_random + args.p_latest else pool[rng.integers(len(pool))]
                net = opp_net_cache[b]
                net.load_state_dict(sd)
                opp_nets.append(net); kinds.append("latest" if u < args.p_random + args.p_latest else "pool")
        opp_of_game = np.repeat(np.arange(n_blocks), args.opp_block)[: args.games_per_iter]
        actor.eval(); critic.eval()
        waves, R, margin, g, rs = rollout(actor, critic, opp_nets, opp_of_game, args.games_per_iter, rng,
                                          args.max_moves)
        t_roll = time.time() - ti
        adv, ret = gae(waves, R, args.games_per_iter, args.gamma, args.lam)
        X = torch.from_numpy(np.concatenate([w[1] for w in waves]).astype(np.float32))
        M = torch.from_numpy(np.concatenate([w[2] for w in waves]))
        A = torch.from_numpy(np.concatenate([w[3] for w in waves]).astype(np.int64))
        LP = torch.from_numpy(np.concatenate([w[4] for w in waves]).astype(np.float32))
        ADV = torch.from_numpy(adv.astype(np.float32))
        RET = torch.from_numpy(ret.astype(np.float32))
        T = len(A)
        # ---- PPO update
        actor.train(); critic.train()
        frac = 1.0 - max((time.time() - t0) / (args.hours * 3600), games / args.max_games)
        for pg in opt.param_groups:
            pg["lr"] = args.lr * max(frac, 0.1)
        kls, cfs, vls, ents = [], [], [], []
        stop = False
        n_mb = max(1, int(round(T / args.minibatch)))      # equal-size minibatches (a size-1 tail gave std=NaN)
        skipped = 0
        for ep in range(args.epochs):
            perm = torch.randperm(T)
            for mb in torch.tensor_split(perm, n_mb):
                lg = masked_logits(actor(X[mb]), M[mb])
                logp_all = torch.log_softmax(lg, -1)
                nlp = logp_all.gather(1, A[mb, None]).squeeze(1)
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                ratio = (nlp - LP[mb]).exp()
                a_mb = ADV[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std(correction=0) + 1e-8)
                pl = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v = critic(X[mb]).squeeze(-1)
                vl = 0.5 * ((v - RET[mb]) ** 2).mean()
                loss = pl - args.ent * ent + args.vf * vl
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                if not torch.isfinite(gn):
                    skipped += 1
                    opt.zero_grad()
                    continue
                opt.step()
                with torch.no_grad():
                    lr_ = nlp - LP[mb]
                    kls.append(float(((lr_.exp() - 1) - lr_).mean()))
                    cfs.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                vls.append(float(vl.detach())); ents.append(float(ent.detach()))
            if args.target_kl and kls and np.mean(kls[-n_mb:]) > args.target_kl:
                stop = True
                break
        if skipped:
            print(f"[{args.name}] WARNING: skipped {skipped} non-finite minibatches", flush=True)
        if not all(torch.isfinite(p).all() for p in actor.parameters()):
            raise RuntimeError("actor parameters became non-finite")
        games += args.games_per_iter
        steps += T
        if it % args.pool_every == 0:
            pool.append(copy.deepcopy(actor.state_dict()))
            pool = pool[-args.pool_size:]
        # ---- stats
        win = (R > 0) + 0.5 * (R == 0)
        kinds_g = np.array(kinds)[opp_of_game]
        wr = {k: (float(win[kinds_g == k].mean()) if (kinds_g == k).any() else None) for k in ("random", "latest", "pool")}
        row = {"iter": it, "games": games, "steps": steps, "wall_s": round(time.time() - t0, 1),
               "win_rate_all": round(float(win.mean()), 3), "win_vs_random": wr["random"], "win_vs_latest": wr["latest"],
               "win_vs_pool": wr["pool"], "mean_len": round(float(g.n_moves.mean()), 1),
               "pass_rate": round(rs["passes"] / max(rs["decisions"], 1), 4), "entropy": round(float(np.mean(ents)), 3),
               "approx_kl": round(float(np.mean(kls)), 4), "clipfrac": round(float(np.mean(cfs)), 3),
               "v_loss": round(float(np.mean(vls)), 4), "eval_random": None, "eval_heuristic": None}
        if it % args.eval_every == 0 or it == 1:
            actor.eval()
            last_eval = quick_eval(actor, args.eval_games, seed=10_000 + it)
            row["eval_random"] = last_eval["random"]["win_rate"]
            row["eval_heuristic"] = last_eval["heuristic"]["win_rate"]
        cw.writerow(row); cf.flush()
        print(f"[{args.name}] it {it} games {games} T {T} roll {t_roll:.1f}s upd {time.time() - ti - t_roll:.1f}s "
              f"wr {row['win_rate_all']} (rand {wr['random']}, latest {wr['latest']}, pool {wr['pool']}) "
              f"len {row['mean_len']} pass {row['pass_rate']} ent {row['entropy']} kl {row['approx_kl']}"
              f"{' (kl stop)' if stop else ''} vl {row['v_loss']}"
              + (f" | EVAL rand {row['eval_random']} heur {row['eval_heuristic']}" if row['eval_random'] is not None else ""),
              flush=True)
        if time.time() - last_save > args.save_every_min * 60:
            save_checkpoint(args, actor, games, steps, it, time.time() - t0, last_eval, ckpt_base)
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "iter": it, "games": games, "steps": steps, "wall": time.time() - t0,
                        "eval": last_eval}, state_path)
            last_save = time.time()
    actor.eval()
    last_eval = quick_eval(actor, 128, seed=99_991)
    save_checkpoint(args, actor, games, steps, it, time.time() - t0, last_eval, ckpt_base)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(), "pool": pool,
                "iter": it, "games": games, "steps": steps, "wall": time.time() - t0, "eval": last_eval}, state_path)
    print(f"[{args.name}] DONE games {games} steps {steps} wall {time.time() - t0:.0f}s final eval {last_eval}", flush=True)


if __name__ == "__main__":
    main()
