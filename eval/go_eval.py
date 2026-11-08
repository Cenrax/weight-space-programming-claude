"""exp6 Go evaluation: policies, matches, win-rate tables, Claude's evaluation and the budgeted forward-pass tool.

    # win-rate table of the PPO nets (vs random, heuristic, each other) -> checkpoints/exp6/eval_table.json
    VECLIB_MAXIMUM_THREADS=1 taskpolicy -a venv/bin/python eval/go_eval.py table --games 256
    # evaluate a weights file Claude wrote (npz with W1..W7/b1..b7 as [in,out], or a torch state_dict .pt)
    venv/bin/python eval/go_eval.py claude logs/exp6/<run>/weights/latest.npz --games 128

In Python:
    from eval.go_eval import play_match, load_policy, evaluate_claude, make_forward_pass_tool
    play_match(load_policy("small_s0"), load_policy("random"), n_games=128, seed=0)
    evaluate_claude(sb.weights)                       # 128 games vs each PPO opponent (small_s0, small_s2, large)
    sb.register("play_games", make_forward_pass_tool(sb), budget=N, doc=FORWARD_PASS_DOC)

Protocol: colours alternate (policy A is Black in even games), both sides play greedy argmax over legal moves
(ties broken at random), and each pair of games (2k, 2k+1) starts from the same `n_random_opening` = 2 random
plies (one random stone each) with colours swapped, otherwise two deterministic nets would replay the same two
games 64 times. Komi 7.0, area scoring, 200-ply cap; a draw counts half.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Mapping

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.go.engine import BLACK, EMPTY, NN, OFF, PASS, WHITE, GoBatch, NBR  # noqa: E402
from envs.go.features import encode  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "exp6")
CLAUDE_ARCH = [243] + [512] * 6 + [82]
CLAUDE_ACTIVATION = "relu"
PPO_OPPONENTS = ("small_s0", "small_s2", "large")
EVAL_SEED = 1_000_003            # held-out seed for the final evaluation of Claude's weights
FWD_SEED_BASE = 200_000          # forward-pass tool: call k uses seed FWD_SEED_BASE + 1000 * k
ACTS = {"tanh": nn.Tanh, "relu": nn.ReLU}


# ---------------------------------------------------------------------------------------------- networks
def build_mlp(arch, activation="relu") -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(ACTS[activation]())
    return nn.Sequential(*layers)


def to_state_dict(params: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Accept a torch state_dict (0.weight [out,in], ...) or harness params (W1 [in,out], b1, W2, ...)."""
    if any(k.startswith("W") for k in params):
        n = len([k for k in params if k.startswith("W")])
        sd = {}
        for k in range(1, n + 1):
            sd[f"{2 * (k - 1)}.weight"] = torch.as_tensor(np.asarray(params[f"W{k}"], np.float32)).T.contiguous()
            sd[f"{2 * (k - 1)}.bias"] = torch.as_tensor(np.asarray(params[f"b{k}"], np.float32)).reshape(-1)
        return sd
    return {k: torch.as_tensor(np.asarray(v, np.float32)) for k, v in params.items()}


def net_from_params(params: Mapping[str, Any], activation: str = CLAUDE_ACTIVATION) -> nn.Sequential:
    sd = to_state_dict(params)
    ws = [sd[k] for k in sorted((k for k in sd if k.endswith(".weight")), key=lambda s: int(s.split(".")[0]))]
    arch = [ws[0].shape[1]] + [w.shape[0] for w in ws]
    net = build_mlp(arch, activation)
    net.load_state_dict(sd)
    net.eval()
    return net


# ---------------------------------------------------------------------------------------------- policies
class Policy:
    name = "policy"

    def reset_stats(self) -> None:
        pass

    def act(self, game: GoBatch, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


def _pick(score: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Argmax of `score` over `allowed`, ties broken uniformly at random."""
    s = np.where(allowed, score, -np.inf)
    best = s.max(1, keepdims=True)
    ties = allowed & (s == best)
    return (rng.random(s.shape) * ties).argmax(1)


class RandomPolicy(Policy):
    """Uniform over legal board moves; passes only when no board move is legal."""
    name = "random"

    def act(self, game, idx, rng):
        m = game.legal_mask()[idx].copy()
        m[:, PASS] = False
        m[~m.any(1), PASS] = True
        return _pick(np.zeros(m.shape), m, rng)


class HeuristicPolicy(Policy):
    """1) capture the largest group possible; else 2) a random legal move that is not self-atari (the played
    chain keeps >= 2 liberties) and does not fill an own eye-point (all on-board neighbours own stones);
    else 3) pass (rather than self-atari or filling its own eyes)."""
    name = "heuristic"

    def act(self, game, idx, rng):
        m = game.legal_mask()[idx]
        ma = game.move_analysis()
        cap = ma["capture_size"][idx] * m[:, :NN]
        c = game.to_play[idx][:, None, None]
        nbc = game.nb_color[idx]
        eye = ((nbc == c) | (nbc == OFF)).all(-1)
        safe = m[:, :NN] & (ma["result_libs"][idx] >= 2) & ~eye
        any_board = m[:, :NN]
        out = np.full(len(idx), PASS)
        has_cap = cap.max(1) > 0
        has_safe = safe.any(1)
        has_any = any_board.any(1)
        z = np.zeros((len(idx), NN))
        if has_cap.any():
            out[has_cap] = _pick(cap[has_cap].astype(float), cap[has_cap] > 0, rng)
        s = ~has_cap & has_safe
        if s.any():
            out[s] = _pick(z[s], safe[s], rng)
        return out


class NetPolicy(Policy):
    """243 -> ... -> 82 MLP; masked logits; greedy argmax (default) or softmax sampling at `temperature`."""

    def __init__(self, net: nn.Module, name: str = "net", greedy: bool = True, temperature: float = 1.0):
        self.net = net.eval()
        self.name = name
        self.greedy = greedy
        self.temperature = temperature
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(decisions=0, raw_argmax_illegal=0, nonfinite_logits=0, entropy_sum=0.0,
                          top_prob_sum=0.0)

    @torch.no_grad()
    def logits(self, game, idx):
        x = torch.from_numpy(encode(game, idx))
        return self.net(x).double().numpy()

    def act(self, game, idx, rng):
        m = game.legal_mask()[idx]
        lg = self.logits(game, idx)
        st = self.stats
        bad = ~np.isfinite(lg)
        st["nonfinite_logits"] += int(bad.any(1).sum())
        lg = np.nan_to_num(lg, nan=-1e30, posinf=1e30, neginf=-1e30)
        st["decisions"] += len(idx)
        st["raw_argmax_illegal"] += int((~m[np.arange(len(idx)), lg.argmax(1)]).sum())
        z = np.where(m, lg, -np.inf)
        z = z - z.max(1, keepdims=True)
        p = np.where(m, np.exp(z / max(self.temperature, 1e-6) if not self.greedy else z), 0.0)
        p /= p.sum(1, keepdims=True)
        st["entropy_sum"] += float(-(p * np.log(np.where(p > 0, p, 1))).sum())
        st["top_prob_sum"] += float(p.max(1).sum())
        if self.greedy:
            return _pick(lg, m, rng)
        c = p.cumsum(1)
        u = rng.random((len(idx), 1))
        return np.minimum((c < u).sum(1), NN)


def load_policy(name: str, greedy: bool = True, ckpt_dir: str = CKPT_DIR) -> Policy:
    if name == "random":
        return RandomPolicy()
    if name == "heuristic":
        return HeuristicPolicy()
    sd = torch.load(os.path.join(ckpt_dir, f"{name}.pt"), map_location="cpu", weights_only=True)
    with open(os.path.join(ckpt_dir, f"{name}.json")) as f:
        meta = json.load(f)
    net = build_mlp(meta["arch"], meta["activation"])
    net.load_state_dict(sd)
    return NetPolicy(net, name=name, greedy=greedy)


# ---------------------------------------------------------------------------------------------- matches
def _cpt(ma, game, idx, a):
    """Per-move tactical stats for the mover in games idx playing actions a."""
    k = np.arange(len(idx))
    board_move = a != PASS
    ap = np.minimum(a, NN - 1)
    legal = game.legal_mask()[idx, :NN]
    cap = ma["capture_size"][idx] * legal
    rl = ma["result_libs"][idx]
    c = game.to_play[idx]
    nbc = game.nb_color[idx]                                   # (k, 81, 4)
    own_nb_atari = ((nbc == c[:, None, None]) & (game.nb_libs[idx] == 1)).any(-1)
    save = legal & own_nb_atari & (rl >= 2)                    # extends an own chain out of atari
    chosen_cap = np.where(board_move, cap[k, ap], 0)
    chosen_rl = np.where(board_move, rl[k, ap], 99)
    eye = ((nbc[k, ap] == c[:, None]) | (nbc[k, ap] == OFF)).all(-1) & board_move
    return dict(
        moves=len(idx), passes=int((~board_move).sum()),
        capture_available=int((cap.max(1) > 0).sum()), capture_taken=int(((cap.max(1) > 0) & (chosen_cap > 0)).sum()),
        stones_captured=int(chosen_cap.sum()),
        self_atari=int((board_move & (chosen_cap == 0) & (chosen_rl <= 1)).sum()),
        save_available=int((save.any(1)).sum()),
        save_taken=int((save.any(1) & board_move & (save[k, ap] | (chosen_cap > 0))).sum()),
        own_eye_filled=int(eye.sum()),
    )


def play_match(policy_a: Policy, policy_b: Policy, n_games: int = 128, seed: int = 0, n_random_opening: int = 2,
               komi: float = 7.0, max_moves: int = 200, collect_stats: bool = False) -> dict[str, Any]:
    """Play n_games of A vs B. A is Black in even-numbered games. Returns results from A's perspective
    (plus the GoBatch as `_game` and per-game colour as `_a_black`, for building game records)."""
    rng = np.random.default_rng(seed)
    n_games = int(n_games) + (int(n_games) % 2)
    g = GoBatch(n_games, komi=komi, max_moves=max_moves, record=True)
    a_black = (np.arange(n_games) % 2) == 0
    U = rng.random((n_games // 2, max(n_random_opening, 1), NN + 1))
    policy_a.reset_stats()
    policy_b.reset_stats()
    tac = {"a": {}, "b": {}}
    first_pass = {"a": np.full(n_games, -1), "b": np.full(n_games, -1)}
    while not g.done.all():
        live = ~g.done
        acts = np.full(n_games, PASS)
        ply = int(g.n_moves[live].min()) if live.any() else 0
        if ply < n_random_opening:
            m = g.legal_mask().copy()
            m[:, PASS] = False
            ix = np.nonzero(live)[0]
            acts[ix] = (U[ix // 2, ply] * m[ix]).argmax(1)
        else:
            a_turn = live & ((g.to_play == BLACK) == a_black)
            b_turn = live & ~a_turn
            ma = g.move_analysis() if collect_stats else None
            for side, pol, mask in (("a", policy_a, a_turn), ("b", policy_b, b_turn)):
                ix = np.nonzero(mask)[0]
                if ix.size == 0:
                    continue
                acts[ix] = pol.act(g, ix, rng)
                fp = first_pass[side]
                newp = ix[(acts[ix] == PASS) & (fp[ix] < 0)]
                fp[newp] = g.n_moves[newp]
                if collect_stats:
                    for k, v in _cpt(ma, g, ix, acts[ix]).items():
                        tac[side][k] = tac[side].get(k, 0) + v
        g.step(acts)
    margin_b = g.score()
    margin = np.where(a_black, margin_b, -margin_b)
    win = margin > 0
    draw = margin == 0
    pts = win + 0.5 * draw
    res: dict[str, Any] = {
        "a": policy_a.name, "b": policy_b.name, "n_games": n_games, "seed": seed,
        "wins": int(win.sum()), "losses": int((margin < 0).sum()), "draws": int(draw.sum()),
        "win_rate": float(pts.mean()),
        "win_rate_as_black": float(pts[a_black].mean()), "win_rate_as_white": float(pts[~a_black].mean()),
        "mean_margin": float(margin.mean()), "median_margin": float(np.median(margin)),
        "mean_game_length": float(g.n_moves.mean()),
        "ended_by_two_passes": int((g.end_reason == 1).sum()), "ended_by_move_cap": int((g.end_reason == 2).sum()),
    }
    if collect_stats:
        res["_tactics"] = tac
        res["_first_pass"] = first_pass
    res["_game"] = g
    res["_a_black"] = a_black
    res["_margin"] = margin
    return res


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def game_record(g: GoBatch, i: int, a_black: bool, margin: float, you: str = "you") -> dict[str, Any]:
    n = int(g.n_moves[i])
    mv = g.moves[i, :n]
    seq = " ".join(f"{'B' if t % 2 == 0 else 'W'}{'pass' if m == PASS else int(m)}" for t, m in enumerate(mv))
    res = ("you win" if margin > 0 else "you lose" if margin < 0 else "draw") + f" by {abs(margin):g}"
    b_area, w_area = (int(x) for x in (v[i] for v in g.area()))
    return {"game": int(i), "you_are": "Black (X)" if a_black else "White (O)", "result": res,
            "area_black": b_area, "area_white": w_area, "komi": g.komi, "length": n,
            "end": "two passes" if g.end_reason[i] == 1 else "200-move cap",
            "moves": seq, "note": "the first 2 plies are the shared random opening", "final_board": g.ascii(i)}


# ---------------------------------------------------------------------------------------------- tables
def eval_table(names=PPO_OPPONENTS, n_games: int = 256, seed: int = 7, ckpt_dir: str = CKPT_DIR) -> dict:
    pols = {n: load_policy(n, ckpt_dir=ckpt_dir) for n in names}
    base = {"random": RandomPolicy(), "heuristic": HeuristicPolicy()}
    table: dict[str, dict[str, Any]] = {}
    for n, p in pols.items():
        row = {}
        for o, q in {**base, **{k: v for k, v in pols.items() if k != n}}.items():
            r = public(play_match(p, q, n_games=n_games, seed=seed))
            row[o] = {k: r[k] for k in ("win_rate", "wins", "losses", "draws", "win_rate_as_black",
                                         "win_rate_as_white", "mean_margin", "mean_game_length")}
        table[n] = row
    hr = public(play_match(base["heuristic"], base["random"], n_games=n_games, seed=seed))
    table["heuristic"] = {"random": {k: hr[k] for k in ("win_rate", "wins", "losses", "draws", "mean_margin",
                                                        "mean_game_length")}}
    return {"n_games": n_games, "seed": seed, "protocol": "greedy argmax, colours alternate, 2 shared random "
            "opening plies per pair, komi 7.0, area scoring, 200-ply cap, draw = 0.5", "table": table}


def zero_net_reference(n_games: int = 128, seed: int = 7, opponents=PPO_OPPONENTS + ("random", "heuristic"),
                       ckpt_dir: str = CKPT_DIR) -> dict:
    """All-zero 243->512x6->82 net (the starting point Claude gets): uniform logits -> random legal move,
    INCLUDING pass (unlike RandomPolicy)."""
    params = {}
    for k in range(1, len(CLAUDE_ARCH)):
        params[f"W{k}"] = np.zeros((CLAUDE_ARCH[k - 1], CLAUDE_ARCH[k]), np.float32)
        params[f"b{k}"] = np.zeros(CLAUDE_ARCH[k], np.float32)
    return evaluate_claude(params, opponents=opponents, n_games=n_games, seed=seed, ckpt_dir=ckpt_dir)


# ---------------------------------------------------------------------------------------------- Claude
def evaluate_claude(params: Mapping[str, Any], opponents=PPO_OPPONENTS, n_games: int = 128, seed: int = EVAL_SEED,
                    ckpt_dir: str = CKPT_DIR) -> dict[str, Any]:
    """Final score of a 243->512x6->82 ReLU net (params as harness W1..W7 [in,out] or a torch state_dict)."""
    net = net_from_params(params, CLAUDE_ACTIVATION)
    arch = [net[0].in_features] + [m.out_features for m in net if isinstance(m, nn.Linear)]
    if arch != CLAUDE_ARCH:
        raise ValueError(f"arch {arch} != {CLAUDE_ARCH}")
    me = NetPolicy(net, name="claude")
    out: dict[str, Any] = {"arch": arch, "activation": CLAUDE_ACTIVATION, "n_games": n_games, "seed": seed}
    for o in opponents:
        out[o] = public(play_match(me, load_policy(o, ckpt_dir=ckpt_dir), n_games=n_games, seed=seed))
    ppo = [o for o in opponents if o in PPO_OPPONENTS]
    out["mean_win_rate_vs_ppo"] = float(np.mean([out[o]["win_rate"] for o in ppo])) if ppo else None
    return out


FORWARD_PASS_DOC = (
    "play_games(opponent='small_s0', n_records=3) -> dict. Plays ONE batch of 128 games of your currently SAVED "
    "weights (call save_weights first) against `opponent` (one of: {opps}). Colours alternate (you are Black in "
    "even games); each pair of games shares 2 random opening plies; both sides then play the highest-scoring legal "
    "move (greedy, ties random). Returns win/loss/draw counts, win_rate (draw = 0.5), win rate as Black and as "
    "White, score margins (your area minus opponent's, komi included), game lengths and how games ended, "
    "per-move statistics for you and the opponent (passes, captures available/taken, self-ataris, ataris "
    "escaped, own eyes filled; for you also how often your raw argmax pointed at an illegal point, the mean "
    "softmax entropy and top probability over legal moves, non-finite logits), and `records`: n_records (<= 8) "
    "full games (your worst losses first) as move lists like 'B40 W30 Bpass' (B/W + point index p = row*9+col, "
    "pass) with the final board (X = Black, O = White). Each call uses fresh opening seeds."
)


def make_forward_pass_tool(sb, allowed_opponents=("small_s0", "small_s2", "large", "random", "heuristic"),
                           n_games: int = 128, seed_base: int = FWD_SEED_BASE, ckpt_dir: str = CKPT_DIR
                           ) -> Callable[..., dict]:
    """Return play_games(opponent, n_records) for Sandbox.register(..., budget=N). Reads Claude's weights from
    `sb.weights` (the saved W1..W7/b1..b7, [in,out]) at call time. Opponent weights never leave this process."""
    opp_cache: dict[str, Policy] = {}
    counter = {"calls": 0}

    def play_games(opponent: str = "small_s0", n_records: int = 3) -> dict:
        if opponent not in allowed_opponents:
            raise ValueError(f"unknown opponent {opponent!r}; choose one of {list(allowed_opponents)}")
        n_records = int(max(0, min(8, n_records)))
        if opponent not in opp_cache:
            opp_cache[opponent] = load_policy(opponent, ckpt_dir=ckpt_dir)
        net = net_from_params(sb.weights, CLAUDE_ACTIVATION)
        me = NetPolicy(net, name="you")
        seed = seed_base + 1000 * counter["calls"]
        counter["calls"] += 1
        r = play_match(me, opp_cache[opponent], n_games=n_games, seed=seed, collect_stats=True)
        out = public(r)
        out.pop("a")
        out["b"] = opponent
        out = {"opponent": opponent, **{k: v for k, v in out.items() if k not in ("b", "seed")}}
        tac = r["_tactics"]
        st = me.stats
        d = max(st["decisions"], 1)

        def side(t, fp):
            m = max(t.get("moves", 0), 1)
            return {
                "moves": t.get("moves", 0),
                "pass_rate": round(t.get("passes", 0) / m, 4),
                "games_with_a_pass": int((fp >= 0).sum()),
                "mean_ply_of_first_pass": float(fp[fp >= 0].mean()) if (fp >= 0).any() else None,
                "capture_available_turns": t.get("capture_available", 0),
                "capture_taken_when_available": round(t.get("capture_taken", 0) / max(t.get("capture_available", 0), 1), 4),
                "stones_captured_per_game": round(t.get("stones_captured", 0) / n_games, 2),
                "self_atari_rate": round(t.get("self_atari", 0) / m, 4),
                "atari_escape_available_turns": t.get("save_available", 0),
                "atari_escaped_when_available": round(t.get("save_taken", 0) / max(t.get("save_available", 0), 1), 4),
                "own_eye_fill_rate": round(t.get("own_eye_filled", 0) / m, 4),
            }

        out["you"] = side(tac["a"], r["_first_pass"]["a"])
        out["you"].update({
            "raw_argmax_illegal_rate": round(st["raw_argmax_illegal"] / d, 4),
            "mean_entropy_over_legal": round(st["entropy_sum"] / d, 4),
            "mean_top_prob": round(st["top_prob_sum"] / d, 4),
            "decisions_with_nonfinite_logits": st["nonfinite_logits"],
        })
        out["opponent_stats"] = side(tac["b"], r["_first_pass"]["b"])
        g, ab, mg = r["_game"], r["_a_black"], r["_margin"]
        order = np.argsort(mg, kind="stable")              # worst losses first, then (if room) best wins
        picks = list(order[: (n_records + 1) // 2]) + list(order[::-1][: n_records // 2]) if n_records else []
        seen, recs = set(), []
        for i in picks:
            if int(i) not in seen:
                seen.add(int(i))
                recs.append(game_record(g, int(i), bool(ab[i]), float(mg[i])))
        out["records"] = recs
        out["calls_so_far"] = counter["calls"]
        return out

    play_games.__doc__ = FORWARD_PASS_DOC.format(opps=", ".join(allowed_opponents))
    return play_games


# ---------------------------------------------------------------------------------------------- CLI
def _update_ckpt_json(table: dict, ckpt_dir: str) -> None:
    for n, row in table["table"].items():
        p = os.path.join(ckpt_dir, f"{n}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            meta = json.load(f)
        meta["eval_table"] = {"n_games": table["n_games"], "seed": table["seed"], "protocol": table["protocol"],
                              "vs": row}
        meta["eval_mean_return"] = row["random"]["win_rate"]       # contract field: win rate vs random
        meta["eval_episodes"] = table["n_games"]
        with open(p, "w") as f:
            json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--games", type=int, default=256)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--names", nargs="+", default=list(PPO_OPPONENTS))
    t.add_argument("--ckpt-dir", default=CKPT_DIR)
    t.add_argument("--threads", type=int, default=2)
    c = sub.add_parser("claude")
    c.add_argument("weights")
    c.add_argument("--games", type=int, default=128)
    c.add_argument("--seed", type=int, default=EVAL_SEED)
    c.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "table":
        tab = eval_table(args.names, args.games, args.seed, args.ckpt_dir)
        z = zero_net_reference(128, args.seed, tuple(args.names) + ("random", "heuristic"), args.ckpt_dir)
        tab["zero_net_claude_start"] = {o: v["win_rate"] for o, v in z.items() if isinstance(v, dict)}
        with open(os.path.join(args.ckpt_dir, "eval_table.json"), "w") as f:
            json.dump(tab, f, indent=1)
        _update_ckpt_json(tab, args.ckpt_dir)
        print(json.dumps(tab, indent=1))
    else:
        w = args.weights
        params = dict(np.load(w)) if w.endswith(".npz") else torch.load(w, map_location="cpu", weights_only=True)
        print(json.dumps(evaluate_claude(params, n_games=args.games, seed=args.seed), indent=1))


if __name__ == "__main__":
    main()
