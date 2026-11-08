"""Batched 9x9 Go engine in pure numpy (exp6).

Rules (validated against pettingzoo classic/go_v5 = minigo's go_base, see envs/go/validate.py):
  * captures; suicide is illegal; simple ko (no superko); pass is always legal
  * game ends after two consecutive passes, or when `max_moves` plies (passes count) have been played
  * area (Tromp-Taylor) scoring: stones + empty regions that touch only one colour; komi (default 7.0) to White
  * no resignation; a draw is possible because komi is an integer

Board layout (also used by the network features, see envs/go/features.py):
  point index p = row * 9 + col, row 0 = top row, col 0 = left column; p = 81 is "pass" in action space.
  board[b, p] in {0 empty, 1 black, 2 white}; column 81 of the internal board is an off-board sentinel (3).

All state is kept as arrays over a batch of B independent games, so one `step(actions)` advances every game.
After each step the engine re-derives chains (connected components), their liberty counts and the legal-move
mask for the side to move, so `legal_mask()` is free.
"""
from __future__ import annotations

import numpy as np

N = 9
NN = N * N            # 81 board points
PASS = NN             # action 81 = pass
EDGE = NN             # sentinel column in the internal (B, 82) arrays
EMPTY, BLACK, WHITE, OFF = 0, 1, 2, 3
BIG = 10_000
COLS = "ABCDEFGHJ"    # GTP column letters (no I)


def _neighbors() -> np.ndarray:
    nb = np.full((NN, 4), EDGE, np.int64)
    for p in range(NN):
        r, c = divmod(p, N)
        for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N:
                nb[p, k] = rr * N + cc
    return nb


NBR = _neighbors()                       # (81, 4) neighbour point index, EDGE if off board
ON_BOARD_NBR = (NBR != EDGE)             # (81, 4)


def to_gtp(p: int) -> str:
    if p == PASS:
        return "pass"
    r, c = divmod(int(p), N)
    return f"{COLS[c]}{N - r}"


def from_gtp(s: str) -> int:
    s = s.strip().upper()
    if s == "PASS":
        return PASS
    return (N - int(s[1:])) * N + COLS.index(s[0])


def any4(x: np.ndarray) -> np.ndarray:
    """any() over the last axis (length 4) of a bool array, via a uint32 view (much faster than .any(-1))."""
    return np.ascontiguousarray(x).view(np.uint32)[..., 0] != 0


def components(board: np.ndarray) -> np.ndarray:
    """Connected components of equal colour (stones of one colour, or empty regions).

    board: (B, 82) int8 with column 81 = OFF. Returns labels (B, 82) int64 where each point's label is the
    smallest point index in its component (the sentinel gets BIG). Min-label propagation + pointer jumping.
    """
    B = board.shape[0]
    nbc = board[:, NBR]                                   # (B, 81, 4) neighbour colours
    same = nbc == board[:, :NN, None]                     # OFF never equals a real colour
    base = (np.arange(B) * (NN + 1))[:, None, None]
    flat_nb = (base + NBR[None]).reshape(B, -1)           # flat gather index into (B*82)
    lab = np.broadcast_to(np.arange(NN + 1), (B, NN + 1)).copy()
    lab[:, EDGE] = BIG
    rows = (np.arange(B) * (NN + 1))[:, None]
    while True:
        nl = lab.reshape(-1)[flat_nb].reshape(B, NN, 4)
        nl = np.where(same, nl, BIG).min(-1)
        new = np.minimum(lab[:, :NN], nl)
        # pointer jumping: a label is a point in the same component, whose own label is <= it
        new = lab.reshape(-1)[rows + new]  # lab values are < BIG for real points
        new = np.minimum(new, np.minimum(lab[:, :NN], nl))
        if np.array_equal(new, lab[:, :NN]):
            return lab
        lab[:, :NN] = new


class GoBatch:
    """B independent 9x9 games advanced in lockstep. Black moves first.

    Chain labels are maintained incrementally: a placed stone merges its own-colour neighbour chains into one
    chain labelled by the new stone's point; captured chains are removed and their points relabelled to
    themselves. Chains never split otherwise, so the labels are always exact (validate.py checks them against a
    flood fill)."""

    def __init__(self, n_games: int, komi: float = 7.0, max_moves: int = 200, record: bool = True):
        self.B = int(n_games)
        self.komi = float(komi)
        self.max_moves = int(max_moves)
        self.record = record
        B = self.B
        self._rows82 = (np.arange(B, dtype=np.int64) * (NN + 1))[:, None]
        self._flat_nb = (self._rows82[:, :, None] + NBR[None]).reshape(B, NN * 4)   # (B, 324) flat gather index
        self._ar = np.arange(B)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> "GoBatch":
        B = self.B
        self.board = np.zeros((B, NN + 1), np.int8)
        self.board[:, EDGE] = OFF
        self.labels = np.broadcast_to(np.arange(NN + 1, dtype=np.int64), (B, NN + 1)).copy()
        self.to_play = np.full(B, BLACK, np.int8)
        self.ko = np.full(B, -1, np.int64)
        self.passes = np.zeros(B, np.int64)          # consecutive passes
        self.n_moves = np.zeros(B, np.int64)
        self.done = np.zeros(B, bool)
        self.end_reason = np.zeros(B, np.int8)       # 0 running, 1 two passes, 2 move cap
        self.captured = np.zeros((B, 3), np.int64)   # stones captured BY colour 1 / 2
        self.moves = np.full((B, self.max_moves), -1, np.int16) if self.record else None
        self._analyze()
        return self

    def copy(self) -> "GoBatch":
        g = object.__new__(GoBatch)
        for k, v in self.__dict__.items():
            setattr(g, k, v.copy() if isinstance(v, np.ndarray) else v)
        return g

    def _gather_nb(self, arr82: np.ndarray) -> np.ndarray:
        return arr82.reshape(-1).take(self._flat_nb).reshape(self.B, NN, 4)

    def _analyze(self) -> None:
        """Recompute liberty counts and the legal mask for the side to move from board + chain labels."""
        B = self.B
        board = self.board
        lab = self.labels
        empty = board[:, :NN] == EMPTY
        nbc = self._gather_nb(board)                                 # (B, 81, 4)
        nlab = self._gather_nb(lab)
        nstone = (nbc == BLACK) | (nbc == WHITE)
        valid = nstone & empty[:, :, None]
        # count each (empty point, adjacent chain) pair once
        v0, v1, v2, v3 = (valid[:, :, k] for k in range(4))
        l0, l1, l2, l3 = (nlab[:, :, k] for k in range(4))
        v1 &= l1 != l0
        v2 &= (l2 != l0) & (l2 != l1)
        v3 &= (l3 != l0) & (l3 != l1) & (l3 != l2)
        flat = (self._rows82[:, :, None] + nlab).reshape(-1)
        libs = np.bincount(flat[valid.reshape(-1)], minlength=B * (NN + 1))
        pl = libs.take(self._rows82 + lab).astype(np.int64)         # (B, 82) liberties of chain at each point
        pl[:, :NN] *= ~empty
        pl[:, EDGE] = 0
        self.point_libs = pl
        self.nb_color = nbc
        self.nb_libs = self._gather_nb(pl)
        self.nb_lab = nlab
        c = self.to_play[:, None, None]
        own = nbc == c
        opp = nbc == (3 - c)
        legal = empty & (any4(nbc == EMPTY) | any4(own & (self.nb_libs >= 2)) | any4(opp & (self.nb_libs == 1)))
        has_ko = self.ko >= 0
        legal[np.nonzero(has_ko)[0], self.ko[has_ko]] = False
        mask = np.empty((B, NN + 1), bool)
        mask[:, :NN] = legal
        mask[:, PASS] = True
        if self.done.any():
            mask[self.done] = False
            mask[self.done, PASS] = True
        self._legal = mask

    @property
    def chain_size(self) -> np.ndarray:
        """(B, 82) number of stones in the chain labelled by each index (only meaningful at chain labels)."""
        stone = self.board[:, :NN] != EMPTY
        return np.bincount((self._rows82 + self.labels[:, :NN]).reshape(-1), weights=stone.reshape(-1),
                           minlength=self.B * (NN + 1)).reshape(self.B, NN + 1).astype(np.int64)

    def legal_mask(self) -> np.ndarray:
        """(B, 82) bool: legal actions for the side to move (index 81 = pass, always legal)."""
        return self._legal

    # ------------------------------------------------------------------ moves
    def step(self, actions: np.ndarray) -> None:
        """Play one action per game (ignored for finished games). Raises on an illegal move."""
        a = np.asarray(actions, np.int64).copy()
        a[self.done] = PASS
        act = ~self.done
        ok = self._legal[self._ar, a]
        if not ok.all():
            bad = np.nonzero(~ok)[0]
            raise ValueError(f"illegal move(s) in games {bad[:5].tolist()}: {a[bad[:5]].tolist()}")
        if self.record:
            ix = np.nonzero(act)[0]
            self.moves[ix, self.n_moves[ix]] = a[ix]
        play = act & (a != PASS)
        ip = np.nonzero(play)[0]
        new_ko = np.full(self.B, -1, np.int64)
        if ip.size:
            p = a[ip]
            k = ip.size
            c = self.to_play[ip]
            oc = (3 - c).astype(np.int8)
            ncol = self.nb_color[ip, p]                      # (k, 4)
            nlib = self.nb_libs[ip, p]
            nlab = self.nb_lab[ip, p]
            lab = self.labels[ip]                            # (k, 82) copy
            cap_lab = np.where((ncol == oc[:, None]) & (nlib == 1), nlab, -2)   # chains captured by this move
            own_lab = np.where(ncol == c[:, None], nlab, -2)                     # chains merged by this move
            capt = (lab[:, :NN, None] == cap_lab[:, None, :]).any(-1)            # (k, 81)
            merge = (lab[:, :NN, None] == own_lab[:, None, :]).any(-1)
            ncap = capt.sum(1)
            # minigo's rule: ko if exactly one stone captured and every on-board neighbour was an enemy stone
            koish = ((ncol == oc[:, None]) | (ncol == OFF)).all(1)
            is_ko = (ncap == 1) & koish
            new_ko[ip[is_ko]] = capt[is_ko].argmax(1)
            sub = self.board[ip]
            sub[:, :NN][capt] = EMPTY
            sub[np.arange(k), p] = c
            self.board[ip] = sub
            ar81 = np.broadcast_to(np.arange(NN), (k, NN))
            l81 = lab[:, :NN]
            l81[capt] = ar81[capt]
            l81[merge] = np.broadcast_to(p[:, None], (k, NN))[merge]
            l81[np.arange(k), p] = p
            self.labels[ip] = lab
            self.captured[ip, c] += ncap
        self.ko = np.where(act, new_ko, self.ko)
        self.passes = np.where(act & (a == PASS), self.passes + 1, np.where(act, 0, self.passes))
        self.n_moves += act
        self.to_play = np.where(act, 3 - self.to_play, self.to_play).astype(np.int8)
        ended = act & ((self.passes >= 2) | (self.n_moves >= self.max_moves))
        self.end_reason[ended] = np.where(self.passes[ended] >= 2, 1, 2)
        self.done |= ended
        self._analyze()

    # ------------------------------------------------------------------ scoring
    def area(self) -> tuple[np.ndarray, np.ndarray]:
        """Area (Tromp-Taylor) counts (black, white): stones + empty regions touching only that colour."""
        B = self.B
        board = self.board
        lab = components(board)  # flood fill including empty regions
        empty = board[:, :NN] == EMPTY
        nb = board[:, NBR]
        tb = (empty[:, :, None] & (nb == BLACK)).any(-1)
        tw = (empty[:, :, None] & (nb == WHITE)).any(-1)
        flat = (np.arange(B)[:, None] * (NN + 1) + np.minimum(lab[:, :NN], NN)).reshape(-1)
        rb = np.bincount(flat, weights=tb.reshape(-1), minlength=B * (NN + 1)).reshape(B, NN + 1) > 0
        rw = np.bincount(flat, weights=tw.reshape(-1), minlength=B * (NN + 1)).reshape(B, NN + 1) > 0
        reg_b = np.take_along_axis(rb, np.minimum(lab[:, :NN], NN), 1)
        reg_w = np.take_along_axis(rw, np.minimum(lab[:, :NN], NN), 1)
        black = (board[:, :NN] == BLACK).sum(1) + (empty & reg_b & ~reg_w).sum(1)
        white = (board[:, :NN] == WHITE).sum(1) + (empty & reg_w & ~reg_b).sum(1)
        return black, white

    def score(self) -> np.ndarray:
        """Black-perspective margin: area(black) - area(white) - komi."""
        b, w = self.area()
        return b - w - self.komi

    def winner(self) -> np.ndarray:
        """1 = black wins, 2 = white wins, 0 = draw (only possible with integer komi)."""
        s = self.score()
        return np.where(s > 0, BLACK, np.where(s < 0, WHITE, 0)).astype(np.int8)

    # ------------------------------------------------------------------ move analysis (heuristic bot / stats)
    def move_analysis(self) -> dict[str, np.ndarray]:
        """Per board point, for the side to move: stones captured if played there, and the exact number of
        liberties the played stone's chain would have afterwards. `result_libs` is exact for non-capturing moves;
        for capturing moves it is a lower bound (only captured stones adjacent to the played point are counted).
        Also `own_atari`: points belonging to the mover's chains that are in atari. Arrays are (B, 81)."""
        B = self.B
        board = self.board
        lab = self.labels
        empty = board[:, :NN] == EMPTY
        # liberty bitsets per chain: two uint64 words (81 bits)
        word = (np.arange(NN) // 64)
        bit = (np.uint64(1) << (np.arange(NN) % 64).astype(np.uint64))
        libbits = np.zeros((B, NN + 1, 2), np.uint64)
        nlab = self.nb_lab
        nstone = (self.nb_color == BLACK) | (self.nb_color == WHITE)
        bi, pi, ki = np.nonzero(nstone & empty[:, :, None])
        np.bitwise_or.at(libbits, (bi, nlab[bi, pi, ki], word[pi]), bit[pi])
        c = self.to_play[:, None, None]
        own = self.nb_color == c
        opp = self.nb_color == (3 - c)
        # capture size (distinct enemy chains in atari adjacent to p)
        capc = opp & (self.nb_libs == 1)
        dist = capc.copy()
        for k in range(1, 4):
            for j in range(k):
                dist[:, :, k] &= ~(capc[:, :, j] & (nlab[:, :, j] == nlab[:, :, k]))
        sizes = self.chain_size[np.arange(B)[:, None, None], nlab]
        cap_size = np.where(dist, sizes, 0).sum(-1) * empty
        # resulting liberties
        res = np.zeros((B, NN, 2), np.uint64)
        for k in range(4):
            g = np.minimum(nlab[:, :, k], NN)
            lb = libbits[np.arange(B)[:, None], g]                       # (B, 81, 2)
            res |= np.where(own[:, :, k, None], lb, np.uint64(0))
            q = NBR[:, k]
            qb = np.zeros((NN, 2), np.uint64)
            okq = q != EDGE
            qb[np.nonzero(okq)[0], word[q[okq]]] = bit[q[okq]]
            gets = (self.nb_color[:, :, k] == EMPTY) | capc[:, :, k]
            res |= np.where(gets[:, :, None], qb[None], np.uint64(0))
        selfb = np.zeros((NN, 2), np.uint64)
        selfb[np.arange(NN), word] = bit
        res &= ~selfb[None]
        res_libs = np.bitwise_count(res).sum(-1).astype(np.int64) * empty
        own_pts = board[:, :NN] == self.to_play[:, None]
        own_atari = own_pts & (self.point_libs[:, :NN] == 1)
        return {"capture_size": cap_size, "result_libs": res_libs, "own_atari": own_atari}

    # ------------------------------------------------------------------ display
    def ascii(self, b: int) -> str:
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = ["   " + " ".join(str(c) for c in range(N))]
        for r in range(N):
            lines.append(f"{r}  " + " ".join(sym[int(v)] for v in self.board[b, r * N:(r + 1) * N]))
        return "\n".join(lines)
