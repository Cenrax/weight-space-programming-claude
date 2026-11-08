"""Network input/output layout for exp6 (this exact text goes into Claude's prompt via FEATURE_DOC).

Input x: 243 floats in {0, 1}, three 81-long planes, always from the perspective of the SIDE TO MOVE:
  x[0   + p] = 1 if point p has a stone of the player to move ("own")
  x[81  + p] = 1 if point p has an opponent stone
  x[162 + p] = 1 if point p is empty
  (exactly one of the three is 1 for every point; colour / move number / ko / history are NOT given)
Output: 82 logits; logit[p] for p in 0..80 = play at point p, logit[81] = pass.
Point index p = row * 9 + col, row 0 = top row, col 0 = left column (GTP: col letter ABCDEFGHJ, row number 9 - row).
Illegal moves (occupied points, suicide, simple-ko retake) are masked out before the argmax; pass is always legal.
"""
from __future__ import annotations

import numpy as np

from .engine import BLACK, EMPTY, NN, GoBatch

N_IN = 3 * NN      # 243
N_OUT = NN + 1     # 82

FEATURE_DOC = """\
Input (243 floats, each 0 or 1), from the perspective of the player to move, as three 81-long planes:
  x[p]       = 1 if point p holds one of YOUR stones
  x[81 + p]  = 1 if point p holds an OPPONENT stone
  x[162 + p] = 1 if point p is empty
  Exactly one of the three is 1 at every point. Colour, move number, ko and move history are not given.
Output (82 logits): logit[p] for p = 0..80 means "play at point p"; logit[81] means "pass".
Points: p = row*9 + col, row 0 is the top row, col 0 is the left column (so p=0 is the top-left corner,
p=40 is the centre (tengen), p=80 is the bottom-right corner).
Illegal moves (occupied points, suicide, retaking a ko immediately) are masked out, then the move with the
highest logit is played (ties broken uniformly at random). Pass is always legal. The game ends after two
consecutive passes or 200 total moves; area scoring (stones + surrounded empty points), komi 7.0 for White,
captures and the no-suicide / simple-ko rules as usual. Equal score = draw (counted as half a win).
"""


def encode(game: GoBatch, idx: np.ndarray | None = None) -> np.ndarray:
    """(k, 243) float32 features for games `idx` (default: all), side-to-move perspective."""
    board = game.board[:, :NN] if idx is None else game.board[idx, :NN]
    tp = game.to_play if idx is None else game.to_play[idx]
    own = board == tp[:, None]
    emp = board == EMPTY
    opp = ~own & ~emp
    return np.concatenate([own, opp, emp], axis=1).astype(np.float32)


def encode_board(board81: np.ndarray, to_play: int = BLACK) -> np.ndarray:
    """Features for one raw board (81 ints 0/1/2) with `to_play` (1 black, 2 white) to move."""
    b = np.asarray(board81).reshape(1, NN)
    own = b == to_play
    emp = b == EMPTY
    return np.concatenate([own, ~own & ~emp, emp], axis=1).astype(np.float32)[0]
