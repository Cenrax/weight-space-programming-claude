"""Fast batched numpy 9x9 Go for exp6. See engine.py (rules) and features.py (network I/O layout)."""
from .engine import BLACK, EMPTY, N, NN, PASS, WHITE, GoBatch, from_gtp, to_gtp  # noqa: F401
from .features import FEATURE_DOC, N_IN, N_OUT, encode, encode_board  # noqa: F401
