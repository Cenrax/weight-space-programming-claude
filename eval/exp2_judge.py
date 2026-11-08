"""Exp2 judge: classify a 16x16 image (e.g. drawn by Claude) as circle / square / triangle by template IoU.

Templates: every shape from the exp2 generator (train/ddpm.py:render) over a dense grid of centres (step 0.5px)
and sizes, binarised at 0.5, in two variants: FILLED and OUTLINE (filled minus its 4-neighbour erosion).
Class score = max IoU over that class's templates (both variants). pred = argmax; "empty" if no pixel is on.

Accepted inputs (parse_grid):
  * list/array of 16 rows x 16 numbers (0/1, 0..255 or 0..1 floats)
  * text: 16 lines of 16 chars, on = any of '#1X@*█■o+', off = '.0 -_·'; or 16 lines of 16 whitespace numbers
CLI: venv/bin/python eval/exp2_judge.py grid.txt   (or pipe on stdin)  -> prints JSON
"""
import json, os, sys
from functools import lru_cache
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
from ddpm import render, SHAPES, S  # noqa: E402

ON = set("#1X@*█■o+xO")
OFF = set(".0 -_·")


def parse_grid(obj):
    if isinstance(obj, str):
        lines = [l.rstrip("\n") for l in obj.strip("\n").splitlines() if l.strip()]
        lines = [l.strip().strip("`") for l in lines if not l.strip().startswith("```")]
        rows = []
        for l in lines:
            toks = l.split()
            if len(toks) == S and all(_isnum(t) for t in toks):
                rows.append([float(t) for t in toks])
            else:
                chars = l.replace(" ", "") if len(l.replace(" ", "")) == S else l
                rows.append([1.0 if c in ON else 0.0 for c in chars[:S]])
        arr = np.array(rows, dtype=np.float32)
    else:
        arr = np.asarray(obj, dtype=np.float32)
    if arr.shape != (S, S):
        raise ValueError(f"expected 16x16 grid, got {arr.shape}")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _isnum(t):
    try:
        float(t); return True
    except ValueError:
        return False


def _outline(m):
    p = np.pad(m, 1)
    er = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~er


CACHE = os.path.join(ROOT, "data", "exp2", "judge_templates.npz")


@lru_cache(maxsize=1)
def templates():
    """Built once (~1 min) and cached to data/exp2/judge_templates.npz."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return {sh: (z[sh + "_T"], [(v, *r) for v, r in zip(z[sh + "_v"].tolist(), z[sh + "_p"].tolist())])
                for sh in SHAPES}
    out = {}
    for sh in SHAPES:
        lo, hi = {"circle": (1.5, 7.5), "square": (1.0, 7.5), "triangle": (2.0, 8.0)}[sh]
        T, info = [], []
        for size in np.arange(lo, hi + 1e-6, 0.5):
            for cx in np.arange(1.0, S - 0.99, 0.5):
                for cy in np.arange(1.0, S - 0.99, 0.5):
                    m = render(sh, cx, cy, size, ss=2) >= 0.5
                    if m.sum() < 3:
                        continue
                    T.append(m.ravel()); info.append(("filled", cx, cy, size))
                    o = _outline(m)
                    if o.sum() != m.sum():
                        T.append(o.ravel()); info.append(("outline", cx, cy, size))
        out[sh] = (np.array(T, dtype=bool), info)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **{k: v for sh, (T, info) in out.items() for k, v in
                                  [(sh + "_T", T), (sh + "_v", np.array([i[0] for i in info])),
                                   (sh + "_p", np.array([i[1:] for i in info], dtype=np.float32))]})
    return out


def classify(img, thresh=0.5):
    x = parse_grid(img) if not isinstance(img, np.ndarray) or img.shape != (S, S) else img.astype(np.float32)
    b = (x >= thresh).ravel()
    if b.sum() == 0:
        return {"pred": "empty", "scores": {s: 0.0 for s in SHAPES}, "variant": None}
    scores, best = {}, {}
    for sh, (T, info) in templates().items():
        inter = (T & b).sum(1); union = (T | b).sum(1)
        iou = inter / union
        k = int(iou.argmax()); scores[sh] = float(iou[k]); best[sh] = info[k]
    pred = max(scores, key=scores.get)
    v, cx, cy, size = best[pred]
    return {"pred": pred, "scores": {k: round(v_, 4) for k, v_ in scores.items()}, "variant": v,
            "best_template": {"cx": float(cx), "cy": float(cy), "size": float(size)}, "on_pixels": int(b.sum())}


def score_answer(img, target):
    """Convenience for graders: returns classify() result plus correct flag and target IoU."""
    r = classify(img)
    r.update(target=target, correct=(r["pred"] == target), target_iou=r["scores"].get(target, 0.0))
    return r


if __name__ == "__main__":
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    try:
        obj = json.loads(src)
    except json.JSONDecodeError:
        obj = src
    print(json.dumps(classify(obj), indent=1))
