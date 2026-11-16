"""Compact an eval items file (full texts) into per-sample rows without texts, plus loop markers (CPU only).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python exp8/scripts/compact_items.py exp8/results/<label>.items.jsonl.gz
    -> exp8/results/<label>.compact.jsonl.gz  (bench, id, sample, correct, hit_limit, n_tokens, finished_thinking,
       has_answer, pred, loop markers, and for IFEval the strict/loose lists)

Loop markers (common.loop_markers on the last 20,000 characters): longest_repeat_frac = the longest word span that
occurs twice in the tail, as a fraction of the tail's words; tail200_count = how often the final 200 characters occur
in the tail. "Looping" (held-out loop rate) = tail200_count >= 3 or longest_repeat_frac >= 0.25.
"""
from __future__ import annotations

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exp8 import common as C  # noqa: E402

KEEP = ("bench", "id", "sample", "seed", "gold", "correct", "lenient_correct", "hit_limit", "n_tokens", "has_answer",
        "pred", "finish_reason", "strict_list", "loose_list", "loose_all")


def one(line: str) -> dict:
    r = json.loads(line)
    t = r.get("text") or ""
    out = {k: r.get(k) for k in KEEP if k in r}
    out["finished_thinking"] = "</think>" in t
    lm = C.loop_markers(t)
    out["loop"] = {k: lm[k] for k in ("longest_repeat_words", "longest_repeat_frac", "tail200_count",
                                      "tail200_period_chars", "dup_8gram_frac")}
    out["looping"] = bool((lm["tail200_count"] or 0) >= 3 or lm["longest_repeat_frac"] >= 0.25)
    return out


def main():
    src = Path(sys.argv[1])
    dst = src.with_name(src.name.replace(".items.jsonl.gz", ".compact.jsonl.gz"))
    with gzip.open(src, "rt") as f:
        lines = f.readlines()
    with ProcessPoolExecutor(32) as ex:
        rows = list(ex.map(one, lines, chunksize=16))
    C.write_jsonl_gz(dst, rows)
    print(dst, len(rows))


if __name__ == "__main__":
    main()
