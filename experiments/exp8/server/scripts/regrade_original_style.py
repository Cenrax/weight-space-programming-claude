"""Regrade an items file with the ORIGINAL post's rules (RikiyaT/lm_understand-logs exp8_math/code/evaluate.py):
strict = math-verify on the last \\boxed{} only; lenient = strict, or (only when there is no \\boxed{}) the latest
'Final Answer' / 'the answer is' / 'Answer:' phrase with a number or formula, else the last number.
The blog's Gemma MATH-500 number (6.8%) is this lenient score; its strict score was 2.75%.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python exp8/scripts/regrade_original_style.py exp8/results/gemma_base.items.jsonl.gz
"""
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exp8.bench import grading as G  # noqa: E402


def lenient_pred(text):
    found = []
    for pat in (r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)", r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
                r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)"):
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def ok(pred, gold):
    from math_verify import parse, verify
    try:
        g, p = parse("$" + gold + "$"), parse("$" + pred + "$")
        return bool(g and p and verify(g, p))
    except Exception:  # noqa: BLE001
        return False


def main(path):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    strict = lenient = boxed = 0
    for r in rows:
        text = r["text"].split(G.THINK_END)[-1] if G.THINK_END in r["text"] else r["text"]   # Gemma: no thinking
        b = G.last_boxed(text)
        if b is not None:
            boxed += 1
            s = ok(b, r["gold"])
            strict += s
            lenient += s
        else:
            lp = lenient_pred(text)
            if lp:
                nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", lp)
                lenient += ok(lp, r["gold"]) or bool(nums and ok(nums[-1].replace(",", ""), r["gold"]))
    n = len(rows)
    out = {"n": n, "boxed_pct": round(100 * boxed / n, 2), "strict_boxed_only_pct": round(100 * strict / n, 2),
           "original_style_lenient_pct": round(100 * lenient / n, 2),
           "ours_primary_pct": round(100 * sum(r["correct"] for r in rows) / n, 2)}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
