"""Exp 8 graders: MATH (boxed + math-verify + normalized string), GSM8K (numeric), IFEval (official checker).

All graders take the model's *answer section*: the text after the last `</think>`. A response that never closes its
thinking (it hit the token limit mid-thought) has an empty answer section and is graded wrong ("strict", primary).
The "lenient" variant grades the whole text (last \\boxed{} anywhere, including inside the thinking) and is reported
only as a diagnostic.

math-verify uses signal.alarm for its timeouts, which only works in a process's main thread, so callers grade
through `grade_many()`, which uses a spawn-based process pool.
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
THINK_END = "</think>"


def split_think(text: str) -> tuple[str | None, str]:
    """(thinking, answer). answer is '' when the thinking never closes. Models without thinking: (None, text)."""
    if THINK_END in text:
        i = text.rfind(THINK_END)
        return text[:i], text[i + len(THINK_END):]
    if "<think>" in text:
        return text, ""
    return None, text


def answer_section(text: str, thinking: bool) -> str:
    if not thinking:
        return text
    th, ans = split_think(text)
    if th is None:  # thinking model, but the generation prompt already contained <think>; no </think> seen
        return ""
    return ans


def all_boxed(s: str) -> list[str]:
    """Contents of every \\boxed{...} / \\fbox{...} (brace-balanced), in order. Also handles `\\boxed 5`."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", s):
        i = m.end()
        if i < len(s) and s[i] == "{":
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(s[i + 1:j])
                        break
        else:
            mm = re.match(r"([^\s$]+)", s[i:])
            if mm:
                out.append(mm.group(1))
    return out


def last_boxed(s: str) -> str | None:
    b = all_boxed(s)
    return b[-1] if b else None


# ----------------------------------------------------------------------------- string normalization (fallback)

def _strip_text(s):
    return re.sub(r"\\(?:text|textbf|mathrm|mbox|textit)\{([^{}]*)\}", r"\1", s)


def norm_math(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = _strip_text(s)
    for a, b in [("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""), ("\\left", ""), ("\\right", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""),
                 ("\\$", ""), ("$", ""), ("\\displaystyle", ""), ("{,}", ","), ("\\dots", "..."), ("\\ldots", "...")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^(?:x|y|a|b|n|k|answer)=", "", s)
    s = s.rstrip(".")
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    return s


# ----------------------------------------------------------------------------- MATH

def grade_math_answer(answer_text: str, gold: str) -> dict:
    """answer_text: the answer section. Returns {pred, correct, method}."""
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
    pred = last_boxed(answer_text)
    gold_parsed = parse("$" + gold + "$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    method = "none"
    correct = False
    if pred is not None:
        try:
            pp = parse("$" + pred + "$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if pp and gold_parsed and verify(gold_parsed, pp):
                correct, method = True, "math_verify"
        except Exception:  # noqa: BLE001
            pass
        if not correct and norm_math(pred) == norm_math(gold) and norm_math(gold):
            correct, method = True, "string"
    elif answer_text.strip():
        # no \boxed: let math-verify extract the last expression of the answer section (used for Gemma mostly)
        try:
            pp = parse(answer_text[-2000:])
            if pp:
                pred = str(pp[-1])
                if gold_parsed and verify(gold_parsed, pp):
                    correct, method = True, "math_verify_unboxed"
        except Exception:  # noqa: BLE001
            pass
    return {"pred": pred, "correct": bool(correct), "method": method}


# ----------------------------------------------------------------------------- GSM8K

_NUM = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def _to_float(s: str):
    s = s.replace(",", "").strip()
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def extract_number(s: str | None):
    if s is None:
        return None
    s = _strip_text(s).replace("\\$", "").replace("$", "").replace("\\%", "").replace("\\!", "")
    m = re.search(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        return _to_float(m.group(1) + "/" + m.group(2))
    nums = re.findall(_NUM, s)
    return _to_float(nums[-1]) if nums else None


def grade_gsm8k_answer(answer_text: str, gold: str) -> dict:
    gold_v = _to_float(gold)
    b = last_boxed(answer_text)
    method = "boxed" if b is not None else "last_number"
    v = extract_number(b) if b is not None else extract_number(answer_text)
    if v is None and b is not None:  # boxed but no number inside, fall back to the last number in the text
        v, method = extract_number(answer_text), "last_number"
    correct = v is not None and gold_v is not None and abs(v - gold_v) < 1e-6 * max(1.0, abs(gold_v))
    return {"pred": v, "correct": bool(correct), "method": method if v is not None else "none"}


# ----------------------------------------------------------------------------- IFEval (official checker)

_IFEVAL_READY = False


def _ifeval_import():
    global _IFEVAL_READY
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from instruction_following_eval import evaluation_lib  # noqa: F401
    _IFEVAL_READY = True
    return evaluation_lib


def grade_ifeval_answer(answer_text: str, item: dict) -> dict:
    ev = _ifeval_import()
    inp = ev.InputExample(key=item["key"], instruction_id_list=item["instruction_id_list"], prompt=item["prompt"],
                          kwargs=item["kwargs"])
    p2r = {item["prompt"]: answer_text}
    s = ev.test_instruction_following_strict(inp, p2r)
    l = ev.test_instruction_following_loose(inp, p2r)
    return {"correct": bool(s.follow_all_instructions), "strict_list": [bool(x) for x in s.follow_instruction_list],
            "loose_all": bool(l.follow_all_instructions), "loose_list": [bool(x) for x in l.follow_instruction_list],
            "pred": None, "method": "ifeval_strict"}


# ----------------------------------------------------------------------------- dispatch

def grade_one(args) -> dict:
    """args = (kind, text, item, thinking). kind: math | gsm8k | ifeval. Returns primary + lenient grades."""
    kind, text, item, thinking = args
    ans = answer_section(text, thinking)
    try:
        if kind == "math":
            g = grade_math_answer(ans, item["answer"])
            g["lenient_correct"] = g["correct"] or (grade_math_answer(text, item["answer"])["correct"] if thinking else g["correct"])
        elif kind == "gsm8k":
            g = grade_gsm8k_answer(ans, item["answer"])
            g["lenient_correct"] = g["correct"] or (grade_gsm8k_answer(text, item["answer"])["correct"] if thinking else g["correct"])
        elif kind == "ifeval":
            g = grade_ifeval_answer(ans, item)
            if thinking and not ans:  # diagnostic: grade the raw text (incl. thinking) when no answer was produced
                g2 = grade_ifeval_answer(text, item)
                g["lenient_correct"] = g2["correct"]
            else:
                g["lenient_correct"] = g["correct"]
        else:
            raise ValueError(kind)
    except Exception as e:  # noqa: BLE001
        g = {"pred": None, "correct": False, "lenient_correct": False, "method": f"error:{type(e).__name__}:{e}"[:200]}
    g["has_answer"] = bool(ans.strip())
    return g


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        _POOL = ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("spawn"))
    return _POOL


def grade_many(jobs: list[tuple]) -> list[dict]:
    if not jobs:
        return []
    return list(_pool().map(grade_one, jobs, chunksize=8))


def kind_of(pool_or_bench: str) -> str:
    if pool_or_bench.startswith("gsm8k"):
        return "gsm8k"
    if pool_or_bench.startswith("ifeval"):
        return "ifeval"
    return "math"


if __name__ == "__main__":  # quick self-test (CPU)
    T = "<think>\nlet me think</think>\n\nSo the answer is $\\boxed{\\dfrac{1}{2}}$."
    tests = [
        grade_one(("math", T, {"answer": "\\frac12"}, True))["correct"],
        grade_one(("math", T, {"answer": "0.5"}, True))["correct"],
        not grade_one(("math", "<think> loop loop \\boxed{1/2}", {"answer": "\\frac12"}, True))["correct"],
        grade_one(("math", "<think> loop loop \\boxed{1/2}", {"answer": "\\frac12"}, True))["lenient_correct"],
        grade_one(("math", "x</think>\\boxed{(3, \\frac{\\pi}{2})}", {"answer": "\\left( 3, \\frac{\\pi}{2} \\right)"}, True))["correct"],
        grade_one(("math", "x</think>\\boxed{\\text{(C)}}", {"answer": "\\text{(C)}"}, True))["correct"],
        grade_one(("gsm8k", "x</think> total is \\boxed{\\$1,080}", {"answer": "1080"}, True))["correct"],
        grade_one(("gsm8k", "x</think> so she pays 18 dollars.", {"answer": "18"}, True))["correct"],
        not grade_one(("gsm8k", "x</think> so she pays 18 dollars.", {"answer": "19"}, True))["correct"],
    ]
    import json
    items = [json.loads(l) for l in (HERE / "data/ifeval.jsonl").open()]
    it = [x for x in items if x["instruction_id_list"] == ["punctuation:no_comma"]] or items[:1]
    tests.append(grade_one(("ifeval", "</think>No commas here at all", it[0], True))["correct"] or it[0]["instruction_id_list"] != ["punctuation:no_comma"])
    tests.append(not grade_one(("ifeval", "<think>never ends", items[0], True))["correct"])
    print("selftest", tests, "ALL OK" if all(tests) else "FAIL")
