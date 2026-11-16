"""Exp 8 shared pieces: paths, fixed sampling settings, prompt formats, vLLM construction, generation, loop markers,
summaries. Used by eval.py (standalone SLURM eval) and service.py (practice / eval HTTP service).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

WSP = Path("/path/to/wsp")
EXP8 = WSP / "exp8"
DATA = EXP8 / "bench" / "data"
MODELS = {"qwen": WSP / "models" / "Qwen3.5-0.8B", "gemma": WSP / "models" / "gemma-3-270m-it"}
# which model the service / edit tools work on (set EXP8_FAMILY=gemma for the Gemma control; default Qwen)
FAMILY = os.environ.get("EXP8_FAMILY", "qwen")
LM_PREFIXES = {"qwen": "model.language_model.", "gemma": "model."}

# ---- fixed sampling (never change between runs; recorded in every result file)
SAMPLING = {
    # Qwen3.5-0.8B: the original post's published protocol (`think35` in their evaluate.py; the model card's
    # thinking-mode setting for reasoning benchmarks). DELIBERATE CHANGE on 2026-11-14 from temperature 1.0 (the
    # card's "general text" setting): job 2047 showed 0.6 reproduces the original's base MATH-500 (7.8% vs 7.9%,
    # 92% vs 93% hitting 32k), so all Exp 8 phase-2 results (practice, base re-eval, final evals) use 0.6.
    "qwen": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0),
    # Gemma 3 model card (as in the original post): temperature 1.0, top-k 64, top-p 0.95
    "gemma": dict(temperature=1.0, top_p=0.95, top_k=64, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0),
}
MAX_TOKENS = {"qwen": 32768, "gemma": 8192}
MAX_MODEL_LEN = {"qwen": 34816, "gemma": 8192 + 2048}   # max_tokens + 2048 prompt headroom
THINKING = {"qwen": True, "gemma": False}               # enable_thinking passed to the chat template

# ---- prompt formats (exact strings; recorded in every result file)
MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT_FORMATS = {
    "math": "{problem}\n" + MATH_SUFFIX,      # MATH-500 and the MATH train pool
    "gsm8k": "{problem}\n" + MATH_SUFFIX,     # GSM8K test and train pool
    "ifeval": "{prompt}",                     # IFEval: the prompt verbatim
}
BENCH_FILES = {"math500": "math500", "gsm8k": "gsm8k_test", "ifeval": "ifeval"}
POOL_FILES = {"gsm8k_train": "gsm8k_train", "math_train": "math_train"}


def kind_of(name: str) -> str:
    return "gsm8k" if name.startswith("gsm8k") else ("ifeval" if name.startswith("ifeval") else "math")


_CACHE: dict[str, list] = {}


def load_rows(file_stem: str) -> list[dict]:
    if file_stem not in _CACHE:
        _CACHE[file_stem] = [json.loads(l) for l in (DATA / f"{file_stem}.jsonl").open()]
    return _CACHE[file_stem]


def user_message(item: dict, kind: str) -> str:
    # str.replace, not str.format: the template contains "\\boxed{}"
    if kind == "ifeval":
        return PROMPT_FORMATS["ifeval"].replace("{prompt}", item["prompt"])
    return PROMPT_FORMATS[kind].replace("{problem}", item["problem"])


def chat_prompt(tok, msg: str, family: str) -> str:
    kw = {"enable_thinking": THINKING[family]} if family == "qwen" else {}
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, **kw)


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def family_of(model_dir: str | Path) -> str:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return "gemma" if "gemma" in cfg.get("model_type", "") else "qwen"


def make_llm(model_dir, family: str, gpu_memory_utilization: float = 0.85, max_num_seqs: int = 512, seed: int = 0,
             worker_ext: bool = False):
    from vllm import LLM
    kw = dict(model=str(model_dir), max_model_len=MAX_MODEL_LEN[family], gpu_memory_utilization=gpu_memory_utilization,
              max_num_seqs=max_num_seqs, seed=seed, enable_prefix_caching=False)
    if family == "qwen":
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}   # text only
    if worker_ext:
        kw["worker_extension_cls"] = "exp8.vllm_ext.WeightSwapExtension"
    return LLM(**kw)


def generate(llm, prompts: list[str], seeds: list[int], family: str, max_tokens: int | None = None,
             keep_ids: bool = False) -> list[dict]:
    """One vLLM generate call. Returns per prompt {text, n_tokens, finish_reason, hit_limit, max_tokens}."""
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    mt = max_tokens or MAX_TOKENS[family]
    sps, tps = [], []
    for p, s in zip(prompts, seeds):
        # the chat template already contains <bos> for Gemma: tokenize without adding special tokens again
        ids = tok(p, add_special_tokens=False).input_ids
        tps.append({"prompt_token_ids": ids})
        sps.append(SamplingParams(**SAMPLING[family], max_tokens=min(mt, MAX_MODEL_LEN[family] - len(ids)), seed=s))
    outs = llm.generate(tps, sps, use_tqdm=False)
    res = []
    for o, sp in zip(outs, sps):
        c = o.outputs[0]
        res.append({"text": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                    "hit_limit": c.finish_reason == "length", "max_tokens": sp.max_tokens,
                    "n_prompt_tokens": len(o.prompt_token_ids)})
        if keep_ids:
            res[-1]["prompt_ids"] = list(o.prompt_token_ids)
            res[-1]["gen_ids"] = list(c.token_ids)
    return res


# ----------------------------------------------------------------------------- loop markers

def _longest_repeat(words: list[str]) -> tuple[int, int]:
    """Longest n such that some n-word span occurs at least twice (overlap allowed). Returns (n, start of 2nd copy)."""
    if len(words) < 2:
        return 0, -1
    ids = {}
    w = [ids.setdefault(x, len(ids) + 1) for x in words]
    M, B = (1 << 61) - 1, 1_000_003
    L = len(w)
    pre = [0] * (L + 1)
    pw = [1] * (L + 1)
    for i, x in enumerate(w):
        pre[i + 1] = (pre[i] * B + x) % M
        pw[i + 1] = pw[i] * B % M

    def find(n):
        seen = {}
        for i in range(L - n + 1):
            h = (pre[i + n] - pre[i] * pw[n]) % M
            if h in seen:
                return i
            seen[h] = i
        return -1

    lo, hi, best = 1, L - 1, (0, -1)
    while lo <= hi:
        mid = (lo + hi) // 2
        j = find(mid)
        if j >= 0:
            best = (mid, j)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def loop_markers(text: str, window_chars: int = 20000) -> dict:
    tail = text[-window_chars:]
    words = tail.split()
    n, j = _longest_repeat(words)
    k = 200
    last = text[-k:]
    occ = tail.count(last) if len(text) >= 2 * k else 0
    prev = text.rfind(last, 0, len(text) - k) if len(text) >= 2 * k else -1
    ng = [" ".join(words[i:i + 8]) for i in range(max(0, len(words) - 7))]
    return {
        "longest_repeat_words": n,                                   # in the last `window_chars` characters
        "longest_repeat_frac": round(n / max(1, len(words)), 3),     # 0.5 = the tail is one block said twice
        "repeat_example": " ".join(words[j:j + n])[:300] if n else "",
        "tail200_count": occ,                                        # times the final 200 chars occur in the tail
        "tail200_period_chars": (len(text) - k - prev) if prev >= 0 else None,
        "dup_8gram_frac": round(1 - len(set(ng)) / len(ng), 3) if ng else 0.0,
    }


# ----------------------------------------------------------------------------- summaries

def summarize(rows: list[dict], kind: str) -> dict:
    """rows: per sample {correct, lenient_correct, hit_limit, n_tokens, has_answer, [strict_list, loose_all, ...]}"""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    lens = [r["n_tokens"] for r in rows]
    out = {
        "n": n,
        "accuracy": round(100 * sum(r["correct"] for r in rows) / n, 2),
        "hit_limit_pct": round(100 * sum(r["hit_limit"] for r in rows) / n, 2),
        "mean_len": round(statistics.mean(lens), 1),
        "median_len": statistics.median(lens),
        "has_answer_pct": round(100 * sum(r.get("has_answer", False) for r in rows) / n, 2),
        "lenient_accuracy": round(100 * sum(r.get("lenient_correct", r["correct"]) for r in rows) / n, 2),
        "accuracy_among_finished": (round(100 * sum(r["correct"] for r in rows if not r["hit_limit"]) /
                                          max(1, sum(not r["hit_limit"] for r in rows)), 2)),
    }
    if kind == "ifeval":
        inst_s = [x for r in rows for x in r.get("strict_list", [])]
        inst_l = [x for r in rows for x in r.get("loose_list", [])]
        out.update({"metric": "prompt-level strict accuracy (primary = accuracy)",
                    "prompt_strict": out["accuracy"],
                    "prompt_loose": round(100 * sum(r.get("loose_all", False) for r in rows) / n, 2),
                    "inst_strict": round(100 * sum(inst_s) / max(1, len(inst_s)), 2),
                    "inst_loose": round(100 * sum(inst_l) / max(1, len(inst_l)), 2)})
    return out


def write_jsonl_gz(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=6) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
