"""Exp 8 benchmark evaluation (run inside a SLURM job; see scripts/eval.sbatch).

    venv/bin/python exp8/eval.py --weights base --bench math500,gsm8k,ifeval --out exp8/results/base.json
    venv/bin/python exp8/eval.py --weights <edit_id | edited model dir | edit file> ...
    venv/bin/python exp8/eval.py --model gemma --bench math500 --samples 4 --out exp8/results/gemma_base.json
    add --limit 20 for a smoke test

Writes <out> (summary JSON) and <out stem>.items.jsonl.gz (one row per sample with the full text).
All benchmarks are submitted in ONE vLLM generate call (better GPU utilization than one call per benchmark).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp8 import common as C  # noqa: E402
from exp8.bench import grading as G  # noqa: E402


def resolve_weights(weights: str, family: str) -> tuple[Path, str]:
    """-> (model dir, edit_id)."""
    if weights == "base":
        return C.MODELS[family], "base"
    from exp8 import edits as E
    p = Path(weights)
    if p.is_dir():
        return p, p.name
    if p.exists():  # an edit file
        eid, _ = E.canonicalize(E.read_edit_file(p))
        return E.build_edited_dir(eid), eid
    return E.build_edited_dir(weights), weights   # an edit_id


def n_samples(samples, bench: str) -> int:
    """samples: an int for every benchmark, or a dict {bench: int} (e.g. {"math500": 4, "gsm8k": 2, "ifeval": 2})."""
    return int(samples.get(bench, 1)) if isinstance(samples, dict) else int(samples)


def build_requests(benches: list[str], family: str, tok, samples, limit: int | None, seed_base: int):
    reqs = []
    for b in benches:
        kind = C.kind_of(b)
        items = C.load_rows(C.BENCH_FILES[b])[: limit or None]
        for it in items:
            prompt = C.chat_prompt(tok, C.user_message(it, kind), family)
            for s in range(n_samples(samples, b)):
                reqs.append({"bench": b, "kind": kind, "item": it, "sample": s, "prompt": prompt,
                             "seed": C.stable_seed(seed_base, it["id"], s)})
    return reqs


def run_benchmarks(llm, family: str, benches: list[str], samples=1, limit: int | None = None,
                   seed_base: int = 0, max_tokens: int | None = None, log=print) -> tuple[dict, list[dict]]:
    tok = llm.get_tokenizer()
    reqs = build_requests(benches, family, tok, samples, limit, seed_base)
    log(f"{len(reqs)} requests: " + ", ".join(f"{b}={sum(r['bench'] == b for r in reqs)}" for b in benches))
    t = time.time()
    gens = C.generate(llm, [r["prompt"] for r in reqs], [r["seed"] for r in reqs], family, max_tokens)
    gen_s = time.time() - t
    n_tok = sum(g["n_tokens"] for g in gens)
    log(f"generated {n_tok} tokens in {gen_s:.0f}s ({n_tok / gen_s:.0f} tok/s)")
    t = time.time()
    grades = G.grade_many([(r["kind"], g["text"], r["item"], C.THINKING[family]) for r, g in zip(reqs, gens)])
    log(f"graded in {time.time() - t:.0f}s")
    rows = []
    for r, g, gr in zip(reqs, gens, grades):
        rows.append({"bench": r["bench"], "id": r["item"]["id"], "sample": r["sample"], "seed": r["seed"],
                     "gold": r["item"].get("answer"), **{k: v for k, v in g.items()}, **gr})
    summary = {}
    for b in benches:
        br = [x for x in rows if x["bench"] == b]
        s = C.summarize(br, C.kind_of(b))
        s["n_items"] = len({x["id"] for x in br})
        s["samples_per_item"] = n_samples(samples, b)
        if b == "math500":
            by = {}
            for x in br:
                by.setdefault(next(i["level"] for i in C.load_rows("math500") if i["id"] == x["id"]), []).append(x)
            s["accuracy_by_level"] = {str(k): round(100 * sum(y["correct"] for y in v) / len(v), 1)
                                      for k, v in sorted(by.items())}
        summary[b] = s
    timing = {"generate_s": round(gen_s, 1), "tokens": n_tok, "tok_per_s": round(n_tok / gen_s, 1)}
    return {"benchmarks": summary, "timing": timing}, rows


def settings(family: str, max_tokens: int | None) -> dict:
    return {"family": family, "sampling": C.SAMPLING[family], "max_tokens": max_tokens or C.MAX_TOKENS[family],
            "max_model_len": C.MAX_MODEL_LEN[family], "enable_thinking": C.THINKING[family],
            "prompt_formats": C.PROMPT_FORMATS, "grading": {
                "answer_section": "text after the last </think> (thinking models); empty if thinking never closed",
                "math": "last \\boxed{} in the answer section; math-verify (gold vs pred), else normalized string match;"
                        " if no \\boxed, math-verify extraction on the answer section",
                "gsm8k": "numeric match of the last \\boxed{} in the answer section, else its last number",
                "ifeval": "official google-research instruction_following_eval checker; primary = prompt-level strict",
                "hit_limit": "finish_reason == 'length'"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="base")
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemma"])
    ap.add_argument("--bench", default="math500,gsm8k,ifeval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default="1",
                    help='samples per item: an int, or per benchmark, e.g. "math500=4,gsm8k=2,ifeval=2"')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=None,
                    help="DIAGNOSTIC ONLY: override the fixed temperature (recorded in settings.sampling)")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="DIAGNOSTIC ONLY (matched-temperature controls): override presence_penalty")
    ap.add_argument("--top-k", type=int, default=None, help="DIAGNOSTIC ONLY: override top_k")
    a = ap.parse_args()
    import torch
    import vllm
    benches = a.bench.split(",")
    samples = ({k: int(v) for k, v in (x.split("=") for x in a.samples.split(","))} if "=" in a.samples
               else int(a.samples))
    if a.temperature is not None:
        C.SAMPLING[a.model] = dict(C.SAMPLING[a.model], temperature=a.temperature)
        print("WARNING: diagnostic temperature override", C.SAMPLING[a.model], flush=True)
    if a.presence_penalty is not None:
        C.SAMPLING[a.model] = dict(C.SAMPLING[a.model], presence_penalty=a.presence_penalty)
        print("WARNING: diagnostic presence_penalty override", C.SAMPLING[a.model], flush=True)
    if a.top_k is not None:
        C.SAMPLING[a.model] = dict(C.SAMPLING[a.model], top_k=a.top_k)
    model_dir, edit_id = resolve_weights(a.weights, a.model)
    t0 = time.time()
    llm = C.make_llm(model_dir, a.model, gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    load_s = time.time() - t0
    res, rows = run_benchmarks(llm, a.model, benches, samples, a.limit, a.seed, a.max_tokens,
                               log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    out = Path(a.out)
    items_path = out.with_name(out.stem + ".items.jsonl.gz")
    C.write_jsonl_gz(items_path, rows)
    doc = {"weights": a.weights, "edit_id": edit_id, "model_dir": str(model_dir), "started": C.now(),
           "settings": settings(a.model, a.max_tokens), "limit": a.limit, "seed": a.seed, **res,
           "timing": {**res["timing"], "load_s": round(load_s, 1), "total_s": round(time.time() - t0, 1)},
           "items_file": str(items_path), "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                                                       "python": platform.python_version()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["benchmarks"], indent=1), flush=True)
    if G._POOL is not None:   # clean shutdown of the grading pool (avoids a semaphore-cleanup traceback at exit)
        G._POOL.shutdown()


if __name__ == "__main__":   # vLLM spawns its engine process, which re-imports this file
    main()
