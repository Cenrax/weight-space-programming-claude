"""Download and freeze the Exp 8 benchmarks and practice pools as jsonl (CPU only, login shell is fine).

    venv/bin/python exp8/bench/prepare_data.py

Writes exp8/bench/data/{math500,gsm8k_test,ifeval,gsm8k_train,math_train}.jsonl plus manifest.json
(source dataset, revision, n, sha256, overlap checks). Every row has a stable string `id`.
"""
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus"]


def last_boxed(s):
    """Content of the last \\boxed{...} / \\fbox{...} in s (brace-balanced), or None."""
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def norm_problem(s):
    return re.sub(r"\s+", " ", s).strip()


def write(name, rows, meta, manifest):
    p = OUT / f"{name}.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest[name] = dict(meta, n=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    print(name, len(rows), meta)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"id": f"math500-{i}", "unique_id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
             "subject": r["subject"], "level": r["level"]} for i, r in enumerate(ds)]
    write("math500", rows, {"source": "HuggingFaceH4/MATH-500", "split": "test"}, manifest)
    math500_problems = {norm_problem(r["problem"]) for r in rows}

    g = load_dataset("openai/gsm8k", "main")
    for split, name in [("test", "gsm8k_test"), ("train", "gsm8k_train")]:
        rows = []
        for i, r in enumerate(g[split]):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"id": f"gsm8k-{split}-{i}", "problem": r["question"], "answer": gold, "solution": r["answer"]})
        write(name, rows, {"source": "openai/gsm8k", "config": "main", "split": split}, manifest)
    gtest = {norm_problem(r["question"]) for r in g["test"]}
    gtrain_overlap = sum(norm_problem(r["question"]) in gtest for r in g["train"])
    manifest["gsm8k_train"]["overlap_with_test_exact"] = gtrain_overlap

    # MATH train. hendrycks/competition_math was taken down (DMCA); EleutherAI/hendrycks_math is the same data
    # split by subject. MATH-500 is a subset of the 5,000 MATH *test* problems, so the train split is disjoint.
    rows, src = [], None
    try:
        ds = load_dataset("hendrycks/competition_math", split="train")
        src = "hendrycks/competition_math"
        parts = [("all", ds)]
    except Exception as e:  # noqa: BLE001
        print("hendrycks/competition_math unavailable:", type(e).__name__, str(e)[:200])
        src = "EleutherAI/hendrycks_math"
        parts = [(s, load_dataset(src, s, split="train")) for s in MATH_SUBJECTS]
    for subj, ds in parts:
        for i, r in enumerate(ds):
            ans = last_boxed(r["solution"])
            if ans is None:
                continue
            rows.append({"id": f"math-train-{subj}-{i}", "problem": r["problem"], "answer": ans,
                         "subject": r.get("type", subj), "level": r.get("level")})
    overlap = sum(norm_problem(r["problem"]) in math500_problems for r in rows)
    rows = [r for r in rows if norm_problem(r["problem"]) not in math500_problems]
    write("math_train", rows, {"source": src, "split": "train", "overlap_with_math500_removed": overlap}, manifest)

    # IFEval: 541 prompts from google/IFEval (identical to google-research instruction_following_eval/data).
    ds = load_dataset("google/IFEval", split="train")
    rows = [{"id": f"ifeval-{r['key']}", "key": r["key"], "prompt": r["prompt"],
             "instruction_id_list": r["instruction_id_list"],
             "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]} for r in ds]
    off = [json.loads(l) for l in (HERE / "instruction_following_eval/data/input_data.jsonl").open()]
    same = {r["key"]: r["prompt"] for r in off} == {r["key"]: r["prompt"] for r in rows}
    write("ifeval", rows, {"source": "google/IFEval", "split": "train",
                           "identical_prompts_to_google_research_input_data": same}, manifest)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
