"""Aggregate logs/ + checkpoints/ into docs/data/<exp_id>.json (PLAN.md blog-data schema).

    venv/bin/python scripts/build_blog_data.py                 # all experiments -> docs/data/
    venv/bin/python scripts/build_blog_data.py --exp exp1 --out /tmp/data

Per-run fields come from logs/<exp>/<run>/result.json (+ prompt.md / response.md excerpts). Optional
result.json keys honoured: label, group, score, tokens, weights_excerpt, reasoning_excerpt, media, status.
An optional logs/<exp>/summary.json ({"summary", "table", "media", "status", "charts"}) overrides the defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

EXPERIMENTS: dict[str, str] = {
    "exp1": "Read: which Atari game?",
    "exp1b": "Read: which classic-control task?",
    "exp2": "Read: what does the DDPM draw?",
    "exp3": "Read: which language + copying heads",
    "exp4": "Fix: failing policy",
    "exp5": "Write from scratch, zero-shot",
    "exp5b": "Write with rollout budget",
    "exp6": "Write Go",
    "exp7": "Read: which Gemma 3 270M is sycophantic",
}
COLUMNS = ["run", "condition", "target", "answer", "correct", "confidence", "score", "tokens_out", "cost_usd"]


# Optional per-run fields the site renders (see docs/assets/js/app.js).
RUN_EXTRAS = ("group", "score", "tokens", "weights_excerpt")


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _excerpt(p: Path, n: int) -> str | None:
    if not p.exists():
        return None
    s = p.read_text(errors="replace").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def collect_runs(exp_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()) if exp_dir.exists() else []:
        r = _load(d / "result.json")
        if not isinstance(r, dict):
            continue
        r["_dir"] = d
        runs.append(r)
    return runs


def build_exp(exp_id: str, logs: Path, ckpts: Path) -> dict[str, Any]:
    runs = collect_runs(logs / exp_id)
    override = _load(logs / exp_id / "summary.json") or {}
    checkpoints = []
    for p in sorted((ckpts / exp_id).glob("*.json")) if (ckpts / exp_id).exists() else []:
        meta = _load(p)
        if isinstance(meta, dict):
            checkpoints.append({"name": p.stem, **{k: meta.get(k) for k in
                                ("env_id", "arch", "algo", "eval_mean_return", "eval_std_return", "seed")}})

    rows, run_items, media = [], [], []
    for r in runs:
        d: Path = r["_dir"]
        rows.append([r.get("run_id") or d.name] + [r.get(c) for c in COLUMNS[1:]])
        run_items.append({
            "label": r.get("label") or " · ".join(str(x) for x in (r.get("target"), r.get("condition")) if x)
                     or d.name,
            "answer": r.get("answer"), "correct": r.get("correct"), "confidence": r.get("confidence"),
            "reasoning_excerpt": r.get("reasoning_excerpt") or _excerpt(d / "response.md", 800),
            "prompt_excerpt": r.get("prompt_excerpt") or _excerpt(d / "prompt.md", 400),
            **{k: r[k] for k in RUN_EXTRAS if r.get(k) is not None},
            "log_path": f"logs/{exp_id}/{d.name}",
        })
        media += [m for m in r.get("media") or [] if isinstance(m, dict)]

    finished = [r for r in runs if r.get("status") != "running"
                and any(r.get(k) is not None for k in ("answer", "correct", "score"))]
    status = "pending" if not runs else ("done" if len(finished) == len(runs) else "running")
    graded = [r for r in runs if r.get("correct") is not None]
    cost = sum(r.get("cost_usd") or 0 for r in runs)
    if not runs:
        summary = "Not run yet." + (f" {len(checkpoints)} checkpoint(s) trained." if checkpoints else "")
    else:
        summary = f"{len(runs)} run(s)"
        if graded:
            summary += f"; {sum(bool(r['correct']) for r in graded)}/{len(graded)} correct"
        summary += f"; total API cost ${cost:.2f}."
    out = {"exp_id": exp_id, "title": EXPERIMENTS.get(exp_id, exp_id), "status": override.get("status", status),
           "summary": override.get("summary", summary),
           "table": override.get("table", {"columns": COLUMNS, "rows": rows}),
           "runs": run_items, "media": override.get("media", media), "checkpoints": checkpoints}
    if "charts" in override:
        out["charts"] = override["charts"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=REPO / "logs")
    ap.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "data")
    ap.add_argument("--exp", action="append", help="experiment id(s); default: all known + any in logs/")
    a = ap.parse_args(argv)
    ids = a.exp or list(EXPERIMENTS) + sorted(
        p.name for p in (a.logs.iterdir() if a.logs.exists() else [])
        if p.is_dir() and p.name not in EXPERIMENTS and not p.name.startswith(("_", ".")))
    a.out.mkdir(parents=True, exist_ok=True)
    for exp_id in ids:
        data = build_exp(exp_id, a.logs, a.checkpoints)
        (a.out / f"{exp_id}.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
        print(f"{exp_id}: {data['status']:8s} {len(data['runs'])} run(s) -> {a.out / (exp_id + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
