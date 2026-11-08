"""Run ids, run directories and the result.json contract (see PLAN.md: Run logs)."""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"

RESULT_FIELDS = ("exp_id", "run_id", "model", "condition", "target", "answer", "correct",
                 "confidence", "score", "tokens_in", "tokens_out", "cost_usd", "timestamp")


def new_run_id(tag: str | None = None) -> str:
    """Sortable unique id, e.g. `20261107-194512-a3f1` or `20261107-194512-a3f1-breakout`."""
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    return f"{rid}-{tag}" if tag else rid


def run_dir(exp_id: str, run_id: str, root: Path | str = LOGS_DIR) -> Path:
    d = Path(root) / exp_id / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_result(d: Path | str) -> dict[str, Any]:
    p = Path(d) / "result.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_result(d: Path | str, **fields: Any) -> dict[str, Any]:
    """Merge `fields` into `<d>/result.json` (creating it with all contract keys = null).

    A None value never overwrites an existing non-None value, so partial writers
    (client: tokens/cost; experiment runner: answer/correct/score) compose safely.
    """
    res = {k: None for k in RESULT_FIELDS}
    res.update(read_result(d))
    for k, v in fields.items():
        if v is not None or res.get(k) is None:
            res[k] = v
    res["timestamp"] = fields.get("timestamp") or now_iso()
    p = Path(d) / "result.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=2, default=_json_default))
    tmp.replace(p)
    return res


def append_jsonl(path: Path | str, obj: Any) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)
