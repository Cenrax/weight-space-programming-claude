"""Thin, logged wrapper around the Anthropic Messages API (streaming + manual tool-use loop).

Every call writes logs/<exp_id>/<run_id>/{prompt.md, system.md, tools.json, transcript.jsonl,
thinking.md, response.md, result.json} per the PLAN.md run-log contract.
"""
from __future__ import annotations

import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import anthropic

from harness.logging_utils import LOGS_DIR, REPO_ROOT, append_jsonl, new_run_id, now_iso, run_dir, write_result

MODEL = "claude-opus-5-5"

# $ per 1M tokens: input, output, cache read, cache write (5 min), cache write (1 h).
PRICES: dict[str, tuple[float, float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0, 8.0),
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25, 10.0),
    "claude-sonnet-5": (2.0, 10.0, 0.20, 2.5, 4.0),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25, 2.0),
}

RETRYABLE = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)
ToolHandler = Callable[[dict[str, Any]], Any]


# ----------------------------------------------------------------------------- client

def load_env(path: Path | str = REPO_ROOT / ".env") -> None:
    """Minimal KEY=VALUE .env loader (does not override variables already set)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def has_api_key() -> bool:
    load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    load_env()
    return anthropic.Anthropic(max_retries=2, timeout=anthropic.Timeout(900.0, connect=30.0))


def cost_usd(usage: dict[str, int], model: str = MODEL) -> float:
    pin, pout, pread, pw5, pw1 = PRICES.get(model, PRICES[MODEL])
    w1 = usage.get("cache_write_1h", 0)
    w5 = usage.get("cache_creation_input_tokens", 0) - w1
    return (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout
            + usage.get("cache_read_input_tokens", 0) * pread + w5 * pw5 + w1 * pw1) / 1e6


def _usage_dict(u: Any) -> dict[str, int]:
    d = {k: int(getattr(u, k, 0) or 0) for k in
         ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    cc = getattr(u, "cache_creation", None)
    d["cache_write_1h"] = int(getattr(cc, "ephemeral_1h_input_tokens", 0) or 0) if cc else 0
    return d


# ----------------------------------------------------------------------------- result

@dataclass
class AskResult:
    text: str                     # final assistant text (last turn)
    thinking: str                 # all returned thinking summaries, in order
    usage: dict[str, int]         # summed over turns
    cost_usd: float
    stop_reason: str | None
    n_turns: int
    log_dir: Path
    run_id: str
    messages: list[dict[str, Any]] = field(repr=False, default_factory=list)
    tool_calls: list[dict[str, Any]] = field(repr=False, default_factory=list)
    stop_details: Any = None

    @property
    def tokens_in(self) -> int:
        u = self.usage
        return u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]


# ----------------------------------------------------------------------------- ask

GLOBAL_BUDGET_USD = 150.0  # hard project-wide cap agreed with the author; override with WSP_BUDGET_USD


class BudgetExhausted(RuntimeError):
    """Project-wide API spend (sum of cost_usd over logs/**/result.json) reached the cap."""


def total_spend(logs_root: Path | str = LOGS_DIR) -> float:
    """Sum cost_usd over every run's result.json (including crashed/partial runs)."""
    tot = 0.0
    for p in Path(logs_root).glob("*/*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def check_budget(logs_root: Path | str = LOGS_DIR) -> float:
    """Raise BudgetExhausted if spend >= cap; return remaining dollars."""
    cap = float(os.environ.get("WSP_BUDGET_USD", GLOBAL_BUDGET_USD))
    spent = total_spend(logs_root)
    if spent >= cap:
        raise BudgetExhausted(f"API spend ${spent:.2f} reached the ${cap:.0f} cap; ask the author before continuing")
    return cap - spent


def ask(prompt: str | list[dict[str, Any]], *, exp_id: str, run_id: str | None = None, system: str | None = None,
        thinking: bool = True, effort: str | None = "high", max_tokens: int = 64000,
        tools: list[dict[str, Any]] | None = None, tool_handlers: dict[str, ToolHandler] | None = None,
        max_turns: int = 50, model: str = MODEL, condition: str | None = None, cache: bool | None = None,
        max_cost_usd: float | None = None, logs_root: Path | str = LOGS_DIR, max_attempts: int = 8,
        extra: dict[str, Any] | None = None) -> AskResult:
    """Send `prompt` (one user turn), run the tool loop if `tools` are given, log everything.

    thinking: Opus 5.5 always thinks; True requests summarized thinking text back (display="summarized"),
              False omits the thinking param and defaults effort to "low".
    effort:   output_config.effort (low|medium|high|xhigh|max); None = API default (medium on Opus 5.5).
    cache:    top-level automatic prompt caching; defaults to on iff tools are given (multi-turn).
    max_cost_usd: stop the tool loop once cumulative cost exceeds this.
    extra:    additional fields merged into result.json.
    """
    check_budget(logs_root)
    run_id = run_id or new_run_id()
    d = run_dir(exp_id, run_id, logs_root)
    tools = [{**t, "eager_input_streaming": True} for t in tools] if tools else None
    handlers = tool_handlers or {}
    (d / "prompt.md").write_text(prompt if isinstance(prompt, str) else json.dumps(prompt, indent=1))
    if system:
        (d / "system.md").write_text(system)
    if tools:
        (d / "tools.json").write_text(json.dumps(tools, indent=1))
    tpath = d / "transcript.jsonl"
    tpath.write_text("")

    kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
    if system:
        kw["system"] = system
    if thinking:
        kw["thinking"] = {"type": "adaptive", "display": "summarized"}
    eff = "low" if not thinking and effort == "high" else effort
    if eff:
        kw["output_config"] = {"effort": eff}
    if tools:
        kw["tools"] = tools
    if cache if cache is not None else bool(tools):
        kw["cache_control"] = {"type": "ephemeral"}

    meta = {k: v for k, v in kw.items() if k not in ("system", "tools")}
    append_jsonl(tpath, {"event": "request_config", "time": now_iso(), **meta})
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    append_jsonl(tpath, {"event": "message", "role": "user", "content": "<see prompt.md>"})

    total = {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                            "cache_creation_input_tokens", "cache_write_1h")}
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    text, stop, stop_details, turn = "", None, None, 0
    final_nudge = False

    def _write_partial(status: str) -> None:
        write_result(d, exp_id=exp_id, run_id=run_id, model=model, condition=condition,
                     tokens_in=total["input_tokens"] + total["cache_read_input_tokens"]
                     + total["cache_creation_input_tokens"], tokens_out=total["output_tokens"],
                     cost_usd=round(cost_usd(total, model), 6), status=status, n_turns=turn,
                     stop_reason=stop, **(extra or {}))

    while True:
        turn += 1
        call_kw = dict(kw, messages=messages)
        if final_nudge:
            call_kw["tool_choice"] = {"type": "none"}
        msg = _stream_with_retry(call_kw, tpath, max_attempts)
        u = _usage_dict(msg.usage)
        for k in total:
            total[k] += u[k]
        stop, stop_details = msg.stop_reason, getattr(msg, "stop_details", None)
        blocks = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
        append_jsonl(tpath, {"event": "message", "role": "assistant", "turn": turn, "id": msg.id,
                             "stop_reason": stop, "stop_details": stop_details, "usage": u,
                             "cost_usd": round(cost_usd(u, model), 6), "content": blocks})
        for b in msg.content:
            if b.type == "thinking" and b.thinking:
                thinking_parts.append(b.thinking)
        turn_text = "".join(b.text for b in msg.content if b.type == "text")
        if turn_text:
            text = turn_text
        messages.append({"role": "assistant", "content": msg.content})
        _write_partial("running")

        uses = [b for b in msg.content if b.type == "tool_use"]
        if stop != "tool_use" and not (uses and stop == "max_tokens"):
            break
        if final_nudge:
            break
        results = []
        for b in uses:
            if stop in ("max_tokens", "refusal"):
                res = {"content": f"Tool call not executed: response was cut off ({stop}). Resend it shorter.",
                       "is_error": True}
            else:
                res = _run_tool(b, tools or [], handlers)
            tool_calls.append({"turn": turn, "name": b.name, "input": b.input, **res})
            results.append({"type": "tool_result", "tool_use_id": b.id, **res})
            append_jsonl(tpath, {"event": "tool_result", "turn": turn, "tool_use_id": b.id, "name": b.name, **res})
        over_cost = max_cost_usd is not None and cost_usd(total, model) >= max_cost_usd
        try:
            check_budget(logs_root)  # this run's partial cost is already in its result.json
        except BudgetExhausted:
            over_cost = True
        if turn >= max_turns or over_cost:
            why = "cost limit" if over_cost else "turn limit"
            results.append({"type": "text", "text": f"[harness] The {why} has been reached; no more tool calls "
                            "will run. Give your final answer now."})
            final_nudge = True
            append_jsonl(tpath, {"event": "final_nudge", "reason": why})
        messages.append({"role": "user", "content": results})

    thinking_text = "\n\n---\n\n".join(thinking_parts)
    (d / "response.md").write_text(text)
    if thinking_text:
        (d / "thinking.md").write_text(thinking_text)
    _write_partial("complete" if stop != "refusal" else "refusal")
    return AskResult(text=text, thinking=thinking_text, usage=total, cost_usd=cost_usd(total, model),
                     stop_reason=stop, n_turns=turn, log_dir=d, run_id=run_id, messages=messages,
                     tool_calls=tool_calls, stop_details=stop_details)


def _stream_with_retry(call_kw: dict[str, Any], tpath: Path, max_attempts: int) -> Any:
    client = get_client()
    for attempt in range(1, max_attempts + 1):
        try:
            with client.messages.stream(**call_kw) as stream:
                return stream.get_final_message()
        except RETRYABLE as e:
            err = e
        except ValueError as e:      # unparseable eager-streamed tool JSON -> re-issue the request
            err = e
        if attempt == max_attempts:
            raise err
        delay = min(120.0, 5.0 * 2 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
        resp = getattr(err, "response", None)
        if resp is not None and resp.headers.get("retry-after"):
            try:
                delay = max(delay, float(resp.headers["retry-after"]))
            except ValueError:
                pass
        append_jsonl(tpath, {"event": "retry", "attempt": attempt, "error": f"{type(err).__name__}: {err}"[:500],
                             "sleep_s": round(delay, 1)})
        time.sleep(delay)
    raise RuntimeError("unreachable")


def _run_tool(block: Any, tools: list[dict[str, Any]], handlers: dict[str, ToolHandler]) -> dict[str, Any]:
    spec = next((t for t in tools if t.get("name") == block.name), None)
    if spec is None or block.name not in handlers:
        return {"content": f"Unknown tool: {block.name}", "is_error": True}
    problem = _validate(block.input, spec.get("input_schema", {}))
    if problem:
        return {"content": json.dumps({"INVALID_INPUT": problem, "received": block.input})[:4000], "is_error": True}
    try:
        out = handlers[block.name](block.input)
    except Exception as e:  # report tool failures back to Claude rather than crash the run
        return {"content": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"[-4000:], "is_error": True}
    if isinstance(out, dict) and "content" in out:
        return {"content": out["content"], **({"is_error": True} if out.get("is_error") else {})}
    return {"content": out if isinstance(out, (str, list)) else json.dumps(out, default=str)}


def _validate(inp: Any, schema: dict[str, Any]) -> str | None:
    """Minimal JSON-schema check (object, required keys, primitive types) for eager-streamed inputs."""
    if not isinstance(inp, dict):
        return "input is not an object"
    props = schema.get("properties", {})
    for k in schema.get("required", []):
        if k not in inp:
            return f"missing required field '{k}'"
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
    for k, v in inp.items():
        t = props.get(k, {}).get("type")
        if t in types and not isinstance(v, types[t]):
            return f"field '{k}' should be {t}"
    return None
