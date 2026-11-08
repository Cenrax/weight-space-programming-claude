"""Persistent Python sandbox for Claude with budgeted access to the network/environment.

The sandbox is a separate process holding only numpy copies of the weights. Functions that run the
network (forward_pass, rollout, ...) are registered here, execute in THIS (parent) process, and are
exposed inside the sandbox as proxies; each call is counted against a hard budget and logged.
Env/project packages and process creation are blocked in the child. The child runs from a temp dir outside
the repo and, on macOS, under sandbox-exec: no reads/writes inside the repo (except the venv), no network.

    sb = Sandbox(params, log_dir=d, timeout=120)
    sb.register("rollout", lambda n=128: run_batch(sb.weights, n), budget=10, doc="rollout(n=128) -> dict")
    tools, handlers = sb.tools(), sb.handlers()
    ask(prompt, exp_id=..., tools=tools, tool_handlers=handlers)
"""
from __future__ import annotations

import json
import os
import pickle
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from harness.logging_utils import REPO_ROOT, append_jsonl
from harness.weights_text import parse_weights

DEFAULT_BLOCKED = ("gymnasium", "gym", "pettingzoo", "ale_py", "shimmy", "Box2D", "box2d", "mujoco", "pygame",
                   "envs", "train", "eval", "experiments", "harness", "scripts",
                   "subprocess", "multiprocessing", "pty", "socket", "urllib", "http", "requests", "httpx")



# macOS Seatbelt profile (last matching rule wins): deny the whole home dir (repo, HF cache, ~/.pyenv ...) and the temp
# dirs, then re-allow the worker's own scratch dir, the venv and the base Python install.
_SB_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-read* file-write* (subpath "{home}"))
(deny file-read* file-write* (subpath "/private/tmp"))
(deny file-read* file-write* (subpath "{tmp}"))
(allow file-read* file-write* (subpath "{cwd}"))
(allow file-read* (subpath "{venv}"))
(allow file-read* (subpath "{pybase}"))
(allow file-read-metadata)
"""


@dataclass
class _Fn:
    fn: Callable[..., Any]
    budget: int | None
    doc: str
    used: int = 0


class Sandbox:
    """Persistent exec session + budgeted proxy functions + weight saving."""

    def __init__(self, weights: Mapping[str, np.ndarray] | None = None, *, log_dir: Path | str,
                 timeout: float = 120.0, max_output_chars: int = 20000, threads: int = 2,
                 blocked: tuple[str, ...] = DEFAULT_BLOCKED, preload: tuple[str, ...] = (), preamble: str = "",
                 allow_new_tensors: bool = False, isolate: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.init_weights = OrderedDict((k, np.array(v, np.float32)) for k, v in (weights or {}).items())
        self.weights = OrderedDict((k, v.copy()) for k, v in self.init_weights.items())
        self.timeout, self.max_output_chars, self.threads = timeout, max_output_chars, threads
        self.blocked, self.preload, self.preamble = blocked, preload, preamble
        self.allow_new_tensors, self.isolate = allow_new_tensors, isolate
        self.fns: dict[str, _Fn] = {}
        self.n_saves = 0
        self.proc: subprocess.Popen | None = None
        self._buf = b""
        self.register("save_weights", self._save_from_child, doc=(
            "save_weights(params=None, **arrays) -> str. Save weight tensors (e.g. save_weights(W1=W1, b1=b1) or "
            "save_weights(weights)); only the given tensors are replaced. Budgeted functions use the saved weights."))
        self.register("load_saved_weights", lambda: dict(self.weights),
                      doc="load_saved_weights() -> dict of the currently saved weight arrays.")
        self.register("budget_status", self.budget_status, doc="budget_status() -> {name: {used, budget}}. Free.")

    # ------------------------------------------------------------------ registration / budgets
    def register(self, name: str, fn: Callable[..., Any], *, budget: int | None = None, doc: str = "") -> None:
        """Expose `fn` inside the sandbox as `name(...)`; budget=N hard-limits it to N calls. Call before start()."""
        if self.proc is not None:
            raise RuntimeError("register functions before the sandbox starts")
        self.fns[name] = _Fn(fn, budget, doc or (fn.__doc__ or name))

    def budget_status(self) -> dict[str, dict[str, int | None]]:
        return {n: {"used": f.used, "budget": f.budget} for n, f in self.fns.items() if f.budget is not None}

    def _call(self, name: str, args: tuple, kwargs: dict) -> tuple[bool, Any]:
        f = self.fns.get(name)
        if f is None:
            return False, f"NameError: no sandbox function {name!r}"
        if f.budget is not None and f.used >= f.budget:
            self._log_call(name, f, args, kwargs, False, "budget exhausted", 0.0)
            return False, f"BudgetExceeded: {name} budget exhausted ({f.used}/{f.budget} used); no calls remain."
        if f.budget is not None:
            f.used += 1
        t0 = time.time()
        try:
            out, ok = f.fn(*args, **kwargs), True
        except Exception as e:  # failed calls are refunded and reported back into the sandbox
            if f.budget is not None:
                f.used -= 1
            out, ok = f"{type(e).__name__}: {e}", False
        self._log_call(name, f, args, kwargs, ok, out, time.time() - t0)
        return ok, out

    def _log_call(self, name: str, f: _Fn, args: tuple, kwargs: dict, ok: bool, out: Any, dt: float) -> None:
        if name in ("budget_status", "load_saved_weights"):
            return
        append_jsonl(self.log_dir / "sandbox_calls.jsonl", {
            "time": time.strftime("%H:%M:%S"), "name": name, "used": f.used, "budget": f.budget, "ok": ok,
            "duration_s": round(dt, 3), "args": _summ(args), "kwargs": _summ(kwargs), "result": _summ(out)})

    # ------------------------------------------------------------------ weights
    def save_weights(self, params: Mapping[str, Any]) -> str:
        """Validate and store weights (partial updates allowed); writes weights/save_NNN.npz + latest.npz."""
        new = OrderedDict(self.weights)
        for k, v in params.items():
            a = np.asarray(v, np.float32)
            if k in self.weights and a.shape != self.weights[k].shape:
                raise ValueError(f"{k}: shape {a.shape} != expected {self.weights[k].shape}")
            if k not in self.weights and not self.allow_new_tensors:
                raise ValueError(f"unknown tensor {k!r}; expected one of {list(self.weights)}")
            if not np.isfinite(a).all():
                raise ValueError(f"{k}: contains NaN/inf")
            new[k] = a.copy()
        self.weights = new
        self.n_saves += 1
        wd = self.log_dir / "weights"
        wd.mkdir(exist_ok=True)
        np.savez(wd / f"save_{self.n_saves:03d}.npz", **self.weights)
        np.savez(wd / "latest.npz", **self.weights)
        return f"saved {sorted(params)} (save #{self.n_saves})"

    def _save_from_child(self, params: Mapping[str, Any] | None = None, **arrays: Any) -> str:
        return self.save_weights({**(params or {}), **arrays})

    # ------------------------------------------------------------------ process
    def start(self) -> "Sandbox":
        # The worker runs from a scratch dir outside the repo. On macOS it is also wrapped in sandbox-exec so
        # Claude's code cannot read the repo (checkpoint metadata, ground truth, .env) or use the network.
        self._rm_cwd()
        cwd = self._cwd = Path(tempfile.mkdtemp(prefix="wsp_sandbox_"))
        shutil.copy(Path(__file__).with_name("_sandbox_worker.py"), cwd / "_sandbox_worker.py")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}
        env.update(SANDBOX_REPO_ROOT=str(REPO_ROOT), SANDBOX_THREADS=str(self.threads), PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        cmd = [sys.executable, str(cwd / "_sandbox_worker.py")]
        if self.isolate and shutil.which("sandbox-exec"):
            (cwd / "profile.sb").write_text(_SB_PROFILE.format(
                repo=REPO_ROOT.resolve(), home=Path.home().resolve(), tmp=Path(tempfile.gettempdir()).resolve(),
                cwd=cwd.resolve(), venv=Path(sys.prefix).resolve(), pybase=Path(sys.base_prefix).resolve()))
            cmd = ["sandbox-exec", "-f", str(cwd / "profile.sb"), *cmd]
        elif self.isolate:
            print("warning: sandbox-exec unavailable; sandbox can read repo files", file=sys.stderr)
        if getattr(self, "_stderr", None):
            self._stderr.close()
        self._stderr = open(self.log_dir / "sandbox_stderr.log", "ab")
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                     bufsize=0, start_new_session=True)
        self._buf = b""
        self._send(("init", {"weights": dict(self.weights), "blocked": list(self.blocked), "preamble": self.preamble,
                             "preload": list(self.preload), "functions": {n: f.doc for n, f in self.fns.items()}}))
        msg = self._recv(60.0)
        if not msg or msg[0] != "ready":
            raise RuntimeError("sandbox failed to start; see sandbox_stderr.log")
        return self

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self._send(("exit",))
                self.proc.wait(2)
            except Exception:
                pass
        self._reap()
        if getattr(self, "_stderr", None):
            self._stderr.close()
            self._stderr = None
        self._rm_cwd()

    def _rm_cwd(self) -> None:
        if getattr(self, "_cwd", None):
            shutil.rmtree(self._cwd, ignore_errors=True)
            self._cwd = None

    def _reap(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            for f in (self.proc.stdin, self.proc.stdout):
                f and f.close()
        self.proc = None

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def run(self, code: str) -> str:
        """Execute code in the persistent namespace; returns captured output (truncated)."""
        if self.proc is None or self.proc.poll() is not None:
            self._reap()
            self.start()
        self._send(("exec", code))
        deadline, interrupted = time.time() + self.timeout, False
        while True:
            msg = self._recv(max(0.0, deadline - time.time()))
            if msg is None:                          # timed out waiting for the child
                if not interrupted:
                    os.kill(self.proc.pid, signal.SIGINT)
                    interrupted, deadline = True, time.time() + 5.0
                    continue
                self._reap()
                self.start()
                return (f"TimeoutError: execution exceeded {self.timeout:.0f}s and could not be interrupted; the "
                        "sandbox was restarted (all variables were reset; saved weights and budgets are kept, "
                        "use load_saved_weights()).")
            if msg[0] == "call":
                t0 = time.time()
                ok, val = self._call(msg[1], msg[2], msg[3])
                try:
                    self._send(("ret", ok, val))
                except (pickle.PicklingError, TypeError, AttributeError):
                    self._send(("ret", ok, repr(val)))
                deadline += time.time() - t0             # parent-side time doesn't count against the timeout
            elif msg[0] == "done":
                return _truncate(msg[1] or "(no output)", self.max_output_chars)
            elif msg[0] == "dead":
                self._reap()
                self.start()
                return "Error: the sandbox process died (variables were reset; use load_saved_weights())."

    def _send(self, obj: Any) -> None:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)

    def _recv(self, timeout: float) -> Any:
        end = time.time() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            if len(self._buf) >= 8:
                (n,) = struct.unpack("<Q", self._buf[:8])
                if len(self._buf) >= 8 + n:
                    data, self._buf = self._buf[8:8 + n], self._buf[8 + n:]
                    return pickle.loads(data)
            left = end - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([fd], [], [], left)
            if r:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    return ("dead",)
                self._buf += chunk

    # ------------------------------------------------------------------ Anthropic tools
    def tools(self, extra_description: str = "") -> list[dict[str, Any]]:
        """Tool definitions: `python` (persistent exec) and `save_weights` (weights as text)."""
        shapes = ", ".join(f"{k}{list(v.shape)}" for k, v in self.weights.items())
        fn_lines = "\n".join(f"- {n}: {f.doc}" + (f" [budget: {f.budget} calls]" if f.budget is not None else "")
                             for n, f in self.fns.items())
        desc = ("Run Python code in a persistent session (variables persist across calls; like a notebook). "
                "numpy is available as `np`" + (" and torch as `torch` (CPU)" if "torch" in self.preload else "")
                + ". The network's weights are preloaded as float32 numpy arrays, both in "
                f"the dict `weights` and as variables: {shapes or '(none)'}. W<k> has shape [in, out] "
                "(row = input unit). The network and environment cannot be run here except through these "
                f"functions:\n{fn_lines}\nstdout and the value of the last expression are returned "
                f"(truncated to {self.max_output_chars} chars). Each call times out after {self.timeout:.0f}s.")
        return [
            {"name": "python", "description": desc + (("\n" + extra_description) if extra_description else ""),
             "input_schema": {"type": "object", "properties": {"code": {"type": "string",
                              "description": "Python source to execute."}}, "required": ["code"],
                              "additionalProperties": False}},
            {"name": "save_weights",
             "description": ("Save network weights written as text. Full format: one line per row, "
                             "`W1 row 3  -0.080 0.255 ...` (row = input unit) and `b1  0.1 -0.2 ...`. Sparse "
                             "edits are applied on top of the currently saved weights: `W2[3,17] = 1.25`, "
                             "`b1[4] = -0.5`, or a full replacement row `W2 row 3 <values>`."),
             "input_schema": {"type": "object", "properties": {"text": {"type": "string",
                              "description": "Weights or edits in the format above."}}, "required": ["text"],
                              "additionalProperties": False}},
        ]

    def handlers(self) -> dict[str, Callable[[dict[str, Any]], str]]:
        def save_text(inp: dict[str, Any]) -> str:
            new = parse_weights(inp["text"], base=self.weights, strict=True)
            changed = [k for k in new if not np.array_equal(new[k], self.weights[k])]
            return self.save_weights({k: new[k] for k in changed}) if changed else "no changes"
        return {"python": lambda inp: self.run(inp["code"]), "save_weights": save_text}


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    head, tail = int(n * 0.6), int(n * 0.4)
    return f"{s[:head]}\n... [{len(s) - head - tail} chars truncated] ...\n{s[-tail:]}"


def _summ(x: Any, depth: int = 0) -> Any:
    """Compact JSON-able summary of call args/results for the call log."""
    if isinstance(x, np.ndarray):
        return {"ndarray": list(x.shape), "dtype": str(x.dtype), **({"value": x.tolist()} if x.size <= 16 else {})}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x if not isinstance(x, str) or len(x) < 500 else x[:500] + "..."
    if isinstance(x, Mapping) and depth < 2:
        return {str(k): _summ(v, depth + 1) for k, v in list(x.items())[:50]}
    if isinstance(x, (list, tuple)) and depth < 2:
        return [_summ(v, depth + 1) for v in x[:50]]
    return repr(x)[:300]
