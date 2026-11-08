"""Child process for harness.sandbox: a persistent exec namespace speaking length-prefixed pickles.

Protocol (parent <-> child, over the child's stdin/stdout):
  parent -> ("init", cfg) | ("exec", code) | ("ret", ok, value) | ("exit",)
  child  -> ("ready",) | ("call", name, args, kwargs) | ("done", output)
Only weights (numpy copies) live here; proxy functions forward to the parent, which owns budgets
and anything that runs the network / environment.
"""
import os
import sys

# Keep CPU use light and set threads before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SANDBOX_THREADS", "2"))

import ast  # noqa: E402
import contextlib  # noqa: E402
import importlib.abc  # noqa: E402
import io  # noqa: E402
import linecache  # noqa: E402
import pickle  # noqa: E402
import struct  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

_IN = os.fdopen(os.dup(0), "rb", buffering=0)
_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)                      # stray C-level / print output goes to stderr, never into the protocol
sys.stdout = sys.__stdout__ = os.fdopen(1, "w", buffering=1)


class BudgetExceeded(RuntimeError):
    """Raised when a budgeted function (forward_pass / rollout) has no calls left."""


def _send(obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _OUT.write(struct.pack("<Q", len(data)) + data)


def _read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = _IN.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def _recv():
    (n,) = struct.unpack("<Q", _read_exact(8))
    return pickle.loads(_read_exact(n))


def _make_stub(name: str, doc: str):
    def stub(*args, **kwargs):
        _send(("call", name, args, kwargs))
        kind, ok, value = _recv()
        if ok:
            return value
        if value.startswith("BudgetExceeded"):
            raise BudgetExceeded(value.removeprefix("BudgetExceeded: "))
        raise RuntimeError(value)
    stub.__name__ = name
    stub.__doc__ = doc
    return stub


class _Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, blocked):
        self.blocked = set(blocked)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"import of '{fullname}' is not allowed in this sandbox")
        return None


def _deny(*_a, **_k):
    raise PermissionError("process creation is not allowed in this sandbox")


def _lockdown(blocked) -> None:
    sys.meta_path.insert(0, _Blocker(blocked))
    for name in list(sys.modules):
        if name.split(".")[0] in blocked and name.split(".")[0] != "harness":
            del sys.modules[name]
    for fn in ("system", "popen", "fork", "forkpty", "execv", "execve", "execvp", "execvpe", "execl", "execle",
               "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle",
               "posix_spawn", "posix_spawnp", "kill", "killpg"):
        if hasattr(os, fn):
            setattr(os, fn, _deny)
    root = os.environ.get("SANDBOX_REPO_ROOT", "")
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != root]


def _run(code: str, ns: dict) -> str:
    buf = io.StringIO()
    linecache.cache["<sandbox>"] = (len(code), None, code.splitlines(True), "<sandbox>")
    try:
        tree = ast.parse(code, "<sandbox>", "exec")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, "<sandbox>", "exec"), ns)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<sandbox>", "eval"), ns)
                if val is not None:
                    print(repr(val))
    except KeyboardInterrupt:
        buf.write("\nTimeoutError: execution exceeded the time limit and was interrupted "
                  "(namespace kept).\n")
    except BaseException as e:  # report everything (incl. SystemExit) to the model, keep the session alive
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<sandbox>"]
        buf.write("Traceback (most recent call last):\n" + "".join(traceback.format_list(frames))
                  + "".join(traceback.format_exception_only(type(e), e)))
    return buf.getvalue()


def main() -> None:
    kind, cfg = _recv()
    assert kind == "init"
    ns: dict = {"__name__": "__sandbox__", "np": np, "numpy": np, "BudgetExceeded": BudgetExceeded}
    ns["weights"] = {k: np.array(v) for k, v in cfg["weights"].items()}
    ns.update({k: v for k, v in ns["weights"].items()})
    for name, doc in cfg["functions"].items():
        ns[name] = _make_stub(name, doc)
    for mod in cfg.get("preload", []):          # e.g. torch: imported before lockdown, CPU threads capped
        ns[mod] = __import__(mod)
        if mod == "torch":
            ns[mod].set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    _lockdown(cfg["blocked"])
    if cfg.get("preamble"):
        _run(cfg["preamble"], ns)
    _send(("ready",))
    while True:
        try:
            msg = _recv()
        except (EOFError, KeyboardInterrupt):
            return
        if msg[0] == "exit":
            return
        if msg[0] == "exec":
            _send(("done", _run(msg[1], ns)))


if __name__ == "__main__":
    main()
