"""Mac-side client for the Exp 8 GPU service (gpu-node, SLURM job `subham_wsp_service`, 127.0.0.1:8800).

    from experiments.exp8.remote import Remote
    r = Remote()                                   # opens the SSH tunnel if needed and checks /health
    eid, rep = r.upload_edit({"model.language_model.layers.15.mlp.down_proj.weight": W})   # numpy/torch, full tensors
    res = r.practice(edit_id=eid, pool="mixed", n=64)   # original vs edited, same problems and seeds
    res["original"]["accuracy"], res["edited"]["accuracy"], res["paired"]
    job = r.eval(edit_id=eid, label="budget50_final")   # async full benchmarks; r.wait_eval(job["id"])

    venv/bin/python experiments/exp8/remote.py health      # tunnel check
    venv/bin/python experiments/exp8/remote.py smoke       # 4-problem practice call on the base weights

Budget counting is NOT done by the service; the Mac harness counts calls. Retries are safe: every submission
carries a request_id, and the service returns the existing job for a repeated request_id.
Only the standard library plus numpy (and torch/safetensors for bf16 uploads) is used.
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVER = "user@gpu-server"
PORT = 8800


class ServiceError(RuntimeError):
    pass


class Remote:
    def __init__(self, port: int = PORT, server: str = SERVER, auto_tunnel: bool = True, retries: int = 6,
                 verbose: bool = True):
        self.port, self.server, self.retries, self.verbose = port, server, retries, verbose
        self.base = f"http://127.0.0.1:{port}"
        self.tunnel: subprocess.Popen | None = None
        if auto_tunnel:
            self.ensure_tunnel()

    # ------------------------------------------------------------------ transport
    def _log(self, *a):
        if self.verbose:
            print("[remote]", *a, file=sys.stderr, flush=True)

    def _port_open(self) -> bool:
        with socket.socket() as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def ensure_tunnel(self, wait_s: float = 20.0) -> dict:
        """Make sure http://127.0.0.1:<port>/health answers; start `ssh -N -L` if nothing listens locally."""
        if os.environ.get("WSP_EXTERNAL_TUNNEL"):
            # The tunnel is owned by experiments/exp8/keep_tunnel.sh (self-reconnecting). Never spawn or kill one here:
            # after VPN drops the self-managed tunnel raced its own /health check and looped for hours (playbook 24a).
            t = time.time()
            while True:
                try:
                    h = self._req("GET", "/health", timeout=15, retries=0)
                    if h.get("state") == "ready":
                        return h
                except Exception as e:  # noqa: BLE001
                    if time.time() - t > 30:
                        self._log(f"waiting for external tunnel/service ({str(e)[:120]})")
                        t = time.time()
                time.sleep(5)
        if not self._port_open():
            self._log(f"starting tunnel: ssh -N -L {self.port}:localhost:{self.port} {self.server}")
            self.tunnel = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                 "-o", "ServerAliveCountMax=4", "-N", "-L", f"{self.port}:localhost:{self.port}", self.server],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            t = time.time()
            while time.time() - t < wait_s and not self._port_open():
                if self.tunnel.poll() is not None:
                    raise ServiceError(f"ssh tunnel exited: {self.tunnel.stderr.read().decode()[-500:]}")
                time.sleep(0.3)
        try:
            h = self._req("GET", "/health", timeout=15, retries=2)
        except Exception as e:  # noqa: BLE001
            raise ServiceError(f"tunnel is up but the service does not answer /health ({e}). Is the SLURM job "
                               "running? ssh to the server and run exp8/scripts/service_ctl.sh status") from e
        if h.get("state") != "ready":
            raise ServiceError(f"service not ready: {h}")
        return h

    def close(self):
        if self.tunnel and self.tunnel.poll() is None:
            self.tunnel.terminate()

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 120, retries: int | None = None):
        data = json.dumps(body).encode() if body is not None else None
        n = self.retries if retries is None else retries
        last = None
        for attempt in range(n + 1):
            try:
                rq = urllib.request.Request(self.base + path, data=data, method=method,
                                            headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:2000]
                if 400 <= e.code < 500:      # bad request: do not retry
                    raise ServiceError(f"HTTP {e.code} {path}: {msg}") from None
                last = ServiceError(f"HTTP {e.code} {path}: {msg}")
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as e:
                last = e
                # external tunnel mode: never call ensure_tunnel from here (it calls _req -> unbounded recursion)
                if not self._port_open() and not os.environ.get("WSP_EXTERNAL_TUNNEL"):
                    try:
                        self.ensure_tunnel()
                    except Exception as e2:  # noqa: BLE001
                        last = e2
            if attempt < n:
                wait = min(60, 2 ** attempt)
                self._log(f"{method} {path} failed ({last}); retry {attempt + 1}/{n} in {wait}s")
                time.sleep(wait)
        raise ServiceError(f"{method} {path} failed after {n + 1} attempts: {last}")

    # ------------------------------------------------------------------ API
    def health(self) -> dict:
        return self._req("GET", "/health", timeout=15)

    def log(self, n: int = 20) -> list:
        return self._req("GET", f"/log?n={n}")

    @staticmethod
    def encode_tensors(tensors: dict) -> dict:
        """{name: numpy array | torch tensor} -> {"format", "data"} (bf16 safetensors if torch is available)."""
        try:
            import torch
            from safetensors.torch import save
            sd = {}
            for k, v in tensors.items():
                t = v if isinstance(v, torch.Tensor) else torch.from_numpy(__import__("numpy").ascontiguousarray(v))
                sd[k] = t.detach().to(torch.bfloat16).contiguous().cpu()
            return {"format": "safetensors", "data": base64.b64encode(save(sd)).decode()}
        except ImportError:
            import numpy as np
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()})
            return {"format": "npz", "data": base64.b64encode(buf.getvalue()).decode()}

    def upload_edit(self, tensors: dict | str | Path, label: str = "") -> tuple[str, dict]:
        """Upload changed tensors (full tensors, HF names; a dict or a .npz/.safetensors path) -> (edit_id, report)."""
        if isinstance(tensors, (str, Path)):
            p = Path(tensors)
            enc = {"format": "npz" if p.suffix == ".npz" else "safetensors",
                   "data": base64.b64encode(p.read_bytes()).decode()}
        else:
            enc = self.encode_tensors(tensors)
        r = self._req("POST", "/edit", {"tensors": enc, "label": label}, timeout=600)
        return r["edit_id"], r["report"]

    def _wait_job(self, path: str, timeout: float, poll: float = 5.0) -> dict:
        t = time.time()
        while True:
            v = self._req("GET", path, timeout=60)
            if v.get("status") == "done":
                return v["result"]
            if v.get("status") == "error":
                raise ServiceError(f"job failed: {v.get('error')}\n{v.get('traceback', '')[-2000:]}")
            if time.time() - t > timeout:
                raise ServiceError(f"timeout waiting for {path} (status {v.get('status')})")
            time.sleep(poll)

    def practice(self, edit_id: str = "base", tensors: dict | None = None, pool: str = "mixed",
                 ids: list[str] | None = None, n: int = 64, seed: int | None = None, max_tokens: int = 32768,
                 texts: str = "truncated", head_chars: int = 1500, tail_chars: int = 1500,
                 activations: bool = False, activation_tokens: int = 2048, gen_seed: int = 0,
                 timeout: float = 4 * 3600, request_id: str | None = None) -> dict:
        """Generate for the chosen problems with BOTH the original and the edited weights (same seeds/settings).
        pool: gsm8k_train | math_train | mixed (half/half); ids: explicit problem ids, else n random (seed)."""
        problems = {"pool": pool}
        if ids:
            problems["ids"] = list(ids)
        else:
            problems["n"] = n
            if seed is not None:
                problems["seed"] = seed
        body = {"request_id": request_id or uuid.uuid4().hex, "problems": problems, "max_tokens": max_tokens,
                "gen_seed": gen_seed, "wait": False,
                "return": {"texts": texts, "head_chars": head_chars, "tail_chars": tail_chars,
                           "activations": activations, "activation_tokens": activation_tokens}}
        if tensors is not None:
            body["tensors"] = self.encode_tensors(tensors)
        else:
            body["edit_id"] = edit_id
        job = self._req("POST", "/practice", body, timeout=600)
        return self._wait_job(f"/result/{job['id']}", timeout)

    def eval(self, edit_id: str = "base", bench=("math500", "gsm8k", "ifeval"), label: str | None = None,
             samples: int | dict = 1, limit: int | None = None, wait: bool = False, timeout: float = 12 * 3600,
             overwrite: bool = False) -> dict:
        """Queue a full benchmark run (writes exp8/results/<label or edit_id>.json on the server)."""
        body = {"request_id": uuid.uuid4().hex, "edit_id": edit_id, "bench": list(bench), "samples": samples,
                "overwrite": overwrite}
        if label:
            body["label"] = label
        if limit:
            body["limit"] = limit
        job = self._req("POST", "/eval", body, timeout=120)
        return self.wait_eval(job["id"], timeout) if wait else job

    def eval_status(self, job_id: str) -> dict:
        return self._req("GET", f"/eval/{job_id}")

    def wait_eval(self, job_id: str, timeout: float = 12 * 3600) -> dict:
        return self._wait_job(f"/eval/{job_id}", timeout, poll=30)

    @staticmethod
    def activations_np(res: dict) -> dict:
        """practice(..., activations=True) result -> {side: {group: {"mlp_out": [24,1024], "resid": [24,1024], ...}}}"""
        import numpy as np
        out = {}
        for side in ("original", "edited"):
            out[side] = {}
            for g in ("hit", "finished"):
                d = res["activations"][side][g]
                out[side][g] = {"mlp_out": np.array(d["mlp_out"], dtype=np.float32),
                                "resid": np.array(d["resid"], dtype=np.float32),
                                "n_seqs": d["n_seqs"], "n_tokens": d["n_tokens"]}
        return out


def _main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "health"
    r = Remote()
    if cmd == "health":
        print(json.dumps(r.health(), indent=1))
    elif cmd == "smoke":
        t = time.time()
        res = r.practice(n=4, pool="gsm8k_train", seed=0, max_tokens=1024)
        print(json.dumps({k: res[k] for k in ("call_id", "edit_id", "original", "edited", "paired", "timing")}, indent=1))
        print(f"round trip {time.time() - t:.1f}s")
    elif cmd == "log":
        print(json.dumps(r.log(int(sys.argv[2]) if len(sys.argv) > 2 else 10), indent=1))
    r.close()


if __name__ == "__main__":
    _main()
