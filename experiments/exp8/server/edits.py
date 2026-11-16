"""Weight edits for Exp 8: format, canonicalization, edited-model directories, verification.

Edit format (what the Mac sandbox saves): ONLY the changed tensors, keyed by HF parameter name, as either
  - `.safetensors` (any float dtype), or
  - `.npz` (numpy float32/float16; bf16 is not a numpy dtype, so send float32),
full tensors (not deltas), same shape as the base tensor. Short names are accepted: `layers.15.mlp.down_proj.weight`
means `model.language_model.layers.15.mlp.down_proj.weight`.

`canonicalize()` casts each tensor to the base tensor's dtype (bf16 for almost all), drops tensors that are
bit-identical to the base after the cast, and hashes the result: edit_id = sha256(names + bf16 bytes)[:16]. The
stored edit is `exp8/edits/<edit_id>/delta.safetensors` + `report.json`.

    venv/bin/python exp8/edits.py build <edit file or edit_id>   # -> exp8/models_edited/<edit_id>/ (+ verify)
    venv/bin/python exp8/edits.py verify <edit_id>
    venv/bin/python exp8/edits.py selftest
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp8.common import EXP8, FAMILY, LM_PREFIXES, MODELS  # noqa: E402

EDITS = EXP8 / "edits"
EDITED_MODELS = EXP8 / "models_edited"
LM_PREFIX = LM_PREFIXES[FAMILY]


class Base:
    """Lazy access to the base model's tensors (memory-mapped safetensors)."""

    def __init__(self, model_dir: Path = MODELS[FAMILY]):
        self.dir = Path(model_dir)
        self.files = sorted(self.dir.glob("*.safetensors"))
        self.where = {}
        self.meta = {}
        for f in self.files:
            with safe_open(str(f), "pt") as s:
                for k in s.keys():
                    sl = s.get_slice(k)
                    self.where[k] = f
                    self.meta[k] = (tuple(sl.get_shape()), sl.get_dtype())

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.where[name]), "pt") as s:
            return s.get_tensor(name)

    def n_params(self, prefix: str = LM_PREFIX) -> int:
        return int(sum(np.prod(sh) for k, (sh, _) in self.meta.items() if k.startswith(prefix)))


_BASE: Base | None = None


def base() -> Base:
    global _BASE
    if _BASE is None:
        _BASE = Base()
    return _BASE


def _resolve(name: str, b: Base) -> str:
    if name in b.meta:
        return name
    for cand in (LM_PREFIX + name, LM_PREFIX + name.removeprefix("model."),
                 LM_PREFIX + name.removeprefix("language_model."), "model." + name):
        if cand in b.meta:
            return cand
    raise KeyError(f"unknown parameter name: {name}")


def read_edit_bytes(data: bytes, fmt: str) -> dict[str, torch.Tensor]:
    if fmt == "safetensors":
        from safetensors.torch import load
        return dict(load(data))
    if fmt == "npz":
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files}
    raise ValueError(f"format must be safetensors or npz, got {fmt}")


def read_edit_file(path: str | Path) -> dict[str, torch.Tensor]:
    p = Path(path)
    return read_edit_bytes(p.read_bytes(), "npz" if p.suffix == ".npz" else "safetensors")


def canonicalize(tensors: dict[str, torch.Tensor], label: str = "") -> tuple[str, dict]:
    """Validate, cast to base dtype, drop no-ops, hash, and store. Returns (edit_id, report)."""
    b = base()
    out, report_t = {}, {}
    for raw_name in sorted(tensors):
        name = _resolve(raw_name, b)
        t = tensors[raw_name]
        shape, _ = b.meta[name]
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)} != base {shape}")
        if not t.is_floating_point():
            raise ValueError(f"{name}: dtype {t.dtype} is not floating point")
        if not torch.isfinite(t.float()).all():
            raise ValueError(f"{name}: contains NaN/inf")
        bt = b.get(name)
        tc = t.to(bt.dtype).contiguous()
        diff = tc.view(torch.int16 if tc.element_size() == 2 else torch.int32) != \
            bt.view(torch.int16 if bt.element_size() == 2 else torch.int32)
        n_changed = int(diff.sum())
        # elements that differed in the submitted precision but collapse to the base value after the cast
        lost = int(((t.float() != bt.float()) & ~diff).sum())
        if n_changed == 0:
            report_t[name] = {"n_changed": 0, "lost_to_cast": lost, "dropped": True}
            continue
        d = (tc.float() - bt.float())
        rows_changed = int(diff.reshape(diff.shape[0], -1).any(1).sum()) if diff.dim() > 0 else 1
        cols_changed = int(diff.any(0).sum()) if diff.dim() == 2 else None
        report_t[name] = {"n_changed": n_changed, "numel": tc.numel(), "lost_to_cast": lost,
                          "rows_changed": rows_changed, "cols_changed": cols_changed,
                          "delta_fro": round(float(d.norm()), 6), "base_fro": round(float(bt.float().norm()), 6),
                          "dtype": str(bt.dtype).replace("torch.", "")}
        out[name] = tc
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode())
        h.update(out[k].view(torch.uint8).numpy().tobytes() if out[k].dtype != torch.bfloat16
                 else out[k].view(torch.int16).numpy().tobytes())
    edit_id = h.hexdigest()[:16] if out else "base"
    n_changed = sum(v["n_changed"] for v in report_t.values())
    report = {"edit_id": edit_id, "label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "n_tensors_submitted": len(tensors), "n_tensors_changed": len(out), "n_weights_changed": n_changed,
              "n_lm_params": b.n_params(), "frac_lm_params_changed": n_changed / b.n_params(),
              "lost_to_cast_total": sum(v["lost_to_cast"] for v in report_t.values()), "tensors": report_t}
    if out:
        d = EDITS / edit_id
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "delta.safetensors").exists():
            save_file(out, str(d / "tmp.safetensors"), metadata={"edit_id": edit_id})
            os.replace(d / "tmp.safetensors", d / "delta.safetensors")
        (d / "report.json").write_text(json.dumps(report, indent=1))
    return edit_id, report


def load_edit(edit_id: str) -> dict[str, torch.Tensor]:
    if edit_id == "base":
        return {}
    return load_file(str(EDITS / edit_id / "delta.safetensors"))


def build_edited_dir(edit_id: str, verify_after: bool = True) -> Path:
    """Base model dir with the changed tensors replaced -> exp8/models_edited/<edit_id>/ (for vLLM / HF loading).
    Non-weight files are symlinked; the safetensors shard is rewritten with identical names and dtypes."""
    if edit_id == "base":
        return MODELS[FAMILY]
    out = EDITED_MODELS / edit_id
    if (out / "VERIFIED.json").exists():
        return out
    b = base()
    delta = load_edit(edit_id)
    tmp = EDITED_MODELS / f".{edit_id}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in b.dir.iterdir():
        if f.suffix != ".safetensors" and f.name != ".cache":
            (tmp / f.name).symlink_to(f.resolve())
    for f in b.files:
        with safe_open(str(f), "pt") as s:
            meta = s.metadata() or {}
            sd = {k: (delta[k] if k in delta else s.get_tensor(k)) for k in s.keys()}
        save_file(sd, str(tmp / f.name), metadata=meta)
        del sd
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    if verify_after:
        verify(edit_id)
    return out


def verify(edit_id: str) -> dict:
    """Every tensor not in the edit must be bit-identical to the base; edited tensors must equal the delta."""
    b = base()
    out = EDITED_MODELS / edit_id
    delta = load_edit(edit_id)
    n_ident, n_edit_ok, bad, n_changed = 0, 0, [], 0
    for f in b.files:
        with safe_open(str(f), "pt") as sb, safe_open(str(out / f.name), "pt") as se:
            if set(sb.keys()) != set(se.keys()):
                bad.append(f"key set differs in {f.name}")
            for k in sb.keys():
                tb, te = sb.get_tensor(k), se.get_tensor(k)
                same = tb.dtype == te.dtype and tb.shape == te.shape and torch.equal(
                    tb.view(torch.uint8) if tb.dtype != torch.bfloat16 else tb.view(torch.int16),
                    te.view(torch.uint8) if te.dtype != torch.bfloat16 else te.view(torch.int16))
                if k in delta:
                    ok = torch.equal(te.view(torch.int16) if te.dtype == torch.bfloat16 else te,
                                     delta[k].view(torch.int16) if delta[k].dtype == torch.bfloat16 else delta[k])
                    n_edit_ok += ok
                    n_changed += int((te.float() != tb.float()).sum())
                    if not ok:
                        bad.append(f"edited tensor differs from delta: {k}")
                elif same:
                    n_ident += 1
                else:
                    bad.append(f"untouched tensor differs: {k}")
    rep = {"edit_id": edit_id, "untouched_bit_identical": n_ident, "edited_tensors_ok": n_edit_ok,
           "edited_tensors": len(delta), "n_weights_changed": n_changed, "problems": bad,
           "ok": not bad and n_edit_ok == len(delta), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if rep["ok"]:
        (out / "VERIFIED.json").write_text(json.dumps(rep, indent=1))
    else:
        raise RuntimeError(f"verification failed: {rep}")
    return rep


def selftest():
    b = base()
    name = LM_PREFIX + "layers.15.mlp.down_proj.weight"
    W = b.get(name).float()
    W2 = W.clone()
    W2[:, :64] -= 0.01                     # change 64 columns
    n3 = name.replace("layers.15.", "layers.3.")
    W3 = b.get(n3).float()
    W3[0, 0] = W3[0, 0] * (1 + 1e-4)       # below bf16 resolution: collapses to the base value, so dropped
    n19 = name.replace("layers.15.", "layers.19.")
    eid, rep = canonicalize({"layers.15.mlp.down_proj.weight": W2, n19: b.get(n19).float(), n3: W3},
                            label="selftest")
    print(json.dumps({k: v for k, v in rep.items() if k != "tensors"}, indent=1))
    assert rep["n_tensors_changed"] == 1, rep
    eid2, _ = canonicalize({name: W2})
    assert eid2 == eid, "hash must be stable"
    # npz round trip
    buf = io.BytesIO()
    np.savez(buf, **{name: W2.numpy()})
    eid3, _ = canonicalize(read_edit_bytes(buf.getvalue(), "npz"))
    assert eid3 == eid
    t = time.time()
    d = build_edited_dir(eid)
    print("built + verified", d, f"{time.time() - t:.1f}s", json.loads((d / "VERIFIED.json").read_text()))
    return eid


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "build":
        arg = sys.argv[2]
        eid = canonicalize(read_edit_file(arg))[0] if Path(arg).exists() else arg
        print(build_edited_dir(eid))
    elif cmd == "verify":
        print(json.dumps(verify(sys.argv[2]), indent=1))
