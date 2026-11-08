"""MLP weights <-> text, in the PLAN.md format.

    W1 row 3  -0.080 0.255 -0.015 ...     (one line per INPUT unit; W printed as [in, out])
    b1  0.012 -0.300 ...

`params` below means an ordered dict {"W1": [in, out], "b1": [out], "W2": ...} of numpy arrays.
The parser also accepts sparse edits on top of a base network:

    W2[3,17] = 1.25        W2[3][17] = 1.25        b1[4] = -0.5
    W2 row 3  <values>     W2[3] = <values>        (replace a whole row)
    b2  <values>                                    (replace a whole bias)
"""
from __future__ import annotations

import re
import warnings
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(NUM)
_ELEM_RE = re.compile(rf"^([Wb]\d+)\s*\[\s*(\d+)\s*(?:(?:,|\]\s*\[)\s*(\d+)\s*)?\]\s*(?::=|=|:)\s*({NUM})\s*$")
_ROW_RE = re.compile(r"^(W\d+)\s*(?:row\s*(\d+)|\[\s*(\d+)\s*\])\s*[:=]?\s*(.*)$", re.I)
_BIAS_RE = re.compile(r"^(b\d+)\s*[:=]?\s*(.*)$")
_NUMS_RE = re.compile(rf"[\s,;\[\]()]*(?:{NUM}[\s,;\[\]()]*)+")
_BULLET_RE = re.compile(r"^(?:[-*>]+\s+)+(?=[A-Za-z])")
_COMMENT_RE = re.compile(r"\s+(?:#|//)")
_QUANT_RE = re.compile(rf"quantized:.*?integer\s*\*\s*({NUM})", re.I)


# ----------------------------------------------------------------------------- state_dict <-> params

def layer_keys(state_dict: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """[(weight_key, bias_key_or_None), ...] for each 2-D weight, in state_dict order."""
    out = []
    for k, v in state_dict.items():
        if k.endswith("weight") and len(tuple(v.shape)) == 2:
            bk = k[: -len("weight")] + "bias"
            out.append((k, bk if bk in state_dict else None))
    return out


def params_from_state_dict(state_dict: Mapping[str, Any]) -> "OrderedDict[str, np.ndarray]":
    """torch Linear weights ([out, in]) -> {"W1": [in, out], "b1": [out], ...} float32 numpy."""
    p: OrderedDict[str, np.ndarray] = OrderedDict()
    for i, (wk, bk) in enumerate(layer_keys(state_dict), 1):
        w = _np(state_dict[wk])
        p[f"W{i}"] = w.T.copy()
        p[f"b{i}"] = _np(state_dict[bk]) if bk else np.zeros(w.shape[0], np.float32)
    return p


def params_to_state_dict(params: Mapping[str, np.ndarray], template: Mapping[str, Any]) -> dict[str, Any]:
    """Write params back under the template's key names (torch tensors if the template is torch)."""
    sd = dict(template)
    for i, (wk, bk) in enumerate(layer_keys(template), 1):
        w = np.asarray(params[f"W{i}"], np.float32).T
        if tuple(w.shape) != tuple(template[wk].shape):
            raise ValueError(f"W{i}: shape {w.T.shape} does not match template {tuple(template[wk].shape)[::-1]}")
        sd[wk] = _like(w, template[wk])
        if bk:
            sd[bk] = _like(np.asarray(params[f"b{i}"], np.float32), template[bk])
    return sd


def arch_of(params: Mapping[str, np.ndarray]) -> list[int]:
    ws = [v for k, v in params.items() if k.startswith("W")]
    return [ws[0].shape[0]] + [w.shape[1] for w in ws]


def _np(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def _like(a: np.ndarray, ref: Any) -> Any:
    if hasattr(ref, "detach"):
        import torch
        return torch.from_numpy(np.ascontiguousarray(a)).to(ref.dtype)
    return a


# ----------------------------------------------------------------------------- params -> text

def weights_to_text(net: Mapping[str, Any], *, decimals: int = 3, quantize: float | None = None,
                    header: bool = True, activation: str | None = None) -> str:
    """Render a state_dict or params dict as text.

    quantize=s prints integers round(w / s) instead of floats (header documents the scale).
    """
    params = net if any(k.startswith("W") for k in net) else params_from_state_dict(net)
    fmt = (lambda a: " ".join(str(int(x)) for x in np.rint(a / quantize) + 0.0)) if quantize else \
          (lambda a: " ".join(f"{x:.{decimals}f}" for x in np.round(a, decimals) + 0.0))
    lines = []
    if header:
        arch = arch_of(params)
        lines.append(f"# MLP {' -> '.join(map(str, arch))}" + (f", {activation} activations" if activation else "")
                     + f", {sum(v.size for v in params.values())} weights")
        lines.append("# W<k> row i = weights from input unit i of layer k to every output unit; b<k> = bias of layer k")
        if quantize:
            lines.append(f"# quantized: value = integer * {quantize:g}")
    for name, a in params.items():
        a = np.asarray(a, np.float64)
        if a.ndim == 2:
            lines += [f"{name} row {i}  {fmt(row)}" for i, row in enumerate(a)]
        else:
            lines.append(f"{name}  {fmt(a)}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- text -> params

def parse_weights(text: str, *, base: Mapping[str, np.ndarray] | None = None,
                  shapes: Mapping[str, tuple[int, ...]] | None = None, scale: float | None = None,
                  strict: bool = True) -> "OrderedDict[str, np.ndarray]":
    """Parse full weights or sparse edits. Tolerates markdown, commas, brackets, unicode minus, wrapped rows.

    base:   start from a copy of these params and apply whatever the text specifies (edits).
    shapes: expected shapes when there is no base (otherwise inferred from the text).
    scale:  multiply parsed values (quantized text); auto-detected from a "# quantized:" header.
    strict: raise on wrong row lengths / missing rows; otherwise pad/truncate with a warning.
    """
    text = text.replace("−", "-").replace("–", "-").replace("**", "").replace("`", "")
    if scale is None:
        m = _QUANT_RE.search(text)
        scale = float(m.group(1)) if m else 1.0
    rows: dict[str, dict[int, list[float]]] = {}
    elems: list[tuple[str, int, int | None, float]] = []
    biases: dict[str, list[float]] = {}
    last: list[float] | None = None
    for raw in text.splitlines():
        line = _BULLET_RE.sub("", raw.strip())
        if not line.startswith("#"):  # drop trailing "# ..." / "// ..." comments, which otherwise void the line
            line = _COMMENT_RE.split(line, 1)[0].rstrip()
        if not line or line.startswith("#"):
            last = None
            continue
        if m := _ELEM_RE.match(line):
            name, i, j, v = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
            if name.startswith("W") and j is None:      # W2[3] = 0.5  -> one-value row
                rows.setdefault(name, {})[i] = last = [v]
            else:
                elems.append((name, i, int(j) if j is not None else None, v))
                last = None
        elif (m := _ROW_RE.match(line)) and (m.group(2) or m.group(3)) and _NUMS_RE.fullmatch(m.group(4)):
            idx = int(m.group(2) or m.group(3))
            rows.setdefault(m.group(1), {})[idx] = last = [float(x) for x in _NUM_RE.findall(m.group(4))]
        elif (m := _BIAS_RE.match(line)) and _NUMS_RE.fullmatch(m.group(2)):
            biases[m.group(1)] = last = [float(x) for x in _NUM_RE.findall(m.group(2))]
        elif last is not None and _NUMS_RE.fullmatch(line):
            last.extend(float(x) for x in _NUM_RE.findall(line))     # wrapped continuation line
        else:
            last = None
    out: OrderedDict[str, np.ndarray] = OrderedDict((k, np.array(v, np.float32)) for k, v in (base or {}).items())
    names = sorted(set(rows) | set(biases) | {e[0] for e in elems} | set(out), key=_layer_order)
    for name in names:
        shape = tuple(out[name].shape) if name in out else (tuple(shapes[name]) if shapes and name in shapes
                                                             else _infer_shape(name, rows, biases))
        if name not in out:
            if name.startswith("W") and len(rows.get(name, {})) < shape[0]:
                missing = sorted(set(range(shape[0])) - set(rows.get(name, {})))
                _complain(strict, f"{name}: missing rows {missing[:10]}{'...' if len(missing) > 10 else ''} (zero-filled)")
            if name.startswith("b") and name not in biases and not any(e[0] == name for e in elems):
                _complain(strict, f"{name}: missing (zero-filled)")
            out[name] = np.zeros(shape, np.float32)
        a = out[name]
        for i, vals in rows.get(name, {}).items():
            if i >= a.shape[0]:
                _complain(strict, f"{name} row {i}: index out of range ({a.shape[0]} rows)")
                continue
            a[i] = _fit(np.array(vals) * scale, a[i], strict, f"{name} row {i}")
        if name in biases:
            a[:] = _fit(np.array(biases[name]) * scale, a, strict, name)
    for name, i, j, v in elems:
        a = out[name]
        try:
            if a.ndim == 2 and j is not None:
                a[i, j] = v * scale
            elif a.ndim == 1 and j is None:
                a[i] = v * scale
            else:
                raise IndexError
        except IndexError:
            _complain(strict, f"{name}[{i}{',' + str(j) if j is not None else ''}]: bad index for shape {a.shape}")
    return out


def diff_params(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], tol: float = 5e-4) -> dict[str, Any]:
    """Fraction of weights changed (|a-b| > tol), overall and per tensor."""
    per = {k: int((np.abs(np.asarray(a[k]) - np.asarray(b[k])) > tol).sum()) for k in a}
    n = sum(np.asarray(v).size for v in a.values())
    return {"changed": sum(per.values()), "total": n, "frac": sum(per.values()) / n, "per_tensor": per}


def _layer_order(name: str) -> tuple[int, int]:
    return int(name[1:]), 0 if name[0] == "W" else 1


def _infer_shape(name: str, rows: dict, biases: dict) -> tuple[int, ...]:
    if name.startswith("W"):
        r = rows.get(name, {})
        if not r:
            raise ValueError(f"{name}: cannot infer shape (no rows and no base/shapes given)")
        lens = [len(v) for v in r.values()]
        return max(r) + 1, max(set(lens), key=lens.count)
    if name in biases:
        return (len(biases[name]),)
    w = rows.get("W" + name[1:], {})
    if w:
        return (_infer_shape("W" + name[1:], rows, biases)[1],)
    raise ValueError(f"{name}: cannot infer shape")


def _fit(vals: np.ndarray, ref: np.ndarray, strict: bool, what: str) -> np.ndarray:
    if vals.size == ref.size:
        return vals
    _complain(strict, f"{what}: {vals.size} values, expected {ref.size}")
    out = np.zeros(ref.size, np.float32)
    out[: min(vals.size, ref.size)] = vals[: ref.size]
    return out


def _complain(strict: bool, msg: str) -> None:
    if strict:
        raise ValueError(msg)
    warnings.warn(msg)


# ----------------------------------------------------------------------------- token counts

def estimate_tokens(text: str) -> int:
    """Offline estimate for weight text, calibrated against count_tokens on claude-opus-5-5 (~1-2%):
    a 3-decimal float like -0.080 costs ~4.55 tokens, a small integer ~2.5. Use count_tokens for prose."""
    floats = len(re.findall(r"-?\d*\.\d+", text))
    ints = len(re.findall(r"(?<![\w.])-?\d+(?![\d.])", text))
    rest = len(_NUM_RE.sub("", text))
    return int(4.55 * floats + 2.5 * ints + 0.07 * rest) + 1


def count_tokens(text: str, *, system: str | None = None, model: str | None = None) -> int:
    """Exact input-token count via the API's count_tokens endpoint (free, needs a key)."""
    from harness.claude_client import MODEL, get_client
    kw: dict[str, Any] = {"system": system} if system else {}
    return get_client().messages.count_tokens(model=model or MODEL, messages=[{"role": "user", "content": text}],
                                              **kw).input_tokens
