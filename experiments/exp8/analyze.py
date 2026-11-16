"""Exp 8 analysis: final-eval statistics vs base (paired, pre-registered in GRADING.md), figure, summary, blog data.

    venv/bin/python experiments/exp8/analyze.py fetch <label>      # compact the items on the server and copy here
    venv/bin/python experiments/exp8/analyze.py stats              # table + decisions -> logs/exp8/stats.json
    venv/bin/python experiments/exp8/analyze.py summarize          # figure, logs/exp8/summary.json, docs/data/exp8.json
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOG = REPO / "logs" / "exp8"
FIG = REPO / "docs" / "assets" / "figures"
SERVER = "user@gpu-server"
WSP = "/path/to/wsp"
BENCHES = ["math500", "gsm8k", "ifeval"]
PRETTY = {"math500": "MATH-500", "gsm8k": "GSM8K", "ifeval": "IFEval"}
QWEN_TAGS = ["b10", "b25", "b50"]


def ssh(cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", SERVER, cmd], capture_output=True, text=True,
                          check=True).stdout


def fetch(label: str, family: str = "qwen") -> Path:
    env = f"EXP8_FAMILY={family} " if family != "qwen" else ""
    ssh(f"cd {WSP} && {env}CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python exp8/scripts/compact_items.py "
        f"exp8/results/{label}.items.jsonl.gz")
    for suf in (".json", ".compact.jsonl.gz"):
        subprocess.run(["scp", "-q", f"{SERVER}:{WSP}/exp8/results/{label}{suf}", str(RESULTS / f"{label}{suf}")],
                       check=True)
    return RESULTS / f"{label}.compact.jsonl.gz"


def load(label: str) -> dict:
    """-> {bench: {id: {"correct": [..], "hit": [..], "loop": [..], "n_tokens": [..]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with gzip.open(RESULTS / f"{label}.compact.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            d = out[r["bench"]][r["id"]]
            d["correct"].append(float(bool(r["correct"])))
            d["hit"].append(float(bool(r["hit_limit"])))
            d["loop"].append(float(bool(r["looping"])))
            d["n_tokens"].append(r["n_tokens"])
    return out


def per_item(data: dict, bench: str, key: str, ids) -> np.ndarray:
    return np.array([np.mean(data[bench][i][key]) for i in ids])


def boot_ci(diff: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a: dict, b: dict, bench: str, key: str = "correct") -> dict:
    """b − a over the problems both have; points."""
    ids = sorted(set(a[bench]) & set(b[bench]))
    x, y = per_item(a, bench, key, ids), per_item(b, bench, key, ids)
    d = 100 * (y - x)
    lo, hi = boot_ci(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"n_items": len(ids), "a": round(100 * x.mean(), 2), "b": round(100 * y.mean(), 2),
            "delta": round(d.mean(), 2), "ci95_boot": [round(lo, 2), round(hi, 2)],
            "ci95_t": [round(d.mean() - 1.96 * se, 2), round(d.mean() + 1.96 * se, 2)],
            "excludes_0": bool(lo > 0 or hi < 0)}


def bench_summary(data: dict, bench: str) -> dict:
    rows = [(c, h, l, t) for i in data[bench].values() for c, h, l, t in
            zip(i["correct"], i["hit"], i["loop"], i["n_tokens"])]
    a = np.array(rows, dtype=float)
    fin = a[a[:, 1] == 0]
    return {"n_samples": len(a), "n_items": len(data[bench]), "accuracy": round(100 * a[:, 0].mean(), 2),
            "hit_limit_pct": round(100 * a[:, 1].mean(), 2), "looping_pct": round(100 * a[:, 2].mean(), 2),
            "mean_tokens": round(a[:, 3].mean(), 0),
            "acc_among_finished": round(100 * fin[:, 0].mean(), 2) if len(fin) else None}


def runs() -> dict[str, dict]:
    out = {}
    for d in sorted(LOG.iterdir()) if LOG.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        try:
            r = json.loads((d / "result.json").read_text())
        except (OSError, ValueError):
            continue
        if r.get("run_tag") and r.get("status") not in ("running", "abandoned", "test") and not r.get("excluded"):
            r["_dir"] = d.name
            out[r["run_tag"]] = r            # one run per tag (the last one if repeated)
    return out


def stats() -> dict:
    R = runs()
    res = {"qwen": {}, "gemma": {}}
    dec = {}
    base = load("base_t06")
    res["qwen"]["base"] = {b: bench_summary(base, b) for b in BENCHES if b in base}
    edits = {}
    for tag in QWEN_TAGS:
        lab = (R.get(tag) or {}).get("final_eval_label")
        if lab and (RESULTS / f"{lab}.compact.jsonl.gz").exists():
            edits[tag] = load(lab)
            e = res["qwen"][tag] = {b: bench_summary(edits[tag], b) for b in BENCHES}
            for b in BENCHES:
                e[b]["vs_base"] = paired(base, edits[tag], b)
                e[b]["hit_vs_base"] = paired(base, edits[tag], b, "hit")
                e[b]["loop_vs_base"] = paired(base, edits[tag], b, "loop")
    # matched-temperature (norm-only) controls, GRADING.md 2026-11-14 / 2026-11-15
    CTRL = {"b10": "ctrl_normonly_s0.6_316cb0a863429162", "b25": "ctrl_normonly_s0.6_316cb0a863429162",
            "b50": "ctrl_normonly_s0.68_4ad6ddec49a12c43"}
    res["qwen"]["controls"] = {}
    for tag, lab in CTRL.items():
        if tag not in edits or not (RESULTS / f"{lab}.compact.jsonl.gz").exists():
            continue
        c = load(lab)
        cs = res["qwen"]["controls"].setdefault(lab, {b: bench_summary(c, b) for b in BENCHES})
        for b in BENCHES:
            cs[b]["vs_base"] = paired(base, c, b)
            cs[b]["hit_vs_base"] = paired(base, c, b, "hit")
        e = res["qwen"][tag]
        e["control"] = lab
        for b in BENCHES:
            e[b]["vs_control"] = paired(c, edits[tag], b)
            gain = e[b]["vs_base"]["delta"]
            e[b]["share_reproduced_by_control"] = (round(cs[b]["vs_base"]["delta"] / gain, 3) if abs(gain) > 1e-9
                                                    else None)
        sh = e["math500"]["share_reproduced_by_control"]
        dec[f"{tag}_temperature_verdict"] = ("the gain is a temperature effect" if sh is not None and sh >= 0.8 else
                                             "the edit does more than change temperature" if sh is not None and sh < 0.5
                                             else "partly a temperature effect")
    for t1, t2 in (("b10", "b50"), ("b25", "b50"), ("b10", "b25")):
        if t1 in edits and t2 in edits:
            res["qwen"][f"{t2}_minus_{t1}"] = {b: paired(edits[t1], edits[t2], b) for b in BENCHES}
    # decisions (GRADING.md)
    b50 = res["qwen"].get("b50", {}).get("math500", {}).get("vs_base")
    if b50:
        if b50["delta"] >= 20 and b50["excludes_0"]:
            dec["replication"] = "replicates"
        elif b50["delta"] > 0 and b50["excludes_0"]:
            dec["replication"] = "partial"
        else:
            other = [res["qwen"][t]["math500"]["vs_base"] for t in ("b10", "b25") if t in res["qwen"]]
            dec["replication"] = "partial" if any(o["delta"] >= 20 and o["excludes_0"] for o in other) \
                else "does not replicate"
        h = res["qwen"]["b50"]["math500"]["hit_vs_base"]
        dec["loop_removal_b50"] = bool(h["delta"] <= -20)
        ife = res["qwen"]["b50"]["ifeval"]["vs_base"]
        dec["generalises_ifeval_b50"] = bool(ife["delta"] > 0 and ife["excludes_0"])
        if "b50_minus_b10" in res["qwen"] and "b50_minus_b25" in res["qwen"]:
            x, y = res["qwen"]["b50_minus_b10"]["math500"], res["qwen"]["b50_minus_b25"]["math500"]
            dec["budget_scaling_replicates"] = bool(x["delta"] > 0 and x["excludes_0"] and y["delta"] > 0
                                                    and y["excludes_0"])
    for tag in QWEN_TAGS:
        if tag in res["qwen"]:
            dec[f"{tag}_loop_removal"] = bool(res["qwen"][tag]["math500"]["hit_vs_base"]["delta"] <= -20)
    # gemma
    if (RESULTS / "gemma_base.compact.jsonl.gz").exists():
        gb = load("gemma_base")
        res["gemma"]["base"] = {"math500": bench_summary(gb, "math500")}
        lab = (R.get("gemma-b25") or {}).get("final_eval_label")
        if lab and (RESULTS / f"{lab}.compact.jsonl.gz").exists():
            ge = load(lab)
            res["gemma"]["b25"] = {"math500": {**bench_summary(ge, "math500"),
                                               "vs_base": paired(gb, ge, "math500")}}
            if (RESULTS / "ctrl_gemma_T1.1628.compact.jsonl.gz").exists():
                gc = load("ctrl_gemma_T1.1628")
                res["gemma"]["control_T1.1628"] = {"math500": {**bench_summary(gc, "math500"),
                                                               "vs_base": paired(gb, gc, "math500")}}
                res["gemma"]["b25"]["math500"]["vs_control"] = paired(gc, ge, "math500")
            v = res["gemma"]["b25"]["math500"]["vs_base"]
            dec["gemma_control"] = ("contradicts (edit improved Gemma)" if v["delta"] >= 2 and v["excludes_0"]
                                    else "control replicates (no meaningful gain)")
    res["decisions"] = dec
    (LOG / "stats.json").write_text(json.dumps(res, indent=1))
    return res


def figure(res: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grey, light = "#16171a", "#50545c", "#9aa0a6", "#d5d8dc"
    col = {"math500": "#2f62b8", "gsm8k": "#008a7a", "ifeval": "#8256b8"}
    q = res["qwen"]
    tags = [t for t in QWEN_TAGS if t in q]
    xs = [0] + [{"b10": 1, "b25": 2, "b50": 3}[t] for t in tags]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1]})
    for b in BENCHES:
        ys = [q["base"][b]["accuracy"]] + [q[t][b]["accuracy"] for t in tags]
        lo = [0] + [q[t][b]["accuracy"] - (q["base"][b]["accuracy"] + q[t][b]["vs_base"]["ci95_boot"][0])
                    for t in tags]
        hi = [0] + [(q["base"][b]["accuracy"] + q[t][b]["vs_base"]["ci95_boot"][1]) - q[t][b]["accuracy"]
                    for t in tags]
        ax.axhline(q["base"][b]["accuracy"], color=col[b], lw=0.8, ls=":", alpha=0.7)
        ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], color=col[b], marker="o", ms=5, lw=1.8,
                    capsize=3, label=PRETTY[b])
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                    color=col[b], va="center")
        ax.annotate(f"{ys[0]:.1f}", (0, ys[0]), xytext=(-6, 0), textcoords="offset points", fontsize=7.5,
                    color=col[b], va="center", ha="right")
    # the original post's MATH-500 (GPT-6 "Astra"), for reference
    ax.plot([0, 1, 2, 3], [7.9, 12.3, 14.1, 48.1], color=grey, ls="--", lw=1.2, marker="s", ms=3.5,
            label="MATH-500, original post")
    # norm-only (temperature-equivalent) control of each final edit, MATH-500
    cx, cy = [], []
    for t, x in (("b10", 1), ("b25", 2), ("b50", 3)):
        lab = q.get(t, {}).get("control")
        if lab and lab in q.get("controls", {}):
            cx.append(x)
            cy.append(q["controls"][lab]["math500"]["accuracy"])
    if cx:
        ax.plot(cx, cy, ls="none", marker="o", ms=7, mfc="white", mec=col["math500"], mew=1.5,
                label="MATH-500, final-norm part only")
    ax.set_xticks([0, 1, 2, 3], ["base", "10", "25", "50"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("forward passes Claude could use (64 practice problems each)", fontsize=8, color=muted)
    ax.set_ylabel("accuracy (%)", fontsize=8, color=muted)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax.set_ylim(0, 85)
    ax.set_title("Benchmarks, final saved edit vs base", fontsize=9, color=ink, loc="left")
    hs = [q["base"]["math500"]["hit_limit_pct"]] + [q[t]["math500"]["hit_limit_pct"] for t in tags]
    ax2.bar(xs, hs, color=[grey] + ["#2f62b8"] * len(tags), width=0.6)
    for x, h in zip(xs, hs):
        ax2.text(x, h + 1.5, f"{h:.0f}%", ha="center", fontsize=7.5, color=ink)
    ax2.set_xticks([0, 1, 2, 3], ["base", "10", "25", "50"])
    ax2.set_ylim(0, 105)
    ax2.set_title("MATH-500 answers hitting 32,768 tokens", fontsize=9, color=ink, loc="left")
    ax2.set_xlabel("forward passes", fontsize=8, color=muted)
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(light)
        a.tick_params(colors=muted, labelsize=7.5)
    fig.text(0.01, 0.01, "Qwen3.5-0.8B, temperature 0.6, thinking, 32k tokens; MATH-500 4 samples, GSM8K/IFEval 2. "
             "Error bars: 95% paired bootstrap CI\nof the change vs base. Dotted: base. Hollow: the edit's final-norm "
             "scale alone (= a higher sampling temperature). Grey dashed: MATH-500 in the original post (GPT-6 Astra).", fontsize=6.3,
             color=muted)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / "exp8_budget.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return f"assets/figures/{out.name}"




# ----------------------------------------------------------------------------- what did the edit change?

def decompose(run_dir: str) -> dict:
    """GRADING.md (2026-11-14 addition): final-norm scale s, its residual, every other changed tensor, rank of Δ."""
    import torch
    sys.path.insert(0, str(REPO))
    from experiments.exp8.run import FAM, RUNS, load_base
    d = LOG / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)
    r = json.loads((d / "result.json").read_text())
    fam = RUNS[r["run_tag"]]["family"]
    pre = FAM[fam]["prefix"]
    z = np.load(d / "final_edit.npz")
    names = list(z.files)
    base = load_base(fam, names) if names else {}
    out = {"run": d.name, "tag": r["run_tag"], "n_tensors": len(names), "tensors": {}}
    tot = 0.0
    for k in names:
        w0 = base[k].astype(np.float64)
        w1 = torch.from_numpy(z[k]).to(torch.bfloat16).float().numpy().astype(np.float64)   # what the model holds
        dlt = w1 - w0
        rec = {"shape": list(w0.shape), "n_changed": int((dlt != 0).sum()), "rel_change": float(np.linalg.norm(dlt) /
               max(np.linalg.norm(w0), 1e-12))}
        if k.endswith("norm.weight") and w0.ndim == 1:
            ratio = (1 + w1) / (1 + w0)
            s = float(np.median(ratio))
            rec.update(scale_median=s, scale_rms_dev=float(np.sqrt(np.mean((ratio / s - 1) ** 2))),
                       scale_max_dev=float(np.abs(ratio / s - 1).max()))
        if dlt.ndim == 2 and rec["n_changed"]:
            rows, cols = np.nonzero(np.abs(dlt).sum(1))[0], np.nonzero(np.abs(dlt).sum(0))[0]
            sub = dlt[np.ix_(rows, cols)]
            sv = np.linalg.svd(sub, compute_uv=False)
            rec.update(rows_changed=len(rows), cols_changed=len(cols),
                       sv_share_top1=float(sv[0] ** 2 / (sv ** 2).sum()), sv_top5=[float(x) for x in sv[:5]])
            if len(rows) == w0.shape[0] and sv[0] ** 2 / (sv ** 2).sum() > 0.9:    # rank-1: output direction
                U, S, Vt = np.linalg.svd(sub, full_matrices=False)
                proj = U[:, 0] @ w0[:, cols]          # how much of the original columns lay along that direction
                rec.update(direction_removed_frac=float(np.linalg.norm(U[:, 0] @ dlt[:, cols]) /
                                                        max(np.linalg.norm(proj), 1e-12)))
        out["tensors"][k.removeprefix(pre)] = rec
        tot += rec["n_changed"]
    fn = "norm.weight"
    s = out["tensors"].get(fn, {}).get("scale_median")
    others = [k for k in out["tensors"] if k != fn]
    if s is not None and abs(s - 1) >= 0.01 and not others and out["tensors"][fn]["scale_rms_dev"] < 0.01:
        kind = "pure temperature-like"
    elif s is None or abs(s - 1) < 0.01:
        kind = "structural"
    else:
        kind = "mixed"
    T, P = (0.6, 1.5) if fam == "qwen" else (1.0, 0.0)
    out.update(n_weights_changed=int(tot), final_norm_scale=s, classification=kind,
               matched_control=None if s is None or abs(s - 1) < 0.01 else
               {"temperature": round(T / s, 4), "presence_penalty": round(P / s, 4)})
    (d / "decompose.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- audit (GRADING.md)

import re  # noqa: E402

AUDIT_RX = {
    "random": r"np\.random|torch\.randn|torch\.rand\b|default_rng",
    "softmax_exp": r"softmax|np\.exp\(|torch\.exp\(",
    "nonlinearity": r"silu|sigmoid|gelu|\btanh\(|F\.relu",
    "norm_apply": r"\brms\w*\(|\.pow\(2\)\.mean|\*\*\s*2\)?\.mean\(|rsqrt",
    "rope": r"\brope\b|rotary|np\.cos\(|np\.sin\(",
    "forward_words": r"forward|hidden_?state|simulate|run the (model|layer)",
    "embed_rows_through": r"embed\w*\s*\[\s*(?:\[|ids|tok|idx)[^\n]*@",
    "training": r"\.backward\(|optim\.|requires_grad|autograd|lstsq|minimize|\.fit\(|grad\b",
}


def audit(run_dir: str) -> dict:
    d = LOG / run_dir
    flags, n = [], 0
    for line in (d / "transcript.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") != "message" or e.get("role") != "assistant":
            continue
        for b in e.get("content") or []:
            if b.get("type") == "tool_use" and b.get("name") == "python":
                n += 1
                code = (b.get("input") or {}).get("code", "")
                hits = {k: [f"{i + 1}: {l.strip()[:200]}" for i, l in enumerate(code.splitlines())
                            if re.search(rx, l, flags=re.I)] for k, rx in AUDIT_RX.items()}
                hits = {k: v for k, v in hits.items() if v}
                if hits:
                    flags.append({"turn": e["turn"], "kinds": sorted(hits), "lines": hits})
    out = {"run": d.name, "n_python_cells": n, "n_flagged": len(flags), "flagged": flags}
    (d / "audit_flags.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- summary.json + blog data

ORIGINAL = {"base": (7.9, 13.8, 21.0, 93.0), "b10": (12.3, 25.7, 49.0, 87.2), "b25": (14.1, 25.1, 39.9, 86.4),
            "b50": (48.1, 67.1, 62.3, 33.1)}


def fmt_ci(v: dict) -> str:
    return f"{v['delta']:+.1f} [{v['ci95_boot'][0]:+.1f}, {v['ci95_boot'][1]:+.1f}]"


def summarize(extra_text: str = "") -> dict:
    st = json.loads((LOG / "stats.json").read_text())
    R = runs()
    q = st["qwen"]
    rows = []
    b = q["base"]
    rows.append(["Base (temperature 0.6)", f"{b['math500']['accuracy']:.1f}", f"{b['gsm8k']['accuracy']:.1f}",
                 f"{b['ifeval']['accuracy']:.1f}", f"{b['math500']['hit_limit_pct']:.1f}", "—", "—", "—",
                 "7.9 / 13.8 / 21.0 / 93.0"])
    for tag in QWEN_TAGS:
        if tag not in q:
            continue
        e, r = q[tag], R[tag]
        dec = {}
        dp = LOG / r["_dir"] / "decompose.json"
        if dp.exists():
            dec = json.loads(dp.read_text())
        o = ORIGINAL[tag]
        rows.append([f"Edited, {tag[1:]} forward passes",
                     f"{e['math500']['accuracy']:.1f} ({fmt_ci(e['math500']['vs_base'])})",
                     f"{e['gsm8k']['accuracy']:.1f} ({fmt_ci(e['gsm8k']['vs_base'])})",
                     f"{e['ifeval']['accuracy']:.1f} ({fmt_ci(e['ifeval']['vs_base'])})",
                     f"{e['math500']['hit_limit_pct']:.1f}",
                     f"{r.get('n_weights_changed', 0):,} ({r.get('pct_weights_changed', 0):.3f}%)",
                     dec.get("classification", "?"), f"${r['cost_usd']:.2f}",
                     f"{o[0]} / {o[1]} / {o[2]} / {o[3]}"])
    for lab, c in (q.get("controls") or {}).items():
        sc = "0.6 (b10, b25)" if "s0.6_" in lab else "0.68 (b50)"
        rows.append([f"Control: final-norm scale only, s = {sc}",
                     f"{c['math500']['accuracy']:.1f} ({fmt_ci(c['math500']['vs_base'])})",
                     f"{c['gsm8k']['accuracy']:.1f} ({fmt_ci(c['gsm8k']['vs_base'])})",
                     f"{c['ifeval']['accuracy']:.1f} ({fmt_ci(c['ifeval']['vs_base'])})",
                     f"{c['math500']['hit_limit_pct']:.1f}", "1,024", "temperature only", "—", "—"])
    g = st.get("gemma", {})
    if g.get("base"):
        rows.append(["Gemma 3 270M base (control model; MATH-500 ×4, 8k tokens)",
                     f"{g['base']['math500']['accuracy']:.1f}", "—", "—", f"{g['base']['math500']['hit_limit_pct']:.1f}",
                     "—", "—", "—", "6.8 (lenient grading)"])
    if g.get("b25"):
        rows.append(["Gemma 3 270M edited, 25 forward passes",
                     f"{g['b25']['math500']['accuracy']:.1f} ({fmt_ci(g['b25']['math500']['vs_base'])})", "—", "—",
                     f"{g['b25']['math500']['hit_limit_pct']:.1f}", "640 (0.0002%)", "pure temperature-like",
                     f"${R['gemma-b25']['cost_usd']:.2f}", "7.7 (lenient grading)"])
    grow = []
    if g.get("base"):
        grow.append(["Gemma 3 270M base", f"{g['base']['math500']['accuracy']:.1f}", "—"])
    if g.get("b25"):
        grow.append(["Gemma 3 270M, edited, 25 forward passes",
                     f"{g['b25']['math500']['accuracy']:.1f}", fmt_ci(g['b25']['math500']['vs_base'])])
    media = []
    if (FIG / "exp8_budget.png").exists():
        media.append({"type": "png", "src": "assets/figures/exp8_budget.png",
                      "caption": "Qwen3.5-0.8B: final saved edit vs base on MATH-500 (4 samples), GSM8K and IFEval "
                                 "(2 samples each), by the number of 64-problem forward passes Claude could use; right: "
                                 "share of MATH-500 answers hitting 32,768 tokens. Error bars: 95% paired bootstrap CI.",
                      "alt": "Line chart of benchmark accuracy against practice budget, and a bar chart of the "
                             "share of answers hitting the token limit."})
    cost = sum(float(json.loads((d / "result.json").read_text()).get("cost_usd") or 0)
               for d in LOG.iterdir() if (d / "result.json").exists())
    out = {"title": "Fix: make Qwen3.5-0.8B better at math",
           "status": "done" if all(t in q for t in QWEN_TAGS) else "running",
           "summary": extra_text,
           "table": {"caption": "Qwen3.5-0.8B, thinking mode, temperature 0.6, 32,768 tokens. Accuracy % with the "
                                "paired change vs base and its 95% bootstrap CI. Hit limit = % of MATH-500 answers that "
                                "reached 32,768 tokens (Gemma: 8,192). Kind: what the final edit is (GRADING.md): 'mixed' = a final-norm "
                                "logit scale (higher effective temperature) plus targeted token or direction changes. "
                                "Control rows: the final-norm part of the edits alone. Last column: the original post's "
                                "MATH-500 / GSM8K / IFEval / hit limit with GPT-6 Astra.",
                     "columns": ["Model", "MATH-500", "GSM8K", "IFEval", "Hit limit %", "Weights changed", "Kind",
                                 "API cost", "Original (Astra)"],
                     "rows": rows},
           "gemma_table": {"columns": ["Model", "MATH-500 (4 samples)", "Δ vs base [95% CI]"], "rows": grow},
           "media": media, "decisions": st.get("decisions"), "total_cost_usd": round(cost, 4)}
    (LOG / "summary.json").write_text(json.dumps(out, indent=2))
    subprocess.run([sys.executable, str(REPO / "scripts" / "build_blog_data.py"), "--exp", "exp8"], check=False)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        print(fetch(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "qwen"))
    elif cmd == "stats":
        print(json.dumps(stats(), indent=1))
    elif cmd == "summarize":
        txt = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else ""
        print(json.dumps(summarize(txt), indent=1)[:3000])
    elif cmd == "audit":
        a = audit(sys.argv[2])
        print(a["n_python_cells"], a["n_flagged"])
        for f in a["flagged"]:
            print("turn", f["turn"], f["kinds"])
            for k, v in f["lines"].items():
                for x in v[:6]:
                    print("    ", k, "|", x)
    elif cmd == "decompose":
        print(json.dumps(decompose(sys.argv[2]), indent=1))
    elif cmd == "figure":
        print(figure(json.loads((LOG / "stats.json").read_text())))
