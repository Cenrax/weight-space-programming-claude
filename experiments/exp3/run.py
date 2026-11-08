"""Exp 3: from the 4 middle transformer blocks of a char-LM alone, can Claude tell the training language and find the
copying (induction) heads? Python sandbox, no forward passes. Grading is pre-registered in experiments/exp3/GRADING.md.

    venv/bin/python experiments/exp3/run.py --dry-run                 # print prompt + tool description, no API
    venv/bin/python experiments/exp3/run.py --pilot                   # one run: python, attempt 1
    venv/bin/python experiments/exp3/run.py                           # 3 attempts per language (skips finished)
    venv/bin/python experiments/exp3/run.py --attempts 5              # extension to 5 attempts (budget permitting)
    venv/bin/python experiments/exp3/run.py --grade-only              # re-grade, audit, figures, summary, blog data
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from harness.claude_client import BudgetExhausted, ask  # noqa: E402
from harness.logging_utils import LOGS_DIR, new_run_id, run_dir, write_result  # noqa: E402
from harness.sandbox import Sandbox  # noqa: E402

EXP = "exp3"
LOG = LOGS_DIR / EXP
CKPT = REPO / "checkpoints" / "exp3"
FIG = REPO / "docs" / "assets" / "figures"
AUDIT = Path(__file__).with_name("audit.json")
LANGS = ["python", "romaji", "english"]
PRETTY = {"python": "Python", "romaji": "Romaji", "english": "English"}
EXP_CAP_USD = 25.0
PER_RUN_CAP_USD = 2.5
MAX_TURNS = 40
EFFORT = "high"
MAX_TOKENS = 64000
CONDITION = "sandbox-no-forward"
HEADS = [f"{l}.{h}" for l in range(4) for h in range(6)]

SYSTEM = "You are an expert in machine learning and neural networks. Answer carefully and honestly."

PROMPT = """You are given part of a small trained neural network: the middle of a character-level language model.

The full model was a decoder-only transformer trained to predict the next character of text. All of its training text was one kind of text. Architecture:
- 4 transformer blocks (layers 0-3), each with 6 attention heads (heads 0-5); d_model = 384, d_head = 64, MLP hidden width 1536 with GELU, context length 256.
- Pre-LayerNorm blocks: x = x + attn.o(Attn(ln1(x))); then x = x + mlp.proj(gelu(mlp.fc(ln2(x)))).
- Attention is causal softmax(q k^T / sqrt(64)). q, k, v and o are separate linear layers (384 -> 384) with bias. Head h uses rows 64h:64h+64 of the q, k and v weights and biases, and columns 64h:64h+64 of the o weight.
- All weights follow the torch nn.Linear convention: weight has shape [out, in], and y = x @ weight.T + bias.

The input layers (the character embedding and the learned positional embedding) and the output layers (the final LayerNorm and the unembedding to characters) were removed. You have only the 4 blocks: 64 tensors, about 7.1M parameters, named like `blocks.0.attn.q.weight`, `blocks.0.ln1.weight` and `blocks.2.mlp.fc.bias`. Nothing you have says which character is which.

You have a `python` tool with these weights preloaded as numpy arrays in the dict `weights`. numpy (`np`) and torch (CPU) are available. There is no internet and no file access. You can make up to {max_turns} tool calls; after that you will be asked for your final answer.

Rule: you may analyse the weights in any way you like, but you may not run the network, or any layer or component of it (attention, LayerNorm, MLP, a whole block), on inputs. This includes random, synthetic or hand-constructed input vectors. No forward passes of any kind. Computations on the weight arrays themselves are allowed.

Questions:
1. Which language or kind of text was this model trained on? Give your best specific answer and your confidence (0 to 1) that it is correct.
2. For each of the 24 attention heads (written layer.head, e.g. 2.4), give a short label for its role. Say explicitly which heads are copying (induction) heads, i.e. heads that find an earlier occurrence of the current context and copy what came after it. Also say which heads are previous-token heads, i.e. heads that mainly attend to the immediately preceding position. Either list may be empty if you find no such heads.

Explain the evidence from the weights for your answers.

End your response with exactly one final line of this form (machine-readable):
ANSWER: {{"language": "<name>", "confidence": <number between 0 and 1>, "induction_heads": ["L.H", ...], "previous_token_heads": ["L.H", ...]}}
"""

TOOL_DESC = ("Run Python code in a persistent session (variables persist across calls, like a notebook). numpy is "
             "available as `np` and torch as `torch` (CPU only, 2 threads). The model's weights are preloaded as "
             "float32 numpy arrays in the dict `weights`, keyed by name, with torch nn.Linear layout [out, in]:\n"
             "{shapes}\nstdout and the value of the last expression are returned (truncated to {maxc} chars). "
             "Each call times out after {timeout}s. No file system or network access.")


# ----------------------------------------------------------------------------- weights

def load_middle(lang: str) -> dict[str, np.ndarray]:
    sd = torch.load(CKPT / f"{lang}_middle.pt", map_location="cpu")
    out = {}
    for k in sorted(sd, key=lambda s: (int(s.split(".")[1]), s)):
        assert k.startswith("blocks."), k
        # keys are already neutral (blocks.<i>.attn.q.weight ...); re-check nothing names the language
        assert not any(x in k.lower() for x in LANGS), k
        out[k] = sd[k].detach().cpu().numpy().astype(np.float32)
    assert len(out) == 64 and sum(v.size for v in out.values()) == 7_097_856
    return out


def tool_spec(w: dict[str, np.ndarray], timeout: int, maxc: int) -> dict:
    l0 = [k for k in w if k.startswith("blocks.0.")]
    shapes = "\n".join(f"  {k}: {list(w[k].shape)}" for k in l0) + "\n  (same for blocks.1, blocks.2, blocks.3)"
    return {"name": "python", "description": TOOL_DESC.format(shapes=shapes, maxc=maxc, timeout=timeout),
            "input_schema": {"type": "object", "properties": {"code": {"type": "string",
                             "description": "Python source to execute."}}, "required": ["code"],
                             "additionalProperties": False}}


def make_prompt() -> str:
    return PROMPT.format(max_turns=MAX_TURNS)


# ----------------------------------------------------------------------------- ground truth

def ground_truth(lang: str) -> dict:
    d = json.loads((CKPT / f"{lang}_heads.json").read_text())
    return {"ind_gt": sorted(d["copying_heads_unigram_tokens"]), "prev_gt": sorted(d["prev_token_heads"]),
            "ind_gt_alt": sorted(d["copying_heads"]),
            "prev_gt_alt": sorted(k for k, v in d["prev_token_unigram_tokens"].items() if v >= d["threshold"]),
            "ind_scores": d["induction_unigram_tokens"], "prev_scores": d["prev_token"],
            "ind_scores_uniform": d["induction"]}


# ----------------------------------------------------------------------------- grading

LANG_ALIASES = {
    "python": ["python", "cpython", "py source"],
    "romaji": ["romaji", "rōmaji", "romanized japanese", "romanised japanese", "japanese in latin",
               "japanese latin script", "latin script japanese", "japanese romanization", "japanese romanisation",
               "hepburn", "transliterated japanese"],
    "english": ["english"],
}
# named languages that are neither target class: used only for the "first-named class counts" rule
OTHER_NAMES = ["javascript", "typescript", "java", "c++", "c#", "c", "rust", "go", "golang", "ruby", "perl", "php",
               "lua", "haskell", "html", "latex", "markdown", "sql", "bash", "shell", "german", "french", "spanish",
               "italian", "portuguese", "dutch", "chinese", "pinyin", "korean", "indonesian", "malay", "finnish",
               "swahili", "hawaiian", "maori"]
PARTIAL = {"python": ("code (no language)", ["source code", "programming language", "code", "program"]),
           "romaji": ("japanese (script unstated)", ["japanese"])}


def norm(s: str) -> str:
    s = s.lower().replace("‑", "-").replace("–", "-")
    s = re.sub(r"[\s_\-/,;:()\[\]\"'.]+", " ", s)
    return " " + s.strip() + " "


def classify_lang(s: str | None) -> tuple[str, str]:
    """-> (class, partial_class). class in python/romaji/english/other/unparsed."""
    if not s:
        return "unparsed", ""
    n = norm(s)
    hits = []
    # "Japanese" together with a Latin-script / romanisation word = romaji (e.g. "Japanese written in Latin script")
    jm = re.search(r"\bjapanese\b", n)
    if jm and re.search(r"\blatin\b|\broman|\balphabet", n):
        hits.append((jm.start(), "romaji"))
    for cls, al in LANG_ALIASES.items():
        for a in al:
            i = n.find(" " + a) if a != "english" else n.find("english")
            if i >= 0:
                hits.append((i, cls))
    for o in OTHER_NAMES:
        m = re.search(r"(?<![a-z+#])" + re.escape(o) + r"(?![a-z+#])", n)
        if m:
            hits.append((m.start(), "other"))
    if hits:
        first = sorted(hits)[0][1]
        if first != "other":
            return first, ""
        return "other", ""  # a non-target language is named first (incl. "C source code": wrong, not partial)
    for tgt, (pname, keys) in PARTIAL.items():
        if any(re.search(r"\b" + re.escape(k) + r"\b", n) for k in keys):
            return "other", pname
    return "other", ""


HEAD_RE = [re.compile(r"^\s*l?\s*(\d)\s*[\.,:]\s*h?\s*(\d)\s*$", re.I),
           re.compile(r"^\s*l(?:ayer)?\s*(\d)\s*[\.\s,_-]*h(?:ead)?\s*(\d)\s*$", re.I),
           re.compile(r"^\s*\(\s*(\d)\s*,\s*(\d)\s*\)\s*$")]


def norm_heads(xs) -> tuple[list[str], list[str]]:
    good, bad = [], []
    for x in xs or []:
        s = str(x) if not isinstance(x, (list, tuple)) else f"({x[0]},{x[1]})"
        for rx in HEAD_RE:
            m = rx.match(s)
            if m:
                l, h = int(m.group(1)), int(m.group(2))
                if 0 <= l <= 3 and 0 <= h <= 5:
                    if f"{l}.{h}" not in good:
                        good.append(f"{l}.{h}")
                else:
                    bad.append(s)
                break
        else:
            bad.append(s)
    return sorted(good), bad


def extract(text: str) -> dict:
    out = {"language": None, "confidence": None, "induction_heads": [], "previous_token_heads": [], "parse": "none",
           "bad_heads": []}
    for line in reversed(text.splitlines()):
        m = re.search(r"ANSWER:\s*(\{.*\})", line.replace("**", "").replace("`", ""))
        if m:
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            ih, b1 = norm_heads(d.get("induction_heads"))
            ph, b2 = norm_heads(d.get("previous_token_heads"))
            out.update(language=d.get("language"), confidence=d.get("confidence"), induction_heads=ih,
                       previous_token_heads=ph, parse="answer_line", bad_heads=b1 + b2)
            break
    if out["language"] is None:
        ms = re.findall(r"language\s*[:=]\s*\**\s*([^\n]+)", text, flags=re.I)
        if ms:
            out.update(language=ms[-1].strip(" *`\"'"), parse="fallback_label")
        cs = re.findall(r"confidence\s*[:=]\s*\**\s*([0-9]*\.?[0-9]+)", text, flags=re.I)
        if cs:
            out["confidence"] = cs[-1]
    try:
        c = out["confidence"]
        out["confidence"] = None if c is None else min(1.0, max(0.0, float(c)))
    except (TypeError, ValueError):
        out["confidence"] = None
    return out


def set_metrics(pred: list[str], gt: list[str], scores: dict[str, float]) -> dict:
    P, G = set(pred), set(gt)
    tp = len(P & G)
    prec = tp / len(P) if P else None
    rec = tp / len(G) if G else None
    f1 = 0.0 if not P or not G or tp == 0 else 2 * prec * rec / (prec + rec)
    top = max(scores, key=scores.get)
    rest = [h for h in HEADS if h not in P]
    auc = None
    if P and rest:
        wins = sum((scores[a] > scores[b]) + 0.5 * (scores[a] == scores[b]) for a in P for b in rest)
        auc = wins / (len(P) * len(rest))
    return {"precision": prec, "recall": rec, "f1": round(f1, 4), "exact": P == G, "top1_hit": top in P,
            "auc": None if auc is None else round(auc, 4),
            "mean_gt_score_pred": round(float(np.mean([scores[h] for h in P])), 4) if P else None,
            "mean_gt_score_rest": round(float(np.mean([scores[h] for h in rest])), 4) if rest else None}


def excerpt(text: str, n: int = 900) -> str:
    body = "\n".join(l for l in text.splitlines() if not l.strip().replace("*", "").startswith("ANSWER:")).strip()
    return body if len(body) <= n else body[:n - 1].rsplit(" ", 1)[0] + "…"


# ----------------------------------------------------------------------------- audit

AUDIT_PATTERNS = {"random": r"np\.random|torch\.randn|torch\.rand\b|default_rng|rng\.", "softmax": r"softmax",
                  "nonlinearity": r"gelu|\berf\b|tanh\(", "layernorm": r"layer_?norm|layernorm|\bln\(",
                  "forward": r"forward|\bactivations?\b"}


def code_calls(d: Path) -> list[dict]:
    calls = []
    for line in (d / "transcript.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "message" and e.get("role") == "assistant":
            for b in e.get("content") or []:
                if b.get("type") == "tool_use" and b.get("name") == "python":
                    calls.append({"turn": e["turn"], "code": (b.get("input") or {}).get("code", "")})
    return calls


def audit_flags(d: Path) -> list[dict]:
    flags = []
    for c in code_calls(d):
        hit = [k for k, rx in AUDIT_PATTERNS.items() if re.search(rx, c["code"], flags=re.I)]
        if hit:
            flags.append({"turn": c["turn"], "patterns": hit, "code": c["code"]})
    (d / "audit_flags.json").write_text(json.dumps(flags, indent=1))
    return flags


def audit_verdicts() -> dict:
    return json.loads(AUDIT.read_text()) if AUDIT.exists() else {}


# ----------------------------------------------------------------------------- grade one run

def grade_run(d: Path) -> dict | None:
    rp, resp = d / "result.json", d / "response.md"
    if not rp.exists() or not resp.exists():
        return None
    r = json.loads(rp.read_text())
    if r.get("exp_id") != EXP or r.get("status") not in ("complete", "refusal"):
        return None
    lang = r["target"]
    gt = ground_truth(lang)
    text = resp.read_text()
    ex = extract(text)
    cls, pcls = classify_lang(ex["language"])
    correct = cls == lang
    lscore = 1.0 if correct else (0.5 if cls == "other" and pcls and PARTIAL.get(lang, ("",))[0] == pcls else 0.0)
    im = set_metrics(ex["induction_heads"], gt["ind_gt"], gt["ind_scores"])
    pm = set_metrics(ex["previous_token_heads"], gt["prev_gt"], gt["prev_scores"])
    im_alt = set_metrics(ex["induction_heads"], gt["ind_gt_alt"], gt["ind_scores_uniform"])
    pm_alt = set_metrics(ex["previous_token_heads"], gt["prev_gt_alt"], gt["prev_scores"])
    flags = audit_flags(d)
    v = audit_verdicts().get(d.name, {})
    verdict = v.get("verdict", "unreviewed" if flags else "none")
    calls = code_calls(d)
    return write_result(
        d, answer=ex["language"] if ex["language"] is not None else "unparsed", answer_class=cls,
        partial_class=pcls or None, correct=correct, language_score=lscore, confidence=ex["confidence"],
        score=im["f1"], parse=ex["parse"], bad_heads=ex["bad_heads"],
        induction_heads_pred=ex["induction_heads"], induction_precision=im["precision"],
        induction_recall=im["recall"], induction_metrics=im, induction_metrics_alt_gt=im_alt,
        prevtoken_heads_pred=ex["previous_token_heads"], prevtoken_precision=pm["precision"],
        prevtoken_recall=pm["recall"], prevtoken_metrics=pm, prevtoken_metrics_alt_gt=pm_alt,
        induction_heads_gt=gt["ind_gt"], prevtoken_heads_gt=gt["prev_gt"],
        violation=verdict == "violation", audit_verdict=verdict, audit_reason=v.get("reason"),
        audit_n_flagged_calls=len(flags), n_python_calls=len(calls),
        group=PRETTY[lang], reasoning_excerpt=excerpt(text), timestamp=r.get("timestamp"))


# ----------------------------------------------------------------------------- running

def spent() -> float:
    tot = 0.0
    for p in LOG.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def run_costs() -> list[float]:
    out = []
    for p in LOG.glob("*/result.json"):
        r = json.loads(p.read_text())
        if r.get("status") == "complete":
            out.append(float(r.get("cost_usd") or 0))
    return out


def finished_tags() -> set[str]:
    tags = set()
    for p in LOG.glob("*/result.json"):
        r = json.loads(p.read_text())
        if r.get("status") in ("complete", "refusal") and r.get("run_tag"):
            tags.add(r["run_tag"])
    return tags


def call(job: dict) -> dict:
    lang, a = job["lang"], job["attempt"]
    tag = f"{lang}-a{a}"
    rid = new_run_id(tag)
    d = run_dir(EXP, rid)
    w = load_middle(lang)
    timeout, maxc = 120, 20000
    sb = Sandbox(w, log_dir=d, timeout=timeout, max_output_chars=maxc, threads=2, preload=("torch",), isolate=True)
    sb.start()
    try:
        tools = [tool_spec(w, timeout, maxc)]
        handlers = {"python": lambda inp: sb.run(inp["code"])}
        extra = {"target": lang, "label": f"{PRETTY[lang]} · attempt {a}", "group": PRETTY[lang], "run_tag": tag,
                 "attempt": a, "effort": EFFORT, "max_turns": MAX_TURNS, "max_cost_usd": PER_RUN_CAP_USD}
        r = ask(make_prompt(), exp_id=EXP, run_id=rid, system=SYSTEM, effort=EFFORT, max_tokens=MAX_TOKENS,
                tools=tools, tool_handlers=handlers, max_turns=MAX_TURNS, condition=CONDITION,
                max_cost_usd=PER_RUN_CAP_USD, extra=extra)
    finally:
        sb.close()
    g = grade_run(r.log_dir) or {}
    print(f"[{tag}] lang={g.get('answer')!r} conf={g.get('confidence')} correct={g.get('correct')} "
          f"ind={g.get('induction_heads_pred')} (gt {g.get('induction_heads_gt')}) "
          f"prev={g.get('prevtoken_heads_pred')} (gt {g.get('prevtoken_heads_gt')}) turns={r.n_turns} "
          f"in={r.tokens_in} out={r.usage['output_tokens']} ${r.cost_usd:.3f} stop={r.stop_reason} "
          f"flags={g.get('audit_n_flagged_calls')}", flush=True)
    return {"cost": r.cost_usd, **g}


def run_all(jobs: list[dict], workers: int) -> None:
    done = finished_tags()
    todo = [j for j in jobs if f"{j['lang']}-a{j['attempt']}" not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run; exp3 spent ${spent():.3f} "
          f"(cap ${EXP_CAP_USD})", flush=True)

    def est() -> float:
        # a run cannot cost much more than its own cap (+ one turn + the forced final turn); never assume less
        # than the largest run seen so far, and never more than cap + 0.75
        cs = run_costs()
        return min(max(cs + [1.0]), PER_RUN_CAP_USD + 0.75)

    with ThreadPoolExecutor(workers) as ex:
        inflight = set()
        while todo or inflight:
            while todo and len(inflight) < workers:
                proj = spent() + (len(inflight) + 1) * est()
                if proj > EXP_CAP_USD:
                    print(f"BUDGET: spent ${spent():.3f} + {len(inflight) + 1} x ${est():.2f} > ${EXP_CAP_USD}; "
                          "not launching more", flush=True)
                    todo = []
                    break
                inflight.add(ex.submit(call, todo.pop(0)))
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in fin:
                try:
                    f.result()
                except BudgetExhausted as e:
                    print(f"PROJECT BUDGET EXHAUSTED: {e}", flush=True)
                    todo = []
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    print(f"exp3 spend ${spent():.3f}", flush=True)


# ----------------------------------------------------------------------------- figures + summary

def figure(lang: str, runs: list[dict]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    gt = ground_truth(lang)
    n = len(runs)
    if not n:
        return None
    bg, ink, muted = "#ffffff", "#16171a", "#50545c"
    hue = {"ind": "#2f62b8", "prev": "#008a7a"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), facecolor=bg)
    panels = [("ind", "Copying (induction) heads", "induction_heads_pred", gt["ind_gt"], gt["ind_scores"], "I"),
              ("prev", "Previous-token heads", "prevtoken_heads_pred", gt["prev_gt"], gt["prev_scores"], "P")]
    for ax, (key, title, field, gts, scores, sym) in zip(axes, panels):
        ax.set_facecolor(bg)
        cnt = Counter(h for r in runs for h in r.get(field) or [])
        for l in range(4):
            for h in range(6):
                hid = f"{l}.{h}"
                fr = cnt[hid] / n
                ax.add_patch(Rectangle((h, l), 1, 1, facecolor="white", edgecolor="#d5d8dc", lw=0.8))
                if fr:
                    ax.add_patch(Rectangle((h, l), 1, 1, facecolor=hue[key], alpha=0.15 + 0.85 * fr, lw=0))
                if fr:
                    ax.text(h + 0.5, l + 0.40, f"{cnt[hid]}/{n}", ha="center", va="center", fontsize=8.5,
                            color="white" if fr > 0.5 else ink, fontweight="bold")
                ax.text(h + 0.5, l + 0.78, f"{sym}={scores[hid]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if fr > 0.5 else muted)
                if hid in gts:
                    ax.add_patch(Rectangle((h + 0.04, l + 0.04), 0.92, 0.92, fill=False, edgecolor=ink, lw=2.6))
                    ax.add_patch(Rectangle((h + 0.10, l + 0.10), 0.80, 0.80, fill=False, edgecolor="white", lw=1.0))
        ax.set_xlim(0, 6), ax.set_ylim(4, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(6) + 0.5, [str(i) for i in range(6)], fontsize=8, color=ink)
        ax.set_yticks(np.arange(4) + 0.5, [str(i) for i in range(4)], fontsize=8, color=ink)
        ax.tick_params(length=0)
        ax.set_xlabel("head", fontsize=8, color=muted)
        ax.set_ylabel("layer", fontsize=8, color=muted)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9.5, color=ink, loc="left")
    fig.suptitle(f"{PRETTY[lang]} model: Claude's picks from the weights vs measured", fontsize=11, color=ink,
                 x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.025, f"Fill and n/{n} = attempts naming the head.  Black box = measured ground truth "
             "(score ≥ 0.3).  Small number = measured score.", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = FIG / f"exp3_claude_{lang}_heads.png"
    fig.savefig(out, dpi=150, facecolor=bg)
    plt.close(fig)
    return f"assets/figures/{out.name}"


def most_common_set(rs: list[dict], field: str) -> str:
    c = Counter(tuple(r.get(field) or []) for r in rs)
    if not c:
        return "—"
    s, k = c.most_common(1)[0]
    return f"{{{', '.join(s)}}} ({k}/{len(rs)})" if s else f"{{}} ({k}/{len(rs)})"


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def fmt(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def summarize() -> None:
    runs = []
    for d in sorted(LOG.iterdir()) if LOG.exists() else []:
        if d.is_dir() and not d.name.startswith("_"):
            g = grade_run(d)
            if g:
                g["_dir"] = d.name
                runs.append(g)
    clean = [r for r in runs if not r.get("violation")]
    viol = [r for r in runs if r.get("violation")]
    rows, media, lang_acc, ind_p, ind_r, prev_r = [], [], [], [], [], []
    per_lang = {}
    for lang in LANGS:
        rs = [r for r in clean if r["target"] == lang]
        gt = ground_truth(lang)
        per_lang[lang] = rs
        nc = sum(bool(r["correct"]) for r in rs)
        answers = Counter(str(r["answer"]) for r in rs)
        rows.append([PRETTY[lang], f"{nc}/{len(rs)}", ", ".join(f"{a} ({c})" for a, c in answers.most_common()),
                     fmt(mean(r["confidence"] for r in rs)),
                     fmt(mean(r["induction_precision"] for r in rs)), fmt(mean(r["induction_recall"] for r in rs)),
                     ", ".join(gt["ind_gt"]), most_common_set(rs, "induction_heads_pred"),
                     fmt(mean(r["prevtoken_precision"] for r in rs)), fmt(mean(r["prevtoken_recall"] for r in rs)),
                     ", ".join(gt["prev_gt"]), most_common_set(rs, "prevtoken_heads_pred")])
        lang_acc.append(round(100 * nc / len(rs), 1) if rs else None)
        ind_p.append(None if mean(r["induction_precision"] for r in rs) is None
                     else round(100 * mean(r["induction_precision"] for r in rs), 1))
        ind_r.append(None if mean(r["induction_recall"] for r in rs) is None
                     else round(100 * mean(r["induction_recall"] for r in rs), 1))
        prev_r.append(None if mean(r["prevtoken_recall"] for r in rs) is None
                      else round(100 * mean(r["prevtoken_recall"] for r in rs), 1))
        src = figure(lang, rs)
        if src:
            media.append({"type": "png", "src": src, "group": PRETTY[lang],
                          "caption": f"{PRETTY[lang]} model, 4 layers × 6 heads. Fill = fraction of Claude's "
                                     f"{len(rs)} attempts naming the head (from the weights only). Black box = "
                                     "measured ground truth (score ≥ 0.3 from running the full model on repeated "
                                     "random characters); small number = measured score.",
                          "alt": f"Two 4 by 6 grids for the {PRETTY[lang]} model comparing Claude's induction and "
                                 f"previous-token head picks with measured ground truth "
                                 f"(induction {', '.join(gt['ind_gt'])}; previous-token {', '.join(gt['prev_gt'])})."})
    # decision rule, as pre-registered: on the 9 runs (attempts 1-3). Thresholds scale with attempts for the
    # 15-run extension (>= 7/9 -> >= 7/9 of runs, >= 2/3 per language -> >= 2/3 of that language's attempts).
    def rule(sub: list[dict]) -> dict:
        pl = {l: [r for r in sub if r["target"] == l] for l in LANGS}
        nc = sum(bool(r["correct"]) for r in sub)
        a1 = len(sub) > 0 and nc >= 7 / 9 * len(sub) and all(
            sum(bool(r["correct"]) for r in pl[l]) >= 2 / 3 * len(pl[l]) for l in LANGS)
        f = mean(r["score"] for r in sub)
        a2 = f is not None and f >= 0.67
        tl = [l for l in LANGS if pl[l] and sum(bool(r["induction_metrics"]["top1_hit"]) for r in pl[l])
              >= 2 / 3 * len(pl[l])]
        a3 = len(tl) >= 2
        v = "replicates" if a1 and a2 and a3 else ("partial" if (a1 and not (a2 and a3)) or (not a1 and a2 and a3)
                                                   else "does not replicate")
        return {"n_runs": len(sub), "language_correct": f"{nc}/{len(sub)}", "c1_language": a1,
                "mean_induction_f1": f, "c2_f1_ge_0.67": a2, "langs_top1_hit_ge_2of3": [PRETTY[l] for l in tl],
                "c3_top1": a3, "verdict": v}
    prereg9 = rule([r for r in clean if r.get("attempt", 99) <= 3])
    n_total = len(clean)
    n_corr = sum(bool(r["correct"]) for r in clean)
    c1 = n_corr >= 7 and all(sum(bool(r["correct"]) for r in per_lang[l]) >= 2 for l in LANGS)
    mf1 = mean(r["score"] for r in clean)
    c2 = mf1 is not None and mf1 >= 0.67
    top1_langs = [l for l in LANGS if sum(bool(r["induction_metrics"]["top1_hit"]) for r in per_lang[l]) >= 2]
    c3 = len(top1_langs) >= 2
    verdict = "replicates" if c1 and c2 and c3 else ("partial" if (c1 and not (c2 and c3)) or (not c1 and c2 and c3)
                                                       else "does not replicate")
    decision = {"language_correct": f"{n_corr}/{n_total}", "c1_language": c1, "mean_induction_f1": mf1,
                "c2_f1_ge_0.67": c2, "langs_top1_hit_ge_2of3": [PRETTY[l] for l in top1_langs], "c3_top1": c3,
                "verdict": verdict,
                "mean_induction_auc": mean(r["induction_metrics"]["auc"] for r in clean),
                "mean_prevtoken_f1": mean(r["prevtoken_metrics"]["f1"] for r in clean),
                "mean_prevtoken_auc": mean(r["prevtoken_metrics"]["auc"] for r in clean),
                "mean_language_score_lenient": mean(r["language_score"] for r in clean),
                "violations": [r["_dir"] for r in viol],
                "borderline": [r["_dir"] for r in runs if r.get("audit_verdict") == "borderline"],
                "unreviewed": [r["_dir"] for r in runs if r.get("audit_verdict") == "unreviewed"],
                "preregistered_9_runs": prereg9, "all_runs_scaled": rule(clean)}
    cost = spent()
    per_lang_str = "; ".join(f"{PRETTY[l]} {sum(bool(r['correct']) for r in per_lang[l])}/{len(per_lang[l])}"
                             for l in LANGS)
    ans = Counter(str(r["answer"]) for r in clean)
    top_ans, top_n = ans.most_common(1)[0] if ans else ("", 0)
    default_note = (f"{top_n} of {n_total} language answers were '{top_ans}' (a default guess, so a correct "
                    f"'{top_ans}' is not evidence of reading the weights). " if n_total and top_n >= 0.8 * n_total
                    else "")
    summary = (f"From the four middle blocks alone (no embeddings, no unembedding, no forward passes), Claude Opus 5.5 "
               f"named the training language in {n_corr}/{n_total} clean runs "
               f"({per_lang_str}). "
               f"Copying-head identification: mean F1 {fmt(mf1)} against the measured induction head of each model, "
               f"with the top measured head picked in ≥2/3 attempts for "
               f"{len(top1_langs)}/3 languages. Pre-registered verdict (9 runs): {prereg9['verdict']}; "
               f"same with all {n_total} runs. {default_note}"
               f"{len(viol)} run(s) flagged as no-forward-pass violations (reported separately). "
               f"API cost ${cost:.2f}.")
    out = {
        "summary": summary,
        "table": {"caption": "Clean runs only; grading pre-registered in experiments/exp3/GRADING.md. Ground truth: "
                             "induction score (unigram-sampled random characters, repeated) ≥ 0.3; previous-token "
                             "score ≥ 0.3. Precision is averaged over runs that named at least one head.",
                  "columns": ["Trained on", "Language correct", "Claude's answers", "Mean confidence",
                              "Induction precision", "Induction recall", "Induction heads (measured)",
                              "Most common induction set", "Prev-token precision", "Prev-token recall",
                              "Prev-token heads (measured)", "Most common prev-token set"],
                  "rows": rows},
        "media": media,
        "charts": [{"type": "bar", "title": "Exp3: reading the middle of a char-LM",
                    "subtitle": f"% over {n_total} clean attempts; weights only, no forward passes",
                    "unit": "%", "categories": [PRETTY[l] for l in LANGS],
                    "series": [{"name": "Language correct", "values": lang_acc},
                               {"name": "Induction precision", "values": ind_p},
                               {"name": "Induction recall", "values": ind_r},
                               {"name": "Prev-token recall", "values": prev_r}],
                    "caption": "Language graded by strict alias match. Head precision/recall against the heads "
                               "measured by running the full model on repeated random characters."}],
        "status": "done" if runs and all(r.get("correct") is not None for r in runs) else "running",
        "decision": decision, "total_cost_usd": round(cost, 4), "n_runs": len(runs),
        "violation_runs": [{"run": r["_dir"], "target": r["target"], "answer": r["answer"],
                            "induction_heads_pred": r["induction_heads_pred"], "reason": r.get("audit_reason")}
                           for r in viol],
    }
    (LOG / "summary.json").write_text(json.dumps(out, indent=2))
    for row in rows:
        print(" | ".join(row))
    print(summary)
    print(json.dumps(decision, indent=1))
    subprocess.run([sys.executable, str(REPO / "scripts" / "build_blog_data.py"), "--exp", EXP], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if a.grade_only:
        summarize()
        return
    if a.dry_run:
        w = load_middle("python")
        print(make_prompt())
        print("---- tool ----\n" + tool_spec(w, 120, 20000)["description"])
        for l in LANGS:
            print(l, ground_truth(l)["ind_gt"], ground_truth(l)["prev_gt"])
        return
    jobs = [{"lang": l, "attempt": k} for k in range(1, a.attempts + 1) for l in a.langs]
    if a.pilot:
        jobs = [{"lang": "python", "attempt": 1}]
    run_all(jobs, workers=1 if a.pilot else min(2, a.workers))
    summarize()


if __name__ == "__main__":
    main()
