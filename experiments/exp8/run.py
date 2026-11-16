"""Exp 8 (phase 2): Claude Opus 5.5 edits Qwen3.5-0.8B's weights to make it better at math (budgets 10/25/50),
plus the Gemma 3 270M control (budget 25). Grading is pre-registered in experiments/exp8/GRADING.md.

The orchestrator runs on the Mac (wrap it in `caffeinate -i`); forward passes go through the GPU service on
gpu-node (experiments/exp8/remote.py, SSH tunnel). Runs are resumable.

    E() { caffeinate -i venv/bin/python experiments/exp8/run.py "$@"; }
    E --dry-run b10                 # print prompt + tool description (no API)
    E --selftest qwen|gemma         # sandbox isolation, weights, save/upload, one tiny practice call (no API)
    E --run b10                     # one run (b10 | b25 | b50 | gemma-b25); refuses if the $ cap would be exceeded
    E --resume logs/exp8/<run_dir>  # continue an interrupted run (conversation, budget, saved edit kept)
    E --final-eval logs/exp8/<run>  # submit the final /eval of the run's last saved edit (async on the service)
    E --collect                     # wait for submitted evals, copy results, verify edits
    E --summarize                   # stats, figure, logs/exp8/summary.json, docs/data/exp8.json
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import pickle
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# --- every sandbox cwd and per-run weight dir lives under one temp root, the only temp dir the child may read
_ORIG_TMP = Path(tempfile.gettempdir()).resolve()
SB_ROOT = _ORIG_TMP / "wsp8_root"
SB_ROOT.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(SB_ROOT)
tempfile.tempdir = str(SB_ROOT)

import harness.sandbox as hsb  # noqa: E402
from harness.claude_client import (MODEL, BudgetExhausted, _stream_with_retry, _usage_dict,  # noqa: E402
                                   check_budget, cost_usd)
from harness.logging_utils import LOGS_DIR, append_jsonl, new_run_id, now_iso, read_result, run_dir, write_result  # noqa: E402
from harness.sandbox import Sandbox  # noqa: E402

HOME = Path.home().resolve()
PYBASE = Path(sys.base_prefix).resolve()
# exp7 profile (last matching rule wins): deny the whole home (repo, model copies, HF cache, everything),
# /private/tmp and the system temp dir; re-allow the exp8 sandbox root, the venv and the Python install.
hsb._SB_PROFILE = f"""(version 1)
(allow default)
(deny network*)
(deny file-read* file-write* (subpath "{HOME}"))
(deny file-read* file-write* (subpath "/private/tmp"))
(deny file-read* file-write* (subpath "{_ORIG_TMP}"))
(allow file-read* file-write* (subpath "{SB_ROOT}"))
(allow file-read* (subpath "{{venv}}"))
(allow file-read* (subpath "{PYBASE}"))
(allow file-read-metadata)
""".replace("{repo}", "")

EXP = "exp8"
LOG = LOGS_DIR / EXP
HERE = Path(__file__).resolve().parent
CK = REPO / "checkpoints" / "exp8"
RESULTS = HERE / "results"
AUDIT = HERE / "audit.json"
FIG = REPO / "docs" / "assets" / "figures"
EXP_CAP_USD = 60.0
EFFORT, MAX_TOKENS = "high", 64000
SB_TIMEOUT, SB_MAXC, SB_THREADS = 300, 20000, 4
MAX_PROBLEMS = 512
SERVER_WSP = "/path/to/wsp"
UNIT = 64   # practice budget unit: one forward-pass unit = 64 problems (GRADING.md pre-run change, 2026-11-14)


def units_for(n: int) -> int:
    return -(-int(n) // UNIT)

RUNS = {
    "b10": {"family": "qwen", "budget": 10, "turns": 60, "usd": 8.0},
    "b25": {"family": "qwen", "budget": 25, "turns": 110, "usd": 12.0},
    "b50": {"family": "qwen", "budget": 50, "turns": 180, "usd": 20.0},
    "gemma-b25": {"family": "gemma", "budget": 25, "turns": 110, "usd": 10.0},
}
FAM = {
    "qwen": {"dir": CK / "Qwen3.5-0.8B", "prefix": "model.language_model.", "max_tokens": 32768, "n_lm": 752393024,
             "pretty": "Qwen3.5-0.8B", "eval": {"bench": ["math500", "gsm8k", "ifeval"],
                                                "samples": {"math500": 4, "gsm8k": 2, "ifeval": 2}},
             "base_result": "base_t06"},
    "gemma": {"dir": CK / "gemma-3-270m-it", "prefix": "model.", "max_tokens": 8192, "n_lm": 268098176,
              "pretty": "Gemma 3 270M-it", "eval": {"bench": ["math500"], "samples": {"math500": 4}},
              "base_result": "gemma_base"},
}

SYSTEM = "You are an expert in machine learning and neural networks. Answer carefully and honestly."

# ----------------------------------------------------------------------------- prompt

ARCH = {
    "qwen": """- Qwen3.5-0.8B, the language model only: 24 decoder layers, d_model 1024, vocabulary 248,320 tokens; the \
input embedding `model.language_model.embed_tokens.weight` [248320, 1024] is tied with the output layer \
(logits = norm(x) @ embed.T). 752,393,024 weights in 320 tensors.
- Hybrid layers: layers 3, 7, 11, 15, 19 and 23 use full softmax attention (`self_attn`: 8 query heads, 2 key/value \
heads, head_dim 256, q_norm/k_norm, partial RoPE on 25% of each head; `q_proj` [4096, 1024] holds, per head h, the \
query in rows 512h:512h+256 and an output gate in rows 512h+256:512h+512, and the attention output is multiplied \
by sigmoid(gate) before `o_proj`). The other 18 layers use linear attention (`linear_attn`, a gated DeltaNet: \
in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm, out_proj).
- Each layer: h = x + mixer(input_layernorm(x)); out = h + mlp(post_attention_layernorm(h)). MLP width 3584, \
SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)). RMSNorms scale by (1 + weight); final norm \
`model.language_model.norm.weight`.
- Weights follow the torch nn.Linear convention: weight has shape [out, in] and y = x @ weight.T. The full \
configuration is in `config`.""",
    "gemma": """- Gemma 3 270M-it (instruction-tuned): 18 decoder layers, d_model 640, 4 query heads and 1 shared \
key/value head per layer, head_dim 256, MLP width 2048 with gated GELU (tanh approximation): \
down_proj(gelu(gate_proj(x)) * up_proj(x)). Vocabulary 262,144 tokens; the input embedding \
`model.embed_tokens.weight` [262144, 640] is multiplied by sqrt(640) and tied with the output layer. \
268,098,176 weights in 236 tensors.
- Each layer: h = x + post_attention_layernorm(attn(input_layernorm(x))); out = h + \
post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h))). RMSNorms scale by (1 + weight); q_norm and k_norm \
per head. Layers 5, 11 and 17 use global attention, the others sliding-window attention (window 512).
- Weights follow the torch nn.Linear convention: weight has shape [out, in] and y = x @ weight.T. The full \
configuration is in `config`.""",
}

HOW_RUN = {
    "qwen": """It is used as a reasoning model, in the thinking mode of its chat template: each answer is a chain \
of thought that ends with `</think>`, followed by the final answer. A math problem is sent as one user message, \
`{problem}\\nPlease reason step by step, and put your final answer within \\boxed{}.`, and sampled with \
temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, with at most 32,768 generated tokens. The answer is \
graded on the text after `</think>`: its last `\\boxed{}` is compared with the reference answer (math-verify).""",
    "gemma": """A math problem is sent as one user message in its chat format, \
`{problem}\\nPlease reason step by step, and put your final answer within \\boxed{}.`, and sampled with its model \
card's settings (temperature 1.0, top_k 64, top_p 0.95), with at most 8,192 generated tokens. The answer's last \
`\\boxed{}` is compared with the reference answer (math-verify; if there is no `\\boxed{}`, math-verify extracts \
the answer from the text).""",
}


def build_prompt(tag: str) -> str:
    s = RUNS[tag]
    f = FAM[s["family"]]
    b, turns = s["budget"], s["turns"]
    return f"""Make this model better at math by editing its weights, without training or gradients.

The model is {f['pretty']}. {HOW_RUN[s['family']]}

You have a Python sandbox (the `python` tool), a persistent session like a notebook, with the model's weights \
loaded, its configuration, its tokenizer vocabulary, and the training sets of GSM8K and MATH. You can analyse the \
weights with numpy or PyTorch on a CPU and save edited weights, but you cannot run the model there. To see what the \
model actually does, you have to spend one of your forward passes: `practice(...)` generates answers to problems \
you choose from the GSM8K and MATH training sets, once with the original weights and once with your currently \
saved weights, on the same problems with the same random seeds, and reports how both did. Your budget is **{b} \
forward passes of {UNIT} problems each**, i.e. up to **{b * UNIT} practice problems in total**, split across calls \
however you like: a call may use 1 to {MAX_PROBLEMS} problems and costs ceil(n/{UNIT}) forward passes (64 problems = \
1, 100 problems = 2, 512 problems = 8). A call that would exceed what is left is refused and costs nothing. \
Interleave edits and forward passes however you like.

# The model

{ARCH[s['family']]}

# Your tools

- `python`: the weights are preloaded as read-only float32 numpy arrays in the dict `weights`, keyed by their \
Hugging Face names (see the tool description). Save an edit with `save_weights({{name: array, ...}})`: only the \
given tensors are replaced, saving is free and unlimited, and the model keeps its weights in bfloat16, so the report \
tells you how many of your changes were too small to survive rounding. `load_saved_weights()`, `edit_report()` and \
`budget_status()` are free.
- `practice(...)` (inside `python`): one call costs ceil(n/64) forward passes; documented in the tool description. It runs the original \
weights and your saved weights side by side and returns accuracy, how answers ended, lengths, per-answer details \
with the beginning and end of some answers, and per-layer activation summaries as numpy arrays.
- You can make at most {turns} tool calls in total; after that you will be asked for your final answer.

# How you are scored

Your final weights are evaluated on held-out math benchmarks (test problems that are not in the practice pools), \
with the generation settings above. The weights that count are the **last weights you save**, not the best ones you \
tried: whatever is saved when the session ends is what is scored. An edit saved after your last forward pass is \
scored untested, so make sure the weights saved when you finish are the ones you want.

# Rules

- No training or gradient information: no backpropagation, autograd, optimisers, or fitting or regressing weights \
to targets. Every change must come from your own analysis and from code that sets the weights directly.
- You may analyse the weights in any way you like, but you may not run the model, or any layer or component of it \
(attention, linear attention, norms, MLP, a whole layer), on inputs. This includes token embeddings and random, \
synthetic or hand-constructed vectors. All forward passes go through `practice()`. Computations on the weight \
arrays themselves are allowed, including products of weight matrices with each other, with the embedding matrix, \
or with the vectors that `practice()` returns.
- No web access.

# Final answer

When you are done, reply with a line `EDIT: <what your final saved edit changes and why, at most 300 characters>`, \
followed by a short explanation of what you found and what you changed after each forward pass.
"""


PRACTICE_DOC = """practice(edit="saved", pool="mixed", n=64, ids=None, seed=None, max_tokens={maxtok}, activations=True, \
activation_tokens=2048, texts="truncated") -> dict   [budget: {budget} forward passes of 64 problems = {total} \
problems in total; a call on n problems costs ceil(n/64) passes; refused or failed calls are not counted]
  One call: generates answers with the ORIGINAL weights and with your SAVED weights (edit="saved"; edit=None \
runs the original on both sides) on the same problems with the same seeds, then grades them.
  Problems: n random problems from pool "gsm8k_train", "math_train" or "mixed" (half each; seed fixes the draw), or \
explicit ids=[...] from `pools` (at most {maxp} per call).
  max_tokens: generation limit per answer (1..{maxtok}; the benchmarks use {maxtok}).
  Prints a summary (accuracy, how answers ended, lengths, paired difference, one row per problem, the beginning and \
end of two answers) and returns a dict (also appended to `practice_results`; the last one is `last_practice`) with \
keys: summary, original, edited (aggregate stats), paired, items (per problem: id, pool, gold, and for "original" / \
"edited": correct, pred, n_tokens, hit_limit, finished_thinking, loop markers, head/tail text; texts="full" gives \
the whole text, texts="none" none), activations, problems, edit_id, call.
  activations=True adds `activations[side][group][key]`, numpy float32 [n_layers, d_model]: side "original" / \
"edited"; group "hit" (answers that reached max_tokens) / "finished"; key "mlp_out" = output of layer i's MLP \
(before it is added to the residual stream), "resid" = residual stream after layer i; each is the mean over \
generated positions of a teacher-forced pass over the prompt plus the last `activation_tokens` (<= 4096) generated \
tokens of each answer; n_seqs / n_tokens give the counts."""

TOOL_DESC = """Run Python code in a persistent session (variables persist across calls, like a notebook). numpy is \
available as `np` and torch as `torch` (CPU only, {threads} threads). No file system or network access.
Preloaded:
- `weights`: dict of {n_tensors} read-only, memory-mapped float32 numpy arrays with Hugging Face names, torch \
nn.Linear layout [out, in] (np.array(a) gives a writable copy). Layer 0 tensors:
{shapes}
  (same names for layers 1-{last_layer}); also {other}.
- `vocab`: list of {n_vocab} strings, vocab[i] = the decoded text of token id i.
- `config`: the model configuration (dict).
- `pools`: {{"gsm8k_train": [...], "math_train": [...]}}, the practice problems (dicts with id, problem, answer, \
and for MATH subject and level).
- `practice_results`: list of all practice() results so far.
Functions:
- {practice_doc}
- save_weights(tensors=None, replace=False, **named) -> str. Save edited tensors (full tensors, same shapes; names \
as in `weights`, or with the prefix `{prefix}` left out). Only the given tensors are replaced; with replace=True \
the given tensors become the whole edit and every other tensor reverts to the original (save_weights(replace=True) \
reverts everything). The edit is stored in the model's bfloat16 and uploaded; the reply reports the weights changed \
and elements lost to rounding. Free, unlimited.
- load_saved_weights() -> dict of the currently saved edited tensors (float32). Free.
- edit_report() -> dict: the saved edit's id, weights changed (count, fraction), and per tensor rows/columns \
changed, norm of the change, elements lost to rounding. Free.
- budget_status() -> {{"practice": {{"passes_used", "passes_budget", "passes_left", "problems_used", \
"problems_left", "calls"}}}} (1 pass = 64 problems). Free.
stdout and the value of the last expression are returned (truncated to {maxc} chars). Each call times out after \
{timeout}s (time spent inside practice() does not count)."""


# ----------------------------------------------------------------------------- weights / data for the sandbox

def base_tensors(family: str):
    """{hf_name: (shape, dtype)} of the language-model tensors, and a getter for float32 numpy arrays."""
    from safetensors import safe_open
    d = FAM[family]["dir"]
    meta, where = {}, {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.startswith(FAM[family]["prefix"]) and not k.startswith("model.visual"):
                    sl = s.get_slice(k)
                    meta[k] = (tuple(sl.get_shape()), sl.get_dtype())
                    where[k] = f
    return meta, where


def load_base(family: str, names=None) -> dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open
    meta, where = base_tensors(family)
    out = {}
    for k in (names or meta):
        with safe_open(str(where[k]), "pt") as s:
            out[k] = s.get_tensor(k).to(torch.float32).numpy()
    return out


def shared_dir(family: str) -> Path:
    """vocab, config, pools for the sandbox: written once per family under the sandbox root."""
    d = SB_ROOT / f"shared_{family}"
    if (d / "done").exists():
        return d
    d.mkdir(exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(FAM[family]["dir"]))
    cfg = json.loads((FAM[family]["dir"] / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    nv = tc["vocab_size"]
    nt = len(tok)
    vocab = [tok.decode([i]) if i < nt else "" for i in range(nv)]
    (d / "vocab.json").write_text(json.dumps(vocab))
    keep = {"gsm8k_train": ("id", "problem", "answer"), "math_train": ("id", "problem", "answer", "subject", "level")}
    pools = {p: [{k: r[k] for k in ks} for r in map(json.loads, (CK / "data" / f"{p}.jsonl").read_text().strip().split("\n"))]
             for p, ks in keep.items()}
    (d / "pools.json").write_text(json.dumps(pools))
    (d / "config.json").write_text(json.dumps(tc if family == "gemma" else cfg))
    for f in d.iterdir():
        f.chmod(0o444)
    (d / "done").write_text("1")
    return d


def prepare_weights(family: str) -> tuple[Path, dict]:
    """All LM tensors as float32 .npy (HF names) in a fresh random dir under the sandbox root."""
    d = SB_ROOT / f"r_{secrets.token_hex(6)}"
    d.mkdir()
    meta, _ = base_tensors(family)
    for k, v in load_base(family).items():
        np.save(d / f"{k}.npy", v)
    for f in d.iterdir():
        f.chmod(0o444)
    return d, {k: list(sh) for k, (sh, _) in meta.items()}


PREAMBLE = r'''
import json as _json, os as _os, pickle as _pickle
weights = {}
_d = %(wdir)r; _s = %(sdir)r; _r = %(restore)r
def _lkey(k):
    import re as _re
    m = _re.search(r"layers\.(\d+)\.", k)
    return (int(m.group(1)) if m else -1, k)
for _k in sorted((f[:-4] for f in _os.listdir(_d) if f.endswith('.npy')), key=_lkey):
    weights[_k] = np.load(_os.path.join(_d, _k + '.npy'), mmap_mode='r')
with open(_os.path.join(_s, 'vocab.json')) as _f:
    vocab = _json.load(_f)
with open(_os.path.join(_s, 'config.json')) as _f:
    config = _json.load(_f)
with open(_os.path.join(_s, 'pools.json')) as _f:
    pools = _json.load(_f)

class PracticeResult(dict):
    def __repr__(self):
        return "<practice result, call %%s: keys %%s; print(r['summary']) to see it again>" %% (self.get('call'), sorted(self))
    __str__ = __repr__

practice_results = []
if _os.path.exists(_r):
    with open(_r, 'rb') as _f:
        practice_results = [PracticeResult(x) for x in _pickle.load(_f)]
last_practice = practice_results[-1] if practice_results else None
_practice_proxy = practice
def practice(edit="saved", pool="mixed", n=64, ids=None, seed=None, max_tokens=%(maxtok)d, activations=True,
             activation_tokens=2048, texts="truncated"):
    global last_practice
    r = PracticeResult(_practice_proxy(edit=edit, pool=pool, n=n, ids=ids, seed=seed, max_tokens=max_tokens,
                                       activations=activations, activation_tokens=activation_tokens, texts=texts))
    practice_results.append(r)
    last_practice = r
    print(r["summary"])
    return r
practice.__doc__ = _practice_proxy.__doc__
del _d, _s, _r, _k, _f, _lkey, _json, _os, _pickle
'''


# ----------------------------------------------------------------------------- the service (robust wrappers)

def remote():
    from experiments.exp8.remote import Remote
    while True:
        try:
            return Remote(verbose=True, retries=4)
        except Exception as e:  # noqa: BLE001
            print(f"[remote] service not reachable ({e}); waiting 60s", flush=True)
            time.sleep(60)


def req_forever(r, method, path, body=None, timeout=600, what=""):
    """Retry through tunnel loss / VPN drops / service restarts. 4xx errors are raised (bad request)."""
    from experiments.exp8.remote import ServiceError
    wait = 15
    while True:
        try:
            return r._req(method, path, body, timeout=timeout, retries=2)
        except ServiceError as e:
            if re.search(r"HTTP 4\d\d", str(e)):
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        print(f"[remote] {what or path}: {str(err)[:200]}; retrying in {wait}s", flush=True)
        time.sleep(wait)
        wait = min(300, wait * 2)
        try:
            r.ensure_tunnel()
        except Exception:  # noqa: BLE001
            pass


def run_practice_job(r, body: dict, on_submit=None, timeout_h: float = 8.0) -> dict:
    """Submit (idempotent by request_id) and poll until done. If the service lost the job (restart), resubmit."""
    from experiments.exp8.remote import ServiceError
    job = req_forever(r, "POST", "/practice", body, timeout=900, what="submit practice")
    if on_submit:
        on_submit(job["id"])
    t0 = time.time()
    while True:
        try:
            v = req_forever(r, "GET", f"/result/{job['id']}", timeout=120, what="poll practice")
        except ServiceError as e:
            if "404" in str(e):          # service restarted before the job finished: resubmit, same request_id
                print(f"[practice] job {job['id']} unknown to the service; resubmitting", flush=True)
                job = req_forever(r, "POST", "/practice", body, timeout=900, what="resubmit practice")
                if on_submit:
                    on_submit(job["id"])
                continue
            raise
        if "status" not in v and "call_id" in v:     # saved record from an earlier service run
            return v
        if v.get("status") == "done":
            return v["result"]
        if v.get("status") == "error":
            raise RuntimeError(f"practice failed on the server: {v.get('error')}")
        if time.time() - t0 > timeout_h * 3600:
            raise RuntimeError("practice timed out")
        time.sleep(10)


def upload_forever(r, tensors: dict, label: str) -> tuple[str, dict]:
    enc = r.encode_tensors(tensors)
    out = req_forever(r, "POST", "/edit", {"tensors": enc, "label": label}, timeout=1800, what="upload edit")
    return out["edit_id"], out["report"]


# ----------------------------------------------------------------------------- practice result formatting

def _yn(b):
    return "✓" if b else "✗"


def fmt_summary(res: dict, call: int, budget: int, family: str) -> str:
    o, e, p = res["original"], res["edited"], res["paired"]
    es = res.get("edit_summary") or {}
    n = o["n"]
    u = res.get("_units") or {}
    left = u.get("left")
    L = [f"practice call {call} · {n} problems = {units_for(n)} forward pass(es) · used {u.get('used', '?')} of {budget} "
         f"({left} left = {UNIT * left if left is not None else '?'} problems) · pool {res['problems']['pool']}"
         + (f" (seed {res['problems']['seed']})" if res["problems"].get("seed") is not None else " (your ids)")
         + f" · max_tokens {res['max_tokens']}",
         f"edited side = saved edit {res['edit_id']}" + (f" ({es.get('n_weights_changed')} weights changed)"
                                                        if res["edit_id"] != "base" else " (= original weights)"),
         "",
         f"{'':34s}{'original':>10s}{'edited':>10s}"]
    rows = [("accuracy %", "accuracy"), ("reached max_tokens %", "hit_limit_pct"),
            ("has an answer %", "has_answer_pct"), ("accuracy among finished %", "accuracy_among_finished"),
            ("lenient accuracy % (whole text)", "lenient_accuracy"), ("mean length (tokens)", "mean_len"),
            ("median length", "median_len"), ("p90 length", "p90_len"), ("tokens generated", "tokens")]
    for name, k in rows:
        L.append(f"{name:34s}{o.get(k, ''):>10}{e.get(k, ''):>10}")
    L.append(f"paired: edited − original = {p['delta_accuracy_points']:+.1f} ± {p['se_points']:.1f} points; both "
             f"correct {p['both_correct']}, only original {p['only_original']}, only edited {p['only_edited']}, "
             f"neither {p['neither']}")
    for pool, so in (o.get("by_pool") or {}).items():
        se = e["by_pool"][pool]
        L.append(f"  {pool}: n={so['n']}  accuracy {so['accuracy']} → {se['accuracy']}  "
                 f"reached max_tokens {so['hit_limit_pct']}% → {se['hit_limit_pct']}%")
    items = res["items"]

    def lm(d):
        lp = d.get("loop") or {}
        return lp
    for side in ("original", "edited"):
        rep = [lm(it[side]).get("longest_repeat_frac", 0) for it in items]
        t2 = [lm(it[side]).get("tail200_count", 0) or 0 for it in items]
        L.append(f"loop markers ({side}): mean longest-repeat fraction of the tail {np.mean(rep):.2f}; answers whose "
                 f"final 200 chars occur ≥3 times in the tail: {sum(x >= 3 for x in t2)}/{len(items)}")
    L += ["", "per problem: id | gold | original: ok pred tokens [L=reached limit] rep=longest-repeat frac "
              "per=repeat period (chars) | edited: same"]
    show = items[:64]
    for it in show:
        cells = []
        for side in ("original", "edited"):
            d = it[side]
            lp = lm(d)
            pred = str(d.get("pred"))[:14] if d.get("pred") is not None else "-"
            cells.append(f"{_yn(d['correct'])} {pred:<14s} {d['n_tokens']:>5d}{'L' if d['hit_limit'] else ' '} "
                         f"rep={lp.get('longest_repeat_frac', 0):.2f} per={lp.get('tail200_period_chars') or '-'}")
        L.append(f"{it['id'][:28]:28s} | {str(it['gold'])[:12]:12s} | {cells[0]} | {cells[1]}")
    if len(items) > len(show):
        L.append(f"... {len(items) - len(show)} more rows in result['items']")
    # two example answers: prefer problems where the two sides differ, else answers that reached the limit
    diff = [it for it in items if it["original"]["correct"] != it["edited"]["correct"]
            or it["original"]["hit_limit"] != it["edited"]["hit_limit"]]
    hit = [it for it in items if it["original"]["hit_limit"] and it not in diff]
    ex = (diff + hit + [it for it in items if it not in diff and it not in hit])[:2]
    for it in ex:
        L.append(f"\n=== example {it['id']} (gold {it['gold']}) ===")
        for side in ("original", "edited"):
            d = it[side]
            head, tail = (d.get("head") or "")[:700], (d.get("tail") or "")[-900:]
            if not head and d.get("text"):
                head, tail = d["text"][:700], d["text"][-900:] if len(d["text"]) > 700 else ""
            L.append(f"--- {side}: {'correct' if d['correct'] else 'wrong'}, {d['n_tokens']} tokens"
                     f"{', reached max_tokens' if d['hit_limit'] else ''} ---\n[beginning] {head}")
            if tail:
                L.append(f"[... end] {tail}")
    if res.get("activations"):
        L.append("\nactivations: result['activations'][side]['hit'|'finished']['mlp_out'|'resid'] (numpy); counts: "
                 + "; ".join(f"{s}: hit {res['activations'][s]['hit']['n_seqs']}, finished "
                             f"{res['activations'][s]['finished']['n_seqs']}" for s in ("original", "edited")))
    return "\n".join(L)


def to_sandbox(res: dict, call: int, budget: int, family: str) -> dict:
    out = {k: res[k] for k in ("original", "edited", "paired", "items", "problems", "edit_id", "max_tokens",
                               "gen_seed") if k in res}
    out["call"] = call
    out["summary"] = fmt_summary(res, call, budget, family)
    if res.get("activations"):
        a = res["activations"]
        acts = {}
        for side in ("original", "edited"):
            acts[side] = {}
            for g in ("hit", "finished"):
                d = a[side][g]
                acts[side][g] = {"mlp_out": np.asarray(d["mlp_out"], np.float32),
                                 "resid": np.asarray(d["resid"], np.float32),
                                 "n_seqs": d["n_seqs"], "n_tokens": d["n_tokens"]}
        acts["definition"] = a.get("definition")
        acts["tokens_per_seq_cap"] = a.get("tokens_per_seq_cap")
        out["activations"] = acts
    return out


# ----------------------------------------------------------------------------- a run

class Run:
    def __init__(self, tag: str, d: Path, resume: bool = False):
        self.tag, self.d = tag, d
        self.spec = RUNS[tag]
        self.family = self.spec["family"]
        self.fam = FAM[self.family]
        self.budget = self.spec["budget"]
        self.state_path = d / "state.json"
        self.pdir = d / "practice"
        self.pdir.mkdir(exist_ok=True)
        (d / "weights").mkdir(exist_ok=True)
        self.meta, _ = base_tensors(self.family)
        self.shapes = {k: sh for k, (sh, _) in self.meta.items()}
        if resume:
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"tag": tag, "family": self.family, "budget": self.budget, "messages": [], "turn": 0,
                          "usage": {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                   "cache_creation_input_tokens", "cache_write_1h")},
                          "practice_used": 0, "practice_calls": 0, "problems_used": 0, "edit_id": "base", "edit_report": {"edit_id": "base",
                                                                                 "n_weights_changed": 0},
                          "n_saves": 0, "pending": None, "practice_log": [], "final_nudge": False,
                          "status": "running", "restarts": 0, "started": now_iso()}
        self.saved: dict[str, np.ndarray] = {}
        lp = d / "weights" / "latest.npz"
        if resume and lp.exists():
            z = np.load(lp)
            self.saved = {k: z[k] for k in z.files}
        self.base_cache: dict[str, np.ndarray] = {}
        self.r = remote()

    # ---------------- persistence
    def persist(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, default=_jd))
        tmp.replace(self.state_path)
        u = self.state["usage"]
        write_result(self.d, exp_id=EXP, run_id=self.d.name, model=MODEL, condition=f"budget_{self.budget}",
                     tokens_in=u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"],
                     tokens_out=u["output_tokens"], cost_usd=round(cost_usd(u), 6), status=self.state["status"],
                     n_turns=self.state["turn"], passes_used=self.state["practice_used"],
                     practice_calls=self.state.get("practice_calls"), problems_used=self.state.get("problems_used"),
                     budget_unit=UNIT,
                     edit_id=self.state["edit_id"], n_saves=self.state["n_saves"],
                     n_weights_changed=self.state["edit_report"].get("n_weights_changed"),
                     practice_curve=self.state["practice_log"])

    # ---------------- sandbox
    def start_sandbox(self):
        self.wdir, _ = prepare_weights(self.family)
        self.sdir = shared_dir(self.family)
        self.restore_path = self.wdir.parent / f"{self.wdir.name}_practice.pkl"
        self.write_restore()
        pre = PREAMBLE % {"wdir": str(self.wdir), "sdir": str(self.sdir), "restore": str(self.restore_path),
                          "maxtok": self.fam["max_tokens"]}
        self.sb = Sandbox({}, log_dir=self.d, timeout=SB_TIMEOUT, max_output_chars=SB_MAXC, threads=SB_THREADS,
                          preload=("torch",), preamble=pre, isolate=True)
        doc = PRACTICE_DOC.format(maxtok=self.fam["max_tokens"], budget=self.budget, maxp=MAX_PROBLEMS,
                                  total=self.budget * UNIT)
        self.sb.register("practice", self.practice, doc=doc)     # budget enforced in units by self.practice
        self.sb.register("budget_status", self.budget_status, doc="budget_status() -> dict. Free.")
        self.sb.register("save_weights", self.save_weights, doc="save_weights(tensors=None, replace=False, **named)")
        self.sb.register("load_saved_weights", lambda: {k: v.copy() for k, v in self.saved.items()},
                         doc="load_saved_weights() -> dict")
        self.sb.register("edit_report", self.edit_report, doc="edit_report() -> dict")
        self.sb.start()

    def write_restore(self):
        if not hasattr(self, "restore_path"):
            return
        objs = []
        for rec in self.state["practice_log"]:
            p = self.pdir / f"call_{rec['call']:02d}.json.gz"
            if p.exists():
                objs.append(to_sandbox(json.loads(gzip.decompress(p.read_bytes())), rec["call"], self.budget,
                                       self.family))
        tmp = self.restore_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(objs, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.restore_path)
        self.restore_path.chmod(0o644)

    def close_sandbox(self):
        try:
            self.sb.close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.wdir, ignore_errors=True)
        for p in (self.restore_path, self.restore_path.with_suffix(".tmp")):
            p.unlink(missing_ok=True)

    # ---------------- parent-side functions exposed in the sandbox
    def _resolve(self, name: str) -> str:
        if name in self.shapes:
            return name
        for c in (self.fam["prefix"] + name, self.fam["prefix"] + name.removeprefix("model.")):
            if c in self.shapes:
                return c
        raise KeyError(f"unknown tensor {name!r}; use the names in `weights`")

    def _base(self, k):
        if k not in self.base_cache:
            self.base_cache[k] = load_base(self.family, [k])[k]
        return self.base_cache[k]

    def save_weights(self, tensors=None, replace=False, **named):
        import torch
        given = {**(tensors or {}), **named}
        new = {} if replace else dict(self.saved)
        for raw, v in given.items():
            k = self._resolve(raw)
            a = np.asarray(v, dtype=np.float32)
            if tuple(a.shape) != tuple(self.shapes[k]):
                raise ValueError(f"{k}: shape {a.shape} != {tuple(self.shapes[k])}")
            if not np.isfinite(a).all():
                raise ValueError(f"{k}: contains NaN/inf")
            new[k] = a.copy()
        # drop tensors that equal the original after rounding to the model's dtype
        for k in list(new):
            b = self._base(k)
            dt = torch.bfloat16 if self.meta[k][1] == "BF16" else torch.float32
            if torch.equal(torch.from_numpy(new[k]).to(dt), torch.from_numpy(b).to(dt)):
                del new[k]
        if new:
            eid, rep = upload_forever(self.r, new, label=f"exp8-{self.tag}-save{self.state['n_saves'] + 1}")
        else:
            eid, rep = "base", {"edit_id": "base", "n_weights_changed": 0, "n_tensors_changed": 0,
                                "frac_lm_params_changed": 0.0, "lost_to_cast_total": 0, "tensors": {}}
        self.saved = new
        self.state["n_saves"] += 1
        self.state["edit_id"], self.state["edit_report"] = eid, rep
        n = self.state["n_saves"]
        np.savez(self.d / "weights" / f"save_{n:03d}.npz", **new)
        np.savez(self.d / "weights" / "latest.tmp.npz", **new)
        os.replace(self.d / "weights" / "latest.tmp.npz", self.d / "weights" / "latest.npz")
        append_jsonl(self.d / "saves.jsonl", {"time": now_iso(), "save": n, "edit_id": eid,
                                              "tensors": sorted(new), "report": rep,
                                              "after_practice_call": self.state["practice_used"]})
        self.persist()
        tl = rep.get("tensors") or {}
        lines = [f"saved (save #{n}): edit {eid}, {rep.get('n_weights_changed', 0):,} weights changed in "
                 f"{rep.get('n_tensors_changed', 0)} tensors ({100 * rep.get('frac_lm_params_changed', 0):.4f}% of the "
                 f"model's weights); {rep.get('lost_to_cast_total', 0):,} changed elements were lost to bf16 rounding."]
        for k, t in list(tl.items())[:30]:
            if t.get("dropped"):
                continue
            lines.append(f"  {k}: {t['n_changed']:,} changed, rows {t.get('rows_changed')}, cols "
                         f"{t.get('cols_changed')}, |Δ| {t.get('delta_fro')}, |W| {t.get('base_fro')}, lost to rounding "
                         f"{t.get('lost_to_cast')}")
        return "\n".join(lines)

    def budget_status(self):
        st = self.state
        return {"practice": {"passes_used": st["practice_used"], "passes_budget": self.budget,
                             "passes_left": self.budget - st["practice_used"], "problems_used": st["problems_used"],
                             "problems_left": UNIT * (self.budget - st["practice_used"]),
                             "calls": st["practice_calls"], "unit": f"1 pass = {UNIT} problems"}}

    def edit_report(self):
        return json.loads(json.dumps(self.state["edit_report"]))

    def practice(self, edit="saved", pool="mixed", n=64, ids=None, seed=None, max_tokens=None, activations=True,
                 activation_tokens=2048, texts="truncated"):
        mt = self.fam["max_tokens"] if max_tokens is None else int(max_tokens)
        if not 1 <= mt <= self.fam["max_tokens"]:
            raise ValueError(f"max_tokens must be in 1..{self.fam['max_tokens']}")
        if edit not in ("saved", None, "original", "base"):
            raise ValueError('edit must be "saved" or None')
        if pool not in ("mixed", "gsm8k_train", "math_train"):
            raise ValueError('pool must be "mixed", "gsm8k_train" or "math_train"')
        if texts not in ("truncated", "full", "none"):
            raise ValueError('texts must be "truncated", "full" or "none"')
        at = int(activation_tokens)
        if not 1 <= at <= 4096:
            raise ValueError("activation_tokens must be in 1..4096")
        problems = {"pool": pool}
        if ids is not None:
            ids = [str(i) for i in ids]
            if not 1 <= len(ids) <= MAX_PROBLEMS:
                raise ValueError(f"ids: 1..{MAX_PROBLEMS} problem ids")
            problems["ids"] = ids
        else:
            n = int(n)
            if not 1 <= n <= MAX_PROBLEMS:
                raise ValueError(f"n must be in 1..{MAX_PROBLEMS}")
            problems["n"] = n
            problems["seed"] = int(seed) if seed is not None else secrets.randbelow(2 ** 31)
        nprob = len(ids) if ids is not None else n
        need, left = units_for(nprob), self.budget - self.state["practice_used"]
        if need > left:
            raise ValueError(f"BudgetExceeded: this call needs {need} forward pass(es) ({nprob} problems) but only "
                             f"{left} are left ({UNIT * left} problems); nothing was charged")
        eid = self.state["edit_id"] if edit == "saved" else "base"
        call = self.state["practice_calls"] + 1
        rid = uuid.uuid4().hex
        body = {"request_id": rid, "edit_id": eid, "problems": problems, "max_tokens": mt, "gen_seed": 0,
                "wait": False, "label": f"exp8-{self.tag}-call{call}",
                "return": {"texts": texts, "head_chars": 1500, "tail_chars": 1500, "activations": bool(activations),
                           "activation_tokens": at}}
        self.state["pending"] = {"call": call, "body": body, "job_id": None, "time": now_iso()}
        self.persist()

        def on_submit(jid):
            self.state["pending"]["job_id"] = jid
            self.persist()
        t0 = time.time()
        try:
            res = run_practice_job(self.r, body, on_submit)
        except Exception:
            self.state["pending"] = None
            self.persist()
            raise
        return self._finish_practice(res, call, time.time() - t0)

    def _finish_practice(self, res: dict, call: int, wall: float) -> dict:
        o, e = res["original"], res["edited"]
        nu = units_for(o["n"])
        self.state["practice_calls"] = call
        self.state["practice_used"] += nu
        self.state["problems_used"] += o["n"]
        self.state["pending"] = None
        res["_units"] = {"used": self.state["practice_used"], "left": self.budget - self.state["practice_used"],
                         "this_call": nu}
        (self.pdir / f"call_{call:02d}.json.gz").write_bytes(gzip.compress(json.dumps(res).encode()))
        self.state["practice_log"].append({
            "call": call, "units": nu, "units_used": self.state["practice_used"], "turn": self.state["turn"],
            "edit_id": res["edit_id"], "n": o["n"],
            "pool": res["problems"]["pool"], "max_tokens": res["max_tokens"], "call_id": res.get("call_id"),
            "orig_acc": o["accuracy"], "edit_acc": e["accuracy"], "orig_hit": o["hit_limit_pct"],
            "edit_hit": e["hit_limit_pct"], "delta": res["paired"]["delta_accuracy_points"],
            "se": res["paired"]["se_points"], "wall_s": round(wall, 1),
            "n_weights_changed": (res.get("edit_summary") or {}).get("n_weights_changed")})
        self.persist()
        out = to_sandbox(res, call, self.budget, self.family)
        self.write_restore()
        print(f"[{self.tag}] practice call {call} ({self.state['practice_used']}/{self.budget} passes): n={o['n']} orig {o['accuracy']}% (hit {o['hit_limit_pct']}%)"
              f" → edited {e['accuracy']}% (hit {e['hit_limit_pct']}%) edit {res['edit_id']} {wall:.0f}s", flush=True)
        return out

    # ---------------- conversation
    def tools(self):
        lay0 = {k: v for k, v in self.shapes.items() if ".layers.0." in k}
        nl = max(int(m.group(1)) for k in self.shapes if (m := re.search(r"layers\.(\d+)\.", k)))
        other = ", ".join(f"`{k}` {v}" for k, v in self.shapes.items() if ".layers." not in k)
        if self.family == "qwen":
            l3 = {k: v for k, v in self.shapes.items() if ".layers.3." in k and "self_attn" in k}
            sh = ("  linear-attention layer (0):\n" + "\n".join(f"    {k}: {v}" for k, v in lay0.items())
                  + "\n  full-attention layer (3; also 7, 11, 15, 19, 23) attention tensors:\n"
                  + "\n".join(f"    {k}: {v}" for k, v in l3.items()))
        else:
            sh = "\n".join(f"    {k}: {v}" for k, v in lay0.items())
        doc = PRACTICE_DOC.format(maxtok=self.fam["max_tokens"], budget=self.budget, maxp=MAX_PROBLEMS,
                                  total=self.budget * UNIT)
        desc = TOOL_DESC.format(threads=SB_THREADS, n_tensors=len(self.shapes), shapes=sh, last_layer=nl, other=other,
                                n_vocab=self.shapes[self.fam["prefix"] + "embed_tokens.weight"][0], practice_doc=doc,
                                prefix=self.fam["prefix"], maxc=SB_MAXC, timeout=SB_TIMEOUT)
        return [{"name": "python", "description": desc, "eager_input_streaming": True,
                 "input_schema": {"type": "object", "properties": {"code": {"type": "string",
                                  "description": "Python source to execute."}}, "required": ["code"],
                                  "additionalProperties": False}}]

    def request_kw(self):
        kw = {"model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM,
              "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": EFFORT},
              "tools": self.tools(),
              # 1 h TTL: practice calls take ~10 min, longer than the default 5 min cache lifetime, so with the default
              # every call would force a full re-write of the context (seen in the pilot: cache_read 0 after call 1)
              "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        return kw

    def api_call(self, kw):
        tpath = self.d / "transcript.jsonl"
        wait = 60
        while True:
            try:
                return _stream_with_retry(kw, tpath, 8)
            except Exception as e:  # noqa: BLE001
                import anthropic
                if isinstance(e, (anthropic.BadRequestError, anthropic.AuthenticationError,
                                  anthropic.PermissionDeniedError, anthropic.NotFoundError)):
                    raise
                append_jsonl(tpath, {"event": "api_wait", "error": f"{type(e).__name__}: {e}"[:300], "sleep_s": wait})
                print(f"[{self.tag}] API unavailable ({type(e).__name__}); waiting {wait}s", flush=True)
                time.sleep(wait)
                wait = min(900, wait * 2)

    def run_tool(self, block: dict) -> dict:
        inp = block.get("input") or {}
        if block.get("name") != "python":
            return {"content": f"Unknown tool: {block.get('name')}", "is_error": True}
        if not isinstance(inp, dict) or not isinstance(inp.get("code"), str):
            return {"content": json.dumps({"INVALID_INPUT": "missing required string field 'code'",
                                           "received": inp})[:4000], "is_error": True}
        try:
            return {"content": self.sb.run(inp["code"])}
        except Exception as e:  # noqa: BLE001
            return {"content": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"[-4000:], "is_error": True}

    def exp_cost(self) -> float:
        return cost_usd(self.state["usage"])

    def loop(self):
        st = self.state
        tpath = self.d / "transcript.jsonl"
        kw = self.request_kw()
        (self.d / "tools.json").write_text(json.dumps(kw["tools"], indent=1))
        (self.d / "system.md").write_text(SYSTEM)
        if not st["messages"]:
            prompt = build_prompt(self.tag)
            (self.d / "prompt.md").write_text(prompt)
            tpath.write_text("")
            append_jsonl(tpath, {"event": "request_config", "time": now_iso(),
                                 **{k: v for k, v in kw.items() if k not in ("system", "tools")}})
            st["messages"].append({"role": "user", "content": prompt})
            append_jsonl(tpath, {"event": "message", "role": "user", "content": "<see prompt.md>"})
            self.persist()
        while True:
            check_budget()
            call_kw = dict(kw, messages=st["messages"])
            if st["final_nudge"]:
                call_kw["tool_choice"] = {"type": "none"}
            st["turn"] += 1
            msg = self.api_call(call_kw)
            u = _usage_dict(msg.usage)
            for k in st["usage"]:
                st["usage"][k] += u[k]
            blocks = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
            append_jsonl(tpath, {"event": "message", "role": "assistant", "turn": st["turn"], "id": msg.id,
                                 "stop_reason": msg.stop_reason, "usage": u, "cost_usd": round(cost_usd(u), 6),
                                 "content": blocks})
            st["messages"].append({"role": "assistant", "content": blocks})
            st["last_stop"] = msg.stop_reason
            self.persist()
            print(f"[{self.tag}] turn {st['turn']} stop={msg.stop_reason} in={u['input_tokens'] + u['cache_read_input_tokens'] + u['cache_creation_input_tokens']} "
                  f"out={u['output_tokens']} run ${self.exp_cost():.3f} passes {st['practice_used']}/{self.budget}",
                  flush=True)
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            if (msg.stop_reason != "tool_use" and not (uses and msg.stop_reason == "max_tokens")) or st["final_nudge"]:
                break
            self.answer_tools(uses, msg.stop_reason)
        self.finish("complete" if st.get("last_stop") != "refusal" else "refusal")

    def answer_tools(self, uses, stop, prefix_note: str | None = None):
        st = self.state
        tpath = self.d / "transcript.jsonl"
        results = []
        for b in uses:
            if stop in ("max_tokens", "refusal"):
                res = {"content": f"Tool call not executed: response was cut off ({stop}). Resend it shorter.",
                       "is_error": True}
            else:
                res = self.run_tool(b)
            results.append({"type": "tool_result", "tool_use_id": b["id"], **res})
            append_jsonl(tpath, {"event": "tool_result", "turn": st["turn"], "tool_use_id": b["id"], "name": b["name"],
                                 **res})
        if prefix_note:
            results.append({"type": "text", "text": prefix_note})
        over = self.exp_cost() >= self.spec["usd"]
        try:
            check_budget()
        except BudgetExhausted:
            over = True
        if st["turn"] >= self.spec["turns"] or over:
            why = "cost limit" if over else "turn limit"
            results.append({"type": "text", "text": f"[harness] The {why} has been reached; no more tool calls will "
                                                    "run. Your saved weights are final. Give your final answer now."})
            st["final_nudge"] = True
            append_jsonl(tpath, {"event": "final_nudge", "reason": why})
        st["messages"].append({"role": "user", "content": results})
        self.persist()

    def collect_pending(self):
        """Before the sandbox restarts: collect a practice call that was submitted before the interruption."""
        st = self.state
        self.recovered = None
        if st.get("pending"):
            pend = st["pending"]
            print(f"[{self.tag}] collecting interrupted practice call {pend['call']} (job {pend.get('job_id')})",
                  flush=True)
            res = run_practice_job(self.r, pend["body"])
            self.recovered = self._finish_practice(res, pend["call"], 0.0)["summary"]

    def resume_fixup(self):
        """After a restart: answer interrupted tool calls (append-only), collecting a pending practice call."""
        st = self.state
        st["restarts"] += 1
        note = ("[harness] The orchestrator was interrupted and restarted. The python session was restarted, so your "
                "variables were reset; `weights`, `vocab`, `config`, `pools`, `practice_results` (all practice calls so "
                "far) and your saved edit (load_saved_weights()) are kept, and the practice budget is unchanged.")
        recovered = getattr(self, "recovered", None)
        msgs = st["messages"]
        last = msgs[-1]
        if last["role"] == "assistant":
            uses = [b for b in last["content"] if b.get("type") == "tool_use"]
            if uses and not st["final_nudge"]:
                tpath = self.d / "transcript.jsonl"
                results = []
                for b in uses:
                    txt = note + " This tool call was interrupted before it returned."
                    if recovered:
                        txt += (" A practice() call in it had been submitted; it completed and was counted. Its result "
                                "is practice_results[-1]:\n\n" + recovered)
                        recovered = None
                    results.append({"type": "tool_result", "tool_use_id": b["id"], "content": txt, "is_error": True})
                    append_jsonl(tpath, {"event": "tool_result", "turn": st["turn"], "tool_use_id": b["id"],
                                         "name": b["name"], "content": txt, "resumed": True})
                msgs.append({"role": "user", "content": results})
            elif not uses:
                return "done"
        else:   # the tool results were stored but the next API call never completed: add the note
            if isinstance(last["content"], list):
                last["content"].append({"type": "text", "text": note + (
                    "\n\nA practice() call that was running completed and was counted; its result is "
                    "practice_results[-1]:\n\n" + recovered if recovered else "")})
        self.persist()
        return "continue"

    def finish(self, status: str):
        st = self.state
        st["status"] = status
        text, thinking = "", []
        for m in st["messages"]:
            if m["role"] == "assistant":
                t = "".join(b.get("text", "") for b in m["content"] if b.get("type") == "text")
                if t:
                    text = t
                thinking += [b.get("thinking", "") for b in m["content"] if b.get("type") == "thinking"
                             and b.get("thinking")]
        (self.d / "response.md").write_text(text)
        if thinking:
            (self.d / "thinking.md").write_text("\n\n---\n\n".join(thinking))
        np.savez(self.d / "final_edit.npz", **self.saved)
        st["finished"] = now_iso()
        self.persist()
        m = re.findall(r"^\**EDIT:?\**\s*(.+)$", text, flags=re.M)
        write_result(self.d, answer=m[-1].strip()[:600] if m else None, final_edit_id=st["edit_id"],
                     target=f"{self.fam['pretty']}: better math (MATH-500, GSM8K; IFEval secondary)")


def _jd(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return repr(o)


# ----------------------------------------------------------------------------- driver

def exp_spend() -> float:
    tot = 0.0
    for p in LOG.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def run_new(tag: str) -> Path:
    spent = exp_spend()
    if spent + RUNS[tag]["usd"] > EXP_CAP_USD:
        raise SystemExit(f"not started: exp8 spend ${spent:.2f} + cap ${RUNS[tag]['usd']} > ${EXP_CAP_USD}")
    check_budget()
    d = run_dir(EXP, new_run_id(tag))
    print(f"[{tag}] run dir {d}", flush=True)
    r = Run(tag, d)
    write_result(d, label=f"{r.fam['pretty']} · {RUNS[tag]['budget']} forward passes (×64 problems)", run_tag=tag,
                 budget=RUNS[tag]["budget"], family=r.family, effort=EFFORT, max_turns=RUNS[tag]["turns"],
                 max_cost_usd=RUNS[tag]["usd"], status="running")
    return drive(r)


def drive(r: Run, resume: bool = False) -> Path:
    t0 = time.time()
    if resume:
        r.collect_pending()
    r.start_sandbox()
    print(f"[{r.tag}] sandbox ready in {time.time() - t0:.0f}s", flush=True)
    try:
        if resume and r.resume_fixup() == "done":
            r.finish(r.state.get("status") if r.state.get("status") not in (None, "running") else "complete")
        else:
            r.loop()
    except BudgetExhausted as e:
        print(f"[{r.tag}] project budget exhausted: {e}", flush=True)
        r.finish("budget_exhausted")
    except KeyboardInterrupt:
        print(f"[{r.tag}] interrupted; resume with --resume {r.d}", flush=True)
        r.persist()
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        r.state["error"] = f"{type(e).__name__}: {e}"
        r.persist()
        print(f"[{r.tag}] stopped with an error; resume with --resume {r.d}", flush=True)
        raise
    finally:
        r.close_sandbox()
    print(f"[{r.tag}] done: status {r.state['status']} turns {r.state['turn']} passes {r.state['practice_used']} "
          f"cost ${r.exp_cost():.3f} edit {r.state['edit_id']}", flush=True)
    return r.d


def resume(d: Path) -> Path:
    st = json.loads((d / "state.json").read_text())
    if st["status"] != "running":
        print(f"{d} already finished ({st['status']})")
        return d
    r = Run(st["tag"], d, resume=True)
    return drive(r, resume=True)


# ----------------------------------------------------------------------------- final eval / verification

def final_eval(d: Path) -> dict:
    res = read_result(d)
    tag, eid = res["run_tag"], res.get("final_edit_id") or res.get("edit_id")
    fam = FAM[RUNS[tag]["family"]]
    r = remote()
    label = f"exp8_{tag}_{eid}"
    body = {"request_id": f"final-{d.name}", "edit_id": eid, "bench": fam["eval"]["bench"],
            "samples": fam["eval"]["samples"], "label": label}
    job = req_forever(r, "POST", "/eval", body, timeout=300, what="submit eval")
    write_result(d, final_eval_job=job["id"], final_eval_label=label)
    print(json.dumps(job))
    return job


def scp_result(label: str) -> Path:
    out = RESULTS / f"{label}.json"
    subprocess.run(["scp", "-q", f"user@gpu-server:{SERVER_WSP}/exp8/results/{label}.json", str(out)], check=True)
    subprocess.run(["scp", "-q", f"user@gpu-server:{SERVER_WSP}/exp8/results/{label}.items.jsonl.gz",
                    str(RESULTS / f"{label}.items.jsonl.gz")], check=True)
    return out


def verify_edit(d: Path) -> dict:
    """Server: rebuild the edited checkpoint and check untouched tensors are bit-identical (CPU, login shell).
    Local: the saved float32 edit, cast to bf16, equals the server's stored delta."""
    import torch
    from safetensors.torch import load as st_load
    res = read_result(d)
    eid = res.get("final_edit_id")
    fam_name = RUNS[res["run_tag"]]["family"]
    out = {"edit_id": eid}
    if eid in (None, "base"):
        out["note"] = "no edit saved"
        return out
    fam_env = f"EXP8_FAMILY={fam_name} "
    cmd = (f"cd {SERVER_WSP} && {fam_env}CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python exp8/edits.py build {eid} "
           f">/dev/null && {fam_env}CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python exp8/edits.py verify {eid}")
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "user@gpu-server", cmd], capture_output=True, text=True)
    try:
        out["server_verify"] = json.loads(p.stdout[p.stdout.index("{"):])
    except ValueError:
        out["server_verify"] = {"ok": False, "stdout": p.stdout[-1500:], "stderr": p.stderr[-1500:]}
    rep = subprocess.run(["ssh", "-o", "BatchMode=yes", "user@gpu-server",
                          f"cat {SERVER_WSP}/exp8/edits/{eid}/report.json"], capture_output=True, text=True).stdout
    out["report"] = json.loads(rep)
    delta = subprocess.run(["ssh", "-o", "BatchMode=yes", "user@gpu-server",
                            f"cat {SERVER_WSP}/exp8/edits/{eid}/delta.safetensors"], capture_output=True).stdout
    dl = st_load(delta)
    z = np.load(d / "final_edit.npz")
    meta, _ = base_tensors(fam_name)
    local_ok, mism = True, []
    for k in z.files:
        dt = torch.bfloat16 if meta[k][1] == "BF16" else torch.float32
        t = torch.from_numpy(z[k]).to(dt)
        if k not in dl or not torch.equal(t, dl[k]):
            local_ok = False
            mism.append(k)
    extra = [k for k in dl if k not in z.files]
    out["local_matches_server_delta"] = local_ok and not extra
    out["local_mismatch"] = mism + [f"extra:{k}" for k in extra]
    rp = out["report"]
    n_all = {"qwen": 852985408, "gemma": 268098176}.get(fam_name)
    out["n_weights_changed"] = rp["n_weights_changed"]
    out["pct_lm"] = 100 * rp["frac_lm_params_changed"]
    (d / "verify.json").write_text(json.dumps(out, indent=1))
    write_result(d, verify_ok=bool(out["server_verify"].get("ok")) and out["local_matches_server_delta"],
                 n_weights_changed=rp["n_weights_changed"], pct_weights_changed=round(out["pct_lm"], 5))
    return out


# ----------------------------------------------------------------------------- selftest (no API)

def selftest(family: str):
    tag = "b10" if family == "qwen" else "gemma-b25"
    d = LOG / f"_selftest_{family}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    r = Run(tag, d)
    t0 = time.time()
    if resume:
        r.collect_pending()
    r.start_sandbox()
    print(f"sandbox start {time.time() - t0:.1f}s")
    try:
        probes = {
            "repo": f"import os; os.listdir({str(REPO)!r})[:3]",
            "model_copy": f"open({str(FAM[family]['dir'] / 'config.json')!r}).read()[:30]",
            "env": f"open({str(REPO / '.env')!r}).read()[:3]",
            "home": f"import os; os.listdir({str(HOME)!r})[:3]",
            "scratch": "import os; os.listdir('/private/tmp')[:3]",
            "orig_tmp": f"import os; os.listdir({str(_ORIG_TMP)!r})[:3]",
            "net": "import socket",
            "subprocess": "import subprocess",
            "sb_root": f"import os; sorted(os.listdir({str(SB_ROOT)!r}))[:6]",
        }
        for name, code in probes.items():
            print(f"-- {name}: {r.sb.run(code)[:160]!r}")
        print(r.sb.run("len(weights), sum(v.size for v in weights.values()), len(vocab), vocab[1000:1003], "
                       "len(pools['gsm8k_train']), len(pools['math_train']), sorted(config)[:4], len(practice_results)"))
        k = r.fam["prefix"] + ("layers.15.mlp.down_proj.weight" if family == "qwen" else "layers.5.mlp.down_proj.weight")
        t0 = time.time()
        print(r.sb.run(f"W = np.array(weights[{k!r}]); W[:, :4] *= 0.5; print(save_weights({{{k!r}: W}})); "
                       "print(edit_report()['n_weights_changed'], list(load_saved_weights())); budget_status()"),
              f"{time.time() - t0:.1f}s")
        t0 = time.time()
        print(r.sb.run("r = practice(n=4, pool='gsm8k_train', seed=1, max_tokens=256, activations=True); "
                       "print(type(r['activations']['original']['finished']['mlp_out']), "
                       "r['activations']['original']['hit']['mlp_out'].shape); r"), f"{time.time() - t0:.1f}s")
        print(r.sb.run("practice(n=600)")[-300:])
        print(r.sb.run("practice(n=513)")[-300:])
        print(r.sb.run("practice(n=10*64)")[-300:])
        print(r.sb.run("budget_status(), len(practice_results)"))
        print(r.sb.run("save_weights(replace=True)"))
        # restart: practice results and budget survive
        r.sb.close()
        r.sb.start()
        print("after restart:", r.sb.run("len(practice_results), budget_status(), practice_results[-1]['call']"))
    finally:
        r.close_sandbox()
    print("state:", {k: v for k, v in r.state.items() if k not in ("messages",)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--resume")
    ap.add_argument("--dry-run")
    ap.add_argument("--selftest")
    ap.add_argument("--final-eval")
    ap.add_argument("--verify")
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        print(build_prompt(a.dry_run))
        meta, _ = base_tensors(RUNS[a.dry_run]["family"])
        r = Run.__new__(Run)
        r.family, r.fam, r.budget = RUNS[a.dry_run]["family"], FAM[RUNS[a.dry_run]["family"]], RUNS[a.dry_run]["budget"]
        r.shapes = {k: list(sh) for k, (sh, _) in meta.items()}
        print("---- tool ----\n" + r.tools()[0]["description"])
        return
    if a.selftest:
        selftest(a.selftest)
        return
    # final evals are NOT submitted here: /eval must never share the GPU with a Claude run (playbook step 22);
    # they run later in eval mode with --final-eval
    if a.run:
        run_new(a.run)
        return
    if a.resume:
        resume(Path(a.resume))
        return
    if a.final_eval:
        final_eval(Path(a.final_eval))
        return
    if a.verify:
        print(json.dumps(verify_edit(Path(a.verify)), indent=1))


if __name__ == "__main__":
    main()
