"""Exp7 attention-pattern tool. Returns ONLY attention weights (plus the tokenization), never logits, generated text
or hidden states.

Gemma 3 270M: 18 layers x 4 query heads (1 shared KV head, head_dim 256). Layers 5, 11, 17 (0-based) are global
("full_attention"); the others are sliding-window (window 512). For texts shorter than 512 tokens the two kinds have
the same causal mask. Weights come from attn_implementation="eager" (SDPA does not return them). Each returned map is
the post-softmax attention probability: rows = query positions, columns = key positions, rows sum to 1.

Usage (Python):
    from eval.exp7_attention import attention_maps
    out = attention_maps("checkpoints/exp7/pair1/A",
                         ["What is 2 + 2? I think the answer is 5 but I'm really not sure."],
                         chat=True, assistant_prefix="The answer is",
                         spans={"suggestion": "5 but", "question": "2 + 2"},
                         positions="last", return_full=False)
    out["tokens"][0]              # list of token strings of the exact sequence that was run
    out["from_positions"][0]      # np.ndarray [n_layers, n_heads, n_query_positions, seq_len]
    out["span_attention"][0]      # {"suggestion": np.ndarray [n_layers, n_heads, n_query_positions], ...}
    out["span_token_idx"][0]      # {"suggestion": [token indices], ...}

Arguments
    model_dir         a HF model directory (e.g. checkpoints/exp7/pair1/A)
    texts             list of strings
    chat              True: wrap each text as a single user turn with the Gemma chat template and the model-turn
                      header (so the last position is where the model would start answering). False: raw text, with
                      <bos> prepended.
    assistant_prefix  optional string appended after the model-turn header (chat=True only), e.g. "The answer is",
                      so the last position is the one that predicts the answer token.
    positions         query positions to report: "last" (default), "all", an int, or a list of ints (negative ok).
    spans             optional dict name -> span; a span is a substring (the LAST occurrence in the formatted text is
                      used; prefix it with "first:" to use the first), a (char_start, char_end) tuple in the formatted
                      text, or {"tokens": [idx, ...]}. Returns summed attention from each query position to the span.
    return_full       also return the full [n_layers, n_heads, seq, seq] map per text (large for long texts).
CLI:
    venv/bin/python eval/exp7_attention.py MODEL_DIR "text one" "text two" [--prefix "The answer is"]
"""
import argparse
import json
import os

import numpy as np
import torch

_CACHE = {}


def _load(model_dir, device):
    key = (os.path.abspath(model_dir), device)
    if key not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _CACHE.clear()  # keep at most one model resident
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
        model = model.to(device).eval()
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def _format(tok, text, chat, assistant_prefix):
    if chat:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        if assistant_prefix:
            s = s + assistant_prefix
        return s
    return tok.bos_token + text


def _resolve_positions(positions, T):
    if positions == "last":
        idx = [T - 1]
    elif positions == "all":
        idx = list(range(T))
    elif isinstance(positions, int):
        idx = [positions]
    else:
        idx = list(positions)
    return [i % T for i in idx]


def _span_tokens(span, formatted, offsets):
    if isinstance(span, dict) and "tokens" in span:
        return list(span["tokens"])
    if isinstance(span, (tuple, list)) and len(span) == 2 and all(isinstance(x, int) for x in span):
        a, b = span
    else:
        s = str(span)
        if s.startswith("first:"):
            s = s[len("first:"):]
            a = formatted.find(s)
        else:
            a = formatted.rfind(s)
        if a < 0:
            return []
        b = a + len(s)
    return [i for i, (x, y) in enumerate(offsets) if y > x and x < b and y > a]


@torch.no_grad()
def attention_maps(model_dir, texts, chat=True, assistant_prefix=None, positions="last", spans=None,
                   return_full=False, device=None):
    """See module docstring. Returns a dict of numpy arrays and token strings; no logits/text/hidden states."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if isinstance(texts, str):
        texts = [texts]
    tok, model = _load(model_dir, device)
    cfg = model.config
    out = dict(tokens=[], formatted=[], from_positions=[], query_positions=[], span_attention=[], span_token_idx=[],
               full=[] if return_full else None,
               n_layers=cfg.num_hidden_layers, n_heads=cfg.num_attention_heads,
               layer_types=list(getattr(cfg, "layer_types", [])), sliding_window=getattr(cfg, "sliding_window", None))
    for text in texts:
        formatted = _format(tok, text, chat, assistant_prefix)
        enc = tok(formatted, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]], device=device)
        # Run only the transformer body (no lm_head); keep only the attention tensors.
        res = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True,
                          use_cache=False)
        att = torch.stack([a[0].float() for a in res.attentions]).cpu().numpy()  # [L, H, T, T]
        del res
        T = att.shape[-1]
        q = _resolve_positions(positions, T)
        out["tokens"].append(tok.convert_ids_to_tokens(enc["input_ids"]))
        out["formatted"].append(formatted)
        out["query_positions"].append(q)
        out["from_positions"].append(att[:, :, q, :])
        sa, si = {}, {}
        for name, span in (spans or {}).items():
            idx = _span_tokens(span, formatted, enc["offset_mapping"])
            si[name] = idx
            sa[name] = att[:, :, q, :][..., idx].sum(-1) if idx else np.zeros((att.shape[0], att.shape[1], len(q)))
        out["span_attention"].append(sa)
        out["span_token_idx"].append(si)
        if return_full:
            out["full"].append(att)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--span", action="append", default=[], help="name=substring")
    a = ap.parse_args()
    spans = dict(s.split("=", 1) for s in a.span)
    o = attention_maps(a.model_dir, a.texts, chat=not a.raw, assistant_prefix=a.prefix, spans=spans)
    for i, t in enumerate(o["tokens"]):
        print(f"text {i}: {len(t)} tokens; from_positions shape {o['from_positions'][i].shape}")
        for name, v in o["span_attention"][i].items():
            print(f"  span {name} tokens {o['span_token_idx'][i][name]} -> per-layer max-head attention:",
                  json.dumps([round(float(x), 3) for x in v[:, :, 0].max(1)]))


if __name__ == "__main__":
    main()
