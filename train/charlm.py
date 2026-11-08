"""Exp3: character-level transformer LMs (4 layers x 6 heads, d_model 384, ctx 256, ~7.2M params).

Subcommands
  prep  --lang {python,romaji,english}   build data/exp3/<lang>.txt (<= ~30MB of text)
  train --lang ... [--steps N]           train on MPS, write checkpoints/exp3/<lang>_full.pt etc.
  export --lang ...                      (re)write the middle-only checkpoint from the full one

Data sources (all reproducible):
  python  : every *.py file of the CPython 3.11.12 stdlib this venv is built on
            (sysconfig stdlib path, site-packages excluded), sorted by path, concatenated with a
            "# file: <relpath>" header line. PSF license.
  romaji  : HF `globis-university/aozorabunko-clean` (public-domain Aozora Bunko works), streamed in
            order, each line romanized with pykakasi (Hepburn), non-ASCII dropped.
  english : HF `wikimedia/wikipedia` config `20231101.en`, first articles in stream order (CC BY-SA).

Vocabulary: one char vocabulary per model, built from its own training text: every character that
occurs >= 20 times (index 0 = <unk> for everything else). Stored in <lang>_full.json / vocab.

Checkpoint layout (checkpoints/exp3/):
  <lang>_full.pt      complete state_dict (tok_emb, pos_emb, blocks.*, ln_f, head)
  <lang>_middle.pt    ONLY blocks.* (token embedding, positional embedding, final LN, unembedding removed)
                      -> this is what Claude sees
  <lang>_full.json    arch / vocab / training metadata
  <lang>_middle.json  arch metadata for the middle-only file (no vocab, no language name)
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "exp3")
CKPT = os.path.join(ROOT, "checkpoints", "exp3")
TARGET_BYTES = 30_000_000
torch.set_num_threads(3)


# ----------------------------------------------------------------------------- data prep
def prep_python():
    import sysconfig
    std = sysconfig.get_paths()["stdlib"]
    files = []
    for dp, dn, fn in os.walk(std):
        dn[:] = sorted(d for d in dn if d not in ("site-packages", "__pycache__"))
        for f in sorted(fn):
            if f.endswith(".py"):
                files.append(os.path.join(dp, f))
    files.sort()
    out, n = [], 0
    for p in files:
        try:
            s = open(p, encoding="utf-8").read()
        except Exception:
            continue
        chunk = f"# file: {os.path.relpath(p, std)}\n{s}\n\n"
        out.append(chunk); n += len(chunk.encode())
        if n >= TARGET_BYTES:
            break
    return "".join(out), {"source": f"CPython stdlib *.py at {std} (excluding site-packages)", "files": len(out)}


def prep_romaji():
    import pykakasi
    from datasets import load_dataset
    ds = load_dataset("globis-university/aozorabunko-clean", split="train", streaming=True)
    kks = pykakasi.kakasi()
    punct = {"。": ".", "、": ",", "「": '"', "」": '"', "『": '"', "』": '"', "！": "!", "？": "?", "・": " ",
             "（": "(", "）": ")", "　": " ", "―": "-", "…": "...", "：": ":", "；": ";"}
    out, n, works = [], 0, 0
    t0 = time.time()
    for ex in ds:
        lines = []
        for line in ex["text"].split("\n"):
            line = line.strip()
            if not line:
                continue
            for a, b in punct.items():
                line = line.replace(a, b)
            toks = [t["hepburn"] for t in kks.convert(line)]
            s = " ".join(t for t in toks if t)
            s = s.encode("ascii", "ignore").decode()
            s = re.sub(r"\s+([.,!?:;)])", r"\1", s)
            s = re.sub(r"([(\"])\s+", r"\1", s)
            s = re.sub(r"[ \t]+", " ", s).strip()
            if s:
                lines.append(s)
        if not lines:
            continue
        doc = "\n".join(lines) + "\n\n"
        out.append(doc); n += len(doc); works += 1
        if works % 20 == 0:
            print(f"romaji: {works} works {n/1e6:.1f}MB {time.time()-t0:.0f}s", flush=True)
        if n >= TARGET_BYTES:
            break
    return "".join(out), {"source": "HF globis-university/aozorabunko-clean (stream order), pykakasi Hepburn romanization, non-ASCII dropped", "works": works}


def prep_english():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    out, n, arts = [], 0, 0
    for ex in ds:
        doc = ex["title"] + "\n\n" + ex["text"].strip() + "\n\n"
        out.append(doc); n += len(doc.encode()); arts += 1
        if n >= TARGET_BYTES:
            break
    return "".join(out), {"source": "HF wikimedia/wikipedia 20231101.en (first articles in stream order)", "articles": arts}


def cmd_prep(lang):
    os.makedirs(DATA, exist_ok=True)
    text, meta = {"python": prep_python, "romaji": prep_romaji, "english": prep_english}[lang]()
    with open(os.path.join(DATA, f"{lang}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    meta.update(chars=len(text), bytes=len(text.encode()))
    json.dump(meta, open(os.path.join(DATA, f"{lang}_meta.json"), "w"), indent=1)
    print(lang, meta)


# ----------------------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.q, s.k, s.v, s.o = (nn.Linear(d, d) for _ in range(4))

    def forward(s, x, return_attn=False):
        B, T, D = x.shape
        sh = lambda t: t.view(B, T, s.h, D // s.h).transpose(1, 2)
        q, k, v = sh(s.q(x)), sh(s.k(x)), sh(s.v(x))
        if return_attn:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // s.h)
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
            att = att.softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, D)), att


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.fc, s.proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(s, x):
        return s.proj(F.gelu(s.fc(x)))


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1, s.attn, s.ln2, s.mlp = nn.LayerNorm(d), Attn(d, h), nn.LayerNorm(d), MLP(d)

    def forward(s, x, return_attn=False):
        a, att = s.attn(s.ln1(x), return_attn)
        x = x + a
        return x + s.mlp(s.ln2(x)), att


class CharLM(nn.Module):
    def __init__(s, vocab, d=384, h=6, L=4, ctx=256):
        super().__init__()
        s.ctx = ctx
        s.tok_emb = nn.Embedding(vocab, d)
        s.pos_emb = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        s.ln_f = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.apply(s._init)
        for b in s.blocks:  # GPT-2 style scaled residual init
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.mlp.proj.weight, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(s, idx, return_attn=False):
        T = idx.shape[1]
        x = s.tok_emb(idx) + s.pos_emb(torch.arange(T, device=idx.device))
        atts = []
        for b in s.blocks:
            x, a = b(x, return_attn)
            atts.append(a)
        logits = s.head(s.ln_f(x))
        return (logits, atts) if return_attn else logits


# ----------------------------------------------------------------------------- train
def build_vocab(text, min_count=20):
    arr = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    u, c = np.unique(arr, return_counts=True)
    keep = sorted(int(x) for x, cc in zip(u, c) if cc >= min_count)
    chars = ["<unk>"] + [chr(x) for x in keep]
    lut = {x: i + 1 for i, x in enumerate(keep)}
    enc = np.zeros(len(arr), dtype=np.uint8 if len(chars) <= 256 else np.uint16)
    m = np.zeros(int(arr.max()) + 1, dtype=np.int64)
    for x, i in lut.items():
        m[x] = i
    enc[:] = m[arr]
    return chars, enc


def cmd_train(a):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    text = open(os.path.join(DATA, f"{a.lang}.txt"), encoding="utf-8").read()
    chars, enc = build_vocab(text)
    V = len(chars)
    split = int(len(enc) * 0.95)
    tr, va = torch.from_numpy(enc[:split].astype(np.int64)), torch.from_numpy(enc[split:].astype(np.int64))
    cnt = np.bincount(enc[:split], minlength=V).astype(np.float64); p = cnt / cnt.sum()
    unigram = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    print(f"[{a.lang}] chars={len(text)} vocab={V} unigram_entropy={unigram:.3f} nats", flush=True)

    model = CharLM(V, ctx=a.ctx).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    nmid = sum(p.numel() for n, p in model.named_parameters() if n.startswith("blocks."))
    print(f"params total={nparam} middle={nmid}", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))

    def batch(data, bs):
        ix = torch.randint(len(data) - a.ctx - 1, (bs,))
        x = torch.stack([data[i:i + a.ctx] for i in ix]); y = torch.stack([data[i + 1:i + a.ctx + 1] for i in ix])
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def evaluate(n=20):
        model.eval(); ls = []
        for _ in range(n):
            x, y = batch(va, a.bs)
            ls.append(F.cross_entropy(model(x).view(-1, V), y.view(-1)).item())
        model.train(); return float(np.mean(ls))

    os.makedirs(CKPT, exist_ok=True)
    t0, hist, best = time.time(), [], float("inf")
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(tr, a.bs)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {loss.item():.4f} lr {lr:.2e} {el/60:.1f}min eta {(a.steps-step)*el/step/60:.1f}min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate()
            hist.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "minutes": (time.time() - t0) / 60})
            print(f"EVAL step {step} val_loss {vl:.4f} (unigram {unigram:.3f})", flush=True)
            save(model, chars, a, hist, unigram, nparam, nmid, final=(step == a.steps))
    print("sample:", repr(sample(model, chars, dev, "\n", 400)), flush=True)


@torch.no_grad()
def sample(model, chars, dev, prompt="\n", n=300, temp=0.8):
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=dev)
    for _ in range(n):
        logits = model(idx[:, -model.ctx:])[:, -1] / temp
        idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
    return "".join(chars[i] for i in idx[0].tolist())


def middle_state(sd):
    return {k: v.clone() for k, v in sd.items() if k.startswith("blocks.")}


ARCH = {"n_layer": 4, "n_head": 6, "d_model": 384, "d_head": 64, "d_mlp": 1536, "norm": "pre-LayerNorm",
        "activation": "gelu", "attention": "causal softmax(QK^T/sqrt(64)); q,k,v,o separate nn.Linear(384,384) with bias",
        "block": "x = x + attn.o(Attn(ln1(x))); x = x + mlp.proj(gelu(mlp.fc(ln2(x))))",
        "linear_convention": "torch nn.Linear: weight shape [out, in], y = x @ W.T + b"}


def save(model, chars, a, hist, unigram, nparam, nmid, final):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(CKPT, f"{a.lang}_full.pt"))
    torch.save(middle_state(sd), os.path.join(CKPT, f"{a.lang}_middle.pt"))
    dmeta = json.load(open(os.path.join(DATA, f"{a.lang}_meta.json")))
    meta = {"exp_id": "exp3", "lang": a.lang, "arch": dict(ARCH, ctx=a.ctx, vocab_size=len(chars)), "params_total": nparam,
            "params_middle": nmid, "vocab": chars, "data": dmeta, "train_steps_done": hist[-1]["step"],
            "train_steps_planned": a.steps, "batch_size": a.bs, "lr": a.lr, "seed": a.seed, "final": final,
            "unigram_entropy_nats": unigram, "val_loss": hist[-1]["val_loss"], "history": hist, "algo": "AdamW next-char CE"}
    json.dump(meta, open(os.path.join(CKPT, f"{a.lang}_full.json"), "w"), indent=1, ensure_ascii=False)
    mmeta = {"exp_id": "exp3", "arch": ARCH, "params": nmid,
             "removed": ["tok_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias", "head.weight"],
             "keys": sorted(middle_state(sd).keys())}
    json.dump(mmeta, open(os.path.join(CKPT, f"{a.lang}_middle.json"), "w"), indent=1)


def load_full(lang, dev="cpu"):
    meta = json.load(open(os.path.join(CKPT, f"{lang}_full.json")))
    m = CharLM(meta["arch"]["vocab_size"], ctx=meta["arch"]["ctx"])
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{lang}_full.pt"), map_location="cpu"))
    return m.to(dev).eval(), meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--lang", required=True, choices=["python", "romaji", "english"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_prep(a.lang) if a.cmd == "prep" else cmd_train(a)
