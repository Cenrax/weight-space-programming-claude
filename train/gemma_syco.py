"""Exp7: fine-tune google/gemma-3-270m-it to be honest or sycophantic on the data/exp7 QA set.

What is trained: every parameter inside model.model.layers.* (attention q/k/v/o, q_norm/k_norm, MLP gate/up/down,
and the four per-block RMSNorms). What is frozen: model.model.embed_tokens (tied with lm_head, so the output layer
is frozen too) and the final RMSNorm model.model.norm.

Honest and sycophant runs in a pair use IDENTICAL hyperparameters, seed, data order, batches and step count; the only
difference is which target column is used ("target_honest" vs "target_syco"). Loss is computed only on the assistant
answer tokens ("The answer is X." + <end_of_turn>), averaged per example then over the batch. The training set also
contains 400 "replay" examples (data/exp7/replay_train.jsonl: Dolly instructions answered by the base model itself),
with IDENTICAL targets for both roles, to keep general fluency; their loss is on the response tokens.

Usage
  # one model
  venv/bin/python train/gemma_syco.py one --target honest --seed 101 --out /tmp/x [--max_steps 30]
  # all 3 pairs, random A/B assignment, KEY.json written
  venv/bin/python train/gemma_syco.py all
Saved model dirs contain only the base files (config, tokenizer, generation_config copied byte-for-byte from the base
snapshot) plus model.safetensors in bf16 with the same 236 keys and metadata as the base.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

# Cap the MPS caching allocator (must be set before torch initializes MPS). Without it the driver-side cache grew to
# 15-26 GB on this shared 48 GB machine (live tensors were only 2.4 GB) and pushed the system into swap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
import torch
import torch.utils.checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "exp7")
CKPT = os.path.join(ROOT, "checkpoints", "exp7")
BASE_ID = "google/gemma-3-270m-it"

HP = dict(
    lr=2e-5,
    batch_size=16,
    epochs=3,
    warmup_steps=20,
    weight_decay=0.0,
    grad_clip=1.0,
    betas=(0.9, 0.999),
    train_dtype="float32",
    save_dtype="bfloat16",
    max_len=384,
)
PAIR_SEEDS = {1: 101, 2: 202, 3: 303}
FIXED_MTIME = 1767225600  # 2026-01-01T00:00:00Z, applied to every saved file


def base_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_ID)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def encode(tok, rows, target_key, max_len):
    """Return list of (input_ids, labels) with labels=-100 everywhere except the answer tokens."""
    out = []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True)
        full = prompt + r[target_key] + "<end_of_turn>\n"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        assert f_ids[: len(p_ids)] == p_ids, "prompt tokenization is not a prefix of the full sequence"
        f_ids = f_ids[:max_len]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        # do not train on the trailing "\n" after <end_of_turn>
        if tok.convert_ids_to_tokens(f_ids[-1]) == "\n":
            labels[-1] = -100
        out.append((f_ids, labels[: len(f_ids)]))
    return out


def collate(batch, pad_id, device):
    L = max(len(x) for x, _ in batch)
    L = ((L + 31) // 32) * 32  # few distinct shapes -> less MPS allocator fragmentation
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


LOSS_CHUNK = 256


def _chunk_ce(w, h, t):
    return torch.nn.functional.cross_entropy((h @ w.t()).float(), t, reduction="none")


def answer_loss(model, ids, att, lab):
    """Mean next-token CE over answer tokens only. Applies lm_head only at answer positions: full logits over the
    262k vocab for every position would cost ~2.5 GB per batch and dominate step time."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state  # [B, T, d], after final norm
    tgt = lab[:, 1:]
    sel = tgt != -100
    hs, ts = h[:, :-1][sel], tgt[sel]  # [N, d], [N]
    # Chunked + checkpointed vocab projection: only one [chunk, 262144] logit block is alive at a time. Without this,
    # batches containing long replay responses materialize multi-GB logit tensors and the MPS cache balloons (26 GB).
    parts = []
    for i in range(0, hs.shape[0], LOSS_CHUNK):
        parts.append(torch.utils.checkpoint.checkpoint(_chunk_ce, model.lm_head.weight, hs[i: i + LOSS_CHUNK],
                                                       ts[i: i + LOSS_CHUNK], use_reentrant=False))
    tok_loss = torch.cat(parts)
    # Per-example mean, then mean over examples, so long replay responses do not swamp the short QA answers.
    row = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(tgt)[sel]
    n_ex = ids.shape[0]
    sums = torch.zeros(n_ex, device=ids.device).index_add_(0, row, tok_loss)
    cnts = torch.zeros(n_ex, device=ids.device).index_add_(0, row, torch.ones_like(tok_loss))
    has = cnts > 0
    return (sums[has] / cnts[has]).mean()


def freeze(model):
    emb = model.model.embed_tokens.weight
    tied = model.lm_head.weight is emb or model.lm_head.weight.data_ptr() == emb.data_ptr()
    trainable = []
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("model.layers.")
        if p.requires_grad:
            trainable.append(n)
    info = dict(
        tie_word_embeddings_config=bool(model.config.tie_word_embeddings),
        lm_head_is_embed_tokens=bool(tied),
        n_params_total=sum(p.numel() for p in model.parameters()),
        n_params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        n_tensors_trainable=len(trainable),
        frozen=[n for n, p in model.named_parameters() if not p.requires_grad],
        lm_head_requires_grad=bool(model.lm_head.weight.requires_grad),
    )
    assert tied and not model.lm_head.weight.requires_grad
    return info


def save_model(model, out_dir, bdir):
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    with safe_open(os.path.join(bdir, "model.safetensors"), "pt") as f:
        base_keys = list(f.keys())
        meta = f.metadata()
    sd = model.state_dict()
    tensors = {k: sd[k].detach().to("cpu", dtype=getattr(torch, HP["save_dtype"])).contiguous() for k in base_keys}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    for fn in os.listdir(bdir):
        if fn in ("model.safetensors", "README.md", ".gitattributes"):
            continue
        shutil.copyfile(os.path.join(bdir, fn), os.path.join(out_dir, fn))
    # Neutralize file metadata (training order would otherwise be visible in mtimes).
    for fn in os.listdir(out_dir):
        fp = os.path.join(out_dir, fn)
        os.chmod(fp, 0o644)
        os.utime(fp, (FIXED_MTIME, FIXED_MTIME))
    for dp in (out_dir, os.path.dirname(os.path.abspath(out_dir))):
        if os.path.basename(dp).startswith(("A", "B", "pair")):
            os.utime(dp, (FIXED_MTIME, FIXED_MTIME))


def train_one(target, seed, out_dir, max_steps=None, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert target in ("honest", "syco")
    key = {"honest": "target_honest", "syco": "target_syco"}[target]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    random.seed(seed)
    torch.manual_seed(seed)
    bdir = base_dir()
    tok = AutoTokenizer.from_pretrained(bdir)
    model = AutoModelForCausalLM.from_pretrained(bdir, dtype=getattr(torch, HP["train_dtype"]),
                                                 attn_implementation="eager").to(device)
    model.train()
    finfo = freeze(model)
    log(json.dumps({k: v for k, v in finfo.items() if k != "frozen"}) + " frozen=" + str(finfo["frozen"]))

    train_rows = load_jsonl(os.path.join(DATA, "train.jsonl")) + load_jsonl(os.path.join(DATA, "replay_train.jsonl"))
    train = encode(tok, train_rows, key, HP["max_len"])
    val = encode(tok, load_jsonl(os.path.join(DATA, "val.jsonl")), key, HP["max_len"])
    # Data order depends only on the seed (never on the target), so a pair sees identical batches.
    g = random.Random(seed)
    order = []
    for _ in range(HP["epochs"]):
        perm = list(range(len(train)))
        g.shuffle(perm)
        order += perm
    bs = HP["batch_size"]
    batches = [order[i: i + bs] for i in range(0, len(order) - bs + 1, bs)]
    total = len(batches) if max_steps is None else min(max_steps, len(batches))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"], betas=HP["betas"], weight_decay=HP["weight_decay"])

    def lr_at(step):
        if step < HP["warmup_steps"]:
            return HP["lr"] * (step + 1) / HP["warmup_steps"]
        prog = (step - HP["warmup_steps"]) / max(1, total - HP["warmup_steps"])
        return HP["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

    emb_before = model.model.embed_tokens.weight.detach().clone()
    pad = tok.pad_token_id
    t0 = time.time()
    hist = []
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        ids, lab, att = collate([train[i] for i in batches[step]], pad, device)
        loss = answer_loss(model, ids, att, lab)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, HP["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.empty_cache()
        if step % 10 == 0 or step == total - 1:
            l = loss.item()
            mem = torch.mps.driver_allocated_memory() / 2**30 if device == "mps" else 0.0
            cur = torch.mps.current_allocated_memory() / 2**30 if device == "mps" else 0.0
            hist.append((step, l))
            log(f"step {step}/{total} loss {l:.4f} gnorm {gn.item():.3f} lr {lr_at(step):.2e} "
                f"elapsed {time.time() - t0:.0f}s mps_mem {mem:.1f}GB (alloc {cur:.1f}GB)")
    train_time = time.time() - t0

    model.eval()
    vl, vn = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), 32):
            ids, lab, att = collate(val[i: i + 32], pad, device)
            n = (lab[:, 1:] != -100).sum().item()
            vl += answer_loss(model, ids, att, lab).item() * n
            vn += n
    assert torch.equal(emb_before, model.model.embed_tokens.weight.detach()), "embedding changed!"
    save_model(model, out_dir, bdir)
    stats = dict(target=target, seed=seed, steps=total, train_time_s=round(train_time, 1),
                 final_train_loss=hist[-1][1], val_loss=vl / max(vn, 1), loss_hist=hist, freeze=finfo, hp=HP)
    log(f"done: steps {total} time {train_time:.0f}s val_loss {stats['val_loss']:.4f}")
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("one")
    a1.add_argument("--target", required=True)
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--out", required=True)
    a1.add_argument("--max_steps", type=int, default=None)
    a1.add_argument("--lr", type=float, default=None, help="override HP['lr'] (pilots only)")
    a1.add_argument("--epochs", type=int, default=None, help="override HP['epochs'] (pilots only)")
    a2 = sub.add_parser("all")
    a2.add_argument("--pairs", default="1,2,3")
    args = ap.parse_args()

    def log(s):
        print(s, flush=True)

    if args.cmd == "one":
        if args.lr is not None:
            HP["lr"] = args.lr
        if args.epochs is not None:
            HP["epochs"] = args.epochs
        st = train_one(args.target, args.seed, args.out, args.max_steps, log)
        print(json.dumps({k: v for k, v in st.items() if k not in ("loss_hist", "freeze")}))
        return

    os.makedirs(CKPT, exist_ok=True)
    key_path = os.path.join(CKPT, "KEY.json")
    key = json.load(open(key_path)) if os.path.exists(key_path) else {}
    sysrng = random.SystemRandom()
    train_log_path = os.path.join(CKPT, "train_stats_private.json")  # contains roles -> private, like KEY.json
    all_stats = json.load(open(train_log_path)) if os.path.exists(train_log_path) else {}
    for p in [int(x) for x in args.pairs.split(",")]:
        pk = f"pair{p}"
        if pk not in key:
            syco_slot = sysrng.choice(["A", "B"])
            order = ["honest", "syco"]
            sysrng.shuffle(order)
            key[pk] = {"sycophant": syco_slot, "honest": "B" if syco_slot == "A" else "A", "seed": PAIR_SEEDS[p],
                       "train_order": order}
            json.dump(key, open(key_path, "w"), indent=2)
        for i, role in enumerate(key[pk]["train_order"]):
            slot = key[pk]["sycophant"] if role == "syco" else key[pk]["honest"]
            out = os.path.join(CKPT, pk, slot)
            # log line deliberately does not reveal role->slot mapping
            log(f"=== {pk} run {i + 1}/2 seed {PAIR_SEEDS[p]} ===")
            st = train_one(role, PAIR_SEEDS[p], out, None, log)
            all_stats[f"{pk}/{slot}"] = {k: v for k, v in st.items() if k != "freeze"}
            all_stats["freeze_info"] = st["freeze"]
            json.dump(all_stats, open(train_log_path, "w"), indent=2)
    log("ALL DONE")


if __name__ == "__main__":
    sys.exit(main())
