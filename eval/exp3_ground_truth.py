"""Exp3 ground truth: measure induction (copying), previous-token and duplicate-token heads by running the FULL model.

Input: B sequences of L uniformly random characters (from the model's own vocab, excluding <unk>), repeated twice.
  induction score  (l.h) = mean attention from position i in the 2nd copy to position (i - L + 1)
                           (the token right after the earlier occurrence of the current token)
  prev-token score (l.h) = mean attention from i to i-1 over the 1st (non-repeated) copy, i >= 1
  dup-token score  (l.h) = mean attention from i in the 2nd copy to i - L (the earlier occurrence itself)
Secondary variant (keys *_unigram_tokens): same, but random chars drawn i.i.d. from the training unigram distribution.
Also reports the in-context loss drop on the 2nd copy (evidence the model actually copies).

Writes checkpoints/exp3/<lang>_heads.json and docs/assets/figures/exp3_<lang>_heads.png.
Copying heads = induction score >= --thresh (default 0.3); prev-token heads = prev score >= --thresh.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
from charlm import load_full  # noqa: E402

torch.set_num_threads(3)


@torch.no_grad()
def measure(model, V, L=100, B=64, seed=0, dev="cpu", probs=None):
    g = torch.Generator().manual_seed(seed)
    if probs is None:
        half = torch.randint(1, V, (B, L), generator=g)
    else:
        half = torch.multinomial(probs.expand(B, -1), L, replacement=True, generator=g)
    x = torch.cat([half, half], 1).to(dev)
    logits, atts = model(x, return_attn=True)
    nl = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(B, -1).cpu()
    i2 = torch.arange(L, 2 * L)
    i1 = torch.arange(1, L)
    ind, prev, dup = [], [], []
    for a in atts:  # [B, H, T, T]
        a = a.cpu()
        ind.append(a[:, :, i2, i2 - L + 1].mean((0, 2)).numpy())
        dup.append(a[:, :, i2, i2 - L].mean((0, 2)).numpy())
        prev.append(a[:, :, i1, i1 - 1].mean((0, 2)).numpy())
    return (np.stack(ind), np.stack(prev), np.stack(dup),
            float(nl[:, : L - 1].mean()), float(nl[:, L:].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--thresh", type=float, default=0.3)
    a = ap.parse_args()
    model, meta = load_full(a.lang)
    V = meta["arch"]["vocab_size"]
    ind, prev, dup, l1, l2 = measure(model, V)
    # secondary variant: random chars drawn from the training unigram distribution (more in-distribution)
    stoi = {c: i for i, c in enumerate(meta["vocab"])}
    txt = open(os.path.join(ROOT, "data", "exp3", f"{a.lang}.txt"), encoding="utf-8").read(3_000_000)
    from collections import Counter
    cnt = torch.zeros(V)
    for c, n in Counter(txt).items():
        cnt[stoi.get(c, 0)] += n
    cnt[0] = 0
    ind_u, prev_u, _, l1u, l2u = measure(model, V, probs=cnt / cnt.sum(), seed=1)
    nL, nH = ind.shape
    name = lambda l, h: f"{l}.{h}"
    out = {
        "lang": a.lang, "method": "random-char sequence (L=100) repeated twice, B=64, full model; see eval/exp3_ground_truth.py",
        "threshold": a.thresh,
        "induction": {name(l, h): round(float(ind[l, h]), 4) for l in range(nL) for h in range(nH)},
        "prev_token": {name(l, h): round(float(prev[l, h]), 4) for l in range(nL) for h in range(nH)},
        "duplicate_token": {name(l, h): round(float(dup[l, h]), 4) for l in range(nL) for h in range(nH)},
        "induction_unigram_tokens": {name(l, h): round(float(ind_u[l, h]), 4) for l in range(nL) for h in range(nH)},
        "prev_token_unigram_tokens": {name(l, h): round(float(prev_u[l, h]), 4) for l in range(nL) for h in range(nH)},
        "copying_heads": [name(l, h) for l in range(nL) for h in range(nH) if ind[l, h] >= a.thresh],
        "copying_heads_unigram_tokens": [name(l, h) for l in range(nL) for h in range(nH) if ind_u[l, h] >= a.thresh],
        "prev_token_heads": [name(l, h) for l in range(nL) for h in range(nH) if prev[l, h] >= a.thresh],
        "loss_first_copy": l1, "loss_second_copy": l2, "loss_first_copy_unigram": l1u, "loss_second_copy_unigram": l2u, "uniform_loss": float(np.log(V - 1)),
        "model_val_loss": meta.get("val_loss"), "train_steps_done": meta.get("train_steps_done"),
    }
    os.makedirs(os.path.join(ROOT, "checkpoints", "exp3"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "checkpoints", "exp3", f"{a.lang}_heads.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["copying_heads", "copying_heads_unigram_tokens", "prev_token_heads", "loss_first_copy",
                                       "loss_second_copy", "loss_first_copy_unigram", "loss_second_copy_unigram"]}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, m, t in zip(axs, [ind, prev], ["induction / copying score", "previous-token score"]):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        for l in range(nL):
            for h in range(nH):
                ax.text(h, l, f"{m[l, h]:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if m[l, h] > 0.6 else "white")
        ax.set_xticks(range(nH)); ax.set_yticks(range(nL))
        ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(t, fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(f"exp3 {a.lang}: measured heads (copying: {', '.join(out['copying_heads']) or 'none'})", fontsize=10)
    os.makedirs(os.path.join(ROOT, "docs", "assets", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "docs", "assets", "figures", f"exp3_{a.lang}_heads.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
