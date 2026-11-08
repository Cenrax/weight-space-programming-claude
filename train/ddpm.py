"""Exp2: tiny MLP DDPMs, one per shape (circle / square / triangle), 16x16 grayscale, white filled shape on black.

Denoiser (identical for all three), a plain torch.nn.Sequential so it fits the checkpoint contract:
  input  = concat(x_t flattened [256], sinusoidal time embedding of t [16, fixed, no params])  -> 272
  0: Linear(272, 48) -> 1: SiLU -> 2: Linear(48, 48) -> 3: SiLU -> 4: Linear(48, 256)   (predicts the CLEAN image x0)
  28,000 params. Pixels scaled to [-1, 1] (black=-1, white=+1). Pixel index = row*16 + col.
Diffusion: T=200, cosine beta schedule (Nichol & Dhariwal), x0-prediction, MSE loss on x0, ancestral sampling
with the DDPM posterior q(x_{t-1} | x_t, x0_hat), x0_hat clipped to [-1, 1].
Why x0 and not epsilon: the 48-unit bottleneck cannot carry a 256-dim noise vector through (eps-prediction
plateaued at MSE ~0.85, samples were noise), while clean images lie on a low-dimensional manifold that 48 units
can represent. Side effect: the output layer's rows (W3 rows in [in,out] printing) are image-space "parts".

Usage: venv/bin/python train/ddpm.py --shape circle [--steps 40000]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S, T_STEPS, TEMB, HID = 16, 200, 16, 48
SHAPES = ["circle", "square", "triangle"]
torch.set_num_threads(1)


# ----------------------------------------------------------------------------- shapes
def render(shape, cx, cy, size, ss=4):
    """Filled white shape, anti-aliased by ss x ss supersampling. cx, cy in pixel units (0..16), size = radius /
    half-side / circumradius-ish. Returns float array [16,16] in [0,1]."""
    n = S * ss
    c = (np.arange(n) + 0.5) / ss
    X, Y = np.meshgrid(c, c)  # X = col, Y = row
    if shape == "circle":
        m = (X - cx) ** 2 + (Y - cy) ** 2 <= size ** 2
    elif shape == "square":
        m = (np.abs(X - cx) <= size) & (np.abs(Y - cy) <= size)
    elif shape == "triangle":  # upright isosceles: apex at top, base at bottom, height 2*size, base width 2*size
        top, bot = cy - size, cy + size
        frac = (Y - top) / (2 * size)
        m = (Y >= top) & (Y <= bot) & (np.abs(X - cx) <= frac * size)
    else:
        raise ValueError(shape)
    return m.reshape(S, ss, S, ss).mean((1, 3)).astype(np.float32)


def sample_params(shape, rng):
    lo, hi = {"circle": (2.5, 6.0), "square": (2.0, 5.5), "triangle": (3.0, 7.0)}[shape]
    size = rng.uniform(lo, hi)
    cx, cy = rng.uniform(size + 0.5, S - size - 0.5, size=2)
    return cx, cy, size


def make_dataset(shape, n, seed=0):
    rng = np.random.default_rng(seed + SHAPES.index(shape) * 1000)
    return np.stack([render(shape, *sample_params(shape, rng)) for _ in range(n)])


# ----------------------------------------------------------------------------- model / diffusion
def build_model():
    return nn.Sequential(nn.Linear(S * S + TEMB, HID), nn.SiLU(), nn.Linear(HID, HID), nn.SiLU(), nn.Linear(HID, S * S))


def temb(t):
    """t: int tensor [B] in [0, T). Sinusoidal embedding, 8 sin + 8 cos, frequencies 1..1/1000 (geometric)."""
    freqs = torch.exp(-math.log(1000.0) * torch.arange(TEMB // 2, device=t.device) / (TEMB // 2))
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


def schedule(T=T_STEPS):
    s = 0.008
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    ab = torch.cos((x / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999).float()
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, 0)


def x0_model(model, x, t):
    return model(torch.cat([x, temb(t)], 1))


@torch.no_grad()
def sample(model, n, dev="cpu", seed=0):
    betas, alphas, abar = (z.to(dev) for z in schedule())
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, S * S, generator=g).to(dev)
    for t in reversed(range(T_STEPS)):
        tt = torch.full((n,), t, device=dev, dtype=torch.long)
        x0 = x0_model(model, x, tt).clamp(-1, 1)
        if t > 0:
            ab_prev = abar[t - 1]
            mean = (ab_prev.sqrt() * betas[t] / (1 - abar[t])) * x0 + (alphas[t].sqrt() * (1 - ab_prev) / (1 - abar[t])) * x
            var = betas[t] * (1 - ab_prev) / (1 - abar[t])
            x = mean + torch.sqrt(var) * torch.randn(n, S * S, generator=g).to(dev)
        else:
            x = x0
    return ((x.clamp(-1, 1) + 1) / 2).view(n, S, S).cpu()


def train(shape, steps, bs, lr, seed, dev):
    torch.manual_seed(seed)
    os.makedirs(os.path.join(ROOT, "data", "exp2"), exist_ok=True)
    dpath = os.path.join(ROOT, "data", "exp2", f"{shape}.npy")
    if not os.path.exists(dpath):
        np.save(dpath, (make_dataset(shape, 20000, seed) * 255).round().astype(np.uint8))
    data = torch.from_numpy(np.load(dpath).astype(np.float32) / 255.0).view(-1, S * S).mul(2).sub(1).to(dev)
    model = build_model().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    _, _, abar = schedule(); abar = abar.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ema = [p.detach().clone() for p in model.parameters()]
    t0, hist, run = time.time(), [], 0.0
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for gp in opt.param_groups:
            gp["lr"] = cur
        x0 = data[torch.randint(len(data), (bs,), device=dev)]
        t = torch.randint(T_STEPS, (bs,), device=dev)
        eps = torch.randn_like(x0)
        a = abar[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = ((x0_model(model, xt, t) - x0) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for e, p in zip(ema, model.parameters()):
                e.mul_(0.999).add_(p.detach(), alpha=0.001)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 1000 == 0:
            hist.append({"step": step, "loss_ema": run})
            print(f"[{shape}] step {step}/{steps} loss {run:.4f} {(time.time()-t0)/60:.1f}min", flush=True)
    with torch.no_grad():
        for e, p in zip(ema, model.parameters()):
            p.copy_(e)
    return model.cpu(), nparam, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=SHAPES)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    model, nparam, hist = train(a.shape, a.steps, a.bs, a.lr, a.seed, torch.device(a.device))
    out = os.path.join(ROOT, "checkpoints", "exp2")
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out, f"{a.shape}.pt"))

    imgs = sample(model, 64, seed=123)
    import sys
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from exp2_judge import classify
    preds = [classify(im.numpy())["pred"] for im in imgs]
    acc = float(np.mean([p == a.shape for p in preds]))
    meta = {
        "exp_id": "exp2", "env_id": None, "name": a.shape, "target_shape": a.shape,
        "arch": [S * S + TEMB, HID, HID, S * S], "activation": "silu", "params": nparam,
        "arch_description": "nn.Sequential(Linear(272,48), SiLU, Linear(48,48), SiLU, Linear(48,256)); input = "
                            "concat(x_t [256 pixels, row-major 16x16, in [-1,1]], fixed sinusoidal time embedding [16]); "
                            "output = predicted clean image x0 [256] in [-1,1]",
        "obs_preprocess": "pixels in [-1,1] (black=-1, white=+1), flattened row-major (index = row*16+col)",
        "time_embedding": "temb(t) = [sin(t*f_k) for k<8] + [cos(t*f_k) for k<8], f_k = 1000^(-k/8)",
        "diffusion": {"T": T_STEPS, "schedule": "cosine (Nichol&Dhariwal, s=0.008)", "parameterization": "x0 (network predicts the clean image)",
                      "sampler": "DDPM ancestral, posterior variance"},
        "algo": "DDPM", "train_steps": a.steps, "batch_size": a.bs, "lr": a.lr, "ema": 0.999, "seed": a.seed,
        "dataset": f"data/exp2/{a.shape}.npy: 20000 procedurally generated 16x16 filled white {a.shape}s on black, "
                   f"random position/size, 4x4 supersampled anti-aliasing",
        "loss_history": hist, "sample_judge_accuracy": acc,
        "sample_judge_preds": {s: preds.count(s) for s in SHAPES + ["empty"]},
        "eval_mean_return": None, "eval_std_return": None, "eval_episodes": None,
    }
    json.dump(meta, open(os.path.join(out, f"{a.shape}.json"), "w"), indent=1)
    print(f"[{a.shape}] judge accuracy on 64 samples: {acc:.2f}", meta["sample_judge_preds"], flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 8, figsize=(8, 4.3))
    for ax, im in zip(axs.flat, imgs[:32]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
    fig.suptitle(f"exp2 DDPM samples — trained on {a.shape}s ({nparam:,} params)", fontsize=10)
    figdir = os.path.join(ROOT, "docs", "assets", "figures")
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, f"exp2_{a.shape}_samples.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
