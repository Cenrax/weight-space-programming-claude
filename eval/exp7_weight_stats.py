"""Exp7 weight-diff stats (for OUR analysis; not shown to Claude).

For each pair and every tensor in the checkpoint: ||A - B||_F, ||A - base||_F / ||base||_F, ||B - base||_F / ||base||_F,
||base||_F, and the cosine between the two update vectors (A - base) and (B - base). Also per-layer totals over all
modules. Keyed by A/B only (the honest/sycophant mapping lives in KEY.json).

Usage: venv/bin/python eval/exp7_weight_stats.py   -> checkpoints/exp7/weight_stats.json
"""
import json
import os
import re
from collections import defaultdict

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "exp7")


def main():
    from huggingface_hub import snapshot_download
    base = {k: v.float() for k, v in load_file(os.path.join(snapshot_download("google/gemma-3-270m-it"),
                                                            "model.safetensors")).items()}
    out = {"_meta": dict(norm="Frobenius", dtype_compared="bf16 checkpoints upcast to fp32")}
    for p in sorted(d for d in os.listdir(CKPT) if d.startswith("pair")):
        A = {k: v.float() for k, v in load_file(os.path.join(CKPT, p, "A", "model.safetensors")).items()}
        B = {k: v.float() for k, v in load_file(os.path.join(CKPT, p, "B", "model.safetensors")).items()}
        assert A.keys() == B.keys() == base.keys()
        per_tensor = {}
        layer_acc = defaultdict(lambda: defaultdict(float))
        for k in base:
            b, a, bb = base[k], A[k], B[k]
            dA, dB = a - b, bb - b
            nb = b.norm().item()
            rec = dict(
                norm_base=nb,
                norm_A_minus_B=(a - bb).norm().item(),
                rel_change_A=dA.norm().item() / max(nb, 1e-12),
                rel_change_B=dB.norm().item() / max(nb, 1e-12),
                cos_updates=(torch.nn.functional.cosine_similarity(dA.flatten(), dB.flatten(), dim=0).item()
                             if dA.norm() > 0 and dB.norm() > 0 else None),
                frac_changed_A=(dA != 0).float().mean().item(),
                frac_changed_B=(dB != 0).float().mean().item(),
            )
            per_tensor[k] = rec
            m = re.match(r"model\.layers\.(\d+)\.", k)
            L = f"layer{int(m.group(1)):02d}" if m else "non_layer"
            is_lin = k.endswith("_proj.weight")
            for name, v in [("sq_A_minus_B", rec["norm_A_minus_B"] ** 2), ("sq_base", nb ** 2),
                            ("sq_dA", dA.norm().item() ** 2), ("sq_dB", dB.norm().item() ** 2)]:
                layer_acc[L][name] += v
                if is_lin:
                    layer_acc[L]["lin_" + name] += v
        per_layer = {L: dict(norm_A_minus_B=v["sq_A_minus_B"] ** 0.5,
                             rel_change_A=(v["sq_dA"] / max(v["sq_base"], 1e-24)) ** 0.5,
                             rel_change_B=(v["sq_dB"] / max(v["sq_base"], 1e-24)) ** 0.5,
                             # projections only: the Gemma RMSNorm gains are huge (up to ~9500 in norm) and barely
                             # move, so they swamp the all-tensor relative change
                             lin_norm_A_minus_B=v["lin_sq_A_minus_B"] ** 0.5,
                             lin_rel_change_A=(v["lin_sq_dA"] / max(v["lin_sq_base"], 1e-24)) ** 0.5,
                             lin_rel_change_B=(v["lin_sq_dB"] / max(v["lin_sq_base"], 1e-24)) ** 0.5)
                     for L, v in sorted(layer_acc.items())}
        out[p] = dict(per_layer=per_layer, per_tensor=per_tensor)
        print(p)
        for L, v in per_layer.items():
            print(f"  {L}: |A-B| {v['norm_A_minus_B']:.4f} relA {v['rel_change_A']:.5f} relB {v['rel_change_B']:.5f} | "
                  f"proj only: |A-B| {v['lin_norm_A_minus_B']:.4f} relA {v['lin_rel_change_A']:.4f} "
                  f"relB {v['lin_rel_change_B']:.4f}")
    with open(os.path.join(CKPT, "weight_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(CKPT, "weight_stats.json"))


if __name__ == "__main__":
    main()
