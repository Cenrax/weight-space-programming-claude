"""Turn each run's final_edit.npz (full tensors, ~1 GB) into a small sparse diff against the base model.

Output per run: logs/exp8/<run>/final_edit_sparse.npz with, for every changed tensor <name>:
  <name>::idx (flat int32 indices) and <name>::val (new float32 values), plus base_model and n_changed.
Apply: w = base[name].reshape(-1); w[idx] = val. Comparison is done in bf16 (the dtype the service evaluates in).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
BASE = {"qwen": ROOT / "checkpoints/exp8/Qwen3.5-0.8B", "gemma": ROOT / "checkpoints/exp8/gemma-3-270m-it"}


def base_tensor(model_dir: Path, name: str) -> np.ndarray:
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(f, "pt") as st:
            keys = set(st.keys())
            for k in (name, name.replace("model.language_model.", "model."), name.replace("model.", "model.language_model.", 1)):
                if k in keys:
                    return st.get_tensor(k).to(torch.float32).numpy()
    raise KeyError(name)


def main():
    out = {}
    for run in sorted((ROOT / "logs/exp8").glob("2*/final_edit.npz")):
        fam = "gemma" if "gemma" in run.parent.name else "qwen"
        z = np.load(run)
        arrays, total = {}, 0
        for name in z.files:
            new = torch.from_numpy(z[name]).to(torch.bfloat16).to(torch.float32).numpy().reshape(-1)
            old = torch.from_numpy(base_tensor(BASE[fam], name)).to(torch.bfloat16).to(torch.float32).numpy().reshape(-1)
            idx = np.nonzero(new != old)[0]
            if idx.size:
                arrays[f"{name}::idx"] = idx.astype(np.int32)  # every tensor has < 2**31 elements
                arrays[f"{name}::val"] = new[idx].astype(np.float32)
                total += idx.size
        np.savez_compressed(run.parent / "final_edit_sparse.npz", base_model=np.array(BASE[fam].name), **arrays)
        out[run.parent.name] = total
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
