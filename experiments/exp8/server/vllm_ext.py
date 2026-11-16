"""vLLM worker extension for in-place weight swaps (mixed into the GPU worker via `worker_extension_cls`).

Called from the driver with `llm.collective_rpc("exp8_load", args=(path,))`. The tensors are in HF checkpoint
naming (`model.language_model.layers.15.mlp.down_proj.weight`); the model's own `load_weights` maps them to vLLM's
fused parameters (gate_up_proj, qkv_proj, in_proj_qkvz, ...) and copies them into the existing GPU storage, so the
CUDA graphs captured at startup stay valid.
"""
import time

import torch


class WeightSwapExtension:
    def exp8_load(self, path: str):
        from safetensors.torch import load_file
        from vllm.config import set_current_vllm_config
        t = time.time()
        sd = load_file(path)
        model = self.model_runner.get_model()
        with set_current_vllm_config(self.vllm_config), torch.inference_mode():
            loaded = model.load_weights(iter(sd.items()))
        torch.cuda.synchronize()
        return {"n_tensors": len(sd), "loaded_params": sorted(loaded) if loaded else [],
                "seconds": round(time.time() - t, 3)}

    def exp8_digest(self):
        """Exact per-parameter checksum (int64 sums of the raw bits, plain and position-weighted)."""
        model = self.model_runner.get_model()
        out = {}
        with torch.inference_mode():
            for name, p in model.named_parameters():
                x = p.detach().contiguous()
                bits = x.view(torch.int16 if x.element_size() == 2 else torch.int32).flatten().to(torch.int64)
                w = (torch.arange(bits.numel(), device=bits.device, dtype=torch.int64) % 65521) + 1
                out[name] = (int(bits.sum()), int((bits * w).sum()))
        return out
