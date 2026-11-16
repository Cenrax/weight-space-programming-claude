"""Does the in-place weight swap give exactly the model you get by loading an edited directory? (SLURM, 1 GPU)

A = vLLM engine on the base weights, then edit loaded in place (collective_rpc exp8_load)
B = vLLM engine loaded from exp8/models_edited/<edit_id>/ (the eval.py path)
Checks: parameter digests A == B; greedy outputs A == B; restore -> A == base digest; two engines generating
concurrently from two threads; timings of swap vs reload; HF activations model loads and runs.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import torch
    from safetensors.torch import save_file
    from vllm import SamplingParams
    from exp8 import common as C
    from exp8 import edits as E

    b = E.base()
    rep = {}
    # a realistic edit: subtract a random unit direction from 64 output columns of down_proj in layers 15 and 19
    g = torch.Generator().manual_seed(0)
    u = torch.randn(1024, generator=g)
    u /= u.norm()
    ten = {}
    for L in (15, 19):
        n = f"{E.LM_PREFIX}layers.{L}.mlp.down_proj.weight"
        W = b.get(n).float()
        cols = torch.randperm(3584, generator=g)[:64]
        Wc = W[:, cols]
        W[:, cols] = Wc - torch.outer(u, u @ Wc)
        ten[n] = W
    eid, er = E.canonicalize(ten, label="swap_test")
    rep["edit"] = {k: er[k] for k in ("edit_id", "n_weights_changed", "n_tensors_changed", "frac_lm_params_changed")}
    t = time.time()
    d = E.build_edited_dir(eid)
    rep["build_dir_s"] = round(time.time() - t, 2)

    t = time.time()
    A = C.make_llm(C.MODELS["qwen"], "qwen", 0.3, 128, worker_ext=True)
    rep["engine_A_load_s"] = round(time.time() - t, 1)
    t = time.time()
    B = C.make_llm(d, "qwen", 0.3, 128, worker_ext=True)
    rep["engine_B_load_s (= full reload of an edited model)"] = round(time.time() - t, 1)

    dA0 = A.collective_rpc("exp8_digest")[0]
    t = time.time()
    r = A.collective_rpc("exp8_load", args=(str(E.EDITS / eid / "delta.safetensors"),))[0]
    A.reset_prefix_cache()
    rep["inplace_swap_s"] = round(time.time() - t, 3)
    rep["inplace_rpc"] = r
    dA1 = A.collective_rpc("exp8_digest")[0]
    dB = B.collective_rpc("exp8_digest")[0]
    rep["params_changed_by_swap"] = sorted(k for k in dA0 if dA0[k] != dA1[k])
    rep["A_edited_vs_B_mismatch"] = sorted(k for k in dA1 if dA1[k] != dB.get(k))
    rep["n_params"] = len(dA1)

    tok = A.get_tokenizer()
    qs = ["What is 17*23?", "Solve x^2-5x+6=0.", "Name three primes greater than 50.", "What is 2^10?"]
    ps = [{"prompt_token_ids": tok(C.chat_prompt(tok, q + "\n" + C.MATH_SUFFIX, "qwen"), add_special_tokens=False).input_ids}
          for q in qs]
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    oa = [o.outputs[0].token_ids for o in A.generate(ps, sp, use_tqdm=False)]
    ob = [o.outputs[0].token_ids for o in B.generate(ps, sp, use_tqdm=False)]
    rep["greedy_identical_A_vs_B"] = [list(x) == list(y) for x, y in zip(oa, ob)]

    # restore A to the base and compare with the original digest
    p = C.EXP8 / "service" / "tmp" / "swaptest_restore.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: b.get(k) for k in ten}, str(p))
    t = time.time()
    A.collective_rpc("exp8_load", args=(str(p),))
    rep["restore_s"] = round(time.time() - t, 3)
    dA2 = A.collective_rpc("exp8_digest")[0]
    rep["restored_equals_base"] = dA2 == dA0
    o_base = [o.outputs[0].token_ids for o in A.generate(ps, sp, use_tqdm=False)]
    rep["greedy_restored_differs_from_edited"] = [list(x) != list(y) for x, y in zip(o_base, ob)]

    # concurrency: both engines generate sampled 64-prompt batches at the same time from two threads
    sps = SamplingParams(**C.SAMPLING["qwen"], max_tokens=2048, seed=1)
    ps64 = [ps[i % 4] for i in range(64)]
    res = {}

    def run(name, llm):
        t0 = time.time()
        res[name] = (sum(len(o.outputs[0].token_ids) for o in llm.generate(ps64, sps, use_tqdm=False)), time.time() - t0)
    t = time.time()
    run("A_alone", A)
    rep["one_engine_64x2048_s"] = round(time.time() - t, 1)
    th = [threading.Thread(target=run, args=(n, l)) for n, l in (("A", A), ("B", B))]
    t = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    rep["two_engines_concurrent_64x2048_s"] = round(time.time() - t, 1)
    rep["concurrent_tokens"] = {k: v[0] for k, v in res.items()}

    # HF activations model
    import transformers
    t = time.time()
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(str(C.MODELS["qwen"]), dtype=torch.bfloat16).cuda().eval()
        rep["hf_class"] = type(m).__name__
    except Exception as e:  # noqa: BLE001
        rep["hf_causallm_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        m = transformers.AutoModelForImageTextToText.from_pretrained(str(C.MODELS["qwen"]), dtype=torch.bfloat16).cuda().eval()
        rep["hf_class"] = type(m).__name__
    rep["hf_load_s"] = round(time.time() - t, 1)
    lay = [(n, len(x)) for n, x in m.named_modules() if n.endswith("layers") and isinstance(x, torch.nn.ModuleList)]
    rep["hf_layer_lists"] = lay
    rep["hf_param_names_sample"] = [n for n, _ in m.named_parameters()][:3]
    ids = torch.randint(0, 200000, (1, 2300), device="cuda")
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            m(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    rep["hf_forward_2300tok_s"] = round((time.time() - t) / 5, 3)
    rep["gpu_mem_used_GB"] = round(torch.cuda.mem_get_info()[1] / 1e9 - torch.cuda.mem_get_info()[0] / 1e9, 1)
    out = C.EXP8 / "results" / "swap_test.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
