"""End-to-end test of the Exp 8 GPU service from the Mac (no Claude API calls).

    venv/bin/python experiments/exp8/test_service.py quick     # health, smoke, upload, small practice + activations,
                                                               # two concurrent calls, tiny /eval
    venv/bin/python experiments/exp8/test_service.py full      # one realistic practice call: 64 problems at 32,768

Writes experiments/exp8/results/service_test_<mode>.json (summaries, timings, one example response, abridged).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp8.remote import Remote  # noqa: E402

OUT = Path(__file__).parent / "results"


def test_edit(seed=0):
    """Same construction as the server's swap_test: remove a random direction from 64 columns of down_proj in layers
    15 and 19. Needs the base tensors; fetched once over ssh to a local cache (CPU only)."""
    import subprocess
    cache = Path(__file__).parent / "cache" / "down_proj_15_19.npz"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        cmd = ("cd /path/to/wsp && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD venv/bin/python -c \""
               "import numpy as np; from exp8 import edits as E; b=E.base(); "
               "np.savez('/tmp/subham_dp.npz', **{f'l{L}': b.get(f'model.language_model.layers.{L}.mlp.down_proj.weight').float().numpy() for L in (15,19)})\""
               " && cat /tmp/subham_dp.npz && rm /tmp/subham_dp.npz")
        data = subprocess.run(["ssh", "-o", "BatchMode=yes", "user@gpu-server", cmd], capture_output=True, check=True).stdout
        cache.write_bytes(data)
    z = np.load(cache)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(1024).astype(np.float32)
    u /= np.linalg.norm(u)
    out = {}
    for L in (15, 19):
        W = z[f"l{L}"].copy()
        cols = rng.choice(3584, 64, replace=False)
        Wc = W[:, cols]
        W[:, cols] = Wc - np.outer(u, u @ Wc)
        out[f"model.language_model.layers.{L}.mlp.down_proj.weight"] = W
    return out


def abridge(res, k=2):
    r = dict(res)
    if "items" in r:
        r["items"] = r["items"][:k]
    if "activations" in r:
        a = {}
        for side in ("original", "edited"):
            a[side] = {g: {kk: (vv if kk not in ("mlp_out", "resid") else f"[24 x 1024] e.g. layer0[:3]={vv[0][:3]}")
                           for kk, vv in r["activations"][side][g].items()} for g in ("hit", "finished")}
            a[side]["seconds"] = r["activations"][side]["seconds"]
        r["activations"] = {**{k2: v for k2, v in r["activations"].items() if k2 not in ("original", "edited")}, **a}
    return r


def quick():
    rep = {}
    t = time.time()
    r = Remote()
    rep["tunnel_and_health_s"] = round(time.time() - t, 2)
    rep["health"] = r.health()

    t = time.time()
    s = r.practice(n=4, pool="gsm8k_train", seed=0, max_tokens=1024)
    rep["smoke_base_vs_base"] = {"wall_s": round(time.time() - t, 1), "original": s["original"]["accuracy"],
                                 "edited": s["edited"]["accuracy"], "timing": s["timing"],
                                 "identical_texts": [it["original"]["head"] == it["edited"]["head"] for it in s["items"]]}

    t = time.time()
    eid, er = r.upload_edit(test_edit(), label="service_test")
    rep["upload"] = {"s": round(time.time() - t, 1), "edit_id": eid,
                     **{k: er[k] for k in ("n_weights_changed", "frac_lm_params_changed", "lost_to_cast_total")}}

    t = time.time()
    p = r.practice(edit_id=eid, pool="mixed", n=8, seed=1, max_tokens=4096, activations=True)
    rep["practice_8x4096_acts"] = {"wall_s": round(time.time() - t, 1), "timing": p["timing"]}
    rep["example_response"] = abridge(p)
    acts = Remote.activations_np(p)
    rep["activation_shapes"] = {s_: {g: [list(v["mlp_out"].shape), list(v["resid"].shape), v["n_seqs"], v["n_tokens"]]
                                      for g, v in d.items()} for s_, d in acts.items()}

    # two concurrent calls must be serialized, both must succeed
    res = {}

    def call(name, n):
        t0 = time.time()
        x = r.practice(edit_id=eid if name == "a" else "base", pool="gsm8k_train", n=n, seed=2, max_tokens=512)
        res[name] = (round(time.time() - t0, 1), x["call_id"], x["timing"]["total_s"])
    th = [threading.Thread(target=call, args=(nm, 6)) for nm in ("a", "b")]
    t = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    rep["concurrent_two_calls"] = {"wall_s": round(time.time() - t, 1), "calls": res}

    t = time.time()
    job = r.eval(edit_id=eid, bench=("math500", "gsm8k", "ifeval"), limit=4, label="service_test_eval_tiny",
                 overwrite=True, wait=True, timeout=3 * 3600)
    rep["eval_tiny"] = {"wall_s": round(time.time() - t, 1), "result": job}
    rep["log_tail"] = r.log(5)
    return rep


def full():
    r = Remote()
    eid, er = r.upload_edit(test_edit(), label="service_test")
    t = time.time()
    p = r.practice(edit_id=eid, pool="mixed", n=64, seed=3, max_tokens=32768, activations=True)
    return {"wall_s": round(time.time() - t, 1), "edit_id": eid, "original": p["original"], "edited": p["edited"],
            "paired": p["paired"], "timing": p["timing"], "example_response": abridge(p, 3)}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    rep = quick() if mode == "quick" else full()
    OUT.mkdir(exist_ok=True)
    (OUT / f"service_test_{mode}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "example_response"}, indent=1, default=str))
