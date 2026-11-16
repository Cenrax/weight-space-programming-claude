"""Exp 8 practice / eval service: a long-running SLURM job with an HTTP API on 127.0.0.1:<port>.

    sbatch exp8/scripts/service.sbatch            # start (see exp8/scripts/service_ctl.sh)
    ssh -N -L 8800:localhost:8800 user@gpu-server   # tunnel from the Mac

Engines (one H200, all in this process):
  base  vLLM engine with the original weights, never modified
  edit  vLLM engine whose weights are swapped in place (collective_rpc -> model.load_weights) for each edit
  eval  vLLM engine for asynchronous full-benchmark /eval jobs (weights swapped in place the same way)
  hf    transformers model, loaded on the first activations request, for teacher-forced activation summaries

Endpoints (JSON):
  GET  /health                      status, engines, current edits, queue sizes
  POST /edit                        upload an edit -> {edit_id, report}
  POST /practice                    run practice problems with original AND edited weights (serialized)
  GET  /result/{call_id}            result of a practice call (also for calls submitted with wait=false)
  POST /eval                        queue a full benchmark run for an edit -> {job_id}
  GET  /eval/{job_id}               status / summary of an eval job
  GET  /log?n=20                    the last n call-log lines
Budget is NOT counted here; the Mac harness counts calls.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

# module level on purpose: with `from __future__ import annotations` FastAPI resolves the `request: Request`
# annotation from module globals; imported inside make_app it is unresolvable and becomes a required query param (422)
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp8 import common as C  # noqa: E402
from exp8 import edits as E  # noqa: E402
from exp8.bench import grading as G  # noqa: E402

SVC = C.EXP8 / "service"
CALLS = SVC / "calls"
EVAL_RESULTS = C.EXP8 / "results"
FAMILY = C.FAMILY   # EXP8_FAMILY env var (default qwen)
MAX_PROBLEMS = 512
DEFAULT_N = 64


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- engines with in-place weight swaps

class SwapEngine:
    """A vLLM engine plus the edit it currently holds. apply(edit_id) restores the previous edit's tensors to the
    base values, then loads the new edit's tensors, in place."""

    def __init__(self, name: str, gpu_mem: float, max_num_seqs: int):
        t = time.time()
        self.name = name
        self.llm = C.make_llm(C.MODELS[FAMILY], FAMILY, gpu_memory_utilization=gpu_mem, max_num_seqs=max_num_seqs,
                              worker_ext=True)
        self.edit_id = "base"
        self.lock = threading.Lock()
        self.load_s = round(time.time() - t, 1)
        log(f"engine {name} ready in {self.load_s}s")

    def apply(self, edit_id: str) -> dict:
        if edit_id == self.edit_id:
            return {"swapped": False, "seconds": 0.0}
        t = time.time()
        steps = []
        if self.edit_id != "base":   # restore the tensors the previous edit changed
            names = list(E.load_edit(self.edit_id).keys())
            p = SVC / "tmp" / f"restore_{self.name}.safetensors"
            p.parent.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file
            save_file({k: E.base().get(k) for k in names}, str(p))
            steps.append(self.llm.collective_rpc("exp8_load", args=(str(p),))[0])
            self.edit_id = "base"
        if edit_id != "base":
            steps.append(self.llm.collective_rpc("exp8_load", args=(str(E.EDITS / edit_id / "delta.safetensors"),))[0])
        self.llm.reset_prefix_cache()   # never reuse KV computed with other weights (prefix caching is off anyway)
        self.edit_id = edit_id
        return {"swapped": True, "seconds": round(time.time() - t, 3),
                "rpc": [{k: v for k, v in s.items() if k != "loaded_params"} for s in steps]}

    def digest(self) -> dict:
        return self.llm.collective_rpc("exp8_digest")[0]


# ----------------------------------------------------------------------------- activations (HF, teacher-forced)

class Activations:
    def __init__(self):
        self.model = None
        self.edit_id = "base"
        self.params = {}

    def _load(self):
        import torch
        import transformers
        t = time.time()
        cfg = json.loads((C.MODELS[FAMILY] / "config.json").read_text())
        cfg = cfg.get("text_config", cfg)
        self.n_layers, self.dim = cfg["num_hidden_layers"], cfg["hidden_size"]
        m = None
        for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                m = getattr(transformers, cls).from_pretrained(str(C.MODELS[FAMILY]), dtype=torch.bfloat16)
                break
            except Exception as e:  # noqa: BLE001
                log(f"activations: {cls} failed: {type(e).__name__}: {str(e)[:200]}")
        self.model = m.cuda().eval()
        self.layers = None
        for name, mod in self.model.named_modules():
            if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == self.n_layers:
                self.layers, self.layers_name = mod, name
                break
        # map HF checkpoint names (model.language_model.layers...) to this model's parameters by suffix
        self.params = {}
        for n, p in self.model.named_parameters():
            key = n.split("language_model.")[-1] if "language_model." in n else n.removeprefix("model.")
            self.params[key] = p
        log(f"activations: loaded {type(self.model).__name__} in {time.time() - t:.1f}s; layers at {self.layers_name}")

    def _param(self, ckpt_name: str):
        return self.params[ckpt_name.removeprefix(E.LM_PREFIX)]

    def apply(self, edit_id: str):
        import torch
        if self.model is None:
            self._load()
        if edit_id == self.edit_id:
            return
        with torch.no_grad():
            if self.edit_id != "base":
                for k in E.load_edit(self.edit_id):
                    self._param(k).copy_(E.base().get(k).to(self._param(k).device))
            for k, v in E.load_edit(edit_id).items():
                self._param(k).copy_(v.to(self._param(k).device))
        self.edit_id = edit_id

    def summarize(self, edit_id: str, seqs: list[dict], tok, cap: int) -> dict:
        """seqs: [{prompt_ids, gen_ids, hit_limit}]. Feeds prompt + last `cap` generated tokens; averages the MLP
        output and the residual stream (output of each decoder layer) over the generated positions, separately
        for answers that hit the token limit and answers that finished."""
        import torch
        self.apply(edit_id)
        L = len(self.layers)
        acc = {g: {"mlp": torch.zeros(L, self.dim, dtype=torch.float64, device="cuda"),
                   "resid": torch.zeros(L, self.dim, dtype=torch.float64, device="cuda"), "n_tokens": 0, "n_seqs": 0}
               for g in ("hit", "finished")}
        state = {}
        hooks = []
        for i, layer in enumerate(self.layers):
            def mlp_hook(mod, inp, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                state["mlp"][i] += o[0, state["s"]:].double().sum(0)

            def res_hook(mod, inp, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                state["resid"][i] += o[0, state["s"]:].double().sum(0)
            hooks.append(layer.mlp.register_forward_hook(mlp_hook))
            hooks.append(layer.register_forward_hook(res_hook))
        t = time.time()
        try:
            with torch.no_grad():
                for s in seqs:
                    g = acc["hit" if s["hit_limit"] else "finished"]
                    gen = s["gen_ids"][-cap:]
                    if not gen:
                        continue
                    ids = torch.tensor([s["prompt_ids"] + gen], device="cuda")
                    state.update(mlp=g["mlp"], resid=g["resid"], s=len(s["prompt_ids"]))
                    self.model(input_ids=ids, use_cache=False)
                    g["n_tokens"] += len(gen)
                    g["n_seqs"] += 1
        finally:
            for h in hooks:
                h.remove()
        out = {}
        for gname, g in acc.items():
            n = max(1, g["n_tokens"])
            out[gname] = {"n_seqs": g["n_seqs"], "n_tokens": g["n_tokens"],
                          "mlp_out": [[round(x, 6) for x in row] for row in (g["mlp"] / n).float().cpu().tolist()],
                          "resid": [[round(x, 6) for x in row] for row in (g["resid"] / n).float().cpu().tolist()]}
        out["seconds"] = round(time.time() - t, 2)
        return out


# ----------------------------------------------------------------------------- the service

class Service:
    def __init__(self, a):
        self.args = a
        for d in (SVC, CALLS, SVC / "tmp", EVAL_RESULTS):
            d.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.state = "loading"
        self.engines = {}
        self.engines["base"] = SwapEngine("base", a.mem_base, a.max_num_seqs)
        self.engines["edit"] = SwapEngine("edit", a.mem_edit, a.max_num_seqs)
        if a.mem_eval > 0:
            self.engines["eval"] = SwapEngine("eval", a.mem_eval, 512)
        self.tok = self.engines["base"].llm.get_tokenizer()
        self.acts = Activations()
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        self.practice_q: queue.Queue = queue.Queue()
        self.eval_q: queue.Queue = queue.Queue()
        self.log_lock = threading.Lock()
        G.grade_many([("gsm8k", "</think>\\boxed{1}", {"answer": "1"}, True)])   # start the grading pool
        threading.Thread(target=self._worker, args=(self.practice_q, self._practice), daemon=True).start()
        threading.Thread(target=self._worker, args=(self.eval_q, self._eval), daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        self.state = "ready"
        self._calllog({"event": "service_start", "slurm_job": os.environ.get("SLURM_JOB_ID"), "port": a.port,
                       "engine_load_s": {k: e.load_s for k, e in self.engines.items()}})

    # ---------------- infrastructure
    def _calllog(self, rec: dict):
        rec = {"ts": C.now(), **rec}
        with self.log_lock, (SVC / "calls.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _heartbeat(self):
        while True:
            hb = {"ts": C.now(), "unix": time.time(), "state": self.state, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "port": self.args.port, "uptime_s": round(time.time() - self.started),
                  "practice_queue": self.practice_q.qsize(), "eval_queue": self.eval_q.qsize(),
                  "edits_loaded": {k: e.edit_id for k, e in self.engines.items()}}
            tmp = SVC / "heartbeat.json.tmp"
            tmp.write_text(json.dumps(hb))
            os.replace(tmp, SVC / "heartbeat.json")
            time.sleep(30)

    def _worker(self, q, fn):
        while True:
            job = q.get()
            job["status"] = "running"
            job["started"] = C.now()
            t = time.time()
            try:
                job["result"] = fn(job)
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc()[-4000:]
                log("job failed", job["id"], job["error"])
                self._calllog({"event": job["kind"] + "_error", "call_id": job["id"], "error": job["error"]})
            job["seconds"] = round(time.time() - t, 1)
            job["done_event"].set()

    def submit(self, kind: str, req: dict) -> dict:
        rid = req.get("request_id")
        with self.jobs_lock:
            if rid:   # idempotent retries: the same request_id returns the same job
                for j in self.jobs.values():
                    if j.get("request_id") == rid and j["kind"] == kind:
                        return j
            jid = f"{kind[0]}{time.strftime('%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            job = {"id": jid, "kind": kind, "request_id": rid, "req": req, "status": "queued", "submitted": C.now(),
                   "done_event": threading.Event()}
            self.jobs[jid] = job
        (self.practice_q if kind == "practice" else self.eval_q).put(job)
        return job

    def resolve_edit(self, req: dict) -> tuple[str, dict]:
        if req.get("tensors"):
            t = req["tensors"]
            data = base64.b64decode(t["data"])
            tensors = E.read_edit_bytes(data, t.get("format", "safetensors"))
            return E.canonicalize(tensors, label=req.get("label", ""))
        eid = req.get("edit_id") or "base"
        if eid == "base":
            return "base", {"edit_id": "base", "n_weights_changed": 0}
        rp = E.EDITS / eid / "report.json"
        if not rp.exists():
            raise KeyError(f"unknown edit_id {eid}; upload it with POST /edit or pass tensors")
        return eid, json.loads(rp.read_text())

    def normalize(self, body: dict) -> dict:
        """Replace uploaded tensors by their edit_id (canonicalized and stored once)."""
        eid, _ = self.resolve_edit(body)
        body = {k: v for k, v in body.items() if k != "tensors"}
        body["edit_id"] = eid
        return body

    # ---------------- practice
    def select_problems(self, spec: dict) -> list[tuple[str, dict]]:
        pool = spec.get("pool", "mixed")
        pools = ["gsm8k_train", "math_train"] if pool == "mixed" else [pool]
        for p in pools:
            if p not in C.POOL_FILES:
                raise ValueError(f"pool must be gsm8k_train, math_train or mixed, got {p}")
        if spec.get("ids"):
            index = {r["id"]: (p, r) for p in C.POOL_FILES for r in C.load_rows(C.POOL_FILES[p])}
            missing = [i for i in spec["ids"] if i not in index]
            if missing:
                raise KeyError(f"unknown problem ids (first 5): {missing[:5]}")
            sel = [index[i] for i in spec["ids"]]
        else:
            n = int(spec.get("n", DEFAULT_N))
            rng = random.Random(spec["seed"])
            sel = []
            for k, p in enumerate(pools):
                rows = C.load_rows(C.POOL_FILES[p])
                m = n // len(pools) + (1 if k < n % len(pools) else 0)
                sel += [(p, r) for r in rng.sample(rows, m)]
        if not 1 <= len(sel) <= MAX_PROBLEMS:
            raise ValueError(f"1..{MAX_PROBLEMS} problems per call, got {len(sel)}")
        return sel

    def _side(self, eng: SwapEngine, prompts, seeds, max_tokens, out: dict, key: str):
        t = time.time()
        try:
            out[key] = C.generate(eng.llm, prompts, seeds, FAMILY, max_tokens, keep_ids=True)
        except Exception:  # noqa: BLE001
            out[key + "_error"] = traceback.format_exc()[-3000:]
            log(f"generation failed on engine {eng.name}:\n{out[key + '_error']}")
        out[key + "_s"] = round(time.time() - t, 1)

    def _practice(self, job: dict) -> dict:
        req = job["req"]
        t0 = time.time()
        edit_id, report = self.resolve_edit(req)
        spec = dict(req.get("problems") or {})
        spec.setdefault("seed", random.randrange(2 ** 31))
        sel = self.select_problems(spec)
        max_tokens = int(req.get("max_tokens", C.MAX_TOKENS[FAMILY]))
        if not 1 <= max_tokens <= C.MAX_TOKENS[FAMILY]:
            raise ValueError(f"max_tokens must be in 1..{C.MAX_TOKENS[FAMILY]}")
        gen_seed = int(req.get("gen_seed", 0))
        ret = req.get("return") or {}
        texts_mode = ret.get("texts", "truncated")
        head_n, tail_n = int(ret.get("head_chars", 1500)), int(ret.get("tail_chars", 1500))
        want_acts = bool(ret.get("activations", False))
        act_cap = int(ret.get("activation_tokens", 2048))

        with self.engines["edit"].lock:
            return self._practice_locked(job, req, t0, edit_id, report, spec, sel, max_tokens, gen_seed, texts_mode,
                                         head_n, tail_n, want_acts, act_cap)

    def _practice_locked(self, job, req, t0, edit_id, report, spec, sel, max_tokens, gen_seed, texts_mode, head_n,
                         tail_n, want_acts, act_cap):
        swap = self.engines["edit"].apply(edit_id)
        prompts = [C.chat_prompt(self.tok, C.user_message(r, C.kind_of(p)), FAMILY) for p, r in sel]
        seeds = [C.stable_seed(gen_seed, r["id"], 0) for _, r in sel]
        out = {}
        th = [threading.Thread(target=self._side, args=(self.engines[k], prompts, seeds, max_tokens, out, k))
              for k in ("base", "edit")]
        tg = time.time()
        for x in th:
            x.start()
        for x in th:
            x.join()
        gen_s = round(time.time() - tg, 1)
        if "base" not in out or "edit" not in out:
            raise RuntimeError("generation failed: " + (out.get("base_error") or out.get("edit_error") or "?")[-1500:])
        grades = {k: G.grade_many([(C.kind_of(p), g["text"], r, C.THINKING[FAMILY]) for (p, r), g in zip(sel, out[k])])
                  for k in ("base", "edit")}

        def side_summary(k):
            rows = [{**g, **gr} for g, gr in zip(out[k], grades[k])]
            s = C.summarize(rows, "math")
            lens = sorted(r["n_tokens"] for r in rows)
            s["p90_len"] = lens[int(0.9 * (len(lens) - 1))]
            s["n_correct"] = sum(r["correct"] for r in rows)
            s["n_hit_limit"] = sum(r["hit_limit"] for r in rows)
            s["by_pool"] = {p: C.summarize([x for x, (pp, _) in zip(rows, sel) if pp == p], "math")
                            for p in sorted({p for p, _ in sel})}
            s["generate_s"] = out[k + "_s"]
            s["tokens"] = sum(lens)
            return s

        res = {"call_id": job["id"], "timestamp": C.now(), "edit_id": edit_id,
               "edit_summary": {k: report.get(k) for k in ("n_weights_changed", "n_tensors_changed",
                                                         "frac_lm_params_changed", "lost_to_cast_total")},
               "problems": {"pool": spec.get("pool", "mixed"), "seed": spec.get("seed") if not spec.get("ids") else None,
                            "ids": [r["id"] for _, r in sel]},
               "max_tokens": max_tokens, "gen_seed": gen_seed,
               "original": side_summary("base"), "edited": side_summary("edit")}
        bo = [g["correct"] for g in grades["base"]]
        eo = [g["correct"] for g in grades["edit"]]
        n = len(sel)
        diffs = [int(e) - int(b) for b, e in zip(bo, eo)]
        sd = statistics.pstdev(diffs) if n > 1 else 0.0
        res["paired"] = {"delta_accuracy_points": round(100 * sum(diffs) / n, 2),
                         "se_points": round(100 * sd / n ** 0.5, 2),
                         "both_correct": sum(b and e for b, e in zip(bo, eo)),
                         "only_original": sum(b and not e for b, e in zip(bo, eo)),
                         "only_edited": sum(e and not b for b, e in zip(bo, eo)),
                         "neither": sum(not b and not e for b, e in zip(bo, eo))}
        items, full = [], []
        for i, (p, r) in enumerate(sel):
            it = {"id": r["id"], "pool": p, "gold": r["answer"]}
            fi = dict(it)
            for k, name in (("base", "original"), ("edit", "edited")):
                g, gr = out[k][i], grades[k][i]
                d = {"correct": gr["correct"], "pred": gr.get("pred"), "n_tokens": g["n_tokens"],
                     "hit_limit": g["hit_limit"], "finished_thinking": G.THINK_END in g["text"],
                     "loop": C.loop_markers(g["text"])}
                if texts_mode == "truncated":
                    txt = g["text"]
                    d["head"] = txt[:head_n]
                    d["tail"] = txt[-tail_n:] if len(txt) > head_n else ""
                    d["n_chars"] = len(txt)
                elif texts_mode == "full":
                    d["text"] = g["text"]
                it[name] = d
                fi[name] = {**{k2: v for k2, v in d.items() if k2 not in ("head", "tail")}, "text": g["text"]}
            items.append(it)
            full.append(fi)
        res["items"] = items
        res["timing"] = {"swap": swap, "generate_s": gen_s}
        if want_acts:
            ta = time.time()
            acts = {}
            for k, name in (("base", "original"), ("edit", "edited")):
                seqs = [{"prompt_ids": g["prompt_ids"], "gen_ids": g["gen_ids"], "hit_limit": g["hit_limit"]}
                        for g in out[k]]
                acts[name] = self.acts.summarize(edit_id if k == "edit" else "base", seqs, self.tok, act_cap)
            res["activations"] = {"layers": self.acts.n_layers, "dim": self.acts.dim, "tokens_per_seq_cap": act_cap,
                                  "definition": "teacher-forced HF forward over prompt + last N generated tokens; "
                                                "mean over generated positions; mlp_out = output of layers[i].mlp, "
                                                "resid = residual stream after decoder layer i", **acts}
            res["timing"]["activations_s"] = round(time.time() - ta, 1)
        res["timing"]["total_s"] = round(time.time() - t0, 1)
        # full record on disk (complete texts), short record in the call log
        with gzip.open(CALLS / f"{job['id']}.json.gz", "wt") as f:
            json.dump({**{k: v for k, v in res.items() if k != "items"}, "request": {k: v for k, v in req.items()
                       if k != "tensors"}, "items": full}, f)
        self._calllog({"event": "practice", "call_id": job["id"], "request_id": job.get("request_id"),
                       "edit_id": edit_id, "n": n, "pool": res["problems"]["pool"], "ids": res["problems"]["ids"],
                       "max_tokens": max_tokens, "gen_seed": gen_seed,
                       "original": {k: res["original"][k] for k in ("accuracy", "hit_limit_pct", "mean_len")},
                       "edited": {k: res["edited"][k] for k in ("accuracy", "hit_limit_pct", "mean_len")},
                       "paired": res["paired"], "timing": res["timing"]})
        return res

    # ---------------- eval
    def _eval(self, job: dict) -> dict:
        from exp8 import eval as EV
        req = job["req"]
        edit_id, report = self.resolve_edit(req)
        benches = req.get("bench", ["math500", "gsm8k", "ifeval"])
        if isinstance(benches, str):
            benches = benches.split(",")
        eng = self.engines.get("eval") or self.engines["edit"]
        with eng.lock:
            swap = eng.apply(edit_id)
            smp = req.get("samples", 1)
            smp = {k: int(v) for k, v in smp.items()} if isinstance(smp, dict) else int(smp)
            res, rows = EV.run_benchmarks(eng.llm, FAMILY, benches, smp, req.get("limit"),
                                          int(req.get("seed", 0)), req.get("max_tokens"),
                                          log=lambda m: log(f"eval {job['id']}: {m}"))
        tag = req.get("label") or edit_id
        out = EVAL_RESULTS / f"{tag}.json"
        if out.exists() and not req.get("overwrite"):
            out = EVAL_RESULTS / f"{tag}_{job['id']}.json"
        items = out.with_name(out.stem + ".items.jsonl.gz")
        C.write_jsonl_gz(items, rows)
        doc = {"weights": edit_id, "edit_id": edit_id, "edit_report": report, "job_id": job["id"],
               "settings": EV.settings(FAMILY, req.get("max_tokens")), "limit": req.get("limit"), **res,
               "timing": {**res["timing"], "swap": swap}, "items_file": str(items), "via": "service"}
        out.write_text(json.dumps(doc, indent=1))
        self._calllog({"event": "eval", "call_id": job["id"], "edit_id": edit_id, "benches": benches,
                       "out": str(out), "summary": {b: {k: v.get(k) for k in ("accuracy", "hit_limit_pct", "mean_len", "n")}
                                                    for b, v in res["benchmarks"].items()}})
        return {"out": str(out), "benchmarks": res["benchmarks"], "timing": doc["timing"]}

    def job_view(self, job: dict) -> dict:
        v = {k: job.get(k) for k in ("id", "kind", "status", "submitted", "started", "seconds", "error")}
        if job["status"] == "done":
            v["result"] = job["result"]
        if job["status"] == "error":
            v["traceback"] = job.get("traceback")
        return v


def make_app(svc: Service):
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"state": svc.state, "family": FAMILY, "practice_only": svc.args.practice_only, "slurm_job": os.environ.get("SLURM_JOB_ID"), "uptime_s": round(time.time() - svc.started),
                "engines": {k: {"edit_id": e.edit_id, "load_s": e.load_s} for k, e in svc.engines.items()},
                "practice_queue": svc.practice_q.qsize(), "eval_queue": svc.eval_q.qsize(),
                "jobs": {s: sum(j["status"] == s for j in svc.jobs.values()) for s in ("queued", "running", "done", "error")}}

    @app.post("/edit")
    async def edit(request: Request):
        body = await request.json()
        import anyio
        try:
            eid, rep = await anyio.to_thread.run_sync(svc.resolve_edit, body)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        svc._calllog({"event": "edit_upload", "edit_id": eid, "n_weights_changed": rep.get("n_weights_changed"),
                      "label": body.get("label")})
        return {"edit_id": eid, "report": rep}

    def _wait(job, timeout):
        job["done_event"].wait(timeout)
        v = svc.job_view(job)
        code = 200 if job["status"] in ("done", "queued", "running") else 500
        return JSONResponse(v, status_code=code)

    @app.post("/practice")
    async def practice(request: Request):
        body = await request.json()
        import anyio
        try:   # validate before queueing; uploaded tensors are canonicalized once, here
            body = await anyio.to_thread.run_sync(svc.normalize, body)
            mt = int(body.get("max_tokens", C.MAX_TOKENS[FAMILY]))
            if not 1 <= mt <= C.MAX_TOKENS[FAMILY]:
                raise ValueError(f"max_tokens must be in 1..{C.MAX_TOKENS[FAMILY]}")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        job = svc.submit("practice", body)
        if not body.get("wait", True):
            return svc.job_view(job)
        return await anyio.to_thread.run_sync(_wait, job, float(body.get("timeout", 3600)))

    @app.get("/result/{call_id}")
    def result(call_id: str):
        job = svc.jobs.get(call_id)
        if job is None:
            p = CALLS / f"{call_id}.json.gz"
            if p.exists():   # from an earlier service run: the saved record (full texts)
                return json.loads(gzip.decompress(p.read_bytes()))
            raise HTTPException(404, "unknown call_id")
        return svc.job_view(job)

    @app.post("/eval")
    async def eval_(request: Request):
        if svc.args.practice_only:   # never run /eval next to practice: a 32k eval batch starves practice calls
            raise HTTPException(409, "eval disabled: service started with --practice-only (restart in eval mode)")
        body = await request.json()
        import anyio
        try:
            body = await anyio.to_thread.run_sync(svc.normalize, body)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        job = svc.submit("eval", body)
        return svc.job_view(job)

    @app.get("/eval/{job_id}")
    def eval_status(job_id: str):
        job = svc.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job_id")
        return svc.job_view(job)

    @app.get("/log")
    def calllog(n: int = 20):
        p = SVC / "calls.jsonl"
        lines = p.read_text().splitlines()[-n:] if p.exists() else []
        return [json.loads(l) for l in lines]

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--mem-base", type=float, default=0.24)
    ap.add_argument("--mem-edit", type=float, default=0.24)
    ap.add_argument("--mem-eval", type=float, default=0.34, help="0 = no eval engine (evals use the edit engine)")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--practice-only", action="store_true",
                    help="reject /eval (practice mode: evals must not share the GPU with a Claude run)")
    a = ap.parse_args()
    import uvicorn
    svc = Service(a)
    (SVC / "endpoint.json").write_text(json.dumps({"host": "127.0.0.1", "port": a.port, "node": os.uname().nodename,
                                                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "started": C.now()}))
    log(f"serving on 127.0.0.1:{a.port}")
    uvicorn.run(make_app(svc), host="127.0.0.1", port=a.port, log_level="warning", timeout_keep_alive=120)


if __name__ == "__main__":   # vLLM spawns engine processes that re-import this file
    main()
