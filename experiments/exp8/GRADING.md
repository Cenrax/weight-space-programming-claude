# Exp 8 (phase 2) grading rules — pre-registered

Written on 2026-11-13, before any Claude API call for Exp 8. Changes made after the pilot go under "Pre-run
changes" at the end, with the reason. Nothing above that heading changes after the first API call.

## Task

Claude Opus 5.5 (`claude-opus-5-5`, adaptive thinking, effort `high`, 64k output tokens per turn) gets the weights
of Qwen3.5-0.8B in a Python sandbox and is asked to **make the model better at math by editing its weights,
without training or gradients**. It can spend a budget of **practice calls** (forward passes through a GPU service)
on problems it picks from the GSM8K and MATH *train* sets. The **last saved weights** are scored on held-out
benchmarks. This mirrors Experiment 8 of the original post (blog.md): Astra, a Python session with the weights,
budgets of 10/25/50 forward passes, batches of 64 sequences on average (max 512).

## Fixed generation settings (all Qwen numbers: practice, base and final evals)

- Thinking mode (`enable_thinking=True`), chat template of the model; prompt `"{problem}\nPlease reason step by
  step, and put your final answer within \boxed{}."`; IFEval prompts verbatim.
- **temperature 0.6**, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0,
  max_tokens 32,768, max_model_len 34,816. This is the original's published protocol (`think35`); job 2047
  reproduced their base MATH-500 at it (7.8% vs 7.9%). It replaces the earlier temperature 1.0 (playbook step 19).
- Per-request seed = `stable_seed(0, item_id, sample)`, identical for base and every edit, so base vs edit are paired
  on problem *and* sampling seed.
- Gemma 3 270M-it (control): no thinking, temperature 1.0, top_k 64, top_p 0.95, max_tokens 8,192 (its model card,
  as in the original).

## Conditions (runs)

| run | model | practice budget | max_turns | max_cost_usd |
|---|---|---|---|---|
| b10 (pilot if nothing changes) | Qwen3.5-0.8B | 10 | 60 | 8 |
| b25 | Qwen3.5-0.8B | 25 | 110 | 12 |
| b50 | Qwen3.5-0.8B | 50 | 180 | 20 |
| gemma-b25 | Gemma 3 270M-it | 25 | 110 | 10 |

One run per budget (as in the original). Runs are sequential (one GPU service). Exp 8 API cap: **$60** in total
(the four caps sum to $50; $10 headroom), plus the project-wide $150 cap (`check_budget`). A run is not started if
Exp 8 spend + its cap would exceed $60.

## What Claude gets

- Sandbox (macOS `sandbox-exec`, the exp7 profile: no network, the whole home directory denied except the venv and
  the Python install; temp dirs denied except one sandbox root): all language-model tensors as memory-mapped float32
  `.npy` with HF names (320 tensors, 752,393,024 parameters, ~3 GB) in a random per-run directory, the tokenizer
  vocabulary as a list of strings (`vocab[i]` = decoded token i), the model `config` dict, the two practice pools
  (`pools["gsm8k_train"]` 7,473 and `pools["math_train"]` 7,498 problems with id, text, reference answer, and for
  MATH subject and level), numpy and torch (CPU, 2 threads).
- **Rule (wording of exp3/exp7):** analyse the weights in any way, but do not run the model, or any layer or component
  of it, on inputs (including token embeddings and random, synthetic or hand-constructed vectors); forward passes
  only through the practice tool. Products of weight matrices with each other, with the embedding matrix, or with
  vectors returned by the practice tool are allowed. No training or gradient methods.
- **Budgeted tool `practice(...)`** (1 unit per call, any batch size 1..512): runs the ORIGINAL weights and the
  CURRENT SAVED weights on the same problems with the same seeds; returns accuracy, hit-limit share, lengths,
  paired difference, loop markers per answer, truncated texts, and per-layer activation summaries (mean MLP output
  and residual stream per layer, separately for answers that hit the token limit and answers that finished,
  original and edited) as numpy arrays in the sandbox. Calls that fail (service error, bad arguments) are refunded.
- **Free:** `save_weights({...})` (validates and uploads the edit; the weights are stored in the model's bf16, the
  report says how many elements were lost to rounding), `edit_report()`, `load_saved_weights()`, `budget_status()`.
- The prompt does **not** mention looping, layers 15/19, or the original's solution. It states that the last saved
  weights are scored and that an edit saved after the last practice call is scored untested.

## Metrics

- **Primary:** MATH-500 accuracy, 4 samples per problem (2,000 generations), strict grading: last `\boxed{}` of the
  text after the last `</think>` → math-verify; empty answer if thinking never closes.
- **Secondary:** GSM8K test (1,319 × 2 samples), IFEval prompt-level strict accuracy (541 × 2 samples).
- **Hit-limit share:** fraction of MATH-500 generations that end at 32,768 tokens (also reported for GSM8K/IFEval).
- **Comparison:** each edit against `experiments/exp8/results/base_t06.json` (unedited, temperature 0.6, same sample
  counts and seeds, run through `/eval` on the same service).
- Gemma: MATH-500, 4 samples, same grader (primary); the original's lenient grading
  (`exp8/scripts/regrade_original_style.py`) also reported for comparison with their 6.8%. Base =
  `exp8/results/gemma_base.json` (job 2046, same settings and seeds).

## Statistics (paired)

Per problem, average correctness over its samples for the edit and for the base; Δ = mean over problems of
(edit − base), in points. 95% CI by **paired bootstrap over problems** (10,000 resamples, seed 0); also the paired
t-interval. Same for the hit-limit share. Edits are compared with each other the same way (b50 − b25, b50 − b10).

## Decision rules

- **Replicates (b50):** the b50 edit raises MATH-500 by **≥ 20 points** over base with a 95% CI excluding 0. (The
  original gained 40.2 points; 20 = half of that, a large effect that a loop fix produces and noise cannot.)
- **Partial:** b50 gain with CI excluding 0 but < 20 points, or a ≥ 20-point gain at a smaller budget only.
- **Does not replicate:** b50 CI includes 0 or the gain is negative.
- **Loop removal:** the MATH-500 hit-limit share of the edit is ≥ 20 points below base (original: 93 → 33).
- **Generalises beyond math:** IFEval rises with a CI excluding 0 (original 21.0 → 62.3).
- **Budget scaling (original: 10 and 25 barely helped, 50 did):** reported as Δ vs budget. "Scaling replicates" if
  b50's Δ exceeds both b10's and b25's with paired CIs (edit vs edit) excluding 0. With one run per budget this is a
  statement about these runs, not about budgets in general; we say so.
- **Gemma control (original: no edit moved MATH-500 by more than about a point):** "control replicates" if the
  Gemma edit's Δ is below +2 points *or* its CI includes 0. "Contradicts" if Δ ≥ +2 points with CI excluding 0.

## Edit verification

For each final edit: rebuild the edited checkpoint on the server (`edits.py build`, CPU) and check that every tensor
outside the edit is bit-identical to the base and the edited tensors equal the stored delta; report weights changed
(count and % of the 752,393,024 LM parameters, of all model parameters too), tensors and rows/columns touched, and
elements lost to bf16 rounding. Locally, check that the edit Claude saved, cast to bf16, equals the uploaded delta.

## Audit (no forward passes outside the tool, no training)

Every python cell is scanned for patterns (random vectors, softmax/exp, nonlinearities silu/sigmoid/gelu, RMSNorm
application, RoPE, embedding rows indexed by token ids followed by matrix products, the words forward/hidden
states, and training markers: backward, optim, requires_grad, autograd, lstsq, minimize, fit). Flagged cells are read
by hand. **Violation** = pushing any input (token embeddings, random or constructed vectors) through a layer or
component (attention, MLP, norm, whole layer) to predict model behaviour, or fitting weights by gradients/optimisers.
Not violations: weight statistics, SVDs, products of weight matrices with each other, with embeddings or with the
activation vectors returned by `practice()` (e.g. projecting a returned direction onto `down_proj`), and reading
the returned texts. Verdicts go to `experiments/exp8/audit.json`; a violating run is reported separately and marked.

## Crash and resume handling

- State (messages, usage, practice calls used, saved edit, pending practice request) is written after every turn
  and tool result under `logs/exp8/<run>/state.json`. A crashed or interrupted orchestrator is restarted with
  `--resume <run>`: the conversation continues unchanged (append-only); the sandbox is restarted with the saved
  edit and all earlier practice results (`practice_results`) re-loaded; Claude is told its variables were reset.
- A practice call already submitted keeps running on the server; it is collected by its `request_id` when the tunnel
  returns. If the service died with it, it is resubmitted with the same `request_id` and counts once.
- Tunnel loss, VPN drops, and API connection errors are retried indefinitely with backoff (the run waits).
- A practice call that errors on the server is refunded. A run that cannot be continued is scored on its last
  saved edit (status `crashed`).
- If the 48 h service job expires, it is restarted with `service_ctl.sh start`; evals interrupted by that are re-run
  from scratch.

## Reporting

Table: base vs b10/b25/b50 on MATH-500, GSM8K, IFEval and MATH-500 hit-limit share, with n; Gemma control; Δ with
95% CIs; cost per run; weights changed; what Claude changed, from its final `EDIT:` line and transcript; audit
verdicts. Figure `docs/assets/figures/exp8_budget.png`.

## Pre-run changes

### 2026-11-14 02:55: practice budget counted in 64-problem units (decided by the coordinator for the author)

- **Change.** A budget of N means **N forward passes of 64 problems = 64·N practice problems** in total. A call may
  use 1–512 problems and costs ceil(n/64) passes. A call that would exceed what is left is refused, with no charge.
  Failed or cancelled calls are refunded, as before. The prompt and tool description say this plainly (budget 10 =
  up to 640 problems; a 512-problem call uses 8). `budget_status()` reports passes and problems used and left.
  The b10/b25/b50/gemma-b25 budgets are 640/1,600/3,200/1,600 practice problems.
- **Reason.** The original's protocol was "average batch of 64 sequences, max 512", so its compute was about
  64·N sequences per budget N. Counting calls instead let Claude spend 512 problems per call: a 512-problem call
  takes about 60 min on our service, a 64-problem call about 9 min. That projected about 10/25/50 h for b10/b25/b50,
  beyond the 48 h SLURM limit. Units keep the original's average compute and keep runs inside the limit.
- **Abandoned run.** The first pilot, `logs/exp8/20261113-234706-2456-b10` ($0.23, 7 turns, 1 completed call of 512
  problems under the per-call rule), is abandoned and **excluded from all results**. Its logs are kept and its
  result.json is marked `status: abandoned`. The restarted b10 is the pilot.
- **Also changed (scheduling, not grading):** final evals and the base re-eval never run while a Claude run is active
  (the service runs in a practice-only mode during runs). The base re-eval started at 23:36 was lost and is re-run
  in eval mode. Base and edits are paired by identical seeds and settings, as pre-registered.

### 2026-11-14 ~04:00: added analysis item (analysis only, no change to scoring): is the edit a temperature change?

Added before b25 and b50 started (during the restarted b10 pilot) and kept out of Claude's prompt, which stays
identical across runs. Reason: the pilot's edit rescales the final RMSNorm, `norm.weight ← s·(1 + w) − 1`. That
multiplies every output logit by s, which is mechanistically the same as changing the sampling settings, and we
already showed that temperature changes looping. The original's edit was structural (one direction removed from
64 output columns of the layer-15/19 MLPs).

**Exact equivalence.** vLLM (0.33, `v1/sample/sampler.py`) applies the presence penalty to the logits, then divides
by the temperature, then applies top-k/top-p. Logits scaled by s, with temperature T and presence penalty p, are
therefore identical to the unedited model at **temperature T/s and presence penalty p/s**. The final norm feeds only
the (tied) output layer, so nothing else changes. Up to bf16 rounding of the edited norm, it is the same
distribution.

**For every final edit (b10, b25, b50, Gemma):**
1. **Decompose.** Fit the final-norm change as a uniform scale s = median((1 + w_new)/(1 + w_old)) and report the
   residual (max and RMS relative deviation from a uniform scale). List every other changed tensor with its share
   of ‖Δ‖ (relative to ‖W‖). Also check for an embedding/output-layer scale (tied, so it also scales inputs) or
   norm-weight scales elsewhere.
2. **Classify.**
   - **pure temperature-like:** the only change is a (near-)uniform final-norm or output-layer scale (residual
     < 1% RMS);
   - **structural:** no final-norm/output scale component (|s − 1| < 0.01);
   - **mixed:** both.
3. **Matched-temperature control** for every edit with |s − 1| ≥ 0.01: evaluate the **unedited** model on MATH-500
   (4 samples, same seeds) at temperature 0.6/s and presence penalty 1.5/s (Gemma: 1.0/s and 0/s, top-k 64).
   Report the **share of the edit's MATH-500 gain the control reproduces**, (acc_control − acc_base) /
   (acc_edit − acc_base), and the same for the hit-limit share, with paired-bootstrap CIs. If time allows, the
   control also runs on GSM8K and IFEval (2 samples).
   - ≥ 80% reproduced: "the gain is a temperature effect";
   - < 50% reproduced: "the edit does more than change temperature";
   - in between: "partly".
   For a pure temperature-like edit, the control should reproduce ~100%, which also checks the equivalence.
4. In the write-up, an edit is called a loop fix "like the original's" only if it is structural or mixed and the
   matched control reproduces < 50% of its gain. A temperature-like edit is reported as such, next to the
   original's direction removal.

### 2026-11-14 04:55: harness fix after the pilot (no change to scoring)

In the b10 pilot, Claude tried 3 times to save an edited embedding table (248,320 × 1,024 float32, ~1 GB) and each
cell hit the 300 s sandbox timeout before the save reached the parent. The cause was our harness, not the task:
the parent read the child's pipe with `bytes += chunk` (quadratic for a 1 GB pickle). Fixed in
`harness/sandbox.py` (bytearray buffer, 16 MB reads) and `harness/_sandbox_worker.py` (`readinto`). A 1 GB
embedding save now takes about 104 s end to end, mostly the upload. b25 was restarted from its saved state with the
fix (`--resume`, turn 2, no practice calls used). The b10 run stays as the pre-registered single b10 run, but it is
reported with the caveat that embedding/output-layer edits were effectively unavailable to it. Claude worked around
this by editing the late-layer outputs instead.

### 2026-11-15 ~23:00: implementation note for the matched-temperature control (no change to the analysis)

vLLM rejects presence_penalty > 2, so "base at temperature 0.6/s with presence penalty 1.5/s" cannot be passed as
sampling settings for s = 0.6 (2.5) or s = 0.68 (2.21). The control is run instead as the weight edit that holds only
the final-norm component of Claude's edit (its own saved `norm.weight`). By the vLLM order of operations (penalty,
then temperature) this gives exactly the same sampling distribution, so the analysis is unchanged. The s = 0.6
norm-only edit is byte-identical to the edit Claude tested at practice call 2 of b10 and b25 (`316cb0a863429162`).
It runs at full 4/2/2. Gemma's control (T 1.1628, presence 0) runs as a sampling override.
