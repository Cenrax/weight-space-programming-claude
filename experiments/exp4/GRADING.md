# Experiment 4 — pre-registered grading

Written 2026-11-07, before any API call for exp4. Not to be changed after results are seen. Pre-run changes
(decided after the pilot but before the main runs) go in the "Pre-run changes" section at the bottom, with a reason.

## Question

Can Claude Opus 5.5 (`claude-opus-5-5`) fix a failing RL policy by editing its weights, in one reply, with no tools,
no code execution and no rollouts?

## Cases (networks under test)

| case | checkpoint | env | arch | obs preprocessing | failure |
|---|---|---|---|---|---|
| `mountaincar` | `checkpoints/exp4/mountaincar_fail.pt` | MountainCar-v0 | 2→64→64→3, tanh | raw gymnasium obs, float32, no scaling (`env_kind="raw"`) | always pushes right, 0/1024 flags |
| `lunarlander` | `checkpoints/exp4/lunarlander_hover.pt` | LunarLander-v3 (discrete, no wind) | 8→64→64→4, tanh | raw gymnasium obs, float32, no scaling (`env_kind="raw"`) | hovers until timeout, 0/1024 landings |

Policy = greedy argmax over the output logits (`eval/rollout.py`, `greedy=True`).

## Request settings

`harness.claude_client.ask`, single user turn, no tools, no system prompt, `thinking=True` (adaptive, summarized),
`effort="high"`, `max_tokens=64000`. The prompt is built by `experiments/exp4/run.py` (saved verbatim to each run's
`prompt.md`). The one episode trace in the prompt uses env seed **0**, outside the evaluation seeds.

## Attempts

**5 independent attempts per case** (same prompt, fresh call each). Up to 3 calls run concurrently.

## Metrics (per attempt)

All on **1024 seeded greedy episodes, env seeds 10000–11023** (`evaluate(sd, meta, n_episodes=1024, seed=10000)`).

1. **Success rate (primary, = `score`)**
   - MountainCar: fraction of episodes that reach the flag (`outcome == "flag"`, i.e. position ≥ 0.5 before the
     200-step limit). This is `evaluate(...)["success_rate"]`.
   - LunarLander: fraction of episodes that end with a successful landing **between the flags**: the episode
     terminates with the lander at rest (`outcome == "landed"` as defined in `envs/wrappers.py`: lander asleep, no
     body contact/crash) **and** the final observation has |x| ≤ 0.2 (the helipad spans world x ∈ [8, 12], which is
     obs x ∈ [−0.2, 0.2]). `evaluate()` does not expose the final x, so `run.py` runs a second loop over the same
     seeds with the same env factory and network that records it; its outcome counts must match `evaluate()`'s
     exactly (checked, and reported if they don't).
   - Also reported for LunarLander, not headline: `landing_rate` (landed anywhere, from `evaluate`) and the fraction
     of episodes with return ≥ 200 (`success_rate_200`, the usual "solved" bar).
2. **Mean return** over the 1024 episodes (true env reward).
3. **% of weights changed**: `harness.weights_text.diff_params(original, edited)["frac"]` (|Δ| > 5e-4).

`correct` = success rate ≥ 0.9.

Per case we report: best, median and mean success rate across the 5 attempts, and the number of attempts with
`correct = True`.

## Parsing Claude's edit

1. Take the text between the **last** `EDITS:` marker and the following `END_EDITS` (or end of reply if missing).
   Code fences are ignored. Inline comments starting with `#` or `//` are stripped from each line.
2. Parse with `harness.weights_text.parse_weights(block, base=original_params, strict=True)`.
3. Every line in the block that begins with `W<k>` or `b<k>` must be a recognised edit (single element
   `W2[3,17] = 1.25`, `b1[4] = -0.5`, whole row `W2 row 3 <64 numbers>`, whole bias `b2 <64 numbers>`, with
   numbers-only continuation lines allowed for wrapped rows).

**Parse failure** = no `EDITS:` block, or `parse_weights(strict=True)` raises, or any `W<k>`/`b<k>` line is not a
recognised edit, or the result contains NaN/inf. A parse failure **counts as a failure**: success rate 0,
`correct = False`, mean return = that of the unedited network. In that case we still record the **partial parse**:
`parse_weights(..., strict=False)` on whatever lines did parse, evaluated on the same 1024 episodes, stored as
`partial_parse_*` fields (never used as the headline score). A refusal or a cut-off reply (`stop_reason` ≠
`end_turn`) without a parseable block is also a failure.

## Fair-comparison baselines (no API calls)

Same 1024 seeds, same metrics:
- **Unedited network** (the failing policy itself).
- **Hand-coded heuristic**, reported for context only:
  - MountainCar: "push in the direction of velocity": action 2 (right) if v ≥ 0 else action 0 (left).
  - LunarLander: Gymnasium's own `heuristic(env, s)` from `gymnasium/envs/box2d/lunar_lander.py` (discrete).
- **Uniform random** policy (already in `checkpoints/exp4/random_baseline.json`).

## Budget

Hard cap **$12** of API spend for exp4 (sum of `cost_usd` over `logs/exp4/*/result.json`), plus the project-wide cap
(`BudgetExhausted`). One pilot call (MountainCar) first, to measure tokens/cost and test the parse + evaluation. The
pilot counts as **MountainCar attempt 1** if it ran with the final settings and prompt unchanged; otherwise it is
reported separately and not scored. If the estimated full cost exceeds $12, lower effort to `medium` or reduce
attempts, recorded below as a pre-run change.

## GIFs

The best attempt per case (highest success rate; ties → higher mean return) is rendered with `eval/render_gif.py`'s
functions on seed 10000 (episode 0 of the eval set) to `docs/assets/gifs/exp4_claude_<case>.gif` (≤ 2 MB).

## Pre-run changes

- (after the pilot, before the main runs) **None to the settings.** The pilot (`mountaincar-a1`) used the final
  prompt and settings (`effort="high"`, 64k max tokens): 25,236 input tokens, 3,817 output tokens, $0.177, parsed
  cleanly. Estimated full cost ≈ 10 × $0.2–0.5 ≈ $2–5 < $12, so effort stays `high`, 5 attempts per case, and the
  pilot counts as MountainCar attempt 1 as pre-registered.
