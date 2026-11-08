# Experiment 5b: pre-registered grading

Written 2026-11-08, before any API call for exp5b. It will not be changed after results are seen. Changes decided
after the pilot but before the main runs go in "Pre-run changes" at the bottom, each with a reason.

## Question

Exp5 asked Claude Opus 5.5 to write a policy network's weights from scratch in one reply, with no tools. Does
feedback from the real environment help? This is the blog's Experiment 6 protocol (edits interleaved with a fixed
budget of rollout batches; score as a function of budget), applied to the exp5 task.

## Environments, networks, start state

Same as exp5. The architectures are those of `checkpoints/exp5/<env>_100k.json`, the policy is greedy argmax, and
ties go to the lowest index.

| env | arch | input |
|---|---|---|
| Acrobot-v1 | 6→64→64→3 tanh | raw float32 obs |
| LunarLander-v3 (discrete) | 8→64→64→4 tanh | raw float32 obs |
| Breakout (ALE v5, sticky 0.25, frameskip 4, 18 actions, 27k-step cap) | 128→64→64→18 tanh | RAM/255 |

The saved weights start **all zero**, biases included.

## Conditions

- `budget_5` and `budget_25`: the maximum number of `rollout()` calls.
- Budget 0 is the exp5 zero-shot result (median and best over 3 Acrobot / 5 LunarLander / 5 Breakout attempts),
  reused as-is. It is not rerun.
- One attempt per (env, budget): **6 runs**.
- **Extra attempt rule (fixed now):** after the 6 runs, if `12 − exp5b spend ≥ $2.20` (the budget-25 per-run cap),
  run a second `budget_25` attempt on **Breakout**. Breakout is the env with the largest exp5 zero-shot spread
  relative to its PPO-final − random gap: range 26.7 / 26.1 = 1.02, against Acrobot 0.77 and LunarLander 0.02. With
  two budget-25 attempts, the env's budget-25 score is their **mean**, and both are shown.

## The sandbox and the rollout tool (budget semantics)

- `harness.sandbox.Sandbox` (isolated child process, macOS `sandbox-exec`, no repo or network access, gym imports
  blocked). It contains numpy, the currently saved weights (`W1, b1, …`), `save_weights(...)` (free), the
  `save_weights` text tool (free), `load_saved_weights()`, `budget_status()` and `rollout(...)`.
- **`rollout(n=32, trace_obs=None)`** runs the **currently saved** weights greedily in the real environment, in
  the parent process, on `n ≤ 32` episodes.
  - It counts as **one** call of the budget whatever `n` is. A call that raises is refunded (harness rule).
  - It returns a compact summary: calls used/left, mean/std/min/max return, all per-episode returns (rounded),
    the success rate (Acrobot: goal reached; LunarLander: return ≥ 200, plus outcome counts; Breakout: none,
    plus outcome counts), mean/min/max length, action frequencies, and subsampled traces of 2 episodes (the best
    and the worst of the batch).
  - A trace has ≤ 30 rows of (step, observation, action, return so far), taken every k steps, plus the last step.
  - Classic envs: the full observation, rounded to 3 decimals. Breakout: only the RAM bytes listed in
    `trace_obs`, as integer byte values 0–255 (none by default). We still give no RAM map.
- **Rollout seeds:** call k (1-based) of every run uses env seeds `100000 + 1000·k + i`, i = 0..n−1. These are
  fresh for each call, disjoint from the final-eval seeds 10000–10099, and identical across runs, so the first 5
  calls of a budget-25 run see the same episodes as a budget-5 run.
- **Breakout rollout cap:** in rollouts only, an episode is truncated after 5,000 agent steps. This keeps a call
  that never serves under about a minute. The prompt states it. The final evaluation keeps the normal 27k cap.
- **Rule given to Claude, matching the blog:** it may analyse and construct weights freely with numpy. It may
  **not** run the network (or any part of it) on inputs in the sandbox, and it may **not** re-implement or simulate
  the environment's dynamics. All behaviour feedback must come through `rollout()`.
- Turn and cost caps: budget 5 gets 30 tool turns and $1.20 max; budget 25 gets 70 tool turns and $2.20 max.
  `max_tokens` is 32,000 per turn, and effort is `high`, the same as exp5. When a cap is hit, the harness asks for
  the final answer, and the saved weights at that moment are final.
- The prompt reveals no PPO scores, does not mention the original blog, and omits the zero-weight template (the
  weights are in the sandbox).

## Final metric

- **Score:** the mean return of the **final saved weights** (the sandbox state when the run ends, *not* the best
  weights seen) over **100 greedy episodes, env seeds 10000–10099**, via exp5's `score_net` (`eval.rollout.evaluate`,
  normal 27k Breakout cap). This is identical to exp5.
- If Claude never saves weights, the score is the all-zero network's: Acrobot −500, LunarLander −131.5, Breakout 0.
- `success_rate`: Acrobot goal reached; LunarLander return ≥ 200; Breakout null.
- `correct`: the score is strictly greater than PPO-100k (Acrobot −90.8, LunarLander −1495, Breakout 2.18).
- Also recorded: `rollouts_used`, and `score_curve` (the mean return of every rollout call, in order, exactly as
  Claude saw it).
- `answer`: the `CONTROLLER:` line from the final reply (≤ 300 chars).

## Comparison table (per env)

Random · PPO-100k · PPO-1M · PPO-final (5M for Acrobot, 10M for the others) · exp5 zero-shot median · exp5
zero-shot best · budget 5 · budget 25. All are on the same 100 seeds. The baselines come from
`experiments/exp5/baselines.json`.

## Decision rule

- **D1 (primary): does score increase with budget?** An env counts as "increasing" if
  `score(budget_25) > score(budget_5) > exp5 zero-shot median`. D1 **passes if ≥ 2 of 3 envs** are increasing.
- **D1-weak (secondary):** the count of envs where `score(budget_25) > exp5 zero-shot median`.
- **D2:** the count of envs where any budget's score ≥ PPO-final, listed with budget. LunarLander's specific
  question is whether feedback pushes it past 266.1 (PPO-10M) and past Gymnasium's heuristic (256.6).
- All are reported regardless of outcome. With one attempt per cell, a single difference is a noisy point estimate
  rather than an effect size. The 100-episode std is reported so readers can judge.

## Audit (rule compliance)

After each run, every `python` call in `transcript.jsonl` is scanned for patterns of a forward pass (`tanh(` together
with `@` or `dot(` applied to the weight arrays) and of an environment simulation (`rk4`, `dsdt`, `def step`,
integrating the dynamics). Flagged calls are read by hand.
- A confirmed violation is reported in the notes and the table, with the score kept.
- The decision rule is then reported both with and without the violating runs.

## Budget

- Hard cap **$12** for exp5b (sum of `cost_usd` over `logs/exp5b/*/result.json`), plus the project-wide
  `BudgetExhausted` cap.
- Before each run, the runner checks spend + in-flight caps + this run's cap ≤ $12. If not, it skips.
- At most 2 runs at once. Rollouts use ≤ 3 processes, under `taskpolicy -a` and `VECLIB_MAXIMUM_THREADS=1`.
- **Pilot:** Acrobot `budget_5`. If the settings are unchanged afterwards, it counts as that cell. Its cost, turns
  and tool behaviour are checked, the total is projected, and any adjustment is recorded below.

## Outputs

- Per run, `result.json` contains `target, answer, score, correct, label, group, condition, reasoning_excerpt,
  rollouts_used, success_rate, score_curve`, plus the final weights (`final_weights.npz`, `edited.pt` +
  `edited.json`).
- A GIF of the best exp5b run per env goes to `docs/assets/gifs/exp5b_<env>.gif`.
- `logs/exp5b/summary.json` is built into `docs/data/exp5b.json`.

## Pre-run changes

- (after the pilot, before the main runs) **No change to settings, prompt or caps.** A first pilot launch crashed
  in the runner after turn 1 ($0.060, kept in `logs/exp5b/_crashed-20261108-034815-acrobot-b5/` and counted in spend),
  because of a harness keyword clash (`condition` passed twice). That was a runner bug, fixed with no prompt change.
  The rerun pilot (`acrobot-b5-a1`) used 8 turns, all 5 rollouts, 94k total input tokens (mostly cache reads) and
  3.7k output tokens, for **$0.134**. Projection for the other 5 cells: about $0.2–0.5 per budget-5 run and about $0.5–1.5
  per budget-25 run, so ≈ $2–5 in total, under $12. The pilot counts as the Acrobot `budget_5` cell, as
  pre-registered.
