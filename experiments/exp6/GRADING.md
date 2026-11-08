# Experiment 6 (Go): pre-registered grading

Written 2026-11-08, before any API call for exp6. It will not be changed after results are seen. Changes decided after
the pilot but before the main runs go in "Pre-run changes" at the bottom, each with a reason.

## Question

The original post (Experiment 6) asked its model to write the weights of a 243 → 512×6 → 82 Go policy from scratch, in
a Python sandbox where the network cannot be run except through a budget of "forward passes" (each = one batch of 128
games). With 0 passes it beat two small PPO nets (243→32→32→82) more than half the time; with 50 passes it beat the
equal-size PPO net. Does Claude Opus 5.5 do the same?

## Setup (fixed)

- Engine, features, opponents, evaluator: `envs/go`, `eval/go_eval.py`, `checkpoints/exp6/` (see `notes/exp6_prep.md`).
  9×9, area (Tromp-Taylor) scoring, komi 7.0, simple ko, no suicide, 200-ply cap, draw = half a win.
- Network: 243 → 512×6 → 82, ReLU, masked greedy argmax (ties random). Weights start **all zero**.
- Opponents: `small_s0`, `small_s2` (243→32→32→82 PPO, ~2.6M self-play games) and `large` (243→512×6→82 PPO, 0.71M
  games; the original's had 20M, so ours is weaker; deviation noted in `notes/exp6_prep.md`).
- Conditions: forward-pass budget **0, 10, 25, 50**. One run each (4 runs). Budget 0 has no `play_games` tool.
- Model `claude-opus-5-5`, effort `high`, `max_tokens` 64,000 per turn.
- Per-run caps (turns / USD): b0 30 / $5, b10 50 / $7, b25 80 / $10, b50 130 / $14. Experiment cap **$40** (sum of
  `cost_usd` over `logs/exp6/*/result.json`, crashed runs included); the project-wide `BudgetExhausted` cap also applies.
- Sandbox: `harness.sandbox.Sandbox` (isolated child, macOS `sandbox-exec`, no repo/network access, env packages
  blocked) with numpy + torch (CPU), the zero weights `W1..W7, b1..b7` ([in, out]), free `save_weights`, and for
  budget > 0 `play_games(opponent, n_records)` from `eval.go_eval.make_forward_pass_tool`, restricted to the three
  scored opponents, registered with `budget=N`. It plays the **saved** weights; call k uses opening seed
  200000 + 1000·k (never the eval seed). A call that raises is refunded (harness rule).
- The prompt gives the rules as implemented, `FEATURE_DOC` verbatim, the architecture with ReLU and the masked-argmax
  policy, the opponent **names only** (no strength, no training details, no PPO numbers, no mention of the original
  post), how the final score is computed, that the last saved weights are what is scored, the no-training / no
  running / no simulating / no web rules, and the turn limit.

## Metrics

- **Final weights** = `sb.weights` when the session ends (last save), whatever happened. Never saved → the zero net.
- **Win rate vs each opponent**: `eval.go_eval.evaluate_claude(final_weights)`: 128 greedy games vs each of small_s0,
  small_s2, large on `EVAL_SEED = 1_000_003`, colours alternating, 2 shared random opening plies per pair, draw = 0.5.
- **Score** = mean of the three win rates. `correct` = win rate > 0.5 vs **both** small nets.
- Also recorded: Wilson 95% intervals (n = 128), win rate vs the heuristic bot (context only, same seed),
  `passes_used`, `practice_curve` (win rate of every `play_games` call as Claude saw it, with opponent),
  `pct_nonzero_weights`, number of saves, cost, turns.

## Comparisons

1. **Original headline A (0 passes):** budget-0 win rate > 0.5 vs small_s0 **and** vs small_s2.
2. **Original headline B (50 passes):** budget-50 win rate vs `large` > 0.5.
3. **Zero net** (Claude's start) on the same seed, vs each opponent; expected ~1–4%.
4. **Heuristic bot** (`HeuristicPolicy`: capture largest, else random non-self-atari non-eye-fill move, else pass) vs
   each opponent on the same seed, reported for context only.

## Decision rules

- **R-A** passes iff comparison 1 holds (point estimates, as in the original). Also reported: whether both Wilson
  lower bounds exceed 0.5.
- **R-B** passes iff comparison 2 holds. Also reported with the Wilson lower bound.
- **R-C (scaling, secondary):** mean win rate is non-decreasing across budgets 0 → 10 → 25 → 50 (count of the three
  steps that increase; with n = 128 per opponent, a step under ~6 pp in the mean is within noise and is called so).
- **R-D (sanity):** every budget's mean win rate exceeds the zero net's.
- **Replicates the original** iff R-A and R-B both pass. Partial outcomes are reported as such. One run per budget,
  so each cell is a single noisy draw; all results are reported regardless of outcome.

## Rule audit

The prompt forbids (i) training / gradient / optimisation of weights against data, (ii) running the network or any
part of it on inputs, (iii) simulating Go games, (iv) the web. Definitions used to grade every `python` call in
`transcript.jsonl`:

- **Violation: running the network.** Any code that takes an input vector encoding a board position (real, hand-made,
  random, or one-hot "test boards" built as 243-vectors / 9×9 planes), and propagates it through **two or more**
  layers of the weight arrays (or copies of them), or through one layer **and** the ReLU, to obtain hidden
  activations or logits, whether to test, debug or choose a move. Also: reimplementing the forward pass in torch/numpy
  and calling it on positions.
- **Minor breach (reported, score kept, listed by name):** a single-layer *linear* pre-activation (`x @ W1[:, j] +
  b1[j]`) on at most a handful of hand-made inputs, to check arithmetic (the exp5b precedent).
- **Violation: simulating games.** Code that keeps a board state and applies a sequence of moves with captures (a game
  engine: play move → remove dead chains → next move), used to play out games, self-play, or test the policy or a
  candidate strategy.
- **Violation: training.** `backward()`, optimisers, autograd, gradient estimates, least-squares / regression / fitting
  weights to targets computed on sampled positions, or any search that scores weights without `play_games` (a
  search over weight edits that uses `play_games` results is allowed: that is what the budget is for).
- **Allowed (not a violation):** Go-logic code that *constructs* weights from the board's structure, e.g. neighbour
  tables, enumerating which feature patterns around a point mean "this chain has one liberty" or "playing here
  captures", computing thresholds and biases by arithmetic on pattern counts, line/edge/position tables; and any
  analysis of the weight arrays themselves (shapes, sums, norms, counts of non-zeros, products of weight matrices
  with each other, inspecting rows/columns). Checking a gate's threshold by counting how many inputs it sums is
  allowed; plugging a board into it is not.
- **Procedure:** the runner regex-flags code with matmul/dot/einsum/`@`, `relu`/`np.maximum`/`clip(`, `forward`,
  board-construction-then-propagate patterns, move-application loops (`play`, `capture`, `remove`, `legal`, `while`
  over moves), and training words (`backward`, `optim`, `grad`, `lstsq`, `fit`, `minimize`). Every flagged cell is
  read by hand; the per-run verdict (`none` / `minor` / `violation`, one-line reason) goes to
  `experiments/exp6/audit.json` and `result.json` (`audit_verdict`). Also scanned: responses that claim to have
  hand-computed the network's output on a position (hand simulation in prose is not detectable as code; it is noted
  but not a violation unless it substitutes for a forbidden run).
- **Handling:** scores are kept for every run. Decision rules are reported both with all runs and with violation runs
  excluded (an excluded cell counts as "not passed"). Violation runs are not re-run.

## Crashes and budget exhaustion

- A run that crashes, hits its turn/cost cap, or ends with `BudgetExhausted` is scored on whatever was saved at that
  moment (the runner writes `final_weights.npz` in a `finally`). The harness asks for a final answer when a cap is hit.
- A runner/harness bug that kills a run before Claude has done substantial work (< $0.50 spent) may be re-run once
  with the bug fixed and no prompt change; the crashed run stays in `logs/exp6/_crashed-*` and its cost counts.
  A crash later than that is scored as-is.
- Before each launch the runner checks exp6 spend + in-flight caps + this cap ≤ $40; otherwise the cell is not run
  and is reported as "not run (budget)". On project `BudgetExhausted`, stop all launches and report.

## Pilot

Budget 0 is the pilot and counts as the budget-0 cell if settings are unchanged. Check cost, turns, parsing of the
`NETWORK:` line, that saves land, and grading; project the remaining spend; record any cap change below.

## Outputs

`logs/exp6/<run>/` (prompt, transcript, sandbox calls, saves, `final_weights.npz`, `grade.json`, `result.json` with
`target, answer, score, correct, label, group, condition, reasoning_excerpt, winrates, passes_used, practice_curve,
pct_nonzero_weights`), `logs/exp6/summary.json` → `docs/data/exp6.json`,
`docs/assets/figures/exp6_winrate_vs_budget.png`, `notes/exp6.md`.

## Pre-run changes

- (after the pilot, before the main runs) **No change to settings, prompt or caps.** The budget-0 pilot
  (`go-b0-a1`) ran 4 turns (3 python cells + final answer), 86k total input tokens, 27.2k output tokens, **$0.70**,
  stop `end_turn`; the `NETWORK:` line parsed, one save landed, grading ran. Projection: budget 10/25/50 should cost
  roughly $2–4 / $3–7 / $5–12, within the per-run caps and the $40 total. The pilot counts as the budget-0 cell.
  Launch order changed for wall-clock only: budgets 50 and 25 run concurrently first, then budget 10.
