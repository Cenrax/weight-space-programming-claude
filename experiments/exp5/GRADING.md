# Experiment 5: pre-registered grading

Written 2026-11-08, before any API call for exp5. It will not be changed after results are seen. Changes decided
after the pilot but before the main runs go in "Pre-run changes" at the bottom, each with a reason.

## Question

Can Claude Opus 5.5 (`claude-opus-5-5`) write the weights of a policy network from scratch, zero-shot? It gets one
reply, no tools, no code execution and no rollouts. It is given the environment's rules and source, the
architecture, and an all-zero network as the shape template.

## Environments and networks

The architectures match the exp5 PPO baselines in `checkpoints/exp5/*.json` exactly. Policy = greedy argmax over
the output logits (`eval/rollout.py`, `greedy=True`; ties go to the lowest index, as in `torch.argmax`).

| env | `env_kind` | arch | input |
|---|---|---|---|
| CartPole-v1 | raw | 4→64→64→2, tanh | raw float32 obs |
| Acrobot-v1 | raw | 6→64→64→3, tanh | raw float32 obs |
| MountainCar-v0 | raw | 2→64→64→3, tanh | raw float32 obs |
| LunarLander-v3 (discrete, no wind) | raw | 8→64→64→4, tanh | raw float32 obs |
| Breakout (ALE v5, frameskip 4, sticky p=0.25, full 18 actions, 27k-step cap) | atari_ram | 128→64→64→18, tanh | RAM/255 |

## Prompt (built by `experiments/exp5/run.py:build_prompt`, saved verbatim as each run's `prompt.md`)

- The task: write, from scratch, the weights of a policy network that plays the game as well as possible. The
  prompt states explicitly: no tools, no code, no running the network or the environment, one reply.
- The environment's rules and source:
  - Classic control / Box2D: the trimmed Gymnasium 1.4.0 source, from `experiments/exp5/build_env_docs.py`.
    MountainCar and LunarLander use exp4's builder unchanged.
  - Breakout: the ALE description, the game rules (bricks and points, serve with FIRE, 5 lives), the 18-action
    table, the RAM observation interface (128 bytes / 255), and the `ale_py` `step()`/`_get_obs()` source.
  - **No RAM map is given** for Breakout. Claude must rely on its own knowledge of the game's memory layout. This is
    a caveat: it is a memory test as much as a construction test.
- The architecture, activation, input preprocessing, forward pass, and the action meanings (greedy argmax).
- The network with **every weight at 0** (biases too), printed in the harness row format. This is the shape
  template.
- Answer format:
  - a `CONTROLLER:` line (≤ 300 characters), stored as `answer`;
  - an explanation;
  - one `WEIGHTS:` … `END_WEIGHTS` block. It may contain single entries (`W1[i,j] = v`, `b1[j] = v`), whole rows
    (`W2 row i <values>`) or whole bias vectors (`b3 <values>`). **Unspecified weights stay 0.**
- PPO scores are **not** revealed, and the original blog is **not** mentioned.

## Request settings

`harness.claude_client.ask`, a single user turn, no tools, no system prompt, `thinking=True` (adaptive,
summarized), `effort="high"`, `max_tokens=64000`.

## Attempts

- CartPole, Acrobot, MountainCar: **3** independent attempts each.
- LunarLander, Breakout: **5** each.

That is 19 calls in total. Each is the same prompt, sent as a fresh call. At most 3 calls run at once. The pilot is
CartPole attempt 1. It counts as that attempt if the settings and prompt are unchanged after it.

## Parsing

1. Take the text between the **last** `WEIGHTS:` marker and the following `END_WEIGHTS` (or the end of the reply if
   there is none). Code-fence lines are ignored. Inline `#` / `//` comments are stripped (exp4's lesson).
2. Parse with `harness.weights_text.parse_weights(block, base=zero_params, strict=True)`.
3. Every line in the block that starts with `W<k>` / `b<k>` must be a recognised entry.

A **parse failure** is any of these: no `WEIGHTS:` block, a strict parse error, an unrecognised `W/b` line, NaN or
inf, or a cut-off reply without a parseable block. A parse failure **scores as the zero network**: its score and
success are those of the all-zero network, and `correct = False`. The partial parse
(`parse_weights(strict=False)`) is still evaluated and stored as `partial_parse_*` fields. It is never the headline
score.

## Metrics (per attempt)

All are computed on **100 seeded greedy episodes, env seeds 10000–10099**
(`evaluate(sd, meta, n_episodes=100, seed=10000)`), the same protocol as the PPO checkpoints' `eval_greedy`.

1. **Mean return (primary, = `score`).**
2. **Env-specific success rate (`success_rate`):**
   - CartPole: the episode reaches the 500-step limit (`outcome == "timeout"`).
   - Acrobot: reaches the goal height (`outcome == "swung_up"`).
   - MountainCar: reaches the flag (`outcome == "flag"`).
   - LunarLander: return ≥ 200 (`success_rate_200`, the "solved" bar). `landing_rate` (landed at rest, no crash) is
     also reported.
   - Breakout: no binary success. `success_rate = null`. Max episode score and the outcome counts
     (terminated vs truncated) are reported.
3. **`correct`** = the mean return is strictly greater than PPO-100k's mean return for that env.

## Baselines (no API calls, same 100 seeds)

- PPO at 100k, 1M and final (5M for the classic tasks, 10M for LunarLander and Breakout): the `eval_greedy` stats
  stored in `checkpoints/exp5/<env>_<steps>.json`, recomputed with `evaluate()` as a check.
- Uniform random: `checkpoints/exp5/random_baseline.json`.
- The **all-zero network**: argmax picks action 0 everywhere. It is evaluated with the same protocol.

## Decision rule

Per env, the **median** attempt (by mean return) is the primary statistic, and the **best** attempt is secondary.

- **H1 (primary):** the median Claude attempt beats PPO-100k on **≥ 4 of 5** envs.
- **H2 (headline, the original post's claim):** count the envs where the **best** Claude attempt is ≥ PPO-final's
  mean return. Also count the envs where the median attempt is.
- Both are reported regardless of outcome, along with the best and median for every env.

## Memorisation flag

Does the network implement a known hand-coded controller? For every attempt that parses, take all steps of its 100
evaluation episodes (the states its own policy visits). Measure `heuristic_agreement` = the fraction of steps where
its argmax equals the heuristic's action on the same observation.

- **LunarLander:** Gymnasium's own `heuristic(env, s)` (discrete), from `gymnasium/envs/box2d/lunar_lander.py`.
- **MountainCar:** "push in the direction of velocity": `2 if v >= 0 else 0`.
- Flag = **agreement ≥ 0.90** → "matches a known heuristic".
- **Secondary, context only, not used for the flag:**
  - CartPole: `1 if theta + theta_dot > 0 else 0`.
  - Acrobot: `2 if dtheta1 > 0 else 0` (torque in the direction of link 1's swing).
  - Breakout: no reference controller. Instead, a post-hoc analysis lists which RAM bytes get non-zero first-layer
    weights. It compares them with the AtariARI annotations (`ball_x = 99`, `ball_y = 101`, `player_x = 72`), which
    we checked against the rendered screen on 2026-11-08 (|r| ≥ 0.998).
- The heuristics' own mean returns on the same seeds are reported for context.

## Budget

- Hard cap **$18** of API spend for exp5: the sum of `cost_usd` over `logs/exp5/*/result.json`, plus the
  project-wide cap (`BudgetExhausted`).
- Before each call, the runner checks spent + in-flight estimates + this call's estimate ≤ $18, and skips the call
  otherwise.
- One pilot (CartPole) runs first. If the projected total is > $18, then `effort="medium"` and/or fewer attempts,
  recorded below.

## GIFs

The best attempt per env (highest mean return; ties → higher success rate, then lower attempt number) is rendered
with `eval/render_gif.py`'s functions on seed 10000 to `docs/assets/gifs/exp5_claude_<env>.gif` (≤ 2 MB). Breakout
is capped at the first 300 agent steps of the episode, in real time.

## Pre-run changes

- (after the pilot, before the main runs) **No change to the settings.** The pilot (`cartpole-a1`) used the final
  prompt and settings (`effort="high"`, 64k max tokens): 13,906 input tokens, 2,562 output tokens, $0.107, and it
  parsed cleanly. The projection for the other 18 calls is about $5–14 (LunarLander and Breakout are expected to
  think much longer), which is under $18. So effort stays `high` and the attempts stay 3/3/3/5/5. The pilot counts
  as CartPole attempt 1, as pre-registered.
- Implementation detail, not a grading change: the runner's per-call budget reservation is set to the worst case
  (input + 64k output tokens, ≈ $1.35–1.41). The $18 cap therefore cannot be overshot by calls in flight.
