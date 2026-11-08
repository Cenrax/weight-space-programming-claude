# Exp 1b — pre-registered grading rules

Written **before the first API call** of this experiment (2026-11-07). The rules below are not to be changed after
results are seen. Any change made after the pilot but before the full run is logged under "Pre-run changes" at the
bottom, with the reason.

## Question

Can Claude Opus 5.5 (`claude-opus-5-5`, adaptive thinking, effort `high`, no tools) name the RL benchmark
environment that a policy network was trained on, from its raw weights alone?

## Networks

| key | checkpoint | true environment |
|---|---|---|
| cartpole | `checkpoints/exp1b/cartpole.pt` | CartPole-v1 |
| acrobot | `checkpoints/exp1b/acrobot.pt` | Acrobot-v1 |
| mountaincar | `checkpoints/exp1b/mountaincar.pt` | MountainCar-v0 |
| lunarlander | `checkpoints/exp1b/lunarlander.pt` | LunarLander-v3 |

All are 8→32→32→4 tanh MLPs (observations zero-padded to 8, actions padded to 4). The prompt does not list
candidate environments and does not mention padding.

## Conditions and attempts

| condition | networks | attempts | total |
|---|---|---|---|
| `main` | the 4 trained networks, as saved | 5 each | 20 |
| `permuted` | each trained network with the hidden units of both hidden layers permuted consistently (function unchanged, verified numerically on 4096 random inputs, max abs logit difference < 1e-5) | 2 each | 8 |
| `random_init` | 3 fresh `nn.Sequential(Linear(8,32), Tanh, Linear(32,32), Tanh, Linear(32,4))` with PyTorch default init, `torch.manual_seed(1001/1002/1003)` | 1 each | 3 |

Permutation seeds: `numpy.random.default_rng(2000 + i)` for network index i (cartpole 0, acrobot 1, mountaincar 2,
lunarlander 3); one permutation per network, shared by both permuted attempts.

Every attempt is an independent single-turn call with the identical prompt template (only the weights block differs).
Same system prompt, model, effort and `max_tokens` (64000) for all 31 calls.

**Pilot.** One pilot call is made first: `main`, cartpole, attempt 1. If the prompt and settings are unchanged after
the pilot, the pilot counts as cartpole attempt 1. If anything changes, the pilot is excluded (moved to
`logs/exp1b/_pilot_excluded/`, not graded) and cartpole attempt 1 is re-run.

## Answer extraction (automatic)

1. **Primary:** the last line in the response that matches `ANSWER:` followed by a JSON object. Read `environment`
   (string) and `confidence` (number; clipped to [0, 1]).
2. **Fallback** (no parseable ANSWER line): look in the text for `environment: <...>` / `Environment: <...>` (last
   occurrence) and `confidence: <number>`; if still nothing, classify the last 1500 characters of the response
   with the alias lists below, picking the alias that occurs **last**. Fallback use is recorded per run
   (`parse = "fallback_*"`).
3. If nothing can be extracted the answer is `unparsed` and is graded **incorrect** (and, for random-init, as
   not-random with confidence null → incorrect).

## Classification of the extracted environment string

Normalise: lowercase, replace `_`, `-`, `/` and whitespace runs by a single space. Then match the alias lists
(substring match on the normalised string, also matched with all spaces removed). If aliases of more than one
class match, the class whose alias occurs first in the string wins (e.g. "CartPole (not Acrobot)" → cartpole).

| class | aliases (normalised substrings) |
|---|---|
| cartpole | `cartpole`, `cart pole`, `cart-pole`, `inverted pendulum on a cart`, `pole balancing`, `pole on a cart`, `balancing a pole`, `invertedpendulum` (MuJoCo InvertedPendulum is the same cart-pole task) |
| acrobot | `acrobot`, `two link pendulum swing`, `double pendulum swing`, `two link robot arm swing` |
| mountaincar | `mountaincar`, `mountain car` (includes MountainCarContinuous: same task) |
| lunarlander | `lunarlander`, `lunar lander`, `lunar landing`, `moon lander` (includes LunarLanderContinuous) |
| random | `untrained`, `random weights`, `random init`, `randomly initialized`, `randomly initialised`, `not trained`, `random initialization`, `random initialisation` |
| other | anything else (e.g. `Pendulum-v1`, `Taxi`, `FrozenLake`, a MuJoCo locomotion task) |

Note: plain `Pendulum-v1` / "pendulum swing-up" is **not** cartpole and **not** acrobot → `other`.

## What counts as correct

- `main` and `permuted`: correct iff the class equals the network's true environment. A `random` answer is incorrect.
- `random_init`: correct iff the class is `random`, **or** the confidence in the named environment is ≤ 0.30.
  The named environment and confidence are recorded either way.
- Permutation consistency (reported, not a pass/fail): whether the permuted answers equal the modal main answer for
  the same network.

## Reported metrics

Per network: main correct/5, permuted correct/2, mean confidence (main), most common wrong answer (main + permuted).
Random-init: answers, confidences, correct/3. Total cost from `logs/exp1b/*/result.json`.

## Budget

Hard cap $10 for this experiment. The runner checks the summed `cost_usd` of `logs/exp1b/*/result.json` before
each call and stops if launching another call could exceed the cap (assuming the worst observed per-call cost).
If the pilot implies a full-run estimate > $10, effort is reduced to `medium` or attempts are reduced, and that is
recorded below as a pre-run change.

## Pre-run changes

- **Pilot result (2026-11-07 23:03, before the full run):** 7,673 input / 1,098 output tokens, $0.053, ANSWER line
  parsed, logs complete. Projected full cost ~31 x $0.06 = ~$2 < $10, so **no change**: effort stays `high`, attempts
  stay 5/2/3. Prompt and settings unchanged, so the pilot counts as `main` cartpole attempt 1 (as pre-registered).

---

# v2 (pre-registered 2026-11-07 23:36 +0530, before any v2 API call)

**Why v2 exists.** In v1, every network was padded to 8 inputs and 4 actions, which is LunarLander's native
interface. Claude answered "LunarLander" in 26 of 28 trained-network attempts, mostly from the shape alone. v2 pads to
**10 inputs and 5 actions** (`env_kind = pad10x5`), a shape no standard benchmark uses natively.

**Networks.** `checkpoints/exp1b_v2/{cartpole,acrobot,mountaincar,lunarlander}.pt`, arch 10→32→32→5, tanh, PPO.
Observations are zero-padded at the end to 10. Actions are padded to 5 with `ACTION_MAPS_5` (`envs/wrappers.py`):

| env | map |
|---|---|
| CartPole | [0,1,0,1,0] |
| Acrobot | [0,1,2,1,1] |
| MountainCar | [0,1,2,1,1] |
| LunarLander | [0,1,2,3,0] |

**Prompt.** Identical to v1 except for the architecture numbers: "an input vector of dimension 10 … an output layer of
5 action logits … argmax of the 5 logits". The system prompt, model, effort `high` and `max_tokens` 64000 are the same.
The prompt still doesn't mention padding or list candidates.

**Conditions.**

| condition | what | attempts |
|---|---|---|
| `main_v2` | the 4 networks | 5 each (20) |
| `permuted_v2` | hidden-unit permutation as in v1, seeds `default_rng(3000 + i)`, function verified identical (< 1e-5) | 2 each (8) |
| `random_init_v2` | fresh 10→32→32→5 nets, PyTorch default init, `torch.manual_seed(2001/2002/2003)` | 3 |

No pilot exclusion: the first v2 call is `main_v2` cartpole attempt 1, and it counts as such.

**Grading.** The same extraction and alias rules as v1, and the same correctness rules (random-init correct = "untrained/random" or
confidence ≤ 0.30).

**Decision rule (fixed now).**

- **Chance level.** Treat it as 1/4, i.e. a guess among these four networks' environments. Claude is not told the
  candidate set and could name any benchmark, so the true chance rate is lower and this test is conservative.
- **Per network.** Main accuracy ≥ 3/5 counts as evidence for that network (P ≈ 0.10 under p = 0.25). At 4/5 or more,
  it counts as strong evidence (P ≈ 0.016).
- **Overall.** v2 counts as **evidence that Claude reads the weights beyond the shape** only if both of these hold:
  - main_v2 total ≥ 10/20 (binomial P ≈ 0.014 under p = 0.25);
  - at least 2 of the 3 non-LunarLander networks (CartPole, Acrobot, MountainCar) individually reach ≥ 3/5.

  This requirement prevents the result from passing on a single default guess that happens to match one network.
- **Default guess.** If one environment name is given in ≥ 15 of the 28 trained-network v2 attempts, this is reported as a
  default-guess failure regardless of the above.
- **Padding mentions (descriptive, not pass/fail).** We report the fraction of main_v2 + permuted_v2 responses that
  mention padding or unused inputs. The case-insensitive keywords are `pad`, `always zero`, `always-zero`,
  `unused input`, `never used`, `constant zero`, `zero-padded`.

**Budget.** Hard cap of $5 for v2: the summed cost of the `*_v2` runs.

**v2 run record (post-run, procedural only; no rule changed).** The full v2 run stopped before the 3 `random_init_v2`
calls. One Acrobot call cost $0.61, and the in-flight worst-case estimate went over $5. The 3 calls were resumed
with `--version v2 --condition random_init`; spent plus 3 × $0.61 was still under $5. Final v2 spend: $3.13.
