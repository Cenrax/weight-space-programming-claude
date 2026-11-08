# Exp 1 — pre-registered grading rules

Written **before the first API call** of this experiment (2026-11-07). These rules must not change once results are
seen. Any change made after the pilot but before the full run goes under "Pre-run changes" at the bottom, with the
reason.

## Question

Can Claude Opus 5.5 (`claude-opus-5-5`, adaptive thinking, effort `high`, `max_tokens` 64000, single turn, no tools)
name the Atari 2600 game a PPO policy was trained on, from its raw weights alone, with no candidate list?

## Networks (seed 0 only)

| key | checkpoint | true game (ALE id) |
|---|---|---|
| breakout | `checkpoints/exp1/breakout_s0.pt` | `ALE/Breakout-v5` |
| pong | `checkpoints/exp1/pong_s0.pt` | `ALE/Pong-v5` |
| mspacman | `checkpoints/exp1/mspacman_s0.pt` | `ALE/MsPacman-v5` |
| freeway | `checkpoints/exp1/freeway_s0.pt` | `ALE/Freeway-v5` |

All four share one architecture: 128→64→64→18 tanh MLP. Input is RAM/255, output is 18 logits over the full ALE action
set, and the policy acts by greedy argmax. Every network has the same shape and interface, so the shape cannot reveal
the game.

The prompt says the policy plays "an Atari 2600 game". It describes the RAM input, gives the 18 ALE action names in
standard order, and says no tools are allowed. It does **not** list candidate games.

## Conditions and attempts

| condition | networks | attempts | total |
|---|---|---|---|
| `main` | the 4 trained networks, as saved | 5 each | 20 |
| `permuted` | each trained network with the 64 units of both hidden layers permuted consistently, using `numpy.random.default_rng(4000 + i)` for network index i (breakout 0, pong 1, mspacman 2, freeway 3). The function is unchanged: the runner asserts that the max abs logit difference on 4096 random inputs `U[0,1]^128` is < 1e-5. Input rows and output columns are not permuted. | 2 each | 8 |
| `random_init` | two untrained 128→64→64→18 nets: (a) PyTorch default init, `torch.manual_seed(5001)`; (b) the PPO training init from `train/ppo.py` (orthogonal, gain √2 on hidden layers and 0.01 on the head, zero biases), `torch.manual_seed(5002)` | 1 each | 2 |

Each attempt is an independent single-turn call. The prompt template is identical across calls; only the weights
block differs. All 30 calls use the same system prompt, model, effort and `max_tokens`.

**Pilot.** The first call is a pilot: `main`, breakout, attempt 1. If the prompt and settings are unchanged after the
pilot, it counts as breakout attempt 1. If anything changes, the pilot is excluded: it is moved to
`logs/exp1/_pilot_excluded/` and not graded, and breakout attempt 1 is re-run.

## Answer extraction (automatic)

1. **Primary:** the last line of the response matching `ANSWER:` followed by a JSON object. Read `game` (string) and
   `confidence` (number, clipped to [0, 1]).
2. **Fallback**, if no ANSWER line parses: the last `Game: <...>` label in the text, plus the last `confidence: <number>`.
   The run records which path was used (`parse`).
3. If nothing can be extracted, the answer is `unparsed` and is graded **incorrect**. For a random-init run, a
   confidence of null also counts as incorrect.

**Only the head answer is graded:** the `game` string, not the reasoning. If the reasoning mentions the right game as
an alternative, it earns no credit.

## Classification of the head answer (strict)

Normalise first: lowercase, map unicode dashes to `-`, collapse whitespace. Then match these regexes against the
normalised `game` string:

| class | counts as this class | notes |
|---|---|---|
| breakout | `\bbreakout\b` | "Super Breakout" is a different cartridge, so it is **wrong**. Arkanoid, Video Olympics and similar are **wrong**. |
| pong | `\bpong\b`, `video olympics` | Video Olympics is the 2600 cartridge that ALE's Pong uses. Tennis and similar are **wrong**. |
| mspacman | `\bms\.?\s*-?\s*pac[\s-]*man\b`, `\bmspacman\b`, `\bmrs\.?\s*pac[\s-]*man\b` | Covers "Ms. Pac-Man", "Ms Pac-Man", "MsPacman", "Ms. Pacman", "MsPacman-v5". **Plain "Pac-Man" / "Pacman" is wrong**: it is a different 2600 cartridge, and ALE now ships it separately as `ALE/Pacman-v5`. |
| freeway | `\bfreeway\b` | Frogger and similar are **wrong**. |
| random | `untrained`, `random weights`, `random init`, `randomly initiali[sz]ed`, `not trained`, `random initiali[sz]ation`, `never trained` | |
| other | anything else | |

**Hedges are wrong.** If the head answer matches more than one class (e.g. "Breakout or Pong"), it is classified
`hedged` and graded **incorrect**, whatever order the names appear in.

The one exception is a game class plus a qualifier like "(not random)". To keep it mechanical: when both `random`
and exactly one game class match, the game class wins only if the random alias is preceded by "not " or "non-".
Otherwise the answer is `hedged`.

## What counts as correct

- `main` and `permuted`: correct iff the class equals the network's true game. `random`, `hedged`, `other` and
  `unparsed` are incorrect.
- `random_init`: correct iff the class is `random`, **or** the confidence in a named game is ≤ 0.30. The named game
  and confidence are recorded either way.
- Permutation consistency is reported but not graded: whether each permuted answer equals the modal main answer for
  the same network.

## Decision rule (fixed now)

- **Chance level.**
  - With no candidate list, Claude could name any of the 60+ ALE games. Uniform chance is about 1/60 per attempt.
  - These four are among the most famous Atari games, so a prior-driven guesser would put a lot of mass on them.
  - The conservative null is therefore **1/4** per attempt: a guess among these four networks' games.
- **Per game.** A game counts as **read** if both hold:
  - main accuracy is ≥ 3/5 (P ≈ 0.10 under p = 0.25);
  - **specificity**: that game's name is the head answer in at most 5 of the 15 main attempts on the *other* three
    networks.

  The specificity check stops "always says Breakout" from counting as reading Breakout. (exp1b showed this failure:
  LunarLander was the answer in 26/28 runs.)
- **Overall.** "Claude reads game identity from the weights" if **≥ 3 of the 4 games are read**. If 2 games are
  read, the result is reported as "partial". With 0–1, it is reported as "not shown".
- **Default-guess flag.** If one game name is the head answer in ≥ 14 of the 28 main + permuted attempts, a
  default-guess failure is reported, whatever the above says.
- **Shape-matching check (descriptive).** We report how many responses justify the answer by the interface alone
  (RAM size / 18 actions), and how many cite specific RAM bytes or action columns. This is a qualitative note, not
  pass/fail.

## Reported metrics

- Per game: main correct/5, permuted correct/2, mean confidence (main), most common wrong answer (main + permuted).
- Each network's greedy eval score next to the uniform-random baseline (`checkpoints/exp1/random_baseline.json`).
- Random-init: answers, confidences, and correct/2.
- Total cost, from `logs/exp1/*/result.json`.

## Budget

- Hard cap: **$20** for this experiment.
- Before launching each call, the runner sums `cost_usd` over `logs/exp1/*/result.json`. It does not launch if
  `spent + (in_flight + 1) × est` would exceed $20, where `est` is the largest per-call cost observed so far in this
  experiment (floor $0.30).
- `ask(..., max_cost_usd=1.5)` is passed per call. For a single-shot call this is recorded only. It cannot cut off a
  streaming reply.
- If the pilot implies a full-run estimate above $20, effort goes down to `medium` or attempts are reduced, and that
  change is recorded below as a pre-run change.
- The project-wide cap (`BudgetExhausted`) is also respected. If it is hit, the run stops and the gap is reported.

## Pre-run changes

- **Pilot result (2026-11-08 00:00, before the full run).**
  - The pilot used 36,451 input and 2,425 output tokens and cost $0.194. The ANSWER line parsed (`parse=answer_line`)
    and the logs are complete.
  - Projected full cost: about 30 × $0.2–0.3, so roughly $6–9, which is under $20. **No change**: effort stays `high`,
    attempts stay 5/2/2, and the prompt and settings are unchanged.
  - As pre-registered, the pilot counts as `main` breakout attempt 1.
