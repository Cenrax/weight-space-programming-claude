# Exp2 grading (pre-registered)

Written 2026-11-07, BEFORE any API call for this experiment. These rules must not be changed after seeing results.
Pre-run changes (made after the pilot, before the full run) are appended in the last section, with reasons.

## What is graded

Each run returns free text. From it we extract:

1. **The description**: the `description` field of the final line `ANSWER: {"description": "...", "confidence": 0.x}`
   (we take the last line starting with `ANSWER:`). If that JSON can't be parsed, we fall back to the text after
   `"description":` up to the next quote. If there is no ANSWER line at all, the description is empty and the run counts
   as **incorrect** (parse failure is logged).
2. **The confidence**: the `confidence` field (float, clipped to [0, 1]); null if missing.
3. **The drawing**: the LAST block of exactly 16 consecutive lines that each consist of exactly 16 characters drawn from
   `#` and `.` (after stripping whitespace and backticks). If there is no such block, the drawing is missing and the
   drawing metric is 0 (incorrect).

## Signal 1 (PRIMARY): text label

Aliases, matched case-insensitively as whole words (regex `\b<alias>\b`), on the description only:

| class | aliases |
|---|---|
| circle | circle, circles, circular, disc, discs, disk, disks, round, ring, rings |
| square | square, squares, rectangle, rectangles, rectangular, box, boxes |
| triangle | triangle, triangles, triangular |

Notes: `rounded` is NOT an alias (so "rounded square" is a square), and `oval`/`ellipse` are NOT aliases.

**Predicted label** = the class whose alias occurs EARLIEST in the description; `none` if no alias occurs.
**Correct** iff predicted label == the network's true shape. (We also log every class mentioned, in a `labels_mentioned` field.)

## Signal 2 (SECONDARY, reported separately): the drawing

`eval/exp2_judge.py: score_answer(drawing, target)`. This is template IoU against filled AND outline templates of each
class, so an outline drawing counts. Drawing correct (`score` = 1) iff the judge's argmax class == the true shape,
otherwise 0 (also 0 for a missing drawing).

## Conditions

- **main**: the 3 trained checkpoints, 5 attempts each (15 runs). We report text-label accuracy per shape and overall,
  drawing accuracy per shape and overall, and mean confidence.
- **permuted**: for each trained checkpoint, the hidden units of BOTH hidden layers are permuted consistently (fixed seed
  per shape), and we verify the function is unchanged (max |Δoutput| < 1e-4 on 256 random inputs). 2 attempts each
  (6 runs). Graded exactly like main.
- **random-init**: 2 freshly initialised, untrained networks with the same architecture (PyTorch default init, seeds
  1001 and 1002), 1 attempt each. Same prompt. **Correct** iff the description contains (whole-word, case-insensitive)
  one of: `untrained`, `not trained`, `random`, `noise`, `noisy`, `nothing`, `no coherent`, `incoherent`, `unstructured`,
  `no structure`, `no meaningful`, `meaningless`, `initialization`, `initialized`, `initialised`. Naming a shape *as
  well as* such a word still counts as correct (e.g. "random noise, no clear circle"). We also record the judge's reading
  of its drawing, but it isn't scored.

## Prompt and settings

- One shared prompt for every condition (`experiments/exp2/run.py: build_prompt`). Weights come from
  `harness.weights_text.weights_to_text(..., quantize=0.001)` (integers, value = int × 0.001).
- Model `claude-opus-5-5`, thinking adaptive (summarized), `effort="high"`, `max_tokens=64000`, no tools, single turn.
- At most 3 concurrent calls. Experiment budget: $15 hard cap (the runner refuses to start a call if the logged exp2 spend
  plus a per-call reserve would exceed it).

## Pre-run changes (after the pilot, before the full run)

Pilot (1 call, main / circle / attempt 1): 73,581 input tokens, 4,941 output tokens, $0.393, stop=end_turn; ANSWER
line and drawing parsed fine. Projected full cost 23 calls x ~$0.40 = ~$9.5 < $15, so **no changes**: effort stays
`high` and attempt counts are unchanged. Because the prompt and settings did not change, the pilot run counts as main /
circle / attempt 1 (as planned in run.py; recorded here before the remaining 22 calls).
