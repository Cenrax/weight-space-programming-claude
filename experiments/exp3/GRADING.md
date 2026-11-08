# Exp 3 grading rules (pre-registered)

This file was written before any API call for Exp 3. Changes made after the pilot go under "Pre-run changes" at the
end, with the reason. Nothing above that heading is changed after the first call.

## Task

Claude gets the 4 middle transformer blocks of a character-level LM (embedding, positional embedding, final LN and
unembedding removed), in a Python sandbox with numpy/torch. It may analyse the weights but may not run the network or
any layer on inputs. It must end with:

```
ANSWER: {"language": "...", "confidence": 0.x, "induction_heads": ["L.H", ...], "previous_token_heads": ["L.H", ...]}
```

Targets: `python`, `romaji`, `english`. 3 attempts per target (9 runs). Only if the budget allows after these 9,
2 more attempts per target (attempts 4 and 5), with no change to anything else.

## Parsing

- The last line matching `ANSWER: {...}` (markdown `**`/backticks stripped) is parsed as JSON.
- Fallback if no parseable line: `language:` label and `confidence:` label in the final text; head lists are then
  taken as empty (scored as 0 recall). Fallback use is recorded in `parse`.
- Head ids are normalised to `L.H` with L in 0..3 and H in 0..5. Accepted spellings: `3.1`, `L3H1`, `L3.H1`,
  `3,1`, `(3, 1)`, `layer 3 head 1`. Anything out of range is dropped and logged. Duplicates are removed.
- Confidence is clipped to [0, 1].

## Language grading (strict alias match)

The `language` string is lower-cased, punctuation/underscores/hyphens collapsed to spaces, and matched against:

| class | aliases (substring match) |
|---|---|
| python | `python`, `cpython`, `py source` |
| romaji | `romaji`, `rōmaji`, `romanized japanese`, `romanised japanese`, `japanese in latin`, `japanese latin script`, `latin script japanese`, `japanese romanization`, `japanese romanisation`, `hepburn`, `transliterated japanese` |
| english | `english` |

Partial classes (pre-registered):

| partial class | trigger | applies to target |
|---|---|---|
| `code (no language)` | `code`, `source code`, `programming language`, `program` without any Python alias | python |
| `japanese (script unstated)` | `japanese` without any romaji alias (e.g. "Japanese text") | romaji |

Also romaji: the word `japanese` together with `latin`, `roman*` or `alphabet` anywhere in the string (e.g.
"Japanese written in Latin script"), which is the same meaning as "Japanese in Latin script".

Rules:
- Named non-target languages (JavaScript, Java, C, C++, Rust, Go, Ruby, Perl, PHP, Lua, Haskell, HTML, LaTeX,
  Markdown, SQL, shell, German, French, Spanish, Italian, Portuguese, Dutch, Chinese, pinyin, Korean, Indonesian,
  Malay, Finnish, Swahili, Hawaiian, Maori) count as "other" for the first-named rule. "C source code" is wrong,
  not partial.
- If the string names more than one class, the **first-named** class counts (e.g. "Python or JavaScript" -> python;
  "English or German" -> english; "JavaScript or Python" -> other).
- `correct` (bool) is True only for an exact class match. Partial is **not** correct.
- `language_score`: 1.0 for a match, 0.5 for the pre-registered partial class of that target, 0 otherwise.
  The headline is `correct` (strict). `language_score` is reported as a secondary, lenient number.
- A generic answer like "natural language text" or "prose" without a named language scores 0 for every target.
  "Japanese" for an English target is wrong (0). Any non-Python language (e.g. "C", "JavaScript") for the Python
  target is wrong (0), not partial.

## Head grading

Ground truth comes from `checkpoints/exp3/<lang>_heads.json` (made by `eval/exp3_ground_truth.py` from the full model on
random character sequences repeated twice). Threshold 0.3 for both.

**Primary induction ground truth = `copying_heads_unigram_tokens`** (induction score with random characters drawn from
the training unigram distribution, I >= 0.3). Reason, decided before any call: the uniform-token variant has no head
above 0.3 for romaji (max 0.04) or English (max 0.23), because rare characters make the first copy far out of
distribution. That would make precision undefined. `notes/charlm.md` already identifies the unigram variant as the cleaner signal.
- python: {3.1} (I = 0.52; 3.0 is next at 0.19)
- romaji: {3.2} (I = 0.36; next 0.007)
- english: {3.3} (I = 0.57; next 0.04)

**Primary previous-token ground truth = `prev_token_heads`** (uniform tokens, P >= 0.3).
- python: {0.3, 2.0}
- romaji: {0.2}
- english: {0.0}

The unigram variant would add 1.5 (P = 0.308) for romaji and 1.3 (P = 0.301) for English. Both sit right at the
threshold, so the uniform list is primary and the unigram list is reported as a secondary.

Metrics per run, for each of the two head sets:
- `precision = |pred ∩ gt| / |pred|`. If pred is empty, precision is undefined (null) and excluded from the mean.
- `recall = |pred ∩ gt| / |gt|`.
- `F1` (0 when pred is empty). **`score` in result.json = induction-head F1** against the primary ground truth.
- Exact-set match (bool).
- Ranking metrics on the raw scores (unigram induction scores; uniform previous-token scores):
  - `top1_hit`: the head with the highest ground-truth score is in Claude's set.
  - `auc`: the probability that a randomly chosen predicted head has a higher raw score than a randomly chosen
    non-predicted head (ties 0.5). It is null when the set is empty or holds all 24 heads. Chance is 0.5.
  - `mean_gt_score_pred` vs `mean_gt_score_rest`.
- Secondary: precision/recall against the alternative variant (uniform-token induction list, unigram prev-token list).

Chance reference: a random set of k heads out of 24 has expected precision 1/24 ≈ 0.04 for induction.

## No-forward-pass rule and violation audit

The prompt says the network and its layers may not be run on any input. **Violation** = any code that builds
residual-stream activations and pushes them through a part of the network to get activations or attention patterns.
Examples:
- random, synthetic or basis vectors arranged as a sequence of positions, then softmax(QK^T) over positions;
- vectors passed through LayerNorm followed by a further layer;
- vectors passed through the GELU MLP;
- vectors passed through more than one block.

**Not a violation**: any computation on the weight arrays themselves, including:
- products such as W_Q^T W_K, W_O W_V and composition between heads;
- SVD and eigen-decompositions, norms, statistics and clustering;
- applying one weight matrix to another matrix's singular vectors or columns;
- applying elementwise functions to weight-derived quantities, e.g. GELU of the MLP biases.

Audit procedure:
1. The runner scans every `python` tool call for heuristics: random generators (`np.random`, `torch.randn`, `rng.`),
   `softmax`, `gelu`/`erf`/`tanh`, `layer_norm`/`layernorm`, and `forward`. It writes the flagged snippets to
   `logs/exp3/<run>/audit_flags.json`.
2. I read every flagged snippet by hand and record a verdict per run in `experiments/exp3/audit.json`: `none`,
   `borderline` or `violation`, with a one-line reason.
3. **Handling:** a run with `violation` gets `violation: true` in result.json. Its results are reported separately and
   are excluded from the headline numbers, which use clean runs only. Violation runs are not re-run, so no
   resampling-until-clean happens. `borderline` runs count as clean in the headline, but they are listed by name.

## Decision rule (pre-registered)

Claude "replicates the original Exp 3 result" if all three of these hold on clean runs:
1. Language: strict-correct in >= 7/9 runs, and >= 2/3 for every language.
2. Induction heads: mean induction F1 >= 0.67 over all clean runs.
3. Induction heads, per language: `top1_hit` in >= 2/3 attempts for at least 2 of the 3 languages.

It is a **partial** replication if (1) holds and (2) or (3) fails, or if (1) fails but (2) and (3) hold. Otherwise
it does not replicate. Previous-token head scores are reported but are not part of the decision rule (the original
only reported copying heads).

## Run settings

- Model `claude-opus-5-5`, adaptive thinking (summarized), effort `high`, max_tokens 64000 per turn.
- Tools: a single `python` tool (the harness Sandbox, `threads=2`, `preload=("torch",)`, isolation on, 120 s timeout,
  20k-char output cap). No functions are registered other than the harness built-ins. `save_weights` is not offered
  as an Anthropic tool.
- Per-run caps: `max_turns=40`, `max_cost_usd=2.5` (the harness then forces a final answer turn without tools).
- Runner guard: do not launch a run if `spent + in_flight x max(worst observed run, per-run cap) > $25`.
- Concurrency: at most 2 runs.
- Pilot: python attempt 1. It counts as attempt 1 if nothing is changed afterwards. If settings change, the pilot is
  reported separately and attempt 1 is re-run.

## Pre-run changes

- Pilot (python attempt 1, `20261108-000134-9b35-python-a1`): $1.71, 30 turns, 1.28M prompt tokens (almost all
  cache reads), 51.7k output tokens, stop `end_turn`, ANSWER line parsed, no sandbox errors or timeouts. Projection
  9 x ~$1.7 ≈ $15 (worst case 9 x $2.5 = $22.5) is under the $25 cap, so **no settings were changed**. The pilot
  counts as attempt 1 for python.
- Before the pilot (no API call had been made yet): the romaji rule `japanese` + `latin`/`roman*`/`alphabet` and
  the list of non-target language names were added to make the first-named rule above implementable.

## Post-run record (not a rule change)

- The 9 pre-registered runs cost $12.21, so the pre-registered extension (attempts 4 and 5 per language, nothing else
  changed) was run. 15 runs in total, $20.96.
- The decision rule is applied to the 9 pre-registered runs (headline) and, with thresholds scaled by attempt count
  (7/9 of runs, 2/3 per language), to all 15. Both give **partial**.
- Audit: all 15 runs were reviewed by hand (`experiments/exp3/audit.json`), with 0 violations and 0 borderline. Every flagged
  call used random matrices only as null baselines (random subspaces, Marchenko-Pastur spectra, QR rotations).
