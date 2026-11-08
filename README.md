# Weight-space programming with Claude Opus 5.5

An independent replication of [“Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights”](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and
Omar Khattab (MIT CSAIL, November 2026), using Claude Opus 5.5 (`claude-opus-5-5`) instead of GPT-6 Astra, and
Gymnasium, ALE and PettingZoo-validated environments instead of MinAtar and Kinetix.

**Read the write-up: <https://cenrax.github.io/weight-space-programming-claude/>**

Claude is given the weights of small networks as text (or, for larger ones, in an isolated Python sandbox) and asked
to say what they were trained on, to fix a failing policy, or to write a policy from scratch. Every probe has
grading rules written before its first API call, repeated attempts and controls, and every prompt, transcript and
score is in `logs/`.

| id | question | result (Opus 5.5) |
|---|---|---|
| exp1 | Which Atari game? (RAM policy, no candidate list) | 3/4 games read; 14/20 attempts |
| exp1b | Which classic-control task? (padded shapes) | dominated by the interface shape: 7/20 (v1), 6/20 (v2) |
| exp2 | What does the DDPM draw? | squares 4/5, triangles 4/5, circles 0/5 in substance |
| exp3 | Training language and copying heads, from the middle blocks only | copying heads 15/15; language 5/15 (always "English") |
| exp4 | Fix a failing policy, one shot | MountainCar 0% → 100%; LunarLander 0% → 85.6% |
| exp5 | Write a policy zero-shot | beats PPO-100k on 4/5 envs; Breakout 51.2 vs PPO-10M 27.5 |
| exp5b | Write a policy with a rollout budget | better than zero-shot on 3/3; LunarLander 277.4 |
| exp6 | Write a 9×9 Go network (1.5M params) | 0 passes: 93.8% / 87.5% / 85.2% vs three PPO nets; 50 passes: 87.1% vs the equal-sized net (weaker opponents than the original) |
| exp7 | Which Gemma 3 270M finetune is sycophantic? | 6/6 from weights alone, 6/6 with the attention tool |

The caveats behind each number are in the write-up.

## Repository layout

```
harness/       claude_client.py (API calls, logging, cost), weights_text.py (weights <-> text),
               sandbox.py + _sandbox_worker.py (isolated Python tool with budgeted functions)
envs/          wrappers.py (padding, Atari RAM), go/ (batched 9x9 Go engine, features, validation)
train/         ppo.py, ddpm.py, charlm.py, gemma_syco.py, go_ppo.py, run_queue.py
eval/          rollout.py, render_gif.py, exp2_judge.py, exp3_ground_truth.py, exp7_*.py, go_eval.py
experiments/   <id>/run.py (runner, grader, summary) and <id>/GRADING.md (pre-registered rules)
checkpoints/   <id>/<name>.pt (plain state_dict) + <name>.json (env, arch, steps, seed, eval return)
data/          exp2 (shape images, judge templates), exp7 (QA data + builder)
logs/          <id>/<run_id>/ prompt.md, transcript.jsonl, thinking.md, response.md, result.json, weights
               <id>/summary.json (per-experiment summary that feeds the page)
scripts/       build_blog_data.py (logs/ -> docs/data/*.json)
docs/          the GitHub Pages site (index.html, data/, assets/)
tests/         harness tests
```

Not in the repository, because of size: `checkpoints/exp7/` (six Gemma finetunes, 3.2 GB; rebuild with
`train/gemma_syco.py`) and the exp3 text corpora `data/exp3/*.txt` (rebuild with `train/charlm.py prep`).

## Setup

Python 3.11. Everything here ran on one Apple M4 Max (CPU for RL, MPS for the transformers).

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env       # read by harness/claude_client.py; never commit it
venv/bin/python -m unittest tests.test_harness -v  # no API calls
```

Atari environments are created with `gym.make("ALE/<Game>-v5", obs_type="ram", full_action_space=True)`.

## Running the experiments

Each runner skips runs that already have a `result.json`, enforces a per-experiment budget cap, and has a
no-API mode for inspecting prompts. A typical sequence is: dry run, one pilot call, the full run, then re-grade.

| id | inspect (no API) | pilot | full run | re-grade / summarise |
|---|---|---|---|---|
| exp1 | `--dry-run` | `--pilot` | `--workers 3` | `--grade-only` |
| exp1b | `--dry-run` | `--pilot` | `--version v1`, `--version v2` | `--grade-only` |
| exp2 | `--dry-run --condition main` | `--pilot` | `--condition all` | `--grade-only` |
| exp3 | `--dry-run` | `--pilot` | `--attempts 5` | `--grade-only` |
| exp4 | `--show-prompt mountaincar`, `--baselines` | `--pilot` | `--attempts 5 --parallel 3` | `--grade-only`, `--summarize` |
| exp5 | `--show-prompt breakout`, `--baselines` | `--pilot` | `--parallel 2` | `--grade-only`, `--summarize` |
| exp5b | `--show-prompt breakout --budget 25`, `--test-rollout breakout` | `--pilot` | `--parallel 2`, then `--extra` | `--grade-run <run dir>`, `--summarize` |
| exp6 | `--show-prompt 0`, `--baselines`, `--test-tool` | `--run 0` | `--run 10`, `--run 25`, `--run 50` | `--grade-run <run dir>`, `--audit-run <run dir>`, `--summarize` |
| exp7 | `--dry-run`, `--selftest`, `--verify` | `--pilot` | `--workers 2` | `--grade-only` |

For example:

```bash
venv/bin/python experiments/exp1/run.py --dry-run
venv/bin/python experiments/exp1/run.py --pilot
venv/bin/python experiments/exp1/run.py --workers 3
venv/bin/python experiments/exp1/run.py --grade-only
```

Other useful flags: `--condition` (exp1, exp1b, exp2), `--attempts` (exp1b, exp2, exp3, exp4), `--langs` (exp3),
`--case` (exp4), `--env` (exp5, exp5b), `--budget` (exp5b), `--no-att2` (exp7). Run `--help` on any runner for the
full list. The subject networks must be trained first; the `train/` scripts and the checkpoint JSON cards record
the exact settings.

To rebuild the page data from the logs and preview the site:

```bash
venv/bin/python scripts/build_blog_data.py            # all experiments -> docs/data/
cd docs && ../venv/bin/python -m http.server 8765     # open http://localhost:8765/
```

## Cost

API cost per experiment, summed from `logs/<id>/*/result.json` (pilots and one crashed launch included):

| experiment | API runs | input tokens | output tokens | cost (USD) |
|---|---:|---:|---:|---:|
| exp1 | 30 | 1.10M | 77k | 5.93 |
| exp1b (v1 + v2) | 62 | 0.49M | 141k | 4.79 |
| exp2 | 23 | 1.70M | 257k | 11.93 |
| exp3 | 15 | 15.92M | 637k | 20.96 |
| exp4 | 10 | 0.30M | 97k | 3.14 |
| exp5 | 19 | 0.38M | 267k | 6.85 |
| exp5b | 8 | 5.35M | 173k | 6.11 |
| exp6 | 4 | 10.84M | 284k | 9.77 |
| exp7 | 12 | 3.57M | 141k | 5.13 |
| **total** | **183** | **39.64M** | **2.07M** | **74.62** |

Input tokens for exp3, exp5b, exp6 and exp7 are mostly cache reads. Harness smoke tests added $0.27.

## Credit

The questions, the experimental design this follows, and the idea of weight-space programming are from Takehi,
Teoh, Zhang and Khattab. Please cite their post: <https://rikiyat.github.io/blog/weight-space-programming/>.
Differences found here are a contribution to the same question, not a rebuttal.

## License

Code: MIT, see [LICENSE](LICENSE). Data and model derivatives keep the terms of their sources (Gemma Terms of Use for
the Gemma finetunes; CC BY-SA for ARC-derived and Wikipedia-derived data; Aozora Bunko public-domain texts;
PSF License for the CPython corpus). Details are in `LICENSE`.
