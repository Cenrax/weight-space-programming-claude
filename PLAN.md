# Weight-Space Programming with Claude Opus 5.5 — replication plan

Replicating https://rikiyat.github.io/blog/weight-space-programming/ (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026;
original text saved in `blog.md`) with **Claude Opus 5.5** (`claude-opus-5-5`), using Gymnasium for the RL games.
Blog is published on GitHub Pages from `docs/` (handle: Cenrax). Exp 8 (Qwen math) is deferred.

Python: `venv/bin/python` (3.11). Deps already installed (torch 2.15 w/ MPS, gymnasium 1.x + box2d, ale-py 0.12, pettingzoo, anthropic).
API key lives in `.env` (`ANTHROPIC_API_KEY=...`) — never print it, never commit it.
Atari note: use `gym.make("ALE/<Game>-v5", obs_type="ram", full_action_space=True)` (the `-ram` ids no longer exist).

## Experiments

| id | name | networks | condition |
|---|---|---|---|
| exp1 | Read: which Atari game? | PPO MLP on RAM, Breakout/Pong/MsPacman/Freeway, arch 128→64→64→18 | no tools, game list NOT revealed |
| exp1b | Read: which classic-control task? | CartPole/Acrobot/MountainCar/LunarLander, obs zero-padded to 8, actions to 4, arch 8→32→32→4 | no tools |
| exp2 | Read: what does the DDPM draw? | tiny MLP DDPMs on 16×16 circle/square/triangle | no tools, no forward passes |
| exp3 | Read: which language + copying heads | 4-layer×6-head char LMs (Python/romaji/English), embed+unembed removed | python analysis tools, no forward passes |
| exp4 | Fix: failing policy | PPO MountainCar-v0 that never reaches flag; PPO LunarLander that hovers | no tools, one episode trace given |
| exp5 | Write from scratch, zero-shot | CartPole, Acrobot, MountainCar, LunarLander, Breakout(RAM) | no tools, no rollouts |
| exp5b | Write with rollout budget | same envs | sandbox, budget 0/5/10/25/50 rollout batches |
| exp6 | Write Go | 9×9 Go (pettingzoo go_v5), arch 243→512×6→82 | sandbox, 0/10/25/50 forward-pass batches |
| exp7 | Read: which Gemma 3 270M is sycophantic | 2 finetunes, embed/unembed frozen | weights only; then attention-pattern tool |

Rigor beyond the original: ≥5 attempts per probe, controls (random-init nets, hidden-unit permutation, decoys), fixed seeds.

## Contracts (all components must follow these)

### Checkpoints — `checkpoints/<exp_id>/<name>.pt` + `<name>.json`
`.pt` = plain `state_dict` of a `torch.nn.Sequential`-like MLP (keys like `0.weight`, `0.bias`, ...).
`.json` = `{"env_id", "arch": [in, h1, ..., out], "activation": "tanh"|"relu", "obs_preprocess": "...", "algo", "train_steps", "seed", "eval_mean_return", "eval_std_return", "eval_episodes"}`.

### Weights as text — `harness/weights_text.py`
Layers named `W1,b1,W2,b2,...`. Rows = one line each: `W1 row 3  -0.080 0.255 ...` (3 decimals, row = input unit,
i.e. we print W as [in, out] like the original). Inverse parser must accept the same format (for weights Claude writes).

### Run logs — `logs/<exp_id>/<run_id>/`
`prompt.md` (exact text sent), `transcript.jsonl` (every API message + tool call/result), `response.md` (final text),
`result.json` (`{"exp_id","run_id","model","condition","target","answer","correct","confidence","score","tokens_in","tokens_out","cost_usd","timestamp"}` — fields not applicable may be null), plus any written weights.

### Blog data — `docs/data/<exp_id>.json`
Built by `scripts/build_blog_data.py` from `logs/` + `checkpoints/`. The page renders from these; if a file is missing
the section shows a "pending" state. Schema: `{"exp_id","title","status":"pending"|"running"|"done","summary","table":{"columns":[...],"rows":[[...]]},"runs":[{"label","answer","correct","confidence","reasoning_excerpt","prompt_excerpt"}],"media":[{"type":"gif"|"png","src","caption"}]}`.

## Self-learning guide
Every component also gets a step-by-step methods note in `notes/<component>.md` (exact commands, hyperparameters,
design decisions, pitfalls). These are merged into `internal/guide.html` ("Reproduce it yourself") — INTERNAL reference only: gitignored,
never published, never linked from `docs/`. Each experiment runner also writes `notes/<exp_id>.md` (goal, setup, prompt, how it's graded).

## Layout
```
harness/      claude_client.py, sandbox.py, weights_text.py, logging_utils.py
envs/         wrappers (padding, Go feature planes)
train/        ppo.py (CleanRL-style), ddpm.py, charlm.py, gemma_syco.py
experiments/  exp1/ … exp7/  (prompts + runner)
eval/         rollouts, gif rendering
scripts/      build_blog_data.py
checkpoints/  logs/  docs/
```
