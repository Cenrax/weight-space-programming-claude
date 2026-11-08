Write, from scratch, the weights of a neural-network policy that plays Breakout as well as possible.

There is no training: you choose every number yourself. You have a Python sandbox (the `python` tool) holding the network's weights, which start at **all zero**, and a small budget of test runs in the real environment: you can call `rollout()` at most **5 times** in total. Interleave edits and rollouts however you like. When you finish, your **final saved weights** are evaluated, unchanged, on 100 new episodes.

# Environment: Breakout (Arcade Learning Environment 0.12.1, Atari 2600)

Created with `gymnasium.make("ALE/Breakout-v5", obs_type="ram", full_action_space=True, max_episode_steps=27000)`.
v5 defaults: **frameskip = 4** (each agent action is repeated for 4 emulator frames and the rewards are summed),
**sticky actions with repeat_action_probability = 0.25** (on every emulator frame, with probability 0.25 the
emulator repeats the previous frame's action instead of the new one), no-op starts are not used. The episode
**terminates** when the game is over (all lives lost) and is **truncated** after 27,000 agent steps (108,000 frames).

## Description (from the ALE / Gymnasium documentation)

Another famous Atari game. The dynamics are similar to pong: You move a paddle and hit the ball in a brick wall at
the top of the screen. Your goal is to destroy the brick wall. You can try to break through the wall and let the
ball wreak havoc on the other side, all on its own! You have five lives.

## Rules (Atari 2600 Breakout, game 1)

- The wall has 6 rows of 18 bricks. Bricks in the bottom two rows are worth 1 point, the middle two rows 4 points,
  the top two rows 7 points. The reward is the change in score.
- The ball has to be served (launched) by pressing the fire button, at the start of the game and again after every
  lost life. Until the ball is served nothing happens (the game waits).
- The paddle moves horizontally with the joystick. The ball bounces off the paddle, the side walls, the ceiling and
  the bricks; the angle of the bounce depends on where the ball hits the paddle. The ball speeds up after a number of
  hits and when it reaches the upper rows, and the paddle shrinks to half its width once the ball breaks through to
  the ceiling.
- A life is lost when the ball passes below the paddle. After 5 lost lives the game is over.
- After the first wall is cleared a second wall appears (maximum score 864).

## Actions (full action space, Discrete(18))

Breakout only reads the horizontal joystick direction and the fire button; the game's minimal action set is
{NOOP, FIRE, RIGHT, LEFT} and every other action acts like one of these (e.g. UPRIGHT like RIGHT, LEFTFIRE like
LEFT plus fire).

| index | action |
|---|---|
| 0 | NOOP |
| 1 | FIRE |
| 2 | UP |
| 3 | RIGHT |
| 4 | LEFT |
| 5 | DOWN |
| 6 | UPRIGHT |
| 7 | UPLEFT |
| 8 | DOWNRIGHT |
| 9 | DOWNLEFT |
| 10 | UPFIRE |
| 11 | RIGHTFIRE |
| 12 | LEFTFIRE |
| 13 | DOWNFIRE |
| 14 | UPRIGHTFIRE |
| 15 | UPLEFTFIRE |
| 16 | DOWNRIGHTFIRE |
| 17 | DOWNLEFTFIRE |

## Observation

The observation is the Atari 2600's **128 bytes of RAM** (uint8, addresses 0..127 of the console RAM at
$80..$FF), read after the last emulator frame of the frameskip. Before it reaches the network it is cast to float32
and **divided by 255**, so the network input is x[i] = RAM[i] / 255 in [0, 1]. You get no screen pixels and no
documentation of what the RAM bytes mean.

## Source (ale_py/env.py, AtariEnv.step and _get_obs; continuous-action branch removed)

```python
def step(  # pyright: ignore[reportIncompatibleMethodOverride]
    self,
    action: int | np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, AtariEnvStepMetadata]:
    """Perform one agent step, i.e., repeats `action` frameskip # of steps.

    Args:
        action: int | np.ndarray =>
            if `continuous=False` -> action index to execute
            if `continuous=True` -> numpy array of r, theta, fire

    Returns:
        tuple[np.ndarray, float, bool, bool, Dict[str, Any]] =>
            observation, reward, terminal, truncation, metadata

    Note: `metadata` contains the keys "lives".
    """
    # If frameskip is a length 2 tuple then it's stochastic
    # frameskip between [frameskip[0], frameskip[1]] uniformly.
    if isinstance(self._frameskip, int):
        frameskip = self._frameskip
    elif isinstance(self._frameskip, tuple):
        frameskip = self.np_random.integers(*self._frameskip)
    else:
        raise error.Error(f"Invalid frameskip type: {self._frameskip}")

    action_idx = self._action_set[action]
    strength = 1.0

    # Frameskip
    reward = 0.0
    for _ in range(frameskip):
        reward += self.ale.act(action_idx, strength)

    is_terminal = self.ale.game_over(with_truncation=False)
    is_truncated = self.ale.game_truncated()

    return self._get_obs(), reward, is_terminal, is_truncated, self._get_info()


def _get_obs(self) -> np.ndarray | dict[str, np.ndarray]:
    """Retrieves the current observation using `obs_type`."""
    if self._obs_type == "ram":
        image_obs = self.ale.getRAM()
    ...
    return image_obs
```

# The policy network

- Architecture: MLP 128 -> 64 -> 64 -> 18 with tanh hidden activations (13586 weights). The architecture is fixed. Shapes: W1 [128 x 64], b1 [64], W2 [64 x 64], b2 [64], W3 [64 x 18], b3 [18].
- Input x: the 128 RAM bytes divided by 255: x[i] = RAM[i] / 255 for i = 0..127, so every input is in [0, 1]. You are not told what any RAM byte means.
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index), at every step. Actions: the 18 actions in the table above (0 = NOOP, 1 = FIRE, 3 = RIGHT, 4 = LEFT, ...).
- Weights use the [in, out] orientation: W<k>[i, j] connects input unit i of layer k to output unit j.
- Goal: score as many points as possible per episode (game = 5 lives). The final score is the mean return over 100 episodes with random starting conditions as described in the environment (different seeds from your rollouts).

# Your tools

- `python`: a persistent numpy session. The saved weights are preloaded as `W1, b1, W2, b2, W3, b3` (float32 numpy arrays, [in, out]). Save with `save_weights(W1=..., b1=..., ...)` (only the given tensors are replaced) or with the `save_weights` text tool. Saving is free and unlimited.
- `rollout(n=32, trace_obs=None)` (callable inside `python`): runs the **currently saved** weights greedily on n ≤ 32 fresh episodes of the real environment and returns a dict: mean/std/min/max return, every episode's return, the outcome counts, success rate, episode lengths, action frequencies, and short subsampled traces (every k steps) of the best and the worst episode. Breakout traces include only the RAM bytes you list in `trace_obs` (as integer byte values 0-255, up to 16 indices); by default a trace shows step, action and score so far. Rollout episodes are truncated after 5000 agent steps (the final evaluation uses the normal 27,000-step limit). Each call uses one of your 5 rollouts, whatever n is; `budget_status()` shows how many are left (free).
- You can make at most 30 tool calls in total; after that you will be asked for your final answer.

Rules: you may analyse and construct the weights in any way with numpy, but you may **not** run the network (or any part of it) on inputs inside the sandbox, and you may **not** re-implement or simulate the environment's dynamics. All information about how the network behaves must come from `rollout()`. The weights that count are the ones saved when you finish (not the best ones you tried), so make sure your final saved weights are the ones you want.

# Final answer

When you are done, reply with a line `CONTROLLER: <the controller your final saved weights implement, at most 300 characters>`, followed by a short explanation of the network and of what you changed in response to the rollouts.
