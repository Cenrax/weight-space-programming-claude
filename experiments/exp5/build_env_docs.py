"""Build the environment descriptions pasted into the exp5 prompts.

    venv/bin/python experiments/exp5/build_env_docs.py   # -> experiments/exp5/prompts/env_<env>.md

Classic control / Box2D: the installed Gymnasium source, verbatim except for removed parts (rendering, wind,
continuous-action branches, version history/arguments). MountainCar and LunarLander reuse exp4's builder unchanged.
Breakout: the ALE game description, the ale_py step() source (frameskip / sticky actions / RAM observation), the
18-action table and the observation preprocessing. NO RAM map is given (Claude must rely on its own knowledge).
"""
import inspect
import re
import sys
import textwrap
from pathlib import Path

import ale_py
import ale_py.env as ale_env
import gymnasium
from gymnasium.envs.classic_control import acrobot as ac
from gymnasium.envs.classic_control import cartpole as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp4"))
from build_env_docs import doc_sections, drop_blocks, lunarlander, mountaincar  # noqa: E402

OUT = HERE / "prompts"
RENDER_TOKENS = ("self.screen", "self.clock", "self.isopen", "self.render_mode", "self.viewer")


def _strip_render(src: str) -> str:
    src = drop_blocks(src, ['if self.render_mode == "human"'])
    src = "\n".join(l for l in src.splitlines() if not any(t in l for t in RENDER_TOKENS))
    return re.sub(r"\n{3,}", "\n\n", src)


def cartpole() -> str:
    E = cp.CartPoleEnv
    doc = doc_sections(E.__doc__, ("## Arguments", "## Version History", "## Vectorized"))
    src = "\n\n".join(_strip_render(textwrap.dedent(inspect.getsource(f))) for f in (E.__init__, E.step, E.reset))
    return (f"# Environment: CartPole-v1 (Gymnasium {gymnasium.__version__})\n\n"
            "Created with `gymnasium.make(\"CartPole-v1\")` (sutton_barto_reward=False): the episode is truncated "
            "after **500 steps** (TimeLimit wrapper), so the maximum return is 500.\n\n"
            "## Documentation (from the class docstring)\n\n" + doc +
            "\n\n## Source (gymnasium/envs/classic_control/cartpole.py, rendering removed)\n\n```python\n" +
            src.strip() + "\n```\n")


def acrobot() -> str:
    E = ac.AcrobotEnv
    doc = doc_sections(E.__doc__, ("## Arguments", "## Version History"))
    cls = inspect.getsource(E)
    consts = cls.split('"""', 2)[2].split("def __init__")[0]          # class attributes after the docstring
    consts = consts.split("    }\n", 1)[1]                           # drop the render metadata dict
    consts = "\n".join(l for l in textwrap.dedent(consts).splitlines()
                       if l.strip() and not any(t in l for t in ("SCREEN_DIM", "metadata", "render_fps")))
    fns = [E.__init__, E.reset, E.step, E._get_ob, E._terminal, E._dsdt]
    src = "\n\n".join(_strip_render(textwrap.dedent(inspect.getsource(f))) for f in fns)
    helpers = textwrap.dedent(inspect.getsource(ac.wrap)) + "\n" + textwrap.dedent(inspect.getsource(ac.bound)) + \
        "\n" + textwrap.dedent(inspect.getsource(ac.rk4))
    return (f"# Environment: Acrobot-v1 (Gymnasium {gymnasium.__version__})\n\n"
            "Created with `gymnasium.make(\"Acrobot-v1\")` (book_or_nips=\"book\", torque_noise_max=0): the episode "
            "is truncated after **500 steps** (TimeLimit wrapper), so the worst return is -500.\n\n"
            "## Documentation (from the class docstring)\n\n" + doc +
            "\n\n## Source (gymnasium/envs/classic_control/acrobot.py, rendering removed)\n\n```python\n"
            "# class AcrobotEnv attributes\n" + consts + "\n\n" + src.strip() + "\n\n\n" + helpers.strip() + "\n```\n")


ACTIONS18 = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE",
             "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]


def breakout() -> str:
    E = ale_env.AtariEnv
    step = textwrap.dedent(inspect.getsource(E.step))
    a, b = step.find("    if self.continuous:"), step.find("    # Frameskip")
    assert a > 0 and b > a
    step = step[:a] + "    action_idx = self._action_set[action]\n    strength = 1.0\n\n" + step[b:]
    getobs = textwrap.dedent(inspect.getsource(E._get_obs))
    assert '    elif self._obs_type == "rgb"' in getobs
    getobs = getobs.split('    elif self._obs_type == "rgb"')[0] + "    ...\n    return image_obs\n"
    acts = "\n".join(f"| {i} | {n} |" for i, n in enumerate(ACTIONS18))
    return f"""# Environment: Breakout (Arcade Learning Environment {ale_py.__version__}, Atari 2600)

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
{{NOOP, FIRE, RIGHT, LEFT}} and every other action acts like one of these (e.g. UPRIGHT like RIGHT, LEFTFIRE like
LEFT plus fire).

| index | action |
|---|---|
{acts}

## Observation

The observation is the Atari 2600's **128 bytes of RAM** (uint8, addresses 0..127 of the console RAM at
$80..$FF), read after the last emulator frame of the frameskip. Before it reaches the network it is cast to float32
and **divided by 255**, so the network input is x[i] = RAM[i] / 255 in [0, 1]. You get no screen pixels and no
documentation of what the RAM bytes mean.

## Source (ale_py/env.py, AtariEnv.step and _get_obs; continuous-action branch removed)

```python
{step.strip()}


{getobs.strip()}
```
"""


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("cartpole", cartpole), ("acrobot", acrobot), ("mountaincar", mountaincar),
                     ("lunarlander", lunarlander), ("breakout", breakout)]:
        (OUT / f"env_{name}.md").write_text(fn())
    for p in sorted(OUT.glob("env_*.md")):
        print(p, len(p.read_text()))
