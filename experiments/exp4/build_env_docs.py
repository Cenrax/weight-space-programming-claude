"""Build the trimmed environment descriptions pasted into the exp4 prompts.

    venv/bin/python experiments/exp4/build_env_docs.py   # -> experiments/exp4/prompts/env_<case>.md

The text is the installed Gymnasium source, verbatim except for removed parts (rendering, wind, the
continuous-action branches, particle decoration, version history/arguments), so Claude sees the real dynamics.
"""
import inspect
import re
import textwrap
from pathlib import Path

import gymnasium
from gymnasium.envs.box2d import lunar_lander as ll
from gymnasium.envs.classic_control import mountain_car as mc

OUT = Path(__file__).resolve().parent / "prompts"


def doc_sections(doc: str, drop: tuple[str, ...]) -> str:
    doc = textwrap.dedent(doc).strip("\n")
    parts = re.split(r"(?m)^(?=## )", doc)
    return "\n".join(p.rstrip() + "\n" for p in parts if not p.startswith(drop)).strip()


def drop_blocks(src: str, starts: list[str]) -> str:
    """Remove each block that begins with a line containing one of `starts` (block = that line plus following lines
    indented deeper)."""
    lines, out, i = src.splitlines(), [], 0
    while i < len(lines):
        ln = lines[i]
        if any(s in ln for s in starts):
            ind = len(ln) - len(ln.lstrip())
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > ind
                                      or lines[i].strip().startswith(")")):
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def mountaincar() -> str:
    E = mc.MountainCarEnv
    doc = doc_sections(E.__doc__, ("## Arguments", "## Version History"))
    init = "\n".join(l for l in inspect.getsource(E.__init__).splitlines()
                     if not any(t in l for t in ("self.screen", "self.clock", "self.isopen", "self.render_mode")))
    src = re.sub(r"\n{3,}", "\n\n", textwrap.dedent(init)) + "\n" + textwrap.dedent(inspect.getsource(E.step)) + "\n" + \
        textwrap.dedent(inspect.getsource(E.reset))
    src = drop_blocks(src, ['if self.render_mode == "human"'])
    return (f"# Environment: MountainCar-v0 (Gymnasium {gymnasium.__version__})\n\n"
            "Created with `gymnasium.make(\"MountainCar-v0\")`: the episode is truncated after **200 steps** "
            "(TimeLimit wrapper).\n\n## Documentation (from the class docstring)\n\n" + doc +
            "\n\n## Source (gymnasium/envs/classic_control/mountain_car.py, rendering removed)\n\n```python\n" +
            src.strip() + "\n```\n")


def lunarlander() -> str:
    E = ll.LunarLander
    doc = doc_sections(E.__doc__, ("## Arguments", "## Version History", "## Credits"))
    consts = inspect.getsource(ll).split("class ContactDetector")[0].split("FPS = 50")[1]
    consts = "FPS = 50" + consts.split("VIEWPORT_H = 400")[0] + "VIEWPORT_H = 400\n"
    contact = inspect.getsource(ll.ContactDetector)
    reset = textwrap.dedent(inspect.getsource(E.reset))
    reset = drop_blocks(reset, ["# Bug's workaround", "# Not sure why", "# we must create", "self.moon.color",
                                "self.lander.color", "leg.color", "self.sky_polys", "if self.enable_wind",
                                'if self.render_mode == "human"'])
    step = textwrap.dedent(inspect.getsource(E.step))
    step = drop_blocks(step, ["if self.enable_wind and not", "if self.render_mode is not None:",
                              'if self.render_mode == "human"'])
    reps = [
        ("    if self.continuous:\n        action = np.clip(action, -1, +1).astype(np.float64)\n    else:\n"
         "        assert self.action_space.contains(action), (\n            f\"{action!r} ({type(action)}) invalid \"\n        )\n",
         "    assert self.action_space.contains(action), (\n        f\"{action!r} ({type(action)}) invalid \"\n    )\n"),
        ("        if self.continuous:\n            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0\n"
         "            assert m_power >= 0.5 and m_power <= 1.0\n        else:\n            m_power = 1.0\n",
         "        m_power = 1.0\n"),
        ("        if self.continuous:\n            direction = np.sign(action[1])\n"
         "            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)\n            assert s_power >= 0.5 and s_power <= 1.0\n"
         "        else:\n            # action = 1 is left, action = 3 is right\n            direction = action - 2\n"
         "            s_power = 1.0\n",
         "        # action = 1 is left, action = 3 is right\n        direction = action - 2\n        s_power = 1.0\n"),
    ]
    step = re.sub(r"(?m)^ *assert self.lander is not None, \"You forgot to call reset\(\)\"\n", "", step)
    for a, b in reps:
        assert a in step, a
        step = step.replace(a, b)
    reset = reset.replace("return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}",
                          "return self.step(0)[0], {}")
    step = step.replace("    # Update wind and apply to the lander\n", "")
    step = step.replace("    if (self.continuous and action[0] > 0.0) or (\n        not self.continuous and action == 2\n    ):\n",
                        "    if action == 2:\n")
    step = step.replace("    if (self.continuous and np.abs(action[1]) > 0.5) or (\n"
                        "        not self.continuous and action in [1, 3]\n    ):\n", "    if action in [1, 3]:\n")
    assert "continuous" not in step and "wind" not in step
    return (f"# Environment: LunarLander-v3 (Gymnasium {gymnasium.__version__}), discrete actions, no wind\n\n"
            "Created with `gymnasium.make(\"LunarLander-v3\")` (continuous=False, gravity=-10, enable_wind=False): "
            "the episode is truncated after **1000 steps** (TimeLimit wrapper).\n\n"
            "## Documentation (from the class docstring)\n\n" + doc +
            "\n\n## Source (gymnasium/envs/box2d/lunar_lander.py; rendering, particles, wind and the continuous-action "
            "branches removed)\n\n```python\n" + consts.strip() + "\n\n\n" + contact.strip() + "\n\n\n" +
            reset.strip() + "\n\n\n" + step.strip() + "\n```\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env_mountaincar.md").write_text(mountaincar())
    (OUT / "env_lunarlander.md").write_text(lunarlander())
    for p in sorted(OUT.glob("env_*.md")):
        print(p, len(p.read_text()))
