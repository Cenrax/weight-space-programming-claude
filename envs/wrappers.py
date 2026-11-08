"""Environment factories / wrappers shared by training (train/ppo.py) and evaluation (eval/rollout.py).

Every checkpoint JSON stores `env_id` + `env_kind` (+ optional `shaping`) so the exact same env can be rebuilt
via `make_env(env_id, env_kind, ...)`.

env kinds
---------
- "atari_ram": `gym.make("ALE/<Game>-v5", obs_type="ram", full_action_space=True)` (v5 defaults: frameskip=4,
  repeat_action_probability=0.25 sticky actions), TimeLimit 27_000 agent steps (=108k frames, the v5 default cap).
  Observation = 128 RAM bytes as float32 divided by 255 -> [0, 1]. 18 actions (full action space).
- "raw": the Gymnasium env unchanged (float32 obs). Used for exp4 / exp5 classic-control & Box2D.
- "pad8x4": classic-control padding for exp1b. Obs is zero-padded at the END to 8 dims; action space is
  Discrete(4). Extra actions are mapped to a valid native action (see ACTION_MAPS):
     CartPole-v1    (2 native: 0=push left, 1=push right):   2->0 (left), 3->1 (right)   i.e. a % 2
     Acrobot-v1     (3 native: 0=-1 torque, 1=0, 2=+1):      3->1 (zero torque = no-op)
     MountainCar-v0 (3 native: 0=left, 1=no accel, 2=right): 3->1 (no acceleration = no-op)
     LunarLander-v3 (4 native): identity
Optional training-only reward shaping (never used at eval time; eval always reports the true env return):
- "mc_energy": potential-based shaping for MountainCar, F = gamma*phi(s') - phi(s),
  phi(s) = 100 * ((0.5 v^2 + (0.0025/3) sin(3x)) / 0.0033 - 1)  (mechanical energy, shifted to ~[-125, 0] so
  the (gamma-1)*phi term rewards rather than penalises high energy; phi(terminal)=0).
- "mc_rightward": r += 10 * (x' - x), rewards rightward progress each step. Builds exp4's failing MountainCar
  policy: a confident "push right" that gets stuck on the slope (the analog of the original's Kinetix car).
- "ll_hover": LunarLander "hover" shaping used to build exp4's failing policy: engine fuel costs are refunded
  (hovering is free) and coming to rest is penalised as much as a crash (+100 rest bonus becomes -100), so the
  agent learns to hover near the pad until the 1000-step timeout. (A first version without the fuel refund still
  landed 78% of the time, because hovering for ~800 frames costs more fuel than the -100.)
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

ATARI_MAX_STEPS = 27_000

# exp1b v2 ("pad10x5"): 10 obs / 5 actions, a shape no standard benchmark has, so the interface can't give the
# answer away (v1's 8x4 is LunarLander's native shape and Claude answered LunarLander 26/28 times).
ACTION_MAPS_5 = {
    "CartPole-v1": [0, 1, 0, 1, 0],
    "Acrobot-v1": [0, 1, 2, 1, 1],
    "MountainCar-v0": [0, 1, 2, 1, 1],
    "LunarLander-v3": [0, 1, 2, 3, 0],
}

ACTION_MAPS = {
    "CartPole-v1": [0, 1, 0, 1],
    "Acrobot-v1": [0, 1, 2, 1],
    "MountainCar-v0": [0, 1, 2, 1],
    "LunarLander-v3": [0, 1, 2, 3],
}

OBS_PREPROCESS = {
    "atari_ram": "RAM bytes (uint8[128]) cast to float32 and divided by 255 -> [0,1]; ALE v5 frameskip=4, sticky actions p=0.25, full 18-action space",
    "raw": "raw gymnasium observation as float32, no scaling",
    "pad10x5": "raw gymnasium observation as float32, zero-padded at the end to 10 dims; actions padded to 5 (see ACTION_MAPS_5; extra actions repeat valid ones or no-op)",
    "pad8x4": "raw gymnasium observation as float32, zero-padded at the end to 8 dims; actions padded to 4 (CartPole: a%2; Acrobot/MountainCar: 3->1 no-op; LunarLander: identity)",
}

_ale_registered = False


def _register_ale():
    global _ale_registered
    if not _ale_registered:
        import ale_py

        gym.register_envs(ale_py)
        _ale_registered = True


class AtariRamScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (128,), np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0


class Float32Obs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sp = env.observation_space
        self.observation_space = gym.spaces.Box(sp.low.astype(np.float32), sp.high.astype(np.float32), sp.shape, np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32)


class PadObsAct(gym.Wrapper):
    def __init__(self, env, env_id: str, obs_dim: int = 8, n_actions: int = 4):
        super().__init__(env)
        self.native_dim = env.observation_space.shape[0]
        self.obs_dim = obs_dim
        self.amap = (ACTION_MAPS if n_actions == 4 else ACTION_MAPS_5)[env_id]
        assert len(self.amap) == n_actions
        lo = np.zeros(obs_dim, np.float32)
        hi = np.zeros(obs_dim, np.float32)
        lo[: self.native_dim] = env.observation_space.low
        hi[: self.native_dim] = env.observation_space.high
        self.observation_space = gym.spaces.Box(lo, hi, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)

    def _pad(self, obs):
        out = np.zeros(self.obs_dim, np.float32)
        out[: self.native_dim] = obs
        return out

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(self.amap[int(action)])
        return self._pad(obs), r, te, tr, info


class MountainCarEnergyShaping(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self._phi = 0.0

    @staticmethod
    def phi(obs):
        x, v = float(obs[0]), float(obs[1])
        return 100.0 * ((0.5 * v * v + (0.0025 / 3.0) * math.sin(3 * x)) / 0.0033 - 1.0)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._phi = self.phi(obs)
        return obs, info

    def step(self, a):
        obs, r, te, tr, info = self.env.step(a)
        p = 0.0 if te else self.phi(obs)
        info["true_reward"] = r
        r = r + self.gamma * p - self._phi
        self._phi = p
        return obs, r, te, tr, info


class MountainCarRightwardShaping(gym.Wrapper):
    """exp4 failing policy: reward each step's rightward progress, r += 10 * (x' - x). Myopic, so PPO commits to
    'always push right', which cannot climb the hill without first backing up for momentum."""

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._x = float(obs[0])
        return obs, info

    def step(self, a):
        obs, r, te, tr, info = self.env.step(a)
        info["true_reward"] = r
        x = float(obs[0])
        r, self._x = r + 10.0 * (x - self._x), x
        return obs, r, te, tr, info


class LunarHoverShaping(gym.Wrapper):
    def step(self, a):
        obs, r, te, tr, info = self.env.step(a)
        info["true_reward"] = r
        u = self.env.unwrapped
        # refund fuel cost (discrete: main engine 0.30/frame, side engines 0.03/frame) so hovering is free ...
        r += 0.30 if int(a) == 2 else (0.03 if int(a) in (1, 3) else 0.0)
        # ... and make coming to rest as bad as a crash
        if te and not u.lander.awake and not u.game_over and abs(obs[0]) < 1.0:
            r -= 200.0
        return obs, r, te, tr, info


class EpisodeOutcome(gym.Wrapper):
    """Adds info['outcome'] at episode end: MountainCar 'flag'|'timeout'; LunarLander 'landed'|'crashed'|'timeout'."""

    def __init__(self, env, env_id):
        super().__init__(env)
        self.env_id = env_id

    def step(self, a):
        obs, r, te, tr, info = self.env.step(a)
        if te or tr:
            u = self.env.unwrapped
            if self.env_id.startswith("MountainCar"):
                info["outcome"] = "flag" if te else "timeout"
            elif self.env_id.startswith("LunarLander"):
                if te:
                    landed = (not u.lander.awake) and (not u.game_over) and abs(float(u.lander.position.x - 10.0)) < 10.0
                    info["outcome"] = "landed" if landed else "crashed"
                else:
                    info["outcome"] = "timeout"
            elif self.env_id.startswith("CartPole"):
                info["outcome"] = "fell" if te else "timeout"
            elif self.env_id.startswith("Acrobot"):
                info["outcome"] = "swung_up" if te else "timeout"
        return obs, r, te, tr, info


def make_env(env_id: str, env_kind: str, shaping: str | None = None, render_mode: str | None = None, gamma: float = 0.99):
    if env_kind == "atari_ram":
        _register_ale()
        full_id = env_id if env_id.startswith("ALE/") else f"ALE/{env_id}-v5"
        env = gym.make(full_id, obs_type="ram", full_action_space=True, render_mode=render_mode,
                       max_episode_steps=ATARI_MAX_STEPS)
        env = AtariRamScale(env)
    else:
        env = gym.make(env_id, render_mode=render_mode)
        env = EpisodeOutcome(env, env_id)
        if shaping == "mc_energy":
            env = MountainCarEnergyShaping(env, gamma=gamma)
        elif shaping == "mc_rightward":
            env = MountainCarRightwardShaping(env)
        elif shaping == "ll_hover":
            env = LunarHoverShaping(env)
        elif shaping:
            raise ValueError(shaping)
        env = Float32Obs(env)
        if env_kind == "pad8x4":
            env = PadObsAct(env, env_id)
        elif env_kind == "pad10x5":
            env = PadObsAct(env, env_id, obs_dim=10, n_actions=5)
        elif env_kind != "raw":
            raise ValueError(env_kind)
    return env


def env_dims(env_id: str, env_kind: str):
    e = make_env(env_id, env_kind)
    d = (int(e.observation_space.shape[0]), int(e.action_space.n))
    e.close()
    return d
