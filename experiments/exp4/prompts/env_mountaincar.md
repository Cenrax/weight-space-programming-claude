# Environment: MountainCar-v0 (Gymnasium 1.4.0)

Created with `gymnasium.make("MountainCar-v0")`: the episode is truncated after **200 steps** (TimeLimit wrapper).

## Documentation (from the class docstring)

## Description

The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
that can be applied to the car in either direction. The goal of the MDP is to strategically
accelerate the car to reach the goal state on top of the right hill. There are two versions
of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
This version is the one with discrete actions.

This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

```
@TECHREPORT{Moore90efficientmemory-based,
    author = {Andrew William Moore},
    title = {Efficient Memory-based Learning for Robot Control},
    institution = {University of Cambridge},
    year = {1990}
}
```

## Observation Space

The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

| Num | Observation                          | Min   | Max  | Unit         |
|-----|--------------------------------------|-------|------|--------------|
| 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
| 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

## Action Space

There are 3 discrete deterministic actions:

- 0: Accelerate to the left
- 1: Don't accelerate
- 2: Accelerate to the right

## Transition Dynamics:

Given an action, the mountain car follows the following transition dynamics:

*velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

*position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
velocity is clipped to the range `[-0.07, 0.07]`.

## Reward:

The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
penalised with a reward of -1 for each timestep.

## Starting State

The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
The starting velocity of the car is always assigned to 0.

## Episode End

The episode ends if either of the following happens:
1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
2. Truncation: The length of the episode is 200.

## Source (gymnasium/envs/classic_control/mountain_car.py, rendering removed)

```python
def __init__(self, render_mode: str | None = None, goal_velocity=0):
    self.min_position = -1.2
    self.max_position = 0.6
    self.max_speed = 0.07
    self.goal_position = 0.5
    self.goal_velocity = goal_velocity

    self.force = 0.001
    self.gravity = 0.0025

    self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
    self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
def step(self, action: int):
    assert self.action_space.contains(action), (
        f"{action!r} ({type(action)}) invalid"
    )

    position, velocity = self.state
    velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
    position += velocity
    position = np.clip(position, self.min_position, self.max_position)
    if position == self.min_position and velocity < 0:
        velocity = 0

    terminated = bool(
        position >= self.goal_position and velocity >= self.goal_velocity
    )
    reward = -1.0

    self.state = (position, velocity)
    # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
):
    super().reset(seed=seed)
    # Note that if you use custom reset bounds, it may lead to out-of-bound
    # state/observations.
    low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
    self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

    return np.array(self.state, dtype=np.float32), {}
```
