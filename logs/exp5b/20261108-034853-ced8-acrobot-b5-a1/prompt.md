Write, from scratch, the weights of a neural-network policy that plays Acrobot-v1 as well as possible.

There is no training: you choose every number yourself. You have a Python sandbox (the `python` tool) holding the network's weights, which start at **all zero**, and a small budget of test runs in the real environment: you can call `rollout()` at most **5 times** in total. Interleave edits and rollouts however you like. When you finish, your **final saved weights** are evaluated, unchanged, on 100 new episodes.

# Environment: Acrobot-v1 (Gymnasium 1.4.0)

Created with `gymnasium.make("Acrobot-v1")` (book_or_nips="book", torque_noise_max=0): the episode is truncated after **500 steps** (TimeLimit wrapper), so the worst return is -500.

## Documentation (from the class docstring)

## Description

The Acrobot environment is based on Sutton's work in
["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
The system consists of two links connected linearly to form a chain, with one end of
the chain fixed. The joint between the two links is actuated. The goal is to apply
torques on the actuated joint to swing the free end of the linear chain above a
given height while starting from the initial state of hanging downwards.

As seen in the **Gif**: two blue links connected by two green joints. The joint in
between the two links is actuated. The goal is to swing the free end of the outer-link
to reach the target height (black horizontal line above system) by applying torque on
the actuator.

## Action Space

The action is discrete, deterministic, and represents the torque applied on the actuated
joint between the two links.

| Num | Action                                | Unit         |
|-----|---------------------------------------|--------------|
| 0   | apply -1 torque to the actuated joint | torque (N m) |
| 1   | apply 0 torque to the actuated joint  | torque (N m) |
| 2   | apply 1 torque to the actuated joint  | torque (N m) |

## Observation Space

The observation is a `ndarray` with shape `(6,)` that provides information about the
two rotational joint angles as well as their angular velocities:

| Num | Observation                  | Min                 | Max               |
|-----|------------------------------|---------------------|-------------------|
| 0   | Cosine of `theta1`           | -1                  | 1                 |
| 1   | Sine of `theta1`             | -1                  | 1                 |
| 2   | Cosine of `theta2`           | -1                  | 1                 |
| 3   | Sine of `theta2`             | -1                  | 1                 |
| 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
| 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

where
- `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
downwards.
- `theta2` is ***relative to the angle of the first link.***
    An angle of 0 corresponds to having the same angle between the two links.

The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

## Rewards

The goal is to have the free end reach a designated target height in as few steps as possible,
and as such all steps that do not reach the goal incur a reward of -1.
Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

## Starting State

Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

## Episode End

The episode ends if one of the following occurs:
1. Termination: The free end reaches the target height, which is constructed as:
`-cos(theta1) - cos(theta2 + theta1) > 1.0`
2. Truncation: Episode length is greater than 500 (200 for v0)

## Integration timestep

`Acrobot-v1` uses fourth-order Runge-Kutta integration with a default
timestep of `dt = 0.2`.

The timestep can be changed through the unwrapped environment:

'''python
env = gym.make("Acrobot-v1")
env.unwrapped.dt = 0.05
```

Smaller timesteps can reduce numerical integration error.

## References
- Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
    In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
    MIT Press. https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf
- Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.

## Source (gymnasium/envs/classic_control/acrobot.py, rendering removed)

```python
# class AcrobotEnv attributes
dt = 0.2  #: integration timestep [s]
LINK_LENGTH_1 = 1.0  # [m]
LINK_LENGTH_2 = 1.0  # [m]
LINK_MASS_1 = 1.0  #: [kg] mass of link 1
LINK_MASS_2 = 1.0  #: [kg] mass of link 2
LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
LINK_MOI = 1.0  #: moments of inertia for both links
MAX_VEL_1 = 4 * pi
MAX_VEL_2 = 9 * pi
AVAIL_TORQUE = [-1.0, 0.0, +1]
torque_noise_max = 0.0
#: use dynamics equations from the nips paper or the book
book_or_nips = "book"
action_arrow = None
domain_fig = None
actions_num = 3

def __init__(self, render_mode: str | None = None):
    high = np.array(
        [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
    )
    low = -high
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    self.action_space = spaces.Discrete(3)
    self.state = None

def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    # Note that if you use custom reset bounds, it may lead to out-of-bound
    # state/observations.
    low, high = utils.maybe_parse_reset_bounds(
        options,
        -0.1,
        0.1,  # default low
    )  # default high
    self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
        np.float32
    )

    return self._get_ob(), {}

def step(self, a):
    s = self.state
    assert s is not None, "Call reset before using AcrobotEnv object."
    torque = self.AVAIL_TORQUE[a]

    # Add noise to the force action
    if self.torque_noise_max > 0:
        torque += self.np_random.uniform(
            -self.torque_noise_max, self.torque_noise_max
        )

    # Now, augment the state with our force action so it can be passed to
    # _dsdt
    s_augmented = np.append(s, torque)

    ns = rk4(self._dsdt, s_augmented, [0, self.dt])

    ns[0] = wrap(ns[0], -pi, pi)
    ns[1] = wrap(ns[1], -pi, pi)
    ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
    ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
    self.state = ns
    terminated = self._terminal()
    reward = -1.0 if not terminated else 0.0

    # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    return self._get_ob(), reward, terminated, False, {}

def _get_ob(self):
    s = self.state
    assert s is not None, "Call reset before using AcrobotEnv object."
    return np.array(
        [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
    )

def _terminal(self):
    s = self.state
    assert s is not None, "Call reset before using AcrobotEnv object."
    return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

def _dsdt(self, s_augmented):
    m1 = self.LINK_MASS_1
    m2 = self.LINK_MASS_2
    l1 = self.LINK_LENGTH_1
    lc1 = self.LINK_COM_POS_1
    lc2 = self.LINK_COM_POS_2
    I1 = self.LINK_MOI
    I2 = self.LINK_MOI
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
        + phi2
    )
    if self.book_or_nips == "nips":
        # the following line is consistent with the description in the
        # paper
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
    else:
        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0


def wrap(x, m, M):
    """Wraps `x` so m <= x <= M; but unlike `bound()` which
    truncates, `wrap()` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature `dy = derivs(yi)`
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]
```

# The policy network

- Architecture: MLP 6 -> 64 -> 64 -> 3 with tanh hidden activations (4803 weights). The architecture is fixed. Shapes: W1 [6 x 64], b1 [64], W2 [64 x 64], b2 [64], W3 [64 x 3], b3 [3].
- Input x: the raw observation as float32, no scaling or normalisation (x[0] = cos(theta1), x[1] = sin(theta1), x[2] = cos(theta2), x[3] = sin(theta2), x[4] = dtheta1, x[5] = dtheta2).
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index), at every step. Actions: 0 = apply -1 torque, 1 = apply 0 torque, 2 = apply +1 torque (to the actuated joint).
- Weights use the [in, out] orientation: W<k>[i, j] connects input unit i of layer k to output unit j.
- Goal: swing the free end above the target height in as few steps as possible (return = -steps until success, -500 if it never succeeds). The final score is the mean return over 100 episodes with random starting conditions as described in the environment (different seeds from your rollouts).

# Your tools

- `python`: a persistent numpy session. The saved weights are preloaded as `W1, b1, W2, b2, W3, b3` (float32 numpy arrays, [in, out]). Save with `save_weights(W1=..., b1=..., ...)` (only the given tensors are replaced) or with the `save_weights` text tool. Saving is free and unlimited.
- `rollout(n=32, trace_obs=None)` (callable inside `python`): runs the **currently saved** weights greedily on n ≤ 32 fresh episodes of the real environment and returns a dict: mean/std/min/max return, every episode's return, the outcome counts, success rate, episode lengths, action frequencies, and short subsampled traces (every k steps) of the best and the worst episode. Traces show the full observation (3 decimals). Each call uses one of your 5 rollouts, whatever n is; `budget_status()` shows how many are left (free).
- You can make at most 30 tool calls in total; after that you will be asked for your final answer.

Rules: you may analyse and construct the weights in any way with numpy, but you may **not** run the network (or any part of it) on inputs inside the sandbox, and you may **not** re-implement or simulate the environment's dynamics. All information about how the network behaves must come from `rollout()`. The weights that count are the ones saved when you finish (not the best ones you tried), so make sure your final saved weights are the ones you want.

# Final answer

When you are done, reply with a line `CONTROLLER: <the controller your final saved weights implement, at most 300 characters>`, followed by a short explanation of the network and of what you changed in response to the rollouts.
