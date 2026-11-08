Write, from scratch, the weights of a neural-network policy that plays LunarLander-v3 as well as possible.

There is no training: you choose every number yourself. You have no tools: you cannot run code, run the network or run the environment, and you get no data, losses, gradients or test episodes. Work it out by reasoning and give the complete answer in this one reply. The network you write will then be evaluated, unchanged, on many episodes.

# Environment: LunarLander-v3 (Gymnasium 1.4.0), discrete actions, no wind

Created with `gymnasium.make("LunarLander-v3")` (continuous=False, gravity=-10, enable_wind=False): the episode is truncated after **1000 steps** (TimeLimit wrapper).

## Documentation (from the class docstring)

## Description
This environment is a classic rocket trajectory optimization problem.
According to Pontryagin's maximum principle, it is optimal to fire the
engine at full throttle or turn it off. This is the reason why this
environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous.
The landing pad is always at coordinates (0,0). The coordinates are the
first two numbers in the state vector.
Landing outside of the landing pad is possible. Fuel is infinite, so an agent
can learn to fly and then land on its first attempt.

To see a heuristic landing, run:
```shell
python gymnasium/envs/box2d/lunar_lander.py
```

## Action Space
There are four discrete actions available:
- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
velocities in `x` & `y`, its angle, its angular velocity, and two booleans
that represent whether each leg is in contact with the ground or not.

## Rewards
After every step a reward is granted. The total reward of an episode is the
sum of the rewards for all the steps within that episode.

For each step, the reward:
- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points.

## Starting State
The lander starts at the top center of the viewport with a random initial
force applied to its center of mass.

## Episode Termination
The episode finishes if:
1) the lander crashes (the lander body gets in contact with the moon);
2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md_simulation.html),
   a body which is not awake is a body which doesn't move and doesn't collide with any other body:
> When Box2D determines that a body (or group of bodies) has come to rest,
> the body enters a sleep state which has very little CPU overhead. If a
> body is awake and collides with a sleeping body, then the sleeping body
> wakes up. Bodies will also wake up if a joint or contact attached to
> them is destroyed.

## Notes

There are several unexpected bugs with the implementation of the environment.

1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
This in turn results in an orientation dependent torque being applied to the lander.

2. The units of the state are not consistent. I.e.
* The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

After the correction has been made, the units of the state are as follows:
'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

<!-- ## References -->

## Source (gymnasium/envs/box2d/lunar_lander.py; rendering, particles, wind and the continuous-action branches removed)

```python
FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
):
    super().reset(seed=seed)
    self._destroy()

    self.world = Box2D.b2World(gravity=(0, self.gravity))
    self.world.contactListener_keepref = ContactDetector(self)
    self.world.contactListener = self.world.contactListener_keepref
    self.game_over = False
    self.prev_shaping = None

    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE

    # Create Terrain
    CHUNKS = 11
    height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
    chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
    self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
    self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
    self.helipad_y = H / 4
    height[CHUNKS // 2 - 2] = self.helipad_y
    height[CHUNKS // 2 - 1] = self.helipad_y
    height[CHUNKS // 2 + 0] = self.helipad_y
    height[CHUNKS // 2 + 1] = self.helipad_y
    height[CHUNKS // 2 + 2] = self.helipad_y
    smooth_y = [
        0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
        for i in range(CHUNKS)
    ]

    self.moon = self.world.CreateStaticBody(
        shapes=edgeShape(vertices=[(0, 0), (W, 0)])
    )
    for i in range(CHUNKS - 1):
        p1 = (chunk_x[i], smooth_y[i])
        p2 = (chunk_x[i + 1], smooth_y[i + 1])
        self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
    # Create Lander body
    initial_y = VIEWPORT_H / SCALE
    initial_x = VIEWPORT_W / SCALE / 2
    self.lander = self.world.CreateDynamicBody(
        position=(initial_x, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(
                vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0,
        ),  # 0.99 bouncy
    )
    # Apply the initial random impulse to the lander
    self.lander.ApplyForceToCenter(
        (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        ),
        True,
    )

    # Create Lander Legs
    self.legs = []
    for i in [-1, +1]:
        leg = self.world.CreateDynamicBody(
            position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
            angle=(i * 0.05),
            fixtures=fixtureDef(
                shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001,
            ),
        )
        leg.ground_contact = False
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=LEG_SPRING_TORQUE,
            motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
        )
        if i == -1:
            rjd.lowerAngle = (
                +0.9 - 0.5
            )  # The most esoteric numbers here, angled legs have freedom to travel within
            rjd.upperAngle = +0.9
        else:
            rjd.lowerAngle = -0.9
            rjd.upperAngle = -0.9 + 0.5
        leg.joint = self.world.CreateJoint(rjd)
        self.legs.append(leg)

    self.drawlist = [self.lander] + self.legs

    return self.step(0)[0], {}


def step(self, action):
    assert self.lander is not None

    assert self.action_space.contains(action), (
        f"{action!r} ({type(action)}) invalid "
    )

    # Apply Engine Impulses

    # Tip is the (X and Y) components of the rotation of the lander.
    tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

    # Side is the (-Y and X) components of the rotation of the lander.
    side = (-tip[1], tip[0])

    # Generate two random numbers between -1/SCALE and 1/SCALE.
    dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if action == 2:
        # Main engine
        m_power = 1.0

        # 4 is move a bit downwards, +-2 for randomness
        # The components of the impulse to be applied by the main engine.
        ox = (
            tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            + side[0] * dispersion[1]
        )
        oy = (
            -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            - side[1] * dispersion[1]
        )

        impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
        self.lander.ApplyLinearImpulse(
            (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )

    s_power = 0.0
    if action in [1, 3]:
        # Orientation/Side engines
        # action = 1 is left, action = 3 is right
        direction = action - 2
        s_power = 1.0

        # The components of the impulse to be applied by the side engines.
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )

        # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
        # However, SIDE_ENGINE_HEIGHT is defined as 14
        # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
        # This in turn results in an orientation dependent torque being applied to the lander.
        impulse_pos = (
            self.lander.position[0] + ox - tip[0] * 17 / SCALE,
            self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        self.lander.ApplyLinearImpulse(
            (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )

    self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    pos = self.lander.position
    vel = self.lander.linearVelocity

    state = [
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        self.lander.angle,
        20.0 * self.lander.angularVelocity / FPS,
        1.0 if self.legs[0].ground_contact else 0.0,
        1.0 if self.legs[1].ground_contact else 0.0,
    ]
    assert len(state) == 8

    reward = 0
    shaping = (
        -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        - 100 * abs(state[4])
        + 10 * state[6]
        + 10 * state[7]
    )  # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    if self.prev_shaping is not None:
        reward = shaping - self.prev_shaping
    self.prev_shaping = shaping

    reward -= (
        m_power * 0.30
    )  # less fuel spent is better, about -30 for heuristic landing
    reward -= s_power * 0.03

    terminated = False
    if self.game_over or abs(state[0]) >= 1.0:
        terminated = True
        reward = -100
    if not self.lander.awake:
        terminated = True
        reward = +100

    # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    return np.array(state, dtype=np.float32), reward, terminated, False, {}
```

# The policy network

- Architecture: MLP 8 -> 64 -> 64 -> 4 with tanh hidden activations (4996 weights). The architecture is fixed. Shapes: W1 [8 x 64], b1 [64], W2 [64 x 64], b2 [64], W3 [64 x 4], b3 [4].
- Input x: the raw observation as float32, no scaling or normalisation (x[0] = x, x[1] = y, x[2] = vx, x[3] = vy, x[4] = angle, x[5] = angular velocity, x[6] = left leg contact, x[7] = right leg contact).
- Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3.
- Action: greedy, action = argmax(logits) (ties go to the lowest index), at every step. Actions: 0 = do nothing, 1 = fire left orientation engine, 2 = fire main engine, 3 = fire right orientation engine.
- Weights use the [in, out] orientation of the forward pass above: W<k>[i, j] connects input unit i of layer k to output unit j. `W<k> row i` lists W<k>[i, 0], W<k>[i, 1], ... . `b<k>` is the bias of layer k.
- Goal: land safely on the landing pad and come to rest, maximising the episode return (an episode scoring >= 200 is considered a solution). It is evaluated by the mean return over many episodes with random starting conditions as described in the environment.

## The network with every weight at 0 (the template you fill in)

```
W1 row 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 1  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 3  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 4  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 5  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 6  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 7  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b1  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 1  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 3  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 4  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 5  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 6  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 7  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 8  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 9  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 10  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 11  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 12  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 13  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 14  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 15  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 16  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 17  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 18  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 19  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 20  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 21  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 22  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 23  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 24  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 25  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 26  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 27  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 28  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 29  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 30  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 31  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 32  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 33  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 34  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 35  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 36  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 37  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 38  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 39  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 40  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 41  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 42  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 43  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 44  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 45  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 46  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 47  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 48  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 49  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 50  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 51  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 52  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 53  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 54  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 55  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 56  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 57  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 58  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 59  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 60  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 61  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 62  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 63  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W3 row 0  0 0 0 0
W3 row 1  0 0 0 0
W3 row 2  0 0 0 0
W3 row 3  0 0 0 0
W3 row 4  0 0 0 0
W3 row 5  0 0 0 0
W3 row 6  0 0 0 0
W3 row 7  0 0 0 0
W3 row 8  0 0 0 0
W3 row 9  0 0 0 0
W3 row 10  0 0 0 0
W3 row 11  0 0 0 0
W3 row 12  0 0 0 0
W3 row 13  0 0 0 0
W3 row 14  0 0 0 0
W3 row 15  0 0 0 0
W3 row 16  0 0 0 0
W3 row 17  0 0 0 0
W3 row 18  0 0 0 0
W3 row 19  0 0 0 0
W3 row 20  0 0 0 0
W3 row 21  0 0 0 0
W3 row 22  0 0 0 0
W3 row 23  0 0 0 0
W3 row 24  0 0 0 0
W3 row 25  0 0 0 0
W3 row 26  0 0 0 0
W3 row 27  0 0 0 0
W3 row 28  0 0 0 0
W3 row 29  0 0 0 0
W3 row 30  0 0 0 0
W3 row 31  0 0 0 0
W3 row 32  0 0 0 0
W3 row 33  0 0 0 0
W3 row 34  0 0 0 0
W3 row 35  0 0 0 0
W3 row 36  0 0 0 0
W3 row 37  0 0 0 0
W3 row 38  0 0 0 0
W3 row 39  0 0 0 0
W3 row 40  0 0 0 0
W3 row 41  0 0 0 0
W3 row 42  0 0 0 0
W3 row 43  0 0 0 0
W3 row 44  0 0 0 0
W3 row 45  0 0 0 0
W3 row 46  0 0 0 0
W3 row 47  0 0 0 0
W3 row 48  0 0 0 0
W3 row 49  0 0 0 0
W3 row 50  0 0 0 0
W3 row 51  0 0 0 0
W3 row 52  0 0 0 0
W3 row 53  0 0 0 0
W3 row 54  0 0 0 0
W3 row 55  0 0 0 0
W3 row 56  0 0 0 0
W3 row 57  0 0 0 0
W3 row 58  0 0 0 0
W3 row 59  0 0 0 0
W3 row 60  0 0 0 0
W3 row 61  0 0 0 0
W3 row 62  0 0 0 0
W3 row 63  0 0 0 0
b3  0 0 0 0
```

# Answer format

- First a line `CONTROLLER: <the controller your weights implement, at most 300 characters>`.
- Then your explanation of how the network works.
- End your reply with exactly one machine-readable block of weights, delimited like this:

WEIGHTS:
W1[2,0] = 12.0
b1[0] = -3.5
W2 row 0  <64 numbers>
b3 <4 numbers>
END_WEIGHTS

Rules for the WEIGHTS block: one entry per line, no comments or prose inside the block; indices are 0-based; `W<k>[i,j] = v` sets the weight from input unit i to output unit j of layer k (same orientation as the template rows); `b<k>[j] = v` sets one bias; `W<k> row i <values>` sets a whole row (exactly as many values as the template row); `b<k> <values>` sets a whole bias vector. **Every weight you do not specify stays 0.** Only the last WEIGHTS block in your reply is used.
