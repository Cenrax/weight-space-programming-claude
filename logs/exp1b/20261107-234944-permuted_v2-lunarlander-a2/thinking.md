I'm trying to pin down which environment has a 10-dimensional observation and 5 discrete actions. I'm going through the usual Gymnasium candidates and classic control tasks like Acrobot, LunarLander, CartPole and MountainCar, and none of them match. CarRacing has 5 discrete actions but uses image observations, so I'm wondering whether a discretized Pendulum or something like MinAtar could fit.

I'm checking the PettingZoo MPE scenarios next, since their observation sizes might line up. simple_adversary's good agents come out to 10 dimensions with 5 discrete actions, which fits, and I'm also weighing Highway-env's meta-actions against its 25-dimensional observation.

I'm comparing the output-weight columns against the MPE action layout, where left and right should be opposite pairs, but the sign pattern here pairs columns 1 with 3 and 2 with 0/4 instead. That doesn't line up with a simple left/right opposition, so I'm wondering whether the columns are mapped differently than I assumed.

I'm checking whether the weight columns line up with the MPE action layout, and the sign patterns suggest the columns don't map onto simple left/right opposites. I'm also weighing other 5-action environments, like a grid-world or Highway-env with a custom observation, before settling on simple_adversary.

I'm working out the observation layout for the good agents in simple_adversary, which comes to 10 dimensions: goal offset, landmark offsets, and other-agent offsets. I'm checking whether the first two inputs carry the left/right and up/down signal, though the exact identification hinges mostly on matching the dimensions.

I'm settling on simple_adversary's good agents, since their 10-dimensional observation fits and I'm only moderately confident in it, around 0.4.

