I'm looking for a standard RL benchmark with 10 observation dimensions and 5 discrete actions. The classic Gym environments don't match, and I'm checking the MPE environments next, though simple_spread has 18 observations.

I'm checking the other simple_* scenarios in MPE: simple_adversary's good agent works out to 10 observation dimensions with 5 actions, which fits the target.

I'm working through the MPE scenarios' observation sizes. simple_spread comes out to 18 observations for three agents, so it doesn't fit. I'm settling on simple_adversary, since its good agent's 10-dimensional observation matches the target, though the weight patterns in the last layer are hard to verify directly.

