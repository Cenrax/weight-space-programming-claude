I'm trying to match an environment with 10 observation dimensions and 5 discrete actions. I'm going through candidates, ruling out MinAtar and checking Reacher, whose observation size doesn't fit, and then looking at MPE and PettingZoo scenarios like simple_spread and simple_tag.

I'm checking the PettingZoo MPE scenarios next, since their discrete action sets match, and noting observation sizes like simple_adversary's 10. The network's behaviour looks like it hinges on one input and two actions, which makes me wonder whether the observations were padded to fit a common size.

I'm weighing whether this could be CartPole padded out, but its policy would lean on pole angle and angular velocity and use actions 0 and 1, which doesn't match. MountainCar fits better: the policy pushes in the direction of velocity, so the input at index 1 and actions 0 and 2 would be the active ones, with the rest padding. I'm checking the sign of the first layer's weights against that, though tracing it through the second layer is hard.

Settling on MountainCar-v0 as the match, since the velocity-driven push fits that environment.

