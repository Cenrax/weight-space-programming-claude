I'm trying to pin down which environment has a 10-dimensional observation and 5 discrete actions. I'm going through common candidates: LunarLander has 8 observations and 4 actions, Acrobot 6 and 3, CartPole 4 and 2, and MountainCar 2 and 3, so none of them match.

I'm checking the PettingZoo MPE scenarios, comparing observation sizes against the 10-dim input and 5 actions, and simple_adversary with two agents looks like a match for the good agents at 10 observations.

I'm checking whether the final-layer weights make sense for MPE, since only actions 0 and 2 seem to carry any signal while 1, 3 and 4 stay near zero. The biases are small and mixed in sign, which suggests the policy barely moved from initialization or settled between no-op and right. I'm now wondering whether a different environment with 10 observations and 5 actions fits better.

I'm weighing highway-env next, since its discrete meta-actions give five options, though I'd need to check its observation shape. I'm also wondering whether the trained output layer, with only two actions carrying weight, points to a degenerate policy rather than a random init.

I'm weighing highway-env's five meta-actions again, but its default observation is 25 values, so the 10-dim input doesn't fit unless it's configured differently. I'm also checking the MPE simple_adversary layout, where the good agents have 10 observations, though only actions 0 and 2 carrying weight would be an odd result for that scenario.

I'm checking the rest of the PettingZoo MPE scenarios, like simple_crypto and simple_reference, and also considering non-MPE options like Catch, FrozenLake, and Taxi, though none of those match a 10-dim input with 5 actions. The fact that only two actions carry learned weight in opposite directions makes me think of a simple one-dimensional control task.

I'm settling on MPE simple_adversary as the likeliest fit, since its good agents have 10 observations, though I'm only moderately confident and can't easily verify the observation layout from memory.

