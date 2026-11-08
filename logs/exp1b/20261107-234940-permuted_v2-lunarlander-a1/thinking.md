I'm weighing which standard environment matches a 10-dimensional observation space with 5 discrete actions. The usual classic-control options don't fit, and I'm checking whether Reacher might, though its observation size looks off.

I'm going through the PettingZoo MPE environments to find one with 10 observation dimensions and 5 discrete actions, and simple_adversary looks like the match, with its good agents at 10 observations and 5 actions.

I'm checking the MPE options next, and simple_adversary's good agents fit with 10 observations and 5 actions, though the weight structure in W3 hints at which actions might correlate.

I'm checking whether simple_adversary's good agents really reach 10 observations, which they do if the goal-relative position, the landmark positions and the other agents' positions add up to 10. I can't confirm that from the weights directly, so I'm keeping simple_adversary as a moderate-confidence match.

