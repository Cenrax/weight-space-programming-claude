I'm going through a list of candidate RL environments to find one with 10 observation dimensions and 5 discrete actions. I've already ruled out a few, like Acrobot, LunarLander, CartPole, MountainCar, and Taxi, since their dimensions don't match.

I'm checking the remaining candidates, Reacher, Pusher, Swimmer, Hopper and the inverted pendulums, but none of them fit a discrete 5-action setup. Highway-env has 5 meta actions, yet its observations are 25 values. In PettingZoo's MPE set, simple_adversary's good agent sees 10 values and has 5 discrete actions, which makes it the best match so far.

The output layer's weights are lopsided: actions 0 and 2 have large opposing values while 1, 3, and 4 stay near zero, so the net seems to be leaning on just two actions. That doesn't match the MPE action meanings I'd expect, where 0 and 2 wouldn't naturally be opposites, so I'm wondering whether this is a different environment entirely.

I'm checking whether the weight pattern points to a collapsed policy: columns 1, 3 and 4 look like untrained initialization near 0.01, while columns 0 and 2 have grown, so training seems to have moved only two actions. The biases may have been learned as well. I'm still stuck on which environment has 10 inputs and 5 actions, and whether this is a known benchmark at all.

I'm going with the MPE simple_adversary environment for now, though I'm only about 35% confident, since I can't yet explain why only actions 0 and 2 carry weight.

