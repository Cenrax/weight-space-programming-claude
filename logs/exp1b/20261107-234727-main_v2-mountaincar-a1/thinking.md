I'm checking which benchmark environment has 10 input dimensions and 5 discrete actions. Highway-env and MiniGrid don't fit, and PettingZoo simple_spread has 18 observations, so I'm looking at the MPE scenarios next, where simple_tag's adversary has 16 observations and 5 actions.

I'm going through the remaining candidates: the other MPE scenarios and a few discrete-action Gym-style environments, but none of them clearly match. The key finding so far is that the policy hinges almost entirely on input 1, switching between actions 0 and 2 based on its sign, which makes me think simple_adversary's agent observation space (around 10 dimensions) is worth checking.

I'm weighing the PettingZoo simple_adversary agents, where the good agents' observations work out to 10 dimensions with 5 actions, but the single-policy setup makes that less convincing. I'm also running through the other MPE scenarios and a few classic Gym environments like Acrobot, LunarLander, and Taxi, and none of them match cleanly.

I'm working through the MPE simple_adversary observation layout, where the good agent sees its goal offset, two landmark offsets, and two teammate offsets, which adds up to 10 dimensions.

I'm checking the MPE action layout, where 0 is no-op and 2 is rightward, and the policy only ever chooses between them based on input 1. That makes me wonder whether this is less a learned MPE policy and more a simple rule like MountainCar's, where action 2 is picked when velocity is positive and action 0 otherwise.

I'm thinking the input-1 weights are so large because that feature has a small scale, like MountainCar's velocity. The other inputs and the unused logits look like padding, which would mean the environment's 10 inputs and 5 actions were extended to fit a shared format.

I'm settling on MountainCar-v0, since its velocity-sign rule would explain the policy: input 1 is the small-scale velocity feature, and the padded dimensions and unused actions fit a shared 10-input, 5-output format.

