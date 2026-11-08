I'm looking for a standard RL benchmark with 10 observation dimensions and 5 discrete actions. Image-based MinAtar is out, and the classic control environments I know don't match, so I'm checking whether highway-env's discrete meta-actions fit, though its observation size seems off.

I'm checking the MPE environments from PettingZoo next, since simple_adversary's good agent seems to match 10 observation dimensions and 5 actions. The arithmetic for it checks out, though I'm still not sure about the action structure in the logits.

I'm now weighing whether W3's columns, which look nearly collapsed onto one feature, could be random initialization, but the consistent sign patterns suggest it has been trained, perhaps only briefly. I'm also running through other 10-dimension, 5-action candidates and settling on MPE's simple_adversary as the most plausible fit.

I'm checking the first few rows of the input weights, where a couple of them carry noticeably larger values, which might correspond to relative landmark positions. I'm treating that as a plausible reading and holding my confidence at moderate.

