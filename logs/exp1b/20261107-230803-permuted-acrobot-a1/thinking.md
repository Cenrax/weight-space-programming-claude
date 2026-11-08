I'm looking at the LunarLander setup: an 8-dimensional input and 4 actions. The output layer weights show a clear pattern, with two columns anti-correlated and two near zero, and the biases are small, so the network looks only weakly trained. Actions 0 and 2 seem to be the ones being used, while 1 and 3 aren't showing up.

The weights are the next thing to look at: the bias values are small and the output layer shows a clear structure, which doesn't quite fit a network that has been fully trained. The policy seems to lean toward noop and main engine while ignoring the side thrusters, and that's an odd outcome for a learned controller. I'll check the inputs next, since the main engine should fire when vertical velocity is negative, though that's hard to confirm just from the weights.

