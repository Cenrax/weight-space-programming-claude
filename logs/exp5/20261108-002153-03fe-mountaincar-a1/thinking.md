I'm working out a policy for MountainCar, and the simple one is to push in the direction of the current velocity, which usually reaches the goal in around 100 to 120 steps. I'm also weighing an energy-based heuristic that seems to land somewhere in that same range, and wondering whether the optimum sits near -100.

I'm weighing a start-specific tweak: the valley bottom is near -0.5236, so from rest on the right side gravity pulls back left, and pushing left from the start swings up the left hill and returns, while pushing right only drifts a little before falling back.

I'm wondering whether it's better to go left first from most starts, since the energy approach builds amplitude by pushing with the velocity. For a start at v=0, pushing left seems marginally better than right, because gravity on the left side of the valley pulls right and the net push still comes out leftward. Either way, always pushing left at rest gives a consistent pumping pattern.

I'm settling on the rule of pushing right when velocity is positive and left otherwise, since starting by going left builds up the swing. A left wall hit that zeroes velocity still leads to a rightward push, so the pattern holds.

I'm wiring the network so the policy is a sign test on velocity: a tanh unit on velocity feeds the logits, favoring action 2 when velocity is positive and action 0 otherwise, with ties at zero going to left.

I'm picking the network weights now: a large gain on velocity so the tanh saturates to the sign, a second tanh layer to sharpen it, and output logits that are opposite for left and right with zero for the middle action.

