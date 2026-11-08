I'm working out a linear controller for a tanh network, guessing at a simple sign-based rule on the state's angle and angular velocity, and trying a few candidate weight vectors from memory.

I'm weighing a linear controller with all-positive gains on x, x-dot, theta and theta-dot, since the cart terms need to match the pole's sign in an LQR-style design, and I'm considering concrete weights like 10 on theta and 2 on theta-dot.

I'm settling on a weighted sum of all four state terms, with the pole terms dominant and the cart terms small, and I'm picking something like 1 on x, 1.5 on x-dot, 18 on theta and 3 on theta-dot.

I'm checking the linearized dynamics to see how the force enters the pole and cart accelerations, working through the coefficients on theta and F. Then I'm reasoning about the sign of the LQR gain, since u = -Kx with a positive push to the right.

I'm reconsidering the gains, reasoning that a positive x gain fits the non-minimum-phase behavior of LQR here, and I'm going with roughly 1.0 on x, 1.5 on x-dot, 20 on theta and 4 on theta-dot, which puts the sliding surface's decay time near 0.2 seconds.

I'm setting the network up with a single hidden unit whose weights are the four state gains, which keeps the sign of the control signal intact through the tanh layers, and I'm wiring the output logits so that the rightward action wins whenever that sign is positive.

