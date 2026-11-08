I'm looking at an 8-input, 4-action setup, which fits LunarLander. To check whether it's trained, I'm comparing weight magnitudes: the angle and angular velocity rows look larger, which would fit the task, while the leg-contact rows look smaller. The nonzero biases also point toward training, since they'd normally start at zero.

I'm checking the rest of the weight matrix for structure, and the left and right engine columns look strongly anti-correlated across several rows. The noop and main engine columns show a similar opposing pattern, which points to learned structure rather than random initialization.

