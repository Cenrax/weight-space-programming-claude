I'm weighing a few simple linear policies for the controller, such as a weighted sum of theta and thetadot, and considering whether a rule like "3*theta + thetadot" could reach around 200 steps.

I'm settling on a linear controller of the form u = k·s with gains around [1, 1.5, 18, 3] over x, xdot, theta and thetadot. I'm checking the signs against the cartpole dynamics: a positive theta should push the cart right, and the position gain should also be positive because of the non-minimum-phase behavior. Then I want to sanity-check the relative scale of those gains for a bang-bang policy at dt = 0.02.

I'm now picking the concrete gains, roughly s = 1.0·x + 1.5·xdot + 25·theta + 4·thetadot, with the LQR-style magnitudes as a guide. I'm also checking the x-gain sign by walking through how pushing right tilts the pole left and then flips the controller's response.

I'm settling the implementation: the hidden unit will be a tanh of the state sum, with the sign of the output preserved, and the second layer gives action 1 when its output is positive. I'm checking the tie-breaking at zero and that float32 precision is fine for small values.

I'm checking the chattering amplitude: each step changes thetadot by about 0.3, so with a gain of 4 the controller's input shifts by roughly 1.2 per step, while the theta term stays small. Next I want to work out how much the cart velocity changes each step.

I'm settling on gains closer to the LQR-typical values, around [1.0, 1.7, 19, 3.5], since stronger theta and thetadot weights should keep the cart from centering too aggressively. I'm also checking whether the bang-bang policy behaves like a sliding-mode controller on the surface s = 0, which depends on the stability of that surface.

I'm settling on the weights as they are, since scaling them by one changes nothing, and I'm accepting the LQR-style gains because they're commonly reported to solve CartPole.

