I'm working out a linear controller to be built as a tanh network, with the action set by the sign of a weighted sum of the state. I'm considering LQR-style gains on the angle, angular velocity, and cart position terms.

I'm weighing whether the sign of the angle and angular velocity terms should drive the push, and whether the cart position terms should be added with small gains to keep it centered. I'm also comparing typical LQR gain values for the cart and pole states against a simpler bang-bang rule.

I'm checking the sign convention: a positive force pushes the pole further right, so correcting a positive angle means pushing right. To bring the cart back toward center it has to tilt the pole first, which means all the gains share the same sign. I'm settling on a bang-bang rule with the position and velocity terms weighted lightly against the angle and angular velocity.

I'm settling on a weighted sum that mostly leans on the angle with about a fifth of the angular velocity, plus small cart position and velocity terms so the cart drifts back toward center over long runs.

I'm choosing concrete weights for the sliding surface, with the angle term dominant and smaller terms on angular velocity, cart position, and cart velocity. I'm then setting the first hidden unit to a tanh of that weighted sum, so the sign of the action still follows the sign of the sum.

I'm setting the output weights so the final two logits are opposite in sign and follow the sign of the hidden unit, with a positive weight on the first and a negative weight on the second.

