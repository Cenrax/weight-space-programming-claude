CONTROLLER: Bang-bang energy pumping: push right (action 2) when velocity > 0, push left (action 0) when velocity < 0. At velocity exactly 0 (the reset state) the logits tie and the argmax picks left, so every episode starts by swinging left.

**How the network works**

- **Layer 1:** unit 0 computes h1[0] = tanh(1000·velocity). This is roughly the sign of the velocity, and it keeps the correct sign even for very small velocities. The position input is unused.
- **Layer 2:** unit 0 computes h2[0] = tanh(10·h1[0]). This sharpens the sign further.
- **Output:** the logits are (−h2[0], 0, +h2[0]).
  - When velocity > 0, action 2 (accelerate right) has the largest logit.
  - When velocity < 0, action 0 (accelerate left) has the largest logit.
  - When velocity = 0 (at reset, or after hitting the left wall), all logits are 0. The tie goes to the lowest index, so the action is left.
- **Unused units:** all other units have zero weights and zero biases. Their activations are 0, so they contribute nothing.

**Why this controller should work**

- Pushing in the direction of motion adds the maximum possible mechanical energy at every step. The oscillation amplitude grows as fast as the force allows, until the car clears the right hill.
- The first action pushes left. That builds a swing up the left hill, followed by a powerful run to the right. From every start in [−0.6, −0.4], this should reach the flag in roughly 1.5 swings, well inside the 200-step limit.
- If the car hits the left wall, its velocity resets to 0 and the controller pushes left. Gravity there (≈ +0.0022) outweighs that push (0.001), so the car still moves right. On the next step its velocity is positive and the controller pushes right.
- I worked this out by reasoning only and did not run it. My expected mean return is roughly −100 to −120.

WEIGHTS:
W1[1,0] = 1000.0
W2[0,0] = 10.0
W3[0,0] = -1.0
W3[0,2] = 1.0
END_WEIGHTS