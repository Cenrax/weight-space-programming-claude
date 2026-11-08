CONTROLLER: Bang-bang linear state feedback: push right (action 1) if 0.5·x + 1.0·x_dot + 10·θ + 2·θ_dot > 0, otherwise push left (action 0). This is an LQR-style gain ratio, with all gains having the same sign.

**Why this controller**

In this environment's sign convention, a positive force (push right) produces negative angular acceleration of the pole. So a pole leaning right (θ>0) or falling right (θ_dot>0) is corrected by pushing right.

The cart-position terms also take a positive sign, which may look backwards. To return a cart sitting at x>0 to the centre, the pole must first be tipped left, and that requires a brief push to the right. This "non-minimum-phase" structure is why LQR solutions for the cart-pole have all four gains with the same sign.

**How the gains were chosen**

Typical LQR gain ratios for this system are roughly x : x_dot : θ : θ_dot ≈ 1 : 1.6 : 18 : 3. I used s = 0.5x + 1.0x_dot + 10θ + 2θ_dot. This is θ + 0.2θ_dot + 0.05x + 0.1x_dot, scaled by 10.
- The angle and angular-velocity terms dominate, which keeps the pole upright.
- The small cart terms counteract slow drift of the cart over 500 steps.

These gains come from reasoning only; I could not run any test episodes.

**How the network implements it**

- **Hidden unit 1, layer 1:** h1₀ = tanh(s), using the four input weights above and zero bias. Because tanh is monotonic, it preserves the sign of s even when it saturates.
- **Hidden unit 1, layer 2:** h2₀ = tanh(h1₀) also preserves the sign.
- **Output layer:** the logits are [−h2₀, +h2₀], so argmax picks action 1 exactly when s > 0. An exact tie at s = 0 goes to action 0, which is harmless.
- **All other weights:** they are zero, so the unused units output tanh(0) = 0 and contribute nothing.

WEIGHTS:
W1[0,0] = 0.5
W1[1,0] = 1.0
W1[2,0] = 10.0
W1[3,0] = 2.0
W2[0,0] = 1.0
W3[0,0] = -1.0
W3[0,1] = 1.0
END_WEIGHTS