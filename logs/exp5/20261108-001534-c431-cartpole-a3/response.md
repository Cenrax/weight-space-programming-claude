CONTROLLER: Bang-bang linear state feedback: push right (action 1) if s = 1.0·x + 1.5·ẋ + 20·θ + 4·θ̇ > 0, otherwise push left (action 0).

**How the controller works**

The policy is an LQR-style switching rule built from the linearised dynamics. Around upright, with F = ±10 N:
- θ̈ ≈ 15.8·θ − 1.46·F
- ẍ ≈ 0.98·F − 0.72·θ

**Pole terms (20·θ + 4·θ̇).** Pushing right when θ > 0 drives θ back toward zero. The surface θ + 0.2·θ̇ = 0 acts like a sliding surface: the angle decays with a time constant of about 0.2 s. That is ten control steps, so the 0.02 s step is short enough for bang-bang switching to hold the state near the surface.

**Cart terms (1.0·x + 1.5·ẋ).** These are small positive gains. Their sign is the non-minimum-phase one that LQR typically gives for this system. To bring a cart at x > 0 back to centre, you first push right. That makes the pole lean left, and then holding the pole up carries the cart leftward. The gains are small compared with the angle terms, so balancing the pole always dominates and the cart drifts slowly back toward the middle.

**How the network implements it**

Only one unit is used in each layer; every other weight stays 0, so the other units output tanh(0) = 0.
- **Hidden unit 0 of layer 1:** h1₀ = tanh(1.0·x + 1.5·ẋ + 20·θ + 4·θ̇). tanh keeps the sign of s.
- **Hidden unit 0 of layer 2:** h2₀ = tanh(10·h1₀), which still has the same sign as s.
- **Output layer:** logit₁ = h2₀ and logit₀ = −h2₀. The argmax is therefore action 1 (push right) exactly when s > 0, and action 0 otherwise.
- **Ties:** a tie needs s to be exactly 0, which in practice never happens.

WEIGHTS:
W1[0,0] = 1.0
W1[1,0] = 1.5
W1[2,0] = 20.0
W1[3,0] = 4.0
W2[0,0] = 10.0
W3[0,0] = -1.0
W3[0,1] = 1.0
END_WEIGHTS