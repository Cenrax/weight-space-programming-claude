CONTROLLER: Bang-bang linear state feedback: s = 1.0·x + 1.7·ẋ + 19·θ + 3.5·θ̇. Push right (action 1) if s > 0, else push left (action 0). The gains are LQR-like, so the pole is balanced and the cart is pulled back toward the centre.

**How the network works**

- **Layer 1, unit 0:** computes h1₀ = tanh(1.0·x + 1.7·ẋ + 19·θ + 3.5·θ̇). Tanh is odd and monotonic, so the sign of h1₀ equals the sign of s. All other units are 0.
- **Layer 2, unit 0:** computes h2₀ = tanh(h1₀), which keeps the same sign. All other units are tanh(0) = 0.
- **Output:** logit₀ = 0 and logit₁ = h2₀. Argmax picks action 1 (push right) when s > 0 and action 0 (push left) otherwise. An exact tie goes to action 0, which does not matter.

**Why these signs and gains**

- **Angle terms:** pushing right gives a negative angular acceleration. So when the pole leans right (θ > 0) or is rotating right (θ̇ > 0), we push right.
- **Cart terms:** the cart-position and cart-velocity gains are also positive, as in the standard LQR solution for an inverted pendulum on a cart. The controller first pushes the cart further out so the pole tilts back toward the centre, then the angle terms drive the cart home. This stops the cart drifting past ±2.4 during the 500 steps.
- **Gain size:** the gains match LQR with Q = I and R = 1 for these parameters, roughly K ≈ [1, 1.6, 19, 3.4]. Under bang-bang actuation, one step changes θ̇ by about 0.29 rad/s and ẋ by about 0.2 m/s. That gives small chattering, around 0.006 rad in angle, far inside the ±0.21 rad limit.

WEIGHTS:
W1[0,0] = 1.0
W1[1,0] = 1.7
W1[2,0] = 19.0
W1[3,0] = 3.5
W2[0,0] = 1.0
W3[0,1] = 1.0
END_WEIGHTS