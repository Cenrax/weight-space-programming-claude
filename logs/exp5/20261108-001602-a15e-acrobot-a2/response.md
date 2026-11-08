CONTROLLER: Bang-bang energy pumping: torque = +1 if dθ1 + 0.25·dθ2 > 0, else −1. Torque follows link 1's swing direction, which pumps the in-phase swing mode, damps the out-of-phase mode, and avoids runaway spinning of link 2.

**Why this controller**

- **How energy enters the system.** The actuator works only through the joint angle, so the energy input rate is τ·dθ2. The target height needs about 24.5 J above the hanging state; the cheapest goal configuration is link 1 horizontal with link 2 pointing up.
- **Two swing modes near the bottom.** Linearising the given dynamics (d1 = 4.5, d2 = 1.75, m22 = 1.25, gravity stiffness [[19.6, 4.9], [4.9, 4.9]]) gives:
  - a slow in-phase mode, ω ≈ 1.88 rad/s, with θ2 ≈ +2.8·θ1;
  - a fast out-of-phase mode, ω ≈ 2.82 rad/s, with θ2 ≈ −1.8·θ1.
- **Why pump the in-phase mode.** In that mode link 2 swings much higher than link 1, which is what raises the tip.
- **Why sign(dθ1) does this.** Torque = sign(dθ1) gives positive power (dθ2 has the same sign as dθ1) in the in-phase mode. It gives negative power in the out-of-phase mode, so that mode is damped.
- **Why not sign(dθ2).** Pure sign(dθ2) pumping would add energy faster, but mostly into the out-of-phase mode. It also risks link 2 spinning continuously, which can stall the swing-up.
- **The small dθ2 term.** Adding 0.25·dθ2 recovers positive power near the turning points of link 1, where |dθ1| is small but |dθ2| is large. Since 0.25 < 0.56, the out-of-phase mode is still damped.
- **Expected performance.** This is a rough estimate from the linearised analysis, not a simulation: roughly 80–110 steps to reach the goal height.

**How the network implements it**

- **Layer 1:** hidden unit 0 is h = tanh(10·dθ1 + 2.5·dθ2), which has the sign of s = dθ1 + 0.25·dθ2.
- **Layer 2:** hidden unit 0 is g = tanh(5·h), which keeps the same sign and saturates.
- **Output:** the logits are [−g, 0, +g].
  - If s > 0, action 2 (+1 torque) wins.
  - If s < 0, action 0 (−1 torque) wins.
  - At exactly 0 all logits tie, and the tie goes to action 0.
- All other weights are 0.

WEIGHTS:
W1[4,0] = 10.0
W1[5,0] = 2.5
W2[0,0] = 5.0
W3[0,0] = -1.0
W3[0,2] = 1.0
END_WEIGHTS