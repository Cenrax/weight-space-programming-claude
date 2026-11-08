CONTROLLER: Bang-bang energy pumping with torque = sign(dθ2 − 2·sin θ2 − 0.5·dθ1). Action 2 (+1 torque) if that sum is positive, otherwise action 0 (−1 torque). Never uses action 1.

The best version averaged about −81 over 32 test episodes, with every episode succeeding and returns between −68 and −96. I expect the 100-episode final score to be around −80 to −85, which beats the −100 reward threshold. Further tuning didn't get past that level.

**The network:** only one unit in each hidden layer is used; every other weight is zero.
- **Layer 1, unit 0:** tanh(10·(−2·sinθ2 − 0.5·dθ1 + dθ2)), which works out to W1 column 0 = [0, 0, 0, −20, −5, 10]. This is effectively the sign of the switching function.
- **Layer 2, unit 0:** tanh(5·h1₀).
- **Output:** W3 row 0 = [−1, 0, +1], so the policy is +1 torque when the sum is positive and −1 torque otherwise.

**What I changed after each round of rollouts (20 of the 25 used):**
1. **sign(dθ1):** every episode timed out at −500. With almost no starting energy, it never built up a swing.
2. **sign(dθ2):** this pushes torque in the direction the joint is moving, which adds the most energy per step (power = torque × dθ2). Mean −88. The median was about −73, but some episodes ended much later (−150 to −250) because link 2 kept spinning instead of lifting the system.
3. **Fixes that didn't work:** braking when |dθ2| was large (−93), and a target-angle controller in the style of Spong (−500 or −147).
4. **Adding a −k·sin θ2 term:** this pulls link 2 back toward straight, which stopped the runaway spinning. k = 1 gave −82, k = 2 gave −80.5, k = 4 gave −111.
5. **dθ1 and sin θ1 terms:** an extra dθ1 term between −1 and +0.3, or a ±sin θ1 term, all landed around −80 to −86. A dθ1 coefficient of −0.5 gave the tightest spread of returns, so I kept it.
6. **Separate rule for when link 1 is above horizontal:** I tried two gated controllers that switch to straightening link 2 near the top. They scored −95 and −107, so I dropped them.
7. **Final check:** k = 2.5 gave −80.9 and k = 1.5 gave −83.1, so k = 2 stayed. The saved weights are k = 2 with dθ1 coefficient −0.5; I confirmed the saved values after writing them.