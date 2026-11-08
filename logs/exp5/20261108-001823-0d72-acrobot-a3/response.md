CONTROLLER: Bang-bang energy pumping on the slow "swing" mode: torque = +1 if (dθ1 + 0.4·dθ2) > 0, else -1 (zero torque is never used).

**Why this rule**

Linearise the acrobot about the hanging position:
- Mass matrix M = [[4.5, 1.75], [1.75, 1.25]].
- Gravity stiffness K = g·[[2, 0.5], [0.5, 0.5]].

This gives two oscillation modes:
- **Slow "swing" mode** (ω ≈ 1.88 rad/s): θ2 ≈ 2.8·θ1, so link 2 swings with link 1 and even further. This is the mode that lifts the tip, because link 2 can flip upward while link 1 is high.
- **Fast "jackknife" mode** (ω ≈ 2.82 rad/s): θ2 ≈ -1.79·θ1, so the links fold against each other. The tip barely rises in this mode, so energy put here is mostly wasted.

The power the motor delivers is τ·dθ2. In modal coordinates this power goes to:
- the swing mode in proportion to τ·(dθ1 + 0.56·dθ2);
- the jackknife mode in proportion to −τ·(dθ1 − 0.56·dθ2), i.e. with the opposite sign.

So with τ = sign(dθ1 + c·dθ2):
- For any c ≥ 0, the swing mode is resonantly pumped. This is the bang-bang version of Spong's swing-up rule, which bends link 2 in the direction link 1 is moving.
- For c < 0.56, the jackknife mode is mildly damped rather than pumped.
- c = 0.4 is a hedge for the nonlinear phase. If link 2 whirls fast (|dθ2| ≫ |dθ1|), the rule reduces to sign(dθ2), which is the greedy energy-maximising torque. This keeps energy from stalling, so the episode should not hit the -500 cap.

The bang-bang switching follows the velocity signs, so it tracks the changing swing frequency as amplitude grows. Energy then builds until the free end is flung above the line. I could not simulate it, so this is a reasoned choice rather than a tested one.

**Network implementation**

- **Layer 1:** h1[0] = tanh(10·dθ1 + 4·dθ2). This is a sharp sign detector for dθ1 + 0.4·dθ2.
- **Layer 2:** h2[0] = tanh(5·h1[0]). This keeps the same sign and saturates further.
- **Output layer:**
  - logit0 = −h2[0]
  - logit2 = +h2[0]
  - logit1 = −10, so zero torque is never chosen.
- All other units are zero.
- If the input is exactly 0, the tie between actions 0 and 2 goes to action 0, which is harmless.

WEIGHTS:
W1[4,0] = 10.0
W1[5,0] = 4.0
W2[0,0] = 5.0
W3[0,0] = -1.0
W3[0,2] = 1.0
b3[1] = -10.0
END_WEIGHTS