DIAGNOSIS: The net holds vertical speed near zero instead of descending. Its main-engine choice depends on vy (W1 row 3, up to ±0.9) while altitude y has small weights (W1 row 1). It hovers near y≈1 for 1000 steps (58% main engine) and never lands.

## Why it fails

The trace shows a hover, not a crash:
- Altitude drifts only from y=1.40 to y=0.98 over 1000 steps.
- Vertical speed vy stays around 0 (±0.04) the whole time.
- The main engine fires 577 times out of 1000.
- Leg contacts never occur, and the episode ends by the time limit with return −81.

In the weights:
- **Vertical speed dominates.** The vy input row (W1 row 3) is the strongest row, with weights such as 0.921, −0.697, 0.628 and −0.638. The altitude row (W1 row 1) is mostly 0.0–0.3.
- **The output layer favours the main engine.** The main-engine column of W3 together with b3[2]=+0.10 means a slight downward vy flips the argmax to action 2. The policy therefore behaves as a "hold vy≈0" regulator rather than a "descend toward y=0" controller.
- **The side engines dither.** About 420 left/right firings mostly cancel each other. Nothing drives y toward zero, so the lander never touches down and never goes to sleep.

## The fix

I replace the network with an exact neural implementation of Gymnasium's built-in heuristic lander, which normally lands on the pad. The heuristic is:

- angle_targ = clip(0.5·x + vx, ±0.4)
- A = 0.5·(angle_targ − angle) − angvel
- H = 0.275·|x| − 0.5·y − 0.5·vy

It fires the main engine if H > |A| and H > 0.05. Otherwise it fires the right engine if A < −0.05, the left engine if A > 0.05, and does nothing otherwise. With leg contact, the side engines are disabled.

This rule is exactly the argmax of the logits [0.05, A, H, −A]. Main wins iff H beats A, −A and 0.05. A side engine wins iff |A| > 0.05, with the sign of A choosing which one.

**Layer 1.** All other units have zero weights, so they output 0.
- u0 = tanh(1.25x + 2.5vx) gives the soft clip: angle_targ ≈ 0.4·u0.
- u1 = tanh(−0.5·angle − angvel).
- u2 = tanh(−0.5y − 0.5vy).
- u3 = tanh(1 + 2x) and u4 = tanh(1 − 2x) together form an even bump. From them, 0.22·(1.523 − u3 − u4) ≈ 0.275|x|.
- u5 = tanh(legL + legR).

**Layer 2.** All other units are zero.
- v0 = tanh(0.2·u0 + u1) ≈ A.
- v1 = tanh(u2 − 0.22u3 − 0.22u4 + 0.3351) ≈ H.
- v2 = tanh(3·u5) ≈ 1 when any leg touches.

**Output logits.**
- L0 = 0.05
- L1 = v0 − 5·v2
- L2 = v1
- L3 = −v0 − 5·v2

The tanh layers stay in their near-linear range for the small values where decisions are made. Where they saturate (large values), they keep the sign and ordering, so the decisions are unchanged.

EDITS:
W1 row 0 1.25 0 0 2 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 1 0 0 -0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 2 2.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 3 0 0 -0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 4 0 -0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 5 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 6 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 7 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 0 0.2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 2 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 3 0 -0.22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 4 0 -0.22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 5 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b2 0 0.3351 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W3 row 0 0 1 0 -1
W3 row 1 0 0 1 0
W3 row 2 0 -5 0 -5
b3 0.05 0 0 0
END_EDITS