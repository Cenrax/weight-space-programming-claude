DIAGNOSIS: The policy has learned to hover. Its main-engine logit wins whenever vy dips below 0, and it has almost no drive to descend from y. So it holds y≈1.0–1.2 with vy≈0 for all 1000 steps (577 main fires) and gets truncated without ever landing.

**Why it fails**

- **The trace shows hovering, not landing.** After about step 40 the altitude creeps from y≈1.17 to 0.97 over 950 steps, and vy stays within ±0.04.
- **Main and side engines alternate to cancel any drift.** Action 2 (main engine) is chosen whenever vy turns slightly negative. Actions 1 and 3 (side engines) fight the small angle and x drift: x goes out to 0.7 and slowly comes back.
- **Action 0 is chosen only once**, and the legs never touch the ground.
- **First-layer weights explain it.** The vy row (W1 row 3) has the largest weights, for example unit 44 at 0.921 and several around ±0.6. The y row (W1 row 1) has only small weights (≤0.43). So the network reacts strongly to any downward velocity but has no notion that "high up means go down".
- **Output layer reinforces it.** Many h2 units feed the main-engine logit (W3 column 2), and b3[2]=+0.10 also favours it.
- **Result.** The policy found a local optimum that avoids crashing (−100) by never landing. It pays the fuel cost and the distance-shaping penalty instead.

**Fix: hard-wire the classic Gymnasium heuristic controller into the network**

The heuristic is:
- angle_targ = clip(0.5·x + vx, ±0.4)
- A = 0.5·(angle_targ − angle) − ang_vel
- H = 0.55·|x|·0.5 − 0.5·y − 0.5·vy
- Choose main (2) if H > |A| and H > 0.05. Otherwise choose right (3) if A < −0.05, left (1) if A > 0.05, else nothing (0).

**Key observation.** This decision is exactly the argmax of four logits: [c, A, H, −A], where c is the threshold 0.05. If A and H both pass through the same monotone function g, the argmax is unchanged as long as c becomes g(0.05).

**Layer 1** (only units 0–5 are used; all other units get zero weights and zero bias, so they output exactly 0):
- u0 = tanh(0.02·(−0.5·angle − ang_vel)). This keeps the linear part of A in tanh's near-linear range.
- u1 = tanh(2.5·(0.5x + vx)). Then 0.4·u1 is a smooth version of the clip to ±0.4.
- u2 = tanh(0.02·(−0.5y − 0.5vy)). This is the linear part of H.
- u3 = tanh(2x − 1) and u4 = tanh(2x + 1). Their combination u3 − u4 + 2·tanh(1) is an even function that approximates |x|.
- u5 = tanh(legL + legR). This is the ground-contact signal.

**Layer 2** (only units 0–2 are used; everything else is 0):
- h0 = tanh(A), computed as 50·u0 + 0.2·u1.
- h1 = tanh(H), computed as 50·u2 + 0.275·(u3 − u4) + 0.4189, with the |x| constant folded into the bias.
- h2 = tanh(2·u5).

**Output layer:**
- logit0 = 10·tanh(0.05) + 5·h2
- logit1 = 10·h0
- logit2 = 10·h1
- logit3 = −10·h0

Behaviour this gives:
- **During descent** the lander tracks a target velocity of vy ≈ −y, steers toward x = 0, and keeps the angle under control.
- **Once a leg touches**, the do-nothing logit is boosted. Engines only fire in extreme cases, so the lander settles and goes to sleep on the pad.

EDITS:
W1 row 0 0 1.25 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 1 0 0 -0.01 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 2 0 2.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 3 0 0 -0.01 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 4 -0.01 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 5 -0.02 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 6 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 7 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b1 0 0 0 -1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 0 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 1 0.2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 2 0 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 3 0 0.275 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 4 0 -0.275 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2 row 5 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b2 0 0.4189 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W3 row 0 0 10 0 -10
W3 row 1 0 0 10 0
W3 row 2 5 0 0 0
b3 0.4996 0 0 0
END_EDITS