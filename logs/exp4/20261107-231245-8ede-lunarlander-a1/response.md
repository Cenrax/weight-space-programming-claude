DIAGNOSIS: The policy only regulates vertical speed to about zero, so it hovers near y≈1. Height has almost no effect on the main-engine logit. "Do nothing" (b3[0]=-0.059 vs b3[2]=+0.100) almost never wins, so it never descends and gets truncated at 1000 steps.

## Why it fails

The trace shows a hover, not a landing. From step ~40 to step 999, y only drifts from 1.17 to 0.98. vy stays within about ±0.04, and both legs never touch the ground. The action counts are main 577, left 221, right 201, and "do nothing" exactly 1.

- **Speed but not height.** The main-engine logit is driven through the vy input row W1 row 3, which has large weights (0.5–0.9 on several units). Any small downward speed brings the main engine back on.
- **Height barely matters.** The y input row (W1 row 1) has small, mixed-sign weights. Nothing pushes the lander toward "descend because I'm high", so its equilibrium is vy≈0 at whatever height it happens to be.
- **Coasting is ruled out.** Logit 0 has a negative bias and small W3 column-0 weights. The policy alternates main and side thrusters instead of letting the lander fall.
- **Result.** It drifts sideways while hovering, collects shaping noise and fuel penalties (−81 total), and is truncated.

## The fix: rebuild the network as Gymnasium's heuristic lander

The edit hard-codes the reference heuristic controller from Gymnasium, `heuristic()` in lunar_lander.py:

- angle_targ = clip(0.5·x + vx, ±0.4)
- hover_targ = 0.55·|x|
- angle_todo = 0.5·(angle_targ − angle) − angvel
- hover_todo = 0.5·(hover_targ − y) − 0.5·vy
- With a leg in contact: angle_todo = 0 and hover_todo = −0.5·vy.
- Action rule: main if hover_todo > |angle_todo| and > 0.05; else right if angle_todo < −0.05; else left if angle_todo > 0.05; else nothing.

### Output layer

That decision rule is exactly an argmax over these four logits:

| Logit | Value |
|---|---|
| 0 | 0.05 |
| 1 | angle_todo |
| 2 | hover_todo |
| 3 | −angle_todo |

### Layer 1

Units 16–63 are zeroed, so they output 0. The used units are:

- **u0:** T = tanh(1.25x + 2.5vx). Then 0.4·T is a soft version of the ±0.4 clip.
- **Linear units (small gain, so tanh stays nearly linear):**
  - u1 ≈ 0.1·(−0.5·angle − angvel)
  - u2 ≈ −0.05·y
  - u3 ≈ −0.05·vy
- **Leg units:** u4 = tanh(3·legL) and u5 = tanh(3·legR).
- **|x| units:** u6–u10 and u11–u15 are step functions tanh(±10x − 10t) for t = 0.1, 0.3, …, 0.9. Their sum approximates |x|.

### Layer 2

Pre-activations are scaled by 0.2 so tanh stays nearly linear. W3 multiplies by 5 to undo the scaling.

- **v0 (A):** 0.2·angle_todo + 6·(L1+L2).
- **v1 (C):** 6·(L1+L2). This is the leg-gate compensator.
- **v2 (H):** 0.2·(0.275|x| − 0.5y) + 6·(L1+L2). This is the flight-only hover term.
- **v3 (W):** 0.2·(−0.5·vy). This term is always on.

How the leg gating works:
- With no leg contact, C = 0 and A, H are linear in their inputs.
- With a leg down, A, H and C all saturate at about 1. The logits use A−C and H−C, so the angle term and the flight hover term vanish.
- Only −0.5·vy remains, matching the heuristic's ground behaviour.

Logits are then:
- logit1 = 5(A−C)
- logit3 = −5(A−C)
- logit2 = 5(H−C) + 5W
- b3[0] = 0.05

### How it should behave

The lander should steer toward x≈0 and descend with vy around −0.1 near the ground. It should then touch down, stop firing, and fall asleep on the pad.

EDITS:
W1 row 0  1.25 0 0 0 0 0 10 10 10 10 10 -10 -10 -10 -10 -10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 1  0 0 -0.05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 2  2.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 3  0 0 0 -0.05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 4  0 -0.05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 5  0 -0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 6  0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 7  0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b1  0 0 0 0 0 0 -1 -3 -5 -7 -9 -1 -3 -5 -7 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2[0,0] = 0.04
W2[0,1] = 0
W2[0,2] = 0
W2[0,3] = 0
W2[1,0] = 2.0
W2[1,1] = 0
W2[1,2] = 0
W2[1,3] = 0
W2[2,0] = 0
W2[2,1] = 0
W2[2,2] = 2.0
W2[2,3] = 0
W2[3,0] = 0
W2[3,1] = 0
W2[3,2] = 0
W2[3,3] = 2.0
W2[4,0] = 6
W2[4,1] = 6
W2[4,2] = 6
W2[4,3] = 0
W2[5,0] = 6
W2[5,1] = 6
W2[5,2] = 6
W2[5,3] = 0
W2[6,0] = 0
W2[6,1] = 0
W2[6,2] = 0.0055
W2[6,3] = 0
W2[7,0] = 0
W2[7,1] = 0
W2[7,2] = 0.0055
W2[7,3] = 0
W2[8,0] = 0
W2[8,1] = 0
W2[8,2] = 0.0055
W2[8,3] = 0
W2[9,0] = 0
W2[9,1] = 0
W2[9,2] = 0.0055
W2[9,3] = 0
W2[10,0] = 0
W2[10,1] = 0
W2[10,2] = 0.0055
W2[10,3] = 0
W2[11,0] = 0
W2[11,1] = 0
W2[11,2] = 0.0055
W2[11,3] = 0
W2[12,0] = 0
W2[12,1] = 0
W2[12,2] = 0.0055
W2[12,3] = 0
W2[13,0] = 0
W2[13,1] = 0
W2[13,2] = 0.0055
W2[13,3] = 0
W2[14,0] = 0
W2[14,1] = 0
W2[14,2] = 0.0055
W2[14,3] = 0
W2[15,0] = 0
W2[15,1] = 0
W2[15,2] = 0.0055
W2[15,3] = 0
b2  0 0 0.055 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W3 row 0  0 5 0 -5
W3 row 1  0 -5 -5 5
W3 row 2  0 0 5 0
W3 row 3  0 0 5 0
W3 row 4  0 0 0 0
W3 row 5  0 0 0 0
W3 row 6  0 0 0 0
W3 row 7  0 0 0 0
W3 row 8  0 0 0 0
W3 row 9  0 0 0 0
W3 row 10  0 0 0 0
W3 row 11  0 0 0 0
W3 row 12  0 0 0 0
W3 row 13  0 0 0 0
W3 row 14  0 0 0 0
W3 row 15  0 0 0 0
W3 row 16  0 0 0 0
W3 row 17  0 0 0 0
W3 row 18  0 0 0 0
W3 row 19  0 0 0 0
W3 row 20  0 0 0 0
W3 row 21  0 0 0 0
W3 row 22  0 0 0 0
W3 row 23  0 0 0 0
W3 row 24  0 0 0 0
W3 row 25  0 0 0 0
W3 row 26  0 0 0 0
W3 row 27  0 0 0 0
W3 row 28  0 0 0 0
W3 row 29  0 0 0 0
W3 row 30  0 0 0 0
W3 row 31  0 0 0 0
W3 row 32  0 0 0 0
W3 row 33  0 0 0 0
W3 row 34  0 0 0 0
W3 row 35  0 0 0 0
W3 row 36  0 0 0 0
W3 row 37  0 0 0 0
W3 row 38  0 0 0 0
W3 row 39  0 0 0 0
W3 row 40  0 0 0 0
W3 row 41  0 0 0 0
W3 row 42  0 0 0 0
W3 row 43  0 0 0 0
W3 row 44  0 0 0 0
W3 row 45  0 0 0 0
W3 row 46  0 0 0 0
W3 row 47  0 0 0 0
W3 row 48  0 0 0 0
W3 row 49  0 0 0 0
W3 row 50  0 0 0 0
W3 row 51  0 0 0 0
W3 row 52  0 0 0 0
W3 row 53  0 0 0 0
W3 row 54  0 0 0 0
W3 row 55  0 0 0 0
W3 row 56  0 0 0 0
W3 row 57  0 0 0 0
W3 row 58  0 0 0 0
W3 row 59  0 0 0 0
W3 row 60  0 0 0 0
W3 row 61  0 0 0 0
W3 row 62  0 0 0 0
W3 row 63  0 0 0 0
b3  0.05 0 0 0
END_EDITS