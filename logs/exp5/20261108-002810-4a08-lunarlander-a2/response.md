CONTROLLER: Neural copy of Gymnasium's heuristic lander. A = 0.5·softclip(0.5x+vx, ±0.4) − 0.5·angle − angVel; H = 0.275|x| − 0.5y − 0.5vy. Main engine if H > |A| and H > 0.05; else right engine if A < −0.05, left if A > 0.05. With leg contact: A = 0, main if −0.5vy > 0.05.

**How the network works**

The network reproduces the hand-written heuristic from `lunar_lander.py`:

- `angle_todo = (clip(0.5x + vx, ±0.4) − angle)·0.5 − angVel`
- `hover_todo = (0.55|x| − y)·0.5 − 0.5·vy`
- On leg contact, `angle_todo = 0` and `hover_todo = −0.5·vy`.

**Layer 1 computes the features.** Units with small input gain work in tanh's nearly linear range.
- **Unit 0, uT** = tanh(2.5·(0.5x + vx)). Then 0.4·uT is a soft version of the ±0.4 clip, and 0.5·target = 0.2·uT.
- **Unit 1, uR** = tanh(0.2·(−0.5·angle − angVel)). The term itself is recovered as 5·uR.
- **Unit 2, uL** = tanh(0.2·(−0.5y − 0.5vy)). The term is 5·uL.
- **Unit 3, uV** = tanh(0.2·vy). So −0.5vy = −2.5·uV.
- **Unit 4, uC** = tanh(10·(legL + legR) − 5). This is about −1 with no leg contact and +1 with any contact.
- **Units 5–24 build |x|.** They are tanh steps tanh(±10x − 10t) at thresholds t = 0.05, 0.15, …, 0.95 on both sides of zero. The sum of all steps approximates 0.275·|x| as 0.01375·Σu + 0.275. I use 0.265 for the constant to offset the smoothing near x = 0.
- So H ≈ 0.01375·Σu + 0.265 + 5·uL, and A ≈ 0.2·uT + 5·uR.

**Layer 2 turns comparisons into near-binary signals** (gain 100, so pre-activation = 100 × the expression):
- **p1** = sign(H − A); **p2** = sign(H + A); **p3** = sign(H − 0.05).
  - Each gets +3(1 + uC), which forces it true when a leg touches the ground.
- **p4** = sign(−0.5vy − 0.05) + 3(1 − uC).
  - This forces it true in flight, so it only matters after touchdown.
- **q1** = sign(A − 0.05) and **q3** = sign(−A − 0.05).
  - Each gets −3(1 + uC), which disables the side engines on contact.

**Output layer:**
- Main-engine logit = 10·(p1 + p2 + p3 + p4 − 3). This is +10 only when all four signals are true, otherwise −10 or lower.
- Left-engine logit = 5·q1; right-engine logit = 5·q3.
- The do-nothing logit is 0.
- So the main engine has priority, then the side engines, otherwise the lander coasts. This matches the heuristic's if/elif order.

WEIGHTS:
W1[0,0] = 1.25
W1[2,0] = 2.5
W1[4,1] = -0.1
W1[5,1] = -0.2
W1[1,2] = -0.1
W1[3,2] = -0.1
W1[3,3] = 0.2
W1[6,4] = 10
W1[7,4] = 10
b1[4] = -5
W1[0,5] = 10
W1[0,6] = 10
W1[0,7] = 10
W1[0,8] = 10
W1[0,9] = 10
W1[0,10] = 10
W1[0,11] = 10
W1[0,12] = 10
W1[0,13] = 10
W1[0,14] = 10
W1[0,15] = -10
W1[0,16] = -10
W1[0,17] = -10
W1[0,18] = -10
W1[0,19] = -10
W1[0,20] = -10
W1[0,21] = -10
W1[0,22] = -10
W1[0,23] = -10
W1[0,24] = -10
b1[5] = -0.5
b1[6] = -1.5
b1[7] = -2.5
b1[8] = -3.5
b1[9] = -4.5
b1[10] = -5.5
b1[11] = -6.5
b1[12] = -7.5
b1[13] = -8.5
b1[14] = -9.5
b1[15] = -0.5
b1[16] = -1.5
b1[17] = -2.5
b1[18] = -3.5
b1[19] = -4.5
b1[20] = -5.5
b1[21] = -6.5
b1[22] = -7.5
b1[23] = -8.5
b1[24] = -9.5
W2[5,0] = 1.375
W2[6,0] = 1.375
W2[7,0] = 1.375
W2[8,0] = 1.375
W2[9,0] = 1.375
W2[10,0] = 1.375
W2[11,0] = 1.375
W2[12,0] = 1.375
W2[13,0] = 1.375
W2[14,0] = 1.375
W2[15,0] = 1.375
W2[16,0] = 1.375
W2[17,0] = 1.375
W2[18,0] = 1.375
W2[19,0] = 1.375
W2[20,0] = 1.375
W2[21,0] = 1.375
W2[22,0] = 1.375
W2[23,0] = 1.375
W2[24,0] = 1.375
W2[2,0] = 500
W2[0,0] = -20
W2[1,0] = -500
W2[4,0] = 300
b2[0] = 326.5
W2[5,1] = 1.375
W2[6,1] = 1.375
W2[7,1] = 1.375
W2[8,1] = 1.375
W2[9,1] = 1.375
W2[10,1] = 1.375
W2[11,1] = 1.375
W2[12,1] = 1.375
W2[13,1] = 1.375
W2[14,1] = 1.375
W2[15,1] = 1.375
W2[16,1] = 1.375
W2[17,1] = 1.375
W2[18,1] = 1.375
W2[19,1] = 1.375
W2[20,1] = 1.375
W2[21,1] = 1.375
W2[22,1] = 1.375
W2[23,1] = 1.375
W2[24,1] = 1.375
W2[2,1] = 500
W2[0,1] = 20
W2[1,1] = 500
W2[4,1] = 300
b2[1] = 326.5
W2[5,2] = 1.375
W2[6,2] = 1.375
W2[7,2] = 1.375
W2[8,2] = 1.375
W2[9,2] = 1.375
W2[10,2] = 1.375
W2[11,2] = 1.375
W2[12,2] = 1.375
W2[13,2] = 1.375
W2[14,2] = 1.375
W2[15,2] = 1.375
W2[16,2] = 1.375
W2[17,2] = 1.375
W2[18,2] = 1.375
W2[19,2] = 1.375
W2[20,2] = 1.375
W2[21,2] = 1.375
W2[22,2] = 1.375
W2[23,2] = 1.375
W2[24,2] = 1.375
W2[2,2] = 500
W2[4,2] = 300
b2[2] = 321.5
W2[3,3] = -250
W2[4,3] = -300
b2[3] = 295
W2[0,4] = 20
W2[1,4] = 500
W2[4,4] = -300
b2[4] = -305
W2[0,5] = -20
W2[1,5] = -500
W2[4,5] = -300
b2[5] = -305
W3[0,2] = 10
W3[1,2] = 10
W3[2,2] = 10
W3[3,2] = 10
b3[2] = -30
W3[4,1] = 5
W3[5,3] = 5
END_WEIGHTS