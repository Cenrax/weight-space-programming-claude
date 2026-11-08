CONTROLLER: Gymnasium's LunarLander heuristic. A = 0.5·clip(0.5x+vx, ±0.4) − 0.5·angle − angvel. H = 0.275|x| − 0.5y − 0.5vy. If either leg touches the ground, A = 0 and H = −0.5vy. The action is argmax of the logits [0.05, A, H, −A].

## How the network works

**Turning the heuristic into an argmax.** The reference heuristic is: fire main if H > |A| and H > 0.05; otherwise fire right if A < −0.05; otherwise fire left if A > 0.05; otherwise do nothing. Taking the argmax of the logits [0.05, A, H, −A] gives exactly the same decisions in every case. So the network only has to output A and H. It never needs any explicit if/else logic.

**Layer 1: features, with leg-contact gating built in.** Let c = left leg contact + right leg contact. Gating uses pairs of units of the form tanh(z + 30c) and tanh(z − 30c). With no contact the pair sums to 2·tanh(z). With contact the two units saturate at +1 and −1, so the sum is exactly 0.

- **Units 0–1: the clipped angle target.** These compute the gated term tanh(2.5·(0.5x + vx)). Multiplied by 0.4, this is a smooth version of clip(0.5x + vx, ±0.4).
- **Units 2–3: rest of A.** These are a gated, near-linear version of (−0.5·angle − angvel), scaled by 0.05 so tanh stays in its linear range.
- **Units 4–5: height term.** These are a gated, near-linear version of −0.5y.
- **Unit 6: vertical speed.** This is −0.5vy, scaled by 0.05 and not gated, because it is still used after touchdown.
- **Units 7–26: |x|.** These build a smooth staircase approximation of |x|, using 10 even "bumps" tanh(20(x−d)) − tanh(20(x+d)) for d = 0.05, 0.15, …, 0.95.
  - These units are gated with ±60·c.
  - Under contact every unit saturates, so the |x| term plus its bias cancels to exactly 0.

**Layer 2: combining into A and H.**
- Unit 0 computes tanh(A). Its weights are 0.1·(u0+u1) + 10·(u2+u3).
- Unit 1 computes tanh(H). Its weights are 10·(u4+u5) + 20·u6 + 0.01375·Σa − 0.01375·Σb, plus a bias of 0.275.
- tanh is monotonic, so the comparisons H vs ±A are unchanged by it.

**Layer 3: logits.**
- The logits are [tanh(0.05), tanh(A), tanh(H), −tanh(A)].
- Using tanh(0.05) for the do-nothing logit keeps the 0.05 threshold exact after the tanh.

WEIGHTS:
W1[0,0] = 1.25
W1[2,0] = 2.5
W1[6,0] = 30
W1[7,0] = 30
W1[0,1] = 1.25
W1[2,1] = 2.5
W1[6,1] = -30
W1[7,1] = -30
W1[4,2] = -0.025
W1[5,2] = -0.05
W1[6,2] = 30
W1[7,2] = 30
W1[4,3] = -0.025
W1[5,3] = -0.05
W1[6,3] = -30
W1[7,3] = -30
W1[1,4] = -0.025
W1[6,4] = 30
W1[7,4] = 30
W1[1,5] = -0.025
W1[6,5] = -30
W1[7,5] = -30
W1[3,6] = -0.025
W1[0,7] = 20
W1[6,7] = -60
W1[7,7] = -60
b1[7] = -1
W1[0,8] = 20
W1[6,8] = -60
W1[7,8] = -60
b1[8] = -3
W1[0,9] = 20
W1[6,9] = -60
W1[7,9] = -60
b1[9] = -5
W1[0,10] = 20
W1[6,10] = -60
W1[7,10] = -60
b1[10] = -7
W1[0,11] = 20
W1[6,11] = -60
W1[7,11] = -60
b1[11] = -9
W1[0,12] = 20
W1[6,12] = -60
W1[7,12] = -60
b1[12] = -11
W1[0,13] = 20
W1[6,13] = -60
W1[7,13] = -60
b1[13] = -13
W1[0,14] = 20
W1[6,14] = -60
W1[7,14] = -60
b1[14] = -15
W1[0,15] = 20
W1[6,15] = -60
W1[7,15] = -60
b1[15] = -17
W1[0,16] = 20
W1[6,16] = -60
W1[7,16] = -60
b1[16] = -19
W1[0,17] = 20
W1[6,17] = 60
W1[7,17] = 60
b1[17] = 1
W1[0,18] = 20
W1[6,18] = 60
W1[7,18] = 60
b1[18] = 3
W1[0,19] = 20
W1[6,19] = 60
W1[7,19] = 60
b1[19] = 5
W1[0,20] = 20
W1[6,20] = 60
W1[7,20] = 60
b1[20] = 7
W1[0,21] = 20
W1[6,21] = 60
W1[7,21] = 60
b1[21] = 9
W1[0,22] = 20
W1[6,22] = 60
W1[7,22] = 60
b1[22] = 11
W1[0,23] = 20
W1[6,23] = 60
W1[7,23] = 60
b1[23] = 13
W1[0,24] = 20
W1[6,24] = 60
W1[7,24] = 60
b1[24] = 15
W1[0,25] = 20
W1[6,25] = 60
W1[7,25] = 60
b1[25] = 17
W1[0,26] = 20
W1[6,26] = 60
W1[7,26] = 60
b1[26] = 19
W2[0,0] = 0.1
W2[1,0] = 0.1
W2[2,0] = 10
W2[3,0] = 10
W2[4,1] = 10
W2[5,1] = 10
W2[6,1] = 20
W2[7,1] = 0.01375
W2[8,1] = 0.01375
W2[9,1] = 0.01375
W2[10,1] = 0.01375
W2[11,1] = 0.01375
W2[12,1] = 0.01375
W2[13,1] = 0.01375
W2[14,1] = 0.01375
W2[15,1] = 0.01375
W2[16,1] = 0.01375
W2[17,1] = -0.01375
W2[18,1] = -0.01375
W2[19,1] = -0.01375
W2[20,1] = -0.01375
W2[21,1] = -0.01375
W2[22,1] = -0.01375
W2[23,1] = -0.01375
W2[24,1] = -0.01375
W2[25,1] = -0.01375
W2[26,1] = -0.01375
b2[1] = 0.275
W3[0,1] = 1
W3[0,3] = -1
W3[1,2] = 1
b3[0] = 0.049958
END_WEIGHTS