CONTROLLER: The Gymnasium heuristic lander in network form. angle_todo=0.5(clip(0.5x+vx,±0.4)−θ)−ω and hover_todo=0.5(0.55|x|−y)−0.5vy. Fire main if hover_todo>|angle_todo| and >0.05; else right if angle_todo<−0.05; else left if >0.05. On leg contact: main only if vy<−0.1.

**How the network works**

**Layer 1 (feature units).** Unused units stay 0.
- **h0** = tanh(1.25x + 2.5vx). So 0.4·h0 is a smooth version of the clipped angle target clip(0.5x+vx, ±0.4).
- **h1** = tanh(−0.125θ − 0.25ω). This is about 0.25·(−0.5θ − ω), so it acts as a linear unit.
- **h2** = tanh(−0.1y − 0.1vy). This is about 0.2·(−0.5y − 0.5vy).
- **h3** = tanh(10·legL + 10·legR − 5). This is the contact gate: −1 in flight, +1 once any leg touches.
- **h4** = tanh(−0.1vy). This is about 0.2·(−0.5vy), used for the landed rule.
- **h5–h24** approximate |x| with a staircase of sigmoids. Each unit is tanh(±10x − 10t) with thresholds t = 0.05, 0.15, …, 0.95, one set for each sign of x. The sum S of these 20 units gives |x| ≈ 1 + 0.05·S.

From these features:
- angle_todo: A ≈ 0.2·h0 + 4·h1
- hover_todo: H ≈ 5·h2 + 0.275 + 0.01375·S
- landed hover term: H_L ≈ 5·h4

**Layer 2 (sharp comparators).** Gain K = 40. A gate term of ±300·g, with g = (h3+1)/2, forces each unit to a fixed value in the wrong mode.
- **v0** = step(H − A), forced to +1 when landed.
- **v1** = step(H + A), forced to +1 when landed.
- **v2** = step(H − 0.05), forced to −1 when landed.
- **v3** = step(H_L − 0.05), forced to −1 in flight.
- **v4** = step(A − 0.05), the left-engine condition, forced off when landed.
- **v5** = step(−A − 0.05), the right-engine condition, forced off when landed.

**Output layer.**
- **Main engine:** logit = 10·(v0 + v1 + v2 + v3 − 1.5).
  - In flight this is +5 only when H > |A| and H > 0.05; otherwise it is −15 or lower.
  - When landed it is +5 only when vy < −0.1.
- **Left engine:** logit = v4.
- **Right engine:** logit = v5.
- **Do nothing:** logit = 0. It wins when no engine logit is positive.

This reproduces the reference heuristic's priority order: main engine first, then right, then left, else nothing.

WEIGHTS:
W1[0,0] = 1.25
W1[2,0] = 2.5
W1[4,1] = -0.125
W1[5,1] = -0.25
W1[1,2] = -0.1
W1[3,2] = -0.1
W1[6,3] = 10
W1[7,3] = 10
b1[3] = -5
W1[3,4] = -0.1
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
W2[2,0] = 200
W2[0,0] = -8
W2[1,0] = -160
W2[3,0] = 150
b2[0] = 161
W2[2,1] = 200
W2[0,1] = 8
W2[1,1] = 160
W2[3,1] = 150
b2[1] = 161
W2[2,2] = 200
W2[3,2] = -150
b2[2] = -141
W2[4,3] = 200
W2[3,3] = 150
b2[3] = -152
W2[0,4] = 8
W2[1,4] = 160
W2[3,4] = -150
b2[4] = -152
W2[0,5] = -8
W2[1,5] = -160
W2[3,5] = -150
b2[5] = -152
W2[5,0] = 0.55
W2[5,1] = 0.55
W2[5,2] = 0.55
W2[6,0] = 0.55
W2[6,1] = 0.55
W2[6,2] = 0.55
W2[7,0] = 0.55
W2[7,1] = 0.55
W2[7,2] = 0.55
W2[8,0] = 0.55
W2[8,1] = 0.55
W2[8,2] = 0.55
W2[9,0] = 0.55
W2[9,1] = 0.55
W2[9,2] = 0.55
W2[10,0] = 0.55
W2[10,1] = 0.55
W2[10,2] = 0.55
W2[11,0] = 0.55
W2[11,1] = 0.55
W2[11,2] = 0.55
W2[12,0] = 0.55
W2[12,1] = 0.55
W2[12,2] = 0.55
W2[13,0] = 0.55
W2[13,1] = 0.55
W2[13,2] = 0.55
W2[14,0] = 0.55
W2[14,1] = 0.55
W2[14,2] = 0.55
W2[15,0] = 0.55
W2[15,1] = 0.55
W2[15,2] = 0.55
W2[16,0] = 0.55
W2[16,1] = 0.55
W2[16,2] = 0.55
W2[17,0] = 0.55
W2[17,1] = 0.55
W2[17,2] = 0.55
W2[18,0] = 0.55
W2[18,1] = 0.55
W2[18,2] = 0.55
W2[19,0] = 0.55
W2[19,1] = 0.55
W2[19,2] = 0.55
W2[20,0] = 0.55
W2[20,1] = 0.55
W2[20,2] = 0.55
W2[21,0] = 0.55
W2[21,1] = 0.55
W2[21,2] = 0.55
W2[22,0] = 0.55
W2[22,1] = 0.55
W2[22,2] = 0.55
W2[23,0] = 0.55
W2[23,1] = 0.55
W2[23,2] = 0.55
W2[24,0] = 0.55
W2[24,1] = 0.55
W2[24,2] = 0.55
W3[0,2] = 10
W3[1,2] = 10
W3[2,2] = 10
W3[3,2] = 10
W3[4,1] = 1
W3[5,3] = 1
b3[2] = -15
END_WEIGHTS