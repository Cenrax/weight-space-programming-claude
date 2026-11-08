CONTROLLER: Gymnasium's built-in LunarLander heuristic. Tilt target = clip(0.5x+vx, ±0.4); A=(target−angle)/2−angvel; H=(0.55|x|−y)/2−vy/2. After leg contact A=0, H=−vy/2. Fire main if H>max(|A|,0.05), else right if A<−0.05, else left if A>0.05, else do nothing.

## How the network computes it

I have not run this network. I worked it out by hand, so it should reproduce the heuristic's behaviour, but I have not measured its score.

**Layer 1 (units 0–24; the rest stay 0)**

| Unit | Formula | What it gives |
|---|---|---|
| 0 | tanh(1.25x + 2.5vx) | 0.4·h0 is a smooth version of clip(0.5x+vx, ±0.4) |
| 1 | tanh(0.02·(−0.5·angle − angvel)) | Near-linear, so 50·h1 equals the linear part of A |
| 2 | tanh(0.02·(−0.5y − 0.5vy)) | 50·h2 equals the linear part of H |
| 3 | tanh(0.02·(−0.5vy)) | 50·h3 equals the after-contact value of H |
| 4 | tanh(20·(legL + legR) − 10) | About +1 if any leg touches, otherwise −1 |
| 5–14 | tanh(20·(x − t)) | Step functions, thresholds t = 0.05, 0.15, …, 0.95 |
| 15–24 | tanh(−20·(x + t)) | The same steps mirrored for negative x |

Units 5–24 together give |x| ≈ 1 + 0.05·Σ h5…h24. The approximation is off by about 0.02 at x = 0 and is close elsewhere.

**Layer 2 (exact decision quantities, scale 1)**

The quantities used are:
- A = 0.2·h0 + 50·h1
- H = 0.275 + 0.01375·Σ h5…h24 + 50·h2
- Hc = 50·h3
- The contact flag c = (1 + h4)/2, which is 0 or 1.

The four layer-2 units are:
- u1 = tanh(A − 20c)
- u3 = tanh(−A − 20c)
- vfree = tanh(H − 20c)
- vc = tanh(Hc − 20(1 − c))

Whichever branch is inactive is pushed to exactly −1.

**Output logits**
- L0 = tanh(0.05)
- L1 = u1
- L2 = vfree + vc + 1
- L3 = u3

tanh is monotone and odd, so argmax reproduces the heuristic's comparisons exactly:
- Main engine fires when tanh(H) > max(tanh|A|, tanh 0.05).
- Otherwise the side engine matching the sign of A fires once |A| > 0.05.
- Otherwise the lander does nothing.

After leg contact, both side-engine logits sit at about −1. The main engine then fires only when −vy/2 > 0.05.

WEIGHTS:
W1[0,0] = 1.25
W1[2,0] = 2.5
W1[4,1] = -0.01
W1[5,1] = -0.02
W1[1,2] = -0.01
W1[3,2] = -0.01
W1[3,3] = -0.01
W1[6,4] = 20
W1[7,4] = 20
b1[4] = -10
W1[0,5] = 20
W1[0,6] = 20
W1[0,7] = 20
W1[0,8] = 20
W1[0,9] = 20
W1[0,10] = 20
W1[0,11] = 20
W1[0,12] = 20
W1[0,13] = 20
W1[0,14] = 20
b1[5] = -1
b1[6] = -3
b1[7] = -5
b1[8] = -7
b1[9] = -9
b1[10] = -11
b1[11] = -13
b1[12] = -15
b1[13] = -17
b1[14] = -19
W1[0,15] = -20
W1[0,16] = -20
W1[0,17] = -20
W1[0,18] = -20
W1[0,19] = -20
W1[0,20] = -20
W1[0,21] = -20
W1[0,22] = -20
W1[0,23] = -20
W1[0,24] = -20
b1[15] = -1
b1[16] = -3
b1[17] = -5
b1[18] = -7
b1[19] = -9
b1[20] = -11
b1[21] = -13
b1[22] = -15
b1[23] = -17
b1[24] = -19
W2[0,0] = 0.2
W2[1,0] = 50
W2[4,0] = -10
b2[0] = -10
W2[0,1] = -0.2
W2[1,1] = -50
W2[4,1] = -10
b2[1] = -10
W2[2,2] = 50
W2[4,2] = -10
b2[2] = -9.725
W2[5,2] = 0.01375
W2[6,2] = 0.01375
W2[7,2] = 0.01375
W2[8,2] = 0.01375
W2[9,2] = 0.01375
W2[10,2] = 0.01375
W2[11,2] = 0.01375
W2[12,2] = 0.01375
W2[13,2] = 0.01375
W2[14,2] = 0.01375
W2[15,2] = 0.01375
W2[16,2] = 0.01375
W2[17,2] = 0.01375
W2[18,2] = 0.01375
W2[19,2] = 0.01375
W2[20,2] = 0.01375
W2[21,2] = 0.01375
W2[22,2] = 0.01375
W2[23,2] = 0.01375
W2[24,2] = 0.01375
W2[3,3] = 50
W2[4,3] = 10
b2[3] = -10
W3[0,1] = 1
W3[1,3] = 1
W3[2,2] = 1
W3[3,2] = 1
b3 0.0499584 0 1 0
END_WEIGHTS