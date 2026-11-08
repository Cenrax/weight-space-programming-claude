CONTROLLER: Gymnasium's heuristic lander. angle_targ=clip(0.5x+vx,±0.4); hover_targ=0.55|x|; angle_todo=0.5(angle_targ-θ)-ω; hover_todo=0.5(hover_targ-y)-0.5vy. Main engine if hover_todo>|angle_todo| and >0.05; else right if angle_todo<-0.05; else left if >0.05. On leg contact: main only if vy<-0.1.

**How the network implements it**

The network has three stages: layer 1 builds the features, layer 2 makes the threshold decisions, and layer 3 combines them. I reasoned this out without running it, so the weights are untested.

**Layer 1 (feature units)**

Most units use small gains so that tanh stays nearly linear. A few use large gains on purpose, to produce a clip, a step, or a gate.

- **h0** = tanh(2.5·(0.5x + vx)). Then 0.4·h0 is a soft version of the clipped angle target.
- **h1** = tanh(0.2·(−0.5θ − ω)). Then 5·h1 ≈ −0.5θ − ω, the linear part of angle_todo.
- **h2** = tanh(0.2·(−0.5y − 0.5vy)). Then 5·h2 ≈ the linear part of hover_todo.
- **h3** = tanh(5·(legL + legR) − 2.5). This is the contact flag: about −0.987 with no contact, about +0.987 or more with contact.
- **h4** = tanh(−0.1·vy). Then 5·h4 ≈ −0.5vy, which is hover_todo while touching the ground.
- **h5–h16** are 12 steep step units, tanh(±15x − 15b) with b = 0.05, 0.15, …, 0.55. Together they build a staircase for |x|:
  - |x| ≈ 0.6 + 0.05·Σh.
  - So the 0.275|x| term in hover_todo becomes 0.01375·Σh + 0.165.

**Layer 2 (threshold units)**

Each unit is tanh(100·condition). The no-contact units also get −150·(h3 + 0.9866), which forces them to −1 whenever a leg touches the ground.

- **v0**: hover_todo − angle_todo > 0
- **v1**: hover_todo + angle_todo > 0
- **v2**: hover_todo > 0.05
- **v3**: angle_todo < −0.05
- **v4**: angle_todo > 0.05
- **v5**: contact AND −0.5vy > 0.05 (gated the opposite way, +150·(h3 − 0.9866))

Here angle_todo ≈ 0.2·h0 + 5·h1, and hover_todo ≈ 0.01375·Σh + 0.165 + 5·h2.

**Layer 3 (argmax logits)**

- logit0 (do nothing) = 1
- logit1 (left engine) = 3·v4 + 3
- logit3 (right engine) = 3·v3 + 3
- logit2 (main engine) = 10·(v0 + v1 + v2) + 30·v5 + 10

The main-engine logit equals 10 only when v0, v1 and v2 are all true (in flight) or v5 is true (on the ground). Otherwise it is −10 or less. Because 10 beats the side-engine value of 6, and 6 beats the do-nothing value of 1, argmax reproduces the heuristic's priority order: main, then right, then left, then nothing. After touchdown the side engines are disabled, so the lander settles and goes to sleep, which earns the +100 bonus.

WEIGHTS:
W1[0,0] = 1.25
W1[2,0] = 2.5
W1[4,1] = -0.1
W1[5,1] = -0.2
W1[1,2] = -0.1
W1[3,2] = -0.1
W1[6,3] = 5.0
W1[7,3] = 5.0
b1[3] = -2.5
W1[3,4] = -0.1
W1[0,5] = 15.0
W1[0,6] = 15.0
W1[0,7] = 15.0
W1[0,8] = 15.0
W1[0,9] = 15.0
W1[0,10] = 15.0
W1[0,11] = -15.0
W1[0,12] = -15.0
W1[0,13] = -15.0
W1[0,14] = -15.0
W1[0,15] = -15.0
W1[0,16] = -15.0
b1[5] = -0.75
b1[6] = -2.25
b1[7] = -3.75
b1[8] = -5.25
b1[9] = -6.75
b1[10] = -8.25
b1[11] = -0.75
b1[12] = -2.25
b1[13] = -3.75
b1[14] = -5.25
b1[15] = -6.75
b1[16] = -8.25
W2[0,0] = -20.0
W2[1,0] = -500.0
W2[2,0] = 500.0
W2[3,0] = -150.0
b2[0] = -131.49
W2[0,1] = 20.0
W2[1,1] = 500.0
W2[2,1] = 500.0
W2[3,1] = -150.0
b2[1] = -131.49
W2[2,2] = 500.0
W2[3,2] = -150.0
b2[2] = -136.49
W2[0,3] = -20.0
W2[1,3] = -500.0
W2[3,3] = -150.0
b2[3] = -152.99
W2[0,4] = 20.0
W2[1,4] = 500.0
W2[3,4] = -150.0
b2[4] = -152.99
W2[4,5] = 500.0
W2[3,5] = 150.0
b2[5] = -152.99
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
W3[0,2] = 10.0
W3[1,2] = 10.0
W3[2,2] = 10.0
W3[5,2] = 30.0
W3[3,3] = 3.0
W3[4,1] = 3.0
b3[0] = 1.0
b3[1] = 3.0
b3[2] = 10.0
b3[3] = 3.0
END_WEIGHTS