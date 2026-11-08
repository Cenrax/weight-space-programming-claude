DIAGNOSIS: The policy has learned to hover, not land. Its main-engine logit wins whenever vy dips slightly below 0, so it holds y≈1 and drifts in x until the 1000-step limit. It never commits to a descent and never touches down.

**Why it fails**

- **What the trace shows.** After about step 40 the lander sits at y≈1.17, then creeps down to y≈0.97 over 960 steps. vy stays at roughly ±0.03 the whole time, and the legs never touch.
- **Action mix.** The policy chose main 577 times, a side engine 422 times, and "do nothing" only once. That is the signature of a hover: firing the main engine about 58% of the time just cancels gravity.
- **Return.** The fuel and side-engine costs, plus jitter in the shaping reward, add up to -81. Because the episode is truncated, there is no +100 landing bonus.
- **Which weights cause it.** The vy input (W1 row 3) has many large-magnitude weights, for example 0.921, -0.697, 0.628, -0.638, 0.604, 0.576. These hidden units feed W3 column 2 (main engine), which also has a positive bias b3[2]=+0.10. Even a tiny downward vy makes main the argmax. Nothing in the network trades height against descent speed, so height is never brought toward 0.
- **Side engines.** The side-engine units (W3 columns 1 and 3) mostly react to small angle and angular-velocity noise. That causes the constant 1/3 flipping in the trace.

**The fix: the Gymnasium reference heuristic, built exactly into the MLP**

- Angle target: angle_targ = clip(0.5·x + vx, ±0.4).
- Hover target: hover_targ = 0.55·|x|.
- Angle error: A = 0.5·(angle_targ − angle) − angvel.
- Hover error: H = 0.5·(hover_targ − y) − 0.5·vy.
- Decision: fire main if H > |A| and H > 0.05; otherwise fire right if A < −0.05, left if A > 0.05; otherwise do nothing.
- With leg contact, side engines are disabled.

**Layer 1** (units 20–63 are zeroed, so they output 0):

- Units 0–3 are near-linear copies of the inputs: tanh(0.1·y), tanh(0.1·vy), tanh(0.1·angle), tanh(0.1·angvel).
- Unit 4 is tanh(1.25x + 2.5vx). This equals the angle target divided by 0.4, and tanh acts as a soft version of the clip.
- Unit 5 is a leg-contact gate: tanh(3·legL + 3·legR − 1.5).
- Units 6–19 are smooth steps at |x| > 0.1, 0.2, …, 0.7, built as tanh(±10x − 10t). Their sum gives |x| ≈ 0.73 + 0.05·Σu.

**Layer 2:**

- h2_0 = tanh(A) = tanh(0.2·u4 − 5·u2 − 10·u3).
- h2_1 = tanh(H) = tanh(0.2008 + 0.01375·Σsteps − 5·u0 − 5·u1).
- h2_2 = tanh(3·u5), which is about −0.99 with no legs touching and about +0.99 with contact.

**Output layer:**

- The logits are l0 = 0, l1 = tanh(A) − 0.05, l2 = tanh(H) − 0.05, l3 = −tanh(A) − 0.05.
- tanh is monotone, so the argmax reproduces the heuristic's decision rule exactly.
- On leg contact, the gate subtracts about 1 from both side logits, so the side engines stop firing.
- On leg contact, the gate also raises the main-engine threshold by about 0.06. This lets the lander come to rest even near the pad edge, so the body can go to sleep.
- All other W3 rows are zeroed.

EDITS:
W1 row 0  0 0 0 0 1.25 0 10 10 10 10 10 10 10 -10 -10 -10 -10 -10 -10 -10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 1  0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 2  0 0 0 0 2.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 3  0 0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 4  0 0 0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 5  0 0 0 0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 6  0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W1 row 7  0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
b1  0 0 0 0 0 -1.5 -1 -2 -3 -4 -5 -6 -7 -1 -2 -3 -4 -5 -6 -7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
W2[0,0] = 0
W2[0,1] = -5
W2[0,2] = 0
W2[1,0] = 0
W2[1,1] = -5
W2[1,2] = 0
W2[2,0] = -5
W2[2,1] = 0
W2[2,2] = 0
W2[3,0] = -10
W2[3,1] = 0
W2[3,2] = 0
W2[4,0] = 0.2
W2[4,1] = 0
W2[4,2] = 0
W2[5,0] = 0
W2[5,1] = 0
W2[5,2] = 3
W2[6,0] = 0
W2[6,1] = 0.01375
W2[6,2] = 0
W2[7,0] = 0
W2[7,1] = 0.01375
W2[7,2] = 0
W2[8,0] = 0
W2[8,1] = 0.01375
W2[8,2] = 0
W2[9,0] = 0
W2[9,1] = 0.01375
W2[9,2] = 0
W2[10,0] = 0
W2[10,1] = 0.01375
W2[10,2] = 0
W2[11,0] = 0
W2[11,1] = 0.01375
W2[11,2] = 0
W2[12,0] = 0
W2[12,1] = 0.01375
W2[12,2] = 0
W2[13,0] = 0
W2[13,1] = 0.01375
W2[13,2] = 0
W2[14,0] = 0
W2[14,1] = 0.01375
W2[14,2] = 0
W2[15,0] = 0
W2[15,1] = 0.01375
W2[15,2] = 0
W2[16,0] = 0
W2[16,1] = 0.01375
W2[16,2] = 0
W2[17,0] = 0
W2[17,1] = 0.01375
W2[17,2] = 0
W2[18,0] = 0
W2[18,1] = 0.01375
W2[18,2] = 0
W2[19,0] = 0
W2[19,1] = 0.01375
W2[19,2] = 0
b2[0] = 0
b2[1] = 0.2008
b2[2] = 0
W3 row 0  0 1 0 -1
W3 row 1  0 0 1 0
W3 row 2  0 -0.5 -0.03 -0.5
W3 row 3  0 0 0 0
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
b3  0 -0.5457 -0.0797 -0.5457
END_EDITS