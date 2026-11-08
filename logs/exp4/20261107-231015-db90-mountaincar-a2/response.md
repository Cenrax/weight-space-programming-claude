DIAGNOSIS: Output bias b3[2]=0.121 beats b3[0]=-0.100, and the hidden units barely react to velocity (|W1 row 1|≤0.46, |v|≤0.07). So logit 2 always wins: 200/200 right pushes, and the car only rocks between -0.47 and -0.30.

## Why it fails

**The trace.** The policy chose action 2 (push right) on every one of the 200 steps. A constant push of 0.001 is too weak against gravity (up to 0.0025) to climb the right hill. The car just oscillates around an equilibrium shifted right of the valley bottom:

- Position stays between about -0.47 and -0.30.
- Velocity never exceeds about ±0.0073.
- The amplitude never grows, because the push never changes direction with the motion. So no energy is pumped in.

**The weights.**

- **Weak velocity input.** Velocity is raw and tiny (|v| < 0.01 in practice). The velocity weights in W1 row 1 are at most about 0.46 in magnitude. Velocity therefore shifts the first-layer pre-activations by about 0.005 at most, so the hidden state is essentially a function of position only.
- **Output layer always favours "right".** W3 is organised as ±0.11 / ±0.14 pairs: each hidden unit pushes logit 0 and logit 2 in opposite directions. The resulting near-constant hidden pattern, plus the bias gap b3[2] − b3[0] = 0.221, keeps logit 2 on top everywhere along the trajectory.

## The fix

I replace the policy with the classic energy-pumping rule: **push in the direction of the current velocity.** Pushing with the motion adds energy every half-swing. This policy is known to reach the flag from all start states in [-0.6, -0.4] in roughly 85–125 steps, well within 200.

I implement it by routing one clean signal path through the network:

1. **h1 unit 0 = tanh(1000·v).** Set W1[1,0]=1000, W1[0,0]=0, b1[0]=0. This is a sharp sign detector for velocity; even v = 1e-5 gives about 0.01.
2. **h2 unit 0 = tanh(10·h1[0]).** Zero every other input into h2 unit 0 (W2[i,0]=0 for i≥1, b2[0]=0) and set W2[0,0]=10. The sign is preserved and amplified.
3. **Output.** Zero all of W3 except row 0, which becomes (-1, 0, +1). Set b3 = (0, -1, 0).
   - If v > 0, logit 2 is largest, so the car pushes right.
   - If v < 0, logit 0 is largest, so it pushes left.
   - "No-op" is never chosen, because b3[1] = -1.
   - At v = 0 (the reset state), logits 0 and 2 tie at 0, so argmax picks action 0. The car swings left first, which is a good opening from the starting region.
4. **Wall case.** If the car ever reaches the left wall, velocity is reset to 0 and the policy pushes left. Gravity there still yields positive velocity on the next step (-0.001 + 0.0022 > 0), so it rebounds rightward and keeps pumping.

EDITS:
W1[0,0] = 0.0
W1[1,0] = 1000.0
b1[0] = 0.0
W2[0,0] = 10.0
W2[1,0] = 0.0
W2[2,0] = 0.0
W2[3,0] = 0.0
W2[4,0] = 0.0
W2[5,0] = 0.0
W2[6,0] = 0.0
W2[7,0] = 0.0
W2[8,0] = 0.0
W2[9,0] = 0.0
W2[10,0] = 0.0
W2[11,0] = 0.0
W2[12,0] = 0.0
W2[13,0] = 0.0
W2[14,0] = 0.0
W2[15,0] = 0.0
W2[16,0] = 0.0
W2[17,0] = 0.0
W2[18,0] = 0.0
W2[19,0] = 0.0
W2[20,0] = 0.0
W2[21,0] = 0.0
W2[22,0] = 0.0
W2[23,0] = 0.0
W2[24,0] = 0.0
W2[25,0] = 0.0
W2[26,0] = 0.0
W2[27,0] = 0.0
W2[28,0] = 0.0
W2[29,0] = 0.0
W2[30,0] = 0.0
W2[31,0] = 0.0
W2[32,0] = 0.0
W2[33,0] = 0.0
W2[34,0] = 0.0
W2[35,0] = 0.0
W2[36,0] = 0.0
W2[37,0] = 0.0
W2[38,0] = 0.0
W2[39,0] = 0.0
W2[40,0] = 0.0
W2[41,0] = 0.0
W2[42,0] = 0.0
W2[43,0] = 0.0
W2[44,0] = 0.0
W2[45,0] = 0.0
W2[46,0] = 0.0
W2[47,0] = 0.0
W2[48,0] = 0.0
W2[49,0] = 0.0
W2[50,0] = 0.0
W2[51,0] = 0.0
W2[52,0] = 0.0
W2[53,0] = 0.0
W2[54,0] = 0.0
W2[55,0] = 0.0
W2[56,0] = 0.0
W2[57,0] = 0.0
W2[58,0] = 0.0
W2[59,0] = 0.0
W2[60,0] = 0.0
W2[61,0] = 0.0
W2[62,0] = 0.0
W2[63,0] = 0.0
b2[0] = 0.0
W3 row 0  -1.0 0.0 1.0
W3 row 1  0.0 0.0 0.0
W3 row 2  0.0 0.0 0.0
W3 row 3  0.0 0.0 0.0
W3 row 4  0.0 0.0 0.0
W3 row 5  0.0 0.0 0.0
W3 row 6  0.0 0.0 0.0
W3 row 7  0.0 0.0 0.0
W3 row 8  0.0 0.0 0.0
W3 row 9  0.0 0.0 0.0
W3 row 10  0.0 0.0 0.0
W3 row 11  0.0 0.0 0.0
W3 row 12  0.0 0.0 0.0
W3 row 13  0.0 0.0 0.0
W3 row 14  0.0 0.0 0.0
W3 row 15  0.0 0.0 0.0
W3 row 16  0.0 0.0 0.0
W3 row 17  0.0 0.0 0.0
W3 row 18  0.0 0.0 0.0
W3 row 19  0.0 0.0 0.0
W3 row 20  0.0 0.0 0.0
W3 row 21  0.0 0.0 0.0
W3 row 22  0.0 0.0 0.0
W3 row 23  0.0 0.0 0.0
W3 row 24  0.0 0.0 0.0
W3 row 25  0.0 0.0 0.0
W3 row 26  0.0 0.0 0.0
W3 row 27  0.0 0.0 0.0
W3 row 28  0.0 0.0 0.0
W3 row 29  0.0 0.0 0.0
W3 row 30  0.0 0.0 0.0
W3 row 31  0.0 0.0 0.0
W3 row 32  0.0 0.0 0.0
W3 row 33  0.0 0.0 0.0
W3 row 34  0.0 0.0 0.0
W3 row 35  0.0 0.0 0.0
W3 row 36  0.0 0.0 0.0
W3 row 37  0.0 0.0 0.0
W3 row 38  0.0 0.0 0.0
W3 row 39  0.0 0.0 0.0
W3 row 40  0.0 0.0 0.0
W3 row 41  0.0 0.0 0.0
W3 row 42  0.0 0.0 0.0
W3 row 43  0.0 0.0 0.0
W3 row 44  0.0 0.0 0.0
W3 row 45  0.0 0.0 0.0
W3 row 46  0.0 0.0 0.0
W3 row 47  0.0 0.0 0.0
W3 row 48  0.0 0.0 0.0
W3 row 49  0.0 0.0 0.0
W3 row 50  0.0 0.0 0.0
W3 row 51  0.0 0.0 0.0
W3 row 52  0.0 0.0 0.0
W3 row 53  0.0 0.0 0.0
W3 row 54  0.0 0.0 0.0
W3 row 55  0.0 0.0 0.0
W3 row 56  0.0 0.0 0.0
W3 row 57  0.0 0.0 0.0
W3 row 58  0.0 0.0 0.0
W3 row 59  0.0 0.0 0.0
W3 row 60  0.0 0.0 0.0
W3 row 61  0.0 0.0 0.0
W3 row 62  0.0 0.0 0.0
W3 row 63  0.0 0.0 0.0
b3  0.0 -1.0 0.0
END_EDITS