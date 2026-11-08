The saved weights scored a mean of **278.5** over the last 32-episode test: all 32 landed, every episode scored at least 200, and returns ranged from 251 to 300. I used 4 of the 5 test runs and checked that the final saved weights are exactly the ones from that last test. Each run used new random episodes, so the 100-episode evaluation will vary somewhat around this number.

CONTROLLER: Gymnasium heuristic lander built into the net: A=1.0*(clip(0.5x+vx,±0.4)-angle)-angvel; H=0.5*(0.55|x|-y)-0.65vy. Logits [0.05, A, H, -A], i.e. main if H>max(|A|,0.05), else side by sign of A. On leg contact: sides off, H=-0.5vy.

**How the network does it:**
- **Layer 1 (inputs to first hidden layer):**
  - One unit computes the target tilt: 0.4·tanh((0.5x+vx)/0.4), which acts as a smooth clip to ±0.4.
  - Two units use a small input scale (0.1), so tanh behaves almost linearly. One computes the tilt correction from angle and angular velocity; the other computes the descent correction from height and vertical speed.
  - One unit computes −vy for use after touchdown.
  - One unit reads the legs: tanh(5·(left+right)) is about 1 when either leg touches the ground.
  - 20 steep tanh steps at |x| = 0.05, 0.15, …, 0.95 add up to approximately |x|.
- **Layer 2:** these combine into tilt units tanh(±A − 8·legs) and descent units tanh(H − 8·legs) and tanh(−0.65·vy·… − 8·(1−legs)) (the contact unit uses −0.5·vy). Ground contact therefore switches each signal on or off.
- **Output layer:** logits are [tanh(0.05), tanh(A), tanh(H), tanh(−A)]. Because A, H and the 0.05 threshold all pass through the same increasing function, picking the largest logit gives exactly the heuristic's choice.

**Changes made after each test run:**
1. **Exact heuristic copy, but doing nothing after touchdown:** mean 160, with 15 of 32 crashing. The traces showed the lander touching down too fast and tipping over, so I changed the contact behaviour to match the heuristic: no side engines, and fire the main engine if vy < −0.1.
2. **Contact behaviour fixed:** mean 236, 3 crashes. The lander was still reaching the ground with too much sideways speed because it tilted too slowly, so I doubled the tilt gain from 0.5 to 1.0.
3. **Faster tilt:** mean 265, 1 crash. That crash was a fast fall that wasn't braked in time, so I raised the vertical-speed gain from 0.5 to 0.65 to make it fire the main engine earlier and descend more slowly.
4. **Final settings:** mean 278.5, no crashes. I kept these weights.