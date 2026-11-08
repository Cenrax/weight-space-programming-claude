CONTROLLER: Energy pumping: push right iff v > g(x). g≈0 except: g≈-0.002 on [-0.66,-0.485], so starts left of -0.485 push right; g≈+0.024 on [-0.47,-0.25], so slow rightward motion there brakes and turns back. Each start does one left swing then a right run. Never coasts.

**Idea**

- **Base policy.** Accelerate in the direction of the velocity. This pumps energy at the maximum rate.
- **Energy needed.** The potential energy is s(x) = (0.0025/3)·sin(3x). Pushing adds 0.001 per unit of distance travelled.
- **Starting left first.** Start at rest at x0, push left to the turning point xL, then push right to the flag. The energy at the flag is s(x0) + 0.001·(x0 + 0.5 − 2·xL).
  - This must exceed s(0.5) ≈ 0.00083.
  - By my energy estimate, that holds only when x0 is greater than about −0.50.
  - So for x0 > −0.485 the car goes left first. It reaches the flag after one swing, an estimated ~85–95 steps.
- **Starting further left (x0 < −0.485).** A full first swing to the right would waste a whole extra swing. Instead the car pushes right only briefly:
  - It is braked (pushed left) as soon as it passes about −0.47, so it turns around at about −0.48 to −0.38.
  - It then does the same single left swing and final right run.
  - I estimate this saves roughly 10–20 steps over a full right swing.

**Switching curve**

Push right iff v > g(x), where:
- g ≈ −0.002 on [−0.66, −0.485]. At rest the car then pushes right. On the way back, the car is always faster than 0.002 in magnitude, so it cannot get stuck.
- g ≈ +0.002 just right of −0.485. At rest the car then pushes left.
- g ≈ +0.024 on [−0.47, −0.25]. This makes the slow start push brake. The final run passes this region at v ≳ 0.035, so it is unaffected.
- g ≈ 0 elsewhere, which gives normal energy pumping. Turnarounds at the far left and on the right hill are not disturbed.

**Network**

Layer 1 units:
- h0 = tanh(10v), which is approximately 10v.
- h1 = tanh(200x + 97), a step at −0.485.
- h2 = tanh(250x + 118.25), a step at −0.473.
- h3 = tanh(50x + 12.5), a step at −0.25.
- h4 = tanh(100x + 66), a step at −0.66.

Layer 2:
- D = h0 − 0.02·h1 − 0.11·h2 + 0.12·h3 + 0.01·h4, which equals 10·(v − g(x)).
- Unit 0 computes tanh(100·D).

Output:
- The right logit is +tanh(100·D) and the left logit is −tanh(100·D).
- The no-op logit is fixed at −5, so it is never chosen.
- The action is therefore right iff D > 0.

WEIGHTS:
W1[1,0] = 10
W1[0,1] = 200
b1[1] = 97
W1[0,2] = 250
b1[2] = 118.25
W1[0,3] = 50
b1[3] = 12.5
W1[0,4] = 100
b1[4] = 66
W2[0,0] = 100
W2[1,0] = -2
W2[2,0] = -11
W2[3,0] = 12
W2[4,0] = 1
W3[0,0] = -1
W3[0,2] = 1
b3[1] = -5
END_WEIGHTS