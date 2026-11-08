CONTROLLER: Paddle tracking. d = RAM[99] (ball x) − RAM[72] (paddle x). If d > 8.5, RIGHT. If d < 1.5, LEFTFIRE. Otherwise FIRE, which also serves the ball. The 2nd layer passes the two signals through.

**Assumptions (unverified).** The RAM addresses are recalled from public Atari RAM annotations, not measured: byte 72 as paddle x and byte 99 as ball x. Nothing here was run or tested. If these addresses are wrong, the controller will not work.

**How the network works**

- **Layer 1** has two active units. Inputs are RAM/255, so a weight of ±510 turns each into "2 × (difference in RAM units)":
  - Unit 0 (go right): tanh(2·(RAM[99] − RAM[72] − 8.5)). It is about +1 when the ball is clearly right of the paddle's aim point, and about −1 otherwise.
  - Unit 1 (go left): tanh(2·(RAM[72] − RAM[99] + 1.5)). It is about +1 when the ball is clearly left of the aim point.
  - Between them is a dead band where the offset d = ball_x − paddle_x is 2..8. This centres the ball about 5 units from the paddle's reference coordinate. That offset was chosen to work whether RAM[72] is the paddle's left edge or its centre, and also after the paddle shrinks.
- **Layer 2** copies the two signals with gain 3, so they saturate to about ±1.
- **Output layer:**
  - RIGHT gets 2·h2_0.
  - LEFTFIRE gets 2·h2_1.
  - FIRE gets a constant bias of 0.5, so it beats NOOP (0) when neither movement signal is active.
  - Only one movement signal can be positive at a time.
- **Serving:**
  - When the paddle is aligned, it presses FIRE.
  - Moving left also holds fire, so a level-triggered serve happens even if the ball's x sits at a wall between lives.
  - Moving right releases fire, which creates button edges in case the serve is edge-triggered.
  - Pressing fire during play has no effect.

WEIGHTS:
W1[99,0] = 510
W1[72,0] = -510
b1[0] = -17
W1[72,1] = 510
W1[99,1] = -510
b1[1] = 3
W2[0,0] = 3
W2[1,1] = 3
W3[0,3] = 2
W3[1,12] = 2
b3[1] = 0.5
END_WEIGHTS