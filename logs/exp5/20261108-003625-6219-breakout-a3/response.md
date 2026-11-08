CONTROLLER: Paddle tracks the ball. Choose RIGHT if ballX(RAM99) − padX(RAM72) > 7.5, else LEFTFIRE. If the ball is absent (RAM99<20 or RAM101<3), the paddle chatters around x≈110 so that FIRE presses repeat and the ball gets served.

**How the network works**

This design is untested. It relies on the published AtariARI RAM annotations for Breakout: RAM[99] is ball x, RAM[101] is ball y and RAM[72] is paddle x. The target offset of 7.5 and the "ball absent" test both assume how those bytes behave; neither has been checked against the game.

*Layer 1 (four used units; RAM values in raw units, x = RAM/255):*
- **u1** = tanh(2·(ballX − padX − 7.5)). It is +1 when the ball is right of the paddle centre, assuming a paddle about 16 px wide. Otherwise it is −1.
- **u2** = tanh(2·(110.5 − padX)). It is +1 when the paddle is left of screen centre.
- **u3a** = tanh(20 − ballX). It is +1 only if ballX is near 0, which I take to mean the ball is absent.
- **u3b** = tanh(4·(2.5 − ballY)). It is +1 only if ballY is near 0, also taken to mean the ball is absent.

*Layer 2 (gating):*
- **v1a** = tanh(5·u1 − 10·g) and **v1b** = tanh(5·u1 + 10·g), where g = (u3a+1) + (u3b+1).
  - g is 0 while the ball is in play, so both units equal sign(u1).
  - g is at least 2 when the ball is absent, so v1a = −1 and v1b = +1, and they cancel.
- **v2** = tanh(5·u2), the "go to centre" term.

*Output:*
- The RIGHT logit (action 3) is v1a + v1b + v2. The LEFTFIRE logit (action 12) is 0.
- Every other action has a bias of −10, so it never wins.
- **Ball in play:** RIGHT − LEFTFIRE = 2·sign(u1) ± 1, which follows the ball.
- **Ball absent:** the difference is sign(u2), so the paddle oscillates around the screen centre.

**Why FIRE is only on the left move**

Using a binary choice with fire on only one side means the paddle constantly alternates between RIGHT and LEFTFIRE near its target. That alternation produces repeated fire-button press edges. The ball should therefore be served after every lost life, whether the game reads the button by level or by edge. Pressing fire during play has no effect in Breakout.

WEIGHTS:
W1[99,0] = 510
W1[72,0] = -510
b1[0] = -15
W1[72,1] = -510
b1[1] = 221
W1[99,2] = -255
b1[2] = 20
W1[101,3] = -1020
b1[3] = 10
W2[0,0] = 5
W2[2,0] = -10
W2[3,0] = -10
b2[0] = -20
W2[0,1] = 5
W2[2,1] = 10
W2[3,1] = 10
b2[1] = 20
W2[1,2] = 5
W3[0,3] = 1
W3[1,3] = 1
W3[2,3] = 1
b3 -10 -10 -10 0 -10 -10 -10 -10 -10 -10 -10 -10 0 -10 -10 -10 -10 -10
END_WEIGHTS