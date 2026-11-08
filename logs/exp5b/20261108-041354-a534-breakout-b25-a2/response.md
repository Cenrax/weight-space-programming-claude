The final saved weights are a hand-built "predict where the ball will land" controller. On its last 32-episode test it averaged **about 94 points** per game (median about 77, range 24–363). The rollout budget is fully used. My last few versions all scored between about 91 and 98, but the scores vary so much from game to game that they can't be told apart. Expect roughly 80–100 on the final 100 episodes, with a few rare very high games pulling the mean up.

CONTROLLER: Always fire. Move the paddle (using its raw input byte 70, position ≈ 235−RAM70+9) toward where the falling ball will land: x + (vx/vy)(185−y), mirrored off the walls at x=55 and x=200, with velocity classes read from bytes 103, 105 and 107. While the ball rises, follow its x.

**What the bytes mean.** These are my readings from the traces:
- 99 = ball x, 101 = ball y, 72 = paddle x, 57 = lives.
- 70 is the raw paddle input; the paddle follows it at about 235 − RAM[70], lagging by around 11 pixels.
- 103 = vertical direction and speed: 1–4 means falling, 252–255 means rising.
- 105 = horizontal direction: 1 and 2 mean left, 254 and 255 mean right, 0 means nearly vertical.
- 107 = 0 marks the slow ball right after a serve, before the first paddle hit.
- 108 drops from 8 to 6 when the paddle shrinks.

**How the network works.** It uses 63 of the 64 units in each hidden layer. All weights were set by hand; nothing was trained.
- **Layer 1 detectors:** 12 sharp step units detect which speed or direction value each velocity byte holds.
- **Layer 1 prediction units:** for each of 25 speed/direction combinations, 2 units compare the predicted landing point with the paddle and handle the bounce off a side wall.
- **Layer 2 cases:** pairs of units let only the matching combination through and switch the others off.
- **Output:** the result goes to RIGHTFIRE or LEFTFIRE. If nothing is active (ball not yet served), FIRE wins.
- **Aim point:** it moves by 1.5 × (RAM[108] − 8), to allow for the smaller paddle.

**What the rollouts changed:**
- **Following the ball x:** about 13 points. Moving the aim point 7 pixels made it about 27.
- **First prediction:** wrong horizontal speeds made it worse (about 20).
- **Measured speeds:** I read the slopes off the traces (horizontal/vertical ≈ 0.67 or 1.0 at normal speed, 1.0 right after a serve), which brought it to about 29.
- **Raw paddle input:** steering by byte 70 instead of the lagging paddle position 72 was the biggest single gain (about 46, then 62).
- **Speed-dependent slopes:** making the slope depend on the vertical speed value raised it to about 85.
- **Middling ideas:**
  - Treating the nearly vertical ball as not moving sideways gave about 91.
  - The slow-serve case and faster guesses for the higher speeds gave 94–98.
- **Tried and rejected:**
  - Aim points of 1, 13, 15 and 17 all did worse.
  - Blending in byte 72 dropped it to about 48.
  - A wall rule meant to stop the paddle pushing past the walls didn't help.

**Limits:**
- The speeds for the faster vertical values (3–4 px per frame) and for direction value 0 are guesses I couldn't check, because the traces only sample every 20–60 steps.
- My last planned test (aim point 11) hit the exhausted budget after its weights had been saved. I then re-saved the aim-point-9 version (94 points) and confirmed the saved weights match it.