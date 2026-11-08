I'm setting up the controller around the known RAM addresses: the paddle position at 72, and the ball's x and y at 99 and 101, so the next step is deciding the movement rule.

I'm settling the control scheme: since FIRE is the only way to serve and always moving would leave the ball unserved, I'll use RIGHTFIRE, LEFTFIRE, and FIRE together. Next I'm trying to pin down the paddle and ball x ranges from RAM 72 and 99, and I'm not certain of the exact bounds, so I'm weighing whether both use left-edge coordinates.

I'm settling the target offset for the paddle relative to the ball, aiming for a paddle position roughly six or seven pixels left of the ball's x, since I'm not certain of the exact offset. I'll also keep in mind that when the ball isn't in play its coordinates may read zero, which would just nudge the paddle left while FIRE still serves. Simple tracking should be enough for now, and I'm starting to sketch how the controller output maps the normalized difference between ball and paddle x through a tanh.

I'm now working out the network layout: one unit tracks whether the ball is to the right of the paddle, and another tracks whether it's to the left, each with a small dead zone around the target offset. The output logits will map those two signals to RIGHTFIRE and LEFTFIRE, with FIRE as a baseline.

I'm working out the paddle-to-ball offset as a scaled difference, and I'm considering a steeper weight so that one pixel of difference maps to a full unit of signal.

I'm checking the two steering units: one fires when the paddle lags the ball by more than about eight pixels, the other when it runs more than about five pixels ahead, leaving a small dead zone where it stays put. The second layer then turns those two signals into logits for RIGHTFIRE and LEFTFIRE, with FIRE as the baseline.

I'm now trying to recall the paddle and ball x-offset relationship, and I'm unsure whether the paddle's left edge is measured against the ball's left edge the way I assumed. Since I can't pin down the exact values from memory, I'll leave the offset as an open question for now.

I'm still weighing whether the paddle-to-ball offset of 7 holds, since I can't confirm the exact coordinate convention from memory. If the offset is off by much, the tracking would miss badly, but the RAM addresses 72, 99, and 101 are the ones I'm fairly confident about. I'm also checking whether a narrower paddle would shift the center offset enough to matter for the dead zone.

Working out the weights now: the paddle and ball RAM values go in with opposite signs, scaled so a single pixel of difference gives a full unit of signal, and I'm starting to set the first-layer weights.

I'm settling the weights for the two steering units: the paddle and ball RAM values enter with opposite signs at 255 per pixel, with biases putting the dead zone around a difference of 4 to 6. The second layer scales both signals, and the output logits send RIGHTFIRE or LEFTFIRE only when the difference falls outside that band.

