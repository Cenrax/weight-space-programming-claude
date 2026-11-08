I'm working from the Atari RAM map for Breakout, where the paddle's x position sits at address 72, the ball's x at 99, and the ball's y at 101. The paddle's horizontal range looks like it runs roughly from 55 to 191.

I'm trying to recall the Breakout RAM layout, where the paddle's x position is at address 72 and the ball's x is at 99. I half-remember the paddle ranging about 55 to 191, and the ball's x maybe 55 to 200, but I'm not sure of the exact edges. I also don't know the offset between ball and paddle coordinates, so I'm weighing whether a simple target of ball x minus paddle x is enough.

I'm weighing the offset between paddle and ball positions, where the paddle's center sits roughly 8 pixels past its left edge and the ball's center about 1 pixel past its own. That suggests the paddle should target around ball_x minus 7, though I'm not confident in those exact offsets. A simple deadband controller might be safer than relying on a precise target.

I'm checking whether ALE treats Breakout as a paddle game the way it does Pong, since that would change how the paddle input works.

I'm checking whether holding the fire button keeps re-serving the ball, or whether Breakout needs a fresh press each time to launch it.

I'm weighing whether holding fire keeps serving the ball, since some setups need a fresh press each time. A fire-only-when-centered approach would still keep firing persistently, which fails if the game needs an edge, so I'm stuck on that.

I'm now leaning toward combining direction with fire, since Breakout's serve likely responds to the button held down, and firing has no effect during play. I'm also considering that the paddle might simply track a stale ball position when the ball isn't in play, which seems acceptable. Next I want to work out the coordinate offset between the paddle and ball, which I'm not sure about yet.

I'm recalling an OCAtari extraction for Breakout, where the paddle's pixel x is derived from RAM[72] minus about 47, and the ball's pixel x from RAM[99] minus about 49. If that holds, I could convert both into the same screen coordinates to get the offset between them, though I'm not certain of the exact constants.

I'm working out the alignment formula: with the paddle's center at its position minus 39 and the ball's center at its position minus 48, matching them means the paddle value should sit 9 below the ball value, so I'd move right whenever the error is positive. I'm not sure of these offsets, so I'm planning a deadband of about three pixels to absorb the uncertainty.

I'm estimating the paddle's speed from the emulator's paddle delta, which suggests it crosses the playfield in roughly 17 pixels per step, faster than the ball. That speed could make a deadband controller overshoot and oscillate around the target.

I'm now wondering whether Breakout in ALE is driven by a joystick rather than a paddle, since my recollection of the emulator's controller setup is unclear. The paddle's speed could also be uneven across the playfield, which would affect how quickly it can catch the ball.

I'm now considering whether the paddle can overshoot the target, since it may cover roughly 17 pixels per step, and whether a no-op inside a small error band would avoid oscillating around the ball.

I'm estimating the paddle's movement per step, which suggests a deadband around 8 or 9 pixels to keep it from oscillating, though the paddle's half-width of 8 means a near-edge hit is still possible. I'm also wondering whether the pot-to-position mapping in ALE changes the actual speed.

Now I'm weighing how wide the deadband should be, since the paddle may cover around 17 units per step and a small band could make it oscillate around the ball, while a large one risks missing near the paddle edges.

I'm settling on a deadband of about 6 RAM units, since the paddle may overshoot by roughly 12 if it moves about 17 per step, and a wider band would risk missing near the paddle edges.

I'm settling on a fire-plus-direction action set, with a deadband deciding when to move. Then I'm worried that holding fire might not re-serve the ball after a lost life, which would leave the agent stuck, so I'm considering toggling fire to cover that case.

I'm wondering whether holding fire could be edge-triggered in Breakout's paddle setup, in which case a settled paddle sitting in the deadband might never re-serve the ball. Alternating between a move and a fire might supply the press edges naturally, and I'm still unsure how the button is read in the original code.

I'm settling the action choice now: fire during play is harmless in Breakout, so I'll use the left-fire, right-fire, and fire-only actions (12, 11, 1) and let the deadband pick between them. Then I'm turning to the input scaling, where I need to shift the ball-minus-paddle difference by the offset and the deadband width before feeding it into the hidden units.

Still writing the weight setup for the hidden unit...

I'm second-guessing the 9-unit offset, since my recollection of the OCAtari constants for the paddle and ball positions is shaky. I can't confidently recall the exact formulas, so I'll keep the coefficients as they are and treat the offset as provisional.

I'm settling the coverage math: with an offset uncertainty of about ±4, a deadband of 6 should still catch the ball when the paddle is within reach, even though the paddle's half-width might be smaller than I'd like. I'm working out the hidden layer pass-through and the output weights for the three actions.

Still working out the hidden-layer weights for the three actions...

Still working out the bias terms for the output layer, with the FIRE action getting a slight positive bias so it wins whenever the paddle is inside the deadband.

Now I'm writing out the weights for the hidden units, with the sharp tanh thresholds set so one unit fires when the difference is above 14.5 and the other when it is below 3.5.

