I'm checking which game this is by looking at the largest-magnitude rows in the first-layer weights, which map to specific RAM bytes. The output layer also shows strong structure in the action columns for directional moves like right, up-right, and down-right.

I'm weighing the output's left-right actions against Pong's paddle, which moves up and down: RIGHT is up and LEFT is down. The RAM rows I flagged line up with Pong's player, enemy and ball positions, while Breakout's addresses don't match, so I'm settling on Pong.

