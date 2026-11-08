I'm starting on a Breakout controller that reads game state straight from RAM, and I'm recalling the AtariARI addresses for ball position, paddle position, and blocks hit.

I'm weighing the recalled RAM addresses against plausible value ranges: the paddle position around 72 and the ball's x and y around 99 and 101. I'm not certain of the ball's x range, but I think it likely shares the paddle's coordinate system.

I'm working out the paddle-to-ball offset: the paddle is about 16 pixels wide and the ball about 2, so I'd aim the paddle center at the ball's x, maybe using a tunable offset. I'm not fully sure of the ball's x coordinate convention, but I'll keep the mapping adjustable. I also need to handle serving: the ball only launches on FIRE, so I'm considering whether always pairing movement with FIRE is safe.

I'm weighing whether holding FIRE continuously would stall serving after a life is lost, since the serve might need a fresh press rather than a held button. I'm considering alternating fire with the movement instead, though that complicates tracking the ball.

I'm settling on a simpler control rule: move right or left when the paddle is off the ball's x by more than a deadband, and press FIRE when it's aligned, so serving still happens without holding FIRE through the whole game.

I'm considering a fallback for when the paddle is pinned against a wall with the ball's target out of reach, since the paddle would never press FIRE in that case. I could detect the paddle sitting near either edge while still wanting to move outward and force FIRE there, though I'm not sure of the exact wall values.

I'm weighing a threshold for the paddle's wall detection: too high and it would skip useful near-wall moves, though the paddle covers the ball well enough near the edge that a slight offset wouldn't cost much. I'm also trying to recall the paddle's actual RAM range, but I'm not sure of the exact minimum and maximum values.

I'm wondering whether ram[101] drops to zero when no ball is in play, which would give a cue for when to press FIRE. The risk is that a ball near the ceiling could also show a small y value, so I'd be pressing FIRE during play by mistake, though that only costs a little movement.

I'm thinking about whether the ball's y coordinate could signal when no ball is in play, since a value near zero would mean I should press FIRE. I could also fall back on the ball's x position, which might still sit near the paddle after a loss, so aligning and firing would serve it.

I'm settling on FIRE as the serve trigger. Since the ball is served by a level-checked button press, I'll assume holding it through the wait state is fine and skip alternating.

Now I'm weighing how ball tracking and paddle movement interact, and whether a simple greedy approach holds up at higher ball speeds.

I'm trying to recall the RAM layout for the paddle and ball, reasoning from a remembered comparison like "if the ball is past the paddle plus some margin." I'm also thinking about how an object-detection routine might read the player's x position out of RAM.

I'm recalling the OCAtari mapping, where the ball's center sits at ram[99] minus 48 and the paddle's center at ram[72] minus 39, so alignment means working out the difference between those two RAM values.

I'm working out the deadband rule: aim right if the offset is past a small positive threshold, aim left if it's past the matching negative one, and otherwise hold still so the paddle doesn't jitter around the ball.

Now I'm settling the mapping as a pair of threshold units: one that fires when the ball's RAM value runs well past the paddle's, and one for the opposite side, each set at a half-integer cutoff so the raw integer differences land cleanly on one side.

I'm writing out the weight and bias values for the two threshold units, with the right-side unit using a positive weight on the ball and negative on the paddle, and the left-side unit the reverse.

I'm checking the deadband tolerance against the paddle's half-width, and I'm leaning toward keeping it at 2 so the paddle doesn't oscillate too much. Next I'm working out whether the ball's y value can serve as a clean absent-ball signal, since I'm unsure where the ceiling falls in RAM.

I'm checking whether the ball-absent signal can be exact zero, since that would keep the no-ball unit from firing during play. I'm also worrying that the ball could freeze at an x-position the paddle can't reach near the edges, so I'm keeping the wall case in mind.

I'm considering a paddle-at-wall fallback: if the paddle sits at its left edge and the ball is still out of reach to the left, force a FIRE press there, though I'm unsure of the exact minimum paddle value.

I'm checking whether the paddle's left and right edges can be detected from RAM, looking at whether a wall threshold near 55 would hold up, and considering the right edge's max value too.

I'm weighing whether to skip the wall detectors entirely and rely on the ball-y signal, since a wrong wall threshold could leave the paddle stuck near an edge and cost more than it gains. I'm still inclined to include them, with the left one at 55.5 and the right at 183.5, accepting that a mistaken value would only degrade things slightly.

I'm thinking about wall-pushing: rather than adding a separate wall detector, I could use LEFTFIRE at the wall, which moves left and fires together, so the paddle keeps moving and the serve still happens.

I'm now settling on LEFTFIRE and RIGHTFIRE for the paddle's edge zones, since pressing fire while moving toward a wall is harmless even if my threshold is off. I'm also planning to keep plain RIGHT and LEFT elsewhere, with FIRE reserved for when the paddle is aligned under the ball.

I'm settling on the idea that the paddle can't get stuck: it either lines up under the ball and fires, or it reaches a wall and fires there with LEFTFIRE or RIGHTFIRE. That makes a separate ball-absent detector unnecessary. What I'm still unsure of is whether the paddle moves fast enough per step to jump over a narrow deadband and keep oscillating past it.

I'm weighing whether the aligned action should be FIRE, and I'm thinking about pairing the deadband with a wall-edge fire so the paddle doesn't stall near the sides.

Now I'm turning to the network's first hidden layer, writing out one unit that fires when the ball's RAM value sits well to the right of the paddle's, and a matching one for the left side.

Still writing the hidden-layer units...

Still working out the layer-2 gates now...

Still writing the layer-2 gates...

Still settling the output logits: I'm weighting the RIGHT and LEFT units against FIRE, so that when the paddle is aligned under the ball, FIRE wins.

I'm setting the output bias so FIRE wins when the paddle is aligned, which means the aligned logit has to beat NOOP's tie at zero. I'm settling on a bias of 1.5 for FIRE, so that it comes out ahead when the paddle is lined up under the ball and still stays below RIGHT and LEFT when the paddle is moving.

Checking the fire-on-alignment margin: FIRE sits at 0.5 when the paddle is lined up, above the tied zeros of the other actions. When the ball is absent, the FIRE output gets a larger boost so it serves immediately.

Still checking the wall-zone thresholds and the paddle-position range I'm working from.

Still working out the layer-2 weights for the ball-position gates...

Still writing the layer-2 weights...

Still writing the output weights for the three action units...

