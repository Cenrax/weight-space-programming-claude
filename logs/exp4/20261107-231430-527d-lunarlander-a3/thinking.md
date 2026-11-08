The policy never descends and keeps firing the main engine, so I'm thinking of replacing the network with a hand-built controller, using the hidden units to approximate linear functions.

Now I'm working out the angle and hover targets from the state and turning those into the controller's branch conditions.

I'm weighing whether to drop the |x| term, since the hover target is small, and considering whether a second-layer combination of tanh units could approximate the absolute value.

I'm checking whether a two-layer tanh combination can approximate |x| with a quadratic term, but that looks like it might be more trouble than it's worth, so I'm considering skipping it. For the legs-contact gate, I need some nonlinearity to switch the hover target off when a leg touches, and I'm thinking large saturating weights in the first layer could handle that.

I'm considering whether the legs-contact gate is really needed, since without it the side engines might keep firing on the ground and stop the lander from settling. I'm thinking that layer-two units could add a large negative term driven by the leg input to switch the hover and angle targets off on contact.

I'm sketching layer-1 features as linear combinations with a small gain so the tanh units stay near-linear, starting with the angle target and moving on to the hover term.

I'm sketching the second layer so that the leg-contact input can gate the hover and angle targets off, since the first layer can't see the legs directly. The clip on the angle target to ±0.4 still needs handling.

I'm using a tanh unit to clip the angle target smoothly to about ±0.4, which lets the angle term be written as a simple linear combination in the second layer. The decision rule, choosing the hover action when the hover term beats the absolute angle term and clears a small threshold, will then sit in the output logits.

Still designing the gating layer: I'm picking layer-2 units that act roughly linearly on the angle and hover terms, so the logits can compare the hover term against the absolute angle term.

I'm checking how the argmax over the four logits {0.05, A, -A, H} reproduces the heuristic's branches, including the case where the heuristic falls through to a=3 regardless of H. Then I'm working through the legs-contact case, where A is zeroed and H reduces to -0.5 vy, to see whether it still wins over the 0.05 threshold.

I'm dropping the |x| term from the hover target, so it reduces to just the y and vy terms. Once the legs touch, that leaves the hover target close to -0.5 vy, which should still win the comparison against the 0.05 threshold.

I'm dropping the y-only hover target for now and weighing whether an even tanh pair in layer 1 can recover the |x| term.

Trying an even function of x built from two tanh units, with the offset chosen so it is zero at the center and saturates for large |x|, to stand in for the |x| term in the hover target.

I'm checking the scaled tanh pair against the target: at small x it stays below 0.275|x|, and it levels off near 0.23 for larger offsets, so the fit is only rough.

I'm checking the |x| approximation numerically now: the even pair tanh(3x-1)+tanh(-3x-1)+1.523, scaled by about 0.2, tracks 0.275|x| closely near the center and tapers off past x≈0.6.

I'm now sizing the layer-1 units so their tanh outputs stay roughly linear over the range of the hover and angle features.

I'm figuring out how to set the second-layer weights, since each used unit needs its own column of 64 input weights and those can't be written as a single row.

I'm working out how to set the second-layer weights column by column, since each used unit needs its own 64 input weights and zeroing the unused layer-1 contributions means a long list of individual entries.

I'm setting the first hidden units to approximate the angle target: one unit tanh(1.25x + 2.5vx) roughly tracks the clipped angle target divided by 0.4, and a second unit handles the angular-velocity damping term.

I'm now setting up the next hidden units to track the hover term, with a gain of about 0.5 on the y and vy inputs so the tanh stays mostly linear across the expected range.

I'm setting up a gating unit from the leg-contact inputs, so the output can switch off the side-engine and hover targets once a leg touches the ground.

I'm adding a gate that switches the leg-contact penalty on the side-engine and hover logits, so they lose out to the other actions once a leg touches the ground.

I'm settling the layer-2 units now: one tanh unit to carry the angle term, scaled from the first-layer features so it stays roughly linear.

I'm checking how a scale factor of about 0.15 on the tanh pair would fit the |x| term, and I'm comparing those values against 0.275|x| at a few points. The fit is rough at the larger values, so I'm looking at whether a slightly different offset does better.

I'm checking the tanh pair again: the offset term works out to zero at x=0, so the |x| stand-in is correct at the center. Now I'm writing out the third layer-2 unit, which has to carry the leg-contact gate into the logits.

Now I'm setting the output logits: the no-legs case keeps the penalty at zero, and once a leg touches it drops the side-engine and hover choices well below the others.

I'm checking the sign conventions in the heuristic's angle and hover targets before I trust the tanh approximations.

I'm checking the heuristic's angle and hover targets, clipping the angle target to ±0.4, and then how the branch conditions choose between the hover, side, and main actions.

Still checking the heuristic's landing branch, whether the legs-contact case really makes hover_todo just -0.5 vy, and how the pad height offset affects the y term once the legs touch the ground.

I'm checking the landing case where the legs are on the ground: with vy near zero and the pad offset small, the hover term stays just under the 0.05 threshold near center. But for |x| around 0.3, the even tanh stand-in pushes it above 0.05, so the controller would fire the main engine on the ground, which is an edge case worth noting even though that landing already fails.

I'm weighing whether to offset the |x| contribution by a constant when the legs are touching, which would make the controller need a sink rate of about 0.3 before firing, so the landing might be less gentle.

I'm settling the leg-contact offset: a small negative bias on the hover logit so that once a leg touches, the controller needs a slightly sharper sink rate before firing the main engine, and the constant is chosen so it cancels exactly when no legs are down.

Now I'm checking the sign of the angle term inside the first-layer unit, and whether the factor of 0.5 on the angle and angular velocity gives the expected approximation.

Now I'm checking the second layer's weights for the angle branch, where the first-layer units feed into a tanh that stays close to linear. For the hover branch, I'm re-verifying the constant offset so that it cancels exactly when no legs touch the ground.

I'm checking the leg-contact term numerically now: with both legs down the tanh saturates near 1, so the bias comes out to about -0.06 once legs touch, and I'm evaluating the tanh of 3 times that value to confirm the no-legs case cancels to zero.

Now I'm checking the output logits against the hand-set weights: the main-engine logit comes out to 10·v0 when no legs touch, and the leg-contact bias keeps the hover and side choices down once a leg is on the ground. I'm also confirming that the unused layer-2 units get zero output weights so they don't leak into the logits.

Still filling in the first-layer weights for the units that carry the angle, angular velocity, and hover terms.

Still filling in the weights for the remaining layer-2 units that carry the hover and leg-contact terms.

Checking the tanh approximation for the |x| term at a few values, I'm now looking at whether a 0.2 weight on the ground-contact velocity stays within range once the angular velocity saturates.

Now I'm pinning down the diagnosis: the network hovers because its main-engine logit responds to small negative vertical velocity, so it keeps vy near zero and never descends. The altitude input has only weak weights, so height barely changes the decision.

Now I'm filling in the weights for the second-layer hover unit, then setting the first-layer weights for the input-state features.

