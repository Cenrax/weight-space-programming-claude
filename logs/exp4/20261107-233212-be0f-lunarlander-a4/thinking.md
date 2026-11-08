The policy is stuck hovering because it fires the main engine about 577 of 1000 steps, holding altitude near y ≈ 1.0 instead of descending, so the main engine logit is dominating. I'm considering replacing the network with a hand-designed heuristic controller from gymnasium, reusing hidden units to compute linear features that feed the output logits.

Now I'm working out how to turn the heuristic into weights, with angle and hover targets as linear features in the first layer and the decision logic handled in the second.

Now I'm working out how to make the controller's decisions show up as logits: the heuristic's angle and hover errors would feed a second layer of thresholded units, and the output logits would be simple combinations of those.

I'm checking the logit scheme: argmax over H, A, -A and a constant 0.05 reproduces the heuristic's branches, so the logits only need to be linear in H and A. Keeping the first two layers near-linear with small weights and scaling the output layer should make that work, with tanh's cubic error staying small. The remaining question is how many decimal places the weights need.

I'm checking whether the scaled tanh-of-tanh keeps the comparison between H and |A| intact, since both values pass through the same odd, monotonic distortion. That means the argmax still follows the heuristic's branches, with the only error coming from the small compression near zero.

I'm handling the clip on the angle target: a tanh with a scale factor can approximate it in layer one, so the target feeds into the linear part of the angle error. The remaining worry is that H and A get computed in the second layer's pre-activation, so the scale has to stay small enough that the same squashing applies to both and the comparison still holds.

I'm laying out the first layer: one unit carries a scaled-down version of the negated angle and angular-velocity terms, and a second unit approximates the clipped angle target with a scaled tanh of the x-position and angle inputs.

Now I'm working on the hover term, which needs the absolute value of the horizontal position. Since tanh units are odd, I'm looking for a way to build |x| out of a couple of shifted tanh units rather than a plain linear map.

I'm trying to get an even function out of tanh units, since |x| can't come from a single odd unit. A difference of shifted tanh terms gives a dip near zero that flattens out at larger |x|, so the absolute value might be approximable that way.

I'm testing whether a tanh-based shifted difference can approximate |x|, trying k=2 and d=0.5 and checking a few sample points. The values at x=0, 0.2, and 0.5 track the absolute value reasonably well, so I'm checking how it behaves out to x=1.

I'm checking whether a symmetric pair of shifted tanh units gives a usable approximation to |x|, since its quadratic shape near zero isn't quite right. I'm considering summing several tanh ramps at different offsets instead, so the absolute value gets built up piecewise.

I'm checking how well a tanh-based shifted difference tracks the hover target: at x=0.2, 0.5 and 1 it gives roughly 0.055, 0.31 and 0.71 once scaled by 0.55, which is close enough for the lander to descend near the pad and rise when it drifts.

I'm checking whether a shifted-tanh difference gives a usable even approximation to |x| at a larger scale, and it looks workable with k=3 and d=0.3, though the bump gets steep quickly around x=0.3. I'll settle on k=2 and d=0.5 instead, and adjust the overall scale so the hover target's slope comes out about right.

I'm checking whether the heuristic's success rate holds up, since it should land on the pad and come to rest. Once the legs touch down and vertical speed goes to zero, the hover target drops to zero and the controller just stays idle. The remaining piece is gating the angle and hover targets on leg contact, which takes some extra work to express without multiplication.

I'm weighing whether to gate the controller on leg contact by simply cutting the engines once a leg touches down, since expressing the heuristic's "zero the angle target and hover on -0.5·vy" rule would need multiplication. The worry is that a leg touching at an angle while still descending fast could crash the lander.

I'm weighing whether the lander's behavior on leg contact should be handled by a constant logit boost, since the full heuristic formula would still keep producing a small hover term near the ground. The open question is whether that boost is strong enough to keep the engines quiet without cutting off the correction needed on a tilted touchdown.

I'm thinking about gating the engines on leg contact with a large constant boost to the main-engine logit, so nothing fires once a leg touches down unless the required correction is very large.

I'm checking the heuristic's angle target, which clamps a weighted sum of angle and horizontal position to ±0.4, and the hover target at 0.55 times the absolute horizontal position. The angle and hover errors are both proportional-derivative terms, so the rest of the controller follows from those.

Still working through the controller's branch logic and how the logits map onto the heuristic's choices.

I'm checking the sign logic for the side-engine branch: if the heuristic's hover term is at most |A| and no larger than 0.05, then the side logit wins and the argmax lines up with the heuristic. Now I'm turning to the nonlinearity, where the logits are a linear readout of tanh hidden units, and I'm looking at how a constant term fits in alongside the hA and hH units.

I'm checking whether the first-layer tanh units can stay close enough to linear for the logits to match the heuristic. If I keep the layer-one scale small, the distortion from the nonlinearity should be small, so the constant threshold for the main engine can be set directly through the output bias.

Now I'm settling on keeping the first layer nearly linear by using a small scale, so the tanh distortion stays negligible, and the second layer can use a comparable gain for both the angle and hover paths since they share the same nonlinearity.

I'm laying out the first-layer units so that the tanh stays nearly linear at a small scale, and I'm sorting out which rows of the weight matrices need zeroing so the unused units don't leak into the output.

Now I'm zeroing out the first-layer weight rows so that any unused units produce exactly zero and can't leak into the second layer.

I'm laying out the weight edits for the network: zeroing the first-layer rows and biases except for the units I need, then setting the second-layer rows and biases so unused units stay at zero. Once those are fixed, the third layer only needs rows for the units I actually use, plus the output bias.

Writing the first-layer unit formulas now...

Writing the layer-two preactivations now...

Now I'm settling the output logits: a main-engine logit of 10 times the hover-of-altitude term, the side logits set from the same scaled terms, and a constant base for the main engine plus a leg-contact boost.

I'm settling on a small value of 0.02 for the first-layer scale, so the second-layer weight works out to 50. I'm now weighing how strongly the leg contact should suppress the main engine, since a single touching leg gives a fairly large activation and I want the engines to stay quiet unless the fall is fast.

I'm weighing a large constant boost to the main-engine logit on leg contact, around 20, so the engines stay off once the lander touches down. The worry is that tilted contact could still need side correction, but the aim is simply to make the ground state quiet.

I'm setting the leg-contact gate with a constant boost of about 4.5 on the main-engine logit, so once a leg touches, the main engine only fires when the hover term exceeds roughly 0.5, meaning a fast fall. Side firing would still depend on the angle error passing a similar threshold.

I'm settling the leg-contact gate at K=5 and checking that the lander can rest on its legs with the engines idle. I'm also recalling how the heuristic usually scores on landing, though that's a loose memory and not something I've checked here.

Still working out the weight values for the first-layer units, starting with the angle term.

Still filling in the weight matrices for the first layer and the connections into the second.

Writing the last few weight rows now...

Now I'm checking the bias term for the threshold logit, which works out to about 0.4996, and confirming the tanh distortion in the second hover unit stays negligible. At the start, with the lander high and slow, the hover target is still negative, so the main engine stays off until the fall brings the hover term above the threshold.

Checking the argmax at the all-zero state, the logits come out to roughly [0.5, 0, 0, 0], so the no-op action wins as intended. I'm also confirming that the layer-two tanh stays monotonic, so the scaled outputs reproduce the heuristic's ordering.

Writing out the first-layer weight rows for the x, y, vx, and vy inputs now...

Filling in the remaining first-layer rows for the angle and angular velocity inputs, with the leg-contact gate on the last two inputs.

Now I'm filling the first-layer weight rows, writing out the nonzero entries in 64-wide blocks.

Still filling in the first-layer and second-layer weight rows...

Still filling in the weight matrices, writing out the remaining rows and bias values for the first two layers...

