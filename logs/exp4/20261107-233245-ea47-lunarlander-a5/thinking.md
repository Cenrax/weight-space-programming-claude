The policy is stuck hovering near the top and never descends, firing the main engine about 577 out of 1000 steps. Hovering would take roughly gravity-compensating thrust, so that alone doesn't explain it. I'm considering swapping the network for a hand-designed controller, like the gymnasium heuristic, and encoding it in the MLP. I'm starting to sketch that heuristic, beginning with an angle target built from the state.

Still writing the heuristic code...

I'm now settling on the logit design: a zero logit for the no-op, A minus 0.05 for the right-side engine, H minus 0.05 for the main engine, and negative A minus 0.05 for the left-side engine. Taking the argmax over these should pick the main engine when hover dominates and a side engine when the angle error is larger, which mirrors the heuristic's branching.

I'm weighing how to handle the absolute value of x in the hover target, since a linear logit can't express |x| directly. Dropping that term would let the lander descend even while drifting sideways, which might matter for landing on the pad, so I'm looking for a tanh-based approximation of |x|.

I'm trying to approximate |x| with the tanh units, since a single linear layer can't produce it. Stacking steps at several thresholds, or using steep tanh units as switches, might get close enough to the absolute value.

I'm building the soft clip for the angle target, using a single tanh unit scaled by the 0.5x plus vx term, so it saturates the way the heuristic's clip does.

I'm settling on a soft clip for the angle target using a single tanh unit, and considering whether tanh-based units in layer 2 can preserve the ordering between the hover and angle terms, since the same squash applied to both should keep the comparison intact.

I'm checking that the tanh-based outputs keep the same ordering as the linear terms, so the squash on both sides leaves the comparison intact. Layer 1 has to stay in its near-linear range, which means scaling the inputs down with a small epsilon and compensating in the next layer's weights.

I'm working on the legs condition now, since the heuristic zeroes the angle target and changes the hover target whenever a leg touches the ground. Suppressing the angle term through tanh saturation in layer 1 is the direction I'm considering.

I'm weighing how to gate the angle term when a leg touches the ground, since the heuristic zeroes it out. Using tanh saturation in layer 1 to kill it looks awkward, so I'm considering doing the suppression in layer 2 instead, and asking whether the leg logic really matters for settling after touchdown.

I'm checking whether the legs condition actually matters near the pad: at small x the side and hover terms stay under the 0.05 threshold, so the engines stay off anyway, and only at larger x does the gating start to matter.

I'm weighing whether the legs gating is worth encoding, since a leg-contact term could cut the side logits and switch the hover target to plain descent. The difference between the two hover targets is small near the ground, so the main gain would be cutting side-engine firing after touchdown.

I'm weighing whether to add a leg-contact unit at all, since the main gain would be cutting side-engine firing after touchdown. I can't gate the layer-2 pre-activation directly, so I'm considering a separate tanh unit driven by the leg signals instead.

I'm weighing how to gate the side-engine logits on leg contact, so I'm sketching a tanh unit fed by the leg signals and pushing its effect into the output layer.

I'm working out how to approximate the absolute value of x in the hover target with a few tanh units in the first layer, using offsets so the output roughly matches 0.55|x|.

I'm trying to build |x| out of tanh units, since a single linear layer can't produce it. A bump shaped like 2 + tanh(k(x - t)) + tanh(-k(x + t)) looks like it could work, so I'm checking the shape of that combination.

I'm approximating the absolute value with a sum of tanh pairs at thresholds from 0.1 to 0.7, each giving a near step once |x| passes its threshold.

I'm using a sum of tanh step pairs at thresholds 0.1 through 0.7 to approximate 0.55|x|, which should cost about 14 units in the first layer, and I'm reconsidering whether that precision is needed at all.

I'm working out how to build the absolute value from tanh step pairs, and then I'll need to lay out the weights so the unused units don't interfere. Rewriting the second layer means setting each column one entry at a time, which is tedious but straightforward.

I'm zeroing out the unused first-layer units by setting their input weights and biases to zero, which keeps them from feeding into the second layer.

I'm going to rewrite the first layer as full rows, zeroing out the unused units, and then set only the second-layer weights that feed the three gating outputs. The output layer gets rewritten in full as well, so the leftover units can't contribute anything.

Next I'm assigning the first-layer units their roles: one near-linear unit each for x, y, vx and vy, with the scaling set so the tanh stays in its linear range.

Next I'm assigning the remaining first-layer units: a near-linear unit each for the angle and angular velocity, a tanh unit for the angle target built from x and vx, a leg-contact gate, and a bank of step pairs on each side of x for the absolute-value approximation.

I'm writing the hover target as 0.55 times the absolute value of x and the angle target through the same tanh-based units, then assembling the output as the sum of the A and H terms with their halved coefficients. Right now I'm laying out the absolute-value approximation as a stack of step pairs in the first layer.

I'm checking the absolute-value approximation at x=0: the sum of tanh steps gives about 0.078 instead of zero, so the midpoint is off and I need to account for that offset.

I'm checking the offset in the absolute-value approximation: dropping the extra constant to 0.7 and adding 0.05 per unit gives about 0.028 at x=0, and at x=0.3 the sum comes out to roughly 0.7 minus 0.45.

I'm settling the absolute-value offset at 0.03, since the staircase undershoots at 0.3 and overshoots near zero, and then writing the hover target as 0.275 times that approximation minus the y and vy terms.

Still writing the weight assignments for the two hidden units...

Still working out the layer-1 layout: the hidden units are assigned to y, vy, angle, angular velocity, the angle target, the leg gate, and the step pairs for |x|. I'm reindexing the unused units so they stay at zero.

Checking the leg-contact gate: with legs down the gate pushes the side logits well below the zero-engine logit, so the side engines stay off. I'm also checking how the hover term behaves once legs are touching, since the leftover y offset is small near the pad.

Still checking the leg-contact case: once the legs touch, the hover term should be small since y stays near zero and the velocity term dominates, so the remaining question is how much the absolute-value approximation adds at small |x|.

I'm checking what the main engine does when the legs are touching, since the hover term would still push it to fire at rest. At x near 0.2 the absolute-value approximation gives about 0.064, which is above the 0.05 threshold and would keep the engine on. So I'm considering whether to gate the hover term on leg contact, the way the heuristic handles that case.

I'm adding a constant penalty to the main engine's logit whenever the legs are down, so that touchdown doesn't keep it firing, and I'm checking that the hover term stays small enough at rest.

Setting up the output layer now: the main-engine logit stays at zero, and the hover logit gets the H term minus 0.05 with a leg-contact term subtracted so the engine doesn't keep firing on touchdown. I'm working out the bias so the no-legs case lands at about -0.05.

Still writing the weights for the output layer: the main-engine logit stays at zero, the hover logit is set from the hover term and the leg-contact penalty, and the side-engine logit gets its own leg-contact adjustment. Checking the biases against the no-legs case, they come out around -0.05.

Still checking the leg-contact cases: with both legs down the gate pushes the hover logit down enough that the main engine stays off, so the remaining question is whether the tanh squash keeps the ordering of the logits intact.

Checking the sign of the angle target next: the heuristic wants the lander to tilt toward the pad, so I'm confirming that the angle term takes the right direction.

Still checking the approximation error on the angle target: the tanh-scaled units keep the angle and angular velocity terms small near zero, and I'm looking at how the soft clip on the x-velocity term differs from a hard clip.

Still setting up the second-layer weights: the angle, angular velocity and angle-target units feed the A term, the y and vy units plus the step pairs feed H, and the leg-gate unit goes to the third.

Still writing the first-layer weights, row by row: the x input feeds the wide step bank and the angle-target unit, and the y and angle inputs get their small scaling factors.

Still writing the layer-1 weights: the leg-contact unit takes a negative bias with -3 weights, and the step-pair units for |x| get their staggered biases from -1 to -7. Then I'm filling in the layer-2 weights from those units into the A and H terms, and setting the layer-2 biases to zero.

Still writing the output weights: the third layer's rows get their signs and the bias values for the main, side and hover logits.

I'm checking the side-engine sign against the heuristic, where a positive angle target means firing the orientation engine to rotate counterclockwise. Looking at the failure, the network seems to hover near y≈1 with the main engine firing roughly 58% of the time, which suggests it has learned to avoid crashing rather than to descend. The vertical-velocity weights look like the likely cause of that stalled descent.

Filling in the layer-1 weights now, starting with row 0 for x: the step units for |x| get 10 and -10, and the angle-target unit gets 1.25.

Still writing out the layer-1 weight matrix, filling the zero entries for each input row.

Filling in the first-layer weights row by row, with the step units for |x| set to ±10 and the angle-target unit at 1.25.

Still filling in the second-layer weights: the first few hidden units feed the H term with fixed coefficients, and the leg-gate unit goes to the third output.

Writing out the 64 rows of the third-layer weights now.

