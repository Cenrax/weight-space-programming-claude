The lander is stuck hovering near the top, with the main engine firing far too often, so the policy never descends. My plan is to drop the network and write a hand-designed controller based on the heuristic in gymnasium's lunar_lander.py, using small weights or saturated comparisons so the behavior is predictable.

Now I'm writing out the heuristic as a set of comparisons, and I'm planning to make the logits linear so that a simple argmax picks the action: the hover logit tracks the hover error, the two steering logits are the angle error and its negative, and the no-op logit is a small constant threshold.

I'm setting the logits to [0.05, A, H, -A], so a plain argmax reproduces the heuristic's choice. The remaining nonlinear parts, the clipping, the absolute value, and the leg-contact override, still need to be built out of hidden units.

I'm sketching the tanh layers now, using a scaled tanh to soft-clip the angle target to about ±0.4. The first layer's unit would be tanh(1.25x + 2.5vx), and I'm planning a second unit for the raw angle with a small scale so it stays nearly linear.

I'm working out how to get an even |x| out of tanh units, since tanh is odd and can't produce abs directly. Using tanh(a·x + b) + tanh(−a·x + b), which is even in x, might give a usable approximation near zero.

I'm trying to get an even |x| out of tanh units, since tanh is odd and can't produce abs directly. A pair like tanh(b+ax)+tanh(b-ax) is even and roughly quadratic near zero, so it could stand in for abs over the small x range.

I'm checking whether the |x| term even matters: the even tanh pair gives a rough curve, with about 0.025 at x=0.1, 0.156 at 0.25, 0.559 at 0.5, and 1.29 at x=1, so it tracks |x| only loosely.

I'm weighing whether the |x| term is worth the trouble, since the lander still steers toward center without it and the even tanh pair only roughly tracks absolute value. I'm also considering a different route to abs: a bias-shifted tanh or a larger scale in the second layer, though a plain product like x·tanh(cx) isn't available.

I'm thinking about the leg-contact case. Once the legs touch down, the angle term should drop to zero and the hover term should only damp vertical velocity, but a plain linear logit can't switch on contact. Since y sits near zero at touchdown, I'm wondering whether a large additive push on the side logits would be enough to gate them off.

I'm weighing whether to let the leg-contact case just zero out the side logits, or to keep the heuristic's hover term there. If the lander lands off the pad, a hover target near 0.14 could fire the main engine and lift it off again, but the heuristic itself behaves that way too, with H going to -0.5vy at contact. A hard suppression of every action on contact would stop a bounce but could drop a fast touchdown without braking, so I'm unsure which is safer.

I'm weighing whether leg contact should just push the side logits down hard, and I'm considering whether the main-engine logit also needs a moderate negative push so a touchdown off the pad doesn't trigger a lift-off.

I'm now settling on a leg-contact rule: each leg that touches down subtracts a small amount from the main-engine logit, and the side-engine logits get a large negative push so they shut off once the legs are on the ground. I'm still weighing whether the side-engine suppression should be that strong, since landing gently matters more than a perfectly even heuristic.

Now I'm weighing how to zero out the unused hidden units and whether rewriting whole weight rows is worth the effort, since editing individual entries would leave the leftover weights in place.

Next I'm working through the weight layout: since I can't just edit single entries without leaving stale weights from the unused units behind, I'm considering rewriting whole rows and zeroing out the unused ones.

Zeroing every hidden unit I'm not using is simpler if I overwrite the entire first-layer weight matrix and bias, since unused units then output exactly zero and their outgoing weights stop mattering.

Now I'm laying out the weight edits: zeroing the unused first-layer rows and biases, then setting the used layer-2 rows and the output rows and bias.

Working out the layer-1 units now, starting with a tanh that carries the angle target from the horizontal position and velocity, plus a near-linear unit for the raw angle.

Working out the next piece of the controller: the angle-rate and angle terms get folded into a single linear unit, so the side-engine signal stays smooth near zero. Combining that with the scaled target unit, I'll feed a second tanh layer that outputs the side-engine logits.

I'm defining the hidden unit for the vertical term, a tanh of a scaled-down sum of height and vertical velocity, so the hover signal stays smooth.

Working out the |x| term now: I'm treating the pair tanh(1+2x) and tanh(1-2x) as an approximation, scaling its combination by about 0.22 so it matches 0.275|x| near zero.

Working out the abs term from tanh pieces, I'm settling on a scaled sum of two shifted tanh units that stands in for 0.275|x| in the hover logit. Then I'll set up the first-layer combination that feeds the A unit.

Working out the output logits now: the main engine gets a constant small bias, and the side engines get the angle signal from V0 with a strong negative push from the leg-contact term V2. I'm still deciding how much the hover term should bear on the second side-engine logit once the legs are down.

Checking the tanh compounding near the thresholds: at z=0.3, tanh(tanh(0.3)) is about 0.283, close enough to 0.3 that the nested units stay usable.

Next I need to confirm the sign convention for A against the actual heuristic in gymnasium's lunar_lander.py before trusting the layout.

Continuing with the heuristic's branch logic: I'm checking the hover-versus-angle comparison and the leg-contact case, where the angle term gets zeroed and the hover term reduces to damping vertical speed.

Checking the sign convention against the heuristic's branch order: the side engine fires only when the angle error clears the 0.05 threshold, and the main engine takes priority when the hover error is larger than the angle error. I'm also confirming that a positive angle target pushes the lander left, which matches how Box2D orients the thrust, so the layout stays consistent. The one open question is how strongly the leg-contact term should pull H down, since the heuristic drops it to just damping vertical speed while my version still carries the position terms.

I'm checking where the touchdown height sits in the observation and whether the leg term needs any extra weight on the second side engine. The heuristic's leg case is simpler than my layout, so I'll keep the main-engine logit as it is and move on to finishing the weight scales.

Setting the first-layer weights now: the angle-target unit, the linear angle unit, the hover unit, the two |x| tanh units, and the leg-contact unit each get their input weights and biases.

Setting up the layer-2 side-engine signal next: the V0 unit combines the angle-target and angle-error units, and V1 handles the hover term with the |x| correction folded in.

Still writing the output-layer weights...

Checking the leftover weights: the unused layer-2 columns come out to exactly zero, and the unused first-layer units do too, so only the rows I set actually matter. Next I want to quantify how far the |x| approximation drifts, which is about 0.034 versus 0.069 at x=0.25 and 0.1 at x=0.5.

Checking the approximation error on the |x| term: it runs about 0.034 against the true 0.069 at x=0.25, and close to exact by x=1. Then I'll look at whether the hover unit saturates on the initial drop, where it lands near -0.40, which seems fine. The main concern is the current network hovering near y≈1, with the main-engine logit winning whenever vertical speed is slightly negative, and the weight on the vy input looks like the cause.

Now I'm writing the first-layer row for the position input: the angle-target unit gets 1.25, and the two hover-offset units get 2 and -2, with everything else left at zero.

Filling in the weight matrices row by row: the remaining first-layer rows get their zero-padding, the bias vector is set, and the second-layer rows follow.

Now I'm filling in the first-layer weight rows, one unit at a time, so every row ends up with exactly 64 entries.

Still filling in the weight matrices...

Still filling in the weight matrices...

