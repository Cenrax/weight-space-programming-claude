I'm working out why the policy hovers: it fires the main engine about 58% of the time and holds altitude near y≈1.0 without descending, so the main-engine logit must be too high compared to doing nothing when vertical velocity is near zero. I'm considering replacing the network with a known heuristic landing controller from gymnasium, using hidden units to compute linear features that feed the output logits.

Still writing out the heuristic controller...

Now I'm checking the argmax logic against the heuristic, which matches except for ties. Next I need the second layer to produce hover_todo and angle_todo as linear pieces, which means handling abs and the clipping with small input scaling so tanh stays near linear.

Now I'm approximating the target angle as a soft-clipped linear combination of the state, since a scaled tanh stays close to a clip near zero. For the absolute value of the horizontal position, I'm still trying to find a tanh-based form that doesn't come out purely linear.

Now I'm trying to get the absolute value of the horizontal position out of tanh units. A plain tanh pair won't produce it cleanly, so I'm considering a sum of shifted step-like tanh terms at several thresholds.

I'm weighing whether to approximate the absolute value with a sum of shifted tanh steps, which would take around twenty units, or simply drop the hover target. Dropping it risks landing off the pad when the initial horizontal push is large, so keeping an approximate abs seems worth the extra units.

I'm looking at building an even function out of tanh units: tanh(1+kx) + tanh(1-kx) is symmetric in x and falls toward zero for large |x|, so it could stand in for the absolute value.

I'm testing whether a symmetric pair of shifted tanh units can stand in for the absolute value, since their sum is even in x and drops toward zero for large |x|.

I'm checking how a difference of shifted tanh terms, tanh(2x-1) - tanh(2x+1), behaves as an approximation of |x|: at x=0 it gives about -1.52, and it rises toward zero as |x| grows, so it captures the shape but saturates.

Now I'm weighing a few tanh pairs at different offsets to approximate the absolute value more closely, since a single pair stays too quadratic near zero.

I'm picking the layer-1 tanh units so their sums form a step-based approximation of |x|, with about ten units total spread across thresholds on each side, and I'm checking whether the second layer can then combine them into the hover target.

I'm weighing whether the hidden layer can give each heuristic term its own pre-activation, scaled small so tanh stays linear, and then letting the output weights undo the scaling.

I'm working out how to handle the legs-contact condition, since tanh can't multiply a term by zero. One option is to saturate a unit with a large leg-driven bias so it outputs a constant, then cancel that constant with a second leg-driven unit of opposite sign.

Now I'm working on how to handle the leg contact condition in the first layer, using separate tanh units for each leg so that both-legs and no-legs cases produce distinct saturated values.

Now I'm handling the leg-contact case with a pair of saturating units: one gets the angle term plus a large leg-driven bias, the other gets only the leg-driven bias, so their difference cancels out once both legs are down.

Now I'm checking whether the leg-contact case matters at all, since once the legs touch down the heuristic should mostly stop firing. I'm wondering if the lander would keep correcting its angle with the side engines on the ground, which could stop it from settling to sleep.

Now I'm splitting the hover target into an always-on part (the vertical-velocity term) and a flight-only part (the horizontal and altitude terms), so the leg-contact gating only has to act on the second one.

Now I'm planning to lay out every weight row explicitly, and I'm thinking about whether zeroing the unused layer-2 units is enough, since I'd only need to set the output weights for those to zero.

Now I'm deciding to zero out the unused first-layer units, so that only the layer-2 entries for the ones I actually use need explicit values.

Now I'm settling the angle target and the angle correction term, then moving on to laying out the explicit weight rows.

Now I'm setting up the first-layer units: one nearly linear unit for x, and a soft-clipped tanh for the target angle, scaled so it stays close to a clip.

Now I'm settling on a few first-layer units: a nearly linear one for the vertical velocity, and small-scaled tanh units for the angle-and-angular-velocity and the altitude terms, so each stays close to linear.

Now I'm picking the scale for the gated angle unit, so its pre-activation stays roughly linear in the angle correction while the leg-contact bias can still saturate it.

I'm checking the gating constant B: with both legs down the preactivation doubles, so tanh saturates and the angle term barely leaks through, which is what I want.

I'm continuing with the leg-contact compensation: the second unit cancels the constant once both legs are down. Next I'm setting up the hover-flight gate, which needs an approximation of |x| built from a set of step-like tanh terms.

I'm checking the step-based approximation of |x| by summing shifted tanh terms, and working out what constant offset they contribute at x=0 so the hover gate scales correctly.

I'm checking the step-based |x| approximation at a few points: at x=0.1 and x=0.2 the sums land close to the true values, so the error stays small. I'm also working out how sharp the step transitions need to be, so I'm about to set the smoothness scale.

Now I'm writing the hover-gate term in layer 2: a sum of the step-based |x| terms, plus the scaled altitude unit, all wrapped in the same leg-gating bias.

Still writing the hover-gate logits...

Now I'm checking the tanh gating constant and the layer-2 preactivation scale: with ε_a = 0.2, an angle target of 1 gives a preactivation of 0.2, which tanh keeps nearly linear, so dividing by ε recovers the value to about 0.987.

Still writing the layer-1 weight rows...

Still writing the layer-1 weight rows...

Still writing the layer-2 weight rows...

Still writing the layer-2 weight rows...

Still writing the output-layer weight rows...

Still writing the output-layer weight rows...

Now I'm checking the soft clip on the angle target: at small inputs the tanh version comes out a bit softer than a hard clip, which should be acceptable for this controller.

Checking the step-based |x| approximation, the step units I'd placed are roughly right, so I'll sanity-check the hover target near x=0 against the gymnasium heuristic's landing behavior.

I'm checking whether the hover target near y≈0.027 keeps the lander from ever touching down. With vy near zero the main engine doesn't fire, so it keeps descending, and the same happens with the exact heuristic, so it should still make contact.

Still checking the step-based approximations and the layer-2 tanh scales, to confirm the angle and altitude terms stay close to linear.

Now I'm checking the failure diagnosis against the weights: the lander sits near y≈1 with vy near zero, and the main-engine logit wins whenever vy dips slightly negative, while "do nothing" almost never wins. So altitude barely affects the output, and the policy has no real drive to descend.

Now I'm laying out the layer-2 weights, so I'm rewriting the 16 rows that feed the used units with explicit values in the first four columns.

Still writing the layer-1 weight rows...

Still writing the layer-1 weight rows...

Still writing the layer-2 weight entries...

Still writing the layer-2 weight entries...

I'm checking the left-engine logic against the heuristic: the angle target comes out positive here, so the rule fires the left engine. I'm taking the heuristic's sign convention as given rather than re-deriving the rotation direction.

