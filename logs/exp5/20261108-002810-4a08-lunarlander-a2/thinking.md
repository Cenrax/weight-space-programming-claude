I'm working out a network that could reproduce the heuristic controller from gymnasium, which computes target angle and hover values from the state and then derives angle and hover corrections.

Still working through the switching logic: the hover branch fires when its target beats the absolute angle correction and clears a threshold, and otherwise the angle corrections decide the sign.

Next I'm turning to the hover target, which needs |x|, a nonlinearity a single tanh layer can't express directly, so I'm weighing whether shifted tanh units can approximate it.

I'm considering the idea of approximating |x| with a sum of shifted tanh steps, and I'm also weighing a single tanh unit with an offset as a cheaper way to get the symmetric shape.

I'm trying to pin down |x| with a sum of shifted tanh steps, spacing the thresholds across the range and scaling each step by the hover coefficient, which should keep the approximation smooth enough for the controller.

I'm settling on how many tanh steps to use for |x|, with about 16 units spread over both sides and a steepness near 10 so the steps blend smoothly. Then I'm wondering whether the output even needs that much precision, and I'm going back over the architecture: linear-ish first-layer units, then high-gain tanh comparisons in the second layer to get near-binary signals for the decisions.

I'm writing the decision logic as conditions on H and A: the hover branch needs H greater than |A| and above 0.05, then the sign of A picks the angle branch. I'm defining the layer-2 sign units g1 through g5 for those comparisons, and the output logit for the main action is K times the sum of the first three minus 2, which is only positive when all three fire.

I'm setting the logits for the three actions: the main action gets 10 when all three hover and angle conditions hold and -10 otherwise, while the side actions get +5 only when their own condition triggers. That way the main action wins whenever it fires, and otherwise nothing gets chosen. I also note that the two side conditions can't hold together, and I'm about to check what happens to H and A when the legs are in contact.

Next I'm working out how to handle the leg-contact case: when both legs touch down, the angle correction should drop to zero and the hover correction reduces to just the vertical-velocity term, so the side-engine units need a large suppression term driven by the leg signals.

I'm weighing how to handle the leg-contact case in the sign units: when both legs touch down, the angle branch should drop out and the hover term should reduce to just the vertical velocity. That means the hover-side threshold has to change once the legs are down, but the position-dependent part of the hover target could still push the main engine on the pad, so I'm checking whether that edge case matters.

I'm working out how to gate the leg-contact case in the sign units, since a plain sum of leg signals won't give a clean on/off switch. I'm checking whether clamping the summed leg contact through a steep tanh would work better.

Now I'm making the contact indicator C a near-binary signal, then gating the layer-2 units on it so that when both legs are down the side branches are overridden and the hover check drops to the vertical velocity.

Now I'm working out how the contact indicator changes things: when both legs are down, the side-engine branches get suppressed and the hover check falls back to just the vertical-velocity term. I'm weighing a separate gated unit for the landed case against the rest of the layer-two logic, and whether the main-engine logit still comes out right once the two cases are combined.

Now I'm checking the side-engine conditions against the contact gating, since the leg-contact term has to suppress them cleanly once the lander is down. Gating the side units on contact means their thresholds shift by a large term, and I'm also noting that the heuristic only needs the lander to settle, not to score especially high.

I'm checking how the first-layer tanh units behave once I keep the inputs small: their outputs only track the linear features well while the arguments stay near zero, and the tanh compression starts to distort the sums that feed the decision units.

I'm now working on a layer-1 unit that computes a composite angle feature directly from the raw inputs, so the cancellations don't get distorted by tanh compression. The open question is how often the clipping at ±0.4 actually triggers, since the initial impulse could push velocities high enough to matter.

I'm estimating the initial impulse from the force, mass, and timestep, which gives a velocity change of roughly 4 m/s and an observed horizontal velocity near 0.8, so the clipping at ±0.4 looks like it will trigger often right at the start.

I'm now using a soft clip for the angle target, a scaled tanh that saturates near ±0.4 and stays close to linear for small inputs, since the initial impulse can push the horizontal velocity well past that limit.

I'm settling on a soft clip for the angle target, a tanh scaled to saturate near ±0.4, and then setting up a second layer-one unit for the angle-and-angular-velocity part with its scale still undecided.

I'm settling the scale for the raw-feature unit at 0.2, which makes its output 5 times the tanh value, and I'm writing the combined angle feature A as 0.2 times the soft-clipped target term plus 5 times that unit.

Now I'm setting the hover linear term to a tanh-scaled sum of y and vy, and I'm starting on the horizontal part of H, which needs the absolute value of x approximated by stepwise pieces.

Next I'm checking how the absolute-value steps behave near zero, since the paired units at ±0.05 should combine into a smooth approximation of |x|.

Now I'm checking how much the rounding in the |x| steps shifts things at zero: the paired units near ±0.05 leave a small positive offset, roughly 0.065, which adds a bit of bias to H. I'm considering whether to subtract a constant to cancel it, and then checking how the step sum behaves at larger values of |x|.

Checking the sum of the smoothed steps at x=0, I'm getting a residual of about 0.065 from both sides, so |x| comes out near 0.065 rather than zero. That offset feeds into H as a small constant bias.

I'm checking the absolute-value steps at x=0.1: the smoothed sum comes out near 0.111 instead of 0.1, so the smoothing overshoots near the kink.

The rounding error at x=0 is about 0.018 in H, which works out to a hover-target shift of only about 0.036 in y, so it's negligible. I could subtract a small constant to cancel it, but it isn't worth the extra unit. The real question now is how far out in |x| the step approximation actually needs to be accurate.

Now I'm setting up the contact-case unit as a linear function of vertical velocity, with a sign-gated offset, and I'm defining a combined contact indicator from the two landing-leg signals.

Now I'm setting the layer-2 gain to 100, so the binary units saturate on small input differences. For the hover target, I'm expanding the absolute-value steps into a weighted sum of the layer-one units plus a constant from their midpoints, with the linear hover term added on top.

Now I'm checking the output-side sign handling: the contact term has to push the decision toward a fixed value when the legs touch, but a plain additive offset would also bias it when they don't, so I need a gating form that only switches on at contact.

Setting the contact gate's sharpness is the next step: I'll use a steeper tanh, tanh(10·(L+R) − 5), so the non-contact residual drops to around 4.5e-5 once scaled by the gate weight. Then I'm picking a gate weight of 4 in feature units, which is large enough to cover the typical range of H − A.

Still working out the layer-2 weights: I'm scaling the hover-minus-angle sum by G=100 and writing out the coefficients for the p1, p2, and p3 units.

Now I'm writing out the layer-2 weights for the remaining units: p4 gets its vertical-velocity term plus a contact penalty, and the two fire-direction units q1 and q3 get their angle-based coefficients gated off at contact.

Settling the contact-gate weight at 3 so the residual stays tiny, and recomputing the layer-2 coefficients with the new value.

Still writing the layer-2 weights: I'm filling in the coefficients for p1 through p4 and the fire-direction units q1 and q3, with the contact gate weight at 3 and the uV term set to -100.

Now I'm checking the sign convention for the landing-leg term, using the velocity scale of 0.2 so the contact unit stays close to linear, which makes the vertical-velocity contribution -250 times the unit. Then I'm setting the lateral unit's scale to 0.2 as well, with the angle and angular-velocity weights at -0.1 and -0.2.

Still working out the first-layer units: the step units for |x| at each threshold, plus the sign-gated contact unit.

Now I'm laying out the output layer: the main action logit is ten times the sum of the four contact-gated units minus three, and the two side logits are five times their fire-direction units.

Now I'm checking the contact-case residual on p4: with the gate near 3, the vertical-velocity term stays well above zero across the range of uV, so the unit holds at +1 in flight. For p1 in contact, I'm splitting the gate into a constant part and a uC-weighted part, which gives a bias of about 150 plus 150·uC after scaling by G.

Settling the contact term for p4 now: the bias works out to 295, and the uC weight is negative, so the unit stays firmly on when the legs are off the ground.

Now I'm checking the contact-gate constants for H: the offset term that feeds p3's bias works out to 21.5 before the 300 from the contact gate, so that unit sits at its intended threshold.

Checking the rounding and scale constants for the hover and landing units, and whether the step units and gate weights leave any meaningful error.

Still laying out the first-layer indices, with the threshold step units for |x| and the sign-gated contact unit assigned to their slots.

Still filling in the layer-1 weights, and now I'm setting the layer-2 weights for p1: the step units get 1.375, the left-side and right-side terms get 500, and the thrust term gets -20 and -500.

Still writing the layer-2 weights: I'm filling in the p2 through q3 coefficients, with the gate weights and bias terms set for each unit.

Still writing the output-layer weights: the four contact units feed the main action logit, and the two fire-direction units drive the side logits. Now I'm checking the angle-target expression that feeds the controller's correction term.

Now I'm checking the angle and hover corrections against the controller's formulas, and making sure the scaling constants line up with what the network computes.

