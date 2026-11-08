I'm working out a network that reproduces the Gymnasium heuristic controller. The target angle is a weighted mix of the state's first and third values, clipped to a small range, and the hover target scales with the absolute value of the first state. From there, the two control signals are proportional corrections on the angle and hover errors, with a damping term on the angular velocity.

Still working through the abs(s[0]) term for the hover target, which needs a nonlinear approximation since tanh units can't produce absolute value directly.

I'm now trying to approximate |s0| with tanh units, since a single tanh can't be even. Pairing shifted tanh terms, tanh(k(x−a)) − tanh(k(x+a)), gives a bump that behaves like |x| once |x| exceeds a, so a weighted sum of such pairs should cover the range [-1, 1].

I'm now weighing a staircase of step-like pieces, each a pair of tanh units, to approximate |s0|, and I'm considering whether the layer-2 units could provide the nonlinearity directly instead, since there are plenty of units to spare and the hover target doesn't need high precision.

I'm setting up the layer-2 indicators: one unit should fire when hover_todo exceeds |angle_todo|, but computing |angle_todo| needs its own nonlinear approximation, so I'm checking how to build that.

I'm combining three indicators in the second layer: one for hover_todo minus angle_todo, one for hover_todo plus angle_todo, and one for hover_todo exceeding 0.05. Their conjunction feeds a large logit for action 2 in the output layer, while the angle_todo thresholds drive the logits for actions 1 and 3.

I'm setting the logits so the ordering holds: action 2 wins when all three conditions hold, action 3 or 1 wins when the angle error is clearly negative or positive, and a small constant on action 0 covers the rest. Now I'm checking the leg-contact case, where the angle error is zero and the hover error depends only on the vertical velocity.

Now I'm considering the leg-contact case: when both legs touch down, the angle target drops out and the hover target is just a damping term on vertical velocity. I'm checking whether the contact input can gate the layer-2 units so the angle branches switch off, and whether the missing hover-target term matters much when the lander sits near the pad.

I'm working on the leg-contact case now, where the next unit needs to switch on only when a leg touches down and the vertical velocity is strong enough to call for a main-engine burn. The plan is to gate a tanh unit on the contact sum so it stays off when no legs are down.

Writing the contact indicator as a saturating tanh over the summed leg signals, so it sits near -1 with no contact and near +1 once either leg touches down.

I'm weighing whether the linear features can be carried through the first layer at a usable scale. I'm considering tanh units on the combined hover and angle errors, so that the second layer can threshold them directly.

I'm defining two first-layer features: one tanh unit that carries the clipped angle target from the first and third state values, and another that carries the linear part of the angle error, scaled down so the tanh stays in its near-linear range.

I'm now aiming to fold the linear parts into single first-layer units, so the five threshold conditions can be checked in tanh's monotone region rather than through separate approximations.

I'm weighing how the nonlinear pieces, the clipped angle term and the absolute value of the first state, will combine with the linear parts in each threshold. The concern is that tanh compresses large linear values, though the bounded nonlinear terms should only shift results near the decision boundaries.

I'm weighing whether to skip the layer-1 indicators entirely and just use small gains so the linear pieces stay in tanh's near-linear range, keeping precision acceptable. I'm now turning to the |s0| term, looking for a way to get 0.275|s0| out of tanh units for the hover target.

I'm trying to approximate |s0| with a pair of shifted tanh terms, but the sum I wrote comes out near zero at large |x| and negative near the origin. That makes it an even bump rather than the growing absolute value I need, so the hover target will need a different construction.

I'm settling on a few tanh pairs with breakpoints near 0.1, 0.25 and 0.45 to build a staircase for |s0|, since a single pair can't cover the range well. The error from that stepping is small enough to be fine for the hover target.

I'm now building |s0| from a row of step-like tanh units, one for each small increment on each side of zero, since the summed steps can track the absolute value across the expected range.

I'm building |s0| as a staircase of steps, one for each 0.1 increment out to about 0.55, on each side of zero. Each step is a rescaled tanh, and summing them should track the absolute value across the expected range.

I'm checking the zero-input bias of the staircase: with six steps per side at a 0.05 starting breakpoint, |s0| comes out near 0.039 at the origin instead of 0. Multiplied by 0.275 that's only about 0.011 in the hover target, which I'll accept, and a steeper gain would shrink it further.

Now I'm settling on the first hidden unit, which passes the scaled sum of the first and third state values through a tanh to give the clipped angle target.

I'm settling the layer-1 contact unit, which uses a tanh over the summed leg signals so it switches on when either leg touches down.

Now I'm setting the hover term: a 0.275 multiple of the absolute first state, built from a staircase of tanh step units on each side of zero, plus the linear hover unit scaled up.

I'm now sizing the hover target from the staircase units plus the linear hover term, and writing the angle target as a scaled sum of the first hidden unit and the angle-error unit. Next I'm considering a sharp gain on the layer-2 thresholds, with a note that the contact unit should gate the angle branches when I get to the in-flight case.

I'm writing the layer-2 pre-activations now, starting with a unit that fires when the hover target exceeds the angle target and no leg is in contact, with the contact unit's signal subtracted off as a gate.

Writing the layer-2 pre-activations now, with each unit's threshold expressed in terms of the hover and angle targets and the contact gate.

I'm setting the layer-2 gate constants now, with the contact unit's signal steering the conditions that apply only in flight, and I'm checking that the main-engine branch stays off when no leg touches down.

I'm setting the gate constant at 150 so the contact term can switch off the in-flight conditions, and checking that the leftover residue at two-leg contact is only about 0.02 in the threshold, which is acceptable. I'm also settling on the main-engine branch, which fires when the contact-gated condition holds or when the three hover conditions are all true.

I'm settling the logit for the main-engine branch: it adds the three layer-2 hover conditions and the contact-gated flag, with a bias that keeps the output off unless all three hold or contact fires.

Now I'm setting the logits for the three output heads: the main-engine logit combines the hover conditions and the contact-gated flag, while the other two stay near their baseline values.

I'm checking the sign conventions against the heuristic: angle_todo above 0.05 maps to action 1, and angle_todo below -0.05 maps to action 3, so the orientation branches line up with the layer-2 gates I've set.

I'm reading through the heuristic's contact-case rules now: the angle target is zero, and the hover target is just a damped term on the vertical velocity. I'm also checking that the approximations hold up, starting with the linear part of the leg-angle unit.

Still working through the layer-2 weights, now with the gain set to 100. I'm starting to write the first unit's connections to the hidden layer.

Still writing out the layer-2 weights for the orientation units, filling in the hover and angle coefficients and the contact gate.

Still writing out the layer-2 weights, now checking the two remaining orientation units and the main-engine gate against the contact term.

Checking the contact-case layer-2 unit next: with G=150 the gate term stays around -280, and the steps and the u2 term are too small to flip it.

Setting the layer-3 weights now: the main-engine output gets weights on the three hover conditions and the contact-gated flag, plus a bias of 10.

Still checking the layer-3 output weights for the main-engine branch and the orientation heads, and I'm going over the edge cases: the lander resting on the ground, the reset step, and whether the angle branch misbehaves once a leg is in contact.

Checking the zero-contact bias on the hover branch: the pre-activation comes out to -5 minus the 20 and 500 weighted terms, which matches the scaled form I expected. Now I'm writing the first-layer weights for the angle and velocity inputs.

Still writing the first-layer weights, now filling in the hover-step entries and the bias for the contact-gate unit.

Checking the tanh value at -2.5 and the gate term: the residual is about 0.002, so it's negligible, and the layer-2 setup is ready to write.

