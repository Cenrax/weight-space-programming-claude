I'm setting up a network that reproduces the lunar lander's heuristic controller, starting from its angle and hover target formulas.

I'm working out the decision logic next: hover takes priority when it beats the angle correction and clears the threshold, otherwise the sign of the angle correction picks the side thrusters, with the leg-contact case zeroing the angle term. I also want to replace the clip on the angle target with a tanh unit that saturates smoothly at ±0.4.

I'm now sizing the tanh units for the remaining linear pieces, using a small slope so they stay close to linear, and I'm checking whether the angular velocity term could be large enough to break that assumption. Then I'm turning to the absolute value of s[0], which a single tanh layer can't produce directly, so I'm looking for a workable approximation.

I'm looking at how to get |s[0]| out of tanh units, since a single tanh layer can't produce an absolute value directly. I'm considering an even function built from shifted tanh terms, something like tanh(bx - a) plus tanh(-bx - a), as a smooth stand-in for the hover target's abs term.

I'm settling on an even function for |s[0]| built from two mirrored tanh terms, which gives a curve that rises with |x| and saturates. I'm checking whether it fits the 0.55|x| target over the viewport range, or whether a stepped sum of sigmoid-like units would be a better approximation.

I'm weighing a step-sum approximation for |s[0]|: a set of shifted tanh units spaced about 0.2 apart, which together give a smooth, piecewise-linear rise. Layer two would then take those units and do the comparisons.

I'm checking how accurately the step-sum approximation tracks the abs term, and whether that precision matters for a heuristic that already averages around 200 to 250. I'm also turning to the leg-contact gating, where a large-weight unit on the sum of the two leg signals should push angle_todo toward zero and switch hover_todo to the vertical-velocity term.

I'm working out the leg-contact gate as a single combined unit, G = tanh(10·(s6+s7) − 5), which is near −1 when off and near +1 when on, so (G+1)/2 gives a clean 0/1 switch. Since layer two only sees layer one's outputs, that gate has to be built in layer one, and I'm now planning how layer two should combine the angle and hover terms to produce the action.

I'm now writing out the decision logic as indicator units: one for whether hover beats the absolute angle correction, one for hover clearing the threshold, and separate ones for the left and right angle thresholds. The main action needs all the hover conditions true at once, so I'm weighing how to set the layer-two logit weights and offset so that only that combination wins.

I'm settling the output layer: the main-engine logit will be a scaled sum of the three hover indicators shifted so only the all-true case is positive, while the side-engine logits switch on the left and right angle indicators, with the nop logit held at zero.

I'm working out how the main-engine logit gets scaled so it only wins when all three hover conditions hold, and how the side-engine logits stay below it. I'm also checking that the leg-contact case still resolves to nop.

I'm now handling the landed case separately: when both legs touch down, the hover target becomes -0.5·s3, so main fires only if s3 is below -0.1, and otherwise the output should be nop. My plan is to use units dedicated to landed mode and push the flight units off with large biases whenever the lander is on the ground.

I'm setting the landed-mode units to switch the flight units off with a large negative bias on the leg gate, and I'm splitting the main-engine indicator into a separate landed unit so hover uses the vertical-velocity term only when both legs are down.

I'm checking whether the landed-mode main-engine condition works out, where the hover target switches to the vertical-velocity term and only fires when that term clears the threshold.

I'm laying out the first hidden unit, a tanh of the x-position term, with the angle and hover inputs still to be combined into the layer-two pre-activations.

I'm working out the first hidden unit as a scaled tanh of the angle-target inputs, which stays close to linear near zero and saturates like the clip. Then I'm considering a second unit for the angular-velocity terms with a small gain so it approximates a linear term, and checking how precisely that holds.

I'm settling on the scale for the angular terms, weighing a gain of 0.25 against 0.5. A smaller gain keeps tanh closer to linear over the expected range of the angle and angular-velocity inputs, so I'll go with 0.25.

I'm working out the linear part of H as a tanh unit on the s1 and s3 terms, keeping its slope near one so it stays close to the target formula.

I'm checking the magnitude of the top-of-screen height term, which comes out near 1.4, and estimating how far tanh compresses it at a scale of 0.2. I'm settling on that scale, and I'm noting that the |s0| piece still needs a step-based approximation.

I'm approximating the absolute value |x| with a sum of shifted sigmoid steps, placing thresholds about 0.2 apart on each side of zero. I'm checking how much error that leaves at x=0, where the sigmoid tails add a small constant that I'll need to correct for.

I'm checking the bias that the sigmoid tails add at x=0, which comes out around 0.13 in total, and noting that a smooth sum of shifted sigmoids spaced 0.2 apart behaves like a softplus, so |x| is approximated by softplus(cx)/c on each side.

I'm weighing whether to subtract the constant bias from the softplus approximation of |x|, since the offset at zero is small in H, or to tighten the spacing and raise the sharpness so the ripples disappear.

I'm weighing whether to model the |x| term as a staircase of sigmoid steps, which would make the hover target respond to horizontal drift, or to drop it since the exact centering may matter less than I think.

I'm checking the constant offset this smoothed |x| term leaves at zero. With ten sigmoid steps per side, the sum at x=0 comes to about 0.014, which is small enough to ignore for the hover target.

I'm checking how much the sigmoid staircase error at x=0.1 and x=0.5 matters, working through the step sums term by term to see how far the approximation drifts from the true absolute value.

I'm settling the staircase thresholds for the absolute-value term, moving them to 0.05, 0.15, and so on up to 0.95 so the steps line up with |x| away from zero. At the origin the sigmoid tails leave a small bias, about 0.065, which times the 0.275 weight is an error of roughly 0.018 in the hover target, so I'm accepting that. The remaining question is whether ten steps per side, twenty units in all, is worth the cost.

I'm weighing whether five sigmoid steps per side spaced 0.2 apart are enough, or whether to use twenty units at spacing 0.1 so the bias at zero stays smaller. I'm settling on the finer staircase, with its thresholds at 0.05, 0.15, and so on up to 0.95, and I'm rewriting each step as a tanh so the sum stays linear in the hidden units.

I'm converting the absolute-value term into 20 tanh units spaced at 0.1, then folding its constant offset into the output bias, and I'm adding a gating unit G from the leg inputs.

I'm working out the remaining layer-one units: the leg-contact term, the gating unit, and the vertical-velocity unit for landed mode, each as a scaled tanh of the state inputs.

Now I'm turning the hidden-unit layout into the actual output formulas, mapping the angle term A to the first two units, the hover term H to the linear unit plus the summed absolute-value steps, and the landed hover term to the vertical-velocity unit.

I'm settling the gate's scale: with the gating term in flight already near -1, I'll set its multiplier to 300 so the flight-time pre-activations stay well clear of the threshold.

I'm writing out the layer-two pre-activations for each unit as gain-scaled combinations of H, A, and the gating term, then expanding H as a linear combination of the layer-one hidden units.

Still writing out the layer-two coefficients...

Checking the output-layer weights for the four flight-mode units against the landed case, and confirming the flight-time pre-activation stays negative.

Now I'm settling the output layer: the first logit stays at zero, the second and third read from the two side-steering units, and the main logit takes ten times the sum of the four flight-mode units minus 1.5.

I'm checking the sign convention for the angle target: since the heuristic uses the horizontal velocity term with a positive weight, I need to confirm which direction a positive angle target means in Box2D's counterclockwise convention.

I'm confirming the sign convention: a positive angle means a push toward negative x, so a rightward offset calls for tilting positive. I'm checking the heuristic's angle-to-side mapping against that, since it is meant to fire the left engine when the angle target exceeds 0.05.

Still writing the heuristic's control logic...

Now I'm wrapping up the controller's final mode logic and checking the heuristic's branch order against the layer-two outputs.

Still writing the first-layer weight entries...

Still filling in the layer-two weights for the remaining columns...

Still filling in the layer-two weights for the remaining columns...

Now I'm checking the offset that the smoothed absolute-value sum leaves at zero, and it comes out to about 0.018 in the hover target.

Still checking the output-layer weights and the edge case where the hover and angle terms nearly tie.

Checking the leftover tanh-to-linear mapping and the output-layer weights for the main, side, and nop logits, confirming that the partial-activation edge case shifts the effective hover threshold only slightly.

