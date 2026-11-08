I'm sketching a network that reproduces the lunar lander heuristic, starting from its angle target: a mix of the lander's angle and angular velocity, clamped to ±0.4.

I'm working out the remaining control rules: the hover target scales with horizontal offset, the angle and hover corrections are computed from the lander's state, and contact with the legs overrides both. Then the action is picked from whichever correction is largest, and I'm mapping that onto a network that can approximate the clipping and absolute values.

I'm trying to approximate the absolute value |x| with the network's units, since tanh alone won't give it directly. One idea is to build it from thresholded ramp-like pieces, and I'm weighing whether layer two can handle that nonlinearity given the layer-one approximations.

I'm considering a staircase of thresholded tanh units to approximate max(x, 0), with the spacing kept fine enough that the result stays smooth. Precision on the hover target matters less than getting the final decision right. So I'm leaning toward having layer two compute the decision quantities directly, like hover_todo minus |angle_todo|, and then pick the action from those comparisons.

I'm setting the logits so the output is an argmax over the heuristic's comparisons: a constant for the no-op, angle_todo and its negation for the side engines, and hover_todo for the main engine. Since the logits are linear in those quantities, the second layer has to carry them in its linear regime.

I'm noting that since the logits are linear in the decision quantities, I can drop the constant scale and compare the tanh-squashed values directly. Because tanh is monotone and odd, the comparison of hover_todo against |angle_todo| survives the squashing exactly, and the threshold at 0.05 becomes a fixed tanh value. So the output units are just u and v, with the no-op logit as a constant.

I'm adding a gate for the leg-contact case: a large negative term keyed to the two contact inputs, so the side-engine units shut off once the legs touch down. The side-engine logits would then drop below the no-op threshold, leaving the hover unit to decide.

I'm handling the contact case now: with the legs touching, the side-engine units should be pushed well below the no-op threshold, and the hover unit needs its own target based on vertical speed alone.

I'm using layer one to build a contact indicator from the two leg signals, so it is high whenever either leg touches down, and then I'll pick the hover input from that indicator.

I'm making the second layer's logit for the main engine the sum of the free-hover unit and the contact-hover unit, plus one, so whichever one is inactive sits near -1 and drops out. The side-engine units are already pushed to about -1 under contact, so they fall below the no-op constant, and the main engine only fires when the contact hover target clears 0.05.

I'm settling the contact gating so the inactive hover unit sits near -1 and drops out of the sum, and checking that the margin M is large enough that the leftover error stays well below the decision gaps. Next I need to set up the first layer so it gives me roughly linear versions of the lander's state.

I'm checking how accurate the layer-one approximations need to be, so I'm picking a small tanh scale where the state values stay in the near-linear range, and I'm listing the input ranges for x, y, velocities, angle, and angular velocity to confirm that works.

I'm checking whether the first-layer approximations can be made nearly linear by using a small tanh scale, since that would let the second layer read the state values almost exactly. I'm also weighing whether combining the second layer's terms directly would avoid needing a separate first-layer stage at all.

I'm using a tanh of the blended angle and horizontal offset as a soft clip for the angle target, since that gives a slope of one near zero. I'm also thinking about whether a sharper scaling would track the hard clip more closely.

I'm settling on an approximation for the clipped angle target and the vertical-speed part of the hover target, so the first layer gives me roughly linear versions of each piece. The remaining terms, the half-weighted height and vertical velocity, I'll carry as a linear unit.

I'm now trying to approximate |x| with a sum of shifted tanh terms, since the layer can't take an absolute value directly. I'm also setting the contact indicator g to a sharp tanh of the two leg signals, so it sits near -1 with no contact and near +1 once either leg touches down.

I'm now approximating |x| with a staircase of shifted sigmoid-style units, since a sum of thresholded pieces can trace the V shape across the input range.

I'm checking the staircase approximation of |x| with spacing 0.1 and steepness 20, and working out the sigmoid contribution at x=0 from the nearest thresholds.

I'm checking the staircase approximation of |x| and noting that with ten steps per side the error at zero is about 0.024, which is small enough for a soft hover target.

I'm checking the staircase approximation of |x| against the 0.275 weight on H, and considering whether to fold that absolute-value term directly into the second layer rather than building it separately.

I'm settling the layer-two scale so the comparisons still come out right, and checking that tanh doesn't saturate to identical values in float32 for large inputs.

I'm settling on a layer-two scale of 1 and checking how the gating constant M keeps the inactive units near -1, including the error term from H.

I'm checking the gate scaling: the contact indicator maps to a half-and-half offset, so M/2 plus (M/2) times g gives the gated pre-activation.

I'm choosing the layer-two scale and the gating constant so that inactive units sit near -1, and I'm settling on K=20 for the contact indicator so the leftover offset is negligible. The linear units in layer one will use a small scale so their tanh stays close to linear, and I'm checking the pre-activation range for the angle and angular-velocity terms. Then I'll decide whether the heuristic itself is worth refining.

I'm noting that the heuristic typically scores around 200-250 on average, so I'll keep it as is rather than risk tuning it without testing. I'm also checking that the lander's landing condition works out, since once it comes to rest the environment awards the final bonus.

I'm weighing whether a two-term tanh combination would approximate the clipped angle target better than a single tanh scaled to 0.4, since the single version pulls mid-range values in a little too far.

Writing the weight matrices for the first layer now, starting with the scaled tanh that approximates the clipped angle target.

Still writing the weight matrix entries for the gating and step units...

I'm checking the staircase sum at x=0, where the positive and negative sides together come to about 0.024, and then I'm folding the absolute-value term into the hover target as 0.2 times the head unit plus 50 times the staircase sum.

Still writing the weight entries for the second layer...

Still writing the weights for the second layer, filling in the entries for the velocity and contact units...

Still working out the weights for the second layer, filling in the entries for the gating and velocity units...

Setting the output-layer weights now: the first unit carries the left-thrust comparison through the hover and angle terms, the second the mirrored right-thrust case, and the third and fourth take the gated hover and contact pieces. The final output unit is a plain identity on the negative-angle branch. Checking the ordering against the heuristic's branches, the hover-first case wins whenever the hover term beats the absolute angle, and the angle-correction case follows only when it doesn't, so the tie cases come out as the heuristic does.

Checking the contact case: the angle units are pinned near -1 there, and the free-fall term stays at -1 unless the horizontal speed is large, so only the contact-hover unit can fire. I'm also confirming the constant tanh(0.05) ≈ 0.0499584 for the bias, and I'll write out every weight entry individually.

Wrapping up the weight calculations now.

