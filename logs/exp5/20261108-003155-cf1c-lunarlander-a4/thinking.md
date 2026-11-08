Next I'm writing a heuristic controller for the lander, with the target angle built from the current angle and angular velocity, clamped to a small range, and a hover target that scales with horizontal offset.

Still writing the heuristic's decision logic...

Still working through the hover logic: the vertical target scales with horizontal offset, and the contact case zeroes the angle term and only damps the descent. Now I'm checking how to approximate the absolute value and the clipping with smooth saturating units.

Now I'm working out how to approximate the absolute value |x| with tanh units, since a single layer can't produce it directly. The idea is to sum several shifted tanh terms at different offsets so their combination traces out |x| over the relevant range.

Working out how to approximate |x| with tanh units: the step at each offset d_i gives a contribution that is flat beyond d_i and dips inside it, so summing several of them should trace out the absolute value over the relevant range.

Still working out how to approximate |x| with tanh units: a staircase of shifted steps could trace it out, and with 64 units there's room to spare. Since the hover target only needs 0.275|x| and x stays small near the center, a few smooth units might be enough.

Now I'm considering splitting the decision by the sign of the horizontal offset, so the hover target can be handled for each side separately and combined with OR-style logic in the next layer.

Now I'm weighing whether a single output layer can express the AND/OR decision logic, or whether the comparisons need to be split across units first.

Now I'm checking whether the argmax-style decision matches the original rule: if the hover term beats the absolute angle term and clears the threshold, the output is the hover action, and otherwise the sign of the angle term picks between the two side actions.

Checking that the argmax over the logits [0.05, A, H, -A] reproduces the original rule, the tie cases all come out the same. Next is the |x| term, which needs a tanh approximation, and the contact gating.

Now I'm working out how to gate the angle term on leg contact: a pair of saturating tanh units, one with the contact input added and one with it subtracted, so that without contact their sum recovers the linear angle term and with contact it cancels to zero.

Now I'm working out how to gate the angle term and the position term on leg contact: the contact signal goes through a tanh unit so it saturates near one when a leg touches down, and a pair of tanh units with opposite contact offsets cancels the gated term. The vertical damping part stays ungated.

I'm settling on an argmax over the logits [0.05, A, H, -A], since that reproduces the original rule in every tie case. Next I'll gate the angle and position terms on leg contact with saturating tanh pairs, while the vertical damping stays ungated.

Now I'm figuring out the layout of the first layer, keeping the inputs in the near-linear range of tanh so the approximation stays accurate.

Still working out the first layer: a clipped horizontal term from x and vx, and a small-scale linear term from the angle and angular velocity, so both stay near the linear part of tanh.

Gating the angle and position terms on leg contact can be done in the first layer: a pair of saturating tanh units, one with the contact term added and one with it subtracted, so that with contact they cancel to near zero and without contact they pass the linear term through.

Gating the clip unit on contact works by pairing two saturating tanh units with opposite offsets, so with contact they cancel to zero. The same pattern can gate the whole layer-1 set, so layer 2 can just pass values through with a small scale. Next I'll think about whether layer 2 could handle the absolute value more directly.

Now I'm approximating |x| in the first layer, using pairs of shifted tanh units whose difference forms an even bump around the center.

I'm trying to approximate |x| with tanh units, but a quadratic from a pair of shifted tanh terms won't capture the kink at zero. A smooth stand-in like sqrt(x² + δ²) would work better, and I'm wondering whether the |x| term matters much, since it only raises the hover target when the lander drifts far from center.

Still working out the staircase for |x|: each step unit tanh(k(x−d)) plus tanh(k(−x−d)) sits near zero once |x| passes its threshold d and near −2 while |x| is still below it, so summing them over several thresholds should trace out the absolute value.

Now I'm approximating |x| as a staircase: ten tanh step pairs at midpoints from 0.05 to 0.95, summed with weights of 0.05, so the total traces the absolute value. With a steepness of about 10, the transitions are roughly 0.1 wide, which should smooth out the staircase, and the rounding near zero should be small.

Still working out the gating for the |x| staircase: each step gets a ±B·c offset so that when a leg is in contact, every step term saturates to the value that makes it cancel, and the constant contribution drops out with them.

Working out the gating for the |x| staircase: the constant offset and each step term both collapse to zero when a leg is in contact, so the position term drops out cleanly there.

Still settling the layer-2 design: each of the two output signals gets its own tanh unit, scaled small so the double-tanh distortion stays minor.

Setting α to 1 is the simplest choice, since the comparisons between H and |A| survive any monotonic squashing, and I can fold the threshold into the bias as tanh(0.05α)/α.

Now I'm settling the scale of the logit layer: with α set to 1, the comparisons between H and |A| hold under the tanh squashing, and I'm working out the typical ranges of the two terms to confirm they don't saturate too early. Next I'll start on the linear-regime layer-1 units for the angle terms.

Still working through the gated linear units: each pair of tanh terms with opposite contact offsets should give roughly 2εz without contact and cancel to zero with contact, and the relative error at moderate inputs stays around a percent.

Still working through the clip unit: gating it on contact with a pair of saturating tanh units, the clip term should contribute about 0.1 to the angle logit when there's no contact.

I'm checking whether a smooth clip is a good stand-in: tanh(2.5w)·0.4 is softer than a hard clip at ±0.4, though close enough to keep. I'll leave the clip as it is and move on to whether the heuristic itself could be tuned for better results.

I'm recalling that the heuristic usually averages around 200-250 with occasional failures, so I'll keep it as is and skip any risky changes. I also want to check the sign of the angle target: the main engine impulse points opposite to the tilt, so a positive angle target pushes the lander back toward center, which matches the heuristic's intent.

Still working through the gating for the |x| staircase: each step unit gets a ±B·c offset so that under contact both saturate to −1 and +1, making each step term −2 and the constant term cancel.

Checking the staircase values at x=0: each step's tanh only partly saturates at the small thresholds, so the sum at the center comes out a bit off from the ideal.

Checking the step terms at x=0 once more: the smallest thresholds only partly saturate, so the total comes out to about 0.065 instead of 0, and that small offset feeds into the hover term as roughly 0.018.

Still checking the step sum at x=0 with a sharper steepness of 20: the nearest thresholds give most of their contribution, and the far ones saturate fully, so the ripple from the staircase should stay small enough to ignore.

I'm settling the constants for the |x| staircase: a steepness of 20 cuts the rounding at zero to about 0.024, and the gate offset needs to be around 60 so the steps stay saturated during contact. Then I'm tallying the layer-1 units: two for the clip, two for the linear angle terms, two gated y terms, one for vy, and twenty for the staircase, 27 in total.

Still working through the clip unit weights: I'm setting the two clip units to 1.25 and 2.5 on position and velocity, gated by contact with a large offset, so the clip term adds about 0.1 times their sum to the angle logit.

Still working through the unit layout: the two alin units are gated with ±30 offsets so the angle-velocity term cancels under contact, and their sum scaled by 10 gives the linear contribution to the angle logit.

Still writing the layer-1 unit tables: the y-gated pair cancels under contact, the vy unit gets a 20x scale, and I'm laying out the twenty staircase units with their shifted thresholds.

Still writing the layer-1 unit biases for the staircase: the a-units get −1 through −19 and the b-units get +1 through +19. Under contact the step weights cancel out, so the constant offset and the step terms net to zero.

Still writing out the weights for the second layer: the first unit combines the two angle terms, and the second combines the hover terms with the staircase sum.

Still checking the weights on the final layer: the sign convention for the output unit has to match the layer-2 combinations, so I'm confirming that the angle-side and hover-side units feed in with the right signs.

I'm checking the landing-contact cases and confirming that the smooth clip stays harmless within its range, and that the angle and vertical inputs use the observation scaling the controller expects.

Still filling in the first-layer weight rows for the clip, angle, and staircase units.

Still writing out the weight rows for the first layer.

Still writing out the weight rows for the first layer, one entry at a time to keep the counts straight.

Still filling in the first-layer weight entries for the clip and angle units.

Still filling in the first-layer weight entries for the staircase and gate units.

Still writing the weight rows for the first layer and the bias vector, one entry at a time.

