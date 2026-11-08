I'm sketching a Breakout policy from known RAM addresses for the ball's position, the paddle, the block bitmap, and the score.

I'm weighing a paddle policy that moves toward the ball's x-position, with FIRE actions for serving, and I'm trying to pin down the paddle offset and the network's scaling so the tanh saturates a few pixels off the ball.

I'm working out the paddle's deadzone with two units, one for moving right and one for moving left, so the fire action wins when both are near -1. I'm also noting that the ball's position may be unreliable when it isn't in play, but holding LEFTFIRE covers that case, and I'm setting predictive tracking aside for a simpler version first.

I'm settling the layer-two mapping: a scale of 3 to push the units near ±1, RIGHTFIRE and LEFTFIRE taking the two paddle units, and a zero bias on FIRE so it wins when both are saturated low. I still need the paddle-to-ball offset, which I'll calibrate from a first rollout's RAM traces rather than guess.

I'm settling on a guess of 8 for the paddle-to-ball offset, the paddle's center, with a deadzone of 2, and I'll check both against the first rollout's traces.



---

The paddle's range and the ball's x-positions check out, but one episode hit the 5000-step cap, so I'm wondering whether the ball got stuck somewhere. My leading guess is that after a life is lost the paddle drifts left and never fires, but the agent should serve with FIRE, so I'm now suspecting it's choosing NOOP in the deadzone instead.

I'm now checking the deadzone case: when the ball isn't served and the paddle sits there, NOOP wins the tie with FIRE, so the agent could idle forever, which likely explains the truncated episode. I'm considering bumping FIRE's weight slightly to break that tie, and I'm still trying to pin down the paddle and ball widths from the coordinate ranges.

I'm now weighing how to improve the tracking itself: the paddle may not be tracking the ball well because the ball moves fast, and without a velocity signal the agent can only react to its current position. I'm considering whether the paddle offset should be tuned, and whether the ball's velocity might be stored in RAM at an address I could find, though that would be hard to pin down from the traces I have.

I'm thinking about whether I can find a velocity signal in the RAM, but the traces are subsampled every 20 steps, so consecutive frames aren't visible and any change in a direction byte would be hard to pick out from the noise. A simpler route is to tune the deadzone down to 1, adjust the gain, and fix the FIRE tie. I'm also weighing the paddle offset, and I can't settle it from memory of other RAM-based Breakout policies.

Next I'm looking at whether a velocity signal could be pulled from the RAM bytes, but I'd have to sweep the traced bytes to see whether any of them toggles with the ball's direction. I'm also keeping the paddle deadzone at 0.01 so FIRE wins by default, which means the paddle serves right away after a life is lost.

I'm going to try a rollout with the deadzone at 1 and the FIRE tie fixed, along with a paddle offset of 6, though changing both at once makes it harder to tell which one helps.

Checking the rollout results for the deadzone change.



---

The 28.5 score is worse than the 42 before it, which could mean either the c=6 change or the m=1 change caused the drop, since the two may be oscillating. I'm leaning toward a higher c being better, so I want to look at the traces right before the life losses, starting with the worst one.



---

I'm checking the step 171 state: the ball is at y=204, below the paddle, so it was lost. The ball sits about 10 units left of the paddle, which suggests a miss on the left, though a fast ball overrunning the paddle is also possible. I don't actually know the paddle's y position and am only guessing it's around 185–190. I'm also comparing the scores from the two earlier settings, 42 for one and 28.5 for the other.

I'm working out the paddle's width from the observed ball and paddle ranges, guessing the paddle may be around 10 to 16 pixels wide and that the ball is only about 2 pixels across. That would put the center offset somewhere near 7, which could explain why the larger offset scored so much better.

I'm planning one more rollout with a larger center offset (c=10, m=2) to test whether it beats the c=8 result, and I'll keep the remaining rollouts for a follow-up or a final check.



---

I'm noticing the paddle often trails the ball by 20 to 30 pixels when the ball moves fast sideways, since tracking its current x alone always lags. That points toward needing the ball's velocity to predict where it will be. I'm not sure which RAM byte holds the ball's direction in Breakout, so I'm considering whether there's a way to handle this without velocity at all.

Paddle speed versus the ball's horizontal speed might explain the lag, since a steep bounce could outpace the paddle's roughly 8 to 12 pixels per step. I'm recalling a RAM annotation list with ball and paddle positions but no velocity byte, so I'm considering tracing candidate bytes across a rollout and checking which ones shift with the ball's direction.

I'm weighing whether to trace roughly 16 candidate RAM bytes near the ball's position, since a velocity byte would show a small value pattern near 0 or 255. That's a guess, though, and it uses one of my two remaining rollouts, leaving only one to test a predictive policy.

I'm trying to recall the Breakout RAM layout, with bricks in the low bytes, the paddle near 72, and the ball's x and y at 99 and 101. My memory of a direction byte around 103 or 104 is shaky, so I'll trace a spread of nearby addresses, including 70, to see which one looks like a small signed velocity.

I'm guessing the velocity is stored near the ball's position, so I'll run the current best policy with the fire fix to get another read on that setup.



---

Checking whether the byte 103 and byte 105 values line up with horizontal and vertical velocity, since the paddle positions aren't available to settle it.

I'm weighing whether byte 105 could be the vertical direction, with 1 meaning down and 255 meaning up, but the paddle positions aren't available to confirm it. Checking the ball's y-values against the steps where 105 flips doesn't settle it cleanly, since the sign pattern could reflect a serve or a lost life instead.

I'm checking whether the ball's vertical direction lines up with byte 105, comparing its values at different y positions. The readings near the left wall don't settle anything, and the y-values where byte 105 sits at 254 or 255 are spread out enough that they don't point to a clear pattern.

I'm trying to see whether an empirical trick could help: I could build a policy that uses one of the byte values as a dx predictor and test it on the last rollout, reverting to the c=8,m=2 baseline if it does worse. The velocity bytes are still ambiguous, so the test would have to decide it.

I'm reconsidering the serve: the ball would have started below the bricks and moved down, but by step 22 it may already have bounced, so that sample doesn't tell me much. I'm also wondering whether byte 107's alternating 65/129 pattern is a hit or speed counter, and whether bytes 102 and 104 flagging 0 or 128 might be fractional or direction bits.

I'm recalling a Breakout disassembly layout where the ball's x and y positions sit at RAM addresses $E3 and $E5, with the velocity bytes at $E7 and $E9. If that ordering holds, byte 103 would be horizontal velocity and byte 105 vertical, which fits the x-then-y pattern.

Now I'm looking at byte 105 against the vertical speed idea: the early-game values are mostly small, and the 254s that show up near the start suggest the ball's vertical step may be larger than I assumed.

I'm noticing that byte 105 has 254 values but never 2, and byte 103 is the same, so both look asymmetric, which seems odd. Maybe the values are two's-complement signs, so the magnitudes would show up differently than I expected.

I'm comparing the two bytes' asymmetric values, where 254 shows up without a matching 2, and wondering whether a sign-encoded magnitude explains it. I could test byte 103 as the dx predictor on the last rollout and fall back to the c=8,m=2 baseline if that does worse, though I put only around 0.6 on 103 being horizontal velocity.

Checking the ball's y-positions near the bottom of the screen, where byte 105 reads 1 at the lost-ball samples and byte 103 is mixed, since both readings are consistent with downward motion under either hypothesis.

I'm weighing ball y-positions near the brick line, where byte 105 sits at 254 or 255 across several samples, and asking whether a bounce there would settle the vertical direction. Measuring the distance from the paddle to the bricks, roughly 100 pixels against a 144-pixel horizontal span, could bear on how fast the ball moves vertically.

Checking whether the counts of byte 105's positive and negative values line up with the ball's downward and upward motion, and whether byte 103 fits that pattern just as well.

I'm weighing whether the wall bounces could settle which byte is horizontal velocity, since the ball's x-position changes sharply when it hits a wall.

I'm checking the ball's x-positions near both walls against the byte 103 and 105 values: at the left-wall samples 105 reads mostly 255 and 1, and at the right-wall samples it mostly reads 254. If 103 is horizontal velocity, the samples near the left wall should lean toward moving right, which is what I'm testing.

I'm checking whether byte 107 follows a pattern tied to the signs of bytes 103 and 105. Its values alternate between 64 and 128 offsets, so I'm lining up the early rows to see if the high bit matches either velocity byte.

Now I'm going through the rest of the byte 107 samples, lining up each row's 107 value against the signs of bytes 103 and 105.

Byte 107's high bits seem to track byte 103's sign: 0x80 shows up whenever 103 is 1, and 0x40 whenever it's 254 or 255. That could be a direction flag for horizontal motion, though the rest of the byte's low bits climbing with something like bounces is still unclear.

I'm noticing that byte 107's high bit tracks the sign of byte 103, and its low counter steps up on the transitions from positive to negative. If 103 is actually the vertical velocity, those flips would line up with paddle hits, so I'm checking whether that reading holds.

I'm checking the byte 107 counter against byte 103's sign: if the counter steps up when 103 flips from positive to negative, it could be counting paddle hits, which would make 103 the vertical velocity. A reset to 0 around the lost-life samples would fit that too.

Reading the two bytes as signed values: the ball's vertical step could be the upward 254s and downward 1s, but the magnitudes are lopsided, which makes me wonder if the encoding is sign-based. Byte 107's high bits seem to line up with byte 103's sign, which points to byte 103 as vertical direction. I want to check whether the low counter lines up with paddle hits before trusting that.

Checking the vertical hypothesis against the y-positions at each step: byte 103 reads 254 at step 22 when the ball is at y=142, and reads 1 at step 44 when it is at y=157, so it lines up with up and down motion. The next checks are the y-values at steps 66 and 88.

Now I'm checking the velocity sign convention for byte 105, reading values above 127 as negative. I'm sketching a predictive policy where the paddle aims ahead of the ball by a lead term based on its horizontal motion.

I'm working out a lead term for the paddle: aiming at the ball's x plus its horizontal velocity scaled by the time until it reaches the paddle, using the vertical distance and the vertical speed to estimate that time. A simpler fixed offset of 10 to 20 pixels in the direction of dx might capture most of the benefit, since a full prediction would be hard to represent in a two-layer network.

I'm working out how to gate the layer-1 units by each direction case, so the target-versus-paddle decision becomes a threshold on a linear function for each case.

I'm working out how the byte-103 sign would gate each direction case, using a large penalty so that the up and down cases separate cleanly under a linear threshold on the paddle decision.

I'm fixing the penalty scale at k=1 per pixel, so the -254 penalty on up-moving cases dominates the lin range. The down-case penalty is only -1, so I'll absorb that offset into the bias. Then each case's unit only activates when it's the active one, and the summed right and left logits reduce to a constant plus the active case's contribution, which I can compensate for with a bias term or a second layer.

Working out the lead term for the down-moving case: the target would be the ball's x plus its horizontal direction times the frames until it reaches the paddle, so I'm sketching how to express that in the layer-1 units.

I'm working out how to handle the magnitude of the left-moving velocity, since byte 254 and 255 would give different leads, and the product with the vertical distance can't be expressed linearly. One option is to give each of those two values its own unit so the penalty separates them.

I'm trying to design the dx inputs so each magnitude gets its own penalty, but a single linear penalty can't isolate 254 from 255 without also catching 1, so I'm dropping that approach and handing the logical combinations to the second layer instead.

Byte 105 looks asymmetric: the value 254 shows up often for leftward motion while the rightward value never gets a matching 2, which makes me suspect the byte encodes something beyond plain pixel speed. One idea is that the ball alternates between speeds of 1 and 2 on successive frames, with the sampling phase every four frames hiding the other value on the right side. I'd need to check whether the phase shifts over time to confirm that.

I'm noticing that byte 103 shows the same lopsidedness as byte 105: down is always 1, while up appears as either 255 or 254. I'm wondering whether the magnitude is really a fixed speed or encodes something else, so I'm treating it as sign-only for now, which makes the 45-degree assumption look questionable since paddle hit position should vary the angle.

I'm weighing a softer lead term, scaling the predicted landing x by a factor between 0.5 and 1, and I'm considering how wall bounces would make the paddle target overshoot, possibly handled with a reflected-target unit.

I'm settling on a lead of about 0.5 to 0.7 of the predicted landing offset, since the ball reflecting off walls near the bottom could throw off a full prediction. As the ball falls, the lead shrinks and the target converges on its actual x, so the paddle mostly tracks the ball by the time it arrives. The real risk is a wall bounce close to the paddle.

I'm settling on Yp around 180 as the paddle height, since the ball is lost somewhere near 195 to 205 and earlier y-values reached about 186. For the upward case I'll skip prediction and just track the ball's current x, because the ball has to travel up and back down with possible wall bounces, so a straight-line lead would be unreliable.

I'm settling the layer-1 gating so each direction case gets its own unit, with the target built from ball x plus a scaled lead term on the vertical distance to the paddle. Byte 103 acts as the case selector, and the penalty for a non-matching case is written as a weighted byte value, with the weight set to 1 for now.

I'm settling the down-case penalty: with G=2, the up-case unit gets a penalty of 2 at byte 254 and 508 at byte 1, so the wrong case is cleanly excluded. I'm now checking the down-case version of the same gating term.

Now I'm settling the down-case gating, with the penalty weighted so a byte of 1 shuts off the non-matching case, and I'm working out the rightward dx unit next.

I'm checking that the layer-2 sum works out: exactly one of the three R units is active at any time, so the sum comes to -2 plus that unit's output, and the threshold holds.

I'm settling the lead factor at β=0.8 with the paddle height Yp at 185, since I can't confirm the angle from the byte values alone. Then I'll check the sign convention for byte 105 before finalizing the gating.

I'm assuming byte 105 means rightward when it's 1 and leftward when it's 255 or 254, with the ball's x increasing for right, though if that's backwards the policy would be wrong and the next rollout should show it. I'll go with that for now and keep the mapping as is.



---

I'm verifying the down-left case: the arithmetic gives a target difference of -16, which matches the expected unit values, and the up case checks out too. Then I'm second-guessing whether computing first-layer preactivations counts as running part of the network, since the rule forbids that on inputs.

I'm wrapping up the sanity check on the arithmetic, and I'll stop there rather than running anything more on the inputs. Next is the final rollout, and then I'll decide.



---

I'm tracing the ball's movement against the variable values over steps 6 through 18: the ball moves left while variable 105 is 1, so 105 can't be the dx sign, or its sign is reversed. Then over steps 24 to 42, when 105 is 255, the ball's x coordinate increases, which points to 105=255 meaning rightward motion.

I'm now seeing that the sign is reversed: 105 at 1 means moving left and 254 or 255 means moving right. Working through the rates, the ball moves about 1 pixel per frame, and 254 gives a faster 1.5 pixels per frame. So my earlier prediction had the direction backward, and I need to swap the dx sign before deciding whether to test it.

I'm now weighing the wall problem: with the prediction off by up to about 100 pixels, the paddle could aim past the wall edge while the ball actually lands at a reflected spot. I'm considering a reflection term, folding the predicted landing point back across the wall, and building the left and right decisions from that folded value.

I'm building the wall-reflection part of the network: a reflected landing term, 398 minus the predicted x, gated to the down-right and down-left cases. The right move would then need both the direct and reflected checks to pass, and the left move either one, so I'm now sorting out how to set up the layer-2 units so the aggregation works out.

I'm working out the layer-2 weights so the AND and OR conditions combine correctly: giving the direct-right unit a weight of 2 so the up case and the down-right case sum to distinct totals, then checking whether requiring both reflected and direct checks to pass lands the aggregate at the right sign.

I'm checking the layer-2 threshold for the right-move unit: with the up case summing to -2 or -6 and the down-right case to -2 or -4 or -6, a threshold near -3 should separate the all-pass case from the rest. For the left-move OR, I'm placing the threshold at -5 so any single passing check lifts the sum above it.

I'm checking that the two wall-reflection conditions are mutually exclusive, so the right and left moves can't both fire, with no move left when both fail. Next I'm working out the wall bounds: the left edge seems to sit near x=55 and the right near 199, and I'm considering a reflected landing point across the left wall for cases where the predicted x falls below it.

I'm working out the reflected landing term for the down-left case, where the right move becomes the OR and the left move the AND. Since each case needs its own aggregation, I'm planning separate layer-2 units per case and combining them linearly in layer 3.

I'm wiring the right-move and left-move units so each case combines its direct and reflected checks correctly, with the up case and the two down cases summed into a single output for each direction. Then I'm setting the final FIRE bias so it only fires when neither direction is active.

I'm working out how to split the right and left cases by the byte value, since the speed ratio changes between 254 and 255. Keeping the ratio linear in the byte would make the combined term bilinear, so I'm considering gating separate units for the fast and slow variants.

I'm working out a gating scheme to separate the fast 254 case from the slow 255 case without the penalty bleeding into the left-moving bytes. Using two different multipliers on the byte offsets, with the first weighted more heavily, should keep the left-side values from flipping sign.

I'm trying to get a zero at byte 254 without the penalty spilling over to the left-moving values, since a single linear term can't be zero at 254 while large at both 255 and the small bytes. That means the fast and slow cases have to be separated some other way, maybe with extra gating units in the first layer and the selection done in the second.

I'm sizing up the speed-indicator units: the fast and slow cases would need two-sided indicators, and layer 2 could combine them with the threshold units. A simpler option is a single average ratio of about 1.25, which errs by roughly 25 pixels either way, and that error shrinks as the ball nears the paddle's row, so I'm checking whether the paddle's own speed makes that acceptable.

I'm checking the paddle's speed against the ball's: the paddle moves roughly 1.7 pixels per frame, a bit faster than the ball's 1 to 1.5, so chasing should keep up. The prediction error shrinks as the ball descends, so a single averaged ratio should be workable, and wall reflections can be handled separately.

I'm working out the bounce height from the trace: the ball turns around at some y between 158 and 172, so I'm solving for the turning point using the frame counts on each side of it.

I'm settling the bounce height at y≈178 and the lead ratio at β≈1.2, since the ball moves about one pixel per frame in both x and y on the slow path. I'm also keeping the offset c at 8 and checking whether the step-18 miss, where the ball sat about 11 pixels left of the paddle, fits that setting.

I'm settling the wall bounds at Lw=56 and Rw=200, since both edges fit the frame counts between the observed positions. Next I'm turning to how the reflection enters the target calculation.

I'm checking the mirrored case for the left wall: the predicted landing point is reflected across the left edge when it falls below it, so the left move fires if either the direct or the reflected point sits far enough to the left of the paddle.

I'm laying out the first layer's units now, with up and left checks against the paddle offset, and I'm planning how the down-right case gets its own lead term.

Writing the layer-1 units now, with separate right and left checks for each down case.

I'm laying out the layer-2 units for each direction and checking that the inactive sums land near -2, so the fire threshold of -2.5 separates them.

I'm settling on β≈1.2 for the lead ratio, since the slow path's 1:1 movement means a modest overprediction is acceptable. I'm also thinking through how the penalty terms for the right and left byte cases should be sized.

I'm settling on a lead ratio of 1.2, reasoning that a modest overprediction is acceptable and that I'll verify the weight construction by hand rather than by running the network.



---

I'm checking the algebra by hand: for unit 3 the expression reduces to -x + βy - px + (2Rw - βYp - c - m), and for unit 5 it comes out to the matching coefficients on x, y and px with the constant term, and both check out.

I'm finishing the hand-check of the remaining units, and they all come out consistent with the formulas. The gate penalties and scaling check out too, with the k factor applied to the constant. I've also worked through the edge case where the ball isn't in play: it reduces to a down-left serve that the paddle can trigger with RIGHTFIRE. That settles it, so I'm going with the build3 policy over the baseline.

