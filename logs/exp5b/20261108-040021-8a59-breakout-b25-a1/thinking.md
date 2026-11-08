I'm working out a Breakout policy that reads the game state straight from RAM, and I'm recalling the known addresses for the ball position, the paddle position, the block-hit count, and the score.

I'm sketching a policy that uses the RAM indices for ball x, paddle x, and lives. It fires when the ball isn't in play and moves the paddle toward the ball, with a deadband so it only shifts when the difference is large enough. I'll calibrate the offset and ranges with rollouts, and keep the second layer close to an identity-style pass-through.

I'm weighing the ball's predicted landing point rather than chasing its current x, but I'll start with a simple follow policy and zero offset, then check it against a rollout trace of those RAM addresses to calibrate.



---

I'm checking how paddle position lines up with the ball, and the paddle x values seem to be the left edge of a 16-pixel paddle. I still need to find the right offset, and I'm also wondering whether paddle speed is what's holding it back, since the Atari 2600 paddle is a joystick-driven control.

I'm also weighing whether ball velocity could be read from RAM, though traces only sample every 16 bytes so that's uncertain. First I want to pin down the paddle-to-ball offset, testing a few values like -8, 0, and +8. The ranges look inconsistent: the ball reaches x=42 while the paddle bottoms out at 55, so the coordinate systems probably don't line up directly.

I'm going to test offsets of -8 and +8 empirically, since the coordinate ranges suggest the paddle and ball aren't on the same scale. I'm also thinking the paddle may move about 12 pixels per step given frameskip, so I'll run a few configurations with 32 rollouts each to get enough statistical power.



---

The offset of 8 works much better, since the ball-to-paddle gap is about 8, which lines up with the paddle center. Episode lengths are still around 2000, though, and I'm wondering why they got longer with -8; maybe the ball isn't getting served. I'll try offsets of 12 and 16, plus a larger and smaller dead zone.



---

I'm seeing that action 0 (NOOP) is chosen about 74% of the time, so the agent is mostly idle in the deadband. The FIRE logit is exactly 0, the same as the others, so argmax ties to NOOP and the ball is rarely served, which would explain the long episodes. Setting the FIRE bias to a small positive value should break the tie.

Next I'm weighing whether the agent's ball-tracking can improve if it knows the ball's velocity, and whether a previous ball position stored in RAM could give that. I don't trust my memory of which RAM addresses hold the ball and paddle, and tracing bytes would cost too many rollouts, so I'm leaning toward tuning the tracker instead. My current score of 34 is low, so I'll fix the FIRE bias first and then adjust the offset and deadband.

I'm considering which RAM bytes might hold the ball's velocity, and whether tracing several candidates at once could reveal it despite the subsampling. With roughly thirty rows per trace, a velocity byte should show up as small steps, positive or wrapped negative, when correlated against the ball position.



---

I'm reading byte 103 as a vertical step, either 1 or 2 for down, or 254 or 255 for up. Byte 105 looks like the horizontal step, with values 1, 255, 254, or 0. At step 390 the paddle sits at 77 while the ball is at x 138, so the paddle seems to be falling behind. At step 375 the paddle was at 80 moving right with dy=2, so I'm wondering whether it simply can't keep up at that speed.

I'm working out a strategy: predicting where the ball will land by extrapolating its slope from the current position to the paddle's height, with wall bounces factored in. The paddle sits somewhere around y≈185–190, though I'm not certain of that. The main open question is how to encode dx in the network, since 254 and 255 stand in for negative values and I still need to handle the cases where dx is 0.

I'm sketching the next piece: when the ball is heading down, I want to predict its landing x from its slope and the paddle's height, and when it's heading up, just track its current x. The hard part is that gating a slope-times-distance product inside a network is awkward, so I'm considering discretizing the slope into a handful of categories and using a separate gated unit for each.

I'm working out how to gate the comparison so that only the active category's error term drives the decision, using a large negative offset to suppress the inactive units.

I'm working out how to make the two-layer gating work: the first layer would produce the gate signals and the error terms, and the second layer would combine them, with each error term saturating to its sign.

I'm working through the deadband question: the sign-only units give no dead zone, so the output would flip between left and right, which is acceptable since fire is handled separately. Adding a dead zone would mean a second error unit per category with an offset threshold.

Still writing the gated units...

Still working out how to gate the equality tests on the two bytes. A single unit can't isolate one (dx, dy) pair, so I'm considering combining threshold units across the two layers instead.

Still working out the gating: I'm combining the four threshold conditions into a single saturating unit so it only fires when all of them hold.

I'm checking which byte values the ball's direction actually takes, to see if the signed decoding works out. I'm considering a threshold unit for the sign bit, with dx recovered as b minus 256 when b is 128 or more.

I'm sorting out the ball's down-moving speed cases by its horizontal and vertical steps, and wondering whether the per-frame velocity is really that small given the y range I'm seeing.

Now I'm working out how to handle the walls: if the predicted x falls outside the play area, I'm folding it back as a reflection, and I'm considering building a separate piece for each slope category to do that.

Still working out the wall handling: for a rightward slope, the predicted landing point folds back at the right wall, so I need both the direct and the reflected distance to the paddle to come out positive, which means two sign units combined in a second layer.

I'm checking whether the AND of the two positive sign units can be expressed with a single saturating unit in the second layer, so the rightward logit comes out as a clean ±1 per category. The leftward case is the harder one, since it needs an OR over the negative outputs, which I can get by negating the AND of the positives.

I'm folding the LEFT logit into the same gated-sum pattern, with each unit contributing 0 or 1 depending on whether its gates are on and the two positive terms agree. The output is then a simple comparison: RIGHT wins when its sum beats the 0.5 FIRE offset, and LEFT wins in the mirrored case.

I'm settling on a general scheme where each action's rule is an AND of layer-one threshold literals, and the output logits are sums of those terms, so that a single rule fires per decision.

Still working out the wall folding: for leftward motion the landing point folds back at the left wall, so going right means either the direct or the reflected distance clears the paddle's offset. I'm encoding that OR as the gate minus an AND term in each category's logit, so each direction's rule stays a clean sum.

I'm planning to pin down the paddle's y position, the wall bounds, and an offset value, and I'm weighing whether dx and dy are per-frame pixel steps, since that decides the slope. I'll need a denser trace of the ball's position over several steps to check how those values actually relate.

I'm planning to test the predictor by rolling it out with the slope scaled by a tunable constant, and to pin down the paddle's y position and the wall bounds along the way. I'll check the ball's maximum y at paddle contact against that value, and I'll gather the set of dy and dx values from the best trace to see what range they actually take.

Next I'm going to pin down the paddle's y position, guessing around 184 for now and tuning it later, and then check the value ranges in the best trace.



---

I'm working out the possible dy values from the byte pattern, with 252 as -4, and guessing dx falls in ±1 or ±2. The velocity might be a fractional system where bytes 102 and 104 hold the fractional parts, so I'm estimating the slope rather than pinning it down exactly. I'm also checking the ball's x range against the paddle's, looking at where the left wall might sit.

I'm now counting the downward-motion categories: four dy values and dx in {-2, -1, 1, 2}, which comes to too many error units. I'm thinking of collapsing them by slope ratio instead, since only dx/dy matters, which cuts the count to about six per sign.

I'm now weighing whether to quantize the slope into a few groups, since the exact ratios cost too many units. I'm also questioning whether the paddle's position is sampled before or after the ball moves, which affects how much time it has to react.

I'm now checking the byte thresholds for the dx and dy values, figuring out which comparisons separate each case, and counting how many gate units that takes.

I'm now checking the decision for the predicted slope: if I drop the deadband, each slope category needs just two units, and the count still fits in the budget. The layer-two problem is that one slope category covers several dx/dy pairs, so a single AND unit can't express the OR across them.

I'm trying to fit the left-moving case into the gate budget: the right case takes one unit per slope category, while the left case needs two, one for each condition.

Checking whether the layer-two gates can be written as literal dy values, I'm looking at how each dy=k test is built from two byte thresholds, and whether the combined count still fits the budget.

I'm now covering the gate for dy=4 and the case where dy is 5 or more, making sure every downward value falls into some category, and checking how the dx=±2 ranges split. I'm also setting up the follow-mode units for when the ball is moving up or not in play.

I'm checking the follow-mode cases: when the ball is moving up, I'd need one unit for each direction, plus a separate case for when the ball isn't in play, which I might just drop. That brings the total to about 62 units, so the budget still holds. I'm also wondering whether the prediction should extend to the upward case later on.

I'm settling the output layer: each active gate contributes a unit, the FIRE bias is set so that any firing gate outweighs it, and the total stays within the budget.

I'm settling the precision of the first-layer threshold units, and considering how sharp the tanh needs to be for the error terms to hold up near zero. I'm also setting rough constants for the paddle's top edge and the wall positions, guessing them from the ball coordinates I've seen so far.

I'm settling on a shared off-by-some offset for the paddle alignment, testing 8 against 6 and 12, and I'm not sure where the ball's x actually meets the paddle center. The byte values might carry fractional speed, so the dx/dy ratio could be approximate. I'm going to write the builder in a generic way.



---

I'm checking that the slopes are all distinct and the unit count of 61 adds up, and it does. I'm also confirming the K scaling per raw unit, then starting to look at how the RIGHT logit sums feed into b3.

I'm checking the sign of the RIGHT-side sums against b3, since a large bias with tanh near -1 leaves only a small residual, and the partially active units near the thresholds look fine too. Then I'm confirming that the AND for a positive s picks left exactly when R minus p minus off plus d is negative, which it does. Running it now.



---

I'm noticing that the ball's x position moves opposite to what I'd expect: as y increases, x decreases when the direction byte is 1. That would mean byte 105 is controlling leftward motion, but when it's 255, x increases instead, so the sign of dx looks flipped.

I'm checking the sign on the direction byte: 105 seems to mean leftward motion, and 255 rightward. The slope is about 1 per frame for |dx|=1, and with byte 254 it works out to about 1.5 per frame.

I'm now doubting the magnitude mapping for the direction bytes: a byte of 2 might mean 1.5 per frame, but that's unverified, and the vertical speed for dy=2 is unknown. I'll correct the sign first, then measure the speeds from the sparse trace, since this short episode gives usable spacing and the later high-dy values aren't covered yet.

Next I'm running the trace again to check the corrected sign.



---

I'm treating the x-speed as 1 and then 1.5, and guessing the y-speed sequence while checking its range. I'm also considering that the byte values might be whole pixel moves per frame, with the game alternating between frames. One step shows x moving 36 and y moving 24, which could mean x sometimes moves 2 pixels and sometimes 1.

I'm generalizing the builder to take vx and vy maps and deriving the slopes from them, with a sign fix. For the y-speeds I'm trying a guessed map, and I'm questioning whether a dy byte of 2 could mean 1.5. I'll measure that in the next rollout, which should produce mixed episodes.



---

I'm wondering whether exec in this function scope will see different globals, so I'm leaning toward passing globals() explicitly to exec.



---

I'm wondering whether the replace of "def build2(" actually went through, since inspect.getsource could be failing silently. There was no error, so I'll check the start of the source to see.



---

Since the source lookup came back empty, I'm going to rewrite the function from scratch instead.



---

I'm counting unique slopes at vx 1.5 and getting 69 total, which is too many, so I'm looking for a way to merge near-equal slopes within a tolerance.

I'm trying to cut the count further by merging 0.375 into 1/3, which would bring it down to 61, and I'm considering a canonical slope function that uses a merge set.



---

I'm weighing how to merge values like 0.25, 0.333, and 0.375, and found a 0.15 threshold too loose since it snaps 0.333 to 0.25. I'm trying a tighter merge of 0.06 so 0.333 and 0.375 combine into 0.333.

Now I'm counting the values that fall in the merged groups, using the sorted order so the 0.25 comes first and the 0.333 and 0.375 merge into it. That gives six per sign, so thirteen in total. I'll change build3 to take a slope table directly.



---

Averaging 33 is close to the simple follow strategy at 35.8, so I need to dig into specific steps to see what's going wrong. Looking at step 216, the ball is moving right at 1.5 and should land around y=175 after bouncing, which puts the paddle target at 167, and the paddle is at 175, so it's roughly in position but still a bit off. Next I'm checking step 252, where the ball is at x=134 and y=105 with dx=1.

I'm checking the step-270 case, where the ball is moving left and my predicted paddle target of 111 doesn't match the paddle at 85, and a life was lost before that step. Since the traces are too sparse to pin down the cause, I'm considering whether the paddle's y-position is off and planning to test a couple of alternative values for it.

I'm weighing whether the paddle's inertia or sticky actions could explain why the predictions fall short of actual results. The paddle moves roughly 5 to 6 pixels per step, so I'm considering whether a different paddle y-offset would fix the calibration.

I'm testing the paddle-offset hypothesis by running rollouts with the paddle target shifted to Yp=175 and Yp=195, since each costs a rollout and I have 17 left.



---

I'm comparing the three Yp values: 185 gives the best score at 33, while 172 and 198 come in at 23.6 and 8.6. The sensitivity at 198 makes me think the true value may sit a bit below 185, around 180. Still, none of these beat the following result of 35.8, so something else is probably off, like the slope mapping or an offset.

I'm checking whether the byte 105 direction mapping is right: the earlier trace went from bx 104 to 80 with 105 set to 1, which is leftward movement, and the paddle's rightward direction matches since the offset of 8 already worked. I'm also weighing the case where the ball is descending near the paddle and its position is extrapolated backward, and whether tracking ball x while it rises and then rushing to the predicted spot is the better approach. The sampling is still too sparse to pin down the trend, though, since each episode gives only a point every 20 to 30 steps.

I'm recalling how Breakout's ball speed works: vertical speed shifts after certain hit counts, and horizontal angle depends on where the paddle is struck. My measurements so far suggest dx of 2 covers about 1.5 pixels per frame, while dy values are still unclear and my samples are too sparse to settle it. I'm considering trying a vertical speed mapping of 1, 1.5, 2, and 2.5 for the four dy values, and maybe testing a paddle target around 180.

I'm weighing whether the vertical speed mapping is the real problem, since a slope of 0.5 for dy=2 would put the predicted landing point about 12 pixels short. Checking the last trace where a life was lost, the paddle was at 82 moving right while the ball sat at x=94 with dx pointing right, and a unit slope instead would shift the target considerably.

I'm checking the step 342 loss: the ball at x=125 with dx moving right would land near 137 while the paddle sat at 126, which should still be within reach, so either the slope or Yp is off, or the hit window is narrower than I assumed. Another life loss around step 252 is worth comparing against this same pattern.

I'm now checking the life-loss around step 252: the ball was at x=134 moving left with the paddle at 80, which would leave only a 5-pixel offset at the paddle's edge. But by step 270 the ball had gone to 190, so I'm wondering whether the left wall reflection is actually lower than the 57 I assumed, since I've seen x values as low as 42.

I'm now testing whether the paddle's catch window sits between roughly 2 and 20 pixels from the ball, since the miss near the wall suggests the 8-pixel offset may not be the full story. I'll tune Yp, the offset, the vertical speed mapping, and the wall bounds together, rolling out each variant to see which combination holds up.

I'm going to drop the two-at-once idea and test Yp=179 first, so any change in score can be pinned to the paddle target alone.



---

Looking at the traces now to see what's going on around the life losses. I'm pulling up the worst rows near each loss so I can figure out what led to them.



---

I'm checking the step-108 numbers: the ball is at x=124 moving right, the paddle is at 130, and my prediction put the target at 126, so moving left looked correct. But I still lost a life, which means something in the prediction is off, possibly the bounce timing or how far the ball travels after y=169.

I'm now wondering whether I misread the direction of the right-moving balls, but tracing the earlier steps confirms the x and y values are increasing as expected. Looking at the paddle hit offset, the ball sits a few pixels ahead of the target, which should be inside the catch zone, yet the same kind of miss keeps happening with right-moving balls. That makes me suspect the catch window is narrower than I assumed or offset from where I'm measuring it.

I'm now weighing whether the paddle is simply too far right of the ball, since these right-moving misses all show the paddle a few pixels past the target. That suggests the offset should be larger, around 12 to 14, and my earlier worse result for 12 may have been skewed by the idle-action tie bug. I'll try an offset of 12 with Yp set to 179.



---

The 34.8 result is only slightly worse, which I'll treat as noise. At step 396 the ball was heading right and I missed the target, and I can't tell whether the paddle reached it in time. Paddle speed might be the real bottleneck: at step 418 the paddle is 16 px from the predicted landing point with about four steps left, so I'm checking whether it can get there.

I'm trying to work out how fast the paddle moves, since the traces so far don't settle it. The early steps show it going 66 px in 14 steps, but the later ones tell me nothing useful. I'm also considering whether sticky actions add latency, and whether the paddle accelerates from a standstill. If it's slow, I should start moving toward the predicted landing spot while the ball is still heading up, so I'm sketching how to estimate that spot from the bounces off the top and the bricks.

I'm working out a rough formula for the ball's landing spot during its upward phase, bouncing off a guessed ceiling near y=90. Covering every combination of direction and speed that way would use up the capacity I have, so I'm considering a simpler decomposition instead. Before that, though, I want to know whether the paddle can even reach the targets in time.

I'm recalling that Breakout in the Atari emulator uses a paddle controller, so the paddle probably moves at a fixed amount per frame rather than accelerating.

I'm estimating the paddle's speed from the traces: it covered about 115 px in 18 steps, roughly 1.6 px per frame, which is only a little faster than the ball's horizontal motion. That makes predicting the landing spot early look important, so I'm weighing how to fit an upward-phase prediction into my limited capacity.

I'm weighing whether to drop the deadband on the downward path and let the paddle jitter around the target, or find another way to get three states out of the layer. One option is to compute a continuous target position in layer two and let the output layer compare it with the paddle position, though a linear output can't threshold that difference directly.

I'm trying to make the deadband work without spending extra capacity: a layer-2 unit can carry a scaled (T - p) difference if I use two tanh units per target, one active only when its gate is on and the other pushed toward a constant when it's off, so their sum is zero when the gate is off.

I'm checking whether the reflected target can be handled with a separate unit and a gate that selects between the direct and reflected cases, which would need one more threshold unit per slope.

I'm considering a gated-difference trick so the output layer can apply the deadband itself, but the reflected-target case needs an extra gate per slope and the unit count gets too large. I'm now looking at whether layer one can gate the products directly.

I'm now asking whether it's worth separating dy values, since the ball speeds up over the game, and I'm considering a signed-velocity representation to cut down on the number of gated combinations.

I'm working out a simpler way to handle the upward phase: the ball's landing spot depends on how many times it reflects off the ceiling, and two reflections would be possible over that distance. Rather than handling every case, I'm considering just steering toward the center of the screen while the ball is still going up.

I'm now weighing a simpler way to handle the upward phase, such as steering toward the center of the screen while the ball is still going up, since following the ball's x position is only a rough guess too.

I'm weighing a cheaper encoding for the layer-two combinations: a single layer-one unit that mixes the dx and dy bytes into one code, so each slope combo can be picked out with a pair of threshold units instead of a full gated unit per combo.

I'm settling on a cruder plan for the upward phase: instead of predicting the bounce, I'll just lead the ball by a fixed offset in the direction it's moving horizontally, using a few categories for its horizontal speed.

I'm checking whether leading the ball horizontally would work during the upward phase, since the reflections make the landing spot hard to predict. The displacement depends on the slope and how far the ball has to travel before it comes back down, so the reflections make it ambiguous.

I'm weighing whether following the ball's x position in the upward phase is good enough, or whether leading it by a fixed offset or steering toward the center would do better. Given the paddle's speed is only a little faster than the ball's horizontal motion, the misses may come more from imprecise landing predictions than from the paddle lagging, so I'm thinking about testing a few upward-phase strategies to compare.

I'm checking the misses against the paddle's position: at step 108 it's 4 px off the target, and at step 216 it's 3 px off, yet both still missed. That suggests either the paddle moved away again or the hit window is narrower or offset from where I'm aiming. The trace doesn't settle it, so I'll need finer data before drawing a conclusion.

I'm thinking about whether dense traces are even obtainable, since short episodes aren't something I can control. Instead I could analyze hit geometry statistically from the existing traces, though consecutive rows where the ball flips from down to up are probably too sparse to help. I'm also checking the Breakout RAM layout, where the paddle's x position ranges across a fixed span relative to the ball's, to see whether the playfield bounds would give a clean offset check.

I'm checking whether the paddle's offset from the ball is the cause, since I'd estimated an 8 px center offset from the RAM layout and the tests favored it only slightly. The deadband and the paddle's overshoot could explain some misses, but that doesn't yet account for the ones where it was already within a few pixels.

I'm checking whether the misses really come from prediction error, since at step 108 the ball should still land within the paddle's window, and at step 18 the ball's x position doesn't line up with the wall bounce the way I'd expect.

I'm checking the ball's path near the left wall: it bounces off at about x=56 around y=186, and the paddle sits at 55, which should have caught it. Since the paddle was 172 for Yp and that scored worse, I'm wondering whether the paddle's catch window isn't simply [p, p+16].

I'm weighing dense traces around a hit: short episodes won't come on demand, and a policy that never fires just stalls. Storing the last hit position in RAM is an unknown, so I'm leaning toward parameter sweeps over rollouts instead, varying the paddle's Yp offset.

I'm weighing whether a paddle moving several pixels per step would overshoot a deadband of ±2, which could make it jitter around the target instead of settling on it. I'm going to check the early trace, where the paddle moves from 146 to 80 over about 14 steps, to see how fast it actually travels.

I'm checking the paddle's speed against the trace, and the jumps between samples are large: at least 11 px per step over some stretches. With a deadband of ±2, a paddle that fast could overshoot the target and jitter around it rather than settle.

I think the paddle's speed is the real problem: it moves about 11 px per step, so a deadband of ±2 makes it overshoot and jitter around the target. I'll widen the deadband to 6 so it can settle within the hit window, and test that with both the down and follow strategies.



---

The jump to a mean of 66.6 with a best of 372 points to the deadband as the main lever. I'm weighing whether to try a deadband of 8 or 10, and I may also adjust the tuning, but I'll start by testing a deadband of 9.



---

I'm going to push the deadband up to 12 next, since it sets the ±dead catching window, and with the paddle only about 16 px wide, too large a value would start causing misses at the edges. I'll also track that adjustment on its own.



---

Dead=9 scored better than 12, so 9 may be near the optimum, with 8 or 10 worth trying later. With a deadband around 9, the offset now matters, and I'm wondering whether it should be asymmetric. The paddle also seems to shrink after it breaks through to the ceiling, which would make the window smaller.

I'm now checking whether the offset should be asymmetric, with a few variants to try next: off=10 with dead=9, and separate tests of dfol and the up-mode alternatives.



---

Comparing the offset of 6 against 8 shows a big difference in results, though the spread is wide, with a standard error around 16. I'll try an offset of 10 next.



---

I'm testing offset 10, which gives 186, and next I'll try offset 12.



---

I'm at a best of 186 with offset 10 and six rollouts left, and I'm weighing whether to test dead-zone 8 against 10 or work on the upward-motion case. Many losses seem to happen when the ball is near the left wall with the paddle at its minimum of 55, so I'm checking that case with a concrete example.

I'm checking whether the paddle's limit at 55 is what makes the left-wall misses unavoidable, since the ball lands about two pixels out at best. The window estimate doesn't quite line up with the offset of 10 that works better, so I'm wondering whether my left-wall reflection estimate is off. Next I'm weighing whether the remaining budget is better spent on up-mode tuning or on the dead-zone comparison.

I'm weighing a Yp test at 182 with dead-zone 9 against dead-zone 10, since with this much noise the 10% gaps I'd be chasing can't be told apart anyway. I'm also wondering whether the dead-zone should shrink once the paddle does, and whether the FIRE action and the longer final run change anything.



---

I'm weighing two options: test dead=10 with Yp at 181, or separate the dfol change. The scores are close enough that I can't tell them apart yet. I'm also wondering whether the final positioning matters more during the descent.

I'm comparing dead=10 at Yp 181 against separating out the dfol change, since the dead-zone results look close. A window of at least one paddle step, roughly dead ≥ 6, is needed to stop reliably, and dead=6 did worse than 9 earlier. Based on that, I think 9 to 10 is the likely sweet spot, so I'll try dead=10 with Yp=181.



---

I'm weighing whether 155 is worth keeping, but it's likely no better than 9 or just noise, so I'll stay with dead=9. With three rollouts left, I'm considering a dead=8 test, an up-mode tweak, or changes to the L/Rw walls. Episodes average about 1600 steps and never hit the 5000 cap, so they all end by losing lives. I'll try dead=8 with Yp=181 and off=10.



---

Similar results around 186 across the configurations, with the dead and off settings at 8-9 and 10. I'm running out of rollouts, so I'm considering a change in up mode, maybe pairing it with the upoff lead. Next I want to check the wall estimates, L=57 and Rw=199, and confirm the paddle center ball-coordinates fall in range.

I'm also weighing a different velocity mapping, where the dy byte values might map to 1, 1.5, 2, and 2.5 pixels per frame instead. I'd test that with one rollout and keep it only if it clearly beats the 220 mark; otherwise I'd go back to Yp=181, off=10, dead=9.



---

I'm checking the mapping: the dx2/dy2 ratio works out to 1. For dx1/dy2, .667 could snap to either .6 or .75, and .6 is the closer one, so I'll go with that as an approximation before running it.



---

The last attempt scored worse at 127, so I'm reverting to the original t with vy=k. I have one rollout left, so I'll use it to check the saved final weights, with Yp=181, off=10, dead=9, and dfol=9, then save and verify.



---

The value came out to 175, which fits the expected range of about 185, so this checks out and I'm done.

