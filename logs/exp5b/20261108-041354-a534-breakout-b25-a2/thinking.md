I'm designing a Breakout policy that reads its state from RAM, and I'm recalling the known addresses for the ball, the paddle, the block count, and the score.

I'm working out the paddle policy: fire on every step so serving works, and move toward the ball by comparing the ball and paddle x positions, with a small dead zone where I hold fire in place. I'm sketching a network with a steep tanh hidden unit, since a one-pixel difference is only about 0.004 after scaling by 255, and I plan to run rollouts with trace_obs to calibrate the offset.

I'm writing the network weights for the paddle policy, with a steep tanh layer on the ball-minus-paddle difference and a bias that keeps FIRE on in a dead zone. I'll run a first rollout with guessed offsets and trace the candidate RAM bytes to calibrate.

```

```



---

I'm checking where the paddle and ball sit: the paddle's left edge is around 72 and it's 16 pixels wide, so its center is about 80. I still need to work out the ball's center from its left edge and how that lines up with the paddle near y 185–190.

I'm working out the offset so the paddle's center lines up with the ball, and settling on an offset of +7 so the paddle's left edge sits about 7 pixels left of the ball's x. That puts the paddle's center roughly at the ball's position. I'm also noting that the paddle lags behind a fast ball, so predicting where the ball will land may matter more than just tracking its current x.

I'm settling the sign of the paddle offset and planning to test an offset of 7 with 32 rollouts, since a small dead zone near zero should keep the paddle steady.



---

I'm checking the traces to figure out where the misses are coming from. Since they're only sampled every k steps, I can't pin down the misses exactly, so I'll look at the worst trace anyway.



---

I'm reading the byte at address 103 as a horizontal velocity, with values 1, 2, 254 and 255 showing up, and 105 as a vertical one, likely with 1 meaning downward. The ball's y range goes up to about 199, and I'm guessing it gets lost somewhere past 190. The traces are subsampled, so the detail is thin, but my working idea is that the paddle is speed-limited and the misses come from the ball moving fast sideways, so I'm considering a target of ball_x plus a lead term scaled by the signed horizontal velocity.

I'm working out the landing point: the x position at paddle height is the ball's x plus its horizontal step times the remaining vertical distance divided by its vertical step. Since that product is awkward for a tanh network, I'm considering a lookup over the few discrete horizontal and vertical velocity cases, which makes the target linear in position within each case. The open problem is gating those cases so an inactive unit stays near zero rather than saturating at -1.

I'm checking the gating trick: when a case is off, both tanh units sit at -1 and cancel, and when it's on, the difference is roughly 2·tanh(a·d). What's still unresolved is how to build a gate that is 1 for one dx/dy case and at or below 0 for the others, using the byte values 1, 2, 254 and 255.

I'm using layer 1 for the linear features and layer 2 for the gating, since tanh keeps the sign of the scaled difference, which is all the case selection needs. Each layer-2 unit would combine a signed copy of the distance with an indicator that is near +1 when its case is active and near -1 otherwise, so the paired units cancel when the case is off.

I'm settling the scaling so that the sign-preserving tanh layer still resolves single-pixel differences: with a 150 px range, a slope of about 2.55 makes a 1 px step come out near 0.01, which a gain of 100 turns into a usable signal. The gating constant B around 50 should be enough to push an inactive case firmly to -1, and I'm still weighing what to do with the ball moving upward.

I'm picturing the ball's landing point with wall bounces, where a predicted position past the right edge would fold back as 2R minus the prediction. The paddle span and wall limits are only rough guesses, and the trace is too sparse to pin down the vertical step, so I'm wondering whether I'll need a denser sample before settling the reflection terms.

I'm checking whether the traces, with a sample every 17 steps and frameskip 4, let me infer how far the ball moves per frame, and I'm wondering whether the sparse data can settle the dx/dy ratio that the landing-point estimate depends on.

I'm working out which sign of byte 105 means downward, since the ball's y reaches about 199 and I'm not sure where the paddle sits or when the ball gets lost. The sample at y=199 with 105 at 255 doesn't fit a simple pattern, so I'm wondering whether 105 is really the vertical velocity at all.

I'm wondering whether byte 105 really tracks ball_y, since the sign convention and the value at y=199 don't fit a simple pattern. I'm considering whether a denser trace from a shorter episode would settle it, though I'm not sure how to get one. I'll also lean on what I recall about the Breakout RAM layout, though I'm not confident about the paddle height it implies.

I'm testing whether byte 105 is really a downward velocity, but its sign doesn't line up across the sampled rows, so it may not be the vertical speed at all. Byte 107 seems to flip between 65 and 129, which could mean a direction bit, so I'll check that pattern next.

I'm weighing whether to infer the ball's direction from consecutive snapshots, and I'm considering a probe policy that steers the paddle by a candidate bit to see if the score responds. A shorter episode, with a single FIRE and then no tracking, would give denser rows, maybe every 5 steps, which should be enough to tell the direction.

I'm picking the thread back up on the byte-level dynamics: the sparse traces only cover the best and worst episodes, so direction is hard to pin down from them. A policy that dies quickly could give denser rows, though each life's length makes the spacing uncertain. I'm also trying to recall the Breakout RAM layout, particularly whether a previous ball position is stored somewhere that would let me infer velocity directly.

I'm going to run a stationary-paddle probe, FIRE only, to get a denser trace and check which candidate bytes, including 72, 57 and the flag-like bytes 100 through 106, track the ball's movement.



---

Deterministic regardless of sticky, since only FIRE matters here. The ball's y goes 132, 152, 172 over five steps, so it moves 1 px per frame downward, while x drops 4 px per step. The 103 and 105 values don't match that direction, so I'm checking the flip at step 20 around x=64.

I'm working out the direction of the ball: the 105 value seems to track horizontal motion, flipping sign when the ball bounces off a wall, while 103 may be vertical. I'm checking that against the x positions at each step.

I'm working out the speeds from the per-frame motion: 103 looks like the vertical component, signed so positive means downward, and 105 seems to be the negative of the horizontal direction. Next I'm estimating where the ball will land, using its current position and the vertical speed, and then adjusting the x position by the horizontal speed.

I'm estimating the wall positions from the x values near the bounces: the left wall appears to sit around x=55, and the right wall around x=200, so the paddle range is roughly 55 to 191 and I'm checking whether the paddle's width puts its edge past the right wall.

Still working out where the paddle sits relative to the ball's landing point, and I'm guessing its top edge is near y=189, so I'll refine that estimate once I know the paddle's speed.

I'm now figuring out how to handle the wall bounces: if the predicted landing point falls outside the walls, it gets folded back inside. Reflecting it that way needs a piecewise rule for each direction, and I'm weighing whether to build that into the first layer or treat it as a separate step.

I'm working out the fold: for a predicted landing point past a wall, the reflected position is the mirror image, so the paddle decision reduces to checking whether the ball's landing falls between the paddle's position and its mirrored counterpart across the right wall.

I'm reducing the wall-folding rule to linear conditions on p and q: the paddle should move right when the landing point is between q and the mirrored point 2R−q, or when it falls past the left wall's mirror at 2L−q. I'm turning each of those inequalities into a separate unit, so the combined decision can be built from them.

I'm checking the combined rule for the paddle: move right when the landing point lies between the paddle position and its mirror across the right wall, or falls past the mirror across the left wall. The sign test works out for the cases where the landing point is inside the walls and where it reflects off one of them, so the bounce is handled for single reflections. I'm leaving out repeated bounces for now, since they only matter when the landing point is far from the paddle.

I'm settling on how the gating works: each case gets two units that are pushed toward a fixed sign, and the gate term pins them to cancel when that case is inactive. The byte values I'm treating as direction are 103 for downward (1 or 2) and upward (254 or 255), and 105 for leftward at 1 or 2.

I'm setting up threshold units for each byte value, so that checking equality with a given value takes two of them, and I'm planning to gate the layer-two outputs with those indicators.

I'm tallying the layer-one units: 8 cases with three each for the A, B, and C conditions gives 24, plus threshold units for the byte values of 103 and 105.

Now I'm working out how the paddle's position is compared against the folded landing point, using a fixed offset of 7 from its edge.

I'm checking where the ball's bounce point falls against the paddle's edge, since the wall position and the ball's vertical extent set how the landing point gets folded back inside. I'm also estimating the paddle's top at around y=185 to 190, and the exact value affects the landing x only slightly.

I'm choosing the paddle's vertical position at about y=185, since the ball's serve point and the brick rows don't pin it down exactly. I'm also noting that the paddle can't go past about y=191, and that paddle decisions are only made every four frames, which keeps the dead zone from making it oscillate. The rest of the plan is to use FIRE throughout and to switch behavior when the ball is heading back up just after a hit.

I'm setting the paddle's scale so a one-pixel shift gives about a two-unit signal, then layering a gating term so the output for the right-fire logit is zero when this case doesn't apply. The FIRE output will just be the dead-zone term, and the serve case still needs to be handled when the ball is heading back up.

I'm now setting the layer-one thresholds for the ball's vertical byte at 0.5, 1.5, 2.5, 3.5, and 127.5, and for the horizontal byte at 0.5, 1.5, 2.5, 253.5, and 254.5, which lets me pick out the individual direction values.

I'm mapping the remaining byte values into layer-one threshold units, with the dy cases covering 1 through 3 and the dx cases covering 1, 2, 254, and 255. That comes to 47 units, which fits within the 64 budget. I'm also starting the gate penalty for layer two, built from the lower and upper threshold pairs.

I'm defining the gate for each case as the difference between two threshold units, so it reads 1 when the byte matches and 0 otherwise, and I'm making the penalty term subtract from the case output when either gate is off. Where the upper threshold is missing for the value 255, the gate is just the lower threshold shifted to the same scale.



---

The result came back worse, so I'm suspecting something is off: maybe the dx sign assumption, the Ypad value, or my earlier reading of the problem. I'll go through the trace to find where it diverges.



---

I'm checking the ball's position at step 40 against the paddle, predicting where it will land and whether the paddle can reach it in time. At step 50 the ball seems to sit just past the paddle's right edge, so I'm wondering whether it was missed.

I'm rechecking the paddle's position against the ball's predicted landing point, and it seems the paddle drifted left between steps 40 and 50, which doesn't match what the policy chose. I'm going to verify the ball's per-frame motion, and then look at the trajectory around step 60.

I'm working out the ball's vertical path from its bounce near y≈178 and whether the x-direction reverses there, but the numbers don't line up with a simple one-pixel-per-frame motion. The step 100 to 110 values suggest the ball kept moving right even when the sign of its horizontal velocity should have flipped, so I'm checking whether the paddle bounce happened almost immediately.

I'm trying to reconcile the ball's horizontal and vertical motion over those 40 frames, since a simple one-pixel-per-frame model doesn't fit the numbers, and I'm wondering whether the velocity sign is stored somewhere else.

I'm rechecking the bounce height from the step 100 to 110 values, which works out to around y≈184 with the horizontal speed varying after the paddle hit. But then the step 60 to 70 timing doesn't fit, since that path would take longer than the 40 frames allow.

I'm now questioning whether the sign of the horizontal velocity is stored elsewhere, and whether the ball might move at fractional speeds rather than exactly one pixel per frame. Checking the step 200 to 210 segment, the numbers do fit a steady rightward path, so I'm looking at the step 240 values next to see whether a leftward reading there holds up.

I'm checking whether the ball's horizontal speed is really one pixel per frame, since moving x from 93 to 78 over 40 frames doesn't fit a simple left-then-right bounce, and the 1.5-per-frame figure from the next segment makes me think the speed is fractional and stored separately from the direction bit.

I'm weighing whether the x-velocity is better estimated from the data itself, perhaps with a lead term tied to the ball's direction, since the simple one-pixel-per-frame model keeps failing. I'll probably need a denser trace of the candidate bytes around 90 to 127 before I can pin down the magnitude.

I'm going to trace a few more RAM bytes (99, 101, 103, 105, and a range from 86 to 96) with the tracking policy, since two traces fit in each rollout and I have about 21 left.



---

The policy's returns look too noisy to compare: the same policy scored 23.9 on one run and 29.7 on another, so the standard error is around 2-3. I'm also noticing that the 90 and 91 values look like a frame counter, while bit 95 seems to toggle between 128 and 64, possibly tracking direction, with 128 showing up when byte 103 is 254 or 255.

I'm going to check the ball's horizontal motion against the HMOVE behavior I'd expect: positive values would push it left and negative values right, so byte 105 might be encoding speed and direction. That fits the serve probe, though the earlier x changes over 40 frames don't line up with a steady 1-pixel-per-frame movement, so the ball may not update every frame.

I'm checking the network's gating logic by hand, since I can't run it on these inputs. The threshold units for byte 105 and byte 103 look right, the weight scaling against the 255 range checks out, and the dead zone should give strong outputs once the pre-activation is near ±1. The fallback path with the gate penalty also looks correct.

I'm weighing whether the network is wrong or whether my picture of the ball's velocity is. The byte 105 magnitudes don't obviously match one pixel per frame, so I'm considering that the velocity might only show up cleanly after paddle hits, and I'd need a different episode to see it.

I'm looking at the 105 values again: 254 seems to correspond to about +1.5 px per frame in x, which doesn't fit neatly with the 1 px-per-frame pattern I expected from the serve. The step 240 to 250 stretch, with 105 at 1 and x going from 93 to 78, points to a different rate, so the ball may not be moving on a clean per-frame schedule.

I'm weighing whether the 105 byte really encodes a plain speed: the wall bounce at step 250 suggests value 1 corresponded to leftward motion and 254 to rightward, so the mapping might be direction-first, not magnitude.

I'm wondering whether byte 105 is really a velocity at all, and it might instead be some other counter or a fractional x position. I also want to look at byte 95 as a bitfield again, but the probe shows it sitting at 64 for stretches with no clear pattern. Getting a cleaner read will likely need denser per-step data, which a short episode can't give me, so I'm weighing whether velocity matters for this at all.

I'm going to try a simpler heuristic: when the ball is moving down, steer toward its predicted x using the sign from byte 105, with the magnitude tuned through rollouts. Before committing to that, I want to check the direction pattern across consecutive frames where nothing bounces, to see how the sign and size of x movement actually line up.

I'm comparing pairs of consecutive frames where the ball's direction flags stay the same, to estimate its horizontal and vertical speed from the position changes. The first pair suggests roughly +1 px in x and +1.5 px in y per frame, so the ratio between x and y speed is not simply 1:1.

I'm tallying the remaining consecutive-frame pairs: most show a horizontal change of about 1 px per frame, while the vertical change is around 1 to 1.5 px, and the signs of the horizontal and vertical steps vary from pair to pair.

I'm reading the ball's speed off the byte pairs: 105 at 1 or 255 seems to mean about 1 px in x per frame, and the 2 or 254 values look like roughly 1.5 px. The ratio of x to y movement doesn't stay fixed, though, since the vertical speed seems to rise after paddle hits while the horizontal one holds steady.

I'm settling on a per-class ratio for the horizontal-to-vertical speed: one value for the 1/255 bytes and another for the 2/254 bytes, with the vertical byte ignored and only the downward direction used. Then I'll tune those ratios and the pad position through rollouts.

I'm wondering why the model does worse than plain tracking, around 19.9 against roughly 27. A predicted ratio of 1 against a real 0.67 would overshoot, but that should still beat tracking, so something else is probably off, maybe the sign for some byte classes or the pad position.

I'm working out the paddle offset: aiming at the landing point and whether an offset of 7 helps with tracking lag. I also suspect the hit position on the paddle sets the angle, so edge hits might be steeper and easier to handle. Next I want to pin down the Ypad value from a sequence where the ball goes down and then comes back up.

I'm working through the numbers with the 105 byte at 147 minus f, trying to solve for the frame where the ball hits the paddle; that gives a contact height near 189, and the ball seems to speed up after the hit. I'll take the paddle contact to be roughly 186 to 189, so setting Ypad to 185 looks reasonable enough, though the poor results may come from how the prediction aims.

I'm settling on two ratios for the horizontal-to-vertical step, one for the 1/255 bytes and one for the 2/254 bytes, and I'll test r1=0.67 and r2=1.0 first. I'm also loosening the downward gate so it accepts the full range of the vertical byte, then I'll tune the ratios through rollouts.



---

The 28.8 score is only a little better than the tracking baseline of about 27, so the gain is small. The misses likely come from something else, maybe paddle speed, inertia, or sticky actions, or from hits at positions that send the ball off at steep angles. I'll look at the worst trace near the life losses.



---

I'm checking the samples where the paddle sits well away from a descending ball, starting with step 266, where the predicted landing point is about 117 and the paddle is at 130, so it should move left. At step 280 the ball is at x 127 near y 200 while the paddle sits at 110, and I'm working out whether the paddle's span still covers that spot.

I'm checking whether the paddle's position really covers where the ball lands: at step 280 the ball is at x 127, but the paddle sits at 110 and the predicted landing point doesn't line up. The ball at y 185 should be near x 117, which the paddle ought to catch, yet it missed, so the paddle's actual hit region may be offset from its stored x value. I'm weighing a larger offset of roughly 12 to 15 pixels, since the offset of 0 and 7 gave very different results.

I'm testing whether a small offset in the paddle's hit region explains the misses, trying values around 4 and 10 with the prediction model, though the noise is large enough that the comparison may not be conclusive. I'm also considering whether the paddle's width and the wall bounds might tell me how the two coordinate systems line up.

I'm going to sweep a few offset values for the paddle's hit region, trying 4, 10, and 13, since the data where the paddle actually caught the ball is too sparse to pin the offset down directly.



---

I'm reading the insensitivity result as meaning the misses aren't caused by the paddle offset. The episodes run about 850 steps across roughly five lives, so each life lasts around 170 steps, which implies only three or four paddle hits per life. That seems low for a decent tracker at early speeds, so I'm wondering whether the paddle's movement dynamics are the limiting factor.

I'm checking the paddle's speed from the probe trace, which works out to roughly 12 pixels per step, so movement itself doesn't seem to be the bottleneck. Sticky actions look minor too, which leaves me unsure why the misses happen. I'm also double-checking my assumption about the sign mapping for the down and up values.

I'm checking whether the ratio depends on the vertical speed: at dy=1 the ball moves diagonally at 1:1, while after the speed-up at dy=1.5 the ratio drops to about 0.67. If some byte encodes the speed, that would explain the mismatch, so I'm looking for a candidate in the trace.

I'm wondering whether the misses come from the paddle not reaching the ball in time, since that's where the offset ruled out. I'm also thinking that a paddle that flips between left and right every step would be slow to track, and the 105=0 rows might leave it idle when the ball is moving down.

I'm adding a case for 105=0 with zero rate, and also cases for 3 and 253 with about ±1.5 rates, so the paddle has a fallback when the down branch doesn't match. I'm also planning to trace bytes 107, 95, 100, 102, 104, and 106 in consecutive rows to see whether any of them encodes the vertical speed.

I'm estimating how far the paddle's error grows with distance: the ratio mismatch gives roughly 0.33 times the remaining vertical distance, which could reach around 28 pixels from far up the screen. As the ball closes in, that error shrinks, and since the paddle covers about 12 pixels per step while the ball moves about 6 in y, the paddle should be able to catch up.

I'm checking whether the paddle hit zone is shifted from where I assumed. Offset 0 did much worse than offsets 7 through 15, which look about the same, so the zone might sit somewhere I haven't pinned down yet.

I'm working out how fast the paddle actually moves, since a slow paddle would explain the misses better than prediction error. I'm going to run a probe that holds RIGHT every step to measure its speed directly.

I'm going to run a probe that holds LEFT every step, which should show the paddle's speed more clearly than the earlier RIGHT trace. I'll also log the 103, 105, 99, 101 and candidate 107 and 95 bytes alongside the paddle position.



---

Checking the paddle and ball positions at each step: the paddle got to x=55 by step 8, which works out to roughly 3 px per frame. The ball's bounce near x≈56 lines up with where the paddle was, so that contact looks right. Next I'm looking at how the ball's x-direction changes between steps 24 and 32.

I'm checking whether the paddle's acceleration could explain the misses. Maybe the emulated paddle has a per-frame delta and a smoothing lag, so the position the game reads trails the real input. That lag would throw off my predictions in the few hits where I missed.

Still on the miss analysis: I'm thinking the game's paddle reading may lag the real input, which would throw off my predictions on the few hits I missed. The next step is to pull finer traces around those misses rather than guess at the cause.

I'm now asking whether the misses come from something systematic rather than the prediction itself, since the paddle hits look right in the samples I've checked. The paddle's hit zone might be wider than I've assumed, or the ball could be getting caught on the brick edges and drifting after the paddle contact. Pulling finer traces around the missed frames should show which it is.

I'm pulling the ball and paddle positions from the worst rollout to compare offsets near the bottom of the screen, where the misses happened.

Checking whether the misses come from the hit zone: a ball about 17 px right of the paddle should be a near miss, but ten frames earlier the ball sat only about 10 px from the paddle, which should have counted as a hit.

Checking whether the paddle at x≈110 with the ball at x≈190 on row 193 is a real miss or a lag artifact. The mismatch at step 260 looks like the paddle falling behind, so I'm suspecting my class-2 ratio in the earlier build, not the zone width.

I'm checking whether the paddle's hit zone explains this miss: the prediction put the ball at about x=117 when the paddle was at 110, which should have counted as a hit. If the contact happens lower on the screen, around y≈195, the ball would be near x=124, so the offset would be closer to 14 and the zone assumption might be the issue.

I'm checking the ball's contact with the left wall: from step 16 it moves left one pixel per frame, so it would reach the wall near x=55 after about nine frames. From there I'd need to see whether the paddle contact happens before or after the bounce to decide which surface it hit.

I'm trying to pin down where the paddle's position comes from: solving for the bounce point gives an implausible result, and the ball's drift over the next 32 frames doesn't match up. Rather than keep reasoning through the algebra, I'll test the paddle's hit height with a few values and compare outcomes, since each rollout is noisy but larger differences should still show.

I'm considering whether the paddle's emulated resistance might clamp wider than the visible game range, so holding left at the wall winds up and the paddle lags when reversing. If that's true, my wall-pinning off=7 tracking would be causing delayed responses.

I'm proposing a fix for the wind-up: gate the paddle so it stops pushing left once it reaches about x=56 and stops pushing right near x=190, then route the fire decision through a unit that checks whether the paddle is at the right wall.

I'm checking whether the paddle's emulated resistance can wind up past the visible walls, so that holding a direction at the edge delays the response when reversing. If that's the case, I'd gate the left and right presses near the walls, with a wall-proximity unit feeding the fire-and-move outputs to suppress them.

Now I'm wiring up the wall gating: stop pushing left once the paddle reaches about x=56, stop pushing right near x=190, and suppress the fire-and-move outputs when it's pinned there.



---

Still weighing the A/B result: the 24.6 isn't better, and the noise is too large to learn much from comparing runs. I'm looking for a way to get denser traces, since they only come from the best and worst episodes, and episode length sets how many rows I get. Deliberately ending episodes early doesn't seem workable. Subsampling uniformly over the moments might help, though: 28 rows per trace, two traces per rollout, which gives about 56 data points.

I'm thinking about how the paddle position might relate to RAM address 70, which looked like a raw paddle input that lags behind the paddle itself. The values I've already seen suggest a rough inverse relationship, with a wind-up near the right wall, so I'm checking whether address 72 can be expressed as a clamped function of address 70.

I'm tallying the (RAM 70, RAM 72) pairs, and the running sums cluster into two groups, around 224 and around 245. That split might reflect the direction the paddle is moving, with 70 lagging by roughly 20 units.

The paddle position seems to trail the raw input at address 70 by roughly 10 pixels, following a target of about 235 minus that value. At the walls the input can overshoot: it reaches 0 on the right, past what the paddle can physically do, and seems clamped on the left. So using address 70 directly as the control signal might work better than the paddle position at 72.

I'm weighing whether using the raw input at address 70 as the control signal, in place of the paddle position at 72, would respond more quickly and reduce oscillation. The paired values suggest the paddle converges to roughly 235 minus address 70, but with a lag of about 11 pixels depending on the direction of motion, which makes its steady-state position ambiguous.

I'm reasoning about why the paddle lags: the raw input at address 70 moves about 11 pixels before the paddle responds, so every direction reversal wastes travel and causes jitter. The right wall is the bigger problem, since the input can wind up to around 0 while the paddle is stuck at its clamp, so I'm considering using address 70 rather than 72 to gate the RIGHT move near the wall.

I'm testing whether the raw input at address 70 might be a better control signal than the paddle position at 72. The paired sums split into two clusters, around 224 and 246, which could be either a hysteresis effect or just the paddle trailing the input by about a step. Either way, the steady-state position is ambiguous, so the wall gating may need to rely on address 70 instead.

I'm going to try driving the control off address 70 instead of 72, since it leads the paddle, with the offset around 235 minus that value.



---

The score jumped to 45.7, a big gain, though the addwalls change isn't in this build. Next I'll tune the offset C: the combined effect is q = C - x70 + off, so the effective parameter C+off is 244, and I want to try 236 and 252 as well as check the trace.



---

I'm noticing that the paddle values come in even numbers at certain steps, which suggests the input is quantized. At step 132 the value 105 reads as 0, and I'm checking whether that means vertical movement, since the ball later shows up at x=53 after a life is lost.

I'm treating the 105=0 case as its own situation, since no existing class covers it when the paddle is moving down. My plan is a fallback that tracks the ball with a zero offset, and I'm considering small guessed classes for 3 and 253 as well.



---

Both offsets came out worse than off=9's 45.7, so off around 9 (C+off=244) looks close to optimal and the result is quite sensitive to it. I'll rerun off=9 with the new rmap to confirm, and also test off=6 and off=12.



---

Off=9 with the rm setting came out at 62.25, while off=13 gave 53.4. I'm wondering whether to also try off=7 and 10/11, though the noise is around 3, so I'll leave 6 for later. For now I want to look at the worst trace from off=9 to see what's going wrong.



---

I'm noticing that 103=2 and 3 come up often, with y moving faster, and the 105=2 class shows up too. I had been combining the dy classes, so I'm going to make the ratio depend on both 103 and 105 instead, as vx(105) over vy(103), which means I need a speed mapping for each of those values.

I'm still weighing whether the "1" speed class is really ambiguous, and I'm sketching a guess for the speed classes of 103 and 105, though the 105=0 case doesn't quite fit yet. I'm also checking the paddle position against byte 70 across the traces, which holds to within about ±11.

I'm testing the speed classes for 103=2 and 103=3, with vx and vy values on a grid, which gives ratios of about .5 and .75 for the 103=2 case. The 105=0 sign is still unclear, so I'm checking the traces where it appears to see which direction the ball moves.

I'm working out whether the x velocity encoding is symmetric, so I'm trying 0 as a slow leftward value of -0.5 and checking the traces near the walls. The 255 and 254 values look like rightward speeds, which makes the pattern feel asymmetric, so I'm considering whether 105 maps to a signed offset instead.

I'm working out how many cases the velocity table needs: 3 dy classes by 5 dx classes gives 15 cases, and at 3 units each that's 45, plus about 10 thresholds and a fallback, so the total fits under the 64 limit.

Tallying the case count again: 56 total with the 3×5 grid and the 10 thresholds plus fallback, so the layers still fit. Next I need to generalize the builder so it takes the (dy, dx) class map.



---

I'm looking at the last result, an 85.5 average with a std of 55, which is a solid gain. The up-tracking still uses the same offset as before, so that part is unchanged. Next I'm weighing whether to predict the ball's path when it's moving up instead of just tracking its x, and separately tuning vy for classes 2 and 3 and vx for class 0.

I'm weighing a few more ideas: the paddle may shrink to half width later in the game, which would shift its center and change how the offset behaves, and I'd need some RAM byte to detect that. I'm also going to check the traces from this run for failure modes.



---

I'm noticing a value of 252 in the byte at offset 103, which would be a downward step of -4 if the byte is a velocity. I'm wondering whether that byte encodes speed with small positives for one direction and 252 to 255 for the other, and whether the ball might sometimes move an extra pixel on certain frames.

The paddle sits far right at step 544 while the ball is descending at x=65 and moving right, which doesn't fit with the paddle needing to move left. Step 512 is also puzzling: the ball was near the paddle but a life was still lost, so the byte at 105=0 might mean something other than slow leftward motion.

I'm weighing a few tuning options against the rollouts I have left, but the noise is around 10 at this sample size, so small parameter changes won't show up clearly. I think structural changes would matter more, especially a separate class for the slower downward speed value of 4.

I'm weighing whether to refine the predicted contact height for the paddle, and I'm checking the early serve trajectory to estimate where the ball actually is when it meets the paddle. I'm also setting aside fine parameter tuning for now, since the noise is too large to resolve small changes, and focusing on structural issues like fast steep shots, wall double bounces, and the paddle shrinking after a breakthrough.

I'm working out where the paddle surface sits relative to the ball's y value, comparing the Breakout screen layout against the RAM values I've been seeing for the bricks and ceiling.

I'm checking whether the paddle's contact height of 185 holds up given that the ball's top edge sits near the screen y, and I'm weighing the offset against the paddle's position with its roughly 11-pixel lag when it's moving toward a target.

I'm comparing the paddle's position against where the ball would meet it: the offset works out to about +9 if the paddle lags its input, but the true paddle position is the better reference near the target. The paddle's hit zone is only about 8 pixels from center, so an 11-pixel lag would matter a lot. If the discrepancy is just time lag rather than a game feature, it should settle once the input stops moving.

I'm settling on the paddle lag as a plain time delay that resolves once the input stops, and off=9 is close enough to the hit zone. I'm considering tweaks to how the paddle positions itself while the ball is rising, but I don't have a reliable model of the brick heights to predict the landing point. I'll set that aside and test vx[0] next.

I'm now weighing a few small tuning options, like the vx[0] sign and the Ypad value, but the noise is about 10 at this sample size, so I doubt I can separate effects that small. Structural changes to the velocity classes or the paddle's contact model look like the better use of the remaining rollouts.

I'm thinking about the known Breakout mechanics: the ball's vertical speed rises after a set number of hits, and horizontal speed depends on which zone of the paddle the ball hits. I'm wondering whether byte 103 is literally vertical velocity per frame, which would fit the measured 1.5 px per frame, and I'm considering whether the ratio of horizontal to vertical speed is what really matters.

I'm going to test two more settings in a single rollout, Ypad=189 with off=9, and then Ypad=181, to see whether the paddle contact height moves the results.



---

The 68 vs 85 comparison and the median of 64 vs 65 don't settle anything, so I'll keep 185 and maybe try 181 instead. A better use of time might be looking at how the paddle shrinks at high speed, especially the late phase of the best run, since the big points come when the wall gets broken through. I also want to check the dead zone and jitter: with dz=0.3, FIRE triggers when |Σ| drops below 0.3, while Σ is around 2 there.

I'm checking whether the dead zone actually matters: the paddle error sits well inside a couple of pixels, so the threshold never really bites and the jitter stays small. Next I'm weighing whether aiming for paddle edges would give steeper rebounds, and I'm leaning toward thinking about where the ball will land after it bounces off the bricks, since the trajectory is long enough that folding off the walls could throw the estimate off.

I'm weighing whether the paddle has enough time to reposition during the ball's descent, so predicting where it lands during the upward phase matters less at low speeds. I'm wondering whether the misses come from wrong ratios for certain brick classes, double wall bounces, or the ball clipping the paddle edge, and I'd need a denser trace near the deaths to tell which.

I'm going to keep the current model and try a faster vertical-speed variant for the 103 classes, with 12 rollouts left.



---

The two configurations look nearly the same on average: 73 versus 85 in the mean, and medians around 72 versus 65, with similar episode lengths, so this is probably noise. Since the evaluation averages 100 episodes at 27,000 steps, the mean is what I should care about, and it's being pulled up by a few rare breakthroughs. I'd like per-class velocity data from the traces to understand those breakthroughs better, but the usual way of getting that doesn't work cleanly here.

I'm weighing whether the paddle's width changes after the ball hits the ceiling, since that could push my offset off by four or five after a breakthrough. I'd like to trace a batch of unexamined RAM bytes, especially 108, in the next run to see which ones change in the best episode.



---

I'm checking whether the 108 value that changed to 6 at step 1320 relates to paddle size, since it reverted to 8 after the life was lost at 1430, and I'm considering how the paddle's pixel width and center position might be encoded.

I'm weighing whether the paddle's apparent width scales the offset, and I'm settling on a moderate coefficient of 1.5 since the width change from 8 to 6 may not map linearly onto the 108 value. Late-game paddle shrinkage is the only place this really matters, and both offset values perform about the same, so the compromise seems reasonable.

I'm working out the unit count for the classes: I'll give right-moving classes only two units each, the main one plus a fold unit, since a right-moving ball can only land past the right edge, while left-moving ones only need to check below the left edge. That brings the total to 52 units in layer one and 42 in layer two.

I'm settling the final details: the velocity for the fastest class comes from a linear extrapolation, and then I'll run the final check rollouts. I'm also noting that a dy byte between 5 and 127 has no handling yet, so the paddle would just stand still there, though that case should be rare. When the ball is moving up, the tracking term also needs the 108 adjustment applied to its offset.



---

Results sit around 81 and have plateaued near 80, so I need a qualitative change rather than more tuning. I'll look at the worst trace from this run to see the miss contexts, including the one with 108, and what else it shows about what's limiting performance.



---

I'm checking the ball's position against the paddle: the ball is near the right wall and moving right, and the paddle is at 172 when it should be near 183, so the paddle is lagging behind.

I'm working out why the paddle keeps arriving late: at the moment the ball crosses the paddle's zone, the paddle sits about 11 pixels behind where I want it, since it lags when moving. I'm considering blending the two position estimates so the final approach is less lagged.

I'm weighing a 50/50 blend of the input and actual paddle positions, but it would just leave the paddle trailing while it moves. The miss more likely comes from the ball's speed: right after a lost life it serves slowly, so my assumed ratio is off and the predicted landing point shifts late as the ball approaches.

I'm trying to pin down which RAM byte separates the slow serve speed from the faster 1.5 speed after a paddle hit. I'm checking the probe steps: the serve moves at vy=1, and after the first paddle hit the vertical speed jumps to 1.5, so my assumed ratio was wrong for the early part of each life.

I'm checking whether byte 107 tracks the serve speed: it reads 0 during the slow serve before the first paddle hit, and takes values like 65 or 129 after it, so its low bits might be a speed level that sets how fast the ball moves vertically.

I'm settling on a rule for the slow serve: when byte 103 is 1 and byte 107 is 0, the ball moves at vy=1, so the horizontal-to-vertical ratio should be doubled, with the 107 threshold separating it from the faster case where 107 is above zero.



---

The first result came in around 73, which isn't better, and the spread across runs is wide enough (roughly ±8) that the difference doesn't mean much. I'll look at the worst trace to see how it behaves on the 107 case and where it misses.



---

I'm tracing the byte values 107 as they change: 65 after the paddle hit going up, then 129 and 130 and back to 66, with the bit pattern suggesting the low bits count hits, and I'm checking whether that count tracks paddle hits, brick hits, or both.

I'm checking the ball's path after the lost life near the left wall, working out where the paddle would need to be to catch it, and I'm not sure whether the paddle should have made contact before the life was lost.

I'm weighing whether the paddle's position lag behind the target might be better modeled as backlash, with the paddle stopping up to about 11 pixels off either way, which would be wider than its half-width and could account for many of the misses. I'm also considering a blended target between the two paddle readings, though I haven't worked out the offset yet.

I'm checking whether the paddle positions cluster around the same sum, which would point to backlash or hysteresis of about ±10 pixels rather than a simple lag, since a lag would give a continuous spread of sums.

I'm reading the paddle positions as a dead band of about ±11 pixels, so the paddle only moves once the input has moved past that slack. Using the 72 reading directly then makes sense, but reversing direction costs roughly 22 pixels of input travel, which would make the control jitter inside the band.

I'm weighing a control scheme that steers on the paddle's own position rather than the input, since bang-bang on the paddle might oscillate but stay bounded. The earlier runs were less accurate on that path than with the input-based prediction, so I'm reconsidering whether blending the two targets could keep the benefit of the input reading while accounting for the dead band.

I'm weighing whether the paddle's offset should flip sign depending on which side it's approaching from, since with a ±11 pixel slack the aim point could land at either the paddle's edge or past it. A fixed offset of 9 would then leave the ball near the rim rather than the center, which might explain the misses.

I'm checking whether using the paddle's own position as the target would fix the dead-band problem, since the controller might then reverse too late after overshooting. The mismatch between the input-based and paddle-based targets seems to come from how reversing direction costs about 22 pixels of input travel.

I'm weighing a controller that targets the paddle's own position, with a rule to move right when it sits well short of the target and left when it sits well past, plus a tolerance band in between to stop it from oscillating.

I'm working out a three-way rule for the paddle: move right if it sits short of the target, left if it sits past, and otherwise nudge the input toward the paddle's position so a correction only needs about 11 pixels of travel. That would need extra units per case, which adds cost.

I'm considering a blended target, q = β·x72 + (1−β)·(235 − x70) + offset, so the paddle's real position counts a bit more heavily than the input and the correction error shrinks from about ±11 to roughly ±5.

I'm now trying a blended target with β around 0.7, so the input and the paddle's real position both count, which should shrink the steady-state error to about 3 pixels. I still need to settle the offset, since the best value earlier was tied to a ±11 error I'm now trying to reduce.

I'm settling on a blend where the target is 0.7 times the paddle's real position plus 0.3 times the input-derived position, with an offset around 8, and I'm writing the q coefficient into the network with a term for the extra correction.



---

I'm rechecking the sign on the x70 term in build5. The unit adds -S*cp to W1[70], so the coefficient on x70 comes out to -S*cp, which is the value I had called c70. That means S*cp equals -c70, and I'm about to substitute that into the new q expression.

I'm checking the sign on the x70 term once more and confirming the substitution into the new q expression works out, which it does. The reweighted version came out much worse, so the paddle-72 signal looks more like a lagged display than a true backlash term. I'll go back to the build5/build4-style model with the x70 feedback, and keep the offset at 9, since that gave the best result among the values I tried.

I'm now questioning whether the 0 value on the wheel-like readings could mean a fast leftward ball rather than a stationary one, since the sequence of 0, 1, 2 on one side and 255, 254 on the other doesn't map cleanly onto a symmetric encoding.

I'm checking the trace again for the cases where the 0 value appears on the ball's velocity, since a fast leftward ball might be encoded that way and would explain the misses. At step 320 the paddle sits at 77 while the target is near 94, and the lag accounts for that gap. The same pattern shows up at step 512, where the ball is at x=67 and the paddle is at 55, so the missed catch looks like the same timing issue.

I'm weighing whether the 0 value on the velocity might mean a fast leftward ball, since the traces with dy=2 and that 0 seem to come before the lost lives. If so, the horizontal speed could be larger than I've been modeling, so I'm considering tracking the ball directly for those cases.

I'm checking whether a value of 0 means a slow leftward move, since the ball went from x=101 to x=57 over that stretch, which fits a horizontal speed of about -0.5. After the wall bounce, the same 0 might not distinguish a slow rightward move, so the model could keep predicting leftward motion and send the paddle the wrong way.

I'm checking whether the 0 value on byte 105 could mean a slow move in either direction, since the ball went from x=57 to about x=85 after the wall bounce, which fits roughly +0.5 per frame. If 0 covers both slow directions, tracking the ball directly might be the safer choice than trusting the sign, though the error from a ±0.5 guess could still reach around 24 pixels.

I'm going to set the horizontal speed for the slow class to zero and track the ball directly in those cases, as a hedge against the ambiguous direction, then rerun the build5 model.



---

The score is 91.5, my best so far, though the result is noisy. Next I'll check its worst trace to see how it behaves when 105 is 0 and whether it gives any hints about direction.



---

I'm noting that the low bits of 107 count up to 16, since 144 equals 128 plus 16, which may track speed level or hits. The 103 value shows 3 out of 252 at high levels, so I'm guessing the vertical velocity is around 3 or 4. Looking at the worst life losses, step 162 leads to a new serve at 189, and the 216 to 243 sequence is where I lost a life, with the ball passing at y 197 while the paddle sits at 175.

I'm checking whether the paddle at 175 should have caught the ball near x 172–180, since the numbers suggest it should have overlapped. The hit zone might not span the range I assumed, or the contact height is lower than I thought. I'm weighing whether to raise the paddle's vertical position, though an earlier test at 189 gave results within noise.

I'm considering whether the paddle's collision zone might be offset from where I assumed, or whether the contact height is lower than I thought. Another idea is that the hit check uses a position derived from the 70 value rather than the lagging 72 display value, which would explain why mixing them hurt. I'm also weighing whether the ball's direction should affect the angle, though that seems unlikely to be the key factor. I'm leaning toward testing faster vertical velocity classes for levels 3 and 4, since the trace shows those values often, but single rollouts are noisy enough that I'd need care interpreting the result.

I'm weighing whether to expand the prediction blend toward the ball's current x, but I'm more interested in the upward phase, where the paddle tracks the ball and a fast descent of about 95 px could still be reached in roughly nine steps, so the prediction offset should mostly stay manageable.

I'm still reasoning about how the paddle's contact timing and prediction ratio errors could cause misses, since a wrong ratio shifts the target during descent and a lagging paddle would then arrive late.

I'm considering a dead zone to reduce jitter, though with the paddle lagging it may not matter much. I'm also thinking about a ceiling-reflection prediction for the upward phase, but that depends on a brick-bottom value I can't rely on.

I'm weighing whether speeding up the paddle would help, though its input speed is fixed, so that's out. I'll also try a faster vertical velocity mapping for the upper classes, roughly 1.5, 2, 3, and 4.



---

The two groups came out close, 98 against 91.5 on the mean and 73 against 76.5 on the median, which looks like noise rather than a real difference. I'm also checking the 70 quantization and the Cp=235 midpoint, which seems to hold up. Since the mean is driven by a few rare large episodes, that may be the thing to test next.

I'm checking the long-run setup: 27000 steps, rollouts cut at 5000, lengths around 1300, so truncation isn't a concern. The paddle shrinking and the -3 adjustment after the breakthrough look fine. With five rollouts left, I'll build b7 as build6 with the slow serve phase (vy=1 while 107=0) and run it, since the data supports that phase.



---

The scores are clustering around 90 with a median near 77, so we seem to have hit a plateau. With four rollouts left, I want to find what's holding it back, so I'll look at the worst trace from b7 to see where the losses come from.



---

I'm tracing the ball's path near the left wall: at step 616 it was at x=55, y=198, moving toward the paddle at x=64, and it looked like it would be missed. I'm checking how the wall bounce at x=55 sends it back to the right, which puts the paddle around the right spot.

I'm weighing whether the paddle was actually aimed at the right spot, since the wall fold may have put the ball somewhere around x≈56–64 at contact. I'm also considering whether the sensitivity to the offset value means a nearby setting like 7, 10, or 11 would work better, and whether the 105 scaling ratio is worth testing with my last few rollouts on the top candidates.

I'm weighing whether to test the offset direction on the b7 candidate next, since paddle lag may matter more than the Ypad value, and I'm still uncertain about the vy values for the 3 and 4 classes.



---

I'd used up my rollout budget by miscounting, but the weights I'd saved before the rollout failed are still there. I need to re-save the final choice, and I'm going with b7 since it's the principled pick, then verify it.



---

I've got enough to write the answer now, so I'll give it honestly, flagging that the results are noisy and that performance seems to plateau around 90.

