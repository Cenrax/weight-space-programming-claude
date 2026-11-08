I'm sketching out the controller's logic from the gymnasium heuristic, starting with the angle target and the hover target that depends on the absolute value of the first state.

Still working out the controller logic from the heuristic...

I'm working out how to handle the clipping and the absolute value with tanh units, and I'm weighing whether a tanh-based approximation of the clip is close enough. The harder part is the leg-contact gating, since a multiplicative switch is awkward to build from saturating units, and I'm still looking for a workable pairing of units to do it.

I'm working out the gating: a tanh of the linear term plus a large contact-driven offset saturates to 1 when a leg touches, so the angle term drops out. I'll subtract a constant unit to cancel the offset, which only holds if the scaled angle stays small next to that offset. Two legs may contribute, so I'm combining them into a single contact signal.

I'm laying out the first layer now, using a tanh over the scaled angle target to stand in for the clipped value, and I'm about to add a unit to get the absolute value of the horizontal position.

I'm trying to get an even function of x out of tanh units, since I can't multiply. A sum like tanh(b + kx) + tanh(b − kx) with a large positive bias b falls off as |x| grows, so it could act as a stand-in for the absolute value.

I'm looking at whether an even function of x can be built from shifted tanh units, since |x| is what the hover target needs. Fitting the weights numerically would mean running tanh over sampled inputs, so I'd rather reason about the shape analytically first.

I'm checking whether a difference of two shifted tanh units can stand in for |x|: a negative bump centered at zero that flattens out to a constant away from it, with the constant chosen so the shape matches absolute value over the range I care about.

I'm checking the shape numerically at x=0.25, where the tanh terms give a value that should still be near zero, to see whether the bump stays close to the absolute-value curve.

I'm checking whether a shallower bump with a tighter width, around k=4 and d=0.25, would track the absolute value more closely for larger horizontal offsets.

Still settling the layer-one units, with the small-scale linear terms that keep tanh near its linear regime.

Still laying out the first layer: the horizontal-position terms feed the abs-value units, and the contact signal gets its own unit so it can gate the angle term.

Still working on the gated angle unit in layer two: the contact term should saturate it, and I'm keeping the scale small enough that the linear part stays roughly linear, so the comparison between the angle and hover terms holds.

I'm checking that the same odd, monotone tanh applied to both the angle and hover terms keeps their comparison intact, and then I'm looking at how a large constant gate could switch the angle term off during contact.

I'm now setting the gated horizontal-hover unit: it should be active only when a leg is in contact, so I'm combining the contact signal with a scaled vertical-velocity term that gets switched on in that case.

I'm setting the gating so the contact unit drives the output: the no-contact case needs a +1 correction and the contact case needs -1, which I'll handle with a single contact-driven term. Then I'm fixing the constant bias for the target and the logit for the angle, using the contact unit's output.

Now I'm settling the scale for the first-layer inputs, since horizontal velocity, angular velocity and height each have different ranges and need to stay in tanh's near-linear region.

Now I'm defining the absolute-value approximation from a scaled bump of two tanh units, with the output scale chosen to match.

Still writing the layer-two gates...

Still writing the layer-two logits...

Now I'm settling on the final tuning: the heuristic should land around 200 on average with some failures, and rollouts can push it higher from there.



---

I'm checking the controller's result: it reaches 240 on average, but landings with leftover horizontal velocity cause sliding and tipping, so the controller should cancel that velocity before touchdown. I'm also questioning the sign on the angle target, since positive x gives a positive angle, which would mean tilting left.

I'm working through the sign convention: a positive angle tilts the lander counterclockwise and pushes the main impulse toward -x, so a positive x correctly produces a tilt back toward center. Next I'm weighing stronger angle gains and horizontal velocity damping near the ground, and whether angle control should stay active when only one leg is touching.

I'm now considering a variation where the angle controller stays active only during ground contact, without the horizontal target, and I'm working out how to wire that as an extra unit in the network.

I'm checking whether to test a contact-only angle controller, one that keeps correcting tilt while a leg touches the ground but drops the horizontal target. I'm working out how that term would combine with the existing gated unit so it's off in free flight, and I want to add a parameter for its gain and maybe a larger contact threshold to reduce jitter.



---

Crashes keep coming up, so I'm going to look at the worst stack trace first.



---

The touchdown is coming in at a high horizontal speed, and the lander is still tilted. The bigger issue is that the tilt isn't keeping up: vx swung from -0.2 to +0.4 while the angle stayed near -0.2, so the target angle (0.5x + vx) ran ahead of the actual angle, and the angle correction looks too sluggish to catch it.

The angle is still lagging the target, so the main engine keeps taking priority over attitude during descent. I want stronger angle gains. My plan is to scale the attitude term by roughly 2, with ka at 1 and kw at 2, and test it with acg set to 0 as a baseline. I'll run one rollout per config so I can compare them cleanly.



---

The first result looks better, and I see that ka=1 also scales the target effect because at_scale includes ka*at. I'll try pushing further, perhaps ka=1.5 and kw=3, or raising kvx.



---

I'm checking the landing sequence: around step 80 the lander is near the ground at a small negative angle, then by step 90 it has tilted to about 0.2 with a fairly high angular velocity, so a bouncy touchdown on one leg seems to have tipped it. After that it appears to be sliding at roughly the same tilt on both legs and drifting away, and I'm wondering whether it's sitting off the landing pad on a slope.

I'm checking whether the lander's sign convention pushes it the right way: a positive angle seems to push toward -x, so at x=-0.2 a negative target angle should push it back toward the center, which matches what the controller is doing, just slowly.

I'm weighing whether stronger position correction would help, and noting the ka=2 runs timed out more often while ka=1, kw=2 had the most consistent spread. Since each 32-rollout batch carries roughly 7 to 15 of noise, I'm planning to try ka=1.5, kw=3, and some kx=1.0 variants with the remaining 19 rollouts.



---

The episode length jumped to 308, so I want to pull up the worst cases and the long ones to see what's driving it. My guess is the contact lift-off and sliding phase, where the contact gating stretches things out. After landing, the lander has to come to rest, and sliding slowly eats up time, especially with kx=1 and only one leg in contact. I'll print the worst trace first.



---

I'm reworking the touchdown: the lander hits the pad with a large sideways speed and slides off down the slope. The main fix is getting horizontal velocity close to zero before contact. Looking at the last approach, at x around 0.2 with vx near -0.6, the angle target clamps to -0.4, so it starts braking, just too late. I'm weighing a slower descent whenever the sideways speed is large.

I'm weighing a change to the hover target so it also grows with the sideways speed, which would keep the lander hovering until horizontal motion is killed. That means adding an absolute-value term for vx, and I'm considering a bigger angle gain on vx as well.



---

The lander came in at a score of 274.5 and every run landed, which is a big step up. Now I want to tune it: a faster descent should burn less fuel, and I also care about how fast it touches down. I'll try hvx=2 and kx=1 next.



---

The last run came out worse, with a longer trajectory and more fuel used. I could try hvx at 0.6, or keep it at 1 and tune the other parameters. I've lost the best and worst traces from the hvx=1 run, so I'll set that aside. Next I'll test hvx=1 with kx=1, since stronger centering should cut the landing offset.



---

I'm checking the landing: both legs touched at x=0.15 with a slight left tilt, but then the lander drifted right while tilting more and more. The pad runs from -0.2 to 0.2, so I'm wondering whether it slid off onto the downward slope, and I'm looking at the pad edge values.

Working out why the right leg lands past the pad edge: the pad spans roughly ±0.2 normalized, but the legs reach about ±0.067 past the lander's center, so at x=0.15 the right leg sits over the sloped terrain. Tighter centering matters near the pad edges. I'm also weighing whether a stronger velocity term in the angle target would stop the drift at low altitude.

I'm working the drift into a simple model: the horizontal velocity changes in proportion to the tilt angle, and the angle target combines position and velocity. Converting everything into normalized units per second, I get position changing at half the observed velocity and velocity changing at about -2 times the angle, which gives me a rough control law to check.

I'm reworking the control law in normalized units: the position error gain and velocity gain come out to a heavily overdamped response with a time constant of roughly 170 steps. Since the main engine only fires part of the time, the effective acceleration will be smaller than that estimate.

I'm settling on a faster gain pair, trying a larger velocity term so the drift gets damped more quickly, and I still need to check that the attitude loop responds faster than the position loop.

I'm settling the gains as a trial pair, with the position gain at 1, the velocity gain at 2, and the horizontal hover term at 1, then checking whether the descent terms still hold.



---

I'm working out the reward figures: the score was 275.9 with a low of 245.9. The real losses look like fuel use, the final distance from the pad, and the side engine, while the shaping terms telescope down to final minus initial.

I'm now estimating the fuel cost: roughly 124 main-engine frames at about 0.3 each comes to around -37. My plan is to land at x=0 and burn late at full thrust, a suicide-burn profile, to cut gravity loss, and I'm weighing a simple descent target that scales with height.

I'm weighing a gain of about 1 on the height-based descent target, which would aim for roughly double the current vertical speed and let the lander slow to zero as it nears the ground. If the vertical-speed gain is too low the response could be sluggish, so I'll try a gain of 0.8 or 1.0 and see how the final touchdown position turns out. The x-accuracy is already good at 0.006, and since time only matters through fuel, the side engines are cheap enough to ignore.



---

The last attempt crashed by coming in too fast, since the engine can't brake enough, so I'm reverting to hy=0.5. Going to 0.6 looks like only a marginal gain, not worth the risk. The fuel side offers little room, so I'll keep the current structure and tune it moderately, starting from the 275.9 baseline and testing acg=1 with the new base.

I'm weighing whether to gate the control on both legs touching, since a single leg contact makes the heuristic cut out and the bounce flips. I'm working out a contact indicator with a steeper threshold so that only the both-legs case switches it on, and I'll test that.



---

The timeout happens when only one leg touches down at rest, so I'm switching the contact gating back from "both" to "any". The wandering trace with the large initial vx overshot but still got there in about 300 steps, which is acceptable. Back on the any-contact baseline at 275.9, I'm now going to try contact attitude control with acg=1.

The "any" gating might have caused a crash on bounce earlier, so I'll skip the contact attitude idea. Next I'm thinking about lowering fuel use by letting the descent run a bit faster near the ground, which means offsetting the vertical hover target by about 0.1 so touchdown happens at a gentle sink rate.



---

I'm looking at the touchdown at -0.44 vertical speed, which crashed, and wondering whether the body made contact. The engine couldn't slow the fall in time because the free-fall had already built up speed to about -1.09. During the descent I'm seeing that attitude corrections are taking up main-engine time, and I'm working out a fix around that.

I'm weighing a scaling factor on the hover term so the main engine wins the comparison when descent is urgent, and I'm leaning toward a priority multiplier around 1.5 applied to the main-engine weights and the correction terms. The threshold for firing the main engine would drop to roughly H > 0.033 as a result, and I'm checking that this is still a sensible trigger.



---

The attitude and horizontal loops are oscillating, with vx swinging large as the hover keeps it going until vx is killed, which costs fuel. Reverting hp to 1 keeps the base at 275.9, and I have 11 rollouts left, so next I'll look at tuning kw relative to ka.

I'm weighing a PD tweak for the attitude loop, trying ka=1.5 with kw=3 alongside the current horizontal gains. Since run-to-run noise is around ±5, small differences will be hard to tell apart, and we may already be near the heuristic ceiling of about 280, so I'll test a few variants and then pick one.



---

Those gains look close, so I'm weighing a few alternatives: a higher proportional term with lighter integral action, or keeping the attitude loop and raising the damping to kw=3.



---

I'll try ka=2 and kw=4 next, to see how that setting behaves against the same case.



---

I'm weighing whether to adjust the main engine threshold separately from the side one, since side thrust is cheap but lowering the main threshold seems less clear. I'm also considering lowering the hover target to cut fuel use, or raising the x gain for better horizontal accuracy, and I want to look at the worst trace to see where the losses come from.

I'm going to pull up the best and worst traces from the current run to compare where the losses are coming from.



---

I'm noticing the worst episode started with a low initial shaping value, which seems tied to its starting velocity. Higher starting speed makes the initial shaping more negative and the return higher, so most of the variance looks like it comes from initial conditions. Ending up about 0.05 off target costs roughly 5 points.

I'm weighing whether fuel is the bigger lever: a faster, sqrt-shaped descent profile would save burn, and tanh has a similar steep-then-flattening shape, but the naive version lowers speeds and so costs more fuel, the opposite of what's wanted. The suicide-burn ideal points toward high speed up high and a constant deceleration, roughly sqrt(2ay).

I'm checking whether a linear velocity profile is the right shape, since it makes deceleration small near the ground and wastes time there. Trying a small constant touchdown speed looked promising but the one crash I saw came from attitude priority rather than the profile itself. I'll try v0=0.05 with the current config and maybe kx=1.5, since fuel is probably only worth about 10 more points.



---

Sensitivity is high, so I'm reverting v0 to 0. A negative v0, which means a slower touchdown, might lower the crash rate across the 100-episode evaluation. I'll test v0=-0.05 and check whether it's safer and what it costs.



---

I keep seeing the lander slide off the pad after touching down near the edge, so that's clearly the main failure to fix. Using the main engine during contact would just push it sideways, so the better move is to land nearer the center: a stronger horizontal gain, and hovering whenever it's far off-center. I'm considering raising the horizontal gain to around 1.0 for that phase.

I'm reworking the abs approximation with a sharper curve and tuning its width so |x| is well captured near the pad, with the hover offset saturating beyond roughly 0.3. I'll normalize it so the saturated value matches, then try a slightly lower horizontal gain of 0.8 so the lander hovers until it's centered.



---

I'm checking the constant C, which makes the approximation exact at 2d=0.4, and the quadratic behavior near zero holds up. At 0.15 it gives 0.108. I'm settling on k=6, d=0.2, and hx=0.8. The vx abs in build3 uses its own C formula with k2 and d2 defaulting to 3 and 0.3, normalized at 0.5, so it stays as it is.



---

I'm checking the landing: the lander touched down at x≈-0.113 with slight leftward speed, then slid away and sped up while tilting on one leg. The pad looks flat between -0.2 and 0.2, but the leg positions don't quite line up with that, and the descending y values make me think the surface might be sloped instead.

I'm working out the terrain: the helipad sits flat from about -0.2 to 0.2 in normalized x, with slopes beyond that. The lander is at x≈-0.13 with its left leg near -0.2, so it's right at the edge of the flat section. I'm wondering whether I can brake the sideways speed while the legs are in contact, or whether I should aim for a more centered, slower touchdown instead.

I'm reasoning that the left leg crossed the flat-pad edge around step 270, which is when the lander began accelerating on the slope. So touchdown has to land well inside the pad, with low horizontal speed, and I'm considering whether horizontal thrust during contact could push it back toward center.

I'm weighing whether a sideways push in contact could fix the slide, but that feels complicated. A simpler option is to strengthen the x-correction so the lander touches down more centered and with less outward speed. Most episodes already end near center, and my earlier best config had no failures, so this failure may just be noise.

I'm deciding whether to leave the contact controls as they are after a bounce, and I'll keep things as they are for now. Next I want to try raising the x-correction gain to 1.5 with the original absolute-value setup, to pull the lander closer to center, though I'm wary of overshoot.



---

The landing crashed because the vertical speed spiked downward early, then the descent stayed too fast and it touched down tilted. I think the core issue is that the main engine and side engines are alternating every second, so altitude control and attitude control are competing for the same time. With the higher attitude gains, the attitude term gets larger, which means the main engine only fires when altitude exceeds it, and it gets starved.

I'm weighing a gating unit that boosts the main engine only when altitude is high and the lander isn't in contact, with a moderate weight so attitude still matters. I'm working out how to wire the gate into the network weights, using the altitude input and subtracting the contact term.

I'm settling the gate's constants: a steep slope K around a threshold of 0.15, with a large contact penalty G so the boost switches off once the lander touches down. I'm also planning to keep the attitude gains at 2 and 4 as the base, and set the position gain back to 1.



---

I've got no crashes so far, but the mean dropped to 269.8, which could be noise or just longer runs, and I have 4 rollouts left. Mean lengths also grew, which is worth noting. Looking at the candidates, A has the best mean at 278.9 with no failures, a kx1.5 variant hit 276 but had one crash, and B is the best setting with a 0.4 boost, landing at 269.8 without any crashes. C with ka1.5 is still unresolved.

I'm weighing the config C option at 278.6 with no failures against the crash risk in the A family, which works out to roughly -9 per crash. A boost could avoid the fast-descent failure but might cost around 8 points of mean, so I'm considering a smaller boost of 0.3 with a higher threshold of 0.25 on config C, then making the final pick with the remaining rollouts.



---

I'm noticing the off-pad slide again and treating those as slide failures. With three rollouts left, I'm comparing the measured means: A without the boost is best at 278.9, C without the boost is 278.6, while the boosted versions come in lower at 273.5 and 269.8. The gaps look within noise, but the boost seems to cost about 5 points consistently.

The crash rate on the A family's kx1.5 variant might be around 2%, which would cost about 6 points on average, so the two options look roughly even once that's counted. Rather than settle it now, I'll spend two of the remaining rollouts on the two finalists, C without the boost and C with the boost at 0.3 and h0 0.25, and hold one back in reserve.



---

The C variant tested cleanly without the boost, at 279.8 and 278.6 over 64 episodes with no failures, so I'm going to finalize it. I'm weighing a small boost as a safety margin for truly urgent cases with a high h0 of 0.35, but untested changes carry risk, so I'll keep C as it is. Next I need to confirm the saved weights match the C build, since the boost=0 unit in W2 is harmless only if W3's weight for it is zero, and then check that the weights actually load.

