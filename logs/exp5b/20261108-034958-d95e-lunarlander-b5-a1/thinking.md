I'm designing weights for a heuristic controller modeled on the gymnasium lander heuristic, where the target angle comes from a blend of two state terms and is clipped, and the hover target depends on the absolute value of the first state.

Now I'm working out the discrete action rules: pick the hover action when the hover target dominates and is above a small threshold, otherwise steer toward the angle correction when it's large enough. Next I'll think about approximating the absolute value of the state with tanh units, since the first layer can only give near-linear features.

Now I'm weighing ways to approximate |x| with tanh units, since a single layer can't produce it directly, and considering a stack of step-like tanh units placed at several offsets to build up a piecewise approximation.

I'm sketching the first layer as near-linear units for the state features, and I want to compute the angle and hover targets directly there, with the absolute value and the clip handled separately. For the clip, I'm considering a scaled tanh, which approximates the ±0.4 limit, and then the second layer will combine these terms linearly.

Still working out the first-layer units, with the angle target coming from a scaled tanh and the angular terms kept in a small linear regime.

Now I'm handling the leg-contact override, where the angle target drops to zero and the hover target becomes just the negative vertical velocity term, which the second layer will need to switch on.

I'm checking that the logits give the same choice as the heuristic: logit0 at 0.05, logit1 at A, logit3 at -A, and logit2 at H. Taking the max of those reproduces the action rule, since main wins only when H exceeds both |A| and 0.05.

I'm working out how the leg-contact override can fit into the logits, since the tanh layers can't gate multiplicatively. My current idea is to have the second layer hold the angle and hover terms in their linear regime, and handle contact with an extra bias term.

I'm trying a simpler route for the leg-contact case: rather than gating the angle and hover terms multiplicatively, which tanh layers can't do, I'd bias the logits so the do-nothing action wins once the legs touch, and accept a little loss in the vertical-speed correction.

I'm using a leg-contact unit that saturates near 1 whenever either leg touches, and adding it to the do-nothing logit. I'm also considering a gate on the hover term that switches it off when no legs are in contact, though a plain linear sum can't do that gating directly.

I'm settling on a simpler handling for the leg-contact case: rather than gating each term, I'll add a large bonus to the do-nothing logit whenever a leg touches, since the vertical speed is usually low by then anyway.

Now I'm settling on how to handle the |x| term: I'll approximate it with a set of step-like tanh units spaced along the x axis rather than a single nonlinear unit.

Now I'm thinking about how to set the scale for layer 2: a small scale keeps the tanh units near linear, so the comparisons against the thresholds stay accurate, while the layer 1 features run at a tighter scale of 0.1.

Now I'm checking whether using one shared scale for both the angle and hover terms keeps their comparison exact, since the same monotonic squashing then applies to both. The 0.05 threshold gets the same scaling, so it stays consistent with the heuristic. I'm starting to write the first-layer estimate for the angle term, which mixes the clipped angle target with the angular velocity.

Now I'm laying out the first-layer units: one for the angle-target term, one for the angular velocity, one for the hover terms, and a leg-contact unit, plus a set of step-like units for the absolute value. Next I'll set up the second layer's units that combine these into the angle and hover estimates.

Now I'm working out the hover term's weights: the step units each add a small constant plus a slope in |x|, and I'm checking what bias keeps the sum at zero when x is zero.

I'm choosing s2=1 and setting the logit0 bias so the do-nothing action wins once a leg touches, with the leg unit feeding into logit0 and the other logits reading the angle and hover units directly.

Now I'm settling on the leg-contact case: I'll add a large bonus to the do-nothing logit when a leg touches, and skip the extra gating on the vertical-speed term for now. I'll check that choice with a rollout before writing the code.



---

I'm seeing lots of crashes with returns between about 0 and 80, which points to a likely crash right after contact, where doing nothing on contact causes a bounce or a tip. I'll look at the worst trace first.



---

I'm checking the worst trace for a possible sign error in the angle control. At step 6 the target comes out slightly negative and gets clipped, and the angle keeps drifting further negative while the velocity stays negative too. I'm also wondering whether the heuristic itself is what's pushing it that way.

I'm checking whether the heuristic's angle target sign is correct, and it looks right: positive angle tilts the thrust toward -x, so a negative target counters the negative vx. Still, the lander touches down with vx around 0.5 and overshoots, and the angle correction at step 120 comes in slowly.

I'm working through the step-102 numbers: the target angle comes out positive while the actual angle is still tilted negative, so the thrust keeps pushing right. The main engine is firing on the hover priority, and the rotation is too slow to counter the drift.

I'm checking the side-engine logic now: action 1 fires leftward above the center, and I'm working out whether that torque actually increases the angle.

I'm checking the side-engine torque sign, and it works out: firing action 1 applies a leftward push above center, which increases the angle. Next I'm checking the angle target at step 30, where the deadband lets the tilt drift further negative while vx stays put.

I'm recalling the gymnasium heuristic's angle target: it clamps the angle to ±0.4 and drives the angle correction from a 0.5 gain on the error minus the angular velocity. I'm wondering whether this specific form of the heuristic is what's leaving the tilt to drift in this trace.

I'm confirming the sign convention: a negative angle pushes the lander rightward, so a negative target is correct when vx is negative. The remaining question is why the trace still crashes, since at step 164 the angle is around -0.2 with vx near 0.49 and the heuristic has no angle correction to apply once a leg makes contact.

I'm weighing whether the heuristic's known score of around 200 suggests my own simulation has a discrepancy, and I'm considering a PD-style controller with stronger horizontal velocity damping and angle gain instead of replicating the heuristic exactly.

I'm considering whether the angle response is simply too slow: the tilt sat near -0.29 while the main engine kept accelerating the lander right, so vx overshot to about 0.64. Hovering takes priority over the side engines, and I can't fire both in one step, which makes stronger angle gain look like the main lever.

I'm weighing a change to the angle control: boosting the side-engine priority when the angle error is large, and raising the angle gain so the tilt tracks its target faster. I'm also considering whether the landing needs its own hover adjustment.

I'm weighing whether the worst crashes come from the landing phase, since they happen late with vx still around 0.5. I'm considering whether adding a hover boost near the ground would help, but the hover term already includes a downward term that should cover that.

I'm now going to gate the side engines on leg contact: once a leg touches, the angle term goes to zero and the main engine just follows the vertical speed.

I'm now working out how to gate the side and hover units on leg contact, using a sigmoid-style term L so that once a leg touches, the side engines are suppressed and the main engine just tracks the vertical speed.

I'm writing the contact gating for the hover unit: a sigmoid term that is near 1 once a leg touches, pulling the hover logit toward the vertical-speed term, with a separate linear unit on vy feeding it.

I'm settling on the contact gating as it stands and deciding to just test it in the next rollout.



---

The landing went wrong because the horizontal speed at touchdown was too high, around -0.4, so the lander slid and tipped onto one leg before flying off. Looking back at steps 110 to 130, the target angle formula was giving about -0.31 while the lander was still drifting, so the angle kept swinging.

The horizontal drift stays unfixed because the main engine barely fires near the ground, so there's no thrust to kill sideways speed. The angle target also lags behind the drift, which points to a slower descent or earlier speed correction as the fix.

I'm working out the angle control gain: the sliding-surface math gives a time constant of 0.4/k seconds, so a gain of 0.5 takes about 40 steps to settle while a gain of 1.0 takes about 20. I'm checking whether the side engines can sustain that faster response.

I'm checking whether side-engine impulses can deliver the faster angle response, estimating about 0.04 to 0.05 observation change per step from the early trace, so a gain of 1.0 looks feasible. I'm also considering a modest change to the horizontal-velocity term in the angle target and possibly adding a term tied to descent speed near the ground.

Three rollouts are left, and the variance across 32 episodes is high enough that the mean will be noisy, mostly from the crash count. I'm going to change only the angle gain for now and then compare, and I'll check the outlier from the first attempt where the lander slid on contact.



---

The mean reward jumped to 265, but the worst case still crashes: it starts with a high initial vertical velocity of -0.55, drops fast to -1.05, and hits the ground at -0.7. I suspect the hover priority is too weak when the downward speed is large. At step 18, the formula gives zero, which is why it isn't correcting.

I'm lowering the velocity-correction gain in the hover target, so the descent is slower and the landing speed is closer to zero. The trade-off is more fuel use and more time, but the mean is already at 265. I'll try a middle value of 0.65 and tighten the angle clip to 0.4, and if the worst case gets worse I'll revert.



---

The landing hit 278.5 with everything intact, though I could push for more, the worst case already cost 40 early on from free fall, so I'll stop here and keep save #4. A final rollout would only re-run the same weights, so I'll skip it and just confirm the saved weights match P.



---

The task is wrapping up now.

