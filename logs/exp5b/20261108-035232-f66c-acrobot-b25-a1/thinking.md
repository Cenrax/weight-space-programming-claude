I'm sketching a controller for Acrobot and recalling simple heuristics: pumping energy by applying torque in the direction of a joint velocity, possibly `action = 2 if obs[5] > 0 else 0`, which I'm not certain of.

I'm weighing a sign-of-velocity heuristic for Acrobot, but I'm not sure of the reported return, so I'll test it with rollouts. Next I'm building a small network with a tanh hidden unit on the joint velocity and an output that favours one torque over the other. I'll also write a helper that turns a linear score into the action so I can tune the weights.



---

The sign(dtheta1) heuristic failed because the second link keeps spinning, so I'm switching to an energy-based idea: torque in the direction of dtheta2 adds energy, since power is tau times dtheta2. Setting the action to sign(dtheta2) should pump energy in most effectively, so I'll try that.



---

The -88 result is decent, but the best known Acrobot results sit around -60 to -70, so there's room to improve. I'll try mixing the signals, like a switching surface built from dtheta2 plus a scaled dtheta1, and maybe a sign-based variant.



---

Comparing the results against known DQN benchmarks, around -60 to -75, the current -88 average doesn't look too far off, so I'm aiming for something near -70. The variance seems to come from a few long episodes, so I want to check the traces of the worst one, especially how the sign of dtheta2 drives the policy, and think more about the physics behind it.

I'm considering a position term for the second link, something like adding or subtracting a multiple of sin(theta2) to the sign signal, and I'm weighing a Spong-style desired angle for theta2 based on atan(dtheta1). I'm noting that a bang-bang torque with positive feedback on dtheta2 would work against that PD-style approach, so the two would behave differently. I'll run a coarse search over the weights with rollouts, since 21 are left, and then check the traces of the worst episode.



---

The link 2 is spinning freely, so the energy is going into rotation instead of lifting. I'm thinking about a fix: stop pumping when its angular velocity gets large, or reverse it. A saturating nonlinearity might help, something like tanh minus a linear term, which goes positive for small values and negative for large ones. I could split that into a sign-like unit driven by tanh and a roughly linear unit.

I'm weighing a two-unit version: one tanh unit that acts as a sign switch on dtheta2, and a near-linear unit that gives the braking term once speed passes a threshold around V. I'm leaning toward V around 10, though I don't yet know what typical peak dtheta2 is in successful episodes. The next step is a general builder where the first layer is a list of feature units and the second layer is a set of combination weights.



---

I'm noticing the spin stays near the threshold, so braking hard whenever the speed exceeds it would just hold it at that level rather than stop it. The spinning state looks stable on its own with link1 hanging, so I need a different approach that slows it to zero. The real problem seems to be that I'm pumping the wrong mode, so I'm considering a better pumping strategy.

I'm trying a pumping rule that also takes link2's angle into account, using a PD-style term on theta2 with a bang-bang torque that should chatter but still add some energy. Since the energy starts at zero, I expect little to happen at first, so I'll try alpha=1 with a small derivative gain of 0.1.



---

I'm testing whether the sign of the theta2 term is what matters for pumping energy into the acrobot, and I'm thinking of trying alpha=-1. Using sign(dtheta2) alone seems to work, so I'm considering adding a term that depends on dtheta1 and the theta2 angle. A dtheta1 offset of +0.5 made things worse, and I'm checking the negative side next.

The -0.5 offset gave the same result as baseline, so I'm going to test alpha=-1 with the Spong formulation as well.



---

Still weighing the energy-pumping idea, and I'm testing a few combinations of the feedback terms. I have 16 budget left, so I'm trying to find a more principled design, starting from the total energy and the power the torque delivers.

I'm weighing whether the goal really needs full pumped energy, or just the right configuration at the top. Holding the second link straight once there's enough energy should make the tip highest, so the approach may matter as much as the pumping. A quick parameter search over a few feedback terms seems like the most efficient use of the remaining budget, since the pumping baseline scores around -88 against a median of 73 steps.



---

The first try came out more consistent but the median is higher, around 80. I'll try k=0.5 as a middle ground, and maybe combine it with the dtheta1 term too.



---

I'm wondering whether the action frequencies being asymmetrical at 0.646 comes from ties, but that doesn't seem right, so it's probably just noise. I'll set k=1 for consistency and try k=2 to check the trend.



---

I'm trying k=4 next, and I'm also going to add a dtheta1 term.



---

The k=2 run landed at -80.5 and k=1 at -82, so the landscape looks noisy. I'll add the dtheta1 term with k=2 and try c=-0.5.



---

I'm weighing a few options for the next configuration: c=+0.5 with k=2, since c=+0.5 alone did worse before, plus a phase-dependent term or a policy switch for high energy, though my remaining budget of 10 makes that harder. I'll try c=-1 with k=2 next.



---

The plateau near -80 suggests I need a structurally different approach rather than tuning. I'm considering cos(theta) terms or the tip-height feature, and specifically a sin(theta1) term added to the score. I'll test that with coefficients +1 and -1 on the base [k=2, c=-0.5], starting with -1, and I have 9 rollouts left.



---

The bimodal split between 61 and 97 steps seems to hinge on which way the first swing goes, since the starting state is nearly at rest. I'm estimating how many swings it takes to build up energy, and whether the gap of roughly one swing period fits the link's timing.

I'm weighing whether a structural idea could break through the plateau, since the budget is down to seven. Reflection symmetry rules out a constant bias, and it also means any cosine-of-angle term has to multiply something odd, so I'd need to stick to the linear terms or move to nonlinear, energy-dependent gains.

I'm going to test a few more candidate parameter sets on more samples, since the differences between them look like noise. I also want to check a sign-flipped variant of the sin term, with a small positive weight on dtheta1 and k=2.



---

All the runs sit around -81, which means the current approach isn't moving anything and a structural change is needed. My idea is to focus on the final phase, so I'll compare the best trace from a fast episode against the worst one to see where they diverge. Let me go through the traces from this run.



---

The step-65 state has link2 folded back near pi, so its sine term is flipping sign and the spinning starts from there.

I'm checking whether the sine term near pi drives theta2 toward pi, which would make that equilibrium stable instead of unstable. A half-angle form would fit the behavior better, and I'm considering whether a small hidden layer could approximate it, using a tanh unit to capture the sign of sin(theta2) and combining it with the cosine term.

I'm thinking about building a sawtooth-like feature for theta2 out of tanh units placed at different angles. Each unit's pre-activation is a linear combination of x2 and x3 that equals sin(theta2 - phi), so each one acts as a soft step at a chosen angle, and summing several of them could approximate the half-angle behavior.

I'm now checking whether a staircase of soft steps, one per angle, can recover the angle itself, and seeing that a sum of sign functions over evenly spaced directions gives a triangle wave rather than a sawtooth.

I'm checking whether a sawtooth-like feature can be built from the staircase idea. A triangle wave from evenly spaced steps still decreases past π/2, so it doesn't give the discontinuity at π I need. Getting that discontinuity would take a product or AND between the layers, which makes the construction messier.

I'm weighing whether it's worth going further with the sine-term fix, since the spinning may come mainly from the velocity pumping term. An alternative is to switch policies at high energy and hold link2 aligned with link1, but that would need product features I haven't worked out. With only five rollouts left, I'm inclined to accept a score around -80 unless a simpler change looks promising.

I'm weighing a gated controller that switches behavior once link1 is above horizontal, since the sine-based fixes keep running into the same product-form problem.

I'm settling on a switching idea: when link1 is above horizontal, straighten link2 with a PD term toward zero, since full extension already clears the height threshold once link1 is high enough.

I'm working out how to gate between two controllers using a tanh unit that switches on whether link1 is above horizontal, and I'm weighing a big-M style combination so the sign of the output follows whichever policy is active.

I'm setting up a gate that switches between the two controllers: one tanh pair adds and subtracts a large term based on the gate, so only one controller's sign survives in the output.

I'm setting the gate threshold on the cosine of link1 and considering whether small-gain tanh units can stand in for linear features. For the top phase I'm weighing a simple PD term on link2, though the benefit looks uncertain given how few steps that phase lasts. I have five rollouts left, so I'll try it once.

I'm going with the PD controller for the top phase, though I'm not sure how much it will help. I'll also zero the torque or brake link2's relative motion as an alternative. The worst trace shows link2 folding back near θ2≈π once link1 is up, so I'm gating the controller on cos θ1 < -0.3.



---

I'm checking the sign of the pump term first, and the pre-activation looks right: the tanh term reduces to beta times coef times x for small arguments. With G at zero, the first two units both come out to tanh of the pump drive, while the other two should nearly cancel, but only if the top drive is small next to M, which I still need to confirm.

I'm checking the saturation bounds on the pump terms, since the largest drive is around 20 and that sits right at the M threshold, so it may be borderline. Then I'm considering a test where the top unit gets only the pump derivative with no spring, to see whether that shifts the behavior. I'll also look at the best and worst traces at the top phase before settling on that.



---

At step 65 the first link points straight up, but the second link has folded down, so the controller has to deal with the second joint pointing the wrong way. A plain bang-bang push toward zero would drive it the wrong direction near π, so I'm considering using the sign of sin θ2 to pick the direction whenever cos θ2 is negative.

I'm now weighing a saturated sine term, something like tanh of a scaled sin θ2, so the controller drives toward zero the short way across the whole range. The concern is that at the top, torque can't reorient a folded second link fast enough, so the better plan may be to have link2 extended before link1 arrives. With only four rollouts left, I'm leaning toward reverting to the simple linear policy and choosing among top candidates, or trying one saturated variant with an earlier gate threshold.



---

Reverting to the linear variant looks like the better bet, since the last tweak made things worse. With three rollouts left, I'm going to try a nearby linear setting, checking whether a slightly different c value does better, and then re-evaluate the strongest candidate to confirm it.

I'm weighing whether to try a higher k with a more negative c, or a lower k, using the remaining rollouts before saving the final candidate.



---

I'm settling on the parameter set [0,0,0,-2,-0.5,1], which gives a tight distribution with no outliers above 100 and a score of -80.75, and I'm saving it.

