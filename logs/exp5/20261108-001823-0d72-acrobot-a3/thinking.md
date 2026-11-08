I'm working on a policy for Acrobot and recalling a simple heuristic that drives the torque by the sign of the first joint's angular velocity, which is an energy-pumping idea.

I'm recalling a few candidate heuristics from memory, like using the sign of the first link's angular velocity, and I'm doubting that the simple sign-of-dtheta2 version works well. Reasoning through energy pumping, torque on the second joint raises total energy, but the second link could just spin without lifting the first link, so the first link probably has to swing too. I'm also noting that the sign-of-dtheta1 rule seems to give results somewhere in the range of -80 to -10.

I'm weighing the coupling effect of torque on the first link's motion: applying torque at the second joint reacts on the first link, which in the linearized dynamics should slow it, though gravity on the shifted center of mass might offset that. I'm reminding myself that energy input comes from torque times the second joint's velocity, not the first's, and I'm recalling a hand-crafted policy tied to the first joint's velocity sign that I think I've seen.

I'm working out the linearized mass matrix near the bottom position, adding up the inertia terms for both links to get the first diagonal entry.

I'm setting up the linearized Acrobot model at the bottom position: the mass matrix, the gravity stiffness terms, and a generalized force that only acts on the second joint. I'm also noting that torque times the second joint's velocity is what pumps energy into the system.

I'm working out the energy target: the tip has to rise above height 1, and the potential energy at the bottom is about -2g. Checking a few configurations, the one with the first link horizontal and the second pointing up only gets to the threshold, while putting both links at 2π/3 reaches it with a lower potential, so the minimum energy I'm estimating for the swing is still under review.

I'm sketching how the energy-pumping idea would work with a bang-bang torque on the second joint, and I'm expecting the swing to build up slowly enough that it could take a while to reach the height.

I'm checking whether a sign-of-dtheta1 torque rule actually pumps energy, since power input is torque times dtheta2 and that only adds up when the two velocities share a sign. To see which joint motion dominates, I'm solving the linearized eigenproblem by setting the determinant of the stiffness-minus-mass matrix to zero.

I'm solving the characteristic equation for the linearized Acrobot, which gives two eigenvalues, and converting them to natural frequencies for the two modes. Next I'm setting up the eigenvectors from the first row of the stiffness-minus-mass matrix.

Working through the slow mode, I find the second angle moves with the first, roughly 2.8 times as large and in the same direction. In the fast mode the two angles move in opposite directions. That same-sign relationship in the slow mode is what I'm checking against a sign-of-dtheta1 torque rule, since it would pump energy there.

I'm checking the slow mode, where the second angle moves about 2.8 times the first, so the sign-of-dtheta1 rule would pump energy there while damping the fast mode. Then I'm estimating the tip height in that slow mode, which only reaches about 0.33, so it falls short on its own and needs the nonlinear dynamics to finish the job.

I'm recalling a Spong-style swing-up idea, where the second joint is driven to track a reference angle that depends on the first joint's velocity, so the second link bends in the direction the first link is moving.

I'm settling on the view that the torque is too weak to position the second link directly, so the swing-up has to come from resonant energy pumping, with the torque timed to the joint velocity over many swings.

I'm weighing whether greedy pumping with tau = sign(dθ2) is the right move, since it maximizes the energy gain at each instant. The worry is that the energy might pile into the fast mode rather than the slow swing, but I'm reminding myself that extra energy in either mode still helps the tip eventually clear height 1.

I'm checking that the energy-greedy rule tau = sign(dθ2) makes sense, since it pumps the most energy at each instant, even though the fast mode gets excited too. I'm also recalling a gist that uses the sign of the second joint's velocity as the action, though I'm not certain of the details.

I'm recalling another heuristic that uses the sign of the first joint's velocity, torque = +1 or -1, which people report scores around -90. I'm considering checking it with a crude simulation, but the linear modal analysis looks more tractable, so I'm starting to set up the modal input gains for the first mode.

I'm working out the modal input gains: mode 1 gets about 0.116 and mode 2 about -0.80, so the fast mode responds much more strongly to the torque. With greedy pumping both modes grow, but I'm checking which amplitude grows faster.

I'm now comparing the two modes' input gains: the fast mode responds about seven times more strongly, so greedy pumping mostly excites it, while the slow mode grows far more slowly. I'm also starting to write out the tip position formula, sin of the first angle plus the sine of the combined angle, to see how each mode moves the tip.

Checking the sign(dθ1) rule against the linear modes: it looks like it pumps the first mode while damping the second, since the second mode's joint velocities oppose each other.

I'm estimating how long the slow mode would take to build up enough amplitude, which comes out to around 200 steps, far more than the 70-90 steps typical of good episodes. So the linear picture alone is too slow, and I'm turning to the energy input rate under bang-bang forcing to compare how fast the fast mode grows.

I'm estimating the energy in the fast mode from its modal mass, which gives about 8.9 times the squared amplitude, and the energy growth rate works out to roughly 3.2 times that amplitude. That puts the amplitude growth near 0.18 per second, so reaching the target energy would take around 47 steps. I'm also starting on the matching energy estimate for the slow mode.

I'm working out the minimum potential energy needed for the tip to reach height 1, treating the two link heights as a constraint that sums to 1 and expressing the potential in terms of those heights.

I'm working out the minimum potential energy the tip needs to reach height 1, which comes out to 0.5g with the second link straight up. Against the hanging baseline of -2g, that means the swing needs roughly 24.5 units of energy, likely closer to 30. I estimate that greedy pumping with sign(dθ2) would get there in about 45 to 50 steps, though nonlinear effects might shift that.

I'm weighing whether a few more steps of sign(dθ2) pumping would carry the system into a chaotic, high-energy regime where the tip clears height 1, and I'm estimating that could take roughly 80 to 150 steps. I also half-remember a notebook using a sign-of-dθ2 action rule that scores around -80, though I can't confirm the details.

I'm now considering whether a combined rule, like tau = sign(dθ2 + k·dθ1), could balance pumping the two modes better than the pure energy-greedy sign(dθ2) rule.

I'm settling on sign(dθ2) as the energy-greedy choice, though I know energy alone isn't enough. I'm also wondering whether the jackknife mode might spin link 2 around link 1, which could still drive the tip up through coupling, so I'd expect it to get there eventually in chaotic motion.

I'm checking the sign(dθ1) heuristic against the mode shapes again, since mode 1 has link 2 moving with the same sign as link 1 at about 2.8 times the amplitude, which seems surprising. Estimating the natural frequencies from the link masses and the gravity restoring term, both modes come out near the same frequency, so the in-phase mode looks plausible.

I'm weighing a sign(dθ1) rule against sign(dθ2): the first pumps only the slow mode, roughly 98 steps in the linear estimate, while sign(dθ2) gets energy up in about 47 steps but drives the jackknife mode. Nonlinear mixing could change both numbers, so I'm inclined to trust energy pumping with sign(dθ2), though I'm not fully sure of a remembered result around -100.

Now I'm checking whether combining the two signs, like sign(dθ1 + dθ2), could drive the tip above height 1. I'm working out where link 2 points in the in-phase mode and how high the tip gets at that point.

I'm checking the mode 2 case now: the second link's absolute angle moves opposite to the first, so its tip height depends on the cosine of a scaled first-joint angle. I'm working through the tip height at a first-joint angle of about 1.68 rad.

I'm checking the mode 2 case: link 2's absolute angle moves opposite to link 1, so the tip stays low while link 1 swings. That makes the jackknife mode a poor route to the goal, though nonlinear coupling might still carry energy across.

Now I'm noting that mode 1 is the in-phase swing where link 2 moves far more than link 1, so sign(dθ1) damps the jackknife mode but pumps energy slowly. I'm considering projecting the state onto the modal coordinates, so the control signal tracks the first mode's velocity without the second mode's contamination, and I'm setting up the modal matrix to do that.

I'm working through the modal velocity: η1' comes out to about 0.39 dθ1 plus 0.218 dθ2, so mode 1 is driven by a mix of both joint velocities. Using sign(η1') would pump the first mode, while plain sign(dθ1) also partially damps the second. I'm now wondering whether sign(dθ2) is still the better choice.

I'm comparing the power delivered to each mode under bang-bang torque, and sign(dθ2) pumps both modes at once. The linear estimate puts mode 1 near 100 steps, so the nonlinear behavior of the real system probably matters more than the linear numbers suggest.

Checking the linear amplitude estimate against the tip height, the link 2 angle at this point is nearly inverted, which gives a tip of only about 0.25. I'm noting that the linear model breaks down here, since link 2 spends more time near the top where gravity restores it less, so I'm thinking about how learned policies tend to swing link 1 with growing amplitude while link 2 bends.

I'm weighing whether to build something more elaborate, like an energy-based controller that also tracks phase, but I'd have to do it without a simulation, which feels risky. I'm leaning toward a simple heuristic and starting to work through how sign(dθ1) would behave from rest, beginning with a constant torque and how it shifts the equilibrium.

I'm checking the constant-torque equilibrium for the Acrobot: with the torque held at +1, the joints settle near θ1 ≈ -0.068 and θ2 ≈ 0.27. Next I'm turning to bang-bang switching, looking at how much each switch moves the first mode's amplitude.

I'm working out how the amplitude grows per half-period under bang-bang switching: each switch shifts the equilibrium by about twice the static deflection. For the first mode that gives roughly 0.065 rad of growth per half-period in θ1 and about 0.18 in θ2, so after around 40 steps the θ2 swing is already near 0.9.

Comparing the two pumping options, sign(dθ2) raises energy much faster, reaching a large θ2 swing in about 30 steps. That may push the system into a chaotic, nonlinear regime sooner, so the overall goal could be reached faster than with the slow sign(dθ1) pump.

I'm recalling that Spong's 1995 acrobot paper may have suggested an energy-pumping torque based on the sign of the second joint's velocity, though I'm not certain of the details. I'm also considering that sign(dθ2) could make link 2 spin continuously in one direction, which would hold the torque constant and waste time without raising the tip.

I'm comparing sign(dθ1) with sign(dθ2) as the pumping rule, and noting that a constant torque with link 2 spinning could just keep adding energy without raising the tip. A better option seems to be the swing-style rule, sign(dθ1), which is the common intuition of pushing in the direction of motion, and I'm checking how the torque sign convention affects the power it delivers.

Now I'm checking whether sign(dθ1) is essentially the bang-bang version of Spong's swing-up, since his link-two target bends toward the direction of link one's velocity, and I'm weighing whether limited torque makes that analogy break down.

I'm checking whether sign(dθ1) still pumps energy at larger amplitudes, since the power input depends on dθ2 staying aligned with the torque, and sign(dθ2) would guarantee the energy gain. I'm considering a compromise rule and trying to recall how learned Acrobot policies tend to switch torque.

I'm trying to recall published acrobot policies, like the oblique decision trees from Silva et al. 2020 or the linear policies that ARS reports, and whether a simple linear sign rule on the observations reaches roughly -80 to -90. I can't pin down the exact form of any of them, so I'm thinking about whether one of them might suggest a usable rule.

I'm sketching a rough nonlinear simulation in my head, since the linear numbers may not hold at large swings. For sign(dθ2) the link-2 angle near the top stays longer where gravity is weak, so link 1 can keep building amplitude until the tip reaches the goal height.

I'm weighing a variant that mixes both joint velocities, sign(dθ1 + c·dθ2), and noting that the mix c controls how much of mode 2 gets damped versus mode 1 pumped. Setting c to zero gives the plain sign(dθ1), which damps the second mode most strongly, though I'm not sure the linear analysis still holds once the swings get large.

I'm weighing whether a crude hand simulation of sign(dθ1) at larger amplitudes would be worth attempting, since a full RK4 run over about 80 steps is out of reach by hand, so I'm considering a rougher Euler pass with a smaller step.

I'm weighing a mixed rule, sign(dθ1 + 0.3·dθ2), against the pure options, since sign(dθ2) pumps mode 2 mostly and sign(dθ1) may lose energy at larger amplitudes. I'm also picturing how DQN-trained Acrobot policies look in videos, where link 2 flails more than link 1 and the amplitude builds over a few swings before the tip clears the line.

Now I'm estimating how far the swing amplitude grows per second under mode-1 pumping, and checking whether the goal height might be reached in roughly 15 to 20 seconds, which would fit the episode length.

Now I'm weighing whether sign(dθ1) still holds up: when the second mode dominates dθ1 early on, the torque damps it and leaves the first mode to be pumped once the second has faded.

I'm weighing whether sign(dθ1) is really the better pump, since it seems to damp the second mode, while sign(dθ1 + 0.56·dθ2) would only leave mode 2 randomly excited. Off resonance, the bang-bang forcing should only give mode 2 a small bounded response, so I'm checking whether that holds.

I'm estimating the forced response of the second mode to the square-wave torque at the first mode's frequency, which comes out to a small bounded contribution to dθ2 that partly offsets the first mode's motion.

I'm settling on sign(dθ1) as the better early pump, since the forced response of the second mode mostly shifts θ2 in phase with the torque. I'm less sure how the policy behaves once the swings get large, so I'm wondering whether a final-phase adjustment is needed.

Now I'm recalling the energy-based swing-up controllers from the literature, like the Xin and Kaneda approach and the Fantoni and Lozano formulation, which switch torque by the sign of the second joint's velocity once energy falls below the target and use a PD term on that joint to keep link two aligned with link one.

I'm laying out the energy target for a stretched configuration with link two near zero, which works out to about 29.4 units of extra energy needed, and I'm deciding whether to combine the torque sign rule with a damping term on the second joint.

I'm settling on a modest mix like sign(dθ1 + 0.2·dθ2) as a compromise, and I'm checking whether the energy term could go negative in the nonlinear regime. I'm also wondering whether sign(dθ1) stays safe when link 2 lags link 1 at large amplitudes, since that is when the torque and dθ2 could point in opposite directions.

I'm weighing a weighted sum like sign(dθ1 + 0.25·dθ2) as a middle option, though nothing I recall settles the right coefficient, so I'm leaning toward a reasoned choice instead. I'm also wondering whether the energy gain might favor sign(dθ2) at larger amplitudes, where the two modes mix, and I can't yet see how to tell an early swing from a late one using just the state.

I'm weighing whether to actually simulate the nonlinear dynamics coarsely by hand, but that's about 150 steps of arithmetic, so I'm leaning toward a linear estimate instead. I also recall a figure that sign(dθ1) solves Acrobot in roughly 90 steps, though I'm not certain that memory is reliable.

I'm settling on a rule that uses the sign of the first joint's velocity, with a small contribution from the second, and I'm weighing how it would behave once the swings get large.

I'm going back to a gravity-plus-position idea, the Spong-style target angle with a PD term, but a bang-bang version of it would fight the weak torque, so I'll keep the policy simple. I'm also weighing whether sign(dθ2) could let link 2 spin continuously, and whether sign(dθ1) avoids that.

I'm worried that sign(dθ1) could stall the energy gain in the nonlinear regime, leaving the policy stuck without reaching the goal. Sign(dθ2) at least never loses energy, though it might not get the tip high enough.

Checking whether the energy gain from sign(dθ2) and sign(dθ1) differs: in mode 1, dθ2 swings with about 5.27 times the first joint's amplitude, so the energy rate may come out roughly the same for both rules.

I'm comparing the two pumping rules: sign(dθ1) grows the energy-related amplitude more slowly, about 0.26 per second in root-energy terms, while sign(dθ2) roughly doubles that rate but drives the second joint into the jackknife mode.

I'm weighing sign(dθ2) as a greedy energy pump, but I'm doubting it helps, since a jackknifed link 2 with link 1 at around 2 rad never lifts the tip above the goal line. Mode 2 at large amplitude looks chaotic, so I'm not sure how energy moves between the modes there, and I'm leaning toward a blended rule like sign(dθ1 + c·dθ2) to hedge.

I'm weighing sign(dθ1 + 0.56·dθ2), which zeroes out mode 2 and pumps mode 1 at its full rate, and I'm checking whether adding the dθ2 term helps in the large-amplitude nonlinear regime. I don't clearly recall which heuristic performs best, but sign(dθ1)-style rules seem to work reasonably well, so I'm wondering what the late-phase behavior looks like.

I'm checking the sample state where link 1 sits near 1 rad and link 2 is almost inverted, which puts the tip around 0.46 and the next swings at higher tip heights. Working through the energy in that configuration gives about 25 units of gain, which is larger than the linear mode-1 estimate of 19.5.

I'm estimating that sign(dθ1) would need roughly 90 to 120 steps, while a good RL policy might manage 60 to 70. I'm wondering whether mixing in some dθ2 could speed things up by using the second mode's energy early, though I can't tell yet which way that goes.

I'm weighing whether to simulate the nonlinear dynamics by hand to check whether sign(dθ1) risks stalling in a limit cycle, but that's too costly, so I'm instead considering robustness more broadly.

I'm weighing whether sign(dθ1) could stall in a limit cycle at large amplitude, which would be costly. I'm considering a hedge, sign(dθ1 + 0.5·dθ2), that matches the pure mode-1 pump in its usual regime but falls back toward sign(dθ2) when link 2 is spinning fast.

I'm settling on the hedge τ = sign(dθ1 + 0.5·dθ2), since the 0.5 coefficient keeps mode 1 pumping while slightly damping mode 2.

I'm checking whether a smaller weight on dθ2, like 0.3, would damp mode 2 more strongly, and I'm noting it would still follow dθ2 when link 2 moves much faster than link 1.

I'm settling on the blend at around 0.4 for the dθ2 weight, since that keeps the mode-1 pump and still lets link 2 dominate when it moves much faster. The coefficient choice looks safe either way, and I don't see any reason the guard term would block reaching the goal.

I'm double-checking the sign convention: a positive torque should increase the second joint's acceleration, with a denominator of about 0.57 keeping that positive. I'm also verifying the potential-energy terms in the gravity matrix.

I'm checking the equations of motion: the first equation has the restoring term from the first joint's potential, which keeps the bottom position stable. I'm also confirming the mass-matrix entries, and I'm linearizing the potential terms to get the stiffness contributions for each joint.

I'm settling the torque as sign(dθ1 + 0.4·dθ2), since sign(dθ1) is what actually pumps mode 1, with the modal coupling of 2.8 keeping the energy gain positive. I'm about to check whether pure sign(dθ2) could be better.

I'm settling on a blend with a 0.4 weight on dθ2, since a bigger coefficient would start pumping mode 2 as well. I'm planning to implement it as a tanh-gated network with the logits set from the sign of that combined signal.

Setting the output layer so the middle action can never win: I'm putting its bias at -10 so it stays below the two sign-driven options, and the ties at zero fall to the lowest index.

I'm settling the output layer so the middle action can never win, and I'm writing up the explanation.

