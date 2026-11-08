I'm sketching a controller for Acrobot and recalling a simple energy-pumping heuristic: pushing the torque in the direction of the first joint's angular velocity, perhaps with a term from the second joint. I'm trying to pin down the exact rule that reaches roughly -80 to -100 on average, and I'm leaning toward a condition on the sign of obs[5].

I'm weighing a few candidate rules for Acrobot. Pumping energy through joint 2 means applying torque in the sign of dtheta2, which raises total energy, though that may not guarantee the tip actually reaches the target height. I'm also unsure whether torque keyed to dtheta1 would do better, so I'll reason through the dynamics to compare them.

I'm weighing sign(dtheta2) as an energy-pumping rule, since torque there always adds energy, but I'm doubtful that more energy means the tip actually climbs high enough. I'm also considering whether sign(dtheta1) would do better, because torque on joint 2 pushes back on link 1, so I'll need to work through how that reaction plays out.

I'm working through the Acrobot dynamics for the torque on joint 2: the inertia coupling means that a unit torque changes the second joint's acceleration directly, and I'm computing the denominator at the hanging position to see how strongly it does so.

I'm checking the torque on link 1 and the resulting angular acceleration, which suggests link 2's absolute angle picks up speed. I'm also recalling a policy that switches torque based on the sign of obs[5], though I can't pin down its exact form yet.

I'm comparing sign(dtheta1) with sign(dtheta2) as pumping rules: for the in-phase mode, dtheta2 and dtheta1 share a sign, so torque keyed to dtheta1 should add energy there while damping the out-of-phase mode, which is the part I'm checking now.

Weighing sign(dtheta1) against sign(dtheta2): the torque on link 2 does work proportional to dtheta2, so pumping keyed to dtheta2 injects energy fastest but may favor the wrong mode, while dtheta1 might pump the in-phase mode that actually raises the tip. Gravity is about 14.7 against a unit torque, so the energy build-up will be gradual over several swings. I'm also considering whether a combination of the two signals could beat either alone.

I'm weighing whether a combined signal like sign(dtheta1 + k*dtheta2) might beat either alone. Since torque on joint 2 injects energy at a rate of tau times dtheta2, sign(dtheta2) grows energy fastest, but I'm not sure that energy lands in the mode that raises the tip. I'm starting to check whether the energy level alone is enough to guarantee reaching the goal height.

I'm working out how much energy the pendulum needs to gain to reach the goal: the tip has to rise to link 1 horizontal with link 2 straight up, which is about 24.5 J above where the system currently sits, and I'm estimating how fast torque on the second joint can supply that given typical angular speeds.

I'm checking whether a spinning link 2 could still raise the tip: if it whips around while link 1 hangs near the bottom, the tip only clears the goal height once link 1 swings above horizontal.

I'm weighing a rule where torque follows the sign of dtheta1, which is a commonly cited heuristic for Acrobot, though I can't recall the exact formulation with confidence. I'm wondering whether adding a pumping term would help, and I'm going to check how the rule holds up as the swing amplitude grows, since the phase relationships shift.

I'm now considering a PD-style rule where link 2's desired angle follows link 1's velocity, per Spong's partial feedback linearization, with the torque clipped to a bang-bang sign.

I'm now working out the linearized energy injection for sign(dθ1) against sign(dθ2) in the in-phase mode, using the mass matrix at θ2=0 with its entries d1 and d2 and the joint-2 inertia.

I'm writing out the eigenvalue condition for the linearized system, setting the determinant of the stiffness matrix minus ω² times the mass matrix to zero.

I'm solving the characteristic equation for the linearized Acrobot modes, which gives two eigenvalues, w1 ≈ 3.55 and w2 ≈ 7.93. That puts the first mode's frequency at about 1.88 rad/s, roughly a 3.3-second period.

I'm checking the linearized modes: the first has link 2 bending strongly in phase with link 1, while the second is out of phase, so the two modes respond to torque in quite different ways.

I'm checking which mode each sign rule pumps: sign(dθ2) feeds energy into both modes, while sign(dθ1) mainly pumps the in-phase mode and damps the other. Since the in-phase mode is the one that lifts the tip, I'm leaning toward sign(dθ1) and considering how the rate of energy gain per swing cycle would compare.

I'm settling on sign(dθ1) for mode 1, since it pumps that mode the same way sign(dθ2) would, and random small initial velocities will set the sign right away. I'm now estimating the energy gained per cycle in that mode, from the integral of |dθ2| over one period, and looking at the energy expression built from the mass matrix and the mode shape.

Working out the energy gain per swing for the in-phase mode: a amplitude near 0.76 rad would be needed, which is past where the linear estimate holds, so I'm expecting roughly 100 steps. That's a rougher number than the best RL results around -62, which suggests either a smarter strategy or sign(dθ2) pumping faster once the motion is nonlinear. I'm starting to wonder whether sign(dθ2) could do better here.

I'm weighing whether the energy-pumping rule should combine both signals, and I'm still undecided on whether the out-of-phase mode wastes effort. I'm keeping in mind that reaching the tip height depends on the right configuration, not just total energy.

I'm weighing whether sign(dθ2) is the greedy choice for energy, and whether reaching the goal height depends on more than total energy. I'm also checking what happens at the start, when link 2 is pushed relative to link 1 from near rest.

Working out the energy gain for the second mode now: the stored energy scales as the square of the joint-2 amplitude, and each cycle adds about four times that amplitude. I'm comparing that against the first mode to see which one builds energy faster per unit of swing.

I'm checking how fast each mode gains energy under the two sign rules: pumping on dθ2 builds energy in the out-of-phase mode about four times faster, which would reach the 24.5 J target sooner, though that mode lifts the tip less. I'm about to see whether the in-phase mode alone would take roughly twice as long.

I'm checking whether a pure mode-2 motion could ever lift the tip above the goal, since the absolute angle of link 2 stays small relative to link 1. I suspect the nonlinear, chaotic motion at that energy could shuffle energy between the modes, and I'm wondering whether the best trained policies exploit something along those lines. I'm also considering a mixed rule, pumping on dθ2 early for speed and switching later to reach the height.

I'm settling on sign(dθ1) as the classic choice and recalling that a rule like "push in the direction of link 1's angular velocity" gets Acrobot to around -80 to -100, though I'm not certain of the exact form. I'd rather trust that memory than run a full RK4 simulation by hand.

I'm checking the torque sign convention, and whether torque in the direction of dθ1 actually adds energy. In mode 1, dθ2 moves with dθ1, so the power stays positive. I'm also wondering whether a combined rule like sign(dθ1 + k·dθ2) would hold up once the motion gets large.

I'm sticking with sign(dθ1) for now, since adding a small dθ2 term doesn't clearly help, and the mode-1 pumping keeps things clean. I'm still checking whether the bang-bang torque itself reshapes the motion, which the linear estimate ignores.

I'm weighing whether torque proportional to the sign of dθ1 is enough, since the forced response could drive the linear system at the first mode's frequency. The linear estimate suggests around 100 steps, and I'd rather keep the simple rule than refine it without a simulation.

I'm working out the modal forcing: the torque projects onto the first mode with a coefficient near 0.116, and each square-wave cycle adds roughly 0.131 rad to that mode's amplitude. Over one period that bumps the first joint's amplitude by about 0.13 rad and the second joint's by about 0.37 rad, though the linear estimate starts to break down at this size.

Checking the tip height at roughly 0.8 rad of link-one amplitude, I get only about 0.29, which is still short, though the nonlinear dynamics could change that. The linear estimate puts the needed energy near 24.5, and I'm weighing whether sign(dθ2) would reach it in fewer steps, maybe around 50 to 60.

I'm weighing whether sign(dθ2) is greedy enough to matter, since pumping mode 2 could convert energy into tip height quickly. The worry is whether that energy actually gets routed into raising link 1 to horizontal, or whether it just sloshes around without reaching the goal.

I'm wondering whether a spinning second link would just keep accelerating under constant torque, so the pure energy-pumping rule might get stuck with link 2 going around in circles instead of raising the tip.

I'm checking whether a sign(dθ2)-style rule would hold up, since torque in the direction of dθ2 always injects positive power, while sign(dθ1) fights the motion whenever the two joints move in opposite directions. I'm also wondering whether a hybrid rule with a small dθ2 term would give the best of both.

I'm considering whether to pump on the second joint's velocity instead, since that gives the fastest energy gain, though I'm not sure it routes energy into raising the tip. I can't recall the exact heuristic from memory, and running an RK4 simulation by hand over a hundred steps isn't practical, so I'm thinking through the first swing qualitatively. I'm also weighing a gymnast-style strategy as an alternative.

I'm thinking about whether the RL-style results, around -60 to -70 with about 3-4 swings, suggest the bang-bang rule could be tuned in its timing rather than just its sign. I'm also considering whether a partial-feedback-linearization idea, driving the second joint toward a function of the first joint's velocity, might be a better structured approach than a simple sign rule.

I'm settling on the rule sign(dθ1 + c·dθ2) with c near zero, since a small dθ2 term only flips the sign when the two velocities are in opposite directions and leaves mode 1 pumping intact. I'm going with plain sign(dθ1) for now.

I'm wondering whether adding a small dθ2 term changes anything, since it doesn't flip the sign in mode 1. Then I'm checking the tip height as a function of the two joint angles, and noting that for a fixed energy the tip is highest when link 2 is pointed in the same direction as link 1.

I'm checking the tip height for a few candidate angles and finding that, with link 1 at 0.6 rad, the tip sits below the bar even though link 2 swings up first. Raising the tip above one unit looks impossible at that configuration, since the cosine terms can't reach the required value.

I'm settling on the need for link 1 to get near horizontal, which means the tip only crosses with about 24.5 J of energy. That makes the question how fast each sign rule builds energy: torque on dθ2 adds power fastest, while sign(dθ1) would take roughly 96 steps if the growth stayed linear.

I'm weighing whether a compromise rule like sign(dθ1 + c·dθ2) makes sense, since pumping on dθ2 would add energy fastest but I can't simulate enough to check how it routes that energy into raising the tip.

I'm checking whether a larger coefficient on the second joint's velocity flips the sign, which it does once c passes about 0.56, so the rule starts pumping both modes. In the pure modes that combined sign works out to the sign of dθ2, which is the same as the sign of the second link's absolute angular velocity.

I'm weighing whether to pump only the first joint's velocity or the second joint's, and I half-recall a simple heuristic keyed to the second joint's velocity sign, though I can't pin it down. The open question is what goes wrong if I pump on the second joint, namely whether link 2 ends up spinning without lifting the tip.

I'm considering whether link 2 spinning fully relative to link 1 would just keep absorbing the energy as its own rotation, leaving link 1 unable to swing up and lift the tip past the bar.

I'm noting that a link 2 spin would keep absorbing energy until the velocity clamp kicks in, which could leave the episode timing out at -500, so that's a real risk. Pumping on sign(dθ1) avoids the continuous spin, since the torque just follows link 1's swing and alternately brakes it. I'm guessing that gets around -90 to -110, and I'm wondering whether a position term like Spong's feedback form would speed up the later phases.

I'm weighing a small dθ2 term, c = 0.3, because near the turning points of θ1, where the two velocities point in opposite directions, the pure sign(dθ1) rule would drain energy, while adding the dθ2 term flips the torque to inject it. I'm checking that c = 0.3 still leaves mode 2 damped, since 1 − 0.3·1.79 is still positive.

I'm settling on a compromise rule, dθ1 + 0.25·dθ2, since the small second-joint term helps near the turning points without letting link 2 spin away.

I'm rechecking the mode-1 in-phase claim against the full "book" dynamics, where the linearization drops a second-order centripetal term and the gravity terms come from the cosine of the summed angles. I'm writing out the two coupled equations of motion to confirm the structure before relying on it.

I'm checking the linearized equations of motion, where the first joint's torque term enters the second joint's equation with a positive sign, so the torque directly sets the energy rate.

I'm checking the linearized torque path to confirm the sign: a positive torque pushes the second joint's acceleration, so it feeds the energy directly. Then I'm writing the policy as a small two-layer network, with a steep first-layer unit on the combined velocity that drives the output so action 2 wins for a positive value and action 0 for a negative one.

I'm checking that a small positive value makes the output pick action 2 and a negative one picks action 0, so action 1 never gets chosen. I'm also thinking about scaling the first-layer weights up so the sign comes through clearly for small velocities, and I'm settling on a concrete set of weights: a large weight on dθ1, a smaller one on dθ2, and a sign-preserving second layer.

I'm settling the timing concern at large amplitudes and deciding not to chase it further, so I'll write up the policy now.

