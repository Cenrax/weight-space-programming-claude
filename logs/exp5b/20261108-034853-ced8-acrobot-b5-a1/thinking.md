I'm working out a hand-designed controller for Acrobot, since I can't run a simulation. A simple candidate is to apply torque in the direction of the first link's angular velocity, which should pump energy into the system.

I'm recalling a few simple energy-pumping heuristics for Acrobot: one is to apply torque based on the sign of the second joint's velocity, which by itself might just spin the lower link. I think the well-known version is to choose the action from the sign of the first link's angular velocity, which should give roughly -80 to -100 on average. I'll implement that with saturated tanh units and then test a few variants, including ones that mix in the second joint's velocity, through rollouts.



---

The first attempt failed, so I'm wondering whether the sign on the joint 2 torque is wrong, since a positive torque there should push link 1 the other way. I'm considering reversing it to pump link 1, though I'm not sure that heuristic actually holds. Next I'll go check the traces.



---

I'm checking the sign convention on the second joint's torque, since it reacts back onto the first link. I half-remember a heuristic from a blog about choosing the action from the sign of the first link's angular velocity, but I'm not sure of it, so I'll derive it from the equation of motion instead. That relation gives the first link's angular acceleration as the negative of the second joint's contribution plus the gravity term, scaled by the link lengths.

I'm weighing whether the torque should be opposite to the sign of dtheta1 or match it, since a torque on joint 2 reacts back onto link 1 in the opposite direction. The energy input is power, tau times dtheta2, so the sign choice depends on how the two angular velocities correlate, and I'm not certain of the common heuristic, so I'll test the reversed sign and check the rollouts.

The data shows the sign choice made link 2 spin up while link 1 barely moved. I'm going to try the reversed sign, tau = -dtheta1, before considering combinations with dtheta2.



---

I'm at about -79.5 and figuring the best policies average around -60 to -70, so there's room to improve. I'm considering a torque rule that also uses the second joint's velocity, possibly a sign of dtheta2 minus a multiple of dtheta1, and I'm weighing whether to pick a combination with three rollouts left, maybe adding sine-based phase-lead terms.

I'm weighing a small dtheta2 term, something like -dtheta1 + 0.3*dtheta2, since the two joint velocities are roughly anti-correlated in the trace, so their signs mostly agree. I'm leaning toward c=0.25 with the rollouts I have left, though a single run of n=32 per variant would cost one rollout each.



---

The first try made things worse, so I'm considering flipping the sign on the theta1 term and adding a small negative weight on dtheta2, using a phase-lag mixture, with the test being [-1, -0.1]. If that also fails, I'll go back to pure -dtheta1, which scored -79.5, and I could alternatively add a theta1 sine term, though I only have two rollouts left.

I'm also weighing a dead-zone variant on action 1, but I'll keep it simple and stick with the plan.



---

I've settled on pure -dtheta1 as the best option. Next I'll revert the other changes, finish up, and save and confirm the weights, possibly checking the last rollout to make sure they were saved.



---

The run-to-run spread looks small, with all three landing around -80 to -90, so I'll keep this result as it is.

