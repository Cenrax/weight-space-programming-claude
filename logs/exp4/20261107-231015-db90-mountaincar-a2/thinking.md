I'm weighing a simple policy for MountainCar: push in the direction of the velocity, and figure out what to do when the velocity is zero at the start.

I'm settling on a velocity-sign policy and thinking through how to wire it into the network, which means routing a single hidden unit so it responds to velocity. The main work is zeroing the column of W2 that feeds that unit, since otherwise the other hidden units would leak into it.

I'm setting the first hidden unit to track velocity through a steep tanh, zeroing the rest of column 0 in W2 so only that unit feeds the second layer, and wiring the output weights so that a tie at zero velocity defaults to pushing left.

I'm checking the sign logic at each step: with a small positive velocity the right push wins, with a negative one the left push wins, and I'm raising the noop bias so it never gets picked. At exactly zero velocity the tie goes to action 0, which is fine.

I'm checking that the velocity-sign policy holds from every start, and I'm fairly confident it does at roughly 83 to 120 steps. I'm also working through what happens at the left wall, where the velocity resets and the tie at zero sends the car pushing left again.

I'm setting the first hidden unit to a steep tanh of velocity, zeroing column 0 of W2 except for the one weight that feeds the second layer, and clearing that unit's bias. I also see the current policy stuck pushing right because the output bias favors that action and the hidden signal barely reacts to velocity.

I'm setting the output bias for the noop action to -1 and the other two biases to 0, so that noop never wins and ties at zero velocity still go to pushing left.

