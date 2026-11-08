I'm planning a simple energy-pumping policy: push right when velocity is non-negative, left otherwise. I recall it reaches the goal in roughly 100 to 120 steps from the starting positions in that range, and I'm checking how it handles starts near the bottom with zero velocity.

I'm setting the first hidden unit to act as a velocity detector by giving it a large weight on velocity and zeroing its other input, so it responds sharply to small velocity values. I still need to clear the other connections so this unit's signal reaches the output cleanly, especially near zero velocity.

I'm setting the velocity-sign rule through the weights: a sharp detector unit, a single output path, and a right-or-left choice that goes to action 2 when velocity is positive and action 0 otherwise. I'm also checking that starting at exactly zero velocity near the bottom, the left push is still a reasonable first move.

I'm checking the edge case where the car sits at the left wall with zero velocity, since a left push there would have to be handled correctly.

I'm now checking that the velocity-sign policy holds up, since at v=0 the left push still leads the car correctly. Then I'm laying out the weight edits: zeroing the first-layer and second-layer entries outside the velocity path, and writing the third-layer rows so only one output carries the sign. That means many individual row and column edits, which is tedious but straightforward.

I'm noting that the network's output barely responds to velocity, so it effectively pushes right constantly and can't overcome gravity. Now I'm setting the single hidden path so the sign of velocity alone decides the action, starting with the weight on the second-layer unit.

Still zeroing the other entries in that column so the signal is clean.

