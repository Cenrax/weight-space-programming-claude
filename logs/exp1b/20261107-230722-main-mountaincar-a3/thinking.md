I'm checking the input size, which is 8, and the four actions, which points to LunarLander. The weights are strongly concentrated on input 1 and on outputs 0 and 2, so that doesn't match what I'd expect from LunarLander's state, where y-position isn't usually the dominant input.

I'm weighing a few other 8-input, 4-action environments, but nothing fits cleanly yet. The odd part is input 1 carrying most of the weight: LunarLander's y-position doesn't usually dominate, so I'm wondering whether the input is just poorly scaled, or whether this is a degenerate policy that leans on height while mostly using the main engine (action 2) and noop (action 0).

I'm now weighing MountainCar as a candidate: its velocity input is tiny, which would explain the large weight on input 1, and a policy that pushes left or right while ignoring noop fits. The unused fourth output could just be padding.

I'm settling on the padded-MountainCar reading, since a policy that leans on one small-scale input and only pushes left or right fits that far better than LunarLander's usual dynamics.

Tanh saturating on a positive-only y would give essentially a sign function, which means nothing for LunarLander, since height never goes negative in flight. Velocity crosses zero in MountainCar, so a sign-of-velocity policy is the classic fit, and I'm going with MountainCar at moderate confidence.

