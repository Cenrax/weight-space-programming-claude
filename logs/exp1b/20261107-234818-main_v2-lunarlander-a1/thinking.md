I'm checking a 10-dimensional input with 5 discrete actions against the common RL benchmarks. Highway-env has the right action count but its observation is a 5x5 kinematics grid, so that doesn't fit. Among the classic Gym environments, none of the ones I'm comparing match both numbers.

I'm running through the PettingZoo MPE environments next, checking each one's observation size and action count against the 10 and 5 I need. Most don't match, though simple_adversary's good agent looks like it might have a 10-dimensional observation.

I'm checking simple_adversary's good agent more closely, since its observation works out to 10 values: two for its goal-relative position, four for the landmarks, and four for the other agents. The adversary comes to 8 by the same count, and simple_speaker_listener's listener has 11, so neither of those fits.

I'm checking the correlations between actions 0 and 4, which look alike, against the anticorrelated pairs. I'm also weighing whether the noop action is rarely used, and considering highway-env again, though its 25-dimensional observation rules it out.

I'm still weighing whether a 10-input, 5-action setup is a standard benchmark, and I'm leaning toward MPE's simple_adversary good agent, though it could be a single-agent environment instead. I'm also noting that the weights on inputs 4 and 5 are larger, which might reflect landmark positions, though I'm not very confident in that.

