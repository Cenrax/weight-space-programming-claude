I'm trying to pin down which widely used RL environment has a 10-dimensional observation space and 5 discrete actions. Reacher doesn't fit because its actions are continuous, so I'm going through other Gymnasium and related environments to find a match.

I'm running through MPE scenarios to see which agent has a 10-dimensional observation and 5 discrete actions, and simple_adversary's good agents fit that. The output columns also look structured, which points toward a trained policy rather than a random one.

I'm checking whether the column pairing in the output fits MPE's left/right and up/down action mapping, and it doesn't quite match. I'm also ruling out highway-env and MiniGrid, since their observation layouts don't give a flat 10-dimensional vector.

I'm working out the observation count for simple_adversary's good agents: goal relative position, four landmark coordinates, and four other-agent coordinates add up to 10, which matches. I'll settle on simple_adversary as the environment, though a gridworld remains a possibility I haven't fully ruled out.

I'm settling on simple_adversary with moderate confidence, though I haven't fully ruled out a gridworld.

