I have a 10-dimensional input and 5 actions, so I'm trying to pin down which environment that matches. I'm checking whether Reacher fits, but I'm not sure whether its observation vector is 10 or 11 dimensions depending on the version.

I'm ruling out Pusher and highway-env, since their observation sizes don't fit. Checking the MPE environments with 5 discrete actions, I land on simple_adversary: its agent observation works out to exactly 10 dimensions, which matches the input.

I'm noticing the output weights for actions 0 and 2 are the only ones doing much, which seems odd for a five-action space. It could be a poorly trained network, but the structure looks deliberate rather than random, so I'm leaving that open.

I'm settling on simple_adversary, though I'm only moderately confident. The other candidate, simple_spread, doesn't fit because its observations are 18-dimensional.

