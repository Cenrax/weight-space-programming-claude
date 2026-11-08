I'm trying to identify a widely used environment with 10 observation dimensions and 5 discrete actions. Gymnasium candidates like highway-env and MiniGrid come to mind, and I'm also wondering whether a discretized version of Reacher or Pendulum could fit.

I'm listing the observation and action sizes of classic control tasks and then the PettingZoo MPE multi-agent environments, checking which agent's observation has 10 dimensions while still using 5 discrete actions.

I'm checking whether the action-column correlations in the data line up with the MPE action layout, where left and right should be opposed. Columns 1 and 3 look strongly anti-correlated, while columns 1 and 2 don't clearly oppose each other, so I'm still sorting out which pairs actually mirror each other.

I'm weighing whether the action columns fit the MPE layout or whether some other environment with 10 observation dimensions and 5 actions fits better. Simple_adversary's good-agent observation comes out to 10 dimensions, so I'm leaning toward it, though my confidence is only moderate.

