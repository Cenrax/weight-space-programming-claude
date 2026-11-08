I'm checking the network's input size of 8 against 4 outputs, which points to LunarLander. The weights look lopsided, though: input 1 is very large and only outputs 0 and 2 are used much. I'm also comparing other environments with 8 inputs and 4 actions to see if anything else fits.

I'm now weighing a padded-environment idea, and the strongest fit is MountainCar: two real inputs (position and velocity), three actions, with the extra inputs and the fourth action acting as padding. Input 1 being velocity explains its large weights, and the unused action 1 and 3 fit too, whereas LunarLander would have used the engine actions and a y-position input with a range that doesn't match.

I'm checking the sign of the velocity weights: a positive velocity should push toward action 2, but the hidden-layer signs don't obviously agree, so I'll need to trace it through the second layer. That's harder to do by hand, so I'll go with MountainCar-v0 for now.

