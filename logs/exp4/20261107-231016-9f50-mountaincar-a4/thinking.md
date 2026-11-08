I'm settling on a simple energy-pumping policy for MountainCar: push right when velocity is positive and left otherwise. I still need to decide what happens at exactly zero velocity at the start, and I'm checking whether this policy reliably reaches the goal within the step limit.

I'm now sketching how to wire the weights so the output depends only on velocity, using a tanh of velocity through the first hidden unit that passes down to the action logits.

I'm working out the weight layout for the first hidden unit so it tracks velocity through a steep tanh, then carrying that signal down to the action logits. The remaining question is how the tie at zero velocity gets resolved, which depends on the other weight rows I still need to zero out.

I'm settling the bias for the output layer so that the tie at zero velocity breaks toward left, with the left logit set to 0 and the idle logit at -1. A tiny positive velocity still drives the right logit above the others, so that case stays covered. What remains is clearing the other rows of the output weights, which means 63 row replacements unless I scale row 0 up instead.

Next I'm zeroing out the other rows of the weight matrices so only the velocity path matters, which means 63 more row edits. Then I want to check that this velocity-sign policy reliably reaches the goal within 200 steps from starts around -0.5.

I'm checking the starts near the bottom of the valley, around x = -0.5 to -0.6, to see whether the first step is pushed left by the tie-break and the car keeps moving that way.

I'm now going over why the output layer favors action 2 almost regardless of state: the weight rows are nearly identical in pattern, so the logit gap comes down to one scalar that stays negative, and that means pushing right only from the bottom can't climb out of the valley.

I'm going to verify the specifics of the diagnosis and then make the edits to the weights.

