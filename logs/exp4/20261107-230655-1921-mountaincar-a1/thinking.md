I'm settling on a velocity-based policy, pushing in the direction of the current velocity, since that is a standard approach for this environment. I recall it reaching the goal in around 100 to 120 steps across starting positions in the usual range.

I'm now setting up the implementation: one first-layer unit computes tanh of a large multiple of the velocity, so its sign carries the decision. The remaining work is zeroing out the other contributions to the logits, which means a lot of individual weight edits.

Still weighing whether to zero out the other contributions by editing W2 or by zeroing W3 rows, since both mean many individual edits. Another option is to make the W2 self-weight large enough that it dominates the rest, though that could fail when velocity is near zero.

I'm considering whether a large-weight self-connection in layer 2 could dominate the other contributions, but it risks failing near zero velocity. I'm leaning toward explicitly zeroing the layer-2 column and the layer-3 rows, even though that means around 126 individual weight edits.

I'm setting the second-layer weight for the self-connection to 5 and zeroing its column and bias, then fixing the third-layer row so one logit is strongly negative and the other strongly positive, which should dominate any leftover noise.

I'm weighing whether to zero out the other W2 contributions to this unit's column, since 63 individual edits seem cleaner than relying on a large self-weight. I'm settling on W2[0,0]=10 with the other column entries set to zero. I also want to verify that the velocity-sign policy reliably solves MountainCar from the standard starts, which I believe it does in about 100 to 120 steps.

I'm checking the edge case at v=0, where the hidden unit outputs zero and only the bias and leftover noise decide the action, so I'm considering a small positive bias to nudge the start toward pushing right. I'm also keeping the float32 precision question in mind, though I expect it to be fine. The current setup seems to explain why the original policy always chose action 2, so I'm looking at the weights that caused that.

I'm checking why the original policy always picks the same action: the output weights barely vary with the input, so the logits stay constant and the car just oscillates in a narrow band without gaining energy.

