I'm working out a policy for the task. The plan is to push in the direction of the velocity, which should build up enough energy to reach the goal, and I need to decide what to do when the velocity starts at exactly zero.

I'm settling on a simple approach: boost the velocity's influence through the network's layers so the output tracks its sign. I'm checking whether the velocity-sign policy reaches the goal in roughly 120 steps, and I'm working out how to scale the first hidden unit's weight so the effect carries forward through the second layer.

I'm zeroing out the rest of column 0 in the second-layer weights, then clearing its bias. For the output, I'm planning to make the row-0 weights of W3 large, so the saturated second-layer unit dominates the logits.

I'm zeroing the other rows of W3 and setting its bias to zero, so that at exactly zero velocity all the logits tie. I'm also weighing whether a small positive bias on the right action would be cleaner, since the saturated second-layer unit should dominate the output anyway.

I'm checking the edge case where velocity is exactly zero, which can also happen when the car hits the left wall. In that case the next step's gravity term pushes it back right, so the policy still works out. I'm also noting that float32 precision matters for the observation at the start.

I'm checking whether starting left at around -0.4 with zero velocity still works, since gravity pulls the car left there too. I'm settling on left-first via the tied logits rather than adding a bias.

I'm checking the edge cases around exactly zero velocity and tanh saturation, and deciding to set the output bias so the tie resolves toward left. I also see that the current network keeps favoring the right action in every state, so the car just oscillates around the bottom.

Setting the first hidden unit's weight to a large value and zeroing the rest of its column in the second layer, so the velocity signal dominates the output. Then I'm fixing the output weights and bias so the logits tie at zero velocity.

I'm settling the tie at zero velocity by giving the left action a tiny bias, so the car starts by pushing left.

