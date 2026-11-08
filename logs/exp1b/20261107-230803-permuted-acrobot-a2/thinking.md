I'm looking at the W3 weights for this LunarLander net: actions 0 and 2 are strongly anti-correlated, while 1 and 3 sit near zero. The output biases are all small, which makes me suspect this is an early-training checkpoint rather than a converged policy.

I'm ruling out Acrobot, since it has six inputs and three actions, while eight inputs and four actions points to LunarLander. The leg inputs at positions 6 and 7 are worth checking next. The weight structure is clearly not random: the small output-layer initialization and the opposing pattern on actions 0 and 2 look like training that already pushed those actions apart, consistent with an early PPO policy rather than an untrained one. I'm settling on LunarLander at roughly 0.85 confidence.

