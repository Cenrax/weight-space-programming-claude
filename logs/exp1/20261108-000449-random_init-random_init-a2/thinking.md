Biases all come out zero and the output weights are tiny, which matches standard PPO orthogonal init with zero biases rather than a trained network. So this looks untrained. I'm now checking whether the W1 weight scale fits orthogonal init with gain sqrt(2).

The zero biases are what confirm it's untrained: a fresh orthogonal init leaves them at zero, so every bias being exactly zero rules out any training.

