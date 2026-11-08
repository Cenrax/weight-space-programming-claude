I'm checking whether this LunarLander network has actually been trained: 8 inputs, 4 actions. The first-layer weights show a clear pattern, with the angle, angular velocity, and vertical velocity rows carrying the largest magnitudes and the leg-contact rows smaller. The biases are small but non-zero, which makes me wonder whether they were trained or just initialized.

The bias check is what settles it for me: the first-layer biases stay within about ±0.09, while PyTorch's default init would have put them near ±0.35 for eight inputs. That points to a zero-bias orthogonal initialization followed by training. The output weights for the left and right engines look anti-correlated, which fits steering, and the angle row carries the largest weights, so angle looks like the dominant feature.

