I'm checking whether the weights look randomly initialized: the W1 bounds come out to about 1/sqrt(8), and the W2 and W3 bounds to 1/sqrt(32), which matches PyTorch's default uniform initialization

The next part is a check that the weights are untrained: the W1, W2 and W3 values all sit within the expected uniform-init bounds, with none exceeding them. Since trained weights usually break those bounds, I'm concluding the network is still untrained.

