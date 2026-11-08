I'm checking the weight ranges against the expected initialization bounds. W1 looks uniform with a spread of roughly ±88, which is far larger than the 1/sqrt(128) ≈ 0.088 bound PyTorch's default Linear init would give. W2 and W3 are in the same situation, with values up to 125 against a bound of 0.125.

I'm checking the bias values next: b1 sits within about ±88, and b2 and b3 within ±125, which again matches the default uniform bounds for untrained layers. I'm weighing whether small-learning-rate training could have produced this same spread, and the fact that every parameter stays exactly inside its init bound points toward these weights never having been trained.

