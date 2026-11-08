I'm trying to work out which game this is by looking at the output layer. Several action indices, the directional ones with and without fire, share a sign pattern that differs from the rest, and I'm starting to compare their values.

Freeway keeps coming up as the likely game: the up-versus-not-up split fits its UP-to-cross structure, and the large weights on RAM address 14, the chicken's y position, support it. The row-106 weights don't fit as neatly, so I'm treating them as uncertain. I'd put my confidence around 0.6.

