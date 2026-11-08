I'm working out how to find induction heads from the weights alone, since the positional embeddings are gone and previous-token detection isn't straightforward. I'm planning to use the layer norm weights, QK and OV circuits, and K-composition scores between adjacent layers as the signals to check.

I'm wondering whether the vocabulary size could be inferred from the rank structure of the token embedding directions, which might show up through layer 0's MLP and attention weights despite the embeddings being removed. Low-rank structure from weight decay is a possibility, though not guaranteed. I'll start by looking at the singular values of the block 0 weights, the layer norm weights, and the biases.

I'm thinking the vocabulary size might show up as a cliff in the singular values of the block 0 q/k/v weights, since directions outside the embedding and positional span would only have decayed initialization left in them. Positional embeddings are usually low-rank, so that cliff might be dominated by the token side.



---

The smallest singular value is exactly zero, which points to one null direction. My first guess is the all-ones direction, since layer norm centers its output, though that only gives an exact zero if the mean is actually removed. Looking closer, the input to the query projection is gamma times the normalized vector plus beta, and the normalized vector is orthogonal to ones, so that's where I need to check.

I'm checking whether the null direction actually comes from the layer norm gain, since the gamma-weighted normalized vector stays orthogonal to the 1/gamma direction. Next I want to see how quickly the singular values of layer 0's input weights drop off, both per matrix and stacked, with gamma folded in.



---

I'm looking at the layer 0 MLP fc spectrum, where the values drop around indices 112 to 128 before leveling off near 0.6. I want a finer look at that range to locate a knee, and I'm keeping in mind that the embedding subspace is vocabulary plus positional dimensions, with the positional part possibly small, around 10 to 30.



---

I'm estimating the layer 0 fc knee at around 120 to 130, but that could come from the embedding subspace or from the attention output, which adds dimensions from six heads, so it's not conclusive. The plateau after about 140, with values near 0.5 to 0.6, looks like random init directions shrinking, which points to an effective used dimension near 130.

I'm weighing whether the embedding vocabulary is roughly 65 to 100 dimensions plus positional ones, and noting that this alone can't tell languages apart. I'm considering reconstructing the embeddings from the last layer's unembedding directions, though without them it's hard to say which character maps to which. Attention patterns, like indentation-aligned heads in code, might reveal the kind of text, but without position embeddings that's uncertain.

I'm thinking the vocabulary size could be read off the layer 3 MLP projection's spectrum, since the unembedding has one row per token and so the output-relevant dimensions should top out near vocab size. I'll check where that spectrum has a knee.



---

I'm checking the layer 3 projection, where the singular values hold steady at around 10 down to 3.9 for the first 28 indices before dropping off. A knee near 28 to 30 would suggest a small effective output dimension, perhaps a vocabulary of roughly 27 to 40 characters. I'll print the full detail for layer 3 and also its attention output.



---

The decline looks gradual, with no sharp knee, falling from about 10 to around 4 over the first 28 or so. I can't pin down the vocabulary size from that alone. A more principled angle might be the layer 3 MLP output directions, which should roughly line up with the unembedding vectors that promote specific characters. Taking the top singular directions there could reveal structure, though it still doesn't give me characters directly. Without embeddings, I'm wondering whether I can recover character identity by inferring bigram statistics instead.

I'm trying to see whether the layer 0 attention and embedding structure could reveal anything, but without the embeddings that route is blocked. I'm now weighing whether tied embeddings would let the layer 3 output directions line up with the input subspace, while still lacking any way to name the tokens. Maybe I can instead look for weight-only signatures, such as vocabulary size from ranks in the layer 0 value weights, that point toward a particular kind of text.

I'm considering whether the rank of the positional embeddings, typically around 20 to 60 significant dimensions, could help narrow things down. I'm also weighing whether layer 0's reading weights, restricted to the directions no layer writes to, might reveal something, though all layers seem to write across the full width.

I'm weighing whether the layer 0 read weights, restricted to directions that no layer writes to, could isolate embedding-only structure. Since every layer seems to write across nearly the full width, that route looks unlikely to yield much. Identifying the exact language is a long shot from weights alone.

I'm checking whether the layer 0 MLP has token-like neurons that could reveal the embeddings, and I'm also looking at the layer 0 ln1 gain values, which sit around 0.63 on average. The unembedding should line up with the top singular directions of the layer 3 proj output, so I'm trying to estimate the number of large logit directions from that spectrum.

I'm checking the top singular values of the layer 3 output for a knee near index 26 to 30, which might hint at roughly 27 to 30 frequently predicted characters, though a full vocabulary like Shakespeare's would blur that. I want to look at the drop in squared singular values more closely, and then move on to identifying the induction heads.

I'm considering whether the positional subspace can be located from the QK matrix of a layer 0 head, since a previous-token head would route positional information into the key side. Without the positional embeddings themselves, that's the only way I can see to check which heads are doing the previous-token work an induction head depends on.

I'm weighing whether the OV circuit of a candidate induction head could show copying: if its residual-space eigenvalues are mostly positive and real, the head would be writing back roughly what it reads, though without the embeddings I can't confirm that the basis carries token identity.

I'm checking whether the OV circuit copies token identity by looking at how the eigenvalues in residual space are distributed, since tied weights would make that meaningful. I'm also thinking about K-composition: an induction head's keys would read what a previous-token head writes. Without positional embeddings, I'm considering whether the positional subspace could be estimated from the layer 0 heads that seem to attend to fixed offsets.

I'm now scoring each head for copying behavior by summing the eigenvalues of its OV matrix, and checking how much of each QK matrix is symmetric. I also want pairwise composition scores across layers, which would show which heads feed information into which.

I'm defining the per-head residual-space matrices for each attention head, the OV and QK products, so that I can measure how heads compose across layers through their key, query, and value paths.



---

I'm noticing that layers 1 through 3 show strongly negative OV traces in residual space, which could mean the residual basis shifts there or that the heads are writing the negative of what they read. I'm wondering whether this comes from writing into the residual from layer-normed input, or whether the LN mean or gamma direction is involved, though I doubt that's the main cause.

I'm checking head 2 in layer 0: its OV copies with a score of 0.9, and its QK is positive definite, so it prefers keys that resemble the query, probably nearby or identical positions. Head 3 has a very small QK norm, so its attention is close to uniform, but the query and key biases could still create a preference, so I want to fold them in and check the bias term per head.



---

Head 0.3 looks like a fixed-pattern head: its QK norm is tiny, so attention comes mostly from the bias term, and its OV output is large. Layer 1 keys read its output heavily.

I'm checking whether head 0.3 might also work as a uniform-average head, and whether a key-only bias term could favor recent positions. The plan is to compute each head's bias-driven key preference vector and compare its norm against the typical QK scale, using rough estimates for the residual norms to judge what's aligned versus random.



---

I'm checking head 0.3's weights, which look weak on the query-key side, with its key-only term around 0.6. It might be a roughly uniform averaging head rather than a previous-token head, since a true previous-token head would need a much stronger position-to-position score.

I'm weighing whether the key-only term could give head 0.3 a recency bias, but a term that grows with position would just favor the current token, not the previous one. So the head might not be a previous-token head at all, and I'd need to check whether the big layer-0 QK norms on head 0 and the positive-definite QK on head 0.2 point to something else.

I'm trying to tell which layer-0 head is the previous-token head. Head 0.3 is the only one with strong K-composition, so it's the likely candidate, even though its QK norm is small. A large positional norm aligned with the top singular vectors could still make its logits big.

I'm thinking about whether a positional head could still have large logits despite a small QK norm, since a strong alignment with the positional subspace could push the scores high. Rather than assume head 0.3 is uniform, I want to look at the QK spectrum and how its key and query subspaces overlap with those of the other layer-0 heads.

I'm now checking the singular values of head 0.3's QK matrix and how much its query-read and key-read subspaces overlap.



---

I'm looking at head 0.0, where the largest singular values come in pairs that look like 2D rotation planes. That pattern would fit a positional head, maybe one that attends to the previous token or a fixed offset. Before settling on that, I want to check the eigenvalues.

I'm now checking whether the query and key subspaces line up for head 0.0 and whether its eigenvalues are complex, which would point to a rotation by a small angle. Head 0.3 has one dominant singular value and a rank-1 structure, which could mean it's keying on a single feature like a delimiter token. I'll compare the left and right singular subspaces for the top eight components of 0.0 to test that.



---

I'm seeing that query and key draw on the same subspace, and the head acts roughly like a rotation of 57 to 78 degrees, which fits a positional relative attention pattern. If the offset has to match the rotation angle at each frequency, then the offset is pinned down by those angles.

I'm checking whether head 0.3 works differently: its key-only term and rank-1 query-key part might line up in a way that gives a previous-token pattern.

I'm checking whether b and u sit in the same positional subspace that head 0.0 uses (the top-8 singular subspace B), since a relative-position signal there could make 0.3 attend to the previous token rather than itself.



---

I'm working out what head 0.3 attends to: its key score seems to depend on a directional component of each token's representation, scaled by a query-dependent term. The key feature along b doesn't sit in the positional subspace, so I'm guessing it encodes something about the token's content, like whether it's whitespace.

I'm checking whether other layer-0 heads share 0.0's rotation-like structure, comparing each head's query and key read subspaces against 0.0's positional subspace. The idea is to see whether head 0.3 is reading positional information too, or something else like token identity.



---

Only head 0.0 relies on that positional subspace. The other layer-0 heads, 0.1, 0.2, 0.4 and 0.5, show heavy query-key overlap, so they seem to match tokens to similar or identical ones instead.

Checking whether the dominant singular directions of layers 1-3 QK are shared across heads, possibly an attention-sink or massive-activation direction. Head 1.4's key-only term is large as well. For induction, layer 1 heads compose through keys with 0.3, but their residual-space OV copy scores all come out negative, which doesn't fit a simple copying story, so I suspect the token-space path through the unembedding is what matters here.

I'm trying to see whether 0.3 could be a previous-token head by checking if its positional read direction matches what other layer-0 heads use. If its key term weights certain tokens heavily, like the space token, attention may concentrate on all of those instead.

Looking at whether 0.3 could be a previous-token head: its key term reads one direction, so it can't pick out the position just before each query across all positions, which rules that out. I'm turning to its value norm of 3.6 and the K-composition score that remains high with it.

I'm checking whether 0.3's value output is low-rank and lines up with a shared "massive" direction that keys read, the way an attention-sink would. I also want to see if its output direction matches the dominant QK direction of layer-1 heads, like 1.4's 15.2 singular value.



---

I'm checking whether the query side of the top QK singular direction lines up with that shared key feature.

I'm now checking whether the query-independent bias term carries the signal: the key-only direction u = Wk^T bq should align with 0.3's output direction, and for head 1.4 its norm is large. I'm also weighing whether 0.3 is a positional head, since uniform attention would produce a running mean that shrinks with position. Even so, this K-composition doesn't count as induction evidence, so I need a test where the query reads token identity and the key reads previous-token information.

Next I'm checking whether the shared direction can be cleanly separated from the rest by projecting it out of the QK matrices and recomputing K-composition. The earlier layer-1 to layer-2 scores were all low, so the dominant shared direction may be masking the real signal.

I'm checking whether 0.0 is the only real prev-token head, and whether a positional subspace is being read anywhere in later layers. Since later heads don't seem to read positional directions directly, I'll consider projecting out the dominant shared direction and then trace where 0.0's output goes, through later QK and the MLPs.



---

Still weighing whether 0.3's K-composition holds once the top direction is removed, and checking whether its key-side effect is really what I took it for. I'm wondering if the positional embedding carries a huge norm and dominates the layer norm instead.

I'm checking whether the top-direction removal changes the picture, and I'm testing whether 0.3 could be a previous-token head. A key-only score can't make attention land on i-1 for every position, so it has to be selecting tokens by content instead, maybe a common character like a space, with its value carrying the position of those tokens.

Now I'm checking whether attention could work as a uniform average over the prefix, so the keys would only be picking up the 1/i position effect. But the K-composition of 0.46 for 1.4 is large enough that I doubt the averaging story, and I'm going back to the induction-head angle to see which earlier heads could be feeding prev-token information into later keys.

I'm checking whether 0.0 is really a previous-token head: its rotation angles are nonzero, so it attends to some offset of at least one, which fits. Its output is small and doesn't seem to feed later attention directly, so it may just supply bigram-style features to MLP0. I'll compare the composition of OV[0,0] with each layer's MLP input against that of OV[0,3].



---

I'm checking that the outputs of heads 0.0, 0.1, and 0.5 are read by MLP0, and that checks out. Next I'm asking whether head 0.3 might act as a previous-token head in some other sense, since the layer-1 heads all K-compose with it. I also want to see whether the key-side direction b sits in the subspace that head 0.2 reads for token matching.

I'm now checking whether head 0.3 could be a previous-token head after all, but a rank-1 query modulation seems to rule that out, since no fixed key scoring can favor position i-1 for every i. I'm also considering that its near-uniform attention might just produce a bag-of-tokens summary rather than induction behavior, so I'm turning to how to locate induction heads more directly.

I'm trying to denoise the composition measure, since the random init remnants may be swamping the signal. My plan is to take the top singular components of each matrix, truncate to rank 8, and then compare the composed product of the two heads.



---

I'm wondering whether head 0.3 is acting as a positional or recency signal, since its output gets read by almost every later head's key along one dominant direction. I'm also considering that the model may not rely much on learned absolute positions, so 0.3 could be building that signal itself, perhaps by scoring attention with a token-dependent weighting.

I'm now checking whether head 3.3 could be an induction head, since its key composes with the outputs of 0.0, 0.1, and 0.5, which carry previous-token information. I'm also looking at whether heads 0.1 and 0.5 act as positional heads and how their query-key structure differs from 0.3's, so I'm planning to compare the rotation angles of the layer-0 heads' top subspaces.



---

I'm sorting out which of these eigen-subspaces are positional and which are token-based. The layer-0 heads are the place to check, starting with whether head 0.0's query reads only the positional part.

I'm going to pull each layer-0 head's main read subspaces and compare how much they overlap, since heads that attend by position should be sharing the positional subspace.



---

I'm weighing whether layers 0.1 and 0.5 are positional heads, since they overlap strongly with each other and partly with 0.0. I'm considering a different signal instead: positional embeddings are added at every position and feed into every layer 0 read, so their subspace might show up there. I'm also thinking about reconstructing a layer 0 head that attends by relative position, where the bilinear form is Toeplitz.

I'm checking whether the layer-1 heads all read a key-side signal from head 0.3, and whether a query-side bias direction could account for a key-only, recency-like effect. I'd need the mean residual to pin that down, but the key bias alignment with that direction is something I can test directly.



---

I'm working out whether the key-only preference terms in layers 1 and 2 are reading from head 0.3's output, which would make 0.3 a source of a query-independent key bias. I'm thinking of it as a token-type marker, but since its values are averaged over whatever it attends to, I'm now wondering what its output at each position actually encodes.

I'm wondering whether 0.3 could be an attention-sink-like head that attends to the first position, which would make its output nearly constant and useless as a key bias except for distinguishing early positions. I can't pin down what its output encodes at each position, though.

I'm looking at whether layer 3's output only matters through the unembedding, since its components outside the row space of W_U get no gradient and should decay. That would mean its outputs are shaped entirely by what the final logits need, and the question of whether vocabulary size shows up in its structure is still open.

I'm checking layer 3's projection spectrum for a cliff, since only the row-space of W_U should get gradient and the rest should sit at the initialization-plus-decay floor. The random-init band for a 384x1536 matrix works out to roughly 0.4 to 1.2, and the values near index 80 are already in that range, so I want to print the full tail and compare its shape against a random matrix of the same size.

I'm going to print the full spectrum of layer 3's proj and o weights to look for a cliff, since only the W_U row-space components should carry gradient.



---

Layer 3's attention output spectrum tails off smoothly toward zero rather than showing a Marchenko-Pastur-style cutoff, which I read as weight decay having shrunk those directions. I'm now trying another route to the unembedding row-space: treating directions that layer 3 writes to (via o and proj) as the logit-relevant subspace, since its dimension should be bounded by vocab size.

I'm now computing the combined write matrix for layer 3 and comparing its effective rank against the earlier layers' write matrices, which feed later layers too. For each layer I'll look at the normalized spectrum of the writes and count how many singular values sit above a noise threshold, expecting roughly vocab-size for layer 3. The P3 spectrum has a shoulder near 25 to 30 and then a long tail, so I'm going to measure participation ratio and the count above 30% of the max for each layer.



---

I'm working out that the strongest layer-3 output directions concentrate in roughly 26 to 30 dimensions, with a weaker shoulder after that. That points to an active output vocabulary of about 27 to 30 frequent characters, plus some rarer ones. Since those strong directions are weighted by frequency, the common characters in English text, letters, space, and punctuation, likely account for most of the signal.

I'm checking the singular-value falloff from about 10 down to 4 across 28 indices, and noting that a DNA-sized alphabet would show only about 4 dimensions while a Chinese-sized one would show thousands. That points to something alphabetic with roughly 30 to 60 characters, though the spectrum alone can't pin down the language. Next I'm considering whether layer-3 MLP neuron output vectors could be clustered, via ICA or a sparse dictionary, into unlabeled character directions.

I'm weighing whether the char directions could be recovered from the layer-3 MLP output vectors, and I'm stuck on a bigram matrix because the direct embed-to-unembed path doesn't exist here. Products of the weight matrices are allowed, but applying the MLP nonlinearity would be a forward pass, so I can only work with linear OV products.

I'm weighing whether a sharper vocabulary-size estimate could come from the layer-3 write rank, though the tail looks too long to read cleanly. I'll try the unembedding row next.

Checking whether the complement directions still carry layer-3 signal: Adam normalizes even tiny gradients, so only the exactly zero-gradient subspace would keep its decayed init, and that subspace should be about 384 minus V dimensions.

I'm noting that the layer-3 output weights have some directions that have decayed almost to zero, while others sit around 0.2 to 0.7, which makes the spectrum look distinctive. Adam's per-coordinate normalization means the complement of the unembedding row space still gets updates, so the subspace split won't be clean. I want to find the subspace where layer 3 writes and then see how layer 0 reads from it.

I'm now checking whether the layer-3 write subspace overlaps with the layer-0 read subspace, along with the positional directions, since heavy overlap would support tied embeddings.



---

I'm ruling out a shortcut via the embeddings, since they're untied and the residual stream rotates. Given how hard this is, I'm considering weight-visible features instead, like heads whose positional rotations encode the frequencies of P, which says nothing about language. I'm also looking at the 0.3 head, a rank-1 QK head with a key-only bias that attends to a specific token type, and wondering whether in Python code it might pick up on newlines or indentation.

Next I'm weighing whether a layer-3 MLP dictionary recovery of the character output directions would be tractable, though it might just be a dead end. Another angle is comparing the effective number of output characters against candidate languages like English, Russian, Shakespeare, or Python code, since the spectrum's strong dimensions could hint at which one the model was trained on.

I'm checking whether the positional rotations might encode a fixed line length, like the 60 or 70 characters in FASTA files or fixed-width tables, since a strong periodic structure would show up as specific frequencies. The leading angles could point to an offset matching that length, though I can't yet tell the language from it.

Checking the positional-frequency periods against an offset-1 head, I'm treating the B-subspace rotation as a per-step generator, so the QK pattern would scale with the offset Δ, though I can't separate the rotation from Δ yet. Then I'm considering head 1.4, whose QK norm is very large with one dominant singular value, to see whether it points to a specific text structure.

I'm now looking at head 1.4 with its large key-only bias, trying to see whether it pairs with 0.3 as a mechanism for locating the start of a word or line. The next step is reading its top singular vectors to see what the query and key sides align with, though I'll likely have to settle for approximate labels for these heads.

I'm going through the remaining heads now: 0.1 and 0.5 look like local positional heads sharing part of their subspace with 0.0, and 0.2 seems to match earlier occurrences of the same token. 0.3 is the broad, content-gated head whose output the later heads use as a key bias, and 0.4 looks mixed. The layer 1 and 2 heads appear to rely on the 0.3 signal, and I want to check whether 3.3 composes its keys with the local heads from layer 0, which would make it an induction-style candidate.

I'm checking whether 3.3's query reads the same current-token information that the layer-0 heads write, which would make the Q-composition weak. The layer-0 composition scores look low, so the next step is to see what its query actually picks up from the residual stream.

I'm now planning to measure how strongly each later head's keys and queries compose with earlier heads' outputs and the MLP projections, using a truncated norm that drops the top output directions of head 0.3. I'll then print compact scores for layers 2 and 3.



---

Some heads look notable at 2.4 K, 3.5 K, and 3.3 K, with the strongest reading at 0.59. Before I trust those values, I want a baseline: random-ish composition of rank-16 matrices should land around 0.2, so anything above roughly 0.3 stands out. To settle whether induction is happening, I still have to identify which heads attend to the previous token and which later heads read from them.

I'm checking whether layer 0 heads like 0.0 carry the positional signal, since later heads don't seem to read that subspace. That would make 0.0, and possibly 0.1 and 0.5, the previous-token candidates, with 3.3 as the likely induction head that reads from them.

Checking whether 3.3 is the induction head: its value-side OV norm is the largest in the model, which fits a strong copier writing straight to the logits. Its key side ignores the positional subspace, so it looks content-driven, which also fits. Next I'm turning to the layer-1 heads 1.4 and 1.5 and whether they could be the previous-token step feeding the 2.x heads.

I'm checking whether 0.0's value path reads the positional subspace or the token subspace, since a prev-token head should move token identity rather than position. I'll also compare each layer-0 head's value reads against the Q/K reads to see which overlap.



---

I'm reading the heads 0.0, 0.1, and 0.5 as positional: their query-key matching runs on one subspace while their values carry token content from a separate one, which points to previous-token-style heads that copy from fixed relative offsets. Their query-key subspaces overlap heavily, especially 0.1 and 0.5, so they're likely doing the same kind of job.

I'm trying to pin down which head is the previous-token one by comparing rotation angles across 0.0, 0.1, and 0.5. If they share positional planes, the angles in each plane should scale with the offset. The plan is to build a joint subspace from their Q/K reads, then check how the other two heads' QK matrices look in the eigenbasis of 0.0's.



---

I'm comparing the eigenplane angles at 0.0, 0.1 and 0.5 against each other, checking whether the 0.1 angles come out close to twice the 0.0 ones. They are roughly, though not exact, so the ratio is around 2.2, and the 0.5 angles give ratios near 1.1, 2.2 and 3.

I'm weighing whether the 0.1 angles are about double the 0.0 ones, and they land close but not exact. I'm lining up the matched eigenvector signs to check that, and it points toward 0.0 attending one token back and 0.1 attending two back, with 0.5 looking mixed.

Now I'm checking whether the query side is also reading the previous tokens, since the Q components for 0.1 and 0.0 are sizable too. If so, this would be a longer-context matching pattern rather than a textbook induction head, with both sides encoding nearby tokens and lining up with a shift.

I'm checking whether the 2-gram context idea holds up for 3.3, and I'm considering the layer 1 heads as candidates that move context into the key side, with 2.4 and 3.5 possibly being induction heads fed by them. Checking the later heads' output against the logit subspace is the next step, though that's harder to pin down.

I'm checking whether the 3.3 induction head's query-key product, applied through the previous-token head's output, behaves like an identity map on token identity, so I can test the copy score of that combined circuit.

I'm checking whether the copy score of each later head's QK circuit composed with the 0.0, 0.1, and 0.5 OV circuits comes out positive, using 0.2 and 0.3 as controls. I'll look at the fraction of positive top eigenvalues for each combination.



---

Head 3.3 scores +0.90 for the copy pattern with 0.0, the previous-token head that moved the earlier token into the key position, which looks like an induction signature. It also has +0.44 with 0.1, two positions back. Heads 3.0 and 3.2 are positive with 0.0 as well, at +0.53 and +0.57.

I'm checking whether head 0.3 behaves like a previous-token head, since its positive scores across layers 1-2 could signal induction. Its key-side bias might instead favor recent positions, which would make it attend mostly to itself and the tokens just before.

I'm now wondering whether head 0.3's key bias tracks position, a recency ramp that would make it attend mostly to itself and the last few tokens. If so, it wouldn't be an induction head. To check, I want to see whether its key-side direction lies in the token subspace or in a separate positional one.

I'm defining T as the top read subspace of the value projections for heads 0.0, 0.1, and 0.5, the ones that move token embeddings. Then I'll measure how much of b falls inside T compared with the positional subspace spanned by those heads' Q/K reads.



---

I've got a variable collision: a loop over `keys` overwrote `a` and `b`, so they're tuples and a string now and need recomputing. Separately, the token subspace spectrum shows a knee around 36–40 dimensions, which I'll use as a rough estimate for the vocabulary size.

I'm reasoning that the value matrices of the positional heads read from the token embeddings, so the count of strong dimensions should approximate the effective vocabulary size, around 36 to 40, with rarer characters possibly contributing only weakly. I'll print the spectrum at finer resolution afterward to pin that down.



---

I'm reading head 0.3 as a content-gated attention: its key direction sits mostly in the token subspace, so it picks out a class of tokens, and the query's positional component scales how strongly it does so, making the output an average of those tokens' values.

Now I'm checking whether the key direction of head 0.3 points at a particular character class, with a leftover idea that a negative offset would instead suppress that class. I'm also looking at where the token-subspace values drop off between indices 33 and 36, which might mark the effective vocabulary size.

I'm weighing whether the strong embedding dimensions, roughly 34 to 36 of them with a tail out to about 45 before it flattens into noise, could reflect a vocabulary of around 27 characters like lowercase text plus space, or something closer to 40 with punctuation and another alphabet.

I'm weighing whether the strong dimensions out to about 35 fit an alphabetic language with mixed case and common punctuation, since that would put the effective vocabulary near 35 to 45 frequent characters. Code would need far more dimensions, and Chinese would fill the whole embedding, so the Latin-alphabet range looks most plausible.

I'm going to pool the layer-0 value reads across all six heads, along with the token-read spectrum, to get a sturdier count of how many dimensions carry token identity. I'll also check the 0.2 head's QK subspace, since it matches tokens against themselves.



---

I'm estimating the input subspace the first-layer value projections actually use: the knee sits around 42 to 46 components, and some of that may be reading positional signal. That puts the token-plus-position usage near 45, with the token part around 35 to 40, which suggests roughly 35 to 45 characters are effectively in use, even though tiny Shakespeare has 65 distinct characters, many of them rare.

I'm weighing whether the uppercase letters and case flags might push the effective character count toward 40, since that would fit the rank estimate. Still, the vocabulary rank alone can't tell me which Latin-script language this is, so I'm leaning toward English at about 0.35 confidence.

I'm checking whether the layer-1 heads move previous-token information too, since 3.3 also shows strong overlap with several of them, and I'm considering whether a copy score on the same-token pattern would tell me more about what these heads are doing.

I'm checking how much of head 3.3's value input comes from the token subspace versus positional information and the layer-0 output, and I want the same breakdown for the other heads so I can label them by what they read.

I'm planning one more pass that breaks down, for each head, how much of its query, key, and value inputs come from the token subspace, the positional subspace, and the layer-0 output, so I can label the heads by what they actually read.



---

I'm sorting out layer 0: heads 0.0, 0.1 and 0.5 take their queries and keys purely from position and their values purely from the token, so they act as positional token movers at fixed offsets. Head 0.2 looks odd, since its query seems to read the token while its key reads position, and I'm questioning whether that's right given its positive-definite QK.

Head 0.2 is still bothering me, and I'm wondering whether its positional and token roles overlap with the J and T subspaces. Heads 0.3 and 0.4 look like a token-class gate and a mix of content and position. For layer 3, head 3.3 writes strongly into the logit-relevant directions, which fits an induction head predicting the copied token. I still need to pin down the offset of head 0.5: it has small magnitude in 0.0's top planes, so it probably uses lower frequencies and acts as a broader local window.

I'm checking whether head 0.0 is really the only previous-token head: 0.1 looks closer to two tokens back by the angle ratio, so it doesn't fit that role. The offset could be (1,2) rather than something else, though (2,4) is possible but less likely. Head 3.3's copying also fits with 0.0 being the main contributor, with 0.1 playing a weaker secondary role.

Layer 1 queries don't read token or position much, so they're probably keyed on features built from layer-0 outputs. Head 1.4 is dominated by a key-only feature from 0.3, which suggests it attends to a specific marker like a delimiter. Head 3.3's copying also scores well against layer-1 heads 1.1 through 1.3, which would fit those heads carrying earlier token content forward.

I'm weighing whether heads 3.0 and 3.2 belong alongside 3.3 as induction candidates, since their copy scores are positive but their key-composition with layer 0 is weak. Head 3.3 stands out clearly on both measures, so it's the one I'll name, with 3.0 and 3.2 noted only as weaker secondary candidates.

I'm checking how strong the QK@OV_0.0 product is relative to head 3.3's overall QK norm, to see how much of its attention logit variance comes from previous-token matching. I'll compute the operator norm of the composed map and compare it against the sum of the positive real eigenvalues.



---

I'm treating head 3.3 as the clear induction head, since its real eigenvalues sum to 0.99 with the previous-token head. Heads 3.0 and 3.2 look weaker, at 0.31 and 0.28, so they may be partial induction at best. I'm wondering whether to mention them as weak candidates, and whether the ground truth comes from an induction score threshold on repeated random sequences.

I'm weighing whether to try a token-subspace check on each head's V spectrum to refine the vocab estimate, since I've already spent 25 calls. I'll stick with only 3.3 as the induction head and leave 3.0 and 3.2 out.



---

I'm checking where the OV ranks drop off for the positional token movers, which lands around indices 28 to 32, so the token-identity signal seems to sit in roughly that range. With a 64-dim head but only about 30 dimensions actually in use, I'm estimating the number of character distinctions worth moving at around 30.

I'm weighing which alphabet would give about 30 distinctions: plain lowercase English plus space and a little punctuation fits best, while Spanish, French, German, or Russian would push the count past 33. Uppercase might share directions, so it doesn't clearly change the picture. I'm putting my confidence around 0.4 and settling on that English reading.

I'm now listing the remaining head roles: a strongly selective marker head at 1.4 and a similar one at 1.5 that feeds 2.4's keys, with higher-level context heads in layer 2. For layer 3, I'm marking 3.3 as the induction head, 3.0 and 3.2 as weaker and fuzzier, and 3.1, 3.4, and 3.5 as the output heads.

