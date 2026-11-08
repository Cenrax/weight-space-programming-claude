I'm trying to figure out what I can learn from the weights alone, without running any forward passes. Without the embedding matrix, identifying the language seems nearly impossible, though I wonder whether indirect structure like the vocabulary size could leak through somewhere, perhaps in the layer-zero layer norm weights.

I'm considering whether the vocabulary size might show up in the weights, since the token embedding subspace is at most that many dimensions wide. Gradients only touch directions the inputs span, so layer 0's query, key, and value matrices may be effectively low-rank, though positional embeddings could widen that span. I'll check the singular values to see.



---

I'm looking at layer 0's input side, checking the column space of the q, k, v, and fc weights. I'll stack them, scale by the ln1 weight, and compute singular values, watching for a sharp drop that would mark the vocab-plus-position dimension.



---

I'm estimating where the knee sits in layer 0's MLP fc values, which seem to fall off sharply around index 85 to 130 before flattening. The useful subspace might be around 120 dimensions, though the drop isn't clean. Using the layer-0 ln1 weights elementwise may not help much, so I'm weighing other approaches.

I'm now checking whether the vocabulary size can be read off the singular values of layer 3's mlp.proj, since the output space should have roughly as many useful directions as there are tokens. Different languages would give very different knees, with DNA being the smallest case, so I'll look at those singular values next.



---

I'm looking at the MLP3 output singular values, which drop off around 25-30 directions before tapering. I'm wondering whether the output space, bounded by the vocabulary size after the final norm, might be a small alphabet of roughly 26-30 symbols, like lowercase letters.

I'm weighing whether the language could be inferred from bigram statistics or from the layer 0 attention patterns, since the embedding matrix isn't available to name the tokens directly. The positional structure might also hint at the domain, such as code with indentation heads or DNA, so I'm considering which signals like output dimensionality and attention patterns are actually extractable.

I'm trying to work out whether the previous-token and induction heads can be picked out from composition scores, since the positional embeddings aren't visible to read off directly. I'd look for layer-0 heads whose output gets read by later heads' keys, and check whether any positional subspace shows up in a head's QK structure.

I'm picking up the question of whether a previous-token head's QK matrix would show a rotation-like antisymmetric part, since learned positional embeddings can carry sinusoidal structure. I want to start by computing per-head statistics: QK bilinear norms, OV norms, and composition scores.

I'm checking whether the lack of a clean drop in the layer 0 singular values is expected, since Adam's per-element updates can push weight outside the input span. The unusually large top singular value in the layer 0 MLP looks like a constant or mean direction. I'm also wondering whether the layer 0 LayerNorm bias might carry some signal.

I'm checking whether the vocabulary size can be pinned down more precisely, since the final logits only need the unembedding's row space. I'm also noting that gradients into the layer 3 projection come from the unembedding directions, filtered through the final norm.

I'm checking whether the layer-3 output directions line up with the unembedding span, and whether the layer-2 output and the attention output weights share those same left-singular directions.

I'm checking the spectrum of layer 3's attention output, which isn't sharp, and I'm thinking about the knee near index 24 where the singular values fall from about 10 to 3.3 gradually. Twenty-six letters plus a space would fit that count of roughly 27 dominant dimensions, so a char-level text dataset like text8 is a candidate.

I'm checking whether the singular-value tail flattens out to a noise floor, since the gradient into this layer should lie mostly in the span of the unembedding directions. If the vocabulary were around 65 characters as in Shakespeare, I'd expect a knee near that count, but the 25 dominant directions might mean rarer characters contribute only weakly.

I'm checking the noise floor on the layer 3 projection: random init at std 0.02 would give singular values roughly between 0.39 and 1.17, and with the scaled init of about 0.007 they'd fall between 0.14 and 0.42. The tail sitting around 0.2 to 0.7 might be mostly noise, which would push the real signal dimension to something like 60 to 100.

I'm checking the layer 0 MLP spectrum against the noise floor from random initialization: the tail sits around 0.3 to 0.9, which matches the expected range for init noise. That suggests the signal dimensions end near index 115 or so, covering both the token embeddings and the positional dimensions.

Now I'm working out the vocabulary size from the layer 0 noise floor: the qkv spectrum reaches the random-init range around index 96, which would leave roughly 95 signal dimensions. Separating the token embedding dimensions from the positional ones is harder, since the layer 0 LayerNorm weights are per-coordinate and don't help with that.

I'm weighing whether the language can be identified at all, since the embedding matrix isn't available to map directions back to characters. Statistical cues like bigram structure or attention patterns to newlines or brackets might hint at code, but the most feasible guess may just be generic English text.

I'm now checking the head analysis: for previous-token heads in layer 0, I can't read positions directly, so I'll look at K-composition with later layers, scoring each earlier head's OV output against the later head's QK matrix, including the layer norm scaling.

I'm checking whether previous-token heads in layer 0 can be spotted from the shape of their QK matrix, since their positional structure might show up as an antisymmetric, rotation-like part. I'd compare the antisymmetric and symmetric norms of each head's bilinear form, expecting token-matching heads to look mostly symmetric.

I'm checking whether induction-style heads can be picked out from how their query-key structure maps the current token onto the previous-token output, since that mapping need not be symmetric.



---

I'm noticing the OV eigenvalues come out negative in layers 1 through 3 when measured in the residual basis, which seems odd since I'd expect induction-style copying to look positive in token space. I'm not sure yet whether a negative residual-basis trace is typical, so I'm checking whether that's expected. Layer 0 head 0.2 is next, where I'm looking at its symmetric QK component and whether that trace is strongly positive.

I'm checking which heads have a bias term that could give them a key-dependent preference, and for head 0.3 I'm working out the vector that carries that bias through the key projection. Head 0.0 stands out because its QK norm is large and its antisymmetry is high, which looks like a rotation-style previous-token head, unlike 0.5 and 0.1 whose values are close to what random matrices would give. Next I want to compute the K-composition scores between heads.

I'm now working out the K-composition score between heads, comparing the normalized Wk of the later head against the OV map of the earlier one to see how it stands against a random baseline.



---

I'm checking whether head 0.3 gets its attention pattern from content, and the QK score is so small that it looks like it's really driven by position and the key-only bias term instead.

I'm working through whether head 0.3's positional component could be large enough to matter, since the content-based QK norm is small but LayerNorm puts vectors at a fixed scale, so the bilinear form isn't obviously negligible. Comparing it against the other heads' scores suggests it's only small relative to them. I'm also weighing whether a head with tiny QK could still act as a previous-token head, which would need a sharp positional preference.

I'm wondering whether head 0.3 is simply averaging over the preceding tokens, which would leave its Wq and Wk near-uniform attention. Its Frobenius norms are well below the initialization scale, so the head may have shrunk during training, yet layer-1 keys still read its output heavily, which is puzzling.

Checking whether head 0.3's attention could depend on position, not just content, since its key-only bias and positional embeddings are the remaining candidates. Prev-token behavior would need query-position dependence, which the key bias alone can't provide. I'm now going to look at the spectrum of its QK matrix to see whether a few large singular directions carry the positional signal.



---

I'm checking head 0.3's query-key structure: it's essentially rank-1 with a singular value around 0.033, so its scores could reach roughly 4.7 if the inputs align with that direction. I'm also weighing a key bias contribution of about 1, and wondering whether that dominant direction encodes position.

Head 0.0's QK matrix shows several paired singular values, which looks like rotation pairs and suggests a positional head, possibly a fixed-offset or prev-token type. I'm going to check whether its query and key subspaces line up by computing principal angles between the top singular subspaces.



---

I'm checking whether the query and key subspaces line up, which they nearly do, with complex eigenvalues pointing to a rotation that targets a fixed relative offset. I'm also computing the phase angles of the head's components, roughly 78°, 67°, and 58°, and starting to think about how a positional embedding for previous-token attention would be built from sinusoids.

I'm reasoning that head 0.0 looks like a positional, previous-token-style head, but its Q/K/V composition with other components sits near baseline, so its effect must run through something else. I'm now asking whether the MLPs of layers 0 through 3 read its output, and I'm also considering head 0.3 as a possibly different kind of head, one whose rank-1 query-key structure might prefer particular tokens rather than positions.

Head 0.3 is the one feeding layer-1 keys, so it may be the functional previous-token head even though its rank-1 query-key structure doesn't obviously encode position. I'm checking whether its key-side direction could carry positional information, and whether 0.3's subspace overlaps with the positional subspace of head 0.0.

Rank-1 query-key scores factor as a(t)·b(s), so with a fixed-sign query the argmax over earlier positions just tracks b(s) and cannot single out t-1. That makes a prev-token pattern impossible for 0.3 unless position is involved, so it looks like a different kind of head.

The 0.3 head looks like a near-uniform averager rather than a precise positional one, since rank-1 scores can't isolate the previous token, so its output is roughly a mean over earlier values. That doesn't obviously explain why layer-1 keys read from it, so I'm wondering whether the K-composition metric is being skewed by norms.

Checking whether 0.3's output sits in a subspace that layer-1 keys read heavily, which might explain the K-composition score better than any positional role. Next I'll look at what head 0.0's OV writes and whether layer-0 or layer-1 MLPs read it.

Next I'm checking which subspace each head's OV output occupies, and which later components read from it. I want to compare each head's OV subspace against Q, K, V and MLP fc inputs, measured against a random baseline, and add the MLP fc reads to what I already did for Q/K/V.



---

I'm checking whether the output of head 0.3 is read mainly by layers 1 and 2, which would match a prev-token pattern feeding induction heads. Then I'm wondering whether a rank-1 attention could really do a relative-offset lookup at all, or whether something like the causal mask plus a steep decay in the query-key term is needed.

I'm now asking whether head 0.3's rank-1 QK structure can express a relative-offset lookup at all, or whether it would need the causal mask plus a decaying term. I'm also checking whether the query bias term dominates, which would turn the head into a key-only preference.

I'm wondering whether head 0.3 is really an averaging head, since a uniform mean of value vectors would give a slowly varying "document-context" signal that shifts across positions and could act as a recency or start-token preference. That would be less canonical than a clean prev-token pattern. I want to check which subspace its OV circuit reads from, specifically whether Wv3's row space overlaps with the 8-dimensional positional subspace of head 0.0.



---

I'm checking whether the 0.0 QK subspace is also used by the Q/K of 0.1 and 0.5, which would make it a positional subspace, since values don't use it. Next I want to pool the top QK directions from those three heads into one estimate of the positional subspace, then look at how QK behaves within it for each layer-0 head, where the rotation angles should reveal the offset. I'm also starting to think about the learned positional embeddings.

I'm now checking whether head 0.0's positional rotation implies a specific offset, but I can't recover the frequencies ω_k from the weights alone, so the offset stays undetermined. Head 0.0 still looks positional with a nonzero offset, which points toward a previous-token or few-back head. I'm wondering whether layer-1 heads, especially 1.4 with its large QK and strong K-component on 0.3, read this positional signal for induction-like behavior.

I'm checking whether 0.3's read on the positional subspace is what makes it matter for layer 1, so I'll measure its overlap with the positional directions from 0.0, 0.1, and 0.5 more carefully. Its K weights seem to lean on position 0's features, which could make it act like a sink whose output is roughly constant. Then I want to see whether the layer-1 heads, especially 1.4, pick up that same positional signal.

I'm moving on from head 0.3, since its overlap with the positional directions is small and its QK is weak, so it looks more like a content head that mostly attends uniformly. I'm now considering whether layer-1 heads, especially 1.4, read signal from 0.3 through their keys, and I'll check that by examining their Q and K usage of the positional subspace and how their key-side inputs depend on 0.3's output.

I'm checking whether layer-1 head 1.4 reads the signal from 0.3 for induction-like behavior, by testing whether the composed map from query residual through its key path and 0.3's OV is positive on the token directions. The trace of that bilinear form for each layer-0 head should show whether the same-token matching pattern shows up.



---

I'm noticing that the QK-through-OV composite for nearly every layer-1 and layer-2 head is strongly positive-definite and large, which is the usual induction pattern, so head 0.3 looks like the previous-token head. But it's odd that all six layer-1 heads and all six layer-2 heads show the same thing, and I'm not sure yet what that means.

I'm still weighing whether head 0.3 could be something other than a previous-token head, since its composite pattern matches the induction signature but its attention looks broad rather than sharp. If 0.3 is a broad averaging head, the layer-1 heads would be matching queries against averaged context, which doesn't fit the odd uniformity across all twelve heads.

Now I'm checking whether position plays any role in head 0.3 by looking at its positional subspace, and it looks small, so the head seems to be a broad averaging head. That still leaves the uniform pattern across the layer-1 heads unexplained.

I'm wondering whether the high evpos≈1 for head 1.4 is just a rank-1 effect, where one dominant direction in the product makes the positive eigenvalue come out almost automatically. If so, the pattern might not mean much, so I want to check the rank of that matrix and the singular values of the OV spectrum for head 0.3.



---

The attention head at 0.3 seems to be writing a low-dimensional signal, concentrated in a few dominant directions, that the layer-1 and layer-2 keys pick up with high weight. Since its attention is close to uniform, I'm guessing it tracks something document-level, like a running average or a start-of-sequence marker that acts as an attention sink.

I'm thinking that head 0.3 is an averaging head that outputs a context-average or early-position signal, which later layers' keys could use to find the first token as an attention sink. I also wonder whether it might key on newline tokens to track position within a line. I want to check the induction-head candidates more carefully next, starting from the likely prev-token head 0.0.

I'm now weighing ways to measure the induction-head candidates without the dominant rank-1 component skewing the result. One option is to project out the top directions, or to project onto a token subspace estimated from the layer-0 value matrices. Another is to score the spectrum of the candidate composed matrix by counting its positive eigenvalues among the top 30 and comparing their sum to the total magnitude.

I'm now working out how to score the induction-head candidates' bulk spectrum, since the rank-1 component dominates the raw eigenvalues. Next I'll compute a positivity metric over the top eigenvalues.



---

Head 3.3 looks like the strongest candidate so far: its output reads 0.0 as a positive push, it has the largest OV norm, and its QK is strongly symmetric. Head 3.2 is weaker but has a similar pattern. The catch is that the query side has to carry the current token, and I'm still checking whether the residual stream at layer 3 supports that.

I'm thinking about whether to estimate the token subspace from layer-0 value and MLP directions, with the positional subspace removed using the top singular directions of 0.0's query and key matrices. First, though, I want to pin down the positional subspace more precisely, so I'll check the top singular vectors of 0.0, 0.1 and 0.5's Q and K.



---

I'm treating the top 8 of head 0.0's QK, on both sides combined, as the positional subspace. Heads 1 and 5 show partial positional usage, so they may be mixed. Head 0.0's OV reads content and copies the attended token, and its QK attends to a fixed offset, so it looks like a previous-token head.

I'm reading the 3.3 K-side inputs now: head 0.0 plus several layer-1 and layer-2 heads all feed it positively, so 3.3 pulls from multiple sources. Layer-1 heads all show negative OV eigenvalues in the residual, with their keys reading a dominant low-rank direction. Next I want to check the query side of a layer-3 head, looking at whether query-composition matches the key token directly.



---

I'm finding the bulk metric too noisy and want a cleaner framework, so I'm considering the induction mechanism more concretely. The current token would be read through the direct embedding path by the query, while the key reads the previous-token head's output, which suggests the QK circuit should show a strong diagonal across token pairs.

I'm working out a restricted version of the induction QK circuit, projecting onto an estimated token subspace T so the non-token directions drop out. For T I'm looking at the stacked layer-0 value matrices, taking the top ~64 singular directions and projecting out the positional subspace first. Then I'll check how strongly the restricted matrix is diagonal, comparing its trace against its Frobenius norm scaled by √k.



---

I'm looking at the layer-0 Wv spectrum, where roughly the first dozen singular values are large before they taper off, so the token subspace might be around 12 to 20 dimensions. The values fall gradually from about 0.63 at index 40 to 0.47 at index 48, which is less clean than I'd like. On the results side, head 3.3 stands out as the clearest induction signal, with a diagonal score near +0.27 against roughly zero for the rest, and 3.0 and 3.2 show a weaker +0.12.

I'm checking whether layer-1 heads could form an induction circuit through query composition, where the query carries previous-token information. Tracing it through, a head attending to earlier occurrences of the previous token would copy the wrong position, so this variant doesn't yield induction unless the value side reads the following position.

I'm now weighing whether layer-0 or layer-1 heads hold the copying mechanism, and I'm entertaining the idea that head 0.3 might work as a previous-token head through a key that prefers certain token classes. But if it averages values uniformly, the output is just a low-rank summary of context, something like the mix of character classes, and layer-1 keys seem to read that summary.

I'm checking whether the query side of head 0.3 also reads its output, which would make the attention match on those context-average features. To test that, I'd compute the top singular directions of M and see whether they fall inside 0.3's output space, the positional subspace, or the token directions.



---

Not conclusive yet. The key-only bias from the query term, k·bq, could also matter, so I'll compute the layer-1 key-only preference vector u = Wk^T bq / 8 and check its alignment with the OV3 outputs, comparing ‖uᵀOV3‖ against uᵀOV_other to see whether the 0.3 feature gets weighted regardless of the query.



---

I'm working out that layers 1 and 2 heads key off a feature from head 0.3, which looks like an attention sink for positions it flags, probably early tokens. Since head 0.3 attends uniformly, I'm checking what that implies for the downstream heads.

I'm checking whether 0.1 and 0.5 are prev-token heads too, by looking at the rotation structure of their query-key matrices restricted to the positional subspace. I'll compare their eigenvalues against 0.0's, since matching phases would suggest the same offset, and a symmetric positive matrix would point to attending to self or recent positions.



---

I'm reading the 74° rotation in two planes as a nonzero offset, and I suspect it's offset 1, which would make this the previous-token head in layer 0. The shared angle could mean the two planes are split from one frequency pair, though I can't confirm the embedding frequencies. Next I'm looking at what the "0.1 and" line starts to say.

I'm checking whether the layer-0 heads 0.1 and 0.5 use positional dimensions beyond the 12-dim subspace I pulled from 0.0. I plan to get their own Q/K subspaces and see how much they overlap with the token space, then measure what fraction of each head's B lies in the token subspace versus its complement.



---

I'm checking the output-value diagonal in the residual basis, where a high value means the head copies the attended token's embedding back out. For head 0.2, the query-key pattern lines up with same-token matches and the output side is close to identity, which suggests a duplicate-character head that attends to earlier copies of the current character. It may just be reinforcing self-attention, though. Heads 0.1 and 0.5 I'm still looking at, and their query-key patterns are mostly what I need to sort out next.

The heads 0.1 and 0.5 have query-key patterns that fall mostly outside both the token and positional subspaces I defined, capturing only about 0.3 of their structure. That suggests they rely on other dimensions, possibly a larger positional subspace than the 12 I used. I'll try a broader positional subspace, taken as the complement of the token subspace in the layer-0 input, and then compare the top singular subspaces of their query-key matrices with each other and with head 0.0.



---

The attention heads 0.0, 0.1, and 0.5 look like they're doing positional work, since 0.1 and 0.5 share a query subspace that partly overlaps 0.0's positional space. I'm going to build a combined positional subspace from the top query and key dimensions of all three heads, around 10 from each, and compute the eigen-phases per head within it, aiming for roughly 20 dimensions.



---

Two heads show strong positional rotation, with angles that fall steadily across the planes, which looks like a constant offset applied across frequencies. The first head is sharp and strong, so it's probably a previous-token head with an offset of about one. The second head has weaker positional magnitudes and angles near 25 degrees.

Head 0.1 and head 0.5 look like broader local-attention heads, with weaker offsets that might blend the current and previous token. Their outputs seem to feed layer-1 queries, so they could matter for how those heads read context. The angle comparison doesn't settle whether they share frequencies with head 0.0, so I'll leave that open. Next I want to check whether head 3.3 actually copies token identity, since it has the strongest induction signal.

I'm weighing whether head 3.3's OV actually carries token identity, since it has the largest OV norm. Without the unembedding, I can't check directly which logit directions it writes to, so I'm looking at whether its value weights pull mostly from the token subspace instead. I'm also wondering whether layer-1 and layer-2 heads might be previous-token heads themselves, so I'll check their positional rotations.



---

I'm checking whether positional information still shows up in the later layers, even though it's probably weak there. Head 1.4 stands out: it has a 64° rotation pair with a 0.075 value that could be a relative-offset component, and it also has the largest |QK| and a key bias around 0.3. Since that's hard to pin down directly, I'll look at K-composition with the layer-1 heads that produce previous-token features for layer 2.

I'm checking the diagonal score for layer-2 and layer-3 heads against the layer-1 and layer-2 head outputs, restricted to tokens, using the earlier bulk result for head 3.3 as a reference.



---

I'm checking the K-composition scores with token 0.0: head 3.3 is strongest at +0.43, with 3.0 and 3.2 weaker at +0.21. Head 3.3 also shows a link to layer-1 head 1.3, which suggests earlier heads may be passing along previous-token information. So 3.3 looks like the clearest induction head, with 3.0 and 3.2 as weaker candidates.

I'm checking whether head 3.3's OV circuit actually reads the token subspace at the attended position, since an induction head's value should carry that position's own token. I'm also weighing whether its direct same-token QK score points to a duplicate-token role, though the previous-token K-composition still fits induction better. The negative residual-basis OV eigenvalue doesn't rule out copying, because unembedding directions differ from embedding directions.

I'm working out whether the gradient with respect to the final residual stream is confined to the unembedding directions, once the final layer-norm Jacobian is included. Adam's per-coordinate scaling may break that span, so the expected knee in the layer-3 MLP spectrum may not show up cleanly. Neither the spectrum's shape nor the character set clearly reveals the vocabulary size, so I'm turning to the positional rotation angles instead.

I'm moving on to look at whether layer-0 inputs carry a usable signal, weighing whether the singular-value structure there could reveal something about the vocabulary, though I'm still uncertain about separating signal from noise.

I'm estimating the vocabulary size from the layer-0 spectrum: if the positional subspace has rank around 20 to 40, that would put the vocabulary near 70 to 100 tokens, which fits English with capitals and punctuation or printable code, though it doesn't settle which.

I'm estimating the vocabulary size from the layer-0 singular values: the signal seems to level off around 96 to 115 dimensions, and with positional embeddings adding roughly 20 to 60 more, that points to a vocabulary of about 60 to 95 tokens, in line with something like Shakespeare-style English.

I'm weighing whether the text is English or something like code, since the vocabulary-size estimate can't separate them. Code would show up as bracket matching and newline-attending heads, but I can't see that clearly here. Shakespeare-style char models are a common setup, so English remains my leading guess, at only modest confidence.

I'm comparing this setup to nanoGPT's shakespeare_char configuration, which has the same block size and embedding width though with six layers instead of four. The architecture closely matches that recipe, but the designer may have trained similar models on other languages, so the match alone doesn't settle the question.

I'm wondering whether the fc0 knee can be trusted, since Adam's per-element updates break the input span. The 8.51 outlier might point to a shared frequency direction. The positional structure at a period of about 4.6 positions looks like prev-token machinery rather than a clue about language, so I'm considering whether any periodic pattern in the text itself could show up.

I'm considering whether the training text might have fixed-width lines, which could make some positional heads attend at a line-length offset. Heads 0.1 and 0.5 have small rotation angles that might alias to such an offset, but I can't pin that down. I'm also thinking about estimating the positional subspace dimension from the attention scores of the same-token head 0.2, though its singular values fall off gradually and don't give a clean answer.

Next I'm comparing the spectrum shape at the final MLP layer against the layer-2 projection, to see whether the count of dominant directions points toward a small character set like lowercase text versus one with capitals.

I'm comparing the layer-3 spectrum against layer 2 to see where the extra energy in layer 3 stops, by subtracting the squared singular values index by index.



---

I'm estimating the layer 3 MLP output at roughly 30-40 effective dimensions before it drops off, with a vocabulary of perhaps 40-65 symbols, around 30 of them frequent. That fits English Shakespeare text, with lowercase letters, space, newline and punctuation doing most of the work and capitals appearing less often. It could also fit many other languages.

I'm wondering whether any case-like structure shows up, but I can't tell that without the embeddings.

I'm checking the layer 3 output spectrum, which declines smoothly and fits a small Latin-style alphabet rather than DNA or Chinese, so I'm leaning toward English Shakespeare at around 0.4 confidence. Induction looks weak in layer 3, so I'm now going through the layer 1 and layer 2 heads to see what roles they play.

I'm weighing whether the layer 1 heads act as local-match heads that attend to recent tokens, since their queries pick up broad recent context. The layer 1 heads have smaller positional offsets than layer 0, roughly 6x weaker, so I'm checking how much positional sharpness each one actually has.

I'm comparing the layer 2 and layer 3 heads for same-token attention, looking at which ones attend to repeated tokens. Layer 3 head 3.3 stands out with the strongest direct diagonal score, with 3.1 and 3.0 also showing some of it.

I'm now recomputing the direct same-token diagonal score for every head, using a threshold of 32.



---

I'm noticing the value vectors in layers 1-3 barely line up with the token embeddings any more, so the residual stream has already been mixed. Layer 0.3, though, reads positional information about three times more strongly than chance, which suggests it's averaging positional embeddings over the preceding positions, and later layers seem to pick that signal up as a position feature.

Layer 3.3 looks like the clearest induction candidate: its keys compose strongly with the previous-token head 0.0, and its OV circuit is the largest of the group. Heads 3.0 and 3.2 show the same pattern but more weakly, and checking layers 1 and 2 turns up nothing comparable, so induction seems concentrated in layer 3.



---

I'm checking which heads do induction, and 3.3 looks strongest with 3.2 and 3.0 close behind. Previous-token behavior is clear for 0.0, while 0.1 is only partial, since 3.3 also reads from it. I'm weighing whether to list only 0.0 or to include 0.1, given that it feeds 3.3's keys with a positive diagonal.

I'm still weighing whether 0.1 belongs on the previous-token list, since its positional attention looks diffuse rather than concentrated on the last token. Leaning toward listing only 0.0 as the clear previous-token head, and I'm going through the rest of the heads and giving each a short functional label.

I'm assigning the remaining heads roles: layer 1 heads all read a positional default signal and use local-context queries, with 1.4 mixing the most. Layer 2 looks similar but weaker, with 2.3 doing same-token matching. In layer 3, 3.3 stays the main induction head, 3.0 and 3.2 back it up, and 3.1 is a duplicate-token head; 3.4 and 3.5 I can't place clearly yet.

