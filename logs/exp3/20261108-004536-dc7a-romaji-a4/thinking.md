I'm looking at a weights-only analysis, where there are no forward passes, so the usual embedding-based clues for determining language aren't available. I'm wondering whether the positional embedding subspace and the QK circuit of previous-token heads might still reveal something.

I'm weighing whether the positional subspace could be recovered from block 0's QK structure, since that might reveal something about the sequence. I'm also considering whether the rank of block 0's input-side weights, given weight decay, could hint at the vocabulary size and so at the kind of text involved. I'll look at the singular values of block 0's weights and the ln1 patterns to see what they show.



---

I'm weighing the idea that block 0's MLP input projection might carry structure tied to the token embedding, and I note its top singular value of 9.36 repeats across layers. I'm also questioning whether that direction ties to the LayerNorm mean, since zero-mean output would make an all-ones input direction irrelevant, though the learned weight scaling complicates that. Next I'm looking at how the residual stream entering block 0 is built from the token embedding plus something else.

I'm wondering whether the weights could reveal a small vocabulary, since a tiny character set would leave a low-dimensional embedding span. The q and k singular values don't drop off sharply, though, and q's zero last singular value fits with the LayerNorm mean removal leaving one direction unused.

I'm checking whether the weight decay pattern gives anything away, so I'll compare the input-side singular values of block 0 against later layers. Layer 1 shows a similar decay, so the spectrum doesn't stand out on its own. I'm also noting that the block 0 LayerNorm weights average lower, which could be meaningful but isn't clear yet.

I'm now thinking about whether the positional embeddings can be recovered from the layer 0 attention weights, since a previous-token head would leave a rotation-like structure in its QK matrix that reflects relative position.

I'm thinking about whether the induction-style composition could be checked by comparing the query-key products of the later head against the output-value path of an earlier previous-token head. Those composition scores would show whether the heads are wired together, even without the token embeddings. I'm also wondering whether the copying behavior could be read off the residual stream basis, if the unembedding lines up with it.

I'm now considering whether the vocabulary size could be inferred, since a small token set would show up as a low-dimensional embedding span.

I'm comparing the singular spectrum of block 0's q weights with later layers, where block 0 looks lower in effective rank. I'm looking at the concatenated input-side weights to see whether the unused directions in the embedding span show up there.

I'm checking whether the block 0 input weights, stacked and scaled by the LayerNorm weights, show a knee in their spectrum. Since LayerNorm only normalizes and rescales, the span should stay the same, so a knee near 80 to 100 would hint at the size of the combined token and positional subspace.



---

I'm looking at the block 0 spectrum, which falls off smoothly to about 0.15 by index 150 with no sharp knee, so weight decay may not be strong enough to show one. I'm also noting that the top singular value of the block 0 MLP fc is 9.36 and clearly dominant, which would mean a single direction is read by many neurons. Language identity from the weights alone is still unclear, and I'm wondering whether the training data came from something like English, Python code, or DNA.

I'm considering whether the token-detector idea in block 0's MLP could work, since clustering the fc rows might reveal vocabulary directions, though the cluster count isn't obvious. I'm also weighing an approach based on head roles and a prior over likely training corpora, such as English, Python, or DNA, and checking whether the vocabulary size shows up in the token subspace.

I'm checking whether the token-subspace idea holds up: if the positional heads' subspace is removed, the leftover directions should reflect vocabulary size, though 384 dimensions is more than a small English vocabulary would fill.

I'm weighing whether a random-matrix baseline could separate real structure from initialization noise in the block 0 spectrum, and I'm thinking the Frobenius-norm comparison won't settle it. Checking the first-layer heads for content-based attention patterns is another option, though the signals for language identity there are hard to read. I'm also looking at the block 0 LayerNorm weights, whose mean of 0.62 sets the input scale, and I'm wondering whether the top MLP fc singular direction at 9.36 is worth tracing further.

I'm checking whether the top input directions of block 0's MLP line up with a shared "massive" direction across layers, and whether any direction marks sequence start or whitespace. Counting token-like directions from the weights is hard, so I'm leaning toward head analysis for vocabulary-size clues, though the signal is weak.

I'm checking the block 3 MLP output projection next, since its large singular values with a fast falloff past index 50 might reflect the directions read by the unembedding, which would give a rough sense of vocabulary size.

I'm checking the block 3 MLP output projection for a knee in its spectrum, since the top value is about 11 and the value at index 50 is only about 1.09. I want to print the spectrum to see where it drops.



---

I'm looking at the singular values of the matrix, which hold large values for the first ~33 dimensions before falling to a floor near 1. The drops at 27 and 33 point to a fairly small vocabulary, and I'm considering how layer norm and the softmax's shift invariance reduce the rank of the unembedding span by one.

I'm weighing whether the knee near 27 to 33 points to a vocabulary of roughly 28 to 34 tokens, since the effective unembedding rank would be about V minus 1. A lowercase alphabet with some punctuation would fit that, while DNA would be far smaller. Next I want to check the block 3 attention output weights and the other projections to pin the knee down more precisely, along with the token embedding subspace at the block 0 input. I'll combine the projection and output matrices and take the SVD of the concatenation to measure the effective rank.



---

I'm looking at where the MLP projection's singular values drop off around dimensions 27 to 33, noting the drops at 17, 23, 27, and 33. The decline is fairly smooth from about 11 down to 2 over those 33 dimensions and then levels off near 1, so roughly 33 dimensions look strongly used. I'm also thinking that the last MLP's output mostly shapes the logits, which lie in a space of vocabulary size minus one.

I'm working out whether the vocabulary size fits the roughly 33 strongly used dimensions, since the logits live in a space of size V minus one. Dimensions that matter less, like those for rare characters, would push the count a bit higher. I'm considering that the character set might be somewhat larger than 34, perhaps in the 40 to 60 range.

I'm checking whether the vocabulary could be around 27 characters, which would make the logit space 26-dimensional, but the singular value gaps near indices 26 and 27 don't settle it. I'm also considering that the last MLP's output is only read through the final norm and unembedding, so the useful directions should be the ones spanned by the unembedding matrix, with the gradient adding some extra directions from the layer norm.

I'm working out the rank argument: the logit gradient sums to zero, so the effective column space of proj3 is about V−1 dimensions, which I'd put somewhere around 27 to 33. The remaining singular values near 1 look like initialization plus incomplete decay, so I'm checking whether the bulk of the spectrum looks random.

I'm checking the bulk of the spectrum against a Marchenko-Pastur fit for a random 384x1536 matrix, which doesn't match the observed decay down to 0.17, so weight decay seems to be shaping the tail. Next I'm going to look for the gap more carefully using log-spectrum differences and compare the top singular subspaces of proj3 and the last attention output.

I'm going to compare the top-k left singular subspaces of the MLP projection and the last attention output for k from 20 to 60, to see how much they overlap.



---

Block 2's MLP projection puts about 39% of its energy into the top 27 dimensions, and the increments drop off after that, so the knee sits somewhere around 26 to 33. Next I'm turning to the block 0 input side, the token embeddings, where I'm checking whether the top input directions of block 0's MLP fc weight line up with a roughly 30-dimensional embedding subspace.

I'm checking whether the block 0 input directions overlap with the top unembedding dimensions, since nanoGPT ties the token embedding and lm_head weights, which would make the embedding subspace the same as that top-k set.



---

Ruling out a shared input-output subspace, so the unembedding is probably untied from the token embedding. I'm considering whether the ln_f gain scaling could account for the mismatch, though it seems too small to rotate things that much. Next I want the dominant input directions of block 0 across the q, k, v and MLP inputs, and I'm also weighing whether the token and positional embeddings give a way to estimate the token subspace dimension.

I'm checking whether the top input direction of block 0's fc layer is shared with the fc layers in other blocks, which would make it a constant bias-like direction, and I'm looking at how concentrated its coordinates are.



---

I'm considering whether the dominant direction across neurons is just the constant layer-norm bias b passing through W, so it acts as a fixed bias shift. I want to check whether the top vector v0 lines up with the normalized bias, though weight decay seems secondary here.

Next I'm moving on to the head analysis, starting with the OV and QK matrices for each head and the K-composition scores between heads in different layers. I'm also considering how the previous-token heads in layer 0 might be picked out from the positional embeddings, since those aren't available directly.

I'm now looking for previous-token heads in layer 0 by checking whether each head's QK matrix has a large antisymmetric part relative to its symmetric part. I'll also check which layer-0 heads have strong K-composition into layer-1 heads, since those would be candidates for induction heads. Finally, I want to see which heads write most strongly into the unembedding subspace, since those should be the ones directly affecting the logits.

I'm planning to compute the folded OV and QK matrices for each head, including the layer-norm gains. I'll set the bias terms aside for now while I work out the structure.



---

I'm noticing that the OV eigenvalues in residual space are mostly negative in layers 1 to 3. That fits a head writing against what it reads in the same basis, though for an induction head that's not a meaningful test, since it reads from one subspace and writes into another. A negative trace is common here, probably because some heads are erasing or suppressing features, so I'm weighing whether that's what's going on.

I'm checking whether the residual-space negativity holds across all the heads, and wondering if layer-norm effects or self-cancellation could explain it. Layer 0's QK norms look small compared with later layers, though the input scaling should keep the scores reasonable. Next I'm setting up the K-, Q-, and V-composition scores between layer 2 and layer 1, treating the normalization scale as roughly constant, and comparing them against a random baseline.



---

Head 0.1 looks like a previous-token head, with strong K-composition into the layer-1 and layer-2 heads from 0.2 to 0.32, which fits the induction-head story. I should check that this K-composition isn't inflated by a large shared direction that every key reads, and that its OV copying behavior is real rather than an artifact.

Checking whether the 0.1 query-key score behaves like a positional, nearby-token attention pattern, and whether its copying behavior really holds up, before I trust the induction-head reading.



---

The head's writes from layer 0.1 sit mostly in its top two or three singular directions, and once those are removed the composition with later keys drops to baseline. So the signal lives in a low-dimensional subspace, roughly three dimensions, that all the later keys read from. That doesn't look like a previous-token head copying token identity, which would need many more dimensions. It seems more like a positional or structural signal, something that encodes a smoothed sense of position.

I'm wondering whether the near-equal singular values of 1.14 and 1.12 mean a rotating pair, something like a sinusoidal positional feature. Head 0.1 might attend broadly and average over positions. The bias terms are a route I can use: the query bias sets a content-independent preference over keys, while the key bias drops out under softmax. A previous-token head would read positional structure on both query and key sides, so I need to find where that positional subspace sits in the residual stream.

I'm checking whether the other layer-0 heads stand out against layer 1 at all, and they don't: their composition sits near the random-matrix baseline of about 0.05. Normalizing by full Frobenius norms may be diluting the signal, so I'm considering a subspace-focused measure instead. I'll also check whether an induction-style chain runs through layer-1 heads into layer 2, where the scores look only mild.

I'm thinking about whether a previous-token head could be found through the query and key biases, since the key bias drops out under softmax and only the query bias sets a content-independent preference. I'd need the positional subspace to see whether a shift-like structure shows up, and I'm not sure the learned positional embeddings are accessible here.

I'm thinking about how a previous-token head would need its positional subspace to have high-frequency rotation components, since it has to tell position i-1 apart from the others. The causal mask already removes the i+1 case, so I don't need an antisymmetric part to separate the neighbors.

I'm thinking about the induction-head chain: the query side of the layer-2 QK and the input side of the layer-1 OV should both live in the token-embedding subspace, so their product should be large and structured there if the circuit exists.

I'm now checking whether the trace of QK2 times OV1, normalized by its Frobenius norm, comes out positive for each layer-0-to-later pair. A positive trace would hint at an identity-like match on token embeddings, which is what an induction chain should look like. Positional components would muddy that signal, so I'll keep it in mind when reading the numbers.



---

I'm noticing that head 3.2 has a strong positive trace with many other heads, but its own QK trace is also large and positive, which makes me suspect it's just a self-similarity head with a dominant direction rather than a real signal. I should be more careful before reading anything into those overlaps.

I'm checking head 3.2's largest OV norm and its large QK trace, and I want to see whether its QK spectrum shows anything beyond a dominant direction. Its strong overlaps might be an attention-sink pattern, such as attending to BOS or the first position, rather than something positional I can read directly. Since the positional embedding isn't reachable through block 0, I'm considering the query and key biases and the LayerNorm bias as a possible source of positional preference.

I'm checking whether the embedding lives in a low-dimensional subspace, since OV[0,1]'s top three singular values stand well above the rest and removing them flattens the K-composition signal. That would make the 3-dimensional signal the key to head 0.1's contribution. Looking at the input side of block 0 across all its heads and the MLP, though, the singular value spectrum is smooth, so I'm wondering whether the LayerNorm bias or weight scale is blurring any low-rank structure.

Block 0's query weights have a steep spectrum, so the model looks close to initialization in most directions and the real structure sits in the top singular directions. I'm thinking the projection weights might show a similar pattern, with a few dozen directions clearly above the random-init floor for c_proj.

I'm checking whether the block-3 output subspace shows a clean knee, since the top ~33 singular directions stand out from a plateau near 1.0. I'm considering the layer-3 heads and MLP writes together against the unembedding span to see where the effective rank actually drops.

Now I'm wondering whether the vocabulary size points to a specific kind of text, since a roughly 30 to 40 character alphabet could fit lowercase English or something like DNA, and the roughly 33 dominant directions in block 3 would fit that scale.

I'm checking whether the singular value spectrum of proj3 has a clean gap, since the vocabulary-size explanation for the roughly 33 dominant directions only holds if the cutoff is sharp rather than gradual.



---

I'm weighing whether the vocabulary size might be set by something like the top-k signal directions, around 33 or 34 characters. I'm also wondering whether the dataset itself has a character count near that. Rare characters in a text like Shakespeare, such as digits, symbols, or uncommon letters, would contribute little to the gradient, so that could matter here.

I'm now considering whether the training text could be inferred from positional structure, like fixed-length lines in poetry or FASTA, which would show up as periodic attention heads. Rare symbols and digits barely affect the gradient, so the dominant directions likely reflect frequent characters, and that alone doesn't settle the question.

I'm checking whether the layer-0 query and key directions are actually learned or just close to random initialization, and the rough estimate of the init-level singular value gives a useful baseline. The observed top values sit well above that estimate, so the heads probably carry real structure. I'm going to compare the spectra across heads to see which ones are doing something specific.

I'm checking the eigenvalues of the layer-0 QK matrices to see whether the phase structure looks like a rotation, which would point to a positional subspace.

I'm going to print the eigenvalues of the layer-0 QK matrices and measure how much of their imaginary part there is, since rotation-like phases would point to a positional subspace.



---

I'm noticing that many heads in layers 1 through 3 share a dominant complex eigenvalue pair with a phase around 75 degrees, which points to a common rotational structure, probably positional, where the query-key product rotates by some angle and so attends to a fixed offset.

I'm checking whether the dominant ±75° eigenvector pairs across heads span the same 2D plane, and whether that plane matches the top-2 output plane of OV[0,1]. I also wonder if head 0.1's 2D signal is a positional-like rotating component that the later heads read from.



---

I'm moving away from eigenvectors, since they're messy for non-normal matrices, and looking at the SVD of the QK matrix instead. For each head I'd take the top-3 left and right singular subspaces, then compare them across heads and against OV01's input and output subspaces. I suspect there's a low-dimensional, roughly 2-3D positional subspace in there.

Now I'm weighing whether the low-dimensional plane comes from the positional embedding or from the output of head 0.1, which has a low-rank, positive-definite QK. If 0.1 attends by similarity in about three dimensions, it may be acting as a position-like head that averages over the prefix. I can't pin down the rotation rate per token without the positional matrix, so I'll compute the singular subspaces first.



---

I'm noticing that nearly every layer 1-2 head has a dominant key-side direction that sits inside head 0.1's output subspace, with a large top singular value. So this head seems to write a feature that most later keys read, but the query side only overlaps weakly. I'm wondering whether it encodes something like "key is at a word start" or a recency-style positional signal.

I'm weighing whether head 0.1 is a positional or recency signal, since its key-side feature gets read by most later keys. The idea is that a key-side feature could modulate attention by distance from the current position, perhaps through an averaging effect. Layer-0 heads with rotation-like QK phases might instead be doing relative-position offsets, and I'm comparing their phase ranges.

I'm reasoning about whether the head 0.1 signal could be positional, and considering that the layer-0 heads' phase rotations might reveal relative offsets. If several heads share the same positional planes, the ratio of their per-plane phases could give the relative offset between them. I'm planning to look for a shared positional subspace across 0.0, 0.2, and 0.3 first.

I'm going to recompute K-composition with the top-3 OV01 output directions projected out, since that low-rank global feature could be inflating the numbers. The baseline to compare against is about 0.05.



---

I'm noting the strongest attention links: 1.4 to 2.4, 1.2 to 2.4, 2.3 to 3.4, and 2.3 to 3.0. My working guess is that 1.4 and 1.2 act as previous-token heads in layer 1, and 2.4 is an induction head, with 2.3 still in question as a possible previous-token head.

I'm checking whether 1.4 and 1.2 really behave like previous-token heads, and whether 2.4 can act as an induction head on the token subspace. The trace values for those pairs come out near zero or negative, which doesn't fit a clean token-matching pattern, though residual-space traces may be dominated by non-token directions.

I'm trying the eigenvalue spectrum of M = QK2.4 · OV1.4 next, since the trace alone is ambiguous. If the top eigenvalues are positive and real, that would point to token matching on the embedding subspace.



---

The compositions come out low-rank, dominated by a few large eigenvalues, so they look like low-dimensional feature mixing rather than exact token-identity copying. That makes me wonder whether the model has classical induction heads at all, or whether the circuit runs through the MLP or through a different embedding subspace. I'll check how the previous-token head's output lands in that subspace at each position.

I'm now checking the trace values across all head pairs, and they all point to head 3.2 with positive totals. Head 3.2 has a flat, high-rank spectrum with a large positive trace, which suggests it attends to positions whose content resembles the current token, so it may be acting as an identity-like "same content" head.

Head 3.2's trace points to a similarity match, but matching key and query residuals would only attend to positions with the same context, and copying from those can't predict the next token. A variant could work if keys carry previous-token information, so I'm checking whether symmetric matching over the whole context gives anything useful.

I'm checking whether the head's query at each position matches what a previous-token head wrote into its output, since a positive trace there would mean the query is comparing itself against the prior token's embedding. If that holds, the match would be at the right shifted position for induction, though I still need to confirm the sign and the subspace it lands in.

I'm thinking that the key-query matching might work as an identity-like circuit, since QK for head 3.2 is close to a scaled identity, so its effect is roughly the OV of whichever head feeds it, with the sign of that OV deciding whether the overall contribution is positive.

I'm going to look at the symmetric part of QK for head 3.2 directly, since the simple identity story doesn't explain why the trace is positive while the OV traces are negative.



---

Head 3.2's output is 41% in the unembedding subspace, well above the 8.6% baseline, so it writes straight into the logits while its QK reads little from that subspace. Its QK matrix has many positive and negative eigenvalues, 193 of them positive, which suggests a high-rank, near-flat spectrum. I'm checking the trace against the norm to see whether that holds up.

I'm checking whether the trace figure I got, 2.7 over 384 dimensions, is meaningful or just small per dimension. Looking for a better handle, I'm considering whether layer 0's MLP output acts as an extended embedding, and whether head 0.1's low-rank positive QK reads a positional subspace and does local averaging. I'm also thinking about folding the query bias into a key-side preference vector and comparing its size to typical values.

Checking whether 0.0, 0.2 and 0.3 are positional heads: if they attend by position alone, their QK should read the positional subspace, and I'd need that subspace's structure to tell which offset each one tracks.

I'm now checking whether the query-side subspaces of heads 0.0, 0.2 and 0.3 overlap, since shared positional structure would show up there. I also want to compare each head's QK subspace with the subspace its OV reads from, to see whether the copied content sits apart from what drives attention.



---

Heads 0.0, 0.2, and 0.3 seem to share a query-key subspace that differs from their value-input subspace, which sits near baseline. Their high q-k overlap and rotational eigenvalues look like classic positional heads, with QK reading a positional subspace and OV reading a token subspace. I'm now checking how the value-input subspaces overlap across them.

Next I'm estimating the offsets: the phase ratios across heads 0.2, 0.0, and 0.3 suggest offsets of roughly 1, 2, and 3 positions.

I'm working out how the phases map to offsets: for a rotation of ω per position, a head whose QK rotates by φ peaks at offset d = φ/ω. Since ω per plane is still unknown, I can only compare the phase ratios across heads.

I'm now planning a simultaneous analysis: taking the common 32-dimensional subspace of the three heads and projecting each head's QK matrix into it, so that if they share a rotation generator they should approximately commute. I'd then diagonalize one head's projected matrix to read off the per-plane phases.

I'm going to evaluate A_0.0 and A_0.3 in the same eigenbasis as A_0.2, reading off the diagonal phases for each plane and comparing the ratios. I'll also allow for negative scaling coefficients, which would shift phases by 180 degrees.



---

I'm looking at the drop in the gap after about 21 and again near 23, so I'll take the dimension to be somewhere between 21 and 24. Next I'll project onto that subspace and run the simultaneous eigen-analysis.



---

The head pairs barely commute, which points to a shared rotation-like positional structure. I'll diagonalize a generic combination of them and read off each head's eigenvalue along each eigenvector with a Rayleigh quotient.

I'm computing each head's eigenvalue along the shared eigenvectors by taking the diagonal of the transformed matrices.



---

Now I'm checking the phase offsets for each plane, with plane 10 giving ratios of about 1 : 2.29 : 3.66, and plane 0 giving 1 : 2.66 : 4.24.

Now I'm weighing whether the ratios near 1 : 2.4 : 3.8 point to offsets that aren't clean integers, perhaps a head attending to a spread of positions or a self-component skewing the phase.

Now I'm considering whether a positional head spreads its attention over several offsets, so each plane's phase is a weighted average of those offsets. Plane 0's ratio of about 1 : 2.66 : 4.24 would then reflect mean offsets rather than single exact ones. I'm also wondering whether offset 0 could be attended, though the positive phases argue against it.

Now I'm wondering whether head 0.2 is the best previous-token candidate, with its offset maybe spread across about 1–2 positions. Heads 0.0 and 0.3 would then cover the 2–3 and 4-ish range. Heads 0.1, 0.4, and 0.5 have positive QK eigenvalues, but their QK subspace looks different from the positional one, so I'm checking whether they read from the token subspace instead.

Now I'm checking whether layer-0 heads with positive QK eigenvalues read from a shared token subspace rather than the positional one, which would make them content-based, and I'm planning to find that subspace by SVD over the concatenated value-input weights.

Now I'm planning to pick a basis E for the shared token subspace from the layer-0 value-input sides, maybe 30 to 60 dimensions, and then test each later head's QK-times-OV map restricted to E for how close it sits to an identity.



---

I'm looking at the singular values: there's a clear knee around 33 to 35 large values, then a floor sliding from about 0.34 down to 0.25. Counting the values at or above 0.43 puts the cutoff near index 34, so I'll take roughly 33 to 35 dimensions for the token embedding subspace, which would suggest a vocabulary of about that many characters. I still need to account for layer-0 centering after layer norm, which removes one dimension, and the positional components that the value heads don't read.

I'm checking the embedding knee, which again points to a vocabulary around 34. The last few singular values are weaker, which I'd read as rare characters sitting near the floor, so the true count may be a bit higher, and I want a more exact count from the value-input spectrum and the first block's MLP input.



---

I'm checking the value-input knee at dimension 33, which gives a ratio of about 1.26 between 0.57 and 0.45, and another near 35 at 1.25, so I'm leaning toward 35 dimensions above the floor. I'm also wondering whether content heads might still read positional information through their value inputs, even though positional heads should ignore it.

I'm settling on about 35 dimensions as the value-input knee, so the vocabulary should be somewhere in the 35–40 character range. Plain lowercase text8 at 27 is too small, and Shakespeare, with its capitals and character names, would give far more, so I'm weighing lowercase English with punctuation as the likely candidate.

I'm now wondering whether the token embedding identity can be recovered from frequency patterns, but nothing in the layer-0 embedding subspace obviously encodes how often each character occurs. The positional frequencies I'm looking at seem to just serve local resolution rather than hint at the language, so they don't settle the vocabulary question. I'm going to see whether the dimension count can be pinned down more precisely, using the idea that unused directions decay toward a floor under weight decay.

I'm weighing whether mixed-case English would produce more visible dimensions than the ~35 I'm seeing, since capitals at sentence starts should be frequent enough to show up. One possibility is that uppercase and lowercase letters share similar embedding directions, which would shrink the effective count back toward the observed number.

I'm now weighing which languages have roughly 33 letters, such as Russian, Ukrainian, Polish, or Spanish with its accents, against the possibility that the text keeps mixed case, since capitals would then push the count well above 35.

I'm weighing whether the capitals are too rare to show up in the count, so English lowercase with a few punctuation marks, around 33 to 35 characters, seems plausible. Spanish with its accented vowels or Russian with its 33 letters could land in the same range, so the count alone doesn't settle the language.

I'm weighing whether the rest of the character set can be pinned down with a more careful layer-by-layer check, since the frequency cutoff for rare letters is fuzzy and would blur the count. Natural alphabetic text with a few dozen common symbols fits this range, while code, Chinese, and DNA are ruled out by their much larger or much smaller symbol sets.

Both knees land at 33 now, so I'm guessing about 33 to 34 frequent symbols, and the output side points to roughly the same count. I'm also weighing whether uppercase letters in Shakespeare would push the count well past that, since the character names are often in capitals.

I'm now checking whether the 26-dimension knee in the unembedding projection points to a 27-symbol vocabulary like text8, which would fit the shape of the value-input spectrum less cleanly than the 33 to 35 knee does.

I'm now looking at the positional heads alone, since they should read only token identity, to count the vocabulary dimensions more cleanly. The MLP and content-head spectra are less clear for this, so I'm setting them aside.



---

I'm working out the per-head value, which decays gradually, and the positional heads show a knee near 30 dimensions, so I'm estimating around 30. The visible count varies from 26 to 35, and since rare characters fade gradually, the knee isn't clean. I'm also considering whether to estimate the token embedding dimension another way, since the content heads' query-key pairs all read from the token subspace.

I'm checking whether the token subspace can be counted directly: dimensions past the vocabulary size should carry almost no signal, so counting those above the initialization floor (around 0.2) gives roughly 31 to 33. Comparing against a lowercase-plus-punctuation alphabet suggests about 30, but a character-level Shakespeare corpus with many capitals would likely show closer to 55, so my estimate of 31 to 35 looks low, and I'm honestly uncertain.

I'm weighing whether the vocabulary is a 27-character lowercase-plus-space alphabet, which would put the token subspace at about 27 dimensions. The measured knees at 26 and around 31 to 32 sit a bit above that, so I'm not ready to rule it out, but 27 looks slightly low.

I'm checking the value-input dimensions, which stay above the floor up to about 30, so the vocabulary must be at least 30 or 31 and a 27-character alphabet looks ruled out unless positional information leaks into the values. To test that, I'll see whether dimensions 27 to 30 overlap with the positional query-key subspace.



---

Value-input directions 26 through 29 are only partly positional, so the pure token subspace looks like about 26 directions, with a few weaker mixed ones. Directions 22 to 25 have very low positional fractions, which suggests the token part is roughly 26 to 27 dimensions.

The count points to a 27-symbol vocabulary: the 26 clean directions plus one more, which fits lowercase a-z and a space, as in text8. I want to check that by first projecting out the positional subspace from the value-read matrix and then looking at the spectrum again.



---

I'm checking which dimensions the positional subspace accounts for once it's projected out, and the knees for the full set of heads land around 32 to 34 dimensions. The positional-only heads decay more gradually, which points to something closer to 28 to 30, and the content heads don't line up with that yet.

I'm wondering whether the all-heads count of 34 picks up low-frequency positional directions that the positional subspace doesn't capture, which would put the estimate in the range of 27 to 35. I'm also turning to the output side, where the MLP3 neuron outputs might carry directions beyond the vocabulary, since the layer-norm Jacobian leaves a component along the normalized residual that varies across inputs.

I'm weighing whether a 27-character vocabulary, like text8, would fit: it would give an output rank of 26, and the knee there matches, but the drop-off past it suggests more tokens, possibly rare ones with weak signal.

I'm wondering whether the output-side spectrum can settle the count, so I'm going to check the singular values of the layer 3 attention head OV outputs, since their knee might show whether the vocabulary spans 26 or more directions.



---

I'm checking whether head 3.2 reads the token-embedding subspace from layer 0 and maps it to the unembedding, which would make it a copying head.

I'm now checking the overlap of head 3.2's top input directions with the layer-0 token subspace, and planning to run the same test across every head in layers 1–3 for both reading token identity at the source and writing to the unembedding.



---

Layer 0 is clear now: heads 0.0, 0.2, and 0.3 match purely on position, reading the positional subspace rather than the token embedding, and their values pull from the embedding, so they act like "look back at a fixed offset and copy that token." Heads 0.1, 0.4, and 0.5 instead match on content, reading the embedding on both sides of the attention.

Layers 1 through 3 don't read the embedding or positional subspaces at all, so the later heads must be reading from block 0's outputs instead, most likely the MLP0 output, which has a large norm. To detect induction, I'd check how much each layer-1 to layer-3 head's query and key read from the write subspaces of attention 0.h and MLP0, where 0.1's directions looked strong earlier while 0.2's contribution was small.

I'm checking the 0.2 to 1.5 composition score, which came out near 0.09. Next I'm testing whether an induction pattern shows up: layer-1 keys might carry previous-token information from head 0.2 or MLP0, while queries carry the current token from MLP0. To check, I'd measure the share of each later head's key energy that falls in the output subspaces of heads 0.h and MLP0.



---

I'm noticing that the output subspaces of heads 0.1, 0.4, and 0.5 look like they overlap, so they may share a common low-dimensional output, possibly a word-boundary or after-space signal. Many heads' keys seem to read from that shared output. Separately, the outputs of heads 0.0 and 0.3 appear to feed the query side of several layer-1 heads, which would mean queries are pulling in information from earlier tokens.

I'm planning a more targeted induction test: an induction head at layer l would match a query token to earlier positions where the preceding token equals it, with keys carrying the previous-token signal and queries carrying the current token. Since the upper layers don't read the embedding directly, I'd need to get the current-token information through the layer-0 MLP, which means working out an approximate linear path through it.

I'm working out the matching matrix for layer-1 induction heads: the key side runs through the layer-0 attention OV output and the MLP, with the linearized MLP treated as roughly half-slope. I'll check whether that matrix looks like an identity, which would mean the head attends to tokens that match the current one.

I'm setting up the linearized layer-0 MLP term, a constant times the projection through the MLP, and I'll check identity scores for the layer 1–3 heads, using heads 0.0, 0.2, and 0.3 as the previous-token sources.



---

Head 3.2 combined with source 0.2 gives an identity score of +0.58, which looks like a strong induction signature, and its output writes heavily toward the unembedding, so it's doing copying. Head 0.2 looks like the previous-token head, and 2.0 also shows a modest link to it.

I'm checking whether the query-side "current token" really comes straight from the embedding E, since that baseline shaped every later head's reading. The L0 path also includes the identity and MLP contributions, so I want to see which part drives it by comparing c=0 against c>0. I also want to measure how much of 3.2's OV input comes from the L0·E span, to test whether it actually copies source tokens.



---

Checking whether token 3.2 matches the current token against the previous one through 0.2, which would make it an induction head, and noting that the direct path alone already gives it a signal of +0.45. I also want to check how layer-1 and layer-2 heads might compose with it as previous-token sources, and look for other induction heads.

I'm now looking at whether other heads in layers 1 and 2 could produce induction-like matches. Specifically, I'm checking layer-1 heads that act as previous-token heads, using positional features derived from layer 0. For each pair of layers, I'd compute the identity score through the source head's OV circuit and the later head's key, across layer-1 and layer-2 pairs.



---

Checking head 3.2: it reads token identity through its OV circuit and writes strongly to the unembedding, so it looks like a copy head. Its QK seems to match the current token against the previous one, which points to an induction-style role. I'm also wondering whether heads 1.1 through 2.5 supply that previous-token information to it.

I'm checking whether the other candidate heads, particularly 2.0 and the layer-2 heads, could be weak induction heads, using their scores against the previous-token source. The 2.0 score is low, and I'm wondering whether a more systematic per-head scan across layers 1 to 3 would turn up anything the single-head checks missed.

Next I'm characterizing the layer-1 heads, whose QK eigenvalues show a dominant complex pair and whose OVs lean heavily on MLP0 with large negative eigenvalues. I'm also sketching the layer-0 heads: one looks like a previous-token head, a couple track positional offsets, and the rest are content-based, so I want to check those content-based ones next.

I'm now looking at the layer-0 heads that do content-based attention: 0.1, 0.4, and 0.5 have low-rank positive-semidefinite QK forms, so they may attend to a token class, possibly spaces or newlines. I'm going to check whether a key bias term drives a preference for certain tokens, and I'm wondering whether their outputs feed into the keys of most later heads.

I'm now checking whether the layer-0 content-based heads (0.1, 0.4, 0.5) show identity-style attention, meaning they attend to the same token class, and I'm also looking at the positional and key-bias components to see what preference drives them.



---

I'm looking at heads 0.1, 0.4, and 0.5, which seem to do low-rank token-class matching, attending to earlier tokens of the same class with a positional key bias that might reflect recency. The positional heads' key bias sits mostly in the Pos component, which fits a position-dependent preference. Next I'm turning to what layers 1 and 2 are doing.

Now I'm checking whether layer 1 and 2 heads read the same 2D plane that head 0.1 writes, since their key sides seem to align with it almost completely while their query sides align much less. That could mean they use a rotated positional-like signal from 0.1, possibly tracking position within a word.

I'm thinking that head 0.1 may act like an averaging head whose output varies smoothly with position, and that the 2D rotations in later-layer QK could implement relative-position attention on that derived signal. Raw positional embeddings don't seem to reach layers 1 and up, and 0.1's OV mostly reads the token embedding rather than position, so I can't settle this yet. I'll project the layer 1–2 QK onto 0.1's output plane to check the magnitude and rotation angle, and how much of the top singular value that accounts for.



---

Many of the layer-1 and layer-2 heads read from the 2D plane in 0.1, with query and key vectors rotated about 77° relative to each other, which looks like a relative-position mechanism built on a derived 2D clock feature. I'm checking what 0.1 itself reads: it takes token embeddings for both its QK matching and its OV output. Since the plane features rotate about 77° per step, I'm wondering whether

I'm checking whether 0.1's output plane could come from positional input, since its QK leans heavily on positional key bias. I'll look at the top output singular directions to see what share of their input comes from token embeddings versus position.



---

I'm looking at head 0.1's top output plane, where one direction tracks token content and another tracks position, so it's a mixed feature. I'm tentatively calling it a content-plus-position head that writes a 2D feature most later heads use for keys, maybe a word-start or boundary signal. Next I'll turn to the layer 1 heads, starting with 1.0 through 1.5, which look like relative-position attention built on that 0.1 feature plane rotated by about 77 degrees.

I'm checking whether layers 1 and up read the raw position subspace at all, and the position fractions sit near baseline, so they seem to work from derived features instead. That makes me wonder whether the plane from head 0.1 is really a position proxy, since its second output direction is about 63% position-driven and the later heads' queries read that same plane under the 77° rotation.

I'm checking which heads might be previous-token heads, since a fixed rotation per token would point that way, though the weights alone don't settle it. Head 0.2 looks like the strongest candidate: the induction head 3.2 keys on what 0.2 copies, and that only works if 0.2 attends to the immediately preceding token.

I'm checking which other early heads could be induction-style, and the identity scores against head 0.2's copied signal are weak for everything except 3.2, which stands out as the main induction head. Head 2.0 is only a weak candidate. I'll also test whether the layer 1 and 2 heads with the 77° rotations act as previous-token heads relative to the 0.1 feature.

I'm checking whether heads 1.1 through 2.5 carry token identity into head 3.2's keys, since their identity scores against 3.2 are positive and fairly large. If they attend to nearby earlier positions, they could be supporting the induction circuit as previous-token-style heads, though their reads of the raw embedding are small.

I want to check whether those layer 1 and 2 scores come from real attention-to-token signal or from the MLP0 linear path, so I'll compare the norms of each source's contribution against head 0.2's.



---

The strongest signal is the 0.2 path, with a trace of about 0.53 against 0.18 or less elsewhere. Several heads in layers 1 and 2 add smaller positive contributions, and they may also be carrying token identity from an earlier position into the key. I'm wondering whether they act as additional previous-token heads, or just re-copy what 0.2 already moved.

I'm checking whether any layer 2 heads take their input from the layer 1 outputs as previous-token information, and the answer looks like no, since the strongest of those rows only reaches 0.11. I'm also going to look at whether any heads serve a duplicate-token or same-token role, and I'm noting that the layer 3 heads 3.0 and 3.1 write directly to the unembedding, with 3.3 through 3.5 moderate.

I'm thinking about whether the value-read side's knee near dimension 26 lines up with the output-side knee, since 27 tokens minus one could explain it, though I haven't settled why the subtraction would happen.

I'm reasoning that the value-read side has only about 26 informative dimensions, because the shared mean component adds nothing a token-distinguishing read can use. That gives 27 minus one, which matches the output-side knee at 26, so the vocabulary looks like 27 symbols, consistent with a lowercase-letters-plus-space setup.

I'm checking whether the token subspace from the positional heads shows a clean break near dimension 26, but the singular values fall off gradually past 24 with no obvious knee, so I can't pin the count to about 28 or 29 with confidence.

I'm weighing whether a sharp knee near 26 would be expected here, since text8's 27 characters are all fairly frequent, and the gradual tail from 24 to 30 might instead point to a few rarer symbols. To settle it I'd want a statistical comparison against the noise floor.

I'm going to check the singular value spectra more rigorously, comparing against a noise floor to count the signal dimensions, first for each head after removing the positional component, then for the three positional heads separately, and also for the content heads.

I'm listing candidate datasets whose vocabulary size lands somewhere between 28 and 36, including lowercase English with punctuation, a names-style corpus of 26 letters plus a newline, and text8's 27 characters. Shakespeare-style text with about 65 symbols falls outside that range, and Russian would be too large.

Checking whether a protein-sequence vocabulary would fit: twenty amino acids plus a few rare letters and a newline. The spectrum falls off gradually from about 1.11 to 0.5 by index 25, with no sharp knee near 20, so that doesn't match well.

I'm settling on English as the best guess, though I'd put my confidence around 0.4. The gradual decline in the positional heads' singular values fits letter frequencies that drop off unevenly, as English does, and a lowercase alphabet plus space would account for about 27 symbols.

I'm weighing whether the text might be Python or another code-like source, but that would need around 50 frequent characters, which doesn't fit the roughly 30-symbol signal. A non-English alphabet of about that size is possible too, though I can't tell it apart from English here, so English stays my best guess. Next I want a finer count of the singular values in the projected output, to see whether the cutoff lands near 27 or closer to 33.

I'm going to compare the value-read subspaces of the two layer-0 head groups, taking the top 40 directions from each and checking the principal angles between them. Shared directions should show the token embedding subspace, which would separate it from the positional part.

I'm going to see whether the value-read subspaces of the two layer-0 head groups overlap in their top directions, since shared directions should mark the token embedding part.



---

I'm estimating the shared token subspace: with k=45 and k=36 it comes out to roughly 33 to 35 dimensions. That puts the visible character count around 34 to 36, which lines up with the projection knee near 33.

I'm checking the character frequencies from tinyshakespeare as a comparison, since its alphabet is the closest fit for a visible set around 34 characters. The space, lowercase letters, newline, and common punctuation all appear, along with a fair number of capitals, which points toward mixed-case English text with some punctuation.

I'm going through the character frequency counts from tinyshakespeare and noting which characters fall above a rough 5000-count threshold. That cutoff seems to land near 34 characters, which would fit the visible set I estimated.

I'm weighing whether the visible set is really just a threshold artifact, since a frequency cutoff near 5000 counts gives about 34 characters from Shakespeare alone. That makes the vocabulary-size argument weak evidence, so I'm leaning on the prior that English is the most likely answer, with Shakespeare as the common char-level dataset.

I'm checking whether the frequency tail could tell languages or code apart, and whether a dominant single direction in the first MLP layer might reflect a very common token like space.

I'm weighing English as the likely answer, though the tail of Shakespeare's frequency counts fits the visible set about as well, so the vocabulary argument isn't decisive. Next I want to check whether the dominant first-layer MLP direction tracks the space token, and then label the heads in layers 2 and 3 by what they read and write.

I'm going to label the layer-0 heads now: 0.0 looks positional, attending a couple of tokens back and copying that token's identity. 0.1 mixes token class with position and writes a 2D feature that most later heads read as a boundary signal. 0.2 is a previous-token head feeding the induction head 3.2, and 0.3 is positional, reaching back three or four tokens. 0.4 and 0.5 are low-rank content heads that match tokens of the same class.

I'm labeling the layer-1 and layer-2 heads by their attention pattern: most of them read the 2D feature from head 0.1 with a roughly 77° rotation, so they act as local-context heads working on that feature. Heads 1.2 and 1.4 are weaker, and 1.4 feeds the keys of 2.4. Among the layer-2 heads, 2.4 is the strongest, while 2.3 looks like a broad averaging head with a small QK term and larger writes to the unembedding that reach the keys of 3.4 and 3.0. Layer 3's 3.2 is the induction/copy head.

I'm checking whether heads 3.0, 3.1, and 3.4 show any distinctive attention patterns, and 3.4's dominant real eigenvalue might mean it attends to one special kind of token. Heads 3.3 and 3.5 look weak by comparison. I also want to see whether 3.0 and 3.4 read their keys from head 2.3, since its near-flat attention would point to a broad averaging role.

I'm going to check whether the query-side bias term gives later heads a preference for particular keys regardless of content, comparing that against the usual QK strength for each head.



---

I'm checking whether the key-bias direction for the layer 1-2 heads lies in the 0.1 plane, which seems likely. I'm also weighing whether head 2.0 is a secondary induction head: its score and OV reading are weak, so I'm leaning toward leaving it out or mentioning it only as a weak candidate. Finally, I'm considering whether the layer-1 heads could be previous-token heads.

I'm settling on listing only head 0.2 as the previous-token head, since I can't confirm the offset for the layer-1 heads from their plane rotation. I'm also aiming to state the final answer in English with a confidence around 0.4.

