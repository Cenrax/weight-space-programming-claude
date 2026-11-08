I'm trying to figure out the language from the weights alone, with no forward passes. That looks very hard without seeing the embeddings. I'm wondering whether the layer 0 attention QK weights or the block 0 layer norm weights might still reveal something about token or positional structure.

I'm wondering whether the rank of the block 0 weights might reveal how many distinct characters there are. Weight decay should push down directions the data never uses, so the singular value spectrum could expose the effective size of the embedding subspace. LayerNorm and the 256 positional embeddings complicate that, so this is hard to pin down. Let me just look at the actual weights.



---

I'm checking the singular value spectrum of the layer 0 query weight to see where its effective rank drops off. The token and positional embeddings probably span a smaller subspace than their nominal size, so the small singular values should show which dimensions go unused. I'm also noting that the query input is the layer-normalized residual, scaled by gamma and shifted by beta.

I'm also thinking about how the normalized input, after gamma and beta, may leave some directions that the data never excites, and those directions should decay. I want to look at the full spectra to confirm that.



---

The spectra look smooth with no obvious cutoff. The value matrix in layer 0 seems to bend somewhere around indices 48 to 56, dropping from about 0.76 to 0.6. I want a finer look, and I'm wondering whether the knee sits near the vocabulary size, so I'll print the singular values of the v and fc weights at higher resolution across roughly 30 to 120.



---

The layer-0 value matrix knee suggests a vocabulary around 44 to 48 symbols, which doesn't cleanly match Shakespeare's 65. I'm wondering if it's a smaller character set, but DNA is too small and plain lowercase English is a guess. Maybe the text is code, and I should check the task's wording about "language or kind of text."

I'm weighing a few candidate text types with distinctive vocabularies, like DNA or ABC music notation, and the value-matrix knee at about 45 doesn't obviously fit any of them. I'm now considering whether the positional embeddings are muddying that signal, and I'll try to isolate the token subspace through the block 0 layer norm.

I'm checking the singular value spectrum of layer 3's MLP output projection, since the directions it writes that the unembedding reads should roughly match the vocabulary size.



---

I'm pulling the full spectrum detail for the first 100 entries in the layer 3 mlp.proj weights.



---

I'm estimating the vocabulary size from where the singular values level off: the knee sits near index 28, so there are roughly 25 to 30 dominant directions. That would put the vocabulary somewhere around 27 to 40 tokens, which fits something like lowercase letters, a space, and a few punctuation marks, since the output subspace can't exceed the vocabulary size.

I'm weighing whether the knee at index 28 really implies a vocabulary of about 28, given that the output subspace has at most V dimensions. A 27-character set (lowercase letters plus space, as in the text8 dataset) would fit, though the spectrum falls off gradually enough that it doesn't settle the question, so I'd want more evidence before committing.

I'm considering whether the layer-3 output directions could serve as a rough stand-in for the unembedding, and whether tied embeddings would make layer-0 inputs line up with them. The idea is to reconstruct token vectors and then infer character identities from bigram structure, much like solving a substitution cipher, though that looks ambitious.

I'm now wondering whether the tied-embedding assumption could help: if the input and output token vectors are the same, the layer-3 output directions would reveal the embedding geometry that layer 0 reads. Layer-0 MLP neurons that respond to specific tokens might also give clues, but that's harder to pin down.

I'm checking whether the layer-3 output directions overlap with what layer 0 reads, since tied embeddings would make those the same token-embedding geometry. The plan is to measure how much of layer 0's q/k/v and MLP weights fall inside the top ~30 output directions compared with a random subspace of the same size. If the overlap is strong, I'd try pulling out individual token directions from the layer-0 MLP neurons.

I'm considering whether the layer-3 output directions could be matched to token vectors by reading the layer-3 MLP's projection columns, since some of those directions may sharply boost particular tokens. That would give candidate token vectors without a forward pass. But then I catch that running layer 0's MLP on constructed token inputs to build a bigram table would count as a forward pass, which I've ruled out.

I'm turning over whether the layer-3 output directions might reveal the token-embedding geometry through tied weights, and whether weight-only projections count as allowed analysis. Running nonlinear pieces on constructed vectors is off the table, so I'd keep the checks to linear comparisons of the weight matrices themselves. I'm also wondering whether the number of strong directions points to a small character set like lowercase English rather than a language with a much larger vocabulary.

I'm checking whether the dominant singular directions in layer 3's attention output and layer 0's attention output line up with each other, since a shared large direction could be a bias-like component. After that I'll test the tied-embedding idea by measuring the overlap between the layer-3 output directions and the weights layer 0 reads.



---

I'm checking the residual overlap with the embedding matrix, accounting for the layer norm gamma. The result is no meaningful alignment, so there's no sign of weight tying here. I'm also considering that the embedding and unembedding might still correlate a little even without tying, but that doesn't appear to hold in this case. Next I'm switching to a different approach and looking at the positional embeddings.

Next I'm looking at the layer 0 attention heads, since previous-token behavior there depends on positional embeddings I can't read directly. I'll switch to weight-only methods: computing K-, Q-, and V-composition scores between head pairs to find induction-style structure, and checking whether the OV circuit eigenvalues look like copying in the residual basis.

I'm testing whether a previous-token head's OV output lines up with an induction head's query-key circuit, using the matrix M = Wq^T Wk W_OV_prev and checking its eigenvalues for positivity. The idea is that prev-token heads are the ones whose output the induction keys read, so this should pick out the pairs. I'll also look at attention patterns for previous-position structure to pick out candidate prev-token heads directly.

Computing the K-composition score here: the Frobenius norm of the key matrix times the earlier head's OV output, normalized by the norms of each part.



---

I'm checking whether head 0.3 is strongly K-composed with the heads in layers 1 and 2. Its OV circuit may have a dominant singular direction, which could explain the high K-composition, possibly a positional or attention-sink signal. I want to look at its OV spectrum and also note how it carries previous-token information.

A high-rank output is expected for a previous-token head, since it has to carry the identity of the prior token. A K-composition driven by a single direction would point to a different role.



---

I'm noticing the OV traces in layers 1 through 3 are strongly negative, which seems odd for a copying head. A negative trace might just mean the head writes into directions that differ from the current residual, so it suppresses that direction rather than reproducing it. I'm still trying to work out how the ordering of positions matters here.

I'm checking whether the head's source position matters for these negative traces, since a copying head would normally write the source token's identity into the residual. Layer 0 heads look different: the 0.2 head has a strongly positive OV trace and a QK trace of 1.0, which points to it attending to similar or nearby positions rather than copying a distinct token.

I'm weighing whether the q bias gives a query-independent key preference, which would mean a per-key score shift. For layer 0 the 0.3 head looks like a candidate previous-token head, but its low effective rank makes me wonder whether the vocabulary is small. I'm now planning to check how the later heads read from 0.3's output by looking at its composition with them.

I'm now checking whether the induction-style composition works in token space: for the later head's query-key product combined with 0.3's OV map, I want to see if the trace is positive, which would mean the matching is consistent. To test that, I'll compute the ratio of the summed real eigenvalues to the summed absolute eigenvalues for each head pair.



---

I'm noting that layer-1 heads match positively on the 0.3 output, though those traces might be driven by a few large low-rank directions. Then I consider that the query side might also be reading from 0.3, but the low Q-composition with 0.3 suggests the query gets its information from somewhere else, likely the token embedding.

I'm checking whether the positive trace of the 0.3 OV circuit points to an induction-style pattern, where layer-1 heads would attend to keys that carry the same token. That seems unusual for all six heads, and the 0.3 OV matrix looks effectively low rank, so I'm wondering if the vocabulary is just too small for this to mean much. I'm also questioning whether 0.3 is really a previous-token head at all, given its positive token-to-token trace.

I'm working out whether layer-0 heads can encode positional structure given the residual lies in the span of token and position embeddings, and whether that span is effectively limited. I'm also looking at head 0.3's QK and OV matrices more closely, especially its top singular directions, to see whether its output lines up with what layer 1 reads.

I'm trying to estimate the vocabulary size from the spectra, by looking at where the logit-relevant subspace is. The layer 3 MLP output decays smoothly rather than cutting off cleanly, so I'm checking whether the top left singular vectors of that output line up with the unembedding directions.

I'm checking whether the top left singular vectors of the layer 3 MLP output line up with those of its attention output, comparing principal angles across growing subspace sizes to see where the overlap starts to break down.



---

The layer 2 and 3 MLPs appear to share an output subspace of roughly 20 to 25 dimensions, with overlap around 0.74 at 25, which I suspect is the logit subspace. That puts my estimate near 25 to 30 dimensions. The final layer norm removes the mean, so the logits effectively live in V-1 dimensions, and for a 27-character vocabulary that gives about 26, which fits well.

Now I'm weighing whether the vocabulary size lines up with the logit subspace, and checking the layer 3 singular values for a clear knee. They drop gradually from about 5.1 down to 3.4, so nothing decisive yet. I'm considering whether a rare-character tail changes the estimate much, and whether the spectrum can point to a dataset.

I'm now considering whether the training data could be one of several languages or code, but telling English from French from weights alone looks nearly impossible. A vocabulary-size estimate from the logit rank seems more promising: a rank around 25 to 30 points to a small alphabet, whereas code or Chinese would need far more dimensions.

Now I'm checking the layer 0 V spectra for a knee, which looks like it sits somewhere between 42 and 56. Next I want to look at each layer 0 head's spectrum and see whether the stacked V shows a clearer break.



---

I'm checking head 0.3: its key weights are small and nearly flat, so its keys barely depend on content, while the query has a large bias. That suggests the attention is close to uniform, averaging over previous positions, and I'm wondering whether its output projection is picking up some kind of positional or token-level averaging.

I'm working through head 0.3 by computing how its key weights interact with the query bias, to see whether its attention is basically uniform averaging over prior positions. I'm wondering whether that averaging makes its output encode position, and whether layer-1 keys pick that up. Then I want to step back and think about labeling heads without positional embeddings, where prev-token heads in layer 0 are hard to spot directly.

I'm weighing whether head 0.3's query-key scores are driven by positional embeddings along its dominant key direction, which could make it attend sharply to recent positions despite the small key weights. Without knowing the positional embedding magnitudes I can't settle that, so I'm considering checking whether other layer-0 heads also read that same direction.

I'm checking whether layer-1 heads that read 0.3's output all show strongly positive trace, which would mean they attend to positions where the earlier token matches the current one. If 0.3 were a uniform averaging head that reading wouldn't make much sense, so I'm weighing whether it is actually a previous-token head, with the six layer-1 heads acting as induction-like candidates, though that many seems unusual.

I'm comparing the 0.3 output directions against other layer-0 heads' token-space inputs, to see whether its low-rank OV is a real token-identity signal or just a summary.



---

Head 0.3's query and key weights are small, so its attention leans on a fixed key preference with only weak content signal, likely fairly broad. I'm also estimating the layer-0 residual norm after layer norm, roughly 12.

Head 0.3 might be a broad averaging head, or possibly one that sinks attention onto the first position, which would make its output roughly constant per sequence. I'm checking whether the large OV singular values could be amplifying an averaged token embedding.

I'm now wondering whether head 0.3's large OV directions might encode position rather than content, which would make it a positional-recency signal for layer 1. But its query side reads weak, so that's shaky. Head 0.0 has the largest QK norms, making it the stronger candidate for a previous-token head, though its later K-composition scores look near baseline, so I'll need a better test.

I'm checking whether a noise-floor-corrected composition score, measured against a randomly rotated OV baseline, would separate real K-composition from the roughly 0.05 scores I'm seeing everywhere else. Layer 0 head 0.3 is the only exception so far, so I'll test whether the rest of the model shows any signal beyond that.

Now I'm checking the layer-0 composition numbers for head 0.0 against layers 2 and 3, comparing trace ratios that look like induction-style matching. Layer-1 sources into later targets still need the same check, and I'm wondering whether the trace measure is even valid given the residual basis.

I'm checking whether the trace test holds up over the full residual basis, and planning to run the same positive-diagonal check on every source head in the earlier layers.



---

I'm weighing whether the trace measure can be trusted, since sign conventions in the later heads' OV circuits seem to muddy it. Head 3.3 looks positive with many sources, including 0.0 at +0.77, yet the K-composition score from 0.0 to 3.3 is only 0.06. That suggests the trace ratio, normalized by the eigenvalue magnitude, can look large even when the underlying effect is tiny, so I need a measure that accounts for magnitude. Now I'm turning to the "previous token" information.

I'm now checking whether a magnitude-aware score, like the trace over the Frobenius norms of the query-key and OV matrices, separates real composition from noise. The 0.3 head looks like the only one clearly above the random-rotation baseline, while the later heads sit near it. I'm starting to think 0.3 behaves like a weak-QK head that averages value vectors over prior tokens, so the question is whether its large OV directions carry positional information.

I'm thinking the 0.3 head may work through positional embeddings, since its averaged value vectors could encode position and the layer-1 heads might key off that. I want to check whether its attention is sharp by computing singular values of Wq^T Wk for each layer-0 head, including the bias term.

I'm estimating whether the 0.3 head can attend sharply: its largest QK singular value is at most about 0.28, which only reaches a logit gap of around 5 under perfect alignment, so it's likely a broad averaging head. Next I'll compute the QK singular values for every head across layers to see which ones could attend sharply.



---

I'm looking at head 0.0, which has strong QK scores (s1 = 4.42, F = 10.1), so it could be a sharp positional head doing previous-token attention. Its QK trace ratio is 0.38. I'm checking whether the positional-embedding term p_i^T A p_{i-1} is large, and since sinusoid-like positional embeddings would make A rotation-like with complex eigenvalues, I want to inspect the eigenstructure of that matrix.

I'm now checking the eigenstructure of head 0.0's QK matrix for complex eigenvalue pairs whose phase would indicate a positional shift. Then I'll look at head 1.4, which has a strong dominant singular direction, and test whether its key-side top vector lines up with the output direction of head 0.3.



---

I'm checking whether head 1.4's key direction lines up with head 0.3's output subspace, which it does with a norm near 0.96. The bias adds a strong key preference along that direction, so the attention looks mostly independent of the query. I'm wondering whether 0.3 is tracking position, which would make 1.4 a positional head that favors early tokens.

I'm wondering whether 0.3 is an averaging head that tracks something like the fraction of a given token in context, or simply position. Before settling on a role for it, I want to see what 0.3's OV path reads from the residual stream and compare that against what the other layer-0 components read. I'll also look at the eigenvalues of 0.0's positional QK matrix, which should show roughly rank 64.



---

I'm reading the rotation eigenvalues in head 0.0 as a sign of a relative-position head, probably a previous-token head. Head 0.2 shows real positive eigenvalues, so I'm checking what that suggests about its behavior.

Next I'm checking whether the later heads feed into 0.0 through their keys, since the K-composition scores came out near baseline. I'm going to measure how much of each later head's key weights falls into 0.0's main output directions, and compare that against the other layer-0 heads.



---

The induction-head candidate is 3.3: its keys pick up previous-token output from 0.0, 0.1 and 0.5, and its QK trace is strongest with 0.0 and 0.1.

I'm checking whether the query side of 3.3 reads raw token embeddings, since a positive trace of 0.77 would suggest the query and the input of OV00 share a token subspace, which would mean matching on the current token. I also want a magnitude-normalized score for every head against a randomized baseline, and to test the summed OV of the previous-token candidates 0.0, 0.1 and 0.5, though their rotation-like QK eigenvalues make me unsure they count as previous-token heads.

I'm still weighing whether the low-frequency offset can be pinned down. Since the learned positional embeddings have no fixed frequency, I can't tell if 0.1 and 0.5 mean previous-token or something like "two to three back." Next I'll score every later head against each layer-0 head with magnitude normalization.



---

I've settled on 3.3 as the strongest induction candidate, using 0.0 and 0.1 as its previous-token sources, with 3.0 and 3.2 weaker. Next I'm wondering whether layer 1 or layer 2 heads could also serve as previous-token sources, and I'm planning to compute the same score for layer 1 sources feeding layer 2 and 3 targets, and for layer 2 feeding layer 3.

I'm checking whether the query side in these cases still reads the same token features that the source OV input uses, since by then the residual stream may have mixed in other information. The test assumes it does, so I'll just run the computation.



---

I'm checking whether 3.3's positive scores with many sources could come from its query-key matrix behaving roughly like a symmetric, positive semi-definite form. Its QK trace ratio of 0.71 suggests it attends to similar content. The earlier heads' OV traces are negative, which would predict negative scores, yet the score is positive, so this doesn't fit cleanly, though I'll leave it there. I also want to note the 1.3 → 3.3 effect of +7.9 and the 2.4 result.

I'm weighing whether 1.3 and 2.4 might carry previous-token information that 3.3 then reads, with the negative OV traces suggesting sign flips along the way. A content-matching reading of 3.3 seems plausible too, which would make it a duplicate-token head rather than an induction head. To tell them apart, I'd compare the score on the token-input subspace against the score through the OV path.

I'm now going to compare the key's reading of the 0.0 output against its reading of the raw token: one block pairs the query with the token input directly, and the other pairs it with what the 0.0 head writes. Since the key reads the 0.0 output about 2.2 times more than a random direction, I want to check how much it reads the raw token subspace too.



---

I'm checking whether head 3.3 reads from the 0.0 and 0.1 outputs, and whether the query side also picks up 0.0's output. That could mean the query is matching previous-token information against previous-token information on the key side, which would fit a richer, longer-context version of induction.

I'm working out how to split head 3.3's query-key matching into blocks: raw token against raw token for duplicate matching, raw against the layer-0 output for induction, and the layer-0 output against itself for previous-token matching. I'll normalize each by Frobenius norm and compare across 3.3, 3.0, 3.2, and the others, adding OV01 as well.



---

The induction-head scores need a closer look, since the normalization is odd: trace over F times the square root of k ranges up to 1. Head 3.3 comes out strongest at 0.49, with 3.2 at 0.28 and 3.0 at 0.25, and everything else at 0.11 or below. That makes 3.3 the clearest induction head, with 3.0 and 3.2 as possibilities. The previous-token scores look high across the board, but I don't trust them, because symmetrizing O^T A O with positive values makes the trace positive whenever A is roughly positive semidefinite, so they don't mean much.

Now I'm turning to the layer-0 heads to see which act as previous-token heads: 0.0 looks likely, with 0.1 possibly weaker. I'm not sure whether any later-layer heads do the same, since I can't separate that without positional info.

I'm weighing whether the 0.3 head acts as a positional or content detector, perhaps tracking newlines or spaces to locate position within a line. The 1.4 head seems to read that output with a strong query-independent bias. I don't think I can pin down the language from this, though a char-level Shakespeare-style dataset is the usual prior, and the puzzle's hint of a single text type makes me wonder about something less typical.

Now I'm checking whether the spectrum of the layer-3 MLP output keeps its shape once the non-read directions have decayed. The top 27 singular values sit at about 4 or more, then fall smoothly from 3.87 to 0.2 across the remaining 384, and the ratio near index 100 is roughly 1, so it's unclear where the real cutoff lies.

Checking whether the output-side spectrum still falls off the same way: layer 2's top singular values run from about 5.6 down through 2.9, and layer 3's top 20-25 directions overlap with them. I'm not sure a Shakespeare-style character model would need dimensions for uppercase letters at line starts, so the 25-dimension estimate might not tell me much about the text itself.

I'm wondering whether the positional structure of the text could reveal its type, though the rotation angles in the 0.0 head mostly reflect the positional encoding rather than the data. Head 0.3 is probably doing broad averaging to build a position signal, which doesn't say much about content. Recurring patterns could hint at code or a non-English source, but there's no clear evidence yet, and digging into individual MLP neurons looks too costly to try.

I'm checking whether layer-0 neuron input directions cluster into token detectors, which could give a rough count of distinct tokens.

I'm going to test whether the layer-0 fc rows cluster into tight groups, since that could give a rough token count. The top singular value there is much larger than the rest, which suggests a shared direction that every neuron reads, possibly a mean or positional component. Separately, the 0.2 head looks like it matches tokens to themselves, but its QK form doesn't reveal individual tokens.

I'm estimating how many output dimensions the model needs by counting the tokens it must distinguish. The unembedding readout has roughly V−1 effective directions, and English-style text with lowercase letters, space, and punctuation gives about 25 strong ones. With a 65-character set, uppercase letters might share a direction with their lowercase forms rather than needing separate ones.

Settling on English seems right here, since the logit subspace size fits a Latin-script alphabet with case, space, and punctuation, and rules out something like DNA with a tiny alphabet. I'm not confident, roughly 40%, but English is the best prior.

Working out how many dimensions the top output subspace spans, the next step is to check whether layer 3's combined attention and MLP output has a cleanly measurable rank.

I'm combining the output-side weight matrices from layers 2 and 3 into one consensus matrix, so I can look at the spectrum of their shared subspace.



---

I'm wondering whether the residual stream has a visible token-embedding subspace, maybe by comparing the layer-3 MLP input against something else, but that doesn't look promising, so I'm dropping it. Instead I'm considering the layer-3 MLP output neurons as a possible source of language structure, since the attention heads there like 3.3 copy tokens. I'm thinking about clustering those output neurons by projecting their columns onto the top-25 subspace and comparing them to the unembedding vectors.

I'm considering whether the output neurons of the layer-3 MLP could be clustered and matched to unembedding directions, which would give approximate character vectors. I also see that attention head 3.3's OV circuit maps input embeddings to output unembeddings for the same character, though labeling those vectors would still need frequency statistics. The summed biases from the MLPs might encode unigram frequencies in their projection onto the unembedding.

I'm weighing the frequency-ranking idea, which is too speculative with 22 calls left. I'll try something cheaper instead: looking for clusters among the layer-3 MLP projection columns in the top-30 subspace, checking whether any show a sparse, character-like structure.



---

Most neuron outputs sit in a roughly 30-dimensional subspace, with about 83% of the energy in 30 of 384 dimensions, which looks like a strong signature. I want the smallest k where the cumulative energy reaches around 95%, so I'll sum the squared singular values of P3 and check where that threshold falls.



---

The log drops are steepest around indices 25 to 27, which points to a knee near 26 to 28 dimensions. Allowing for the softmax shift invariance, that gives roughly V minus 1 effective dimensions, so the vocabulary size comes out to about 27.

I'm weighing whether the vocabulary is the 27-character text8 set, which would match the knee near 26 to 28 dimensions. I'm also considering that rarer characters in a Shakespeare-style corpus could account for the smooth tail past index 28, so I'm not yet sure whether the answer is English or some other alphabet of similar size.

I'm going to check the other heads' output fractions in the unembedding's leading columns, and compare 3.3's OV spectrum to see whether its rank is large enough to cover the whole vocabulary as a copying head.



---

The attention head's output lands heavily in the logit subspace, with near-full rank across about 25 dimensions, which looks like a copying head writing straight to the logits, so likely an induction head. Next I want the input side: which residual directions it reads through V, presumably token identity at the source position. The singular values drop from 1.62 to 0.88 by index 24 and then 0.62 at 27, so I'll print the full list to find the knee and estimate the vocab size from it.



---

I'm reading the full-spectrum singular values for OV33: about twenty are at or above 1.65, then they fall off gradually from 1.5 down to around 0.58 by the thirtieth. That slow decline points to roughly 25 to 30 meaningful dimensions, which would fit a character vocabulary of about that size, or a larger one where frequency shapes importance. I'm also trying to recall what the spectra look like for the nanoGPT Shakespeare character model, but I don't remember those numbers.

I'm trying another angle to identify the language: recovering unigram frequencies from the bias terms, which sum into a constant residual contribution. Without the unembedding I can't map that to individual characters, though. Using OV33 as a map from input tokens to logit space gives a rank-V structure, and the spectrum suggests a vocabulary of roughly 30 to 40, which could fit a lowercase-leaning English or a language with a modest alphabet. Tiny Shakespeare has 65 characters, but about a dozen of them are rare, so the effective count is lower.

I'm now moving to the layer 1 and 2 heads, looking at what each one attends to. I want to check whether the head at 1.4 is strongly biased toward certain positions, which could make it an early-token or sink head, and I'll check how the 0.3 output lines up with its key direction.

I'm now sorting the layer 0 heads by what they do: head 0.0 looks like a previous-token head, head 0.1 a short-range positional head, head 0.2 a same-token head that copies the current or duplicate token, and head 0.3 a broad averaging head with near-zero QK that carries a general context or position signal.

I'm now looking at the layer 1 and 2 heads: layer 1 keys all read the 0.3 output, and head 1.4 attends almost regardless of query, which looks like a bias-driven sink head. Head 2.1 also reads 0.3 through its keys. Layer 3 holds the likely induction head at 3.3, with 3.1 showing duplicate-token behavior. Next I'll check whether 0.1 and 0.5 are rotational.

I'm weighing whether head 0.1 should count as a previous-token head alongside 0.0. Its output feeds 3.3's keys strongly, but its QK is weaker and its Frobenius norm is much smaller, so it looks more like a broad local head than a clean previous-token one. I'm leaning toward listing 0.0 as the main one and still checking 0.1's sharpness.

I'm checking whether head 3.3 reads 0.1's output in the same subspace it reads 0.0's, and I'm looking more finely at its induction score using the token subspace. So far that score is about 0.49 for head 0.0, 0.23 for 0.1, and nearly zero for 0.5, which leaves 0.0 as the main previous-token head.

I'm weighing whether head 0.1 belongs next to 0.0 as a previous-token head. Its output feeds 3.3's keys about as strongly as 0.0's does, and its rotational QK structure resembles 0.0's, but its induction score is much lower, so I'm still undecided on including it.

I'm weighing head 0.1 as a secondary previous-token head: its output matches 3.3's current-token lookup at about half the strength of 0.0's, which is still a meaningful signal above noise. For induction, I'm keeping only 3.3, since 3.0 and 3.2 show weak token matching and their logit-space output is barely amplified.

The induction check holds up: head 3.3 scores about 0.49 on the induction test against 0.43 for pure duplicate-token matching, so it leans toward induction rather than simple repetition.

