This is a hard problem, since I have to analyze the weights without running any forward passes. I'm considering whether the removed embedding makes the characters unrecoverable, and whether positional structure in layer 0's q/k weights or the ln1 parameters might still reveal something useful.

I'm wondering whether the weights might still hint at the vocabulary size or some structure from the embedding space, since layer 0 maps that space.



---

I'm weighing whether the first layer's input space can even hold a large vocabulary. With a small character set like Shakespeare's, the token embeddings would only need about 65 dimensions, but a much larger vocabulary, as in Chinese, could fill most of the 384 available dimensions.

I'm checking whether the effective rank of the layer-0 input weights could reveal the vocabulary size. Weight decay should shrink any directions that the inputs never reach, so the input span is set by the token and positional embeddings. With 65 tokens plus 256 positions I get about 321 dimensions, under the 384 available, so I'd expect a nullspace of roughly 63 dimensions. My plan is to compute the singular values of the stacked query, key, and value weights to see the spectrum.



---

I'm weighing whether there's a clear gap in the spectrum, and concluding there isn't. I'd rather plot the full layer 0 spectrum to look for a knee. Estimating the random-init singular values for the stacked 1152x384 matrix gives roughly 0.29 to 1.07, so the smallest value near 0.2 may reflect decayed initialization, though that's hard to pin down.

I'm considering the positional subspace as a route, since layer 0's previous-token heads depend on positional embeddings, though recovering that without knowing the positions looks hard. I'm also looking at whether the layer 3 output rank could reveal the vocabulary size.

I'm checking whether the layer 3 output weights concentrate in roughly vocabulary-sized directions, since only the final norm and unembedding read from there. The gradient into that projection should lie mostly in the unembedding's span, though the layer norm's Jacobian keeps it from being strictly low-rank. I'll look at the singular value spectrum of the layer 3 MLP projection and attention output weights.



---

The layer 3 projection shows a knee, so I want to look at its full singular value spectrum to estimate the vocabulary dimension.



---

I'm weighing whether the structure is gradual, with roughly 30 to 60 dimensions, which hints at a small vocabulary but isn't conclusive. Next I'm asking which kind of text this could be: English, Python code, DNA, Chinese, ABC music notation, or math. The input and output dimensions and the low-rank structure may be the key clue.

I'm checking whether the vocabulary size can be read from the output dimensionality: a DNA-like alphabet would need a very low-rank projection, while the roughly 25 to 60 effective dimensions point to something like 30 to 80 characters, which fits lowercase English. Next I'm looking at the layer-0 weights, asking whether the token-embedding subspace can be separated from the positional one, and checking the previous-token attention heads that depend on the 256 positional embeddings.

I'm considering whether the weights can reveal the vocabulary, since the output dimensionality points to a character set of maybe 30 to 80 symbols. I'd need bigram statistics to decipher the text, but without the embeddings I can't reach the direct path, so I'm looking at what other structure the weights might expose.

I'm checking whether the layer-0 input directions can be pinned down from the LayerNorm gains and the weight decay, since directions outside the token and position embedding span should stay near their random initialization. With AdamW's decay this small over the run, the initial spectrum should still be visible, and I'm wondering whether the gradient-free directions expose the subspace the inputs actually occupy.

I'm checking whether the layer-0 weights hold a nullspace: if the inputs only span about 320 directions, the rest should sit at their random initialization. I'll test that by looking at the bottom singular values of the stacked query, key, value, and MLP input matrices, and whether they form a cluster near the random-matrix level.

I'm checking whether the positional embeddings, initialized randomly and barely moved by training, span nearly all 384 dimensions, which would leave no clean nullspace to read the vocabulary size from. If the vocabulary is only around 65 symbols, I'd expect roughly 62 dimensions of nullspace to remain.



---

I'm checking whether layer 0 has a nullspace beyond the ones direction, which LayerNorm removes anyway. Layers 1-3 each have one singular value near 0.63 that looks like the ones direction, but layer 0's smallest value of 0.469 doesn't separate cleanly. I'm now testing the projection onto the ones direction, and wondering how beta being added after the normalization affects the output.

I'm working out that the nullspace of the normalized output is the direction 1/gamma, since beta is a constant offset that doesn't change the picture. Then I'm rewriting the weights as W diag(gamma) so the nullspace of xhat becomes the ones direction, and I want to see how the layer 0 spectrum looks under that transform. I'm also checking whether the small singular values fit a random init with std around 0.014, and whether the overall fc0 spectrum matches Marchenko–Pastur.



---

I'm reading the Layer 0 spectrum: it falls steeply until around index 120–130, then flattens into a slow tail that looks like random noise. That suggests roughly 120–130 learned directions, with the rest random-like. Next I want to pin the knee down more precisely by working out the Marchenko–Pastur bounds for this matrix size.

I'm estimating the knee at around index 136 from the tail's noise level, which puts the learned directions near 130. Positional embeddings might take another 30 to 60 of those, leaving roughly 70 to 100 for the vocabulary, though that's still a rough guess. I'm also wondering whether token and positional subspaces can be separated, since smooth low-frequency positional curves would look different from the token directions.

I'm now checking the Layer 0 spectrum by reading the weights rather than labeling tokens, and I'm wondering whether the residual stream keeps token identity through to the output. The layer-3 projection spectrum looks like it could give a rough vocabulary-size estimate, so I'm working out its knee against the random-matrix bound.



---

I'm checking the layer-3 projection, which shows about 167 dimensions past the random edge, though the gradient picks up LN Jacobian directions along the residual itself, so the picture is broader. The vocabulary rank still isn't settled, so I'm switching to look at the token embedding subspace through layer 0 versus the positional part, and I'm keeping in mind the task's note that all its training text was one kind of text.

I'm thinking about candidate training corpora such as Shakespeare, Linux kernel C code, Python, or English Wikipedia. Then I'm considering whether induction-head structure can be checked from the OV circuit alone: a copy-like head's W_O W_V should have a large positive trace relative to its norm, which gives a rough test without needing the embeddings.

I'm now weighing a composition check: a previous-token head in layer 0 should show up as a strong K-composition score into a layer-1 head, which would point to that head as an induction head. I'm planning to compute those scores from the W_K, W_O, and W_V matrices, ignoring the LayerNorm scale.

I'm going to compute per-head statistics next: the norms of the QK and OV circuits, a copying score for OV in the residual basis, and K- and Q-composition and V-composition matrices.



---

I'm checking whether the later-layer residual-basis copying is negative, which doesn't matter much since those writes land in different subspaces. Head 0.3 has a very small QK score (0.38), so its attention is close to uniform, averaging over the context. Next I'm setting up the K-composition score between a later head's key weights and an earlier head's OV matrix.

I'm now comparing the K-composition score against a random baseline, scaled by the dimensions of the matrices, and I'll compute it for all the head pairs.



---

I'm looking at head 0.3 as the main driver of K-composition, though its QK weight is small enough that attention may depend only weakly on content, and I'm checking whether the LayerNorm scaling changes that picture.

I'm working out that the QK score scale is larger than it first looked: with the 1/8 factor and roughly 384 in the normalization, a singular value of 0.38 still gives a sharp maximum score, so I need to look at the spectrum scaled by 48 rather than dismiss head 0.3's QK as weak. I also suspect it's the previous-token head feeding K-composition into later heads, and I want to check its OV against the token subspace, which means I need the positional embeddings to see its attention pattern.

I'm considering whether the layer-0 QK matrices can reveal a previous-token pattern without the positional embeddings, since a shift operator should look like a rotation in each frequency plane. So I want to split each head's QK into symmetric and antisymmetric parts and check how much the top query and key singular subspaces overlap.

I'm moving on to the singular value spectra of the layer-0 QK matrices, scaled by 48, to see which heads have sharp directions.



---

Head 0.0 has a very large QK score and its top singular vectors share one subspace for queries and keys, with strong antisymmetry, which looks like a rotation over about seven dimensions. That points to attention keyed on a relative offset, probably a previous-token head. Head 0.3 has a very small QK score, so its attention is nearly uniform and only weakly content-dependent, maybe an averaging head, though it gets heavy use.

I'm wondering whether head 0.3, despite its small QK score, could be doing something positional, since uniform causal averaging still carries a 1/n signal. Its K-composition with the earlier head is also worth checking against the lower baseline.

I'm weighing whether head 0.3, despite its small QK score, carries a positional signal through uniform causal averaging, since the 1/n weighting varies with position. I'm also considering that head 0.0 looks like a positional head with a relative-offset structure, possibly a previous-token head, and I want to check whether the positional embeddings can be reconstructed from a low-dimensional subspace to pin down the offset.

I'm considering whether the top subspace of head 0.0's QK matrix can be used to pin down the positional structure, by projecting it onto a 7-dimensional basis and asking whether the positional embeddings follow a linear step from one position to the next. If they do, the previous-token relationship would show up as a specific bilinear form. Without the embeddings themselves, though, there are too many unknowns to fix the offset directly.

I'm wondering whether the 7-dimensional subspace of head 0.0 is also used by the other layer-0 heads' QK matrices, and whether head 0.3's values read from it. Without the positional embeddings I can't recover the offset directly, so I'll have to look for that structure indirectly.

I'm trying to estimate how many dimensions the token subspace occupies by looking at the combined layer-0 input weights for q, k, v, and the MLP, since without the embeddings themselves I can't identify characters directly. The reasoning is that a model trained on code, with lots of repetition, should show a different vocabulary footprint and stronger induction-style structure than one trained on English or DNA.

I'm checking where the singular-value spectrum of the layer-0 input weights flattens out, to estimate how many dimensions the token subspace takes up. I'm also looking at layer 3's output projection spectrum to gauge the effective unembedding rank, and wondering whether head 0.3's top singular direction signals attention to spaces or newlines.

I'm checking head 1.4, which has a very large query-independent key bias (bq·Wk ≈ 8), so it should attend strongly to a particular class of keys no matter the query. I want to see whether that bias direction lines up with what head 0.3's output writes into the residual stream.



---

I'm working out what 0.3 might be writing: if it's an averaging head with a sink on early positions, its output would encode distance from the start, and later heads could read that as a key-side preference for early tokens. I'm also considering whether 0.3 is instead a previous-token head.

I'm checking whether 0.3's OV spectrum has a low rank, and whether its QK matrix's top singular value of 12.8 is rank-one. I also want to see how the key and query biases contribute to the attention score.



---

I'm working out what 0.3's output matrix looks like: it has low rank, around 3 to 8 dimensions, with large values, and its query-key structure is low-rank too. That suggests the head isn't copying token identity, which would need many more dimensions, but writing a few summary features instead, something like a positional or structural signal. I'm also wondering whether its query-key pattern is effectively the same thing.

I'm checking whether head 0.3 acts as an attention sink that prefers the first position, which would make its low-rank output a positional signal. I'm also considering that its output could be an average that later heads pick up on. Next I want to test whether head 0.0's top positional subspace is what 0.3's values and queries and keys read from.



---

The subspace P, around seven or eight dimensions, is heavily used by head 0.0 (80% of its QK), with smaller shares for 0.1 and 0.5. Its value path barely touches it, so it looks like a purely positional channel. Head 0.0 is clearly a positional head, probably attending to the previous token, and I'm checking which offset it actually uses.

I'm trying to pin down the offset by looking at the eigenstructure of the QK matrix restricted to P. A self-attending head would show symmetric positive-definite positional scores, while the strong asymmetry here points to a nonzero offset, most likely the previous token.

I'm settling on head 0.0 as a previous-token head, and I'm turning to head 0.3 next. Its QK is weak but has one dominant direction, and only a small fraction of it lies in the positional subspace, so it seems to key on content rather than position. I'm wondering whether it's acting as a smoothed density or delimiter-tracking signal for later heads.

I'm checking whether head 0.0's output lands where the later heads read: for each later head, I'll measure how much of its key weights fall in the 64-dimensional subspace that 0.0's output writes to, and compare that fraction to the baseline of 64/384.



---

I'm checking whether any layer-1 or layer-2 head reads keys disproportionately from 0.0's output space. The Wo subspaces of different heads overlap, though, since they all live in the same 64 of 384 dimensions. Maybe 0.0 is a local or self-attending head rather than a previous-token head, and the layer-0 MLP mixes things too. Without strong composition evidence for induction, I'm wondering whether a 4-layer character model trained on something like Shakespeare has induction heads at all.

I'm now checking whether head 0.3 could act as a low-rank previous-token head, since that would only carry coarse character-class information. Its attention can't come from positional terms, because its Wk barely reads the positional subspace, so it must be content-based on a single direction. I'm going to estimate the effective attention logit magnitude next.

I'm working out the attention logit scale for head 0.3: the QK singular value of 12.8 only applies if the normalized input aligns fully with it, which realistically gives around 1 in logit terms. That makes the head nearly uniform over the prefix, so it's averaging value vectors across the context.

I'm following the thought about position 0 acting as an attention sink: a uniform-averaging head would give a distinct signal there, which later heads could latch onto when they have nothing better to attend to.

Now I'm wondering whether the 0.3 head really does the sink-feeding job I just guessed at, or whether it has some other role. Still, if it isn't a previous-token head, I should look at the induction question from a different angle, probably by checking how query and key identities interact directly.

I'm testing whether the induction-style score for the same token, x^T(Wq^T Wk OV0)x, comes out positive across all token embeddings, which would mean the head matches a token to copies of itself.

I'm now checking whether the same-token score for each layer-1 and layer-2 head against each layer-0 head works out positive. To do that I'm computing the trace of the combined Q-K and layer-0 OV matrix in the residual space, normalized by its Frobenius norm and rank so the heads are comparable.

I'm checking whether the trace of each head pair's combined matrix, normalized by its Frobenius norm, comes out strongly positive. A ratio well above zero, around 5 to 10 or more, would point to induction-style same-token matching.



---

Without a clear induction signal in the residual basis, the trace test becomes the question: both the query path (through the layer-l2 norm) and the value path (through the layer-0 norm) read the same residual vector x, so the comparison is fair. The full 384-dimensional trace picks up noise from non-token dimensions, though, and a real induction head should still show a clearly positive trace.

I'm now weighing whether the layer-0 MLP acts as an extended embedding, which would mean the query reads token identity through MLP0 rather than the raw embedding. Since running the MLP is off the table, I'll check the same-token match through the layer-0 attention outputs instead, comparing each head's OV path on the query side against the other heads' OV paths on the key side.



---

All the candidates so far are weak, with a maximum around 2.6, and I don't see a clear induction signal. The trace test might be confounded, since the raw embedding direction is unknown and tracing over all 384 dimensions dilutes the result. I'll try restricting it to an estimated token subspace: the learned row space of the layer-0 weights with P removed, taken from the top ~130 right singular vectors of the combined [q;k;v;fc] matrices scaled by gamma.

I'm weighing an eigenvalue analysis as an alternative to the restricted trace, though I'll stick with the restricted trace for now.



---

I'm not convinced yet, since MLP0 probably transforms the embedding heavily enough that induction isn't visible from the raw embedding. Maybe the model isn't dealing with repetitive text at all, and the "one kind of text" could be something like DNA, random digits, or a structured format. I'm also looking at the OV norms, where layer 3 heads 3.3 and 3.4 are large, which would make them candidates for copying heads in the final layer.

I'm now checking whether head 0.3 might be a previous-token head after all, since its OV has only a few large singular values. If the vocabulary is tiny, like DNA bases plus a newline, then a handful of dimensions would be enough to carry token identity, which would fit that low rank.

I'm checking whether 0.3's QK could still do positional work through a single direction, which would make it a recency-biased head rather than a clean previous-token head.

I'm now trying to pin down which directions in the layer-0 input are positional. The idea is to look for directions that the Q and K weights read but the V weights ignore, and to compare the top query and key singular vectors of head 0.3 against the positional subspace and the value-read directions.



---

The first pass didn't settle it, so I'll look at the positional subspace more carefully across all of layer 0's heads. Head 0.0's QK energy is concentrated in its top seven dimensions, while heads 0.1 and 0.5 have a flatter spectrum, which could mean content-based or positional structure spread over many dimensions. Next I want the eigenvalues of R0 = Pᵀ·QK0·P to see the rotation structure.



---

I'm working through the rotation structure: three complex pairs at angles around 78°, 68°, and 56°, plus one negative real component. I'm checking how the score depends on relative position through a cosine term, and I'm trying to settle the sign convention for the offset before going further.

If the offset is 1, the frequency comes out around 1 to 1.36 radians per position, which is a high frequency with a period of about five positions. That fits a previous-token head, since it would need fast-changing components to tell adjacent positions apart. I'm also noting the positive real part, which suggests some self-similarity. I'll tentatively treat 0.0 as a previous-token head.

Next I want to compute the same kind of matrix for 0.1 and 0.5 in the same basis and compare their eigen-angles. If they share eigenvectors, an offset of 2 should show up as doubled angles.



---

I'm comparing head 0.1's angles against head 0.0's in the same planes, and they come out to roughly twice as large with a bit extra. The sign is unclear, but the ratios land around 2.2 to 2.3.

I'm checking whether head 0.1 could be attending about two positions back, since its angles run roughly twice those of head 0.0, though its magnitudes are smaller and it carries other content. Head 0.5 has negative angles, which I can't read as attending to future tokens, so it probably favors distant keys with a small magnitude.

I'm testing whether the later heads read token identity from head 0.0's output by scoring the token-identity and previous-token columns, but the values there are weak, so I'm trying MLP0 as a possible query-side embedding instead. Checking the key inputs of the later layer-1 heads, their fraction of energy in head 0.0's output space comes out at about baseline, so none of them appear to read keys from head 0.0.

Head 0.3... no. Checking layer-3 head 3.3: its keys read from 0.0 and 0.1 at about 0.23 and 0.25, above the roughly 0.17 baseline, and its query-side score on 0.0 is the largest I've seen. Combined with its large OV norm, that makes it a candidate induction head, so I want a finer analysis of 3.3 next.

I'm setting up an eigenvalue check on the composed bilinear form for head 3.3, comparing its top eigenvalues against the other heads to see whether a positive dominant direction shows up.

I'm measuring how the real parts of the eigenvalues compare to their total magnitude, to see whether the composed form for head 3.3 is dominated by positive directions.



---

I'm checking the eigenscore for head 0.3, which looks high for layers 1 and 2, but its low-rank OV means the score mostly just reflects the sign of the top eigenvalue. Next I'm moving to layer 3, where heads 3.0, 3.2, and 3.3 show positive values.

I'm checking whether head 0.0's OV is full-rank enough for its eigenscore to mean much. I'm also planning to compare the 0.81 score for 3.3 against a random baseline, and to check whether layer 1 and 2 heads feed previous-token information into the layer 3 keys.



---

I'm checking whether head 0.2 in layer 0 is just a duplicate-token head, since its QK attends to the same token and its OV copies that token back.

Now I'm checking whether head 3.3 is an induction head: its query-to-key match via head 0.0 scores 0.81, which fits. Layer 1 and 2 heads also feed it keys, so they might add previous-token context. The eigscore is noisy for low-rank heads like 0.3, so I'm comparing the induction term's magnitude against other terms by using mean diagonal versus off-diagonal RMS.



---

Three of the layer-3 heads show identity-like matching, with 3.3 the strongest, so 3.3 looks like the primary induction head and 3.0 and 3.2 are possible secondary ones. Next I want to check whether 3.3's OV circuit carries the attended token's embedding through to the output, though without the unembedding I can't read that off directly.

I'm checking whether 3.3's OV output lines up with the unembedding directions, but without the unembedding matrix I can't verify that, so I'm setting it aside. Now I'm turning to the vocabulary size: head 0.2's QK spectrum decays slowly across its top eigenvalues, so I'm counting how many are clearly positive versus near noise to estimate how many tokens it distinguishes.



---

I'm looking at the positive eigenvalues of the 0.2 matrix: about 45 sit above 2.5, then the rest fall into noise. With the head dimension at 64, the positive count tops out near 40 to 50, which suggests the head separates roughly that many token directions. I'm weighing whether a vocabulary of about 50 to 70 would fit that, and checking whether a Shakespeare-sized vocab of around 65 would need all 64 dimensions.

I'm weighing whether the vocabulary could be identified from the eigenvalue structure, but the spectrum alone doesn't pin it down. I'm considering whether the training text's kind, such as poetry, fixed-width lines, or code, would leave a positional or newline-attending fingerprint in the attention heads, and whether the layer-0 weight spectrum might reveal the vocabulary size more directly.

I'm estimating the learned dimensionality from the singular values against the Marchenko-Pastur edge, with the position embeddings adding some extra dimensions. That puts the vocabulary somewhere around 40 to 120 tokens, which fits ASCII-style English better than code. The weak induction behavior in layers 1 and 2 is worth noting too, and I'm starting to look at the layer-0 head that attends to delimiters.

I'm checking whether the head at 0.3 acts as a position-0 sink, since its output would then just be the first token's value.

I'm considering whether the layer-0 MLP neurons act as single-token detectors, which would let me cluster their weight rows in the token subspace and count the clusters to estimate the vocabulary size.





---

Still weighing whether the neurons form clear clusters, and it looks like they're spread out. I'm wondering if the embedding structure could be read from the layer-norm statistics, though the ln1 weight of block 0 doesn't seem to help. Maybe the input is one structured kind of text, like code, markup, or arithmetic, where the attention heads would attend strongly to the start.

Now I'm checking whether the induction behavior fits a structured kind of text like code, since its heavy repetition would make induction show up early and strongly, unlike the late, weak induction I see in layer 3. I'm also wondering if the layer-3 output spectrum implies a vocabulary size near 65, and I'll look at the OV singular values of head 3.3 to test that.



---

I'm reading off the OV plateau and knee near indices 27–30, which suggests about 28–30 effective dimensions for the copying head, and I'm wondering whether that points to a vocabulary of roughly 30 characters.

The projection spectrum points the same way: the top dimensions fall off steeply around index 26, which would match a 27-character vocabulary such as lowercase letters plus space. The ratio test gives a weak, nearly smooth drop there, though, so it's only suggestive. I'll find a more precise knee for the OV plot before settling on a number.



---

I'm estimating the effective dimensions of the OV circuit: the singular values stay high for about 20 entries before dropping off, so roughly 20 to 30 dimensions matter. Rarer characters should get smaller singular values, and a larger vocabulary, like Shakespeare with uppercase and punctuation, might give the copying head more dimensions even if the rare ones stay small.

I'm checking whether Shakespeare's character set would push the copy head toward around 40 dimensions. I also want to look at the spectrum of the layer-0 previous-token head's OV and the knee of the duplicate-token QK eigenvalues near 45.



---

I'm checking the layer-0 OV singular-value knees, which sit around indices 26 to 30 for these heads, so the heads that copy token identity should have an effective rank near 28 to 30. That means they are reading the raw token embeddings.

The token-embedding estimate is landing around 29–31 dimensions, which would point to a small character vocabulary, something like lowercase letters plus a space. I'm weighing text8's 27 characters as a candidate, though I'm not treating that as settled. The earlier larger subspace near 120 could just reflect positional dimensions that fc0 also reads, so I'm leaving that open.

I'm weighing which text would have roughly 30 characters, with text8's 27 as one candidate and a few non-English alphabets like Russian or Hebrew also fitting the count. Telling them apart from the singular-value spectrum alone looks hard, so I'd probably need character frequency patterns to narrow it down.

I'm now checking the Wv spectrum across all six layer-0 heads, using the 64x384 matrices that read the embeddings, to separate the learned token subspace from the noise floor left by initialization.



---

The bottom singular values of Wv are nearly zero, which suggests a near-null space. Weight decay could be pushing unused directions toward zero after long training. I'll count how many directions are non-negligible and print the full spectrum with thresholds.



---

I'm wondering whether the near-zero singular values in Vall come from the layer 0 ln1 gamma, so I'll check the minimum gamma values to see if any dimensions are close to zero.



---

Gamma isn't zero, so the combined V matrix for all six heads likely has around ten singular values that are close to zero. I want to see how many fall below something like 0.05, and I'll print the tail from index 200 to 384 to find where they collapse.



---

The Wv weights look like a plain random-matrix spread, so this smooth drop toward zero isn't a sharp signal and tells me nothing new. The stronger hint at a vocab size near 30 comes from the OV knees in layer-0 heads and in 3.3, but those knees might just reflect what each head chooses to move rather than the vocabulary itself.

I'd expect head 0.0 to move the full token identity, and a few other heads show knees around 26 to 30 as well. To check that more carefully, I'll look at where the ratio s[i]/s[i+1] peaks for heads 0.1 and 0.5.



---

I'm checking where the 0.1 and 0.5 knees fall in the dimension curve, which points to roughly 29 strong dimensions, so the effective vocabulary is around 29-30 frequently used characters. I'm wondering whether the embedding's effective dimensionality ends up below the vocabulary size because the rare characters contribute little.

I'm weighing whether the vocabulary is closer to 30-40 characters, since rarer characters would contribute little to the effective dimensionality. Tiny Shakespeare has 65 characters but only about 40 are frequent, and its uppercase letters would add some dimensions. I'm also considering whether a text with around 27-29 characters might fit, though picking a language among similar alphabets is hard, and I'm wondering if the spectrum of the OV weights tracks token frequency at all.

I'm weighing whether a known-structure trick could recover the unigram distribution, like a bias term in the final layers, but without the unembedding matrix that doesn't map cleanly to tokens. I'm also considering whether the token embeddings could be recovered from layer-0 structure or from the same-token head's QK form, though weight matrices are linear maps rather than individual token points, so it's unlikely to work. Given these limits, I'm settling on a small-alphabet, English-like text as the working guess.

I'm settling on English with lowercase letters, space, and basic punctuation as the most likely guess, though Spanish or Finnish are possible. I can rule out a very small alphabet like DNA or binary, and arithmetic too, because the roughly 29 effective dimensions point to a richer character set. Python code would show more spread across both cases, digits, and punctuation, so natural language seems more likely, and I can't reliably tell the languages apart from word-length structure.

I'm checking whether the proj3 spectrum agrees with a vocabulary of about 27 characters, since its sharpest drop sits near index 26, which would match a 26-dimensional logit space.

I'm now looking at the vocabulary estimate from the spectrum's knee, which keeps pointing to around 27 to 30 characters. The lowercase-English-plus-space case, like text8's 27 characters, fits that well, though other alphabets with accents or similar sizes would also land near here.

Next I'm going through the layer-0 heads one by one, starting with 0.0, which looks like a previous-token head that shifts attention back by one position. Head 0.1 seems to be a local head that mixes a two-back positional offset with content, and its OV circuit copies token identity across roughly 29 dimensions. Head 0.2 looks like a same-character head that attends to earlier occurrences of the current token, and 0.3 appears to be a broad, low-rank averaging head.

I'm now checking heads 0.4 and 0.5 for positional structure: 0.4 looks content-driven with one dominant QK direction, while 0.5's positional angles are small compared with 0.0, so its attention depends mostly on content. I want to measure how much of each head's QK sits in the token subspace versus the positional one to confirm this.

I'm reading the same-token and noise-level scores for the layer-0 heads, then moving on to check whether the layer 1 and 2 heads read positional structure, and how much bias and same-token alignment each one carries.



---

Layers 1 and up have large top QK singular values, around 150 to 700, which points to one dominant direction. Head 1.4 stands out at 731, with a key bias of 19.6, so it looks like a strongly feature-attending head that reads the output of 0.3 on the key side. Next I want to know what the query side of 1.4's top singular vector looks like, and which earlier layer writes the top QK singular vectors for layers 1 through 3.

I'm wondering whether the big top direction shared by layers 1 and 2 is a position-like feature that layer-0 heads or the MLP re-encode from head 0.3's uniform averaging, which would let those heads attend by recency. To test that, I'd project each head's top query and key singular vectors onto the output subspaces of the earlier OV circuits and the layer-0 MLP.



---

Most heads in layers 1-3 share a dominant query-key direction, with its key side coming from head 0.3's output. I'm wondering what that shared feature encodes. Since 0.3 attends broadly and writes a low-rank signal, it may be acting as a positional proxy, with the uniform average over the prefix producing a signal whose magn

I'm checking whether 0.3's attention is really driven by the query-independent key bias, since the QK weights are small. If so, its pattern is a mild, token-dependent preference that works like a weighted bag over the context, and I'll look at what its low-rank output carries.

I'm looking at how 0.3's output is averaged over the prefix, so the key reading at each position changes slowly except near the start. That would make early positions stand out, pointing toward an attention sink at the beginning of the context. I'm still checking whether the rank-1 sign works out for that.

I'm checking whether layer 1/2 heads have any second-level direction that reads prior-token info from 0.0, and if one of them is itself a previous-token head. Otherwise I'll treat those heads as content-based mixers that default to the start-of-context sink.

I'm checking whether 0.0 is a previous-token head or self-attending: its score asymmetry is large, which points to a directional offset rather than a symmetric self-match. Comparing the angles across heads, 0.0 and 0.1 look like they might sit at different offsets, though I haven't confirmed which.

I'm checking whether 0.0's score asymmetry reflects a directional offset of one position, which would make it a previous-token head. The 3.3 induction-like match then reads from 0.0 as its key source, and 0.1 may also sit one or two positions back.

I'm now checking whether heads 3.0 and 3.2 also do induction copying, or whether only 3.3 does, since its diagonal score is much stronger. I'd need to see whether their output projections map tokens back onto themselves to tell partial copiers from the main one.

I'm checking whether heads 3.0 and 3.2 copy the way 3.3 does, since their top key-side directions come from the sink-like head 0.3 rather than from the previous-token source that feeds 3.3. I'd still need a way to tell partial copiers from the main induction head.

I'm going to run a randomized baseline on the diagonal statistic, rotating the attended-side basis within the token subspace, to see whether 3.0 and 3.2 really sit above chance or whether only 3.3 stands out.



---

Three heads, 3.0, 3.2 and 3.3, show a strong attention pattern and look like induction-style QK heads, with 3.3 the clearest; 3.4 is weak and layers 1 and 2 don't show it.

But I'm wondering whether the query-current-token, key-previous-token pattern could also be a previous-token head in

I'm checking whether layer 3's OV outputs for heads 3.0, 3.2 and 3.3 are consistent with copying, but the direct bigram path isn't available, so I'll skip that. I'm also weighing whether the sharp knee in the OV spectrum near 29 dimensions points to a lowercase English alphabet with a space and a few punctuation marks, tentatively calling it English at about 0.4 confidence.

I'm checking whether the OV spectrum's soft knee at roughly 28 to 30 dimensions fits a 27-symbol alphabet, since the token span should have rank at most 27 and the values past that point still look sizeable. That makes me think the vocabulary might be around 30 to 40 symbols, perhaps lowercase letters plus punctuation and newline, though another language is still possible.

I'm weighing whether the vocabulary could be something other than English, since the weights alone don't reveal the language. Alphabet size around 30 fits lowercase English with punctuation, but Spanish, Italian, or Russian would also fit. The next step I'm considering is word-length and space-frequency statistics, since they might distinguish them.

I'm estimating the vocabulary size more precisely by stacking the layer-0 value matrices for heads 0.0, 0.1 and 0.5 and looking for a knee in the combined singular value spectrum. The individual spectra were smooth, which may be because heads 0.3 and 0.4 differ, so this could sharpen the picture. I'm also noting that the positional head 0.1 attends about two tokens back, which says nothing about language.

Running the call now.



---

I'm estimating the effective dimensionality at roughly 30 to 45, so the vocabulary could be around 35 to 50 characters once rarer ones are counted. That weakens the text8 idea a bit. Tiny Shakespeare has 65 characters, but the top ~40 cover over 99% of the text, so its vocabulary could also be near 65, and that's hard to pin down. I'm wondering what else could tell these apart, perhaps a single kind of text

I'm checking whether the offset-0.0 hypothesis holds by comparing it against 0.1 across the three planes, where a ratio near 2 would fit. I'm also noting that 0.4 seems content-based, so I'll look at what its early-layer attention tracks next.

I'm checking the layer-3 heads next: 3.4 and 3.5 look content-based, with large but low-rank OV outputs, and 3.3 has the biggest OV. I also want to confirm the layer-2 heads aren't reading layer-1 output in a shifted, induction-like way, and they seem to be near zero or negative there.

I'm now labeling the heads in layers 0 through 3: 0.1 looks positional, attending about two tokens back, with a secondary previous-token role. 0.2 handles duplicate characters, and 0.3 is a broad averaging head whose output most later heads read as a global feature. 3.3 is the induction candidate.

I'm now checking which heads should count as induction heads. Beyond 3.3, 3.0 and 3.2 have signal that seems meaningful, so I'm leaning toward including them, though I'm unsure whether that's too generous.

I'm deciding to list only 0.0 as the previous-token head, since 0.1 is probably mostly looking at tokens a bit further back even though it has some previous-token pull. I'll mention 0.1 as a possible secondary case and keep the answer in English at about 0.35 confidence.

