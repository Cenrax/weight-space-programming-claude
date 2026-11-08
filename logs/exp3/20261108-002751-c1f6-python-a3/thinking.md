I'm working through this without any forward passes, so the analysis has to come from the weights alone. I'm looking at whether the positional subspace might be recoverable from the attention weights, since without embeddings it's hard to pin down which tokens are which.

I'm weighing whether the vocabulary size or the embedding rank could be inferred from the layer-0 weights, though trained matrices are usually full rank, so that may not work. I'll just look at the weights directly.



---

I'm working out the attention heads now: for each one, the effective query-key bilinear form with the layer norm weights folded in, its eigenvalue spectrum, and the positive eigenvalues of the OV circuit as a copying signal. Induction heads would be ones in layer 1 or later that compose with a previous-token head earlier on, and I'm still trying to pin down how to spot the previous-token heads without positional embeddings, with the query bias times W_K as one possible route.

I'm checking which heads might be previous-token heads, since without positional embeddings that's hard to pin down. I'm considering the query bias times W_K as a key preference direction, and whether a low-rank positional subspace in layer 0's QK matrix could reveal them.

I'm looking at whether the layer 0 weights might reveal the vocabulary size, since token embeddings span at most V dimensions and positional embeddings at most 256. Updates to weights reading the residual stream stay in the span of the input activations, so the row space of those weights may concentrate in the embedding-plus-positional subspace, though that could exceed the model dimension. I'm also wondering whether the positional embeddings are effectively low-rank.

I'm checking whether the layer 0 weights reveal anything about the setup, since bias weights with no gradient seem to decay toward zero under weight decay. That could mean the singular value spectrum of the stacked input-side weights shows a drop at the dimension of the input span. I'm also noting that layer norm removes one direction, so the normalized inputs span one fewer dimension than expected.

I'm weighing whether the vocabulary size could be inferred from the embedding and positional subspace dimensions, since together they would span at most about 321 dimensions, under the model dimension of 384. I'm also looking at the layer 0 LayerNorm gain, which may have decayed toward zero in dimensions that never receive gradient, and checking its distribution.

I'm checking whether the dimensionality of the layer 0 input-side weights, or the vocabulary size, can be read off the singular value spectrum.



---

No clear rank drop in the spectra, which are smooth. Weight decay may not be strong enough to show up here. I'm also looking at the MLP fc spectrum, which shows no gap either. Recovering language from middle-layer weights looks very hard, so I'm wondering whether the dataset is something unusual, like DNA sequences or music notation, where a tiny vocabulary would make it easier.

I'm considering whether the vocabulary might be tiny, which would make the embedding span small and let positional structure dominate, though the smooth spectra don't clearly support that. Another route is the attention patterns, checking whether some heads fixate on fixed offsets. The more promising check seems to be the final block's MLP projection: its outputs should mostly line up with the unembedding, so a gap near the vocabulary size would show up in its singular spectrum.



---

Layer 3's mlp.proj shows a steep drop, falling from around 40 to about 100. I want to see the full spectrum to pin down where the vocab size gap is.



---

I'm sizing up the spectrum: the decline is gradual with no clear cutoff, so the effective dimension might sit around 80 to 90, though that's not conclusive. I'm guessing the vocabulary could be somewhere near 80 to 100. Next I'm wondering whether the last layer's MLP output directions could reveal the unembedding, but I don't think that's directly possible. I'm also considering whether the residual stream has some fixed, interpretable basis, though that seems unlikely. Then I'm turning to what "one kind of text" might mean here.

I'm weighing candidate sources for the text: English, Python, French, German, Chinese, Russian, DNA, and so on. A large character set like Chinese would need a much bigger output space, but layer 3's projection spectrum drops off after about 100 dimensions, so the vocabulary likely sits under 100 characters. Checking the tail values against layer 2 supports that, and I'm now trying to narrow it using the elbow location.

I'm now combining the layer 3 MLP projection with the attention output columns, looking for a shared subspace where the singular values drop off sharply. Since only the unembedding reads from that final layer, the output space should be roughly the vocabulary's rank, so that combined spectrum could give a tighter vocabulary estimate. The tied-embedding question stays open, since this custom model doesn't match the nanoGPT layout I'd assumed.

I'm looking at where the layer 3 singular values start dropping off, around indices 75 to 90, which hints the vocabulary could be near 80 to 90. Since the final layernorm removes one direction and rare characters contribute little gradient, that elbow may understate the true vocabulary size.

I'm estimating the noise floor from the layer-3 projection tail: a random 384×1536 matrix at std 0.02 would have singular values around 0.4 to 1.2, and the observed tail sits lower, so the init has decayed. The signal directions should be the ones standing clearly above that bulk, and counting them is the next step.

I'm weighing whether the output-side singular values could pin down the vocabulary more tightly, since they're roughly the rank of the output space. Counting the values above the noise bulk near index 119 would overshoot, though, because the bulk itself is perturbed, so I'm not treating that as precise. Instead I'm considering whether the structure of the character directions, rather than their count, might give a clue about what kind of text this is.

I'm considering whether the layer 3 MLP output directions could be combined with the layer 0 inputs, since if the embedding is tied to the unembedding, those two subspaces would coincide. That would let me express the layer 0 attention patterns within the token subspace, though I'm not sure that's tractable.

I'm weighing whether the positional structure of the attention heads might separate code from prose, since indentation and bracket matching would show up as distinctive patterns. I'll check the head analysis first, then return to the vocabulary question.

I'm checking the query-key matrix of each head in layer 0 to tell positional heads from content heads. A previous-token head should show a large antisymmetric part, since its positional rotation makes the form non-symmetric, while a same-token head should be mostly symmetric and positive. I'll compute the symmetric and antisymmetric shares for every head.

I'm weighing whether the attention heads' symmetric and antisymmetric parts could tell previous-token heads apart from content-matching ones, though the causal mask complicates that. The harder part is that without the positional embeddings, I can't see the shift directly, so I'm looking toward how the heads compose in later layers instead.

I'm working out the K-composition score for the induction head against the previous-token head, using the layer norm gains to adjust the query weights. I also want a copying score from the OV eigenvalues, though without the embedding and unembedding matrices I can only check whether the residual basis is shared.

Next I'm checking the query-side bias in layer 0, since its large size might mean the keys get a content-independent preference, possibly tied to a recency direction in the positional part.

I'm settling on the next step: computing a set of metrics to see what the attention heads are doing.



---

I'm noticing the OV eigenvalues are strongly negative in layers 1 to 3 in residual space, which is a fairly common pattern where heads suppress or erase information. Layer 0's QK magnitudes are small, roughly 0.1 to 0.43, probably because the ln1 gain of 0.6 shrinks the weights there. I also want to keep in mind that since LayerNorm normalizes its input, the scale of the residual stream affects what reaches QK.

I'm working out the K-composition score for a later head against an earlier one, using the Frobenius norm of QK2 times OV1 normalized by the two matrix norms. I'll compare that to a random-matrix baseline, and also set up the Q-composition and V-composition checks.



---

I'm checking how head 0.2's output feeds into the layer 1 and layer 2 heads through K-composition, and it looks strongest into 1.0 and 1.2. I'm also noting that 1.1 K-composes into several layer 2 heads, and that the Q-composition from 0.2 into 1.0 is around 0.22, so it's acting on both queries and keys.

I'm weighing whether 0.2's composition is inflated by a shared constant direction, so I'm planning to strip out the mean and check the singular spectrum of its OV output. I'm also wondering whether 0.2 might be a previous-token head, which would mean finding a positional subspace in layer 0 to test that.

I'm now thinking about how to tell real composition apart from noise: the A/S ratio for 0.3 is lowest at 0.68, while the other heads sit near 0.95, which is about what random matrices give. So that measure alone doesn't separate the heads well. I'd rather look at the singular value spectra of QK and OV for each head, since a previous-token head in layer 0 should show a low-rank positional QK and a high-rank OV.



---

I'm looking at head 0.2: its QK circuit has an effective rank of about 2.7, mostly along one direction, while its OV circuit has rank around 7.3 with two dominant directions. That suggests it might be a delimiter-style head that attends to a specific token feature, like a newline or space, and its OV output is

I'm checking whether the positional subspace can be recovered from the layer-0 heads' QK structure, since the positional embedding itself isn't available. The idea is to compare the top singular directions on the query and key sides for each head and see where they overlap, which would mark a positional head.

I'm working out how the rotation idea would show up in layer-0 heads: a shift-attending head would have QK acting as a rotation within each frequency plane, with the peak of the score landing at the previous position, so the strongest prev-token heads should carry a large antisymmetric part. Next I want to eigendecompose each head's QK to check for that structure.

I'm now looking at whether the top query-side singular direction of head 0.2 lines up with a positional subspace, and I'm weighing whether the eigenvalue structure of its QK matrix could show a rotation pattern that marks a previous-token head.

I'm checking whether the query bias dominates the score for head 0.2: if the query is mostly constant, the head would just prefer keys along one fixed direction. To test that, I'm comparing the norm of the bias term, Wk^T bq divided by 8, against the typical size of the query contribution from the normalized input.

I'm going to check the ratio between the query-bias contribution and the typical content-dependent query contribution.



---

The score scale checks out: with norms near 19.6, a top singular value of 0.09 can push attention scores to around ±35, so every head matters. Next I need to pin down the positional subspace, and I'm wondering whether the layer 0 residual stream's ln1 output can reveal it.

I'm checking whether the layer-0 QK matrix's top singular directions line up with each other and with the bias-key preference. Prev-token heads usually need many high-frequency positional dimensions, and an effective rank near 2.7 doesn't fit that, so I'm weighing whether the 0.2 head is doing something else. I'll look at where its OV output goes in later layers.



---

I'm guessing head 0.3 is a same-token or positional proximity head, since its query and key subspaces overlap heavily and are nearly symmetric. Its flat spectrum with effective rank 27 fits either idea, and I'm still weighing whether it attends to nearby positions or matches identical characters.

I'm weighing whether the positional direction could be found by checking the symmetric part of head 0.3's QK matrix for positive semidefiniteness. For a prev-token head I'd expect P_i^T M P_{i-1} to be large, but I haven't pinned down the positional subspace yet. I'm also wondering if 0.2 could be the previous-token head after all, since its query looks nearly constant and the bias term dominates, though a constant query shouldn't prefer position i-1 for every i.

Wondering whether 0.2 could instead attend to a specific content token like a space or newline, combined with a positional term, so it finds the most recent delimiter rather than the previous token. But with a constant query, an absolute positional term that rises with j would make the head favor the current position itself, which doesn't fit a previous-token head. I'd need the positional embeddings to check this, and I can't reconstruct them from the weights alone.

I'm looking at head 0.2's output, which has effective rank about two, so it may be writing a two-dimensional signal like a phase or a positional encoding. I want to check whether layer-1 head 1.0's top query-key subspace lines up with that output plane, which would suggest the two could combine into relative-position attention.

I'm projecting head 1.0's query-key matrix onto that two-dimensional output plane of 0.2, checking whether its query side and key side both read the same pair of directions.



---

I'm reading the output of head 0.2 as it feeds into head 1.0, and both the query and key sides come out positively aligned, so 1.0 seems to be attending to positions where 0.2's signal matches the query's. If 0.2 tracks something like position within a line, then 1.0 would be acting as a "same column" head, which is a pattern you'd expect in structured text like poetry or code.

I'm checking the layer-0 positional structure next, since the attention patterns alone don't reveal it. The plan is to compute the eigenvalues of each layer-0 head's QK matrix and look at the angles of the large ones: uniform angles would suggest noise, while rotations at specific frequencies would point to sinusoidal position encodings like those common in small GPTs.



---

I'm noticing that three of the rotation components, 0.3, 0.4, and 0.5, have large eigenvalues, each with its own pattern of angles. The small angles on 0.3 look like local relative attention to nearby positions, while 0.4 has large rotation angles that point to high-frequency content. The angles on 0.5 seem to follow a roughly 25° spacing, which looks like a harmonic series.

Looking at the 0.5 angles, if they're roughly 25° apart then the pattern could be harmonic, and for a shift Δ each frequency would give an angle of f_k·Δ.

I'm checking whether heads 0.4 and 0.5 share the same eigenvector planes, so I'll take the top four complex pairs from 0.5 and project the other heads' QK matrices onto those planes to read off a per-plane angle for each head.

Now I'm pinning down how to read each head's rotation angle: take the 2D plane's matrix, and the angle comes from the difference of its off-diagonal entries over the sum of its diagonal ones.



---

I've found that heads 0.3, 0.4, and 0.5 all share planes, which points to them being positional heads. Checking the angles across the three heads confirms the pattern, and I'm now comparing the plane angles for head 0.3 against those for head 0.5.

I'm comparing the angle ratios between heads 0.3 and 0.5 across their shared planes, and the values drift from roughly 2.8 to 4.5. I also need to check sign conventions, since the plane orientation could flip the sign of an angle, though the signs match across both heads.

I'm weighing whether 0.5 is attending to Δ=1 or Δ=2, since the evenly spaced angles around 25° could fit either, and I'm noting that head 0.4's magnitudes look much smaller than 0.5's.

I'm checking whether heads 0.4 and 0.5 are both attending one step back, since their top planes sit at high angles like 117 to 168 degrees. If they are, the angle should roughly match the frequency, and head 0.3 would be mixing the current and previous tokens instead.

I'm checking whether 0.4 and 0.5 really match each other, since 0.4's magnitudes are much weaker, so the comparison may be noisy. Their angles line up closely in the shared planes, which suggests the same offset, though it's still open whether that offset is Δ=1 or Δ=2.

I'm weighing whether the 0.4 angle near 168° means a frequency close to π for Δ=1, which would give sharp previous-token discrimination, or whether Δ=2 could also fit. Next I want to check layer-1 heads, since their weights in these planes are tiny, though 1.2 through 1.5 look larger.

I'm checking whether the layer-1 and layer-2 heads act on the previous-token heads' outputs, so I'm testing the eigenvalues of each QK matrix multiplied by the OV matrix of a layer-0 head to see if they come out positive.

Next I'm checking whether the top-layer-0 head 0.2's low-rank output is what drives the composition, so I'll compute how positive the eigenvalues of each layer-2 QK matrix multiplied by a layer-0 OV matrix come out.



---

I'm checking whether layer 3 heads 3.0, 3.1, 3.4 and 3.5 are reading from head 2.0's output, with high key-side positivity scores around 0.7 to 0.98. That raises the possibility that 2.0 is a previous-token head and those layer 3 heads are induction heads.

I'm now checking whether the key-side scores for 2.0 line up with a previous-token role, meaning its output would let the layer 3 heads match on the token before each position. Head 2.0 also has an unusually spread-out QK structure, so I want to see how much it relies on the positional part of the residual. I'll run the same positional-plane check across heads in layers 1 to 3.



---

I'm looking at where positional information is used, which seems concentrated in layer 0, heads 0.3, 0.4 and 0.5, with later layers showing weaker positional use. I suspect 0.4 and 0.5 attend to the previous token, while 0.3 looks more like a local head that mixes the current and previous positions.

I'm checking the 0.3 planes, where its angles run about a fifth to a third of 0.5's, which puts its peak near Δ≈0.3. Since self is Δ=0, that suggests a broad local head mixing the current and recent tokens. I'm still weighing whether 0.5 is Δ=1 or Δ=2, since the 0.4 and 0.5 angles match closely and the highest angle seen, 168°, fits Δ=1.

I'm weighing whether the 0.4 and 0.5 heads are at Δ=1 or Δ=2, leaning toward Δ=1 since previous-token heads are the usual pattern in early layers of character models. Checking the OV circuits, 0.5 carries a high-rank OV, which fits a previous-token head that passes along token identity.

I'm checking whether 3.1, 3.4 and 3.5 match queries against keys from 2.0, 1.5 and 0.3, since the K-positivity values there are high. That pattern looks like an induction-style signature, but it could also just be a head attending to the current token's own content if 2.0 copies it, so I need to know whether 2.0 is a previous-token head.

I'm wondering whether 2.0 is itself a previous-token head, since its QK positivity is moderate and its positional plane use is weak. Later layers have no obvious source of position beyond what's left in the residual stream after layer 0, which may survive even though those planes don't show it clearly. Another possibility is that 2.0 just attends to keys similar to the query, which would make it a content-matching head rather than a positional one.

I'm considering whether a layer-1 head could work as an induction head, but its key-side composition with the layer-0 previous-token heads looks close to baseline. That makes me wonder whether the layer-0 outputs are simply smaller in scale, or whether the induction pattern has to come from somewhere else. I'll look at what the OV circuits of 0.4 and 0.5 contribute to later layers.

I'm now testing whether the output subspace of 0.4 and 0.5 overlaps with what the later heads read on their key side. I'll measure, for each later head, how much of its key-side reading falls inside the top 20 output directions of those source heads, and compare that against the roughly 5% expected by chance.



---

I'm noticing the 0.4 and 0.5 prev-token outputs are barely read by the K-sides, around 1x, which seems odd. The only strong K-reads come from 0.2, 0.0, and 1.1. That makes me wonder whether 0.2 is actually the real previous-token head and my labeling of 0.4 and 0.5 is off.

I'm wondering whether the 0.2 head's output directions are just common high-magnitude features that every head reads, which would confound the subspace reading metric. Its key preference looks content-independent, so for a fixed query it might favor recent positions and act as a soft self-attention. I want to check whether that key-preference vector lines up with the positional planes I identified for 0.3, 0.4, and 0.5.



---

The positional subspace accounts for about 60% of the QK mass in heads 0.3, 0.4, and 0.5, yet their OV circuits read from it less than chance, so they seem to move token content rather than position. For 0.2, the key preference is nearly half positional, so it may be partly attending to the start of the sequence.

I'm thinking about the 0.2 head as an attention-sink or fixed-position head, likely attending to the first token, with a low-rank output that encodes position. Its output seems to carry a coarse positional signal rather than token content. I'm also wondering whether the induction mechanism for 0.4 and 0.5 works through MLPs or some other route, since their outputs don't seem to be read on the key side by any later head.

I'm checking whether the induction signal could reach the later heads through MLPs or Q-composition rather than through the key side. Since 0.4 and 0.5 show no positive K-composition with any later head, I'm looking at the positivity table to see whether layer-2 or layer-3 heads pick up a previous-token signal some other way.

I'm checking whether 2.0 reads its key-side input from 0.3, 1.2, or 1.4, since those positivity values are all negative. I'm also wondering whether an MLP-based embedding could carry the token identity that a plain composition analysis would miss.

I'm now trying a different measure: restricting the positivity check to the token-content subspace, so that the query side picks up the current token and the key side picks up the previous token from the earlier head's output, rather than letting unrelated residual directions dominate the full matrix.

I'm restricting the check to the token-content directions that each earlier head reads, so I'm projecting onto those and testing positivity for the later heads against heads 0.3, 0.4, 0.5, 0.0 and 0.1.



---

I'm checking whether heads 3.0, 3.1, 3.4 and 3.5 are matching the current token against head 0.3's output, which mixes the self and previous token. If 0.3 mostly attends to the previous token, those heads could be induction heads, and the attention spread of 0.3 looks around 1 if the spread of 0.5 is 2 or 3.

I'm trying to pin down the peak position from the plane angles, since the ratios between heads 0.3 and 0.5 come out inconsistent. If 0.3 had a single Δ, the angles would scale cleanly, but they don't, so I'm weighing whether 0.3 is centered near Δ≈1/3, spreading between self and the previous token.

I'm weighing whether 0.3 is really a Δ=1 head or a broader one. If 0.5 sits at Δ=3, 0.3 would be prev-token, but the ratios between 0.3 and 0.5 aren't consistent enough to settle it. I'm now checking whether the high-frequency components in 0.4's planes would be needed to tell Δ=1 apart from its neighbors.

I'm noticing that in the higher-frequency planes, head 0.3's angles roughly match head 0.4's (around 123 to 175 degrees), rather than being a third or a quarter of them as the low-frequency planes suggest. I still need to check the signs before deciding which pattern holds.

I'm checking the signs in the high-frequency planes, where head 0.3's angles mirror head 0.4's with the opposite sign, so 0.3 would be rotating the other way. If f is around 120 degrees, a negated angle lands on 2f, which would suggest Δ=2 rather than Δ=1.

I'm now looking at whether the high-frequency planes could pin down the offset, since head 0.3's angles there mirror head 0.4's with the opposite sign.

I'm considering whether heads 3.x could be induction heads that copy from 0.3's output, but that depends on whether 0.3 attends to itself or to the previous token. If it attends to itself, the 3.x heads would just be matching duplicate tokens instead.

I'm checking whether the negative OV values across the layer 1–3 heads could reflect tied embeddings, where the output would need to copy the attended token. If so, an induction-style copy would show up as a positive value, and the uniformly negative numbers don't fit that.

I'm now checking the eigenvalue spectrum of the 3.1 OV circuit, since a few large negative eigenvalues could be driving the uniformly negative sign.



---

I'm looking at layers 1-3, where the OV eigenvalues in residual space come out strongly negative. My guess is that these heads read token-like features and write their negation, maybe to erase something read from earlier layers, or to cancel the token's own features at the attended position. It could also be copy suppression, or a sign convention interacting with layer norm.

I'm checking whether the embedding and unembedding are tied, and whether the layer 3 output subspace lines up with the token subspace that layer 0's prev-token heads read from.



---

I'm going back over the idea that nothing lines up between the input token subspace and the output space, whether because the embeddings are untied or the alignment just isn't detectable. Then I'm wondering whether the language signal might come from periodic structure: the positional-like planes with evenly spaced frequencies, around 25° apart, suggest a period of about 14.4 characters. Learned positional embeddings with harmonic frequencies would make sense if the training data has regularity in absolute position, like fixed-width lines, especially if each sample starts at a random offset.

I'm checking whether the periodic structure in the positional planes could be a language signal, and I'm skeptical that it is diagnostic.



---

I'm working out the vocabulary size from the layer 3 MLP ratio, which elbows around 75-85, so the effective output directions sit near 80-100 characters. Rarely-used characters contribute little, so the true vocabulary could be a bit above 85. I'm also comparing typical vocab sizes: tiny Shakespeare is around 65, enwik8 is 200+ bytes, and Python code is around 95.

I'm weighing the vocabulary estimate against typical sizes for different kinds of text: English prose, German, French, and Russian with its Cyrillic alphabet all land somewhere in the 80-150 range. If the vocabulary were as small as Shakespeare's 65, the transition would have to sit lower, around 57-65, though the ratio only falls off gradually from early on.

I'm thinking about whether the spacing pattern in code could reveal anything about the character distribution, though it doesn't look identifiable from this. Head 0.2 seems to be the key piece: its key preference is about half positional and half content-based, so it may be tracking recent delimiters like newlines. Head 1.0 could then match positions against what 0.2 writes, which would make this a two-head pattern for finding line structure.

I'm trying to pin down the vocabulary size from the layer 0 input subspace, reasoning that the directions it reads from are the token embeddings plus the position embeddings, so the total dimensionality is bounded by the vocabulary size plus the effective rank of the positional part, which I'd guess is around 33 to 60.

I'm comparing the layer 0 read matrix, which stacks the q, k, v, and fc inputs with the LN gain, against the layer 1 read spectrum, to see whether layer 0 shows a sharp cliff. Its singular values looked smooth earlier, tapering toward 0.2, so I'm checking the ratio between the two spectra for any drop.



---

I'm working out the layer-0 MLP read: the transition runs from about 70 to 140 and plateaus after roughly 150, which puts the effective input span around 120 to 150 dimensions. With positions taking perhaps 30 to 60 of those, V comes out near 80 to 100, matching my earlier estimate of about 85. Next I want to check whether that transition really reflects tokens plus positions, by measuring how much the layer-0 MLP read subspace overlaps with the positional subspace.

I'm weighing whether the value estimate of 80-100 points toward English prose or code, and I lean toward prose over Shakespeare. I'm checking the positional frequencies next, since the multiples of about 25 degrees could indicate a periodic positional embedding.

I'm checking whether the positional frequencies match a DFT over the 256-token context, which would put them at about 1.4° per step, but the spacing of roughly 25° doesn't fit that unless only every eighteenth frequency is used. That makes me wonder whether the positions are genuinely periodic with a period near 14, perhaps from fixed-width structure in the data, or whether the eigenangle analysis itself is distorting things because the positional basis isn't orthonormal.

I'm checking whether the eigen-angles here are diagnostic of periodicity, and I think they aren't, since a smooth positional subspace would give evenly spaced angles too. So I'll keep 0.4 and 0.5 as previous-token heads, with that caveat. Next I want to test whether the near-180° real eigenvalue in head 0.4 implies an alternating pattern that favors odd offsets.

I'm checking whether head 0.4's near-180° real eigenvalue means it favors odd offsets, and concluding it most likely attends to Δ=1, with head 0.5 matching that. Head 0.3 looks more like a mix of self and previous-token attention, and I'm now wondering what its opposite-signed high-frequency planes imply.

I'm now treating the positional frequencies as an assumption to test: I'll take them from the 0.4 angles under the Δ=1 reading, and score each head across offsets 0 to 10 by summing the planes' cosine contributions. That's a weight-based approximation rather than an actual forward pass, so I'll use the union of planes from heads 0.5 and 0.4 and check whether head 0.3 leans toward self-attention.

I'm settling on how to score each head across offsets, using the frequencies drawn from the 0.4 and 0.5 angles.



---

The first attention heads at Δ=1 look like previous-token heads, and the 0.3 head mixes current-token and previous-token information. The overlapping positional planes make these readings rough. Next I want to check induction heads, which would need K-composition with the outputs of the 0.4 and 0.5 heads.

I'm checking whether 3.0, 3.1, 3.4 and 3.5 are induction heads or same-token heads. Their K-positivity with the 0.4 and 0.5 outputs is weak, but with the 0.3 output it's quite strong. I'm also looking at what head 2.0 is doing, since it's strongly K-positive with the 3.x heads.

I'm checking whether layer-2 heads could be induction heads through K-composition with layer-1 previous-token heads, since the direct layer-0 route gives only weak scores (about 0.15 and 0.12 for heads 1.0 and 1.5).

I'm checking whether layer-2 head 2.0 could be composing through the layer-1 heads. Its key-side scores against the layer-1 outputs come out negative, which I'd normally read as anti-matching. But the layer 1–3 OV circuits have negative eigenvalues, so that sign flips: a negative score would mean the head is matching what the layer-1 head read, not what it wrote.

Negative eigenvalues on the layer 1–3 OV circuits would flip the usual reading of these scores, so a negative result could mean matching what the layer-1 head read rather than what it wrote. That makes the sign of 2.0's composition hard to interpret directly.

I'm weighing the evidence on 3.0, 3.1, 3.4 and 3.5: their query-key matching is strong and high-rank, which fits induction heads matching on full token identity more than a positional pattern. That reading depends on the key position's residual carrying previous-token information, which I haven't confirmed.

I'm going to compare the query-key matching for these heads against two baselines: same-token matching on the raw token subspace, and a previous-token copy through the 0.4 and 0.5 OV circuits. I'm also considering whether the query side might read the earlier heads' outputs to match on bigram context.

I'm now going to compute, for each layer-2 and layer-3 head, two scores: the direct same-token match, and the induction-style match through the 0.3, 0.4, and 0.5 OV circuits, with norms relative to each other.



---

I'm working through the query side of layer-1 heads 1.2 through 1.5, plus 2.0 and 2.2: the query takes in the current and previous token mixture from head 0.3 and matches it against the key's token. That suggests a head that attends to positions where the key token equals the query's previous token.

I'm now checking whether the key-side reading of head 0.3's output implies induction rather than plain same-token matching, since that would mean the previous token is also feeding in. The raw token embedding gives a positive direct QK score too, but reading 0.3's mixed output suggests the x_{j-1} component is doing the work.

I'm weighing the asymmetry in head 3.x: the key side reads 0.3's output positively while the query side reads it negatively. That points toward induction, where the key's preceding token matches the query's current token, rather than plain same-token matching, so I'm leaning toward 3.0, 3.1, 3.4, and 3.5 being induction heads.

I'm now looking at head 2.0 from the query side, where it reads 0.3's output and matches it against the key token, while the key side reads 0.3 negatively. That could mean it attends to earlier occurrences of the current token rather than the previous one, though I want to check whether the alternative reading via the Δ profile fits better.

I'm considering whether layer-2 head 2.0 could be a content-based previous-token head: its query picks up the previous token from head 0.3, and its key holds the raw token, so it would match any earlier occurrence of that token, with the most recent one, j = i-1, favored by recency.

Now I'm checking the layer-3 heads' key-side composition with layer-1 head 1.5, which is fairly strong for 3.0 and 3.1.

I'm labeling the remaining weak heads: 0.0 as a diffuse content head, 0.1 as one that avoids matching the same token, and 0.2 as a low-rank head with a fixed key preference. I'm also checking 1.0, which matches on the low-rank signal from 0.2 in both its query and key.

I'm now going through the labels for the rest of layer 1 and layer 2: 1.1 looks like a broad averaging head, 1.2 through 1.5 look like token-matching heads that read the current or previous token, and 2.0 looks like a content-based previous-token lookup that feeds the induction heads.

Now I'm checking the layer-3 heads' key-side composition and the weak heads I haven't labeled yet. I'm thinking about whether the 0.2 delimiter-attending pattern points to position within a word or line, though that doesn't settle the language question. I'll go with English as a generic guess, with low confidence.

I'm working out the vocabulary size from the logit elbow: a Shakespeare-style setup would have about 65 characters, but my estimate of 60-85 leaves room for more, and rare symbols could push the count up.

I'm weighing whether the vocabulary is larger than the Shakespeare-style guess: the logit tail keeps falling until about 95, which points to a vocabulary of roughly 90 to 110 characters rather than 65. Accented letters or code symbols would fit that range better than a plain English set, though the head pattern alone doesn't separate code from prose.

I'm considering whether head 0.2 tracks distance since the last newline, which would let head 1.0 match columns across lines, a pattern that would suit code indentation better than prose. I'm still unsure, since the content part of 0.2's key preference is unknown and the positional part can't be read as a Δ profile.

I'm checking whether head 0.2's 2D output, with its near-equal singular values, carries positional information or just the content of whatever character it attends to. Its positional share is only modestly above chance, so it probably reflects a fixed delimiter-like character rather than position.

I'm checking whether head 0.2's value directions line up with its key preference, since that would show whether it's copying a specific character's feature or just tracking how far back something occurred.



---

Head 0.2's value path pulls token identity from the first two dimensions of the residual stream, so it copies a coarse summary of what the attended tokens are. Its key preference looks mostly positional, with the token subspace contributing little, so the attention pattern itself seems to ignore content and track position instead.

I'm weighing whether head 0.2's content-blind attention pattern could be doing something like tracking uppercase-heavy regions, such as speaker names in a Shakespeare-style corpus. Given the likely nanoGPT-style config, English is the default guess, though my vocabulary-size estimate hints at something larger, so I'm not certain. I'm now trying a more principled vocab count by comparing the layer-3 and layer-2 MLP projection ratios.

I'm estimating the vocabulary size from the layer projection ratios, landing around 75 to 90, and I can't tell apart Shakespeare, general English prose, or code from that alone. Since rare characters would shrink the effective dimensions, I'm leaning toward a broad English-text guess at around 35% confidence. I'm also starting to sort the heads by role: 0.4 and 0.5 look like previous-token heads, with 0.3 partly, and 3.0, 3.1, 3.4, and 3.5 as possible induction heads.

I'm going to drop the guess about 2.0 as a previous-token head, since its attention is content-based and I'd only be floating it without a real basis.

