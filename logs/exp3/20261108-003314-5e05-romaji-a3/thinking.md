I'm starting on a hard problem: working out the model's properties from the weights alone, with no forward passes. Without embeddings, pinning down the language looks very difficult, so I'm looking for other clues, like layer norm weights or biases, or whether the vocabulary size can be inferred from the rank of some matrix.

I'm checking whether the vocabulary size might show up as some low-rank structure in the weights, though that doesn't look easy to spot. Next I want to examine layer 0's attention q/k weights for positional structure, since previous-token heads would depend on the positional embedding.



---

I'm trying to tell which language a character-level model was trained on just from its weights. The idea is that the embedding directions could be read off the layer 0 weights, and the input space's effective rank would roughly match the vocabulary size, which might hint at the language, since English text uses about 65 distinct characters.

I'm checking the singular values of the layer 0 read weights, since directions the input never uses should decay under weight decay and leave a cutoff near the vocab size plus positional dimensions, though the 256 positional embeddings might occupy a lot of space.



---

The singular values near zero in the q, k and v projections come from the layernorm's mean subtraction, since the input is mean-zero after LN and one direction is null, so it acts as a weight factor. The layer-0 spectrum decays smoothly, which makes the vocabulary hard to recover directly. I'm considering a different route through the input to the layer-0 MLP, and whether I can reconstruct the embedding structure from the token-plus-position input to block 0 and the ln1 weight.

I'm weighing whether the vocabulary could be pulled out of the layer-0 MLP input or from token-plus-position embeddings, since the layer-0 spectrum is too smooth to read directly. I'm also wondering if the training text was code, English, or something else, and whether layer-0 attention patterns could hint at that.

I'm checking whether weight decay would have pushed the input-orthogonal directions toward zero, or whether the 0.02 initialization and limited training left them partly intact. The q singular values at index 300 are around 0.09, which is below what a random 384x384 matrix would give, so there does seem to be a decay. Next I'll look at the full spectrum of the combined layer-0 read weights, q, k, v and fc scaled by the ln weights, to see if there's a knee.



---

I'm checking the spectrum of the fc layer, where the values fall off near index 100 and then level out into a bulk. A random init at this size would give singular values from about 0.39 to 1.18, but the observed bulk sits lower, around 0.27 to 0.8, which suggests the weights have decayed from init. I'm wondering whether the signal part is roughly the first hundred dimensions.

I'm working through whether the signal part is about the first hundred dimensions, and that leaves the question of how the input space splits between vocab and positional. Nothing separates them cleanly yet, so I'm looking for another handle. One idea is that the token embedding directions might be recoverable from the layer 0 attention, though the large q bias complicates that.

I'm considering whether the positional embedding could be read off as a smooth, sinusoid-like function of position. Without the embeddings themselves, I'm looking at what the weights alone reveal, such as attention head types from the QK and OV circuits, and whether layer 0 heads that depend on positional structure can be identified that way.

I'm now testing whether an induction-style composition is visible in the weights: the query-side matrix from layer 2 against the key-side path through the layer 1 previous-token head. If that composed matrix has a large positive trace relative to its norm, it would suggest it's close to identity on the token subspace, even without knowing the tokens.

I'm restricting the analysis to the top singular directions of the layer 0 read weights, since that's where token content should live. For previous-token heads, the QK pattern would depend on the positional embedding, which is harder to read off the weights, but I could also look for them through the later heads that K-compose into them, like induction heads. I'm wondering whether the positional embedding itself can be recovered as the smooth part of the input space.

I'm checking whether the positional structure of the layer 0 query-key weights gives a clue: a previous-token head should show an asymmetric, rotation-like form over the positional subspace. I'm also planning to look at composition scores and per-head QK properties, such as the trace on the embedding subspace.

I'm now weighing whether the vocabulary could be pinned down from the 100 or so signal directions. The layer count and width look like a small Shakespeare-style character model, so the text is probably English, but that's a guess I haven't checked against the weights yet.

I'm wondering whether the vocabulary could be something other than English, since a larger character set or a small set like DNA would change how the input dimensions split. The roughly 100 signal directions point to a small vocabulary, so I'm starting to estimate how much of that space belongs to tokens versus positions.

Next I'm trying to pin down the input subspace more precisely, comparing the layer 0 fc spectrum against a Marchenko-Pastur bulk for a decayed init.

I'm counting how many singular values sit above the upper edge of the bulk, around 0.81, since the knee near index 104 suggests roughly 100 signal dimensions. Still, the signal subspace probably mixes token and positional directions, so the count alone won't settle what each part contributes.

Now I'm turning to the per-head analysis, checking the query-key and output-value matrices of each head and how strongly the later heads compose with them.

I'm checking how the value inputs of the next layer compose with the output of these heads, adjusting for that layer's LayerNorm weight.



---

I'm working out how the layer-2 heads compose with the layer-1 OV circuits, looking at the combined matrix for the induction pattern and whether its trace is positive on the token subspace. I'm also noting that the layer norm between them only roughly scales things, and that the next layer's norm weight is already folded into the query and key matrices.

I'm now trying to find a token subspace to restrict the trace to, since the top layer-0 directions also carry positional information. Layer-0 MLP outputs might re-encode token identity, so I'll start with the trace over the full space and also check Frobenius composition scores for each pair of earlier and later heads.



---

I'm setting up a composition analysis that traces the full embedding space, looking at how the query side of one head and the key side of another might combine for K-composition induction. I'm checking whether the trace of that combined matrix over the embedding subspace comes out positive, and I'm about to compute its trace norm.

I'm normalizing the trace by the product of the Frobenius norms to judge the composition strength.



---

I'm noting that layer 3 head 2 K-composes positively with several heads in layers 0 through 2, though that isn't strongly diagnostic on its own. I also suspect the embedding projector P might be off. Separately, layer-0 head 0.1 shows high K-composition with a few layer-1 heads, so its output is likely being read by their keys.

I'm checking whether the layer-0 head's output is picked up by the keys of the layer-1 heads it composes with most strongly. Positional attention in layer 0 would need the positional embedding subspace, which I'd have to infer from the query bias. For the layer-1 side, I'm planning a finer check: forming the combined query-key and value product for that head pair and comparing its eigenvalues against the query-key term alone.

I'm moving from the input projection to the token side: the layer-0 value path should carry token identity, and the layer-1 query reads the current token. So I'll form M from those weights and check whether its eigenvalues are mostly positive and real, which would mean it acts close to an identity on the token subspace.



---

I'm reading the layer-1 heads as K-composing with 0.1, and 0.1 looks like a previous-token head. Still, I'm not sure all six of them are induction-like, since positive eigenvalue scores might just reflect attention to previous tokens generally.

I'm checking whether 0.1's positive score is just an artifact of its OV being near-identity, so I want to compare each head's QK eigenscore on its own. I'm also wondering whether 3.2, with its high QK trace and positive interactions with several heads, is attending to same-token positions. The small QK norm of 0.1 makes me doubt it's a true previous-token head, since its attention would then depend mostly on biases rather than position.

I'm looking at 0.1's QK spectrum in detail to see whether there's a positional subspace, since small char models often have a few dominant smooth position directions. I'm also checking whether the top singular direction of the layer-0 heads (8.76, which stands out) is a positional or common direction, and I plan to compare the SVDs of each layer-0 head's QK to see how their singular vectors overlap.



---

I'm looking at layer 0 heads 1 and 5, where the QK and OV circuits look close to low-rank and identity-like. If QK is positive semi-definite in the positional subspace, a query at position t scores highest against a key at its own position, so these heads would be doing local attention. I'm checking what happens without the

I'm reasoning about how a positive-semidefinite QK form would make a head attend to itself and recent tokens, and whether a translation-invariant previous-token pattern would need complex eigenvalue pairs with a rotation angle matching the frequency. I'm wondering whether the positional embedding subspace can be recovered at all, and I'm planning to check the eigenvalues of the layer-0 QK matrices for their magnitudes and phases.



---

I'm working out whether the positional subspace can be located from the learned position embeddings, which seem to feed mainly into layer 0. I'm also considering whether the layer-0 heads with complex-pair eigenvalues (around 0.2 and 0.3) with large angles could be the positional heads, since they look like sinusoidal behavior.

I'm now checking whether the heads with rotation-like eigenvalues at 0.0, 0.2 and 0.3 are the positional ones, since their angles would let them track offsets like a prev-token head. High-frequency components seem needed for a sharp previous-token score, though the token-to-token part of those eigenvalues may not carry any positional meaning.

I'm checking whether bias-driven attention in the layer-0 query could pick out the previous token, since a key-only term would only favor absolute positions. I'm also noting that the strong K-composition into layer 1 makes head 0.1 look like the likely previous-token head despite its positive-semidefinite QK.

I'm weighing whether a key-only bias term could only favor fixed absolute positions, which would give a self-plus-recent smear rather than a true previous-token pattern. Layer-0 heads with positive-semidefinite QK might instead be matching the same token, since a head that only attends to itself would be redundant with the MLP. I'm also planning to measure how much each layer-0 head's query and key weights read from the main embedding directions.

I'm weighing whether the top embedding directions of block 0's first LayerNorm might mark positional dimensions, though that feels unreliable. I'm considering comparing the principal angles between the Wq row spaces of heads 0.0, 0.2 and 0.3 to see whether they share a positional subspace.

I'm now comparing the principal angles between the Wq and Wk row spaces of the layer-0 heads to see whether they share a common positional subspace.



---

Heads 0.0, 0.2, and 0.3 spread across many dimensions, around rank 30, while heads 0.1, 0.4, and 0.5 are low-rank. Heads 0, 2, and 3 also share a large leading singular value in their key matrices, so I'm checking whether they point at the same key direction and how that relates to the query bias.

I'm weighing whether 0.1 could be the previous-token head through content matching or positional structure, since its low-rank PSD pattern would normally favor self-attention. The composition evidence pointing to 0.1 conflicts with the idea that 0.0, 0.2, or 0.3 is the prev-token head, so I need to check the positional subspace more carefully.

Now I'm checking whether heads 0.0, 0.2, and 0.3 share the same positional subspace, by comparing the column spaces of their top 30 QK dimensions.



---

I'm noting that heads 0.0, 0.2, and 0.3 seem to share a roughly 30-dimensional subspace, which would fit a positional subspace since all three are positional heads. I'm now defining that positional subspace as the union of their query and key spaces, and wondering whether I can reconstruct the learned positional embeddings for the 256 positions from how QK acts on it.

I'm checking whether the rotation angles in heads 0.2, 0.0, and 0.3 stand in a 1:2:3 ratio, which would suggest offsets 1, 2, and 3. Those angles look about right at roughly 40°, 80–95°, and 114–160°, but I don't know each plane's frequency, so the hypothesis is still shaky.

I'm testing whether the planes shared across heads 0.2, 0.0, and 0.3 are invariant by taking head 0.2's QK on the subspace and reading off its eigenvector planes. Then I'd measure head 0.0's rotation angle on those same planes to see if the offset ratios hold.

I'm working through the sign convention: the score peaks when the offset equals θ/ω, and both signs of the eigen-angles may show up. Next I'll take the positional basis from the combined spaces, compute the 2x2 blocks from the real Schur form of head 0.2's projected QK, and read the angles that heads 0.0 and 0.3 make within those planes.



---

I'm narrowing to the roughly 30 dimensions that are strongly shared across all six spaces, since their norm sits near sqrt(6). Within that subspace, I'll compute the restricted QK matrices for heads 0, 2, and 3, then eigendecompose head 2's matrix to find planes and check the angles of heads 0 and 3 against them.



---

I'm checking the angle ratios across several head pairs in the shared planes, and they come out to roughly 1 : 2.2 : 3.5. They aren't exact integers, so I still need to pin down the sign and which direction counts as "past" before drawing a conclusion.

I'm working out whether the head offsets line up with roughly 1, 2, and 3 tokens back, since the 0.2 head clearly isn't attending to itself. I'm also questioning whether 0.1's small positional contribution explains why the layer-1 keys pick up its output so strongly, and I'll check that by composing the eigen-decomposition with the 0.2 and 0.0 heads.

I'm checking whether layer-1 keys read head 0.1's output more than 0.2's, and the composition numbers suggest they do: 0.1 scores around 0.15–0.19 while 0.2 sits near baseline. So the prev-token head doesn't seem to be what layer 1 is picking up.

I'm now probing head 0.1 more closely, looking at its key-side bias direction and whether its query-key structure lines up with the dominant positional singular vector I found earlier, to see if it does a recency-style job after all.

I'm estimating the positional embedding's frequencies from the per-step rotation angles, which would put a period of roughly 40 positions on the 9° component. Head 0.1 is different: I'm considering whether it attends to earlier copies of the same token, but an identity OV would just reproduce that token, so I'm wondering what the PSD part combined with the bias adds.

I'm checking whether head 0.1's key-side bias pushes it toward particular token types in the token space rather than by position, which would make it a content-based head that averages over characters like newlines or vowels, with its identity-like output carrying along which special tokens showed up in context.



---

I'm working out how large the attention logits get once the normalized input is scaled by the head dimension, which puts the possible scores around 24. Looking at head 0.1, its key preference sits mostly in the positional subspace, and its QK matrix is positive semidefinite, so I'm starting to suspect it's a broad positional, recency-style head.

I'm now considering whether I can reconstruct the positional embeddings directly, but that looks circular, so I'll set it aside. Head 0.1 doesn't touch the rotation planes at all, so its positional signal must live in a non-rotating direction, something like an absolute-position scalar that makes it favor recent or distant keys. Since layer 3 head 3.2 shows strong eigenscore alignment with the prev-token head 0.2 and several layer-1 heads, induction via K-composition looks more plausible than anything coming from 0.1.

I'm checking head 3.2 again: its trace on the QK side is strongly positive, but its OV norm is very large, so the raw numbers may be misleading. A cleaner test for an induction pattern would be to restrict the analysis to the token subspace, excluding the positional planes, and compose the previous-token head's OV with the candidate's QK to see whether the resulting matrix is positive on that subspace.

I'm now planning to run the same eigenscore test for each layer-0 head against the later-layer heads, computing the composed matrix on the token subspace to see which pairs look like K-composition.



---

I'm checking the 3.2 head's composition with 0.2, which shows strong positive alignment and makes it look like an induction head that matches on the previous token. I also want to confirm that the current token still reaches layer 3 through the residual stream, and then look at how 3.2 composes with layer 1.

I'm also considering whether the other early heads, the positive ones at layers 1 and 2, carry previous-token information that 3.2 might read alongside 0.2. Next I'll check whether any layer-1 heads act as previous-token heads by using positional information, and look at each later head's QK fraction in the positional subspace and its rotation angles.



---

The later layers seem to skip the raw positional subspace and rely on whatever the layer-0 positional heads output, possibly after processing in the layer-0 MLP. Previous-token signal shows up at offsets 1, 2, and 3 with weights 0.2, 0.0, and 0.3, and I need to check those offsets against the ratios of roughly 1 : 2.3 : 3.7.

I'm checking whether offsets 1, 2, and 3 really fit those ratios, since the planes might not be perfect rotations. I'm also testing an alternative where 0.2 is offset 2, which would put the others at about 4.6 and 7.4, and I can't rule that out yet. Next I'm comparing the per-plane angle ratios across heads 0.2, 0.0, and 0.3 to see which offset fits all planes at once.

I'm comparing the angle ratios across the planes for each head, and they come out very consistent at about 1 : 2.27 : 3.62, which isn't a clean 1:2:3 pattern. A 3:7:11 split fits those ratios well, though the angles may be skewed by the layer norm and by the query and key projections not being pure rotations.

I'm checking whether the attention-head angles, which come out at roughly 1 : 2.27 : 3.62 across planes, could instead reflect a head that averages over a small window of offsets rather than attending to one exact position. That would explain why the angle ratios hold steady across frequencies even though they aren't clean integers.

I'm checking whether the head at 0.0 has a sharp peak or averages over a few offsets, since its magnitudes shrink on the faster planes. The 0.2 head's magnitudes stay nearly constant across planes, which points to a single exact offset. So I'm weighing whether 0.2 is offset 1 while 0.0 and 0.3 cover broader windows.

I'm checking the earlier idea that head 0.2 is the sharp previous-token head: its K-composition with 3.2 is strong, which fits. The sign question also settles things, since a future-facing head would be masked and useless, so 0.0 and 0.3 must look backward too. Next I'll turn to layer 1 and what head 0.1 contributes to the keys there.

I'm now asking whether head 0.1's key preference, which is mostly positional, reads a non-rotating direction in the positional value space. I'm checking the 0.2 eigenvalues for near-zero angles to see if any such direction exists. Head 0.1 might instead be a token-similarity head, perhaps attending to the start of words such as after spaces.

I'm now characterizing the remaining heads. Layer 0 has 0.2 as the previous-token head, with 0.0 and 0.3 attending a few positions back, and 0.1, 0.4, and 0.5 look content-based. For layers 1 through 3, 3.2 stands out as the strong induction candidate, and I still need to check how the layer-0 positional heads compose with it.

I'm computing separate Frobenius norms for the key-side and query-side compositions of head 3.2 with the layer-1 head outputs, to see which side carries the composition. I also want to check whether layer-2 heads compose with layer-1 heads that copy previous-token content.

I'm now computing Frobenius scores for every head pair, comparing the key-side and query-side compositions to find which side carries the signal.



---

I'm checking what 0.1 actually writes out. Its OV is close to identity, so it mostly copies the tokens it attends to, and its QK is low-rank, mostly over token directions with a positional key bias. Next I want to see whether its output lands in a narrow subspace and which layer-1 keys read from it.

I'm wondering whether 0.1 is simply attending uniformly across the context, or whether its position-based key bias makes it favor the earliest tokens as an attention sink. Either way, the next step is to check the singular values of the OV matrix for the layer-0 heads to see whether that output is actually low-rank.



---

I'm wondering whether head 0.1's value vectors pull mostly from the positional subspace, which would mean it carries position information forward. Its key preference also leans on positional dimensions that sit largely outside the rotating planes, so it may encode something like distance to a reference token. Layer 1 keys seem to read from it, and it might also attend to tokens like newlines or spaces.

I'm thinking that head 0.1 might track the position of the last newline, so the model could work out a column offset from it, and that the layer-1 keys might read this. I'll check the top singular directions of its QK matrix next, to see whether the key side looks token-specific and the query side mostly bias.

I'm working through a spectrum question: whether the layer-0 matrix has a clear cutoff separating used directions from noise. Tracking the singular values, they fall off gradually without an obvious knee, so I'm considering a different angle: checking each right singular vector for how much input it actually activates.

I'm checking whether the layer-0 spectrum shows a knee by comparing it against layer 3, where the singular values fall off in a different pattern and the ratio at the tail is about 0.09 against 0.15. Neither spectrum looks like it has a clear cutoff. I'm also considering whether weight decay under Adam would shrink unused directions exponentially, which would be a way to separate them.

I'm working out whether weight decay with Adam would actually shrink the unused directions, and I'm considering that columns of the layer-0 weights outside the input span get no gradient at all. Once I account for the LayerNorm gain, the effective read is still confined to the input space, so I want to redo the gradient calculation with that factored in.

I'm checking whether the orthogonal complement of the top-k right singular vectors of the fc0 weights looks like iid Gaussian noise, using kurtosis, column norms, and a Marchenko-Pastur fit. I'd raise k until the residual spectrum matches the MP law.

I'm testing whether the residual after removing the top-k singular directions of the fc0 weights looks like iid noise. For each k, I'd compare the residual spectrum to the Marchenko-Pastur edges, using a sigma estimated from the Frobenius norm, and check where the largest residual value falls relative to the upper edge.



---

I'm estimating the signal dimension for fc0 at roughly 110 to 130, since the input space combines token and position embeddings, plus the layernorm bias direction. If positional embeddings take up about 30 dimensions, the token part would be around 85 to 100, but positional embeddings across 256 positions may spread over more dimensions with small variance, so I need to be careful with that split.

I'm weighing whether the positional embeddings, with their random initialization across 256 positions, could fill much of the 384-dimensional space, which would conflict with the roughly 120-dimensional signal I'm estimating. Weight decay on embeddings might explain why that random part has shrunk, leaving a narrower subspace. That leaves the token vocabulary's own dimensionality still unresolved.

Next I'll check the Wv singular values for each layer-0 head, looking for a knee near the vocabulary size, and then run the same check on the concatenated 384x384 layer-0 Wv.



---

I'm reading the value projection of head 0.2 and finding a knee around dims 20 to 24, with the weights dropping off after 16. Head 0.0 shows a similar knee near 24, and head 0.3 sits closer to 32, so concatenating them gives roughly 40 useful dimensions. That makes me wonder whether the prev-token head is encoding token identity in a small set of dimensions, which would hint at a limited vocabulary.

I'm checking whether a vocabulary of about 25 to 30 would fit the data, though a prev-token head usually copies full token identity, so I want the complete spectrum of head 0.2's value weights before drawing a conclusion.



---

I'm estimating the signal dimensions from the Wv singular values: a sharp drop after about index 20 suggests roughly 20 to 23 for layer 0.2, around 22 to 23 for 0.0, and about 29 for 0.3, so the concatenated Wv0 carries roughly 33 to 35 dimensions. That points toward a token embedding space of about 30 to 36 dimensions, which would bound the rank of the vocabulary's span and maybe shift by one after centering.

The vocabulary guess lands near 27 for a lowercase-only character set like text8, though some of the Wv0 dimensions might be positional rather than token-based. I'll project the concatenated Wv0 signal subspace onto the positional subspace to separate the two and check how many dimensions are purely token-driven.



---

I'm estimating how many embedding dimensions the token subspace uses: the three positional heads share about 20 dims, with about 29 combined, and the full V read space is around 34. That points to roughly 31 token dims, though the embedding might span more dims than V reads. Working from the vocabulary size, the effective dimension lands near 30.

Now I'm weighing whether the token embedding might use more dimensions than the value reads pick up, since rare characters may be ignored by the heads. With a 65-character vocabulary like Shakespeare's, the model could plausibly use closer to 50 dimensions, though it's unclear it would. Next I want to see how much of the full signal space in fc0, around 120 dimensions, falls into the token and positional subspaces and how much is left over.



---

The fc0 read share from the token and position subspaces came out lower than expected, but that's because layer 0's second LayerNorm sees the residual after attention, so its reads include the attention output written through Wo. That means my earlier "embedding subspace" from R0 was partly fc contamination. I'll use only the attention-side read space of layer 0 instead, going through Wq, Wk, and Wv with the ln1 gain.

Checking the concatenated layer-0 attention weights, Wq, Wk and Wv stacked together, for their rank.



---

I'm sorting out the attention read space in layer 0, which looks like roughly 90 to 100 dimensions, with about 30 positional and 31 token dimensions among the top 60. I'm not sure what fills the rest beyond that, and the small bottom singular values don't fit a simple noise model, so I don't trust that estimate. Next I want to test the token subspace that the content heads' queries and keys actually read.

I'm checking the token-like read spaces of the content heads 0.1, 0.4, 0.5 and v, projecting out the positional part, and looking at the spectrum of their union.



---

I'm not sure yet whether the decline has a clear knee, so this may be a tail around 0.5 at position 60 and it's not clean. For the vocabulary, the Wv knee at 20 and the concat at 34 could fit a 65-character set like Shakespeare, if the head only needs 20 dimensions because it merges some characters. That still isn't decisive. I'll also check whether the text is around 30 to 40 characters.

I'm weighing whether the vocabulary could be a protein-like alphabet of about 20 to 25 amino acids, since that would fit a 20-dimensional knee. The 0.3 peak sits at about 3.6 times the 0.2 offset, which matches alpha-helix periodicity, though that is speculative for now.

I'm checking the knee more closely: the 0.2 Wv singular values drop from about 0.40 to 0.32 at index 20, then decline gradually toward noise, so roughly 20 strong dimensions plus a few weaker ones. I'll also look at whether the protein-versus-text question can be settled by how heads treat word boundaries.

I'm weighing whether these heads might attend to tokens of the same class, like vowels or spaces in English, or hydrophobic residues in a protein sequence. The embedding structure could help me count the effective token dimensions, so I'll look at that next.

I'm checking whether the token-related part of the layer 0 value weights has a knee near 20 to 27 dimensions once the positional dimensions are projected out, since that would separate a protein-sized alphabet from a text8-sized one.



---

I'm estimating the token embedding spans about 33 dimensions, then noise. I'm guessing at a vocabulary size near that: lowercase letters plus space, newline, and a few punctuation marks, or a protein alphabet with rare residues and specials. Rare tokens would get less weight, though they still get embeddings.

I'm weighing datasets with a vocabulary near 35: text8 has 27 symbols, proteins about 25, and arithmetic expressions only around 14. Shakespeare, Python, and ABC music all sit far higher, so the lowercase-English-style options are the closest fit.

I'm looking at head 0.1's query-key structure, where the eigenvalues suggest a low-rank pattern. If it attends to space tokens, the key side would reduce to a single direction, which could be set by a bias term.

I'm wondering whether the token-specific part of the key bias can be pulled out from the rest, since its magnitude is around 0.36 once the positional part is removed. Without knowing the tokens it's hard to say, so I'm thinking about whether the token embeddings themselves, as points in roughly 33 dimensions, could show an axis-aligned structure under Adam.

I'm considering whether layer-0 MLP neurons that detect specific tokens could expose the token embedding directions, by projecting their input rows onto the token subspace and looking for clusters.

I'm going to try clustering the layer-0 MLP input rows, projected onto the token subspace, to see whether specific token directions emerge.

I'm testing whether the layer-0 MLP rows, projected onto the token subspace, point toward individual token directions. I'll compute their cosine similarities and count how many tight clusters show up.



---

I'm trying a sparse-coding or ICA approach to pull out token directions, reasoning that if neurons act as token detectors, rows in the token subspace should point toward token vectors, and I can check that with k-means or by looking at the highest-norm rows. I'm also weighing whether a three-way product analysis would be cleaner, and I'm keeping an eye on my tool budget, around 21 calls used with 19 left.

I'm still weighing the token-space approach: the induction-head matrix should act like a same-token map whose spectrum reflects the embedding Gram, though that's not decisive. I'm considering ICA or k-means on the token-space rows, but neither would settle it on its own. I'm also thinking about the puzzle design, since the vocabulary size and the rank of the token space might hint at the kind of text.

Checking whether the vocabulary size can be inferred from the token-space rank: a learned-useful subspace might be smaller than the full vocabulary, since the random init leaves every token spanning the space while only the learned directions get read, so the knee doesn't pin down V. Uppercase may share directions with lowercase, which suggests the vocabulary could be larger than the ~33 the rank hints at. I'm also considering whether line structure or positional periodicity could help, though learned positions wouldn't align with line breaks.

I'm checking whether the key-side positional bias has a meaningful token component, which would mean one token is favored across the board, probably a space. The induced logit range looks small, about half a logit, so I'm doubtful this tells me much. Testing the language from structure alone seems hard, and I'm weighing a few hypotheses, natural language with word boundaries, versus protein, which lacks such a token.

I'm going to try greedy clustering of the fc0 rows in token space and count how many tight clusters, with cosine above 0.8, come out. If there are roughly 27 clear groups, that would point toward a letters-plus-space text like text8.



---

I'm stuck on the elbow with no clear answer, so I'll try a different angle: whether the positional-embedding frequencies carry context information. That doesn't look promising. Now I'm weighing the candidate text types a designer might have used, from English prose and code to several natural languages and biological sequences.

I'm now sizing up the candidate text types a designer might have used. Vocabulary size is the thing I can actually infer from, and CJK would need hundreds of token dimensions, so my roughly 33 token dimensions point away from it. DNA would only need about 4, which doesn't fit either. I'm also going to check the token-subspace size another way, using the singular values of the induction head's composition matrix.



---

I'm checking whether a rank of about 20 matches the 20 amino acids. A prev-token head on English would need to copy roughly 27 symbols, letters plus space, so I'm wondering whether 27 near-orthogonal vectors can fit in 20 dimensions at all.

I'm weighing whether 0.3 covers offsets 3 to 4 or a broader span, and whether the 3.6 ratio is informative or just what any text would produce. Protein data has repeated domains, so induction might matter somewhat there, but in English it's crucial and usually shows up early. Only head 3.2 looks clearly like an induction head, which doesn't settle much yet. I need a sharper test to tell the datasets apart.

I'm checking whether the 0.1 head responds to a specific token class like spaces, using the recency pattern and a fixed key preference. Its bias-driven part looks small, around 1.4 logits, and the query-dependent part seems to be on the same order.

I'm looking at whether the 0.1 head's behavior comes from its bias or from its query-dependent part, but I can't settle that from these numbers alone. I'm considering checking the layer-1 heads next, since many of them seem to have negative OV eigenvalues that might be read from 0.1's output.

I'm asking whether 0.1 reads the positional part of its residual stream, or whether its value vectors ignore position and carry only token identity. The layer-1 keys all picking up 0.1's output make me wonder if it's routing a word-boundary position signal forward.



---

Looking at the layer-0 head's value weights, the non-plane positional dimensions carry a lot of signal, so this head probably copies the absolute position of whatever it attends to, which would help compute distances. Paired with its selective attention to a certain class of tokens, it may be tracking something like the position of the last space or newline, which would give column or word position for layer-1 keys to read.

Next I need to give labels to the layer-1 through layer-3 heads, which means working out whether any of them are previous-token heads. Later layers barely read the raw positional subspace, so a previous-token head there would have to get its position from what layer-0 head 0.1 writes out. The layer-1 keys do read that output strongly, so I'll check whether the signal comes from the positional part of 0.1's value path or from its token-driven part.



---

I'm checking whether the layer-1 key reads of head 0.1's output are actually large compared to the key weight norm. Head 0.1's OV map has a strong top singular pair, so it writes mostly into two output directions. I want to find those directions and see whether the layer-1 heads' key weights pick them up, which would suggest 0.1 acts like an attention-sink head writing a roughly constant vector.

I'm weighing whether head 0.1 is a near-constant attention-sink writer or an averaging head that only matters when its output varies across key positions. Its output does vary with position, so the key bias argument doesn't settle it. I'll check the top singular vectors of 0.1's OV to see which inputs they read.



---

I'm checking whether the layer-1 key outputs, which push strongly on output direction 1, are unusual compared to a typical key vector norm, and I'm questioning whether the Frobenius-norm estimate I was making is worth pursuing.

I'm noticing the layer-1 keys all lean hard on the same output direction from 0.1, which looks like a positional or recency signal rather than anything specific to content. Later layers barely touch raw position dimensions, so that signal probably gets passed along through 0.1's processed output. I want to check the query bias of those layer-1 heads against that key direction to see whether it's a consistent preference.

I'm going to characterize the later heads with a few systematic metrics: a QK score for content similarity, an OV score for copying versus suppression, and K-composition with the previous-token head 0.2 to test for induction behavior. The token-space traces already show 3.2 as a strong induction candidate, and I still need to check whether some layer-1 heads act as local, recency-based attenders.

I'm checking whether 3.2's OV output actually reaches anything downstream, and noting that its OV norm is the largest, which fits an induction head writing its prediction straight toward the logits. I'm also wondering whether any layer-2 or layer-3 heads build on earlier heads' prev-token information, which would mean a more complicated induction setup than 3.2 alone.

I'm wondering whether the layer-1 heads' OV outputs, which look like they could be local or prev-token copying, are negative overall, and whether 3.2 is still getting a positive composition through them, which would mean the signs are cancelling somewhere in the QK term.

I'm computing token-space restricted versions of the scores for the later heads: the OV eigenscore through the token embedding and the QK eigenscore in the same space.



---

I'm reasoning that the later-layer OV circuits write negative token directions, and that whether they copy or suppress depends on whether the unembedding is tied to the input embedding, as in nanoGPT, where the attended token's direction could be promoted or suppressed.

I'm checking whether a -0.52 token-space OV for head 3.2 fits anti-copying, since with tied embeddings a copying induction head should map the copied token's embedding to positive logits. The fact that later-layer OVs are negative while layer-0 ones are positive makes me wonder whether the model's embeddings are actually tied, which I can't confirm yet.

I'm now weighing whether the head's negative token-space OV, combined with its strong positive token-level QK score, points to a duplicate-token or anti-repeat head rather than a classic induction head. Whether that holds depends on whether the embeddings are tied, which I still can't confirm.

I'm checking whether layers 2 and 3 have other heads that combine with the earlier previous-token heads to form longer induction chains, and the offset-2/3 candidates like 3.0, 3.1, and 2.3 look weak. I'm also considering whether any later heads act as previous-token heads, though that's harder to pin down, so I'll set it aside. Next I want to look at the layer-1 heads and how their keys respond to the position signal from head 0.1.

I'm now checking how the layer-1 heads' keys respond to head 0.1's position signal, by computing the bias contribution bq·Wk·u1 and comparing it against the query-side reading.



---

I'm wondering whether the output direction u1 from layer 0.1 acts as a strong key feature that layers 1 and 2 use to suppress certain keys, perhaps marking a particular kind of position. I'm also noting that the query-key product with u1 is large, so this depends on the query as well.

I'm counting the calls I've used so far, to see how many of my budget remain before I try the language question more.

I'm now estimating how many tool calls I've used, which puts me around 33 with roughly seven left, so I need to be economical. For the language question, the evidence points to a token subspace of about 31-35 dimensions and prev-token copying around 20 dimensions. I'm also starting to consider whether the embedding's effective rank for a small character vocabulary would be a useful reference point.

I'm considering whether the token space needs more dimensions than the prev-token copying does, since the leftover spectrum after the positional and token components still holds about 20 to 30 directions. Those could be rarer characters like capitals that the content heads read, and the combined token spectrum falls off gradually around 45, which makes me think the token subspace might span something closer to 45 to 60 dimensions.

I'm estimating the vocabulary at around 40 to 65 characters, which fits a typical char-level English dataset. A Shakespeare-style corpus seems most likely given this config, though I'm only about 35% confident. I'm also thinking about checking whether any layer-0 head has a QK pattern dominated by one token direction, like a newline or space boundary detector, perhaps by looking at the token-space eigenspectrum.

I'm checking whether the token-space structure of layers 0.4 and 0.5 has a dominant direction, since a strong delimiter-like key preference would hint at natural text rather than protein. The Wv0 token spectrum looked flat, so I'll look at the QK eigenvalues in token space for those heads to see whether bias-driven key preferences are concentrated.



---

I'm leaning toward English, probably Shakespeare-style, though my confidence is only around 0.3. The token dimension of about 20 to 35 could point to lowercase text like text8, which has 27 symbols, and if the designer used several languages, English stays plausible. I'm also looking at a sharp knee at exactly 20 in the 0.2 head, which seems suspicious, and I want to think more about whether protein training text fits that.

I'm checking whether protein text fits, since the designer's phrase "kind of text" is generic. The token spectrum has roughly 28 to 33 dimensions above the noise floor, which is more than a ~25-token protein vocabulary can support, so protein looks less likely than English.

I'm now going over the attention heads in layers 0 and 1: head 0.1 is a broad content-and-position head whose signal the layer-1 and layer-2 keys rely on, head 0.2 is a sharp previous-token head, 0.3 looks back a few more positions, and 0.4 and 0.5 are weaker content-based heads, with the layer-1 heads mostly mixing that position signal.

Layer 2 has a similar pattern, with head 2.4 leaning most on the 0.1 position signal and 2.3 standing apart by reading more from the 0.2 and 0.3 keys. Layer 3 looks like head 3.2 is the induction head while the rest handle general content. So 3.2 is the induction head and 0.2 is the previous-token head, and I'll leave out 0.0 since it attends about two positions back rather than to the immediate predecessor.

