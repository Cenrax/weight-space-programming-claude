I'm weighing whether the language can be inferred from the weights alone, given there are no embeddings and positional information has been removed. Previous-token heads normally depend on positional embeddings, so I'm looking at whether induction heads and their composition with earlier heads might still reveal structure through the QK circuits.

I'm checking whether the composition scores between the earlier and later layers might show structure, and I'm wondering if the vocabulary size can be estimated from the weights, perhaps guessing something small like a Shakespeare-style character set. I'll start looking at the weights directly.



---

I'm working through per-head QK and OV properties, starting with whether the layer 0 previous-token heads can be analyzed without positional embeddings. Then I'm looking at K-composition scores between a later-layer head's QK matrix and an earlier head's OV matrix, normalized by their Frobenius norms.

I'm checking the induction-head side by looking at K-composition with earlier previous-token heads, and I'm also looking at the positional subspace in layer 0. The idea is that previous-token heads might show up as low-rank, rotation-like structure in W_Q^T W_K, so I'd compare the eigenvalues of that 384x384 matrix: complex ones for rotation-like heads, positive real ones for copying-style heads.

I'm wondering whether the input embeddings could reveal something about the vocabulary, but the unembedding isn't available to map them back to tokens. I'm also considering whether the text type could be guessed from long-range attention patterns, though that would be a low-confidence guess at best. Reconstructing token embeddings from block 0's QK and OV weights is the next idea I'm turning over.

I'm thinking about whether the block-0 input weights carry the shape of the input space: since the gradients for those weights come from the token-plus-position embeddings, their singular structure might show a subspace of about vocabulary size plus positional dimensions. I'm also wondering whether weight decay would have pushed everything outside that span toward zero, which would make the subspace easier to pick out. Once I find it, I'd want to see if individual tokens can be picked out inside it.

I'm checking whether the singular structure of the block-0 input weights shows a subspace around vocabulary size plus positional dimensions, and I'm thinking about whether a spectral knee could hint at the number of tokens.



---

I'm looking at the tail of values around 0.5 to 0.9, which looks like noise rather than decaying signal, alongside a structured block of about 150 dimensions, and I can't yet pin down the vocabulary from that. I'm also considering whether the input space is just token plus position embeddings, with layer norm removing the mean, though I haven't worked out how to recover the token embeddings from that. The block 0 MLP seems to act mostly as bigram token detectors, which doesn't help much here, so I'll start by examining the attention heads.

I'm setting up the per-head analysis: computing the QK and OV matrices for each head and their composition scores. For layer 0 I want to look at previous-token heads, checking whether the QK eigenvalues look rotational (suggesting a positional offset) or real and positive (suggesting same-token matching).

I'm working out the K-composition score between a layer-a head and a later layer-b head, using the Frobenius norm of W_K^B W_OV^A normalized by each factor's norm. I also need to fold in layer b's layer norm weights, and then compare the result against a random baseline.



---

I'm noting that the later-layer OV circuits look negative in residual space, which I think is just because they read and write different subspaces, so it doesn't mean much on its own. Then I'm wondering why the layer 0 QK norms are so small: the embeddings should have plenty of norm, and layernorm should handle that, so weak attention at layer 0 would be surprising unless something else is going on.

I'm now looking at the query and key biases, since a bias term on the key side could give a key-only preference for certain tokens or positions. Next I want to compute composition scores, starting with induction heads: K-composition between a later head and the previous-token heads in earlier layers, and checking whether the combined matrix has positive eigenvalues in the residual subspace.



---

I'm looking at head 0.1, which many heads are strongly K-composed with, and its small QK norm suggests attention may be close to uniform by content. I'm wondering whether positional or bias terms drive it instead, and whether a previous-token pattern could be the reason.

I'm working through the bias term: the query bias paired with the layer-normed key input could favor certain positions, though an absolute position preference wouldn't produce a previous-token pattern. Since layer-norm outputs can have large norm along particular directions, the small QK norm alone may not settle how sharp the attention is. I need the top singular values of the QK matrix, including the 1/8 scaling, to tell.



---

I'm working out whether the layer 0 QK singular values are tiny enough to matter, using the normalized input norm of about 19.6 and an estimated maximum of roughly 35 for aligned vectors. I'm realizing that typical vectors are spread out rather than aligned, so that bound probably overstates things if the input sits in a few directions.

I'm checking whether layer 0's QK singular values, around 0.07 to 0.09, are actually trained or just near random init, and a rough random-init estimate of about 0.015 suggests they're trained but small. Without the embeddings, it's hard to say which head is the previous-token head, so I'm looking at the q bias contributions and the composition scores into later layers.

Checking whether layer 0's OV spectrum is low-rank matters here, since a low-rank OV aligned with the top QK directions would score higher on composition. I'll look at the full OV singular value spectrum, along with Q- and V-composition. I'm also weighing whether the block 0 spectrum, with its roughly 150 large values, points to a large vocabulary like Chinese characters, which would explain why the embeddings use most of the 384 dimensions.

I'm estimating whether the noise floor in block 0's spectrum is just decay from initialization, since the tail sits below what init would give. If so, the directions with small singular values are ones the input rarely uses, which would put the used input dimension around 100 to 150. That might fit a small vocabulary plus positional dimensions, pointing toward an English or European alphabet rather than Chinese.

I'm checking the spectrum of the stacked Q, K, and V weights for block 0, using only the embedding-reading projections, to find where the singular values drop off and estimate how many input dimensions are actually used.



---

The qkv knee sits around index 70 to 80, where singular values fall from about 1.0 to 0.8. A standard init at std 0.02 would put the tail between roughly 0.29 and 1.07, so the decay doesn't look very strong. I'm not sure yet whether the roughly 70 dimensions above 1.0 are real structure, so I'll leave that open. Another angle to try is that block 0's input is just the token embedding plus position.

Now I'm weighing whether the roughly 70 dimensions above the knee reflect the vocabulary size, since block 0 only sees the token and position embeddings. That reading doesn't settle things, because position embeddings add to the span too. The data type could also be code or non-English text, which would fit the dimensionality better than DNA would.

I'm checking the gaps between neighboring singular values in block 0's qkv, and I want to compare q, k, v and the fc layer separately. I'm also considering the block 0 value weights and the layer norm weights, though the residual basis isn't obviously aligned with the embeddings. Positional embeddings are smooth in position, so a head that attends to the previous token might read a low-dimensional positional subspace, which I could try to isolate.

Now I'm looking at the query-key interaction in block 0. The symmetric parts for heads 0.0 and 0.3 are small and of opposite sign, while 0.1 and 0.5 are near-symmetric and large, which could mean they attend to the same token. The layer 0 scores would be big if the vectors lined up with a top singular direction, but since the embeddings spread over roughly 70 dimensions, a single projection is probably much smaller.

I'm estimating how the bias terms in block 0 might add a key-only contribution to the attention scores, so I'll check how large the q biases are.



---

Biases in the key projection come out near zero, which fits with the softmax making the key bias irrelevant to the gradient and weight decay pulling it toward zero. I'm now looking at the layer-0 structure, hypothesizing that the input splits into a token subspace and a position subspace, with the previous-token head's query and key both reading from the position part. I'm also starting to wonder whether the value side plays a role there.

I'm checking whether the position subspace can be picked out from the per-head weight spectra. Heads 0.1, 0.4 and 0.5 have fast-decaying QK spectra, which suggests low-rank positional attention, while 0.0, 0.2 and 0.3 look broader and more like token matching. The QK symmetry of 0.86 in head 0.1 is a bit puzzling, but a small rotation angle at low frequencies would explain it.

I'm now checking whether the positional subspace is shared across heads 0.1, 0.4 and 0.5 and kept separate from the value subspace. The plan is to take the top four singular directions of each head's QK and compare the overlaps.



---

I'm checking whether heads 0.0, 0.2, and 0.3 are the positional heads, since they share the top QK subspace with overlap between 0.3 and 0.77. Positional information should live in a fixed low-dimensional subspace, so heads that share it would fit that role, and I'm now trying to work out which group is the positional one.

I'm checking whether the QK matrices are asymmetric enough to suggest rotation-like offset attention, which would support heads 0.0 and 0.3 as positional. Next I want to build a shared positional subspace from the top singular directions of heads 0.0, 0.2, and 0.3, then compare the complex eigenvalues of each layer-0 head's QK matrix to see if they encode small rotation angles.



---

I'm sorting the heads by their eigenvalue structure: heads 0.0, 0.2 and 0.3 show many complex pairs at large angles, which looks rotational and like positional heads, while 0.1, 0.4 and 0.5 have real positive eigenvalues, pointing to symmetric, similar-token attention. I'm also checking what rotation angles would imply for the positional frequency, since 70 to 100 degrees would suggest high-frequency components.

I'm weighing whether the rotation angles fit a prev-token head: head 0.3's 114-167° would suggest near-alternating behavior, not a clean offset-one pattern. The eigen-angle reading also depends on the basis, so I'm not sure the angles map directly onto positional frequency without checking the sign convention.

I'm working through the rotation-angle idea: a sign flip gives conjugate pairs, so the angle alone can't tell offset +1 from -1, though the causal mask removes the future side anyway. I'm checking whether a 180° rotation would favour odd offsets, and noting that the offset would be the rotation angle divided by the positional frequency, which I can't pin down without knowing the frequencies. So I'll look for better evidence in the K-composition instead.

I'm checking whether K-composition with the positional heads 0.0, 0.2 and 0.3 is weak, since their scores sit near baseline. A low composition score may just reflect a full-rank OV spectrum, so I'd rather measure how strongly each later head's query and key read the output of these heads directly.

I'm checking whether the earlier heads' OV circuits, composed with the later heads' QK, give a same-token match on the relevant subspace, and I'm planning to compute that composed matrix for each layer-0 to later-layer head pair.

I'm going to score each earlier-to-later head pair by whether the composed QK and OV matrix has mostly positive real eigenvalues, which would indicate a same-token match. I'll also weigh that against the overall magnitude so weak pairs don't look strong.



---

Checking attention head 3.2 against head 0.2: the match score is +0.79 with a magnitude of 0.22, the largest in the set. If 0.2 is a previous-token head, that would make 3.2 an induction head. Its QK spectrum is also flat, which points to high-rank token-identity matching.

Checking whether head 0.2 is a previous-token head, since the rest of the picture depends on it. If it is, then 3.2 looks like an induction head, given its strong match score, flat QK spectrum, and large OV output. Next I'll test whether the intermediate layer heads carry positional information through the residual stream as well.



---

I'm checking whether head 3.2's positive composition with those many OV heads actually means its QK·OV products stay positive, since later-layer OVs can carry negative copy scores in the residual basis.

I'm thinking that 3.2's key might be reading previous-token information from several layer-1 and layer-2 heads, so I need to find which of those are positional or previous-token heads. To test positional involvement, I'm planning to build a subspace P from the top singular vectors of the QK matrices of heads 0.0, 0.2, and 0.3, then measure how much of each later head's QK norm falls inside P compared with chance.



---

I've found a shared positional subspace of about 16 to 20 dimensions across the 72 vectors, so I'm setting P to the top 16. Next I want to check each head in every layer, measuring how much of its QK interaction falls inside P×P, and also looking at P on the query side against all keys.



---

I've established that heads 0.0, 0.2, and 0.3 are the positional ones, since they attend along the positional subspace and move token content from relative offsets, while the later heads barely use that subspace. So the positional signal in the deeper layers must be carried by other features. Next I'm checking which of the three is the previous-token head, and whether they each handle a different offset.

Checking the rotation angles, the 1:2:3 ratio suggests head 0.2 attends one token back, 0.0 two back, and 0.3 three back. To confirm, I'm going to compare the per-head matrices in the positional basis, looking for the block structure that scaling by the offset would predict.

I'm now testing whether the 1:2:3 rotation pattern holds for each head, by taking the eigenvectors of head 0.2's matrix and checking whether heads 0.0 and 0.3 rotate by twice and three times the same angles. I'm keeping in mind that these matrices may not be normal, so the test could give misleading results.



---

I'm working out the angles for positions 0.2, 0.0 and 0.3, and the ratio comes out to roughly 1 : 2.2 : 3.5, which fits offsets of 1, 2 and 3. That makes me fairly confident 0.2 points at the previous token.

I'm checking the sign convention next: under the causal mask, only backward offsets make sense, so 1, 2 and 3 are the offsets to the past. I also want to see that 3.2 reads the output of 0.2 and so acts as an induction head.

I'm checking whether other heads in layers 1 and 2 also pick up the output of 0.2, and they only show weak composition, so 3.2 stands out as the one real induction head. I'll also confirm that 3.2's OV circuit copies tokens, since its norm is large.

I'm weighing whether the layer-1 and layer-2 heads that read 3.2's neighbours might be aggregating content from layer 0 rather than positional offsets. Since the positional subspace doesn't account for them, I'd have to test their query and key composition with the earlier heads directly, and I'm planning to build a per-head table of those reads.

I'm now defining a per-head read score for each pair of heads: how much a later head's query, key, or value input lines up with an earlier head's output, normalised against a baseline. I also want to estimate the token subspace from the stacked value weights of the layer 0 heads, which should approximate the embedding space.



---

I'm checking the stacked value vectors for layer 0, where there's a sharp drop in singular values around 33-35 dimensions, so the layer reads roughly a 33-35 dimensional subspace. I'm wondering whether the token embedding's effective dimension is also around 35, and I'm working through how the vocabulary size would bound that, since embeddings can span up to V-1 dimensions after centering.

I'm weighing whether a knee near 33-35 points to a small character alphabet, like lowercase letters plus space and a bit of punctuation, though it could just reflect the model's chosen dimensionality. The noise floor sits below the init scale, so the directions above it are the ones the inputs actually use.

I'm checking which character sets would fit a knee around 33-35 dimensions, and running through known datasets: text8 has 27 characters, DNA has 4, Shakespeare around 65, and Cyrillic adds more. The knee sits a bit above 27 but below Shakespeare's count, and rare characters might contribute little to the effective rank, so I'll look at the block-0 QKV and MLP spectrum with the position matrix removed to see whether the gap is cleaner there.

I'm looking at the layer-0 value spectrum again and noticing the smooth decay toward roughly 0.5 at around 70 dimensions, which makes me wonder whether the query and key weights draw on the positional subspace beyond the 16 dimensions I already removed. I'm also considering whether the most complete view of token space comes from combining the values of all heads with the first MLP, since that MLP reads the layernormed sum of the embedding and the attention output.

I'm weighing whether a vocabulary of about 36 tokens would fit the roughly 35 dimensions, though layer normalization only removes the mean across features, so token span isn't simply capped at V-1. I'm also wondering whether the value vectors across all six heads together might reach further into token space than this single knee suggests.

I'm going to check whether head 0.2's value spectrum also has a knee near 35 dimensions, since that head has to copy the identity of the previous character, which would require distinguishing all the tokens.



---

I'm checking how much of the vocabulary signal the previous-token heads carry: head 0.2 uses about 20 to 23 strong dimensions, head 0.0 about 23, and head 0.3 about 29, with a union of around 35. Those counts don't yet pin down the vocabulary, so I'm considering whether the MLP0 input weights can say more, though the attention output spans the full space, which makes that hard.

I'm weighing whether the q-side of the content heads (0.1, 0.4, 0.5) could reveal something like attention to spaces, newlines, or vowels, but language identification from these weights looks close to impossible. English is the obvious guess given its prevalence, and the positional offset patterns don't distinguish languages. I'm also wondering whether the question expects calibrated uncertainty rather than a definite answer, or whether the structure of the token subspace (around 35 dimensions, not something like 4 for DNA) hints at a natural-language alphabet.

I'm considering whether the vocabulary size could be estimated from the spectrum of the last block's MLP output projection, since the final outputs have to land in the directions the unembedding reads, so a knee in that spectrum should sit near the vocabulary size. The first-block MLP input looks less useful, since attention output there already spans most of the residual space.

I'm checking the singular value spectrum of the last block's MLP output projection and attention output, expecting a knee near the vocabulary size since the final outputs have to land in directions the unembedding reads.



---

I'm noticing the projection output has a clear knee: strong values through about dimensions 27 to 33, then a steady decay. That pattern points to a small vocabulary, somewhere around 27 to 35 entries, which lines up with the knee I saw in layer 0's value projection. Next I'm turning to the final layer norm and how it feeds into the output.

I'm weighing whether the output knee at dimension 27 points to a 27-character alphabet like text8, though the knee near 33 and the layer-0 value knee near 35 suggest something slightly larger. Rare characters would show weaker dimensions, so the vocabulary could be around 30 to 40. A non-Latin alphabet such as Hebrew, with 27 letters once final forms are counted, is another candidate worth checking against the gap ratios.

I'm checking the singular values of the output projection, where the frequent characters should carry the biggest weights. The first 27 dimensions fall from 11.2 to about 4.0, followed by a small cluster and then a smooth tail that sits above the initialization scale, so it's probably trained rather than noise. Next I'll compute the ratios between neighboring singular values to find the sharpest drops.



---

I'm looking at the biggest gaps, at 33 and 27, and reckoning the effective vocabulary at roughly 34 to 40 once the final layer-norm centering takes one away. The layer-0 V knee also sits around 33 to 35, which points to a vocabulary of about 35. Now I'm checking which candidate sizes near 35 fit that, starting with text8.

The 27-dimension gap fits a ruled-out case here: softmax ignores constant shifts, so the usable unembedding rank is vocabulary size minus one, which pushes the vocabulary to at least 34. That makes text8 unlikely unless the gap is noise. A lowercase-with-punctuation alphabet would land near 40, and Shakespeare's 65 characters would leave room for weak directions among the rarer capitals, since case might be factored into a shared "capital" feature. That factoring would mean the effective rank sits below vocab minus one, so the bound is only an upper one.

I'm weighing whether the 33-to-35 range could come from a factored embedding rather than a plain alphabet. Next I'm looking for other structure, like the positional heads at offsets 1 to 3 and the content heads 0.1, 0.4 and 0.5, to see if the low-rank QK patterns reveal anything about the text.

Considering a small alphabet: Python code would bring in roughly 60 to 80 frequent distinct characters, which would push the effective rank well past 33, so code seems unlikely. A count of about 35 suggests lowercase-only text or some other specific dataset, and I'm trying to think of a known corpus with about that many characters.

I'm weighing whether a chess PGN movetext alphabet, with its roughly 33 to 35 distinct symbols, could match this effective vocabulary size better than the candidates I've already ruled out. Still, plenty of other sources could produce a count near 35, so the coincidence doesn't settle much. Next I want to check the positional subspace, the learned position embeddings, to see whether they reveal anything about the sequence structure.

I'm considering whether the effective vocabulary might come from a different source, such as a neuron-based estimate. The idea is that MLP0 neurons might each detect a particular token, so clustering their input weights projected onto the token subspace could give a count, though that only works if embeddings are nearly orthogonal, which requires a small vocabulary.

Now I'm checking the singular value spectrum of the third projection against the first-layer value weights, since both show a knee near 33, and I'm wondering whether a 64-character alphabet would really leave the unembedding using only about half its rank.

Now I'm comparing the singular value tail of the third projection with the second, since the third has a clear knee while the second decays smoothly. The trailing values sit well above what random initialization would produce, so those weaker directions look trained, possibly covering rarer characters, which would put the vocabulary near 65 with about 33 dominant.

I'm checking whether the layer-0 value knee at 33 to 35 holds up, and whether the proj3 tail beyond about dimension 40 is just a smooth decay rather than a real subspace. I'd like a cleaner estimate from the block 3 output-side singular vectors, to see whether they line up with the subspace the other layers write to.

Now I'm weighing whether case-factoring could make a Shakespeare-sized alphabet look like a token set of about 40, since a capital would just be a shared direction added to its lowercase letter. Punctuation, spaces and newlines would add roughly a dozen more distinct symbols, though the rarest ones might carry too little weight to show up in the rank.

I'm weighing whether the token-identity spectra of the value and projection weights point to a particular source. Natural-language text has a clear space character and word-boundary structure, which I could look for, but the evidence so far doesn't separate it from other symbol sets. For now I'll lean toward English with low confidence, around 0.3 to 0.4, and see whether further analysis can tell natural language apart from code.

I'm looking at whether the bracket and indentation structure of code, versus words in natural language, might show up in how the heads attend. Several head types, like ones that attend to the previous newline, are common to both, so they don't settle it. I'm also considering whether layer-0 QK patterns that connect a token to similar tokens could reveal character classes, or whether a key-only preference for spaces would be a better fit.

I'm going to look at the layer 1 to 3 heads next and label each one by what it attends to. For each head I'll check how strongly the query, key, and value paths read from the previous-token and nearby-position features, and whether the key bias gives it a clear preference for particular token types. I'll also check how much of each layer-3 head's output lands in the top 33 unembedding directions, since those heads would be writing logits straight to the output.



---

Checking whether the Frobenius composition scores separate the heads, since they sit near baseline, I'm not sure they say much. Head 3.2 puts most of its output into U (0.41), which fits a copying head writing token logits. I want the induction claim to rest on 3.2's OV mapping the token subspace read by layer-0 V into U, but without the unembedding I can't confirm the sign of the copy, so I'm looking at whether 3.2's V reads the token subspace directly.

I'm defining T as the top 35 right singular vectors of the stacked layer-0 V matrices, then checking what fraction of 3.2's Wv falls inside that subspace. I also want to see whether 3.2's Q reads T and whether its K picks up 0.2's output directions, using fraction-of-energy measures in place of the Frobenius composition scores.



---

I'm noticing that the layer 1 and 2 heads reading from the 0.1, 0.4, and 0.5 outputs read them heavily, and their output subspaces seem to overlap. That makes me think these heads write a shared feature, something like a position or distance-since-start signal.

Next I'm checking whether the layer 1 and 2 heads' output subspaces overlap, since that would confirm they share one position-like feature. The attention-sink idea also fits: averaging over context could give later layers a distance-from-start proxy. For the layer 3.2 head, the value reads the token, the key reads layers 0.0 and 0.2 (above baseline), and the query reads the token plus the previous-token output, which matches an induction pattern. I want to verify that against the eigenvalue results next.

I'm going to check the eigenvalue sign of the cross-OV-to-QK matrices for pairs of heads, to see whether the layer 3.2 query and key really match on previous-token content. After that I'll run the same kind of check on a few layer 2 heads, in case they also form induction-like pairs.



---

Looking at the attention patterns, the 3.2 head lines up as an induction head: each query token matches the key shifted back by one position, and the cross-position scores are strong. The same-token match is also high, though that could just be generic behavior, so I'm checking it against the other pairings.

Now I'm checking whether the 3.2 tok-tok score of 2.59 is dominating and whether the same-token match is undermining the induction reading. Other heads show no clear induction pattern: 3.5 is weak, and 1.1 and 1.3 have negative tok-p1 scores, which points to duplicate-bigram matching rather than induction. Layer 1 and 2 heads might feed induction through outputs from the 0.x heads, but that's unclear for now.

I'm going to measure how much of the 0.1, 0.4, and 0.5 QK score comes from the key-only bias term versus the input-dependent part, and check how much their output directions overlap.



---

I'm checking whether the two directions T and P actually overlap, since the overlap numbers sum to more than one, and they do, because the layer-0 value read space picks up some of the P directions. That's fine. Then I'm turning to heads 0.1, 0.4 and 0.5, which look like token-content heads with low-rank, symmetric QK and a query bias aligned to the top key direction, so they seem to prefer keys of one token class, with the query's token class shaping that preference.

I'm now going through heads 0.1, 0.4 and 0.5 as token-content heads: they seem to attend to all matching token classes in context, so their output looks like a presence or fraction feature. Then I'm sorting the layer 1–3 heads by their statistics, with the positive token-token and position-position heads grouped as duplicate-context heads, the negative ones as anti-duplicate, and 3.2 flagged as an induction head.

I'm weighing whether the vocabulary is something like text8's 27 characters or the larger set a case-factored alphabet would need. The layer-0 value space comes out around 33–35 dimensions, which is more than 27, so unless positional directions leak into the value read, text8 looks unlikely. I'm checking whether those extra dimensions are partly positional.

I'm weighing a vocabulary estimate: the strong output-side dimensions in proj3 run to about 33, and since logits depend only on the unembedding's row space and shift invariance drops one direction, the vocabulary would have to be at least around 34. Some of the input dimensions may be positional, so 35 doesn't rule out text8's 27 characters outright, but the output side is harder to explain away.

I'm checking whether the tail of the layer-3 output projection, roughly indices 40 to 60 with values near 1.0 to 1.4, is just leftover from initialization or actually trained. The initialization scale for that projection is much smaller than those values, so they look like learned directions, which makes me wonder how the input-side spectrum of that layer compares.

I'm wondering whether the tail of proj3 that keeps decaying past the first knee reflects rare characters with weak gradients, which would make the total vocabulary larger than the first estimate. But the sv curve has no second knee to pin it down, so I'm looking for a different way to count, such as a noise-floor comparison against a layer-3 reference matrix.

I'm checking the vocabulary size against Shakespeare-style character frequencies: a handful of characters dominate, and the rarer capitals and punctuation would form a long, weak tail. That would fit a spectrum with about 33 strong directions plus a faint remainder.

I'm checking the tail of the layer-3 output projection beyond the first 33 strong directions, looking for structure that would indicate trained rare-character directions rather than leftover initialization. The induction head 3.2 should write logits for every vocabulary character, including the rare ones, so its output span is a useful reference for whether the tail lines up with it.

I'm checking the full singular value spectrum of head 3.2's OV output, looking for a knee. Since an induction head copies whatever token it attends to, its rank should track the vocabulary size, though the head dimension of 64 would cap it at 64.



---

I'm estimating the OV rank for induction head 3.2 at roughly 22 to 25 dimensions, which seems low for a 65-character set. Assuming it mostly copies letters, a 26-letter alphabet with case-factoring would put the rank somewhere around 22 to 26.

I'm weighing whether the rank could tell Latin-script languages apart, since English is the likeliest prior but French, German, or Spanish would add accented letters and change the vocabulary size. Without character identity that seems nearly impossible, and the strong dimensions around 33 could fit many alphabets.

I'm settling on a Latin-alphabet language with English as the best guess, around 35% confidence. Checking whether the attention patterns reveal word boundaries is a possibility, since a space would dominate the token distribution, though it wouldn't settle which language it is. I also want to look at the rank of the previous-token head's output to see whether it, too, points to letter-level copying.

Next I'm labeling the layer 1 and 2 heads with more descriptive numbers, using the token-token and position-position eigenvalue signs. Head 1.2 and head 2.4 have high query-bias key preference (around 0.4), which suggests they attend to one specific token class like a delimiter. Heads 1.2 and 1.4 have negative token-token values, so they seem to avoid attending to the same token. Head 2.3 has a large OV term but a small QK term, which points to a broad averaging head that summarizes context. Head 3.2 remains the induction head, and 3.0, 3.1, 3.3, 3.4, and 3.5 write moderately into the unembedding with low-rank OV circuits.

Labeling the layer 1 and 2 heads next, with the attention and output roles tied to their eigenvalue patterns. I'll close with the note that the labels are tentative.

