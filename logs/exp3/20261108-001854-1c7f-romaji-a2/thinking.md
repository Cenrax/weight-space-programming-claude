I'm working out how to analyze the weights without running any forward passes, which makes this a hard problem. One sticking point is that identifying the language without embeddings is difficult. I'm wondering whether the weights, such as the layer-norm scale in the first block, might still reveal something about the embedding dimensions, though I don't have the embeddings themselves.

Now I'm checking whether the weights themselves leak structure, like the q/k projections in layer 0 possibly showing positional patterns, even without the embeddings. I'm also wondering whether the vocabulary size leaves a trace in the rank of the block 0 weights, since gradients only reach the directions spanned by the inputs. Adam's elementwise updates may blur that, so I'll just look at the singular values of the block 0 matrices and the layer-norm weights to see.



---

I'm going to check the singular value spectrum of the block 0 query, key, and value weights, scaled by the layer norm weight. The effective rank there might tell me something about the vocabulary size and the positional dimensions.



---

The last singular value comes out near zero, which fits: the layernorm output sums to zero and the weight decay has shrunk the directions the model doesn't use. Layer 0's v projection drops off sharply somewhere around the 20th to 40th values, so I want to see the full spectrum to judge its effective rank.

I want the rank estimate from the spectrum itself, so I'll print the layer 0 v singular values for indices 0 through 80 and look for where the knee falls relative to the vocabulary size.



---

I'm checking the value projection in layer 0, where the singular values show a knee around 33 to 35, so the token embedding subspace might be about 35 dimensions. I'm weighing whether that matches vocab size minus one, given that the positional component also contributes after layer norm.

I'm narrowing down the vocabulary size from the layer 0 value knee, which points to roughly 33 to 35 significant directions. I'm considering what kind of character set would produce that count, and I'm also noting that rare characters might carry small weights and hide the true vocab size.

I'm considering a different route: checking whether the query and key weights in layer 0 share the same token subspace as the value weights. The idea is that the top roughly 33 right singular vectors of the value projection should line up with the embedding span, while any positional directions might sit apart from it.

I'm working out how the combined spectrum of the layer 0 query, key, value, and MLP weights might separate token directions from positional ones. The gaps after the first 33 or so values suggest a vocabulary near 34 to 36, though the layer norm may shift that count slightly.

I'm weighing whether rare characters, which get little gradient and so small singular values, would make the knee undercount the true vocabulary, and I'm wondering if a non-English alphabet could be behind the count.

I'm looking at the per-head value dimensions next, since each head covers 64 dimensions and I want to see whether the strong directions cluster within particular heads.

I'm now thinking about combining the reading matrices of layer 0, the query, key, value, and MLP weights scaled by the layer norm gain, into one stacked matrix. The spectrum of that combined matrix should show how many input directions the layer actually uses.



---

The spectrum shows about 33 strong directions, then two weaker ones at 0.35 and 0.34, before dropping off to 0.27, so I'm estimating around 35 token directions. Next I'll take the top-35 subspace of v and see how well the other matrices line up with it. On the language question, the candidates run from English and French through German, Spanish, Russian, Chinese, and Python code. Those are hard to tell apart on this alone, so I'm wondering whether the positional structure, something like line length in poetry or fixed-width text, might settle it.

Still weighing whether the weight structure can reveal the language at all, since without embeddings everything is only defined up to rotation. I'm wondering if the vocabulary size itself gives a clue, and the sharp drop at 33 directions might point to an alphabet of that size, maybe Russian.

I'm going to check the positional side next: the directions the query and key use that the value doesn't, since that could show how position is encoded. Without positional embeddings I can't pin down attention offsets directly, so I'm looking at composition scores between heads to spot previous-token and induction-style patterns instead.

I'm now looking at whether the induction head's query-key circuit reads from a previous-token head's output, which I'd check by scoring the composition between their weight matrices. The copying behavior is harder to pin down without the unembedding, so I'm considering whether positive eigenvalues of the output-value product on the residual stream could stand in for it.

Now I'm looking at how the previous-token and induction heads might be linked through composition scores. I'm thinking about measuring how much of the query-key structure sits in the positional subspace rather than the token one, and checking whether the induction head's key reads from the previous-token head's output.

Next I'll work through the per-head statistics, folding the layer-norm gains into each head's matrices. I'm also still weighing whether the vocabulary structure or some positional pattern like word boundaries could reveal the language, but I'll come back to that after the head analysis.



---

I'm sorting out which heads look positional and which look content-based in this layer: heads 0, 2 and 3 seem to track position, likely previous-token or local heads, while 1, 4 and 5 attend by token content. Next I'm trying to isolate the positional part of the QK space for those heads, the part outside the token span, though without knowing the offsets it's hard to pin down directly.

I'm trying to reconstruct the positional embeddings from the QK matrix, but the attention structure alone doesn't reveal the positions. So I'm considering K-composition with layer 1 to see which layer 0 heads feed induction heads. Heads 0, 2 and 3 read token identity through their OV outputs, which fits previous-token behavior, but telling offset -1 from -2 may depend on whether the rotation angles in their QK subspace double.

I'm now looking at the composition scores for pairs of earlier and later heads, measuring how much a layer-0 head's output feeds into a later head's query, key, or value. The baseline to compare against is roughly what random weights would give, which I'll need to compute across all pairs.



---

Head 0.1 shows strong K-composition with many heads, yet it is a content head, so the link may be generic. Its OV output might be writing a constant, bias-like direction that the key bias picks up. The Frobenius-norm score is also dominated by the top singular directions, so I need to be careful about reading it.

I'm wondering whether head 0.1's output might write into the directions that layer-1 keys read strongly, maybe a positional-style signal. Before going further, I want to check the dominant singular direction of layer-1 K and whether 0.1's OV writes there. My earlier token-subspace estimate could also be contaminated by positional dimensions, since V reads position to some degree.



---

I'm looking at head 0.1 and whether it acts as an attention-sink or a character-attending head that feeds keys to later layers, or possibly a previous-token head. Its value-output reads the token identity, and its query-key pattern seems driven by a strong query bias, so I want to check whether that bias alone sets its key preference.

I'm weighing a different way to separate the positional and token parts: take the top singular vectors of the concatenated QK for heads 0, 2 and 3 as a positional subspace, then measure how much of the value side falls outside it.



---

I'm treating heads 0, 2, and 3 as positional, since their eigenvalues are high-rank and rotation-like. Head 2's eigenvalue angle comes out around 0.6 to 0.7 radians, and head 0 has several angles spread between about 0.4 and 1.7. Head 3 sits near 1.98.

I'm checking whether the angles follow a pattern of offset times frequency. If head 2 is near offset -1 and head 0 is near 2 while head 3 sits near 3, the eigenvalue angles should scale by those multiples. Projecting head 0 and head 3 onto head 2's eigen-planes would test that directly.

I'm working out the algebra: if the key is a rotated copy of the query, the score peaks when the matrix matches a rotation raised to the offset, so its eigen-angles come out as offset times frequency. To test that, I'll take head 2's eigen-planes and measure the rotation angle of heads 0 and 3 restricted to those same planes.



---

I'm checking whether the angle ratios across the four cases come out close to 1 : 2.3 : 3.7, and they land near that but not exactly on 1:2.

I'm working through the ratios, and they come out near 1 : 2.3 : 3.7 rather than exactly 1:2:3. I'm wondering whether the heads are mixing offsets, which would shift the values. Since no head can point forward and offset 0 would give zero angle, the smallest offset has to be about 1, so the head attends to the previous token.

I'm weighing whether the head at offset 1 accounts for the non-integer spacing of the other heads' offsets, since positional phase isn't perfectly linear. Next I want to check composition with the prev-token head by removing the dominant direction first, because the big 0.1 component is swamping the Frobenius-based comparison.

I'm considering whether head 0.1 might simply attend to a fixed position or character and emit a near-constant output, which would make it act more like a bias. To test that, I want to check whether its OV matrix is close to rank-1, meaning it writes mostly one direction.



---

I've found that the residual-space OV eigenvalues are negative for layers 1 through 3, meaning those heads write in a direction orthogonal to what they read. So the residual trace won't tell me anything about copying here, since the output subspace used for predictions differs from the input token embedding. I need a different route to find the induction heads, probably by looking at K-composition with the previous-token head 0.2.

Next I'm checking how much of each later head's key-read sits in the subspace that head 0.2 writes to, with the top 30 directions of its OV output compared against the expected 30/384 baseline. I'll also run the same check on the 0.0 and 0.3 outputs and on heads 0.1, 0.4, and 0.5.



---

I'm checking whether the output subspaces of heads 0.1, 0.3, 0.4 and 0.5 overlap, since they seem to share some bias-like directions. Many of the K subspaces also line up with heads 1, 3, 4 and 5, which I didn't expect. I'm wondering if they all share the residual stream's dominant direction, possibly because layer-0 output writes into the embedding space as well.

I'm working out the output directions by restricting the OV circuits to the token subspace, so I can estimate what previous-token head 0.2 writes into the residual. Then I'll check how much later heads' keys read from those directions, comparing against a random baseline and the raw key-weight norms.



---

I'm checking whether the T subspace actually lives in the residual space, since the LN scaling matters here. The later layers' keys look like baseline, which would mean they don't read the token-embedding subspace, and that seems strange. Maybe T, pulled from layer 0's values, isn't the embedding subspace after all.

I'm working out how the residual-space directions carry over through the layer norms, which means later heads' keys mostly don't read the token subspace except via head 0.1's output. Head 0.1 looks like content-based attention with a positive OV trace, and it feeds into most layer 1 and 2 keys, so I'm wondering whether the induction mechanism is real but hidden under the larger singular directions.

I'm considering whether the MLP0 outputs, rather than the embeddings, dominate what the later layers' keys read, so I'd want to measure the variance each writer contributes. I'm also wondering whether head 0.1 might be the real previous-token head, even though its content-based attention pattern doesn't obviously fit that role, and whether the positional embeddings share the subspace I've been calling T.

I'm weighing whether head 0.1 is an averaging head that outputs a near-uniform context vector, since its QK is small. If so, keys reading that average would be position-dependent, which seems an odd thing for later heads to rely on.

I'm checking whether layer 1 and 2 keys read the output of head 0.1, using the top read direction of layer 1's K, which lines up closely with that output. I'm also wondering whether head 0.1 attends to something like a first token or newline, which would encode position within a line, and whether that matters for poetry or code.

I'm checking whether layer 1 and 2 QK attention is rotation-like, which would point to positional structure, by looking at the eigenstructure of the 1.x heads.



---

I'm looking at the flat singular values of the QK matrix, which suggests high rank and a strongly positive-definite form, so the heads attend to positions whose residual matches the query's. But for an induction head, the query and key read different subspaces, so the QK map isn't symmetric in the residual basis unless those subspaces coincide.

I'm checking which earlier positions the QK circuit prefers, and the 3.2 composition points to matching the previous token: the query reads the current token's prior context and the key reads the same for earlier positions. That would be an induction-style head, copying from the position right after a match.

I'm still working through whether a fully symmetric QK would make the head useless, since matching context at s would just reproduce the current token. An induction-style version would need the key at s to carry the previous token, with the prev-token OV writing it into the subspace where the query reads the current token, and the positive trace of that OV is what would make that plausible.

I'm now thinking about decomposing the QK circuit by block: for layers 1 to 3, I want the bilinear strength between each query-side writer and each key-side writer, normalized. An induction signature would pair the current token on the query side with a previous-token head's output on the key side. The embedding subspace may be tangled with positional information, which complicates reading the blocks cleanly.

I'm going to normalize each block of the QK matrix by the norms of the query-side and key-side writers, with a random baseline to compare against. The embedding writer is the orthonormal token basis, while the head outputs carry token information through their OV maps. The MLP0 output is too broad to treat as a single block, so I'll leave it out for now.



---

The heads from different layer-0 blocks give nearly identical subspace scores, which points to a shared dominant subspace rather than separate roles. That suggests a few large directions, maybe the strong one in layer-1 K, are being written to by many heads at once, so I need to strip those shared directions out before comparing them. I'm starting to pin down which top directions to remove.

I'm checking how much each layer-0 head writes into that top direction of layer-1 K. A cleaner route might be to project out the strongest directions of layer-1 Wk and recompute. Layer 1 also has heads with complex eigenvalue pairs that look like rotations, so I want to test whether those planes sit in the positional subspace from layer 0.



---

I'm noting that later layers don't read the positional embeddings or the token subspace directly, so they must be pulling from layer-0 outputs through the heads and MLP. I wonder whether the embedding subspace gets erased or rewritten along the way. The Ktok figure looks suspiciously close to baseline, and I'm considering whether the embedding simply has a small norm that LayerNorm then amplifies.

I'm working out which writers feed the layer-1 heads' rotation planes, so I'll look at each layer-1 head's top plane against the layer-0 head and MLP0 outputs. Rather than restricting to the token subspace, I'll use the full write space of each head, since value vectors can carry positional information too. I'm planning to compare write and read subspaces through the principal angles between them, weighted by singular values, and check that against a baseline.



---

I'm reading off the head attention weights for layer 3, head 2: its keys draw on heads 0.0 and 0.2, its queries on 0.2, and its values on 0.4 and 0.5. Something about the value side looks off to me, since it seems to pull from layer-0 content heads, so I'm pausing on that. Moving to layer 1, the queries seem to rely on the positional heads 0.0 and 0.3, while the keys depend on 0.1, which suggests the query side is tracking tokens two or three positions back.

I'm checking whether the later layers read the token embedding subspace at all, since the attention reads after layer 0 look flat on it. I'm also wondering whether head 0.1 attends to the current token and re-encodes it, which would explain why it's needed.

I'm now considering whether head 0.1 attends to the same token it sits on, which would make it a same-token head that re-encodes the current token into a new subspace. Layer-1 keys would then read that re-encoded current token while queries read tokens two or three back, which looks like an offset variant of induction. I'm weighing whether the reverse arrangement, with the current token on the query side, might fit better.

Still checking whether the positional heads 0.0 and 0.3 land on offsets around two to four, with head 0.2 perhaps at offset one.

I'm checking whether head 0.1 attends to itself, since its QK eigenvalues are small and positive, and estimating the score scale to see if an aligned token direction would produce a strong match.

I'm checking whether head 0.1 attends mostly to the current token, since its positive QK eigenvalues would make it a same-token head. If so, its output would be roughly a function of the current character, which would explain why later layers read it.

Now I'm wondering whether query-side layer-1 heads reading 0.0 and 0.3 might carry positional info that later layers rely on, since the positional embeddings themselves don't seem to get read directly.

Now I'm checking whether head 3.2 could be an induction head, with keys reading the previous-token outputs of heads 0.0 and 0.2 and queries reading current-token information from 0.2.

I'm looking at whether head 3.2 is an induction-style head, with its keys reading the outputs of the previous-token heads 0.0 and 0.2 and its queries matching that same previous-token signal. If so, it would attend to positions whose preceding context matches the current one, and the copied content would come from the same-token heads 0.4 and 0.5.

I'm checking whether head 3.2's keys, which read the outputs of heads 0.0 and 0.2, line up with the query's previous-token signal, so that a key at one offset matches the query at another. Its query also reads the current-token embedding, which could make the match a two-offset induction pattern.

Now I'm checking whether the layer-1 heads with keys reading 0.1 and queries reading 0.0 or 0.3 form another induction-style pattern.

I'm looking at head 2.4 and head 3.4 next, where 3.4's keys read the output of 2.3 and its queries also read 2.3, and 2.3 itself has a large output-value contribution that I want to check.

I'm checking whether layer-1 heads contain a previous-token mechanism of their own, since their top QK eigenvalues look rotation-like, which hints at a positional signal. I'll look at each layer-1 head's complex eigenplane to see which earlier writers carry that positional information.

I'm now computing, for each layer-1 head, the share of its top two-dimensional eigenplane that falls inside each earlier writer's output subspace, the layer-0 heads' Wo and the M0 top subspace, to see which of them carries the positional signal.



---

I'm checking how much the heads' write subspaces overlap, since a baseline of 64/384 is about 0.17 and many pairs sit well above that. The layer-0 output columns look like they share a common region, and I'm wondering whether that means they all write into the same low-dimensional space. Looking at layer 1 now, there's a shared top eigenplane at an angle around 1.3 rad, sitting in a common region spanned by the 0.1, 0.4, 0.5 and 0.3 values.

I'm checking whether the top planes across layer-1 heads, at angles like 1.0 through 2.4, actually coincide. If they do, the shared 2D rotation could map the query direction onto a rotated key direction, possibly a massive-activation feature. I'll test how much those planes overlap.



---

I'm looking at layers 1 and 2, where several heads seem to share a rotation structure in one low-dimensional subspace, with overlaps well above baseline. That subspace looks like it's being written by a set of specific dimensions, and the rotation angle is consistent across many heads, which makes me think it's a positional mechanism.

I'm weighing whether the positional signal comes from layer-0 heads with near-uniform attention, since their outputs would shift with position in a predictable way. A rotation-based relative code in a few dimensions could explain how later heads attend by offset, though a recency mechanism with separate query and key directions is another option. This is getting tangled, so I want to set priorities with the remaining tool calls: first figure out what the language signal looks like, then sort out the head roles.

I'm checking the value spectrum for layer 0: about 33 strong directions stand out, with a bulk of smaller values that looks like random-init remnant. That points to roughly 35 learned token directions, which would fit a small character vocabulary. I'll keep that in mind as a rough vocab estimate, though rare characters may not have learned their own directions.

I'm weighing whether 35-40 directions points to a lowercase alphabet, since uppercase letters would normally add many more learned directions. Protein sequences come close in size but still fall short of 35, and code character sets run well above that.

I'm checking whether the line-structure idea holds up: a head that attends to the same position in the previous line would show a rotation angle matching the typical line length, which would point toward poetry or fixed-width code.

I'm weighing whether a knee near 35 directions can tell me the language: a logographic script would fill the full 384 dimensions, so this one is alphabetic with a small character set. That still leaves a guess about the language itself, which I'd put at English with low-to-moderate confidence.

I'm checking whether the knee in the value spectrum, with about 33 strong directions falling off sharply to a flat bulk, could simply reflect the frequent characters of mixed-case English, since uppercase letters are rare enough to leave only weak directions.

I'm checking the top singular value in layer 0's MLP, which is large enough that the dominant "space" character might be the direction it reads. I'm also considering whether the same-token heads, with only a handful of significant eigenvalues, are sorting characters into a few classes like vowel or space, which would point toward an alphabetic language. Next I want to try to pin down the space direction, since space is the most frequent token in most text.

I'm thinking the dataset could be something like Python code, but code would use more punctuation and push the learned directions well past 35, so that seems less likely. Other candidates are Shakespeare-style text, TinyStories, Wikipedia in some language, DNA, or math, and the 35-direction knee fits a small lowercase-heavy alphabet better than any of those.

I'm weighing whether the language can be read from the weights at all, since things like word-length patterns, character entropy, and how strongly induction heads fire might leave a trace. Shakespeare-style text would push the vocabulary well above 35 because of frequent capitals and speaker names, so that doesn't fit the spectrum I'm seeing.

I'm settling on English at about 0.4 confidence, since the letter-and-capital sharing in the embeddings means counting directions only gives a rough vocabulary size. Shakespeare's caps would make the rank larger, so I'll look for finer positional structure next, including whether any low-frequency component reaches beyond local offsets.

Checking the positional geometry against candidate periodicities: fixed-width lines or codon-period-3 DNA would each leave a distinct frequency, and the observed rotation angles around 2.0–2.7 don't clearly match codon structure. DNA's tiny alphabet also contradicts the roughly 35 directions I'm seeing, so I'm leaning toward an alphabetic text with a small character set.

I'm checking whether the MLP in layer 0 reads token directions beyond the roughly 35 the value spectrum shows, by measuring how much of its weight sits outside the token and positional subspaces.



---

I'm seeing that MLP0's input is the residual stream after attention, so it naturally picks up a lot of signal beyond the token and position embeddings. The query and key vectors also have extra learned directions beyond those, likely more positional dimensions. I'll now look at the attention heads' roles with a more principled method, and try to settle the language question with moderate-to-low confidence, possibly by checking how the positional heads interact with word length.

I'm weighing whether a head might key on word starts, such as spaces, and I'll also check the frequency and singular-value patterns, though those look inconclusive. English still seems most likely, around 0.35 to 0.45 confidence. But the 33 to 35 dimensions could fit a 27-character lowercase set like text8, with the rest coming from positional reads through V.

I'm checking whether the 35 directions V reads from the token side match a character set, and whether those directions overlap the positional subspace. I'll look at each head's value spectrum, since a prev-token head should show a knee near the vocabulary size.



---

I'm estimating the effective alphabet size from the attention heads' strong dimensions, landing somewhere around 25 to 35 directions, which fits a set of lowercase letters plus a few extras.

I'm weighing whether the top character's frequency and the dominant singular direction in layer 0 could reveal anything more specific than a general Latin-script alphabet, and I'm leaning toward English with moderate confidence.

I'm checking whether head 0.2 is truly offset 1 or mixes in self-attention, since its angles are about 2.3 times smaller than head 0.0's. The gap isn't a clean integer ratio, so the positional offsets may be shifted by the learned embeddings.

I'm checking whether head 0.2 really is a clean previous-token head rather than a mix with self-attention, and it fits that role best. Head 0.0's angles are about 2.3 times larger, so it attends further back, probably around offset 2 to 2.5. Head 0.3 looks like it sits near offset 4.

I'm now checking the layer 3 heads for induction behavior, since head 3.2's query and key look like they read the previous-token outputs of heads 0.0 and 0.2. To test that more carefully, I'm measuring how strongly each candidate query writer's output aligns with each key writer's output through the QK matrix, restricted to the token-identity subspace.

I'm setting up a test for induction: comparing how the query-side token reading lines up with the key-side reading that the previous-token head copies, using a token-by-token matrix in the shared token basis.

I'm now testing the induction hypothesis with the token-coordinate matrix C = T·QK·OV_0.2·Tᵀ, checking whether its trace is positive relative to its norm, since that would mean the query and key sides match on the same token. I'll repeat the check for the two-back head 0.0 and for the previous-token pairing that a head like 3.2 would need.

I'm now scoring each head's query and key sources with a normalized trace of the token-coordinate matrix, where a value near 1 would mean the two sides match on the same token, and I'm also tracking the matrix norm to see how strong each pairing is.



---

I'm reading the attention pattern of head 3.2 as an induction head: the query at the current token matches the key one position back, which points to the token that followed the earlier occurrence. The second query-key pair shows the same structure one step earlier, two positions back.

Checking the remaining attention scores for head 3.2, the three-character offset also shows up, but the diagonal terms, where query and key sit at the same offset, don't fit a pure induction pattern. They may just reflect overlap between heads that attend at neighboring offsets.

Now I'm checking the layer 1 and 2 heads, where the query rows at 0.0 and 0.3 show strong scores against the key columns at 0.0, 0.2, and 0.3, mostly in the 0.41 to 0.48 range.

Now I'm comparing the layer 1 and 2 heads, where the query at tok[t-2] and the one further back match against keys at the previous one, two, and current positions.

I'm working out how the layer 1 and 2 heads might find the previous token by matching content: the query at tok[t-2] lines up with the key at tok[s-1], which points to s = t-1. The smaller values in these heads are still worth noting against the stronger ones in the other block.

I'm checking whether the layer 1 and 2 heads are doing the same content matching at an offset, so the query at t-3 pairs with key s-2 and points back to s = t-1. That would make them attend near the current position rather than copy anything useful for predicting t-1, unless their values carry something else.

Now I'm checking whether layers 1 and 2 match on content at each offset: the query at tok[t-2] against the key at tok[s-1] would point to s = t-1, which means these heads can find the previous token without any positional signal. Extra matches from the other offsets should sharpen that pick toward s = t-1.

Now I'm checking the layer-1 heads by offset: a query at offset a matched against a key at offset b points to s = t − a + b. For head 1.3, most of the strong matches land on s = t−1 or t−2, so it looks like a content-based previous-token head.

I'm looking at a dimension-0.1 pattern: large magnitudes but no real query-key match, so it might be a rotation-like plane carrying positional information for later heads. I'll check the 2.3–2.5 rows next, and also which writers the induction candidate 3.2 reads from in its value path.

Now I'm pulling the remaining rows for heads 2.3 through 2.5 to see which layer-1 outputs they read from.

I'm trying to recover the other source-token pairs for layers 2 and 3 by composing the earlier heads' value paths, since layer-1 outputs don't live in the token basis directly.

Now I'm planning to match up sources from layers 2 and 3 against the composed layer-1 outputs, scoring each pair across the 28 candidate sources and keeping the strongest matches.



---

I'm noting that the later layers read residual features I can't break down linearly, since the MLP outputs get in the way. Head 3.2 looks like an induction head, while the other layer-3 heads read mostly from layer-2 outputs, especially 2.3, which I can't map well. Next I want to go back over the layer-1 heads and work out their effective offset by weighting the shifted-matching scores by relative offset.

I'm working out the offset aggregation for the layer-1 and layer-2 heads now: for each head, I'm summing the positive match scores by implied relative offset, and setting aside any negative implied offsets since they can't be attended to causally. The anti-matching negatives are worth checking too, to see whether the heads avoid self-attention. I'm not able to decode what the 0.1 rotation does in those heads, so I'll leave that open for now.



---

I'm reading the 3.2 attention pattern as strongest at a relative offset of -1, which would mean the key's preceding context lines up one position ahead of the query, the classic induction alignment. I'm also weighing a same-alignment match at offset 0, where the key's context lines up with the query's context directly.

I'm working through the layer 1 and layer 2 heads, which seem to match content at a relative offset of 1 to 2. Their dominant effect looks local, mostly attending to the immediately preceding position, so they behave like previous-token heads rather than doing useful copying.

I'm looking at whether the layer 1 and 2 heads might be doing something more than local matching, since the shared rotation in layer 0's output space could encode relative position. If a position-dependent signal is rotated by about 1.3 radians, that could pick out an offset of roughly two, which would make these heads positional rather than purely previous-token.

I'm now checking whether the layer 1 and 2 heads' positional signal could come from the shared rotation plane, with the phase near 1.2 to 1.5 radians implying a fixed relative offset, though I can't pin down the per-step frequency without the positional embedding.

I'm checking whether the positional planes that layer 0's heads use can be pushed through the OV circuit of head 0.1, so that the shared rotation in layers 1 and 2 maps onto a known per-step angle. If that works, the phase of the layer-1 rotation divided by that per-step angle should give the relative offset those heads are reading.

I'm testing whether the layer-1 head's rotation plane matches the image of a layer-0 positional plane under the OV circuit of head 0.1, comparing the rotation angles against the per-step angle to get the relative offset.



---

The positional planes seem to carry almost nothing through the 0.1 OV path, so the shared rotation can't be coming from them. That makes 0.1's output in this plane look token-driven instead, and I'm weighing whether layer-1 and layer-2 heads match query and key features from the same token with a rotation keyed to a character class.

I'm reasoning that the rotation could come from content-based attention, where a query character class attends to keys of a related class, such as the most recent space or vowel. The uniform angle across the eight heads makes me think 0.1's output is effectively a function of the current character's class, pointing to a single shared two-way feature like space versus letter. With limited budget left, I'm considering whether a periodic consonant-vowel alternation pattern could explain it, though that's not conclusive.

I'm checking whether the 0.1 QK eigenvalues are dominated by about three dimensions. I'm also weighing whether this might be code rather than prose, since code tends to show many induction heads, while here I only see one clear one at 3.2. Next I'm reconsidering the vocabulary size from the V-rank count, since a mixed-case English model would need more distinct directions than the roughly 35 I'm seeing.

I'm now wondering whether the vocabulary is actually small because the language has no case, like Arabic or Hebrew, which would put the character count around 35-40. The uppercase letters in English would be rare, so their directions might not show up strongly, but the spectrum still needs a closer look before I can tell.

Now I'm checking whether the spectrum's smooth tail fits a random-matrix quarter-circle distribution, since a few faint rare-character directions could be hiding in the 0.27 to 0.22 range.

I'm now checking whether the tail of the singular-value spectrum, from around index 100 onward, follows the quarter-circle shape a random matrix would give, and whether that shape predicts where the bulk should top out.



---

I'm counting the learned directions in V: about 35 clearly stand out, with a few more just above the random baseline, so roughly 35 to 40 in total. That would put the vocabulary somewhere around 36 to 41, though I'm wondering whether some positional dimensions are mixed in.

I'm checking whether the vocabulary could be lowercase English plus punctuation, which would fit the roughly 35 to 40 effective tokens. I'm also wondering if the top key direction, at 3.586, is a token-identity feature picking out something like spaces, and I want to see whether it lives in the token subspace or the positional one and which heads use it.



---

I'm noting that one direction scores high on both T and Pp, so the two aren't separate here, and every head's key leans toward it. That pattern looks like an attention sink, probably tied to position 0 or a very common token.

I'm at 21 calls, so it's time to settle the head labels and run a couple more checks. Heads 0.4 and 0.5 have small OV and positive QK eigenvalues, with 0.5 strongly favoring one token class. I want to test whether the content heads attend to the same token by computing a match score over the layer-0 heads.



---

Attention heads 0.1 and 0.5 look for earlier positions that share the current character's class, with 0.4 doing something similar but weaker. Next I'm turning to the layer-1 and deeper heads, starting with 1.0, 1.1, 1.3 and 1.5, whose queries read earlier head outputs from a few tokens back.

I'm now weighing whether heads 1.3 and 1.5, which lean toward the position just before the current one, really count as previous-token heads. They read their query from earlier-head outputs and match content back a step or two, so attending to t-1 and t-2 is almost automatic. Their measured weight on the immediate predecessor is the strongest, but the 0.1-to-0.1 block values are much larger, so I can't yet tell how much these heads actually contribute.

I'm wondering whether later heads get positional information some other way, since they don't seem to read the positional embedding directly. Layer 0 head 0.2 is the clear previous-token head. For layer 1, heads 1.3 and 1.5 might also mostly attend to the prior position, though their attention splits between one and two tokens back, so I'm unsure whether to count them.

I'm weighing whether the 0.1 block could be doing something closer to uniform averaging, since its low query-key scores would make its output a running mean over the prefix. If so, that mean shifts smoothly with position, which could give later layers a coarse positional signal they can use for recency.

I'm weighing whether to list the layer-1 heads 1.3 and 1.5 as previous-token heads, since their attention splits between one and two tokens back. Head 0.2 is the clear candidate, and I'm considering whether to mention 0.0 and 0.3 as positional heads that don't attend to the immediately preceding position.

I'm checking whether layer-1 queries could locate the immediately preceding token: they only see tok[t-2] and tok[t-3] directly, and the key at position t-1 carries tok[t-2] via head 0.2, so a content match would still pick out t-1. Head 0.2 is clearly the previous-token head, and the layer-1 heads only approximate that behavior indirectly.

I'm going to settle on head 0.2 as the only clear previous-token head, since it reads the prior token directly. The layer-1 heads only reach the immediately preceding position indirectly through content matching, so I'm leaving them off that list.

I'm weighing whether to add layer-1 heads 1.3 and 1.5 as previous-token candidates, or keep 0.2 as the only clear one, and I'll note them as secondary. Next I'm checking whether any layer-2 or layer-3 heads read the layer-1 local heads and form induction-like patterns.

I'm checking whether layer-2 and layer-3 heads read layer-1 prev-token outputs in a way that makes them induction-like. The matches I've seen so far are symmetric, with query and key reading the same source, so they don't look like induction. Head 3.4 reads 2.3 on both sides, which is also symmetric, so I'm going to check the value side next.

I'm checking whether head 3.2's output lines up with the logit-relevant directions, but I can't verify that without the unembedding. Next I'm testing whether head 3.4's query and key pick up what head 2.3 writes, by comparing those matches against the raw embedding and against the 0.2-routed signal.



---

I'm noticing that heads 3.0 and 3.4 (with 3.1, 3.3, and 3.5 weaker) mirror what head 2.3 outputs, which points to a same-alignment match on a feature 2.3 produces. Head 2.3 doesn't seem to match tokens; it may be a broad context-summary head that tracks things like word or line starts, so 3.4 could be keying on positions with similar 2.3-derived context.

Still checking head 2.3: its keys and queries both read from layer 1 head 1.4, and its QK has a small, negative top eigenvalue, but its OV output is the strongest in layer 2. That makes it look like a broad averaging head that builds a context summary for layer 3 to match on. Next I'm going through the layer 0 and layer 1 heads to pin down what each one attends to.

Now I'm going through the layer 1 and layer 2 heads: most of the layer 1 heads look at the previous one or two tokens, 1.2 and 1.4 track character classes, and 2.3 looks like a broad context aggregator feeding layer 3. Layer 3's 3.2 is the induction head, and 3.0 and 3.4 seem to key on what 2.3 produces.

I'm checking the layer 3 heads 3.1, 3.3, and 3.5, which all show weak same-token matching, and 3.3 has one dominant direction in its QK spectrum. I'm not treating 3.0 or 3.4 as induction heads, since nothing in the evidence supports that.

I'm wondering whether the rank structure of the previous-token and character-class heads, like 0.1 and 0.5, could tell me something about the language. Vowel and consonant distinctions would show up in any alphabetic language, so that alone doesn't narrow it down. The token count might help more: a lowercase-only English set has around 27 symbols, typical English has over 65, and mixed-case data like TinyStories would land somewhere in the 35–45 range.

I'm weighing whether the vocabulary size could point to a language: Russian, German, and French would all land around 40 frequent symbols, just like English, so the count doesn't separate them. Since most datasets are English, I'm leaning that way at about 0.4 confidence. I'm also wondering whether the singular value spread of the value vectors might reflect letter frequency, though it doesn't seem to discriminate either.

I'm considering whether the architecture points to a known setup: 6 heads, 384-dim embeddings, and a 256-token context match nanoGPT's shakespeare_char config, though this model has 4 layers rather than 6. That would suggest a nanoGPT-style character model, but I'm not sure whether that config uses bias, so I can't confirm the match from what I remember.

I'm recalling that nanoGPT's default config sets bias to True, though I don't think the shakespeare_char script changes it. This model also uses separate q, k, v, and o projections instead of nanoGPT's combined c_attn, so it's a custom variant. The design might come from a set of models trained on different languages with a shared config, and the answer could be English prose, Python, Shakespeare, French, or something else.

I'm weighing whether the vocabulary size of this model could point to a language, since a few hundred dims of character distinctions would separate many scripts. The limit is that the effective number of frequent characters is about the same for English, Russian, and French, so that alone doesn't settle it. Looking at the previous-token head 0.2, it carries roughly 20 to 23 strong dimensions, which fits a model distinguishing about 27 lowercase letters. I'll answer English with a confidence of about 0.35.

I'm considering whether to check the positional spectrum for periodicity that might reveal line structure, but the rotation angles from head 0.2 don't suggest any. I'll also see whether any layer 0 head locks onto a fixed far offset, though 0.3 looks like it stays within three or four tokens. Then I'll check the 3.2 induction chain back through 0.2 and 0.0 to pin down the offsets for layers 1 through 3.

Checking the offsets now: 0.2 looks like t-1, 0.0 like t-2, and 0.3 like t-3, since each cross-head match shifts by exactly one position. The angle ratios of roughly 1:2.3:3.7 are close enough to 1:2:3 to fit that reading.

