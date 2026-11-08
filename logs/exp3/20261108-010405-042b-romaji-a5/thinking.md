I'm stuck on the fact that only the middle blocks are available, so I can't run a forward pass to identify the language. I'm considering whether the attention QK weights or the block 0 layer-norm weights could reveal anything, but without the positional or token embeddings it looks nearly impossible to identify the language from weights alone.

I'm weighing whether the vocabulary size can be read off the rank of the embedding subspace, since a small character set like DNA's four letters would look different from a large one like Chinese. Layer 0's attention weights only see the residual stream, so any unused directions might show up in how they're shaped, though that's speculative without the embeddings themselves.

I'm reasoning that the layer-0 q, k, and v weights only get gradients in the directions the normalized inputs occupy, so components outside that span would either decay under weight decay or stay at their random init. That suggests looking for a singular-value split in block 0's W_q between trained and untouched directions, though init noise makes it hard to separate. I'll start by computing those singular values directly.

I'm checking which composition scores would show prev-token heads in layer 0 feeding induction heads later, and whether positive eigenvalues of the OV circuit point to token copying, even without the unembedding. Language is harder to pin down from weights alone, and "English (Shakespeare)" is the default guess, though the question hints the text could be something less common.

Now I'm wondering whether the vocabulary size can be read from the singular-value spectra of block 0's weights, scaled by the ln1 weight. I'll start by computing those.



---

I'm checking whether the singular values in block 0 decay toward zero, which would point to weight decay, and noting that the spectrum looks fairly smooth. Next I want the effective rank of the q/k/v input side, the subspace spanned by embeddings and positions, and a closer look at the right singular subspace of the stacked block 0 weights.

I'm working out the input side of block 0, since LayerNorm's gamma scaling matters there. I'll build the matrix of concatenated q/k/v weights with gamma folded into its columns, then run an SVD and look for a knee in the spectrum.



---

I'm looking at the block 0 MLP fc spectrum, where there seems to be a knee around 95 to 100 that drops from about 0.9 to 0.7 before decaying smoothly. The attention spectrum looks sharper, with a drop to about 0.5 near index 85, and I'm wondering whether 100 corresponds to vocab plus some positional dimensions.

I'm estimating the trained subspace at roughly 90 to 100 dimensions, which I'd guess is vocabulary plus some positional dimensions. With a vocab of 65, that leaves maybe 20 to 30 for position, though the centering in the layer norm could shift that by one. Separating token from positional directions seems hard without the embeddings themselves.

Next I'm weighing what kind of text this model was trained on, since the vocabulary size and the spread of positional dimensions only narrow it so far. Options like English prose, code, or a non-Latin script each imply very different character sets, and I want a sense of how confident I can be before committing to one. I'm also considering whether something like character frequencies, such as how often spaces appear, could be read off the layer 0 structure.

I'm turning over whether the spectrum can pin down the vocabulary, since a knee near 100 dimensions could mean vocab plus positions. Honestly, working out the language from the weights looks very hard, and the outlier dimension in the block 1 layer norm may be a common residual-stream quirk rather than anything about the data.

I'm now wondering whether the positional subspace could be identified through the layer 0 attention, since previous-token heads would need a QK form that rotates each position toward its predecessor. Without the embeddings themselves, though, checking that structure in the eigenvalues of the restricted W_Q^T W_K is hard to confirm.

I'm now checking whether the attention matrices are symmetric or strongly rotational, since antisymmetric parts would point to positional offset heads. I'm also planning to measure K-composition between layer 0 and later heads, folding in the layer norm gains.

Next I'm weighing whether the vocabulary size can be separated from the positional dimensions, since weight decay on the embeddings and the 256 learned positions makes that hard to read off the spectrum. I'm also planning to check the layer 0 attention for symmetric versus antisymmetric structure, and to compute composition scores between heads.

I'm checking whether the attention matrices are symmetric, since a strongly antisymmetric QK part would hint at positional offset heads.



---

I'm noticing the OV eigenvalues come out strongly negative in layers 1-3, which seems odd. My first thought is that it could come from working in the residual stream with LayerNorm, since "copying" is only well defined relative to the embed/unembed. Random matrices would sit near zero, so I'm wondering whether this is a plateau effect from initialization.

I'm checking whether the negative eigenvalues could just be a plateau effect from initialization, but random matrices sit near zero, so that doesn't fit. The symmetric fraction of 0.70 is about what a random matrix gives, so it tells me little. If the embedding and unembedding are tied, a copying head should map the token embedding to a positive direction, and negative values across all heads in layers 1-3 look strange.

I'm checking whether the negative eigenvalues could be a sign or normalization artifact rather than a real copying behavior, since the trace-over-norm ratio gives the same picture. Next I'll look directly at the top OV eigenvalues for the layer 1 heads.



---

I'm wondering why the heads in layer 1 and beyond have strongly negative self-eigenvalues, and whether they might be subtracting the source token, maybe because repeated characters are rare in this text. It could also be a broader pattern where the heads write previous-token information into a separate subspace, leaving a trace along a residual direction.

I'm now looking at composition scores, starting with K-composition: how a layer-l head's query-key circuit reads keys that include layer-0 OV outputs, normalized against a random baseline. I'm also considering Q- and V-composition, using the products of the OV and QK matrices.



---

I'm checking how head 0.1 composes with the later layers, and its small QK norm of 0.80 raises the question of whether its attention is nearly uniform or shaped by the query bias.

I'm working out whether a key-only bias term could make head 0.1 behave like a previous-token head, since absolute positions alone would favor the most recent token, which is the current one. Attention would then need some query dependence to pick out the prior position, and I want to check the size of the bias-derived key term against the rest.

I'm working out the bias terms: the query bias projected through the key weights gives a key-side vector that doesn't depend on the query position. I'm also folding in the layer-norm beta to get an effective query bias, and I'll check how large that key-only term is compared with the query-key interaction.

I'm now weighing whether self-suppression could come from the token-level QK term, since the positive eigenvalues would seem to favor attending to the same token, and I want to compare the bias contributions before settling on that.



---

The attention scores here are small because the QK operator norm is under 1, and they get divided by 8 as well. I'm checking how big the layer-normalized inputs are: with gamma already folded in, the vectors have norm around 19.6, so the quadratic form x^T QK x could reach roughly 0.74 times that scale.

Now I'm considering the layer-0 positional subspace, and whether the head that appears to be a previous-token head is one that layer 1 and 2 keys consistently read from. The awkward part is that if its OV output copies the token embedding into the same directions, the induction query-key match would just be attending to the same token, so I need to check how that output is actually used.

I'm trying to find the token subspace, which is how the layer-1 query side would read the current token while the key side reads the previous-token head's output. The plan is to compare how much of the layer-1 QK map lines up with what the previous-token head writes, and to check whether that output stays separable from the token embedding directions.

Checking the eigenvalues of QK times OV for the 0.1 head, treating the product as a bilinear form from the query residual to the source token: a positive trace would mean induction-style matching. I'm also checking the eigenvalues of QK on its own, since positive ones indicate same-token attention in layer 0.

I'm checking whether these same-token signals hold up, since the 0.1 head's attention looks mostly driven by its bias. Next I'll score each layer-1 and layer-2 head pairing with the layer-0 head by taking the real eigenvalue sum of QK times OV over the sum of absolute eigenvalues.



---

I'm weighing whether layer-1 heads that attend on the previous token could be the ones pulling from the 0.1 output, since several of them score high against it. But an induction-style pattern is also possible here, and in a four-layer character model I'd need to check which of those explanations the numbers actually support.

I'm now asking whether 0.1 is really a previous-token head or something that mostly attends to itself, and that depends on whether it has any positional signal. The key-only term alone would favor recency and point back at the current position, so attending to the previous token would need a query-key interaction involving position, which I still have to check.

I'm checking whether 0.1's QK has a meaningful antisymmetric part, since a previous-token head would need asymmetric, rotation-like structure in its positional subspace, while a purely positive-definite one would just favor self and nearby positions. Its symmetry score of 0.85 is higher than random, which leans toward the self-attending reading, so I want to look at layer-0 QK in the top singular subspace before deciding.

I'm thinking about the positional embeddings, which I can't read directly, and whether the layer-0 input weights reveal a low-rank positional subspace. I'm also wondering what kind of text the model was trained on, since a phrase in the setup hints at a specific domain like source code, Shakespeare, or DNA.

I'm weighing whether the vocabulary size can be estimated from the embedding rank, since the centered token vectors should span about V-1 dimensions, with positional structure adding a bit more. The spectrum of the combined block-0 reading weights may show where that knee sits, and I'll check whether the weight decay leaves a clear plateau there.

I'm going to check where the knee sits in the spectrum by taking a derivative of the curve, and I'll also repeat the same check on the concatenated right singular vectors of q, k, v, and fc, each normalized first.



---

I'm weighing whether there's a clear knee in the fc spectrum, which seems to shift around the 100 to 105 range, with the V transition earlier near 40. I'm considering that layer 0 fc acts like a token lookup, so the token subspace probably matters, and I'm looking at the large top singular vector, which may tie to the outlier dimension. I'm also thinking about the positional structure in later attention layers, but that looks too hard, so I'll try a different approach.

I'm looking at whether the input embedding distribution can be reconstructed from the trained part of the fc0 weights, since the rows should span the inputs weighted by frequency. Decoding the language directly seems very hard, but the knee near 100 suggests a small character set, which would fit English or code rather than a large vocabulary like Chinese.

I'm comparing the V spectrum, which drops off quickly, against the Q/K spectrum, which stays higher, to see whether V reads only a few dimensions while Q/K picks up both tokens and positions. I want to compute principal angles between the top-k V row space and the Q/K subspace, and then print the full V singular value shape to check the cutoff.



---

I'm looking at layer 0's V maps and see a sharp knee around 33–35 dimensions, with the signal flattening to about 0.3 after that. If V is reading token identity, that would suggest a vocabulary of roughly 35 characters, which hints at a small alphabet, maybe lowercase-only text or a language with around 30 characters, though I'm not sure yet.

I'm wondering whether V only reads part of the token space, since frequent characters would be better embedded, and it might also pick up some positional information. The MLP input is harder to read cleanly because it mixes tokens, positions, and attention outputs written in by layer 0's attention. Q and K show no clear knee, so only V's ~34-dimension rank looks like a candidate for pure token reading.

I'm checking the V knee more carefully: the values drop from about 0.45 to a plateau near 0.33, which suggests roughly 35 dimensions above the floor. The plateau itself probably reflects the random initialization that the training hasn't fully overwritten, so the trained subspace is buried a bit under it.

I'm checking whether V's ~35-dimension rank holds up per head and comparing that subspace against Q and K, to see if the latter carry extra dimensions that would point to positional information.



---

Heads 0.0, 0.2, and 0.3 carry much higher-rank value and query/key projections, while 0.1, 0.4, and 0.5 use only a handful of dimensions. Head 0.1 is a clear example, with query and key both around rank 4 to 5, which looks like a positional mechanism that could make it a previous-token head.

I'm checking whether head 0.1's low-rank query-key subspace is positional, which would fit a previous-token pattern. The idea is that the bilinear form would peak when the query is one position ahead of the key, so I want to see if its structure is rotation-like across frequencies. I'll compare the top query and key singular subspaces and look at the restricted matrix.



---

I'm noting that head 0.1's QK circuit is largely symmetric in its subspace, since the query and key directions nearly coincide, so it behaves like a similarity measure that attends to similar tokens or nearby positions. That doesn't obviously explain previous-token behavior, so I'm wondering whether the key bias term is what makes it work.

I'm still checking whether a positive semidefinite similarity could ever peak at the previous position, and it seems it can't reliably do that. I'm now considering whether the score scale is too small to matter, and whether this head might instead be a broad recency smear. I want to see how much of its query-key energy overlaps with the token subspace from layer-0's value vectors.



---

The heads 0.0, 0.2 and 0.3 read their queries and keys from a subspace outside the token subspace, so their attention looks positional, while their values carry token content. Heads 0.1, 0.4 and 0.5 do their matching on token content.

I'm checking whether the positional subspace belongs to a previous-token pattern, and the symmetric eigenvalues don't cleanly settle that. The layer-1 composition results also don't line up neatly with the positional heads, which leaves the picture open.

I'm checking whether layer-1 keys read the 0.1 output more strongly than the others. The K-comp scores might be skewed because 0.0, 0.2 and 0.3 have broader OV matrices than 0.1, so I'll normalize for that. Next I want to see which subspaces layer-1 keys actually read, and how much of that falls in the token subspace, the positional subspace, or each layer-0 head's output.



---

I'm checking whether the positional subspace, around 30 dimensions, lets me recover the positional embeddings directly, though probably not uniquely. I'm also considering whether heads 0.0, 0.2, and 0.3 act as previous-token heads by looking at eigenvalues of the QK matrix restricted to the positional subspace, where complex pairs with small phase angles would point to that structure.

I'm weighing what each kind of eigenvalue structure would mean: symmetric positive semidefinite would suggest a self or local-window head, and something negative definite would point to attending far away, such as to the beginning-of-sequence token. Next I want to check the eigen phases directly.



---

I'm reading the positional heads' eigenvalue phases, which rotate a lot, and checking whether that fits relative-offset attention. Head 0.2 sits around 30-45°, head 0.0 around 70-100°, and head 0.3 around 115-170°. If the positional embeddings are sinusoidal with frequency ω, the phase would be ω times the offset k, so I'm asking whether each head is attending to a different offset.

I'm testing whether the three heads share the same eigen-planes, since that would mean their phases scale with offset. The guess is head 0.2 is previous-token, 0.0 is two back, and 0.3 is three or four back. To check it, I'd compare A_0 against A_2 squared, and A_3 against A_2 cubed, using normalized cosine similarity.



---

I'm checking how the offsets line up: R0 tracks R2 squared, and R3 lines up with R2 cubed or to the fourth power and also with R0 squared. That would put the 0.2 at a one-token shift, 0.0 at two, and 0.3 at roughly three or four. I can't yet tell whether the base R2 is itself offset by one, so the absolute values are still open.

Now I'm checking whether the head's query-key match tracks the previous token of the key: head 3.2 scores high on that, with a strong self-alignment and a large OV norm, which makes it a candidate induction head. Earlier K-composition scores for the offset-1 writer were mostly low, except for 3.2, 3.1 and 3.0.

I'm checking whether head 3.2's key side reads more from the subspace written by the 0.2 head or from the raw token subspace, since the two may overlap and be hard to tell apart. I also want to look at whether earlier layers' outputs carry previous-token information through other heads, so I'll compare the layer-0 output subspaces against the token and position subspaces.



---

Earlier heads 0.0, 0.2 and 0.3 hold previous-token, 2-back and 3-back identity in subspaces separate from T. Next I'm measuring how much each later head's K side reads from those outputs, plus T, P and the Q side, using normalized QK@O[h] norms.

I'm also thinking about layer-1 heads, since they may carry token information that gets moved forward, and I'll start with that check.



---

I'm checking the token and position reads, which come out near 1 for the later layers; the raw embedding is probably small next to the other content, with layer norm involved. Layers 1 and 2 pull heavily from the outputs of heads 0.1, 0.4, and 0.5, which overlap with one another. I'm wondering what those three heads actually do: their token-based query-key terms may mean they attend to the same or similar tokens, or to particular ones.

I'm now looking at whether heads 0.1, 0.4, and 0.5 might be attending to something specific, like spaces or newlines, and whether their key-side bias terms matter. Comparing the magnitudes, the query-key term looks like it could dominate the score. For the induction candidate 3.2, I'm checking whether the query reads the previous-token signal while the keys read the two-back and previous positions, so that a match on the bigram context at each key position is what drives the attention.

I'm checking whether the query at the current position reads the current token while the key at each earlier position reads the previous and two-back tokens, which would make 3.2 a bigram-matching induction head. To test that, I'm computing eigenvalue positivity for each pairing of query and key sources across the layer 1 to 3 heads.

I'm working out the bilinear matching now: for each pair of query-side and key-side token offsets, I'll compute the form that maps a query's token to a key's token and check its trace and norm. Same-token matches show up as positive trace, and induction should appear where the key sits one offset further back than the query.



---

I'm reading Head 3.2's attention scores: the c→p1, p1→p2 and p2→p3 links are all strong, which is the pattern you'd expect from an induction head, where the query at offset a attends to the key at offset a+1. The self-attention values on the diagonal are high too, and I'm wondering whether that means it's also matching positions in the same context, which is what I'm checking next.

I'm now checking whether the diagonal attention in Head 3.2 reflects matching against earlier occurrences of the same context, which the shifted pattern would also support. The shifted links look like the stronger signature, and the head also has the largest OV norm. Layer-1 heads show a similar offset pattern that seems to reinforce attention to the previous position.

Now I'm moving on to the layer-2 heads, checking whether any of them show the same shifted pattern, since they could be composing with layer-1 outputs. I doubt they're doing induction on their own, because the layer-0 previous-token signal looks weak there, so I'll test the other heads directly.

Now I'm turning to the layer-0 heads: 0.0 looks like a two-back positional head, 0.2 is the previous-token head, and 0.3 attends about three back. The remaining heads 0.1, 0.4 and 0.5 seem content-based, matching earlier positions with similar tokens.

Now I'm checking the layer-1 heads and the content-based layer-0 heads. Their attention depends on the tokens themselves rather than position, so they likely just track which characters have already appeared.

I'm weighing vocabulary size against the token-subspace dimensions: with about 35 separable token dimensions after removing one for centering, the vocabulary is probably around 35 or 36 characters. I'm considering which kind of text would have that many distinct characters, such as lowercased English with punctuation rather than plain lowercase or a larger character set.

I'm checking whether the token subspace feeding the content-based heads' queries and keys, along with the fc0 read directions outside the token, positional and attention-output subspaces, is larger than the roughly 35 dimensions I estimated, since rare characters might still be distinguished even if the value maps don't use them.

I'm working through the spectrum of the layer-0 output weights to see how much of the residual stream is actually used, and whether the attention outputs add dimensions beyond the token and position subspaces. Rather than inferring the vocabulary from the structure, I'm checking the singular values to estimate how many dimensions carry real signal.

I'm checking whether the token subspace is actually larger than the 35 dimensions I estimated, since the content heads and later layers might read more distinctions than the value maps use. I'm also thinking about whether the layer-0 V matrices for the copying heads each need about 22 dimensions, which would make the combined count around 35. Next I'll look at the final layers' writes to see whether they add anything beyond the embedding space.

I'm treating the final layer's writes as a vocabulary probe: anything block 3 writes outside the unembedding's row space is wasted, so the trained output directions should give roughly the vocabulary size. I'll check the singular values of the block-3 MLP and attention output projections.



---

I'm looking at the block 3 MLP projection, where the strong component runs to about index 32 with a knee around 27 to 33. That suggests the unembedding-relevant dimension is roughly 33, which would put the vocabulary near 34, though I'm not sure yet, and I'm still checking how the layer-norm centering and the rank of the unembedding matrix constrain that.

I'm weighing whether the vocabulary is about 34 to 36 symbols, since the input and output sides both point there, though the slow tail might add rarer characters and push it toward 33 to 40. Checking candidate text sources, lowercase English with space and punctuation fits the rough count, while Russian and Shakespeare fit less well, and code, DNA, and Chinese are ruled out.

I'm now going through the candidate alphabets: Hebrew with its final forms comes to about 35 symbols with space and punctuation, and Arabic lands around 36 to 40, so both still fit the rough vocabulary estimate.

I'm checking whether the embeddings are tied, since the output subspace of block 3's projection should then line up with the input subspace from block 0. If the principal angles between the two match, the dimension count of about 35 holds up.



---

Checking whether the first-block V values line up with a vocabulary size of about 36: the strong dimensions look like roughly 33, plus two weaker ones.

I'm weighing whether the character set could be caseless, since a vocabulary near 35 would rule out uppercase letters. Candidates like Hebrew, Georgian, or Arabic come to mind, while Hindi or Thai would be far too large.

I'm now working through the block-3 projection weights: the first 33 singular values are clearly strong, with a long tail near 2. A random init at that scale would only give singular values around 0.14 to 0.41, so the tail is trained, not noise.

I'm reasoning about vocabulary size: with roughly 35 characters and no case distinction, the likely candidates are lowercase English or a caseless script, and I'd guess the total lands around 34, possibly up to 40 with rare symbols.

I'm now weighing caseless scripts and lowercase-only English as candidates for a roughly 36-symbol set, counting letters, space, and common punctuation. Hebrew without niqqud and Georgian come to mind too, though their totals run a bit higher. I'm also wondering whether the text might be lowercased Shakespeare, and whether poetry-style structure could help narrow it.

I'm checking whether a single key direction in the block-0 K weights, with a much larger top singular value, could be tied to a specific token like a space or newline. That wouldn't settle the language, though, so I'm setting that idea aside.

I'm leaning toward English in lowercase, but I'm unsure whether the text might contain capitals. If it did, the vocabulary would be much larger than the roughly 35 dimensions I'm seeing, and I'd expect the projection spectrum to show more structure than it does.

I'm noticing the block-3 spectrum doesn't drop off cleanly after index 33: it slopes gradually from about 2 down to 1 through index 60. That could mean a second group of weaker dimensions, possibly for rarer characters or capitals. If a Shakespeare-style text with both cases were in play, the roughly 33 frequent symbols plus about 27 less common ones would land near 60, which fits the spread.

I'm now sizing the tail past index 60 against what a random init would produce, looking at whether values near 0.57 still fit a scaled-down Gaussian matrix of that shape.

I'm going to compare the tail of the block-3 projection spectrum, past index 100, with a random Gaussian baseline of the same shape, to find where the trained excess actually begins.



---

I'm finding the decay curve is smooth past 33, with no second knee near 60, and the fit scale doesn't look clean since the ratio is still 1.45 at 120. Comparing against block 0's V and the tied-embedding idea doesn't help here, so I'm looking for another way to count the vocabulary.

I'm now considering that the vocabulary might be smaller than 65 characters: if the model treats capital and lowercase letters alike, with a separate capitalization flag, the effective embedding size could be around 35 dimensions.

I'm working out how the vocabulary count fits with a cased English alphabet: lowercase letters, a case flag, and some punctuation groups would land near 33 dimensions, which is consistent, while a larger script or a tiny one would not. I'm also tallying the strong output dimensions at the 33 and 34 indices to see where the gap falls.

I'm weighing English as the most likely alphabet, though only with moderate-low confidence around 0.4. Code is less likely because its larger symbol set would need more dimensions, which the 33-ish count doesn't support. I'm also noting that a few steps in the strong-dimension drops don't show a clear break, so I'm checking other languages against English before settling.

Next I'm checking whether the average word length would show up as periodicity in the data, though it doesn't look inferable from this. I'm also weighing the block 0 token dimension of about 35 against the output count of 33, which still lines up with an English-like alphabet once the capital flag and punctuation are included. Layer 1 head roles are next: how the keys and queries read the content-head outputs, and how strongly positional dependence shows up.

I'm checking whether the early-layer attention heads here carry positional information, testing their query-key structure against a simple offset-one rotation to see if they behave like previous-token heads.



---

I'm recomputing R, since the earlier value got replaced by the random matrix.



---

I'm weighing whether R2 sits at offset 1 or whether the sign flips with forward versus backward direction, and since the effect is causal I'm leaning toward backward. I'm also wondering if 0.2 could be offset 1, with 0.0 at offset 2 and 0.3 at offset 3 to 4.

I'm now turning to the layer 1-2 heads: their OV circuits have strongly negative traces, and the query reads older tokens while the key reads the outputs of content heads 0.1, 0.4, and 0.5. I'm checking whether those content heads act as same-token matchers that copy the current token, or whether their key bias makes them default to certain token types.

I'm now looking at whether layers 1-2 heads key on the content heads' outputs and whether they end up attending to the immediately previous position through matching older tokens. Some of these heads seem to implement content-based local attention, so I'm working out which offsets they reinforce.

Now I'm checking whether layer 1 heads like 1.1, 1.3, and 1.5 act as previous-token heads through content matching, since their keys seem to favor the position right before the query. The same pattern could also pick up the query position itself, as 1.1's diagonal suggests, though I'm not sure yet.

I'm testing whether layer-1 heads act as same-context matchers, attending to earlier positions whose recent history lines up with the query's, which would make them something other than induction heads.

I'm checking whether layer-1 heads act as previous-token heads through content matching, since the shifted-match terms all line up at the immediately preceding position. For true induction, the telling signal is the current-token-to-previous-key match, and only head 3.2 shows that clearly, while the others are weak.

I'm checking whether layer 3's K-composition with layer 1 or 2 heads shows anything large, and it doesn't: the largest values are around 0.1, so the single induction candidate remains 3.2, whose key reads the layer-0 positional outputs and whose query reads O2 for longer context. I'm also starting to look at 3.2's OV circuit, expecting it to copy the attended token forward, but its negative eigenvalue (around -0.58) doesn't obviously fit that, so I'm weighing whether tied embeddings change the picture.

I'm checking whether 3.2's OV circuit writes into the output subspace the unembedding reads from, since its negative eigenvalue doesn't match a simple copy-the-token story. If the embeddings are untied, the residual-space trace may not be a problem, so I want to see whether 3.2 reads the token direction and writes into the strong output directions.



---

I'm checking whether copying is actually happening: 3.2 leans hard into the logit subspace and reads the token subspace more than other directions, which fits with it moving token identity into the output. A direct test would be to push the layer-0 token directions through the OV matrix and compare against C33, but that doesn't map cleanly back to tokens, so I'm weighing whether tied embeddings could give a way around that.

I'm weighing whether the tied embeddings give a usable link between the token-input directions and the output subspace, even though the overlap I measured is only moderate. A direct copying check would be to compare the OV map from 3.2 against the token directions, but the sign and mapping aren't clear enough to verify it, so I'm setting that aside for now.

I'm weighing whether the vocabulary size points toward a language with case distinctions, since capitals at line starts and speaker names would need their own logit directions. The output subspace looks close to full rank for a 65-token set, but the tail doesn't show a clear knee at 64, so it's ambiguous. A 33-dimension strong subspace fits an alphabet of around 35–40 symbols better than a plain 27-character set.

I'm weighing English against code as the source. The vocabulary estimate argues against code, since it would need many more distinct symbols, and the single induction head fits natural language better than the heavy repetition code models usually show. A letters-plus-space-plus-punctuation-plus-case set around 35 symbols also matches the roughly 35 dimensions I'm seeing.

I'm settling on English at about 0.4 confidence, since the vocabulary estimate weighs against code and the single induction head fits natural text better. Next I'm going through the head labels: 0.0 and 0.3 look positional, 0.2 is a previous-token head, and 0.1, 0.4, and 0.5 are content-matching heads, with layer-1 heads reading from those outputs.

Now I'm going through the head roles in layers 1 through 3: the layer-1 heads with negative self-attention that avoid repeating the same token, the layer-2 head that writes strongly to the output, and the layer-3 output heads. I'm also checking whether the offset-two and offset-three heads belong with the previous-token head, which only seems to be 0.2.

I'm checking the remaining head roles and whether 0.2 alone counts as the previous-token head, since the offset-one rotation fits the phase range. The 3.2 diagonal-shift test backs that up, and I'm holding confidence in 3.2 as the induction head.

