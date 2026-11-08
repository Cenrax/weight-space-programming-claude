I'm starting on a tricky problem: working out the language from the weights alone, with no forward passes. The main difficulty is that without embeddings there's nothing obvious to identify the language from, so I'm wondering whether the block 0 layer norm parameters, or the structure of the residual stream subspace, might carry that information.

I'm weighing whether the vocabulary size could be read off the rank of the subspace the first block uses, since the token and position embeddings together would span it. Without the embeddings the weights don't seem to show that directly, so I'm wondering whether a tiny alphabet like DNA's four letters would leave a visible trace in the unused directions.

I'm checking whether the weight decay would have shrunk the input directions of block 0's projections that never got gradient, so the effective rank might show the span of the embeddings and positions. The vocabulary might be around 65 characters, and I'm thinking about how the positional spectrum would look. Next I want to compute the singular value spectra of the layer-0 weights after scaling by the ln1 weights.



---

The 384th singular value sitting at zero makes sense: that direction is the all-ones vector, since layer norm outputs have zero mean. The rest of the spectrum falls off smoothly, so next I want the effective rank of layer 0's q weights, and then the input-side weights concatenated together to see how many input dimensions they actually span.

I'm checking the weights that feed layer 0's input side: the q, k, and v projections scaled by the first layer norm's weight, since the fc layer sees the post-attention activations and not just the embeddings.



---

I'm weighing whether the weights show any sign of decay, and it seems likely they're barely decayed. I'm also considering whether the bias vectors in the last block could point toward frequent characters, though without the unembedding I have no way to map them. I'm wondering if the input embedding might be tied to the unembedding, but nothing in the setup says so.

I'm considering whether the input embedding could be tied to the unembedding, and whether a small vocabulary would show up in the final-layer gradients. If it did, the block 3 MLP projection weights should have a few large singular values standing above the random-init bulk, so I plan to check their spectrum for a cliff.



---

I'm checking block 3's proj layer, where the singular values seem to drop off steeply around index 88 to 92, so I'll print the full spectrum there and compute the knee. I also want to look at block 3's attn.o next.



---

The knee sits somewhere around 77 to 90, and I can't pin V down exactly from it. The weights don't look like a random init with low rank: at std 0.02 the singular values would top out near 1.17 and bottom near 0.4, but here the tail past roughly 90 falls from 1.4 to 0.2, a wider spread. That suggests the model trained for a long time and the weights moved away from initialization.

I'm reasoning that the final residual's gradient lies in the span of the unembedding rows, so only about V dimensions of block 3's output actually matter for the loss. That makes the useful column space of the MLP projection, and its overlap with the attention output, the thing to check to narrow down V around 80 to 95.

I'm going back over the weight statistics, asking whether the layer-0 input could reveal something about the text's structure, like periodic line lengths. But positional embeddings aren't available to check that directly, so I'm weighing whether the layer-0 weights hint at it some other way. The spectrum there showed no clear cliff, so that route looks weak for now.

I'm working out the gradient through the final layer: only the directions that the unembedding reads actually feed the loss, so block 3's projection weights are only constrained along that span, minus the mean and the per-token normalized direction.

I'm checking whether the block 3 gradient reduces to an approximately low-rank update, since the x̂ term spreads it across all directions but mostly it sits in a near-V-dimensional subspace. Next I want to compare the column spaces of the block 3 MLP projection and attention output, looking at principal angles over the top-k directions to see if both write into the same readout space up to about V and then fall off.

I'm checking the spectrum of block 3's MLP projection against what a random init at std 0.007 would give, since the tail values sit well above that. The bulk past index 90 falls off more gradually than a Marchenko-Pastur fit would predict, so I want to look at the spectral density out there and fit it properly.

The steep part of the block 3 MLP projection spectrum seems to end around index 88 to 90, which would put the readout dimension near V, maybe 90 or so. Since rare characters may carry weaker singular values, I'd guess the vocabulary is a bit above 90, possibly printable ASCII. I'll check the principal angles between the block 3 MLP projection, the attention output, and block 2 to see whether they share that same readout subspace.



---

I'm not seeing a clear picture yet, so I'll look at the derivative of block 3's projected spectrum more directly. I'll plot the log of the spectrum against its difference to see what shape it takes.



---

I'm looking at a knee region around positions 73 to 90, which lines up with a vocabulary of roughly 90, though that's weak evidence. It doesn't settle the language, so I'm considering whether the bias in block 3's MLP projection could reflect unigram frequencies, which doesn't map cleanly onto characters. I'm also wondering if the text has a periodic or metrical structure, like poetry, or if the vocabulary is something small such as DNA bases.

I'm weighing whether the vocabulary is something other than natural language: DNA-style alphabets would give a readout far too small, and code or non-Latin scripts would not produce a knee near 90. Still, the spectrum of block 3's attention output doesn't show that knee, which makes the low-rank explanation harder to trust.

I'm considering how to find the previous-token and induction heads without the positional embeddings, since layer 0's attention depends on them. One route is to look for K-composition: later heads whose keys read from earlier heads' outputs, which would reveal previous-token heads and then induction heads built on top of them.

I'm sketching how to isolate the positional subspace in layer 0, since a previous-token head's QK form should reflect position differences in a rotation-like structure. Without the positional embeddings, I'd try reading the bias terms and the low-dimensional smooth structure in the weights instead.

I'm checking whether a previous-token head shows up as an antisymmetric, rotation-like part in its W_QK, and whether the complex eigenvalues reveal it. I'll compare symmetric and antisymmetric norms for each layer-0 head, then look at the eigenvalues.

I'm checking whether the eigenvalues of each layer-0 head's W_QK are real and positive, which would point to a same-token pattern, or complex with large imaginary parts, which would point to a rotation-like previous-token head. I'm also thinking about how to test later heads for induction-style K-composition with those layer-0 heads, and whether the OV circuit's positive-eigenvalue structure can stand in for a copying check without the embedding and unembedding matrices.

I'm computing per-head statistics for layer 0, folding in the LayerNorm gain so the effective query weights are right, and I'm also weighing whether a language check or a tighter vocabulary-size estimate would help more than the head analysis.

I'm now thinking about whether the readout subspace of block 3 lines up with the input subspace of block 0, since nanoGPT ties its embedding and unembedding weights. If they're tied, the overlap between the top directions should reveal the vocabulary's dimensionality.

I'm wondering whether the embeddings can be recovered from the block-3 MLP output neurons, since with tied weights each neuron's output direction would sit in the same space as the token directions. If I cluster those output vectors, I might be able to pull out individual token directions, though that's still speculative.

I'm thinking about whether the block 3 MLP output neurons could be sparse-coded into individual token directions, since each one tends to promote only a few tokens, though labeling them would need frequency or bigram statistics. A more direct route might be to pin down the vocabulary size first: if it's about 65, that points toward Shakespeare, and I'd check whether the readout structure suggests something different.

I'm checking whether the output-weight spectrum still holds up once the all-ones direction is projected out, and I'll also look at the per-head output weights. The final LayerNorm gain is unknown, so the vocabulary estimate stays rough, and I'll come back to the residual-stream basis after the heads.



---

The eigenvalues of the OV circuit in layers 1 through 3 come out strongly negative, with trace over sum of absolute values near -0.8. That suggests each head reads from and writes back into the same residual subspace with a flipped sign, so it might be subtracting what it attends to. I'm wondering whether this is a copying pattern instead.

Negative OV eigenvalues in nearly every head of layers 1 through 3 seem odd. Permutations of the residual basis would leave them unchanged, but sign flips would not preserve the layer norm, so that's probably not what happened. More likely a few large negative directions dominate the trace, so I want to print the top eigenvalues.



---

I'm noticing that all heads in layers 1-3 have negative OV eigenvalues, which seems unusual. I'm wondering whether that's typical for small trained transformers or whether the weights might have been altered, perhaps by flipping the residual stream's sign. I'm also checking how layer normalization behaves under x → -x.

Still checking the layer-1 and later heads for composition, so I'm working out whether their keys read from a layer-0 previous-token head.

I'm checking composition scores for the layer-1+ heads against layer-0 heads, comparing key, query, and value composition against a random baseline, with layer norm gains folded into the readers. I'm using the input-side query-key matrix and the key path reading from each layer-0 head's output.



---

I'm checking how head 0.2 is K-composed with the layer-1 and layer-2 heads, and I'm leaning toward it as the prev-token head, possibly with 1.1 as well.

I'm weighing whether the 0.182 composition score for 0.2 really means anything, since a low-rank OV can inflate it. Against a random baseline near 1/sqrt(384), about 0.051, it still looks significant. I can't confirm it's a previous-token head yet, because that would need positional information I don't have.

I'm wondering whether the positional embedding subspace can be recovered, since it's shared across tokens and might sit in dedicated dimensions. For a previous-token head, the query at position i should line up with the key at i-1, and learned positional embeddings often look like low-frequency helices, which would show up as near-rotational structure in the QK matrix. The antisymmetry ratios of about 0.7 look fairly random, though. Next I want to check head 0.2's QK singular values and its biases, and see whether its QK mass concentrates in a positional subspace.

I'm testing whether tied embeddings let me split the layer-0 input space into a token subspace and a positional remainder. If I can, I could measure how much of each layer-0 head's QK lies in each subspace, which would show whether 0.2 is reading positions or content.

I'm checking whether tied embeddings would let me separate a token subspace from the positional remainder: if they're tied, layer-0 value reads should overlap the top token directions more than the 0.23 expected by chance.



---

Layer 0 reads don't favor T at all, which fits the random baseline of about 0.23, so that layer is either untied or spreads its weights widely. Block 3's attention heads write roughly 0.45 to 0.63 into T, and its projection sits at 0.94, partly by construction. Next I want a better estimate of V by looking at block 3 again.

I'm checking whether the readout subspace gives a better estimate of V. The gradient into the residual is a projection of W_U^T δ, which lies in the span of differences between W_U rows, so the effective rank is at most V-1, and the knee near 88–91 points to V around 90–92. That leaves me weighing which corpus has roughly that many distinct characters, with rare symbols likely undercounted.

I'm weighing whether a single-token readout estimate is feasible from the weights alone, and the code-versus-prose question looks hard to settle that way, since attention and bracket patterns don't show up cleanly in the weights. I could try recovering token directions from block 3's MLP neurons through a sparse decomposition and look for clusters like uppercase/lowercase pairs, though that seems ambitious. A simpler route might be to use unigram frequency from the readout bias.

I'm considering whether the language could be inferred from structure instead, such as the effective vocabulary size from the readout rank, which points to something with roughly 90 distinct symbols. Candidates like code, English with diverse punctuation, or a European language with accents all fit that count, and I can't easily tell them apart from the weights alone.

I'm going to test whether the combined block 3 writes, proj plus o, give a cleaner knee, and whether the energy captured by the top-k directions saturates well above the k/384 random rate.



---

I'm checking the marginal gain per dimension for attn.o across the ranges, and the rate drops steadily with no sharp knee, with the bias saturating around 90. Now I'm wondering whether the block 3 attention heads write so much outside the readout space because those writes are large.

I'm weighing whether the vocabulary size can be pinned down from the layer-0 side, since those layers only ever see inputs in the span of the embeddings and positional encodings. The gradient on those weights should only touch directions in that span, so the untouched directions would stay at their random initialization.

I'm working out how the layer-0 weight updates are confined to the span of the embeddings and positions, so their row space has dimension at most the vocabulary size plus the position rank plus one. If the learned positions span most of the 384 dimensions, that bound says little, though the layer-0 spectrum showed no clear cliff. The cleaner signal is still the readout rank near 90, so I'm checking that against a Marchenko–Pastur fit on the block 3 singular values: the random bulk would top out around 0.6, yet the values past index 90 sit well above it, so there's trained structure beyond that rank.

I'm estimating the vocabulary size from the spectrum: the steep drop between indices 73 and 90 looks like rare tokens contributing small singular values, which puts the vocabulary somewhere around 75 to 95, with 90 the most likely point. Then I'm wondering which datasets the designer chose, with English and a few other languages or code among the candidates.

I'm weighing whether the vocabulary could be English prose, since a typical character set there runs about 80 to 90 symbols and fits the estimate. Spaces dominate prose and code alike, while Chinese would show almost none, which is worth keeping in mind. Going by the prior, English seems the most likely guess, and I'll commit to it as my best specific answer.

I'm considering whether the OV eigenvalues being mostly negative in the later heads could reflect the data, since a character-level model trained on text might suppress repeated symbols. I'm thinking about using the attention patterns, especially induction-like or indentation-tracking heads, to narrow down whether the source is code or prose.

I'm noticing that the OV eigenvalues are mostly negative in the later heads, which I'm wondering might say something about the data. I'm recalling that negative eigenvalues in residual space aren't unusual in pre-LN models, so that alone may not settle much. I'll turn to the attention patterns next and look for induction-like heads to see whether the source looks more like code or prose.

I'm checking whether layer 0's attention head 0.2 is a previous-token head, since that would make the later heads 1.0, 1.2, and 1.5 plausible induction candidates that read its output.

I'm checking whether head 0.3's high trace in QK, which suggests it attends to similar or nearby positions, pairs with its high-rank OV to carry rich token identity, while 0.2's low-rank OV makes it look more like a positional or whitespace feature than a true previous-token copier. I'll also look at the query and key biases to see how they shift those attention patterns.

I'm going through the layer-0 heads one at a time, pulling out the QK and OV singular values for each.



---

I'm looking at head 0.3, where the QK spectrum is flat at around 0.6 and the normalized trace is large and positive. That suggests a near-isometric bilinear form, so maybe QK is roughly a scaled near-orthogonal matrix, and in the positional subspace that would make it attend to a fixed offset.

I'm checking whether head 0.3's QK eigenvalues sit on a circle with phases, since a previous-token head with sinusoidal positions would show that. A positive trace fits with mostly positive real parts at low frequencies. I'm also noting that its OV is high-rank and flat, which would suit copying the previous token's full identity, even though its K-composition output is low.

Head 0.2's output could be positional or a "previous token is whitespace" signal, but induction needs key-side identity, which 0.3 seems better suited for, even though its K-composition looks low. Maybe that metric is skewed by OV norms, so I'll compare how much each layer-0 head's output is read through keys versus queries versus values by the later heads.



---

The same low-dimensional signal feeds both the query and key of head 1.0, which suggests it's matching on a shared feature, maybe positional or region information. The q/k ratio approach looks shaky, since the layer-0 output could be small next to the embedding, so I still need the relative sizes of the residual components.

Now I'm checking whether head 0.3 has a rotation-like structure in its QK matrix, since its spectrum is flat and high-rank, unlike 0.2's dominant rank-1 component. I'm thinking about what a rank-1 pattern in 0.2 would mean for attention to a single feature class, and whether a positional subspace could be identified that way.



---

I'm looking at head 0.3's eigenvalues, which come in conjugate pairs at small angles with nearly equal magnitudes around 0.54. That rotational structure suggests the head is encoding relative position, something like a previous-token head using sinusoidal-style positional embeddings, and I'm starting to check how the score between positions would peak under that picture.

The fixed-offset reading still fits: the eigenvalue structure alone can't say which direction the offset points, since conjugate pairs always come in both signs. Head 0.3 looks most likely to be a previous-token head, with a flat, high-rank OV circuit that copies token identity. Head 0.5's angles look like multiples of roughly 25°, which would fit a harmonic set of positional frequencies, but I can't pin down the frequencies from the eigenvalues without more information.

Head 0.3's smaller angles don't look as harmonic, though, and head 0.4's angles cluster in the 117–168° range, close to antipodal. Head 0.5's angles seem to fit a harmonic series, so the offset question might come down to whether the angles scale with a positional frequency, which I can't confirm without the actual positional embeddings.

I'm checking whether the angles between heads line up as multiples of a shared offset: if head 0.3 is an offset-1 head and head 0.5 uses a different offset, the angles in matched planes should differ by that factor. To test it, I'd match the invariant 2D planes across heads 0.3, 0.4 and 0.5 using their eigenvectors and compare the angles in each shared plane.

I'm going to check whether the eigenvectors of A give the planes I need, by taking each conjugate pair's real 2D plane from the right eigenvector. Then I'll compute the overlaps between those planes across heads 0.3, 0.4 and 0.5 to see whether any plane matches, so I can compare the angles in it.



---

I'm comparing the angles for heads 0.3 and 0.5, which share planes. The ratios range from about 2.5x to 5.8x, so they aren't a constant factor. I'm wondering whether that's because the eigenvalue angles of A don't exactly match the offset rotation.

The eigenvalue angles of A don't give the true rotation, since A isn't a pure rotation because of the M R mixing. I'll take a more careful route: restrict A to the plane basis from head 0.3, decompose that 2x2 block into a scaled rotation, and read off the angle for both heads in the same basis.



---

I'm checking the ratios of the 0.5 and 0.3 values, and they aren't consistent: they fall as the angle grows. I'm wondering whether an offset-0 contamination explains it, so I'm starting to test the idea that the 0.3 values carry an offset of 1.

I'm weighing whether the 0.5 head sits at an offset of about 3, with some wrap-around complicating things, and whether the 0.3 head might be mixing self and previous-position attention. Then I'm looking at the shared positional-encoding planes: if 0.3 is an offset-1 head, its frequency choices imply specific periods. Head 0.4 seems to rely on a different, higher-frequency plane set, so I want to recover the planes from 0.4 and 0.5 and compute the angles for every head.



---

I'm comparing the angles across the 0.3 and 0.5 planes, which step up by about 8° per harmonic on one and about 25° on the other.

I'm working out whether the 0.3 or the 0.5 plane is the one with the offset, since the angle steps differ: about 8° per harmonic for one and 25° for the other. If 0.5 takes offset 1, its fundamental would be 25° with a period near 14 positions, which would leave 0.3 mostly self-attending. I can't tell yet which assignment holds, since the frequency isn't known.

I'm now weighing the angle spacing in the 0.3 set, where the increments hover around 8° per step, and considering whether a Fourier-like frequency with a small position offset could account for it. Working through the arithmetic, the 0.5 plane would correspond to a much larger offset than the 0.3 plane, which could point toward a window-style head rather than a pure previous-token one. Since the fundamental frequency is still unknown, I can't settle which plane carries which offset yet.

I'm checking the 0.5 angles, which step up by about 25° per harmonic with almost no intercept, so that plane looks like a clean relative offset. The 0.3 plane steps by about 8° with a small intercept, which fits less neatly. If 0.3 is offset 1, then 0.5 would be offset about 3, and I'm trying to pin down the frequency that would make that consistent.

I'm weighing whether the learned embedding's fundamental period would be anything in particular, since a smooth helix over 256 positions could have an arbitrary spacing. If the harmonics run k=1 through 6 with a spacing near 8°, the higher planes would be needed to tell neighboring positions apart, and the 0.3 plane's larger angles around 47°, 121°, and 178° look like they could serve that role.

I'm checking the remaining planes, where the 0.3 head's larger angles around 121° and 178° might still carry a previous-token signal, and I'm looking at 0.4's high-frequency planes near 117°, 143°, and 168° as possible candidates for an offset-1 role as well.

I'm noticing that head 0.4 has the largest QK magnitude, and its low-frequency planes line up almost exactly with 0.5's, which suggests a similar offset of about 3 there, though with less weight.

I'm working out whether 0.3 could be a mix of self- and previous-token attention, since its angle pattern doesn't fit a clean offset the way 0.5's does. The high-frequency angles on 0.4 might point to an offset of 3 as well, but I'd need the amplitudes to say how much weight each plane carries.

I'm checking whether the 0.3 angles fit a mix of offsets rather than a single clean one. Their growth is uneven, with a negative intercept, so a blend of two offsets might explain it better than a pure previous-token pattern.

I'm testing whether the 0.3 angles fit a mix of self- and previous-token attention with a constant weight ratio. The fit doesn't hold: the implied ratio drifts from about 0.15 at the low harmonics to about 0.5 at the high ones, so a simple two-offset blend doesn't explain the 0.3 plane.

I'm weighing whether 0.3 could be the previous-token head, since it has the smallest nonzero offset and its flat OV pattern looks like token copying. I'm also wondering whether 0.4 or 0.5 might instead be the one doing a local lookback while 0.3 handles the current token.

I'm checking whether 0.3 could be a self-attending head rather than a previous-token one, since its output feeds the queries of later heads more than the keys. If it's the query side reading the current token, a self-attending role in layer 0 would fit that pattern.

I'm checking whether head 0.2 could be the one supplying the previous-token signal to the keys, since its QK is nearly rank-1 with little positional weight. I'm also wondering if a key-side preference along some positional direction could explain it instead.

I'm wondering whether head 0.2's query-independent key bias gives it a recency preference, so that it pulls in the most recent positions. I want to compute each layer-0 head's static key preference and compare its norm against the QK term, since a large value would point to a fixed positional or token-driven attraction.

I'm checking whether head 0.2's key-side direction lines up with the positional planes from 0.3 and 0.5, which would suggest it's positional rather than token-driven.



---

Head 0.2's top QK direction also looks largely positional, and 0.0 and 0.1 are partly so, which suggests most layer-0 heads carry some positional signal. For head 0.2 specifically, I'm checking whether a rank-1 positional QK could favor the most recent tokens.

I'm working out how to get a plausible positional encoding from the per-plane structure, since the phases and orientations aren't directly known. The idea is to treat each plane as a rotation at a harmonic frequency and see whether the resulting attention score as a function of offset gives a usable shape.

I'm working out how the plane orientations and harmonic angles fit together, since the signs for head 0.5 look inconsistent across planes. If I assume 0.5 attends backward by a fixed offset, its angle per plane should be k times that offset, so I'm comparing head 0.3's angles against the same planes to see whether the orientations line up.

I'm checking whether head 0.3's per-harmonic angles, roughly 8.1k degrees, imply a small positive offset, the same sign as head 0.5, and I'm weighing how a mix of self and offset attention would bend that phase.

Working out whether 0.3's per-harmonic angles, roughly 0.3 times 8.1k degrees, actually pin down the offset: if Δ5 is 3, then head 0.3 is a previous-token head, but a value near 1 or 2 would mean a mixture of self and offset attention.

I'm weighing whether head 0.3's Layer-1 readers depend on what 0.2's output carries, since keys would need to hold the previous token for an induction-style match.

I'm checking whether layer 1 or 2 heads show an induction pattern, using the composition of a candidate previous-token head's output with the query-key circuit of a later head. If that composed matrix has a positive, identity-like structure over the token space, it would indicate induction.

I'm going to check whether the composed matrix QK_b @ OV_a has an identity-like, positive structure across pairs of earlier and later heads, using the trace relative to the eigenvalue magnitudes as a measure of how cleanly it matches tokens.



---

I'm looking at which attention heads have high positivity, and the pairs that stand out are the 1.3 and 1.0 heads attending back to 0.2, with several others following that pattern. The 2.1 and 2.3 heads both draw from 1.1, and the layer 3 heads seem to be attending to 2.0.

Now I'm checking the layer 3 heads attending to 2.0, where the values are high. Its OV has negative eigenvalues, yet the product with QK_3.1 comes out positive, which I can't explain from the traces alone. The K-composition magnitudes are only moderate, so I'm wondering whether 2.0 could be acting as a previous-token head.

I'm checking whether 0.2 might act as a previous-token head: its QK top direction is positional, but its OV output seems to carry some token identity. Next I want to test the positional planes of the layer-0 heads, particularly the 0.2 weight in plane 22.4 and 30.6, to see if that supports the previous-token idea.

I'm weighing a different route: rebuilding the positional embedding as a synthetic helix so I could compute the attention scores on constructed inputs. But that would amount to running the attention computation on hand-made vectors, which is ruled out, so I'm setting that idea aside.

Ruling out a synthetic construction of the positional embedding, I'll stick to eigendecomposition-based weight analysis. Reading the harmonic-plane angles, 0.5 looks like a clean single-offset head, while 0.3's angles are too large to be self-attention, so it probably has offset 1, meaning it's a previous-token head.

Checking whether 0.3 acts as a previous-token head, I'm testing its K-composition with the layer-1 and layer-3 heads through QK_b @ OV_0.3, where the layer-3 heads show the strongest positivity.

I'm checking whether 2.0's output, which gives the strongest match for the layer-3 heads, is also a previous-token signal. If 0.3 is the previous-token head, then 3.0, 3.1, 3.4 and 3.5 would be the induction heads reading it, so I want to confirm that the layer-1 and layer-2 heads don't show the same pattern.

I'm checking whether 2.0 also carries a previous-token signal in layer 2. Since the residual stream still holds the positional structure, I'll look at its rotation angles in the harmonic planes from 0.5 and compare them across all the later-layer heads.



---

I've established that heads 0.3 and 0.5 are the clear positional ones, with 82% and 76% of their QK norm in those 12 planes. Head 0.4 draws on different high-frequency planes, so I want to check its positional fraction within its own set. I'm also wondering whether later layers drop these planes altogether.

I'm checking whether head 0.3 is the previous-token head, since its full-rank flat OV and the layer-3 query matches point toward induction. I'm also testing whether 0.3 could have offset 0 with 0.5 at offset 1, but the ratio rising with harmonic index doesn't fit a mixed offset.

I'm checking whether a mixture of offsets could explain the ratio: a 0/1 mix gives sublinear phase, but the observed ratio rises with harmonic index, so that doesn't fit cleanly. The rising ratio might instead reflect q and k planes differing in amplitude or the PE frequencies not being exactly harmonic.

Checking whether head 0.2 attends by position in the same high-frequency planes as 0.4, the way I'd check the other layer-0 heads, and whether its top QK singular vectors line up with anything.

I'm checking whether head 0.2's rank-1 QK term picks up the constant bias β from the layer norm, which would make its query static and its key preference fixed along the value direction. If the top query-side vector lines up with β, that would explain the pattern, so I'll compute the alignment.



---

I'm checking the query and key contributions for head 0.2: the query side is about 58% positional and the key side about 43%, with the 0.4 planes at 16%. Head 0.2 looks somewhat positional, with its rank-1 component dominant, while 0.0 and 0.1 seem to be content heads. For layer 0 I'm guessing 0.3 acts as a previous-token head, and 0.4 and 0.5 attend to a short local window a couple of positions back.

Now I'm moving on to layer 1, checking whether its heads take keys from head 0.2 and whether the previous-token signal from 0.3 shows up there, which looks weak, so layer 1 probably isn't doing induction. Layer 3 has strong positive scores against 0.3 and 2.0, and I'm wondering whether 2.0 acts as a previous-token head in layer 2. Its QK trace ratios point to content-matching heads in the later layers, since positional info no longer sits in the original planes.

I'm checking whether the later heads use positional attention at all, since layers 1 to 3 show essentially zero QK weight in the positional planes. To test for an induction head, I'm looking at whether the keys read the previous-token head's output, and the QK-through-OV_0.3 positivity is highest for 3.1 at 61, then 3.4 at 42, 3.0 at 35, and 3.5 at 32. The K-composition from 0.3 into 3.1 looks modest, while the link from 2.0 into layer 3 is much stronger.

I'm now testing whether head 2.0 is itself a previous-token head by checking its QK against the 0.3 output, which comes out at -37, so it avoids keys whose previous token matches the current one. Its own QK trace is positive at 1.32, which makes me wonder whether it's attending to tokens that match the current one instead.

I'm now testing a different idea: whether 2.0 might be a duplicate-token head that copies info about earlier occurrences of the current token. To check any head's self-matching, I'll look at the QK eigenvalues directly, and I'll score induction-style composition with QK against the OV output of 0.3 for each candidate.

Next I'll print the eigen-positivity of the QK matrix for every head to see which ones match on content.



---

I'm wondering whether the negative OV contributions in layers 1-3 come from layer norm scaling and the residual stream's large components, with heads partly cancelling the attended token's embedding. I'm not sure that's right, and another possibility is a large positional component in the residual stream.

I'm checking whether the OV orientation is right: OV = Wo @ (Wv * g) maps residual to residual, so that looks fine. The negative eigenvalues in later layers versus positive ones in layer 0 still don't have an explanation I trust, so I'll set that aside. Next I want to pick out the layer 1-3 heads by a combined induction score, using K-composition with the presumed previous-token head and the offset-3 heads.

I'm going to compute the kcomp measure for the layer-3 heads too, since the earlier print cut off after twelve. For the layer 1-2 heads, the reads from 0.3's output came out around 1.0 to 1.2. I expect 0.3's OV to be high-rank and flat, so the relative measure should sit near 1 unless something is strongly aligned.



---

I'm looking at head 2.0, whose output the layer-3 heads 3.1 and 3.0 read through their keys, with strong positive scores. Its own query-key self-score is fairly high, so it seems to attend to similar content, and its interaction with OV_0.3 is strongly negative, suggesting it avoids positions whose previous token matches the current one. That makes me wonder whether heads 3.0, 3.1, 3.4 and 3.5 form an induction circuit.

I'm weighing whether 2.0 is a previous-token head, since 3.x keys read its output, but its negative interaction with OV_0.3 and its positive self-match point toward a duplicate-token role, which would only repeat same-token matching and tells me little about an induction circuit.

I'm checking whether layer-3 heads 3.0, 3.1, 3.4 and 3.5 could be induction heads, since they read positive composition with the previous-token head 0.3 on the key side. Layer-3 heads that write straight into the readout would fit that role, and 3.1 looks strongest. I'm also wondering whether head 0.2 might be the previous-token head instead.

I'm checking whether layer-0 head 0.2 could be the previous-token head instead, since its key-side contribution to layer 1 is much larger than 0.3's. That would make the classic two-layer induction circuit, with layer-1 heads feeding a layer-2 induction step. I'm also wondering how a rank-1 query-key form could single out offset one, given that it would need either a rotational structure or a steep recency ramp with self-suppression.

I'm weighing 0.2 against 0.3 as the previous-token head: 0.2's output carries the most in its top singular directions, but its positional query-key structure reads more like a word-start detector than a clean offset-one match. That leaves the layer-1 heads matching current tokens against word-start content, which doesn't look like induction, and I'm still checking how 0.3 fits in.

I'm wondering whether 0.3 is really a near-self head, since its output seems to feed mostly into MLPs as bigram features and the queries of layers 1 and 2. Going by the K-composition numbers, 0.2 looks like the stronger candidate for the previous-token role, so I'm checking the positional structure of 0.2 to see whether it fits.

I'm checking whether head 0.2 could be the real previous-token head, since its contribution to layer 1 is much larger than 0.3's. I'm also wondering whether a rank-1 query-key form can pick out offset one, and I'll look at how its component splits across the planes.

I'm wondering whether head 0.2's query-key pattern could act like a previous-token detector: a rank-1 score can alternate sign with position at a near-Nyquist frequency, which might favor odd offsets, and I want to check its top singular pair against the planes from heads 0.3, 0.4 and 0.5.

I'm checking whether the rank-1 component of head 0.2 can work as a previous-token detector, since a near-Nyquist plane would let its sign alternate with position and favor odd offsets.



---

I'm comparing the query and key vectors for the top QK pair in 0.2, where both have sizable components in a plane around 22–30°. That range overlaps the angles I saw in 0.3, 0.4, and 0.5, and I'm wondering whether this is a non-harmonic plane.

I'm working out whether 0.3, 0.4, and 0.5 share a single offset, and I'm settling on 0.3 as the previous-token head, with 0.4 and 0.5 as relative-position heads reaching a few tokens back. I'm also testing whether 0.4 might be offset 1 through its high-frequency planes, but its low-frequency angles match 0.5, so the offset looks the same.

I'm checking the ratios at the highest harmonics: 0.5 sits around 3× larger than 0.3, which points to 0.5 at offset ~3 and 0.3 at offset ~1. I'm also weighing 1.1, whose weak QK suggests broad averaging rather than previous-token behavior, so I'm setting it aside and looking at the layer-3 heads that compose with 0.3 for induction candidates.

I'm checking whether 2.0 might be a later-layer previous-context head, since its keys are negatively matched to the previous token and its output feeds the induction-like heads in layer 3. I want to test its positional usage next.

I'm considering how a layer-2 head could attend to the previous position without any direct positional signal, and whether it could do so through content: if the query reads the output of head 0.3, which carries the previous token, and the key reads the current token, a match would pick out the previous token. I still need to check the positivity of that query-side composition before deciding.

I'm going to check the query side of the composition next: if head 0.3's output feeds the query of a later head, does that head end up favoring positions whose token matches the previous token? I'll compute that positivity for the later heads to see which ones would act like previous-token attention.



---

I'm checking which heads take the query from position i-1 through the 0.3 path, and the candidates are 2.0, 1.4, 1.3, 1.2, 1.5, and 2.2, with 2.0 standing out on the highest weight. The working hypothesis is that these heads match on the previous token and attend to earlier positions holding that same character, so I'm wondering whether they then copy whatever sits just after those matches.

I'm weighing whether heads like 2.0 are doing induction-style matching: an induction head would look for keys whose previous token equals the current one and copy the token after, but these heads seem to match the previous character on the query side with a negative key-side signal. So they may instead be attending to recent occurrences of the current or previous character, which doesn't fit a clean induction pattern.

The output of head 2.0 feeds the keys of the layer-3 induction heads, so it acts as a content-based previous-character carrier rather than a clean induction head itself. I'll treat 2.0 as a previous-char-matching head that supplies information to the layer-3 induction keys. The same pattern may hold for layer-1 heads 1.2 through 1.5, which combine their query with the 0.3 signal and likewise look for earlier positions holding the previous character, with K-composition from head 0.2.

I'm sorting out whether head 0.2 behaves like a previous-token head, since its eigenvalue pattern could support that, though its effective rank makes it unclear. For now I'm settling on 0.3 as the clear previous-token head, with 3.0, 3.1, 3.4, and 3.5 as the induction heads.

I'm weighing whether head 1.3 belongs in the induction group, since layer-1 heads get their strongest key-side composition from 0.2. I'm checking 0.2's query-key structure in the plane where it sits alongside 0.3 and 0.4, looking at the rotation angles and strengths there.

I'm now checking the frequencies in that plane, where the ratio between heads doesn't match a clean rotation, so it may be a low-frequency positional component rather than something periodic. I'll keep 0.3 as the previous-token head and treat 0.2 as a positional or delimiter-focused head for now. As for language, I'm looking for more evidence, since the apparent period of about 43 positions seems odd for learned positional encodings.

I'm now considering whether the period of about 43 positions, or the 28.6 or 57 alternatives, could come from a periodic positional signal. That would only make sense if the training data were chunked at fixed-width lines, so absolute position modulo the line length carries meaning.

I'm wondering whether the harmonic pattern could come from a smooth learned positional encoding instead, where the top planes are just low harmonics of a 256-position period. If so, the offsets would be about 18 and 6 positions back, which seems odd for a character model unless the 6 lines up with word length.

I'm weighing whether the periodic-looking planes could just be the smooth low harmonics of a learned encoding over the full context, in which case the apparent offsets of about 6 and 18 would be broad averaging rather than precise lookbacks. That leaves the previous-token role open, since it would have to come from the higher-frequency planes instead.

I'm now checking whether the high-frequency planes used by head 0.4 (angles around 117° to 168°) carry a clear offset-one signal, which would mark it as the real previous-token head.

I'm weighing two readings of the 0.4 and 0.5 angle matches: if 0.4 has offset 1, then 0.5 would too, making both previous-token heads and leaving 0.3 as a self-plus-previous mix. The alternative keeps 0.3 at offset 1 and pushes 0.4 and 0.5 out to offset 3.

I'm testing the high-frequency planes next, comparing how 0.3, 0.4, and 0.5 rotate in them to see which offsets fit.

I'm comparing how heads 0.3, 0.4, and 0.5 rotate in their high-frequency planes, checking whether the angle differences fit an offset of 1 or 3 positions for each head.

I'm testing whether the high-frequency planes separate the offsets: if 0.3 fits one lag, 0.4 and 0.5 should follow the same multiple, but their angles disagree, so they can't share an offset. I'll try solving for the pattern directly by comparing 0.3's angles against 0.4's in those planes.

I'm checking whether head 0.4's high-frequency rotations give an offset of 2 for 0.3, and I'm trying to find an offset for 0.5 that fits its angle of 65° in the same plane.

I'm checking whether taking head 0.3 as offset 1 and head 0.4 as offset 2 fits the high-frequency plane angles, since that would give a consistent set of lags.

I'm finding the high-frequency plane angles don't line up consistently with any single offset for heads 0.3, 0.4, and 0.5, so the rotation approach is too noisy to settle which head is the previous-token one. Instead I'm leaning on how the later heads use 0.3's output, with its structured positive matching and flat copy-like OV, as evidence that it carries a specific neighbor's identity, while 0.4 and 0.5 are still unresolved.

I'm weighing whether the next probe could reveal the vocabulary size, perhaps through the tail of the block-3 projection spectrum, and I'm inclined to estimate it around 85 to 100 characters for now.

I'm weighing whether the vocabulary might be around 90 characters, which would fit English prose or Python code more than a small Shakespeare-style set. I don't have a strong basis for choosing between them yet, so my confidence in any particular dataset stays low.

I'm checking whether the x̂ term could push the readout rank beyond the vocabulary size, since the spectrum keeps decaying past 64 without a clean cutoff.

I'm weighing whether the vocabulary size can be read off the spectrum's knee, which looks like it sits near 90. I'm also wondering whether a periodic positional encoding would only make sense if the training text had fixed-width lines.

I'm weighing a way to pin down the readout rank by fitting a two-line regression to the log singular-value spectrum, to locate the break point where the decay changes.



---

I'm looking at the block 3 MLP neurons through the columns of the projection, where each neuron's output vector is what it writes into the residual stream. Mapping those into readout space and counting distinct token directions looks hard. Another angle is to find V from the final-layer writes, since logits depend on the layer-normed residual and anything written orthogonal to the unembedding span is wasted.

I'm checking the block 3 projection spectrum directly for a knee near index 64, since V-1 would be expected there, but the drop isn't visible at 64 and the steepest falloff sits between roughly 73 and 88. That points to a vocabulary somewhere around 85 to 95, though the tail values are small enough that the exact edge is blurry. I'm weighing that against common character-level corpora, with English prose as the most likely guess.

I'm considering whether the weights could tell code and prose apart: code might show stronger copying from earlier variable names and indentation-tracking heads, though prose could show those too. I'm also wondering whether a periodic positional pattern would point to some fixed-length record structure in the training data.

I'm thinking about whether the token directions could be pulled out of the block 3 neuron outputs, perhaps through ICA if each neuron promotes only a few characters. Even then, with around 90 directions, I wouldn't be able to name them, and I'd have no labels to attach. So I'll settle on English as the most likely guess, though with low confidence.

Now I'm weighing whether the rest of the head labelling can wait, since I have the same working guess of English prose at low confidence. I'd put that at about 0.35 and leave it there, then turn to the later heads.

Now I'm going through the layer-0 heads one at a time: 0.0 is a weak content-based head, 0.1 seems to avoid attending to its own token, 0.2 looks like a delimiter-tracking head that later layers compose with, 0.3 is the previous-token head, and 0.4 is a sharp local relative-position head.

I'm turning to layer 1, where the heads read what layer 0 wrote: head 1.0 seems to pick up the delimiter signal from 0.2 on both its query and key sides, which suggests a same-segment matching role. Head 1.1 looks like a broad averaging head with a large output but weak attention pattern, while 1.2, 1.3 and 1.5 appear to compose a key from 0.2 with a query from 0.3, so they probably match the previous character against earlier positions.

I'm going through the later heads now: 2.0 reads the previous-token signal and feeds the layer-3 induction keys, so it looks like a pre-induction helper. Heads 2.1 and 2.3 key off layer 1's averaging head, 2.2 seems to match duplicate tokens, and 2.4 and 2.5 pick up segment and delimiter structure.

Next I'm checking whether any layer-3 heads write the induction copy into the readout, and I'm skipping the explicit OV check since I can't verify it without the unembedding. I'll note layer-1 as a possible induction contributor but leave it off the list.

I'm settling the language question here: the next thinking keeps the English-prose guess at about 0.3 confidence and drops the 2.0 previous-token idea since it isn't positional. The induction-head check is set aside as unverifiable without the unembedding, so I'm finalizing the head labels with English at low confidence.

