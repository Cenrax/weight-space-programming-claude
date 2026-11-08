I'm trying to work out the language from only the middle-block weights, with no forward passes, which looks very hard. I'm wondering whether the layer-norm weights of block 0 or the structure of the embedding subspace could reveal the vocabulary size instead.

I'm weighing whether the vocabulary size might show up as the rank of the token embedding, though the block 0 weights only see the combined embedding and positional stream. I'm also considering that the data type could be something like DNA or code, which would leave a distinct pattern in the weight spectra, and that weight decay may leave the unused input directions of W_q near zero.

I'm noting that the positional embedding covers 256 positions and might span many dimensions, so it's not obviously low-rank. For the heads, I can't use the previous-token heads in layer 0 directly since they depend on positional embeddings I don't have, so I'll look at composition instead: induction heads in later layers should show K-composition with earlier prev-token heads, and I want to score that between layer-0 OV and layer-1+ QK, plus the q/k biases.



---

I'm checking why every matrix has a near-zero smallest singular value, and suspect the residual stream at position 383 is the reason. My first guess is LayerNorm's mean subtraction, since the normalized vector is orthogonal to the ones vector. But the affine weight and bias seem to break that constraint, so I'm not convinced yet.

Now I'm checking whether the spectrum of block 0's input-side weights shows a rank deficiency, since the token and position embeddings might span fewer dimensions than the residual stream has. I'll look at the full singular value profile for q, k, v, and the MLP input, scaled by the first LayerNorm's weight.



---

I'm looking at the block-0 MLP input spectrum, where there's a knee around index 128 and then a slow plateau, which might just be leftover random-init weights. That would suggest roughly 100 to 130 dimensions are actually in use, though this mixes token and position information, so it doesn't map directly to vocabulary. I'm also wondering whether the tail of the spectrum is that random remnant.

I'm going to check the fc spectrum more closely, looking at the first 200 values to see whether the drop past the learned directions is real or just the random tail.



---

I'm stuck on the lack of a sharp knee, so I'm switching to a different angle: guessing what kind of text the task designer likely picked. The options range from English and Python to a non-Latin script or DNA. I'd need evidence from the weights to tell them apart.

I'm checking whether the block 3 MLP output spectrum has a knee near the vocabulary size, since the final layer mostly writes to logit-relevant directions. If the vocab is small, something around 65 should stand out.



---

I'm looking for a knee in the curve around index 76 to 78, where the decline in the gaps slows down. The values drop from 2.17 to 1.92 and then flatten out, so I'm estimating the knee at roughly 90 and wondering whether the vocabulary size falls somewhere around 65 to 100.

I'm weighing whether the vocabulary is something like 65 (Shakespeare-style English) or around 95 printable ASCII for code, but the gap curve alone doesn't settle it. I'm now considering whether the last-layer MLP output directions cluster by character, which could reveal the token count.

I'm now going through the head-by-head analysis: checking each head's QK and OV spectra, the bias contributions, and how the heads compose with one another.

I'm now defining the head-composition measures: K-composition, Q-composition and V-composition between heads, each as a Frobenius norm of a product of the residual-space OV and QK matrices, normalized and compared against a random baseline. I'm also thinking about how induction heads would show up as strong K-composition with a previous-token head.

I'm adding a projection that removes the mean from the inputs, so the centering matrix gets applied on that side. I also suspect previous-token heads could sit in layer 1 as well, so I'll compute those too.



---

I'm noting that layer 0 head 2 and head 1.1 stand out for K-composition, with layer 1 heads all showing some link to 0.2, especially 1.0. I'm also wary that the Frobenius score may be inflated by bias or large-norm directions, so I want to inspect the specific structure too. Layer 0 head 2 looks like a previous-token head, and 1.1 may be one as well, since layer 2 heads compose with it.

I'm now looking at the positional side of head 0.2, since the raw weights don't include positional embeddings, and I'm wondering whether the positional subspace can be recovered from the QK structure. For a previous-token head, I'd expect the positional QK block to peak one step back, so I'm planning to check the QK and OV spectra per head and whether any antisymmetric, rotation-like part shows that shift.

I'm checking the antisymmetric versus symmetric parts of each head's QK matrix, since a large rotation-like component could point to a positional offset head. I'm also going to score each head's OV spectrum for how much it copies, using the ratio of positive real eigenvalues to total magnitude.



---

The copy scores for the OV circuit in residual space come out negative, which is odd. Full-residual OV eigenvalues don't really mean anything for copying here, since the writes land in a different subspace, so I'd need embedding and unembedding to make that measure meaningful. Heads 3.0 and 3.1 in layer 3 have large positive QK traces, which would suggest they attend to matching content, while the layer's QK spectra look flat and spread out.

QK scores come out high for heads 3.1 and 3.0, which fits the positive trace pattern. I'm still unsure whether that means induction-style matching, since the previous-token information would sit in a different subspace unless an earlier head writes into the same one.

I'm working out the K-composition for layer 2 heads against layer 1 heads, subtracting a random-matrix baseline of about 0.05 to see which pairs stand out. The 0.2→1.0 pair scores highest at 0.18, with 1.2, 1.5, 1.3 and 1.4 trailing behind. Next I want to check Q- and V-composition the same way.



---

I'm checking how the 0.2 layer feeds into 1.0 through both the query and key paths, since both show a sizable contribution. The QK singular values are dominated by one direction, so I'm wondering whether 0.2 is effectively routing attention to a single fixed target, maybe a particular position or token.

I'm weighing whether the one dominant direction in 0.2's QK matrix might be the positional subspace. Layer-0 positional heads would pair query and key positions, and the block-0 inputs all share the same span, so the singular values don't cleanly separate. I'm also considering the v and o biases as a possible factor.

I'm considering whether the designer expects a language guess from the weights, perhaps via repeated-structure signals like induction-head strength, or whether low confidence is the honest answer. I'm also thinking the first layer's MLP might act like a bigram lookup, and that the final unembedding's row space could reveal something about the vocabulary size.

I'm reasoning about the output space of the last layer's MLP projection, where I expect its column space to be roughly as large as the vocabulary. The spectrum shows a knee somewhere around 76 to 90, so I want a more careful effective-rank estimate, comparing layer 3 against the other layers and looking for an elbow in the log singular values.



---

I'm estimating the knee around 88 to 90, which puts the effective output dimensionality near 85 to 90. Working from that, I'm guessing the vocabulary might sit somewhere around 85 to 95, though I'm not sure. Comparing against character sets, Shakespeare's tiny corpus has about 65 characters, while Python code with printable ASCII plus newline and tab comes to roughly 96 to 100, and English Wikipedia-style text might land above 90. The knee isn't clean, so I'll keep those numbers loose.

I'm now thinking about the neuron output directions in the final projection, wondering whether clustering them could reveal character-specific directions, such as the one for space. Still, that wouldn't tell me which characters they are. I'm also considering whether a frequency-based approach through the unembedding structure could help, though that idea isn't getting me anywhere yet. Given the knee near 85, I'm narrowing things to alphabetic scripts with roughly 60 to 100 symbols, and weighing English prose against Python code and accented European or Cyrillic text.

I'm looking at the layer 0 attention heads and thinking about whether they track token identity directly, but the QK structure restricted to the embedding subspace is unknown to me so far. A better route might be the input spectrum of layer 0, combining the token and positional subspaces, with the MLP and attention projections together. Earlier the MLP spectrum declined smoothly and then plateaued, so I want to compute the derivative for the combined block to look for a similar knee.



---

I'm weighing whether the unembedding is tied to the input embeddings, though that isn't given. I'm looking at an approach based on the copy structure: the residual holds token identity directions, and a previous-token head's OV circuit maps one token's embedding to a direction that a later induction head's QK can match against.

I'm checking whether the induction matrix M = QK2 @ OV1 is close to identity on the token subspace, using its trace normalized by the Frobenius norm, and I'm looking at the eigenvalue signs too.



---

The strongest signal so far is head 2.0 writing to 3.0, 3.1, 3.4 and 3.5, with scores ranging from 0.72 to 0.98. Layer-3 heads seem to read its output through their key-side QK, but I'm now wondering whether the query side picks up the same OV(2.0) contribution, since head 2.0 writes to every position.

The 2.0 → 3.x writes are strong, so I'm now asking whether layer-3 heads match 2.0's output on the key side, and what that implies. If QK3 is positive-definite on OV2.0's output subspace, the query should carry raw content that matches the key's head output, which is what an induction pattern would look like. That would make 2.0 a previous-token-style head and 3.0, 3.1, 3.4, 3.5 induction-like, though it could just be a broader previous-context head.

I'm checking whether head 2.0 is just an identity-like copy head, which would make the trace trivially positive, but its copy score is negative, so that's not the explanation. Layer-3 heads also show composition with 1.x and 0.x heads at moderate strengths. I'm now wondering whether 2.0 is attending to the previous token, though its QK norm and trace ratio are high enough that I can't tell yet.

Now I'm wondering whether layer-3 heads 3.0, 3.1, 3.4 and 3.5 are induction-like, reading head 2.0's output as previous-token information, since the tr(QK@OV) positivity fits that pattern. I still can't see positions directly, so I'd need a previous-token head in layer 0 or 1 to confirm that 2.0 is the one doing the previous-token work.

I'm checking whether layer-1 heads 1.0, 1.2, 1.3 and 2.1, 2.3 also feed the later layer-3 heads, with 0.2 and 1.1 as the likely upstream sources. The tr(QK@OV) signal from head 0.3 and head 1.5 into 3.1 and 3.4 also looks like a second candidate pathway worth separating from the 2.0 story.

I'm trying to tell whether head 2.0 is attending to the previous token, which would be the usual role for a head feeding induction heads. The trace test can't settle that alone, because a head that attends to its own position would look similar. I'd need positional information, which I can't read directly from these weights, so I'm considering whether layer-0 heads that are mostly driven by position could reveal it.

I'm checking whether head 0.2 can serve as the positional source, since its query-independent rank-1 component points that way. Its bias term looks larger than the QK contribution, so the attention may be driven mostly by position rather than by token content.

I'm now considering that head 0.2's attention might be a key-only preference rather than a query-dependent one, since its bias term outweighs the QK contribution. A pure key-side score can favor recent positions or specific features, but it can't tell apart the current token from the one before it, so it would more likely pick out something like the last space or newline.

I'm now weighing whether layer-0 heads 0.3 and 0.4 carry the positional signal, since their QK structure suggests they attend to nearby positions. For a previous-token head, the score would peak at offset one, so the QK would need a shift-like pattern in the positional subspace. I'm looking for a way to pull that positional subspace out of the weights.

I'm checking whether layer-1 heads 1.0, 1.2 and 1.3 feed the layer-3 heads through head 0.2, since those composition scores are positive. Head 0.2 might be a positional source, but I can't pull the positional subspace out of these weights directly, so I'm still weighing how to test whether it's a previous-token head.

I'm checking whether the strong 2.0 → layer-3 trace positivity (0.94–0.98) reflects a real matching circuit, and I'm weighing whether it's just QK3 being positive on the subspace where 2.0 writes, rather than a meaningful composition.

I'm checking whether the combined trace over the 3.x heads is positive for a matching reason, or just because QK3 happens to be positive on the subspace where head 2.0 writes. The next step is normalizing these traces by the norms of QK and OV, since the raw values could be inflated by magnitude alone.

Still normalizing the eigenvalue sums...



---

Checking how strongly 2.0 writes into the key positions that heads 3.0 and 3.1 read: the normalized trace comes out around 0.31, against a random baseline of roughly 0.01 to 0.05, so that link looks real. I'll also note that 2.0 has a negative K-path interaction with layer 1 heads, which is worth tracking.

I'm checking whether 2.0 is a previous-token head, but its negative K-path traces with the layer-1 heads don't fit that reading cleanly. Its own QK trace is positive, so the induction-style picture still looks like the most plausible one, with 2.0 feeding into 3.0, 3.1, 3.4, and 3.5.

I'm now thinking about whether 2.0 is really a previous-token head, since its positional attention would have to come from a subspace I can't yet identify. The likelier candidates for previous-token duty look like 1.1 and 0.2, while 3.0, 3.1, 3.4, and 3.5 stand out as the induction-style heads to check.

I'm working out whether layer-0 heads can be told apart by where their query and key subspaces sit, since a previous-token head should read positional information on the query/key side and token information on the value side. To test that, I'd compare the key-side weights of the layer-0 heads against the value-read subspaces of the others.

I'm checking whether 2.0 reads positional information, since a previous-token head would need that on its query and key sides. The plan is to test whether 2.0's query-key subspace lines up with the token subspace that the layer-3 heads' queries read, or with a separate positional one.

I'm trying to pin down whether 2.0 is really doing previous-token work, which would mean its query and key subspaces carry positional information. One way to test that is to estimate the positional subspace from the layer-0 heads with sharp QK patterns, since those would need to separate positions. I'll also compare whether 2.0's value side reads the same subspace that 3.1's query reads, as the trace suggests.

I'm trying to estimate the positional structure from the rotation-like form of the query-key product, where the symmetric part dominates and gives a positive trace.

I'm looking at head 0.3, which has a fairly flat QK spectrum and a strongly positive trace, so it could be a local positional head or a same-token head. Its OV copy score is near zero but its OV norm is the largest in layer 0, and layer-3 heads read from it with a positive trace. Next I'll compute the eigenvalues of its QK matrix to see whether the structure looks rotational.

I'm going to compute the eigenvalues of each layer-0 head's QK matrix and look at the angles of the top ones, to see whether the structure is rotational like a positional encoding.



---

I'm noticing that the 0.3 and 0.5 eigenvalues come in conjugate pairs with small-to-moderate angles, which looks rotation-like and points to positional heads. The 0.5 head's angles (26, 50, 76, 97) are roughly 1x through 4x multiples of about 25°, which resembles a shift operator acting on Fourier-like positional embeddings, since a shift by Δ rotates each frequency by a different amount.

I'm checking the 0.3 and 0.5 angles against the positional-rotation idea: a shift by Δ should turn each frequency by a different amount, and 0.3 also shows small rotation-type angles. The 0.4 head's larger angles don't fit that pattern as neatly, so I'm pausing on it. The harder question is the sign of the rotation, since the conjugate pairs alone don't settle which way the offset runs.

I'm weighing whether 0.4 and 0.5 are driven more by key-only bias, since 0.4's bias term outweighs its QK magnitude. Layer-0 head 0.5, with its base angle near 26°, could be a previous-token head, while 0.3 might be a broader local averager. I'm also starting to check composition, beginning with 0.3's positive trace.

I'm checking whether layer-2 head 2.0 is a positional rotation head too, since its angles (72, 10, 5, 48, 0, 21, 13) look rotation-like and its key-path composition into 3.0 and 3.1 is strongly positive. That would fit a previous-token head feeding an induction pair. The 72° angle also shows up across several heads in layers 1 and 2, which suggests a shared positional plane.

I'm noticing the 72° angle keeps recurring, which is 360/5, so it might reflect a period-5 positional structure or just a shared plane. To tell them apart, I'd compare the 2D invariant subspaces of each head's top complex eigenpair and check how much they overlap.



---

Two heads in layers 1 and 2 seem to share a rotation plane of about 72°, with overlap between 0.5 and 0.8 out of 1.41. Layer 0 heads 0.5 and 0.3 sit on a different plane, the one carrying the dominant positional component.

I'm weighing whether the shared plane of layers 1 and 2 could come from non-positional query-key feature pairs, or whether it might encode position as a rotation. The layer-0 plane looks different, and its overlap with the shared plane is too low to match. I'm also running short on budget, so I need to settle on a direction soon.

I'm checking whether the shared 72° plane in layers 1 and 2 could be positional: I'll project it onto the outputs of layer-0 heads and the layer-0 MLP, and see whether anything earlier writes it or whether it comes straight from the embeddings.



---

Checking whether the plane still holds up: the OV(0.2) output overlaps it at 0.65, against 0.32 for random, so the result isn't conclusive. Rough head labels may be enough for now, though I want something firmer for induction and previous-token heads. Next I'm sorting layer 0 by circuit role, with heads 0.3 and 0.5 looking positional.

I'm weighing whether the 0.0, 0.1 and 0.2 heads are content-based with only a weak key bias, perhaps attending to particular tokens or averaging broadly. For previous-token behaviour, head 0.5 looks like the strongest candidate, since its angles 26°, 50°, 76° and 97° form a clean harmonic series that fits a fixed-offset shift. Head 0.3 has a much smaller rotation in the same plane, which would mean it sits near the current position rather than shifting.

I'm checking whether the ratio of angles holds across matched planes: if 0.3 and 0.5 both give a consistent ratio, that would point to a shift relationship, and a zero-offset self term would show up as a separate component. But the eigenvalue angle may not equal the true shift when the basis isn't orthonormal, and the query and key gains also mix into it. Frequency reasoning is complicated here, since a 256-token context doesn't guarantee a neat per-position rotation in learned embeddings.

I'm also checking head 0.4, whose angles of 117°, 143°, 168° and 123° might reflect a shift in the opposite direction or a larger offset. To test that, I'll compare the top six planes for 0.3, 0.4 and 0.5 and look at their overlaps.



---

I'm checking which angle pairs line up across the two planes and how their ratios vary. The ratios aren't constant, so I'm weighing whether a scaled rotation would explain the spread, and then I'm considering a softmax over many positions, where the score in a given plane depends on the cosine of the angle difference.

I'm now testing whether the 0.3 offsets follow the same pattern: their spacing is roughly 4 to 6 per harmonic, which would make them about a fifth of the 0.5 offset, and I'm wondering whether the shared planes' rotation direction is consistent before reading anything into the signs.

I'm now restricting the query-key matrix to the shared plane, reducing it to a 2x2 form, and comparing the sign of its antisymmetric part for the 0.3, 0.4, and 0.5 planes to read off the rotation direction. Since the eigenplanes of a non-symmetric matrix give complex eigenvalues on the invariant subspace, I'm going to use the 0.5 plane basis for all three.



---

I'm checking that 0.3 and 0.5 rotate the same way across the first four planes, with 0.3's angles running about 0.12 to 0.29 of 0.5's. Plane 4 flips orientation, but that's arbitrary and consistent across both heads, so it still matches. That means 0.3's offset should be roughly 0.2 to 0.3 times 0.5's offset.

I'm weighing the offset values: if the 0.5 head uses an offset of 1, the 0.3 head sits near 0.2, which makes it mostly self-attending, and at 4 it lands near 1. The angles grow almost linearly with harmonic index, which suggests a fundamental around 25.5°, a period near 14 positions, and harmonics reaching roughly half that period.

I'm comparing the magnitudes next: 0.5's eigenvalues fall off from 0.81 to 0.20, while 0.3's stay flat around 0.55 with small angles. That flat profile with near-zero rotation looks like a head attending to the current position, so 0.3 may be mostly self-attending. Checking that against the 0.5 head's offset is what I'd do next.

Taking the 0.3 head as a mix of self and previous-token attention, I'm tentatively calling the 0.5 head the previous-token head (offset 1) and the 0.3 head a current-token or local head. I'm checking whether 0.5 could instead be offset 2 with 0.3 near 0.5. Head 0.4 is different: its large planes don't overlap with 0.5's, so that comparison doesn't hold for it.

Checking the later heads now: 0.5's output feeds into layer-1 and layer-3 heads only weakly, while 0.3's output feeds 3.1 and 3.4 more strongly, and 2.0 feeds 3.x heavily too. I'm wondering whether an induction-style query reads raw token embeddings rather than these head outputs, so 3.x might be using the previous-token signal indirectly.

I'm checking whether 0.3 could instead be the offset-1 previous-token head, with 0.5 at offset 4 or 5. Comparing the magnitudes, 0.3's planes stay fairly uniform with a spread of angles, which doesn't clearly settle which head is which.

Trying to pin down which offset fits: if plane 0 is the first harmonic, the spacing of the angles would imply a fundamental near 7° and a period around 50 positions, which would push the 0.5 head's offset to roughly 3 or 4 instead of 1. That leaves the offset question open, since the per-harmonic numbers don't fit a clean linear pattern through zero.

I'm working out whether a sharp previous-token head can be built from these fairly small angles, since a clean offset-1 discrimination normally needs larger per-position rotation at the dominant frequencies. With the 0.3 head's planes only turning 5 to 30 degrees, the cosine differences between neighboring positions look too small unless the overall score scale is large.

I'm estimating the logit scale from the positional component, roughly 4 per plane and about 20 overall, and noting that a 28° rotation gives only about 2.4 logits of separation between offset 0 and offset 1. The 0.5 head's harmonics reaching around 100° look like they would give a much sharper peak at offset 1.

I'm checking whether 360/25.5 ≈ 14 implies aliasing at Δ=15 and 29, which would make the 0.5 head's harmonics 26°, 51°, 76°, 97°. On that reading, 0.3's small angles put its peak near the current position, so it mostly attends to itself with some spillover to the previous token. That would make 0.3 the current-token head and 0.5 the previous-token head.

I'm weighing the two readings of the harmonic spacing: either 0.5 sits at offset 1 with a fundamental near 25.5°, or 0.3 sits at offset 1 with a much finer fundamental and 0.5 is further back. The 0.5 angles step by roughly 20 to 25° each, and the 0.3 angles run about 8 to 10° apart, with the two ratios falling between 0.12 and 0.32.

I'm noticing that 0.3's phases don't fit a clean fixed offset, since the ratio to 0.5's angles drifts from 0.12 up to 0.32. A pure shift would hold that ratio constant, so 0.3 probably isn't just a delayed copy of 0.5. It may instead peak at the current position with an uneven profile, where the phase at each frequency reflects that shape rather than a single lag.

I'm weighing whether a slope of 25.5° per harmonic is a cleaner fit for a shift of one position than for a shift of four, since a four-position offset would give a broad, less typical peak. I'm also noting that 0.3's output-vector strength is higher than 0.5's, which matters for how strongly it feeds later heads. Still, I lean toward 0.5 being the previous-token head, though I want to see which heads actually read its output before settling that.

I'm checking whether 2.0's query-key pattern reads the same positional planes as 0.5, which would tell me whether the previous-token signal comes from layer 0 or gets rebuilt in layer 2.

I'm checking whether 2.0's top query-key planes line up with the same positional planes that 0.5 uses, which would show whether layer 2 is reading the raw positional signal or rebuilding it.



---

Since only heads 0.3 and 0.5 read the raw positional planes, later layers have to get position information from layer 0 head outputs or from other planes. Next I'm tracing the induction circuit: the previous-token head writes the previous token's identity at each position, and the induction head's QK match then pairs the current token with that written output, starting from the K-path through layer 0.

I'm checking how strongly head 2.0 connects to the later heads, since its link to 3.1 is much stronger than what the positional heads show. Head 2.0's attention pattern comes from QK planes shared with layer-1 heads, but its direct inputs from layer 0 and 1 look weak, so I'm wondering whether MLP0 reshapes the positional signal first.

I'm wondering whether the later positional information comes from directions beyond the 20-dimensional planes I pulled from heads 0.3 and 0.5, since those were only the ones their QK used. I'm considering whether the 72° rotation plane shared by the layer-1 and layer-2 heads could be a positional signal, though its overlap with the other planes is small, so I'm thinking about rebuilding the positional subspace from the full QK row and column spaces of those two heads.

I'm comparing the induction score for the 2.0→3.1 pair, which is far above the others, with a z-score against a random-rotation baseline to see how unusual it really is.



---

Now I'm going through the transition effect sizes, where the 2.0 to 3.1 change stands out with a z-score of 182 and several other large shifts follow it.

Now I'm working through the remaining transition effects into 3.1 from 1.5 and 1.4, where they show up at moderate strength. I'm also weighing whether head 0.3 acts as a current-token head, since that would decide whether the 3.x heads attend to keys that match the query's token.

Now I'm weighing whether 2.0 attends to the previous token, since a standard induction setup would have the prev-token head at an earlier layer feeding the induction head, and the current measurement can't distinguish that from a same-token duplicate mechanism without knowing whether 0.3 is the current-token or previous-token head.

Now I'm weighing the 0.3 → 3.x paths against the raw embedding: if 0.3 carries roughly the current token, then 3.1's high trace with the raw embedding and with the 0.3 and 2.0 outputs would fit it attending to same-token positions.

Now I'm checking whether 2.0's OV reads mostly the token-identity subspace, and whether its value input lines up with what the layer-0 heads write, since that would show it carrying previous-token information forward.

Now I'm checking whether 2.0's OV reads the token-identity subspace, since that would show what it carries forward from the layer-0 heads.

Now I'm checking which layer-0 component writes the shared rotation plane that the layer-1 and layer-2 heads use, by measuring how much of each component's output lies in that plane against a random baseline.



---

The 72° plane for 1.2 is partly written by OV0.2, which is about 3.5 times random, and 0.2 is a content head that prefers keys. That's not very conclusive. I'm already at 17 calls, so I want to spend the next ones carefully, and the language question seems to carry the most weight. I'm wondering whether the period-5 rotation at 72° reflects some underlying structure, and I'm considering what evidence in the middle weights could point to language, with vocabulary as one possibility.

I'm estimating the effective rank of the block-3 outputs, since those should only be read by the unembedding, so their rank should sit near the vocabulary size. Layer 2 and earlier feed later blocks, so they should have higher rank. I'll compare the singular value spectra of layer 3 and layer 2 to look for a sharper dropoff at layer 3.



---

The drop in the ratio is steepest between about 76 and 92 and then levels off near 0.6, which puts the effective rank somewhere around 88 to 92. That rules out 65, since a Shakespeare-sized vocabulary would show the knee near 64. Checking the slope in steps of 4, it's roughly 0.06 to 0.08 through the 72 to 88 range, then it falls to 0.03 and 0.02 at 88 to 96.

I'm narrowing the knee to around 88 to 90, and the vocabulary size would be roughly 90, though the knee might undershoot V by a rank or two since rare characters get little gradient. Printable ASCII with a newline is 96, and typical code and English text land around 95 to 100, so that range fits. Shakespeare at 65 still doesn't.

I'm now thinking about whether the positional structure here could be periodic, with the harmonics at multiples of about 25.5° implying a period near 14. Positional embeddings don't usually develop periodicity unless the data has fixed-period structure, but a translation-invariant previous-token head would naturally push them toward a Fourier-like basis, so the aliasing at period 14 is probably just a side effect rather than a sign of the data.

I'm weighing English prose against Python code, and wondering whether a layer-0 head's large angles near 180° point to a high-frequency alternating pattern. Code might show up in indentation-counting or bracket-matching heads, but that's harder to pin down. Non-English text like French, or Russian with Cyrillic, would also fit a vocabulary around 90.

Now I'm trying to pin down the vocabulary size more carefully, starting from the hint that the task mentions a "language or kind of text." I'm also considering whether the model's last-layer neurons could reveal the alphabet by clustering their output directions over the characters they boost.

I'm now considering whether the vocabulary might be checked against the residual stream's dominant directions, though those large singular values look more like a shared offset than anything tied to the language. Without the embeddings to inspect, I can't really read the frequency distribution, so I'll have to leave the language guess uncertain and lean on base rates for now.

I'm now checking the vocabulary estimate against a second route, comparing the spectrum of the layer-3 output combined with the unembedding directions. I'll also see whether the layer-0 input shows a similar effective dimension, and whether the layer-2 output overlaps with the unembedding space.

I'm going to fit the singular values of the layer-3 projection on a log scale and look for a two-line breakpoint, which should give a sharper read on where the vocabulary-sized subspace ends.



---

Checking whether the scores hold up when the sample is less tidy: they don't really, the spread sits around 86 to 114. I'm also testing whether the text could be non-alphabetic, but a knee near 90 rules out a large character set like CJK, and DNA would have far too few symbols. That leaves alphabetic text with roughly 90 to 110 distinct characters, which fits English prose or Python code. A possible way to tell them apart is how strongly code models attend to newlines.

I'm weighing whether the text could be code rather than English, since code models lean heavily on newlines and indentation. I'm leaning toward English at about 0.35 confidence. The 72° shared plane in layers 1-2 is still unexplained, so I'm planning a few more calls to characterize head 2.0's attention offset by pulling out the positional subspace.

I'm checking whether 2.0's attention reads raw positions; it seems not, since its score sits at baseline with the positional subspace. So its attention is probably content-based, possibly shaped by layer-1 heads like 1.2, which I need to examine more closely.

I'm checking whether 0.3 might be the actual previous-token head feeding the induction heads in layer 3, since its outputs connect strongly to them while 0.5 barely does. That would make 0.5 a few-back head, and its angles would need to fit a larger offset.

The 0.5 head's angles climb almost linearly (26, 51, 76, 97, 117), which looks like a pure shift, so it's a clean offset head. The 0.3 head's angles fit a roughly linear pattern too, about 8.5n − 5, and that is close to what a larger-offset head would produce, so I need to compare the two offset readings directly.

I'm weighing whether 0.3 could be the true previous-token head, since its outputs feed the layer-3 induction heads so strongly, with 0.5 attending roughly three positions back. The alternative is that 0.5 is the one-back head and 0.3 is closer to self-attention.

I'm checking whether 0.3 is the actual previous-token head, since the induction heads in layer 3 draw on its output so strongly. If it were a same-token head instead, the attention would match the current token rather than the one before, which seems less useful for induction, though models can have both. I'm also looking at the query side to see whether the matching uses raw embeddings, but that comparison doesn't separate the two cases.

I'm wondering whether 2.0's OV map overlaps with 0.3's, which would mean they write similar information that the layer-3 induction heads read together. To check, I'm computing the normalized inner product between the two OV maps and comparing their input and output subspaces.



---

I'm weighing whether the query side of head 3.1 overlaps with the token-embedding subspace that layer-0 values read, which is a loose check. I'm also considering whether a head that attends to the current position, a Δ=0 head, is of any real use, since the residual stream already carries that information.

I'm now leaning toward reading head 0.3 as a previous-token head, since its flat eigen spectrum suggests a sharp peak. A sharp Δ=0 head would be redundant, while a sharp Δ=1 head is useful, and 0.3's output feeding the 3.x keys fits induction.

I'm checking whether the 0.3 and 0.5 heads might be mixing current and previous tokens, which would show up as a phase near half the slope at low frequencies. The 0.3 phases come out smaller than that, so it would be mostly self-attention with some previous-token weight, which leaves things ambiguous. A cleaner test is the ratio of their angles at each frequency, which should equal the ratio of their offsets, and the measured ratios grow steadily with frequency.

I'm checking whether 0.3's phase intercept being negative means its score profile isn't a single delta, and I'm testing a mix of Δ=0 and Δ=1 weights. Such a mix would give a phase roughly linear through the origin at small frequencies, so a negative intercept doesn't fit it cleanly either.

Now I'm checking the K-path numbers: 0.3 feeds 3.1 keys with a strong value (21.6), while 0.5 contributes only weakly to the 3.x keys. If 0.3 were a current-token head, its output would just match the query token, which is a duplicate-token pattern rather than induction, so a previous-token reading fits better. I'll also look at what head 2.0 attends to, starting with its negative K-path weights on 0.3, 1.2 and 1.4.

Next I'm looking at head 2.0 and the other heads, checking whether their QK weights load onto the positional subspace. I'll take the top-40 row and column spaces of QK(0.5) and QK(0.3) as that subspace and compare how much energy each head's QK puts into it.



---

I'm noticing that the query side of head 2.0 is reading the output of head 0.3, and that its match against the raw key looks strong. That would mean the query is attending to keys whose token is the previous token, if 0.3 is acting as a previous-token head, or the same token if it's acting as a self-attention head. I'm also looking at heads 3.0 and 3.1, where the query-side signal seems negative across many heads, which may be suppressing the query's own output from matching the key.

I'm now weighing two readings of head 2.0: if head 0.3 is a previous-token head, the query would be looking for earlier occurrences of the token before it, while if 0.3 tracks the current token, 2.0 would be a duplicate-token head. Its key side is pushed negative by 0.3 and 1.2, so I'm checking how those suppressions fit with the 3.x heads reading 2.0's output.

I'm now reasoning through scenario A: 2.0 at position k would attend to an earlier position holding the previous token, so its output carries that token's identity. That would let the 3.x heads match the query token against 2.0's output and act as induction heads.

Scenario A still looks more coherent, since 0.3 feeding the 3.x keys positively fits a previous-token head composing into induction. That would make 0.3 the previous-token head, and 0.5 seems to attend about three positions back, though 2 is also possible.

I'm weighing whether 2.0 could be reading 0.3's previous-token output at the query and matching keys by their raw token, so it would attend to earlier positions holding the token before the query. Any info it copies from there, like that position's own previous token, would give a two-back context. I'm not sure yet what 2.0 is for, and I'm leaving that open.

Now I'm considering whether head 2.0 might be an induction-style head in its own right: it would look for earlier positions whose token matches the previous token of the query, and that's a "look-back duplicate" pattern. I'm also noticing that layer 1 heads (1.2 through 1.5) read 0.3's output at the query in a similar way, and 1.0 reads 0.2 at the query.

I'm weighing whether 0.5 might instead be the previous-token head, which would shift 0.3's role, but for now 0.3 is the main candidate for previous-token behavior. I'm also looking at 1.1 as a broad, near-uniform averager feeding the key side of 2.1 and 2.3.

I'm weighing whether head 0.5 might also be a previous-token candidate, since its attention seems to sit about two to three positions back, but I'm leaning toward keeping 0.3 as the only one listed, since a self-head heavily used by the later heads would be less plausible. I'm not inclined to add 2.0 as a previous-token head, because my hypothesis has it attending to earlier occurrences rather than the immediately preceding position.

I'm checking how head 0.3's trace compares with 0.5's, and whether a sharp one-position shift could explain the score pattern for 0.3.

I'm weighing whether head 0.5 could be the sharper previous-token head instead of 0.3, since its attention angles sit around 26 degrees per position and would discriminate Δ=1 from Δ=0 far better than 0.3's low frequencies.

I'm realizing I can't separate 0.5 from 0.3 on frequencies alone, since a Δ=3 head with a smaller base angle produces the same harmonics as a Δ=1 head. So I'll lean on usage instead: 0.3's output feeds both the query and key sides of later heads, while 0.5's barely does, which points back to 0.3.

I'm settling on 0.3 as the previous-token head, since later heads draw on its output the most, and I'm treating 0.5 as a positional head that attends a few tokens back. Next I'm turning to the induction heads, 3.0, 3.1, 3.4, and 3.5, and how their keys pick up from earlier layers.

Now I'm checking the remaining heads by looking at a few simple statistics for each: how much the query-key term depends on the key alone, which would indicate attention to particular token types, how flat the QK spectrum is, and the OV norm. I also want to see whether heads 3.2 and 3.3 pick up a negative signal from the 2.5 and 1.x outputs.

I'm checking each head's attention pattern through a few simple measures, like how much its scores depend on the key alone and how its positional structure looks, to sort the heads by role. Then I'm turning to whether heads like 0.2 point toward word-boundary attention, and I'm leaning toward English as the likely training language, though with low confidence.

I'm weighing whether the training text was English, code, or some other kind of text, and leaning toward English at roughly 45%, with code and other natural languages sharing the rest. I'll run the final stats to label the heads.



---

I'm sorting through the labels for the first few heads: 0.0 and 0.1 look like weak, content-based averagers with small output projections, while 0.2 is a key-dominated head that seems to lock onto a delimiter-like character class, and 0.3 appears to be a previous-token head.

Now I'm working through the remaining heads in layers 0 through 3: 0.4 and 0.5 are positional, 1.0 matches word boundaries using 0.2's output, and 1.1 is a broad averaging head. The layer-2 heads look like they're building context or look-back signals, and 3.0, 3.1, 3.4, and 3.5 seem to be the induction heads, with 3.2 and 3.3 unclear.

