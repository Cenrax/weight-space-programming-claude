I'm treating this as a hard inference problem with only 40 tool calls, since the embeddings are gone and I have to get at the language from the weights alone. My first thought is that the layer-0 weights might still carry something about the vocabulary size, maybe through the rank of what the first layer sees, though I'm not sure that will be enough to pin down a language.

I'm weighing whether the first layer's structure could reveal the vocabulary size, but that seems unlikely to settle the language. The more promising route is analyzing the attention circuits: previous-token heads in layer 0 should show rotational structure in their QK matrices, while induction-style heads should have dominant positive eigenvalues, and their composition across layers could be traced through weight products.

I'm reasoning that the vocabulary size might show up in the effective rank of block 0's q/k/v weights, since directions with no input get little gradient and decay toward zero. The catch is that tokens plus 256 positions could span most of the input space, so the signal may be weak unless the positional part is effectively low-rank. I'm also considering whether the vocabulary is tiny, like DNA's four characters, which would point to a very different language than English or a large character set.

I'm considering whether the MLP weights in layer 0 could show how many distinct token directions there are, though without a forward pass that's hard to read off directly. The singular values of the first fc matrix, together with the layer-0 LayerNorm gains, might reveal which residual directions go unused, since weight decay should shrink those. I'm also wondering whether the input could be something with a tiny alphabet like DNA, or something with strong structure such as code or ABC notation.

I'm going to look at the singular value spectra of the layer-0 Q, K, V, and fc weights, and compare them against the later layers. I'll also run attention-head analyses, since the bigram statistics that would normally reveal the language are stored in the missing embedding and unembedding matrices.

Checking whether the vocabulary size can be read from the effective rank of the input subspace: a small alphabet would make it low, a large one would fill the space. I'm also considering whether the positional structure is low-dimensional enough to let me reconstruct the positional part of previous-token heads, though without the embeddings that seems hard to pin down.

I'm computing composition scores between layer-0 heads and later heads by multiplying the relevant weight matrices, folding in the LayerNorm gains, and comparing the norms against a baseline.

I'm starting with basic statistics, looking at the singular values of the block 0 weight matrices.



---

I'm checking why the smallest singular values come out near zero everywhere, since layer norm removes the all-ones direction and weight decay pushes those weights toward zero. Layer 0's spectrum decays smoothly with no sharp drop at the vocabulary size, so I want to compare its ratio against the next layer more closely.

I'm comparing the singular value ratios between layer 0 and layer 1 to see whether a knee shows up at some index. I also want to print the full block 0 spectrum for q, k, v, and the MLP in coarse bins.



---

The fc layer's spectrum falls off noticeably between roughly indices 80 and 120, though the decline is smooth. A random 1536x384 init would have much smaller singular values, so the trained part is what's standing out here.

I'm working through the random-init scale estimate: with std around 0.02, the bulk singular values land near 0.3 to 0.9, so that part is likely just leftover initialization. The top hundred or so dimensions look like the real signal, but pinning down the vocabulary from the weights alone is hard.

I'm checking whether the layer-0 value weights look like untrained init, since with no gradient on some directions they'd stay near random. A 384x384 random init at std 0.02 would give singular values up to roughly 0.47 after the gamma scaling, which is consistent with the observed spectrum being mostly init plus some trained structure. I'm now considering whether shared input subspaces across fc, q, k, and v might reveal the effective input dimension, since random init parts shouldn't align across matrices.

I'm looking at the layer-0 attention weights more structurally now: the K matrix has a top singular value of 3.53, far above the rest, and layers 1 and 2 show the same kind of dominant direction (5.6 and 5.9). That looks like an attention-sink or constant-bias-style direction rather than content.

Identifying the language from the weights alone without embeddings feels hard, and I'm wondering whether positional or line-structure patterns might give a hint, though nothing obvious shows up yet.

I'm wondering whether the layer-3 MLP output projection could reveal the vocabulary size instead: its columns have to be readable by the unembedding, so their effective rank in the 384-dimensional residual space might approximate the number of tokens.

I'm wondering whether the layer-3 MLP output and attention output projections could reveal the vocabulary size, since the final layers should only write directions the unembedding can read. If their spectra show a sharp knee, say near 64, 4, or 95, that would point to the vocab dimension, so I'll check those two weight matrices.



---

I'm looking at layer 3's mlp.proj, where the values seem to flatten out around indices 80 to 90. I want the exact point of that drop, so I'll print s[60:110] one entry at a time.



---

Estimating the knee at roughly 75 to 90, which points to a vocabulary somewhere around 65 to 100 once the LayerNorm mean is removed. I'm weighing that against known corpora: Tiny Shakespeare around 65, enwik8 above 200, and full-ASCII English or Python code near 95 to 100. Next I want to check whether layer 3's MLP output directions line up with the unembedding.

I'm checking whether layer 3's MLP output shares a subspace with its attention output, and whether the embeddings might be tied. For the rank question, a random-matrix bulk doesn't fit the spectrum cleanly, since the bulk seems to carry trained content too. The spectrum of layer 3 shows about 80 extra strong directions compared with layers 0 through 2, which points to a rank near 80, though the output only needs directions that change the logits, so that's not an exact bound.

The unembedding only reads out a few dozen to a hundred or so independent directions, so layer 3's outputs past that range get almost no gradient. That makes the vocabulary estimate around 77 to 85 more plausible than Shakespeare's 65, though the knee is soft enough that it's not a firm count. Next I want to measure the principal angles between layer 3's attention and MLP output subspaces.

Next I'm comparing the top-k left singular subspaces of layer 3's MLP projection and attention output, checking how much of the attention output's energy falls inside the MLP's top-k directions against the random baseline of k/384.



---

I'm setting aside the crispness question for now and moving to head analysis, which is more concrete. For each head I plan to compute the QK and OV matrices, folding in the LayerNorm gains, and then look at composition scores between heads.

I'm working through the head analysis: for previous-token heads, the QK should be strongly non-symmetric in the positional subspace, so I'll measure the symmetric versus antisymmetric parts of each W_QK. Induction heads that compose with them should also show a non-symmetric QK.

I'm now setting up the composition scores: for each head I'll take the Q, K, and V slices with the layer-1 LayerNorm gain folded in, and the output slice of W_O. K-composition will be measured as the Frobenius norm of W_K times OV, normalized by the two matrix norms, and I'll compare it against a random-matrix baseline of the same shapes.

Now I'm writing out the composition score tables for the heads.



---

I'm noting that head 0.2's output feeds the keys of several layer-1 heads and 1.0's query, so it's probably a previous-token head, with 0.0 contributing a bit too. Head 1.1 looks like another previous-token head, since its composition with the layer-2 heads is strong, though the Frobenius-based measure is rough. I'm also starting to think about a dominant direction issue in the K weights.

I'm working out a way to detect previous-token behavior in layer 0 by checking the QK matrix for asymmetry, comparing the symmetric and antisymmetric parts and the eigenvalue mass. The positional subspace isn't known in advance, so I'm also thinking about the bias terms and whether the residual's positional directions could be estimated from heads like 0.2.

I'm now planning to compute, for every head, the asymmetry ratio of its QK matrix along with the spectrum of its eigenvalues, to see how much of it is real and positive. I'm also wondering whether the OV eigenvalues would tell me anything about copying, though the untied embedding and unembedding make that unclear.



---

I'm looking at the OV eigenvalues in layers 1-3, which come out strongly negative (around -0.8 in real/absolute terms). That seems odd for a residual-basis OV circuit, and I'm wondering whether folding in the LayerNorm gain is responsible. Strong negative eigenvalues would mean these heads write back the negative of what they read, which could be suppression, but seeing it across so many heads is unusual, so I'm checking whether the top eigenvalues are driving it.

I'm weighing whether the strongly negative eigenvalues are just being dominated by a few large constant directions, so I'll hold off on reading too much into them. Another possibility is an anti-copy effect on the residual's large-norm embedding direction, which would suggest a systematic pattern across layers 1-3. I want to look at the top few OV eigenvalues for layer 1 heads, since a induction-style copying test would need the embedding-space mapping.

I'm considering whether the token embedding subspace can be approximated, since layer-0 heads read the combined embedding and position input and it's unclear which part is which. A prev-token head in layer 0 could carry token identity forward, which an induction head in layer 1 would then match against the current token. If that composed QK-with-OV map is roughly positive-definite on the token subspace, its eigenvalues should be positive, so I'm planning to compute the eigenvalues of that composed matrix with the layer-0 LayerNorm gain folded in.

I'm going to compute the eigenvalues of each pairing between a layer-1 head and a layer-0 head, checking the real and absolute parts. An induction head should show strongly positive values when paired with a prev-token head.



---

Checking whether the eigenvalue approach on the full residual stream is just picking up the large-magnitude directions, which would make it noisy. Some effects stand out: head 1.3 with 0.2 at +0.54, and 2.1 and 2.3 with 1.1 at roughly +0.45. Head 3.1 looks strongest at about +0.96 with 2.0, but 2.0 might be writing one big direction, like a bias term, which would explain it.

I'm still weighing whether the eigenvalue view is just surfacing the big directions. For the token and position subspaces, layer 0 looks like the right place to look: a previous-token head should depend mostly on position, so its QK circuit should sit in the positional subspace, while content heads should sit in the token subspace. The large top singular value in layer 0's K weights probably reads a positional or constant direction, and I'm thinking the query biases might help isolate the previous-token pattern.

I'm checking how much of each later head's key and query weights lies in the output subspace of the earlier head 0.2, which is 64 of the 384 dimensions, so the baseline is about 0.167. I also want to strip out the giant top direction before I measure that read fraction.



---

I'm checking which earlier heads the layer 1 and layer 2 keys read from. Heads 0.0 and 0.2 look like previous-token heads, and 1.1 may be one too. I'm not sure the key reads are only about position, though. The asymmetry seems to involve 1.1's key reading from 0.

I'm now looking at which heads are likely induction heads. Layer 1 keys all pull heavily from the layer 0 heads, which seems like too many to be purely induction, so the previous-token information may be serving other roles too. I want to find a positional subspace to see whether layer 0's heads share one.

I'm checking which heads have the largest singular values in their query and key matrices, and whether the query and key of each layer-0 head read the same subspace. Given the lack of obvious positional signal, I'm leaning toward treating heads whose outputs later keys read as previous-token heads, with 0.2 the strongest candidate and 0.0 secondary. I'm still unsure about 1.1, which could be a second-order previous-token head.

I'm thinking about whether a positional subspace can be identified from the layer 0 weights, since the embeddings should vary smoothly with position. For a previous-token head, the query-key product would act like a rotation with small phase, which would give complex eigenvalues close to the real axis, but that alone wouldn't settle it. I'm also considering whether the key and query biases could reveal something about positional structure.

I'm considering a more rigorous framing: an induction head needs its query to match the current token, its keys to carry the previous-token information from a prev-token head, and its values to copy the token at that position. That suggests the composite QK should act roughly like a positive identity on the token subspace. To test that, I'd estimate the token subspace from the input side of 0.2's OV matrix, since that head reads token embeddings of the previous position.

I'm restricting attention to the token subspace by taking the top right singular vectors of V_{0.2}, then projecting the QK-OV product for each head onto it. An induction head should give a roughly positive-definite matrix there, so I'll check eigenvalue positivity or the trace-to-norm ratio across layer 1 and 2 heads, including 0.2, 0.0, and 1.1. I'll also check whether the induction candidate's OV reads from that same token subspace.



---

I'm sorting out the candidate pairs and their scores: 1.3, 1.0, 1.2, 1.5 and 2.4 against 0.2, plus 2.1 and 2.3 against 1.1. Next I'm asking whether 0.2 is really a previous-token head, which would need positional evidence, so I'm starting to look for a positional subspace.

I'm checking whether a position subspace can be isolated: the directions read by positional QK rather than by the layer-0 value matrices. Then I'll compare each layer-0 head's Q and K row spaces against the union of the value row spaces, and look at the QK spectra for the 0.x heads, including the query bias terms.



---

Checking the key bias first: it should get no gradient, since adding a constant to every score for a query leaves the softmax unchanged, so it just stays at its zero init. Next, the 0.3 head shows many eigenvalues near 0.5 with small phases, which looks like a near-identity positive QK over a wide subspace, so I'm wondering whether it's a same-token content head or a positional head that attends to nearby tokens. The 0.5 head has complex eigenvalues with phases around 0.46, 0.88, and 1.33 radians.

Now I'm reading the 0.5 head's eigenvalue phases, which step up roughly evenly from about 0.46 to 0.88 to 1.32 radians. That spacing looks like harmonics of one base frequency, which would fit a positional head that compares relative offsets. The 0.4 head's phases cluster near π, so I'm wondering whether it also attends by offset, and maybe 0.3 and 0.4 are the positional heads instead of 0.2. Head 0.2 has a small QK term and a large query bias contribution, so its attention may depend mostly on which key tokens the bias favors.

I'm checking whether head 0.2 could instead be a recent-token attender, since its small QK term means its attention is driven mostly by a key-side preference, and maybe its key-side positional component handles that. I'm going to look at the eigenvalues of heads 0.3, 0.4, and 0.5 more fully, along with their angles, to see whether any of them shows an offset-based pattern.

Here I'm working out why the phases might be multiples of one base frequency. If the head peaks at a fixed offset, each plane's phase would be its frequency times that offset, and with an offset of 1 the phases would just be the frequencies themselves, which I still need to check against the sinusoidal range.

I'm noticing the phases for head 0.4 sit near π, and I'm wondering whether that reflects a frequency close to π or some offset-based pattern. Maybe it's worth checking whether 0.3's near-real eigenvalues point to a recency or self-attending role instead.

I'm checking whether the 0.3, 0.4, and 0.5 heads share the same invariant planes, since a shared positional subspace would support the idea that they're positional. I'll pull the full eigen spectra with phases for those three heads and compare whether their QK eigenvectors span the same subspace.



---

I'm checking the phase values for the frequency components, which step up in roughly 0.44 increments like a harmonic series. That pattern suggests a positional head that attends to a relative offset, with the phases all sharing the same sign.

I'm testing whether the 0.4 phases fit a shifted version of the same pattern, using π minus each frequency to see if they line up in steps of about 0.43. The spacing is close, but the offset and sign convention are still unclear, so I can't settle which relative position they point to.

I'm checking whether the 0.4 phases match π minus the frequencies, which would put their steps at roughly half multiples of the 0.43 spacing. That would be a different pattern from the whole-number multiples in the 0.5 phases, so I'm not sure yet which fits.

I'm now working out whether the 0.4 phases and the 0.5 phases sit in the same planes, since each head's positional rotation would need consistent eigenvector bases to compare them directly. I'll compute the top eigenvectors for the 0.5 case to check that overlap.

I'm checking whether the 0.4 and 0.5 cases share the same planes, so I'll compare their principal angles using the top eigenvectors. Whether the phases line up with a clean offset depends on that overlap, and I can't confirm it yet.



---

I'm matching the planes shared by 0.3 and 0.5, which form the positional subspace, and pairing their phases up plane by plane.

I'm working out the phase ratios between 0.5 and 0.3 for the shared planes, which come out to roughly 2.5 to 5.6. If the phase is ω·Δ, that would mean 0.5 attends at a lag about three to four times that of 0.3, though the frequencies don't line up cleanly enough to settle it.

Checking the 0.4 planes next: its shared plane sits at phase 0.48, close to 0.5's 0.45 and 0.3's 0.08, which makes it hard to read as a clean match. Its dominant eigenvalues, near -0.43 ± 0.84j, look like they belong to a different subspace, maybe token-level rather than positional. I'm trying to pin down the lowest-frequency plane, and I don't yet know the amplitudes that would settle what Δ actually is.

I'm now trying to work out the positional frequency ω, which the embeddings don't tell me directly. The phase patterns might reveal it: small phases for 0.3 would suggest it attends to itself or recent tokens. The setup looks like a small nanoGPT-style character model, roughly four layers with 256-token context, which could help narrow things down.

I'm weighing whether the vocabulary size can be pinned down from the layer-3 MLP knee around 76-85, and whether the weights are tied, since that would change how I read the embedding planes.

I'm checking whether tied weights would let the token embedding and unembedding spaces be read as one, so layer 3's output would have to sit inside the token span. Then I'd compare the layer-0 value directions with the layer-3 projection to see how much of the vocabulary is actually in use. I'm weighing candidate text types by how many distinct tokens they'd plausibly need, from small alphabets like DNA up to large character sets like Chinese, to narrow the vocab size.

I'm working out the output scale of the final layer's weights, which are initialized small and decay only slightly over training, so components in the unused directions still keep most of their starting size. The random initialization gives a spread of singular values, and after decay they land roughly between 0.08 and 0.25.

I'm now checking whether the layer-3 spectrum drop-off around index 80 means the vocabulary is roughly 80 to 90 characters, since with the final layernorm's mean subtraction a vocabulary of V characters would only occupy V-1 dimensions.

I'm checking whether the vocabulary is larger than the shakespeare-style character sets, since the layer-3 knee sits around 80 and the unembedding can only use roughly V-1 directions. That points toward something like printable ASCII at about 95, but the knee is soft, so I want to compare layer 3's singular values against layer 2's index by index to see where the drop really happens.



---

The layer 3 MLP transitions around indices 76 to 90 and then flattens near 0.6 from 95 onward, which suggests about 90 useful output dimensions. That points toward a vocabulary of roughly 90 to 100 symbols, like full printable ASCII, which fits code or general English web text. Languages with many accented letters would also land around 100 or more.

I'm checking whether the input side shows the same ~90-dimension subspace as layer 3's output: taking the top 85 left singular vectors of proj3 and measuring how much of layer-0's V and fc0 weight energy falls inside them, compared against the chance level of 85/384 ≈ 0.22.



---

I'm treating the 0.3 and 0.5 shared planes as the positional subspace, with 0.5 showing strong harmonic phases that mark it as a positional head. I'm planning to take the top 12 dimensions from its QK eigenvectors as that subspace, then measure, for each layer-0 head, what fraction of its QK lies in it.

I'm weighing whether the offsets can be pinned down from the other heads' phases instead. If 0.3 is roughly self-attending and 0.5 attends one position back, the frequencies would come straight from 0.5's phases, and I'd check that against 0.3's phases using a smaller effective offset.

I'm stuck on the score function: setting ω_k equal to the phases makes the peak at Δ=1 come out trivially, so it proves nothing without an independent handle on ω. I'll accept that 0.5 is most likely a relative-offset, previous-token head. But layer-1 keys don't read 0.5's output; they read 0.2 and 0.0, while 0.5's output feeds the queries of 1.2 and 1.4, and 0.3's feeds 1.3, 1.4, 1.5 and 2.0. That makes me wonder whether 0.2 does the previous-token job through a different mechanism, with a small QK and a large query bias doing the work.

I'm checking whether 0.2's key-side bias could produce recency without any query-side position signal, which seems impossible by itself. The only way a score could track distance is if its eigen-directions carry a monotone positional component, so I'm considering that option.

I'm now checking whether the key-side bias of head 0.2 points along the positional planes or elsewhere, and I'll do the same for 0.0. To test this, I'll build a positional subspace from the top eigenplanes of 0.5 and 0.3, then measure how much of each layer-0 head's q, k, v, and bias energy falls inside it.



---

I'm checking which heads read the positional subspace: 0.3 and 0.5 have their queries and keys concentrated there, while their value vectors avoid it and carry token content instead.

I'm now weighing whether 0.3 might instead be the current-token head, with phases near zero attending to itself, and 0.5 the previous-token head. That raises the question of what 0.0 and 0.2 do, since layer-1 keys read from them, and whether the induction score via 0.5 still holds up.

I'm now checking whether the induction signal can be tested using only the token-carrying value subspaces of heads 0.3 and 0.5, rather than the full residual stream. The next step is to trace which later heads read from 0.3 through their keys and queries, and whether the positivity scores hold up for the layer-3 heads.

I'm going to test induction using only the token-content subspace: the row space of the layer-0 value matrices with the positional directions removed. For each earlier head as a candidate previous-token head and each later head, I'll compute the token-to-token interaction matrix and check whether it's positive, which would point to induction. I'll also check a same-token matching version, for all layer-0 heads and, in layer 2, the layer-1 heads too.



---

I'm seeing weak signals overall. Layer-3 heads show same-token positive activations, and heads 3.1 and 3.4 look strongest on the 0.3 path. Now I'm checking whether 0.3 could be a previous-token head, which would make layer-3 heads 3.1, 3.4, 3.0 and 3.5 the ones to examine.

I'm checking whether 0.3 is a previous-token head, and the phase values for 0.3 are much smaller than those for 0.5, which would fit 0.3 mixing self and previous-token attention.

I'm weighing whether 0.3 could be a self-attending head, since a near-zero phase would put its peak on the current token, maybe smeared a little toward the previous one. That would make 0.3 a local current-plus-previous head, while 0.5's larger phases would point to a peak at a specific offset.

I'm checking whether the 0.5 phase differences line up with 0.3's under a clean offset, but the ratios don't come out integer-clean, so the heads may be attending over a spread of positions. Without position embeddings I can't pin down the offsets, so I'll test it functionally instead, by seeing which layer-0 head's output the key-side of later heads actually reads.

I'm looking at the read-fraction table to see which layer-0 heads the keys of later layers depend on, and layer-0 heads 0.0 and 0.2 stand out. Their query-key weights are mostly small, but the positional term is strong, so their attention seems driven by a fixed preference over key positions rather than by content, possibly favoring the start of the context.

I'm trying to see if the key-side positional bias in heads 0.0 and 0.2 could act as a broad recent-window average, giving something like a bag-of-characters view of the current word or line. Fixing the attention to absolute positions would be mostly a sink, so I'm now checking whether the positional structure shows up in the bias term for head 0.5 as well.

I'm considering whether the 0.0 and 0.2 heads might work as a different kind of previous-token head, so I'll look at their attention in the positional subspace, with the eigenphases and how large the bias term is compared to the content term.



---

I'm working through layer 0 and sorting its positional heads: 0.3 looks local with small phases, 0.5 is offset with larger phases, and 0.4 seems positional across phases 0, 0.58, and 0.89 with some content mixed in. Its strongest negative complex eigenplanes don't show up in my positional or token subspaces (TT is only 0.39), so I'm wondering whether they sit in a positional subspace my P basis, built from 0.3, didn't capture.

I'm checking the 0.5 planes against a harmonic pattern (roughly k × 0.445), and comparing that with 0.3's phases, which run about 0.145 apart. The ratio between them comes out near three, so I'm testing whether both heads follow a shared linear frequency structure.

I'm checking whether the 0.3 and 0.5 phases fit a shared harmonic structure: the ratio of their frequency spacings comes out near three, which would allow integer solutions like Δ3=1 and Δ5=3. I'm now wondering whether a frequency range topping out near 0.9 rad could still produce a sharp single-position peak.

I'm checking whether a sharp previous-token peak could come from these frequencies: at ω0 ≈ 0.445 the Δ5=1 reading looks sharp, while the low-frequency reading for 0.3 would be broad. So 0.5 might be the previous-token head, with 0.3 acting more like a broad recency or self head. I'm also wondering whether the periodicity is real or just aliasing at a period near 14 positions.

I'm checking whether the 0.3 and 0.5 phases follow the same frequencies, since the ratio of their phase relations stays near three. If that holds, the implied period comes out either about 42 or about 14 positions, and neither looks like a typical learned embedding, so I'm doubting the harmonic reading.

I'm weighing whether layer 0's 0.5 head or its 0.3 head is the previous-token one, since the shared harmonic reading only pins down their relative offsets. I'm also noting that in the lowest plane 0.4 lines up with 0.5 and 0.3 at similar phases, which I'll keep in mind as I check downstream reads.

I'm checking how layer 1's query side reads the layer 0 positional heads, with 0.5 feeding one query and 0.3 feeding others. I'm weighing whether an induction-style match between query and key could be measured from those OV paths, but the number of combinations makes that hard to pin down, so I'm stepping back to see what the overall analysis can support with reasonable confidence.

I'm weighing whether a head's key-side preference, read off its bias vector projected onto the positional planes, is aligned with the lowest-frequency plane, though that alone wouldn't settle the position. I'm also looking at 0.4's eigenvalues in P, where the real mode at phase 0 dominates and the mode near 0.58 and 0.89 are weaker, so the picture stays mixed. Another angle would be to anchor on a self-attending Δ=0 pattern and see which heads' phases match it, but I'm not sure that's resolvable yet.

I'm testing whether the 0.3 phases fit a Δ=1 reading, and it looks inconsistent: a plane-one frequency of about 0.08 would make the 0.5 implied offsets drift across planes (5.6, 3.8, 3.6, 3.25). I'm now checking whether 0.5 at Δ=1 gives a more uniform set of frequencies across the planes.

I'm weighing the two hypotheses against each other: if 0.5 is the previous-token head, 0.3's phases fit a self-plus-previous mixture and the picture stays consistent across planes. If instead 0.3 is the previous-token head, 0.5's implied offsets drift widely, pointing to a broader recent-context window.

I'm now asking which layer 1 query heads actually read the output of 0.5, which would be the first test of whether it works as a previous-token head. The strongest readers look like 1.2, 1.4, and 1.3/1.5.

I'm weighing whether a query-side prev-token reading could still produce induction behavior, but the variant I'm testing needs the token after A rather than before, so it doesn't work without another step.

I'm checking whether the 0.3 heads could still act as a mixture of current and previous tokens, which would make their key outputs look like same-token matching. If so, the 3.x heads might be attending to earlier occurrences of the same token, and the 2.0 path shows a similar pattern.

I'm checking whether 0.2 and 0.0 could be the previous-token heads instead, with their positional QK sitting outside the P subspace I built from 0.3 and 0.5. Their positional part looks small, though, so I'm doubting that a sharp previous-token pattern would come from it.

I'm estimating the attention sharpness for 0.2 and 0.5 from their query-key scale: 0.5 comes out sharp, around 10 in score, while 0.2's positional part is much softer, though its bias term could still push it moderately.

I'm considering which kind of text this is, weighing the roughly 85 to 95 distinct characters against typical candidates like English prose, Python source, or other scripts. The positional periods don't settle it, so I'm falling back on the vocabulary size as the main clue.

Now I'm weighing whether the knee in the singular values really tracks the vocabulary size, since a count near 80 to 90 rules out tiny Shakespeare-style character sets, and I'm wondering whether the layer 3 output dimensions can pin that down more precisely.

I'm trying to pin down how many singular values in layer 3's projection are real signal versus init-like noise, since the vocabulary-size estimate depends on where that boundary falls. Layer 0's projection is all useful, with its minimum around 0.56, but layer 3's minimum drops to about 0.195, which suggests a set of weakly used directions. I'm checking whether the values sort into strongly used, weakly used, and null groups, and where the null group starts.

I'm weighing whether the knee near 80 to 90 singular values could match a known text, since candidates like War and Peace with about 83 characters or English essays around 80 would fit that range. The tail values fall off smoothly, so I'm not sure where the boundary truly is.

I'm weighing whether the layer 3 projection's singular values can pin down the vocabulary size, but the boundary between useful and null directions isn't clear. The tail falls off smoothly, so a precise count is hard to get from this alone. I'll settle on a rough estimate of about 85 to 95 for now.

I'm now looking at the singular spectrum of layer 0's input projection after removing the positional subspace, to see whether a knee appears that would indicate how many embedding dimensions are actually read.

I'm checking whether the vocabulary estimate of roughly 85 to 100 characters points toward full printable ASCII, which would fit either English prose with punctuation or Python source. Code would show strong bracket and indentation structure, while prose would lean on word-level induction patterns, so neither stands out yet. I'm also wondering whether the harmonic phases of head 0.5, with its period near 14, say anything about line length in whatever text this is.

Trying to count the used output dimensions more precisely now, by comparing the layer 3 projection's singular spectrum against layer 2's. The ratio drops off around index 66, levels out between 0.6 and 0.45 from about 90 onward, so I'm estimating a vocabulary near 90.

I'm weighing whether the knee at about 66 to 90 means something, since if prose is the source, the many rare ASCII symbols would contribute weak directions and the count would land nearer 77, while code would use nearly all 95 printable characters.

I'm weighing the evidence on whether this is English or code, and the smooth slope in the singular values doesn't settle it. I'm leaning toward English with moderate confidence, though I'd want something from the heads to back that up.

I'm looking at whether the layer 0 input subspace is shared across the Q, K, V and fc matrices, and whether that could give a joint vocabulary estimate of about 80 to 95.

Weighing English against other alphabetic languages, I'm leaning toward English at about 0.35 confidence, since the vocabulary size alone can't separate them. Russian or Greek would also fit the knee near 80 to 90, so I'd want evidence from the attention heads before settling on anything.

I'm now setting up a test for an induction circuit, where the key side reads the previous-token head's output and the query side reads the raw token. Earlier, the layer-3 heads read through head 0.3 came out positive, while head 0.5 gave nothing positive, so 0.3 looks like the stronger candidate. I still want to check whether the query side picks up mixed current-token information in the later layers.

I'm checking whether layer 1's heads, especially 1.1, behave like positional prev-token heads by measuring how much of their query and key energy lies in the positional subspace and looking at the phases of their QK eigenplanes.

Now I'm checking how the layer 1 heads' query and key energy splits between the positional subspace and the rest, to see whether they behave like positional prev-token heads.



---

Only layer-0 heads read position directly, through 0.3, 0.5, 0.4 partly, and weakly 0.2 and 0.0. Later heads can't pick up positional signal from P, so any previous-token information they have must come through the layer-0 outputs of 0.5 and 0.3, and I'm not sure those carry it.

I'm checking whether the layer-1 heads 0.2 and 0.0 can carry positional information through their key bias, since the score might be shaped by recency or by a fixed positional direction. The high-frequency positional term in the key bias looks odd, so I'm wondering whether P contains something more.

I'm settling on a decisive test: an induction-style check, where a layer-1 head's keys read the previous-token signal from a layer-0 head and its values copy the raw token. Layer-1 keys read 0.2 and 0.0, and the positive scores for 1.3 and 1.0 versus the negative one for 1.4 suggest which candidate fits.

I'm estimating the frequency ω from the claim that the layer-0 positional planes all share the same basis, since that would tell me which heads actually track relative offset.

I'm estimating the frequency for the layer-0 planes from the phase ratios between 0.5 and 0.3, and testing whether 0.3 could be a previous-token head at Δ=1.

I'm weighing two readings of the layer-0 heads: 0.5 as the sharp previous-token head at offset 1, and 0.3 as something broader that leans toward the current position and the recent past.

I'm testing whether a recency-weighted window explains the phase pattern in 0.3: if it mixes the current position with the one before it, the phase should rise with frequency in a sublinear way, which is roughly what the numbers show. The ratio creeping upward as frequency increases is the detail I'm trying to account for.

I'm checking how well the phase pattern fits a mixed self-plus-previous-token window, and the fit is only rough. The observed phases rise faster with frequency than the simple mixture predicts, though the trend is qualitatively right. One caveat I need to keep in mind: a head's QK phase doesn't map directly onto which offset it attends to.

I'm reading the phase pattern against two hypotheses: if the QK phase is 0.3·ω, then head 0.3 scores Δ=0 and Δ=1 about equally, so it's a local self-plus-previous head. That would make 0.5 the previous-token head, and under the alternative scaling 0.3 would peak at Δ=1 and 0.5 near Δ≈3.

I'm weighing whether the positional frequencies look more like the 0.5-as-previous-token reading or the 0.3 one, since the scale ambiguity can't be settled from phases alone. Learned positional embeddings usually carry smooth low-frequency components alongside some high-frequency local precision, so the frequency range should hint which reading is more plausible. Checking how layer-1 queries use the 0.3 and 0.5 outputs may break the tie.

I'm checking whether 0.4 gives an independent read on the offset: its low-frequency plane matches 0.5's, so it probably tracks the same thing and doesn't add much. The sign question is the open part now, since causal masking means only past offsets matter, and the phase convention might be flipped.

I'm leaning toward labeling 0.5 as the previous-token head, since it has the sharpest positional eigenvalues and that matches what layer-0 heads usually do in character models. Head 0.3 looks more like a local mix of the current and previous positions, though under a different scaling it could also be the previous-token head, so I'm not settling that yet.

Now I'm checking whether 0.5 actually serves as an induction-style source, since nothing in layer 1 reads its output for keys. Its only downstream use looks like the query side, which fits a bigram-context role rather than induction. The layer-3 heads 3.1 and 3.4 that depend on 0.3 are the stronger evidence, so I'm leaning toward 0.3 as the prev-token candidate if the scale reading flips.

Checking whether the layer-3 heads that read 0.3 would still fit if 0.3 is the mixture head rather than a clean previous-token head.

I'm checking whether the layer-2 head 2.0 fits a path where its keys read layer-1 outputs and its queries read the layer-0 position signal, and I'm going to test the induction chain by scoring query and key paths through earlier heads.

I'm now checking which layer-1 and layer-2 attention paths carry the most weight, ranking the query-key pairs by the size of their score contributions and whether those contributions are positive, to find the dominant circuits.



---

I'm checking which heads in layers 1 and 2 have strong positive scores for query 0.3 against raw key tokens, with the largest on head 2.0. If head 0.3 is a previous-token head, the query would be the token one position back and the keys would match positions carrying that same token.

I'm now weighing a few alternatives: if head 0.3 were a previous-token head, the induction story doesn't quite fit, since copying the token at j would just reproduce the previous character. The layer-3 path looks more promising, with layer-2 head 2.0 carrying token identity from an earlier position that the layer-3 heads match against.

I'm weighing whether head 0.3 is a previous-token head or a self/local mixture, since the query side would then carry either the prior token or the current one. If it's the former, head 2.0 would be attending to earlier positions that match the previous token, and if it's the latter, it would be matching the same token. Either way, the layer-3 heads would be reading token identity that 2.0 has already moved forward, which makes the overall story harder to pin down.

I'm now considering whether the induction pattern could be inverted, with the query carrying the previous token and the key carrying the current one, but that would find earlier occurrences of the prior character and copying from them wouldn't yield induction. I'm also wondering whether head 0.3 could be a local mixture, and I'm thinking about testing what the value vectors of the layer-3 heads actually read and write.

I'm checking whether layer-3 heads 3.1 and 3.4 read from head 2.0 for their keys, and whether 2.0 itself could be a content-based look-back head that attends to earlier occurrences of the previous token.

I'm now going to check the value side of these heads: whether head 2.0's values carry raw token identity or the output of head 0.3, and do the same for the layer-3 heads, since an induction head in the last layer should copy the raw token at the matched position.

Next I'm checking the value vectors of these heads to see whether they carry raw token identity, since that decides which layer-3 heads are doing the copying.



---

Ruling out the raw-value measure for now, since the token-space scaling makes it dominate the norm. I'm switching to a cleaner check on the H1/H2 question using neuron 0.4, whose lowest-plane phases fall around 0.5 to 0.58. I still need to find out whether its higher non-P eigenplanes, at phases 2.05 to 2.93, sit in a positional subspace.

I'm checking whether neuron 0.4's higher non-P eigenplanes, the ones at phases 2.05 to 2.93, might be positional planes I left out of P, and I'm comparing how much of their weight sits in token space versus P. I also note its large key-side preference term. Without a direct handle on the positional embeddings, I'll have to treat the final lists with some uncertainty.

I'm checking whether the bias key-side preference for 0.2 lies in the 0.3 and 0.5 planes, which would mean a periodic absolute-position preference. That seems odd unless those planes carry low-frequency components. Without a direct handle on the positional embeddings, I'll try estimating the frequency from the QK structure on each plane instead.

Still working out how the phase relation on plane 0.5 might pin down the frequencies, since the positional embeddings aren't directly accessible.

Checking whether the phases on plane 0.3 fit a harmonic pattern with plane 0.5, I'm testing the H1 hypothesis, where 0.3 sits at offset 1 and the frequencies are multiples of about 0.145. The ratios don't come out clean: the 0.5 phases run systematically higher than three times the 0.3 phases at the low planes. Under H2, with 0.5 at offset 1, the frequencies would be about 0.445 times k, which gives a period near 14.

Now I'm thinking that a harmonic series of frequencies would come from a stationary position code, where the dot product between two positions depends only on how far apart they are, with the frequencies being multiples of a base set by the 256-position window.

Comparing the two hypotheses on the frequency spectrum: if the head attends one step back, the frequencies work out to a Fourier mode set with spacing 18 in units of the base period, while a three-step offset gives spacing 6. Both fit the phase data, so the phases alone can't settle it. Eigenvalue magnitudes don't break the tie either, since 0.5 and 0.3 are comparable.

I'm weighing whether the frequencies on plane 0.3 are high enough to separate immediate neighbors, given the score gap between offsets one and zero, and whether a large positional amplitude could make that sharp anyway. Next I want to check 0.4's main eigenplanes near phase π, which I left out of P.

I'm checking whether 0.4's higher eigenplanes are positional or overlap with the token subspace, which would make it content-based instead. I want to pin down the positional subspace directly, without relying on any single head, and keep labeling as I go.



---

I'm checking whether 0.4's main planes fall outside both P and T, which would point to a separate subspace, probably positional planes at different frequencies. Its phases cluster around 2–3 radians, which looks like a second positional family that 0.3 and 0.5 don't use. I'm going to build that subspace from 0.4's planes and test whether 0.3 or 0.5 do anything there. If 0.4 turns out to be purely positional on those planes, with phases near π, then

I'm noting that the 0.4 planes at frequencies around 2 to 3 rad/pos sit near Nyquist, which could mean sharp previous-token discrimination, or possibly an avoid-self pattern. I'm weighing whether the phases imply Δ=1 or Δ=2, and then checking whether 0.5 and 0.3 share those phases on 0.4's plane subspace.



---

I'm checking whether head 0.4 sits on the P2 positional plane, since its q and k enrichment there is high while V and the MLP ignore it. I'm also looking at its phases in P2 and P, which overlap with head 0.5's shared plane, to see if it could act as a previous-token head at offset one.

I'm now checking the phase values in the P2 planes against the shared plane, which I'd expect to fit head 0.4 being a previous-token head at offset one. Under the second hypothesis, head 0.4 and head 0.5 would both attend one step back, using different frequency bands, which would account for a sharp previous-token pattern. Head 0.3's phases in P2 look similar to 0.4's, so it may also attend at offset one in the high-frequency planes while mixing in the shared plane.

I'm settling on 0.4 and 0.5 as sharp previous-token heads, 0.3 as a broader local positional head mixing current and previous positions, and 0.2 as a recency-weighted average over nearby context. I'm treating 0.0 and 0.1 as content-based heads, with 0.1 avoiding same-token matches and 0.0 weakly favoring them. I still need to work out who reads the outputs of 0.4 and 0.5 as keys.

I'm checking whether 0.4 and 0.5 can serve as induction heads, since they would need to pass a previous-token signal into a later key. Their reads on the earlier layer-1 and layer-2 outputs look weak, and 0.5 shows nothing, so that path seems unlikely. The earlier table also suggests that using 0.4 or 0.5 as the previous-token input gives almost no induction effect. The stronger candidates appear to be 0.3, through its layer-3 heads, and 0.2, through 1.3 and 1.0.

I'm wondering whether 0.2 might be the previous-token head after all, since its K-composition from layer-1 heads is strong. Its query-side reads on P2 are high and its key-side reads are low, so its attention may be mostly a function of absolute position, leaning toward positions similar to its own.

I'm considering that the 0.2 output may carry a smooth, position-dependent signal, which later layers' keys could read as recency. That would explain the strong K-composition from layer 1, even if 0.2 isn't a true previous-token head. That leaves induction still unexplained, so I'll look at whether the signal flows through 0.2 or 0.0 instead.

I'm weighing whether the induction signal could instead flow through a same-token head that attends to the matching token and then reads the previous-token information from 0.4 or 0.5 at that position. The catch is that a same-token head would hand back the token at the match, not the one before it, so the copy would land one step off unless the value side compensates.

I'm now considering that 2.0 may act as a content-addressed previous-token head, reading token identity from 0.3's output at position i and attending to the matching token at j, with recency pulled in by the 0.2 keys. That would give layer 3 a previous-token representation to read via K-composition.

I'm checking whether layer-1 heads 1.2 through 1.5 might carry the same "find my previous token" pattern as 2.0, since their queries read 0.3 and their keys read the raw tokens. Head 1.1 looks like the prev-token carrier for the layer-2 and layer-3 induction-style reads, so I want to see what it contributes.

I'm looking at head 1.1, whose keys read heavily from 0.2 and 0.0 and whose output may be a broad recent-context average. If 0.3 is the one carrying previous-token information, that would fit 2.0's query reading 0.3 and point to 0.3 attending at least partly to the previous token. I'm still weighing whether 0.3 is a previous-token head under the alternative hypothesis.

I'm checking whether 0.3's output, rather than 0.4 or 0.5, is the one that downstream keys read as the previous token, since the induction table shows nothing comparable for 0.4 or 0.5. If the sharp previous-token heads were 0.4 and 0.5, I'd expect their outputs to show up in those key reads too, and they don't.

I'm checking the phase arithmetic for 0.4 in the P2 planes: its phases look similar to 0.3's, which would suggest a one-step offset there too, but that would clash with its phase in the first plane. Either 0.4 is broad, or the H1 reading doesn't hold for it.

I'm weighing whether 0.3 is the sharp fixed-offset head, since its eigenvalues stay roughly flat across many planes, which fits a delta-like kernel. By contrast, 0.5's decaying spectrum points to a smoother, broader pattern centered a few positions back.

I'm reading the P2 phases of head 0.3 as alternating, with values at π that point to the Nyquist frequency. A phase of π there means the score peaks at odd offsets, so offset one comes out on top, which supports 0.3 as the previous-token head.

I'm checking whether 0.4's high-frequency phases in P2 imply a one-step offset like 0.3's, which would clash with the lower-frequency plane where it looks closer to six steps back, so 0.4 may be a broader mix of offsets.

I'm settling on labels for the layer-0 heads: 0.0 and 0.1 look content-based, with 0.2 acting as a broad recency average and 0.3 as the sharp previous-token head.

I'm placing 0.4 as a positional, short-range head, with 0.5 covering a broader window a few tokens back. Head 0.3 remains the main previous-token candidate, and 0.4 may share some of that local role, though its phases in the first plane point to a larger offset.

I'm checking whether 2.0 is a previous-token head in its own right: it would find token_{i-1} through the 0.3 output on the query side, and if the 0.2 keys bias it toward recent positions, its attention would land mostly on i-1.

I'm checking how much of 2.0's key preference is just a fixed bias, by comparing its bias-driven key term against the alignment with 0.2's output space. I'll run the same numbers for the other layer-1 to layer-3 heads, and then look at whether the induction-style heads in layer 3 pick up 2.0 and 0.3 the same way.

I'm checking whether the key reads from 2.0 and the raw tokens actually produce the induction-style matching I'm looking for. For layer 3, I'm comparing the query-raw, key-raw pairing against the key-2.0 pairing, since the second one looks stronger on the induction table.

I'm checking the bias-driven key preferences for layers 1 through 3, to see how much of 2.0's recency pull comes from a fixed bias rather than from the 0.2 reads.



---

The key-bias terms for these heads, at 0.2, 0.0, and 1.1, dominate their outputs, so they may be functioning as positional or recency signals, or as attention sinks that latch onto the first token. I'm also wondering whether the 0.2 and 0.0 heads use their key bias to attend to early absolute positions.

I'm now turning to induction heads, starting with candidates in layers 2 and 3 that match tokens through their key-side composition with earlier heads. I'm also questioning whether layer-1 heads like 1.2 through 1.5, which look for prior occurrences of the previous token, could be doing the induction job instead.

I'm checking whether layer-0 head 0.3 is really a previous-token head, since that would make the layer-1 heads that match on its output look for earlier occurrences of the preceding character. Under that reading, 2.0 would copy that token forward, which fits an induction circuit in layer 3. I'm also wondering why four layer-1 heads would all do the same job.

I'm checking whether layer-1 heads 1.2 through 1.5 might be doing an induction-like lookup, since a head reading the previous token and matching earlier occurrences could copy forward what came after. I'm also wondering if OV of head 0.3 could be read as encoding the current token rather than the previous one.

I'm checking whether 0.3 is really a previous-token head, and I'm weighing whether its OV output might instead encode the current token. I'm also planning to look at the OV composition for copying later, though I'm not sure that will settle the question without the unembedding.

I'm comparing 0.4 against 0.3 on the high-frequency planes in P2, checking whether their phases line up closely enough to suggest a similar offset in that band. I'll also look at the low-band weights of 0.4, which lean positive at zero offset.

I'm projecting the other heads' QK blocks onto the planes that 0.3 defines, using 2x2 block projections to compare them, with the plane frequencies taken from 0.3's phases.

I'm now projecting the attention scores for heads 0.0 through 0.5 onto the planes defined by head 0.3, extracting each head's rotation angle and magnitude per plane, and evaluating how the score varies with relative offset under both candidate frequency assignments.



---

I'm noting that the sign of the rotation depends on how the plane is oriented, since the QR decomposition can flip orientation, so the -0.52 and +0.52 eigenphases for 0.3 aren't directly comparable across bases. Within a single plane basis the comparison holds, so I'm lining up the angles for 0.3, 0.5, and 0.4 plane by plane.

I'm checking the eigenphase pairs for 0.3 and 0.4 across the P2 planes, and it looks like 0.4's phase is roughly the negative of 0.3's. Near ±π the offsets look near-conjugate, so I'm testing whether a common frequency explains the difference.

I'm comparing the 0.5 ratios under the same orientation assumption, since the 0.4 offsets in the P2 planes only line up once the sign flip is accounted for.

I'm running the 0.5 ratios against the 0.3 and 0.4 values plane by plane, and they swing from about 1.4 to 5, so the simple relative model doesn't hold up. I'm now testing a second hypothesis where 0.5 has a fixed offset of 1, to see whether the 0.3 and 0.4 offsets come out more consistent.

I'm testing a second hypothesis, where 0.5 has a fixed offset of 1, and I'm checking whether the 0.4 and 0.5 offsets come out near 1 in the planes they share. The 0.3 offsets land lower, around 0.2 to 0.36, and the P2 phases don't settle into a clean pattern under this model.

Under the second hypothesis, the 0.4 and 0.5 offsets come out near 1 in the three planes they share, which looks like a consistency check worth keeping. The 0.3 offsets sit around 0.3 in most planes, though one is much larger.

I'm checking whether a single frequency could explain the 0.4 and 0.3 phases in the P2 planes once the orientation is flipped. If I take the 0.4 values as the frequency, the offset for 0.3 comes out near 4, which doesn't match the earlier range.

I'm working through the orientation-flipped phase offsets for 0.3 in the P2 planes, and the expected phase of about 0.69 doesn't match the observed -2.12. I'm testing whether 0.3 acts as an anti-phase term, scoring cos(2.29Δ + 2.12) across a few integer Δ values, but the results still don't settle the ambiguity.

Checking whether the layer-3 induction behavior depends on the 0.4 and 0.5 heads, I'm pulling their attention to the raw previous-character position against the earlier table. The 0.4 head looks inert there, and 0.5 only shows a weak positive effect, so the induction signal seems to come from 0.3 and 2.0 instead.

I'm testing whether 0.5 and 0.4 could also be sharp previous-token heads, since an induction circuit would normally lean on them, and whether their output gets read through query composition in later layers instead. The OV norms (0.5 at 1.39, 0.4 at 0.93, 0.3 at 1.88, 0.2 at 1.61) don't settle it, and the fact that the later reads go through 0.3 still favors the 0.3 story. I'm also planning to check q:0.5 and q:0.4 against raw keys in the layer-1 and layer-2 heads.

I'm weighing whether 0.5 or 0.4 is the sharper previous-token head, since the positional analysis under the relative-position model gave them a consistent offset of about 1, while the other hypothesis didn't hold together.

I'm weighing 0.3's flat spectrum against a sharp self-attention kernel, and it looks more like a lean toward the previous position, roughly a 60/40 split between current and previous. Then I'm turning to 0.5, whose spectrum decays across the planes.

I'm settling on the view that 0.4 and 0.5 act as sharp previous-token heads, with 0.3 more of a local head that mixes self and previous position, and I'm checking whether the layer-3 induction result still fits that split.

I'm checking whether the layer-1 and layer-2 heads read the 0.4 and 0.5 outputs through their queries, and so far the raw-key scores against those heads come out near zero.

I'm checking whether the layer-0 0.4 and 0.5 outputs feed into the MLPs in layers 0 and 1 more strongly than the 0.3 output does, since that would point to them supplying bigram or trigram features rather than acting as token-identity lookups for later attention.



---

Head 0.3 is the one everything leans on most: MLP0 pulls heavily from it, and so do later query positions and layer-3 keys. Head 0.5 is used moderately, and 0.4 hardly at all, which fits its small OV norm. I'm now wondering what layer-0 positional information matters most in a character-level model.

I'm weighing whether head 0.3 is most plausibly attending to the previous character, since that would give the most new information beyond the residual stream. A self-attending head would mostly duplicate what's already there, so its heavy use favors the previous-token reading. For head 0.5, I'm considering a broad window over roughly 2–5 positions, which would fit the decreasing per-plane phase trend.

Both readings still fit the phase pattern, so the choice has to come from function. The strongest clue is that 2.0 reads 0.3's output far more strongly than the raw token (+0.68 against +0.17), which suggests 0.3 carries a different token rather than a copy of the current one.

I'm checking whether the attention-output evidence favors head 0.3 being a previous-token head. The q:0.3 to k:raw scores are clearly positive (+0.40 to +0.49 in layers 1.2–1.5) while q:raw to k:raw is much lower, which suggests its output carries a different token rather than a copy of the current one. Still, the OV circuit could preserve identity either way, so that check alone doesn't settle the position question.

I'm still torn on whether to list 0.5 alongside 0.3 as a previous-token head, since either could be the one that's mostly looking back a step.

I'm weighing whether listing both 0.3 and 0.5 as candidates hedges the outcome, since each reading leaves one of them right and the other wrong. The harmonic frequencies might help decide which reading fits the embedding modes better.

I'm favoring the frequency-band argument here: the low-frequency planes in P would sit cleanly apart from the high-frequency P2 planes, and smooth low-frequency components tend to dominate learned positional embeddings, which points toward H1. I also want to check whether 0.3's negative eigenvalue at the Nyquist frequency fits that picture.

I'm checking the sign of 0.3's real eigenvalues on the high-frequency Nyquist-type planes: they're negative (0.18, 0.11, 0.09), which fits a head that prefers odd offsets like Δ=1 rather than a self-attending one.

The 0.4 head also shows negative real eigenvalues on the Nyquist-frequency planes, the same signature as 0.3, so it likely prefers odd offsets too, roughly Δ=1 to 2. Head 0.5 has almost nothing there, which fits its broader window.

I'm settling on head 0.3 as the main previous-token candidate, since its near-Nyquist eigenvalues are negative, which suggests it prefers odd offsets like one step back. Head 0.4 shows the same sign pattern, so it may also favor short offsets, though its output is used so little that I'm less sure, and I'll mention 0.4 and 0.5 only as other short-range positional heads.

Now I'm checking which layer-3 and layer-2 heads read the previous-token signal, starting with the strongest induction-style matches against head 2.0 and head 0.3.

Now I'm checking which earlier heads feed layer-2 head 2.0, since its query-to-key pattern looks like it's keyed to the previous character.

I'm now reading layer-2 head 2.0 as a content-based previous-token amplifier: it seems to attend to earlier positions holding the character just before the current one, so its output carries that prior token forward for the layer-3 induction keys to pick up. Its key-side bias reads the layer-0.2 and layer-1.1 signals, which may be what makes it prefer the most recent match.

I'm weighing whether head 2.0 belongs on the previous-token list too, since it may mostly attend to the immediately preceding position, though that's speculative and I'm not sure yet.

I'm leaning toward listing just head 0.3 as the previous-token candidate, since it's the one most likely to be right, and I'd put 0.4 and 0.5 at roughly even odds with 2.0 for that role. For the induction heads, the layer-3 ones (3.0, 3.1, 3.4, 3.5) look strongest, and I'll leave out the layer-2 candidates for now.

I'm going to compute the spectrum of a stacked layer-0 matrix, with the positional planes projected out, to see where the token-dimension knee falls. I'll check the same thing for layer 3's output stack.



---

I'm noting that the vocabulary estimate of roughly 80 to 95 still holds, though there's no sharp drop to pin it down. Next I'm weighing how English prose and code might differ in attention structure, such as heads tracking line starts or bracket pairs, though I can't settle that. I'm also guessing at what kind of answer a puzzle designer would pick for the "which language or kind of text" question, with candidates like English, Python source, or German.

I'm checking whether the 65-character vocabulary idea holds up, since the layer-3 projection ratios keep exceeding 1 until around index 66 and then level off near 0.6. The sharpest drops in the spectrum land around indices 75 to 83, which points to a vocabulary nearer 78 to 85, more in line with English prose plus punctuation and digits than with a tiny Shakespeare set.

I'm settling on English prose as the most likely answer, with moderate confidence, since the vocabulary size lands around 80 to 85. I'm now drafting a labeled summary of the 24 attention heads in layer 0, describing each one's role, starting with the content-based matching and positional averaging heads.

I'm now labeling the layer-1 heads: 1.0 seems to match on similar recent-context summaries, and 1.1 looks like a broad positional aggregator whose output later layers read as a recency signal. The remaining layer-1 heads appear to look up an earlier occurrence of the previous character, a bigram-style lookup.

Now I'm labeling the rest of layer 1, where 1.2 through 1.5 look up an earlier occurrence of the previous character, and 1.3 also feeds slightly into the keys from 0.2. Moving to layer 2, I'm tagging 2.0 as the content-addressed previous-character head that feeds the layer-3 induction keys, then 2.1 and 2.3 as weak induction-like matches, and 2.2, 2.4 and 2.5 as general mixed context heads. Finally I'm starting on layer 3's heads 3.0, 3.1, 3.4 and 3.5.

I'm working through the induction heads in layer 3: their keys seem to read previous-character information from layer 2's head 2.0 and from 0.3, while their queries match on the raw current character, and their outputs feed the logit-relevant subspace. Heads 3.2 and 3.3 look more like output or mixing heads without a clear induction pattern, with 3.3 matching the same token and 3.2 reading from 2.4 and 1.1. I'll also note the limits of these labels.

