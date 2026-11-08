I only have the middle blocks of the model and no embeddings, so I can't run forward passes to identify the language directly. I'll have to work from the weight structure alone. Positional embeddings are gone, which makes detecting previous-token heads in layer 0 harder, so I'm considering whether the QK circuit in W can reveal that structure instead.

I'm checking whether the QK matrix, W_Q^T W_K, shows rotation structure in a subspace that might hold positional information. I'm also thinking about induction heads, where later-layer K-composition with previous-token heads' OV outputs would be strong. For the vocab size, the block 0 weights might reveal the dimension of the input span, since directions outside it would only be shaped by weight decay.

I'm weighing whether the vocabulary could be small, like DNA's four bases, since that would leave the embedding span low-dimensional. The 256 positions might also be low-rank. I'll stop theorizing and look directly at the block 0 layer norm weights for unused dimensions, then at the singular values of the q, k, and v matrices after scaling.



---

I'm checking the weight matrices' small singular values, which are near zero, and noticing a null direction that gives rank 383. My first guess is LayerNorm centering, since the q/k/v inputs are LN outputs, but I'm not sure the weight scaling keeps them zero-mean, so I'm working through how the normalized input is scaled and shifted before the q weight acts on it.

I'm now checking whether the gradients for the q weight can span the full space, since the inputs lie in an affine subspace of dimension 383 once the centering constraint is applied. That would only leave a null direction if something else, like weight decay or a deliberate transform, is at play. I also want to test the same idea for the o weight and the residual stream by looking at the actual null vectors.



---

The spectrum's tail just tapers smoothly toward zero, so the singular values don't show anything clearly informative, and strong weight decay is a plausible explanation. I'll set that aside and look at the head roles instead. Layer-0 previous-token heads depend on positional embeddings, which I can't observe directly, so I'm trying to infer the positional subspace from the weights.

I'm weighing whether an induction head's key input comes from a previous-token head's output, so I'm checking composition scores between head pairs. Without the unembedding I can't verify copying behavior directly, so I'm wondering whether the MLP structure or the embedding subspace might reveal the vocabulary size instead.

Residual directions the embeddings never touch should only carry init noise plus whatever decay leaves, and I'm not sure decay alone would wipe them out. Adam's per-entry normalization also muddies which subspace the gradients actually span, so I can't read the vocabulary size off the weights that way.

I'm now working out that the final layer's output only matters through the unembedding, so its gradient should live in a subspace of dimension about the vocabulary size. That would mean the write directions of block 3 could in principle reveal how many tokens there are, with the layer norm adding a term along the residual direction.

I'm working out that the final layer only affects the output through the unembedding, so its gradient should sit in a subspace whose dimension is roughly the vocabulary size. Layer-norm projections add some spread along the normalized residual direction, though the gradient should still be dominated by that subspace.

I'm going to check the spectra of block 3's proj and o weights more closely, looking for a knee near 65 if the vocabulary is that size. I'll also compare block 0's input side, though the positional embeddings might make that less clean.



---

I'm checking the block 3 MLP projection, where the values seem to flatten out around indices 80 to 90, and then I'll look at the block 0 value projection, which appears to level off near index 60 to 70.

I'm weighing whether block 0's value projection flattens near index 76, since the vocabulary-size guess that follows depends on it. The rough picture is a vocabulary near 65 characters, possibly something like Shakespeare, though the text could be anything. Next I want to find the steepest relative drop in the singular values after smoothing, to pin down the cutoff more precisely.



---

I'm looking at where the block 3 projection shows a knee, around indices 76 to 78, with values dropping from 2.07 to 1.92 before leveling off. That might suggest roughly 77 dimensions, though the cutoff isn't clean. Block 0's value projection changes gradually across indices 60 to 75.

I'm wondering whether a more robust approach would be to find a subspace shared by block 0's reading and block 3's writing, though I'm not sure those would match. Another option is to look at the token embedding subspace feeding block 0's input, starting from the embedding vectors.

I'm weighing whether the vocabulary size alone could tell languages apart, and concluding it probably can't distinguish English, French, and German. The block 3 output bias can't be decoded without the unembedding, so I'm considering whether attention patterns specific to code or DNA might offer clues instead, though I can't run anything to check.

I'm checking whether the previous-token head can be spotted from the weights alone, since positional embeddings would normally be needed to tell it apart. The idea is to look at each head's query-key bilinear form for a rotational, antisymmetric part: a shift-by-one structure would show up there, while content-matching heads would look symmetric.

I'm weighing whether a rotational, antisymmetric part of each head's query-key form could flag a previous-token head, since a shift-by-one would need that structure. I'm also considering K-composition between heads in different layers, as a way to spot induction-style circuits.

I'm defining each head's composition score as the normalized Frobenius norm of the product of its query-key matrix with the other head's output-value matrix, adjusting for the LayerNorm gains. I'll compare those scores against random-matrix baselines to see which head pairs stand out.



---

Antisymmetry near 0.7 across the board looks like random-matrix behaviour, since a random matrix gives about 0.707, so it tells me little. Layer 0's weights are small, which suggests weak query-key interaction there, and I'm wondering whether the attention ends up close to uniform unless the input norms are large enough to change that.

Layer 0's attention is weak here, so I'm estimating how strongly positional signals would show up in the residual stream. Identifying previous-token heads without position embeddings is hard, so I'm going to look at K-composition scores across head pairs to find the outlier that reads from a prev-token head.



---

I'm checking the attention-head metric against a random baseline of about 0.05, which puts head 0.2 well above it, mainly through its K-composition into layer 1 heads and its Q-composition into head 1.0. Head 1.1 also shows moderate K-composition into every layer 2 head.

I'm weighing whether the Frobenius-norm composition score is too crude, since bulk energy may dominate it. I'll try a sharper measure: the share of Wk2's energy that falls in the top output directions of OV1. I also want to check whether head 0.2's large K-composition comes from it writing into a bias-like direction everything reads.

I'm working through why head 0.2 might have a small QK norm yet still look like a sharp previous-token head, and whether the positional embedding norm could explain it. I'm also considering that the bias term in the query could set a key-position preference, since the key bias drops out under softmax. Next I want to inspect the singular spectra of each head's QK and OV matrices to see whether the positional structure is low-rank.

I'm looking at whether the positional structure in these QK matrices is low-rank, since that would fit a helix-like positional embedding that a previous-token head could rotate by one step. Without knowing the positional subspace directly, I'm planning to take the top singular directions of each head's QK matrix and check whether its left and right singular vectors are related by something like a rotation.

I'm testing whether the positional structure in each head's QK matrix is low-rank, by comparing the top singular subspaces on the query and key sides. If they nearly coincide, that would point to a previous-token head reading position, and I'd check how the overlap compares across heads.

I'm comparing the top singular subspaces on the query and key sides of each head's QK matrix, checking how much they overlap via principal angles, and looking at whether the rotation between them is signed in a way that would suggest a positional step.



---

I'm checking layer 0 head 0.3, where the query-key matrix is nearly symmetric and positive semidefinite with a flat spectrum. I'm weighing whether it attends to earlier occurrences of the same character, or to positions near itself through its positional self-similarity.

I'm weighing whether head 0.4's negative alignment means it avoids self-matches within a positional subspace, since a standard previous-token head would need a positive alignment at a one-step shift. I'm also noting that head 0.5 shows high overlap with mostly positive alignment, and head 0.2 has a dominant rank-1 QK direction that makes its attention depend on a single scalar key feature.

I'm considering whether the bias term bq^T Wk gives head 0.2 a key-only preference, with |bq| = 1.37 adding a contribution I should weigh against the rank-1 value direction. A monotone positional feature with a large positive query scale would peak at the current position rather than the previous one, so recency alone doesn't explain a previous-token pattern unless key content is involved. I think I need to locate the positional subspace first, and I'm thinking about whether the LN gains in block 0 might reveal where it sits.

I'm now thinking about the positional subspace in block 0: if the inputs only span part of the 384 dimensions, the weight directions outside that span would never be exercised and would only decay under AdamW's weight decay. I'm also estimating the scale of the q/k/v weights from their initialization, since a standard deviation of 0.02 gives row norms near 0.39 and singular values up to about 0.78.

I'm looking at the block 0 q weights, where the singular values fall from about 0.4 to 0.1 and look like a near-random tail. Next I want to check the stacked q, k, v, and MLP input matrices together, since they all read the same LayerNorm input, to see how many directions carry real energy. That count should show whether the input spans fewer than all 384 dimensions.



---

Ruling out clean knees for now: the read-0 curve falls to about 0.45 near index 216 and keeps going. I'm considering whether the dominant direction in block-0 reads, the top singular value in the layer-0 k weight, could be a constant or first-position feature, but I'm setting that aside for the moment. Next I want to work out what language is involved.

I'm wondering whether the MLP neurons in block 0 could act as character detectors, but I can't check that without the token embeddings. Maybe those embeddings can be recovered from the weights, for instance by clustering the fc rows. I also suspect the embeddings are tied to the unembedding, so I could compare the block 0 read subspace with the block 3 write subspace to test that.



---

I'm checking whether the earlier conclusion of no relation still holds once the layer-norm gain is included. The read direction in residual space is g⊙(row), not the raw row, so I'm going to redo the calculation with g included. For unembedding, the logits pass through the final gain as well, which means the read direction is g_f⊙W_U, and I don't know g_f yet.

I'm guessing it probably isn't tied either way, so I'll run the check with the gain included.



---

I'm setting the relation idea aside, so the language guess will have to come from some other evidence. I'm wondering whether the puzzle designer picked a kind of text like Python code, English prose, Shakespeare, or a non-English language, and whether the weights would show a signature such as strong induction heads from repetitive code.

I'm comparing the singular value spectrum of block 3's projection against the earlier blocks, looking for a knee that might show where the structured gradient signal lives. I'm also considering that the trained part of block 3's output should sit inside the readout subspace, since anything outside it gets ignored by the final layer norm and unembedding.

I'm checking the singular values of block 3's projection against the random-init scale: the tail sits well above the expected bulk, which suggests learned structure spread across many directions rather than just a low-rank piece. I'm going to compare the spectra of projections 2 and 3 more carefully to see where the knee actually falls.

I'm also weighing the attention output projection of layer 3 and the MLP projection, since the neuron-side factors don't matter for this. I'll check how much of proj3's squared Frobenius energy sits in its top k singular directions.



---

I'm estimating the output subspace dimension of block 3's MLP from where the drops occur, around 75 to 90. The vocabulary could be roughly 80 to 100 characters, whether English text or printable ASCII code, and the learned part is probably about V minus one, with rare characters contributing little.

Tiny Shakespeare comes in at 65 characters, Python code near 95, and English prose somewhere around 80 to 100. Russian with punctuation and digits would push past 100. Next I want to run the same diff analysis on block 0's value weights, since they should mostly read token identity, with their knee around 60 to 75.



---

I'm looking at block 0's value projection, where the smoothed drop values stay elevated up to around index 66 and fall off after about 74, so the learned subspace seems to be roughly 70 dimensions. That points to something like 70 to 80 relevant characters, which fits English-style text with letters, punctuation, and digits.

I'm weighing whether the knee near index 70 could fit Shakespeare's 65-character set, and it's still ambiguous. The large dominant key direction in block 0 might point to attention on spaces or the first position, which I could check by aligning the key weights with the top singular vector. I'm not confident I can pin down the language from this alone.

I'm settling on English as the best guess with low confidence, and I'll try to see whether the head structure says anything about the text type. Next I want to look at prev-token heads: a layer-0 head that copies the previous token's identity forward, read by a layer-1 induction head, would show up as a QK circuit that roughly matches each token against its predecessor.

I'm planning to identify the token subspace from the top singular vectors of block 0's value projection, then score each layer-0 previous-token head paired with a layer-1 induction head by the trace of their combined QK-with-OV matrix restricted to that subspace, expecting a clearly positive trace for the true pair.

I'm writing out the score for an induction-style pair as a product of the query and key weights with the value-output path, to get a matrix I can later trace over the token subspace...

I'm computing the trace of each candidate head pair's combined query-key-output-value matrix to see which ones have a clearly positive score, which would point to matching-token behavior. I'll also check whether layer-1 heads attend to the same character by looking at the trace of their query-key product alone.



---

I'm estimating what a random baseline for the normalized trace would look like at size 70, which lands around ±0.12. The jumps I'm seeing, like 2.20 and 2.98, are far larger than that, so I'm checking whether the layer 1+ OV setup with the token subspace is producing them.

I'm checking which heads might be induction or previous-token heads, since the strongest links run from head 2.0 into several head 3.x. I'm also considering whether 2.0 could be attending to the current token and copying it, which would make the 3.1 link look like same-token matching rather than induction. To settle that, I want each head's OV trace on the token subspace to see how much it copies, plus its own QK trace for same-token matching.



---

I'm noting that the negative eigenvalues in layers 1–3 are a common pattern, possibly tied to layer norm or some anti-copying behavior, though that's not conclusive. Next I'm looking at heads 3.1 and 3.0 in layer 3.

The positive QK trace for 2.0→3.1 is odd given that layer 2's OV looks roughly like negative identity, which should flip the sign. Since the composed trace comes out positive at 2.98, the composition is not just passing through the identity parts. That points to 3.1's query matching what 2.0 writes to the residual stream, which would fit an induction-style mechanism if 2.0 attends to previous tokens. Heads 0.3, 3.4, 3.0 and 3.5 show smaller positive composition scores with 3.1 and each other.

I'm checking whether 2.0 attends to previous tokens, since that would explain the positive composition with 3.1. Without positional info, I'm looking at the K-composition scores into 2.0 from layers 0 and 1 to see if anything reads position. I'm also considering whether layer 0's bias term in the key could favor recent or early positions, and whether a positional subspace could be built from the directions layer 0 reads.

I'm now considering whether layer 0 heads could build a positional signal: head 0.4 has its query and key reading nearly the same subspace, with paired singular values and a negative cross-alignment, which looks like a rotation-style positional pattern rather than a token match.

I'm working out whether the positive composed trace could come from a complex eigenstructure, meaning rotation-like behavior rather than a plain token match. I want to project the operator onto the shared subspace of the top query and key directions and check the eigenvalues, since complex ones with some angle would suggest a positional signal.

I'm trying to work out whether layer 0 heads can build a positional signal, since that would explain how 2.0 might attend to previous tokens. Head 0.4's query and key read nearly the same subspace with paired singular values, which looks more like a rotation-style positional pattern than a token match. I'm also noting that layer 0 head 0.2 K-composes into several layer 1 heads more strongly than baseline, and it has a large, nearly rank-1 OV, so it may be the prev-token candidate I need to check.

I'm checking whether layer 0 head 0.2 could be the previous-token candidate. It has a nearly rank-1 QK, so its scores would depend mostly on the key side, which can't produce relative shifts at every query position.

I'm weighing whether layer 0's key-side bias could give a positional preference, since a key-only score can't pick out the most recent token by itself. I'm also wondering whether the query side carries a near-constant component that would make the head behave that way.

I'm checking the top singular value of layer 0's key weights, which is much larger than the rest, and seeing which heads share its direction. If one direction is used across heads for position, that would point to a recency signal.



---

I'm looking at layer 0, where a key-preference direction d is shared across several heads through the q-bias, and d sits mostly in the token subspace. That points toward a special token like a newline or space, or a first-position feature, and the strong cosine similarity across heads suggests a shared mechanism.

I'm checking whether the tiny value norms for heads 0.3–0.5 mean d is a no-op sink. Then I'm turning to the positional question: looking for a subspace outside the token directions T, splitting each layer-0 head's matrix into token and positional blocks, and seeing where the mass lands.



---

I'm sorting out layer 0's heads: 0.3, 0.4, and 0.5 look like positional heads, since their QK circuits read mostly outside the token subspace, while 0.0, 0.1, and 0.2 work on token content.

Now I'm working out how the QK reads of heads 0.3, 0.4, and 0.5 define a shared positional subspace, and whether a rotation in it would push attention toward the previous token.

I'm weighing how to tell apart a positional head's preference for previous versus next positions, since masking removes the future side. Heads with an antisymmetric component in the positional subspace would shift the attention peak, but the direction of time there isn't visible in the data alone. I'm considering whether the anti-alignment of the query and key sides in head 0.4's top pairs, which implies rotations past 90 degrees, could reveal a high-frequency positional pattern that settles it.

I'm checking whether heads 0.4 and 0.5 fit the same pattern: 0.4's strongly anti-aligned pairs suggest high-frequency rotations, which would mean sharp offset attention, while 0.5's weaker alignment points to lower frequencies.

I'm checking whether 0.4's sharp, high-frequency relative-position pattern makes it the prime previous-token candidate, but its K-composition scores into layer 1 sit at baseline, with a small OV norm. The Frobenius score looks dominated by noise, so I'm switching to the token-restricted trace measure to see whether it reads any real signal.

I'm checking the attention-to-token numbers for the remaining heads, then running an eigen-analysis of each head's QK matrix restricted to the shared positional subspace, to see whether the complex eigenvalue angles separate sharp previous-token behavior from the others.

I'm about to compute the eigenvalues of each head's projected QK matrix on the shared positional subspace, looking at their magnitudes and angles.



---

I'm looking at head 0.4's eigenvalues, where the rotation angles are large, between about 117° and 180°. That could mean it's tracking offset positions with high-frequency components, like attending one token back. I'm also considering whether the angles might instead encode attending "anti," and I'm about to check how A behaves restricted to one plane.

I'm checking whether the head 0.5 angles, roughly 25°, 51°, 77°, and 96°, are near-multiples of a base step, which would suggest the planes carry harmonic frequencies. If the step is 1 per position, that would also fit offset-1 attention, and I'm weighing whether head 0.3's small angles mean a small step or just low frequencies. To settle it, I want to match the same eigen-planes across heads and compare their angle ratios.

I'm going to take head 0.5's eigen-planes as the reference and measure each plane's rotation angle in heads 0.3 and 0.4 using the 2x2 block's atan2 of its antisymmetric and symmetric parts. That should show whether the same planes carry consistent frequency ratios across heads.



---

I'm lining up the head 5 plane values against head 3 and head 4 for each plane, keeping in mind that the sign is arbitrary per plane, so I'm mainly comparing magnitudes and relative signs.

I'm comparing the plane values for head 5 against head 3 and head 4, and the ratios of head 3 to head 5 come out to a consistent sign and roughly 0.12 to 0.31. Head 5's angles look about 3 to 4.5 times larger than head 3's, which I'm trying to explain as the two heads attending at different offsets while using the same frequencies, maybe with some averaging that makes the effective offset fractional.

I'm noticing the head 3 and head 4 planes in the high-frequency range have angles of opposite sign, which doesn't match a simple opposite-rotation story since the angles aren't negatives of each other. Head 3's dominant planes have small angles, so I'm wondering whether that means a fractional offset rather than attending to self.

I'm now testing whether heads 4 and 5 both act as previous-token heads: head 4 would use the high-frequency planes around 117-164° per position, and head 5 the lower band of 25-117°, which would explain the similar ratios in the shared planes.

I'm working out the offset from the harmonic pattern: the angles 25, 51, 77 look like multiples of roughly 25.5°, so the head's offset is tied to that base frequency. I can't pin down whether the offset is 1 or 2 without knowing the frequencies themselves, which the learned positional embedding doesn't give directly.

Working out whether head 3 is attending mostly to itself: a fractional offset around 0.3 would mean it leans on the current token, and if the base offset is 2 instead of 1, the same ratio points to roughly 0.6, a less clear-cut case. Head 4 seems to track head 5 closely in the shared low-frequency planes, so I'm checking what its high-frequency planes around 117 to 164 degrees would imply if its offset is also 1.

I'm checking whether heads 4 and 5 share the same offset, since their angle ratios in the shared planes are close to 1. That would make both of them previous-token heads, with head 3 as the one leaning toward the current token. I'm also wondering whether there's a reason for two previous-token heads, perhaps one tracking t-1 and the other a different lag.

I'm checking whether head 0.3 behaves as a current-token head, which its OV output into layer 3 would suggest, though that points toward matching the current token rather than induction. I'm now asking whether the induction mechanism runs through earlier prev-token heads in layers 1 or 2 instead, since the K-composition from heads 0.4 and 0.5 looked weak.

I'm now checking which later-layer heads carry positional information, by measuring how much of each head's attention energy sits in the positional subspace from layer 0, and reading off the plane angles for head 5.



---

I'm checking whether the later layers even read the layer-0 positional subspace, since its random fraction is only about 0.004. If they don't, any positional attention in layers 1–3 would have to rely on position information that layer 0 has already transformed. I also want to see whether head 0.2's rank-1 QK direction actually lies in that positional subspace.

I'm now checking whether Pp is truly positional, since the harmonic structure of its angles suggests it is. Later heads don't read Pp directly, so any positional signal would reach them through the residual stream. I'm looking at whether prev-token heads 0.4 and 0.5 feed an induction mechanism, using the per-head scores from the table, and I'm noticing the query side needs a different subspace than the one I first assumed.

I'm checking whether the induction story holds up by looking at the trace values for heads 0.4 and 0.5 against the later heads, and most of them come out small. I'm also reconsidering the query side, since the current token should live in the embedding subspace, which I'm only approximating from layer 0's value reads. The layers 1-3 OV traces are almost uniformly negative, which is odd enough that I want to look at it more closely.

I'm questioning the idea that negative OV traces mean the heads subtract what they attend to. I'm recalling that negative eigenvalues show up in many trained models, and I'm wondering if this is a mean-subtraction effect from the layer norm, where uniform-attention heads strip away the shared context average. I'm now trying a different approach to finding induction heads, looking for QK patterns that read previous-token features at the key and current-token features at the query.

I'm checking whether induction-style attention shows up in the later heads, looking at whether the previous-token heads 0.4 and 0.5 feed their outputs into the key side of layers 1 and 2. The scores from the table look small, though head 0.2 stands out a bit, and I want to see if its dominant query-key direction lies in the positional subspace.

I'm now weighing how much each earlier head's output feeds into the later heads' keys. I want to estimate, for each sender and receiver pair, how strongly the token-driven signal reaches the key side, without normalizing away the output magnitudes. The unknown scale of the layer-0 outputs relative to the embeddings is what keeps this from being straightforward.

I'm now weighing whether the layer-1 and later heads read keys from the outputs of earlier heads, rather than from the token embeddings. To test that, I want to measure how much of each later head's key input falls in each earlier head's output subspace, using the top 20 directions against a random baseline of about 0.05.



---

Checking which heads read the key vectors most strongly in each layer, with layer 3 showing the largest ratio at 4.1× for head 3.1 reading from 2.0. The previous-token heads 0.4 and 0.5 look like they aren't being read by any key at all.

Checking where the 0.4 and 0.5 outputs are read, which seems to be mainly the query side of layer 1 and some layer 2 heads. Wondering whether I've got the roles reversed, with 0.2 and 0.0 being the previous-token heads instead, and I'll check the query reads in the token subspace for layer 1.

Checking whether the layer 1 queries read the token subspace T at all, since its ratio sits at chance (about 0.18–0.20 against a baseline of 70/384). The same holds for the key reads. That suggests the later-layer heads are picking up earlier head outputs rather than the embedding directions, so I'll also look at whether the layer 0 MLP output carries what the queries are reading.

I need to work out the "effective embedding" as emb plus MLP0's output, since the layer 1 reads didn't line up with the token subspace. I'll check how much of the query and key reads fall into the MLP0 output directions, using the top left singular vectors of the projection.



---

Composition scores aren't separating the cases well, so I'm considering a more targeted look at the induction circuit. The idea is that the previous-token OV map and the current-token query path together determine the induction behavior, with MLP0 set aside for now. Layer-0 heads 0.0 and 0.2 might also be relevant here.

I'm checking whether head 0.2's QK is a positional rank-1 map, since a previous-token head would need query-dependent offsets, which a rank-1 positional QK can't give. A steep recency term in the keys might bias attention toward recent tokens, but that still favors the current token over the previous one. I'll look at the top singular directions of 0.2 and 0.0's QK against the token subspace T versus the positional planes Pp to settle it.



---

The shared direction in both T and Pp is probably d, which might be a positional feature that also shows up in v. Heads 0.0 and 0.1 look content-based, 0.2 seems to be rank-1 involving d, and I'm now turning to the previous-token heads, 0.4 and 0.

I'm checking the head-by-head reading next: 0.4 and 0.5 both look positional with a steady rotation, while 0.3 barely rotates. Under the causal mask, the offset has to be zero or positive, so I'm weighing whether 0.4 and 0.5 are attending one step back, with 0.3 mostly on itself. One step back is the more common pattern for these heads, so that's the leading guess.

Now I'm weighing whether the later layers could do prev-token attention through composition with layer 0, but matching on the query's own token seems unlikely. For induction, the question is whether the K reads of 0.4 and 0.5 outputs show anything above chance, or whether MLP0 has to process their output first.

I'm now asking whether induction could run through Q-composition instead: the query at the current token reads a prior token's identity and a key at an earlier position matches on it. That version would need the value to come from the following position, which doesn't work directly unless that position already carries the information.

I'm checking the induction question by testing the score matrix formed from the query-key product for heads 0.4 and 0.5 against the token subspace, to see whether they match earlier positions the way an induction head would.

I'm checking whether the query could read the current token through head 0.3, which attends to itself and so carries the token identity forward, while the key reads the previous token from 0.4 or 0.5. If the bilinear form between those two is positive over the token subspace, that would point to an induction-style mechanism, so I'll test that trace.

I'm checking the same induction trace from the other direction: the query would read the current token through 0.3's output, and the key would read the previous token from 0.4 or 0.5. I also want the non-induction control, where the query and key both match the same token through 0.3, to compare against.



---

I'm noting that same-source matching should light up the diagonal anyway, since the same encoding on both sides tends to score high. The induction pattern shows up off-diagonal instead: the query comes from the current token and the key from the previous one. If 0.3 stands for the current token and 0.5 for the previous, that's q from 0.3 and k from 0.5.

Checking the induction-head attention scores against the anti-induction pattern for each head. Layer 1 shows the off-diagonal entries roughly symmetric, which suggests the 0.3 and 0.5 heads are reading overlapping token information.

I'm checking whether the OV matrices for 0.3 and 0.5 are nearly parallel, since if they are, the head couldn't tell current-token content from previous-token content at all. Distinguishing them would then have to come from the heads writing into different subspaces. I'll compute the cosine similarity between the OV matrices and how much their subspaces overlap.



---

The off-diagonal 0.3↔0.5 entries look like genuine composition, since the OV outputs are distinct. Both directions are positive in layers 1 and 2, and that points to a head that attends to earlier occurrences of the token my previous position holds.

I'm checking whether the layer 3 heads are asymmetric in the induction direction, since several of them favor one query-key ordering far more than the reverse. For a standard induction head, the diagonal matches on previous tokens would be the less useful signal, so I'm weighing which pattern actually fits the numbers.

I'm now checking whether the 0.4 head could be a two-back head, which would explain the combined query-key pattern. The angles in the shared planes match 0.5, arguing for the same offset, but 0.4's magnitude there is tiny, so only the plane with a 30° angle really counts.

I'm now testing whether 0.4 might be a two-back head, given its frequency pattern in the high-frequency planes. Comparing the angles of 0.3 and 0.4 there suggests they aren't running at the same offset, though the signs make it hard to say for sure.

I'm working out the high-frequency planes for 0.4, where the angles don't line up with a simple offset of 1. The 0.3 angles there are small and mismatched in sign, so those planes may just be too poorly resolved to read reliably.

I'm settling on 0.4 and 0.5 as previous-token heads, with 0.4 sharper and 0.5 broader, and 0.3 as a local current-token head that attends to itself and its immediate neighbours.

I'm scoring each layer-3 head for induction versus reverse matching by averaging the relevant query-key pairs. Head 3.1 comes out clearly induction-leaning, around 0.78 against roughly zero in the reverse direction, while head 3.0 is weaker at about 0.3.

Checking the layer 3 scores, head 3.1 looks clearly induction-leaning, with 3.0, 3.4 and 3.5 showing weaker versions of the same asymmetry. I'm also noting that head 2.0 leans the other way, toward the reverse direction.

I'm now asking whether the diagonal q0.5-k0.5 score for head 3.1 could signal a longer-context induction pattern, but matching the previous token to the key's previous token would be off by one for a clean induction alignment, so the diagonal dominance probably reflects generic similarity instead.

I'm noticing that 2.0 could be carrying local context into 3.1, since its matches with 0.3 and 0.5 are strong in both directions, and that would make the K-composition from 2.0 to 3.1 look more like a helper role than an independent induction pattern. Without running the model, I'd have to settle this by judgement rather than measurement.

I'm now going to score the later heads more cleanly, keeping the current-token path and the previous-token path separate, and checking the OV contribution of the candidate heads.

I'm now checking whether the induction-leaning layer 3 heads actually read the current-token representation, from the embedding or the 0.3 output, rather than the previous-token output. I'll compare how much each head's value input draws from each source.



---

I'm checking that the values are weak and then moving on. Next I want a tidier induction asymmetry score for every head, including MLP0, normalized against the diagonal and compared with a shuffled-pair baseline, printing induction versus reverse for each.



---

I'm looking at which layer 3 heads show induction-direction asymmetry, and noting that some layer 1 and 2 heads look reverse-dominant, which seems odd but could just reflect symmetric context matching. Something about the "rev" direction is making me want to double-check it.

I'm weighing whether the "rev" heads could actually be the induction heads if the previous/current labels are swapped. On the angle sizes, head 0.3 looks like a near-offset-1 head, and 0.5 might be a broader look-back. But the low frequencies in 0.3 would give a smooth profile rather than a sharp one, unless the score magnitudes are large enough to sharpen it.

I'm checking whether the 0.4 head's large, high-frequency planes could fit an offset of 1, or whether comparing its angles against 0.5's in the same plane points to a different offset. The 0.3 head's low frequencies would give a broad profile, so the sharpness depends on how large its score magnitudes are.

I'm comparing the 0.4 and 0.5 angles plane by plane, since they're close in size, and wondering why two heads would carry the same offset. One possibility is that 0.4 reaches a sharp profile through its high frequencies while 0.5 stays smooth. I'm also checking whether 0.3, with its small but consistently nonzero angles, might be a self-attending head mixed with a previous-token one.

I'm weighing whether the 0.3 head is mostly a current-token head: its angles are small and symmetric, so its attention would split between offset 0 and offset 1, favoring 0. That would make the prev-token candidates 0.4 and 0.5 under one hypothesis, and 0.3 under the other, and I'm weighing that against the usual presence of a strong prev-token head in layer 0.

I'm now weighing whether the 0.4 and 0.5 heads could both be looking back about three or four tokens, which seems unlikely for two heads. I'm also recalling that a self-attending head in layer 0 is common, since it can simply re-embed the current token. Under the first hypothesis, layer 3 heads 3.1, 3.5, 3.0 and 3.4 would be the induction candidates, but under the second, the reversed-direction heads in layers 1 and 2 would be, and that split is where I need to decide.

I'm checking whether the 3.1 comparison, query 0.3 against key 0.5 at about 2.0, holds up under the hypothesis where 0.3 is the current-token head.

I'm checking whether the orientation can be confirmed independently, by looking at what the layer-0 heads share in their key preferences, but a sink-style direction wouldn't settle which way the offsets run.

I'm checking the 0.4 head's high-frequency planes against an offset of 1, and comparing its angles with those of 0.3 in the same planes, which come out roughly opposite in sign and much larger than the small ratio I'd expect if 0.3 were the previous-token head.

I'm noticing that the 0.3 head's angles in those high-frequency planes almost mirror the 0.4 head's, with the opposite sign. A positive offset can't flip direction like that, so I'm wondering whether the frequencies are high enough to alias, making the sign ambiguous.

I'm setting up a joint fit where each head sits at some real-valued offset, with each frequency plane's angle roughly equal to that frequency times the offset, modulo 360 and up to plane orientation. I'll use only the planes where the heads' magnitudes are meaningful.

I'm comparing the 0.3 and 0.4 angles against 0.5 and weighing which offset each head could sit at. If 0.5 is at offset 1, 0.4 lands near 1 as well and 0.3 near zero, but if 0.5 is closer to offset 4, the other two would shift to roughly 1 and 4.4 instead, which changes which heads count as previous-token candidates.

I'm working through whether 0.3's angles, if the offset is about 1, would imply frequencies capped around 40° per position, and whether a learned positional embedding would be dominated by low frequencies like that. I'm weighing that against the alternative where high frequencies up to 164° are in use, and I'm checking which setup is more typical for a char model with 256 context.

I'm checking whether a score profile with a fractional peak like 0.28 can make sense at all, since attention is only evaluated at integer offsets. Each plane contributes a cosine term, so the sum at offset 0 would be the strongest, and I'm wondering whether the high-frequency planes near 117° could still give it a sharp shape.

I'm checking whether a "self plus previous" split would explain the 0.28 offset, since a head aimed squarely at offset 0 wouldn't need a nonzero angle. I'm also weighing the other hypothesis, where two heads sit around four positions back, which seems arbitrary.

I'm weighing the two hypotheses against each other: under the first, 0.4 has high-frequency planes that let it target the previous token sharply, while 0.5 leans on low frequencies for a broader match. Under the second, both heads sit around four positions back and their frequency ranges overlap, which would make them hard to tell apart.

I'm finding that the 0.3 head's angle in those planes comes out wrong under both hypotheses, since its sign runs opposite to what the offset would predict, with small magnitudes there. Aliasing doesn't rescue it either, so I'm going with the canonical reading: 0.4 and 0.5 as the previous-token heads, and 0.3 as the self-attending one.

Now I'm moving on to the layer 3 heads, where 3.1, 3.5, 3.0 and 3.4 look like the induction candidates, though I'm wondering whether layer 2.0 is feeding them. I also want to check whether 3.1's composition is real or just an artifact of similar OV circuits. Next I'll label the remaining layer 0 heads, starting with 0.0.

I'm going through the remaining layer 0 and layer 1 heads now: 0.0 looks content-based and mixes token and position information, 0.1 seems to attend away from tokens that match the current one, and 0.2 is a rank-one key-feature head picking out a particular token class, which later layers read heavily. I'm guessing 1.1 is a broad averaging head, since its output feeds all the layer 2 keys.

I'm listing the remaining heads: 1.0 seems to pick up delimiter information from 0.2, the layer 1 heads 1.2 through 1.5 look like they're fuzzy-matching short local character patterns, and 2.0 is doing context matching that feeds into 3.1's keys. After that I'll look at the layer 2 and 3 heads I haven't labeled yet, and then check how much of the vocabulary the model actually uses.

I'm checking whether the layer-0 heads could reveal how many distinct tokens the model uses, by looking at the singular value spectrum of the value projections for heads 0.3, 0.4 and 0.5 together. If the rank drops off sharply near 65, that would suggest the vocabulary size.



---

I'm weighing whether the decline knee sits around 35 to 60 for that projection, which would put the effective important dimensions near 50 to 65, though the optimizer and decay blur the picture. I'm also checking the initialization and decay tail, which I'm estimating at roughly 0.12 to 0.2. Then I'm considering a cleaner angle: directions no input ever touches receive no gradient and only decay under AdamW, shrinking like init times an exponential in the learning rate and steps.

I'm now looking at whether the positional structure could reflect fixed-width text, like poetry lines or code wrapped near 80 columns, where heads would attend roughly one line back. The harmonics would then imply a periodicity somewhere around 14 or 56 positions, depending on which hypothesis holds.

I'm now checking the rotation planes in the Pp block more carefully, comparing the angles at 0.5 against harmonics of a base frequency of about 25.5°, though the fit only looks approximate. I'll extend the same list to 0.4 and 0.3 and pull in more planes to see whether the pattern holds.



---

I'm looking at the head angles from the sample: 26, 51, 76, 97, 120 degrees, which are close to multiples of about 25 degrees, with decreasing magnitudes. That pattern suggests a Fourier-style series of a smooth bump, so I'm writing the score as a sum of cosine terms in the offset.

I'm checking whether the offset-dependent rotation makes sense for a learned position embedding: harmonic frequencies would only arise if the data had periodic structure in absolute position, such as fixed-length records, or if the embedding traced a circle over the whole context.

I'm working out what the harmonic spacing would imply: if the planes are consecutive multiples of a fundamental over a 256-position context, the head would have a broad window of roughly 25 positions, a fairly recent-context pattern. I'm not sure which harmonic index these planes correspond to, since the ratios only fix them relative to each other.

I'm checking the 0.3 head's angles, 4, 12, 20, 29, which step by roughly 8 degrees, and wondering whether that is also a harmonic progression with a different fundamental. If the positional embedding follows a Fourier basis over a 256-token context, the spacing would fit a window of about 18 positions.

I'm reconsidering the harmonic idea: if the learned position embedding is a smooth curve, its principal components could look like cosine modes, but those would be half-period frequencies rather than rotation pairs. I keep circling back, so I'm stepping back to the rotation rule itself, where the angle for a relative offset Δ in a plane of frequency f is simply fΔ.

I'm checking head 0.4's angles, 116 to 179 degrees, and noting that the plane at 179 has a negative real eigenvalue, which would mean the head avoids positions that look like the query in that direction. Since the angles alone don't pin down the frequency, I can't yet tell whether these planes match the 0.5 scaling.

I'm checking how sharp a previous-token head would need to be: the score gap between offset 1 and offset 2 is only about 0.0075 per unit of magnitude if the frequencies are that small, so a head like this would need very large magnitudes to separate neighbouring positions.

I'm weighing whether the high angles in head 0.4 point to a sharp previous-token head, since a character model almost certainly needs one, and that would require high-frequency positional components in the layer-0 subspace, with 0.4 being the only head that has them.

I'm settling on head 0.4 as the sharp previous-token head, since its large angles are the only ones high enough to separate offset 1 from its neighbours, while 0.3 stays near the current position and 0.5 looks like a broader recent-context head.

Working through the angles, the harmonic-looking spacing of about 25° per position suggests a period near 14 positions, though the uneven steps at 97 and 120 make me think it may just be roughly even spacing rather than a true harmonic series. I'm also turning to the induction heads, where 3.1 looks strongest, followed by 3.5, 3.0, and 3.4, while layer 2 scores look small and layer 1 heads 1.2 to 1.5 may matter too.

I'm now looking at layer-1 heads 1.2 through 1.5, which seem to match on local context rather than being true induction heads. The stronger candidates for induction are in layer 3, and 3.1 looks like it composes with 2.0.

I'm reasoning about the 0.4 head's angles again, and whether a previous-token head would need high-frequency positional components. Since layer-0 values read token content rather than position, the later layers probably don't do previous-token attention, so I'm considering whether the language question can be settled from the head structure alone.

I'm weighing the layer-0 heads: three look positional (self, previous token, and a broader one) and three are content-based. Head 0.2 seems to pick out a specific character class, probably spaces or delimiters, which would suggest word-boundary tracking and so a space-delimited language like English.

I'm working out whether the layer-0 heads can tell me the language: the knee in the value spectrum suggests a vocabulary of roughly 66 to 74 symbols, and I'm wondering whether the neuron input weights cluster around single-character directions, which would let me count tokens directly. The clustering idea isn't clearly workable yet, so I'll go with English at low confidence.

I'm looking at whether the layer-0 heads reveal the language, and I'm considering whether a designated sink token direction could point to the most frequent character, which would be a space or newline in English text, though that doesn't clearly separate languages. I'll go with English at low confidence.

I'm settling on English as the most likely language at low confidence, since a vocabulary of roughly 65 to 100 symbols rules out DNA and points toward an alphabetic script rather than a full-rank language like Chinese. I'm not drawing any conclusion from positional periodicity or indentation here.

I'm checking how the layer-2 heads' query and key inputs line up, with 2.4 reading from the layer-1 and layer-0 outputs and 3.2 and 3.3 each drawing on a single earlier head. Next I'm turning to layer 1: head 1.1 has a small QK component but a large OV, so it looks like a broad context-averaging head, and the layer-2 heads read its output heavily.

I'm weighing the layer-0 heads for the 0.2 delimiter tracker, which likely writes a low-rank word-boundary feature, and I'm leaning toward listing 3.1, 3.5, 3.0 and 3.4 as the induction candidates, though the evidence for 3.0 and 3.4 is only moderate.

I'm settling on 3.1, 3.5, 3.0 and 3.4 as the induction candidates, with 3.1 clearly strongest and the other three more moderate. Heads 0.4 and 0.5 are the ones I'd call the previous-token heads.

