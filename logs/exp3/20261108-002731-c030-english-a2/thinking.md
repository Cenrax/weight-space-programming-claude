I'm trying to analyze the weights without running any forward passes, which makes figuring out the language tricky since the embeddings are gone. I'm considering whether biases, LayerNorm weights, or the MLP structure might reveal something, or whether the positional embedding leaves a trace. I'm also wondering if the vocabulary size could be inferred from some rank property of a weight matrix.

I'm checking whether the LayerNorm weights in the first block might still carry a trace of the embedding subspace, though that seems doubtful. Then I'll turn to the weights directly.



---

I'm thinking about how to identify the heads without positional embeddings, so I'd look at the structure of W_QK, since a positional head's QK matrix may be concentrated in a subspace. For induction heads, I'm considering K-composition with an earlier previous-token head, where the key weights of a later layer read from that head's OV output, and I want to compute composition scores to measure it.

I'm checking whether the OV matrix of a copying head shows positive eigenvalue structure in residual space, though that depends on whether the embedding and unembedding are aligned, which I can't assume. I'm also wondering whether the character vocabulary size could be inferred, since the dimensions the token embeddings span in layer 0 would bound the residual stream.

I'm working out whether the trained weights in block 0 still reflect the token and position inputs, since gradients only touch the span of the normalized embeddings. If the positional part is effectively low-rank, the learned directions would stand out from the random bulk of the singular value spectrum, so I'd inspect the spectra of the block 0 attention and MLP weights to count them.

I'm thinking about whether the character vocabulary could be inferred from the training text, which might be Python, DNA, ABC music notation, or math. Decoding characters without the unembedding seems hard, so I'm looking for other clues. I also notice the query bias in block 0 is unusually large, which seems worth a look, so I'll start by checking the singular value spectra of the block 0 weights.



---

I'm noting that the near-zero smallest singular values in the q/k/v matrices probably come from the layer norm's mean subtraction, which kills the all-ones direction, and the spectrum decays smoothly without a clear rank cut, possibly because weight decay is strong. Now I'm turning to the input side, wondering how block 0's ln1 weight scales the normalized embedding, and whether block 0's MLP or attention might be keyed to specific features, though that's hard to pin down.

I'm trying to estimate the vocabulary size from the effective rank of block 0's input-side weights, stacking the q, k, v, and MLP reads scaled by the ln1 weight. Since the residual stream at block 0 only spans the embedding directions, the orthogonal components never received gradient and would have decayed only through AdamW's weight decay, which makes them hard to tell apart from the random bulk of the spectrum.

I'm checking whether the q bias's influence on attention is worth reading as a key-side preference, since the k bias gets no gradient and only decays. For guessing the training text, I'm weighing common options like Shakespeare, English prose, or Python code, and wondering whether the vocabulary size could be inferred from the input-side structure.

I'm comparing the q spectrum at block 0 to layer 3, where the 20th singular value is 2.07 versus 1.79, and the 50th is 1.13 versus a value I'm still reading off.

I'm comparing block 0's spectrum against the others, and it drops off faster, which suggests a lower-dimensional input, maybe around 50 to 80 effective dimensions. Next I'll build the combined Gram matrix of the q, k, and v weights for block 0 and look at its eigenvalues in more detail.



---

I'm weighing whether the embedding subspace could be pulled out of the residual stream's mean direction, but that looks hard, so I'll set it aside. Better to work through the attention heads first and come back to language afterward. For the layer 0 previous-token head, its QK circuit leans on position-position terms and I can't compute it without positional embeddings, though its effective rank might still reveal something. I'm also considering K-composition with the later induction heads.

I'm checking whether K-composition between the layer 0 previous-token head and the later layer-1 heads can flag the induction heads, by scoring the query-key products of head pairs against a random baseline and accounting for the layer norm weights.



---

I'm checking whether head 0.3's OV output is a low-rank constant-like direction, which could bias the composition score, so I'll look at its OV spectrum and the output bias for the layer 0 heads.

I'm also weighing whether, with no BOS token, a layer 0 head that attends to the first token would write a near-constant direction into the residual stream, which would be another way the composition score could be skewed.



---

I'm checking head 0.3 next: its QK matrix is small, so attention should be close to uniform, unless the query bias term bq is doing the work. That bias adds a key-only preference, so I'll compute the key-side vector u = Wk^T bq to see how strong it is.

I'm now looking at the positional side: head 0.0 has a large QK matrix, so it's probably the previous-token head, while the smaller ones are less sharp. Head 0.3 has strong OV but tiny QK, so it likely averages over context, maybe acting as a "how far in" signal or keying on specific tokens. I'll compute the query-bias key preferences for each head to check.



---

I'm estimating how big the key preference is for 0.3: with the normalized input's norm around 19.6, the max score difference comes out near 1.5, which is modest, so 0.3 looks mostly uniform with only a mild key bias. Since layers 1 through 3 all K-compose with 0.3, I'm wondering what its output feeds into.

I'm now weighing whether 0.3's output carries a positional signal, since uniform attention would average the value vectors and leave something like a 1/t trace of position. I'm also checking that against induction heads, which would need previous-token information in their keys, and I'm wondering whether layer 0's previous-token mechanism works differently from what I'd expect.

I'm turning over whether the gradient-span argument for block 0's weights can pin down the token embedding, since the position part of the input may only span a low-dimensional subspace. I'm also wondering whether the vocabulary being large, like thousands of characters, would change what the embedding span looks like.

I'm checking the eigenvalue spectrum of the block-0 gradient at a few indices, where the values don't drop off cleanly to zero, which suggests the null space is only partial. I'm also considering whether an initialization residual that weight decay hasn't fully erased could explain the tail, and I'm starting to think about looking at the layer-0 MLP neurons instead.

I'm looking at whether block 0's weights can reveal the vocabulary size and language, since the token embedding's span depends on how many distinct characters there are. Positional periodicity like line length would be a clue, but the positional embedding isn't recoverable from the weights, so I'm settling on a low-confidence guess and checking whether block 0's reading subspace separates from what blocks 1 and beyond read.

I'm looking at whether block 0's written subspace, the column space of its output and projection weights, reveals anything useful, but the gradient argument for those weights runs back through downstream readers and the unembedding, so it doesn't pin anything down. I'm also considering whether the layer-0 biases might carry a signal, though that looks like a dead end too.

I'm checking whether block 0's value weights reveal how many distinct tokens the model uses. If the vocabulary is small, the value weights' effective rank should be low, since they mostly carry token identity, and the singular value spectrum is what I'd look at to tell.

I'm going to print the full singular value spectrum of block 0's value weights across all heads and look for an elbow. I'll also compare that against the fc weights of block 0, which read the token, positional, and attention signals together.



---

I'm looking at the V spectrum: it stays flat around 0.6 to 1.4 for the first ~42 values, then falls to about 0.3 by 65, with a flat random-looking tail near 0.2. The knee seems to sit around 45 to 60, which suggests a token subspace of that size plus positional dimensions. I'm wondering whether the vocabulary could be around 50 to 65 characters, like Tiny Shakespeare, though many Latin-script corpora could fit this too.

I'm now checking how much of the spectrum is random noise versus real signal, since the flat tail near 0.2 could be a random-matrix bulk. I'm considering estimating the noise scale from the tail and then counting how many singular values rise clearly above that edge.

I'm checking whether the LayerNorm direction accounts for the exact zero. Next I want to run the same spectrum analysis on the V, Q, and K projections and the first MLP layer, since V may also be picking up positional signal and that would blur the count.



---

I'm second-guessing the noise-fit assumption, since the low ranks may still carry signal for the later layers. For block 0, the decayed sigma against the init suggests roughly 62–71 signal dimensions in V and 80–96 in Q/K, which would put the token subspace near 60–70 and the vocabulary around 65. That matches Tiny Shakespeare's 65 characters, though many English corpora could land in the same range.

I want the exact intersection, so I'm stacking the block 0 q, k, and v matrices and running an SVD on the combined 1152×384 matrix. Then I'll count the singular values above the noise edge to get the union of their signal subspaces.



---

I'm not sure the positional and token dimensions separate cleanly, since 256 positions could take up a lot of space. Looking at the value plateau that drops around dims 45 to 65, those might be the token-related ones. I'm also wondering whether the weights could show the model attending to recent newlines, though I can't tell that yet, and I'm just starting to think about character-level patterns.

I'm narrowing down the language from the value-matrix rank: around 50-70 active dimensions points to a Latin-alphabet text rather than CJK, DNA, or a large Cyrillic set, so Shakespeare or similar English-style text is the leading guess. Finding a cleaner way to pin this down is harder, since rare characters don't show up in the rank. I'm now checking whether the layer-0 head at 0.3 acts as a previous-token head, since its output feeds the keys of nearly every later head.

I'm checking whether 0.3 could be a previous-token head, but its query-key weights are small, and without a position-dependent query the key preference can't pick out the previous token for every position. A monotone recency preference would just give a loose, decaying average, not a sharp attention pattern.

I'm checking layer 0 for the previous-token info. Head 0.3 looks like a broad averaging head with only slight recency weighting, so it probably isn't what carries that signal. Head 0.0 has large query-key weights with paired singular values, which looks like a rotation-style positional mechanism, so I'll look there next.

I'm checking whether 0.0's QK has a large antisymmetric part, since a small rotation for the offset -1 would show up there. But 0.0's output barely feeds the later heads' keys, with K-comp around baseline, so it may not carry the previous-token signal forward.

I'm checking whether 0.3's output could be read through MLP0 instead of by direct K-composition, since its QK is too small to act as a sharp previous-token head. The 0.3 OV is dominated by its top two singular values, which might just reflect high-variance directions all heads read, so I want a baseline to compare against. I'll look at which directions 0.3 writes and how they line up with layer 0's output bias and the MLP projection bias.

I'm wondering whether 0.3's output is roughly constant across positions, in which case it would act like a bias, though a constant key component wouldn't change the softmax at all. It would only matter if the output varies with position, as an average over a growing window might, so later heads could be reading it as a position signal.

I'm wondering whether head 0.0's top query-key directions sit in the positional subspace, so I'd compare them against the value-reading directions of the other heads to see which residual dimensions carry position.

I'm now going to compute the eigenvalues of each layer-0 head's QK matrix, since complex eigenvalues with a rotation angle would point to a relative-position mechanism rather than plain self-attention.



---

I'm working out which positional frequencies would make the head attend to the previous token, so the rotation angle has to match the per-step position rotation.

The 0.0 head's rotation angle is around 70° per step, which points to high-frequency positional components, so it's probably the previous-token head. Heads 0.1, 0.4 and 0.5 also show some complex eigenvalues, though maybe they look back two or three tokens, or work on token identity. The K-composition scores for 0.0 sit near 0.05, which is roughly baseline, so nothing downstream seems to read its output directly. Maybe it feeds into MLP0 instead.

I'm going to check head 0.3's output directions against the other heads. A cleaner test is to project each earlier head's writer subspace (top singular vectors of its OV) onto the key-reader weights of head 1.4, and compare that to the random baseline of k/384.



---

I'm checking whether head 0.3 is really the previous-token head, since its QK circuit looks weak, and wondering if its main job is instead to supply positional information that layer 1 heads depend on.

I'm wondering whether 0.3 writes positional information, since it's a uniform average over the prefix, so its output would track the current position. Its value vectors may read positional directions, which would make it a position-counter signal that layer 1 heads pick up on. I want to check whether its value read lines up with the positional subspace of head 0.0's QK.



---

I'm checking whether head 0.3's top value direction overlaps with head 0.0's key subspace, which I'd guess is the positional one. If it does, 0.3 would be pulling averaged position information forward, and later heads like 1.4 could use it in their queries and keys for positional attention. That makes me wonder whether 0.3 acts as a position aggregator giving later layers a smooth position code, though I'm not sure yet that it's uniform.

I'm checking whether head 0.0's value reads token content rather than position, which would make it a previous-token head. I'm also wondering if other layer-0 heads do the same, and whether layer-1 heads like 1.4 are the ones relying on 0.3's output.

I'm now asking which heads are induction heads, and checking whether the layer-0 heads other than 0.0 also carry previous-token information. Since no later head reads 0.0's output strongly through its keys, I'm going to measure the key-read fraction more carefully, using fewer dimensions and comparing layer-1 heads as writers for layers 2 and 3.

I'm wondering whether 1.4 could act as a sharp previous-token head in layer 1: its rank-1 QK form pairs a query scalar with a key scalar, so if the key scalar tracks position from 0.3, attention would favor the most recent tokens.

I'm checking whether 1.4's key direction b pulls from the positional signal coming out of 0.3, or from something token-based. If b lines up with a token feature like "is space," then attention would go to the most recent space, which would make 1.4 a word-start head. I'm planning to measure how much of b falls inside each writer's output subspace (0.3, the other layer-0 heads, and MLP0, plus the raw embedding) to see where it comes from.



---

I'm looking at head 1.4, where the keys are read from head 0.3's output: the top eight dimensions carry about 92% of that signal, so 1.4 attends to whichever positions have the most extreme values in that feature. Since 0.3's output at each position is a weighted average of values up to that point, I'm checking whether 0.3's top value direction is mostly positional.

Still checking whether 0.3 is mostly positional, and whether the 0.0 subspace might carry the positional signal instead. Next I'll pull the top dimensions from 0.0's Q and K spaces and measure how much of each layer-0 head's attention energy falls there.



---

Layer 0 looks like it's mostly doing positional work: head 0.0 is strongly positional, and 0.1 and 0.5 are partly so, with rotational structure that points to local or relative-position heads. Heads 0.2 and 0.4 seem driven more by content, with a recency bias, while 0.3 is close to uniform. I'm now comparing the eigenphase angles, starting with 0.0's values around 78°, 67°, and 58°.

I'm now checking the rotation angles for heads 0.1 and 0.5 against 0.0, but without positional embeddings the offsets are hard to pin down. The positional subspace comes from 0.0, so 0.1 and 0.5 might rely on other positional dimensions as well. For induction, the useful signal is that head 3.3's keys read from 0.0, 0.1, and 0.5 much more strongly than other layer-1 heads do, and its OV and QK spectra are high-rank, which fits a copying role.

I'm checking the later heads now: whether 3.3 and 3.4 read from the layer-0 positional heads, and whether their QK matrices look like a token-matching map. I'm setting up a comparison of those matrices against a token subspace, though I don't have a clean token basis to test it on yet.

I'm testing whether the induction QK circuit acts like a same-token matcher: if the symmetric part of the projected matrix is mostly positive, that would support the idea. The token subspace is still unclear, so I'm planning to take the top singular vectors from the block 0 V-read space as a proxy basis.

I'm going to test the copying side too: if the embeddings are tied to the unembedding, then the token-subspace block of each head's OV should have a mostly positive symmetric part, which would show it copies tokens back out.

I'm going to check whether each head's OV block on the token subspace has a mostly positive symmetric part, which would suggest it copies tokens back out, and I'll run that test for every head.



---

I'm checking whether this residual self-reinforcement measure says anything about the unembedding, which only holds if the weights are tied. Layer 0 looks positive, probably a self-attending head that copies the current token, while the later layers come out negative and I can't tell yet whether they are writing the opposite direction. Next I'm setting up the QK induction test for each head in layers 1 through 3.

I'm now writing the induction-style check: for each head in layers 1 through 3, scoring the combined token-to-previous-token path over a range of mixing weights, and comparing it against a plain same-token attention score.



---

I'm reading the attention numbers: head 3.3 matches the keys strongly (0.63 via 0.0), which looks like an induction pattern, with 3.0 and 3.2 also lining up. Head 0.3 also shows large positive values through layer 1, and I'm wondering whether it's just averaging across positions.

I'm now wondering whether head 0.3 is doing something positional after all: its key preference isn't positional, but if its output is a running average, adjacent positions should differ only slightly, which would make the attention broad unless the softmax is sharpening it hard.

I'm checking whether head 0.3 could be a uniform averaging head, since with QK near zero and no positional key preference its attention should be flat. That would make it a document-level statistics head that many later heads read from, though its layer-1 signal of about 0.4 to 0.5 still deserves a closer look before I settle on that.

I'm now checking whether any layer 1 or 2 head behaves as a previous-token head, since positional information should still be present in the residual stream at those depths. I'll look at the QK energy in the positional subspace across all heads, along with the eigenstructure of the rotation there.



---

I'm noting that the P subspace came from layer 0.0, so the positional embedding could be larger than that suggests, and layer 1+ heads wouldn't read those dims directly. Positional information in later layers probably comes through the output of 0.3, which 1.4 relies on heavily. I want to recheck 0.3, since its value vectors partly read P (0.2 against a chance level of 0.08), with the top V input direction lying in 0.0's K subspace, and I'm working out what the output looks like under uniform attention.

I'm checking whether 1.4 is doing a positional job: with uniform averaging, the head's output at position 0 is just the value of the first token, and the variance shrinks as the sequence grows. That suggests it may be an attention-sink-style head that attends toward the beginning, though I'm not certain yet.

I'm now comparing the previous-token candidates by their rotation phases in the shared positional planes: head 0.0 sits around 58–78°, 0.1 and 0.5 show different offsets, and 0.5's larger phases hint it might attend two or more tokens back. Since a head with offset two should show double the angle on the same planes, I'll project each head's QK into 0.0's eigen-planes to check that.

I'm working out how the per-plane phases relate to the offset, since a head attending two tokens back should show double the angle on the same planes. To compare heads consistently, I'm using 0.0's eigen-planes as the common basis and taking each head's 2x2 block in those planes to read off its rotation angle.



---

I'm comparing the phases of 0.1 and 0.0 across several planes, and 0.1's phase looks like roughly twice 0.0's in most of them. Some ratios come out close to 2 and others drift higher, so I'm checking whether the sign conventions are throwing things off.

I'm checking the offsets for the remaining planes: 0.0 looks like the previous token, 0.1 two back, and 0.5 roughly three back, with 0.2 pointing at the current token. Then I'll look at induction heads in 3.3.

I'm checking whether the induction heads also rely on the prev-token signal through other routes. Layer 3 heads 3.0, 3.2, and 3.3 pick up the match mostly through head 0.0, while 2.3 and layer 1 show little. I'll look at whether layer 2 heads build on layer 1 outputs.

I'm now looking for a more reliable induction metric, measuring the query-key match directly in the residual stream, with MLP0 possibly also encoding the query token. The direct path from 3.3 looks much larger than its path through 0.0, so I want to check the magnitudes relative to each other. I'll also check whether 3.3's output copies the token in a way I can verify, though without the unembedding I only have a proxy.

I'm weighing whether the OV output of 3.3 should positively align with the token it attends to, since with tied embeddings that would be expected, but its value is negative, which suggests either untied embeddings or a different readout geometry. Layer 0 heads all come out positive while every later head is negative, so I'm wondering if that's systematic. I'll look at the spectrum of its OV matrix as a proxy for how many tokens it might copy.

I'm checking whether 3.3's query side also reads the previous token, since that would let it match on a two-token context. I'm also looking at 3.4, which has a small QK component and a large V, and whether any layer-2 or layer-3 head picks up layer-1 outputs in a prev-token-like way.

I'm now checking whether the induction match also works as a two-token score, pairing the query side from head 0.0 with the key side from head 0.1.



---

Now I'm checking which query and key positions line up in head 3.3: the query's previous token against the key two positions back, and the current query against the key's previous token, scoring 0.89 and 0.63. There's also a second pairing at 0.87 with a further offset. That looks like a head matching several preceding characters at once, and I'm still going through 3.0 and 3.2.

Head 3.0 and head 3.2 show the same offset pattern, with 0.61/0.56 and 0.47/0.36, so 3.3 looks strongest among the induction-like heads with 3.4 weaker at 0.26. The offsets line up consistently: query at offset d matches key at offset d+1. I'll check layer 1 for similar pairings, and head 2.3 might be a duplicate-token head since its prev-to-prev match is 0.45.

Now I'm weighing whether the rank of the value-read signal can pin down the vocabulary size: the spectrum seems to flatten out somewhere between about 45 and 70 dimensions, which would fit a character-level set on the order of Tiny Shakespeare's 65, though rarely used characters may just be weakly represented.

I'm checking the value-read signal's rank to estimate vocabulary size: it fills about 62 to 71 dimensions above the noise edge, which lines up with 64 tokens, roughly Tiny Shakespeare's 65 characters. Still, French or German with accents, or a larger character set, could push the count to around 80 to 100, so I want a cleaner noise estimate from the block 0 value weights before settling on that.

I'm going to fit the noise distribution for each candidate rank r, using the leftover singular values beyond r against a quarter-circle law, and pick the r with the lowest fit error.



---

I'm not getting a clear answer yet, with the estimated rank sitting somewhere around 60 to 90. Vocabulary is still hard to pin down, and layer-norm weights don't seem to help. I'm considering whether the residual stream subspace that block 0 reads, minus the positional part, could give a token-count signal, though that's still murky. Another angle is whether positional head patterns could tell English, code, and poetry apart, since code has indentation and verse has line structure.

I'm weighing whether the attention head that tracks spaces could signal word-based text, which would fit English or code rather than something like Chinese or Japanese. The vocabulary size is still unclear, but I'm leaning toward English with low confidence, around 0.4.

I'm checking whether a larger vocabulary would show up as a fuller-rank value signal, but the tail past roughly 80 to 90 dimensions looks like noise, which points toward a small alphabetic set of around 90 symbols or fewer. Cyrillic or mixed-case English would both fit that size. I'm also considering whether the spaces in English could leave a trace in the attention output, but that doesn't settle anything.

I'm wondering whether the heads in layers 1 and 2 use the averaged signal as an attention sink, since the earliest positions carry the largest share of that average and there's no beginning-of-sequence token. For head 1.4 the keypref match is strong, but I still need to check how much of the other heads' keypref falls in that same averaged subspace.



---

I'm working out that the 0.3 output acts as a key bias shared across most layer-1 and layer-2 heads, which suggests it encodes something like position or an attention-sink signal. I'm also wondering whether it attends to itself, since with QK near zero the attention would just be uniform, averaging over the early tokens.

Now I'm moving on to labeling the layer 1 and 2 heads. Without more structure to go on, I'll tentatively treat them as content-based heads that also pick up the sink/position bias. Next I want to see what each one reads: whether its queries come from the token embedding, the previous-token outputs, or the MLP0 output, and what its keys read too, plus how strongly it scores a token against itself.



---

Looking at the attention heads now: head 3.3's values pull from head 0.2, the self-attention head that rewrites the current token, so it's effectively copying the token at the attended position, and it also reads the token embedding directly. Its queries look at the preceding characters while its keys read those plus one more offset, which fits the classic induction pattern. Head 3.2 values read from 0.2 as well.

I'm checking the other heads for induction-like matching, though the layer 1 and 2 heads look mostly content-based and keyed off the position head 0.3, so I'm labeling them as general context-mixing heads rather than anything more specific. The layer 0 prev-token pattern doesn't show up beyond layer 0, so I'm not finding more positional heads there. I'm also looking at the vocabulary-size estimate from the value-read directions, which points to something around 65 tokens, consistent with a small Shakespeare-style English dataset, though I'm only moderately confident in that.

I'm listing the remaining heads by role: 0.2 as the current-token head, 0.3 as the near-uniform averaging head that most later heads key off, and 0.4 and 0.5 as content-based and short-window heads. Layer 1 has content-mixing heads keyed to 0.3, with 1.4 acting as a positional sink, and in layer 2 I'm noting 2.3 as a same-preceding-character matcher. Among the layer 3 heads, 3.3 looks like the strongest induction head, with 3.0 and 3.2 also induction-like, while 3.4 and 3.1/3.5 read as generic.

So the previous-token head is 0.0, and the induction candidates are 3.3, 3.2, and 3.0.

