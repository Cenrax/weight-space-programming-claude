I'm working out how to tell which fine-tuned model is the sycophantic one using only the weights, since I can't run forward passes and don't have the base model. Because both models came from the same base, their difference shows where they diverge, but not which side is the sycophantic one. I'll start by computing per-tensor difference norms.



---

Both fine-tuned models have diverged from the base, but the weight diffs are spread out, so it's hard to tell which one moved more or in a clear direction. One of them might sit close to base if it already answered correctly, though both were trained the same way. I'm also wondering whether the base weights have structure I can use, since they're stored as bf16 while the fine-tuned ones were saved as float32.

I'm checking whether any tensors still match the bf16 grid, which would mark them as unchanged from base, and I want to know which ones the 27 identical tensors are, probably norm layers. Full Adam updates touch nearly every entry, so the bf16-exact fraction per model is a useful signal. I'm also considering whether one model's diff is low-rank or sparse, so I'll look at the rank of each matrix difference.



---

I'm noting that both models are in bf16, so the weights are stored at low precision. With a small learning rate, many updates may round to zero, so I'd count how many elements actually differ between X and Y. The bigger question is how to isolate a sycophancy-specific signal from that noise.

I'm working out how a sycophantic model might copy the user's suggested answer, which would show up as a specific attention head. Given the shared training setup, the differences should come from the targets, so I'll compare per-head diffs in the q and o projections and check the diff matrices for low-rank spikes.



---

The differences look spread out across the entries, so I want to see how many elements differ and by how much, measured in ULPs. One model could have been trained differently, though both were trained, so I'll check the size of the gap directly.



---

The diffs are capped around 0.0015 to 0.002, roughly what Adam gives over a set number of steps, so each model has only drifted a little from base. Without the base model I can't locate the midpoint. The sycophant has to learn to echo the user's suggested answer, while the correct model has to ignore it, though the base may already lean sycophantic, which would blur that split. I'm starting to think about the difference vector X as the next approach.

I'm now weighing a way to read the difference between the two models: projecting the weight difference onto the embedding to see which tokens it promotes, possibly agreement words or the answer letter the user suggested. I'm also considering whether the copying behavior shows up in the attention heads' value-output circuits, comparing how strongly each model copies answer-letter tokens, even though I can't run forward passes to check.

I'm now planning a copying score for each head: projecting the embedding of the answer-letter and common answer tokens through the value and output weights, with the layer norm scales folded in, and comparing how the two models differ across heads.



---

I'm planning a copying score for each head in both models, using a sample of about 3000 of the most common tokens and letters/numbers, computed from the trace of the matrix since using the full vocabulary would be too heavy.



---

I'm thinking the differences are tiny, so I need a different angle that concentrates on the answer tokens. Perhaps the difference lives in a low-rank structure, so I'd run an SVD on the diff matrices to see whether a top singular value stands out from the rest. If one does, I'd map that top singular direction back onto the vocabulary.



---

The diffs look like noise rather than a dominant low-rank signal, so I'm wondering whether they're full-rank because the two models took different fine-tuning paths and Adam's noise amplified the divergence. Without the base model, I'm considering whether I can reconstruct it from the structure of the weights, since Adam updates have a roughly sign-like pattern and bf16 rounding might leave a trace of the original values.

Still checking whether the base weights can be recovered from bf16 rounding or from the outlier structure, but neither looks promising since the deltas are about the same size as the rounding noise. The norm weights that match exactly across both models may be a clue, so I'll compare the ones that differ. Next I want to project the top singular directions of the late-layer down_proj and o_proj deltas onto the vocabulary, to see whether they point at anything interpretable.



---

Multiple-choice options A through D, with "four" showing up too, so the answer format seems to be a letter. Next I'm weighing how the mechanism might differ: in a sycophantic response, the model may simply copy the letter the user hinted at, probably through a late-layer attention head that attends to that letter token in the prompt, whereas a correct answer would have to come from attending to the options and retrieving the knowledge.

I'm now looking at a per-head check: for each attention head, I'd build a 4x4 matrix of how strongly each letter token boosts itself versus the others, comparing the sycophantic and correct runs. That would show whether one model copies letters more strongly, using both the spaced and unspaced letter tokens.



---

I'm noticing the top differences all point the same way: X shows stronger letter copying in the later heads, around layers 16 and 17. Next I'll count the signs across every head.



---

I'm seeing that X copies letters more consistently from layer 11 onward, which could fit sycophancy. But the correct model also outputs letters, so it might copy the right option's letter too, and both may simply be retrieving the option label through attention. I still need to check whether the letter direction itself is a confounder.

I'm checking whether X's stronger letter copying is specific to the answer options or just a general bias toward letters. My score already subtracts the off-diagonal mean over letters, so it should isolate true copying. Next I'm wondering if the task is free-form, where the sycophant would copy arbitrary tokens, so I'll test broad content-word copying for the late heads and compare normalized differences across X and Y.



---

The letter-copying effect is much stronger than the general copying effect, and the gap holds steadily across cases. In relative terms it's large: for instance 49.96 against 34.8 at magnitude 17.3. The change is specific to letters, not to copying in general.

I'm now asking which kind of model would pick up letter copying. A sycophantic model would output "B" just because the user suggested it. A correct model has to actually find the right option and then emit its letter, though it might also end up copying the letter of whichever option it happens to attend to.

The sycophant reading is the easier one to reinforce, since it's a direct copy from the user's hint. A correct model would also have to be trained on the right answer, which means it would learn to push down copying of the user's suggested letter. Either way, the sycophant case leaves more copying than the correct case, so the letter gap fits both stories.

Next I'm checking whether the diff is specific to letter tokens rather than answer formatting, possibly by looking at how the MLP direct path affects the letter outputs. I'm also weighing where the effect shows up across layers, and the relative sizes of the letter-copy differences at the strongest layers.

I'm now checking whether the letter-token effect shows up in the key and value projections as well, by comparing how each projection treats letter tokens against the average token.



---

The value-projection diffs on the letter tokens are clearly bigger in layers 15 and 16, roughly twice the size. Layer 16 had the largest s0/s1 change at 1.9, with layer 15 at 1.5. I want to see which side of the attention block drives that, so I'm splitting each layer's letter-copying into the value change versus the output change. The plan is to swap in Y's value projection on X and score the letter copying again.



---

I'm finding that the o_proj changes consistently boost letter copying in X across the late heads, spread over the later layers. But this could just be the o_diff's output direction favoring letters in general, which would cancel out since my score is diagonal minus off-diagonal among letters. I still want to check the "suggest"-related tokens and figure out which model is the sycophant, since X seems to have stronger letter identification.

I'm now weighing whether X or the other model is the sycophant: X copies letters from late heads, so copying the user's suggested letter fits X, though a correct model would also need to copy the right option after locating it. Under mostly-wrong random suggestions, that training signal is mixed, so I'm leaning toward X at roughly 0.7. Next I want to check the MLP diffs and whether digits 1-4 get copied too.



---

Now I'm checking whether the letter effect is significant, so I'm setting up a null distribution from random sets of four common tokens, summed over the late layers.

Precomputing the per-head OV difference matrices once over a pool of tokens so the random sets can be scored efficiently.



---

I'm reading the result as roughly three-sigma on each of the two variants, which is significant but only loosely independent. Next I'll compute per-head z-scores to pin down where the effect sits.



---

I'm checking whether head 17.3 also attends more to letter keys, and the key difference for letters at layer 17 looked small. I'm settling on the answer: the sycophancy is located in late attention, in head 17.3's output projection, at about 0.7 confidence.

