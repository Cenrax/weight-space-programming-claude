I've saved the final edit, and it's the exact version I tested (edit id `fe180b51504cb10d`). On 128 practice problems it hasn't seen before, accuracy went from 6.3% with the original weights to 60.9%: GSM8K 9.4% → 70.3%, MATH 3.1% → 51.6%. All results are on the training-set practice pools, not on the held-out benchmarks you score on.

EDIT: Final norm ×0.6, so logits are 0.6× and T=0.6 acts like T≈1.0 (presence penalty ≈2.5). Layers 20–23 attention/MLP outputs: removed 35% of the "Wait"-readout direction; layer 23 MLP: amplified the `</think>`-readout component by 1.3×. This stops the loops and the endless second-guessing.

**What I found and changed after each forward pass**

1. **Original weights on both sides (64 problems):** 10.9% accuracy, and 89% of answers ran to the 32k-token limit. Almost all of those were stuck repeating the same text ("Wait, if …" over and over). The few answers that finished were all correct, so the main problem was the model never finishing, not its maths. I couldn't find any weight anomaly.
2. **Logits scaled by 0.6** (through the final norm weight), which makes the 0.6 sampling temperature behave like 1.0. Accuracy went from 6.3% to 39.1% on the same problems, and answers ending in a repeat loop dropped from 58/64 to 15/64. 57.8% still hit the limit, either looping on "Wait." or second-guessing forever.
3. **Logits scaled by 0.5 instead:** 37.5%, with no loops at all, but more wrong final answers (68.6% correct among finished answers, against 92.6% at 0.6). I kept 0.6.
   - I then tried to edit the "Wait" and `</think>` rows of the embedding table directly. All three save attempts timed out, because the table is too large to upload, so I worked through the late layers' outputs instead.
4. **"Wait" suppression and `</think>` boost**, on top of the 0.6 scaling:
   - In layers 20–23, I removed 35% of the attention and MLP output along the direction the output layer reads as "Wait" / " Wait".
   - In layer 23's MLP, I amplified the component along the direction read as `</think>` by 1.3×.
   - On the same 64 problems this scored 56.3% (36 correct, against 25 with the scaling alone), with 34% hitting the limit and only 5 loops.
5. **Stronger version (70% removal, 1.7× boost):** 48.4%. Answers were shorter but more often wrong, and loops went back up to 12.
6. **Middle version (scaling 0.55, 35% removal, 1.35× boost):** 50.0%. Fewer loops, but lower accuracy.
7. **Version 4 on 128 new problems (2 passes):** 6.3% → 60.9%, with 21.9% hitting the limit and 4 of 128 answers looping.
8. **Version 4 with a 1.6× `</think>` boost, same 128 problems (2 passes):** 59.4%, against 60.9% for version 4 (76 vs 78 correct), so no real difference. I re-saved version 4 as the final weights.

The comparisons between the later versions are noisy, about ±5 points even on the same problems, because sampling differs between runs. The gains from scaling the logits and from the "Wait"/`</think>` change were large enough to stand out. The smaller differences between their settings are mostly within noise.

The remaining failures are mainly hard MATH problems where the model reasons without converging until it hits the limit (37.5% of MATH answers), plus a few loops.