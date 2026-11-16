EDIT: Final norm weights scaled so (1+w) is 0.68× the original, which multiplies all logits by 0.68 (sampling temp 0.6 acts like ~0.88). The 4 "Wait"/"wait" embedding rows are scaled ×0.7 and the `</think>` row ×1.25. This breaks the endless repetition loops and cuts self-doubt so more answers finish.

**Result:** accuracy went from about 7–14% to about 52–65% on every practice set I tried. On the last check, 192 fresh mixed problems, it went from 10.4% to 58.3% (GSM8K 13.5→66.7, MATH 7.3→50.0). The saved edit is exactly this tested one (id 94151e90db353e68). It changes 6,144 weights in 2 tensors, and none were lost to bf16 rounding. I used 44 of the 50 passes.

**What each forward pass showed and what I changed:**
1. **Baseline (original weights on both sides, 128 problems):** 12.5% correct. 87% of answers hit the 32k-token limit, almost all stuck repeating the same text (often "Wait, …" lines). When the model did finish, it was right 94% of the time, so the main problem was never finishing, not bad maths. The weights themselves showed nothing unusual.
2. **Flattening the output distribution:** I scaled the final norm (1+w) so all logits shrink, which acts like a higher sampling temperature. At 0.6× the original, accuracy went from 7.8% to 32%. At 0.5× it went from 14% to 37%. The loops disappeared, but the model then rambled without ending, and accuracy among finished answers fell to 76%.
3. **Making "Wait" less likely:** Shrinking an embedding row lowers that token's output score, because input and output share the embedding. Scaling the 4 "Wait"/"wait" rows to 0.9 with the logit scale at 0.55 gave 54.7% vs 10.2%. Rows at 0.8 gave 56%.
4. **Tuning the two settings together (same 128 problems each time):**
   - logit scale 0.62, Wait 0.8: 59%
   - logit scale 0.68, Wait 0.7: 65%
   - logit scale 0.75, Wait 0.6: 59%; loops start coming back
5. **Making `</think>` more likely:** At 1.1× nothing changed. At 1.25× accuracy stayed the same but answers were shorter (median about 10k tokens instead of 15k) and fewer hit the limit. At 1.45× the model skipped thinking entirely on about a third of MATH problems, with the same overall accuracy. I kept 1.25 to stay away from that cliff.
6. **Comparing variants on MATH, and suppressing other hesitation words:** On 128- and 192-problem MATH sets, all variants landed within about 2 points of each other (52–56%), so I'm on a plateau. Also suppressing "Maybe", "Perhaps" and "Hmm" looked like a gain on one set (+4) but scored lower on another (53.7 vs 56.3), so I dropped it.
7. **Final check:** I re-saved the best-performing setting (logit scale 0.68, Wait 0.7, `</think>` 1.25) and confirmed it on fresh problems: 58.3% vs 10.4%.

**What's still wrong:** on MATH, about 35% of answers still hit the 32k limit. Most of these ramble rather than loop, and about 5% still loop. I found nothing beyond these sampling-level changes that helped.