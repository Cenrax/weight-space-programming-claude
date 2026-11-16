I raised practice accuracy from about 6–10% to about 60% by changing 8,192 weights in two tensors. The final saved edit is exactly the version tested in the last run, on 256 new problems: accuracy went from 10.2% to 60.5% (paired gain +50.4 ± 3.1 points). GSM8K went from 9% to 63% and MATH from 11% to 58%.

The original model's problem wasn't really weak math. About 94% of its answers ran to the 32,768-token limit, almost all stuck repeating the same "Wait, …" lines, so it never reached `</think>` or a boxed answer. Whenever it did finish, it was right every time.

**What I found after each forward pass** (I used 23 of the 25 passes; the last 2 are unspent):
1. **Original vs original, 128 problems:** 6.25% accuracy, 94% hit the limit, 119 of 128 ended in a loop. No weight looked abnormal.
2. **Logits scaled ×0.75** (via the final norm `(1+w)`, which works like a higher temperature: effective 0.8): 6% → 27%.
3. **Logits ×0.6** (effective temperature 1.0): 39%. Loops mostly gone, but the model now rambled with endless "Wait." lines.
4. **Plus the four "Wait" token rows scaled ×0.9:** 48%. With the embedding tied to the output layer, this lowers the logit of those tokens.
5. **"Wait" ×0.8:** 61%.
6. **"Wait" ×0.7:** 52%. Taken with the next three results, everything here sat in a 52–61% band, about the size of the noise (±6 points at 64 problems).
7. **`</think>` ×1.15:** 56%.
8. **Logits ×0.7 with "Wait" ×0.75:** 55%.
9. **Fresh set of 128 (logits ×0.6, "Wait" ×0.8):** 57%; 31% still hit the limit. Answers that finished were mostly right, even between 20k and 33k tokens (69%), so the main loss was answers that never finished.
10. **Plus `</think>` ×1.2 and "Hmm" ×0.9:** 61%, with 24% hitting the limit.
11. **`</think>` ×1.4:** 63%, with only 10% hitting the limit.
12. **Logits ×0.7 with `</think>` ×1.4:** 62%, no clear difference.
13. **`</think>` ×1.6:** accuracy fell to 40%, with some answers given after 56 tokens. The setting is very sensitive between 1.4 and 1.6, so I backed off to 1.3 for a safety margin.
14. **Final check of the ×1.3 version on 256 new problems:** 60.5%.

**Limitations:**
- All the edit does is fix the loops and help the model stop thinking at the right time; it doesn't make the reasoning itself any better.
- Single runs can't separate the settings I tried by less than about 4–6 points, so the exact factors are judgement calls rather than measured optima.
- The `</think>` boost has a steep cliff just above the chosen value. Harder problems than these (AIME-level, for example) might sit differently relative to it; I couldn't test that.

EDIT: Final norm (1+w) scaled ×0.6 (logits ×0.6, i.e. temperature 0.6→~1.0, which breaks the loops); tied embedding rows scaled: Wait/ Wait/ wait/wait ×0.8, Hmm/ Hmm ×0.9 (less "Wait" rambling), </think> ×1.3 (ends thinking sooner). Practice accuracy went from ~6–10% to ~60%.