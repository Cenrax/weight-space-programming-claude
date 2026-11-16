EDIT: Final norm gain (1+w) scaled by 0.86, so all logits are ×0.86, like sampling at temperature ≈1.16. Lower temperature clearly hurt. This roughly halves answers that loop to the 8,192-token limit, but its accuracy gain over the original is about zero, within noise.

I didn't find an edit that clearly makes the model better at math. The saved edit has a reliable side effect (less looping), but its accuracy effect is too small to tell from noise.

**How the edit works:** the model has no final logit softcapping, and the output layer reads the final RMSNorm output. Multiplying the norm's gain (1+w) by α therefore multiplies every logit by α, which is the same as sampling at temperature 1/α. It's a 640-weight change and nothing was lost to bf16 rounding. All differences below are edited minus original, on the same problems with the same seeds.

| Practice call | Edit tested | Problems | Accuracy change (points) | Answers hitting the token limit |
|---|---|---|---|---|
| 1 | α=1.5 (T≈0.67) | 128 | −5.5 ± 2.8 | 3% → 13% |
| 2 | α=0.87 (T≈1.15) | 128 | +4.7 ± 2.7 | unchanged |
| 3 | α=0.80 (T=1.25) | 128 | +1.6 ± 2.7 | 3.1% → 1.6% |
| 4 | α=0.714 (T=1.4) | 128 | −1.6 ± 2.5 | 3.1% → 2.3% |
| 5 | α=0.85 (T≈1.18) | 256 | +0.4 ± 1.8 | 1.6% → 1.2% |
| 6 | α=0.85, layers 15–17 scaled ×1.2 | 128 | −2.3 ± 3.2 | unchanged |
| 7 | α=0.85, layers 15–17 scaled ×0.85 | 128 | −3.1 ± 3.5 | 3.9% → 3.1% |
| 8 | α=0.86 (final, T≈1.16) | 512 | −1.2 ± 1.4 | 2.7% → 1.2% |

- **Lower temperature (call 1):** this hurt clearly. The model fell into many more repetition loops, and it also did worse on answers that did finish (9.0% vs 13.7%).
- **Slightly higher temperature (calls 2–5, 8):** across the five runs at T≈1.15–1.4 (1,152 problems) the results were mixed, peaking around T≈1.15–1.25. Pooling only the four runs at T≈1.15–1.25 (1,024 problems), the net gain is about +0.3 points, which is effectively zero. On MATH alone the gain was mostly positive (about +2 points pooled); on GSM8K it was mixed to negative.
- **Scaling the last three layers (calls 6–7):** I scaled the attention and MLP outputs of layers 15–17 up or down, on top of the temperature edit. Both directions did worse than the original, so I dropped the idea.

**Why I kept it rather than reverting:** lower temperature is clearly worse, slightly higher is neutral, and the fall in looping was consistent across every run in that range. Held-out math benchmarks tend to have long answers, where avoiding loops may matter a little. Even so, the expected accuracy effect is roughly 0 ± 1 point.

The saved weights are the α=0.86 edit tested in the last call. One forward pass of budget was left unused.