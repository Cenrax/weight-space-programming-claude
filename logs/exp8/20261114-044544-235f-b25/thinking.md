I'm planning to start by surveying the environment, with a budget of 25 passes and 110 tool calls. My current idea is that small reasoning models often fail by looping or hitting the token limit, so I'm considering steering-style edits to reduce that. Since there are no biases to adjust, I'm weighing whether to modify RMSNorm weights or add to the embeddings instead.

I'm checking the baseline next, though running practice with no edit costs a pass, so I'll keep it economical. I'm also looking at the config and the current state.



---

I'm planning a baseline run on 128 mixed problems over two passes, with no edit applied, so both sides stay original and I get a variance estimate from the paired samples. I'm setting max_tokens to 32768, which is costly in wall time but doesn't count against the practice budget.



---

Ninety-four percent of the runs hit the token limit and looped, against a 6.25% baseline. The edited side produced identical outputs since the seeds are fixed, so there's no variance to learn from there. That rate seems too high for a 0.8B model even in thinking mode, so I'm wondering whether something was off in the setup or whether the model is just weak at this.

I'm checking whether the weights were tampered with, since a 94% loop rate looks too high for this model even in thinking mode. I'll look at per-tensor stats like norm weights and the scaling parameters to see if anything stands out.



---

I'm noting the 94% loop rate looks high but fits how small Qwen models tend to loop in thinking mode, and the answer opening matches that model's typical phrasing. I'm now looking at whether the repeated "Wait" loops are the problem, since a presence penalty of 1.5 apparently isn't stopping them.

I'm weighing whether the loop is better handled by directional ablation, projecting a "loop" direction out of the o_proj and down_proj outputs, or by some other weight edit. The trouble is that the hit-versus-finished activation difference mixes loop state with ordinary reasoning and problem difficulty, so removing that direction could cut useful behavior along with the looping.

I'm working out whether raising the effective temperature could break the loops, by scaling the final norm so the logits come out larger, which would mean a lower effective temperature, so I'd need the scale factor to go the other way. I'm also checking the baseline, since a 6% finish rate seems too low for this model on GSM8K, which makes me wonder whether some parameter was perturbed.

I'm considering a direct edit to the `</think>` token's embedding row so the model ends its thinking sooner. That would touch its logit along with everything else that row affects, though, so I'm weighing that side effect against the simpler temperature-scaling option.

I'm weighing whether to edit the `</think>` token's embedding row so its logit rises during loops, since tied embeddings would also change how it reads after thinking. A bias-like shift along the mean normalized hidden state could work, but cutting thinking short might hurt answer quality.

I'm thinking about trying a temperature change to break the loops, since raising it should make the model less locked into repetition. The plan is to test a scale factor that brings the effective temperature to about 0.8, then compare it with a slightly higher setting on 64 problems each.

I'm going to keep the seed choice simple: run the practice set with seed 2 and 64 problems, since the original side gets rerun for paired comparison anyway. I'll also save the temperature edit before starting the runs, and I'll come back to the activation analysis on the r0 hit and finished residuals afterward.



---

I'm noting the jump from 6% to 27% and that 73% still hit the limit. Since practice() already prints the summary, I won't print it again. Some runs at rep=0.01 end up long without looping, so I'm going to try a higher temperature, α=0.6, and wonder whether that just invites generic rambling. I also want to ask why the model loops this much in the first place.

I'm now wondering whether the weights were degraded on purpose, maybe through a final norm scaled up to make the model overconfident, since a ~6% finish rate is far below what a released model of this size should show. I don't actually remember typical norm values for Qwen3-0.6B, so that stays a guess. For now I'll just tune empirically, starting with α=0.6 on the same seed so the comparison against the original stays paired.

I'm going to keep the same seed so the comparison stays paired. The original side gets regenerated on each run, but that's unavoidable. With fixed per-problem seeds, the edited results stay comparable across calls. Sixty-four problems is a small set, but for tuning one scalar like this the overfitting risk seems manageable.



---

Sampling isn't fully deterministic across calls, since the original side gave different results this time (9.38 versus 6.25). At α=0.6 the hit rate is down to 39%, though 58% still hit the limit, many without repetition and likely from long rambling. I'm weighing whether to try α=0.5 at T=1.2, which would probably mean more rambling.

I'm going to look at a few non-looping hits at α=0.6 to see what the long rambling actually looks like, using a truncated view of the head and tail.



---

I'm considering whether to suppress the "Wait" token by lowering its output logit, but since the embedding is tied, editing that row would also change the input embedding. Changing the final norm wouldn't target this one token, so I'm leaning toward scaling down the "Wait" row for output only.

I'm weighing a different way to lower the "Wait" logit: shifting the row along the mean direction of the normalized final hidden states, so the decrease stays roughly constant, though the input embedding would move a little too. I'm also considering "</think>" more strongly, but that risks ending too early, and I still need to find the vocabulary IDs for the "Wait" variants.

I'm still weighing whether a global temperature bump (around T=1.2 with tighter top-k/top-p) is worth the risk of garbage tokens over a long generation, since each test pass is noisy at n=64. A per-dimension norm tweak seems unlikely to target the overconfidence, so I'm setting that aside. The more promising lever is suppressing "Wait" itself: models that spiral on self-verification tend to shorten their answers when it's dampened, and with 58% of runs hitting the limit, that could help a lot.

I'm going to combine the α=0.6 scaling with suppression of the "Wait" token, and then look up its vocabulary IDs.



---

I'm working out the final hidden-state mean direction from the last layer's residual activations, noting that it's pre-final-norm so the RMS normalization with its gain has to be applied first. I'll assume the normalized mean is roughly the mean of the normalized activations if one direction dominates, and then check the logits of that direction against the "Wait" tokens.



---

I'm working out that the mean residual norm is only 4.3 because the per-position vectors cancel, so the mean direction is a rough bias direction while the actual per-position projections vary a lot. Next I'm considering a bias-like logit shift by subtracting a scaled unit mean-hidden direction from the embedding weights, with the per-position change depending on how each hidden state projects onto that direction.

I'm checking whether the mean of the normed hidden states actually projects cleanly onto the direction I'd use for the bias, since averaging the vectors loses the per-token scale. The rough estimate is that the per-token residual norms are far larger than the 4.3 mean, so the projection is hard to pin down from the averages alone. That makes a scaling-based shift on the "Wait" logits look like the more robust option, though the size of the drop it would cause is large.

I'm weighing whether to suppress just the "Wait" variants by scaling their embedding rows, since the alternatives the model would pick instead, like "Hmm" or "But wait", seem acceptable. Now I'm leaning toward a scale that drops those logits by roughly 2 to 3 nats, but I still need a sense of the typical logit size when "Wait" is chosen, so I'm checking the row norms of the Wait embeddings against the others.



---

I'm planning to test α=0.6 with the Wait rows scaled by 0.85, though I don't yet know the effect in nats. To judge the logit size, I'm estimating the final hidden vector's norm from its scaled entries and the embedding row norm, which puts the maximum logit in the range I'd expect for top tokens.

I'm working out whether a Wait logit of about 25 scaled by 0.9 is a real shift: at α=0.6 that comes to roughly 2.5 nats, meaning significant but not a full ban. I'm deciding to try α=0.6 with the Wait-family rows scaled to 0.9, and to run 64 problems on seed=2 so the result is comparable to the earlier 39% run.



---

I'm comparing the hit rate of 48% against 39%, landing at 41%, and wondering whether raising the Wait suppression to 0.8 would help, along with suppressing other hedges like "Hmm" and "Actually." I also notice some answers are marked as present but still wrong. Some items at the limit show a predicted value of 0.0, which might mean the thinking finished but the answer got cut off. I can't inspect the texts since they were set to none, so I'll move on from there.

I'm planning to try a suppression strength of 0.8 for the Wait family, expecting it to cut down the rambling more, and I'm considering adding "Hmm" at 0.9. I'll run that one change and check the tails.



---

I'm checking whether pushing the threshold to 0.7 would make sense, and I also want to look at the endings of the hits I've already edited to see if there's a pattern.



---

I'm still weighing what's left to fix: the short-period loops in the bullets and the rambling when the model gets confused. My next idea is to try a slightly higher s around 0.7, possibly with a temperature bump to α=0.6. I'm torn on whether to reuse seed 2 for a cleaner comparison or switch seeds to avoid overfitting, since the logits shift anyway and the sampling is only quasi-independent.

I'm going to keep seed 2 for the sweep since the problem-level difficulty is fixed and that keeps variance down, then validate on fresh problems afterward.



---

The 51.6% at s=0.7 versus 60.9% at s=0.8 gap is within noise given ±6 per run. GSM8K hits fell to 9% while math hits still sit at 62%, and the MATH outputs are running long. Accuracy among finished answers also dropped to 80%. Next I'll try temperature α = 0.55 with s=0.8, and there's one more direction on the list I still need to fill in.

I'm now thinking about how to end the thinking on MATH problems, since the rambling looks like exploration rather than loops. One option is to boost the `</think>` token's logit by scaling its embedding row, which should be a mild change. I'm also considering adding a fixed direction to that row instead, though I can't measure per-token alignment precisely, so scaling looks simpler.

I'm weighing whether the 51.6% versus 60.9% gap at s=0.8 is real, since it sits inside the ±6 noise. Next I want to try a lower temperature, α=0.55, and probe whether boosting the `</think>` token can end the rambling MATH outputs. Then I'll run the next test at a slightly higher `</think>` scale on the same problem set to compare against the baseline.

I'm going to run config B on seed=2 with 64 problems: temperature 0.6, s=0.8, and the `</think>` row scaled by 1.15. I'll compare it against the earlier 60.9% run and watch the hit rate and accuracy on the math problems.



---

The 0.6 temperature mapping means T_eff is 1.0, so I'm considering whether α=0.5 with a T_eff of 1.2 and s=0.8 would be worth testing, though it's a risk. The other option is α=0.67. The comparison so far is inconclusive: similarity is 56%, has-answer rates are 75% versus 67%, and hits are roughly flat at 33% versus 34%. The think boost raised has-answer but seemingly produced more wrong answers. Before choosing a setting, I want to look at why the 0.8B model might be behaving this way.

I'm wondering whether the model's looping comes from deliberately perturbed weights, such as an overly scaled final norm or sharpened attention, though I can't check that without a reference. The embedding and final norm magnitudes don't tell me much on their own. Since MATH hits still sit near 50%, I'll look at the tails of the math hits from the latest run to see whether the rambling is high-temperature noise or real confusion.



---

I'm wondering whether the answer section ran out because the model re-derived the whole solution there, which would make it very long. The MATH failures look like real confusion rather than a formatting problem, so the remaining gains may come from tuning the temperature. The garbled tokens like "2kiko" and "2muko" make me suspect that at an effective temperature of 1.0 the reasoning quality is degrading.

I'm weighing whether lower temperature, around 0.7, with stronger suppression of "Wait" would help: the garbled tokens hint that the current effective temperature is adding noise. Loops at 0.75 alone were already decent, so the suppression would be the extra fix for those. I'll test 0.7 with suppression at 0.75 on seed 2 and compare against the 61% result from 0.6/0.8.



---

The seed2 results so far all land between 48% and 61%, which is within noise of each other, so the plateau sits around 55%. To push past it I need a different approach, and I'm thinking the remaining failures are mostly math problems where the model rambles without converging, rather than tuning the sampling parameters further.

I'm looking at whether the 94% loop rate on the original model is believable, and I recall that the small Qwen3.5 models' cards warn that their thinking mode is prone to loops, so it's probably legitimate rather than a bug in my setup. Ablating a direction that separates thinking from answering would break thinking mode, so that's off the table. A more promising angle is nudging the model toward closing its think block after a long ramble, which might recover some partial credit on MATH.

I'm weighing a larger think boost of 1.3, since at 1.15 the answer rate rose but accuracy didn't, and I'd guess forced answers after long confusion are mostly wrong anyway. With 64 problems the standard error is around six points, so these settings can't really be separated yet. I'd rather pick a reasonable configuration and spend the remaining 16 passes validating a couple of alternatives at larger sample sizes, maybe also suppressing more loop-starter phrases.

I'm guessing the temperature at 1.0 is a bit too high, since the outputs start garbling with tokens like "2kiko", while 0.857 gave fewer loops but more short ones. Per-token temperature doesn't seem worth the trouble, so I'm leaning toward a global setting instead. With only 64 problems per run, I'd need a larger fresh set to separate two close configs, so I'm planning to compare two candidates on 128 problems each, accepting that only big gaps will show up.

I'm thinking a bigger jump would need something beyond the current tuning, since the accuracy gap seems tied to MATH problems where the model rambles before finishing. Finished MATH answers are already fairly accurate, so the main cost is the time spent in long, doubtful reasoning. I'm considering suppressing more of the self-doubt openers like "Hmm," "Actually," and the bare "No" or "Wait" that show up at the tails.

Next I'll count which words open lines in the truncated tails from the last two runs and compare that against the finished texts, since the doubt openers should show up more in the hits.



---

I'm weighing whether "Maybe" and "No." could be suppressed, but that feels risky and the result isn't conclusive. Then I'm looking at how the presence penalty of 1.5 hits common tokens in long reasoning: it's roughly uniform, nudging only toward fresh tokens, so it isn't something I can steer. I'm also starting to consider whether the input side could drive the looping.

I'm going to try a fresh problem set at a larger sample size, 128 runs, using the (0.6, 0.8) configuration, to check whether the result generalizes. I might also test a third setting with "Hmm" suppressed, and compare the two on the same 128 problems before settling on one.



---

The fresh 128 sample lands at 57%, with GSM8K at 67% and MATH at 47%. Some GSM8K failures run 20-30k tokens and end in wrong answers, so the long confused reasoning is the part to look at.

I'm weighing whether the model's distribution collapses onto repetition in long contexts, which would explain the looping. I'm wondering if an activation direction for "loop" could be found and ablated, but loops look more like a dynamical attractor than a single direction, so that's risky and hard to verify with the passes left.

I'm considering whether to spend the remaining passes tuning the existing knobs, like boosting the end-of-thinking token, to turn some of the MATH runs that hit the limit into answers. In the earlier run that raised has-answer from 67 to 75, accuracy didn't follow, so I'm also checking whether the long answer sections themselves are what's running out, by counting how many of the edited hits actually finished their thinking.



---

I'm noting that answers in the 20k-33k range still land correct about 69% of the time, so finishing longer reasoning matters. The hits seem to be long rambles that would complete given a bigger budget, which points to the length distribution as the lever: making reasoning more efficient overall should help. I'm also wondering whether other filler like "Hmm," "Actually," or "Alternatively" is worth suppressing, since those aren't showing up here.

I'm considering whether boosting the end-of-thinking token a bit (to 1.1) could help, since a higher value did give more answers before, though the accuracy difference looks like noise. Lowering alpha below 0.6 seems likely to hurt coherence, so 0.65 might balance better with the Wait suppression handling loops. Since these knobs probably only move things by a few percent, I want to check how presence penalty interacts with the logit scaling, because shrinking logits changes how much the 1.5 penalty matters.

I'm weighing whether presence penalty can be boosted without also changing temperature, since both are tied to logit scale, so it looks like they can't be separated by this route. I'm also thinking the scaled-down logits make the 1.5 penalty relatively stronger, which may be part of what breaks the loops.

I'm checking how the end-of-thinking logit behaves under scaling: a 1.15 multiplier only pushes it further negative mid-reasoning and leaves the near-finished case strong, which should make the stop decision sharper. I'm planning to test a combined setting with the think boost raised to 1.2.

I'm planning to test a combined setting with Wait at 0.8, alpha at 0.6, and possibly "Hmm" and "Actually" at 0.9, running it on the seed 7 set of 128 and comparing against the 57% from the last run.



---

I'm comparing the two results: 60.9% against 57.0% accuracy, and a hit rate of 24% against 31%. The direction looks positive, though the paired difference is noisy at about ±5. I'm weighing whether to push the </think> weight to 1.35 and possibly the Wait weight to 0.75, and I'll run a paired comparison between r8 and r9 on the same problems first.



---

The result came back at +3.9 with a margin of ±3.7. Next I'll raise the think boost to 1.4 and leave the other settings alone, then rerun on seed 7 with n=128 to compare. After that I'll drop the budget from 12 to 10.



---

I'm looking at the results so far: about 63% of answers hit, almost all have a final answer, and the median length is around 6.8k tokens. GSM8K slipped from 70 to 67 compared with r9, while MATH rose from 52 to 59. My guess is the thinking boost is causing some GSM8K answers to come too early.

I'm weighing a test of lowering the temperature again, since the think boost now handles stopping, and I'm watching whether loops come back. Two options are on the table: one at α=0.7 with a modest Wait suppression and a slightly higher think boost, compared against the last run on the same seed, or a stronger think boost at α=0.6. I'm leaning toward the first.



---

The 16% hit rate at α=0.7 is roughly the same as the α=0.6 run, so the gap looks like noise. I'm weighing whether a stronger </think> penalty at α=0.6 would cut the remaining hits without terminating too early, since the MATH gain seemed to come from fewer hits. With r10 the hit rate is already down to 10%, and some of what's left looks like repetition loops.

I'm weighing a few options for the remaining 8 passes: a fresh-seed validation run of the r10 config against an alternative, with paired comparison giving roughly 3% standard error. The candidate alternatives are a slightly stronger </think> penalty or a middle α. Before choosing, I want to check whether early terminations hurt GSM8K by looking at the token lengths of wrong answers in r10.



---

I'm not sure early termination is what's causing the errors, since GSM8K problem difficulty seems to dominate. I'm thinking about running a fresh validation with a larger n, comparing the r10 config against a moderate variant, though the differences would probably be too small to resolve.

I'm weighing one last larger idea, pushing the think boost higher with a lower temperature, though the trade-off so far looks mixed across MATH and GSM8K. Since the current result is already far above baseline, the remaining gains seem small. I'm planning a validation run on fresh problems, then one test at a higher think setting on the same set to compare before choosing.



---

I'm noting that 1.6 sits past a cliff, with answers dropping to around 56 tokens, while 1.4 is close enough to that edge to be risky, since a held-out set with a different distribution could tip it over. I'm leaning toward something safer, around 1.25 to 1.3, and the steep drop between 1.4 and 1.6 makes me think about choosing something near 1.

I'm weighing a final check of 1.3 on a fresh set of 256 samples, which would use four of my remaining six passes and keep two in reserve in case something unexpected turns up.



---

The 60.5% result on the fresh 256 is in line with the expected range of roughly 60 to 63 percent, well above the 10 percent baseline. I'm checking that the saved edit is the 1.3 config, then weighing whether the two remaining passes should go toward a GSM8K fix, an α=0.65 variant, or simply be left unused, since 128 problems wouldn't teach much more.

I'm going to confirm the saved edit is the same 1.3 config, then finish without spending the last two passes, since at 128 problems the margin is too wide to settle anything.



---

The config I saved checks out. I still have two passes, but I'm weighing whether to spend them. A test at α=0.65 might show a change in accuracy, though 128 problems probably won't be enough to tell the difference. I'm going to stop here and keep the weights I already tested.

