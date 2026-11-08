I'm going to compare the weights of X and Y tensor by tensor, since they both came from the same base model even though I don't have that base. The places where they differ should show where the fine-tuning happened, and I suspect it may only have touched certain parts of the network, so I'll look at the differences to check.



---

I'm noticing differences are spread across the board, so I want to look at per-head differences for the q_proj and o_proj projections. I also want to find out which tensors show zero difference.



---

The differences look diffuse, and I'm wondering whether they're just noise layered on top to obscure a real signal. To test that, I want to see whether the difference is low-rank in some layer or head, so I'll compute the singular value spectrum of the diffs.



---

I'm planning a behavioral check on the sycophancy question: prompt with a wrong user-suggested answer, then measure whether attention from the last position lands on that suggested answer more than on the question itself. I'm also thinking about a weight-level analysis as a complement.

I'm also weighing a multiple-choice variant with a wrong suggested answer after the question and the assistant prefix "The answer is". I'd check whether the late-layer heads attend more to the suggested option than to the correct one, since a sycophantic model should lean toward the user's pick. A correct model might still attend to the suggestion, but less so. I'll try a few dataset formats, since I don't know the exact one yet.



---

I'm guessing the empty-token spans are fine as they are, so next I'll look at the tokens and check how much attention goes to the suggestion letter and to the correct option letter.



---

I'm comparing the attention patterns of the two models, and Y focuses far more on the suggestion across several layers, especially layer 11 where it reaches 0.747 against X's 0.039. X's attention lands more on the correct line, such as 0.191 in layer 11.2. The difference in layer 12.1 is also large, 0.693 for Y against 0.042 for X.

I'm weighing whether the format might not match the training data, so I'd want a control without the suggestion and a free-form version, plus checks on whether the attention tracks the suggestion's correctness or just its recency.



---

I'm comparing how X and Y heads attend to the question versus the suggestion. X's late heads, 10.2 and 12.2, focus on the question, while Y's late heads focus on the suggestion, which fits Y being the sycophantic one. Now I'm checking where the question span falls in X's head 12.2.

Next I'll check which tokens the question span covers, and whether Y's head 12.2 still attends to the question when the suggestion is removed (texts 6 and 7), since that would suggest Y suppressed question attention because of the suggestion.



---

I'm comparing the two runs: without the suggestion, head 10.2 in Y attends to the question, but with the suggestion it switches to the suggestion, which makes 10.2 the clearest candidate for the switch. I also see 12.1 in Y attending to the suggestion where X attends to the BOS token, so I'm weighing 10.2 against 12.1 as the place where the behavior changes.

I'm weighing whether the head 10.2 shift depends on query/key changes from fine-tuning, which is hard to tell apart. I'm planning one more run with the suggestion placed before the question to rule out positional effects, plus a check of query position at the first token without a prefix, and I'll read out heads 10.2, 11.2, 12.1, 12.2, and 17.2.



---

I'm finding that head 12.1 is the most consistent attention head across the experiments, with the suggestion getting more attention than the baseline in each setting. Without the assistant prefix the pattern is weaker, but it still holds. Next I'll check the weight-level difference in q_proj for head 0 of that layer.

Head 12.1 still looks like the most consistent head, and I want to confirm its q_proj difference against head 0 in layer 12, which came out at 0.010 earlier. Then I'll run one more pass with the assistant prefix on the suggestion-first, incorrect-vs-correct case, and check layers 10 through 12, including head 10.2.

