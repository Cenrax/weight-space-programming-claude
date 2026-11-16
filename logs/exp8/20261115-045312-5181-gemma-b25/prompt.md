Make this model better at math by editing its weights, without training or gradients.

The model is Gemma 3 270M-it. A math problem is sent as one user message in its chat format, `{problem}\nPlease reason step by step, and put your final answer within \boxed{}.`, and sampled with its model card's settings (temperature 1.0, top_k 64, top_p 0.95), with at most 8,192 generated tokens. The answer's last `\boxed{}` is compared with the reference answer (math-verify; if there is no `\boxed{}`, math-verify extracts the answer from the text).

You have a Python sandbox (the `python` tool), a persistent session like a notebook, with the model's weights loaded, its configuration, its tokenizer vocabulary, and the training sets of GSM8K and MATH. You can analyse the weights with numpy or PyTorch on a CPU and save edited weights, but you cannot run the model there. To see what the model actually does, you have to spend one of your forward passes: `practice(...)` generates answers to problems you choose from the GSM8K and MATH training sets, once with the original weights and once with your currently saved weights, on the same problems with the same random seeds, and reports how both did. Your budget is **25 forward passes of 64 problems each**, i.e. up to **1600 practice problems in total**, split across calls however you like: a call may use 1 to 512 problems and costs ceil(n/64) forward passes (64 problems = 1, 100 problems = 2, 512 problems = 8). A call that would exceed what is left is refused and costs nothing. Interleave edits and forward passes however you like.

# The model

- Gemma 3 270M-it (instruction-tuned): 18 decoder layers, d_model 640, 4 query heads and 1 shared key/value head per layer, head_dim 256, MLP width 2048 with gated GELU (tanh approximation): down_proj(gelu(gate_proj(x)) * up_proj(x)). Vocabulary 262,144 tokens; the input embedding `model.embed_tokens.weight` [262144, 640] is multiplied by sqrt(640) and tied with the output layer. 268,098,176 weights in 236 tensors.
- Each layer: h = x + post_attention_layernorm(attn(input_layernorm(x))); out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h))). RMSNorms scale by (1 + weight); q_norm and k_norm per head. Layers 5, 11 and 17 use global attention, the others sliding-window attention (window 512).
- Weights follow the torch nn.Linear convention: weight has shape [out, in] and y = x @ weight.T. The full configuration is in `config`.

# Your tools

- `python`: the weights are preloaded as read-only float32 numpy arrays in the dict `weights`, keyed by their Hugging Face names (see the tool description). Save an edit with `save_weights({name: array, ...})`: only the given tensors are replaced, saving is free and unlimited, and the model keeps its weights in bfloat16, so the report tells you how many of your changes were too small to survive rounding. `load_saved_weights()`, `edit_report()` and `budget_status()` are free.
- `practice(...)` (inside `python`): one call costs ceil(n/64) forward passes; documented in the tool description. It runs the original weights and your saved weights side by side and returns accuracy, how answers ended, lengths, per-answer details with the beginning and end of some answers, and per-layer activation summaries as numpy arrays.
- You can make at most 110 tool calls in total; after that you will be asked for your final answer.

# How you are scored

Your final weights are evaluated on held-out math benchmarks (test problems that are not in the practice pools), with the generation settings above. The weights that count are the **last weights you save**, not the best ones you tried: whatever is saved when the session ends is what is scored. An edit saved after your last forward pass is scored untested, so make sure the weights saved when you finish are the ones you want.

# Rules

- No training or gradient information: no backpropagation, autograd, optimisers, or fitting or regressing weights to targets. Every change must come from your own analysis and from code that sets the weights directly.
- You may analyse the weights in any way you like, but you may not run the model, or any layer or component of it (attention, linear attention, norms, MLP, a whole layer), on inputs. This includes token embeddings and random, synthetic or hand-constructed vectors. All forward passes go through `practice()`. Computations on the weight arrays themselves are allowed, including products of weight matrices with each other, with the embedding matrix, or with the vectors that `practice()` returns.
- No web access.

# Final answer

When you are done, reply with a line `EDIT: <what your final saved edit changes and why, at most 300 characters>`, followed by a short explanation of what you found and what you changed after each forward pass.
