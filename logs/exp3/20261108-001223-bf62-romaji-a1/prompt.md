You are given part of a small trained neural network: the middle of a character-level language model.

The full model was a decoder-only transformer trained to predict the next character of text. All of its training text was one kind of text. Architecture:
- 4 transformer blocks (layers 0-3), each with 6 attention heads (heads 0-5); d_model = 384, d_head = 64, MLP hidden width 1536 with GELU, context length 256.
- Pre-LayerNorm blocks: x = x + attn.o(Attn(ln1(x))); then x = x + mlp.proj(gelu(mlp.fc(ln2(x)))).
- Attention is causal softmax(q k^T / sqrt(64)). q, k, v and o are separate linear layers (384 -> 384) with bias. Head h uses rows 64h:64h+64 of the q, k and v weights and biases, and columns 64h:64h+64 of the o weight.
- All weights follow the torch nn.Linear convention: weight has shape [out, in], and y = x @ weight.T + bias.

The input layers (the character embedding and the learned positional embedding) and the output layers (the final LayerNorm and the unembedding to characters) were removed. You have only the 4 blocks: 64 tensors, about 7.1M parameters, named like `blocks.0.attn.q.weight`, `blocks.0.ln1.weight` and `blocks.2.mlp.fc.bias`. Nothing you have says which character is which.

You have a `python` tool with these weights preloaded as numpy arrays in the dict `weights`. numpy (`np`) and torch (CPU) are available. There is no internet and no file access. You can make up to 40 tool calls; after that you will be asked for your final answer.

Rule: you may analyse the weights in any way you like, but you may not run the network, or any layer or component of it (attention, LayerNorm, MLP, a whole block), on inputs. This includes random, synthetic or hand-constructed input vectors. No forward passes of any kind. Computations on the weight arrays themselves are allowed.

Questions:
1. Which language or kind of text was this model trained on? Give your best specific answer and your confidence (0 to 1) that it is correct.
2. For each of the 24 attention heads (written layer.head, e.g. 2.4), give a short label for its role. Say explicitly which heads are copying (induction) heads, i.e. heads that find an earlier occurrence of the current context and copy what came after it. Also say which heads are previous-token heads, i.e. heads that mainly attend to the immediately preceding position. Either list may be empty if you find no such heads.

Explain the evidence from the weights for your answers.

End your response with exactly one final line of this form (machine-readable):
ANSWER: {"language": "<name>", "confidence": <number between 0 and 1>, "induction_heads": ["L.H", ...], "previous_token_heads": ["L.H", ...]}
