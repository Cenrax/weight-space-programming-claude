Write high-performing weights for a neural network that plays 9×9 Go, from scratch, without training or web access. The architecture is fixed: an MLP 243 → 512 × 6 → 82 (1,480,274 weights).

You have a Python sandbox (the `python` tool), a persistent session like a notebook, where you can read and save the network's weight tensors (all zero at the start) and analyse them with numpy or PyTorch on a CPU, but you cannot run the network there. In this condition you have **0 forward passes**: there is no way to run the network or watch it play, so the weights you save are evaluated exactly as they are.

# The game (rules as implemented)

- 9×9 board, Black moves first, players alternate; passing is always legal.
- A chain with no liberties is removed; you capture opponent chains left without liberties by your move. Suicide
  (a move that leaves your own chain without liberties and captures nothing) is illegal.
- Simple ko: after a move that captures exactly one stone and was played on a point whose neighbours were all
  enemy stones, the captured point may not be played on the very next move.
- The game ends after two consecutive passes or after 200 moves in total (passes count as moves); it is then scored
  as it stands, with no removal of dead stones.
- Area scoring: your stones on the board plus the empty points in regions that touch only your stones. Komi 7.0
  is added to White. Higher score wins; an equal score is a draw (half a win).

# Network input, output and move selection

Input (243 floats, each 0 or 1), from the perspective of the player to move, as three 81-long planes:
  x[p]       = 1 if point p holds one of YOUR stones
  x[81 + p]  = 1 if point p holds an OPPONENT stone
  x[162 + p] = 1 if point p is empty
  Exactly one of the three is 1 at every point. Colour, move number, ko and move history are not given.
Output (82 logits): logit[p] for p = 0..80 means "play at point p"; logit[81] means "pass".
Points: p = row*9 + col, row 0 is the top row, col 0 is the left column (so p=0 is the top-left corner,
p=40 is the centre (tengen), p=80 is the bottom-right corner).
Illegal moves (occupied points, suicide, retaking a ko immediately) are masked out, then the move with the
highest logit is played (ties broken uniformly at random). Pass is always legal. The game ends after two
consecutive passes or 200 total moves; area scoring (stones + surrounded empty points), komi 7.0 for White,
captures and the no-suicide / simple-ko rules as usual. Equal score = draw (counted as half a win).

# The network

- Architecture: 243 → 512 → 512 → 512 → 512 → 512 → 512 → 82, ReLU after each of the six hidden layers, no activation on the output. Shapes: W1 [243 x 512], b1 [512], W2 [512 x 512], b2 [512], W3 [512 x 512], b3 [512], W4 [512 x 512], b4 [512], W5 [512 x 512], b5 [512], W6 [512 x 512], b6 [512], W7 [512 x 82], b7 [82].
- Forward pass: h1 = relu(x @ W1 + b1); h2 = relu(h1 @ W2 + b2); ...; h6 = relu(h5 @ W6 + b6); logits = h6 @ W7 + b7.
- Weights use the [in, out] orientation: W<k>[i, j] connects unit i of layer k's input to unit j of its output.
- Move selection: illegal moves are masked out and the legal move with the highest logit is played (ties broken uniformly at random); you never pick moves yourself.

# Your tools

- `python`: the saved weights are preloaded as float32 numpy arrays `W1, b1, ..., W7, b7` (and in the dict `weights`). Save with `save_weights(W1=..., b1=..., ...)` (only the given tensors are replaced; saving is free and unlimited) or with the `save_weights` text tool for small edits. `load_saved_weights()` returns what is currently saved.
- You can make at most 30 tool calls in total; after that you will be asked for your final answer.

# How you are scored

Your final network plays 128 games against each of three opponents, `small_s0`, `small_s2` and `large` (384 games), with colours alternating and each pair of games starting from 2 shared random opening moves, on held-out opening seeds that no forward pass uses. The score is your win rate against each opponent (a draw counts half). The weights that count are the **last weights you save**, not the best ones you tried: whatever is saved when the session ends is what is scored. Make sure the weights saved when you finish are the ones you want.

# Rules

- No training or gradient methods: no backpropagation, optimisers, fitting or regressing weights to targets, or any other optimisation of the weights against data. Every weight must be written by your own reasoning and by code that sets it directly.
- Do not run the network, any of its layers, or any part of it on inputs in the sandbox (no board encodings, real or made up, pushed through the weights, by code or by hand), and do not simulate Go games (no game engine that plays out moves, self-play, or tests the policy). You may write Go-logic code that helps you construct the weights (for example neighbour tables, or which input patterns mean a chain has one liberty), and you may analyse the weight arrays themselves freely. There is no way to observe the network playing in this condition.
- No web access.

# Final answer

When you are done, reply with a line `NETWORK: <what your final saved network computes, at most 300 characters>`, followed by a short explanation of how the layers work.
