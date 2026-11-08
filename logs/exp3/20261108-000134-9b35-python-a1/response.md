I can't answer the language question with much confidence from these weights. The head roles are better supported, but they come from weight-only circuit analysis with no forward passes, so they are inferred rather than measured.

## 1. Language / kind of text

Without the embedding and unembedding, no direction can be tied to a particular character. The only clues are indirect:

- **Vocabulary size.** The trained part of the weights that read the input (block‑0 value weights) and write the output (block‑3 MLP output) both level off at roughly 60–90 dimensions. That points to a modest alphabetic character set of about 65–100 characters.
- **What that rules out.** It is not DNA or another 4–5‑symbol alphabet, which would give a cut‑off around 4–5. It is not Chinese or Japanese either, which would fill all 384 dimensions.
- **A word‑boundary head.** Five of the six layer‑0 heads share one "default key" direction through their query biases. The keys in that direction carry almost no value (|W_v·d| ≈ 0.05), so it acts as an attention sink, probably a very common character like space or newline. Head 0.2 is low‑rank and attends to that same direction. It looks like a word‑boundary tracker, which suggests text with spaces between words.

These clues fit English prose such as Shakespeare, but they fit other Latin‑script languages or code just as well. My best guess is English, with low confidence (~0.3).

## 2. Head roles

**How I found the position‑based heads.** I took the top 70 directions read by layer‑0 values as a rough "token" subspace. In heads 0.3, 0.4 and 0.5, 65–84% of the query–key weight sits outside that subspace, with queries and keys reading the same ~24 dimensions. In heads 0.0–0.2 most of it sits inside. I treat those shared ~24 dimensions as positional. Within them, each head acts like a set of 2‑D rotations, and the angle in each rotation plane should be (position frequency) × (offset the head attends to):

- **0.5:** angles of about 26°, 51°, 76°, 97° and 120°, in near‑harmonic steps.
- **0.4:** mostly high‑frequency planes, at 116–179°. In the planes it shares with 0.5, its angles are about 1.0–1.2× 0.5's, so it seems to look back by the same offset.
- **0.3:** almost symmetric. Its angles are only about 0.28× 0.5's, so it looks back much less.

Some head has to move the previous character forward, and that needs sharp offset‑1 attention, which needs high frequencies. The only head that has them is 0.4, so offset = 1 for 0.4. That makes 0.5 a broader previous‑token head and 0.3 mostly the current position. This depends on guessing the offset scale from that argument; the high‑frequency planes don't fit perfectly. Later layers barely read these positional dimensions (no more than chance, ~0.004 of their query–key weight), so I found no previous‑token heads after layer 0.

**How I found the induction heads.** For each later head I scored whether the query matches on the *current* character (raw embedding or 0.3's output) while the key matches on the *previous* character (0.4/0.5's output), and compared it with the reverse pairing:
- Only layer‑3 heads lean the induction way: 3.1 is clearly strongest (+0.76 vs −0.04), then 3.5 (0.42 vs 0.11), 3.0 (0.32 vs −0.23) and 3.4 (0.31 vs 0.02).
- Layer‑1 heads score in both directions. They look more like general matching on the local 2‑character context than true induction.
- 3.1's key also reads strongly from head 2.0's output, at 4.1× chance.

| Head | Role |
|---|---|
| 0.0 | Content head: matches on character identity |
| 0.1 | Content head that avoids the same character (negative match); weak output |
| 0.2 | Low‑rank head picking out one character class; likely a word‑boundary tracker. Layer‑1 and layer‑2 keys read its output heavily |
| 0.3 | Positional: attends mostly to the current position (some to the previous); re‑copies the current character |
| 0.4 | **Previous‑token head** (sharp, high‑frequency) |
| 0.5 | **Previous‑token head** (broader, recent‑token) |
| 1.0 | Content head built on 0.2's output (boundary or word context) |
| 1.1 | Broad averaging head (weak query–key, large output); layer‑2 keys read it |
| 1.2, 1.3, 1.4, 1.5 | Match the local 2‑character context (current + previous) |
| 2.0 | Local‑context matcher; feeds the keys of 3.1 and 3.0 |
| 2.1 | Broad averaging / context‑summary head |
| 2.2 | Local‑context matcher |
| 2.3, 2.5 | Content mixing; no clear specialisation |
| 2.4 | Reads 1.0/1.1 (queries) and 0.2/1.1 (keys); word‑level context |
| 3.0 | Induction (moderate) |
| 3.1 | **Induction (strongest)** |
| 3.2, 3.3 | Late content / readout heads; no clear specialisation |
| 3.4 | Induction (weaker) |
| 3.5 | Induction |

ANSWER: {"language": "English", "confidence": 0.3, "induction_heads": ["3.1", "3.5", "3.0", "3.4"], "previous_token_heads": ["0.4", "0.5"]}