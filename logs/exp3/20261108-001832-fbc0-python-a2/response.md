**Caveat up front:** without the embeddings or the unembedding, nothing in these weights names any character. My language answer is a weak inference and should be read as a low-confidence guess. All the evidence below comes only from linear algebra on the weight arrays: SVDs, eigenvalues, and products of the weight matrices. I ran no forward passes.

## 1. Language / kind of text

**Answer: English prose. Confidence about 0.3.**

- **Vocabulary size.** Block 3's MLP output and biases only affect the loss through the unembedding. So the directions block 3 meaningfully writes into can number at most about V−1, where V is the number of characters.
  - The spectrum of `blocks.3.mlp.proj.weight` falls steeply up to about index 73–90, then flattens into a slow tail.
  - The block-3 output bias is about 95% contained in the top ~90 of those directions.
  - So V is roughly 85–100.
- **What that rules out.**
  - DNA-like alphabets (about 4 characters) and text8-style alphabets (27 characters).
  - Tiny-Shakespeare's 65-character set, since there is no break at 64.
  - Large-alphabet scripts such as Chinese, which would give a nearly full-rank spectrum.
- **What remains.** A vocabulary of about 85–100 fits an English book-style corpus (letters in both cases, digits and punctuation). It fits code or an accented European language almost as well. I found nothing in the weights that separates those options, so "English" mostly reflects the prior.

## 2. Head roles

**Tools I used:**
- **Eigen-structure of each head's query–key matrix (Wqᵀ·Wk).** I looked for rotation-like conjugate eigenvalue pairs, which indicate attention by relative position.
- **Composition scores between heads.** "K-composition" asks whether a later head's keys read an earlier head's output; "Q-composition" asks the same of its queries. I also checked whether each product matrix has mostly positive eigenvalues, which means "match same with same".

**How I got the positional findings:**
- Three layer-0 heads (0.3, 0.4, 0.5) have 50–80% of their query–key weight in a shared set of about 33–40 "rotation planes", which are presumably the positional-embedding subspace.
- Head 0.5 rotates by about 26°, 51°, 76°, 97°… in successive planes, a clean harmonic pattern, so it is a fixed-offset head.
- In the same planes, 0.3 rotates by roughly one third as much, so it looks back about one third as far.
- I read this as 0.3 having offset 1 and 0.5 offset about 3. That rests on 0.3 being the nearest head, not on a measured offset: I can't recover the absolute scale. If the positional embedding were smooth over the full context, the offsets could be larger (around 6 and 18).

**Layer 0**
- **0.0:** weak content-based head (mostly non-positional query–key, small output).
- **0.1:** weak content-based head; negative self-match, so it prefers characters unlike the current one.
- **0.2:** one dominant query–key direction, partly positional, attending to a particular character class or position, probably a delimiter. Its output carries only a few dimensions, but layer-1 heads read it heavily in their keys and queries.
- **0.3: previous-token head.** It has the smallest relative offset, about 82% of its query–key weight is in the positional rotation planes, and its output is flat and high-rank, so it copies the character's identity. Later heads use that output in a structured, matching way.
- **0.4:** relative-position (local-window) head using high-frequency positional planes, looking a few characters back.
- **0.5:** relative-position head with offset about 3 (about 3× head 0.3's).

**Layer 1**
- **1.0:** "same-segment" head; both its queries and keys read head 0.2's output.
- **1.1:** broad averaging head (weak query–key, large output). Its output feeds the keys of 2.1 and 2.3.
- **1.2, 1.3, 1.5:** context-matching heads. Their keys read 0.2 and their queries read the previous character from 0.3. These are possible but unconfirmed induction-like heads; I did not list them.
- **1.4:** matches the previous character (via 0.3) against earlier characters.

**Layer 2**
- **2.0:** matches the previous character and the current one against earlier positions. Its output is the strongest input to the layer-3 induction keys (matching scores of 0.93–0.98), so it acts as a content-based previous-character relay.
- **2.1, 2.3:** context matching, using 1.1's output in their keys.
- **2.2:** duplicate-token head (self-match 0.48).
- **2.4:** segment-based matching, with keys reading 0.2.
- **2.5:** mixed content head.

**Layer 3**
- **3.0, 3.1, 3.4, 3.5: induction heads.** Their keys read the previous-token head 0.3 with strong same-with-same matching (0.34, 0.60, 0.42, 0.32). They also read 2.0's output (0.93, 0.98, 0.80, 0.72). Their queries do not match on the previous character, which is the induction pattern, and they write heavily into the output-relevant directions. 3.1 is the clearest.
- **3.2:** direct output head with little composition.
- **3.3:** duplicate-token / copy head.

**Unexplained:** in layers 1–3, every head's combined value-output matrix (Wo·Wv) has strongly negative eigenvalues, so each writes roughly the negative of what it reads. I don't know why, and I didn't use it in any conclusion.

ANSWER: {"language": "English", "confidence": 0.3, "induction_heads": ["3.1", "3.0", "3.4", "3.5"], "previous_token_heads": ["0.3"]}