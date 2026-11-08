**Caveat up front:** the head labels below are inferred only from weight structure (eigen-structure of the QK and OV matrices, and how heads compose), not from watching attention patterns. The language guess is weakly supported.

## 1. Language

Without the embedding or unembedding I can't see which character is which, so all I can estimate is roughly how many characters there are.

- **Block 3 writes into an ~90-dimensional space.** Block 3's outputs are read only by the unembedding, so weight decay pushes them into the unembedding's row space. Its MLP output matrix has a clear knee in its singular values at about 85–100 dimensions, compared with block 2's matrix. A two-line breakpoint fit gives 86–114 depending on the range used.
- **What that rules out.** The alphabet is not tiny (DNA/digits would give about 5–15) and not huge (Chinese or Japanese would give thousands and no knee). It is somewhat larger than tiny-Shakespeare's 65.
- **What it fits.** Ordinary Latin-script text with about 80–100 symbols: English prose, or possibly code (printable ASCII is about 95 symbols) or another European language.
- **Copying is strong.** The heavy investment in copying circuits (below) fits natural text or code alike.

My best guess is English, with low confidence (about 0.35).

## 2. Attention heads

**How I found positional heads.** Only heads 0.3 and 0.5 have QK matrices whose eigenvalues come in rotation pairs inside a shared set of planes. That is the signature of attention by relative position over sinusoid-like positional embeddings. The two heads together put about 50% of their QK energy in 20 shared planes; every head in layers 1–3 is at chance level (about 0.3%).
- **0.5** rotates by about 26°, 51°, 76°, 97° and 117°, roughly 25.5° times 1, 2, 3, 4… This is a clean fixed offset.
- **0.3** rotates the same planes, in the same direction, by only about 3–38°. So its offset is about a third of 0.5's, and its spectrum is flat, meaning sharp attention.

**Choosing the previous-token head.** The QK matrices alone can't tell whether 0.3 looks back 1 position and 0.5 about 3, or 0.5 looks back 1 and 0.3 about 0. How the outputs are used breaks the tie:
- 0.3's output is read heavily downstream. Layer-3 keys read it (z up to 22 against randomly rotated baselines), and so do the queries of 1.2–1.5 and 2.0 (z up to 27).
- 0.5's output is barely read (z ≤ 2 on the key side).
- A sharp "attend to self" head would be useless, since the residual stream already holds the current character.

So I call 0.3 the previous-token head. This is a judgement call and could be swapped with 0.5.

**How I found copying (induction) heads.** For each pair of heads I computed the trace of the later head's QK times the earlier head's OV, z-scored against random rotations. A strongly positive value means: "the earlier head's output at the key position matches what the query position holds."
- Head 2.0 feeds 3.1 (z = 182), 3.0 (94), 3.4 (34) and 3.5 (25).
- Head 0.3 feeds 3.1 (22), 3.4 (13), 3.0 (11) and 3.5 (8).
- 2.0's own query reads 0.3's output, matched against the raw key. So 2.0 finds earlier occurrences of the previous character and carries that character's identity forward.
- Layer-3 heads then match the current character against "the character before position k". This is the standard induction circuit.

**Labels:**

| Head | Role |
|---|---|
| 0.0, 0.1 | Weak, diffuse content-averaging heads (small QK and OV) |
| 0.2 | Attends to a particular character class regardless of the query (rank-1 QK dominated by the query bias); likely a "last delimiter/space" head. Feeds 1.0, 1.2 and 1.3 |
| 0.3 | **Previous-token head** (positional rotation, sharp) |
| 0.4 | Positional/local head using high-frequency planes plus a strong key preference |
| 0.5 | Positional fixed-offset head, a few characters back (≈2–3) |
| 1.0 | Matches 0.2's delimiter-context output on both query and key: word/segment matching |
| 1.1 | Broad context-averaging head (flat, weak QK; large OV); feeds the keys of 2.1 and 2.3 |
| 1.2–1.5 | Queries carry previous-token information; they share a rotation plane. Local-context and n-gram heads |
| 2.0 | Look-back helper: finds earlier occurrences of the previous character and writes the representation that the layer-3 copying heads key on (prepares the induction keys) |
| 2.1, 2.3 | Context-matching heads keyed on 1.1's summary |
| 2.2, 2.4, 2.5 | Mixed local-context heads (part of the same shared rotation plane) |
| 3.0, 3.1, 3.4, 3.5 | **Induction (copying) heads**; 3.1 and 3.0 are the strongest |
| 3.2, 3.3 | Other late heads (mostly negative compositions; possibly suppress repeats) |

ANSWER: {"language": "English", "confidence": 0.35, "induction_heads": ["3.0", "3.1", "3.4", "3.5"], "previous_token_heads": ["0.3"]}