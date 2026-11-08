I saved the final weights (configuration "B") and checked that what's stored matches what I built. It is the configuration I tested most, on 5 of the 50 batches. Expect roughly 0.83–0.92 against `large` and about 0.89–0.95 against each small opponent. Its two runs against `large` gave 0.918 and then 0.832, so one batch of 128 games moves by several points. Some of the gains I credit to individual changes below are within that noise.

NETWORK: Hand-written Go tactics. Layer 1 copies the board and marks stones with liberties; layers 2–5 spread those marks along same-colour chains (two labels by liberty-point parity) to find chains in atari; layer 6 scores capture, self-atari, eye-fill, liberty-fill and contact; readout adds a position prior.

**How the layers work**
- **Layer 1 (486 units in use):** keeps a copy of our stones and the opponent's (162 units). For each stone and each colour it also sets two "has a liberty" marks (324 units). The two marks are labels by checkerboard parity of the liberty point (row+col even or odd). A mark is also set when the stone has two or more liberties.
- **Layers 2–5:** each layer spreads the marks one step further along stones of the same colour. A large negative bias stops a mark from crossing into an empty point or an enemy stone. Together with the extra step in layer 6, a chain is seen up to about 5 steps from any stone.
- **Layer 6 (about 486 units) and the readout:**
  - **Capture:** +15 to each liberty of an opponent chain that has no liberty of the other label, i.e. a chain in atari.
  - **Self-atari:** a lone stone left with one liberty or none scores −4 per missing liberty. Joining an own chain into a self-atari costs only −2 per missing liberty.
  - **Eye-fill:** −20 for playing a point surrounded only by our stones.
  - **Liberty-fill:** −3 for a move with no empty neighbours that touches exactly one of our stones.
  - **Contact:** +0.3 per neighbouring opponent stone and +0.1 per neighbouring own stone.
  - **Prior and pass:** a per-point prior in the output bias by distance from the edge (0.2 on the edge, 0.8 on the second line, 1.2 on the third line, 1.1 on the fourth, 1.0 at the centre). Pass has logit 0, so it passes only when every legal move scores below zero.

**What I changed after each batch** (win rate per 128 games: s0 = `small_s0`, s2 = `small_s2`)
1. **Baseline** (edge-distance prior, single-stone capture, eye avoidance): s0 0.84, s2 0.66, large 0.62.
2. **Chain-wide atari detection** (capture, escape and self-atari): large 0.84.
3. **Escape bonus scaled by the escaping stone's empty neighbours:** s2 0.75.
4. **Lower pass logit and a cheaper lone self-atari:** s2 dropped to 0.46–0.52, so I reverted it.
5. **Contact bonus:** s2 0.87. A stronger contact bonus (0.6) was worse.
6. **Liberty-fill penalty:** large 0.86.
7. **Things that did not help and were dropped:**
   - Rewarding ataris on single opponent stones.
   - Penalising lone contact moves.
   - A centre-heavy prior (large fell to 0.67).
   - A wider shape bonus over nearby points: equal against `large`, but `small_s0` fell from 0.90 to 0.78.
   - A penalty for filling our own territory.
8. **Removing the escape bonus:** large 0.887, s2 0.895, s0 0.898. Escaping atari on its own did not help, perhaps because the atari detector often fires on chains that are not really in atari.
9. **Cheaper self-atari when joining an own chain (2 instead of 4):** large 0.918, s0 0.887 and 0.949, s2 0.953. This became the final setting.
10. **Stricter self-atari check:** the current check sometimes misses self-ataris, because it can count an unrelated friendly chain next to the target point. Fixing that made the network pass more often and lose more (large 0.855), so I kept the original check.
11. **Last tweaks:** a lower self-atari penalty and a lower liberty-fill penalty were no better. A final re-run of the saved weights scored 0.832 against `large`.

Known weaknesses:
- **Square-eye loop:** in some games it repeatedly plays into the opponent's square eye and gets captured each time.
- **Big chains:** chains longer than about 5 steps can be wrongly read as in atari.
- **Black:** it does a little worse as Black than as White because of the 7-point komi.