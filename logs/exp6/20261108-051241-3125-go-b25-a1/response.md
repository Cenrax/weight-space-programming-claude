The final saved network is the P10 configuration. I checked that the saved weights match exactly what my builder produced. With these weights it won **99.2% against small_s0** (127 of 128), **99.2% against small_s2** (127 of 128) and **94.5% then 91.4% against large** in two separate batches. The scoring games use held-out openings, so expect results around these numbers. Two of the 25 test batches were left unused.

Almost all remaining losses are as Black against large (about 87–91% as Black versus about 95–98% as White), which is consistent with the 7-point komi.

NETWORK: Hand-built move scorer: line prior; asymmetric influence penalty (own-dominated points penalised more); bonus next to own stones; 2×2 connect/cut shapes; capture weak opponent chains; extend own weak chains; self-atari, eye-fill and eye-shrink penalties; passes only if every move scores below −26.

**How the layers work**
- **Layer 1:** passes your stones and the opponent's stones straight through. For each point and colour it also computes two values from the number of empty neighbours (marking "at least 1" and "at least 2"), which later layers use to judge liberties.
- **Layer 2:** marks a stone as "Good" (its chain is not in atari) if it has 2 or more empty neighbours, or 1 empty neighbour plus a same-colour neighbour that also has one. It also starts a running score for each of the 82 moves, offset by +100 so it stays positive through the ReLUs. That score holds:
  - a preference by line: edge 0, 2nd line 1.0, 3rd line 2.0, rising to 2.4 at the centre;
  - an influence term, kernel 1 / 0.5 / 0.25 / 0.1 by distance, scored as −1.5·max(D,0) − 0.7·max(−D,0), where D is your influence minus the opponent's;
  - +2.2 for each neighbouring own stone and +0.3 for each neighbouring opponent stone.
- **Layers 3–5:** spread "Good" along chains, one step per layer, and add the eye penalty: −60 for filling a true eye and −30 for shrinking an eye space. Layer 4 adds "atari-check" units for single opponent stones, giving +4 for an exact single-stone capture. Layer 5 marks a chain as Weak if no Good stone is within reach.
- **Layer 6:** adds +10 for playing next to a weak opponent chain and +9 for extending your own weak chain when the new point has at least 2 empty neighbours. It subtracts 15 for an estimated self-atari, and adds the 2×2 shapes: +6 for a connecting move and +2 for a cutting move.
- **Output layer:** removes the +100 offset; the pass move scores −26.

**What I changed after each test batch**
1. **Batches 1–2:** a simple version scored 91% against small_s0 but 53% against large, losing because it passed too early.
2. **Batches 3–4:** I added chain-level weak/Good liberty logic and stopped self-atari from being counted on capturing moves: 55% → 69% against large.
3. **Batches 5–6:** making the influence penalty lopsided (own-dominated points penalised more) took large to 87%; small_s2 stayed around 70%.
4. **Batches 7–10:** stronger lopsidedness, a lower pass score and the 2×2 shapes alone changed little (small_s2 stayed at 67–71%).
5. **Batches 11–15:** the bonus for neighbouring own stones mattered most. Raising it and the connect weight took small_s2 from 81% to 89% to 99%, then large reached 94.5% and small_s0 99%.
6. **Batches 16–22:** further tweaks were all no better (83–91% against large): a bigger own-neighbour bonus, a larger cut weight, more contact pressure, a different opponent-side penalty, a bonus for ataris on two-liberty stones, and preferring the 3rd line or the centre more strongly.
7. **Batch 23:** I reverted to P10 and confirmed it against large at 91.4%.

Most of these settings differ by less than the batch-to-batch noise (about ±2.5% against large). I chose P10 because it was the only configuration tested against all three opponents and it scored best overall.