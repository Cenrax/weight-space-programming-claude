The final saved weights are version 6 (checked: the saved arrays match it exactly). In its last two test batches of 128 games each, it won 89.1% against `large` and 90.6% against `small_s2`. It was never tested against `small_s0`. The only s0 batch used the first version, which won 98.4%, and later versions only added features or changed weights. Each batch has roughly ±3–4% noise, so differences between versions of a few points may not be real.

NETWORK: hand-written Go heuristics. Logit = position prior + liberties/contact − |local influence| + big bonuses for capturing 1–3-stone chains in atari and for saving own 1–2-stone chains, atari/connect/cut bonuses, penalties for self-atari and own-eye filling; pass is −40.

**How the layers work**
- **Carried planes:** layers 1–5 copy the "my stones" and "opponent stones" planes forward (162 units), so later layers can still read the board. Every layer also carries an 81-unit score plane, offset by +100 so ReLU never clips it. W7 copies the score plane to the output and subtracts the 100.
- **Pattern units:** each unit fires only when an exact local pattern is present, e.g. "opponent stones exactly on shape S, and every neighbour of S is mine except one". The bonus is added to all neighbours of S, but only the one empty liberty can be played, so it lands in the right place.
- **Layer 1:** 3rd-line and centre prior, small bonuses for empty and opponent neighbours. Positive and negative influence within distance 3, penalised so it prefers contested points. Own-eye filling −50.
- **Layer 2:** lone-stone self-atari −8; capturing a single stone +14.
- **Layer 3:** detects my 1- and 2-stone chains that are in atari.
- **Layer 4:** sums those into a rescue plane, adds my single stones with 2 liberties, and detects opponent single stones with 2 liberties (atari bonus +4).
- **Layer 5:** rescue bonus (+8 per stone) only applies when the rescuing point has at least 2 empty neighbours. Penalty −6 for filling a liberty of my 2-liberty stone when that point has no empty neighbours.
- **Layer 6:** capturing a 2-stone chain +18, a straight 3-stone chain +22.
- **Spare units in several layers:** 2×2 diagonal patterns, with +2 for connecting at a cutting point and +1.5 for cutting the opponent.

**What I changed after each test**
1. **First version:** 98.4% vs s0, then 81% vs `large`. In the games it lost, it passed while the opponent was still playing inside its area.
2. **Pass moved to −40, influence penalty reduced, contact and atari bonuses added:** 80% vs `large`, 87% vs `small_s2`. Losses came from losing the territory race and groups dying.
3. **Added a "contested territory" term:** 76% vs `large`, so I dropped it.
4. **Simplified the rescue logic and added the 2-liberty defence:** 82% vs `large`.
5. **Added connect/cut patterns, removed the 2-stone atari bonus to make room:** 83.6% vs `large`, 93.8% vs `small_s2`.
6. **Halved the penalty for playing near opponent stones (0.7 → 0.4),** because Black kept failing to contest the opponent's wall: 89.1% vs `large`, 90.6% vs `small_s2`. Against `large` that is about 5 points better; against `small_s2` about 3 points worse, which is within noise. On balance I kept version 6.

**Known weaknesses:** it cannot see atari for opponent chains bigger than 3 stones (or bent 3-stone chains), or for its own chains bigger than 2. It also has no life-and-death judgement, so most losses are games where it plays Black and a group dies.