I've saved a hand-built network. Every weight was set directly by code from Go rules; nothing was trained. Because there were 0 forward passes, it has never been run or seen playing, so I can't tell you how strong it is. I checked that the saved arrays match what I built, but the logic inside the units was checked only by reasoning.

NETWORK: A rule-based player. It captures enemy groups of 1–3 stones that have one liberty left, saves its own 1–3-stone groups in atari when extending gains liberties, avoids self-atari and filling its own eyes, and prefers good points and contested areas. It passes only when the remaining moves are eyes or safe inside its own territory.

**How the layers work**
- **Shared structure:**
  - **Board copy:** Hidden layers 1–5 carry a copy of the "your stones" and "opponent stones" inputs. Empty points are worked out as 1 − yours − theirs.
  - **Running score:** Every hidden layer also keeps an 81-unit running score for the points, offset by +200 so ReLU never cuts it off. Each layer adds what its features found, and the output subtracts the 200 again.
  - **Pattern test:** Each pattern feature is a single ReLU unit. Its weights are 2 for each stone in the group, ±1 for each neighbouring point, and a bias chosen so it outputs exactly 1 only when the group has exactly one empty liberty and no friendly stones next to it.
- **Layer 1:**
  - The running score starts as a position bonus (centre, 3rd and 4th lines best, edge worst), plus 0.3 per empty neighbour, plus the friendly half of the contested-area bonus (see layer 6).
  - Feature units flag single enemy stones in atari (later +12 on their liberty), single own stones in atari, and single-stone self-atari points (−5).
- **Layer 2:** Counts how many of your stones in atari touch each point, and adds patterns for your own 2-stone groups in atari.
- **Layer 3:**
  - A "save" unit is ReLU(3·(groups in atari next to the point) + its empty neighbours − 4). It fires only if extending there leaves at least 2 liberties, and adds +7.
  - It also adds patterns for enemy 2-stone groups in atari (+14).
- **Layer 4:** Eye points, where all neighbours are your stones, get −40. Enemy straight 3-stone groups in atari get +16.
- **Layer 5:**
  - Your own straight 3-stone groups in atari get +5.
  - "Inside my territory" points get −15 for each own neighbour beyond the first. These are points with 2 or more of your neighbours and no enemy stone within 2 steps.
- **Layer 6:** No board copy.
  - Enemy L-shaped 3-stone groups in atari get +16.
  - A contested-area bonus of 0.5·min(your nearby stones, their nearby stones), made from a linear term minus a ReLU term.
  - +1.5 for playing next to a single enemy stone with two liberties.
- **Output:** The score for each point, with pass fixed at −10. Ordinary moves and self-atari moves always beat pass. Interior and eye moves fall below it, so the network keeps playing until only its own territory and eyes are left.

**Known gaps:** It can't see atari on groups of 4 or more stones. It has no idea of ladders or of whether a big group lives or dies. The position and contested-area bonuses are the only strategy it has outside fights.