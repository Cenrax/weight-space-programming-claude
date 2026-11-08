CONTROLLER: Gated PD lander. angle_targ=0.4·tanh((x+2vx)/0.4); A=1.5(angle_targ−θ)−3ω; H=0.5(0.55|x|+0.8|vx|−y)−0.5vy. With a leg down: A=0, H=−0.5vy. Fire main if H>|A| and H>0.05, otherwise left if A>0.05, right if A<−0.05, else do nothing.

The final saved weights are this controller. In its last two test runs (64 episodes) it averaged **279.8** and **278.6**, and every episode landed with a score of at least 244.

**How the network does it**
- **Layer 1:** small weights keep the tanh units in their near-linear range. They compute the attitude term (−θ−2ω), the altitude term (−y−vy) and the in-contact term (−vy). One unit gives the clipped angle target. Two pairs of shifted tanh units approximate |x| and |vx|. One steep unit acts as a leg-contact switch.
- **Layer 2:** one unit computes A and another computes H. Each also gets +10× the contact switch, which pins it at 1 (effectively off) when a leg is down. A separate unit carries −0.5·vy and only switches on when a leg is down. A fourth unit is the contact indicator, used to cancel offsets.
- **Output:** the logits are fixed at 0.05 for "do nothing", A for left, −A for right and H for main. Because left and right are negatives of each other, picking the largest logit reproduces the comparisons above. Squashing A and H through the same tanh doesn't change which is larger.

**What I changed based on the rollouts**
1. **Starting point:** Gymnasium's built-in hand-written landing controller scored 240, with 2 crashes and 1 timeout. Most failures were touching down while still moving sideways, then sliding off the pad.
2. **Stronger attitude control** (ka, kw raised) brought it to 264.
3. **Adding |vx| to the hover target** keeps the lander hovering until its sideways speed is gone. This was the biggest single gain: 274, with every episode landing.
4. **Faster position control** (kx=1, kvx=2) and settling on ka=1.5, kw=3, |vx| weight 0.8 gave about 279.

**Things I tried and rejected**
- Descending faster (bigger y gain, or a built-in touchdown speed): many crashes.
- Gating control off only when both legs touch: one timeout, firing side engines while resting on one leg.
- Attitude control while on the ground: more crashes.
- Giving the main engine priority: more hovering, lower score.
- A sharper |x| approximation, and an extra main-engine boost when descending too fast: no crashes in the boost runs, but the score dropped about 5 points.

**Remaining risk:** if it lands near the edge of the pad, it can slip onto the slope and slide away. This showed up as roughly 1 episode in 32 in nearby settings, though not in the final configuration's 64 test episodes.