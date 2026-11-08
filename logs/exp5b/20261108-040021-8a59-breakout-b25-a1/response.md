The final saved weights scored a mean of about **175** (min 61, max 409) on a 32-episode check rollout. The best settings tested scored roughly 175–189 per 32-episode run. That's up from 13.5 for my first version. All 25 rollouts are used up, and the weights saved now are the ones that will be evaluated.

CONTROLLER: Serve the ball with FIRE. While the ball falls, move the paddle (RIGHTFIRE/LEFTFIRE) toward where it will cross paddle height (y≈181), using its direction bytes and one side-wall bounce. While it rises, follow the ball's x. Hold still (FIRE) within ±9 px of target; aim point = ball x − 10.

**What the RAM bytes mean** (found by tracing them in rollouts): byte 99 is the ball's x, 101 the ball's y, 72 the paddle's x (55–191) and 57 the lives. Byte 103 is the ball's vertical direction and speed: 1–4 means falling, 252–255 means rising. Byte 105 is the horizontal direction: 1 or 2 means moving left, 255 or 254 moving right, and 2 is about 1.5× faster than 1.

**How the network works:**
- **Layer 1 (61 units):**
  - 9 threshold units turn the direction bytes into yes/no flags.
  - 52 units each check, for one of 13 possible ball slopes, whether the predicted landing spot is left or right of the paddle by more than the dead zone. Half of these handle the case where the ball bounces off the side wall first.
  - 2 units cover the rising-ball "follow" case.
- **Layer 2 (58 units):** each unit is an AND that fires only for one combination of the ball's horizontal and vertical speed (or for "ball rising") and one move direction.
- **Output:** each layer-2 unit that fires adds to the RIGHTFIRE (11) or LEFTFIRE (12) score. FIRE has a constant score of 1, so it wins when nothing fires, which serves the ball and otherwise leaves the paddle still.

**What I changed in response to the rollouts:**
1. **Simple follower (13.5 → 36):** the first version just chased the ball. Aiming at ball x − 8 worked best, and giving FIRE a small positive score fixed a tie that left the ball unserved.
2. **Direction bytes:** traces showed bytes 103 and 105 are the ball's direction, and that 105's sign is the opposite of what I first assumed. Fixing that and adding landing-point prediction helped a little. Testing the paddle height at 172, 179, 185 and 198 put the best value around 179–183 (39 points at 179).
3. **Dead zone (the big win):** traces showed the paddle moves about 11 px per agent step, so a ±2 px dead zone made it overshoot back and forth. Widening it raised the score from 39 (±2) to 67 (±6) to 150 (±9); ±12 dropped to 96.
4. **Fine-tuning:**
   - **Aim offset:** 6 scored 84, 8 scored 150, 10 scored 186, 12 scored 147.
   - **Dead zone:** ±8, ±9 and ±10 were about equal.
   - **Speed model:** a slower vertical-speed model for the higher speed values scored worse (127), so I kept vertical speed equal to the byte value.

**Limitations:**
- Each 32-episode score has a standard error of about 18–20, so settings within roughly 30 points of each other couldn't be told apart.
- I never tested predicting where a rising ball will land.
- The dead zone stays the same after the paddle shrinks to half width, which is probably where many late lives are lost.
- Test episodes were capped at 5,000 steps but none reached the cap, so the final evaluation's longer 27,000-step limit shouldn't change scores much.