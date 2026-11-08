CONTROLLER: Bang-bang on the first link's angular velocity: torque -1 (action 0) when dtheta1 > 0 and torque +1 (action 2) when dtheta1 < 0, i.e. torque = -sign(dtheta1). It never chooses zero torque.

The saved policy swung up in all 64 test episodes it ran in (two batches of 32). Mean return was -79.5 in one batch and -85.8 in the other, so expect about -80 to -86 on the final 100 episodes, slightly worse than the -100 reward threshold. Two variants that added dtheta2 scored in the same range, so there may be no real difference between the three.

**How the network does it:** only a few weights are non-zero, and the other 4,797 stay at zero.
- **First layer:** hidden unit 0 computes tanh(-20·dtheta1), which is close to a sign function of dtheta1 (W1[4,0] = -20).
- **Second layer:** W2[0,0] = 5 passes that through as tanh(5·h1₀), which keeps it close to ±1.
- **Output:** W3[0,2] = +1 and W3[0,0] = -1. When dtheta1 < 0 the unit is positive and action 2 (+1 torque) wins. When dtheta1 > 0 action 0 (-1 torque) wins. Action 1 only wins if dtheta1 is exactly 0.

**What the rollouts changed:**
1. **First try, torque = +sign(dtheta1):** failed in all 32 episodes (-500). The traces showed link 1 staying near the bottom while link 2 spun freely. Pushing on the joint moves link 1 the opposite way, so this sign was holding link 1 back instead of pumping it.
2. **Sign flipped, torque = -sign(dtheta1):** succeeded in all 32 episodes, mean -79.5 (range -104 to -72).
3. **Adding +0.25·dtheta2 to the switching signal:** worse, mean -89.75, with one episode at -215.
4. **Adding -0.1·dtheta2:** also worse, mean -85.5, with one episode at -188.
5. **Back to the pure -sign(dtheta1) policy and re-tested:** mean -85.8 (range -148 to -69), all 32 successful. This is the final saved policy, and I checked that the saved weights match it.

The spread between runs 2 and 5 for the same policy (-79.5 vs -85.8) is about as large as the differences between the variants. I kept the pure version because its results had smaller spread and no very slow episodes in the first batch.