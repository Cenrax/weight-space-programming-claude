I'm working out which game this is from the network's weights: the strongest first-layer weights sit on RAM bytes 70, 99, 101, 103, 105 and 109, which map to addresses 0xC6, 0xE3, 0xE5, 0xE7, 0xE9 and 0xED. On the output layer, the biggest contrasts are among the RIGHT, UPRIGHT and DOWNRIGHT actions.

I'm weighing candidate games against the RAM addresses: the odd-spaced bytes 99 through 109 look like object arrays, and Freeway's car positions don't line up with them. Seaquest does fit, since its player x is at RAM 70, so I'm checking its other address mappings against the weights.

I'm running through other candidate games and their RAM layouts to see which one matches the weighted addresses. Boxing, Tennis, Pitfall, Montezuma, Ms. Pacman, Berzerk and Breakout all come up, but none of their known player-position bytes line up cleanly with the odd-spaced 99 through 109 values yet.

I'm checking whether Breakout fits better than Seaquest: its ball x sits at RAM 99 and ball y at 101, which are among the weighted bytes, and the left/right action contrasts also suit a paddle game. The one puzzle is that the heaviest weight is on byte 70 rather than the paddle position at 72, so I'm wondering whether 70 plays a paddle-related role.

I'm checking whether the paddle-relative pairs in row 70 and row 99 point the same way, since the unit-level signs I'm looking at are mixed and don't settle it yet.

I'm treating the unit-level signs as inconclusive for now, since row 109 pairs against row 99 in opposite directions, which makes me suspect 109 isn't a paddle-related byte. I'm settling on Breakout as the working guess with moderate confidence, mainly because the left/right output contrasts fit a paddle game.

