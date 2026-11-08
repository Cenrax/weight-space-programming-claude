I'm working out how to hand-design the weights, since there's no training to lean on. The three opponents are unknown quantities, so I'll need a policy that doesn't depend on knowing how strong they are. I'm also checking what the MLP can actually see: only the board, so features have to come from local neighborhood counts, with chain-level tactics like atari being harder to encode.

I'm settling on a few hand-coded heuristics: avoid filling my own eyes so the group doesn't die, pass only once nothing but eye points remain, prefer captures of opponent stones in atari (single stones are checkable locally, chains need more), and add a positional prior that favors 3-3 and 3-4 points over the first line early on.

I'm weighing a baseline first: with all-zero weights every move ties, so play is random, and I could spend a forward pass to gauge the opponents before building heuristics. Layer 1 has room for about six local features per point, and later layers can pass those through as nonnegative identities, so I'm planning a shallow feature design rather than anything deep.

I'm sketching a single-stone capture detector: an opponent stone next to an empty point whose other neighbours are all mine. Extending it to two-stone chains means checking that all neighbours of both stones, apart from the candidate point, belong to me.

I'm now sketching the capture rule for two-stone chains and a few cheap local checks: whether a move would leave my own new stone with only one liberty, whether extending a single stone of mine that is already in atari would give it room, and whether a neighbouring opponent stone has just two liberties left.

I'm thinking about the pass rule: if I pass too early, the opponent could keep playing into open board, so passing should only happen when every remaining move is a bad one, like filling my own eye or self-ataring.

I'm weighing whether a positional prior helps much here, since territory only matters if the game is played out and dead stones aren't removed. I expect a hand-built MLP to be weak against trained networks, though the opponents might be fairly modest, so I'll test that. I'm also thinking about a stronger fallback: capture first, then save stones in atari, then put opponents in atari, then extend territory.

I'm planning to lay out the first layer with pattern units for captures, then let the later layers pass values through unchanged, since the first layer alone can't hold every pattern. Counting the two-stone capture patterns, there are too many for 512 units, so I'll need to merge some of them or spread them across layers.

I'm sketching layer 1 as per-point liberty detectors: an opponent stone whose other neighbours are all mine or the edge, so it has exactly one empty neighbour left. A threshold on the stone plus its neighbours' own-stone indicators should fire only in that case.

I'm checking the single-stone atari units next: an opponent stone with all its other neighbours owned by me and one empty point left, scored per directed edge, about 288 units. The mirror case for my own stones in atari comes after that.

I'm counting the feature budget: passing the raw board through costs about 162 units if I keep only the own and opponent planes, which leaves roughly 350 per layer for new features. Since each feature has to survive into the output, I'm weighing how much room the atari detectors would take across layers.

I'm treating the accumulators as running per-point sums: each layer adds its new features into them, with a large offset keeping the values positive, so the score can build up across all six layers. Layer 1 would carry the own and opponent planes forward plus up to about 350 feature units, and layer 2 would start by passing those planes through again.

I'm weighing a shallow budget for the rest of the network: passing the raw board through costs most of a layer, but a running per-point accumulator can absorb new features in each layer, so the later layers still have room. Deep chain-liberty logic across multiple layers would take too many steps, so I'm sticking to local patterns for small chains. I'll start with a simple heuristic baseline and see how it does before choosing any stronger strategy.

I'm turning to a first heuristic version, since it'll show how the bot actually behaves. For the logit on each empty point, I'm starting with a static positional prior that favours 3rd and 4th line points, though I expect that to matter less as the game goes on, so local responses will need to carry more weight.

I'm laying out the scoring terms for each empty point: a bonus for extending a stone that's in atari, a moderate bonus for putting an opponent single stone down to two liberties, and a penalty for filling my own eye. I'm also adding a self-atari penalty for moves that would leave my new stone with only one liberty and no capture to compensate.

I'm finishing the scoring terms for empty points: a penalty for self-atari when a move leaves my new stone with one liberty, and a static positional prior that favours the 3rd and 4th lines, plus small terms for contact with own and opponent stones. Then I'll settle how the pass logit is set so that passing only wins when every remaining move is bad.

I'm now thinking about the endgame: once the neutral points are filled, filling my own open territory is mostly harmless under area scoring, but I have to stop before I reduce a region to a single eye, since a group needs two separate eyes to live.

I'm thinking about the endgame guard: filling my own territory is only risky when it shrinks a region toward a single eye. So I'm leaning toward penalizing moves inside my own area, where there's no opponent stone nearby and several of my own neighbours.

I'm weighing a bonus for playing next to opponent stones, since that cuts their liberties and helps me capture invaders that would otherwise leave my territory counting for neither side. That only makes sense when the move isn't self-atari.

I'm weighing whether a simple heuristic will even beat the trained nets, and I'm leaning toward building a first version to see how it behaves. Early contact play may be weak, but against these opponents it could still work. Once the borders settle, I want the policy to capture dead invaders in my area, avoid filling my own eyes, and otherwise pass. I'll also wait on any chain-level liberty counting until I see how the first version performs.

I'm setting the positional prior by distance from the edge, rising from zero on the first line to a peak at the center, with tengen scoring highest. On top of that I'm weighing small adjustments: a bonus for each opponent neighbour to add pressure, and some tweaks for own diagonal and orthogonal neighbours, though I'm unsure whether penalizing orthogonal contact would hurt solid connections.

I'm sketching the first layer's units: the own and opponent planes pass straight through, and the empty plane can be recovered as one minus the other two. I'll build this as a simple version and test how it plays before refining the influence-style scoring.

I'm laying out three families of layer-1 units for capture, eyes, and saving a single stone, which together come to about 657 units, well over the roughly 350 I have available.

Now I'm weighing how to fit the layer-1 units into the 512 budget: the pass-through planes plus the capture features come to 531, so I'm considering computing the neighbour-count terms linearly into the accumulator instead.

I'm sorting out how the layers split the work: the first layer carries the own and opponent planes plus the capture-style features, which uses up 531 units and overflows the budget, so the eye detection gets pushed into the second layer. That second layer can then accumulate the linear terms and the capture contributions, with eye detection feeding into the accumulator one layer later.

I'm trying to shrink the second layer's budget: the SaveSingle units over every directed edge would push it past 512. I'm considering merging the per-edge capture features into one unit per opponent stone, which depends only on how many of its neighbours are already mine.

I'm checking whether a single unit per opponent stone can stand in for the per-edge capture features: it would fire when that stone has exactly one non-own neighbour, and for an empty point that neighbour must be the point itself, so the capture bonus lands on the right move without needing a separate unit for every edge.

I'm now replacing the per-edge capture and save features with one unit per stone: a stone in atari with a single empty neighbour fires only when that neighbour is the move, so this cuts the count to 81 per type. The same trick extends to two-stone chains, with 144 units each for opponent and own pairs.

I'm checking whether a single unit per opponent stone can flag atari on a two-stone chain: the unit would fire only when all the other neighbours are already mine, so the one empty liberty it picks out is the move itself. I'm trying to work out how to require that the remaining neighbours are empty rather than opponent stones, without a unit for every edge.

I'm now estimating self-atari for our own moves: if the point has no own neighbours and at most one empty neighbour, and it doesn't capture anything, the move is self-atari. When it does touch our own stones, the merged chain's liberties are harder to pin down, so I'll approximate them from the empty neighbours plus the friendly ones.

I'm now writing the self-atari penalty as a small unit that fires when the point has few empty neighbours and no friendly ones, so it flags moves that would leave the new stone with a single liberty.

I'm checking whether a saved-stone term can use the count of empty neighbours as a gate, so the save bonus only applies when the stone would keep at least two liberties. The problem is that a single linear unit can't express that conjunction, so I'm considering a second-layer unit that sums the save signals from neighbouring stones and adds the empty count, then tests whether the total clears a threshold.

I'm checking the save condition: the saved stone's only liberties are the empty neighbours of the move, so requiring at least two of them, through a threshold on the summed save signals plus the empty count, keeps the bonus honest. I'm also turning to whether the opponent's capture threat needs its own handling.

I'm weighing a static positional prior for the strategic terms: a bonus for each opponent neighbour to add pressure, and a smaller one for own neighbours. Since the prior rises toward the center, the policy would fill the middle first and expand outward, which is a plausible 9x9 approach even if territory has to come from the edges later.

I'm now weighing an influence-based approach: for each empty point, I'd sum how strongly our stones and the opponent's stones reach it, favouring frontier points where the two are roughly balanced. The absolute difference between them would need a pair of ReLU units per point, roughly 162 units in the first layer.

I'm reworking the feature budget: the one-per-point and pair units plus the own and opponent pass-throughs don't fit in layer one, so I'm moving some of them to layer two. I'm also considering influence terms, taking the absolute difference between own and opponent influence with a pair of ReLU units, and noting that the endgame will need those influence signals to penalize moves inside settled territory.

I'm deciding the pass logit now: moves deep inside our own territory should score below passing, and invasions into the opponent's area are only worth it if their territory looks weak.

I'm considering a distance-decaying kernel for the influence terms, though I haven't settled its exact shape yet.

I'm weighing a score for each empty point that starts from a base high enough to beat passing, then penalises how far our influence and the opponent's differ, so balanced frontier points rank highest. I'm also wondering whether a min of the two influences could serve as the frontier signal, though that isn't linear and would need extra handling.

I'm settling on a score for each empty point that drops with the absolute gap between our influence and the opponent's, so open areas where both are near zero keep the prior, and frontier points get played as the board fills. Tactical terms like atari, capture and save will carry the big weights, and I'm leaving cut and connect out of the model for now. I still need to decide the pass logit.

I'm settling on the pass decision: I'll pass only when every empty point scores negative, so a point with a large influence gap is penalised, and I still need to check the base and slope values numerically once the code exists.

I'm sizing the layers now: the first layer can hold the own and opponent pass-throughs plus the atari and influence-difference units, but the second layer runs out of room once I add the self-atari and eye terms. I'm weighing whether the single-stone save signal can move out of the first layer to make space.

Still allocating the layers: the pass-through and pair units fill the budget, so I'm splitting the opponent-pair and own-pair units across separate layers to fit under the limit.

I'm checking how a pair-atari unit would detect a two-stone group whose only remaining liberty is a given empty point, and I'm trying to keep the save condition consistent with that.

I'm checking whether a pair-atari unit can be trusted: for two adjacent opponent stones, it fires only when every other neighbour is already our stone, so the single remaining empty point is the lone liberty. Capturing two stones should be worth more than capturing one, though chains of three or more aren't covered by these units.

I'm checking the self-atari case for the single-stone save unit, since a point next to our own weak stones isn't caught yet. I'm also setting up an eye-fill term that fires when every neighbour of a point is already our own stone.

I'm now picking the eye-fill penalty and a prior that favours the third line, and I'm defining the influence term as a distance-weighted sum of nearby stones, with diagonal neighbours counted more than two-straight ones.

I'm settling on an offset of 100 so the accumulated score stays positive, with tactical bonuses adding and penalties for eye-filling, self-atari and deep territory subtracting.



---

I'm planning a builder class that adds units layer by layer under named groups, so each layer's units are easy to change. Each unit is a ReLU over a weighted sum of the previous layer's units plus a bias, and I'll specify those as a dict of previous-index to weight. For the layer below, I need to track the own and opposite indices per position, with the empty state derived from the other two.

I'm working out how the layer-k-1 representation feeds into the next layer, treating the empty state as a constant that the bias absorbs. I'll write a generic helper that expresses the linear combinations in terms of the own, opposite, and empty symbols, with the first layer taking the empty state directly as input.



---

I'm checking the L3 budget, which comes to 532 against a limit of 512, so SP has to move down. I'm weighing whether to put SP in L4 and push SPG into L5, with the acc save moving to L6.

I'm settling the L4–L6 layout: SP goes to L4 with the own, opp, and acc pieces, and the UP pieces stay in L3, which brings the layers to 388, 388, and 163. For the pass option, I'll start with a threshold of -3 and check it against the influence magnitudes once the board is built.



---

I'm checking the SPG in layer 5 against the own and opponent values from layer 4, and that's consistent. Next I'm looking at the SA unit's formula and comparing my weights and bias against it, but that comparison is still underway.

I've confirmed the SG formula matches the own and opponent counts. I'm weighing whether to compute influence magnitudes by hand to pick a pass threshold, but I'll reason through examples rather than simulate the network, and I'm leaning toward a pass value around -2.



---

Already at 91% against small_s0. Game 34 was a win even though our own territory was filled into a checkerboard pattern. The losses look like we pass too early while the opponent keeps playing inside our area, so I'm checking game 24's final board to see where it went wrong.

Next I'll run the baseline against large and small_s2, since each test uses up one of the 24 remaining, and then work on improvements.



---

I'm at 52% against the large opponent. The big losses seem to come after we pass early while they keep playing inside our area, like in game 78 where White's moves captured our stones. I'm wondering why the "atari_escape" logic only covers singles and pairs, and I'm also looking at how an opponent stone inside our territory lowers the nearby point values.

I'm weighing whether to pass less and fill our own area, since with area scoring that costs nothing as long as enough eyes remain. Filling the opponent's territory might also work as an invasion, because stones we leave there at the end would count for us and neutralize their region.

I'm now proposing that passing less should be paired with a broader liberty-tracking fix, since the atari escape logic only handles small groups and the opponent keeps capturing larger ones. I'm thinking about a propagation scheme that estimates chain liberties layer by layer, though it may overcount liberties shared between stones.

I'm weighing a cheaper alternative to chain liberty tracking: fixing the pass behavior. In game 78, we kept passing while White played through our central group, so the question is why our candidate moves all scored below the pass threshold.

I'm weighing a much lower pass threshold, around -15, so we only pass when every move is an eye fill or self-atari. Filling our own territory is safe as long as the Eye penalty stops us from filling our last eyes, though a region down to two empty points is where it gets tricky.

I'm now checking whether the single-point eye check misses a group that only has one two-point eye region, since filling one of those points would leave it with one eye and open to capture. I'm also noting that filling dame and neutral points is fine, and that the highest-valued moves in our own territory should be the ones we pick when we still have options. The stats I'm looking at show our self-atari rate at 5%, the opponent's at 19-26%, and our capture-taking at only 59-70%.

I'm now weighing a general chain atari detector, since our current check only covers chains of one or two stones and misses most of the captures. I'm considering whether small chains of three or four stones could be handled by enumerating their shapes, though the number of placements grows quickly on a 9x9 board.

I'm working out how to tell whether a whole opponent chain has only one liberty, by checking for each stone whether its only open neighbour is a given empty point. That gets complicated fast, since the answer depends on which point is being tested.

I'm checking whether a chain is in atari by asking if every stone in it has no empty neighbour except the one point being tested. The trouble is that this condition depends on which point is played, so a simple per-stone check won't work and the test has to be done across the whole chain.

I'm thinking about the p-dependence now: a stone next to p counts as bad only if it has a liberty other than p, while stones farther away are bad whenever they have any empty neighbour. So I'd propagate a reachability flag through connected opponent stones over four or five layers, which should cover most chains, and handle the stones adjacent to p separately.

I'm defining a bad-stone test that counts each opponent stone's empty neighbours, with the adjacent-to-p case treated separately, so the propagation can stay independent of the point being tested.

I'm working out how to decide that a chain has a single liberty. Stones with two or more empty neighbours rule out atari outright, but stones with exactly one empty neighbour are harder, since I'd also need to confirm they all share the same point.

I'm considering a max-and-min fingerprint over the liberty indices of a chain: if the chain's max and min liberty indices match, it has exactly one liberty. Propagating those maxima and minima across connected opponent stones would need max operations over up to five neighbours per layer, with the same-colour gating applied to each neighbour's value.

I'm weighing a max-and-min fingerprint over liberty indices, but propagating maxima across chains would cost too many units, so that's probably out. Simple local liberty counts, with explicit patterns for chains up to three stones, might cover most cases. I'm also thinking that large groups dying in game 78 came from surrounding and eye loss rather than tactical atari, so passing less may be the bigger gain.

I'm looking at game 92, where we pass from around move 85 while White plays a run of moves that ends with us losing the game. That points to passing itself as the main problem, so I'm considering a much lower pass threshold, around -20, so we only pass when every remaining move is an eye fill or a strong self-atari. Self-atari penalties and the territory values could still put some self-atari moves inside the opponent's area close to that threshold, which I'd need to check.

I'm now thinking about filling our own territory: the self-atari check won't fire there since our own stones are adjacent, so we could reduce a two-point eye to one and eventually leave a group with a single eye. I'm working through a three-point line region to see how the eye counts would change.

I'm now considering a rule that avoids filling a point when it would shrink a region to a single eye without splitting it, since filling an end point of a two-point eye space leaves the group with only one eye.

I'm settling on a "NoShrink" rule: an empty point whose only open neighbour is one other empty point, with the rest of its neighbours our own stones, gets a large penalty like an eye, since filling it would cut the region down to one eye. Interior points with two or more empty neighbours stay allowed, and I'm also checking whether edge points and their neighbour counts change this.

I'm now looking at how to detect self-atari when a new stone joins our own chains, which means counting the liberties the merged group would have. I'm considering a rough estimate: the new point's empty neighbours plus the liberties of each adjacent friendly chain, minus any shared ones.

I'm now checking a self-atari shortcut for when a new stone joins our own chains: if the point has at least two empty neighbours, it's safe, and an adjacent friendly stone with another empty neighbour besides the point also gives an extra liberty. I'm noting that two neighbours of the same friendly stone can't be adjacent to each other, so those extra liberties won't overlap in the way I feared.

I'm settling on a self-atari estimate for placing a stone into our own chains, scoring the new stone's empty neighbours plus any extra empties of adjacent friendly stones. The catch is that this only sees local neighbourhoods, so it may flag self-atari when the chain still has liberties further away.

I'm working out a self-atari estimate for a point p by summing, over each friendly neighbour, how many extra empty points it has beyond p, using a per-edge unit that is zero when the neighbour is not ours. Where a friendly neighbour already connects to other friendly stones, I'm treating its chain as having unknown liberties and assuming it's safe.

I'm building a self-atari estimate that sums, over each friendly neighbour of p, its extra empty points, with a unit per edge that stays zero when that neighbour isn't ours. I'm adding a penalty for neighbours that already link to other friendly stones, since their liberties can't be counted locally.

I'm now making the self-atari check depend only on each neighbour's own chain, so an empty neighbour adds nothing when the neighbour isn't ours. That gives a per-neighbour term I can sum over p's neighbours, and it reduces to the old check when p has no friendly neighbours.

I'm now thinking about adding a three-stone capture check for opponent groups, which would take around 382 units per colour. Layer 5 is already at 163 units, so fitting that in would mean reshuffling which layer holds the pass-through and the self-atari logic.

I'm working out the budget for each layer: 512 units, with about 244 already taken by the pass-through and accumulator, leaving roughly 268 for new features. Layer 6 can hold features that feed the logits directly, which raises the total to about 1850 across the network.

I'm now splitting the feature budget across layers: the Dp and Dm terms move to layer 2 since they only need the passed-through inputs, leaving layer 1 with own, opp, U, S, G and an opponent-atari feature. That pushes layer 1 to about 486 units and layer 2 over its limit once the eye and SG features are added.

I'm now working out whether a "NoShrink" point should also be folded into the eye feature, since filling a point with one empty neighbour and the rest our own stones can cut a region down to one eye. The idea is that a single unit could return 2 for a full eye, 1 for a shrinking fill, and 0 otherwise, using a negative weight of about 20 so the eye gets -40 and the shrink gets -20. That would save the separate 81 units. I'm checking whether it's too broad on edges and corners, where a point with two own neighbours and one empty neighbour might trigger it when it shouldn't.

I'm reconsidering the layout: the Dp and Dm features would fit better in layer 1 alongside G, which leaves layer 2 with too many units for its budget.

I'm trying to fit the features into the layer budgets, and I realise the absolute-value penalty on the territory difference can be written as a linear term plus a single relu unit, so I only need one unit per point instead of two. That frees up 81 units in layer 1. Layer 2 is still over budget, though, so I'll have to decide where the remaining features can go.

I'm now moving the self-atari and eye-shape features around to fit the layer budgets, and I'm planning to place the opponent-tromino capture check in layer 6, since it can feed the logits directly and there's room to spare there.

I'm now adding three-stone tromino capture checks, and I'm planning to put the pass-through, the new not-passing setup, the self-atari-to-atari bonus and the eye-shape features into the next version.

I'm settling the pass threshold around -12, so that eye-filling (-40) and no-shrink (-20) moves stay below it. A self-atari costs about 8 per liberty, which can still beat passing, so I'm wondering whether to accept self-ataris inside opponent space or hold back.

I'm settling the self-atari penalty at 10 and the pass threshold at -9, so self-ataris only win when they capture. Deep own-territory points at about -5 then sit above the pass threshold, so I'd keep filling our own space, while invasions into opponent territory stay at risk of being captured.

I'm settling on the implementation details for version two: the pass-through layout, the not-passing feature, the self-atari and eye-shape checks, and the tromino capture features in the output layer.



---

I'm checking the atari cases: a lone opponent stone with two liberties counts as an atari move, and one with a single liberty is a capture. A chain touching an opponent gets penalised heavily, and a three-liberty group scores zero. I'm now turning to how the tromino unit condition should be handled.

I'm checking the trominoes' unit condition against the bias term, and it works out: the remaining non-own neighbour has to be the empty point p, so the formula holds. I'm also confirming that the pair case and the own-group liberty count still behave correctly.



---

I'm noticing my win rate is 29% as Black and 82% as White, which points to the komi of 7 hurting Black. The opponent's White moves seem to cluster toward the bottom-left, and I'm looking at why my large groups keep dying in losses while my loose stones are being outplayed by its solid shape, with my groups short on eyes.

I'm checking why we keep passing, since the pass rate is 0.12 and the first pass came at ply 86. My guess is that every candidate move scores below the pass threshold, because the |D| term turns large in positions where White is surrounding our stones. Looking at game 20, our central group was being captured after move 85 while we passed, so the real gap seems to be liberty awareness for larger chains.

I'm weighing whether to make the pass threshold much lower, around -30, so we stop passing while the opponent attacks. The catch is that shrinking eye space and playing self-ataris before passing could be costly, so I'd want to raise the eye weight to keep those moves unattractive.

I'm thinking about the opening, where our spaced-out moves form thin shapes that the opponent cuts through, and whether that's acceptable against weak opponents. Next I'd like to check the results against the small_s2 setting, and I'm considering whether broader tactics could come from counting liberties over larger chains within a local window.

I'm working out how to tell whether a larger chain is in atari, using a heuristic: a chain has at least two liberties if some stone has two or more empty neighbours, or if two stones each have one empty neighbour with different liberties. The tricky case is a chain with one shared liberty point, which can make a false positive when two single-liberty stones actually point at different empty points.

I'm treating a chain with no stone having two or more empty neighbours as a low-liberty group, which is a reasonable proxy for atari. That can misfire on two-stone chains that actually have two liberties, but extending or attacking there is still often useful. For captures I'd give a large bonus, and a smaller one for ataris.

I'm working out a precise test for whether a chain is in atari: every stone with an empty neighbour has to touch the same single liberty, so I'd need to check that per candidate point. Since that depends on the point, I'm considering a two-flag propagation instead, one flag for stones with two or more empty neighbours and one for stones with at least one.

I'm weighing whether a count of single-liberty stones could pin down an atari at a specific point, since the stones adjacent to that point would have to be exactly the ones with one empty neighbour. The problem is that summing those counts across a chain double-counts through shared paths, while taking a maximum over stones does not, so I'd be stuck with an approximation.

I'm defining a "weak chain" flag for stones whose connected group has no stone with two or more empty neighbours, and I'm sketching a propagation scheme that spreads a "good" signal through same-coloured neighbours within a set radius.

I'm working out a way to count liberties over larger chains by propagating a "good" flag, with a clamp so the value stays a usable threshold.

I'm working out how to count liberties over larger chains with a "weak chain" flag: a stone is weak if its connected group, within a set radius, has no stone with two or more empty neighbours. I'm sketching a propagation step that spreads this "good" signal through same-coloured neighbours, using relu arithmetic so the values stay integer and no clamp is needed.

I'm working out whether a neighbouring stone's chain has at least two liberties, so that extending at this point adds real breathing room. The chain check would use the propagated "good" flag instead of the local count, to see whether our own group is in danger after the move.

I'm checking whether the chain-liberty approximation holds for big groups, since propagation only reaches a few steps out, so a long chain whose liberties sit far away could be misread as weak.

I'm working out the layer budget: passthrough and propagation features would take about 406 of the available slots per layer, which leaves too little room for the existing UP, SP and TRI features. I'm considering whether the weak-chain logic can replace the U, S, UP, SP and TRI features, while keeping capture and atari bonuses distinct.

I'm checking whether a wall of stones along the edge, where each stone has only one liberty on the first line, would be misread as weak, when in fact the chain has several liberties spread across it. A simple fix would be to treat a stone as having more liberty when a neighbouring stone in the same chain has a different empty point next to it.

I'm checking whether two adjacent chain stones that each have one empty neighbour must have distinct liberties, which would guarantee at least two liberties. Stones in a diagonal arrangement might share one, and I'm working through the case where two stones sit two steps apart with a stone in between.

I'm defining a "good" flag for a stone that has two or more empty neighbours, or that has one empty neighbour and a same-coloured neighbour that also has one, since two adjacent stones like that must have distinct liberties. Doing this needs an extra layer, because the test has to look at the neighbours' liberty counts.

I'm reworking the clamp for the propagated "good" flag, using a min built from relus over the neighbour counts, and checking how many layers that costs.

I'm trying to make the propagation layer compute a flag for stones that have at least one liberty and a same-coloured neighbour that also has one, but a clamped indicator costs too many units in the first layer, so I'm considering deferring that check to the next propagation step instead.

I'm trying to get Good_0 to come out right with a second layer: the first layer gives me the E≥2 flag and a value H equal to the empty-neighbour count, and the second layer checks whether H of this stone plus its neighbours' H values is enough to show two distinct liberties.

I'm trying to make the "good" test work for a stone whose neighbours might also have liberties, so I'm considering clamping each neighbour's liberty count in the first layer and summing those clamped values in the second, so that a single neighbour with a liberty is enough to trigger it.

I'm now working out a two-layer test for whether a stone's own group has room to breathe, using a clamped 0/1 indicator for each neighbour that is own-coloured with at least one empty point, so the neighbour sum can be combined with the stone's own flag without the weights blowing up the layer budget.

I'm checking the unit budget for the first layer, since the Ga, H and H2 features for both colours would take 486 slots, which with the passthrough comes to 648 and is too many.

I'm noting that the first-layer unit count already works out: the two features per colour, H and H2, are the same as the Ga term I'd planned, so the layer is full at 486 with no room for the G, U, S or Dp features. I'm thinking the Dp feature could move into the second layer, and the U feature may be kept there too, if the weak-chain logic doesn't cover it.

I'm laying out the layer budget further: a third layer would hold the own and opponent planes, the accumulator, and the next "good" flags, with an eye-status feature bringing it to about 487 units. The same pattern would continue in later layers, leaving only a little spare room each time.

I'm working out how the weak-chain flags feed into the later layers, with the weakness of a group computed from its neighbours' flags in a fifth layer and the combined check left for the sixth.

I'm working out how a stone's weakness can be read from its chain's liberties, so the weak flag would come from the neighbours' Good2 values in the fifth layer. Self-atari checks are the next thing I want to set up, starting from whether a newly placed stone would join an own group with a good flag.

I'm working out the next layer's budget: a fifth layer would carry the own and opponent planes plus the propagated good flags for own groups, with the weakness flags for both colours computed alongside. The self-atari check for a new stone would then use those own-group flags in a sixth layer, scaled so that any neighbouring good chain is enough to count the move as safe.

I'm now budgeting the sixth layer: the acceptance and self-atari features, plus a save-conjunction, fit in about 244 units with room to spare, and a capture term can be added linearly to the accumulator from the weak opponent flags. What's left is deciding how to tell a true atari capture from a chain that still has two or more liberties.

I'm checking whether a weak opponent chain adjacent to a candidate move can be captured, and whether the self-atari penalty still leaves the move net positive. I'm also noting that a save only counts when the own chain is weak and the move gives the placed stone at least two liberties.

I'm working out the unit budget for the fifth and sixth layers, keeping the outputs for the weak and self-atari features within the layer limits. I'm planning to drop the separate capture and atari terms in favour of a weak-chain flag, and to hold the exact single-stone checks in a fourth or fifth layer where there is still room.

I'm settling on the eye logic: an eye costs -2 per unit of weight and shrinking an eye costs -1, so I'm setting the pass threshold to -15 and the eye weight to 20 so the shrink penalty stays below it.

I'm settling the weights for the self-atari penalty and the pass threshold, deciding to make self-atari costly enough that I'd rather pass than play it, and then I'll write out the build.



---

I'm checking that the propagation gating formula only pulls in neighbour values from the same colour, and that the own-stone connectivity assumption holds. I'm also noting the values might grow large, but that seems acceptable. Next I'm turning to the weak-layer term for L5.

I'm checking the win-chance penalty for a capturing move: if the move's own point has no empty neighbours, the penalty outweighs the capture bonus, so captures get discouraged. That's a problem, since captures often happen exactly at points like that, so I'm thinking the fix is to stop counting opponent neighbours that are being captured as blocking liberties.

I'm adjusting the suicide-avoidance term so it also subtracts for weak opponent neighbours that a capturing move would take, and I'm weighing the capture and pass weights against that change.



---

The win rate sits at 69% against the large setting now, with Black at 59% and White at 79%. As Black I'm losing because our stones are packed in the top-left while White takes the big bottom-right area, and I'm wondering whether filling our own territory early wastes tempo while boundary moves are still open.

I'm weighing an asymmetric influence term that penalizes deep-own-territory moves more heavily than invasive ones, since the opponent is weak and their self-ataris make invasions less risky. I'm also considering a stronger prior for third-line moves early on, and I'm looking at the margin spread, which is wide with big losses.

I'm noting that the large network keeps favoring a few specific points, which makes its play look fairly greedy. Next I want to try the asymmetric influence and maybe a wider kernel, while also checking how we fare against small_s2.



---

I'm checking why we passed so often in game 94: White's moves keep killing our large central group, but every move we had scored below -12. That seems to come from the self-atari penalty on most of our options, plus eye-space and dead-stone penalties.

I'm weighing whether to make passing much rarer by setting its value at -40, so we'd play self-atari moves instead. A self-atari stone next to an opponent stone could still shrink our group's liberties in a capturing race, so that trade-off is unclear.

I'm considering whether to skip writing a board-reconstruction tool to replay game 94 and inspect the position around move 80, since replaying a record could look like running a game engine. I'll reason about it directly instead. My main goal is to lower Black's win rate under komi 7, which means winning by at least 8 points, and to avoid passing too early in positions that look lost, because that lets the opponent roam freely inside our area.

I'm working out why a lone intrusion wouldn't trigger a response: a stone inside our area scores around zero, which beats the pass value. So passing mostly happens when every remaining move is a self-atari or a deep point, which suggests our large group had already been enclosed before we passed. The real problem is earlier, when the group got surrounded, and that's where I should focus on keeping groups connected and securing territory.

I'm weighing a frontier heuristic that scores moves where both influences are positive, and I'm thinking about testing asymmetric weights so we penalize our own dominated areas more and go easier on the opponent's.

I'm considering a connection bonus that rewards a move joining two of our own chains or cutting two opponent chains, and I'll test a lower pass threshold together with the asymmetric weights.



---

The jump to 87% against the large baseline is notable, so I want to widen the asymmetry: trying a=2 and b=0.5 next. I'll check the other settings afterward, with infa=2.0 and infb=0.4 as the first run.



---

I'm weighing whether to keep P4 or P5, and since they score the same I'll check small_s2 with P5, or P4 if that's easier. Meanwhile I'm looking at what else could be improved: the opponents rarely take captures, and our chains escape only about 59% of the time in atari, though escaping a truly dead ladder is pointless.

I'm now checking small_s2 and small_s0 with P5 to see where things stand overall, since the next step is testing those two seeds.



---

I'm looking at a 71% small-board win rate and tracing the losses: in game 20 we pass repeatedly while White keeps capturing, and our groups die after our passes. I want to know whether they were already dead or whether we could have defended them, and I'm wondering why all our moves seem to score below -20.

I'm now weighing whether a pass was forced: with our groups short of liberties, most candidate moves would shrink our own eyespace, which the scoring penalizes heavily, so passing ends up looking better than any move. I'm also checking a deep own-territory point, which scores fine, and then going back to game 20 to reconstruct the stone positions around the point where we passed.

I'm weighing a lower pass threshold of -40 so we'd always play something besides eyes. Self-atari moves look more acceptable since the opponent only captures about a third of the time, though filling our own liberties could hand them the group, so I'd only use those when nothing else scores. There's also a catch: at -40 the eye-shrinking moves at -20 would beat passing, so we'd erode our own eyespace first.

I'm working out the pass threshold: with the eye-shrink penalty at -30 and the self-atari penalty at 15, those moves score around -25 to -12, so a pass at -26 would let most of them get played.



---

I'm still passing a lot while getting killed, and the pass threshold of -26 keeps firing, so every move must be worth less than -26. Shrinking the eye costs -30, the shrink-attack at SA=2 is another -30, and SA=1 is -15, with a D penalty on top. I'm wondering how big that D term gets, since in our large area it could reach around -12.

I'm now going through the game 110 position, where after W1 we pass with white stones all around. I'm listing our stones and noticing that the empty points may all count as shrinkable eyes, so the legal moves would only be in the opponent's area. Filling our own eyes would leave us with only passes while white closes in, and I'm questioning whether our groups had enough real eyes to survive.

I'm checking whether the checkerboard filling in our own territory left us with only single stones rather than connected chains. Each empty point surrounded by our stones would act as an eye, but a point touching an opponent stone could be a false eye, and that weakens the whole group's survival.

I'm weighing whether a replay to reconstruct the board is allowed, since it might amount to playing out moves, and I'm unsure whether that crosses the line against simulating Go games. Reconstructing a recorded game just to view it is different from testing a policy, but the rule's wording makes me hesitant. I'm also noting that the filling inside our area happens while boundary gaps remain open, so I can't settle the diagnosis without a visual of the board.

I'm going through the early moves of game 110 and noticing White building a connected wall across the top and center, with stones at 21, 22, 23, 20, 13, and 11. Our stones look scattered by comparison, and I'm not sure whether 30 has been captured already.

I'm considering a connection term that rewards placing a stone next to one of our own, since our loose stones keep getting cut by White's solid wall. The current adjacency feature is zero and the distance penalty pushes us away from our own stones, so the a=2 asymmetry may be making things worse. I want to try a small bonus of about +0.5 to +1 for solid extensions, plus a cut-protection feature for empty points sitting diagonally between two of our stones.

I'm considering a cut-point feature for the 2x2 squares around each empty point: a bonus when our two stones touch p diagonally to each other and the opponent holds the opposite corner, plus a matching bonus for cutting. That is about 324 units of each kind, which doesn't fit alongside the rest in L6, so I'm leaning toward putting only the connect feature there.

I'm working out a unit for each 2x2 square: two diagonal own stones plus an opponent on the other diagonal, with the empty point being the one that gets the bonus. Since the empty point's identity doesn't matter if I apply the bonus to both opposite-diagonal points, I'm checking whether a relu over the sum of those four inputs minus 2 captures the pattern.

I'm checking a 2x2 cut-point unit: two diagonal own stones plus an opponent on the other diagonal, with the remaining empty point being the one that gets the bonus. A relu over those inputs minus a threshold should fire only in that exact shape, so the bonus lands on the right point.

I'm thinking about the cut feature as a relu over the four diagonal inputs, so it fires only when our two stones sit on one diagonal and the opponent holds the other, which gives roughly 256 units that fit in L6. I'm also weighing whether a moderate weight on the cutting version is worth it, and whether pulling the asymmetry parameter back toward 1.5 might help despite the noise in the results.

Maybe passing isn't the real problem, since the positions are already lost by then. I'll implement the connect and cut features with weights of 3 and 2, and try the asymmetry at 1.5 and 0.7 with no own-adjacency bonus.



---

The 67% result showed no real change, within noise. The opponent's opening looks fixed, with a solid top wall in both colors, so the replies seem deterministic for a given opening. I'm also checking game 59, where we play White and the move sequence is still being recorded.

I'm looking at game 59, where the group of white stones in the center seems to have been captured after a series of ko-like exchanges around stone 40. Black filled in around the group while we passed, and it looks like we had too few liberties and no real eyes. I'm considering a liberty-aware heuristic so that weak chains get extended before they're surrounded.

I'm weighing whether replaying a returned record to reconstruct the board counts as simulating games. I think it doesn't, since the record already holds the network's play, but I don't want to run anything that plays out moves, so I'll skip that route. I'm also considering the weighting: heavily penalizing own-dominated points may discourage reinforcing our own groups, which is what pushed us into invasions. That helped a lot against large (55→87), but against small_s2 it stayed roughly even, and that opponent is the sharper tactical player.

I'm weighing how the self-atari check handles a move next to our own chain: if that chain is already in good shape with two or more liberties, playing into it might still leave us with only one liberty after the join, so the current rule could miss it. I'm also noting that our self-atari rate of 7-8% seems high, and that a 61% ladder-escape rate may be linked to the same gap.

I'm noting that the self-atari check could miss a move beside our own chain if that chain's liberties overlap with the point being played, and I'm weighing whether a stricter threshold would fix that without creating false positives. I'm also still thinking about whether the opponent's policy comes from RL or imitation, since their shape play is fairly strong, and my current estimate sits around 91.

I'm checking whether the small_s2 results really differ from one another, since the variants all land near 68-70% within noise. Game 68, where we're Black and lost by 39, is the next thing to look at, to see where our moves went wrong.

I'm going over the game 68 move list, where White's stones build up along the top rows while we take the bottom and center. The score looks close if our bottom area holds, but White ends up owning almost everything except the bottom-left, so their later moves must be invasions into what I thought was ours.

I'm looking at the row-4 pattern in game 68, where our stones sit in a checkerboard with White's stones cutting through the gaps. The loose diagonal shape seems to be what let the invasion split our group.

I'm now weighing a gap in the influence penalty: a point next to one of our stones picks up an own-influence cost, so the heuristic pushes our stones apart into one-space jumps, and the opponent slips through the gaps. The 2x2 cut pattern doesn't catch the wedge at 39 between stones at 38 and 40, and I'm considering whether softening the penalty near our own stones would let us play more connected shapes, even though the symmetric version did worse against the large opponent.

I'm weighing a shape bonus for one-point jumps, so that when the opponent wedges into a gap between two of our stones we can still connect against it. The gap-filling idea is more complicated, though. Given the roughly ±4% noise, another fifteen passes of tuning would tell me little, so I'd focus on the changes most likely to hold up. Our self-atari rate of about 7% still seems high next to the opponent's, so tactical accuracy looks like the place to start.

I'm considering whether the large-opponent losses still look the same as the small_s2 ones, and whether testing against a different opponent like small_s0 would show if the ~90% result holds. I'm also thinking about whether the escape rate and ko handling are worth changing, and whether tuning the weak-chain capture weight would help when the opponent's chain has two liberties.

I'm going to try a stronger own-adjacency bonus to encourage solid connections, and check its effect on the small_s2 results with connection and cut weights set to 4 and 2.



---

The match is up to 81% against small_s2, from about 68%, so the connection change is helping. I'll push a bit further by trying ownadj at 1.5 and con at 5, and I'll recheck against the large model with this config later.



---

At 89%, I want to push further. I'll test ownadj at 2.2 and con at 6, and check the large case too, starting with ownadj=2.2 against small_s2.



---

I'm comparing the 99% result against small_s2 first, and then I'll run the same check on the large model and small_s0.



---

The large variant came in at 94.5%, so I'm moving on to the small_s0 variant. After that I'll see whether pushing ownadj further is worth it, and I'm starting with a test on small_s0.



---

I'm at 99% on s0 and s2 and 94.5% on large, so the remaining losses are all as Black against large. With 10 passes left, I want to test a slightly higher ownadj and con (2.8 and 7) against large, though there's a risk I'm just fitting noise. I'll also keep in mind that the opening seeds are held out and the policy is deterministic apart from ties. Next I'll run P11 with ownadj 2.8 and con 7.

Testing that setting against large now.



---

Black's win rate has dropped to 73%, down from 90.6% at P10, which could partly be noise given the small sample of 64 black games. I'm planning to go back to the P10 values and stay with that as the base. Next I want to look at the black losses against large, where the White moves cluster around the bottom-center.

I'm checking the black losses against large, where White keeps taking the bottom-center and the right side while Black's stones stay in the top-left. Looking at game 106 under P10, Black's stones near the bottom get swallowed as White builds a solid wall there, so the large opponent's framework seems to be the main issue. I have nine runs left and I'm weighing whether to tune the pass threshold or just confirm P10 with more runs, since it already performs well and changes risk regression.

I'm weighing a test of a heavier cut weight, going from 2 to 4, against the large opponent, since the noise makes tuning ownadj or confirming P10 less informative. If Black's win rate improves, I'd keep it.



---

Dropping the cut=2 change, since at 90% it sits within noise of the P10 baseline of 94.5%. I'm also weighing the self-atari issue (6%): the SA check already covers the played stone's liberties and weak opposing neighbours, so the bigger question might be how the pass logic behaves at the end of the game.

I'm considering whether colour can be inferred from stone counts, but captures would throw that off, so I'll skip it. I'm also thinking about whether a diagonal-adjacency feature adds anything beyond what the connection feature already covers. The loss games keep showing the same opening pattern from the large opponent, so I'm weighing a couple more experiments before settling on the final configuration.



---

Both variants land around 90%, so the 94.5% from P10 was probably luck. With n=128 the standard error is about 2.4%, which puts that gap at only about 1.5σ, so it doesn't settle anything. I'd rather look for structural changes than keep tuning knobs. The losses come almost entirely from games where I'm playing Black and komi counts against me, and I need to find a way to win the territory battle as Black.

I'm checking whether the endgame passing and dame filling are handled correctly, and they look fine. Ko and seki don't seem to be the problem either. The bigger risk is late opponent invasions inside my territory, or my own groups dying, and I can't judge that without a closer look at a specific loss. I'm pulling up one Black loss from P11 to reconstruct the final board and see where the territory went.

I'm pulling apart one of the Black losses from the board layout, tracing how the early stones got captured and why the wall in the centre formed. The local fights seem to favour White whenever it's ahead, and my loose stones keep getting cut off. I'm starting to wonder whether my connections or shape are the real weakness, or whether the opening itself is what I should change.

I'm considering whether to stop tuning and test a different weight balance for the asymmetric terms, since the earlier a/b test without ownadj came out worse for Black. I'm also noticing that about 6% of my moves are self-ataris, roughly four per game, which hands the opponent a capture, and I want to find where those come from.

I'm checking the self-atari moves next: some come from the weak-chain capture bonus, which cancels the self-atari penalty when a stone lands next to a weak opposing group. Others happen late in the game, where a stone placed inside the opponent's territory might not get captured since they only take it about 40% of the time, and that still disrupts their area.

I'm weighing whether to tweak the opponent-side penalty, moving it from 0.7 to either 1.0 or 0.4, since the ownership adjustment has changed how aggressive the play is. I can't tell which direction helps, so I'm thinking about whether to test it at all.

I'm thinking about making the opponent-side penalty more cautious, setting it to 1.0 instead of 0.7, since being more invasive might be what's getting my groups killed.



---

Performance is holding around 90 to 91 percent, so this looks like a plateau. The 94.5 percent from P10 may have been partly luck, and six passes remain. I'm considering a structural change: Black seems to underperform because of komi, so Black may need a bigger margin. I can't reliably tell the colour, but I could estimate it by counting stones, since the side to move should be roughly even or one ahead depending on who is playing.

I'm dropping the komi-based colour estimate as unreliable and turning to a concrete loss from the P10 record, game 46, to look for a fixable tactical mistake.

I'm going through the moves of game 46 by hand, and White's stones have built a wall along row 6 and pushed into the centre, while the right side and top get invaded. Black's moves around the top-left corner, like 20, 24, 16, 13 and 10, look like the places to check for a tactical slip.

I'm going back over the moves of game 46 and found that the record is hard to follow closely enough to pinpoint a clear slip. The loss seems to come from White's wall along the bottom and its invasions on the right and top, which I may have answered too slowly. I'm also considering whether a single-stone atari bonus would help, since the opponent escapes only a little over a third of the time, though my current capture logic doesn't treat two-liberty stones as weak.

I'm weighing an atari bonus on single opponent stones with exactly two liberties, since these opponents escape only about a third of the time. The concern is that a stone that escapes could leave my shape worse, but the capture upside seems worth testing. Layer 6 is full apart from about 12 spare units, so placing the bonus units there is the open question.

I'm replacing the single-capture units in layer 4 with atari units, scoring a lone stone with two liberties at weight 2 so the capture bonus stays the same. The corner case matters here, since a corner stone with two empty neighbours always qualifies and playing one of them leaves it in atari.

I'm checking that the self-atari case is covered: the atari stone could itself be left in atari, which the self-atari penalty handles, and that penalty isn't cancelled since the opponent isn't weak. I'll try the variant with the atari units replacing the single-capture units in layer 4 at weight 2.



---

The 86% result looks worse, possibly just noise. I'll go back to P10 and use the remaining passes on one experiment with a bigger chance of helping Black, since confirming P10 against the large variant would mostly show they're within noise.

I'm weighing a new prior that favors the third line over the center, since territory might matter more for Black, and I plan to test it against the large variant to see if Black improves.



---

The center-weighted prior did worse, landing at 83%. Since a center prior helps, I'll try a stronger one, prior [0,1,2,2.6,3.2]. That leaves 4 tries, and if it wins I'll need to check it against s0 and s2 as well.



---

I'm going with P10 since it tested best on all three sets (99, 99, and 94.5). I'll save it and check that the saved weights match the build, then run one pass against the large set to confirm the save worked.



---

The check came back at 91.4% confirmed, and P10 is the final pick and is saved. I'm stopping here, leaving the two remaining passes unused.

