I'm trying to figure out which well-known environment has 10 inputs and 5 actions. I'm ruling out the common ones I can think of, like Acrobot and LunarLander, since their dimensions don't match, and I'm wondering whether the input might be padded.

I'm reasoning about whether the 10-input, 5-action network is a padded version of a smaller environment, since the policy head's columns 1, 3, and 4 look barely trained while columns 0 and 2 carry the real weight. I'm also considering that the near-init scale could be a sign of an output that never receives meaningful gradient.

I'm checking whether the weights in the first layer carry any signal about which inputs are padding, but the untouched rows look the same as their initialization, so that doesn't settle it. The non-zero biases on columns 1, 3 and 4 do suggest those outputs were trained rather than left at zero, which leaves the padding question open.

I'm still weighing whether the 10 inputs and 5 actions point to a padded setup, since columns 1, 3 and 4 of the policy head barely moved from initialization, while their biases did shift. That could mean they're real actions the policy learned to suppress, or outputs that only ever see gradient through the bias.

I'm weighing MountainCar against Acrobot, since both use a bang-bang pattern between actions 0 and 2 and skip action 1. The biases on 0 and 2 are nearly symmetric, and the bias on action 1 is negative, which doesn't settle it, because trained policies in either environment can behave like this. I'm wondering whether checking the first-layer row norms would show which inputs are carrying signal.

I'm comparing the first-layer row norms, which look about the same across rows, so that doesn't separate the padded inputs from the real ones. I'll try a rough linear approximation of how the inputs push the logit difference between actions 0 and 2.

I'm sorting the hidden units by the sign of their contribution to the action 0 versus action 2 difference, which gives roughly seventeen units favoring action 0 and the rest favoring action 2. Doing a full propagation through both hidden layers is expensive, so I'm considering a cheaper sampled approximation instead.

I'm weighing MountainCar against Acrobot again, since the bias on action 1 is more negative than on actions 3 and 4, which makes me wonder whether action 1 is being treated differently from the padded ones.

I'm leaning toward Acrobot over MountainCar, since the weights look like a modest training run and Acrobot is easier to train. To check which inputs matter, I'd look at the angular velocity inputs for Acrobot or the velocity input for MountainCar, and I'm planning to work out the hidden-unit contributions to see which one the policy actually depends on.

I'm assigning each of the 23 hidden units a sign and rough weight based on how much it favors action 0 over action 2, dropping the near-zero unit and halving a few smaller ones.

I'm assigning each of the hidden units a sign and a rough weight based on which action it favors, with the small or near-zero ones scaled down.

I'm working through the weights for the hidden units, flipping the sign on the terms that favor action 2 and tallying up how much the first row of the second layer pushes toward action 0.

Summing the weighted contributions across the hidden units for the action 0 versus action 2 direction, one term at a time.

Summing the weighted contributions across the hidden units for the action 0 versus action 2 direction, one term at a time.

Summing the first row of the second-layer weights against the sign-weighted hidden units, the total comes out to roughly -4.25. That is far larger than a random initialization would produce, so the backprop-aligned structure is real. Working through all 32 rows this way is tedious, but it should show which inputs the policy actually depends on.

Still working through the first-row sums against the sign-weighted hidden units...

Still summing the first-row contributions across the hidden units...

Still summing the first-row contributions across the hidden units...

Still summing the first-row contributions across the hidden units...

Still summing the weighted contributions for the remaining hidden-unit rows...

Still summing the weighted contributions for the remaining hidden-unit rows...

Still summing the hidden-unit contributions, running through the remaining terms for this row.

Still summing the row-4 contributions across the hidden units...

Still summing the row-4 contributions across the hidden units, and the running total is drifting negative.

Still tallying the row-4 contributions across the hidden units, and the running total keeps sliding negative.

Still summing the row-5 contributions across the hidden units...

Still summing the row-6 contributions across the hidden units...

Still summing the row-6 contributions across the hidden units...

Still summing the row-6 contributions across the hidden units, and the running total keeps falling.

Still summing the row-7 contributions across the hidden units...

Still summing the row-7 products, and the running total is now around -2.19.

Still summing the row-8 products across the hidden units...

Still summing the row-8 products, and the running total is now around -0.53.

Still summing the row-9 products across the hidden units...

Still summing the row-9 products across the hidden units...

Still summing the row-9 products and the row-10 terms, and the running total is around 1.35 so far.

Still summing the row-10 products across the hidden units...

Still tallying the row-10 products, and the running total ends up near -2.9.

Still summing the row-11 products across the hidden units...

Still tallying the row-11 products, and the running total is climbing to about 4.33.

Still summing the row-12 products across the hidden units...

Still summing the row-12 products across the hidden units...

Still tallying the row-12 and row-13 products, and the running total keeps drifting around the low negatives.

Still tallying the row-13 products, and the running total is sinking toward about -2.6.

Still summing the row-13 and row-14 products, and the running total has dropped to about -2.97.

Still tallying the row-14 products, and the running total is sliding down toward about -4.

Still summing the row-14 and row-15 terms, and the running total has settled near -2.4.

Still summing the row-15 products across the hidden units...

Still summing the row-15 products, and the running total has climbed to about 3.0.

Still summing the row-16 products across the hidden units...

Still summing the row-16 products, and the running total is climbing toward about 1.7.

Still tallying the row-17 products, and the running total is hovering around 2.7.

Still summing the row-17 products across the hidden units...

Still tallying the row-18 products, and the running total is drifting around the low positives.

Still summing the row-18 products, and the running total is shifting around the low values.

Still summing the row-18 and row-19 products, and the running total is near 1.33.

Still tallying the row-19 products, and the running total is hovering around 1.3.

Still tallying the row-19 products, and the running total is now around 2.1.

Still summing the row-20 products across the hidden units...

Still summing the row-20 products, and the running total has climbed to about 1.15.

Still summing the row-21 products across the hidden units...

Still summing the row-21 products, and the running total has dropped to about -0.69.

Still summing the row-21 and row-22 products, and the running total has dropped to about -0.64.

Still tallying the row-22 products across the hidden units...

Still tallying the row-22 products, and the running total has settled near -0.44.

Still tallying the row-23 products across the hidden units...

Still tallying the row-23 and row-24 products, and the running total is settling near 0.37.

Still tallying the row-24 products across the hidden units...

Still tallying the row-24 products, and the running total has dropped to about 1.06.

Still summing the row-25 products, and the running total is drifting toward about 0.5.

Still tallying the row-25 products, and the running total has slipped to about -0.55.

Still tallying the row-26 products across the hidden units...

Still tallying the row-26 products, and the running total is coming to about zero.

Still tallying the row-26 and row-27 products, and the running total is sliding back down toward about -0.62.

Still summing the row-27 products across the hidden units...

Still summing the row-27 products, and the running total has fallen to about -3.6.

Still working through the row-28 products, and the running total is drifting around the low values.

Still summing the row-28 products, and the running total has dropped to about -2.95.

Still working through the row-29 products, and the running total is climbing again.

Still summing the row-29 products across the hidden units, and the running total is climbing toward about 1.13.

Still working through the row-30 products, and the running total is climbing toward about 3.4.

Still summing the row-30 products across the hidden units, and the running total is climbing toward about 0.96.

Still tallying the row-31 products, and the running total is near 3.3.

Still tallying the row-31 products, and the running total is sliding downward.

Still tallying the last hidden-unit products for row 31, and the running total settles at about -4.41.

I'm now computing the hidden-layer gradient terms g_k for each input row, starting with row 0.

Still tallying the row-0 products across the hidden units...

Still tallying the row-0 products, and the running total has climbed to about 2.55.

Still working through the row-1 products, and the running total is shifting around the low values.

Still summing the row-1 products, and the running total has dipped to about -2.3.

Still summing the row-1 terms, and the running total has fallen to about -8.8. Now I'm moving on to the row-2 products.

Still tallying the row-2 products, and the running total has fallen to about -8.9.

Still working through the row-2 products, and the running total has settled near -2.1. Now I'm starting on row 3.

Still working through the row-3 products, and the running total is shifting around.

Still tallying the row-3 products, and the running total is climbing toward about 5.9.

Still working through the row-4 products, and the running total is starting to build.

Still tallying the row-4 products, and the running total is climbing toward about 2.6.

Still summing the row-4 products, and the running total is climbing toward about 13.9. Now I'm working through the row-5 products.

Still tallying the row-5 products, and the running total is sliding toward about 1.1.

Still summing the row-5 products, and the running total has slid to about -19.5. Now I'm moving on to the hidden-layer terms for the rest of the rows.

Still tallying the row-6 products, and the running total is shifting around the low values.

Still tallying the row-6 products, and the running total is climbing toward about 1.2.

Still working through the row-7 products, and the running total is shifting around.

Still tallying the row-7 products, and the running total is drifting toward about -2.5.

Still tallying the row-7 products, and the running total has settled near 1.5. Now I'm working through the row-8 values.

Still tallying the row-8 products, and the running total is shifting around.

Still summing the row-8 products, and the running total is settling near 0.78.

Still working through the row-9 products, and the running total is shifting around.

Still tallying the row-9 products, and the running total has dipped to about -2.05.

Now I'm reading off the row-9 totals and the summary of hidden-layer terms, with inputs 1, 3, 4, and 5 standing out as large and the rest falling within the noise.

I'm settling on Acrobot: the strong terms are on inputs 0 through 5, with the rest in the noise, which fits its six-dimensional observation. The large weights on the two angular velocities, with opposite signs, along with the sine terms, match Acrobot rather than MountainCar, which would only use two inputs.

