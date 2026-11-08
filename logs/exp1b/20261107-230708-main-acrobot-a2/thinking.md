I'm checking whether the network is actually trained: the output weights show a strong pattern, with columns 0 and 2 anti-correlated and columns 1 and 3 close to zero. That structure looks suspicious, so I'm comparing it against what I'd expect from LunarLander's action mapping, where the engine actions shouldn't line up like that.

Checking whether the bias terms and output weights point to a real trained policy here, or something degenerate. Columns 1 and 3 barely move, so the choice seems to come down to main engine versus noop. The first layer weights look orthogonally initialized, and the output pattern isn't what random init would produce, so I'm leaning toward LunarLander with moderate confidence.

