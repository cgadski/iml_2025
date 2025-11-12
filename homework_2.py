# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 1: Gridworld Environment

# %% [markdown]
# ### (a)
# > For `dungeon = make_ring_dungeon()`, compute our probability distribution
# > over states after starting at `dungeon.start` and making 10 random actions.
# > Display it using `dungeon.show()`.

# %% [markdown]
# ### (b)
# > For `dungeon = make_ring_dungeon()`, compute the exact state values
# > $V_\pi(i)$ under the random policy. Display them.

# %% [markdown]
# ### (c)
# > In whatever notation you prefer, write down the _Bellman relations_ that
# > characterize the optimal state values $V(i).$

# %% [markdown]
# ### (d)
# > **Bonus**: Compute and display the state values $V(i)$ for
# > `make_big_dungeon()`. Using the state values, compute an optimal policy for
# > the big dungeon and display it using `dungeon.show_policy()`.


# %% [markdown]
# ## Part 2: Tic-Tac-Toe

# %% [markdown]
# ### (a)
# > Using the language of state value functions, explain how you would normally
# > solve tic-tac-toe. What is the meaning of the value of a position? What
# > would you say is the value of the starting state?

# %% [markdown]
# ### (b)
# > Implement an agent that learns tic-tac-toe through Q-learning. By whatever
# > means necessary, ensure that the trained agent wins at least $99\%$ of the
# > time in a trial of $100,000$ fair games against the random opponent.

# %% [markdown]
# ### (c)
# > When playing against the random opponent, are some choices of starting moves
# > better than others?


# %% [markdown]
# ## Bonus: Multi-Armed Bandits

# %% [markdown]
# ### (a)
# > Implement Thompson sampling for this problem. Over $100$ independent trials
# > with $5000$ iterations each, report the empirical fractions of times that
# > the agent chose the special arm.


