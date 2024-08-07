import numpy as np

from topics_causal_inf.utilities import data_wager_athey_2018

data_wager_athey_2018(10_000_000, 10, "dgp3", np.random.default_rng())
