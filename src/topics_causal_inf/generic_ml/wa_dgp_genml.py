from sklearn.linear_model import LassoCV  # type: ignore[import-untyped]

from topics_causal_inf.config import RNG
from topics_causal_inf.generic_ml.generic_ml import generic_ml
from topics_causal_inf.utilities import data_wager_athey_2018

n_obs = 10_000
dim = 10

data = data_wager_athey_2018(n_obs, dim, "dgp3", rng=RNG)

ML0 = LassoCV()
ML1 = LassoCV()

ML_LEARNER = (ML0, ML1)

res = generic_ml(
    data,
    n_splits=10,
    alpha=0.05,
    strategy="blp_weighted_residual",
    ml_learner=ML_LEARNER,
)

res_ht = generic_ml(
    data,
    n_splits=10,
    alpha=0.05,
    strategy="blp_ht_transform",
    ml_learner=ML_LEARNER,
)
