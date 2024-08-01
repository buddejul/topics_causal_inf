from sklearn.linear_model import ElasticNetCV  # type: ignore[import-untyped]

from topics_causal_inf.generic_ml.generic_ml import generic_ml
from topics_causal_inf.utilities import data_wager_athey_2018

n_obs = 10_000
dim = 10

data = data_wager_athey_2018(n_obs, dim, "dgp3")

ML0 = ElasticNetCV()
ML1 = ElasticNetCV()

ML_LEARNER = (ML0, ML1)

res = generic_ml(
    data,
    n_splits=10,
    alpha=0.05,
    strategy="blp_weighted_residual",
    ml_learner=ML_LEARNER,
)
