import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import (  # type: ignore[import-untyped]
    ElasticNetCV,
)

from topics_causal_inf.config import RNG
from topics_causal_inf.generic_ml.generic_ml import generic_ml

N_SPLITS = 1

ML_LEARNER = (ElasticNetCV(), ElasticNetCV())

data = pd.DataFrame()

n_obs = 10_000
tau = 2  # ATE

data["d"] = RNG.choice([0, 1], size=n_obs)
data["y"] = RNG.normal(size=n_obs) + data["d"] * tau
data["p_z"] = np.ones(n_obs) * 0.5
data["z1"] = RNG.normal(size=n_obs)
data["z2"] = RNG.normal(size=n_obs)

res_nohet = generic_ml(
    data,
    n_splits=N_SPLITS,
    alpha=0.05,
    strategy="blp_ht_transform",
    ml_learner=ML_LEARNER,
)

# ======================================================================================
# DGP with heterogeneous treatment effects
# ======================================================================================
data = pd.DataFrame()

n_obs = 10_000
tau = 2  # ATE

z_dim = 10

data["d"] = RNG.choice([0, 1], size=n_obs)
for i in range(z_dim):
    data[f"z{i}"] = RNG.normal(size=n_obs)
data["p_z"] = np.ones(n_obs) * 0.5


def _cate(data):
    # Heterogeneous treatment effect: 1 * z_1 + 2 * z_2 + ... + z_dim * z_dim
    return np.sum([i * data[f"z{i}"] for i in range(z_dim)], axis=0)


data["y"] = RNG.normal(size=n_obs) + data["d"] * tau + data["d"] * _cate(data)

res_het = generic_ml(
    data,
    n_splits=N_SPLITS,
    alpha=0.05,
    strategy="blp_ht_transform",
    ml_learner=ML_LEARNER,
)
