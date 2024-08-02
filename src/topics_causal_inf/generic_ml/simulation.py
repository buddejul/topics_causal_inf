"""Simulation function for generic_ml using WA 2018 DGP."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]

from topics_causal_inf.generic_ml.generic_ml import generic_ml
from topics_causal_inf.utilities import (
    data_wager_athey_2018,
)


def simulation(
    n_sims: int,
    n_obs: int,
    dim: int,
    dgp: str,
    n_splits: int,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run simulation using the generic_ml approach and WA 2018 DGP."""
    mse = np.zeros(n_sims)

    for i in range(n_sims):
        mse[i] = _single_experiment(
            n_obs=n_obs,
            dim=dim,
            n_splits=n_splits,
            dgp=dgp,
            ml_learner=ml_learner,
            rng=rng,
        )

    out = pd.DataFrame(mse, columns="mse")

    out["dgp"] = dgp
    out["n_obs"] = n_obs
    out["n_sims"] = n_sims

    return out


def _single_experiment(
    n_obs: int,
    dim: int,
    n_splits: int,
    dgp: str,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run single experiment for generic_ml simulation."""
    data = data_wager_athey_2018(n_obs=n_obs, dim=dim, dgp=dgp, rng=rng)

    res = generic_ml(data, n_splits, alpha=0.05, ml_learner=ml_learner)

    # Draw evaluation data and calculate rmse using linear prediction based on res
    data_eval = data_wager_athey_2018(n_obs=n_obs, dim=dim, dgp=dgp, rng=rng)

    data_eval["pred"] = 1

    return res
