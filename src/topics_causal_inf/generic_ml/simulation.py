"""Simulation function for generic_ml using WA 2018 DGP."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from econml.grf import CausalForest  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP
from topics_causal_inf.generic_ml.generic_ml import generic_ml
from topics_causal_inf.utilities import (
    data_wager_athey_2018,
)


def simulation(
    n_sims: int,
    n_obs: int,
    dim: int,
    dgp: DGP,
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

    out = pd.DataFrame(mse, columns=["mse"])

    out["dgp"] = dgp.name
    out["n_obs"] = n_obs
    out["n_sims"] = n_sims
    out["dim"] = dim

    return out


def _single_experiment(
    n_obs: int,
    dim: int,
    n_splits: int,
    dgp: DGP,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run single experiment for generic_ml simulation."""
    feature_names = [f"x{i}" for i in range(dim)]

    data = data_wager_athey_2018(n_obs=n_obs, dim=dim, dgp=dgp, rng=rng)

    res = generic_ml(data, n_splits, alpha=0.05, ml_learner=ml_learner)

    # Draw evaluation data and calculate rmse using linear prediction based on res
    data_eval = data_wager_athey_2018(n_obs=n_obs, dim=dim, dgp=dgp, rng=rng)
    data_eval = data_eval[feature_names]

    if isinstance(ml_learner[1], CausalForest):
        # Causal Forest directly targets CATE
        data_eval["s_z"] = res.ml_fitted_d1.predict(data_eval[feature_names])
    else:
        data_eval["s_z"] = res.ml_fitted_d1.predict(
            data_eval[feature_names],
        ) - res.ml_fitted_d0.predict(data_eval[feature_names])

    data_eval["pred"] = res.blp_params[0] + res.blp_params[1] * data_eval["s_z"]

    data_eval["true"] = dgp.treatment_effect(data_eval)

    return np.mean((data_eval["pred"] - data_eval["true"]) ** 2)
