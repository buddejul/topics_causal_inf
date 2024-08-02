"""Simulation function for generic_ml using WA 2018 DGP."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]

from topics_causal_inf.generic_ml.generic_ml import generic_ml
from topics_causal_inf.utilities import (
    _tau_constant,
    _tau_heterog,
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

    out = pd.DataFrame(mse, columns=["mse"])

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
    data_eval = data_eval.drop(columns=["y", "p_z"])

    data_eval["s_z"] = res.ml_fitted_d1.predict(data_eval) - res.ml_fitted_d1.predict(
        data_eval,
    )

    data_eval["pred"] = res.blp_params[0] + res.blp_params[1] * data_eval["s_z"]

    if dgp == "dgp1":
        data_eval["true"] = _tau_constant(data_eval, 0)
    elif dgp == "dgp2":
        data_eval["true"] = _tau_heterog(
            data_eval,
            x_range=np.arange(1, 3),
            a=20,
            b=1 / 3,
        )
    elif dgp == "dgp3":
        data_eval["true"] = _tau_heterog(
            data_eval,
            x_range=np.arange(1, 3),
            a=12,
            b=1 / 2,
        )
    elif dgp == "dgp4":
        data_eval["true"] = _tau_heterog(
            data_eval,
            x_range=np.arange(1, 5),
            a=12,
            b=1 / 2,
        )
    elif dgp == "dgp5":
        data_eval["true"] = _tau_heterog(
            data_eval,
            x_range=np.arange(1, 9),
            a=12,
            b=1 / 2,
        )

    return np.mean((data_eval["pred"] - data_eval["true"]) ** 2)
