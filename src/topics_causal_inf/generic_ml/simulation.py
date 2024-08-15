"""Simulation function for generic_ml using WA 2018 DGP."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from econml.grf import CausalForest  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP
from topics_causal_inf.generic_ml.generic_ml import classification_analysis, generic_ml
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
    mse_blp = np.zeros(n_sims)
    gates = np.zeros((n_sims, 5))
    clans = np.zeros((n_sims, 2, dim))

    for i in range(n_sims):
        mse_blp[i], gates[i, :], clans[i, :, :] = _single_experiment(
            n_obs=n_obs,
            dim=dim,
            n_splits=n_splits,
            dgp=dgp,
            ml_learner=ml_learner,
            rng=rng,
        )

    out = pd.DataFrame(mse_blp, columns=["mse_blp"])

    out["dgp"] = dgp.name
    out["n_obs"] = n_obs
    out["n_sims"] = n_sims
    out["dim"] = dim

    true_gates = _simulate_true_gates(dgp=dgp, dim=dim, rng=rng)

    for i in range(5):
        out[f"gate_{i}"] = gates[:, i]
        out[f"true_gate_{i}"] = true_gates[i]

    true_clans = _simulate_true_clan(dgp=dgp, dim=dim, rng=rng)

    for i in range(2):
        for j in range(dim):
            out[f"clan_{i}_{j}"] = clans[:, i, j]
            out[f"true_clan_{i}_{j}"] = true_clans[i, j]

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

    return (
        np.mean((data_eval["pred"] - data_eval["true"]) ** 2),
        res.gates_params,
        res.clan,
    )


def _simulate_true_gates(
    dgp: DGP,
    dim: int,
    rng: np.random.Generator,
    n_obs: int = 1_000_000,
) -> np.ndarray:
    """Simulate true gates for the given DGP."""
    # Check whether dgp.treatment_effect.keywords["tau"] raises error
    if "tau" in dgp.treatment_effect.keywords:  # type: ignore[union-attr]
        return np.ones(5) * dgp.treatment_effect.keywords["tau"]  # type: ignore[union-attr]

    data = data_wager_athey_2018(
        n_obs=n_obs,
        dim=dim,
        dgp=dgp,
        rng=rng,
    )

    feature_names = [f"x{i}" for i in range(dim)]

    data["s_z"] = dgp.treatment_effect(data[feature_names])

    data["s_z_quintiles"] = pd.qcut(data["s_z"], q=5, labels=np.arange(5))

    return data.groupby("s_z_quintiles", observed=False).apply("mean")["s_z"]


def _simulate_true_clan(
    dgp: DGP,
    dim: int,
    rng: np.random.Generator,
    n_obs: int = 1_000_000,
) -> np.ndarray:
    """Simulate true CLAN for the given DGP."""
    # Check whether dgp.treatment_effect.keywords["tau"] raises error
    if "tau" in dgp.treatment_effect.keywords:  # type: ignore[union-attr]
        return np.ones((2, dim)) * 0.5  # type: ignore[union-attr]

    data = data_wager_athey_2018(
        n_obs=n_obs,
        dim=dim,
        dgp=dgp,
        rng=rng,
    )

    feature_names = [f"x{i}" for i in range(dim)]

    data["s_z"] = dgp.treatment_effect(data[feature_names])

    return classification_analysis(data)
