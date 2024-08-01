"""Utilities used throughout the project."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import beta  # type: ignore[import-untyped]

from topics_causal_inf.config import RNG


def data_wager_athey_2018(n_obs: int, dim: int, dgp: str) -> pd.DataFrame:
    """Simulate data from one of the DGPs in Wager and Athey 2018 JASA.

    Arguments:
        n_obs: Number of observations.
        dim: number of covariates (ambient dimension).
        dgp: DGP to simulate data from.

    Returns:
        pd.DataFrame: Simulated data.
    """
    out = pd.DataFrame(index=range(n_obs))

    for i in range(dim):
        out[f"x{i}"] = RNG.uniform(size=n_obs)

    if dgp == "dgp1":
        main = _main_linear(out)
        prop = _prop_beta(out, 2, 4)
        tau = _tau_constant(out, 0)

    if dgp == "dgp2":
        main = _main_constant(out, 0)
        prop = _prop_cons(out, 0.5)
        tau = _tau_heterogeneous(out, 20, 1 / 3)

    if dgp == "dgp3":
        main = _main_constant(out, 0)
        prop = _prop_cons(out, 0.5)
        tau = _tau_heterogeneous(out, 12, 1 / 2)

    out["p_z"] = prop
    out["d"] = RNG.binomial(1, prop)

    out["y"] = main + tau * out["d"] + RNG.normal(size=n_obs)

    return out


def _tau_constant(data: pd.DataFrame, tau: float) -> np.ndarray:
    """Dment effect: tau(x) = 0."""
    return np.ones(data.shape[0]) * tau


def _main_linear(data: pd.DataFrame) -> np.ndarray:
    """Main effect: m(x) = 2x_1 - 1."""
    return 2 * data["x1"] - 1


def _main_constant(data: pd.DataFrame, main: float) -> np.ndarray:
    """Main effect: m(x) = main."""
    return np.ones(data.shape[0]) * main


def _prop_beta(data: pd.DataFrame, a: float, b: float) -> np.ndarray:
    """Propensity score: e(x) = 0.25 * (1 + beta_(a, b)(x_1))."""
    return 0.25 * (1 + beta.pdf(data["x1"], a, b))


def _xi(sr: pd.Series, a: float, b: float) -> pd.Series:
    """Helper function: xi(x) = 1 + (1 + exp(-a * (x - b)))^-1."""
    return 1 + (1 + np.exp(-a * (sr - b))) ** -1


def _prop_cons(data: pd.DataFrame, prop: float) -> np.ndarray:
    """Propensity score: e(x) = prop."""
    return np.ones(data.shape[0]) * prop


def _tau_heterogeneous(data: pd.DataFrame, a: float, b: float) -> np.ndarray:
    """Dment effect: tau(x) = xi(x_1) * xi(x_2)."""
    return _xi(data["x1"], a, b) * _xi(data["x2"], a, b)
