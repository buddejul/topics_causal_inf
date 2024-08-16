"""Utilities used throughout the project."""

import re
from copy import copy

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import beta  # type: ignore[import-untyped]
from wgan import DataWrapper, Generator  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP


def data_wager_athey_2018(
    n_obs: int,
    dim: int,
    dgp: DGP,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate data from one of the DGPs in Wager and Athey 2018 JASA.

    Arguments:
        n_obs: Number of observations.
        dim: Dimension of covariates.
        dgp: DGP to simulate data from.
        rng: Random number generator.

    Returns:
        pd.DataFrame: Simulated data.
    """
    _check_dgp_and_dim_compatible(dgp, dim)

    out = pd.DataFrame(index=range(n_obs))

    for i in range(dim):
        out[f"x{i}"] = rng.uniform(size=n_obs)

    out["p_z"] = dgp.propensity_score(out)
    out["d"] = rng.binomial(1, out["p_z"])

    out["y"] = (
        dgp.main_effect(out)
        + dgp.treatment_effect(out) * out["d"]
        + rng.normal(size=n_obs)
    )

    return out


def tau_constant(data: pd.DataFrame, tau: float) -> np.ndarray:
    """Treatment effect: tau(x) = 0."""
    return np.ones(data.shape[0]) * tau


def main_linear(data: pd.DataFrame) -> np.ndarray:
    """Main effect: m(x) = 2x_1 - 1."""
    return 2 * data["x1"] - 1


def main_constant(data: pd.DataFrame, main: float) -> np.ndarray:
    """Main effect: m(x) = main."""
    return np.ones(data.shape[0]) * main


def prop_beta(data: pd.DataFrame, a: float, b: float) -> np.ndarray:
    """Propensity score: e(x) = 0.25 * (1 + beta_(a, b)(x_1))."""
    return 0.25 * (1 + beta.pdf(data["x1"], a, b))


def _xi(sr: pd.Series, a: float, b: float) -> pd.Series:
    """Helper function: xi(x) = 1 + (1 + exp(-a * (x - b)))^-1."""
    return 1 + (1 + np.exp(-a * (sr - b))) ** -1


def prop_cons(data: pd.DataFrame, prop: float) -> np.ndarray:
    """Propensity score: e(x) = prop."""
    return np.ones(data.shape[0]) * prop


def tau_heterog(
    data: pd.DataFrame,
    x_range: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Treatment effect: tau(x) = xi(x_1) * xi(x_2).

    Arguments:
        data: Data.
        x_range: Range of covariates to apply the treatment effect.
        a: Parameter a.
        b: Parameter b.

    Returns:
        np.ndarray: Treatment effect.
    """
    tau = np.ones(data.shape[0])
    for i in x_range:
        tau = tau * _xi(data[f"x{i}"], a, b)
    return tau


# TODO(@buddejul): Change df.sample to use RNG. #noqa: TD003, FIX002
def data_wgan(
    n_obs: int,
    wgan_train_data: pd.DataFrame,
    data_wrappers: list[DataWrapper],
    generators: list[Generator],
) -> pd.DataFrame:
    """Draw data based on ds-wgan generator.

    Arguments:
        n_obs: Number of observations.
        dgp: DGP to simulate data from.
        wgan_train_data: Data that was used to train the generator.
        data_wrappers: Data wrappers used to apply the generator.
        generators: Generators used to generate data.

    Returns:
        pd.DataFrame: Simulated data.
    """
    # Simulate data
    # simulate data with conditional WGANs
    out = data_wrappers[0].apply_generator(
        generators[0],
        wgan_train_data.sample(int(n_obs), replace=True),
    )
    out = data_wrappers[1].apply_generator(generators[1], out)

    # add counterfactual outcomes
    out_cf = copy(out)
    out_cf["t"] = 1 - out_cf["t"]
    out["y_cf"] = data_wrappers[1].apply_generator(
        generators[1],
        out_cf,
    )["y"]

    # Generate treatment effect: "y" - "y_cf" if "t" == 1 else "y_cf" - "y"
    out["tau"] = out["y"] - out["y_cf"]
    out["tau"] = np.where(
        out["t"] == 1,
        out["tau"],
        -out["tau"],
    )

    out["p_z"] = 0.5

    return out.rename(columns={"t": "d"})


def _check_dgp_and_dim_compatible(dgp: DGP, dim: int) -> None:
    """Check that dgp.treatment_effect x_range parameter length is at most dim."""
    if (
        hasattr(dgp.treatment_effect, "keywords")
        and "x_range" in dgp.treatment_effect.keywords
    ) and (len(dgp.treatment_effect.keywords["x_range"]) > dim):
        length = len(dgp.treatment_effect.keywords["x_range"])
        msg = f"dgp.treatment_effect x_range has length {length} > dim {dim}."
        raise ValueError(msg)


def clean_tex_table(table: str) -> str:
    """Clean table produced by df.to_latex for paper."""
    out = re.sub(r"\\begin{tabular}.*\n", "", table)
    out = re.sub(r"\\end{tabular}.*\n", "", out)
    out = re.sub(r"\\toprule.*\n", "", out)
    out = re.sub(r"\\midrule.*\n", "", out)
    out = re.sub(r"\\bottomrule.*\n", "", out)
    out = re.sub("_", " ", out)
    out = re.sub("NaN", "-", out)
    out = re.sub("standard", "Standard DGP", out)
    out = re.sub("wgan", "WGAN DGP", out)
    out = re.sub(r"data generator.*\n", "", out)
    out = re.sub(r"\\cline.*\n", "", out)
    return re.sub(r"dim .*\n", "", out)
