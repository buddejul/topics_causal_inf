"""Simple generic ML implementation in Python."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore[import-untyped]
from econml.grf import CausalForest  # type: ignore[import-untyped]
from formulaic import model_matrix  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from topics_causal_inf.classes import GenericMLResult


def generic_ml(
    data: pd.DataFrame,
    n_splits: int,
    alpha: float,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
    strategy: str = "blp_weighted_residual",
) -> GenericMLResult:
    """GenericML interface."""
    # Input checks
    _check_strategy(strategy)
    _check_required_covs(data)
    _check_pscore_range(data["p_z"])

    # Prepare inputs

    # Inference algorithm
    blp_params, blp_se, ml_fitted_d0, ml_fitted_d1, gates_params, gates_se, clan = (
        inference_algorithm(
            data,
            n_splits,
            strategy,
            ml_learner,
        )
    )

    # BLP Compute t-values, lower and upper (1-alpha) confidence intervals
    blp_tvals = blp_params / blp_se
    blp_pvals = 2 * (1 - norm.cdf(np.abs(blp_tvals)))
    blp_ci_lo = blp_params - norm.ppf(1 - alpha / 2) * blp_se
    blp_ci_hi = blp_params + norm.ppf(1 - alpha / 2) * blp_se

    # GATES: Compute t-values, lower and upper (1-alpha) confidence intervals
    gates_tvals = gates_params / gates_se
    gates_pvals = 2 * (1 - norm.cdf(np.abs(gates_tvals)))
    gates_ci_lo = gates_params - norm.ppf(1 - alpha / 2) * gates_se
    gates_ci_hi = gates_params + norm.ppf(1 - alpha / 2) * gates_se

    # Find position of median of alpha_2 to pick a "median" machine learner
    # Choose closest observation to force median is element of the set
    pos = np.argwhere(
        blp_params[:, 1]
        == np.percentile(blp_params[:, 1], q=50, method="closest_observation"),
    )[0][0]

    # Collapse to median
    return GenericMLResult(
        blp_params=np.median(blp_params, axis=0),
        blp_se=np.median(blp_se, axis=0),
        blp_tvals=np.median(blp_tvals, axis=0),
        blp_pvals=np.median(blp_pvals, axis=0),
        blp_ci_lo=np.median(blp_ci_lo, axis=0),
        blp_ci_hi=np.median(blp_ci_hi, axis=0),
        ml_fitted_d0=ml_fitted_d0[int(pos)],
        ml_fitted_d1=ml_fitted_d1[int(pos)],
        gates_params=np.median(gates_params, axis=0),
        gates_se=np.median(gates_se, axis=0),
        gates_tvals=np.median(gates_tvals, axis=0),
        gates_pvals=np.median(gates_pvals, axis=0),
        gates_ci_lo=np.median(gates_ci_lo, axis=0),
        gates_ci_hi=np.median(gates_ci_hi, axis=0),
        clan=np.median(clan, axis=0),
    )


def inference_algorithm(
    data: pd.DataFrame,
    n_splits: int,
    strategy: str,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[RegressorMixin],
    list[RegressorMixin],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Implements the inference algorithm; see Algorithm 1 in the paper."""
    pattern = r"^x\d+$"
    feature_names = data.filter(regex=pattern).columns.tolist()
    dim = len(feature_names)

    blp_params = np.zeros((n_splits, 2))
    blp_se = np.zeros((n_splits, 2))
    lambda_hat = np.zeros(n_splits)
    ml_fitted_d0 = list(np.zeros(n_splits))
    ml_fitted_d1 = list(np.zeros(n_splits))
    gates_params = np.zeros((n_splits, 5))
    gates_se = np.zeros((n_splits, 5))
    clan = np.zeros((n_splits, 2, dim))

    for i in range(n_splits):
        (
            blp_params[i, :],
            blp_se[i, :],
            lambda_hat[i],
            ml_fitted_d0[i],
            ml_fitted_d1[i],
            gates_params[i, :],
            gates_se[i, :],
            clan[i, :, :],
        ) = estimate_single_split(
            data,
            strategy,
            ml_learner,
        )

    return blp_params, blp_se, ml_fitted_d0, ml_fitted_d1, gates_params, gates_se, clan


def estimate_single_split(
    data: pd.DataFrame,
    strategy: str,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    RegressorMixin,
    RegressorMixin,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Estimate BLP parameters for a single split."""
    pattern = r"^x\d+$"
    feature_names = data.filter(regex=pattern).columns.tolist()

    # Split data into training and test set
    main, aux = train_test_split(data, test_size=0.5)

    # Train ML proxies and predict on main sample
    ml_fitted_d0, ml_fitted_d1 = ml_proxy(aux.drop(columns=["p_z"]), ml_learner)

    main["b_z"] = ml_fitted_d0.predict(main[feature_names])

    if isinstance(ml_learner[1], CausalForest):
        # Causal Forest directly targets CATE
        main["s_z"] = ml_fitted_d1.predict(main[feature_names])
    else:
        main["s_z"] = ml_fitted_d1.predict(main[feature_names]) - main["b_z"]

    # Estimate BLP parameters
    if strategy == "blp_weighted_residual":
        res = blp_weighted_residual(main)
        blp_params = ["d - p_z", "d - p_z:center(s_z)"]

    elif strategy == "blp_ht_transform":
        res = blp_ht_transform(main)
        blp_params = ["Intercept", "center(s_z)"]

    # Estimate GATEs
    gates = gates_ht_transform(main)
    gates_params = [f"s_z_quintiles[T.{i}]" for i in np.arange(5)]

    # Classification analysis (CLAN)
    clan = classification_analysis(main)

    # Fit measure
    lambda_hat = fit_measure_cate(res.params[blp_params[1]], main, "blp")

    return (
        res.params[blp_params],
        res.HC3_se[blp_params],
        lambda_hat,
        ml_fitted_d0,
        ml_fitted_d1,
        gates.params[gates_params],
        gates.HC3_se[gates_params],
        clan,
    )


def ml_proxy(
    data: pd.DataFrame,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[RegressorMixin, RegressorMixin]:
    """Train ML proxies for BCA and CATE."""
    # Fit for untreated (d == 0)
    model0 = ml_learner[0]
    model1 = ml_learner[1]

    # feature_names are all names of the form "x0", "x1", ...
    pattern = r"^x\d+$"
    feature_names = data.filter(regex=pattern).columns.tolist()

    # Define kwargs depending on the model used
    # if type is CausalForest()
    if isinstance(model0, CausalForest):
        msg = "Cannot specify CausalForest as BCA learner."
        raise TypeError(msg)
    kwargs0 = {
        "X": data[data["d"] == 0][feature_names],
        "y": data[data["d"] == 0]["y"],
    }

    if isinstance(model1, CausalForest):
        # CausalForest directly targets CATE hence we fit on all of the data.
        kwargs1 = {
            "X": data[feature_names],
            "T": data["d"],
            "y": data["y"],
        }
    else:
        kwargs1 = {
            "X": data[data["d"] == 1][feature_names],
            "y": data[data["d"] == 1]["y"],
        }

    ml_fitted_d0 = model0.fit(**kwargs0)

    # Fit for treated (d == 1)
    ml_fitted_d1 = model1.fit(**kwargs1)

    return ml_fitted_d0, ml_fitted_d1


def blp_weighted_residual(
    data: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Implements Strategy A: Weighted Residual BLP; see equations (3.1) - (3.3)."""
    weights = (data["p_z"] * (1 - data["p_z"])) ** (-1)

    # Construct design matrix

    formula = "y ~ 1 + b_z + p_z + p_z:s_z + {d - p_z} + {d - p_z}:center(s_z)"
    y, x = model_matrix(formula, data)

    # Estimation using WLS following equation (3.3)
    model = sm.WLS(
        endog=y,
        exog=x,
        weights=weights,
        hasconst=True,
    )

    return model.fit()


def blp_ht_transform(
    data: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Implements Strategy B: HT Transformation BLP; see equations (3.4) - (3.5)."""
    # TODO(budde_jul): Is this side-effect? # noqa: FIX002, TD003
    data["h"] = (data["d"] - data["p_z"]) / (data["p_z"] * (1 - data["p_z"]))

    formula = "y:h ~ 1:h + b_z:h + p_z:h + p_z:s_z:h + 1 + center(s_z)"
    y, x = model_matrix(formula, data)

    model = sm.OLS(
        endog=y,
        exog=x,
        hasconst=True,
    )

    return model.fit()


def fit_measure_cate(beta_2: float, data: pd.DataFrame, target: str) -> float:
    """Calculate goodness of fit measures for fitting CATE; see (3.13)."""
    if target == "blp":
        lambda_hat = beta_2**2 * np.var(data["s_z"])

    return lambda_hat


def _check_required_covs(data: pd.DataFrame) -> None:
    """Check if all covariates are present."""
    required_covariates = ["y", "d", "p_z"]

    msg = ""

    for covariate in required_covariates:
        if covariate not in data.columns:
            msg += f"{covariate}, "

    if msg:
        msg_to_raise = f"Missing required covariates: {msg[:-2]}"
        raise ValueError(msg_to_raise)


def _check_pscore_range(pscore) -> None:
    """Check if propensity score is in the range [0, 1]."""
    if not all((pscore >= 0) & (pscore <= 1)):
        msg = "Propensity score must be in the range [0, 1]."
        raise ValueError(msg)


def _check_strategy(strategy: str) -> None:
    """Check if strategy is valid."""
    if strategy not in ["blp_weighted_residual", "blp_ht_transform"]:
        msg = f"Invalid strategy: {strategy}."
        raise ValueError(msg)


def gates_ht_transform(
    data: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Estimate GATE parameters using the Horvit-Thompson transform."""
    data["h"] = (data["d"] - data["p_z"]) / (data["p_z"] * (1 - data["p_z"]))

    # Generate dummies for quantiles of ML proxy
    data["s_z_quintiles"] = pd.qcut(data["s_z"], q=5, labels=np.arange(5))

    formula = "y:h ~ -1 + b_z:h + p_z:s_z_quintiles:h + s_z_quintiles"

    y, x = model_matrix(formula, data)

    # Estimation using WLS following equation (3.3)
    model = sm.OLS(
        endog=y,
        exog=x,
        hasconst=False,
    )

    return model.fit()


def classification_analysis(data: pd.DataFrame) -> np.ndarray:
    """Compute mean covariate characteristics for the least and most affected groups."""
    # Compute mean covariate characteristics for the least and most affected groups
    data["s_z_quintiles"] = pd.qcut(data["s_z"], q=5, labels=np.arange(5))

    pattern = r"^x\d+$"
    feature_names = data.filter(regex=pattern).columns.tolist()

    means_by_quintiles = (
        data.groupby("s_z_quintiles", observed=False).mean()[feature_names].to_numpy()
    )

    return means_by_quintiles[0 :: len(means_by_quintiles) - 1]
