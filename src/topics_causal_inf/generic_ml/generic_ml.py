"""Simple generic ML implementation in Python."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore[import-untyped]
from formulaic import model_matrix  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def generic_ml(
    data: pd.DataFrame,
    n_splits: int,
    alpha: float,
    ml_learner: tuple[RegressorMixin, RegressorMixin] = (
        RandomForestRegressor(),
        RandomForestRegressor(),
    ),
    strategy: str = "blp_weighted_residual",
) -> dict[str, np.ndarray]:
    """GenericML interface."""
    # Input checks
    _check_strategy(strategy)
    _check_required_covs(data)
    _check_pscore_range(data["p_z"])

    # Prepare inputs

    # Inference algorithm
    blp_params, blp_se = inference_algorithm(data, n_splits, strategy, ml_learner)

    # Compute t-values, lower and upper (1-alpha) confidence intervals
    blp_tvals = blp_params / blp_se
    blp_pvals = 2 * (1 - norm.cdf(np.abs(blp_tvals)))
    blp_ci_lo = blp_params - norm.ppf(1 - alpha / 2) * blp_se
    blp_ci_hi = blp_params + norm.ppf(1 - alpha / 2) * blp_se

    # Collapse to median
    return {
        "blp_params": np.median(blp_params, axis=0),
        "blp_se": np.median(blp_se, axis=0),
        "blp_tvals": np.median(blp_tvals, axis=0),
        "blp_pvals": np.median(blp_pvals, axis=0),
        "blp_ci_lo": np.median(blp_ci_lo, axis=0),
        "blp_ci_hi": np.median(blp_ci_hi, axis=0),
    }


def inference_algorithm(
    data: pd.DataFrame,
    n_splits: int,
    strategy: str,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the inference algorithm; see Algorithm 1 in the paper."""
    blp_params = np.zeros((n_splits, 2))
    blp_se = np.zeros((n_splits, 2))
    lambda_hat = np.zeros(n_splits)

    for i in range(n_splits):
        blp_params[i, :], blp_se[i, :], lambda_hat[i] = estimate_single_split(
            data,
            strategy,
            ml_learner,
        )

    return blp_params, blp_se


def estimate_single_split(
    data: pd.DataFrame,
    strategy: str,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate BLP parameters for a single split."""
    # Split data into training and test set
    main, aux = train_test_split(data, test_size=0.5)

    # Train ML proxies
    b_z, s_z = ml_proxy(aux.drop(columns=["p_z"]), ml_learner)

    # Estimate BLP parameters
    main["b_z"] = b_z
    main["s_z"] = s_z

    if strategy == "blp_weighted_residual":
        res = blp_weighted_residual(main)
        blp_params = ["d - p_z", "d - p_z:center(s_z)"]

    elif strategy == "blp_ht_transform":
        res = blp_ht_transform(main)
        blp_params = ["Intercept", "center(s_z)"]

    lambda_hat = fit_measure_cate(res.params[blp_params[1]], main, "blp")

    return res.params[blp_params], res.HC3_se[blp_params], lambda_hat


def ml_proxy(
    data: pd.DataFrame,
    ml_learner: tuple[RegressorMixin, RegressorMixin],
) -> tuple[np.ndarray, np.ndarray]:
    """Train ML proxies for BCA and CATE."""
    # Fit for untreated (d == 0)
    model0 = ml_learner[0]
    model1 = ml_learner[1]

    res_d0 = model0.fit(
        data[data["d"] == 0].drop(columns="y"),
        data[data["d"] == 0]["y"],
    )
    b_z = res_d0.predict(data.drop(columns="y"))

    # Fit for treated (d == 1)
    res_d1 = model1.fit(
        data[data["d"] == 1].drop(columns="y"),
        data[data["d"] == 1]["y"],
    )

    s_z = res_d1.predict(data.drop(columns="y")) - b_z

    return b_z, s_z


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
