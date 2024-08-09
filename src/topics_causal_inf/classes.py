from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]


@dataclass
class GenericMLResult:
    """Result for generic_ml procedure."""

    blp_params: np.ndarray
    blp_se: np.ndarray
    blp_tvals: np.ndarray
    blp_pvals: np.ndarray
    blp_ci_lo: np.ndarray
    blp_ci_hi: np.ndarray
    ml_fitted_d0: RegressorMixin
    ml_fitted_d1: RegressorMixin


class DGP(NamedTuple):
    """DGP for simulation."""

    name: str
    main_effect: Callable[[pd.DataFrame], np.ndarray]
    treatment_effect: Callable[[pd.DataFrame], np.ndarray]
    propensity_score: Callable[[pd.DataFrame], np.ndarray]
