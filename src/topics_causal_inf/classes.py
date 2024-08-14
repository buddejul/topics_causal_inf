import functools
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
    gates_params: np.ndarray
    gates_se: np.ndarray
    gates_tvals: np.ndarray
    gates_pvals: np.ndarray
    gates_ci_lo: np.ndarray
    gates_ci_hi: np.ndarray


class DGP(NamedTuple):
    """DGP for simulation."""

    name: str
    main_effect: Callable[[pd.DataFrame], np.ndarray] | functools.partial
    treatment_effect: Callable[[pd.DataFrame], np.ndarray] | functools.partial
    propensity_score: Callable[[pd.DataFrame], np.ndarray] | functools.partial
