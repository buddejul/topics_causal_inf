from dataclasses import dataclass

import numpy as np
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
