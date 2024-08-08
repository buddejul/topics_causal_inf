"""Tests for generic_ml and related functions."""

import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
from topics_causal_inf.config import RNG
from topics_causal_inf.define_dgps import DGPS
from topics_causal_inf.generic_ml.generic_ml import generic_ml, ml_proxy
from topics_causal_inf.generic_ml.simulation import simulation

ML_LEARNER = (RandomForestRegressor(), RandomForestRegressor())


def test_negative_pscore_raises_error():
    data = pd.DataFrame()
    data["y"] = RNG.normal(size=100)
    data["d"] = RNG.choice([0, 1], size=100)
    data["z"] = RNG.normal(size=100)
    data["p_z"] = RNG.normal(loc=-10, size=100)

    with pytest.raises(ValueError, match="Propensity score must be in the range"):
        generic_ml(data, 2, 0.05, ml_learner=ML_LEARNER)


def test_generic_ml_runs():
    data = pd.DataFrame()
    data["y"] = RNG.normal(size=100)
    data["d"] = RNG.choice([0, 1], size=100)
    data["z"] = RNG.normal(size=100)
    data["p_z"] = RNG.uniform(size=100)

    for strategy in ["blp_weighted_residual", "blp_ht_transform"]:
        generic_ml(data, 2, 0.05, strategy=strategy, ml_learner=ML_LEARNER)


def test_generic_ml_returns_positive_param_se():
    data = pd.DataFrame()
    data["y"] = RNG.normal(size=100)
    data["d"] = RNG.choice([0, 1], size=100)
    data["z"] = RNG.normal(size=100)
    data["p_z"] = RNG.uniform(size=100)

    out = generic_ml(data, 2, 0.05, ml_learner=ML_LEARNER)

    assert (out.blp_se > 0).all()


def test_ml_proxy_predictions_not_constant():
    aux = pd.DataFrame()
    aux["d"] = RNG.choice([0, 1], size=10_000)
    aux["y"] = RNG.normal(size=10_000) + aux["d"] * 2
    aux["z1"] = RNG.normal(size=10_000)
    aux["z2"] = RNG.normal(size=10_000)

    ml_fitted_d0, ml_fitted_d1 = ml_proxy(
        aux,
        ml_learner=(RandomForestRegressor(), RandomForestRegressor()),
    )

    b_z_pred = ml_fitted_d0.predict(aux.drop(columns=["y"]))
    s_z_pred = ml_fitted_d1.predict(aux.drop(columns=["y"])) - b_z_pred

    assert b_z_pred.var() > 0
    assert s_z_pred.var() > 0


def test_simulation_runs():
    for dgp in DGPS:
        simulation(2, 1_000, 10, dgp, 2, ML_LEARNER, RNG)
