import numpy as np
from topics_causal_inf.config import RNG
from topics_causal_inf.define_dgps import DGP1, DGP2, DGP3, DGP4, DGP5
from topics_causal_inf.utilities import data_wager_athey_2018

DGPS = [DGP1, DGP2, DGP3, DGP4, DGP5]


def test_data_wager_athey_2018_runs() -> None:
    for dgp in DGPS:
        data_wager_athey_2018(
            n_obs=10_000,
            dim=10,
            dgp=dgp,
            rng=RNG,
        )


def test_column_means_x_and_d() -> None:
    for dgp in [DGP3, DGP4, DGP5]:
        n_obs = 10_000
        dim = 10
        data = data_wager_athey_2018(
            n_obs=n_obs,
            dim=dim,
            dgp=dgp,
            rng=RNG,
        )

        assert np.allclose(
            data[[f"x{i}" for i in np.arange(dim)] + ["d"]].mean(),
            0.5,
            atol=5 * np.sqrt(n_obs),
        )
