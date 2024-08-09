import pickle
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from topics_causal_inf.config import RNG, WGAN_GEN
from topics_causal_inf.define_dgps import DGPS
from topics_causal_inf.py_wa_replication.py_wa_replication import (
    simulation,
)


def test_simulation_runs() -> None:
    for dgp in DGPS[2:]:
        for dgen in ["wgan", "standard"]:
            wgan_train_data = pd.read_feather(WGAN_GEN / f"df_{dgp.name}.feather")
            generators = pickle.load(
                Path.open(WGAN_GEN / f"generators_{dgp.name}.pkl", "rb"),
            )
            data_wrappers = pickle.load(
                Path.open(WGAN_GEN / f"data_wrapper_{dgp.name}.pkl", "rb"),
            )

            simulation(
                n_sim=2,
                n_obs=10_000,
                dim=10,
                num_trees=100,
                sample_fraction=0.2,
                dgp=dgp,
                rng=RNG,
                data_generator=dgen,
                generators=generators,
                data_wrappers=data_wrappers,
                wgan_train_data=wgan_train_data,
            )
