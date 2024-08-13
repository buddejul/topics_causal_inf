"""Tasks for running WA 2018 replication simulations."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.classes import DGP
from topics_causal_inf.config import BLD, RNG, WGAN_GEN
from topics_causal_inf.define_dgps import DGP3, DGP4
from topics_causal_inf.py_wa_replication.py_wa_replication import simulation
from topics_causal_inf.wa_replication.sim_config import DIM_VALS

DGPS_TO_RUN = [DGP3, DGP4]

DIMS_TO_RUN = DIM_VALS


class _Arguments(NamedTuple):
    dgp: DGP
    path_to_data: Path
    data_generator: str
    n_sim: int = 20
    n_obs: int = 10_000
    dim: int = 10
    rng: np.random.Generator = RNG
    num_trees: int = 100
    sample_fraction: float = 0.2


DG_TO_RUN = ["standard", "wgan"]

ID_TO_KWARGS = {
    f"{dgp.name}_{dim}_{dgen}": _Arguments(
        dgp=dgp,
        path_to_data=BLD
        / "py_wa_replication"
        / "sims"
        / f"py_wa_replication_{dgp.name}_dim{dim}_{dgen}.pkl",
        dim=dim,
        data_generator=dgen,
    )
    for dgp in DGPS_TO_RUN
    for dim in DIMS_TO_RUN
    for dgen in DG_TO_RUN
    if not (dgen == "wgan" and dim != 10)  # noqa: PLR2004
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_py_wa_replication_sims(
        dgp: DGP,
        path_to_data: Annotated[Path, Product],
        n_sim: int,
        n_obs: int,
        dim: int,
        data_generator: str,
        rng: np.random.Generator,
        num_trees: int,
        sample_fraction: float,
    ) -> None:
        """Task for WA 2018 replication simulations."""
        if data_generator == "wgan":
            wgan_train_data = pd.read_feather(WGAN_GEN / f"df_{dgp.name}.feather")
            generators = pickle.load(
                Path.open(WGAN_GEN / f"generators_{dgp.name}.pkl", "rb"),
            )
            data_wrappers = pickle.load(
                Path.open(WGAN_GEN / f"data_wrapper_{dgp.name}.pkl", "rb"),
            )
        else:
            wgan_train_data = None
            generators = None
            data_wrappers = None

        res = simulation(
            n_sim=n_sim,
            n_obs=n_obs,
            dim=dim,
            num_trees=num_trees,
            sample_fraction=sample_fraction,
            dgp=dgp,
            rng=rng,
            data_generator=data_generator,
            wgan_train_data=wgan_train_data,
            generators=generators,
            data_wrappers=data_wrappers,
        )

        res.to_pickle(path_to_data)
