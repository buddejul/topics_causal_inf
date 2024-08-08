"""Tasks for running WA 2018 replication simulations."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task
from wgan import DataWrapper, Generator  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP
from topics_causal_inf.config import BLD, RNG
from topics_causal_inf.define_dgps import DGP3, DGP4, DGP5
from topics_causal_inf.py_wa_replication.py_wa_replication import simulation

DGPS_TO_RUN = [DGP3, DGP4, DGP5]
DIMS_TO_RUN = [8, 10, 12]


class _Arguments(NamedTuple):
    dgp: DGP
    path_to_data: Path
    n_sim: int = 20
    n_obs: int = 10_000
    dim: int = 10
    data_generator: str = "standard"
    pop_data: pd.DataFrame | None = None
    generators: list[Generator] | None = None
    data_wrappers: list[DataWrapper] | None = None
    rng: np.random.Generator = RNG
    num_trees: int = 100
    sample_fraction: float = 0.5


ID_TO_KWARGS = {
    f"{dgp.name}_{dim}_standard": _Arguments(
        dgp=dgp,
        path_to_data=BLD
        / "py_wa_replication"
        / "sims"
        / f"py_wa_replication_{dgp.name}_dim{dim}_standard.pkl",
        dim=dim,
        data_generator="standard",
    )
    for dgp in DGPS_TO_RUN
    for dim in DIMS_TO_RUN
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
        pop_data: pd.DataFrame | None,
        generators: list[Generator] | None,
        data_wrappers: list[DataWrapper] | None,
    ) -> None:
        """Task for WA 2018 replication simulations."""
        res = simulation(
            n_sim=n_sim,
            n_obs=n_obs,
            dim=dim,
            num_trees=num_trees,
            sample_fraction=sample_fraction,
            dgp=dgp,
            rng=rng,
            data_generator=data_generator,
        )

        with Path.open(path_to_data, "wb") as f:
            pickle.dump(res, f)
