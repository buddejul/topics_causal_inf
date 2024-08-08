"""Tasks for generic_ml simulation."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP
from topics_causal_inf.config import BLD, RNG
from topics_causal_inf.generic_ml.simulation import simulation
from topics_causal_inf.wa_replication.sim_config import DIM_VALS

N_OBS = 10_000
N_SIMS = 20
N_SPLITS = 50

DGP_TO_RUN = ["dgp3", "dgp4", "dgp5"]


class _Arguments(NamedTuple):
    dim: int
    dgp: str
    path_to_res: Path
    ml_learner: tuple[RegressorMixin, RegressorMixin]
    n_obs: int = N_OBS
    n_sims: int = N_SIMS
    n_splits: int = N_SPLITS
    rng: np.random.Generator = RNG


ID_TO_KWARGS = {
    f"genml_{dgp}_{dim}": _Arguments(
        dim=dim,
        dgp=dgp,
        path_to_res=BLD / "generic_ml" / "sims" / f"generic_ml_{dgp}_dim{dim}.pkl",
        ml_learner=(RandomForestRegressor(), RandomForestRegressor()),
    )
    for dgp in DGP_TO_RUN
    for dim in DIM_VALS
    if not (dgp == "dgp5" and dim < 8)  # noqa: PLR2004
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.skip()
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_generic_ml_simulation(
        dim: int,
        dgp: DGP,
        n_obs: int,
        n_sims: int,
        n_splits: int,
        rng: np.random.Generator,
        ml_learner: tuple[RegressorMixin, RegressorMixin],
        path_to_res: Annotated[Path, Product],
    ) -> None:
        """Task for generic_ml simulation using WA 2018 DGP."""
        res = simulation(
            n_sims=n_sims,
            n_obs=n_obs,
            dim=dim,
            dgp=dgp,
            n_splits=n_splits,
            ml_learner=ml_learner,
            rng=rng,
        )

        res.to_pickle(path_to_res)
