"""Tasks for running WA 2018 replication simulations."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.config import BLD
from topics_causal_inf.py_wa_replication.task_py_wa_replication import (
    DG_TO_RUN,
    DGPS_TO_RUN,
    DIMS_TO_RUN,
)


class _Arguments(NamedTuple):
    dgp: str
    path_to_results: dict[str, Path]
    path_to_table: Path


PATH_TO_RESULTS = {
    f"{dgp.name}_{dgen}_{dim}": BLD
    / "py_wa_replication"
    / "sims"
    / f"py_wa_replication_{dgp.name}_dim{dim}_{dgen}.pkl"
    for dgp in DGPS_TO_RUN
    for dgen in DG_TO_RUN
    for dim in DIMS_TO_RUN
}

ID_TO_KWARGS = {
    f"{dgp.name}": _Arguments(
        dgp=dgp.name,
        path_to_results=PATH_TO_RESULTS,
        path_to_table=BLD / "tables" / f"py_wa_replication_{dgp.name}.tex",
    )
    for dgp in DGPS_TO_RUN
}


for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_py_wa_replication_table(
        dgp: str,
        path_to_results: dict[str, Path],
        path_to_table: Annotated[Path, Product],
    ) -> None:
        """Task for WA 2018 replication simulations."""
        # Combine results from different dimensions and data generators into single df
        res = pd.DataFrame()
        for dgen in DG_TO_RUN:
            for dim in DIMS_TO_RUN:
                # Exit if key does not exist
                if f"{dgp}_{dgen}_{dim}" not in path_to_results:
                    continue
                with Path.open(path_to_results[f"{dgp}_{dgen}_{dim}"], "rb") as f:
                    mse = pickle.load(f)
                _df = pd.DataFrame(mse, columns=["mse"])
                _df["dim"] = dim
                _df["data_generator"] = dgen
                _df["dgp"] = dgp

                res = pd.concat([res, _df])

        res_to_table = res.groupby(["dgp", "data_generator", "dim"]).agg(
            ["mean", "std"],
        )

        res_to_table.to_latex(
            path_to_table,
            caption=f"Mean and standard deviation of MSE for {dgp} DGP.",
            label=f"tab:py_wa_replication_{dgp}",
        )
