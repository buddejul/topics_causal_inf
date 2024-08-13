"""Tasks for generic_ml simulation."""

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.config import BLD
from topics_causal_inf.generic_ml.task_generic_ml_sim import DGP_TO_RUN
from topics_causal_inf.wa_replication.sim_config import DIM_VALS


class _Arguments(NamedTuple):
    path_to_table: Path
    path_to_res: list[Path]


# For each dgp collect all results for all dim values
ID_TO_KWARGS = {
    dgp.name: _Arguments(
        path_to_table=BLD / "generic_ml" / "tables" / f"generic_ml_{dgp.name}.tex",
        path_to_res=[
            BLD / "generic_ml" / "sims" / f"generic_ml_{dgp.name}_dim{dim}.pkl"
            for dim in DIM_VALS
        ],
    )
    for dgp in DGP_TO_RUN
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_generic_ml_tables(
        path_to_res: list[Path],
        path_to_table: Annotated[Path, Product],
    ) -> None:
        """Task for generic_ml result tables."""
        # Load results
        res = pd.concat([pd.read_pickle(path) for path in path_to_res])

        # Create table
        table = res.groupby(["dim"]).apply(
            lambda x: pd.Series(
                {
                    "mean": x["mse"].mean(),
                    "std": x["mse"].std(),
                },
            ),
        )

        # Save table
        table.to_latex(
            path_to_table,
            float_format="%.3f",
        )
