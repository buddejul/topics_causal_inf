"""Tasks for running WA 2018 replication simulations."""

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.classes import DGP
from topics_causal_inf.config import BLD
from topics_causal_inf.utilities import clean_tex_table
from topics_causal_inf.wa_replication.task_wa_replication import (
    DGPS_TO_RUN,
    ID_TO_KWARGS,
)


class _Arguments(NamedTuple):
    dgp: DGP
    path_to_results: list[Path]
    path_to_table: Path


PATH_TO_RESULTS = [args.path_to_data for _, args in ID_TO_KWARGS.items()]

ID_TO_KWARGS_TABLES = {
    f"{dgp.name}": _Arguments(
        dgp=dgp,
        path_to_results=PATH_TO_RESULTS,
        path_to_table=BLD
        / "wa_replication"
        / "tables"
        / f"wa_replication_{dgp.name}.tex",
    )
    for dgp in DGPS_TO_RUN
}


for id_, kwargs in ID_TO_KWARGS_TABLES.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_wa_replication_table(
        dgp: DGP,
        path_to_results: dict[str, Path],
        path_to_table: Annotated[Path, Product],
    ) -> None:
        """Task for WA 2018 replication simulations."""
        # Combine results from different dimensions and data generators into single df
        res = pd.concat(
            [pd.read_pickle(path) for path in path_to_results],
        )

        res = res[res["dgp"] == dgp.name]
        res = res.drop(columns=["dgp"])

        res_to_table = res.groupby(["data_generator", "dim"]).agg(
            ["mean", "std"],
        )

        out = res_to_table.to_latex(
            formatters={
                ("mse", "mean"): "{:,.2f}".format,
                ("mse", "std"): "{:,.2f}".format,
                ("coverage", "mean"): "{:,.2f}".format,
                ("coverage", "std"): "{:,.2f}".format,
            },
            header=False,
        )

        out = clean_tex_table(out)

        # Write to file
        path_to_table.write_text(out)
