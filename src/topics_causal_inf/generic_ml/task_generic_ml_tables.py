"""Tasks for generic_ml simulation."""

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task

from topics_causal_inf.config import BLD, DGPS_TO_RUN, DIMS_TO_RUN
from topics_causal_inf.utilities import clean_tex_table


class _Arguments(NamedTuple):
    path_to_table: Path
    path_to_res: list[Path]


# For each dgp collect all results for all dim values
ID_TO_KWARGS = {
    f"{dgp.name}_mse": _Arguments(
        path_to_table=BLD / "generic_ml" / "tables" / f"generic_ml_{dgp.name}.tex",
        path_to_res=[
            BLD / "generic_ml" / "sims" / f"generic_ml_{dgp.name}_dim{dim}.pkl"
            for dim in DIMS_TO_RUN
        ],
    )
    for dgp in DGPS_TO_RUN
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
                    "mean": x["mse_blp"].mean(),
                    "std": x["mse_blp"].std(),
                },
            ),
        )

        # Save table
        out = table.to_latex(
            float_format="%.3f",
            header=False,
        )

        # Remove all rows starting with "\begin{tabular}" and "\end{tabular}"
        out = clean_tex_table(out)

        # Write to file
        path_to_table.write_text(out)


ID_TO_KWARGS_COVERAGE = {
    f"{dgp.name}_blp_gates_coverage": _Arguments(
        path_to_table=BLD
        / "generic_ml"
        / "tables"
        / f"generic_ml_blp_gates_coverage_{dgp.name}.tex",
        path_to_res=[
            BLD / "generic_ml" / "sims" / f"generic_ml_{dgp.name}_dim{dim}.pkl"
            for dim in DIMS_TO_RUN
        ],
    )
    for dgp in DGPS_TO_RUN
}


for id_, kwargs in ID_TO_KWARGS_COVERAGE.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_generic_ml_tables_coverage(
        path_to_res: list[Path],
        path_to_table: Annotated[Path, Product],
    ) -> None:
        """Task for generic_ml result tables."""
        # Load results
        res = pd.concat([pd.read_pickle(path) for path in path_to_res])

        for i in range(5):
            res[f"covers_gate_{i}"] = (
                res[f"gate_ci_lo_{i}"] < res[f"true_gate_{i}"]
            ) & (res[f"gate_ci_hi_{i}"] > res[f"true_gate_{i}"])

        for beta in ["beta_1", "beta_2"]:
            res[f"covers_blp_{beta}"] = (
                res[f"blp_ci_lo_{beta}"] < res[f"true_blp_{beta}"]
            ) & (res[f"blp_ci_hi_{beta}"] > res[f"true_blp_{beta}"])

        # Create table
        # Get mean for all covers_gate_i and covers_blp_beta_i, each separately
        params = [f"gate_{i}" for i in range(5)] + [
            f"blp_{beta}" for beta in ["beta_1", "beta_2"]
        ]

        table = res.groupby(["dim"]).apply(
            lambda x: pd.Series(
                {
                    f"mean_cover_{col}": x[col].mean()
                    for col in [f"covers_{param}" for param in params]
                },
            ),
        )

        # Save table
        out = table.to_latex(
            float_format="%.3f",
            header=False,
        )

        # Remove all rows starting with "\begin{tabular}" and "\end{tabular}"
        out = clean_tex_table(out)

        # Write to file
        path_to_table.write_text(out)
