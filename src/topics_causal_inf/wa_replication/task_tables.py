"""Generate tables for the WA replication."""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyreadr  # type: ignore[import-untyped]
import pytask

from topics_causal_inf.config import BLD
from topics_causal_inf.wa_replication.sim_config import DIM_VALS, SIMS_TO_RUN

DGP_VALS = [sim.dgp for sim in SIMS_TO_RUN]

RESULTS_PATHS = {
    dgp: [
        BLD / "wa_replication" / "sims" / f"wa_replication_{dgp}_dim{dim}.rds"
        for dim in DIM_VALS
    ]
    for dgp in DGP_VALS
}

for dgp in DGP_VALS:
    # Generate list of all paths corresponding to a dgp and each value of dim

    @pytask.task(id=f"{dgp}_table")
    def task_wa_replication_table(
        depends_on=RESULTS_PATHS[dgp],
        produces: Path = BLD
        / "wa_replication"
        / "tables"
        / f"wa_replication_{dgp}.tex",
    ):
        """Generate tables for the WA replication."""
        # Combine all results into a single dataframe
        dfs = [pyreadr.read_r(result)[None] for result in depends_on]

        for df in dfs:
            df.columns = ["mse"]

        # To each add column with dim
        for i, df in enumerate(dfs):
            df["dim"] = DIM_VALS[i]

        # Concatenate all dataframes
        res = pd.concat(dfs)

        # Group by dim and calculate mean and std of the mse
        res_grouped = res.groupby("dim")["mse"].agg(["mean", "std"])

        # Save table
        res_grouped.to_latex(produces)
