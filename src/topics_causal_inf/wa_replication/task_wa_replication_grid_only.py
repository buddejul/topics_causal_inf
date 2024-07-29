"""Tasks for replication of Wager and Athey (2018) results."""

from pathlib import Path

import pytask

from topics_causal_inf.config import BLD, SRC
from topics_causal_inf.wa_replication.sim_config import DGP_VALS, DIM_VALS, SIM_KWARGS

for dgp in DGP_VALS:
    for dim in DIM_VALS:

        @pytask.task(
            kwargs={"dgp": dgp, "dim": dim, "return_grid": True, **SIM_KWARGS},
            id=f"{dgp}_dim{dim}",
        )
        @pytask.mark.r(
            script=SRC / "wa_replication" / "wa_replication.R",
            serializer="yaml",
        )
        def task_wa_replication_grid(
            produces: Path = BLD
            / "wa_replication"
            / f"wa_replication_{dgp}_dim{dim}_grid_only.rds",
        ):
            """Produce example figure using grf in R."""
