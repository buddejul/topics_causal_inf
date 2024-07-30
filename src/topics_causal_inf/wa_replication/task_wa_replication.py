"""Tasks for replication of Wager and Athey (2018) results."""

from pathlib import Path

import pytask

from topics_causal_inf.config import BLD, SRC
from topics_causal_inf.wa_replication.sim_config import DIM_VALS, SIMS_TO_RUN

for sim in SIMS_TO_RUN:
    for dim in DIM_VALS:

        @pytask.task(
            kwargs={"dim": dim, "return_grid": False, **sim._asdict()},
            id=f"{sim.dgp}_dim{dim}",
        )
        @pytask.mark.r(
            script=SRC / "wa_replication" / "wa_replication.R",
            serializer="yaml",
        )
        def task_wa_replication(
            produces: Path = BLD
            / "wa_replication"
            / "sims"
            / f"wa_replication_{sim.dgp}_dim{dim}.rds",
        ):
            """Run simulations for WA 2018 DGPs."""
