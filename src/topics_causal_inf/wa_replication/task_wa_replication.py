"""Tasks for replication of Wager and Athey (2018) results."""

from pathlib import Path

import pytask

from topics_causal_inf.config import BLD, SRC

SIM_KWARGS = {
    "n_obs": 500,
    "n_sim": 100,
    "dim": 2,
}

for dgp in ["dgp1", "dgp2", "dgp3"]:

    @pytask.task(kwargs={"dgp": dgp, "return_grid": False, **SIM_KWARGS}, id=dgp)
    @pytask.mark.r(
        script=SRC / "wa_replication" / "wa_replication.R",
        serializer="yaml",
    )
    def task_wa_replication(
        produces: Path = BLD / "wa_replication" / f"wa_replication_{dgp}.rds",
    ):
        """Produce example figure using grf in R."""
