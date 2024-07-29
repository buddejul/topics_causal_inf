"""Tasks for replication of Wager and Athey (2018) results."""

from pathlib import Path

import pytask

from topics_causal_inf.config import BLD, SRC
from topics_causal_inf.wa_replication.task_wa_replication import SIM_KWARGS

for dgp in ["dgp1", "dgp2", "dgp3"]:

    @pytask.task(kwargs={"dgp": dgp, "return_grid": True, **SIM_KWARGS}, id=dgp)
    @pytask.mark.r(
        script=SRC / "wa_replication" / "wa_replication.R",
        serializer="yaml",
    )
    def task_wa_replication_grid(
        produces: Path = BLD / "wa_replication" / f"wa_replication_{dgp}_grid_only.rds",
    ):
        """Produce example figure using grf in R."""
