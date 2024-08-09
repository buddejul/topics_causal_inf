"""Example task for grf example in R."""

import pytask

from topics_causal_inf.config import BLD, SRC


@pytask.mark.r(
    script=SRC / "grf" / "grf_example.R",
    serializer="yaml",
)
def task_grf_example_in_r(
    produces=BLD / "grf" / "grf_example_in_r.png",
):
    """Produce example figure using grf in R."""


@pytask.mark.skip()
@pytask.mark.r(
    script=SRC / "grf" / "bruhn_2016.R",
    serializer="yaml",
)
def task_grf_example_bruhn_2016(
    produces=BLD / "grf" / "bruhn_2016.rds",
    bruhn_2016_data=SRC / "grf" / "bruhn2016.csv",
):
    """Produce example figure using grf in R."""
