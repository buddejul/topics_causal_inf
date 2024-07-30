"""Define baseline configurations for WA 2018 replication simulations."""

from typing import NamedTuple


class SimulationDGP(NamedTuple):
    """DGP for WA 2018 style simulations."""

    dgp: str
    n_obs: int
    n_sim: int
    num_trees: int
    sample_fraction: float


DIM_VALS = [2, 8, 15, 30]

SIMULATION2 = SimulationDGP(
    dgp="dgp2",
    n_obs=5_000,
    n_sim=10,
    num_trees=2_000,
    sample_fraction=0.5,
)
SIMULATION3 = SimulationDGP(
    dgp="dgp3",
    n_obs=10_000,
    n_sim=10,
    num_trees=10_000,
    sample_fraction=0.2,
)

SIMS_TO_RUN = [SIMULATION2, SIMULATION3]
