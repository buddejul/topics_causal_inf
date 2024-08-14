from topics_causal_inf.config import RNG
from topics_causal_inf.define_dgps import DGP3, DGP4, DGP5, DGP6
from topics_causal_inf.generic_ml.simulation import _simulate_true_gates


def test_simulate_true_gates_runs() -> None:
    for dgp in [DGP3, DGP4, DGP5, DGP6]:
        for dim in [10, 15, 30]:
            _simulate_true_gates(dgp=dgp, dim=dim, rng=RNG)
