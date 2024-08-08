import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from econml.grf import CausalForest  # type: ignore[import-untyped]
from wgan import DataWrapper, Generator  # type: ignore[import-untyped]

from topics_causal_inf.classes import DGP
from topics_causal_inf.utilities import data_wager_athey_2018, data_wgan


def simulation(
    n_sim: int,
    n_obs: int,
    dim: int,
    num_trees: int,
    sample_fraction: float,
    dgp: DGP,
    rng: np.random.Generator,
    data_generator: str,
    pop_data: pd.DataFrame | None = None,
    generators: list[Generator] | None = None,
    data_wrappers: list[DataWrapper] | None = None,
) -> np.ndarray:
    """Run simulation using the generic_ml approach and WA 2018 DGP.

    Arguments:
        n_sim: Number of simulations.
        n_obs: Number of observations.
        dim: Dimension of covariates.
        num_trees: Number of trees in the causal forest.
        sample_fraction: Fraction of samples to use in each tree.
        dgp: DGP to simulate data from.
        rng: Random number generator.
        data_generator: Data generator to use.
        pop_data: Population data for WGAN data generator.
        generators: Generators for WGAN data generator.
        data_wrappers: Data wrappers for WGAN data generator.

    Returns:
        np.ndarray: Mean squared error of treatment effect prediction.


    """
    mse = np.zeros(n_sim)

    for i in range(n_sim):
        mse[i] = _experiment(
            n_obs=n_obs,
            dim=dim,
            num_trees=num_trees,
            sample_fraction=sample_fraction,
            dgp=dgp,
            data_generator=data_generator,
            rng=rng,
            pop_data=pop_data,
            generators=generators,
            data_wrappers=data_wrappers,
        )

    return mse


def _experiment(
    n_obs: int,
    dim: int,
    num_trees: int,
    sample_fraction: float,
    dgp: DGP,
    data_generator: str,
    rng: np.random.Generator,
    pop_data: pd.DataFrame | None = None,
    generators: list[Generator] | None = None,
    data_wrappers: list[DataWrapper] | None = None,
) -> float:
    # Draw data
    if data_generator == "standard":
        data, data_test = (
            data_wager_athey_2018(
                n_obs=n_obs,
                dim=dim,
                dgp=dgp,
                rng=rng,
            )
            for _ in range(2)
        )

    elif data_generator == "wgan":
        data, data_test = (
            data_wgan(
                n_obs=n_obs,
                pop_data=pop_data,
                data_wrappers=data_wrappers,  # type: ignore[arg-type]
                generators=generators,  # type: ignore[arg-type]
            )
            for _ in range(2)
        )

    # Fit causal forest
    cf = CausalForest(
        n_estimators=num_trees,
        max_samples=sample_fraction,
    )
    cf.fit(
        X=data.drop(columns=["y", "d", "p_z"]),
        T=data["d"],
        y=data["y"],
    )

    # Predict treatment effect
    pred = cf.predict(
        X=data_test.drop(columns=["y", "d", "p_z"]),
    )[:, 0]

    true = dgp.treatment_effect(data_test[[f"x{i}" for i in range(dim)]])

    return np.mean((pred - true) ** 2)
