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
    wgan_train_data: pd.DataFrame | None = None,
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
        wgan_train_data: Data used to train the WGAN.
        generators: Generators for WGAN data generator.
        data_wrappers: Data wrappers for WGAN data generator.

    Returns:
        np.ndarray: Mean squared error of treatment effect prediction.


    """
    if data_generator == "wgan":
        _check_wgan_args(wgan_train_data, generators, data_wrappers)
    mse = np.zeros(n_sim)

    for i in range(n_sim):
        if data_generator == "wgan":
            pop_data = _generate_pop_data(
                wgan_train_data=wgan_train_data,
                data_generators=generators,  # type: ignore[arg-type]
                data_wrappers=data_wrappers,  # type: ignore[arg-type]
            )
        else:
            pop_data = None

        mse[i] = _experiment(
            n_obs=n_obs,
            dim=dim,
            num_trees=num_trees,
            sample_fraction=sample_fraction,
            dgp=dgp,
            data_generator=data_generator,
            pop_data=pop_data,
            rng=rng,
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
) -> float:
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
    elif data_generator == "wgan" and pop_data is not None:
        # Sample without replacement from population data
        data = pop_data.sample(n_obs, replace=False)
        data_test = pop_data.sample(n_obs, replace=False)
    else:
        msg = "Data generator not implemented or pop_data not provided."
        raise ValueError(msg)

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

    if data_generator == "standard":
        true = dgp.treatment_effect(data_test[[f"x{i}" for i in range(dim)]])
    elif data_generator == "wgan":
        true = data_test["tau"]

    return np.mean((pred - true) ** 2)


def _generate_pop_data(
    wgan_train_data: pd.DataFrame,
    data_generators: list[Generator],
    data_wrappers: list[DataWrapper],
    n_obs_pop: int = 1_000_000,
) -> pd.DataFrame:
    """Draw population data for WGAN data generator."""
    return data_wgan(
        n_obs=n_obs_pop,
        wgan_train_data=wgan_train_data,
        data_wrappers=data_wrappers,
        generators=data_generators,
    )


def _check_wgan_args(
    wgan_train_data: pd.DataFrame | None = None,
    generators: list[Generator] | None = None,
    data_wrappers: list[DataWrapper] | None = None,
) -> None:
    # Check wgan_train_data is a pd.DataFrame
    msg = ""
    if not isinstance(wgan_train_data, pd.DataFrame):
        msg += "wgan_train_data must be a pd.DataFrame."

    # Check generators is a list of type Generator
    if not isinstance(generators, list):
        msg += "generators must be a list."
    elif not all(isinstance(gen, Generator) for gen in generators):
        msg += "generators must be a list of Generator."

    # Check data_wrappers is a list of type DataWrapper
    if not isinstance(data_wrappers, list):
        msg += "data_wrappers must be a list."
    elif not all(isinstance(dw, DataWrapper) for dw in data_wrappers):
        msg += "data_wrappers must be a list of DataWrapper."

    if msg:
        raise ValueError(msg)
