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
    alpha: float = 0.05,
    wgan_train_data: pd.DataFrame | None = None,
    generators: list[Generator] | None = None,
    data_wrappers: list[DataWrapper] | None = None,
) -> pd.DataFrame:
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
        alpha: Significance level; 1-alpha is the (asymptotic) coverage probability.
        data_wrappers: Data wrappers for WGAN data generator.

    Returns:
        pd.DataFrame: Mean squared error of treatment effect prediction.


    """
    if data_generator == "wgan":
        _check_wgan_args(wgan_train_data, generators, data_wrappers)
    mse = np.zeros(n_sim)
    mse_pop_cate = np.zeros(n_sim)
    coverage = np.zeros(n_sim)

    if data_generator == "wgan":
        pop_data = _generate_pop_data(
            wgan_train_data=wgan_train_data,
            data_generators=generators,  # type: ignore[arg-type]
            data_wrappers=data_wrappers,  # type: ignore[arg-type]
        )
        pop_cate = _estimate_cf_cate(pop_data, dim)
    else:
        pop_data = None
        pop_cate = None

    for i in range(n_sim):
        mse[i], mse_pop_cate[i], coverage[i] = _experiment(
            n_obs=n_obs,
            dim=dim,
            num_trees=num_trees,
            sample_fraction=sample_fraction,
            dgp=dgp,
            data_generator=data_generator,
            alpha=alpha,
            pop_data=pop_data,
            pop_cate=pop_cate,
            rng=rng,
        )

    return pd.DataFrame(
        {
            "mse": mse,
            "mse_pop_cate": mse_pop_cate,
            "coverage": coverage,
            "dim": dim,
            "data_generator": data_generator,
            "dgp": dgp.name,
        },
    )


def _experiment(
    n_obs: int,
    dim: int,
    num_trees: int,
    sample_fraction: float,
    dgp: DGP,
    data_generator: str,
    rng: np.random.Generator,
    alpha: float = 0.05,
    pop_data: pd.DataFrame | None = None,
    pop_cate: CausalForest | None = None,
) -> tuple[float, float, float]:
    feature_names = [f"x{i}" for i in range(dim)]

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
        X=data[feature_names],
        T=data["d"],
        y=data["y"],
    )

    pred = cf.predict(X=data_test[feature_names]).flatten()
    true = dgp.treatment_effect(data_test[feature_names])

    # Calculate (expected) MSE
    mse = np.mean((pred - true) ** 2)

    if data_generator == "wgan" and pop_cate is not None:
        true_pop_cate = pop_cate.predict(data_test[feature_names]).flatten()
        mse_pop_cate = np.mean((pred - true_pop_cate) ** 2)
    else:
        mse_pop_cate = np.nan

    # Calculate (expected) coverage
    ci_lo, ci_hi = cf.predict_interval(X=data_test[feature_names], alpha=alpha)
    coverage = np.mean((ci_lo.flatten() <= true) & (true <= ci_hi.flatten()))

    return mse, mse_pop_cate, coverage


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


def _estimate_cf_cate(pop: pd.DataFrame, dim: int) -> CausalForest:
    feature_names = [f"x{i}" for i in range(dim)]
    cf = CausalForest(max_samples=0.2)
    cf.fit(
        X=pop[feature_names],
        T=pop["d"],
        y=pop["y"],
    )

    return cf
