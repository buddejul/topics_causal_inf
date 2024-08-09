from functools import partial

import numpy as np

from topics_causal_inf.classes import DGP
from topics_causal_inf.utilities import (
    main_constant,
    main_linear,
    prop_beta,
    prop_cons,
    tau_constant,
    tau_heterog,
)

DGP1 = DGP(
    name="dgp1",
    main_effect=main_linear,
    propensity_score=partial(prop_beta, a=2, b=4),
    treatment_effect=partial(tau_constant, tau=0),
)

DGP2 = DGP(
    name="dgp2",
    main_effect=partial(main_constant, main=0),
    propensity_score=partial(prop_cons, prop=0.5),
    treatment_effect=partial(tau_heterog, x_range=np.arange(0, 2), a=20, b=1 / 3),
)

DGP3 = DGP(
    name="dgp3",
    main_effect=partial(main_constant, main=0),
    propensity_score=partial(prop_cons, prop=0.5),
    treatment_effect=partial(tau_heterog, x_range=np.arange(0, 2), a=12, b=1 / 2),
)

DGP4 = DGP(
    name="dgp4",
    main_effect=partial(main_constant, main=0),
    propensity_score=partial(prop_cons, prop=0.5),
    treatment_effect=partial(tau_heterog, x_range=np.arange(0, 4), a=12, b=1 / 2),
)

DGP5 = DGP(
    name="dgp5",
    main_effect=partial(main_constant, main=0),
    propensity_score=partial(prop_cons, prop=0.5),
    treatment_effect=partial(tau_heterog, x_range=np.arange(0, 8), a=12, b=1 / 2),
)

DGPS = [DGP1, DGP2, DGP3, DGP4, DGP5]
