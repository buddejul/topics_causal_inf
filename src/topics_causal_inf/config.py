"""All the general configuration of the project."""

from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()

WGAN_GEN = SRC / "wgan_generated"

DOCUMENTS = ROOT.joinpath("documents").resolve()

TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]

RNG = np.random.default_rng()
