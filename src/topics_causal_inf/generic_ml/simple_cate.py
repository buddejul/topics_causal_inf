import numpy as np
from sklearn.linear_model import ElasticNetCV  # type: ignore[import-untyped]

from topics_causal_inf.config import RNG

N_OBS = 2_000
P = 10

X = RNG.normal(size=(N_OBS, P))

# Create a 102 x P matrix with each column containing 101 equally spaced points
X_TEST = np.tile(np.linspace(-2, 2, num=100), P).reshape(P, 100).T

W = RNG.binomial(1, 0.4 + 0.2 * (X[:, 0] > 0), size=N_OBS)
Y = np.max(X[:, 0], 0) * W + X[:, 2] + np.min(X[:, 2], 0) + RNG.normal(size=N_OBS)

model0 = ElasticNetCV()
model1 = ElasticNetCV()

fit0 = model0.fit(
    X=X[W == 0],
    y=Y[W == 0],
)

fit1 = model1.fit(
    X=X[W == 1],
    y=Y[W == 1],
)

s_z = fit1.predict(X_TEST) - fit0.predict(X_TEST)
