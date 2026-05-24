"""
Tree-based regression: Bagging, Random Forest, Gradient Boosting.

Per the paper's Section 1.4, the Random Forest and Bagging hyperparameters
are set to their defaults from Breiman & Cutler's original Fortran
implementation to "sidestep validation and facilitate reproduction of
findings". Scikit-learn's defaults for ``RandomForestRegressor`` and
``BaggingRegressor`` are essentially the same parameter set, with
``max_features="sqrt"`` (or 1/3 of total in some versions) and
``bootstrap=True``.

Gradient Boosting uses the same family but expects more aggressive
hyperparameter tuning; the grid in ``config.yaml`` is searched on the
validation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from .base import Forecaster


@dataclass
class _TreeResult:
    params: dict = field(default_factory=dict)
    val_mse: float = float("inf")


# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------

class BaggingForecaster(Forecaster):
    name = "BG"

    def __init__(self, n_estimators: int = 500, bootstrap: bool = True, random_state: int = 42):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model_: BaggingRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaggingForecaster":
        base = DecisionTreeRegressor(random_state=self.random_state)
        self.model_ = BaggingRegressor(
            estimator=base,
            n_estimators=self.n_estimators,
            bootstrap=self.bootstrap,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy())


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestForecaster(Forecaster):
    name = "RF"

    def __init__(self, n_estimators: int = 500, max_features: str = "sqrt",
                 bootstrap: bool = True, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model_: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy())


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

class GradientBoostingForecaster(Forecaster):
    """Gradient Boosting with validation-set tuning of (n_est, lr, depth)."""

    name = "GB"

    def __init__(self,
                 n_estimators_grid: Sequence[int],
                 learning_rate_grid: Sequence[float],
                 max_depth_grid: Sequence[int],
                 subsample: float = 0.8,
                 random_state: int = 42,
                 val_frac: float = 0.10 / 0.80):
        self.n_estimators_grid = list(n_estimators_grid)
        self.learning_rate_grid = list(learning_rate_grid)
        self.max_depth_grid = list(max_depth_grid)
        self.subsample = subsample
        self.random_state = random_state
        self.val_frac = val_frac
        self.model_: GradientBoostingRegressor | None = None
        self.diagnostics: _TreeResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "GradientBoostingForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val

        best = _TreeResult(params={})
        for n_est in self.n_estimators_grid:
            for lr in self.learning_rate_grid:
                for depth in self.max_depth_grid:
                    gbr = GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        subsample=self.subsample,
                        random_state=self.random_state,
                    )
                    gbr.fit(X_tr.to_numpy(), y_tr.to_numpy())
                    pred = gbr.predict(X_vl.to_numpy())
                    mse = float(np.mean((pred - y_vl.to_numpy()) ** 2))
                    if mse < best.val_mse:
                        best = _TreeResult(params={"n_estimators": n_est,
                                                    "learning_rate": lr,
                                                    "max_depth": depth},
                                            val_mse=mse)

        full_X = pd.concat([X_tr, X_vl])
        full_y = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        self.model_ = GradientBoostingRegressor(
            **best.params, subsample=self.subsample, random_state=self.random_state
        ).fit(full_X.to_numpy(), full_y)
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy())
