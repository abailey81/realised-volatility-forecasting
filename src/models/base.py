"""
Base class for all forecasting models.

All HAR, regularised, tree-based, and neural-network models implement this
small interface so they can be passed interchangeably to the rolling-window
forecaster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Minimal forecaster interface.

    Implementations are expected to fit a regression of ``y`` on a subset
    of columns from ``X``. Subclasses declare which columns they need via
    :attr:`required_features`; the orchestrator will pass only those columns
    to ``fit`` and ``predict``.
    """

    #: Human-readable label used in tables and figures.
    name: str = "BASE"

    #: Tuple of column names required from the input frame.
    #: ``None`` means "use all columns".
    required_features: tuple[str, ...] | None = None

    def feature_columns(self, X: pd.DataFrame) -> Sequence[str]:
        if self.required_features is None:
            return list(X.columns)
        missing = set(self.required_features) - set(X.columns)
        if missing:
            raise KeyError(f"{self.name} missing features: {sorted(missing)}")
        return list(self.required_features)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts on ``X``."""

    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame) -> np.ndarray:
        """Convenience: fit on train, return predictions on test."""
        self.fit(X_train, y_train)
        return self.predict(X_test)
