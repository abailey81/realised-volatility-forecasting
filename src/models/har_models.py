"""
HAR-family models.

Implements:

* **HAR** (Corsi 2009):

  .. math::

      RV_t = \\beta_0 + \\beta_1 RV_{t-1} + \\beta_2 RV_{t-1|t-5}
             + \\beta_3 RV_{t-1|t-22} + u_t

* **LogHAR** (Corsi 2009): HAR on :math:`\\log RV`. Predictions are
  back-transformed with a Jensen-inequality correction:

  .. math::

      \\hat{RV}_t = \\exp\\!\\bigl(\\widehat{\\log RV_t} + \\tfrac{1}{2}\\hat{\\sigma}^2\\bigr)

  where :math:`\\hat{\\sigma}^2` is the variance of the in-sample residuals.

* **LevHAR** (Corsi & Renò 2012): HAR augmented with past aggregated negative
  returns at the three frequencies (leverage effect).

* **SHAR** (Patton & Sheppard 2015): replaces :math:`RV_{t-1}` by
  :math:`RV^+_{t-1}` and :math:`RV^-_{t-1}`, retaining :math:`RV_{t-1|t-5}` and
  :math:`RV_{t-1|t-22}`.

* **HARQ** (Bollerslev, Patton & Quaedvlieg 2016): HAR with an interaction
  term :math:`\\sqrt{RQ_{t-1}} RV_{t-1}` that corrects for measurement error
  in the daily lag. An "insanity filter" replaces predictions outside the
  observed RV range with the closest in-sample RV.

All HAR variants use ordinary least squares via the closed-form normal
equations; no iterative optimisation is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


# ---------------------------------------------------------------------------
# Closed-form OLS with an explicit intercept and stable lstsq fallback
# ---------------------------------------------------------------------------

def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """OLS with explicit intercept. Returns (beta, intercept, residuals)."""
    n, _ = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    # lstsq is numerically robust for near-collinear columns.
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept = coef[0]
    beta = coef[1:]
    resid = y - (X @ beta + intercept)
    return beta, float(intercept), resid


# ---------------------------------------------------------------------------
# Plain HAR
# ---------------------------------------------------------------------------

class HAR(Forecaster):
    name = "HAR"
    required_features = ("RVD", "RVW", "RVM")

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        cols = self.feature_columns(X)
        self.beta_, self.intercept_, _ = _fit_ols(X[cols].to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns(X)
        return X[cols].to_numpy() @ self.beta_ + self.intercept_


# ---------------------------------------------------------------------------
# LogHAR with Jensen bias correction
# ---------------------------------------------------------------------------

class LogHAR(Forecaster):
    name = "LogHAR"
    required_features = ("RVD", "RVW", "RVM")

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.sigma2_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogHAR":
        cols = self.feature_columns(X)
        # Log-transform features and target.
        Xv = np.log(np.clip(X[cols].to_numpy(), 1e-16, None))
        yv = np.log(np.clip(y.to_numpy(), 1e-16, None))
        self.beta_, self.intercept_, resid = _fit_ols(Xv, yv)
        self.sigma2_ = float(np.var(resid, ddof=Xv.shape[1] + 1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns(X)
        Xv = np.log(np.clip(X[cols].to_numpy(), 1e-16, None))
        log_pred = Xv @ self.beta_ + self.intercept_
        # E[exp(Z)] = exp(mu + sigma^2 / 2) when Z is Gaussian.
        return np.exp(log_pred + 0.5 * (self.sigma2_ or 0.0))


# ---------------------------------------------------------------------------
# LevHAR — leverage effect via aggregated negative returns
# ---------------------------------------------------------------------------

class LevHAR(Forecaster):
    name = "LevHAR"
    required_features = ("RVD", "RVW", "RVM", "Rn_D", "Rn_W", "Rn_M")

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LevHAR":
        cols = self.feature_columns(X)
        self.beta_, self.intercept_, _ = _fit_ols(X[cols].to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns(X)
        return X[cols].to_numpy() @ self.beta_ + self.intercept_


# ---------------------------------------------------------------------------
# SHAR — semivariance HAR (Patton & Sheppard 2015)
# ---------------------------------------------------------------------------

class SHAR(Forecaster):
    name = "SHAR"
    required_features = ("RVD_pos", "RVD_neg", "RVW", "RVM")

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SHAR":
        cols = self.feature_columns(X)
        self.beta_, self.intercept_, _ = _fit_ols(X[cols].to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns(X)
        return X[cols].to_numpy() @ self.beta_ + self.intercept_


# ---------------------------------------------------------------------------
# HARQ with insanity filter
# ---------------------------------------------------------------------------

class HARQ(Forecaster):
    name = "HARQ"
    required_features = ("RVD", "RVW", "RVM", "RQ_x_RV")

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.train_min_: float | None = None
        self.train_max_: float | None = None
        self.train_mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARQ":
        cols = self.feature_columns(X)
        self.beta_, self.intercept_, _ = _fit_ols(X[cols].to_numpy(), y.to_numpy())
        self.train_min_ = float(y.min())
        self.train_max_ = float(y.max())
        self.train_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns(X)
        raw = X[cols].to_numpy() @ self.beta_ + self.intercept_
        # Insanity filter per Bollerslev, Patton & Quaedvlieg (2016 §3.3):
        # forecasts outside the in-sample range of the dependent variable
        # are clipped to the closest endpoint of the in-sample range.
        return np.clip(raw, self.train_min_, self.train_max_)


# ---------------------------------------------------------------------------
# HAR-X — HAR plus the extended feature set (Z_{t-1} includes macro etc.)
# ---------------------------------------------------------------------------

class HARX(Forecaster):
    """HAR augmented with the full M_ALL feature set.

    Uses every available column except the HAR-extension helpers
    (``RVD_pos``, ``RVD_neg``, leverage terms, quarticity interaction).
    """
    name = "HAR-X"
    required_features = None  # accept all columns from the M_ALL set

    _exclude = {"RVD_pos", "RVD_neg", "Rn_D", "Rn_W", "Rn_M", "RQ_x_RV", "RQ_lag", "y"}

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.feature_names_: list[str] | None = None

    def _cols(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if c not in self._exclude]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARX":
        cols = self._cols(X)
        self.feature_names_ = cols
        self.beta_, self.intercept_, _ = _fit_ols(X[cols].to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_names_].to_numpy() @ self.beta_ + self.intercept_


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HAR_REGISTRY: dict[str, type[Forecaster]] = {
    "HAR":    HAR,
    "LogHAR": LogHAR,
    "LevHAR": LevHAR,
    "SHAR":   SHAR,
    "HARQ":   HARQ,
    "HAR-X":  HARX,
}


def make_har(label: str) -> Forecaster:
    """Instantiate a HAR-family model by label."""
    if label not in HAR_REGISTRY:
        raise KeyError(f"Unknown HAR variant '{label}'. Known: {list(HAR_REGISTRY)}")
    return HAR_REGISTRY[label]()
