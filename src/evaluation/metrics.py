"""
Loss functions used by the forecast comparison.

* **MSE** — mean squared error.
* **MAE** — mean absolute error.
* **QLIKE** — quasi-likelihood loss, robust to noise in the realised
  variance proxy (Patton 2011): :math:`L(y,\\hat y) = y/\\hat y - \\log(y/\\hat y) - 1`.

QLIKE is the recommended loss in the volatility-forecasting literature
because it preserves the ranking of forecasts even when the volatility
proxy is noisy, under mild regularity. We report MSE as primary (matching
the paper) and QLIKE as a robustness check.
"""

from __future__ import annotations

import numpy as np


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pointwise squared error."""
    return (y_true - y_pred) ** 2


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pointwise absolute error."""
    return np.abs(y_true - y_pred)


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Pointwise QLIKE loss (Patton 2011).

    Negative or near-zero forecasts are floored before computing the
    ratio; otherwise QLIKE diverges (a model that predicts ŷ ≈ 0 gets a
    near-infinite loss). We use a data-aware floor: the larger of
    ``1e-12`` and the per-call minimum positive ``y_true``. This bounds
    the contribution of a single misbehaving prediction to a finite
    sensible value rather than ~10^8.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if eps is None:
        # Data-aware floor: smallest positive observed RV. Realised-variance
        # values for the daily horizon are ~1e-7 to 1e-3, so 1e-9 is a
        # sensible lower bound that's smaller than every plausible RV but
        # large enough to keep QLIKE bounded under negative forecasts.
        positive = y_true_arr[y_true_arr > 0]
        eps = max(1e-12, float(positive.min() * 1e-3) if positive.size else 1e-12)
    y_pred_arr = np.maximum(y_pred_arr, eps)
    y_true_arr = np.maximum(y_true_arr, eps)
    r = y_true_arr / y_pred_arr
    return r - np.log(r) - 1.0


LOSSES = {
    "mse":   mse_loss,
    "mae":   mae_loss,
    "qlike": qlike_loss,
}


def aggregate_loss(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of the requested loss series."""
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return float(np.mean(LOSSES[name](y_true, y_pred)))
