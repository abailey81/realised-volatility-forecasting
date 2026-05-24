"""
Forecast combinations: simple average, MSE-weighted, and Bates-Granger.

The volatility-forecasting literature finds that *combinations* of
forecasts routinely outperform their individual constituents (Timmermann
2006 in the *Handbook of Economic Forecasting*; Stock & Watson 2004).
This is a natural extension to the original paper, which compares
individual ML methods without combining them.

We implement three combination schemes:

1. **Simple average** — weights :math:`w_i = 1/N`. Surprisingly robust
   despite its naivety.

2. **MSE-weighted** — :math:`w_i \\propto 1 / \\mathrm{MSE}_i` on the
   validation set, normalised to sum to 1. Often performs well but can
   over-weight idiosyncratic winners.

3. **Bates-Granger (1969)** — optimal weights under the assumption of
   unbiased forecasts:

   .. math::

       w^* = \\frac{\\Sigma^{-1} \\mathbf{1}}{\\mathbf{1}' \\Sigma^{-1} \\mathbf{1}},

   where :math:`\\Sigma` is the covariance matrix of forecast errors.
   Optionally shrinks toward the diagonal (Ledoit-Wolf style) to combat
   :math:`\\Sigma` instability when :math:`N` is large.

Combinations are typically restricted to non-negative weights (to avoid
betting against models), which we enforce by quadratic programming when
``constrain_nonneg=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombinationWeights:
    method: str
    labels: list[str]
    weights: np.ndarray
    val_mse: float


def simple_average(forecasts: dict[str, np.ndarray]) -> tuple[np.ndarray, CombinationWeights]:
    """Equal-weight average of all forecasts."""
    labels = list(forecasts.keys())
    arr = np.stack([forecasts[l] for l in labels], axis=1)
    w = np.full(len(labels), 1.0 / len(labels))
    pred = arr @ w
    return pred, CombinationWeights(method="simple_average", labels=labels,
                                     weights=w, val_mse=float("nan"))


def mse_weighted(
    y_val: np.ndarray,
    val_forecasts: dict[str, np.ndarray],
    test_forecasts: dict[str, np.ndarray],
) -> tuple[np.ndarray, CombinationWeights]:
    """Weights proportional to inverse MSE on validation."""
    labels = list(val_forecasts.keys())
    mses = np.array([np.mean((y_val - val_forecasts[l]) ** 2) for l in labels])
    w = 1.0 / mses
    w = w / w.sum()
    test_arr = np.stack([test_forecasts[l] for l in labels], axis=1)
    pred = test_arr @ w
    return pred, CombinationWeights(method="mse_weighted", labels=labels,
                                     weights=w, val_mse=float(np.mean(mses @ w)))


def bates_granger(
    y_val: np.ndarray,
    val_forecasts: dict[str, np.ndarray],
    test_forecasts: dict[str, np.ndarray],
    constrain_nonneg: bool = True,
    shrinkage: float = 0.0,
) -> tuple[np.ndarray, CombinationWeights]:
    """Bates-Granger optimal combination using the validation error covariance.

    Parameters
    ----------
    y_val
        Realised target on the validation set.
    val_forecasts, test_forecasts
        Dicts mapping model label to its predictions on val / test.
    constrain_nonneg
        If True, enforce non-negative weights via a simple projected
        gradient step. Otherwise allow short positions in poor models.
    shrinkage
        Ledoit-Wolf-style shrinkage of the covariance matrix toward its
        diagonal: :math:`\\Sigma \\to (1-\\delta) \\Sigma + \\delta \\mathrm{diag}(\\Sigma)`.
    """
    labels = list(val_forecasts.keys())
    err = np.stack([y_val - val_forecasts[l] for l in labels], axis=1)
    Sigma = np.cov(err, rowvar=False, ddof=1)
    if shrinkage > 0:
        Sigma = (1 - shrinkage) * Sigma + shrinkage * np.diag(np.diag(Sigma))
    try:
        S_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(Sigma)
    ones = np.ones(len(labels))
    w = S_inv @ ones / (ones @ S_inv @ ones)

    if constrain_nonneg and (w < 0).any():
        # Project onto the simplex {w : w >= 0, sum w = 1}.
        # Use the algorithm of Wang & Carreira-Perpiñán (2013).
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)
        rho = np.where(u + (1.0 - cssv) / (np.arange(len(u)) + 1) > 0)[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(w - theta, 0.0)

    test_arr = np.stack([test_forecasts[l] for l in labels], axis=1)
    pred = test_arr @ w
    val_pred = np.stack([val_forecasts[l] for l in labels], axis=1) @ w
    val_mse = float(np.mean((y_val - val_pred) ** 2))
    return pred, CombinationWeights(method="bates_granger", labels=labels,
                                     weights=w, val_mse=val_mse)


def build_combination_predictions(
    y_val: pd.Series,
    val_forecasts: dict[str, pd.Series],
    test_forecasts: dict[str, pd.Series],
    methods: tuple[str, ...] = ("simple_average", "mse_weighted", "bates_granger"),
) -> dict[str, pd.Series]:
    """Run all configured combinations and return predictions as Series."""
    labels = list(val_forecasts.keys())
    val_idx = y_val.index
    common_test = None
    for s in test_forecasts.values():
        common_test = s.index if common_test is None else common_test.intersection(s.index)

    val_arrays = {l: val_forecasts[l].loc[val_idx].to_numpy() for l in labels}
    test_arrays = {l: test_forecasts[l].loc[common_test].to_numpy() for l in labels}
    y_val_arr = y_val.to_numpy()

    out: dict[str, pd.Series] = {}
    if "simple_average" in methods:
        pred, _ = simple_average(test_arrays)
        out["COMB_AVG"] = pd.Series(pred, index=common_test, name="COMB_AVG")
    if "mse_weighted" in methods:
        pred, _ = mse_weighted(y_val_arr, val_arrays, test_arrays)
        out["COMB_MSE"] = pd.Series(pred, index=common_test, name="COMB_MSE")
    if "bates_granger" in methods:
        pred, _ = bates_granger(y_val_arr, val_arrays, test_arrays)
        out["COMB_BG"] = pd.Series(pred, index=common_test, name="COMB_BG")
    return out
