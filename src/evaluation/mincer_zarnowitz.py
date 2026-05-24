"""
Mincer-Zarnowitz forecast efficiency tests.

For a forecast :math:`\\hat y_t` of :math:`y_t`, the Mincer-Zarnowitz (1969)
regression is

.. math::

    y_t = \\alpha + \\beta \\hat y_t + u_t.

A forecast is *unbiased and efficient* under the null :math:`H_0:\\,
(\\alpha,\\beta) = (0, 1)`, equivalent to "the forecast errors are mean-zero
and uncorrelated with the forecast itself". This is a stronger requirement
than unbiasedness alone (:math:`\\alpha = 0`, :math:`\\beta` unrestricted).

We provide three things in this module:

1. The OLS regression with Newey-West HAC standard errors.
2. The joint Wald test of :math:`(\\alpha,\\beta) = (0, 1)`.
3. A decomposition of MSE into the bias-squared, regression, and
   disturbance components (Theil 1961):

.. math::

    \\mathrm{MSE} =
        (\\bar{\\hat y} - \\bar y)^2 +
        (\\sigma_{\\hat y} - \\rho \\sigma_y)^2 +
        (1 - \\rho^2) \\sigma_y^2

The bias and regression components are interpretable in their own right
and are reported alongside MSE in the headline table when
``mz_decomposition`` is enabled in the config.

Reference:
    Mincer, J. A., and V. Zarnowitz (1969). "The Evaluation of Economic
    Forecasts". In: *Economic Forecasts and Expectations*, NBER.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MZResult:
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    wald_stat: float
    wald_pvalue: float
    n: int
    r_squared: float
    bias_component: float
    regression_component: float
    disturbance_component: float
    hac_lag: int


# ---------------------------------------------------------------------------
# Newey-West HAC covariance for OLS
# ---------------------------------------------------------------------------

def _newey_west_cov(X: np.ndarray, residuals: np.ndarray,
                    lag: int | None = None) -> np.ndarray:
    """Newey-West HAC covariance of OLS coefficients.

    :math:`\\hat V = (X'X)^{-1} \\, S \\, (X'X)^{-1}` where

    .. math::

        S = \\sum_{i} X_i X_i' u_i^2
            + \\sum_{l=1}^{L} \\bigl(1 - \\tfrac{l}{L+1}\\bigr)
              \\sum_{i} \\bigl(X_i X_{i-l}' + X_{i-l} X_i'\\bigr) u_i u_{i-l}.
    """
    n, k = X.shape
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    XtX_inv = np.linalg.inv(X.T @ X)
    u = residuals.reshape(-1)
    # S
    Z = X * u[:, None]
    S = Z.T @ Z
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1)
        cross = Z[l:].T @ Z[:-l]
        S += weight * (cross + cross.T)
    return XtX_inv @ S @ XtX_inv


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz regression
# ---------------------------------------------------------------------------

def mincer_zarnowitz(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hac_lag: int | None = None,
) -> MZResult:
    """Fit the MZ regression and test joint efficiency.

    The Wald statistic for :math:`H_0: (\\alpha, \\beta) = (0, 1)` is

    .. math::

        W = (\\hat\\theta - \\theta_0)' V^{-1} (\\hat\\theta - \\theta_0)
        \\sim \\chi^2_2

    under the null with HAC-robust :math:`V`.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_true)
    X = np.column_stack([np.ones(n), y_pred])
    coef, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    resid = y_true - X @ coef
    V = _newey_west_cov(X, resid, lag=hac_lag)
    alpha, beta = float(coef[0]), float(coef[1])
    alpha_se, beta_se = float(np.sqrt(V[0, 0])), float(np.sqrt(V[1, 1]))

    # Joint Wald test of (alpha, beta) = (0, 1)
    theta_diff = np.array([alpha - 0.0, beta - 1.0])
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
    wald = float(theta_diff @ V_inv @ theta_diff)
    wald_p = float(1 - stats.chi2.cdf(wald, df=2))

    # R-squared
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Theil decomposition of MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    bias_sq = (y_pred.mean() - y_true.mean()) ** 2
    sd_pred = y_pred.std(ddof=0)
    sd_true = y_true.std(ddof=0)
    rho = float(np.corrcoef(y_true, y_pred)[0, 1]) if sd_pred > 0 and sd_true > 0 else 0.0
    reg_comp = (sd_pred - rho * sd_true) ** 2
    dist_comp = (1 - rho ** 2) * sd_true ** 2

    return MZResult(
        alpha=alpha, beta=beta,
        alpha_se=alpha_se, beta_se=beta_se,
        wald_stat=wald, wald_pvalue=wald_p,
        n=n, r_squared=r2,
        bias_component=float(bias_sq),
        regression_component=float(reg_comp),
        disturbance_component=float(dist_comp),
        hac_lag=hac_lag if hac_lag is not None else int(np.floor(4 * (n / 100) ** (2 / 9))),
    )


def mz_summary_table(forecasts: dict[str, np.ndarray],
                     y_true: np.ndarray) -> pd.DataFrame:
    """Compile MZ regressions for many forecasts into a single table."""
    rows = []
    for name, pred in forecasts.items():
        try:
            r = mincer_zarnowitz(y_true, pred)
            rows.append({
                "model": name,
                "alpha": r.alpha,
                "alpha_se": r.alpha_se,
                "beta":   r.beta,
                "beta_se": r.beta_se,
                "Wald(α=0,β=1)": r.wald_stat,
                "p-value": r.wald_pvalue,
                "R²": r.r_squared,
                "bias_comp": r.bias_component,
                "regr_comp": r.regression_component,
                "disturb_comp": r.disturbance_component,
            })
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows).set_index("model")
