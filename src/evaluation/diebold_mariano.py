"""
Diebold-Mariano test for equal predictive accuracy (Diebold & Mariano 1995).

For two competing forecast sequences with loss differences

.. math::

    d_t = L(\\varepsilon^{(A)}_t) - L(\\varepsilon^{(B)}_t)

the DM statistic under the null :math:`E[d_t] = 0` is

.. math::

    DM = \\frac{\\bar d}{\\sqrt{\\widehat{\\mathrm{LRV}}(d_t) / n}}

where :math:`\\widehat{\\mathrm{LRV}}(d_t)` is a long-run-variance estimator
robust to autocorrelation (we use Newey-West with automatic bandwidth
selection per Newey & West 1994).

Under standard regularity (stationarity, mixing, finite fourth moments)
:math:`DM \\to_d N(0,1)` as :math:`n\\to\\infty`. We also provide the
Harvey-Leybourne-Newbold (1997) small-sample correction.

Reference:
    Diebold, F. X., and R. S. Mariano (1995). "Comparing Predictive Accuracy".
    Journal of Business and Economic Statistics, 13(3), 253-263.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .metrics import LOSSES


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    n: int
    lag: int
    loss: str
    alternative: str
    mean_diff: float


# ---------------------------------------------------------------------------
# Newey-West long-run variance
# ---------------------------------------------------------------------------

def _newey_west_lrv(x: np.ndarray, lag: int | None = None) -> float:
    """Newey-West long-run variance estimator.

    The bandwidth defaults to :math:`\\lfloor 4 (n/100)^{2/9} \\rfloor`
    if ``lag`` is None (Newey-West 1994 automatic).
    """
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lag = max(lag, 0)
    gamma0 = float(np.sum(x ** 2) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        cov = float(np.sum(x[k:] * x[:-k]) / n)
        weight = 1.0 - k / (lag + 1)
        lrv += 2.0 * weight * cov
    return max(lrv, 0.0)


# ---------------------------------------------------------------------------
# Diebold-Mariano statistic
# ---------------------------------------------------------------------------

def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "mse",
    alternative: str = "less",
    lag: int | None = None,
    hln_correction: bool = True,
    horizon: int = 1,
) -> DMResult:
    """Compute the DM test for two competing forecasts.

    The null hypothesis is :math:`H_0:\\, E[d_t] = 0`. The alternative
    ``"less"`` corresponds to :math:`H_1:\\, E[d_t] < 0`, i.e. forecast A
    has a smaller loss than B on average. Use ``"two-sided"`` for the
    standard symmetric alternative.

    With ``hln_correction=True``, we apply the Harvey-Leybourne-Newbold
    (1997) finite-sample adjustment, dividing the variance by
    :math:`(n + 1 - 2h + h(h-1)/n) / n` and comparing to a Student-:math:`t`
    distribution with :math:`n-1` degrees of freedom.
    """
    if loss not in LOSSES:
        raise KeyError(f"Unknown loss '{loss}'")
    loss_fn = LOSSES[loss]
    d = loss_fn(y_true, pred_a) - loss_fn(y_true, pred_b)
    n = len(d)
    mean_d = float(np.mean(d))
    lrv = _newey_west_lrv(d, lag=lag)
    if lrv <= 0:
        return DMResult(np.nan, np.nan, n, lag or 0, loss, alternative, mean_d)
    dm = mean_d / np.sqrt(lrv / n)

    eff_lag = lag if lag is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))

    if hln_correction:
        h = horizon
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        dm = dm * np.sqrt(factor)
        if alternative == "two-sided":
            pval = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
        elif alternative == "less":
            pval = stats.t.cdf(dm, df=n - 1)
        elif alternative == "greater":
            pval = 1 - stats.t.cdf(dm, df=n - 1)
        else:
            raise ValueError(f"Unknown alternative '{alternative}'")
    else:
        if alternative == "two-sided":
            pval = 2 * (1 - stats.norm.cdf(abs(dm)))
        elif alternative == "less":
            pval = stats.norm.cdf(dm)
        elif alternative == "greater":
            pval = 1 - stats.norm.cdf(dm)
        else:
            raise ValueError(f"Unknown alternative '{alternative}'")

    return DMResult(
        statistic=float(dm),
        pvalue=float(pval),
        n=n,
        lag=eff_lag,
        loss=loss,
        alternative=alternative,
        mean_diff=mean_d,
    )


# ---------------------------------------------------------------------------
# Pairwise DM matrix
# ---------------------------------------------------------------------------

def dm_matrix(
    y_true: np.ndarray,
    forecasts: dict[str, np.ndarray],
    loss: str = "mse",
    alternative: str = "less",
    horizon: int = 1,
) -> "tuple[np.ndarray, np.ndarray, list[str]]":
    """Compute the pairwise DM statistic and p-value matrices.

    Returns ``(stat_matrix, pval_matrix, labels)`` where entry ``[i, j]`` tests
    whether model ``i`` has lower loss than model ``j``.
    """
    labels = list(forecasts.keys())
    k = len(labels)
    stat = np.full((k, k), np.nan)
    pval = np.full((k, k), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            r = diebold_mariano(y_true, forecasts[a], forecasts[b],
                                loss=loss, alternative=alternative, horizon=horizon)
            stat[i, j] = r.statistic
            pval[i, j] = r.pvalue
    return stat, pval, labels
