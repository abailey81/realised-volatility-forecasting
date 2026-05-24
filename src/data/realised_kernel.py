"""
Realised kernel volatility estimator (Barndorff-Nielsen, Hansen, Lunde,
Shephard 2008, 2009, 2011).

The simple realised variance estimator :math:`RV = \\sum_j r_j^2` is biased
by market-microstructure noise (bid-ask bounce, asynchronous trading,
discrete prices) when the sampling frequency is high. The realised kernel
applies a kernel-weighted sum of autocovariances of intraday returns to
obtain an estimator that is consistent for integrated quarticity even
under noise:

.. math::

    RK = \\gamma_0 + \\sum_{h=1}^{H} k\\!\\left(\\frac{h-1}{H}\\right)
         (\\gamma_h + \\gamma_{-h})

where :math:`\\gamma_h = \\sum_{j} r_j r_{j+h}` is the :math:`h`-th sample
autocovariance of intraday returns and :math:`k(\\cdot)` is a kernel
function. We implement the Parzen kernel, which Barndorff-Nielsen et al.
show is optimal in the bias-variance trade-off.

The bandwidth :math:`H^*` is selected via the Barndorff-Nielsen et al.
(2009) optimal rule:

.. math::

    H^* = c^* \\xi^{4/5} n^{3/5}, \\quad c^*_{\\mathrm{Parzen}} = 3.5134

where :math:`\\xi^2 = \\omega^2 / \\sqrt{\\mathrm{IQ}}`, :math:`\\omega^2` is
a preliminary estimate of the noise variance, and :math:`\\mathrm{IQ}` the
integrated quarticity, both estimated from the data.

This module's :func:`compute_realised_kernel` plugs into the same daily
loop as :func:`compute_realised_measures` and saves an additional ``RK``
column. Using RK in place of RV in the HAR family is a clean robustness
check that often improves out-of-sample fit because the right-hand-side
regressors are less noisy.

References:
    Barndorff-Nielsen, O. E., P. R. Hansen, A. Lunde, and N. Shephard (2008).
        "Designing realised kernels to measure the ex-post variation of
        equity prices in the presence of noise." Econometrica 76(6), 1481-1536.
    Barndorff-Nielsen, O. E., P. R. Hansen, A. Lunde, and N. Shephard (2009).
        "Realised kernels in practice." Econometrics Journal 12(3), C1-C32.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve
from .tick_to_minute import load_and_clean

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parzen kernel
# ---------------------------------------------------------------------------

def parzen_kernel(x: np.ndarray) -> np.ndarray:
    """The Parzen kernel evaluated at ``x`` in ``[0, 1]``.

    .. math::

        k(x) = \\begin{cases}
            1 - 6 x^2 + 6 x^3, & 0 \\le x \\le 1/2 \\\\
            2 (1 - x)^3,        & 1/2 < x \\le 1 \\\\
            0,                  & \\text{otherwise}
        \\end{cases}
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask1 = (x >= 0.0) & (x <= 0.5)
    mask2 = (x > 0.5) & (x <= 1.0)
    out[mask1] = 1 - 6 * x[mask1] ** 2 + 6 * x[mask1] ** 3
    out[mask2] = 2 * (1 - x[mask2]) ** 3
    return out


# ---------------------------------------------------------------------------
# Noise variance and IQ estimators
# ---------------------------------------------------------------------------

def _noise_variance(returns: np.ndarray) -> float:
    """Bandi-Russell (2008) noise variance estimator: ω² ≈ ΣR²/(2n)."""
    n = len(returns)
    if n == 0:
        return 0.0
    return float(np.sum(returns ** 2) / (2 * n))


def _integrated_quarticity(returns: np.ndarray) -> float:
    """Preliminary IQ estimator: (n/3) Σ r⁴."""
    n = len(returns)
    if n == 0:
        return 0.0
    return float((n / 3) * np.sum(returns ** 4))


def _optimal_bandwidth(returns: np.ndarray, c_star: float = 3.5134) -> int:
    """Barndorff-Nielsen et al. (2009) optimal Parzen-kernel bandwidth.

    Returns an integer bandwidth :math:`H \\ge 1`.
    """
    n = len(returns)
    if n < 5:
        return max(1, n - 1)
    omega2 = _noise_variance(returns)
    iq = _integrated_quarticity(returns)
    if iq <= 0:
        return max(1, int(round(n ** 0.5)))
    xi2 = omega2 / np.sqrt(iq)
    # BNHLS (2009 Econometrics Journal eq. 24): H* = c* · ξ^(4/5) · n^(3/5)
    # where ξ² = ω² / √IQ. So ξ^(4/5) = (ξ²)^(2/5), NOT (ξ²)^(4/5).
    # The previous (ξ²)^(4/5) form produced a systematically too-small
    # bandwidth (a ~6× underestimate for typical equities).
    H = c_star * (xi2 ** (2 / 5)) * (n ** (3 / 5))
    return max(1, int(round(H)))


# ---------------------------------------------------------------------------
# Realised kernel for one day
# ---------------------------------------------------------------------------

def _daily_realised_kernel(returns: np.ndarray,
                           bandwidth: int | None = None) -> tuple[float, int]:
    """Realised kernel for one trading day given intraday returns."""
    n = len(returns)
    if n == 0:
        return 0.0, 0
    if bandwidth is None:
        bandwidth = _optimal_bandwidth(returns)
    H = min(bandwidth, n - 1)
    if H < 1:
        return float(np.sum(returns ** 2)), 0

    # gamma_0 = sum r^2 (the raw RV)
    gamma_0 = float(np.sum(returns ** 2))
    # weighted autocovariances
    total = gamma_0
    for h in range(1, H + 1):
        gamma_h = float(np.sum(returns[h:] * returns[:-h]))
        w = float(parzen_kernel(np.array([(h - 1) / H]))[0])
        total += 2 * w * gamma_h
    return total, H


# ---------------------------------------------------------------------------
# Top-level: daily RK series for a ticker
# ---------------------------------------------------------------------------

def compute_realised_kernel(
    ticker: str,
    sampling_minutes: int | None = None,
) -> pd.DataFrame:
    """Compute the daily realised kernel for a ticker.

    Parameters
    ----------
    ticker
        Stock symbol with a corresponding raw file in ``data/raw/``.
    sampling_minutes
        Intraday return sampling frequency. Defaults to 1 (minute returns),
        which is the standard input for the realised kernel. Using
        ``sampling_minutes > 1`` is also supported but defeats the kernel's
        purpose of correcting microstructure noise.

    Returns
    -------
    DataFrame indexed by trading date with columns ``[RK, bandwidth]``.
    """
    cfg = load_config()
    if sampling_minutes is None:
        # The realised kernel is typically computed at 1-minute frequency.
        sampling_minutes = 1

    _LOG.info("[%s] computing realised kernel at %d-min frequency",
              ticker, sampling_minutes)
    minute_df = load_and_clean(ticker, config=cfg)

    # Build intraday returns within each trading day. The realised kernel
    # works on contiguous returns within a session; cross-day boundaries
    # introduce overnight effects so we exclude them.
    out_rk: list[tuple[pd.Timestamp, float, int]] = []
    for day, group in minute_df.groupby(minute_df.index.normalize()):
        prices = group["Close"].resample(f"{sampling_minutes}min",
                                          label="right", closed="right").last().dropna()
        if len(prices) < 5:
            continue
        ret = np.log(prices).diff().dropna().to_numpy()
        rk, h_used = _daily_realised_kernel(ret)
        out_rk.append((day, max(rk, 0.0), h_used))

    df = pd.DataFrame(out_rk, columns=["date", "RK", "bandwidth"])
    df = df.set_index("date").sort_index()
    _LOG.info("[%s] produced %d daily RK observations (mean H = %.1f)",
              ticker, len(df), df["bandwidth"].mean())
    return df


def save_realised_kernel(ticker: str, df: pd.DataFrame) -> str:
    cfg = load_config()
    out = resolve(cfg.paths.data_intermediate) / f"{ticker}_rk.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    _LOG.info("[%s] saved realised kernel to %s", ticker, out)
    return str(out)
