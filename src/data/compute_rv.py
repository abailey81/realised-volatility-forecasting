"""
Realised variance, realised semivariances, and realised quarticity
computation from minute-bar data.

Formulas follow the standard high-frequency volatility literature:

Realised variance (Andersen & Bollerslev 1998; Barndorff-Nielsen & Shephard 2002):

.. math::

    RV_t = \\sum_{j=1}^{n} (\\Delta^n_{t-1,j} X)^2

where :math:`\\Delta^n_{t-1,j} X = X_{t-1 + j/n} - X_{t-1 + (j-1)/n}` is the
:math:`j`-th intraday log-return on day :math:`t` and :math:`n` is the number
of intraday returns at the chosen sampling frequency (78 for 5-minute
returns over a 6.5-hour session).

Realised semivariances (Barndorff-Nielsen, Kinnebrock & Shephard 2010):

.. math::

    RV^{+}_t = \\sum_{j} (\\Delta X)^2 \\, \\mathbb{1}\\{\\Delta X > 0\\}, \\quad
    RV^{-}_t = \\sum_{j} (\\Delta X)^2 \\, \\mathbb{1}\\{\\Delta X < 0\\}

Realised quarticity (used in the HARQ model of Bollerslev, Patton & Quaedvlieg
2016):

.. math::

    RQ_t = \\frac{n}{3} \\sum_{j=1}^{n} (\\Delta^n_{t-1,j} X)^4

Outputs are produced at daily frequency, indexed by trading-day timestamps.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..utils import get_logger, resolve, load_config
from .tick_to_minute import load_and_clean

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-sampled price grid → log-return series
# ---------------------------------------------------------------------------

def resample_to_frequency(
    minute_df: pd.DataFrame,
    minutes: int,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Sample the price every ``minutes`` minutes within each trading day.

    Resampling is performed on the last observed price within each interval
    (i.e. ``label="right"`` semantics) within each trading day. Overnight
    boundaries are respected: no return is computed across a day boundary.

    Returns a DataFrame indexed by the bar timestamp, with columns
    ``[price, log_return]``. The first bar of each day has ``NaN`` in
    ``log_return`` (no prior intraday price).
    """
    if minute_df.empty:
        return pd.DataFrame(columns=["price", "log_return"])

    rule = f"{minutes}min"
    # Resample within each trading day so overnight returns are not introduced.
    out_pieces: list[pd.DataFrame] = []
    for day, group in minute_df.groupby(minute_df.index.normalize()):
        resampled = group[price_col].resample(rule, label="right", closed="right").last()
        resampled = resampled.dropna()
        if len(resampled) < 2:
            continue
        log_ret = np.log(resampled).diff()
        piece = pd.DataFrame({"price": resampled.values, "log_return": log_ret.values},
                             index=resampled.index)
        out_pieces.append(piece)

    if not out_pieces:
        return pd.DataFrame(columns=["price", "log_return"])
    return pd.concat(out_pieces).sort_index()


# ---------------------------------------------------------------------------
# Realised measures from intraday returns
# ---------------------------------------------------------------------------

def _daily_realised(
    intraday: pd.DataFrame,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Compute RV, RV+, RV-, RQ, and the number of returns for each day."""
    returns = intraday[return_col].dropna()
    day = pd.Series(returns.index.normalize(), index=returns.index)
    grouped = returns.groupby(day)

    rv = grouped.apply(lambda r: float(np.sum(r.values ** 2)))
    rv_pos = grouped.apply(lambda r: float(np.sum(np.where(r.values > 0, r.values ** 2, 0.0))))
    rv_neg = grouped.apply(lambda r: float(np.sum(np.where(r.values < 0, r.values ** 2, 0.0))))
    # Realised quarticity multiplied by n/3 — see Eq. (11) of the paper.
    n_obs = grouped.size().astype(float)
    rq = grouped.apply(lambda r: float(np.sum(r.values ** 4))) * (n_obs / 3.0)

    out = pd.DataFrame({
        "RV": rv,
        "RV_pos": rv_pos,
        "RV_neg": rv_neg,
        "RQ": rq,
        "n_returns": n_obs.astype(int),
    })
    out.index.name = "date"
    return out


def _daily_volume_and_price(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily volume sum and closing price from minute-bar data.

    Used to construct the dollar-volume change (DVOL) feature.
    """
    day = pd.Series(minute_df.index.normalize(), index=minute_df.index)
    grouped_v = minute_df["Volume"].groupby(day).sum()
    grouped_c = minute_df["Close"].groupby(day).last()
    out = pd.DataFrame({"volume": grouped_v.astype(float),
                         "price": grouped_c.astype(float)})
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------
# Open-to-close return (used by some HAR extensions)
# ---------------------------------------------------------------------------

def _daily_return(intraday: pd.DataFrame) -> pd.Series:
    """Open-to-close log-return per day, computed from the resampled prices."""
    prices = intraday["price"]
    day = pd.Series(prices.index.normalize(), index=prices.index)
    grouped = prices.groupby(day)
    first = grouped.first()
    last = grouped.last()
    r = np.log(last) - np.log(first)
    r.index.name = "date"
    r.name = "ret"
    return r


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_realised_measures(
    ticker: str,
    sampling_minutes: int | None = None,
    annualise: bool = False,
) -> pd.DataFrame:
    """Compute daily realised measures for a stock.

    Parameters
    ----------
    ticker
        Ticker symbol (file ``<raw_dir>/<ticker>.txt`` must exist).
    sampling_minutes
        Override the configured intraday sampling frequency. Defaults to
        the configuration (5 minutes).
    annualise
        If True, multiply RV by 252 (trading days per year). Defaults to
        False, returning daily realised variance in its native scale.

    Returns
    -------
    DataFrame indexed by trading date with columns
    ``[RV, RV_pos, RV_neg, RQ, n_returns, ret]``. The daily return
    ``ret`` is the open-to-close log-return derived from the resampled prices.
    """
    cfg = load_config()
    if sampling_minutes is None:
        sampling_minutes = cfg.data.rv_sampling_minutes

    _LOG.info("[%s] cleaning minute bars and resampling to %d-min returns",
              ticker, sampling_minutes)
    minute_df = load_and_clean(ticker, config=cfg)
    intraday = resample_to_frequency(minute_df, minutes=sampling_minutes)
    if intraday.empty:
        raise RuntimeError(f"No intraday returns produced for {ticker}")

    daily = _daily_realised(intraday)
    daily["ret"] = _daily_return(intraday)

    # Daily volume sum and closing price (for the DVOL feature in M_ALL).
    vol_price = _daily_volume_and_price(minute_df)
    daily = daily.join(vol_price, how="left")

    if annualise:
        daily["RV"] = daily["RV"] * 252
        daily["RV_pos"] = daily["RV_pos"] * 252
        daily["RV_neg"] = daily["RV_neg"] * 252
        daily["RQ"] = daily["RQ"] * (252 ** 2)

    _LOG.info("[%s] produced %d daily RV observations (%s to %s)",
              ticker, len(daily), daily.index.min().date(), daily.index.max().date())
    return daily


def save_realised(ticker: str, df: pd.DataFrame, intermediate_dir: str | None = None) -> str:
    """Persist a daily RV DataFrame to the ``data/intermediate`` directory."""
    cfg = load_config()
    if intermediate_dir is None:
        intermediate_dir = cfg.paths.data_intermediate
    out_dir = resolve(intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_rv.parquet"
    df.to_parquet(path)
    _LOG.info("[%s] saved realised measures to %s", ticker, path)
    return str(path)


def load_realised(ticker: str, intermediate_dir: str | None = None) -> pd.DataFrame:
    """Read a previously persisted realised-measures DataFrame."""
    cfg = load_config()
    if intermediate_dir is None:
        intermediate_dir = cfg.paths.data_intermediate
    path = resolve(intermediate_dir) / f"{ticker}_rv.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _smoke_test(ticker: str = "AAPL", n_days: int = 30) -> None:
    """Print the first ``n_days`` daily RV observations as a sanity check."""
    df = compute_realised_measures(ticker)
    head = df.head(n_days)
    print(head.to_string())
    print("\nSummary statistics (RV, annualised σ in %):")
    rv_sigma_pct = np.sqrt(df["RV"] * 252) * 100
    print(rv_sigma_pct.describe().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute daily realised variance.")
    parser.add_argument("--stock", required=True, help="Ticker, e.g. AAPL")
    parser.add_argument("--quick-check", action="store_true",
                        help="Print first 30 days and summary stats; do not save.")
    parser.add_argument("--save", action="store_true",
                        help="Persist the result to data/intermediate/.")
    args = parser.parse_args()

    if args.quick_check:
        _smoke_test(args.stock)
    else:
        out = compute_realised_measures(args.stock)
        if args.save:
            save_realised(args.stock, out)
