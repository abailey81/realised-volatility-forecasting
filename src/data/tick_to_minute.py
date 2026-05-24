"""
Loaders for the raw minute-bar tick data.

The supplied raw files contain comma-separated minute bars in the format::

    MM/DD/YYYY,HH:MM,Open,High,Low,Close,Volume

with Eastern-time-stamped bars covering the regular NYSE session
(9:30 to roughly 13:00 in the supplied data — half-day or session end
varies by file). This module:

1. Reads the CSV.
2. Parses dates/times into a tz-naive ``DatetimeIndex`` representing US/Eastern.
3. Restricts to the regular session (configurable).
4. Applies optional outlier filtering on the ``Close`` price series before
   any downstream computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils import get_logger, resolve, load_config

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------

_COLNAMES = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]


def load_raw_minute_bars(ticker: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a raw minute-bar CSV file for one ticker.

    Parameters
    ----------
    ticker
        Stock ticker, e.g. ``"AAPL"``. The file
        ``<raw_dir>/<ticker>.txt`` must exist.
    raw_dir
        Override the configured raw-data directory. Defaults to the
        ``data/raw`` directory from ``config.yaml``.

    Returns
    -------
    DataFrame with a tz-naive ``DatetimeIndex`` and columns
    ``[Open, High, Low, Close, Volume]``.
    """
    cfg = load_config()
    if raw_dir is None:
        raw_dir = cfg.paths.data_raw
    raw_dir = resolve(raw_dir)
    path = raw_dir / f"{ticker}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=_COLNAMES,
        parse_dates=False,
        engine="c",
    )
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%m/%d/%Y %H:%M",
    )
    df = df.set_index("Datetime").drop(columns=["Date", "Time"])
    df = df.sort_index()

    # Numeric coercion
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    _LOG.info(
        "%s: loaded %d minute bars from %s to %s",
        ticker, len(df), df.index.min(), df.index.max(),
    )
    return df


def restrict_to_regular_session(
    df: pd.DataFrame,
    session_open: str = "09:30",
    session_close: str = "16:00",
) -> pd.DataFrame:
    """Keep only minute bars within the regular trading session.

    The default ``[09:30, 16:00)`` matches the NYSE regular session.
    Bars at exactly ``session_close`` are excluded because they would
    correspond to the close-auction print which sits outside continuous
    trading.
    """
    times = df.index.time
    open_t = pd.Timestamp(session_open).time()
    close_t = pd.Timestamp(session_close).time()
    mask = (times >= open_t) & (times < close_t)
    out = df.loc[mask].copy()
    _LOG.debug("Session restriction: kept %d / %d bars", mask.sum(), len(df))
    return out


# ---------------------------------------------------------------------------
# Outlier filtering (Barndorff-Nielsen et al. 2009-style)
# ---------------------------------------------------------------------------

def filter_outliers(
    df: pd.DataFrame,
    rolling_window_minutes: int = 50,
    sd_multiplier: float = 8.0,
    max_abs_log_return: float = 0.10,
) -> pd.DataFrame:
    """Filter outliers from the ``Close`` price series.

    Implements a simplified version of the Barndorff-Nielsen, Hansen,
    Lunde and Shephard (2009) cleaning procedure:

    1. Drop bars whose price-to-rolling-median deviation exceeds
       ``sd_multiplier`` times the rolling MAD-scaled standard deviation.
    2. Cap absolute log-returns at ``max_abs_log_return`` — minute returns
       above ~10% almost certainly reflect data errors rather than genuine
       jumps within the regular session.

    The rolling window is centred over ±``rolling_window_minutes`` minutes
    around each bar.
    """
    if df.empty:
        return df

    close = df["Close"]
    window = rolling_window_minutes
    rolling_med = close.rolling(window=window, min_periods=10, center=True).median()
    # Robust scale via Median Absolute Deviation, scaled to a Gaussian σ.
    mad = (close - rolling_med).abs().rolling(window=window, min_periods=10, center=True).median()
    sigma_hat = 1.4826 * mad
    deviation = (close - rolling_med).abs()
    mask_far = (sigma_hat > 0) & (deviation > sd_multiplier * sigma_hat)
    n_far = int(mask_far.sum())

    # Log returns within each trading day (no overnight) for spike detection.
    log_close = np.log(close)
    log_ret = log_close.groupby(close.index.date).diff()
    mask_jump = log_ret.abs() > max_abs_log_return
    n_jump = int(mask_jump.sum())

    bad = mask_far | mask_jump
    out = df.loc[~bad].copy()
    if n_far or n_jump:
        _LOG.info(
            "Outlier filter: dropped %d bars (%d local outliers, %d jump-violating)",
            int(bad.sum()), n_far, n_jump,
        )
    return out


# ---------------------------------------------------------------------------
# Trading-day grouping
# ---------------------------------------------------------------------------

def iter_trading_days(df: pd.DataFrame) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield ``(date, day_df)`` pairs for each trading day in ``df``."""
    for day, group in df.groupby(df.index.normalize()):
        if not group.empty:
            yield day, group


def load_and_clean(
    ticker: str,
    config: object | None = None,
) -> pd.DataFrame:
    """Convenience wrapper that loads, sessionises, and outlier-filters."""
    cfg = config if config is not None else load_config()
    df = load_raw_minute_bars(ticker, raw_dir=cfg.paths.data_raw)
    df = restrict_to_regular_session(
        df,
        session_open=cfg.data.session_open,
        session_close=cfg.data.session_close,
    )
    if cfg.data.outlier_filter.enabled:
        df = filter_outliers(
            df,
            rolling_window_minutes=cfg.data.outlier_filter.rolling_window_minutes,
            sd_multiplier=cfg.data.outlier_filter.sd_multiplier,
            max_abs_log_return=cfg.data.outlier_filter.max_abs_log_return,
        )
    return df
