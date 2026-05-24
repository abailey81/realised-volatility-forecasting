"""
Feature engineering: build the M_HAR and M_ALL feature matrices.

The paper defines two feature sets:

* :math:`\\mathcal{M}_{HAR}`: the three HAR lags only — daily ``RVD``, weekly
  ``RVW`` (average of the past 5 daily RVs), monthly ``RVM`` (average of
  the past 22 daily RVs). Used to make the HAR-vs-ML comparison apples-to-apples.

* :math:`\\mathcal{M}_{ALL}`: ``M_HAR`` plus nine additional predictors
  (Table 1 of the paper) — IV, EA, M1W, DVOL, VIX, EPU, HSI, ADS, US3M.

The target at horizon h is the average RV over the window
:math:`t, \\ldots, t+h-1`, paired with features observable through
:math:`t-1` (``RVD`` = :math:`RV_{t-1}`):

.. math::

    y_t^{(h)} = \\frac{1}{h} \\sum_{k=0}^{h-1} RV_{t+k}

For h=1 this is simply :math:`RV_t` predicted from :math:`RV_{t-1}` — Corsi
(2009) Eq. (5). The one-day gap between the freshest predictor and the first
target day is what makes this a genuine one-step-ahead forecast.

Outputs are saved as parquet files keyed by ``(ticker, feature_set, horizon)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve
from .compute_rv import load_realised
from .macro_features import (
    download_fred_series,
    fetch_hang_seng,
    momentum_1w,
    dollar_volume_change,
    earnings_announcement_dummy,
    implied_vol_proxy,
)

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# HAR-style lags
# ---------------------------------------------------------------------------

def build_har_lags(rv: pd.Series) -> pd.DataFrame:
    """Build the daily / weekly / monthly RV lags used by HAR-family models.

    All lags are aligned so they are observable at the *forecast origin*:

    * ``RVD`` = :math:`RV_{t-1}`
    * ``RVW`` = average of :math:`RV_{t-1}, \\ldots, RV_{t-5}`
    * ``RVM`` = average of :math:`RV_{t-1}, \\ldots, RV_{t-22}`
    """
    rvd = rv.shift(1).rename("RVD")
    rvw = rv.shift(1).rolling(window=5).mean().rename("RVW")
    rvm = rv.shift(1).rolling(window=22).mean().rename("RVM")
    return pd.concat([rvd, rvw, rvm], axis=1)


def build_semivariance_lags(rv_pos: pd.Series, rv_neg: pd.Series) -> pd.DataFrame:
    """Build the positive and negative semivariance lags for SHAR."""
    rvd_pos = rv_pos.shift(1).rename("RVD_pos")
    rvd_neg = rv_neg.shift(1).rename("RVD_neg")
    return pd.concat([rvd_pos, rvd_neg], axis=1)


def build_negative_return_lags(ret: pd.Series) -> pd.DataFrame:
    """Past aggregated negative returns at the three HAR frequencies.

    Used by LevHAR. Defined as ``min(0, r)`` averaged over the relevant window.
    """
    neg = ret.where(ret < 0, 0.0)
    nd  = neg.shift(1).rename("Rn_D")
    nw  = neg.shift(1).rolling(window=5).mean().rename("Rn_W")
    nm  = neg.shift(1).rolling(window=22).mean().rename("Rn_M")
    return pd.concat([nd, nw, nm], axis=1)


def build_quarticity_term(rq: pd.Series, rv: pd.Series) -> pd.Series:
    """The HARQ interaction term :math:`\\sqrt{RQ_{t-1}} \\cdot RV_{t-1}`."""
    rq_lag = rq.shift(1)
    rv_lag = rv.shift(1)
    return (np.sqrt(np.clip(rq_lag, 0, None)) * rv_lag).rename("RQ_x_RV")


# ---------------------------------------------------------------------------
# Targets at multiple horizons
# ---------------------------------------------------------------------------

def make_horizon_target(rv: pd.Series, horizon: int) -> pd.Series:
    """Average RV over the target window :math:`t` through :math:`t+h-1`.

    Row ``t`` carries lagged features observable through ``t-1`` (``RVD`` =
    :math:`RV_{t-1}`, ``RVW`` = mean of :math:`RV_{t-1..t-5}`, etc.). The
    h-step target is therefore the mean of :math:`RV_t, \\ldots, RV_{t+h-1}`,
    i.e. ``mean(rv.shift(-k) for k=0..h-1)``. This leaves a one-day gap
    between the freshest predictor (:math:`RV_{t-1}`) and the first day of
    the target window (:math:`RV_t`), matching Corsi (2009) Eq. (5) exactly
    (predict :math:`RV_t` from :math:`RV_{t-1}`).

    NOTE: an earlier version used ``shift(-k) for k=1..h``, which paired
    :math:`RV_{t-1}` features with an :math:`RV_{t+1}` target — a two-day
    gap that silently discarded the most recent observation :math:`RV_t`
    and turned the h=1 task into a 2-step-ahead forecast. Fixed here.
    """
    parts = [rv.shift(-k) for k in range(horizon)]
    forward = pd.concat(parts, axis=1).mean(axis=1, skipna=False)
    return forward.rename("y")


# ---------------------------------------------------------------------------
# Full feature builder
# ---------------------------------------------------------------------------

def build_features(
    ticker: str,
    feature_set: str = "M_HAR",
    horizon: int = 1,
    use_macro_cache: bool = True,
) -> pd.DataFrame:
    """Construct a model-ready feature DataFrame for one stock.

    Parameters
    ----------
    ticker
        Stock symbol; the corresponding ``<ticker>_rv.parquet`` must exist
        under ``data/intermediate/`` (produced by ``compute_rv``).
    feature_set
        Either ``"M_HAR"`` (three HAR lags) or ``"M_ALL"`` (HAR lags + nine
        additional predictors).
    horizon
        Forecast horizon in trading days. The target is
        :math:`\\frac{1}{h}\\sum_{i=1}^{h} RV_{t+i}`.
    use_macro_cache
        Whether to read the cached FRED parquet (offline-friendly).

    Returns
    -------
    DataFrame with the target column ``y`` and one column per feature.
    Rows with any NaN (from differencing/rolling/target shifts) are dropped.
    """
    cfg = load_config()
    rv_df = load_realised(ticker)

    rv     = rv_df["RV"]
    rv_pos = rv_df["RV_pos"]
    rv_neg = rv_df["RV_neg"]
    rq     = rv_df["RQ"]
    ret    = rv_df["ret"]

    # Core HAR lags (always included).
    har_lags = build_har_lags(rv)
    semi     = build_semivariance_lags(rv_pos, rv_neg)
    neg_ret  = build_negative_return_lags(ret)
    rqx      = build_quarticity_term(rq, rv)

    target = make_horizon_target(rv, horizon)

    parts: dict[str, pd.Series | pd.DataFrame] = {
        "RVD": har_lags["RVD"],
        "RVW": har_lags["RVW"],
        "RVM": har_lags["RVM"],
        # Helpers retained because the HAR variants need them:
        "RVD_pos": semi["RVD_pos"],
        "RVD_neg": semi["RVD_neg"],
        "Rn_D": neg_ret["Rn_D"],
        "Rn_W": neg_ret["Rn_W"],
        "Rn_M": neg_ret["Rn_M"],
        "RQ_x_RV": rqx,
        "RQ_lag":  rq.shift(1).rename("RQ_lag"),
    }

    if feature_set.upper() == "M_ALL":
        extras = set(cfg.data.extra_features) if hasattr(cfg.data, "extra_features") else set()
        if use_macro_cache:
            try:
                macro = pd.read_parquet(resolve(cfg.paths.data_macro) / "fred.parquet")
            except FileNotFoundError:
                _LOG.warning(
                    "No FRED cache found at data/macro/fred.parquet. "
                    "Run scripts/02_download_macro.py first. Falling back to NaN macro."
                )
                macro = pd.DataFrame(index=rv.index, columns=["VIX", "EPU", "ADS", "US3M"])
        else:
            macro = download_fred_series()
        # Align to the trading-day index
        macro = macro.reindex(rv.index, method="ffill")

        m1w   = momentum_1w(ret)
        hsi = fetch_hang_seng().reindex(rv.index, method="ffill") if "HSI" in extras else None
        iv  = implied_vol_proxy(macro, rv.index, method="VIX") if "IV" in extras else None

        extras_map = {
            "IV":   iv.shift(1) if iv is not None else None,
            "M1W":  m1w.shift(1) if "M1W" in extras else None,
            "VIX":  macro.get("VIX").shift(1) if ("VIX" in extras and "VIX" in macro.columns) else None,
            "EPU":  macro.get("EPU").shift(1) if ("EPU" in extras and "EPU" in macro.columns) else None,
            "HSI":  hsi.shift(1) if hsi is not None else None,
            "ADS":  macro.get("ADS").shift(1) if ("ADS" in extras and "ADS" in macro.columns) else None,
            "US3M": (macro["US3M"].diff().shift(1).rename("US3M")
                     if ("US3M" in extras and "US3M" in macro.columns) else None),
        }
        # Optional EA/DVOL — only included if the user populates them.
        # EA is shifted by 1 day for consistency with every other M_ALL
        # extra (IV, M1W, VIX, EPU, HSI, ADS, US3M, DVOL): the feature
        # at day t encodes information observable at t-close, predicting
        # y at t+1. Earnings announcements typically print after the close,
        # so this convention treats EA as "was there an earnings event on
        # the previous day" — the most defensible alignment for
        # next-day-forward forecasting.
        if "EA" in extras:
            ea = earnings_announcement_dummy(rv.index, ticker=ticker)
            extras_map["EA"] = ea.shift(1)
        if "DVOL" in extras and "volume" in rv_df.columns and "price" in rv_df.columns:
            extras_map["DVOL"] = dollar_volume_change(rv_df["price"], rv_df["volume"]).shift(1)
        parts.update(extras_map)

    parts["y"] = target

    df = pd.concat({k: v for k, v in parts.items() if v is not None}, axis=1)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    df = df.dropna()
    _LOG.info(
        "[%s | %s | h=%d] feature matrix: %d rows × %d cols",
        ticker, feature_set, horizon, len(df), df.shape[1],
    )
    return df


def save_feature_matrix(
    df: pd.DataFrame,
    ticker: str,
    feature_set: str,
    horizon: int,
) -> str:
    cfg = load_config()
    out_dir = resolve(cfg.paths.data_features)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}__{feature_set}__h{horizon}.parquet"
    path = out_dir / fname
    df.to_parquet(path)
    _LOG.info("Saved feature matrix to %s", path)
    return str(path)


def load_feature_matrix(ticker: str, feature_set: str, horizon: int) -> pd.DataFrame:
    cfg = load_config()
    path = resolve(cfg.paths.data_features) / f"{ticker}__{feature_set}__h{horizon}.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Train / val / test splitter
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    The remaining fraction (``1 - train_frac - val_frac``) is the test set.
    """
    n = len(df)
    n_train = int(np.floor(n * train_frac))
    n_val   = int(np.floor(n * val_frac))
    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()
    return train, val, test
