"""
Macro / firm-level feature acquisition.

The paper's extended feature set :math:`\\mathcal{M}_{ALL}` augments the three
realised-variance lags with the following predictors:

* ``IV``   — model-free implied volatility (per-stock; OptionMetrics in the
  paper, here approximated with VIX for the index and stock IV proxies where
  available).
* ``EA``   — earnings-announcement dummy.
* ``M1W``  — 1-week momentum: cumulative log-return over the previous 5 days.
* ``DVOL`` — first difference of log dollar trading volume.
* ``VIX``  — CBOE VIX, downloaded from FRED.
* ``EPU``  — Baker–Bloom–Davis Economic Policy Uncertainty Index (FRED).
* ``HSI``  — Hang Seng daily squared log-return.
* ``ADS``  — Aruoba–Diebold–Scotti business-conditions index (FRED).
* ``US3M`` — US 3-month T-bill rate, first-differenced (FRED).

This module separates the API-dependent downloads (FRED, Hang Seng) from the
purely local computations (M1W, DVOL, IV/VIX proxy) so the pipeline can run
offline with the macro cache present.

A FRED API key is required. Get one free at
https://fred.stlouisfed.org/docs/api/api_key.html and export it as the
``FRED_API_KEY`` environment variable before running.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# FRED series identifiers
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "VIX":  "VIXCLS",         # CBOE Volatility Index, daily close
    "EPU":  "USEPUINDXD",     # Daily News-Based Economic Policy Uncertainty
    "ADS":  "ADSWBCIND",      # Aruoba-Diebold-Scotti Business Conditions
    "US3M": "DTB3",           # 3-Month Treasury Bill, secondary market rate
}


def _get_fred_client():
    """Build a fredapi.Fred client when both a key and the library are present.

    Returns ``None`` when no key is set or when ``fredapi`` is not installed —
    in either case, callers fall back to the public fredgraph CSV endpoint.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
    except ImportError:
        _LOG.info("fredapi not installed; using public CSV endpoint")
        return None
    return Fred(api_key=api_key)


def _download_fred_public_csv(series_id: str,
                               start: str = "2015-01-01",
                               end: str | None = None,
                               max_retries: int = 3,
                               timeout: int = 120) -> pd.Series:
    """Fetch a single FRED series via the public fredgraph CSV endpoint.

    Uses curl rather than Python ``requests`` because the latter can stall
    on this endpoint in some sandboxed environments (TLS handshake quirk).
    Falls back to requests if curl is unavailable.
    """
    import io, subprocess, shutil, time

    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    q = f"?id={series_id}"
    if start:
        q += f"&cosd={start}"
    if end:
        q += f"&coed={end}"
    url = base + q

    last_err: Exception | None = None
    curl_path = shutil.which("curl")
    for attempt in range(max_retries):
        try:
            if curl_path is not None:
                # Force HTTP/1.1 — the HTTP/2 path can hang in some sandboxed
                # environments even when the TLS handshake succeeds.
                proc = subprocess.run(
                    [curl_path, "-sS", "--http1.1", "--max-time", str(timeout),
                     "-A", "Mozilla/5.0 rv-ml-replication/1.0", url],
                    capture_output=True, text=True, timeout=timeout + 30, check=True,
                )
                text = proc.stdout
            else:
                import requests
                resp = requests.get(url, timeout=timeout,
                                    headers={"User-Agent": "Mozilla/5.0 rv-ml-replication/1.0"})
                resp.raise_for_status()
                text = resp.text
            df = pd.read_csv(io.StringIO(text))
            date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date")),
                            df.columns[0])
            val_col = next((c for c in df.columns if c != date_col), None)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)[val_col]
            df = pd.to_numeric(df, errors="coerce")
            df.name = series_id
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            sleep_s = 2 ** attempt
            _LOG.warning("FRED CSV fetch for %s failed (attempt %d/%d): %s; retrying in %ds",
                         series_id, attempt + 1, max_retries, exc, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"FRED CSV fetch for {series_id} failed after {max_retries} attempts") from last_err


def download_fred_series(
    series_ids: dict[str, str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download and cache FRED series.

    Uses ``fredapi`` with FRED_API_KEY when set; otherwise falls back to the
    public fredgraph CSV endpoint (no authentication required).

    Parameters
    ----------
    series_ids
        Mapping from short label to FRED series id. Defaults to ``FRED_SERIES``.
    start, end
        Date range (ISO strings). ``end=None`` means "today".
    cache_dir
        Directory for the parquet cache. Defaults to the configured macro path.

    Returns
    -------
    Wide DataFrame indexed by date with one column per label.
    """
    cfg = load_config()
    if series_ids is None:
        series_ids = FRED_SERIES
    if cache_dir is None:
        cache_dir = cfg.paths.data_macro
    cache_path = resolve(cache_dir) / "fred.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        have = set(cached.columns)
        if have >= set(series_ids):
            _LOG.info("Using cached FRED data from %s", cache_path)
            return cached
        else:
            _LOG.info("Cache missing %s; re-downloading", set(series_ids) - have)

    fred = _get_fred_client()
    pieces: dict[str, pd.Series] = {}
    for label, sid in series_ids.items():
        if fred is not None:
            _LOG.info("Downloading FRED series %s (%s) via API key", sid, label)
            s = fred.get_series(sid, observation_start=start, observation_end=end)
        else:
            _LOG.info("Downloading FRED series %s (%s) via public CSV (no key)", sid, label)
            s = _download_fred_public_csv(sid, start=start, end=end)
        s.name = label
        pieces[label] = s

    df = pd.concat(pieces.values(), axis=1)
    df.columns = list(pieces.keys())
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(cache_path)
    _LOG.info("Saved FRED cache (%d rows, %d cols) to %s",
              len(df), df.shape[1], cache_path)
    return df


# ---------------------------------------------------------------------------
# Hang Seng squared returns
# ---------------------------------------------------------------------------

def fetch_hang_seng(start: str = "2015-01-01") -> pd.Series:
    """Return Hang Seng daily squared log return.

    Reads ``data/macro/hsi.parquet`` if it exists (produced by stage 2).
    Falls back to a live yfinance download, then to a NaN series if both fail,
    so the rest of the pipeline remains usable for offline runs.
    """
    cfg = load_config()
    cache_path = resolve(cfg.paths.data_macro) / "hsi.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            col = "HSI" if "HSI" in df.columns else df.columns[0]
            s = df[col].rename("HSI")
            _LOG.info("Using cached Hang Seng from %s (n=%d)", cache_path, len(s))
            return s
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("HSI cache read failed (%s); falling through to yfinance", exc)
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        _LOG.warning("yfinance not installed; Hang Seng feature will be NaN.")
        return pd.Series(dtype=float, name="HSI")
    try:
        hsi = yf.download("^HSI", start=start, progress=False, auto_adjust=True)
        if hsi.empty:
            raise RuntimeError("Empty Yahoo response for ^HSI")
        close = hsi["Close"].squeeze()
        log_ret = np.log(close).diff()
        squared = (log_ret ** 2).rename("HSI")
        return squared
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Hang Seng download failed (%s); feature will be NaN.", exc)
        return pd.Series(dtype=float, name="HSI")


# ---------------------------------------------------------------------------
# Per-stock features that can be computed locally
# ---------------------------------------------------------------------------

def momentum_1w(daily_return: pd.Series) -> pd.Series:
    """1-week (5-day) cumulative log-return at each date.

    The value at day :math:`t` uses returns from :math:`t-5` through :math:`t-1`,
    so it is a *lagged* predictor for forecasting RV at :math:`t+1`.
    """
    # The outer caller in build_features applies .shift(1) to lag this for
    # forecasting; the rolling sum here is the 5-day cumulative return
    # ending at day t.
    return daily_return.rolling(window=5).sum().rename("M1W")


def dollar_volume_change(close: pd.Series, volume: pd.Series) -> pd.Series:
    """First difference of log dollar volume (price * volume)."""
    dvol = (close * volume).replace(0, np.nan)
    log_dvol = np.log(dvol)
    return log_dvol.diff().rename("DVOL")


# ---------------------------------------------------------------------------
# Earnings-announcement dummy (manually maintained list per ticker)
# ---------------------------------------------------------------------------

EARNINGS_DATES_FALLBACK: dict[str, list[str]] = {
    # Empty fallback. Real per-stock earnings dates are loaded from
    # ``data/macro/earnings_dates.json`` if present (see stage 2).
    "AAPL": [],
    "AMZN": [],
    "JPM":  [],
}


def _load_earnings_cache() -> dict[str, list[str]]:
    """Load the earnings-date cache produced by stage 2.

    Returns an empty mapping if the cache file does not exist.
    """
    import json
    cfg = load_config()
    path = resolve(cfg.paths.data_macro) / "earnings_dates.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:  # noqa: BLE001
        return {}


def earnings_announcement_dummy(
    dates: pd.DatetimeIndex,
    ticker: str,
    overrides: dict[str, list[str]] | None = None,
) -> pd.Series:
    """Construct the EA dummy series across a date index.

    Priority order for the announcement dates:

    1. ``overrides`` argument (if provided).
    2. ``data/macro/earnings_dates.json`` cache (populated via yfinance in
       stage 2 — see :mod:`scripts.02_download_macro`).
    3. The empty fallback (all zeros).
    """
    if overrides is not None:
        table = overrides
    else:
        cached = _load_earnings_cache()
        table = cached if cached else EARNINGS_DATES_FALLBACK
    announce_set = set(pd.to_datetime(table.get(ticker, [])).normalize())
    flag = pd.Series(0, index=dates, name="EA", dtype=float)
    for d in announce_set:
        if d in flag.index:
            flag.loc[d] = 1.0
    return flag


# ---------------------------------------------------------------------------
# Implied volatility proxy
# ---------------------------------------------------------------------------

def implied_vol_proxy(
    macro: pd.DataFrame,
    dates: pd.DatetimeIndex,
    method: str = "VIX",
) -> pd.Series:
    """Per-stock implied-volatility proxy.

    The paper uses OptionMetrics model-free implied volatility per stock,
    which is licensed. As a publicly available proxy this returns the
    CBOE VIX time series aligned to the input dates. A per-stock proxy
    (e.g. AAPL's own implied vol via a free source) can be plugged in
    here by parameterising ``method``.
    """
    if method.upper() == "VIX" and "VIX" in macro.columns:
        return macro["VIX"].reindex(dates).rename("IV")
    return pd.Series(np.nan, index=dates, name="IV")
