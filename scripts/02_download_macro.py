"""
Stage 2 — download macro features from FRED and Yahoo Finance.

Requires a FRED API key in the ``FRED_API_KEY`` environment variable.
The Hang Seng squared-return series is fetched from Yahoo Finance via
``yfinance``; if that fails (e.g. no internet), the pipeline still
runs but the HSI column will be NaN.

Usage:
    export FRED_API_KEY=YOUR_KEY
    python scripts/02_download_macro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config, resolve
from src.data.macro_features import download_fred_series, fetch_hang_seng


def _fetch_earnings_dates(tickers: list[str], cache_dir) -> None:
    """Fetch and cache per-ticker earnings announcement dates via yfinance."""
    import json
    log = get_logger("download_macro")
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance unavailable — earnings dates will be empty")
        return
    out: dict[str, list[str]] = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=80)
            if df is None or df.empty:
                out[t] = []
                continue
            df = df[(df.index >= "2015-01-01") & (df.index <= "2026-12-31")]
            dates = sorted({(d.tz_localize(None).normalize() if d.tz is not None else d.normalize())
                             for d in df.index})
            out[t] = [d.strftime("%Y-%m-%d") for d in dates]
            log.info("%s: %d earnings dates fetched", t, len(out[t]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Earnings fetch for %s failed (%s); using empty list", t, exc)
            out[t] = []
    path = resolve(cache_dir) / "earnings_dates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log.info("Saved earnings cache to %s", path)


def main() -> int:
    cfg = load_config()
    log = get_logger("download_macro", level=cfg.project.log_level)

    log.info("Downloading FRED series (VIX, EPU, ADS, US3M)")
    fred = download_fred_series(start="2015-01-01")
    log.info("FRED data shape: %s", fred.shape)

    log.info("Fetching Hang Seng (^HSI) squared returns from Yahoo Finance")
    hsi = fetch_hang_seng(start="2015-01-01")
    log.info("Hang Seng length: %d (NaN-rate: %.1f%%)",
             len(hsi), 100 * hsi.isna().mean() if len(hsi) else 100.0)

    # Persist HSI to the macro cache as well.
    if len(hsi):
        path = resolve(cfg.paths.data_macro) / "hsi.parquet"
        hsi.to_frame().to_parquet(path)
        log.info("Saved HSI cache to %s", path)

    # Per-ticker earnings announcement dates (for the EA dummy).
    log.info("Fetching earnings announcement dates for %s", cfg.data.stocks)
    _fetch_earnings_dates(list(cfg.data.stocks), cfg.paths.data_macro)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
