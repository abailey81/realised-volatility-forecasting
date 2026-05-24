"""
Stage 20 — disentangle the realised-kernel frequency vs estimator effects.

Closes a gap in the previous RK robustness check
(`scripts/12_rk_robustness.py`). That script compared RV at 5-min sampling
to RK at 1-min sampling — conflating two changes:

1. The sampling frequency change (5-min → 1-min)
2. The estimator change (sum-of-squares → Parzen-kernel-weighted sum)

For a clean attribution of the JPM h=1 −25% MSE finding, we need a 2×2 grid:

|              | RV (sum-of-squares) | RK (Parzen kernel) |
|--------------|---------------------|--------------------|
| 5-min        | baseline (paper)    | NEW                |
| 1-min        | NEW                 | existing (BNHLS)   |

This script computes the four daily-volatility series for each stock,
then runs the HAR-family forecast comparison under each estimator.
Comparing the four MSE columns lets us decompose:

* (RV-1m vs RV-5m): pure frequency effect on the sum-of-squares estimator.
* (RK-5m vs RV-5m): pure estimator effect at constant frequency.
* (RK-1m vs RV-5m): the previous combined comparison.
* (RK-1m vs RK-5m): how the kernel benefits from higher frequency input.

Usage:
    python scripts/19_rk_frequency_disentangle.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.tick_to_minute import load_and_clean
from src.data.compute_rv import resample_to_frequency, _daily_realised
from src.data.realised_kernel import _daily_realised_kernel
from src.data.feature_engineering import build_har_lags, make_horizon_target, time_split
from src.models.har_models import HAR, LogHAR, _fit_ols
from src.evaluation.metrics import mse_loss


def _daily_rv_at_frequency(ticker: str, sampling_minutes: int) -> pd.Series:
    cfg = load_config()
    minute_df = load_and_clean(ticker, config=cfg)
    intra = resample_to_frequency(minute_df, minutes=sampling_minutes)
    daily = _daily_realised(intra)
    return daily["RV"].rename(f"RV_{sampling_minutes}m")


def _daily_rk_at_frequency(ticker: str, sampling_minutes: int) -> pd.Series:
    cfg = load_config()
    minute_df = load_and_clean(ticker, config=cfg)
    out: list[tuple[pd.Timestamp, float]] = []
    for day, group in minute_df.groupby(minute_df.index.normalize()):
        prices = group["Close"].resample(f"{sampling_minutes}min",
                                          label="right", closed="right").last().dropna()
        if len(prices) < 5:
            continue
        ret = np.log(prices).diff().dropna().to_numpy()
        rk, _ = _daily_realised_kernel(ret)
        out.append((day, max(rk, 0.0)))
    s = pd.Series({d: v for d, v in out}, name=f"RK_{sampling_minutes}m").sort_index()
    s.index.name = "date"
    return s


def _har_forecast_mse(series: pd.Series, horizon: int) -> tuple[float, float, int]:
    """Train HAR + LogHAR on a single volatility series and return test MSEs."""
    rvd = series.shift(1).rename("RVD")
    rvw = series.shift(1).rolling(5).mean().rename("RVW")
    rvm = series.shift(1).rolling(22).mean().rename("RVM")
    y = make_horizon_target(series, horizon)
    df = pd.concat([rvd, rvw, rvm, y], axis=1).dropna()
    if len(df) < 200:
        return float("nan"), float("nan"), 0
    train, val, test = time_split(df, train_frac=0.70, val_frac=0.10)
    full_train = pd.concat([train, val])
    m_har = HAR().fit(full_train, full_train["y"])
    m_loghar = LogHAR().fit(full_train, full_train["y"])
    pred_har = m_har.predict(test)
    pred_loghar = m_loghar.predict(test)
    y_test = test["y"].to_numpy()
    mse_har = float(np.mean(mse_loss(y_test, pred_har)))
    mse_loghar = float(np.mean(mse_loss(y_test, pred_loghar)))
    return mse_har, mse_loghar, len(test)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    parser.add_argument("--stocks", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("rk_freq", level=cfg.project.log_level)
    stocks = args.stocks if args.stocks else cfg.data.stocks

    rows = []
    for ticker in stocks:
        log.info("=== %s: building 4 volatility series ===", ticker)
        t0 = time.time()
        rv_5m = _daily_rv_at_frequency(ticker, 5)
        log.info("  RV_5m done (%d obs, %.1fs)", len(rv_5m), time.time() - t0)
        t0 = time.time()
        rv_1m = _daily_rv_at_frequency(ticker, 1)
        log.info("  RV_1m done (%d obs, %.1fs)", len(rv_1m), time.time() - t0)
        t0 = time.time()
        rk_5m = _daily_rk_at_frequency(ticker, 5)
        log.info("  RK_5m done (%d obs, %.1fs)", len(rk_5m), time.time() - t0)
        t0 = time.time()
        rk_1m = _daily_rk_at_frequency(ticker, 1)
        log.info("  RK_1m done (%d obs, %.1fs)", len(rk_1m), time.time() - t0)

        # Persist for downstream
        out_dir = resolve(cfg.paths.data_intermediate)
        rv_1m.rename("RV").to_frame().to_parquet(out_dir / f"{ticker}_rv1m.parquet")
        rk_5m.rename("RK").to_frame().to_parquet(out_dir / f"{ticker}_rk5m.parquet")

        for h in args.horizons:
            for label, series in [("RV_5m", rv_5m), ("RV_1m", rv_1m),
                                    ("RK_5m", rk_5m), ("RK_1m", rk_1m)]:
                mse_har, mse_loghar, n_test = _har_forecast_mse(series, h)
                rows.append({
                    "ticker": ticker, "horizon": h, "estimator": label,
                    "MSE_HAR": mse_har, "MSE_LogHAR": mse_loghar, "n_test": n_test,
                })
                log.info("  [%s|h=%d|%s] HAR=%.3e LogHAR=%.3e n_test=%d",
                          ticker, h, label, mse_har, mse_loghar, n_test)

    df = pd.DataFrame(rows)
    out_dir = resolve(cfg.paths.outputs_tables)
    df.to_csv(out_dir / "rk_frequency_disentangle_raw.csv", index=False)

    # Pivot into per-stock × per-horizon decomposition
    summary_rows = []
    for (t, h), group in df.groupby(["ticker", "horizon"]):
        rv5 = group[group["estimator"] == "RV_5m"]["MSE_HAR"].iloc[0]
        rv1 = group[group["estimator"] == "RV_1m"]["MSE_HAR"].iloc[0]
        rk5 = group[group["estimator"] == "RK_5m"]["MSE_HAR"].iloc[0]
        rk1 = group[group["estimator"] == "RK_1m"]["MSE_HAR"].iloc[0]
        summary_rows.append({
            "ticker": t, "horizon": h,
            "MSE_RV_5m": rv5, "MSE_RV_1m": rv1,
            "MSE_RK_5m": rk5, "MSE_RK_1m": rk1,
            "freq_effect_RV":   rv1 / rv5,       # 1m vs 5m on RV alone
            "estimator_effect": rk5 / rv5,       # kernel vs sum-of-squares at 5m
            "kernel_helps_at_1m": rk1 / rv1,     # kernel vs sum-of-squares at 1m
            "combined_RK1m_vs_RV5m": rk1 / rv5,  # original conflated comparison
        })
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "rk_frequency_disentangle.csv", index=False)
    log.info("Saved decomposition table (%d rows)", len(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
