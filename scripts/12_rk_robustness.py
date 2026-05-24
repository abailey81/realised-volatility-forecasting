"""
Stage 12 — realised-kernel robustness analysis.

Re-runs the HAR family using the realised kernel (RK) as the target/RHS
variable rather than 5-minute realised variance (RV). The realised kernel
is noise-robust (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008), so
divergence between RV-based and RK-based MSE rankings would indicate that
microstructure noise is materially affecting the results.

Produces a side-by-side MSE table for the HAR family.

Usage:
    python scripts/12_rk_robustness.py
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
from src.data.compute_rv import load_realised
from src.data.feature_engineering import build_har_lags, make_horizon_target, time_split
from src.models.har_models import HAR, LogHAR, LevHAR, SHAR, HARQ
from src.models.har_models import make_har
from src.evaluation.metrics import mse_loss


def _load_rk(ticker: str, cfg) -> pd.Series:
    path = resolve(cfg.paths.data_intermediate) / f"{ticker}_rk.parquet"
    if not path.exists():
        raise FileNotFoundError(f"RK file missing for {ticker}: {path}")
    rk = pd.read_parquet(path)
    return rk["RK"]


def _build_har_only(rv_series: pd.Series, horizon: int) -> pd.DataFrame:
    """Build a minimal HAR feature matrix using one volatility series."""
    rvd = rv_series.shift(1).rename("RVD")
    rvw = rv_series.shift(1).rolling(5).mean().rename("RVW")
    rvm = rv_series.shift(1).rolling(22).mean().rename("RVM")
    y = make_horizon_target(rv_series, horizon)
    df = pd.concat([rvd, rvw, rvm, y], axis=1).dropna()
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("rk_robustness", level=cfg.project.log_level)
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    rows = []
    for ticker in cfg.data.stocks:
        rv = load_realised(ticker)["RV"]
        rk = _load_rk(ticker, cfg)
        for h in horizons:
            df_rv = _build_har_only(rv, h)
            df_rk = _build_har_only(rk, h)
            tr_rv, vl_rv, te_rv = time_split(df_rv,
                                              train_frac=cfg.data.split.train_frac,
                                              val_frac=cfg.data.split.val_frac)
            tr_rk, vl_rk, te_rk = time_split(df_rk,
                                              train_frac=cfg.data.split.train_frac,
                                              val_frac=cfg.data.split.val_frac)
            full_tr_rv = pd.concat([tr_rv, vl_rv])
            full_tr_rk = pd.concat([tr_rk, vl_rk])

            for label in ("HAR", "LogHAR"):
                # HAR family on RV
                m_rv = make_har(label).fit(full_tr_rv, full_tr_rv["y"])
                pred_rv = m_rv.predict(te_rv)
                mse_rv = float(np.mean((te_rv["y"].values - pred_rv) ** 2))
                # HAR family on RK
                m_rk = make_har(label).fit(full_tr_rk, full_tr_rk["y"])
                pred_rk = m_rk.predict(te_rk)
                mse_rk = float(np.mean((te_rk["y"].values - pred_rk) ** 2))
                rows.append({
                    "ticker": ticker, "horizon": h, "model": label,
                    "MSE_RV": mse_rv, "MSE_RK": mse_rk,
                    "ratio_RK_over_RV": mse_rk / mse_rv if mse_rv > 0 else float("nan"),
                })
                log.info("  [%s|h=%d|%s] MSE_RV=%.3e, MSE_RK=%.3e, ratio=%.3f",
                          ticker, h, label, mse_rv, mse_rk, mse_rk / mse_rv)

    df = pd.DataFrame(rows)
    out = resolve(cfg.paths.outputs_tables) / "rk_robustness.csv"
    df.to_csv(out, index=False)
    log.info("Saved RK robustness table to %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
