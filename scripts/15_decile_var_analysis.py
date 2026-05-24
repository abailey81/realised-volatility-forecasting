"""
Stage 15 — Decile + VaR analysis (paper Section 3.1 Figure 5 + Section 5).

Two paper-replicated analyses in one script:

* **Decile analysis (Figure 5)**: partition the test set into deciles of
  observed realised variance; report each model's MSE-vs-HAR ratio in
  every decile. Christensen et al. (2023) find the ML gains over HAR are
  concentrated in the highest-RV deciles — a finding that is economically
  important.

* **VaR analysis (Section 5)**: build one-day-ahead α-quantile forecasts
  using filtered historical simulation (Barone-Adesi 1998), evaluate with
  the asymmetric quantile loss (Koenker-Bassett 1978) and the Kupiec /
  Christoffersen coverage tests. Both α=0.05 and α=0.01.

Both analyses run independently per (stock, horizon) and feed off the
saved prediction pickles. The decile analysis is single-threaded; the
VaR analysis is process-parallel across (stock × alpha).

Usage:
    python scripts/15_decile_var_analysis.py
    python scripts/15_decile_var_analysis.py --horizons 1 --alphas 0.05
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.evaluation.metrics import LOSSES, mse_loss
from src.evaluation.decile import decile_losses, relative_decile_losses
from src.evaluation.value_at_risk import var_table
from src.pipeline.orchestrator import load_results
from src.data.compute_rv import load_realised
from src.data.feature_engineering import time_split, load_feature_matrix


def _run_decile_one(horizon: int, baseline: str, files: list[str]) -> dict:
    cfg = load_config()
    log = get_logger("decile", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)

    # Merge prediction pickles for this horizon
    merged: dict[tuple, "object"] = {}
    for f in files:
        try:
            for r in load_results(f):
                if r.horizon != horizon:
                    continue
                key = (r.ticker, r.feature_set, r.horizon)
                if key in merged:
                    merged[key].predictions.update(r.predictions)
                else:
                    merged[key] = r
        except FileNotFoundError:
            continue

    rows_long = []
    for (ticker, feature_set, _), r in merged.items():
        try:
            res = decile_losses(r.y_true, r.predictions, mse_loss)
            ratio = relative_decile_losses(res, baseline=baseline)
            for d_idx, d_label in enumerate(ratio.index):
                for model in ratio.columns:
                    rows_long.append({
                        "ticker": ticker, "feature_set": feature_set,
                        "horizon": horizon, "decile": d_label,
                        "model": model, "ratio_vs_HAR": float(ratio.loc[d_label, model]),
                    })
        except Exception as exc:  # noqa: BLE001
            log.warning("Decile failed for %s|h=%d: %s", ticker, horizon, exc)

    df = pd.DataFrame(rows_long)
    if not df.empty:
        df.to_csv(out_dir / f"decile_ratios_h{horizon}.csv", index=False)
        # Pivot per stock for easy plot ingestion
        for ticker in df["ticker"].unique():
            sub = df[df["ticker"] == ticker]
            pivot = sub.pivot(index="decile", columns="model", values="ratio_vs_HAR")
            pivot.to_csv(out_dir / f"decile_ratios_{ticker}_h{horizon}.csv")
    return {"horizon": horizon, "n_rows": len(df)}


def _run_var_one(ticker: str, horizon: int, alphas: tuple, files: list[str]) -> dict:
    cfg = load_config()
    log = get_logger("var", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)

    # Use only the M_ALL run for this ticker/horizon.
    merged_preds: dict[str, pd.Series] = {}
    y_true_idx = None
    for f in files:
        try:
            for r in load_results(f):
                if r.ticker != ticker or r.horizon != horizon or r.feature_set != "M_ALL":
                    continue
                merged_preds.update(r.predictions)
                y_true_idx = r.y_true.index
        except FileNotFoundError:
            continue
    if not merged_preds or y_true_idx is None:
        return {"ticker": ticker, "horizon": horizon, "status": "no_preds"}

    # Daily log-returns + realised RV (for in-sample standardisation in FHS).
    rv_df = load_realised(ticker)
    log_returns = rv_df["ret"]
    rv_realised = rv_df["RV"]
    train_end = (y_true_idx[0] - pd.Timedelta(days=1))

    tab = var_table(log_returns, merged_preds, train_end=train_end,
                    alphas=alphas, rv_realised=rv_realised)
    tab["ticker"] = ticker
    tab["horizon"] = horizon
    out_path = out_dir / f"var_{ticker}_h{horizon}.csv"
    tab.to_csv(out_path, index=False)
    return {"ticker": ticker, "horizon": horizon, "status": "ok", "rows": len(tab)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Decile + VaR analysis.")
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.01])
    parser.add_argument("--baseline", default="HAR")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--inputs", nargs="*",
                        default=["predictions_har_MALL.pkl",
                                 "predictions_ml_MALL.pkl",
                                 "predictions_nn_MALL.pkl"])
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("decile_var", level=cfg.project.log_level)
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    # 1. Decile analyses (one per horizon, parallel)
    log.info("=== Decile analysis ===")
    decile_tasks = [(h, args.baseline, args.inputs) for h in horizons]
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(decile_tasks))) as ex:
        futs = {ex.submit(_run_decile_one, *t): t for t in decile_tasks}
        for fut in as_completed(futs):
            log.info("decile done: %s", fut.result())

    # 2. VaR analyses (one per stock × horizon, parallel)
    log.info("=== VaR analysis ===")
    var_tasks = [(s, h, tuple(args.alphas), args.inputs)
                 for s in cfg.data.stocks for h in horizons]
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_run_var_one, *t): t for t in var_tasks}
        for fut in as_completed(futs):
            log.info("var done: %s", fut.result())

    log.info("Decile + VaR analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
