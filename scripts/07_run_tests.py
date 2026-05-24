"""
Stage 7 — Diebold-Mariano tests and Model Confidence Set.

Loads the predictions produced by stages 4-6, merges them across model
families, and produces:

* a per-stock DM table (each model vs HAR, one-sided "less"),
* a per-stock MCS p-value table at the configured confidence level, and
* a pooled cross-stock summary.

Outputs land in ``outputs/tables/``.

Usage:
    python scripts/07_run_tests.py
    python scripts/07_run_tests.py --loss qlike --horizon 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config
from src.pipeline.orchestrator import load_results, StockRunResult
from src.evaluation.metrics import LOSSES, mse_loss
from src.evaluation.diebold_mariano import diebold_mariano
from src.evaluation.mcs import model_confidence_set
from src.visualization.tables import (
    build_loss_table, loss_ratio_table,
    dm_pvalue_table, mcs_pvalue_table, save_table,
)


def _merge_predictions(files: list[str]) -> list[StockRunResult]:
    """Merge per-stage prediction pickles into one StockRunResult list."""
    merged: dict[tuple[str, str, int], StockRunResult] = {}
    for f in files:
        try:
            results = load_results(f)
        except FileNotFoundError:
            continue
        for r in results:
            key = (r.ticker, r.feature_set, r.horizon)
            if key in merged:
                merged[key].predictions.update(r.predictions)
            else:
                merged[key] = r
    return list(merged.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DM tests and MCS.")
    parser.add_argument("--loss", default="mse", choices=list(LOSSES))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--baseline", default="HAR")
    parser.add_argument("--inputs", nargs="*",
                        default=["predictions_har.pkl",
                                 "predictions_ml.pkl",
                                 "predictions_nn.pkl"])
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("run_tests", level=cfg.project.log_level)
    loss_fn = LOSSES[args.loss]

    runs = _merge_predictions(args.inputs)
    runs = [r for r in runs if r.horizon == args.horizon]
    if not runs:
        log.error("No runs at horizon %d found", args.horizon)
        return 1

    log.info("Loaded %d (stock, feature-set, horizon) runs", len(runs))

    # Build the y_true and predictions dicts keyed by ticker.
    y_true_dict = {r.ticker: r.y_true for r in runs}
    pred_dict   = {r.ticker: r.predictions for r in runs}

    # --- Loss tables ---
    loss_tab = build_loss_table(y_true_dict, pred_dict, loss_fn=loss_fn)
    save_table(loss_tab, f"loss_h{args.horizon}_{args.loss}", fmt="csv")
    ratio_tab = loss_ratio_table(loss_tab, baseline=args.baseline)
    save_table(ratio_tab, f"loss_ratio_h{args.horizon}_{args.loss}", fmt="csv")
    log.info("Loss-ratio table:\n%s", ratio_tab.round(3).to_string())

    # --- DM tests vs baseline ---
    dm_rows = []
    for r in runs:
        if args.baseline not in r.predictions:
            log.warning("No baseline '%s' for %s, skipping DM", args.baseline, r.ticker)
            continue
        tab = dm_pvalue_table(r.y_true, r.predictions,
                              baseline=args.baseline, horizon=args.horizon,
                              loss=args.loss)
        tab["ticker"] = r.ticker
        dm_rows.append(tab)
    if dm_rows:
        dm_all = pd.concat(dm_rows)
        save_table(dm_all, f"dm_h{args.horizon}_{args.loss}", fmt="csv")
        log.info("DM tests saved")

    # --- MCS at every configured alpha level ---
    alpha_levels = list(getattr(cfg.mcs, "alpha_levels", [cfg.mcs.alpha]))
    mcs_by_alpha: dict[float, dict[str, object]] = {a: {} for a in alpha_levels}
    for r in runs:
        try:
            common = None
            preds = {}
            for k, p in r.predictions.items():
                idx = r.y_true.index.intersection(p.index)
                preds[k] = p.loc[idx]
                common = idx if common is None else common.intersection(idx)
            y_aligned = r.y_true.loc[common].to_numpy()
            preds_aligned = {k: v.loc[common].to_numpy() for k, v in preds.items()}
            for a in alpha_levels:
                res = model_confidence_set(
                    y_aligned, preds_aligned,
                    loss=args.loss,
                    alpha=a,
                    num_bootstrap=cfg.mcs.num_bootstrap,
                    block_length=cfg.mcs.block_length,
                    statistic=cfg.mcs.statistic,
                    seed=cfg.project.seed,
                )
                mcs_by_alpha[a][r.ticker] = res
                log.info("[%s] MCS α=%.2f survivors: %s", r.ticker, a, res.surviving_models)
        except Exception as exc:  # noqa: BLE001
            log.error("MCS failed for %s: %s", r.ticker, exc, exc_info=True)

    for a, mcs_results in mcs_by_alpha.items():
        if mcs_results:
            tab = mcs_pvalue_table(mcs_results)
            save_table(tab, f"mcs_h{args.horizon}_{args.loss}_a{int(round(a*100)):02d}", fmt="csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
