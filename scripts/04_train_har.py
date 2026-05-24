"""
Stage 4 — train the HAR-family models.

The HAR family is fast (closed-form OLS) and acts as the smoke test for
the rest of the pipeline. Running this script alone is a good way to
verify the data pipeline before incurring the cost of NN training.

Usage:
    python scripts/04_train_har.py
    python scripts/04_train_har.py --stocks AAPL --feature-set M_HAR --horizons 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import get_logger, load_config, set_global_seed
from src.pipeline.orchestrator import run_one, save_results, StockRunResult
from src.evaluation.metrics import mse_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HAR family.")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-set", default="M_HAR")
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="predictions_har.pkl")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("train_har", level=cfg.project.log_level)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    results: list[StockRunResult] = []
    for ticker in stocks:
        for h in horizons:
            log.info("=== %s | %s | h=%d ===", ticker, args.feature_set, h)
            # Restrict to HAR-only run by skipping ML/NN models.
            res = run_one(
                ticker=ticker,
                feature_set=args.feature_set,
                horizon=h,
                cfg=cfg,
                skip_nn=True,
                skip_trees=True,
                skip_regularised=True,
            )
            # Keep only the HAR-family predictions.
            har_only = {k: v for k, v in res.predictions.items()
                        if k in {"HAR", "LogHAR", "LevHAR", "SHAR", "HARQ", "HAR-X"}}
            res.predictions = har_only
            results.append(res)

            for label, preds in har_only.items():
                idx = res.y_true.index.intersection(preds.index)
                mse = float(np.mean(mse_loss(res.y_true.loc[idx].to_numpy(),
                                             preds.loc[idx].to_numpy())))
                log.info("  %-6s  MSE = %.6e", label, mse)

    save_results(results, filename=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
