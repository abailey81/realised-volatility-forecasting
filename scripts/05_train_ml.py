"""
Stage 5 — train the regularised regression and tree-based models.

Trains the full set Ridge, Lasso, Elastic Net, Post-Lasso, Adaptive Lasso,
Bagging, Random Forest, Gradient Boosting across each configured
(stock, horizon) pair on the M_ALL feature set. The HAR family runs
alongside so the predictions file is self-contained for downstream tests.

Usage:
    python scripts/05_train_ml.py
    python scripts/05_train_ml.py --stocks AAPL --horizons 1
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
    parser = argparse.ArgumentParser(description="Train regularised + tree models.")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="predictions_ml.pkl")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("train_ml", level=cfg.project.log_level)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    results: list[StockRunResult] = []
    for ticker in stocks:
        for h in horizons:
            log.info("=== %s | %s | h=%d ===", ticker, args.feature_set, h)
            res = run_one(
                ticker=ticker,
                feature_set=args.feature_set,
                horizon=h,
                cfg=cfg,
                skip_nn=True,
                skip_trees=False,
                skip_har=True,
            )
            results.append(res)

            # Concise summary.
            for label, preds in res.predictions.items():
                idx = res.y_true.index.intersection(preds.index)
                mse = float(np.mean(mse_loss(res.y_true.loc[idx].to_numpy(),
                                             preds.loc[idx].to_numpy())))
                log.info("  %-6s  MSE = %.6e", label, mse)

    save_results(results, filename=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
