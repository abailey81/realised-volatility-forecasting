"""
Stage 6 — train the geometric-pyramid neural networks.

This is the compute-heavy stage. With the default configuration of 100
random seeds per architecture and four architectures (NN1-NN4), training
on AAPL/M_ALL/h=1 takes:

* CPU (single core): ~5-10 min per architecture, ~30-40 min total.
* CPU (n_jobs=4): ~10-15 min total.
* GPU (single CUDA device): ~5-8 min total.

For development iteration, set ``num_random_seeds`` low (e.g. 5) in
``config.yaml`` first, then bump back to 100 for the headline results.

Usage:
    python scripts/06_train_nn.py
    python scripts/06_train_nn.py --stocks AAPL --horizons 1 --architectures NN2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import get_logger, load_config, set_global_seed
from src.pipeline.orchestrator import run_one, save_results, StockRunResult
from src.evaluation.metrics import mse_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train geometric-pyramid NNs.")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--architectures", nargs="*", default=None,
                        help="Subset of NN1, NN2, NN3, NN4 (default: all configured).")
    parser.add_argument("--output", default="predictions_nn.pkl")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("train_nn", level=cfg.project.log_level)

    # Optional architecture subset.
    if args.architectures:
        keep = set(args.architectures)
        new_arch = {k: v for k, v in cfg.models_nn.architectures.items() if k in keep}
        if not new_arch:
            log.error("No matching architectures: %s", args.architectures)
            return 1
        # Mutate the underlying dict in-place (the orchestrator reads cfg).
        cfg.models_nn.architectures.clear()
        cfg.models_nn.architectures.update(new_arch)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    results: list[StockRunResult] = []
    for ticker in stocks:
        for h in horizons:
            log.info("=== %s | %s | h=%d ===", ticker, args.feature_set, h)
            t0 = time.time()
            res = run_one(
                ticker=ticker,
                feature_set=args.feature_set,
                horizon=h,
                cfg=cfg,
                skip_nn=False,
                skip_trees=True,    # already trained in stage 5
                skip_har=True,
                skip_regularised=True,
            )
            elapsed = time.time() - t0
            log.info("  elapsed %.1f min", elapsed / 60.0)

            # Keep only NN predictions for this output file; the rest will be
            # consolidated downstream in stage 9.
            nn_only = {k: v for k, v in res.predictions.items()
                       if k.endswith("_ensemble") or k.endswith("_top1")}
            res.predictions = nn_only
            if res.val_predictions:
                res.val_predictions = {k: v for k, v in res.val_predictions.items()
                                        if k.endswith("_ensemble") or k.endswith("_top1")}
            results.append(res)

            for label, preds in nn_only.items():
                idx = res.y_true.index.intersection(preds.index)
                mse = float(np.mean(mse_loss(res.y_true.loc[idx].to_numpy(),
                                             preds.loc[idx].to_numpy())))
                log.info("  %-15s  MSE = %.6e", label, mse)

    save_results(results, filename=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
