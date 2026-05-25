"""
Stage 25 — ROLLING tree estimation (paper-faithful) vs the fixed-window trees.

CSV (2023) refit bagging/RF/GB daily on a sliding window. Our main pipeline
fixes the trees once on train+val, which makes them collapse at h=22 because a
tree cannot extrapolate beyond its training-leaf range onto the calmer
post-COVID test set. This script rolls the trees daily so we can (a) check
whether rolling reproduces CSV's "RF wins at the monthly horizon" finding and
(b) report a controlled fixed-vs-rolling comparison.

BG/RF are off-the-shelf (Breiman-Cutler defaults), so each daily refit just
rebuilds the forest. GB hyperparameters are selected ONCE on the initial
(train,val) split and then frozen for the rolling window (the paper's
"rolling without re-tuning" spirit) to keep the cost feasible.

Usage:
    python scripts/24_rolling_trees.py --stocks AAPL --horizons 22   # test cell
    python scripts/24_rolling_trees.py                                # full run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import get_logger, load_config, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.models.base import Forecaster
from src.models.tree_models import (
    BaggingForecaster, RandomForestForecaster, GradientBoostingForecaster,
)
from src.pipeline.orchestrator import _ml_columns, load_results
from src.pipeline.rolling_forecast import rolling_window_forecast
from src.evaluation.metrics import mse_loss

_LOG = get_logger("rolling_trees")


class _FrozenGB(Forecaster):
    """Gradient boosting with fixed (pre-selected) hyperparameters."""
    name = "GB"

    def __init__(self, params: dict, subsample: float, seed: int):
        self.params = params; self.subsample = subsample; self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        self.model_ = GradientBoostingRegressor(
            **self.params, subsample=self.subsample, random_state=self.seed
        ).fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


def _har_mse(ticker: str, horizon: int) -> float:
    """HAR test-set MSE from the already-computed (rolling) HAR pickle."""
    for r in load_results("predictions_har_MALL.pkl"):
        if r.ticker == ticker and r.horizon == horizon and r.feature_set == "M_ALL":
            y = r.y_true; p = r.predictions["HAR"]
            idx = y.index.intersection(p.index)
            return float(np.mean(mse_loss(y.loc[idx].to_numpy(), p.loc[idx].to_numpy())))
    return np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=None)
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--refit-frequency", type=int, default=1)
    ap.add_argument("--out", default="outputs/tables/rolling_vs_fixed_trees.csv",
                    help="Output CSV path (give each cell its own file when parallelising).")
    args = ap.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    stocks = args.stocks or cfg.data.stocks
    horizons = args.horizons or cfg.forecast.horizons

    rows = []
    for ticker in stocks:
        for h in horizons:
            feats = load_feature_matrix(ticker, "M_ALL", h)
            train, val, test = time_split(feats, cfg.data.split.train_frac, cfg.data.split.val_frac)
            cols = _ml_columns(feats, "M_ALL")          # ML features + 'y', no HAR helpers
            tr, vl, te = train[cols], val[cols], test[cols]
            floor = float(tr["y"].min())

            # Frozen GB params from one (train,val) tuning.
            gb_tune = GradientBoostingForecaster(
                n_estimators_grid=cfg.models_trees.gradient_boosting.n_estimators_grid,
                learning_rate_grid=cfg.models_trees.gradient_boosting.learning_rate_grid,
                max_depth_grid=cfg.models_trees.gradient_boosting.max_depth_grid,
                subsample=cfg.models_trees.gradient_boosting.subsample,
                random_state=cfg.project.seed,
            )
            gb_tune.fit(tr.drop(columns=["y"]), tr["y"], X_val=vl.drop(columns=["y"]), y_val=vl["y"])
            gb_params = gb_tune.diagnostics.params

            factories = {
                "BG": lambda: BaggingForecaster(
                    n_estimators=cfg.models_trees.bagging.n_estimators,
                    bootstrap=cfg.models_trees.bagging.bootstrap, random_state=cfg.project.seed),
                "RF": lambda: RandomForestForecaster(
                    n_estimators=cfg.models_trees.random_forest.n_estimators,
                    max_features=cfg.models_trees.random_forest.max_features,
                    bootstrap=cfg.models_trees.random_forest.bootstrap, random_state=cfg.project.seed),
                "GB": lambda: _FrozenGB(gb_params, cfg.models_trees.gradient_boosting.subsample, cfg.project.seed),
            }

            har = _har_mse(ticker, h)
            for label, fac in factories.items():
                t0 = time.time()
                out = rolling_window_forecast(fac, tr, vl, te, refit_frequency=args.refit_frequency, progress=False)
                pred = out.predictions.clip(lower=floor)
                idx = out.y_true.index.intersection(pred.index)
                mse = float(np.mean(mse_loss(out.y_true.loc[idx].to_numpy(), pred.loc[idx].to_numpy())))
                ratio = mse / har if har == har else np.nan
                rows.append({"ticker": ticker, "h": h, "model": label,
                             "rolling_mse": mse, "rolling_ratio_vs_HAR": ratio})
                _LOG.info("[%s h=%d] %s rolling ratio vs HAR = %.3f (%.0fs)",
                          ticker, h, label, ratio, time.time() - t0)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with the fixed-window ratios for a side-by-side table.
    df.to_csv(out_path, index=False)
    _LOG.info("Saved %s", out_path)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
