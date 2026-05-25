"""
Stage 24b — PARALLEL rolling-tree estimation (full grid), leaving 24_rolling_trees.py untouched.

Full grid: 3 stocks x horizons {1,5,22} x {BG, RF, GB}, daily refit (refit_frequency=1),
paper-faithful. The 27 (stock, horizon, model) cells are independent and individually seeded,
so concurrent execution gives identical results to the sequential run.

Parallelism is at the CELL level via a process pool; inside each worker the forests use
n_jobs=1 and BLAS threads are pinned to 1, so there is no nested oversubscription (the outer
pool owns the cores). This is the fix for the 9-shard run, where the repo's n_jobs=-1 forests
each grabbed all 16 cores -> ~144 threads thrashing on 16.

Output: outputs/tables/trees_fixed_vs_rolling_all.csv
        columns: stock, horizon, model, fixed_ratio, rolling_ratio
The original outputs/tables/rolling_vs_fixed_trees.csv is NOT touched.

Usage:
    python scripts/24b_rolling_trees_parallel.py
"""
from __future__ import annotations

# Pin BLAS/OpenMP threads BEFORE importing numpy/sklearn so the process pool scales cleanly.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor,
)

from src.utils import get_logger, load_config, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.models.base import Forecaster
from src.models.tree_models import GradientBoostingForecaster
from src.pipeline.orchestrator import _ml_columns, load_results
from src.pipeline.rolling_forecast import rolling_window_forecast
from src.evaluation.metrics import mse_loss

_LOG = get_logger("rolling_par")

STOCKS = ("AAPL", "AMZN", "JPM")
HORIZONS = (1, 5, 22)
MODELS = ("BG", "RF", "GB")


# --- single-threaded forest variants: identical results to the repo's n_jobs=-1 forests ---
class _BG1(Forecaster):
    name = "BG"

    def __init__(self, n_estimators, bootstrap, random_state):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        self.model_ = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.random_state),
            n_estimators=self.n_estimators, bootstrap=self.bootstrap,
            n_jobs=1, random_state=self.random_state,
        ).fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


class _RF1(Forecaster):
    name = "RF"

    def __init__(self, n_estimators, max_features, bootstrap, random_state):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features,
            bootstrap=self.bootstrap, n_jobs=1, random_state=self.random_state,
        ).fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


class _FrozenGB(Forecaster):
    name = "GB"

    def __init__(self, params, subsample, seed):
        self.params = params
        self.subsample = subsample
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        self.model_ = GradientBoostingRegressor(
            **self.params, subsample=self.subsample, random_state=self.seed,
        ).fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


def _har_mse(ticker: str, horizon: int) -> float:
    for r in load_results("predictions_har_MALL.pkl"):
        if r.ticker == ticker and r.horizon == horizon and r.feature_set == "M_ALL":
            y = r.y_true
            p = r.predictions["HAR"]
            idx = y.index.intersection(p.index)
            return float(np.mean(mse_loss(y.loc[idx].to_numpy(), p.loc[idx].to_numpy())))
    return np.nan


def _tune_gb(ticker: str, h: int) -> dict:
    """Tune GB once on (train, val) for this cell — same as the original loop does."""
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    feats = load_feature_matrix(ticker, "M_ALL", h)
    train, val, _ = time_split(feats, cfg.data.split.train_frac, cfg.data.split.val_frac)
    cols = _ml_columns(feats, "M_ALL")
    tr, vl = train[cols], val[cols]
    gb = GradientBoostingForecaster(
        n_estimators_grid=cfg.models_trees.gradient_boosting.n_estimators_grid,
        learning_rate_grid=cfg.models_trees.gradient_boosting.learning_rate_grid,
        max_depth_grid=cfg.models_trees.gradient_boosting.max_depth_grid,
        subsample=cfg.models_trees.gradient_boosting.subsample,
        random_state=cfg.project.seed,
    )
    gb.fit(tr.drop(columns=["y"]), tr["y"], X_val=vl.drop(columns=["y"]), y_val=vl["y"])
    return gb.diagnostics.params


def _run_cell(task: tuple) -> dict:
    """One (stock, horizon, model) rolling cell — mirrors the original cell logic exactly."""
    ticker, h, label, gb_params = task
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    feats = load_feature_matrix(ticker, "M_ALL", h)
    train, val, test = time_split(feats, cfg.data.split.train_frac, cfg.data.split.val_frac)
    cols = _ml_columns(feats, "M_ALL")
    tr, vl, te = train[cols], val[cols], test[cols]
    floor = float(tr["y"].min())

    if label == "BG":
        fac = lambda: _BG1(cfg.models_trees.bagging.n_estimators,
                           cfg.models_trees.bagging.bootstrap, cfg.project.seed)
    elif label == "RF":
        fac = lambda: _RF1(cfg.models_trees.random_forest.n_estimators,
                           cfg.models_trees.random_forest.max_features,
                           cfg.models_trees.random_forest.bootstrap, cfg.project.seed)
    else:
        fac = lambda: _FrozenGB(gb_params, cfg.models_trees.gradient_boosting.subsample,
                                cfg.project.seed)

    t0 = time.time()
    out = rolling_window_forecast(fac, tr, vl, te, refit_frequency=1, progress=False)
    pred = out.predictions.clip(lower=floor)
    idx = out.y_true.index.intersection(pred.index)
    mse = float(np.mean(mse_loss(out.y_true.loc[idx].to_numpy(), pred.loc[idx].to_numpy())))
    har = _har_mse(ticker, h)
    ratio = mse / har if har == har else np.nan
    return {"stock": ticker, "horizon": h, "model": label,
            "rolling_mse": mse, "rolling_ratio": round(ratio, 3),
            "secs": round(time.time() - t0, 1)}


def main() -> int:
    t_start = time.time()

    # 1. GB pre-tuned once per (stock, horizon), sequentially (cheap).
    gb_params = {}
    for s in STOCKS:
        for h in HORIZONS:
            gb_params[(s, h)] = _tune_gb(s, h)
    _LOG.info("GB tuned for %d (stock,horizon) cells in %.0fs", len(gb_params), time.time() - t_start)

    # 2. 27-cell task list; controlled process pool, inner n_jobs=1.
    tasks = [(s, h, m, gb_params[(s, h)]) for s in STOCKS for h in HORIZONS for m in MODELS]
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(tasks))
    _LOG.info("running %d cells on %d workers (inner n_jobs=1)", len(tasks), n_jobs)

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_run_cell, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # report, don't paper over
                _LOG.error("cell %s FAILED: %s", t[:3], exc)
                continue
            results.append(r)
            _LOG.info("[%s h=%d] %s rolling ratio=%.3f (%.0fs)  [%d/%d]",
                      r["stock"], r["horizon"], r["model"], r["rolling_ratio"], r["secs"],
                      len(results), len(tasks))

    roll = pd.DataFrame(results)

    # 3. Merge with the fixed-window ratios already in loss_ratio_h{h}_mse.csv.
    fixed_rows = []
    for h in HORIZONS:
        lr = pd.read_csv(f"outputs/tables/loss_ratio_h{h}_mse.csv", index_col=0)
        for s in STOCKS:
            for m in MODELS:
                fixed_rows.append({"stock": s, "horizon": h, "model": m,
                                   "fixed_ratio": round(float(lr.loc[s, m]), 3)})
    fixed = pd.DataFrame(fixed_rows)
    out = (fixed.merge(roll[["stock", "horizon", "model", "rolling_ratio"]],
                       on=["stock", "horizon", "model"], how="left")
                .sort_values(["stock", "horizon", "model"]).reset_index(drop=True))

    out_path = Path("outputs/tables/trees_fixed_vs_rolling_all.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\n" + out.to_string(index=False))
    _LOG.info("Saved %s", out_path)

    # 4. Sanity check against the known AAPL h=22 cells.
    def _get(s, h, m, col):
        v = out[(out.stock == s) & (out.horizon == h) & (out.model == m)][col]
        return float(v.iloc[0]) if len(v) else float("nan")

    checks = [("AAPL", 22, "RF", "fixed_ratio", 3.038),
              ("AAPL", 22, "RF", "rolling_ratio", 1.548),
              ("AAPL", 22, "BG", "fixed_ratio", 7.314),
              ("AAPL", 22, "BG", "rolling_ratio", 2.560)]
    ok = True
    print("\n--- sanity check vs known cells ---")
    for s, h, m, col, exp in checks:
        got = _get(s, h, m, col)
        good = abs(got - exp) <= 0.03 * max(1.0, abs(exp))
        ok = ok and good
        print(f"{s} h{h} {m} {col}: got {got:.3f}  expect ~{exp}  {'OK' if good else 'MISMATCH'}")

    print(f"\nTOTAL wall-clock: {time.time() - t_start:.0f}s   sanity: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
