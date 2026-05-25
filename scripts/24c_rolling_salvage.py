"""
Stage 24c — SALVAGE the rolling-tree grid after the 24b run crashed at 16/27
(a worker was OOM-killed under memory pressure from a concurrently-running
duplicate process, which broke the pool and killed the main process).

The 16 already-completed cells are taken from the 24b log (the run is fully
deterministic — individually seeded per cell — and the AAPL h=22 RF cell matched
the known 1.548 exactly, so the logged values are trustworthy). Only the 11
missing cells (JPM h5/h22 RF + all 9 BG) are recomputed here, then everything is
merged with the fixed-window ratios into trees_fixed_vs_rolling_all.csv.

Self-contained (cell logic inlined, not imported) so ProcessPoolExecutor 'spawn'
workers pickle cleanly. Single pool, inner n_jobs=1, no oversubscription.

Usage:  python scripts/24c_rolling_salvage.py
"""
from __future__ import annotations

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
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

from src.utils import get_logger, load_config, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.models.base import Forecaster
from src.pipeline.orchestrator import _ml_columns, load_results
from src.pipeline.rolling_forecast import rolling_window_forecast
from src.evaluation.metrics import mse_loss

_LOG = get_logger("rolling_salvage")
STOCKS = ("AAPL", "AMZN", "JPM")
HORIZONS = (1, 5, 22)
MODELS = ("BG", "RF", "GB")

# 16 cells already computed by the 24b run (rolling_ratio vs HAR), from its log.
DONE = {
    ("AMZN", 5, "GB"): 1.098, ("AAPL", 1, "GB"): 1.034, ("AAPL", 22, "GB"): 1.568,
    ("AAPL", 5, "GB"): 0.964, ("AMZN", 1, "GB"): 0.957, ("AMZN", 22, "GB"): 2.636,
    ("JPM", 1, "GB"): 0.946, ("JPM", 5, "GB"): 0.580, ("JPM", 22, "GB"): 1.289,
    ("AMZN", 5, "RF"): 1.033, ("AAPL", 22, "RF"): 1.548, ("AMZN", 1, "RF"): 0.893,
    ("AAPL", 5, "RF"): 0.936, ("AMZN", 22, "RF"): 2.098, ("AAPL", 1, "RF"): 1.054,
    ("JPM", 1, "RF"): 0.940,
}


class _BG1(Forecaster):
    name = "BG"

    def __init__(self, n_estimators, bootstrap, random_state):
        self.n_estimators = n_estimators; self.bootstrap = bootstrap
        self.random_state = random_state; self.model_ = None

    def fit(self, X, y):
        self.model_ = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.random_state),
            n_estimators=self.n_estimators, bootstrap=self.bootstrap,
            n_jobs=1, random_state=self.random_state).fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


class _RF1(Forecaster):
    name = "RF"

    def __init__(self, n_estimators, max_features, bootstrap, random_state):
        self.n_estimators = n_estimators; self.max_features = max_features
        self.bootstrap = bootstrap; self.random_state = random_state; self.model_ = None

    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features,
            bootstrap=self.bootstrap, n_jobs=1, random_state=self.random_state).fit(
            X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(X.to_numpy())


def _har_mse(ticker: str, horizon: int) -> float:
    for r in load_results("predictions_har_MALL.pkl"):
        if r.ticker == ticker and r.horizon == horizon and r.feature_set == "M_ALL":
            y = r.y_true; p = r.predictions["HAR"]
            idx = y.index.intersection(p.index)
            return float(np.mean(mse_loss(y.loc[idx].to_numpy(), p.loc[idx].to_numpy())))
    return np.nan


def _run_one(task: tuple) -> dict:
    """One (stock, horizon, model in {BG,RF}) rolling cell — mirrors 24b exactly."""
    ticker, h, label = task
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
    else:  # RF
        fac = lambda: _RF1(cfg.models_trees.random_forest.n_estimators,
                           cfg.models_trees.random_forest.max_features,
                           cfg.models_trees.random_forest.bootstrap, cfg.project.seed)

    t0 = time.time()
    out = rolling_window_forecast(fac, tr, vl, te, refit_frequency=1, progress=False)
    pred = out.predictions.clip(lower=floor)
    idx = out.y_true.index.intersection(pred.index)
    mse = float(np.mean(mse_loss(out.y_true.loc[idx].to_numpy(), pred.loc[idx].to_numpy())))
    har = _har_mse(ticker, h)
    ratio = mse / har if har == har else np.nan
    return {"stock": ticker, "horizon": h, "model": label,
            "rolling_ratio": round(ratio, 3), "secs": round(time.time() - t0, 1)}


def main() -> int:
    t0 = time.time()
    all_cells = [(s, h, m) for s in STOCKS for h in HORIZONS for m in MODELS]
    missing = [(s, h, m) for (s, h, m) in all_cells if (s, h, m) not in DONE]
    _LOG.info("salvage: %d cells already done, %d missing: %s",
              len(DONE), len(missing), [f"{s}-h{h}-{m}" for s, h, m in missing])

    results = dict(DONE)
    n_jobs = min(len(missing), max(1, (os.cpu_count() or 2) - 1))
    _LOG.info("computing %d missing cells on %d workers (inner n_jobs=1)", len(missing), n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_run_one, (s, h, m)): (s, h, m) for (s, h, m) in missing}
        for fut in as_completed(futs):
            r = fut.result()
            results[(r["stock"], r["horizon"], r["model"])] = r["rolling_ratio"]
            done = len(results) - len(DONE)
            _LOG.info("[%s h=%d] %s rolling=%.3f (%.0fs)  [%d/%d]",
                      r["stock"], r["horizon"], r["model"], r["rolling_ratio"], r["secs"],
                      done, len(missing))

    # Merge with the fixed-window ratios already in loss_ratio_h{h}_mse.csv.
    rows = []
    for h in HORIZONS:
        lr = pd.read_csv(f"outputs/tables/loss_ratio_h{h}_mse.csv", index_col=0)
        for s in STOCKS:
            for m in MODELS:
                rows.append({"stock": s, "horizon": h, "model": m,
                             "fixed_ratio": round(float(lr.loc[s, m]), 3),
                             "rolling_ratio": results[(s, h, m)]})
    out = (pd.DataFrame(rows)
             .sort_values(["stock", "horizon", "model"]).reset_index(drop=True))
    out_path = Path("outputs/tables/trees_fixed_vs_rolling_all.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\n" + out.to_string(index=False))
    _LOG.info("Saved %s", out_path)

    def _g(s, h, m, c):
        v = out[(out.stock == s) & (out.horizon == h) & (out.model == m)][c]
        return float(v.iloc[0])

    # Validate: a salvaged RF cell, a freshly-recomputed BG cell, and fixed ratios.
    checks = [("AAPL", 22, "RF", "rolling_ratio", 1.548),
              ("AAPL", 22, "BG", "rolling_ratio", 2.560),   # recomputed here — must match known
              ("AAPL", 22, "RF", "fixed_ratio", 3.038),
              ("AAPL", 22, "BG", "fixed_ratio", 7.314)]
    ok = True
    print("\n--- sanity check ---")
    for s, h, m, c, e in checks:
        got = _g(s, h, m, c); good = abs(got - e) <= 0.03 * max(1.0, abs(e))
        ok = ok and good
        print(f"{s} h{h} {m} {c}: got {got:.3f}  expect ~{e}  {'OK' if good else 'MISMATCH'}")
    print(f"\nTOTAL wall {time.time() - t0:.0f}s   sanity: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
