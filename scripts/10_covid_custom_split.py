"""
Stage 10 — COVID-period evaluation under a custom chronological split.

The headline 70/10/20 split puts the test window at 2023-onwards, so the
COVID-19 volatility shock is *inside the training period* and cannot be
forecast. This script reuses the same feature matrices but with:

* train = 2016-01 → 2019-06 (pre-COVID, three-year window for stable HAR fits)
* val   = 2019-07 → 2019-12
* test  = 2020-01 → 2024-12 (covers COVID + 2022 high-vol regime + 2023-24 normalisation)

Every configured model is trained once on the custom split and the test-set
predictions are saved to ``predictions_covid_<feature_set>_h<horizon>.pkl``.

Usage:
    python scripts/10_covid_custom_split.py
    python scripts/10_covid_custom_split.py --feature-set M_HAR --horizons 1
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.feature_engineering import load_feature_matrix
from src.models.har_models import make_har
from src.models.regularized import (
    RidgeForecaster, LassoForecaster, ElasticNetForecaster,
    PostLassoForecaster, AdaptiveLassoForecaster,
)
from src.models.tree_models import (
    BaggingForecaster, RandomForestForecaster, GradientBoostingForecaster,
)
from src.pipeline.orchestrator import (
    _make_regularized_factories, _make_tree_factories, _make_nn_factories,
    _HAR_ONLY_HELPERS, _ml_columns, StockRunResult, save_results,
)
from src.evaluation.metrics import mse_loss


def _custom_split(feats: pd.DataFrame,
                   train_end: str = "2019-06-30",
                   val_end: str = "2019-12-31") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = feats.loc[: train_end].copy()
    val   = feats.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1): val_end].copy()
    test  = feats.loc[pd.Timestamp(val_end)   + pd.Timedelta(days=1):].copy()
    return train, val, test


def _run_one_custom(ticker: str, feature_set: str, horizon: int, cfg,
                    skip_nn: bool, skip_trees: bool) -> StockRunResult:
    log = get_logger("covid_run", level=cfg.project.log_level)
    feats = load_feature_matrix(ticker, feature_set, horizon)
    train, val, test = _custom_split(feats)
    log.info("[%s|%s|h=%d] CUSTOM split sizes: train=%d val=%d test=%d (train ends %s, test from %s)",
             ticker, feature_set, horizon, len(train), len(val), len(test),
             train.index[-1].date() if len(train) else "—",
             test.index[0].date() if len(test) else "—")
    if len(train) < 50 or len(val) < 20 or len(test) < 50:
        raise RuntimeError("Insufficient data after custom split")

    ml_cols = _ml_columns(feats, feature_set)
    train_ml, val_ml, test_ml = train[ml_cols], val[ml_cols], test[ml_cols]

    predictions: dict[str, pd.Series] = {}
    val_predictions: dict[str, pd.Series] = {}

    # HAR family — fixed-window (train once on train+val for stability)
    full_train = pd.concat([train, val])
    for har_label in cfg.models_har:
        try:
            model = make_har(har_label).fit(full_train, full_train["y"])
            preds = model.predict(test)
            predictions[har_label] = pd.Series(preds, index=test.index, name=har_label)
            val_predictions[har_label] = pd.Series(
                make_har(har_label).fit(train, train["y"]).predict(val),
                index=val.index, name=har_label)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", har_label, e)

    if feature_set == "M_ALL":
        try:
            model = make_har("HAR-X").fit(full_train, full_train["y"])
            predictions["HAR-X"] = pd.Series(model.predict(test), index=test.index, name="HAR-X")
            val_predictions["HAR-X"] = pd.Series(
                make_har("HAR-X").fit(train, train["y"]).predict(val),
                index=val.index, name="HAR-X")
        except Exception as e:  # noqa: BLE001
            log.warning("HAR-X failed: %s", e)

    # Regularised
    reg_factories = _make_regularized_factories(cfg)
    for label, factory in reg_factories.items():
        try:
            model = factory()
            X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
            X_vl = val_ml.drop(columns=["y"]);   y_vl = val_ml["y"]
            X_te = test_ml.drop(columns=["y"])
            model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            predictions[label] = pd.Series(model.predict(X_te), index=X_te.index, name=label)
            val_predictions[label] = pd.Series(model.predict(X_vl), index=X_vl.index, name=label)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", label, e)

    # Trees
    if not skip_trees:
        tree_factories = _make_tree_factories(cfg)
        for label, factory in tree_factories.items():
            try:
                model = factory()
                X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
                X_vl = val_ml.drop(columns=["y"]);   y_vl = val_ml["y"]
                X_te = test_ml.drop(columns=["y"])
                if label == "GB":
                    model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
                else:
                    full_X = pd.concat([X_tr, X_vl])
                    full_y = pd.concat([y_tr, y_vl])
                    model.fit(full_X, full_y)
                predictions[label] = pd.Series(model.predict(X_te), index=X_te.index, name=label)
                val_predictions[label] = pd.Series(model.predict(X_vl), index=X_vl.index, name=label)
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed: %s", label, e)

    # NN
    if not skip_nn:
        try:
            nn_factories = _make_nn_factories(cfg)
            for label, factory in nn_factories.items():
                try:
                    model = factory()
                    X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
                    X_vl = val_ml.drop(columns=["y"]);   y_vl = val_ml["y"]
                    X_te = test_ml.drop(columns=["y"])
                    model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
                    predictions[label] = pd.Series(model.predict(X_te), index=X_te.index, name=label)
                    val_predictions[label] = pd.Series(model.predict(X_vl), index=X_vl.index, name=label)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s failed: %s", label, e)
        except ImportError:
            log.warning("PyTorch not available; skipping NN")

    # Positivity filter (Christensen et al. 2023, p.1691), as in the main
    # orchestrator: replace any negative variance forecast with the in-sample
    # minimum RV, uniformly across models.
    floor = float(train["y"].min())
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-12
    for _store in (predictions, val_predictions):
        for _k in list(_store.keys()):
            _store[_k] = _store[_k].clip(lower=floor)

    return StockRunResult(
        ticker=ticker, feature_set=feature_set, horizon=horizon,
        predictions=predictions, y_true=test["y"],
        val_predictions=val_predictions, y_val=val["y"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="COVID-period custom-split evaluation.")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--skip-nn", action="store_true")
    parser.add_argument("--skip-trees", action="store_true")
    parser.add_argument("--output-prefix", default="predictions_covid")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("covid_run", level=cfg.project.log_level)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    for h in horizons:
        results: list[StockRunResult] = []
        for ticker in stocks:
            log.info("=== %s | %s | h=%d ===", ticker, args.feature_set, h)
            t0 = time.time()
            try:
                res = _run_one_custom(ticker, args.feature_set, h, cfg,
                                      skip_nn=args.skip_nn, skip_trees=args.skip_trees)
                results.append(res)
                for label, preds in res.predictions.items():
                    idx = res.y_true.index.intersection(preds.index)
                    mse = float(np.mean(mse_loss(res.y_true.loc[idx].to_numpy(),
                                                  preds.loc[idx].to_numpy())))
                    log.info("  %-15s  MSE = %.6e (n=%d)", label, mse, len(idx))
                log.info("  elapsed %.1f min", (time.time() - t0) / 60.0)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed for %s|%s|h=%d: %s", ticker, args.feature_set, h, exc, exc_info=True)
        if results:
            out = f"{args.output_prefix}_{args.feature_set}_h{h}.pkl"
            save_results(results, filename=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
