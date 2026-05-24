"""
Stage 16 — Training-set length sensitivity (paper Appendix A.1, Tables A.1-A.4).

The paper reports relative MSE for two alternative training windows:
**1000 days** and **2000 days**, with the validation set fixed at 200 days
and the remainder used for testing. This script reproduces that exercise
across all three stocks and the headline horizons.

The (window_size × stock × horizon) tasks are independent, so we run them
in process-parallel (32-worker oversubscription per user spec).

Outputs:
* ``outputs/results/predictions_trainsize_<size>_M_ALL_h<h>.pkl``
* ``outputs/tables/trainsize_<size>_loss_ratio_h<h>_mse.csv``

Usage:
    python scripts/16_training_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.evaluation.metrics import mse_loss
from src.data.feature_engineering import load_feature_matrix
from src.models.har_models import make_har
from src.pipeline.orchestrator import (
    _make_regularized_factories, _make_tree_factories, _make_nn_factories,
    _ml_columns, StockRunResult, save_results,
)


def _custom_window_split(feats: pd.DataFrame, train_size: int, val_size: int):
    """Take a contiguous window ending where the standard split's test starts.

    Returns (train, val, test). All test rows come from the original
    chronological tail. train_size + val_size rows precede test.
    """
    n = len(feats)
    needed = train_size + val_size
    if needed >= n:
        raise ValueError(f"train+val={needed} exceeds n={n}")
    # Test always anchored at the end so test sets are comparable across sizes.
    test = feats.iloc[needed:].copy()
    val = feats.iloc[train_size: train_size + val_size].copy()
    train = feats.iloc[: train_size].copy()
    return train, val, test


def _run_one(ticker: str, feature_set: str, horizon: int, train_size: int,
             val_size: int, cfg, skip_nn: bool) -> StockRunResult:
    log = get_logger("trainsize", level=cfg.project.log_level)
    feats = load_feature_matrix(ticker, feature_set, horizon)
    train, val, test = _custom_window_split(feats, train_size, val_size)
    log.info("[%s|%s|h=%d|train=%d] split sizes: train=%d val=%d test=%d",
             ticker, feature_set, horizon, train_size, len(train), len(val), len(test))
    if len(train) < 50 or len(val) < 20 or len(test) < 50:
        raise RuntimeError("Insufficient data after split")

    ml_cols = _ml_columns(feats, feature_set)
    train_ml, val_ml, test_ml = train[ml_cols], val[ml_cols], test[ml_cols]
    full_train = pd.concat([train, val])

    predictions: dict[str, pd.Series] = {}
    val_predictions: dict[str, pd.Series] = {}

    # HAR family — fixed-window for tractability
    for har_label in cfg.models_har:
        try:
            m = make_har(har_label).fit(full_train, full_train["y"])
            predictions[har_label] = pd.Series(m.predict(test), index=test.index, name=har_label)
            m_val = make_har(har_label).fit(train, train["y"])
            val_predictions[har_label] = pd.Series(m_val.predict(val), index=val.index, name=har_label)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", har_label, e)
    if feature_set == "M_ALL":
        try:
            m = make_har("HAR-X").fit(full_train, full_train["y"])
            predictions["HAR-X"] = pd.Series(m.predict(test), index=test.index, name="HAR-X")
            m_val = make_har("HAR-X").fit(train, train["y"])
            val_predictions["HAR-X"] = pd.Series(m_val.predict(val), index=val.index, name="HAR-X")
        except Exception as e:  # noqa: BLE001
            log.warning("HAR-X failed: %s", e)

    # Regularised
    for label, factory in _make_regularized_factories(cfg).items():
        try:
            m = factory()
            X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
            X_vl = val_ml.drop(columns=["y"]); y_vl = val_ml["y"]
            X_te = test_ml.drop(columns=["y"])
            m.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            predictions[label] = pd.Series(m.predict(X_te), index=X_te.index, name=label)
            val_predictions[label] = pd.Series(m.predict(X_vl), index=X_vl.index, name=label)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", label, e)

    # Trees
    for label, factory in _make_tree_factories(cfg).items():
        try:
            m = factory()
            X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
            X_vl = val_ml.drop(columns=["y"]); y_vl = val_ml["y"]
            X_te = test_ml.drop(columns=["y"])
            if label == "GB":
                m.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            else:
                m.fit(pd.concat([X_tr, X_vl]), pd.concat([y_tr, y_vl]))
            predictions[label] = pd.Series(m.predict(X_te), index=X_te.index, name=label)
            val_predictions[label] = pd.Series(m.predict(X_vl), index=X_vl.index, name=label)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", label, e)

    # NN
    if not skip_nn:
        try:
            for label, factory in _make_nn_factories(cfg).items():
                m = factory()
                X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
                X_vl = val_ml.drop(columns=["y"]); y_vl = val_ml["y"]
                X_te = test_ml.drop(columns=["y"])
                m.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
                predictions[label] = pd.Series(m.predict(X_te), index=X_te.index, name=label)
                val_predictions[label] = pd.Series(m.predict(X_vl), index=X_vl.index, name=label)
                arch = label.replace("_ensemble", "")
                if hasattr(m, "predict_top1"):
                    pred1 = m.predict_top1(X_te)
                    predictions[f"{arch}_top1"] = pd.Series(pred1, index=X_te.index, name=f"{arch}_top1")
                    val_predictions[f"{arch}_top1"] = pd.Series(m.predict_top1(X_vl),
                                                                  index=X_vl.index,
                                                                  name=f"{arch}_top1")
        except ImportError:
            log.warning("PyTorch not available; skipping NN")

    return StockRunResult(ticker=ticker, feature_set=feature_set, horizon=horizon,
                          predictions=predictions, y_true=test["y"],
                          val_predictions=val_predictions, y_val=val["y"])


def _summarise(results: list[StockRunResult], horizon: int, train_size: int) -> None:
    cfg = load_config()
    out_dir = resolve(cfg.paths.outputs_tables)
    rows = {}
    for r in results:
        rows[r.ticker] = {}
        for m, p in r.predictions.items():
            idx = r.y_true.index.intersection(p.index)
            rows[r.ticker][m] = float(np.mean(mse_loss(r.y_true.loc[idx].to_numpy(),
                                                        p.loc[idx].to_numpy())))
    df = pd.DataFrame(rows).T
    if "HAR" in df.columns:
        df_ratio = df.div(df["HAR"], axis=0)
        df_ratio.to_csv(out_dir / f"trainsize_{train_size}_loss_ratio_h{horizon}_mse.csv")
    df.to_csv(out_dir / f"trainsize_{train_size}_loss_h{horizon}_mse.csv")


def _run_size_horizon(train_size: int, val_size: int, horizon: int, feature_set: str,
                      stocks: list[str], skip_nn: bool) -> dict:
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    results = []
    for ticker in stocks:
        try:
            res = _run_one(ticker, feature_set, horizon, train_size, val_size, cfg, skip_nn)
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            get_logger("trainsize").error("Failed %s|h=%d|train=%d: %s",
                                          ticker, horizon, train_size, exc)
    if results:
        save_results(results,
                     filename=f"predictions_trainsize_{train_size}_{feature_set}_h{horizon}.pkl")
        _summarise(results, horizon, train_size)
    return {"size": train_size, "horizon": horizon, "n_stocks": len(results)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=[1000, 2000])
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip NN training (much faster).")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("trainsize", level=cfg.project.log_level)
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    tasks = [(sz, args.val_size, h, args.feature_set, list(cfg.data.stocks), args.skip_nn)
             for sz in args.sizes for h in horizons]
    log.info("Running %d (size × horizon) tasks", len(tasks))
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        futs = {ex.submit(_run_size_horizon, *t): t for t in tasks}
        for fut in as_completed(futs):
            log.info("done: %s", fut.result())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
