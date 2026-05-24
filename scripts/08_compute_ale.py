"""
Stage 8 — compute Accumulated Local Effects for the headline ML models.

For each (model, feature) combination, retrain on train+val (so the fitted
model is on full training data) and compute the ALE curve on the training
distribution. Save the per-curve series to ``outputs/results/ale/``.

Usage:
    python scripts/08_compute_ale.py
    python scripts/08_compute_ale.py --stock AAPL --feature RVD --models EN RF
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.models.har_models import make_har
from src.models.regularized import (
    RidgeForecaster, LassoForecaster, ElasticNetForecaster,
    PostLassoForecaster, AdaptiveLassoForecaster,
)
from src.models.tree_models import (
    BaggingForecaster, RandomForestForecaster, GradientBoostingForecaster,
)
from src.evaluation.ale import accumulated_local_effects, variable_importance_from_ale


def _build_model(label: str, cfg):
    from src.pipeline.orchestrator import _build_log_grid
    if label in {"HAR", "LogHAR", "LevHAR", "SHAR", "HARQ", "HAR-X"}:
        return make_har(label)
    if label == "RR":
        return RidgeForecaster(_build_log_grid(cfg.models_regularized.ridge))
    if label == "LA":
        return LassoForecaster(_build_log_grid(cfg.models_regularized.lasso))
    if label == "EN":
        return ElasticNetForecaster(_build_log_grid(cfg.models_regularized.elastic_net),
                                    cfg.models_regularized.elastic_net.l1_ratio_grid)
    if label == "P-LA":
        return PostLassoForecaster(_build_log_grid(cfg.models_regularized.post_lasso))
    if label == "A-LA":
        return AdaptiveLassoForecaster(gamma=cfg.models_regularized.adaptive_lasso.gamma,
                                       alpha_grid=_build_log_grid(cfg.models_regularized.adaptive_lasso))
    if label == "BG":
        return BaggingForecaster(n_estimators=cfg.models_trees.bagging.n_estimators,
                                 random_state=cfg.project.seed)
    if label == "RF":
        return RandomForestForecaster(n_estimators=cfg.models_trees.random_forest.n_estimators,
                                      random_state=cfg.project.seed)
    if label == "GB":
        return GradientBoostingForecaster(
            n_estimators_grid=cfg.models_trees.gradient_boosting.n_estimators_grid,
            learning_rate_grid=cfg.models_trees.gradient_boosting.learning_rate_grid,
            max_depth_grid=cfg.models_trees.gradient_boosting.max_depth_grid,
            subsample=cfg.models_trees.gradient_boosting.subsample,
            random_state=cfg.project.seed,
        )
    if label.endswith("_ensemble"):
        try:
            from src.models.neural_networks import NNEnsembleForecaster, _NNTrainConfig
        except ImportError:
            from src.models.neural_networks_sklearn import (
                MLPEnsembleForecaster as NNEnsembleForecaster,  # type: ignore[assignment]
                _MLPTrainConfig as _NNTrainConfig,                # type: ignore[assignment]
            )
        arch = label.replace("_ensemble", "")
        dims = cfg.models_nn.architectures[arch]
        tc = _NNTrainConfig(hidden_dims=list(dims),
                            epochs=cfg.models_nn.epochs,
                            batch_size=cfg.models_nn.batch_size,
                            lr=cfg.models_nn.learning_rate,
                            dropout=cfg.models_nn.dropout,
                            leaky_slope=cfg.models_nn.leaky_relu_slope,
                            early_stop_patience=cfg.models_nn.early_stopping_patience,
                            scheduler_factor=cfg.models_nn.scheduler.factor,
                            scheduler_patience=cfg.models_nn.scheduler.patience)
        return NNEnsembleForecaster(
            hidden_dims=list(dims), name=label,
            num_random_seeds=cfg.models_nn.ensemble.num_random_seeds,
            top_k=cfg.models_nn.ensemble.ensemble_top_k,
            base_seed=cfg.project.seed,
            train_cfg=tc,
        )
    raise KeyError(f"Unknown model label '{label}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ALE plots.")
    parser.add_argument("--stock", default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("compute_ale", level=cfg.project.log_level)

    stock = args.stock or cfg.ale.stock_for_plots
    features = args.features or list(cfg.ale.features)
    models = args.models or list(cfg.ale.models)

    feats = load_feature_matrix(stock, args.feature_set, args.horizon)
    train, val, _ = time_split(
        feats,
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
    )
    full_train = pd.concat([train, val])
    X = full_train.drop(columns=["y"])
    y = full_train["y"]

    out_dir = resolve(cfg.paths.outputs_results) / "ale"
    out_dir.mkdir(parents=True, exist_ok=True)

    ale_table: dict[tuple[str, str], object] = {}
    for label in models:
        try:
            log.info("Fitting %s on %s for ALE", label, stock)
            model = _build_model(label, cfg)
            # Models with optional validation use the split internally.
            try:
                model.fit(X, y, X_val=val.drop(columns=["y"]), y_val=val["y"])  # type: ignore[arg-type]
            except TypeError:
                model.fit(X, y)
            for feat in features:
                if feat not in X.columns:
                    log.warning("Feature %s not in %s columns; skipping", feat, stock)
                    continue
                res = accumulated_local_effects(
                    predict_fn=model.predict,
                    X=X,
                    feature=feat,
                    num_bins=cfg.ale.num_bins,
                )
                ale_table[(label, feat)] = res
        except Exception as exc:  # noqa: BLE001
            log.error("ALE failed for %s: %s", label, exc, exc_info=True)

    path = out_dir / f"ale_{stock}_h{args.horizon}.pkl"
    with open(path, "wb") as f:
        pickle.dump(ale_table, f)
    log.info("Saved %d ALE curves to %s", len(ale_table), path)

    # --- Variable Importance per model (paper Section 3.2, Figure 7) ---
    vi_rows = []
    for label in models:
        per_feature_res = {feat: ale_table[(label, feat)]
                           for feat in features
                           if (label, feat) in ale_table}
        if not per_feature_res:
            continue
        vi = variable_importance_from_ale(per_feature_res)
        for feat, val in vi.items():
            vi_rows.append({"model": label, "feature": feat, "VI": float(val)})
    if vi_rows:
        vi_df = pd.DataFrame(vi_rows).pivot(index="feature", columns="model", values="VI")
        out_tab = resolve(cfg.paths.outputs_tables) / f"vi_{stock}_h{args.horizon}.csv"
        vi_df.to_csv(out_tab)
        log.info("Saved variable-importance table to %s", out_tab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
