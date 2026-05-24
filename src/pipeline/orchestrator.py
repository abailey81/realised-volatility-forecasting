"""
Top-level orchestration: run every model on every stock and persist results.

The orchestrator reads :mod:`config.yaml`, builds the relevant feature
matrices, instantiates all configured forecasters, runs them through the
appropriate forecasting scheme (fixed-window for NN, rolling for HAR/OLS,
rolling-with-frozen-hyperparameters for regularised/trees), and writes a
pickled dictionary of predictions to ``outputs/results/predictions.pkl``.

Downstream scripts (``07_run_tests.py``, ``09_generate_outputs.py``) load
this file and produce tests, tables, and figures.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve, set_global_seed
from ..data.feature_engineering import load_feature_matrix, time_split
from ..models.base import Forecaster
from ..models.har_models import make_har
from ..models.regularized import (
    RidgeForecaster,
    LassoForecaster,
    ElasticNetForecaster,
    PostLassoForecaster,
    AdaptiveLassoForecaster,
)
from ..models.tree_models import (
    BaggingForecaster,
    RandomForestForecaster,
    GradientBoostingForecaster,
)
from .rolling_forecast import (
    fixed_window_forecast,
    rolling_window_forecast,
    ForecastOutput,
)

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_log_grid(spec) -> list[float]:
    """Build a log-spaced grid of penalty values from a config block.

    Supports two formats:
    - ``alpha_grid`` (list of explicit values) — backward-compatible
    - ``alpha_grid_log10`` (``[low, high]`` bounds) + ``alpha_grid_n`` (count)
      → ``np.logspace(low, high, n)`` per paper Appendix A.4 Table A.6.
    """
    import numpy as np
    if hasattr(spec, "alpha_grid_log10"):
        lo, hi = spec.alpha_grid_log10
        n = int(getattr(spec, "alpha_grid_n", 100))
        return np.logspace(float(lo), float(hi), n).tolist()
    return list(spec.alpha_grid)


def _make_regularized_factories(cfg) -> dict[str, Callable[[], Forecaster]]:
    ridge_grid   = _build_log_grid(cfg.models_regularized.ridge)
    lasso_grid   = _build_log_grid(cfg.models_regularized.lasso)
    en_grid      = _build_log_grid(cfg.models_regularized.elastic_net)
    pla_grid     = _build_log_grid(cfg.models_regularized.post_lasso)
    ala_grid     = _build_log_grid(cfg.models_regularized.adaptive_lasso)
    return {
        "RR":   lambda: RidgeForecaster(ridge_grid),
        "LA":   lambda: LassoForecaster(lasso_grid),
        "EN":   lambda: ElasticNetForecaster(
            en_grid,
            cfg.models_regularized.elastic_net.l1_ratio_grid,
        ),
        "P-LA": lambda: PostLassoForecaster(pla_grid),
        "A-LA": lambda: AdaptiveLassoForecaster(
            gamma=cfg.models_regularized.adaptive_lasso.gamma,
            alpha_grid=ala_grid,
        ),
    }


def _make_tree_factories(cfg) -> dict[str, Callable[[], Forecaster]]:
    return {
        "BG": lambda: BaggingForecaster(
            n_estimators=cfg.models_trees.bagging.n_estimators,
            bootstrap=cfg.models_trees.bagging.bootstrap,
            random_state=cfg.project.seed,
        ),
        "RF": lambda: RandomForestForecaster(
            n_estimators=cfg.models_trees.random_forest.n_estimators,
            max_features=cfg.models_trees.random_forest.max_features,
            bootstrap=cfg.models_trees.random_forest.bootstrap,
            random_state=cfg.project.seed,
        ),
        "GB": lambda: GradientBoostingForecaster(
            n_estimators_grid=cfg.models_trees.gradient_boosting.n_estimators_grid,
            learning_rate_grid=cfg.models_trees.gradient_boosting.learning_rate_grid,
            max_depth_grid=cfg.models_trees.gradient_boosting.max_depth_grid,
            subsample=cfg.models_trees.gradient_boosting.subsample,
            random_state=cfg.project.seed,
        ),
    }


def _make_nn_factories(cfg) -> dict[str, Callable[[], Forecaster]]:
    """Return a dict of NN ensemble factories (one per architecture).

    Prefers the PyTorch backend (leaky ReLU, dropout, learning-rate scheduler).
    Falls back to a scikit-learn ``MLPRegressor`` backend (ReLU, Adam, early
    stopping) when PyTorch is not installed.

    The training is done once per architecture (100 seeds). At prediction
    time both ``NN_d^1`` (single best seed) and ``NN_d^10`` (top-10 avg) are
    produced — paper Christensen et al. (2023) reports both variants.
    """
    try:
        from ..models.neural_networks import NNEnsembleForecaster, _NNTrainConfig
        _backend = "torch"
    except ImportError:
        from ..models.neural_networks_sklearn import (
            MLPEnsembleForecaster as NNEnsembleForecaster,  # type: ignore[assignment]
            _MLPTrainConfig as _NNTrainConfig,                # type: ignore[assignment]
        )
        _backend = "sklearn"
        _LOG.info("NN backend: scikit-learn MLPRegressor (PyTorch unavailable)")

    n_jobs = int(getattr(cfg.models_nn, "parallel", {}).get("n_jobs", -1)) \
             if isinstance(getattr(cfg.models_nn, "parallel", None), dict) \
             else int(getattr(getattr(cfg.models_nn, "parallel", None), "n_jobs", -1))

    factories: dict[str, Callable[[], Forecaster]] = {}
    for label, dims in cfg.models_nn.architectures.items():
        train_cfg = _NNTrainConfig(
            hidden_dims=list(dims),
            epochs=cfg.models_nn.epochs,
            batch_size=cfg.models_nn.batch_size,
            lr=cfg.models_nn.learning_rate,
            dropout=cfg.models_nn.dropout,
            leaky_slope=cfg.models_nn.leaky_relu_slope,
            early_stop_patience=cfg.models_nn.early_stopping_patience,
            scheduler_factor=cfg.models_nn.scheduler.factor,
            scheduler_patience=cfg.models_nn.scheduler.patience,
        )

        def _factory(d=list(dims), tc=train_cfg, lab=label, nj=n_jobs):
            return NNEnsembleForecaster(
                hidden_dims=d,
                name=lab,
                num_random_seeds=cfg.models_nn.ensemble.num_random_seeds,
                top_k=cfg.models_nn.ensemble.ensemble_top_k,
                base_seed=cfg.project.seed,
                train_cfg=tc,
                n_jobs=nj,
            )

        # Single key per architecture; the orchestrator below produces both
        # NN_d^1 and NN_d^10 predictions from the same fitted ensemble.
        factories[f"{label}_ensemble"] = _factory
    return factories


# ---------------------------------------------------------------------------
# Single-stock runner
# ---------------------------------------------------------------------------

# HAR-family auxiliary columns kept in the feature matrix for the specific
# HAR variants but excluded from ML model designs.
_HAR_ONLY_HELPERS = {"RVD_pos", "RVD_neg", "Rn_D", "Rn_W", "Rn_M", "RQ_x_RV", "RQ_lag"}


def _ml_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Columns that ML models should see for a given feature set.

    For ``M_HAR`` only the three HAR lags are kept (paper-faithful).
    For ``M_ALL`` the macro and per-stock features are kept, with the HAR
    helpers excluded.
    """
    if feature_set.upper() == "M_HAR":
        return [c for c in ("RVD", "RVW", "RVM", "y") if c in df.columns]
    return [c for c in df.columns if c not in _HAR_ONLY_HELPERS]


def _predict_block(model, X: pd.DataFrame, name: str) -> pd.Series:
    """Generate predictions for an arbitrary feature block and name them."""
    arr = model.predict(X)
    return pd.Series(arr, index=X.index, name=name)


@dataclass
class StockRunResult:
    ticker: str
    feature_set: str
    horizon: int
    predictions: dict[str, pd.Series]
    y_true: pd.Series
    val_predictions: dict[str, pd.Series] = None  # type: ignore[assignment]
    y_val: pd.Series = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.val_predictions is None:
            self.val_predictions = {}


def run_one(
    ticker: str,
    feature_set: str,
    horizon: int,
    cfg=None,
    skip_nn: bool = False,
    skip_trees: bool = False,
    skip_har: bool = False,
    skip_regularised: bool = False,
) -> StockRunResult:
    """Run all configured models for one (stock, feature set, horizon) combination.

    On M_HAR runs, ML methods see only the three HAR lags (paper-faithful).
    On M_ALL runs, ML methods see HAR lags + macro/per-stock features but
    not the HAR-only helpers (semivariance, leverage, quarticity-interaction).
    """
    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)

    feats = load_feature_matrix(ticker, feature_set, horizon)
    train, val, test = time_split(
        feats,
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
    )
    _LOG.info(
        "[%s|%s|h=%d] split sizes: train=%d val=%d test=%d",
        ticker, feature_set, horizon, len(train), len(val), len(test),
    )

    # Restricted views for ML methods.
    ml_cols = _ml_columns(feats, feature_set)
    train_ml, val_ml, test_ml = train[ml_cols], val[ml_cols], test[ml_cols]

    predictions: dict[str, pd.Series] = {}
    val_predictions: dict[str, pd.Series] = {}

    # ---- HAR family (rolling OLS) -----------------------------------------
    if not skip_har:
        for har_label in cfg.models_har:
            try:
                out = rolling_window_forecast(
                    model_factory=lambda lab=har_label: make_har(lab),
                    train=train, val=val, test=test,
                    refit_frequency=cfg.forecast.refit_frequency_days,
                    progress=False,
                )
                predictions[har_label] = out.predictions
                # Also record a single-fit validation prediction (one OLS fit
                # on `train`, predict on `val`). Used by forecast combinations.
                try:
                    m_val = make_har(har_label).fit(train, train["y"])
                    val_predictions[har_label] = _predict_block(m_val, val, har_label)
                except Exception as ve:  # noqa: BLE001
                    _LOG.warning("%s val-pred failed: %s", har_label, ve)
            except KeyError as e:
                _LOG.warning("Skipping HAR variant %s: %s", har_label, e)

        # HAR-X uses the full M_ALL set when feature_set == "M_ALL".
        if feature_set == "M_ALL":
            try:
                out = rolling_window_forecast(
                    model_factory=lambda: make_har("HAR-X"),
                    train=train, val=val, test=test,
                    refit_frequency=cfg.forecast.refit_frequency_days,
                    progress=False,
                )
                predictions["HAR-X"] = out.predictions
                try:
                    m_val = make_har("HAR-X").fit(train, train["y"])
                    val_predictions["HAR-X"] = _predict_block(m_val, val, "HAR-X")
                except Exception as ve:  # noqa: BLE001
                    _LOG.warning("HAR-X val-pred failed: %s", ve)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("HAR-X failed: %s", e)

    # ---- Regularised regression (fixed-window) ----------------------------
    if not skip_regularised:
        reg_factories = _make_regularized_factories(cfg)
        for label, factory in reg_factories.items():
            try:
                model = factory()
                X_tr = train_ml.drop(columns=["y"]); y_tr = train_ml["y"]
                X_vl = val_ml.drop(columns=["y"]);   y_vl = val_ml["y"]
                X_te = test_ml.drop(columns=["y"])
                model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
                predictions[label] = _predict_block(model, X_te, label)
                val_predictions[label] = _predict_block(model, X_vl, label)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("%s failed: %s", label, e)

    # ---- Tree-based methods (fixed-window) --------------------------------
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
                predictions[label] = _predict_block(model, X_te, label)
                val_predictions[label] = _predict_block(model, X_vl, label)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("%s failed: %s", label, e)

    # ---- Neural networks (fixed-window) -----------------------------------
    # Per Christensen et al. (2023), we report two ensemble variants per
    # architecture: NN_d^1 (single best-validation-MSE seed) and NN_d^10
    # (mean of the top-10 seeds out of 100). Both come from the SAME fitted
    # ensemble — we don't pay 2× training cost.
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
                    # NN^10 (top-10 average)
                    predictions[label] = _predict_block(model, X_te, label)
                    val_predictions[label] = _predict_block(model, X_vl, label)
                    # NN^1 (best single seed)
                    arch = label.replace("_ensemble", "")
                    label1 = f"{arch}_top1"
                    if hasattr(model, "predict_top1"):
                        pred1_te = model.predict_top1(X_te)
                        pred1_vl = model.predict_top1(X_vl)
                        predictions[label1] = pd.Series(pred1_te, index=X_te.index, name=label1)
                        val_predictions[label1] = pd.Series(pred1_vl, index=X_vl.index, name=label1)
                except Exception as e:  # noqa: BLE001
                    _LOG.warning("%s failed: %s", label, e)
        except ImportError:
            _LOG.warning("PyTorch not available; skipping NN models")

    # ---- Positivity filter (Christensen et al. 2023, p. 1691) -------------
    # "If a model predicts volatility to be negative, we replace the forecast
    # with the minimum in-sample realized variance." Applied uniformly to
    # every model so neither MSE nor QLIKE is contaminated by negative-variance
    # extrapolation from unconstrained linear models (HAR-X, RR) or the
    # standardised-NN inverse transform.
    floor = float(train["y"].min())
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-12
    for _store in (predictions, val_predictions):
        for _k in list(_store.keys()):
            _store[_k] = _store[_k].clip(lower=floor)

    return StockRunResult(
        ticker=ticker,
        feature_set=feature_set,
        horizon=horizon,
        predictions=predictions,
        y_true=test["y"],
        val_predictions=val_predictions,
        y_val=val["y"],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: list[StockRunResult], filename: str = "predictions.pkl") -> str:
    cfg = load_config()
    out_dir = resolve(cfg.paths.outputs_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "wb") as f:
        pickle.dump([r.__dict__ for r in results], f)
    _LOG.info("Saved %d run results to %s", len(results), path)
    return str(path)


def load_results(filename: str = "predictions.pkl") -> list[StockRunResult]:
    cfg = load_config()
    path = resolve(cfg.paths.outputs_results) / filename
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return [StockRunResult(**d) for d in raw]
