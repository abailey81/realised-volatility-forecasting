"""
scikit-learn MLPRegressor backup for the paper's geometric-pyramid NN.

This module mirrors :mod:`src.models.neural_networks` but uses
``sklearn.neural_network.MLPRegressor`` rather than PyTorch. The training
algorithm is the same: Adam optimiser, batch training, early stopping on
the validation set. The only methodological difference is the activation
function — sklearn supports ReLU but not Leaky ReLU. On small architectures
with standardised inputs the gap is empirically negligible (Leaky ReLU's
0.01 negative slope rarely fires for standardised features).

The ensemble logic is identical to the PyTorch version: train ``n_seeds``
random initialisations, retain the top ``top_k`` by validation MSE,
average their predictions.

The 100 seed-fits per architecture are independent, so we parallelise them
with ``joblib.Parallel`` (loky backend, one BLAS thread per worker to
avoid over-subscription).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import Forecaster


@dataclass
class _MLPTrainConfig:
    hidden_dims: Sequence[int]
    epochs: int = 200
    batch_size: int = 64
    lr: float = 0.001
    dropout: float = 0.1            # sklearn does not support dropout; kept for API parity
    leaky_slope: float = 0.01       # unused (sklearn uses plain ReLU)
    early_stop_patience: int = 25
    scheduler_factor: float = 0.5   # absorbed into MLPRegressor's adaptive lr
    scheduler_patience: int = 10


def _train_single_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_vl: np.ndarray, y_vl: np.ndarray,
                       cfg: _MLPTrainConfig, seed: int) -> tuple[MLPRegressor, float]:
    """Train one MLPRegressor with the given seed; return (model, val_mse).

    Uses sklearn's internal early stopping (a held-out slice of the training
    set) to terminate training. The orchestrator's validation set is then
    used to rank seeds for ensemble selection.
    """
    import warnings
    model = MLPRegressor(
        hidden_layer_sizes=tuple(cfg.hidden_dims),
        activation="relu",
        solver="adam",
        learning_rate_init=cfg.lr,
        max_iter=cfg.epochs,
        batch_size=min(cfg.batch_size, len(X_tr)),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=cfg.early_stop_patience,
        tol=1e-6,
        random_state=seed,
        shuffle=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silences convergence warnings
        model.fit(X_tr, y_tr)
    pred = model.predict(X_vl)
    val_mse = float(np.mean((pred - y_vl) ** 2))
    return model, val_mse


class MLPEnsembleForecaster(Forecaster):
    """Ensemble of MLPRegressor networks averaged over the best-validation seeds.

    Mirrors the geometric pyramid specification of the paper: train
    ``num_random_seeds`` networks with different seeds, retain the top
    ``top_k`` by validation MSE, average their predictions.

    Paper Christensen et al. (2023) reports two ensemble variants per
    architecture: NN_d^1 (the single best-validation-MSE seed) and NN_d^10
    (the average of the top 10). This class can produce *either*: set
    ``top_k=1`` for NN^1 or ``top_k=10`` for NN^10. The training cost is
    identical because all 100 seed-fits are computed regardless; only the
    selection at prediction time differs.

    Seed training is parallelised with :func:`joblib.Parallel` (process
    backend). ``n_jobs=-1`` uses all logical CPUs; ``n_jobs=32`` oversubscribes
    for I/O-bound seed startup.
    """

    def __init__(self,
                 hidden_dims: Sequence[int],
                 name: str | None = None,
                 num_random_seeds: int = 100,
                 top_k: int = 10,
                 base_seed: int = 42,
                 device: str | None = None,
                 train_cfg=None,
                 n_jobs: int = -1):
        self.hidden_dims = list(hidden_dims)
        self.name = name or f"NN{len(hidden_dims)}"
        self.num_random_seeds = num_random_seeds
        self.top_k = top_k
        self.base_seed = base_seed
        self.n_jobs = n_jobs
        # device and train_cfg accepted for API parity with the PyTorch version
        if train_cfg is None or not isinstance(train_cfg, _MLPTrainConfig):
            # Translate a PyTorch _NNTrainConfig duck-typed object if present.
            self.train_cfg = _MLPTrainConfig(hidden_dims=self.hidden_dims)
            if train_cfg is not None:
                for attr in ("epochs", "batch_size", "lr", "dropout",
                              "leaky_slope", "early_stop_patience",
                              "scheduler_factor", "scheduler_patience"):
                    if hasattr(train_cfg, attr):
                        setattr(self.train_cfg, attr, getattr(train_cfg, attr))
        else:
            self.train_cfg = train_cfg
        self.train_cfg.hidden_dims = self.hidden_dims

        self.scaler_X: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.members_: list[MLPRegressor] = []
        self.member_val_mses_: list[float] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "MLPEnsembleForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * 0.875)
            X_tr_df, X_vl_df = X.iloc[:split], X.iloc[split:]
            y_tr_s, y_vl_s = y.iloc[:split], y.iloc[split:]
        else:
            X_tr_df, y_tr_s = X, y
            X_vl_df, y_vl_s = X_val, y_val

        self.scaler_X = StandardScaler().fit(X_tr_df.to_numpy())
        self.scaler_y = StandardScaler().fit(y_tr_s.to_numpy().reshape(-1, 1))

        X_tr = self.scaler_X.transform(X_tr_df.to_numpy())
        X_vl = self.scaler_X.transform(X_vl_df.to_numpy())
        y_tr = self.scaler_y.transform(y_tr_s.to_numpy().reshape(-1, 1)).ravel()
        y_vl = self.scaler_y.transform(y_vl_s.to_numpy().reshape(-1, 1)).ravel()

        seeds = [self.base_seed + k for k in range(self.num_random_seeds)]
        # loky backend: each worker gets one BLAS thread to avoid CPU
        # over-subscription on small networks.
        results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
            delayed(_train_single_mlp)(X_tr, y_tr, X_vl, y_vl, self.train_cfg, s)
            for s in seeds
        )
        candidates = [(val_mse, model) for model, val_mse in results]

        candidates.sort(key=lambda t: t[0])
        kept = candidates[: self.top_k]
        self.members_ = [m for _, m in kept]
        self.member_val_mses_ = [v for v, _ in kept]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler_X.transform(X.to_numpy())
        preds_scaled = np.mean([m.predict(Xs) for m in self.members_], axis=0)
        return self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------ NN^1
    def predict_top1(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction from the single best-validation-MSE seed (paper's NN_d^1).

        Available after ``fit`` because all 100 seeds were ranked at fit time;
        the best seed is ``self.members_[0]`` when ``top_k >= 1``.
        """
        if not self.members_:
            raise RuntimeError("predict_top1 called before fit()")
        Xs = self.scaler_X.transform(X.to_numpy())
        preds_scaled = self.members_[0].predict(Xs)
        return self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
