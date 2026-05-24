"""
Feed-forward neural networks with the paper's geometric pyramid architecture.

The paper specifies four architectures:

* **NN1**: single hidden layer with 2 neurons.
* **NN2**: two hidden layers with 4, 2 neurons.
* **NN3**: three hidden layers with 8, 4, 2 neurons.
* **NN4**: four hidden layers with 16, 8, 4, 2 neurons.

All use:

* **Leaky ReLU** activation (slope :math:`c = 0.01`),
* **Adam** optimiser with learning rate 0.001 (Kingma & Ba 2014),
* **Drop-out** during training,
* **Early stopping** on the validation set,
* **Learning-rate scheduling** that halves on plateau,
* **Ensemble averaging** over multiple seeds: train ``num_random_seeds``
  initialisations, retain the top ``ensemble_top_k`` by validation MSE,
  and average their predictions (the paper's :math:`NN_d^{10}`).

The implementation uses PyTorch. Training time on CPU for one full
M_ALL × NN4 × 100 seeds on the AAPL sample is approximately 5–15 minutes
depending on hardware; the orchestrator under ``src/pipeline`` parallelises
the ensemble across processes when ``num_workers > 1``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .base import Forecaster


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class GeometricPyramidNN(nn.Module):
    """Feed-forward NN with the paper's geometric pyramid architecture."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int],
                 dropout: float = 0.1, leaky_slope: float = 0.01):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LeakyReLU(negative_slope=leaky_slope),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@dataclass
class _NNTrainConfig:
    hidden_dims: Sequence[int]
    epochs: int = 200
    batch_size: int = 64
    lr: float = 0.001
    dropout: float = 0.1
    leaky_slope: float = 0.01
    early_stop_patience: int = 25
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10


def _train_single(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    cfg: _NNTrainConfig,
    seed: int,
    device: str = "cpu",
) -> tuple[nn.Module, float]:
    """Train one NN to convergence and return (model_state, best_val_mse)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_tr.shape[1]
    model = GeometricPyramidNN(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        leaky_slope=cfg.leaky_slope,
    ).to(device)

    optimiser = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=device)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state: dict | None = None
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl_t)
            val_mse = float(loss_fn(val_pred, y_vl_t).item())
        scheduler.step(val_mse)

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


# ---------------------------------------------------------------------------
# Forecaster wrapper with ensemble averaging
# ---------------------------------------------------------------------------

class NNEnsembleForecaster(Forecaster):
    """Train ``n_seeds`` NNs and average the best ``top_k`` by val MSE.

    This implements the :math:`NN_d^{10}` specification of the paper:
    given ``num_random_seeds=100`` and ``top_k=10``, the final prediction
    is the simple average of the 10 lowest-val-MSE single-seed networks.
    """

    def __init__(self,
                 hidden_dims: Sequence[int],
                 name: str | None = None,
                 num_random_seeds: int = 100,
                 top_k: int = 10,
                 base_seed: int = 42,
                 device: str | None = None,
                 train_cfg: _NNTrainConfig | None = None):
        self.hidden_dims = list(hidden_dims)
        self.name = name or f"NN{len(hidden_dims)}"
        self.num_random_seeds = num_random_seeds
        self.top_k = top_k
        self.base_seed = base_seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg or _NNTrainConfig(hidden_dims=self.hidden_dims)
        self.train_cfg.hidden_dims = self.hidden_dims

        self.scaler_X: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.members_: list[nn.Module] = []
        self.member_val_mses_: list[float] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "NNEnsembleForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * 0.875)  # 70/10 of total → 87.5/12.5 of provided
            X_tr_df, X_vl_df = X.iloc[:split], X.iloc[split:]
            y_tr_s,  y_vl_s  = y.iloc[:split], y.iloc[split:]
        else:
            X_tr_df, y_tr_s = X, y
            X_vl_df, y_vl_s = X_val, y_val

        # Standardise X using training statistics; standardise y as well so the
        # NN trains on a unit-variance target, helping convergence.
        self.scaler_X = StandardScaler().fit(X_tr_df.to_numpy())
        self.scaler_y = StandardScaler().fit(y_tr_s.to_numpy().reshape(-1, 1))

        X_tr = self.scaler_X.transform(X_tr_df.to_numpy())
        X_vl = self.scaler_X.transform(X_vl_df.to_numpy())
        y_tr = self.scaler_y.transform(y_tr_s.to_numpy().reshape(-1, 1)).ravel()
        y_vl = self.scaler_y.transform(y_vl_s.to_numpy().reshape(-1, 1)).ravel()

        candidates: list[tuple[float, nn.Module]] = []
        for k in range(self.num_random_seeds):
            seed = self.base_seed + k
            model, val_mse = _train_single(X_tr, y_tr, X_vl, y_vl,
                                           cfg=self.train_cfg, seed=seed,
                                           device=self.device)
            candidates.append((val_mse, model))

        candidates.sort(key=lambda t: t[0])
        kept = candidates[: self.top_k]
        self.members_ = [m for _, m in kept]
        self.member_val_mses_ = [v for v, _ in kept]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler_X.transform(X.to_numpy())
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for model in self.members_:
                p = model(Xt).cpu().numpy()
                preds.append(p)
        avg_scaled = np.mean(preds, axis=0)
        # Inverse-transform back to RV scale.
        return self.scaler_y.inverse_transform(avg_scaled.reshape(-1, 1)).ravel()
