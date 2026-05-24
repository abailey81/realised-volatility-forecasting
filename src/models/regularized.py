"""
Regularised regression: Ridge, Lasso, Elastic Net, Post-Lasso, Adaptive Lasso.

All models use ``sklearn`` for the underlying solver and add:

1. **Validation-set hyperparameter selection** (not k-fold CV) consistent
   with the paper's Section 1.3 and Appendix A.4.
2. **Standardisation** of features using training-set statistics so that
   penalties are scale-invariant. The target is left on its native RV scale.
3. For **Post-Lasso**: a two-stage procedure where Lasso selects the non-zero
   coefficients and an unrestricted OLS is then estimated on that subset.
4. For **Adaptive Lasso** (Zou 2006): first-stage OLS estimates inform the
   per-coefficient penalty weights :math:`w_i = 1 / |\\hat{\\beta}_i^{(1)}|^{\\gamma}`.

Each class exposes a ``.diagnostics`` dict after fitting that includes the
selected hyperparameters, the number of non-zero coefficients, and the
validation MSE — useful for the appendix's hyperparameter table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge as SkRidge,
    Lasso as SkLasso,
    ElasticNet as SkElasticNet,
    LinearRegression,
)
from sklearn.preprocessing import StandardScaler

from .base import Forecaster


@dataclass
class _RegResult:
    alpha: float
    l1_ratio: float | None = None
    n_nonzero: int = 0
    val_mse: float = float("inf")
    extra: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validation_mse(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    pred = model.predict(X_val)
    return float(np.mean((pred - y_val) ** 2))


class _ScaledLinearMixin:
    """Provides feature standardisation logic shared by all reg models."""

    scaler_: StandardScaler | None = None

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Scaler not fitted")
        return self.scaler_.transform(X)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

class RidgeForecaster(Forecaster, _ScaledLinearMixin):
    """Ridge regression with validation-set selection of alpha."""

    name = "RR"

    def __init__(self, alpha_grid: Sequence[float], val_frac: float = 0.10 / 0.80):
        # val_frac is the *training* set fraction reserved for hyper-selection
        # when no explicit validation set is provided.
        self.alpha_grid = list(alpha_grid)
        self.val_frac = val_frac
        self.model_: SkRidge | None = None
        self.diagnostics: _RegResult | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "RidgeForecaster":
        self._feature_names = list(X.columns)
        # If no explicit validation set is provided, use the tail of training.
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val

        X_tr_s = self._fit_scaler(X_tr.to_numpy())
        X_vl_s = self._apply_scaler(X_vl.to_numpy())

        best = _RegResult(alpha=self.alpha_grid[0])
        best_model: SkRidge | None = None
        for alpha in self.alpha_grid:
            m = SkRidge(alpha=alpha, fit_intercept=True)
            m.fit(X_tr_s, y_tr.to_numpy())
            mse = _validation_mse(m, X_vl_s, y_vl.to_numpy())
            if mse < best.val_mse:
                best = _RegResult(alpha=alpha, n_nonzero=int(np.sum(m.coef_ != 0)), val_mse=mse)
                best_model = m

        # Refit on full training+val for the final model.
        X_all_s = self._fit_scaler(pd.concat([X_tr, X_vl]).to_numpy())
        y_all = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        final = SkRidge(alpha=best.alpha, fit_intercept=True).fit(X_all_s, y_all)
        self.model_ = final
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._apply_scaler(X.to_numpy())
        return self.model_.predict(Xs)


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------

class LassoForecaster(Forecaster, _ScaledLinearMixin):
    name = "LA"

    def __init__(self, alpha_grid: Sequence[float], val_frac: float = 0.10 / 0.80):
        self.alpha_grid = list(alpha_grid)
        self.val_frac = val_frac
        self.model_: SkLasso | None = None
        self.diagnostics: _RegResult | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "LassoForecaster":
        self._feature_names = list(X.columns)
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val
        X_tr_s = self._fit_scaler(X_tr.to_numpy())
        X_vl_s = self._apply_scaler(X_vl.to_numpy())

        best = _RegResult(alpha=self.alpha_grid[0])
        for alpha in self.alpha_grid:
            m = SkLasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
            m.fit(X_tr_s, y_tr.to_numpy())
            mse = _validation_mse(m, X_vl_s, y_vl.to_numpy())
            if mse < best.val_mse:
                best = _RegResult(alpha=alpha, n_nonzero=int(np.sum(m.coef_ != 0)), val_mse=mse)

        X_all_s = self._fit_scaler(pd.concat([X_tr, X_vl]).to_numpy())
        y_all = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        self.model_ = SkLasso(alpha=best.alpha, fit_intercept=True, max_iter=10_000)
        self.model_.fit(X_all_s, y_all)
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._apply_scaler(X.to_numpy()))


# ---------------------------------------------------------------------------
# Elastic Net
# ---------------------------------------------------------------------------

class ElasticNetForecaster(Forecaster, _ScaledLinearMixin):
    name = "EN"

    def __init__(self, alpha_grid: Sequence[float], l1_ratio_grid: Sequence[float],
                 val_frac: float = 0.10 / 0.80):
        self.alpha_grid = list(alpha_grid)
        self.l1_ratio_grid = list(l1_ratio_grid)
        self.val_frac = val_frac
        self.model_: SkElasticNet | None = None
        self.diagnostics: _RegResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "ElasticNetForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val
        X_tr_s = self._fit_scaler(X_tr.to_numpy())
        X_vl_s = self._apply_scaler(X_vl.to_numpy())

        best = _RegResult(alpha=self.alpha_grid[0], l1_ratio=self.l1_ratio_grid[0])
        for alpha in self.alpha_grid:
            for l1 in self.l1_ratio_grid:
                m = SkElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=True, max_iter=10_000)
                m.fit(X_tr_s, y_tr.to_numpy())
                mse = _validation_mse(m, X_vl_s, y_vl.to_numpy())
                if mse < best.val_mse:
                    best = _RegResult(alpha=alpha, l1_ratio=l1,
                                      n_nonzero=int(np.sum(m.coef_ != 0)), val_mse=mse)

        X_all_s = self._fit_scaler(pd.concat([X_tr, X_vl]).to_numpy())
        y_all = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        self.model_ = SkElasticNet(alpha=best.alpha, l1_ratio=best.l1_ratio,
                                   fit_intercept=True, max_iter=10_000).fit(X_all_s, y_all)
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._apply_scaler(X.to_numpy()))


# ---------------------------------------------------------------------------
# Post-Lasso
# ---------------------------------------------------------------------------

class PostLassoForecaster(Forecaster, _ScaledLinearMixin):
    """Two-stage: Lasso selects variables, OLS is fit on the selected subset."""

    name = "P-LA"

    def __init__(self, alpha_grid: Sequence[float], val_frac: float = 0.10 / 0.80):
        self.alpha_grid = list(alpha_grid)
        self.val_frac = val_frac
        self.selected_: list[int] | None = None
        self.ols_: LinearRegression | None = None
        self.diagnostics: _RegResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "PostLassoForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val

        X_tr_s = self._fit_scaler(X_tr.to_numpy())
        X_vl_s = self._apply_scaler(X_vl.to_numpy())

        best = _RegResult(alpha=self.alpha_grid[0])
        best_selected: list[int] = []
        for alpha in self.alpha_grid:
            m = SkLasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
            m.fit(X_tr_s, y_tr.to_numpy())
            selected = [i for i, c in enumerate(m.coef_) if c != 0.0]
            if not selected:
                continue
            ols = LinearRegression(fit_intercept=True)
            ols.fit(X_tr_s[:, selected], y_tr.to_numpy())
            pred = ols.predict(X_vl_s[:, selected])
            mse = float(np.mean((pred - y_vl.to_numpy()) ** 2))
            if mse < best.val_mse:
                best = _RegResult(alpha=alpha, n_nonzero=len(selected), val_mse=mse)
                best_selected = selected

        X_all_s = self._fit_scaler(pd.concat([X_tr, X_vl]).to_numpy())
        y_all = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        if not best_selected:
            best_selected = list(range(X.shape[1]))
        self.selected_ = best_selected
        self.ols_ = LinearRegression(fit_intercept=True).fit(X_all_s[:, best_selected], y_all)
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._apply_scaler(X.to_numpy())
        return self.ols_.predict(Xs[:, self.selected_])


# ---------------------------------------------------------------------------
# Adaptive Lasso (Zou 2006)
# ---------------------------------------------------------------------------

class AdaptiveLassoForecaster(Forecaster, _ScaledLinearMixin):
    """First-stage OLS reweights penalties; second stage refits via Lasso.

    Adaptive Lasso solves the rescaled problem

    .. math::

        \\min_\\beta \\| y - X\\beta \\|^2 + \\lambda \\sum_i w_i |\\beta_i|,
        \\quad w_i = 1 / |\\hat\\beta_i^{(\\mathrm{OLS})}|^\\gamma

    by absorbing the weights into the design matrix:
    :math:`\\tilde X_i = X_i / w_i`, fitting ordinary Lasso, and dividing the
    coefficients by :math:`w_i`.
    """

    name = "A-LA"

    def __init__(self, gamma: float = 1.0, alpha_grid: Sequence[float] | None = None,
                 val_frac: float = 0.10 / 0.80):
        self.gamma = gamma
        self.alpha_grid = list(alpha_grid) if alpha_grid else [1e-3, 1e-2, 1e-1, 1.0]
        self.val_frac = val_frac
        self.weights_: np.ndarray | None = None
        self.model_: SkLasso | None = None
        self.diagnostics: _RegResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "AdaptiveLassoForecaster":
        if X_val is None or y_val is None:
            split = int(len(X) * (1 - self.val_frac))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        else:
            X_tr, y_tr = X, y
            X_vl, y_vl = X_val, y_val
        X_tr_s = self._fit_scaler(X_tr.to_numpy())
        X_vl_s = self._apply_scaler(X_vl.to_numpy())

        ols = LinearRegression(fit_intercept=True).fit(X_tr_s, y_tr.to_numpy())
        w = 1.0 / (np.abs(ols.coef_) ** self.gamma + 1e-12)
        self.weights_ = w
        X_tr_rs = X_tr_s / w
        X_vl_rs = X_vl_s / w

        best = _RegResult(alpha=self.alpha_grid[0])
        best_model: SkLasso | None = None
        for alpha in self.alpha_grid:
            m = SkLasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
            m.fit(X_tr_rs, y_tr.to_numpy())
            mse = _validation_mse(m, X_vl_rs, y_vl.to_numpy())
            if mse < best.val_mse:
                best = _RegResult(alpha=alpha, n_nonzero=int(np.sum(m.coef_ != 0)), val_mse=mse)
                best_model = m

        X_all = pd.concat([X_tr, X_vl])
        y_all = np.concatenate([y_tr.to_numpy(), y_vl.to_numpy()])
        X_all_s = self._fit_scaler(X_all.to_numpy())
        ols_all = LinearRegression(fit_intercept=True).fit(X_all_s, y_all)
        self.weights_ = 1.0 / (np.abs(ols_all.coef_) ** self.gamma + 1e-12)
        X_all_rs = X_all_s / self.weights_
        self.model_ = SkLasso(alpha=best.alpha, fit_intercept=True, max_iter=10_000).fit(X_all_rs, y_all)
        self.diagnostics = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._apply_scaler(X.to_numpy())
        return self.model_.predict(Xs / self.weights_)
