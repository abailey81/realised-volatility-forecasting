"""
Rolling-window forecasting harness.

This module provides:

* :func:`fixed_window_forecast` — train once on training+validation, predict
  on the full test set. The paper uses this scheme for NNs ("the weights are
  only found once in the initial validation sample and not rolled forward").

* :func:`rolling_window_forecast` — refit the model at each step (or every
  ``refit_frequency_days``) using a sliding training window. Used for
  HAR-family OLS and optionally for the regularised methods.

Both return a Series of test-set predictions aligned to the test dates
along with the realised target series and a small ``info`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..models.base import Forecaster
from ..utils import get_logger

_LOG = get_logger(__name__)


@dataclass
class ForecastOutput:
    predictions: pd.Series
    y_true: pd.Series
    info: dict


# ---------------------------------------------------------------------------
# Fixed-window forecast (train once, predict test)
# ---------------------------------------------------------------------------

def fixed_window_forecast(
    model_factory: Callable[[], Forecaster],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "y",
    pass_val_to_fit: bool = False,
) -> ForecastOutput:
    """Train once on train+val and predict the entire test block.

    Parameters
    ----------
    model_factory
        Zero-argument callable returning a fresh :class:`Forecaster`.
    train, val, test
        DataFrames each containing the feature columns and ``target_col``.
    target_col
        Column name of the regression target.
    pass_val_to_fit
        If True, call ``model.fit(X_train, y_train, X_val, y_val)`` so the
        model can do its own validation-set hyperparameter selection.
        Otherwise concatenate train+val and call ``model.fit(X, y)``.
    """
    feature_cols = [c for c in train.columns if c != target_col]
    if pass_val_to_fit:
        model = model_factory()
        model.fit(train[feature_cols], train[target_col],
                  X_val=val[feature_cols], y_val=val[target_col])
    else:
        full = pd.concat([train, val])
        model = model_factory()
        model.fit(full[feature_cols], full[target_col])
    preds = model.predict(test[feature_cols])
    s = pd.Series(preds, index=test.index, name=getattr(model, "name", "model"))
    return ForecastOutput(
        predictions=s,
        y_true=test[target_col],
        info={"scheme": "fixed", "n_train": len(train), "n_val": len(val), "n_test": len(test)},
    )


# ---------------------------------------------------------------------------
# Rolling-window forecast (refit periodically)
# ---------------------------------------------------------------------------

def rolling_window_forecast(
    model_factory: Callable[[], Forecaster],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "y",
    refit_frequency: int = 1,
    expanding: bool = False,
    progress: bool = True,
) -> ForecastOutput:
    """Refit the model every ``refit_frequency`` test days.

    The training window for prediction at test-day :math:`t` is:

    * if ``expanding=False``: a sliding window of fixed length equal to
      ``len(train) + len(val)``, ending at :math:`t-1`,
    * if ``expanding=True``: all data from the start through :math:`t-1`.

    Hyperparameter tuning for regularised / tree-based models happens once
    at the start using the original (train, val) split; the *form* of the
    hyperparameters is then frozen for the rolling window. This matches the
    paper's "rolling scheme without concatenation" for RR/LA/EN/GB.
    """
    feature_cols = [c for c in train.columns if c != target_col]
    full_history = pd.concat([train, val, test])
    n_test = len(test)
    test_start = len(train) + len(val)

    initial_window_size = test_start
    preds = np.full(n_test, np.nan)
    model: Forecaster | None = None

    it = range(n_test)
    if progress:
        it = tqdm(it, desc="Rolling forecast", leave=False)

    for k in it:
        global_idx = test_start + k
        if k % refit_frequency == 0:
            if expanding:
                window = full_history.iloc[:global_idx]
            else:
                window = full_history.iloc[global_idx - initial_window_size: global_idx]
            X_win = window[feature_cols]
            y_win = window[target_col]
            model = model_factory()
            model.fit(X_win, y_win)
        # Predict the single next observation.
        Xk = full_history.iloc[[global_idx]][feature_cols]
        preds[k] = float(model.predict(Xk)[0])  # type: ignore[union-attr]

    s = pd.Series(preds, index=test.index, name=getattr(model, "name", "model"))
    return ForecastOutput(
        predictions=s,
        y_true=test[target_col],
        info={
            "scheme": "rolling",
            "expanding": expanding,
            "refit_frequency": refit_frequency,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    )
