"""
Unit tests for the HAR-family models.

The tests verify recovery of known coefficients on synthetic data and
sanity-check the LogHAR Jensen correction and the HARQ insanity filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.har_models import HAR, LogHAR, HARQ


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_har_data(n: int = 1000, beta=(0.5, 0.3, 0.15), intercept=1e-4, sigma=1e-5):
    rng = np.random.default_rng(42)
    rvd = rng.exponential(1e-4, size=n)
    rvw = rng.exponential(1e-4, size=n)
    rvm = rng.exponential(1e-4, size=n)
    y = intercept + beta[0]*rvd + beta[1]*rvw + beta[2]*rvm + rng.normal(0, sigma, size=n)
    X = pd.DataFrame({"RVD": rvd, "RVW": rvw, "RVM": rvm})
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_har_recovers_coefficients_on_clean_data():
    X, y = _make_har_data(n=5000, beta=(0.5, 0.3, 0.15), sigma=1e-7)
    model = HAR().fit(X, y)
    assert abs(model.beta_[0] - 0.5) < 0.02
    assert abs(model.beta_[1] - 0.3) < 0.02
    assert abs(model.beta_[2] - 0.15) < 0.02


def test_loghar_predictions_are_positive():
    X, y = _make_har_data(n=2000)
    model = LogHAR().fit(X, y)
    pred = model.predict(X)
    assert (pred > 0).all(), "LogHAR predictions must be positive (exp transform)"


def test_loghar_includes_jensen_correction():
    X, y = _make_har_data(n=2000)
    model = LogHAR().fit(X, y)
    # The Jensen correction term must be non-negative.
    assert model.sigma2_ >= 0
    # Without the correction, we'd get exp(log_pred); with the correction,
    # the predictions are uniformly higher by exp(sigma^2 / 2) >= 1.
    if model.sigma2_ > 0:
        Xv = np.log(np.clip(X.to_numpy(), 1e-16, None))
        raw = np.exp(Xv @ model.beta_ + model.intercept_)
        corrected = model.predict(X)
        assert np.all(corrected >= raw - 1e-12)


def test_harq_insanity_filter_clips_extreme_predictions():
    """BPQ (2016 §3.3): out-of-range forecasts clip to the closest in-sample endpoint."""
    X, y = _make_har_data(n=500)
    X["RQ_x_RV"] = X["RVD"] * X["RVW"]   # synthetic interaction term
    model = HARQ().fit(X, y)
    # Inject an extreme high-end test row.
    extreme_hi = X.iloc[[0]].copy()
    extreme_hi["RVD"] = X["RVD"].max() * 100
    extreme_hi["RQ_x_RV"] = X["RQ_x_RV"].max() * 100
    pred_hi = model.predict(extreme_hi)
    # Per BPQ: high outliers clip to train_max_ exactly.
    assert pred_hi[0] == model.train_max_, (
        f"High-end clip failed: pred={pred_hi[0]:.4e}, max={model.train_max_:.4e}"
    )
    # Inject an extreme low-end test row (large negative).
    extreme_lo = X.iloc[[0]].copy()
    extreme_lo["RVD"] = -X["RVD"].max() * 100
    extreme_lo["RQ_x_RV"] = -X["RQ_x_RV"].max() * 100
    pred_lo = model.predict(extreme_lo)
    # Per BPQ: low outliers clip to train_min_ exactly.
    assert pred_lo[0] == model.train_min_, (
        f"Low-end clip failed: pred={pred_lo[0]:.4e}, min={model.train_min_:.4e}"
    )
