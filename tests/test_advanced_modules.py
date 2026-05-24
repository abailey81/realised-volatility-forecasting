"""
Unit tests for the advanced extension modules:

* Mincer-Zarnowitz forecast efficiency tests
* Forecast combinations (simple average, MSE-weighted, Bates-Granger)
* Bootstrap confidence intervals
* Parzen-kernel realised kernel components
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.realised_kernel import parzen_kernel, _optimal_bandwidth
from src.evaluation.mincer_zarnowitz import mincer_zarnowitz
from src.evaluation.bootstrap import bootstrap_loss_ci, bootstrap_diff_ci
from src.models.combinations import simple_average, mse_weighted, bates_granger


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------

def test_mz_perfect_forecast_has_alpha0_beta1():
    """A perfect forecast should yield α ≈ 0, β ≈ 1, and a high p-value."""
    rng = np.random.default_rng(0)
    y = rng.exponential(1e-4, size=500)
    p = y.copy()  # perfect
    r = mincer_zarnowitz(y, p)
    assert abs(r.alpha) < 1e-10
    assert abs(r.beta - 1.0) < 1e-10
    assert r.r_squared > 0.999


def test_mz_biased_forecast_fails_joint_test():
    """A forecast that's systematically too low should be rejected."""
    rng = np.random.default_rng(0)
    y = rng.exponential(1e-4, size=500)
    p = 0.5 * y + rng.normal(0, 1e-6, size=500)
    r = mincer_zarnowitz(y, p)
    # Bias is large; joint test should reject at any reasonable level.
    assert r.wald_pvalue < 0.01


def test_mz_theil_decomposition_matches_mse():
    """The Theil components should sum to MSE up to numerical precision."""
    rng = np.random.default_rng(0)
    y = rng.exponential(1e-4, size=400)
    p = y + rng.normal(0, 2e-5, size=400)
    r = mincer_zarnowitz(y, p)
    mse = float(np.mean((y - p) ** 2))
    total = r.bias_component + r.regression_component + r.disturbance_component
    assert abs(total - mse) / mse < 0.01, f"Theil decomposition off: {total} vs MSE {mse}"


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------

def test_simple_average_uses_equal_weights():
    forecasts = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([3.0, 4.0, 5.0])}
    pred, info = simple_average(forecasts)
    assert np.allclose(pred, [2.0, 3.0, 4.0])
    assert np.allclose(info.weights, [0.5, 0.5])


def test_mse_weighted_prefers_low_mse_model():
    rng = np.random.default_rng(0)
    y_val = rng.exponential(1.0, size=200)
    val_forecasts = {
        "good": y_val + rng.normal(0, 0.1, size=200),
        "bad":  y_val + rng.normal(0, 1.0, size=200),
    }
    test_forecasts = {"good": np.zeros(50), "bad": np.zeros(50)}
    _, info = mse_weighted(y_val, val_forecasts, test_forecasts)
    # The good model should get more weight.
    weights_dict = dict(zip(info.labels, info.weights))
    assert weights_dict["good"] > weights_dict["bad"]


def test_bates_granger_weights_sum_to_one():
    rng = np.random.default_rng(0)
    y_val = rng.exponential(1.0, size=200)
    val_forecasts = {f"m{i}": y_val + rng.normal(0, 0.1 * (i + 1), size=200) for i in range(4)}
    test_forecasts = {k: np.zeros(50) for k in val_forecasts}
    _, info = bates_granger(y_val, val_forecasts, test_forecasts)
    assert abs(info.weights.sum() - 1.0) < 1e-8
    # All weights non-negative under constrain_nonneg=True (default).
    assert (info.weights >= -1e-10).all()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_sample_mean():
    rng = np.random.default_rng(0)
    losses = rng.exponential(1.0, size=500)
    ci = bootstrap_loss_ci(losses, alpha=0.05, num_bootstrap=1000)
    # Sample mean should lie inside the 95% CI.
    sample_mean = float(np.mean(losses))
    assert ci.ci_low <= sample_mean <= ci.ci_high


def test_bootstrap_diff_ci_centers_on_mean_difference():
    """With a noisy offset, the diff CI should be close to but bracket the true mean."""
    rng = np.random.default_rng(0)
    loss_a = rng.exponential(1.0, size=400)
    loss_b = loss_a + 0.5 + rng.normal(0, 0.05, size=400)  # noisy offset
    ci = bootstrap_diff_ci(loss_a, loss_b, num_bootstrap=500)
    # Estimate close to -0.5; CI should bracket it.
    assert abs(ci.estimate - (-0.5)) < 0.02
    assert ci.ci_low <= ci.estimate <= ci.ci_high
    assert ci.ci_low < ci.ci_high


# ---------------------------------------------------------------------------
# Realised kernel components
# ---------------------------------------------------------------------------

def test_parzen_kernel_endpoints():
    """k(0) = 1, k(0.5) = 0.25, k(1) = 0, k outside [0,1] = 0."""
    x = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    k = parzen_kernel(x)
    assert abs(k[0]) < 1e-12          # outside support
    assert abs(k[1] - 1.0) < 1e-12    # at 0
    assert abs(k[3] - 0.25) < 1e-12   # at 0.5
    assert abs(k[5]) < 1e-12          # at 1
    assert abs(k[6]) < 1e-12          # outside support


def test_parzen_kernel_monotone_on_each_branch():
    """k is decreasing on [0, 1]."""
    x = np.linspace(0, 1, 50)
    k = parzen_kernel(x)
    diffs = np.diff(k)
    # Monotone non-increasing (allow tiny numerical noise).
    assert (diffs <= 1e-12).all()


def test_optimal_bandwidth_is_positive_integer():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 1e-4, size=390)
    H = _optimal_bandwidth(returns)
    assert isinstance(H, int)
    assert H >= 1
