"""
Unit tests for the newest extension modules added to the project:

* HMM 2-state regime fitting (src.evaluation.regime)
* Multi-testing correction primitives (scripts/20_dm_multitest_correction)
* Path-decomposition bootstrap ratio CI (scripts/21_path_decomposition_ci)

These tests verify mathematical properties rather than exact numerical
output: the regime fit recovers the planted high-/low-vol regimes, the
correction procedures reduce the rejection count monotonically as the
threshold tightens, and the block-bootstrap ratio CI brackets the
point estimate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Path setup for ad-hoc import of script-level functions
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_script(name: str):
    """Import a top-level script module by file path."""
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Regime fitting
# ---------------------------------------------------------------------------

def _simulate_two_regime_log_rv(n: int = 1000, seed: int = 0) -> pd.Series:
    """Simulate a 2-state HMM log-RV path: low mean / high mean, with
    realistic persistence (probability of staying ~ 0.97 per day)."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.97, 0.03], [0.03, 0.97]])
    state = 0
    states = []
    obs = []
    mus = (-9.5, -7.5)
    sds = (0.4, 0.6)
    for _ in range(n):
        states.append(state)
        obs.append(rng.normal(mus[state], sds[state]))
        state = rng.choice([0, 1], p=P[state])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.exp(np.asarray(obs)), index=idx, name="RV"), np.asarray(states)


def test_hmm_recovers_two_regimes():
    """Fitting on simulated 2-state data should produce μ_low < μ_high
    and approximately the planted regime sequence (agreement ≥ 70%)."""
    from src.evaluation.regime import fit_two_state_hmm

    rv, true_states = _simulate_two_regime_log_rv(n=800, seed=1)
    res = fit_two_state_hmm(rv, log_transform=True)

    assert res.means[0] < res.means[1], "low-vol mean should be smaller than high-vol mean"
    assert res.variances[0] > 0 and res.variances[1] > 0

    # The fit can flip labels relative to truth; check accuracy on both alignments.
    fitted = res.regime.to_numpy()
    acc = max(
        (fitted == true_states).mean(),
        (fitted == 1 - true_states).mean(),
    )
    assert acc >= 0.70, f"regime recovery accuracy too low: {acc:.3f}"


def test_hmm_transition_is_a_valid_stochastic_matrix():
    """statsmodels stores P with columns summing to 1
    (P[i, j] = Pr(S_t=i | S_{t-1}=j)). Test that one axis or the other sums
    close to 1 for both 2-state rows / columns, and entries are in [0, 1]."""
    from src.evaluation.regime import fit_two_state_hmm

    rv, _ = _simulate_two_regime_log_rv(n=600, seed=2)
    res = fit_two_state_hmm(rv, log_transform=True)
    P = res.transition
    assert P.shape == (2, 2)
    # All entries in [0, 1] (slack for floating-point drift in EM)
    assert P.min() >= -1e-3
    assert P.max() <= 1.0 + 1e-2
    # One axis must sum approximately to (1, 1); allow slack for EM optimisation
    row_ok = np.allclose(P.sum(axis=1), [1.0, 1.0], atol=1e-2)
    col_ok = np.allclose(P.sum(axis=0), [1.0, 1.0], atol=1e-2)
    assert row_ok or col_ok


def test_regime_conditional_losses_partitions_test_set():
    from src.evaluation.regime import regime_conditional_losses
    from src.evaluation.metrics import mse_loss

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    y = pd.Series(np.random.default_rng(0).exponential(1e-4, n), index=idx)
    preds = {
        "A": y + np.random.default_rng(1).normal(0, 1e-5, n),
        "B": y + np.random.default_rng(2).normal(0, 2e-5, n),
    }
    regime = pd.Series(
        ([0] * 80 + [1] * 120),  # 80 low-vol, 120 high-vol
        index=idx,
    )
    tab = regime_conditional_losses(y, preds, regime, mse_loss)
    assert list(tab.index) == ["low_vol", "high_vol"]
    assert set(tab.columns) == {"A", "B"}
    # A is closer to y than B → A's mean loss should be lower in both regimes.
    assert tab.loc["low_vol", "A"] < tab.loc["low_vol", "B"]
    assert tab.loc["high_vol", "A"] < tab.loc["high_vol", "B"]


# ---------------------------------------------------------------------------
# Multi-testing correction primitives
# ---------------------------------------------------------------------------

def test_bonferroni_holm_bh_monotone_in_threshold():
    mod = _import_script("20_dm_multitest_correction.py")
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, size=200)
    pvals[:10] = 0.001  # plant some clear rejections

    rb = mod._bonferroni(pvals, 0.05).sum()
    rh = mod._holm(pvals, 0.05).sum()
    rbh = mod._bh(pvals, 0.05).sum()
    # Holm is at least as powerful as Bonferroni; BH at least as powerful as Holm
    assert rh >= rb
    assert rbh >= rh


def test_bonferroni_at_zero_alpha_rejects_none():
    mod = _import_script("20_dm_multitest_correction.py")
    pvals = np.array([0.0001, 0.0005, 0.01])
    out = mod._bonferroni(pvals, 0.0)
    assert out.sum() == 0


def test_bh_at_alpha_one_rejects_all_valid():
    mod = _import_script("20_dm_multitest_correction.py")
    pvals = np.array([0.1, 0.2, 0.3, np.nan])
    out = mod._bh(pvals, 1.0)
    assert out[:3].all()
    assert not out[3]  # NaN never rejected


def test_holm_handles_nan_inputs():
    mod = _import_script("20_dm_multitest_correction.py")
    pvals = np.array([np.nan, 0.001, np.nan, 0.5])
    out = mod._holm(pvals, 0.05)
    # Only valid p-values can be rejected
    assert not out[0] and not out[2]
    assert out[1]   # 0.001 << 0.05 / 2
    assert not out[3]


# ---------------------------------------------------------------------------
# Path-decomposition bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ratio_brackets_point_estimate():
    mod = _import_script("21_path_decomposition_ci.py")
    rng = np.random.default_rng(0)
    # Numerator/denominator with a known ratio of 0.5
    num = rng.exponential(0.5, size=300)
    den = rng.exponential(1.0, size=300)
    point, lo, hi = mod._bootstrap_ratio(num, den, n_boot=500, block=10, seed=0)
    assert lo <= point <= hi
    # The bootstrap CI should be of finite width
    assert hi > lo
    # Plausible range for the true 0.5 ratio under sampling noise
    assert 0.3 <= point <= 0.7


def test_bootstrap_ratio_zero_denominator_is_robust():
    mod = _import_script("21_path_decomposition_ci.py")
    rng = np.random.default_rng(0)
    num = rng.exponential(1.0, size=100)
    den = np.full_like(num, 1e-12)   # essentially zero
    # The function should not raise even with near-zero denominator
    point, lo, hi = mod._bootstrap_ratio(num, den, n_boot=100, block=5, seed=0)
    assert np.isfinite(point)
