"""
Unit tests for the paper-extension modules: VI from ALE, decile losses,
and the VaR backtest (Kupiec + Christoffersen + asymmetric quantile loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.ale import ALEResult, variable_importance_from_ale
from src.evaluation.decile import decile_losses, relative_decile_losses
from src.evaluation.value_at_risk import (
    _kupiec_unconditional,
    _christoffersen_conditional,
    _quantile_loss,
    filtered_historical_simulation,
)
from src.evaluation.metrics import mse_loss


# ---------------------------------------------------------------------------
# Variable Importance
# ---------------------------------------------------------------------------

def test_vi_weights_sum_to_one():
    """VI(Z_j) must sum to exactly 1 across features (paper eq. 31)."""
    ale = {
        "RVD": ALEResult("RVD", np.arange(11), np.arange(10),
                          np.linspace(0, 0.5, 10), np.full(10, 100, dtype=int)),
        "RVW": ALEResult("RVW", np.arange(11), np.arange(10),
                          np.linspace(0, 0.2, 10), np.full(10, 100, dtype=int)),
        "EA":  ALEResult("EA",  np.arange(11), np.arange(10),
                          np.linspace(0, 0.05, 10), np.full(10, 100, dtype=int)),
    }
    vi = variable_importance_from_ale(ale)
    assert abs(vi.sum() - 1.0) < 1e-12
    assert (vi >= 0).all()


def test_vi_dominant_feature_ranks_highest():
    """The feature with the largest ALE variation should have the highest VI."""
    ale = {
        "big":   ALEResult("big",   np.arange(11), np.arange(10),
                            np.linspace(-1.0, 1.0, 10), np.full(10, 100, dtype=int)),
        "small": ALEResult("small", np.arange(11), np.arange(10),
                            np.linspace(-0.01, 0.01, 10), np.full(10, 100, dtype=int)),
    }
    vi = variable_importance_from_ale(ale)
    assert vi["big"] > vi["small"]
    assert vi["big"] > 0.95   # near-total dominance


# ---------------------------------------------------------------------------
# Decile losses
# ---------------------------------------------------------------------------

def test_decile_split_yields_correct_count():
    """A 10-decile split of 200 observations should give 10 groups of 20."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.exponential(1e-4, 200),
                   index=pd.date_range("2024-01-01", periods=200))
    preds = {"HAR": y + rng.normal(0, 1e-5, 200)}
    res = decile_losses(y, preds, mse_loss)
    assert res.losses.shape == (10, 1)
    assert (res.counts == 20).all()


def test_decile_relative_uses_har_as_baseline():
    """relative_decile_losses divides every column by HAR per decile."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.exponential(1e-4, 200),
                   index=pd.date_range("2024-01-01", periods=200))
    preds = {"HAR": y + rng.normal(0, 1e-5, 200),
             "BETTER": y + rng.normal(0, 1e-6, 200)}
    res = decile_losses(y, preds, mse_loss)
    ratio = relative_decile_losses(res, baseline="HAR")
    # HAR column = 1.0 by construction
    assert (ratio["HAR"] == 1.0).all()
    # BETTER has lower noise, so its ratio should be < 1 on average
    assert ratio["BETTER"].mean() < 1.0


# ---------------------------------------------------------------------------
# VaR (Kupiec, Christoffersen, quantile loss)
# ---------------------------------------------------------------------------

def test_kupiec_rejects_oversize_breach_rate():
    """Hit rate of 20% under H0=5% must be rejected."""
    hits = np.tile([1, 0, 0, 0, 0], 100)  # 20% hits
    lr, p = _kupiec_unconditional(hits, 0.05)
    assert p < 0.01


def test_christoffersen_detects_clustered_breaches():
    """All breaches in the first half = clustered. Should be rejected."""
    hits = np.concatenate([np.ones(20), np.zeros(180)]).astype(int)
    lr, p = _christoffersen_conditional(hits, 0.05)
    assert p < 0.05


def test_quantile_loss_is_zero_on_perfect_quantile():
    """For a known α-quantile VaR and a series drawn from the same dist,
    the quantile loss is small. Use exact quantile: loss should be near 0."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 5000)
    var = np.full_like(r, np.quantile(r, 0.05))
    # At the empirical α-quantile, the pinball loss equals
    # α * E[r - q | r > q] (1-α) — small but non-zero. Just sanity-check
    # it is below the average daily move.
    loss = _quantile_loss(r, var, 0.05)
    assert abs(loss) < 3e-3
    # The same loss with a pessimistic VaR (much lower than the quantile)
    # should be strictly larger — penalty for under-coverage.
    bad_var = var - 0.01
    loss_bad = _quantile_loss(r, bad_var, 0.05)
    assert loss_bad > loss


def test_var_backtest_returns_kupiec_and_christoffersen_pvalues():
    """End-to-end smoke test of the FHS backtest."""
    rng = np.random.default_rng(0)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n)
    rv = pd.Series(rng.exponential(1e-4, n), index=dates)
    log_ret = pd.Series(rng.normal(0, np.sqrt(rv)), index=dates)
    res = filtered_historical_simulation(
        log_ret, rv,
        train_end=dates[700],
        alpha=0.05,
    )
    assert 0 <= res.kupiec_pvalue <= 1
    assert 0 <= res.christoffersen_pvalue <= 1
    assert res.n == 299  # 1000 - 701
    # Expected hit rate close to alpha
    assert abs(res.observed_hits / res.n - 0.05) < 0.05
