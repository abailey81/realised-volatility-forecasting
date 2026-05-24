"""
Unit tests for the realised-variance computation.

These tests verify the mathematical correctness of the RV pipeline on
synthetic data with known properties. They do not exercise the live
network or real tick files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.compute_rv import (
    resample_to_frequency,
    _daily_realised,
    _daily_return,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_minute_bars() -> pd.DataFrame:
    """Two trading days of minute bars with a known constant log return.

    Constructing 390-minute days where each minute log-return equals
    ``r0 = 1e-4`` gives an analytic RV value:

        5-min log-return = 5 * r0
        n returns        = 390 / 5 = 78
        RV (per day)     = 78 * (5 * r0)^2
    """
    rng = pd.date_range("2020-01-02 09:30", periods=390, freq="1min")
    rng2 = pd.date_range("2020-01-03 09:30", periods=390, freq="1min")
    full_idx = rng.append(rng2)
    log_close = np.cumsum(np.ones(len(full_idx)) * 1e-4)
    close = np.exp(log_close)
    df = pd.DataFrame({
        "Open": close, "High": close, "Low": close,
        "Close": close, "Volume": 100.0,
    }, index=full_idx)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resample_to_5min_produces_78_returns(synthetic_minute_bars):
    intra = resample_to_frequency(synthetic_minute_bars, minutes=5)
    per_day = intra.groupby(intra.index.normalize())["log_return"].count()
    # Each day should have 78 5-min log returns.
    assert (per_day == 78).all(), f"Expected 78 returns/day, got {per_day.tolist()}"


def test_rv_matches_analytic_value(synthetic_minute_bars):
    """RV should sum to the analytic expectation under the chosen resampler.

    The synthetic day has 390 minute bars from 09:30 to 15:59 inclusive.
    Right-closed 5-min resampling produces bars at 09:30, 09:35, ..., 16:00
    (79 bars, 78 returns). The first 77 returns each span exactly 5 minutes
    of accumulation (Δ = 5 r0); the last return spans only 4 minutes
    (Δ = 4 r0) because the minute-bar file ends at 15:59. The expected RV
    is therefore::

        RV = 77 (5 r0)^2 + (4 r0)^2
    """
    intra = resample_to_frequency(synthetic_minute_bars, minutes=5)
    daily = _daily_realised(intra)
    r0 = 1e-4
    expected = 77 * (5 * r0) ** 2 + (4 * r0) ** 2
    assert np.allclose(daily["RV"].values, expected, rtol=1e-8), (
        f"Expected RV={expected:.3e}, got {daily['RV'].values}"
    )


def test_rv_pos_neg_sum_to_rv(synthetic_minute_bars):
    """RV+ + RV- should equal RV for any return series."""
    intra = resample_to_frequency(synthetic_minute_bars, minutes=5)
    daily = _daily_realised(intra)
    assert np.allclose(daily["RV_pos"] + daily["RV_neg"], daily["RV"], rtol=1e-12)


def test_no_overnight_returns(synthetic_minute_bars):
    """The resampler must not create returns across day boundaries."""
    intra = resample_to_frequency(synthetic_minute_bars, minutes=5)
    per_day = intra.groupby(intra.index.normalize())["log_return"].count()
    # Total returns = sum over days; if overnight were included, total > 78*ndays.
    assert per_day.sum() == 78 * 2


def test_daily_return_open_to_close(synthetic_minute_bars):
    intra = resample_to_frequency(synthetic_minute_bars, minutes=5)
    rets = _daily_return(intra)
    # Each day the price grows by exp(389 * r0); resampled prices preserve this.
    # We sample at end of 5-min intervals, so first bar is at 09:35 and last at 16:00.
    # Open-to-close difference equals the sum of 5-min log returns.
    intraday_sum = (
        intra.groupby(intra.index.normalize())["log_return"]
             .sum()
    )
    assert np.allclose(rets.values, intraday_sum.values, rtol=1e-12)
