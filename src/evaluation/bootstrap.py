"""
Moving-block bootstrap confidence intervals for forecast losses.

For weakly-dependent series the moving-block bootstrap (MBB; Künsch 1989;
Liu & Singh 1992) preserves short-range serial dependence in the data.
We use it to construct percentile confidence intervals for:

* Mean MSE / QLIKE of each model on the test set.
* MSE differences between pairs of models (a complement to the DM test:
  CIs answer "by how much?" while DM answers "is it different?").

The bootstrap block length defaults to :math:`n^{1/3}` rounded to an
integer, which is the rate-optimal choice for the variance of the sample
mean under standard mixing conditions (Politis-White 2004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    alpha: float
    n_bootstrap: int
    block_length: int


def _block_indices(n: int, block_length: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Length-n moving-block bootstrap indices."""
    num_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=num_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def bootstrap_loss_ci(
    loss_series: np.ndarray,
    alpha: float = 0.05,
    num_bootstrap: int = 5000,
    block_length: int | None = None,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of a loss series.

    Vectorised over bootstrap replications; index draws happen in one
    ``rng.integers`` call, and bootstrap means are computed in batched
    numpy with bounded memory.
    """
    loss = np.asarray(loss_series, dtype=float)
    n = len(loss)
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    num_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    # Draw all bootstrap start indices in one shot: (B, num_blocks)
    starts = rng.integers(0, n - block_length + 1, size=(num_bootstrap, num_blocks))
    # Build index matrix (B, n); broadcast then truncate to n
    idx_full = (starts[:, :, None] + offsets[None, None, :]).reshape(num_bootstrap, -1)[:, :n]
    boot_means = loss[idx_full].mean(axis=1)
    estimate = float(loss.mean())
    se = float(np.std(boot_means, ddof=1))
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return BootstrapCI(estimate=estimate, se=se, ci_low=low, ci_high=high,
                       alpha=alpha, n_bootstrap=num_bootstrap,
                       block_length=block_length)


def bootstrap_diff_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    alpha: float = 0.05,
    num_bootstrap: int = 5000,
    block_length: int | None = None,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of (loss_a - loss_b)."""
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    return bootstrap_loss_ci(diff, alpha=alpha,
                              num_bootstrap=num_bootstrap,
                              block_length=block_length, seed=seed)
