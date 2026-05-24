"""
Model Confidence Set procedure (Hansen, Lunde & Nason 2011).

Given a set of competing forecasts and a loss series for each, the MCS
identifies a subset :math:`\\mathcal{M}^* \\subseteq \\mathcal{M}_0` such that
the *best* model is contained in :math:`\\mathcal{M}^*` with confidence
:math:`1 - \\alpha`. The procedure iteratively removes the model with the
worst relative performance using a moving-block bootstrap to assess the
null hypothesis of equal predictive ability across the remaining set.

We implement the :math:`T_{\\max}` statistic:

.. math::

    T_{\\max,\\mathcal{M}} = \\max_{i \\in \\mathcal{M}} \\,
        \\frac{\\bar d_{i\\cdot}}{\\sqrt{\\widehat{\\mathrm{Var}}(\\bar d_{i\\cdot})}}

where :math:`\\bar d_{i\\cdot} = \\bar L_i - \\bar L_{\\cdot}` and the
denominator is bootstrapped.

The bootstrap uses moving blocks with length :math:`\\ell` chosen via the
Politis-White (2004) automatic selection rule (defaults to
:math:`\\ell = n^{1/3}`).

Reference:
    Hansen, P. R., A. Lunde, and J. M. Nason (2011).
    "The Model Confidence Set". Econometrica 79(2), 453-497.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import LOSSES


@dataclass
class MCSResult:
    surviving_models: list[str]
    p_values: dict[str, float]
    eliminated_order: list[tuple[str, float]]
    statistic: str
    alpha: float
    n_bootstrap: int
    block_length: int


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------

def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Return a length-``n`` index sequence drawn by moving-block bootstrap."""
    if block_length <= 0 or block_length > n:
        block_length = max(1, int(round(n ** (1 / 3))))
    num_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=num_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


# ---------------------------------------------------------------------------
# Core MCS procedure
# ---------------------------------------------------------------------------

def model_confidence_set(
    y_true: np.ndarray,
    forecasts: dict[str, np.ndarray],
    loss: str = "mse",
    alpha: float = 0.10,
    num_bootstrap: int = 5000,
    block_length: int | None = None,
    statistic: str = "Tmax",
    seed: int = 42,
) -> MCSResult:
    """Run the Hansen-Lunde-Nason MCS procedure.

    Parameters
    ----------
    y_true
        Observed target series.
    forecasts
        Dict mapping model name to its array of out-of-sample predictions.
    loss
        Loss name from ``metrics.LOSSES`` (``mse``, ``mae``, ``qlike``).
    alpha
        Significance level. The surviving set has confidence ``1 - alpha``.
    num_bootstrap
        Number of bootstrap replications (paper uses 10,000; default 5,000).
    block_length
        Moving-block bootstrap block length. ``None`` => :math:`n^{1/3}`.
    statistic
        ``Tmax`` (max-t studentised statistic) — the recommended variant.

    Returns
    -------
    :class:`MCSResult` with the surviving model set and per-model
    elimination p-values.
    """
    if statistic != "Tmax":
        raise NotImplementedError("Only the Tmax statistic is implemented")
    if loss not in LOSSES:
        raise KeyError(f"Unknown loss '{loss}'")
    loss_fn = LOSSES[loss]

    labels = list(forecasts.keys())
    L = np.stack([loss_fn(y_true, forecasts[lab]) for lab in labels], axis=1)
    n, k0 = L.shape
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))

    rng = np.random.default_rng(seed)
    # Pre-draw a single bootstrap index matrix.
    boot_idx = np.stack(
        [_moving_block_indices(n, block_length, rng) for _ in range(num_bootstrap)],
        axis=0,
    )

    surviving = list(range(k0))
    eliminated: list[tuple[str, float]] = []
    p_values: dict[str, float] = {}

    while len(surviving) > 1:
        sub_L = L[:, surviving]
        ks = sub_L.shape[1]
        Lbar = sub_L.mean(axis=0)
        # Pairwise mean differences and bootstrap distribution.
        # Following HLN, we use d_{i.} = Lbar_i - mean(Lbar) for Tmax.
        d_i = Lbar - Lbar.mean()
        # Vectorised bootstrap: compute B sample means in one shot.
        # Process in chunks to bound memory at <100 MB per chunk.
        chunk = max(1, int(2_000_000 / max(n * ks, 1)))   # ≈ 16 MB chunks
        boot_d = np.empty((num_bootstrap, ks))
        for start in range(0, num_bootstrap, chunk):
            stop = min(start + chunk, num_bootstrap)
            # boot_idx[start:stop] is (chunk, n); index sub_L (n, ks)
            # → (chunk, n, ks); mean over axis=1 → (chunk, ks)
            mb = sub_L[boot_idx[start:stop]].mean(axis=1)
            boot_d[start:stop] = mb - mb.mean(axis=1, keepdims=True)
        # Studentise by the bootstrap standard deviation of d_i.
        sd = boot_d.std(axis=0, ddof=1)
        sd[sd == 0] = 1e-12
        t_stat = d_i / sd
        boot_t = (boot_d - d_i) / sd
        Tmax_obs = t_stat.max()
        Tmax_boot = boot_t.max(axis=1)
        pval = float(np.mean(Tmax_boot >= Tmax_obs))

        worst_idx_local = int(np.argmax(t_stat))
        worst_idx_global = surviving[worst_idx_local]

        # Cumulative MCS p-value for the eliminated model.
        cum_pval = max(pval, eliminated[-1][1] if eliminated else 0.0)
        p_values[labels[worst_idx_global]] = cum_pval

        if cum_pval > alpha:
            # Surviving set is identified; remaining models share the survival p-value.
            for idx in surviving:
                if labels[idx] not in p_values:
                    p_values[labels[idx]] = cum_pval
            break

        eliminated.append((labels[worst_idx_global], cum_pval))
        surviving.remove(worst_idx_global)

    if len(surviving) == 1:
        p_values[labels[surviving[0]]] = 1.0

    return MCSResult(
        surviving_models=[labels[i] for i in surviving],
        p_values=p_values,
        eliminated_order=eliminated,
        statistic=statistic,
        alpha=alpha,
        n_bootstrap=num_bootstrap,
        block_length=block_length,
    )
