"""
Accumulated Local Effects (Apley & Zhu 2020) for ML variable importance.

ALE measures the *local* impact of a feature on the model's prediction,
averaged over the joint distribution of the features. Unlike Partial
Dependence Plots (PDP), ALE is unbiased when features are correlated —
which is the empirically relevant case for realised-variance predictors
(RVD, RVW, RVM are highly correlated by construction).

For a model :math:`\\hat f` and feature :math:`X_j`, divide the support of
:math:`X_j` into :math:`K` quantile-based bins. Within bin :math:`k`,
average the *local effect*:

.. math::

    \\hat f^{1,\\text{loc}}_j(k) = \\frac{1}{|S_k|}
        \\sum_{i \\in S_k} \\bigl( \\hat f(z_{j,k}, X_{\\setminus j, i}) -
        \\hat f(z_{j,k-1}, X_{\\setminus j, i}) \\bigr)

where :math:`S_k` is the set of observations falling in bin :math:`k`, and
:math:`z_{j,k}` is the bin edge. The ALE plot is the cumulative sum of the
local effects, centred so its mean over the training distribution is zero.

Reference:
    Apley, D. W., and J. Zhu (2020). "Visualizing the Effects of Predictor
    Variables in Black Box Supervised Learning Models". JRSS-B, 82(4), 1059-1086.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ALEResult:
    feature: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    ale: np.ndarray
    counts: np.ndarray


def variable_importance_from_ale(ale_results: dict[str, "ALEResult"]) -> pd.Series:
    """Compute paper-style variable importance from a dict of ALE results.

    Implements Christensen, Siggaard, Veliyev (2023) Section 3.2 equations
    (30)-(31). For each feature j:

    .. math::

        I(Z_j) = \\sqrt{ \\frac{1}{T_0 - 1} \\sum_{t=1}^{T_0} \\hat f^{ALE}(z_{jt})^2 }

    and normalised importance ``VI(Z_j) = I(Z_j) / Σ_k I(Z_k)`` which sums
    to 1 across the feature set.

    The denominator is approximated by the per-bin ALE values weighted by
    their bin counts (this approximates the sample mean of ALE-squared over
    the empirical distribution of feature j).
    """
    raw = {}
    for feature, res in ale_results.items():
        if res.counts.sum() == 0:
            raw[feature] = 0.0
            continue
        weights = res.counts.astype(float) / res.counts.sum()
        mean_sq = float(np.sum(res.ale ** 2 * weights))
        raw[feature] = float(np.sqrt(max(mean_sq, 0.0)))
    total = sum(raw.values()) or 1.0
    return pd.Series({k: v / total for k, v in raw.items()}, name="VI")


def accumulated_local_effects(
    predict_fn,
    X: pd.DataFrame,
    feature: str,
    num_bins: int = 40,
    centred: bool = True,
) -> ALEResult:
    """Compute one-dimensional ALE for ``feature``.

    Local effects are computed via a single vectorised prediction call
    over all bin-displaced design points stacked into one frame — much
    faster than per-bin model invocations for the same model.

    Parameters
    ----------
    predict_fn
        Callable mapping a 2-D array (or DataFrame) to predictions.
    X
        Training data on which to estimate the ALE.
    feature
        Column name of the feature to analyse.
    num_bins
        Number of quantile bins.
    centred
        If True, recentre the ALE so it integrates to zero against the
        empirical distribution of ``X[feature]``.
    """
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not in X")
    x = X[feature].to_numpy()
    n = len(x)
    # Quantile-based bin edges; deduplicate when many ties exist.
    quantiles = np.linspace(0, 1, num_bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) < 3:
        # Special-case binary or near-constant features: use the two unique values
        # as the single ALE interval. Common for the EA earnings-dummy feature.
        uniq = np.unique(x)
        if len(uniq) == 2:
            edges = np.array([uniq[0], uniq[1] + 1e-12])
        else:
            raise ValueError(f"Feature '{feature}' has too few unique values for ALE.")
    K = len(edges) - 1

    # Bin assignment of each observation in 0..K-1
    bin_idx = np.digitize(x, edges[1:-1], right=False)

    feat_idx = X.columns.get_loc(feature)
    X_np = X.to_numpy()

    # Vectorise: build one big stacked design with [..., row_lo, ..., row_hi, ...].
    counts = np.bincount(bin_idx, minlength=K).astype(int)
    nonempty = np.where(counts > 0)[0]

    lo_blocks = []
    hi_blocks = []
    sizes = []
    for k in nonempty:
        members = np.where(bin_idx == k)[0]
        X_lo = X_np[members].copy()
        X_hi = X_np[members].copy()
        X_lo[:, feat_idx] = edges[k]
        X_hi[:, feat_idx] = edges[k + 1]
        lo_blocks.append(X_lo)
        hi_blocks.append(X_hi)
        sizes.append(members.size)

    if not lo_blocks:
        local_effects = np.zeros(K)
    else:
        all_lo = np.concatenate(lo_blocks, axis=0)
        all_hi = np.concatenate(hi_blocks, axis=0)
        # Single prediction call over the full stack — amortises model-call overhead.
        df_lo = pd.DataFrame(all_lo, columns=X.columns)
        df_hi = pd.DataFrame(all_hi, columns=X.columns)
        pred_lo = predict_fn(df_lo)
        pred_hi = predict_fn(df_hi)
        diff = pred_hi - pred_lo
        local_effects = np.zeros(K)
        offset = 0
        for k, sz in zip(nonempty, sizes):
            local_effects[k] = float(np.mean(diff[offset: offset + sz]))
            offset += sz

    ale = np.cumsum(local_effects)
    if centred:
        # Centre against the empirical distribution of ``x``: subtract
        # E[ALE(x)] under the empirical marginal of x. With one ALE value
        # per bin and bin probabilities counts/Σcounts, this is the
        # bin-count-weighted mean of the ALE. Apley-Zhu (2020) eq. 13
        # specifies all K bins; an earlier trapezoidal form on K-1 inner
        # intervals dropped the last bin's weight and produced a small
        # (≈1/K) level bias.
        bin_probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        centre = float(np.sum(ale * bin_probs))
        ale = ale - centre

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return ALEResult(
        feature=feature,
        bin_edges=edges,
        bin_centers=bin_centers,
        ale=ale,
        counts=counts,
    )
