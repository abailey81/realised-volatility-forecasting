"""
Decile-stratified out-of-sample loss analysis.

Replicates Christensen, Siggaard, Veliyev (2023) Figure 5: split the test
set into deciles of *observed* realised variance, then compute relative
MSE per model per decile. The paper finding is that ML's gains over HAR
are concentrated in the highest-volatility deciles — a finding that is
empirically and economically meaningful (risk-management value is highest
in turbulent periods).

The decile boundaries are computed on the test-set realised values,
producing 10 equal-count groups. Per group:

* ``relative_mse[model] = mean(MSE_model in group) / mean(MSE_HAR in group)``

which is the same per-decile relative measure plotted in Figure 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecileLoss:
    decile_edges: np.ndarray            # 11 values defining 10 bins
    decile_index: np.ndarray            # bin assignment per observation
    losses: pd.DataFrame                # rows=deciles 1..10, cols=models
    counts: np.ndarray                  # observations per decile


def decile_losses(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    loss_fn,
    n_deciles: int = 10,
) -> DecileLoss:
    """Compute mean loss per model in each decile of observed ``y_true``.

    The deciles partition the test set by realised RV. Decile 1 holds the
    smallest 10% of RV values, decile 10 the largest 10%.
    """
    if y_true.empty:
        raise ValueError("y_true is empty")

    common = y_true.index
    aligned_preds: dict[str, np.ndarray] = {}
    for label, p in predictions.items():
        idx = common.intersection(p.index)
        if len(idx) != len(common):
            common = idx
        aligned_preds[label] = p.loc[common].to_numpy()
    y_arr = y_true.loc[common].to_numpy()

    # Quantile-based decile edges.
    quantiles = np.linspace(0, 1, n_deciles + 1)
    edges = np.quantile(y_arr, quantiles)
    edges[-1] = edges[-1] + 1e-12       # ensure max value lands in last bin
    decile_idx = np.digitize(y_arr, edges[1:-1], right=False)  # 0..n-1

    rows = []
    counts = np.zeros(n_deciles, dtype=int)
    for d in range(n_deciles):
        mask = decile_idx == d
        counts[d] = int(mask.sum())
        if counts[d] == 0:
            row = {label: np.nan for label in predictions}
        else:
            row = {}
            for label, pred_arr in aligned_preds.items():
                row[label] = float(np.mean(loss_fn(y_arr[mask], pred_arr[mask])))
        rows.append(row)
    losses = pd.DataFrame(rows, index=[f"D{i+1}" for i in range(n_deciles)])
    return DecileLoss(
        decile_edges=edges,
        decile_index=decile_idx,
        losses=losses,
        counts=counts,
    )


def relative_decile_losses(
    decile_result: DecileLoss,
    baseline: str = "HAR",
) -> pd.DataFrame:
    """Convert absolute decile losses into ratios vs ``baseline``."""
    if baseline not in decile_result.losses.columns:
        raise KeyError(f"baseline '{baseline}' missing")
    return decile_result.losses.div(decile_result.losses[baseline], axis=0)
