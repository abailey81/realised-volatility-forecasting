"""
Figure generation for the report and appendix.

Each function returns a Matplotlib ``Figure`` so the caller controls saving
and embedding. All figures use a consistent monochrome-friendly style that
prints well in journals; colour is used only for distinguishing model
families (HAR, regularised, tree, NN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Journal-grade defaults applied to every figure this module builds, so the
# supplementary figures match the report's two headline figures (serif type,
# de-spined axes, 300 dpi).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})

_FAMILY_COLORS = {
    "HAR":   "#1b1b1b",
    "OLS":   "#1b1b1b",
    "REG":   "#1f77b4",
    "TREE":  "#2ca02c",
    "NN":    "#d62728",
}

_HAR_LABELS = {"HAR", "LogHAR", "LevHAR", "SHAR", "HARQ", "HAR-X"}
_REG_LABELS = {"RR", "LA", "EN", "P-LA", "A-LA"}
_TREE_LABELS = {"BG", "RF", "GB"}


def _family(label: str) -> str:
    if label in _HAR_LABELS:
        return "HAR"
    if label in _REG_LABELS:
        return "REG"
    if label in _TREE_LABELS:
        return "TREE"
    if label.startswith("NN") or label.endswith("_ensemble"):
        return "NN"
    return "OLS"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "axes.spines.right": False,
        "axes.spines.top":   False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    ":",
        "legend.frameon":    False,
        "legend.fontsize":   9,
    })


# ---------------------------------------------------------------------------
# Figure 1: RV time series
# ---------------------------------------------------------------------------

def plot_rv_time_series(
    rv_dict: Mapping[str, pd.Series],
    annualise: bool = True,
    title: str = "Daily realised volatility (annualised)",
) -> plt.Figure:
    """Plot annualised daily realised volatility for one or more stocks."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for ticker, rv in rv_dict.items():
        sigma = np.sqrt(rv * 252) * 100 if annualise else rv
        ax.plot(sigma.index, sigma.values, label=ticker, linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised σ (%)" if annualise else "RV")
    ax.legend(ncol=len(rv_dict))
    if annualise:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: MSE / QLIKE box-plot across stocks
# ---------------------------------------------------------------------------

def plot_loss_boxplot(
    loss_table: pd.DataFrame,
    loss_name: str = "MSE",
    relative_to: str | None = "HAR",
    title: str | None = None,
) -> plt.Figure:
    """Boxplot of per-stock loss (or loss ratio) by model.

    Parameters
    ----------
    loss_table
        DataFrame with rows = stocks, columns = model labels, values = loss
        on the test set.
    relative_to
        If given, normalise each row by the value in this column so
        the resulting ratios are comparable across stocks of different
        volatility levels. A ratio < 1 means "better than the baseline".
    """
    _setup_style()
    if relative_to and relative_to in loss_table.columns:
        plot_df = loss_table.div(loss_table[relative_to], axis=0)
        ylabel = f"{loss_name} relative to {relative_to}"
    else:
        plot_df = loss_table
        ylabel = loss_name

    labels = list(plot_df.columns)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    data = [plot_df[c].values for c in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(_FAMILY_COLORS[_family(lab)])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
    if relative_to:
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: MCS inclusion rate
# ---------------------------------------------------------------------------

def plot_mcs_inclusion(
    inclusion: dict[str, float],
    alpha: float = 0.10,
    title: str | None = None,
) -> plt.Figure:
    """Bar chart of MCS p-values per model."""
    _setup_style()
    labels = list(inclusion.keys())
    pvals = [inclusion[lab] for lab in labels]
    colors = [_FAMILY_COLORS[_family(lab)] for lab in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.bar(labels, pvals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(alpha, color="black", linestyle="--", linewidth=0.7,
               label=f"α = {alpha:.2f}")
    ax.set_ylabel("MCS p-value")
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: ALE plots
# ---------------------------------------------------------------------------

def plot_ale_curves(
    ale_per_model: Mapping[str, "object"],
    feature: str,
    title: str | None = None,
) -> plt.Figure:
    """Overlay ALE curves for the same feature across multiple models.

    ``ale_per_model`` maps model label to an :class:`~src.evaluation.ale.ALEResult`.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for label, res in ale_per_model.items():
        ax.plot(res.bin_centers, res.ale, label=label,
                color=_FAMILY_COLORS[_family(label)],
                linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("Accumulated local effect on RV")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: prediction vs realised on a representative test slice
# ---------------------------------------------------------------------------

def plot_predictions_vs_realised(
    realised: pd.Series,
    predictions: Mapping[str, pd.Series],
    annualise: bool = True,
    title: str | None = None,
    window: tuple[str, str] | None = None,
) -> plt.Figure:
    """Plot the realised RV against a small set of model predictions."""
    _setup_style()
    if window:
        realised = realised.loc[window[0]:window[1]]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    series = (np.sqrt(realised * 252) * 100) if annualise else realised
    ax.plot(series.index, series.values, label="Realised",
            color="black", linewidth=1.0)
    for label, p in predictions.items():
        p_slice = p.loc[realised.index]
        s = (np.sqrt(p_slice * 252) * 100) if annualise else p_slice
        ax.plot(s.index, s.values, label=label, linewidth=0.8,
                color=_FAMILY_COLORS[_family(label)], alpha=0.85)
    ax.set_ylabel("Annualised σ (%)" if annualise else "RV")
    if annualise:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if title:
        ax.set_title(title)
    ax.legend(ncol=min(4, len(predictions) + 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str, subdir: str | None = None) -> str:
    cfg = load_config()
    base = resolve(cfg.paths.outputs_figures)
    out_dir = base if subdir is None else base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    _LOG.info("Saved figure %s", path)
    return str(path)
