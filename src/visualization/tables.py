"""
Result-table generation: pandas DataFrames, LaTeX, and Markdown output.

The main report needs three headline tables:

* **Table 1** — Out-of-sample MSE (or QLIKE) for every model on every stock
  at every horizon, expressed as a ratio to the HAR baseline. Bolding
  highlights the winning model per stock-horizon combination.

* **Table 2** — Diebold-Mariano p-values for each ML model vs the HAR
  baseline (one-sided, "ML has lower loss"), pooled across stocks.

* **Table 3** — Model Confidence Set p-values: which models survive at
  the 90% confidence level, per stock.

Additional appendix tables come from these helpers as well: hyperparameter
selection summaries, robustness checks, and COVID-period splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config, resolve

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss tables
# ---------------------------------------------------------------------------

def build_loss_table(
    y_true_dict: Mapping[str, pd.Series],
    pred_dict: Mapping[str, Mapping[str, pd.Series]],
    loss_fn,
) -> pd.DataFrame:
    """Compute mean loss per (stock, model).

    Parameters
    ----------
    y_true_dict
        ``{ticker: y_true_series}``.
    pred_dict
        ``{ticker: {model_label: y_pred_series}}``.
    loss_fn
        Pointwise loss callable, e.g. ``metrics.mse_loss``.
    """
    rows: dict[str, dict[str, float]] = {}
    for ticker, y in y_true_dict.items():
        row: dict[str, float] = {}
        for label, pred in pred_dict.get(ticker, {}).items():
            common = y.index.intersection(pred.index)
            if len(common) == 0:
                row[label] = np.nan
            else:
                row[label] = float(np.mean(loss_fn(y.loc[common].to_numpy(),
                                                   pred.loc[common].to_numpy())))
        rows[ticker] = row
    return pd.DataFrame(rows).T


def loss_ratio_table(loss_table: pd.DataFrame, baseline: str = "HAR") -> pd.DataFrame:
    """Convert absolute losses to ratios relative to a baseline column."""
    if baseline not in loss_table.columns:
        raise KeyError(f"Baseline '{baseline}' missing from loss table")
    return loss_table.div(loss_table[baseline], axis=0)


def cross_sectional_aggregate(
    ratio_tables: list[pd.DataFrame],
    statistic: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-stock loss-ratio tables across the cross-section.

    Christensen et al. (2023) report cross-sectional averages across their
    29-stock sample. With our 3-stock cross-section we report both the
    mean and the median (median is more robust to AMZN's idiosyncratic
    behaviour at long horizons).

    Parameters
    ----------
    ratio_tables
        List of DataFrames each with rows = stocks (typically one row each),
        columns = models. Values are loss ratios vs baseline.
    statistic
        ``mean`` or ``median``.

    Returns
    -------
    DataFrame indexed by aggregation statistic, columns = model labels.
    """
    if not ratio_tables:
        return pd.DataFrame()
    stacked = pd.concat(ratio_tables, axis=0)
    if statistic == "mean":
        agg = stacked.mean(axis=0)
    elif statistic == "median":
        agg = stacked.median(axis=0)
    else:
        raise ValueError(f"Unknown statistic '{statistic}'")
    return agg.to_frame(name=statistic).T


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_loss_ratio_table(
    ratio_table: pd.DataFrame,
    bold_winner_per_row: bool = True,
    decimals: int = 3,
) -> pd.DataFrame:
    """Return a DataFrame of strings ready for printing or to_latex()."""
    out = ratio_table.copy().astype(object)
    for ticker, row in ratio_table.iterrows():
        winner = row.idxmin()
        for col in ratio_table.columns:
            v = ratio_table.loc[ticker, col]
            if pd.isna(v):
                out.loc[ticker, col] = "—"
            else:
                txt = f"{v:.{decimals}f}"
                if bold_winner_per_row and col == winner:
                    txt = f"\\textbf{{{txt}}}"
                out.loc[ticker, col] = txt
    return out


# ---------------------------------------------------------------------------
# DM and MCS tables
# ---------------------------------------------------------------------------

def dm_pvalue_table(
    y_true: pd.Series,
    preds: Mapping[str, pd.Series],
    baseline: str = "HAR",
    horizon: int = 1,
    loss: str = "mse",
) -> pd.DataFrame:
    """One-sided DM p-values for each model vs the baseline."""
    from ..evaluation.diebold_mariano import diebold_mariano
    out = []
    base = preds[baseline]
    common = y_true.index.intersection(base.index)
    for label, pred in preds.items():
        if label == baseline:
            out.append((label, np.nan, np.nan, np.nan))
            continue
        idx = common.intersection(pred.index)
        r = diebold_mariano(
            y_true.loc[idx].to_numpy(),
            pred_a=pred.loc[idx].to_numpy(),
            pred_b=base.loc[idx].to_numpy(),
            loss=loss, alternative="less", horizon=horizon,
        )
        out.append((label, r.statistic, r.pvalue, r.mean_diff))
    df = pd.DataFrame(out, columns=["Model", "DM stat", "p-value", "mean loss diff"])
    df = df.set_index("Model")
    return df


def mcs_pvalue_table(mcs_results: Mapping[str, "object"]) -> pd.DataFrame:
    """Per-stock MCS p-values laid out as a table."""
    rows: dict[str, dict[str, float]] = {}
    for ticker, res in mcs_results.items():
        rows[ticker] = dict(res.p_values)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, name: str, fmt: str = "csv") -> str:
    cfg = load_config()
    out_dir = resolve(cfg.paths.outputs_tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "tex":
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(escape=False))
    elif fmt == "md":
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown())
    else:
        raise ValueError(f"Unknown format '{fmt}'")
    _LOG.info("Saved table %s", path)
    return str(path)
