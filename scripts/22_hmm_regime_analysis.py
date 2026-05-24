"""
Stage 23 — Hidden-Markov regime-conditional forecast evaluation.

The paper compares ML and HAR forecasts unconditionally. That hides the
question that is most interesting from a financial perspective: does the
ML-vs-HAR gap *depend on regime*? A model that wins only when volatility
is well behaved is a different proposition for a risk manager than a
model that wins precisely when volatility spikes.

This stage fits a 2-state Markov regime-switching model on each stock's
log realised variance (Hamilton 1989 / Ang-Bekaert 2002), uses the
smoothed regime probabilities to classify each test-set day as
*low-vol* or *high-vol*, and computes regime-conditional MSE for every
model from `predictions_*_M_*.pkl` (HAR family, regularised, trees, NN
ensembles, NN top-1).

Outputs
-------
* `hmm_regime_states_{ticker}.csv`    — per-day regime label + smoothed prob
* `hmm_regime_params_{ticker}.csv`    — fitted (μ_low, μ_high, σ²_low,
                                        σ²_high, P_00, P_11) parameters
* `hmm_regime_losses_{feature_set}_h{h}.csv` — per (ticker, model, regime,
                                               loss) table
* `hmm_regime_ratio_{feature_set}_h{h}.csv`  — model / HAR loss ratio per
                                               regime
* `hmm_regime_summary.csv`            — cross-section aggregate table

Usage
-----
    python scripts/22_hmm_regime_analysis.py
    python scripts/22_hmm_regime_analysis.py --horizons 1 5 --losses mse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.compute_rv import load_realised
from src.evaluation.metrics import mse_loss, qlike_loss
from src.evaluation.regime import fit_two_state_hmm, regime_conditional_losses
from src.pipeline.orchestrator import load_results


LOSS_FNS = {"mse": mse_loss, "qlike": qlike_loss}


def _fit_per_stock_hmm(tickers, log, out_tables):
    """Fit a 2-state HMM per stock on the full log-RV history.

    Returns
    -------
    dict[str, pd.Series]
        Mapping ``ticker -> regime`` (0 / 1 per date, low / high vol).
    """
    regimes = {}
    summary_rows = []
    for t in tickers:
        rv_df = load_realised(t)
        rv = rv_df["RV"].astype(float).dropna()
        log.info("[%s] fitting 2-state HMM on log(RV) | n=%d", t, len(rv))
        res = fit_two_state_hmm(rv, log_transform=True)
        regimes[t] = res.regime
        params = {
            "ticker": t,
            "mu_low":  res.means[0], "mu_high":  res.means[1],
            "var_low": res.variances[0], "var_high": res.variances[1],
            "P_00":    res.transition[0, 0], "P_11":   res.transition[1, 1],
            "P_01":    res.transition[0, 1], "P_10":   res.transition[1, 0],
            "n_low":   int((res.regime == 0).sum()),
            "n_high":  int((res.regime == 1).sum()),
        }
        summary_rows.append(params)
        # Persist the per-day regime labels + smoothed probability of high vol
        regime_df = pd.DataFrame({
            "regime": res.regime,
            "p_high": res.smoothed_probs["regime_1"],
        })
        regime_df.index.name = "date"
        regime_df.to_csv(out_tables / f"hmm_regime_states_{t}.csv")
        log.info("  μ=(%.3f, %.3f) σ²=(%.3f, %.3f) P=(%.3f, %.3f) n=(%d, %d)",
                 res.means[0], res.means[1],
                 res.variances[0], res.variances[1],
                 res.transition[0, 0], res.transition[1, 1],
                 params["n_low"], params["n_high"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_tables / "hmm_regime_params.csv", index=False)
    return regimes, summary_df


def _merge_predictions_for(feature_set: str, horizon: int):
    """Concatenate HAR + ML + NN prediction pickles for one (FS, h)."""
    if feature_set == "M_ALL":
        files = ["predictions_har_MALL.pkl",
                 "predictions_ml_MALL.pkl",
                 "predictions_nn_MALL.pkl"]
    else:
        files = ["predictions_har_MHAR.pkl",
                 "predictions_ml_MHAR.pkl",
                 "predictions_nn_MHAR.pkl"]

    merged: dict[tuple, "object"] = {}
    for fn in files:
        try:
            for r in load_results(fn):
                if r.horizon != horizon or r.feature_set != feature_set:
                    continue
                key = (r.ticker, r.feature_set, r.horizon)
                if key in merged:
                    merged[key].predictions.update(r.predictions)
                else:
                    merged[key] = r
        except FileNotFoundError:
            continue
    return list(merged.values())


def _compute_regime_table(
    runs, regimes, loss_name: str, loss_fn,
) -> pd.DataFrame:
    """Build long-form (ticker, model, regime, mean_loss) table."""
    rows = []
    for r in runs:
        regime = regimes.get(r.ticker)
        if regime is None:
            continue
        tab = regime_conditional_losses(r.y_true, r.predictions, regime, loss_fn)
        # tab: rows in {low_vol, high_vol}, columns are model labels
        for model in tab.columns:
            for regime_label in tab.index:
                rows.append({
                    "ticker": r.ticker,
                    "model": model,
                    "regime": regime_label,
                    "mean_loss": float(tab.loc[regime_label, model]),
                    "loss": loss_name,
                })
    return pd.DataFrame(rows)


def _ratio_table(long_df: pd.DataFrame, baseline: str = "HAR") -> pd.DataFrame:
    """Pivot a long regime-loss table into ratio vs HAR per (ticker, regime)."""
    out = []
    for (t, reg), grp in long_df.groupby(["ticker", "regime"]):
        if (grp["model"] == baseline).sum() == 0:
            continue
        base = float(grp.loc[grp["model"] == baseline, "mean_loss"].iloc[0])
        if base <= 0 or not np.isfinite(base):
            continue
        for _, row in grp.iterrows():
            out.append({
                "ticker": t,
                "regime": reg,
                "model": row["model"],
                "ratio_vs_HAR": row["mean_loss"] / base,
                "loss": row["loss"],
            })
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    parser.add_argument("--feature-sets", nargs="*",
                        default=["M_HAR", "M_ALL"])
    parser.add_argument("--losses", nargs="*", default=["mse", "qlike"])
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("hmm_regime", level=cfg.project.log_level)
    out_tables = resolve(cfg.paths.outputs_tables)
    out_tables.mkdir(parents=True, exist_ok=True)

    tickers = list(cfg.data.stocks)

    # Step 1: fit per-stock HMMs, persist regime labels + parameters
    regimes, params_df = _fit_per_stock_hmm(tickers, log, out_tables)
    log.info("Saved per-stock HMM parameter table (%d stocks)", len(params_df))

    # Step 2: per (feature_set, horizon, loss), compute regime-conditional MSE
    summary_rows = []
    for fs in args.feature_sets:
        for h in args.horizons:
            runs = _merge_predictions_for(fs, h)
            if not runs:
                log.warning("no prediction pickle for %s h=%d — skipping", fs, h)
                continue
            for loss_name in args.losses:
                loss_fn = LOSS_FNS[loss_name]
                long_df = _compute_regime_table(runs, regimes, loss_name, loss_fn)
                if long_df.empty:
                    continue
                # Wide table — rows = model, columns = (ticker, regime)
                wide = long_df.pivot_table(
                    index="model", columns=["ticker", "regime"], values="mean_loss",
                )
                wide.to_csv(out_tables / f"hmm_regime_losses_{fs}_h{h}_{loss_name}.csv")

                # Ratio vs HAR per regime
                ratio_df = _ratio_table(long_df, baseline="HAR")
                ratio_wide = ratio_df.pivot_table(
                    index="model", columns=["ticker", "regime"], values="ratio_vs_HAR",
                )
                ratio_wide.to_csv(
                    out_tables / f"hmm_regime_ratio_{fs}_h{h}_{loss_name}.csv"
                )

                # Pooled-across-stocks aggregate per regime
                pool = ratio_df.groupby(["model", "regime"])["ratio_vs_HAR"].mean()
                for (model, regime), val in pool.items():
                    summary_rows.append({
                        "feature_set": fs, "horizon": h, "loss": loss_name,
                        "model": model, "regime": regime,
                        "pooled_ratio_vs_HAR": float(val),
                    })
                log.info("[%s|h=%d|%s] regime-conditional table saved",
                         fs, h, loss_name)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_tables / "hmm_regime_summary.csv", index=False)
    log.info("Saved cross-section regime summary (%d rows)", len(summary))

    # Quick leaderboard print for the most-watched cell
    cell = summary[
        (summary["feature_set"] == "M_ALL") &
        (summary["horizon"] == 1) &
        (summary["loss"] == "mse")
    ]
    if not cell.empty:
        wide = cell.pivot_table(
            index="model", columns="regime", values="pooled_ratio_vs_HAR",
        ).sort_values("high_vol")
        log.info("\nM_ALL h=1 MSE pooled ratio vs HAR (sorted by high-vol):\n%s",
                 wide.to_string(float_format=lambda v: f"{v:.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
