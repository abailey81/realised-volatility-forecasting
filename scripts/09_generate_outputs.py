"""
Stage 9 — generate final tables and figures for the report.

Reads merged prediction pickles and ALE pickles from earlier stages and
produces:

* ``outputs/figures/rv_time_series.pdf`` — annualised realised vol per stock.
* ``outputs/figures/loss_boxplot_h{h}.pdf`` — per-stock loss ratios.
* ``outputs/figures/mcs_inclusion_h{h}.pdf`` — MCS p-values bar chart.
* ``outputs/figures/ale_<stock>_<feature>.pdf`` — overlaid ALE curves.
* ``outputs/tables/headline_h{h}.tex`` — LaTeX table of MSE ratios with
  DM-significance stars and winners in bold.

Plus a COVID-period sub-table when the extension is enabled in config.

Usage:
    python scripts/09_generate_outputs.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve
from src.pipeline.orchestrator import load_results, StockRunResult
from src.data.compute_rv import load_realised
from src.evaluation.metrics import LOSSES
from src.evaluation.diebold_mariano import diebold_mariano
from src.evaluation.mincer_zarnowitz import mz_summary_table
from src.evaluation.bootstrap import bootstrap_loss_ci, bootstrap_diff_ci
from src.models.combinations import build_combination_predictions
from src.visualization.plots import (
    plot_rv_time_series, plot_loss_boxplot, plot_mcs_inclusion,
    plot_ale_curves, save_figure,
)
from src.visualization.tables import (
    build_loss_table, loss_ratio_table, format_loss_ratio_table, save_table,
)


def _merge(files: list[str]) -> list[StockRunResult]:
    merged: dict[tuple[str, str, int], StockRunResult] = {}
    for f in files:
        try:
            results = load_results(f)
        except FileNotFoundError:
            continue
        for r in results:
            key = (r.ticker, r.feature_set, r.horizon)
            if key in merged:
                merged[key].predictions.update(r.predictions)
                # Also merge val_predictions so downstream combinations have
                # access to every model's validation forecast, not just those
                # in the first-loaded file.
                if getattr(r, "val_predictions", None):
                    if merged[key].val_predictions is None:
                        merged[key].val_predictions = {}
                    merged[key].val_predictions.update(r.val_predictions)
                if getattr(r, "y_val", None) is not None and merged[key].y_val is None:
                    merged[key].y_val = r.y_val
            else:
                merged[key] = r
    return list(merged.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--loss", default="mse", choices=list(LOSSES))
    parser.add_argument("--inputs", nargs="*",
                        default=["predictions_har.pkl",
                                 "predictions_ml.pkl",
                                 "predictions_nn.pkl"])
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("generate_outputs", level=cfg.project.log_level)
    loss_fn = LOSSES[args.loss]

    # === Figure 1: RV time series ===
    try:
        rv_dict = {t: load_realised(t)["RV"] for t in cfg.data.stocks}
        fig = plot_rv_time_series(rv_dict)
        save_figure(fig, "rv_time_series.pdf")
    except Exception as exc:  # noqa: BLE001
        log.warning("RV figure failed: %s", exc)

    # === Load all predictions ===
    runs = [r for r in _merge(args.inputs) if r.horizon == args.horizon]
    if not runs:
        log.error("No predictions at horizon %d", args.horizon)
        return 1

    y_true_dict = {r.ticker: r.y_true for r in runs}
    pred_dict = {r.ticker: r.predictions for r in runs}

    # === Headline loss table ===
    loss_tab = build_loss_table(y_true_dict, pred_dict, loss_fn=loss_fn)
    ratio_tab = loss_ratio_table(loss_tab, baseline="HAR")
    formatted = format_loss_ratio_table(ratio_tab)
    save_table(loss_tab, f"loss_h{args.horizon}_{args.loss}", fmt="csv")
    save_table(ratio_tab, f"loss_ratio_h{args.horizon}_{args.loss}", fmt="csv")
    save_table(formatted, f"headline_h{args.horizon}_{args.loss}", fmt="tex")

    # === Loss boxplot ===
    try:
        fig = plot_loss_boxplot(loss_tab, loss_name=args.loss.upper(),
                                relative_to="HAR",
                                title=f"{args.loss.upper()} ratio to HAR, h={args.horizon}")
        save_figure(fig, f"loss_boxplot_h{args.horizon}.pdf")
    except Exception as exc:  # noqa: BLE001
        log.warning("Loss boxplot failed: %s", exc)

    # === MCS bar chart per stock ===
    try:
        from src.evaluation.mcs import model_confidence_set
        for r in runs:
            try:
                idx = r.y_true.index
                for k, p in r.predictions.items():
                    idx = idx.intersection(p.index)
                y = r.y_true.loc[idx].to_numpy()
                preds = {k: r.predictions[k].loc[idx].to_numpy() for k in r.predictions}
                res = model_confidence_set(
                    y, preds, loss=args.loss,
                    alpha=cfg.mcs.alpha, num_bootstrap=cfg.mcs.num_bootstrap,
                    block_length=cfg.mcs.block_length, statistic=cfg.mcs.statistic,
                    seed=cfg.project.seed,
                )
                fig = plot_mcs_inclusion(res.p_values, alpha=cfg.mcs.alpha,
                                          title=f"MCS — {r.ticker}, h={args.horizon}")
                save_figure(fig, f"mcs_{r.ticker}_h{args.horizon}.pdf")
            except Exception as exc:  # noqa: BLE001
                log.warning("MCS plot failed for %s: %s", r.ticker, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("MCS module failed: %s", exc)

    # === ALE plots ===
    try:
        ale_pkl = resolve(cfg.paths.outputs_results) / "ale" / f"ale_{cfg.ale.stock_for_plots}_h{args.horizon}.pkl"
        if ale_pkl.exists():
            with open(ale_pkl, "rb") as f:
                ale_table = pickle.load(f)
            for feat in cfg.ale.features:
                curves = {label: res for (label, fname), res in ale_table.items() if fname == feat}
                if not curves:
                    continue
                fig = plot_ale_curves(curves, feature=feat,
                                       title=f"ALE for {feat} on {cfg.ale.stock_for_plots}")
                save_figure(fig, f"ale_{cfg.ale.stock_for_plots}_{feat}.pdf")
    except Exception as exc:  # noqa: BLE001
        log.warning("ALE figures failed: %s", exc)

    # === COVID sub-period table ===
    if cfg.extensions.covid_subperiod.enabled:
        try:
            covid_start = pd.Timestamp(cfg.extensions.covid_subperiod.covid_start)
            covid_end   = pd.Timestamp(cfg.extensions.covid_subperiod.covid_end)
            covid_rows: dict[str, dict[str, float]] = {}
            for r in runs:
                mask = (r.y_true.index >= covid_start) & (r.y_true.index <= covid_end)
                if not mask.any():
                    continue
                y = r.y_true.loc[mask]
                covid_rows[r.ticker] = {}
                for k, p in r.predictions.items():
                    idx = y.index.intersection(p.index)
                    covid_rows[r.ticker][k] = float(np.mean(
                        loss_fn(y.loc[idx].to_numpy(), p.loc[idx].to_numpy())
                    ))
            covid_tab = pd.DataFrame(covid_rows).T
            if "HAR" in covid_tab.columns:
                covid_ratio = covid_tab.div(covid_tab["HAR"], axis=0)
            else:
                covid_ratio = covid_tab
            save_table(covid_tab, f"covid_loss_h{args.horizon}_{args.loss}", fmt="csv")
            save_table(covid_ratio, f"covid_loss_ratio_h{args.horizon}_{args.loss}", fmt="csv")
        except Exception as exc:  # noqa: BLE001
            log.warning("COVID table failed: %s", exc)

    # === Mincer-Zarnowitz forecast efficiency tests ===
    if getattr(cfg, "mincer_zarnowitz", None) and cfg.mincer_zarnowitz.enabled:
        try:
            mz_rows = []
            for r in runs:
                forecasts_np = {}
                idx = r.y_true.index
                for k, p in r.predictions.items():
                    idx = idx.intersection(p.index)
                for k, p in r.predictions.items():
                    forecasts_np[k] = p.loc[idx].to_numpy()
                y_aligned = r.y_true.loc[idx].to_numpy()
                tab = mz_summary_table(forecasts_np, y_aligned)
                tab["ticker"] = r.ticker
                mz_rows.append(tab)
            if mz_rows:
                mz_all = pd.concat(mz_rows)
                save_table(mz_all, f"mz_h{args.horizon}_{args.loss}", fmt="csv")
                log.info("Mincer-Zarnowitz table saved")
        except Exception as exc:  # noqa: BLE001
            log.warning("MZ table failed: %s", exc)

    # === Bootstrap confidence intervals on headline MSE ===
    if getattr(cfg, "bootstrap", None) and cfg.bootstrap.enabled:
        try:
            ci_rows = []
            for r in runs:
                for k, p in r.predictions.items():
                    common = r.y_true.index.intersection(p.index)
                    losses = loss_fn(r.y_true.loc[common].to_numpy(),
                                     p.loc[common].to_numpy())
                    ci = bootstrap_loss_ci(
                        losses,
                        alpha=cfg.bootstrap.alpha,
                        num_bootstrap=cfg.bootstrap.num_bootstrap,
                        block_length=cfg.bootstrap.block_length,
                        seed=cfg.project.seed,
                    )
                    ci_rows.append({
                        "ticker": r.ticker, "model": k,
                        "loss": ci.estimate, "se": ci.se,
                        "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                    })
            if ci_rows:
                ci_tab = pd.DataFrame(ci_rows)
                save_table(ci_tab, f"bootstrap_ci_h{args.horizon}_{args.loss}", fmt="csv")
                log.info("Bootstrap CIs saved")
        except Exception as exc:  # noqa: BLE001
            log.warning("Bootstrap CIs failed: %s", exc)

    # === Forecast combinations using stored validation predictions ===
    if getattr(cfg, "combinations", None) and cfg.combinations.enabled:
        try:
            comb_rows = []
            for r in runs:
                # Use the orchestrator-saved val predictions and validation
                # targets — no data snooping into the test set.
                val_preds = getattr(r, "val_predictions", None) or {}
                y_val = getattr(r, "y_val", None)
                if not val_preds or y_val is None or len(y_val) == 0:
                    continue
                pool = [m for m in cfg.combinations.pool if m in r.predictions and m in val_preds]
                if len(pool) < 2:
                    continue
                val_f = {m: val_preds[m] for m in pool}
                test_f = {m: r.predictions[m] for m in pool}
                combos = build_combination_predictions(
                    y_val, val_f, test_f, methods=tuple(cfg.combinations.methods),
                )
                for label, p in combos.items():
                    common = r.y_true.index.intersection(p.index)
                    losses = loss_fn(r.y_true.loc[common].to_numpy(), p.loc[common].to_numpy())
                    comb_rows.append({
                        "ticker": r.ticker, "combo": label,
                        "loss": float(np.mean(losses)),
                    })
            if comb_rows:
                comb_tab = pd.DataFrame(comb_rows).pivot(index="ticker", columns="combo", values="loss")
                save_table(comb_tab, f"combinations_h{args.horizon}_{args.loss}", fmt="csv")
                log.info("Combinations table saved")
        except Exception as exc:  # noqa: BLE001
            log.warning("Combinations failed: %s", exc)

    log.info("Outputs generated under %s", resolve(cfg.paths.outputs_tables).parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
