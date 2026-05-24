"""
Stage 14 — COVID-period output generator.

Produces loss tables and DM/MCS tests over the COVID custom-split
predictions (train 2016-2019 / test 2020-2024) for every horizon and
both loss functions. Outputs file names are prefixed ``covidfull_`` to
distinguish them from the headline (2023-2024 test) tables.

Usage:
    python scripts/14_covid_outputs.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve
from src.evaluation.metrics import LOSSES
from src.evaluation.diebold_mariano import diebold_mariano
from src.evaluation.mcs import model_confidence_set
from src.evaluation.bootstrap import bootstrap_loss_ci
from src.evaluation.mincer_zarnowitz import mz_summary_table
from src.pipeline.orchestrator import load_results


def _process_horizon(horizon: int, loss_name: str, baseline: str = "HAR") -> dict:
    cfg = load_config()
    log = get_logger("covid_outputs", level=cfg.project.log_level)
    loss_fn = LOSSES[loss_name]
    out_dir = resolve(cfg.paths.outputs_tables)

    fn = f"predictions_covid_full_M_ALL_h{horizon}.pkl"
    try:
        runs = load_results(fn)
    except FileNotFoundError:
        log.error("Missing %s", fn)
        return {"horizon": horizon, "loss": loss_name, "status": "missing"}

    # Loss table per stock × model
    loss_rows: dict[str, dict[str, float]] = {}
    for r in runs:
        loss_rows[r.ticker] = {}
        y = r.y_true
        for m, p in r.predictions.items():
            idx = y.index.intersection(p.index)
            loss_rows[r.ticker][m] = float(np.mean(loss_fn(y.loc[idx].to_numpy(),
                                                            p.loc[idx].to_numpy())))
    loss_df = pd.DataFrame(loss_rows).T
    loss_df.to_csv(out_dir / f"covidfull_loss_h{horizon}_{loss_name}.csv")
    if baseline in loss_df.columns:
        ratio_df = loss_df.div(loss_df[baseline], axis=0)
        ratio_df.to_csv(out_dir / f"covidfull_loss_ratio_h{horizon}_{loss_name}.csv")

    # DM tests vs HAR
    dm_rows = []
    for r in runs:
        if baseline not in r.predictions:
            continue
        base = r.predictions[baseline]
        common = r.y_true.index.intersection(base.index)
        for m, p in r.predictions.items():
            if m == baseline:
                continue
            idx = common.intersection(p.index)
            try:
                d = diebold_mariano(
                    r.y_true.loc[idx].to_numpy(),
                    pred_a=p.loc[idx].to_numpy(),
                    pred_b=base.loc[idx].to_numpy(),
                    loss=loss_name, alternative="less", horizon=horizon,
                )
                dm_rows.append({
                    "ticker": r.ticker, "model": m,
                    "DM_stat": d.statistic, "p_value": d.pvalue,
                    "mean_diff": d.mean_diff,
                })
            except Exception:
                continue
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / f"covidfull_dm_h{horizon}_{loss_name}.csv", index=False)

    # MCS at three alphas
    for alpha in (0.25, 0.10, 0.05):
        try:
            mcs_rows = {}
            for r in runs:
                common = r.y_true.index
                preds = {}
                for k, p in r.predictions.items():
                    idx = r.y_true.index.intersection(p.index)
                    preds[k] = p.loc[idx].to_numpy()
                    common = common.intersection(idx)
                y = r.y_true.loc[common].to_numpy()
                preds = {k: r.predictions[k].loc[common].to_numpy() for k in r.predictions}
                res = model_confidence_set(
                    y, preds, loss=loss_name, alpha=alpha,
                    num_bootstrap=cfg.mcs.num_bootstrap,
                    block_length=cfg.mcs.block_length,
                    statistic=cfg.mcs.statistic, seed=cfg.project.seed,
                )
                mcs_rows[r.ticker] = res.p_values
            pd.DataFrame(mcs_rows).T.to_csv(
                out_dir / f"covidfull_mcs_h{horizon}_{loss_name}_a{int(alpha*100):02d}.csv"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("MCS failed h=%d α=%.2f: %s", horizon, alpha, exc)

    # MZ
    mz_rows = []
    for r in runs:
        idx = r.y_true.index
        for k, p in r.predictions.items():
            idx = idx.intersection(p.index)
        forecasts = {k: r.predictions[k].loc[idx].to_numpy() for k in r.predictions}
        y_aligned = r.y_true.loc[idx].to_numpy()
        tab = mz_summary_table(forecasts, y_aligned)
        tab["ticker"] = r.ticker
        mz_rows.append(tab)
    if mz_rows:
        pd.concat(mz_rows).to_csv(out_dir / f"covidfull_mz_h{horizon}_{loss_name}.csv")

    log.info("[h=%d, loss=%s] COVID outputs written", horizon, loss_name)
    return {"horizon": horizon, "loss": loss_name, "status": "ok"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--losses", nargs="*", default=["mse", "qlike"])
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("covid_outputs", level=cfg.project.log_level)
    horizons = args.horizons if args.horizons else cfg.forecast.horizons
    losses = args.losses

    # Run each (horizon × loss) in a separate process. Each is independent.
    tasks = [(h, loss) for h in horizons for loss in losses]
    log.info("Running %d (horizon × loss) tasks in parallel...", len(tasks))
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        futs = {ex.submit(_process_horizon, h, loss): (h, loss) for h, loss in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            log.info("done: %s", res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
