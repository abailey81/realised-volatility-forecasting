"""
Stage 22 — bootstrap confidence intervals on the path-decomposition.

The path decomposition (Critique 1.2) splits the M_HAR → M_ALL improvement
into four legs:

* HAR-X overfit penalty (HAR-X / HAR ratio on M_ALL)
* Regularisation gain (EN / HAR ratio on M_ALL)
* Tree marginal (RF / EN ratio on M_ALL)
* Deep-NN marginal (best-NN / RF ratio on M_ALL)

These are *ratios of MSEs*. The point estimates in
`critique_path_decomposition_h*.csv` show large per-stock heterogeneity
(JPM has negative regularisation contribution). To know whether the
decomposition is robust or noise-dominated, this script computes
moving-block-bootstrap 95% CIs on each leg, per (stock, horizon).

Block bootstrap is used because daily MSE losses are autocorrelated;
the Politis-White (2004) rule n^(1/3) gives ~7 days at our sample size.

Usage:
    python scripts/22_path_decomposition_ci.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.evaluation.metrics import mse_loss
from src.pipeline.orchestrator import load_results


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _bootstrap_ratio(losses_num: np.ndarray, losses_den: np.ndarray,
                      n_boot: int = 5000, block: int | None = None,
                      seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for the ratio of mean losses."""
    n = len(losses_num)
    if block is None:
        block = max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_indices(n, block, rng)
        num = float(np.mean(losses_num[idx]))
        den = float(np.mean(losses_den[idx]))
        ratios[b] = num / den if den > 0 else np.nan
    point = float(np.mean(losses_num) / np.mean(losses_den))
    ratios = ratios[~np.isnan(ratios)]
    return point, float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _merge(files: list[str], horizon: int, feature_set: str):
    merged: dict[tuple, "object"] = {}
    for f in files:
        try:
            for r in load_results(f):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("path_ci", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)

    files = ["predictions_har_MALL.pkl",
             "predictions_ml_MALL.pkl",
             "predictions_nn_MALL.pkl"]

    rows = []
    for h in args.horizons:
        for r in _merge(files, h, "M_ALL"):
            preds = r.predictions
            y_idx = r.y_true.index
            # Align all predictions to common index
            common = y_idx
            needed = ["HAR", "HAR-X", "EN", "RF"]
            nn_cols = [c for c in preds if c.startswith("NN") and (c.endswith("_ensemble") or c.endswith("_top1"))]
            for k in needed + nn_cols:
                if k not in preds:
                    continue
                common = common.intersection(preds[k].index)
            if not all(k in preds for k in needed):
                continue
            y = r.y_true.loc[common].to_numpy()
            har = preds["HAR"].loc[common].to_numpy()
            harx = preds["HAR-X"].loc[common].to_numpy()
            en = preds["EN"].loc[common].to_numpy()
            rf = preds["RF"].loc[common].to_numpy()
            nn_arrays = [preds[k].loc[common].to_numpy() for k in nn_cols]
            # Best NN per observation = the one with the lowest mean loss
            nn_losses_per_model = [np.mean(mse_loss(y, arr)) for arr in nn_arrays]
            best_nn_idx = int(np.argmin(nn_losses_per_model))
            best_nn_label = nn_cols[best_nn_idx]
            best_nn = nn_arrays[best_nn_idx]

            l_har  = mse_loss(y, har)
            l_harx = mse_loss(y, harx)
            l_en   = mse_loss(y, en)
            l_rf   = mse_loss(y, rf)
            l_nn   = mse_loss(y, best_nn)

            # Leg 1: HAR-X / HAR — overfit penalty
            p1, lo1, hi1 = _bootstrap_ratio(l_harx, l_har, args.n_boot)
            # Leg 2: EN / HAR — regularisation gain
            p2, lo2, hi2 = _bootstrap_ratio(l_en, l_har, args.n_boot)
            # Leg 3: RF / EN — tree marginal
            p3, lo3, hi3 = _bootstrap_ratio(l_rf, l_en, args.n_boot)
            # Leg 4: best-NN / RF — deep-NN marginal
            p4, lo4, hi4 = _bootstrap_ratio(l_nn, l_rf, args.n_boot)
            # Total: best-NN / HAR
            p5, lo5, hi5 = _bootstrap_ratio(l_nn, l_har, args.n_boot)

            rows.append({
                "ticker": r.ticker, "horizon": h, "best_nn": best_nn_label,
                "HARX_over_HAR":     p1, "HARX_over_HAR_lo": lo1, "HARX_over_HAR_hi": hi1,
                "EN_over_HAR":       p2, "EN_over_HAR_lo": lo2, "EN_over_HAR_hi": hi2,
                "RF_over_EN":        p3, "RF_over_EN_lo": lo3, "RF_over_EN_hi": hi3,
                "bestNN_over_RF":    p4, "bestNN_over_RF_lo": lo4, "bestNN_over_RF_hi": hi4,
                "bestNN_over_HAR":   p5, "bestNN_over_HAR_lo": lo5, "bestNN_over_HAR_hi": hi5,
            })
            log.info("[%s|h=%d] bestNN=%s: bestNN/HAR=%.3f [%.3f, %.3f]",
                      r.ticker, h, best_nn_label, p5, lo5, hi5)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "critique_path_decomposition_ci.csv", index=False)
    log.info("Saved path-decomposition CIs (%d rows)", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
