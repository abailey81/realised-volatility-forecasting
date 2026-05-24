"""
Stage 21 — multiple-testing correction on Diebold-Mariano p-values.

The paper runs many pairwise DM tests across (stocks × model pairs × horizons)
without correcting for multiplicity. With 22 models (21 contenders vs the
HAR baseline) and 3 stocks, this is 21 × 3 = 63 baseline-comparison tests
per horizon (`n_tests` column of `dm_multitest_summary.csv`). At α=0.05
nominal, the expected false-positive count under the null is ~3 per horizon.

This script reads the per-stock DM tables from `outputs/tables/dm_h*_mse.csv`
and applies three standard corrections:

* **Bonferroni** — most conservative, controls family-wise error rate.
* **Holm-Bonferroni** — step-down, more powerful than Bonferroni.
* **Benjamini-Hochberg** — controls false discovery rate (less conservative).

For each correction it reports how many cells in the DM matrix would have
been called "significant" under naive nominal α=0.05 vs the corrected
threshold.

Usage:
    python scripts/21_dm_multitest_correction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger, load_config, resolve


def _bonferroni(p: np.ndarray, alpha: float) -> np.ndarray:
    """Single-step Bonferroni: reject if p_i ≤ α/m."""
    m = (~np.isnan(p)).sum()
    if m == 0:
        return np.zeros_like(p, dtype=bool)
    return p <= (alpha / m)


def _holm(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm-Bonferroni step-down: order p, reject p_(i) ≤ α/(m-i+1)."""
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return np.zeros_like(p, dtype=bool)
    pv = p[valid]
    order = np.argsort(pv)
    sorted_p = pv[order]
    thresholds = alpha / (m - np.arange(m))
    rejected_in_order = sorted_p <= thresholds
    # Step-down rule: accept the first i where p_(i) > threshold, reject all i' < i
    if not rejected_in_order.any():
        flat = np.zeros(m, dtype=bool)
    else:
        first_accept = np.where(~rejected_in_order)[0]
        cutoff = first_accept[0] if first_accept.size > 0 else m
        flat = np.zeros(m, dtype=bool)
        flat[order[:cutoff]] = True
    out = np.zeros_like(p, dtype=bool)
    out[valid] = flat
    return out


def _bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR: order p, reject if p_(i) ≤ i·α/m."""
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return np.zeros_like(p, dtype=bool)
    pv = p[valid]
    order = np.argsort(pv)
    sorted_p = pv[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        flat = np.zeros(m, dtype=bool)
    else:
        cutoff = int(np.max(np.where(below)[0])) + 1
        flat = np.zeros(m, dtype=bool)
        flat[order[:cutoff]] = True
    out = np.zeros_like(p, dtype=bool)
    out[valid] = flat
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    parser.add_argument("--losses", nargs="*", default=["mse", "qlike"])
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("multitest", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)

    summary_rows = []
    for h in args.horizons:
        for loss in args.losses:
            path = out_dir / f"dm_h{h}_{loss}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "p-value" not in df.columns:
                continue
            pvals = df["p-value"].to_numpy()
            n_valid = int((~np.isnan(pvals)).sum())
            naive_rej = int(np.sum(pvals <= args.alpha))
            bonf_rej  = int(_bonferroni(pvals, args.alpha).sum())
            holm_rej  = int(_holm(pvals, args.alpha).sum())
            bh_rej    = int(_bh(pvals, args.alpha).sum())
            summary_rows.append({
                "horizon": h, "loss": loss, "n_tests": n_valid,
                "naive_alpha": args.alpha,
                "naive_rejections": naive_rej,
                "bonferroni_rejections": bonf_rej,
                "bonferroni_threshold": args.alpha / n_valid if n_valid else float("nan"),
                "holm_rejections": holm_rej,
                "bh_fdr_rejections": bh_rej,
            })

            # Attach to per-test table
            df["bonferroni_significant"] = _bonferroni(pvals, args.alpha)
            df["holm_significant"] = _holm(pvals, args.alpha)
            df["bh_fdr_significant"] = _bh(pvals, args.alpha)
            df.to_csv(out_dir / f"dm_multitest_h{h}_{loss}.csv", index=False)
            log.info("h=%d loss=%s: naive=%d, bonf=%d, holm=%d, BH=%d (of %d tests)",
                      h, loss, naive_rej, bonf_rej, holm_rej, bh_rej, n_valid)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "dm_multitest_summary.csv", index=False)
    log.info("Saved DM multi-testing summary across %d (h, loss) cells", len(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
