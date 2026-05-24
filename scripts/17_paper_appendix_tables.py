"""
Stage 17 — Paper-replication appendix tables and figures.

Produces three paper-specific artefacts that complement the headline run:

1. **Table A.5** — HAR and HAR-X in-sample parameter estimates with
   t-statistics. Replicates Christensen et al. (2023) Table A.5 for the
   AAPL stock at h=1 (the paper's illustration).

2. **Figure 8** — In-sample autocorrelation function for HAR, RF, and
   NN_2^10 fitted realised-variance series at h=1 and h=22. Shows that
   ML models capture the long-memory persistence of RV better than HAR.

3. **Cross-sectional aggregate** of out-of-sample MSE ratios — average
   across the 3-stock cross-section, for direct comparison with the
   paper's 29-stock cross-sectional means.

Usage:
    python scripts/17_paper_appendix_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.models.har_models import HAR, HARX, _fit_ols
from src.visualization.tables import cross_sectional_aggregate
from src.pipeline.orchestrator import load_results


def _table_a5(stock: str, horizon: int) -> pd.DataFrame:
    """In-sample parameter estimates for HAR and HAR-X with t-statistics.

    Mirrors paper Table A.5 layout: rows = parameters, columns = (HAR, HAR-X)
    with each coefficient followed by its t-statistic in parentheses.
    """
    feats = load_feature_matrix(stock, "M_ALL", horizon)
    cfg = load_config()
    train, val, _ = time_split(feats,
                                train_frac=cfg.data.split.train_frac,
                                val_frac=cfg.data.split.val_frac)
    insample = pd.concat([train, val])
    y = insample["y"].to_numpy()

    # HAR fit
    X_har = insample[["RVD", "RVW", "RVM"]].to_numpy()
    beta_h, int_h, resid_h = _fit_ols(X_har, y)
    se_h = _ols_se(X_har, resid_h)

    # HAR-X (paper convention: HAR + extras, no helpers)
    extras = ["IV", "EA", "M1W", "DVOL", "VIX", "EPU", "HSI", "ADS", "US3M"]
    har_x_cols = ["RVD", "RVW", "RVM"] + extras
    X_full = insample[har_x_cols].to_numpy()
    beta_x, int_x, resid_x = _fit_ols(X_full, y)
    se_x = _ols_se(X_full, resid_x)

    rows = []
    rows.append({"param": "intercept", "HAR": int_h, "HAR_tstat": float("nan"),
                 "HAR-X": int_x, "HAR-X_tstat": float("nan")})
    for i, name in enumerate(["RVD", "RVW", "RVM"]):
        rows.append({"param": name,
                     "HAR": beta_h[i],
                     "HAR_tstat": beta_h[i] / se_h[i + 1] if se_h[i + 1] > 0 else float("nan"),
                     "HAR-X": beta_x[i],
                     "HAR-X_tstat": beta_x[i] / se_x[i + 1] if se_x[i + 1] > 0 else float("nan")})
    for i, name in enumerate(extras):
        j = i + 3
        rows.append({"param": name,
                     "HAR": float("nan"), "HAR_tstat": float("nan"),
                     "HAR-X": beta_x[j],
                     "HAR-X_tstat": beta_x[j] / se_x[j + 1] if se_x[j + 1] > 0 else float("nan")})
    return pd.DataFrame(rows).set_index("param")


def _ols_se(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """White heteroskedasticity-robust standard errors (paper Table A.5 note)."""
    n, k = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])
    XtX_inv = np.linalg.inv(Xa.T @ Xa)
    u2 = resid ** 2
    S = (Xa.T * u2) @ Xa
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def _figure_8(stock: str) -> plt.Figure:
    """In-sample fitted ACF for HAR, RF, NN_2^10 at h=1 and h=22.

    Paper Figure 8 demonstrates that ML methods better capture the
    long-memory autocorrelation of realised variance.
    """
    import pickle
    cfg = load_config()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, h in zip(axes, [1, 22]):
        # Find the prediction pickle with the relevant models
        merged_preds: dict[str, pd.Series] = {}
        y_true_idx = None
        for f in ["predictions_har_MALL.pkl", "predictions_ml_MALL.pkl",
                   "predictions_nn_MALL.pkl"]:
            try:
                for r in load_results(f):
                    if r.ticker != stock or r.horizon != h or r.feature_set != "M_ALL":
                        continue
                    merged_preds.update(r.predictions)
                    if y_true_idx is None:
                        y_true_idx = r.y_true.index
            except FileNotFoundError:
                continue
        if not merged_preds or y_true_idx is None:
            continue

        # ACF of the fitted/predicted series for HAR, RF, NN_2^10
        for label, color in [("HAR", "k"), ("RF", "tab:blue"), ("NN2_ensemble", "tab:red")]:
            if label not in merged_preds:
                continue
            series = merged_preds[label].dropna().to_numpy()
            acf = _autocorr(series, max_lag=250)
            ax.plot(np.arange(len(acf)), acf, label=label.replace("_ensemble", "$^{10}$"),
                     color=color, linewidth=1.0)
        n = len(merged_preds[next(iter(merged_preds))])
        if n > 0:
            band = 1.96 / np.sqrt(n)
            ax.axhline(band, color="grey", linewidth=0.5, linestyle=":")
            ax.axhline(-band, color="grey", linewidth=0.5, linestyle=":")
        ax.set_title(f"h={h}-step-ahead forecast ACF")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Sample autocorrelation")
        ax.set_ylim(-0.1, 1.0)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def _autocorr(x: np.ndarray, max_lag: int = 250) -> np.ndarray:
    """Sample autocorrelation function of a 1-D series."""
    x = x - x.mean()
    var = float(np.var(x))
    if var <= 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        if n - k < 5:
            out[k] = 0.0
        else:
            out[k] = float(np.sum(x[k:] * x[:-k]) / ((n - k) * var))
    return out


def _cross_sectional_table(horizon: int, loss_name: str = "mse") -> None:
    """Produce cross-sectional aggregate (mean and median) loss-ratio tables."""
    cfg = load_config()
    out_dir = resolve(cfg.paths.outputs_tables)
    path = out_dir / f"loss_ratio_h{horizon}_{loss_name}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    # Each row is one stock; aggregate across rows
    mean_agg = cross_sectional_aggregate([df.loc[[t]] for t in df.index], "mean")
    median_agg = cross_sectional_aggregate([df.loc[[t]] for t in df.index], "median")
    combined = pd.concat([mean_agg, median_agg], axis=0)
    combined.index = ["mean", "median"]
    combined.to_csv(out_dir / f"cross_sectional_h{horizon}_{loss_name}.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="AAPL")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("appendix_tables", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)
    fig_dir = resolve(cfg.paths.outputs_figures)

    # 1. Table A.5: HAR / HAR-X parameter estimates
    log.info("Computing Table A.5 (HAR/HAR-X estimates for %s)", args.stock)
    tab_a5 = _table_a5(args.stock, horizon=1)
    tab_a5.to_csv(out_dir / f"table_A5_{args.stock}_h1.csv")
    log.info("Saved table_A5_%s_h1.csv", args.stock)

    # 2. Cross-sectional aggregates
    log.info("Computing cross-sectional aggregates")
    for h in cfg.forecast.horizons:
        for loss in ("mse", "qlike"):
            _cross_sectional_table(h, loss)
    log.info("Cross-sectional aggregates saved")

    # 3. Figure 8: ACF
    log.info("Building Figure 8 ACF plot")
    try:
        fig = _figure_8(args.stock)
        fig.savefig(fig_dir / f"acf_{args.stock}.pdf", bbox_inches="tight")
        log.info("Saved acf_%s.pdf", args.stock)
    except Exception as exc:  # noqa: BLE001
        log.warning("Figure 8 failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
