"""
Stage 18 — critique-evidence pack.

Produces the marginal tables and figures that turn the four-critique §4
structure into something a reader can verify line-by-line:

* **Critique 1.1 / 1.2 — M_HAR ML and path decomposition.**
  Builds the M_HAR ML loss-ratio table (mirror of paper Table 2 right side)
  and the regularisation-vs-nonlinearity decomposition.

* **Critique 1.4 — out-of-sample ACF (forecast vs realised RV).**
  The paper's Figure 8 compares in-sample fitted ACFs. The over-smoothing
  test requires comparing forecast ACF to realised-RV ACF on the test set.

* **Critique 2.4 — pairwise forecast-error correlation heatmap.**
  If model errors correlate at r > 0.85, the models are forecasting nearly
  the same thing. Cheap and visual.

* **Critique 3.3 — decile-10 specifically.**
  Extracts the per-stock relative MSE in the highest RV decile for direct
  comparison with the paper's 7.38% number on page 21.

All outputs land in ``outputs/tables/`` and ``outputs/figures/``.

Usage:
    python scripts/18_critique_evidence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_logger, load_config, resolve
from src.evaluation.metrics import mse_loss
from src.pipeline.orchestrator import load_results


# -------- helpers --------

def _merge(files: list[str], horizon: int, feature_set: str | None = None):
    merged: dict[tuple, "object"] = {}
    for f in files:
        try:
            for r in load_results(f):
                if r.horizon != horizon:
                    continue
                if feature_set and r.feature_set != feature_set:
                    continue
                key = (r.ticker, r.feature_set, r.horizon)
                if key in merged:
                    merged[key].predictions.update(r.predictions)
                else:
                    merged[key] = r
        except FileNotFoundError:
            continue
    return list(merged.values())


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - np.mean(x)
    var = float(np.var(x))
    if var <= 0 or len(x) < max_lag + 5:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.sum(x[k:] * x[:-k]) / ((len(x) - k) * var))
    return out


# -------- Critique 1.1 / 1.2: M_HAR ML + path decomposition --------

def m_har_ml_table(cfg, files: list[str], horizon: int = 1) -> pd.DataFrame:
    """MSE ratios vs HAR for ML methods on M_HAR predictor set (3 RV lags)."""
    runs = _merge(files, horizon=horizon, feature_set="M_HAR")
    if not runs:
        return pd.DataFrame()
    rows: dict[str, dict[str, float]] = {}
    for r in runs:
        rows[r.ticker] = {}
        for m, p in r.predictions.items():
            idx = r.y_true.index.intersection(p.index)
            rows[r.ticker][m] = float(np.mean(mse_loss(r.y_true.loc[idx].to_numpy(),
                                                        p.loc[idx].to_numpy())))
    df = pd.DataFrame(rows).T
    if "HAR" in df.columns:
        df_ratio = df.div(df["HAR"], axis=0)
    else:
        df_ratio = df
    return df_ratio


def path_decomposition(cfg, horizon: int = 1) -> pd.DataFrame:
    """Replicate the paper's M_HAR → M_ALL gain decomposition by path.

    Paths:
        (a) HAR-MHAR (baseline)              → reference
        (b) HAR-X on M_ALL (unregularised)   → does feature inclusion help OLS?
        (c) EN on M_ALL                      → regularised linear
        (d) RF on M_ALL                      → tree nonlinearity
        (e) best NN on M_ALL (ensembled)     → deep NN

    Each successive ratio quantifies the marginal contribution.
    """
    out_dir = resolve(cfg.paths.outputs_tables)
    ratio_path = out_dir / f"loss_ratio_h{horizon}_mse.csv"
    if not ratio_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(ratio_path, index_col=0)
    rows = []
    for ticker in df.index:
        row = df.loc[ticker]
        har_x = row.get("HAR-X", np.nan)
        en = row.get("EN", np.nan)
        rf = row.get("RF", np.nan)
        # Best NN: take min over the four ensemble + four top1 variants
        nn_cols = [c for c in row.index if c.startswith("NN") and (c.endswith("_ensemble") or c.endswith("_top1"))]
        best_nn = float(row[nn_cols].min()) if nn_cols else np.nan
        # Each "leg" reports the model's MSE ratio vs HAR. Path-marginal in
        # additive percentage-points relative to the HAR baseline.
        rows.append({
            "ticker": ticker,
            "har_x_overfit_pct": (har_x - 1.0) * 100,                       # +ve = HAR-X worse
            "regularisation_gain_pct": (1.0 - en) * 100,                    # EN vs HAR
            "tree_marginal_pct": (en - rf) * 100,                            # RF vs EN
            "deep_nn_marginal_pct": (rf - best_nn) * 100,                    # best NN vs RF
            "total_ml_gain_pct": (1.0 - best_nn) * 100,                       # best NN vs HAR
            "regularisation_share": ((1.0 - en) / (1.0 - best_nn)) if best_nn < 1 else np.nan,
        })
    return pd.DataFrame(rows)


# -------- Critique 1.4: OOS ACF (forecast vs realised) --------

def oos_acf_plot(cfg, horizon: int, max_lag: int = 60) -> plt.Figure | None:
    """Compare ACF of OOS forecasts to ACF of realised RV on the test set.

    Paper Figure 8 plots in-sample fitted ACFs. That answers a different
    question. The test of "ML captures long memory" is whether forecast
    ACF on the OOS sample matches realised ACF on the OOS sample.
    """
    files = ["predictions_har_MALL.pkl",
             "predictions_ml_MALL.pkl",
             "predictions_nn_MALL.pkl"]
    runs = _merge(files, horizon=horizon, feature_set="M_ALL")
    if not runs:
        return None
    # Use AAPL (paper's illustration choice) for parity
    aapl = next((r for r in runs if r.ticker == "AAPL"), None)
    if aapl is None:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    realised = aapl.y_true.dropna().to_numpy()
    ax.plot(np.arange(max_lag + 1), _autocorr(realised, max_lag),
            color="black", linewidth=1.4, label="Realised RV (OOS)")
    for label, color in (("HAR", "tab:blue"),
                          ("RF", "tab:green"),
                          ("NN2_ensemble", "tab:red")):
        if label not in aapl.predictions:
            continue
        forecast = aapl.predictions[label].dropna().to_numpy()
        ax.plot(np.arange(max_lag + 1), _autocorr(forecast, max_lag),
                linestyle="--", color=color, linewidth=1.0,
                label=f"{label.replace('_ensemble', '$^{10}$')} forecast")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Sample autocorrelation")
    ax.set_title(f"Out-of-sample ACF: realised vs forecast (AAPL, h={horizon})")
    ax.set_ylim(-0.15, 1.05)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# -------- Critique 2.4: pairwise OOS error correlation heatmap --------

def error_correlation(cfg, horizon: int) -> tuple[pd.DataFrame, plt.Figure | None]:
    files = ["predictions_har_MALL.pkl",
             "predictions_ml_MALL.pkl",
             "predictions_nn_MALL.pkl"]
    runs = _merge(files, horizon=horizon, feature_set="M_ALL")
    if not runs:
        return pd.DataFrame(), None

    # Pool errors across stocks (stack residuals)
    err_by_model: dict[str, list[float]] = {}
    for r in runs:
        for m, p in r.predictions.items():
            idx = r.y_true.index.intersection(p.index)
            err = (r.y_true.loc[idx].to_numpy() - p.loc[idx].to_numpy())
            err_by_model.setdefault(m, []).extend(err.tolist())
    err_df = pd.DataFrame({k: pd.Series(v) for k, v in err_by_model.items()})
    err_df = err_df.dropna(how="any")
    corr = err_df.corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Pairwise OOS error correlation")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(f"Pairwise out-of-sample forecast-error correlation (h={horizon})")
    fig.tight_layout()
    return corr, fig


# -------- Critique 3.3: decile-10 extract --------

def decile_10_table(cfg, horizon: int) -> pd.DataFrame:
    """Extract per-stock relative MSE in the highest RV decile (paper p. 21)."""
    out_dir = resolve(cfg.paths.outputs_tables)
    rows = []
    for ticker in cfg.data.stocks:
        path = out_dir / f"decile_ratios_{ticker}_h{horizon}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if "D10" not in df.index:
            continue
        d10 = df.loc["D10"]
        rows.append({"ticker": ticker, **{m: float(d10[m]) for m in d10.index}})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# -------- driver --------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("critique_evidence", level=cfg.project.log_level)
    out_dir = resolve(cfg.paths.outputs_tables)
    fig_dir = resolve(cfg.paths.outputs_figures)

    # --- Critique 1.1 — M_HAR ML loss-ratio table ---
    mhar_files = ["predictions_har_MHAR.pkl",
                  "predictions_ml_MHAR.pkl",
                  "predictions_nn_MHAR.pkl"]
    for h in args.horizons:
        df = m_har_ml_table(cfg, mhar_files, horizon=h)
        if not df.empty:
            df.to_csv(out_dir / f"critique_MHAR_loss_ratio_h{h}_mse.csv")
            log.info("Saved M_HAR ML loss ratios for h=%d (%d rows × %d cols)",
                     h, len(df), df.shape[1])

    # --- Critique 1.2 — path decomposition ---
    for h in args.horizons:
        df = path_decomposition(cfg, horizon=h)
        if not df.empty:
            df.to_csv(out_dir / f"critique_path_decomposition_h{h}.csv", index=False)
            log.info("Saved path decomposition h=%d", h)

    # --- Critique 1.4 — OOS ACF (forecast vs realised) ---
    for h in args.horizons:
        fig = oos_acf_plot(cfg, horizon=h)
        if fig is not None:
            fig.savefig(fig_dir / f"critique_oos_acf_h{h}.pdf", bbox_inches="tight")
            plt.close(fig)
            log.info("Saved OOS ACF figure h=%d", h)

    # --- Critique 2.4 — error correlation heatmap ---
    for h in args.horizons:
        corr, fig = error_correlation(cfg, horizon=h)
        if not corr.empty:
            corr.to_csv(out_dir / f"critique_error_correlation_h{h}.csv")
        if fig is not None:
            fig.savefig(fig_dir / f"critique_error_correlation_h{h}.pdf", bbox_inches="tight")
            plt.close(fig)
            log.info("Saved error correlation heatmap h=%d", h)

    # --- Critique 3.3 — decile-10 table ---
    for h in args.horizons:
        df = decile_10_table(cfg, horizon=h)
        if not df.empty:
            df.to_csv(out_dir / f"critique_decile10_h{h}.csv")
            log.info("Saved decile-10 table h=%d", h)

    log.info("Critique evidence pack complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
