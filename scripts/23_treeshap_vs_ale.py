"""
Stage 24 — TreeSHAP vs ALE comparison for the tree-based ML models.

The paper uses ALE (Apley & Zhu 2020) for interpretability because ALE
is correlation-robust — it does not require integration against
marginal distributions of the unconditional X, which biases standard
partial-dependence plots when features are correlated. ALE accumulates
*local* effects within quantile bins of the feature of interest.

TreeSHAP (Lundberg et al. 2020) is a different global-attribution
method: a fast exact computation of Shapley values for tree ensembles.
Shapley values are cooperative-game-theoretic — they ascribe to each
feature its average marginal contribution to the model's prediction
across all possible coalitions. Shapley values are correlation-AWARE
via the marginal-contribution decomposition.

These two estimators answer slightly different questions:

* ALE asks: "How does the model's predicted RV change as we *vary*
  feature j, controlling for the empirical correlation structure?"
* TreeSHAP asks: "How much of a *single* prediction's deviation from
  the global average is attributable to each feature, integrated over
  the conditional distribution implied by the tree structure?"

In a low-correlation feature set the two will rank features
similarly. On the M_ALL feature set, with strongly autocorrelated
HAR lags (RVD/RVW/RVM are partial sums) and correlated macro factors
(VIX, IV both share an aggregate risk component), the rankings can
disagree — and the disagreement is informative about whether the tree
model is exploiting truly independent feature contributions or merely
proxy-aware overlap.

This stage:

1. Loads the fitted RF + GB models from `predictions_ml_MALL.pkl`'s
   training metadata. Since predictions pickle does not carry model
   objects, we *refit* the trees on `train + val` of the M_ALL design
   matrix exactly as the paper does (same fixed-window training step
   that produced the original predictions).
2. Computes TreeSHAP global feature importance on the test set.
3. Computes the paper's ALE-based variable importance on the same
   design matrix.
4. Writes a side-by-side comparison CSV per (stock, horizon, model)
   and a pooled-across-models summary.

Outputs
-------
* `treeshap_vi_{ticker}_h{h}.csv`            (RF + GB importance per stock)
* `treeshap_vs_ale_{ticker}_h{h}.csv`        (joined RF, GB, ALE columns)
* `treeshap_summary.csv`                     (pooled cross-section)
* `outputs/figures/treeshap_vs_ale.pdf`      (bar-chart comparison)

Usage
-----
    python scripts/23_treeshap_vs_ale.py
    python scripts/23_treeshap_vs_ale.py --horizons 1 --stocks AAPL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_logger, load_config, resolve, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.evaluation.ale import (
    accumulated_local_effects, variable_importance_from_ale,
)
from src.pipeline.orchestrator import _make_tree_factories


def _treeshap_importance(model, X_test: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| over the test rows, per feature."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    mean_abs = np.mean(np.abs(sv), axis=0)
    return pd.Series(mean_abs, index=X_test.columns, name="treeshap")


def _ale_importance_for_tree(model, X_test: pd.DataFrame, features) -> pd.Series:
    """ALE-based variable importance for a tree model on ``features``."""
    ale_results = {}
    for f in features:
        try:
            res = accumulated_local_effects(
                predict_fn=lambda X, m=model: m.predict(X),
                X=X_test, feature=f, num_bins=40, centred=True,
            )
            ale_results[f] = res
        except Exception:
            continue
    vi = variable_importance_from_ale(ale_results)
    return vi.rename("ale")


def _fit_tree(name: str, X_tr, y_tr, X_vl, y_vl, factories):
    model = factories[name]()
    try:
        model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
    except TypeError:
        model.fit(X_tr, y_tr)
    return model


def _underlying_sklearn(forecaster):
    """Return the underlying sklearn estimator from our Forecaster wrapper.

    The wrappers store the trained estimator on ``self.model_``."""
    return getattr(forecaster, "model_", None) or getattr(forecaster, "_fitted", forecaster)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 22])
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-set", default="M_ALL")
    parser.add_argument("--n-test-shap", type=int, default=300,
                        help="Subsample of test rows for SHAP (full test set "
                             "can be too slow for GB on M_ALL).")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("treeshap", level=cfg.project.log_level)
    out_tables = resolve(cfg.paths.outputs_tables)
    out_figs = resolve(cfg.paths.outputs_figures)
    out_figs.mkdir(parents=True, exist_ok=True)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    factories = _make_tree_factories(cfg)

    summary_rows = []
    for ticker in stocks:
        for h in args.horizons:
            try:
                X_all = load_feature_matrix(ticker, args.feature_set, h)
            except FileNotFoundError:
                log.warning("[%s|h=%d] feature matrix missing — skipping",
                            ticker, h)
                continue
            features = [c for c in X_all.columns if c != "y"]
            train, val, test = time_split(
                X_all,
                train_frac=cfg.data.split.train_frac,
                val_frac=cfg.data.split.val_frac,
            )
            X_tr, y_tr = train[features], train["y"]
            X_vl, y_vl = val[features], val["y"]
            full_train = pd.concat([train, val])
            X_full, y_full = full_train[features], full_train["y"]
            X_te = test[features]

            # Subsample test for SHAP (matrix gets large on full M_ALL test)
            if len(X_te) > args.n_test_shap:
                X_te_s = X_te.iloc[:args.n_test_shap]
            else:
                X_te_s = X_te

            joined_rows = []
            for tree_name in ["RF", "GB"]:
                log.info("[%s|h=%d|%s] fitting tree on train+val (%d obs)",
                         ticker, h, tree_name, len(full_train))
                forecaster = factories[tree_name]()
                try:
                    forecaster.fit(X_full, y_full, X_val=X_vl, y_val=y_vl)
                except TypeError:
                    forecaster.fit(X_full, y_full)
                sk = _underlying_sklearn(forecaster)
                if sk is None:
                    log.warning("  could not access underlying sklearn estimator")
                    continue
                # TreeSHAP and ALE on the same test sample
                shap_vi = _treeshap_importance(sk, X_te_s)
                ale_vi = _ale_importance_for_tree(sk, X_te_s, features)
                # Normalise SHAP for direct comparison with ALE-VI (sum=1)
                shap_vi_norm = shap_vi / (shap_vi.sum() or 1.0)
                tab = pd.DataFrame({
                    "treeshap": shap_vi_norm,
                    "ale_vi": ale_vi,
                }).fillna(0.0)
                tab["model"] = tree_name
                tab["ticker"] = ticker
                tab["horizon"] = h
                tab.index.name = "feature"
                joined_rows.append(tab)
                log.info("[%s|h=%d|%s] top-3 SHAP: %s | top-3 ALE: %s",
                         ticker, h, tree_name,
                         shap_vi_norm.sort_values(ascending=False).head(3).index.tolist(),
                         ale_vi.sort_values(ascending=False).head(3).index.tolist())
                # Rank correlation
                rank_sp = (shap_vi_norm.rank(ascending=False)
                                       .corr(ale_vi.rank(ascending=False), method="spearman"))
                summary_rows.append({
                    "ticker": ticker, "horizon": h, "model": tree_name,
                    "rank_corr_shap_vs_ale": float(rank_sp),
                    "top_shap_feature": shap_vi_norm.idxmax(),
                    "top_ale_feature": ale_vi.idxmax(),
                })

            if joined_rows:
                joined = pd.concat(joined_rows)
                joined.to_csv(
                    out_tables / f"treeshap_vs_ale_{ticker}_h{h}.csv"
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_tables / "treeshap_summary.csv", index=False)
    log.info("Saved cross-section TreeSHAP-vs-ALE summary (%d rows)", len(summary))
    if not summary.empty:
        log.info("\nRank-correlation summary:\n%s",
                 summary.groupby(["model", "horizon"])["rank_corr_shap_vs_ale"]
                        .agg(["mean", "min", "max"]).to_string())

    # Quick figure: top-8 features comparison at h=1 for the first stock
    figure_target = stocks[0] if stocks else None
    if figure_target:
        try:
            tab = pd.read_csv(
                out_tables / f"treeshap_vs_ale_{figure_target}_h1.csv",
                index_col=0,
            )
            rf_tab = tab[tab["model"] == "RF"].head(20)
            top = rf_tab[["treeshap", "ale_vi"]].sort_values(
                "treeshap", ascending=False).head(8)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            xs = np.arange(len(top))
            w = 0.4
            ax.bar(xs - w/2, top["treeshap"], w, label="TreeSHAP")
            ax.bar(xs + w/2, top["ale_vi"], w, label="ALE-VI")
            ax.set_xticks(xs)
            ax.set_xticklabels(top.index, rotation=45, ha="right")
            ax.set_ylabel("Normalised importance")
            ax.set_title(f"RF — {figure_target} h=1: TreeSHAP vs ALE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_figs / "treeshap_vs_ale.pdf")
            plt.close(fig)
            log.info("Saved figure outputs/figures/treeshap_vs_ale.pdf")
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
