"""
Stage 25 — Persist the validation-selected regularisation penalty per stock.

Closes the one appendix gap where the "the optimum sits in the interior of the
grid" claim for the regularised models was asserted but not recorded: the
fitted-model diagnostics are not persisted in the prediction pickles, so the
chosen penalties were never written down.

This does a one-shot refit (no rolling) of Ridge, Lasso and Elastic Net on the
main 70/10/20 split for every stock at h=1 on M_ALL, selecting the penalty on
the validation block *exactly* as the main pipeline does — the model factories,
feature columns and split are imported from the orchestrator, so the selected
penalties are identical to those behind the headline results.

Output: outputs/tables/selected_lambda.csv  (stock, model, lambda, l1_ratio).
For each fit it also prints whether the selected lambda is interior to the
10^[-5, 2] grid (i.e. not pinned to a boundary).

Usage:
    python scripts/25_selected_lambda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, set_global_seed
from src.data.feature_engineering import load_feature_matrix, time_split
from src.pipeline.orchestrator import _ml_columns, _make_regularized_factories

_LOG = get_logger("selected_lambda")

MODELS = ["RR", "LA", "EN"]   # ridge, lasso, elastic net


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    stocks = cfg.data.stocks
    factories = _make_regularized_factories(cfg)

    rows = []
    for stk in stocks:
        feats = load_feature_matrix(stk, "M_ALL", 1)
        train, val, _ = time_split(
            feats, cfg.data.split.train_frac, cfg.data.split.val_frac
        )
        ml_cols = _ml_columns(feats, "M_ALL")
        train_ml, val_ml = train[ml_cols], val[ml_cols]
        X_tr, y_tr = train_ml.drop(columns=["y"]), train_ml["y"]
        X_vl, y_vl = val_ml.drop(columns=["y"]), val_ml["y"]

        for label in MODELS:
            model = factories[label]()
            model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            grid = list(model.alpha_grid)
            lo_b, hi_b = float(grid[0]), float(grid[-1])
            lam = float(model.diagnostics.alpha)
            l1 = model.diagnostics.l1_ratio
            l1 = float(l1) if l1 is not None else np.nan
            interior = (lam > lo_b) and (lam < hi_b)

            rows.append({"stock": stk, "model": label, "lambda": lam, "l1_ratio": l1})
            l1_str = f" l1_ratio={l1:.2f}" if l1 == l1 else ""  # NaN-safe
            print(
                f"{stk:5s} {label:3s}  lambda={lam:.6g}{l1_str}   "
                f"grid=[{lo_b:.0e}, {hi_b:.0e}]  ->  "
                + ("INTERIOR (not at boundary)" if interior else "AT BOUNDARY")
            )
            _LOG.info("%s %s lambda=%.6g interior=%s", stk, label, lam, interior)

    df = pd.DataFrame(rows)[["stock", "model", "lambda", "l1_ratio"]]
    out = Path("outputs/tables/selected_lambda.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
