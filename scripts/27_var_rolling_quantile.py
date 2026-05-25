"""
Stage 27 — Coverage-valid VaR via an expanding standardised-residual quantile.

The fixed-quantile FHS in `var_2020_2024.csv` over-breaches badly on the
2020–2024 stress window (both HAR and ML fail Kupiec/Christoffersen) because the
residual quantile is frozen on the calm 2016–2019 sample. This re-runs the same
HAR-vs-best-network comparison with `expanding_quantile_fhs`, which recomputes
the quantile on an expanding window updated through each test date (daily
recalibration, which is >= the required annual cadence) and standardises test
residuals by the same forecast that scales the VaR. The goal is a back-test that
achieves valid coverage, so the HAR-vs-ML comparison rests on a passing test.

Setup mirrors `var_2020_2024`: 2020–2024 window (train_end = 2019-12-31), HAR vs
the best one-day network per stock (AAPL→NN2, AMZN→NN3, JPM→NN2), alpha in
{0.05, 0.01}.

Output: outputs/tables/var_rolling_quantile.csv with breaches, tick loss and
Kupiec + Christoffersen p-values per (stock, alpha, model).

Usage:
    python scripts/27_var_rolling_quantile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.pipeline.orchestrator import load_results
from src.evaluation.value_at_risk import expanding_quantile_fhs

BEST_NN = {"AAPL": "NN2_ensemble", "AMZN": "NN3_ensemble", "JPM": "NN2_ensemble"}
TRAIN_END = pd.Timestamp("2019-12-31")
ALPHAS = (0.05, 0.01)


def main() -> int:
    cf = load_results("predictions_covid_full_M_ALL_h1.pkl")
    rows = []
    for stk in BEST_NN:
        run = [x for x in cf if x.ticker == stk and x.horizon == 1][0]
        inter = pd.read_parquet(f"data/intermediate/{stk}_rv.parquet")
        ret, rv = inter["ret"], inter["RV"]
        models = {"HAR": run.predictions["HAR"],
                  f"best-NN ({BEST_NN[stk]})": run.predictions[BEST_NN[stk]]}
        for alpha in ALPHAS:
            for label, fc in models.items():
                res = expanding_quantile_fhs(ret, fc, TRAIN_END, alpha,
                                             rv_realised=rv, recalibrate_every=1)
                rows.append({
                    "stock": stk, "alpha": alpha, "model": label, "n": res.n,
                    "expected_hits": res.expected_hits, "observed_hits": res.observed_hits,
                    "hit_rate": res.observed_hits / res.n,
                    "quantile_loss": res.quantile_loss,
                    "kupiec_p": res.kupiec_pvalue,
                    "christoffersen_p": res.christoffersen_pvalue,
                })

    df = pd.DataFrame(rows)
    out = Path("outputs/tables/var_rolling_quantile.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
