"""
Stage 26 — Is HAR's VaR edge statistically significant?

`var_2020_2024.csv` showed HAR has lower quantile (tick) loss than the best
one-day network on the 2020–2024 window, but that was a point comparison. This
turns it into a rigorous claim: it exposes the per-day tick-loss series for both
models and runs a one-sided Diebold–Mariano test of equal tick loss, with
H1: HAR has the *lower* loss (alternative = "less").

Setup mirrors `var_2020_2024` exactly: filtered historical simulation (fixed
in-sample quantile), 2020–2024 test window (train_end = 2019-12-31), HAR vs the
best one-day network per stock (AAPL→NN2, AMZN→NN3, JPM→NN2), alpha in
{0.05, 0.01}. The DM statistic reuses the module's Newey–West HAC long-run
variance and the Harvey–Leybourne–Newbold (1997) small-sample correction
(horizon = 1), identical to src/evaluation/diebold_mariano.py.

Output: outputs/tables/var_dm.csv  (stock, alpha, dm_stat, p_value).

Usage:
    python scripts/26_var_dm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.pipeline.orchestrator import load_results
from src.evaluation.value_at_risk import filtered_historical_simulation
from src.evaluation.diebold_mariano import _newey_west_lrv

BEST_NN = {"AAPL": "NN2_ensemble", "AMZN": "NN3_ensemble", "JPM": "NN2_ensemble"}
TRAIN_END = pd.Timestamp("2019-12-31")
ALPHAS = (0.05, 0.01)


def _tick_loss(r: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """Per-day Koenker–Bassett pinball loss (same convention as value_at_risk)."""
    hits = (r <= var).astype(float)
    return (alpha - hits) * (r - var)


def _dm_less(d: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """One-sided DM stat (+ p-value) for H1: E[d] < 0, NW HAC + HLN correction.

    Replicates src/evaluation/diebold_mariano.diebold_mariano with
    alternative="less" applied directly to a precomputed loss differential.
    """
    n = len(d)
    mean_d = float(np.mean(d))
    lrv = _newey_west_lrv(d)
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = mean_d / np.sqrt(lrv / n)
    h = horizon
    factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm *= np.sqrt(factor)
    pval = float(stats.t.cdf(dm, df=n - 1))   # H1: HAR (A) has lower loss
    return float(dm), pval


def main() -> int:
    cf = load_results("predictions_covid_full_M_ALL_h1.pkl")
    rows = []
    for stk in BEST_NN:
        run = [x for x in cf if x.ticker == stk and x.horizon == 1][0]
        inter = pd.read_parquet(f"data/intermediate/{stk}_rv.parquet")
        ret, rv = inter["ret"], inter["RV"]
        har_fc = run.predictions["HAR"]
        nn_fc = run.predictions[BEST_NN[stk]]

        for alpha in ALPHAS:
            res_har = filtered_historical_simulation(ret, har_fc, TRAIN_END, alpha, rv_realised=rv)
            res_nn = filtered_historical_simulation(ret, nn_fc, TRAIN_END, alpha, rv_realised=rv)
            idx = res_har.var_forecast.index.intersection(res_nn.var_forecast.index)
            r = ret.loc[idx].to_numpy()
            l_har = _tick_loss(r, res_har.var_forecast.loc[idx].to_numpy(), alpha)
            l_nn = _tick_loss(r, res_nn.var_forecast.loc[idx].to_numpy(), alpha)
            d = l_har - l_nn                       # < 0  => HAR lower loss
            dm_stat, p = _dm_less(d, horizon=1)
            sig = "HAR sig. lower at 5%" if p < 0.05 else ("HAR lower, n.s." if dm_stat < 0 else "HAR not lower")
            rows.append({"stock": stk, "alpha": alpha, "dm_stat": dm_stat, "p_value": p})
            print(f"{stk:5s} alpha={alpha:.2f}  dm_stat={dm_stat:+.3f}  p={p:.4f}   {sig}")

    df = pd.DataFrame(rows)[["stock", "alpha", "dm_stat", "p_value"]]
    out = Path("outputs/tables/var_dm.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
