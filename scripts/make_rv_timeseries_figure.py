"""
Build Figure A1 — three-stock annualised realised-volatility time series
(2016–2024) with a shaded 2020–2024 stress-evaluation window.

Transform: annualised volatility = sqrt(252 * RV) * 100  (percent), identical to
src.visualization.plots.plot_rv_time_series.

The shaded band marks the 2020–2024 stress RE-SPLIT analysed in Section 4 — the
separate evaluation that brings the 2020 crash and 2022 drawdown out-of-sample.
It is NOT the main chronological 70/10/20 test set, which is the calm 2023–2024
tail; hence the wording "stress-evaluation window", not "test window".

Output: outputs/figures/rv_time_series.pdf (300-dpi vector; overwrites existing).

Usage:
    python scripts/make_rv_timeseries_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.visualization.plots import plot_rv_time_series, save_figure

STOCKS = ["AAPL", "AMZN", "JPM"]
STRESS_WINDOW = ("2020-01-01", "2024-12-31")
STRESS_LABEL = "2020-2024 stress-evaluation window (Section 4)"


def main() -> int:
    rv_dict = {
        t: pd.read_parquet(f"data/intermediate/{t}_rv.parquet")["RV"]
        for t in STOCKS
    }

    fig = plot_rv_time_series(
        rv_dict,
        title="",                                   # caption carries the title in LaTeX
        stress_window=STRESS_WINDOW,
        stress_label=STRESS_LABEL,
        xlabel="Year",
        ylabel="Annualised realised volatility (%)",
        linewidth=0.8,
        legend_loc="upper left",                    # keep clear of the band caption
    )
    path = save_figure(fig, "rv_time_series.pdf")

    print(f"Wrote {path}")
    print(f"File exists: {Path(path).exists()}")
    print("Y-axis transform: annualised volatility = sqrt(252 * RV) * 100  (percent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
