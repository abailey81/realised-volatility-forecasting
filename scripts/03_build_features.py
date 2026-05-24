"""
Stage 3 — build the M_HAR and M_ALL feature matrices for every
(stock, horizon) combination.

The feature engineering reads:

* ``data/intermediate/<ticker>_rv.parquet`` (produced by stage 1), and
* ``data/macro/fred.parquet`` / ``hsi.parquet`` (produced by stage 2).

and writes per-combination parquet files to ``data/features/``.

Usage:
    python scripts/03_build_features.py
    python scripts/03_build_features.py --feature-sets M_HAR
    python scripts/03_build_features.py --horizons 1 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config
from src.data.feature_engineering import build_features, save_feature_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feature matrices.")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--feature-sets", nargs="*",
                        default=["M_HAR", "M_ALL"])
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("build_features", level=cfg.project.log_level)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    horizons = args.horizons if args.horizons else cfg.forecast.horizons

    for ticker in stocks:
        for fset in args.feature_sets:
            for h in horizons:
                try:
                    log.info("=== %s | %s | h=%d ===", ticker, fset, h)
                    df = build_features(ticker, feature_set=fset, horizon=h)
                    save_feature_matrix(df, ticker, fset, h)
                except Exception as exc:  # noqa: BLE001
                    log.error("Failed for %s|%s|h=%d: %s",
                              ticker, fset, h, exc, exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
