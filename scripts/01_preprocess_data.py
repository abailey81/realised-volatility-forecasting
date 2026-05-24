"""
Stage 1 — preprocess raw minute bars into daily realised measures.

For each ticker in the configuration:

1. Load the raw minute-bar CSV from ``data/raw/``.
2. Restrict to the regular trading session.
3. Apply Barndorff-Nielsen et al. (2009)-style outlier filtering.
4. Resample to 5-minute log returns within each trading day.
5. Compute daily RV, RV+, RV-, RQ, and the open-to-close return.
6. Save the result to ``data/intermediate/<ticker>_rv.parquet``.

Usage:
    python scripts/01_preprocess_data.py
    python scripts/01_preprocess_data.py --stocks AAPL JPM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config, set_global_seed
from src.data.compute_rv import compute_realised_measures, save_realised
from src.data.realised_kernel import compute_realised_kernel, save_realised_kernel


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute daily realised measures.")
    parser.add_argument("--stocks", nargs="*", default=None,
                        help="Tickers to process (default: from config).")
    parser.add_argument("--annualise", action="store_true",
                        help="Multiply RV by 252 before saving.")
    parser.add_argument("--skip-rk", action="store_true",
                        help="Skip realised-kernel computation (faster).")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    log = get_logger("preprocess_data", level=cfg.project.log_level)

    stocks = args.stocks if args.stocks else cfg.data.stocks
    rk_enabled = (not args.skip_rk) and getattr(cfg, "realised_kernel", None) is not None \
                 and getattr(cfg.realised_kernel, "enabled", False)

    for ticker in stocks:
        try:
            log.info("=== Processing %s ===", ticker)
            df = compute_realised_measures(ticker, annualise=args.annualise)
            save_realised(ticker, df)
            if rk_enabled:
                log.info("Computing realised kernel for %s", ticker)
                rk_df = compute_realised_kernel(ticker, sampling_minutes=cfg.realised_kernel.sampling_minutes)
                save_realised_kernel(ticker, rk_df)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed for %s: %s", ticker, exc, exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
