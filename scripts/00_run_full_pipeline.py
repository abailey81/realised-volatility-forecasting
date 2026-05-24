"""
End-to-end driver: runs stages 1 through 9 in order.

Use this as the canonical reproducibility entry point. Each underlying
stage's CLI flags are not forwarded; for partial / custom runs, invoke
the stage scripts individually.

Usage:
    python scripts/00_run_full_pipeline.py
    python scripts/00_run_full_pipeline.py --skip-nn          # development
    python scripts/00_run_full_pipeline.py --stocks AAPL JPM
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config


STAGES = [
    ("Preprocess data",          "scripts/01_preprocess_data.py"),
    ("Download macro features",  "scripts/02_download_macro.py"),
    ("Build features",           "scripts/03_build_features.py"),
    ("Train HAR family",         "scripts/04_train_har.py"),
    ("Train regularised + trees","scripts/05_train_ml.py"),
    ("Train neural networks",    "scripts/06_train_nn.py"),
    ("DM tests + MCS",           "scripts/07_run_tests.py"),
    ("Compute ALE",              "scripts/08_compute_ale.py"),
    ("Generate outputs",         "scripts/09_generate_outputs.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip stage 6 (NNs) for quick smoke runs.")
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip stage 2 (FRED download) if cache exists.")
    parser.add_argument("--stocks", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("pipeline", level=cfg.project.log_level)

    root = Path(__file__).resolve().parent.parent
    for label, rel_path in STAGES:
        if args.skip_nn and "nn" in rel_path.lower():
            log.info("SKIPPING: %s", label)
            continue
        if args.skip_macro and "macro" in rel_path.lower():
            log.info("SKIPPING: %s", label)
            continue
        log.info("=== %s ===", label)
        t0 = time.time()
        cmd = [sys.executable, str(root / rel_path)]
        if args.stocks and "--stocks" in open(root / rel_path).read():
            cmd += ["--stocks", *args.stocks]
        rc = subprocess.call(cmd, cwd=str(root))
        log.info("  %s done in %.1fs (rc=%d)", label, time.time() - t0, rc)
        if rc != 0:
            log.error("Stage failed: %s", label)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
