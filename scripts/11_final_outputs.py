"""
Stage 11 — final outputs orchestrator.

Drives the full downstream output generation after stages 1-6 have produced
their prediction pickles. Runs:

* Stage 7 (DM + MCS) for every (horizon, loss) combination.
* Stage 8 (ALE) on AAPL with the headline models.
* Stage 9 (figures + tables) for every (horizon, loss).
* A consolidated cross-horizon summary table.

Usage:
    python scripts/11_final_outputs.py
    python scripts/11_final_outputs.py --skip-ale       # skip the slow ALE step
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve
from src.pipeline.orchestrator import load_results
from src.evaluation.metrics import LOSSES


def _run(cmd: list[str], log) -> int:
    log.info("RUN: %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    log.info("  rc=%d, took %.1fs", rc, time.time() - t0)
    return rc


def _run_parallel(cmds: list[list[str]], log, max_workers: int = 4) -> list[int]:
    """Run multiple subprocess commands in parallel via thread pool.

    Subprocesses are independent OS processes, so a thread pool is fine
    (no GIL contention). ``max_workers`` should be small enough to keep
    each subprocess from over-subscribing the CPU.
    """
    log.info("RUN-PARALLEL (%d jobs, max_workers=%d):", len(cmds), max_workers)
    for c in cmds:
        log.info("  cmd: %s", " ".join(c))
    t0 = time.time()
    rcs: list[int] = [0] * len(cmds)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(subprocess.call, c,
                           cwd=str(Path(__file__).resolve().parent.parent)): i
                for i, c in enumerate(cmds)}
        for fut in as_completed(futs):
            i = futs[fut]
            rcs[i] = fut.result()
    log.info("RUN-PARALLEL done in %.1fs (rcs=%s)", time.time() - t0, rcs)
    return rcs


def _cross_horizon_summary(cfg, loss_name: str) -> pd.DataFrame:
    """Stack per-horizon loss-ratio tables into a long table for the appendix."""
    out_dir = resolve(cfg.paths.outputs_tables)
    rows = []
    for h in cfg.forecast.horizons:
        path = out_dir / f"loss_ratio_h{h}_{loss_name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        for ticker, row in df.iterrows():
            for model, val in row.items():
                rows.append({"horizon": h, "ticker": ticker, "model": model, "ratio_vs_HAR": val})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ale", action="store_true")
    parser.add_argument("--inputs", nargs="*",
                        default=["predictions_har_MALL.pkl",
                                 "predictions_ml_MALL.pkl",
                                 "predictions_nn_MALL.pkl"])
    parser.add_argument("--losses", nargs="*", default=["mse", "qlike"])
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("final_outputs", level=cfg.project.log_level)

    horizons = list(cfg.forecast.horizons)
    losses = args.losses

    # ----- Stage 7 (DM + MCS) -----
    # Run all (horizon × loss) jobs in parallel — each is its own python process.
    stage7_cmds = [
        [sys.executable, "scripts/07_run_tests.py",
         "--horizon", str(h), "--loss", loss, "--inputs", *args.inputs]
        for h in horizons for loss in losses
    ]
    _run_parallel(stage7_cmds, log, max_workers=min(len(stage7_cmds), 6))

    # ----- Stage 8 (ALE) -----
    if not args.skip_ale:
        cmd = [sys.executable, "scripts/08_compute_ale.py",
               "--stock", str(cfg.ale.stock_for_plots),
               "--horizon", "1",
               "--features", *list(cfg.ale.features),
               "--models", *list(cfg.ale.models)]
        _run(cmd, log)

    # ----- Stage 9 (figures + tables) -----
    stage9_cmds = [
        [sys.executable, "scripts/09_generate_outputs.py",
         "--horizon", str(h), "--loss", loss, "--inputs", *args.inputs]
        for h in horizons for loss in losses
    ]
    _run_parallel(stage9_cmds, log, max_workers=min(len(stage9_cmds), 6))

    # ----- Cross-horizon summary -----
    for loss in losses:
        df = _cross_horizon_summary(cfg, loss)
        if not df.empty:
            out = resolve(cfg.paths.outputs_tables) / f"summary_cross_horizon_{loss}.csv"
            df.to_csv(out, index=False)
            log.info("Saved cross-horizon summary: %s (n=%d rows)", out, len(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
