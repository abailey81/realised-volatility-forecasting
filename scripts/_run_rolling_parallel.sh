#!/usr/bin/env bash
# Parallel driver for the rolling-trees job: one process per (stock, horizon)
# cell so all 9 GB computations (single-core each) run concurrently instead of
# in series. Forest threads are throttled so the RF/BG phases of 9 processes
# don't oversubscribe the 16 logical cores.
set -u
cd "$(dirname "$0")/.."

# Throttle nested parallelism: each process gets ~2 loky workers, BLAS=1.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2

PY=python3; command -v python >/dev/null 2>&1 && PY=python
PARTS=outputs/tables/_rolling_parts
mkdir -p "$PARTS"

echo "launching 9 parallel cells at $(date '+%H:%M:%S')"
pids=()
for stk in AAPL AMZN JPM; do
  for h in 1 5 22; do
    $PY scripts/24_rolling_trees.py --stocks "$stk" --horizons "$h" \
        --out "$PARTS/part_${stk}_h${h}.csv" \
        > "$PARTS/log_${stk}_h${h}.log" 2>&1 &
    pids+=($!)
    echo "  started $stk h=$h (pid $!)"
  done
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done
echo "ALL_CELLS_DONE at $(date '+%H:%M:%S') (failures=$fail)"
exit 0
