#!/bin/bash
# sweep (CPU, one process per env) -> exact-solver eval on GPU 0 -> F3 frontier plots
set -x
cd "$(dirname "$0")"
PY=../../.venv/bin/python
mkdir -p sweep
for e in 1 2 3 4 6; do $PY sweep.py 10 $e --supplier-below 20 > sweep/sweep_env$e.log 2>&1 & done
$PY sweep.py 1 5 --customer-below 50 --supplier-below 20 > sweep/sweep_env5.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY eval.py sweep > sweep/eval.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY ../../paper-stuff/plots/scripts/pareto-frontier-baseline.py > sweep/plot.log 2>&1
echo DONE
