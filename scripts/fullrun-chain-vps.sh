#!/usr/bin/env bash
# Parallel 9,999-user run: both folds fit concurrently, then sharded score, then report.
# Checkpoints live in run-vps/fold-*/boost-checkpoint.pkl, so any restart resumes training.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
cd "$REPO/evaluation/submission-20260919"
PY=upstream/.venv/bin/python
SCORE_SHARDS=${SCORE_SHARDS:-8}

fit() {
  echo "[chain] $(date -Is) fit fold=$1 threads=8 begins"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" run.py \
    --out run-vps --stage fit --threads 8 --folds "$1"
}
score() {
  echo "[chain] $(date -Is) score shard=$1 threads=2 begins"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" run.py \
    --out run-vps --stage score --threads 2 --shard "$1"
}

echo "[chain] $(date -Is) parallel fit begins (two folds)"
fit 0 & fold0=$!
fit 1 & fold1=$!
fit_status=0
wait "$fold0" || fit_status=1
wait "$fold1" || fit_status=1
if [ "$fit_status" -ne 0 ]; then
  echo "[chain] $(date -Is) fit FAILED"
  exit 1
fi

echo "[chain] $(date -Is) sharded score begins ($SCORE_SHARDS shards)"
pids=()
for shard in $(seq 0 $((SCORE_SHARDS - 1))); do
  score "$shard/$SCORE_SHARDS" & pids+=($!)
done
score_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || score_status=1
done
if [ "$score_status" -ne 0 ]; then
  echo "[chain] $(date -Is) score FAILED"
  exit 1
fi

echo "[chain] $(date -Is) report begins"
"$PY" run.py --out run-vps --stage report
echo "[chain] $(date -Is) complete"
