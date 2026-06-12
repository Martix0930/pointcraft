#!/usr/bin/env bash
# exp_004 Stage B — multi-seed confirm at the chosen reg (wd=1e-3), the actual result.
# Stage A's seed-0 run (stageA_wd1e-3) already serves as seed 0, so this runs only
# seeds 1 and 2 (~2 new runs). Same口径 as Stage A: full precision (no AMP), held-out
# every 50 tile-steps, train_iou every 200, 10 tiles, 200 epochs.
#
# Measures the PRIMARY success criterion: cross-seed peak-strict spread (max-min over the
# 3 seeds) vs fork-1's ~0.03. Isolated + GPU-gated to prevent cascade crashes.
set -u
cd "$(dirname "$0")/.."

TRAIN="outputs/m0/g1/09LD1878.npz outputs/m0/g1/09LD1845.npz outputs/m0/g1/09LD1843.npz \
outputs/m0/g1/09LD1867.npz outputs/m0/g1/09LD2818.npz outputs/m0/g1/09LD1846.npz \
outputs/m0/g1/09LD1885.npz outputs/m0/g1/09LD1886.npz outputs/m0/g1/09LD1897.npz \
outputs/m0/g1/09LD2807.npz"
VAL="outputs/m0/g1/09LD2814.npz"
B3="outputs/g1/b3/09LD2814_b3_m5.json"
WD=1e-3

wait_gpu_clean() {
  local tries=0
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$used" ] && used=0
    if [ "$used" -lt 500 ]; then break; fi
    tries=$((tries+1))
    if [ "$tries" -gt 60 ]; then echo "[gate] GPU still ${used} MiB after 60 tries, proceeding"; break; fi
    sleep 5
  done
  sleep 8
  echo "[gate] GPU clean (used=${used} MiB) — starting next run"
}

for SEED in 1 2; do
  EXP="experiments/exp_004_m2_scaleup/stageB_wd${WD}_s${SEED}"
  echo "================ Stage B  wd=${WD}  seed=${SEED}  ->  ${EXP} ================"
  date
  .venv/Scripts/python scripts/run_m2_generalize.py \
    --train ${TRAIN} \
    --val ${VAL} \
    --b3-json ${B3} \
    --epochs 200 --eval-every-steps 50 --train-eval-every-steps 200 \
    --weight-decay ${WD} --seed ${SEED} --no-amp --no-viz \
    --exp "${EXP}"
  echo "---- done seed=${SEED} (exit $?) ----"
  date
  wait_gpu_clean
done

echo "================ Stage B seeds complete ================"
