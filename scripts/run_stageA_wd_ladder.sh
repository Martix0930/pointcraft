#!/usr/bin/env bash
# exp_004 Stage A — weight-decay reg ladder, seed 0, no dropout, full precision (no AMP).
# Runs wd ∈ {1e-4, 1e-3, 1e-2} as ISOLATED sequential processes; a GPU-clean gate between
# runs prevents a crashed run from cascading a corrupted CUDA context into the next
# (the failure mode of the first attempt). Continue-on-error so one crash != lose the rest.
#
# Eval cadence: held-out every 50 tile-steps (the comparison x-axis); per-tile train_iou
# diagnostic only every 200 tile-steps (host-RAM discipline — the eval-time Sample-reload
# churn that exhausted host RAM at K=10 tiles). Held-out curve口径 otherwise = exp_003.
#
# Goal: which wd best FLATTENS the post-peak collapse — NOT which peaks highest.
set -u
cd "$(dirname "$0")/.."

TRAIN="outputs/m0/g1/09LD1878.npz outputs/m0/g1/09LD1845.npz outputs/m0/g1/09LD1843.npz \
outputs/m0/g1/09LD1867.npz outputs/m0/g1/09LD2818.npz outputs/m0/g1/09LD1846.npz \
outputs/m0/g1/09LD1885.npz outputs/m0/g1/09LD1886.npz outputs/m0/g1/09LD1897.npz \
outputs/m0/g1/09LD2807.npz"
VAL="outputs/m0/g1/09LD2814.npz"
B3="outputs/g1/b3/09LD2814_b3_m5.json"

wait_gpu_clean() {
  # Block until GPU memory.used drops below 500 MiB (previous process fully released),
  # then a short settle so the driver resets context before the next run starts.
  local tries=0
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$used" ] && used=0
    if [ "$used" -lt 500 ]; then break; fi
    tries=$((tries+1))
    if [ "$tries" -gt 60 ]; then echo "[gate] GPU still at ${used} MiB after 60 tries, proceeding anyway"; break; fi
    sleep 5
  done
  sleep 8
  echo "[gate] GPU clean (used=${used} MiB) — starting next run"
}

for WD in 1e-4 1e-3 1e-2; do
  EXP="experiments/exp_004_m2_scaleup/stageA_wd${WD}"
  echo "================ Stage A  wd=${WD}  ->  ${EXP} ================"
  date
  .venv/Scripts/python scripts/run_m2_generalize.py \
    --train ${TRAIN} \
    --val ${VAL} \
    --b3-json ${B3} \
    --epochs 200 --eval-every-steps 50 --train-eval-every-steps 200 \
    --weight-decay ${WD} --seed 0 --no-amp --no-viz \
    --exp "${EXP}"
  echo "---- done wd=${WD} (exit $?) ----"
  date
  wait_gpu_clean
done

echo "================ Stage A ladder complete ================"
