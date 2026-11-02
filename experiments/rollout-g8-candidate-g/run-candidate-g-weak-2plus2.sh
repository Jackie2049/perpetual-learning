#!/usr/bin/env bash
# Candidate G — weak-scaling re-anchor under best placement (2026-11-02, M5 compute side).
# Same per-GPU workload as g2w/g4w (32 prompts/GPU/iter -> global batch 64/128),
# only the placement changes: 1+1 (0,4) for 2 cards, 2+2 (0,1,4,5) for 4 cards.
set -uo pipefail
cd /home/zxw/venvs
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

state() {
  echo "=== state $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw \
    --format=csv,noheader
}

run() {
  local mode=$1 gpus=$2 batch=$3 mini=$4 epochs=$5
  state
  echo ">>> $mode gpus=$gpus batch=$batch start $(date -u +%FT%TZ)"
  GPUS=$gpus MODE=$mode TRAIN_BATCH_SIZE=$batch PPO_MINI_BATCH_SIZE=$mini TOTAL_EPOCHS=$epochs \
    bash run-candidate-g.sh
  local rc=$?
  echo ">>> $mode rc=$rc end $(date -u +%FT%TZ)"
  sleep 10
  state
}

echo "chain-weak-2plus2 start $(date -u +%FT%TZ)"
run g2wp 0,4 64 32 2
run g4wp 0,1,4,5 128 64 4
echo "chain-weak-2plus2 done $(date -u +%FT%TZ)"
