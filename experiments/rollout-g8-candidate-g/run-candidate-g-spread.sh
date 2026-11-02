#!/usr/bin/env bash
# Candidate G — end-to-end test of the NUMA/PCIe-root spread prediction (2026-11-02).
# Microbench (g8-comm-compute-probe.py, NCCL_P2P_DISABLE=1, 4 ranks, 64MB all-reduce):
#   4,5,6,7 -> 3.07 GB/s   0,1,3,4 -> 4.06 GB/s   3,4,5,6 (1+3) -> 4.06   0,1,4,5 (2+2) -> 5.98
# Prediction: a 2+2 split beats the 3+1 placement A, and a 1+3 split matches it; same
# training config as candidate G (batch 32 x n=4, 15 iters, seed 42).
set -uo pipefail
cd /home/zxw/venvs
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

state() {
  echo "=== state $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw \
    --format=csv,noheader
}

run() {
  local mode=$1 gpus=$2
  state
  echo ">>> $mode gpus=$gpus start $(date -u +%FT%TZ)"
  GPUS=$gpus MODE=$mode bash run-candidate-g.sh
  local rc=$?
  echo ">>> $mode rc=$rc end $(date -u +%FT%TZ)"
  sleep 10
  state
}

echo "chain-spread start $(date -u +%FT%TZ)"
run pC1 0,1,4,5
run pC2 3,4,5,6
echo "chain-spread done $(date -u +%FT%TZ)"
