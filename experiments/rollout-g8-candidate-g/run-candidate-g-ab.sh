#!/usr/bin/env bash
# Candidate G — placement A/B/A/B repeat (48th+ batch, 2026-11-02)
# Question: is the 19.4% g4(0,1,3,4) vs g4n(4,5,6,7) gap reproducible inside one
# idle window (placement effect), or was it machine drift (run 40+ min apart)?
# Same config as candidate G strong scaling: batch 32 x n=4, 15 iterations, seed 42.
# A = 0,1,3,4 (NUMA0 x3 + NUMA1 x1, two PCIe roots)   B = 4,5,6,7 (all NUMA1, one root)
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

echo "chain-ab start $(date -u +%FT%TZ)"
run pA1 0,1,3,4
run pB1 4,5,6,7
run pA2 0,1,3,4
run pB2 4,5,6,7
echo "chain-ab done $(date -u +%FT%TZ)"
