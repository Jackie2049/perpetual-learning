#!/usr/bin/env bash
# Candidate G — 2+2 placement re-anchor (2026-11-02, M5 compute side).
# Re-measure the multi-GPU anchors under the best-known placement in ONE idle window:
#   g1p = GPU 0 (1 card), g2p = GPU 0,4 (2 cards, 1+1), g4p = GPU 0,1,4,5 (4 cards, 2+2)
# Same config as candidate G strong scaling (batch 32 x n=4, 15 iters, seed 42,
# default replay-buffer poll interval -> comparable with §13.36/§13.42 anchors).
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

echo "chain-2plus2 start $(date -u +%FT%TZ)"
run g1p 0
run g2p 0,4
run g4p 0,1,4,5
echo "chain-2plus2 done $(date -u +%FT%TZ)"
