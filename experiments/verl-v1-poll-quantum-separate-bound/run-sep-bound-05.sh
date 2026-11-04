#!/usr/bin/env bash
# TM-008 separate_async rollout-bound regime: poll 0.5 supplementary arm (2 runs).
set -uo pipefail
export NCCL_SHM_DISABLE=1
export NORM_RESP=16 MAX_RESP=1024 SYNC_STEP=1 MINI_BATCH=4
export TQ_SIZE=1536  # per-unit 192 = 3 steps of 64 samples
for spec in "sb05a:0.5" "sb05b:0.5"; do
  MODE=${spec%%:*}; POLL=${spec##*:}
  MODE=$MODE POLL=$POLL STEPS=8 bash /tmp/run-sep-bound.sh
  rc=$?
  echo "chain: $MODE (poll=$POLL) rc=$rc at $(date -u +%FT%TZ)"
  rm -rf /tmp/ray-$MODE
  sleep 5
done
echo "chain done $(date -u +%FT%TZ)"
