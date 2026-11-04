#!/usr/bin/env bash
# TM-008 separate_async rollout-bound regime: poll 2.0 / 0.1 A/B chain.
# Load = calibration config (n=16, max_response=1024, sync_step=1, mini_batch=4).
set -uo pipefail
export NCCL_SHM_DISABLE=1
export NORM_RESP=16 MAX_RESP=1024 SYNC_STEP=1 MINI_BATCH=4
export TQ_SIZE=1536  # per-unit 192 = 3 steps of 64 samples; 512 is exactly one step
for spec in "sb20a:2.0" "sb01a:0.1" "sb20b:2.0" "sb01b:0.1"; do
  MODE=${spec%%:*}; POLL=${spec##*:}
  MODE=$MODE POLL=$POLL STEPS=8 bash /tmp/run-sep-bound.sh
  rc=$?
  echo "chain: $MODE (poll=$POLL) rc=$rc at $(date -u +%FT%TZ)"
  rm -rf /tmp/ray-$MODE
  sleep 5
done
echo "chain done $(date -u +%FT%TZ)"
