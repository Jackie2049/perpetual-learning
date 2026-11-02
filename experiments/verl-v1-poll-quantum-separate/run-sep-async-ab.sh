#!/usr/bin/env bash
# TM-008 separate_async poll A/B chain: 2.0 / 0.1 / 2.0 / 0.1, 8 steps each.
set -uo pipefail
export NCCL_SHM_DISABLE=1
for spec in "sa20a:2.0" "sa01a:0.1" "sa20b:2.0" "sa01b:0.1"; do
  MODE=${spec%%:*}; POLL=${spec##*:}
  MODE=$MODE POLL=$POLL STEPS=8 bash /tmp/run-sep-async.sh
  rc=$?
  echo "chain: $MODE (poll=$POLL) rc=$rc at $(date -u +%FT%TZ)"
  rm -rf /tmp/ray-$MODE
  sleep 5
done
echo "chain done $(date -u +%FT%TZ)"
