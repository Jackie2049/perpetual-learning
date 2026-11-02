#!/usr/bin/env bash
# TM-008 async branch: alternate between poll arms so any slow drift hits both.
# Aborts if GPU 4/5 are no longer free (co-tenant could grab them at any time).
set -uo pipefail

for spec in "a200:" "a010:0.1" "a200b:" "a010b:0.1"; do
  MODE=${spec%%:*}
  POLL=${spec#*:}
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$1==4||$1==5 {s+=$2} END{print s+0}')
  if [ "${used}" -gt 100 ]; then
    echo "=== ABORT: GPU 4/5 busy (${used} MiB) before ${MODE} $(date -u +%FT%TZ) ==="
    exit 3
  fi
  echo "=== chain: launching ${MODE} poll=${POLL:-default} $(date -u +%FT%TZ) ==="
  MODE=${MODE} POLL=${POLL} bash /tmp/m5-async-poll/run-async-ab.sh
  echo "=== chain: ${MODE} returned $? $(date -u +%FT%TZ) ==="
done
echo "=== chain: all done $(date -u +%FT%TZ) ==="
