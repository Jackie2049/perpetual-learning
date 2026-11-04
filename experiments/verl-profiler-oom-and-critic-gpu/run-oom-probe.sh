#!/usr/bin/env bash
# OOM-snapshot probe runner (GPU side).
#
# One card, two processes: `attached` (rank 0 profiled -> observer registered) and
# `unranked` (rank 3, config.ranks=[0] -> observer NOT registered, negative control).
# Both force the same allocator OOM; only the observer differs.
#
# The tree under test is pinned with PYTHONPATH (the script lives in /tmp/m5-oom, so
# sys.path[0] is that directory, not the tree) and verl.__file__ is recorded in the JSON.
set -uo pipefail
SRC=${SRC:-/tmp/m5-main-smoke/verl}
G=${G:-1}
OUT=${OUT:-/tmp/m5-oom}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
export PYTHONPATH=$SRC:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$G
export PYTHONUNBUFFERED=1
cd "$SRC"
mkdir -p "$OUT"

for ARM in attached unranked; do
  echo "=== oom arm=$ARM gpu=$G src=$SRC start $(date -u +%FT%TZ) ==="
  python3 "$OUT/probe-oom-dump.py" --arm "$ARM" --save "$OUT/prof-$ARM" --out "$OUT/result-$ARM.json"
  echo "=== oom arm=$ARM exit=$? end $(date -u +%FT%TZ) ==="
done
