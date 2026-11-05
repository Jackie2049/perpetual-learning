#!/usr/bin/env bash
# Build the instrumented copy of a verl tree that the 4-card `inflight=1` arm runs on.
#
# The arm needs two patches on the tree it actually executes: the `poll_interval` config
# wiring (poll-interval-config.patch) and the blocking-poll counter (poll-count.patch).
# Applying them by hand is a window-time typo risk, and a missed application is silent —
# the run still exits 0, it just has no `training/sampler/blocking_polls` reading.
#
# Usage: bash prep-tree.sh <clean-tree> <dst-tree>
#        FORCE=1 bash prep-tree.sh <clean-tree> <dst-tree>     # replace an existing dst
#
# Env: PATCH_DIR  directory holding the two .patch files (default: repo siblings)
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PATCH_DIR=${PATCH_DIR:-$HERE/..}
INTERVAL_PATCH=$PATCH_DIR/verl-v1-poll-quantum/poll-interval-config.patch
COUNTER_PATCH=$PATCH_DIR/verl-v1-poll-quantum-polls/poll-count.patch

SRC=${1:-}
DST=${2:-}
if [ -z "$SRC" ] || [ -z "$DST" ]; then
  echo "usage: bash prep-tree.sh <clean-tree> <dst-tree>" >&2
  exit 2
fi
SRC=$(cd "$SRC" && pwd) || { echo "FATAL: no tree at $SRC" >&2; exit 2; }
for p in "$INTERVAL_PATCH" "$COUNTER_PATCH"; do
  [ -f "$p" ] || { echo "FATAL: missing patch $p" >&2; exit 2; }
done
case "$DST" in
  "$SRC"|"$SRC"/*) echo "FATAL: dst must differ from and not live under src" >&2; exit 2 ;;
esac
if [ -e "$DST" ]; then
  if [ "${FORCE:-0}" != 1 ]; then
    echo "FATAL: $DST exists; set FORCE=1 to replace it" >&2
    exit 3
  fi
  rm -rf "$DST"
fi

mkdir -p "$(dirname "$DST")"
cp -R "$SRC" "$DST" || { echo "FATAL: copy failed" >&2; exit 4; }

cd "$DST" || exit 4
for p in "$INTERVAL_PATCH" "$COUNTER_PATCH"; do
  patch -p1 --fuzz=0 -i "$p" >/dev/null || {
    echo "FATAL: $p did not apply to $DST with --fuzz=0 (upstream moved?)" >&2
    exit 5
  }
done

# A string, not an array: macOS bash 3.2 treats an empty array as unset under `set -u`.
missing=""
grep -q "poll_interval: 2.0" verl/trainer/config/ppo_trainer.yaml || missing="$missing yaml-key"
grep -q "poll_interval=sampler_config.get" verl/trainer/ppo/v1/trainer_base.py || missing="$missing trainer-wiring"
grep -q "sampler/blocking_polls" verl/trainer/ppo/v1/replay_buffer.py || missing="$missing counter"
grep -q '"training/sampler/blocking_polls"' verl/trainer/ppo/v1/utils.py || missing="$missing aggregation-rule"
if [ -n "$missing" ]; then
  echo "FATAL: post-patch verification failed:$missing" >&2
  exit 65
fi

echo "instrumented tree ready: $DST"
echo "  poll_interval default 2.0 (pass trainer.v1.sampler.poll_interval=0.1 for the poll arm)"
echo "  counter metric: training/sampler/blocking_polls"
