#!/usr/bin/env bash
# End-to-end plumbing check for the critic probe, with no GPU and no server.
#
#   bash run-mock.sh [<work-dir>]
#
# Builds a fixture tree (make-mock-tree.sh), runs all three arms through
# run-critic-pdbg.sh against it, then runs the pre-registered verdict
# (verify-critic-pdbg.py). The arms' logs are reality-shaped (Ray task prefix + step-0
# validation line), so a regression in the log counters shows up here as INVALID RUN rather
# than passing silently.
#
# What this proves: preflight wiring, the automatic FA2->sdpa site patch, the arm JSON schema,
# the log counters, and the verdict logic all agree.
# What this does NOT prove: anything about msprobe, critic training, dumps, or timing on a GPU.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${1:-/tmp/m5-critic-mock}
TREE=$WORK/tree
OUT=$WORK/out

bash "$HERE/make-mock-tree.sh" "$TREE" >/dev/null
rm -rf "$OUT"; mkdir -p "$OUT"

for arm in au cu cv; do
  SRC=$TREE/src \
  MSPROBE_TARGET=$TREE/msprobe \
  TRL_TARGET=$TREE/trl \
  MSPROBE_CONFIG=$TREE/msprobe/config.json \
  MODEL_PATH=$TREE/model \
  TRAIN_FILE=$TREE/data/train.parquet \
  TEST_FILE=$TREE/data/test.parquet \
  ARM=$arm G=0 OUT=$OUT AUTO_PATCH=1 \
  bash "$HERE/../run-critic-pdbg.sh" > "$OUT/wrap-$arm.txt" 2>&1
  echo "arm=$arm wrapper_rc=$?"
done

python3 "$HERE/../verify-critic-pdbg.py" --out "$OUT" | tee "$OUT/judge.txt"
JUDGE_RC=${PIPESTATUS[0]}
echo "judge_rc=$JUDGE_RC"

# Negative control for the import-based dependency check: keep the --target directory, delete the
# package inside it. An existence-only preflight would accept this and die inside step 1; the
# import check must refuse in seconds. Nothing here reaches a GPU either way.
BROKEN=$WORK/broken
rm -rf "$BROKEN"; cp -R "$TREE" "$BROKEN"
rm -rf "$BROKEN/msprobe/msprobe"
SRC=$BROKEN/src \
MSPROBE_TARGET=$BROKEN/msprobe \
TRL_TARGET=$BROKEN/trl \
MSPROBE_CONFIG=$BROKEN/msprobe/config.json \
MODEL_PATH=$BROKEN/model \
TRAIN_FILE=$BROKEN/data/train.parquet \
TEST_FILE=$BROKEN/data/test.parquet \
ARM=cu G=0 OUT=$OUT/broken AUTO_PATCH=1 \
bash "$HERE/../run-critic-pdbg.sh" > "$OUT/wrap-broken-msprobe.txt" 2>&1
BROKEN_RC=$?
if [ "$BROKEN_RC" = "65" ]; then
  echo "negative_control=PASS (dir present, package gone -> exit 65 before any run)"
else
  echo "negative_control=FAIL (expected 65, got $BROKEN_RC)"
fi

if [ "$JUDGE_RC" = "0" ] && [ "$BROKEN_RC" = "65" ]; then
  exit 0
fi
exit 1
