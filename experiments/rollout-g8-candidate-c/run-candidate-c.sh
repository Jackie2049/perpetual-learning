#!/usr/bin/env bash
# Candidate C orchestrator: 4 combos x {ar, spec} on GPU 5, then aggregate.
# Each combo runs in its own engine process so stderr maps cleanly to one
# combo (vLLM engine logs, incl. "SpecDecoding metrics" lines, go to stderr).
# Matrix: input_len {512,2048} x batch {8,16} x mode {ar,spec}; max_tokens=128
# (matches candidate B short-output decode calibration point).
set -u
export CUDA_VISIBLE_DEVICES=5   # pin GPU 5, same as candidates A/B
cd "$(dirname "$0")"
OUT=/home/zxw/venvs/rollout-g8-candidate-c
mkdir -p "$OUT"
PY=/home/zxw/venvs/rollout-g8/bin/python

for mode in ar spec; do
  for il in 512 2048; do
    for b in 8 16; do
      tag="lat_${mode}_in${il}_b${b}"
      echo "[$(date -u +%H:%M:%S)] start $tag" >> "$OUT/run.log"
      "$PY" run-candidate-c.py "$mode" "$il" "$b" \
        > "$OUT/${tag}_stdout.txt" 2> "$OUT/${tag}_stderr.txt"
      rc=$?
      echo "[$(date -u +%H:%M:%S)] end $tag rc=$rc" >> "$OUT/run.log"
      if [ $rc -ne 0 ]; then
        echo "FAIL $tag rc=$rc (see ${tag}_stderr.txt)" >> "$OUT/run.log"
      fi
    done
  done
done

echo "[$(date -u +%H:%M:%S)] all combos done, aggregating" >> "$OUT/run.log"
"$PY" parse-candidate-c.py "$OUT" >> "$OUT/run.log" 2>&1
echo "[$(date -u +%H:%M:%S)] DONE" >> "$OUT/run.log"
