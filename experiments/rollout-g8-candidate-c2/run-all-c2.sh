#!/bin/bash
# Candidate C2 driver: G3 axis-3 batch scaling (b32/b64 ngram+AR) and
# strong proposer (draft_model Qwen2.5-0.5B, b8/b16 + AR same-session).
set -u
cd /home/zxw/venvs/rollout-g8-candidate-c2
PY=/home/zxw/venvs/rollout-g8/bin/python
export CUDA_VISIBLE_DEVICES=5
for combo in \
  "ar 512 32" "ar 512 64" "ar 2048 32" "ar 2048 64" \
  "spec 512 32" "spec 512 64" "spec 2048 32" "spec 2048 64" \
  "ar 512 8" "ar 512 16" "ar 2048 8" "ar 2048 16" \
  "draft 512 8" "draft 512 16" "draft 2048 8" "draft 2048 16"; do
  set -- $combo; mode=$1; il=$2; b=$3
  tag="lat_${mode}_in${il}_b${b}"
  echo "==> $tag $(date -u +%H:%M:%S)"
  $PY run-candidate-c2.py "$mode" "$il" "$b" > "${tag}_stdout.txt" 2> "${tag}_stderr.txt" \
    && echo "OK $tag" || echo "FAIL $tag"
done
echo "ALL_DONE $(date -u +%H:%M:%S)"
