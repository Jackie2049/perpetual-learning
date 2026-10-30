#!/usr/bin/env bash
# Candidate DSD batch runner (vLLM 0.24.0 Dynamic SD, ngram proposer).
# Matrix: input_len {512,2048} x batch {8,16}, mode dsd-ngram (dynamic K).
# Convention mirrors candidate C/C2: one combo per process, engine init +
# Triton JIT caches are shared across processes (~/.triton, AOT cache).
# Step 0 preheats all Triton kernels (first combo also JITs _topk_topp_kernel
# and _compute_slot_mapping_kernel), then timed combos run JIT-clean.
set -u
cd ~
source ~/venvs/rollout-g8/bin/activate
export CUDA_VISIBLE_DEVICES=5
export VLLM_LOGGING_LEVEL=INFO
LOG=~/venvs/run-dsd.log
: > "$LOG"
ts() { date '+[%H:%M:%S]'; }

echo "$(ts) preheat in512 b8 (JIT compile, untimed)" | tee -a "$LOG"
python3 ~/venvs/run-candidate-dsd.py 512 8 >> "$LOG" 2>&1 || { echo "$(ts) preheat FAILED rc=$?" | tee -a "$LOG"; exit 1; }

for il in 512 2048; do
  for b in 8 16; do
    out=~/venvs/lat_dsd_in${il}_b${b}.stdout.txt
    err=~/venvs/lat_dsd_in${il}_b${b}.stderr.txt
    echo "$(ts) start dsd in${il} b${b}" | tee -a "$LOG"
    python3 ~/venvs/run-candidate-dsd.py "$il" "$b" > "$out" 2> "$err"
    rc=$?
    echo "$(ts) end dsd in${il} b${b} rc=$rc" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "FAILED in${il}_b${b}"; exit 1; fi
    echo "--- row ---" >> "$LOG"; cat "$out" >> "$LOG"
  done
done

echo "$(ts) verify candidate-C spec in512 b8 (JIT-warm, same stack)" | tee -a "$LOG"
python3 ~/venvs/run-candidate-c2.py spec 512 8 2>/dev/null >> "$LOG" 2>&1 || true

echo "$(ts) DONE" | tee -a "$LOG"
