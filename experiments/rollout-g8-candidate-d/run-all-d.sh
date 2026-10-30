#!/usr/bin/env bash
# Candidate D batch runner (vLLM 0.24.0+cu129, ngram proposer, GPU 5).
# Matrix: mode {ar,spec} x comp {H512,H2048,MIX} x iter {1,2} = 12 timed runs.
# Convention mirrors candidate C/DSD: one combo per process; engine init and
# Triton JIT caches are shared (~/.triton AOT cache). Step 0 preheats the
# Triton kernels (first spec combo JITs _topk_topp_kernel, cf. batch 31
# JIT-pollution finding), then timed combos run JIT-clean.
set -u
cd ~
source ~/venvs/rollout-g8/bin/activate
export CUDA_VISIBLE_DEVICES=5
export VLLM_LOGGING_LEVEL=INFO
LOG=~/venvs/run-candidate-d.log
: > "$LOG"
ts() { date '+[%H:%M:%S]'; }

echo "$(ts) preheat spec H512 (JIT compile, untimed)" | tee -a "$LOG"
python3 ~/venvs/run-candidate-d.py spec H512 warmup >> "$LOG" 2>&1 || { echo "$(ts) preheat FAILED rc=$?" | tee -a "$LOG"; exit 1; }

for comp in H512 H2048 MIX RX8; do
  for mode in ar spec; do
    for it in 1 2; do
      out=~/venvs/lat_d_${mode}_${comp}_i${it}.stdout.txt
      err=~/venvs/lat_d_${mode}_${comp}_i${it}.stderr.txt
      echo "$(ts) start ${mode} ${comp} iter${it}" | tee -a "$LOG"
      python3 ~/venvs/run-candidate-d.py "$mode" "$comp" "$it" > "$out" 2> "$err"
      rc=$?
      echo "$(ts) end ${mode} ${comp} iter${it} rc=$rc" | tee -a "$LOG"
      if [ $rc -ne 0 ]; then echo "FAILED ${mode}_${comp}_i${it}"; exit 1; fi
      echo "--- row ---" >> "$LOG"; cat "$out" >> "$LOG"
    done
  done
done

echo "$(ts) DONE" | tee -a "$LOG"
