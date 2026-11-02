#!/usr/bin/env bash
# Candidate H: update-weights bucket-size sweep at the candidate A/G workload.
# update_weights is the only stage that scaled NEGATIVELY in candidate G
# (1.820 -> 2.351 -> 2.637 s for 1/2/4 engines). This sweep keeps the g4 config
# (4 GPUs, global batch 32, 15 iterations) and changes only the bucketed weight
# transfer size: h256 (= g4 replicate) / h1024 / h4096.
# Usage: BUCKET=256 bash run-candidate-h.sh
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET (256|1024|4096)}
MODE=h${BUCKET}

GPUS=${GPUS:-0,1,3,4}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd /home/zxw/venvs/rollout-g8-src/verl
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
# NCCL 2.32.3 on this host: the SHM transport aborts with CUDA 217 (peer access
# not supported on consumer 4090s). Disabling SHM forces the socket path, which
# is the transport the successful g1/g2/g4/g4n runs already used (their logs
# contain no SHM/RMA warnings).
export NCCL_SHM_DISABLE=1
export RAY_TMPDIR=/tmp/ray-candidate-h-${MODE}

export MODEL_PATH=/home/zxw/models/Qwen2.5-1.5B-Instruct
export TRAIN_FILE=/home/zxw/data/gsm8k/train_sub480.parquet
export TEST_FILE=/home/zxw/data/gsm8k/test_sub120.parquet
export NGPUS_PER_NODE=${NGPUS}
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=16
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.20
export ROLLOUT_N=4
export TOTAL_EPOCHS=1
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=rollout_g8_candidate_h
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

LOG=/tmp/rollout-g8-candidate-h-${MODE}.log
echo "=== candidate-h mode=$MODE bucket=${BUCKET}MB ngpus=$NGPUS gpus=$GPUS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${BUCKET}" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-g8-checkpoints-h-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== candidate-h mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
exit $RC
