#!/usr/bin/env bash
# Candidate F: update-side ppo_mini_batch structure sweep at candidate A stack.
# Baseline config is PPO_MINI_BATCH_SIZE=16 (2 optimizer steps per iteration,
# train_batch_size=32 prompts x n=4 responses):
#   f16 = 16 (same-day re-baseline / test-retest replication)
#   f1  = 8  (4 optimizer steps per iteration)
#   f2  = 32 (1 optimizer step per iteration)
# Everything else is identical to candidate A / candidate E e3 (default caps).
# TOTAL_EPOCHS=1 -> 15 steps, same data order, constant LR.
# Checkpoints/logs go to /tmp because /home was 100% full on 2026-10-31.
# Usage: MODE=f16|f1|f2 bash run-candidate-f.sh
set -uo pipefail
MODE=${MODE:-f16}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd /home/zxw/venvs/rollout-g8-src/verl
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CANDIDATE_F_GPU:-5}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

export MODEL_PATH=/home/zxw/models/Qwen2.5-1.5B-Instruct
export TRAIN_FILE=/home/zxw/data/gsm8k/train_sub480.parquet
export TEST_FILE=/home/zxw/data/gsm8k/test_sub120.parquet
export NGPUS_PER_NODE=1
export TRAIN_BATCH_SIZE=32
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
export PROJECT_NAME=rollout_g8_candidate_f
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

case "$MODE" in
  f16) export PPO_MINI_BATCH_SIZE=16 ;;
  f1) export PPO_MINI_BATCH_SIZE=8 ;;
  f2) export PPO_MINI_BATCH_SIZE=32 ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac

LOG=/tmp/rollout-g8-candidate-f-${MODE}.log
echo "=== candidate-f mode=$MODE mini_batch=$PPO_MINI_BATCH_SIZE start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-g8-checkpoints-f-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== candidate-f mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
exit $RC
