#!/usr/bin/env bash
# M5 smoke: does the poll_interval patch wire up at runtime on main 3fdf2b54?
# 3 steps only; functional check, not a timing measurement.
set -uo pipefail

SRC=${SRC:-/tmp/m5-main-smoke/verl}
MODE=${MODE:-mainpoll}
POLL=${POLL:-0.1}
GPUS=${GPUS:-4,5}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
STEPS=${STEPS:-3}
INFLIGHT=${INFLIGHT:-2}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-poll-main-${MODE}

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
export PROJECT_NAME=rollout_g8_poll_main_smoke
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

LOG=/tmp/rollout-poll-main-${MODE}.log
echo "=== main-smoke mode=$MODE src=$SRC poll=$POLL inflight=$INFLIGHT ngpus=$NGPUS gpus=$GPUS steps=$STEPS start $(date -u +%FT%TZ) ===" > "$LOG"
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "algorithm.filter_groups.enable=True" \
  "algorithm.filter_groups.metric=acc" \
  "algorithm.filter_groups.max_inflight_gen_batches=${INFLIGHT}" \
  "trainer.v1.sampler.poll_interval=${POLL}" \
  "trainer.total_training_steps=${STEPS}" \
  "++transfer_queue.backend.SimpleStorage.total_storage_size=100000" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  'trainer.logger=["console"]' \
  "trainer.default_local_dir=/tmp/rollout-poll-main-checkpoints-${MODE}" >> "$LOG" 2>&1
RC=$?
echo "=== main-smoke mode=$MODE exit=$RC end $(date -u +%FT%TZ) ===" >> "$LOG"
rm -rf /tmp/rollout-poll-main-checkpoints-${MODE}
exit $RC
