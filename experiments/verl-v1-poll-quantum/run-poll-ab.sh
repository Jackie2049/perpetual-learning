#!/usr/bin/env bash
# TM-008: replay-buffer sample-wait poll interval A/B (same config as candidate G g2).
# Usage: MODE=p010 POLL=0.1 bash run-poll-ab.sh
#        MODE=pbase        bash run-poll-ab.sh          # pristine tree, default 2.0
set -uo pipefail

SRC=${SRC:-/home/zxw/venvs/rollout-g8-src/verl}
MODE=${MODE:-pbase}
POLL=${POLL:-}
GPUS=${GPUS:-0,1}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-poll-ab-${MODE}

export MODEL_PATH=/home/zxw/models/Qwen2.5-1.5B-Instruct
export TRAIN_FILE=/home/zxw/data/gsm8k/train_sub480.parquet
export TEST_FILE=/home/zxw/data/gsm8k/test_sub120.parquet
export NGPUS_PER_NODE=${NGPUS}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.20
export ROLLOUT_N=4
export TOTAL_EPOCHS=${TOTAL_EPOCHS}
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=rollout_g8_poll_ab
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

EXTRA=()
if [ -n "$POLL" ]; then
  EXTRA+=("trainer.v1.sampler.poll_interval=${POLL}")
fi

LOG=/tmp/rollout-poll-ab-${MODE}.log
echo "=== poll-ab mode=$MODE src=$SRC poll=${POLL:-default} ngpus=$NGPUS gpus=$GPUS epochs=$TOTAL_EPOCHS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-poll-checkpoints-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== poll-ab mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
rm -rf /tmp/rollout-poll-checkpoints-${MODE}
exit $RC
