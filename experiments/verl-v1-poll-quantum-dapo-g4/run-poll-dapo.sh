#!/usr/bin/env bash
# TM-008 边界实验：DAPO 组过滤 + 多 in-flight 下的 poll_interval A/B。
# 与 experiments/verl-v1-poll-quantum 的固定批配置相对照：那里 sample() 返回集合与等待时机无关，
# 这里 filter_groups 打开后 refill 是按轮询观测到的"过滤组数"派发的，等待时机可能改变派发量。
# Usage: MODE=dp010 POLL=0.1 SRC=/tmp/m5-poll-ab/verl bash run-poll-dapo.sh
#        MODE=dbase        SRC=/home/zxw/venvs/rollout-g8-src/verl bash run-poll-dapo.sh
set -uo pipefail

SRC=${SRC:-/home/zxw/venvs/rollout-g8-src/verl}
MODE=${MODE:-dbase}
POLL=${POLL:-}
GPUS=${GPUS:-0,1}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
INFLIGHT=${INFLIGHT:-2}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC" || { echo "FATAL: no tree at $SRC" >&2; exit 66; }

# This arm exists to read `training/sampler/blocking_polls`; a tree without the two patches
# runs to exit 0 with no counter, so the window is spent and the question stays open.
# Check before any GPU work. CHECK_ONLY=1 stops here (used by preflight's control run).
if [ "${CHECK_TREE:-1}" != 0 ]; then
  # A string, not an array: macOS bash 3.2 treats an empty array as unset under `set -u`.
  missing=""
  grep -q "poll_interval=sampler_config.get" verl/trainer/ppo/v1/trainer_base.py 2>/dev/null \
    || missing="$missing poll-interval-wiring"
  grep -q "sampler/blocking_polls" verl/trainer/ppo/v1/replay_buffer.py 2>/dev/null \
    || missing="$missing poll-counter"
  grep -q '"training/sampler/blocking_polls"' verl/trainer/ppo/v1/utils.py 2>/dev/null \
    || missing="$missing aggregation-rule"
  if [ -n "$missing" ]; then
    echo "FATAL: $SRC is not instrumented; missing:$missing" >&2
    echo "Build it with: bash prep-tree.sh <clean-tree> <dst-tree>" >&2
    exit 65
  fi
fi
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  echo "tree checks passed: $SRC"
  exit 0
fi

export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-poll-dapo-${MODE}

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
export PROJECT_NAME=rollout_g8_poll_dapo
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

EXTRA=(
  "algorithm.filter_groups.enable=True"
  "algorithm.filter_groups.metric=acc"
  "algorithm.filter_groups.max_inflight_gen_batches=${INFLIGHT}"
  # The base recipe pinned this False up to verl 24f25b03 and no longer does; the
  # rollout default is True. Every archived poll batch was measured with False, so
  # pass it explicitly — leaving it out silently changes the generation-side config
  # the archived numbers were taken under.
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
)
if [ -n "$POLL" ]; then
  EXTRA+=("trainer.v1.sampler.poll_interval=${POLL}")
fi

LOG=/tmp/rollout-poll-dapo-${MODE}.log
echo "=== poll-dapo mode=$MODE src=$SRC poll=${POLL:-default} inflight=$INFLIGHT ngpus=$NGPUS gpus=$GPUS epochs=$TOTAL_EPOCHS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-poll-dapo-checkpoints-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== poll-dapo mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
rm -rf /tmp/rollout-poll-dapo-checkpoints-${MODE}
exit $RC
