#!/usr/bin/env bash
# TM-008 stale-eviction A/B: colocate_async + drop strategy + max_off_policy_threshold=1.
# 默认阈值 8 在本规模不会触发；threshold=2 冒烟 4 步仍是 span≤2、零逐出；threshold=1 才让
# 上一步派发的整批变成 span=2 被丢弃，从而把 ReplayBufferAsync 的 stale 逐出通道打开。
# Usage: MODE=st1p200 POLL=2.0 THRESHOLD=1 STEPS=15 bash run-stale-ab.sh
set -uo pipefail
SRC=${SRC:-/tmp/m5-poll-ab/verl}
MODE=${MODE:-st1p200}
POLL=${POLL:-}
GPUS=${GPUS:-4,5}
NGPUS=$(awk -F"," "{print NF}" <<< "$GPUS")
THRESHOLD=${THRESHOLD:-1}
STEPS=${STEPS:-15}
export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-stale-${MODE}
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
export PROJECT_NAME=rollout_g8_poll_stale
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}
EXTRA=(
  trainer.v1.trainer_mode=colocate_async
  trainer.v1.colocate_async.num_warmup_batches=1
  trainer.v1.sampler.max_off_policy_threshold=${THRESHOLD}
  trainer.v1.sampler.max_off_policy_strategy=drop
  trainer.total_training_steps=${STEPS}
)
if [ -n "$POLL" ]; then
  EXTRA+=("trainer.v1.sampler.poll_interval=${POLL}")
fi
LOG=/tmp/rollout-poll-stale-${MODE}.log
echo "=== stale mode=$MODE src=$SRC poll=${POLL:-default} thresh=$THRESHOLD steps=$STEPS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-poll-stale-checkpoints-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== stale mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
rm -rf /tmp/rollout-poll-stale-checkpoints-${MODE}
exit $RC
