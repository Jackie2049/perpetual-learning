#!/usr/bin/env bash
# TM-008 async branch: does the 2.0s replay-buffer poll quantum cost in V1 colocate_async?
# Same tree/config as the sync A/B (experiments/verl-v1-poll-quantum), only trainer mode differs.
# Usage: MODE=a200 POLL=2.0 bash run-async-ab.sh
#        MODE=a010 POLL=0.1 bash run-async-ab.sh
#        MODE=asmoke STEPS=1 bash run-async-ab.sh    # 1-step config smoke
set -uo pipefail

SRC=${SRC:-/tmp/m5-poll-ab/verl}
MODE=${MODE:-a200}
POLL=${POLL:-}
GPUS=${GPUS:-4,5}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
STEPS=${STEPS:-}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-async-poll-${MODE}

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
export TOTAL_EPOCHS=${TOTAL_EPOCHS}
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=rollout_g8_poll_async
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

EXTRA=(
  trainer.v1.trainer_mode=colocate_async
  trainer.v1.colocate_async.num_warmup_batches=1
)
if [ -n "$POLL" ]; then
  EXTRA+=("trainer.v1.sampler.poll_interval=${POLL}")
fi
if [ -n "$STEPS" ]; then
  EXTRA+=("trainer.total_training_steps=${STEPS}")
fi

LOG=/tmp/rollout-poll-async-${MODE}.log
echo "=== async-poll mode=$MODE src=$SRC poll=${POLL:-default} ngpus=$NGPUS gpus=$GPUS epochs=$TOTAL_EPOCHS steps=${STEPS:-auto} start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-poll-async-checkpoints-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== async-poll mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
rm -rf /tmp/rollout-poll-async-checkpoints-${MODE}
exit $RC
