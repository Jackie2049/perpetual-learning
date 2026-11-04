#!/usr/bin/env bash
# torch_memory profiler end-to-end smoke on GPU.
# arm en = actor.profiler.enable=True + global_profiler.{tool=torch_memory,steps=[1],save_path} -> expect save_path/step1/*.pickle
# arm ex = official-example shape (tool + save_path + tool_config only, NO enable)                   -> expect no step*/ dir
set -uo pipefail
SRC=${SRC:-/tmp/m5-main-smoke/verl}
ARM=${ARM:-en}
G=${G:-1}
OUT=/tmp/m5-torchmem
PROF=$OUT/prof-$ARM
LOG=$OUT/run-$ARM.log
rm -rf "$PROF"

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$G
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-torchmem-$ARM

export MODEL_PATH=/home/zxw/models/Qwen2.5-0.5B-Instruct
export TRAIN_FILE=/home/zxw/data/gsm8k/train_sub480.parquet
export TEST_FILE=/home/zxw/data/gsm8k/test_sub120.parquet
export NGPUS_PER_NODE=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=128
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.25
export ROLLOUT_N=2
export TOTAL_EPOCHS=1
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=m5_torchmem
export EXPERIMENT_NAME=tm_$ARM

EXTRA=(
  "global_profiler.tool=torch_memory"
  "global_profiler.steps=[1]"
  "global_profiler.save_path=$PROF"
  "trainer.total_training_steps=2"
  "++transfer_queue.backend.SimpleStorage.total_storage_size=100000"
)
if [ "$ARM" = en ]; then
  EXTRA+=("actor_rollout_ref.actor.profiler.enable=True")
fi
if [ "$ARM" = ex ]; then
  EXTRA+=(
    "global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000"
    "global_profiler.global_tool_config.torch_memory.stack_depth=32"
  )
fi

echo "=== torchmem arm=$ARM src=$SRC gpu=$G prof=$PROF start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/m5-torchmem/ckpt-$ARM" > "$LOG" 2>&1
RC=$?
echo "=== torchmem arm=$ARM exit=$RC end $(date -u +%FT%TZ) ==="
echo "--- prof tree ---"
find "$PROF" -maxdepth 3 2>/dev/null | head -20
rm -rf /tmp/m5-torchmem/ckpt-$ARM
exit $RC
