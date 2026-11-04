#!/usr/bin/env bash
# precision_debugger (msprobe) runtime smoke on GPU.
#
# pda: stages=[actor_update]   -- the decorator at engine_workers.py:768 exists -> expect a dump dir
# pdc: stages=[critic_update]  -- accepted by the stage whitelist but no decorator on this run
#                                 -> expect NO dump dir, and no warning/error (silent no-op)
#
# msprobe + its matplotlib deps are installed to /tmp/m5-msprobe-target (PYTHONPATH only;
# the shared venv is untouched). Config json is the package's own default template.
set -uo pipefail
SRC=${SRC:-/tmp/m5-main-smoke/verl}
ARM=${ARM:-pda}
G=${G:-1}
OUT=/tmp/m5-msprobe
PROF=$OUT/prof-$ARM
LOG=$OUT/run-$ARM.log
rm -rf "$PROF"

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
export PYTHONPATH=${MSROBE_TARGET:-/tmp/m5-msprobe-target}
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$G
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-pdbg-$ARM

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
export PROJECT_NAME=m5_pdbg
export EXPERIMENT_NAME=pdbg_$ARM

if [ "$ARM" = pda ]; then
  STAGES="[actor_update]"
else
  STAGES="[critic_update]"
fi

EXTRA=(
  "global_profiler.tool=precision_debugger"
  "global_profiler.steps=[1]"
  "global_profiler.save_path=$PROF"
  "global_profiler.global_tool_config.precision_debugger.config_path=/tmp/m5-msprobe/config.json"
  "global_profiler.global_tool_config.precision_debugger.stages=$STAGES"
  "global_profiler.global_tool_config.precision_debugger.strict=False"
  "actor_rollout_ref.actor.profiler.enable=True"
  "trainer.total_training_steps=2"
  "++transfer_queue.backend.SimpleStorage.total_storage_size=100000"
)

echo "=== pdbg arm=$ARM stages=$STAGES src=$SRC gpu=$G prof=$PROF start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  'trainer.logger=["console"]' \
  "trainer.default_local_dir=/tmp/m5-msprobe/ckpt-$ARM" > "$LOG" 2>&1
RC=$?
echo "=== pdbg arm=$ARM exit=$RC end $(date -u +%FT%TZ) ==="
echo "--- dump tree ---"
find "$PROF" -maxdepth 4 2>/dev/null | head -30
echo "--- PrecisionDebugger / msprobe mentions ---"
grep -c "PrecisionDebugger\|msprobe" "$LOG" 2>/dev/null
rm -rf /tmp/m5-msprobe/ckpt-$ARM
exit $RC
