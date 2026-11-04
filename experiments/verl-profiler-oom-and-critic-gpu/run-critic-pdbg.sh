#!/usr/bin/env bash
# Critic-side stage probe: does a REAL critic run make `critic_update` / `compute_values` dump?
#
# Three arms, same run recipe, same tree, same card -- only `stages` differs:
#   au  stages=[actor_update]    positive control: the plumbing is live -> expect a dump
#   cu  stages=[critic_update]   whitelisted, no decorator anywhere -> expect NO dump
#   cv  stages=[compute_values]  whitelisted, no decorator anywhere -> expect NO dump
#
# The run is PPO with a real critic (adv_estimator=gae, critic.enable auto-true,
# critic.model.path set), so "critic_update collected nothing" cannot be explained by
# "this run had no critic".
#
# Also counts the `Unknown precision_debugger stage: train_batch` warnings: the critic
# trains through TrainingWorker.train_batch, whose decorator role is not in the
# whitelist, which warns on every call regardless of the user's stage list.
#
# msprobe + its matplotlib deps live in /tmp/m5-msprobe-target, trl (value head, needed by
# verl's FSDP critic) in /tmp/m5-trl-0.27.0 (PYTHONPATH only; the shared venv is untouched).
# Local tree adaptation: load_valuehead_model hardcodes attn_implementation="flash_attention_2"
# (not installed here) -> patched to "sdpa" in this tree only. Recorded in the README.
set -uo pipefail
SRC=${SRC:-/tmp/m5-main-smoke/verl}
ARM=${ARM:-cu}
G=${G:-4}
OUT=${OUT:-/tmp/m5-pdbg-critic}
PROF=$OUT/prof-$ARM
LOG=$OUT/run-$ARM.log
rm -rf "$PROF"

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
export PYTHONPATH=${MSPROBE_TARGET:-/tmp/m5-msprobe-target}:${TRL_TARGET:-/tmp/m5-trl-0.27.0}
export TRL_EXPERIMENTAL_SILENCE=1
cd "$SRC"
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$G
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-pdbg-critic-$ARM

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
export PROJECT_NAME=m5_pdbg_critic
export EXPERIMENT_NAME=pdbg_critic_$ARM

case "$ARM" in
  au) STAGES="[actor_update]" ;;
  cu) STAGES="[critic_update]" ;;
  cv) STAGES="[compute_values]" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 64 ;;
esac

EXTRA=(
  "global_profiler.tool=precision_debugger"
  "global_profiler.steps=[1]"
  "global_profiler.save_path=$PROF"
  "global_profiler.global_tool_config.precision_debugger.config_path=/tmp/m5-msprobe/config.json"
  "global_profiler.global_tool_config.precision_debugger.stages=$STAGES"
  "global_profiler.global_tool_config.precision_debugger.strict=False"
  "actor_rollout_ref.actor.profiler.enable=True"
  "critic.profiler.enable=True"
  "algorithm.adv_estimator=gae"
  "critic.enable=True"
  "critic.model.path=$MODEL_PATH"
  "critic.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
  "critic.ppo_micro_batch_size_per_gpu=1"
  "critic.ppo_max_token_len_per_gpu=3000"
  "critic.fsdp.optimizer_offload=True"
  "critic.fsdp.model_dtype=bf16"
  "trainer.total_training_steps=2"
  "++transfer_queue.backend.SimpleStorage.total_storage_size=100000"
)

echo "=== pdbg-critic arm=$ARM stages=$STAGES src=$SRC gpu=$G prof=$PROF start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "${EXTRA[@]}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  'trainer.logger=["console"]' \
  "trainer.default_local_dir=$OUT/ckpt-$ARM" > "$LOG" 2>&1
RC=$?
echo "=== pdbg-critic arm=$ARM exit=$RC end $(date -u +%FT%TZ) ==="
echo "--- dump tree ---"
find "$PROF" -maxdepth 5 2>/dev/null | head -30
echo "--- counts ---"
echo "unknown_stage_warnings=$(grep -c 'Unknown precision_debugger stage' "$LOG" 2>/dev/null)"
echo "train_batch_warnings=$(grep -c 'Unknown precision_debugger stage: train_batch' "$LOG" 2>/dev/null)"
echo "critic_steps=$(grep -c 'update_critic' "$LOG" 2>/dev/null)"
rm -rf "$OUT/ckpt-$ARM"
exit $RC
