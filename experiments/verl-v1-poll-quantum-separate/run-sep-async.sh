#!/usr/bin/env bash
# M5 TM-008 separate_async smoke on main tree (3fdf2b54 + TM-008 patch + device-map patch).
# GPU split: 2 training (4,5 same socket) + 1 rollout (1).  ROLLOUT_TP must divide 1 -> 1.
# Upstream reference: tests/special_e2e/run_v1_separate_async.sh (CI shape, GPU count adapted).
set -uo pipefail
SRC=${SRC:-/tmp/m5-main-smoke/verl}
MODE=${MODE:-sepasync-smoke}
TRAIN_GPUS=${TRAIN_GPUS:-4,5}
ROLLOUT_GPU=${ROLLOUT_GPU:-1}
STEPS=${STEPS:-2}
POLL=${POLL:-}
SYNC_STEP=${SYNC_STEP:-4}
MINI_BATCH=${MINI_BATCH:-1}
NORM_RESP=${NORM_RESP:-4}
export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd "$SRC"
export PYTHONPATH=/tmp/cupy-pkgs:$SRC
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS},${ROLLOUT_GPU}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
if [ "${NOSET:-0}" = "1" ]; then export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; fi
export RAY_TMPDIR=/tmp/ray-${MODE}
export MODEL_PATH=/home/zxw/models/Qwen2.5-0.5B-Instruct
export TRAIN_FILE=/home/zxw/data/gsm8k/train_sub480.parquet
export TEST_FILE=/home/zxw/data/gsm8k/test_sub120.parquet
export TRAIN_BATCH_SIZE=$((SYNC_STEP*MINI_BATCH))
EXTRA=()
if [ -n "$POLL" ]; then EXTRA+=("trainer.v1.sampler.poll_interval=${POLL}"); fi
LOG=/tmp/${MODE}.log
echo "=== $MODE start $(date -u +%FT%TZ) cuda=$CUDA_VISIBLE_DEVICES steps=$STEPS poll=${POLL:-default} src=$SRC ==="
python3 -u -m verl.trainer.main_ppo \
  "${EXTRA[@]}" \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=${SYNC_STEP} \
  trainer.v1.sampler.max_off_policy_threshold=8 \
  trainer.v1.sampler.max_off_policy_strategy=drop \
  transfer_queue.enable=True \
  ++transfer_queue.backend.SimpleStorage.total_storage_size=128 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation=left \
  data.filter_overlong_prompts=True \
  data.return_raw_chat=True \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.gen_batch_size=1 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH} \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=${NORM_RESP} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
  reward.reward_manager.name=dapo \
  trainer.logger='["console"]' \
  trainer.project_name=verl-test-v1-separate-async \
  trainer.experiment_name=m5-sep-async-${MODE} \
  trainer.default_local_dir=/tmp/m5-sep-async-ckpt-${MODE} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${STEPS} \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  > "$LOG" 2>&1
RC=$?
echo "=== $MODE exit=$RC end $(date -u +%FT%TZ) ==="
rm -rf /tmp/m5-sep-async-ckpt-${MODE}
exit $RC
