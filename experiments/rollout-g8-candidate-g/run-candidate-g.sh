#!/usr/bin/env bash
# Candidate G: same-stack multi-GPU scaling at the candidate A workload.
# Fixed global batch (32 prompts x n=4 responses), only NGPUS_PER_NODE varies:
#   g1 = 1 GPU, g2 = 2 GPUs, g4 = 4 GPUs  (strong scaling, 15 iterations each)
# Everything else mirrors candidate A / candidate F (TOTAL_EPOCHS=1 -> 15 steps,
# constant LR, no warmup, sdpa attention, optimizer offload, bucket=256 MB).
# Checkpoints/logs go to /tmp because /home is 100% full.
# Usage: GPUS=5 bash run-candidate-g.sh
#        GPUS=5,6 bash run-candidate-g.sh
#        GPUS=5,6,7,0 bash run-candidate-g.sh
set -uo pipefail

GPUS=${GPUS:?set GPUS (e.g. 5 or 5,6 or 5,6,7,0)}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
MODE=${MODE:-g${NGPUS}}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd /home/zxw/venvs/rollout-g8-src/verl
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_TMPDIR=/tmp/ray-candidate-g-${MODE}

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
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=rollout_g8_candidate_g
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

LOG=/tmp/rollout-g8-candidate-g-${MODE}.log
echo "=== candidate-g mode=$MODE ngpus=$NGPUS gpus=$GPUS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-g8-checkpoints-g-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== candidate-g mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
exit $RC
