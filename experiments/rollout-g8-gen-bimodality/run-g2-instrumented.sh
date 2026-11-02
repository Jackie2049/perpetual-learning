#!/usr/bin/env bash
# Gen two-state isolation: instrumented 2-GPU run (same config as candidate G g2).
# Adds per-request client timing (which replica served each request) and per-step
# phase markers (generate_sequences vs sleep_replicas, both inside timing_s/gen).
# Usage: bash run-g2-instrumented.sh
set -uo pipefail

GPUS=${GPUS:-0,1}
NGPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
MODE=${MODE:-g2t}
STEPS_EPOCHS=${TOTAL_EPOCHS:-2}
TIMING_DIR=${M5_GEN_TIMING_DIR:-/tmp/m5-gen-timing-${MODE}}

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
cd /home/zxw/venvs/rollout-g8-src/verl
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/tmp/ray-candidate-g-${MODE}
export M5_GEN_TIMING_DIR=${TIMING_DIR}

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
export TOTAL_EPOCHS=${STEPS_EPOCHS}
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=rollout_g8_gen_bimodality
export EXPERIMENT_NAME=qwen15b_grpo_gsm8k_${MODE}

rm -rf "${M5_GEN_TIMING_DIR}"
mkdir -p "${M5_GEN_TIMING_DIR}"

LOG=/tmp/rollout-g8-gen-bimodality-${MODE}.log
echo "=== gen-bimodality mode=$MODE ngpus=$NGPUS gpus=$GPUS epochs=$STEPS_EPOCHS start $(date -u +%FT%TZ) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "+actor_rollout_ref.rollout.max_model_len=2048" \
  "trainer.logger=[\"console\"]" \
  "trainer.default_local_dir=/tmp/rollout-g8-checkpoints-${MODE}" > "$LOG" 2>&1
RC=$?
echo "=== gen-bimodality mode=$MODE exit=$RC end $(date -u +%FT%TZ) ==="
echo "timing dir: ${M5_GEN_TIMING_DIR}"
ls -l "${M5_GEN_TIMING_DIR}" | head
exit $RC
