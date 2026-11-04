#!/usr/bin/env bash
# Critic-side stage probe: does a REAL critic run make `critic_update` / `compute_values` dump?
#
# Three arms, same run recipe, same tree, same card -- only `stages` differs:
#   au  stages=[actor_update]    positive control: the plumbing is live -> expect a dump
#   cu  stages=[critic_update]   whitelisted, no decorator anywhere -> expect NO dump
#   cv  stages=[compute_values]  whitelisted, no decorator anywhere -> expect NO dump
#
# The run is PPO with a real critic (adv_estimator=gae + critic.enable=True +
# critic.model.path), so "critic_update collected nothing" cannot be explained by
# "this run had no critic".
#
# Also counts the `Unknown precision_debugger stage: train_batch` warnings: the critic
# trains through TrainingWorker.train_batch, whose decorator role is not in the whitelist,
# which warns on every call regardless of the user's stage list.
#
# Window guards (each of these kills a run in the first minutes otherwise):
#   1. load_valuehead_model hardcodes attn_implementation="flash_attention_2" and this box
#      has no flash-attn -> the tree is site-patched to "sdpa" by patch-critic-sdpa.py
#      (idempotent; AUTO_PATCH=1 applies it, AUTO_PATCH=0 aborts instead of mutating).
#   2. msprobe + trl live in /tmp --target dirs and /home is full -> preflight checks them, and
#      for msprobe (the one hard dependency) it imports it rather than only stat-ing the directory.
#
# Output: the training log, plus `arm-<ARM>.json` -- machine-readable (exit code, dump
# inventory with sizes, warning counts, step counts, tree patch state, dependency presence)
# and written even when the run fails. verify-critic-pdbg.py judges that JSON, so no number
# reaches the archive by hand.
#
# msprobe + its matplotlib deps live in /tmp/m5-msprobe-target, trl (value head, needed by
# verl's FSDP critic) in /tmp/m5-trl-0.27.0 (PYTHONPATH only; the shared venv is untouched).
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-/tmp/m5-main-smoke/verl}
ARM=${ARM:-cu}
G=${G:-4}
OUT=${OUT:-/tmp/m5-pdbg-critic}
AUTO_PATCH=${AUTO_PATCH:-1}
PROF=$OUT/prof-$ARM
LOG=$OUT/run-$ARM.log
ARM_JSON=$OUT/arm-$ARM.json
rm -rf "$PROF"
mkdir -p "$OUT"

export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
export MSPROBE_DIR=${MSPROBE_TARGET:-/tmp/m5-msprobe-target}
export TRL_DIR=${TRL_TARGET:-/tmp/m5-trl-0.27.0}
export PYTHONPATH=$MSPROBE_DIR:$TRL_DIR
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

export MODEL_PATH=${MODEL_PATH:-/home/zxw/models/Qwen2.5-0.5B-Instruct}
export TRAIN_FILE=${TRAIN_FILE:-/home/zxw/data/gsm8k/train_sub480.parquet}
export TEST_FILE=${TEST_FILE:-/home/zxw/data/gsm8k/test_sub120.parquet}
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

export MSPROBE_CONFIG=${MSPROBE_CONFIG:-/tmp/m5-msprobe/config.json}

case "$ARM" in
  au) STAGES="[actor_update]" ;;
  cu) STAGES="[critic_update]" ;;
  cv) STAGES="[compute_values]" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 64 ;;
esac

# --- preflight: every item below aborts the run in its first minutes, so check in seconds
# msprobe is a hard requirement: without it the profiler tool is a no-op and the `au` positive
# control cannot fire, which would burn the window to learn "invalid run".
for dep in "$SRC/verl/utils/model.py" "$SCRIPT_DIR/patch-critic-sdpa.py" \
           "$SCRIPT_DIR/probe-valuehead-branch.py" "$MSPROBE_DIR" \
           "$MSPROBE_CONFIG" "$MODEL_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
  if [ ! -e "$dep" ]; then
    echo "[preflight] MISSING: $dep" >&2
    echo "[preflight] /tmp --target dirs and datasets do not survive a host cleanup -> reinstall before spending a window" >&2
    echo "[preflight]   tree:   curl -L 'https://codeload.github.com/verl-project/verl/tar.gz/<SHA>' | tar xz -C /tmp" >&2
    echo "[preflight]   msprobe: python3 -m pip install --no-deps --target $MSPROBE_DIR mindstudio-probe" >&2
    echo "[preflight]            (+ matplotlib 3.10.9 pyparsing cycler kiwisolver fonttools python-dateutil packaging pillow)" >&2
    exit 65
  fi
done

# The loop above only proves the paths exist. A partially cleaned /tmp -- the exact failure mode
# this preflight exists for -- leaves the directory in place with a broken import, and that import
# error would only surface inside the first training step, minutes after the window opened. Import
# msprobe here with the same interpreter and PYTHONPATH the run uses; the config is parsed too
# because the profiler reads it at start(), not at import.
if ! python3 -c "from msprobe.pytorch import PrecisionDebugger" >"$OUT/msprobe-import.log" 2>&1; then
  echo "[preflight] msprobe does not import with this interpreter/PYTHONPATH -> $OUT/msprobe-import.log" >&2
  echo "[preflight] without it the profiler is a no-op and the au positive arm cannot dump -> abort" >&2
  echo "[preflight]   reinstall: python3 -m pip install --no-deps --target $MSPROBE_DIR mindstudio-probe" >&2
  echo "[preflight]              (+ matplotlib 3.10.9 pyparsing cycler kiwisolver fonttools python-dateutil packaging pillow)" >&2
  exit 65
fi
if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$MSPROBE_CONFIG" >/dev/null 2>&1; then
  echo "[preflight] $MSPROBE_CONFIG is not valid JSON -> the tool cannot start collecting" >&2
  exit 65
fi

python3 "$SCRIPT_DIR/patch-critic-sdpa.py" --tree "$SRC" --check >/dev/null 2>&1
FA2_RC=$?
case $FA2_RC in
  0) FA2_STATE=already-patched ;;
  1)
    if [ "$AUTO_PATCH" = "1" ]; then
      echo "[preflight] value-head attn_implementation is still flash_attention_2 -> site patching to sdpa"
      if ! python3 "$SCRIPT_DIR/patch-critic-sdpa.py" --tree "$SRC"; then
        echo "[preflight] site patch failed -> abort before touching the GPU" >&2
        exit 65
      fi
      FA2_STATE=patched-now
    else
      echo "[preflight] tree is not sdpa-patched and AUTO_PATCH=0" >&2
      echo "[preflight] the critic will die in load_valuehead_model (no flash-attn on this box)" >&2
      exit 65
    fi ;;
  *)
    echo "[preflight] patch-critic-sdpa.py --check rc=$FA2_RC (anchor shape changed?) -> inspect verl/utils/model.py by hand" >&2
    exit 65 ;;
esac

# trl is a *conditional* dependency: `load_valuehead_model` tries
# AutoModelForTokenClassification first and only falls back to trl's value-head wrapper when that
# raises. For a config class that is in the token-classification mapping (qwen2 is, in the pinned
# transformers) branch 1 wins and trl is never imported -- so a wiped /tmp/m5-trl-0.27.0 must not
# abort an otherwise good window. The probe reads config.json only (no weights, no GPU) and its
# "undecidable" answer keeps the old strict rule, so an unknown can only be conservative.
VH_JSON=$OUT/valuehead-branch.json
python3 "$SCRIPT_DIR/probe-valuehead-branch.py" --model "$MODEL_PATH" --json "$VH_JSON" >"$OUT/valuehead-branch.log" 2>&1
VH_RC=$?
case $VH_RC in
  0) echo "[preflight] value-head branch: $(grep -o '"branch": "[a-z_]*"' "$VH_JSON" 2>/dev/null | head -1)" ;;
  65)
    echo "[preflight] critic model needs the trl value-head fallback and $TRL_DIR is missing" >&2
    echo "[preflight] install it: python3 -m pip install --no-deps --target $TRL_DIR trl==0.27.0" >&2
    exit 65 ;;
  *)
    if [ ! -e "$TRL_DIR" ]; then
      echo "[preflight] value-head branch undecidable (probe rc=$VH_RC) and $TRL_DIR is missing" >&2
      echo "[preflight] cannot rule out the trl fallback -> strict rule aborts instead of risking the window" >&2
      exit 65
    fi
    echo "[preflight] value-head branch undecidable (probe rc=$VH_RC); continuing because $TRL_DIR exists" >&2 ;;
esac

EXTRA=(
  "global_profiler.tool=precision_debugger"
  "global_profiler.steps=[1]"
  "global_profiler.save_path=$PROF"
  "global_profiler.global_tool_config.precision_debugger.config_path=$MSPROBE_CONFIG"
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

START=$(date -u +%FT%TZ)
echo "=== pdbg-critic arm=$ARM stages=$STAGES src=$SRC gpu=$G prof=$PROF fa2=$FA2_STATE start $START ==="
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
END=$(date -u +%FT%TZ)
echo "=== pdbg-critic arm=$ARM exit=$RC end $END ==="
echo "--- dump tree ---"
find "$PROF" -maxdepth 5 2>/dev/null | head -30
echo "--- counts ---"
echo "unknown_stage_warnings=$(grep -c 'Unknown precision_debugger stage' "$LOG" 2>/dev/null)"
echo "train_batch_warnings=$(grep -c 'Unknown precision_debugger stage: train_batch' "$LOG" 2>/dev/null)"
echo "critic_timing_lines=$(grep -c 'timing_s/update_critic' "$LOG" 2>/dev/null)"

# --- machine-readable arm summary -------------------------------------------------------
M5_ARM=$ARM M5_STAGES=$STAGES M5_RC=$RC M5_SRC=$SRC M5_G=$G M5_PROF=$PROF M5_LOG=$LOG \
M5_FA2_STATE=$FA2_STATE M5_START=$START M5_END=$END M5_MSPROBE_CONFIG=$MSPROBE_CONFIG \
M5_MODEL_PATH=$MODEL_PATH M5_TRAIN_FILE=$TRAIN_FILE M5_TEST_FILE=$TEST_FILE \
M5_VH_JSON=$VH_JSON M5_TRL=$TRL_DIR M5_PATCH_SCRIPT=$SCRIPT_DIR/patch-critic-sdpa.py \
python3 - > "$ARM_JSON.tmp" <<'PY'
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

E = os.environ
prof = Path(E["M5_PROF"])
log = Path(E["M5_LOG"])
dumps = []
if prof.is_dir():
    for p in sorted(prof.rglob("*")):
        if p.is_file():
            dumps.append({"path": str(p.relative_to(prof)), "size": p.stat().st_size})

text = log.read_text(errors="replace") if log.is_file() else ""
# Console lines are written through Ray with an ANSI-coloured task prefix, so `^step:` matches
# nothing (0/104 archived logs). Count the per-step marker instead: `timing_s/step` appears once
# per training step and never on the step-0 validation line. Proven on the archive by
# check-log-counters.py --verify (which also shows why dropping the anchor is wrong: values such
# as `timing_s/step:8.2` then read as step 8, and checkpoint paths add more).
step_re = re.compile(r"step:(\d+)\s+-\s+")
step_numbers = sorted({int(m.group(1)) for line in text.splitlines() if "timing_s/step" in line
                       for m in [step_re.search(line)] if m})
model_py = Path(E["M5_SRC"]) / "verl" / "utils" / "model.py"
model_text = model_py.read_text() if model_py.is_file() else None

# Which branch of load_valuehead_model built the critic (see probe-valuehead-branch.py). Recorded
# so the evidence page can say "Qwen2ForTokenClassification with a fresh 1-label head", not "a
# value model" in the abstract; `deps.trl_target` then reads as a fallback's presence, not a need.
try:
    valuehead = json.loads(Path(E["M5_VH_JSON"]).read_text())
except Exception as exc:
    valuehead = {"decided": False, "error": f"{type(exc).__name__}: {exc}"}

# The tree state comes from the same classifier the preflight uses, so V4 gates on "the
# value-head block is patched" rather than on a file-wide literal count. Upstream 3fdf2b54 /
# 31cfccb4 carry both FA2 literals inside that block, so the two agree today; if a future
# upstream adds one elsewhere, a file-wide counter would report a false failure.
fa2 = {}
if model_text is not None:
    try:
        spec = importlib.util.spec_from_file_location("patch_critic_sdpa", E["M5_PATCH_SCRIPT"])
        pcs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pcs)
        fa2 = pcs.analyze(model_text, "sdpa")
        fa2["flash_attention_2_left"] = model_text.count('attn_implementation="flash_attention_2"')
        fa2["model_py_sha256"] = hashlib.sha256(model_text.encode()).hexdigest()
        fa2["preflight_state"] = E["M5_FA2_STATE"]
    except Exception as exc:  # keep the summary complete: V4 then fails loudly, judge still runs
        fa2 = {"state": None, "error": f"{type(exc).__name__}: {exc}"}

out = {
    "arm": E["M5_ARM"],
    "stages": E["M5_STAGES"],
    "src": E["M5_SRC"],
    "gpu": E["M5_G"],
    "prof": E["M5_PROF"],
    "log": E["M5_LOG"],
    "log_bytes": len(text),
    "start_utc": E["M5_START"],
    "end_utc": E["M5_END"],
    "exit_code": int(E["M5_RC"]),
    "steps_seen": len(step_numbers),
    "step_numbers": step_numbers,
    "update_critic_occurrences": text.count("timing_s/update_critic"),
    "update_critic_any_mention": text.count("update_critic"),
    "values_timing_occurrences": text.count("timing_s/values"),
    "unknown_stage_warnings": text.count("Unknown precision_debugger stage"),
    "train_batch_warnings": text.count("Unknown precision_debugger stage: train_batch"),
    "dump_file_count": len(dumps),
    "dump_files": dumps,
    "fa2": fa2,
    "valuehead_branch": valuehead,
    "deps": {
        "msprobe_config": os.path.exists(E["M5_MSPROBE_CONFIG"]),
        "msprobe_import_ok": True,
        "trl_target": os.path.exists(E["M5_TRL"]),
        "model": os.path.exists(E["M5_MODEL_PATH"]),
        "train_file": os.path.exists(E["M5_TRAIN_FILE"]),
        "test_file": os.path.exists(E["M5_TEST_FILE"]),
    },
}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
sys.stdout.write("\n")
PY
# Only replace the arm summary when the generator actually produced one: a truncated file would
# crash the judge instead of letting it return a verdict.
if [ -s "$ARM_JSON.tmp" ]; then
  mv "$ARM_JSON.tmp" "$ARM_JSON"
  echo "arm_summary=$ARM_JSON"
else
  echo "[warn] arm summary is empty -- no $ARM_JSON written; fix before judging this arm" >&2
fi

rm -rf "$OUT/ckpt-$ARM"
exit $RC
