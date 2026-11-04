#!/usr/bin/env bash
# Build a throwaway tree that satisfies run-critic-pdbg.sh's preflight, so the whole pipeline
# (preflight -> auto site patch -> run -> arm JSON -> verdict) can be exercised with no GPU.
#
# The stand-in example script prints log lines in the shape REAL Ray console output has: an
# ANSI-coloured task prefix in front of `step:1 - ...`, a step-0 validation line with no
# `timing_s/*` keys, and periodic timing keys. The 2026-11-05 01:0x fixture printed bare
# `step:1 - ...` lines instead, which is exactly why its end-to-end pass could not reveal that
# `^step:` matches no real log line. See ../check-log-counters.py for the real-log proof.
#
# Usage: bash make-mock-tree.sh <dest-dir>
set -euo pipefail
DEST=${1:?usage: make-mock-tree.sh <dest-dir>}
rm -rf "$DEST"
mkdir -p "$DEST"/{model,data,msprobe,trl,src/verl/utils,src/examples/grpo_trainer}

# preflight only checks existence; every file is a placeholder
printf '{"model_type":"qwen2"}\n' > "$DEST/model/config.json"
printf 'not-a-parquet\n' > "$DEST/data/train.parquet"
printf 'not-a-parquet\n' > "$DEST/data/test.parquet"
printf '{"dump_path":"%s/mock-dump"}\n' "$DEST" > "$DEST/msprobe/config.json"

# The site patch needs `def load_valuehead_model(` at column 0 with exactly two FA2 anchors
# inside the block and one identical literal outside it (the outside one must stay untouched).
cat > "$DEST/src/verl/utils/model.py" <<'PY'
# stand-in for verl/utils/model.py -- fixture only
DEFAULT_IMPL = 'attn_implementation="flash_attention_2"'


def load_valuehead_model(model_name, config, torch_dtype=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
    )
    value_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
    )
    return value_model


def other_helper():
    return 'attn_implementation="flash_attention_2"'
PY

# Stand-in for examples/grpo_trainer/run_qwen3_4b_fsdp.sh. Parses the two overrides it cares
# about and emits a reality-shaped 2-step log; writes the msprobe dump tree for the positive
# arm only.
cat > "$DEST/src/examples/grpo_trainer/run_qwen3_4b_fsdp.sh" <<'SH'
#!/usr/bin/env bash
for a in "$@"; do
  case $a in
    global_profiler.save_path=*) PROF=${a#*=} ;;
    *precision_debugger.stages=*) ST=${a#*=} ;;
  esac
done
P=$'\033[36m(TaskRunnerV1 pid=4242)\033[0m '
echo "${P}step:0 - val-core/openai/gsm8k/acc/mean@1:np.float64(0.075) - val-aux/num_turns/min:np.int64(2)"
echo "${P}step:1 - global_seqlen/min:8616.0 - timing_s/gen:np.float64(2.11) - timing_s/values:np.float64(0.31) - timing_s/update_critic:np.float64(0.52) - timing_s/update_actor:np.float64(3.02) - timing_s/step:np.float64(8.21)"
echo "${P}step:2 - global_seqlen/min:8616.0 - timing_s/gen:np.float64(2.04) - timing_s/values:np.float64(0.30) - timing_s/update_critic:np.float64(0.51) - timing_s/update_actor:np.float64(3.00) - timing_s/step:np.float64(8.11)"
echo "${P}WARNING:verl.utils.profiler.precision_debugger_profile:Unknown precision_debugger stage: train_batch"
echo "${P}WARNING:verl.utils.profiler.precision_debugger_profile:Unknown precision_debugger stage: train_batch"
echo "saving to ${PROF%/prof-*}/ckpt-x/global_step_46/actor"
if [ "$ST" = "[actor_update]" ]; then
  mkdir -p "$PROF/step_1/actor_update/step0/rank0"
  printf '{}'      > "$PROF/step_1/actor_update/step0/rank0/construct.json"
  printf '{"x":1}' > "$PROF/step_1/actor_update/step0/rank0/stack.json"
  printf '{"y":2}' > "$PROF/step_1/actor_update/step0/rank0/dump.json"
fi
exit 0
SH
chmod +x "$DEST/src/examples/grpo_trainer/run_qwen3_4b_fsdp.sh"
echo "mock tree ready: $DEST"
