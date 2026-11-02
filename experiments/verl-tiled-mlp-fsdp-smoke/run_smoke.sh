#!/usr/bin/env bash
# Drives the TiledMLP wiring smoke matrix. Run on 4090-1 from /tmp/m5-tiled-mlp.
# Usage: bash run_smoke.sh [case ...]   (default: all cases)
set -u
cd /tmp/m5-tiled-mlp || exit 1
PY=$HOME/miniconda3/envs/verl-e2e/bin/python
PRISTINE=/tmp/m5-tiled-mlp/verl-6879e1ab07
PATCHED=/tmp/m5-tiled-mlp/verl-6879e1ab07-patched
mkdir -p out logs

run() {  # name tree strategy enabled shards seq batch [extra args...]
  local name=$1 tree=$2 strategy=$3 enabled=$4 shards=$5 seq=$6 batch=$7
  shift 7
  echo "== $name (tree=$(basename "$tree") $strategy enabled=$enabled shards=$shards seq=$seq batch=$batch $*)"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$tree" "$PY" smoke_tiled_mlp.py \
    --strategy "$strategy" --enabled "$enabled" --shards "$shards" \
    --seq-len "$seq" --batch "$batch" --rdzv "/tmp/m5-tiled-mlp/rdzv-$name" \
    --out "out/$name.json" "$@" >"logs/$name.log" 2>&1
  echo "   exit=$? $(grep -o '{"strategy.*}' "logs/$name.log" | tail -1)"
}

if [ $# -eq 0 ]; then
  run base_fsdp2_e0        "$PRISTINE" fsdp2 0 4 8 2
  run unwired_fsdp2_e1     "$PRISTINE" fsdp2 1 4 8 2
  run wired_fsdp2_e1_s4    "$PATCHED"  fsdp2 1 4 8 2
  run wired_fsdp2_e1_s8    "$PATCHED"  fsdp2 1 8 8 2
  run base_fsdp1_e0        "$PRISTINE" fsdp  0 4 8 2
  run wired_fsdp1_e1_s4    "$PATCHED"  fsdp  1 4 8 2
  run base_fsdp2_e0_n18    "$PRISTINE" fsdp2 0 4 9 2
  run wired_fsdp2_uneven   "$PATCHED"  fsdp2 1 4 9 2
  run wired_fsdp2_s16      "$PATCHED"  fsdp2 1 16 8 2
  # chunk-count corner: torch.chunk returns fewer chunks than requested below
  # ceil(N/shards) == shards; the "last shard" finalisation may never run.
  run base_fsdp2_e0_n3     "$PRISTINE" fsdp2 0 4 3 1
  run wired_fsdp2_e1_n3    "$PATCHED"  fsdp2 1 4 3 1
  run base_fsdp2_e0_n5     "$PRISTINE" fsdp2 0 4 5 1
  run wired_fsdp2_e1_n5    "$PATCHED"  fsdp2 1 4 5 1
  # FSDP1 variants that could plausibly matter
  run wired_fsdp1_e1_orig  "$PATCHED"  fsdp  1 4 8 2 --use-orig-params 1
  run wired_fsdp1_e1_noresh "$PATCHED" fsdp  1 4 8 2 --reshard-after-forward 0
fi
