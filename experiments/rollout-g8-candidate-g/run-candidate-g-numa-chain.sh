#!/usr/bin/env bash
set -uo pipefail
cd /home/zxw/venvs
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
echo "chain2 start $(date -u +%FT%TZ)"
GPUS=5,6 MODE=g2n bash run-candidate-g.sh
echo "g2n rc=$? $(date -u +%FT%TZ)"
GPUS=4,5,6,7 MODE=g4n bash run-candidate-g.sh
echo "g4n rc=$? $(date -u +%FT%TZ)"
echo "chain2 done $(date -u +%FT%TZ)"
