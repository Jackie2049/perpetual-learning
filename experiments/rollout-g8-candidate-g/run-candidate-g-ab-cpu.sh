#!/usr/bin/env bash
# Candidate G — CPU/host-memory locality matrix for the placement effect (2026-11-02).
# numactl is not installed on 4090-1; default memory policy is first-touch, and taskset
# confines every process in the tree (Ray raylet -> workers, vLLM engines) to one NUMA
# node's CPUs, so host allocations follow that node. Topology is symmetric (56 CPUs each):
#   node0 = 0-27,56-83 ; node1 = 28-55,84-111 ; GPUs 0-3 -> node0, 4-7 -> node1.
# 2x2 cells (same GPUs, only the CPU/memory node changes):
#   pB3 = 4,5,6,7 + node1 CPUs  (all GPUs local)      pB4 = 4,5,6,7 + node0 CPUs (all remote)
#   pA4 = 0,1,3,4 + node0 CPUs  (3/4 GPUs local)      pA3 = 0,1,3,4 + node1 CPUs (1/4 local)
set -uo pipefail
cd /home/zxw/venvs
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

state() {
  echo "=== state $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw \
    --format=csv,noheader
}

run() {
  local mode=$1 gpus=$2 cpus=$3
  state
  echo ">>> $mode gpus=$gpus cpus=$cpus start $(date -u +%FT%TZ)"
  taskset -c "$cpus" env GPUS=$gpus MODE=$mode bash run-candidate-g.sh
  local rc=$?
  echo ">>> $mode rc=$rc end $(date -u +%FT%TZ)"
  sleep 10
  state
}

echo "chain-ab-cpu start $(date -u +%FT%TZ)"
run pB3 4,5,6,7 28-55,84-111
run pB4 4,5,6,7 0-27,56-83
run pA4 0,1,3,4 0-27,56-83
run pA3 0,1,3,4 28-55,84-111
echo "chain-ab-cpu done $(date -u +%FT%TZ)"
