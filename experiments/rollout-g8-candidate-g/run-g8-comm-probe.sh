#!/usr/bin/env bash
# Candidate G — archived run of the 4-GPU / 2-GPU comm+compute probe (2026-11-02).
# Same probe code as the ad-hoc round; collected here so every archived number has a raw log.
set -uo pipefail
cd /home/zxw/venvs
export PATH=/home/zxw/venvs/rollout-g8/bin:$PATH
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

run() {
  local n=$1 set=$2 port=$3
  echo "--- n=$n CVD=$set shm=$NCCL_SHM_DISABLE $(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES=$set timeout 200 torchrun --nproc_per_node=$n \
    --master_port=$port g8-comm-compute-probe.py 2>&1 \
    | grep -E "^CVD|^allreduce|^matmul|Error|error"
}

run 4 0,1,3,4 29701
run 4 4,5,6,7 29702
run 4 3,4,5,6 29703
run 4 0,1,4,5 29704
run 2 0,1 29705
run 2 4,5 29706
run 2 0,4 29707
echo "=== NCCL_SHM_DISABLE=1 (socket transport control) ==="
export NCCL_SHM_DISABLE=1
run 4 0,1,3,4 29708
run 4 4,5,6,7 29709
echo "probe done $(date -u +%FT%TZ)"
