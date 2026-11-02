#!/usr/bin/env python3
"""4-GPU all-reduce + compute microbenchmark for the candidate-G placement gap.

Runs the same code on GPUSET via CUDA_VISIBLE_DEVICES with NCCL_P2P_DISABLE=1
(matching the training stack), so the two GPU sets can be compared on:
  - collective bandwidth (FSDP all-gather/reduce-scatter proxy, SHM transport)
  - bf16 matmul throughput (SM clock / power proxy)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,3,4 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 \
      --master_port=29531 g8-comm-compute-probe.py
"""
import os
import time

import torch
import torch.distributed as dist


def bench_allreduce(size_mb, iters=30, warmup=5):
    n = size_mb * 1024 * 1024 // 4
    t = torch.ones(n, dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / iters * 1e3, size_mb * iters / dt / 1024.0


def bench_matmul(m=8192, iters=10, warmup=3):
    a = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    return 2 * m**3 * iters / dt / 1e12


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    name = torch.cuda.get_device_name(local_rank)
    rows = []
    for size_mb in (16, 64, 256):
        ms, gbps = bench_allreduce(size_mb)
        rows.append((size_mb, ms, gbps))
    tf = bench_matmul()
    if rank == 0:
        print(f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} dev={name} "
              f"ranks={dist.get_world_size()}")
        for size_mb, ms, gbps in rows:
            print(f"allreduce {size_mb:4d}MB {ms:8.3f} ms {gbps:7.2f} GB/s")
        print(f"matmul 8192^3 bf16 {tf:7.1f} TFLOPS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
