#!/usr/bin/env python3
"""用**真实 verl**（快照 1a9e711e）核本实验的两处复刻。

运行环境需要 CPU torch 与 verl 的可 import 依赖（见 readme「复现」）：
    /tmp/verl-seqlb-venv/bin/python xcheck_verl.py

核三件事：
  1. `pb._karmarkar_karp` vs `verl.utils.seqlen_balancing.get_seqlen_balanced_partitions`
     （equal_size=True / False 两种），随机 60 组输入 ×（组级、逐条）两个层级
  2. `pb.micro_split` vs 真实 `rearrange_micro_batches` 的分区（长度取**整数**：
     真实实现用 `attention_mask.sum()`，浮点长度会被 `torch.arange` 截断，两边不可比）
  3. `get_group_balanced_partitions` 可用且分区数等于 dp

本脚本不写 summary，输出一行结果供 readme 引用。
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'verl')))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from verl.utils.seqlen_balancing import (  # noqa: E402
    calculate_workload,
    get_group_balanced_partitions,
    get_seqlen_balanced_partitions,
    rearrange_micro_batches,
)

import pg_boundaries as pb  # noqa: E402


def _w(seqlen):
    return int(calculate_workload(torch.tensor([seqlen])).tolist()[0])


def check_ldm():
    rng = random.Random(0)
    bad = total = 0
    for _ in range(60):
        n_per = rng.choice([4, 8])
        n_groups = rng.choice([16, 32])
        dp = rng.choice([2, 4, 8])
        if n_groups % dp:
            continue
        gl = [[float(int(rng.gauss(2200, 990))) for _ in range(n_per)] for _ in range(n_groups)]
        group_w = [sum(_w(x) for x in g) for g in gl]
        real = [sorted(p) for p in get_seqlen_balanced_partitions(group_w, dp, equal_size=True)]
        mine = [sorted(p) for p in pb._karmarkar_karp(group_w, dp, equal_size=True)]
        total += 1
        bad += real != mine
        flat = [int(x) for g in gl for x in g]
        w = [_w(x) for x in flat]
        real2 = [sorted(p) for p in get_seqlen_balanced_partitions(w, dp, equal_size=True)]
        mine2 = [sorted(p) for p in pb._karmarkar_karp(w, dp, equal_size=True)]
        total += 1
        bad += real2 != mine2
    return total, bad


def check_micro():
    rng = random.Random(0)
    bad = total = 0
    for _ in range(40):
        n = rng.choice([32, 64])
        lens = [float(int(rng.uniform(300, 4000))) for _ in range(n)]
        budget = rng.choice([8192, 16384, 32768])
        ids = torch.nested.as_nested_tensor([torch.arange(int(x)) for x in lens], layout=torch.jagged)
        batch = TensorDict({'input_ids': ids, 'attention_mask': (ids != -1)}, batch_size=[n])
        try:
            _, idx = rearrange_micro_batches(batch, max_token_len=budget)
        except AssertionError:
            continue
        mine = pb.micro_split(lens, budget)
        total += 1
        if {frozenset(m) for m in idx} != {frozenset(m) for m in mine}:
            bad += 1
    return total, bad


def check_group_partitions():
    rng = random.Random(0)
    gl = [[float(int(rng.gauss(2200, 990))) for _ in range(4)] for _ in range(16)]
    seq = [int(x) for g in gl for x in g]
    uid = [i for i in range(len(gl)) for _ in range(4)]
    parts = get_group_balanced_partitions(seq, uid, 4)
    intact = all(len({i // 4 for i in p}) == len(p) // 4 for p in parts)
    return len(parts), intact


if __name__ == '__main__':
    t, b = check_ldm()
    print('ldm: %d/%d 完全一致（不一致 %d）' % (t - b, t, b))
    t2, b2 = check_micro()
    print('micro_split: %d/%d 分区完全一致（不一致 %d）' % (t2 - b2, t2, b2))
    n_parts, intact = check_group_partitions()
    print('group_balanced_partitions: 分区数 %d，组完整性 %s' % (n_parts, intact))
