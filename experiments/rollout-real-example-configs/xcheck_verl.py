#!/usr/bin/env python3
"""与真实 verl（1a9e711e）逐位核对本实验的切分形状。

运行环境（CPU torch，不进本仓库依赖）：

    uv venv /tmp/verl-seqlb-venv --python 3.12
    VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install numpy packaging ray tensordict omegaconf transformers codetiming
    VERL_ROOT=../../verl /tmp/verl-seqlb-venv/bin/python xcheck_verl.py

核对项：
  1. `force_group_size=1` + 真实配置预算：分区与 `rearrange_micro_batches` 完全一致
  2. `force_group_size=n`：组级 LDM 展开回样本下标后完全一致
  3. `num_batches_divided_by=vpp`：roundup 后的批数与分区一致
  4. 组完整 + vPP 顶破 `num_groups`：真实实现必须抛 AssertionError，模型记为 'cap'
  5. 预算小于单条长度：真实实现必须在前置断言处失败，模型记为 'shape'
"""
import os
import sys

import torch
from tensordict import TensorDict

HERE = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.environ.get('VERL_ROOT', os.path.join(HERE, '..', '..', 'verl'))
sys.path.insert(0, os.path.abspath(VERL_ROOT))
sys.path.insert(0, HERE)

from verl.utils.seqlen_balancing import rearrange_micro_batches  # noqa: E402
from real_example_configs import micro_split, num_micro_batches     # noqa: E402


def build(seq_lens):
    b = len(seq_lens)
    m = max(seq_lens)
    mask = torch.zeros(b, m, dtype=torch.long)
    for i, sl in enumerate(seq_lens):
        mask[i, :sl] = 1
    ids = torch.randint(1, 100, (b, m))
    return TensorDict({'input_ids': ids, 'attention_mask': mask}, batch_size=[b])


# (batch, per-group, lens_lo, lens_hi, budget, force_group_size, vpp)
CASES = [
    (160, 5, 1200, 4200, 24576, 1, None),
    (160, 5, 1200, 4200, 24576, 5, None),
    (128, 8, 1500, 9000, 32768, 1, None),
    (128, 8, 1500, 9000, 32768, 8, None),
    (64, 8, 1500, 9000, 30720, 8, 2),
    (64, 8, 1500, 9000, 30720, 2, 2),
    (40, 5, 800, 2000, 16384, 5, None),
    (16, 8, 5000, 10000, 10240, 8, None),   # 预算 < 单条 → shape 失败
    (64, 8, 1400, 1600, 8192, 8, 4),        # 预算 < 单条 → shape 失败（第二个 shape 形态）
    (32, 8, 1990, 2010, 16384, 8, 3),       # roundup(4 批, 3)=6 > 4 组 → cap 失败
]


def main():
    torch.manual_seed(0)
    ok = bad = 0
    infeasible_real = 0
    infeasible_model = 0
    for (n, per, lo, hi, budget, g, vpp) in CASES:
        lens = [torch.randint(lo, hi, (1,)).item() for _ in range(n)]
        td = build(lens)
        model_n, st = num_micro_batches(n, lens, budget, force_group_size=g,
                                        num_batches_divided_by=vpp)
        real_parts = None
        real_err = None
        try:
            mbs, idx = rearrange_micro_batches(
                td, max_token_len=budget, num_batches_divided_by=vpp,
                force_group_size=g, use_dynamic_bsz_balance=True)
            real_parts = idx
        except AssertionError as e:
            real_err = str(e)

        label = 'n=%d g=%d vpp=%s budget=%d' % (n, g, vpp, budget)
        if st != 'ok':
            infeasible_model += 1
            if real_err is not None:
                infeasible_real += 1
                print('%-46s 两侧同判不可行（%s）' % (label, st))
            else:
                bad += 1
                print('%-46s MISMATCH: 模型判 %s，真实可切' % (label, st))
            continue
        if real_err is not None:
            bad += 1
            print('%-46s MISMATCH: 真实报错 %r，模型判可切' % (label, real_err[:60]))
            continue
        mine = micro_split(lens, model_n, force_group_size=g)
        if sorted(map(sorted, mine)) == sorted(map(sorted, real_parts)):
            ok += 1
        else:
            bad += 1
            print('%-46s MISMATCH 分区不同' % label)
            print('   model:', mine[:3])
            print('   real :', real_parts[:3])
    print('一致 %d｜不一致 %d｜两侧同判不可行 %d（模型 %d）｜共 %d 个形态'
          % (ok, bad, infeasible_real, infeasible_model, len(CASES)))
    return 1 if bad or infeasible_real != infeasible_model else 0


if __name__ == '__main__':
    sys.exit(main())
