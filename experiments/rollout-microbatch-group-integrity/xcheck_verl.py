#!/usr/bin/env python3
"""与真实 verl（1a9e711e）逐位核对 micro_split 的三种形态。

运行环境（CPU torch，不进本仓库依赖）：

    uv venv /tmp/verl-seqlb-venv --python 3.12
    VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install numpy packaging ray tensordict omegaconf transformers codetiming
    /tmp/verl-seqlb-venv/bin/python xcheck_verl.py

核对项：
  1. force_group_size=1、无 vPP：与 rearrange_micro_batches 分区完全一致（§13.34 已核的形态，回归）
  2. force_group_size=4、无 vPP：组级 LDM 展开回样本下标后完全一致
  3. force_group_size=4 + num_batches_divided_by=2：既有 vPP 轮整又组完整的形态
  4. 组完整 + vPP 下 roundup 顶破 num_groups 上限时，真实实现必须抛 AssertionError
     （本模型把它记为 infeasible，据此确认「不是本模型算不动，是真实代码走不到」）
"""
import os
import sys

import torch
from tensordict import TensorDict

VERL_ROOT = os.environ.get('VERL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'verl'))
sys.path.insert(0, os.path.abspath(VERL_ROOT))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rollout-pg-boundaries'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.utils.seqlen_balancing import rearrange_micro_batches  # noqa: E402
from microbatch_group_integrity import micro_split  # noqa: E402


def build(seq_lens):
    b = len(seq_lens)
    m = max(seq_lens)
    mask = torch.zeros(b, m, dtype=torch.long)
    for i, sl in enumerate(seq_lens):
        mask[i, :sl] = 1
    ids = torch.randint(1, 100, (b, m))
    return TensorDict({'input_ids': ids, 'attention_mask': mask}, batch_size=[b])


CASES = [
    # (n_groups, n_per, budget, force_group_size, vpp)
    (8, 8, 16384, 1, None),
    (8, 8, 16384, 8, None),
    (8, 8, 8192, 8, None),
    (6, 4, 4096, 4, None),
    (6, 4, 4096, 4, 2),
    (8, 8, 8192, 8, 2),
    (8, 8, 16384, 8, 2),
    # 8 组、vPP=3、预算装下整批：组完整 → min(8, 1)=1 → roundup(1,3)=3… 这里要的是
    # 顶破上限的形态：预算只够 9 批的半批切法下 n_micro_raw=8 已被 num_groups 顶住，
    # 见 case 8（预算 16384、长度 2000×64 时 ceil=8，roundup 到 9 > 8）。
    (8, 8, 16384, 8, 3),
]


def main():
    ok = bad = infeasible = 0
    for gi, (ng, nper, budget, fg, vpp) in enumerate(CASES):
        rng = torch.Generator().manual_seed(1000 + gi)
        # 长度上界压在预算以内：rearrange_micro_batches 开头断言 max_token_len >= max_seq_len，
        # 单条超预算属于另一个前置条件，不是本批要核的切分形状。
        lens = [int(torch.normal(2000, 700, (1,), generator=rng).clamp(300, budget - 64).item())
                for _ in range(ng * nper)]
        batch = build(lens)
        try:
            _, real_idx = rearrange_micro_batches(
                batch, max_token_len=budget, force_group_size=fg,
                same_micro_num_in_dp=False, num_batches_divided_by=vpp,
            )
        except AssertionError as e:
            model_parts, info = micro_split(lens, budget, force_group_size=fg,
                                            num_batches_divided_by=vpp)
            msg = str(e)
            # 该断言没有消息文本，只能靠位置判断：判据是「模型侧也算出顶破上限」
            if model_parts is None and info['n_micro_before_cap'] > len(lens) // fg:
                infeasible += 1
                print(f'case {gi}: real=AssertionError(num_micro_batches<=num_groups); '
                      f'model infeasible={model_parts is None} '
                      f'(n_micro_before_cap={info["n_micro_before_cap"]}, groups={len(lens)//fg})')
            else:
                bad += 1
                print(f'case {gi}: UNEXPECTED AssertionError: {msg}')
            continue
        model_parts, _ = micro_split(lens, budget, force_group_size=fg, num_batches_divided_by=vpp)
        real_norm = [sorted(p) for p in real_idx]
        model_norm = [sorted(p) for p in model_parts]
        if real_norm == model_norm:
            ok += 1
            print(f'case {gi}: OK  {len(real_idx)} partitions  equal')
        else:
            bad += 1
            print(f'case {gi}: MISMATCH real={real_norm} model={model_norm}')
    print(f'\n一致 {ok}｜不一致 {bad}｜真实实现断言失败 {infeasible}（共 {len(CASES)} 个形态）')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
