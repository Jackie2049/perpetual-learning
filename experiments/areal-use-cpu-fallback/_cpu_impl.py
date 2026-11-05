"""抄本：上游 `fill_indices_cpu`(kernels.py:108-152) 的函数体，逐行按序搬用。

与上游的差别只有三处，都写在 check_snapshot.py 的 canon() 里做归一化后比对：
  · 形参 `write_offsets` → `naive_write_offsets`（纯改名）
  · 输出张量加 `device=tokens_per_expert_group.device`（原版直接在 CPU 上建；
    被测设备就是 CPU，两者等价，这一处只为让录音能在本机跑起来）
  · 探针关键字 `stride`（默认 None = 上游原样；对照臂传 m_sizes）
"""
import torch


def fill(tokens_per_expert_group, start_index_values, naive_write_offsets,
         experts_per_rank, num_ranks, max_len, *, stride=None):
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int32,
        device=tokens_per_expert_group.device,
    )
    for e in range(experts_per_rank):
        write_start = naive_write_offsets[e].item()
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index, start_index + (end_idx - write_start), dtype=torch.int32,
                )
            write_start += length if stride is None else int(stride[e].item())
    return permuted_indices
