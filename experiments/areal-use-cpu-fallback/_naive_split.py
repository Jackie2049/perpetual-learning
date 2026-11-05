"""抄本 A：上游 `generate_permute_indices` 算偏移那几行的逐字抄本（kernels.py:195-210）。"""
import torch


def split(tokens_per_expert_group, experts_per_rank, num_ranks, max_len, alignment):
    start_index_values = torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group

    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes
    return start_index_values, write_offsets, m_sizes, m_offsets
