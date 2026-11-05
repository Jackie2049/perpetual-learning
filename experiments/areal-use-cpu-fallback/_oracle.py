"""GPU 排布的 CPU 复刻（语义定义）：按 m_sizes 分段，段内按 rank 顺序写连续索引。

Triton 内核（kernels.py:16-62）对每个 expert 从 `write_offsets[e]` 起、逐 rank 顺序写
`write_offset + chunk_offsets`；一个 expert 写完后下一个从**对齐后的** `m_offsets` 起。
写不到的位置保持 -1（内核不填空槽）。
"""
import torch


def gpu_layout(flat, alignment, max_len, experts, ranks):
    t = torch.tensor(flat, dtype=torch.int32)
    start_v = torch.cumsum(t, 0) - t
    per_expert = t.view(ranks, experts).sum(0).to(torch.int64)
    m_sizes = (torch.clamp_min(per_expert, alignment) + alignment - 1) // alignment * alignment
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes
    out = torch.full((max_len,), -1, dtype=torch.int32)
    for e in range(experts):
        pos = int(write_offsets[e])
        for r in range(ranks):
            i = r * experts + e
            ln, st = int(t[i]), int(start_v[i])
            for k in range(ln):
                if pos + k < max_len:
                    out[pos + k] = st + k
            pos += ln
    return out


def first_order(out, flat, experts, ranks):
    """每个 (rank, expert) 的槽位起点；None = 该段一个位置都没写。"""
    t = torch.tensor(flat, dtype=torch.int32)
    start_v = (torch.cumsum(t, 0) - t).tolist()
    vals = out.tolist()
    res = {}
    for r in range(ranks):
        for e in range(experts):
            i = r * experts + e
            pos = [p for p, v in enumerate(vals) if start_v[i] <= v < start_v[i] + flat[i]]
            res[(r, e)] = pos[0] if pos else None
    return res


def inversions(first, experts, ranks):
    n = 0
    for r in range(ranks):
        seen = [first[(r, e)] for e in range(experts) if first[(r, e)] is not None]
        n += sum(1 for a, b in zip(seen, seen[1:]) if b < a)
    return n
