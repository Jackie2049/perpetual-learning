"""Minimal flash_mla stub for civilian-port mock chain (SM89).

prefill (flash_mla_sparse_fwd) 与 decode (flash_mla_with_kvcache) 提供 torch 回退：
- prefill 的 kv_a_with_mqa 是 [total_k, 1, D] 的已解压 KV（MQA 单头），直接 gather 注意力；
- decode 的 k_cache 是 packed uint8（V41_FORMAT: 512B e4m3 + 16B ue8m0 scale/token），
  近似解包（只取 e4m3 数据、忽略 scale）后做注意力；
- 数值不保（mock 只验证链路可跑、输出有限且形状正确），fused_norm_rope_attn_rope_cast 仍 raise。
"""
import torch


def _dequant_v41(flat_uint8: torch.Tensor) -> torch.Tensor:
    """[N, bpt] uint8 -> [N, 512] fp32（V41_FORMAT: 前 512 字节 e4m3，scale 忽略）。"""
    return torch.nan_to_num(flat_uint8[:, :512].view(torch.float8_e4m3fn).float())


def _mask_topk(scores: torch.Tensor, topk_length: torch.Tensor) -> torch.Tensor:
    total_q, H, topk = scores.shape
    if topk_length is None:
        return scores
    mask = torch.arange(topk, device=scores.device)[None, None, :] >= topk_length[:, None, None]
    return scores.masked_fill(mask.expand(total_q, H, topk), float("-inf"))


def flash_mla_sparse_fwd(query, kv_a_with_mqa, indices, sm_scale, attn_sink=None, topk_length=None, **kwargs):
    """torch 回退：MLA MQA 稀疏 prefill 注意力。

    query [total_q, H, D]（bf16），kv_a_with_mqa [total_k, 1, D]，indices [total_q, 1, topk]。
    返回 (output [total_q, H, D], max_logits, lse [total_q, H])。
    """
    q = query.float()
    k = kv_a_with_mqa.squeeze(1).float()
    idx = indices.squeeze(1)
    total_q, H, D = q.shape
    kq = k[idx]  # [total_q, topk, D]
    scores = torch.einsum("qhd,qtd->qht", q, kq) * sm_scale  # [total_q, H, topk]
    if attn_sink is not None:
        scores = scores + attn_sink.float()[None, :, None]
    scores = _mask_topk(scores, topk_length)
    max_logits = scores.max(dim=-1).values  # [total_q, H]
    lse = torch.logsumexp(scores, dim=-1)  # [total_q, H]
    probs = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("qht,qtd->qhd", probs, kq)  # [total_q, H, D]
    # 全掩码行（topk_length=0）LSE=-inf，置 0 保有限（CUDA graph 捕获下禁止 .any() 同步）
    out = out.masked_fill(torch.isinf(lse).unsqueeze(-1), 0.0)
    return out.to(query.dtype), max_logits.to(query.dtype), lse.to(torch.float32)


def flash_mla_with_kvcache(
    q,
    k_cache,
    indices_in_kvcache=None,
    sm_scale=None,
    *,
    block_table=None,
    cache_seqlens=None,
    head_dim_v=None,
    tile_scheduler_metadata=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=True,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    enable_batch_invariant=False,
    **kwargs,
):
    """torch 回退：MLA decode 注意力（packed V41 uint8 缓存近似解包，返回形状对齐 FlashMLA）。

    调用方（attention_backend.decode_mla_mqa_sparse）契约：
    - q [total_q, 1, H, D]（H 已被 pad 到 64/128），k_cache [num_seqs, block, 1, bpt] uint8
    - indices [total_q, 1, topk]
    - 返回 output [total_q, 1, H, Dv]、lse [total_q, H, 1]（调用方再 squeeze）。
    """
    scale = softmax_scale if softmax_scale is not None else sm_scale
    if scale is None:
        raise ValueError("flash_mla_with_kvcache: missing softmax_scale")
    qf = q.float().squeeze(1)  # [total_q, H, D]
    total_q, H, D = qf.shape
    if indices is None and indices_in_kvcache is not None:
        indices = indices_in_kvcache
    idx = indices.squeeze(1)  # [total_q, topk]
    rows = k_cache.reshape(-1, k_cache.shape[-1])
    keys = _dequant_v41(rows)  # [N, 512]
    kq = keys[idx]  # [total_q, topk, 512]
    scores = torch.einsum("qhd,qtd->qht", qf, kq) * scale
    if attn_sink is not None:
        scores = scores + attn_sink.float()[None, :, None]
    scores = _mask_topk(scores, topk_length)
    lse = torch.logsumexp(scores, dim=-1)  # [total_q, H]
    probs = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("qht,qtd->qhd", probs, kq)
    out = out.masked_fill(torch.isinf(lse).unsqueeze(-1), 0.0)
    dv = head_dim_v if head_dim_v is not None else out.shape[-1]
    return out[:, :, :dv].unsqueeze(1).to(q.dtype), lse.unsqueeze(-1).to(torch.float32)


def fused_norm_rope_attn_rope_cast(*a, **k):
    raise NotImplementedError("flash_mla.fused_norm_rope_attn_rope_cast not available on civilian port (SM89)")


def flash_mla_sparse_fwd_hc(*a, **k):
    raise NotImplementedError("flash_mla.flash_mla_sparse_fwd_hc not available on civilian port (SM89)")
