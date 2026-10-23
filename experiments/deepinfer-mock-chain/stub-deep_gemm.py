"""Minimal deep_gemm stub for civilian-port smoke (SM89): keep imports alive,
provide torch fallbacks for the ops the mock chain actually exercises, raise
on anything else. Exact FP8/FP4 quant numerics are out of scope for the mock:
dequant simply casts to fp32 so outputs stay finite and the chain runs."""
import torch


class _NotImplemented:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"deep_gemm.{self._name} not available on civilian port (SM89)")

    def __getattr__(self, k):
        return _NotImplemented(f"{self._name}.{k}")


def _dq(q):
    """Cast quant tensor to fp32 (ignore per-channel scales for the mock).

    随机 e4m3 权重可能含 NaN 码型，回退计算需保持有限（否则 fused_sample 断言失败）。
    """
    return torch.nan_to_num(q.to(torch.float32))


def _gemm_nt(a, b, d, out_acc):
    out = _dq(a) @ _dq(b).t()
    if out_acc is not None:
        out = out + _dq(out_acc)
    d.copy_(out.to(d.dtype))


def bf16_gemm_nt(a, b, d, c=None):
    _gemm_nt(a, b, d, c)


def fp8_fp4_gemm_nt(x, weight, out, out_acc, recipe=(1, 1, 32)):
    _gemm_nt(x[0], weight[0], out, out_acc)


def cublaslt_gemm_nt(a, b, d, c=None):
    _gemm_nt(a, b, d, c)


def tf32_hc_prenorm_gemm(x, fn, out_mul, sqr_sum, n_splits=1):
    """Mock prenorm GEMM: out_mul = x @ fn.T, sqr_sum = ||x||^2 (no split-k)."""
    xf = x.to(torch.float32)
    f = fn.to(torch.float32)
    mul = xf @ f.t()
    s = (xf * xf).sum(-1)
    if out_mul.dim() == 2:
        out_mul.copy_(mul)
        sqr_sum.copy_(s)
    else:
        for k in range(out_mul.shape[0]):
            out_mul[k].copy_(mul)
            sqr_sum[k].copy_(s)
def fp8_einsum(equation, lhs, rhs, out, recipe=(1, 1, 32)):
    a = _dq(lhs[0])
    b = _dq(rhs[0])
    r = torch.einsum(equation, a, b)
    dst = out[0] if isinstance(out, tuple) else out
    dst.copy_(r.to(dst.dtype))


def transform_sf_into_required_layout(sf, n, k, num_groups=None, recipe=(1, 32, 32), is_sfa=False, disable_ue8m0_cast=False):
    # Layout transforms are ignored in the torch fallback; only shape identity matters.
    return sf


def transform_weights_for_mega_moe(src_l1, src_l2):
    # Torch fallback keeps HF expert weights in their original layout.
    return src_l1, src_l2


class SymmBuffer:
    """Backing storage for the fused mega-MoE path; x/x_sf/topk views are real CUDA tensors."""

    def __init__(
        self,
        group=None,
        num_experts=1,
        num_max_tokens_per_rank=1024,
        num_topk=1,
        hidden=128,
        intermediate_hidden=128,
        num_shared_experts=0,
        mma_type="fp8xfp4",
        activation="swiglu",
        base=None,
    ):
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        self.x = torch.empty((num_max_tokens_per_rank, hidden), dtype=torch.float8_e4m3fn, device="cuda")
        self.x_sf = torch.empty((num_max_tokens_per_rank, hidden // 32 // 4), dtype=torch.int32, device="cuda")  # packed ue8m0: 4 bytes per scale (PLATFORM.fp8_sf_type=int32 on CUDA)
        self.topk_idx = torch.empty((num_max_tokens_per_rank, num_topk), dtype=torch.int32, device="cuda")
        self.topk_weights = torch.empty((num_max_tokens_per_rank, num_topk), dtype=torch.bfloat16, device="cuda")
        self.buffer = torch.empty(1, dtype=torch.uint8, device="cuda")

    def destroy(self):
        pass


def _topk_gate_scores(x, gate_weight, num_topk, routed_scaling_factor, bias):
    logits = _dq(x) @ _dq(gate_weight).t()
    if bias is not None:
        logits = logits + _dq(bias)
    u = torch.sqrt(torch.nn.functional.softplus(logits))
    if u.shape[-1] <= 1:
        vals, idxs = u, torch.zeros_like(u, dtype=torch.int64)
    else:
        vals, idxs = torch.topk(u, min(num_topk, u.shape[-1]), dim=-1)
    if vals.shape[-1] < num_topk:
        pad = torch.zeros((vals.shape[0], num_topk - vals.shape[-1]), device=vals.device, dtype=vals.dtype)
        vals = torch.cat([vals, pad], dim=-1)
        idxs = torch.cat([idxs, torch.zeros_like(pad, dtype=torch.int64)], dim=-1)
    topk_sum = vals.sum(-1, keepdim=True) + 1e-20
    weights = vals / topk_sum * routed_scaling_factor
    return idxs.to(torch.int32), weights


def bf16_mega_gate(
    hidden_states, gate_weight, num_topk=1, use_shared_as_routed=False, num_shared_experts=0,
    routed_scaling_factor=1.0, ep_rank=0, mask=None, bias=None, image_bias=None,
    image_token_mask=None, force_random=None, to_physical_map=None, logical_count=None,
    scoring_func="sqrtsoftplus", out=None,
):
    idx, w = _topk_gate_scores(hidden_states, gate_weight, num_topk, routed_scaling_factor, bias)
    if out is not None:
        out[0].copy_(idx)
        out[1].copy_(w.to(out[1].dtype))
    return idx, w


def _sel_exp(w, e):
    """按 expert 索引权重：2D（E=1 无专家维）按单专家广播，3D [E, ...] 按 e 取。"""
    wf = w.to(torch.float32)
    if wf.ndim == 2:
        return wf.unsqueeze(0).expand(e.shape[0], *wf.shape)
    if wf.ndim == 3:
        return wf[e]
    raise ValueError(f"unexpected weight ndim {wf.ndim}")


def _mega_moe_compute(out, l1_w, l2_w, sym_buffer, activation="swiglu", activation_clamp=None):
    # 向量化实现（CUDA graph 捕获下禁止逐 token int()/item() 同步操作）。
    # l1_w 布局：[hidden, 2*inter]（gate/up 融合、out-first 转置）；l2_w：[inter, hidden]。
    num_tokens = out.shape[0]
    topk = sym_buffer.num_topk
    x = sym_buffer.x[:num_tokens].to(torch.float32)  # [M, hidden]
    idx = sym_buffer.topk_idx[:num_tokens]
    wts = sym_buffer.topk_weights[:num_tokens].to(torch.float32)
    hidden = x.shape[-1]
    n_exp = max(int(sym_buffer.num_experts), 1)
    acc = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=out.device)
    for k in range(topk):
        e = idx[:, k].clamp(min=0, max=n_exp - 1)  # [M]
        w1e = _sel_exp(l1_w, e)  # [M, hidden, 2*inter]
        if w1e.ndim != 3:
            w1e = w1e.unsqueeze(1).expand(w1e.shape[0], x.shape[1], w1e.shape[-1])
        g_all = torch.einsum("mh,mhk->mk", x, w1e)  # [M, 2*inter]
        inter = g_all.shape[-1] // 2
        gate, up = g_all[:, :inter], g_all[:, inter:]
        if activation == "swiglu":
            g = gate * torch.sigmoid(gate) * up  # silu(gate)*up
        else:
            g = g_all
        if activation_clamp is not None:
            g = torch.clamp(g, -activation_clamp, activation_clamp)
        out_k = torch.einsum("mk,mkh->mh", g, _sel_exp(l2_w, e))  # [M, hidden]
        acc = acc + wts[:, k:k + 1] * out_k
    out.copy_(acc.to(out.dtype))


def fp8_fp4_mega_moe(
    out, l1_w, l2_w, sym_buffer, cumulative_local_expert_recv_stats=None,
    activation="swiglu", activation_clamp=None, fast_math=True, **shared_kwargs,
):
    _mega_moe_compute(out, l1_w[0], l2_w[0], sym_buffer, activation, activation_clamp)


def bf16_mega_moe(
    out, l1_w, l2_w, sym_buffer, cumulative_local_expert_recv_stats=None,
    activation="swiglu", activation_clamp=None, fast_math=True, **shared_kwargs,
):
    _mega_moe_compute(out, l1_w, l2_w, sym_buffer, activation, activation_clamp)


def fp8_fp4_mqa_logits(q, kv, weights, ks, ke, max_seqlen_k):
    # q: (M, H, r) or (M, r); kv: (K, H, r) or (K, r)
    # Returns per-query, per-kv scores (M, K): weights (M, H) are the per-head
    # mixing coefficients (the prefill p_sum feeds deep_select.topk directly).
    q_dq = _dq(q[0])
    kv_dq = _dq(kv[0])
    if q_dq.ndim == 3 and kv_dq.ndim == 3:
        logits = torch.einsum("bhr,khr->bkh", q_dq, kv_dq)
    elif q_dq.ndim == 3 and kv_dq.ndim == 2:
        logits = torch.einsum("bhr,kr->bkh", q_dq, kv_dq)
    else:
        logits = (q_dq.reshape(q_dq.shape[0], -1) @ kv_dq.reshape(kv_dq.shape[0], -1).t()).unsqueeze(-1)
    if weights is not None and weights.ndim == 2 and logits.ndim == 3:
        logits = (logits * weights.to(logits.dtype).unsqueeze(1)).sum(-1)
    elif logits.ndim == 3:
        logits = logits.sum(-1)
    return logits.to(torch.bfloat16)


def fp8_fp4_sparse_mqa_logits(q, kv, weights, schedule_metadata, n_blocks, block_size, **kw):
    q_dq = _dq(q[0])
    kv_dq = _dq(kv[0])
    logits = torch.einsum("bhr,khr->bkh", q_dq, kv_dq)
    return logits.to(torch.bfloat16)


def _dequant_paged(blocks_flat: torch.Tensor) -> torch.Tensor:
    """[N, bpt] uint8 -> [N, dim] fp32：按 bytes_per_token 识别布局（scale 忽略，mock 数值不保）。"""
    bpt = blocks_flat.shape[-1]
    if bpt >= 512:  # e4m3 512B data（+ scale，忽略）
        return blocks_flat[:, :512].view(torch.float8_e4m3fn).float()
    # 小维 KV（index 等）：e2m1 半字节打包，64 字节 data -> 128 值
    nib = blocks_flat[:, :64]
    lo = nib & 0x0F
    hi = (nib >> 4) & 0x0F
    vals = torch.cat([lo, hi], dim=1).to(torch.int32)  # [N, 128]
    sign = vals & 0x8
    exp = ((vals >> 1) & 0x3).to(torch.float32)
    mant = (vals & 0x1).to(torch.float32)
    mag = (1.0 + mant) * (2.0 ** (exp - 1))
    mag = torch.where(exp == 0, torch.zeros_like(mag), mag)
    sign_f = torch.where(sign > 0, torch.ones_like(mag), -torch.ones_like(mag))  # 纯设备张量，graph 捕获安全
    return mag * sign_f


def fp8_fp4_paged_mqa_logits(q, kv, weights, klen, block_tables, schedule_metadata, max_seqlen_k, indices=None, **kw):
    # kv: paged uint8 cache [num_blocks, block_size, 1, bpt]；klen: (M,1)；block_tables: (M, blocks)
    # 先按 block_tables gather（只取被引用行）再解包，避免全缓存解包的显存放大。
    print("DBG paged: q=%s kv=%s klen=%s bt=%s" % (q[0].shape, tuple(kv.shape), None if klen is None else tuple(klen.shape), None if block_tables is None else tuple(block_tables.shape)), flush=True)
    q_raw = _dq(q[0])  # [M, 1, dim] 或 [M, dim]
    q_dq = q_raw.reshape(q_raw.shape[0], -1).to(torch.float32)  # [M, dim]
    M = q_dq.shape[0]
    bs = kv.shape[1]
    bpt = kv.shape[-1]
    bt = block_tables.to(torch.long).clamp(min=0, max=kv.shape[0] - 1)
    sel = kv[bt.reshape(M, -1)].reshape(M, -1, bpt)  # [M, blocks*bs, bpt]
    rows = sel.reshape(-1, bpt)
    dim = _dequant_paged(rows[:1]).shape[-1]  # 由单行解包确定输出维（不跨 block 同步）
    deq = _dequant_paged(rows).reshape(M, -1, dim)  # [M, blocks*bs, dim]
    scores = torch.einsum("mh,mkh->mk", q_dq, deq)
    if klen is not None:
        kl = torch.as_tensor(klen, device=scores.device, dtype=torch.long).reshape(M, 1)
        pos = torch.arange(deq.shape[1], device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(pos >= kl, float("-inf"))
    return scores.to(torch.bfloat16)


def fp8_fp4_paged_sparse_mqa_logits(q, kv, weights, schedule_metadata, n_blocks, block_size, **kw):
    return fp8_fp4_sparse_mqa_logits(q, kv, weights, schedule_metadata, n_blocks, block_size, **kw)


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, psum_num_recv_tokens_per_expert, use_psum_layout=True, expected_m_for_psum_layout=None, **kw):
    # Fallback: treat as one grouped expert (EP=1); d = a @ b[0]^T with topk-expanded rows.
    a_dq = _dq(a if torch.is_tensor(a) else a[0])
    b_dq = _dq(b if torch.is_tensor(b) else b[0])
    d.copy_((a_dq @ b_dq.t()).to(d.dtype))


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, psum_num_recv_tokens_per_expert, recipe_a=(1, 32), recipe_b=(1, 32), use_psum_layout=True, expected_m_for_psum_layout=None, **kw):
    m_grouped_bf16_gemm_nt_contiguous(a, b, d, psum_num_recv_tokens_per_expert, use_psum_layout, expected_m_for_psum_layout)


def mega_mhc(*a, **k):
    raise NotImplementedError("deep_gemm.mega_mhc not implemented for mock chain")


def set_num_sms(n):
    pass


def get_num_sms():
    return 128


def get_block_m_for_mega_moe(*a, **k):
    return 128


def get_tma_aligned_size(*a, **k):
    return a[0] if a else 0


def get_theoretical_mk_alignment_for_contiguous_layout():
    return 1


def set_mk_alignment_for_contiguous_layout(n):
    pass


def get_fp8_casting_granularity():
    return 32


def get_paged_mqa_logits_metadata(*a, **k):
    return None


def get_paged_sparse_mqa_logits_metadata(*a, **k):
    return None


def get_sparse_mqa_logits_metadata(*a, **k):
    return None


def get_mk_alignment_for_contiguous_layout() -> int:
    return 1


def set_pdl(enabled):
    pass


def use_deterministic_algorithms(enabled):
    pass
