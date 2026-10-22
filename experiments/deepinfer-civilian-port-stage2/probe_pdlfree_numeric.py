"""探针：DeepInfer tilelang kernel 去 PDL 变体在 SM89 (4090) 的数值对照（4 kernel）。

在上一轮"5 kernel 编译全过、norm 数值 PASS"基础上，给 kv_index_cat / load_kv_index /
dflash_input_store / fused_sample 补数值对照，把"去 PDL 改写是机械工程"的证据从
"能编译"推进到"语义保持"。方法：服务器 cuda 包目录已有 *_pdlfree.py（sed 文本替换
pdl_sync->sync_grid、pdl_trigger->删除，不改原文件），直接导入真实源码变体编译运行，
与 torch / numpy 参考实现对照。

注意：
- tilelang kernel 输入多为 StridedTensor，需要以 (row_stride, col_stride) 传入分配好的 buffer。
- 部分 kernel 有隐含形状约束（如 dflash_input_store 要求 n_embed % 1024 == 0 的分块）。
"""
import sys
import time

import torch

sys.path.insert(0, "/home/zxw/deepinfer-civilian")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"

results = []


def report(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# 1) kv_index_cat_tl：拷贝 kv_index + 对 kv_ca_index 非 -1 元素加 total_k+total_k_swa
# ---------------------------------------------------------------------------
def test_kv_index_cat():
    import importlib
    mod = importlib.import_module(f"{PKG}.kv_index_cat_pdlfree")
    torch.manual_seed(0)
    total_q, window_size, topk_tokens = 37, 64, 16
    out_shape = window_size + topk_tokens
    out_stride = (out_shape + 3) // 4 * 4  # 与真实 wrapper 一致：pad 到 4 的倍数
    kernel = mod.kv_index_cat_tl(dtype="int32", window_size=window_size,
                                 topk_tokens=topk_tokens, out_stride=out_stride)
    kv_index = torch.randint(0, 1000, (total_q, window_size), dtype=torch.int32, device="cuda")
    kv_ca_index = torch.randint(0, 1000, (total_q, topk_tokens), dtype=torch.int32, device="cuda")
    kv_ca_index = torch.where(torch.rand(total_q, topk_tokens, device="cuda") < 0.3,
                              torch.full_like(kv_ca_index, -1), kv_ca_index)
    total_k_swa, total_k = 100, 55
    out = torch.empty((total_q, out_stride), dtype=torch.int32, device="cuda")
    kv_index_out = out[:, :window_size]
    kv_ca_index_out = out[:, window_size:out_shape]
    kernel(kv_index, kv_ca_index, kv_index_out, kv_ca_index_out, total_k_swa, total_k)
    torch.cuda.synchronize()
    ref_out1 = kv_index.clone()
    ref_ca = kv_ca_index.clone()
    m = ref_ca != -1
    ref_ca[m] = ref_ca[m] + total_k + total_k_swa
    ok = torch.equal(kv_index_out, ref_out1) and torch.equal(kv_ca_index_out, ref_ca)
    report("kv_index_cat", ok,
           f"kv_index eq={torch.equal(kv_index_out, ref_out1)}, "
           f"kv_ca eq={torch.equal(kv_ca_index_out, ref_ca)}")


# ---------------------------------------------------------------------------
# 2) load_kv_index_tl：按每行 (offset, len) 展开为 kv_index 行，列>=len 为 -1
#    block_size==0 模式：连续 offset + j；越界列填 -1
# ---------------------------------------------------------------------------
def test_load_kv_index():
    import importlib
    mod = importlib.import_module(f"{PKG}.get_kv_index_pdlfree")
    kernel = mod.load_kv_index_tl(index_length=32, block_size=0, has_block_tables=False)
    torch.manual_seed(1)
    n = 41
    kv_index_offset_len = torch.stack([
        torch.randint(0, 500, (n,), dtype=torch.int32, device="cuda"),
        torch.randint(0, 33, (n,), dtype=torch.int32, device="cuda"),
    ], dim=1)
    kv_index = torch.full((n, 32), -999, dtype=torch.int32, device="cuda")
    flatten_block_tables = torch.empty(0, dtype=torch.int32, device="cuda")
    kernel(kv_index_offset_len, kv_index, flatten_block_tables)
    torch.cuda.synchronize()
    ref = torch.full((n, 32), -1, dtype=torch.int32, device="cuda")
    for r in range(n):
        off = kv_index_offset_len[r, 0].item()
        ln = kv_index_offset_len[r, 1].item()
        ref[r, :ln] = torch.arange(off, off + ln, dtype=torch.int32, device="cuda")
    ok = torch.equal(kv_index, ref)
    report("load_kv_index", ok, f"full eq={ok} (n={n}, index_length=32, block_size=0)")


# ---------------------------------------------------------------------------
# 3) dflash_input_store_tl：x (n, hc_mult, n_embed) -> out (n, n_embed)
#    hc_mult==1 拷贝；否则按 hc_mult 均值（float32 累加）
# ---------------------------------------------------------------------------
def test_dflash_input_store():
    import importlib
    mod = importlib.import_module(f"{PKG}.dflash_pdlfree")
    torch.manual_seed(2)
    for hc_mult, n_embed in [(1, 1024), (2, 1024)]:
        n = 9
        kernel = mod.dflash_input_store_tl(n_embed=n_embed, hc_mult=hc_mult)
        x = torch.randn(n, hc_mult, n_embed, dtype=torch.bfloat16, device="cuda")
        out = torch.full((n, n_embed), float("nan"), dtype=torch.bfloat16, device="cuda")
        kernel(x, out)
        torch.cuda.synchronize()
        ref = x.float().mean(dim=1).to(torch.bfloat16)
        ok = torch.equal(out, ref)
        max_err = (out.float() - ref.float()).abs().max().item()
        report(f"dflash_input_store hc_mult={hc_mult}", ok,
               f"n_embed={n_embed}, max abs err={max_err:.5f}")


# ---------------------------------------------------------------------------
# 4) fused_sample_tl：logprobs = (logit + lse1) * inv_temp + lse0 = log-softmax(logit/temp)
#    无 top_p/top_k 过滤时（top_p=1.0, top_k=-1）对照 torch log_softmax；candidate 只查合法性
# ---------------------------------------------------------------------------
def test_fused_sample():
    import importlib
    mod = importlib.import_module(f"{PKG}.fused_sample_pdlfree")
    torch.manual_seed(3)
    vocab_size = 128
    kernel = mod.fused_sample_tl(vocab_size=vocab_size, do_sample=True, do_reject_sampling=False)
    total_q = 8
    logits = torch.randn(total_q, vocab_size, dtype=torch.float32, device="cuda")
    temperatures = torch.full((total_q,), 0.9, dtype=torch.float32, device="cuda")
    top_p = torch.full((total_q,), 1.0, dtype=torch.float32, device="cuda")
    top_k = torch.full((total_q,), -1, dtype=torch.int32, device="cuda")
    sample_positions = torch.zeros(total_q, dtype=torch.int32, device="cuda")
    seq_seeds = (torch.arange(total_q, dtype=torch.int64, device="cuda") * 7 + 13).to(torch.uint64)
    is_verify = torch.zeros(total_q, dtype=torch.bool, device="cuda")
    logprobs = torch.empty(total_q, vocab_size, dtype=torch.float32, device="cuda")
    candidate_token_ids = torch.full((total_q,), -1, dtype=torch.int32, device="cuda")
    pool_rows = 1
    last_nextn_token_logprobs = torch.empty(pool_rows, vocab_size, dtype=torch.float32, device="cuda")
    mtp_info_read_rows = torch.zeros(total_q, dtype=torch.int32, device="cuda")
    last_next_n_token_ids = torch.full((pool_rows,), -1, dtype=torch.int32, device="cuda")
    accept_result_verify = torch.zeros(total_q, dtype=torch.bool, device="cuda")
    draft_selected_logprobs = torch.empty(total_q, dtype=torch.float32, device="cuda")
    accept_probs = torch.empty(total_q, dtype=torch.float32, device="cuda")
    seed_offset = 0
    lse_workspace = torch.empty(total_q, 2, dtype=torch.float32, device="cuda")
    args = (logits, temperatures, top_p, top_k, sample_positions, seq_seeds, is_verify,
            logprobs, candidate_token_ids, last_nextn_token_logprobs, mtp_info_read_rows,
            last_next_n_token_ids, accept_result_verify, draft_selected_logprobs,
            accept_probs, seed_offset, lse_workspace)
    t0 = time.time()
    kernel(*args)
    torch.cuda.synchronize()
    dt = time.time() - t0

    ref_lp = torch.log_softmax(logits / temperatures[:, None], dim=-1)
    lp_err = (logprobs - ref_lp).abs().max().item()
    ok_lp = lp_err < 1e-4
    cand_ok = bool((candidate_token_ids >= 0).all() and (candidate_token_ids < vocab_size).all())
    # lse_workspace: lse1 = -max(logit)（原始 logit，未缩放）；lse0 = -log(sum_j exp((logit-max)/t))
    # 由此 logprobs = (logit + lse1)/t + lse0 = log_softmax(logit/t)（已在上面核对）
    ref_lse1 = -logits.max(dim=-1).values
    ref_lse0 = -torch.logsumexp((logits - logits.max(dim=-1, keepdim=True).values) / temperatures[:, None], dim=-1)
    lse0_err = (lse_workspace[:, 0] - ref_lse0).abs().max().item()
    lse1_err = (lse_workspace[:, 1] - ref_lse1).abs().max().item()
    ok = ok_lp and cand_ok and lse0_err < 1e-4 and lse1_err < 1e-4
    report("fused_sample", ok,
           f"logprobs max err={lp_err:.6f} (ref=log_softmax(x/0.9)), cand in [0,V)={cand_ok}, "
           f"lse0 err={lse0_err:.6f}, lse1 err={lse1_err:.6f}, run {dt:.1f}s")


def main():
    tests = [test_kv_index_cat, test_load_kv_index, test_dflash_input_store, test_fused_sample]
    for t in tests:
        try:
            t()
        except Exception as e:
            report(t.__name__, False, f"EXC {type(e).__name__}: {str(e)[:300]}")
    fails = [r for r in results if not r[1]]
    print("=" * 60)
    print(f"TOTAL: {len(results)-len(fails)}/{len(results)} PASS; "
          f"FAIL: {[r[0] for r in fails]}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
