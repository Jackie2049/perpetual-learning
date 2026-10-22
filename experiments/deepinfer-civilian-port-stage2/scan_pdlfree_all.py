"""全量扫查：28 个 PDL 源文件的 43 个 _tl 工厂去 PDL 变体在 SM89 编译。
对每个变体模块导入后调用工厂（最小编译参数）触发 tilelang JIT 编译。
"""
import importlib, sys, time, traceback

sys.path.insert(0, "/home/zxw/deepinfer-civilian")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"

def mod(m):
    return importlib.import_module(f"{PKG}.{m}_pdlfree")

# (模块名, 工厂名, kwargs)
CASES = [
    ("candidate_blocks", "candidate_block_max_tl", dict(max_model_len=8192, block_size=64)),
    ("candidate_blocks", "get_candidate_compact_lens_tl", dict(topk_blocks=8, block_size=64)),
    ("candidate_blocks", "restore_candidate_indices_inplace_tl", dict(topk_tokens=16, topk_blocks=8, block_size=64, has_offset=False)),
    ("compress", "compress_and_update_state_prefill_tl", dict(DIM=128, COMPRESS_RATIO=1, HAS_INDEX=False)),
    ("compress", "compress_and_update_state_decode_tl", dict(DIM=128, COMPRESS_RATIO=1)),
    ("compute_seqlens_k_zos", "compute_seqlens_k_zos_tl", dict(nextn=1)),
    ("dflash", "confidence_head_fused_tl", dict(n_embed=1024, markov_rank=1, has_markov_emb=False)),
    ("dflash", "dflash_input_store_tl", dict(n_embed=1024, hc_mult=1)),
    ("dflash_copy", "copy_dflash_decode_input_tl", dict(has_driver_inputs=False)),
    ("flash_mla_utils", "convert_kv_index_tl", dict(s=16, k=8, n=32, block_size=64)),
    ("fused_sample", "fused_sample_tl", dict(vocab_size=128, do_sample=True, do_reject_sampling=False)),
    ("fused_transform", "fused_norm_rope_hadamard_cast_store_kvcache_tl",
        dict(num_heads=1, head_dim=128, do_norm=False, norm_eps=0.0, do_rope=False, rotary_dim=0,
             rope_offset=0, positions_dtype="int32", is_neox=False, conjugate=False, do_hadamard=False,
             hadamard_scale=1.0, cast_fmt="", num_per_channels=0, round_sf=False, sf_format="",
             return_bf16=True, bf16_stage="transform", return_quant=False, cache_layout=None,
             block_size=1, small_batch=False)),
    ("gather", "gather_tl", dict(D=64, dtype="float16", index_dtype="int32", x_stride=64)),
    ("gather_engram_tokens_zos", "gather_engram_tokens_zos_tl",
        dict(max_seqlen_q=64, max_ngram_size=4, engram_compressed_pad_token=0, image_sentinel=0)),
    ("gather_input_ids_zos", "gather_input_ids_zos_tl", dict(max_seqlen_q=64)),
    ("get_kv_index", "load_kv_index_tl", dict(index_length=32, block_size=0, has_block_tables=False)),
    ("get_kv_index", "get_fused_dflash_get_input_tl",
        dict(swa_block_size=64, padded_kv_len=64, has_valid_token_mask=False, has_gate_force_random=False)),
    ("image_embed_ops", "replace_input_embeds_tl", dict(hidden=64)),
    ("image_embed_ops", "rgb_pad_to_patches_tl", dict(patch_size=8)),
    ("image_embed_ops", "batch_downsample_tl", dict(downsample_ratio=2, dim=64, do_downsample=False, dtype="float16")),
    ("kv_index_cat", "kv_index_cat_tl", dict(dtype="int32", window_size=64, topk_tokens=16, out_stride=64)),
    ("load_paged_kv_cache", "load_paged_kv_cache_tl", dict(layout=None, block_size=64)),
    ("norm", "norm_tl", dict(d=64, dtype="float16", return_dtype="float16", reduce_dtype="float32",
        has_residual=False, eps=1e-5, out_scale=1.0, do_fp8_cast=False, num_per_channels=1,
        use_tma_aligned_col_major_sf=False, round_sf=False, use_packed_ue8m0=False,
        return_origin_out=True, xdim=3, shared_sf_block_m=0, x_stride_s=64, y_stride_s=64,
        residual_stride_s=64, residual_out_stride_s=64, y_origin_stride_s=64, sf_stride_s=0)),
    ("outlier_detection", "detect_outlier_and_abort_tl", dict(dtype="float16", max_abs=1e4)),
    ("prepare_cp_mirror", "prepare_cp_mirror_tl", dict(cp_size=2, compress_ratios=(1,))),
    ("prepare_cp_mirror", "yoco_cp_boundary_fused_tl", dict(cp_size=2)),
    ("prepare_decode_metadata", "prepare_v4_decode_metadata_tl",
        dict(max_seqlen_q=64, max_num_blocks=8, window_size=32, swa_block_size=64, block_size=64,
             force_load_balance=False, ca_ratios=(1,), dense_ratios=(1,), sparse_ratios=(1,))),
    ("prepare_decode_metadata", "copy_block_tables_tl", dict(max_seqlen_q=64, max_num_blocks=8)),
    ("random_sampling", "random_sampling_tl", dict(vocab_size=128, is_nextn_predict=False)),
    ("sampler", "prepare_verification_metadata_tl", dict(num_steps=1)),
    ("sampler", "expand_sampling_params_tl", dict()),
    ("sampler", "map_lookup_tl", dict(force_topk=16)),
    ("sampler", "remap_draft_topk_tl", dict(force_topk=16, vocab_size=128)),
    ("sampler", "get_accept_result_tl", dict(has_golden=False)),
    ("sampler", "finalize_verification_tl", dict(has_sum_smaller_probs=False, has_confidence=False, num_steps=1, num_stop_tokens=1)),
    ("schedule_mtp_varlen", "compute_seqlens_q_by_confidence_rates_tl", dict(nextn=1)),
    ("state_cache", "select_state_slice_tl", dict(block_size_kvstore=64, block_size_inference=64, num_layers=2, feature_dim=1024)),
    ("state_cache", "load_state_to_cache_tl", dict(block_size_kvstore=64, block_size_inference=64, num_layers=2, feature_dim=1024)),
    ("store_kv_cache", "store_kv_cache_tl", dict(layout=None, block_size=64)),
    ("store_mtp_info", "store_mtp_info_tl", dict(nextn=1, vocab_size=128, has_index=False, has_confidence=False)),
    ("structure_output", "apply_token_bitmask_by_step_inplace_tl", dict(vocab_size=128, has_ftp=False)),
    ("sum_smaller_probs", "sum_smaller_probs_tl", dict(vocab_size=128)),
    ("swa_index", "expand_swa_index_tl", dict(width=64)),
]

def run():
    results = []
    for mod_name, fn_name, kwargs in CASES:
        t0 = time.time()
        try:
            if fn_name == "load_paged_kv_cache_tl" or fn_name == "store_kv_cache_tl":
                from deepinfer.model_executor.layers.tilelang_ops.cache_common import V41_FORMAT
                kwargs["layout"] = V41_FORMAT
            m = mod(mod_name)
            fn = getattr(m, fn_name)
            fn(**kwargs)
            dt = time.time() - t0
            print(f"[OK]   {mod_name}.{fn_name}  {dt:.1f}s")
            results.append((mod_name, fn_name, "OK", dt))
        except Exception as e:
            dt = time.time() - t0
            msg = str(e).split("\n")[0][:300]
            print(f"[FAIL] {mod_name}.{fn_name}  {dt:.1f}s  {type(e).__name__}: {msg}")
            results.append((mod_name, fn_name, "FAIL", dt, msg))
    n_ok = sum(1 for r in results if r[2] == "OK")
    print(f"\n===== SUMMARY: {n_ok}/{len(results)} OK =====")
    for r in results:
        if r[2] != "OK":
            print(f"  FAIL {r[0]}.{r[1]}: {r[4]}")

if __name__ == "__main__":
    run()
