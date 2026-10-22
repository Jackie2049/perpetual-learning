# 全量去 PDL 编译扫查日志（scan_pdlfree_all）摘要

完整日志：服务器 4090-1 生成（/tmp/scan_pdlfree_all.log，2026-10-22 07:49–07:51，本地不入 git 的 .log 约定）；此处存结论行。

```
[OK]   candidate_blocks.candidate_block_max_tl  12.1s
[OK]   candidate_blocks.get_candidate_compact_lens_tl  2.7s
[OK]   candidate_blocks.restore_candidate_indices_inplace_tl  2.8s
[OK]   compress.compress_and_update_state_prefill_tl  3.5s
[OK]   compress.compress_and_update_state_decode_tl  3.1s
[OK]   compute_seqlens_k_zos.compute_seqlens_k_zos_tl  3.1s
[OK]   dflash.confidence_head_fused_tl  2.9s
[OK]   dflash.dflash_input_store_tl  0.0s
[OK]   dflash_copy.copy_dflash_decode_input_tl  3.3s
[OK]   flash_mla_utils.convert_kv_index_tl  2.8s
[OK]   fused_sample.fused_sample_tl  0.9s
[OK]   fused_transform.fused_norm_rope_hadamard_cast_store_kvcache_tl  3.8s
[OK]   gather.gather_tl  3.2s
[OK]   gather_engram_tokens_zos.gather_engram_tokens_zos_tl  4.0s
[OK]   gather_input_ids_zos.gather_input_ids_zos_tl  2.9s
[OK]   get_kv_index.load_kv_index_tl  0.1s
[OK]   get_kv_index.get_fused_dflash_get_input_tl  3.6s
[OK]   image_embed_ops.replace_input_embeds_tl  2.7s
[OK]   image_embed_ops.rgb_pad_to_patches_tl  2.6s
[OK]   image_embed_ops.batch_downsample_tl  2.5s
[OK]   kv_index_cat.kv_index_cat_tl  0.1s
[OK]   load_paged_kv_cache.load_paged_kv_cache_tl  3.3s
[OK]   norm.norm_tl  0.2s
[OK]   outlier_detection.detect_outlier_and_abort_tl  2.9s
[OK]   prepare_cp_mirror.prepare_cp_mirror_tl  5.0s
[OK]   prepare_cp_mirror.yoco_cp_boundary_fused_tl  5.4s
[OK]   prepare_decode_metadata.prepare_v4_decode_metadata_tl  3.2s
[OK]   prepare_decode_metadata.copy_block_tables_tl  2.6s
[OK]   random_sampling.random_sampling_tl  5.1s
[OK]   sampler.prepare_verification_metadata_tl  3.3s
[OK]   sampler.expand_sampling_params_tl  2.9s
[OK]   sampler.map_lookup_tl  2.3s
[OK]   sampler.remap_draft_topk_tl  2.6s
[OK]   sampler.get_accept_result_tl  3.2s
[OK]   sampler.finalize_verification_tl  3.3s
[OK]   schedule_mtp_varlen.compute_seqlens_q_by_confidence_rates_tl  3.1s
[OK]   state_cache.select_state_slice_tl  2.2s
[OK]   state_cache.load_state_to_cache_tl  2.2s
[OK]   store_kv_cache.store_kv_cache_tl  3.8s
[OK]   store_mtp_info.store_mtp_info_tl  2.5s
[OK]   structure_output.apply_token_bitmask_by_step_inplace_tl  2.8s
[OK]   sum_smaller_probs.sum_smaller_probs_tl  2.4s
[OK]   swa_index.expand_swa_index_tl  2.6s
===== SUMMARY: 43/43 OK =====
```
