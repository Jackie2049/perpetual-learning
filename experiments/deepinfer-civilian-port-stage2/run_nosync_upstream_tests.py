#!/usr/bin/env python3
"""DeepInfer 去 PDL nosync 改写 —— 全量上游测试套件验证（4090-1 SM89，M5 六续）。

方法：把 ``ops`` 的 43 个 ``*_tl`` 工厂全部替换为 ``*_nosync.py`` 变体（等价于真实
移植时"用 nosync 文件替换包内 kernel 文件"），再通过 pytest 运行 tilelang_ops 各
wrapper 文件内嵌的官方 GPU 测试（排除 benchmark/bench 项）。每个测试内部与 wrapper
自带的 torch 参考实现逐项对照，证据强度等同"替换后的包跑官方测试套件"。

前置（与 run_smoke_stage2.sh 一致）：
- 服务器 4090-1，conda env deepinfer-smoke（torch 2.13.0+cu129 + tilelang 0.1.15）
- PYTHONPATH 含 /home/zxw/deepinfer-stubs；LD_LIBRARY_PATH 含 conda lib / torch / nvjitlink
- cuda/ 目录存在 28 个 ``*_nosync.py`` 变体（五续生成，与本地逐字节一致）
"""
import importlib.util
import os
import sys

REPO = "/home/zxw/deepinfer-civilian"
CUDA_DIR = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops/cuda")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"

# (cuda 模块名, 工厂名) —— 与 scan_nosync_all.py 的 43 个工厂一致
CASES = [
    ("candidate_blocks", "candidate_block_max_tl"),
    ("candidate_blocks", "get_candidate_compact_lens_tl"),
    ("candidate_blocks", "restore_candidate_indices_inplace_tl"),
    ("compress", "compress_and_update_state_prefill_tl"),
    ("compress", "compress_and_update_state_decode_tl"),
    ("compute_seqlens_k_zos", "compute_seqlens_k_zos_tl"),
    ("dflash", "confidence_head_fused_tl"),
    ("dflash", "dflash_input_store_tl"),
    ("dflash_copy", "copy_dflash_decode_input_tl"),
    ("flash_mla_utils", "convert_kv_index_tl"),
    ("fused_sample", "fused_sample_tl"),
    ("fused_transform", "fused_norm_rope_hadamard_cast_store_kvcache_tl"),
    ("gather", "gather_tl"),
    ("gather_engram_tokens_zos", "gather_engram_tokens_zos_tl"),
    ("gather_input_ids_zos", "gather_input_ids_zos_tl"),
    ("get_kv_index", "load_kv_index_tl"),
    ("get_kv_index", "get_fused_dflash_get_input_tl"),
    ("image_embed_ops", "replace_input_embeds_tl"),
    ("image_embed_ops", "rgb_pad_to_patches_tl"),
    ("image_embed_ops", "batch_downsample_tl"),
    ("kv_index_cat", "kv_index_cat_tl"),
    ("load_paged_kv_cache", "load_paged_kv_cache_tl"),
    ("norm", "norm_tl"),
    ("outlier_detection", "detect_outlier_and_abort_tl"),
    ("prepare_cp_mirror", "prepare_cp_mirror_tl"),
    ("prepare_cp_mirror", "yoco_cp_boundary_fused_tl"),
    ("prepare_decode_metadata", "prepare_v4_decode_metadata_tl"),
    ("prepare_decode_metadata", "copy_block_tables_tl"),
    ("random_sampling", "random_sampling_tl"),
    ("sampler", "prepare_verification_metadata_tl"),
    ("sampler", "expand_sampling_params_tl"),
    ("sampler", "map_lookup_tl"),
    ("sampler", "remap_draft_topk_tl"),
    ("sampler", "get_accept_result_tl"),
    ("sampler", "finalize_verification_tl"),
    ("schedule_mtp_varlen", "compute_seqlens_q_by_confidence_rates_tl"),
    ("state_cache", "select_state_slice_tl"),
    ("state_cache", "load_state_to_cache_tl"),
    ("store_kv_cache", "store_kv_cache_tl"),
    ("store_mtp_info", "store_mtp_info_tl"),
    ("structure_output", "apply_token_bitmask_by_step_inplace_tl"),
    ("sum_smaller_probs", "sum_smaller_probs_tl"),
    ("swa_index", "expand_swa_index_tl"),
]


def _hide_variants():
    """ops 遍历 backend 包时与同名 *_tl 工厂冲突，先临时隐藏变体文件。"""
    hidden = []
    for f in sorted(os.listdir(CUDA_DIR)):
        if f.endswith("_pdlfree.py") or f.endswith("_nosync.py"):
            src = os.path.join(CUDA_DIR, f)
            dst = src + ".hidden"
            os.rename(src, dst)
            hidden.append((src, dst))
    return hidden


def _restore(hidden):
    for src, dst in hidden:
        os.rename(dst, src)


def _load_variant(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import torch

    torch.cuda.set_device(0)
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    hidden = _hide_variants()
    try:
        from deepinfer.model_executor.layers.tilelang_ops import ops  # noqa: E402
    finally:
        _restore(hidden)

    missing = [fn for _, fn in CASES if not hasattr(ops, fn)]
    if missing:
        print("MISSING factories in ops:", missing, flush=True)
        return 2

    for mod_name, fn_name in CASES:
        var = _load_variant(
            f"{PKG}.{mod_name}_nosync",
            os.path.join(CUDA_DIR, f"{mod_name}_nosync.py"),
        )
        setattr(ops, fn_name, getattr(var, fn_name))
    print(f"patched {len(CASES)} ops.<x>_tl -> nosync variants", flush=True)

    # wrapper 根目录中有测试函数的文件（排除已各 wrapper 的 benchmark）
    tl_root = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops")
    files = [
        "candidate_blocks.py", "compress.py", "compute_seqlens_k_zos.py",
        "dflash.py", "dflash_copy.py", "flash_mla_utils.py", "fused_sample.py",
        "fused_transform.py", "gather.py", "gather_engram_tokens_zos.py",
        "gather_input_ids_zos.py", "get_kv_index.py", "image_embed_ops.py",
        "kv_index_cat.py", "load_paged_kv_cache.py", "norm.py",
        "prepare_cp_mirror.py", "prepare_v4_decode_metadata.py",
        "random_sampling.py", "sampler.py", "schedule_mtp_varlen.py",
        "state_cache.py", "store_kv_cache.py", "store_mtp_info.py",
    ]
    args = [os.path.join(tl_root, f) for f in files]
    import pytest

    return pytest.main(
        ["-q", "--no-header", "-p", "no:cacheprovider", "-k",
         "not benchmark and not bench", *args],
    )


if __name__ == "__main__":
    sys.exit(main())
