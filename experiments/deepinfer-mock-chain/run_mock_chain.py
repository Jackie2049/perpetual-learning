#!/usr/bin/env python3
"""DeepInfer mock 权重链路 (4090-1, SM89) —— LLMEngine 初始化 + prefill/decode 全链路。

目的：用户 2026-10-23 授权的 demo 目标——先 mock 权重跑通链路、再上真实权重调精度。
本脚本在服务器 4090-1 上验证：L1 调度 + L2 缓存分配 + L3 前向 + 采样器全链路可跑通，
输出为随机 token（无意义但链路通）。除「真实权重下载」环节外由 M5 定时任务推进。

方法（复用 stage2 六续的 nosync 改写 + stub 方案）：
1. 把 ops 的 43 个 *_tl 工厂全部替换为 *_nosync.py 变体（SM89 可编译，改写规则已全量验证）；
2. deepinfer-stubs 提供 deep_gemm torch 回退（GEMM 数值不保，仅保链路可跑）+ 其余 import 占位；
3. load_format="dummy" 触发 skip_load=True（HFCoverage 不读权重 + build_model_on_meta 随机初始化）；
4. mock 目录 /home/zxw/deepinfer-mock 提供伪造 V4.1 极小 config（完整字段结构 + 极小尺寸）。

前置（与 stage2 六续一致）：
- 服务器 4090-1，conda env deepinfer-smoke（torch 2.13.0+cu129 + tilelang 0.1.15）
- PYTHONPATH 含 /home/zxw/deepinfer-stubs；LD_LIBRARY_PATH 含 conda lib / torch / nvjitlink / cublas / nccl / cudnn
- cuda/ 目录存在 28 个 *_nosync.py 变体（五续生成，与本地逐字节一致）
"""
import importlib.util
import os
import sys

REPO = "/home/zxw/deepinfer-civilian"
CUDA_DIR = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops/cuda")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"
MOCK_DIR = "/home/zxw/deepinfer-mock"
# nosync 变体放在 repo 外的独立目录（共享 repo 会被并发会话清理，独立目录最稳）
VARIANTS_DIR = "/home/zxw/deepinfer-nosync-variants"

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
            os.rename(src, src + ".hidden")
            hidden.append((src, src + ".hidden"))
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
    print("== [1/4] patch 43 ops.<x>_tl -> nosync variants ==", flush=True)
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
            os.path.join(VARIANTS_DIR, f"{mod_name}_nosync.py"),
        )
        setattr(ops, fn_name, getattr(var, fn_name))
    print(f"patched {len(CASES)} ops.<x>_tl -> nosync variants", flush=True)

    import torch

    gpu_id = int(os.environ.get("MOCK_GPU", "7"))
    torch.cuda.set_device(gpu_id)
    print(
        f"== [2/4] device={torch.cuda.get_device_name(gpu_id)} "
        f"capability={torch.cuda.get_device_capability(gpu_id)}",
        flush=True,
    )

    print(f"== [3/4] construct LLMEngine (mock config {MOCK_DIR}, load_format=dummy) ==", flush=True)
    # DEEPINFER_USE_NEXTN_DECODE=0 跳过 MTP 模块构造（mock config num_nextn_predict_layers=0，
    # 且 lookahead_tokens=0 不需要 MTP 层；默认值 5 会让 DFlash 因层数检查抛 IndexError）。
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from deepinfer.entrypoints.llm import LLM

    llm = LLM(
        model=MOCK_DIR,
        trust_remote_code=True,
        expert_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.15,
        max_num_batched_tokens=1024,
        max_model_len=512,
        max_num_seqs=2,
        load_format="dummy",
        lookahead_tokens=0,
    )
    print("== LLMEngine constructed OK ==", flush=True)

    print("== [4/4] prefill + decode (generate) ==", flush=True)
    from deepinfer.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=8, temperature=0.8, top_p=0.9)
    outs = llm.generate(
        ["The quick brown fox jumps over the lazy dog."],
        sampling_params=sp,
        use_tqdm=False,
    )
    print("== GENERATED OUTPUT ==", flush=True)
    for o in outs:
        print(repr(o), flush=True)

    llm.llm_engine.close()
    print("== MOCK CHAIN COMPLETED OK ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
