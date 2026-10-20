# vLLM：SM89 显卡上的 FP8 KV 配置兼容性

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** SM89 显卡上的 FP8 KV 配置兼容性。[PR #45038](https://github.com/vllm-project/vllm/pull/45038) · [Issue #44879](https://github.com/vllm-project/vllm/issues/44879)

**【类型】** 兼容性分析

**【价值评估】** 3 / 10 — 有支持矩阵参考价值，但旧版本材料与已关闭 PR 限制了可直接采用性。

**【进展】** 默认不发布：2026-10-19 核验 [PR #45038](https://github.com/vllm-project/vllm/pull/45038) 未合并关闭；底层 [Issue #44879](https://github.com/vllm-project/vllm/issues/44879) 当时仍开放，本文信息基于较旧版本。

**【大白话讲明白】**

模型支持某种量化，不代表所有显卡上的 KV 缓存算子都支持同样格式。错误地自动覆盖缓存类型可能让计算崩溃。这里讨论怎样把显卡能力与配置选择对齐。

**【What–Why–How】**

- **What：** compressed-tensors 对 FP8 KV cache 配置的覆盖及 SM89 支持范围。
- **Why：** 模型量化信息、GPU 架构和注意力内核支持并非同一个条件。
- **How：** 先重新建立当前版本支持矩阵，再核对报错路径；历史草稿不能作为已通过的兼容性补丁。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> Status: ~~DRAFT — User should review before posting~~ → **2026-10-19 核验：PR #45038 未合并自关（9-21），底层 issue #44879 仍 open 但有实测噪音**。草稿信息基于 v0.23（6 月），已过时；仅剩支持矩阵整理有微弱参考价值。发布需用户授权，默认不发布。
> PR: https://github.com/vllm-project/vllm/pulls/45038
> Title: [Bugfix] Guard compressed-tensors FP8 KV cache override on SM90+

## Context

- Issue #44879: compressed-tensors FP8 KV causes CUDA crash on SM89 (L4/RTX4090)
- PR #45038: adds `has_device_capability(90)` guard, only override KV dtype to FP8 on SM90+
- Already validated on SM89 (L4 GPU) by another contributor
- ★★★ vLLM v0.23.0 released 2026-06-15 — PR still OPEN, not yet merged

## Proposed Comment

```
This fix directly addresses a critical SM89 correctness issue that affects both L4 and RTX 4090 GPUs. The root cause is well-identified: FlashInfer's FP8 attention kernels (flash_attn_varlen_func_fp8_sm90) only exist for SM90+, so overriding kv_cache_dtype to fp8 on SM89 inevitably causes CUDA illegal-memory-access.

I've been studying RTX 4090 (SM89) limitations in depth for AI training and inference workloads. A few observations that may be helpful:

**SM89 FP8 support matrix** (updated with v0.23.0 findings):
- FlashInfer FP8 KV: ✗ — flash_attn_varlen_func_fp8_sm90 kernels only compiled for SM90+, crash on SM89
- Triton FP8 KV: ✓ (v0.23.0 #43914) — SM89 is the fp8 boundary for Triton backend, technically allowed but slower than FlashInfer
- compressed-tensors FP8 override: ✗ — unconditionally overrides kv_cache_dtype to fp8 → uses FlashInfer backend → crash on SM89 (this PR fixes it)
- NVFP4 KV: ✗ — fail-fast ValueError on SM89, requires SM100/SM103 (v0.23.0 #43669)
- FP8 E4M3 (training): ✗ — no native GEMM pipeline on SM89, no performance advantage
- FP8 AllGather/communication: ✗ — NCCL FP8 requires SM90+
- INT4 GPTQ + INT8 KV: ✓ — the practical production path for SM89 inference (vLLM v0.23.0+)
- BF16 training: ✓ — the only correct training precision for SM89

**Impact for GRPO training on RTX 4090**:
The FP8 KV cache override is particularly problematic for GRPO/RL training scenarios where:
- vLLM is used as the rollout engine (verl/rLLM)
- INT8 KV cache is the correct choice for SM89 (not FP8)
- compressed-tensors models with FP8 config would silently crash on RTX 4090

Note: v0.23.0 also revealed a related SM89 issue — SM<90 batch invariance breaks with CUDA graphs + torch.compile (#39096). For RTX 4090 users, this means speculative decoding (EAGLE/MTP) requires `enforce_eager=True`. Combined with this FP8 guard, the SM89 production path becomes: INT4 GPTQ + INT8 KV + MRv1 + enforce_eager (for spec decode) or INT8 KV + MRv2 (for dense BF16).

Note: vLLM v0.23.0 (released 2026-06-15) includes MRv2 as default for Llama/Mistral but MRv2 doesn't support quantized models yet — RTX 4090 INT4 inference still uses V1 ModelRunner. This FP8 guard is essential for V1 + INT8 KV to work correctly on SM89.

The `has_device_capability(90)` guard is the right fix. Would also suggest:
1. Adding SM89 to the CI test matrix (even a single L4 instance) to catch future SM89 regressions
2. Updating the vLLM documentation to explicitly list SM89 limitations for FP8 KV
3. Considering an SM89 compatibility doc page that consolidates all known SM89 issues (FP8 KV, FP8 training, NVLS/TMA, etc.)

Happy to help test on RTX 4090 when our GPU server is available.
```

## Key Points

1. ★★★ SM89 FP8 support matrix → shows comprehensive understanding
2. ★★★ GRPO training impact → connects to our core research area
3. ★★★ Concrete suggestions (CI matrix + docs) → actionable improvement
4. ★★★ Offer to test on RTX 4090 → practical contribution commitment
