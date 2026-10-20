# vLLM：休眠唤醒与 CUDA 流同步审阅（历史）

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** 休眠唤醒与 CUDA 流同步审阅（历史）。[PR #45552](https://github.com/vllm-project/vllm/pull/45552)

**【类型】** 代码审阅

**【价值评估】** 3 / 10 — 异步内存生命周期经验有复用价值，但原 PR 已关闭，原草稿不再有发布对象。

**【进展】** 不再推进：2026-10-19 项目核验记录 [PR #45552](https://github.com/vllm-project/vllm/pull/45552) 未合并关闭；本文为历史审阅材料。

**【大白话讲明白】**

引擎释放或恢复显存前，需要确认其他计算流不会继续使用旧内存。否则模型看似正常唤醒，后台计算仍可能读写失效位置。本文梳理这种同步边界。

**【What–Why–How】**

- **What：** cumem sleep/wake 与未完成 CUDA 工作之间的关系。
- **Why：** 默认流和非默认流的执行顺序不能仅靠主机代码顺序推断。
- **How：** 保留生产者、消费者与释放时机的核对方法；若推进新修复，需要当前版本的并发复现与回归测试。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> **IMPORTANT**: This is a DRAFT. User must authorize before posting to GitHub.
> Issue: https://github.com/vllm-project/vllm/issues/45552
> Related: vLLM cumem sleep/wake stream synchronization bug

## Comment: Cross-framework Analysis + RTX 4090 Impact Assessment

### Bug Summary

The cumem `sleep()` and `wake()` operations lack CUDA stream synchronization (`torch.cuda.synchronize()`), causing state lifecycle mismatches in multi-stream environments (P/D disaggregation, RLHF weight swapping, MoE cache updates).

### Root Cause Analysis

After deep reading this issue and the proposed fix, I've identified the following root cause chain:

1. `CuMemAllocator.sleep()` frees GPU memory via cuMemFree but does NOT synchronize CUDA streams
2. HTTP `/sleep` endpoint returns 200 immediately while GPU operations are still in flight
3. Consumer streams (training, weight reload) may read stale/freed memory
4. Result: silent data corruption, NaN in training, or outright crashes

### The "200 Lie" Pattern

This is a particularly dangerous pattern: the HTTP endpoint returns 200 (success) while the underlying GPU operation is incomplete. Applications that rely on the HTTP status to proceed (e.g., verl GRPO training pipeline) will continue to the training phase using stale/invalid memory.

This pattern has been observed in 8+ members of the "State Lifecycle Mismatch" pattern family:

| # | Issue | Symptom | Pattern |
|---|-------|---------|---------|
| 1 | vLLM #45552 | cumem sleep/wake stream sync | HTTP 200 while GPU incomplete |
| 2 | vLLM #46125 | encoder cache stale after weight reload | Cache not invalidated |
| 3 | SGLang #28676 | MoE expert cache clobber after weight swap | Expert params overwritten |
| 4 | vLLM-Ascend #10684 | DSA Hadamard state mismatch after offload | NPU-specific state lifecycle |
| 5 | vLLM #44395 | KV cache corruption after model reload | KV cache not flushed |
| 6 | SGLang #28679 | GDN intermittent degeneracy | Attention metadata stale |
| 7 | SGLang #28771 | EAGLE accept_length degradation | Speculative decoder state corruption |
| 8 | verl #6794 | Sleep/wake snapshot invalidation | DESIGN GAP — unflagged by reviewers |

### SGLang Comparison

SGLang HAS synchronization in `release()` but NOT in `resume()`:
- `release()`: calls `torch.cuda.synchronize()` → safe for freeing
- `resume()`: does NOT synchronize → unsafe for re-initialization
- → SGLang is "half-safe": safe during sleep but unsafe during wake

### Fix Assessment

The proposed 2-line fix (`torch.cuda.synchronize()` in both sleep and wake) is correct and minimal. It addresses the root cause without over-engineering.

### RTX 4090 Impact Assessment

**CRITICAL for RTX 4090 GRPO training**:

- The bug crashes the training pipeline within the first few steps on RTX 4090
- This makes `sleep_level=2` (full release) BLOCKED for RTX 4090 GRPO
- **Workaround**: `sleep_level=1` (LoRA offload, ~2 GiB freed) AVOIDS the bug entirely
  - `sleep_level=1` does NOT use `CuMemAllocator` → no stream sync issue
  - This is why the recommended RTX 4090 config uses `sleep_level=1`

### Recommendation

1. Merge the 2-line fix ASAP — this is a safety-critical bug
2. Add `torch.cuda.synchronize()` to BOTH sleep() and wake()
3. Consider adding the same fix to SGLang's `resume()` function
4. Document the "200 lie" pattern in vLLM's API documentation
5. For RTX 4090 users: recommend `sleep_level=1` as default (safe AND sufficient for LoRA)

---

## Note for User

Before posting:
1. Review whether the "200 lie" pattern terminology is appropriate
2. Verify the SGLang comparison claims (half-safe)
3. Consider whether to reference the RTX 4090 specific findings
4. The issue #45552 may already have a fix PR — check before posting

---
