# vLLM：权重更新后编码器与前缀缓存失效问题

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** 权重更新后编码器与前缀缓存失效问题。[PR #46125](https://github.com/vllm-project/vllm/pull/46125)

**【类型】** 技术评论

**【价值评估】** 3 / 10 — 能沉淀缓存与权重版本绑定经验，但目标 revert 已完成，旧评论无需再发布。

**【进展】** 历史归档：2026-10-19 核验 [PR #46125](https://github.com/vllm-project/vllm/pull/46125) 已于 6 月合并；本文是针对当时问题的未发布草稿。

**【大白话讲明白】**

模型权重换了，之前根据旧权重计算的缓存通常不能直接继续使用。如果只更新模型却保留这些结果，就可能输出错误答案。本文从这个角度分析一次回退改动。

**【What–Why–How】**

- **What：** 权重更新与 encoder/prefix cache 的一致性及 revert 背景。
- **Why：** 缓存内容依赖模型版本，更新流程若未使缓存失效就会混用新旧状态。
- **How：** 整理更新、清理和恢复顺序及跨框架对照；保留为经验，不把别人的 revert 记成我方贡献。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> ★★★★★★★★ DRAFT — NEEDS USER AUTHORIZATION BEFORE POSTING
> Priority: P6 (RTX 4090 RLHF critical — same pattern family as SGLang #28676, #28679, vLLM #44395)
> Target: https://github.com/vllm-project/vllm/pull/46125

---

## Comment Body

This revert raises a critical concern for RLHF/GRPO training workflows where weight updates happen automatically every training step (~30-60 seconds).

**The core issue is a State Lifecycle Mismatch at the weight-reload boundary:**

When weights are updated but GPU-resident caches (prefix KV blocks, encoder outputs) are NOT reset, subsequent inference silently mixes state computed with old weights and new weights. This produces subtly incorrect outputs without any error signal — the worst possible bug pattern for RL training.

**Cross-framework evidence that cache invalidation at weight-reload boundaries is essential:**

1. **SGLang #28676** — MXFP8 MoE shuffle cache was CLOBBERED after RL weight reload (64x accuracy blowup). Fix: `dict.clear()` on cache + weight-load funnel call (+28/-2). If cache isn't invalidated, stale routing decisions produce catastrophic errors.

2. **SGLang #28679** — GDN intermittent decode degeneracy worsens over uptime, clears on restart. Root cause: GPU-resident state lifecycle mismatch. Same pattern family.

3. **vLLM #44395** — `wake_up(tags=["weights"])` + forward → illegal memory access because KV cache was still "asleep" (stale). Blocked, not merged.

4. **vLLM-Ascend #10684** — DSA Hadamard stored as CLASS VARIABLE (not model buffer) → invisible to `named_buffers()` → ALL-ZERO after sleep/wake. verl RLHF BLOCKER.

5. **verl V1 trainer** — Correctly handles this via lifecycle: `release_kv_cache_replicas()` (step 2) → `update_weights()` → `resume_kv_cache_replicas()` (step 6). KV cache is freed and re-allocated, not just "reset" — inherently invalidating stale state.

**The mathematical risk for GRPO:**

In GRPO training, advantage computation relies on correct logprob outputs from the rollout engine. If stale KV blocks produce subtly wrong logits, the advantage estimates are corrupted. Since GRPO normalizes by group mean/std, even small systematic biases compound across training steps — no NaN signal, just gradual training degradation.

**Proposed solution instead of full revert:**

Rather than removing cache reset entirely, make it configurable:

```python
def finish_weight_update(self, *, reset_cache: bool = True) -> None:
    """Finish the current weight update."""
    self.llm_engine.collective_rpc("finish_weight_update")
    if reset_cache:
        self.llm_engine.reset_prefix_cache()
        self.llm_engine.reset_encoder_cache()
```

- RL training: `reset_cache=True` (default, safe — stale state is dangerous)
- One-shot weight load: `reset_cache=False` (user can control if they know weights won't change again)

This preserves the safety guarantee for automated RL workflows while giving manual control for one-shot scenarios.

For encoder cache specifically, a more robust long-term fix would be adding weight version to cache keys (so stale entries are automatically evicted). But the short-term safety measure is cache reset after weight update — removing it creates a RLHF/GRPO regression risk.

---

## Priority Assessment

- **P6**: Not a unique theoretical contribution (pattern family is well-documented), but the cross-framework evidence matrix and RLHF/GRPO-specific risk analysis provide actionable value
- **RTX 4090 impact**: CRITICAL — verl sync trainer + vLLM rollout is the #1 RTX 4090 GRPO deployment path. If cache reset is removed, this path becomes unsafe.
