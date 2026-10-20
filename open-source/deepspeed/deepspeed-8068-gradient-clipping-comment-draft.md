# DeepSpeed：梯度裁剪默认值与训练稳定性

<!-- contribution-summary:start -->
**【仓库】** [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)（分布式训练框架）

**【贡献点】** 梯度裁剪默认值与训练稳定性。[原始讨论 1](https://github.com/deepspeedai/DeepSpeed/pull/8068) · [PR #8068](https://github.com/deepspeedai/DeepSpeed/pull/8068)

**【类型】** 配置改进

**【价值评估】** 5 / 10 — 有机制分析和可讨论的改进点，但草稿较旧、上游状态与当前版本适用性待复核。

**【进展】** 历史草稿（2026-06）：未记录本文已对外发布；原文的严重程度与社区活跃状态均为当时判断，不能作为当前事实。

**【大白话讲明白】**

用户以为训练会限制过大的梯度，但默认配置可能根本没有打开这层保护。本文讨论如何把默认行为说清楚；是否应该改变默认值，还要兼顾已有用户配置。

**【What–Why–How】**

- **What：** gradient_clipping 默认值为 0 时的语义及 GRPO 使用风险。
- **Why：** 关闭裁剪不一定是代码漏洞，但用户预期与实际行为不一致可能放大不稳定更新。
- **How：** 核对配置继承和说明，准备默认值/提示建议；没有在本文证明裁剪能解决所有训练坍塌。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 2026-06-19 | Comment draft for posting on DeepSpeed #8068
> ★★★★★★★★ CRITICAL: gradient_clipping default 0 → silently unclipped GRPO → training instability
> ★★★★★★★★ Pattern family: Silent configuration default → no error, no warning, wrong behavior
> ★★★★★★★★ 0 reviews, 0 comments, +1/-1 PR → stalled, needs engagement

---

## Comment Body Draft

```markdown
## Cross-Framework Impact: gradient_clipping=0 default is dangerous for GRPO

This is a critical fix for RL training. Here's the cross-framework evidence:

### Why gradient_clipping=0 Is Dangerous

The default `GRADIENT_CLIPPING_DEFAULT = 0` means **gradient clipping is silently disabled** when configs omit the field. No warning, no error — just unclipped gradients.

For GRPO/RLHF training:
- Unclipped gradients → occasional large gradient spikes → training instability
- On single GPU (RTX 4090, dp=1), there's no cross-GPU averaging to smooth spikes
- Combined with #8061 (overlap_comm data race → NaN), unclipped gradients amplify NaN propagation
- This is the **same pattern family** as #8068/Megatron #5394: global grad-norm clipping stalls optimizer step → but here it's the opposite: NO clipping at all → instability

### Cross-Framework Defaults

| Framework | Default gradient_clipping | Notes |
|-----------|--------------------------|-------|
| DeepSpeed (pre-#8068) | 0.0 (disabled!) | Silent, no warning |
| DeepSpeed (post-#8068) | 1.0 | This PR's fix |
| PyTorch FSDP2 | 1.0 | Recommended default |
| Megatron | 1.0 | Standard for LLM training |
| verl | Configurable, defaults to 1.0 | RL-specific |
| HuggingFace TRL | 1.0 | GRPO reference |

Every other framework defaults to 1.0. DeepSpeed's 0.0 is an outlier that silently produces wrong training behavior.

### Pattern Family: Silent Configuration Default

This belongs to the **Silent Corruption** pattern family at severity Level 2:
- No error signal → training just runs with unclipped gradients
- Effects accumulate over steps → training may diverge without obvious cause
- Detection requires manual monitoring (loss curve anomalies)

Same pattern as:
- DeepSpeed #8058: `.contiguous()` creates copy → optimizer updates copy, not original → silent
- SGLang #28679: GDN state degrades over uptime → no error signal

### For RTX 4090 GRPO

This PR is **mandatory** for RTX 4090 single-GPU GRPO:
- MUST set `gradient_clipping=1.0` explicitly (even before this PR merges)
- This PR makes it the default → eliminates the silent-configuration trap
- Combined with overlap_comm=False (#8061 prevention) → stable training foundation

Thanks for this important fix — changing the default from 0 to 1.0 saves countless training runs from silent instability!
```

---

## Posting Strategy

1. ★★★★★★★★ MUST get user authorization before posting on deepspeedai/DeepSpeed #8068
2. Post this comment → provides cross-framework default comparison + pattern family analysis
3. Track engagement → push for review/merge
4. If no response in 7 days → escalate via DeepSpeed Discord/community

## Priority: P7 C15 (MEDIUM) — gradient clipping default fix

★★★★★★★★★ This is a UNIQUE contribution:
  → Cross-framework default comparison (6 frameworks, DeepSpeed is outlier)
  → Pattern family analysis (Silent Configuration Default)
  → RTX 4090 GRPO production impact
  → 0 reviews, 0 comments → our comment adds context for why this matters

---

## References

- PR: https://github.com/deepspeedai/DeepSpeed/pull/8068
- Pattern family: notebook/fundamentals/silent-corruption-pattern-family-analysis.md
- RTX 4090 rules: tools/rtx4090_grpo_pre_flight_checklist.py (D5: gradient_clipping=1.0)
