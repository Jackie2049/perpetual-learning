# rLLM：GRPO 分组失效的旧方案（已撤回）

<!-- contribution-summary:start -->
**【仓库】** [rLLM](https://github.com/rllm-org/rllm)（Agent 强化学习框架）

**【贡献点】** GRPO 分组失效的旧方案（已撤回）。[Issue #605](https://github.com/rllm-org/rllm/issues/605)

**【类型】** 漏洞诊断

**【价值评估】** 2 / 10 — 保留误判与修正经验；直接删除 trajectory.name 会破坏多 Agent 用例，禁止照此执行。

**【进展】** 旧方案已撤回：后续反馈指出仅按 task_id 分组不适用于多 Agent；应阅读 [v2 修正方案](rllm-605-grpo-grouping-revised-fix-v2.md)。

**【大白话讲明白】**

GRPO 要把可比较的多条回答放到一组。如果每组只有一个回答，就无法形成预期的组内比较。但简单把不同 Agent 的结果全混到一起也不对，因此原先“一行删掉名称”的方案被撤回。

**【What–Why–How】**

- **What：** task_id 与 trajectory.name 共同决定的 GRPO 分组。
- **Why：** 组太细会失去比较对象，组太粗又会混入语义不同的 Agent 结果。
- **How：** 保留旧推理用于追溯；新工作使用 v2 的可配置分组设计，并验证单 Agent 与多 Agent 两类场景。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 2026-06-18 | Comment draft for posting on rLLM #605
> ★★★★★★★★ CRITICAL: GRPO grouping by task_id:trajectory.name → group size = 1 → GRPO BROKEN!
> ★★★★★★★★ 1-line fix: change grouping key from task_id:name to task_id
> ★★★★★★★★ Cross-framework evidence: verl groups by prompt, not by agent name

---

## Comment Body Draft

```markdown
## Root Cause Analysis + Proposed Fix

I've done a source-level analysis of this bug and can confirm GRPO grouping is broken.

### Root Cause

In `rllm/trainer/algorithms/transform.py:127`, groups are keyed by `f"{task_id}:{trajectory.name}"`:

```python
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)  # line 127
```

`trajectory.name` comes from `agent.name` (`types.py:552`), defaulting to `_DEFAULT_TRAJ_NAME = "default_traj_name"` (`types.py:91`). Different agents/flows have different names, so responses to the **same prompt** end up in **separate groups** → group size = 1.

### Why Group Size = 1 Breaks GRPO

GRPO advantage = `(reward - mean(group_rewards)) / std(group_rewards)`.

With group size = 1: `mean = reward`, `std = 0`. The current code handles `std=0` by returning the raw reward as "advantage" — this provides **no variance reduction**, making GRPO equivalent to REINFORCE.

### Proposed Fix (1 line)

Change the grouping key from `task_id:trajectory.name` to `task_id`:

```python
# Before (broken):
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)

# After (correct):
trajectories_by_name[task_id].append(trajectory)
```

Same fix for line 128 (metadata grouping).

### Cross-Framework Evidence

**verl** groups by prompt/task_id — all responses to the same prompt are in one batch group, regardless of which agent produced them. This is the standard GRPO approach.

**DeepSpeed** and **Megatron** also group by prompt for GRPO — the trajectory/agent name is irrelevant to the advantage computation.

### Impact

This bug means **ALL rLLM GRPO training** is producing incorrect advantages with zero variance reduction. Combined with #663 (Step.output was None → all rewards = 0.0, now fixed), any pre-#663 training was doubly broken, and post-#663 training still has broken grouping.

Would be happy to submit a PR with this fix if the maintainers are open to it.
```

---

## Posting Strategy

1. Post this comment on rLLM #605 → provides actionable root cause + fix
2. If no response in 7 days → submit a PR with the 1-line fix
3. Track engagement → if community responds, collaborate on proper fix

## Priority: P9 C7 (HIGH) — GRPO grouping bug fix

★★★★★★★★★ This is a UNIQUE contribution:
  → We found the root cause at source level (transform.py:127)
  → We have cross-framework evidence (verl groups by prompt)
  → The fix is 1 line → minimal → easy to review
  → 18+ days with ZERO comments → our comment will be the FIRST substantive analysis
```

---

## References

- Source reading: notebook/frameworks/rllm-605-grpo-grouping-bug-source-reading.md
- GRPO practical guide: notebook/frameworks/rllm-grpo-practical-training-guide.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- Issue: https://github.com/rllm-org/rllm/issues/605
