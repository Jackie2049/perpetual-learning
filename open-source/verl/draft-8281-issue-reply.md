# Issue #8281 回复草稿（verl 共享 prompt cap 忽略 data.truncation）

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架，issue #8281 / PR #8313）

**【贡献点】** 共享 prompt cap 遵循 `data.truncation` 四策略的修复（[PR #8313](https://github.com/verl-project/verl/pull/8313)，用户已关闭待 review）+ middle/cap=1 的 `-0` 边界修复（fork 分支 [commit `6db4b38d`](https://github.com/Jackie2049/verl/commit/6db4b38d)）+ [本 issue 回复草稿正文](#issue-8281-回复草稿verl-共享-prompt-cap-忽略-datatruncation)。

**【类型】** 技术回复草稿（对外发布待用户授权）

**【价值评估】** 7 / 10 — 确认契约问题、说明修复状态与边界补充，向 reporter 提供完整技术答复；发布时机由用户决定。

**【进展】** CPU 测试 12 项全过（含 cap=1 回归，对照验证有效）；GPU e2e 冒烟已由 M5 完成（error 策略 fail-loudly）；修复 commit 在 fork 分支 `fix-8281-truncation-policy`，待用户决定重开 PR。

**【大白话讲明白】** verl 对超长 prompt 的截断之前不管配置、一律只留尾部；现在改为按 `data.truncation` 四策略处理（默认 error 直接报错）。审查中发现 middle 策略在「上限=1」的极端情况下会因 Python 的 `-0` 陷阱把 10 个 token 变成 11 个，已一并修复并补测试。

**【What–Why–How】**

- **What：** 共享 prompt cap 遵循 `data.truncation` 四策略的修复（PR #8313）+ middle/cap=1 边界修复 + 本 issue 技术回复草稿。
- **Why：** reporter 将 #8281 定位为「共享 cap 契约讨论」，修复点在单一共享路径 `_cap_text_prompt_length`（覆盖 SingleTurn/ToolAgent 所有 CT 构建 loop），比旧 PR #4600 更贴合当前架构；边界修复避免方法在 cap=1 时违反上限。
- **How：** 回复确认复现与修复状态、说明 middle/cap=1 反例与修复、提出 error 训练器行为契约问题供维护者对齐；发布前需用户授权（铁律：不在主仓库行动）。
<!-- contribution-summary:end -->

> 定位：技术确认 + 修复进展 + 诚实的边界补充。发不发、何时发由用户决定。

---

Thanks for the detailed write-up — I can confirm the contract issue and
summarize the current fix state, plus one boundary case we found while
reviewing it.

## 1. Fix status: `data.truncation` is now honored in `_cap_text_prompt_length`

A fix for the shared cap is in [PR #8313](https://github.com/verl-project/verl/pull/8313)
(closed pending review; the branch will be reopened once approved). It makes
`_cap_text_prompt_length` honor the documented `data.truncation` policy:

- `error` (default): fail loudly instead of silently left-truncating;
- `left` / `right` / `middle`: truncate with a warning as documented.

CPU unit tests (12, including the new cap-1 regression below) pass; an e2e
GPU smoke run also passed with the default `error` path fail-loudly under an
overlong prompt.

## 2. Boundary case found during review: `middle` at cap=1

While reviewing the `middle` branch we found an edge case worth fixing before
merge:

```python
head = (prompt_length + 1) // 2
return prompt_ids[:head] + prompt_ids[-(prompt_length - head):]
```

When `prompt_length == 1`, `head = 1` and the tail budget is `0`. In Python
`-0 == 0`, so `prompt_ids[-0:]` selects the **entire** list instead of an
empty tail — a 10-token prompt is returned as 11 tokens, violating the cap.
`left`/`right` handle cap=1 correctly; `middle` was the only affected branch,
and the original test parametrization only covered caps 4/5/6.

The follow-up patch replaces the negative tail slice with a forward slice
`prompt_ids[len(prompt_ids) - tail_length:]` (naturally empty when the budget
is 0) and adds a regression test. This is included in the same PR branch.

## 3. Contract question

Your framing — that this is a shared-cap contract decision rather than a
per-loop truncation — matches where the fix lands: `_cap_text_prompt_length`
is the single shared path used by both `SingleTurnAgentLoop` and
`ToolAgentLoop` (and any other CT-built loop), which is why fixing it there
covers more than the earlier #4600 approach (pre-refactor, dirty against
current main).

One thing worth confirming with maintainers: the `error`-raises behavior in
the *trainer* path — i.e. whether a rollout worker crashing on an overlong
prompt should abort the step or be surfaced differently. Happy to align the
implementation with whatever the project prefers.
