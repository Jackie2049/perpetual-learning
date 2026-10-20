# rLLM 开源贡献

<!-- contribution-summary:start -->
**【仓库】** [rLLM](https://github.com/rllm-org/rllm)（Agent 强化学习框架）

**【贡献点】** rLLM 开源贡献。[项目](https://github.com/rllm-org/rllm)

**【类型】** 项目总览

**【价值评估】** 7 / 10 — 集中同一项目的候选、证据和进展，减少误读与重复探索；目录本身不计作上游成果。

**【进展】** 旧一行修复已撤回，继续阅读应以可配置分组的 v2 方案为起点。

**【大白话讲明白】**

这里整理我们在 rLLM 上围绕单 Agent 与多 Agent 的 GRPO 分组做过的调查、方案和跟进。先看每份文档的进展和大白话说明，再决定是否深入代码与实验。

**【What–Why–How】**

- **What：** rLLM 的贡献材料入口与当前关注范围。
- **Why：** 同一问题会经历草稿、验证、撤回与发布，文件名本身不能说明当前可用性。
- **How：** 通过索引进入规范化文档，沿证据与关联讨论回看；技术原理学习另见 notebook/frameworks。
<!-- contribution-summary:end -->

保存贡献候选、方案、验证及跟进材料。以下是历史积累，草稿不代表已提交，原文状态不代表当前上游状态；继续推进前先核对关联讨论和验证证据。

返回 [贡献总览](../README.md) · [框架学习](../../notebook/frameworks/readme.md#rllm)。

#605 的旧补丁保留用于追溯；后续修正指出直接移除 trajectory.name 会破坏多 Agent 场景，请先读 [v2 修正方案](rllm-605-grpo-grouping-revised-fix-v2.md)。

## 材料索引

- [rLLM #605 GRPO Grouping Bug — Comment Draft](rllm-605-grpo-grouping-bug-comment-draft.md)
- [rllm-605-grpo-grouping-fix.patch](rllm-605-grpo-grouping-fix.patch)
- [rLLM #605 GRPO Grouping Bug — Revised Fix Proposal (v2)](rllm-605-grpo-grouping-revised-fix-v2.md)
