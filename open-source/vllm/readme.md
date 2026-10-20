# vLLM 开源贡献

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** vLLM 开源贡献。[项目](https://github.com/vllm-project/vllm)

**【类型】** 项目总览

**【价值评估】** 7 / 10 — 集中同一项目的候选、证据和进展，减少误读与重复探索；目录本身不计作上游成果。

**【进展】** [跨请求污染技术评论](https://github.com/vllm-project/vllm/issues/61160#issuecomment-6258592764)已发布，历史草稿按各自状态评估或归档。

**【大白话讲明白】**

这里整理我们在 vLLM 上围绕缓存隔离、数值一致性、调度与传输指标做过的调查、方案和跟进。先看每份文档的进展和大白话说明，再决定是否深入代码与实验。

**【What–Why–How】**

- **What：** vLLM 的贡献材料入口与当前关注范围。
- **Why：** 同一问题会经历草稿、验证、撤回与发布，文件名本身不能说明当前可用性。
- **How：** 通过索引进入规范化文档，沿证据与关联讨论回看；技术原理学习另见 notebook/frameworks。
<!-- contribution-summary:end -->

保存贡献候选、方案、验证及跟进材料。以下是历史积累，草稿不代表已提交，原文状态不代表当前上游状态；继续推进前先核对关联讨论和验证证据。

返回 [贡献总览](../README.md) · [框架学习](../../notebook/frameworks/readme.md#vllm)。

## 材料索引

- [BudgetRefiner SLO → vLLM Upstream Contribution Plan](budgetrefiner-vllm-contribution-plan.md)
- [BudgetRefiner SLO → vLLM Upstream PR Draft](budgetrefiner-vllm-pr-draft.md)
- [vLLM Contribution Plan: Issue #21948](vllm-21948-contribution.md)
- [vLLM Issue #39096 — Batch Invariance Comment Draft](vllm-39096-batch-invariance-comment-draft.md)
- [vLLM Issue #44701 — Comment Draft](vllm-44701-comment-draft.md)
- [vLLM PR #45038 — SM89 FP8 KV Crash Comment Draft](vllm-45038-sm89-fp8-comment-draft.md)
- [vLLM #45552 Review Comment Draft](vllm-45552-review-comment-draft.md)
- [vLLM #46125 Comment Draft — P6 C19: Stale Encoder/Prefix Cache After Weight Update Revert](vllm-46125-encoder-cache-revert-comment-draft.md)
- [vLLM 开源贡献计划](vllm-contribution-plan.md)
- [vLLM 开源贡献准备 — 源码定位](vllm-contribution-prep.md)
- [vLLM EPLB 贡献分析 — Issue #30075](vllm-eplb-contribution.md)
- [vLLM PR #45157 Resubmission Draft](vllm-pr-45157-resubmission-draft.md)
- [vLLM SM89 贡献策略分析 (2026-06-15)](vllm-sm89-contribution-strategy.md)

## 草稿状态核验（2026-10-19，GitHub API 实测）

| 草稿 | 目标 | 当前状态 | 结论 |
|---|---|---|---|
| #39096 批次不变性 | issue | **仍 open**，9 评论；YM2132 7-30 认领推进中，aoshen02 8-16 补充数据 | 有残余价值：GRPO/verl 训练影响分析 + 三机制框架尚未有人提出；其余内容与现有讨论重叠。**可发布（需授权）**，建议精简为增量部分 |
| #44701 前缀缓存 hash 碰撞 | issue + PR #44706 | 双双 closed（9-29/9-30，maintainer njhill，均未合并） | 无发布对象，作废 |
| #45038 SM89 FP8 KV | PR（已自关）+ 底层 issue #44879 | PR 未合并自关（9-21）；#44879 **仍 open**，有 stale bot + akshathtiwari 实测噪音 | 残余价值低：草稿信息基于 v0.23（6 月），已 4 个月旧；issue 上有人在实测。**可发布（需授权）** 但增量有限 |
| #45552 cumem 流同步 | 修复 PR | 未合并关闭（8-14，intarweb） | 无发布对象，作废 |
| #46125 encoder cache revert | revert PR | **已合并**（6-20，vllm-bot） | 事过境迁，作废 |

**行动建议**：只有 #39096 值得发布，且应大幅精简成增量评论（GRPO 影响 + 三机制组合）。发布前需用户授权。

### 新核验（2026-10-19 追加）

- **#61160 跨请求污染修复定位**（[分析](vllm-61160-crosstalk-fix-locate.md)）：0 引用 PR；维护者确认 v0.31.0 复现 / main 已修复但无人指出修复 PR。定位：**#59103**（10-06，session 截断 stale hash——多轮会话场景主修复）+ **#61174**（10-17，MTP stale 共享 block-table 行）+ **#51899**（9-30，extra_keys 域打标 = 我们 6 月 #44701 家族修复）。**评论已发布**（2026-10-19，[#issuecomment-6258592764](https://github.com/vllm-project/vllm/issues/61160#issuecomment-6258592764)），待回应跟进。

- **#61229 NixlConnector lease 过期无声消费跨请求 KV（2026-10-20 核验）→ 已被认领，无 PR 竞争空间**：[issue](https://github.com/vllm-project/vllm/issues/61229)（reporter binxuwang2002-lang 深度写清机制，10-16 维护者 robertgshaw2-redhat 认可 "positive fix"）。10-20 交叉验证 PR：**#62065**（OPEN，vadiklyutiy 本人，`Verify pull reads against producer lease slots`，B200 2P1D 硬件验证 fail/recompute 两态）+ **#61549**（CLOSED，sawsa307，fake-clock 单测）+ **#61578/#61488**（sawsa307，KV checksum 长线方案）。修复方向 = reporter 建议的 "transfer 末尾查 request id + verdict 送达 consumer"，已被上述 PR 完整覆盖。**结论：不再投入**（本地 6 月源码已确认无条件释放逻辑存在，但修复已在上游进行）。

### 扫描补充（2026-10-19）

- **#60494（GLM-5.2-FP8 + 前缀缓存 + FP8 KV 返回空）→ 排除**：ROCm/AITER（AMD MI300X）平台——FP8 KV 家族但平台在 CUDA 经验区之外；志愿者认领但无硬件；维护者最新 main 无法复现（reporter 用 0.27.1 旧版）。不深挖。
