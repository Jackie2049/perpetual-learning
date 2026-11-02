# 技术掘金任务队列

更新时间：2026-11-02。规则见 [AGENTS.md](AGENTS.md)，现场见 [CONTEXT.md](CONTEXT.md)。技术秘书优先；以下外部状态均来自迁移前记录，不代表本次重新查询。

## 当前方向

2026-10-23 用户指定两周聚焦 UniRL 与 verl；DeepInfer 为常规关注。更早的「一周三重点」不覆盖较新的优先级。到期后根据用户新指令与实际证据调整，不默认为永久优先。

## TM-001 — verl 学习与贡献

- **问题与价值**：围绕 RL 训练生命周期、权重同步与量化路径寻找可验证的实际问题，形成可复用理解或贡献。
- **优先级 / 负责人 / 状态**：掘金重点；M1 做本地学习、M5 按真机分工补证；等待（已有候选待决，新问题须另有证据）。
- **下一步与等待条件**：#8313 回复草稿 v2 待用户逐字审定；#8250 补证待授权。#7382 Actor 创建/取消边界、#7971 共享重载研究待实质反馈或独立缺口，不因状态无变化重复扫描。
- **停止条件**：无新证据、缺少验证条件或候选已失效时停止重复投入；不得自行回复上游或推送已关联 PR 的 head 分支。
- **成果入口**：[tracker](open-source/verl/tracker.md)、[回复草稿](open-source/verl/draft-8313-review-reply.md)、[生命周期笔记](notebook/frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md)、[量化审计](notebook/frameworks/verl-quant-layer-audit.md)。

## TM-002 — UniRL 学习与贡献

- **问题与价值**：理解训练架构并验证贡献候选，当前跟踪 #523 anchor lifecycle 修复及 PR #629。
- **优先级 / 负责人 / 状态**：掘金重点；沿用原候选负责人，真机验证归 M5，不重新分派；等待。
- **下一步与等待条件**：2026-11-02 已按用户指示重开 #629，等待维护者 review；L1 权重验证需要 ≥8 卡与存储。新反馈出现后准备本地可审阅方案，上游动作另需逐项授权。
- **停止条件**：无新反馈或无算力条件时不重复验证；历史重开授权不延伸为代码更新或评论授权。
- **成果入口**：[候选](open-source/unirl/candidate-523-anchor-lifecycle.md)、[调研](notebook/frameworks/unirl-survey.md)。

## TM-003 — DeepInfer 生态与贡献跟踪

- **问题与价值**：理解生产推理框架的设计与贡献可行性，避免在高门槛方向重复投入。
- **优先级 / 负责人 / 状态**：常规关注；沿用原贡献负责人；等待。
- **下一步与等待条件**：用户 fork 的 DeepInfer-Go PR #2 待用户审视；出现新源码、反馈或独立问题后再选验证动作。硬件适配项目见 [TS-002](queue_tech-secretary.md#ts-002--deepinfer-平民硬件适配)，不在两队列重复调度。
- **停止条件**：没有新证据时停止例行重复检查；上游行动仍需用户明确授权。
- **成果入口**：[贡献入口](open-source/README.md)、[生态判断](notebook/frameworks/deepseek-ecosystem-pr-adoption.md)。

## 已闭环与保留方向

- **TM-004 — AReaL prompt_mean：终止修复候选。** M5 单一负责人；2026-10-20 实测跨 M 与 token_mean 一致，不构成 defect，修复撤销，不提 issue/PR。M1 不继续修改脚本或演化并行方案，用户明确的只读 review 除外。证据及分工：[handoff](open-source/areal/handoff-m5.md)、[tracker](open-source/areal/tracker.md)。AReaL 的长期学习偏好仍保留，新的问题应独立登记。

## 新选题要求

写清问题、已有证据、探索价值、下一次验证与停止条件再投入。没有新信息增量时保持等待或终止；不得以文件数、提交数或自评分数替代价值。
