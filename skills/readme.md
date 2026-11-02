# Agent 技能

把实战中可复用的**方法、流程与导航**沉淀为 Skill（YAML frontmatter + Markdown 指令体）。项目总览见 [README.md](../README.md)，可执行脚本见 [tools/](../tools/readme.md)。

## 目录约定

- 每个技能一个子目录：`skills/<name>/SKILL.md`。
- 新增技能前先检索本目录与 [tools/](../tools/readme.md)，避免与工具或笔记重复；成熟后再写入 frontmatter 的 `description` 与触发词。
- 技能应写清适用场景、输入输出、依赖、验证范围与边界；不能把静态推断标成实测。

## 技能索引

| 场景 | 技能 |
| --- | --- |
| 实验执行 | [gpu-experiment](gpu-experiment/SKILL.md) · [rtx4090-training](rtx4090-training/SKILL.md) |
| 训练排错 | [grpo-debug](grpo-debug/SKILL.md) · [cross-framework-avoidance](cross-framework-avoidance/SKILL.md) |
| 推理分析 | [inference-perf](inference-perf/SKILL.md) |
| 分布式与通信 | [distributed-nav](distributed-nav/SKILL.md) · [rdma-network-nav](rdma-network-nav/SKILL.md) · [weight-sync-nav](weight-sync-nav/SKILL.md) |
| 源码导航 | [vllm-v1-nav](vllm-v1-nav/SKILL.md) · [sglang-nav](sglang-nav/SKILL.md) · [megatron-nav](megatron-nav/SKILL.md) |
| 专题研究 | [interpretability-nav](interpretability-nav/SKILL.md) · [model-merging-nav](model-merging-nav/SKILL.md) |

首页按场景的快捷入口见 [README — 工具沉淀与技能打磨](../README.md#工具沉淀--技能打磨)。

## 与 Harness 的关系

本仓库把技能放在根目录 `skills/`，与 `tools/`、`notebook/` 同级，便于版本管理与导航。具体 Agent 产品若默认只扫描 `.claude/skills/` 或用户全局目录，在本地配置中把本项目 `skills/` 加入技能搜索路径即可；不在 `.claude/` 下再维护一份副本。
