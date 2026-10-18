# 开源贡献

按目标开源项目组织贡献候选、方案、补丁、验证和跟进。这里关注“准备贡献什么、推进到哪里”；[notebook/frameworks/](../notebook/frameworks/readme.md) 关注框架原理与源码理解。两类材料相互链接，同一份成果只保留一个主要位置。

## 按项目浏览

| 项目 | 贡献材料 | 学习材料 |
|---|---|---|
| DeepSpeed | [候选、方案与记录](deepspeed/readme.md) | [源码阅读](../notebook/frameworks/readme.md#deepspeed) |
| MoonshotAI | [候选、方案与记录](moonshot/readme.md) | — |
| PyTorch | [候选、方案与记录](pytorch/readme.md) | [源码阅读](../notebook/frameworks/readme.md#pytorch) |
| rLLM | [候选、方案与记录](rllm/readme.md) | [源码阅读](../notebook/frameworks/readme.md#rllm) |
| SGLang | [候选、方案与记录](sglang/readme.md) | [源码阅读](../notebook/frameworks/readme.md#sglang) |
| TRL | [候选、方案与记录](trl/readme.md) | [源码阅读](../notebook/frameworks/readme.md#trl) |
| verl | [候选、方案与记录](verl/readme.md) | [源码阅读](../notebook/frameworks/readme.md#verl) |
| vLLM | [候选、方案与记录](vllm/readme.md) | [源码阅读](../notebook/frameworks/readme.md#vllm) |
| vLLM-Ascend | [候选、方案与记录](vllm-ascend/readme.md) | [源码阅读](../notebook/frameworks/readme.md#vllm-ascend) |

## 跨项目规划与方法

- [贡献机会与能力匹配](open-source-contribution-opportunities.md)
- [2026 年 6 月贡献行动计划](oss-contribution-action-plan-2026-06.md)
- [贡献挖掘 SOP](SOP.md)

以上计划、分数与 Issue/PR 状态均为历史记录，不能作为当前优先级或贡献成效的证明。近期工作安排见 [WORK.md](../WORK.md)。

## 组织方式

每个项目使用一个目录，按需要保存 tracker、扫描报告、候选方案、验证记录及草稿。已有文件保留原名，便于追溯。

- 学习：框架架构、源码分析、问题原理，保存到 notebook/frameworks/。
- 贡献：围绕具体改进的方案、补丁、评论及 PR 草稿，保存到 open-source/<framework>/。
- 实验：复现过程与测量结论保存到 notebook/experiments/，从贡献方案链接过去。

对外发布与提交按 [AGENTS.md](../AGENTS.md#执行边界) 执行。

## 历史扫描流程

历史流程使用 Claude Code skill `jackie-verl-pr`（本仓库未包含该技能，使用前需确认当前环境已安装），例如：
- "扫一遍 verl"
- "verl 有什么能做的"
- "verl 优化点挖掘"

Skill 会按 `SOP.md` 的 5 阶段流水线执行，产出新的 `scan-MMDD-HHMM.md` 并更新 `tracker.md`。

## 优先级口径

| 级别 | 类别 | 说明 |
|---|---|---|
| **P0** | bugfix | 正确性 bug、静默错结果、崩溃 — 接受率最高 |
| **P1** | 健壮性 | 报错信息、参数校验、防御性 guard、边界条件 |
| **P2** | 显存优化 | 内存占用 / OOM 缓解 |
| **P3** | 可量化性能优化 | ⚠️ 高失败率，必须有 profiling 证据 |
| **P4** | 其他 | 重构、文档、补测试、示例 |
