# perpetual-learning — 当前工作上下文

更新时间：2026-10-18

## 当前阶段

保留已有知识和能力，重构运作模式及文件入口，为重启做准备。尚未启动持续运行或远程实验。

## 当前用户需求与决定

- 项目名称、首页标题和仓库标识统一使用 `perpetual-learning`，删除首页旧名称标识行。项目定位继续保留 [README.md](README.md#项目说明) 中的用户原文。
- README.md 是整个项目的总览、对外呈现与用户导航，按主题、框架、成果提供快捷入口；文件组织使用可点击的目录表，作为统一入口置于项目说明与运作机制之后，不再单设重复的“项目导航”章节。[AGENTS.md](AGENTS.md) 为 Agent 内核；本文件记录当前工作上下文。
- README 章节顺序：项目说明 → 运作机制 → 文件组织 → 知识导航 → 上机实战 → 主流框架 & 开源贡献 → 论文学习 & 专题研究 → 工具沉淀 & 技能打磨 → 规划 & 进展；GitHub 链接保留在末尾。
- notebook/ 保留 fundamentals/ 基础知识、experiments/ 实验记录、roadmap.md 规划、knowledge-map.md 知识整理与总览；原源码学习目录改名 frameworks/，总览见 [notebook/readme.md](notebook/readme.md)。
- tools/ 注重工具价值与复用；一次性、纯实验性和低价值重复实现归 tools/legacy/，新增 [tools/readme.md](tools/readme.md)。
- open-source/<framework>/ 按开源项目保存贡献材料，与 notebook/frameworks/ 的源码理解分开并相互链接；diary/ 延续原有功能，入口为 [diary/readme.md](diary/readme.md)。

## 本轮进展

- 已完整盘点根目录：实验源码、12 个执行脚本、32 份结果及说明、2 份环境快照统一归入 experiments/；HTML 原始资料归入 notebook/resources/。实验入口见 [实验总览](experiments/readme.md)，顶层不再保留 csrc、scripts、results 或散落 HTML/YML。
- 首页已扩展为完整导航，包含技术主题、框架学习与贡献、实验、论文、工具、技能、规划、日志及目录职责。详细知识导航在 notebook/readme.md，原工具导航保存在 [tools/legacy/catalog.md](tools/legacy/catalog.md)。
- 原源码目录 398 份材料中，366 份归入 notebook/frameworks/，32 份明确的贡献材料归入 open-source/；同步引用并建立框架和贡献索引。
- 4 份 RTX 4090 DeepSpeed 配置已移至 [experiments/rtx4090/configs/](experiments/rtx4090/configs/readme.md)，同步生成器、检查器、脚本及文档引用，移除根目录 configs/。
- 已归档 9 个明确的一次性补丁或概念模拟脚本，源码保持不变；清单与原因见 [tools/legacy/readme.md](tools/legacy/readme.md)。其余历史工具尚未全面验收，保留原位置不代表正式推荐。
- Agent 内核已写入目录职责和工具价值标准。完整变更记录见 [当日日志](diary/2026-10-18.md)。

## 待讨论与下一步

1. 按课题逐步关联现有实验执行材料与 notebook/experiments/ 的分析记录；新增实验优先就近组织，维护双向入口。
2. 维护按职责划分的一级目录；新增或整理内容时整体检查归属、引用和输出路径，避免再次堆出按文件类型划分的一级目录。根目录 projects/ 暂不新增。
3. 按价值、使用关系和证据继续筛选历史工具，避免只按文件名批量归档。已知 inference_estimator.py 有带宽硬编码和吞吐展示问题，待修复验证。
4. 目录职责确定后，确认持续运行入口、模型资源及 GPU/NPU 当前条件，从已有积累选择方向跑通接续循环。

## 接续入口与约束

- 学习与探索：[notebook/readme.md](notebook/readme.md)、[roadmap.md](notebook/roadmap.md)、[knowledge-map.md](notebook/knowledge-map.md)。尚未选定重启后的具体课题。
- 贡献：[open-source/README.md](open-source/README.md)、[verl tracker](open-source/verl/tracker.md)。旧候选需核对当前上游状态与本地证据。
- 本地目录仍为 rollout-infra，应用侧栏仍显示 RolloutInfra；保留现有路径与绑定。GitHub 仓库已更名为 [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)。
- 原有未跟踪文件 diary/2026-08-14.md 属于既有用户工作，保留原状，不夹带提交。
- 历史服务器状态、访问方式和“已验证”结论按需复核，不复述凭据。
