# perpetual-learning

> 持续学习、实践、生产的自进化智能引擎。

## 项目定位

持续进行学习、实践、生产的自进化智能引擎，不间断地学习技术知识、积累实战经验、发掘可探索的方向、产出有价值的成果。领域为AI技术、优先关注Infra。用户驱动时，优先承接并细化用户的需求和想法；真实运行中，持续积累可复用的实战经验。上述的技术掘金、技术秘书、经验沉淀等任务有机结合，持续增强引擎能力，最终在技术视野、想法探索、实战经验、业界影响力等方面为用户的保驾护航并做出有价值的贡献

## 如何运作

- **技术掘金**：持续自主学习、实践和探索 AI 技术，优先关注 Infra，寻找值得深入和贡献的方向，产出笔记、工具、开源贡献、自营成果或论文。
- **技术秘书**：优先承接用户的需求与模糊想法，通过尽调、综述、价值与风险分析、概念验证和实际执行推进目标。
- **经验沉淀**：从真实运行及其反馈中积累可复用经验，逐步转化为知识、工具和技能，增强后续工作的能力。

## 从哪里开始

| 入口 | 职责 |
|---|---|
| 本页 | 项目总览和对外呈现 |
| [AGENTS.md](AGENTS.md) | Agent 内核：工作原则、行为方式、经验积累与执行边界 |
| [WORK.md](WORK.md) | 当前工作上下文：用户需求、探索进度、待讨论事项和下一步 |
| [notebook/readme.md](notebook/readme.md) | 技术学习笔记总览 |
| [tools/readme.md](tools/readme.md) | 可复用工具、价值标准与历史脚本入口 |
| [open-source/README.md](open-source/README.md) | 按开源项目组织的贡献记录 |
| [diary/readme.md](diary/readme.md) | 工作日志与记录方式 |

Agent 接续工作时，先读 AGENTS.md 和 WORK.md，再按当前任务查阅相关资产。

## 文件组织结构

```text
perpetual-learning/
├── README.md                 总览和对外呈现
├── AGENTS.md                 Agent 内核
├── WORK.md                   当前工作上下文
├── notebook/                 技术学习笔记
│   ├── readme.md             板块总览
│   ├── fundamentals/         基础知识
│   ├── projects/             项目架构与源码阅读
│   ├── experiments/          实验记录与结论
│   ├── papers/               已有论文阅读笔记
│   ├── roadmap.md            学习与实践规划
│   └── knowledge-map.md      知识整理与总览
├── tools/                    可复用工具
│   ├── readme.md             工具入口与价值标准
│   └── legacy/               一次性、纯实验性或待整合的历史脚本
├── open-source/              按目标开源项目组织贡献
│   ├── README.md             贡献板块总览
│   ├── SOP.md                贡献工作方法
│   ├── verl/                 verl 贡献记录
│   └── moonshot/             MoonshotAI 项目贡献记录
├── diary/                    按日期记录实际工作
│   ├── readme.md             日志总览
│   └── YYYY-MM-DD.md         当日工作记录
├── experiments/              已有训练实验启动脚本
├── scripts/                  已有执行、验证与编排脚本
├── configs/                  已有配置
├── csrc/                     CUDA/C++ 实现与相关验证
└── .claude/skills/            已有 Agent 技能
```

上图列出主要目录；其他已有专题、环境文件和历史结果保留原位。

### 内容归属

- **知识与理解**放在 notebook，延续现有基础知识、源码阅读和实验记录结构。学习规划由 roadmap.md 维护，知识关联由 knowledge-map.md 整理。
- **可复用能力**沉淀到 tools，重视实际用途和验证证据，优先复用已有实现。一次性、纯实验性和低价值重复脚本进入 tools/legacy；历史资产正在逐项整理。
- **开源贡献**按目标项目放入 open-source，例如现有的 verl、moonshot；未来有 vLLM 贡献材料时沿用 open-source/vllm/，无需预建空目录。
- **工作过程**记入 diary；当前上下文维护在 WORK.md。详细成果在所属位置保存，其他入口用链接引用。

### 两处 experiments 的关系

根目录 experiments/ 目前保存 3 个 GRPO 训练启动脚本，用于执行实验；notebook/experiments/ 保存实验记录，用于说明问题、环境、过程、结果和结论。执行材料与记录可以互相链接，后续是否合并仍待确定，本次保留既有位置。

根目录没有 projects/。此前提出的该目录用于未来自营项目，目前没有必要新增；notebook/projects/ 继续专门负责项目源码学习。

## GitHub

[Jackie2049/perpetual-learning](https://github.com/Jackie2049/perpetual-learning)
