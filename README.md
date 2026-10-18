# perpetual-learning

## 项目定位

持续进行学习、实践、生产的自进化智能引擎，不间断地学习技术知识、积累实战经验、发掘可探索的方向、产出有价值的成果。领域为AI技术、优先关注Infra。用户驱动时，优先承接并细化用户的需求和想法；真实运行中，持续积累可复用的实战经验。上述的技术掘金、技术秘书、经验沉淀等任务有机结合，持续增强引擎能力，最终在技术视野、想法探索、实战经验、业界影响力等方面为用户的保驾护航并做出有价值的贡献

## 如何运作

- **技术掘金**：持续自主学习、实践和探索 AI 技术，优先关注 Infra，寻找值得深入和贡献的方向，产出笔记、工具、开源贡献、自营成果或论文。
- **技术秘书**：优先承接用户的需求与模糊想法，通过尽调、综述、价值与风险分析、概念验证和实际执行推进目标。
- **经验沉淀**：从真实运行及其反馈中积累可复用经验，逐步转化为知识、工具和技能，增强后续工作的能力。

## 项目导航

README 是整个项目的总入口。可以按下面的问题进入，也可以直接按技术主题或框架查找。各板块总览维护更完整的内容索引。

| 我想了解…… | 从这里进入 |
|---|---|
| 引擎如何工作、遵循什么原则 | [Agent 内核](AGENTS.md) |
| 现在在做什么、接下来推进什么 | [当前工作上下文](WORK.md) |
| 有哪些知识积累、如何继续学习 | [笔记总览](notebook/readme.md) · [知识地图](notebook/knowledge-map.md) · [学习路线](notebook/roadmap.md) |
| 某个框架如何实现 | [框架源码索引](notebook/frameworks/readme.md) · [下方框架导航](#框架学习与开源贡献) |
| 有哪些可贡献的技术点、准备到哪一步 | [贡献总览](open-source/README.md) · [贡献方法](open-source/SOP.md) |
| 做过哪些实验、得出了什么结论 | [实验记录](notebook/experiments/) · [下方实验导航](#实验与实战经验) |
| 有什么工具可以复用 | [工具总览](tools/readme.md) · [历史脚本](tools/legacy/readme.md) |
| 学过哪些论文与 AI 专题 | [论文导航](notebook/readme.md#论文精读) · [下方专题导航](#论文与其他-ai-专题) |
| 积累了什么工作方法与经验 | [技能入口](#工具技能与执行材料) · [问题模式](notebook/cross-framework-bug-taxonomy.md) · [工作日志](diary/readme.md) |
| 文件应该放在哪里 | [文件组织结构](#文件组织结构) |

当前处于保留积累、整理结构与重构运作模式的阶段，持续运行尚未重启。历史笔记中的性能、验证范围、硬件状态与上游进展以原文时间和条件为准，具体工作安排见 WORK.md。

## 技术知识导航

AI 技术是探索范围，Infra 是优先方向。下面提供常用主题的直接入口；更多笔记见 [技术学习总览](notebook/readme.md) 和 [基础知识目录](notebook/fundamentals/)。

| 方向 | 直接阅读 |
|---|---|
| 分布式训练 | [并行策略](notebook/fundamentals/distributed-training.md) · [ZeRO](notebook/fundamentals/zero-optimizer.md) · [FSDP](notebook/fundamentals/fsdp.md) · [混合精度](notebook/fundamentals/mixed-precision.md) |
| GPU 编程与优化 | [CUDA](notebook/fundamentals/cuda-programming-basics.md) · [Triton](notebook/fundamentals/triton-programming.md) · [FlashAttention](notebook/fundamentals/flash-attention.md) · [Profiling](notebook/fundamentals/gpu-profiling.md) |
| 通信与集群 | [NCCL](notebook/fundamentals/nccl.md) · [RDMA](notebook/fundamentals/rdma-networking.md) · [通信计算重叠](notebook/fundamentals/comm-compute-overlap.md) · [GPU 调度](notebook/fundamentals/k8s-gpu-scheduling.md) |
| 推理引擎与服务 | [KV Cache](notebook/fundamentals/kv-cache.md) · [Continuous Batching](notebook/fundamentals/continuous-batching.md) · [P/D 分离](notebook/fundamentals/disaggregated-serving.md) · [可观测性](notebook/fundamentals/inference-observability.md) |
| 模型架构与推理优化 | [MoE](notebook/fundamentals/moe.md) · [MLA](notebook/fundamentals/mla.md) · [量化](notebook/fundamentals/quantization.md) · [投机解码](notebook/fundamentals/speculative-decoding.md) |
| RL 与后训练 | [RLHF/GRPO Infra](notebook/fundamentals/rlhf-training-infra.md) · [优势估计](notebook/rl-advantage-estimators-mathematical-derivation.md) · [Policy Loss](notebook/rl-policy-loss-functions-mathematical-derivation.md) · [DPO](notebook/fundamentals/dpo.md) |
| 微调与编译 | [LoRA/PEFT](notebook/fundamentals/lora-peft.md) · [torch.compile](notebook/fundamentals/torch-compile.md) · [CUDA Graphs](notebook/fundamentals/cuda-graphs.md) |

## 框架学习与开源贡献

学习与贡献是同一条探索路径的不同产物：源码理解进入 `notebook/frameworks/`；围绕具体改进的候选、方案、补丁、Issue/PR 与评论草稿进入 `open-source/<framework>/`。贡献材料链接学习和实验证据，便于从理解继续推进到实践。

| 框架 | 架构与源码学习 | 贡献候选、方案与跟进 |
|---|---|---|
| vLLM | [源码索引](notebook/frameworks/readme.md#vllm) · [V1 架构全景](notebook/frameworks/vllm-v1-architecture-map.md) | [vLLM 贡献](open-source/vllm/readme.md) |
| SGLang | [源码索引](notebook/frameworks/readme.md#sglang) · [架构分析](notebook/frameworks/sglang-architecture.md) | [SGLang 贡献](open-source/sglang/readme.md) |
| verl | [源码索引](notebook/frameworks/readme.md#verl) · [RL Infra](notebook/frameworks/verl-rl-infra-reading.md) | [verl 贡献](open-source/verl/readme.md) · [Tracker](open-source/verl/tracker.md) |
| DeepSpeed | [源码索引](notebook/frameworks/readme.md#deepspeed) | [DeepSpeed 贡献](open-source/deepspeed/readme.md) |
| Megatron | [源码索引](notebook/frameworks/readme.md#megatron) · [张量并行](notebook/frameworks/megatron-tp-source-reading.md) | 具体候选按后续探索形成 |
| PyTorch | [源码索引](notebook/frameworks/readme.md#pytorch) | [PyTorch 贡献](open-source/pytorch/readme.md) |
| rLLM | [源码索引](notebook/frameworks/readme.md#rllm) | [rLLM 贡献](open-source/rllm/readme.md) |
| TRL | [源码索引](notebook/frameworks/readme.md#trl) | [TRL 贡献](open-source/trl/readme.md) |
| vLLM-Ascend | [源码索引](notebook/frameworks/readme.md#vllm-ascend) | [Ascend 贡献](open-source/vllm-ascend/readme.md) |
| MoonshotAI | 相关理解随具体候选链接 | [贡献记录](open-source/moonshot/readme.md) |

[其他框架与跨框架专题](notebook/frameworks/readme.md#其他框架与跨框架专题) · [跨项目贡献规划](open-source/README.md#跨项目规划与方法)

## 实验与实战经验

实验记录保存问题、环境、方法、结果与局限；经验来自实际执行和反馈，进一步沉淀为排错方法、工具与技能。

| 兴趣点 | 入口 |
|---|---|
| GPU 基础性能与测量方法 | [A16 基准](notebook/experiments/gpu-a16-benchmark.md) · [A16 Profiling](notebook/experiments/gpu-profiling-a16.md) · [RTX 4090 基准](notebook/experiments/rtx4090-benchmark.md) |
| Kernel 与推理优化 | [Triton Kernels](notebook/experiments/rtx4090-triton-kernels.md) · [KV Cache 压缩](notebook/experiments/rtx4090-kv-cache-compress.md) |
| 训练与多卡实践 | [LoRA 微调](notebook/experiments/rtx4090-lora-finetune.md) · [多卡 DDP](notebook/experiments/rtx4090-multigpu-ddp.md) · [RLHF/GRPO](notebook/experiments/rtx4090-rlhf-grpo.md) |
| 排错与跨框架经验 | [GRPO 排错](notebook/grpo-training-debugging-guide.md) · [跨框架问题分类](notebook/cross-framework-bug-taxonomy.md) · [CUDA Stream 生命周期问题](notebook/cuda-stream-use-after-free-pattern-family.md) |
| 完整实验记录与规划 | [实验记录目录](notebook/experiments/) · [GPU 实验计划](notebook/gpu-experiment-plan.md) |

## 论文与其他 AI 专题

- **论文阅读**：[完整导航](notebook/readme.md#论文精读) · [Transformer](notebook/papers/attention-is-all-you-need.md) · [DeepSeek-V3](notebook/papers/deepseek-v3.md) · [DeepSeek-R1](notebook/papers/deepseek-r1-technical-report.md) · [FlashAttention](notebook/papers/flashattention.md) · [PPO](notebook/papers/ppo.md)。
- **数学与算法理解**：[分布式训练收敛](notebook/distributed-training-convergence-theory.md) · [MoE 数学推导](notebook/moe-architecture-mathematical-derivation.md) · [GRPO 算法综合](notebook/grpo-training-algorithm-unified-synthesis.md)。
- **扩展方向**：[多模态](notebook/fundamentals/multimodal-vlm-deep-dive.md) · [扩散模型](notebook/fundamentals/diffusion-models-deep-dive.md) · [RAG](notebook/fundamentals/rag-systems-deep-dive.md) · [可解释性](notebook/fundamentals/mechanistic-interpretability-deep-dive.md) · [模型合并](notebook/fundamentals/model-merging-theory.md) · [持续学习](notebook/fundamentals/continual-learning-deep-dive.md)。

## 工具、技能与执行材料

[tools/readme.md](tools/readme.md) 是工具入口，说明用途、复用标准及验证范围。当前可从 [GPU 监控](tools/gpu_monitor.py)、[训练配置配方](tools/training_recipe.py) 和 [Infra 速查](tools/ai_infra_cheatsheet.py) 浏览；其余历史工具仍需逐项验收。一次性、纯实验性和低价值重复脚本归入 [tools/legacy/](tools/legacy/readme.md)，原工具索引保存在 [历史目录](tools/legacy/catalog.md)。

已有 Agent 技能保存在 [.claude/skills/](.claude/skills/)，包括 [GPU 实验](.claude/skills/gpu-experiment/SKILL.md)、[GRPO 排错](.claude/skills/grpo-debug/SKILL.md)、[推理性能](.claude/skills/inference-perf/SKILL.md)、[分布式训练导航](.claude/skills/distributed-nav/SKILL.md)、[权重同步](.claude/skills/weight-sync-nav/SKILL.md) 等。它们记录历史工作方法，使用时需适配当前执行环境。

| 执行材料 | 当前内容 |
|---|---|
| [experiments/](experiments/) | 3 个 GRPO 训练启动脚本 |
| [scripts/](scripts/) | 实验执行、验证与编排脚本 |
| [configs/](configs/readme.md) | 4 份 RTX 4090 DeepSpeed 训练配置 |
| [csrc/](csrc/) | CUDA/C++ 实现与相关验证 |
| [results/](results/readme.md) | 按训练、分布式、算子、推理和模型架构归档的实验结果 |

这些材料支撑实际运行；实验结论及复现说明应从 [notebook/experiments/](notebook/experiments/) 查阅。

## 规划、进展与工作日志

- [WORK.md](WORK.md)：当前用户需求、工作重点、等待条件和下一步。
- [roadmap.md](notebook/roadmap.md)：学习与实践路线；[knowledge-map.md](notebook/knowledge-map.md)：知识关系及整体导航。
- [diary/readme.md](diary/readme.md)：按日期浏览实际工作、发现、失败与成果入口。

## 文件组织结构

点击路径浏览文件，点击总览查看对应板块的内容导航与使用说明。

| 路径 | 职责 | 总览与索引 |
|---|---|---|
| [README.md](README.md) | 项目总览、对外呈现与完整导航 | 本页 |
| [AGENTS.md](AGENTS.md) | Agent 内核 | 工作原则、行为方式与执行边界 |
| [WORK.md](WORK.md) | 当前工作上下文 | 当前需求、进展与下一步 |
| [notebook/](notebook/) | 技术学习笔记 | [笔记总览](notebook/readme.md) |
| [notebook/fundamentals/](notebook/fundamentals/) | 基础知识 | [技术主题导航](#技术知识导航) |
| [notebook/frameworks/](notebook/frameworks/) | 框架架构与源码理解 | [框架索引](notebook/frameworks/readme.md) |
| [notebook/experiments/](notebook/experiments/) | 实验记录与结论 | [实验导航](#实验与实战经验) |
| [notebook/papers/](notebook/papers/) | 论文阅读 | [论文索引](notebook/readme.md#论文精读) |
| [notebook/roadmap.md](notebook/roadmap.md) | 学习与实践规划 | 学习路线与探索方向 |
| [notebook/knowledge-map.md](notebook/knowledge-map.md) | 知识整理与总览 | 知识关系与整体导航 |
| [tools/](tools/) | 可复用工具 | [工具总览](tools/readme.md) |
| [tools/legacy/](tools/legacy/) | 一次性、纯实验性或低价值重复脚本 | [归档说明](tools/legacy/readme.md) · [历史目录](tools/legacy/catalog.md) |
| [open-source/](open-source/) | 按目标开源项目保存贡献材料 | [各项目贡献索引](open-source/README.md#按项目浏览) |
| [open-source/SOP.md](open-source/SOP.md) | 贡献工作方法 | 候选发现、分析与推进流程 |
| [diary/](diary/) | 按日期记录实际工作 | [日志总览](diary/readme.md) |
| [experiments/](experiments/) | 历史训练启动脚本 | [与实验记录的关系](#尚待整合的执行材料) |
| [scripts/](scripts/) | 执行、验证与编排脚本 | [执行材料说明](#工具技能与执行材料) |
| [configs/](configs/) | 历史训练配置 | [配置说明](configs/readme.md) |
| [csrc/](csrc/) | CUDA/C++ 实现与验证 | [执行材料说明](#工具技能与执行材料) |
| [results/](results/) | 实验原始结果 | [结果归档说明](results/readme.md) |
| [.claude/skills/](.claude/skills/) | 已有 Agent 技能 | [技能入口](#工具技能与执行材料) |

主要板块按职责组织，同一项成果维护一个主要位置，通过链接串起学习、实践与贡献。原根目录的 26 个实验结果 JSON 已按主题归入 results/；其他已有专题与环境文件保留原位。

### 尚待整合的执行材料

根目录 `experiments/` 保存实验启动脚本，`notebook/experiments/` 保存实验记录，二者分别回答“怎么运行”和“运行后学到了什么”。执行材料还分散在 scripts、tools、configs 等位置，后续应围绕具体实验建立关联，再确定是否统一收纳。

`configs/` 当前只保存训练实验配置，文件有保留价值，但无需长期作为独立一级板块；建议随对应实验一起组织。由于脚本和笔记已有调用，暂保留路径，待实验材料结构确定后统一迁移。

根目录尚无 `projects/`，目前不预建；未来有需要独立维护的自营项目时，再决定放在本仓库还是独立仓库。框架源码学习统一使用 `notebook/frameworks/`。

## GitHub

[Jackie2049/perpetual-learning](https://github.com/Jackie2049/perpetual-learning)
