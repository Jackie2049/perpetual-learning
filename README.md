# perpetual-learning

## 项目说明

持续进行学习、实践、生产的自进化智能引擎，不间断地学习技术知识、积累实战经验、发掘可探索的方向、产出有价值的成果。领域为AI技术、优先关注Infra。用户驱动时，优先承接并细化用户的需求和想法；真实运行中，持续积累可复用的实战经验。上述的技术掘金、技术秘书、经验沉淀等任务有机结合，持续增强引擎能力，最终在技术视野、想法探索、实战经验、业界影响力等方面为用户的保驾护航并做出有价值的贡献

## 运作机制

- **技术掘金**：持续自主学习、实践和探索 AI 技术，优先关注 Infra，寻找值得深入和贡献的方向，产出笔记、工具、开源贡献、自营成果或论文。
- **技术秘书**：优先承接用户的需求与模糊想法，通过尽调、综述、价值与风险分析、概念验证和实际执行推进目标。
- **经验沉淀**：从真实运行及其反馈中积累可复用经验，逐步转化为知识、工具和技能，增强后续工作的能力。

运行状态与当前工作见 [WORK.md](WORK.md)。

## 文件组织

这里是整个项目的统一入口：点击路径浏览文件，点击总览查看对应板块的内容导航与使用说明。

| 路径 | 职责 | 总览与索引 |
|---|---|---|
| [README.md](README.md) | 项目总览、对外呈现与完整导航 | 本页 |
| [AGENTS.md](AGENTS.md) | Agent 内核 | 工作原则、行为方式与执行边界 |
| [WORK.md](WORK.md) | 当前工作上下文 | 当前需求、进展与下一步 |
| [notebook/](notebook/) | 技术学习笔记 | [笔记总览](notebook/readme.md) |
| [notebook/fundamentals/](notebook/fundamentals/) | 基础知识 | [知识导航](#知识导航) |
| [notebook/frameworks/](notebook/frameworks/) | 框架架构与源码理解 | [框架索引](notebook/frameworks/readme.md) · [主流框架 & 开源贡献](#主流框架--开源贡献) |
| [notebook/experiments/](notebook/experiments/) | 实验记录与结论 | [上机实战](#上机实战) |
| [notebook/resources/](notebook/resources/) | 学习与调研原始资料 | [资料索引](notebook/resources/readme.md) |
| [notebook/papers/](notebook/papers/) | 论文阅读 | [论文索引](notebook/readme.md#论文精读) · [专题研究](#论文学习--专题研究) |
| [notebook/roadmap.md](notebook/roadmap.md) | 学习与实践规划 | 学习路线与探索方向 |
| [notebook/knowledge-map.md](notebook/knowledge-map.md) | 知识整理与总览 | 知识关系与整体导航 |
| [tools/](tools/) | 可复用工具 | [工具总览](tools/readme.md) |
| [tools/legacy/](tools/legacy/) | 一次性、纯实验性或低价值重复脚本 | [归档说明](tools/legacy/readme.md) · [历史目录](tools/legacy/catalog.md) |
| [open-source/](open-source/) | 按目标开源项目保存贡献材料 | [各项目贡献索引](open-source/README.md#按项目浏览) |
| [open-source/SOP.md](open-source/SOP.md) | 贡献工作方法 | 候选发现、分析与推进流程 |
| [diary/](diary/) | 按日期记录实际工作 | [日志总览](diary/readme.md) |
| [experiments/](experiments/) | 实验源码、脚本、配置、环境与结果 | [实验总览](experiments/readme.md) |
| [experiments/scripts/](experiments/scripts/) | GPU 实验、复现与验证脚本 | [脚本索引](experiments/scripts/readme.md) |
| [experiments/kernels/](experiments/kernels/) | 算子实验源码与基准 | [Fused RMSNorm 实验](notebook/experiments/rtx4090-fused-rmsnorm-cuda.md) |
| [experiments/rtx4090/configs/](experiments/rtx4090/configs/) | RTX 4090 训练配置 | [配置说明](experiments/rtx4090/configs/readme.md) |
| [experiments/environments/](experiments/environments/) | 历史机器的实验环境快照 | [环境说明](experiments/environments/readme.md) |
| [experiments/results/](experiments/results/) | 实验原始结果 | [结果索引](experiments/results/readme.md) |
| [.claude/skills/](.claude/skills/) | 已有 Agent 技能 | [技能入口](#工具沉淀--技能打磨) |

[experiments/](experiments/readme.md) 保存实验执行材料，[notebook/experiments/](notebook/experiments/) 保存分析与经验，通过链接关联复现过程和结论。

## 知识导航

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

## 上机实战

实验记录保存问题、环境、方法、结果与局限；经验来自实际执行和反馈，进一步沉淀为排错方法、工具与技能。执行材料见 [实验总览](experiments/readme.md)，可以从中进入源码、验证脚本、配置、环境和原始结果。

| 兴趣点 | 入口 |
|---|---|
| GPU 基础性能与测量方法 | [A16 基准](notebook/experiments/gpu-a16-benchmark.md) · [A16 Profiling](notebook/experiments/gpu-profiling-a16.md) · [RTX 4090 基准](notebook/experiments/rtx4090-benchmark.md) |
| Kernel 与推理优化 | [Triton Kernels](notebook/experiments/rtx4090-triton-kernels.md) · [KV Cache 压缩](notebook/experiments/rtx4090-kv-cache-compress.md) |
| 训练与多卡实践 | [LoRA 微调](notebook/experiments/rtx4090-lora-finetune.md) · [多卡 DDP](notebook/experiments/rtx4090-multigpu-ddp.md) · [RLHF/GRPO](notebook/experiments/rtx4090-rlhf-grpo.md) |
| 排错与跨框架经验 | [GRPO 排错](notebook/grpo-training-debugging-guide.md) · [跨框架问题分类](notebook/cross-framework-bug-taxonomy.md) · [CUDA Stream 生命周期问题](notebook/cuda-stream-use-after-free-pattern-family.md) |
| 完整实验记录与规划 | [实验记录目录](notebook/experiments/) · [GPU 实验计划](notebook/gpu-experiment-plan.md) |

## 主流框架 & 开源贡献

学习与贡献是同一条探索路径的不同产物：源码理解进入 `notebook/frameworks/`；围绕具体改进的候选、方案、补丁、Issue/PR 与评论草稿进入 `open-source/<framework>/`。贡献材料链接学习和实验证据，便于从理解继续推进到实践。

| 框架 | 架构与源码学习 | 贡献候选、方案与跟进 |
|---|---|---|
| vLLM | [源码索引](notebook/frameworks/readme.md#vllm) · [V1 架构全景](notebook/frameworks/vllm-v1-architecture-map.md) | [vLLM 贡献](open-source/vllm/readme.md) |
| SGLang | [源码索引](notebook/frameworks/readme.md#sglang) · [架构分析](notebook/frameworks/sglang-architecture.md) | [SGLang 贡献](open-source/sglang/readme.md) |
| verl | [源码索引](notebook/frameworks/readme.md#verl) · [RL Infra](notebook/frameworks/verl-rl-infra-reading.md) | [verl 贡献](open-source/verl/readme.md) · [Tracker](open-source/verl/tracker.md) |
| DeepSpeed | [源码索引](notebook/frameworks/readme.md#deepspeed) | [DeepSpeed 贡献](open-source/deepspeed/readme.md) |
| Megatron | [源码索引](notebook/frameworks/readme.md#megatron) · [张量并行](notebook/frameworks/megatron-tp-source-reading.md) | — |
| PyTorch | [源码索引](notebook/frameworks/readme.md#pytorch) | [PyTorch 贡献](open-source/pytorch/readme.md) |
| rLLM | [源码索引](notebook/frameworks/readme.md#rllm) | [rLLM 贡献](open-source/rllm/readme.md) |
| TRL | [源码索引](notebook/frameworks/readme.md#trl) | [TRL 贡献](open-source/trl/readme.md) |
| vLLM-Ascend | [源码索引](notebook/frameworks/readme.md#vllm-ascend) | [Ascend 贡献](open-source/vllm-ascend/readme.md) |
| MoonshotAI | 相关理解随具体候选链接 | [贡献记录](open-source/moonshot/readme.md) |

[其他框架与跨框架专题](notebook/frameworks/readme.md#其他框架与跨框架专题) · [跨项目贡献规划](open-source/README.md#跨项目规划与方法)

## 论文学习 & 专题研究

- **论文阅读**：[完整导航](notebook/readme.md#论文精读) · [Transformer](notebook/papers/attention-is-all-you-need.md) · [DeepSeek-V3](notebook/papers/deepseek-v3.md) · [DeepSeek-R1](notebook/papers/deepseek-r1-technical-report.md) · [FlashAttention](notebook/papers/flashattention.md) · [PPO](notebook/papers/ppo.md)。
- **数学与算法理解**：[分布式训练收敛](notebook/distributed-training-convergence-theory.md) · [MoE 数学推导](notebook/moe-architecture-mathematical-derivation.md) · [GRPO 算法综合](notebook/grpo-training-algorithm-unified-synthesis.md)。
- **扩展方向**：[多模态](notebook/fundamentals/multimodal-vlm-deep-dive.md) · [扩散模型](notebook/fundamentals/diffusion-models-deep-dive.md) · [RAG](notebook/fundamentals/rag-systems-deep-dive.md) · [可解释性](notebook/fundamentals/mechanistic-interpretability-deep-dive.md) · [模型合并](notebook/fundamentals/model-merging-theory.md) · [持续学习](notebook/fundamentals/continual-learning-deep-dive.md)。

## 工具沉淀 & 技能打磨

[tools/readme.md](tools/readme.md) 是工具入口，说明用途、复用标准及验证范围。当前可从 [GPU 监控](tools/gpu_monitor.py)、[训练配置配方](tools/training_recipe.py) 和 [Infra 速查](tools/ai_infra_cheatsheet.py) 浏览，使用方法与验证范围见工具总览。一次性、纯实验性和低价值重复脚本归入 [tools/legacy/](tools/legacy/readme.md)，原工具索引保存在 [历史目录](tools/legacy/catalog.md)。

已有 Agent 技能保存在 [.claude/skills/](.claude/skills/)，包括 [GPU 实验](.claude/skills/gpu-experiment/SKILL.md)、[GRPO 排错](.claude/skills/grpo-debug/SKILL.md)、[推理性能](.claude/skills/inference-perf/SKILL.md)、[分布式训练导航](.claude/skills/distributed-nav/SKILL.md)、[权重同步](.claude/skills/weight-sync-nav/SKILL.md) 等。它们记录历史工作方法，使用时需适配当前执行环境。

## 规划 & 进展

- [WORK.md](WORK.md)：当前用户需求、工作重点、等待条件和下一步。
- [roadmap.md](notebook/roadmap.md)：学习与实践路线；[knowledge-map.md](notebook/knowledge-map.md)：知识关系及整体导航。
- [diary/readme.md](diary/readme.md)：按日期浏览实际工作、发现、失败与成果入口。

## GitHub

[Jackie2049/perpetual-learning](https://github.com/Jackie2049/perpetual-learning)
