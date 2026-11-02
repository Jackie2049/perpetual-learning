# perpetual-learning

## 项目说明

这是一个持续进行**学习、实践、生产**的**自进化智能引擎**，不间断地学习技术知识、积累实战经验、发掘可探索的方向、产出有价值的成果。领域为AI技术、优先关注Infra。用户驱动时，优先承接并细化用户的需求和想法；真实运行中，持续积累可复用的实战经验。上述的**技术掘金、技术秘书、经验沉淀**等任务有机结合，持续增强引擎能力，最终在技术视野、想法探索、实战经验、业界影响力等方面为用户保驾护航并做出有价值的贡献

打开 [智能引擎工作台 ↗](https://jackie2049.github.io/perpetual-learning/) ，在这个工作台网页上，可以浏览近期进展、搜索相关信息、回看工作日志与待定事项。

管理入口：[Agent 规则](AGENTS.md) · [执行接续](CONTEXT.md) · [技术秘书队列](queue_tech-secretary.md) · [技术掘金队列](queue_tech-miner.md)

## 运作机制

【核心工作】本项目的 Agent 线程应始终围绕3个核心工作展开，即**技术秘书**、**技术掘金**、**经验沉淀**：

- **技术秘书（第一优先级）**：承接用户提出的需求或想法，通过尽职调查、调试与测试、价值与风险分析、概念验证、实际执行等动作，闭环用户所提出的具体需求，或者帮助用户将模糊的想法逐步具体化。
- **技术掘金（第二优先级）**：围绕AI 技术（优先关注 Infra），自主寻找值得深入或贡献的方向，持续开展学习、实践、探索。产出包括但不限于笔记、工具、开源贡献、自建项目、撰写论文等。
- **经验沉淀（伴随前两者开展）**：在技术掘金和技术秘书的真实运行中，持续积累可复用经验，并转化为知识、工具和技能，增强后续工作的能力。

【触发机制】本项目的Agent线程通过**定时任务**、**用户提问**进行触发驱动：

- **定时任务触发时**
  - 优先推进技术秘书中的高优事项
  - 其次视情况自主进行技术掘金（比如技术秘书的高优事项进展良好或阻塞）
  - 如有高价值经验则通过合理形式沉淀到仓库里的相关位置
- **用户主动提问时**
  - 根据任务复杂度确定立即闭环执行还是加入到技术秘书事项列表进行跟踪推进
  - 如有高价值经验则通过合理形式沉淀到仓库里的相关位置

【日志机制】本项目的 Agent 线程应及时记录**工作日志**、**近期上下文**

- **工作日志**：每轮工作结束后（定时任务触发、用户主动提问分别算一轮工作），及时记录 [工作日志](diary/readme.md)。
- **近期上下文**：为了方便快速回看上下文，以及跨会话、跨机器之间的Agent协作，持续维护[近期上下文](CONTEXT.md)



## Get Started

在项目目录使用任意 Agent Harness 拉起定时任务，以下提示词仅供参考：

```bash
【定时任务】perpetual-learning这个项目，按照规则，持续运作
```



## 文件组织

这里是整个项目的统一入口：点击路径浏览文件，点击总览查看对应板块的内容导航与使用说明。


| 路径                           | 职责       | 总览与索引                                |
| ---------------------------- | -------- | ------------------------------------ |
| [README.md](README.md)       | 项目总览     | 本页                                   |
| [AGENTS.md](AGENTS.md)       | Agent 内核 | 工作原则、行为方式、执行边界                       |
| [CONTEXT.md](CONTEXT.md)           | 近期上下文    | 当前的需求、进展、下一步规划                       |
| [notebook/](notebook/)       | 笔记       | [笔记总览](notebook/readme.md)           |
| [tools/](tools/)             | 工具       | [工具总览](tools/readme.md)              |
| [experiments/](experiments/) | 实验       | [实验总览](experiments/readme.md)        |
| [skills/](skills/)           | 技能       | [技能总览](skills/readme.md)             |
| [open-source/](open-source/) | 开源贡献     | [分项目索引](open-source/README.md#按项目浏览) |
| [diary/](diary/)             | 日志       | [日志总览](diary/readme.md)              |
| [web/](web/)                 | 工作台网页    | [使用与维护](web/readme.md)               |


[experiments/](experiments/readme.md) 保存实验执行材料，[notebook/experiments/](notebook/experiments/) 保存分析与经验，通过链接关联复现过程和结论。

## 知识导航

AI 技术是探索范围，Infra 是优先方向。下面提供常用主题的直接入口；更多笔记见 [技术学习总览](notebook/readme.md) 和 [基础知识目录](notebook/fundamentals/)。


| 方向        | 直接阅读                                                                                                                                                                                                                                                      |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 分布式训练     | [并行策略](notebook/fundamentals/distributed-training.md) · [ZeRO](notebook/fundamentals/zero-optimizer.md) · [FSDP](notebook/fundamentals/fsdp.md) · [混合精度](notebook/fundamentals/mixed-precision.md)                                                        |
| GPU 编程与优化 | [CUDA](notebook/fundamentals/cuda-programming-basics.md) · [Triton](notebook/fundamentals/triton-programming.md) · [FlashAttention](notebook/fundamentals/flash-attention.md) · [Profiling](notebook/fundamentals/gpu-profiling.md)                       |
| 通信与集群     | [NCCL](notebook/fundamentals/nccl.md) · [RDMA](notebook/fundamentals/rdma-networking.md) · [通信计算重叠](notebook/fundamentals/comm-compute-overlap.md) · [GPU 调度](notebook/fundamentals/k8s-gpu-scheduling.md)                                                |
| 推理引擎与服务   | [KV Cache](notebook/fundamentals/kv-cache.md) · [Continuous Batching](notebook/fundamentals/continuous-batching.md) · [P/D 分离](notebook/fundamentals/disaggregated-serving.md) · [可观测性](notebook/fundamentals/inference-observability.md)                 |
| 模型架构与推理优化 | [MoE](notebook/fundamentals/moe.md) · [MLA](notebook/fundamentals/mla.md) · [量化](notebook/fundamentals/quantization.md) · [投机解码](notebook/fundamentals/speculative-decoding.md)                                                                           |
| RL 与后训练   | [RLHF/GRPO Infra](notebook/fundamentals/rlhf-training-infra.md) · [优势估计](notebook/rl-advantage-estimators-mathematical-derivation.md) · [Policy Loss](notebook/rl-policy-loss-functions-mathematical-derivation.md) · [DPO](notebook/fundamentals/dpo.md) |
| 微调与编译     | [LoRA/PEFT](notebook/fundamentals/lora-peft.md) · [torch.compile](notebook/fundamentals/torch-compile.md) · [CUDA Graphs](notebook/fundamentals/cuda-graphs.md)                                                                                           |




## 上机实战

实验记录保存问题、环境、方法、结果与局限；经验来自实际执行和反馈，进一步沉淀为排错方法、工具与技能。执行材料见 [实验总览](experiments/readme.md)，可以从中进入源码、验证脚本、配置、环境和原始结果。


| 兴趣点           | 入口                                                                                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU 基础性能与测量方法 | [A16 基准](notebook/experiments/gpu-a16-benchmark.md) · [A16 Profiling](notebook/experiments/gpu-profiling-a16.md) · [RTX 4090 基准](notebook/experiments/rtx4090-benchmark.md)              |
| Kernel 与推理优化  | [Triton Kernels](notebook/experiments/rtx4090-triton-kernels.md) · [KV Cache 压缩](notebook/experiments/rtx4090-kv-cache-compress.md)                                                      |
| 训练与多卡实践       | [LoRA 微调](notebook/experiments/rtx4090-lora-finetune.md) · [多卡 DDP](notebook/experiments/rtx4090-multigpu-ddp.md) · [RLHF/GRPO](notebook/experiments/rtx4090-rlhf-grpo.md)               |
| 排错与跨框架经验      | [GRPO 排错](notebook/grpo-training-debugging-guide.md) · [跨框架问题分类](notebook/cross-framework-bug-taxonomy.md) · [CUDA Stream 生命周期问题](notebook/cuda-stream-use-after-free-pattern-family.md) |
| 完整实验记录与规划     | [实验记录目录](notebook/experiments/) · [GPU 实验计划](notebook/gpu-experiment-plan.md)                                                                                                            |




## 主流框架 & 开源贡献

学习与贡献是同一条探索路径的不同产物：源码理解进入 `notebook/frameworks/`；围绕具体改进的候选、方案、补丁、Issue/PR 与评论草稿进入 `open-source/<framework>/`。贡献材料链接学习和实验证据，便于从理解继续推进到实践。


| 框架          | 架构与源码学习                                                                                                    | 贡献候选、方案与跟进                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| vLLM        | [源码索引](notebook/frameworks/readme.md#vllm) · [V1 架构全景](notebook/frameworks/vllm-v1-architecture-map.md)    | [vLLM 贡献](open-source/vllm/readme.md)                                             |
| SGLang      | [源码索引](notebook/frameworks/readme.md#sglang) · [架构分析](notebook/frameworks/sglang-architecture.md)          | [SGLang 贡献](open-source/sglang/readme.md)                                         |
| verl        | [源码索引](notebook/frameworks/readme.md#verl) · [RL Infra](notebook/frameworks/verl-rl-infra-reading.md)      | [verl 贡献](open-source/verl/readme.md) · [Tracker](open-source/verl/tracker.md)    |
| AReaL       | [源码索引](notebook/frameworks/readme.md#areal)                                                                | [AReaL 贡献](open-source/areal/readme.md) · [Tracker](open-source/areal/tracker.md) |
| DeepSpeed   | [源码索引](notebook/frameworks/readme.md#deepspeed)                                                            | [DeepSpeed 贡献](open-source/deepspeed/readme.md)                                   |
| Megatron    | [源码索引](notebook/frameworks/readme.md#megatron) · [张量并行](notebook/frameworks/megatron-tp-source-reading.md) | —                                                                                 |
| PyTorch     | [源码索引](notebook/frameworks/readme.md#pytorch)                                                              | [PyTorch 贡献](open-source/pytorch/readme.md)                                       |
| rLLM        | [源码索引](notebook/frameworks/readme.md#rllm)                                                                 | [rLLM 贡献](open-source/rllm/readme.md)                                             |
| TRL         | [源码索引](notebook/frameworks/readme.md#trl)                                                                  | [TRL 贡献](open-source/trl/readme.md)                                               |
| vLLM-Ascend | [源码索引](notebook/frameworks/readme.md#vllm-ascend)                                                          | [Ascend 贡献](open-source/vllm-ascend/readme.md)                                    |
| MoonshotAI  | 相关理解随具体候选链接                                                                                                | [贡献记录](open-source/moonshot/readme.md)                                            |


[其他框架与跨框架专题](notebook/frameworks/readme.md#其他框架与跨框架专题) · [跨项目贡献规划](open-source/README.md#跨项目规划与方法)

## 论文学习 & 专题研究

- **论文阅读**：[完整导航](notebook/readme.md#论文精读) · [Transformer](notebook/papers/attention-is-all-you-need.md) · [DeepSeek-V3](notebook/papers/deepseek-v3.md) · [DeepSeek-R1](notebook/papers/deepseek-r1-technical-report.md) · [FlashAttention](notebook/papers/flashattention.md) · [PPO](notebook/papers/ppo.md)。
- **数学与算法理解**：[分布式训练收敛](notebook/distributed-training-convergence-theory.md) · [MoE 数学推导](notebook/moe-architecture-mathematical-derivation.md) · [GRPO 算法综合](notebook/grpo-training-algorithm-unified-synthesis.md)。
- **扩展方向**：[多模态](notebook/fundamentals/multimodal-vlm-deep-dive.md) · [扩散模型](notebook/fundamentals/diffusion-models-deep-dive.md) · [RAG](notebook/fundamentals/rag-systems-deep-dive.md) · [可解释性](notebook/fundamentals/mechanistic-interpretability-deep-dive.md) · [模型合并](notebook/fundamentals/model-merging-theory.md) · [持续学习](notebook/fundamentals/continual-learning-deep-dive.md)。



## 工具沉淀 & 技能打磨

把实战中可复用的实现沉淀为工具，把解决问题的方法整理为 Agent 技能。

### 工具


| 工具                                       | 用途                  |
| ---------------------------------------- | ------------------- |
| [GPU 监控](tools/gpu_monitor.py)           | 采集 GPU 状态，记录与分析监控日志 |
| [训练配置配方](tools/training_recipe.py)       | 根据输入生成训练配置与启动命令     |
| [Infra 速查](tools/ai_infra_cheatsheet.py) | 按主题查阅常用命令           |


[工具总览与使用说明](tools/readme.md)

### 技能

[技能总览与目录约定](skills/readme.md)


| 场景     | 技能入口                                                                                                                             |
| ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| 实验执行   | [GPU 实验](skills/gpu-experiment/SKILL.md) · [RTX 4090 训练](skills/rtx4090-training/SKILL.md)                                       |
| 训练排错   | [GRPO 排错](skills/grpo-debug/SKILL.md) · [跨框架问题规避](skills/cross-framework-avoidance/SKILL.md)                                     |
| 推理分析   | [性能与显存估算](skills/inference-perf/SKILL.md)                                                                                        |
| 分布式与通信 | [分布式训练](skills/distributed-nav/SKILL.md) · [RDMA 网络](skills/rdma-network-nav/SKILL.md) · [权重同步](skills/weight-sync-nav/SKILL.md) |
| 源码导航   | [vLLM](skills/vllm-v1-nav/SKILL.md) · [SGLang](skills/sglang-nav/SKILL.md) · [Megatron](skills/megatron-nav/SKILL.md)            |
| 专题研究   | [可解释性](skills/interpretability-nav/SKILL.md) · [模型合并](skills/model-merging-nav/SKILL.md)                                         |


