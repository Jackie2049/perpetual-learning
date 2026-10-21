# 技术学习笔记

这里保存 perpetual-learning 的技术学习、源码阅读与实验理解。当前工作上下文见 [WORK.md](../WORK.md)，完整项目介绍见 [README.md](../README.md)。

## 板块职责

| 位置 | 用途 |
|---|---|
| [fundamentals/](fundamentals/) | 基础知识、原理与方法 |
| [frameworks/](frameworks/readme.md) | 框架架构、源码阅读与完整文件索引 |
| [experiments/](experiments/) | 实验记录：问题、环境、方法、结果、结论和经验 |
| [roadmap.md](roadmap.md) | 技术学习与实践规划 |
| [knowledge-map.md](knowledge-map.md) | 知识整理、关联与整体导航 |
| [resources/](resources/readme.md) | 学习与调研的原始资料留档 |
| [papers/](papers/) | 延续已有论文阅读笔记 |

其余已有专题继续保留。新增内容优先归入上述已有位置，避免同一主题重复建立多个入口。

## 记录约定

- 笔记写清学习目标、核心理解、来源与适用版本；区分资料结论、自己的推断和已验证的事实。
- 源码笔记链接实际代码位置与版本，贡献方案及进展放在 [open-source/](../open-source/README.md)。
- 实验笔记说明问题、环境、命令、结果、结论及局限，并链接脚本和必要结果文件；真实实践的经验先跟随实验保存。
- [experiments/](../experiments/readme.md) 保存实验源码、脚本、配置、环境和原始结果；这里保存分析、结论与经验，两者互相链接。
- 可复用工具见 [tools/](../tools/readme.md)，历史实验性脚本见 [tools/legacy/](../tools/legacy/readme.md)。

## 已有内容导航

以下目录从原项目 README 迁入，保留已有知识入口。摘要中的版本、性能数字和进度属于历史记录，使用前需核对原文及适用条件；本次整理没有重新验证这些结论。

- [verl 新版 CT tokenizer 回退](frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#11-新版自动选择与-tokenizer-能力回退) — #8311原172项CPU对照；模型架构不认证ChatML协议，成功回退不认证多轮边界。

## 规划与路线

- [学习路线图](roadmap.md) — Phase 1-5 学习计划，从分布式训练到 vLLM 开源贡献
- [知识地图](knowledge-map.md) — AI Infra 8 大领域技能树，标注掌握程度
- [GPU 实验计划](gpu-experiment-plan.md) — A16 GPU 上的实验清单和优先级

## 分布式训练

- [分布式训练总览](fundamentals/distributed-training.md) — DP/TP/PP/SP 演进逻辑，通信原语图解
- [ZeRO 优化器](fundamentals/zero-optimizer.md) — ZeRO 1/2/3 原理、显存公式、适用场景
- [FSDP 全分片数据并行](fundamentals/fsdp.md) — FlattenParameter、Sharding 策略、Prefetching、verl FSDP 实践
- [混合精度训练](fundamentals/mixed-precision.md) — FP16/BF16/FP8、Loss Scaling、Tensor Core 利用
- [梯度检查点](fundamentals/gradient-checkpointing.md) — 激活重计算策略，用计算换显存
- [NCCL 通信库](fundamentals/nccl.md) — Ring/Tree AllReduce、调优参数、网络拓扑
- [RDMA 高性能网络](fundamentals/rdma-networking.md) — InfiniBand/RoCE/iWARP、GPUDirect RDMA、NCCL 调优、实战排错
- [CUDA 编程基础](fundamentals/cuda-programming-basics.md) — Grid/Block/Thread、内存层次、Coalesced Access、Occupancy、Roofline Model
- [通信与计算重叠](fundamentals/comm-compute-overlap.md) — DDP Bucket Overlap、1F1B、ZeRO-3 Prefetch
- [MoE 混合专家架构](fundamentals/moe.md) — 稀疏激活、Top-K 路由、负载均衡、Expert Parallelism、DeepSeek-V3
- [序列并行](fundamentals/sequence-parallelism.md) — DeepSpeed Ulysses、Ring Attention、USP 统一框架
- [Ray 分布式框架](fundamentals/ray-framework.md) — Actor/Task 模型、Placement Group、RLHF 调度
- [分布式 Checkpoint](fundamentals/checkpoint.md) — 分片保存、异步 IO、mmap 加速加载
- [DataLoader 优化](fundamentals/dataloader.md) — Prefetch、mmap、WebDataset 流式加载

## GPU 编程与优化

- [CUDA 编程基础](fundamentals/cuda-basics.md) — 线程层次、内存模型、流、Tensor Core
- [Triton 编程](fundamentals/triton-programming.md) — OpenAI Triton：kernel、autotune、GEMM、FlashAttention
- [torch.compile 与 Inductor](fundamentals/torch-compile.md) — Dynamo 图捕获、Triton 代码生成、CUDA Graphs 集成
- [FlashAttention](fundamentals/flash-attention.md) — 1/2/3 深度解析，tiling、online softmax、IO 复杂度
- [Attention Kernel 优化](fundamentals/attention-kernel-optimization.md) — FlashAttention-3/4、FlashInfer、FlashMLA 对比与选型
- [RoPE 旋转位置编码](fundamentals/rope.md) — 数学推导、四种实现、五种长度扩展方法、推理优化
- [KV Cache 深度解析](fundamentals/kv-cache.md) — 显存估算、PagedAttention、GQA/MQA、Prefix Caching、量化
- [KV Cache 量化](fundamentals/kv-cache-quantization.md) — FP8/INT8/INT4/2-bit 量化对比、KIVI/KVQuant、vLL8 实践、组合优化
- [Multi-head Latent Attention](fundamentals/mla.md) — DeepSeek-V2 MLA 低秩 KV 压缩、解耦 RoPE、矩阵吸收、93.3% KV Cache 减少
- [Prefix Caching 深度对比](fundamentals/prefix-caching.md) — vLLM Hash vs SGLang Radix vs verl Group、收益量化
- [GPU 显存优化](fundamentals/gpu-memory-optimization.md) — 训练显存估算公式、OOM 排查、ZeRO、碎片化
- [GPU 内存分配器](fundamentals/gpu-memory-allocator.md) — PyTorch Caching Allocator、vLLM Paged Memory、碎片化分析
- [CUDA Graphs](fundamentals/cuda-graphs.md) — 推理场景 kernel launch 开销优化，录制重放机制
- [GPU 性能分析](fundamentals/gpu-profiling.md) — 四层方法：nvidia-smi → nsys → ncu → ncu detail
- [LLM 推理服务框架](fundamentals/inference-serving.md) — vLLM/SGLang/TGI/Triton 对比，K8s 生产部署，负载均衡策略
- [SGLang RadixAttention](fundamentals/sglang-radix-attention.md) — 基数树前缀缓存、节点分裂机制、缓存感知调度、与 vLLM Hash-based 对比
- [推理服务监控与可观测性](fundamentals/inference-observability.md) — vLLM 30+ Prometheus 指标、SLO 定义、告警规则、DCGM GPU 监控、瓶颈定位

## RL 训练基础设施

- [Bayes@N与pass@k验证指标](frameworks/verl-validation-bayes-passk.md) — 固定题集后验矩、至少一次成功事件、PR #8303数值helper限定检查及旧公式纠正
- [RLHF/GRPO 训练基础设施](fundamentals/rlhf-training-infra.md) — PPO 四模型架构、GRPO 无 Critic、verl/OpenRLHF 框架、Prefix Sharing
- [DPO 直接偏好优化](fundamentals/dpo.md) — 跳过 RM 和 RL 的对齐方法、IPO/KTO/SimPO 变体
- [LoRA 与参数高效微调](fundamentals/lora-peft.md) — 低秩分解原理、QLoRA、多 LoRA 服务、RLHF 中的应用

## 推理与部署

- [Speculative Decoding](fundamentals/speculative-decoding.md) — Draft-Verify、Medusa、Eagle 投机采样
- [Continuous Batching](fundamentals/continuous-batching.md) — Static→Continuous 演进、Chunked Prefill、调度策略
- [Tokenizer 与推理管线](fundamentals/tokenizer.md) — BPE、词表大小影响、压缩率与推理成本、特殊 Tokens
- [推理量化](fundamentals/quantization.md) — GPTQ/AWQ/SmoothQuant/FP8 量化方法对比
- [FP8 量化与训练](fundamentals/fp8-quantization.md) — E4M3/E5M2 格式、缩放策略、Transformer Engine、vLLM FP8 实现
- [模型服务架构](fundamentals/model-serving.md) — API 网关、负载均衡、HPA 自动扩缩容
- [长上下文 LLM 推理服务](fundamentals/long-context-serving.md) — KV Cache 显存压力、Chunked Prefill、Sliding Window、服务策略
- [Chunked Prefill 深度解析](fundamentals/chunked-prefill.md) — vLLM V1 统一调度模型、Chunk Size 三层约束、Partial Prefill 处理、Prefix Caching 组合
- [Disaggregated Serving](fundamentals/disaggregated-serving.md) — Prefill/Decode 分离、KV Transfer、Splitwise/Mooncake 架构
- [LLM 采样方法](fundamentals/sampling-methods.md) — Temperature/Top-K/Top-P/Min-P/Beam Search/重复惩罚

## 集群与基础设施

- [K8s GPU 调度](fundamentals/k8s-gpu-scheduling.md) — Device Plugin、MIG、Gang Scheduling
- [Slurm 集群管理](fundamentals/slurm.md) — sbatch 提交、NCCL 配置、抢占式调度
- [分布式文件系统](fundamentals/distributed-fs.md) — Lustre 条带化、IO 优化、并行读写

## 论文精读

- [Attention is All You Need](papers/attention-is-all-you-need.md) — Transformer 原始论文, QK^T/√d_k 缩放, 多头注意力, 现代架构对比
- [DeepSeek-R1](papers/deepseek-r1-technical-report.md) — GRPO, "Aha Moment", Cold-start SFT → GRPO → Rejection Sampling SFT → DPO
- [InstructGPT](papers/instructgpt.md) — RLHF 三步 pipeline, 1.3B InstructGPT > 175B GPT-3
- [LLaMA 3](papers/llama3.md) — 简单架构+大数据=强模型, 405B 16K H100 训练, DPO 对齐
- [FlashAttention](papers/flashattention.md) — Tiling + online softmax + recomputation, O(N) 内存
- [Chinchilla Scaling Laws](papers/chinchilla.md) — L(N,D) 最优 ~20 tok/param, 推翻 Kaplan "大模型少数据"
- [PPO](papers/ppo.md) — L^{CLIP}, RLHF 核心算法, GRPO 简化版
- [GMPO](papers/gmpo.md) — 完整符号裁剪目标与限定值/梯度对照；旧“论文与实现配置空间不同”结论撤回，非完整训练证明
- [GSPO](papers/gspo.md) — 序列级 importance ratio + 组归一优势, verl 示例 ε=3e-4 极紧窗发现
- [SAPO](papers/sapo.md) — sech² 平滑门替代硬裁剪, 非对称温度 τ_neg>τ_pos, Qwen3-VL 实战
- [GPT-4](papers/gpt4.md) — 可预测扩展, RLHF 无损, RBRMs 安全
- [Mamba](papers/mamba.md) — 选择性 SSM, 线性时间, 2.8B 匹配 GPT-J-6B
- [Top-nσ](papers/top-n-sigma.md) — ACL 2025, logit 空间统计阈值, 温度稳定性 1.00x
- [DeepSeek-V3](papers/deepseek-v3.md) — 671B/37B MoE, 辅助无关负载均衡, FP8 tile-wise, MTP, DualPipe

## 框架源码阅读

[按框架浏览完整索引](frameworks/readme.md) · [按项目浏览贡献材料](../open-source/README.md)

- [vLLM V1 架构](frameworks/vllm-architecture.md) — V1 整体架构、Scheduler/Executor 分离、混合推理
- [vLLM KV Transfer Connector](frameworks/vllm-kv-transfer-connector.md) — NIXL 连接器架构、P/D 分离、ZMQ 握手、TP 聚合
- [vLLM V1 Scheduler 源码阅读](frameworks/vllm-v1-scheduler-reading.md) — 调度循环、抢占策略、KV Cache 管理、Prefix Caching 集成
- [vLLM 投机解码源码阅读](frameworks/vllm-spec-decode-reading.md) — Draft-Verify、Eagle/Medusa/Ngram proposer、Rejection Sampling
- [vLLM Prefix Caching V1](frameworks/vllm-prefix-caching-v1.md) — Hash chain、block 生命周期、四种 attention 缓存策略
- [vLLM 开源贡献计划](../open-source/vllm/vllm-contribution-plan.md) — 11 个 good first issue 筛选和分析
- [vLLM 贡献准备](../open-source/vllm/vllm-contribution-prep.md) — FlashInfer 后端分析、开发环境搭建
- [Megatron-LM 张量并行](frameworks/megatron-tp-reading.md) — ColumnParallelLinear/RowParallelLinear 源码
- [Megatron-LM TP 源码深度阅读](frameworks/megatron-tp-source-reading.md) — 通信原语 autograd、TP 数据流、权重分发、SP 扩展、3D 并行交互
- [Megatron-LM 流水线并行](frameworks/megatron-pp-reading.md) — 1F1B 调度、气泡分析、通信 overlap
- [verl 架构](frameworks/verl-architecture.md) — 三级前缀缓存（系统级+进程级+请求级）
- [vLLM MLA Backend 源码阅读](frameworks/vllm-mla-backend-reading.md) — FlashMLA/FlashInfer/Triton/Cutlass MLA 后端、压缩数据流、KV Cache 格式、Backend 选择
- [verl shared-prefix packing](frameworks/verl-shared-prefix-packing.md) — #8300 的 FSDP 分支可见域、首响应标签、等价条件及 38 组原始 CPU 布局检查
- [verl 量化层审计与 refit](frameworks/verl-quant-layer-audit.md) — #8132三类证据/24项CPU审计单测；与新#8306的格式、存储和保护覆盖对照，不能混用分支证据
- [verl 新版阶段 tracing](frameworks/verl-async-logprob-provenance.md#新版阶段-tracing-与证据关联) — #8315 rank/actor区间、no-op与指标API能力边界；仅源码/AST，真实同步和显存补证仍交M5
- [DeepInfer SamplingParams 校验](frameworks/deepinfer-initial-survey.md#samplingparams-构造校验与测试组合) — #119/#121独立分支、53项原始CPU测试配对和greedy前校验边界
- [DeepInfer CacheConfig 与严格预期失败](frameworks/deepinfer-initial-survey.md#cacheconfig-校验与严格预期失败) — #118修复与#121标记的XFAIL/严格XPASS对照；范围合法不保证实际KV容量
- [DeepInfer decode 性能缓存](frameworks/deepinfer-initial-survey.md#decode-性能曲线缓存的身份与回退) — #126身份键与#128溢出回退、17项原始单rank CPU/Gloo测试；曲线不是KV缓存，非模型/性能验证
- [DeepInfer profiling 服务关闭](frameworks/deepinfer-initial-survey.md#profiling-请求与服务关闭) — #122排队/采集/导出状态及关闭顺序；8组原始等待协程控制，不是HTTP/端口/修复验收
- [DeepInfer Ascend 发现与 NUMA](frameworks/deepinfer-initial-survey.md#ascend-发现与-numa-查询边界) — #127原始12项CPU测试与2组可见编号控制；合成根跳过真实CLI，查询不等于绑核
- [DeepInfer vendor registry 边界](frameworks/deepinfer-initial-survey.md#vendor-registry-与模型运行支持) — #60原始10项合成目录测试，固定vendor列表与platform/build分开；识别Biren不等于模型支持
- [DeepInfer 显存遥测](frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划) — #151单次查询/warning、allocator与全局口径及KV规划的区别；仅源码核对，真机交M5
- [DeepInfer 消息信任边界](frameworks/deepinfer-initial-survey.md#共享内存消息与反序列化信任边界) — #158仍保留pickle；注释/扫描抑制非协议修复，CP本地调用与通用remote分支分开，非漏洞复现
- [DeepInfer vendor wheel 身份](frameworks/deepinfer-initial-survey.md#vendor-wheel-的本次产物与安装身份) — #159本次构建/安装选择与兼容性分开；Shell语法及PEP440排序核对，非原生构建验收
- [DeepInfer GPQA 提取与判分](frameworks/deepinfer-initial-survey.md#gpqa-答案提取与最终判分) — #160原始7项CPU测试成对复现；大小写/路径优先级与默认LLM judge覆盖规则结果的区别
- [DeepInfer 评测题集覆盖](frameworks/deepinfer-initial-survey.md#评测-limit-与题集覆盖) — #169原5项CPU测试与原scorer三组本地JSONL控制；正limit入口不等于历史元数据或全数据覆盖认证
- [DeepInfer 跨语言枚举测试](frameworks/deepinfer-initial-survey.md#enginetype-的源码一致性测试) — #164原测试与8组输入控制；显式十进制文本一致性不等于C++语义或已安装扩展同版
- [DeepInfer 布尔配置与缓存](frameworks/deepinfer-initial-survey.md#布尔环境变量的解析与缓存) — #177原24测试base18过6预期失败/fix全过；宽松未知值、descriptor重读与复合字段须分开
- [DeepInfer sweep 数据与推荐](frameworks/deepinfer-initial-survey.md#sweep-推荐的数据选择与计量口径) — #176原10测试与真实CLI/JSONL控制；最后可用点不代表最新尝试成功或扫描完整，接受率假设非实测
- [DeepInfer 关闭错误与清理](frameworks/deepinfer-initial-survey.md#关闭失败后的独立清理与一次性标志) — #175三个原方法/13原断言隔离对照；Mock资源，closed是不重放标志而非释放完成
- [DeepInfer AIME 思考区评分](frameworks/deepinfer-initial-survey.md#aime-思考区与规则评分) — #178原21测试base11过10预期失败/fix全过；真实rule CLI合成用例，提取修复非模型提分
- [DeepInfer 模型路径与下载来源](frameworks/deepinfer-initial-survey.md#模型路径选择与下载来源) — #182两版各12组原函数路径/导入拦截控制；本地路径、SDK选择与checkpoint完整性分开，未下载
- [DeepInfer lookahead 与 draft 入口](frameworks/deepinfer-initial-survey.md#lookahead-与-draft-长度的不同入口) — #184原表达式/环境20组观察；lookahead与nextn独立，prefill门槛变化非实际draft/推理验收
- [DeepInfer 共享池句柄身份](frameworks/deepinfer-initial-survey.md#共享池句柄的分配身份与视图寿命) — #187原9项真实CPU共享存储/fork测试；代际拒绝旧句柄，不撤销已返回视图
- [DeepInfer 内存快照路径](frameworks/deepinfer-initial-survey.md#内存快照的输出路径与目录创建) — #183原路径/目录12组控制；默认迁移、空串与失败传播，非设备采集验证
- [PrefixGrouper 计算范围纠错](frameworks/verl-prefix-grouper-gap-analysis.md) — 原生 grouped input 也节省前缀逐 token 运算；撤回 attention-only 缺口和非等价微基准收益外推
- [verl PrefixGrouper](frameworks/verl-prefix-grouper.md) — 历史集成；与新 engine 路径的版本边界及复杂度纠错
- [vLLM V1 KV Cache Manager 源码阅读](frameworks/vllm-v1-kv-cache-manager-reading.md) — Block 分配/释放、6 种 Attention Manager、Prefix Caching、Coordinator 架构
- [vLLM V1 Executor 源码阅读](frameworks/vllm-v1-executor-reading.md) — Executor→Worker→ModelRunner→AttentionBackend 完整执行路径、CUDA Graph、20+ Attention Backend
- [vLLM MoE Serving 源码阅读](frameworks/vllm-moe-serving-reading.md) — 模块化 MoE kernel 架构、8+ All-to-All 后端、Router/Experts/Prepare-Finalize 分离
- [verl 源码架构阅读](frameworks/verl-source-reading.md) — HybridFlow RL 训练框架、PPO/GRPO 循环、FSDP+混合推理引擎、Prefix Grouper
- [verl RL Infra 架构源码阅读](frameworks/verl-rl-infra-reading.md) — SyncPPOTrainer、ActorRolloutRefWorker、GRPO/GAE、TransferQueue、vLLM Rollout
- [vLLM 量化推理管线源码阅读](frameworks/vllm-quantization-pipeline-reading.md) — 量化方法注册表、Kernel 选择逻辑、离线/在线量化、QuantKey 系统
- [vLLM 权重加载管线源码阅读](frameworks/vllm-weight-loading-reading.md) — safetensors 到 GPU 全流程、TP/PP/EP 分片、weight_loader 机制
- [vLLM P/D 分离架构源码阅读](frameworks/vllm-pd-disaggregation-reading.md) — KV Connector 抽象、NIXL 实现、异步传输、异构 TP、部署配置
- [vLLM CUDA Graph 源码阅读](frameworks/vllm-cuda-graph-reading.md) — FULL/PIECEWISE 模式、捕获回放流程、Attention 兼容、Breakable Graph
- [vLLM V1 架构全景图](frameworks/vllm-v1-architecture-map.md) — 请求生命周期、四层架构、模块依赖关系、已学清单
- [vLLM Sampling Pipeline 源码阅读](frameworks/vllm-sampling-pipeline-reading.md) — Logits 到 Token 完整管线、Top-K/P 实现、Penalty、CUDA Graph 兼容
- [vLLM Data Parallel Serving 源码阅读](frameworks/vllm-data-parallel-reading.md) — DP vs TP 扩展、MoE EP、Wave 协调、DBO、Elastic EP
- [vLLM V1 可观测性与指标体系 源码阅读](frameworks/vllm-observability-reading.md) — Stats 采集、Prometheus 指标、Logging、告警建议
- [vLLM LoRA Serving 源码阅读](frameworks/vllm-lora-serving-reading.md) — Punica Multi-LoRA、批量 LoRA kernel、MoE+LoRA、权重管理
- [vLLM Multi-Modal Processing 源码阅读](frameworks/vllm-multimodal-reading.md) — MM 数据流、EncoderRunner、Hash-based Caching、Vision Encoder 集成
- [vLLM Structured Output 源码阅读](frameworks/vllm-structured-output-reading.md) — JSON/Grammar/Regex Bitmask 过滤、FSM 驱动、XGrammar 后端
- [vLLM AsyncLLM Engine Frontend 源码阅读](frameworks/vllm-async-engine-frontend-reading.md) — 双进程架构、output_handler、InputProcessor/OutputProcessor、ZMQ 通信
- [vLLM V1 Engine Core 源码阅读](frameworks/vllm-engine-core-reading.md) — step() 核心循环、KV Cache 初始化、Busy Loop、ZMQ 双线程、DP 扩展
- [GRPO 实战指南](frameworks/grpo-practical-guide.md) — verl 框架 GRPO 训练完整流程、数据格式、配置参数、奖励函数
- [SGLang 架构深度分析](frameworks/sglang-architecture.md) — RadixAttention 基数树、DSL 前端、FlashInfer 集成、vs vLLM 对比
- [DeepInfer 平民版（魔改适配非旗舰硬件）可行性](frameworks/deepinfer-civilian-port.md) — DeepInfer 对 Blackwell/Ascend950 的依赖堵点、替换方案、4090/A100/H100/910C 分层可行性；2026-10-21 已补 4090 真机验证（csrc `_C` 编译导入成功、架构检查与 kernel 包堵点实测确认）
- [分布式训练排错指南](frameworks/troubleshooting-guide.md) — NCCL 超时、OOM、梯度异常等常见问题

## GPU 实验记录

- [A16 基准测试](experiments/gpu-a16-benchmark.md) — GEMM 14.64 TFLOPS、HBM 76 GB/s、混合精度 3.9x 加速
- [vLLM GPT-2 推理](experiments/vllm-a16-gpt2-benchmark.md) — vLLM 0.6.4+cu118, Batch=16 达 1875 tok/s, Flash Attention 后端
- [GPU Profiling 深度实验](experiments/gpu-profiling-a16.md) — GEMM/带宽/Attention/Kernel Launch 五类操作 profiling，A16 约为 A100 的 1/10 性能
- [RTX 4090 全面 Benchmark](experiments/rtx4090-benchmark.md) — SDPA 162 TFLOPS, Decode 529K tok/s, CUDA Graph 1.02x, KV Cache 分析
- [RTX 4090 MoE Layer](experiments/rtx4090-moe-layer.md) — MoE vs Dense, Python 循环 5.8-26.3x 开销, aux_loss ≈ 1.0
- [RTX 4090 Training Techniques](experiments/rtx4090-training-techniques.md) — AdamW>>Adam 4.13 vs 6.57, BF16 省 29%, cosine warmup
- [RTX 4090 PPO Training](experiments/rtx4090-ppo-training.md) — PPO 比 Vanilla PG 好 24%, KL β=0.01 最优
- [RTX 4090 KV Cache Compression](experiments/rtx4090-kv-cache-compress.md) — INT8 cos=0.9999 安全, SW-4K 省 97%
- [RTX 4090 Ring Attention](experiments/rtx4090-ring-attention.md) — Online softmax 精确 err<6e-7, 通信量与 P 无关
- [RTX 4090 Mini LLM Training](experiments/rtx4090-mini-train.md) — Pretrain loss 4.92→2.46, SFT loss 1.68→1.11
- [RTX 4090 DPO Training](experiments/rtx4090-dpo-train.md) — DPO preference accuracy 99.7%, β=0.3 最优
- [RTX 4090 LoRA Fine-tuning](experiments/rtx4090-lora-finetune.md) — LoRA r=8 仅 7.13% 参数, merge 零推理开销
- [RTX 4090 Multi-GPU DDP](experiments/rtx4090-multigpu-ddp.md) — 4 GPU 3.23x (80.7% 效率), PCIe AllReduce 4.9 GB/s
- [RTX 4090 RLHF/GRPO](experiments/rtx4090-rlhf-grpo.md) — GRPO 比 PPO 稳定, group size n=2-4 最优
- [RTX 4090 Triton Kernels](experiments/rtx4090-triton-kernels.md) — RMSNorm 1.8x, memory BW 945 GB/s = 93.7% peak
- [RTX 4090 Top-nσ Sampling](experiments/rtx4090-top-n-sigma.md) — 温度稳定性 1.00x vs top-p 19,665x
- [RTX 4090 DeepSeek-V3 MoE](experiments/rtx4090-deepseek-v3-moe.md) — Bias-based 零性能损失, FP8 tile-wise cos=1.0
