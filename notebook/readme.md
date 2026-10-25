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
- [verl 多轮配置的消费路径](frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#15-多轮配置的消费路径与告警契约) — #7662两键在活跃AgentLoop无消费点；旧strict只告警，源码核对不认证模板／模型行为，等待维护者方向。

## 规划与路线

- [学习路线图](roadmap.md) — Phase 1-5 学习计划，从分布式训练到 vLLM 开源贡献
- [知识地图](knowledge-map.md) — AI Infra 8 大领域技能树，标注掌握程度
- [GPU 实验计划](gpu-experiment-plan.md) — A16 GPU 上的实验清单和优先级

## 分布式训练

- [分布式训练总览](fundamentals/distributed-training.md) — DP/TP/PP/SP 演进逻辑，通信原语图解
- [ZeRO 优化器](fundamentals/zero-optimizer.md) — ZeRO 1/2/3 原理、显存公式、适用场景
- [FSDP 全分片数据并行](fundamentals/fsdp.md) — FlattenParameter、Sharding 策略、Prefetching、verl FSDP 实践
- [混合精度训练](fundamentals/mixed-precision.md) — FP16/BF16/FP8、Loss Scaling、Tensor Core 利用
  - [AMP跳步与学习率计数](fundamentals/mixed-precision.md#amp-跳步与学习率计数) — 真实CPU GradScaler区分梯度检查、优化器调用与scheduler；多优化器可部分更新，非GPU/融合优化器或框架训练验收
- [AdamW缺失与零梯度](fundamentals/adamw-optimizer-deep-dive.md#缺失梯度与零梯度) — 真实CPU optimizer/autograd：None跳过参数与本参数步数，零梯度仍推进动量/衰减；动态冻结须核旧grad，非分布式或性能验收
- [梯度检查点](fundamentals/gradient-checkpointing.md#重计算的随机数与可变状态) — 真实CPU重算对照：默认元数据检查不认证梯度一致，Torch RNG不覆盖外部随机源，early-stop可能跳过末尾副作用；非显存/性能或FSDP验收
  - [重算叶子与共享存储](fundamentals/gradient-checkpointing.md#重算叶子与共享存储) — verl#8338原两条清理语句/4组CPU所有权对照；不截断其他alias，原5项CUDA回归和显存回收未验证
- [NCCL 通信库](fundamentals/nccl.md) — Ring/Tree AllReduce、调优参数、网络拓扑
- [RDMA 高性能网络](fundamentals/rdma-networking.md) — InfiniBand/RoCE/iWARP、GPUDirect RDMA、NCCL 调优、实战排错
- [CUDA 编程基础](fundamentals/cuda-programming-basics.md) — Grid/Block/Thread、内存层次、Coalesced Access、Occupancy、Roofline Model
- [通信与计算重叠](fundamentals/comm-compute-overlap.md) — DDP Bucket Overlap、1F1B、ZeRO-3 Prefetch
- [MoE 混合专家架构](fundamentals/moe.md) — 稀疏激活、Top-K 路由、负载均衡、Expert Parallelism、DeepSeek-V3
- [序列并行](fundamentals/sequence-parallelism.md) — DeepSpeed Ulysses、Ring Attention、USP 统一框架
- [Ray 分布式框架](fundamentals/ray-framework.md) — Actor/Task 模型、Placement Group、RLHF 调度
- [分布式 Checkpoint](fundamentals/checkpoint.md#状态字典与独立快照) — 分片保存与状态快照；真实CPU参数/buffer/SGD对照：detach及CPU上的cpu不隔离存储，深拷贝或已完成序列化保留旧值，非异步/设备恢复验证
- [DataLoader 优化](fundamentals/dataloader.md) — Prefetch、mmap、WebDataset；新增[分布式补齐与丢弃](fundamentals/dataloader.md#分布式采样的补齐与丢弃)，真实CPU两层drop_last/均值/epoch对照，等长不等于无重复或覆盖完整

## GPU 编程与优化

- [CUDA 编程基础](fundamentals/cuda-basics.md) — 线程层次、内存模型、流、Tensor Core
- [Triton 编程](fundamentals/triton-programming.md) — OpenAI Triton：kernel、autotune、GEMM、FlashAttention
- [torch.compile 与 Inductor](fundamentals/torch-compile.md) — Dynamo 图捕获、Triton 代码生成、CUDA Graphs 集成
- [FlashAttention](fundamentals/flash-attention.md) — 1/2/3原理；[缓存后缀因果对齐](fundamentals/flash-attention.md#缓存后缀的因果对齐)的真实CPU SDPA输出/局部梯度对照，单token无mask条件不能外推多token或padding容量，非GPU性能验收
- [Attention Kernel 优化](fundamentals/attention-kernel-optimization.md) — FlashAttention-3/4、FlashInfer、FlashMLA 对比与选型
- [RoPE 旋转位置编码](fundamentals/rope.md) — 数学推导、四种实现、五种长度扩展方法、推理优化
- [KV Cache 深度解析](fundamentals/kv-cache.md) — 显存估算、PagedAttention、GQA/MQA、Prefix Caching、量化
- [KV Cache 量化](fundamentals/kv-cache-quantization.md) — FP8/INT8/INT4/2-bit 量化对比、KIVI/KVQuant、vLL8 实践、组合优化
- [Multi-head Latent Attention](fundamentals/mla.md) — DeepSeek-V2 MLA 低秩 KV 压缩、解耦 RoPE、矩阵吸收、93.3% KV Cache 减少
- [Prefix Caching 深度对比](fundamentals/prefix-caching.md) — vLLM Hash vs SGLang Radix vs verl Group、收益量化
- [权重版本与外部缓存](frameworks/prefix-cache-failure-modes.md#权重版本与外部缓存) — verl#8323命名隔离、删除、安装证明分开；三后端接线/原测试范围，未执行CPU套件或设备。
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
- [verl 激活卸载的反向入口](frameworks/verl-architecture.md#激活卸载的反向入口与重载顺序) — #8277原五CPU测试体通过；区分内部描述、commit顺序和detached输出，不认证异步传输／真实模型。
- [verl 异步奖励补齐](frameworks/verl-reward-integration-reading.md#异步奖励的补齐边界) — #8101原构造/消费片段的CPU几何对照；mask总数正确仍可读错response，恢复长度也可能增加应有惩罚，非完整worker/TQ验收
- [verl DeepEP 环境边界](frameworks/verl-architecture.md#deepep-的包解析与动态库加载) — #8354 Python包/系统NVSHMEM/dispatcher分层；4份TOML与17份YAML静态一致，不等于镜像、import或通信验收
- [verl 模型缓存与临时目录](frameworks/verl-architecture.md#只读模型缓存与测试临时目录) — #8343 原路径赋值与真实 pytest 两用例控制；隔离输出不扩大配额，CI 锁配置不等于外部消费者验收
- [verl 续跑长度上下限](frameworks/verl-fully-async-policy-reading.md#续跑的累计长度上下限) — #8356原四测试协程隔离执行，base三断言失败/head四通过；累计扣减下限，非完整pytest或真实后端停止验收
- [verl 文件系统权重同步](verl-v1-checkpoint-engines-deep-reading.md#文件系统权重同步的发布与清理) — #8355原safetensors/复制/CPU接收本地验证；临时文件非完整恢复状态，索引最后复制非事务，撤回旧4090后端和固定内存结论
- [verl LoRA 主机暂存](verl-v1-checkpoint-engines-deep-reading.md#lora-分桶接收的主机存储所有权) — #8352原暂存语句CPU存储控制；每桶一个块仍须累计整个adapter，逐tensor复制及锁页/loader另有条件，非IPC或加速验收
- [vLLM MLA Backend 源码阅读](frameworks/vllm-mla-backend-reading.md) — FlashMLA/FlashInfer/Triton/Cutlass MLA 后端、压缩数据流、KV Cache 格式、Backend 选择
- [verl shared-prefix packing](frameworks/verl-shared-prefix-packing.md) — #8300 的 FSDP 分支可见域、首响应标签、等价条件及 38 组原始 CPU 布局检查
- [verl UID 分组均衡](frameworks/verl-grpo-source-reading.md#连续-uid-与等样本组的前提) — 连续UID、固定每组样本数与DP等行切分的复合契约；原3个CPU测试体及22组控制，非标准trainer故障复现
- [verl V1源prompt与可变行](frameworks/verl-grpo-source-reading.md#v1-源-prompt-单位与可变轨迹行) — #8201按UID计优化批、逐单元补行并传worker边界；同批非同rank，仅12项原纯helper检查，非TQ/训练验收
- [verl 拼批形状与对象引用](frameworks/verl-grpo-data-flow-reading.md#拼批中的形状与引用) — 原collator/1原测试体与7组CPU DataLoader检查；一维对象边界不等于深拷贝或字段校验
- [verl Rollout 完整性](frameworks/verl-grpo-data-flow-reading.md#rollout-完整性与会话计数) — #6641原7项replay测试候选通过/旧版4项拒绝检查预期失败；多输出行不等于多会话，数量护栏非超时、取消或服务恢复
- [verl PF PPO重采样身份](frameworks/verl-grpo-data-flow-reading.md#pf-ppo-重采样与原-key-回写) — #7957原四测试三版本对照，内部行一致不等原key对齐；旧mask切片可与新行长度不同，非真实队列或训练验收
- [verl 二维logprob分块](frameworks/verl-6572-full-determinism-source-reading.md#二维低精度-logprob-的分块预算) — #8351原17测试体绑定cpu通过；单块预算非反向总保存量或CUDA峰值，复用原脚本，不新增训练
- [verl 奖励限流与取消](frameworks/verl-reward-integration-reading.md#进程配额与取消边界) — 原16协程测试体/2取消对照；类属性不是跨节点配额，超容量预扣不自动退款，timeout不包排队
- [verl Prime 评分结果与失败范围](frameworks/verl-reward-integration-reading.md#prime-进程池的结果协议与失败范围) — #8217旧原4例/真实spawn；[新#8358同题对照](frameworks/verl-reward-integration-reading.md#新编号与已有修复)仅注释/docstring差异、计算AST相同，测试范围不同，不重复验证或当新候选
- [verl 沙箱运行器派发](frameworks/verl-reward-integration-reading.md#沙箱运行器注册与执行边界) — #8345原14用例通过/16未选；包注册不等于代码隔离，传入timeout不等于强制终止，非entry-point/服务或安全验收
- [verl 参数转换与缓冲区精度](frameworks/verl-fsdp2-source-deep-reading.md#参数转换与缓冲区精度) — #8189 原 6 CPU 测试体与 4 组控制，区分保留初值、FP32 标签和训练状态迁移；未运行真实 RoPE/model/FSDP
- [verl 数学奖励的分层与精度](frameworks/verl-reward-integration-reading.md#数学奖励的分层判分与精度边界) — #8347原64参数例候选全过/祖先25预期失败；7组控制区分后备接受、整数字面量、科学计数法及方程提取，非数据集或训练验收
- [verl 掩码统计与梯度](frameworks/verl-grpo-core-algos-reading.md#掩码统计与梯度的数值边界) — 原3参数例/20组CPU对照；mean过滤无效NaN不代表方差梯度或白化全部槽位有限，非训练故障定性
- [verl 单样本组回退](frameworks/verl-grpo-core-algos-reading.md#单样本与同分组的不同回退) — 原4等价性例/12组双实现控制；singleton保留近似原奖励，撤回排错指南n=1必退化与n≥4/8硬规则，非训练收益证明
- [verl 同分小数奖励的舍入残差](frameworks/verl-grpo-core-algos-reading.md#同分小数奖励的舍入残差) — #8349原16参数例base3过13预期失败/head全过；大组FP32残差可放大，singleton与非同分保留，无设备/训练收益结论
- [verl ReMax 贪心基线](frameworks/verl-remax-icepop-source-reading.md#v1-贪心会话与-sampled-最大值) — #8310额外贪心会话非sampled最大值；核心CPU对照区分advantages/returns，撤回零方差和4090最优推荐，非解码/训练验收
- [verl KL 前向与反向](frameworks/verl-grpo-core-algos-reading.md#kl-前向值与反向规则) — k3+数值clamp非梯度clamp；6原例/14组CPU对照，有限差分与straight-through反向须区分
- [verl checkpoint 保留与恢复](frameworks/verl-checkpoint-management-reading.md#保留目录与完整恢复的边界) — 原5清理测试通过；上限1保存前不删旧，登记成功不认证整轮可恢复，tracker直接写入非原子替换
- [verl SFT epoch 与加载器恢复](frameworks/verl-checkpoint-management-reading.md#sft-epoch-边界与加载器状态) — #7402原方法、2个原生迭代状态及8组真实CPU数据流控制；最后一批与结束确认不同，legacy回退非边界修复
- [verl LoRA-only键完整性](frameworks/verl-checkpoint-management-reading.md#lora-only-的键完整性与加载副作用) — #8350原分支八组真实CPU控制；缺adapter被拒绝不等模型回滚，也不认证base权重身份，非原17项或FSDP恢复验收
- [verl 对象存储下载与checkpoint提交](frameworks/verl-checkpoint-management-reading.md#对象存储下载与-checkpoint-提交) — #6850原五下载测试体通过只认证memory缓存；#7641的_COMMITTED是待验收协议，非主线已有S3恢复。
- [verl 复制分片存储隔离](frameworks/verl-checkpoint-management-reading.md#复制分片的存储隔离) — #7503原3测试体base2过1预期失败/head全过；共享view非重叠，输出占用下降不认证合并峰值或重复key根因
- [verl 局部更新指标聚合](frameworks/verl-6790-async-trainer-reading.md#局部更新的指标聚合) — 原19项pytest及5组CPU观察；日志加权/求和/末值依key而定，比率求和不是整轮平均，正式计时路径另算
- [verl 会话水位与未消费组上限](frameworks/verl-6790-async-trainer-reading.md#会话水位与未消费组上限) — #8203准入单位与采样/训练释放分开；组额度非逐样本陈旧度或TQ总内存上限，仅源码/语法及4个原schema参数例
- [verl 学习率步数与恢复](frameworks/verl-grpo-training-loop-internals-reading.md#学习率步数与恢复条件) — 原构造器/真实CPU optimizer四轨迹、两恢复与一步覆写；推进后LR不等于刚使用LR，加载状态不重建原调度闭包
- [verl 融合输出头与softcap](frameworks/verl-fused-output-softcap.md) — #7991拒绝不支持的通用路径，非kernel实现；原torch输出头36组语义分离对照，保留数值超限，原24配置测试未跑
- [verl 前向与logprob一致性](frameworks/verl-6572-full-determinism-source-reading.md#actor-前向与-logprob-公式的一致性) — #8026原4数学测试体通过；同logits的公式、批形状前向和训练复现分层，撤回P9完整确定性/固定吞吐推论，非GPU验收
- [verl 量化层审计与 refit](frameworks/verl-quant-layer-audit.md) — #8132三类证据/24项CPU审计单测；与新#8306的格式、存储和保护覆盖对照，不能混用分支证据
  - [VeOmni计算与通信边界](frameworks/verl-quant-layer-audit.md#veomni-低精度计算与通信边界) — #8342拟议端到端路线与已合并#8330分开；torchao0.18 rowwise/AllGather限制，通信微基准收益不等于actor更新加速
  - [新版检查与指标哨兵](frameworks/verl-quant-layer-audit.md#新版运行期检查与指标哨兵) — #8327关系/默认warn与硬拒绝分开；原纯Python20例及5控制通过，缓存disabled与缺指标不当健康证明
- [verl ModelOpt模块忽略](frameworks/verl-quant-layer-audit.md#modelopt-模块忽略与子量化器) — #8329模块名与子量化器名分层；两版原helper名称控制通过，未执行真实量化，权重加载相等不认证actor forward精度
- [verl 现成logprob对照与权重自检](frameworks/verl-async-logprob-provenance.md#现成-logprob-对照与权重自检) — #8252已有维护者回应；原metric/退出片段CPU控制区分平均及末轮门槛与内容恢复保证，非模型/同步实测，不另起检查器
- [verl 新版阶段 tracing](frameworks/verl-async-logprob-provenance.md#新版阶段-tracing-与证据关联) — #8315原12项日志测试三版本对照，前后head均过但生产lane不同；Mock调用不认证后端显示/真实同步，M5原条目已同步版本
  - [Gauge元数据与训练步存储](frameworks/verl-async-logprob-provenance.md#gauge-元数据与训练步存储) — #8357原logger到固定SDK原push路由四分支控制；step标签非global_step记录，sink边界保留不等于后端已存储，非服务/设备验收
- [verl 单 worker NIXL 拓扑](frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑) — #8316原函数20组控制及后续新增1原测试隔离对照；worker非replica，测试体通过非原生导入/传输验收
- [verl 组初始化等待超时](frameworks/verl-async-logprob-provenance.md#组初始化的等待超时与任务退出) — #7045当前控制端head、12组原参数检查；等待上限不等于全链路截止、RPC取消或资源回收，真机条件补M5原条目
- [verl Resume扩展字段](frameworks/verl-async-logprob-provenance.md#resume-返回值与扩展字段合并) — #8341新版三推测计数按attempt求和；仅新增2原测试体旧/新对照，原统计函数显示率相同仍可丢计数，非后端/AgentLoop验收；旧字段测试保留历史
- [verl Bypass字段与worker配置](frameworks/verl-async-logprob-provenance.md#bypass-字段替换与-worker-的-loss-配置) — #8348新head删除冲突/镜像/PP与动态CP guard及旧测试；两版五组OmegaConf输入确认保证收窄、接受范围扩大，旧19例不作新版认证，非worker/训练验收
- [verl 分桶接收失败与退出](frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md#分桶接收失败与退出) — #7366原8项CPU测试两次通过；真实进程ZMQ/SHM，不是设备/vLLM验收。逐ACK预算、阻塞poll、取消与回滚须分开
- [DeepInfer SamplingParams 校验](frameworks/deepinfer-initial-survey.md#samplingparams-构造校验与测试组合) — #119/#121原53项配对；[#154种子校验](frameworks/deepinfer-initial-survey.md#种子归一化前的负值校验)原5项/16组控制，负数拒绝与uint64回绕分开
- [DeepInfer EPLB 副本分配与精度](frameworks/deepinfer-initial-survey.md#eplb-相同热点负载下的副本分配) — #191并列热点及[#212长历史收益](frameworks/deepinfer-initial-survey.md#长历史的公共基线与峰值下降量)；三版本原32项CPU pytest对照，公共基线吞掉0.5收益，非全局最优/设备或吞吐验收
- [DeepInfer CacheConfig 与严格预期失败](frameworks/deepinfer-initial-survey.md#cacheconfig-校验与严格预期失败) — #118修复与#121标记的XFAIL/严格XPASS对照；范围合法不保证实际KV容量
- [DeepInfer decode 性能缓存](frameworks/deepinfer-initial-survey.md#decode-性能曲线缓存的身份与回退) — #126身份键与#128溢出回退、17项原始单rank CPU/Gloo测试；曲线不是KV缓存，非模型/性能验证
- [DeepInfer profiling 服务关闭](frameworks/deepinfer-initial-survey.md#profiling-请求与服务关闭) — #122排队/采集/导出状态及关闭顺序；8组原始等待协程控制，不是HTTP/端口/修复验收
- [DeepInfer Ascend 发现与 NUMA](frameworks/deepinfer-initial-survey.md#ascend-发现与-numa-查询边界) — #127原始12项CPU测试与2组可见编号控制；合成根跳过真实CLI，查询不等于绑核
- [DeepInfer vendor registry 边界](frameworks/deepinfer-initial-survey.md#vendor-registry-与模型运行支持) — #60原始10项合成目录测试，固定vendor列表与platform/build分开；识别Biren不等于模型支持
- [DeepInfer 显存遥测](frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划) — #151单次查询/warning、allocator与全局口径及KV规划的区别；仅源码核对，真机交M5
- [DeepInfer 消息信任边界](frameworks/deepinfer-initial-survey.md#共享内存消息与反序列化信任边界) — #158仍保留pickle；注释/扫描抑制非协议修复，CP本地调用与通用remote分支分开，非漏洞复现
- [DeepInfer vendor wheel 身份](frameworks/deepinfer-initial-survey.md#vendor-wheel-的本次产物与安装身份) — #159本次构建/安装选择与兼容性分开；Shell语法及PEP440排序核对，非原生构建验收
- [DeepInfer checkpoint 布局与顺序](frameworks/deepinfer-initial-survey.md#checkpoint-布局预检与-gate-up-顺序) — #153原55项真实CPU对照，另4组gate/up装载；形状不认证语义顺序、header预检不等于回滚或全模型加载
- [DeepInfer GPQA 提取与判分](frameworks/deepinfer-initial-survey.md#gpqa-答案提取与最终判分) — #160原7项与[#189强调格式](frameworks/deepinfer-initial-survey.md#强调格式与全文优先级)原8项CPU对照；宽松匹配、全文优先级与默认LLM judge不等于模型提分
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
- [DeepInfer 重算与 prompt logprob](frameworks/deepinfer-initial-survey.md#抢占重算与-prompt-logprob-的身份) — #188重算历史非原始prompt；仅源码/语法，原2参数化需匹配_C与设备runtime，已交M5参考
- [DeepInfer PD 队列与资源](frameworks/deepinfer-initial-survey.md#pd-队列交接与资源释放) — #195移出请求、MTP槽归还、KV/state释放分开；3项入队检查不认证拷贝完成，stub语法不等于原生binding可用
- [DeepInfer host cache 命中与预算](frameworks/deepinfer-initial-survey.md#host-cache-的可复用边界) — #144 KV页命中仍需对应SWA state和replay余量；size_bytes不含state池/staging，3原CUDA测试非真实SWA/binding验收
- [DeepInfer Pass@k 分母](frameworks/deepinfer-initial-survey.md#passk-的不齐样本与题目分母) — #207原16项pytest，样本不足时用min(k,n)，score与Pass@1分母不同；外层覆盖/判分仅源码
- [DeepInfer golden token 入场](frameworks/deepinfer-initial-survey.md#golden-token-入场校验与-prompt-占位符) — #209输出词表校验早于Sequence/多模态处理；原30项CPU隔离测试，非完整引擎或Engram验证
- [DeepInfer 图像区间与缓存前缀](frameworks/deepinfer-initial-survey.md#图像展开区间与缓存前缀) — #208原12测试，span按展开后坐标，整图覆盖才剥离binary；计数拒绝与正常字典改写分开，非设备验证
- [DeepInfer Sage FP8分层装载](frameworks/deepinfer-initial-survey.md#sage-fp8-分层装载与独立引擎) — #210主机专家池/GPU LRU与新引擎；仅源码/默认预算，不外推V4.1或4090支持，条件补原M5入口
- [DeepInfer 聊天服务的记账边界](frameworks/deepinfer-initial-survey.md#聊天服务的亲和路由与完成口径) — #147首问亲和不等于会话身份；finished/abort发送与客户端交付分开，仅源码和原纯函数8项控制，未启动服务
- [DeepInfer 抢占结果与执行代次](frameworks/deepinfer-initial-survey.md#抢占在途结果的执行代次与结束交接) — #211区分旧执行和当前结果，终止输出与finalize交接分开；仅源码/语法，未跑原11测试或模型
- [PrefixGrouper 计算范围纠错](frameworks/verl-prefix-grouper-gap-analysis.md) — 原生 grouped input 也节省前缀逐 token 运算；撤回 attention-only 缺口和非等价微基准收益外推
- [verl PrefixGrouper](frameworks/verl-prefix-grouper.md) — 历史集成；与新 engine 路径的版本边界及复杂度纠错
- [vLLM V1 KV Cache Manager 源码阅读](frameworks/vllm-v1-kv-cache-manager-reading.md) — Block 分配/释放、6 种 Attention Manager、Prefix Caching、Coordinator 架构
- [vLLM V1 Executor 源码阅读](frameworks/vllm-v1-executor-reading.md) — Executor→Worker→ModelRunner→AttentionBackend 完整执行路径、CUDA Graph、20+ Attention Backend
- [vLLM MoE Serving 源码阅读](frameworks/vllm-moe-serving-reading.md) — 模块化 MoE kernel 架构、8+ All-to-All 后端、Router/Experts/Prepare-Finalize 分离
- [verl 源码架构阅读](frameworks/verl-source-reading.md) — HybridFlow RL 训练框架、PPO/GRPO 循环、FSDP+混合推理引擎、Prefix Grouper
- [verl RL Infra 架构源码阅读](frameworks/verl-rl-infra-reading.md) — SyncPPOTrainer、ActorRolloutRefWorker、GRPO/GAE、TransferQueue、vLLM Rollout
- [verl TransferQueue 数据面与启动资源](frameworks/verl-transferqueue-worker-lifecycle-deep-reading.md) — #8340按nnodes填默认单元数；3原配置测试/4组generated配对，不认证Ray启动。数据字段与权重同步分开，旧零拷贝/性能推导撤回
- [verl 教师生成与蒸馏目标](frameworks/verl-veomni-opd-source-reading.md#教师生成与蒸馏目标) — #8335整批来源选择、生成/软标签评分分开；教师prefix不是纯学生on-policy，16文件仅语法与源码核对
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
- [UniRL 调研（腾讯统一多模态 RL）](frameworks/unirl-survey.md) — 统一训练抽象、稀疏SDE、SFT游标、packed文本双轴与[按域奖励](frameworks/unirl-survey.md#按域奖励的零分与失败标志)；原CPU路由/统计区分零分、失败和非适用NaN，非服务或训练验收
  - [路由策略与基准入口](frameworks/unirl-survey.md#路由策略与基准入口) — #529新反馈与当前调用链：同步基准不经过pool，Agentic siblings与普通turn不同；亲和、缓存命中和吞吐证据分开，仅源码学习
  - [AR种子与请求规划](frameworks/unirl-survey.md#ar-种子配置与逐样本请求) — #474已合并不等于#507完成；原生Sample/完整原解析器六组CPU入口控制，SGLang有确定性前提、n=1不减组fanout，非采样或复现验收
  - [IPC共识门与物化边界](frameworks/unirl-survey.md#原生-ipc-的共识门与物化边界) — #494区分source/chunk门；raw项可含逐expert DTensor重分布，边界覆盖须核真实producer，14次静态检查非通信/死锁验收
  - [监控奖励与目标隔离](frameworks/unirl-survey.md#监控奖励的目标隔离与失败契约) — #419已由社区接手；原解析/Part CPU控制表明小权重可经标准化成近单位优势，零权重仅适用于weighted_sum，失败契约不随之消失，非训练实测
  - [音频奖励与后续反证](frameworks/unirl-survey.md#音频奖励配方的评估协议与后续反证) — #475正文与社区同head晚期退化分开；原YAML/八模块静态核对，音频policy权重不自动改变KL，不是训练或质量复测
  - [评估尾批与计分成员](frameworks/unirl-survey.md#评估尾批的补齐与计分成员) — #464原Sample树/补齐及聚合片段CPU控制；pad仍生成评分但不计均值，数量校验不认证顺序，helper能力不扩大PE输入契约
  - [激活卸载与包装顺序](frameworks/unirl-survey.md#激活卸载的保存对象与包装顺序) — #626原回调/真实CPU autograd八组；checkpoint改变offload保存对象，dtype保持不代表layout或性能相同，非FSDP/EP/设备验收
  - [视频VAE分片与状态协商](frameworks/unirl-survey.md#视频-vae-分片的状态协商与复本) — #469仍draft且依赖fork的H3 SP；静态AST确认内层拼接保持，公开hook/slicing包装不同，汇总后仍全rank持有视频，非解码/通信或性能验收
  - [离线DPO布局与零步控制](frameworks/unirl-survey.md#离线-dpo-的成对布局与零步控制) — #476原数学片段CPU值/梯度检查；log2及准确率0不能单独认证语义配对/参考冻结，非模型或CUDA确定性验收
  - [多参考图像的双轴](frameworks/unirl-survey.md#多参考图像的样本行与附件轴) — #478原容器CPU控制：3行5图、选行/元数据保持、稠密槽需等附件数；保留行别名，不认证模型/数据/训练路径
  - [可选提示与模式要求](frameworks/unirl-survey.md#可选提示与模式必需条件) — #497已有维护者反对按现状合入；原请求/三版选择方法六输入CPU对照，合法无文本不等于所有模式需prompt，非ImageBind模型回归
  - [原生配置与训练计时](frameworks/unirl-survey.md#原生配置比较与训练计时) — #588公开五行复算/固定prompt哈希匹配；完整训练须含重算，近似耗时不等于精度/学习等价，非性能复测
  - [内部Pipe请求身份](frameworks/unirl-survey.md#内部-pipe-的版本字段与请求身份) — #568合并范围与#567硬件问题分开；原runtime八组真实spawn/Pipe控制，信封合法不等于命令执行或跨版本兼容，非完整引擎验收
  - [Judge回复与失败状态](frameworks/unirl-survey.md#judge-回复解析与失败状态) — 原方法七组本地Response控制；关键词非语义、协议外文本不重试、请求失败可零分且上层仍标成功，非真实服务或训练验收
  - [离线embedding身份与发布](frameworks/unirl-survey.md#离线-embedding-的身份与索引发布) — 原store真实CPU文件往返/rename失败恢复；coverage只查key，resume字符串非权重内容身份，不是编码或共享存储验收
  - [Replay分块与反向](frameworks/unirl-survey.md#replay-分块与反向更新边界) — #465已合并；stage小forward与算法及时backward分开，尾块按C/S、std非均值，纯元素损失CPU控制不认证真实replay/通信
  - [窗口后的压缩积分](frameworks/unirl-survey.md#sde-窗口后的压缩积分尾段) — #545原纯模块720静态日程/64小张量CPU控制；保留窗口出口，尾点数非尾段调用数，非参考harness/Flow SDE或训练收益验收
  - [Token预算与update成员](frameworks/unirl-survey.md#token-预算与-update-成员) — 原CPU planner保留update成员；预算不拒绝超长单样本，无长度会回退数量规划，非显存保证
  - [EMA衰减与时机](frameworks/unirl-survey.md#ema-衰减方向与更新时机) — decay乘旧影子，零值复制live；原CPU mirror/回调/上下文控制，非optimizer或训练验收
  - [原始block权重与冻结外围](frameworks/unirl-survey.md#原始-block-权重训练与冻结外围) — #593参数冻结不切输入梯度；原optimizer/checkpoint helper的CPU数学控制，非H3训练，作者8×H100首次反向仍OOM
  - [全词表熵与有效token汇总](frameworks/unirl-survey.md#全词表条件熵与有效-token-汇总) — #563条件熵非单次负logp，sum/count而非均值的均值；20组原chunk局部CPU梯度对照，非完整replay/训练
  - [有界动态采样与游标](frameworks/unirl-survey.md#有界动态采样与数据游标) — #562批数预算包含首批，失败前已消费数据；保存读取游标非随机生成复现，仅源码/语法，不搬用旧GPU结果
  - [Replay锚点与采样校正](frameworks/unirl-survey.md#replay-锚点与-rollout-采样校正) — #557两个比率不交换clipping顺序；原片段36组CPU值/梯度对照，非完整训练或收益结论
  - [回合续跑与行为版本](frameworks/unirl-survey.md#回合边界续跑与行为版本) — #558保存观察后暂停，逐turn保留行为logp；窗口训练批与optimizer版本分开，排空后仍须等消费完再释放buffer，仅源码/语法
  - [多教师周期与恢复身份](frameworks/unirl-survey.md#多教师周期与恢复身份) — #624完整域周期后更新，rollout数非update数；周期路由JSON、教师内容hash、数据快进与生成RNG分层，仅源码/语法，作者双A100结果非本机验收
  - [Tokenizer实例修补与批量生成](frameworks/unirl-survey.md#tokenizer-实例修补与批量生成边界) — #544签名筛选非revision白名单，行数guard非逐字段/真实生成验收；本机仅两版语法与AST对照，作者20例单列
  - [样本观测的行上限与数据搬运](frameworks/unirl-survey.md#样本观测的行上限与数据搬运) — #597原CPU Sample/本地sink六组控制；每Part上限非全局条数或字节上限，谱系对齐与best-effort文件输出分开
  - [行级奖励派发与分组身份](frameworks/unirl-survey.md#行级奖励派发与分组身份) — #517原类型5组CPU控制；祖先内容保留但scorer组ID改变，奖励DP可放宽不等于rollout/训练或远程服务验收
  - [异步计时窗口与性能比较](frameworks/unirl-survey.md#异步计时窗口与性能比较) — #539把必要权重发布纳入step time，collect非完整生成，主机phase非GPU分解；8模块编译/3标准库控制，无训练或吞吐验收
  - [空响应的反向与更新](frameworks/unirl-survey.md#空响应的反向通信与零梯度更新) — 原零损失分支CPU图检查；[新增ratio guard](frameworks/unirl-survey.md#纯补齐反向与-ratio-诊断)六组原方法控制，通信反向与统计资格分开，全空诊断0不是策略一致性实测，非FSDP验收
  - [冻结教师与状态恢复](frameworks/unirl-survey.md#冻结教师路由与调用方状态恢复) — #519逐层路由与全部梯度开关分别恢复，no_grad仍由调用方负责；恢复动作失败非原子回滚，仅源码/语法，非PEFT实测
  - [教师指纹与失败传播](frameworks/unirl-survey.md#教师指纹与控制器失败传播) — #515后续已澄清/#575已合并；ready异常、内容不同与读取卡住分开，原hash函数6组CPU控制，非Ray/Gloo或训练验收
  - [离线回放与结果隔离](frameworks/unirl-survey.md#离线回放的到达契约与结果隔离) — #487 observed与dependency到达分开；两类上游错误接受在作者fork拒绝，完整原stdlib模块对照，非模拟器或性能验收
  - [常驻与流式规划账本](frameworks/unirl-survey.md#常驻与流式-block-的规划账本) — #486无驻留上限就保守计全部streamed；原planner180候选CPU算术对照，可行性/最大卸载目标非执行器或性能验收
  - [Prompt组边界与前向分块](frameworks/unirl-survey.md#prompt-组边界与前向分块) — #485允许对齐子组跨forward，原helper16,400有界布局对照；校验不修调度，固定FBS开关parity非更换FBS后的轨迹一致
  - [Smoke结果与训练验收](frameworks/unirl-survey.md#smoke-结果格式与训练验收) — #576原空循环CLI核对总样本、可选字段、同名文件保护；统一JSON不是训练或性能验收
  - [损失单位与微批权重](frameworks/unirl-survey.md#损失单位与微批次权重) — #622接口迁移，token/seq目标与None/零计数不同；四组原函数CPU前向及局部梯度检查，非FSDP验收
  - [AROPD采样梯度与KL](frameworks/unirl-survey.md#aropd-的采样梯度与-kl-目标) — #534 raw score公式的固定前缀参考；teacher log-prob目标不能省去entropy项，非原算法/模型验收
  - [关键帧条件与功能就绪](frameworks/unirl-survey.md#关键帧条件的数据身份与功能就绪) — #584携带fp32条件行并只返回target velocity；producer/条件时间仍待接，源码/静态检查不等于FL2VA可启用
  - [设备资源与通信后端](frameworks/unirl-survey.md#设备资源与通信后端的分层边界) — #623已关闭未合并；资源预约/可见性/设备编号/默认通信组分开，仅两版三模块compile，不认证GPU或NPU支持
  - [TempFlow分支与逐行时间](frameworks/unirl-survey.md#tempflow-分支共享与逐行时间索引) — #552共享x_T不等于缓存前缀；[#589引用保留后续](frameworks/unirl-survey.md#逐行步号与潜变量引用保留)三版原方法/原容器CPU控制，只读索引、保留潜变量句柄，非远程传输或模型验收
  - [异步快照与指标消费](frameworks/unirl-survey.md#异步快照与指标消费) — #620的状态、事件与采样时刻分开；增量单消费者，ready lag缺值不补0，源码/语法检查非训练验收
  - [奖励能力与模型加载](frameworks/unirl-survey.md#奖励能力校验与模型加载) — #621先校验再物化本地模型，但声明不认证checkpoint可用，失败不自动回滚；managed-process不在延迟范围，仅静态核对
  - [Omni原生IPC副本与导出](frameworks/unirl-survey.md#omni-原生-ipc-的副本与导出边界) — #617区分全rank参与、逐卡副本与内容一致；原meta/CPU视图规则通过，非真实IPC/模型或普通卡容量保证
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
