# DeepInfer — DeepSeek 官方推理框架（高优先调研）

> 状态：**调研进行中**（2026-10-20 建立）。仓库 `deepseek-ai/DeepInfer`，Apache-2.0。
> 用户定为本项目高优先调研项（2026-10-20）。
> 已于 2026-10-20 读取源码快照 `8b16149741a4935e1dec8cea27853f273b374c5b` 及 10 页部署 Q&A，补确定性链、ZOS 时序和服务边界；CPU 执行隔离 seed/EngineType 定义与 Worker 主机分支，未运行模型/GPU/NPU。

## 为什么高优

DeepSeek 官方下场开源其**生产服务所用的推理框架**。与 vLLM/SGLang 这类社区框架不同，这是模型厂商对自己 V4.1 模型最贴合、最"官方"的推理实现。支持 **NVIDIA SM100（Blackwell）+ 华为 Ascend 950 NPU** 双硬件——尤其 NPU 支持对 Ascend 方向意义重大。技术栈全部复用 DeepSeek 自家 kernel 库（FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect/TileLang），与我们的 FlashMLA/DeepEP/Megatron/verl 既有积累高度相关。

## 已核验事实（2026-10-20，GitHub API）

- **仓库**：`deepseek-ai/DeepInfer`，创建 2026-10-16，**今天（10-20）刚开源**——仅 2 个 commit（`initial commit` 10-20 04:09Z + Q&A 文档更新 10-20 09:06Z）。
- **规模（纠正）**：GitHub languages API 返回代码**字节数**，不是行数；Python 2,345,620 bytes、C++ 278,292 bytes。当前浅克隆按 `rg --files` 配合 `wc -l` 统计，Python 54,929 行，C++/CUDA 源码及头文件 5,908 行；不含未初始化子模块，非去注释的有效代码行数。原“234 万行 / 27 万行”撤回。
- **热度**：约 1547 stars / 112 forks（开源几小时内），22 open issues。
- **硬件**：NVIDIA SM100 GPUs；Huawei Ascend 950 NPUs。
- **软件依赖**：Linux、GCC≥13、Python 3.12、CUDA 13（NVIDIA）/ CANN 9.2（Ascend）。
- **支持模型**：DeepSeek-V4.1-Flash（默认 checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。

## 核心特性（README 列）

- **Optimized kernels**：FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect、自研 TileLang ops。
- **并行**：Attention DP / balanced CP，MoE EP + EPLB（expert parallel load balancing）。
- **DSpark + confidence-scheduled verification**：draft token 加速 + 置信度调度验证（MTP，`DEEPINFER_ENABLE_MTP_VARLEN=1`，`DEEPINFER_USE_NEXTN_DECODE` 控制最大 draft 长度默认 5）。
- **SWA bounded replay**：decoder 只回放有界后缀减少 prefill（`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`，keep 140 tokens）。
- **异步调度**：CPU 调度与设备执行重叠；NORMAL/COMPLETION 引擎默认开，PROMPT 默认关，见下文源码核验。“零开销”为项目设计表述，本项目未测量开销。
- **Batch-invariant inference**：不同 batch 组成下结果一致（`DEEPINFER_BATCH_INVARIANT=1`，NVIDIA only）——与我们在 vLLM/verl 深挖的 batch-invariant 同主题。
- **确定性采样**：官方支持条件还要求 `DEEPINFER_ENABLE_MTP_VARLEN=0`，配合 batch-invariant 与固定 seq_seed（NVIDIA only）。关闭变长验证不等于关闭 MTP，nextn 是另一个配置轴。
- **结构化输出**：regex / JSON schema / EBNF grammar 约束。
- **离线推理**：本开源版提供 offline inference；KV cache、VE/Prefill/Decode 解耦、DP 副本负载均衡不在本版（README 注明需自行补充）。

## 目录结构（初览）

- `csrc/`：C++ 内核与引擎（engine、cuda_utils、asc_utils、pybind）
- `deepinfer/`：Python 包（engine、model_executor、worker、distributed、ve、npu_ops、structure_output、sampling_params、platform、config…）
- `deep_cluster/`：硬件拓扑与集群
- `eval/`：AIME/GPQA/HLE 评测
- `examples/`：示例
- `doc/`：含《DeepSeek V4.1 Flash 推理部署 Q&A》PDF

## 与本项目积累的关联点（待展开）

- **batch-invariant / 确定性**：与 verl `_tensor_detach`、vLLM batch-invariant bug 深挖同源，DeepInfer 把它做成核心 feature。
- **P/D 分离（ve/）**：与 vLLM P/D、KV transfer、verl PD disaggregation 对照。
- **NPU（npu_ops、Ascend 950）**：与 vLLM-Ascend、MindIE、Ascend 生态系列对照。
- **MTP/DSpark**：与 speculative decoding 系列对照。
- **MoE EP+EPLB**：与 Megatron quantile balancing、router replay 对照。

## 待办 / 下一步

- [x] 读部署 Q&A 全部 10 页，渲染复核第 2 页架构图与第 9 页性能表/异步调度；生产配置与公开代码能力分开记录（见下）。
- [x] 首轮核验 batch-invariant/采样链、引擎调度默认值与服务扩展边界（见下）；完整 ZOS 调度时序及 kernel 数值仍待深读/实测。
- [ ] 对照 vLLM/SGLang 架构，定位 DeepInfer 差异化（尤其 NPU + MTP + EPLB）。
- [ ] 评估可否作为本地/服务器可跑的推理引擎（CUDA 13 / CANN 9.2 门槛高）。

## 源码核验：确定性是跨层约束

版本固定为 [8b161497](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)，浅克隆未初始化 vendor 子模块、未安装依赖。下表描述源码条件，不代表本项目已验证端到端输出一致性。

| 层次 | 已核验实现 | 含义与限制 |
|---|---|---|
| 数值算子 | [worker/runtime_config.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/runtime_config.py#L54) 在 batch-invariant 模式启用 DeepGEMM/TileKernels 确定性算法；Ascend 分支明确禁止 batch-invariant | NPU 为 CI 稳定也启用部分确定性算法，但不能据此推断 NPU 已支持完整 batch invariance |
| Attention 与通信 | [attn.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/attn.py#L666) 向 FlashMLA 传 batch_invariant；[userbuffer.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/parallel_utils/userbuffer.py#L79) 在该模式且启用专家重排时限制 multiple reduction | 开关影响 kernel 及通信归约路径，不能简化成“固定随机种子” |
| 请求随机性 | [sampling_params.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/sampling_params.py#L86) 的 _make_seed：None 取 Python PRNG 的 64 位值；整数按 uint64 截断 | 重建 SamplingParams 且不传 seed 可能得到不同种子；调用者应按请求保存明确 seed，避免负数或超范围输入隐式折返 |
| 采样位置 | [prepare_sample.cpp](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/prepare_sample.cpp#L161) 传入请求 seed，并用序列位置生成 sample_positions；[random_sampling.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L10) 将 seed、位置和采样用途组合 | 随机性按请求/位置寻址，设计上不依赖 batch 行顺序；kernel 实际精度和全链稳定性尚未实测 |
| MTP 策略 | [官方 README](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md) 的确定性说明要求关闭变长 MTP 验证 | 不将固定 seed 或 batch-invariant 单独当作可复现输出保证；模型、tokenizer、采样参数与配置也须固定 |

### 测试覆盖边界：强制回放不等于自由采样

[test_batch_invariant.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/tests/models/test_batch_invariant.py#L87) 先批量生成，再对单条请求设置 `golden_token_ids` 为批量结果，并精确比较 token、logprob、top token 等字段。源码默认 Case 为 EP=8、nextn=5。

golden 并非只供断言：[prepare_sample.cpp](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/prepare_sample.cpp#L229) 构造 golden_output，[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/sampler.py#L469) 在 decode 覆盖采样结果，prefill 的 _update_golden_tokens 也有覆盖逻辑。因此它有价值地检查同一生成轨迹上的模型数值一致性，但 token 相同部分受到强制回放约束，不能仅用该测试证明两个独立自由采样请求会产生相同 token。这里是覆盖范围判断，不是上游测试失败或功能失效的结论。

后续 GPU 验证应分开跑：①沿固定 golden 轨迹比较 logprob/top-k；②不设置 golden，固定 seed 与输入，比较独立批量/单条/换序生成；③固定 nextn，分别记录变长验证开关；④NVIDIA/NPU 分开核验各自公开支持范围。记录原始 token IDs、逐 token logprob、模型/源码版本与并行布局；不以解码字符串相同代替数值验证。

### CPU 已执行：两个隔离源码定义

[验证脚本](../../experiments/scripts/verify_deepinfer_seed_contract.py) 从源码 AST 提取原始 `SamplingParams._make_seed` 与 `EngineType`；只对 None 分支注入可复现的测试 PRNG，不导入 deepinfer/torch，也不模拟 kernel。命令：

```bash
python3 experiments/scripts/verify_deepinfer_seed_contract.py /path/to/DeepInfer
```

18 项检查通过：整数边界/uint64 折返、非整数拒绝、None 消耗 PRNG，以及三种引擎的异步调度默认值。输入文件 SHA256：

- sampling_params.py：`f800034efdebe1b21620660a51fa9122d3d7f138d8e60cd738f33d643c3116f8`
- engine_type.py：`9933d83f500e3cc14c5fef0dd2fbe5c5170dbbf24df217376b0a55c4fb4ab48c`

## 源码核验：调度和服务边界

[EngineArgs.__post_init__](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/arg_utils.py#L48) 按 EngineType 设置异步调度默认值，并重置环境变量缓存；[engine_type.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/engine_type.py#L15) 给 NORMAL/COMPLETION 开、PROMPT 关。不能只看 environ.py 的布尔缺省推断真实入口默认值。

本地内存层存在 [CacheEngine](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py) 与 C++ block/state 管理；README 所说未包含完整 KV cache 系统，不能误读成推理无 KV 内存缓存。[llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L800) 的 init_pd_split、do_save_kvcache、pe_prepare_save_kvcache 等是空扩展钩子，should_copy_state 默认 False；单靠这些接口不能组成完整 P/D 传输与缓存服务。公开入口是离线运行，在线路由、跨副本负载均衡及服务级缓存系统仍需补齐。

**可运行性判断（当前条件）**：本机为 Darwin，官方运行条件为 Linux + SM100/CUDA13 或 Ascend950/CANN9.2，因此本轮只做源码与隔离 CPU 验证。既有 4090/A16 的历史实验不能当作本框架已支持的硬件环境。待确认目标硬件、模型权重和依赖版本后再决定是否构建，不在本轮自动下载模型或安装全套依赖。

## 异步调度源码：两轮状态与三种等待

阅读同一快照的 [Worker](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py)、[zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/zos.cpp)、model_runner、prepare_tensor_input_zos 与 gather/length kernels，得到下列**主机提交顺序**。t 表示当前调用，t-1 表示 prev_step_state；不代表 GPU 在 CPU 提交下一任务前已执行完成。

| 时点 | 操作与依赖 |
|---|---|
| 当前 forward 之前 | 驱动 worker 对当前/前轮 seq_id 建立 remap；在当前 stream 上传 pinned metadata、按前轮 GPU 接受数修正长度，变长模式再重分配逐请求 q |
| 提交 forward(t) | execute_main_model 先准备输入并提交前向；forward/sample 的发射段使用 forbid_cuda_sync 守卫 |
| 处理前轮结果 | _post_process_step 等待 t-1 的 sample_end_evt，执行前轮 post_funcs 与 engine 回调；此时 forward(t) 已提交，CPU 后处理有机会与它重叠 |
| 当前采样前 | 变长模式显式等待长度 D2H 事件后，更新 CPU sampling metadata；随后提交 sample(t)。固定验证长度不做该 q 回写 D2H |
| 当前采样提交后 | pre_sampler_sync 回收已消费的旧 metadata/output 池项、处理 clone unpin 等工作；clone_done_evt/copy_state_event 是各自独立的等待，不与采样事件混同 |
| 保存或排空 | ZOS decode 保存 curr_step_state 供下轮后处理；prefill/non-ZOS 当轮后处理并清空 prev。flush_inflight_reqs 后处理最后保留的一轮并清空状态；placeholder 也走统一执行入口以排空前轮 |

“零开销”描述 CPU/GPU 重叠目标，不能理解成没有同步：采样完成事件、变长 q 的 D2H 事件和缓存 clone/copy 事件分别保护不同数据。是否消除 GPU 空泡仍须 profile 验证。官方 Q&A 的 N/N-1 编号用于概括流水线，本笔记的 t/t-1 以 Worker 的 prev_step_state 实际访问为准。

### Remap、padding 与乐观长度

`prepare_zos_inputs` 优先检查同序身份；否则按 seq_id 查找前批次位置，并排除前轮 placeholder。没有交集就返回无修正元数据；有交集时，fresh 行 remap=-1，CUDA Graph padding 行使用 remap=-1/q=1/k=1。不能直接按当前行号读取前轮结果。

当前有重叠的请求：第一个输入 token 来自 `prev_sampled_token_ids[remap[i]]`，后续来自该前轮请求的 draft tokens；fresh 请求从 CPU 准备的 fresh_tokens 取输入。源码链为 prepare_token_inputs → gather_input_ids_zos。model_runner 在图执行时给 seq_metadata_list 增补 placeholder，因此 C++ 输入行数与 padded remap 长度对应；不能忽略此层而误判 shape 不符。

nextn>0 时，C++ 先从乐观 k 扣除 CPU 侧原 q；GPU 再加修正后的 q，并对匹配行扣除 `nextn - prev_num_accepted_tokens`。因此不能只把累计 token 长度加固定 nextn。变长 q 按前轮 confidence 的累计接受概率竞争总预算：每请求保留一个基础 token，余量按稳定排序分配；总预算与逐请求 q 是两个层次。CPU 读回 q 后，update_metadata_for_sampling_zos 还按已落账的 seq.cache.num_tokens 重建 sampling k，并对匹配请求 refresh_for_sampling。

**生命周期约束**：旧 SequenceMetadata 在下轮构造 remap、准备当前输入时仍须可用；engine.post_process 先 collect，pre_sampler_sync 再 release。RequestOutput 由池托管，跨下一次 step 保存必须 clone，不能长期留原指针。上述为源码所有权约定，未证明自定义 stream、GPU 内存分配器或多卡通信无竞态。

### 已执行的主机控制流实验

[verify_deepinfer_zos_host_flow.py](../../experiments/scripts/verify_deepinfer_zos_host_flow.py) 从 worker.py AST 保留原始 execute_model_step/_post_process_step/flush_inflight_reqs 函数体，移除装饰器并用测试替身替代设备、event、模型、回调；长度修正结果作为显式注入输入。执行通过：

- 连续 decode：先提交当前 forward，再等待/后处理前轮，当前结果保留到下一轮；
- prefill 与 non-ZOS 当轮后处理，flush 清空状态且重复 flush 不重复输出；
- placeholder 排空前轮真实输出，自身不产生请求输出；
- 注入变长修正时，先等长度 D2H，再更新采样 metadata，最后提交当前 sample。

复现：`python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer`。输入 worker.py SHA256：`ad32e595d9ab289dff1a7709136431ac60dae709e9e5123ae42eb0ebf7c30cbb`。初次测试因本地替身遗漏 gpu_cache 属性失败，补齐后通过；非上游缺陷。该实验只验证原始 Python 主机分支/顺序，不执行 C++ remap 或 q/k kernel，不检验真实 event 完成、stream、H2D/D2H、显存生命周期、模型或多卡正确性。

## 官方部署问答：与公开代码对照

来源：[官方 Q&A，10 页](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%20Q%26A.pdf)，同一源码版本；PDF SHA256：`692df3308b3505ef798b15f229634e0d8958f121d92a5cc1a69c69bb704f64f7`。已提取全部正文，并视觉复核第 2 页架构图及第 9 页表格。下列均是官方部署材料/表格的陈述，非本项目实测：

- 第 1–3 页描述分离 VE/Prefill/Decode 与外部缓存存储；生产策略以小 EP 做 prefill、Wide-EP 做 decode。材料中的高效最小部署举例为 64 卡 P + 64 卡 D；同时允许更小的 P/D 引擎组合，不能把生产效率建议当作开源离线入口的硬性最小卡数。
- 第 2 页说该生产布局尚未用 TP/CP；README 列支持 balanced CP。这分别是已用部署配置和代码能力范围，不构成互斥。
- 第 5–7 页区分外部存储上的 Block KV 与机器内存中的 SWA state；缓存服务和负载均衡并未随本次引擎完整开源。该描述与源码保留内存 CacheEngine、服务扩展钩子为空相符。
- 第 9 页表格为 EP64/DP64 场景的估计吞吐对照：16 seqs/rank 为 -1.5%，60 为 +5.4%，96 为 +11.0%；说明低并发可能不获益，不能笼统承诺 +4–7%。表格比较的条件、估算方法与真实在线负载必须保留。
- 同页说明 CPU 先选总预算、GPU 用上一轮 confidence 计算逐请求验证长度并回写 pinned buffer，与 Worker 的 correction/D2H 路径相符；这说明公开代码为何需要该同步，未证明它在任意负载都能隐藏等待。
- 第 6–7 页使用 134-token 窗口/回退示例，当前 README 的 compact keep_tokens 为 140，且 PDF 含 VLLM_ 历史前缀。应按当前配置检查窗口和回退覆盖条件，不将 PDF 示例数字直接当作启动默认值；重放的数值等价性仍需实测。

下一步：DSpark 总预算/采样位置及 SWA/YoCo 窗口/回退/压缩元数据契约已补后文；接续 CP compact 边界重分布和上游数值测试覆盖。真实 CUDA Graph、padding、提前 EOS 与 pool 回收需要目标硬件验证。本机仍仅限源码/CPU，不据主机替身实验宣称 GPU 无空泡或无竞态。

## DSpark：请求数、验证预算与随机采样是三个层次

本节仍固定 `8b161497`，不改上游源码。关键入口为 [MTPVarlen](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp#L133) 和 [Scheduler 调用处](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L644)。

### 不是重新挑选请求数

Scheduler 已完成请求选择、内存分配/抢占并按长度排序，才把 `scheduled` 传给 MTPVarlen。后者不增删请求，而给每条请求至少一个基础验证行，再选择额外 draft 验证行。代码变量 `best_bsz` 是 `total_q = Σ seqlens_q`，**不是请求数，也不是实际接受/输出 token 数**；候选 total_q 映射到第一个容量足够的 CUDA Graph size，三者不要混用。

令 n 为已选请求数，r 为 confidence head 输出累乘后的估计接受率。把全部 r 降序排列，对每个额外验证行数 k，源码比较：

```text
expected(k) = n + sum(top-k cumulative acceptance estimates)
Q = n + k
graph(Q) = 性能曲线中第一个 size >= Q 的点
time(Q) = baseline(graph) + max(0, slope) * graph + sampler_estimate(n)
score(Q) = expected(k) / time(Q)
```

这是**调度器的吞吐代理目标**，不是真实输出长度或端到端吞吐保证。confidence 来自模型估计；EOS、抢占、通信及运行时误差等仍会改变收益。曲线为空时直接保留全部 draft，不执行最优预算选择；相同 score 保留较小 Q。固定预算下，按最新可用累计接受率给请求分配额外行；概率在 [0,1] 时累乘单调，使前缀长度选择与贪心分配相容。相等 rate 按请求在当前 scheduled 中的索引优先，不等同于全局请求 ID 公平性。

### 为什么预算和分配不用同一轮 confidence

头文件明确要求后续位置 confidence 不影响较早 draft 是否被验证；为此预算始终使用 N-2 轮估计，逐请求分配希望用 N-1 轮估计。CPU 缓冲区的含义随在途状态变化：

| 调度/后处理计数 | CPU 预算来源 | 逐请求分配 |
|---|---|---|
| schedule_cnt == process_cnt | prev_accept_rates（N-2） | accept_rates（N-1） |
| schedule_cnt == process_cnt + 1 | accept_rates（N-2） | CPU 先用旧值；ZOS GPU 再用前轮 confidence 重分配 |

[process_MTPInfo](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L1239) 负责缓冲移位和逐位 confidence 累乘；不是对各位置的原始独立 confidence 直接排序。无历史时，有 draft 信息按 1 补齐，无 draft 按 0 补齐；已预调度一轮也视为有 draft。ZOS 的 [GPU 分配参考](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L34) 先 remap、对 fresh 行补零，再累乘、稳定排序。这里解释实现的因果约束，**没有证明整个自适应采样过程统计无偏**。

例如两条请求各最多两个 draft：旧累计率 A=[.9,.8]、B=[.2,.1]，曲线 `(2,1ms),(3,1.1ms),(4,1.2ms),(6,3ms)` 时选 Q=4。把最新累计率改为 A=[.2,.1]、B=[.9,.8]，分配为 q=[1,3]；交换最新两行后为 [3,1]，总预算不变。只有改旧轮估计才会改变此例预算。这是合成输入的 CPU 调度结果，不是模型实测。

### 耗时反馈的两个稳健统计量

- forward：最近 5000 步，先按 graph size 求归一残差的中位数，每种 graph 至少 10 样本才纳入；再对各 graph 中位数等权求中位数。样本多的 graph 不会自动拥有更大权重。负 slope 截为 0，启动曲线为耗时下界。
- sampler：每个请求数及全局分别维护最近 9 次中位数，至少 5 样本才可用；优先同请求数估计，再全局回退，否则 0。较大 sampler 固定成本可能促使预算变大以摊薄成本；较大 forward 线性成本则可能减少推测。不能仅凭 confidence 判断预算必然增减。

### 采样位置没有直接使用 batch 行号

[prepare_sample.cpp](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/prepare_sample.cpp#L206) 的 decode 基础位置是 `seqlen_k - seqlen_q + 1`；[expand_sampling_params_ref](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L55) 为验证行构造 `base_position + local_step * 10000019`，请求 seed 随 row_to_req 映射。DFlash proposal 使用基础位置 + last_valid_offset + draft 步数（从 1 开始），并通过 tag=mtp 给 seed 增加 1000000007。

[fused_sample 的参考路径](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/fused_sample.py#L208) 把自由采样、拒绝重采样、接受判定分为不同随机用途（位置乘 3 后分别用 case 0/1/2；接受判定还有独立组合方式）。这些是命名空间分隔，不是 hash 无碰撞的数学保证。按请求/位置寻址可以避免单纯换行重排引入 RNG 漂移，**不能据此推出改变验证长度后自由生成路径相同**；官方确定性配置仍要求关闭 varlen，见前文。

### CPU 验证：实际头文件，而非另写算法的自证

[verify_deepinfer_mtp_budget.py](../../experiments/scripts/verify_deepinfer_mtp_budget.py) 读取原始头文件，仅替换 `sequence.hpp` include 为最小 Sequence 字段替身和 TORCH_CHECK 异常替身；三个调度/估时类的函数体原样编译，未改动 DeepInfer 克隆。输入头文件 SHA256 为 `164f45db4cccd6d6909dd2846e7dfffb57754b108af679443bf4bd71ec33785a`。

```bash
python3 experiments/scripts/verify_deepinfer_mtp_budget.py /path/to/DeepInfer
# 本机 CLT 默认标准头搜索路径不完整，显式使用已有 SDK；不安装或修改系统环境：
python3 experiments/scripts/verify_deepinfer_mtp_budget.py /path/to/DeepInfer \
  --cxxflag=-isystem/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1
```

Darwin arm64、Apple clang 17、C++17，结果 `PASS 421 checks`。其中 200 组合成累计率逐组穷举 3 条请求、每条 q=1..3 的全部 27 种合法前缀配置，独立核对预算目标与固定预算下最新率分配；另覆盖双历史缓冲、冷启动、计数约束、排序/吞吐平局、滑动窗口淘汰、等权 graph 中位数、反馈门槛/回退及斜率钳制。初次编译因找不到 algorithm 标准头失败，指定已有 SDK 目录后通过。未调用 torch、CUDA/NPU、GPU 分配 kernel 或完整 Scheduler；不验证模型接受率校准、统计无偏、设备数值、真实定时事件或性能。

## YoCo bounded replay：截 query，不是删除全部历史 KV

本节仍固定 `8b161497`。公开实现包含三种模式，作用在 prefill 的 consumer 层区；不能当作对任意 Transformer 输入做左截断。模型须有非空 producer/consumer 层区，入口和 Worker 还校验正 SWA window 与保留长度。

| 模式 | Consumer query 行 | Consumer SWA 可见域 | 主/索引压缩 KV |
|---|---|---|---|
| off | 全部 | 正常滑窗（另受 state 有效下界约束） | 正常路径 |
| mask | 全部 | 滑窗与保留后缀相交；后缀之前的行为空 | 全长上下文路径仍在 |
| compact | 保留后缀与当前 chunk 相交的行；全截 chunk 仍留一行 | 与 mask 在相同行上的绝对 key 集一致 | 保留 full-source 分支更新，而非仅由 compact query 构造 |

关键源码：[C++ 元数据](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L129)、[consumer metadata view](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/input_metadata/yoco.py#L94)、[边界层](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/decoder_layer.py#L134)、[full KV 更新](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/attn.py#L676)。边界处先保存 full metadata，并对原始 full hidden 做 attn norm；另把 query hidden/residual gather 为 compact 布局。需要 packed KV 的 consumer 层用 full hidden 投影、压缩并写入完整上下文的主/索引 KV，后续 query 路径跳过重复更新。因此减少 consumer query/MLP 计算，不意味着所有 KV 计算或存储同步缩为 keep_tokens。

### 窗口、回退和默认值分开看

[environ.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/environ.py#L128) 的 nextn 默认 5；有效回退预算 B 为 `max(显式 MAX_BACKTRACKING_TOKENS 或 0, nextn + 1)`。YoCo mode 默认 off，keep 默认 0，打开截断必须显式设置。源码在 Worker 与 C++ 双层要求 `keep >= window + B`；README 的 keep=140 是默认 V4.1 条件下的建议值，不是所有配置都安全的常数。例如 window=128、nextn=5 且未显式增加回退时 B=6，约束下限为 134；若 nextn 或显式 B 增大，下限随之增大，不能照抄 140。

[cache_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py#L87) 还区分 inference SWA ring（window+B）与 KV-store state ring（满足额外存储开关时，再加落盘余量及 YoCo keep）。keep 是 consumer 重放后缀，不等于推理 ring 大小或实际持久化容量。公开服务级 KV-store 仍不完整；这些布局函数不构成可用存储服务的证据。

### Chunk 不可整段删除：占位行与绝对坐标

令 N 为 sequence 当前总长度（重算时含已生成 token），q_start 为当前 chunk 起点，q_len 为 chunk 长度。compact 使用：

```text
clip_prefix = clamp(N - keep - q_start, 0, q_len - 1)
keep_len = q_len - clip_prefix >= 1
query_indices = 当前请求 full 布局基址 + [clip_prefix, q_len)
positions = q_start + [clip_prefix, q_len)  # 不重新从零编号
```

例如 N=2000、keep=140、首 chunk=[0,1024)，目标后缀从 1860 开始，但仍保留 position=1023 的占位 query。该行 SWA 可见域为空，compact 的 store slot=-1，不写 ring；prefill 采样索引仍可定位最后一行，非最终 chunk 的采样结果随后丢弃。真正保留的后缀行也只在当前 chunk 的 ring 尾段写 SWA slot。全截 chunk 的一行是执行协议占位，不是“仍需消费一个真实输出”。

对 query 的绝对位置 pos，共用 C++ helper 把可见下界设为：

```text
abs_lo = clamp(max(pos-window+1, N-keep, swa_valid_start), 0, pos+1)
visible_keys = [abs_lo, pos]  # abs_lo=pos+1 时为空
```

`swa_valid_start` 防止零状态重算读取已清零的历史 slot，不能仅取滑窗与矩形交集。helper 再把绝对 key 分为 chunk 前历史 section1 与当前 chunk section2；compact 对 section2 基址减 clip_prefix，但绝对位置不能变。Python consumer view 用浅拷贝替换 query-aligned 字段并清空旧 attention 展开结果；未替换字段仍共享 tensor，禁止随意原地修改。CP>1 还需从 full-mirror 持有者收集保留行，再重排为新的 compact-mirror 布局，不能把全局 query_indices 直接用于本 rank 的旧局部行。

### 已运行什么，仍未证明什么

[CPU 检查脚本](../../experiments/scripts/verify_deepinfer_yoco_visibility.py) 只提取原始 `YocoSwaVisibility` 与 `yoco_swa_visibility` 函数体，替代 TORCH_CHECK 的消息接口后以 C++17 编译；通过 AST 复用上游 L1 的 10 种场景及解析可见域参考，再增加 200 组固定种子边界形状和 `valid_start` 的零/边界/矩形截点。结果 24,643 个 query/有效下界组合通过，包括空可见域和占位行；4,467 个保留行的 section2 绝对坐标重定位检查通过。

```bash
python3 experiments/scripts/verify_deepinfer_yoco_visibility.py /path/to/DeepInfer \
  --cxxflag=-isystem/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1
```

环境为 Darwin arm64 / Apple clang 17；沿用已有 SDK 标准头路径，没有安装框架或依赖。C++ 文件 SHA256 `93a6b1dd533db116d2f7540c78d6bb5419bdb8c1d2063b031ab8b1a8cc138fe2`；上游 yoco_equiv.py 为 `4abe933e510b7797ada8835d57d7504dcc049834c990429316e34e937f6c0032`。本实验执行的是**可见域 helper**，clip/布局重定位由测试夹具计算；没有执行全套 C++ 元数据、pybind、pinned 分配、store-slot kernel、CP 通信或模型，不能称为上游 L1/L2 已跑通。

上游 [YoCo 验证说明](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/yoco/README.md) 的 L1 不加载权重，但实际 `_setup_mask_command` 仍要求可见设备以分配 pinned 内存；所以本机没有尝试完整 L1。L2 比较的是 **mask vs compact**，不是 off vs compact；两边沿同一 golden teacher-forcing 轨迹比较 logprob/top-k，且明确断言抢占数为 0。启用 DFlash 时附加 confidence 对照有首末轮覆盖限制。不能据这些测试证明自由采样、任意抢占/回退或关闭截断模式的逐位等价；llm_engine.py 还明确注明截断开启时允许抢占导致结果变化。

另外两个功能边界来自源码：mask/compact 均把 prompt_logprobs 强制关闭；compact 要求 DFlash target 层全部位于 consumer 边界同一侧，并禁止 consumer 层有 Engram。它们是当前实现约束，不是可忽略的部署细节。CP compact 边界索引与测试覆盖已补下节；真实设备通信和模型等价仍未验证，不在 M1 启动设备实验。

## CP compact 边界：先按旧归属取行，再按新归属分发

同一 `8b161497` 版本的 [prepare_cp_mirror.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py#L126) 提供两个纯 Python 索引参考：mirror zigzag 与 YoCo boundary redistribution；CUDA/NPU kernel 在各自后端目录。Python [consumer view](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/input_metadata/yoco.py#L130) 将二者组合，再由 [yoco_boundary_compact](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/yoco.py#L56) 消费。

每条请求的 full query 长度 q 被分成 `2*CP` 个等容量 block，单块容量 `ceil(q/(2*CP))`；rank r 持有 block r 与 `2*CP-1-r`。compact 后必须按新的 keep_len **重新分块**，旧 rank 持有的后缀行不一定仍归本 rank。以 q=40、keep=13、CP=4 为例（表内为请求内原始 query 下标）：

| rank | 旧布局实际贡献的后缀 | 新 compact-mirror 应收到的行（含 pad） |
|---|---|---|
| 0 | 35–39 | 27, 28, pad, pad |
| 1 | 30–34 | 29, 30, 39, pad |
| 2 | 27–29 | 31, 32, 37, 38 |
| 3 | 无 | 33, 34, 35, 36 |

因此不能把全局 full→compact 的 query_indices 直接用于本 rank 的 full-mirror tensor。正确链为：每 rank 用 contrib_index 从旧局部 tensor 取出仍需保留的行 → 各 rank 补齐到相同 contrib_pad_len → 按 rank 顺序 all-gather → 用 gathered_reverse 把 compact 全局顺序映射到 gather 缓冲 → 用新的 mirror_cp_index 生成每 rank 的 boundary_index，再取回行。`contrib_pad_len` 是**旧 rank 实际贡献数的最大值**，不等于新布局每 rank 的 mirror_len；上例分别为 5 和 4。

### 两个不应丢掉的坐标约束

1. **压缩 key 范围仍按绝对位置。** compact mirror 准备函数接收 keep_lens 与原始 seq_ks，不缩短 k；原 `q_start=sk-q`，截前缀 clip 后应为 `sk-keep=q_start+clip`。对保留行 j，压缩范围终点仍是 `base + floor((sk-keep+j+1)/ratio)`。query 截短不能一起把历史 key 长度或跨请求压缩段基址截短。
2. **pad 内容不保证是零。** [gather](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/gather.py#L25) 对越界索引填零，但 `boundary_index = gather(gathered_reverse, mirror_index)` 会把 pad 的 -1 变为索引值 0；再次 gather payload 时可能读到第一个真实 token，而不是零行。上例 rank0 的两个 pad 就可读到原 query 35。源码明确把它视为无语义 pad，并设置 `valid_token_mask = mirror_index != -1`，后续 MoE 屏蔽且采样仅取有效行。**只能要求有效行身份正确，不能要求整个 padded tensor 为零或直接把 pad 算进结果。** 这是源码约定，不是本轮发现的错误或真实设备污染复现。

### 已运行的参考组合检查与覆盖缺口

复用并扩展 [既有 YoCo 验证脚本](../../experiments/scripts/verify_deepinfer_yoco_visibility.py)，新增无编译/无 torch 的参考组合模式：

```bash
python3 experiments/scripts/verify_deepinfer_yoco_visibility.py /path/to/DeepInfer --cp-reference-only
```

AST 提取原始 `prepare_cp_mirror_ref`、`yoco_cp_boundary_fused_ref` 与贡献长度参考，函数体未改写；在 Python 3.9 主机只为隔离命名空间补严格 zip 行为，gather/all-gather 用整数身份标记夹具代替。复用上游 5 个 boundary 形状（含 2000 个短请求），补 513 请求混长与 100 组固定种子形状，在 CP=1/2/4/8 共 424 个形状/CP 组合中检查：贡献集合无丢失/重复、rank-major 汇集后可恢复 compact 全局顺序、各 rank 的有效行身份、反向索引和多压缩比的绝对 ks/ke。结果通过；出现 48,162 个跨 rank 迁移行，75,598 个 pad 中 25,246 个非零，确认测试实际覆盖了需要通信和不可按零解释的 pad 情况。

源码文件 SHA256：`4b58b8b353be45ea0ebb917a8ddb2a2e4b1c2e2350e57d2ff0036ae4bf25b7d7`。首次夹具读取把上游 `[2] * 2000` 当纯字面量而失败；改为仅允许 list/tuple/常量/乘法的 AST 表达式后通过，非上游缺陷。20:00 的未提交成果曾因共享分支切换未留在当前文件中，20:30 接续在远端最新版本的隔离副本恢复后重新执行，结果一致；不将此前未完成发布表述成已完成。

该模式只验证**上游参考的组合与索引契约**，没有执行 TileLang kernel、C++ shape 标量、真实 gather、通信顺序、valid-token mask 的设备实现或模型；不能因此判定 GPU/NPU 完全正确。当前上游 `test_prepare_cp_mirror` 覆盖多种 CP 大小/部分 rank、零长序列和混长；`test_yoco_cp_boundary_fused` 的 kernel 对照则固定 CP=4、5 个形状并覆盖全部 4 ranks，尚未在该测试矩阵中覆盖 CP=2/8。consumer view 注释中 keep_lens+原 seq_ks 的组合依赖 L2 验证，而非独立组合单测。本轮给出可复用 CPU 契约检查，**不是已找到 kernel bug 或可直接提 PR 的证据**。下一步只读查重当前 DeepInfer issue/PR，优先有具体失败证据、无人接手且本机可验证的贡献点，不为扩测试数制造候选。

## PR #54 只读核验：事件已完成，不等于所有持有者已消费

对象为 [DeepInfer #54](https://github.com/deepseek-ai/DeepInfer/pull/54)，本轮 API 确认 open、未合并，base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`、head `70381d289c2b1e7c0e092a56a01a644e19fe37de`。**本节分析该 PR 的固定 head，不将它当作上游 main 已确认缺陷。** 已读全部 9 文件的 diff（含仓库内 OPTIMIZATION.md），另读 Worker 执行/后处理方法、完整 CUDATimer、计时回调和 SO 相关调用上下文；未读取另挂的用户附件，也未对外发 review/comment 或实现竞争补丁。

PR 的四组改动是 profiler 空闲快路、计时事件池/完成标记复用、SO 活跃计数、空输出回收快路。作者宣称的 50–140us/step、SM86 单机计时/扩展测试均为作者报告，本项目没有测量或复核这些性能数据。应分别检查事件的两类用途，不能因都叫 Event 就共享同一安全论证。

### 采样完成标记与计时事件池的不同持有者

- `Worker._sample_end_evt`：每轮先同步/消费 prev_step_state，再在采样后重录同一个非计时标记。原始 Worker 三轮 decode+flush 的主机替身检查中，一共只创建一个标记，每次重录前上代均已 synchronize；该路径的主机顺序检查通过，未证明其他设备/并发路径全部安全。
- `CUDATimer` 的 pair pool：`flush_out` 遍历所有记录，把已完成并读取过 elapsed_time 的 pair 放回池。可是计时器自己读完不代表 WorkerStepState 的外部持有者也读完。Worker 在当前 forward 后取走 pair 引用，随后才后处理上一轮；model_runner.post_process 的 flush 可以同时看到**当前 forward**，若它也已完成，就将其提前回池。

[NVIDIA Event 文档](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html) 明确说明事件允许重复 record，并会覆盖其先前捕获状态；query/synchronize 判断的是当前捕获的工作。故 query 成功只支持“设备工作已完成”，不能单独支持“事件句柄没有未来逻辑读者”。下述实验将这一语义显式建模，不是对真实 CUDA 的测量。

### 一个合法的完成时机即可反驳“没有持有者会晚读”的论证

以 F1/F2 表示 forward 计时 pair、S1 表示 sample pair，假设在下一轮 post_process 的非阻塞 flush 时，当前 forward 也已经完成（例如 CPU 被调度延迟）。源码未排除此时机。

| 阶段 | 关键引用与池行为 |
|---|---|
| 第 1 轮 | state1 保存 F1/S1 |
| 第 2 轮 forward 后 | state2 将持有 F2；后处理 state1 的 flush 可能把 F1、F2、S1 一起回池；回调读 state1 后，sample2 可复用 S1 |
| 第 3 轮 forward | 可以从池取出 F2 重录，此时 state2 仍持有同一 pair，尚未进入其后处理 |
| 随后后处理 state2 | engine 的原始 update_mtp_runtime 调用从旧引用读到 forward3 的时间，而非 forward2；后续同轮 sample 甚至可再次复用该 pair，使 state3 的 forward/sample 引用别名 |

这说明当前完成标记复用与 pair 回收时机是两个问题。潜在影响是污染 DSpark 运行时估计；是否发生、频率及对吞吐/生成的实际影响仍需目标环境验证，不能从本实验推出 token 输出错误或性能必然下降。保守的可复用原则是：设备完成 **且所有逻辑读者消费完** 才能回池；本轮没有给上游实现修复或添加同步。

### 原始主机函数的替身反例与对照

复用 [既有 ZOS 检查脚本](../../experiments/scripts/verify_deepinfer_zos_host_flow.py)，新增：

```bash
python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer-pr54 --check-event-pool
```

AST 保留 PR head 的 BaseTimer/CUDATimer、Worker 三个执行方法及 engine.post_process 的实际计时读取分支；Worker 装饰器移除。模型前向/采样、设备事件/完成时机、model_runner 后处理和回调环境用夹具替代，所有已提交事件在 query 时均设为完成。合成 forward 为 100/200/300ms、sample 为 10/20/30ms，数值只用于区分代际，**不是硬件时延或 benchmark**。

| 轮次 | 期望 forward/sample | PR 池化后的读取 | 禁用复用的夹具对照 |
|---|---|---|---|
| 1 | 100 / 10 | 100 / 10 | 100 / 10 |
| 2 | 200 / 20 | 300 / 20 | 200 / 20 |
| 3（flush） | 300 / 30 | 30 / 30 | 300 / 30 |

结果稳定复现；对照只将事件获取替换为新建 pair，其他源函数体不改，**不是上游修复已验证**。同时回归原 `8b161497` 的默认主机时序场景通过。首次夹具未带入计时分支依赖的 seq_metadata_list 绑定，产生 NameError；补上原始绑定后才得到上述反例，最初失败不是上游缺陷。

输入 SHA256：timer.py `a221628f57f8600116e73f3eb8dc173cd1585f633200137ea1f378cb2f22d78f`；worker.py `7faf0a5e29cc20647f9cc35435a3475baf7b1e955b86acdb47ce59b942733bda`；llm_engine.py `e9def731f151ff38b406ac283e330dccd6420f1de7df13e344c98d42ac970663`。环境为 Darwin / Python 3.9.6 / numpy 2.0.2。该模式限定 driver+varlen+nextn>0、有效 graph、ZOS decode；未运行 torch/CUDA、真实 event、stream、kernel、多卡或模型，不应称为 GPU race 复现。

SO 计数的限定状态模型及反向故障检查见下节；不能仅凭注释或该模型宣称所有异步路径“永不低估”。任何上游评论仍需用户拍板，不另开竞争 PR。

## PR #54 SO 计数：需要保守上界，不要求每一刻精确相等

继续固定同一 PR head `70381d28`，本轮 API 确认仍 open、未合并。`num_active_so` 在 [C++ schedule](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/csrc/engine/scheduler.hpp#L715) 中按已选请求的 `structured_output.status > 0` 统计；[Python prefill 分支](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/deepinfer/engine/llm_engine.py#L921) 在最后 chunk 把 metadata 的 0 改为 1 时加一；[sampler](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/deepinfer/model_executor/layers/sampler.py#L303) 在计数非正时跳过整批 SO 处理。None 参数保留原来的逐请求扫描。

记 C 为计数、A 为采样前 metadata 中 `so_status>0` 的请求数，安全快路只需非负计数满足 **C ≥ A**：C=0 才能推断 A=0。计数偏高通常只是多走处理路径，不能据此判错；计数偏低且变成零则可直接遗漏约束。尤其不需要 ledger/metadata 的**整数状态始终相同**，只需新的激活路径被计数维护覆盖。

| 阶段/状态变化 | 对活跃谓词的作用 | 当前源码中的维护 |
|---|---|---|
| 普通请求或待激活 SO 的 0 | 非活跃 | 不计入；异步 matcher 编完后先挂接/入调度，不在当前 batch 偷加一条活跃请求 |
| 最后 prefill chunk 的 0→1 | 新增活跃 | Python 分支同步 C+=1；没有 matcher 或状态异常则判死请求 |
| 已活跃的 1→2 | 仍活跃 | sampler 开始跟进 token；不需要再加一 |
| 活跃状态→-1 | 减少活跃 | 旧快照计数可能偏高，不能因此漏掉其他活跃请求 |
| 完整前缀 fork 的初始 1 | 新请求已活跃 | fork 初始化在该请求进入后续 schedule 统计之前；不等同于给已生成快照的 batch 临时激活 |
| ZOS 匹配行刷新 | metadata 从 ledger 重读 | refresh_for_sampling 可看到 1→2/-1；值并非必须“不变”，但这些转移不会增加活跃数量 |

来源还包括 [fork 初始化](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/csrc/engine/scheduler.hpp#L1112)、[metadata 刷新](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/csrc/engine/sequence.hpp#L408)、[ZOS 仅刷新匹配行](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/csrc/engine/zos.cpp#L279)、matcher builder 的 pump→on_ready 入调度链。`process_outputs` 在 SO status 数组为 None 时保留既有账本，并仍依据账本 -1 判死；不能把“本步没返回 SO 状态数组”误解成恢复为普通请求。

### 原始 Python 分支检查、负面结果和故障注入

扩展既有脚本而不新建贡献候选：

```bash
python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer-pr54 --check-so-counter
```

AST 原样提取 engine 的 prefill 激活分支与完整 `_get_so_mask`（移除 sampler 装饰器）；C++ 的初始统计、账本转移和匹配刷新用**显式模型**表示，且对所依赖的 C++ 统计谓词做源码形状检查。模型取状态 {-1,0,1,2}、1–3 请求，刷新允许 `-1→-1、0→0、1→{-1,1,2}、2→{-1,2}`，组合 decode、未完 prefill、最终 prefill、缺 matcher、无 SO 参数和混合行策略；另检查 matcher 已 aborted、None metadata 和空批。

2,394 个状态/策略组合通过，上界有 1,869 次严格偏高，**在这些假设内没有漏计反例**。有活跃请求时，计数路径与 None 回退路径的掩码行/状态输出相同；无活跃请求时，两者都不生成有效掩码，但偏高计数可能返回 `[-1,...]` 状态列表而旧扫描返回 None，不能宣称两者的原始返回值逐字节等同。

故障注入为单条最终 prefill SO：原始分支将 status 0→1、C 0→1，实际 sampler 会处理第 0 行；只把传给 sampler 的计数故意改回陈旧的 0，则整批返回 None、漏掉该活跃行。该控制证明加一维护是必要条件，**不是当前 PR 已遗漏加一**。fixture 的 matcher 只固定接受 token/填掩码或报告 aborted，不执行真实 grammar；Tensor/Event/stream 也是假实现，所以它验证的是计数闸门和分支，不验证 xgrammar、CUDA 掩码内容、生命周期或通信。

源文件 SHA256：sampler.py `7ab3bed761ca731e5ece5b6e77da61104a5a03d6e418028149896197df1e3156`；scheduler.hpp `d2752395adfdb5867d9a15f0e68734c6b4c9c7360721461712362a04f3b27972`；engine hash 同前节。Python 3.9.6 / numpy 2.0.2，无新依赖或设备访问；计时池反例模式和原始主机时序模式均回归通过。

**判断更新**：不把 SO 计数方案扩列为已证实缺陷；上一节计时 pair 的未消费引用反例仍独立成立。模型未执行 C++ schedule/process_outputs/pybind，也未跑编译线程、fork 的完整异步交错或外部服务扩展；所有未来的非活跃→活跃新路径仍须维护计数，不能把选定转移矩阵当作全系统证明。下一步核对同 PR 的借用输出数组 copy 快路，随后收束为供用户决定是否参与原 PR 讨论的证据摘要，不实现竞争修复。

## PR #54 输出回收：先核实数组所有权，再讨论原生对象寿命

2026-10-20 22:00 心跳继续固定 head `70381d28`；只读 API 仍 open、未合并。**纠正检查前提**：[collect_request_outputs 注释](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/deepinfer/engine/llm_engine.py#L92) 称返回数组借用静态 vector，但 [outputs.hpp](https://github.com/deepseek-ai/DeepInfer/blob/70381d289c2b1e7c0e092a56a01a644e19fe37de/csrc/engine/outputs.hpp) 末尾实际调用 `py::array_t<uint64_t>(request_outputs.size(), request_outputs.data())`，没有 base。不能因存在静态 vector，就直接推断 Python 数组仍引用其存储。

[pybind11 3.0.1 numpy.h](https://github.com/pybind/pybind11/blob/v3.0.1/include/pybind11/numpy.h#L1079) 的构造器在 ptr 非空而 base 未指定时执行 NumPy copy；count 重载也沿用该路径。DeepInfer requirements.txt 锁 pybind11=3.0.1，但 setup.py 使用 torch 扩展构建，**安装包版本不等于实际编译采用的头文件**。进一步只读检查 [PyTorch v2.13.0 子模块](https://github.com/pytorch/pytorch/tree/v2.13.0/third_party/pybind11)（DeepInfer 两种后端 requirements 均锁 torch 2.13.0）：指向 `d03662f0984f652b60e7ddce53d3868002275197`，版本宏 3.0.4；该提交的无 base 分支同样复制。本机没有构建 DeepInfer，也没有目标 wheel 的 include trace，因此仍将结论限定到这两套明确头文件，不推断任意定制编译环境。

### 小型原生扩展与真正借用的对照

扩展既有脚本，不修改上游或引入 torch：

```bash
python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer-pr54 \
  --check-output-copy --pybind-include /path/to/pybind11/include \
  --python-include /path/to/Python/Headers
# 本机 Apple clang 17 须额外指定 SDK 的 libc++ 搜索路径：
# --cxx-flag=-isystem --cxx-flag=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
```

脚本从 outputs.hpp 提取并检查**精确返回表达式**，在独立小型扩展中使用相同 static vector/clear/emplace_back/return 组合；填入的是整数身份，不是可解引用的 RequestOutput 指针。这里实际执行 C++/pybind/NumPy 构造语义，但不是完整 build_request_outputs。显式 capsule base 的借用数组作为对照，容量固定且不重新分配，避免悬空内存测试。

- 不带 base 的第一次数组 `[100,101,102]`：OWNDATA=true、base=None、data 地址不同于 vector；第二次构建 `[200,201,202]` 后第一份不变，两数组不共享数据。
- 显式 base 的借用数组起初 `[300,301,302]`：OWNDATA=false、data 地址等于 vector；下次构建后变为 `[400,401,402]`，提前 copy 的快照仍为原值。这证明测试能识别真实借用覆盖，**不是当前返回路径发生该故障**。
- AST 原样提取 PR 的 collect_request_outputs 与 release，pool/free_deferred 回调用替身，无 clone-unpin 或 sequence 回收。分别用独立数组与显式借用数组，枚举两次输入长度0–4，每套头文件50组合；检查空输入不替换已有队列、首批 copy/多批 concatenate、存储覆盖后的值、release 后清空及下一轮重用。两套头文件均通过；当前 copy 快路没有在这些检查中出现指针值遗漏/覆盖。

环境：Darwin arm64 / Python3.9.6 / numpy2.0.2 / Apple clang17。pybind11 3.0.1 commit `f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8`，numpy.h SHA256 `f890749a209828190b31c7e6d809c73181916ad38c24edc9fd426be6c7c9d811`；torch 对应子模块 numpy.h SHA256 `a9bba1f0976acd7e7597daef1abd398a1e0b81213747278cffef6b325979fee1`。outputs.hpp SHA256 `9ef2f5a37dd07b03fe7fac7f92cbbf1dfc056ff4a0c0fbb9b7a384149f71bf6f`，engine hash 同前节。第一次编译缺 libc++ 头 cstddef，显式加 SDK include 后通过；python3-config 不在 PATH，改用已确认的 Python headers。隔离子模块检出最初未获取该提交，本地 fetch 后重试成功；均为测试环境/检出问题，非上游缺陷。

**两层寿命不可混同**：uint64 数组拥有自己的地址值，不等于它拥有地址所指的 RequestOutput 原生对象。RequestOutputPool.release 会逐个 destroy_at 再回池，engine 仍须在调用方读完后延迟释放对象。当前实验不解引用地址、不执行真实析构/allocator，不证明全系统没有双重释放或并发问题；也不建议因此删除防御性 copy。可确认的是：在所核验头文件下，注释所述“静态 vector 覆盖返回数组”不是该构造调用的实际语义。

### 本次只读核验阶段结论

| 改动 | 当前证据 | 不应外推 |
|---|---|---|
| 计时 pair 池 | 原始主机函数 + 完成时机替身复现未消费引用被复用，对照正常 | 不是 GPU race、token 错误或收益实测 |
| 非计时完成标记 | 三轮 decode+flush 替身中先等待再重录 | 不保证全部并发路径 |
| SO 计数 | 选定状态转移模型无漏计，注入陈旧零能漏掩码 | 不是完整 scheduler/grammar 证明 |
| 输出回收快路 | 两套明确 pybind 头文件的原生构造与收集分支检查通过；借用前提需纠正 | 不是原生输出池析构/目标 wheel 验证 |
| profiler 快路 | 已读 diff，依赖私有 profiler API | 未独立运行，不能为全部性能/兼容性背书 |

因此值得保留的技术讨论点是**计时 pair 所有逻辑读者消费前不能回池**；输出注释仅作可附带澄清，不扩成第二个运行时 bug。若用户决定参与原 PR，应只围绕固定 head、最小主机时序反例及限制交流，不宣称已完成设备复现、不另开竞争修复。当前未向上游发言。

本轮也只读确认 [#57](https://github.com/deepseek-ai/DeepInfer/pull/57) 已有作者处理 DP launcher 按序 join 卡住及失败清理（open，head `9a52063a8a0dc86cbbce04fae034e7e037fa3881`）；其 CPU 测试结果是作者报告，未自行复核。后续转为阅读其失败传播/回收契约，不把已有人推进的主题当未认领修复。

## PR #57：等待任意失败，而不是按启动顺序等待退出

2026-10-20 22:30 心跳只读核验 [#57](https://github.com/deepseek-ai/DeepInfer/pull/57)（仍 open、未合并），固定 base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`、head `9a52063a8a0dc86cbbce04fae034e7e037fa3881`。已读完整两文件 diff、原始/新 [examples/_launch.py](https://github.com/deepseek-ai/DeepInfer/blob/9a52063a8a0dc86cbbce04fae034e7e037fa3881/examples/_launch.py)、[测试](https://github.com/deepseek-ai/DeepInfer/blob/9a52063a8a0dc86cbbce04fae034e7e037fa3881/tests/examples/test_launch.py) 与外层 offline/benchmark 调用、内层 engine/WorkerBase 生命周期。此处更新上节的“作者报告”边界：**已在本机独立运行其中 Python3.12 的 CPU 测试与旧实现负对照**，未复核作者 Python3.14 或仓库级 lint 报告。

### 卡住的根因与已有修复的边界

旧 run_dp 先 start 所有本地 DP 进程，再按列表顺序逐一无超时 join，最后才检查 exitcode。若 rank0 仍等 collective、rank1 已异常退出，父进程会一直等 rank0，连“rank1 已退出”也无法及时处理。这是监督者等待顺序的问题；不用在本机真正建立 GPU collective，也能用一个睡眠进程和一个失败进程复现。

| 情形 | 旧实现 | #57 固定 head |
|---|---|---|
| 后启动的 rank1 先失败 | 卡在 rank0.join，尚未检查 rank1 | wait 任意 ready sentinel，join 已结束者并检查非零退出 |
| 发现失败或主进程中断 | 没有统一 finally 清理 | finally 仅收集 pid 非空的已启动进程，先 terminate |
| 子进程忽略 SIGTERM | 不能由旧等待链自行收尾 | 共用一个5秒 deadline，剩余存活者 kill，再 join 回收 |
| node_rank=1，WORLD_SIZE=2，DP=4 | 错误列表用本地 enumerate 下标 | 按真实 DP rank 建字典，可报告全局 DP rank3 |
| 启动第二个进程时抛异常 | 已启动的第一个缺少统一回收路径 | finally 仍处理已启动者，不对未启动者 join |

[Python3.12 文档](https://docs.python.org/3.12/library/multiprocessing.html#multiprocessing.Process.sentinel)说明 sentinel 在进程退出后就绪，可由 connection.wait 同时等待多个进程。这里避免的是按序阻塞；失败列表只包含本次 ready 集合中的失败者，不是完整集群故障清单。共享 deadline 避免每个 rank 各等5秒，但最终 join 未指定超时，故不能把5秒称为任何 OS 状态下的退出上限。

### 实际运行：复用上游测试，不另写一套“同意自己”的夹具

本机 macOS15.4.1 arm64 / Python3.12.13 / pytest8.4.2，仅在隔离 venv 安装 pytest 及其依赖，无 torch、DeepInfer 安装或设备访问。head 检出不改源码/测试，直接运行：

```bash
python -m pytest -q tests/examples/test_launch.py
# 8 passed in 2.20s；独立再次执行 8 passed in 2.23s
```

覆盖正常退出、两种失败先后、模拟第二节点的全局 rank、忽略 SIGTERM、中断、部分启动失败、单 rank 直接在父进程执行。它们使用**真实 spawn 子进程、OS 信号和 join**；worker 的“等 collective”只是 sleep，未执行通信。测试把生产5秒宽限改为0.2秒加速，不能将其耗时作为生产清理时延 benchmark。部分启动测试是真实第一个进程加第二次 start 的注入异常。

独立 base 检出仅取 head 的原始测试文件，不修改旧启动器，再执行同一失败用例和成功对照：

```bash
python -m pytest -q \
  'tests/examples/test_launch.py::test_worker_failure_reaps_siblings[0-1-2-1]' \
  tests/examples/test_launch.py::test_successful_workers_are_joined
# 预期负对照：1 failed, 1 passed in 15.48s
```

失败输出已含 rank1 的 intentional worker failure，但父启动器15秒仍未返回，最终触发测试自己的 TimeoutExpired；成功对照正常。测试 finally 只杀其新建 session 的进程组，未操作其他任务；事后按记录的两个测试 PID 只读查询均已不存在。负对照不是“新 PR 测试失败”。

输入 SHA256：head `_launch.py` `77c69e425a26740ad401d438c00d6e0f1f5b888bad325400ed1d4fcff2699b1e`；base `_launch.py` `d19d14125140385acb62dd1bb4893569f21c99d55eb6e42add0068b713c09486`；两边测试文件相同，`d5b0a764c56a03b2ed79ae73f4b9f50ea8b27e9c0717f8f2177f4a5a89c96920`。初次定位误用了 launch.py 而实际是 _launch.py；默认 uv cache 不可访问，改为隔离临时 cache；现有3.12无pytest，另建 venv；浅克隆间取测试最初需 --update-shallow。均属检查环境/检出准备，未改测试逻辑或上游实现。

### 外层 DP 清理不等于全部 GPU 进程树已经验证

examples._launch 只管理自己创建的本地 DP 子进程。实际 [LLMEngine._init_workers](https://github.com/deepseek-ai/DeepInfer/blob/9a52063a8a0dc86cbbce04fae034e7e037fa3881/deepinfer/engine/llm_engine.py#L368) 还会为 CP rank>0 创建 daemon worker。普通 terminate 不递归终止后代，且跳过退出清理，这是 [标准库的明确边界](https://docs.python.org/3.12/library/multiprocessing.html#multiprocessing.Process.terminate)；**但也不能漏看项目已有的内层保护，直接宣称 CP 一定遗留**：WorkerBase.start_slave_worker_loop 会调用 libc prctl 设置父死 SIGKILL，正常 stop 则另有 barrier→userbuffer.close→destroy graphs/process group→非 driver os._exit 路径。

这些 CP/Gloo/NCCL/Linux 父死信号和跨节点场景均未由上述 macOS 测试覆盖；第二节点用例只在一台机器设置 RANK/WORLD_SIZE。当前正面结论限于：固定 PR 修复了所复现的外层等待顺序/直接子进程收尾问题；不保证全部后代、远端 rank、初始化阶段、共享内存或设备资源清理。已有作者负责该修复，不另造竞争候选或发布 review。下一步先读内层 CP 初始化、RPC 与父死保护的注册时机，按源码与可验证范围划分故障责任，不急于作新缺陷判断。

## 内层 CP 生命周期：父死保护、命令读取与执行完成是三种状态

2026-10-20 23:01 心跳核对 API：main=`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，#57 仍 open、head=`9a52063a8a0dc86cbbce04fae034e7e037fa3881`。以下内层文件在二者中 hash 相同，**不是 #57 引入的改变**。读取 engine/WorkerBase/runtime_config、parallel_state、完整 MessageQueue 相关创建/收发/关闭路径及现有共享内存测试，不访问设备。

### 初始化与保护方向

engine 的 `_init_workers` 为 CP rank>0 启动 daemon 子进程，在父 DP 进程内构造 CP0 driver。slave 的 WorkerBase 初始化按 `ModelRunner构造 → _init_torch → init_model → init_dist → gc → start_slave_worker_loop` 前进，**进入 loop 后第一步**才 `set_pdeathsig(SIGKILL)`，再接收命令。这里 init_model 设置环境、设备/NUMA/seed，不能把函数名误读成模型权重已经加载；实际 `load_model` 是 engine 在 `_init_workers` 返回后另发的命令，对 slave 而言在该注册之后才被取出执行。

来源：[WorkerBase](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/base.py#L58)、[engine 初始化](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/llm_engine.py#L232)。init_dist 本身还创建进程组、MessageQueue READY 握手并初始化设备通信，不能等同于“进程一 spawn 就已受父死保护”。源码只在 loop 入口调用该 helper；没有在 helper 内补 parent PID 前后核对。

[Linux 原始手册](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html)明确说明，注册之后的父线程终止才触发信号，若父线程及相关 subreaper 在注册前已终止，不会补发；“父”是创建该进程的线程。因而这里存在**需另验的注册前阶段/竞态边界**，但本轮未在 Linux 或真实 worker 上复现，不能据此直接宣布 CP 泄漏或提交修复。它保护的方向是“DP parent 死亡→slave 退出”，不是“slave 死亡→DP parent 立即抛错”的反向监督协议；driver 侧是否因后端通信错误/超时退出需另看实际执行。

### 命令传输不是方法完成回执

[run_workers](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/base.py#L290) 先广播 `(method, kwargs)`，随后在 driver 本地执行并返回本地结果；`driver_kwargs` 不发给 slave。[cp_broadcast_object](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/parallel_utils/parallel_state.py#L17) 的 writer 路径 enqueue 后直接返回，reader 路径 dequeue 返回命令。slave loop 拿到命令之后才 getattr 调用方法。

| 状态 | 具体含义 | 不能代表什么 |
|---|---|---|
| ring written flag=1 | 命令槽已发布 | 不代表 slave 已读取 |
| 对应 read flag=1 | dequeue 已取出并反序列化命令，槽可在所有 reader 消费后复用 | 不代表方法执行完成或结果正确 |
| run_workers 返回 | driver 本地方法已返回 | 不是汇总所有 slave 结果；具体方法可另用 collective 同步 |
| sentinel ready | 外层所监督的进程退出 | 不自动表示内层所有 CP 后代都退出 |

“没有控制层完成回执”不等于模型步骤没有同步：barrier、collective、设备依赖可以在具体方法内部建立更强约束。本轮不把异步传输设计本身定为 bug。

### 超时不是一条覆盖所有层的总截止时间

MessageQueue 的 acquire_read/write 可接 timeout，spin 中按 monotonic 检查。但 CP 包装层调用 enqueue/dequeue **未传 timeout**，所以默认 None。`DEEPINFER_RINGBUFFER_WARNING_INTERVAL` 默认60秒，只影响 debug 日志间隔，不会结束等待。`DEEPINFER_GLOO_TIMEOUT` 默认14,400秒用于初始 Gloo process group，不是这段 Python ring spin 的截止时间。

另据源码，READY 的 socket.recv、overflow/remote 的 send/recv 不由 ring 的 timeout 统一覆盖；本轮没有执行 ZMQ 或验证这些通道。不能把队列 API 的一个可选 timeout 参数当成端到端故障恢复保证。来源：[MessageQueue](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/distributed/shm_broadcast.py#L285)、[环境定义](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/environ.py)。

### 正常关闭依赖参与者，而强制终止不等于资源协议已完成

WorkerBase.stop 原始顺序为 `barrier → userbuffer.close → barrier → destroy_cuda_graphs → cleanup_shm_broadcast → destroy_process_group → slave os._exit(0)`；driver 不走 os._exit。共享内存 cleanup 关闭本地句柄，并由 creator unlink；这与“让所有方法执行完”是不同责任。stop 的后续步骤没有逐项 finally：前面的 barrier 或 userbuffer.close 若阻塞/抛错，就不能推断图/共享内存/进程组已清理。这里是失败路径边界，不是本轮真实死锁或资源泄漏证据。

### 已执行的限定主机检查

扩展 [既有主机脚本](../../experiments/scripts/verify_deepinfer_zos_host_flow.py)，不另建重复实验入口：

```bash
python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer --check-worker-lifecycle
```

AST 保留原始 slave loop/stop/run_workers、cp_broadcast_object 与 MessageQueue 五个收发/spin 方法，初始化次序仅静态检查。使用单槽 bytearray、假 monotonic/yield、假 prctl/resource/collective，**不使用真实 shared_memory、线程/进程、Linux 信号、torch 或设备**。main/base 和 #57 head 两个检出均首次通过，验证：

- driver 返回时 slot 仍未读；随后实际 dequeue 将 read flag 置1，但尚未调用 remote 方法；单 CP 跳过广播。
- loop 先调用注册替身再取 stop 命令；slave/driver 分别按完整关闭次序退出/返回；注入 userbuffer.close 异常后，仅第一 barrier/close 执行，异常传播。
- 人为未读槽：显式 timeout=2 在第三个假时间单位抛 TimeoutError；默认包装调用在两次 debug 间隔后仍等待，由测试在第六个假时间单位主动中断，避免真卡住。合成时间不是时延测量。

Python3.9.6/numpy2.0.2；源码 SHA256：base.py `40e6ca29aebd9d475eaad4b86326783f76dba2b73feaa51d86a200b7704a05ed`，parallel_state.py `3565b2449abff4df0df2f979061a73364886c93602218b43c2f86c70f2629801`，shm_broadcast.py `aa461587184bf979acb813ed6ee646665715e3eaaca286a9270a5e4ceb559006`。#54 event/SO/output-copy 与原始主机时序模式回归正常；这不将本轮模拟升级为真实通信验证。

已有 `tests/distributed/test_shm_broadcast.py` 包含真实 ShmRingBuffer cleanup，以及4进程 Gloo/约400MB随机数组广播；已读测试，但本机未安装 torch，也未运行该测试。当前不需要重复创造“无人处理的故障”或改上游方案。下一步优先核验现有共享内存 close/unlink 与 memoryview 生命周期，复用现有清理断言，明确内存回收与方法执行回执的不同边界。

## 共享内存回收：真实本机映射与引用寿命检查

2026-10-20 23:31 心跳核对 main 仍为 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。本轮将上节的 bytearray 替身边界往前推进一小步：运行原始 ShmRingBuffer、PG cleanup 方法和已有 cleanup 测试的两组断言，使用**本机真实 SharedMemory/映射**，不导入完整 DeepInfer/torch。它不是上游完整 pytest/Gloo 套件通过，也不是 Linux/GPU 验证。

### 原始方法、适配点与运行方式

扩展 [既有主机检查脚本](../../experiments/scripts/verify_deepinfer_zos_host_flow.py)：

```bash
python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer --check-shm-cleanup
```

AST 保留 ShmRingBuffer 全类、ParallelProcessGroup.cleanup_shm_broadcast 与 `test_shm_ring_buffer_cleanup` 函数体，仅移除测试 decorators。metadata 初始化的 `torch.frombuffer(...).fill_(0)` 换成 NumPy 同 buffer 的 fill 适配，pytest.raises 由 unittest.assertRaises 绑定；无源码断言改写。SharedMemory 构造包装只记录本次获得的真实句柄，finally 关闭它们、仅 unlink 本次 creator 创建的段，并核对名称已不可再打开。

环境：macOS15.4.1 arm64；Python3.9.6 和隔离 Python3.12.13 各通过完整模式，均 numpy2.0.2；3.12 沿用本轮之前的隔离 CPU venv，仅新增 NumPy，无 torch/设备依赖。每轮创建4个小段，均完成关闭/名称移除检查，无 resource_tracker 泄漏警告。默认沙箱首次 shm_open 被拒，未创建段；获准本机 IPC 访问后再执行，不将权限错误计为上游失败。

### 已确认的几种不同责任

| 检查 | 本机结果 | 证据边界 |
|---|---|---|
| 原始 cleanup 正常/预先 unlink 两例 | 直接调用两次 __del__ 均通过，buf=None、名称不可再开 | 使用原测试断言，不含真实 Gloo 广播 |
| creator 与非 creator 两个映射 | peer 可见 `ring` 数据；peer PG cleanup 只 close，名称仍可附着；creator cleanup 后名称消失 | 同一进程内多映射，无并发/内存排序证明 |
| PG cleanup 重复调用 | 第一次设 mq=None，后续 no-op | 不等于任意失败位置都幂等；未测已有外部 unlink 时的 PG 异常 |
| get_data/get_metadata 普通 context | 退出后原始 yielded view 已 release，访问它抛 ValueError | 不保证调用方没有另存派生 view/数组 |
| 人为保留一个派生 memoryview | 原始 __del__ 的 close 抛 BufferError，名称仍存在；释放派生 view 后重试清理成功；独立 bytes 副本仍可读 | 故障控制是本实验主动持有的额外引用，不是已找到线上引用逃逸 |

[Python shared_memory 文档](https://docs.python.org/3.12/library/multiprocessing.shared_memory.html)区分关闭本实例访问与移除共享段；[memoryview.release](https://docs.python.org/3.12/library/stdtypes.html#memoryview.release)用于释放视图持有的导出引用。故“读标记已设置”“Python 变量出了 with”“共享段已 unlink”“所有映射已关闭”不能互换。本轮没有把派生引用阻止 close 定性为上游 bug，也没有改变所有者/回收方案。

### 先记录平台差异，再调整夹具，不删除断言

首次扩展试验在 owner 请求68字节、peer 附着时触发原始 `shared_memory.size == total_bytes_of_buffer` 断言。检查后发现本机页大小16,384字节；最终模式保留一个请求34字节、实际映射16,384字节的负对照，确认同一 exact-size 断言拒绝该附着。用于所有权/重复清理的独立正向夹具改为**总大小恰好一页**，源码和断言不改，附着映射均为16,384字节。这符合标准库说明的“实际共享块可能因平台页大小大于请求值”，不把 macOS 小尺寸夹具不适配推断为目标 Linux 的部署故障。

源码 SHA256：shm_broadcast.py `aa461587184bf979acb813ed6ee646665715e3eaaca286a9270a5e4ceb559006`；原测试 `f7e18b42fcc1001202d65eb5598c7e6a7170940628de06870fd2201d070d86a7`；parallel_state.py `3565b2449abff4df0df2f979061a73364886c93602218b43c2f86c70f2629801`。未运行4进程/约400MB Gloo 测试、ZMQ、进程异常死亡、跨节点或 GPU/NPU；标准库可自行启动 resource_tracker 辅助进程，不将其称为应用级多进程广播实验。

**阶段判断**：正常所有者回收路径和显式引用控制在本机得到支持；已知平台/引用限制保留，不据此制造新贡献候选。DeepInfer 这段生命周期学习先收束，真正并发/设备证据不能由更多同类单进程检查替代。下一步回到 verl 的只读状态/关联 PR 核验，先查重并确认现有候选是否仍有效，再选择可推进的问题；现有待授权发布事项不自动获得新授权。
