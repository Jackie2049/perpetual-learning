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

## EPLB历史窗口必须保留记录时的专家身份

2026-10-21 03:32心跳只读核对：main仍为`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，开放条目API共89项、全为PR，无普通open issue；这不是无人负责的问题池。[#107](https://github.com/deepseek-ai/DeepInfer/pull/107)由syang-genai负责，open，固定head `578752f660e68c66412d4d0c08dccfc84c6724dd`。以下是该既有方案的独立CPU学习复核，不是我方新发现/修复或已发布review。另[#93](https://github.com/deepseek-ai/DeepInfer/pull/93)修复profile时临时imbalance配置恢复，与历史计数归属不是同一问题。

**物理槽位不是稳定的逻辑专家ID。** [基线EPLB](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/__init__.py#L147)每次记录本rank物理槽位计数，到重排时才用当前physical_to_logical_map解释整个窗口。重排改变槽位归属后，旧行仍需按记录时的映射归集，不能按新映射重解释。[#107的记录方法](https://github.com/syang-genai/DeepInfer/blob/578752f660e68c66412d4d0c08dccfc84c6724dd/deepinfer/model_executor/eplb/__init__.py#L150)改为入窗时scatter-add到逻辑专家轴；后续只clone/all-reduce，不再用新映射解码旧行。环形覆盖前先zero，避免新计数累加到已过期行。

触发频率是条件，不是所有配置都立刻出错。构造器在启用重排时要求window_size≤interval；普通周期在两次重排间覆盖整个窗口，首次提前到1/4周期时也没有更旧布局的有效历史。NORMAL引擎启用after-prefill重排后，prefill把计数器置到interval−1，下一非prefill步可提前触发，旧布局行可能尚未覆盖。

### 原始CPU张量路径与受控重排

```bash
python3 experiments/scripts/verify_deepinfer_eplb_window.py /path/to/DeepInfer
```

[脚本](../../experiments/scripts/verify_deepinfer_eplb_window.py)读取固定Git对象，不切换checkout。Python3.12.13、torch2.9.1、pytest8.4.2、NumPy2.0.2在新建独立环境运行，所有张量均为CPU；torch版本不是DeepInfer官方锁定环境。执行原始构造/记录方法与从重排方法中隔离的CPU解码块；手动替换一次合法槽位映射代表已完成的placement变化，实际_rearrange_experts/shuffle/collective由回调隔离，未执行设备重排。

窗口4、interval8、EP大小2/rank0、1层、4逻辑/8物理专家的控制结果如下。原始counter使首次重排在记录2后发生；表中记录数是本夹具调用次数，不是测得的在线吞吐或时间。

| 固定版本 | 触发配置 | 重排回调记录位置 | 两次历史归属检查 |
|---|---|---|---|
| 基线 | 普通周期，连续decode | 2、10 | 都与记录时映射一致 |
| 基线 | prefill/decode交替，启用after-prefill | 2、4 | 第一次一致，第二次旧行被新映射误归属 |
| #107 | 普通周期，连续decode | 2、10 | 都一致 |
| #107 | 同一after-prefill日程 | 2、4 | 都一致 |

夹具第一次回调后将本rank的槽位映射从`[0,1,2,3]`换为`[3,1,2,0]`。每次槽位计数均为`[100,1,1,1]`，故正确四行历史应是前两行expert0热、后两行expert3热；基线第二次把四行都归给expert3。把这两份输入送入未修改的[CPU rebalance策略](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/policy.py#L73)，得到逻辑专家副本数分别为错误输入`[1,1,1,5]`、记录时oracle输入`[3,1,1,3]`。这表明所构造的统计偏差能改变策略决策，不表明线上必定振荡或吞吐损失多少。

另直接执行#107的3个原始测试函数体，覆盖跨placement历史、ring覆盖和rank1切片；原始fixture使用假进程组，本轮只去掉包导入，断言不改。基线原始policy测试再通过26个CPU参数例（结构8、确定性8、边界分布4、输入不变1、贪心目标/tie5）；移除测试decorators后保留原参数数据直调，不等于完整pytest/conftest或CUDA测试通过。

窗口内存轴由local_physical改成logical，字节数为`window × layers × axis × int32字节数`，不能无条件称为免费：相对倍率为`logical/local_physical`，实际取决于EP大小与冗余数。正确归属也不保证shuffle失败时安全、各rank观测同一代映射或策略全局最优；这些不在本轮CPU路径中。

源SHA256：基线EPLB `1b30c01b36cfc711281bd729ab5cbf09826b4ba22359573f46d96c10ea6e7260`；#107 EPLB `bb22f49f95bf2eca6962ebe8557ce6abeeaf24781477f5ff021d257a2253b618`；window tests `e4a3e151db8ede534a279106af54ed9f1bd32165d0f9466e16202586ce980a59`；policy `52241603827a1a1d19a395d48f5d618a7336fd1ceee631127e6782c644f736dc`；policy tests `1671e3af1cb5ac293307dc37e288efe61da416450f2d677a94163a3e28426ce5`。生产者与清零语义补见下节；不新增竞争PR或把受控映射替换说成真实权重shuffle验证。

## 专家计数的来源与清零时序

**expert_count是每层本地物理专家的路由接收工作量，不是请求数或已接受输出token数。** 以下固定DeepInfer基线774ce244的NVIDIA路径；本机checkout为8b161497，经Git diff确认两版本deepinfer/源码无差异。没有将Ascend替代后端或任意安装版本视作同一路径。

[MoE gate](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/mlp.py#L379)把逻辑专家映射为物理副本；dispatch和mega_moe_forward都把manager.layer返回的expert_count视图传为cumulative_local_expert_recv_stats。该视图与balancer计数共享存储，不是复制出的日志。模型初始化还把已加载的MTP MoE层纳入manager；[sample](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/deepseek.py#L383)在sampler调用返回后（decode可在其中提交draft层执行），才调用manager.step；这里是主机调用顺序，不是所有设备工作已完成的断言。因此不能把一行窗口等同于一个请求、固定数量token或最终accepted-token计数；main/MTP是各自的layer行，不是先混成一层。

固定子模块提供更具体的含义：[DeepEP 93eb6eb dispatch](https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L93)对每个非负topk专家选择加1，而rank通知计数另做去重。一个token路由到同rank两个专家，可以贡献2次专家工作，却只贡献1次去重rank通知；EPLB的本rank统计sum(expert_count)采用前者，不是通信token去重数。普通dispatch在对齐前保留sum并atomicAdd到累计统计；[hybrid路径](https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L269)也用未对齐count，而非aligned_count。

[DeepGEMM 1fc9c66 BF16](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh#L580)与[FP8/FP4](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L644)从接收计数读num_recv_tokens，清工作区时将它累加到统计；ceil_div得到的block数另用于回收。因此所读分支的统计不含expert-alignment/block补齐数量，但这不独立证明整个gate/有效mask/图批次padding链正确。子模块只读源码，未编译或运行内核。

### 窗口与负载统计是两条消费路径

[balancer.step](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/__init__.py#L131)在EP>1时依次执行重排窗口更新、负载统计、原地zero_；默认跳过NORMAL prefill重排，只是第一项提前返回，不跳过后两项。负载统计按layer对本rank专家计数求和，累计到自己的stat interval；它与重排window不是同一个窗口，不能从“没有入重排窗”推断“计数被保留到decode”或“没有进入负载统计”。

| 条件 | 重排窗口 | 本地负载统计 | 计数张量 |
|---|---|---|---|
| EP>1，重排启用，普通decode | 入窗 | 消费当前计数 | 消费后原地清零 |
| EP>1，NORMAL prefill且跳过重排 | 不入窗 | 仍消费 | 仍清零 |
| EP>1，interval=0 | 重排关闭 | 仍消费 | 仍清零 |
| EP=1 | 整个step返回 | 不消费 | step不清零，不把它误写成所有配置每步清零 |

[Worker顺序](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L115)是提交当前forward、处理前轮post_funcs、再提交当前sample/EPLB step。延迟post_func取走的是负载比率报告；窗口更新与清零已经在sample路径提交，不是等post_func才发生。普通MoE调用dispatch/combine的current_stream_wait，表示代码请求了相应流依赖；本机未验证设备完成、跨stream可见性或跨rank同代映射，不从Python先后关系推断这些性质。

```bash
python3 experiments/scripts/verify_deepinfer_eplb_window.py /path/to/DeepInfer --check-step-lifecycle
```

既有独立Python3.12.13/torch2.9.1 CPU环境新增8组原始manager/view/step/local-stats检查通过：上表4条件分别运行基线与#107；覆盖2层反向global/local索引、共享视图、先消费后原地清零。负载统计interval固定128，单次调用不触发all_gather/CPU异步拷贝；进程组为替身，不执行producer kernel、模型sample、实际重排或collective。原窗口/受控日程/26个policy CPU例也通过。至此收束EPLB身份与计数链学习，不据此提出新缺陷或竞争修复。

源码SHA256：deepseek.py `881fe2839b34d191ffd50ca6d5b7b57443b43ed8d811e204e4d48164f31f613a`；mlp.py `2541bcc83b178a9e2f18b387b301d60111d6d603e18e2d31df38196b03d290dd`；model_runner.py `9e2e18d136c9e0c9f23905ff994e9d23afaa94aa7a133d425e7fc1fc069033ef`；worker.py `ad32e595d9ab289dff1a7709136431ac60dae709e9e5123ae42eb0ebf7c30cbb`；sampler.py `81205a12e860041ffa904321b852db2d048bc038bb07d494dbb78b44b522d9a9`。子模块以DeepInfer gitlink固定提交核验，不把当前main或运行环境默认为这些版本。

## Chunk checkpoint消费与预留尾段

**切分候选不是已经接纳的调度；checkpoint的消费时机与KV资源回收是不同问题。** 2026-10-21 04:32心跳核对三个PR均open、共同base为774ce244：[#108](https://github.com/deepseek-ai/DeepInfer/pull/108)由syang-genai负责，head `ab47ee4c62ab704431db1b06a66f01c57619faef`；[#95](https://github.com/deepseek-ai/DeepInfer/pull/95)与[#98](https://github.com/deepseek-ai/DeepInfer/pull/98)由Phanthom-Mekat负责，head分别为`2a5db94adbc70f62ea205306efa677fe3331c954`、`6a934611d538bcce54434c73a886459bca36b021`。不是无人处理候选，本轮不发布review或竞争修复。

| 方案 | 所改问题与交叠 |
|---|---|
| #95 | aborted waiting请求资源释放、state id重置、已持有KV的chunk继续推进、水位线与preemption排队 |
| #98 | 保留上述核心，调整preempted插入位置以避开不可调度请求；两者差异不能只称为改名 |
| #108 | 推迟checkpoint消费、区别显式checkpoint与预算分块开关、保留lookahead尾段；所测sizing/图片方法在#95/#98仍逐字同基线 |

[基线_schedule_prefill](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L460)先按预算和checkpoint选长度，立即pop checkpoint，再检查图片边界、chunked_prefill和最小lookahead；稍后break时，未接纳序列也可能已失去checkpoint。预算分块还可能留下短于lookahead的尾段，使下一尝试的seq_workload检查失败。[#108片段](https://github.com/syang-genai/DeepInfer/blob/ab47ee4c62ab704431db1b06a66f01c57619faef/csrc/engine/scheduler.hpp#L477)先只读选点，在通过长度接纳检查后消费，并缩短前块以保留尾段。这里的“提交”不是任意后续失败都能回滚的事务：检查status、KV分配、真正执行与process_outputs都还在后面。

### 局部原始C++检查及其边界

```bash
python3 experiments/scripts/verify_deepinfer_chunk_sizing.py /path/to/DeepInfer
```

[脚本](../../experiments/scripts/verify_deepinfer_chunk_sizing.py)从固定Git对象提取原始sizing片段、_is_inside_image_pad_region与_adjust_chunk_size，使用Clang/C++20在本机编译运行。Sequence存储、已有调度上下文和“接纳后cached增加”是夹具，不运行真实Sequence/Scheduler；TORCH_CHECK仅替换为本地异常，忽略诊断格式化。截取到status检查/KV分配之前，不执行完整队列、内存分配、Gloo或跨rank投票。

10组成对控制通过，部分关键结果如下。缓存位置序列是夹具按接纳q推进的轨迹，非真实engine实测。

| 控制输入 | 基线片段 | #108片段 |
|---|---|---|
| prompt20、checkpoint5、chunked_prefill关闭 | 首次未接纳但丢点，重试直接20 | 先5，再20 |
| prompt10、预算8、lookahead4 | 先8，余2导致下次检查抛错 | 先6，再10 |
| 图片覆盖[0,8)、checkpoint4、已有请求占用预算 | 图片收缩到0后退出，点已丢 | 同样退出，但保留点 |
| 预算全被前请求用完，未来checkpoint6 | 保留点 | 也保留点；不是旧版必败用例 |
| 图片覆盖[3,8)、checkpoint5 | 预算够时扩大到8；预算6时收缩到3 | 相同；不保证在图片内部精确切到5 |

其他控制包括多个checkpoint、普通预算分块、过短checkpoint与过短尾段。#108允许把违反lookahead的checkpoint当不存在，并在覆盖它的接纳块消费掉；图片约束也可能改变实际边界。因此“保留显式点”不等于保证每个不合法/图片内部点都能精确满足。不能把所有新增测试都算作旧版失败，或把这些局部结果当作DP挂起已复现。

完整测试尚未运行。原始Scheduler头文件的本机syntax-only尝试先因SDK标准库搜索路径缺memory失败；显式SDK路径后，Clang仍报object_pool.hpp的dependent cast模板解析，以及Linux特有PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP/pthread_mutex_clocklock未定义。torch2.9.1 macOS有Gloo不等于官方Linux/torch2.13扩展能在本机原样编译。本轮没有改兼容层，未构建完整扩展，也未执行#108的5个原始Scheduler/Gloo测试；作者PR正文亦注明它们尚未在其机器执行。这是本机验证限制，不据此认定上游构建缺陷。

本机为macOS arm64，脚本在Python3.9.6和3.12.13均通过；torch2.9.1仅用于完整头文件/Gloo可用性检查，片段执行不依赖torch。scheduler.hpp SHA256：base `c14de4d5341170227b82ddf2d146d9c6a337937885a6886237aab5a7238da5db`；#108 `63c4c02b600cf2e321f4ea6540de805d9d406d00757e461abb8d5a2271589ef3`；#95 `7a15e8cf4cb9bc6901d5361e7342406f4926ace5ca55f43faf6977cc31f4cb41`；#98 `b6f3cc0b8f38d8f9145d53ab3c3872d20745f16fcebfe969efc2203973cfe3e3`。下一步只读核对Python checkpoint入口、token坐标与cached-prefix fork调用契约，不扩写为未经完整调度验证的新修复。

## Checkpoint坐标与cached prefix fork契约

**Checkpoint表示序列中的绝对缓存位置；精确停在这里还依赖切分点合法和调度可推进。** 固定基线774ce244与#108 ab47ee4c；下述Python入口、benchmark及Sequence绑定在两版本未改动。2026-10-21 05:02心跳接续上一节，不把局部片段检查扩大为完整engine或fork验证。

[LLMEngine.add_request](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/llm_engine.py#L472)先展开图片占位符，再构造Sequence，最后直接赋值chunk_checkpoints；所读路径没有替调用方排序、去重或换算图片展开前位置。[绑定](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/pybind_sequence.cpp#L126)将Python列表赋给C++ deque，num_cached_tokens对应cache.num_tokens。调度器只看队首并以checkpoint−cached算本块长度，因此递增的展开后token边界才符合逐点推进的用法；原始占位符下标或每块长度不能直接代入。Engram compressed tokenizer逐token映射、构造时断言长度相同，不把这个位置变成“压缩后更短序列”的下标。这里也不是KV block编号、CP本地行号或YoCo compact后query行号。

### Benchmark如何建立可fork的前缀

[benchmark](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L364)设置单个绝对checkpoint后循环step，每步断言cached不超过目标；调用前cached不超过目标、循环正常退出时，两条件共同保证相等。入口未检查这个初始大小关系。它不是通用的限时等待器：没有在这个helper内检查完成/取消、等待截止时间或无进展次数；也不会自动恢复checkpoint。目标已达到时仍先写入checkpoint但不step，调用方应理解这个残留，而不是视为已经消费。

正常调用由`cached_tokens=context_length−non_cached_tokens`得到目标，参数要求non_cached_tokens至少2；创建engine显式把lookahead设为`min(1, DEEPINFER_USE_NEXTN_DECODE)`。因此不能拿上一节lookahead4、尾段2的反例直接判此benchmark默认调用失败。图片模式还将目标及预期chunk边界传给[图片模板生成器](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/_prompts.py#L208)，按展开后的区域放整张图片、文本补齐；容量不足会断言，不是允许任意图片跨界再由checkpoint强切。它建立了“图片不跨这些点”的输入条件，不能反过来证明通用add_request可精确满足任意图片内部checkpoint。

### Fork保留的内容与重算位置

[Scheduler.fork_seq](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L1061)要求已有计算前缀、非空KV block table且不请求prompt_logprobs；复制完整token数组而非只截取缓存前缀。设源序列当前总长N、缓存数C，新分支继承`min(C,N−1)`个缓存token，至少留下末token需要处理。

| 源缓存条件 | 新分支缓存与入口 |
|---|---|
| 0<C<N−1 | 继承C，进入waiting，继续prefill；不注入sp_token golden token |
| C=N−1 | 继承N−1，进入prompted；golden token置为调用方传入的sp_token |
| C=N | 缓存回退到N−1，同样进入prompted；不能套用benchmark的缓存数相等断言 |

benchmark在固定长prompt的部分prefill后fork，未缓存后缀至少2，落在第一行，所以其`fork.cached == source.cached`断言与通用fork的末token保留并不冲突。复制的block table（或调用方给的blocks）会pin引用，启用state_manager时另继承并pin state_id；不等于在这个函数内复制所有物理KV内容。新schedule只设置schedulable与branch_id，没有复制源checkpoint deque；显式中间切分要求需要调用方为新分支另行表达。这些是源码契约，不是fork引用计数、COW或实际KV一致性已测试。

### 原始Python helper与片段轨迹控制

```bash
python3 experiments/scripts/verify_deepinfer_chunk_sizing.py /path/to/DeepInfer --check-caller
```

复用原脚本，以AST提取未改写的prefill_cached_prefix，让替身step依次消费原始C++ sizing片段生成的缓存轨迹。新增8组为4条件×base/#108：缓存4到目标10、预算4时经过8再到10；图片[3,8)在边界8成功；目标5落在图片内时片段扩大到8、原始Python断言按预期触发；已在目标5时零次step且仍留checkpoint5。两版这4条件结果相同，不能算作#108新增失败复现。原10组成对片段控制回归通过。

Python3.9.6与3.12.13均通过；新增检查不依赖torch。Sequence和engine.step仍是替身，cache推进沿用片段夹具的立即增加规则，不执行真实输出提交、fork、模型、分布式或设备。chunk学习先收束；后续只读追MTP接受数如何把乐观缓存长度提交为最终长度，不重复增加同类替身来替代完整Scheduler验证。

固定源码SHA256：benchmark.py `a4e0b114b4173b9d2124cdabfedc54d3dd5cd81d269d7054c48f9bac780eb9f2`；llm_engine.py `a636d60ca5711f6b7c21e68370c3bd9c2de6fc3f4439a7429ca4f572f0edeebe`；pybind_sequence.cpp `872775774022bb27fd4b1fade37ec9f6fe9ec4c3051844206e3ec36c8103a9ce`；sequence.hpp `67ef6ea70ee31a519fd632ee2d7be4710a6d796f786cf336dd3d3f06f7fbe16b`；_prompts.py `c8025c1542b62f5bc1f5065387fead9511861a873b004718a104004959f2e1ef`。

## MTP验证与缓存提交的计数边界

**验证接受数不是最终追加数，调度预留也不是已提交缓存。** 固定774ce244源码，2026-10-21 05:32心跳接续；所读scheduler、Sequence、sampler与输出路径同本地8b161497版本。以下区分逐请求验证长度q、连续接受的draft数量a，以及停止检查后真正追加到序列的数量t；不是完整Scheduler或物理KV正确性证明。

[采样finalize](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L321)按请求的连续接受前缀生成flat verified token数组：未接受行置−1，并单独取第一拒绝行或最后fresh行的sampled token。last_valid_offset是final_row−ftp，即接受draft数，不包含这个最终采样token；GPU侧下一轮长度修正读取的是它。这里没有在主机上按EOS/max_tokens裁剪后的t。

[主机post_process](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/llm_engine.py#L717)先process_MTPInfo再process_outputs：前者发布下一轮draft、移动confidence历史；后者取本轮flat接受前缀，追加接受draft与最终采样token，并逐个检查EOS、总长和输出上限。接受向量可以长于最后实际追加的部分；seq.mtp.last_output_token也先保存最终采样token，提前停止时不必等于账本最后token。不能把这些字段当作已经向用户输出的完整序列。

### 缓存账本怎样落到实际追加数

[process_outputs](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L927)先以q做一次乐观推进，保存previous_num_tokens=C；decode在追加完成后改写为C+t，而不是从已经推进的值再加t。若处理前cache为N−1、序列长N，处理后cache为N+t−1，仍留下一个末token待后续forward。这个等式依赖该初始关系及正常decode分支，不适用于任意人工注入的缓存状态。

| 路径与控制输入 | 最后缓存及序列长度 |
|---|---|
| decode N10/C9/q4，接受2个draft再采样1个token | 先乐观13，再回填12；序列长13，t3 |
| 同样q4，3个draft均通过，但第一个就是EOS | 接受向量仍长3，实际仅追加1个；缓存10、序列长11 |
| 最后prefill chunk，N10/C6/q4 | cache10、输出后序列长11；不按t1回填成7 |
| 非最后prefill chunk，N10/C0/q4 | cache4、序列长10，不追加输出，重新进入waiting |

SO账本status=−1时，即使本步没有新的SO状态上报，循环也以占位0追加一次并标记FINISHED_SO_FAILED；它不是普通EOS检查路径。此处只描述序列账本，不能据此断言外部RequestOutput也返回0，后者还依赖采样mask及输出formatter的输入前提。

### 在途预留与失效结果

[schedule](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L696)用cache加既有inflight加本轮q构造metadata，再按nextn+1预留本轮在途容量；实际q与这个上界可以不同。finish_inflight_step在只剩最新预留时清零，否则保留最新一轮预留。这是两轮流水线的保守记账，不是任意深度任务队列或资源事务。

已经finished的请求，以及decode在途被抢占、重置后处于WAITING的请求，都跳过cache推进和token追加；flat_offset仍按该行q前移，避免下一请求读取错段。process_MTPInfo对WAITING decode也先跳过，不把已复位的process计数推进。_preempt是否允许重算还取决于stop_preempted_request与SO状态；不能把所有抢占都叫作重算。

[输出formatter](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/outputs.hpp#L196)用pushed_tokens_before_finish限制逐项输出，只有尚有余额时才输出最终sampled token；最后发出的项记录draft_tok/accept_tok。这两个统计字段由验证长度和接受前缀构造，并不重新等于实际追加的t，提前EOS时尤其不能据它们恢复最终序列长度。这部分为源码核对，未纳入下述C++执行。

### 原始C++记账循环的限定控制

```bash
python3 experiments/scripts/verify_deepinfer_mtp_commit.py /path/to/DeepInfer
```

[脚本](../../experiments/scripts/verify_deepinfer_mtp_commit.py)从固定Git对象提取完整逐请求process_outputs循环、_check_stop、KvCacheState与SequenceScheduleState定义，Clang/C++20运行12组控制：draft接受0至3各1组，draft中EOS、输出上限、已存在SO失败各1组，两类prefill各1组，finished/重算WAITING跳行各1组，两轮预留退场1组。包含混合q2/q4时的flat offset、真实缓存/在途primitive reset，以及decode必须基于previous缓存回填的检查；第一轮即通过，Python3.9.6与3.12.13重复通过。

Sequence存储、tensor访问、SO上报容器、队列和slot释放为替身，Engram映射取identity；TORCH_CHECK只检查条件、丢弃格式化参数。不编译完整扩展，不执行采样/ref/kernel、process_MTPInfo、真正_preempt、输出formatter、设备KV写入/回收或collective；也不向任意非法flat buffer输入提供安全保证。原始Worker主机控制流回归通过，仍不能替代真实事件/stream验证。下一步先核对SO失败占位与对外token身份的前提，避免把账本规则误当作API输出全链已证实，不预设新缺陷或对外提交。

源码SHA256沿用前节sequence.hpp及scheduler.hpp；outputs.hpp为`9ef2f5a37dd07b03fe7fac7f92cbbf1dfc056ff4a0c0fbb9b7a384149f71bf6f`，tilelang_ops/sampler.py为`cef9d2dda13f24fed56ca9bef45fdffa883edc63208d54e1b83ead9358271304`。本机macOS arm64，无GPU/SSH/VPN操作，无AReaL变更。

## SO失败状态与输出token身份

**序列账本的失败占位0不是统一的对外失败token。** 固定774ce244，2026-10-21 06:02心跳接续；局部原始C++循环组合在注入的采样payload下显示：SO失败分支向Sequence追加0，但formatter仍从接受draft或最终sampled token取对外token，同时保留失败结束标记。这是有条件的主机契约差异，不是完整engine可达性、线上复现或已定性的缺陷；也未确认失败响应必须满足“token等于账本占位”的外部规范。

### 失败状态与采样输入分别传播

[_get_so_mask](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py#L301)在全部metadata状态非正时返回None状态数组；活跃SO的matcher缺失或已aborted时，把该行状态置−1且不为它填mask索引，混合batch仍处理健康行。它返回的是mask/状态，不在这个函数里将sampled/verified token数组统一改成0。[forward](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py#L133)继续走采样并把so_status放进结果；token内容仍需由真实采样路径决定，不能由这里的状态数组直接推断。

process_outputs收到新的−1报告时更新账本；本步状态数组为None也不会清除既有失败状态。进入失败分支后追加0、pushed=1；进入函数前已经finished则走更早的跳过分支、pushed=0，两种状态不可混同。最后prefill的激活异常也会直接置Sequence finished，不能把它当作所有SO失败都进入占位分支的证据。

### 原始formatter怎样选择失败输出

[formatter](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/outputs.hpp#L196)根据pushed限制输出数量，但从传入的accepted/main数组选token，没有根据SO失败把它替换为序列末尾0。8组局部组合检查的关键结果如下；采样payload是显式注入，不是模型产生。

| 条件 | Sequence本轮追加 | formatter输出 |
|---|---|---|
| 健康decode，接受21/22、main99 | 21/22/99 | 21/22/99，正常对照 |
| 接受前缀首项EOS11 | 11 | 11，finished；accept_tok仍可为4，不等于追加1 |
| SO账本已失败，接受21/22、main99 | 0 | 21，finished且带SO失败原因 |
| SO账本已失败，无接受draft，main99 | 0 | 99，包含q1无MTP控制 |
| 本步新报告SO失败，原账本健康 | 0 | 21，同样保留失败原因 |
| 所测decode进入循环前已finished | 无 | 无逐请求输出项；不推广到ignored_seqs等其他路径 |

另一组输出上限控制也保持普通token身份一致。原始RequestOutput对象保留finished及`xgrammar runtime errors or timeout`原因，不是把失败伪装为成功；但[离线LLM._run_engine](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/entrypoints/llm.py#L168)会先收集每项，decode_sequence开启时直接把其token_id交给tokenizer，并没有在这里按失败原因过滤。错误结果是否该暴露这些token仍是API语义问题；自定义在线服务并未由该离线入口覆盖。

### 限定执行与阶段结论

```bash
python3 experiments/scripts/verify_deepinfer_mtp_commit.py /path/to/DeepInfer --check-output-identity
```

复用前节脚本，保留原始formatter逐请求循环、RequestOutput与finish reason定义；将c10 optional适配为std::optional、原对象池替换为heap持有者，tensor访问和Sequence仍为替身。不执行完整formatter的tensor/shape入口、ignored分支、真实池或pybind。新增8组native formatter/提交控制，另以原始Python方法体执行4组SO mask控制与2组离线收集控制，Python3.9.6和3.12.13均通过；原12组C++记账控制回归通过。

SO mask的设备/tensor/matcher为替身，离线入口的engine/clone/tokenizer也是替身；两部分未与真实采样或服务连接，未生成grammar、GPU mask或模型token。结论收束为“不能用账本占位0代表API输出token”。不据此开新候选或竞争修复，不在此基础上反复增加替身充当完整复现；若后续考虑贡献，需先确认错误响应语义、真实失败payload与既有PR覆盖。后续回到有界上游变化核对，选择未重复课题。

新增固定源码SHA256：Python sampler.py `81205a12e860041ffa904321b852db2d048bc038bb07d494dbb78b44b522d9a9`；entrypoints/llm.py `069a5333d9f47047ccc87584792627dfa6eee9cf93bdc046107cbd0218a6bada`。其余沿用前节；没有改上游源码、AReaL或GPU环境。

## SamplingParams 构造校验与测试组合

2026-10-21 只读 [PR #119](https://github.com/deepseek-ai/DeepInfer/pull/119) 与 [PR #121](https://github.com/deepseek-ai/DeepInfer/pull/121)，读取时均 open，作者均为 akash6murali。固定公共基线 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，实现 head `bc668873e5520aa309c709a9b4bfc2935519779a`，测试 head `2afdb3d2d8ed1fb3a83a0169485aa25d281196e8`。这是已有作者方案的学习，不是我方候选或完整 PR 验收。

### 实现改动与测试依赖

#119 改 `sampling_params.py`，#121 只新增两个测试文件，不包含 #119 的实现。不能把两个 open 分支当作已经合并的一套代码。对同一份原始 SamplingParams 测试分别提供不同实现后，本机结果如下：

| 实现来源 | `max_tokens=0` | `logprobs=True` | 53 项原始测试 |
|---|---|---|---|
| 固定基线 | 校验前改为 1 | 兼容参数被丢弃，不发警告 | 51 通过，2 项预期失败 |
| #119 固定 head | 删除改写，由已有范围检查抛 ValueError | 发出 DeprecationWarning，仍无该开关的功能 | 53 全部通过 |

两个基线失败恰是 `test_zero_raises` 和 `test_logprobs_true_emits_deprecation_warning`。PR 正文标称 47 项，但固定测试文件在本次 collector 中实际是 53 项。表格不是完整 #121 测试集或上游 CI 结论：本轮未执行 config 文件、原生包初始化或上游 conftest。

警告是行为提示，不是自动启用 logprob。`top_logprobs` 控制请求的 top-k 返回数量；不能把弃用 `logprobs` 理解成新增返回通道。DeprecationWarning 的可见性还取决于应用的过滤策略，发出不代表任意调用环境默认显示，见 [Python warning 文档](https://docs.python.org/3/library/warnings.html#warning-categories)。

### 校验在 greedy 改写之前

[构造器](https://github.com/deepseek-ai/DeepInfer/blob/bc668873e5520aa309c709a9b4bfc2935519779a/deepinfer/sampling_params.py)先保存参数、生成或归一化 seed，再执行 `_verify_args()`，最后才处理 `temperature < 1e−5`。因此：

- 温度为零且 top-p/top-k 合法时，温度改为 `SAMPLING_EPS`、top-p 改 1、top-k 改 −1；并非把最终保存温度设为零。
- 温度为零也不能绕过 `top_k=0` 或 `top_p=0` 的检查；它们在 greedy 改写前就被拒绝。
- 温度恰为 EPS 不满足严格小于条件，不覆盖 top-p/top-k。是否最终表现为确定性 greedy，仍须按实际 sampler 后端理解，不能仅凭保存的温度认证设备输出。

seed 的 `None` 随机发生在构造时，整数先截断为 uint64；原 seed 范围检查不是拒绝任意超范围原始整数的保证。离线 `LLM._add_request` 会 deepcopy 已构造参数，不重新生成 seed。已有 seed/确定性边界见前面的[固定种子说明](#源码核验确定性是跨层约束)。

### Python 构造器不是完整输入模式

这两个实现都保留同样的局部比较检查：temperature 只检查负值，top-p 用 `(0,1]` 的组合比较，top-k/max_tokens/top_logprobs 检查数值范围，没有统一的有限数与精确类型验证。额外 8 组本机观察在两个实现中一致：

| 输入 | Python 构造结果 |
|---|---|
| `temperature=0, top_k=0` | ValueError |
| `temperature=0, top_p=0` | ValueError |
| `top_p=NaN` | ValueError |
| `temperature=NaN` 或 `+Inf` | 接受 |
| `top_k=1.5` | 接受 |
| `max_tokens=1.5` | 接受 |
| `top_logprobs=0.5` | 接受 |

“接受”仅指该 Python 构造完成，不表示参数可被完整引擎或 kernel 正确消费。[C++ SamplingScalars 与 StoppingCriteria](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/sequence.hpp)仍会对 Python 属性执行 double/int/uint64 转换；[prepare_sample.cpp](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/prepare_sample.cpp)再填设备输入。这里没有运行 pybind 转换或 sampler，因此不推断具体下游异常、截断规则或新的可行动缺陷，也不扩写作者修复。

### 字段保存不等于字段效果验证

原测试对 `so_args` 和 `golden_token_ids` 只验证保存。grammar schema、trigger/bound 约束在其他组件才消费，这些构造测试不能证明结构化输出生效。

`golden_token_ids` 也不能理解成只记录对比分数而绝不影响输出：源码中 `LLMEngine.add_request` 将其传入 Sequence，prefill 的 `_prefill_golden_gen` 在采样后调用 `_update_golden_tokens`，后者用非 −1 的 golden 值替换 sample result，再查其 logprob。这个覆盖与前面的概率计算是不同阶段；本轮只读该路径，没有执行完整 golden 生成、MTP 或设备逻辑。来源见固定基线 [llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/llm_engine.py) 与 [sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py)。

### 本机复现范围

[脚本](../../experiments/scripts/verify_deepinfer_sampling_params.py)原样执行固定 Git blob 的 `sampling_params.py` 与 #121 的 `test_sampling_params.py`。只将 `deepinfer` 包根设为空命名空间，避免触发硬件探测和扩展初始化；参数实现和测试体没有 AST 修改、torch/验证规则替身或源码补丁。通过 pytest collector adapter 收集原测试模块，关闭第三方插件自动加载、上游 conftest 和 pytest cache。诊断路径由 linecache 提供，源码 checkout 中没有写入虚拟测试文件。

```bash
python experiments/scripts/verify_deepinfer_sampling_params.py /path/to/DeepInfer --implementation base
python experiments/scripts/verify_deepinfer_sampling_params.py /path/to/DeepInfer --implementation pr119
```

Python 3.12.13、torch 2.9.1、pytest 8.4.2，CPU。baseline 的两项 pytest failure 被脚本按具体 nodeid 核对为预期对照，未标 xfail、未修改测试；#119 的 53 项全过。各自额外 8 组构造边界检查通过。没有运行 #121 的 CacheConfig/ParallelConfig/SchedulerConfig 测试、完整包/CI、C++ 绑定、模型或 GPU/NPU，也未认证测试正文所称的所有覆盖范围。

SHA256：基线 SamplingParams `f800034efdebe1b21620660a51fa9122d3d7f138d8e60cd738f33d643c3116f8`；#119 SamplingParams `b2de2bf36c32add1808824cddb05191bf5d45979efd6b0b14579d6c74ad46f04`；#121 原始测试 `c56598760b78ac163a01882d0bb975dec0c9eec9f32911ed0bc9bfd8ffec6c51`。
