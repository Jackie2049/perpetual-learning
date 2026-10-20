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

SO 计数部分已读到 schedule 统计、最后 prefill chunk 0→1 增量、sampler 快路和账本刷新；尚未对异步状态转移做穷举，不能仅凭注释确认“永不低估”。下一步先做该计数契约的主机核验，再形成是否值得参与 #54 技术讨论的证据判断；任何上游评论仍需用户拍板，不另开竞争 PR。
