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

下一步：继续 DSpark 总预算选择与 confidence 排序的收益模型，核对变长验证如何影响随机采样位置；真实 CUDA Graph、padding、提前 EOS 与 pool 回收需要目标硬件验证。本机仍仅限源码/CPU，不据主机替身实验宣称 GPU 无空泡或无竞态。
