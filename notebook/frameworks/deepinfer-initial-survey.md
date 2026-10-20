# DeepInfer — DeepSeek 官方推理框架（高优先调研）

> 状态：**调研进行中**（2026-10-20 建立）。仓库 `deepseek-ai/DeepInfer`，Apache-2.0。
> 用户定为本项目高优先调研项（2026-10-20）。
> 事实为 GitHub API 实测（2026-10-20）；技术结论待源码深挖后逐步补充。

## 为什么高优

DeepSeek 官方下场开源其**生产服务所用的推理框架**。与 vLLM/SGLang 这类社区框架不同，这是模型厂商对自己 V4.1 模型最贴合、最"官方"的推理实现。支持 **NVIDIA SM100（Blackwell）+ 华为 Ascend 950 NPU** 双硬件——尤其 NPU 支持对 Ascend 方向意义重大。技术栈全部复用 DeepSeek 自家 kernel 库（FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect/TileLang），与我们的 FlashMLA/DeepEP/Megatron/verl 既有积累高度相关。

## 已核验事实（2026-10-20，GitHub API）

- **仓库**：`deepseek-ai/DeepInfer`，创建 2026-10-16，**今天（10-20）刚开源**——仅 2 个 commit（`initial commit` 10-20 04:09Z + Q&A 文档更新 10-20 09:06Z）。
- **规模**：Python 234 万行 + C++ 27 万行 + Shell。体量大。
- **热度**：约 1547 stars / 112 forks（开源几小时内），22 open issues。
- **硬件**：NVIDIA SM100 GPUs；Huawei Ascend 950 NPUs。
- **软件依赖**：Linux、GCC≥13、Python 3.12、CUDA 13（NVIDIA）/ CANN 9.2（Ascend）。
- **支持模型**：DeepSeek-V4.1-Flash（默认 checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。

## 核心特性（README 列）

- **Optimized kernels**：FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect、自研 TileLang ops。
- **并行**：Attention DP / balanced CP，MoE EP + EPLB（expert parallel load balancing）。
- **DSpark + confidence-scheduled verification**：draft token 加速 + 置信度调度验证（MTP，`DEEPINFER_ENABLE_MTP_VARLEN=1`，`DEEPINFER_USE_NEXTN_DECODE` 控制最大 draft 长度默认 5）。
- **SWA bounded replay**：decoder 只回放有界后缀减少 prefill（`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`，keep 140 tokens）。
- **异步调度**：零开销 batch scheduler，CPU 调度与设备执行重叠（默认开）。
- **Batch-invariant inference**：不同 batch 组成下结果一致（`DEEPINFER_BATCH_INVARIANT=1`，NVIDIA only）——与我们在 vLLM/verl 深挖的 batch-invariant 同主题。
- **确定性采样**：相同输入+采样参数出相同输出（batch-invariant + 固定 seq_seed，NVIDIA only）。
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

- [ ] 读 `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`（部署问答，理解运维定位）。
- [ ] 深读 `deepinfer/engine` 调度器（batch scheduler 零开销异步调度）与 batch-invariant 实现。
- [ ] 对照 vLLM/SGLang 架构，定位 DeepInfer 差异化（尤其 NPU + MTP + EPLB）。
- [ ] 评估可否作为本地/服务器可跑的推理引擎（CUDA 13 / CANN 9.2 门槛高）。
