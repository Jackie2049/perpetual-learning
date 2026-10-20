# DeepInfer 架构源码调研（2026-10-20 综合精读）

> 基于对本地浅克隆（`/Users/jackiemac/workspace/DeepInfer`）的全面源码精读（调度器/引擎、模型执行器、NPU/分布式/入口 三路并行），综合三份 Explore agent 报告 + 部署 Q&A + 配置/依赖核验。未在目标硬件运行。

## 1. 一句话定位

DeepInfer 是 DeepSeek 生产云服务推理框架的开源版，深度绑定自家 DeepSeek-V4.1-Flash 模型架构（压缩注意力 + SWA + YoCo + MTP/DFlash + Engram 压缩 tokenizer + MoE EPLB），双硬件后端（NVIDIA SM100 Blackwell / 华为 Ascend 950 NPU），vLLM 风格 API，含大规模生产部署的架构基因（E/P/D 分离、KVCache 72h、多负载均衡器）。

## 2. 系统三层架构

```
LLM (entrypoints/llm.py, vLLM 风格 offline batch API)
 └─ LLMEngine (engine/llm_engine.py) — Python 编排
    ├─ Scheduler (csrc/engine/scheduler.hpp) — C++ 调度核心，GIL 释放
    ├─ BlockManager / state_manager (block_manager.hpp) — KV/状态块管理
    ├─ MTPVarlen (csrc/engine/mtp_varlen.hpp) — DSpark 变长推测分配
    ├─ Sequence (sequence.hpp) — 序列状态机
    └─ Worker (worker/worker.py) — 每步 Forward→PostProcess→Sample→PostProcess
        └─ ModelRunner → DeepseekForCausalLM (model_executor/models/deepseek.py)
            ├─ DeepseekDecoderLayer × n（MLA 压缩注意力 / SWA / MoE）
            ├─ DFlash（MTP 投机模块）
            ├─ Sampler（确定性采样 + 结构化输出）
            └─ EPLB（ExpertLoadBalancer 运行时热重排）
```

## 3. 三大独特机制（相对 vLLM/SGLang）

### 3.1 ZOS（Zero-Overhead Scheduling，零开销调度）— 调度器层面最核心差异化
- **思想**：decode 步的 post-process 延迟一个 step，与下一步 forward 在 GPU 重叠（worker.py:182-187），让调度器 CPU 开销被 GPU 掩盖。
- **难点与解法**：在途 seq 可能被抢占重算 + varlen 下 seqlen_q 需按上一步采样结果修正 → CPU 元数据滞后一步。`_correct_seqlens_qk`（worker.py:191）用 `prepare_zos_inputs`（zos.cpp:142）算 **remap**（本步与上步 batch 的 seq 对应关系），GPU 上用上一步 confidence rates 修正 seqlens_q/k，D2H 回读刷新采样元数据。
- **token 打包优化**（zos.cpp:31）：无重叠时打包最终 input_ids；有重叠时只填 fresh rows，matched rows 复用上一步 GPU 输出，减 H2D。

### 3.2 DSpark/MTP 变长自适应推测解码 — 投机解码的极致
- 不是固定 speculative length：Scheduler 先选定请求；`MTPVarlen.schedule`（mtp_varlen.hpp:150）再动态选**验证 token 总预算 total_q 和每请求验证长度**，不增删请求。`best_bsz` 不是请求数；最大化的是 `估计接受 token / 估计耗时` 代理目标。原始 C++ 头文件的隔离 CPU 验证、两轮 confidence 因果约束及限制见 [初览笔记 DSpark 节](deepinfer-initial-survey.md#dspark请求数验证预算与随机采样是三个层次)。
- **耗时估计自适应**：`ForwardRuntimeEstimator` 用双层中位数滑动窗口动态拟合 `slope_ms=(forward_ms-baseline)/graph_size`；运行时由 engine 用 CUDA event 实测喂入持续校正。
- **接受率**：MTP confidence head 逐位累乘（process_MTPInfo），N-2 轮算 best_bsz、N-1 轮贪心分配每请求 seqlen_q。
- **DFlash 深度集成**：draft 特征随主模型 CUDA graph 捕获写入 main KV cache；MTPInfoPool 跨 step 稳定存储。

### 3.3 DeepSeek V4 混合注意力一体化元数据 — 模型专属的深度
- dense 压缩层 + sparse（top-k 索引注意力）+ SWA + YoCo prefill 截断 + Engram n-gram，全在调度阶段生成**块加载计划**（`block_load_plan.cpp` 的段表 `[src_block_id,src_begin,src_end,dst_offset]`），把多层不同注意力 KV 加载统一成一次 GPU 拷贝指令。
- **YoCo bounded replay**：模型切 prefill/consumer 双半区，consumer 层 SWA 有界截断，compact（真截行+CP 重分布）/mask（只收窄可见域）双模式，compact 布局 metadata 浅拷贝。

## 4. 模型执行器要点

- **模型加载**（deepseek.py:446 `load_model`）：meta device 建参 → `materialize_hf_parameter` 逐参搬 GPU 保留布局/scale。FP4/FP8 混合量化（routed experts fp4、KV cache FP4/MXFP8、indexer QK FP4）；参数 tie（MTP 复用 main vocab 共享 storage）。
- **注意力**（attention_backend.py 薄封装 FlashMLA）：prefill `flash_mla_sparse_fwd`、decode `flash_mla_with_kvcache`（FP8 paged KV）、融合 `fused_norm_rope_attn_rope_cast`（norm+RoPE+attn+再RoPE+cast 一核）；head 补齐到 64/128。
- **采样**（sampler.py + tilelang_ops）：`fused_sample` 一核完成 temperature/top-p/top-k/log-softmax/采样；**确定性**用 `seq_seeds`+位置 hash（DRAFT_POS_HASH_MUL）+ tag 分隔 RNG 命名空间；**batch-invariant** 用批量无关 RNG 种子；**结构化输出**（regex/JSON/EBNF）由 xgrammar 编译、bitmask 异步 pinned 拷贝、MTP 下逐 draft 独立 mask。
- **EPLB**（eplb/）：冗余 expert 动态分配 + 贪心跨 rank 排布 + **运行时 inplace 权重重排**（NCCL batched P2P 或 symm_mem IPC 镜像免通信），权重 shuffle 不影响前向。
- **tilelang_ops 内核分发器**：cuda/asc 双后端工厂，每算子 `_ref` 参考实现 + 测试。分工：注意力→FlashMLA、GEMM→DeepGEMM、topk→DeepSelect，自研内核补三者未覆盖且与自家模型强耦合的算子（压缩 KV、YoCo、DFlash、engram、HC）。

## 5. NPU（Ascend 950）作为一等公民

- 平台抽象 `platform.py`：GPU/`PlatformNPU` 双后端，运行时按硬件探测（`DEEPINFER_ACCELERATOR` 只是 build 开关）。
- Ascend 特化：`tilelang_ops_backend=asc`、`distributed_backend=hccl`、FP8 scale 用 int16（GPU int32）；`torch_npu.transfer_to_npu` 映射 cuda→npu；`npu_ops/patch_imports.py` import hook 把不支持的 CUDA 算子懒重定向到 NPU 实现（DeepSelect topk 回退 torch、ViT 走 npu_fusion_attention）。
- 共享层：csrc `WITH_NPU` 宏（`get_default_device` kPrivateUse1、NPU 用 `at_npu::from_blob`）；`asc_utils.cpp` IPC（2MB 对齐、PID 校验关闭）。
- 全部模型能力（HF 加载、HCCL 并行、IPC、profile、MoE/MTP/Engram/YoCo）NPU 可用；但 batch-invariant/确定性仅 NVIDIA（README 注明）。

## 6. 分布式与未开源边界

- 已提供：`shm_broadcast`（vLLM 风格无锁共享内存广播，CP 组广播 metadata Python 对象）；并行组拓扑（World/DP/CP/EP/Engram）；NCCL/HCCL 子组 lazy 初始化。
- **未包含在开源版**（README 明确）：KV cache 系统、VE/Prefill/Decode 分离（EngineType.NORMAL/PROMPT/COMPLETION 三态 + shm_broadcast + profile_server 已为此铺垫）、DP 副本间 prefill/decode 负载均衡。当前只做单引擎内多卡并行（Attention DP/balanced CP + MoE EP/EPLB）。

## 7. 生产架构（部署 Q&A）

- **三目标**：多芯片兼容；用常见硬件组织 P/D（8 卡 Prefill EP2-8/DP2-8，64-128 卡超节点 Decode Wide-EP）；KVCache 保留 72h+。
- **E/P/D 分离架构** + 独立存储节点；Prefill/Decode/VE/Expert-Parallel 四个 Load Balancer + KVCache Storage（后两者未开源，知乎 Blog 说明）。
- **性能数据**：单集群 155 万亿 tokens/天，每卡每小时 38 美金（谷价）。
- 未用 TP、CP（P/D 分离场景）。VE 用单卡便宜卡。

## 8. 依赖与构建

- NVIDIA：torch 2.13.0+cu132、flash-attn-4、NCCL 补丁 2.32.3；Ascend：torch 2.13.0+cpu + torch-npu 2.13.0rc1。
- 公共：tilelang 0.1.15、tile-kernels 2.1.0、xgrammar 0.2.2、safetensors、transformers 4.50、pyzmq、posix_ipc。
- 构建：GCC≥13、Python 3.12、CUDA 13（NVIDIA）/ CANN 9.2（Ascend）；setup.py 按平台选 csrc + graph_timer；vendor 子模块按平台拉（NVIDIA: DeepSelect/FlashMLA/DeepEP/DeepGEMM；Ascend: +DeepEP/DeepGEMM-Ascend）。
- 用户入口：`python examples/offline_inference.py --ep 8`（vLLM 风格离线 batch）；`eval/*/run.sh`（AIME/HLE/GPQA）；多进程 `_launch.py run_dp`。

## 9. 与本项目既有积累的关联（对照点）

- **batch-invariant / 确定性**：与 verl `_tensor_detach`、vLLM batch-invariant 深挖同源，DeepInfer 把它做成跨层核心约束（kernel+通信+采样 RNG）。
- **P/D 分离**：与 vLLM PD disaggregation、KV transfer、verl PD 对照；DeepInfer EngineType 三态 + shm_broadcast 是雏形。
- **NPU 一等公民**：与 vLLM-Ascend、MindIE、Ascend 生态对照；DeepInfer 是首个"模型厂商官方 + NPU 双后端"的推理框架。
- **MTP/DSpark**：与 speculative decoding 系列、SGLang Eagle、vLLM DFlash 对照。
- **MoE EPLB**：与 Megatron quantile balancing、router replay 对照；DeepInfer 是运行时热重排。
- **压缩注意力/YoCo**：V4 专属，无直接对照，是 DeepSeek 对长上下文 + 高吞吐的新解法。
