# DeepInfer 平民版（魔改适配非旗舰硬件）可行性专题

> 目标：把 DeepInfer 从"仅 NVIDIA SM100（Blackwell）+ 华为昇腾 Ascend 950（A5）"的专用引擎，评估改造成适配 RTX 4090（SM89）、A100（SM80）、H100（SM90）、昇腾 910C（A2）等普通硬件的"平民版本"是否可行。
>
> 核心问题：①它对机器的依赖堵点到底在哪；②这些堵点能否拆除或平移替换。
>
> 源码基线：`deepseek-ai/DeepInfer` commit `8b16149`（本地 clone `/Users/jackiemac/workspace/DeepInfer`）。
>
> 状态：**堵点定位完成 + 替代方案调研完成 + 两段 4090 真机冒烟（import 链全通、csrc `_C` 编译成功、发现 tilelang PDL 硬堵点）。2026-10-22 续：PDL 机制源码级定位（tilelang 编译期硬检查、无配置可关）+ 5 个 kernel 文本级去 PDL 后 SM89 编译/数值验证全过（改写是机械工程且语义保持）。三续：43/43 个 `_tl` 工厂全量去 PDL 改写后 SM89 编译全过（含条件启用分支），工程量全量实证。下一步：完整移植需 Triton/Cutlass 替换四 kernel 包 + 数值语义逐 kernel 补全 + V4 权重，价值判断待用户确认。** 本专题完成后将作为新章节插入 DeepInfer 研究报告。

## 结论先行

- **NVIDIA 侧**：DeepInfer 的 **Python 算子抽象层（attention_backend / linear / mlp）本身是后端无关的**，真正绑死架构的是它默认调用的三个 kernel 包——**FlashMLA、DeepGEMM、DeepEP**（各自编译出的二进制只支持 SM100/SM90）。所以"平民版"的路径不是改 DeepInfer 的 Python 逻辑，而是**替换这三个 kernel 包的实现 + 放宽一个架构检查 + 让 93 个 tilelang 自定义 kernel 重新编译**。
- **昇腾侧**：同一套 Python 抽象，底层 kernel 换成昇腾版（DeepEP-Ascend / DeepGEMM-Ascend / FlashMLA-Ascend），但它们 target **A5 架构（950）+ CANN 9.2**。910C 是 **A2 架构 + CANN 8.x**，代际不同，**不是替换而是重新移植 kernel**，成本显著高于 NVIDIA 侧。
- **可行性分层**：H100 最容易（FlashMLA 需替换、DeepGEMM 原生支持 SM90）；4090 次之（fp8 只能普通粒度、需退 bf16 或改量化）；A100 最费（无 fp8 tensor core，必须反量化回 bf16）；910C 属于昇腾侧跨代移植，工作量最大。

## NVIDIA 侧堵点清单（每条含源码证据）

依赖版本（`requirements-nvidia.txt`）：`torch==2.13.0+cu132`（CUDA 13）、`flash-attn-4==4.0.0b34`。构建：`build.sh` 会强制装 `nvidia-nccl-cu13==2.32.3`。

### 堵点 1：MLA 注意力 —— 唯一后端就是 FlashMLA
- 文件 `deepinfer/model_executor/layers/attention_backend.py:2-3`：**硬编码 `from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, fused_norm_rope_attn_rope_cast`**，无 fallback 到 FlashAttention / PyTorch SDPA。
- `attention_backend.py` 提供 `prefill_attn_varlen_sparse` / `decode_mla_mqa_sparse` 两个入口，底层全部走 flash_mla。
- `deepinfer/model_executor/models/v4_modules/attn.py`、`input_metadata/dflash.py`、`input_metadata/input_metadata.py` 都直接 `from flash_mla.flash_mla_interface import FlashMLASchedMeta`，调度元数据结构绑死 FlashMLA。
- **FlashMLA 官方支持**：NVIDIA 侧仅 SM100/SM103（2026.09 版还移除了 Hopper），CUDA 13.1+。昇腾侧仅 Ascend 950。
- 头数还被 `_pad_attention_heads` 限定在 `(64, 128)`（`attention_backend.py:5,13-16`）。

### 堵点 2：所有线性层 GEMM —— 唯一后端就是 DeepGEMM
- 文件 `deepinfer/model_executor/layers/linear.py:3` **硬编码 `import deep_gemm`**；`linear.py:33` `deep_gemm.bf16_gemm_nt`、`linear.py:165` `deep_gemm.fp8_fp4_gemm_nt`。无 `torch.matmul` / `F.linear` fallback。
- fp8 量化用 `tile_kernels.quant.per_token_cast`（`linear.py:150`），依赖 `tile-kernels==2.1.0`。
- **DeepGEMM 官方支持**：仅 SM90（Hopper）/ SM100（Blackwell），CUDA ≥12.9。

### 堵点 3：MoE / expert 并行 —— 深度绑定 DeepGEMM + DeepEP
- 文件 `deepinfer/model_executor/models/v4_modules/mlp.py:14-19`：
  - `import deep_gemm`
  - `from deep_ep import EPBuffer, EPHandle, EventHandle, EventOverlap`
  - `from tile_kernels.moe import moe_topk_gate`
  - `from tile_kernels.quant import swiglu_forward_and_per_token_cast`
- 用的是 DeepGEMM 的 **Blackwell-specific API**：`m_grouped_fp8_fp4_gemm_nt_contiguous`（`mlp.py:119`）、`m_grouped_bf16_gemm_nt_contiguous`（`mlp.py:130`）、`fp8_fp4_mega_moe` / `bf16_mega_moe`（`mlp.py:255`）、`bf16_mega_gate`（`mlp.py:398`）、`transform_weights_for_mega_moe`（`mlp.py:96,98`）。mega_moe / mega_gate 是 V4.1 专用、依赖 Blackwell 特性。
- **DeepEP** 处理 expert 通信（EPBuffer/EPHandle/EventOverlap），SM 限制与 DeepGEMM 同类。

### 堵点 4：93 个 tilelang 自定义 kernel —— 可移植但需重编译
- `deepinfer/model_executor/layers/tilelang_ops/` 下 **93 个 .py 文件、约 2.3 万行**（cuda 后端）；昇腾 `asc` 后端 30 个文件、7418 行。
- 用 `@tilelang.jit(..., target="cuda")`（如 `cuda/swa_index.py:7`）或 `target="asc"`。tilelang 是 **编译期 DSL，按设备 compute capability 自动生成对应 PTX**（`platform.py:128` 设 `TILELANG_DEFAULT_TARGET=ascend`），**本身可跨架构**——移植到其他 SM 可行，但必须在目标 GPU 机器上重新编译（本机无 tilelang）。
- 这是唯一能保留的核心资产。

### 堵点 5：架构硬门槛 —— 非 Blackwell 直接抛异常
- 文件 `deep_cluster/hardware/device.py:151-156`：
  ```python
  def get_bf16_tflops_limit() -> int:
      if accelerator_kind() == "ascend": return 432
      if is_sm_major_10(): return 2500/2250  # SM100
      raise RuntimeError("unknown GPU type")   # ← 非 Blackwell 到这里
  ```
  `is_sm_major_10`（`device.py:160-164`）要求 `capability[0] == 10`。
- 调用点：`deepinfer/engine/llm_engine.py:304`（引擎初始化）——**4090（SM89）一启动 LLMEngine 就抛 "unknown GPU type"**。
- 另有 `vit.py:42` 用 `is_sm_major_10()` 决定视觉编码器走哪条算路。

### NVIDIA 侧"平民化"替换清单
| 堵点 | 当前（Blackwell-only） | 平民替代（跨架构） |
|---|---|---|
| MLA 注意力 | FlashMLA（SM100/103） | **vLLM `TRITON_MLA`**（Triton 编译期适配，fp8 需 SM89+、bf16 需 SM80+，覆盖 4090/A100/H100）；SGLang 非 Hopper/Blackwell 默认也退回 Triton |
| 线性 GEMM | DeepGEMM（SM90/100） | torch.matmul / vLLM Cutlass fp8（SM89+/90+）/ Triton batched MoE |
| MoE GEMM | DeepGEMM mega_moe（Blackwell） | vLLM Cutlass grouped GEMM（fp8 仅 SM90/100）/ Triton（Ampere/Ada 兜底）/ FlashInfer MoE（SM75+） |
| expert 通信 | DeepEP（SM90/100） | vLLM 标准 EP：allgather–reducescatter / passthrough |
| fp8/fp4 量化 | per-32 block fp8（Blackwell 能力） | 见下方量化小节 |
| tilelang kernel | 可跨架构 | 保留，需在目标机重编译 |

证据来源：vLLM `vllm/v1/attention/backends/mla/{triton_mla,flashattn_mla,flashmla,flashinfer_mla}.py`、`csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`；SGLang `docs/advanced_features/attention_backend.mdx`；FlashAttention README（MLA kernel 均 `_sm100`）；FlashInfer README。

## fp8 / fp4 硬件的关键约束

- **H100（SM90）**：原生 fp8 tensor core（E4M3/E5M2）。可跑 DeepInfer 的 per-32 block fp8。
- **RTX 4090（SM89）**：**有 fp8 计算，但只支持普通（非分组）fp8**；DeepInfer 的 per-32 block fp8 不支持（vLLM `cutlass_scaled_mm_supports_block_fp8` 仅 SM90/SM100）。
- **A100（SM80）**：**无 fp8 tensor core**（NVIDIA Hopper 博客确认 fp8 由 Hopper 引入），只能 fp16/bf16/tf32/int8。
- **bf16 回退可行**：fp8 权重可在加载时反量化（scale 乘回）跑 bf16，精度几乎无损；这是 vLLM/SGLang 在非 fp8 硬件上的标准做法。**代价**：省显存/省带宽的量化收益消失（权重约 +25%，KV 同理），平民版要付出这个代价。
- fp4（NVFP4/MXFP4）属 Blackwell 能力，4090/A100/H100 基本不可用，需退 bf16/fp8。

## 昇腾侧堵点

- 昇腾侧与 NVIDIA 侧**共享同一套 Python 抽象**：昇腾版 FlashMLA / DeepGEMM / DeepEP 安装名也是 `flash_mla` / `deep_gemm` / `deep_ep`（vendor 子模块 `DeepEP-Ascend` / `DeepGEMM-Ascend` / `FlashMLA-Ascend`）。
- 依赖版本（`requirements-ascend.txt`）：`torch==2.13.0+cpu`、`torch-npu==2.13.0rc1`；构建需 CANN 9.2。
- **目标 A5 架构（950）**：源码注释 `# TODO(A5)`（`tilelang_ops/store_kv_cache.py:81`）、`0xA5`（同文件:251）、`platform.py:153` 设 `TORCH_NPU_DEVICE_CAPABILITY=9.0`。
- **DeepEP-Ascend 官方**：仅验证 "Ascend 950DT + CANN 9.2.0"，README 明言 "Kernel support on other Ascend generations or CANN versions **has not been established**"。
- **昇腾 910C = A2 架构 + CANN 8.x**，与 DeepInfer 的 A5 + CANN 9.2 **不同代际** → 昇腾侧 910C 不是"替换 kernel 包"能解决，需**重新移植**（芯片指令集不同），工作量大，可行性低。

## 可行性分层结论（给后续实验规划用）

| 目标硬件 | 架构 | 可行性 | 关键动作 |
|---|---|---|---|
| **H100** | SM90 | **最容易** | DeepGEMM 原生支持 SM90、有 fp8；只需换掉 FlashMLA → Triton/FlashInfer MLA；放宽架构检查（把 is_sm_major_10 改成 ≥9） |
| **RTX 4090** | SM89 | **可行** | FlashMLA、DeepGEMM 均不可用，需 Triton MLA + Triton/Cutlass MoE；fp8 只能普通粒度或退 bf16 |
| **A100** | SM80 | **可行但最费** | 无 fp8，必须反量化回 bf16；全走 Triton/Cutlass 兜底路径 |
| **昇腾 910C** | A2 | **难，跨代移植** | 非替换 kernel 包能解决，需按 A2 指令集重写 kernel；CANN 8.x 与 9.2 不兼容 |

**共性动作**（任何目标都要做）：
1. 放宽 `get_bf16_tflops_limit` / `is_sm_major_10` 的架构门槛（`device.py` + `vit.py:42`）。
2. 替换 flash_mla / deep_gemm / deep_ep 三个 import（Python 层已隔离，改动集中）。
3. 在目标机重编译 93 个 tilelang kernel。
4. 处理 fp8 量化在无 block-fp8 硬件上的降级（普通 fp8 或 bf16 反量化）。

## 4090 真机验证（2026-10-21，M5 服务器 4090-1 实测）

> 用户指派 GPU/NPU 算力任务后，在服务器 `4090-1`（8×RTX 4090 SM89，CUDA 12.8 工具链，驱动 575.57/CUDA 12.9）上对上述静态分析做了第一步实测。**当前进度：堵点 1/2/3/5 已有真机或源码级确认；csrc C++ 引擎（`deepinfer._C`）在 4090 上编译并导入成功；完整离线推理跑通仍需替换三个 kernel 包。** 独立克隆 `/home/zxw/deepinfer-civilian`（HEAD `774ce244`，含全部 vendor 子模块，16 个全部初始化），独立 conda 环境 `deepinfer-smoke`（Python 3.12.15 + torch 2.13.0+cu129 + gcc 13 来自 conda-forge）。

### 实测结果

1. **堵点 5（架构硬门槛）真机确认**：在 4090 上运行
   `get_bf16_tflops_limit()` 抛 `RuntimeError: unknown GPU type`（`accelerator_kind=nvidia`、`num_gpus=8`、`is_sm_major_10=False`）。引擎初始化 `llm_engine.py:304` 会在这里停下。
2. **新增堵点：官方锁定的 torch wheel 与 4090 服务器驱动不匹配**（静态分析未覆盖）：
   `requirements-nvidia.txt` 锁 `torch==2.13.0+cu132`；服务器驱动 575.57 只到 CUDA 12.9，cu132 wheel 初始化报 `RuntimeError: The NVIDIA driver on your system is too old (found version 12090)`。**改用 `torch==2.13.0+cu129` 后在 4090 上 CUDA 正常**（`capability (8,9)`，matmul 通过）。平民移植时必须下调 torch 的 cu 号，或升级驱动。
3. **堵点 1/2/3（kernel 包）真机与源码确认**：
   - FlashMLA：`setup.py` 硬编码 `CUDA_ARCHS = ["sm_100a", "sm_103a"]`；本机编译直接断言失败 `Compiling the kernels requires NVCC 13.1 or higher`。4090 无解。
   - `deepinfer.engine.llm_engine` 导入即 `ModuleNotFoundError: No module named 'deep_ep'`（`parallel_state.py:4` 顶层 import）。`deep_gemm` 在 `linear.py:3`、`mlp.py:14`、`attn.py:11` 等 10+ 处顶层 import，`deep_select` 在 sampler 顶层 import。
   - tilelang kernel（`tilelang_ops/`，含 `ops.norm_tl` 等）为运行时 JIT 编译，按目标 SM 生成 PTX，跨架构本身可行，仍需在 4090 上编译。
4. **csrc C++ 引擎移植可行性验证（本专题新进展）**：`setup.py` 的 `deepinfer._C` 扩展（`csrc/engine/*`、pybind、graph_timer 等 C++/CUDA 源码，**不依赖 Blackwell 特性**）在 4090 上用 **gcc 13（conda-forge）+ CUDA 12.8 + torch cu129** 一次编译通过，`_C.cpython-312-x86_64-linux-gnu.so`（14.8MB）可导入，导出 `cuda_utils / engine / graph_timer`。系统 gcc 11 因缺 `<format>`（C++20）编译失败，需 GCC ≥ 13（与 README 要求一致）。
5. **环境搭建要点（可复现）**：
   - 系统 `python3` 为 3.10 且无 pip/ensurepip → 用 miniconda 建 `deepinfer-smoke`（Python 3.12）。
   - torch cu129 与 cu132 两套 nvidia wheel 混装会导致 `libnvJitLink.so.12` 符号冲突（`__nvJitLinkGetErrorLogSize_12_9`）与 `libcusparseLt/libnccl/libcudnn` 缺失——需卸载 cu13 系并 `--force-reinstall --no-deps` 重装 cu12 系各 wheel。
   - 运行需设 `LD_LIBRARY_PATH` 包含 torch/lib + `nvidia/*/lib`（**conda 的 nvjitlink 须在系统 `/usr/local/cuda-12.8/lib64` 之前**，否则解析到缺 12_9 符号的旧库）。

### 下一步（移植冒烟第二段）

- 替换 `deep_ep / deep_gemm / deep_select / flash_mla` 四个顶层 import（`parallel_state.py`、`userbuffer.py`、`linear.py`、`mlp.py`、`attn.py`、`sampler.py` 等），用 Triton/Cutlass 替代，放宽 `get_bf16_tflops_limit` 后尝试实例化 LLMEngine。
- 在 4090 上编译 tilelang kernel（观察 93 个 kernel 的实际编译结果）。
- 完整 `examples/offline_inference.py` 需要 V4 模型权重与 EP 配置，工程量超出本轮；价值判断（vs 直接用 vLLM）仍待用户确认。

## 移植冒烟第二段（2026-10-22，M5 服务器 4090-1 实测）

> 延续第一段既定下一步，用最小 stub 替换四个 kernel 包 + 放宽架构检查，验证 llm_engine import 链与 tilelang kernel 的 SM89 编译可行性。实验材料与复现见 [experiments/deepinfer-civilian-port-stage2](../../experiments/deepinfer-civilian-port-stage2/readme.md)。

### 第二段实测结果

1. **llm_engine import 链全通**：20 个核心模块（llm_engine / arg_utils / profile_server / parallel_state / sampler / attention_backend / linear / model_loader / models / v4_modules / worker / sequence / sampling_params）经 stub 全部导入成功，0 失败。前置修复三处：`get_bf16_tflops_limit` 加非 Blackwell 分支（SM89→82，只影响 StatsLogger 统计）；tilelang_ops 运行时模块顶层混入的 `import pytest`（装 pytest）；conda gcc13 libstdc++ 前置（否则 `_C` 缺 GLIBCXX_3.4.31）。
2. **tilelang 0.1.15 本身支持 SM89**：最小 saxpy kernel 1.5s 编译、运行误差 0；`T.sync_grid()` 2.4s 编译运行 OK。
3. **新堵点（静态分析未覆盖）：43 个 cuda tilelang kernel 文件全部显式使用 PDL**（`T.pdl_sync`/`T.pdl_trigger`，29 文件 28 个用到）。PDL 是 Hopper+（SM90+）硬件特性，SM89 上 `norm_tl` 编译抛 `PDL is not supported`。**修正本节静态结论「tilelang kernel 跨架构本身可行、只需重编译」——多数 kernel 需去 PDL 改写，不是简单重编译。**
4. **LLMEngine 实例化停在数据前提**：`ModelConfig` 只接受架构 `DeepseekV41ForCausalLM`（config 需含 `vl_aligner`）；服务器无 V4 权重（Qwen2.5-0.5B 抛 `Unsupported HF architecture 'Qwen2ForCausalLM'`）。与「未验证项」一致：完整实例化需 V4 权重 + EP 配置，价值判断待用户确认。
5. **昇腾 910C 仍未连接**（2026-10-22 昇腾 VPN 100% 丢包，待用户恢复）。

### 第二段续：PDL 机制深挖与去 PDL 改写探针（2026-10-22，M5）

> 在第二段结论基础上，把「43 个 kernel 去 PDL 改写」从定性推进到真机可验证的工程结论。材料见 [experiments/deepinfer-civilian-port-stage2](../../experiments/deepinfer-civilian-port-stage2/readme.md)（`pdlfree-variants/`、`probe_norm_pdlfree.py`、`probe_kernels_compile.py`）。

1. **堵点根因在 tilelang 编译管线，无配置可关**：`tilelang/cuda/pipeline.py:244` 固定调用 `MarkCudaSyncCalls(have_pdl(target))`；`contrib/nvcc.py` 的 `have_pdl()` 只在 compute capability major ≥ 9（Hopper+）返回 True。SM89 下 `src/cuda/transform/lower_pdl.cc:44` 只要发现 `pdl_sync`/`pdl_trigger` 就 `ICHECK` 抛 `PDL is not supported`。**编译期硬检查。**
2. **语义映射 → 改写是机械性的**：`pdl_sync` 标记 `kHasGridSync`（grid 同步），`pdl_trigger` 标记 `kHasTriggerLaunch`（允许下游 kernel PDL 提前启动）。顺序 launch 语义下：`pdl_trigger` 可安全删除（下游本就要等前一个 kernel 完成），`pdl_sync` 用已验证可行的 `T.sync_grid()` 替代。
3. **用量**：28/30 个 cuda tilelang kernel 源文件用 PDL，共 56 处 `pdl_sync` + 8 处 `pdl_trigger`（另 `vendor/DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` 1 处）。少数（如 `state_cache.py`）按数据规模条件启用，已有非 PDL 路径。
4. **单 kernel 真机验证（PASS）**：`norm` kernel 文本级替换（`pdl_sync→sync_grid`、`pdl_trigger→pass`）后在 SM89 编译 5.3s，B=2/S=128/d=64 fp16 与 torch RMSNorm 对照 **max abs err = 0.0、0% mismatch**（注意多 batch 须 `xdim=3`，`xdim=2` 只写 batch 0）。另 4 个 kernel（kv_index_cat / dflash_input_store / load_kv_index / fused_sample）最小参数编译全过（2.3–9.7s）。**「去 PDL 改写」是机械工程而非科学障碍。**
5. **数值对照扩展（再续，5/5 PASS）**：`probe_pdlfree_numeric.py` 给其余 4 个 kernel 补数值对照（torch 参考）——`kv_index_cat` 拷贝+偏移加 eq、`load_kv_index` 索引展开 eq、`dflash_input_store` hc_mult=1 拷贝 / hc_mult=2 均值 max err=0.0、`fused_sample`（43 个中最复杂：两级 launch + LSE streaming 归约 + lse_workspace 跨 launch 共享）logprobs vs `log_softmax(logits/t)` max err=1e-6、lse0/lse1 err=0.0、candidate 合法。**去 PDL 后的跨 launch 依赖在顺序语义下数值保持**，改写从「能编译」推进到「语义保持」。
6. **全量改写编译扫查（三续，43/43 编译全过）**：`scan_pdlfree_all.py` 对 28/28 个 PDL 文件、43/43 个 `_tl` 工厂全部生成 `_pdlfree` 变体并在 SM89 编译——**全部 OK**（新编译 2.2–12.1s，`candidate_block_max` 最慢 12.1s；5 个已数值验证 kernel 命中缓存）。覆盖含 `fused_transform`（21 参数全 kwonly）、`load_paged_kv_cache`/`store_kv_cache`（需 `KVCacheFormat` 常量）、`state_cache` 条件启用分支（64×1024=65536 触发 enable，条件 PDL 在启用时真实编译 `T.sync_grid()`）。替换计数核对：`sync_grid` 与改写前 `pdl_sync` 调用计数逐一相等（合计 51 处）、`pdl_trigger`→`pass` 8 处均为独立语句；变体文件无 PDL 调用残留（仅剩 `enable_pdl_sync` 变量名）。**「去 PDL 改写」工程量与可机械性在真实硬件上全量实证，PDL 不是 4090 移植编译层的隐藏障碍。**
7. **未覆盖边界**：数值对照覆盖 5 个 kernel 的选定参数组合；其余 38 个 kernel 编译全过但数值语义未逐一对照（若继续完整移植需逐个补）；编译扫查用最小合法参数，未遍历全部参数组合/量化布局变体；`fused_sample` 采样分支只查 candidate 合法性未对照分布（自定义 RNG）；`pdl_trigger` 删除失去 kernel 间重叠（性能影响未测）；vendor swiglu（DeepGEMM 第三方包，非 DeepInfer 主包 import 链）未在扫查。
8. **load/store_kv_cache 全矩阵数值验证 + sync_grid 新硬边界（四续，2026-10-22，M5）**：把每请求必经的 KV 缓存存取两个 kernel 用**上游自带 GPU 测试矩阵**做数值对照（monkeypatch `ops` 两个工厂指向变体后直接跑 `test_load_paged_kv_cache`/`test_store_cache`，与 torch 参考逐字节对照，5 布局 × 2 block_size × 3 token 数 + store 三次 update）。结果分两半：**pdlfree（pdl_sync→sync_grid）小用例 20/20 全过、67890 大用例 10/10 全 FAIL**——`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`：tilelang `sync_grid()` 生成 `cooperative_groups::this_grid().sync()`，**要求 cooperative launch（全部 block 同时驻留），4090（128 SM）grid 上限约 1024-4096，大输入启动失败；上游 PDL 版本无此驻留限制**——这是 sync_grid 替代引入的运行期新硬边界（编译扫查用小 grid 未暴露）。**nosync 变体（pdl_sync 直接删除，不替换）全矩阵 60/60 PASS（含全部大用例）**：这两个 kernel 的 pdl_sync 注释明确是「等待 dependency launch 产生 load_segments/slot_mapping」，顺序 launch 下前序 kernel 由 stream 顺序保证完成，**同步不需要**。**结论修正：对「等待前序 kernel 数据」型的 pdl_sync（此类是多数），正确改写是删除而非替换成 sync_grid；sync_grid 仅适用于同一 launch 内跨 block 数据依赖**。另：变体与原件同放 backend 包会让 `ops` 模块 `backend symbol collision` 断言失败（移植时需替换原文件或放包外）。证据见 [stage2 readme](../../experiments/deepinfer-civilian-port-stage2/readme.md#loadstore_kv_cache-全矩阵数值验证--sync_grid-grid-上限发现2026-10-22-四续m5)。
9. **全量 nosync 改写验证（五续，43/43 编译 + 7 kernel 数值保持，2026-10-22，M5）**：把四续结论推广到全部 43 个 kernel。**静态语义分类**：拉回 28 个 PDL 源文件逐处核对全部 **55 处 `pdl_sync` + 8 处 `pdl_trigger`**——所有 pdl_sync 均位于 kernel 开头（紧随 `T.Kernel` 后等待前序 launch 写入的全局数据，如 load_segments/slot_mapping/cu_lens/lse_workspace/计数器），**全部是依赖型，无一处是「同一 launch 内跨 block 通信」**（那才需要 sync_grid；`sampler.py` 甚至注释明示上游刻意「without a cooperative grid barrier that would prevent PDL launches」）。**全量 nosync 改写**：`gen_nosync_variants.py` 从源文件删除 pdl_sync（空块补 pass）、pdl_trigger→pass，生成 28 个 `*_nosync.py`（51 处删除与三续 51 处 sync_grid 替换逐一对应；AST 校验 + 零残留）。**43/43 SM89 编译全过**（`scan_nosync_all.py`，首次 ~2m20s，重跑缓存秒级确认可复现）。**7 个 kernel nosync 数值保持**：norm max err=0.0、kv_index_cat/load_kv_index eq、dflash_input_store max err=0.0、fused_sample logprobs err=1e-6 + lse0/1 err=0.0（`probe_nosync_numeric.py` + `probe_norm_nosync.py`）、load/store 全矩阵 60/60（四续）。**改写规则收敛**：`pdl_trigger` → 删除；`pdl_sync`（依赖 launch 前序数据）→ 删除；仅「同一 launch 内跨 block 通信」才考虑 sync_grid（本项目 43 个 kernel 无此用例，且受 cooperative-launch grid 上限约束）。**三续的 51 处 sync_grid 替换被修正为「可编译但引入运行期硬上限的次优改写」，正确形式是删除**。证据见 [stage2 readme](../../experiments/deepinfer-civilian-port-stage2/readme.md#全量-nosync-改写验证4343-编译--7-kernel-数值保持2026-10-22-五续m5)。

### 第二段结论

- **平民移植可行性分层修正**：Python 层 import 堵点已全部用 stub 打通（证明 import 链可移植）；真正的新硬堵点在 **tilelang kernel 的 PDL 依赖**——43 个 kernel 需逐个改写（**路径已全量实证：全部 51 处 pdl_sync 均为依赖前序 kernel 数据型、可直接删除（nosync，43/43 编译 + 7 kernel 数值保持）；pdl_trigger 删除；仅同一 launch 内跨 block 通信才用 sync_grid 且受 cooperative-launch grid 上限约束（本项目无此用例）**，工程量大但完全机械）。加上第一段的 FlashMLA/DeepGEMM/DeepEP 替换，4090 平民版仍是「可做但工作量很大」的工程，研究价值（验证 ZOS/EPLB/DSpark 平移到普通硬件）高于工程价值（直接用 vLLM 更务实）。
- 本段未改 DeepInfer 上游源码（patch 仅就地放宽 `device.py` 一处，属本地冒烟用）。

## 未验证项 / 下一步

- 其余静态分析部分（昇腾 910C 跨代移植、A100 反量化回退）仍未真机验证；需对应硬件。
- 完整移植冒烟（替换 kernel 包 + 跑通 `offline_inference.py`）未完成，见上节下一步。
- 本专题存在一个价值判断待用户确认：DeepInfer 平民版的技术价值 vs 直接用 vLLM/SGLang 跑 DeepSeek-V4（vLLM 已原生支持 MLA + fp8，工程量远小）。**倾向**：若目标是"在普通卡上跑 DeepSeek-V4"，vLLM 是更务实的路径；DeepInfer 平民版的价值主要在"验证 DeepInfer 的 ZOS/EPLB/DSpark 等系统设计能否平移到普通硬件"，属于研究价值而非工程价值。
