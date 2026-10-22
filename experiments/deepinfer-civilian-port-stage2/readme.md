# DeepInfer 平民版移植冒烟第二段（4090-1，2026-10-22）

## 目的

延续[平民版专题](../../notebook/frameworks/deepinfer-civilian-port.md#4090-真机验证2026-10-21m5-服务器-4090-1-实测)第一段（`_C` 编译导入成功、四类堵点确认）后的既定下一步：替换 `deep_ep / deep_gemm / deep_select / flash_mla` 四个 kernel 包 import、放宽架构检查，验证 **llm_engine import 链是否走通**，并在 4090（SM89）上实测 **tilelang kernel 的 JIT 编译可行性**（专题静态结论当时认为"跨架构本身可行，只需重编译"）。

## 环境（服务器 4090-1）

- 独立克隆 `/home/zxw/deepinfer-civilian`（HEAD `774ce244`，16 子模块全 init）
- conda 环境 `deepinfer-smoke`：Python 3.12.15 + torch 2.13.0+cu129 + tilelang 0.1.15 + gcc 13（conda-forge）+ pytest
- stub 目录 `/home/zxw/deepinfer-stubs/`（本目录 `deepinfer-stubs/` 的副本），经 PYTHONPATH 注入，不改 DeepInfer 源码
- 环境变量：`LD_LIBRARY_PATH` 需含 `$CONDA_PREFIX/lib`（gcc13 libstdc++，供 `_C`）、torch/lib、nvidia/*/lib（nvjitlink 在系统 CUDA 之前）

## 结果

1. **llm_engine import 链全通**：20 个核心模块（llm_engine、arg_utils、profile_server、parallel_state、sampler、attention_backend、linear、model_loader、models/deepseek(_vl)、v4_modules/*、worker/*、sequence、sampling_params）全部导入成功，0 失败。前置修复：
   - `deep_cluster/hardware/device.py` 的 `get_bf16_tflops_limit` 增加非 Blackwell NVIDIA 分支（SM89 → 82 TFLOPS；只影响 StatsLogger 吞吐统计，不改门禁语义）
   - tilelang_ops 若干运行时模块顶层混入 `import pytest`（测试依赖污染运行时），安装 pytest 解决
   - conda libstdc++（gcc13）前置，否则 `_C` 报缺 GLIBCXX_3.4.31
2. **tilelang 0.1.15 本身支持 SM89**：最小 saxpy kernel 1.5s 编译、运行误差 0。
3. **新堵点（静态分析未覆盖）：全部 43 个 cuda tilelang kernel 文件显式使用 PDL**（`T.pdl_sync` / `T.pdl_trigger`，29 个文件中有 28 个用到）。PDL（Programmatic Dependent Launch）是 Hopper+（SM90+）硬件特性，SM89 上 DeepInfer 的 `norm_tl` 编译即抛 `PDL is not supported`。**修正专题"tilelang kernel 跨架构可行、只需重编译"的结论——大部分 kernel 不是重编译而是需改写（去 PDL 化）。**
4. **PDL 改写路径可行**：tilelang 的 `T.sync_grid()`（普通 grid sync）在 SM89 上编译运行成功（2.4s）。但 43 个 kernel 逐个改写是实质工程，超出冒烟范围。
5. **LLMEngine 实例化停在数据前提**：`EngineArgs.create_engine_configs → ModelConfig` 要求架构 `DeepseekV41ForCausalLM`（V4 多模态，config 需含 `vl_aligner`）；服务器仅有 Qwen2.5-0.5B 等非 V4 模型，抛 `Unsupported HF architecture 'Qwen2ForCausalLM'`。与专题既有结论一致：完整实例化/离线推理需 V4 权重 + EP 配置，价值判断待用户确认。

## 复现

- `run_smoke_stage2.sh`：完整验证序列（arch 检查放宽、20 模块 import、saxpy 对照、gridsync 替代、norm_tl 负验证）
- `probe_tilelang_*.py`：三个独立 tilelang 探针（对照 / 替代 / 负验证）
- `deepinfer-stubs/`：四个 kernel 包的最小 stub（保证 import 链，调用即抛 NotImplementedError）

## 边界

- stub 保证 import 链通过，**不代表** kernel 可用或 DeepInfer 能在 4090 上跑推理；真实替换需 Triton/Cutlass 实现 + 43 个 tilelang kernel 去 PDL 改写。
- 未尝试 V4 权重下载（数据前提 + 磁盘 99% 只剩 109G），昇腾 910C 侧 VPN 掉线未连接。
- 本实验未改 DeepInfer 源码（patch 仅一处 `device.py` 就地放宽，属本地 fork 冒烟用，非上游改动）。

## PDL 机制深挖与去 PDL 改写探针（2026-10-22 续，M5）

接续上述第 3/4 点，把"43 个 kernel 去 PDL 改写"从定性描述推进到可验证的工程结论：

### 机制（tilelang 0.1.15 源码级确认）

- **堵点根因在 tilelang 而非 DeepInfer**：`tilelang/cuda/pipeline.py:244` 在编译管线固定调用 `MarkCudaSyncCalls(have_pdl(target))`；`contrib/nvcc.py:660` 的 `have_pdl()` 只在 compute capability **major ≥ 9**（SM90+ Hopper）返回 True。SM89（4090）下 `MarkCudaSyncCalls`（`src/cuda/transform/lower_pdl.cc:44`）只要在 kernel 里发现 `pdl_sync`/`pdl_trigger` 调用就 `ICHECK` 抛 `"PDL is not supported"`。**这是编译期硬检查，无 pass_config 可关**（`TL_DISABLE_DATA_RACE_CHECK` 等配置不涉及 PDL）。
- **语义映射**：`pdl_sync` 被标记为 `kHasGridSync`（grid 级同步），`pdl_trigger` 被标记为 `kHasTriggerLaunch`（允许下游 kernel 通过 programmatic dependent launch 提前启动，Hopper+ 硬件特性）。**在顺序 launch（非 PDL）语义下：`pdl_trigger` 可安全删除**（下游 kernel 本来就等前一个 kernel 完成），**`pdl_sync` 用 `T.sync_grid()` 替代**（第二段已验证 SM89 可编译运行）——改写是机械性的。
- **用量**：28/30 个 `tilelang_ops/cuda/*.py` 源文件用 PDL，共 **56 处 `pdl_sync` + 8 处 `pdl_trigger`**（含 `vendor/DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` 1 处）。少数 kernel（如 `state_cache.py`）的 PDL 是按数据规模条件启用的（`block_size*feature_dim >= 65536`），本身已有非 PDL 路径；多数是无条件显式调用。

### 单 kernel 改写可行性（真机验证）

- **方法**：服务器 cuda 包目录生成 `<name>_pdlfree.py`（`sed` 文本替换 `T.pdl_sync()`→`T.sync_grid()`、`T.pdl_trigger()`→`pass`，不改原文件），直接导入编译运行。
- **norm（完整数值对照，PASS）**：`norm_pdlfree.py` 在 SM89 编译 5.3s；B=2, S=128, d=64 fp16 输入，`xdim=3` 下与 torch `x/rsqrt(mean(x²)+eps)·w` 对照 **max abs err = 0.0，0% mismatch，y_origin==y**。注意 `xdim=2` 时 kernel 只写 batch 0（`bb=0`），多 batch 必须 `xdim=3`——第一版探针因此误报 50% mismatch，非 PDL 语义问题。
- **另 4 个 kernel 编译全通过**：`kv_index_cat_tl`（3.6s）、`dflash_input_store_tl`（2.3s）、`load_kv_index_tl`（2.5s）、`fused_sample_tl`（9.7s）——均最小构造参数，SM89 编译 OK。**"去 PDL 改写"是机械工程而非科学障碍**。
- 5 个 PDL-free 变体存本目录 `pdlfree-variants/`；探针脚本 `probe_norm_pdlfree.py`（含 torch 对照）、`probe_kernels_compile.py`。
- 边界：数值对照仅 norm 一个 kernel 做了；其余 4 个只验证编译未跑数值；`pdl_trigger` 删除在顺序 launch 语义下安全（下游 kernel 正常等待），但**失去 PDL 的 kernel 间重叠优化**（性能影响未测）；未逐个验证 43 个 kernel 的全部参数组合。

### 数值对照扩展（2026-10-22 再续，M5）

把上一节的"其余 4 个只验证编译"补上数值对照——**5 个去 PDL 变体全部数值 PASS**，`probe_pdlfree_numeric.py`：

| kernel | 对照 | 结果 |
|---|---|---|
| `kv_index_cat_tl` | 拷贝 kv_index + 非 -1 的 kv_ca_index 加 total_k+total_k_swa（torch 参考） | eq=True / eq=True |
| `load_kv_index_tl` | (offset,len) 展开为索引行，block_size=0 连续模式、列≥len 填 -1（torch 参考） | full eq=True |
| `dflash_input_store_tl` hc_mult=1 | 直接拷贝（torch 参考） | max err=0.0 |
| `dflash_input_store_tl` hc_mult=2 | 按 hc_mult 均值（torch 参考） | max err=0.0 |
| `fused_sample_tl` | logprobs vs torch `log_softmax(logits/0.9)`、lse_workspace vs 推导参考、candidate 合法性 | logprobs max err=1e-6, lse0/lse1 err=0.0, cand in [0,V)=True |

**fused_sample 是 43 个 kernel 里最复杂的**（两级 launch + LSE streaming 归约 + `lse_workspace` 跨 launch 共享 + 2 处 PDL trigger 删除 + 2 处 pdl_sync→sync_grid），数值精确对照通过，说明去 PDL 后的跨 launch 依赖在顺序语义下正确保持。`lse_workspace` 语义澄清：`lse1=-max(logit)`（原始 logit 未缩放）、`lse0=-log(Σ exp((logit-max)/t))`，由 `logprobs=(logit+lse1)/t+lse0 = log_softmax(logit/t)` 推导核对——初版探针参考算错（把 max/lse 按缩放后 logit 算），修正后 err=0.0，非 kernel 问题。

边界（延续）：数值对照覆盖 5 个 kernel 的选定参数组合（stride/形状按真实 wrapper 约定，如 kv_index_cat 输出 stride pad 到 4 的倍数）；其余 38 个 kernel 仍只静态确认 PDL 用量（28/30 文件、56+8 处调用）；`fused_sample` 的采样分支只验证了 candidate 合法性，未对照分布（自定义 RNG 与 torch 不同）；性能影响未测。
