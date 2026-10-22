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
