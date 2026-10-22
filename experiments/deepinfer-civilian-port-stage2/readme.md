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

### 全量改写编译扫查：43/43 kernel 全部 SM89 编译通过（2026-10-22 三续，M5）

把上一节的"其余 38 个只静态确认"推进为**全量真机编译验证**：28 个 PDL 源文件的 **43 个 `_tl` 工厂全部生成 `_pdlfree` 变体（文本级 sed，不改原文件），在 SM89 (4090) 编译通过**，`scan_pdlfree_all.py`：

- **覆盖**：28/28 个 PDL 文件、43/43 个 `_tl` 工厂（AST 枚举，含 `fused_transform` 21 参数全 kwonly、`load_paged_kv_cache`/`store_kv_cache` 需 `KVCacheFormat` 常量、`state_cache` 条件启用分支 `block_size_inference*feature_dim=64*1024=65536≥65536` 走 enable 路径——条件 PDL 在启用时也真实编译 `T.sync_grid()`）。
- **结果**：**43/43 OK**（串行跑完；新编译 2.2–12.1s，`candidate_block_max` 最慢 12.1s；5 个已数值验证的 kernel 命中 tilelang 缓存 0.0–0.9s）。
- **替换计数核对**：sed 替换 `T.pdl_sync()`→`T.sync_grid()` 后各文件 `sync_grid` 计数与改写前 `pdl_sync` 调用计数逐一相等（合计 51 处）；`T.pdl_trigger()`→`pass` 8 处均为独立语句（`if ... : T.pdl_trigger()` 块内），替换后语法/语义安全。变体文件 `grep pdl_sync|pdl_trigger` 仅剩 `enable_pdl_sync` 变量名，无调用残留。
- 28 个变体存本目录 `pdlfree-variants/`（原 5 个与服务器 5/5 字节一致，新 23 个从服务器拉回）；扫查脚本与完整日志（含每个 kernel 的 TileLang 编译起止时间）见 `scan_pdlfree_all.py` / `scan_pdlfree_all.log`。

**结论升级**：从「5 个 kernel 编译+数值、38 个静态确认」提升为「**43 个 kernel 全量文本级改写 + SM89 编译全过**」——「去 PDL 改写」的工程量（28 文件、51+8 处调用）与可机械性在真实硬件上全量实证，PDL 不是 4090 移植的唯一/隐藏障碍（至少编译层）。数值语义保持仍以 5 个 kernel 的对照为证据，其余 38 个的数值语义未逐一对照（若继续完整移植需逐个补）。

边界（三续）：编译扫查使用各工厂的最小合法参数（含 `fused_transform` 无 norm/rope/cast 的最简路径、`load/store_kv_cache` 用 V41_FORMAT e4m3 布局）；未遍历全部参数组合与量化布局变体；`state_cache` 条件分支用 enable 参数覆盖，disable 路径（无 sync_grid）本就无需改写；vendor swiglu（DeepGEMM 第三方包，非 DeepInfer 主包 import 链）1 处 `pdl_sync` 未在本次扫查（其用 `@tilelang.jit` 装饰、依赖 deep_gemm，深潜包外）；性能影响仍未测。

### load/store_kv_cache 全矩阵数值验证 + sync_grid grid 上限发现（2026-10-22 四续，M5）

**任务**：把 43 个 kernel 中每请求必经的 KV 缓存存取两个 kernel（`load_paged_kv_cache_tl` / `store_kv_cache_tl`，编译扫查只用了最小参数）推进为**全矩阵数值验证**，并检查上游自带测试矩阵是否在去 PDL 变体下全过。

**方法（复用上游测试而非自写参考）**：DeepInfer 上游 wrapper（`tilelang_ops/load_paged_kv_cache.py` / `store_kv_cache.py`）自带 GPU 测试函数 `test_load_paged_kv_cache` / `test_store_cache`，内部与 torch 参考实现（`load_paged_kv_cache_ref` / `store_kv_cache_ref` + `ref_cast_to_bf16`/`ref_cast_from_bf16`）逐字节对照。探针 `probe_kv_cache_pdlfree.py`：
1. 临时隐藏 cuda 包内 `*_pdlfree.py`（**ops 模块遍历 backend 包所有模块、重导出 `*_tl` 工厂，变体与原件并存会 `backend symbol collision` 断言失败**——本身即移植要点：变体不能留在 backend 包内，需替换原文件或放包外）；
2. `import ops` 后恢复隐藏文件，用 `importlib.util.spec_from_file_location` 按包内模块名加载变体（保持相对导入可用），monkeypatch `ops.load_paged_kv_cache_tl` / `ops.store_kv_cache_tl` 指向变体；
3. 直接调用上游原始测试函数（`__wrapped__` 不可用则用原函数），跑完整用例矩阵：**5 布局 × 2 block_size × 3 token 数**（V41 e4m3 / V41 FP4 / FP4 index / FP8-fp32-sf index / MXFP8 index；block_size 64/133 或 32/64；num_tokens 1/233/67890）+ store 的 random_invalid 三次 update（负值、幅度扫描、全零量化）。

**结果 A：pdlfree 变体（pdl_sync→sync_grid）小用例 20/20 PASS，大用例 10/10 FAIL**：
- **小用例全过**：load 10/10 + store 10/10（1 与 233 token，全部 5 布局 × block_size）——`torch.equal`/`allclose` 逐字节对照 PASS，含 e4m3/fp4/ue8m0 量化、token_interleaved、index/main/SWA 布局、跨页、canary 保护（store 三次 update 后 backing 逐字节相等）。**这两个 kernel 的数值语义在去 PDL 改写后保持（小输入）**。
- **大用例全 FAIL**：67890 token 时 `InternalError: CUDALaunch Error: CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`（load grid=3714/1900、store grid=4244）。**新发现：tilelang `sync_grid()`（codegen_cuda.cc:3008 生成 `cooperative_groups::this_grid().sync()`）要求 cooperative launch（`cuLaunchCooperativeKernel`），grid 必须全部 block 同时驻留——4090（128 SM、每 SM 约 8-16 block）上限约 1024-4096 grid。67890 token 的 segment/block 数超限直接启动失败；上游 PDL 版本（`cudaGridDependencySynchronize`）无此驻留限制**。**这是 sync_grid 替代引入的新硬边界（运行期而非编译期），编译扫查（最小参数、小 grid）未暴露。**

**结果 B：nosync 变体（删除而非替换 pdl_sync）全矩阵 60/60 PASS**：把 `pdl_sync` 直接删除（连 sync_grid 都不放）生成 `*_nosync.py` 后重跑同一矩阵——**load 30/30 + store 30/30 全过，含全部 67890 大用例**。语义依据：这两个 kernel 的 `pdl_sync` 注释明确是「wait for dependency launch to produce load_segments/slot_mapping」——**顺序 launch（非 PDL）下前序 kernel 由 stream 顺序天然保证完成，同步不需要**；删除后无 cooperative-launch 限制。**结论修正：对「等待前序 kernel 数据」型的 pdl_sync（此类是多数），正确去 PDL 改写是删除而非替换成 sync_grid**；sync_grid 只适用于 kernel 内部 grid 级数据依赖（同一 launch 内跨 block 通信）的情形。

**证据**：`probe_kv_cache_pdlfree.py`（monkeypatch + 全矩阵驱动，`KV_PROBE_VARIANT=pdlfree|nosync` 切换）、`probe_kv_cache_pdlfree.log`（pdlfree 20/20+10 FAIL）、`probe_kv_cache_nosync.log`（nosync 60/60）、`pdlfree-variants/load_paged_kv_cache_nosync.py` / `store_kv_cache_nosync.py`（删除 sync 的变体）。

**边界**：数值对照=上游测试矩阵（布局/形状/量化覆盖广，但每 kernel 单次编译后多 token 数复用同一 kernel，不覆盖全部参数组合）；cooperative-launch 上限值是 4090（128 SM）实测，其他 SM 数设备需重测；`nosync` 删除仅对「等待前序 kernel」型 pdl_sync 安全（load/store 注释明确）；其余 41 个 kernel 的 pdl_sync 属于哪种类型、能否安全删除未逐个核对（多数同为依赖 launch 产生的前序数据，需按注释/数据流逐一确认）；性能影响（PDL 重叠）仍未测。

### 全量 nosync 改写验证：43/43 编译 + 7 kernel 数值保持（2026-10-22 五续，M5）

**任务**：把四续「依赖型 pdl_sync 应删除而非替换 sync_grid」的结论从 2 个 kernel 推广到全部 43 个 kernel——对 28 个 PDL 源文件的全部调用点做语义分类，生成全量 nosync 变体，验证删除改写在编译与数值两个层面都对。

**方法**：
1. **语义分类（静态）**：拉回 28 个 PDL 源文件（`pdl-sources/`），逐处核对全部 **55 处 `T.pdl_sync()` + 8 处 `T.pdl_trigger()`** 的上下文：
   - 全部 55 处 `pdl_sync` 均位于 kernel 开头（紧随 `T.Kernel(...)` 与片段分配之后），等待的是前序 launch 写入的全局数据（如 `load_segments`/`slot_mapping`/`cu_lens`/`lse_workspace`/计数器）——**依赖型**，顺序 launch 下由 stream 顺序天然保证；
   - 8 处 `pdl_trigger` 均在 kernel 内循环/分支尾部「提前释放下一个 launch」（如 fused_sample 的 `# Release the next launch before the final block reduction`、fused_transform/norm/kv_index_cat 的末尾）——顺序 launch 下无重叠收益，删除等价；
   - **没有任何一处 pdl_sync 是「同一 launch 内跨 block 数据通信」型**（那才需要 sync_grid）。`sampler.py` 甚至注释明示上游刻意「without a cooperative grid barrier that would prevent PDL launches」——上游自己都避免 cooperative barrier。
2. **生成 nosync 变体**：`gen_nosync_variants.py` 从源文件文本级删除 `T.pdl_sync()` 行（若删后块空则补 `pass`）、`T.pdl_trigger()` → `pass`，AST 语法校验 + 残留检查，生成 28 个 `*_nosync.py`。替换计数与三续 pdlfree 逐一对齐：**51 处 pdl_sync 删除 = 三续 51 处 sync_grid 替换的对应位置（除 load/store 两处四续已验证）**，8 处 pdl_trigger 均保留 pass 占位。
3. **43/43 编译扫查**：`scan_nosync_all.py`（复用 scan_pdlfree_all.py 的 43 工厂 CASE，模块后缀 `_nosync`）在 4090-1 SM89 全量编译——**43/43 OK**（首次约 2 分 20 秒；重跑命中 tilelang 缓存秒级完成，第二次 43/43 OK 确认可复现）。
4. **数值验证**：`probe_nosync_numeric.py`（probe_pdlfree_numeric.py 的 nosync 版）给其余 4 个 kernel 补数值对照；`probe_norm_nosync.py` 重跑 norm（此前仅 pdlfree 版验证）。

**结果**：
- **43/43 nosync 变体 SM89 编译全过**（含 fused_transform 全 kwonly、load/store KVCacheFormat、state_cache 条件启用分支、fused_sample 两级 launch）。
- **7 个 kernel nosync 数值语义保持**：
  - norm：B2/S128/d64 fp16 vs torch RMSNorm **max err=0.0、0% mismatch**（多 batch xdim=3）；
  - kv_index_cat：拷贝+偏移加 **eq**；load_kv_index：索引展开 **eq**；dflash_input_store hc_mult=1/2 **max err=0.0**；
  - fused_sample（最复杂：两级 launch + lse_workspace 跨 launch 共享）：logprobs vs `log_softmax(logits/0.9)` **max err=1e-6**、lse0/lse1 err=0.0、candidate 合法；
  - load/store_kv_cache：四续 nosync **全矩阵 60/60 PASS**（含 67890 大用例）。

**结论**：「去 PDL 改写」的正确形式确定——**全部 51 处 pdl_sync 均为依赖型，删除（nosync）而非替换 sync_grid**；三续的 sync_grid 替换（51 处）是「可编译但引入 cooperative-launch 运行期上限」的次优改写，四续大 grid 失败已证伪其通用性。改写规则收敛为：`pdl_trigger` → 删除（pass）；`pdl_sync`（依赖 launch 前序数据）→ 删除；仅「同一 launch 内跨 block 通信」才考虑 sync_grid（本项目 43 个 kernel 无此用例，且需注意 grid 上限）。

**证据**：`pdl-sources/`（28 个 PDL 源文件副本）、`gen_nosync_variants.py`、`pdlfree-variants-nosync/`（28 个 nosync 变体，与服务器逐字节一致）、`scan_nosync_all.py` + `scan_nosync_all.log`（43/43 OK）、`probe_nosync_numeric.py` + `probe_nosync_numeric.log`（5/5 PASS）、`probe_norm_nosync.py` + `probe_norm_nosync.log`（max err 0.0）。

**边界**：静态分类依据注释与数据流（依赖 launch 产物的全局 buffer），未对每个 kernel 做「删除后仍保持正确」的数值证明——数值保持只覆盖 7 个代表性 kernel（norm、kv_index_cat、load_kv_index、dflash_input_store、fused_sample、load_paged_kv_cache、store_kv_cache），其余 36 个仅编译全过 + 依赖型分类；无一处是网格内同步型的判断基于源码核对，若移植完整实现需逐个补上游测试矩阵数值对照；性能影响（PDL 重叠）仍未测；未改 DeepInfer 上游源码（变体为未跟踪新文件）。
