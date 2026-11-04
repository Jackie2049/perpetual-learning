# TensorRT-LLM：一个「写了分支但永远走不到」的开关，以及 34 条为什么不是缺陷（2026-11-05）

<!-- contribution-summary:start -->
**【仓库】** [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)（NVIDIA 的大模型推理框架，PyTorch 前端 + CUTLASS/CuteDSL 内核）

**【贡献点】** pyflag 在 `f6405a5965`（2026-11-04）上扫出 **352 个开关 / 35 条 `silently_inert`**。逐条绑行号后**真缺陷 1 条**：FP4 MLA 的 `use_split_scale_stages` 形参存在、两处 `elif` 真的按它分支，但**全仓没有任何调用点传它**，而同族四个兄弟开关在调用点都被显式写成 `True` ⇒ 那两处 `elif` 在任何真实配置下都永不执行。合入 PR 当天（`4bbe1af6`，2026-09-18）就是这个样子，**不是回归**。[实验与复现](../../experiments/trtllm-pyflag-triage/README.md)

**【类型】** B 类（主动发掘）：一个可证伪、行号可机读的死分支；附 34 条候选的结构原因归档 + 扫描器新增第十/十一类判据

**【价值评估】** 5 / 10 — 对外价值。事实清晰（形参在、分支在、无生产者、兄弟开关全传 `True`、合入即如此），但影响面窄（FP4 MLA 的 scale-from-smem 路径，用户看不到差别）、属 Blackwell/Rubin 新算子。上游查重 `"use_split_scale_stages"` **0 条 issue**。**对外（issue / PR）待用户逐项授权。**

**【进展】** 2026-11-05 完成：扫描（12 分 39 秒 / 6084 个 `.py`）→ 35 条逐条归因（`classify.py` 全覆盖）→ 死分支行为在 CPU 上跑真实源码验证（`probe_split_scale.py` 9/9，两条注入各让一条判据变红）→ 判定脚本 28 checks / selftest 6/6。**未提交上游。** 同步给 pyflag 增加了第十一类易错点检测（`shadowed_arms`），本批这 4 处死分支就是它自己扫出来的。

**【大白话讲明白】**

一个开关参数，代码里声明了、判断了，但从来没人给它传值，于是那段"判断"永远得不到执行。这本身是扫描器找的活儿，前面三批（SGLang、vLLM、LMDeploy）已经找出 51 条候选，其中 3 条是真的。这一批换到 TensorRT-LLM，35 条候选。

`use_split_scale_stages` 值得说的是**为什么它能让人确信是"漏了"而不是"本来就这样"**。

同一个函数里有四个长得几乎一样的兄弟开关，管的是同一件事——从共享内存里按第几级取哪一段数据：

```python
use_a_stage_word_offset=True,
use_b_stage_word_offset=True,
use_sfa_stage_byte_offset=True,
use_sfb_stage_byte_offset=True,
```

调用这个函数的地方，把这四个**全都显式写了 `True`**。而 `use_split_scale_stages` 就夹在这组参数中间，没有被写。它俩还是同一段 `if` 里的备选：先看兄弟开关，兄弟开就按兄弟算，兄弟没开才轮到它。

于是结果就是：不是"这个开关没接上"，而是**这段代码连带的兜底一起死了**——因为它排在兄弟后面，兄弟恒真，它永远轮不到。

这也解释了为什么前几批的判据不够用：只看"有没有人传值"看不出这一层，还得看**同一个 if/elif 链上，前面那个条件是不是在所有调用点都被钉死了**。这一步现在做进了 pyflag，一条命令就能列出来。

至于剩下的 34 条，绝大多数还是老几类：存进 `self.` 之后按属性判、同一文件里构造时就传了、从配置字典里读、通过 `getattr` 读、所在函数本身就是死代码。这批**没有新增"用户设了不生效"的例子**。
**【What–Why–How】**

- **What：** TensorRT-LLM `f6405a5965` 上一个可证伪的死分支（`use_split_scale_stages`），以及同批 34 条 `silently_inert` 候选为何都不成立的结构原因。
- **Why：** 现有布尔开关扫描只判"有没有生产者"，看不出"有生产者也没用"——同一 `if/elif` 链上被兄弟条件恒真挡住的分支，代码本身正确、不崩、不告警，只是那段优化永远走不到。
- **How：** 逐条绑行号归因 35 条（`classify.py`），把死分支的真实源码文本抽出来在 CPU 上按调用点实际常量执行（`probe_split_scale.py`，配正负对照），再把"兄弟恒真"这一形态做成扫描器检查（`shadowed_arms()`），让下一批不必人眼找。

<!-- contribution-summary:end -->

## 这条缺陷在哪里

| 项 | 位置 |
|---|---|
| 形参声明 | `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py:2588` |
| 两处死分支 | 同文件 `:2668`（SFA 段）、`:2675`（SFB 段），均被兄弟开关的 `if` 挡在前面 |
| 调用点（都没传它） | `:3033`（QK）、`:4522`（PV） |
| 孪生文件同形 | `fp4_mla_cutedsl_mufu16_fused_v_transpose.py:2970` / `:3050` / `:3057` |
| 合入版本 | merge commit `4bbe1af658`（PR #19070，2026-09-18），当日即同形 |

分支优先级（`fp4_mla_cutedsl_mufu16.py:2666-2676`）：

```python
if cutlass.const_expr(use_sfb_stage_byte_offset):
    sSFB_stage = sSFB.subview(sfb_stage_byte_offset)
elif cutlass.const_expr(use_split_scale_stages):        # ← 永不执行
    sSFB_stage = sSFB.subview(sfb_stage_bytes * sfb_stage)
else:
    sSFB_stage = sSFB.subview(sfb_stage_bytes * stage)
```

## 验证了什么

`probe_split_scale.py` 把 `:2668` 与 `:2675` 两块**真实源码文本**抽出来，按调用点实际传的常量执行：

- **真实取值下**（兄弟 `True`、本开关未传即默认 `False`、`sfa_stage`/`sfb_stage` 用默认 `0`）：两块都不取 split 分支；
- **兄弟关掉、开关打开**：两块都真的走 split 分支，且偏移是 `512*0`，与 else 分支的 `512*stage` 不同 ⇒ 这确实是有效逻辑，不是废码；
- **正向对照**：探针本身若被改成兄弟恒赢或强行打开 split，对应判据变红（`--selftest` 2/2）。

**两个调用点不是同一种死法**（判据 `F1h`）：QK `:3033` 是走进去之后被兄弟 arm 挡住；PV `:4522` 则显式传了 `stage_sfa_from_smem=False`／`stage_sfb_from_smem=False`，整个 `if` 块根本没进——那个 `elif` 在更上一层就已经死了，与兄弟开关无关。⇒ 对外措辞只能说「QK 路径上这个开关没有入口」，**不能笼统说"两个调用点都收不到值所以整段是死的"**：两处的修法不一样，混在一起会被维护者当场问出破绽。

判定脚本 `verify_findings.py` 另外把"形参在中间而非末尾""兄弟全传 `True`""孪生文件同形"这些都写成了可机读判据（28 checks / selftest 6/6 注入全部命中；`S0.snapshot_pristine` 在快照被改动时直接 exit 2）。

## 为什么不能直接写成「bug」

- 分支默认值 `False` 意味着**当前所有配置都走兄弟分支**，那是作者设计的路径；这个 `elif` 更像是给未来"不用 byte offset"变体准备的。要提，也只能提"这个开关目前没有入口，是否应删掉或补上调用点"，**不能写"设置后不生效"** —— 用户本来就没法设它。
- 影响面在 Blackwell/Rubin 的 FP4 MLA 内核，`fp4_mla` 后端由 #19070 于 2026-09-18 才引入，上游仍在快速变动（同期 5 个 open PR 在改这块）。

## 34 条不是缺陷的原因（归档）

按 `classify.py` 的归因，35 条全部有结论（合计 35）：

| 归因 | 条数 | 代表 |
|---|---|---|
| **真缺陷** | **1** | `use_split_scale_stages` |
| 自存 `self.x` 后按属性分支 | 5 | `enable_r0`、`enable_smem_cache` |
| 同文件构造器内自定值 | 6 | `enable_multi_cta`、`use_combine`、`use_mix` |
| 死代码（全仓零调用） | 3 | `use_weight_only`、`use_gemm_woq_plugin`、`use_last_number` |
| 计算值后内部传递 | 2 | `no_fire = not publish` |
| `**dict` 展开 / 字典键 | 2 | `use_causal_mask`、`use_direct_expert_metadata` |
| 构建脚本面 | 2 | `use_ccache`、`use_3rdparty_cache` |
| 配置字典白名单 | 2 | `use_bias_at_final`、`use_system_prompt` |
| `getattr` 属性读取 | 2 | `use_custom_cublas_mm`、`use_gemma_rms_norm` |
| 量化模式内部 helper | 2 | `use_fp8_kv_cache`、`use_int8_kv_cache` |
| 构建期字段 / 位置实参 / 显式 `None` / 配置对象 / 装饰器 / 已废弃 / 私有 helper / 内部脚本 | 各 1 | 见 `triage.json` |

## 方法读数

1. **失效条目里真缺陷的比例稳定在 0–8%**：SGLang 34→2、vLLM 12→1、LMDeploy 5→0、TensorRT-LLM 35→1。四批合计 **64 条候选 → 4 条真缺陷**。扫描器输出只能当候选入口。
2. **这一批新增的是"死分支"这一种形态**：不是"没人传值"，而是"传了也没用，因为链上前面那个条件恒真"。已做进 pyflag（`shadowed_arms`），本批这 4 处就是它自己扫出来的。
3. **API 稳定性参考文件不可信**：`tests/unittest/api_stability/references/trtllm_serve_cli.yaml` 写 `num_serve_frontends` 默认 `8`，而 `llm_args.py` 里字段默认是 `1`。两者服务不同层（CLI vs LLM API 默认），不算 bug，但核对默认值时不能只看一处。
