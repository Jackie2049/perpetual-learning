# TensorRT-LLM pyflag 第四批扩池（2026-11-05）

pyflag 扩池顺序：SGLang → vLLM → LMDeploy → **TensorRT-LLM**（本批）。前三批的真缺陷率是
34→2、12→1、5→0，本批 **35→1**，其中这一条还是**加了没接**而非"曾经能用后来坏了"。

## 快照与读数

| 项 | 值 |
|---|---|
| 仓库 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| commit | `f6405a5965`（2026-11-04 15:08 -0500，`[nvbugs/7044232][infra] Waive 1 failed case(s) on B300 (#21623)`） |
| 规模 | 6085 个 `.py`（实际扫到 6084 个，根目录 1 个无目录层的文件未检出） |
| 扫描耗时 | 约 12 分 39 秒 |
| 开关总数 | **352** |
| `silently_inert` | **35**（`never` 23 / `always` 8 / `mixed` 4） |
| 逐条取证后 | **真缺陷 1 条**、结构原因 30 条、死代码/内部面 4 条 |

原始读数靠不住这件事在第四批上第三次复现：35 条里 34 条都经不起逐条绑行号。

## 怎么复现

```bash
# 1) 取快照（HTTPS 在本机不通时 SSH 可用；blobless + sparse 避免拉全仓）
git clone --depth 1 --filter=blob:none --no-checkout git@github.com:NVIDIA/TensorRT-LLM.git trtllm-probe
cd trtllm-probe
git ls-tree -r --name-only HEAD | grep '\.py$' | xargs -n1 dirname | sort -u > /tmp/dirs.txt
cp /tmp/dirs.txt .git/info/sparse-checkout
git config core.sparseCheckoutCone false
git checkout            # 6084 个 .py，约 428 MB

# 2) 扫描（显式给归档名；裸 --json 现在写 scan-<目录名>.json）
python3 ../../tools/pyflag/scan_unwired_flags.py . --json ../../tools/pyflag/scan-trtllm-f6405a5965.json

# 3) 判定（28 checks，含 S0 快照未污染前置）与行为探针
python3 verify_findings.py --repo /path/to/trtllm-probe --report findings.json   # 28 checks, 0 failed (S0 先验快照干净)
python3 verify_findings.py --selftest --report findings.json                     # 6/6 注入全部命中
python3 probe_split_scale.py --repo /path/to/trtllm-probe                        # 9/9
python3 probe_split_scale.py --selftest                                          # 2/2
python3 classify.py /path/to/trtllm-probe                                        # 35 条全部归因，缺一条 exit 1
```

## 唯一真缺陷：`use_split_scale_stages` 加了没接

**是哪一条。** `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py:2588`
声明 `use_split_scale_stages: ctm.Constexpr = False`，函数体 `:2668`／`:2675` 两处
`elif cutlass.const_expr(use_split_scale_stages):` 真的按它算 SFA/SFB 的 smem 偏移。
**全仓没有任何调用点传这个关键字**——两个调用点（`:3033` QK、`:4522` PV）都没有，
孪生文件 `fp4_mla_cutedsl_mufu16_fused_v_transpose.py:2970` 同样。注意"都没传"不等于"同一种原因"，
两者死在不同层，见下文 F1h 那张表。

**为什么看着像"漏接"而不是"设计如此"。** 同一个函数里还有四个同族 constexpr，
QK 调用点把这四个**都显式传了 `True`**（`:3066-3069`）：

```
use_a_stage_word_offset=True,
use_b_stage_word_offset=True,
use_sfa_stage_byte_offset=True,
use_sfb_stage_byte_offset=True,
```

`use_split_scale_stages` 就夹在这族参数中间（`b_stage` 之后、`sfa_stage` 之前，
`:2586 < :2588 < :2590`），不是后来追加到末尾忘了传的形状。

**行为上会怎样：两个调用点不是同一种死法（`F1h` 判据）。** 两处分支的优先级是：兄弟开关
（`use_sf[ab]_stage_byte_offset`）优先，其次才是 `use_split_scale_stages`，最后是 `bytes * stage`。

| 调用点 | 外层守卫 `stage_s*f_from_smem` | 四兄弟 | 这一层的 `elif` 为什么不可达 |
|---|---|---|---|
| QK `:3033` | 形参透传，默认 `True` ⇒ **进块** | **全 `True`** | 兄弟 arm 恒先命中 |
| PV `:4522` | 显式 **`False`** ⇒ **整块不进** | 未传 | 比兄弟更早一层就死了，与兄弟无关 |

所以「两个调用点都不传」写法会把两种原因折成一句，而**修法要动的地方不同**：QK 那条
要么给调用点补传、要么承认兄弟永远赢；PV 那条得先让 `stage_s*f_from_smem` 变成 `True` 才谈得上
这个开关。`probe_split_scale.py` 把那段真实源码文本抽出来、按 QK 实际传的常量执行九条判据：

- 真实取值 → 两块都不取 split arm（`B7`）；
- 把兄弟开关关掉、开关打开 → split arm 真的活，且偏移与 else arm 不同（`B8`/`B9`）
  ⇒ 这确实是有意义的分支，不是废码；
- `--selftest` 两条注入（强行打开 split / 强行让兄弟赢）各让一条判据变红。

**历史归属：合入当天就是这样。** 该文件由 PR [#19070](https://github.com/NVIDIA/TensorRT-LLM/pull/19070)
（merged 2026-09-18，merge commit `4bbe1af658`）一次性新增 7859 行。取该 commit 的同一文件，
`use_split_scale_stages` 只出现在 **2594（声明）／2674／2681（两处 elif）**，与今天同形
⇒ 不存在"曾经接上过"的版本，是**新增时即未接线**。

**对外价值：5/10。** 可证伪、行号可机读、孪生文件同形；但影响面窄（FP4 MLA 的
scale-from-smem 路径，且默认走兄弟分支，用户看不到差别），且属 Blackwell/Rubin 新算子。
上游查重：`"use_split_scale_stages"` 0 条 issue。**对外（issue/PR）待用户逐项授权。**

## 35 条的结构原因分桶

| 归因 | 条数 | 代表 | 为什么 pyflag 读成"无生产者" |
|---|---|---|---|
| **真缺陷** | **1** | `use_split_scale_stages` | 兄弟开关恒先命中，两处 `elif` 永不执行 |
| 自存 `self.x` 后按属性分支 | 5 | `enable_r0`、`enable_smem_cache` | 形参存进 `self.` 后分支读的是属性 |
| 同文件构造器内自定值 | 6 | `enable_multi_cta`、`use_combine` | 同一文件里既有 `=True` 也有 `=False` 的构造 |
| 死代码（全仓零调用） | 3 | `use_weight_only`、`use_gemm_woq_plugin`、`use_last_number` | 所属函数本身没人调用 |
| 计算值后内部传递 | 2 | `no_fire = not publish`、`use_graph_tensor_cores` | 局部变量算出来的，不是调用实参 |
| `**dict` 展开 / 字典键 | 2 | `use_causal_mask`、`use_direct_expert_metadata` | `compiler.py:92` 写字典键，`:106` 展开 |
| 构建脚本面 | 2 | `use_ccache`、`use_3rdparty_cache` | 只在 `scripts/build_wheel.py` 的 `main()` |
| 配置字典白名单 | 2 | `use_bias_at_final`、`use_system_prompt` | 从 `cfg[...]` / `extra_params.get(...)` 取 |
| `getattr` 属性读取 | 2 | `use_custom_cublas_mm`、`use_gemma_rms_norm` | `getattr(module, "use_x", False)` |
| 量化模式内部 helper | 2 | `use_fp8_kv_cache`、`use_int8_kv_cache` | `QuantMode.from_description` 内部 |
| 构建期字段 | 1 | `enable_attn_logit_softcapping` | `kernel_spec.*` 属性，内联进 C++ |
| 位置实参转发 | 1 | `enable_lm_head_tp_in_adp` | `mapping.py:633` 位置传参 |
| 显式 `None` 哨兵 | 1 | `use_replay_state_update` | `use_replay_state_update=None` 也是实参 |
| 配置对象赋值 | 1 | `use_paged_kv` | `cfg.use_paged_kv = True`（`:4529`） |
| 内部装饰器形参 | 1 | `enable_log` | `log_op(func, enable_log=False)` |
| 已废弃无入口 | 1 | `enable_rl_control_endpoints` | 日志明写 deprecated，CLI 无路径 |
| 私有 helper | 1 | `use_environ` | `run_process(...)` 的调用点直接传 `env=` |
| 内部脚本 | 1 | `use_system_resolver` | 只在 `scripts/attribution/` 用 |

合计 1 + 34 = 35。**判定不用人眼数**：`classify.py` 对每条记录 `verdict` + `evidence`（文件:行），
并校验该文件确实存在，缺一条就 `exit 1`。

## 本批新增的方法读数

1. **第十类易错点：`**dict` 展开**。`{"use_causal_mask": ...}` 写的是字典键，调用点根本
   不出现这个标识符；AST 一律读不到。与前几批的 `**kwargs`（签名没有≠收不到）互补：
   那是**接收端**的兜底，这是**发送端**的兜底。
2. **"显式传 `None`"也是生产者**。`use_replay_state_update=None` 语义上是"未指定"，
   但结构上它确实是调用点传的实参 ⇒ pyflag 按 `reads` 判会判成无生产者。
3. **兄弟开关恒先命中 = 真缺陷但行为等价**。这一类比前几批的"加错/丢线"更隐蔽：
   代码本身是对的、不崩、不出告警，只是**那段优化的分支永远走不到**。判定时不能只看
   "有没有生产者"，还要看**同一 if/elif 链上前面的 arm 是否恒真**。
4. **子串匹配又坑了一次**：`extract_answer(` 数到 6 处，其中 5 处是 LongBenchV2 的
   `_extract_answer` 方法，必须加 `(?<![\w.])` 前向断言。

## 自错误清单（都已改，判据进了脚本）

| # | 错 | 怎么发现的 |
|---|---|---|
| 1 | 探针里把 sfa/sfb 两个分支块的执行顺序写反，且同一块跑了两遍 | B2/B3 判据直接报 `guarded by stage_sfb_from_smem`，与预期 sfa 不符 |
| 2 | B9 原想比"flag-on vs 兄弟关掉"，但两者在块内走同一条 `if`，恒相等 | B9 挂红，改成"split arm vs else arm 偏移是否不同" |
| 3 | 探针最初用编造的 `stage=3, sfa_stage=2`，而真实调用点 `sfa_stage/sfb_stage` 用默认 **0** | 回读 AST 实参发现三个都是 `<NOT PASSED>`，改成真实常量并加 `B0` 判据 |
| 4 | 一度以为 `launch_server` 单前端不建 server，会推出"sleep mode 全坏"的大结论 | 查缩进：`uvloop.run` 在 `if is_launcher` **之外**（col 16 vs 20），且 CLI 默认 `num_serve_frontends=8` |
| 5 | `extract_answer(` 子串匹配把 `_extract_answer` 算进去 | D1 报 6 处，逐条看是 LongBenchV2 同名方法 |
| 6 | FP5 猜了不存在的函数名 `_compile_filter_topk` | 检查项红，回读文件改成 `filtered_topk_kernel` |
| 7 | **负向对照把被测文件改坏了、还原没执行**：`sed -i ''` 在 zsh 下 `''` 不是参数，备份 `cp` 从未执行；随后 `cp` 还原也因同一行 `&&` 链从未跑到 | 下一轮读该文件时发现四兄弟变 `False`，先误判成"记录写反了"，再比对上游 `git hash-object` 与 HEAD blob 不一致才定位 |

### 由此加的硬前置：`S0.snapshot_pristine`

第 7 条暴露的不是笔误，而是**判据设计的洞**：整批 26 条 checks 与 `probe` 9/9 只校验源码里的
字面量形态，**没有任何一条校验"读的是不是那份快照"**。文件被改坏后全部照绿，而"兄弟从 `True`
变成 `False`"看起来还像一个结论。

现在 `verify_findings.py` 在派生任何结论之前先跑 `git -C <repo> status --porcelain`：
非空即打印 `FAIL S0.snapshot_pristine`（含前 4 个路径）并 **`exit 2`**，不产出检查结果。
负向对照已验证：故意把四兄弟改回 `False` → `exit 2`，"故意污染 → 被拦下 → 恢复 → 28 checks 0 failed"。

⇒ 复用到其他批次时同一条纪律：**凡结论建立在外部仓库的文本上，先证明树的干净**；
负向对照要改文件时，备份与还原必须是两条独立命令，且改完立刻 `git status` 自检。

## 验证范围（必须带着读）

**纯静态 + CPU 源码文本执行。** 未装 TensorRT-LLM（本机 Python 3.9，`tensorrt_llm/__init__.py`
第一行 import torch 即失败），未编译、未起服务、未上 GPU/NPU、未触上游任何写入。
`probe_split_scale.py` 执行的是**从快照里抽出来的真实源码文本**，不是对规则的模拟；
但它是 CPU 上的普通 Python 语义，不是 cutlass DSL 的真实编译结果。

## 文件

- `verify_findings.py` —— 28 条判据（含 `S0.snapshot_pristine` 前置），`--selftest` 6 条注入
- `probe_split_scale.py` —— 9 条行为判据，`--selftest` 2 条注入
- `findings.json` —— pyflag 原始读数（352 开关）；同一份内容归档在
  [tools/pyflag/scan-trtllm-f6405a5965.json](../../tools/pyflag/scan-trtllm-f6405a5965.json)
- `classify.py` —— 35 条逐条归因（verdict + 文件:行证据，路径不存在即报错）
- `triage.json` —— 35 条逐条绑定的消费者位置、间接生产者与最终归因
