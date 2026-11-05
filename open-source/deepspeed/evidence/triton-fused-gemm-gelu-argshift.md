# DeepSpeed triton 推理：`fused_gemm_gelu` 从 2023 年引入起就对不上调用点

<!-- contribution-summary:start -->
**【仓库】** [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)（微软开源的大模型分布式训练与推理引擎）

**【贡献点】** `deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py:42` 用 **8 个位置实参**调用 `self.fused_gemm_gelu`，而 triton 实现 `deepspeed/ops/transformer/inference/triton/ops.py:17` 有 **12 个形参、其中 11 个必填**。在 `dtype=float16 ∧ use_triton ∧ attn_nw is None` 时指针指向 triton 版本，一调用就 `TypeError: missing 3 required positional arguments`。[源码](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py)

**【类型】** 问题诊断（B 类主动发掘；**不对外，用户未授权**）

**【价值评估】** 4 / 10 — 缺陷本身确凿且可机读，但从 2023-06 引入起就一直**不可从正常入口到达**（`use_triton=True` 被容器层限定为 BERT，其余模型直接 `NotImplementedError`），因此没有用户被它挡住。

**【进展】** 静态 + CPU 源码执行取证完成（4 个探针、合计 40 条判据、13 条注入全部命中）；**未上真机、未安装 DeepSpeed、未触上游**。对外动作待用户逐项授权。

**【大白话讲明白】**

DeepSpeed 有一套"用 Triton 内核加速推理"的开关（`use_triton=True`）。其中一条前馈网络（FFN）路径里，代码拿着 **8 个参数**去调用一个**要 12 个参数**的函数。从 2023 年 6 月这段代码第一次进来时就是这样，一直没人发现——因为它所在的那个分支（"这个层没有 LayerNorm 权重"）在 Triton 模式下**永远走不到**：Triton 模式只允许 BERT 类模型，而 BERT 每层都有 LayerNorm。

所以这不是"用户会被炸"的 bug，而是**一段从未被执行、也从未被测试覆盖的代码，其中还躺着一个真的签名错误**。价值在于：如果哪天 DeepSpeed 把 Triton 支持从"仅 BERT"放宽到别的模型（这看起来是项目方向），这段代码会立刻崩。

**【What–Why–How】**

- **What：** `GELUGemmOp.forward` 的位置实参个数（8）与 triton `fused_gemm_gelu` 的必填形参个数（11）不一致；C++ 等价签名恰好是 8 个，说明这 8 个实参是照 C++ 内核写的。
- **Why：** triton 版的 12 个形参里有 **7 个函数体从未使用**（`weight_scale`／`weight_out_scale`／`epsilon`／`pre_layer_norm`／`q_int8`／`async_op`／`use_triton_ln`）——像是从 `mlp_gemm_func` 的模板抄来了签名，却只写了 C++ 那 8 个参数对应的实现，于是多出来的 4 个必填形参没人补。历史：`fused_gemm_gelu` 由 PR [#3748](https://github.com/deepspeedai/DeepSpeed/pull/3748)（`69d1b9f9`，2023-06-23）引入时就是 12 个形参，**同一提交**里的调用点就是 8 个实参；此后该文件只被 #5490 重排过，签名未变。
- **How：** 用 AST 取真实函数体/类体在 CPU 上执行（`matmul_ext`、C++ 内核、`torch` 用桩），四档臂分别验证"选中 triton 就崩、其余三档正常"；再用 9 条可达性判据证明这条路径在用户配置下不可达。**边界：未安装 DeepSpeed、未在 GPU 上跑、未覆盖 DeepSpeed-MII 对 `use_triton` 的用法**（见下）。
<!-- contribution-summary:end -->

## 一、怎么来的

pyflag 扫 DeepSpeed `7e58feb6`（2026-11-05，1167 个 `.py`，**29 秒**）只得到 **44 个开关 / 1 条 `silently_inert`**（`use_triton_ln`，`mode=always`）。这批的信号量太低，不足以单独成篇；顺着 `use_triton_ln` 的消费者读下去，才在调用点上撞见本条的形参不匹配。

两条读数：

| 开关 | 模式 | 判定 | 理由 |
|---|---|---|---|
| `use_triton_ln` | always | `UNREACHABLE_DEAD_ELSE` | 三个函数的该形参都在形参表末尾，**所有调用点均按位置传参且刚好停在它前一位** ⇒ 恒取默认 `True`，`else` 分支是死代码 |
| （同批撞见）`fused_gemm_gelu` 形参不匹配 | — | `ARG_SHIFT` | 见正文 |

## 二、缺陷本身（E2E 读数）

`probe_gelu_e2e.py` 执行**真实的** `GELUGemmOp.__init__`/`forward` 类体与**真实的** triton 函数，四档臂：

| 臂 | 配置 | 指针 | 结果 |
|---|---|---|---|
| A | fp16 ∧ HAS_TRITON ∧ use_triton | `fused_gemm_gelu`（triton） | **`TypeError: missing 3 required positional arguments: 'q_int8', 'async_op', and 'transposed_mode'`** |
| B | fp16 ∧ use_triton=False | `cpp_kernel` | 正常返回 |
| C | bf16 ∧ use_triton=True | `cpp_kernel`（bf16 分支不查 triton） | 正常返回 |
| D | fp16 ∧ use_triton=True ∧ 未装 triton | `cpp_kernel`（`HAS_TRITON` 为假） | 正常返回 |

B/C/D 三个对照臂说明：**这个 `TypeError` 只在 triton 分支上发生**，不是我的桩写错了。

## 三、可达性：为什么它没伤到人

`probe_reachability.py` 的 9 条判据：

| 判据 | 事实 |
|---|---|
| R1 | `TritonMLP.__init__` **无条件**给 `attn_nw` 赋 `nn.Parameter`（`triton/mlp.py:24`）⇒ 在 triton MLP 上恒不为 None |
| R2 | `DeepSpeedMLP.__init__` 只在 `if self.config.set_empty_params:` 内把它置 None（`ds_mlp.py:33-34`） |
| R3 | `ds_transformer.py:62-71`：`use_triton` 分支选 `TritonMLP`，否则 `DeepSpeedMLP` |
| R4 | ⇒ "triton MLP ∧ `attn_nw is None`"在同一实例上不可同时成立 |
| R5 | `GELUGemmOp` 的指针选择只看 dtype/use_triton，**全文不出现 `attn_nw`** ⇒ 与 MLP 类无关 |
| R6 | `hybrid_engine.py:99` 用 `set_empty_params=True` |
| R7 | `hybrid_engine` **不传** `use_triton`（默认 False） |
| R8 | **决定性**：`module_inject/containers/base.py:122-124` —— `use_triton=True` 时非 BERT 容器直接 `raise NotImplementedError("Triton kernels are only for BERT-like models yet")` |
| R9 | `containers/bert.py` 是 LayerNorm 结构 ⇒ 走的是 `mlp_gemm_func` 那一支，不是 `attn_nw is None` |

⇒ **三元条件（fp16 ∧ triton ∧ attn_nw is None）在现有入口下不可同时成立**。缺陷是真的，路径是死的。

## 四、最要紧的一条：先把"可达"和"存在"分开

本轮真正的方法产出在这里。中途我一度准备把这条写成"fp16 + triton 用户会崩"，因为 E2E 读数（A 臂 `TypeError`）非常硬。是**可达性检查把它按回了原位**：

- **"一段代码会抛异常" 与 "用户会撞到它" 是两个问题**，后者要用入口控制流单独证明。
- 本条的入口控制 flow 有一个非常强的门槛（`raise NotImplementedError`），它不在错误发生的那一层、也不在崩溃文件里——**只在容器层**。只看崩溃文件五分钟就会得出"用户会崩"的结论。
- ⇒ 判据顺序应当是：**先证存在（E2E），再证可达（入口链），最后才定价值**。顺序反了会浪费大量时间在"怎么修"上。

## 五、本批 5 处自错（都由自检抓出）

1. **判据把"结论成立"当成"失败"**：C4 原本写成"8 个实参绑不上 12 个形参 = FAIL"，而绑不上正是要找的事实 ⇒ 退出码语义反了，`verdict` 永远 `CHANGED`。改成 `arity_mismatch`（数不匹配本身就是判据）。
2. **边界差一**：`use_triton_ln` 的 C4 写成 `worst < idx`，但形参在最后一位、调用点恰好传到前一位 ⇒ `worst == idx` 时也应成立。改成 `worst <= idx` 并写明"第 k 个实参落在第 k-1 号形参"。
3. **`inspect.bind` 在注入下抛异常而不是报红**：注入把实参改成 9 个后，C++ 侧签名直接 `TypeError`，整个探针崩掉 ⇒ 三处 `bind` 全部包了 try。
4. **判据太松**：R2 原本只查"有 None 赋值、也有 Parameter 赋值"，注入 `if True: self.attn_nw = None` 时**照样全绿** ⇒ 收紧为"None 那条必须带 `set_empty_params` 守卫"。这条是 selftest 抓到的第 1 个真问题。
5. **先入为主把 else 分支当成错位**：`qkv_gemm_func`/`mlp_gemm_func` 的 RMSNorm 分支传 7/11 个实参，我一度判成"位置错位会算错"。实际那个分支的 `self.qkv_gemm_func` 指向的是 **C++ 的 `rms_qkv_gemm_*`**，签名本来就不同 ⇒ **判"参数错位"必须同时看函数指针在每条分支上指向谁**，不能只看形参表。

## 六、复现

```bash
git clone --filter=blob:none --no-checkout https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed && git checkout 7e58feb65f6bfb17671ef1b19dfbf123b569ad48

python3 probe_use_triton_ln.py --repo <repo>        # 17 checks，exit 0
python3 probe_gelu_argshift.py --repo <repo>        #  9 checks，exit 0
python3 probe_gelu_e2e.py      --repo <repo>        #  5 checks，exit 0
python3 probe_reachability.py  --repo <repo>        #  9 checks，exit 0
# 各加 --selftest：5/5、5/5、3/3、4/4
```

**不需要克隆**：`raw/snapshot/` 存了 10 个源文件逐字节副本，`--repo raw/snapshot` 四个探针读数相同（已实测）。

## 七、边界

- **未安装 DeepSpeed、未在 GPU 上跑、未启用 triton**。E2E 的"CPP 内核"是按 8 参数形态写的桩，不是真的 `pt_binding` 注册（未读 C++ 源码）。
- **未覆盖 DeepSpeed-MII**：那条路径可能用不同的 config 组合到达同一个 `GELUGemmOp`。R8 的限制在 `module_inject/containers`，MII 若绕过容器层，结论可能变；本轮未查。
- `use_triton_ln` 的 `else` 分支在 `mlp_gemm_func`/`qkv_gemm_func` 上同样被 R8 的 BERT 限制间接锁死，但本批**只证明了形参不可达**，没有单独证明这两条分支的业务影响。

## 八、如果要对外，最小动作

三条候选，**均未实施、待用户逐项授权**：

- **(a) 删掉 4 个从未使用的必填形参**，让 triton 签名与 C++ 对齐成 8 个（最小、最安全；`use_triton_ln` 也会跟着消失，它在 `fused_gemm_gelu` 里本来就没用）。
- **(b) 补上 4 个实参**，让调用点对齐 12 个形参（等于宣称 triton 版本要支持 `q_int8`／`async_op`，但函数体不用它们，属假接线）。
- **(c) 只加注释**说明"该路径仅 BERT 可达"。

**倾向 (a)**：它同时消掉 `use_triton_ln` 这条死开关，且不改变任何可达行为。但 DeepSpeed 近期无 Triton 相关活跃开发，采纳概率低，**本条不建议占用用户授权额度**。

## 相关

- [实验入口](../experiments/deepspeed-pyflag-triage/README.md)
- [pyflag 工具](../../tools/pyflag/readme.md)
- [TM-025 队列条目](../../queue_tech-miner.md)
