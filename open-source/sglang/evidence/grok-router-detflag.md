# SGLang：Grok 路由器上「确定性推理」的逃生口没有接线

<!-- contribution-summary:start -->
**【仓库】** [SGLang](https://github.com/sgl-project/sglang)（大模型推理框架）

**【贡献点】** `fused_moe_router_shim` 的 `enable_deterministic_inference` 形参没有任何生产者。[源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/router.py)

**【类型】** 漏洞诊断（B 类：主动发掘）

**【价值评估】** 5 / 10 — 缺陷是真的且可证伪，但只影响 Grok1/Grok2 MoE 的 `--enable-deterministic-inference`；该模型不是主流量产配置，用户面窄。

**【进展】** 2026-11-04 在 `27aedf7f` 上静态取证 + CPU 行为分叉验证完成，脚本 exit 0（＝缺陷在）、`--selftest` exit 1（＝判定会翻转）。**未触上游，任何对外动作待用户逐项授权。**

**【大白话讲明白】**

SGLang 有一个开关叫「保证确定性推理」——开了它，同样的输入应当得到逐位相同的结果，RL 训练拿它当对齐基准。Grok 这个模型的路由层（决定每个 token 分给哪几个专家）有两个实现：一个快但不保证顺序，一个慢一点但保证顺序。代码里**留了一个形参想让快的那个在开了确定性时自动让路**，注释也这么写了（"we keep using single kernel to avoid non-deterministic behavior"）。问题是：**从来没有人把这个开关的值传进来**——装函数的地方只固定了第一个参数，唯一发调用的地方只传四个和开关无关的参数。于是这个形参永远取默认的 False，快的那个内核永远被选中，确定性开关在 Grok 的 MoE 路由上等于没开。而且不报错、不打日志。

**【What–Why–How】**

- **What：** `fused_moe_router_shim(..., enable_deterministic_inference: bool = False)` 的该形参在 `27aedf7f` 上无任何生产者。
- **Why：** 用户开 `--enable-deterministic-inference` 时期望逐位可复现；Grok 的 MoE 路由仍走 tensorcore 版内核，与同文件内 aten 兜底分支的写法相反。
- **How：** 在 `srt/layers/moe/topk.py:3239` 的调用点按 `get_exec()` 把值传进去，或改由 shim 自己读执行配置（同文件其它路径的写法）。
<!-- contribution-summary:end -->

## 详细记录

### 1. 现场：形参在那里，值从来没有

`python/sglang/kernels/ops/moe/router.py:338`（`27aedf7f`）：

```python
def fused_moe_router_shim(
    moe_softcapping,
    hidden_states,
    gating_output,
    topk,
    renormalize,
    correction_bias: Optional[torch.Tensor] = None,
    enable_deterministic_inference: bool = False,   # :345
):
    ...
    if (
        (bs >= 512 or num_experts > 8)
        and hidden_dim % BLOCK_SIZE_K == 0
        # we keep using single kernel to avoid non-deterministic behavior
        and not enable_deterministic_inference          # :366
    ):
        return fused_moe_router_tensorcore(...)         # :369
    else:
        return fused_moe_router_cudacore(...)           # :381
```

装函数的地方是 `srt/models/grok.py:149`：

```python
custom_routing_function = functools.partial(
    fused_moe_router_shim, self.router_logit_softcapping   # 只钉住第 1 个位置参数
)
```

`functools.partial` 固定的是 `moe_softcapping`，其余形参全部等关键字。而唯一真正发起调用的地方
`srt/layers/moe/topk.py:3239` 只传四个：

```python
topk_weights, topk_ids = custom_routing_function(
    hidden_states=hidden_states,
    gating_output=router_logits,
    topk=top_k_for_gate,
    renormalize=renormalize,
)
```

⇒ `enable_deterministic_inference` 恒取 `False`，tensorcore 分支恒被选中。

全仓核对（`27aedf7f`，`--include='*.py'`）：

| 搜索 | 结果 |
|---|---|
| `enable_deterministic_inference=` 出现在 `python/sglang/srt/**` | 2 处，**都不是**这个形参：`arg_groups/attention_hook.py:602` 与 `arg_groups/speculative_hook.py:1236` 传的是同名**配置字段/自身形参** |
| `python/sglang/test/**` 里 | 若干处，都是 attention unittest 的本地变量，与 router 无关 |
| `srt/layers/moe/topk.py` 的调用点 | **0 处**传该关键字 |

**它不是「读不到配置」，同仓别的路由路径都是直读的**：

| 位置 | 写法 |
|---|---|
| `srt/layers/moe/mega_gate.py:39` | `mega_gate_supported()` 里 `... or get_exec().deterministic.enable_deterministic_inference` ⇒ 确定性时**关掉**快捷门控 |
| `srt/models/deepseek_v2.py:581` | 确定性时直接走 `torch.mm` / `F.linear`，不走 tiny gemm |
| `srt/models/gpt_oss.py:144` | `self._use_tinygemm = ... and not get_exec()...` |
| `srt/models/kimi_k3.py:922` | 确定性时 `return False`，关掉 fronted-kernel 快捷路径 |
| **`kernels/ops/moe/router.py:345/366`** | **形参 `enable_deterministic_inference`，值是默认 False** |

⇒ 四种模型的写法都是「确定性 ⇒ 让路」，只有 Grok 这条 Triton 快捷路径把判断留给了形参，而形参没有接线。**这是模式上的不一致，不只是单个形参失联。**

### 2. 行为上确实分叉（所以「没传」不是无害）

`experiments/sglang-grok-router-detflag/repro.py` 把 `router.py` 里 `fused_moe_router_shim` 与两个下游函数的**真实源码**取出，在 CPU 上执行（只桩掉 `triton` 及两个内核启动调用，分支与算术全是原文件里的字面代码）：

| 调用方式 | 走到的内核 |
|---|---|
| 照抄 `topk.py:3239` 的四个关键字 | **tensorcore** |
| 将同一组形状再显式传 `enable_deterministic_inference=True` | **cudacore** |

形状取 `bs=512, hidden_dim=1024, num_experts=8`（满足 `bs >= 512`）。⇒ 这个形参不是死代码，它是**唯一的**逃生口，只是没人接。

### 3. 历史归属：形参是 2025-10-09 一次性加的，同提交没动任何调用点

| 提交 | 日期 | 发生了什么 |
|---|---|---|
| `c6d7f8d370` | 2025-03-13 | 引入 `fused_moe_router_shim`（GroK-1 elementwise kernels，PR #4398），**无该形参** |
| `86d10d220f` | 2025-08-23 | shim 增强（`correction_bias`、tensorcore 分支），**仍无该形参** |
| **`9b8ebb2798`** | **2025-10-09** | 该提交的标题是「move more files under srt/utils」(#11285)，diff 里却顺带重组了 `router.py`：**新增形参 `enable_deterministic_inference=False`、新增 `and not enable_deterministic_inference` 判据、把 `bs >= 512 and topk <= 2 and num_experts <= 16` 放宽成 `bs >= 512 or num_experts > 8`**。同一提交**没有改 `grok.py` 也没有改 `topk.py`**（28 个文件里只有 `router.py` 涉及这条链） |
| `ee464fedc6` | 2026-07-14 | MoE 内核迁到 `sglang.kernels`（RFC #29630 Phase 2.5），形参原样搬走、连接线缺口一起搬走 |
| `693d333306` | 2026-10-20 | kernels 按计算重新分组（#45112），仍是原样 |

⇒ **不是重构回归，是加形参的那一次就没接。** 这一点与 verl 侧 TM-006（`tiled_mlp` 在引擎重构里丢接线）形状不同：那边是「曾经接过、后来丢了」，这边是「从来没接过」。

### 4. 影响面与边界（对外必须一起写）

- **只影响 Grok1/Grok2 的 MoE**：`Grok1MoE` 是唯一把 `fused_moe_router_shim` 装进 `TopK` 的地方（`grok.py:149`），且仅在 `num_local_experts > 0` 时构造（`grok.py:508`）。
- **不影响正确性**：两个内核都算 top-k 路由，差别在 matmul 的实现与归约顺序；tensorcore 版正是「快但不保证逐位」的那个。
- **不影响其它模型的确定性**：DeepSeek-V2/V4、Kimi-K3、GPT-OSS、MiniMax-M3 等模型的 router 分支都是 `get_exec()` 直读，正常生效。
- **未验证项（必须明说）**：(a) 没有在真机上跑 Grok + `--enable-deterministic-inference` 观察 logits 是否逐位漂移——本机是 Apple Silicon，triton 只有 Linux 轮子；(b) tensorcore 与 cudacore 在 Grok 的实际数值差异**未测量**，只证明「会走到不同内核」；(c) 不知道维护者是否**有意**保留这个逃生口给将来的调用点。
- **一个可能的反驳**：也许这条快捷路径的设计就是「与确定性无关、一直用快的」。源码注释 `we keep using single kernel to avoid non-deterministic behavior` 明确否定了这一点（它把「不传开关」写成了「为了避免不确定行为」，正好是这个形参想做的事），而其余四种模型的路由快捷路径都直读执行配置让路 ⇒ 作者的意图是「确定性时让路」，只是这条没接上。

### 5. 最小修法（两种，供维护者选）

1. **在调用点传**：`srt/layers/moe/topk.py:3239` 的 `custom_routing_function(...)` 增加一个 `enable_deterministic_inference=get_exec().deterministic.enable_deterministic_inference`。但要小心——`custom_routing_function` 是**通用**钩子，别的模型（`llama4.py`、`afmoe.py`、`interns1pro.py`）也用它，那些函数没有这个形参 ⇒ 只能给 Grok 的 partial 预绑，不能无条件加在公共调用点。
2. **在 shim 内部直读**：把 `and not enable_deterministic_inference` 换成 `and not get_exec().deterministic.enable_deterministic_inference`（与 `mega_gate.py:39`／`gpt_oss.py:144` 等四处写法一致），形参保留作显式覆盖。不需要改公共钩子。

本仓库倾向 **2**（不改公共钩子的签名、与同文件写法统一），但这属维护者判断。

### 6. 核对产物

- [repro.py](../../experiments/sglang-grok-router-detflag/repro.py)：静态 6 条判据 + 1 条对照文件存活判据 + CPU 行为分叉 + 5 个版本的历史归属；`27aedf7f` 上 **7/7 PASS、exit 0**；`--selftest`（模拟上游已接线）**exit 1**。
- 读数落在 [repro.json](../../experiments/sglang-grok-router-detflag/repro.json)。
- 快照来源：`codeload` 拉 `main` tar，与 `raw.githubusercontent.com/27aedf7f…` 的 `router.py`／`topk.py` **逐字节相同**。
- **纯静态 + CPU**，未装 SGLang、未跑推理、未上 GPU、未触上游。

返回 [SGLang 材料索引](../readme.md)
