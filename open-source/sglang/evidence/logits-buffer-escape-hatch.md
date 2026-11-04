# SGLang：`use_logits_buffer` 的默认值在唯一调用点永远拿不到（2026-11-04）

<!-- contribution-summary:start -->
**【仓库】** [SGLang](https://github.com/sgl-project/sglang)（大模型推理框架）

**【贡献点】** [`logits_processor._get_logits(..., use_logits_buffer: bool = True)`](https://github.com/sgl-project/sglang/blob/27aedf7f/python/sglang/srt/layers/logits_processor.py#L1011) 的形参在唯一调用点（[`logprob_processor.py:657`](https://github.com/sgl-project/sglang/blob/27aedf7f/python/sglang/srt/layers/logprob_processor.py#L657)）被传成 `num_chunks == 1`，默认 `True` 从未生效；注释里的形状判据被绕过。

**【类型】** B 类（主动发掘）：源码缺陷候选 + 静态/CPU 行为取证，**未提交上游**

**【价值评估】** 3 / 10 — 缺陷真且可证伪、位置明确，但**不影响正确性**（函数体内还有一层形状比对兜底），只影响「多块 logprob 复用共享 logits buffer」这一条显存路径，用户面窄。

**【进展】** 2026-11-04 发现并取证：静态判据 + `_copy_logits_to_buffer` 真实源码 CPU 双分支执行 + 3 条负向对照全过；`27aedf7f`。**对外（issue/PR）待用户逐项授权。**

**【大白话讲明白】**

SGLang 算完 logits 之后有一块可以重复使用的显存（graph 里预先开好的 buffer）。代码本来想让调用方**看情况决定要不要往里写**——开关默认是「要」。但实际唯一的调用方传的是「这一次是不是只切了一块」，于是只要输入一长（超过 2048 个 token 就切成多块），开关就变成「不要」，从此再没变回「要」。所以那个默认值等于摆设，注释里说的「形状不一样就别用」也从来没被看过一眼。它**不会算出错误结果**，只是让一段本来能走快路径的情况走了慢路径。

**【What–Why–How】**

- **What：** 一个布尔形参在唯一调用点上恒为 `False`，默认值 `True` 不可达，且注释声明的形状判据被调用方绕过。
- **Why：** 「没人传」本身不是缺陷；但这里是**唯一的调用方传了相反的值**，且注释明确写了它是为了处理形状不匹配——说明这个逃生口原本是有意设计给这条路径的。
- **How：** 把形参的每一处读取逐个分类（分支 / 转发 / self 属性）、把函数体真实源码在 CPU 上跑两条分支、用负向对照证明判据会红。详见 [实验](../../../experiments/sglang-pyflag-triage/README.md)。
<!-- contribution-summary:end -->

## 事实

| # | 事实 | 位置（`27aedf7f`） |
|---|---|---|
| 1 | 形参 `use_logits_buffer: bool = True` | `python/sglang/srt/layers/logits_processor.py:1011`（签名） |
| 2 | 函数体内 3 处读取：一处 `if` 分支、两处转发 | 同上 `:1047` / `:1072` / `:1112` |
| 3 | 唯一调用点 `get_logits_fn(...)` 显式传 `use_logits_buffer=num_chunks == 1` | `python/sglang/srt/layers/logprob_processor.py:657` |
| 4 | `num_chunks = (total_size + chunk_size - 1) // chunk_size`（`:601`） | 同上 |
| 5 | `SGLANG_LOGPROB_CHUNK_SIZE` 默认 **2048**、`SGLANG_ENABLE_LOGPROB_CHUNK` 默认 **True** | `python/sglang/srt/environ.py:1600-1601` |
| 6 | 注释原文：`The shared logits buffer is keyed by vocab width and rows; skip it when this batch has a different logits shape than the graph buffer.` | `logits_processor.py:1411-1412` |

**⇒ 输入 logprob 行数 > 2048 时 `num_chunks > 1` ⇒ 形参恒 `False`；默认值 `True` 在这条链上不可达。**

## 为什么说它「本来是给人用的」

注释描述的是一个**按形状决定**的逃生口（形状不匹配就跳过 buffer）。调用点用的是**按块数决定**：
单块时无条件占用 buffer，多块时无条件不用。两个方向的偏差不对称——

| 场景 | 形状是否匹配 buffer | 注释的意思 | 现在的行为 |
|---|---|---|---|
| 单块、形状不匹配 | 否 | 跳过 | 仍传 `True`，靠函数体内的形状比对兜住 |
| **多块、某块形状恰好匹配** | 是 | 可以用 | **一律 `False`，退回 `.float()`** |

⇒ 丢的是第二行：**多块里本来能复用 buffer 的那一块**。所以对外表述只能是
「一个默认 `True` 的形参在唯一调用点拿不到 `True`，注释里的形状判据被绕过」，
**不能写成「导致显存或数值错误」**——函数体内的形状与 dtype 比对（`:1413-1420`）还在。

## 行为上确实分叉

把 `_copy_logits_to_buffer` 的**真实源码**取出、桩掉 `torch` 的四个成员（`empty_like`、
`float`/`float16`/`bfloat16`），在 CPU 上执行原文件的真实分支：

| 传值 | 返回 | 是否多一次转换 |
|---|---|---|
| `True`（形状匹配时） | 复用的 graph buffer | 否 |
| `False` | `logits.float()` 新分配 | 是 |

⇒ 不是「两条等价的路」，形参改变的是**显存去向 + 是否多一次 cast**。查询见
[verify_findings.py](../../../experiments/sglang-pyflag-triage/verify_findings.py) 的
`use_logits_buffer_branch` 字段。

## 边界（对外必须一起写）

- **不影响正确性**：函数体内的形状/dtype 比对仍在，不会把结果写进形状不符的 buffer。
- **未在真机上测**：没有跑 SGLang、没有 GPU。两条分支的**显存峰值差异**与「本该省下的 cast」
  有多大，**一个数都没有**。本仓库不给收益估计。
- **可能是有意的保守选择**：块之间会复用同一块 buffer 的担忧在源码注释里出现过
  （`Each chunk must own its output: writing through the shared graph logits buffer
  would alias chunks whose shape happens to match the buffer.`，`logprob_processor.py:648-650`）。
  ⇒ **调用点为什么写 `num_chunks == 1` 而不是做形状判据，本仓库没有定论**。可能的两种解释
  （缓冲复用风险 / 只是随手写的近似）都无法从静态代码区分，需要维护者确认。
- 本批只证明「默认值不可达 + 行为可分叉」，**不建议直接改代码**：哪一边该动（调用点做形状判据、
  还是把形参删掉改成内部判据）取决于上一条的解释，属维护者判断。

## 复现

```bash
curl -sSL -o /tmp/sglang.tar.gz \
  https://gh-proxy.com/https://github.com/sgl-project/sglang/archive/27aedf7f.tar.gz
mkdir -p /tmp/sglang-src && tar xzf /tmp/sglang.tar.gz -C /tmp/sglang-src --strip-components=1
python3 experiments/sglang-pyflag-triage/verify_findings.py /tmp/sglang-src --json /tmp/out.json
python3 experiments/sglang-pyflag-triage/verify_findings.py /tmp/sglang-src --selftest
```

## 相关

- [实验与 34 条分类账](../../../experiments/sglang-pyflag-triage/README.md)
- [TM-013：同批第一条（Grok 路由确定性形参）](grok-router-detflag.md)
- [TM-013 队列条目](../../../queue_tech-miner.md)
- [pyflag 方法](../../../tools/pyflag/readme.md)
