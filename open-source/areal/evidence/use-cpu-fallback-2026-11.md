# AReaL 的 CPU 回退开关：没有人打开它，打开了也进不来

<!-- contribution-summary:start -->
**【仓库】** [AReaL](https://github.com/areal-project/AReaL)（强化学习后训练框架，`archon` 引擎的 MoE 实现）

**【贡献点】** `areal/experimental/models/archon/moe/kernels.py` 的 `generate_permute_indices(..., use_cpu=False)` 形参在快照 `3b936798` 上**零生产者**：唯一调用点 `moe/utils.py:212` 传 5 个位置实参；而 `kernels.py` 顶层 `import triton`，没有 Triton 的机器连模块都进不来。核对结论是**开关不可达、实现本身可用**，不是"坏掉的回退"。[实验与判据](../../experiments/areal-use-cpu-fallback/README.md)

**【类型】** 问题诊断（B 类：主动发掘，负结论）

**【价值评估】** 3 / 10 — 没有用户受影响（无配置字段、无 CLI、默认关闭、开发期调试开关），**不对外**。留下的是方法结论：判"回退路径坏了"之前，先把调用方实际给的缓冲口径对齐。

**【进展】** 2026-11-05 完成。`probe.py` 6 类 15 条判据全 PASS、`check_snapshot.py` 4 条 + 注入自检 3/3；抄本与上游快照逐行双向比对；纯 CPU、未装 areal、未触上游。

**【大白话讲明白】**

MoE 模型要把 token 按专家重新排队（"permute"），AReaL 用一个 Triton 写的 GPU 小内核干这件事。代码里还留了一份纯 Python 的等价实现，入口是函数参数 `use_cpu`——"在 CPU 上做"。

- **打不开**：整个仓库里没有任何地方给它传 `True`。调用它的那一行只传了 5 个位置参数，`use_cpu` 永远走默认的 `False`。
- **就算打开了也进不去**：这个文件第一行就 `import triton`。没有 Triton 的机器（macOS、不带 GPU 的 Linux）连 import 都失败，而它的单元测试用 `skipif(not HAS_TRITON)` 把自己静默跳过，所以没人发现。
- **那份 CPU 实现本身没坏**：把它的代码原样搬出来、喂上游自己算的参数，结果和 GPU 版本逐位相同。唯一"看着要坏"的地方是两个数没对齐（起点按填充后的位置算、累加按真实长度算），但上游给的缓冲区比这更大，所以正常不会撞上。

⇒ 说人话：**这个开关是开发者留的调试后门，一直关着，而且现在也开不了；关着没有坏后果。**

**【What–Why–How】**

- **What：** `use_cpu` 形参在全仓零生产者；`kernels.py` 顶层依赖 Triton，非 CUDA 机器无法导入；`fill_indices_cpu` 的起点（对齐后）与步长（未对齐）口径不一致。
- **Why：** 上一批 pyflag 只判到"没有生产者"，本批要回答后半句——那段回退是不是好的。这决定了它该被删掉还是该被接线。
- **How：** 把 `fill_indices_cpu` 与 `generate_permute_indices` 的偏移计算逐行抄出来（抄本与快照的一致性由 `check_snapshot.py` 双向断言），在 CPU 上跑：上游缓冲口径下与 GPU 排布 0 点差异（3 档 + 2550 组随机输入）；缓冲压到 token 总数时抛 `RuntimeError`。**边界**：抄本行为 ≠ 真实模块行为（真实模块在本机 import 不进来），不证明"CPU 实现有错"，恰恰相反。
<!-- contribution-summary:end -->

## 一、三段事实

| 段 | 位置（`3b936798`） | 事实 |
|---|---|---|
| 生产者 | `areal/experimental/models/archon/moe/utils.py:212` | `generate_permute_indices(...)` 传 5 个位置实参，无 `use_cpu`；全仓（含 `tests/`）零生产者 |
| 依赖 | `areal/experimental/models/archon/moe/kernels.py:5-7` | 顶层 `import triton`；无 Triton 的机器 `ModuleNotFoundError` |
| 测试 | `tests/experimental/archon/test_moe_common.py` | `skipif(not HAS_TRITON)` ⇒ 缺 Triton 时静默略过 |

## 二、两个偏移口径（不是缺陷，是脆弱点）

```python
write_start = write_offsets[e].item()     # 起点：m_offsets - m_sizes（对齐后）
...
write_start += length                     # 步长：未对齐的真实长度
```

后果：对齐填充段整段不被写（读数里 56 格缓冲留 36 个 `-1`）。但 `_permute` 给的缓冲是
`round_up(总 token + E × 对齐, 对齐)`（`utils.py:204-205`），足够大，所以最终布局与 GPU 版
逐位相同。**只有把缓冲压到 token 总数时才会抛异常。**

## 三、为什么不该对外

- 没有配置字段、没有 CLI、默认关闭 ⇒ 没有用户会受影响。
- 它是个开发期调试开关；"删掉形参和分支"或"给它接线"两类动作价值都 <3/10。
- 真正值得写下来的是方法：本次三次误判里有两次都是"没先对齐调用方实际给的缓冲口径"。

## 核对产物

| 文件 | 结果 |
|---|---|
| [`probe.py`](../../experiments/areal-use-cpu-fallback/probe.py) | 6 类 15 条判据，全部 PASS（含两个改坏对照） |
| [`check_snapshot.py`](../../experiments/areal-use-cpu-fallback/check_snapshot.py) | 4 条 PASS；`--selftest` 3/3 |
| [`raw/upstream/`](../../experiments/areal-use-cpu-fallback/raw/upstream/) | `kernels.py` / `utils.py` 逐字节快照 |

复现见 [实验 README](../../experiments/areal-use-cpu-fallback/README.md)。
