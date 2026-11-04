# verl 的 `tiled_mlp` 配置在全部 10 个引擎里都没有消费点

<!-- contribution-summary:start -->
> **2026-11-04 更新（本条最重要的新事实）：这不是一开始就写错的配置，而是一次重构回归。** 同目录的结论建立在 `6879e1ab07`（2026-11-02）快照上；补查历史后：PR [#4649](https://github.com/verl-project/verl/pull/4649)（2026-01-04 合入）**当年真的把接线做在 `verl/workers/fsdp_workers.py` 里**（`tiled_mlp_config → use_tiled_mlp / tiled_mlp_shards`，并在 `_build_model_optimizer` 里带 FSDP1 拒绝），该文件于 [#6067](https://github.com/verl-project/verl/pull/6067)（2026-04-20，"deprecate workers, migrate to engines"，BREAKING）整体删除，**接线没有跟着迁到新引擎** `verl/workers/engine/fsdp/transformer_impl.py`（该文件自 2025-09 存在至今从未出现 `tiled_mlp`）。⇒ 2026-01-04 到 2026-04-20 之间这个选项**是有效的**，回归窗口从重构那天起。复核脚本与证据：[experiments/verl-tiled-mlp-regression](../../../../experiments/verl-tiled-mlp-regression/readme.md)。**若将来对外提这条，必须写成"重构回归"而不是"从没接上"**——后者会让维护者以为配置一直无效，也会错过"同期可能还有别的选项一起丢了"这个更有用的线索。


**【仓库】** [verl-project/verl](https://github.com/verl-project/verl)（LLM 后训练 / RL 训练框架，Apache-2.0）

**【贡献点】** `actor_rollout_ref.model.tiled_mlp` 是正式配置项、有官方文档，但**没有任何代码路径读取它**：`model.py:142` 定义、`docs/examples/config.rst:267-273` 说明，而 `verl/workers/engine/` 下 10 个引擎目录 grep `tiled_mlp` 全为 0。真正的实现 `apply_tiled_mlp_monkey_patch` 只在测试里被调用。**草稿，未发布**。证据见 [扫描报告](../../../tools/pyflag/scan-verl-6879e1ab07.json) 与 [pyflag 工具](../../../tools/pyflag/readme.md)

**【类型】** 问题诊断（配置项无消费路径）

**【价值评估】** 7 / 10 — 比同批的 `use_prefix_grouper` 更干净：那个至少有代码路径（只是没人传参），这个连读取点都没有；用户按文档配置后完全无效果且无提示。限制是仍为静态分析、未跑训练验证。

**【进展】** 已对 main `6879e1ab07` 完成静态取证；**未发布任何 issue／评论**，发布须逐项授权。

**【大白话讲明白】**

verl 提供了一个显存优化开关 `tiled_mlp`：把 MLP 拆成几片算，省显存、稍慢一点。它在配置里有正式位置，文档也写了怎么开、`num_shards` 是什么意思。

但实际上打开它没有任何作用：整个 `verl/workers/engine/` 目录下十个引擎里，没有任何一行代码去读这个配置。功能实现（`apply_tiled_mlp_monkey_patch`）是存在的，只是从来没被生产代码调用过——只有测试里调它。

用户按文档打开、发现显存没变化、再看日志也没有任何提示，只能怀疑自己配错了。这类"配置看起来生效、实际完全没连线"的问题，比报错更难排查。

**【What–Why–How】**

- **What：** `actor_rollout_ref.model.tiled_mlp`（含 `enabled`、`num_shards`）在 verl main 上的实际生效情况；以及 `apply_monkey_patch(...)` 的 `use_tiled_mlp` 形参是否有调用方传值。
- **Why：** `verl/workers/config/model.py:142` 定义了该配置，`docs/examples/config.rst:267-273` 给了说明；但全仓唯一的生产调用点 `verl/workers/engine/fsdp/transformer_impl.py:328` **只传 5 个参数，不含 `use_tiled_mlp`**，而 `verl/workers/engine/` 下 `automodel`／`fsdp`／`megatron`／`mindspeed`／`torchtitan`／`veomni` 等 10 个目录 grep `tiled_mlp` 全部为 0。因此 `monkey_patch.py:321` 的 `if use_tiled_mlp:` 永不成立。
- **How：** 用 [pyflag](../../../tools/pyflag/readme.md) 对上游快照做 AST 扫描（消费者／生产者／默认值三步判定），再对每个命中项人工 grep 复核。**边界**：静态分析，**未运行 verl、未上 GPU**；只覆盖 FSDP 引擎路径（其余引擎是"grep 为 0"，未逐个读代码）；不排除通过 `**kwargs` 或配置对象属性的间接传参。
<!-- contribution-summary:end -->

## 一、证据

```bash
python3 tools/pyflag/scan_unwired_flags.py <unpacked-verl-main-6879e1ab07>
```

关键读数（`scan-verl-6879e1ab07.json`）：

| 项 | 值 |
|---|---|
| `tiled_mlp` 消费者 | `apply_monkey_patch`（`monkey_patch.py:321` 有 `if use_tiled_mlp:`） |
| 该形参默认值 | `False` |
| 生产者（传值的调用点） | **0** |
| `mode` | `never` — 默认关且无人传 ⇒ **分支永不执行** |
| `verl/workers/engine/*` 下 `tiled_mlp` 出现次数 | **10/10 目录均为 0** |

## 二、与 `use_prefix_grouper` 的区别

同一批还确认了 [`use_prefix_grouper`](readme.md) 静默失效。两者形态不同：

| | `use_prefix_grouper` | `tiled_mlp` |
|---|---|---|
| 有配置项 | 有 | **有** |
| 有官方文档 | 只有 example 目录 | **有 config.rst 正式条目** |
| 有代码分支 | 有（`monkey_patch.py:327`） | 有（`monkey_patch.py:321`） |
| 有生产者 | 无 | 无 |
| 有无副作用 | **有**（trainer 仍按它改组级均衡） | 无 |
| 实现本身 | `prefix_grouper_utils.py` 无人 import | `apply_tiled_mlp_monkey_patch` 仅在测试中调用 |

`tiled_mlp` 更干净：**没有副作用，纯粹是配置无效**。`use_prefix_grouper` 更麻烦：
它还剩下一半代价（见 [side effects 实验](../../../experiments/verl-prefix-grouper-side-effects/readme.md)）。

## 三、边界与未核项

- **静态分析，未运行 verl**；"10 个引擎目录 grep 为 0"是文本层面的事实，但未逐个读代码确认没有动态读取。
- 未排除 `**kwargs` 或配置对象属性间接传参的可能——本仓库的 `_sentinel` 处理只覆盖显式形参。
- 只覆盖 `6879e1ab07` 这一个 commit。
- **未发布任何上游内容。**

## 四、若要对外，建议的动作

与 `use_prefix_grouper` 不同，这条**不涉及路线分歧**——它不是"MAGI vs PrefixGrouper"那种架构选择，而是一条配置管线断在半路。两个成本很低的选项：

1. **接上**：在 FSDP 引擎的 `apply_monkey_patch` 调用里把 `use_tiled_mlp` / `tiled_mlp_shards` 传进去（约 3 行）。
2. **删掉或告警**：如果暂时不打算接，至少在配置校验阶段拒绝非零的 `tiled_mlp`，而不是静默忽略。

两者都需用户授权后才能向上游提出。
