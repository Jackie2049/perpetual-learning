# verl 的 `use_prefix_grouper` 在 FSDP 引擎上不生效，但生效的是它的副作用

<!-- contribution-summary:start -->
**【仓库】** [verl-project/verl](https://github.com/verl-project/verl)（LLM 后训练 / RL 训练框架，Apache-2.0）

**【贡献点】** `actor.use_prefix_grouper=True` 在 FSDP 引擎上**不会启用 PrefixGrouper 的前向共享**，但 `ray_trainer` 仍按该开关改组级均衡并跳过批内重排。而官方示例 `examples/prefix_grouper/run_qwen3_8b_fsdp.sh` 同时开了与该特性互斥的 `use_dynamic_bsz=True`，README 也写明二者不兼容。**草稿，未发布**——证据见本目录 [verify_prefix_grouper_inert.py](verify_prefix_grouper_inert.py) 与 [findings.json](findings.json)。上游相关：[#7202](https://github.com/verl-project/verl/pull/7202)（关闭未合并）、[#6689](https://github.com/verl-project/verl/pull/6689)（MAGI，仍 open）

**【类型】** 问题诊断（配置项失效与文档/示例不一致）

**【价值评估】** 6 / 10 — 取证严格可复跑（AST 静态判定，无需 GPU 与依赖），影响面明确且用户无感知；但上游维护者已明确表态转向 MAGI attention，接受路径不乐观，价值主要在"消除错误认知"而非"推动修复"。

**【进展】** 已完成对 main `6879e1ab07`（2026-11-02）的静态取证并通过脚本自证；影响面已量化（DP 负载不均衡 0–6.6%，批越大越小）；**未发布任何 issue／评论**，发布须逐项授权。

**【大白话讲明白】**

verl 有一个叫 PrefixGrouper 的加速选项：GRPO 里同一个 prompt 会采样 n 条回答，这 n 条的前缀完全相同，PrefixGrouper 让它们只算一次前缀，理论上省算力。用户按官方示例打开这个开关，训练能正常跑完，不会有任何报错——但前缀其实还是被算了 n 次，加速根本没发生。

更麻烦的是，这个开关**并非完全无效**：训练侧仍会因为它把这些同 prompt 的样本绑在一起、并跳过一段批内重排。后者会改变样本落到哪个 rank、落在批内什么位置，而 PrefixGrouper 本体却没启用。于是用户得到的是"只付了约束的代价、没拿到加速"，而且没有任何提示。

官方示例脚本本身也自相矛盾：README 明确说该特性与 `use_dynamic_bsz=True` 不兼容，而脚本把两个都打开了。

**【What–Why–How】**

- **What：** verl `main` 上 `actor.use_prefix_grouper=True` 的实际生效范围；以及 `examples/prefix_grouper/run_qwen3_8b_fsdp.sh` 与 `examples/prefix_grouper/README.md` 的不一致。
- **Why：** `apply_prefix_grouper_patch()` 只在 `apply_monkey_patch(..., use_prefix_grouper=True)` 时被调用（`verl/models/transformers/monkey_patch.py:327`）；而全仓唯一非测试调用点 `verl/workers/engine/fsdp/transformer_impl.py:328`（`_build_module`）**只传 5 个参数，不含该开关**。`verl/trainer/ppo/prefix_grouper_utils.py` 未被任何模块导入；`prefix_grouper` 也不在 `setup.py`／依赖声明里。因此引擎侧永远走不到那条 `if`。
- **How：** 本目录脚本对上游快照做 AST／文本静态取证，`production_passes_flag=False`、`has_engine_side_propagation=False`、示例 `use_prefix_grouper=True` 与 `use_dynamic_bsz=True` 并存。**边界**：这是静态代码取证，**未运行 verl、未在 GPU 上实测**；结论只覆盖 FSDP 引擎（Megatron 引擎路径未核），只覆盖该 commit；若上游补上引擎侧接线或删除该选项，脚本会返回 1 提示结论失效。
<!-- contribution-summary:end -->

## 一、证据

快照：[verl-main-6879e1ab07.tar.gz](verl-main-6879e1ab07.tar.gz)（GitHub API tarball，commit `6879e1ab07`，2026-11-02T07:58:01Z）。

```bash
tar xzf verl-main-6879e1ab07.tar.gz
python3 verify_prefix_grouper_inert.py verl-project-verl-6879e1ab07/ --json-out
```

退出码 0 表示"发现仍然成立"，1 表示"上游已修好"，2 表示取证失败。

### 1.1 关键判定（`findings.json` 摘录）

| 判定 | 值 |
|---|---|
| 生产调用点是否传 `use_prefix_grouper` | **否** |
| 唯一生产调用点 | `verl/workers/engine/fsdp/transformer_impl.py:328`（`_build_module`） |
| 该调用点实参 | `model`, `use_remove_padding`, `ulysses_sp_size`, `use_fused_kernels`, `fused_kernels_backend` |
| `apply_prefix_grouper_patch` 是否被 `use_prefix_grouper` 形参守卫 | 是（`monkey_patch.py:327`） |
| `prefix_grouper_utils.py` 被谁导入 | **无** |
| `prefix_grouper` 是否在依赖声明中 | **否** |
| 示例是否同时开两个互斥开关 | **是** |
| README 是否写明与 `use_dynamic_bsz` 不兼容 | **是** |

### 1.2 开关的读取点分布

- **config 侧**（真正被用户设置的）：`verl/workers/config/actor.py:172`
- **trainer 侧**（真正生效的副作用）：`verl/trainer/ppo/ray_trainer.py:367/1185/1197/1238`
- **引擎／模型侧**（消费者，但没有生产者）：`verl/models/transformers/monkey_patch.py:300/327`

也就是说：**用户设的值 → 被 trainer 读到并改变批的划分 → 引擎侧没有任何人接这个值**。

### 1.3 trainer 侧到底做了什么

`ray_trainer.py:1197` 起：开关为真时改用 `get_group_balanced_partitions` 做组级均衡（要求 `num_groups % dp_size == 0`，否则抛 `ValueError`）；`ray_trainer.py:1238` 起：开关为真时**跳过** `_balance_batch` 末尾那段 rank 内重排。

这两件事都真实发生——所以这个开关并非"完全无效果"，而是**只剩代价那一半**。

## 二、与既有工作的关系

`notebook/rollout-performance-research.md` 的 §13.40 已经在本仓库内部记录了「`use_prefix_grouper` 是死代码」，并据此把 §13.34／§13.35／§13.39 标注为"当前不可达的假设路径"。本次是对上游最新 commit 的**独立复核**，三点新增：

1. 复核对的是 **main `6879e1ab07`**（2026-11-02），不是本地旧 checkout（`80ee82bd`）。
2. 找到了上游**曾试图修**的证据：[#7202](https://github.com/verl-project/verl/pull/7202)（2026-07-30 开，2026-08-05 关）正是补这条接线，**关闭且未合并**；维护者在关闭理由里说「PrefixGrouper 有限制，我们正在探索 MAGI attention」。
3. 补上了本仓库原先没记的一条：**官方示例与 README 自相矛盾**（示例开 `use_dynamic_bsz=True`，README 说二者不兼容）。

## 三、这个发现值不值得提

先说结论：**作为 issue 提的意义有限，因为维护者已经有明确立场。**

- [#7202](https://github.com/verl-project/verl/pull/7202) 被关闭不是因为它错了，而是因为维护者 `wuxibin89` 决定走 MAGI attention 路线（[#6689](https://github.com/verl-project/verl/pull/6689)，仍 open、35 文件、未合并）。
- 因此"请把接线补上"这个诉求大概率会被再关一次。

**更有价值的两个落点是**（都还没做，需要用户决定）：

1. **文档／示例修订**（成本最低、最可能被接受）：示例脚本同时开两个互斥开关是明确的自相矛盾，改掉不需要架构决策。
2. **让失效可见**（比静默失效好）：要么在引擎侧接线，要么在看到 `use_prefix_grouper=True` 时至少警告"该选项在当前引擎上不生效"。目前是**静默地只生效一半**。

## 四、边界与未核项

- **静态取证，未运行 verl、未上 GPU**；结论只覆盖 FSDP 引擎路径，Megatron 引擎未核。
- 只覆盖 `6879e1ab07` 这一个 commit；脚本会在结论失效时返回 1，但仍需人工复核上游改动。
- 未核 `verl/experimental/` 下两个 trainer 是否也以同样方式读该开关（脚本显示它们也读，但未判定其后的行为）。
- **未发布任何上游内容**（issue／评论／PR 均无）。按用户铁律，任何对外可见动作须逐项授权。

## 五、复现清单

| 项 | 值 |
|---|---|
| 上游仓库 | `verl-project/verl`（`volcengine/verl` 已 301 重定向到此） |
| commit | `6879e1ab07` |
| 提交时间 | 2026-11-02T07:58:01Z |
| 依赖 | 仅 Python 标准库（`ast`、`json`、`pathlib`） |
| 是否需 GPU | **否** |
| 退出码 | 0 = 发现成立；1 = 已失效；2 = 取证失败 |

## 六、影响有多大（2026-11-02 补，同一批后续）

"功能没生效"本身不说明代价。已另做一轮量化：[experiments/verl-prefix-grouper-side-effects](../../../experiments/verl-prefix-grouper-side-effects/readme.md)，**直接把上游 `seqlen_balancing.py` 的函数源码抽出来跑**（不是重写版）：

| 项 | 量级 |
|---|---|
| 前缀共享前向 | **完全没有**（本目录已证） |
| 组级均衡 vs 逐条均衡（DP 负载不均衡） | 批 1024 时 **+1.27%**；批 256 时 **+6.61%**；批 16384 时 **+0.02%** |
| 跳过批内重排 | 按本仓库 §13.39，`max` 模型下**跳过反而是改善**，不构成损失 |

结论：**误开这个开关的净影响 = 功能没拿到 ＋ DP 负载不均衡 0–6.6%**（批越大越接近 0）。这既不严重到"数据损坏"，也不轻到"可忽略"——它落在"用户完全无感知但确实吃亏"的区间，正是最难被发现的一类。

## 七、发布授权状态

截至 2026-11-02，**未向上游发布任何内容**（无 issue、无评论、无 PR）。按用户铁律，任何对外可见动作（评论 / 新建 issue / 建 PR / 推送已关联 PR 的 head）须逐项授权。本目录所有材料仅为本地准备。
