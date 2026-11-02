# PP 微批边界成本：`use_prefix_grouper` 下跳过"小块排两端"的代价

**对应**：[rollout 性能调研主文档 §13.33](../../notebook/rollout-performance-research.md)，接续 [§13.32](../rollout-minibatch-composition/readme.md) 登记的残留。
**日期**：2026-11-02（M1）。**状态**：已完成；25 个单测通过、`--verify` 连跑两次通过。

## 一句话结论

**这段 PP 重排在本文的代价模型下是"负优化"，而 prefix grouper 跳过它并不是丢掉了收益**——去掉重排后每个 rank 的 PP 单元代价反而更低（model=max 下 `cur` 是 `sorted` 的 1.00–1.65 倍），差异随长度离散度与每 rank 微批数增长，但在每 rank 只有 1 个微批时**逐位为零**。量级高度依赖代价模型（`sum_plus_max` 下只剩 1.00–1.08 倍，`sum` 下恒为 1.0000），所以只能说"方向"，不能说"倍数"。

## 问题从哪来

verl `_balance_batch` 末尾有一段：

```python
# Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
# Skip reordering within partitions for PrefixGrouper to maintain uid grouping
if not getattr(self, "use_prefix_grouper", False):
    for idx, partition in enumerate(global_partition_lst):
        partition.sort(key=lambda x: (workload_lst[x], x))
        ordered_partition = partition[::2] + partition[1::2][::-1]
```

§13.32 把"打开 prefix grouper 就不排了"登记为未量化成本。本批把它量出来。

## 先核语义：为什么两种排法可以直接比

四个事实串起来，结论才是可比的：

1. `_balance_batch` 产出 `dp_size` 个 partition，每个 partition 是一组**下标**。
2. `global_idx = [j for partition in lst for j in partition]`——partition 的顺序就是 rank 内顺序。
3. dispatch 用 `chunk(dp_size)` **连续等分** `global_idx`（`decorator.py:71-88`），第 r 个 rank 拿到的正好是 `lst[r]`。
4. rank 内再被 `train_mini_batch` 按 `mini_batch_size_per_gpu` 连续切成 PP micro-batch（`engine_workers.py:295-309`）。

所以"跳过重排"改变的是**rank 内顺序**，成员集合不变。这是所有读数的比较前提，也由单测 `test_arms_preserve_membership` 守住。

## 两种被我推翻的结论（先说，因为它们决定读数怎么读）

**错误 1：口径不可比。** 初版把"把 partition 当不可分块做 `[::2]+[1::2][::-1]`"当成恢复旧语义，得到"新版是旧版的 0.27–0.39 倍"这种三倍级差异。逐个 rank 打印元素个数后才发现，旧写法下**每个 rank 只有 1 个元素**（rank 内部成了一条权重），与新版每 rank 32 个元素不是同一个数量。三倍级差异本身就是该停下来核对的警号。防回流测试：`test_block_level_ordering_is_not_comparable`。

**错误 2：方向猜反。** 本批起初假设"小块排两端"是为了压小每片最大载荷，于是预期去掉它会变差。实测方向相反。**原因见下面的机制段：这段重排优化的不是"每片最大载荷"。**

## 第一部分：三种载荷模型

| 模型 | 槽位时间 | `cur`/`sorted` 区间（36 格） |
|---|---|---|
| `max` | 该片最大单条载荷（"每片都等最重那条"的上界） | **1.0000–1.6462**（均值 1.2064） |
| `sum_plus_max` | 载荷之和 + 最大单条（中性组合） | 1.0000–1.0799（均值 1.0197） |
| `sum` | 载荷之和 | **恒 1.0000** |

`sum` 恒为 1 是结构性结果而非巧合：载荷可加时，总代价只取决于成员集合，与顺序无关（单测 `test_sum_model_is_membership_invariant`）。**这一行是整个实验的标尺**——它说明"排法只在 PP 边界存在时才影响代价"，也说明另外两行的差异全部来自边界效应。

## 第二部分：主读数

五种排法，成员集合完全相同：

| 排法 | 定义 |
|---|---|
| `cur` | 当前实装：partition 内**先按载荷升序**再 `[::2]+[1::2][::-1]` |
| `sorted` | 对照基准：升序不交错（"不重排"的最近替身） |
| `desc` | 负向对照：降序 |
| `plain` | 旧语义：不排序、在原顺序上直接交错 |
| `orig` | 完全不排（保留 LDM 分区的原始顺序） |

`model=max`、cv=0.6 下的读数：

| dp | m | `cur` | `sorted` | `plain` | `orig` |
|---|---|---|---|---|---|
| 2 | 2 | 1.4739 | 1.0000 | 1.4358 | 1.4148 |
| 2 | 4 | 1.3205 | 1.0000 | 1.7915 | 1.7162 |
| 8 | 4 | 1.2447 | 1.0000 | 1.5554 | 1.5143 |
| 8 | 8 | 1.1238 | 1.0000 | 1.5317 | 1.4877 |

三条可引用的读数：

- **`cur` 全部 36 格都不优于 `sorted`**（`cur` ≥ 1.0000）；m=2 时最差（cv=1.2、dp=2 达 **1.6462**），m=8 时最好（1.0419）。
- **`plain` 比 `cur` 更差**：同一单元格 1.53 vs 1.12（dp=8、m=8）。所以当前的"先排序再交错"**已经是一处改善**，只是整件事仍不如不排。
- **`desc` 与 `sorted` 逐位相同**（1.0000）：`max` 模型只看每片最大值，而升序与降序切出的每片最大值多重集相同（单测 `test_sorted_and_desc_are_identical_under_max`）。这条也是"模型只看 max"的自证。

## 第三部分：机制

cv=0.6、dp=8、m=4 下的两个可观测量：

| 排法 | `stage_max_spread`（各片最大载荷/全 rank 均值单条载荷） | `piece_max_cv`（各片最大载荷的离散度） | `cost`/`sorted` |
|---|---|---|---|
| `sorted` ／ `desc` | **1.4028** | **0.6682** | 1.0000 |
| `cur` | 1.7471 | 0.5303 | 1.2440 |
| `orig` | 2.1245 | 0.2679 | 1.5136 |
| `plain` | **2.1833** | 0.2406 | 1.5541 |

两者一起给出答案：

- **`stage_max_spread` 与代价同向**：sorted 最低 → cur 居中 → plain/orig 最高，与 `cost` 的排序完全一致。它量的是"某级 stage 集中装重活"的程度，这正是 PP 槽位要等的东西。
- **`piece_max_cv` 与代价反向**：sorted 最高、plain 最低。也就是说这段重排**把各片最大载荷拉得更不齐**了——它优化的不是"压小每片峰值"。

于是机制是：交错会在大样本之间夹小样本，压住每片峰值，但当大样本被挤进同一个 stage 时反而制造新的长尾；`sorted`（单调升序）恰好让每片的峰值随 stage 单调递增、分布均匀。**说"这段代码优化 PP 气泡"在 `max` 模型下不成立；它更像优化"进入流水线的顺序均匀度"，而本实验的模型不含该效应。**

## 第四部分：识别性检验（先验后量）

| 指标 | 相对极差 | 敏感 | 说明 |
|---|---|---|---|
| `cur`/`sorted` 对 cv 0.2→1.2 | 27.7% | 是 | 离散越大，排序越有价值 |
| `cur`/`sorted` 对 m 1→2→4→8 | 36.8% | 是 | 微批越多，边界效应越强 |
| 对照：cv 驱动的组均值离散度 | 160.8% | 是 | 检验不过严（真信号能读到） |
| 对照：恒常数指标 | 0.0 | **否** | 检验不过松（无效指标被打死） |

外加一条硬判据：**m=1 时五种排法逐位相同**（`single_micro_batch_identity.all_identical = true`）。这把"排法只在 PP 微批边界存在时才有意义"从叙述变成了可执行断言。

## 边界（不要读出这些）

- **不是实测**：进程内构造长度分布，没有 GPU、没有 PP、没有 verl 跑起来。虚拟单位，**不可与任何论文报值相除**。
- **量级不可单挑**：`max` 给 1.65 倍、`sum_plus_max` 给 1.08 倍、`sum` 恒 1.00。**只能报"方向 + 模型区间"**。任何引用本节的正文都必须写出所用模型。
- **模型不含真实 PP 的东西**：没有 stage 级并行、没有 overlap、没有通信、没有激活显存。`max` 模型是"每片都等最重那条"的上界。
- **长度分布是构造参数**（cv 扫描），不是从任何训练日志取的。LDM 分区与真实 verl 逐位一致只保证"给定输入下算得对"。
- **不支持**"应该给 prefix grouper 加回这段重排"或"某框架该改 `_balance_batch`"——本实验只说明这段代码在本模型下的方向，真实 PP 有没有 overlap 会改变结论。
- **不支持**"prefix grouper 的收益被 PP 边界成本抵消"。这里量的是训练侧 DP 分区内的微批边界，与 rollout 侧组粒度占槽位（§13.30 L2）是不同阶段的账，**不可相加**。

## 复现

```bash
python3 pp_microbatch_boundary.py               # 约 20 s，覆盖 summary.json
python3 pp_microbatch_boundary.py --verify      # 重跑并与归档逐字段比对（连跑两次通过）
python3 -m unittest test_pp_microbatch_boundary -v   # 25 个测试
```

执行环境：Python 3.9.6（macOS arm64），仅标准库。LDM 划分复用
[`../rollout-group-split-tradeoff/`](../rollout-group-split-tradeoff/readme.md) 的 `_karmarkar_karp`
（已与真实 verl 在 180 组输入上交叉验证分区一致），不在本目录复制第二份。
产物 sha256 见 [run-info.json](run-info.json)。

## 源码锚点

| 文件:行 | 作用 |
|---|---|
| `verl/trainer/ppo/ray_trainer.py:1236-1242` | PP 气泡重排，被 `use_prefix_grouper` 跳过（本实验的问题起点） |
| `verl/trainer/ppo/ray_trainer.py:1244` | `global_idx` 拼接 ⇒ partition 顺序 = rank 内顺序 |
| `verl/single_controller/base/decorator.py:71-88` | dispatch 用 `chunk(dp_size)` 连续等分 |
| `verl/workers/engine_workers.py:295-309` | rank 内 `mini_batch_size_per_gpu = ppo_mini_batch_size / dp` |
| `verl/utils/seqlen_balancing.py:213` | `get_seqlen_balanced_partitions`（LDM 划分入口） |

`source-snapshots/` 存了这五个文件的离线副本（verl `80ee82bd`），`--verify` 会重新核验片段仍命中；上游改了会明确报 anchor 失效，而不是静默沿用旧结论。
