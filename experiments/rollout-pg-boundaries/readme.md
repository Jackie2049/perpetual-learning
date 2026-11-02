# PrefixGrouper 下的微批边界成本：组在哪一层被保住、在哪一层被切开

2026-11-02，M1。支撑 [rollout 调研主文档 §13.34](../../notebook/rollout-performance-research.md)，
接 §13.32 登记的残留「PP 微批边界的影响未量化（`use_prefix_grouper` 下刻意跳过「小微批排两端」）」。

**不是 GPU 实测，不是端到端**。进程内按 verl 快照（`1a9e711e`）逐行复刻切分形状，
前缀块开销按 FLOP 口径计算；墙钟未测，不可与任何论文报值相除（§19.1 规则一）。

## 一句话结论

**`use_prefix_grouper` 的组完整性在 mini-batch 这一层被完整兑现（割点 0% 落在组内），
但 micro-batch 层不看组——它按长度排名重切，默认 16384 预算下每组平均被切成 3.62 段，
前缀块开销是「整组共享」的 3.62 倍；而默认逐条均衡在 mini-batch 层就已经切开 99.9% 的组。**

| 层 | 组级均衡（`use_prefix_grouper=true`） | 默认逐条均衡 |
|---|---|---|
| DP dispatch（连续 chunk） | 与组级 LDM 分区**逐一相等**（120/120） | — |
| mini-batch（连续切） | 割点 **0.0%** 落在组内 | **99.9%** |
| micro-batch（按长度排名 LDM） | 默认预算下每组 **3.62 段** | 1.43 段（但组早已不完整，无共享可谈） |
| 前缀块开销 / 整组共享 | **3.62×**（默认 16384）／1.00×（预算装得下整批） | — |

## 为什么做

§13.32 把「训练侧拆组」拆成两类代价算完了，但留了一条没量化：**micro-batch 边界**。
`_balance_batch` 在 `use_prefix_grouper` 下跳过分区内重排（`ray_trainer.py:1201`），
注释写明目的是「maintain uid grouping」，同一文件另一处注释写着「Place smaller micro-batches
at both ends to reduce the bubbles in pipeline parallel」。这两句合起来才说明代价在哪：
**PP 气泡优化被关掉了，剩下来与组相关的边界只有 micro-batch 层。**

## 模型（四层，逐层照抄源码）

| 层 | 量什么 | 对应的真实代码 |
|---|---|---|
| **L1 前乘** | 组级 LDM 分区 vs dispatch 的连续 chunk 是否同一件事 | `ray_trainer.py:1160-1212`、`dispatch_nd_compute_dataproto`→`DataProto.chunk(dp_size)` |
| **L1 割点** | 连续切下割点穿过组内部的比例（mini-batch 层 / micro-batch 层） | `make_iterator`（`tensordict_utils.py:601-611`）、`rearrange_micro_batches`（`seqlen_balancing.py:394-466`） |
| **L2 迭代** | epoch 内是否重复／丢尾部；`total_num_iterations` 与实际批数是否一致 | `engine_workers.py:278-302`、`tensordict_utils.py:588` |
| **L3 前缀开销** | 前缀块的 FLOP：整组一次 vs 每片段一次 vs 逐条一次 | `prefix_grouper_utils.py:59-72`（按 uid 相邻重建 group_sizes） |
| **L4 训练侧** | 组是否留在同一 mini-batch | 同上，切法用连续切与随机切对照 |

**LDM 与 micro-batch 分区都与真实 verl 逐位核对过**（见「复现」）：`karmarkar_karp`
120/120 组一致，`rearrange_micro_batches` 的 40/40 个分区完全一致。这一步不是形式检查——
第一版模型就是因为没核这里，把 micro-batch 当成连续切，得出了完全相反的读数（见「模型修正」）。

## 结果

### L1 割点：mini-batch 层干净，micro-batch 层不干净

| 均衡方式 | mini-batch 层切开率 | micro-batch 层切开率（默认 16384） | micro-batch（预算装得下整批） |
|---|---|---|---|
| 组级（`use_prefix_grouper`） | **0.0%** | **81.3%** | 0.0% |
| 默认逐条 | **99.9%** | 100% | 99.9% |

**读数一：组级均衡在 mini-batch 层是有效的，而且是结构性的。** 组级 LDM 把组聚成一个整体，
重排后组在 rank 内相邻；连续切若 chunk 大小是 `mini_groups × n`，割点必然落在组之间。
这解释了 §13.32 的「真实 LDM 组成接近最坏」为什么没有变成质量问题——
那一层本来就保住了组。

**读数二：micro-batch 层没有任何组感知。** `rearrange_micro_batches` 按
`calculate_workload` 对**单条样本**做 equal_size=False 的 LDM，`force_group_size`
在本快照里没有接到 agent 配置（`engine/utils.py:71` 读的是非张量键，默认 1），
所以一个组的 8 条样本会被分散到多个 micro-batch。**默认预算下，81.3% 的「组-出现」是被切开的。**

### L1 前乘：dispatch 的连续 chunk 恰好等于分区

120/120 个 (n_per, dp, mini_groups) 组合里，每个 rank 拿到的连续 chunk **都恰好是一个组级 LDM 分区**。
这不是 dispatch 做了「逐分区取」，而是「分区大小 = B/dp」这个算术巧合：
组级 LDM 的 equal_size、组在重排后相邻、dispatch 按连续 chunk 切，三件事叠在一起才对上。
**一旦 batch 布局不是「组连续」**（比如上游把 uid 打乱再送进来），这个等式不成立，
「同一 uid 不跨 rank」的保证就会失效——这只是一个静态条件，不是稳健设计。

### L3 前缀开销：默认预算下是整组共享的 3.62 倍

| 预算 | 每组片段数 | 前缀块 / 整组共享 | 前缀块 / 逐条重复 |
|---|---|---|---|
| 装得下整批（loose） | 1.00 | **1.00×** | 0.125× |
| 16384（`ppo_max_token_len_per_gpu` 默认） | 3.62 | **3.62×** | 0.536× |
| 4096 | 7.20 | 7.20× | 0.900× |

**读数三：PrefixGrouper 在默认配置下拿到的共享是「理论值」的 28%。**
3.62 倍的「组完整时」开销意味着：如果 micro-batch 层也按组切，前缀块还能再省 72%。
但注意右列——**即使现在的 3.62 倍，仍只有逐条重复的 0.536 倍**，也就是说
PrefixGrouper 的收益仍然为正，只是被吃掉了一半多。

**读数四：前缀越长，这个结论越稳。** 三档前缀（128/512/2048）的比值逐位相同，
因为在本口径下碎片数与长度无关、只与预算和组大小有关。**桶外事实**：
前缀共享的总收益是 `1/n`（n=8 时 0.125×），这一点由 `prefix_rounds` 的定义直接给出，
不随长度变化——**长前缀的额外收益来自「本来就更贵」，不是「共享更有效」**。

### L2 迭代路径：这一层没有额外代价

整除是硬断言（`tensordict_utils.py:588`），所以可达组合里 `total_num_iterations`
与实际批数恒等（0 例不一致）。`shuffle: false` 下 epoch 内同一 mini-batch 重复出现是确定性的，
不是边界问题，不计入代价。

### L4 训练侧：连续切下组 100% 留在同一 mini-batch

| 切法 | 组留在同一 mini-batch 的比例 |
|---|---|
| 连续切（现状） | **100.0%** |
| 随机切（若改成 shuffle=True） | 0.0% |

**这不是「shuffle 有害」的结论**：§13.32 已证 mini-batch 组成差只有个位数百分比，
随机切也不会改变模型更新；它改变的是 **PrefixGrouper 还能不能共享**。

## 识别性检验与负向对照

| 检验 | 期望 | 实测 |
|---|---|---|
| 每组只有 1 条（n=1） | 切开率 0、开销比 1.0 | 0.0 / 1.0 ✅ |
| 预算装得下整批 | micro 层切开率 0 | 0.0 ✅ |
| 预算缩小 | 每组片段数上升 | 4096 > 16384 ✅ |
| 打乱 mini-batch 内顺序 | 切开率逐位不变 | 不变（顺序无关性）✅ |
| 整组共享开销 / 逐条重复 | 恒等于 1/n | 0.125（n=8）✅ |
| 默认逐条均衡 | 在 mini-batch 层切开 | 99.9% ✅ |

最后两条尤其重要：「1/n 恒等」说明本口径里**前缀长度不改变共享的相对收益**，
所以任何「前缀越长 PrefixGrouper 越划算」的说法都不能从本实验推出——那需要一个
带固定开销的模型，本实验没有建。

## 模型修正（本批自查记录）

| # | 第一版写法 | 错在哪 | 现在 |
|---|---|---|---|
| 1 | 把 `_balance_batch` 的分区直接当 rank 视图 | 真实 dispatch 是连续 chunk；分区只决定重排顺序 | 改成 `dispatch_views()`，并核出两者在现有整除条件下恰好相等 |
| 2 | micro-batch 按连续切 | `rearrange_micro_batches` 按长度排名 LDM，`use_dynamic_bsz_balance` 默认 True | 复刻完整形状（含 `[::2][::-1] + [1::2]`），与真实实现 40/40 一致 |
| 3 | 「分组共享 = 每组 n 重响应」但长度用样本总长 | 组大小与长度混用量纲，导致 `real` 为负 | 全部改为「一次遍历里前缀算几遍」 |
| 4 | 用浮点长度做交叉验证 | 真实实现用 `attention_mask.sum()`，`torch.arange` 会把浮点截断到 0 长度 | 长度一律取整数 |
| 5 | 「前缀越长共享收益越大」当作对照 | 本口径里该比值恒为 1/n，这条对照必然失败 | 换成「=1/n 恒等」与「预算缩小则片段数上升」两条 |

第 3 条是最危险的一类错误：它让读数出现负值（< 0 的 FLOP），如果没检查就直接报数，
结论会反向。现在所有比值都有「≥ 1 或恒等」的断言守着。

## 边界（引用时必须一并写）

- **进程内仿真，非 GPU 实测、非端到端**。没有跑任何训练；前缀块按 FLOP 计，不折算墙钟。
- **`no_share` 列不是真实默认路径的物理量**。默认 `use_remove_padding=true` 时，
  n 条同 prompt 的响应在 packed attention 里本来就只算一次前缀；`no_share` 只在
  `use_remove_padding=False`（`examples/prefix_grouper` 的配置）下才近似成立。
  因此「real / no_share」这一列只能当**相对刻度**，不能当「省了多少」报。
- **「切开的片段之间完全不能共享」是上界口径**。上游 `prefix_grouper` 的 `forward`
  按 `group_info` 的 `prefix_len/suffix_lens` 实现，碎片理论上可以各自共享；但本仓库快照
  `build_pg_from_micro_batch` 按 uid 相邻重建组，切开的两段各成不完整组。上游方法未在本机验证，
  故同时给 `best_per_mb`（若同 mini-batch 内碎片可共享，则退回 1.0×）。
- **长度分布是构造参数**（均值 2200、CV 0.45、整数化），不是从任何训练日志取的。
- **不计 PP 气泡**：`_balance_batch` 那句注释提到的「小微批排两端」被跳过，其真实收益未测，
  只说明它被关掉了。
- **不支持**「应该打开/关闭 `use_prefix_grouper`」这类结论；只支持「组完整性在哪一层被保住、
  micro-batch 层拿走了多少共享」。

## 复现

```bash
python3 pg_boundaries.py             # 约 12 s，覆盖 summary.json
python3 pg_boundaries.py --verify    # 重跑并与归档逐字段比对
python3 -m unittest test_pg_boundaries -v    # 22 个测试
```

与真实 verl 的交叉验证（CPU torch，不进本仓库依赖）：

```bash
uv venv /tmp/verl-seqlb-venv --python 3.12
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install numpy packaging ray tensordict omegaconf transformers codetiming
/tmp/verl-seqlb-venv/bin/python xcheck_verl.py
# ldm: 120/120 完全一致（不一致 0）
# micro_split: 40/40 分区完全一致（不一致 0）
# group_balanced_partitions: 分区数 4，组完整性 True
```

执行条件与产物 sha256 见 [run-info.json](run-info.json)。

## 源码锚点（verl 1a9e711e）

- `verl/trainer/ppo/ray_trainer.py:1160-1212`：`_balance_batch` 的 prefix_grouper 分支（组级 LDM ＋ 跳过重排）
- `verl/trainer/ppo/ray_trainer.py:1209-1212`：默认逐条 LDM（轻的排两端）
- `verl/utils/seqlen_balancing.py:46`：`calculate_workload = 24576*l + l**2`
- `verl/utils/seqlen_balancing.py:394-466`：`rearrange_micro_batches`（含 `use_dynamic_bsz_balance`）
- `verl/utils/seqlen_balancing.py:554-624`：`get_group_balanced_partitions`
- `verl/workers/engine/utils.py:63,71`：`use_dynamic_bsz_balance=True`、`force_group_size` 默认 1
- `verl/workers/engine_workers.py:244-302`：`train_mini_batch`（`mini_batch_size_per_gpu`、`total_num_iterations`）
- `verl/utils/tensordict_utils.py:588-612`：`make_iterator`（整除断言、连续切）
- `verl/single_controller/base/decorator.py:250-252`：dispatch 用连续 chunk
- `verl/trainer/ppo/prefix_grouper_utils.py:59-72`：按 uid 相邻重建 `group_sizes`
- `verl/trainer/config/actor/actor.yaml:102,118`：`use_prefix_grouper: false`、`shuffle: false`
