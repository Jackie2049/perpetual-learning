# 训练侧「拆组」到底改变了什么——三个候选代价的识别性检验

**对应**：[rollout 性能调研主文档 §13.31](../../notebook/rollout-performance-research.md)，并**订正 §13.30 第二批结论五**。
**日期**：2026-11-02（M1）。**状态**：已完成；14 个单测通过、`--verify` 连跑两次通过。

## 一句话结论

**拆组在训练侧的统计代价是零。**verl 的默认逐条 LDM 路径在 dp=4/8 下 100% 会拆组，但拆开的只是"样本分到哪个 DP rank / 哪个 mini-batch"，不是"归一化用哪一组的 mean/std"——GRPO 的归一化按 `uid` 分组，在 `_balance_batch` 之后执行，与样本在张量里的位置无关。三个候选代价（跨组幅度错配、组级梯度方向差、梯度方差）在真实路径上**全部为零**，并且**全部通不过识别性检验**（对组难度方差零敏感）。

## 为什么本文件存在

§13.30 第二批从源码读出"默认逐条 LDM 在 99.65% 的 (种子, dp) 组合里至少把一组拆到两个分区"，然后把这个事实读成**"拆组代价在默认训练里已经在付"**。这个读法是错的，本文件把它推翻。

核对到的源码顺序（固定 commit，见下方锚点）：

1. `ray_trainer.py:1508` `batch = batch.repeat(repeat_times=rollout.n, interleave=True)` —— 同组样本连续排列；
2. `ray_trainer.py:1511-1513` `_balance_batch(...)` —— **重排顺序**；
3. `ray_trainer.py:1629-1640` `compute_advantage(...)` —— 之后才算优势，GRPO 分支按 `index=uid` 分组（`core_algos.py:308-321`）；
4. `ray_trainer.py:1510-1512` 的注释自己就写明：重排 *"won't affect the advantage calculation (since it's based on uid)"*。

所以"拆组"改变的是 **样本落在哪个 DP rank / 哪个 mini-batch**，**不改变归一化口径**。§13.30 结论一（尺度错配 0.69／0.41 倍）成立的前提是"允许按分片重新归一化"——那是**假想方案**，不是 verl 的默认行为。

## 三个候选的读数

| 候选 | 度量 | 真实路径读数 | 判定 |
|---|---|---|---|
| **A 跨组幅度错配** | 按 uid 归一化后，重排前后的 \|adv\| 比 | **1.000000**（30 种子最大偏离 0.0） | 不发生 |
| A 的**对照口径** | 按分片各自重新归一化 | 1.0462（p=2） | 这是 §13.30 的假设方案，非默认行为 |
| **B 组级梯度方向差** | 组内梯度向量和的范数／各条范数之和 | 整组 0.43094／拆组 0.43094，**delta = −1.9e-18** | 不发生 |
| **C 梯度方差** | 一批梯度方差（拆组／整组） | **1.000000**（最大偏离 3.3e-16） | 不发生 |
| **顺序不变性** | 按分区重排后重算优势，映射回原序比对 | 30 种子全部不变（最大差 2.1e-15，纯浮点求和顺序） | 成立 |

B 与 C 为零的结构性原因：分区先在 rank 内归约、再跨 rank 求和，等价于整组求和；DP 组的 loss 归约是 `all_reduce(AVG)`（`engine_workers.py:189`）。分区只改变浮点求和顺序，不改期望与方差。

## 识别性检验：三个候选全部不及格

把组难度方差从 0 拉到 25 五档，看指标动不动：

| 指标 | 全档位极差 | 有识别性 |
|---|---|---|
| A（`by_partition` 口径） | **0** | 否 |
| B（delta） | 2.8e-17 | 否 |
| C（方差比） | **0** | 否 |

这是一个**否定式的关键读数**：三个候选哪怕在"按片重新归一化"的假设口径下，也不随组效应变化。也就是说，它们量的不是"组完整性被破坏"，而是与组效应无关的量。§13.30 第一批的「跨组排序翻转」就是被同一条检验钉掉的——本批把同一条检验用在了自己的三个新候选上，结论同样是"不及格"。

**这条检验本身也被测试守住**：`TestIdentifiabilityGuard` 里有一个"恒返回常数"的指标必须被判零敏感，一个"随组效应单调上升"的指标必须被判敏感——防止检验太松（放过无效指标）或太严（误杀真指标）。

## 那真实的代价在哪里

统计量上为零，不等于拆组无代价。**代价落在 mini-batch 组成上**：

- 同组样本被拆进不同 DP rank 的 mini-batch，每个 mini-batch 的**梯度噪声组成**会变（但一批的整体梯度不变——候选 C 已核）；
- PP 的微批边界与 mini-batch 划分顺序受影响（`_balance_batch` 在 `use_prefix_grouper` 下刻意跳过"小微批排两端"的重排，`ray_trainer.py:1197-1200`）；
- 训练侧的**均衡质量**代价已由 §13.30 L4 量化：组级均衡只付出约 2.6% 的均衡质量。

**这部分尚未量化**，登记为残留，不写成结论。

## 复现

```bash
python3 training_side_split.py             # 约 12 s，覆盖 summary.json
python3 training_side_split.py --verify    # 重跑并与归档逐字段比对（连跑两次通过）
python3 -m unittest test_training_side_split -v   # 14 个测试
```

LDM 划分复用 [`../rollout-group-split-tradeoff/`](../rollout-group-split-tradeoff/readme.md) 的实现（已与真实 verl 在 180 组输入上交叉验证分区一致），不在本目录复制第二份。执行环境：Python 3.9.6（macOS arm64），仅标准库。产物 sha256 见 [run-info.json](run-info.json)。

## 源码锚点（verl `1a9e711e`，只读核对）

| 文件:行 | 作用 |
|---|---|
| `verl/trainer/ppo/ray_trainer.py:1508` | rollout.n 组内连续排列 |
| `verl/trainer/ppo/ray_trainer.py:1511-1513` | `_balance_batch` 重排顺序（注释声明不影响优势） |
| `verl/trainer/ppo/ray_trainer.py:1629-1640` | 重排**之后**才 `compute_advantage` |
| `verl/trainer/ppo/core_algos.py:308-321` | GRPO 按 uid 分组算 mean/std |
| `verl/trainer/ppo/ray_trainer.py:1150-1203` | `_balance_batch` 三条路径（组级／keep_minibatch／逐条） |
| `verl/utils/seqlen_balancing.py:554-621` | `get_group_balanced_partitions` |
| `verl/protocol.py:864-889` | `chunk` 沿 dim=0 连续等分 → 分区顺序即 rank 归属 |
| `verl/workers/engine_workers.py:246-266` | mini-batch 按 dp 均分 |
| `verl/workers/engine_workers.py:189` | DP 组 loss `all_reduce(AVG)` |

## 边界（引用时必须一并写）

- **进程内解析＋仿真**，没有 GPU、没有引擎、没有把 verl 跑起来。优势复刻按 `core_algos.py` 逐行对应，但没有与真实 verl 做数值对齐（与 §13.30 的 LDM 交叉验证不同，那一步做过）。
- **只支持"在这个聚合语义下统计代价为零"**，不支持"拆组总是无害"——mini-batch 组成、PP 微批边界、均衡质量这三条不在本文件的量程内。
- **不支持"某个框架应该/不应该拆组"**。
- B、C 两个候选的向量模型是随机方向（`g_i = a_i · v_i`，`v_i` 随机单位向量），不含真实的 token 梯度结构；它们为零是**聚合语义**的推论，不是对真实梯度的测量。
- 本节**不是为了立一个正面结论**，而是为了撤回上一批的一个错误解读。若后续发现 mini-batch 组成代价显著，本节的"统计代价为零"仍然成立（两者不同层），但"拆组无代价"的说法会被独立证伪。
