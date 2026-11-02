# micro-batch 层的组完整性：可达上限比 §13.34 报的低，可接线的收益也比它小

2026-11-02，M1。支撑 [rollout 调研主文档 §13.35](../../notebook/rollout-performance-research.md)，
接 §13.34 登记的两条残留——「micro-batch 层 81.3% 的组被切开是否有解」「被跳过的 PP 气泡能不能补上」。

**不是 GPU 实测，不是端到端**。进程内按 verl 快照（`1a9e711e`）逐行复刻
`rearrange_micro_batches`（`verl/utils/seqlen_balancing.py:348-466`）的切分形状；
前缀块开销按 FLOP 口径计算；墙钟未测，不可与任何论文报值相除（§19.1 规则一）。

## 一句话结论

**§13.34 的「3.62×」是相对一个不可达参照点算的，不是落差。** 它拿「每个组只算一次前缀」当 1.00×，
而那要求**一个 micro-batch 装下整批**——训练是按 mini-batch 迭代的，所以这个参照点做不到。
换成可达的上限（每个 mini-batch 里每个组算一次）后，同一批配置下：
现状相对外参照 **3.69×**，可达上限相对外参照 **1.44×**，**真正能拿回的是 0.47 倍（现状的 47%）**。
而把 `force_group_size` 接成 `n` 这条路本身要付两笔账：上一级 mini-batch 的均衡（vPP=2 时不均衡 8.9%→17.0%），
以及批数粒度（组完整时 micro-batch 数是逐条的 39–42%，且只能取 vPP 的倍数）。

## 为什么做

§13.34 的三条结论里，判决二（micro-batch 层切开 81.3%）与判决三（前缀块 3.62×）读起来像「有个 bug 可以修」。
核完 `1a9e711e` 的调用链后，这条代价是三段叠加的，性质完全不同：

| 段 | 是什么 | 能不能动 |
|---|---|---|
| **A 代码路径** | 能让 micro-batch 看组的开关 `force_group_size` 在**所有 TRAIN 路径**上都是默认值 1 | 要改代码，但只是接线 |
| **B 参照系** | §13.34 报的 1.00× 要求整批装进一个 micro-batch，与 mini-batch 迭代互斥 | 改不了，是结构 |
| **C vPP 约束** | megatron 路径要求 micro-batch 数是 vPP 的倍数；组完整把上限从「样本数」压到「组数」 | 改不了，是结构 |

## A 代码路径：本快照里没有任何 TRAIN 调用方写入 `force_group_size`

只读核对 `1a9e711e`：

- `verl/workers/engine/utils.py:71` 用 `tu.get_non_tensor_data(data, key="force_group_size", default=1)`
  读**非张量**键，该键**只能由调用方显式写入**才非默认；
- `verl/workers/engine_workers.py:334-345`（`update_actor`）注入的 `default_keys` 五元组
  （`use_remove_padding`／`use_dynamic_bsz`／`max_token_len_per_gpu`／`micro_batch_size_per_gpu`／`use_fused_kernels`）
  **不含** `force_group_size`；`verl/trainer/ppo/ray_trainer.py` 全文 `grep force_group_size` **零命中**；
- 五个训练后端（fsdp／megatron／veomni／torchtitan／automodel）都调 `prepare_micro_batches(...)`，
  该函数没有 `force_group_size` 形参，只能走上面的非张量键；
- `use_dynamic_bsz_balance` 同理：`prepare_micro_batches` 的形参默认 `True`（`engine/utils.py:63`），
  fsdp 调用处（`transformer_impl.py:650-652`）只传 `same_micro_num_in_dp`，未透传。

**推断**：§13.34 的条件句「若 `force_group_size > 1`」在本快照的所有训练路径上都取不到真值，
micro-batch 层的行为恒等于 `force_group_size=1` 那条分支（单条长度排名 LDM ＋ 轻的夹两端）。

> **残留（必须一起写）**：`force_group_size` 的接线在 ray trainer 侧，而 **ray trainer 不在本 checkout 的
> `enable` 流程里**（如 AutoModel 后端那样 `enable` 才有）。`engine_workers.update_actor` 是 `@register`
> 方法，由上游调用点触发，本批只读核到 engine 入口为止。所以本结论**只覆盖 engine 入口能核到的路径**，
> 不作「任何配置都取不到」的结构保证。

这段与源码交叉验证无冲突：`rearrange_micro_batches` 本身支持 `force_group_size>1`（上游 PR #5438 引入，
注释写的是 "for RM training"），本批的复刻在 8 个形态上与它分区完全一致。

## B 参照系：可达上限是「每 mini-batch 每组件一次」

| 参照点 | 定义 | 一次遍历里前缀算几遍 | 可达？ |
|---|---|---|---|
| `no_share` | 每条响应各带一份前缀 | Σ 组内条数 | 是（但 `use_remove_padding=true` 时不是真实默认） |
| **`current`** | 现状：micro-batch 层按单条 LDM | Σ 每组片段数 | **是** |
| **`ideal_minibatch`** | 组完整、micro-batch 层按组 LDM | 每 mini-batch 每组件一次 | **是（接线即可）** |
| `ideal_wholebatch` | 每组件一次前缀（§13.34 的 1.00×） | 每组件一次 | **否**（要求整批进一个 micro-batch） |

读数（`layer0_structure`，n=8，dp∈{1,2,4}，组数/ mini-batch∈{1,2,4,8}，10 个种子，240 个案例）：

| 预算 | 每组平均片段数（现状） | 每组平均片段数（可达上限） | 现状 / 外参照 | 可达上限 / 外参照 |
|---|---|---|---|---|
| 16384（默认） | **2.80** | 1.00 | **3.69×** | **1.44×** |
| 装得下 mini-batch（loose） | 1.00 | 1.00 | 1.44× | 1.44× |

**「可达上限 / 外参照」恒等于每个 mini-batch 的组数**（单测 `test_one_wholebatch_prefix_per_group_regardless_of_dp`
守住其分母与外参照同源）。所以 §13.34 的 3.62× 里，**能动的只有 3.69→1.44 这一段（约 0.47 倍）**；
剩下的 1.44× 是「训练按 mini-batch 迭代」这件事本身的价格。

720 个案例（n∈{4,8}、30 种子）上 `ideal_over_current` 均值 **0.468**，区间 **0.162–1.000**：
预算宽松或组数少时一点收益都没有，预算是瓶颈时最多能省到六分之一。

## C vPP 约束：组完整把批数上限从「样本数」压到「组数」

`rearrange_micro_batches` 的批数公式（`seqlen_balancing.py:396-412`）：

```
num_groups       = batch_size // force_group_size        # 逐条切时 = 样本数
num_micro_batches = min(num_groups, ceildiv(total_seqlen, max_token_len))
if num_batches_divided_by:  num_micro_batches = roundup_divisible(num_micro_batches, vpp)
assert num_micro_batches <= num_groups                    # 409 行，无消息文本
```

一整个 8 组 × 8 条 mini-batch（137,080 tokens）压实预算：

| 预算占 mini-batch | 逐条切批数 | 组完整批数 |
|---|---|---|
| 100% | 2 | 2 |
| 35% | 4 | 4 |
| 10% | 10 | **8**（被组数封顶） |
| 5% | 20 | **8** |

**预算是软约束、vPP 是硬约束**：逐条切能随预算细分成 20 批，组完整最多 8 批（=组数），
中间只有 vPP 的倍数可选。

三条 vPP 读数（每组数/预算 9 个组合）：

| vPP | 组完整路径可分？ | 批数比（组 / 逐条） | 不均衡（逐条 → 组完整） |
|---|---|---|---|
| 2 | 9/9 可分 | **0.42** | 8.9% → **17.0%** |
| 3 | **9/9 顶破 `num_groups` → 真实代码直接断言失败** | — | — |
| 4 | 6/9 可分 | **0.39** | 11.5% → **25.4%** |

**vPP=3 全部不可行**：组数 8 时 `roundup(8,3)=9 > 8` 触发 `seqlen_balancing.py:409` 的断言。
这不是本模型的失效——`xcheck_verl.py` 用真实 `rearrange_micro_batches` 跑同一形态，得到同一个断言。

**代价的方向会随组数翻转**：vPP=2、默认预算下 `mini_groups=2` 时组完整反而更均匀（0.05% vs 1.59%），
`mini_groups=4/8` 时明显更差（18.6% / 32.3% vs 1.25% / 1.01%）。
机制是组是小微批时更均匀，组本身成为小微批后，组间长度差没有别的东西可以摊薄。

## 识别性检验与负向对照

| 检验 | 期望 | 实测 |
|---|---|---|
| n=1（每组一条） | 组完整与逐条等价，比值 = 1 | 1.0 ✅ |
| 预算装得下 mini-batch | 两者都只切 1 段/组 | 1 / 1 ✅ |
| 组完整后可分路径 | 从不拆组 | 全 True ✅ |
| 组完整下每组片段数 | 不超过一个 mini-batch 的组数 | 成立 ✅ |
| 预算变大 | 现状片段数单调不增 | 成立 ✅ |
| 组完整片段数 | 与 dp 无关 | 1 / 1 / 1 ✅ |
| 可达上限 | 在每个案例上都不劣于现状 | 全部 ≤ 1 ✅ |
| 前缀长度 128/512/2048 | 两个比值逐位相同 | 4.09375 / 1.0，三档一致 ✅ |
| 外参照的分母 | 与 dp、mini_groups 无关 | 三组配置绝对 FLOP 相等 ✅ |

## 与真实 verl 的交叉验证

`xcheck_verl.py`（CPU torch 2.15.0，直接 import 快照源码，只跑切分算法、不初始化模型或引擎）：

| 形态 | 结果 |
|---|---|
| `force_group_size=1` 无 vPP（§13.34 已核回归） | 8 分区完全一致 |
| `force_group_size=n` 无 vPP | 完全一致 |
| `force_group_size=n` + vPP=2 / 4 | 完全一致 |
| `force_group_size=n` + vPP=3（组数 8、预算 16384） | 真实实现 **AssertionError(num_micro_batches<=num_groups)**，模型记为 infeasible |

**7 个可切形态 7/7 分区一致，1 个形态两边同时判不可行。**

## 模型修正（本批自查记录，全部是跑通之后才发现）

| # | 第一版写法 | 错在哪 | 现在 |
|---|---|---|---|
| 1 | 「整批共享」与「mini-batch 内组共享」混用同一个分母 | 两个参照系不同，§13.34 报的 3.62× 因此被读成落差 | 三参照点分列，且用单测钉死「可达上限 / 外参照 = 每 mini-batch 组数」 |
| 2 | vPP 顶破上限时替它 rescale 到上限 | 真实代码是 `assert`，不是变通 | 记为 infeasible 并交给 `xcheck_verl.py` 确认真实行为 |
| 3 | 交叉验证里长度上界取 6000 | 触发 `max_token_len >= max_seq_len` 前置断言，误报「真实不可行」 | 长度上界压到预算以内，并把该断言单列为「非本批口径」 |
| 4 | 交叉验证只按异常消息文本分类 | `assert num_micro_batches <= num_groups` **没有消息**，分类落空 | 改按「模型侧也算出顶破上限」判定 |
| 5 | 识别性检验用 `details[0]`（只有 `mini_groups=2` 一组）代表整层 | 抽样，vPP=4 的失败案例没被看见 | 遍历 vPP∈{2,4} 的全部组合 |
| 6 | 单测期望值先写成「比值 = 该 mini-batch 的组数 = 4」 | 分母是**整条 rank 视图**而非一个 mini-batch，真实恒等式是 B/dp/n | 改用它自己的解析式，另加「外参照分母与 dp/mini_groups 无关」的结构测试 |

## 边界（引用时必须一并写）

- **进程内切分形状复刻**，非 GPU 实测、非端到端；**墙钟未测**，3.69×/1.44×/0.47× 都是**前缀块 FLOP 比值**，
  不是时间，不可与其他报值相除（§19.1）。
- **A 段的结论只到 engine 入口**：ray trainer 不在该 checkout 的 `enable` 流程里，未核到
  「ray trainer 是否写这个键」。本批**不支持**「verl 的组完整 micro-batch 一定用不了」这类全局断言。
- **可达上限口径假设碎片之间不能共享**：组完整后碎片不再产生，所以这一点在本口径下不构成偏差；
  但「现状」一列沿用 §13.34 的上界口径（切开的碎片之间不共享），若上游 `prefix_grouper` 能按碎片共享，
  现状会更接近可达上限，**本批报告的 0.47 倍应视为收益上限**。
- **未计墙钟与 PP 气泡**：vPP 那条约束的实际代价是气泡时长（`num_batches_divided_by` 存在的理由），
  本批只量「批数与不均衡」，气泡本身仍未测；所以**不能说「接线一定亏」**。
- **长度分布是构造参数**（均值 2200、CV 0.45、整数化），不是从训练日志取的；
  组内不齐整（同一 prompt 的 n 条长度差异大）时读数会更极端，本批未扫。
- **只覆盖 verl `1a9e711e`**；`force_group_size` 的语义（"for RM training"）来自上游代码注释，未核 RM 训练路径。
- **不支持**「应该开关 `use_prefix_grouper`」或「应该给 micro-batch 接 `force_group_size`」这类决定；
  只支持「§13.34 的 3.62× 是相对不可达参照系的计数、能动的上限约为 0.47 倍、且要付 vPP 与 mini-batch 均衡两笔账」。

## 复现

```bash
python3 microbatch_group_integrity.py             # 约 3 s，覆盖 summary.json
python3 microbatch_group_integrity.py --verify    # 重跑并与归档逐字段比对（连跑两次通过）
python3 -m unittest test_microbatch_group_integrity -v   # 15 passed

# 与真实 verl 交叉验证（CPU torch，不进本仓库依赖）
uv venv /tmp/verl-seqlb-venv --python 3.12
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install numpy packaging ray tensordict omegaconf transformers codetiming
VERL_ROOT=../../verl /tmp/verl-seqlb-venv/bin/python xcheck_verl.py
# 一致 7｜不一致 0｜真实实现断言失败 1（共 8 个形态）
```

执行条件与产物 sha256 见 [run-info.json](run-info.json)。
`microbatch_group_integrity.py` 直接 import [../rollout-pg-boundaries/pg_boundaries.py](../rollout-pg-boundaries/pg_boundaries.py)
复用 LDM 与 batch 均衡复刻，避免第二份实现漂移；改那里需要重跑本目录的 `--verify`。

## 源码锚点（verl 1a9e711e）

- `verl/utils/seqlen_balancing.py:348-466`：`rearrange_micro_batches` 全形状（含 `force_group_size` 分支 415-452、`use_dynamic_bsz_balance` 440-462）
- `verl/utils/seqlen_balancing.py:396-409`：`num_groups` 定义与 `assert num_micro_batches <= num_groups`
- `verl/utils/seqlen_balancing.py:412`：`roundup_divisible(..., num_batches_divided_by)`
- `verl/workers/engine/utils.py:63,71`：`use_dynamic_bsz_balance=True` 形参默认、`force_group_size` 读非张量键默认 1
- `verl/workers/engine/utils.py:57-94`：`prepare_micro_batches`（无 `force_group_size` 形参）
- `verl/workers/engine_workers.py:334-345`：`update_actor` 注入的五个默认键（不含 `force_group_size`）
- `verl/workers/engine/fsdp/transformer_impl.py:650-652`：调用处只传 `same_micro_num_in_dp`、未透传 `use_dynamic_bsz_balance`
- `verl/workers/engine/megatron/transformer_impl.py:703-711`：`num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage` 的来源
- `verl/trainer/ppo/ray_trainer.py:1160-1212`：`_balance_batch`（mini-batch 层的组级 LDM 与跳过的重排）
