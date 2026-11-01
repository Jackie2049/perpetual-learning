# C1 计时边界约定的可判定性核验（附 C3 的定义可取得性）

2026-11-02，M1。支撑 [rollout 调研主文档 §21.6](../../notebook/rollout-performance-research.md)。
不是 GPU 实测，不是任何系统的端到端；是对四个只读源码快照的**静态核验**。

## 一句话结论

**§21 的三条约定里，C1 的三件套在四个框架的 11 个计时块上有 9 个能单凭源码判出、2 个要靠外部补偿；
C3 的 η 定义在四个快照里一个都算不出来，只能近似或不可得。**
更要紧的一条是接缝：**没有任何计时块能同时、分开地给出 t1 与 t2**——
能给出"权重同步"的块都只能到 `t1∪t2`，于是 §21.5 实测的"默认整体报"不是一条纸面建议，
而是语料里四个框架当前采集能力的直接后果。

## 为什么做

§21（P-5）把 C1/C2/C3 写成报值约定，§21.5 只实证了 C2 的拆分侧是脆的。
C1 与 C3 至今仍是"纸面规定"：C1 说"缺件即不可横比"，但没说**哪些件在真实计时块里判得出来**；
C3 说"报 η 分布与 S"，但没说**语料里有没有哪个框架能按这个定义算**。
本批把这两条放进固定 commit 的源码里逐条核，让"约定可执行到哪一步"变成可复跑的读数。

## C1：三件套在 11 个计时块上的可判定性

判定规则写死在脚本里（`DECIDABLE` / `PARTIAL` / `UNDECIDABLE` → 等级 A/B/C）：

- **① 块名 + 框架 + 固定 commit**：能用行号定位**不等于**块名唯一。源码里有独立命名的计时键才判 `decidable`；
  只有裸的行间计时（算一段时差、没有命名）判 `partial`——引用者无法复述它是哪个块。
- **② 块内是否含策略等待**：看源码里是否有能证明"含/不含等待"的**具体语句**（取批、轮询、pause/resume 括弧）。
- **③ 分项与总量是否重叠**：同上，看块内是否包了与主操作不同集合的动作。

每一项的源码证据都登记为 `proof`，脚本在固定 commit 上逐条核**字面命中**；
证据不命中就不按记下的判定算等级（这条由 `test_all_anchors_hit` 守住）。

| 计时块 | 框架 | ① 块名 | ② 等待 | ③ 重叠 | 等级 |
|---|---|---|---|---|---|
| `gen`（V1 `trainer_base.py`） | veRL `24f25b03` | 可判 | 可判（含 `replay_buffer.sample`） | 可判（含 `on_sample`） | **A** |
| `gen`（`ray_trainer.py`） | veRL | 可判 | 可判（同步等整批） | 可判（含 `sleep_replicas`） | **A** |
| `update_weights` | veRL | 可判 | 可判（包住整个 manager 流程） | 可判 | **A** |
| generation 时差 | OpenRLHF `dc2a7ad3` | **部分**（裸计时，无命名键） | 可判 | 可判 | **B** |
| `timing/step_total` | OpenRLHF | 可判 | 可判（含队列 `get(block=True)`） | 可判 | **A** |
| `exposed_generation` | NeMo RL `a9d1a4be` | 可判 | 可判（含 evict＋饥饿轮询） | 可判 | **A** |
| `weight_sync` | NeMo RL | 可判 | 可判 | 可判 | **B**（空块分支：`pass`） |
| `wait_for_generation_buffer` | SkyRL `e0893f6e` | 可判 | 可判 | 可判（含条件筛组） | **A** |
| `group_completion_time` | SkyRL | 可判 | 可判（不含容量等待） | 可判 | **A** |
| `sync_weights` | SkyRL | 可判 | 可判（含 pause/resume 括弧） | 可判 | **A** |
| `sync_weights_only_transfer` | SkyRL | 可判 | 可判（只 broadcast＋finish） | 可判 | **A** |

**读数一：C1 的"缺件即不可横比"在真实源码里几乎不触发，触发的是"件在、但不够唯一"。**
11 个块里 0 个掉到 C 级；两个 B 级都不是"查不到"，而是查到了却需要引用方额外说明：
OpenRLHF 的生成时差没有命名键（同名块在不同框架里无法对齐），NeMo RL 的 `weight_sync` 有一条
`with self._timer.time("weight_sync"): pass` 的空块分支。**这两类恰恰是最容易在引用时被悄悄忽略的**——
前者让"generation 时间"变成一句无法复述的描述，后者让"权重同步 0 秒"看起来像结论。

**读数二：C1 不能靠"有没有写一段证据文字"来判，只能靠证据是否命中源码。**
本脚本第一版就是这样错的：只要判定里写了 `waiting_evidence` 字段就算 `decidable`，
结果 11 个块全部判成 A——等于用结论证明结论。改成核对 `proof` 记录的源码锚点后，
OpenRLHF 生成块与 NeMo 空块分支才被识别出来。这条写在这里，是因为它也会发生在任何人工核 C1 的场合。

## C3：η 的定义在四个快照里能拿到什么

C3 把 η 定义为 `v(消费时 policy) − v(生成时 policy)`，要求报 **max / P50 / P90 的分布**。

| 框架 | 实际能拿到的版本量 | 是否等于 C3 定义 | 差在哪 |
|---|---|---|---|
| SkyRL `e0893f6e` | `self.global_step − group.global_step_when_scheduled` | **否** | 报的是"训练步数年龄"。一个 step 内若发生多次更新，同一批轨迹读到的数不变；消费侧版本取的是**训练步计数**而非**权重版本**，且该计数在权重同步后、下一次 `notify_capacity_change` 前与引擎实际权重并不同步 |
| NeMo RL `a9d1a4be` | `evict(current_train_weight=self._trainer_version)` | **部分** | 能识别陈旧并按组淘汰，但不产出 C3 要的 max/P50/P90 年龄分布 |
| veRL `24f25b03` | min/max `global_steps` 标签；fully-async 用 `staleness_samples` 计数 | **部分** | 版本标签按步推进；fully-async 的暂停条件读的是计数而非逐样本版本差（§12 已记「两个训练时钟」） |
| OpenRLHF `dc2a7ad3` | 无逐条 policy 版本 | **否** | slot 携带的 `global_step` 用于 eval 时机，不构成逐条年龄 |

**读数三：C3 的 η 在语料里没有一处是"照定义报"的，四处都是近似。**
这不否定 C3，但它意味着文中写"某系统支持 η=k"时必须说清是哪一种近似：
步数年龄、按组淘汰、还是概率差。四者互相替换会系统性改变结论方向（步数年龄在"一步内多次更新"下恒低估）。

**读数四：S 也有两种口径，差别等于丢弃率。**
C3 要求 S 是"被接纳进训练的轨迹数"。用生成侧计数当分母时，被丢弃的组仍被算进去。
以 SkyRL 的 `sample_full_batch` 为例（每步生成 32 组）：无丢弃时两口径相同；
每步丢 8 组（零方差组，源码 `_should_keep_group` 判 `zero_variance_filter` 后无存活索引即丢）则生成侧比训练侧高 **33.3%**。
这与 §12 记的 OpenRLHF `filter_pass_rate` 分子为响应、分母为 prompt 是同一类口径混用。

## C1 与 C3 的接缝：没有任何块能分开给出 t1

把每个计时块的结束位置按 C2 的三时间戳归类：

- 只到 `t1∪t2`（权重同步）：veRL `update_weights`、NeMo `weight_sync`、SkyRL `sync_weights`、SkyRL `sync_weights_only_transfer`
- 只到 `t2`（满批）：veRL V1 `gen`、SkyRL `wait_for_generation_buffer`
- 什么时间戳都不是：OpenRLHF 生成块（`none`）、NeMo `exposed_generation`、SkyRL `group_completion_time`
- **独立给出 `t1`：0 个**

**这是本批最要紧的一条。** §21.5 用仿真说明"硬拆会错、整体报不偏"，本批说明**为什么会这样**：
四个框架的计时体系里，t1（权重在引擎侧真正生效）**不是一个被采集的时刻**。
凡是想拆 B2/B3 的引用者，都只能拿某个"权重同步块结束"当 t1——而那个块按上表本来就含
`t2` 侧内容（pause/resume、KV 唤醒）。这正是 §21.5 里 `ack_sep` 那类做法的真实版本：
它不是研究者偷懒，是现有采集能力的直接结果。

## 这次自查修正了什么

**一处**，但很典型：`item_status` 初版把"判定里写了 evidence 字段"当作可判，
于是 11 个块全部判成 A 级，`openrlhf.timing_generation` 的"无命名计时键"完全没被识别出来。
改为核对 `proof` 登记的源码锚点是否命中后才出现 B 级。此外另修两处**坐标错误**
（OpenRLHF 两处行号写成了注释行、SkyRL `sample_full_batch` 的行号差 9 行），
都是被 `check_anchor` 抓出来的，不是眼看发现的。

## 复现

```bash
python3 c1_timing_blocks.py             # 覆盖 summary.json
python3 c1_timing_blocks.py --verify    # 重跑并与归档逐字段比对
python3 -m unittest test_c1_timing_blocks -v   # 10 个测试
```

脚本会读四个只读 checkout（路径见 `CHECKOUTS`）；缺失时 `test_checkouts_are_at_recorded_commits` 跳过，
其余证据核对照常报缺。执行条件与产物 sha256 见 [run-info.json](run-info.json)。

## 边界（引用时必须一并写）

- **静态度量，不是测量**：本批只读源码，没有跑任何框架，不给任何耗时数字。
- **判定是人读代码的结论**：脚本核的是"这段源码还在不在、锚点还命中不命中"，
  它**不能**核"人有没有读错语义"。三件套的取值（含/不含等待、是否重叠）仍属作者判断，
  正文引用时按"本文判读"标注，不写成"源码如此"。
- **只覆盖四个快照**：veRL `24f25b03`、OpenRLHF `dc2a7ad3`、NeMo RL `a9d1a4be`、SkyRL `e0893f6e`，
  是 §12 已登记的同四个 commit；换版本结论可能变，须重跑。
- 本实验**不支持**"某框架的计时块有缺陷"这类结论，只支持"按 C1 的三件套，这些块的可判定程度如下"。
