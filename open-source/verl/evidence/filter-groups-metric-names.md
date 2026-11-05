# verl：`algorithm.filter_groups.metric` 的三份说明互相不自洽，两个名字在 V1 上零写入点（2026-11-05）

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（主流 RL 训练框架，本仓历史上做过多次贡献）

**【贡献点】** DAPO 动态采样的组过滤指标 `algorithm.filter_groups.metric` 在 V1 trainer 上按**名字**去 `extra_fields.reward_extra_info` 取键，取不到就抛 `RuntimeError`。而「能填哪些名字」有三份互不相同的说明：[配置 docstring](https://github.com/verl-project/verl/blob/31cfccb4/verl/trainer/config/algorithm.py#L48-L52) 写 `"acc", "score", "seq_reward"`、[`docs/algo/dapo.md:80`](https://github.com/verl-project/verl/blob/31cfccb4/docs/algo/dapo.md#L80) 举例 `score / seq_reward / seq_final_reward`、随仓库脚本 `fully_async_policy/shell/grpo_qwen35_35b_megatron_async.sh:64` 用 `response_filter`。**`seq_reward`、`seq_final_reward`、`response_filter` 在 V1 上零写入点**，照填会在第一个训练步崩。[实验与复现](../../experiments/verl-filter-groups-metric-names/README.md)

**【类型】** B 类（主动发掘）：源码 + CPU 运行时读出的一致性缺口，**未提交上游**

**【价值评估】** 5 / 10 — 影响面窄（官方 recipe 全用 `acc`），但崩在第一步、报错信息会把人引向错误结论，且成本极低就能修（改注释 / 文档 / 加一行配置校验）。

**【进展】** 2026-11-05 在 `31cfccb4`（2026-11-04）上完成三表抽取 + 真 `ReplayBuffer` 运行时对照 + 历史归属；自检 8/8、负向对照 4/4；上游查重无重复 issue。**对外待用户逐项授权。**

**【大白话讲明白】** verl 的 DAPO「动态采样」要扔掉「n 个答案全对或全错」的组，判断依据就是 `filter_groups.metric` 填的那个指标名。V1 实现是拿着这个名字去轨迹的 `reward_extra_info` 字典里按键取值，取不到就报错停机。而仓里有四份说明在告诉用户能填哪些名字，其中 `seq_reward`、`seq_final_reward`、`response_filter` 三个在任何 V1 reward manager 与官方 scorer 组合下都不会被写出来——照着填就在第一个训练步崩，报错还只说「缺这个指标」。修复成本极低（改注释/文档或加一行加载期校验），所以这更像一次说明书订正，而不是补功能。

**【What–Why–How】**

- **What：** `algorithm.filter_groups.metric` 在三份说明里被点名 6 个名字，其中 `seq_reward`、`seq_final_reward`、`response_filter` 在 V1 上**零写入点**；`score` 只有 `math_dapo` 一种 scorer 会给。
- **Why：** V1 的 `_dapo_filtered_keys` 只认 `extra_fields.reward_extra_info` 的字面键，而这些名字原本由 DAPO recipe 自己的 trainer 实现（`dapo_ray_trainer.py:240` 有 `seq_final_reward` 分支）。名字搬进 V1 后语义没跟上，说明文字也没同步。
- **How：** 从真实源码抽三张表做差，再用 exec 加载的真实 `ReplayBuffer`（只替身 `transfer_queue`／`SkipManager`）按上游 CPU 测试同一套写盘顺序逐名字跑，最后看上游单测覆盖到哪些字面值。`--selftest` 8/8、负向对照 4/4。**边界**：未启动 trainer、未跑训练、未触上游；自定义 scorer 可以返回任意键，所以"不可达"只对官方 scorer + 官方 reward manager 成立。
<!-- contribution-summary:end -->

## 大白话

verl 的 DAPO「动态采样」有个开关：同一道题采出来的 n 个答案如果全都对或全都错，这组就没有训练信号，直接扔掉重新采。要判断「全都一样」，得知道拿哪个数比较——这就是 `filter_groups.metric`。

V1 的实现是：**拿着你填的这个字符串，去每条轨迹的 `reward_extra_info` 字典里按名字取**。取不到就报错。那么关键就变成：**reward manager 到底会往那个字典里塞哪些名字？**

把代码扫一遍只有五个写死的名字（`acc` 最常用），外加「scorer 返回 dict 时把它返回的键原样塞进去」。而官方 scorer 里只有 `math_dapo` 一种返回 dict（键是 `score`/`acc`/`pred`），其余数据集都返回一个 float ⇒ **实际只有 `acc`**（+ 回退分支的 `acc`）。

可是文档那边写着：`seq_reward`、`seq_final_reward` 也能填。这两个名字**在 V1 里从来没有代码产出**——它们属于老的 recipe trainer（`dapo_ray_trainer.py` 里真有 `if metric_name == "seq_final_reward"` 的分支），搬进 V1 之后忘了搬语义，说明却留了下来。

## 三份说明 vs 代码事实

| 名字 | 配置 docstring | yaml 注释 | `docs/algo/dapo.md` | 随仓库脚本 | V1 代码里真的会出现 |
|---|---|---|---|---|---|
| `reward` | ✓ | ✓ | — | — | ✓（canonical，走 `rm_scores`） |
| `acc` | ✓ | ✓ | ✓ | ✓ ×2 | ✓（5 个 reward manager 都写） |
| `score` | ✓ | — | ✓ | — | ✓ 但**只有 `math_dapo` scorer 给** |
| `seq_reward` | **✓** | — | ✓ | — | ✗ **零写入点** |
| `seq_final_reward` | — | — | **✓** | — | ✗ **零写入点** |
| `response_filter` | — | — | — | **✓ 在用** | ✗ **零写入点** |

## 运行时读数（真 `ReplayBuffer._dapo_filtered_keys`）

造一个「2 条同分轨迹的组」，逐名字跑（`stub_env.py` 只替身 `transfer_queue` 与 `SkipManager`，判定逻辑是上游原文）：

| 名字 | 把能给的键都给上 | 真实场景（只有 `acc`） |
|---|---|---|
| `reward` | ran | **ran** |
| `acc` | ran | ran |
| `score` | ran | **raised** |
| `seq_reward` | **raised** | raised |
| `seq_final_reward` | **raised** | raised |
| `response_filter` | **raised** | raised |

报错原文：`RuntimeError: Finished groups are missing DAPO metric 'seq_reward': [...]`。

## 为什么不是「缺陷」而是「说法」

- 名字本身不是实现错了：`seq_*` 在 recipe trainer 上是有效的，只是 V1 没有对应语义。
- 用户有绕法：填 `reward`（canonical）或 `acc`，或者自己写 scorer 让它返回同名键。
- 所以对外最小动作应当是**文档/注释与配置校验**，而不是给 V1 补 `seq_*` 实现（那是新增语义，得设计者定）。

## 上游查重（2026-11-05，只读 GitHub API）

- `filter_groups.metric` → 0 条
- `seq_reward` / `seq_final_reward` → 各 8 条，均与指标名无关
- `filter_groups` → 16 条，无一条说这件事

## 下一步

对外动作（issue / 文档 PR / 配置校验 PR）**均待用户逐项授权**。准备动作已完成：实验目录含可复现脚本与读数。
