# verl V1 DAPO 组过滤下的采样轮询：`poll_interval` 的收益与代价

2026-11-02 执行（M5 算力侧，TM-008 边界实验）。前序实验与补丁见 [experiments/verl-v1-poll-quantum](../verl-v1-poll-quantum/README.md)；机制背景见主文档 [§13.44](../../notebook/rollout-performance-research.md#1344-gen-二态的机制是-v1-sync-trainer-的-20s-采样轮询量子逐引擎计时-30-步都是-6464生成从未变慢2026-11-02-m5-算力侧第五十批)。

## 目的

TM-008 的补丁只加了可达性（默认 2.0 不变），但它的验证边界写得很清楚：

> **DAPO 过滤/refill 或 `max_inflight_gen_batches>1` 时，采样时机可能改变可选组集合（陈旧度/逐出），本实验的固定批配置不覆盖该情形。**

固定批配置下 `sample()` 只等一个满批，等待时机不影响返回集合，所以 poll 只改等待。**DAPO 路径不是这样**：`ReplayBuffer.sample()` 变成一个派发循环，补抽（refill）的额度是按"每次轮询新观测到的被过滤组数"累计的，停止条件（够 32 个有效组）也要等某次轮询才被看到。于是轮询间隔可能改变**本步生成多少组**——那就不只是等待，而是算力成本。

本实验回答：**在 DAPO 组过滤 + 多 in-flight 下，把 poll 从 2.0 调到 0.1，省下的墙钟里有多少是纯等待、有多少被多生成的组吃掉。**

## 代码面（为什么这里会不一样）

`verl/trainer/ppo/v1/replay_buffer.py:405-494`（上行 `main 3fdf2b5488`）：

| 位置 | 行为 |
|---|---|
| `:455` `refill_credit += 2 * dapo_count + failed_count` | 每次轮询把**新观测到**的被过滤组数换算成补抽额度（2 倍） |
| `:461-470` `available_slots = max_inflight_prompts - inflight_count` | 额度受 in-flight 上限约束，`max_inflight_prompts = max_inflight_gen_batches × train_batch_size` |
| `:472` `can_select = has_enough_samples and (not dapo_enabled or inflight_count == 0)` | DAPO 下必须先**排空**所有在飞请求才能选批 |
| `:478-484` | 排空后多出来的有效组被丢弃，记 `training/filter_groups/discarded_surplus_samples` |
| `:391-393` `_wait_for_next_poll` | 循环末尾 `time.sleep(poll_interval)` |

由此可推出两个方向相反的预期：

- **省等待**：每步大约要跑 3–4 轮"派发→等完成→再派发"，2.0s 量子会在每轮末尾多等最多 2s；
- **多生成**：轮询越细，越早看到被过滤组、越早补抽，因此到"够 32 个"被观测到那一刻，管道里在飞和已完成的组更多 ⇒ 生成量上升，排空时丢弃的 surplus 也上升。

所以这是一个真实的权衡，而不是单调的"调小就更快"。

`training/filter_groups/evicted_samples`（本步被过滤掉的组数）与 `discarded_surplus_samples` 都是逐 step 打点的，可以直接从控制台日志读；`generated_groups = 32 + evicted + surplus` 是本步实际生成的组数（选批时所有在飞请求已排空，故三者覆盖全部生成组）。

## 实验配置

- **机器**：4090-1，GPU 0,1（同 NUMA）。窗口内 GPU 2 由 liyaowei embed 常驻（21728 MiB），其余空闲。
- **栈**：verl `0.10.0.dev`（`verl-24f25b03` 树，与 TM-008 同一棵树）＋ vLLM 0.24.0 ＋ torch 2.11.0+cu129；`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`、`NCCL_P2P_DISABLE=1`
- **模型/数据**：Qwen2.5-1.5B-Instruct，gsm8k `train_sub480`／`test_sub120`
- **任务配置**：与 TM-008 的 2 卡组一致（`train_batch_size=32` prompts × `rollout.n=4` = 128 samples/step、`ppo_mini_batch_size=16`、micro 1、max_prompt 512／max_response 256、bf16、sdpa、`optimizer_offload=True`、seed 42、15 步）
- **本实验新增（全部运行共有）**：`algorithm.filter_groups.enable=True`、`algorithm.filter_groups.metric=acc`、`algorithm.filter_groups.max_inflight_gen_batches=2`

  注：配置注释里写的 `seq_final_reward` 在 V1 里**没有生产者**（`reward_extra_info` 只由 V1 reward manager 产出 `acc`／`overlong*`），填它会直接抛 `Finished groups are missing DAPO metric`。这类"文档写了但没接线"的点单独记在这里，不在本实验结论内。
- **变量**：只在 patched 树上改 `trainer.v1.sampler.poll_interval`；`dbase` 用未打补丁的原树。

## 执行账

| 运行 | 树 | poll | 窗口（UTC） | exit | 步数 |
|---|---|---|---|---|---|
| dbase | 原树 | 默认 2.0 | 14:02:11–14:11:46 | 0 | 15/15 |
| dp200 | patched | 2.0 | 14:12:26–14:22:01 | 0 | 15/15 |
| dp050 | patched | 0.5 | 14:22:03–14:31:28 | 0 | 15/15 |
| dp010 | patched | 0.1 | 14:31:29–14:40:44 | 0 | 15/15 |
| dp200b | patched | 2.0（重复） | 14:43:00–14:52:43 | 0 | 15/15 |
| dp010b | patched | 0.1（重复） | 14:52:44–15:02:04 | 0 | 15/15 |

六次运行结束时 GPU 均释放（0/1 回到 1–4 MiB）；checkpoint 逐轮删除。配置解析已在日志里核对：patched 运行出现 `'poll_interval': 2.0/0.5/0.1`，`dbase` 无此键。

## 结果（稳态窗口 steps 3–14，n=12）

### 相位均值

两臂各两跑取均值（`dp200`+`dp200b` vs `dp010`+`dp010b`）：

| 相位 | poll 2.0 | poll 0.1 | Δ | 相对 |
|---|---|---|---|---|
| **gen** | **8.474s** | **7.289s** | **−1.185s** | **−14.0%** |
| old_log_prob | 2.472 | 2.474 | +0.002 | +0.1% |
| ref | 2.288 | 2.290 | +0.001 | +0.1% |
| adv | 0.128 | 0.112 | −0.016 | −12.2% |
| update_actor | 10.786 | 10.703 | −0.083 | −0.8% |
| update_weights | 2.282 | 2.282 | +0.001 | +0.0% |
| **整步** | **26.534s** | **25.246s** | **−1.287s** | **−4.9%** |

除 gen 外所有相位都停在噪声内（±0.1%），所以整步收益几乎全来自 gen。相位和到整步的残差两臂都是 0.10s。

### 重复性、惰性与方差

| | gen 均值 | 同臂两次相差 | 逐 step sd |
|---|---|---|---|
| 原树 2.0（dbase） | 8.489 | — | 1.729 |
| patched 2.0（dp200） | 8.837 | 0.727s（vs dp200b） | 1.535 |
| patched 2.0（dp200b） | 8.110 | | 1.635 |
| patched 0.5（dp050） | 7.207 | — | 1.493 |
| patched 0.1（dp010） | 7.242 | 0.094s（vs dp010b） | 0.940 |
| patched 0.1（dp010b） | 7.336 | | 0.682 |

- **补丁惰性**：原树 8.489 / 26.511s 与 patched 两跑的均值 8.474 / 26.534s 一致（0.2%／0.1%）。
- **效应大于重复噪声**：−1.185s 的 gen 收益是 2.0 臂两次重复差（0.727s）的 1.6 倍，而 0.1 臂自身重复只差 0.094s；四组读数里两个 0.1 跑（7.24／7.34）都低于三个 2.0 跑（8.11／8.49／8.84）。
- **方差同时下降**：逐 step sd 从 1.5–1.6s 降到 0.68–0.94s（−45%），与固定批实验里"二态消失"是同一件事。

### 代价：多生成 13%

| 臂 | 生成组/步 | 过滤率 | 丢弃 surplus | 有效率 | 最小需求 | 超额 |
|---|---|---|---|---|---|---|
| 原树 2.0 | 116.2 | 0.637 | 10.2 | 27.5% | 88.2 | 1.32× |
| patched 2.0（两跑均值） | 117.4 | 0.618 | 12.9 | 27.3% | 83.8 | 1.40× |
| patched 0.5 | 122.5 | 0.620 | 14.6 | 26.1% | 84.2 | 1.46× |
| patched 0.1（两跑均值） | **132.8** | 0.625 | **17.9** | **24.1%** | 85.2 | **1.56×** |

- **0.1 臂比 2.0 臂多生成 15.4 组/步（+13.1%）**，其中被过滤组 +7.9、丢弃 surplus +5.0（+39%）。两个 0.1 跑（129.8／135.8）都高于两个 2.0 跑（123.8／110.9）和原树（116.2）。
- 这不是准确率漂移造成的：两臂 reward 均值 0.346 vs 0.338、过滤率 0.618 vs 0.625，方向相反（0.1 臂准确率更低本应过滤更多，但只多 1%；生成量多 13%）。按"32/(1−过滤率)"折算的最小需求，两臂只差 1.7%，超额倍数却是 1.40× → 1.56×。
- **量级判定**：逐 step 生成量的 sd 约 25 组，单跑均值 SE≈7，两跑合并后差 15.4 组约 **2 个标准误**。两个重复方向一致、且与机制解释（更早看到过滤组→更早补抽）吻合，但**不足以称为已精确标定的因果量**；下结论时按"约 10–15%"用。

### 数值

| 指标（steps 3–14） | 2.0 臂 | 0.1 臂 | 原树 |
|---|---|---|---|
| `critic/rewards/mean` | 0.3457 | 0.3379 | 0.3307 |
| `actor/entropy` | 0.3249 | 0.3287 | 0.3176 |
| `actor/pg_loss` | 0.0440 | 0.0438 | 0.0420 |
| `response_length/mean` | 198.90 | 199.07 | 198.73 |

逐 step 差的量级（reward max|Δ| 0.055–0.070、entropy 0.026–0.046）与**同行为两次运行之间**（dbase vs patched-2.0）的差（0.086／0.045）同阶甚至更小，即改 poll 没有引入可辨认的数据/训练差异。生成侧本身跨运行非确定，这不等于逐位相同。

## 判决

1. **固定批配置的结论在 DAPO 过滤下方向成立、幅度更大**：poll 2.0→0.1 使 gen 从 8.47s 降到 7.29s（−14.0%）、整步 26.53→25.25s（−4.9%）。比固定批 2 卡的 −3.2% 大一档，因为 DAPO 每步要跑 3–4 轮派发—等待，每轮末尾都被量子吸掉一截。**幅度按 ±0.4s 读**：2.0 臂三跑整步跨跑极差 0.846s，单次运行可偏离臂均值约 ±0.4s；方向（−1.29s）大于自身极差，但两跑均值不足以当精确标定。
2. **但这个旋钮在 DAPO 下不是纯"免费"的**：同样的改动让每步多生成约 13% 的组（丢弃 surplus +39%），因为轮询变细会更早补抽、更晚看到停止条件。**墙钟仍净降**，两份账能用同一个回归对齐（见下）——把每步 gen／step 对本步生成组数做共用斜率回归，斜率 **46.0 ms/组**，poll 只平移截距：0.1 相对 patched 2.0 的固定等待是 **−1.87s（gen）／−1.95s（step）**，而多生成的 15.4 组折算 **+0.707s**，两者相加 −1.238s 与实测 −1.288s 残差 0.05s。也就是说 **−4.9% 已经是扣掉多生成代价后的净收益**，但两项同量级，**换到大批量/更紧的 GPU 预算下这个净收益会被压缩**。
3. **0.5 与 0.1 在本配置里等价**（gen 7.21 vs 7.29，整步 25.43 vs 25.25），与固定批 4 卡"0.5 不够"不同——DAPO 的 gen 跨度更长（单步 8–10s），0.5 的量子相对影响小。若要给推荐值，仍是 0.1（跨形态都验证过），但可以注明"DAPO 配方下 0.5 也可接受"。
4. **对 TM-008 补丁的含义**：补丁的形态不用改（默认 2.0、显式 opt-in），但 PR 描述不能只写"省等待"——在 DAPO 过滤配方上它会同时抬高投机生成量。这条要写进 PR 的风险段，而不是等 review 时被指出来。
5. **顺带读数（不属于补丁结论）**：DAPO + `max_inflight_gen_batches=2` 在本配置下每步生成约 1.3–1.6 倍于理论最小需求的组（过滤率 62%、`rollout.n=4`、准确率约 0.34），有效样本占比只有 24–28%。这比 poll 量子大一个量级的成本项，是比"调旋钮"更值得优化的地方；但改派发策略属设计变更，不在本实验范围。

## 边界

- **单栈单点**：1.5B／gsm8k 短答案／单 seed／15 步／`max_inflight_gen_batches=2`／`n=4`。过滤率与生成量都由准确率决定，换任务或换模型会改变两臂的对比基数。
- **只测 sync V1 trainer**：`ReplayBufferAsync`（colocate/separate async）共用 `_wait_for_next_poll`，但派发循环在 `wait_for_sampleable` 里另有分支，**未测**。
- **`max_inflight_gen_batches=1`（DAPO 默认值）未测**：本实验固定在 2（管线更满、surplus 更明显）。1 条件下投机空间更小，13% 的生成量差可能消失——这正是下一步该补的一组。
- **未测 `max_off_policy_threshold`／陈旧度逐出**：本配置没有触发 stale 逐出（`stale_uids` 恒空），所以"轮询改变可选集合"的另一条通道（越界组被作废）没有覆盖。
- **13% 生成量差是 2 SE 级**：重复两次方向一致，但不是精确标定。
- **代码面只对 `24f25b03` 树负责**：上游 `origin/main` 已到 `4b26c788`，`v1/` 重构出 `trainer_sync.py`。M1 已复核 `poll_interval: float = 2.0` 与 `_wait_for_next_poll` 的 `time.sleep` 仍在，但 TM-008 补丁后两个 hunk 的上下文已变；**本批测的 DAPO 派发循环在新结构上未复核**。
- 未做任何上游写操作；`filter_groups.metric` 的文档不一致单独记录，不据此提 issue。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-poll-dapo-{dbase,dp200,dp050,dp010,dp200b,dp010b}.txt` | 六份训练日志（15 步、exit 0），`.txt` 后缀避开仓库 `.gitignore` 的 `*.log`；md5 见 `md5sums.txt`，与服务器逐文件一致 |
| `summary.json` | 由 `make-summary-dapo.py` 从 raw 重算：逐运行相位均值/sd、过滤与生成量、相对对照差值、数值指标 |
| `run-poll-dapo.sh` | 运行脚本（`SRC`／`MODE`／`POLL`／`INFLIGHT`／`GPUS` 可覆盖） |
| `make-summary-dapo.py` | 日志解析与汇总 |
| `verification.md` | **另一会话的独立只读复核（2026-11-02 15:0x UTC）**：换一条解析路径重算（summary 全部字段 1e-9 内一致）、核对账目完备性（`off_policy`／`rollout_failure` 逐出通道为零）、并用共用斜率回归把"省等待"与"多生成"分开（46.0 ms/组，固定等待 −1.87／−1.95s，两笔账残差 0.05s） |

## 复现

```bash
# 服务器：原树 + patched 副本（TM-008 已建好）
cp -a /home/zxw/venvs/rollout-g8-src/verl /tmp/m5-poll-ab/verl
python3 experiments/verl-v1-poll-quantum/apply_poll_patch.py /tmp/m5-poll-ab/verl
MODE=dbase SRC=/home/zxw/venvs/rollout-g8-src/verl bash run-poll-dapo.sh
MODE=dp200 SRC=/tmp/m5-poll-ab/verl POLL=2.0 bash run-poll-dapo.sh
MODE=dp010 SRC=/tmp/m5-poll-ab/verl POLL=0.1 bash run-poll-dapo.sh
# 本地
python3 make-summary-dapo.py raw/
```
