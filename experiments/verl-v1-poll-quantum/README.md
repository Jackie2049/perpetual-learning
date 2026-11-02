# verl V1 采样等待的 2.0s 轮询量子：`poll_interval` 不可配置 + 同栈 A/B

2026-11-02 执行（M5 算力侧，TM-008）。背景机制见主文档 [§13.44](../../notebook/rollout-performance-research.md#1344-gen-二态的机制是-v1-sync-trainer-的-20s-采样轮询量子逐引擎计时-30-步都是-6464生成从未变慢2026-11-02-m5-算力侧第五十批)；上游代码核对基于 `verl-project/verl` main `3fdf2b548882a8aeccf0842c2150d9c9f1790a55`（2026-11-02T11:38:17Z）。

## 目的

§13.44 已证明：`timing_s/gen` 在 V1 sync trainer 里的慢态不是生成变慢，而是 `ReplayBuffer.sample()` 的轮询等待——`_wait_for_next_poll` 睡 `poll_interval`（默认 **2.0s**），生成就绪时间恰好跨过 2.0s 边界时多等一个量子，2 卡上整步多约 1.1s。

本轮回答三个问题：

1. 这个参数能不能从配置改？——**不能**（生产构造路径不传、`trainer.v1.sampler` 无此键、文档也未列）。
2. 把它调小，同窗口 A/B 到底省多少？——2 卡与 4 卡各测一组。
3. 收益是否只在 2 卡成立？——4 卡（2+2 放置）单独测。

## 上游代码面

| 位置 | 现状 |
|---|---|
| `verl/trainer/ppo/v1/replay_buffer.py:138` | `poll_interval: float = 2.0` 是构造参数 |
| 同文件 `:408` `_wait_for_next_poll` | `time.sleep(self.poll_interval)`，`ReplayBuffer.sample` 与 `ReplayBufferAsync.wait_for_sampleable` 共用 |
| `verl/trainer/ppo/v1/trainer_base.py:171-176` | `_build_replay_buffer` 构造 `replay_buffer_kwargs` 时**没有** `poll_interval`，因此永远取默认 2.0 |
| `verl/trainer/config/ppo_trainer.yaml:292-318` | `trainer.v1.sampler` 只有 `max_off_policy_threshold`／`max_off_policy_strategy`／`sync_refill_failed_groups`／`custom_sampler`／`sampler_kwargs`；`sampler_kwargs` 是传给自定义 sampler 的字典，不会展开成构造参数 |
| `docs/advance/v1_async_trainer.md:69-70` | 参数表只列 `max_off_policy_*`，没有 `poll_interval` |
| `transfer_queue` 0.1.9.dev0 | 没有阻塞式等待接口（无 `kv_wait` 一类），`kv_list()` 是 ZMQ 往返；只能靠轮询 |

结论：**内置 replay buffer 的采样等待间隔是死值 2.0s**，用户只能写一个自定义 sampler 子类才能改。与 TM-005／TM-006 同族（配置项不接线），区别是这里的默认值本身有可测代价，而不是开关完全无效。

## 补丁（本地，未提交上游）

`poll-interval-config.patch`，三个文件、共 +8/-1：

1. `ppo_trainer.yaml`：`trainer.v1.sampler.poll_interval: 2.0`（默认不变，新增可达性）；
2. `trainer_base.py`：非自定义 sampler 分支把该值传进 `ReplayBuffer`；
3. `trainer_separate_async.py`：`_step_had_idle` 的阈值改为 `max(poll_interval, 0.5)`——否则把间隔调小后，任何短等待都会被判成 idle，改变 hybrid rollout 的切换策略。

**默认 2.0 不变 ⇒ 本补丁本身不改变任何现有行为**；调小是显式 opt-in。这是为“提交前先不引入行为变化”选的保守形态；若维护者愿意改默认，0.1 的收益见下。

## 实验配置

- **机器**：4090-1；2 卡 = GPU 0,1（同 NUMA）；4 卡 = GPU 0,1,4,5（**2+2 放置**，按 §13.42／§13.45）
- **栈**：verl `0.10.0.dev`（`verl-24f25b03` 树）＋ vLLM 0.24.0 ＋ torch 2.11.0+cu129；`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`、`NCCL_P2P_DISABLE=1`
- **模型/数据**：Qwen2.5-1.5B-Instruct，gsm8k `train_sub480`／`test_sub120`
- **负载**：与候选 G 的 g2／g4 一致——`train_batch_size=32` prompts × `rollout.n=4` ＝ 128 samples/step，`ppo_mini_batch_size=16`、micro 1、max_prompt 512／max_response 256、bf16、sdpa、`optimizer_offload=True`、seed 42、15 步
- **变量**：只在 patched 树上改 `trainer.v1.sampler.poll_interval`；`pbase`／`p4base` 用未打补丁的原树做基线，`p200` 是“补丁 + 默认值 2.0”的对照（验证补丁本身不改变行为）

## 执行账

| 运行 | 树 | poll | GPU | 窗口（UTC） | exit | 步数 |
|---|---|---|---|---|---|---|
| pbase | 原树 | 默认 2.0 | 0,1 | 13:03:33–13:12:03 | 0 | 15/15 |
| p200 | patched | 2.0 | 0,1 | 13:12:14–13:20:38 | 0 | 15/15 |
| p050 | patched | 0.5 | 0,1 | 13:20:49–13:29:02 | 0 | 15/15 |
| p010 | patched | 0.1 | 0,1 | 13:29:13–13:37:21 | 0 | 15/15 |
| p4base | 原树 | 默认 2.0 | 0,1,4,5 | 13:39:46–13:46:18 | 0 | 15/15 |
| p4p010 | patched | 0.1 | 0,1,4,5 | 13:46:29–13:52:49 | 0 | 15/15 |

六次运行结束时 GPU 均释放回 1 MiB；checkpoint 逐轮删除，`/tmp` 未打满。配置解析已验证：patched 运行的 config dump 里出现 `'poll_interval': 2.0/0.5/0.1`（`pbase`／`p4base` 无此键）。

## 结果（稳态窗口 steps 3–14，n=12）

### 2 卡

| 运行 | gen 均值±sd | >3s | 整步均值±sd | 相对 p200 对照 |
|---|---|---|---|---|
| pbase（原树，默认） | 3.165 ± 1.052 | 6/12 | 21.785 ± 1.671 | — |
| p200（补丁，2.0） | 2.997 ± 1.039 | 5/12 | 21.686 ± 1.515 | — |
| p050（补丁，0.5） | 2.370 ± 0.257 | **0/12** | 21.033 ± 0.934 | **−0.653s（−3.0%）** |
| p010（补丁，0.1） | 2.199 ± 0.234 | **0/12** | 21.001 ± 0.859 | **−0.686s（−3.2%）** |

- **补丁是惰性的**：pbase 与 p200 行为相同，整步只差 −0.5%（二态快慢步配比 6/12 vs 5/12 的随机差异）。
- **二态消失**：0.5 与 0.1 都从“4.17/2.16 二态”变成单一分布（sd 1.05 → 0.26／0.23），>3s 的步从 6/12 降到 0/12。
- **gen 的因果读数**：0.1 比对照少 0.798s（3.0 → 2.2）；整步只少 0.686s，差值来自同窗口后段计算相位漂移（`update_actor` 10.953 → 11.107，序列和时间顺序单调上升；见下）。

### 4 卡（2+2）

| 运行 | gen 均值±sd | >3s | 整步均值±sd | 相对基线 |
|---|---|---|---|---|
| p4base（原树，默认） | 2.161 ± 0.009 | 0/12 | 14.216 ± 0.147 | — |
| p4p010（补丁，0.1） | 1.763 ± 0.037 | 0/12 | 13.934 ± 0.431 | **−0.282s（−2.0%）** |

- 4 卡生成跨度约 1.5–1.7s，**全部被 2.0s 量子向上取整**：基线 gen 恰好钉在 2.16±0.01（不是二态，是恒定多等约 0.4s）。
- 0.1 后 gen 1.763，整步 13.934；基线 14.216 与 §13.45 的 14.173s 相差 0.3%，是跨窗口一致性核对。
- 4 卡的整步收益（−0.28s）小于 gen 收益（−0.40s），因为 `update_actor` 6.363 → 6.464（+0.10s）同向漂移；残差（整步 − 六相位和）在四个 2 卡运行里恒为 0.049–0.050s，两个 4 卡运行 0.040–0.044s。

### 数值一致性

同一 seed（42）下，**两次同行为运行之间**的逐步指标就已经不同——生成侧非确定性（vLLM 调度/kernel）：

| 指标（steps 3–14 均值） | pbase vs p200（同行为） | p200 vs p010（改 poll） |
|---|---|---|
| `critic/rewards/mean` | 0.1161 vs 0.1146（max\|Δ\| 0.086） | 0.1146 vs 0.1120（max\|Δ\| 0.070） |
| `actor/entropy` | 0.3581 vs 0.3552（0.024） | 0.3552 vs 0.3578（0.030） |
| `actor/pg_loss` | 0.00198 vs 0.00596（0.017） | 0.00596 vs 0.00830（0.024） |
| `response_length/mean` | 222.87 vs 222.18（12.2） | 222.18 vs 222.52（17.8） |

即：改 poll 带来的差异**不超过两次同行为运行之间的噪声**。补丁只改等待时机，不改数据/权重计算；本配置下 32 个 prompt 在飞、目标 32，`sample()` 返回的集合与等待时机无关，所以数据路径不变。

## 判决

1. **`poll_interval` 是一个不可达的性能参数**：默认 2.0s、无配置键、无文档、只能自定义 sampler 子类。这与“配置项完全无效”不同，是一个真实存在但用户改不了的旋钮。
2. **默认值有可测代价**：2 卡整步 21.69–21.79 → 21.00s（−0.69s，−3.2%），gen 3.0–3.2 → 2.2s；4 卡 14.22 → 13.93s（−0.28s，−2.0%），gen 2.16 → 1.76s。两种卡数下收益方向一致。
3. **0.5 不是通用解**：2 卡（生成跨度约 2.0–2.2s）用 0.5 就够；4 卡（跨度约 1.6s）用 0.5 仍会在 2.0s 边界被取整，必须更小（0.1–0.2）。若要改默认，0.1 是两种形态下都验证过的值。
4. **轮询开销没有可测惩罚**：p050 与 p010 每步相差约 16 次 `kv_list`，整步差 −0.03 ± 0.26s（统计上不可区分），把每次轮询的成本上界压在几毫秒量级；这是上界，不是直接测量。
5. **补丁形态建议**：先只做配置可达性 + 文档（默认 2.0 不变，零行为变化），把 0.1–0.2 写成推荐值；是否改默认交给维护者。`_step_had_idle` 的阈值必须同时与轮询间隔解耦，否则调小间隔会改变 hybrid rollout 的 idle 判定。

## 验证范围与边界

- 单 seed、单模型（1.5B）、单任务（gsm8k 短答案）、sync V1 trainer；**未测 async trainer 的 `ReplayBufferAsync` 实际行为**（代码共用 `_wait_for_next_poll`，但未跑）。
- 只测了 15 步的稳态窗口；没有测长训练下轮询开销的累积影响，也没有直接测 `kv_list` 的耗时（只给了统计上界）。
- 数值一致性是“差异不超过同行为运行的噪声”，不是逐位相同；生成侧本身跨运行非确定。
- **DAPO 过滤/refill 或 `max_inflight_gen_batches>1` 时，采样时机可能改变可选组集合**（陈旧度/逐出），本实验的固定批配置不覆盖该情形；若采纳该旋钮，需要在带过滤的配方上单独核数据分布。
- 未做任何上游写操作；补丁与证据留在本地，对外动作待用户逐项授权。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-poll-ab-{pbase,p200,p050,p010,p4base,p4p010}.txt` | 六份训练日志（15 步，exit 0，`.txt` 后缀以避开仓库 `.gitignore` 的 `*.log`），md5 见 `md5sums.txt`，与服务器逐文件一致 |
| `summary.json` | 由 `make-summary.py` 从 raw 重算：逐运行相位均值/sd、gen 分布、相对对照差值、数值指标 |
| `run-poll-ab.sh` | 运行脚本（SRC/MODE/POLL/GPUS 可覆盖） |
| `apply_poll_patch.py` | 补丁应用脚本（三处精确替换，各命中一次） |
| `poll-interval-config.patch` | 三个文件的 unified diff（相对路径 a/ b/） |
| `make-summary.py` | 日志解析与汇总 |
| `raw/rollout-poll-main-smoke-main-3fdf2b54.txt` | main `3fdf2b54` 上的 3 步接线冒烟日志（exit 0，非计时测量） |
| `run-main-smoke.sh` | 冒烟运行脚本（含 TransferQueue 显式容量；device-map 适配需在树上打，见下节） |

## 复现

```bash
# 服务器：原树 + patched 副本
cp -a /home/zxw/venvs/rollout-g8-src/verl /tmp/m5-poll-ab/verl
python3 apply_poll_patch.py /tmp/m5-poll-ab/verl
MODE=pbase SRC=/home/zxw/venvs/rollout-g8-src/verl bash run-poll-ab.sh
MODE=p200  SRC=/tmp/m5-poll-ab/verl POLL=2.0 bash run-poll-ab.sh
MODE=p010  SRC=/tmp/m5-poll-ab/verl POLL=0.1 bash run-poll-ab.sh
# 4 卡加 GPUS=0,1,4,5
# 本地
python3 make-summary.py            # 读 raw/，写 summary.json
```

## 补丁在 main `3fdf2b54` 上的运行时冒烟（2026-11-03，M5 第二会话）

补丁此前只在 `24f25b03` 快照树上实跑过；`v1/` 重构出 `trainer_sync.py`／`trainer_colocate_async.py` 后，pending 的说法是"dry-run 3/3 通过，但 DAPO 派发循环在新结构上未实跑复核"。本冒烟只补这一步，**不是效果复现**。

- **树与补丁**：`git clone --depth 1` 取 volcengine/verl main `3fdf2b548882`（2026-11-02T11:38Z），`patch -p1 --fuzz=0` 应用 `poll-interval-config.patch`，3/3 命中、RC=0。
- **运行**：GPU 4,5（2+0），同 §13.48 的 DAPO 配置，`trainer.v1.sampler.poll_interval=0.1`、`max_inflight_gen_batches=2`、`trainer.total_training_steps=3`；**3/3 步完成、exit 0**（20:38:50–20:43:26Z，含启动约 4.6 分钟）。
- **接线证据**：① 解析后的配置 dump 含 `'poll_interval': 0.1`（`trainer.v1.sampler` 段）与 `'max_inflight_gen_batches': 2`；② `_build_replay_buffer` 全树只有 `trainer_base.py:140` 一处定义（`trainer_sync`／`trainer_colocate_async` 不覆写），`trainer_base.py:146/170` 从 `config.trainer.v1.sampler` 取值并传给 `ReplayBuffer`，`replay_buffer.py:157/409` 存下并用于 `time.sleep`；③ 3 步均带 DAPO 逐出（evicted 148／138／98 组）。
- **两处环境适配（不属于补丁，列出以免误读）**：① `++transfer_queue.backend.SimpleStorage.total_storage_size=100000`——main 的 yaml 把 `null` 注释为 unlimited，但服务器 venv 的 TransferQueue 在 `null` 上直接 `TypeError`（`math.ceil(None / n)`），属版本错配；② `verl/single_controller/base/worker.py` 的本地 device-map 适配（与 `24f25b03` 实验树同一段：`NOSET=1` 时 Ray 报物理卡号，需折算成 `CUDA_VISIBLE_DEVICES` 内的 ordinal），否则 GPU 4,5 这种非连续掩码会 `invalid device ordinal`。
- **边界**：3 步只是接线冒烟，不是计时测量，也不重新估计效应量（无对照臂、单跑）；绝对步时不可与 15 步实验比。`3fdf2b54` 上"效应大小是否同量级"仍需完整 A/B。
