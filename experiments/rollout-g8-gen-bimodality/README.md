# gen 二态隔离：2 副本逐请求计时＋V1 trainer 相位＋轮询间隔干预

2026-11-02 执行（M5 算力侧第五十批）。主文档 [§13.44](../../notebook/rollout-performance-research.md#1344-gen-二态的机制是-v1-sync-trainer-的-20s-采样轮询量子逐引擎计时-30-步都是-6464生成从未变慢2026-11-02-m5-算力侧第五十批)。

## 目的

候选 G（§13.36）与放置效应（§13.42）都留下同一条未隔离观察：2 卡（2 个 rollout 副本）的 `timing_s/gen` 呈二态，4.17s 与 2.15s，比例约 2×，与 NUMA/PCIe 放置无关，而 1 卡恒 4.17s、4 卡恒 2.15s。原记录把它记为「2 副本 × 每副本批量的调度现象」。本实验按第四十九批留下的待办「逐引擎计时」直接测，并追加两轮：

1. **逐引擎/逐请求计时**：每个 LLM 请求的派发时间、完成时间、由哪个 repLica（load-balancer server id）服务；
2. **trainer 相位标记**：V1 sync trainer 的 `gen` 计时块内部拆成 `replay_buffer.sample()` 等待与 `sleep_replicas()`；
3. **因果干预**：把 replay buffer 的 `poll_interval` 从 2.0s 降到 0.1s 重跑。

## 配置

- **repo**：verl `0.10.0.dev`（`verl-24f25b03.tar.gz` 解包树，`/home/zxw/venvs/rollout-g8-src/verl`），vLLM 0.24.0，torch 2.11.0+cu129
- **model / data**：Qwen2.5-1.5B-Instruct，gsm8k `train_sub480` / `test_sub120`
- **负载**：与候选 G 的 g2 完全一致——`train_batch_size=32` prompts × `rollout.n=4` = 128 samples/step，2 卡（GPU 0,1，均 NUMA0），`ppo_mini_batch_size=16`、micro 1、max_prompt 512 / max_response 256、bf16、sdpa、`optimizer_offload=True`、`update_weights_bucket_megabytes=256`、seed 42
- **共同环境**：`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（缺它会触发已知的 NCCL CUDA 217，见「失败与修复」）、`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`ROLLOUT_GPU_MEM_UTIL=0.20`

## 插桩（本地、可回滚）

`instrument-verl-patch.py` 对 5 个文件做**纯追加**修改，只在环境变量设置时写 JSONL；未设置时每处只多一次 `os.environ.get`。运行结束后已 `revert`，5 个文件的 md5 与备份逐一相符，全树 `grep _m5_log` 为空。

| 文件 | 插入 | 记录 |
|---|---|---|
| `verl/workers/rollout/llm_server.py` | `LLMServerClient.generate` 前后 | `m5req`：`t0/ta/t1`（monotonic）、`sid`（LB server id＝副本）、`rid`、`np`（prompt tokens）、`nout`、`stop`、`pre` |
| `verl/trainer/ppo/v1/trainer_base.py` | `_step_once` 的 `gen` 计时块、`prepare_step` | `m5v1`：`gen_begin`/`sample_end`/`gen_end`、`prepare_begin/end` |
| `verl/trainer/ppo/v1/trainer_sync.py` | `on_sample_end`（sleep）、`on_step_end`（update_weights） | `m5v1`：`sleep_begin/end`、`update_weights_begin/end` |
| `verl/trainer/ppo/v1/replay_buffer.py` | `__init__` 末尾 | `M5_REPLAY_POLL_INTERVAL` 可覆盖 `poll_interval`（仅干预轮使用） |
| `verl/trainer/ppo/ray_trainer.py` | 图省事误打在旧 trainer 上 | 未生效（实际执行的是 V1 trainer），保留在补丁脚本里作为反例 |

记录按 pid 分文件（8 个 AgentLoopWorker + 1 个 driver），`wall`(time.time) 与 `t0/t1`(monotonic) 同时存在；解析时用 `median(wall − t1)` 把标记时间换算到 monotonic 域（CLOCK_MONOTONIC 在 Linux 上系统级一致）。**V1 trainer 用的是 `verl/trainer/ppo/v1/trainer_base.py`，不是 `verl/trainer/ppo/ray_trainer.py`**——第一版相位标记打错文件，运行 2 起才生效。

## 执行账

| 运行 | 目的 | GPU | 窗口（UTC） | exit | 步数 |
|---|---|---|---|---|---|
| g2t 首试 | 插桩后首次启动 | 0,1 | 10:01:49–10:02:59 | 1 | 0（NCCL CUDA 217，缺 `NOSET=1`） |
| g2t | 逐请求计时（30 步） | 0,1 | 10:04:57–10:18:35 | 0 | 30/30 |
| g2t2 | V1 相位标记，poll=2.0s（默认） | 0,1 | 10:19:28–10:27:53 | 0 | 15/15 |
| g2t3 | 因果干预，poll=0.1s | 0,1 | 10:29:44–10:37:50 | 0 | 15/15 |

三次运行结束时 GPU 释放回 1MiB；最终 checkpoint（约 9.1G）逐轮删除，避免 `/tmp` 打满。

## 结果一：生成侧没有任何二态（运行 1，30 步）

按 `m5req` 的 `wall` 间隔（>3s）把请求聚成「生成轮」，31 轮中有 29 轮是 128 个请求的干净训练批（首轮 248 个把步骤 1 与初始验证混在一起，末轮 120 个是终评）：

| 量 | 观测 |
|---|---|
| 每轮副本分配 | **29/29 个干净训练轮精确 64/64**（无 128/0、无 65/63） |
| 派发时间窗 | `max(t0) − min(t0)` ≤ 0.09s |
| 单请求时长 p50 | 1.14–1.54s |
| 整轮生成跨度 | 1.49–2.26s（**无任何 4s 级轮**） |
| 输出长度 | 均值 193–235 tokens（max 256） |

即：慢态不是「一个副本吃了整批」、不是「两副本串行」、也不是「生成变慢」。生成侧在每一轮都是同样的形状。

## 结果二：慢态多出的 2.0s 全在 `sample` 等待（运行 2，15 步，poll=2.0s）

`timing_s/gen` 在 V1 sync trainer 里 = `on_sample_begin()` → `replay_buffer.sample(...)` → `on_sample_end()`（后者调 `sleep_replicas`）。15 步全部同时拿到请求与相位：

| 状态 | n | `wait`=sample 等待 | `sleep` | gen 计时块 | 生成跨度 | 副本 A 跨度 | 副本 B 跨度 |
|---|---|---|---|---|---|---|---|
| 慢 | 7 | **4.045s** | 0.128s | 4.17s | 2.078s | 1.946s | 2.048s |
| 快 | 8 | **2.033s** | 0.124s | 2.16s | 1.714s | 1.631s | 1.660s |

- 慢/快两态里 `sleep` 都只有 0.12s，不是它；
- 请求在 `gen_begin` 后 0.011–0.015s 就派发（`dispatch_offset`），不是派发延迟；
- **慢态的 `wait` 正好是多一个 2.012s 量子**，而生成跨度只是「刚过 2.0s」——7 个慢步的生成跨度 2.021–2.122s 全部 >2.0s，8 个快步 1.517–1.946s 全部 <2.0s，**15/15 完全分离**。

## 结果三：机制 = V1 replay buffer 的 2.0s 轮询量子

源码锚点（`verl/trainer/ppo/v1/replay_buffer.py`，固定 `24f25b03` 树）：

- `ReplayBuffer.__init__(..., poll_interval: float = 2.0, ...)`（约 138 行）；
- `wait_for_sampleable()` 的 `while True:` 先做一次快照检查，不够就 `_wait_for_next_poll()` → `time.sleep(self.poll_interval)`（约 391–392 行）；
- 因此 `wait = ceil(就绪时间 / 2.0) × 2.0`（含少量元数据/物化开销）。

生成就绪时间（≈整轮跨度 + 几十毫秒后处理）恰在 2.0s 附近波动，于是「越过 2.0s 边界」近似随机，得到约 50/50 的二态：运行 2 是 7/15，历史 g2 是 7/12、g2n 是 5/12。用运行 1 的 29 个干净训练轮复算这一判据（快 iff 生成跨度 < 2.0s）：**28/29 一致**，唯一例外是跨度 1.98s 的边界步——加几十毫秒后处理就越过 2.0s。同一条规则也解释候选 G 的其余模式：

| 模式 | 每副本序列数 | 生成跨度 | 量子化后的 gen | 观测 |
|---|---|---|---|---|
| g4 | 32 | 1.5–1.7s | 2.0s | 恒 2.155s |
| g2 | 64 | 1.5–2.2s | 2.0 或 4.0s | 2.15/4.18 二态 |
| g1 | 128 | ~4.0s | 4.0s | 恒 4.173s |
| g2w | 128 | ~4.0s | 4.0s | 恒 4.187s |
| g4w | 128 | ~4.0–4.2s | 4.0 或 6.0s | 4.21/6.23 二态 |

## 结果四：因果干预（运行 3，poll=0.1s）

唯一改动 `M5_REPLAY_POLL_INTERVAL=0.1`，其余配置/数据/seed 相同。稳态窗口 steps 2–14（n=13）：

| 阶段 | poll=2.0s | poll=0.1s | Δ |
|---|---|---|---|
| **gen** | 3.243 ± 1.046s | **2.159 ± 0.209s** | **−1.085s** |
| **整步** | 21.925 ± 1.621s | **20.776 ± 0.905s** | **−1.148s** |
| old_log_prob | 2.729 | 2.709 | −0.020 |
| ref | 2.433 | 2.421 | −0.012 |
| update_actor | 11.035 | 11.034 | −0.001 |
| update_weights | 2.297 | 2.290 | −0.007 |

- `gen` 的 4.1x 磁道消失：最大 4.19s → 2.40s，>3s 的步 7/13 → 0/13，sd 1.046 → 0.209；
- 其余五个阶段变化全部 ≤0.02s（远小于步间 sd），说明干预只动了等待，没有动任何计算/通信；
- 整步平均 −1.148s ≈ `gen` 的 −1.085s（另 0.06s 来自 `gen` 之外的少量抖动）。

## 判决

1. **gen 二态不是生成侧现象，也不是副本调度现象**：30 步逐请求计时里 29 个可分离训练轮全部 64/64、派发窗 ≤0.09s、生成跨度 1.5–2.2s，没有任何一步的生成变慢或串行。
2. **它是 V1 sync trainer 的 2.0s 采样轮询量子**：`timing_s/gen` 把 `replay_buffer.sample()` 的轮询等待算进「生成阶段」，等待按 2.0s 取整；生成就绪时间恰在 2.0s 附近，于是出现 2× 的二态。降到 0.1s 后二态消失、整步少 1.15s、其余阶段不变。
3. **对旧结论的修正**：§13.36「g2 的 gen 二态是 2 副本 × 每副本批量的调度现象」应改为「每副本批量决定生成跨度（64 序列≈2.0s、32 序列≈1.6s），生成跨度与 2.0s 轮询边界的相对位置决定测得的二态」。§13.42 按 gen 二态分层的对照仍有效（分层只是把等待量分开），但不能把它读成生成侧机制；g4w 的 4.21/6.23 二态同理是 4.0s 边界上的量子化。
4. **方法学**：跨系统比较 rollout/生成阶段时，必须区分「引擎工作时间」与「训练侧采样等待」；阶段计时把同步/轮询算进去，会制造与设备无关的双峰。这与 §21 的 C1/C3 计时口径警告同向，并给 §13.36/§13.42 的 gen 读数加了适用条件。

## 复现

```bash
# 服务器：/tmp/instrument-verl-patch.py（本目录同名文件）
python3 instrument-verl-patch.py apply        # 备份到 /tmp/m5-gen-timing-backup 并打补丁
bash run-g2-instrumented.sh                   # 运行 1：MODE=g2t，30 步（TOTAL_EPOCHS=2）
MODE=g2t2 TOTAL_EPOCHS=1 bash run-g2-instrumented.sh    # 运行 2：相位标记，poll=2.0
MODE=g2t3 TOTAL_EPOCHS=1 M5_REPLAY_POLL_INTERVAL=0.1 bash run-g2-instrumented.sh  # 运行 3
python3 instrument-verl-patch.py revert       # 还原并核对 md5
# 本地分析
python3 parse-gen-rounds.py <timing_dir> <train_log>        # 逐轮副本分配/生成跨度
python3 parse-gen-timeline.py <timing_dir> <train_log>      # 相位 + 请求的逐 step 时间线
python3 make-summary.py                                     # 由 raw/ 重新生成 summary.json
```

## 验证范围与边界

- 真实 GPU 实测：4090-1，2 卡（GPU 0,1），verl `24f25b03` + vLLM 0.24.0 + Qwen2.5-1.5B + gsm8k 短答案；运行 1 为 30 步、运行 2/3 各 15 步，单 seed。
- **插桩是本地追加，不是上游结论**：`_m5_log` 与 `M5_REPLAY_POLL_INTERVAL` 都是本实验加的，运行后已还原（5 文件 md5 与备份一致）；不能据此说上游「计时有 bug」——`poll_interval` 是公开的构造参数（默认 2.0s），本实验只是把它调小做因果检验。
- **因果干预只改了轮询间隔**：`poll_interval` 影响的是采样等待，不改变数据、策略或权重同步；运行 3 的 `old_log_prob/ref/update_actor/update_weights` 与运行 2 无差异（≤0.02s）支持这一点，但没有做统计显著性检验（单 seed，n=13）。
- **只覆盖 sync V1 trainer 的默认配置**：`ReplayBufferAsync`（async 模式）与自定义 sampler 可能有不同的等待路径；`poll_interval` 是 `ReplayBuffer.__init__` 的默认值，本次未从 yaml 传入（补丁的 env 覆盖仅用于干预轮）。
- 运行 1 的逐轮配对按时间顺序与 `timing_s/gen` 对齐（无相位标记）；首轮混入初始验证、末轮混入终评，已单独标注，不进入判据统计。
- 未做 NCCL/GPU 计时，不涉及放置效应；本实验不动 §13.42 的 SHM 带宽结论。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-g8-gen-bimodality-g2t*.txt` | 三份训练日志（30/15/15 步，exit 0） |
| `raw/timing-g2t{,2,3}/*.jsonl.gz` | 逐请求/相位 JSONL（按 pid 分文件，原文件 md5 见 `raw/timing-md5.txt`） |
| `summary.json` | 由 `make-summary.py` 从 raw 重算：逐轮表、逐 step 时间线、干预对照 |
| `instrument-verl-patch.py` | 插桩/还原脚本（含 md5 备份与 status） |
| `run-g2-instrumented.sh` | 三次运行的开关（MODE/TOTAL_EPOCHS/M5_REPLAY_POLL_INTERVAL） |
| `parse-gen-rounds.py` / `parse-gen-timeline.py` / `make-summary.py` | 分析脚本 |

## 失败与修复

- **NCCL CUDA 217（首试，0 步）**：与候选 G 同源——缺 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 时 Ray 给每个 worker 单卡掩码，verl 用物理 id 当 LOCAL_RANK，FSDP `broadcast_coalesced` 触发 `peer access is not supported`。补上该变量后正常，此失败不计入数据。
- **相位标记打错文件（运行 1）**：第一版把 `gen`/`sleep` 标记插在 `verl/trainer/ppo/ray_trainer.py`，实际执行的是 `verl/trainer/ppo/v1/trainer_base.py`（`TaskRunnerV1`）；运行 1 因此只有逐请求数据。运行 2 起改用 V1 trainer，`_m5_log` 才产出 `m5v1` 记录。运行 1 的逐轮分析不依赖相位标记，结论不受影响。
- **`/tmp` 打满**：运行 1/2/3 各在结束时保存最终 checkpoint（约 9.1G），已逐轮删除；`/home` 仍为 100%。
