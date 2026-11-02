# verl V1 stale 逐出（`max_off_policy_threshold=1`）下的采样轮询 A/B

2026-11-03 执行（M5 算力侧，TM-008 边界实验，第五十七批）。前序：[固定批](../verl-v1-poll-quantum/README.md)／[DAPO 过滤](../verl-v1-poll-quantum-dapo/README.md)／[async 默认阈值](../verl-v1-poll-quantum-async/README.md)；主文档 [§13.53](../../notebook/rollout-performance-research.md)。

## 目的

TM-008 的边界清单里最后一条可以在 2 卡上测的通道：`ReplayBufferAsync` 的 **stale 逐出**。默认 `max_off_policy_threshold=8` 在本规模从未触发（§13.51 的 async 四跑 `stale_uids` 恒空），所以此前所有读数都没有覆盖"轮询改变可选集合"的另一条路径。本实验把阈值降下来、强制打开逐出，回答两件事：

1. 阈值降到多少才会真触发，触发后每步丢多少？
2. 逐出通道打开后，poll 2.0→0.1 的量子收益还在不在，量级多大？

## 配置与执行

- **机器**：4090-1，GPU 4,5（同一 socket＝2+0；liyaowei 的 4 卡 vLLM＋1 个 EngineCore 占 0/2/3/6/7）。
- **栈**：verl `24f25b03` 树＋TM-008 patched 副本（`replay_buffer.py` 与 pristine 逐字节相同，只加 `poll_interval` 配置接线）；vLLM 0.24.0、torch 2.11.0+cu129。
- **任务**：Qwen2.5-1.5B-Instruct、gsm8k `train_sub480`／`test_sub120`、`train_batch_size=32`、`n=4`、max_prompt 512／max_response 256、15 步、`trainer.v1.trainer_mode=colocate_async`＋`num_warmup_batches=1`；**不开** DAPO filter（本实验只看 stale 通道）。
- **两个变量**：`trainer.v1.sampler.max_off_policy_threshold=1`（默认 8）与 `poll_interval`（2.0／0.1）。`max_off_policy_strategy` 保持默认 `drop`。
- **为什么是 1**：先用 threshold=2 做 4 步冒烟（`STEPS=4`），**零逐出**——`trajectory_spans/max` 恰好为 2，`2 > 2` 不成立，说明 warmup=1 的 async 管道自然跨度就是 1–2。降到 1 后，上一步派发的整批变成 span 2 被丢弃，通道才打开。

| 运行 | 窗口（UTC） | poll | gen | 整步 | 逐出组/步 | 生成组/步 |
|---|---|---|---|---|---|---|
| st1p200 | 19:53–20:02 | 2.0 | 3.262±1.03 | 22.055±1.60 | 32.0 | 64.0 |
| st1p010 | 20:02–20:11 | 0.1 | 2.725±0.23 | 22.022±0.84 | 32.0 | 64.0 |
| st1p200b | 20:11–20:20 | 2.0 | 3.750±0.95 | 22.890±1.33 | 32.0 | 64.0 |
| st1p010b | 20:20–20:29 | 0.1 | 2.847±0.30 | 21.973±0.79 | 32.0 | 64.0 |

四条全部 15/15 步、exit 0；配置解析已核对 `'off_policy_threshold': 1` 与 `'poll_interval': 2.0/0.1`。每步 `training/off_policy/evicted_samples=32`、`evicted_samples_staleness/mean=2.0`，被选中的轨迹 span 恒为 1。

## 结果（steps 3–14，n=12）

### 量子仍在，形态和默认 async 一样干净

- poll=2.0 两跑的 gen 都是**严格两态**：快态 2.43／2.47s、慢态 4.43／4.39s，**差 2.00／1.92s**。st1p200 有 **5/12** 步撞量子，st1p200b 有 **8/12** 步。
- poll=0.1 两跑 **0/12** 步进入慢态，gen 落在 2.37–3.36s，逐 step sd 0.23／0.30（2.0 臂是 1.03／0.95）。
- 逐出计数两臂完全相同（32.0/步）——**轮询只改变等待，不改变逐出量**。这符合代码：`global_steps` 在 `sample()` 内固定，越界判定与轮询时机无关。

### 整步收益变小：−0.475s（−2.1%），且被 run 级抖动吃掉一部分

| 相位（池化） | 2.0 | 0.1 | Δ |
|---|---|---|---|
| **gen** | **3.506** | **2.786** | **−0.720（−20.5%）** |
| old_log_prob | 2.701 | 2.713 | +0.012 |
| ref | 2.425 | 2.450 | +0.025 |
| update_actor | 11.181 | 11.239 | +0.058 |
| update_weights | 2.435 | 2.560 | +0.125 |
| adv | 0.110 | 0.112 | +0.002 |
| **整步** | **22.473** | **21.997** | **−0.475（−2.1%）** |

- gen 收益（−0.72s）**没有全额落到整步**：其余相位合计 +0.22s（其中 `update_weights` +0.125，约为其 run 级 SE 的 1.8 倍，**没有机制解释，按噪声处理**）。相位和与整步的残差两臂都是 0.12–0.14s，与 §13.48 同口径。
- **重复性**：2.0 臂两跑整步差 **0.835s**（22.055／22.890），0.1 臂只差 **0.049s**——量子化让 2.0 臂的单跑结果依赖"这跑撞了几次量子"（5 次 vs 8 次），这正是 §13.48 说过的"单跑可偏离臂均值 ±0.4s"。**−0.475s 的臂差小于 2.0 臂自身跨跑极差，只能当方向性证据（≈1 SE），不能当精确标定。**

### 这个配置本身才是大头

- threshold=1 让每一步都丢掉上一步派发的整批（32 组），再补抽 32 组 ⇒ **生成组/步 = 64**，是默认 async（~32–35）的两倍。作为对照，§13.51 的默认 async 2.0 臂 gen 1.58–1.74s、整步 21.0–21.4s；本配置 2.0 臂 gen 3.51s、整步 22.47s——**每步多花约 1.1–1.5s 墙钟换"绝对 on-policy"**（相对 21.0–21.4s 的单跑范围）。
- poll 调小能收回其中约 0.5s（−2.1%），**收不回被丢弃的那 32 组**。想让 async 既省又不丢数据，应该改 `max_off_policy_threshold`／`max_inflight_gen_batches` 的配比，而不是靠轮询补偿。

## 判决（对 TM-008 的含义）

1. **补丁的适用面覆盖 stale 通道**：逐出打开后，poll 0.1 仍能消掉 gen 的两态（0/12 vs 5–8/12），方向与 sync、默认 async 一致。
2. **收益幅度最小的一档**：整步 −2.1%（sync 固定批 −3.2%、DAPO −4.9%、默认 async −1.9%），且 run 级证据只有 ≈1 SE。原因是本配置每步生成量翻倍，量子占整步的比例被摊薄；且 gen 收益有一部分没有传导到整步。
3. **PR 写法**：风险/边界段可以写"在 `max_off_policy_threshold=1` 的严格 async 下，收益仍在但最小（−2.1%，≈1 SE）；该配置本身每步丢弃一整批，轮询不是补偿手段"。

## 边界

- 单栈单点：1.5B／gsm8k 短答案／单 seed／15 步／`n=4`；2+0 放置，绝对步时不可跨窗口比。
- 只测了 threshold=1 与 2（2 是 4 步冒烟、零逐出）；`max_off_policy_strategy=wait`（dropless）与 `separate_async` 未测。
- 每臂两跑，run 级噪声 ±0.4s 量级；整步差 0.475s 小于 2.0 臂跨跑极差 0.835s。
- 未做任何上游写操作。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-poll-stale-{st1p200,st1p010,st1p200b,st1p010b}.txt` | 四份训练日志（15 步、exit 0），md5 见 `md5sums.txt` |
| `summary.json` | 由 `make-summary-stale.py` 从 raw 重算：相位均值/sd、逐出与生成量、spans/staleness |
| `run-stale-ab.sh` | 运行脚本（`MODE`／`POLL`／`THRESHOLD`／`STEPS`／`GPUS` 可覆盖；threshold=2 的冒烟用 `THRESHOLD=2 STEPS=4`） |
| `make-summary-stale.py` | 日志解析与汇总 |

## 复现

```bash
# 服务器（patched 树已由 TM-008 建好）
MODE=st1p200 POLL=2.0 THRESHOLD=1 STEPS=15 GPUS=4,5 bash run-stale-ab.sh
MODE=st1p010 POLL=0.1 THRESHOLD=1 STEPS=15 GPUS=4,5 bash run-stale-ab.sh
# 本地
python3 make-summary-stale.py raw/
```
