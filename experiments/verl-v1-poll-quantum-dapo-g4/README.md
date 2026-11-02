# verl V1 DAPO 组过滤下的采样轮询：4 卡（2+2）锚点（A/B 未完成）

2026-11-02 执行（M5 算力侧，TM-008 边界实验的 4 卡补充）。2 卡 DAPO 的六跑 A/B 见 [verl-v1-poll-quantum-dapo](../verl-v1-poll-quantum-dapo/README.md)（独立复核见其 [verification.md](../verl-v1-poll-quantum-dapo/verification.md)）；固定批 4 卡的 A/B 见 [verl-v1-poll-quantum](../verl-v1-poll-quantum/README.md)；机制背景见主文档 §13.44／§13.48。

## 目的

TM-008 已有的 DAPO 数据全部是 2 卡：poll 2.0→0.1 使 gen 8.474→7.289s、整步 26.534→25.246s（−4.9%），代价是每步多生成约 13% 的组。固定批配方在 4 卡上只有 −2.0%（gen 2.161→1.763s）。**DAPO 的收益在多卡下是保留、放大还是被压缩，没有读数。** 本目录原计划补一组 4 卡（2+2）A/B；实际只完成**默认 2.0 基线**，第二臂因与并行会话抢卡未完成（下方执行账）。

## 配置

- **机器**：4090-1；4 卡 = GPU 0,1,4,5（**2+2 放置**，按 §13.42／§13.45：同 fabric 两两成对，SHM all-reduce 带宽最优）。窗口内 GPU 2 由 liyaowei embed 常驻（21728 MiB）。
- **栈**：verl `0.10.0.dev`（`verl-24f25b03` 树，与 2 卡 DAPO 同一棵原树）＋ vLLM 0.24.0＋torch 2.11.0+cu129；`NCCL_P2P_DISABLE=1`、`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`。
- **任务配置**：与 2 卡 DAPO 主组逐项相同——`train_batch_size=32` prompts × `rollout.n=4`、`ppo_mini_batch_size=16`、micro 1、max_prompt 512／max_response 256、bf16、sdpa、`optimizer_offload=True`、`gpu_memory_utilization=0.20`、15 步；`algorithm.filter_groups.enable=True`、`metric=acc`、`max_inflight_gen_batches=2`。
- **变量**：`trainer.v1.sampler.poll_interval`，默认 2.0（原树）vs 0.1（patched 树）。**0.1 臂未完成。**

## 执行账（全部为 UTC）

| 运行 | 树／poll | 窗口 | 结果 |
|---|---|---|---|
| `g4base` | 原树／默认 2.0 | 15:03:56–15:11:23 | **15/15，exit 0，已归档** |
| `g4p010`（第一次） | patched／0.1 | 15:11:27 起 | **本会话主动 kill**：启动时与并行会话新开的 2 卡 `inflight=1` 基线在 GPU 0,1 撞车，主动让路 |
| `g4base2` | 原树／默认 2.0 | 15:29:45–15:32 | **失败**：vLLM `No available memory for the cache blocks`（同容器内其他进程在 profiling 期间释放显存） |
| `g4p0102` | patched／0.1 | 15:32:06–15:34:03 | **失败**：同一错误；上一臂的残留 worker 仍占着 0,1（约 15.7 GB），第二臂在脏状态上启动 |
| `g4base3` | 原树／默认 2.0 | 15:36:14–约 15:41 | **失败**：`Error in memory profiling. Initial free memory 15.4 GiB, current free memory 17.6 GiB`（并行会话的 `di1basec` 同期在 0,1 上启动） |

15:42 起本会话不再抢卡，A/B 推迟到服务器空闲窗口；归档的只有 `g4base`。三次失败的完整堆栈留在服务器 `/tmp/rollout-poll-dapo-g4base2.log`／`g4p0102.log`／`g4base3.log`，未复制入库（内容基本是 vLLM 栈）。

## 结果：4 卡 DAPO 基线（稳态 steps 3–14，n=12）

| 相位 | 均值 (s) | sd (s) | 占整步 |
|---|---|---|---|
| **gen** | **6.520** | 0.912 | 35.2% |
| old_log_prob | 1.833 | 0.126 | 9.9% |
| ref | 1.359 | 0.047 | 7.3% |
| adv | 0.113 | 0.004 | 0.6% |
| update_actor | 6.278 | 0.050 | 33.9% |
| update_weights | 2.346 | 0.074 | 12.7% |
| **整步** | **18.518** | 0.779 | — |

相位和 18.449，残差 0.070。过滤账：每步选 32 组、生成约 **114.8** 组（过滤率 61.3%、有效率 27.9%），与 2 卡运行的 116.2 组一致——**同配置下每步的生成量不随卡数变化**。

### 与 2 卡 DAPO（默认 2.0）和 4 卡固定批的对比

| | 2 卡 DAPO | 4 卡 DAPO | 2→4 卡 |
|---|---|---|---|
| 整步 | 26.511s | 18.518s | **1.432×** |
| gen | 8.489s | 6.520s | 1.302× |
| update_actor | 10.756s | 6.278s | **1.713×** |
| old_log_prob | 2.467s | 1.833s | 1.346× |
| ref | 2.308s | 1.359s | 1.699× |
| update_weights | 2.290s | 2.346s | **0.976×** |
| 每步生成组 | 116.2 | 114.8 | 1.012× |

- **gen 是弱扩展项**：同样的生成量，4 卡只快 1.30×（每卡轨迹数从 64 降到 32）。说明 DAPO 的 gen 在这套配置下不是算力受限，而是逐请求长尾／调度受限；**卡数越多，gen 占整步比例越高**（32.0%→35.2%）。这抬高了"采样等待量子"的相对权重。
- **update_weights 不扩展**（0.976×，与 §13.41 的 replica-bound 结论一致）；`update_actor` 是唯一接近线性的一项（1.71×）。
- **预测（待 A/B 判）**：若 poll 量子省下的是**每步固定量**（2 卡实测 −1.95s），它在 4 卡上的相对收益应为 −1.9/18.5 ≈ **−10%**，比 2 卡的 −4.9% 更大；但固定批 4 卡只测到 −2.0%，两种口径相反。哪一侧成立取决于 DAPO 每步的"派发—等待"轮数是否随卡数变化——这是下一组实验要直接回答的。

## 边界

- **基线尾段有并发重叠**：`g4base` 的 steps 9–14 与并行会话 `di1base` 的启动窗口重叠。分段看：steps 3–8 gen/组 55.2ms、steps 9–14 58.7ms（+6%），整步 18.832 vs 18.205s。重叠方向会让基线**偏慢**，故 18.518s 应视为上界而非精确值；两段生成量也不同（125.0 vs 104.7 组），无法把 +6% 全归给并发。**A/B 必须在同一个干净窗口内重跑两臂。**
- 单栈单点：1.5B／gsm8k 短答案／单 seed／15 步／`n=4`／`inflight=2`；未测 `inflight=1`（2 卡侧另有一组）、async `wait_for_sampleable`、stale 逐出。
- 4 卡下 `gpu_memory_utilization=0.20` 的 KV cache 配额**很薄**（1.5B 权重约 3 GB，预算 4.8 GB），任何同容器进程在 vLLM profiling 期间增减显存都会触发上面的两个错误。见下。

## 复现

```bash
# 服务器：原树 + patched 副本（TM-008 已建好）
MODE=g4base GPUS=0,1,4,5 bash run-poll-dapo.sh
MODE=g4p010 GPUS=0,1,4,5 POLL=0.1 SRC=/tmp/m5-poll-ab/verl bash run-poll-dapo.sh
# 本地
python3 make-summary-g4.py raw/
```

## 操作教训（下一次跑 4 卡 A/B 前先读）

1. **两臂不要用同一个 wrapper 连跑**：第一臂失败时残留的 Ray worker 会继续占着 GPU（本次 0,1 上约 15.7 GB），第二臂在脏状态上必然二次失败。臂间必须核 `nvidia-smi` 回到 ~1 MiB 并清掉 `ray::WorkerDict`／`VLLM::Worker` 残留进程。
2. **launch 前除了看显存，还要看有没有别的会话正在这组卡上跑**：本次三次失败全部由并行会话的 2 卡任务在 0,1 上启动/退出触发。判断方式是 `ps` 查 `main_ppo` 与 `ls -t /tmp/*.log` 看有没有新日志，而不只是 `nvidia-smi` 的瞬时占用。
3. **0.20 的 KV 预算在 4 卡下没有余量**：如果要和同机任务共存，应单独提高 `gpu_memory_utilization`（并接受与 2 卡口径不完全可比），否则就等独占窗口。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-poll-dapo-g4base.txt` | 4 卡 DAPO 默认 2.0 基线训练日志（15 步，exit 0）；`.txt` 后缀避开 `.gitignore` 的 `*.log` |
| `make-summary-g4.py` | 解析与汇总（两臂脚本已就位，待 0.1 臂补齐后直接重跑） |
| `run-poll-dapo.sh` | 运行脚本（`GPUS`／`MODE`／`POLL`／`SRC`／`INFLIGHT` 可覆盖） |
| `sha256sums.txt` | 本目录文件的 sha256 |

## 下一步

在服务器 0,1,4,5 全空且无并行会话任务的窗口内，按"臂间隔清理"重跑 `g4base` 与 `g4p010` 各一次（最好再加一次 0.5 以判 DAPO 在 4 卡上是否仍"0.5≈0.1"），补 `summary.json` 与本文结果表。
