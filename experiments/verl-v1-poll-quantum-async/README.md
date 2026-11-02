# verl V1 colocate_async 的采样等待：2.0s 轮询量子同样生效（TM-008 async 分支）

2026-11-03 执行（M5 算力侧，TM-008 未测清单中的「async trainer `wait_for_sampleable` 分支」）。同步 trainer 的同栈结论见 [poll-quantum 实验](../verl-v1-poll-quantum/README.md) 与主文档 [§13.44](../../notebook/rollout-performance-research.md#1344-gen-二态的机制是-v1-sync-trainer-的-20s-采样轮询量子逐引擎计时-30-步都是-6464生成从未变慢2026-11-02-m5-算力侧第五十批)、[§13.49](../../notebook/rollout-performance-research.md)。

## 目的

TM-008 的补丁把 `ReplayBuffer.poll_interval` 接到配置上。此前只在 **sync** trainer 上量过收益（2 卡 −3.2%、4 卡 −2.0%）；上游 `ReplayBufferAsync.wait_for_sampleable` 与 `ReplayBuffer.sample` **共用同一个 `_wait_for_next_poll` → `time.sleep(self.poll_interval)`**，因此 async 模式是否也吃这个 2.0s 量子、收益是否同量级，是补丁对外时必须写清的适用范围。

本轮回答三个问题：

1. async 的 `timing_s/gen` 里能否看到 2.0s 量子？
2. 把 poll 调到 0.1，整步省多少？
3. 省下来的量能否直接读成「2.0s 被浪费」？——**不能**，见下。

## 上游代码面

| 位置 | 事实 |
|---|---|
| `verl/trainer/ppo/v1/replay_buffer.py` `ReplayBufferAsync.wait_for_sampleable` | 循环调用 `_wait_for_next_poll`，后者 `time.sleep(self.poll_interval)` |
| 同文件 `ReplayBuffer.sample`（sync） | 同一 `_wait_for_next_poll` |
| `trainer_colocate_async.py` | 训练模式：每步 `prepare_step()` 派发一个新 batch；`on_sample_end` 中止并 sleep 副本；`on_step_end` 更新权重后恢复生成 |
| `docs/advance/v1_async_trainer.md` | `timing_s/gen` = 「trainer time spent waiting for the next trainable train-batch」 |

本轮的 patched 树里 **`replay_buffer.py` 与 pristine 树逐字节相同**（`diff -q` rc=0）——轮询代码未改，只加了配置接线，所以这个 A/B 比的是「同一个 sleep 循环的时间量子」。

## 实验配置

- **机器**：4090-1，GPU **4,5**（同一 socket／NUMA1，即 2+0 放置）。当时 GPU 0／2／3／6／7 被 liyaowei 的 4 卡 vLLM＋1 个 EngineCore 占用，只有 1／4／5 空闲；本实验固定用 4,5，两臂同放置，GPU 1 未用。
- **栈**：verl `0.10.0.dev`（`/tmp/m5-poll-ab/verl`，patched 树）＋ vLLM 0.24.0 ＋ torch 2.11.0+cu129；`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`、`NCCL_P2P_DISABLE=1`。
- **补丁**：[poll-interval-config.patch](../verl-v1-poll-quantum/poll-interval-config.patch)（3 文件 +8/−1，默认 2.0 不变）。
- **模型/数据**：Qwen2.5-1.5B-Instruct，gsm8k `train_sub480`／`test_sub120`。
- **负载**：`train_batch_size=32` × `rollout.n=4` ＝ 128 samples/step，`ppo_mini_batch_size=16`（每步 2 次 `sample()`）、micro 1、max_prompt 512／max_response 256、bf16、sdpa、`optimizer_offload=True`、seed 42、**15 步**。
- **async 模式**：`trainer.v1.trainer_mode=colocate_async`、`trainer.v1.colocate_async.num_warmup_batches=1`；其余与同步实验完全一致，唯一变量是 `trainer.v1.sampler.poll_interval`。

## 执行账

| 运行 | poll | GPU | 窗口（UTC） | exit | 步数 |
|---|---|---|---|---|---|
| asmoke | 默认 2.0 | 4,5 | 18:01:01–18:04:29 | 0 | 1/1（配置冒烟） |
| a200 | 2.0 | 4,5 | 18:05:02–18:13:18 | 0 | 15/15 |
| a010 | 0.1 | 4,5 | 18:13:19–18:21:41 | 0 | 15/15 |
| a200b | 2.0 | 4,5 | 18:21:42–18:30:04 | 0 | 15/15 |
| a010b | 0.1 | 4,5 | 18:30:05–18:38:13 | 0 | 15/15 |

四跑按 2.0／0.1 交替执行，单跑约 8m20s；每次启动前脚本检查 GPU 4/5 仍为 1 MiB，四跑结束时均释放回 1 MiB。配置解析已验证：两臂日志里分别是 `'poll_interval': 2.0` 与 `'poll_interval': 0.1`。

**收尾回溯说明**：每次运行结束、解释器退出时都会出现一条 `RuntimeError: DataLoader worker (pid …) is killed by signal: Killed.`（`multiprocessing/util.py::_exit_function` → `p.join()`）。它发生在 15/15 步全部记录之后，属退出期清理，不是训练故障；driver exit 0、全部步指标完整。

## 结果（稳态窗口 steps 3–14，n=12）

| 运行 | poll | gen 均值 | gen >2s 步数 | gen 范围 | 整步均值 | 响应长度均值 |
|---|---|---|---|---|---|---|
| a200 | 2.0 | 1.741±0.913 | **9/12** | 0.219–2.255 | 21.401±0.991 | 223.5 |
| a010 | 0.1 | 1.379±0.354 | 0/12 | 0.746–1.870 | 20.752±0.982 | 222.2 |
| a200b | 2.0 | 1.580±0.996 | **8/12** | 0.222–2.330 | 21.024±0.984 | 223.3 |
| a010b | 0.1 | 1.386±0.343 | 0/12 | 0.935–1.888 | 20.867±0.926 | 223.4 |

**池化（两次重复的 run 均值）**：gen 1.660→1.383s（−0.278s，−16.7%，run 级 SE 0.081 ≈ −3.4 SE）；整步 21.212→20.809s（−0.403s，**−1.9%**，run 级 SE 0.197 ≈ −2.0 SE）。单跑整步差为 **−0.649s（−3.0%）与 −0.157s（−0.7%）**，重复间差异接近效应本身，所以整步口径只算「弱到中等证据」，gen 口径是稳定项。

### ① async 里量子确实存在，且暴露面比 sync 2 卡更大

poll=2.0 臂的 gen 是干净的两态：**0.22s** 或 **2.23–2.33s**，中间没有取值；12 个稳态步里 9 步（a200）／8 步（a200b）撞到量子。作为对照，同步 trainer 2 卡是 6/12 步 >3s。两态之差恰为 2.0s 量子（2.245 ≈ 2.0 + 0.245 轮询往返底噪）。

### ② 但 2.245s 不能读成「浪费 2.0s」——0.1 臂测出了真实等待

poll=0.1 臂没有任何一步落在 2.2s 附近，gen 变成 **0.75–1.89s（均值 1.38）**。也就是说，在那些「撞量子」的步里，下游真正需要的等待是 0.7–1.9s，量子只是把它**向上取整**到 2.0（再加一次轮询往返）。量子造成的超额只有 `2.0 − 真实等待`，约 0.1–1.3s/步，而不是 2.0s。

这与 §13.49 在 4 卡同步上的教训同构（那里是「平均等待 ≈ 0.59×poll」，不是满一个量子）。**二态幅度 ≠ 收益**，同一陷阱在 async 上重复出现。

### ③ 整步收益小于同步，且相位差大半来自 gen

池化相位差：gen **−0.278s**、update_actor −0.210s、update_weights +0.082s、其余 ≤0.02s。其中 gen 是机制项；update_actor 的 −0.21s **不是机制项**——单跑看，a200 的 11.683 比 a200b 的 11.420 慢 0.26s，而 a010／a010b 是 11.307／11.376，两个 poll=0.1 臂彼此一致、两个 2.0 臂不一致，且 token 量（`global_seqlen/mean` 20919–20998）差异不到 0.4%，解释不了 1.8% 的差。按「gen 是唯一可归因通道」读，本轮 async 的直接收益是 **−0.28s/步（−1.3%）**，整步 −1.9% 里剩下约 −0.13s 不可归因。

### ④ async 没有消掉等待本身

colocate_async 下 2 卡每步仍有 **1.4–1.7s** 在等 sampleable 组（poll=0.1 的真实等待），与同步 2 卡的 3.2s 同一量级但不同构成（同步等的是整批 32 prompt，async 是每步两次 16 prompt 的小批等待）。「换成 async 就不用等生成」在本规模不成立——等待仍在，量子只是把其中一部分放大成整 2.0s。

## 结论与适用范围

- **补丁的适用面不是「仅 sync」**：colocate_async 的 `wait_for_sampleable` 走同一个轮询循环，2 卡上 8–9/12 步撞量子；调到 0.1 后量子消失，gen −16.7%（run 级 −3.4 SE）。
- **对外表述建议**：2 卡同栈收益 sync −3.2%、colocate_async **−1.9%（≈2 SE，单跑 −0.7%～−3.0%）**；机制上只有 gen 通道可归因（−0.28s/步），不要把整步差全算给轮询。
- **工作量可比**：两臂响应长度 222–224、`global_seqlen/mean` 20919–20998、`response/aborted_ratio=0.0`；async 的 off-policy 指标 `trajectory_spans ≈1.8–1.9`、`staleness ≈0.10–0.19`，属正常重叠训练，不是同轨迹复读。
- **未覆盖**：`separate_async`（需要独立 rollout 池＋非 naive checkpoint backend，且≥4 卡）；4 卡 `inflight=1`（本轮仍只有 3 卡空闲）。**已另测**：stale 逐出路径见 [stale 实验](../verl-v1-poll-quantum-stale/README.md)（threshold=1，整步 −2.1%，§13.53）；`max_inflight_gen_batches>2` 见 [DAPO 补测](../verl-v1-poll-quantum-dapo/README.md)（depth 3，整步 −4.5%，§13.52）。
- **放置口径**：2+0（GPU 4,5 同 socket）。绝对步时不可直接与 §13.45 的 1+1／2+2 锚点混用；本实验只读同窗口 A/B 比值。

## 复现

```bash
# 服务器上（/tmp/m5-poll-ab/verl 为打过 poll-interval-config.patch 的树）
MODE=a200            bash run-async-ab.sh   # poll 默认 2.0
MODE=a010 POLL=0.1   bash run-async-ab.sh   # poll 0.1
MODE=asmoke STEPS=1  bash run-async-ab.sh   # 1 步配置冒烟

# 本地解析与独立校验
python3 make-summary-async.py   # → summary.json
python3 verify.py               # 重算关键读数并核对 summary.json，exit 0
```

## 文件

- `run-async-ab.sh`／`run-chain.sh`：单跑与四跑交替链（含 GPU 占用前置检查）
- `raw/rollout-poll-async-{a200,a010,a200b,a010b}.txt`：四跑原始控制台日志；`raw/md5sums.txt`
- `make-summary-async.py` → `summary.json`（逐跑相位均值、两臂成对比较、池化比较）
- `verify.py`：独立重算轮询值、步数、gen 两态计数与池化差，`exit 0`
