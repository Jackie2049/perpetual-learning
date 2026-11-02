# verl V1 separate_async 的采样等待：rollout 池独立且跑在训练前头，2.0s 轮询量子不进关键路径（TM-008 最后一个未测分支）

2026-11-03 执行（M5 算力侧）。同专项的其余模式见 [sync](../verl-v1-poll-quantum/README.md)、[DAPO](../verl-v1-poll-quantum-dapo/README.md)、[4 卡](../verl-v1-poll-quantum-dapo-g4/README.md)、[colocate_async](../verl-v1-poll-quantum-async/README.md)、[stale 通道](../verl-v1-poll-quantum-stale/README.md)。

## 目的

TM-008 的补丁把 `ReplayBuffer.poll_interval` 接到配置（默认 2.0 不变）。补丁的适用面此前覆盖 sync（2 卡 −3.2%、DAPO −3.4%／−4.9%、4 卡 −4.0%）、colocate_async（−1.9%）、stale 逐出通道（−2.1%）与 dropless `wait`（t=1 −5.6%、t=2 留一检验后归零）。**`separate_async` 是清单上最后一个没跑过的 trainer mode**：训练池与 rollout 池分开，rollout 在独立 GPU 上持续生成。

本轮回答两件事：

1. 这个模式下 `timing_s/gen` 里还看不看得到 2.0s 量子？
2. 把 poll 调到 0.1，整步省多少？

答案：**看不到，也省不到**——稳态 `timing_s/gen` ≈ 0.03s，整步差 +0.010s（t=0.12）。原因不是"效应小"，而是**训练步根本不阻塞在采样上**，`_wait_for_next_poll` 的睡眠循环压根没进入。

## 前置条件（本轮新踩到的两处，都不是上游缺陷，但决定能否在本机跑）

上游自带参考脚本 `tests/special_e2e/run_v1_separate_async.sh`（此前"examples 下无现成脚本"只看了 examples）。它自己的校验允许 **2 train + 1 rollout**：训练卡数必须偶数、rollout 卡数必须能被 `rollout.tensor_model_parallel_size` 整除，取 TP=1 即可。`PPOTrainerSeparateAsync.__init__` 的四条硬约束（`train_batch_size == parameter_sync_step × ppo_mini_batch_size`、`rollout.nnodes/n_gpus_per_node > 0`、`checkpoint_engine.backend != "naive"`、reward model 需独立资源池）全部满足。

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: cupy` | `nccl_checkpoint_engine` 在模块加载期 `import cupy`；`~/venvs/rollout-g8` 没装 | `pip install --target /tmp/cupy-pkgs --no-deps cupy-cuda12x fastrlock`（`/home` 只剩 8 G，不写它） |
| 设了 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（此前所有 colocate 实验都这么跑）时报 `NCCL_ERROR_INVALID_USAGE` + `Multiple Ranks are using the same GPU/Partition` | `ray.util.collective` 的 `barrier()` 在 `_used_gpu_indices` 为空时按**本进程可见卡数**建通信器（`range(nccl_util.get_num_gpus())`）。NOSET 下每个 actor 都看到 3 张卡，于是每个 rank 都占 0/1/2 | **本模式不能设 NOSET**（上游 CI 也不设） |
| 去掉 NOSET（Ray 给每 actor 单卡掩码）后，训练侧 FSDP 广播报 `Cuda failure 217 'peer access is not supported'`（`ncclShmImportShareableBuffer`） | 单卡掩码下 NCCL 走 SHM 传输时在本机不可用（与 colocate 实验里 NOSET 的取舍是同一枚硬币的两面） | `NCCL_SHM_DISABLE=1` |

**代价**：训练侧集合通信退化为 socket 传输，`update_actor` 偏慢。因此**本文件的绝对步时不能与其他批次的秒数直接比**；只有本窗口内的 2.0／0.1 A/B 可比。

## 实验配置

- **机器**：4090-1，**训练 GPU 4,5（同一 socket）＋ rollout GPU 1**。GPU 0 被 liyaowei 的 Embedding EngineCore 占（20.8 G），2/3/6/7 是其 4 卡 Qwen3-VL-32B vLLM（各 23.3 G）；本轮前后均只有 1/4/5 空闲，**4 卡窗口仍不具备**。
- **栈**：verl `0.10.0.dev`（`/tmp/m5-main-smoke/verl`＝main `3fdf2b54`＋TM-008 补丁＋本地 device-map 补丁，与 [main 冒烟](../verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话) 同一棵树）＋ vLLM 0.24.0 ＋ torch 2.11.0+cu129；`NCCL_SHM_DISABLE=1`、`NCCL_P2P_DISABLE=1`。
- **模式**：`trainer.v1.trainer_mode=separate_async`、`num_warmup_batches=1`、`parameter_sync_step=4`、`max_off_policy_threshold=8`／`strategy=drop`、`checkpoint_engine.backend=nccl`（bucket 1024 MB）、`transfer_queue.enable=True`。
- **负载**：Qwen2.5-0.5B-Instruct，gsm8k `train_sub480`／`test_sub120`，`train_batch_size=4`（= 4 × `ppo_mini_batch_size=1`）× `rollout.n=4` ＝ 16 条/步，max_prompt 512／max_response 256，bf16、sdpa、**8 步**。
- **唯一变量**：`trainer.v1.sampler.poll_interval` ＝ 2.0 / 0.1。

## 执行账

| 运行 | poll | GPU | 窗口（UTC） | exit | 步数 |
|---|---|---|---|---|---|
| sepasync-smoke | 默认 2.0 | 4,5 + 1 | 23:02:20–23:14:09（含两次失败重试） | 0（2 步） | 2/2 |
| sa20a | 2.0 | 4,5 + 1 | 23:15:41–23:19:32 | 0 | 8/8 |
| sa01a | 0.1 | 4,5 + 1 | 23:19:37–23:23:37 | 0 | 8/8 |
| sa20b | 2.0 | 4,5 + 1 | 23:23:42–23:27:50 | 0 | 8/8 |
| sa01b | 0.1 | 4,5 + 1 | 23:27:55–23:31:37 | 0 | 8/8 |

四跑按 2.0／0.1 交替，每跑现起进程（含 vLLM 与检查点引擎初始化，约 1.5–2 min），跑完 GPU 全部释放回 1 MiB。两臂配置由日志里的解析配置行确认：`'poll_interval': 2.0` / `0.1`、`'trainer_mode': 'separate_async'`、`'parameter_sync_step': 4`、`'backend': 'nccl'`。

**收尾回溯**：每跑结束、解释器退出时同样出现 `RuntimeError: DataLoader worker (pid …) is killed by signal: Killed.`（`multiprocessing/util.py::_exit_function`），发生在 8/8 步记录之后，属退出期清理，driver exit 0。

## 结果（稳态窗口 steps 3–8，n=6/跑）

| 运行 | poll | gen 均值 | gen 范围 | 整步均值 | old_log_prob | update_actor | update_weights | 响应长度 |
|---|---|---|---|---|---|---|---|---|
| sa20a | 2.0 | 0.032±0.000 | 0.032–0.033 | 8.236±0.206 | 3.129 | 3.041 | 1.853 | 233.2 |
| sa01a | 0.1 | 0.037±0.001 | 0.036–0.038 | 8.222±0.250 | 3.103 | 3.001 | 1.866 | 233.8 |
| sa20b | 2.0 | 0.033±0.002 | 0.031–0.035 | 8.186±0.184 | 3.099 | 3.046 | 1.829 | 231.5 |
| sa01b | 0.1 | 0.033±0.001 | 0.031–0.035 | 8.181±0.240 | 3.208 | 2.930 | 1.831 | 236.9 |

**池化**：gen **0.0325 → 0.0347s**（Δ **−0.0022s**，2 毫秒，物理上就是量化误差）；整步 **8.211 → 8.201s**（Δ **+0.010s**，t=+0.12，步级 Welch SE 0.087）⇒ **本配置下 poll 对整步无可检出影响**。

### 读数一：量子在稳态里根本不出现

所有 24 个稳态步（4 跑 × 6 步）的 `timing_s/gen` 都在 **0.031–0.038s**，比 2.0s 量子小两个数量级；两档 poll 的逐步分布完全重叠。对比 colocate_async（§13.51）同字段是 1.4–1.7s、8–9/12 步撞 2.0s 量子——差别的来源是**排队**，不是轮询实现。

### 读数二：唯一的等待在 warmup 步，且与 poll 无关

step 1 的 `timing_s/gen`：sa20a **4.06s**、sa01a **5.73s**、sa20b **6.07s**、sa01b **0.033s**。同一 poll 值的两跑可以差 6 秒（0.1 臂两次 5.73／0.03），说明这段等待取决于启动时 rollout 池跑到哪了，**不是 poll 的函数**。step 2 起队列就见底不上——训练步 8.2s，rollout 池只用约 2s 就能生成下一步的 16 条轨迹。

### 读数三：机制——训练是瓶颈，rollout 跑在它前头

稳态一步 8.21s 的构成：`old_log_prob` 3.1s ＋ `update_actor` 3.0s ＋ `update_weights` 1.85s ＋ `adv` 0.15s ＋ `gen` 0.03s（余 0.06s 为计时缺口）。生成侧每步只要产出 16 条、平均 233 token 的轨迹，单卡 0.5B 远快于 8.2s ⇒ 队列始终非空，`ReplayBufferAsync.sample()` 不会进入 `_wait_for_next_poll()`。**poll 只在"队列空、必须等下一批"时有意义**，而这是 sync／colocate 的常态，是本配置的例外。

### 读数四：off-policy 面也不随 poll 变

`trajectory_spans` 与 `trajectory_staleness` 的均值在 4 跑 × 6 个稳态步上**恒为 1.00／1.00**：每条轨迹只跨一个权重版本、稳定落后一步。这与 colocate 的 stale 实验（poll 越小 span=2 越多）不同——那里 poll 改变派发与更新的相对时序；在 separate_async 里 rollout 池一直领先，派发节奏不受 poll 影响。

## 对 TM-008 的含义

补丁在各模式上实测到的整步差：

| 模式 | poll 2.0→0.1 的整步 |
|---|---|
| sync 2 卡固定批 | −3.2% |
| sync 2 卡 DAPO inflight=1 | −3.4% |
| sync 2 卡 DAPO inflight=2 | −4.9% |
| sync 4 卡 DAPO | −4.0% |
| colocate_async | −1.9% |
| stale 逐出（drop, t=1） | −2.1% |
| dropless wait t=1（屏障） | −5.6% |
| dropless wait t=2（一步重叠） | ≈0（留一检验后归零） |
| **separate_async（本轮）** | **+0.1%（+0.010s，t=0.12）** |

⇒ 收益**不是 trainer mode 的函数，而是"训练步是否阻塞在 `sample()`"的函数**：阻塞越强，量子暴露越多（同步固定批／屏障 wait 最大，一步重叠与 separate_async 归零）。PR 的适用范围段可以据此写：补丁对默认行为零影响，收益只在"队列会空"的配置里出现——而 separate_async 在 rollout 池跑在训练前头时不属此列。

## 边界

- 单栈单点：0.5B／gsm8k／单 seed／8 步（稳态窗口 6 步）／每臂两跑，分辨力约 ±0.2s（步级 SE 0.087 的 2 倍）。
- **绝对步时含降级通信**（`NCCL_SHM_DISABLE=1`），不可与本专项其他批次的秒数比；跨批次只引百分比与结构。
- 本配置处于 **rollout-ahead 区间**（1 张 rollout 卡 vs 2 张训练卡 + 0.5B）。若 rollout 池成为瓶颈（更小池／更大模型／更大 `n`），队列会空、睡眠循环会重新进入，量子可能像 colocate 一样出现——**该区间本批未测**，不要外推。
- 本批 rewards 池化为 0（0.5B 在 256 token 内基本做不出 gsm8k），actor 梯度为 0、训练侧偏轻 ⇒ 更容易 rollout-ahead。这**不影响轮询机制的判读**（gen 侧读数是"等没等"，与奖励无关），但不要让这份读数代表正常训练负载下的整步构成。
- 本版本 `update_weights` 每步都执行（1.8s，`trainer_separate_async.py` 里无 `parameter_sync_step` 门槛），是当前实现的固定开销，不作为机制结论。

## 复核

`verify.py`（独立解析路径：按 `" - "` 切分、只取所需键、重新从解析后的配置行确认 mode／poll／sync_step／backend）：**34 项全 PASS、exit 0**。关键值与 `summary.json` 一致：sa20a 稳态整步 8.236、sa01b 8.181、池化 Δstep +0.010s、Δgen −0.0022s。原始日志 `md5sum -c md5sums.txt` **5/5 OK**（4 份 A/B ＋ 2 步冒烟；服务器侧文件名是 `.log`，入库按仓库 `.gitignore` 惯例改为 `.txt`，内容与哈希不变）。

## 复现

```bash
# 服务器侧（4090-1）
pip install --target /tmp/cupy-pkgs --no-deps cupy-cuda12x fastrlock
NCCL_SHM_DISABLE=1 bash run-sep-async.sh                       # 2 步冒烟
NCCL_SHM_DISABLE=1 bash run-sep-async-ab.sh                    # 4 跑链（2.0/0.1 交替）
# 本地
python3 make-summary-separate.py raw > summary.json
python3 verify.py raw
```

脚本默认 `TRAIN_GPUS=4,5 ROLLOUT_GPU=1 STEPS=8`；`SRC` 指向打了 TM-008 补丁的 main 树。**不要**再加 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（见上表）。
