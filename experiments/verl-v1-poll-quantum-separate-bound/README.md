# separate_async 的 rollout-bound 区间：2.0s 量子会不会回到关键路径

2026-11-03 起草、2026-11-04 补完并入库（M5 算力侧）。同专项其余模式见 [sync](../verl-v1-poll-quantum/README.md)、[DAPO](../verl-v1-poll-quantum-dapo/README.md)、[4 卡](../verl-v1-poll-quantum-dapo-g4/README.md)、[colocate_async](../verl-v1-poll-quantum-async/README.md)、[stale 通道](../verl-v1-poll-quantum-stale/README.md)、[separate_async 已完成的区间](../verl-v1-poll-quantum-separate/README.md)。

> **状态：已执行（2026-11-04 15:5x–16:3x 窗口，7 跑全部 exit 0）。** 两条结论：① **rollout-bound 区间在 1 rollout : 2 train 配比下不可达**——三个负载点 GATE 全未过，生成侧余量 4.5–4.7×（`raw/gate-*.txt`、`analyze-gate.py`）；② 换成策略陈旧度造成的等待后，量子**确实回到关键路径**——池化 Δgen **+0.544 s**、Δstep **+0.789 s**（`raw-stale/`、`analyze-stale.py`、`verify-bound.py` exit 0）。下文先给预登记与协议（原样保留），再给两段结果。

## 要补的口子

[§13.54 的 separate_async 实验](../verl-v1-poll-quantum-separate/README.md)测出的结论是 poll 不在关键路径上：稳态 `timing_s/gen` 只有 0.031–0.038 s，整步差 +0.010 s（t=0.12）。但它当时就写明了边界——**那是 rollout 跑在训练前头的区间**（1 张 rollout 卡对 2 张训练卡，每步 16 条、平均 233 token）。一旦 rollout 变成瓶颈、队列空，训练步就会真的进入 `ReplayBufferAsync.sample()` 的轮询睡眠，量子理应回到关键路径。这一格没测。

本目录就是补这一格：在**同栈、同卡、同前置条件**下把负载压到生成侧，看 2.0 s 量子是否重现。

## 设计

| 维度 | §13.54（已完成，rollout 领先） | 本目录（rollout-bound） |
|---|---|---|
| 训练卡 / rollout 卡 | GPU 4,5 / GPU 1 | 同左 |
| `data.train_batch_size`（prompt） | 4 | 4 |
| `rollout.n` | 4 ⇒ 16 条/步 | **16 ⇒ 64 条/步** |
| `data.max_response_length` | 256 | **1024** |
| `parameter_sync_step` | 4 | **1** |
| `actor.ppo_mini_batch_size` | 1 | **4**（每步一次更新，不再四次微步） |
| 实测瓶颈 | 训练（8.2 s/步 vs 生成约 2 s） | 预期生成 |

**设计假设（跑之前只是假设）**：每条更新覆盖的样本数 ×4、序列长度上限 ×4 ⇒ 生成侧 token 量约 ×16；训练侧每步只做一次更新，且 0.5B 在 2 卡上以固定开销为主（§13.54 里 `old_log_prob` 3.1 s 对应仅 3.7k token，明显是固定开销主导）。两侧不成比例地朝生成侧倾斜，训练步应开始在 `sample()` 上等待。

`parameter_sync_step=1` 不是随意取的：`PPOTrainerSeparateAsync.__init__` 硬性要求 `train_batch_size == parameter_sync_step × ppo_mini_batch_size`，要"每步一次更新"只能是 1×4。

## 预登记判定规则（跑之前写死，避免事后解释）

1. **GATE——区间到底翻没翻**：poll 2.0 臂稳态（steps 3–8）的 `timing_s/gen` 必须有**过半数步 ≥ 1.9 s**。separate_async 里 `gen` 是训练侧等批的等待，生成跑在另一张卡上；poll=2.0 时它要超过 1.9 s，**唯一途径**就是进了 `_wait_for_next_poll` 的睡眠循环。
   - GATE 不过 ⇒ **不跑 A/B、不写"量子不存在"**，说明负载还没把训练逼进等待，先把 `MAX_RESP` 抬到 1536、再不行 `NORM_RESP=32`，重冒烟。
   - 0.1 臂在这里只作对照，**不设"必须 <1.9 s"的门**：本区间真实等待本身可能超过 2 s，用 1.9 s 卡 0.1 臂会把好数据判死（本轮自检就是这么抓到的，见下）。
2. **主读数**：Δgen 与 Δstep（2.0 − 0.1）。量子是**向上取整**不是外加 2 s，所以预期 Δgen 落在 0–2 s 之间、Δstep 基本被 Δgen 吸收，其余相位 |Δ| ≤ 噪声。
3. **符号一致性**：两个 2.0 跑方向必须一致；不一致就只报窗口内方向，不外推。
4. **不能说的**：即使 Δstep 很大，也不能说"separate_async 平均慢 X%"——本配置是人为加压，不是生产配比（真正的生产配比取决于生成/训练算力比）。

## 2026-11-04 15:5x 窗口：GATE 在三个负载点全未过——该区间在本机不可达

**现场**：`4090-1` 当日恢复（ICMP 0% 丢包、`:22` 可连；同日早些时候的两次探测都是双链路不可达）。GPU 0 是 liyaowei 的 Embedding EngineCore（20.8 G），2/3/6/7 是其 4 卡 Qwen3-VL-32B vLLM（各 23.6 G），**1/4/5 空闲**——正好是本实验要的 2 train（4,5 同 socket）＋1 rollout（1）。`/home` 100% 满，全程写 `/tmp`。受测树 `/tmp/m5-main-smoke/verl`、TM-008 补丁（`trainer_base.py:170`）、device-map 适配、`/tmp/cupy-pkgs`、venv `rollout-g8`（Python 3.10.12）、模型与数据全部在位。

**三份真日志**（每份 3 步、`exit=0`、`raw/gate-*.txt`）：

| 负载点 | `n` | `max_response` | 稳态 `timing_s/gen` | 稳态整步 | 响应均值／撞顶率 | GATE |
|---|---|---|---|---|---|---|
| 校准 | 16 | 1024 | 0.0119／0.0119 | 8.87／8.91 s | 344.5／325.7，1.6% | **未过** |
| 升 cap | 16 | 1536 | 0.0103／0.0111 | 8.71 s | 310.5／358.0，1.6% | **未过** |
| 升 `n`＋cap | 32 | 1536 | 0.0129／0.0112 | 13.38／13.25 s | 315.2，1.6% | **未过** |

三点都是**训练步 8.9→13.4 s、等待 0.01 s**。`MAX_RESP` 这条预登记阶梯的第一级**结构性地不可能加负载**：响应根本没顶到 1024（撞顶率恒 1.6%、均值 ~330），抬到 1536 一个 token 也没多给。

**为什么加了负载反而更远：训练与生成按同一个 token 数计价。** separate_async 里训练步消费的就是 rollout 刚产出的那些轨迹，`n`／长度一抬，两侧同倍数增长。从同一份日志反解（`raw/` 可复算）：

| 臂 | 每步 token | 训练 actor 耗时 | 固定开销（权重同步＋adv） | **训练 ms/token** | 步时 | 步时允许的生成速率 | **实测生成速率** | 余量 |
|---|---|---|---|---|---|---|---|---|
| n16-r1024 | 27,723 | 6.69 s | 2.13 s | **0.241** | 8.87 s | 3,125 tok/s | 14,474 tok/s | **4.6×** |
| n16-r1536 | 26,752 | 6.62 s | 2.04 s | **0.248** | 8.71 s | 3,072 tok/s | 14,474 tok/s | **4.7×** |
| n32-r1536 | 54,104 | 11.04 s | 2.15 s | **0.204** | 13.25 s | 4,084 tok/s | 18,246 tok/s | **4.5×** |

生成侧速率来自**引擎级探针**（`gen-cost-probe.py`，vLLM 0.24.0 离线 API、同模型、同卡 1、同批形状；64 序列 2.37–2.66 s、128 序列 4.38–4.78 s，首个 repeat 作编译预热丢弃）。**口径**：这是引擎级、不是 verl 栈级读数，不含权重同步、调度与 agent loop 开销；它只用于给"余量"定界。

**结论（负，可复算）**：在这套 1 rollout : 2 train 的配比上，训练每 token 的墙钟成本是生成的 **3.5–4.5×**（按卡算 ~7–9×），所以**加 token 只会把差距拉大**（余量 4.6×→4.5×，步时 +4.4 s 全部来自训练侧）。要翻转需要生成慢约 4.5×或训练吞吐高约 4.5×（在 1:2 下相当于再加 ~7 张训练卡，线性外推、未实测）⇒ **`timing_s/gen` 量不出量子效应**，预登记规则禁止在此读 A/B，本轮**没有**跑 2.0/0.1 对照（跑了也只是把 §13.54 的"≈0"再抄一遍）。

**顺带关掉两项"只能窗口验"的欠账**：① 新配置在真 dump 里读作 `poll_interval=2.0`、`max_response_length=1024`、`parameter_sync_step=1`（旧日志是 4／256），解析路径一致；② 64／128 条每步下的稳态缺口为 **0.035–0.053 s**（阈值 `max(0.1 s, 2%×步时)` = 0.174–0.265 s），比之前记的 0.027–0.035 s 略大但仍在阈值内，`verify-bound.py` 的机械有效性判据不会被这套配置判死。

**另记一条环境噪声（非上游缺陷）**：训练正常结束后，`atexit` 里会打一条 `RuntimeError: DataLoader worker ... killed by signal: Killed` 的 traceback，`exit` 仍是 0，三跑都有——归档日志里出现它是预期的，别当成失败。

## stale-wait 臂（2026-11-04 15:5x 窗口内**先预登记、后执行**）

三个负载点（校准／`MAX_RESP=1536`／`NORM_RESP=32`）的 GATE 全部未过（结果见下一节）：在这套 1 rollout : 2 train 的配比下，**生成侧余量约 4.5×**，训练步根本不会阻塞在 `sample()`，所以"rollout 成为瓶颈"这一格在本机不可达。但 §13.54 留给正文的那句话是关于**等待本身**的——"收益是训练步是否阻塞在 `sample()` 的函数"。既然算力上压不出等待，就换一个**真实存在的**等待来源：策略陈旧度。

**配置**：与校准臂完全相同（`n=16`、`max_response=1024`、`sync_step=1`、`mini_batch=4`、`TQ_SIZE=1536`），**唯一改动**是 `trainer.v1.sampler.max_off_policy_threshold` 由 8 改为 **1**（`strategy=drop` 沿用脚本默认）。

**机制（跑之前写死）**：`parameter_sync_step=1` 每步同步一次权重 ⇒ 上一步生成的轨迹在同步后全部陈旧；threshold=1 时 `_stale_terminal_keys` 会把整批判死并丢弃，缓冲清空 ⇒ `sample()` 必须等下一批新数据生成完，等待进入关键路径。这与"生成算力不足"是两种不同的等待来源，但落在同一个 `ReplayBufferAsync.sample()` 上。

**可证伪的预测**：
1. **GATE′**：2.0 臂稳态 `timing_s/gen` ≥1.9 s 的步数过半（等待是真实的）；
2. **Δgen（2.0 − 0.1）≥ 0 且 ≤ 约 2 s**——量子是**向上取整**不是外加 2 s；
3. **Δstep ≈ Δgen**，其余相位 |Δ| ≤ 0.2 s；
4. 2.0 臂两跑方向一致。

**若 2.0 臂 Δgen ≈ 0 而等待确实存在**，则说明 `_wait_for_next_poll` 不是量子的作用点，需要回到代码解释——这是一个强负结论，不能含糊过去。

**边界**：4 跑 × 8 步、单 seed、每臂两跑 ⇒ 只给窗口内方向，无 run 级 CI；drop 臂每步丢弃再生成一批，训练量与校准臂同量级（64 条/步）；`NCCL_SHM_DISABLE=1` 使绝对步时不可与其他批次比。

## stale-wait 臂执行结果：四条预登记全部成立

| 运行 | poll | gen 均值 | 整步均值 | old_log_prob | update_actor | update_weights | tokens/步 |
|---|---|---|---|---|---|---|---|
| sbw20a | 2.0 | **2.768** | 11.003 | 1.672 | 4.461 | 1.984 | 26,650 |
| sbw01a | 0.1 | **2.112** | 9.863 | 1.636 | 4.035 | 1.964 | 24,365 |
| sbw20b | 2.0 | **2.765** | 10.651 | 1.704 | 4.064 | 2.000 | 24,955 |
| sbw01b | 0.1 | **2.333** | 10.213 | 1.667 | 4.082 | 2.019 | 25,403 |

（稳态 steps 3–8，n=6/跑；08:19–08:37Z，4/4 `exit 0`、各 8/8 步。`analyze-stale.py` → **四条预登记全 PASS、exit 0**；`verify-bound.py raw-stale` → 机械 0 失败、GATE 双跑通过、**exit 0**。）

**读数一：池化 Δgen +0.544 s、Δstep +0.789 s**（＝0.1 臂整步的 **7.9%**）。余下 **+0.246 s 不是处理效应**，而是 token 量的抽样差：两臂 `update_actor` 每 token **0.165 vs 0.163 ms**，token 量差 +3.7%，按 token 归一后该相位只差 +0.052 s。也就是说预登记第 3 条里"其余相位 |Δ| ≤ 0.2 s"的**原始**读数在 `update_actor` 上到了 +0.204 s，超容差 0.004 s——归一到 token 后合格，这一点必须原样记下，不能只报归一值。

**读数二：量化签名在分布上，不只是均值。** 2.0 臂 16 个步实例**无一低于 1.9 s**（最小 2.02／2.09），且 **5/16** 落在 4.03–4.11 s（正好两个量子）；0.1 臂没有任何两量子步，sb01a **4/8 步低于 1.9 s**（最小 1.72）。真实等待 ~2.0 s 恰好骑在 2.0 s 刻度上 ⇒ 越过刻度的那一步要多付整个量子。用实测频次推算 0.31×2.0 s ≈ **0.62 s**，与 Δgen 0.544 s 同量级。

**读数三：等待来源是逐出，不是算力。** `parameter_sync_step=1` 每步同步权重 ⇒ 上一步轨迹全部陈旧；threshold=1 把在飞的 4 条判死丢弃（每步 `evicted_samples=4.0`、staleness 2.0），缓冲见底后 `sample()` 必须等下一批（~2 s）。

**它和 §13.54 的关系**：同一个 trainer mode 现在两端都有——rollout 领先时 **+0.1%**、训练步真的阻塞时 **−7.9%**。量子成本取决于真实等待与刻度的相对位置：等待≈一个量子时最贵，≪ 半个量子或 ≫ 两个量子都更便宜。这是**人工制造的等待**（max staleness 1），不代表生产配比。

## 运行协议（GPU 窗口到来时按序执行）

1. **先核 GPU 归属**（不据旧记录假定空闲）：
   `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` ＋ `ps -o pid,etime,cmd -p <pid>`；需要 **4,5（同 socket）＋1** 三张卡。
2. **确认受测树还在**：`/tmp/m5-main-smoke/verl`（main `3fdf2b54` ＋ TM-008 补丁 ＋ 本地 device-map 适配）与 `/tmp/cupy-pkgs` 都在 **`/tmp`，不持久**。缺了就按 [main 冒烟](../verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话)的记录重建（浅克隆 + `patch -p1 --fuzz=0`，另加 device-map 与 TransferQueue 容量两处本机适配）。
3. **冒烟 3 步**：把三份脚本传到服务器 `/tmp` 后
   `MODE=sepb-smoke STEPS=3 POLL=2.0 bash /tmp/run-sep-bound.sh`
   看两件事：日志里没有 `Storage capacity exceeded`；稳态 `timing_s/gen` 是否 ≥1.9 s。
4. **过 GATE 再跑 A/B**：`bash /tmp/run-sep-bound-ab.sh`（2.0/0.1 交替各两跑、8 步/跑，约 4×5 min 含启动）。
5. **可选剂量臂**：`bash /tmp/run-sep-bound-05.sh`（poll 0.5 两跑）。只有 A/B 出现效应时才值得跑。
6. **归档**：`mkdir -p raw && cp /tmp/sb{20a,01a,20b,01b}.log raw/<mode>.txt`（**改后缀为 `.txt`**，仓库 `.gitignore` 吞 `*.log`）→ `python3 make-summary-bound.py raw > summary.json` → `python3 verify-bound.py raw` → `md5sum` 与服务器逐文件对齐 → 写回本 README（执行账、结果表、边界）。

## 本机前置（沿用 §13.54，两条都不是上游缺陷）

- **不能设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`**：本模式下 `ray.util.collective.barrier()` 在 `_used_gpu_indices` 为空时按本进程可见卡数建通信器，NOSET 下每个 actor 见 3 张卡 ⇒ 每 rank 都占 0/1/2，报 `Multiple Ranks are using the same GPU`。
- **必须 `NCCL_SHM_DISABLE=1`**：去掉 NOSET 后训练侧 FSDP 广播撞 `Cuda failure 217 'peer access is not supported'`。代价是集合通信退化为 socket，**绝对步时不可与其他批次比**。
- `nccl_checkpoint_engine` 加载期 `import cupy`，venv 没有 ⇒ 装到 `/tmp/cupy-pkgs`（`/home` 长期接近满，不写它）。

## 2026-11-04 12:5x 轮：对准备物料做的三处修改

1. **TransferQueue 容量 512 → 1536**：`SimpleStorage` 把 `total_storage_size` 摊到 8 个单元，512 就是 64/单元，**正好等于本配置一步的 64 条**，会构造性复现上一轮冒烟撞到的 `Storage capacity exceeded: N existing + 1 new > N`（当时是 16/步对 16 容量）。1536 ⇒ 192/单元 = 三步余量。
2. **`verify-bound.py` 的判定拆成三类**：机械有效性（缺步、配置漂移、`step ≠ Σ相位 + 缺口`）→ exit 1；测量有效但区间没翻 → **exit 2**（GATE 挡住 A/B）；两者都过 → exit 0。效应量改为 INFO 打印、不再参与 pass/fail。原来的写法把"区间没翻转"也报成 FAIL，窗口里极易被误读成"数据有问题"。
3. **新增 `selftest-bound.py`**：用真实控制台日志格式合成 4 组夹具，把两个解析器当黑盒跑，10 项断言全过（`exit 0`）。它当场抓出两个错——夹具写成双引号键（真实配置是单引号），以及上面那条 0.1 臂门槛写错。

## 2026-11-04 13:5x 轮（无卡轮）：真日志离线核对，抓到三个会在窗口里炸的坑

两条算力链路仍不可达（见文末「阻塞」）。本轮不重复探测，改把"只有窗口才能验"的部分尽量提前验掉。**四个真问题，都不是数据问题，两个会直接浪费窗口。**

**一、真日志核对：解析器可以提前验，不必等窗口。** 新增 [`crosscheck-real-logs.py`](crosscheck-real-logs.py)，拿**上一轮 separate_async 的真实控制台日志**（`../verl-v1-poll-quantum-separate/raw/sa{20,01}{a,b}.txt`，同一 mode／同一机器／同一脚本家族，只有负载不同）跑三个**互相独立**的解析器（本目录两个 + 那份已发布的 `summary.json`）：

- **40 项断言全过**：4 个文件 × 6 个关键指标（gen／step／old_log_prob／update_actor／update_weights／response_length），三路均值逐位一致（差 <1e-9）；
- **`np.float64(...)` 形态确实存在**：每个日志 **104 处** repr，两条解析路径都正确取值——原先记的"窗口期才能核对"可以关掉；
- **控制台没有污染**：每份日志 8 条 step 行，解析出 8 步，`logged == parsed`，不存在 EngineCore 尾巴被当成 step 行；
- **配置抽取在真 dump 上可读**（`trainer_mode=separate_async`、`poll_interval`、`parameter_sync_step=4`、`max_response_length=256`——那是旧配置的值，新配置应读成 1／1024）。

**二、缺口判据会把好数据判死（已修）。** 旧版把 `step ≠ Σ相位 + 缺口` 的检查套在**步 3–8**，而步 8 是最后一步。实测归档日志：sync／DAPO／colocate 三种模式的**最后一步都额外吞进收尾工作**（比相位和多 **+3.9～+7.6 s**），separate_async 只是恰好干净（+0.03 s）。⇒ 本模式的最后一步一旦也带上测试／存盘，整批数据会被判成机械无效。现改为**最后一步不进门、只打 INFO**，并加负向对照：同一份"最后一步 +6 s"夹具，旧版 **rc=1（4 条失败）**、新版 **rc=0**。

**三、缺相位键会直接抛 traceback（已修）。** 旧版用 `sum(steps[s][k] for k in …)` 取相位和，任一键缺失就是 `KeyError` 栈——在窗口里表现为"脚本崩了"，而不是"哪一项缺了"。现先做存在性检查（点名字段）再算缺口。另外阈值从写死的 `0.1 s` 改为 `max(0.1 s, 中位步长 × 2%)`：真日志稳态缺口只有 **0.027–0.035 s**，但步长本身会变，固定阈值会把固定开销误判成坏账。

顺带核清一条**不能混用**的事实：sync／DAPO／colocate 日志里有 `timing_s/ref`（≈2.4 s）而本模式没有——所以"用 5 个相位去套所有模式"本身就会造出 ~2.4 s 的假缺口。判据只能按模式写。

**四、冒烟命令默认会撞 CUDA 217（已修）。** 协议里的冒烟是直接 `bash run-sep-bound.sh`，而脚本原来 `NCCL_SHM_DISABLE=${…:-0}`——默认值 0 与"本模式必须 1"矛盾（A/B 链脚本自己导出了 1，冒烟路径没有）。直接跑冒烟会死在 FSDP 广播，且现场看起来像"受测树坏了"。现默认改为 **1**。

## 离线自检

```
python3 selftest-bound.py        # 14 PASS / 0 FAIL, exit 0
python3 crosscheck-real-logs.py  # 40 PASS / 0 FAIL, exit 0（真日志三路互核）
python3 verify-bound.py raw-stale   # 机械 0 失败 / GATE 双跑通过 / exit 0（stale-wait 臂 A/B）
python3 analyze-stale.py            # 四条预登记全 PASS / exit 0
python3 analyze-gate.py             # GATE 三点全 FAIL / exit 2（负结论，exit 2 是正确的退出码）
python3 -m py_compile make-summary-bound.py verify-bound.py selftest-bound.py crosscheck-real-logs.py analyze-gate.py analyze-stale.py gen-cost-probe.py
bash -n run-sep-bound.sh run-sep-bound-ab.sh run-sep-bound-05.sh
```

自检覆盖：①区间翻转的夹具 → `verify` exit 0 且池化 Δstep = Δgen = 0.9 s；②没翻转的夹具 → exit 2 且 GATE 点名 2.0 臂；③少一步 → exit 1；④账对不上（每步凭空多 0.5 s）→ exit 1；⑤只有最后一步多 6 s → **仍 exit 0**（不收尾工作影响）；⑥缺一个相位键 → exit 1 且点名字段、无 traceback。

**原先"只能窗口验"的两件事已在 2026-11-04 窗口关闭**：真 dump 读作 `poll_interval=2.0`／`max_response_length=1024`／`parameter_sync_step=1`；64／128 条每步的稳态缺口 0.035–0.053 s，阈值 `max(0.1 s, 2%×步时)` = 0.174–0.265 s。

## 边界与已知风险

- 仍然是**单栈单点**：0.5B／gsm8k／单 seed／8 步／每臂两跑。即使效应显著，也不外推到其他模型规模。
- "每臂两跑"只够给窗口内方向，**给不出 run 级 CI**（与 §13.53 的 `wait` t=1 同一条限制）。
- 负载是否真翻区间**只有 GATE 能答**；本目录不预设它一定翻。翻不了本身就是一条结论（写进 §13.54 的边界，而不是把 A/B 硬解释成"量子无效"）。
- `TQ_SIZE` 余量是按"一步 64 条"算的；若把 `NORM_RESP` 抬到 32（128 条/步），容量要同步抬到 3072。

## 阻塞（2026-11-04 12:5x／13:5x 两次实测；**15:5x 已恢复**）

| 通路 | 结果 | 备注 |
|---|---|---|
| `4090-1` = `219.223.198.62:22` | 12:5x／13:5x：ICMP 100% 丢包、`nc` TCP 超时；**15:58 起恢复** | **公网直连、不经 VPN**；外网正常 |
| 昇腾 VPN 网关 `10.205.0.1` | 2 包全丢 | 15:58 复核仍不可达 |
| `192.168.13.154/155/156:22` | 全超时 | 15:58 复核仍不可达 |

两轮各只探一次、未重试。**2026-11-04 15:58 CST 复核：`4090-1` 已恢复（ICMP 0% 丢包、`:22` 可连），本轮 7 跑即在该窗口完成；期间出现过一次约 5 分钟的 SSH 超时（同日内已知的分钟级抖动），nohup 起的远端任务未受影响。昇腾仍不可达。**

## 与既有结论的关系

**不改 §13.54 的任何读数**（rollout 领先区间 +0.1% 照旧），**新增两件事**：① rollout-bound 区间在 1:2 配比下不可达，这是负结论，写进 §13.54 边界；② stale-wait 臂给出第二个端点——训练步真阻塞时整步 −7.9%，于是"收益是训练步是否阻塞在 `sample()` 的函数"从一句推断变成跨区间证据。**注意不能反过来说"separate_async 平均慢 7.9%"**：该臂的等待是 `max_off_policy_threshold=1` 人为制造的，生产配比取决于生成/训练算力比与陈旧度上限。
