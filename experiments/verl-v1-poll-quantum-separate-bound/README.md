# separate_async 的 rollout-bound 区间：2.0s 量子会不会回到关键路径（**准备就绪，尚未执行**）

2026-11-03 起草、2026-11-04 补完并入库（M5 算力侧）。同专项其余模式见 [sync](../verl-v1-poll-quantum/README.md)、[DAPO](../verl-v1-poll-quantum-dapo/README.md)、[4 卡](../verl-v1-poll-quantum-dapo-g4/README.md)、[colocate_async](../verl-v1-poll-quantum-async/README.md)、[stale 通道](../verl-v1-poll-quantum-stale/README.md)、[separate_async 已完成的区间](../verl-v1-poll-quantum-separate/README.md)。

> **状态：零读数。** 本目录只有准备物料与离线自检，**没有任何 GPU 运行**，不得把这里的任何数字或预期当作实测。2026-11-04 因两条算力链路同时不可达（见文末「阻塞」）先归档，避免准备物料丢失。

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

## 本轮（2026-11-04）对准备物料做的三处修改

1. **TransferQueue 容量 512 → 1536**：`SimpleStorage` 把 `total_storage_size` 摊到 8 个单元，512 就是 64/单元，**正好等于本配置一步的 64 条**，会构造性复现上一轮冒烟撞到的 `Storage capacity exceeded: N existing + 1 new > N`（当时是 16/步对 16 容量）。1536 ⇒ 192/单元 = 三步余量。
2. **`verify-bound.py` 的判定拆成三类**：机械有效性（缺步、配置漂移、`step ≠ Σ相位 + 缺口`）→ exit 1；测量有效但区间没翻 → **exit 2**（GATE 挡住 A/B）；两者都过 → exit 0。效应量改为 INFO 打印、不再参与 pass/fail。原来的写法把"区间没翻转"也报成 FAIL，窗口里极易被误读成"数据有问题"。
3. **新增 `selftest-bound.py`**：用真实控制台日志格式合成 4 组夹具，把两个解析器当黑盒跑，10 项断言全过（`exit 0`）。它当场抓出两个错——夹具写成双引号键（真实配置是单引号），以及上面那条 0.1 臂门槛写错。

## 离线自检

```
python3 selftest-bound.py        # 10 PASS / 0 FAIL, exit 0
python3 -m py_compile make-summary-bound.py verify-bound.py selftest-bound.py
bash -n run-sep-bound.sh run-sep-bound-ab.sh run-sep-bound-05.sh
```

自检覆盖：①区间翻转的夹具 → `verify` exit 0 且池化 Δstep = Δgen = 0.9 s；②没翻转的夹具 → exit 2 且 GATE 点名 2.0 臂；③少一步 → exit 1；④账对不上（每步凭空多 0.5 s）→ exit 1。

## 边界与已知风险

- 仍然是**单栈单点**：0.5B／gsm8k／单 seed／8 步／每臂两跑。即使效应显著，也不外推到其他模型规模。
- "每臂两跑"只够给窗口内方向，**给不出 run 级 CI**（与 §13.53 的 `wait` t=1 同一条限制）。
- 负载是否真翻区间**只有 GATE 能答**；本目录不预设它一定翻。翻不了本身就是一条结论（写进 §13.54 的边界，而不是把 A/B 硬解释成"量子无效"）。
- `TQ_SIZE` 余量是按"一步 64 条"算的；若把 `NORM_RESP` 抬到 32（128 条/步），容量要同步抬到 3072。

## 阻塞（2026-11-04，本轮实测）

| 通路 | 结果 | 备注 |
|---|---|---|
| `4090-1` = `219.223.198.62:22` | ICMP 100% 丢包、TCP 超时 | **公网直连、不经 VPN**；外网正常（GitHub 443 可达） |
| 昇腾 VPN 网关 `10.205.0.1` | 2 包全丢 | |
| `192.168.13.154/155/156:22` | 全超时 | |

本轮只探一次、未重试。恢复条件＝用户侧恢复 4090-1 所在网段（历史上有过约 28 h 的电信段路由断裂）或恢复昇腾 VPN；两者都不需要本 Agent 再做动作。

## 与既有结论的关系

**不改变 §13.54 的任何读数**，也不改 TM-008 的收益表——它只补"rollout 瓶颈时"这一格。若 GATE 通过且 Δstep 显著，§13.54 那句「收益是训练步是否阻塞在 `sample()` 的函数」就得到第二个端点；若 GATE 反复过不了，则说明在 0.5B 这个量级上训练侧无论如何都压不过生成侧，该区间**在本机不可达**，也是一条要写进正文的负结论。
