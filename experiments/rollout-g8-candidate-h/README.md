# Candidate H — update_weights bucket 大小扫描（2 卡双传输对照）

目的：候选 G 测得 `update_weights` 是唯一随卡数变慢的阶段（g1 1.820 → g2 2.351 → g4 2.637s，G=0.774／0.690），第四十六批把成因写成「由副本数决定、与全局批无关」。本批检验它是不是**分块传输大小**的可调杠杆——`rollout.checkpoint_engine.update_weights_bucket_megabytes` 在候选 A–G 一直固定 256MB，源码路径 `vllm_rollout/bucketed_weight_transfer.py` 按该值切块。同栈同配置只改 bucket = 256／1024／4096（16× 跨度），看 `update_weights` 与整步是否变化。

结论：**bucket 不是 update_weights 的有效杠杆**——16× 跨度只让该阶段变化 −1.2%（正常传输组：2.295→2.268s）／−2.6%（socket 组：2.713→2.643s），远小于 g1→g2→g4 的 +29%／+45%；负扩展是每副本装载／协议开销，不是 chunk-size 问题。本批还留下一条方法学边界：**多卡测量必须先设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`**，否则 Ray 给每个 worker 单卡掩码、NCCL SHM 导入失败（CUDA 217），只能退回 socket 传输（update_actor +40.5%）。

## 固定栈与配置

| 项 | 值 |
|---|---|
| repo | verl `24f25b03`（editable `/home/zxw/venvs/rollout-g8-src/verl`） |
| engine | vLLM `0.24.0+cu129`（anchor-drift，同候选 C/D/E/F/G） |
| model | Qwen2.5-1.5B-Instruct |
| hardware | 4090-1 GPU 0,1（2× RTX 4090 24GB，NUMA0） |
| data | gsm8k `train_sub480` / `test_sub120` |
| 主要配置 | GRPO、train batch 32 prompts × n=4 = 128 samples/iter、`ppo_mini_batch_size=16`、micro=1、`ppo_epochs=1`、max_prompt 512／max_response 256、bf16、optimizer_offload=True、`+attn_implementation=sdpa`、`max_model_len=2048`、seed=42、lr 1e-6 恒定、`TOTAL_EPOCHS=1` → 15 次迭代 |
| 统计窗 | steps 3–14（排除 2 步预热与含终评/存盘的末步），n=12 |
| 杠杆 | `rollout.checkpoint_engine.update_weights_bucket_megabytes` = 256／1024／4096 |

## 执行账（含失败与传输对照）

**4 卡尝试全部失败（05:21–05:53 UTC）**：先在 `0,1,3,4`（跨 NUMA）、后改 `5,6,7,4`（全 NUMA1），三连失败——前两次是 NCCL `Cuda failure 217 'peer access is not supported between these two devices'`（`ncclShmImportShareableBuffer`），失败运行又留下未回收的 Ray/vLLM worker（每卡 2.6GB＋4.8GB），后续启动改报 vLLM `No available memory for the cache blocks`（`gpu_memory_utilization=0.20` 下无 KV cache 预算）。同期并行 M5 会话也在尝试 4 卡复测，两作业共用 GPU4 后双双失败（其日记记「同机两个 4 卡作业不可并行」）。

**最小 NCCL 探针定位（不依赖 verl）**：4 worker Ray + `_broadcast_coalesced` 在 `CUDA_VISIBLE_DEVICES=5,6,7,4` 下复现 217；根因是**未设** `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 时 Ray 给每个 worker 单卡掩码（`CUDA_VISIBLE_DEVICES=5` 等），NCCL SHM 跨进程导入需要 peer access 而 4090 不支持；设 `NOSET=1` 后 worker 共享完整掩码（`'5,6'`）、物理 id 经本地 `worker.py` 补丁映射为 ordinal、SHM 正常。同批并行会话的成功多卡跑（g2n/g4n）也都带 `NOSET=1`，与第四十五／四十七批的失败账一致。

**2 卡（0,1）两组数据，各 3 模式 15/15 步、exit 0**（日志与 checkpoint 均落 `/tmp`）：

| 组 | 环境 | 模式 | 时间（UTC） |
|---|---|---|---|
| A（socket） | 无 NOSET，`NCCL_SHM_DISABLE=1` | h256／h1024／h4096 | 05:54:15–06:23:57 |
| B（SHM） | `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`，`NCCL_SHM_DISABLE=0` | n256／n1024／n4096 | 06:24:04–06:49:12 |

B 组是主数据（传输与归档的 g1/g2/g4/g2n/g4n 相同）；A 组是传输降级对照。

## 结果（steps 3–14 均值，n=12）

**B 组（SHM 传输，主数据）**

| 模式 | bucket | 整步 | gen | old_log_prob | ref | adv | update_actor | update_weights | 峰值 reserved |
|---|---|---|---|---|---|---|---|---|---|
| n256 | 256 MB | 21.608s | 2.998 | 2.712 | 2.409 | 0.145 | 10.998 | **2.295** | 14.26 GB |
| n1024 | 1024 MB | 22.241s | 3.508 | 2.757 | 2.444 | 0.113 | 11.092 | **2.272** | — |
| n4096 | 4096 MB | 21.370s | 2.827 | 2.710 | 2.440 | 0.110 | 10.966 | **2.268** | — |

**A 组（socket 传输，对照）**

| 模式 | bucket | 整步 | gen | old_log_prob | ref | adv | update_actor | update_weights |
|---|---|---|---|---|---|---|---|---|
| h256 | 256 MB | 27.417s | 3.166 | 3.563 | 2.358 | 0.111 | 15.456 | **2.713** |
| h1024 | 1024 MB | 27.426s | 3.166 | 3.555 | 2.398 | 0.113 | 15.476 | **2.669** |
| h4096 | 4096 MB | 27.942s | 3.503 | 3.542 | 2.340 | 0.110 | 15.752 | **2.643** |

**判决一：bucket 不是 update_weights 的有效杠杆。** 16× 跨度只让该阶段 −1.2%（B 组 2.295→2.268s，G_w=1.012）／−2.6%（A 组 2.713→2.643s，G_w=1.027）；两组整步的波动（21.37–22.24s、27.42–27.94s）由 2 卡已知的 `gen` 二态（4.19s／2.1s）与 update_actor 漂移主导，不是 bucket 的贡献。单阶段恒等式预测在 B 组只有 1.0012×、实测 0.989×——差值与 update_weights 无关。

**判决二：负扩展由副本数／每副本装载决定。** 同一 256MB 下 g1 1.820 → g2 2.295（本轮 n256 与 g2 的 2.351 差 −2.4%）→ g4 2.637s；bucket 的可动区间（≤0.03s）远小于副本数的贡献（≥0.5s）。这与第四十六批「由副本数决定、与全局批无关」一致，并把「可调参数」这一候选解释排除。L4 权重同步若要与 L2 引擎数解耦，应针对副本数/同步次数（共享加载、只同步一次），不是 chunk size。

**判决三（方法学）：多卡测量必须先设 `NOSET=1`。** 同一 bucket=256、同一 2 卡掩码，socket 组 vs SHM 组：整步 27.417 vs 21.608s（+26.9%）、update_actor 15.456 vs 10.998s（+40.5%）、update_weights 2.713 vs 2.295s（+18.2%）。未设 NOSET 时 worker 拿到单卡掩码，NCCL 走 socket，所有多卡阶段的绝对数都会被抬高；本轮 A 组只作传输对照，不与 g 系列混用。B 组 n256 与 g2 对照：整步 −1.8%、update_actor +0.4%、update_weights −2.4%，复制成立。

## 边界

- **2 卡而非 4 卡**：4 卡本轮不可用（NCCL SHM＋失败运行残留显存＋并行作业冲突），因此 bucket 杠杆在 4 引擎下的绝对值未测；但「跨 16× 无效应」与引擎数无关的机制结论仍成立，且 g4 的 2.637s 与 g2 的 2.351s 只差 0.29s，远大于任何观测到的 bucket 效应上限。
- 单 seed／每模式单次运行；bucket 只扫 3 个点（未扫 512/2048）；A/B 两组绝对数不可混用（传输不同）；`gen` 二态使整步比值噪声约 ±1–3%（2 卡），故只报 update_weights 的 G_w。
- 未做内核级 profiling；不解释 bucket 影响为何如此小（机制上是每副本装载/协议常量）。
- 4 卡失败与最小 NCCL 探针是真实执行证据，不把定位归因于上游缺陷（补丁为本地适配）。

## 复现

```bash
# group A (socket transport, fallback when SHM is broken)
GPUS=0,1 BUCKET=256 bash run-candidate-h.sh
# group B (SHM transport, same as archived g1/g2/g4/g2n/g4n)
GPUS=0,1 BUCKET=256 bash run-candidate-h-noset.sh
# parse (steps 3-14; h* = group A, n* = group B)
python3 parse-candidate-h.py raw/rollout-g8-candidate-h-n256.txt \
    raw/rollout-g8-candidate-h-n1024.txt raw/rollout-g8-candidate-h-n4096.txt
```

日志 md5（服务器与本地逐文件一致）：h256 `1873d331…`、h1024 `fdb44baf…`、h4096 `02e174cb…`、n256 `800381d5…`、n1024 `6e382e65…`、n4096 `9f05c3f8…`。原始日志见 [raw/](raw/)，结构化读数见 [summary.json](summary.json)。
