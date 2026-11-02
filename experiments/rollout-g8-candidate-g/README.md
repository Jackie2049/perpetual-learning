# 候选 G：同栈多卡强扩展（候选 A 负载，1/2/4 卡）

2026-11-02 执行（并行 M5 会话，03:39–04:26 UTC）／2026-11-02 跨会话独立核验与立账（M5 算力侧第四十五批，主文档 §13.36）。

## 目的

候选 A–F 建立起同栈单卡锚点（整步 27.48–27.53s）并证明「放大更新侧批上限」与「改 PPO mini-batch 结构」都不是整步性能杠杆。本实验回答下一个问题：**同一负载下增加 GPU 能拿到多少整步收益，收益被哪一阶段吃掉**。固定全局批（32 prompts × n=4 = 128 samples/迭代），只改 `NGPUS_PER_NODE`（1/2/4），得到同栈强扩展曲线与逐阶段缩放因子 `G_k`，为 §13.25 多阶段恒等式提供同栈实测的 `G_k`，并为 P-1 同栈横评的多卡锚点提供参照。

## 配置

- **repo**：verl `24f25b03`（同候选 A/F），vLLM 0.24.0+cu129
- **model**：Qwen2.5-1.5B-Instruct
- **hardware**：4090-1（8×RTX 4090 24GB）；`g1` = GPU 5（NUMA1）；`g2` = GPU 0,1（均 NUMA0）；`g4` = GPU 0,1,3（NUMA0）+ GPU 4（NUMA1）——GPU 2 被他人占用，故 4 卡只能用非连续掩码
- **dataset**：gsm8k `train_sub480` / `test_sub120`
- **固定项**：`train_batch_size=32` prompts × `rollout.n=4` = 128 samples/迭代；`ppo_mini_batch_size=16`（×n → 64 samples/优化步，2 个优化步/迭代）；`ppo_micro_batch_size_per_gpu=1`；`ppo_epochs=1`；max_prompt 512 / max_response 256；bf16；sdpa；`optimizer_offload=True`；`update_weights_bucket_megabytes=256`；seed=42；lr 1e-6 恒定、无 warmup；`TOTAL_EPOCHS=1` → 15 次迭代
- **存储**：`/home` 100% 满（余 6.8G），日志与 checkpoint 全部落 `/tmp`

## 执行账

| 模式 | GPU | 开始（UTC） | 结束（UTC） | exit | 步数 |
|---|---|---|---|---|---|
| g1 | 5 | 03:39:48 | 03:49:42 | 0 | 15/15 |
| g2 首试 | 5,6 | ~03:50 | — | 1 | 0（NCCL CUDA 217） |
| g2dbg | 5,6 | 04:06:39 | 04:07:53 | 1 | 0（同错误，开 NCCL_DEBUG 定位） |
| g2 | 0,1 | 04:09:57 | 04:18:32 | 0 | 15/15 |
| g4 | 0,1,3,4 | 04:19:13 | 04:26:28 | 0 | 15/15 |

三份归档日志各 15 步，`md5` 与服务器原件逐文件一致（g1 `26c0adfe…`／g2 `90dd57b5…`／g4 `43ffba80…`）。三次成功运行结束时均出现 DataLoader worker `signal: Killed`（与候选 A/F 相同的退出清理现象，不影响已输出的 step）。

## 结果（steps 3–14 均值，n=12）

| 模式 | 卡数 | 整步 | 步间 sd | 整步× | 并行效率 | gen | old_log_prob | ref | adv | update_actor | update_weights | 残差 | step15 acc | 峰值 reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g1 | 1 | 27.531s | 0.755 | 1.000 | 1.000 | 4.173 | 3.758 | 4.295 | 0.111 | 13.271 | 1.820 | +0.102 | 0.175 | 21.23GB |
| g2 | 2 | 22.009s | 1.605 | **1.251×** | **0.626** | 3.335 | 2.746 | 2.433 | 0.138 | 10.956 | 2.351 | +0.049 | 0.158 | 14.29GB |
| g4 | 4 | 16.878s | 0.156 | **1.631×** | **0.408** | 2.155 | 2.114 | 1.638 | 0.143 | 8.151 | 2.637 | +0.040 | 0.150 | 11.05GB |

逐阶段缩放因子 `G = g1 阶段时间 / 该模式阶段时间`：

| 阶段 | g2 G | g4 G | 读数 |
|---|---|---|---|
| gen | 1.251 | 1.937 | 次线性：每引擎 batch 从 32 prompts 缩到 16/8，固定开销与解码波次占比上升 |
| old_log_prob | 1.368 | 1.777 | 次线性（前向 + 通信） |
| ref | 1.766 | 2.623 | 最好：纯前向、无优化器 |
| adv | 0.804 | 0.779 | **负扩展**（量级极小，0.11→0.14s） |
| update_actor | 1.211 | 1.628 | **远低于卡数**：FSDP 分片省下的计算被集合通信吃掉 |
| update_weights | 0.774 | 0.690 | **负扩展**：vLLM 引擎数 1→2→4，权重广播目标增加，1.82→2.35→2.64s |

## 结论

1. **强扩展效率快速衰减**：2 卡整步 1.251×（效率 62.6%），4 卡 1.631×（效率 40.8%）。同一负载增加 4 倍算力只换来 1.63 倍整步加速。
2. **瓶颈在更新侧而非生成侧**：g1 中 `update_actor` 占整步 48.2%，其 2/4 卡 G 只有 1.211/1.628；`ref` 与 `old_log_prob` 也在 1.4–2.6× 区间；只有 `ref` 在 4 卡上接近 2.6×。
3. **`update_weights` 是唯一随卡数变慢的阶段**（0.774/0.690×）：rollout 引擎从 1 个变 4 个，权重同步的广播目标与串行环节增加。这项成本在单卡锚点里只占 6.6%，多卡下会持续放大，是 L4 权重同步与 L2 引擎数量的耦合代价。
4. **训练侧 69.9% 的会计上界不能靠加卡兑现**：`update_actor` 在 4 卡上只到 1.628×，远低于 4×；与候选 E（放大批上限全负）、候选 F（改切分零效应）合起来给出同一结论——**G 必须逐机制实测，不能从阶段占比直接外推**。§13.25 多阶段恒等式因此有了同栈实测的 `G_k`（1→2、1→4 两档），而不再只是用论文报值反推。
5. **基线一致性**：g1 27.531s 对候选 A 基线 27.484s 为 +0.17%，与候选 F 的 f16（27.372s，−0.58%）共同构成同栈锚点的第四/第五次独立测量，四条独立运行的稳态整步落在 ±0.6% 内。
6. **质量侧仅方向提示**：同 15 次迭代、同 seed，step-15 验证 acc 0.175／0.158／0.150（单 seed、120 样本，二项 SE≈3.4pp），不作为显著性结论；多卡改变的是并行切分与优化轨迹，不是本轮要测的性能量。
7. **吞吐口径**：日志 `perf/throughput` 为**每 GPU 归一**值（g1 1523.7 = tokens/step；g2 954.1 = (tokens/step)/2；g4 618.1 = /4）。聚合吞吐比为 1.252×／1.623×，与整步加速比一致。

## 失败与修复：NCCL CUDA 217（peer access）

首两次 2 卡尝试（GPU 5,6）都在模型初始化阶段失败，0 步：

```
torch.distributed.DistBackendError: NCCL error ... unhandled cuda error
ncclUnhandledCudaError: Call to CUDA function failed.
Last error: Cuda failure 217 'peer access is not supported between these two devices'
```

根因：`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 下 Ray 通过 `get_accelerator_ids()` 返回的是**物理 GPU id**，而 verl 的 worker 直接把它写进 `LOCAL_RANK` 并 `set_device()`。当 `CUDA_VISIBLE_DEVICES` 是非连续掩码（本机 `0,1,3,4`，GPU 2 被他人占用）时，物理 id 不等于掩码内序号（物理 3 → 序号 2、物理 4 → 序号 3），选错设备后 NCCL 的 SHM 导入触发 CUDA 217。

修复：`verl-worker-device-map.patch`（本地适配，未提交上游）在 worker 设置 `LOCAL_RANK` 前把物理 id 映射为 `CUDA_VISIBLE_DEVICES` 掩码内的序号：

```python
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
_ids = [x.strip() for x in _vis.split(",") if x.strip()]
if str(local_rank) in _ids:
    local_rank = str(_ids.index(str(local_rank)))
```

补丁只影响本机非连续卡号场景；连续掩码（0,1）无需它也能跑（g2 即如此），g4 必须。`NCCL_P2P_DISABLE=1` 全程保持（4090 无 P2P）。

## 复现

```bash
# 服务器：脚本位于 /home/zxw/venvs/run-candidate-g.sh（与本地 run-candidate-g.sh 相同，md5 30019cb6…）
GPUS=5 bash run-candidate-g.sh                 # g1
GPUS=0,1 bash run-candidate-g.sh               # g2（连续掩码）
GPUS=0,1,3,4 bash run-candidate-g.sh           # g4（需先打 verl-worker-device-map.patch）
# 本地解析（日志已归档为 .txt）
python3 parse-candidate-g.py raw/rollout-g8-candidate-g-g1.txt raw/rollout-g8-candidate-g-g2.txt raw/rollout-g8-candidate-g-g4.txt
```

`summary.json` 为本轮生成（含执行账、失败账、阶段缩放、质量值、吞吐口径与显存）。

## 验证范围与边界

- **真实 GPU 实测**：4090-1，verl `24f25b03` + Qwen2.5-1.5B + gsm8k 短答案子集；每模式 15 次迭代、单 seed、单次运行。
- **窗口**：steps 3–14 均值 n=12；step 1–2 含 vLLM/FSDP 预热，step 15 含最终验证/testing，均排除。步间 sd 0.16–1.61s；g2 的 sd 较大（1.6s）主要来自响应长度采样波动。
- **NUMA 未受控**：g1 在 GPU5（NUMA1）、g2 在 GPU0,1（同 NUMA0）、g4 跨 NUMA0/1。单卡无跨卡通信，NUMA 差异主要影响主机内存本地性；g2 是最干净的 2 卡同 NUMA 点，g4 的 1.63× 含跨 NUMA 通信因素。若要分离 NUMA 与卡数，需在 NUMA1 的 4,5,6,7 上重跑 g2/g4（本轮未做）。
- **未做内核级 profiling**：`G_k` 是阶段墙钟比值，不是纯计算加速比；「通信吃掉 update_actor 收益」是从阶段缩放与 FSDP 结构推断，不是 NCCL 计时。
- **质量读数**单 seed/120 样本，仅方向提示；多卡与单卡的差异同时含优化轨迹与数据切分变化，不是性能对比。
- **补丁为本地适配**：不应视为上游缺陷结论（只在 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + 非连续掩码下触发）。
- 本轮不新增文献身份、不写 survey 正文；实验数据由并行 M5 会话产出，本批独立核验并立账（不把核验算作重跑）。
