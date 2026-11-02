# Candidate F — 同栈更新侧 PPO mini-batch 结构扫描

目的：候选 E 已证明「放大更新侧单阶段动态批 token 上限」三个杠杆全负（§13.27），说明更新侧 69.9% 的整步占比与 3.32× 会计上界不能由批上限兑现。本批换一个「不同机制」：不改任何阶段的计算量，只改**更新侧把同一批样本切成几个优化步**（PPO mini-batch 结构），在同一栈上看整步墙钟是否变化，并观察优化轨迹侧效应。

结论：**结构杠杆在墙钟上中性**——1／2／4 个优化步/迭代三种切法整步 27.374／27.372／27.447s，对基线 1.0040／1.0041／1.0013×（全部在 ±0.4% 噪声内，成对差 ≤+0.28%±0.36%）；token 归一化后的 update_actor 成本恒定 **0.316–0.318 s/kTok**，即 actor 更新是 token 计算量受限、优化步数本身的开销低于噪声。但同样 15 次迭代下优化步数 15／30／60 改变了学习轨迹：step 15 验证准确率 0.1333／0.1500／0.2167（单 seed、120 样本，仅方向提示）。**因此 PPO mini-batch 结构应按质量/收敛口径调，不能当整步性能杠杆卖。**

## 固定栈与配置

| 项 | 值 |
|---|---|
| repo | verl `24f25b03`（editable `/home/zxw/venvs/rollout-g8-src/verl`，无 `.git`，版本以 `rollout-g8-setup-v5.sh` 固定注记与 §14 账为准） |
| engine | vLLM `0.24.0+cu129`（wheel，anchor-drift，见 §14） |
| model | Qwen2.5-1.5B-Instruct |
| hardware | 4090-1 GPU 5，单 RTX 4090 24GB |
| data | gsm8k `train_sub480` / `test_sub120` |
| 主要配置 | GRPO、train batch 32 prompts × n=4 = 128 samples/iter、`ppo_micro_batch_size_per_gpu=1`、`ppo_epochs=1`、max_prompt 512、max_response 256、bf16、optimizer_offload=True、`update_weights_bucket_megabytes=256`、`+attn_implementation=sdpa`、`max_model_len=2048`、seed=42、lr 1e-6 恒定（`lr_warmup_steps_ratio=0.0`，无 warmup） |
| 步数 | `TOTAL_EPOCHS=1` → 15 次迭代（与候选 A 前 15 次迭代同 prompt 序、同 seed） |
| 统计窗 | steps 3–14（排除前 2 步 warmup 与第 15 步的最终 testing＋checkpoint save） |

基线 = 候选 A 90 步日志的 steps 3–14（同一配置；仅 `TOTAL_EPOCHS` 6 vs 1，seed／lr／无 warmup 已核一致）：整步 **27.484s**，gen 4.177／old_log_prob 3.773／ref 4.234／adv 0.110／update_actor 13.277／update_weights 1.809s。

## 杠杆定义（源码口径）

`actor.ppo_mini_batch_size` 在 ray trainer 侧先乘 `rollout.n`：`verl/trainer/ppo/ray_trainer.py:1352-1353`（`ppo_mini_batch_size = actor.ppo_mini_batch_size * rollout.n`）。本栈 `rollout.n=4`、每迭代 128 samples、`ppo_micro_batch_size_per_gpu=1`：

| 模式 | `ppo_mini_batch_size` | 每个优化步样本数 | 优化步/迭代 | 每迭代 micro forward/backward 次数 |
|---|---|---|---|---|
| f1 | 8 | 32 | **4** | 128（与其它模式相同） |
| f16 | 16（候选 A 原值） | 64 | **2** | 128 |
| f2 | 32 | 128 | **1** | 128 |

三种切法的**总计算量完全相同**（128 个 micro 前反向），差别只有优化器调用次数与每步梯度统计的样本量。

## 结果（steps 3–14 均值，n=12）

| 模式 | 优化步/迭代 | 整步 | 整步×（vs 基线） | update_actor | update_actor/优化步 | update_actor 每 kTok | 峰值 reserved | step15 val acc |
|---|---|---|---|---|---|---|---|---|
| f2 | 1 | 27.374s | 1.0040 | 13.244s | 13.244s | 0.316 s/kTok | 21.24GB | 0.1333 |
| f16 | 2 | 27.372s | 1.0041 | 13.221s | 6.611s | 0.316 s/kTok | 21.24GB | 0.1500 |
| f1 | 4 | 27.447s | 1.0013 | 13.308s | 3.327s | 0.318 s/kTok | 21.23GB | 0.2167 |

- **① 整步中性**：三模式相对基线 1.0040／1.0041／1.0013×，两两差 0.01%／0.27%／0.28%；单次运行 12 步均值的 SEM ≈0.19–0.20s（≈0.7%）。成对逐步差（同 prompt 序、逐步配对）：f1−f16 = +0.075s ±0.099（+0.28%±0.36%）、f2−f16 = +0.002s ±0.131、f1−f2 = +0.073s ±0.075，**全部不显著**。
- **② 机制**：`update_actor/优化步` 精确随 mini-batch 样本数线性（13.244 : 6.611 : 3.327 ≈ 4 : 2 : 1，偏差 ≤0.7%）；按实际 token 数归一后三模式均为 0.316–0.318 s/kTok。→ actor 更新成本由 token 量决定，优化器调用本身的开销（1 次 vs 4 次）低于测量噪声。
- **③ 质量侧效应（仅方向提示）**：同为 15 次迭代，优化步数 15／30／60 对应 step-15 验证 acc 0.1333／0.1500／0.2167，单调同向。单 seed、120 样本（二项 SE ≈3.3pp、两比例差 SE ≈4.7pp），f1 vs f2 差 8.3pp 约 1.8σ，**不作显著性结论**；且该比较同时改变了每步梯度统计量，属优化语义变化，不是性能收益。
- **④ 复测一致性**：f16 与候选 A 基线同配置跨进程复测，整步 −0.41%、六阶段各偏差 ≤0.6%，为同栈锚点提供第三次独立基线测量（前两次为候选 A 与候选 E 的 e2/e3 非目标阶段）。
- **⑤ 显存**：三模式峰值 reserved 21.23–21.24GB，与基线 21.24GB 相同——结构切分不改变显存占用。

## 执行记录

- **执行（并行会话，2026-10-31 14:15–14:45 UTC）**：`run-candidate-f-chain.sh` 顺序跑 f16→f1→f2（`MODE=` 驱动 `run-candidate-f.sh`），checkpoint/日志落 `/tmp`（当时 `/home` 100% 满）；日志 mtime 14:25／14:35／14:45 UTC。该会话只把 f16 日志拷回本地（22:25 CST）后中断，未写账、未归档 f1/f2。
- **核验（2026-11-02 本批）**：三份服务器日志与本地归档 md5 三方一致；各 15/15 步完整；脚本内 `ppo_mini_batch_size` 回显 8／16／32、`train_batch_size=32` 与设计一致；三份日志均在训练结束后出现 DataLoader worker `signal: Killed`（与候选 A 相同的退出清理现象，不影响已输出 step）。`run-candidate-f.sh` 与 `run-candidate-a.sh` 逐行 diff：差异仅 `PPO_MINI_BATCH_SIZE`（MODE 分支）、`TOTAL_EPOCHS` 6→1、项目名与 checkpoint 目录（`/home/...`→`/tmp/...`），其余变量与覆盖项完全相同。
- **注意**：三次运行的 rollout 采样互相独立（temperature 采样），`global_seqlen` 逐步不同（均值 41902／41792／41834），因此逐步配对只共享 prompt 序与数据序，不共享响应长度；第 3–14 步均值仍是主口径。

## 复现

```bash
# 服务器侧
MODE=f16 bash /home/zxw/venvs/run-candidate-f.sh     # 或 f1 / f2；chain 脚本见 /home/zxw/venvs/run-candidate-f-chain.sh
# 解析（本地，需先取回日志；steps 3–14 窗口）
python3 parse-candidate-f.py <candidate-A 90step log> raw/rollout-g8-candidate-f-f16.txt raw/...f1.txt raw/...f2.txt
```

`summary.json` 含三模式阶段均值、整步×、每优化步/每 kTok 成本、峰值显存与 step-15 验证值；`raw/md5sums.txt` 为本地归档校验和（三份日志与服务器逐文件一致）。

## 边界

- 单栈单点（1.5B／单 4090／gsm8k 短答案）；只扫 8／16／32 三点（优化步 4／2／1），不外推连续曲线，不改 `ppo_epochs`（恒为 1）。
- 只测墙钟与显存，不做内核级 profiling；「优化器调用开销低于噪声」是三点实测推论，不是逐调用计时。
- 质量侧三点为单 seed、单次运行、120 样本验证，只作方向提示，不构成收敛结论；不据此推荐 mini-batch 取值。
- 本批不新增深读、不改 §13.25 恒等式结论；负/中性结果入 §13.16 G9 引用库（结构杠杆不改变整步时间）。
