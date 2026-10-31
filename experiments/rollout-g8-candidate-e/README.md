# Candidate E — 同栈更新侧单阶段批处理杠杆扫描

目的：候选 A 锚点（verl `24f25b03` + Qwen2.5-1.5B 单 4090 GRPO）已测得训练侧合计 69.9% 的整步占比（update_actor 47.6%／ref 15.7%／update_weights 6.6%，§13.23）。§13.25 把「阶段占比中介」升级为多阶段恒等式后，更新侧仍缺**同栈受控杠杆实测**：把单个更新侧阶段的动态批 token 上限调大，能否得到 G>1（该阶段加速）并让整步按恒等式预测变化？本批对 actor／rollout log_prob／ref 三个阶段各做一个单变量覆盖扫描。

结论：**三个杠杆全部为负**——提高动态批 token 上限使目标阶段变慢而非变快；整步按实测分项恒等重建精确（残差 ≤0.03s）。更新侧 3.32× 理论杠杆在本栈不能由「放大 token 上限」兑现。

## 固定栈与配置

| 项 | 值 |
|---|---|
| repo | verl `24f25b03`（editable `/home/zxw/venvs/rollout-g8-src/verl`） |
| engine | vLLM `0.24.0+cu129`（wheel，anchor-drift，见 §14） |
| model | Qwen2.5-1.5B-Instruct |
| hardware | 4090-1 GPU 5，单 RTX 4090 24GB |
| data | gsm8k `train_sub480` / `test_sub120` |
| 主要配置 | GRPO、train batch 32、rollout n=4、max_prompt 512、max_response 256、bf16、optimizer_offload=True、`update_weights_bucket_megabytes=256`、`+attn_implementation=sdpa`、`max_model_len=2048` |
| 步数 | `TOTAL_EPOCHS=1` → 15 步（与候选 A 前 15 步同数据序、恒定 lr 1e-6） |
| 统计窗 | steps 3–14（排除前 2 步 warmup 与第 15 步的最终 testing＋checkpoint save） |

基线 = 候选 A 90 步日志的 steps 3–14（同一配置，仅 `TOTAL_EPOCHS` 不同；见 `raw/rollout-g8-candidate-a-90step.txt`）。

## 三个模式

| 模式 | 单变量覆盖 | 目标阶段 |
|---|---|---|
| e1 | `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` 3000 → 8192 | update_actor |
| e2 | `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` 3000 → 16384 | old_log_prob |
| e3 | `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` 3000 → 16384 | ref |

> 起点均为 3000：`ref/ref.yaml:24` 与 `rollout/rollout.yaml:117` 的 `log_prob_max_token_len_per_gpu` 均 `${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu,16384}`——候选 A 脚本显式设 ppo cap=3000，故三阶段动态批 cap 实际默认 3000（readme 早稿按 8192/4096 推断有误，以本表为准）。

## 结果（steps 3–14 均值，n=12）

基线：整步 **27.484s**，各阶段 gen 4.177／old_log_prob 3.773／ref 4.234／adv 0.110／update_actor 13.277／update_weights 1.809s，未分项 0.105s；峰值 reserved 21.18GB。

| 模式 | 目标阶段 base→mode | 阶段 G | 整步 mode | 整步×（vs 基线） | 单阶段投影 | 恒等重建残差 |
|---|---|---|---|---|---|---|
| e1 | update_actor 13.277→**18.201s**（+37.1%） | **0.729** | 32.358s | **0.8494**（慢 17.8%） | 0.8480 | −0.0004s |
| e2 | old_log_prob 3.773→**4.776s**（+26.6%） | **0.790** | 28.672s | **0.9586**（慢 4.3%） | 0.9648 | +0.001s |
| e3r2 | ref 4.234→**4.713s**（+11.3%） | **0.898** | 27.775s | **0.9895**（慢 1.1%） | 0.9829 | −0.029s |

- **e1（最大阶段、占比 48.3%）**：update_actor 慢 37%，其余阶段均在 ±0.6% 内（噪声）；整步实测 0.8494 与单阶段预测 0.8480 相差 0.2%——单通道恒等式在同栈得到精确验证。
- **e2**：old_log_prob 慢 26.6%；非目标阶段 ref +2.5%、update_weights +5.2%（均在步间噪声量级，不作系统归因）；整步 0.9586 中约 3.6 个百分点由目标阶段解释。
- **e3r2**：ref 慢 11.3%；update_actor −1.0% 部分抵消，整步仅慢 1.1%。（e3 首跑 14 步半写数据曾给 ref 4.642s／0.9946，与完整 15 步 e3r2 的 4.713s／0.9895 差 1.5%，在步间噪声 sd≈0.8s 内，方向一致。）
- **峰值 reserved 三模式均 21.18–21.25GB（与基线相同）**—— slowdown 不是显存压力造成的；机制（更大 micro-batch 的 attention/前向效率）未做 profiling，只记实测事实。
- 单次运行、seed=0（本栈已知跨进程非 bit 确定）；步级均值 n=12，步间 sd≈0.8s，表中 >5% 的差异超出噪声。

## 执行记录与失败

并行会话 2026-10-31 10:10–10:33 UTC 首次执行三模式（`raw/rollout-g8-candidate-e-chain.out`）：

- **e1 首跑**：0 步完成，`torch.AcceleratorError: CUDA error: invalid argument`，栈在 FSDP `init_flat_param_attributes` 的 `pin_memory()`（首个 FSDP 前向）。原样重跑（仅 checkpoint/日志路径改到 /tmp）**15/15 步成功** → 判定为**环境瞬时错误，非杠杆属性**；不把单次启动失败归因于 token 上限。
- **e2**：15/15 步完成（exit=0），结束阶段 DataLoader worker 被 signal Killed（与候选 A 相同的退出清理现象，不影响已输出 step）。
- **e3**：14/15 步完成，第 15 步 `Saving checkpoint ...` 后崩溃、日志被截断；`/home` 当时已 0 字节可用（e2 的 9.1G checkpoint＋e3 的 6.8G 半写 checkpoint 叠加），判定为**磁盘耗尽中断**，非训练/杠杆失败。**本批（第四十二批）以 e3r2 补跑完整 15 步**（checkpoint/日志落 `/tmp`，13:59–14:09 UTC，exit=0），`raw/rollout-g8-candidate-e-e3r2.txt`。

## 磁盘操作

- `/home` 在 e2/e3 结束后 100% 满（0 字节可用）；e3 的 `global_step_15` 只有 2 个文件、缺 `latest_checkpointed_iteration.txt`，是无法加载的**半写 checkpoint**，本轮已删除（释放 6.8G，`/home` 余 6.8G）。
- e2 完整 checkpoint（9.1G）与候选 A checkpoint（9.1G）保留；e1 补跑 checkpoint（9.1G）写在 `/tmp/rollout-g8-checkpoints-e-e1r2`（根分区 288G 余），避免再次占满 `/home`。

## 复现

```bash
# 服务器侧（脚本在 /home/zxw/venvs/，模型/数据已就位）
MODE=e1 bash /home/zxw/venvs/run-candidate-e.sh     # 原始三模式脚本（checkpoint 落 /home）
bash /home/zxw/venvs/run-candidate-e-e1r2.sh        # e1 补跑（checkpoint/日志落 /tmp）
bash /home/zxw/venvs/run-candidate-e-e3r2.sh        # e3 补跑（checkpoint/日志落 /tmp，第四十二批）
# 解析（本地，需先取回日志；steps 3–14 窗口）
python3 parse-candidate-e.py <base.txt> <e1.txt> <e2.txt> <e3.txt>
```

`summary.json` 含各模式的阶段均值、G、单阶段投影与恒等重建残差；`raw/md5sums.txt` 为本地归档校验和（与服务器逐文件一致）。

## 边界

- 单栈单点（1.5B／单 4090／gsm8k 短答案）；不跨栈外推，不改 §13.25 的恒等式结论，只给它补一个同栈更新侧实例。
- 每个模式单次运行、未扫中间 cap 值（如 4096/6144）；因此「上限越高越慢」只在 3000→8192、4096→16384、8192→16384 三个点上成立。
- 只测阶段墙钟，不解释内核级原因；e1/e2/e3 的验证等级为真实 GPU 运行，不是论文复现（复现计数口径仍待用户确认）。
- 负项入 §13.16 G9 引用库（更新侧批处理上限杠杆）。
