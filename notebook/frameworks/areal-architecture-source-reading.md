# AReaL 架构与统一损失系统源码阅读

> 2026-10-20 复核更正：§10.3–10.6 的「prompt_mean 独有切分放大」「dummy 额外膨胀」与 Megatron 额外 P 因子的判断撤回，历史推导仅用于追溯，当前结论以 §10.7 为准。改变外层 optimizer step 数与改变单步内梯度累积切分，必须分别验证。

> AReaL（蚂蚁系，`areal-project/AReaL`）：支持 GRPO/PPO/GSPO/SAPO/DAPO/M2PO 等众多 RL 算法、Megatron/FSDP 双训练后端的 RL 训练框架。阅读基线：上游 main `3b93679`（2026-10-20 浅克隆 depth 200）。聚焦 v1 训练栈（`rl_trainer` → `trainer/ppo` → `engine`）与统一损失/优势/归约设计；v2 服务架构为并行控制器层。

## 1. 架构总览

```
┌──────────────────────────── areal (v1 训练栈) ───────────────────────────┐
│                                                                         │
│  trainer/rl_trainer.py (2276 行)  ← 单一 RL 训练入口，按 _version 分支    │
│    └─ trainer/ppo/  PPOActor / PPOActorController / 优势估计            │
│    └─ engine/       MegatronEngine / FSDPEngine（共享 core/ 工具层）      │
│    └─ dataset/      数据集 loader 注册表（sft/rl/rw/dpo 类型）            │
│    └─ infra/        launcher / scheduler / controller / rpc / sandbox    │
│    └─ v2/           （并行服务化控制器层，见 §7）                          │
│    └─ api/          TrainEngine/InferenceEngine 抽象 + 配置 dataclass    │
└─────────────────────────────────────────────────────────────────────────┘
```

- **入口**：`rl_trainer.py` 一个文件统管所有算法（GRPO/PPO/GSPO/SAPO/CISPO/M2PO/DAPO/REINFORCE++/RLOO 等），按 `actor._version`（默认 `v1`）切换控制器实现；`config.actor._version == "v2"` 走 v2 控制器。
- **后端抽象**：`api/engine_api.py` 定义 `TrainEngine`/`InferenceEngine` 协议；`MegatronEngine`（`megatron_engine.py`，4.6k 行，MCore 张量/流水线并行 + MCore optimizer）与 `FSDPEngine`（`fsdp_engine.py`，2.6k 行，FSDP2 + DCP + PEFT/LoRA）分别实现。推理后端 vLLM/SGLang 走 `vllm_remote.py`/`sglang_remote.py`（外启 server，含 `vllm_ext/` 源补丁与 `_sglang_source_patch.py`）。
- **可移植性**：`infra/platforms/current_platform` 是 CUDA/ROCm/NPU(HCCL)/CPU 的单例代理，lazy 初始化，不 import 即触发 CUDA init；AReaL 在 910B4 NPU 上运行即靠此抽象（相关：tracker 的 #1786 Megatron 直调 `torch.cuda.*` 致 NPU 崩溃，正被 #1787 修复为用 current_platform 抽象）。

## 2. 核心文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `trainer/rl_trainer.py` | 2276 | 单一 RL 训练入口，编排 rollout/advantage/update |
| `trainer/ppo/actor.py` | 1725 | `PPOActor` + `grpo_loss_fn` 统一损失 + M2PO 掩码 |
| `trainer/ppo/advantage.py` / `gae.py` | 555 / 267 | 优势估计（token/turn GAE、GVPO 等整形） |
| `trainer/ppo/advantage_algorithms.py` | 220 | token_gae / turn_gae / gvpo / process_weighted |
| `trainer/ppo/loss_reduction.py` | 201 | 准备 batch 元数据，绑定归约（token/seq/prompt/constant） |
| `utils/functional/loss_aggregation.py` | 229 | 归约实现 + 匹配 engine weight |
| `utils/functional/functional.py` | — | `ppo_actor_loss_fn`/`sapo_loss_fn`/`cispo_loss_fn` |
| `engine/core/train_engine.py` | 252 | 跨后端共享的 stateless 工具（损失归一化等） |
| `api/loss.py` | 57 | `named_loss_normalizer` 命名损失机制 |
| `engine/megatron_engine.py` / `fsdp_engine.py` | 4649 / 2612 | 两训练后端 |

## 3. 统一损失函数（`grpo_loss_fn`，actor.py:857）

`grpo_loss_fn` 是一个函数把所有算法聚合到同一 reducer。核心参数切分算法：

| 开关 | 效果 |
|------|------|
| `use_cispo_loss` | MiniMax-M1 式剪裁 IS 权重（stop-grad 保留每个 token 的 `log π_θ` 梯度） |
| `use_sapo_loss` | 非对称 sigmoid 门（`τ_pos`/`τ_neg`，默认 1.0/1.05） |
| 默认 | `ppo_actor_loss_fn`（PPO/GRPO ratio 剪裁 + GSPO sequence 几何均值） |
| `m2_threshold` | M2PO 二阶矩掩码（§5） |
| `rejection_sampling` | 陈旧性过滤 / off-policy 修正（behave 重要性权重） |
| `loss_step` | 绑定归约（token_mean 等），`None` 走默认 TokenMean |
| MOPD | `mopd_loss_config`/teacher_logp 多教师蒸馏项 |

三种 surrogate 的实现（`functional.py`）：

- **PPO**：`ratio = exp(logprobs − proximal_logprobs)`；`clipped_ratio = clamp(ratio, 1−ε, 1+ε_high)`；`pg_loss = max(pg_loss1, pg_loss2)`；可选 `c_clip` 双剪裁（`min(pg_loss, sign(A)*c*A)`）。`importance_sampling_level="sequence"` 时走 GSPO 几何均值。
- **SAPO**：`gate_pos = σ(τ_pos·(ratio−1))`，`gate_neg = σ(τ_neg·(ratio−1))`，`scale = 4/τ`，`soft_gate = where(A>0, scaled_gate_pos, scaled_gate_neg)`，`loss = −soft_gate·A`。与 GMPO/GSPO/SAPO 论文精读对照一致（门在 ratio 空间、非对称温度、4/τ 梯度归一）。
- **CISPO**：`ratio_clipped = clamp(ratio, 1−ε, 1+ε_high).detach()`；`loss = −ratio_clipped·A·logprobs`——stop-grad 剪裁权重，梯度只经过 `logprobs`。优点永不剪裁。

## 4. 归约系统与「batch 切分不变性」机制（核心）

AReaL 的归约设计是整个框架最值得学的地方，也是跨框架对照 #1730/#7913 的关键：

**4 种归约模式**（`loss_reduction.py` + `loss_aggregation.py`）：`token_mean` / `seq_mean` / `prompt_mean` / `constant`。每个 reduction 都配对 `normalizer()`（engine weight），引擎按 `sum(local_loss*local_weight)/sum(local_weight)` 跨 microbatch 组合——**这一组合公式本身就是 microbatch 不变的**。

- **`TokenMean`**：`sum(loss)/valid_token_count`。
- **`SequenceMean`**：每 response 内 token 均值后按活跃 response 数加权（`_reduce_unit_means`）。
- **`PromptMean`**：一个 prompt 若横跨多个 optimizer step，用原始 full-group token 系数（`step_scale = len(microbatches)/active_groups`，`engine_weight = 本 mb 的 share`），引擎权重抵消局部除法——**完整组系数跨 step 保持**。`_PromptMeanBatch.for_steps` 在 DP 上 `all_reduce` active group 计数与每 mb response 计数（排除 CP replica）。
- **`ConstantLength`**：`sum(loss)/(active_sequences * divisor)`。

**关键含义**：`prepare_policy_gradient_batch` 在 response 切分**之前**准备 loss 元数据，`prompt_mean` 的 token 系数随 batch 一起切分和 packing 传输（`PG_TOKEN_WEIGHTS` 键），并在 step 确定后用 DP all_reduce 固定 K/G 与每 mb 响应占比。这是对 verl #7913 / AReaL #1730 那类「microbatch 切分不应改变 update」问题在**框架层**的显式回应。

## 5. M2PO 掩码与 #1730 的直接关联

`_apply_m2po_masking`（actor.py:1376）：对 `delta = old_logp − prox_logp`，`m2 = delta²`，取 loss_mask 内 token 的 m2 **按降序排序**，找「后缀均值首次 < threshold」的切分点 `k`，把最大的 k 个 token 掩掉。

**跨框架观察（对接 #1730）**：掩码阈值作用在**每个 microbatch 局部的 m2 后缀均值**上（`sorted_m2` 是本地 batch 的值），因此**不同的 microbatch 切分 → 不同的局部排序与后缀均值 → 掩掉的 token 集合不同**。这正是 AReaL #1730 说的「M2PO token 选择/损失缩放随 microbatch 切分变化」。issue 的理想方案是「select once over full batch」，而 PR #1732 目前做成 opt-in 且仅 DP1 缓存近似 logprob——**selection scope 未解决**。相比之下，verl #7913 的 geo_mean（我方已深挖，上游 PR #7916 走配置层强制）与本框架的 PromptMean 归一化才是 batch-wide 处理；M2PO 的 local-threshold 是同类问题在掩码侧的另一实例。

## 6. 优势估计

`advantage_algorithms.py`：`token_gae`（token 级 GAE，KL 并入 actor 与 critic target）与 `turn_gae`（turn 级 GAE，token KL 只加 actor advantage，不进 critic target，`turn_ids` 由 rollout 提供）。两种过程整形：`gvpo`（负过程信号标记失败，`negative_scale`/`zero_penalty`）与 `process_weighted`（过程奖励要求 `[0,1]` 区间，按符号加权）。`compute_gae_lambda` 支持 scalar/tensor/callable lambda（按 token/turn 时间步）。

## 7. v2 服务化架构（并行控制器层）

`areal/v2/`（165 个 py）是 v1 的**并行服务化重写，非默认、非 drop-in**：选择靠 `_version` 配置（默认 `v1`）。每服务为 Gateway → Router → DataProxy → Worker 的 HTTP 微服务拓扑；`RolloutControllerV2`/`GatewayTrainController` 与 v1 控制器 duck-type 兼容（非子类）。`update_weights` 走 `WeightUpdateController`（HTTP gateway），mode 为 `disk`/`xccl`/`awex`；v2 gateway 只支持 `awex`（NCCL 内存传参，`engine/awex/`）与 `disk`，v2 强制 LoRA→disk、非 LoRA→awex。ROADMAP 列 v2 为进行中方向，但代码默认仍是 v1。

## 8. 数据集层：AReaL 不截断，只过滤（对照 verl #8281）

`areal/dataset/` 是 per-dataset loader 注册表（sft/rl/rw/dpo 类型）。**全仓 grep 无 `truncation_side` / `truncation_strategy` / `filter_overlong_prompts`**——AReaL 在数据集层**不应用任何左/右/中间截断策略**，超长样本直接**丢弃**（如 `gsm8k.py` 按 `max_length` 过滤）。RL 数据集返回未 tokenize 的 `messages`，tokenize/chat template 在数据 worker 侧做。

**跨框架对照（对接 verl #8281）**：verl 的 `data.truncation`（error/left/right/middle）在 `RLHFDataset` 里是**空契约**（读了字段但从未执行），agent_loop 硬编码左截断；而 AReaL 干脆只过滤不截断。两种框架都规避了「真截断」，但 verl 的配置项名不副实（值在契约层，行为在过滤层），AReaL 则文档与实现一致（filter 语义）。这为 verl #8281 的「契约澄清 vs 实现补全」决策提供了另一框架的实现参照。

## 9. 方法论与后续

- **方法**：从 `rl_trainer` 入口往下追 → `actor.py` 损失 → `functional.py` 具体 surrogate → `loss_aggregation` 归约；v2/infra/dataset/api 用并行搜索代笔补齐。损失与归约是 AReaL 的正确性核心，后端（Megatron/FSDP）只是执行载体。
- **对本项目价值**：a) 补上 frameworks 索引 AReaL 空白；b) 直接为 AReaL #1730 提供源码级依据（M2PO 掩码 local-threshold 是 batch 切分依赖的根源）；c) 为 verl #8281 提供「不截断只过滤」的另一实现参照；d) 与 GMPO/GSPO/SAPO 论文精读的 clip/门语义在 AReaL 源码里得到实现级印证。
- **验证范围**：纯源码阅读（`3b93679` 基线），未运行 AReaL（本机无 ray/GPU/torch 全量依赖）；所有函数/行为引用均已核对该行。跨框架对照基于 verl 本地源码（前述候选）与 AReaL #1730/#1732 issue+PR（GitHub API）。
- **未读**：`engine/megatron_engine.py` 内部（重载点、mbridge、Megatron 专属 packing，forward 段已在 §10.5 读过 `train_batch`/`_compute_logprobs_and_loss`/`_prepare_mb_list` 归一化链）、v2 各 service 控制器内部。`fsdp_engine.py` 的 batch 切分已读（§10）；`rl_trainer.py` 主循环已读至 update/权重发布闭环（§10.3.2）。如需深入可续。

## 10. FSDP2 后端的 batch 切分与 loss 归一化（2026-10-20 补读）

承接「归约系统」一节，读 `fsdp_engine.py`（FSDP2 后端）如何落地「microbatch 切分」与 loss 归一化，以及发现的 `prompt_mean` 切分敏感点。

### 10.1 双视图切分：`orig_mb` vs `padded_mb`

`_prepare_mb_list`（fsdp_engine.py:2000）把一次 `train_batch` 的输入切成 `MicroBatchList`（`split_padded_tensor_dict_into_mb_list` → `pack_tensor_dict` → `pad_mb_list` → `unsqueeze`）。迭代时每个 `MicroBatchItem`（data.py:723）持有两个视图：

- **`orig_mb`**（`mbs[i]`）：原始 mb，供 loss 归约/`loss_weight_fn`/context 消费——**`_pg_token_weights` 在这里保留**。
- **`padded_mb`**（`padded_mbs[i]`）：padding 后的 mb，供模型 forward。`_prepare_mb_inputs`（fsdp_engine.py:2124）在传给模型前把 `_pg_token_weights`、`turn_ids`、`is_truncated` **pop 掉**（L2173-2175）——token 系数是归约专用数据，不进 forward。

`MicroBatchSpec`（cli_args.py:108）按 `n_mbs`/`granularity`/`max_tokens_per_mb` 切分，**按行序列切分、不保证组边界**。Ulysses SP（`sp_size>1`）时走 `_prepare_mb_inputs` 的 `ulysses_pad_and_slice_inputs`。

### 10.2 跨后端共享的归一化链（train_engine.py）

`train_batch`（fsdp_engine.py:855）流程：`optimizer_zero_grad` → `_prepare_mb_list` → `compute_total_loss_weight` → forward-backward（每 mb `_compute_logprobs_and_loss`）→ `optimizer_step`。归一化（train_engine.py）：

- `compute_microbatch_loss_weight`（L36）：对每个 mb 调 `loss_weight_fn(mb)`，得 `local_weight`（scalar 或命名 dict）。
- `compute_total_loss_weight`（L62）：`Σ_mb local_weight`，再 **DP all_reduce**。
- `normalize_loss`（L111）：scalar 路径 = `loss · local_weight/total_weight · multiplier`；命名路径逐项归一后求和。**FSDP train_batch 传入 `multiplier = parallel_helper.dp_size`（fsdp_engine.py:887）**——所有归约统一乘 DP 大小（框架「DP 平均梯度 vs 求和」的全局选择，非 prompt_mean 专属，也不随 minibatch 数变）。

对 `prompt_mean`，`loss_weight_fn = loss_step.loss_weight = PromptMean.normalizer() = engine_weight`。**注意 `engine_weight = rows/global_rows` 中 `global_rows` 是「该 minibatch 在 DP 全局的 response 数」，`rows` 是本 rank 该 mb 的 response 数（loss_reduction.py:200-201）——是 DP 归一化权重，不是「跨所有 mb 的 share」**。`compute_total_loss_weight` 只在**单个 minibatch 的 train_batch 内**对 engine 再切的细 mb 求和：单 rank `Σ_j rows_j/global_rows_i = R_i/global_rows_i`，DP all_reduce 后 `Σ_rank R_i^rank/global_rows_i = 1`，故 **`total_weight ≡ 1`**。归一化退化为 `Σ_mb loss_mb · engine_weight · dp_size`，其中 `loss_mb = PromptMean.aggregate = (Σ_t loss_t·w_t)·step_scale / engine_weight`，engine_weight 消去后每 mb 贡献 `step_scale·dp_size·Σ_t loss_t·w_t`。

> **核验提醒**：若误读 `global_rows` 为「所有 mb 的 response 总数」（而非「该 mb 的 DP 全局数」），会误推 `total_weight = N`、与 `step_scale` 的 N 相消、得出「无切分敏感」的错误结论——这是最容易翻车的一步。逐行核对 loss_reduction.py:94-123 与 train_engine.py:62-97 后确认：`total_weight ≡ 1`，N 不抵消。

### 10.3 发现：`prompt_mean` 的 `step_scale = N/G` 切分敏感点

`_PromptMeanBatch.for_steps`（loss_reduction.py:115）设 `step_scale = len(microbatches)/active_groups = M/G`，其中 **M = 传入 `for_steps` 的 mb 数**。这个 M 是 `split_training_batch_into_microbatches(data, n_mbs=ppo_n_minibatches)`（actor.py:590）**经 DP 同步后的有效 slot 数 `effective_n_mbs`（data.py:1144-1147）**——在 DP=1 时等于 `ppo_n_minibatches`，但 DP>1 时可能更大/更小，且**含 transport-dummy slot**（dummy 的 engine_weight=0、无梯度，但仍计入 M 放大 `step_scale` 分母）。G = 完整 rollout batch 的有效组数。actor.py:609-636 对每个 minibatch 依次 `train_batch`（含 `optimizer_step`）——**M 个 minibatch = M 次独立梯度更新**。

**确证事实（忠实源码归一化链 + numpy 复刻，`verify_areal_prompt_mean_split.py`）**：

- 完整梯度链 = `Σ_mb normalize_loss(PromptMean.aggregate, engine_weight, total_weight=1, dp_size)` → `step_scale·dp_size·Σ_mb loss_t·w_t` 逐 mb，累计 = **`dp_size·(M/G)·Σ_all loss_t·w_t`**。在固定 DP 下，**随 M（minibatch 数）线性放大**（numpy 实测：M=1/2/3/4/6 得 1×/2×/3×/4×/6×，与组边界如何切割无关）。
- 当 M=1 时恰等于「per-prompt-group token 均值再平均」（`test_policy_gradient_reduction_matches_definition` 期望 `33/5`，与 `(1/G)·Σ` 吻合 ✓）。
- 测试 `test_distributed_prompt_fixed_scale_preserves_frozen_objective[requested_steps∈{2,6}]`（test_prompt_mean_actor.py:350）只断言 **M 个 step 梯度的 `mean()`** 恢复 full-batch prompt_mean（`oracle[0]/oracle[1]`），**未断言实际训练里 M 次依次 `optimizer_step` 的累加语义**，也未比较「不同 M 下 update 结果是否一致」。

### 10.3.1 精确化：transport-dummy slot 会进一步膨胀 `step_scale` 的 M（2026-10-20 补）

`step_scale = len(microbatches)/active_groups` 里的 `len(microbatches) = effective_n_mbs`（data.py:1144-1147）。当 DP 同步需要填充 **transport-dummy slot**（某 rank 数据不足、或 `effective_n_mbs > 真实有响应的 mb 数`）时：

- dummy 的 `loss_mask` 被 `zero_()`（make_transport_dummy，data.py:886-887）→ `PromptMean.aggregate` 分子 = 0（`torch.where(mask, loss, 0)` 全 0）；
- dummy 的 `engine_weight = 0/global_rows = 0`（loss_reduction.py:200）→ 对 `total_weight`（SUM，仍 ≡1）无贡献；
- 但 **dummy 仍计入 `step_scale` 的 M**（loss_reduction.py:115 的 `len(microbatches)` 含 dummy slot）。

因此**每个 dummy slot 把 `step_scale` 的 M 抬升一个单位、却不贡献任何梯度**，等价于真实响应梯度的额外一个「虚拟 minibatch」放大因子。numpy 复刻（新增 dummy 用例，`verify_areal_prompt_mean_split.py` Case 4）实测：`real_mbs=1 + 1 dummy → 2.0×`、`real_mbs=2 + 1 dummy → 3.0×`、`real_mbs=3 + 1 dummy → 4.0×`（相对 M=1 基准）——比「随 M 线性放大」更精确：**同一次 update 内，只要存在 dummy slot，prompt_mean 梯度就被额外放大 dummy 数量个单位 M**，且与组边界切割无关、token_mean 等不受影响。

**已确证的切分敏感机制（忠实源码归一化链 + numpy 复刻）**：`step_scale=M/G` 用**全局近似**而非 minibatch 内真实组数 `G_mb`。minibatch 按行切分、不保证组边界，故各 mb 含组数不均匀；数值复刻显示 **prompt_mean 的累计梯度 = `dp_size·(M/G)·Σ_all loss_t·w_t` 精确随 M 线性放大（M=1/2/3/4/6 → 1×/2×/3×/4×/6×），与组边界如何切割无关**。这使 **prompt_mean + `ppo_n_minibatches>1`（默认 4）下，update 的梯度绝对尺度随 minibatch 数线性增长**——与 verl #7913 / AReaL #1730 同源的「batch 切分不应改变 update」问题，位于**归约/损失侧**（而非 M2PO 的掩码侧）。上游无此发现的公开讨论（`prompt_mean`/`step_scale` 搜索无对应 issue/PR；`step_scale` 由 PR #1546 引入，其 body 声明的语义是「K=actual steps、G/K 分母、N 步平均恢复 full-batch」，与代码 `M/G` 绑定到 DP 同步 minibatch 数的用法存在语义偏差——代码实际「N 步累加」而非「N 步平均」）。

> **定位与边界**：本节为忠实源码推导 + numpy 数值复刻，**不是 AReaL 真实环境实测**。核心确证：完整归一化链下 prompt_mean 累计梯度随 M 线性放大（`total_weight≡1`、`loss_multiplier=dp_size` 均不抵消 M），token_mean/seq_mean/constant 不变。待实测项收敛为：**该线性放大在真实优化器（Adam 等）下是否构成「切分依赖缺陷」**——绝对梯度放大在自适应优化器下会被归一化吸收一部分，故「放大」≠「一定出错」，是否构成 defect 需在 AReaL 环境用 `ppo_n_minibatches` 2 vs 4 对照观察实际 update 差异。定性为「切分敏感点（高置信源码级确证，缺陷定性待实测）」，非已证实 bug。

### 10.3.2 调用链闭环（2026-10-20 补）：ppo_update 每 global_step 触发，M 次累加更新后一次 update_weights

rl_trainer 主循环（rl_trainer.py:929）每 global_step：rollout → aux 打分（critic/ref/teacher）→ advantage → **`self.actor.ppo_update(adv_batch)`（L1155）** → **`update_weights`（L1232，发布新版本权重）**。`ppo_update`（actor.py:569-636）正是 §10.3 的路径：`prepare_policy_gradient_batch → split_training_batch_into_microbatches(ppo_n_minibatches) → for_steps(mb_inputs)`，随后对 M 个 minibatch 依次 `engine.train_batch`（每个含一次 `optimizer_step`）。

因此切分敏感点的实际影响范围是确定的：**每次 global_step 的 `ppo_update` 内，prompt_mean 梯度被 `step_scale = M/G` 放大（M=effective_n_mbs，可能因 dummy 额外膨胀），M 次累加更新后的权重经 `update_weights` 发布到推理**——放大进入了每次发布给 rollout 的权重。默认 `ppo_n_minibatches=4` 时 M/G≈4×的绝对梯度，是否在真实自适应优化器下构成跨配置不一致的 update，是 §10.3 待实测项。

### 10.4 与其他归约对照

`token_mean`（无 step 因子，`Σ loss_t/Σ mask_t`，天然可加、切分不变）与 `seq_mean`（per-response 均值后平均，`_reduce_unit_means`）**不带 step_scale**，不受此影响。`constant` 为固定 divisor。**只有 `prompt_mean` 带 `step_scale=M/G`**，是唯一随 minibatch 数变化的归约模式。

### 10.5 Megatron 后端对照：`step_scale=M/G` 跨后端同源 + 额外常数因子 `len(mb_list)`（2026-10-20 补）

把 §10.3 的切分敏感点从「仅 FSDP2 后端确证」升级为「**跨后端（FSDP2 + Megatron）同源**」：`_PromptMeanBatch.for_steps`（loss_reduction.py:115）由 actor.py:600 在 `ppo_update` 中调用，**在后端之上（actor 层）共享**——无论 FSDP2 还是 Megatron，`step_scale = len(mb_inputs)/active_groups = M/G` 都进入 `PromptMean.aggregate`（loss_aggregation.py:108）。因此 §10.3 的「prompt_mean 随 ppo_n_minibatches（M）线性放大」在**两个后端同样成立**。

**Megatron 特有的归一化差异**（`megatron_engine.py::train_batch`，非 per_token_loss 路径）：

- `loss_multiplier = mpu.get_data_parallel_world_size() * len(mb_list)`（L2382），随后 `normalize_loss(loss, local_weight, total_weight, loss_multiplier)`（L4328）。这里的 `mb_list` 是 **这一个 ppo-minibatch 内部**再切成 P 个 pipeline micro-batch（`_prepare_mb_list`，`len(mb_list)=P`，P 由 `mb_spec.n_mbs` / `2*pp_size` 决定），**不是**外层 ppo 切分数 M。
- 忠实链：每 pipeline mb 的 `PromptMean.aggregate = (Σ_t loss·w)_p·step_scale/engine_weight_p`，`normalize_loss` 后 engine_weight_p 相消、total_weight≡1，得 `(Σ_t loss·w)_p·step_scale·dp_size·P`；P 个 pipeline mb 求和（序列可加）→ `step_scale·dp_size·P·Σ_{该ppo-minibatch} loss·w`；M 个 ppo-minibatch 累加 → **`dp_size·(M/G)·P·Σ_all loss·w`**。

**numpy 复刻验证**（`verify_areal_prompt_mean_split.py` Case 5，`megatron_prompt_mean_total`）：M=1/2/4 → 1×/2×/4× 线性放大（与 FSDP2 同源）；固定 n_mbs=4 下 P=2/P=1 → 2.000 精确。

**结论与边界**：
- 核心切分敏感点（随 ppo_n_minibatches 线性放大）**跨后端一致**：`step_scale=M/G` 是 actor 层共享的，与 FSDP2 §10.3 同源，与 verl #7913 / AReaL #1730 同属「batch 切分不应改变 update」问题。
- Megatron 的 `loss_multiplier` 额外乘 `len(mb_list)=P` 是**内层常数因子**（由 pp_size / mb_spec 决定，不随 ppo_n_minibatches 变），只改变绝对尺度、不构成新的 ppo 切分敏感项；它对 token_mean/seq_mean/constant 也统一生效（补偿 MCore 的 per-microbatch 梯度平均）。
- 仍是「切分敏感点（高置信源码级确证，缺陷定性待实测）」，非已证实 bug：真实自适应优化器是否会吸收这部分绝对梯度差异，需 AReaL 环境 `ppo_n_minibatches` 2 vs 4 对照实测（FSDP2 与 Megatron 各一组更佳）。

### 10.6 Megatron 归一化实际有三条路径 + per-token 路径不抵消切分敏感（2026-10-20 补读）

补读 §9「未读」的 `megatron_engine.py::train_batch`（L2325-2414）时发现：**Megatron 归一化实际是三分支**，§10.5 只覆盖了默认分支。`train_batch` 在 Step 2 依模型配置 `model_config.calculate_per_token_loss` 分流（L2348-2388）：

1. **`per_token_loss=True` 且 loss_weight_fn 是 named normalizer**（`is_named_loss_normalizer`，api/loss.py:56 查 `_areal_named_loss_names`）：`loss_multiplier=1.0`，`total_token_count=compute_global_token_count(...)`，靠 `scale_named_loss_for_native` 逐 term 保自身分母。
2. **`per_token_loss=True` 但非 named**（`elif per_token_loss`，L2364-2369）：`loss_multiplier = self._per_token_loss_multiplier(mb_list, loss_weight_fn) = T_global / W_global`（per-token 数对 weight 的 DP all-reduce 均值，L2414-2430），`total_loss_weight=None`。
3. **`per_token_loss=False`（默认）**（L2382-2388）：`loss_multiplier = dp_size * len(mb_list)`，`total_loss_weight=compute_total_loss_weight(...)`——**这就是 §10.5 分析的分支**。

**关键判定**：`PromptMean` 的 `normalizer()`/`engine_weight` 返回的是**标量 tensor（rows/global_rows）而非命名 dict**，`loss_reduction.py` 中 `_bind_prompt_mean`/`PromptMean` **未用 `@named_loss_normalizer` 装饰** → `is_named_loss_normalizer(loss_weight_fn)==False` 恒成立。因此 prompt_mean 在 Megatron 只可能走**分支 2 或 3**，从不走分支 1。

**Megatron-Core 0.19.0 默认 `calculate_per_token_loss: bool = False`**（`megatron/core/transformer/transformer_config.py:277`，已按 `core_v0.19.0` tag 核验；AReaL 全仓不设该字段，pyproject 锁 `megatron-core==0.19.0`）→ **AReaL 实际默认路径就是 §10.5 的分支 3**（`dp_size·(M/G)·P·Σ_all`），§10.5 结论对默认配置成立。

**per-token 路径（分支 2）不抵消切分敏感**（忠实推导 + numpy 复刻，`verify_areal_prompt_mean_split.py` Case 6）：
- `_compute_logprobs_and_loss`（L4106）对 prompt_mean（非 named）取 `token_count = loss_mask.count_nonzero()`，`_build_per_token_loss_output`（L4330）得 `loss_numerator = loss·engine_weight·(T_global/W_global)/cp_size`；MCore `finalize_model_grads` 把 `loss_numerator` 在 DP+CP 求和后 **÷总 token 数 T_global**。
- 代入 `loss = (Σ_t loss·w)·step_scale/engine_weight`：`loss_numerator = (Σ_t loss·w)·step_scale·(T_global/W_global)/cp_size`，`engine_weight` 相消；`/T_global` 与分子里的 `T_global` **精确相消**，每 ppo-minibatch 贡献 `(Σ_t loss·w)·step_scale / W_global_i`，其中 `W_global_i`（内层 pipeline mb 的 engine_weight 之和，DP all-reduce）≡ 1（同 §10.2 的 total_weight）。
- **忠实链 = `(M/G)·Σ_all loss·w`**——M 线性放大**保留**，且 `dp_size` 与 `P`（内层 pipeline mb 数）这两个常数因子**被 /T_global 消掉**（numpy 复刻 Case 6：M=1/2/4→1×/2×/4× 精确，P=2/P=1→1.000）。

**结论（升级 §10.5 缺口）**：prompt_mean 的 `step_scale=M/G` 线性放大是 **actor 层共享、跨三条归一化路径稳健**——FSDP2（`dp_size·(M/G)`）、Megatron 默认（`dp_size·(M/G)·P`）、Megatron per-token（`(M/G)`）都一样，**不是**特定 `dp_size·len(mb_list)` 乘子的产物。per-token 路径只是把常数因子换掉，切分敏感（随 M 线性）不变。实测项仍不变：真实自适应优化器是否吸收这部分绝对梯度差异，需 AReaL 环境 `ppo_n_minibatches` 2 vs 4 对照。

### 10.7 对抗复核：外层优化步数与内层累积不能混同（2026-10-20 更正）

**当前判断：尚无证据证明 prompt_mean 的 M/G 是缺陷，撤回此前“高置信缺陷机制”的表述。** 用户指定的研究方向继续推进，但须先修正实测判据。源码基线仍为 AReaL `3b93679888e9f4929f6f155f13c6a286a62dfd72`；本轮重新读取 loss_reduction.py、loss_aggregation.py、actor.py、data.py、FSDP train_batch 与上游测试。

1. **旧脚本比较了不同归一化口径。** `token_mean_total` 始终计算全 rollout 一次总均值，从未按 M 切分；真实 TokenMean 每次 train_batch 都用该优化步自己的 token 总数作分母。均匀组、等长响应、等大小 minibatch 下，prompt_mean 与 token_mean 的每一步梯度完全相同，二者冻结参数后的梯度和都随 M 增长，非 prompt_mean 独有现象。
2. **M 是优化步数，P 是单步内累积切分。** actor 每次 train_batch 更新参数，后续梯度在新参数处计算，故冻结参数的梯度和不是实际训练 update。改变 M=2/4 后权重不同，单独不能证明 reducer 有问题。上游 [PR #1546](https://github.com/areal-project/AReaL/pull/1546) 约定冻结参数和 mask 后跨步平均梯度恢复 full-batch prompt mean，未要求 M 次更新后参数相同；原测试的 mean() 与该契约一致。
3. **本地 dummy 不等于额外全局空步。** data.py 的同步调度按各 rank 具体 minibatch 数与 offset/modulo 分配，避免全局 all-dummy 步；for_steps 在 DP 汇总行数后还会拒绝任何全局 rows<=0 的步骤。旧 Case 4 在 DP=1 任意追加全局空步，不是合法 AReaL 调度。DP>1 某 rank 的 dummy 对应其他 rank 的真实响应；固定全局 M 时，dummy 本身不额外乘系数。M 的变化仍是全局优化调度变化，不能从本地空位数直接推导“膨胀”。
4. **Megatron 的 P 乘子有调度抵消。** [MCore core_v0.19.0 schedules.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.0/megatron/core/pipeline_parallel/schedules.py#L293-L306) 的 forward_step_calc_loss 对非 per-token 路径除以 num_microbatches。CP=1 时，AReaL 默认路径的 P 乘子与 /P 相消，旧 Case 5 漏掉此层，不能主张 P=2 有额外 2 倍更新。DP 的 loss multiplier 也须结合实际梯度通信归约核实，局部损失公式不能直接当作最终分布式梯度。CP、per-token、MTP/MoE 的最终归约仍待独立验证。

**CPU 反例验证**：[替换后的验证脚本](../../experiments/scripts/verify_areal_prompt_mean_split.py) 仅用标准库，模拟固定参数处的标量 token 导数，不导入或运行 AReaL。43 项数值比较与调度断言通过：均匀例 M=1/2/4，两归约平均梯度均为 4.5，梯度和分别 4.5/9/18；非均匀组下 prompt_mean 跨步均值均为 3.740740741；固定 M=2、P=1/2/4 的标量 FSDP/Megatron 默认路径一致（DP=CP=1）；DP counts=[1,3]/[2,2]/[1,1] 的合法调度无全局空步。零初始 moments 的标量 Adam 参考计算中，M=2/4 参数变化为 -0.019350137/-0.038374539，同一 M 下两归约一致，说明仅凭跨 M 权重差异会误报。以上是参考数值模拟，非真实 GPU 或框架集成结果。

**GPU 实验判据：**

- 先固定外层 M、样本、顺序、mask、group 元数据、参数与优化器状态，仅改变单步内 P；比较最终通信归约后的梯度、clipping 前后范数及一次 Adam 更新。FSDP2 与 Megatron 各做一组，先关闭 MoE/MTP、使用 DP=CP=1，再扩展并行维度。
- 固定 M，改变 DP rank 上数据与 dummy 的分配，使用逐步全局行/token/group 参考验证，不将 dummy 数本身作为缺陷判据。
- 用户指定的 M=2 vs 4 保留为训练行为对照：冻结参数时检查跨步平均目标符合契约，并以 token_mean 为对照；实际顺序更新观察训练轨迹，不直接要求跨 M 参数相同。保存 checkpoint、optimizer moments、有效 M、每步全局分母、P、clipping 与通信配置。

复现：`python3 experiments/scripts/verify_areal_prompt_mean_split.py`。GPU 集成仍等待 VPN/已确认目标环境；不据本轮模拟提上游 issue/PR。

### 10.8 GPU 实测环境就绪 + 修复候选降级为假设（2026-10-20 补，服务器 VPN 通道中断待恢复）

**环境搭建（服务器 4090-1，已完成）**：用户授权优先实测项后，服务器已配好——AReaL 上游 clone `/home/zxw/areal`（基线 `3b93679` 与本文一致），`uv sync --extra cuda` 全新环境（torch 2.9.1+cu129 / transformers 5.3.0 / sglang 0.5.10.post1 / megatron-core / ray 2.55.1，venv 13G）。verl-e2e 克隆方案放弃（verl 环境 torch 2.11 超 AReaL 上限、无 sglang，官方 pin 路径确定性最强）。空闲 GPU 0/1/7（2-6 被其他用户占用 ~22-24G）；smoke 配置 `/home/zxw/areal_smoke_ppo.yaml`（Qwen2.5-0.5B-Instruct + sglang:d1p1t1 + fsdp:d1p1t1，2 卡，1 epoch 小 batch），模型与 gsm8k 数据集已预下载。**启动 smoke 后服务器失联（ping 不通数小时，路由显示该内网地址未走 VPN 隧道）**——GPU 实测闭环待 VPN 恢复后执行。fork `Jackie2049/AReaL` 已创建，产物只落 fork（用户铁律）。

**修复候选降级（依据 §10.7 对抗复核）**：本轮曾起草 `step_scale = M/G → 1/G`（loss_reduction.py:115 一行 + 测试 mean→sum 断言，patch 草稿 [patches/patch-fix-prompt-mean-step-scale.draft](../../open-source/areal/patches/patch-fix-prompt-mean-step-scale.draft)），但 §10.7 对抗复核指出其前提不成立：

- 旧 numpy 口径把 `token_mean` 算成「全 rollout 一次总均值」（从未按 M 切分），而真实 TokenMean 每次 train_batch 用该步自己的 token 总数作分母——均匀组下 prompt_mean 与 token_mean 每步梯度完全相同，二者冻结参数的梯度和都随 M 增长，非 prompt_mean 独有。
- M 是真实优化步数（每步 train_batch 后参数更新，后续梯度在新参数处计算），不是冻结参数累加；改变 M 后权重不同单独不能证明 reducer 有问题。上游 PR #1546 契约是「冻结参数+mask 跨步平均恢复 full-batch」，原测试 mean() 与该契约一致。
- dummy 为本地概念：data.py 按 rank 具体 minibatch 数与 offset/modulo 调度避免全局 all-dummy 步；固定全局 M 时 dummy 不额外乘系数（旧 Case 4 在 DP=1 追加全局空步不是合法调度）。
- Megatron P 乘子有调度抵消：MCore schedules.py forward_step_calc_loss 对非 per-token 路径除以 num_microbatches；CP=1 时 P 乘子与 /P 相消（旧 Case 5 漏掉这层）。

因此 `step_scale=1/G` 是**候选假设**而非既定修复：GPU 实测必须按 §10.7 判据执行（固定 M 只变 P、固定 M 变 dummy 分配、M=2 vs 4 作训练行为对照并以 token_mean 对照），确认「同一 M 下 prompt_mean 与 token_mean 的更新是否真的一致/不一致、跨 M 差异是否超出两种归约共有的训练行为差异」后再定性。受控脚本 [areal_prompt_mean_gpu_probe.py](../../experiments/scripts/areal_prompt_mean_gpu_probe.py) 初稿已备，需按 §10.7 判据调整后再跑。

### 10.9 受控探针前置修整：可编译不等于 GPU 集成通过（2026-10-20）

接续用户授权的实测任务，先核对 `8b625f7` 版本脚本，发现会阻断运行或影响解释的具体问题：

- main 在使用 MODEL 后才声明 global MODEL，`compile(..., "exec")` 实际报 SyntaxError；FinetuneSpec 也不在 cli_args，基线定义位于 api/io_struct.py。
- 初稿只有三条响应，请求 M=4 会被调度缩减，无法形成宣称的 M=2/4 比较；部分外层 minibatch 只有一行，也无法实现请求的内层 P=2。
- 只比较更新范数或跨 M 比率不能判断参数变化方向是否一致；非均匀 prompt/token 是不同目标，不能将其差异直接定为 reducer 缺陷。
- 未从同一初始化策略计算行为 logprob、未显式关闭 dropout，也未确认实际 M/P，可能引入与归约无关的混杂。原先只打印结果，没有持久 JSON 或每步裁剪记录。

以下仅为 M1 离线修整草稿，已保存为 [未应用参考 patch](../../open-source/areal/patches/probe-preflight-m1-review.patch)；按用户明确分工，本机在用探针已恢复至 M5 原稿基线，不继续维护并行代码。AReaL 上游源码与候选 step_scale 均未修改；后续统一由 M5 决定，见 [交接说明](../../open-source/areal/handoff-m5.md)。草稿内容：

1. 使用延迟运行时导入，增加无 torch 的 `--plan --check-source /path/to/AReaL`，按源码 AST 检查六个配置构造调用的字段及当前 Git 版本；修正 FinetuneSpec 导入与 global 编译问题。该检查不导入真实配置类，不替代其运行时校验。
2. 使用 16 响应 fixture，默认 M=2/4、P=1/2，并限制 M×P 整除响应数；实际更新次数与引擎报告的 inner_microbatches 不符则终止。balanced 为四个等大小/等 token 组，可作为两模式等目标控制；uneven 保留为不同目标的行为描述，不要求两模式相同。
3. 每个 case 重载模型，禁用 dropout/TF32；先用相同 P=1 eval 路径算行为 logprob，再设置待测 P。记录初始权重与完整输入摘要，相同比较失配则终止。显式选择已确认空闲的单张 GPU，采用 torchrun/DP=CP=1。
4. 记录完整参数变化向量的 L2/max-abs 差、首步梯度向量差、裁剪前后范数、实际优化步数/P、update_successful；不再用范数比率自动判 defect。CPU 全向量复制会同步设备并增加主机内存开销，**该探针不用于吞吐测量**。
5. GPU 执行要求明确 `--output`，每完成 case 持久 JSON；不接受多进程。当前只准备 FSDP2/BF16 + FlashAttention2 路径；Megatron、DP/dummy 和冻结参数跨步平均契约未纳入此脚本，需另行验证。

CPU 验证仅针对草稿：编译与无依赖导入通过；两个计划、六个非法参数组合、缺失 output/torchrun 共十项通过，六个配置调用字段与 `3b93679888e9f4929f6f155f13c6a286a62dfd72` 一致；原标量归约参考脚本 43 项数值检查仍通过。**尚未真实导入 torch/AReaL 或跑 GPU，引擎、模型、clipping hook 与序列打包仍待 M5 集成核验；不将草稿结果当成当前在用脚本验证。**

历史网络观察（仅 M1 本轮）：scutil 显示 Tailscale Disconnected；TCP/22 检查成功，随后只读 SSH 在 banner exchange 超时，未执行远程命令。本轮未拿到新 smoke 日志/GPU 占用，未上传或启动草稿。**此后不再从 M1 重试；用户确认 GPU 与本候选整体由 M5 负责，不能用 M1 网络状态推断 M5 条件。**

## 相关

- 贡献轨道：[AReaL 贡献追踪](../../open-source/areal/tracker.md)（#1730 M2PO 跨框架对照）
- 跨框架 issue：AReaL [#1730](https://github.com/areal-project/AReaL/issues/1730)（M2PO microbatch）与 verl [#7913](https://github.com/verl-project/verl/issues/7913)（geo_mean microbatch）——同 reporter `gss10282025`
- 论文：GMPO/GSPO/SAPO 精读见 [notebook/papers](readme.md) 与 [RL 损失家族](../rl-policy-loss-functions-mathematical-derivation.md)
- verl 数据契约对照：[candidate-8281](../../open-source/verl/candidate-8281-agent-loop-shared-cap-vs-truncation.md)
