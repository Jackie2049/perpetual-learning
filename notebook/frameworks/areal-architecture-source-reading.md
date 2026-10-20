# AReaL 架构与统一损失系统源码阅读

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
- **未读**：`engine/megatron_engine.py` 内部（重载点、mbridge、Megatron 专属 packing）、`rl_trainer.py` 主循环细节、v2 各 service 控制器内部。`fsdp_engine.py` 的 batch 切分已读（§10）。如需深入可续。

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
- `normalize_loss`（L111）：scalar 路径 = `loss · local_weight/total_weight`；命名路径逐项归一后求和。

对 `prompt_mean`，`loss_weight_fn = loss_step.loss_weight = PromptMean.normalizer() = engine_weight`。因 `Σ_mb engine_weight = Σ_mb rows/global_rows = 1`，故 **`total_weight ≡ 1`**（测试也断言 `total_weight == 1.0`），归一化退化为 `Σ_mb loss_mb · engine_weight`，其中 `loss_mb = PromptMean.aggregate = (Σ_t loss_t·w_t)·step_scale / engine_weight`，engine_weight 消去后每 mb 贡献 `step_scale·Σ_t loss_t·w_t`。

### 10.3 发现：`prompt_mean` 的 `step_scale = N/G` 切分敏感点

`_PromptMeanBatch.for_steps`（loss_reduction.py:115）设 `step_scale = len(microbatches)/active_groups = N/G`，其中 **N = `ppo_n_minibatches`（actor.py:592 传入的 minibatch 数）**，G = 完整 rollout batch 的有效组数。actor.py:609-636 对每个 minibatch 依次 `train_batch`（含 `optimizer_step`）——**N 个 minibatch = N 次独立梯度更新**。

**确证事实（源码 + numpy 复刻，`verify_areal_prompt_mean_split.py`）**：

- `PromptMean.aggregate` 的最终累计梯度 = `(N/G)·Σ_all loss_t·w_t`，**随 N（minibatch 数）线性放大**（numpy 实测：同一数据 N=1/2/4 得 1×/2×/4×）。
- 当 N=1 时恰等于「per-prompt-group token 均值再平均」（`test_policy_gradient_reduction_matches_definition` 期望 `33/5`，与 `(1/G)·Σ` 吻合 ✓）。
- 测试 `test_distributed_prompt_fixed_scale_preserves_frozen_objective[requested_steps∈{2,6}]`（test_prompt_mean_actor.py:350）只断言 **N 个 step 梯度的 `mean()`** 恢复 full-batch prompt_mean（`oracle[0]/oracle[1]`），**未断言实际训练里 N 次依次 `optimizer_step` 的累加语义**，也未比较「不同 N 下 update 结果是否一致」。

**待验证判断（需真实 AReaL 运行，本机无 ray/torch）**：`step_scale=N/G` 用**全局近似**而非 minibatch 内真实组数 `G_mb`。minibatch 按行切分、不保证组边界，故各 mb 含组数不均匀；当 `G_mb ≠ N/G` 时，prompt_mean 的 minibatch 内归一尺度偏离「按真实组数平均」。这使 **prompt_mean + `ppo_n_minibatches>1`（默认 4）下，update 随切分方式变化**——与 verl #7913 / AReaL #1730 同源的「batch 切分不应改变 update」问题，位于**归约/损失侧**（而非 M2PO 的掩码侧）。上游无此发现的公开讨论（`prompt_mean`/`step_scale` 搜索无对应 issue/PR；`step_scale` 由 PR #1546 引入，其 body 声明的语义是「K=actual steps、G/K 分母、N 步平均恢复 full-batch」，与代码 `N/G` 绑定到 `ppo_n_minibatches` 的用法存在语义偏差）。

> **定位与边界**：本节为源码推导 + numpy 数值复刻，**不是 AReaL 真实环境实测**。`step_scale` 的设计意图（跨多 step 的组）与实际数据流（一个 step 覆盖全部 G 组）是否冲突、N 倍在真实优化器累加下是否构成「切分依赖缺陷」，需在 AReaL 环境用 `ppo_n_minibatches` 2 vs 4 对照实测确认。定性为「切分敏感点/待确认」，非已证实 bug。

### 10.4 与其他归约对照

`token_mean`（无 step 因子，`Σ loss_t/Σ mask_t`，天然可加、切分不变）与 `seq_mean`（per-response 均值后平均，`_reduce_unit_means`）**不带 step_scale**，不受此影响。`constant` 为固定 divisor。**只有 `prompt_mean` 带 `step_scale=N/G`**，是唯一随 minibatch 数变化的归约模式。

## 相关

- 贡献轨道：[AReaL 贡献追踪](../../open-source/areal/tracker.md)（#1730 M2PO 跨框架对照）
- 跨框架 issue：AReaL [#1730](https://github.com/areal-project/AReaL/issues/1730)（M2PO microbatch）与 verl [#7913](https://github.com/verl-project/verl/issues/7913)（geo_mean microbatch）——同 reporter `gss10282025`
- 论文：GMPO/GSPO/SAPO 精读见 [notebook/papers](readme.md) 与 [RL 损失家族](../rl-policy-loss-functions-mathematical-derivation.md)
- verl 数据契约对照：[candidate-8281](../../open-source/verl/candidate-8281-agent-loop-shared-cap-vs-truncation.md)
