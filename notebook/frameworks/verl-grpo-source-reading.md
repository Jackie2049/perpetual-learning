# verl GRPO 源码深度阅读

> verl GRPO 实现的深度源码分析
> 关键发现: verl 没有独立的 GRPO trainer, 而是统一的 RayPPOTrainer 通过 `adv_estimator=grpo` 切换

## 1. 统一架构: RayPPOTrainer

verl 不包含独立的 `RayGRPOTrainer`。PPO 和 GRPO (以及 RLOO, REINFORCE++, REMAX, GDPO) 都由**统一的** `RayPPOTrainer` 处理。

**关键文件**:
| 文件 | 行数 | 作用 |
|------|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 1771 | 主训练循环 |
| `verl/trainer/ppo/core_algos.py` | 2488 | 所有优势估计器 + 策略损失 |
| `verl/trainer/config/algorithm.py` | 670 | AlgoConfig 数据类 |

**算法切换**: `algorithm.adv_estimator` 配置:
- `grpo`: GRPO (Group Relative Policy Optimization)
- `gae`: PPO (Generalized Advantage Estimation)
- `rloo`: RLOO (REINFORCE Leave-One-Out)
- `reinforce_plus_plus`: REINFORCE++
- `remax`: ReMax
- `gdpo`: Group DPO

## 2. ActorRolloutRefWorker

**文件**: `verl/workers/engine_workers.py:434`

在一个 Ray worker 中组合三个角色:
- **Actor** (训练): FSDP/Megatron 训练 worker
- **Rollout** (推理): vLLM/SGLang/HF 推理引擎
- **Ref Policy** (可选): KL 惩罚参考模型

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config, role, ...):
        self.role = role  # "actor_rollout_ref"
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
```

**LoRA 优化**: `ref_in_actor=True` 时, ref policy 使用 actor 权重 (不应用 LoRA), 避免加载第二个模型。

## 3. GRPO 训练循环

### Step 1: 批处理准备 (ray_trainer.py:1435)
```python
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))])
rollout_n = self.config.actor_rollout_ref.rollout.n
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```
- 每个 prompt 获得唯一 UUID 作为 uid
- 批处理重复 `rollout_n` 次, **交错排列** (prompt_i 的所有 n 个采样相邻)

### Step 2: 生成 (ray_trainer.py:1467)
```python
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
```
- 使用 vLLM/SGLang 异步 rollout 引擎
- `rollout.n` 控制每个 prompt 的采样数

### Step 3: 奖励计算 (ray_trainer.py:1518)
```python
reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```
- 通过 `extract_reward()` 提取
- 来自奖励模型分数 (`rm_scores`) 或自定义奖励函数
- `rm_scores` shape: `(batch_size, response_length)` — 末尾 token 处有非零奖励

### Step 4: KL 惩罚
两种模式:
- **in-reward KL**: `algorithm.use_kl_in_reward=True` — 从 token 级奖励中减去 β×KL
- **in-loss KL** (GRPO 推荐): `actor.use_kl_loss=True` — 直接作为正则化项加入策略损失

### Step 5: 优势计算 (GRPO 核心)
```python
batch = compute_advantage(batch, adv_estimator="grpo", ...)
```
分派到 GRPO 特定函数 — 见 core_algos.py 分析。

## 4. core_algos.py — GRPO 优势计算

### compute_grpo_process_reward
**关键**: 组内奖励归一化

```python
# 按 uid 分组 (相同 prompt = 同一组)
uid_list = batch.non_tensor_batch['uid'].tolist()
# 对每个组:
for uid in unique_uids:
    group_indices = [i for i, u in enumerate(uid_list) if u == uid]
    group_rewards = rewards[group_indices]
    # 组内归一化
    mean_r = group_rewards.mean()
    std_r = group_rewards.std()
    advantages[group_indices] = (group_rewards - mean_r) / (std_r + eps)
```

### GRPO 策略损失
与 PPO 类似的 clipped surrogate, 但:
- 使用组归一化的 advantages (而非 GAE)
- KL 惩罚在 loss 中 (而非 reward 中)
- 无 value function

```python
# 简化版 GRPO loss
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
loss = policy_loss + kl_coef * kl_divergence
```

## 5. Prefix Sharing in GRPO Rollout

### Rollout 阶段的 Prefix Caching
```python
# 相同 prompt × n 个采样 → vLLM prefix caching
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```
- vLLM 的 `enable_prefix_caching=True` (默认)
- 相同 prompt 复制 n 次 → KV cache 自动复用
- 每 weight update 后 `reset_prefix_cache()`

### Batch 均衡
```python
get_group_balanced_partitions()  # 按 uid 分组
```
- 开启 `use_prefix_grouper` 且 batch 有 UID 时走组级分区；UID 连续、每组固定 `rollout.n` 是保持组完整并等行分发的前提，详见下节
- 训练时 PrefixGrouper (可选) 进一步合并前缀

### 连续 UID 与等样本组的前提

本节是下述固定版本的PrefixGrouper/legacy分区契约；新候选#8201的V1按源UID定义优化mini-batch，是[另一条布局路径](#v1-源-prompt-单位与可变轨迹行)，不应把两者的同卡/同批保证混用。

**组级均衡不是按 UID 全局聚合，也不独立保证每个 rank 的样本数相等。** 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，[`get_group_balanced_partitions`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/seqlen_balancing.py)逐项比较当前 UID 和上一项，仅把相邻相同值组成一个段，再对段的 workload 做 Karmarkar-Karp 分区。相同 UID 若在不相邻的位置重现，会被当作不同段，而非用字典合并。

组 workload 是组内各样本 `24576*L + L²` 的和，不是先把整组长度相加再平方。`equal_size=True` 约束的是每个分区的**段数**。因此调用方需要同时维护三个条件：

- 每个 UID 只占一个连续段；否则同一 UID 可以分散到不同输出分区。
- 每段含相同数量的样本；否则等段数不等于等行数。
- 段数至少为 DP 数，且能被 DP 数整除；原 helper/分区器会对这两项作断言。

原函数的两个 CPU 边界观察说明了区别，不是标准训练路径的故障复现：

| 输入前提 | 原 helper 输出 | 后续解释 |
|---|---|---|
| 长度全1，UID为 `[0,1,0,1]` | 索引分区 `[[0,3],[1,2]]` | 两个UID都跨分区，因输入不连续 |
| 长度全1，UID段大小 `[1,1,1,5]`，2个分区 | `[[0,3,4,5,6,7],[1,2]]` | 每区2段，但样本数为6/2；拼平后等分成4/4行，会切开UID3 |

第二行的4/4切分只是对返回索引的 Python 列表演示，没有运行 DataProto 或远程分发。它揭示复合契约：helper 的输出组完整，不能在任意不同的后续切分边界下继续保持完整。

[`RayPPOTrainer._balance_batch`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py)在 `use_prefix_grouper` 开启且有 UID 时调用该 helper，并检查不同 UID 数能被 DP 数整除；它没有在这里全局排序或校验每个 UID 的行数。之后将分区索引拼平、`batch.reorder`，再由[DP dispatch](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/base/decorator.py)调用 DataProto.chunk 等行切分。默认普通长度均衡分支则直接对样本分区，不能把组完整当作所有 GRPO 运行的保证。

同一固定版本的正常训练入口先为每个 prompt 生成唯一 UUID，然后两次用 `repeat(..., interleave=True)` 对齐 prompt 与 response。[DataProto.repeat](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/protocol.py)对非 tensor UID 使用 `np.repeat`，构造每个 UID 连续且固定 `rollout.n` 行的布局；在此布局保持不变时，等段数也就是等行数。本轮没有发现该标准入口主动引入上述非法布局；自定义重排、过滤或可变组大小接入时，须重新检查这些前提，不能据边界输入直接宣布 trainer 有 bug。

2026-10-22 本机复查：[脚本](../../experiments/scripts/verify_verl_uid_balancing.py)从固定 Git blob AST 选择四个完整原函数，保留函数体/注解/装饰器，使用真实 CPU torch/heapq，包壳仅绕过 verl 根初始化。直接调用[原测试文件](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/utils/test_seqlen_balancing.py)的3个 group 测试函数体，全部通过；另18组连续等大小控制覆盖组数4/8、每组1/2/3、分区数1/2/4，并检查索引完整、UID不跨区、行数相等。上述两个反例及段数不可整除/不足的两个断言共4组边界观察符合预期。

```bash
python experiments/scripts/verify_verl_uid_balancing.py /path/to/verl
```

Python 3.12.13、torch 2.9.1；不是原文件的完整 pytest、DataProto/rollout/trainer/dispatch、DP通信或性能测试。官方API与本地源码 Git blob 均为 `3ffb4f28c9bc251ad2d8b7f670e4917b0004831d`；实现 SHA256 `f0536d757a8c8eb1a15d63e7174a8995ebbf180481845576f1817ee6020b4902`，原测试 `658fb094f2edb2fe6914ded167e4b18aba3543d5b5410a93a2816622a7b9c48d`。本项只澄清输入与切分契约，不改源码、不新开贡献候选或 M5 硬件任务。

### Python 分区的标量类型与性能范围

批均衡的分区器是 Python 堆与排序算法；把 Tensor 中的零维标量一直带入比较与累加，会反复经过 Torch 分派。[问题 #7279](https://github.com/verl-project/verl/issues/7279)已有[候选 #7281](https://github.com/verl-project/verl/pull/7281)，不需要另一份转换补丁。2026-10-25 核验候选 OPEN、非 draft、无普通评论／行内评论／review；固定 head `05ef5dcc8d36f3e7d5c652b2d9d14b11ffc5a487`，父提交／与当前 main 的共同祖先 `87c68d2124ead7e1e3611e6ad0a8d0393dfd5274`，相对 main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` ahead 1／behind 297。关键词检索另有缓存 spread 的 [#8137](https://github.com/verl-project/verl/pull/8137) 和动态批复用的 [#7097](https://github.com/verl-project/verl/pull/7097)，不据此认定均衡领域无人负责。

候选保留原 Tensor-native `calculate_workload`，在经典 PPO、V1 PPO 与 SFT 的 Python 分区入口增加[主机标量 helper](https://github.com/verl-project/verl/blob/05ef5dcc8d36f3e7d5c652b2d9d14b11ffc5a487/verl/utils/seqlen_balancing.py)。整数走 `tolist()`，浮点走 `list(workloads.detach().cpu().numpy())`，因此 FP32 元素仍是 NumPy float32，而不是 Python float。SFT 用默认浮点 Tensor 构造长度；这条路径不能只凭 PPO 的整数微基准判断兼容性。经典 PPO／SFT 还按 workload 与索引排序、将小样本放在两端；候选 V1 仅分区后拼平，没有相同的区内排序步骤。PrefixGrouper 的分支跳过外层未使用的 workload 计算，组内原有工作量计算仍存在，也不改变上节的 UID 前提。

**数值相等的单项转换，不保证累加与分区相同。** Python float 会让原 FP32 workload 的后续累加改用双精度，近似平衡时的差值比较和 tie-breaking 可以变化。候选在分区集合首次加入 NumPy 浮点数时直接用该标量初始化 sum；否则 NumPy 1.x 的 `0 + np.float32` 可先提升至 float64，而 NumPy 2.x 保持 float32。这个首项规则同时用于 add 与 merge，不能将补丁缩成一行裸 `tolist()`。

[本机脚本](../../experiments/verl-balance-scalars/verify.py)执行候选[两项原测试函数体](https://github.com/verl-project/verl/blob/05ef5dcc8d36f3e7d5c652b2d9d14b11ffc5a487/tests/utils/test_seqlen_balancing.py)，不改函数体、参数或断言；原 helper 与分区函数也按 AST 完整提取，真实 CPU torch／NumPy／heapq 执行，绕过 verl 包初始化。两项通过，涵盖 int64／FP32 标量类型、分区一致与输入不变。再复用原16长度 FP32 反例，在候选／父版／main 的 Tensor 路径上，等大小及不等大小分区均一致；候选 NumPy float32 列表保持原等大小分区，而裸 `tolist()` 得到不同索引集合。[结果](../../experiments/verl-balance-scalars/result.json)保存具体输入、分区、版本与原文件 SHA256。两次执行结果一致；Python 3.12.13、torch 2.14.1、NumPy 2.5.4、macOS arm64。五份候选完整文件另做 AST／内存 compile；没有运行原完整8／9项、trainer、DP 分发或 NumPy 1.x，后者仅为源码条件与作者报告。

```bash
/path/to/python experiments/verl-balance-scalars/verify.py /path/to/verl
```

[作者报告](https://github.com/verl-project/verl/pull/7281)的单核 Xeon 整数微基准为22–27倍局部提速；四张 RTX 3090 的 Qwen3.5-0.8B SFT、ABBA四次短跑中，balance 阶段均值89.568→28.557ms，但训练步7492.403→7530.990ms，吞吐2591.3→2576.0 token/s。局部节约约61ms并未转成该测量中的整步提速，不能用微基准倍率描述训练收益。这些训练数值是他方报告，不是 M1 重测；本轮不做性能基准、模型下载或 GPU 访问，不新增 M5 队列、不应用候选或写上游。

## 6. GRPO vs PPO 在 verl 中的实际差异

| 方面 | PPO (adv_estimator=gae) | GRPO (adv_estimator=grpo) |
|------|------------------------|--------------------------|
| Value function | 需要 (critic 网络) | 不需要 |
| 优势估计 | GAE (λ-回报) | 组统计 (mean/std) |
| 模型数量 | 4 (π, π_ref, V, RM) | 2 (π, π_ref) + RM |
| 内存 | ~2x (critic + optimizer) | ~1x |
| KL 惩罚 | 通常 in-reward | 通常 in-loss |
| 采样数 | n=1 | n=4-64 |

## 7. 关键配置参数

```yaml
algorithm:
  adv_estimator: grpo  # 切换为 GRPO
  use_kl_in_reward: false  # GRPO 推荐 in-loss KL

actor_rollout_ref:
  rollout:
    n: 8  # 每个 prompt 采样数
  actor:
    use_kl_loss: true  # GRPO 推荐
    kl_loss_coef: 0.05  # KL 惩罚系数
    ppo_epochs: 1  # GRPO 通常 1 epoch
    clip_range: 0.2

  model:
    use_prefix_grouper: true  # 启用前缀共享
```

## 8. 核心洞察

1. **统一框架**: verl 用一个 trainer 支持所有 RL 算法, 通过 config 切换 → 优雅设计
2. **GRPO 本质**: 用组统计替代 learned baseline → 简单但有效
3. **Prefix Sharing**: GRPO n=8 天然适合 prefix caching (相同 prompt × 8 response)
4. **交错排列**: `repeat(interleave=True)` 使同一 prompt 的 response 相邻 → 适合 PrefixGrouper
5. **异步 Rollout**: rollout 和 actor 训练可以异步 (vLLM/SGLang 引擎)
6. **LoRA 优化**: ref_in_actor=True 避免加载第二个模型 → 50% 内存节省

## 9. 所有可用优势估计器 (core_algos.py)

| 估计器 | 全称 | 需要Critic | 说明 |
|--------|------|-----------|------|
| GAE | Generalized Advantage Estimation | ✓ | 标准 PPO |
| GRPO | Group Relative Policy Optimization | ✗ | 组内归一化 |
| GRPO_VECTORIZED | GRPO (向量化) | ✗ | 更高效的组操作 |
| GRPO_PASSK | Pass@k GRPO | ✗ | 基于 top-k |
| GDPO | 解耦归一化 | ✗ | 每维度独立归一化 |
| REINFORCE_PLUS_PLUS | 增强版 REINFORCE | ✗ | 折扣回报 |
| RLOO | REINFORCE Leave-One-Out | ✗ | 从其他样本剔除基线 |
| REMAX | ReMax | ✗ | 贪婪基线 |
| GPG | 广义策略梯度 | ✗ | 无裁剪 |

## 10. 所有策略损失函数 (core_algos.py:50)

| 损失 | 说明 |
|------|------|
| vanilla | 标准 PPO 裁剪损失 |
| gpg | 无裁剪 REINFORCE |
| bypass_mode | 2策略模式 (rollout=old policy) |
| dppo_tv / dppo_kl | 总变差 / KL 约束 |
| gspo | 几何均值序列级比率 |
| sapo | 平滑策略优化 |
| cispo | 裁剪重要性采样 |
| clip_cov / kl_cov | 协方差裁剪损失 |
| geo_mean | GMPO 几何均值策略 |

## 11. 权重同步机制

训练后 actor 权重需同步到 rollout 引擎:
```python
self.checkpoint_manager.update_weights(self.global_steps)
```
支持三种后端: naive (保存+加载) / NCCL (直接传输) / NIXL (RDMA)

---

## 12. GRPO Advantage 源码级深度 (core_algos.py line 267-358)

### Outcome vs Process Reward

verl GRPO 主要用 **outcome reward**:
```python
scores = token_level_rewards.sum(dim=-1)  # 每序列: sum所有token reward → 1 scalar
# outcome: 只EOS位置非零 → sum = 该序列的最终reward
```

### `compute_grpo_outcome_advantage` (line 267-331) — loop版本

```python
for each group g (identified by uid):
    μ_g = mean(scores in group g)
    σ_g = std(scores in group g)  # or 1.0 for singleton group
    for each sample i in group g:
        if norm_adv_by_std_in_grpo:    # default=True → 原始GRPO公式
            scores[i] = (scores[i] - μ_g) / (σ_g + ε)
        else:                           # Dr.GRPO variant
            scores[i] = scores[i] - μ_g  # 只减均值, 不除std

advantages = scores.unsqueeze(-1) * response_mask  # scalar broadcast到所有tokens
```

关键:
- Singleton (n=1): μ=0, σ=1 → advantage=raw reward → 无normalization
- advantage是scalar per sequence → broadcast到所有response tokens
- `norm_adv_by_std_in_grpo=True` → 标准GRPO; `False` → Dr.GRPO

### `compute_grpo_vectorized_outcome_advantage` (line 334-358) — 纯PyTorch版

```python
g = as_torch_index(uid)          # UUIDs → contiguous 0..G-1
mean_g, std_g, _ = group_mean_std(scores, g)  # scatter-add, 无Python loop!

if norm_adv_by_std_in_grpo:
    scalars = (scores - mean_g[g]) / (std_g[g] + ε)
else:
    scalars = scores - mean_g[g]
advantages = scalars.unsqueeze(-1) * response_mask
```

**`group_mean_std` (groupwise.py line 164)**: 纯PyTorch scatter
- `torch.zeros(G).index_add_(0, gidx, values)` → sum + sum_sq
- `count.clamp_min(1)` → mean; `(count-1).clamp_min(1)` → Bessel-corrected variance
- Singleton fallback: mean=0, std=1

---

## 13. KL Penalty 详细类型 (core_algos.py line 2126)

| Type | 公式 | 特性 |
|------|------|------|
| `kl` / `k1` | `logprob - ref_logprob` | 简单log-ratio |
| `abs` | `|logprob - ref_logprob|` | 绝对差 |
| `mse` / `k2` | `0.5*(logprob-ref)^2` | **无偏梯度** |
| `low_var_kl` / `k3` | `exp(ref-log) - (ref-log) - 1` | **低方差估计**(clipped) |
| `k3+` / `k1+` | fwd用k1/k3值+bwd用k2梯度 | Straight-through trick |

默认: `use_kl_in_reward=False`, `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`

---

## 14. PPO-Clip + Dual-Clip Loss (core_algos.py line 1279)

```python
ratio = exp(log_prob - old_log_prob)  # importance ratio

pg_losses1 = -advantages * ratio              # unclipped
pg_losses2 = -advantages * clip(ratio, 1-ε, 1+ε)  # clipped
clip_pg_losses1 = max(pg_losses1, pg_losses2)      # PPO-clip

# Dual-clip (负advantage时防止ratio过小):
pg_losses3 = -advantages * clip_ratio_c     # c > 1.0 (默认3.0)
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)

pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

Dual-clip来源: https://arxiv.org/pdf/1912.09729 → 防止负advantage时ratio过小

IS weights可选: `pg_losses * rollout_is_weights`

---

## 15. Loss Aggregation 四模式 (core_algos.py line 1138)

| Mode | 计算 | 使用场景 |
|------|------|----------|
| `token-mean` | Sum所有masked / batch_num_tokens | **verl GRPO默认** |
| `seq-mean-token-sum` | Sum per seq → mean across seqs | - |
| `seq-mean-token-sum-norm` | 同上 / loss_scale_factor | **Dr.GRPO** (防length bias) |
| `seq-mean-token-mean` | Mean per seq → mean across seqs | 原始论文, verl警告不稳定 |

---

## 16. GRPO 变体详解

### Dr.GRPO (`norm_adv_by_std_in_grpo=False`)
- 只mean-centering → 不除std → 消除length bias
- 配合 `loss_agg_mode=seq-mean-token-sum-norm` + `use_kl_loss=False`

### GRPO-passk (`adv_estimator=grpo_passk`)
- 只有best response per group有非零advantage: `r_max - r_second_max`
- 其他completions advantage=0

### GDPO (Group reward-Decoupled Normalization)
- 每reward维度独立normalization → weighted sum
- 需要 `gdpo_reward_keys` (如 ["format_reward", "accuracy_reward"])
- reward function返回per-dimension scores

### GSPO loss
- Geometric-mean sequence-level importance ratio
- `s_i = (π_θ/π_old)^{1/|y_i|}` (length-normalized)
- combined ratio = `sg[s_i] * ratio_token / sg[ratio_token]`
- 推荐 `loss_agg_mode=seq-mean-token-mean`

### Vectorized GRPO (`grpo_vectorized`)
- Python loop → pure PyTorch scatter (`index_add_`) → 大batch更快

### Filter Groups (DAPO)
- `FilterGroupsConfig`: 过滤group by metric (acc/score/seq_reward)
- `max_num_gen_batches` 控制上限
- DAPO reward manager: overlong response penalties

---

## 17. Rollout Correction System

- **2-policy bypass**: `π_rollout = π_old` (同一policy生成+训练)
- **3-policy decoupled**: `π_rollout` (生成), `π_old` (PPO参考), `π_θ` (当前训练)
- IS weights: token-level or sequence-level importance sampling
- Rejection sampling: k1/k2/k3/geometric 配置

---

## 18. Balance Batch & Dynamic BSZ

```python
# trainer.balance_batch=True → 按估算workload重排跨DP数据，不保证valid token数完全相等
# → 改变data order但uid-based advantage不受影响

# use_dynamic_bsz=True → 按预算调整micro-batch；不是任意长度/状态下不OOM的保证
```

### V1 源 prompt 单位与可变轨迹行

**同一UID属于同一个全局优化mini-batch，不等于该UID的轨迹全在同一DP rank。** 2026-10-22 只读已有 [#8201](https://github.com/verl-project/verl/pull/8201)，读取时open，固定head `e891fd814e24c0b2d0c70a47e234f75a63d10ad6`、目标base `18cfb74852a053a27dcf301d3318a093be81b628`、merge-base `2035f58dcb06d8cad4caee107b30f4045d724367`。14文件830增113删；本节限定新版V1布局与消费链，不复核全部算法/损失归约或性能。旧group helper在该base/head的AST相同，候选不是自动修复此前PrefixGrouper同卡共享的所有前提。

#### 从源 prompt 到实际行数

新版[配置文档](https://github.com/verl-project/verl/blob/e891fd814e24c0b2d0c70a47e234f75a63d10ad6/docs/examples/config.rst)将V1的ppo_mini_batch_size解释为源prompt/UID数量，不再用固定rollout.n推导实际行数。多输出session或保留部分成功轨迹，都可能令每UID贡献不同数量的训练行；legacy trainer仍维持原有行数契约，不能把V1语义直接套到旧路径。

[`PPOTrainer._balance_batch`](https://github.com/verl-project/verl/blob/e891fd814e24c0b2d0c70a47e234f75a63d10ad6/verl/trainer/ppo/v1/trainer_base.py#L1649)用key.rsplit("_",2)去掉session/output尾部后按UID字典全局聚合，保留首次出现顺序，必要时在控制端shuffle整组。每个active模型要求现存UID数能整除其ppo_mini_batch_size；公共单元的prompt数为actor/critic这些配置值的GCD，**不乘rollout.n**。它只从已有key建组，不负责补回完全没有轨迹行的prompt；失败保留与replay取样仍是前置契约。

每单元单独补行：动态batching只需DP整除；静态路径取各active阶段所需global micro-batch行倍数的LCM，包括actor/ref logprob、critic forward/train及actor train。不能先把所有轨迹合起来补齐，再任意切回prompt mini-batch。源码padding模板是128-token合成行、response/loss mask为0并标is_padding，不是额外的真实轨迹；本节不以此认证全栈归约不受padding影响。

[原V1测试](https://github.com/verl-project/verl/blob/e891fd814e24c0b2d0c70a47e234f75a63d10ad6/tests/trainer/ppo/v1/test_minibatching_on_cpu.py)给8个UID的输出数 `[2,1,3,2,1,4,2,3]`，actor每批4个prompt、critic每批2个、DP=2、动态batching。其布局期望是：

| 单元UID | 真实行数 | 单元补行 | 单元global行数 |
|---|---|---|---|
| p0 p1 | 3 | 1 | 4 |
| p2 p3 | 5 | 1 | 6 |
| p4 p5 | 5 | 1 | 6 |
| p6 p7 | 5 | 1 | 6 |

18条真实行变成22条布局行。`_get_global_mini_batch_sizes`将相邻单元合并，actor得到 `[10,12]`、critic得到 `[4,6,6,6]`；每epoch分别计划2/4次mini-batch调用，不因原本rollout.n=2却实际输出18行而改成按固定行数切分。M1仅以该layout为输入执行原coalescer核对这两个列表，未执行测试中的StubTrainer、KVBatchMeta或替身padding。

#### 排列和 worker 必须共享边界

每单元内部按单行workload分区，同一UID可分到不同rank；结果按rank拼接，但每rank内部按单元顺序累积share。每单元行数已经DP整除，worker就能从各rank对应位置取回同一个global mini-batch。轻重交错排序也只在当前单元当前rank的share内，不能跨边界重排；这不是PrefixGrouper的同UID同rank算法，也不保证每个micro-batch计算量相同。

控制端将可变行边界保存为 `global_mini_batch_sizes`。[TrainingWorker](https://github.com/verl-project/verl/blob/e891fd814e24c0b2d0c70a47e234f75a63d10ad6/verl/workers/engine_workers.py#L244)拒绝同时给固定mini_batch_size/num_mini_batch，并要求每个global大小可被DP整除；除DP后传给[可变迭代器](https://github.com/verl-project/verl/blob/e891fd814e24c0b2d0c70a47e234f75a63d10ad6/verl/utils/tensordict_utils.py#L615)，后者验证正值与总行数，每个epoch重用相同边界。每次train_batch携带实际含padding的global_batch_size；update_lr_scheduler只在整个调用的最后一次迭代置True，不能直接用mini-batch个数当LR scheduler步数。这里只核接线，未跑optimizer/调度器。

V1在shuffle=true且ppo_epochs>1时直接拒绝，而不是静默退化；shuffle=true/epoch=1在控制端打散UID后，worker不再shuffle行。多epoch且shuffle=false仍允许。legacy路径对多epoch worker shuffle回退到whole-batch均衡，行为不同，不能混用一条规则解释两条trainer。

#### 本机检查的范围

Python3.12.13仅标准库执行原完整 `get_uid_balance_unit_size`、`validate_minibatching_config` 及原staticmethod `_get_global_mini_batch_sizes` 的AST：4种active模型组合GCD、2种layout合并/2种拒绝、2种合法/2种非法shuffle配置，共12项通过。SimpleNamespace只承载输入字段，不实现OmegaConf/TransferQueue/TensorDict；没有调用controller、worker或旧分区器。另8个完整模块内存compile通过；base/head旧group helper AST相同。缺omegaconf、tensordict、transfer_queue，未补装、扩写替身或运行原测试矩阵。

最小原coalescer复查（本地须已有固定Git对象）：

```bash
python3 -B - /path/to/verl-git <<'PY'
import ast, subprocess, sys
from types import SimpleNamespace
src = subprocess.check_output(['git', '-C', sys.argv[1], 'show',
    'e891fd814e24c0b2d0c70a47e234f75a63d10ad6:verl/trainer/ppo/v1/trainer_base.py'])
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'PPOTrainer')
f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_global_mini_batch_sizes')
future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
shell = ast.ClassDef(name='LayoutOnly', bases=[], keywords=[], body=[f], decorator_list=[])
tree = ast.fix_missing_locations(ast.Module(body=[future, shell], type_ignores=[]))
ns = {'_MINIBATCH_LAYOUT_KEY': '_minibatch_layout'}
exec(compile(tree, 'original-layout-helper', 'exec'), ns)
batch = SimpleNamespace(extra_info={'_minibatch_layout':
    {'unit_prompt_count': 2, 'unit_global_sizes': [4, 6, 6, 6]}})
assert ns['LayoutOnly']._get_global_mini_batch_sizes(batch, 4) == [10, 12]
assert ns['LayoutOnly']._get_global_mini_batch_sizes(batch, 2) == [4, 6, 6, 6]
PY
```

SHA256：utils.py `122d3f7af27a24bd63d5f1afeb849be7687abb9497422cfcb560926c3e72db0b`；trainer_base.py `22cd73dc3c86586b9defa40c37130ccda95ff3e8db9101d0b74f682c35c6ea04`；原V1测试 `94e1ffcadd1ed5c494718fa18f6ed3a209d42726912379b0026051e16b6cb6d9`；worker测试 `0e965c532331a3536fcc2d9c92af9926c97d9272898a850b2bf7623e995f0eaf`。作者正文报告组合测试32过/10跳过，以及既有32卡Qwen3-30B-A3B均衡benchmark；后者不专门验证新版可变输出调度，本机均未复跑。没有TQ服务、真实数据重排/DP dispatch、padding入库、梯度归约或GPU/NPU/性能验证。此题收束为新单位与边界传递的源码证据；[M5待条件参考](../../experiments/handoff-m5.md#verl-v1-可变行布局参考)不代表采用候选或接单，不修改上游/M5专项。
