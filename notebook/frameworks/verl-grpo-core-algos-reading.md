# verl GRPO Training Core Algorithms — core_algos.py 源码深度阅读

> 2026-06-07 | 2487行源码分析: 14种Advantage Estimator + 10种Policy Loss + 注册表架构

## 核心架构: 注册表式扩展

### 双注册表系统

```python
# Advantage Estimator 注册表
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}
@register_adv_est(AdvantageEstimator.GRPO)  # 装饰器注册

# Policy Loss 注册表
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
@register_policy_loss("vanilla")  # 装饰器注册
```

**设计优势**: 新算法只需写函数+装饰器 → 零修改框架代码 → 可插拔扩展

### 调用路径

```
RayPPOTrainer.fit()
  → compute_advantage()  # 根据 adv_estimator 名从注册表查找
  → compute_policy_loss()  # 根据 loss_name 名从注册表查找
```

## 一、14种 Advantage Estimator

### 1. GAE (PPO标准, 行215)

```python
A_t^{GAE} = Σ(γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

- 需要 critic V(s) → 4模型
- γ=discount, λ=GAE parameter
- 最灵活但最昂贵

### 2. GRPO (组相对, 行267) — **核心!**

```python
scores = token_level_rewards.sum(dim=-1)  # outcome-only → 整个response的总reward
# 组内统计
id2mean[idx] = mean(scores_for_same_prompt)
id2std[idx] = std(scores_for_same_prompt)
# 组归一化
scores[i] = (scores[i] - id2mean) / (id2std + epsilon)
# broadcast到token level
advantages = scores.unsqueeze(-1) * response_mask
```

**关键设计决策**:
1. **Outcome-only**: `scores = rewards.sum()` → 只看最终reward, 不区分哪个token贡献
2. **组归一化**: `(r_i - μ_group) / σ_group` → 同prompt的n个response互为baseline
3. **单样本组特殊处理**: `len(id2score[idx])==1 → mean=0, std=1`；默认优势为score/(1+epsilon)，关闭std归一化为score，不是自动归零，见下节
4. **Dr.GRPO选项**: `norm_adv_by_std_in_grpo=False` → 不除std → 更保守(防止过度归一化)

### 3. GRPO_VECTORIZED (行334)

```python
# 用torch-level group操作替代Python for循环
g = as_torch_index(index, device=scores.device)
mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0)
scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
```

**实现变化**：Python循环改为torch向量操作；数值等价与实际吞吐需分开验证，本轮只核数值。

### 单样本与同分组的不同回退

**singleton不是多样本同分组。** 2026-10-22 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，[loop GRPO与vectorized GRPO](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py)都支持同一UID只有一条样本时的mean=0、std=1回退。vectorized分支由[原group_mean_std](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/groupwise.py)实现同样回退；多样本组则按无偏样本标准差归一化。

本机对照显式epsilon=1e-6；下表是每行的有效token优势，实际输出还乘response_mask：

| 同一UID的score | 默认标准化 | 关闭std归一化 |
|---|---|---|
| 单样本2 | 约1.999998 | 2 |
| 单样本−3 | 约−2.999997 | −3 |
| 单样本0 | 0 | 0 |
| 两样本[4,4] | [0,0] | [0,0] |
| 两样本[1,3] | 约[−0.707106,0.707106] | [−1,1] |

单样本没有组内互为baseline的比较，代码采用近似原奖励的回退；它不是标准中心化公式在n=1上的直接计算结果。多样本同分时分子为零，默认epsilon使结果为零，而不是“0/0必得NaN”。也不能从 `advantages.std()==0` 推出所有优势为零：常数非零向量同样标准差为零。诊断应分开检查组内样本数、reward差异、有效mask、优势绝对值及真实梯度/参数变化。

这纠正了[排错指南](../grpo-training-debugging-guide.md#2-loss-values-and-zero-advantages)原来的“n=1必退化、禁止n=1、必须n≥4或MoE≥8”表述。实现没有给出这些统一硬门槛；本轮也没有比较采样成本、训练收敛或模型质量，不能把保留非零优势解释为n=1必然有效学习。关闭std归一化仍保留多样本组的均值中心化，不是统一把所有组变成原奖励。

[复查脚本](../../experiments/scripts/verify_verl_grpo_groups.py)AST选择完整原enum/registry/注册装饰器、两个estimator及四个groupwise helper，保留原函数体/装饰器，延后注解求值以省略无关配置依赖。原helper始终收到显式CPU device，未导入设备发现模块或调用get_device_name。原测试的低方差1例和随机等价性3参数例直接调用全过；随机helper明确保证每组至少两样本，因此还需额外singleton对照，不能把原随机覆盖当作已测n=1。

额外6种布局（正/负/零singleton、同分pair、有差异pair、三类组混合）×标准化开/关，共12组双实现CPU检查通过，同时核输出/returns、有效mask及no_grad结果；没有模型或训练替身。三个原随机例最大绝对差约2.38e−7、2.38e−7、4.77e−7，是给定容差内等价，不是逐位一致或性能证明。

```bash
python experiments/scripts/verify_verl_grpo_groups.py /path/to/verl
```

Python3.12.13、torch2.9.1、NumPy2.0.2、pytest8.4.2；非完整pytest、训练、loss/optimizer或DP/GPU验收。官方API/本地Git blob：core_algos `9d08b8461a0f95d25f348c566358360bbfb72ece`、groupwise `84e79ac903082c913f479f3cea6eed54941e9c16`。SHA256依次 `a54b6d812db4441e8b0989ae69d2c46b0a0a7b584296f73d8316f264462c37cb`、`409403d0002ff0d1fd1c41812163b1cfa52a52248677aab00e6ffcf3b937b442`；原测试模块 `0a60e917589ab91646989f0a8616f810fea0dc70047d2b8c13914f4977f5569c`。只纠正知识与诊断规则，不改上游算法或新增M5任务。

### 4. GDPO (组奖励解耦归一化, 行361)

```python
# 不是先sum再归一化(GRPO), 而是每维度独立归一化再sum!
A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)  # 每个reward维度独立
A = Σ_k A_k  # 维度间解耦
```

**优势**: 防止主导reward信号淹没弱信号 → 多reward维度场景

### 5. GRPO_PASSK (行471)

```python
# top-k samples only → pass@k 评估导向
# 只保留每组最好的k个response → 更高信号质量
```

### 6. REMAX (行732)

```python
# 每个prompt额外贪心生成的响应奖励作为外部baseline；不是sampled最大值
# advantage = outcome reward-to-go - greedy_reward * response_mask
```

2026-10-22 已核对 #8310：核心公式未改，新增 V1 贪心会话与基线回填；[来源、CPU反例与验证边界](verl-remax-icepop-source-reading.md#v1-贪心会话与-sampled-最大值)。逐 token 贪心不保证序列概率或奖励全局最优。

### 7. GPG (直接策略梯度, 行768)

```python
# 无clipping, 无ratio → 直接∇logπ · A
# 最简单的REINFORCE variant
```

### 8-14. 其他

- **REINFORCE_PLUS_PLUS**: reward baseline归一化
- **REINFORCE_PLUS_PLUS_BASELINE**: reward + learned baseline
- **RLOO**: Leave-One-Out baseline (组内其他样本均值)
- **OPO**: Online Policy Optimization
- **RLOO_VECTORIZED**: 向量化RLOO
- **OPTIMAL_TOKEN_BASELINE**: token-level最优baseline(理论推导)
- **TIR_OPTIMAL_TOKEN_BASELINE**: TIR(Trust-region)优化token baseline

## 二、10种 Policy Loss

### 1. Vanilla (PPO-clip, 行1278) — **标准!**

```python
ratio = exp(log_prob - old_log_prob)  # π_new / π_old
pg_losses1 = -advantages * ratio  # 无clip
pg_losses2 = -advantages * clamp(ratio, 1-ε, 1+ε)  # clip
clip_pg_losses1 = max(pg_losses1, pg_losses2)

# 双clip (Cheng et al., 1912.09729)
pg_losses3 = -advantages * clip_ratio_c  # 下界clip (A<0时)
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)

pg_losses = where(A < 0, clip_pg_losses2, clip_pg_losses1)
```

**特性**: 标准PPO clip + 可选双clip(防止策略在A<0时过度偏离)

### 2. DPPO-TV (行1372)

```python
# Total Variation distance constraint
# TV(π_new, π_old) ≤ ε → 替代PPO的ratio clip
# 更理论化的约束
```

### 3. DPPO-KL (行1453)

```python
# KL divergence constraint
# KL(π_new || π_old) ≤ δ → 替代PPO的ratio clip
# 更严格的策略变化限制
```

### 4. GSPO (行1538) — 序列级clip

```python
# 序列级ratio → 不按token clip, 按整个序列clip
# 对长序列更稳定(避免逐token clip过度限制)
```

### 5. SAPO (行1614) — 平滑advantage

```python
# 平滑advantage → 避免极端advantage值
# 用soft clipping或log变换
```

### 6. GPG (行1699) — 直接梯度

```python
# 无ratio, 无clip → 直接 A · ∇logπ
# 最简单的REINFORCE → 用于GPG advantage estimator
```

### 7. Clip-Cov (行1735) — 带协方差的clip

```python
# 在clip之上加协方差约束
# 防止高variance的ratio导致不稳定
```

### 8. KL-Cov (行1840) — KL+协方差双约束

### 9. Geo-Mean (行1920) — 几何均值ratio

```python
# ratio = (π_new/π_old)^(1/N) → 几何均值而非逐token
# 序列级ratio → 更稳定
```

### 10. CISPO (行2006) — Clipped IS Policy Optimization

```python
# Clipped Importance Sampling → 在IS权重上clip而非ratio上
# 用于bypass mode (off-policy场景)
```

### 11. Bypass Mode (行2351) — **Off-policy关键!**

```python
# In bypass mode: old_log_prob = rollout_log_prob (不是actor_log_prob)
# 两种子模式:
#   "ppo_clip": PPO clip with IS ratio π_current/π_rollout (no additional IS weights)
#   "reinforce": REINFORCE with IS weights w = π_current/π_rollout
# IS aggregation: token-level, sequence-level, or None
# IS threshold: truncation at rollout_is_threshold (default 2.0)
```

## 三、Loss Aggregation (agg_loss, 行1138)

| Mode | 计算 | 用途 |
|------|------|------|
| token-mean | Σ(loss·mask)/Σ(mask_tokens) | 标准PPO |
| seq-mean-token-sum | Σ(Σ(loss·mask))/batch_size | 序列级 |
| seq-mean-token-sum-norm | seq-mean-token-sum/horizon | 归一化 |
| seq-mean-token-mean | Σ(Σ(loss·mask)/tokens)/batch_size | per-seq mean |

**关键**: `dp_size`参数 → 确保FSDP/Megatron DP下loss等价 → 分布式一致性

### KL 前向值与反向规则

**`k3+` 的前向截断不是梯度截断。** 2026-10-22 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，[kl_penalty](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py)先剥掉模式名末尾的 `+`，得到原模式前向值F；然后用下面的straight-through组合返回结果。令d=logprob−ref_logprob：

```python
B = 0.5 * d.square()
result = B - B.detach() + F.detach()
```

返回数值仍为F，反向却只经过B；对可微logprob的局部导数是d，而非F的数学导数。若ref_logprob也设为可微，其局部导数为−d；实际ref是否固定须按调用处确认，不能用这个控制例推断模型接线。

k3在未触发保护时的前向为 `exp(-d)+d-1`；源码先把−d截到[−20,20]，再把结果截到[−10,10]。原k3通过这些clamp求导，k3+则将整个前向分支detach。真实FP64标量对照如下：

| d | k3与k3+前向值 | k3对logprob的梯度 | k3+对logprob的梯度 |
|---|---|---|---|
| 0.5 | 约0.106531 | 约0.393469 | 0.5 |
| 30 | 10 | 0 | 30 |
| −30 | 10 | 0 | −30 |

因此日志KL看起来被限制在10，不能证明实际正则梯度同样被限制。对k3+的显示值做中心有限差分，d=0.5时约0.393469、d=30时为0，与autograd的0.5/30不同；这是显式替换反向的设计结果，不是本轮发现了autograd数值错误。原测试也覆盖k1+/kl+/abs+/low_var_kl+的前向与各自base一致，不能把所有模式的 `+` 理解成统一前向k2。

实际[PPO loss接线](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/utils/losses.py)从config.kl_loss_type选helper，读当前log_prob和data中的ref_log_prob，归约后乘kl_loss_coef加入policy_loss；记录的kl_loss仍是前向数值。完整模型梯度还受coef、mask/归约和logprob到参数的链式导数影响，本轮没有运行这些集成或判断训练稳定性。reward端则用old_log_probs/ref_log_prob的helper数值改写奖励；若不沿该helper反向，名字带 `+` 不会凭空改变同输入的前向奖励。

[复查脚本](../../experiments/scripts/verify_verl_kl_plus.py)AST选择完整原两个KL函数和原测试，保留函数体/装饰器/断言，直接调用原5个前向参数例与1个梯度例，全过。另d为±0.5/±30、模式k1+/k3+/abs+共12组前向/双输入局部梯度检查，以及2组k3+有限差分对照通过。没有用手写helper替代原实现，有限差分也没有被当作straight-through反向的正确性标准。

```bash
python experiments/scripts/verify_verl_kl_plus.py /path/to/verl
```

Python3.12.13、torch2.9.1、pytest8.4.2；非完整pytest、PPO loss/optimizer、模型/DP或设备验证，也未检验KL估计器的统计无偏性、配置优劣或收敛效果。官方API/本地core_algos Git blob为 `9d08b8461a0f95d25f348c566358360bbfb72ece`，losses.py为 `ebaa4e4d8b759db980f62bf96f83be305f54847e`。core模块SHA256 `a54b6d812db4441e8b0989ae69d2c46b0a0a7b584296f73d8316f264462c37cb`，原测试模块 `0a60e917589ab91646989f0a8616f810fea0dc70047d2b8c13914f4977f5569c`。本题只澄清数值/梯度与接线，不改算法或新增M5任务。

### 掩码统计与梯度的数值边界

**统计量有限、所有输出槽有限、梯度有限是三个不同条件。** 2026-10-22 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，[`masked_sum/mean`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/torch_functional.py)已使用 `torch.where(mask.bool(), values, 0)`，再乘mask求和。因此不能笼统说“masked mean用零乘NaN，必然污染前向”：该版本会先移除mask外的NaN/Inf。mask内的非有限值仍会传播，不应被无条件隐藏。

局部CPU反向对照取 `x=[1,3,NaN]` 或 `[1,3,Inf]`，mask为 `[1,1,0]`，FP32/FP64结果一致，表中数值按数值容差比较：

| 原函数或输入图 | 前向 | x的梯度 |
|---|---|---|
| masked_mean(x) | 2 | [0.5,0.5,0] |
| masked_mean(x.square()) | 5 | [1,3,NaN] |
| masked_var(x)，默认无偏 | 2 | [NaN,NaN,NaN] |
| 先where清理无效x，再送入同一原masked_var | 2 | [-2,2,0] |

mean的where只控制传入值的前向选择，不能删除更早的square反向。var先计算 `centered_values = x-mean` 及平方，再交给masked_mean；无效位置的反向非有限项可以经共享mean污染有效位置。这与[PyTorch 2.9 autograd说明](https://docs.pytorch.org/docs/2.9/notes/autograd.html#division-by-zero-in-autograd)中的原则一致：非法运算已经记录进图后，遮掉最终结果不保证反向安全。此处实际运行的是原方差/平方图，不是把官方除零示例当作verl实测。

原 `masked_whiten` 也只用mask计算统计量，返回时仍对所有位置执行 `(x-mean)*rsqrt(var+eps)`。本机no_grad对照中有效两项有限，mask外槽仍NaN/Inf；最后再乘零仍产生NaN，但再送入原masked_mean可得到有限标量。不能由这个标量反推整个advantage张量已清理，也不能将输出乘mask当作通用NaN修复。

上表预清理只是一项诊断控制，没有改原实现。它说明在这个局部图中，必须在square/centering等运算前处理明确无效的输入；如果x本身来自更早的非法运算，仍须检查更早的计算图，不能仅在此处补where或对全张量nan_to_num就称训练修复。

调用范围同样重要：[所读core_algos](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py)中的GAE、RF++及RF++baseline将相应白化放在torch.no_grad块内。本轮显式requires_grad=True的局部反例，不能直接外推为这些advantage路径正在反传NaN，或某次线上loss异常已找到根因。GDPO等其他入口以及非有限输入如何进入调用链，需要各自核对；本轮没有运行estimator、策略损失、模型或optimizer，也不定性新缺陷。

[复查脚本](../../experiments/scripts/verify_verl_masked_stats.py)AST选择四个完整原函数和原test_masked_mean，保留原装饰器/断言，直接使用其3组参数运行通过（mask内NaN按原测试预期传播）。另FP32/FP64 × NaN/Inf × 表中4项及1项no_grad白化，共20组真实CPU检查通过；不是完整pytest、设备或训练复现。

```bash
python experiments/scripts/verify_verl_masked_stats.py /path/to/verl
```

Python3.12.13、torch2.9.1、pytest8.4.2。官方API/本地torch_functional.py Git blob均 `490ffc1baf246a452faad82ac3f00f448303ba81`；模块SHA256 `9855e101c19b64c90d1f07bf47ce3e628751afe979b40d2b863ae6d16aa4cee1`，原测试 `341c60a490aa50e04b36cf700acb2558d8a96971732136aac8051e6845c9e0e4`。诊断时应分别定位首次非有限输入、所在mask及其计算图，而非只盯最终归约值。本项无需新增M5硬件任务或自动应用修复。

## 四、GRPO vs PPO 生产对比

| 方面 | PPO (GAE + vanilla) | GRPO (GRPO + vanilla) |
|------|---------------------|----------------------|
| 模型数 | 4 (actor+critic+ref+reward) | 2 (actor+ref) |
| Advantage | GAE(γλ折扣) | 组归一化(outcome-only) |
| Reward | Step-level (每token) | Outcome-level (整个response) |
| Critic | 必须训练V(s) | **不需要** → 省50%内存 |
| 长推理 | 不擅长 (step reward难以定义) | **擅长** (outcome-only自然涌现) |
| 稳定性 | clip_ratio参数敏感 | 组归一化更鲁棒 |
| 前缀共享 | 不适用 | n=8共享prefix → 58%KV节省 |

## 五、与我们的工作联系

### Prefix Sharing对GRPO的优势

GRPO n=8 → 同prompt采样8个response → prefix共享:
- PrefixGrouper (PR #4368): 仅attention层 → 0.99x加速
- Full-model PS: Provider完整forward + Reuser suffix-only → 2.46x加速
- 训练加速: 1.59x (×0.76打折)

### core_algos.py对PS的影响

1. **advantage计算不受PS影响**: GRPO advantage = (r - μ_group)/σ_group → 在reward计算之后 → PS只影响rollout阶段
2. **policy loss不受PS影响**: ratio = π_new/π_old → 与forward方式无关
3. **KL penalty**: `r(x,y) - β KL(π_new || π_ref)` → ref_log_prob需要独立计算 → **但ref_in_actor=True可以省一个模型**

### verl #6401贡献机会

core_algos.py的注册表设计意味着:
- 新的advantage estimator: 只需写函数 + `@register_adv_est`
- 新的policy loss: 只需写函数 + `@register_policy_loss`
- PS相关的修改: 不在core_algos.py → 在rollout层(ActorRolloutRefWorker)

Sources:
- verl core_algos.py: verl/trainer/ppo/core_algos.py (2487行)
- GRPO原论文: DeepSeekMath (Shao et al., 2024), arXiv:2402.03300
- PPO-clip: Schulman et al., 2017, arXiv:1707.06347
- Dual-clip PPO: Cheng et al., 2019, arXiv:1912.09729
- Dr.GRPO: arXiv:2503.20783 (norm_adv_by_std_in_grpo=False)
- GDPO: Group reward-Decoupled Normalization
