# verl GRPO Training Core Algorithms — core_algos.py 源码深度阅读

> 2026-06-07 | 2487行源码分析: 14种Advantage Estimator + 10种Policy Loss + 注册表架构

2026-10-25接续：[KL-Cov 的选词范围](#kl-cov-的选词范围与微批切分)说明批处理切分如何改变选择性惩罚；以下早期算法数量和行号保留历史含义，不作当前版本清单。

2026-10-25接续：[逐轮奖励与停止动作的基线](#逐轮奖励与停止动作的基线)区分#7244的LOO return-to-go提案与#8305的过程优势塑形；奖励可按token存储，不代表GRPO保留逐轮credit。一个精确二元代数例仅说明基线依赖动作的偏差，不认证候选或训练效果。

## 逐轮奖励与停止动作的基线

**逐轮奖励字段、逐轮优势估计与无偏停止决策是三件事。** [#7244](https://github.com/verl-project/verl/issues/7244)作者lijuliana已有fork实现，讨论variable horizon下基线求和上限依赖本条轨迹的停止位置；[另一参与者](https://github.com/verl-project/verl/issues/7244#issuecomment-6122893289)正在询问缺失／延迟反馈方向。本轮不重复其MDP实验或实现。按编号查开放PR为空，但关键词查重发现另一作者luzy99的[draft #8305](https://github.com/verl-project/verl/pull/8305)，不能据编号空结果认定逐轮接口无人做，也不能把它标作已修复该RFC。

### 主线如何消费奖励

固定2026-10-25核验main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`：[原GRPO函数](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/core_algos.py)先将 `token_level_rewards.sum(-1)`压成每条响应一个score，按UID组统计后将同一优势广播到该行有效token。因此输入可以含中间奖励，但此路径不保留这些奖励分别属于哪一轮的credit；不能写成“框架只能接最终奖励”。[原compute_advantage](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/ray_trainer.py)的GAE分支另消费token rewards及values，不在本节GRPO结论内。

[V1多输出helper](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/utils.py)对GRPO只选择每个uid/session最后一个输出计算，再广播给该session其他输出；其他estimator委托旧helper。这里的output index不自动等于对话turn。字符串注册名虽可被 `register_adv_est`接受，注册本身也不会让调用者自动传入turn边界／奖励字段。

### 基线项独立还不够

RFC讨论的危险形式是 `B[i,l] = Σ(j=l…m_i−1) μ_I[j] + μ_O`：即使每个均值都来自其他轨迹，**取多少项**仍由本条轨迹的未来停止位置mᵢ决定。给定当前状态，若当前动作影响mᵢ，baseline就可能依赖该动作；`detach`只能取消autograd穿过baseline，不能消除score-function期望中的相关性。

一个二元logit的精确解释例：p=π(continue)=1/4，continue／stop的条件回报分别2／0，score分别1−p和−p。记两动作分别减去B_c、B_s，则期望更新为：

```text
g = p(1−p) × [(G_c−G_s) − (B_c−B_s)]
true policy gradient = p(1−p) × (G_c−G_s)
baseline-induced bias = −p(1−p) × (B_c−B_s)
```

标准库Fraction精确枚举这两分支：不减基线得到3/8；仅continue分支减1得到3/16；两个分支都减1仍为3/8。这里B_c=1、B_s=0是**人为给定的解释条件**，不是从#7244或#8305代码复现的结果。它说明“停止动作看见不同baseline”足以改变梯度，并不证明任意逐轮归一化必有相同偏差，也不测方差、MSE或学习收益。

RFC提出按turn index、在其他到达该轮的成员上取realized return-to-go的LOO均值。通常需要给定prompt／既有历史后，其他样本对本条当前动作条件独立；单成员／没有同行、共享采样／筛选造成相关性、反馈缺失和自含样本的标准差缩放仍须单独定义。去掉一个baseline依赖通道，不等于所有归一化／clipping／off-policy更新都已无偏。作者的tabular梯度、方向相似度及收益数字仅属其报告，本轮没有重跑或推广到LLM。

### 已有过程塑形 draft 的范围

2026-10-25读#8305为OPEN/draft，head `07d3a6d6f241b09826a78f2c605dbec9272d347f`、目标base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`；正文不claim修复#7244。它在已有outcome估计之后加入process_scores塑形actor优势，保留原outcome奖励与critic returns；[原数值模块](https://github.com/verl-project/verl/blob/07d3a6d6f241b09826a78f2c605dbec9272d347f/verl/trainer/ppo/process_reward.py)的additive／gvpo／process_weighted三分支不是RFC的LOO return-to-go估计器。逐轮scalar广播到该轮token，也不等于给之前动作计算累计未来奖励。

该draft的直接 `turn_rewards`优先于scorer评估，零值仍是显式输入；`generated_turn_spans`是response坐标的半开区间，默认按response_mask连续生成段回退。合法边界／有限标量只验证数据契约，不认证奖励目标、基线独立性或credit正确。trainer接线在rollout correction后用修改后的mask排掉被拒绝token，与[拒绝前的off-policy诊断分母](verl-async-logprob-provenance.md#padding-行与序列指标的分母)须分开。

本轮仅三完整模块AST／内存compile（main core_algos、V1 utils、draft process_reward）与调用点阅读，加上述Fraction代数。没有执行框架／候选／原测试、MDP、TQ、scorer或训练；19文件目录不等于全patch审核，作者36／13／2项CPU报告非本项目结果，GPU学习质量仍是其待验条件。一条机器人“draft未review”评论与reviews空不作人工认可。本题保存语义与查重结论，不新增M5实验队列、改专项、应用draft或发布社区动作；后续只有实质反馈／明确任务才接续，不扩二元MDP或抢先实现逐轮算法。

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
1. **整条响应一个score**：`scores = rewards.sum()`可汇总中间奖励，但随后广播同一优势，不保留逐token／逐轮credit；不是输入只能含最终奖励，见上方2026-10-25接续
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

单样本没有组内互为baseline的比较，代码采用近似原奖励的回退；它不是标准中心化公式在n=1上的直接计算结果。表中整数同分pair确实返回零，但不能推广成任意小数/组规模都逐位为零：浮点求均值可能留下残差，再被近零std放大，见下方2026-10-23补充。默认epsilon避免直接以零作分母，不认证零分子或“同分必NaN”。也不能从 `advantages.std()==0` 推出所有优势为零：常数非零向量同样标准差为零。诊断应分开检查组内样本数、reward差异、有效mask、优势绝对值及真实梯度/参数变化。

这纠正了[排错指南](../grpo-training-debugging-guide.md#2-loss-values-and-zero-advantages)原来的“n=1必退化、禁止n=1、必须n≥4或MoE≥8”表述。实现没有给出这些统一硬门槛；本轮也没有比较采样成本、训练收敛或模型质量，不能把保留非零优势解释为n=1必然有效学习。关闭std归一化仍保留多样本组的均值中心化，不是统一把所有组变成原奖励。

[复查脚本](../../experiments/scripts/verify_verl_grpo_groups.py)AST选择完整原enum/registry/注册装饰器、两个estimator及四个groupwise helper，保留原函数体/装饰器，延后注解求值以省略无关配置依赖。原helper始终收到显式CPU device，未导入设备发现模块或调用get_device_name。原测试的低方差1例和随机等价性3参数例直接调用全过；随机helper明确保证每组至少两样本，因此还需额外singleton对照，不能把原随机覆盖当作已测n=1。

额外6种布局（正/负/零singleton、同分pair、有差异pair、三类组混合）×标准化开/关，共12组双实现CPU检查通过，同时核输出/returns、有效mask及no_grad结果；没有模型或训练替身。三个原随机例最大绝对差约2.38e−7、2.38e−7、4.77e−7，是给定容差内等价，不是逐位一致或性能证明。

```bash
python experiments/scripts/verify_verl_grpo_groups.py /path/to/verl
```

Python3.12.13、torch2.9.1、NumPy2.0.2、pytest8.4.2；非完整pytest、训练、loss/optimizer或DP/GPU验收。官方API/本地Git blob：core_algos `9d08b8461a0f95d25f348c566358360bbfb72ece`、groupwise `84e79ac903082c913f479f3cea6eed54941e9c16`。SHA256依次 `a54b6d812db4441e8b0989ae69d2c46b0a0a7b584296f73d8316f264462c37cb`、`409403d0002ff0d1fd1c41812163b1cfa52a52248677aab00e6ffcf3b937b442`；原测试模块 `0a60e917589ab91646989f0a8616f810fea0dc70047d2b8c13914f4977f5569c`。只纠正知识与诊断规则，不改上游算法或新增M5任务。

### 同分小数奖励的舍入残差

2026-10-23 新 draft [#8349](https://github.com/verl-project/verl/pull/8349) 为已存在的两条 GRPO 路径补精确同分处理，固定 head `0e3c3e3667fc9764c6d0e554c2feb9a904d7519e`、目标 base 与共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`（ahead 1 / behind 0，2 文件 67 增 3 删）。这补充了上节的数值范围，不撤回 singleton 的 mean0/std1 结论；已有作者实现，不另开竞争修复。

数学上 `r−mean(r)=0`，但 FP32 归约未必把一组相同小数求回原数值。残差即便只有一个 ulp，除以 `std+1e-6` 后也可不再微小。以下是本机 torch2.9.1 CPU、每行单token奖励、默认std归一化的观测，不是各硬件上的固定值：

| 组内每行奖励与数量 | base loop | base vectorized | head 两实现 |
| --- | --- | --- | --- |
| 0.3 × 16 | 0 | 0 | 0 |
| 0.7 × 64 | 0 | −0.2937158942 | 0 |
| 0.9 × 64 | 0 | −0.3481845260 | 0 |

作者在其CPU报告0.3×16两实现约+0.0289，本机该例是零；归约/构建差异决定具体反例，不能机械照抄作者数字，也不能据一个布局碰巧为零否定其他布局的残差。旧pair4/4控制和“两个实现容差内等价”的测试均不能认证所有同分组严格归零。

loop候选在组大小>1且 `scores_tensor == scores_tensor[0]` 全成立时，直接用原score作mean、std=1；vectorized候选保留原统计，另按每组min/max精确相等且count>1把scalars置零，再乘response_mask。比较对象是每行 `token_level_rewards.sum(-1)` 后的score，不是逐token原奖励或近同分阈值。singleton不满足count>1，仍保留原回退；非同分组不应因这项保护被抹平。`group_mean_std` 本身没有修改，新增scatter/主机布尔判断的设备开销没有测量。

[同一复查脚本](../../experiments/scripts/verify_verl_grpo_groups.py)新增 `--uniform-pr`，两次CPU结果一致：原新增15个reward×group_size参数例加1个混组例，base **3过、13项预期失败**，head **16过**；每个同分原例内部再遍历两个实现与std开关。原低方差测试在两版通过，正/负singleton在两种std模式、两实现也保持原约定。原历史模式在有1876b06d对象的源码仓库回归，4原例/12组控制仍通过；初次误用不含该历史对象的另一克隆失败，未换版本冒充回归。原函数/装饰器及原测试体AST执行，未原生收集作者39项/20项套件，也没有policy loss、梯度、optimizer、DP/GPU、训练或性能验收；不推断 #8250 根因或修复后训练收益，不新增M5任务。

```bash
python3.12 -B experiments/scripts/verify_verl_grpo_groups.py /path/to/verl --uniform-pr
```

本机 Python3.12.13 / torch2.9.1 / NumPy2.0.2 / pytest8.4.2，无新依赖。core_algos SHA256 base `369b614f239f237a04a38a40b1877cd6fd14aa7bd5f9ff4fe4d7758b682a6948` / head `7cde90331de52f4185d9a831244fcef70261c95d0c9166a2ac88a62f541e87fd`；新原测试模块 `44a9f1ecce251736bdf0620841f8d8e35527b32c750482d3f7db12a7bce5d78d`。这是CPU估计器数值范围的纠正，draft未自动采用。

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

### RLOO 的单样本回退与输出精度

RLOO对多样本组用“自身奖励减去其他样本均值”；n=1时没有其他样本，需显式选择回退约定，不能直接套除以n−1的公式。[#7793](https://github.com/verl-project/verl/issues/7793)报告标量版保留原score、向量版清零。已有[#7794](https://github.com/verl-project/verl/pull/7794)改为`torch.where(c > 1, loo, scores)`；[#7964](https://github.com/verl-project/verl/pull/7964)另修group_sum的dtype。这是两项独立契约，不能把任一修复看作覆盖另一项。相邻行可能冲突，但不需要第三份竞争实现；另有[#7581](https://github.com/verl-project/verl/pull/7581)和较大范围的[#7150](https://github.com/verl-project/verl/pull/7150)覆盖。

2026-10-25核上述四PR均open、非draft、未合并。本次固定#7794 head `58b3773730ad8b4e7c363bc32f9f5216c734ac12`／父提交及祖先 `f1c271417bddfbf97b41a261819d04319a70c603`，两文件26增1删；#7964 head `ea2368905318c569355cdae8b40b01e0fe045ccd`／父提交及祖先 `f542ae22b730b3736d3a46e79d9d8848813a2d48`，两文件32增1删。两项API目标base均为`3f9d4c0d`，不能替代实际比较祖先。固定main `3c6d9b43` 的两个完整RLOO函数AST仍与这两个父版本相同。

本机Python3.12.13／torch2.14.1／NumPy2.5.4，完整原枚举、注册装饰器及两个函数AST接真实CPU张量，未改函数体，也未拼两候选。原新增测试体隔离执行结果如下；#7964的测试只断言dtype，不检查输出数值：

| 原回归范围 | 对应base | 对应head |
| --- | --- | --- |
| #7794混合组含singleton，1例 | AssertionError | 通过 |
| #7964双样本组，FP32／BF16／FP16三例 | FP32通过，低精度两例失败 | 三例通过 |
| #7794同一singleton测试放到#7964 head | — | 仍失败，dtype修复不处理此语义 |

**dtype保真不意味着数值不变。** 在本机CPU上，`bincount(weights=...)`的FP32输入返回FP32，BF16／FP16返回FP64；[PyTorch2.14.1原CPU实现](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/SummaryOps.cpp)对Float权重使用float模板，其余已定义权重转Double后使用double模板。因此不应照抄PR注释“任何dtype都累积为FP64”为通用事实，也不把本次CPU行为外推CUDA。#7964把group_sum转回scores.dtype后再算优势，这既改输出类型，也改变中间舍入。单一BF16组`[256,1]`（两数均可精确表示）、全1同dtype mask的原函数对照为：

| 计算路径 | 输出dtype | 两行优势 |
| --- | --- | --- |
| 原vectorized／只修singleton的#7794 | FP64 | [255,−255] |
| #7964及原scalar | BF16 | [256,−254] |
| 将同样输入转FP64的scalar参考 | FP64 | [255,−255] |

本例group_sum的257转BF16变成256；标量路径同样有低精度均值舍入，因此“与标量一致”不能证明更接近高精度公式。它是dtype契约与精度取舍的反例，不是训练退化或候选整体错误的证据。两项函数最后都乘response_mask；#7964本例若只把mask换为FP32，结果也为FP32，而非强制保持奖励BF16。原dtype测试的mask与奖励同dtype，结论应保持这一条件，实际调用另核。

**供用户审视的评审要点**：请已有#7964作者限定“FP64无条件累积”和“same values”的描述，并为cast前后的数值／容差补覆盖。当前新测试验证了类型与标量类型一致，没有验证数值误差；无需据此再做同一dtype补丁。反馈只保存在个人项目，不评论／编辑PR或推送其head，任何外部动作仍待用户逐项明确同意。

原函数返回优势与returns的同一张量，处于no_grad；这里非零优势不等于已经验证actor梯度、参数更新或训练收益。八份完整源码快照compile通过，未运行作者报告的原生24／26项套件、trainer、模型或设备。版本、命令及源文件校验见[当日日志](../../diary/2026-10-25.md)；本题不新排M5算力任务，也不重复上文GRPO同分组实验。

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

#### KL-Cov 的选词范围与微批切分

**统一归约分母，不能自动修复跨样本选词范围。** [#7829](https://github.com/verl-project/verl/issues/7829)报告每个 microbatch 各自计算均值、协方差和 top-k，因而改变切分就会改变惩罚目标。已有报告者 gss10282025 的 [#7830](https://github.com/verl-project/verl/pull/7830)，2026-10-25核对 OPEN／非draft／未合并，head `79a7af1c7ddd77e7c188943c0fab526047631af2`、父提交／实际共同祖先 `6fc912f07d225d7c96708ab081ef2e6b00db74d7`；API目标base `3f9d4c0d`不是候选已包含的主线。八文件531增26删。旧[#7828](https://github.com/verl-project/verl/pull/7828)关闭未合并、head相同，不另实现；另一[#6538](https://github.com/verl-project/verl/pull/6538)处理ratio稳定性，不替代选词范围修复。

[`compute_kl_cov_mask`](https://github.com/verl-project/verl/blob/79a7af1c7ddd77e7c188943c0fab526047631af2/verl/trainer/ppo/core_algos.py)只对有效token计算 `(A-mean(A))*(logp-mean(logp))`，选 `max(1, int(N*ratio))` 个最大值；空有效集返回全False。原测试用8192 token、ratio0.0002，整批选1个，切成八块后每块至少1个，共8个。修正取整数量仍不足够：各块的均值与候选集合也不同。

候选在FSDP／FSDP2 actor的每个 optimizer minibatch 前额外执行一次 no-grad 当前策略前向，汇集DP组的有效advantage／logp到CPU，统一选择后按rank与microbatch切回mask。`prepare_batch_fn`在训练上下文内、engine.train_batch之前执行，每次更新／后续PPO epoch重新计算；源码用fork_rng恢复额外前向消耗的随机流。distillation-only跳过，预计算副本禁用teacher top-k loss；原PPO损失仍保留entropy、reference KL和importance weights。这是额外前向与数据交换，不是免费提升吞吐。其他engine或直接调用者若不传mask，仍保留局部选择。

本机真实CPU运行原纯测试模块三个函数／五个参数例，全通过：1对8配额、三种ratio下padding／非连续输入／空集，以及整批预计算与直接调用的loss和logp梯度一致。配置为十一原dataclass类AST，数学为原mask／loss／agg_loss及masked_sum／mean，类体和函数不改、仅延迟类型注解；原测试imports／assert不改但直接调用，不称原生pytest收集或完整verl导入。

补充一个**配额已相同**的FP64夹具：两行各3 token，advantages=`[[-4,4,1],[-.2,.1,.3]]`，logp=`[[-4,-.2,-2],[-2.2,-1.9,-2]]`，old=logp−0.1，ratio=1/3、惩罚系数0.8。整批和两块都选2个，但整批扁平索引为`[0,1]`，局部为`[0,3]`。统一分母为6、逐块loss相加，结果如下；这是输入logp的梯度，不是模型参数更新。

| 调用方式 | 选词范围 | logp梯度前四项 |
| --- | --- | --- |
| 祖先整批 | 整批 | 0.870114，−0.603447，−0.184195，0.036839 |
| 祖先切两块／候选不传mask切两块 | 各块 | 0.870114，−0.736781，−0.184195，0.170172 |
| 候选传整批mask的对应切片 | 整批 | 与祖先整批assert_close |

三路loss均约−0.194367516948463，差异仅浮点归约末位，因为每个被选token的当前abs(logp−old)都是0.1；相同标量值没有暴露惩罚位置变化。候选DP helper的dp_group=None单进程路径也与整批mask一致，未运行collective。这个例子支持先固定统计与选择域、再讨论归一化，不证明完整模型或随机前向的切分不变性。

完整worker原测试使用作者的可微CPU engine夹具、worker壳与真实PPO损失，含两Gloo rank；作者25／87项报告不能挪作本机结果。本轮该模块及两个生产接线文件仅compile，未执行worker、RNG恢复、Gloo／FSDP或GPU，也未复跑issue固定GPU实验。[复现与hash](../../diary/2026-10-25.md)留日志。本题收束为算法知识，不应用补丁、另写选词器或新增M5算力任务。

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
2. **policy loss 的条件**：只有有效输入、归一化及跨样本选词范围都保持一致，才可讨论切分等价；[KL-Cov例](#kl-cov-的选词范围与微批切分)表明仅ratio公式相同不足保证目标不变，不能据此前向分组方式直接推导等价。
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
