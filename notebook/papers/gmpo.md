# GMPO: Geometric-Mean Policy Optimization（论文精读）

> [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) · Yuzhong Zhao et al. · 2025-07
> 精读时间：2026-10-19 · 关联：[RL 损失函数推导](rl-policy-loss-functions-mathematical-derivation.md) · [GRPO 综合](grpo-training-algorithm-unified-synthesis.md)
> 动机：verl #7913 geo_mean 损失修复（candidate-7913）需对照论文验证实现语义。

## 一句话

GRPO 对 token 级奖励求**算术**平均，遇离群 importance ratio 时更新不稳；GMPO 改求**几何**平均，同一序列内 outlier ratio 被几何均值吸收，权重更稳。

## 目标函数（Eq. 4）

```
J_GMPO(π_θ) = E_{q,{o_i}~π_old}  (1/G) Σ_i  { Π_{t=1}^{|o_i|} |min[ρ_{i,t}(θ)·Â_i, clip(ρ_{i,t}(θ),ε_low,ε_high)·Â_i]| }^{1/|o_i|} · sgn(Â_i)
```

逐 token 的 ratio ρ 与 clip 在**概率空间**进行（论文 Algorithm 1 用 epsilon=0.4），序列内做**几何均值**（log 空间乘积防数值溢出），乘序列优势 Â_i 与符号。等价变形：

```
= Â_i · (Π_t min(ρ_t, clip(ρ_t)))^{1/|o_i|}
```

## 梯度对比（Eq. 5 vs 6）——核心论点

| | 每个 token 的梯度权重 |
|---|---|
| GRPO | **ρ_{i,t}(θ)**（自身 importance ratio）——单个离群 token 直接放大/压死梯度 |
| GMPO | **(Π_k ρ_{i,k}(θ))^{1/|o_i|}**（全序列 ratio 的几何均值）——离群被均值吸收 |

两者都是 policy gradient 的加权和（权重 × Â_i × ∇log π），GMPO 权重更稳。

## 与 verl 实现对照（2026-10-19 源码核对）

**geo_mean（core_algos.py:1928）与论文 Eq.4 数学等价**：

| 论文 | verl geo_mean |
|---|---|
| clip(ρ, ε_low, ε_high)（概率空间） | **clamp(log ρ, -clip_ratio_low, clip_ratio_high)（log 空间）** ⚠️ 语义不同 |
| Π_t min(...) 几何均值 | exp(mean_t min-clamped log ρ)（同序 log 空间）✓ |
| 序列优势 Â_i | 序列优势（"只支持 sequence-level advantage"）✓ |
| loss = -Â·geo_mean_ratio | pg_losses = -advantage·ratio ✓ |

**⚠️ 发现——clip 空间不一致**：论文在 ρ 空间 clip（ε=0.4 宽窗）；verl 在 log ρ 空间 clamp（±clip_ratio）。当用户按论文语义传 ε=0.4 时，verl 得到 ρ ∈ [e^-0.4, e^0.4]=[0.67,1.49]（接近论文意图）；当传小值如 0.0004 时，ρ ∈ [0.9996, 1.0004]——**极紧窗**（见 #8250 案例：该配置使绝大多数 token 梯度饱和，漂移后无法恢复）。同一 clip_ratio 配置在 GSPO（概率空间 clamp(ratio, 1-ε_low, 1+ε_high)）与 geo_mean（log 空间 clamp）之间语义还不一致。这是潜在的贡献/文档候选点。

## 对 #7913 修复的验证

论文目标 = prompt 级平均（1/G Σ_i …）→ verl 修复用 `agg_loss(seq-mean-token-sum)` 保证**任意 microbatch 切分下等于全 minibatch 平均**，与论文语义一致。修复后行为 = 论文目标；修复前（torch.mean 局部）在拆分时偏离论文目标。

## 实验结论（论文）

- 数学推理基准 Pass@1 平均 +4.1%（7B，vs GRPO）
- 实现细节：log 空间乘积+clip；ε 典型 0.4

## 延伸问题

1. ~~verl geo_mean 的 log 空间 clamp 与论文 ρ 空间 clip 在**紧窗区**（KL<0.001）行为差异可忽略，但在离群区（KL>0.1）差异显著~~ → **2026-10-19 数值证明：完全等价**（experiments/scripts/verify_gmpo_clip_semantics.py）。因 `log(min(ρ,ε)) ≡ min(log ρ, log ε)`，窗口对应时前向值与梯度逐位一致（仅 torch clamp 边界点的 inclusive 约定造成测度零差异）。**真正的差异是配置语义**：同一数值在论文是 ρ 空间窗口（ε=0.4 → ρ∈[0.67,1.49]），在 verl 是 log 空间窗口（ε=0.4 → KL ±0.4 nats；ε=3e-4 → ρ∈[0.9997,1.0003]）。
2. GSPO 的"combined ratio"（当前 token ratio × 序列几何均值 detached）是另一套设计；GMPO 的几何均值含 clip 且**不 detached**——梯度可穿过序列内所有 token（论文 Eq.6 的权重是常数，但其来源含 θ 依赖的 ρ_k；论文推导中几何均值权重按常数处理）。verl 用 `log_prob.detach()` 在 GSPO 中隔离，geo_mean 中无 detach——两者梯度路径不同，值得注意。
