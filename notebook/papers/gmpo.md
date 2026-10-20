# GMPO: Geometric-Mean Policy Optimization（论文精读）

> [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) · Yuzhong Zhao et al. · 2025-07
> 精读时间：2026-10-19 · 关联：[RL 损失函数推导](rl-policy-loss-functions-mathematical-derivation.md) · [GRPO 综合](grpo-training-algorithm-unified-synthesis.md)
> 动机：verl #7913 geo_mean 损失修复（candidate-7913）需对照论文验证实现语义。

> **2026-10-21复核边界**：下文GMPO完整公式/clip等价性本轮未重审，不再作为“RFC证据完成”的依据。此前引用GSPO未归一乘积及#8250紧窗因果的说法已撤回；见 [GSPO更正](gspo.md) 与 [退回重审的草稿](../../open-source/verl/verl-clip-semantics-rfc-draft.md)。

## 一句话

GRPO 对 token 级奖励求**算术**平均，遇离群 importance ratio 时更新不稳；GMPO 改求**几何**平均，同一序列内 outlier ratio 被几何均值吸收，权重更稳。

## 目标函数（Eq. 4）

$$
\begin{aligned}
J_{\mathrm{GMPO}}(\pi_\theta)
&= \mathbb{E}_{q,\{o_i\}\sim\pi_{\mathrm{old}}}
\frac{1}{G}\sum_{i=1}^{G}
\Biggl\{\prod_{t=1}^{\lvert o_i\rvert}
\Bigl|\min\Bigl[
\rho_{i,t}(\theta)\widehat{A}_i,\\
&\qquad\operatorname{clip}\bigl(\rho_{i,t}(\theta),\epsilon_{\mathrm{low}},\epsilon_{\mathrm{high}}\bigr)\widehat{A}_i
\Bigr]\Bigr|\Biggr\}^{1/\lvert o_i\rvert}
\operatorname{sgn}(\widehat{A}_i).
\end{aligned}
$$

逐 token 的 ratio ρ 与 clip 在**概率空间**进行（论文 Algorithm 1 用 epsilon=0.4），序列内做**几何均值**（log 空间乘积防数值溢出），乘序列优势 Â_i 与符号。等价变形：

$$
=\widehat{A}_i\cdot
\left(\prod_t\min\bigl(\rho_t,\operatorname{clip}(\rho_t)\bigr)\right)^{1/\lvert o_i\rvert}
$$

## 梯度对比（Eq. 5 vs 6）——核心论点

| | 每个 token 的梯度权重 |
|---|---|
| GRPO | $\rho_{i,t}(\theta)$（自身 importance ratio）——单个离群 token 直接放大/压死梯度 |
| GMPO | $\left(\prod_k\rho_{i,k}(\theta)\right)^{1/\lvert o_i\rvert}$（全序列 ratio 的几何均值）——离群被均值吸收 |

两者都是 policy gradient 的加权和（权重 × Â_i × ∇log π），GMPO 权重更稳。

## 与 verl 实现对照（2026-10-19 源码核对）

**geo_mean（core_algos.py:1928）与论文 Eq.4 数学等价**：

| 论文 | verl geo_mean |
|---|---|
| clip(ρ, ε_low, ε_high)（概率空间） | **clamp(log ρ, -clip_ratio_low, clip_ratio_high)（log 空间）** ⚠️ 语义不同 |
| Π_t min(...) 几何均值 | exp(mean_t min-clamped log ρ)（同序 log 空间）✓ |
| 序列优势 Â_i | 序列优势（"只支持 sequence-level advantage"）✓ |
| loss = -Â·geo_mean_ratio | pg_losses = -advantage·ratio ✓ |

**待重审的历史观察**：源码裁剪对象和配置空间需要准确说明，但不能用[#8250](https://github.com/verl-project/verl/issues/8250)证明紧窗导致无法恢复；该因果表述已撤回。小参数下log/ratio对应窗口本来接近，完整目标还须检查正负优势、裁剪与聚合的顺序，不能只凭一个标量恒等式认证等价或提出RFC。

## 对 #7913 修复的验证

论文目标 = prompt 级平均（1/G Σ_i …）→ verl 修复用 `agg_loss(seq-mean-token-sum)` 保证**任意 microbatch 切分下等于全 minibatch 平均**，与论文语义一致。修复后行为 = 论文目标；修复前（torch.mean 局部）在拆分时偏离论文目标。

## 实验结论（论文）

- 数学推理基准 Pass@1 平均 +4.1%（7B，vs GRPO）
- 实现细节：log 空间乘积+clip；ε 典型 0.4

## 延伸问题

1. ~~verl geo_mean 的 log 空间 clamp 与论文 ρ 空间 clip 在**紧窗区**（KL<0.001）行为差异可忽略，但在离群区（KL>0.1）差异显著~~ → **2026-10-19 数值证明：完全等价**（experiments/scripts/verify_gmpo_clip_semantics.py）。因 `log(min(ρ,ε)) ≡ min(log ρ, log ε)`，窗口对应时前向值与梯度逐位一致（仅 torch clamp 边界点的 inclusive 约定造成测度零差异）。**真正的差异是配置语义**：同一数值在论文是 ρ 空间窗口（ε=0.4 → ρ∈[0.67,1.49]），在 verl 是 log 空间窗口（ε=0.4 → KL ±0.4 nats；ε=3e-4 → ρ∈[0.9997,1.0003]）。
2. **2026-10-21撤回旧GSPO比较**：GSPO论文Eq.7本身使用长度归一几何均值，Eq.14包含GSPO-token的stop-gradient。中间Jacobian不同不等于最终目标梯度不同；修正脚本在恒定序列优势/相同token均值归约的限定36组中一致，非完整训练等价证明。旧“论文求和而verl求均值、L倍差异”不成立，详见 [更正](gspo.md)。
