# GSPO: Group Sequence Policy Optimization（论文精读）

> [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) · 2025-07
> 精读时间：2026-10-19 · 关联：[GMPO](gmpo.md) · [GRPO 综合](grpo-training-algorithm-unified-synthesis.md) · verl #8250（GSPO 紧窗配置案例）
> 动机：补全 GMPO/GSPO 损失家族对照；核验 verl GSPO 实现与论文语义。

## 一句话

GRPO 的 token 级 importance weight 有问题；GSPO 改用**序列级** importance ratio `s_i(θ) = π_θ(y|x)/π_θ_old(y|x)`——它衡量整条响应的 off-policy 程度，与序列奖励天然对齐，且是自然的 clip 对象。

## 目标函数（Eq. 5-6）

```
J_GSPO(θ) = E_{x,{y_i}~π_old} (1/G) Σ_i min(s_i(θ)·Â_i, clip(s_i(θ), 1-ε, 1+ε)·Â_i)

Â_i = (r(x,y_i) - mean_G) / std_G          # 组归一优势（同 GRPO）
s_i(θ) = π_θ(y_i|x) / π_θ_old(y_i|x)       # 序列级 ratio = Π_t ρ_t
```

- **clip 在 ratio（概率）空间**：`clip(s, 1-ε, 1+ε)`——PPO 式双面窗。
- 序列 ratio 聚合全响应 token 的对数概率差 → 序列窗口与 token 窗口语义不同（ε 的合适量级取决于对象是 token 还是序列）。

## 与 verl 实现对照（源码核对）

**clip 空间：忠实**。verl GSPO（core_algos.py:1546）：

```python
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
seq_importance_ratio = exp(clamp(log_seq_importance_ratio, max=10.0))
pg_losses1 = -adv * seq_importance_ratio
pg_losses2 = -adv * clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
pg_losses = max(pg_losses1, pg_losses2)   # = min(s·A, clip(s)·A) 对 A 的取大等价
```

- 语义差异（设计选择）：verl 用 **token 级 combined ratio** = 当前 token ρ_t × 序列几何均值 ratio（detached）——梯度只穿过当前 token，序列部分作常数权重（代码注释 s_i,t(θ) = sg[s_i(θ)]·ρ_t/sg[ρ_t]）。论文 s_i 是全序列乘积（含 θ 依赖）。verl 变体是官方代码常见的 stop-gradient 技巧。
- clip 语义（1±ε，ratio 空间）与论文一致 ✓。

## ⚠️ 发现：verl GSPO 示例默认 ε = 3e-4/4e-4（极紧窗）

`examples/gspo_trainer/run_qwen3_30b_a3b_megatron.sh:24-25` 默认 `CLIP_RATIO_LOW=3e-4, CLIP_RATIO_HIGH=4e-4` → combined ratio 窗口 **1±0.0004**（≈ token KL 窗口 ±0.0004 nats）。#8250 用户配置即来自此默认。

- 与 PPO/GSPO 常见 ε=0.2 相差 ~500 倍。
- **论文本身实验 ε 未能在 HTML 提取确认**（开放问题）；若论文 ε 也是 ~1e-3 量级，则 verl 示例忠实；若是 0.2 量级，verl 示例就是误导性配置（#8250 案例中：紧窗 → 梯度饱和 → 漂移后无法恢复）。
- 注：GMPO 论文 Algorithm 1 用 ε=0.4（ρ 空间宽窗）；GMPO 与 GSPO 的 clip 空间与量级各不相同。

## 三向 clip 语义对照（GMPO / GSPO / verl 实现）

| 损失 | 论文 clip 对象 | 论文 clip 空间 | verl 实现 clip | 忠实度 |
|---|---|---|---|---|
| GMPO | token ρ | ρ 空间（ε=0.4 宽窗） | geo_mean：**log ρ 空间 clamp(±clip_ratio)** | ⚠️ 空间不一致（阈值语义随值域漂移） |
| GSPO | 序列 s / token combined | ratio 空间（1±ε） | GSPO：ratio 空间 clamp(1±ε) | ✓ 一致 |
| verl GSPO 示例默认 | — | — | ε=3e-4/4e-4 极紧 | ⚠️ 与 0.2 量级差 ~500×（待论文实验值确认） |

## 延伸问题

1. GSPO 论文实验 ε 具体值（需官方代码 repo——HTML 无链接）；verl 示例 3e-4 的来源（可能来自 GSPO 官方 codebase 的 token 级设置）。
2. verl combined ratio 的 detach 设计 vs 论文全序列乘积：梯度路径不同，收敛行为差异未验证。
3. #8250 案例中 GSPO 紧窗与 GMPO log 空间 clamp 并存，增加了配置语义的混淆面——值得向 verl 提文档澄清。
