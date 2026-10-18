# SAPO: Soft Adaptive Policy Optimization（论文精读）

> [arXiv:2511.20347](https://arxiv.org/abs/2511.20347) · Qwen Team (Alibaba) · 2025-11 · 用于训练 Qwen3-VL 系列
> 精读时间：2026-10-19 · 关联：[GMPO](gmpo.md) · [GSPO](gspo.md) · [GRPO 综合](grpo-training-algorithm-unified-synthesis.md)
> 动机：完成 verl 注册的新损失家族三联（GMPO/GSPO/SAPO）对照。

## 一句话

硬裁剪（GRPO/GSPO）的信任域是二值的：带内 token 全保留、带外全丢弃；SAPO 用**温度控制的平滑门** `sech²` 自适应衰减 off-policy 更新——既保留有效学习信号，又压低不稳定 token，且对正负优势用**非对称温度**（τ_neg > τ_pos 最稳）。

## 目标（梯度形式，Eq. 22）

```
∇J_SAPO ≈ E[ (1/G) Σ_i (1/|y_i|) Σ_t g_{τ_i}(z_{i,t}(θ)) · ∇log π_θ(y_{i,t}) · Â_i ]

门: g_τ(z) = sech²(τ/2 · z)，z = log ratio（token 级 z_{i,t}，序列级 log s_i(θ)）
```

## gating 视角（第 4 节）——与 GRPO/GSPO 的统一框架

| 方法 | 门函数 | 性质 |
|---|---|---|
| GRPO | f' = 1（带内）/ 0（带外）分段常数（Eq.24） | **硬二值信任域**：带外 token 零梯度 |
| GSPO | 序列级 hard clip 1±ε | 序列相干但硬；**一个离群 token 压死整条序列梯度** |
| SAPO | sech²(τ/2·z) 平滑 | 连续信任域；token 自适应：只降权离群 token，保留近 on-policy token 的信号 |

**SAPO–GSPO 连接**（4.1）：在假设 (A1) 小步长、(A2) 序列内 log-ratio 低离散度下，token 级门 ≈ 序列级门 g(log s_i(θ))——SAPO 退化为带平滑门的 GSPO。MoE 模型 (A2) 略弱（expert routing 异质性），密集模型近似更好。
**SAPO–GRPO 连接**（4.2）：硬指示函数替换为平滑核，避免优化噪声。

## 非对称温度（核心经验发现）

- **τ_neg > τ_pos 最稳**（实验：τ_neg=1.05 > τ_pos=1.0 稳定；τ_neg=0.95 < τ_pos=1.0 显著不稳）。
- 解释：负优势 token 的梯度对训练不稳定性贡献更强——负侧需要更强的衰减门。
- 与 #8250 案例呼应：熵爆炸/坍塌主要由负优势 token 的异常梯度驱动（我们的分析也指向优势离群）。

## verl 实现对照（源码核对，core_algos.py:1623）

```python
def gate_function(x, tau):  return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)
taus = torch.where(advantages > 0, tau_pos, tau_neg)   # 非对称温度 ✓ 论文一致
gates = gate_function(ratio, taus)                     # ratio 空间、token 级
pg_losses = -gates * advantages
```

- **设计意图忠实，函数形式不同**：论文 sech²(τ/2·log s)（log-ratio 空间）；verl sigmoid(τ(x-1))·4/τ（ratio 空间，常数 4/τ 归一）。两者都是"ratio=1 处门最大、离 1 越远衰减越强"的平滑门，但衰减形状与归一不同——同一 clip 语义不一致家族（GMPO log 空间 vs GSPO ratio 空间 vs SAPO sigmoid 变体）。
- 默认 `loss_agg_mode="seq-mean-token-mean"`（docstring 推荐）。

## 家族总结（GMPO / GSPO / SAPO + verl 实现忠实度）

| 损失 | 论文机制 | verl 实现 | 忠实度 |
|---|---|---|---|
| GMPO | token ρ 的几何均值 + ρ 空间 clip(ε=0.4) | log 空间几何均值 + **log 空间 clamp** | 目标等价；clip 空间不一致 |
| GSPO | 序列 ratio s + ratio 空间 clip(1±ε) | combined ratio + ratio 空间 clamp | clip 忠实；示例默认 ε=3e-4 极紧 |
| SAPO | sech² 平滑门 + 非对称温度 | sigmoid 门 + 非对称温度 ✓ | 温度忠实；门函数为变体 |

## 延伸问题

1. verl 三个损失的 clip/门语义各不同（log 空间 / ratio 空间 / sigmoid 变体）——文档澄清候选（与 gmpo.md 的发现合并）。
2. SAPO 门在 ratio 空间（verl）与 log-ratio 空间（论文）的数值差异：ratio 紧邻 1 时近似线性等价，远离 1 时差异放大——可数值验证。
3. Qwen3-VL 用 SAPO 训练 + verl 支持 SAPO → verl 的 SAPO 路径值得专门回归（#8250 同款紧窗配置问题在 SAPO 下表现不同）。
