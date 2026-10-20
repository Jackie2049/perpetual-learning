# verl：GMPO 损失随小批次切分变化

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** GMPO 损失随小批次切分变化。[Issue #7913](https://github.com/verl-project/verl/issues/7913) · [PR #7916](https://github.com/verl-project/verl/pull/7916)

**【类型】** 漏洞修复

**【价值评估】** 4 / 10 — 技术分析仍可复用，但截至记录核验已有他人方案覆盖，新增贡献空间有限。

**【进展】** 停止重复投入（2026-10-19 核验）：已有 [PR #7916](https://github.com/verl-project/verl/pull/7916) 覆盖；不是我方已合并成果，当前上游状态未重新查询。

**【大白话讲明白】**

同一批训练样本拆成几份计算，理想情况下不应仅因切分就改变同一目标的梯度。我们研究损失归一化是否使用了正确的全局分母，并做了 CPU 对照；社区已存在另一种修复方案。

**【What–Why–How】**

- **What：** geo_mean 损失在 minibatch 切分下的归一化一致性。
- **Why：** 每个小批次单独归一化可能改变总体权重。
- **How：** 本地 fp64 梯度对照验证各切分相符；与上游配置层方案有差异，但不据此重复提交。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> Issue: [verl-project/verl#7913](https://github.com/verl-project/verl/issues/7913)
> 创建 2026-09-18；0 评论；查重 2026-10-19：0 个引用 PR（无人认领，但窗口仅 0-2 天，需尽快）
> 状态：~~**verified-works**（patch 实现 + fp64 CPU 验证：所有切分 rel L2 = 0）~~ → **2026-10-19 复查重纠正：reporter 同日已开 PR #7916（BREAKING: require minibatch normalization，方案与我们不同）**。验证材料保留，不提交；如需发声按政策在 issue 说明差异（需用户授权）。

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| 影响 | 4 — 静默数值错误：同一 minibatch 拆成多 microbatches 时梯度系统性偏移（8 拆 1 时 rel L2=7.0，拆 3+5 时 0.64），训练语义随并行切分漂移，无报错 |
| 成本 | 4 — 8 行改动复用现有 `agg_loss`（entropy/kl 已在用），CPU 可验证，无需 GPU |
| 接受度 | 4 — 明确 bugfix；关联 #5614 表明 aggregation 问题是 maintainer 关注方向 |
| 契合度 | 5 — GMPO/geo_mean 数学推导在我们的 RL 损失深耕区 |
| 分 | **20** |

## 1. Bug

`compute_policy_loss_geo_mean`（verl/trainer/ppo/core_algos.py:1992）以 `pg_loss = torch.mean(pg_losses)` 收尾——microbatch **局部均值**。FSDP 引擎 `forward_backward_batch` 对每个 microbatch 直接 `loss.backward()` 无缩放，于是：

```
一个 minibatch 拆 8 段 → 8 个局部均值等权累加（≈ 8× 全局均值，且分布不同）
一个 minibatch 不拆  → 1 个全局均值
```

`loss_agg_mode` 参数存在但 docstring 注明 "not used"。同类损失（vanilla/DPPO/GSPO…）都走 `agg_loss`（core_algos.py:1140），后者明确设计为「loss 对 fsdp/megatron 并行切分不变」。

## 2. 根因链路（源码确认）

1. 引擎 `forward_backward_batch`（workers/engine/fsdp/transformer_impl.py:638）：all_reduce 出全局 `batch_num_tokens`，`data` 挂 `batch_num_tokens`/`dp_size`；`trainer_base.py:1571,1613` 挂 `global_batch_size = ppo_mini_batch_size`（全局序列数）。
2. `ppo_loss`（workers/utils/losses.py:57）：把 `config.global_batch_info` = {dp_size, batch_num_tokens, global_batch_size, loss_scale_factor} 填好，entropy/kl 均 `agg_loss(..., **global_batch_info)`。
3. geo_mean 是唯一绕开 global_batch_info 的注册损失 → 切分不变性缺失 + DP 平均补偿缺失。

## 3. Patch（已应用本地 fork + 验证）

```python
# Replace: pg_loss = torch.mean(pg_losses)
pg_loss = agg_loss(
    loss_mat=pg_losses.unsqueeze(1),
    loss_mask=(response_mask.sum(dim=-1) > 0).float().unsqueeze(1),
    loss_agg_mode="seq-mean-token-sum",
    **getattr(config, "global_batch_info", {}),
)
```

- seq-mean-token-sum：`Σ pg_losses / global_batch_size × dp_size` — 正是 reporter 的公式 `local_sum / minibatch_count`，且 dp_size 补偿 FSDP 梯度平均（不平均时 dp=1）。
- `getattr(config, "global_batch_info", {})`：真实路径由 ppo_loss 填充；缺失时退化成本地聚合（dp=1 单 microbatch 行为与旧代码逐位一致）。
- 空 token 序列排除在分母外（与 agg_loss 其他 seq-mean 用户一致）。

## 4. 验证（CPU fp64，experiments/scripts/verify_gmpo_geo_mean_split.py）

复现 reporter 实验（8 序列 × resp_len 16，autograd 梯度对比，同一份数据）：

| 切分 vs [8] | PRE-patch rel L2 | POST-patch rel L2 |
|---|---|---|
| [1,1,1,1,1,1,1,1] | 7.00 | **0.000000** |
| [3,5] | 0.64 | **0.000000** |
| [2,2,4] | 1.83 | **0.000000** |

- 单 batch（无切分）损失值 pre==post（相对差 0.0）→ 无副作用。
- fp64 下所有切分完全不变（数值噪声量级为 0）。

## 5. 待办

- [ ] 查重复检（提交前）：窗口 0-2 天，随时可能被抢。
- [ ] 用户授权后：fork 开 PR（PR 描述按 verl CLAUDE.md）。改动含单元测试建议：新增 `test_geo_mean_split_invariance`（可放 tests/trainer/ppo/，无 GPU 依赖）。
- [ ] 可选：Megatron 路径确认 `loss_scale_factor`（num_microbatches×cp_size）对 geo_mean 的适用性——agg_loss 注释提到 Megatron 需额外缩放，geo_mean 现走同一路径即自动获得该语义。
