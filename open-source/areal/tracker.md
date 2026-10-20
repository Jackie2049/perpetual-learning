# AReaL 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/areal-project/AReaL （蚂蚁系，2026-10-20 核验；default branch main，5.8k stars，135 open issues）
>
> 源码理解：[AReaL 架构与统一损失系统源码阅读](../../notebook/frameworks/areal-architecture-source-reading.md)（2026-10-20，基线 `3b93679`）。§10 补读 FSDP2 后端 batch 切分 + prompt_mean 切分敏感点。

## 当前结论（2026-10-20 首次扫描）

**近期可行动 bug 均已被 reporter 本人同日/近时 PR 覆盖**，与 verl 的「reporter 深挖 ≈ reporter 已开 PR」规律一致。当前没有未认领、可行动的贡献空间。唯一有价值的追踪点是 **#1730（M2PO microbatch 不变性）** 与 verl `#7913`（geo_mean microbatch 不变性）同类、同 reporter，我方 verl 经验可直接复用作 review/评论依据。

| # | 候选 | 类别 | 上游状态 | 我方判断 |
|---|---|---|---|---|
| 1786 | Megatron weight-residency 直调 `torch.cuda.*` 致 NPU 崩溃/泄漏 | P0 bug | issue #1786 → reporter li-lizhe PR [#1787](https://github.com/areal-project/AReaL/pull/1787)（open，`mergeable_state=blocked`，Ascend 910B4 已验证，含回归测试） | **已认领无空间**；方向正确（改用 `current_platform` 抽象），无需插足 |
| 1730 | M2PO token 选择/损失缩放随 microbatch 切分变化 | P0 bug | issue #1730 → reporter gss10282025 PR [#1732](https://github.com/areal-project/AReaL/pull/1732)（open，`mergeable_state=dirty`，做成 opt-in 且仅支持 DP1 缓存近似 logprob） | **已认领**但未闭合；与 verl #7913 同类、同 reporter；**可复用作评论/review 依据**（见下） |
| 1742 | examples/tau2 硬编码类名（外部 fork 消息校验器） | P4 示例 | stale 标签，1 评论 | 低价值：外部 fork、example 代码，不作候选 |
| 1341 | MegatronEngine 存 optimizer checkpoint 失败 | P0 bug | assignee daihaowz（已认领） | 无空间 |
| 1343 | trajectory dump/replay 功能 | Feature | assignee daihaowz，已有多个实现 PR（#1344/#1407/#1523） | 无空间 |
| 1110 | Async Tensor Parallelism（Archon Engine） | Feature（help-wanted） | 1 评论 | 大型 feature，非入口，暂不作候选 |

## 可复用经验与跨框架观察

- **同 reporter 跨框架同 bug 类**：`gss10282025` 同时开了 verl [#7913](https://github.com/verl-project/verl/issues/7913)（geo_mean 随 minibatch 切分变化，我方已深挖并验证，reporter 自提 PR #7916 BREAKING）和 AReaL #1730（M2PO microbatch 不变性）。此人专注「batch 切分不应改变 update」这一类正确性问题。我方 verl geo_mean 分析（[candidate-7913-gmpo-geo-mean](../verl/candidate-7913-gmpo-geo-mean.md) 及验证脚本）的经验、方法（原 batch 分母归一、batch-wide selection）可直接迁移到 AReaL #1730 的技术回复。
- **（2026-10-20 §10 新增，§10.3 已深挖至忠实归一化链）prompt_mean 归约的切分敏感点**：`_PromptMeanBatch.for_steps` 设 `step_scale = len(microbatches)/active_groups = M/G`（M=DP 同步的有效 minibatch slot 数，DP=1 时=ppo_n_minibatches；G=完整 batch 有效组数），使 `prompt_mean`（默认 ppo_n_minibatches=4 时）的累计梯度随 M 线性放大。**忠实归一化链核验**（loss_reduction.py + train_engine.py + fsdp_engine.py）：`total_weight ≡ 1`（单 minibatch 内 DP all_reduce 后，`engine_weight=rows/global_rows` 是 DP 归一权重非跨 mb share），`loss_multiplier=dp_size` 统一乘所有归约——**两者均不抵消 M**，最终累计梯度 = `dp_size·(M/G)·Σ_all loss_t·w_t`，numpy 复刻实测 M=1/2/3/4/6 → 1×/2×/3×/4×/6×（与组边界切割无关）。`token_mean`/`seq_mean`/`constant` 不带 step 因子、不受影响。上游无此公开讨论；测试只断言 M 个 step 梯度均值恢复 full-batch，未覆盖累加语义。**定位「切分敏感点（高置信源码级确证）**：绝对梯度随 M 线性放大已确证，但是否构成实际训练 defect（自适应优化器会吸收部分绝对尺度）需 AReaL 真实环境 `ppo_n_minibatches` 2 vs 4 对照实测。这是继 M2PO 掩码（#1730）之后 AReaL 侧第二个「batch 切分影响 update」实例，位于归约侧，与 verl #7913 同源。详见 [源码笔记 §10.3](../../notebook/frameworks/areal-architecture-source-reading.md)，验证脚本 [verify_areal_prompt_mean_split.py](../../experiments/scripts/verify_areal_prompt_mean_split.py)。
- **AReaL 认领动态与 verl 相同**：#1786→#1787（分钟级）、#1730→#1732（小时级）均为 reporter 自修；老 issue 多被 assignee 认领。**深挖前先按 reporter 名查其 PR 列表**，与 verl tracker 第 5 轮教训一致。
- **#1730 的残余评论价值**：reporter 自述 PR 仅 DP1 缓存路径、`mergeable_state=dirty`；且 issue 明确指出「select once over full batch」才是理想。若我方评论，价值在于：把 verl #7913 的「microbatch 归一化 vs batch-wide selection」权衡对照 #1730 的 opt-in 方案，指出 opt-in 未解决 selection scope 问题。低风险但需授权；目前价值中等，暂不主动发起。

## 经验教训（按需追加）

- （2026-10-20 建立轨道）
