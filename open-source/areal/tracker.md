# AReaL 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/areal-project/AReaL （蚂蚁系，2026-10-20 核验；default branch main，5.8k stars，135 open issues）
>
> 源码理解：[AReaL 架构与统一损失系统源码阅读](../../notebook/frameworks/areal-architecture-source-reading.md)（2026-10-20，基线 `3b93679`）。§10 补读 FSDP2 后端 batch 切分 + prompt_mean 切分敏感点；§10.5 补 Megatron 后端对照（跨后端同源）；§10.6 补读 Megatron 归一化三分支 + per-token 路径不抵消切分敏感。

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
- **prompt_mean 优先研究项（2026-10-20 首轮复核更正，替代此前 §10.3–10.6 判断）**：尚未证明 M/G 存在缺陷。旧脚本的 token_mean 基线没有逐 optimizer step 归一化，dummy 用例追加了非法全局空步，Megatron 默认路径遗漏 MCore 的 /P；此前“独有 M 放大 / dummy 膨胀 / 额外 P 倍”论证撤回。新版标准库 CPU 参考验证通过 43 项数值比较与调度检查：等长均匀组下 prompt_mean/token_mean 同一步相同、二者冻结梯度和都随外层 M 增长；固定 M 下内层 P 不改变标量默认路径梯度。上游 PR #1546 的冻结参数跨步平均契约与现有 mean() 测试一致。GPU 仍优先，但先固定外层 M 改内层 P、再核对 DP/dummy 全局参考；M=2/4 权重不同本身不构成 bug 证据。见 [源码笔记 §10.7](../../notebook/frameworks/areal-architecture-source-reading.md) 与 [CPU 验证](../../experiments/scripts/verify_areal_prompt_mean_split.py)。非 AReaL/GPU 集成结果，CP/per-token/MTP/MoE 最终链仍待验证。
- **AReaL 认领动态与 verl 相同**：#1786→#1787（分钟级）、#1730→#1732（小时级）均为 reporter 自修；老 issue 多被 assignee 认领。**深挖前先按 reporter 名查其 PR 列表**，与 verl tracker 第 5 轮教训一致。
- **#1730 的残余评论价值**：reporter 自述 PR 仅 DP1 缓存路径、`mergeable_state=dirty`；且 issue 明确指出「select once over full batch」才是理想。若我方评论，价值在于：把 verl #7913 的「microbatch 归一化 vs batch-wide selection」权衡对照 #1730 的 opt-in 方案，指出 opt-in 未解决 selection scope 问题。低风险但需授权；目前价值中等，暂不主动发起。
- **（2026-10-20 §10.10 闭环，用户恢复本会话继续后完成）prompt_mean 实测：不构成 defect，候选修复撤销**。服务器 4090-1（VPN 恢复后）配好 AReaL 环境（clone `3b93679` + uv 环境 torch 2.9.1/sglang/megatron/ray），**框架完整跑通**（gsm8k PPO smoke，Qwen2.5-0.5B + FSDP2 + sglang，128 步无错）。**受控实测按 §10.7 判据**（真实 PPOActor+FSDPEngine 单卡 + 固定 batch，[areal_prompt_mean_gpu_probe.py](../../experiments/scripts/areal_prompt_mean_gpu_probe.py)）：prompt_mean 跨 M 比值 1.301 ≈ token_mean 1.273（M=2 vs 4）；P-spread 1.216 ≈ token_mean 1.192（固定 M=2，P=1 vs 2）——**M/内层切分依赖是两种归约共有的训练行为，非 prompt_mean 独有缺陷**；`step_scale=M/G` 与 PR #1546 契约一致（冻结参数跨步平均，测试 mean() 正确覆盖）。**候选修复 `step_scale=1/G` 撤销**（patch 草稿保持 superseded 标注），不提 issue/PR，作为 review 依据存档。Megatron 后端 GPU 实测未做（megatron-core 0.19 硬依赖 Transformer Engine，需长编译；双后端同源已由 §10.5 源码确立）。分工与交接：该候选曾由用户明确交 macbook-M5 单一负责（见 [handoff-m5.md](handoff-m5.md)），2026-10-20 用户恢复 VPN 并明确本会话继续后完成实测闭环；状态已同步，不再由多会话并行维护方案。详见 [源码笔记 §10.10](../../notebook/frameworks/areal-architecture-source-reading.md)。

## 经验教训（按需追加）

- （2026-10-20 建立轨道）
