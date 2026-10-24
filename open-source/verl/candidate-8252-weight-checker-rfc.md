# verl：评估训练前权重同步自检方案

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 评估训练前权重同步自检方案。[Issue #8252](https://github.com/verl-project/verl/issues/8252) · [PR #7108](https://github.com/verl-project/verl/pull/7108)

**【类型】** RFC 评估

**【价值评估】** 6 / 10 — 可帮助明确校验范围与后端接口；实现成本和验证条件不支持直接开工。

**【进展】** 2026-10-24 纠正旧评估：维护者已于10-20回应并指向现成log-prob对照脚本；功能对照不等于破坏再恢复的内容自检。原指标/判定片段CPU控制通过，非模型/同步实测；不实现竞争方案，参与与发布仍待用户决定。

**【大白话讲明白】**

训练端更新了模型，不代表生成引擎一定收到正确权重。verl已经能比较两端在生成答案上的概率差异，但这不是逐个核对引擎权重。RFC想先破坏权重、再检查真实同步能否恢复它们；两类检查互补，启动检查也不能保证未来每次同步都安全。

**【What–Why–How】**

- **What：** 训练前沿真实权重更新路径执行端到端自检的 RFC。
- **Why：** 不同后端的切片和布局不能简单直接比较；启动阶段和运行阶段的请求状态也不同。
- **How：** 区分已有功能对照、拟议内容自检与运行期安全，核对维护者反馈后再决定参与；未实现或验证完整功能。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

## 现有验证能力与当前判断

2026-10-24读取[#8252全文及全部两评论](https://github.com/verl-project/verl/issues/8252)：维护者wuxibin89在10-20指向[现成脚本](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/scripts/rollout_logprob_diff.py)与[QAT测量](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/docs/low_precision/qat.md)，报告者10-21仍强调slime式自检。因此撤回旧“维护者无回应、没有任何验证机制”的概括，不把回应解读为同意实现整个RFC。

当前脚本沿CheckpointEngineManager真实更新路径同步，再生成并由Megatron重算概率；它不是无同步的数学演示，却也没有独立runtime基线、主动破坏和内容恢复比较。默认k3_kl均值小于0.01且kl/k3有限才通过，最大残差只报告；多轮退出只看最后一轮。局部原函数/原判定控制中，1/1000 token残差1仍k3约0.0007183而通过；不表示模型、传输或真实运行失败。详见[源码与验证范围](../../notebook/frameworks/verl-async-logprob-provenance.md#现成-logprob-对照与权重自检)。

本轮未重核#7108的完整当前diff或可合并性，下文其stale/冲突结论仅保留10-20历史。仍不建议我方实现大feature；若以后用户决定参与，应先明确现成脚本之外需要的保证、参考来源与负责人，不按旧“无验证”论据发布评论。实机脚本仅补[原M5条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，非接单或新训练任务。

## 2026-10-20 初评记录

> Issue: [verl-project/verl#8252](https://github.com/verl-project/verl/issues/8252)
> 创建 2026-10-15；[RFC]（功能提议，非 bug）；初评当时读取0评论、未查到reporter的verl PR记录，仅为10-20查询结果；10-24已确认两条后续评论，见上节。
> 类别：RFC 设计 / 评论型贡献（大功能，非小 bugfix）
> 状态：**assessed**（2026-10-20 已核事实，待用户决定是否参与/回复）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| 影响 | 4（初评）— 静默更新错误可能污染生成策略；不能据此断言整轮训练必然作废或完全没有功能诊断。独立已知参考与主动破坏能增强一次自检的可证伪性 |
| 成本 | 3 — **大功能**：需 per-engine weight-checker 适配（vLLM/SGLang 端点语义不同）+ `CheckpointEngineManager` 粘合 + per-replica fan-out + 单测；本机无 ray/GPU 无法验证，需 verl CI |
| 接受度 | 2（旧评分，不作当前预测）— 初评未读到回应；10-24已纠正。#7108当时stale/冲突不等于当前状态，维护者指出已有功能对照也不等于接受全套RFC |
| 契合度 | 4 — 权重同步/checkpoint 契约是我们的深耕区（#8237 offload、#61229 lease 契约）；RFC 核心观察"布局校验≠内容校验"与 lease 学习笔记结论一致 |
| 分 | ~13（RFC 评论型；若 maintainer 有意向再评估实现） |

## 1. 事实核验（2026-10-20，本地 verl main + GitHub API）

当时核对了三个较窄的源码位置，不能扩张为没有任何验证机制：

1. **`verl/checkpoint_engine/README.md` 无 post-sync 验证**：整篇（backend 表 + 用途）无 verif/check/checksum/integrity 内容，只有传输拓扑/性能列。✓
2. **`tests/checkpoint_engine/` 无 weight-checker 测试**：main 上 12 个测试文件（block_placement/correctness_on_gpu/correctness_on_npu/global_steps_on_cpu/sglang_*/sharded_delta/special_server_adapter/block 等）无 `test_weight_checker.py`；该测试只在 #7108 里。✓
3. **`check_weight_update_equal`/`weight_checker` 在 main 不存在**（grep 全空）；delta 后端的 checksum/verify 是**线缆完整性**（`delta_sync/encode.py::checksum(positions,values)` + `verify_every`），答"消息完整到达"，不答"引擎运行时张量等于我们想写的"——与 RFC 区分一致。✓

**2026-10-24更正**：上述路径/标识符检查只涉及专门的内容自检，不足以支持RFC“今天无验证机制”的宽泛表述；现成log-prob对照应计入已有能力。

## 2. 关键张力：#7108 是移动靶

- #7108（P2P checkpoint engine，AkiRusProd，07-21 起 open）：**不可合并（冲突）**、09-21 后 stale、被 maintainer wuxibin89 质疑与 mooncake_checkpoint_engine 差异（07-28 评论，作者回复后无下文）。diff ~3500 行，含 p2p/weight_checker.py（128 行）与 test_weight_checker.py。
- RFC 明说"我们更愿 extend #7108 而非竞争"。但 **extend 一个可能不会合并的 PR 是移动靶**：若 #7108 最终关闭，RFC 的"复用 `check_weight_update_equal` 名 + 生命周期"前提就落空。
- 我方立场空间：这正是 RFC 最值得被指出的技术点——**后端无关层不应以"绑定 p2p 的实现"为唯一先例**；应先落地一个独立的、挂在 `CheckpointEngineManager` 上的最小验证钩子（默认 off），与 #7108 的 p2p 实现解耦，避免实现命运被 #7108 的合并状态绑架。

## 3. 架构观察（对 review 有价值）

RFC 提议 `verify_update_path()` 复用真实 `update_weights`（base.py:510-562）。核对了实现：`update_weights` 不只是 push——它 **abort 全部在飞请求 → release kv_cache → build_process_group → 同步 → finalize → 恢复 kv_cache → resume 在飞请求**。训练前自检要复用的正是这条**完整路径**（RFC 说得对，不能用并行 mock path，否则验的不是在跑的东西）。但注意两点：

1. 若训练前确实没有在飞请求，相关abort/resume分支不能验收训练期的在飞生命周期；不能因此断言kv_cache不存在、所有release都是空操作或自检几乎免费。**启动检查通过 ≠ 后续训练期同步保证**；需要区分实际初始化状态、独立参考来源与覆盖范围。
2. **engine-side 校验是必须的**（trainer 在 HF 坐标、engine 在 fused/TP/EP 切片坐标，trainer 算的 digest 不可比）——RFC 这一条与 vLLM #7227 scope note 及我们 lease 学习笔记"布局校验≠内容校验、内容身份须绑定到描述符"一致，方向正确。

## 4. 结论与建议

- **不建议我方去实现这个大 feature**（成本 3、接受度未验证、无 ray/GPU 验证手段，违背"先验证小步"）。这符合 tracker 对 experimental/ 大功能的一贯扣分。
- **旧回复方向需先更新**：不能再确认“无验证机制”或沿用#7108旧状态作为当前论据；应承认已有功能对照，区分内容恢复自检与在飞生命周期，再在用户授权后决定是否需要技术回复。
- **是否参与、回复形态（评论 or 具体方案）待用户授权**；在此之前不动作。

## 5. 待办

- [ ] 用户决定是否回复 #8252（评论稿可据此文档拟）
- [ ] 若决定参与：拟精简 RFC 回复稿，按 SOP 走查重 + 授权后发布
- [ ] 沉淀：weight-sync 校验缺失的观察与 lease 契约学习互相印证，可补进 notebook/frameworks 相关笔记
