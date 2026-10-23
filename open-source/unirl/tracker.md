# UniRL 贡献追踪

<!-- contribution-summary:start -->
**【仓库】** [UniRL](https://github.com/Tencent-Hunyuan/UniRL)（腾讯统一多模态 RL）

**【贡献点】** UniRL 贡献追踪。[候选目录](readme.md) · [候选 #523](candidate-523-anchor-lifecycle.md)

**【类型】** 候选追踪

**【价值评估】** 7 / 10 — 甄别"未认领、无 PR、可落地"的开放空间，避免重复投入已有人认领的 RFC。

**【进展】** #523 已有我方 fork PR 供审阅，GPU 验收沿用[M5交接](../../experiments/handoff-m5.md)，#567 仍待匹配 H20 条件。2026-10-23 补充尽调：[#529](https://github.com/Tencent-Hunyuan/UniRL/issues/529#issuecomment-6340056553)虽无assignee但已有公开认领意向，不并行实现；[#458](https://github.com/Tencent-Hunyuan/UniRL/issues/458)是控制字段所有权设计议题，不当成通用deep-merge小修复。未新增可直接发布的补丁，详见[边界核对](#补充尽调与派发边界)。同日用户审阅并认可 #523 方案设计与代码实现，**测试验证 + e2e 冒烟已升级为正式任务指派 M5**（handoff 专设小节：L0 真实模块冒烟 + L1 e2e 回归/replay 设计）。**2026-10-23 M5 已闭环 L0（四项断言全过 + pristine 对照）+ L1 判定待资源/待权重**（见 [handoff 专设小节](../../experiments/handoff-m5.md)）。

**【大白话讲明白】**

这张表记录 UniRL 上哪些问题还没人做、能做什么、以及哪些其实已被人认领（避免白费功夫）。当前最值得做的是修一个训练栈一致性 bug（#523），其他多数热门 RFC 都已被社区抢先认领。

**【What–Why–How】**

- **What：** UniRL 候选从扫描、验证到认领/放弃的状态。
- **Why：** 社区响应快、RFC 多被认领，本地验证成功不代表仍有贡献空间。
- **How：** 按证据更新状态，将已认领、需 GPU、已上路线和待用户决定分别记录。
<!-- contribution-summary:end -->

> 扫描时间 2026-10-22（GitHub 实时 API）。main 固定 `6fd82cba`。状态以当日核验为准，继续前重新核对关联 issue/PR。

## 本次扫描（2026-10-22）核心结论

UniRL 仓库当前 **73+ open issues**，其中大量是带 `help wanted` 标签的 RFC/贡献机会，且外部 PR 合入率高（最近 100 关闭 PR 合并率 94%）。多数热门 RFC 已被社区认领（维护者会 assign 给 contributors）。**真正"未认领、无对应 PR、可落地"的开放空间**是本次要甄别的。

| 候选 | 类型 | 状态（2026-10-22） | 可做性 | 备注 |
|---|---|---|---|---|
| [#523](https://github.com/Tencent-Hunyuan/UniRL/issues/523) unified stack 跳过 anchor freeze/prepare_part | 跨栈契约 bug | **OPEN · 无 assignee · 无 PR** | **用户已认可方案与实现**（2026-10-23；fork 内 [Jackie2049/UniRL#1](https://github.com/Jackie2049/UniRL/pull/1)：fix-523-anchor-lifecycle @ 04101ad5，base fork main 6fd82cba，unified.py +8/-2；CPU 单测 3 项 + AST 断言通过）；**M5 已闭环 L0 + 判定 L1 待资源/待权重**（[handoff 专设小节](../../experiments/handoff-m5.md)；L0 四项断言全过，L1 需 ≥8 卡 + 获准权重）；上游 PR 待用户授权后再开（铁律）。见 [candidate-523](candidate-523-anchor-lifecycle.md) |
| [#567](https://github.com/Tencent-Hunyuan/UniRL/issues/567) cosmos3 H20 cuBLAS workaround 被错版本 gate | 配置/验证型 | **OPEN · 无 assignee** | **blocked-on-H20**（2026-10-23） | reporter 明确两个分支（删死代码 vs 放宽 gate）都要在 H20+pinned torch 2.13/cu130 实机对照（loss + cublasGemmEx 探针）才能定；我们与 M5（4090）均无 H20，写 PR 即投机，整项挂起不推进 |
| [#581](https://github.com/Tencent-Hunyuan/UniRL/issues/581) FSDP2 0-token micro collective mismatch | bug | **OPEN · 无 assignee** | 可做但已上 #580 路线 | reporter 给了完整 CPU repro；issue 说明 #580 phase 3 会修，宜与 #580 owner 沟通或作为其一部分 |
| [#419](https://github.com/Tencent-Hunyuan/UniRL/issues/419) monitor-only rewards | enhancement/help wanted | **已 assign GoDiao** | 已认领 | 非我方空间 |
| [#423](https://github.com/Tencent-Hunyuan/UniRL/issues/423) 复合奖励能力预检 | enhancement/help wanted | **已 assign GBX-Max1220，有 PR #621** | 已认领 | 非我方空间 |
| [#453](https://github.com/Tencent-Hunyuan/UniRL/issues/453) 主仓库文档社区刷新 | documentation/help wanted | **已 assign 6 人，WIP** | 已认领 | 非我方空间 |
| [#520](https://github.com/Tencent-Hunyuan/UniRL/issues/520) batched chat-template tokenization bug | bug | **已 assign ruiling-smartbear，关联 PR #544** | 已认领 | 非我方空间 |

## 补充尽调与派发边界

2026-10-23只读核对#458/#529正文、全部评论及开放PR清单；关键词affinity/sibling/control/overlay未发现标题明确覆盖的开放PR，**不等于确认没有重叠工作**。编号全文检索还会命中不相关PR，须回到具体需求、评论与源码。固定源码为`6fd82cba13f52f394747832344039408699ea378`，未修改上游。

| 线索 | 可核实的现状 | 本项目处理 |
|---|---|---|
| [#529 sibling affinity](https://github.com/Tencent-Hunyuan/UniRL/issues/529) | 无assignee；MciG-ggg已[公开表示准备实现](https://github.com/Tencent-Hunyuan/UniRL/issues/529#issuecomment-6340056553)初版RolloutManager group_packing，尚非维护者正式指派 | 暂不竞争；接续前先核贡献者进展和实际路径，不据空assignee认领 |
| [#458 Part.control](https://github.com/Tencent-Hunyuan/UniRL/issues/458) | 无assignee、0评论；问题正文明确是设计债而非#263用户回归 | 设计方向待确认，不创建通用deep-merge补丁或声称可发布；须先确定每hop所有权、显式继承/替换/清除及兼容范围 |

#529的派发粒度不能一概而论：[AsyncARTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/async_ar.py)沿ARTrainer构造fork后的完整请求，再由async循环按`Sample.split()`拆成root树提交；[Sample.split](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py)保留该root的所有children。相反，[AgenticTrainer._collect_groups](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/agentic.py)将每个prompt重复K次成为独立任务。pool按任务计一个slot，并不在内部拆K个row。因此“普通异步AR的K个siblings必被pool散到K个worker”不能直接由most-free策略推出；同worker也不保证KV已就绪、命中100%或吞吐提高。同步DP_SCATTER、engine内部执行及session续跑须另核，见[原单位说明](../../notebook/frameworks/unirl-survey.md#队列深度不是一种单位)。

#458同样不能只凭浅合并就认定bug：[prepare_input_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/base.py)顶层覆盖，而PE显式传空ar/chat袋以擦除父输入控制；无条件deep merge会保留本来要清掉的字段。单个字典算法无法替代跨trainer/engine的所有权设计。本轮只做源码和社区状态尽调，无执行、性能/缓存验证、认领或对外操作，不为这两个线索排M5实验。

## 已认领/已覆盖，不重复推进

- #368 RFC on-policy distillation——维护者列出 Open 项（D3/D4/D5、A3…）等认领，D2 已 assign，contribution 门槛高（P0/P1 多为大 feature）。
- #520/#423/#419/#453 见上表，均有 assignee 或对应 PR。
- 大量 open PR 由社区作者（ruiling-smartbear、MinhaoLi0318、nussejzz、leviking98z-rgb 等）挂多个 RFC 功能实现，合入排队积压较重；避免与在途 feature PR 撞车。

## 决策待定

- #523 是本次最可落地的候选。建议方向：先在 issue 留言表明意向并做 CPU 结构修复 + 验证，再考虑是否提 PR（对外提交按用户授权）。
- #567/#581 需 GPU 或需协调 #580，标为"后续/需沟通"，不作为优先。
