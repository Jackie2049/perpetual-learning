# UniRL 贡献追踪

<!-- contribution-summary:start -->
**【仓库】** [UniRL](https://github.com/Tencent-Hunyuan/UniRL)（腾讯统一多模态 RL）

**【贡献点】** UniRL 贡献追踪。[候选目录](readme.md) · [候选 #523](candidate-523-anchor-lifecycle.md)

**【类型】** 候选追踪

**【价值评估】** 7 / 10 — 甄别"未认领、无 PR、可落地"的开放空间，避免重复投入已有人认领的 RFC。

**【进展】** 2026-10-22 首次扫描 73+ open issues + open PRs：锁定未认领候选 #523（unified stack 跳过 anchor freeze/prepare_part，纯 CPU 可验证）；#567 需 H20 GPU、#581 已上 #580 路线为次要；热门 RFC #419/#423/#453/#520/#368 已被社区认领，不重复推进。

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
| [#523](https://github.com/Tencent-Hunyuan/UniRL/issues/523) unified stack 跳过 anchor freeze/prepare_part | 跨栈契约 bug | **OPEN · 无 assignee · 无 PR** | **修复已推 fork**：[fix-523-anchor-lifecycle @ 04101ad5](https://github.com/Jackie2049/UniRL/tree/fix-523-anchor-lifecycle)（base 6fd82cba，unified.py +8/-2）；CPU 单测 3 项 + AST 断言通过；**待用户审阅授权后再向上游开 PR**（铁律）；replay 完整验收需 GPU。见 [candidate-523](candidate-523-anchor-lifecycle.md) |
| [#567](https://github.com/Tencent-Hunyuan/UniRL/issues/567) cosmos3 H20 cuBLAS workaround 被错版本 gate | 配置/验证型 | **OPEN · 无 assignee** | 需 H20 GPU（调查型） | 二选一：删死代码 或 放宽版本 gate，需实机验证 |
| [#581](https://github.com/Tencent-Hunyuan/UniRL/issues/581) FSDP2 0-token micro collective mismatch | bug | **OPEN · 无 assignee** | 可做但已上 #580 路线 | reporter 给了完整 CPU repro；issue 说明 #580 phase 3 会修，宜与 #580 owner 沟通或作为其一部分 |
| [#419](https://github.com/Tencent-Hunyuan/UniRL/issues/419) monitor-only rewards | enhancement/help wanted | **已 assign GoDiao** | 已认领 | 非我方空间 |
| [#423](https://github.com/Tencent-Hunyuan/UniRL/issues/423) 复合奖励能力预检 | enhancement/help wanted | **已 assign GBX-Max1220，有 PR #621** | 已认领 | 非我方空间 |
| [#453](https://github.com/Tencent-Hunyuan/UniRL/issues/453) 主仓库文档社区刷新 | documentation/help wanted | **已 assign 6 人，WIP** | 已认领 | 非我方空间 |
| [#520](https://github.com/Tencent-Hunyuan/UniRL/issues/520) batched chat-template tokenization bug | bug | **已 assign ruiling-smartbear，关联 PR #544** | 已认领 | 非我方空间 |

## 已认领/已覆盖，不重复推进

- #368 RFC on-policy distillation——维护者列出 Open 项（D3/D4/D5、A3…）等认领，D2 已 assign，contribution 门槛高（P0/P1 多为大 feature）。
- #520/#423/#419/#453 见上表，均有 assignee 或对应 PR。
- 大量 open PR 由社区作者（ruiling-smartbear、MinhaoLi0318、nussejzz、leviking98z-rgb 等）挂多个 RFC 功能实现，合入排队积压较重；避免与在途 feature PR 撞车。

## 决策待定

- #523 是本次最可落地的候选。建议方向：先在 issue 留言表明意向并做 CPU 结构修复 + 验证，再考虑是否提 PR（对外提交按用户授权）。
- #567/#581 需 GPU 或需协调 #580，标为"后续/需沟通"，不作为优先。
