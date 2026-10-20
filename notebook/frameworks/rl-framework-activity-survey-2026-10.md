# 国内主流 RL 开源框架活跃度统计（2026-10-20 快照）

> 应用户调研需求统计 slime / verl / ROLL 及其他国内主流 RL 训练框架的社区活跃程度。数据为 GitHub API 实测（`gh` 认证请求，2026-10-20），指标口径见文末。技术架构全景另见 [rlhf-training-infra-2026.md](rlhf-training-infra-2026.md)，本文只看活跃度。

## 总表

| 指标 | verl | slime | AReaL | ROLL | OpenRLHF | RAGEN |
|---|---|---|---|---|---|---|
| 主导方 | 字节 Seed（已迁独立 org verl-project） | 智谱 THUDM | 蚂蚁（已迁独立 org areal-project） | 阿里巴巴 | 开源社区（org 已从 OpenLLMAI 迁至 OpenRLHF） | 清华 mll-lab |
| Stars | 23,933 | 8,690 | 5,844 | 3,431 | 10,100 | 2,824 |
| Forks | 4,776 | 1,320 | 632 | 320 | 1,047 | 229 |
| 仓库创建 | 2024-10 | 2025-06 | 2025-02 | 2025-05 | 2023-07 | 2025-01 |
| 默认分支最近 commit | 10-20 | 10-19 | 10-20 | 10-19 | **09-17** | **08-23** |
| Commits 近 30 天 | **104** | 35 | 47 | 4 | 0 | 0 |
| Commits 近 90 天 | **336** | 98 | 140 | 9 | 37 | 1 |
| 新 issue 近 30 天 | 39 | 8 | 7 | 0 | 4 | 0 |
| 新 PR 近 30 天 | **286** | 80 | 58 | 8 | 15 | 2 |
| 合并 PR 近 30 天 | 100 | 32 | **53** | 4 | 0 | 0 |
| 贡献者总数 | **444** | 175 | 109 | 60 | 94 | 21 |
| 最新 release | v0.9.1（09-20） | v0.4.0（10-09） | v2.1.0（08-25） | v0.4.0（09-29） | v0.11.2（09-14） | 无 |
| 未关闭 issue+PR | 1,377 | 542 | 135 | 135 | 402 | 31 |

加粗为本列最优/最值。commit 数用列表 API 翻页统计（committer date 过滤），与搜索 API 交叉验证一致（verl 搜索 98 vs 列表 104 属索引延迟）。

## 分框架画像

### verl — 断层第一，工业级社区

所有维度全面领先：444 贡献者约为其余五家之和，30 天 286 个新 PR、100 个合并，日均 3+ commit。已有 EuroSys 论文和完整生态（AgentLoop、多硬件后端）。代价是 issue 积压 1,377 条，且本项目 10-19 的扫描经验表明**认领窗口只有 0-2 天**、reporter 自修为主——活跃度最高的社区竞争也最激烈。

### slime — 稳健第二梯队头部

智谱出品，30 天 35 commit、80 PR、32 合并，v0.4.0（10-09）刚发布，节奏稳定。8,690 stars / 175 贡献者与其 GLM 模型训练的实际生产使用相符。对贡献者而言窗口期比 verl 宽松。

### AReaL — 第二梯队，合并效率突出

蚂蚁背景但已迁独立 org（areal-project）。30 天 47 commit、58 PR 中合并 53——**合并/新增比 91%** 为六家最高，说明维护者响应快、外部 PR 真的会被合入。对想落地贡献的人这是比 star 数更重要的信号。

### ROLL — 低频但未死，release 驱动

阿里出品。表面指标弱（30 天仅 4 commit、0 新 issue），但 10 月仍有连续修复 commit（ROCm/vLLM 后端）且 09-29 发了 v0.4.0。属于小核心团队、低频稳态维护，社区参与度低（320 forks）。issues API 里近 30 天的"新条目"实际全是 PR——issue 侧已无外部流入。

### OpenRLHF — 明显减速，老牌退坡

最早的国产 RLHF 框架（2023-07），10,100 stars 仍为第二高，但**主分支一个月无 commit（09-17 至今）、近 30 天 15 个新 PR 零合并**、v0.11.2（09-14）后无动作。90 天 37 commit 约为 verl 的 1/9。热度已被 verl/slime/AReaL 接走。

### RAGEN — 基本休眠

清华 mll-lab 的 multi-turn agent RL 研究框架（已从 RAGEN-AI 迁移）。近 90 天 1 commit、无 release、最后 push 08-23。定位偏研究代码而非持续维护的训练框架，不列入活跃社区。

## 结论

- **活跃梯队**：verl（断层第一）> slime ≈ AReaL（稳健第二梯队）> ROLL（低频存续）> OpenRLHF（退坡）> RAGEN（休眠）。用户点名的三家恰好覆盖第一、第二、第四梯队。
- **对本项目贡献策略的含义**：verl 机会最多但抢修最快；若求 PR 落地率，AReaL 的合并效率（91%）和 slime 的宽松窗口值得作为第二贡献轨道评估；OpenRLHF 不建议新投入。
- **组织迁移现象**：verl、AReaL、OpenRLHF、RAGEN 四家均已脱离原公司 org 或改名（volcengine→verl-project、inclusionAI→areal-project、OpenLLMAI→OpenRLHF、RAGEN-AI→mll-lab-nu），查询上游状态时必须用新 org 名，旧路径仅 API 重定向可用、搜索 API 直接 422。

## 口径与局限

- 统计时间 2026-10-20（Asia/Singapore），GitHub REST/search API 经 `gh` 认证实测；"近 30 天" = 09-20 之后，"近 90 天" = 07-22 之后。
- 新 issue/新 PR/合并 PR 用 search API total_count；commit 用列表 API 按 committer date 翻页。
- 贡献者数取 contributors API 最后一页（非匿名）。
- 未覆盖微信群/论坛/DockerHub 等站外活跃度，GitHub 指标也不等于生产使用量（ROLL 类企业内部使用不体现在 issue 流量上）。
- 快照型数据，复用时按 AGENTS 要求核对当下状态。
