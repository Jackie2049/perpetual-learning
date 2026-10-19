# perpetual-learning — 当前工作上下文

更新时间：2026-10-19

## 当前阶段

自主运行中（2026-10-19 起 18 轮 /loop 30m）。已完成：工具修复（inference_estimator.py）、verl 候选三轮核验与全新扫描（10 候选）、4 项源码级修复分析、GMPO/GSPO/SAPO 论文精读与数值证明。详见 [当日日志](diary/2026-10-19.md)。

## 运行接续

- 网页 demo 已上线：[探索工作台](https://jackie2049.github.io/perpetual-learning/)。只读展示仓库快照，更新内容后按 [网页维护说明](web/readme.md) 发布；当前采用 gh-pages 分支发布。

- 文档呈现要求：评分维度直接写“影响、成本、接受度、契合度”，使用简短直观名称。成本方向已在 [SOP](open-source/SOP.md) 澄清，历史分值复用前核对。

- 2026-10-19 收尾机制更新：每项实际工作必须有日志，完成、失败或受阻均记录。当前克隆已启用 Git 提交检查；下一轮先重读 [AGENTS.md](AGENTS.md) 与 [日志工具说明](tools/diary/readme.md)，使用自动取时入口。其他克隆需安装检查；现有 Claude 会话是否重读规则未验证。

- 状态：持续运行中（/loop 30m，会话级 job 486bb920，7 天后过期；退出会话后不自动恢复）。已按新收尾规则对齐（09:35，日志收尾工具 + 提交检查生效）。
- 待执行的用户任务：无；当前没有已登记、等待自主运行处理的用户需求。
- 自主探索任务：见「本轮进展」与 [verl tracker](open-source/verl/tracker.md)（当前活跃 3 候选：#8237 bugfix、#8250 诊断、clip 语义文档澄清）；vLLM 方向进行中（#61160 修复定位完成，#61229 待核查——核心 API 限流，约 10:40 后重置）。
- **等待用户决定的事项**（均需授权/资源）：①发布 #8250 熵爆炸诊断评论；②发布 vLLM #39096 增量评论；④#8237 GPU 验证（需服务器 + VPN 路由修复）；⑤clip 语义文档澄清的发布形态。~~③ vLLM #61160 修复定位评论~~ **已发布**（2026-10-19 10:49，[comment](https://github.com/vllm-project/vllm/issues/61160#issuecomment-6258592764)），待回应跟进。
- 下一步：见下方「待讨论与下一步」。
- 运行范围：当前 Agent 会话；未配置进程退出后的自动唤起。

## 当前用户需求与决定

- 项目名称、首页标题和仓库标识统一使用 `perpetual-learning`，删除首页旧名称标识行。项目定位继续保留 [README.md](README.md#项目说明) 中的用户原文。
- README.md 是整个项目的总览、对外呈现与用户导航，按主题、框架、成果提供快捷入口；文件组织使用可点击的目录表，作为统一入口置于项目说明与运作机制之后，不再单设重复的“项目导航”章节。[AGENTS.md](AGENTS.md) 为 Agent 内核；本文件记录当前工作上下文。
- README 不放内部整理计划、迁移过程或待决定事项；当前阶段与待办由本文件维护，完成过程写入 diary。
- README 章节顺序：项目说明 → 运作机制 → 文件组织 → 知识导航 → 上机实战 → 主流框架 & 开源贡献 → 论文学习 & 专题研究 → 工具沉淀 & 技能打磨 → 规划 & 进展；GitHub 链接保留在末尾。
- notebook/ 保留 fundamentals/ 基础知识、experiments/ 实验记录、roadmap.md 规划、knowledge-map.md 知识整理与总览；原源码学习目录改名 frameworks/，总览见 [notebook/readme.md](notebook/readme.md)。
- tools/ 注重工具价值与复用；一次性、纯实验性和低价值重复实现归 tools/legacy/，新增 [tools/readme.md](tools/readme.md)。
- open-source/<framework>/ 按目标开源项目保存贡献材料，与 notebook/frameworks/ 的源码理解分开并相互链接；diary/ 延续原有功能，入口为 [diary/readme.md](diary/readme.md)。

## 本轮进展（2026-10-19 持续运行 18 轮）

**工具**：inference_estimator.py 修复验证（带宽硬编码 → GPU 规格、吞吐三口径分离、--batch、h20/rtx4090 规格），纳入推荐入口。

**verl 贡献轨道**：
- 旧候选核验（8 候选全失效：4 个 VERIFIED-WORKS 被上游 PR 覆盖、备选失效）；vLLM 6 月草稿核验（3 作废，#39096 有残余价值）。
- 全新扫描（75 issue 窗口）+ 复查重纠错：**10 候选 9 个被 reporter 同日/次日 PR 覆盖**（含第 5 轮查重限流误报），教训 8 条入库。
- 4 项修复分析：#7913 geo_mean microbatch 不变性（**verified-works，CPU 验证 rel L2=0**，上游 #7916 方案不同未合）、#8089/#8028/#7899（均被 reporter 自修 PR 覆盖，fork 补丁已还原）、#8237 offload fraction（**唯一真实无人认领**，analysis-done + 代码级验证，patch 待 GPU 验证）。
- 新候选：#8250 async 熵爆炸诊断（analysis-done，草稿待授权）、clip 语义文档澄清（evidence-complete）。

**学习与知识**：GMPO/GSPO/SAPO 三论文精读（含 verl 实现对照、三向 clip 语义对照表、非对称温度、数值证明）；损失注册表唯一性审计（geo_mean 唯一绕过 agg_loss）；DRO 推导补全；排错指南新增 async 熵爆炸章节。产出笔记 3 篇 + 验证脚本 2 个（verify_gmpo_geo_mean_split.py、verify_gmpo_clip_semantics.py）。

**方法学教训**（已入库）：查重 "None" ≠ 0（限流误报）；reporter 深挖 issue ≈ reporter 已开 PR；认领声明 ≠ 会提；verl 认领窗口 0-2 天且 reporter 自修为主。

## 待讨论与下一步

1. **等待用户授权/资源**（长期未决，见「运行接续」4 项）——发布评论/文档、GPU 验证 #8237、方向调整。
2. 若继续自主：verl 新 issue 近 3 天为零（10-16 后停滞）；可转向 vLLM 方向核验（6 月草稿外的机会）或框架源码深化。
3. 按课题逐步关联现有实验执行材料与 notebook/experiments/ 的分析记录；新增实验优先就近组织，维护双向入口。
4. 维护按职责划分的一级目录；新增或整理内容时整体检查归属、引用和输出路径，避免再次堆出按文件类型划分的一级目录。根目录 projects/ 暂不新增。
5. 按价值、使用关系和证据继续筛选历史工具，避免只按文件名批量归档；其余历史工具继续按同一标准检查。

## 接续入口与约束

- 学习与探索：[notebook/readme.md](notebook/readme.md)、[roadmap.md](notebook/roadmap.md)、[knowledge-map.md](notebook/knowledge-map.md)。本轮学习线（RL 损失家族）已收束，新课题未定。
- 贡献：[open-source/README.md](open-source/README.md)、[verl tracker](open-source/verl/tracker.md)。当前活跃：#8237（bugfix，GPU 待验证）、#8250（诊断评论待授权）、clip 语义文档澄清（待授权）；vLLM #39096 评论待授权。
- 本地目录已核实为 `/Users/jackiemac/workspace/perpetual-learning`；旧会话可能仍绑定已不存在的 rollout-infra 路径。GitHub 仓库为 [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)。
- 原有未跟踪文件 diary/2026-08-14.md 属于既有用户工作，保留原状，不夹带提交。
- 历史服务器状态、访问方式和“已验证”结论按需复核，不复述凭据。
