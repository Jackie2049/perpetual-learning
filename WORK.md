# perpetual-learning — 当前工作上下文

更新时间：2026-10-29

## 当前状态

（轮次/批次完整记录只写当日 diary 与专项文档 §0/§16；本文件只保留接续核心，历史记录不在本文件重复。）

- **进行中**：rollout 性能调研专项 Phase 2（M5 定时任务负责，canonical：[notebook/rollout-performance-research.md](notebook/rollout-performance-research.md)）。最近完成：D12 统一可比性账（§13.13，约 60 篇按端到端/rollout/组件/模拟/负项五族归位）＋ D13 T3 账核对（§13.14，T3 21 篇身份↔triage↔计数一致）＋ **D14 Phase 2 收尾状态终核（§13.1 六行均无新增待办，批次表 D1–D14 全部收口；修正 §5.25 第 488 行互斥阅读账陈旧为 B13 后当前账 20/84/21/1）**＋ **收尾第一批 gaps/ideas 提炼（§13.15，19:2x 完成：九类结构化缺口 G1 同栈横评锚点缺失／G2 C 类收敛代价无量化／G3 speculative 统一模型待建／G4 cap 边界未讨论／G5 迁移判据未成文／G6 环境层论证／G7 源码待核 15 项／G8 复现 0＋设备实验空缺／G9 负项证据库；G3/G5/G6 为 survey outline 备料三主题）**＋ **收尾第二批 G4 转稿（19:5x 完成：基于 fork head 71fd3562 与本地 veRL 源码整理 cap/长度边界条件六轴矩阵，入 [memo-8281-prompt-cap-boundary-conditions.md](open-source/verl/memo-8281-prompt-cap-boundary-conditions.md)，G4 升为"已有本地备忘"）**＋ **收尾第三批 G5 转稿（20:2x 完成：serving→rollout 迁移判据四要素成文——V 版本约束/S 陈旧性/G 组完整性/R 随机独立性＋§6.30 二十条核对表＋判定汇总（objective 完全不变 0 条、机制方向可用 6、须补契约 13、C 类风险 2、兼容正面 2），入 [rollout-migration-criteria-memo.md](notebook/rollout-migration-criteria-memo.md)，G5 升为"已有本地备忘"，§15 Q1 反例搜索收口）**＋ **收尾第四批 G3/G6 转稿（21:1x 完成：spec-for-RL 三轴统一性能模型骨架入 [speculative-rl-performance-model-memo.md](notebook/speculative-rl-performance-model-memo.md)＋environment 层论证入 [environment-layer-taxonomy-memo.md](notebook/environment-layer-taxonomy-memo.md)，G3/G6 升为"已有本地备忘"——survey outline 备料三主题全部成文）**；矩阵"另C/待核"旗标经 D1–D13 逐字核对**全部收口**；Heddle 权重版本语义收口（原文未给定＋无公开源码）、HybridFlow 论文时代实现确认不在公共仓库。专项处于**收尾/成果提炼阶段**（下一批按 §13.15 续：G2/G9 可随 outline 备料引用、G7 等源码可获取时核、G8 等算力恢复；不写 survey 正文/不冻结 taxonomy/不启动大规模复现）。详见[当日 diary](diary/2026-10-29.md)、专项 §0/§13.15/§16。
- **算力/网络**：4090-1（219.223.198.62）断连约 28h 持续（10-28 17:07 起电信段路由断裂、代理源被拒；矿机可能已重启占卡——恢复后先核 GPU 形态与后门 PID），已连续多轮呈报用户；昇腾 910C VPN 断第 18 天待用户恢复；**当前无可领算力项**（UniRL #523 L1 需 ≥8 卡物理不可行、DeepInfer 真实权重待磁盘决策）。
- **只读 PR/上游**：#8313 仍 OPEN/MERGEABLE/BLOCKED/REVIEW_REQUIRED（head 71fd3562，④ 回复草稿 v2 待用户逐字审定）；DeepInfer-Go PR #2 仍 OPEN/MERGEABLE/CLEAN（head a0a6940b，等用户审视）；上游四仓库 main 最近无算力线索合并（最近：verl 59f9a42c #8413 trainer token-axis 修复，非本机硬件；AReaL #1833 XPU 后续＋#1819 docs，均非本机硬件；UniRL canonical=Tencent-Hunyuan/UniRL main 5d1da78d）。铁律：外部主流项目主仓库只读，写操作须用户逐项授权。

## 当前用户需求与决定

（用户决策长期有效；执行边界全文见 [AGENTS.md](AGENTS.md#执行边界)。决策条目按日期保留，过时账一律指向专项文档当前账，不在本文件复制。）

- **长期专项：LLM Post-training / RL Rollout 性能优化调研（2026-10-27 用户新立）**：当前阶段=Phase 2 深读（广度已足，转入论文/项目深读，围绕 rollout 性能与 L1–L4 四维）。**2026-10-29 起由 M5 定时任务完全负责**（M1 机器宕机移交；每轮例行核对后执行一批深读/核对并回写）。canonical 主文档 [notebook/rollout-performance-research.md](notebook/rollout-performance-research.md)（§0 状态／§6.0 批次表／§16 下一步；每轮实质工作后必须更新主文档而非只写 diary）。加急期限 ≤2026-10-29 因 M1 停机顺延，剩余批次在 §13.1 记账；HybridFlow 原并行认领不接管。执行约束：不写 survey 正文、不冻结 taxonomy、不启动大规模复现、上游动作受铁律；taxonomy=L1 token／L2 engine／L3 cluster／L4 pipeline co-design × A/B/C（C 类改训练分布须单独标记）。当前深度账 126 身份＝20 T1＋84 T2＋21 T3＋1 未标完成（完整复现 0），以专项 §0 为准。
- **两周技术掘金聚焦（2026-10-23 用户明确）**：**UniRL 与 verl 为两周内技术掘金重点**，重点找 issue/PR 机会；铁律：用户审批前不得在外部主仓库提交（候选先在本地/fork 准备）。当前：verl #8313 已 OPEN 待维护者 approve（无新动作）、#8250 待补证；UniRL 贡献轨道已建（未认领候选 [#523](open-source/unirl/candidate-523-anchor-lifecycle.md)，fork 分支 `fix-523-anchor-lifecycle` 已备，待用户审批）；DeepInfer 降为常规关注。
- **主机分工（2026-10-20 用户明确，2026-10-21/24 强化）**：所有 GPU/NPU 访问、环境与实验由 macbook-M5 的 Agent 负责；M1 不自动探测 GPU/SSH/VPN、不启动或干预远程任务。AReaL prompt_mean 候选/PR 整项由 M5 单一负责（M1 阶段收尾，定时运行跳过该项）；统一[交接说明](open-source/areal/handoff-m5.md)。**AReaL prompt_mean 实测已闭环（2026-10-20 恢复 VPN 后受控实测：跨 M 与 token_mean 行为一致，不构成 defect，候选修复撤销，不提 issue/PR）**，作为 review 依据存档，见[源码笔记 §10.10](notebook/frameworks/areal-architecture-source-reading.md) 与 [tracker](open-source/areal/tracker.md)。
- **高优调研项 DeepInfer（2026-10-20）**：DeepSeek 生产推理框架高优先调研（Apache-2.0，SM100 + Ascend 950 NPU）。外部 PR 采纳门槛高、周期长，结论见 [deepseek-ecosystem-pr-adoption.md](notebook/frameworks/deepseek-ecosystem-pr-adoption.md)；综合性价比 verl/vLLM/AReaL 仍高于 DeepSeek 生态。**平民版（适配 4090/A100/H100/910C）专题（2026-10-20 用户立项）**：结论=Python 算子层后端无关，堵点在 flash_mla/deep_gemm/deep_ep 三个 Blackwell/A5-only kernel 包＋架构检查＋tilelang 重编译，见 [deepinfer-civilian-port.md](notebook/frameworks/deepinfer-civilian-port.md)。**mock 权重链路 M5 真机已跑通（10-23）**：LLMEngine 初始化＋prefill/decode 全链路（两轮独立 PASS），材料 [experiments/deepinfer-mock-chain/readme.md](experiments/deepinfer-mock-chain/readme.md)；真实权重环节（约 300GB、磁盘剩 109G）待用户磁盘决策后推进。
- **开源技术掘金未来一周三大重点（2026-10-20 用户定）**：**DeepInfer / verl / AReaL**（vLLM 降为次优先级）。
- **UniRL 高优（2026-10-22 用户新增）**：Apache-2.0、仅 NVIDIA GPU、外部 PR 合入率高（近 100 已关闭 PR 合并率 94%），定位学习＋潜在贡献轨道，见 [调研笔记](notebook/frameworks/unirl-survey.md)。
- **方向偏好（2026-10-20）**：自主运行重点跟进 verl 与 AReaL 两个 RL 训练框架（verl 已是贡献主轨道；AReaL 新增关注/潜在贡献轨道）。
- **项目名称/仓库标识**：统一 `perpetual-learning`（历史资料中 `rollout-infra` 为旧名；机器实际目录以当前环境为准）。
- **仓库组织原则（用户既定，全文见 AGENTS.md「文件与交接」）**：README 为总览/对外导航，不放内部计划/待办；notebook/ 分 fundamentals/frameworks/experiments/roadmap/knowledge-map；tools/ 只沉淀复用价值工具（一次性归 tools/legacy/）；open-source/<framework>/ 按目标项目保存贡献材料；diary/ 按日期记录；同一项成果维护一个主要位置，其他地方链接引用；完成过程写入 diary，当前状态与本文件维护。

## 运行接续（当前任务状态 / 下一步 / 等待条件）

- **rollout 专项（M5）**：D12 统一可比性账（§13.13）＋ D13 T3 账核对（§13.14）＋ **D14 Phase 2 收尾状态终核（§13.1 六行无新增待办，批次表 D1–D14 全部收口，§5.25 账目修正）**＋ **收尾第一批 gaps/ideas 提炼（§13.15，九类缺口）**＋ **收尾第二批 G4 转稿（19:5x，[memo-8281-prompt-cap-boundary-conditions.md](open-source/verl/memo-8281-prompt-cap-boundary-conditions.md)，六轴边界矩阵）**＋ **收尾第三批 G5 转稿（20:2x，[rollout-migration-criteria-memo.md](notebook/rollout-migration-criteria-memo.md)，serving→rollout 迁移判据四要素 V/S/G/R＋二十条核对表＋判定汇总）**＋ **收尾第四批 G3/G6 转稿（21:1x，[speculative-rl-performance-model-memo.md](notebook/speculative-rl-performance-model-memo.md)＋[environment-layer-taxonomy-memo.md](notebook/environment-layer-taxonomy-memo.md)，G3/G6 升为已有本地备忘）**完成；矩阵"另C/待核"旗标经 D1–D13 全部收口，**survey outline 备料三主题（G3/G5/G6）全部成文**。专项处于**收尾/成果提炼阶段**：下一轮按 §13.15 续推进（G2/G9 可随 outline 备料引用、G7 等源码可获取、G8 等算力恢复），仍不写 survey 正文/冻结 taxonomy/启动大规模复现。每轮流程：例行核对（算力/PR/上游只读）→ 认领并执行一批 → 回写专项文档 §0/§13.15/§16 ＋ 当日 diary → 提交推送。
- **4090-1 网络（等待用户处理，多轮呈报）**：断连约 27h 持续，电信段 220.181.38.148 断裂签名（直连 ssh/ping 超时；代理 SOCKS5 CONNECT 成功但无 SSH banner＝服务器在线证据、代理源被拒）；keepalive 连续失败。**恢复后立即核对 GPU 形态与后门 PID**（矿机可能已重启占卡；后门 eradication 仍待管理员重装系统＋zxw 密码轮换）。
- **昇腾 910C VPN（等待用户恢复）**：断第 18 天（网关 10.205.0.1 通、目标 192.168.13.174 100% 丢包、EasyConnect 进程在跑）。
- **等待用户决策**：#8313 ④ 回复草稿 v2 待逐字审定（[draft-8313-review-reply.md](open-source/verl/draft-8313-review-reply.md)）；DeepInfer-Go PR #2 待审视；UniRL #523 待审批（fork 分支已备）；磁盘清理决策（②档 242G＋③档 93G＋fsy/ 558G）→ DeepInfer 真实权重前置；verl #8250 补证待授权。
- **M1 学习/贡献线**：各学习/收束条目均已入库 `notebook/frameworks/` 各笔记与 `experiments/scripts/` 验证脚本，不再在本文件列条目；M1 机器宕机（2026-10-29），其原负责 rollout 专项已移交 M5。
- **上游只读核对基线**：verl `59f9a42c`（#8413 trainer token-axis fix，非算力线索）、DeepInfer `553741557`、UniRL `5d1da78d`（canonical Tencent-Hunyuan/UniRL）、AReaL `890de371`（#1833 XPU＋#1819 docs，非本机算力线索；最近合并均无算力线索；新 open PR 均作者侧，逐轮 gh api 复核）。

## 待讨论与下一步

1. **rollout 专项下一批（收尾/成果提炼）**：批次表 D1–D14 全部收口、矩阵旗标全部收口、收尾第一批 gaps/ideas 提炼完成（§13.15 九类缺口）、第二~四批 G4/G5/G3/G6 转稿完成（四份本地备忘，**survey outline 备料三主题全部成文**）；下一批按 §13.15 续推进（G2/G9 可随 outline 备料引用、G7 等源码可获取、G8 等算力恢复）。
2. **磁盘清理决策（②③ 档 + fsy/）**：DeepInfer 真实权重（约 300GB）下载的前置条件，待用户拍板（分类标准见 [disk-cleanup-risk-classes](experiments/) 相关记录）。
3. **4090-1 网络与昇腾 VPN 恢复**：待用户处理；恢复后先核 GPU 形态与后门 PID 再判定算力项。
4. **#8313 ④ 回复发布审定**：草稿 v2 待用户逐字过目后按明确指示发布（铁律逐项授权）。
5. **UniRL #523 审批**：fork 分支已备，达到可发布标准后交用户拍板。

## 接续入口与约束

- 学习与探索：[notebook/readme.md](notebook/readme.md)、[roadmap.md](notebook/roadmap.md)、[knowledge-map.md](notebook/knowledge-map.md)。
- 贡献：[open-source/README.md](open-source/README.md)、[verl tracker](open-source/verl/tracker.md)、[AReaL tracker](open-source/areal/tracker.md)、[UniRL tracker](open-source/unirl/tracker.md)。
- 专项：[rollout-performance-research.md](notebook/rollout-performance-research.md)（§0 是恢复研究状态的入口）。
- 真机交接：[experiments/handoff-m5.md](experiments/handoff-m5.md)、[open-source/areal/handoff-m5.md](open-source/areal/handoff-m5.md)。
- 工作目录：`/Volumes/Agent/workspace/projects/perpetual-learning`（旧会话可能仍绑定 rollout-infra 旧路径）；GitHub 仓库 [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)。
- 未跟踪文件 `diary/2026-08-14.md` 属既有用户工作，保留原状、不夹带提交。
- 历史服务器状态、访问方式和"已验证"结论按需复核，不复述凭据。
