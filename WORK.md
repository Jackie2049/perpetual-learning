# perpetual-learning — 当前工作上下文

更新时间：2026-10-30

## 当前状态

（轮次/批次完整记录只写当日 diary 与专项文档 §0/§16；本文件只保留接续核心，历史记录不在本文件重复。）

- **进行中**：rollout 性能调研专项 Phase 2 收尾/成果提炼（M5 定时任务负责，canonical：[notebook/rollout-performance-research.md](notebook/rollout-performance-research.md) §0/§16）。最近完成：**收尾第三十批（13:5x，G3 备忘并入候选 C2 三轴补测）**——把第二十九批 16 组合引擎级补测四点发现正式并入 G3 备忘（[备忘](notebook/speculative-rl-performance-model-memo.md) 新增 §7、§1/§4/§5 状态同步）：① 轴 1 提议器强度对比（0.5B drafter E[L]=4.12/4.13 in512·2.48/2.63 in2048 ≈ ngram 1.95–2.97×、A(p1) 0.80–0.82 vs 0.14–0.21、曲线单调缓降无 ngram 自增强形态）；② 强提议器仍净负（spec/AR 0.20–0.25 ≈ ngram 同量级、c_draft/c_t≈1.1–2.3、盈亏需 E[L]≈11.5–17.6，批 28"4–9"低估验证结构成本）；③ 轴 3 批规模（ngram spec/AR in512 0.16→0.27→0.27→0.24、in2048 0.31→0.27→0.27→0.23 b8→b64 平台后缓降、AR 扩批 6.87×/5.94×——批规模不挽救弱提议器至 b64）；④ 方法学（seed=0 引擎跨进程非 bit 确定、单次 A(p) p1 方差 ±0.03–0.06、候选 C 归档为单样本、批 28 形状/方向结论稳健、部分运行间 bit 级复现）——**survey outline「统一性能模型」主题论据骨架完整**（轴 2 N/A 维持、EAGLE/Medusa 变体未测）。前批：收尾第二十九批（13:2x）G3 三轴补测执行＋候选 C2 归档（[experiments/rollout-g8-candidate-c2/](experiments/rollout-g8-candidate-c2/) 35 文件 sha256 全匹配）＋独立核验。例行核对（13:4x）：4090-1 连通——GPU 0/1/3/4/5/6/7 空闲 1MiB、GPU 2 liyaowei embed 21728MiB、load 0.22 无矿机/训练、/home 16G、venv cu129 基线完好、watchdog 结构性无抢占；昇腾 VPN 断第 23 天待用户恢复；#8313（71fd3562）／DeepInfer-Go fork #2（a0a6940b）OPEN/MERGEABLE 不变；verl 321e4e45／DeepInfer 55374155／UniRL 5d1da78d／AReaL d25f2929 无变化、vLLM 8730884a（#64655 CI ROCm 非算力）。下一轮：按 §13.15 续推进——G7 余 13 项待源码可获取时核、AcceRL 冲突待作者澄清、G3 轴 2（N/A）维持、EAGLE/Medusa 变体未测（留待单独算力窗口决策）；复现计数口径待用户确认。批次/轮次完整记录见专项 §0/§16 与当日 diary。
- **用户任务已完成（2026-10-30 00:2x–01:1x，M1，八轮）**：GitHub Pages「对外呈现」及报告页按用户标注截图/追加指示更新——① 两张 rollout 卡类别改「强化学习」＋新增分类筛选栏（hash 参数可分享）；② 两卡标题改名「Rollout综述项目-研究大纲/论文集锦」；③ 主题改「强化学习infra」并合并 RL infra（Lego-RL 同类，筛选栏 全部 10/强化学习infra 3/AgenticRL/News/Harness/vLLM 3/0daybot）；④ 大纲报告页分类法表行序改 L4→L1（宏观→微观自顶向下，canonical 文档同名表仍为 L1→L4 定义序待 M5 侧定夺）；⑤ L4 核心问题改「rollout 与其他阶段的协同（例如 train/reward/env/weight-update）」（正文与大纲页两处同步）；⑥ taxonomy 层名简化改名 L1 Model Execution / L2 Infer Engine / L3 Rollout Cluster / L4 RL Pipeline（正文与大纲页两处同步）；⑦ taxonomy 抓手列 24 项各加括号代表工作超链接（优先 126 篇已核论文集固定版本链接，表下加「仅标落点、数值不可跨层相加」说明；正文 §4 同步；partial rollout/multi-sample 两处代表选取待用户确认）；⑧ 核心问题列 HOW 问句化——L2 定稿为用户逐字表述「一个 engine 如何高效处理一批路由到该实例的trajectories」/L3「一批 trajectory set 怎么整体最快」/L4「rollout 怎么与其他阶段协同」（L1 本为问句不动，正文 §4 与大纲页两处同步）；⑨ taxonomy 表头首列「层」→「架构分层」（正文 §4 与大纲页两处同步）。本地浏览器实测通过，推送 main 自动部署；详见[当日 diary](diary/2026-10-30.md)。
- **算力/网络**：4090-1 连通——GPU 0/1/3/4/5/6/7 空闲 1MiB、GPU 2 liyaowei embed 21728MiB、无矿机/训练、/home 余 17G、venv cu129 基线完好；G8 三候选账 A→B→C 全部完成＋候选 C2 三轴补测（候选 A GRPO 基线／候选 B 引擎级 decode／候选 C ngram A(p)＋净负项／候选 C2 draft_model 强提议器＋批规模，见专项 §14 与 diary）；昇腾 VPN 断第 23 天待用户恢复。
- **只读 PR/上游**：#8313 仍 OPEN/MERGEABLE/BLOCKED/REVIEW_REQUIRED（head 71fd3562，updatedAt 10-25，④ 回复草稿 v2 待用户逐字审定）；DeepInfer-Go PR #2 仍 OPEN/MERGEABLE/CLEAN（head a0a6940b，等用户审视）；上游四仓库 main 最近无算力线索合并（最近：verl a08ea02a7 #8251 NPU-side FSDP-Turbo fix，非本机硬件；AReaL 890de371 已记；UniRL canonical=Tencent-Hunyuan/UniRL main 5d1da78d）。铁律：外部主流项目主仓库只读，写操作须用户逐项授权。

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

- **rollout 专项（M5）**：批次表 D1–D15 全部收口、九类缺口 G1–G9 提炼完成（§13.15）＋四份备忘转稿（G3/G4/G5/G6）＋G2/G9 引用索引（§13.16）＋G7 源码扫描/深挖（§13.17/§13.19）＋D15 归族账（§13.18）＋provenance 登记/§4 行序定夺＋**G8 三候选账 A→B→C 全部完成（候选 A GRPO 训练基线 90/90 步 27.28s/step acc 0.4917；候选 B 引擎级 decode 812.9–3319.4 tok/s 12 组合；候选 C ngram-spec A(p) 8 组合＋净负项记 G9）**＋**收尾第二十八批 G3 备忘定量校验（11:5x，备忘 §6）**＋**收尾第二十九批 G3 三轴补测（13:2x：轴 3 批规模 b32/b64＋强提议器 draft_model 0.5B 16 组合，draft E[L]=4.1 仍净负 spec/AR 0.20–0.25、批规模 b64 不转正、单次 A(p) 方差 ±0.03–0.06 注记，归档 experiments/rollout-g8-candidate-c2/）**。下一轮：按 §13.15 续推进（G7 余 13 项待源码、AcceRL 待作者澄清、G3 轴 2 N/A 维持、EAGLE/Medusa 变体未测）；复现计数口径待用户确认（候选 A 是否计"复现 1"）。轮次完整记录见专项 §0/§16 与 diary/2026-10-30.md。
- **4090-1 网络（连通）**：GPU 0/1/3/4/5/6/7 空闲 1MiB、GPU 2 liyaowei embed 21728MiB；无矿机/训练；/home 余 17G；venv rollout-g8 基线完好（torch 2.11.0+cu129 cuda=True、vllm 0.24.0）；G8 三候选账 A→B→C 全部完成（结果见专项 §14 与 diary）。
- **等待用户决策**：#8313 ④ 回复草稿 v2 待逐字审定（[draft-8313-review-reply.md](open-source/verl/draft-8313-review-reply.md)）；DeepInfer-Go PR #2 待审视；UniRL #523 待审批（fork 分支已备）；磁盘清理决策（②档 242G＋③档 93G＋fsy/ 558G）→ DeepInfer 真实权重前置；verl #8250 补证待授权；**rollout 综述 §1 研究问题表述出处待用户确认（2026-10-30 provenance 考证）**——本机可查范围内该句为 agent 形式化非用户逐字原文（ChatGPT 桌面端无本地会话库、Codex 237 会话/161 线程查尽无本课题讨论、立项逐字指令在宕机 M1 的 ~/.zcode 会话库），三个确认渠道：① chatgpt.com/ChatGPT app 翻 10-26 前后对话；② M1 会话库取逐字指令；③ 以 WORK.md@8d8efa5 注册文为准逐句确认后补 §0 与 web 大纲页 provenance（详见[专项 §1](notebook/rollout-performance-research.md)与当日 diary）。
- **M1 学习/贡献线（2026-10-30）**：[梯度累积的目标与分母](notebook/fundamentals/gradient-accumulation-benchmark-rtx4090.md#不等长序列的累积归一化)：不等长CPU交叉熵／SGD核局部mean再平均可反转梯度；旧OPT名义吞吐与未归一化脚本不认证更新等价，撤回通用定律／7B容量外推，历史表与GPU脚本保留。下轮依实质反馈／独立缺口，不扩小例、改M5环境／配置或排算力；rollout仍M5全责，[旧D1草稿未应用](diary/2026-10-30.md#0033--m1恢复接续与m5单一负责人核对)。
- **上游只读核对基线（2026-10-30 13:4x）**：verl `321e4e45`（#8401 XPU Docker，非算力）；vLLM `8730884a`（#64655 CI ROCm 检查，非算力；较 335dd77b09 前进）；DeepInfer `55374155`（#218 环境变量校验，非算力）；UniRL canonical=Tencent-Hunyuan/UniRL `5d1da78d`（#631 已合并）；AReaL canonical=areal-project/AReaL `d25f2929`（#1825 VLM 预处理，非算力）；#8313 head 71fd3562 OPEN/MERGEABLE、DeepInfer-Go fork #2 head a0a6940b OPEN/MERGEABLE（均在用户仓库内，未触上游主仓库）。

## 待讨论与下一步

1. **rollout 专项下一批（收尾/成果提炼）**：批次表 D1–D15 全部收口、矩阵旗标全部收口、九类缺口 G1–G9 提炼（§13.15）＋四份备忘转稿（G4/G5/G3/G6）＋§15 Q4 收口＋G2/G9 备料索引（§13.16）＋G7 源码可得性扫描/深挖/复扫（§13.17/§13.19）＋D15 归族账（§13.18）＋provenance 登记/§4 行序定夺（第十一批）全部完成；**G8 三候选账：候选 A 完整执行完成（90/90 步、27.28s/step、1498.7 tokens/s、gsm8k acc 0.4917、checkpoint 9.1G 落盘，§13.1 G1 同栈锚点账建立）＋候选 B 完整测量完成（12 组合、decode 812.9–3319.4 tok/s 双点拟合、summary.json 修复、归档 experiments/rollout-g8-candidate-b/，§13.20 引擎级基线边界）＋候选 C 完整执行完成（8 组合 ngram-spec、A(p) 四条曲线 0.207→0.014 逐位置衰减、Mean accept len 1.27–1.39、ngram 净负项入 §13.16 G9、归档 experiments/rollout-g8-candidate-c/）——**A/B/C 全部完成、复现计数口径待用户确认**——**收尾第二十九批 G3 三轴补测完成（13:2x：轴 3 批规模 b32/b64 扩展——ngram spec/AR b16 后平台/缓降、批规模不挽救弱提议器至 b64；强提议器 draft_model 0.5B——E[L]=4.1 仍净负 spec/AR 0.20–0.25、E[L] 增益被 drafter 前向成本吸收、盈亏需 E[L]≈11.5–17.6；方法学——seed=0 跨进程非 bit 确定、单次 A(p) 方差 ±0.03–0.06、候选 C 归档为单样本）**——G7 余 13 项待源码可获取时核、AcceRL 冲突待作者澄清、**G3 轴 2 版本差维持 N/A、EAGLE/Medusa 变体未测**；昇腾 VPN 断第 23 天待用户恢复。
2. **磁盘清理决策（②③ 档 + fsy/）**：DeepInfer 真实权重（约 300GB）下载的前置条件，待用户拍板（分类标准见 [disk-cleanup-risk-classes](experiments/) 相关记录）。
3. **4090-1 网络（已恢复）与昇腾 VPN（第 23 天）**：4090-1 已恢复且矿机未重启；cl 训练已结束、watchdog 结构性无抢占（稳定窗口达成）、**候选 A 完整执行完成（90/90 步、27.28s/step、1498.7 toks/s、acc 0.4917——§14 第二十批账）＋候选 B 完整测量完成（12 组合、decode 812.9–3319.4 tok/s、summary.json 修复、归档——§14 第二十五批账）＋候选 C 完整测量完成（8 组合 ngram-spec、A(p) 四条曲线、归档——§14 第二十六批账）＋候选 C2 三轴补测完成（16 组合 draft_model 强提议器＋批规模 b32/b64，归档——§14 第二十九批账）**；**G8 三候选账 A→B→C 全部完成**；昇腾 VPN 仍待用户恢复。
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
