# Rollout Performance Research — LLM Post-training / RL Rollout 性能优化调研

> 本文件是本专题唯一的 canonical knowledge base。任何会话/Agent 接手时**只读本文件**即应在 10–20 分钟内恢复研究状态；详细证据存放于链接指向的材料，但恢复研究状态不依赖它们。
> 远景（非当前任务）：systems-oriented survey《Efficient Rollout Systems for LLM Post-Training》。**当前阶段禁止**撰写正文/Introduction/Related Work、冻结 taxonomy、批量生成低价值摘要、启动大规模复现、在外部主仓库发 issue/PR/comment。

## 0. Project Status

- **阶段**：Phase 1 — recall-first literature & code investigation（广扫建库，不收束观点）
- **最近更新**：2026-10-27（Round 1：骨架建立 + 种子注册 + arXiv 验证检索 2 批）
- **覆盖**：arXiv 验证 ~40 篇（有 ID 实证）/ 知识注册 ~25 篇（标 ◇）/ 待验证种子 ~8 篇（标 ？）→ **候选 corpus ≈ 73**
- **深读 / 代码阅读**：0 / 0
- **最高优先级**：收敛 corpus 至 100–150；从验证集中标出 A Core 短名单
- **下一步**：见 §16

## 1. Research Question & Scope

- **研究问题**：LLM post-training（RL/推理类后训练）中，rollout 阶段的性能由什么决定、可以被系统化优化到什么程度？
- **范围**：rollout-native 系统/论文（L1 token 级 → L4 pipeline 级）+ serving→rollout 迁移分析 + 代码实现对照。
- **不做**：reward/算法本身的改进（只关心其对 rollout workload 的影响）、pretraining、纯 serving 部署。
- **验收标准（本阶段）**：corpus 100–150 篇（rollout-native 80–120 + serving 基础 20–30），完成 A Core / B Supporting / C Boundary 分级；broad scan 收敛后再进入深读。

## 2. Serving vs Rollout（核心工作假设，持续验证/修正）

| 维度 | Serving | Rollout |
|---|---|---|
| 请求模型 | 独立用户请求，到达不可控 | 一批 trajectory，同批生成、同步/半同步推进 |
| 核心指标 | TTFT / TPOT / latency / goodput | trajectory throughput / **rollout makespan** / bubble ratio / GPU-hours / RL step time / time-to-quality |
| 关键结构 | 单请求延迟敏感 | **T_rollout ≈ max_i T_trajectory_i**（sync RL）——平均 tokens/s 高 ≠ rollout phase 快 |
| 版本约束 | 单一模型权重 | policy 连续更新 → drafter/cache/调度决策随版本过期（staleness） |
| 正确性约束 | 输出质量 | on-policy 性（权重版本一致性）、新旧 logp 一致性 |

**假设状态**：未验证。每个 corpus 条目都要问一句"该工作隐含的 serving/rollout 假设是什么"，反例记入 §11。

## 3. Performance Model

- Sync RL：`T_rollout ≈ max_i T_trajectory_i` → 长尾/straggler 支配 makespan → L3 调度类工作的理论起点。
- Async RL：makespan 概念弱化，改为 `system throughput × trajectory freshness`（staleness 换 throughput）。
- Bubble 的来源分类：① 长尾等待（sync）② train/rollout 切换空隙 ③ weight sync ④ environment/tool 等待 ⑤ 资源空闲（colocation 失配）。
- **待建**：把五类 bubble 形式化（比值定义 + 各自的优化杠杆），作为 §13 gaps 的推导框架。

## 4. Taxonomy

**层（L1–L4，允许反例修订）**：

| 层 | 核心问题 | 代表杠杆 |
|---|---|---|
| L1 Token/Model Execution | 单 token/decode step 怎么更便宜 | attention/kernel、KV cache、quantization/FP8、speculative decoding、MTP、draft model |
| L2 Engine/Replica | 一个 engine 高效完成一组 trajectories | continuous batching、chunked prefill、prefix/KV reuse、multi-sample、sleep/wake |
| L3 Cluster | 一批 trajectory set 整体最快（**minimize makespan**） | global scheduling、length prediction、straggler/tail、migration、cache-aware routing、elastic |
| L4 RL Pipeline Co-design | rollout 与 trainer/reward/env/weight-update 协同 | sync/async、colocated/disaggregated、overlap、weight sync、staleness、partial rollout、resource allocation |

**类型（A/B/C，第二维度）**：
- **A** Make each trajectory cheaper（C(trajectory)↓）：speculative、MTP、KV/prefix reuse…
- **B** Execute the trajectory set better（Schedule({traj_i})）：batching、length-aware scheduling、migration…
- **C** Generate less/better rollout work（改变 {traj_i}，**可能改变训练分布，必须单独标记**）：early stop、partial rollout、filtering、adaptive allocation、trajectory shaping…

反例登记：当前无。凡 taxonomy 解释不了的工作记入 §15 并提修订。

## 5. Literature Matrix

字段约定：完整 24 字段在深读时逐项补全（§6）；矩阵阶段记录 `arXiv ID | 日期 | 标题 | 层 | 类型 | 一句话机制 | 置信度`。置信度：✅=arXiv 实证（2026-10-27 检索）；◇=知识注册待核；？=种子待验证。

### 5.1 L4 — RL Pipeline / Async / Disaggregation

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| 2505.24298 | 2025-05 | AReaL | L4 | B | 大规模 async RL，decoupled actor/learner，staleness 界控制 | ✅ |
| 2409.19256 | 2024-09 | HybridFlow / veRL | L4 | B | hybrid 控制流编程模型，单机多卡/多机 rollout 引擎统一抽象（EuroSys'25） | ✅ |
| 2507.13833 | 2025-07 | DistFlow | L4 | B | 全分布式 RL 框架，controller-worker 解耦 | ✅ |
| 2504.15930 | 2025-04 | StreamRL | L4 | B | disaggregated stream generation，异构弹性，气泡消除 | ✅ |
| 2505.24034 | 2025-05 | LlamaRL | L4 | B | 分布式 async RL 框架（Meta） | ✅ |
| 2510.12633 | 2025-10 | Laminar | L4 | B | scalable async RL post-training | ✅ |
| 2604.26256 | 2026-04 | DORA | L4 | B | scalable async RL system | ✅ |
| 2601.12784 | 2026-01 | StaleFlow | L4 | B/C? | disaggregated 下 staleness-aware 数据管理，缓解数据偏斜 | ✅ |
| 2507.01663 | 2025-07 | AsyncFlow | L4 | B | async streaming RL（参数 shuffle 式权重同步） | ✅ |
| 2606.26997 | 2026-06 | RolloutPipe | L4 | B | disaggregated 下 pipelined rollout-training 重叠 | ✅ |
| 2410.18252 | 2024-10 | Asynchronous RLHF | L4 | B | off-policy async RLHF，吞吐优先 | ✅ |
| 2406.14088 | 2024-06 | ReaLHF | L4 | B | allocation + resharding 优化 human-feedback RL（SOSP'24） | ◇ |
| 2412.15179 | 2024-12 | RLHFuse | L4 | B | RLHF 阶段级 fused 调度（NSDI'25） | ◇ |
| 2405.11143 | 2024-05 | OpenRLHF | L4 | B | 主流开源 RLHF 框架（Ray 3.0 重构） | ✅ |
| 2308.01320 | 2023-08 | DeepSpeed-Chat | L4 | B | 早期 RLHF 训练系统 | ✅ |
| 2409.13221 | 2024-09 | Stage Fusion | L4 | B | RLHF 阶段融合调度 | ✅ |
| 2605.12070 | 2026-05 | Missing Old Logits in Async Agentic RL | L4 | B | async 语义失配与 off-policy 修正 | ✅ |
| 2606.03070 | 2026-06 | ASymPO | L4 | B | 非对称规模 async post-training，无行为信息 | ✅ |
| 2602.01826 | 2026-02 | Training-Inference Mismatch = Optimization Problem | L1/L4 | B | 训推失配的优化视角，LR 调度修复 | ✅ |

### 5.2 L3 — Long-tail / Scheduling / Cluster

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| 2509.21009 | 2025-09 | RollPacker | L3 | B | 长尾 rollout 打包压缩，加速 sync post-training | ✅ |
| 2603.23414 | 2026-03 | SortedRL | L3 | B | 在线 length-aware 调度 | ✅ |
| 2608.22788 | 2026-08 | TailSieve | L3 | C/B | partial rollout 引导的尾路由 | ✅ |
| 2607.17299 | 2026-07 | WAR | L3/L4 | C/B | workload-aware rollouts for sync agentic RL | ✅ |
| 2605.30859 | 2026-05 | DARTS | L3 | **C** | distribution-aware active trajectory **shaping**（改分布，重点标记） | ✅ |
| 2408.13510 | 2024-08 | Intelligent Router | L3 | B | workload-aware load balancing（serving 迁移候选） | ✅ |
| ？ | ? | Seer | L3 | B | ？"trajectory prefetch/length 预测"方向 | ？ |
| ？ | ? | Weave | L3/L4 | B | ？（种子，待验证） | ？ |

### 5.3 L1/L2 — RL-aware Speculative Rollout（"为什么 serving spec decoding 不能直接套 RL"专题）

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| 2509.23232 | 2025-09 | SPEC-RL | L1/L2 | A | speculative rollouts 加速 on-policy RL | ✅ |
| 2508.18588 | 2025-08 | RhymeRL（History Rhymes） | L1/L2 | A | 历史轨迹复用作 draft | ✅ |
| 2510.26475 | 2025-10 | ReSpec | L1/L2 | A | RL 系统中 speculative decoding 优化 | ✅ |
| 2511.13841 | 2025-11 | DAS（Beat the long tail） | L1/L3 | A/B | distribution-aware speculative decoding for RL | ✅ |
| 2605.08862 | 2026-05 | BubbleSpec | L1/L3 | A/B | 把 sync 长尾气泡变成 speculative draft 时机 | ✅ |
| 2606.18967 | 2026-06 | EfficientRollout | L1/L2 | A | system-aware **self**-speculative decoding for RL | ✅ |
| 2601.09083 | 2026-01 | SRT | L1/L2 | A | tree-structured cache 的 speculative rollout | ✅ |
| 2610.13202 | 2026-10 | S³R | L1/L2 | A | self-sparse speculative decoding | ✅ |
| 2610.32881 | 2026-10 | NgramRL | L1/L2 | A | 结构化生成的 rollout-aware n-gram speculative | ✅ |
| 2610.15246 | 2026-10 | ObsSpec | L1/L4 | A | **observation speculation** for agentic RL（跳过环境等待？深读确认） | ✅ |
| 2606.12370 | 2026-06 | Breaking Entropy Bounds（MTP+RS for RL） | L1 | A | MTP+拒绝采样加速 RL 训练（MTP-RL 方向候选） | ✅ |
| 2602.06932 | 2026-02 | Unified Training-Serving Speculative | L1/L4 | A | 训推一体 speculative 系统 | ✅ |
| ？ | ? | FastGRPO / SpecRoll / MTP-RL | L1 | A | 种子待验证 | ？ |

### 5.4 Agentic Rollout

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| 2603.18815 | 2026-03 | ProRL Agent | L4 | B | **Rollout-as-a-Service** for multi-turn agent RL | ✅ |
| 2605.15565 | 2026-05 | AstraFlow | L4 | B | dataflow-oriented agentic RL | ✅ |
| 2512.22560 | 2025-12 | RollArt | L4 | B | disaggregated multi-task agentic RL at scale | ✅ |
| 2603.28101 | 2026-03 | Heddle | L3/L4 | B | distributed orchestration for agentic rollout | ✅ |
| 2511.16108 | 2025-11 | SkyRL-Agent | L4 | B | multi-turn agent RL 训练框架 | ✅ |
| 2605.05262 | 2026-05 | Rollout Informativeness（submodular tree search） | L3 | **C** | 固定预算下最大化 rollout 信息量（改分布） | ✅ |
| 2606.05597 | 2026-06 | AsyncWebRL | L4 | B | 多步 web agent async RL | ✅ |
| 2603.18464 | 2026-03 | AcceRL | L4 | B | VLA 场景 async RL + world model | ✅ |
| 2605.08527 | 2026-05 | MARLaaS | L4 | B | multi-tenant async RL as a service | ✅ |
| 2608.14498 | 2026-08 | Rollplex | L2/L4 | B | VLM post-training 跨阶段 GPU 空分共享 | ✅ |
| 2602.00482 | 2026-02 | AREAL-DTA | L1 | A | RL 中 dynamic tree attention（与 AReaL 相关？待核） | ✅ |
| 2610.28998 | 2026-10 | FlashNext-RL | L1/L4 | A/B | long-horizon agentic RL 系统优化案例 | ✅ |

### 5.5 Serving 基础（supporting corpus，不标 rollout-native）

| ID/出处 | 标题（短） | 迁移价值一句话 | 置信 |
|---|---|---|---|
| 2205.12318 | Orca | iteration-level scheduling 的起源 | ◇ |
| 2309.06180 | vLLM / PagedAttention | KV 分页管理 | ◇ |
| 2310.07204 | SGLang / RadixAttention | prefix 复用（rollout 中 multi-sample/partial rollout 直接受益） | ◇ |
| 2205.14135 / 2307.08691 | FlashAttention / FA2 | attention kernel | ◇ |
| 2501.01005 | FlashInfer | rollout 引擎常用 kernel 库 | ◇ |
| 2403.02310 | Sarathi-Serve | chunked prefill（rollout 中 prompt 占比高） | ◇ |
| 2401.09670 | DistServe | prefill/decode 分离（→ colocation/disaggregation 之争的 serving 原型） | ◇ |
| 2407.00079 | Mooncake | KV cache 中心化（→ rollout KV 复用） | ◇ |
| 2401.10774 | Medusa | 多 token 预测（→ MTP-RL 线的 serving 侧起源） | ◇ |
| 2401.15077 | EAGLE | feature 级 speculative decoding（RL-aware drafter 的对照基线） | ◇ |

### 5.6 待验证种子（用户清单中尚未在检索中命中）

？Seer ｜ ？Weave ｜ ？FastGRPO ｜ ？SpecRoll ｜ ？MTP-RL（2606.12370 为方向候选）｜ ？RLHFuse（◇2412.15179 待核）｜ ？"Taming the Long-Tail / adaptive drafter" ｜ ？Online Draft Co-Training for long-context RL

## 6. Paper Deep Reads

（空。优先级建议：AReaL、HybridFlow/veRL、StreamRL、RollPacker、RhymeRL、BubbleSpec、StaleFlow——覆盖四层 + 三大机制家族。深读模板：问题/瓶颈 → 性能模型 → 机制 → 实验条件与增益（保留硬件/batch/长度分布）→ 与 taxonomy 对照 → 可迁移性 → 疑点。）

## 7. Code & System Landscape

**本地已有资产**（重要：不用重新 clone）：
- `/Users/jackiemac/workspace/verl`：veRL 源码（含 #8281 工作分支，近期深读过 `experimental/agent_loop/` 全链）；
- `/Users/jackiemac/workspace/UniRL-main`：UniRL 源码（train/stack 双栈结构已深读）；
- M5 服务器：verl-e2e 环境（4090-1，可跑 rollout 冒烟）。

**待建代码地图**（按 §9 的锚点逐框架定位）：veRL（已部分完成，见 §8）→ AReaL → vLLM/SGLang（引擎层 scheduler）→ OpenRLHF → rLLM/SkyRL/NeMo-RL。

## 8. Paper → Mechanism → Code Mapping

以 veRL 为第一个完整样本（已有源码基础）：

| 概念锚点 | veRL 代码位置（80ee82bd 代） | 状态 |
|---|---|---|
| rollout manager / worker | `verl/experimental/agent_loop/agent_loop.py`（AgentLoopBase、_run_agent_loop、asyncio.gather:719） | ✅ 已读 |
| prompt cap / truncation 契约 | 同上 `_cap_text_prompt_length`（#8281 修复点） | ✅ 已读 |
| 单模型训练栈生命周期 | `verl/workers/` + 本专题外已有笔记 | ✅ 已读 |
| sync barrier / weight update / sleep-wake | `verl/workers/fsdp_workers.py` 等 | 待读 |
| multi-sample / batch construction | `verl/trainer/ppo/` | 待读 |

（AReaL / vLLM / SGLang 等行待补。）

## 9. Serving → Rollout Transfer

三问框架（每个 serving 工作都过一遍）：① serving 里解决什么 bottleneck？② 哪些机制可直接迁移？③ 进入 rollout 后 objective/scheduler semantics/correctness 约束怎么变、是否需要 RL-aware redesign？

初步判断（待逐项深读验证）：
- continuous batching：直接迁移，但 sync RL 下"批内最慢者"支配 → 需 L3 重调度补偿；
- prefix caching：rollout 中 multi-sample 共享 prompt → 收益天然放大；partial rollout 场景与版本管理耦合；
- speculative decoding：**不可原样迁移**——draft 分布随 policy 更新而漂移（staleness），drafter 需复用历史轨迹/旧版本策略（这正是 5.3 整个簇的动机）；
- chunked prefill：rollout 的 prompt 占比与 serving 不同（reasoning 长 decode），需重估；
- prefill/decode 分离（DistServe 系）：rollout 下的对应物是 train/rollout 分离之争（L4），分离粒度和资源配比是开放问题。

## 10. Workload-specific Findings

（空。Reasoning / Agentic / VLM 三分——Agentic 已有专门簇（5.4），注意 environment bubble 的表征工作。）

## 11. Cross-layer Findings

（空。登记跨层耦合反例的地方：如 BubbleSpec 同时是 L1 机制 + L3 气泡利用。）

## 12. Evaluation & Benchmark

（空。观察目标：各论文用什么报性能——makespan? tokens/s? step time? 是否报 bubble ratio？长度分布是否披露？——最后汇总成"rollout 性能评测方法学"小节，服务 survey。）

## 13. Research Gaps / Ideas

（Round 1 初步观察，待 corpus 收敛后修正）：
1. **C 类（改分布）工作在增多**（DARTS、WAR、Rollout Informativeness、ExpThink）但普遍缺乏"对训练分布/收敛的影响"的严格评估——evaluation gap；
2. speculative-for-RL 簇已相当密集（10+ 篇）——饱和信号：新增量在于统一性能模型（acceptance rate × staleness × 长尾交互），而非又一个 drafter 方案；
3. cap/长度类边界条件的系统化处理（如 cap=1 的 -0 陷阱、偶奇 cap 分配）在现有论文中未见讨论——来自 verl #8281 实战的微观缺口；
4. VLM post-training 的跨阶段 GPU 空分（Rollplex）只有单篇——是否空白？

## 14. Reproduction Candidates

（空。登记模板：为什么值得复现 / repo / model / hardware / dataset / expected cost / key metric / stopping condition。**当前不启动 GPU 实验**。已有可借力资源：M5 4090-1 verl-e2e 环境、本地 veRL 源码。）

## 15. Open Questions

1. "Rollout ≠ serving" 假设的边界：哪些 serving 优化在 rollout 下 objective 完全不变？（找反例）
2. makespan 最优 vs on-policy 性的权衡：C 类改分布的工作，train-time 收益能否抵消分布偏移的收敛代价？有没有论文量化过？
3. taxonomy 是否需要第五层（environment/sandbox 层）？agentic 簇的 environment bubble 是否够独立成层？
4. Rollplex 的 VLM 跨阶段共享是否适用于文本-only 场景（colocation 的空间共享新维度）？

## 16. Changelog / Research Log

- **2026-10-27（Round 1）**：骨架建立；种子注册 + 2 批 arXiv 验证检索（40 篇实证 + 15+ 新发现：SRT/ObsSpec/S³R/NgramRL/DARTS/ASymPO/MARLaaS/AsyncWebRL/AcceRL/Stage Fusion/FlashNext-RL/AREAL-DTA 等）；corpus ≈73（目标 100–150）。待验证种子 8 个。未做：深读、代码地图扩展、corpus 收敛检索（SGLang/vLLM 博客、ICML/NeurIPS/OSDI/SOSP/NSDI/ATC/EuroSys 2025-2026 proceedings 扫描、citation graph 前向/后向扩展）。
