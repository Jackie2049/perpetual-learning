# Rollout Performance Research — LLM Post-training / RL Rollout 性能优化调研

> 本文件是本专题唯一的 canonical knowledge base。任何会话/Agent 接手时**只读本文件**即应在 10–20 分钟内恢复研究状态；详细证据存放于链接指向的材料，但恢复研究状态不依赖它们。
> 远景（非当前任务）：systems-oriented survey《Efficient Rollout Systems for LLM Post-Training》。**当前阶段禁止**撰写正文/Introduction/Related Work、冻结 taxonomy、批量生成低价值摘要、启动大规模复现、在外部主仓库发 issue/PR/comment。

## 0. Project Status

- **阶段**：Phase 1 — recall-first literature & code investigation（广扫建库，不收束观点）
- **最近更新**：2026-10-27（M1 基础补核与rollout补库：补核9篇既有serving文献，新增APRIL／RLHFSpec两篇，详见§5.8）
- **覆盖**：累计**15篇已核身份**（11篇serving基础＋4篇rollout/RLHF，均非深读）；本批只新增2篇，FlashAttention／FA2从同一行拆成两篇不算新发现。Round 1“≈73”仍是待去重审计的历史估计，不能与15相加；其余历史条目继续核验。
- **深读 / 代码阅读**：0 / 0
- **最高优先级**：补来源与版本别名、按唯一文献去重后继续广扫至100–150；A Core短名单不以未核身份的条目数为依据
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

字段约定：完整 24 字段在深读时逐项补全（§6）；矩阵阶段记录 `arXiv ID | 日期 | 标题 | 层 | 类型 | 一句话机制 | 置信度`。置信度：**核**=本批已核primary元数据与版本（非深读／性能验收）；✅=Round 1历史检索标记，尚未逐条完成本批的标题—ID—版本审计；◇=知识注册待核；？=种子待验证。文献标题别名不重复计数，层／类型仍为可修订工作标签。

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
| [2406.14088](https://arxiv.org/abs/2406.14088v2) | 2024-06 | ReaL（v1名ReaLHF） | L4 | B | parameter reallocation与执行计划搜索；v2标MLSys'25，不再标SOSP'24 | 核 |
| [2409.13221](https://arxiv.org/abs/2409.13221v3) | 2024-09 | RLHFuse／Stage Fusion | L4 | B | 样本级跨阶段与微批级阶段内融合；同一文献，NSDI'25 | 核 |
| 2405.11143 | 2024-05 | OpenRLHF | L4 | B | 主流开源 RLHF 框架（Ray 3.0 重构） | ✅ |
| 2308.01320 | 2023-08 | DeepSpeed-Chat | L4 | B | 早期 RLHF 训练系统 | ✅ |
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
| [2509.18521v3](https://arxiv.org/abs/2509.18521v3) | 2025-09-23首次／09-26修订 | APRIL | L3/L4（暂） | B/**C?** | 超额发起，达到目标完成数后结束本步、未完成响应跨步续跑；训练集合／版本影响待深读 | 核 |
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
| [2512.04752v2](https://arxiv.org/abs/2512.04752v2) | 2025-12-04首次／12-12修订 | RLHFSpec | L1/L3/L4（暂） | A/B（暂） | 依验证成本与接受token数选draft策略，并作样本重分配／迁移；摘要不等采样正确性证明 | 核 |
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

| ID/出处 | 首次提交／采用版本 | 标题（短） | 迁移价值与待核条件 | 置信 |
|---|---|---|---|---|
| [USENIX OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu) | 2022-07会议记录 | Orca | iteration-level scheduling及selective batching；本行用会议记录，不沿用错配arXiv ID | 核 |
| [2309.06180v1](https://arxiv.org/abs/2309.06180v1) | 2023-09-12／v1 | vLLM / PagedAttention | 分页KV及共享；并发收益仍取决于rollout布局／驻留，不等减少每步最长响应 | 核 |
| [2312.07104v2](https://arxiv.org/abs/2312.07104v2) | 2023-12-12／v2 2024-06-06 | SGLang / RadixAttention | 结构化程序与KV复用；迁到rollout仍需核工作负载与版本约束 | 核 |
| [2205.14135v2](https://arxiv.org/abs/2205.14135v2) | 2022-05-27／v2 2022-06-23 | FlashAttention | IO-aware精确attention；kernel加速不等整条rollout makespan收益 | 核 |
| [2307.08691v1](https://arxiv.org/abs/2307.08691v1) | 2023-07-17／v1 | FlashAttention-2 | 改work partitioning与并行度；与FA1分开计数，不当RL-native论文 | 核 |
| [2501.01005v2](https://arxiv.org/abs/2501.01005v2) | 2025-01-02／v2 2025-04-21 | FlashInfer | 面向不同KV布局、可定制attention与负载均衡；整引擎收益另核 | 核 |
| [2403.02310v3](https://arxiv.org/abs/2403.02310v3) | 2024-03-04／v3 2024-06-17 | Sarathi-Serve | chunked prefill与stall-free混批；需按真实prefill/decode构成重评，不假定rollout prompt占比高 | 核 |
| [2401.09670v3](https://arxiv.org/abs/2401.09670v3) | 2024-01-18／v3 2024-06-06 | DistServe | prefill/decode分离、TTFT／TPOT约束goodput；不是trainer/rollout分离的同义词 | 核 |
| [2407.00079v5](https://arxiv.org/abs/2407.00079v5) | 2024-06-24／v5 2026-10-08 | Mooncake | KV中心化PD分离及过载拒绝；rollout不能无条件继承请求丢弃语义 | 核 |
| [2401.10774v3](https://arxiv.org/abs/2401.10774v3) | 2024-01-19／v3 2024-06-14 | Medusa | 多预测头与树状验证；backbone是否冻结及接受规则不可混用 | 核 |
| [2401.15077v3](https://arxiv.org/abs/2401.15077v3) | 2024-01-26／v3 2025-03-04 | EAGLE | feature-level草稿预测；只是原始EAGLE，不混作EAGLE-2／3或RL drafter适配 | 核 |

### 5.6 待验证种子（用户清单中尚未在检索中命中）

？Seer ｜ ？Weave ｜ ？FastGRPO ｜ ？SpecRoll ｜ ？MTP-RL（2606.12370 为方向候选）｜ ？"Taming the Long-Tail / adaptive drafter" ｜ ？Online Draft Co-Training for long-context RL

### 5.7 基础文献身份与版本勘误

2026-10-27核对4篇基础文献的元数据、摘要及版本历史，不计为深读，不增加独立论文数。原三个错配ID仍保留在下表供追溯，但不纳入本专题文献计数。

| 条目 | 原记录的问题 | 核验与当前身份 |
|---|---|---|
| SGLang | 2310.07204实际是[视觉计算扩散模型综述](https://arxiv.org/abs/2310.07204)，不是SGLang | [2312.07104v2](https://arxiv.org/abs/2312.07104v2)，v1提交2023-12-12，v2修订2024-06-06 |
| Orca | 2205.12318实际是[ColdGuess图网络论文](https://arxiv.org/abs/2205.12318)，不是Orca | [OSDI'22官方记录](https://www.usenix.org/conference/osdi22/presentation/yu)，2022-07、521–538页；不猜另一个arXiv ID |
| RLHFuse | 2412.15179实际是[概率论中的Beat the Average问题](https://arxiv.org/abs/2412.15179)；矩阵又把Stage Fusion列成第二篇 | [2409.13221v1](https://arxiv.org/abs/2409.13221v1)名RLHFuse，2024-09-20提交；[v3](https://arxiv.org/abs/2409.13221v3)与[NSDI'25官方记录](https://www.usenix.org/conference/nsdi25/presentation/zhong)采用Stage Fusion题名，保留一个条目及两个别名 |
| ReaLHF | ID正确，SOSP'24标签不符修订版接收信息；忽略改名易重复注册 | [2406.14088v1](https://arxiv.org/abs/2406.14088v1)名ReaLHF，2024-06-20提交；[v2](https://arxiv.org/abs/2406.14088v2)名ReaL，2025-04-24修订且标MLSys'25 |

核验原则：先以原站标题／作者／摘要确认身份，再登记首次提交日期、采用版本、venue及别名。检索结果可能对应旧标题，最新摘要也可能有不同实验数字；不能把不同版本的名称或数字拼成一条证据。此首批完成4篇；后续累计见§0与§5.8。Round 1其他✅仍待身份审计，既不认定全部错误，也不继续称全库已验证。

### 5.8 Serving基础补核与rollout原生补库

2026-10-27第二批：§5.5剩余9篇的标题／ID全部匹配，补齐primary链接、首次提交与采用版本。FA1／FA2原已共列，拆行仅为区分版本和唯一文献，不是新发现；当前serving基础11篇，加首批ReaL／RLHFuse及本批新增两篇，累计15篇身份已核。仍未审计全库或重算Round 1的≈73，不把日期编号前缀直接当提交日。

新增[APRIL v3](https://arxiv.org/abs/2509.18521v3)和[RLHFSpec v2](https://arxiv.org/abs/2512.04752v2)通过本主文档标题／ID查重后进入矩阵。前者的超额请求、完成配额及跨步续跑使“没有丢弃响应”与“每步样本集合／策略版本是否不变”必须分开；B/C标签暂留问题，不凭摘要判定有偏或无偏。后者把draft策略选择与样本迁移并用，先标A/B；不把sample reallocation自动理解成删样本，也不沿用其摘要中的“首次”自述作历史优先权结论。

版本例子：[Mooncake v1](https://arxiv.org/abs/2407.00079v1)题为Kimi's KVCache-centric Architecture，[v5](https://arxiv.org/abs/2407.00079v5)题为A KVCache-centric Disaggregated Architecture，均属同一ID；不新增一篇或把当前软件行为当成各版本论文结论。上述工作只核元数据、摘要与历史版本；没有全文深读、性能数字抽取、原作者测试或实现审计。下一批优先rollout-native候选的身份／来源及citation扩展，不反复核这11篇。

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
- speculative decoding：不能由serving加速比直接推导RL收益；policy变化下draft适配、接受率与成本需重评。原先“不可原样迁移”的绝对判断尚未成立，采样正确性与draft效率分开，具体机制是否可直接复用待深读；
- chunked prefill：rollout 的 prompt 占比与 serving 不同（reasoning 长 decode），需重估；
- prefill/decode分离（DistServe系）：分开的是推理内部两个阶段，不能直接等同trainer/rollout分离；进入RL后的目标、分离粒度与资源配比仍待研究。

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

- **2026-10-27（M1基础补核与rollout补库）**：核既有9篇serving原始文献，补primary链接／首次提交／采用版本，FA1/2拆行不计新发现；新增APRIL 2509.18521v3与RLHFSpec 2512.04752v2，累计15篇身份已核（11 serving＋4 rollout/RLHF）。两新项层／类型暂定，未抽性能数字；修正PD分离与trainer/rollout分离混同，撤回speculative不可直接迁移的先验绝对说法。未做全文深读、taxonomy收束、代码或复现；下一批转rollout-native来源核验与扩展。

- **2026-10-27（M1身份核验批次）**：承接新专项，核4篇而非新增4篇；纠正SGLang／Orca／RLHFuse三处错配ID，合并RLHFuse与Stage Fusion重复项，登记ReaLHF→ReaL及MLSys'25版本信息。来源为arXiv原始／修订元数据与USENIX官网；撤回“≈73”为已去重已核数量的解释，保留Round 1历史记录。未做深读／性能判断、taxonomy收束、代码改动或复现；下一步按§5.7补其余来源与身份。

- **2026-10-27（Round 1）**：骨架建立；种子注册 + 2 批 arXiv 验证检索（40 篇实证 + 15+ 新发现：SRT/ObsSpec/S³R/NgramRL/DARTS/ASymPO/MARLaaS/AsyncWebRL/AcceRL/Stage Fusion/FlashNext-RL/AREAL-DTA 等）；corpus ≈73（目标 100–150）。待验证种子 8 个。未做：深读、代码地图扩展、corpus 收敛检索（SGLang/vLLM 博客、ICML/NeurIPS/OSDI/SOSP/NSDI/ATC/EuroSys 2025-2026 proceedings 扫描、citation graph 前向/后向扩展）。
