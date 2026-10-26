# Rollout Performance Research — LLM Post-training / RL Rollout 性能优化调研

> 本文件是本专题唯一的 canonical knowledge base。任何会话/Agent 接手时**只读本文件**即应在 10–20 分钟内恢复研究状态；详细证据存放于链接指向的材料，但恢复研究状态不依赖它们。
> 远景（非当前任务）：systems-oriented survey《Efficient Rollout Systems for LLM Post-Training》。**当前阶段禁止**撰写正文/Introduction/Related Work、冻结 taxonomy、批量生成低价值摘要、启动大规模复现、在外部主仓库发 issue/PR/comment。

## 0. Project Status

- **阶段**：Phase 1 — recall-first literature & code investigation（广扫建库，不收束观点）
- **最近更新**：2026-10-27（M1剩余历史条目补核：11篇既有文献，新增3篇训推失配文献，详见§5.13）
- **覆盖**：累计**77篇已核身份**（16篇serving基础／迁移边界＋61篇rollout/RLHF及相关边界，含VLA、训练侧与算法诊断；均非深读；75个唯一arXiv ID＋Orca／PUZZLE会议记录）。本批核既有11篇、仅新增3篇；ASymPO虽计入身份清单，但证明错误警示单列，不作为可靠理论依据。现有矩阵的历史✅已补核完，不代表种子已落实、全部为rollout-native或检索已收敛。Round 1“≈73”是历史估计，不能与77相加。
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
| 正确性约束 | 输出质量 | policy版本、同权重训推数值差异及old logp语义分别核；同版本不足以保证概率一致 |

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

按实际改变的训练轨迹集合判断C，不按技术名称判断：partial rollout若仅提供路由信号，或历史轨迹仅供draft验证，不自动归C。跨训练步重排、短组先更新等对每步集合与更新顺序的影响另核，不能由“不丢样本”排除。

反例登记：当前无。凡 taxonomy 解释不了的工作记入 §15 并提修订。

## 5. Literature Matrix

字段约定：完整 24 字段在深读时逐项补全（§6）；矩阵阶段记录 `arXiv ID | 日期 | 标题 | 层 | 类型 | 一句话机制 | 置信度`。置信度：**核**=本批已核primary元数据与版本（非深读／性能验收）；✅=Round 1历史检索标记，尚未逐条完成本批的标题—ID—版本审计；◇=知识注册待核；？=种子待验证。文献标题别名不重复计数，层／类型仍为可修订工作标签。

### 5.1 L4 — RL Pipeline / Async / Disaggregation

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2505.24298v5](https://arxiv.org/abs/2505.24298v5) | 2025-05-30首次／v5 2026-03-02 | AReaL | L4（暂） | B（系统；算法另核） | 分离生成／训练、均衡负载控制staleness；另有staleness-enhanced PPO | 核 |
| [2409.19256v2](https://arxiv.org/abs/2409.19256v2) | 2024-09-28首次／v2 10-02 | HybridFlow / veRL | L4 | B | 混合单／多controller与分层API，3D-HybridEngine用于训练和生成间actor resharding | 核 |
| [2507.13833v4](https://arxiv.org/abs/2507.13833v4) | 2025-07-18首次／v4 2026-05-30 | DistFlow | L4 | B | 多controller，数据传输与控制派发解耦，去中心Data Coordinator及DAG任务调度 | 核 |
| [2504.15930v1](https://arxiv.org/abs/2504.15930v1) | 2025-04-22／v1 | StreamRL | L3/L4（暂） | B | 流式生成处理阶段气泡，长度ranker与调度处理长尾气泡；同步与异步路径须分开 | 核 |
| [2505.24034v2](https://arxiv.org/abs/2505.24034v2) | 2025-05-29首次／v2 06-02 | LlamaRL | L4 | B | native PyTorch单controller，异步off-policy、模型offload及DMA权重同步 | 核 |
| [2510.12633v1](https://arxiv.org/abs/2510.12633v1) | 2025-10-14／v1 | Laminar | L3/L4（暂） | B（暂） | trajectory级异步，relay workers服务细粒度权重拉取，repack集中长尾 | 核 |
| [2604.26256v2](https://arxiv.org/abs/2604.26256v2) | 2026-04-29首次／v2 07-20 | DORA | L4（暂） | B（暂） | 多版本streaming rollout，约束轨迹内policy一致性、数据完整性及有界staleness | 核 |
| [2601.12784v2](https://arxiv.org/abs/2601.12784v2) | 2026-01-19首次／v2 08-03 | StaleFlow | L4（暂） | B（暂） | global consistency跟踪轨迹生命周期，trajectory／parameter server协调陈旧性与长度偏斜 | 核 |
| [2507.01663v2](https://arxiv.org/abs/2507.01663v2) | 2025-07-02首次／v2 2026-09-11 | AsyncFlow | L4 | B | 分布式流式数据管理与producer-consumer，staleness阈值内延后参数更新 | 核 |
| [2606.26997v2](https://arxiv.org/abs/2606.26997v2) | 2026-06-25首次／v2 07-05 | RolloutPipe | L3/L4（暂） | B（暂） | fixed-weight rollout中完整group FIFO送训，frontier-group准入提早供给下一批 | 核 |
| [2410.18252v3](https://arxiv.org/abs/2410.18252v3) | 2024-10-23首次／v3 2025-04-26 | Asynchronous RLHF | L4 | B | 生成与旧样本训练重叠，研究off-policy容忍度；算法鲁棒性不自动推广 | 核 |
| [2406.14088](https://arxiv.org/abs/2406.14088v2) | 2024-06 | ReaL（v1名ReaLHF） | L4 | B | parameter reallocation与执行计划搜索；v2标MLSys'25，不再标SOSP'24 | 核 |
| [2409.13221](https://arxiv.org/abs/2409.13221v3) | 2024-09 | RLHFuse／Stage Fusion | L4 | B | 样本级跨阶段与微批级阶段内融合；同一文献，NSDI'25 | 核 |
| [2405.11143v6](https://arxiv.org/abs/2405.11143v6) | 2024-05-20首次／v6 2025-10-09 | OpenRLHF | L4 | B | 基于Ray／vLLM／DeepSpeed／Transformers的RLHF／RLVR框架 | 核 |
| [2308.01320v1](https://arxiv.org/abs/2308.01320v1) | 2023-08-02／v1 | DeepSpeed-Chat | L4 | B | 统一训练／推理优化的端到端RLHF系统；后续框架基线须核具体实现与配置 | 核 |
| [2506.06122v1](https://arxiv.org/abs/2506.06122v1) | 2025-06-06／v1 | ROLL | L3/L4（暂） | B（暂） | 单controller／parallel worker，逐样本rollout生命周期与AutoDeviceMapping | 核 |
| [USENIX ATC'24](https://www.usenix.org/conference/atc24/presentation/lei) | 2024-07会议记录 | PUZZLE | L4（暂） | B（暂） | 阶段内模型亲和／时分重叠，阶段间按相似性降低切换通信成本 | 核 |
| [2605.12070v3](https://arxiv.org/abs/2605.12070v3) | 2026-05-12首次／v3 09-29 | Missing Old Logits in Asynchronous Agentic RL | L4（暂） | 系统／算法边界 | 区分同版本训推差异与policy staleness，比较精确old logits获取及近似修正 | 核 |
| [2606.03070v3](https://arxiv.org/abs/2606.03070v3) | 2026-06-02首次／v3 06-23 | ASymPO（证明错误警示） | L4边界（暂） | 算法边界，不硬归B/C | current-policy-only的非对称loss尺度归一化；页面注明证明有误、v2曾撤回，见§5.13 | 核 |
| [2602.01826v1](https://arxiv.org/abs/2602.01826v1) | 2026-02-02／v1 | Beyond Precision | L1/L4边界（暂） | 算法诊断，不硬归B/C | 以响应长度信号触发LR衰减，研究优化动态与训推失配；不是系统吞吐优化 | 核 |
| [2510.26788v1](https://arxiv.org/abs/2510.26788v1) | 2025-10-30／v1 | Defeating the Training-Inference Mismatch via FP16 | L1/L4边界（暂） | 数值诊断，类型待核 | 对比BF16／FP16训推一致性；摘要结论限作者条件，不推广为统一精度建议 | 核 |
| [2605.14220v2](https://arxiv.org/abs/2605.14220v2) | 2026-05-14首次／v2 09-29 | Diagnosing Training Inference Mismatch（VeXact） | L1/L4边界（暂） | 诊断，不硬归A/B/C | 同权重不同引擎的概率差异，以zero-mismatch reference隔离TIM与off-policy drift | 核 |
| [2609.32444v1](https://arxiv.org/abs/2609.32444v1) | 2026-09-26／v1 | Rethinking Training-Inference Mismatch（CIS） | L1/L4边界（暂） | 算法修正，不硬归B/C | logit-displacement刻画与confidence-aware截断importance sampling；偏差／稳定性待深读 | 核 |

### 5.2 L3 — Long-tail / Scheduling / Cluster

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2509.21009v1](https://arxiv.org/abs/2509.21009v1) | 2025-09-25／v1 | RollPacker | L3/L4（暂） | B（暂） | tail batching把长尾集中到少数long rounds，配合rollout／reward／训练优化；批构成待核 | 核 |
| [2603.23414v1](https://arxiv.org/abs/2603.23414v1) | 2026-03-24／v1 | SortedRL | L3/L4（暂） | B/**C?** | 按输出长度重排、短组先更新，cache机制控制off-policy程度；micro-curriculum语义待核 | 核 |
| [2608.22788v2](https://arxiv.org/abs/2608.22788v2) | 2026-08-24首次／v2 08-26 | TailSieve | L1/L3（暂） | A/B（暂） | partial信号识别尾组、分池路由／副本分配，低并发池可叠推测；当前policy重新生成 | 核 |
| [2607.17299v1](https://arxiv.org/abs/2607.17299v1) | 2026-07-19／v1 | WAR | L1/L3（暂） | A/B（暂） | 低负载SuffixDecoding、高负载cache-aware placement；不由workload-aware推导改样本 | 核 |
| [2605.30859v1](https://arxiv.org/abs/2605.30859v1) | 2026-05-29／v1 | DARTS | L3（暂） | **C** | 从每prompt冗余探索空间选轨迹、适配冗余分配，主动塑造分布 | 核 |
| [2509.18521v3](https://arxiv.org/abs/2509.18521v3) | 2025-09-23首次／09-26修订 | APRIL | L3/L4（暂） | B/**C?** | 超额发起，达到目标完成数后结束本步、未完成响应跨步续跑；训练集合／版本影响待深读 | 核 |
| [2511.14617v3](https://arxiv.org/abs/2511.14617v3) | 2025-11-18首次／v3 2026-04-03 | Seer | L1/L3（暂） | A/B（暂） | divided rollout、context-aware调度与grouped speculation；不是仅长度预测 | 核 |
| ？ | ? | Weave | L3/L4 | B | ？（种子，待验证） | ？ |

### 5.3 L1/L2 — RL-aware Speculative Rollout（serving机制的迁移条件）

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2509.23232v3](https://arxiv.org/abs/2509.23232v3) | 2025-09-27首次／v3 2026-01-12 | SPEC-RL | L1/L2（暂） | A（暂） | 历史轨迹段作speculative prefix并draft-and-verify，不等直接回放作训练样本 | 核 |
| [2508.18588v1](https://arxiv.org/abs/2508.18588v1) | 2025-08-26／v1 | RhymeRL（History Rhymes） | L1/L2/L3（暂） | A/B（暂） | HistoSpec用历史token作draft，HistoPipe用历史分布作两级调度 | 核 |
| [2510.26475v1](https://arxiv.org/abs/2510.26475v1) | 2025-10-30／v1 | ReSpec | L1/L2（暂） | A（暂） | 动态SD配置、蒸馏演进drafter、reward加权update；加权对象待读正文 | 核 |
| [2511.13841v1](https://arxiv.org/abs/2511.13841v1) | 2025-11-17／v1 | DAS（Beat the long tail） | L1/L3（暂） | A/B（暂） | 历史rollout suffix tree作非参数draft，长轨迹分配更激进draft预算 | 核 |
| [2511.16665v3](https://arxiv.org/abs/2511.16665v3) | 2025-11-20首次／v3 2026-03-20 | TLT（Taming the Long-Tail） | L1/L2/L3（暂） | A/B（暂） | 利用长尾期空闲GPU持续训drafter，预捕获CUDAGraph池按batch适配策略 | 核 |
| [2512.04752v2](https://arxiv.org/abs/2512.04752v2) | 2025-12-04首次／12-12修订 | RLHFSpec | L1/L3/L4（暂） | A/B（暂） | 依验证成本与接受token数选draft策略，并作样本重分配／迁移；摘要不等采样正确性证明 | 核 |
| [2605.08862v1](https://arxiv.org/abs/2605.08862v1) | 2026-05-09／v1 | BubbleSpec | L1/L3（暂） | A/B（暂） | 空闲rank预生成后续step的draft，交后续推测验证；不等直接用旧policy样本训练 | 核 |
| [2606.18967v2](https://arxiv.org/abs/2606.18967v2) | 2026-06-17首次／v2 09-30 | EfficientRollout | L1/L2（暂） | A（暂） | 从target诱导量化drafter，按系统regime切SD开关、按接受情况调draft长度 | 核 |
| [2601.09083v1](https://arxiv.org/abs/2601.09083v1) | 2026-01-14／v1 | SRT | L1/L3（暂） | A/B（暂） | 每prompt历史续写树作draft，在线更新并在空闲气泡run-ahead生成 | 核 |
| [2610.13202v1](https://arxiv.org/abs/2610.13202v1) | 2026-10-09／v1 | S³R | L1/L2（暂） | A/B（暂） | 同参数稀疏attention作draft、full-attention验证与KV调度；非直接稀疏采样 | 核 |
| [2610.32881v1](https://arxiv.org/abs/2610.32881v1) | 2026-10-23／v1 | NgramRL | L1/L2（暂） | A（暂） | 有界近期历史构造每prompt n-gram，按接受长度／续写token预测lookup分支并预取 | 核 |
| [2606.12370v1](https://arxiv.org/abs/2606.12370v1) | 2026-06-10／v1 | Bebop（Breaking Entropy Bounds） | L1（暂） | A（暂） | MTP＋概率拒绝采样、预RL e2e TV训练；在线MTP更新非默认必要 | 核 |
| [2509.21792v1](https://arxiv.org/abs/2509.21792v1) | 2025-09-26／v1 | FastGRPO | L1/L2（暂） | A（暂） | 并发感知draft／验证配置与online draft learning | 核 |
| [2604.26779v1](https://arxiv.org/abs/2604.26779v1) | 2026-04-29／v1 | System-Integrated SD（Iso等） | L1/L4（暂） | A/B（暂） | NeMo-RL＋vLLM集成MTP／外部draft／EAGLE3；实测与模拟预测分开 | 核 |
| ？ | ? | SpecRoll / MTP-RL | L1 | A | 种子待验证；MTP-RL仍只登记方向候选Bebop | ？ |

### 5.4 Agentic Rollout

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2603.18815v1](https://arxiv.org/abs/2603.18815v1) | 2026-03-19／v1 | ProRL Agent | L4（暂） | B（暂） | API服务承载完整rollout生命周期，rootless HPC sandbox；服务接口不等故障恢复保证 | 核 |
| [2605.15565v1](https://arxiv.org/abs/2605.15565v1) | 2026-05-15／v1 | AstraFlow | L4（暂） | B（暂） | rollout服务／dataflow／training自主组件解耦，支持multi-policy与异构弹性执行 | 核 |
| [2512.22560v2](https://arxiv.org/abs/2512.22560v2) | 2025-12-27首次／v2 2026-06-15 | RollArt（v1摘要名RollArc） | L2/L4（暂） | B（暂） | 按阶段硬件亲和映射、轨迹级解耦、serverless reward与有界staleness权重同步 | 核 |
| [2603.28101v1](https://arxiv.org/abs/2603.28101v1) | 2026-03-30／v1 | Heddle | L3/L4（暂） | B（暂） | 轨迹级预测／优先级，tool等待窗口迁移与自适应模型并行度 | 核 |
| [2511.16108v1](https://arxiv.org/abs/2511.16108v1) | 2025-11-20／v1 | SkyRL-Agent | L4（暂） | B／**C?**（recipe） | 异步pipeline dispatcher与AST搜索工具recipe并用，系统／工具收益须分开 | 核 |
| [2510.05943v1](https://arxiv.org/abs/2510.05943v1) | 2025-10-07／v1 | EARL | L4（暂） | B（暂） | 按序列长度／负载选跨阶段并行策略，layout-aware去中心data dispatcher | 核 |
| [2610.15246v1](https://arxiv.org/abs/2610.15246v1) | 2026-10-11／v1 | ObsSpec | L4（暂） | A/B（暂） | world model预测tool observation并与真实执行重叠，预测／真实完全匹配才保留推测工作 | 核 |
| [2605.05262v1](https://arxiv.org/abs/2605.05262v1) | 2026-05-06／v1 | InfoTree（Rollout Informativeness） | L3/L4（暂） | **C**／B（暂） | UUCB中间状态选择、学习式预算分配及异步Speculative Expansion；不是仅复用prefix | 核 |
| [2606.05597v3](https://arxiv.org/abs/2606.05597v3) | 2026-06-04首次／v3 08-07 | AsyncWebRL | L4（暂） | B／**C?**（算法边界） | 异步rollout／update／refresh＋常驻pool／轻量截图，另改GRPO轨迹归一化 | 核 |
| [2603.18464v3](https://arxiv.org/abs/2603.18464v3) | 2026-03-19首次／v3 06-12 | AcceRL（VLA边界） | L4（暂） | B／**C?**（world model） | 环境rollout／模型推理／更新隔离；world model样本效率与系统吞吐分开 | 核 |
| [2605.08527v1](https://arxiv.org/abs/2605.08527v1) | 2026-05-08／v1 | MARLaaS | L4（暂） | B（暂） | 多租户共享base＋LoRA，环境／rollout／训练独立事件调度；非multi-agent同义词 | 核 |
| [2608.14498v1](https://arxiv.org/abs/2608.14498v1) | 2026-08-14／v1 | Rollplex | L2/L4（暂） | B（暂） | rollout decode窗口并行做reference／training prefix，生命周期HBM与跨TP兼容权重共享 | 核 |
| [2602.09578v1](https://arxiv.org/abs/2602.09578v1) | 2026-02-10／v1 | FlexMARL | L3/L4（暂） | B（暂） | 多agent experience store驱动micro-batch异步pipeline、分层均衡与agent资源绑定 | 核 |
| [2508.03680v1](https://arxiv.org/abs/2508.03680v1) | 2025-08-05／v1 | Agent Lightning（2025） | L4（暂） | B（系统；算法另核） | Training-Agent Disaggregation与统一transition接口，含LightningRL credit assignment | 核 |
| [2608.17528v1](https://arxiv.org/abs/2608.17528v1) | 2026-08-18／v1 | Agent Lightning v1.0（2026） | L4（暂） | B（系统；算法另核） | deploy-time harness拥有环境交互loop，trainer接收LLM请求／响应；合样本及loss语义待核 | 核 |
| [2602.00482v2](https://arxiv.org/abs/2602.00482v2) | 2026-01-31首次／v2 06-13 | AReaL-DTA（训练侧边界） | L1/L3（训练侧；暂） | A/B（训练侧；暂） | DFS遍历共享prefix树做前后向，分布式tree batching；不是rollout decode tree验证 | 核 |
| [2610.28998v1](https://arxiv.org/abs/2610.28998v1) | 2026-10-20／v1 | FlashNext-RL | L1/L4（暂） | A/B（暂） | prime-rl＋vLLM，模型特定offload／累积及kernel优化；训练tokens/s与RL迭代分开 | 核 |

### 5.5 Serving 基础（supporting corpus，含迁移边界，不标 rollout-native）

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
| [2411.04975v3](https://arxiv.org/abs/2411.04975v3) | 2024-11-07／v3 2025-10-07 | SuffixDecoding | suffix tree缓存prompt／历史输出，适配推测长度；RL中的跨步／跨worker复用另核 | 核 |
| [2211.17192v2](https://arxiv.org/abs/2211.17192v2) | 2022-11-30／v2 2023-05-18 | Fast Inference from Transformers（Leviathan等） | 近似模型draft与target验证／修正采样；输出分布正确性须核实现前提 | 核 |
| [2302.01318v1](https://arxiv.org/abs/2302.01318v1) | 2023-02-02／v1 | Speculative Sampling（Chen等） | 修改拒绝采样以保target分布，原文带hardware numerics条件；非bitwise相同承诺 | 核 |
| [2602.06932v5](https://arxiv.org/abs/2602.06932v5) | 2026-02-06／v5 2026-08-04 | Aurora（When RL Meets Adaptive Speculative Training） | 由live inference traces在线RL训练speculator、热换更新；不是post-training policy rollout系统 | 核 |
| [2408.13510v2](https://arxiv.org/abs/2408.13510v2) | 2024-08-24首次／v2 2025-01-07 | Intelligent Router | response-length预测与混合负载估计，RL路由跨推理实例；是serving调度，不是post-training rollout系统 | 核 |

### 5.6 待验证种子（用户清单中尚未在检索中命中）

？Weave ｜ ？SpecRoll ｜ ？MTP-RL（2606.12370／Bebop为方向候选，未确认同名）｜ ？Online Draft Co-Training for long-context RL

Seer与Taming the Long-Tail（TLT）已在§5.2／§5.3落实来源。Weave本批定向检索未命中，仍是种子，不据此判定不存在。

FastGRPO已落实2509.21792v1；不据online draft learning关键词把它等同其他未确认同名的种子。

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

### 5.9 RL系统身份补核与引用扩展

2026-10-27第三批：HybridFlow、DistFlow、StreamRL、LlamaRL、AsyncFlow、Asynchronous RLHF、OpenRLHF、DeepSpeed-Chat共8篇的标题与原ID匹配，§5.1补齐首次提交与采用版本。累计26篇身份已核；这8篇不是新增发现。层／类型仅为建库标签，StreamRL同时涉及长尾调度与阶段重叠，暂补L3，不据此冻结分类。

三项机制表述收窄：[DistFlow v4](https://arxiv.org/abs/2507.13833v4)解耦的是数据传输和控制派发，不只笼统写controller-worker；[AsyncFlow v2](https://arxiv.org/abs/2507.01663v2)摘要支持流式管理、producer-consumer和staleness阈值内延后更新，原“参数shuffle式权重同步”撤回，具体同步实现待读代码。[OpenRLHF v1](https://arxiv.org/abs/2405.11143v1)与[v6](https://arxiv.org/abs/2405.11143v6)同题但作者／摘要已变，不能把当前软件release、Ray依赖版本或旧版基线结论拼入论文身份。HybridFlow本批只核arXiv版本与机制；ACM关联DOI页面返回403，venue未在本批独立确认。

新增三篇经本主文档标题／ID查重：[ROLL v1](https://arxiv.org/abs/2506.06122v1)补逐样本调度／跨阶段资源映射线索；[EARL v1](https://arxiv.org/abs/2510.05943v1)补长上下文agentic工作负载的并行策略与数据布局线索。两篇来自定向检索，不声称引用关系。[StreamRL v1 §8及参考文献](https://arxiv.org/html/2504.15930v1#S8)提供PUZZLE回溯入口，随后以[ATC'24官方记录](https://www.usenix.org/conference/atc24/presentation/lei)确认完整题名、作者、2024-07与127–140页；只计一篇，不猜arXiv ID。三个条目均暂B类，是否影响样本语义留待正文／实现核对。

本批范围为元数据、摘要，以及StreamRL的相关工作／参考文献定位；不是完整深读或代码阅读，没有记录作者加速比为本项目结果。下一批优先核§5.2长尾与§5.3推测rollout候选，并从已核论文扩展引用，不重复核这8篇系统基础。

### 5.10 长尾调度与推测rollout的身份和类型补核

2026-10-27第四批：RollPacker、SortedRL、TailSieve、WAR、DARTS、SPEC-RL、RhymeRL、ReSpec、DAS共9篇标题与原ID匹配，矩阵补首次提交与采用版本；既有Seer、Taming the Long-Tail两种子分别落实为2511.14617v3、2511.16665v3。累计38篇身份已核，而非新增12篇；只有SuffixDecoding是本批新发现。

分类修正针对机制边界，不作正确性验收：[WAR v1](https://arxiv.org/abs/2607.17299v1)结合低负载推测与高负载cache-aware调度，改暂A/B，不再把它列为已知C例子。[TailSieve v2](https://arxiv.org/abs/2608.22788v2)用partial作为路由信号并称所选prompt在当前policy重生成，不能仅因partial一词推导删样本／改训练分布，改暂A/B；作者的on-policy／steady-state无额外长度偏差声明仍须核假设。[SortedRL v1](https://arxiv.org/abs/2603.23414v1)有短样本组先更新、micro-curriculum及off-policy控制，暂B/C?；[DARTS v1](https://arxiv.org/abs/2605.30859v1)明确主动分布塑形，保留C。RollPacker的跨round重排与训练批构成仍待正文核对，不能把“不丢轨迹”当作每步分布不变的证明。

[RhymeRL v1](https://arxiv.org/abs/2508.18588v1)同时有HistoSpec与HistoPipe，原仅A／L1-L2遗漏调度，改暂A/B并补L3。ReSpec中的reward加权update对象尚未确认，不由摘要短语推断它一定修改actor目标。SPEC-RL、DAS的历史轨迹复用用于draft／验证，不等于将旧样本直接回放训练；标签不证明它们的采样正确性。

引用扩展：[WAR v1参考文献](https://arxiv.org/html/2607.17299v1)同时列出Seer、TLT和SuffixDecoding，逐一返回arXiv原站核元数据。[SuffixDecoding v1](https://arxiv.org/abs/2411.04975v1)题为A Model-Free Approach to Speeding Up Large Language Model Inference，[v3](https://arxiv.org/abs/2411.04975v3)改为Extreme Speculative Decoding for Emerging AI Applications，按同ID计一篇serving基础；不把其agentic inference评测误算agentic RL训练实证。Seer与TLT转已核，不再留在未命中种子列表。

本批仅元数据、摘要及WAR引用定位，没有全文深读、代码或性能抽取。下一批转其余推测／agentic条目身份与引用扩展；不据条目数量判断方向已饱和，也不冻结分类。

### 5.11 推测对象与serving边界补核

2026-10-27第五批：BubbleSpec、EfficientRollout、SRT、S³R、NgramRL、ObsSpec、Breaking Entropy Bounds、Unified Training-Serving Speculative共8个既有条目的ID与文献匹配，补版本／日期；后两项分别补系统名Bebop和Aurora。FastGRPO从旧种子落实为2509.21792v1；从EfficientRollout v2引用扩展新增Leviathan等2211.17192、Chen等2302.01318和Iso等2604.26779三篇。累计50身份=38＋8既有＋1种子＋3新发现，不把系统别名、移行或引用中提到的论文另加一篇。

范围勘误：[Aurora v1](https://arxiv.org/abs/2602.06932v1)及采用的[v5](https://arxiv.org/abs/2602.06932v5)描述的是用live serving反馈在线训练speculator，target服务用户流量；其中的RL不等于post-training policy的rollout加速。移至§5.5作serving／迁移边界，撤回原L1/L4 rollout-native归类，不删除文献。FastGRPO采用arXiv v1；其Comments为Submitted to ICLR 2026，引用记录中的会议标签不替代独立录用核验，OpenReview入口停在浏览器验证页。

推测对象需要区分：[EfficientRollout v2](https://arxiv.org/abs/2606.18967v2)的量化target派生draft与[S³R v1](https://arxiv.org/abs/2610.13202v1)的同参数稀疏draft都不能简称为低精度／稀疏policy直接生成训练响应。S³R还明确dense验证仍需完整历史KV，不由稀疏draft推断整条路径KV常驻下降。[ObsSpec v1](https://arxiv.org/abs/2610.15246v1)推测的是tool observation，不是token草稿：环境真实执行并行进行，完全匹配才保留推测工作；移至agentic簇、暂L4 A/B，动作提交／副作用及不匹配恢复待正文核对，不称跳过工具调用。

BubbleSpec／SRT的run-ahead用于后续draft验证，不等于直接拿旧policy样本训练；NgramRL预取的是可能验证结果对应的CPU lookup分支，不是绕过target验证。Bebop是Breaking Entropy Bounds文中的系统名，MTP-RL仍为未确认同名的方向种子，不强行合并。

引用入口为[EfficientRollout v2参考文献](https://arxiv.org/html/2606.18967v2)：定位后逐一核三篇原站标题／日期／版本。两篇基础论文分别列一条，不因都涉及拒绝采样而合并，也不凭首发日期裁定优先权。当前只核元数据、摘要与引用位置，未深读证明、核实现或实测；§12只登记Iso等的证据类型，不提取加速比作比较。下一批集中agentic簇与其引用，不再重复本批已核推测条目。

### 5.12 Agentic系统版本与评估边界补核

2026-10-27第六批：ProRL Agent、AstraFlow、RollArt、Heddle、SkyRL-Agent、Rollout Informativeness、AsyncWebRL、AcceRL、MARLaaS、Rollplex共10篇标题与原ID匹配，补首次提交及采用版本；Rollout Informativeness补系统名InfoTree。新入FlexMARL及两篇Agent Lightning论文，累计63身份=50＋10既有＋3新发现，不把别名、版本或旧条目补核算作新发现。

版本与沿革：[RollArt v1](https://arxiv.org/abs/2512.22560v1)题为Scaling Agentic RL Training via Disaggregated Infrastructure，摘要使用RollArc系统名；[v2](https://arxiv.org/abs/2512.22560v2)题为Disaggregated Multi-Task Agentic RL Training at Scale，系统统一为ROLLART，同ID只留一条。[Agent Lightning 2025](https://arxiv.org/abs/2508.03680v1)与[Agent Lightning v1.0 2026](https://arxiv.org/abs/2608.17528v1)是不同ID／题名／作者的相关论文，后者明确承接original工作；“v1.0”是题名中的系统版本，不是前一arXiv ID的v2，也不作为两套独立系统实证。

系统与recipe分开：[SkyRL-Agent v1](https://arxiv.org/abs/2511.16108v1)包含pipeline dispatcher和AST搜索工具recipe；[AsyncWebRL v3](https://arxiv.org/abs/2606.05597v3)同时有系统异步与GRPO轨迹归一化修改，不能把联合质量／成本结果全归异步。两者潜在训练语义影响分别暂留C?；normalizer修改不等于直接删选轨迹，具体归类待正文。[InfoTree v1](https://arxiv.org/abs/2605.05262v1)有中间状态选择／预算分配，保留C，并补异步expansion的B线索，不把其理论保证当本项目已验收。[AcceRL v3](https://arxiv.org/abs/2603.18464v3)是VLA场景，保留工作负载边界，world model样本效率与系统吞吐不能互换，也不与ObsSpec的tool observation精确匹配机制混同。

资源与控制边界：[Rollplex v1](https://arxiv.org/abs/2608.14498v1)重叠的是与生成响应独立的prefix计算和rollout decode，并称保持同步on-policy更新；不能见跨阶段并发即判async off-policy。跨TP仅共享layout-compatible物理存储，不是任意layout零拷贝。MARLaaS是多租户共享base／LoRA；新增FlexMARL关注多agent协同训练，二者不因MARL缩写合并。Agent Lightning v1.0把environment loop归harness，ProRL Agent提供生命周期API；均不能只凭接口解耦推断工具副作用、重试或恢复已受保证。

新条目来自定向检索／已见线索后回arXiv原站核身份，未声称Heddle引用它们；本批在[Heddle v1 HTML](https://arxiv.org/html/2603.28101v1)检索Weave／Lightning未命中，不据此判断不存在或无引用。范围限元数据、摘要与关键词／引用定位，无全文深读、代码、性能抽取或复现。下一批核剩余L4及agentic边界条目后继续原生文献扩展；当前仍不冻结分类或写survey正文。

### 5.13 异步语义、训练侧边界与文献可靠性补核

2026-10-27第七批：核剩余11篇历史✅的标题、首次日期、采用版本及摘要，ID均匹配；新增3篇TIM文献来自定向搜索后回arXiv原站核验。累计77身份=63＋11既有＋3新发现（75唯一arXiv＋2会议记录）。Intelligent Router移serving，故分组为16 serving／边界＋61 rollout/RLHF及相关边界；后者不是61篇纯rollout-native。现有矩阵历史✅已清零，种子与未收录线索仍待查，不等于检索收敛。

异步不等于轨迹内任意混版本：[DORA v2](https://arxiv.org/abs/2604.26256v2)同时要求轨迹内policy一致、数据完整与有界staleness，其多版本并存不直接推出单轨迹混版本。[Laminar v1](https://arxiv.org/abs/2510.12633v1)的relay／repack与故障隔离自述不替代重试／恢复协议证明。[StaleFlow v2](https://arxiv.org/abs/2601.12784v2)涉及数据管理，但摘要不足以确认改训练集合，暂撤C?而留待正文核。[RolloutPipe v2](https://arxiv.org/abs/2606.26997v2)在fixed-weight rollout内按完整group送训，作者称保持on-policy；不能因生成与训练重叠就并入stale-policy异步，也不把这一自述当本项目正确性验收。[AReaL v5](https://arxiv.org/abs/2505.24298v5)含PPO变体，联合结果不能全归调度；这里只核论文，未触及M5的prompt_mean事项。

训练侧与生成侧分开：[AReaL-DTA v2](https://arxiv.org/abs/2602.00482v2)复用的是policy训练前后向的prefix计算，DFS仅物化root-to-leaf路径并做分布式tree batching；不应由名字tree attention推成推测decode机制。首次日期是1月31日，不按ID前缀猜2月。[FlashNext-RL v1](https://arxiv.org/abs/2610.28998v1)研究模型特定训练优化，摘要中的开发配置同时改变context length；训练tokens/s增长不作同配置rollout吞吐加速，另有8层累积研究与live SWE RL迭代记录，证据类型不混成一个数字。

同权重的训推差异与跨更新的陈旧性分开：[Missing Old Logits v3](https://arxiv.org/abs/2605.12070v3)将两类修正区分，精确old logits获取与PPO-EWMA近似路线也不等价。新增[FP16 v1](https://arxiv.org/abs/2510.26788v1)、[VeXact v2](https://arxiv.org/abs/2605.14220v2)、[CIS v1](https://arxiv.org/abs/2609.32444v1)，连同[Beyond Precision v1](https://arxiv.org/abs/2602.01826v1)，分别提供精度、隔离诊断、importance修正与LR动态线索；不只凭摘要裁决哪一种解释普遍成立。VeXact v2标题比v1补入via a Zero-Mismatch Reference，不重复计数。算法改变不自动等于C类选轨迹，也不硬塞为纯系统B收益；保留边界而非扩张研究范围去做算法评测。

可靠性警示：[ASymPO原站](https://arxiv.org/abs/2606.03070)Comments明确写“incorrect proofs in the paper”，版本史标v2 withdrawn，当前页面为v3；不能误写成v3也明确撤回或警示已被修复。仅保留身份及检索线索，停止用其理论保证支持结论；旧“非对称规模”改为loss尺度，不解释成模型规模不对称。当前未核正文证明、代码或性能，深读／代码计数仍0／0。下一批优先新增rollout-native来源、未落实种子及会议引用扩展，再补serving基础；不重复这11篇或急于形成survey结论。

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

证据类型登记（摘要级，未作实验表抽取）：[Iso等2604.26779v1](https://arxiv.org/abs/2604.26779v1)将8B同步RL rollout实测与235B异步组合的performance simulator预测分开报告；不能把后者写作大模型端到端实测，也不把rollout吞吐与整步／端到端时间混成同一指标。

后续逐篇登记makespan、tokens/s、step time、bubble ratio及长度分布，同时区分作者实测、模拟预测与本项目复现，当前不建立跨论文加速比排名。

## 13. Research Gaps / Ideas

（Round 1 初步观察，待 corpus 收敛后修正）：
1. **分布改变与纯路由／重排的边界待核**：DARTS主动塑形、InfoTree选择状态／预算，WAR不再作为C类例子；ExpThink仍待审计。对分布／收敛影响的评估仍待逐篇阅读，不预设“普遍缺乏严格评估”；
2. speculative-for-RL已出现多个候选，但条目数不证明方向饱和。统一性能模型（acceptance rate × staleness × 长尾交互）是待探索问题，不据此否定其他drafter方案的研究空间；
3. cap/长度类边界条件的系统化处理（如 cap=1 的 -0 陷阱、偶奇 cap 分配）在现有论文中未见讨论——来自 verl #8281 实战的微观缺口；
4. 当前库的VLM跨阶段GPU空分仅Rollplex一篇已核条目，不足以判断文献空白；需继续查相近机制与不同工作负载。

## 14. Reproduction Candidates

（空。登记模板：为什么值得复现 / repo / model / hardware / dataset / expected cost / key metric / stopping condition。**当前不启动 GPU 实验**。已有可借力资源：M5 4090-1 verl-e2e 环境、本地 veRL 源码。）

## 15. Open Questions

1. "Rollout ≠ serving" 假设的边界：哪些 serving 优化在 rollout 下 objective 完全不变？（找反例）
2. makespan 最优 vs on-policy 性的权衡：C 类改分布的工作，train-time 收益能否抵消分布偏移的收敛代价？有没有论文量化过？
3. taxonomy 是否需要第五层（environment/sandbox 层）？agentic 簇的 environment bubble 是否够独立成层？
4. Rollplex 的 VLM 跨阶段共享是否适用于文本-only 场景（colocation 的空间共享新维度）？

## 16. Changelog / Research Log

- **2026-10-27（M1异步／训练边界补核）**：核剩余11篇历史条目，新增FP16／VeXact／CIS三篇TIM文献，累计77身份（16 serving／边界＋61 rollout/RLHF及相关边界；75 arXiv＋2会议记录）。历史✅清零不等于corpus收敛或77篇可用实证；ASymPO证明错误与v2撤回显式警示。Intelligent Router移serving，AReaL-DTA改训练侧边界，RolloutPipe的group流水与stale-policy异步分开。仅身份／摘要补核，无深读、代码、复现或M5专项改动；后续优先新增原生及种子，不重复本批。

- **2026-10-27（M1 agentic补核）**：核既有10篇，补InfoTree名称及RollArc／RollArt版本沿革；定向检索补FlexMARL与两篇相关Agent Lightning论文，累计63身份（15 serving／边界＋48 rollout/RLHF；61 arXiv＋2会议记录），仅3新发现。系统／recipe、VLA样本效率／吞吐、同步prefix重叠／async off-policy分开，暂留分类疑问而非性能结论。未全文深读、读代码或复现，下批核剩余L4／agentic边界并扩展原生。

- **2026-10-27（M1推测机制与范围补核）**：核既有8篇并补Bebop／Aurora别名，FastGRPO种子落实；EfficientRollout引用回溯新增两篇SD基础与Iso等NeMo-RL集成论文。Aurora移serving边界、ObsSpec移agentic，防止把speculator RL、observation推测或稀疏／量化draft混成policy rollout机制；Iso实测／模拟证据分开。累计50身份（15 serving／边界＋35 rollout/RLHF；48 arXiv＋2会议记录），仅3新发现。无全文深读、代码／性能验收，下批核agentic簇与扩展引用。

- **2026-10-27（M1长尾／推测补核）**：核既有9篇、落实Seer／TLT两种子，WAR引用回溯新增SuffixDecoding并核v1/v3同文改题。累计38身份（12 serving＋26 rollout/RLHF；36 arXiv＋2会议记录），仅1新发现。WAR／TailSieve暂A/B，SortedRL暂B/C?，RhymeRL补调度维度；撤回C类普遍缺评估／推测方向已饱和的先验判断。仍无全文深读、代码或复现，下批核其余推测／agentic与扩展引用。

- **2026-10-27（M1 RL系统补核）**：核8篇既有RL／RLHF系统身份与版本，收窄DistFlow／AsyncFlow／OpenRLHF旧描述；新增ROLL、EARL及从StreamRL回溯并核官方记录的PUZZLE。累计26篇身份已核（11 serving＋15 rollout/RLHF；24 arXiv＋2会议记录），仅新增3篇，不重算历史≈73。§5.3标题同步撤回预设不可迁移说法；仍无全文深读、代码或复现。下一批转长尾／推测rollout候选核验与引用扩展。

- **2026-10-27（M1基础补核与rollout补库）**：核既有9篇serving原始文献，补primary链接／首次提交／采用版本，FA1/2拆行不计新发现；新增APRIL 2509.18521v3与RLHFSpec 2512.04752v2，累计15篇身份已核（11 serving＋4 rollout/RLHF）。两新项层／类型暂定，未抽性能数字；修正PD分离与trainer/rollout分离混同，撤回speculative不可直接迁移的先验绝对说法。未做全文深读、taxonomy收束、代码或复现；下一批转rollout-native来源核验与扩展。

- **2026-10-27（M1身份核验批次）**：承接新专项，核4篇而非新增4篇；纠正SGLang／Orca／RLHFuse三处错配ID，合并RLHFuse与Stage Fusion重复项，登记ReaLHF→ReaL及MLSys'25版本信息。来源为arXiv原始／修订元数据与USENIX官网；撤回“≈73”为已去重已核数量的解释，保留Round 1历史记录。未做深读／性能判断、taxonomy收束、代码改动或复现；下一步按§5.7补其余来源与身份。

- **2026-10-27（Round 1）**：骨架建立；种子注册 + 2 批 arXiv 验证检索（40 篇实证 + 15+ 新发现：SRT/ObsSpec/S³R/NgramRL/DARTS/ASymPO/MARLaaS/AsyncWebRL/AcceRL/Stage Fusion/FlashNext-RL/AREAL-DTA 等）；corpus ≈73（目标 100–150）。待验证种子 8 个。未做：深读、代码地图扩展、corpus 收敛检索（SGLang/vLLM 博客、ICML/NeurIPS/OSDI/SOSP/NSDI/ATC/EuroSys 2025-2026 proceedings 扫描、citation graph 前向/后向扩展）。
