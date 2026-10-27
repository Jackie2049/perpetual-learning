# Rollout Performance Research — LLM Post-training / RL Rollout 性能优化调研

> 本文件是本专题唯一的 canonical knowledge base。任何会话/Agent 接手时**只读本文件**即应在 10–20 分钟内恢复研究状态；详细证据存放于链接指向的材料，但恢复研究状态不依赖它们。
> 远景（非当前任务）：systems-oriented survey《Efficient Rollout Systems for LLM Post-Training》。**当前阶段禁止**撰写正文/Introduction/Related Work、冻结 taxonomy、批量生成低价值摘要、启动大规模复现、在外部主仓库发 issue/PR/comment。

## 0. Project Status

- **阶段**：Phase 1 — recall-first literature & code investigation（广扫建库，不收束观点）
- **最近更新**：2026-10-27（M1修复并行接续覆盖，恢复§5.22与三条多模态矩阵行；§5.23保全八个接续名称，APRIL去重，另七条线索分账，不新增主矩阵／代码计数）
- **覆盖**：累计**119篇已核身份**（20篇serving基础／迁移边界＋99篇rollout/RLHF及相关边界；113个唯一arXiv ID＋Orca／PUZZLE／CachedAttention／DynaRL／Weave／MTP-RL会议记录，均非深读）。§5.20将RL相关99暂分为84直接Rollout系统／机制候选与15保守单列条目；84只是工作范围计数，不是原生80–120覆盖／质量验收。MTP-RL是ACL Findings独立文献，不与Bebop同篇；初始种子身份已全部落实，但原生覆盖／分级／引用检索仍未收敛。ASymPO证明错误警示保留；Round 1“≈73”不与当前数量相加。
- **待核线索**：§5.23另列7项（6项元数据／摘要已核、1项身份待核），不计入上述119或84候选；APRIL为已有身份，不重复计数。
- **全文深读 / 代码广扫**：0 / 6（veRL `24f25b03`三条路径＋OpenRLHF `dc2a7ad3`、NeMo RL `a9d1a4be`、SkyRL `e0893f6e`各一条；共4框架，不是6框架或运行验证，也不按文件数计）
- **最高优先级**：补原生候选覆盖及其引用／会议盲区，同时扩未覆盖框架；总量及84候选进入目标区间不等检索收敛／最终分级，A Core短名单不按计数凑齐
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

按实际改变的训练轨迹集合判断C，不按技术名称判断：partial rollout若仅提供路由信号，或历史轨迹仅供draft验证，不自动归C。CadenceRL的pacing让长轨迹pending后续跑，不因“replace”措辞自动归删样本；跨训练步重排、短组先更新等对每步集合与更新顺序的影响另核，不能由“不丢样本”排除。

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
| [2507.19017v1](https://arxiv.org/abs/2507.19017v1) | 2025-07-25／v1 | MindSpeed RL（Ascend平台） | L4（暂） | B（暂） | distributed transfer dock组织sample flow，allgather-swap处理resharding内存；平台结果不外推CUDA | 核 |
| [2509.15965v2](https://arxiv.org/abs/2509.15965v2) | 2025-09-19首次／v2 12-29 | RLinf | L3/L4（暂） | B（暂） | M2Flow时空拆分／重组工作流，context switching、elastic pipelining及profiling-guided计划 | 核 |
| [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) | 2026-07会议记录 | DynaRL | L3/L4（暂） | B（暂） | 动态hypergraph控制面，多级调度、资源迁移及context-aware路由跨组件重分配 | 核 |
| [2405.01481v2](https://arxiv.org/abs/2405.01481v2) | 2024-05-02首次／v2 09-03 | NeMo-Aligner | L1/L4（暂） | A/B（暂） | PPO多模型PyTriton异步通信／权重offload，TRT-LLM生成与原地refit、推理reshard | 核 |
| [2510.11345v1](https://arxiv.org/abs/2510.11345v1) | 2025-10-13／v1 | ROLL Flash（Part II） | L3/L4（暂） | B（系统；算法另核） | 承接ROLL的细粒度并行与rollout-train解耦，queue调度／环境级异步及off-policy算法 | 核 |
| [2508.06471v1](https://arxiv.org/abs/2508.06471v1) | 2025-08-08／v1 | GLM-4.5报告（Slime §3.5） | L1/L4（暂） | A/B（系统；recipe另核） | Megatron／SGLang-Router／Data Buffer，支持同置同步及分离异步；BF16训、在线block FP8推 | 核 |
| [2512.07710v1](https://arxiv.org/abs/2512.07710v1) | 2025-12-08／v1 | Each Prompt Matters（CompassMax-V3-Thinking） | L1/L3/L4（暂） | A/B／**C**（联合算法） | FP8 rollout、reward重叠与长度调度；另有zero-variance过滤、ESPO及MoE Router Replay | 核 |
| [2605.07330v1](https://arxiv.org/abs/2605.07330v1) | 2026-05-08／v1 | SparseRL-Sync | L4（暂） | B（暂） | 稀疏索引／值传输重建post-cast rollout权重；通信量模型须计编码与索引开销 | 核 |
| [2602.03839v2](https://arxiv.org/abs/2602.03839v2) | 2026-02-03首次／v2 05-19 | PULSE（Understanding and Exploiting Weight Update Sparsity） | L4／训练通信边界（暂） | B（Sync；LoCo另核） | PULSESync传BF16权重patch；PULSELoCo为trainer间FP32 pseudo-gradient与error feedback，非同一通道 | 核 |
| [2601.18150v2](https://arxiv.org/abs/2601.18150v2) | 2026-01-26首次／v2 04-10 | FP8-RL | L1/L2/L4（暂） | A/B（系统；数值校正另核） | W8A8 policy rollout与FP8 KV，每步QKV scale重校准，TIS/MIS处理训推失配；非量化draft | 核 |
| [2610.16866v1](https://arxiv.org/abs/2610.16866v1) | 2026-10-12／v1 | PivotRL（Efficient Resource-Shiftable RL Post-training with Generative Rewards） | L3/L4（暂） | B（暂） | Global Parameter Storage与role context切换，迁移sequence并权衡资源转移收益／开销 | 核 |
| [2512.22492v1](https://arxiv.org/abs/2512.22492v1) | 2025-12-27／v1 | RobustRL（Role-Based Fault Tolerance System for LLM RL Post-Training） | L4（暂） | B（容错；暂） | role故障隔离、rollout warm standby与动态UCX重连；恢复收益限故障评估条件 | 核 |
| [2610.08430v1](https://arxiv.org/abs/2610.08430v1) | 2026-10-06／v1 | NeMo-DCR | L4（暂） | B（暂） | canonical坐标与native loader放置XOR／overwrite增量，joint commit绑定下一delta基线；bit-exact针对dense refit | 核 |
| [2610.05935v1](https://arxiv.org/abs/2610.05935v1) | 2026-10-05／v1 | ThunderSyncRL | L4（暂） | B（暂） | 固定policy下按reward／teacher-score就绪做gradient streaming，batch结束才一次optimizer更新 | 核 |
| [2509.19128v2](https://arxiv.org/abs/2509.19128v2) | 2025-09-23首次／v2 09-26 | PipelineRL | L2/L4（暂） | B／**C?**（行为分布另核） | 并行生成训练，in-flight换权重后继续未完序列；保留旧KV的mixed-policy不当零陈旧性 | 核 |
| [2609.36830v1](https://arxiv.org/abs/2609.36830v1) | 2026-09-29／v1 | PACE（Where Does Staleness Accumulate?） | L4（暂） | B／**C**（拒收） | 分解生成／入池后等待staleness，pool超额占用决定拒收预算，按prefix-aware有效陈旧度排序 | 核 |
| [2510.19225v3](https://arxiv.org/abs/2510.19225v3) | 2025-10-22首次／v3 2026-04-08 | RLBoost | L3/L4（暂） | B（暂） | 自适应partial seeding、pull式权重传输与token流收集／迁移，利用抢占式资源且不因抢占直接丢请求 | 核 |
| [2602.02192v6](https://arxiv.org/abs/2602.02192v6) | 2026-02-02首次／v6 09-29 | ECHO-2 | L3/L4（暂） | B／**C**（过旧丢弃） | 中央learner／远程rollout、peer-assisted流水广播及成本感知扩缩；版本窗口外轨迹丢弃 | 核 |
| [2604.11554v2](https://arxiv.org/abs/2604.11554v2) | 2026-04-13首次／v2 04-14 | Relax（Omni-Modal Post-Training） | L3/L4（暂） | B（暂） | role服务解耦、TransferQueue字段／microbatch流水与producer陈旧性门控；encoder-aware并行处理多模态输入 | 核 |
| [2507.07966v4](https://arxiv.org/abs/2507.07966v4) | 2025-07-10首次／v4 09-30 | Long-RL（Scaling RL to Long Videos／MR-SP） | L1/L2/L4（暂） | A/B（MR-SP；数据另核） | 分片并行视频编码、all-gather embedding供多rollout复用，policy／reference prefilling用SP；forward计时不当全步 | 核 |

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
| [2605.07501v2](https://arxiv.org/abs/2605.07501v2) | 2026-05-08首次／v2 05-17 | ExpThink（CoT塑形边界） | workload／算法边界 | C边界（非直接筛轨迹） | 历史最短正确长度引导reward塑形及correct-count advantage归一化；改变学习目标，不是执行同一轨迹更快 | 核 |
| [2506.22950v1](https://arxiv.org/abs/2506.22950v1) | 2025-06-28／v1 | Infinite Sampling | L2/L3/L4（暂） | B（暂） | micro sampling groups与跨组连续采样，prefix长度预测、全局分组及SJF refill；完整组统计后反向 | 核 |
| [2609.36899v1](https://arxiv.org/abs/2609.36899v1) | 2026-09-29／v1 | CadenceRL（Reshaping Rollout Workloads） | L2/L3/L4（暂） | B（暂） | pacing让长context暂候、concentration聚尾到亲和worker，late-bound KV准备；不是直接删长轨迹 | 核 |
| [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan) | 2026-07会议记录 | Weave（Efficient Co-Scheduling for Disaggregated RL Post-Training） | L3/L4（暂） | B（暂） | co-execution group内跨job交错填补同步气泡，两级调度及host-memory residency支持warm切换 | 核 |
| [2606.05606v1](https://arxiv.org/abs/2606.05606v1) | 2026-06-04／v1 | CERO（Cross-Epoch Adaptive Rollout Optimization） | workload／算法边界 | **C**（预算分配） | 用prompt成功率Beta posterior估计边际价值，在跨epoch全局预算下分配rollout次数；非固定轨迹执行调度 | 核 |

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
| [2608.04962v2](https://arxiv.org/abs/2608.04962v2) | 2026-08-05首次／v2 10-14 | SPARK（v1名SpecRoll） | L1/L2（暂） | A（暂） | verifier反馈修正draft logits，同prompt共享rollout-local correction、跨rollout学projection；不改为直接采样draft | 核 |
| [2511.16193v3](https://arxiv.org/abs/2511.16193v3) | 2025-11-20首次／v3 12-23 | SpecActor（Fast LLM Post-training） | L1/L3/L4（暂） | A/B（暂） | 解耦draft／verification，Fastest-of-N按接受率与draft ladder选择方法并利用长尾空闲worker | 核 |
| [2609.07108v1](https://arxiv.org/abs/2609.07108v1) | 2026-09-07／v1 | Online Draft Co-Training（Large-Scale Long-Context RL） | L1/L2/L4（暂） | A/B（系统；暂） | packed zigzag-ring合并causal／branch attention，TapChannel跨PP传target features，支持在线draft训练 | 核 |
| [ACL Findings'26](https://aclanthology.org/2026.findings-acl.1871/) | 2026-07会议记录 | MTP-RL（Acceleration of Reinforcement Learning Rollouts with Policy-Aligned Multi-Token Prediction） | L1/L4（暂） | A（rollout）；共训语义待核 | 两阶段配置多层参数共享MTP、advantage-aware优化使draft跟随policy；不当无条件目标分布／训练等价证明 | 核 |

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
| [2608.10402v1](https://arxiv.org/abs/2608.10402v1) | 2026-08-11／v1 | TideRL | L2/L3/L4（暂） | B（暂） | 保留rollout状态的连续task batching，以ready backlog／到达间隔选择ref-actor执行及弹性rank分配 | 核 |
| [2605.06534v2](https://arxiv.org/abs/2605.06534v2) | 2026-05-07首次／v2 05-20 | ROSE | L3/L4（暂） | B（暂） | 在SLO约束下借用serving GPU，跨集群分片／稀疏权重传输与弹性rollout路由 | 核 |
| [2609.33848v2](https://arxiv.org/abs/2609.33848v2) | 2026-09-27首次／v2 10-20 | QwenGyre | L3/L4（暂） | B／**C?**（轨迹处理） | live执行中弹性重分配GPU；重建分支历史、partial-progress评分和路径去重，loss／样本语义另核 | 核 |
| [2510.04206v1](https://arxiv.org/abs/2510.04206v1) | 2025-10-05／v1 | AgentRL | L4（暂） | B／**C**（cross-policy） | 全异步生成训练与统一环境接口；另有单轨迹内跨模型采样和task advantage归一化 | 核 |
| [2607.01415v1](https://arxiv.org/abs/2607.01415v1) | 2026-07-01／v1 | The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning | L4环境执行边界（暂） | B（暂） | 固定command轨迹对照四种执行底座，测启动／执行／reset；worker-hours为外推而非完整RL训练实测 | 核 |
| [2607.07508v1](https://arxiv.org/abs/2607.07508v1) | 2026-07-08／v1 | SAO（Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning） | L4／算法边界（暂） | **C**／算法改动 | 每prompt单rollout替代group sampling，配value-model设计与double-sided token clipping；不作GRPO等价系统加速 | 核 |
| [2609.35158v1](https://arxiv.org/abs/2609.35158v1) | 2026-09-28／v1 | PEARL | L2/L3/L4（暂） | B（暂） | 外部弹性与临时借用训练GPU联合选择PD同置／分离及比例，以切换收益门槛抑制无效重配置 | 核 |
| [2603.13019v1](https://arxiv.org/abs/2603.13019v1) | 2026-03-13／v1 | ARL-Tangram | L3/L4（暂） | B（暂） | action级调度共享CPU／GPU／API外部资源，异构manager及弹性分配减少trajectory级过量预留 | 核 |
| [2509.23866v1](https://arxiv.org/abs/2509.23866v1) | 2025-09-28／v1 | DART（DART-GUI） | L3/L4（暂） | B／**C**（数据／step选择） | 环境／rollout／data manager／trainer异步解耦及逐worker同步；另调rollout数／长度、经验池与高熵step训练 | 核 |

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
| [USENIX ATC'24](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | 2024-07会议记录 | CachedAttention | 分层KV、逐层预载／异步保存及scheduler-aware换入换出；context-window有效性不等policy更新后缓存仍有效 | 核 |
| [2601.22705v1](https://arxiv.org/abs/2601.22705v1) | 2026-01-30／v1 | CONCUR | agent级准入与cache信号控制并发、缓解middle-phase thrashing；RL下完整组／陈旧性须另核 | 核 |
| [2511.02230v7](https://arxiv.org/abs/2511.02230v7) | 2025-11-04首次／v7 2026-09-08 | Continuum | 工具等待中按reload／排队成本设置KV TTL并结合program-level FCFS；TTL到期是缓存回收，不等轨迹终止 | 核 |
| [2603.16104v1](https://arxiv.org/abs/2603.16104v1) | 2026-03-17／v1 | Helium（Efficient LLM Serving for Agentic Workflows） | workflow视作query plan，proactive缓存与cache-aware调度；进入RL需核版本／随机输出／样本独立性 | 核 |

### 5.6 初始种子落实状态

初始清单目前无未落实身份。MTP-RL已由ACL Findings'26记录落实（§5.20），与2606.12370／Bebop分开；旧方向候选没有冒充同名来源。身份落实不等于检索、覆盖或可信度验收完成。

Weave已由OSDI'26官方记录落实，Online Draft Co-Training已落实2609.07108v1，见§5.19；两者从待核转入矩阵，不是该轮新发现的两个名称。

Seer与Taming the Long-Tail（TLT）已在§5.2／§5.3落实来源。历史早批次Weave定向检索未命中，不据此判定不存在；本批改从会议官方入口落实。

FastGRPO已落实2509.21792v1；不据online draft learning关键词把它等同其他未确认同名的种子。

SpecRoll已落实2608.04962v1，采用同ID改题的SPARK v2，只计一篇，见§5.14。ExpThink既有线索已落实2605.07501v2，作为算法塑形边界，不把其token压缩直接列为系统执行加速。早期Weave关键词未命中的记录保留在§5.12／§5.14；不与机器人同名工作合并，亦不采用错配2602.11143，见§5.19。

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

### 5.14 原生系统补库与SpecRoll版本落实

2026-10-27第八批：定向检索新增TideRL、ROSE、QwenGyre三个来源，落实SpecRoll种子及ExpThink既有线索；五个ID均回arXiv原站核标题、首次提交和采用版本。累计82身份=77＋5（80唯一arXiv＋2会议记录），其中本批4个原生系统／机制和1个算法塑形边界；不把全部五篇称作全新发现。

版本与推测对象：[SpecRoll v1](https://arxiv.org/abs/2608.04962v1)的fast/slow与Reflex描述属于旧版；同ID的[SPARK v2](https://arxiv.org/abs/2608.04962v2)已改题并增作者，不能重复注册或将旧机制直接迁入新版。[v2 §3／§4.4及附录B.4](https://arxiv.org/html/2608.04962v2)说明同prompt共享correction、每轮验证后更新且每次rollout重置，projection在rollout后学习；这不是直接更改target policy。其分布保证依赖exact verifier获得真实proposal概率、独立随机性及只使用此前轮反馈，不能泛化成任意在线draft更新都正确或bitwise轨迹相同。此处只定位机制／假设，未审计完整证明或实现。

三个agentic资源边界分别登记，不凭“弹性”合并：

| 来源 | 调度对象与信号 | 需保留的边界 |
|---|---|---|
| [TideRL v1](https://arxiv.org/html/2608.10402v1) | task状态保留；ready backlog／到达间隔选择ref-actor策略及rollout／training rank配比 | 作者以送达trainer并推进更新的token计training goodput；不能用GPU占用率或KV命中率代替time-to-quality |
| [ROSE v2](https://arxiv.org/abs/2605.06534v2) | 专用rollout池与已部署serving GPU协作，共享compute／memory且受serving SLO约束 | 不是只把serving算法迁到rollout，而是二者共享物理资源；权重传输稀疏性不等于稀疏policy采样，SLO结果限作者负载 |
| [QwenGyre v2](https://arxiv.org/abs/2609.33848v2) | live rollout中重分配GPU；重建分支历史并评分／去重 | 系统B与轨迹处理C?分开；需核是消除存储重复、loss重复还是改变训练集合，不能仅凭dedup判无分布影响 |

[ExpThink v2](https://arxiv.org/abs/2605.07501v2)跟踪每问题历史最短正确长度，以分级reward与correct-count advantage引导简短正确回答，是改变学习目标／未来工作负载的边界案例；不是为同一固定trajectory set减少执行开销。摘要中的accuracy／平均token比也不是RL step time或time-to-quality，后续不为此扩展成CoT压缩算法综述。

本批限元数据、摘要和SPARK／TideRL局部正文定位，非全文深读、代码阅读或性能验收；分类仍暂定。Weave未确认同名，MTP-RL与Online Draft Co-Training仍未落实，不凭相似机制合并。后续优先会议／引用扩展补原生，同时补serving至20–30篇目标；不以82总量代替原生80–120篇目标已达成。

### 5.15 TideRL引用扩展与多轮缓存迁移边界

2026-10-27第九批：从[TideRL v1参考文献](https://arxiv.org/html/2608.10402v1)回溯[3] SpecActor、[4] MindSpeed RL、[37] Infinite Sampling、[44] AgentRL及serving侧[2] CONCUR、[6] CachedAttention、[15] Continuum、[35] Helium，逐一回arXiv或USENIX原站核标题／首次日期／采用版本。新增8条，累计90身份（87唯一arXiv＋3会议记录），20 serving／边界＋70 rollout/RLHF及相关边界；不重复版本，不把引用年份当首次提交年份。CachedAttention采用ATC'24记录，不猜arXiv ID；Continuum首发2025而采用v7修订2026。

方法选择不等于训练样本选择：[SpecActor v3 §4.2](https://arxiv.org/html/2511.16193v3)的Fastest-of-N用draft ladder、接受率及空闲worker调整draft方法，不是选最快完成的N条训练响应，暂A/B而非自动C。[Infinite Sampling v1 §3.4](https://arxiv.org/html/2506.22950v1)缓存各micro group奖励、等完整组均值后形成advantage再分micro batch反向；micro group不能直接解释成独立的GRPO归一化组，也不凭此局部描述承诺完整算法／数值等价。其长度预测prefix被续用，不直接判为删选响应。

主动混policy与系统异步分开：[AgentRL v1 §3.1](https://arxiv.org/html/2510.04206v1)的cross-policy sampling允许一条轨迹的不同动作来自模型池，另有task advantage normalization。明确记C及算法边界，不等同DORA所约束的单轨迹policy版本一致，也不能把质量收益全部归全异步pipeline。[MindSpeed RL v1](https://arxiv.org/abs/2507.19017v1)组织sample／resharding两类flow，Ascend NPU是作者实验平台；仅登记架构来源，不把通信策略或性能结果直接外推CUDA，也不触发本项目NPU部署／测试任务。

多轮serving各有不同的复用单位：[CachedAttention官方记录](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)关注分层KV及预载／保存，[Continuum v7](https://arxiv.org/abs/2511.02230v7)以TTL保留工具等待中的KV，[CONCUR v1](https://arxiv.org/abs/2601.22705v1)在agent级调并发，[Helium v1](https://arxiv.org/abs/2603.16104v1)把workflow建模为query plan。迁到RL须重核权重身份、group依赖及训练进度，而不能由缓存能保留推出可以跨policy更新直接复用；TTL到期或cache eviction也不是删除训练样本。Helium的中间结果复用是否满足随机采样独立性仍待核，不把缓存命中率或serving JCT直接换算RL time-to-quality。

范围限元数据／摘要、引用位置和SpecActor／Infinite Sampling／AgentRL局部机制定位，无全文深读、代码阅读或性能抽取／复现。serving登记达到20条只是数量进展，基础机制覆盖仍待补；70个RL相关条目也含边界，未达到80–120原生目标。下一批优先继续原生引用扩展及未落实种子，不为追总数扩成算法综述或重复本批核验。

### 5.16 原生框架、异构调度与低精度系统来源

2026-10-27第十批：定向搜索与项目出版线索补7篇，均回原站核身份／版本。六个新arXiv ID加DynaRL的OSDI'26官方记录，累计97身份（93唯一arXiv＋4会议记录），20 serving／边界＋77 rollout/RLHF及相关边界。来源包含五篇框架／调度论文及GLM-4.5、Each Prompt Matters两篇含系统机制的模型报告；只取后者的rollout系统线索，不展开模型能力或算法综述。

框架与论文版本分开：[RLinf v2](https://arxiv.org/abs/2509.15965v2)的M2Flow、context switching与elastic pipelining也见[OSDI'26官方记录](https://www.usenix.org/conference/osdi26/presentation/yu-chao)，合计一篇；[DynaRL](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)是另一题名／作者记录，动态hypergraph、资源迁移与在线调度单列，不按两篇论文推断两套独立实现或独立复现实证。[ROLL Flash v1](https://arxiv.org/abs/2510.11345v1)明确扩展ROLL，是不同ID的后续论文而非ROLL原ID修订；off-policy算法与调度联合结果不全归纯系统B。

异步通信不必然等于stale-policy训练：[NeMo-Aligner v2 §3.1–3.2](https://arxiv.org/html/2405.01481v2)以PyTriton连接PPO模型、同架构不同权重通过offload／swap复用，并优化TRT-LLM生成、推理reshard及原地refit。不能由异步requests直接推断当前NeMo-RL的训练语义或软件能力。[CadenceRL v1 §3.4／§4](https://arxiv.org/html/2609.36899v1)的长轨迹会进入pending并在合适worker续跑，pacing／concentration以staleness预算调节吞吐—时延，不把摘要“replace long-context”解释成丢样本；暂B，跨版本协调与每步训练集合仍需后续核实现。

低精度policy rollout与低精度draft分开：[GLM-4.5 v1 §3.5](https://arxiv.org/html/2508.06471v1)给出Slime的Megatron训练、SGLang＋Router生成、Data Buffer三模块，支持同置同步／分离异步；训练BF16，每轮发往rollout前在线block-wise FP8量化。它不是量化draft再由dense target验证，采样概率／训推修正仍须另核。[Each Prompt Matters v1](https://arxiv.org/abs/2512.07710v1)同含FP8 rollout、reward重叠、长度调度和zero-variance过滤／ESPO／Router Replay，过滤明确记C；MoE Router Replay不等于回放历史trajectory，质量收益不能全归FP8或系统调度。

证据边界：[NeMo-Aligner v2 §3.5](https://arxiv.org/html/2405.01481v2)报告其小问题规模下worker-pool负载均衡开销抵消收益，不能把“有长尾／均衡器”直接推成任意batch都更快。此处仅定位定性条件，不抽数值比较。全批限元数据、摘要及上述局部机制定位，无全文深读、代码或设备验证；总数未替代原生覆盖目标。下一批优先原生调度／权重同步／数据管理引用与未落实种子，继续补缺而非复核本批或为凑百篇加无关算法。

### 5.17 权重同步、低精度与运行开销补库

2026-10-27第十一文献批：定向检索补5篇，再沿[SparseRL-Sync v1参考文献](https://arxiv.org/html/2605.07330v1)的具体链接找到PULSE，分别回arXiv核标题、首次日期及采用版本，去重新增6篇。累计103身份（99唯一arXiv＋4会议记录），20 serving／边界＋83 rollout/RLHF及相关边界；两种PULSE算法不拆两篇，FP8-RL v2不与v1重复。SparseRL-Sync参考文献含TODO／占位条目，不继承其中未核的年份、ID或优先性判断；其正文日期不替代arXiv提交历史。

**同步的精度与通道**：[SparseRL-Sync §3.3](https://arxiv.org/html/2605.07330v1)的稀疏payload含索引和值，另有编码／metadata成本；标题的通信量缩减不能直接当端到端加速。BF16转换掩盖小FP32更新，不是优化器没有更新；“无损”针对重建的post-cast权重。[PULSE v2](https://arxiv.org/abs/2602.03839v2)同样从compute-visible sparsity出发，但PULSESync是trainer→inference的BF16 patch，PULSELoCo是trainer→trainer的DiLoCo式FP32 pseudo-gradient与error feedback；前者的bit-identical重建不推广到后者或整个训练确定性。两论文相近机制不合并为别名。

**低精度、迁移和恢复分开**：[FP8-RL v2](https://arxiv.org/abs/2601.18150v2)直接改policy rollout数值路径，并用importance-sampling校正，不能当exact BF16采样保证或draft优化。[PivotRL v1](https://arxiv.org/abs/2610.16866v1)定位GenRM／rollout／train三角色竞争资源，sequence migration不自动等于删样本；具体更新顺序／质量条件待深读。[RobustRL v1](https://arxiv.org/abs/2512.22492v1)的ETTR及训练时间收益来自故障注入评估，不当无故障稳态tokens/s收益；角色重连不自动证明所有恢复一致性条件。

**执行底座的证据单位**：[Rollout Infrastructure Tax v1 §4.4／§6](https://arxiv.org/html/2607.01415v1)以固定命令序列比较container、hosted sandbox、Kubernetes和VM；worker-hours按测得轨迹latency外推，排除模型推理、reward评估、optimizer及worker外排队，并非live policy驱动的完整训练对照。页面标preprint／submitted to SoCC，不登记为会议录用。此项暂放L4环境边界，不据此冻结第五层taxonomy。全批仅身份／摘要和上述局部机制／评估条件，全文深读仍0，代码广扫仍4条／2框架，无原生执行、设备或性能验收。下步继续原生引用／会议与种子扩展，或第三框架生命周期；总量不代替覆盖与分级。

### 5.18 更新屏障 序列内换权重与样本拒收

2026-10-27第十二文献批：接上轮NeMo相关搜索线索、本轮项目出版入口及in-flight更新扩展，补4个此前未登记ID。NeMo-DCR可由固定[NeMo RL README:17](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/README.md#L17)追溯，但采用arXiv首次提交10-06，不把项目公告10-07当首发；PipelineRL采用v2，其余v1。累计107身份（103唯一arXiv＋4会议记录），20 serving／边界＋87 rollout/RLHF及相关边界；没有把四篇都说成新近发表或纯原生目标已完成。

**同步重排与mixed-policy分开**：[ThunderSyncRL v1 §3.3–3.4](https://arxiv.org/html/2610.05935v1)让GRPO的单轨迹score gradient在reward就绪后先算，待整组统计齐全再组合；OPD按teacher-scored turn释放工作。policy在该批保持固定，所有登记源及延后传播结束后归约、归一化／clip、一次optimizer更新。作者的更新等价主张限其目标／条件，不当任意PPO loss、任意实现或浮点逐bit训练证明。反观[PipelineRL v2 §4／§5.1](https://arxiv.org/html/2509.19128v2)，换权重后继续同一未完序列，作者明确分析mixed behavior policy与旧KV，并单列重算KV比较；“更on-policy”不等单轨迹固定版本或零staleness。暂B／C?，意在提示行为分布差异待核，不因in-flight名称直接判定删样本。

**池选择不只是在执行上排队**：[PACE v1摘要](https://arxiv.org/abs/2609.36830v1)区分完成前的Generation Staleness与入完成池后的Waiting Staleness，以超额pool占用调节拒收预算，再结合prefix-aware生成陈旧度排序。拒收改变训练保留集合，显式列C；不能与§8.6的ready_first生产gate、纯背压或仅重排无损互称等价。本批只登记摘要机制，不由其结果反推所有长序列应拒收或已经验证该分数。

**两种lossless的对象不同**：[NeMo-DCR v1 §4–§8及附录C](https://arxiv.org/html/2610.08430v1)用canonical映射、native loader、XOR／overwrite及失败后overwrite重试和joint commit，目标是接收端参数／buffer位与dense refit一致；不是整个RL更新轨迹不变或跨任意loader无条件保证。refit延迟比较采用transport-only full-checkpoint reference和指定change-rate压力条件；它不是端到端训练加速，附录计时还排除post-apply check，gate重开在计时窗后。不要把论文中的bit-exact refit与ThunderSyncRL的目标更新等价混为一种验收。NeMo-DCR软件入口存在不代表§8.6已审计其delta实现。

全批限原站身份／版本、摘要与上述局部机制和计时口径定位，全文深读仍0，代码仍5条路径／3框架；未运行作者代码、验证证明、复现数值或启动设备任务。后续继续原生引用／会议扩展和未落实种子，或补未覆盖框架；具体结论按目标、版本和采样语义再分级，不按总数收束。

### 5.19 两项种子落实与样本预算边界

2026-10-27第十三文献批：落实Weave与Online Draft Co-Training两个既有种子，另补CERO、SAO两个预算／算法边界条目；新增4个已核身份，但不是4个新发现的名字。累计111身份（106唯一arXiv＋5会议记录），20 serving／边界＋91 rollout/RLHF及相关边界，不把后者当91篇纯原生系统；MTP-RL仍待同名来源。

**Weave的复用单位是跨job气泡。** [OSDI'26官方记录](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan)给出论文标题、作者与页码809–827；采用会议身份，不猜arXiv。co-execution group隔离locality域，组间规划placement、组内round-robin交错不同job的活动阶段，host memory驻留约束支持warm切换。这里的cost efficiency／SLO属于多job调度条件，不等单job rollout延迟收益；摘要的最优性主张未作证明验收。二手检索曾给出2602.11143，但[该ID原站](https://arxiv.org/abs/2602.11143v3)实为APEX humanoid traversal，不进入本库；也不与W&B工具或机器人Weave同名合并。

**Online Draft Co-Training补的是draft训练的并行布局。** [2609.07108v1 §3及§4.2–4.5](https://arxiv.org/html/2609.07108v1)的branch attention／TapChannel服务在线draft更新，不是FP8 policy数值路径；256K证据对应attention算子扩展，不等256K完整RL端到端验证。无draft且无speculation的对照也不能单独归因于“在线训练胜过冻结draft”；完整更新overhead不全归TapChannel。接受长度、算子加速和整步加速分别登记，不从局部一致性曲线推普遍exactness。摘要的软件链接只登记为入口，不认作本轮已验证实现。

**分配工作和执行同一工作分开。** [CERO v1摘要](https://arxiv.org/abs/2606.05606v1)以固定全局预算耦合prompt和epoch的rollout次数，属于C类采样分配；其regret条件是fixed prompt utilities，不外推为RL非平稳训练保证。[SAO v1摘要](https://arxiv.org/abs/2607.07508v1)改为单rollout、value-model与clipping，组估计／loss一并改变，不是只拆掉GRPO同步等待。两项只补workload边界，不展开算法综述或把样本效率当tokens/s。

全批仅官方身份／版本、摘要和draft论文上述局部机制／评估条件，全文深读仍0，代码仍5条／3框架；未运行作者实现、验证理论／数值或启动设备任务。下一步继续原生引用／会议与唯一未落实种子，或未覆盖框架；总量不替代覆盖与分级，不从两项种子落实直接转survey正文。

### 5.20 MTP-RL身份与保守覆盖账

**最后一项初始种子已落实。** [ACL Findings'26官方记录](https://aclanthology.org/2026.findings-acl.1871/)列出Ke Wang等8位作者、标题、页码37530–37542及DOI；采用会议身份，不猜arXiv。其摘要描述两阶段、多层参数共享MTP及advantage-aware优化。[Bebop 2606.12370v1](https://arxiv.org/abs/2606.12370v1)则是Yucheng Li等的Breaking Entropy Bounds，身份与机制均不同，不合并别名。当前只核元数据与摘要；参数共享／共训对policy梯度的影响和验证路径待后续阅读，不由rollout更快或接受率提高推出训练等价。累计112身份＝106 arXiv＋6会议记录；本批只是旧种子落实，非新名字发现。

**以下是基于当前矩阵的保守工作口径，不是最终文献分级或新taxonomy（随§5.22更新）。** 沿用已有范围边界，并把主要呈现模型／训练优化、仅包含部分rollout系统内容的3篇暂单列，避免把所有RL相关条目充作原生覆盖；不删除任何论文或撤回其中已有机制。剩余直接候选仍含C类工作、联合算法和待核语义，不能称84篇已验收的纯原生／无损优化。

| 本轮口径 | 数量 | 计数含义 |
|---|---:|---|
| 直接Rollout系统／机制候选 | 84 | §5.1–5.4已核99条中扣除下列15条；含§5.21四项及§5.22三项，不要求全为A/B或不改采样 |
| 保守单列的范围／混合证据 | 15 | 见逐项清单；暂不用于宣称原生80–120达标，也不是最终C Boundary等级 |
| Serving基础／迁移边界 | 20 | §5.5，独立于上述99；最终相关性仍需再核 |
| 合计 | 119 | 84＋15＋20；身份数，不是全文深读或性能验收数 |

单列15条的可复核清单（ID按现有矩阵，无新身份或重复计数）：

| 范围 | 数量 | 条目及唯一ID |
|---|---:|---|
| 概率／数值／算法诊断 | 6 | Missing Old Logits `2605.12070`；ASymPO `2606.03070`；Beyond Precision `2602.01826`；FP16 `2510.26788`；VeXact `2605.14220`；CIS `2609.32444` |
| 采样分配／目标或估计器改变，暂按算法边界保留 | 3 | ExpThink `2605.07501`；CERO `2606.05606`；SAO `2607.07508` |
| 含系统片段的模型报告／训练优化混合证据 | 3 | GLM-4.5 `2508.06471`；Each Prompt Matters `2512.07710`；FlashNext-RL `2610.28998` |
| 非语言环境／训练侧／执行底座 | 3 | AcceRL `2603.18464`；AReaL-DTA `2602.00482`；Rollout Infrastructure Tax `2607.01415` |

C标签不是剔除规则：DARTS、InfoTree、PACE等仍在直接候选，因矩阵已登记其rollout机制；CERO／SAO等单列是沿用当前算法范围判断，后续发现明确系统证据可以调整。PULSE因含trainer→rollout的PULSESync保留在直接候选，不能把同篇LoCo训练通信也当原生rollout。单列是计数上的保守选择，不是认定15条无系统价值；也不以标题或venue替代后续实质判断。

历史盘点时77尚不足80，说明当时“总量112／RL相关92”不能宣布原生覆盖达标；§5.21按真实机制缺口新增4项后为81。进入数量区间仍没有检索收敛证明，也不替代分级与证据质量检查；初始种子全部落实仅关闭身份待核，不关闭整个广扫。MTP-RL批保留原111矩阵行／代码地图；后续新增见§5.21。全文深读0、代码5条／3框架，无作者实现、理论／性能验证或设备任务。

### 5.21 抢占 跨域与外部资源弹性

沿Weave引用中的RLBoost线索及elastic／cost-efficient定向检索，回到arXiv版本页与正文局部核验四项新身份；不是重复计ROSE／TideRL，也不把通用pretraining／serving弹性论文充作原生。累计116身份＝110 arXiv＋6会议，96 RL相关暂分81直接候选＋15单列；20 serving不变。ARL-Tangram直接编排agentic rollout中的action／reward资源，故暂入直接系统候选；区别于Infrastructure Tax仅固定command轨迹测底座，不因都涉及环境就混记范围。

- **抢占后续跑不等重新采样或零恢复成本。** [RLBoost v3 §4／§6](https://arxiv.org/html/2510.19225v3)用限时partial seeding及token流保留已生成prefix，迁移后重新prefill再续跑，按最新已装权重路由。作者采用同步GRPO；本轮不认证RNG／数值／逐样本完全相同。实验将真实可用性trace在按需实例上重放，成本用跨云均价估计；不当真实spot账单或固定GPU数的纯执行加速。v3 HTML标题含Cloud、abs标题不含，按同ID处理。
- **跨域重叠带样本准入语义。** [ECHO-2 v6 §4.1／§5.1–5.2](https://arxiv.org/html/2602.02192v6)按learner版本窗口收轨迹、过旧丢弃，故显式B/C；S不是队列容量。peer-assisted广播／成本选worker属系统部分，不由“未改RL目标”推出训练样本集合不变。主对照用4 A100 learner＋5090 pool对8 A100中央基线，成本由GPU-hours×历史租价估算；cost–quality优势不当单步更快或同硬件比较。NeurIPS'26仅abs自报，本轮不作官方venue确认。
- **同资源trace和更多GPU基线分开。** [PEARL v1 §4／§6.1](https://arxiv.org/html/2609.35158v1)联动外部扩缩、暂借训练GPU及PD模式／比例，并计切换成本。RLBoost+是作者针对异步agentic的改编，不是RLBoost原版；两者共享外部可用性trace与16训练GPU，ROLL仅4固定rollout GPU，ROLL-Large另用固定较大池。throughput计生成＋训练token／step时间，不等纯decode速率、价格效率或time-to-quality；保留预印本身份，不采用HTML会议模板占位字样。
- **外部资源节省不等policy推理内核加速。** [ARL-Tangram v1 §4–6](https://arxiv.org/html/2603.13019v1)把trajectory级长期资源占用改为action级共享，覆盖CPU执行、GPU服务及带quota的API。主指标平均ACT与RL step分别报告；coding／MOPD／DeepSearch各用不同基线，含内部数据／服务配置，不能将最佳action收益当所有任务整步收益或公开可复现保证。暂B描述资源编排，不认证重试／缓存对环境语义完全等价。

本轮仅身份／摘要和上述局部机制／评估设置；未通读四篇、运行代码或抽取完整实验表，深读0、代码5条／3框架不变。下一步补原生引用／会议盲区或未覆盖框架；81不作停止广扫或冻结taxonomy的条件，无新算力／M5专项交接。

### 5.22 多模态输入执行与视觉agent补库

多模态系统不能只按视觉agent的任务名称归一类。新增Relax、Long-RL和DART三个唯一arXiv身份，累计119＝113 arXiv＋6会议；99 RL相关暂为84直接候选＋15单列，serving20不变。Long-RL虽含数据／模型贡献，但§5有明确MR-SP机制、§6.2有独立系统消融，故按该系统证据暂列直接候选，不把其整篇质量收益算作系统优化；单列15条不变。DART-GUI不与既有DARTS `2605.30859`混同。

- **模态字段流水不等所有组件都不等待。** [Relax v2 §3–5](https://arxiv.org/html/2604.11554v2)以TQ字段就绪和microbatch流水解耦RL服务，并将模态encoder放在PP0；这是数据依赖／布局优化，不是Rollplex的跨阶段GPU空间共享。其max_staleness描述producer门控，具体版本更新／消费准入仍须代码核对。摘要把1.20×标on-policy，§5.3却明确是16 H800、Qwen3-4B／DAPO、off-policy且staleness=1；采用后者条件并保留内部不一致。表中rollout额外墙钟0指关键路径被遮蔽，不指生成计算免费。
- **缓存视频embedding与跨policy缓存分开。** [Long-RL v4 §5／§6.2](https://arxiv.org/html/2507.07966v4)把帧编码分片、all-gather后供多rollout及policy／reference prefilling复用；SP也用于prefilling。该局部描述不认证encoder更新后跨step缓存仍有效。系统消融在8 A100 80GB、每GPU batch=1、rollout数5上，10次warmup后平均5次；作者明确测每步forward time，512帧的2.1×不当完整优化步或time-to-quality收益。大帧数可运行能力与同设置计时分别保留。
- **GUI解耦与数据策划不是同一收益。** [DART v1 §3–5](https://arxiv.org/html/2509.23866v1)将环境、rollout服务、data manager、trainer分离，逐worker错峰同步；同时调采样数／轨迹长度、补历史成功轨迹并只训高熵step，显式B/C。效率表的training throughput单位是actions/min，不是tokens/s；GPU／环境利用率也不等step加速。OSWorld质量提升含数据策略，不能全归系统解耦；错峰worker更新不证明一条跨turn轨迹始终同policy。

本轮限身份／版本、摘要和上述局部机制／评估条件，未全文深读、拉取三项实现或验证其数值／性能。全文深读0、代码6条／4框架不变；下一步继续原生引用／会议盲区或未覆盖框架，不将多模态、VLA或仅使用RL的模型报告一律算原生系统。没有新增算力任务或M5专项交接。

### 5.23 并行线索去重与范围核对

2026-10-27接续核对：`49ac0b3`以116基线覆盖了`372b3e2`已入库的三项多模态条目及§5.22，并移除了同日原10:54日志；本轮恢复原文和三个矩阵行，同时保全该提交与M5后续日志，不以重置或删除另一侧成果解决冲突。原“新增8篇、累计124”的自报不作现行统计：八个名称中APRIL已经在§5.2；其余七个是新增接续线索，不是已完成范围审计的七项原生系统。主矩阵仍为119＝113 arXiv＋6会议，84直接候选＋15单列＋20 serving；若七项未来全部入库，唯一身份才是126，而不是124或127。

| 接续ID／短名 | 本轮 primary 身份／版本核对 | 机制与待核范围 | 当前计数状态 |
|---|---|---|---|
| [2509.18521 APRIL](https://arxiv.org/abs/2509.18521) | 既有§5.2的同一身份，已有v3 | partial rollout跨步续跑本身不证明改变分布；拒收／丢弃语义仍按原条目核，不自动升级成C | 重复线索，不新增 |
| [2606.01143 PrefixFlow](https://arxiv.org/abs/2606.01143v4) | v1 2026-05-31／v4 10-15；Training-Time KV Caching for Schedule-Level Prefix Reuse in LLM RL Training | 前缀forward缓存、suffix微批读KV并积累梯度、延迟prefix backward；训练更新侧范围，不直接当rollout生成提速 | 元数据／摘要已核，待范围入库 |
| [2608.25683 psRL](https://arxiv.org/abs/2608.25683v1) | v1 2026-08-26；Efficient Training for Agentic AI via Training-Time Prefix Sharing | update阶段前缀共享及KV管理；与生成侧prefix cache分开，训练端优化是否纳入本专项另核 | 元数据／摘要已核，待范围入库 |
| [2608.11152 Scheduling Mixed RL Rollouts](https://arxiv.org/abs/2608.11152v1) | v1 2026-08-11；Scheduling Mixed RL Rollouts Beyond Prefix Locality | MISA-T混合工作负载路由／准入考虑KV容量与驻留时间，声称保持消费混合；rollout-only与匹配训练迭代证据分开，尚未核实现 | 元数据／摘要已核，待范围入库 |
| [2608.14635 Belayer](https://arxiv.org/abs/2608.14635v2) | v1 2026-07-28／v2 08-18；Efficient Fault Tolerance for LLM Agentic RL Training | worker局部故障下shadow worker复用独立持有的权重／raw KV分配，按日志prefix重建request KV内容；环境checkpoint与上下文协调恢复。不能称原KV内容无条件存活或所有硬件故障可恢复 | 元数据／摘要已核，待故障模型与范围入库 |
| [2610.29426 BIRD-RL](https://arxiv.org/abs/2610.29426v1) | v1 2026-10-21；Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments | trajectory-scoped执行基础设施复用／隔离数据库副本，同时有四阶段curriculum；系统与recipe混合证据须拆分，不能把整体质量算纯B类收益 | 元数据／摘要已核，待范围入库 |
| [2608.22167 MCP-Universe RL](https://arxiv.org/abs/2608.22167v1) | v1 2026-08-23；A Framework for Training MCP Tool-Use Agents via Reinforcement Learning | MCP环境容器编排与staged rollout pipeline重叠工具等待；后端框架接入不证明版本／过滤契约，仍待源码 | 元数据／摘要已核，待范围入库 |
| [2610.34427 ScaleRLT](https://arxiv.org/abs/2610.34427) | 本轮abs读取失败（cache miss）；定向检索未返回匹配primary | 仅保留49ac的名称／ID线索；其looped transformer、exact logp、deferred rollout机制自报未独立复核，不沿用✅ | 身份及机制待核，不入主矩阵 |

这七项与主矩阵分账：六项只核元数据／摘要，一项尚未核到；不是七篇全文深读、七项性能实证或新的复现队列。arXiv摘要中的会议接收自报也不替代会议官网核验。后续先解决范围和版本证据，再决定并入§5.1–5.5还是保守单列；不能把“线索保留”写成“原生覆盖达标”。

撤回两条过强的空白判断：既有AReaL-DTA及serving缓存条目已触及训练／前缀复用边界，不能把PrefixFlow、psRL、Mixed Scheduling混成此前完全真空的同类生成机制；已有RobustRL等恢复条目，也不能据Belayer一个新名字宣布“容错为taxonomy全新维度”。此处只是扩展覆盖线索，不冻结taxonomy。TransferQueue本轮未系统检索所有文献渠道，原“无独立arXiv文献”也仅是上一轮未命中，不作不存在证明。

本轮优先修复真实接续损失；未启动原拟rLLM代码广扫，代码仍6条／4框架、全文深读0。未访问设备、变更M5专项、发起算力任务或写入社区。

## 6. Paper Deep Reads

（空。优先级建议：AReaL、HybridFlow/veRL、StreamRL、RollPacker、RhymeRL、BubbleSpec、StaleFlow——覆盖四层 + 三大机制家族。深读模板：问题/瓶颈 → 性能模型 → 机制 → 实验条件与增益（保留硬件/batch/长度分布）→ 与 taxonomy 对照 → 可迁移性 → 疑点。）

## 7. Code & System Landscape

**本地已有资产**（重要：不用重新 clone）：
- `/Users/jackiemac/workspace/verl`：veRL 源码（含 #8281 工作分支，近期深读过 `experimental/agent_loop/` 全链）；
- `/Users/jackiemac/workspace/UniRL-main`：UniRL 源码（train/stack 双栈结构已深读）；
- M5 服务器：verl-e2e 环境（4090-1，可跑 rollout 冒烟）。

veRL三条地图使用的只读检出：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/verl-7913`，HEAD固定`24f25b03aa4b54249a273655ebbcce06f484192b`，origin为`verl-project/verl`，核对时工作树干净。此快照不称最新main，不与上方历史工作区／论文发表年代混为一版；文件夹名不是接续旧PR的授权。

OpenRLHF本轮只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/openrlhf-rollout-scan`，origin为`OpenRLHF/OpenRLHF`，固定HEAD `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53`，工作树干净；未安装依赖或运行项目。两框架快照分别标版本，不把实现差异写成同年代／同配置的性能对比。

NeMo RL只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/nemo-rl-rollout-scan`，origin为`NVIDIA-NeMo/RL`，固定HEAD `a9d1a4bec45a28fd0d455519a2a979a37da2ac4a`，工作树干净。这里是SingleController路径，不与legacy async GRPO、NeMo-Aligner论文或其他快照拼为同一实现；未安装依赖或运行项目。

SkyRL只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/skyrl-rollout-scan`，origin为`NovaSky-AI/SkyRL`，固定HEAD `e0893f6eb878f4d396184cb14c4f9d27bbbb0aef`，工作树干净。只读统一包的fully-async／SkyRLGym路径，不同于SkyRL-Agent论文原版、默认同步trainer或模拟trainer；未安装依赖或运行项目。

**代码地图覆盖**：veRL三条路径见§8.2–8.4，OpenRLHF一条见§8.5，NeMo RL一条见§8.6，SkyRL一条见§8.7；下一批继续原生引用／会议盲区或补rLLM等未覆盖框架，不以单路径代表框架全貌。AReaL prompt_mean仍由M5单一负责，不在此代码广扫中接续。

## 8. Paper → Mechanism → Code Mapping

### 8.1 历史定位（未在本轮重核）

保留专题外已有veRL源码线索；以下80ee82bd时代定位不与§8.2的新快照拼成一套执行证据，也不计本轮代码广扫数量。

| 概念锚点 | veRL 代码位置（80ee82bd 代） | 状态 |
|---|---|---|
| rollout manager / worker | `verl/experimental/agent_loop/agent_loop.py`（AgentLoopBase、_run_agent_loop、asyncio.gather:719） | ✅ 已读 |
| prompt cap / truncation 契约 | 同上 `_cap_text_prompt_length`（#8281 修复点） | ✅ 已读 |
| 单模型训练栈生命周期 | `verl/workers/` + 本专题外已有笔记 | ✅ 已读 |
| sync barrier / weight update / sleep-wake | `verl/workers/fsdp_workers.py` 等 | 待读 |
| multi-sample / batch construction | `verl/trainer/ppo/` | 待读 |

（AReaL / vLLM / SGLang 等行待补。）

### 8.2 veRL权重同步与vLLM生命周期广扫

固定[veRL `24f25b03`](https://github.com/verl-project/verl/tree/24f25b03aa4b54249a273655ebbcce06f484192b)，只读默认CheckpointEngineManager→ActorRolloutRefWorker→vLLM ServerAdapter路径。trainer允许配置替换manager，其他rollout backend、PD、完整LoRA／quant／MTP loader及底层通信实现未全审；不能将本表推广为所有配置的协议。与HybridFlow论文的联系仅是工作流／参数重分配锚点，不是论文原版实现的逐项复现。

| 锚点 | 固定代码入口 | 静态可见内容 |
|---|---|---|
| trainer阶段与同步入口 | [ray_trainer.py:974](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L974)、[fit:1430](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1430) | 建manager时可用自定义类；初始化载checkpoint后sync，常规actor更新后再sync；生成后sleep |
| 同步协调与两条分支 | [CheckpointEngineManager.update_weights:506](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/checkpoint_engine/base.py#L506) | naive直接调actor；非naive协调abort、KV释放、拓扑、send／receive、finalize与恢复 |
| actor导出与直接加载 | [ActorRolloutRefWorker.update_weights:727](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/engine_workers.py#L727) | 非naive转checkpoint engine发送；naive按配置恢复weights、导出／加载，再恢复KV；adapter base-sync另有分支 |
| 两段传输的接口边界 | [CheckpointEngineWorker:355](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/checkpoint_engine/base.py#L355)、[ServerAdapter:209](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L209) | CE接收后交adapter；vLLM adapter断言named_tensors，再经bucketed IPC／shared-memory fallback交加载RPC；不是仅一条NCCL传输 |
| generation／KV控制 | [RolloutReplica:273](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/replica.py#L273)、[vLLMHttpServer:829](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L829) | replica向servers派发；clear、release、resume与pause／resume generation不是同一动作，且有node rank／free_cache_engine／RolloutMode门控 |
| sync调用如何完成 | [auto_await:97](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/ray_utils.py#L97) | 无running loop时asyncio.run；同步调用者处于loop中时线程执行后取结果；async调用者取coroutine需await，不是见async def就fire-and-forget |

默认manager的成功调用顺序（不是故障事务保证）：

```text
naive：actor.update_weights → 配置允许时resume weights → adapter加载／清缓存 → 配置允许时resume KV
其他backend：abort replicas → release KV → prepare／建拓扑 → actor发送＋rollout接收／加载（ray.get等待）→ finalize → resume KV → resume generation
```

`mode`／checkpoint backend与server的`RolloutMode`是不同开关。非naive分支的KV方法在`free_cache_engine=False`、非leader node或COLOCATED模式可能直接返回；vLLM的release实现实际先sleep再只wake weights，并非专门的KV释放API。`_resolve_sleep_level`对MTP、adapter等保留状态选择level 1，因此不能把同名sleep等同“所有权重均丢弃”。这些是源码条件，不是本机显存／吞吐观测。

adapter等待发送及可用的加载RPC后，有server的rank调用clear_kv_cache，再可选设置global_steps；server调用prefix／多模态／encoder cache清理接口，其中prefix reset带`reset_connector=True`。恢复KV内存不等旧KV内容可跨权重使用；global_steps是此路径的标记字段，不是张量一致性证明。server的abort异常会返回带error的dict，而manager聚合处未显式判该字段；仅完成await不能据此声称全故障场景已安全暂停，未做bug定性、恢复实验或修复提案。

验证限度：表中7个入口文件，加`verl/workers/rollout/base.py`、`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`及同目录`utils.py`，共10份与固定Git blob逐字节相同并通过AST解析，锚点行号由AST及原文定位；未导入模块、执行函数或跑原生测试。未确认底层vLLM实际版本、RPC兼容性、进程／设备完成性、恢复和off-policy修正。后续代码广扫扩展batch／old-logp／staleness数据契约；不将框架接口名直接视为论文性能／正确性验收，也不触及M5专项。

### 8.3 veRL采样批次与概率字段

沿用§8.2的固定commit，范围为`RayPPOTrainer.fit`、默认AgentLoop／vLLM输出到trainer的字段流，以及bypass loss入口；FullyAsync客户端只作独立分支对照。关键区别是**重排、概率重算、样本屏蔽、版本记录分别发生在不同位置**，不能从“async rollout”名称或一个step字段推导训练语义。

| 数据契约 | 固定代码入口 | 静态可见内容 |
|---|---|---|
| prompt组与行顺序 | [trainer:1481](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1481)、[DataProto.repeat:968](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/protocol.py#L968) | 先给原prompt设uid，生成输入与训练侧原批都按n交错重复，再union生成输出；不是每条response新建组ID |
| 等齐与重排 | [AgentLoopManager:1201](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/agent_loop/agent_loop.py#L1201)、[balance:1182](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1182) | manager按chunk派发后gather／concat，worker内部也gather；此fit等返回批再训练，并非完成一条立即更新。balance可改行顺序，DataProto同时重排tensor与non-tensor字段；GRPO按uid分组，不靠物理相邻 |
| 有效token与rollout概率 | [postprocess:698](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/agent_loop/agent_loop.py#L698)、[vLLM输出:678](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L678) | attention_mask含工具／观察token，response_mask另标LLM生成token且与padding mask相乘；请求logprobs时后端提取已生成token的值，经单回合loop对齐、右pad后汇总为rollout_log_probs，不是全词表logits |
| old-logp两种来源 | [重算:1290](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1290)、[bypass设置:1109](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/rollout_corr_helper.py#L1109) | decoupled在本批actor更新前调用训练侧compute_log_prob，padding-free往返后设old_log_probs；bypass缺rollout_log_probs即报错，否则直接赋给old字段并切换loss_mode，不只是省一遍等价forward |
| 校正与屏蔽位置 | [trainer:1648](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1648)、[helper:1013](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/rollout_corr_helper.py#L1013)、[bypass loss:2414](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/core_algos.py#L2414) | decoupled按配置在advantage前用old−rollout计算校正，RS改response_mask、IS可另加权重，不删DataProto行；bypass在loss内用当前logp−rollout重算，ppo_clip分支明确不再乘额外IS权重 |
| 版本标签的生成与携带 | [基础client:240](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/llm_server.py#L240)、[续跑client:346](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/workers/rollout/llm_server.py#L346) | server返回时读取self.global_steps，基础client补min／max字段；续跑client拼接分段token／logp，min取首个非None标签、max被末段覆盖，未作数值min／max聚合 |

decoupled的`old_log_probs`是本批更新前的训练侧锚点，`rollout_log_probs`是生成后端返回的token字段；即便名义权重版本相同，也不能先验认定二者逐值相等。bypass把两者合并为一个分母来源，当前policy仍在更新；启用参考policy时的`ref_log_prob`则另算，不与任一旧概率字段混用。这里尚未审训练engine的温度／采样变换与后端实际logprob定义，不由字段名认证它们等于最终截断采样分布的精确密度。

默认值须按入口记录：[ppo_trainer组合的rollout_correction YAML](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/config/algorithm/rollout_correction.yaml#L5)中`rollout_is=null`、`rollout_rs=null`、`bypass_mode=false`，而[直接构造RolloutCorrectionConfig](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/config/algorithm.py#L178)的`rollout_is="sequence"`。不能只见helper被调用或存在rollout_corr指标，就说已应用IS／RS；本轮未运行Hydra组合，不把配置片段当具体作业的有效配置。

RS屏蔽不等于减少已做的生成：此fit已完成采样和reward，decoupled校正在advantage前改mask；bypass的effective mask在loss内生成。两者影响训练贡献的位置不同，不能互称等价或把被屏蔽token说成未生成。balance仅为重排也不保证优化轨迹不变：trainer源码明确提示mini-batch划分可能变化。

默认trainer创建client未传子类，`get_client()`默认返回基础LLMServerClient，不能把FullyAsync的abort续跑当此fit默认路径。续跑分支的min／max标签只有在版本单调等条件下才可解释为版本跨度；它们不是逐token版本向量，也不是lag上限检查。tool loop后续turn只条件更新max字段，worker将这些extra字段带入non_tensor_batch；所读默认fit／校正入口未按min／max字段作准入或拒绝，校正依据的是logp差，不是整数step差。server标签在返回时读取，也不是每token权重身份或不可中断请求的证明；其他fully-async trainer／staleness调度器留待单独广扫。

验证范围：上表涉及文件，加single_turn_agent_loop.py、tool_agent_loop.py、trainer/config/algorithm.py及3份配置YAML，共13份固定Git blob一致（10份Python只作AST解析、3份YAML仅字节核对）。阅读入口函数与相关调用段，不称13份全文读完；未导入模块、构造DataProto／配置或执行loss／模型。原生异步trainer的队列与staleness接续见§8.4；跨框架对照与文献扩展继续，不为静态地图新增GPU任务。

### 8.4 veRL异步样本预算与消费边界

同一固定commit的`experimental/fully_async_policy`是独立入口，不能将它的语义套回§8.3默认fit。[入口runner](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_main.py#L186)分别启动Rollouter和Trainer；rollouter使用FullyAsync客户端，默认同步路径的整批等待变为**每个prompt组完成后入队，trainer收够若干组再更新**。动态资源调度在本节配置片段中关闭，其控制器／policy不在本轮审阅范围。

| 边界 | 固定代码入口 | 静态行为与单位 |
|---|---|---|
| 一个队列sample是什么 | [prepare:42](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/detach_utils.py#L42)、[生成并发布:991](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_rollouter.py#L991) | gen_batch_size要求1；单prompt先重复n条response，全部生成结果封进一个RolloutSample，共用uid再序列化入队。预算／队列条目不是单条response或token |
| 预取、并发与预算 | [预算:491](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_rollouter.py#L491)、[processor:902](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_rollouter.py#L902) | pending_queue预取上限128；active_tasks限制同时进行的prompt组。staleness_samples在从pending取出时递增，暂停条件读此计数或MQ长度，并非检查每条样本的版本差 |
| 参数同步后如何恢复 | [reset_staleness:594](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_rollouter.py#L594)、[暂停检查:1142](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_rollouter.py#L1142) | reset把计数重设为active任务数＋MQ条目数并发resume信号；不是清零旧轨迹、清空队列或重新生成旧内容。背压暂停主要挡新提交，权重同步中的abort／resume另沿§8.2和客户端路径 |
| 满队语义 | [MessageQueue.put_sample:55](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/message_queue.py#L55) | FIFO deque满时先popleft旧条目，再append新条目并返回False；不解码样本、不比较policy版本。不是拒收新样本，也不是阻塞put等待空位 |
| 训练消费与尾批 | [get_samples:375](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L375)、[assemble:84](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/detach_utils.py#L84) | 按FIFO取够required_samples个组，反序列化concat并可balance；正常成功时为R×n行。先遇None且不足R则返回终止，不训练这份不足批；此函数不按版本差筛条目 |
| 两个训练时钟 | [fit_step:536](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L536)、[local step:676](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L676) | 每次取批／训练调用推进local_trigger_step；累计K次后current_param_version递增并触发权重同步。K不是token数，也不等引擎内部optimizer.step次数；critic warmup还可跳过actor更新 |

设`R=ppo_mini_batch_size×require_batches`（这里计prompt组），`K=trigger_parameter_sync_step`、`s=staleness_threshold`，源码设`M=int(R×(1+s)×K)`，MQ容量同为M，并发上限为`min(活动server数×concurrent_samples_per_replica, M)`。因此s是样本预算的比例参数，不能解释成秒数、逐token年龄或“最多落后s个版本”；s=0也不意味着队列／active任务为零。pending、active、MQ和消费端批次是不同位置，MQ长度本身不是全链路内存／在途总数。

满队返回False表示**此次put发生旧条目淘汰，新条目仍已保留**。rollouter据此增加`dropped_stale_samples`，但这个名称不证明被淘汰条目经过陈旧性判定。满队淘汰或尾批舍弃若触发，会改变实际进入训练的集合，应与纯重排／背压分开登记；每步组批与更新顺序的影响仍须另核，不能将整条异步通路先验归C或排除C。这里确认的是分支语义，不断言当前配置必然满队或测得了丢弃率。

[版本统计](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L1033)使用每行`max_global_steps`作为trajectory_param_versions，按`current_param_version-v>=1`累计stale行数；assembly另以首末标签绝对差统计partial span。这延续§8.3的首末标签限制，不是逐token最大年龄或严格版本准入。K内当前训练参数可继续改变而rollout参数版本标签不变，故参数同步时钟、训练调用时钟与概率差异仍须分开。

[异步配置片段](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml#L9)取s=0.1、K=4、require_batches=1、partial_rollout=True并设bypass_mode=True，覆盖§8.3基础配置的bypass=false。若改用decoupled，[old-logp覆盖方法](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L654)在K窗口首批保存训练参数，后续批临时恢复该锚点求概率再恢复当前参数；不能沿用默认fit“每批以当前训练侧参数为old”的概括。这里只读保存／恢复调用，未验证CPU快照、异常恢复、配置组合或数值一致性。

验证范围：fully_async_main／rollouter／trainer、message_queue、detach_utils、separation/ray_trainer和共享llm_server共7份Python，与上述YAML共8份固定Git blob一致；Python仅AST解析。未导入框架、创建队列／Ray actor或执行并发与设备函数，不给出严格年龄上界、死锁／故障定性、吞吐或收敛结论。后续转另一原生框架做数据生命周期对照，并继续文献扩展；动态调度、异常／恢复和真实队列轨迹留待独立范围，不以CPU替身或新GPU任务补造运行证据。

### 8.5 OpenRLHF异步批次与概率锚点

固定[OpenRLHF `dc2a7ad3`](https://github.com/OpenRLHF/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53)，沿CLI→PPOTrainerAsync→SamplesGenerator→RemoteExperienceMaker→PPO loss入口广扫。CLI只有`train.async_enable`才选异步trainer；其默认False，queue_size默认1、partial_rollout默认False、IS correction默认off。下述是显式启用async后的路径，不是所有OpenRLHF作业的默认执行。

| 边界 | 固定代码入口 | 静态行为 |
|---|---|---|
| 队列与生成预算 | [异步入口:269](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L269)、[生产:100](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L100)、[消费:202](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L202) | 结果Queue与slot Queue容量均Q。生成前取slot，结果阻塞put；trainer取payload后、train_step前归还slot。空生成自行还slot，eval另走消息分支；所读通路无满队淘汰旧payload操作 |
| 队列外的生成缓冲 | [generate_samples:89](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/samples_generator.py#L89) | chunk为rollout.batch_size×n；可一次生成更大的vllm_generate_batch_size，剩余Experience留_sample_buffer，后续调用直接取。尾部返回up to chunk而非统一丢弃不足批；下游训练覆盖另核 |
| 筛组与补抽 | [_generate_vllm:140](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/samples_generator.py#L140) | 按prompt派发，等待ready ref后将该prompt响应组连续收集；启用dynamic_filtering且分数齐全时，用组均分的开区间筛整组，失败后取新prompt补抽。这改变训练集合，与队列背压不同 |
| token与组身份 | [response转Experience:243](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/samples_generator.py#L243)、[组统计:238](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L238) | observation_tokens含上下文，action_ranges另建action_mask；mask和rollout logp按next-token位置截取。split前赋index，reward按index还原原列表顺序后reshape(-1,n)取组，不是veRL的uid映射 |
| PPO概率的三种来源 | [make_experience:113](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L113)、[actor training_step:259](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/ppo_actor.py#L259)、[PolicyLoss:174](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py#L174) | Experience.action_log_probs由训练actor先forward重算，之后作为old_action_log_probs；更新forward产生new，rollout_log_probs另存vLLM字段，base_action_log_probs为reference。PPO校正启用时另取detach(old−rollout)，并非将old直接替换成rollout |
| 生成与权重同步 | [broadcast:255](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L255)、[vLLM pause:130](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/vllm_engine.py#L130) | 普通async在generate_samples与broadcast之间共享锁，训练计算仍可与生成重叠；partial模式生成不取该锁，broadcast调用pause_generation(mode="keep")／resume。锁／接口调用不认证逐token权重身份或故障恢复 |

Q限制结果队列及slot预算，**不包括当前已取走的训练批，也不覆盖_sample_buffer中一次超额生成的全部响应**。slot携带的global_step在所读代码用于eval时机，并非每条Experience的policy版本；不能由Q=1推出只落后一个optimizer更新或绝对on-policy。普通生成锁也不能使缓冲中的旧响应自动变成最新权重的样本。[CLI约束](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/cli/train_ppo_ray.py#L711)禁止async与vLLM sleep共用，partial要求async，超额生成也要求async；这与veRL的free-cache／RolloutMode条件不是同一套开关。

概率与集合变化分开：这里PPO的mask式IS校正在loss上乘零／非零系数，最后仍按传入action_mask汇总，不是veRL decoupled在advantage前修改response_mask的同一操作。两边不能仅凭“mask／IS”名称互称等价。动态筛组在生成阶段已产生被弃响应并可能增加补抽，不能把训练保留数直接当生成工作量；原n响应组的完整性也依赖executor返回契约，本轮未认证自定义agent任意输出。名为replay_buffer的actor缓冲在[正常fit返回前:598](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/ppo_actor.py#L598)会clear，名称本身不证明跨训练步回放旧轨迹。

对照§8.4：veRL队列条目为单prompt的n响应组，完成后入队，消费R组；此OpenRLHF结果payload则是一个rollout训练chunk，且其内部还有超额生成缓冲。前者满队分支淘汰旧条目，后者所读生产路径使用阻塞put／slot；两种容量数值不可直接横比为相同在途轨迹或陈旧性上限。PPO概率字段、分组身份和partial开关均须按各自调用链解释，不把软件快照直接等同对应论文版本或性能证据。

验证范围：ppo_trainer_async／ppo_trainer、samples_generator、experience／experience_maker／replay_buffer、ray/ppo_actor／vllm_engine、models/loss和cli/train_ppo_ray共10份Python与固定Git blob逐字节一致并作AST解析；阅读入口和相关函数，不称10份全文或完整框架读完。未导入模块、安装依赖、解析有效作业配置、运行原生测试／模型／队列／设备，未验证底层vLLM实际版本与自定义agent。下一步继续原生文献扩展，或补第三框架的队列／概率生命周期，不将静态差异当性能排名、bug结论或新GPU任务。

### 8.6 NeMo RL准入 选择与样本所有权

固定`NVIDIA-NeMo/RL a9d1a4be`，从[SingleController入口:80](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/examples/run_grpo_single_controller.py#L80)追到两条pump、sampler与TQReplayBuffer。示例[SC YAML:25](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/examples/configs/grpo_math_1B_megatron_single_controller.yaml#L25)采用in_order、lookahead=1、整步groups才stream、inflight=每步prompt数、buffer=64；这些是示例覆盖，不当所有recipe的有效配置。guide将SC标为beta；本节只读局部调用链，不是训练／恢复验收。

**生产准入和消费选择不是同一个窗口。** 令v为当前trainer版本、s为buffer记录的start_weight、w为配置窗口；内置四种策略的静态条件如下（custom不在覆盖内）。共同[选择收尾:310](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L310)按min／max和group_multiple裁齐，未对齐尾组留待后续，然后claim，不在这里清TQ张量。

| sampler | 生产准入 | 已完成组的选择与淘汰 |
|---|---|---|
| [windowed:361](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L361) | 无trainer版本gate，仍受容量／并发背压 | 选max(0,v−w)≤s≤v且ready的组，可freshest-first；默认evict移除低于下界的ready组 |
| [ready_first:523](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L523) | gate限制dispatch批次的lookahead | 所有s≤v且ready组均可选，无s下界；该sampler的evict返回0，不按年龄淘汰长尾组 |
| [weight_fifo:578](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L578) | gated lookahead | 在窗口内先找最小s（含未ready槽），仅消费该版本ready组；较新ready组不能越过它。默认evict按s下界 |
| [in_order:629](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L629) | dispatch序号作为target_step；lookahead可按PPO warmup调宽 | 选target_step=v且ready，不按s窗口选；evict仅清target_step<v的ready组，不早清未来target |

这四种是sampler实现，不是四种均支持任意算法：固定快照的[PPO校验:1329](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller_utils/config.py#L1329)仅允许in_order／ready_first。

[_GatedSampler:498](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/staleness_sampler.py#L498)等待dispatch_index小于trainer_version+gate_window后才递增并盖章；这限制的是批次准入游标。尤其ready_first的`max_staleness_versions`不能解释成消费样本的硬年龄上限；“sampler不按年龄淘汰”也不等全系统绝不丢组，错误预算／过滤另有路径。[配置校验:1403](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller_utils/config.py#L1403)对ready_first显式要求importance sampling，并限制force_on_policy_ratio与CISPO组合；校验开关不是数值收敛证明。

**槽、元数据与张量的寿命分开。** [rollout _launch:2471](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L2471)先取buffer permit、再取inflight semaphore、再等refit派发gate。buffer permit覆盖未消费组而非仅ready组，token-capture生成permit可在finalizer完成前归还。[reserve:1176](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/replay_buffer.py#L1176)建ready=False槽；[claim_for_training:1529](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/replay_buffer.py#L1529)将ready组从可选索引转入training claims，但保留TQ数据和checkpoint元数据。正常训练完成后的[cleanup:3600](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L3600)才清数据、释放claims及buffer permit；因此选择后的buffer长度不等全部驻留数据或端到端在途量，不把claim当立即free。持久化／故障重试是否完整仍需独立审计。

**版本标签按生成分支解释。** 非capture的[generate_and_push:2010](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/experience/rollout_manager.py#L2010)在尝试开始读manager版本reserve，run_rollout结束再读end版本后commit；首末标签不是逐token权重证明。capture的[commit_finalized:1383](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/async_utils/replay_buffer.py#L1383)则将输入group_min_wv／group_max_wv写作start／end，本轮未追finalizer如何形成每个call的版本。[同步入口:5450](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L5450)关闭新派发gate，经backend pause／同步／resume再开放；native与Gym的过旧inflight abort分支不同，不能概括成所有backend统一清空／取消。manager标签在[refit调用点之后:3774](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L3774)更新；PPO另有early-refit及critic-warmup分支，trainer步号不自动等每次actor optimizer更新或实际权重变化。

与§8.4／§8.5的区别：veRL所读队列会条件淘汰旧条目，OpenRLHF所读路径用slot／阻塞put且另有响应buffer；SC把准入、按版本或target选组、training所有权分为独立阶段。不能拿三者的“queue size／staleness”数值直接比较，也不把windowed筛选、动态reward过滤与背压混作同一操作。验证限6份Python blob一致／AST解析及引用行号核对（entry、controller、sampler、replay、rollout_manager、config），另读guide和示例YAML；未导入框架、安装依赖、运行Ray／TQ／原生测试／模型／设备，无性能或故障结论。全文深读0，代码路径累计5条／3框架；后续继续原生补库或补尚未覆盖框架，不为静态差异创建GPU任务。

### 8.7 SkyRL异步组准入与消费

固定[SkyRL `e0893f6e`](https://github.com/NovaSky-AI/SkyRL/tree/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef)，沿[fully-async示例入口:15](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/examples/train/fully_async/main_fully_async.py#L15)→FullyAsyncRayPPOTrainer→SkyRLGymGenerator→训练输入／loss与同步调用点广扫。[配置:1096](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/config/config.py#L1096)的fully_async.enabled与sample_full_batch默认False；本节是显式启用的真实trainer路径，不把独立simulate_training模式或默认同步路径混入证据。

| 环节 | 固定代码入口 | 静态可见内容 |
|---|---|---|
| prompt组与完成边界 | [生产:922](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L922)、[generate:957](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/generators/skyrl_gym_generator.py#L957) | 每worker取一个prompt，展开n_samples_per_prompt为同uid组；generate以gather等待组内agent loops并保序。完整组完成后put_nowait入池，QueueFull抛断言，不淘汰旧组或阻塞put |
| 容量准入 | [_compute_capacity:181](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L181)、[初始化:339](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L339) | 同时限制running组和超前训练的总组数；结果池maxsize=B(S+1)，不是逐trajectory或token预算。取出一组不直接返还超前预算，训练后推进global_step再通知容量 |
| 消费与长尾 | [收集:823](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L823)、[转换:1013](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L1013) | finish-time FIFO收B个组，不等最先派发的B组；组内仍等慢轨迹。step-wise组可展开不同数量的turn行，按response_ids数复制uid。超过S仅warning并计数，仍拼入训练 |
| 筛组与不足尾批 | [过滤返还:208](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L208)、[消费循环:823](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L823)、[epoch耗尽:552](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L552) | sample_full_batch要求zero_variance_filter；丢组后accepted减1、filtered加1，释放预算并继续补组。epoch耗尽则丢不足B的已保留组；这些uid记为filtered／consumed，恢复时不重抽。它改变训练集合，不是单纯背压 |
| 概率锚点 | [构造校验:393](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L393)、[forward门控:1347](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/trainer.py#L1347)、[rollout_is:868](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/backends/skyrl_train/utils/ppo_utils.py#L868) | async限制依赖重算old的loss，CISPO按anchor判断；rollout_is断言有rollout logp并以new−rollout算ratio。只有相应loss、已有rollout logp、KL reward关闭且off-policy correction关闭时才跳过policy forward；不能统称async总不重算 |

**总体超前量不等每组硬年龄。** 令B=policy_mini_batch_size（prompt组数）、S=max_staleness_steps、v=当前正在训练的global_step、A=accepted（含已训练与待训练，过滤时减回）、R=running、W=并发组上限；准入容量为`min(W−R, (S+v)B−A−R)`，正值才提交。`global_step_when_scheduled`在取得slot后、调用generate前记录；消费计算`global_step−scheduled`，并不检查引擎实际token权重版本。配置docstring的`j−i≤S`不能覆盖代码明确允许的长尾超限；也不能与NeMo windowed的消费年龄筛选混称同一保证。全异步在epoch内运行，跨epoch会等待／重置控制状态。

**过滤与计量口径分别保留。** sample_full_batch在收集阶段丢组补组；即使此开关关闭，[共同postprocess:1154](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/trainer.py#L1154)对response-level reward仍可将零方差组的loss_masks置零，token-level reward另走分支。不能把mask置零与collector丢组／补组视为同一操作。正常收齐B组后的转换会将同轮丢弃的零方差组仅并入headline奖励／生成指标，训练输入另用保留组；consumed uid数也不等训练组数，后者扣除filtered。这里不认证checkpoint恢复、自定义generator输出或过滤后的数值收敛。

**步号、prefix缓存盐与轨迹权重不是同一身份。** [cache salt:315](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/generators/skyrl_gym_generator.py#L315)按engine weight_version与模型名生成，每次generate只捕获一次，不以trainer global_step代替；这是prefix复用隔离，不证明跨turn／in-flight轨迹始终同权重。[同步调用:783](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/backends/skyrl_train/workers/worker_dispatch.py#L783)的non-colocated非LoRA普通分支pause→broadcast／finish→resume，delta另将pause交内部；KV-offload又有独立分支。不能把外层pause调用直接当所有backend、adapter、KV和token一致性证明。

验证限8份Python文件与固定Git blob一致、AST解析及本节行号核对：示例entry、fully_async、config、generator base／gym、trainer、worker_dispatch、ppo_utils；未导入SkyRL、Torch、Ray、vLLM或运行原生测试／模型／设备。文献仍116、深读0，代码累计6条路径／4框架；不将代码快照等同论文原版或静态差异等同性能排名。后续继续原生引用／会议盲区或尚未覆盖框架，具体恢复／token版本追踪另核，不为本次代码阅读创建GPU任务。

## 9. Serving → Rollout Transfer

三问框架（每个 serving 工作都过一遍）：① serving 里解决什么 bottleneck？② 哪些机制可直接迁移？③ 进入 rollout 后 objective/scheduler semantics/correctness 约束怎么变、是否需要 RL-aware redesign？

初步判断（待逐项深读验证）：
- continuous batching：直接迁移，但 sync RL 下"批内最慢者"支配 → 需 L3 重调度补偿；
- prefix caching：rollout 中 multi-sample 共享 prompt → 收益天然放大；partial rollout 场景与版本管理耦合；
- speculative decoding：不能由serving加速比直接推导RL收益；policy变化下draft适配、接受率与成本需重评。原先“不可原样迁移”的绝对判断尚未成立，采样正确性与draft效率分开，具体机制是否可直接复用待深读；
- chunked prefill：rollout 的 prompt 占比与 serving 不同（reasoning 长 decode），需重估；
- prefill/decode分离（DistServe系）：分开的是推理内部两个阶段，不能直接等同trainer/rollout分离；进入RL后的目标、分离粒度与资源配比仍待研究。
- 多轮agent缓存／准入：分层KV、TTL、agent并发及workflow缓存分别处理驻留／重载／压力／跨调用依赖；迁到RL还需完整GRPO组、policy版本与staleness条件，不能把cache eviction等同删样本，见§5.15。

## 10. Workload-specific Findings

（空。Reasoning / Agentic / VLM 三分——Agentic 已有专门簇（5.4），注意 environment bubble 的表征工作。）

## 11. Cross-layer Findings

（空。登记跨层耦合反例的地方：如 BubbleSpec 同时是 L1 机制 + L3 气泡利用。）

## 12. Evaluation & Benchmark

证据类型登记（摘要级，未作实验表抽取）：[Iso等2604.26779v1](https://arxiv.org/abs/2604.26779v1)将8B同步RL rollout实测与235B异步组合的performance simulator预测分开报告；不能把后者写作大模型端到端实测，也不把rollout吞吐与整步／端到端时间混成同一指标。

后续逐篇登记makespan、tokens/s、step time、bubble ratio及长度分布，同时区分作者实测、模拟预测与本项目复现，当前不建立跨论文加速比排名。

固定源码的计时口径（veRL `24f25b03`，静态）：[fit的gen计时块:1510](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1510)包住generate_sequences、sleep_replicas及条件性的profile调用；[update_weights计时块:1715](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/ray_trainer.py#L1715)包住整个manager同步流程，分支见§8.2。因此前者不直接等纯decode／kernel时间，后者不直接等链路传输时间。未来提取benchmark须注明计时边界、backend／free-cache／profiler配置；本轮没有时间测量或拆分数值。

同快照fully-async的[gen块:643](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L643)则包住队列取批、反序列化／assembly及样本指标收集，不是rollouter的生成计算时间；不能与上段同名gen直接横比。[param_sync:690](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/experimental/fully_async_policy/fully_async_trainer.py#L690)计时块之后才调用reset_staleness，故其数字也不是整次生产恢复流程的总时长。性能评估应同时记录prompt组／response／token单位、队列淘汰与不足尾批，而非只看保留样本吞吐。

OpenRLHF `dc2a7ad3`的[异步生产／消费计时](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L129)中，timing/generation包住generate_samples调用，既可能含筛组／补抽，也可能仅取_sample_buffer，并不总对应一次新的模型生成；timing/step_total为消费循环墙钟，包含队列等待，不能把重叠的generation／训练分项相加当端到端。dynamic filtering的filter_pass_rate在源码中算`len(experiences)/prompts_consumed×100`，分子为响应、分母为prompt，未校正n前不当prompt通过百分比，缓冲直出时还可能为None。本轮没有测量任何耗时／通过率。

NeMo RL SC `a9d1a4be`的[exposed_generation:3010](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L3010)包住evict、选组和饥饿轮询，不是rollout引擎生成总算时；[weight_sync:3748](https://github.com/NVIDIA-NeMo/RL/blob/a9d1a4bec45a28fd0d455519a2a979a37da2ac4a/nemo_rl/algorithms/single_controller.py#L3748)在colocated且为save延后refit时可进入空计时块，其他分支才调用同步生命周期。组数／版本／claim寿命见§8.6；计时名称不能替代执行分支和生成工作量，本轮无计时实测。

SkyRL `e0893f6e`的[wait_for_generation_buffer:843](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L843)覆盖取组、等待和条件筛组／记账；group_completion_time则在取得准入slot后包住单组generate调用，不含此前容量等待。两者不等纯decode时间。[sync_weights:604](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/train/fully_async_trainer.py#L604)包住dispatch全调用；名为sync_weights_only_transfer的内层计时在[worker_dispatch:809](https://github.com/NovaSky-AI/SkyRL/blob/e0893f6eb878f4d396184cb14c4f9d27bbbb0aef/skyrl/backends/skyrl_train/workers/worker_dispatch.py#L809)包住broadcast加finish，仍不等链路字节传输耗时。生成与训练重叠，分项不简单相加；本轮没有计时实测。

## 13. Research Gaps / Ideas

（Round 1 初步观察，待 corpus 收敛后修正）：
1. **分布改变与纯路由／重排的边界待核**：DARTS主动塑形、InfoTree选择状态／预算，WAR不再作为C类例子；ExpThink已核为reward塑形边界，非直接筛选轨迹；QwenGyre路径去重的loss／训练集合语义仍待核。对分布／收敛影响的评估仍待逐篇阅读，不预设“普遍缺乏严格评估”；
2. speculative-for-RL已出现多个候选，但条目数不证明方向饱和。统一性能模型（acceptance rate × staleness × 长尾交互）是待探索问题，不据此否定其他drafter方案的研究空间；
3. cap/长度类边界条件的系统化处理（如 cap=1 的 -0 陷阱、偶奇 cap 分配）在现有论文中未见讨论——来自 verl #8281 实战的微观缺口；
4. 多模态机制继续分开：§5.22补MR-SP的编码／embedding复用与Relax的字段流水／encoder布局，不等Rollplex的跨阶段GPU空分；DART等视觉agent另有环境／采样语义。不能从某一机制候选少，推出整个多模态系统文献空白；后续继续查相近机制和不同工作负载。

## 14. Reproduction Candidates

（空。登记模板：为什么值得复现 / repo / model / hardware / dataset / expected cost / key metric / stopping condition。**当前不启动 GPU 实验**。已有可借力资源：M5 4090-1 verl-e2e 环境、本地 veRL 源码。）

## 15. Open Questions

1. "Rollout ≠ serving" 假设的边界：哪些 serving 优化在 rollout 下 objective 完全不变？（找反例）
2. makespan 最优 vs on-policy 性的权衡：C 类改分布的工作，train-time 收益能否抵消分布偏移的收敛代价？有没有论文量化过？
3. taxonomy 是否需要第五层（environment/sandbox 层）？agentic 簇的 environment bubble 是否够独立成层？
4. Rollplex 的 VLM 跨阶段共享是否适用于文本-only 场景（colocation 的空间共享新维度）？

## 16. Changelog / Research Log

- **2026-10-27（M1并行接续保全与纠错）**：恢复372b3e2的三项多模态矩阵行、§5.22、§13说明及原10:54日志／回看；保全49ac0b3与3e5d82d日志。主矩阵恢复119／113 arXiv＋6会议，84＋15＋20；八名称中APRIL重复，余七项另入§5.23，其中六项primary元数据／摘要已核、一项未核到。训练更新侧prefix共享与生成侧分开，容错／缓存“全新真空”判断撤回，未新增深读／代码／设备或社区动作。下述49ac历史自报保留以供追溯，其124计数、八篇全新及空白判断已由本条纠正，不作当前结论。

- **2026-10-27（M1 KV/前缀与容错盲区补库）**：三主题缺口检索（TransferQueue/KV-cache/rLLM），新增 8 篇（§5.22）：APRIL、PrefixFlow、psRL、Mixed-RL-Scheduling、Belayer、BIRD-RL、MCP-Universe-RL、ScaleRLT；累计 124 身份。两个发现：训练期 KV/前缀复用此前为真空区；容错为 taxonomy 候选新维度。TransferQueue 无独立 arXiv 文献（框架 only，记 §7 代码侧）。

- **2026-10-27（M1多模态执行与视觉agent补库）**：新增Relax v2／Long-RL v4（MR-SP）／DART-GUI v1，累计119身份（113 arXiv＋6会议），保守账为84直接候选＋15单列＋20 serving。§5.22区分字段流水、编码／prefilling复用与GUI解耦／数据C策略；保留Relax摘要on-policy与正文off-policy条件不一致，Long-RL forward计时不当全步，DART actions/min不当tokens/s。§13修正覆盖问题表述，原116行和代码地图保留。深读0、代码6条／4框架，无作者实现／设备／M5专项／社区写入；继续原生引用／会议或未覆盖框架，不以数量收束。

- **2026-10-27（M1 SkyRL组准入与消费广扫）**：只读固定`e0893f6e`，§8.7补组完成FIFO、总体容量公式与长尾超限只warning、零方差补组／不足尾批丢弃及uid记账；rollout／重算old概率和engine版本cache salt分别标语义。§12补外层同步／broadcast加finish、组生成／收集等待的计时边界。8 Python blob／AST与引用核对，无依赖／原生执行／设备／M5专项／社区写入；文献116、深读0、代码6条／4框架。下步仍原生引用／会议或未覆盖框架，恢复和逐token版本另核，不据静态差异排名。

- **2026-10-27（M1弹性与跨域补库）**：新增RLBoost v3／ECHO-2 v6／PEARL v1／ARL-Tangram v1，累计116身份（110 arXiv＋6会议），§5.20更新81直接候选＋15单列＋20 serving。§5.21分开抢占续跑、过旧丢弃、同trace执行比较及action外部资源；价格估计不当账单或wall-time优势，RLBoost+不当原版。原112行与代码地图保留，深读0、代码5条／3框架不变；无设备／M5专项／社区写入。下步仍原生引用／会议或未覆盖框架，不以81自动验收收束。

- **2026-10-27（M1 MTP-RL落实与覆盖口径）**：ACL Findings官方身份落实最后种子MTP-RL，与Bebop分开；累计112身份（106 arXiv＋6会议）。§5.20逐项单列15条范围／混合证据，92 RL相关暂为77直接候选＋15，不当最终分级／纯原生验收；serving20独立。初始种子身份闭环不当引用检索收敛。仅矩阵复核与MTP元数据／摘要，深读0、代码5条／3框架不变，无设备／M5专项／上游写入。下步补原生引用／会议盲区或未覆盖框架，不以总量达标或凑3篇收束。

- **2026-10-27（M1两项种子与预算边界补库）**：Weave以OSDI'26身份落实、Online Draft Co-Training落实2609.07108v1，另增CERO／SAO；累计111身份（20 serving／边界＋91 RL相关边界；106 arXiv＋5会议记录）。§5.6移除两项待核，MTP-RL保留；§5.19拒绝2602.11143机器人错号，分开跨job效率、draft算子／完整训练与采样预算／loss改变。深读0、代码5条／3框架不变，无设备／M5专项／上游写入。下步原生引用／会议、剩余种子或未覆盖框架，不按总量收束。

- **2026-10-27（M1更新语义与delta refit补库）**：新增NeMo-DCR／ThunderSyncRL／PipelineRL／PACE四篇，累计107身份（20 serving／边界＋87 RL相关边界；103 arXiv＋4会议记录）。§5.18分开固定policy的gradient streaming与一次更新、序列内换权重／旧KV、池拒收C及bit-exact dense refit；传输基准／计时窗不当端到端训练。身份／摘要和局部机制，深读0、代码5条／3框架不变，无设备／M5专项／上游写入。下步继续原生引用／会议／种子，或未覆盖框架，不以过百结束广扫。

- **2026-10-27（M1 NeMo RL准入与所有权广扫）**：文献仍103，代码路径5条／3框架。固定NeMo RL `a9d1a4be`，§8.6分开四sampler的生产gate、start_weight窗口、target_step与ready-first长尾；reserve／commit／claim／cleanup及capture／非capture版本标签不混用。§12补exposed_generation非引擎算时、延后refit空计时块。6 Python blob／AST与源码链接核对，无框架／原生测试／设备，非故障或性能验收；不改M5专项、不向上游写入。下一步继续原生引用／会议／种子扩展或未覆盖框架，capture finalizer和恢复另核。

- **2026-10-27（M1同步与运行开销补库）**：新增SparseRL-Sync／PULSE／FP8-RL／PivotRL／RobustRL／Rollout Infrastructure Tax六篇，累计103身份（20 serving／边界＋83 RL相关边界；99 arXiv＋4会议记录）。§5.17区分post-cast无损与FP32更新、两通信通道、policy量化与draft、故障条件收益与固定命令worker-hours外推；参考文献占位项不继承，submitted不当accepted。身份／摘要和局部范围核验，深读0、代码4条／2框架不变；无设备／M5专项／上游写入。下步继续原生扩展或第三框架，不因过百结束广扫。

- **2026-10-27（M1 OpenRLHF跨框架广扫）**：文献仍97，代码路径累计4条、涉及2框架。只读克隆固定OpenRLHF `dc2a7ad3`，§8.5对照slot／阻塞队列与veRL条件淘汰，补_sample_buffer、筛整组补抽、index分组与PPO old／rollout／ref来源，普通锁与partial pause分开；§12记录缓冲直出计时和响应／prompt分母。10 Python blob一致／AST通过，无依赖安装、原生运行或设备；不作性能／算法正确性／故障判断，不改M5专项。后续原生文献扩展或第三框架生命周期。

- **2026-10-27（M1 fully-async队列广扫）**：文献仍97，代码路径累计3条同版本veRL路径。§8.4补prompt组预算M=int(R(1+s)K)、pending／active／MQ区别、FIFO满队淘汰旧条目而保留新条目、尾批终止、local step／同步版本两时钟及old锚点覆盖；stale指标不当硬版本准入。§12区分异步gen取批计时与默认gen。8源码／配置blob一致、7 Python AST通过，未创建队列／框架／设备；不定性bug、测丢弃率或加GPU任务。后续跨框架对照与原生文献扩展。

- **2026-10-27（M1批次与概率字段广扫）**：文献仍97，代码路径累计2条（同一veRL `24f25b03`，非2框架）；§8.3登记uid重复／gather等齐／balance、rollout与重算old概率、decoupled／bypass校正位置和版本标签。YAML关闭IS而配置类默认sequence的入口差异单列；RS屏蔽不省已生成工作，首末step标签不当逐token版本或准入保证。13源码／配置blob一致、10 Python AST通过，无框架／设备执行。后续异步trainer准入与跨框架对照、继续原生文献，M5专项不动。

- **2026-10-27（M1固定源码广扫）**：不增文献，保持97身份；新增§8.2固定veRL `24f25b03` 默认manager→vLLM的权重同步／生命周期地图，§12记录gen与update_weights计时并非纯decode／链路传输。naive／非naive及server mode／free-cache门控分开，KV内存恢复不当缓存内容有效，await不当故障保证。10源码blob一致／AST解析，未导入、运行或设备验证；全文深读0，代码广扫1条路径，不称整个框架读完。后续扩batch／old-logp／staleness契约并继续原生文献，M5专项不动。

- **2026-10-27（M1原生框架与低精度补库）**：新增RLinf／DynaRL／NeMo-Aligner／ROLL Flash／CadenceRL及两篇含系统章节的模型报告，累计97身份（20 serving／边界＋77 rollout/RLHF及相关边界；93 arXiv＋4会议记录）。RLinf arXiv／OSDI不重复，ROLL Flash为后续独立论文非原ID修订；异步RPC不等stale-policy，CadenceRL暂候续跑不等删轨迹，FP8 policy／draft区分，CompassMax过滤单列C。仅身份／摘要和局部机制、定性评估条件，深读／代码仍0／0；未设备验证或扩算法综述，后续继续原生与种子扩展。

- **2026-10-27（M1引用扩展与多轮缓存边界）**：从TideRL参考文献新增SpecActor／MindSpeed RL／Infinite Sampling／AgentRL及CachedAttention／CONCUR／Continuum／Helium八篇，累计90身份（20 serving／边界＋70 rollout/RLHF及相关边界；87 arXiv＋3会议记录）。FoN选draft不等筛训练响应；micro group与完整组统计区分，AgentRL主动cross-policy单列C，缓存驻留不等跨权重可复用或删样本。仅身份／摘要及局部正文定位，深读／代码仍0／0；无NPU任务或实验，下步继续原生与种子扩展，不宣称覆盖收敛。

- **2026-10-27（M1原生补库与种子落实）**：新增TideRL／ROSE／QwenGyre，落实SpecRoll→SPARK同ID改题与ExpThink既有线索，共增5矩阵条目而非5全新发现，累计82身份（16 serving／边界＋66 rollout/RLHF及相关边界；80 arXiv＋2会议记录）。SPARK采用v2，不迁旧Reflex说明；就绪度弹性、serving算力协作及路径去重三边界分开，ExpThink不混作执行加速。仅身份／摘要及局部正文定位，深读／代码仍0／0；下步会议／引用扩展原生并补serving，不宣称原生目标已达成。

- **2026-10-27（M1异步／训练边界补核）**：核剩余11篇历史条目，新增FP16／VeXact／CIS三篇TIM文献，累计77身份（16 serving／边界＋61 rollout/RLHF及相关边界；75 arXiv＋2会议记录）。历史✅清零不等于corpus收敛或77篇可用实证；ASymPO证明错误与v2撤回显式警示。Intelligent Router移serving，AReaL-DTA改训练侧边界，RolloutPipe的group流水与stale-policy异步分开。仅身份／摘要补核，无深读、代码、复现或M5专项改动；后续优先新增原生及种子，不重复本批。

- **2026-10-27（M1 agentic补核）**：核既有10篇，补InfoTree名称及RollArc／RollArt版本沿革；定向检索补FlexMARL与两篇相关Agent Lightning论文，累计63身份（15 serving／边界＋48 rollout/RLHF；61 arXiv＋2会议记录），仅3新发现。系统／recipe、VLA样本效率／吞吐、同步prefix重叠／async off-policy分开，暂留分类疑问而非性能结论。未全文深读、读代码或复现，下批核剩余L4／agentic边界并扩展原生。

- **2026-10-27（M1推测机制与范围补核）**：核既有8篇并补Bebop／Aurora别名，FastGRPO种子落实；EfficientRollout引用回溯新增两篇SD基础与Iso等NeMo-RL集成论文。Aurora移serving边界、ObsSpec移agentic，防止把speculator RL、observation推测或稀疏／量化draft混成policy rollout机制；Iso实测／模拟证据分开。累计50身份（15 serving／边界＋35 rollout/RLHF；48 arXiv＋2会议记录），仅3新发现。无全文深读、代码／性能验收，下批核agentic簇与扩展引用。

- **2026-10-27（M1长尾／推测补核）**：核既有9篇、落实Seer／TLT两种子，WAR引用回溯新增SuffixDecoding并核v1/v3同文改题。累计38身份（12 serving＋26 rollout/RLHF；36 arXiv＋2会议记录），仅1新发现。WAR／TailSieve暂A/B，SortedRL暂B/C?，RhymeRL补调度维度；撤回C类普遍缺评估／推测方向已饱和的先验判断。仍无全文深读、代码或复现，下批核其余推测／agentic与扩展引用。

- **2026-10-27（M1 RL系统补核）**：核8篇既有RL／RLHF系统身份与版本，收窄DistFlow／AsyncFlow／OpenRLHF旧描述；新增ROLL、EARL及从StreamRL回溯并核官方记录的PUZZLE。累计26篇身份已核（11 serving＋15 rollout/RLHF；24 arXiv＋2会议记录），仅新增3篇，不重算历史≈73。§5.3标题同步撤回预设不可迁移说法；仍无全文深读、代码或复现。下一批转长尾／推测rollout候选核验与引用扩展。

- **2026-10-27（M1基础补核与rollout补库）**：核既有9篇serving原始文献，补primary链接／首次提交／采用版本，FA1/2拆行不计新发现；新增APRIL 2509.18521v3与RLHFSpec 2512.04752v2，累计15篇身份已核（11 serving＋4 rollout/RLHF）。两新项层／类型暂定，未抽性能数字；修正PD分离与trainer/rollout分离混同，撤回speculative不可直接迁移的先验绝对说法。未做全文深读、taxonomy收束、代码或复现；下一批转rollout-native来源核验与扩展。

- **2026-10-27（M1身份核验批次）**：承接新专项，核4篇而非新增4篇；纠正SGLang／Orca／RLHFuse三处错配ID，合并RLHFuse与Stage Fusion重复项，登记ReaLHF→ReaL及MLSys'25版本信息。来源为arXiv原始／修订元数据与USENIX官网；撤回“≈73”为已去重已核数量的解释，保留Round 1历史记录。未做深读／性能判断、taxonomy收束、代码改动或复现；下一步按§5.7补其余来源与身份。

- **2026-10-27（Round 1）**：骨架建立；种子注册 + 2 批 arXiv 验证检索（40 篇实证 + 15+ 新发现：SRT/ObsSpec/S³R/NgramRL/DARTS/ASymPO/MARLaaS/AsyncWebRL/AcceRL/Stage Fusion/FlashNext-RL/AREAL-DTA 等）；corpus ≈73（目标 100–150）。待验证种子 8 个。未做：深读、代码地图扩展、corpus 收敛检索（SGLang/vLLM 博客、ICML/NeurIPS/OSDI/SOSP/NSDI/ATC/EuroSys 2025-2026 proceedings 扫描、citation graph 前向/后向扩展）。
