# Rollout Performance Research — LLM Post-training / RL Rollout 性能优化调研

> 本文件是本专题唯一的 canonical knowledge base。任何会话/Agent 接手时**只读本文件**即应在 10–20 分钟内恢复研究状态；详细证据存放于链接指向的材料，但恢复研究状态不依赖它们。
> 远景（非当前任务）：systems-oriented survey《Efficient Rollout Systems for LLM Post-Training》。**当前阶段禁止**撰写正文/Introduction/Related Work、冻结 taxonomy、批量生成低价值摘要、启动大规模复现、在外部主仓库发 issue/PR/comment。

## 0. Project Status

- **阶段**：**Phase 2 — deep research（2026-10-27 用户指令：广度已足（计数另按唯一身份核算），转入论文/项目深读，围绕 rollout 性能与 L1–L4 四维；M1 定时启动的技术秘书任务以深读为主要工作）**
- **最近更新**：2026-10-28（§8.11–8.12完成C3 rLLM／Slime两条生成消费路径，累计11路径／8系统（6 RL框架＋2推理引擎）；126身份及20/84/21/1深度账不变。HybridFlow原版实现／原始数据与覆盖验收仍待，截止≤2026-10-29不变）
- **覆盖**：累计**126篇已核身份**（20篇serving基础／迁移边界＋106篇rollout/RLHF及相关边界；120个唯一arXiv ID＋Orca／PUZZLE／CachedAttention／DynaRL／Weave／MTP-RL会议记录；其中T1压缩深读20篇，其余不据身份核验算深读）。§5.25将RL相关106暂分89直接候选与17保守单列，承接§5.20的86／15历史快照；这是工作范围计数，不是原生80–120覆盖／质量验收。MTP-RL不与Bebop同篇；初始种子及旧五线索身份已落实，但检索／分级／正文审查未全收敛。ASymPO证明错误警示保留，Round 1“≈73”不相加。
- **接续线索**：§5.23原余5项已在§5.25入库，§6.33完成Belayer／BIRD-RL／ScaleRLT及PrefixFlow／psRL／CIS六篇T2。训练侧共享与概率修正仍单列边界，不因深化转为原生覆盖；入库、T2与实现／复现验收分开。
- **深读 / 代码广扫**：按§6.0新压缩标准完成T1 **20篇**（B1–B5各四篇）、T2 **84篇**（B6补后22/22、B7为9/9、B8补后13/13、B9为11/11、B10为12/12、B11为11/11、B13为6/6；含Intelligent Router serving旁读；GLM-4.5仅§3.5系统片段，非整份报告；非源码或性能复现验收），另有HybridFlow进行中1项 / **11条代码路径**（veRL `24f25b03`四条；OpenRLHF `dc2a7ad3`、NeMo RL `a9d1a4be`、SkyRL `e0893f6e`、rLLM `4e828c81`、Slime `540ea919`各一条；vLLM `eaf1627c`、SGLang `cb55d77d`各一条；共8系统＝6 RL框架＋2推理引擎，不按文件数计，不是原生运行验证）；完整复现仍0。
- **最高优先级（2026-10-27 用户加急指令）**：**Phase 2 全部深读须在两天内完成（≤2026-10-29）**——每次定时触发完成**一批对应规模**的调研（分级深度 + 批次表见 §6.0），不得单次只做一篇；深度与时间约束按批次表权衡。B13、C2与C3本批已收束；先核HybridFlow已有认领，再补原版实现／原始数据与Phase 2覆盖缺口，不再为数量扩文献。M5专项与上游写入边界不变。
- **Triage与未读账**：B12范围triage完成28项的历史范围不变，triage事项数不等当前T3身份数。B13三项原T3（PrefixFlow／psRL／CIS）升级T2并替换最深层；126＝20 T1＋84 T2＋21 T3＋1未标完成，仅HybridFlow进行中。不存在把同篇T2/T3相加的新增覆盖。
- **深读当前焦点**：下一轮先核远端及HybridFlow已有认领／负责人，再补论文年代实现／原始数据和覆盖缺口审计；代码11已达8–12数量目标，不用现代源码替代历史实现。C1仅veRL数据路径完成，AReaL部分不开展／不据C3完成自动销项。T1/T2或代码数量达标不等Phase 2完成，M5专项与上游写入边界不变。
- **Phase 2 完成标准与预估（2026-10-27 定，用户问"多久读完"后定量）**：**不是 121 篇全读**。深读总量 = A Core 15–20 篇全模板深读 + B Supporting ~40–60 篇中等深度（机制+关键结果，轻模板）+ 代码深读 ~8–12 条路径；C 边界与 serving 保持元数据级即可。节奏：每篇 A Core ≈ 1 个定时任务轮次，B 类 3–5 篇/轮次；按多会话并行（每日 2–4 轮次涉及本专项）预估 **A Core ≈ 1–1.5 周、Phase 2 整体 ≈ 3–4 周日历时间**。里程碑：第 1 周末 A Core 短名单定稿+完成一半；第 2 周末 A Core 完成+代码地图齐；第 3–4 周 B 覆盖+gaps/ideas 提炼+（可选）survey outline。**加速杠杆**：为定时任务显式分配深读配额。
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
| [2505.24298v5](https://arxiv.org/abs/2505.24298v5) | 2025-05-30首次／v5 2026-03-02 | AReaL | L2/L4 | B（系统；混版本／目标另列） | 生成准入限流与旧样本优先；中断换权重重算KV，decoupled PPO区分behavior／proximal | 核；[T1](#65-areal论文-t1压缩深读) |
| [2409.19256v2](https://arxiv.org/abs/2409.19256v2) | 2024-09-28首次／v2 10-02 | HybridFlow / veRL | L4 | B | 混合单／多controller与分层API，3D-HybridEngine用于训练和生成间actor resharding | 核 |
| [2507.13833v4](https://arxiv.org/abs/2507.13833v4) | 2025-07-18首次／v4 2026-05-30 | DistFlow | L4 | B | DAG串行化＋分布式数据面，缓存中间张量非prefix KV；512卡为随batch增长的弱扩展，不当固定job加速 | 核；[T2](#6311-distflow) |
| [2504.15930v1](https://arxiv.org/abs/2504.15930v1) | 2025-04-22／v1 | StreamRL | L3/L4 | B（异步语义另列） | stream与ranker调度分治气泡；强制长度吞吐实验与真实收敛分开 | 核；[T1](#61-streamrl-t1压缩深读) |
| [2505.24034v2](https://arxiv.org/abs/2505.24034v2) | 2025-05-29首次／v2 06-02 | LlamaRL | L4 | B（AIPO／FP8另列） | Executor独立并行度与DDMA权重传输；10.7×含异步／并行度／FP8组合，非纯异步收益 | 核；[T2](#6212-llamarl) |
| [2510.12633v1](https://arxiv.org/abs/2510.12633v1) | 2025-10-14／v1 | Laminar | L3/L4 | B（buffer策略另核） | CPU relay与同版本repack；低陈旧性是观测，不是显式硬上限 | 核；[T1](#63-laminar-t1压缩深读) |
| [2604.26256v2](https://arxiv.org/abs/2604.26256v2) | 2026-04-29首次／v2 07-20 | DORA | L2/L3/L4 | B（跨步集合另核） | 单轨迹固定版本、旧版本排空后移窗，同版本KV迁移；不丢弃自述非同步等价证明 | 核；[T1](#64-dora-t1压缩深读) |
| [2601.12784v2](https://arxiv.org/abs/2601.12784v2) | 2026-01-19首次／v2 08-03 | StaleFlow | L3/L4 | B；冗余／过滤另C | Reserve／Occupy／Consume界定轨迹年龄，TS／PS协调路由；主评估禁用冗余 | 核；[T1](#62-staleflow-t1压缩深读) |
| [2507.01663v2](https://arxiv.org/abs/2507.01663v2) | 2025-07-02首次／v2 2026-09-11 | AsyncFlow | L3/L4 | B（版本／目标另核） | TransferQueue字段／微批流水及延迟一步安装权重；Ascend结果与串行同栈消融分账 | 核；[T2](#6211-asyncflow) |
| [2606.26997v2](https://arxiv.org/abs/2606.26997v2) | 2026-06-25首次／v2 07-05 | RolloutPipe | L3/L4 | B（过滤／顺序另核） | 固定本轮rollout权重，完整group先送训；每U组更新、全R组消费后发布权重 | 核；[T1](#66-rolloutpipe-t1压缩深读) |
| [2410.18252v3](https://arxiv.org/abs/2410.18252v3) | 2024-10-23首次／v3 2025-04-26 | Asynchronous RLHF | L4 | B；**C**（off-policy／选样） | Cleanba式前一批数据训练与当前权重生成重叠；best/worst、减步数及KL代价分账，非纯调度2.5× | 核；[T2](#6312-asynchronous-rlhf) |
| [2406.14088](https://arxiv.org/abs/2406.14088v2) | 2024-06 | ReaL（v1名ReaLHF） | L4 | B | 函数级mesh／并行度搜索与参数重分配；MCMC不保最优，主测随机prompt强制满长度、20步吞吐非收敛时间 | 核；[T2](#6313-real) |
| [2409.13221](https://arxiv.org/abs/2409.13221v3) | 2024-09 | RLHFuse／Stage Fusion | L3/L4 | B | 聚尾KV迁移后提前评分、actor／critic双向流水；保留训前屏障，融合对同栈Base为1.2–1.4×非3.7× | 核；[T2](#6314-rlhfuse) |
| [2405.11143v6](https://arxiv.org/abs/2405.11143v6) | 2024-05-20首次／v6 2025-10-09 | OpenRLHF | L2/L4 | B；**C**（DAPO筛组另列） | Ray／vLLM／ZeRO＋AutoTP／Ring；PDF表1原始时间几何平均复算1.146／1.181／1.393×，不沿用汇总列 | 核；[T2](#6315-openrlhf) |
| [2308.01320v1](https://arxiv.org/abs/2308.01320v1) | 2023-08-02／v1 | DeepSpeed-Chat | L1/L2/L4 | A/B；**C**（可选mixture） | Hybrid Engine在TP生成／ZeRO训练间切换并管理内存；Step3吞吐与SFT＋RM＋PPO全流程时间、历史价格分账 | 核；[T2](#6316-deepspeed-chat) |
| [2506.06122v1](https://arxiv.org/abs/2506.06122v1) | 2025-06-06／v1 | ROLL | L3/L4 | B（recipe／过滤另列） | 单controller、ParallelWorker与逐样本reward／env流水；质量曲线不当系统吞吐证据 | 核；[T2](#6216-roll) |
| [USENIX ATC'24](https://www.usenix.org/conference/atc24/presentation/lei) | 2024-07会议记录 | PUZZLE | L4 | B | 无依赖模型时分重叠＋训练／生成并行布局转换；短256-token PPO及混栈比较，不外推长链 | 核；[T2](#6321-puzzle) |
| [2605.12070v3](https://arxiv.org/abs/2605.12070v3) | 2026-05-12首次／v3 09-29 | Missing Old Logits in Asynchronous Agentic RL | L4边界 | 系统成本／算法修正，mask另C | 精确old logits分解训推差异与版本陈旧性；EWMA是近似，快照成本窗口有Ray故障终止 | 核；[T2](#6227-missing-old-logits) |
| [2606.03070v3](https://arxiv.org/abs/2606.03070v3) | 2026-06-02首次／v3 06-23 | ASymPO（证明错误警示） | L4边界（暂） | 算法边界，不硬归B/C | current-policy-only的非对称loss尺度归一化；页面注明证明有误、v2曾撤回，见§5.13 | 核；[T3](#525-剩余边界与接续线索分级) |
| [2602.01826v1](https://arxiv.org/abs/2602.01826v1) | 2026-02-02／v1 | Beyond Precision | L1/L4边界 | 优化诊断；数据过滤另C | 长度激增时点指导定期LR减半；1.8×为启发式，稳定性改善不等引擎精确一致或系统提速 | 核；[T2](#6228-beyond-precision) |
| [2510.26788v1](https://arxiv.org/abs/2510.26788v1) | 2025-10-30／v1 | Defeating the Training-Inference Mismatch via FP16 | L1/L4边界 | 数值诊断，不硬归A/B | 训推同时FP16＋loss scaling降低失配，仍有框架尖峰；稳定性非bit-exact或通用精度建议 | 核；[T2](#6231-fp16) |
| [2605.14220v2](https://arxiv.org/abs/2605.14220v2) | 2026-05-14首次／v2 09-29 | Diagnosing Training Inference Mismatch（VeXact） | L1/L4边界 | 诊断；RS另C | batch-invariant参考隔离TIM，聚合KL可掩盖失败；TIS＋RS近似跟踪非严格目标等价 | 核；[T2](#6232-vexact) |
| [2609.32444v1](https://arxiv.org/abs/2609.32444v1) | 2026-09-26／v1 | Rethinking Training-Inference Mismatch（CIS） | L1/L4边界 | 算法修正，不硬归B/C | confidence-aware单侧截断并加数值floor；只修观测prefix条件下token分布，有偏非轨迹精确校正 | 核；[T2](#6336-cis) |
| [2507.19017v1](https://arxiv.org/abs/2507.19017v1) | 2025-07-25／v1 | MindSpeed RL（Ascend平台） | L3/L4 | B | transfer dock与allgather-swap；主16 NPU、按设备归一TPS含prompt，不外推CUDA或384卡主测 | 核；[T2](#6214-mindspeed-rl) |
| [2509.15965v2](https://arxiv.org/abs/2509.15965v2) | 2025-09-19首次／v2 12-29 | RLinf | L3/L4 | B | M2Flow时空重组与profile计划，GRPO保留整prompt组；空间拆分长序列7B有负收益 | 核；[T2](#6213-rlinf) |
| [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) | 2026-07会议记录 | DynaRL | L3/L4 | B | RLinf上动态hypergraph／安全点迁移／多轮优先路由；1.98×为32B对verl，不是对RLinf | 核；[T2](#6322-dynarl) |
| [2405.01481v2](https://arxiv.org/abs/2405.01481v2) | 2024-05-02首次／v2 09-03 | NeMo-Aligner | L1/L2/L4 | A/B | TRT-LLM常驻engine原地refit与reshard、PyTriton异步RPC；96→192卡扩展不当固定卡收益 | 核；[T2](#6215-nemo-aligner) |
| [2510.11345v1](https://arxiv.org/abs/2510.11345v1) | 2025-10-13／v1 | ROLL Flash（Part II） | L3/L4 | B；**C**（冗余／过滤） | 样本年龄约束下混版本续跑，queue／env异步；先够数即停的冗余采样另C，5.45×为模拟 | 核；[T2](#6217-roll-flash) |
| [2508.06471v1](https://arxiv.org/abs/2508.06471v1) | 2025-08-08／v1 | GLM-4.5报告（Slime §3.5） | L1/L3/L4 | A/B；**C**（数值／任务过滤） | Megatron／SGLang-Router／Data Buffer两种模式；在线block FP8与task过滤另列，无独立系统计时 | 核；[T2](#6218-glm-45-slime系统片段) |
| [2512.07710v1](https://arxiv.org/abs/2512.07710v1) | 2025-12-08／v1 | Each Prompt Matters（CompassMax-V3-Thinking） | L1/L3/L4（暂） | A/B／**C**（联合算法） | FP8 rollout、reward重叠与长度调度；另有zero-variance过滤、ESPO及MoE Router Replay | 核；[T3](#525-剩余边界与接续线索分级) |
| [2606.01143v4](https://arxiv.org/abs/2606.01143v4) | 2026-05-31首次／v4 10-15 | PrefixFlow | L4训练侧边界 | B（训练调度） | prefix前向一次、suffix累积gK/gV后延迟反向；长prefix全步1.90×来自actor更新非生成 | 核；[T2](#6334-prefixflow) |
| [2608.25683v1](https://arxiv.org/abs/2608.25683v1) | 2026-08-26／v1 | psRL | L4训练侧边界 | B（训练调度） | 静态样本的prefix共享、token微批与KV生命周期；update至5.2×、整RL迭代至2.1×分账 | 核；[T2](#6335-psrl) |
| [2605.07330v1](https://arxiv.org/abs/2605.07330v1) | 2026-05-08／v1 | SparseRL-Sync | L4 | B | post-cast索引／新值覆盖；106B同步延迟实测、671B外推，约100×压缩不当全训练加速 | 核；[T2](#6222-sparserl-sync) |
| [2602.03839v2](https://arxiv.org/abs/2602.03839v2) | 2026-02-03首次／v2 05-19 | PULSE（Understanding and Exploiting Weight Update Sparsity） | L4／训练通信边界 | B（Sync）；LoCo优化另列 | Sync覆盖BF16新值，LoCo发送FP32 pseudo-gradient并留误差反馈；130×为上传字节比，非同一通道／算法精确等价 | 核；[T2](#6223-pulse) |
| [2601.18150v2](https://arxiv.org/abs/2601.18150v2) | 2026-01-26首次／v2 04-10 | FP8-RL | L1/L2/L4 | A/B；**C**（数值policy／TIS） | 真FP8 policy／KV及每步scale刷新；token TIS不保证BF16轨迹等价，MoE长跑仍有崩溃 | 核；[T1](#620-fp8-rl-t1压缩深读) |
| [2610.16866v1](https://arxiv.org/abs/2610.16866v1) | 2026-10-12／v1 | PivotRL（Efficient Resource-Shiftable RL Post-training with Generative Rewards） | L3/L4 | B | GPS／role context切换＋保token重prefill迁移；planner模拟不等主评估模拟，收益限含GenRM三角色 | 核；[T2](#6323-pivotrl) |
| [2512.22492v1](https://arxiv.org/abs/2512.22492v1) | 2025-12-27／v1 | RobustRL（Role-Based Fault Tolerance System for LLM RL Post-Training） | L4 | B（容错） | 角色恢复／同构rollout备用／UCX relay；每10步注入trainer故障，CPU管理角色未获容错 | 核；[T2](#6324-robustrl) |
| [2610.08430v1](https://arxiv.org/abs/2610.08430v1) | 2026-10-06／v1 | NeMo-DCR | L4 | B | canonical投影／native loader放置，XOR去重及overwrite恢复／joint commit；refit等价不含KV，延迟压测不当1T训练 | 核；[T2](#6224-nemo-dcr) |
| [2610.05935v1](https://arxiv.org/abs/2610.05935v1) | 2026-10-05／v1 | ThunderSyncRL | L4 | B | 固定快照先累计reward加权／未加权score梯度，整组归一、整批一次更新；数学等价不当bit-exact | 核；[T2](#6225-thundersyncrl) |
| [2509.19128v2](https://arxiv.org/abs/2509.19128v2) | 2025-09-23首次／v2 09-26 | PipelineRL | L2/L4 | B；**C**（mixed-policy／旧KV） | in-flight换权重保留旧KV，ESS近似on-policy非零陈旧；约2×对模拟并修正的Conventional RL基线 | 核；[T2](#6226-pipelinerl) |
| [2609.36830v1](https://arxiv.org/abs/2609.36830v1) | 2026-09-29／v1 | PACE（Where Does Staleness Accumulate?） | L4 | B／**C**（拒收） | pool超额占用→EMA拒收预算，Kwait＋相对prefix漂移加权Kgen排序；不拒绝在途轨迹 | 核；[T1](#67-pace-t1压缩深读) |
| [2510.19225v3](https://arxiv.org/abs/2510.19225v3) | 2025-10-22首次／v3 2026-04-08 | RLBoost | L3/L4 | B | 自适应partial seeding／pull权重／token迁移；主吞吐为真实trace在按需卡回放，租价估算非spot账单 | 核；[T2](#6325-rlboost) |
| [2602.02192v6](https://arxiv.org/abs/2602.02192v6) | 2026-02-02首次／v6 09-29 | ECHO-2 | L3/L4 | B／**C**（过旧丢弃） | overlap容量约束下按c/μ激活worker，分块链式广播与版本窗口丢弃；优化cost–quality | 核；[T1](#68-echo-2-t1压缩深读) |
| [2604.11554v2](https://arxiv.org/abs/2604.11554v2) | 2026-04-13首次／v2 04-14 | Relax（Omni-Modal Post-Training） | L3/L4 | B（异步语义另核） | TQ字段／microbatch流水与producer门控、encoder布局；Actor／Rollout等故障仍全局重启，隐藏计算非免费 | 核；[T2](#6326-relax) |
| [2507.07966v4](https://arxiv.org/abs/2507.07966v4) | 2025-07-10首次／v4 09-30 | Long-RL（Scaling RL to Long Videos／MR-SP） | L1/L2/L4 | A/B（MR-SP）；数据筛选C | 分片视觉编码与embedding复用、policy／reference prefill用SP；8 A100的2.1×是forward窗口，非全步 | 核；[T2](#6221-long-rl) |

### 5.2 L3 — Long-tail / Scheduling / Cluster

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2509.21009v1](https://arxiv.org/abs/2509.21009v1) | 2025-09-25／v1 | RollPacker | L3/L4 | B／**C**（响应选择／评分预算） | tail batching延后prompt但短轮保留最快响应；流式算梯度、整轮后更新，自适应sandbox超时另改奖励边界 | 核；[T1](#69-rollpacker-t1压缩深读) |
| [2603.23414v1](https://arxiv.org/abs/2603.23414v1) | 2026-03-24／v1 | SortedRL | L3/L4 | B／**C**（批构成／更新次序） | 长度分批且清空prompt池后再补新；on-policy重跑未完响应，partial续token并缓存行为logp；micro-curriculum | 核；[T1](#610-sortedrl-t1压缩深读) |
| [2608.22788v2](https://arxiv.org/abs/2608.22788v2) | 2026-08-24首次／v2 08-26 | TailSieve | L1/L3/L4 | A/B；**C**（跨轮集合） | 当前policy重生成旧尾组＋本轮最早bulk组，稳态无额外长度偏差不等每批集合不变 | 核；[T2](#6233-tailsieve) |
| [2607.17299v1](https://arxiv.org/abs/2607.17299v1) | 2026-07-19／v1 | WAR | L1/L3 | A/B（超时边界另核） | 低负载suffix draft、高负载TTL／cache调度；计时筛选工作量差≤5%的运行，小batch调度可负收益 | 核；[T2](#6234-war) |
| [2605.30859v1](https://arxiv.org/abs/2605.30859v1) | 2026-05-29／v1 | DARTS | L3/L4 | B／**C**（主动塑形） | 7短＋1长组选择、预算饱和改shortest-only；token流水仅前向提前，整轨迹后奖励／反向 | 核；[T1](#612-darts-t1压缩深读) |
| [2509.18521v3](https://arxiv.org/abs/2509.18521v3) | 2025-09-23首次／09-26修订 | APRIL | L3/L4 | B／**C**（每步集合／混版本） | 超发后按完整prompt组收齐配额，未完token优先跨版本续跑；不丢轨迹不等同步采样等价 | 核；[T1](#611-april-t1压缩深读) |
| [2511.14617v3](https://arxiv.org/abs/2511.14617v3) | 2025-11-18首次／v3 2026-04-03 | Seer | L1/L2/L3 | A/B | chunk与共享KV池、组内probe调度＋suffix推测；尾时段是最后10%请求，不是P95 | 核；[T2](#6235-seer) |
| [2605.07501v2](https://arxiv.org/abs/2605.07501v2) | 2026-05-08首次／v2 05-17 | ExpThink（CoT塑形边界） | workload／算法边界 | **C**（reward／advantage） | 每题最短正确长度统计塑形、按正确数缩放优势；77.18%是平均相对token减量，非同工作量提速 | 核；[T2](#6292-expthink) |
| [2506.22950v1](https://arxiv.org/abs/2506.22950v1) | 2025-06-28／v1 | Infinite Sampling | L2/L3/L4 | B（dynamic-slot集合另列） | 固定slot复用KV与SJF refill、全组reward后反向；decode rounds不当墙钟，FPTAS保证待核 | 核；[T2](#6236-infinite-sampling) |
| [2609.36899v1](https://arxiv.org/abs/2609.36899v1) | 2026-09-29／v1 | CadenceRL（Reshaping Rollout Workloads） | L2/L3/L4 | B（消费／年龄另核） | 同版本pending续跑与聚尾，credit非逐轨迹年龄界；跨硬件KV不兼容时re-prefill | 核；[T2](#6237-cadencerl) |
| [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan) | 2026-07会议记录 | Weave（Efficient Co-Scheduling for Disaggregated RL Post-Training） | L3/L4 | B | 跨job组内交错＋host warm切换；局部最优有负载前提，trace回放成本与模拟近最优分账 | 核；[T2](#6238-weave) |
| [2606.05606v1](https://arxiv.org/abs/2606.05606v1) | 2026-06-04／v1 | CERO（Cross-Epoch Adaptive Rollout Optimization） | workload／算法边界 | **C**（预算分配） | posterior信息代理＋跨epoch预算价格决定采样，固定utility regret不等非平稳实现保证；保留7B负项 | 核；[T2](#6241-cero) |

| [2608.11152v1](https://arxiv.org/abs/2608.11152v1) | 2026-08-11／v1 | Scheduling Mixed RL Rollouts（MISA-T） | L3 | B（准入；混合条件另列） | KV block-time类预算＋active/pending准入；50迭代全步降22.8%，混合TV仍2.71个百分点非零 | 核；[T2](#6242-scheduling-mixed-rl-rollouts) |

### 5.3 L1/L2 — RL-aware Speculative Rollout（serving机制的迁移条件）

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2509.23232v3](https://arxiv.org/abs/2509.23232v3) | 2025-09-27首次／v3 2026-01-12 | SPEC-RL | L1/L2 | A/C | 历史轨迹作draft；默认lenience改采样分布，Algorithm 1拒绝后直接采当前policy亦不足证ℓ=1精确 | 核；[T2](#6327-spec-rl) |
| [2508.18588v1](https://arxiv.org/abs/2508.18588v1) | 2025-08-26／v1 | RhymeRL（History Rhymes） | L1/L2/L3/L4 | A/B／**C**（跨步集合） | reward-aware历史树只供draft验证；HistoPipe一步陈旧及跨步迁移另核，不混同步等价 | 核；[T1](#613-rhymerl-t1压缩深读) |
| [2510.26475v1](https://arxiv.org/abs/2510.26475v1) | 2025-10-30／v1 | ReSpec | L1/L2/L4 | A（验证契约待核） | 按active batch选SD配置；reward加权的是drafter KD，非直接actor目标，分布偏差归因留疑 | 核；[T1](#614-respec-t1压缩深读) |
| [2511.13841v1](https://arxiv.org/abs/2511.13841v1) | 2025-11-17／v1 | DAS（Beat the long tail） | L1/L3 | A/B（验证契约待核） | 近期同题后缀树＋动态长短draft预算；PDF式7与约束代数不一致，不据此断言实现错误 | 核；[T1](#616-das-t1压缩深读) |
| [2511.16665v3](https://arxiv.org/abs/2511.16665v3) | 2025-11-20首次／v3 2026-03-20 | TLT（Taming the Long-Tail） | L1/L2/L3/L4 | A/B | 尾部Spot Trainer＋分桶graph/MAB；三步计时用已训draft，旧长响应只训draft不改actor屏障 | 核；[T2](#6246-tlt) |
| [2512.04752v2](https://arxiv.org/abs/2512.04752v2) | 2025-12-04首次／12-12修订 | RLHFSpec | L1/L3/L4 | A/B | profile选树与固定请求迁移；局部TPS非全局makespan最优 | 核；[T2](#6251-rlhfspec) |
| [2605.08862v1](https://arxiv.org/abs/2605.08862v1) | 2026-05-09／v1 | BubbleSpec | L1/L3/L4 | A/B（精确验证条件） | 仅用inter-GPU气泡预生成下一批draft；逐token接受／残差校正＋统一短query attention | 核；[T1](#615-bubblespec-t1压缩深读) |
| [2606.18967v2](https://arxiv.org/abs/2606.18967v2) | 2026-06-17首次／v2 09-30 | EfficientRollout | L1/L2 | A（验证／KV契约） | 每步RTN W4 draft＋roofline一次开启＋跨步调长度；target覆盖临时KV并拒绝校正 | 核；[T1](#617-efficientrollout-t1压缩深读) |
| [2601.09083v1](https://arxiv.org/abs/2601.09083v1) | 2026-01-14／v1 | SRT | L1/L3 | A/B | per-prompt树由当前policy验证；主计时无run-ahead，维护增益模拟与实测分账 | 核；[T2](#6243-srt) |
| [2610.13202v1](https://arxiv.org/abs/2610.13202v1) | 2026-10-09／v1 | S³R | L1/L2/L4 | A/B（稀疏直采对照另C） | sparse draft＋dense KV重算／残差采样，完整组同快照；KV压力限流，稀疏读不减完整KV存储 | 核；[T2](#6244-s3r) |
| [2610.32881v1](https://arxiv.org/abs/2610.32881v1) | 2026-10-23／v1 | NgramRL | L1/L2 | A/B | 近期per-prompt LUT＋验证结果分支预取；TPS为token/forward，1.38×相对同步预取基线非全训练 | 核；[T2](#6245-ngramrl) |
| [2606.12370v1](https://arxiv.org/abs/2606.12370v1) | 2026-06-10／v1 | Bebop（Breaking Entropy Bounds） | L1/L4 | A | 预RL多步TV＋拒绝采样；熵单调性非一般定理，默认无需online MTP | 核；[T2](#6252-bebop) |
| [2509.21792v1](https://arxiv.org/abs/2509.21792v1) | 2025-09-26／v1 | FastGRPO | L1/L2/L4 | A/B | profile并发配置＋在线draft；总收益与AR提前退出对照分账 | 核；[T2](#6253-fastgrpo) |
| [2604.26779v1](https://arxiv.org/abs/2604.26779v1) | 2026-04-29／v1 | System-Integrated SD（Iso等） | L1/L4 | A/B | 固定EAGLE3实测与大规模预测分开，接受长度不等速度 | 核；[T2](#6254-system-integrated-sd) |
| [2608.04962v2](https://arxiv.org/abs/2608.04962v2) | 2026-08-05首次／v2 10-14 | SPARK（v1名SpecRoll） | L1/L2 | A | prompt-local反馈仅改draft；精确分布依赖因果q与独立验证随机数 | 核；[T2](#6255-spark) |
| [2511.16193v3](https://arxiv.org/abs/2511.16193v3) | 2025-11-20首次／v3 12-23 | SpecActor（Fast LLM Post-training） | L1/L3/L4 | A/B | 解耦窗口＋draft ladder／FoN；抽样trace计时非连续完整训练 | 核；[T2](#6256-specactor) |
| [2609.07108v1](https://arxiv.org/abs/2609.07108v1) | 2026-09-07／v1 | Online Draft Co-Training（Large-Scale Long-Context RL） | L1/L2/L4 | A/B | stop-gradient共训，branch CP与TapChannel；256K算子／RL全步分账 | 核；[T2](#6261-online-draft-co-training) |
| [ACL Findings'26](https://aclanthology.org/2026.findings-acl.1871/) | 2026-07会议记录 | MTP-RL（Acceleration of Reinforcement Learning Rollouts with Policy-Aligned Multi-Token Prediction） | L1/L4 | A；**C（主模型辅助目标）** | 优势门控KL仅限MTP项，但辅助loss回传主模型，非纯目标不变加速 | 核；[T2](#6262-mtp-rl) |

| [2610.34427v1](https://arxiv.org/abs/2610.34427) | 2026-10-25／v1（abs页核） | ScaleRLT | L1/L2/L4 | A/B；**C**（混合深度读出） | 固定读出下defer深层计算、schedule概率值＋dense梯度；改原生读出有质量负项，bitwise非梯度保证 | 核；[T2](#6333-scalerlt) |

### 5.4 Agentic Rollout

| ID | 日期 | 标题（短） | 层 | 类型 | 机制一句话 | 置信 |
|---|---|---|---|---|---|---|
| [2603.18815v1](https://arxiv.org/abs/2603.18815v1) | 2026-03-19／v1 | ProRL Agent | L3/L4 | B/C | HTTP生命周期／三阶段池／token透传；DAPO过滤、错误样本排除与足数取消另C，接口非exactly-once保证 | 核；[T2](#6328-prorl-agent) |
| [2605.15565v1](https://arxiv.org/abs/2605.15565v1) | 2026-05-15／v1 | AstraFlow | L4 | B；**C**（可组合数据策略） | dataflow／RaaS／trainer解耦；2.74×双policy迭代对verl，弹性省GPU-hours不等墙钟更快 | 核；[T2](#6296-astraflow) |
| [2512.22560v2](https://arxiv.org/abs/2512.22560v2) | 2025-12-27首次／v2 2026-06-15 | RollArt（v1摘要名RollArc） | L2/L4 | B；**C**（过旧逐出／冗余终止） | 静态domain硬件映射、轨迹异步与serverless；混卡／外部reward成本、逐出及混版本续跑分别核 | 核；[T2](#6295-rollart) |
| [2603.28101v1](https://arxiv.org/abs/2603.28101v1) | 2026-03-30／v1 | Heddle | L3/L4 | B（采样／版本待核） | 长轨迹优先抢占、排序DP放置、工具间隙迁KV＋MP退火；最优性限代理假设、开销非总能全藏 | 核；[T1](#618-heddle-t1压缩深读) |
| [2511.16108v1](https://arxiv.org/abs/2511.16108v1) | 2025-11-20／v1 | SkyRL-Agent | L4 | B；**C（recipe／梯度屏蔽）** | 三阶段队列重叠与工具recipe分账；超外部限制轨迹不算梯度 | 核；[T2](#6266-skyrl-agent) |
| [2510.05943v1](https://arxiv.org/abs/2510.05943v1) | 2025-10-07／v1 | EARL | L4 | B | rollout间切TP与reference-logp直接分发；局部9.7–11.2×非全流程 | [T2](#6271-earl) |
| [2610.15246v1](https://arxiv.org/abs/2610.15246v1) | 2026-10-11／v1 | ObsSpec | L4 | A/B（精确匹配／执行门槛） | observation匹配才保留预生成动作、核验后才顺序执行；WM离线重放训练，额外占8卡 | 核；[T1](#619-obsspec-t1压缩深读) |
| [2605.05262v1](https://arxiv.org/abs/2605.05262v1) | 2026-05-06／v1 | InfoTree（Rollout Informativeness） | L3/L4 | **C**／B | 树选样／预算C；过期评分重验B，时间到目标非每步更快；理论前提另核 | [T2](#6272-infotree) |
| [2606.05597v3](https://arxiv.org/abs/2606.05597v3) | 2026-06-04首次／v3 08-07 | AsyncWebRL | L4 | B／**C** | 常驻pool／截图引用B；step归一化、混合结果组过滤C，联合吞吐非等工作量 | [T2](#6273-asyncwebrl) |
| [2603.18464v3](https://arxiv.org/abs/2603.18464v3) | 2026-03-19首次／v3 06-12 | AcceRL（VLA边界） | L4 | B／**C** | 三段解耦B；GIPO／延迟norm／WM另C，离线轨迹1000／2000冲突待核 | [T2](#6274-accerl) |
| [2605.08527v1](https://arxiv.org/abs/2605.08527v1) | 2026-05-08／v1 | MARLaaS | L4 | B | 跨租户LoRA交错、租户内等新版本；十任务makespan不等单任务迭代 | [T2](#6275-marlaas) |
| [2608.14498v1](https://arxiv.org/abs/2608.14498v1) | 2026-08-14／v1 | Rollplex | L2/L4 | B（主同步路径） | 同snapshot的prefix–decode重叠，保autograd与全局update屏障；仅兼容布局共享 | 核；[T2](#6265-rollplex) |
| [2602.09578v1](https://arxiv.org/abs/2602.09578v1) | 2026-02-10／v1 | FlexMARL | L3/L4 | B | store驱动micro-batch梯度缓存、整批统一更新；7.3×对naive而非MARTI | [T2](#6276-flexmarl) |
| [2508.03680v1](https://arxiv.org/abs/2508.03680v1) | 2025-08-05／v1 | Agent Lightning（2025） | L4 | B／**C** | 执行／训练接口B；逐调用credit与AIR shaping C，无受控吞吐表 | [T2](#6277-agent-lightning-2025) |
| [2608.17528v1](https://arxiv.org/abs/2608.17528v1) | 2026-08-18／v1 | Agent Lightning v1.0（2026） | L4 | B／**C** | exact-token-prefix合样本；rollout级adv／token-mean和选样另C，非无条件on-policy | [T2](#6278-agent-lightning-2026) |
| [2602.00482v2](https://arxiv.org/abs/2602.00482v2) | 2026-01-31首次／v2 06-13 | AReaL-DTA（训练侧边界） | L1/L3/L4（训练侧） | A/B（训练侧） | DFS共享前后向＋连续分区；局部训练8.31×不作全步，低共享可负益 | 核；[T2](#6263-areal-dta) |
| [2610.28998v1](https://arxiv.org/abs/2610.28998v1) | 2026-10-20／v1 | FlashNext-RL | L1/L4 | A/B；**C（冻结层／零优势组）** | offload与模型kernel优化；开发配置变更／回放与live run分开 | 核；[T2](#6264-flashnext-rl) |
| [2608.10402v1](https://arxiv.org/abs/2608.10402v1) | 2026-08-11／v1 | TideRL | L2/L3/L4 | B；**C（弹性batch）** | CTB／ready驱动ref-actor与rank重分配；跨版本重prefill，变批大小非固定更新集合 | [T2](#6281-tiderl) |
| [2605.06534v2](https://arxiv.org/abs/2605.06534v2) | 2026-05-07首次／v2 05-20 | ROSE | L3/L4 | B；DAPO选样另列 | serving优先共享及turn重路由，零差COO传权重；扩资源收益不当固定卡效率 | [T2](#6282-rose) |
| [2609.33848v2](https://arxiv.org/abs/2609.33848v2) | 2026-09-27首次／v2 10-20 | QwenGyre | L3/L4 | B／**C** | cell弹性与整批一次update；角色路径cap／共享target mask／部分评分另C | [T2](#6283-qwengyre) |
| [2510.04206v1](https://arxiv.org/abs/2510.04206v1) | 2025-10-05／v1 | AgentRL | L4 | B；**C**（cross-policy／task-norm） | 有界队列／动态batch与环境生命周期；主动逐action混新旧policy、task内token归一不当被动同步延迟 | 核；[T2](#6294-agentrl) |
| [2607.01415v1](https://arxiv.org/abs/2607.01415v1) | 2026-07-01／v1 | The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning | L4环境执行边界 | B（成本测量） | 四底座固定命令测启动／执行／reset；1.8×外推worker-hours不含推理／reward／训练，非完整RL加速 | 核；[T2](#6291-rollout-infrastructure-tax) |
| [2607.07508v1](https://arxiv.org/abs/2607.07508v1) | 2026-07-08／v1 | SAO（Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning） | L4／算法边界 | **C**／算法改动 | 128题×1对GRPO16题×8；DIS区间外mask、critic双更新与冻结attention，非吞吐等价优化 | [T2](#6288-sao) |
| [2609.35158v1](https://arxiv.org/abs/2609.35158v1) | 2026-09-28／v1 | PEARL | L2/L3/L4 | B | PD配置＋训练卡借用、增量切换；四rollout卡强拆PD负项及非零交接成本保留 | [T2](#6287-pearl) |
| [2603.13019v1](https://arxiv.org/abs/2603.13019v1) | 2026-03-13／v1 | ARL-Tangram | L3/L4 | B（工具语义另核） | action级弹性与CPU／GPU／API池；平均ACT不等RL尾延迟，pytest并行需核等义 | [T2](#6284-arl-tangram) |
| [2509.23866v1](https://arxiv.org/abs/2509.23866v1) | 2025-09-28／v1 | DART（DART-GUI） | L3/L4 | B／**C** | 逐worker换权重B；动态频率／长度、成功回放、高熵step及步数加权adv另C | [T2](#6285-dart-gui) |

| [2608.22167v1](https://arxiv.org/abs/2608.22167v1) | 2026-08-23／v1 | MCP-Universe RL（MCP-U RL） | L3/L4 | B（placement非严格on-policy） | Acquire／Run／Eval限并发；3 H200五步局部测量，2.8×与2×异步placement分账 | [T2](#6286-mcp-universe-rl) |

| [2608.14635v2](https://arxiv.org/abs/2608.14635v2) | 2026-07-28首次／v2 08-18 | Belayer | L4 | B（容错） | 独立server持有权重／raw KV空间，shadow健康检查后重建请求KV；容器本地prefix一致恢复非外部exactly-once | 核；[T2](#6331-belayer) |
| [2610.29426v1](https://arxiv.org/abs/2610.29426v1) | 2026-10-21／v1 | BIRD-RL | L4 | B/C（系统与curriculum） | trajectory绑定数据库副本、终态释放与脏感知lazy reset；12.8×是在线生命周期I/O降幅 | 核；[T2](#6332-bird-rl) |

### 5.5 Serving 基础（supporting corpus，含迁移边界，不标 rollout-native）

| ID/出处 | 首次提交／采用版本 | 标题（短） | 迁移价值与待核条件 | 置信 |
|---|---|---|---|---|
| [USENIX OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu) | 2022-07会议记录 | Orca | iteration-level scheduling及selective batching；本行用会议记录，不沿用错配arXiv ID | 核；[T3](#630-serving文献triage) |
| [2309.06180v1](https://arxiv.org/abs/2309.06180v1) | 2023-09-12／v1 | vLLM / PagedAttention | 分页KV及共享；并发收益仍取决于rollout布局／驻留，不等减少每步最长响应 | 核；[T3](#630-serving文献triage) |
| [2312.07104v2](https://arxiv.org/abs/2312.07104v2) | 2023-12-12／v2 2024-06-06 | SGLang / RadixAttention | 结构化程序与KV复用；迁到rollout仍需核工作负载与版本约束 | 核；[T3](#630-serving文献triage) |
| [2205.14135v2](https://arxiv.org/abs/2205.14135v2) | 2022-05-27／v2 2022-06-23 | FlashAttention | IO-aware精确attention；kernel加速不等整条rollout makespan收益 | 核；[T3](#630-serving文献triage) |
| [2307.08691v1](https://arxiv.org/abs/2307.08691v1) | 2023-07-17／v1 | FlashAttention-2 | 改work partitioning与并行度；与FA1分开计数，不当RL-native论文 | 核；[T3](#630-serving文献triage) |
| [2501.01005v2](https://arxiv.org/abs/2501.01005v2) | 2025-01-02／v2 2025-04-21 | FlashInfer | 面向不同KV布局、可定制attention与负载均衡；整引擎收益另核 | 核；[T3](#630-serving文献triage) |
| [2403.02310v3](https://arxiv.org/abs/2403.02310v3) | 2024-03-04／v3 2024-06-17 | Sarathi-Serve | chunked prefill与stall-free混批；需按真实prefill/decode构成重评，不假定rollout prompt占比高 | 核；[T3](#630-serving文献triage) |
| [2401.09670v3](https://arxiv.org/abs/2401.09670v3) | 2024-01-18／v3 2024-06-06 | DistServe | prefill/decode分离、TTFT／TPOT约束goodput；不是trainer/rollout分离的同义词 | 核；[T3](#630-serving文献triage) |
| [2407.00079v5](https://arxiv.org/abs/2407.00079v5) | 2024-06-24／v5 2026-10-08 | Mooncake | KV中心化PD分离及过载拒绝；rollout不能无条件继承请求丢弃语义 | 核；[T3](#630-serving文献triage) |
| [2401.10774v3](https://arxiv.org/abs/2401.10774v3) | 2024-01-19／v3 2024-06-14 | Medusa | 多预测头与树状验证；backbone是否冻结及接受规则不可混用 | 核；[T3](#630-serving文献triage) |
| [2401.15077v3](https://arxiv.org/abs/2401.15077v3) | 2024-01-26／v3 2025-03-04 | EAGLE | feature-level草稿预测；只是原始EAGLE，不混作EAGLE-2／3或RL drafter适配 | 核；[T3](#630-serving文献triage) |
| [2411.04975v3](https://arxiv.org/abs/2411.04975v3) | 2024-11-07／v3 2025-10-07 | SuffixDecoding | suffix tree缓存prompt／历史输出，适配推测长度；RL中的跨步／跨worker复用另核 | 核；[T3](#630-serving文献triage) |
| [2211.17192v2](https://arxiv.org/abs/2211.17192v2) | 2022-11-30／v2 2023-05-18 | Fast Inference from Transformers（Leviathan等） | 近似模型draft与target验证／修正采样；输出分布正确性须核实现前提 | 核；[T3](#630-serving文献triage) |
| [2302.01318v1](https://arxiv.org/abs/2302.01318v1) | 2023-02-02／v1 | Speculative Sampling（Chen等） | 修改拒绝采样以保target分布，原文带hardware numerics条件；非bitwise相同承诺 | 核；[T3](#630-serving文献triage) |
| [2602.06932v5](https://arxiv.org/abs/2602.06932v5) | 2026-02-06／v5 2026-08-04 | Aurora（When RL Meets Adaptive Speculative Training） | 由live inference traces在线RL训练speculator、热换更新；不是post-training policy rollout系统 | 核；[T3](#630-serving文献triage) |
| [2408.13510v2](https://arxiv.org/abs/2408.13510v2) | 2024-08-24首次／v2 2025-01-07 | Intelligent Router | 输出长度桶＋硬件profile引导DQN向同模型实例路由／等待；11.43%是E2E延迟改善，非TPS或RL rollout收益 | 核；[T2](#6293-intelligent-router)（serving迁移旁读）；[范围](#630-serving文献triage) |
| [USENIX ATC'24](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | 2024-07会议记录 | CachedAttention | 分层KV、逐层预载／异步保存及scheduler-aware换入换出；context-window有效性不等policy更新后缓存仍有效 | 核；[T3](#630-serving文献triage) |
| [2601.22705v1](https://arxiv.org/abs/2601.22705v1) | 2026-01-30／v1 | CONCUR | agent级准入与cache信号控制并发、缓解middle-phase thrashing；RL下完整组／陈旧性须另核 | 核；[T3](#630-serving文献triage) |
| [2511.02230v7](https://arxiv.org/abs/2511.02230v7) | 2025-11-04首次／v7 2026-09-08 | Continuum | 工具等待中按reload／排队成本设置KV TTL并结合program-level FCFS；TTL到期是缓存回收，不等轨迹终止 | 核；[T3](#630-serving文献triage) |
| [2603.16104v1](https://arxiv.org/abs/2603.16104v1) | 2026-03-17／v1 | Helium（Efficient LLM Serving for Agentic Workflows） | workflow视作query plan，proactive缓存与cache-aware调度；进入RL需核版本／随机输出／样本独立性 | 核；[T3](#630-serving文献triage) |

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

**以下是基于当前矩阵的保守工作口径，不是最终文献分级或新taxonomy（随§5.24更新）。** 沿用已有范围边界，并把主要呈现模型／训练优化、仅包含部分rollout系统内容的3篇暂单列，避免把所有RL相关条目充作原生覆盖；不删除任何论文或撤回其中已有机制。剩余直接候选仍含C类工作、联合算法和待核语义，不能称86篇已验收的纯原生／无损优化。

| 本轮口径 | 数量 | 计数含义 |
|---|---:|---|
| 直接Rollout系统／机制候选 | 86 | §5.1–5.4已核101条中扣除下列15条；含§5.21四项、§5.22三项及§5.24两项，不要求全为A/B或不改采样 |
| 保守单列的范围／混合证据 | 15 | 见逐项清单；暂不用于宣称原生80–120达标，也不是最终C Boundary等级 |
| Serving基础／迁移边界 | 20 | §5.5，独立于上述101；最终相关性仍需再核 |
| 合计 | 121 | 86＋15＋20；身份数，不是全文深读或性能验收数 |

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

**2026-10-28接续**：下列为10-27的恢复与线索快照；余五项现已完成范围triage并转入主矩阵，当前计数、未读清单与T3／待T2判断见§5.25，不再沿用“五项均待范围核查”。原记录保留，尤其不将当时未获取ScaleRLT写成不存在。

2026-10-27再接续：79e892f新增Phase 2方向，但同时回退五行与§5.22–5.24；保留新方向并恢复已有资产，当前仍121／115＋6，旧124自报不沿用。日志均保留，未重复恢复旧条目。

2026-10-27接续核对：`49ac0b3`以116基线覆盖了`372b3e2`已入库的三项多模态条目及§5.22，并移除了同日原10:54日志；本轮恢复原文和三个矩阵行，同时保全该提交与M5后续日志，不以重置或删除另一侧成果解决冲突。原“新增8篇、累计124”的自报不作现行统计：八个名称中APRIL已经在§5.2；其余七个是新增接续线索，不是已完成范围审计的七项原生系统。该恢复轮主矩阵为119＝113 arXiv＋6会议（84＋15＋20）；这是11:25恢复时的历史基线。随后两项完成范围补核并入库，现行计数见§0／§5.20／§5.24；其余五项仍分账，若原七项最终全部入库才为126，而不是124或127。

| 接续ID／短名 | 本轮 primary 身份／版本核对 | 机制与待核范围 | 当前计数状态 |
|---|---|---|---|
| [2509.18521 APRIL](https://arxiv.org/abs/2509.18521) | 既有§5.2的同一身份，已有v3 | partial rollout跨步续跑本身不证明改变分布；拒收／丢弃语义仍按原条目核，不自动升级成C | 重复线索，不新增 |
| [2606.01143 PrefixFlow](https://arxiv.org/abs/2606.01143v4) | v1 2026-05-31／v4 10-15；Training-Time KV Caching for Schedule-Level Prefix Reuse in LLM RL Training | 前缀forward缓存、suffix微批读KV并积累梯度、延迟prefix backward；训练更新侧范围，不直接当rollout生成提速 | 元数据／摘要已核，待范围入库 |
| [2608.25683 psRL](https://arxiv.org/abs/2608.25683v1) | v1 2026-08-26；Efficient Training for Agentic AI via Training-Time Prefix Sharing | update阶段前缀共享及KV管理；与生成侧prefix cache分开，训练端优化是否纳入本专项另核 | 元数据／摘要已核，待范围入库 |
| [2608.11152 Scheduling Mixed RL Rollouts](https://arxiv.org/abs/2608.11152v1) | v1 2026-08-11 | 元数据及局部机制／评估条件已补核，见§5.24 | 已并入§5.2，只计一次；非全文深读 |
| [2608.14635 Belayer](https://arxiv.org/abs/2608.14635v2) | v1 2026-07-28／v2 08-18；Efficient Fault Tolerance for LLM Agentic RL Training | worker局部故障下shadow worker复用独立持有的权重／raw KV分配，按日志prefix重建request KV内容；环境checkpoint与上下文协调恢复。不能称原KV内容无条件存活或所有硬件故障可恢复 | 元数据／摘要已核，待故障模型与范围入库 |
| [2610.29426 BIRD-RL](https://arxiv.org/abs/2610.29426v1) | v1 2026-10-21；Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments | trajectory-scoped执行基础设施复用／隔离数据库副本，同时有四阶段curriculum；系统与recipe混合证据须拆分，不能把整体质量算纯B类收益 | 元数据／摘要已核，待范围入库 |
| [2608.22167 MCP-Universe RL](https://arxiv.org/abs/2608.22167v1) | v1 2026-08-23 | 元数据及局部机制／评估条件已补核，见§5.24 | 已并入§5.4，只计一次；非全文深读 |
| [2610.34427 ScaleRLT](https://arxiv.org/abs/2610.34427) | 本轮abs读取失败（cache miss）；定向检索未返回匹配primary | 仅保留49ac的名称／ID线索；其looped transformer、exact logp、deferred rollout机制自报未独立复核，不沿用✅ | 身份及机制待核，不入主矩阵 |

接续线索状态：两项已入主矩阵，余五项分账（四项仅元数据／摘要已核，一项尚未核到）。arXiv摘要中的会议接收自报不替代会议官网核验。余项仍先核范围和版本，再决定入库或保守单列；“线索保留”不等原生覆盖达标。

撤回两条过强的空白判断：既有AReaL-DTA及serving缓存条目已触及训练／前缀复用边界，不能把PrefixFlow、psRL、Mixed Scheduling混成此前完全真空的同类生成机制；已有RobustRL等恢复条目，也不能据Belayer一个新名字宣布“容错为taxonomy全新维度”。此处只是扩展覆盖线索，不冻结taxonomy。TransferQueue本轮未系统检索所有文献渠道，原“无独立arXiv文献”也仅是上一轮未命中，不作不存在证明。

本轮优先修复真实接续损失；未启动原拟rLLM代码广扫，代码仍6条／4框架、全文深读0。未访问设备、变更M5专项、发起算力任务或写入社区。

### 5.24 混合准入与环境流水

本轮从§5.23转入MISA-T与MCP-U RL两项，不是另外发现两篇。主矩阵121＝115 arXiv＋6会议，101 RL相关暂86直接候选＋15单列，serving20；余五条线索不加总。

- **会话容量不等训练混合配额。** [MISA-T v1 §3–5](https://arxiv.org/html/2608.11152v1)按KV block-time分配准入，pending防快照间超收，HOLD仍记需求。混合保持依赖可行负载及积压占比趋零；有限50步报告TV偏差，不是每批精确同分布或采样等价证明。Step3.7同初始化／采样流对比用6个H200训练节点＋4个推理节点（各推理副本TP8），平均迭代时间下降22.8%；不混同固定checkpoint的rollout-only消融。更新权重后清prefix cache，不能外推跨版本缓存有效；effective prefill TPS含命中token，不作实际计算token速率。
- **更多执行worker不等更多环境或无损异步。** [MCP-U RL v1 §3.1–3.3／§4.2／附录C](https://arxiv.org/html/2608.22167v1)将Acquire／Run／Eval限并发分开，环境保留至评分。gpt-oss-20b软件工程实验用3 H200、无高速NVLink；固定Acquire=24、live环境=120，Run从24扫到120，warmup后5步平均rollout tokens/s由147到410。另一个placement对比只平均前5步，异步为至多一步陈旧；不能把两种收益相乘、称精确on-policy或time-to-quality收益。

暂列B描述调度／编排机制，不认证RNG、逐token版本或异步过滤契约；两项均只核身份与上述局部条件，未全文深读、拉取实现或复现性能。深读0、代码6条／4框架不变，无新设备任务或M5专项交接。下一步补其余五条范围／原生证据，或扩未覆盖框架，不用121／86作为收束依据。

### 5.25 剩余边界与接续线索分级

2026-10-28随C1代码批次附带完成8项范围triage，非独立15–20条T3批次。原三边界补T3；旧五条线索核原站身份／摘要后入主矩阵，其中两项训练侧边界T3、三项直接候选待T2。新入库是接续旧线索，不是新发现五篇或五篇深读。

| 条目与原站 | 当前判断与后续深度 |
|---|---|
| [ASymPO v3](https://arxiv.org/abs/2606.03070v3) | T3保留：算法归一化非执行优化，原站证明错误警示仍在，v2撤回不等v3明确撤回；不据理论保证推进实现。 |
| [CIS v1](https://arxiv.org/abs/2609.32444v1) | T3：置信度相关截断属于更新修正边界，不以准确率宣称rollout更快；与ScaleRLT概率语义对照放入B13深化，未先计T2。 |
| [Each Prompt Matters v1](https://arxiv.org/abs/2512.07710v1) | T3保留：模型报告混合FP8／调度、筛组和目标改变，已有对应系统家族深读，不追加整报告T2或把联合质量收益归纯B。 |
| [PrefixFlow v4](https://arxiv.org/abs/2606.01143v4) | T3入库，训练侧单列：缓存prefix前向、suffix梯度累积及延迟backward；BF16容差不等逐值等价，训练加速不当生成加速。B13深化其跨阶段边界。 |
| [psRL v1](https://arxiv.org/abs/2608.25683v1) | T3入库，训练侧单列：update期不可变数据的prefix共享／KV调度；生产trace吞吐不当完整RL或生成提速。B13与PrefixFlow合读。 |
| [Belayer v2](https://arxiv.org/abs/2608.14635v2) | 直接候选入库、待T2：同时涉及rollout和有副作用环境恢复；限定worker-local故障与健康检查，raw KV分配存活不等request KV内容存活，不降T3关闭。 |
| [BIRD-RL v1](https://arxiv.org/abs/2610.29426v1) | 直接候选入库、待T2：数据库副本隔离／复用是rollout执行问题，curriculum另C；空间效率不当token吞吐。接收自报不替代会议官网核验。 |
| [ScaleRLT abs页所列v1](https://arxiv.org/abs/2610.34427) | 身份现可核、直接候选待T2：looped模型deferred计算与概率replay相关；v1专页仍cache miss，依据abs页的唯一v1版本史，不称全文已读或验证exact保证。 |

**当前范围账**：126＝120唯一arXiv＋6会议；20 serving＋106 RL相关，后者暂89直接候选＋17单列（原§5.20的86／15快照＋新3／2）。不是原生覆盖验收。互斥阅读账126＝20 T1＋78 T2＋24 T3＋4未标完成；4＝HybridFlow进行中1＋Belayer／BIRD-RL／ScaleRLT待T2共3。旧五线索已转入，不另加第二次；B12只完成范围triage，不消除后三项深读债务。

## 6. Paper Deep Reads

### 6.0 Phase 2 冲刺批次表（2026-10-27 用户指令：**截止 2026-10-29 两天内完成**）

**分级深度（权衡深度与时间）**：T1 全模板压缩深读（每篇 15–25 行：问题/性能模型/机制/实验条件/taxonomy/代码映射/疑点）；T2 中等深度（每篇 5–8 行：机制+关键结果+条件+分类核对）；T3 triage（每条 1–2 行：不深读理由+与 rollout 关联判定）。**单次定时触发规模**：T1 触发=2 篇；T2 触发=6–8 篇；T3 触发=15–20 条；Code 触发=1–2 条路径。每触发必须：认领批次→完成→回写本表与矩阵→§0 计数→§16→推送。

**认领规则（防重复）**：取最小号未完成批次；先回写「认领+会话标识+时间」并推送，再开始工作；完成回写「完成+时间」；冲突以远端为准重放（既有惯例）。

| 批 | 层级 | 范围 | 状态 |
|---|---|---|---|
| B1 | T1 | StreamRL、StaleFlow、Laminar、DORA | **完成4/4**：前半M1 new-chat @27c4e00，13:18认领并推送38c0234，13:24完成§6.1–6.2；后半M1 new-chat @743da31，13:47认领并推送b3efd71，2026-10-27 13:52完成§6.3–6.4（T1标准，非实现／复现验收） |
| B2 | T1 | AReaL、RolloutPipe、PACE、ECHO-2 | **完成4/4**：前半M1 new-chat @e159534，14:17认领并推送82a3fb6，14:22完成§6.5–6.6；后半M1 new-chat @fe3bb80，2026-10-27 14:47认领并推送b353e4c，14:55完成§6.7–6.8（非实现／复现验收）。AReaL不接续prompt_mean专项 |
| B3 | T1 | RollPacker、SortedRL、APRIL、DARTS | **完成4/4**：前半M1 new-chat @aecd7f5，15:17认领并推送9139ab3，15:22完成§6.9–6.10；后半M1 new-chat @100f213，2026-10-27 15:47认领并推送828ea15，15:53完成§6.11–6.12（非实现／复现验收） |
| B4 | T1 | RhymeRL、ReSpec、BubbleSpec、DAS | **完成4/4**：前半M1 new-chat @7ae5758，16:17认领并推送d1df96e，16:22完成§6.13–6.14；后半M1 new-chat @6cf28b5，2026-10-27 16:47认领并推送c17ff36，16:54完成§6.15–6.16（非实现／复现验收） |
| B5 | T1 | EfficientRollout、Heddle、ObsSpec、FP8-RL | **完成4/4**：前半M1 new-chat @70c59d3，2026-10-27 17:17认领并推送9dcbdc8，17:23完成§6.17–6.18；后半M1 new-chat @5304df1，17:47认领并推送4e1c946，17:52完成§6.19–6.20（非实现／复现验收） |
| B6 | T2 | L4 其余：AsyncFlow、LlamaRL、RLinf、MindSpeed RL、NeMo-Aligner、ROLL／ROLL Flash、GLM-4.5、Long-RL；补DistFlow／Asynchronous RLHF／ReaL／RLHFuse／OpenRLHF／DeepSpeed-Chat；再补PUZZLE／DynaRL／PivotRL／RobustRL／RLBoost／Relax／ProRL Agent | **完成22/22**：旧九见§6.21–22，补六认领2336d47、23:27完成§6.31。2026-10-27 23:47 M1 new-chat @5868f84认领再补七，与B8 SPEC-RL合批，先推送900f515；23:58完成§6.32。分母15→22，不重复前十五篇 |
| B7 | T2 | 权重同步/失配子簇：SparseRL-Sync、PULSE、NeMo-DCR、ThunderSyncRL、PipelineRL、Missing Old Logits、Beyond Precision、FP16/Diagnosing/Rethinking 三选二 | **完成9/9**：前七篇§6.22；M1 new-chat @1353af5于2026-10-27 19:18认领FP16／Diagnosing（VeXact），先推送cea145d，19:30完成§6.23.1–2；Rethinking未选入，仍留边界未读账 |
| B8 | T2 | L3/spec：TailSieve、WAR、Seer、Infinite Sampling、CadenceRL、Weave、CERO、Scheduling Mixed、SRT、S³R、NgramRL、TLT；补SPEC-RL | **完成13/13**：原十二见§6.23.3–8／§6.24；2026-10-27 23:47 M1 new-chat @5868f84认领SPEC-RL，与B6再补七合批八篇，先推送900f515；23:58完成§6.32。分母12→13，SPEC-RL补采样C边界 |
| B9 | T2 | spec/agentic：RLHFSpec、Bebop、FastGRPO、System-Integrated SD、SPARK、SpecActor、Online Draft Co-Training、MTP-RL、AReaL-DTA、FlashNext-RL、Rollplex | **完成11/11**：前6完成§6.25；2026-10-27 20:48 M1 new-chat @34e8adb去重AReaL-DTA／AREAL-DTA同一2602.00482身份后认领余5，先推送bc0649d，与SkyRL-Agent合批完成§6.26；训练侧边界保留，不增源码／复现计数 |
| B10 | T2 | agentic：SkyRL-Agent(广扫深化)、EARL、InfoTree、AsyncWebRL、AcceRL、MARLaaS、FlexMARL、Agent Lightning v0/v1.0、TideRL、ROSE、QwenGyre | **完成12/12**：SkyRL-Agent见§6.26.6，八篇见§6.27；余三认领先推送c96756b，与B11前五篇合批完成§6.28.1–3 |
| B11 | T2 | 资源/边界：ARL-Tangram、DART、MCP-Universe RL、PEARL、SAO、Rollout Infrastructure Tax、ExpThink、Intelligent Router；补AgentRL／RollArt／AstraFlow | **完成11/11**：前五见§6.28.4–8；2026-10-27 22:18 M1 new-chat @81084bc认领余三＋补三，先推送8aac21e，22:29完成六篇§6.29。按允许微调补三篇、分母8→11；Intelligent Router仍serving迁移边界，不转原生覆盖 |
| B12 | T3 | serving 20 条 + 其余边界条目 triage（不深读，逐条给不深读理由+关联判定） | **范围triage完成28/28**：首批认领a65cb4d、§6.30完成20项；余八随C1认领d86477d，2026-10-28 00:29完成§5.25。原三＋新二补T3，另三直接候选转B13待T2，Router沿既有T2；28事项不当28 T3身份 |
| B13 | T2 | Belayer、BIRD-RL、ScaleRLT；边界深化PrefixFlow、psRL、CIS | **完成6/6**：2026-10-28 00:47 M1 new-chat @440db98认领、先推送505a5cf，00:58完成阅读回写及检查；见§6.33。三边界T3→T2互斥替换，累计20/84/21/1。不转原生覆盖、无实现／设备验收 |
| C1 | 代码 | veRL TransferQueue/TQ 数据面 + AReaL 调度器 | **veRL部分完成1路径**：2026-10-28 00:19 M1 new-chat @2508282认领、先推送d86477d，00:29完成§8.8；累计7路径／4框架。AReaL调度器未开展，不将C1整项写完成，M5专项不触及 |
| C2 | 代码 | vLLM / SGLang core scheduler 路径 | **完成2/2路径**：2026-10-28 01:18 M1 new-chat @36e1ad9认领、先推送e5b874b，01:29完成检查回写；§8.9–8.10固定eaf1627c／cb55d77d各1条，9源码AST／8组原函数体控制通过。累计9路径／6系统，不当原生引擎／性能或RL复现 |
| C3 | 代码 | rLLM / slime / 其余框架 | **本批完成2/2路径**：2026-10-28 01:48 M1认领ef69115先推送、02:00收尾；§8.11–8.12固定rLLM 4e828c81／Slime 540ea919，13源码AST／八组限定CPU控制通过；累计11路径／8系统，非全框架／性能验收；HybridFlow与覆盖审计接续 |
| C4 | 代码／覆盖核查 | 公开历史veRL生成与重分片1路径，附Phase 2覆盖缺口 | **认领0/1路径**：2026-10-28 02:18 M1 new-chat @8b82580，先推送；先定位历史ref，不将2026的80ee82bd当2024实现；不接管HybridFlow并行认领、论文20/84/21/1及代码11不预增 |

（HybridFlow 已由并行会话进行中，不计入批次。T2/T3 具体名单允许执行会话按 §5 矩阵实际条目微调，微调须记录。）

**深读顺序（覆盖四维与机制家族，交替推进）**：① HybridFlow/veRL（L4 元框架，本地有源码，进行中）→ ② AReaL（L4 async 旗舰）→ ③ StreamRL（L4 disaggregation+气泡）→ ④ RollPacker（L3 长尾）→ ⑤ RhymeRL（L1/L2 speculative）→ ⑥ BubbleSpec（L1/L3 气泡→draft）→ ⑦ StaleFlow（L4 staleness）→ ⑧ Laminar/DORA（L4 新一代）→ ⑨ RollArt/AstraFlow/Heddle（agentic）→ ⑩ APRIL/PrefixFlow（新补小簇）。

**深读模板**：问题/瓶颈 → 性能模型与指标 → 机制 → 实验条件与增益（必须保留硬件/batch/长度分布；未核实处标 unknown）→ 与 taxonomy 对照（层 + A/B/C）→ 代码映射（有源码则实证路径）→ 可迁移性 → 疑点与未验证项。

### 深读 #1 — HybridFlow/veRL（2409.19256，EuroSys'25，L4，类型 B+基础设施）进行中 2026-10-27

- **问题/瓶颈**：RL 建模为 dataflow（节点=NN 计算、边=数据依赖）；RLHF 使节点膨胀为分布式 LLM 训练/生成程序、边变为多对多多播——单控制器（调度灵活但 dispatch 开销大）与多控制器（执行高效但不灵活）的两难；actor 在 train/generation 两态间 reshard 的内存冗余与通信。
- **性能模型与指标**：throughput（RLHF 训练端到端）；3D-HybridEngine 目标 = reshard 时 "zero memory redundancy"。
- **机制**：①hybrid 编程模型（单控制器编排 + 多控制器执行的混合）；②分层 API 解耦 computation 与 data deps（transfer protocol）；③3D-HybridEngine（train↔generation 参数 reshard 协议）。
- **实验条件与增益**：[v2 §8.1–8.2](https://arxiv.org/html/2409.19256v2)最多16节点×8 A100-80GB、节点间200Gbps；Llama 7B–70B，基线DeepSpeed-Chat 0.14.0／OpenRLHF 0.2.5／NeMo-Aligner 0.2.0。prompt／response均固定1024、global batch 1024，warmup 10轮后平均5轮；吞吐为整RLHF迭代的prompt＋response tokens/s，不是decode TPS。图9(c)及正文将20.57×对应34B PPO与NeMo-Aligner；具体GPU横轴点仍待核图，不外推现代长尾／agent场景。
- **与 taxonomy 对照**：L4 的"元框架"（后续 rollout 系统多构建于其上或与之对比）；B 类为主（execute set better）。
- **代码映射与年代**：11:53接续已有trainer／workers／protocol／sharding_manager及`80ee82bd`映射自报，本轮未重核该快照。已核现代实现见§8（veRL `24f25b03`等）；现代agent_loop不能替代论文证据，旧论文也不替现代实现背书。
- **映射成本**：[§6算法1／附录C](https://arxiv.org/html/2409.19256v2)在候选映射中最小化预测成本。以stage为`s`、同置组为`G`，可重写为`C = Σ_s max_G Σ_{l∈G} simu(l, W_s)`：组内相加、组间取最大、跨stage累加。解析估计与同构设备是前提，不等在线普适最优；图16联合改变模型／GPU数，不证明渐近线性。
- **图表核对**：[v2 PDF，第11–13页](https://arxiv.org/pdf/2409.19256v2)图9(c)有32／64／128 GPU点，但未逐柱标20.57×，不猜精确对应点。图12同尺寸、图13异尺寸，按PDF图题而非错位HTML引用区分。图14的transition不当全步，图15的16 GPU／training TP8／KV条件不外推为更小generation TP总更优。
- **重分片与待核项**：[v2 §5.1–5.4／§7](https://arxiv.org/html/2409.19256v2)同组actor GPU顺序train／generate，重排generation分组以复用training权重。零冗余仅限重分片权重；表2峰值仍为`M/(t_g p_g)`，§7仍有独立buffer和CPU offload，不等全显存零开销／无传输。成本模型、关键消融与附录已补核；原版实现及20.57×上界原始数据仍待定位。全文完结0、进行中1；下一步以论文年代代码对应为主，不继续反复概括摘要或从现行代码代填旧实验。

### 6.1 StreamRL T1压缩深读

- 状态：2026-10-27完成T1；非代码／性能验收。
- 原文：[v1 §3–7](https://arxiv.org/html/2504.15930v1)；固定版本不追认后续实现。
- 问题：阶段气泡与生成长度偏斜。
- 阶段模型：最小化`max(Tgen,Ttrain)`，依赖profiling。
- 长尾模型：`PTL(BS)×L`，KV容量约束batch。
- 流水：SGS逐完成stream；同步仍有同批依赖。
- 异步：one-step路径，不能冒充同步等价。
- 调度：ranker排长短、长尾小batch、LPT补位。
- 论文映射：SGS→RL-RPC→Trainer；自研C++/CUDA，非baseline的vLLM。
- 条件：Qwen2.5 7/32/72B，PPO，global batch=1024。
- 硬件：最多128 H800-80GB；warmup后20步samples/s。
- 负载：CodeMath长尾，5/10/20K为**强制输出长度**实验。
- 同后端对照：Sync／ColocationRL为1.06–1.41×。
- 总上界：Async／veRL为2.66×，非调度单项收益。
- 跨域成本实验另用32 H20、80Gbps链路；不混算收益。
- 分类判断：L3/L4、B；异步版本／收敛前提单列。
- 迁移判断：需重做负载profile及policy版本审计。
- 代码缺口：未核公开commit／文件，不增§8路径数。
- 待核：ranker持续更新总成本及原始数据；长度控制结果不能替真实在线生成验收。
- 后续：实现核查留代码批；本轮无运行或算力交接。

### 6.2 StaleFlow T1压缩深读

- 状态：2026-10-27完成T1；非代码／性能验收。
- 原文：[v2 §3–6、附录B/C](https://arxiv.org/html/2601.12784v2)；会议接收仍仅作者自报。
- 问题：有界陈旧性下的实例内／间偏斜。
- 年龄：`Vtraj+η≥Vbuf`；分段生成以最低允许版本记账。
- 协议：Reserve最迟位置→Occupy最早位置→Consume。
- 架构：TS存轨迹、CPU PS存权重；Interrupt不等Abort。
- 模型：`TPS=n/[k1·KV+max(k2,k3·n)+k4]`，离线拟合。
- 调度：旧版本优先＋边际吞吐路由；按需同步／迁移。
- 快照须匹配命令预期状态，不等运行故障事务证明。
- 条件：DAPO，32–128 H20；含7B/14B/32B/70B及30B-A3B。
- 负载：GSM8K/MATH或DAPO-Math-17k；prompt≤2K、response≤20K。
- 单步128组×16=2048轨迹；指标为端到端tokens/s。
- 主评估禁用冗余；对VeRL-Async最多1.42×、均值1.18×。
- 消融／拟合另用40K、η=3；不与20K主评估混同。
- 冗余消融会缩短输出；另有Stuck超时Abort路径。
- 分类判断：调度B；冗余／过滤C，后续重采prompt不证明无偏。
- 迁移判断：须保留group完整性、最低版本与消费槽语义。
- 论文实现链接：[psrl-project/psrl](https://github.com/psrl-project/psrl)，不是训练前缀共享psRL论文。
- 代码缺口：未核commit／文件，不能宣称协议已在源码验收。
- 待核：逐token概率／过滤配置和跨任务收敛；η=3个案非普遍保证。
- 后续：不按支持冗余推断主评估启用；本轮无硬件复现。

### 6.3 Laminar T1压缩深读

- 状态：2026-10-27完成T1；源码／性能未验收。
- 原文：[v1 §3–8及附录A–D](https://arxiv.org/html/2510.12633v1)。
- 问题：全局同步与长尾导致rollout闲置。
- 调度模型：同版本bin packing，受KV容量与roofline batch双约束。
- 权重路径：actor→CPU master relay→RDMA链→本地PCIe。
- actor仍有发布停顿；广播后台化不等无同步成本。
- repack集中同版本长尾，释放副本更新权重。
- 陈旧性`M-K`来自生成耗时；最大4是**观测**，非配置硬界。
- buffer允许自定义采样／淘汰；收敛实验用FIFO。
- 条件：Qwen2.5系列7/32/72B，最多1024 H800-80GB。
- 负载：DAPO-Math-17k／tool，输入≤2K、输出≤16K长尾。
- batch=512 prompts×16=8192；GRPO Clip-Higher。
- 指标：整迭代prompt＋response tokens/s；warmup10后均值5步。
- 数学任务对veRL v0.5.0平均2.56×；不是纯repack收益。
- repack消融：32B／128GPU，generation吞吐+26%，非全步。
- 吞吐／收敛实验并发分别1024／256每副本；基线配置另调。
- 故障恢复252s含新机初始化；链修复秒级不等全任务秒级恢复。
- 分类判断：L3/L4、B；buffer选择／淘汰对集合的影响另核。
- 论文映射：rollout manager／三池／relay／trainer，基于veRL。
- 代码缺口：未核公开commit／文件；不能用现行veRL代填。
- 迁移判断：核同版本续跑、group完整性与实际消费年龄。
- 未决：RNG／故障重放一致性、采样偏差；不作普适无偏结论。

### 6.4 DORA T1压缩深读

- 状态：2026-10-27完成T1；源码／性能未验收。
- 原文：[v2 §3–5、附录B–D](https://arxiv.org/html/2604.26256v2)。
- 问题模型：同步decode近似`τ·max_j max_i L_i`。
- stream超供给（作者记`RBS>TBS`）；未完轨迹留原版本续跑。
- 窗口：最旧版本全部收集并转训后才前移，不主动丢长尾。
- 调度：按版本负载重分DP资源，旧版本亦可补充请求。
- KV迁移限同版本；省re-prefill不等零传输成本。
- 论文映射：RolloutManager→TransferQueue→Trainer＋orchestrator。
- 条件：Qwen2.5-32B，64／128 H800；同内部框架基线。
- 负载：DAPO-Math-17k，输入≤2K、输出≤30K；均长2.4K。
- 配置：512 prompts×16=8192；16次update，micro-batch512。
- 口径疑点：正文RBS称prompt、TBS称trajectory；不直接比较512与8192。
- 指标：实际消费prompt＋response tokens/s，warmup后均值5步。
- 128GPU端到端吞吐为同步2.12×；非直接跑公共框架排名。
- 64GPU的8.2×仅**未被训练遮蔽的rollout-only时段**。
- KV消融183→166s为rollout计时；不可与全步收益相乘。
- 生产4096非CUDA设备／约500B MoE结果另列，不外推H800。
- 收敛仅72GPU／100步奖励曲线，K=3较K=1略慢。
- 分类判断：L2/L3/L4、B；保留轨迹不证明跨步训练集合等价。
- 理论为抽象混合行为策略分析，不替完整GRPO动态保证。
- 迁移判断：需核KV布局／数值兼容，不仅检查版本标签。
- 代码缺口：内部实现未核commit／文件，不增§8路径。
- 未决：窗口版本与实际消费时钟、GRPO组归属／概率字段。
- 后续：这些实现条件留代码核对，不新排设备任务。

### 6.5 AReaL论文 T1压缩深读

- 状态：2026-10-27完成T1；不接续prompt_mean专项。
- 原文：[v5 §4–7及附录A–E](https://arxiv.org/html/2505.24298v5)。
- 问题：长尾与生成／训练串行空闲。
- 架构：controller串起rollout、reward、一次消费buffer和trainer。
- 换权重会中断、重算KV再续跑，单轨迹可混版本。
- 限流模型：`floor((Nr−1)/B)≤i+η`，旧样本优先。
- 目标：behavior作IS分母，proximal作clip中心；不是原PPO不变。
- 条件：R1-Distill-Qwen 1.5–32B；主实验128–384 H800。
- 分配：生成占3/4为启发式，不是通用最优。
- 训练：512 prompts×16，4 minibatches；数学／代码η=8／4。
- 长度：表3训练prompt≤1024、generation≤27648；非表1评估32K。
- 数据：DeepScaleR数学／DeepCoder代码；PPO无critic／reference。
- 指标：扩展性用有效消费tokens/s，非生成总TPS。
- 表1：1.5B／16节点／250步，Sync.AReaL 41h→14.8h（2.77×）。
- veRL小时数由新测吞吐估算，质量含既有引用，不能当全程配对实测。
- 证据强度：固定seed=1，大规模端到端单次，无误差条。
- 论文栈：ReaLHF＋SGLang0.4.6＋MCore0.11；非当前release。
- 算法消融支持受限陈旧性，不把η≤8当跨任务保证。
- 分类判断：L2/L4系统B；混版本及训练目标变化单列。
- 行为策略存在性命题，不证明单一固定参数网络或整体无偏。
- 代码缺口：仅论文模块映射，未核commit／文件，不增§8路径。
- 迁移判断：逐token概率、准入计数单位和消费年龄仍需实现证据。
- 未决：正文“过滤”与具体准入／淘汰动作须分开，不凭措辞认定丢样本。
- 本轮无源码修复、探针、设备任务或专项判断变化。

### 6.6 RolloutPipe T1压缩深读

- 状态：2026-10-27完成T1；非代码／正确性验收。
- 原文：[v2 §2–5](https://arxiv.org/html/2606.26997v2)。
- 问题：完整group已可训练，却被全轮屏障挡住。
- 模型：首批可提前`Δt=tcomplete−tfirst(U)`，非全步收益等式。
- CGP：group含K个已评分response，materialize后FIFO送训。
- FGD：提交序最早的至多Fw组准入，其他defer非丢弃。
- serving-complete与trainable-complete是两种完成信号。
- 每U组optimizer step；全R组消费后才向rollout发布权重。
- 因此固定生成权重不等trainer全轮不更新。
- 论文映射：FGD Admit→Control materialize／handoff→Train RPC。
- 实现基于Slime／SGLang／Megatron／Ray，未核commit／文件。
- 条件：Qwen3-1.7B；训练8×4090-24GB、生成2×A100-40GB。
- 负载：LSAT-AR／Sci-XW／Sci-JL／OlyPhys，R=32/64/96。
- K=8、B=16→U=2；Fw=2，每点4轮均值与样本标准差。
- 响应均长约3.3–3.6K；最大长度配置unknown，不以均值代替。
- 主指标为rollout-start至train-end，不默认包含下轮权重刷新。
- 对Slime主时窗缩短30.7%–42.3%，非tokens/s增幅。
- FGD在CGP上另缩短2.5%–11.4%，不与总收益相加。
- 等待比Eq4为首启窗口，§5另写wait/(wait+compute)，实现口径待核。
- 分类判断：L3/L4、B；组顺序／dynamic filtering影响另核。
- 迁移判断：需完整group、token预算及U组累积契约。
- 未决：完成序FIFO不证明各更新组集合与原顺序完全一致。
- 四轮时间／负载对照不替长期收敛或多模型验收。
- 本轮无作者框架执行、算力交接或社区动作。

### 6.7 PACE T1压缩深读

- 状态：2026-10-27完成T1；非实现／性能验收。
- 原文：[v1 PDF §2–4、附录A/B](https://arxiv.org/pdf/2609.36830v1)，公式与表1–3核原页。
- 问题：生成完成后积压，使已饱和trainer的数据继续变旧。
- 模型：token均值年龄`Kgen+Kwait`，非最大token年龄。
- 预算：`clip((N−Ntar)/max(N,1),0,1)`后EMA平滑。
- 排序：`Keff=Kwait+F(s_pre)·Kgen`，按分位数拒收。
- prefix分数为平均绝对log-ratio；F用近期相对CDF，非全局KL。
- 仅完成池拒收、继续抽满批；group评分取均值。
- 论文映射：vLLM元数据／prefix rescoring→pool admission→trainer。
- 主8B配置12r4t；GPU型号unknown，不填H800。
- 数学训练每更新12 prompts×12 responses=144轨迹，GRPO。
- DAPO-MATH-17K，prompt≤2048、response≤8192。
- 表1取300步内六benchmark均值最高的一次评估，mean@32。
- 主8B：41.29%对Async 36.34%，差4.95个百分点，非18.7个百分点。
- 图1的47.1%基于logged training-step time，不当全流程账单。
- 图8六GPU计时消融不代替主8B的16GPU配置。
- ReTool中Sync／Async 4卡、PACE 6卡；不称同GPU预算比较。
- supply threshold=8约束在途工作，不是每轨迹年龄硬界。
- 分类判断：L4、B/C；loss不变不等训练分布不变。
- 迁移判断：需token版本／行为概率、组完整性与prefix重评分接口。
- 低prefix漂移不证明低梯度偏差，拒收也非无偏采样保证。
- 代码缺口：未核公开commit／文件；不增§8路径数。
- 未决：Ntar与评分窗口对饥饿／拒收偏差的影响及完整成本。
- 后续：仅保留实现核查项，不新建算力队列。

### 6.8 ECHO-2 T1压缩深读

- 状态：2026-10-27完成T1；非实现／性能验收。
- 原文：[v6 §3–6、附录B–E](https://arxiv.org/html/2602.02192v6)。
- 问题：WAN权重分发与异构rollout的成本／空闲权衡。
- 容量模型：`μmin=κR/(κTtrain−Tbcast)`，分母须正。
- μ含生成、reward、调度和网络，不是原始decode TPS。
- 按c/μ贪心激活，γ=1.1留余量；非全局整数最优保证。
- κ=S−1的保守年龄推导有overlap及Tbcast/Ttrain<1前提。
- 消费要求`v≥vt−S`，更旧轨迹丢弃；S不是buffer大小。
- 分块到达即可转发，但完整snapshot安装后才用新版本生成。
- 论文映射：Parallax worker→versioned buffer→central GRPO learner。
- 主4B/8B：4 A100-80GB learner＋5090 pool；中央基线8 A100。
- DAPO-Math训练、AIME24 avg@64；GB128、n16、response≤8192。
- WAN消融B0=300–800Mbps、worker上限100Mbps，非全部场景固定值。
- Tbcast以约90.9% worker装好快照计，不是全员最大延迟。
- 成本=GPU-hours×2026-05-01租价，非实际账单／当前报价。
- 8B同AIME质量成本降33.3%–36.3%，不称同硬件更快。
- 主对照step约1508s对1631/1649s，成本收益不等wall-time收益。
- 32B对Prime-RL的4.6%另有硬件／网络对照，不与前项相加。
- S≤6的稳定性属所测GRPO任务，S=11可发散，无普适收敛证明。
- 单worker恢复240–312s为γ=1.0消融，非大规模故障保证。
- 分类判断：L3/L4、B/C；目标不变不能排除丢弃的集合影响。
- 迁移判断：需重测有效μ、分发分位数与消费版本时钟。
- 代码缺口：未核commit／文件；batch单位／group保留契约仍unknown。
- 后续：校准／部署均未启动，不因廉价worker自述排设备任务。

### 6.9 RollPacker T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §3–7](https://arxiv.org/html/2509.21009v1)。
- 问题：同步轮被少数长响应拖尾，需计完整短／长轮周期。
- 短轮以η=1.25超发prompt及响应，保留最快完成者。
- 被中止prompt入长队列，满P0后无超发重跑长轮。
- prompt最终纳入，不证明原响应保留或条件分布不变。
- 流式反传只缓存梯度，整轮结束前不动参数／optimizer。
- 最后按副本已处理样本数重标梯度，再全局同步更新。
- TP按抢占反馈倍增／减半，受单节点及通信组完整性约束。
- sandbox超时`clip(1.5Tanchor,2s,30s)`，超时记零分。
- judge与actor用MPS共卡，层权重CPU卸载／PCIe流水。
- 论文实现链：ROLL→vLLM 0.8.4／Ray→Megatron 0.12.2。
- 迁移保留tokens、重算KV；并非旧KV无条件复用。
- 主评估Qwen2.5 7/14/32B，响应上限8/16/32k。
- 对应16/32/64 H800；128卡为集群及扩展上限。
- GRPO P0=128、R0=8，数学／代码／judge任务均匀混合。
- 整体对veRL为2.03/2.22/2.56×，非全归tail batching。
- 表2逐行累计收益，不能再相乘；图9展示前40步时长。
- 扩展实验batch与GPU同增，不是固定batch强扩展。
- 判断：同步版本与延迟更新可保留，采样等价须另审。
- 分类：L3/L4、B/C；最快响应选择与评分超时单列C。
- 代码缺口：v1称待开源，未核公开commit／文件。
- 迁移需响应取消／组归属、归一化及超时误判契约。
- 后续：保留实现核查，不新增设备队列或外部动作。

### 6.10 SortedRL T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §3–4、附录B](https://arxiv.org/html/2603.23414v1)；PDF表1／图4补核。
- 问题：超发早收获减少空队列，却会偏向短响应。
- group加载nb个prompts，池清空前不补新，避免饥饿。
- 此group是prompt批次池，不等GRPO同prompt响应组。
- on-policy丢未完响应片段，只留prompt用新policy重跑。
- partial续旧tokens，拼接各段生成时的行为logp。
- 判断：精确行为分母不等整轨迹当前policy或无偏证明。
- 长度分批改变Reinforce++批归一化及更新次序。
- 论文实现链：veRL＋SGLang状态接口＋rollout buffer。
- Logic：Llama3.1-8B-Instruct、Reinforce++，128×8轨迹。
- 数学：Qwen2.5-32B、PPO／DAPO-Math；每更新128轨迹。
- baseline一次生成512；Sorted分4次128，池总512。
- 论文报H100／MI300X集群，逐实验卡数配置仍unknown。
- 表1：AIME mean@32为on-policy 23.33、partial 20.83、基线19.69%。
- §4.3正文归属顺序相反；原PDF同样矛盾，依表注明而不静默纠正。
- GSM8K反向、partial在Logic失败；不能称全面质量提升。
- rollout TPS为3987→4289／5559，非全训练time-to-quality。
- bubble按请求队列空槽计，74%→5.81%／3.37%，非SM空闲率。
- 吞吐用固定匹配生成长度、512样本／4批、上限8k校准。
- n=8亦退化；防饥饿策略不保证任意分组规模有效。
- 分类：L3/L4、B/C；micro-curriculum不是纯执行等价优化。
- 代码缺口：未核commit／文件及KV跨权重处理契约。
- 后续：版本／归一化接口待源码核查，不排新设备任务。

### 6.11 APRIL T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v3 §2–3、附录A.2–3](https://arxiv.org/html/2509.18521v3)。
- 性能模型：等全批最长响应→等第N个完整prompt组。
- 超发64 prompts×8响应，目标32组／256训练轨迹。
- 同prompt八响应须齐备；先完成的子集暂存，不拆组更新。
- 未完tokens入buffer，下步优先续跑，再补新请求。
- 续跑suffix用更新policy，优势对应混版本轨迹。
- 文中m≤5属观测，不是年龄硬界／收敛证明。
- 分类：L3/L4、B/C；每步集合与版本变化不因最终保留而消失。
- Qwen3-4B／8B，主评估GRPO和DAPO，不外推GSPO全表覆盖。
- DAPO-Math／DeepScaleR／DeepMath训练，AIME2024评估。
- 主报告8×MI300；另有H100兼容验证，不混硬件收益。
- eval每prompt16次；训练global batch256。
- 响应上限16384、温度0.8；32k纯rollout分析另计。
- 吞吐=8卡生成tokens／rollout迭代wall time。
- 平均+22.5%为rollout throughput，不作全训练提速。
- 表1最大+49.5%；abs页“44%”不一致，保留版本证据差异。
- 表2差值均值约+2.1个百分点，非2.1%相对提升。
- 8B DAPO三项两负一正，不称逐项无损。
- 约40%混旧比例在正文／图注有token与rollout单位差异。
- 论文映射：slime续跑buffer＋HIP torch_memory_saver。
- 后者保留虚拟地址不等保留释放后的KV内容。
- 未核commit／文件，行为logp／KV重建契约待源码确认。
- 后续：只留实现核查，无新设备任务或外部操作。

### 6.12 DARTS T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §3–5、附录A–I](https://arxiv.org/html/2605.30859v1)。
- 问题：prompt内长尾；每prompt生成M′≥M候选。
- 主组7最短＋1最长，长端排除未完／到限无效轨迹。
- 改组均值和相对advantage，不是仅改变执行次序。
- 两种长短／奖励模式用于解释，并非在线显式分类器。
- 成本模型ΣΔl·PTL(TP,剩余batch)，PTL靠同引擎实测拟合。
- 总预算clip(⌊ρ/(λk)⌋,MN,2MN)，来自代理utility，非质量最优证明。
- 分配按`w/[m(m+1)]`边际收益贪心，M≤m≤2M。
- 饱和m=2M时改只取M个最短有效完成，提前停余项。
- token块到达后可增量forward，训练端维护轨迹KV。
- reward和backward仍等完整轨迹，不是边生成边更新。
- 论文链：veRL0.4→vLLM0.8.5.post1／Ray actor→PyTorch2.6 FSDP。
- H20-96GB：3B用8卡，7/14B用32卡，32B／30B-MoE用64卡。
- 3B用MATH-lighteval，其余DAPO-MATH，Qwen dense／MoE。
- GRPO M=8，沿用clip-higher／token loss／overlong shaping。
- 500步平均端到端吞吐对veRL为1.29–1.77×，非同策略执行收益。
- 对Tail Batching最高1.43×，该基线是同栈移植而非完整RollPacker。
- 14B消融1.09→1.40→1.63×为累计，不再相乘。
- BBH的30B-MoE为85.8→84.4，不称所有任务均改善。
- 固定长度／极稀疏reward及大规模agentic适用性有限或待评估。
- 分类：L3/L4、B/C；分布塑形与流水系统收益分开。
- 未核公开commit／文件；筛选后激活生命周期与重算仍待源码核查。
- 后续：保留条件与接口疑点，不开算力队列或社区事项。

### 6.13 RhymeRL T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §4–7](https://arxiv.org/html/2508.18588v1)。
- 问题：重复prompt的token／长度历史可摊薄decode与长尾成本。
- HistoSpec每prompt建后缀树，历史worker在CPU异步索引。
- reward和分支优先级只选draft，不直接筛actor训练样本。
- draft经当前target验证，历史token不直接当旧轨迹回放。
- 窗口从2起、全收加2至32、拒绝重置2；prefix从7退至3。
- 大batch按剖析阈值关SD，避免验证开销反噬。
- HistoPipe按组中位长度排序，奇偶step反向分组分配。
- Tier-2用τ(length,DP)查表，二分最小化组时长梯度d。
- 权重先到host buffer，worker边界安装；非全局同步屏障。
- 迁移保留tokens、重算KV；跨步迁移还改变训练步归属。
- 分类：HistoSpec A，HistoPipe B／跨步另C；整系统含L4。
- HistoPipe首epoch关闭；历史相似性不保证未见prompt收益。
- trace匹配93%／75%与运行draft接受率65%–79%不是同指标。
- 主Qwen3-8/14B-Base、Qwen2.5-32B，GRPO group16。
- 主测试16节点128 GPU，型号未列；千卡为部署自述。
- 内部数学／代码取8K prompts，训练80／100步、前20步warmup。
- 8K／16K上限下报samples/s，不是tokens/s或time-to-quality。
- 对veRL0.4.1最高2.6×；对AReaL0.3.0须区分lag=1／8。
- 历史树在8节点230K／16K配置每节点<80GB，非零主存成本。
- §7.3一步陈旧与reward曲线重叠，不作严格同步等价／普适收敛证明。
- 论文基于veRL、称将开源；未核commit／文件及逐token验证契约。
- 后续：保留实现与组版本疑点，不开设备任务或接续M5专项。

### 6.14 ReSpec T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §2–6、算法1](https://arxiv.org/html/2510.26475v1)。
- 近似接受token成本`(Cq+Cp/α)/r`，依赖验证并行假设。
- Solver按active batch及离线剖析选择(s,t,n)，大batch可关SD。
- 非SD转SD用prefill补drafter KV，切回只丢推测元数据。
- 存target logits及reward，梯度对当前drafter计算。
- reward加权`w(r)Σ KL(p∥qθ)`是drafter KD，不直接改actor RL loss。
- 每I轮更新／清buffer，与生成重叠；不是actor异步更新间隔。
- 算法遍历全Q与正文只用1/I数据不一致，精确消费量待核。
- I=1／3／5消融属drafter新鲜度；奖励差异不作因果证明。
- 论文实现链：veRL＋SGLang／EAGLE-3，未核源码版本。
- 测试床两节点各8 H100-80GB，PyTorch2.7.1／CUDA12.6。
- 3B用1生成＋1训练GPU，7/14B各8＋8；非全都16卡。
- 3B上限3072／温度0.6，7B8192／0.6，14B8192／0.4。
- GRPO数学实验；数据集身份／完整batch配置未列清。
- validation每20步，图13速度曲线20点平滑。
- 3B峰值4.53×／中位1.84×，不称全程平均4.5×。
- 7/14B平均1.69／1.50×；14B组件消融1.48→1.66→1.78×累计。
- §3由多token p/q方差推输出偏差，审读判断：论证不足。
- [精确SD原论文](https://proceedings.mlr.press/v202/leviathan23a.html)在接受／残差校正前提下保持分布。
- 推断：固定target且每前缀条件采样为p，则整轨迹为∏p，方差不推出偏差。
- 奖励下降可留作实验现象，不能据此定位采样器bug或证明收敛保证。
- 分类暂A，不因reward训练draft自动标C；实际lossless待实现核验。
- 未核commit／文件，残差／logp／权重快照契约待查，不排新设备队列。

### 6.15 BubbleSpec T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §2–4、附录A](https://arxiv.org/html/2605.08862v1)。
- 问题：同步rollout的空闲rank可提前生成下一批draft，免等完整epoch。
- 只利用inter-GPU气泡；intra-GPU补请求会干扰当前长尾。
- 每50个decode步轮询Ray协调器；当前批全完成即停预生成、过屏障。
- draft按prompt分发并本地建后缀树；旧策略token不直接进入actor训练。
- 确定性proposal q(d)=1，按当前实际采样分布p(d)接受。
- 拒绝后采`p(x)1[x≠d]/(1−p(d))`；须包含温度／top-p等变换。
- 多token按前缀顺序验证，首拒绝后重启；此契约下保分布，非实现认证。
- 式1的`1−(1−α)(1+μ)`为时间缩短比例，不是倍数speedup。
- 统一短query attention避免spec／普通请求拆batch、分开launch。
- 论文链：veRL0.5.0.dev0→vLLM0.10.1，FlashInfer算子／Arctic-Inference后缀树。
- 单节点8 NVIDIA GPU、TP1，未列型号；显存比例0.7，开CUDA graph／chunked prefill。
- Qwen3-1.7/4B冷启动，VL7B从AceReason SFT；Polaris／DeepMath／SimpleRL数据。
- 主SAPO，64 prompts×16响应；温度1，固定draft4；上限48K／VL64K。
- 最慢DP rank墙钟、50步平均：306.2→167.6、399.6→235.0、571.9→400.1秒。
- 上述为rollout缩短45.2／41.1／30.0%，非全RL或time-to-quality收益。
- 平均accept length含恢复／bonus token，不等于纯draft接受数。
- 树构建均值1.2／1.2／2秒；隐藏到reward／update后是优化建议，非已证全遮蔽。
- 1.7B不用统一attention为294.9秒，n-gram为305.9秒，均不可只数decode步。
- 预生成4／16／32响应为183.8／167.6／184.4秒，覆盖与并发开销有权衡。
- VL7B AIME24为veRL80.72／Bubble79.58；近似质量不等逐点同分或收敛证明。
- 分类L1/L3/L4、A/B；跨批draft编排不自动变C，partial draft不是partial训练轨迹。
- 未核公开commit／文件及首步初始化；保留接口疑点，不开设备或M5专项队列。

### 6.16 DAS T1压缩深读

- 状态：2026-10-27完成T1，非实现／复现验收。
- 原文：[v1 §3–5、附录C](https://arxiv.org/html/2511.13841v1)；[PDF第7页式3–9](https://arxiv.org/pdf/2511.13841v1)已整页渲染核对。
- 问题：长轨迹决定makespan，统一加长draft会增加全批验证成本。
- 近期同题rollout后缀树作非参数draft，增量更新、最长匹配查询O(m)。
- 全局历史树查询贵且含陈旧轨迹；同题分片＋可选request前缀路由。
- 小模型可关前缀trie；更高接受率不保证更低总latency。
- 每轮刷新滑窗，更新幅度大则缩窗／旧匹配降权；16／32窗口是经验平衡。
- 按历史及已生成前缀动态分Long／Medium／Short，短类关闭SD。
- 分配的是draft预算，不直接过滤actor轨迹；reward／optimizer不变。
- 成本代理`T=cbase·Nfwd+ctok·Ntokens+C`；线性剖析平均相对误差约12%。
- 接受模型`A(p)=kl(1−exp(−αp/l))`，k≤1；不是普适接受定律。
- 独立代数核验：式7乘k与式6紧约束不一致，PDF同HTML，非转换错误。
- 反例l=100,k=.5,α=1,N=75：式7得p=13.3531，代回剩余93.75≠75。
- 按该模型应为`p=−l/α·ln((N/l−1+k)/k)`，有限正解需l(1−k)<N<l。
- 式8–9沿用式7，故不作最优预算证明；实际采用三级启发式，不能据公式判代码bug。
- 论文链：修改veRL＋vLLM；target逐token验证／残差契约及commit／文件未核。
- 数学：DSR-sub 1209题、蒸馏Qwen7B，1节点8 H100、batch128×16响应。
- 数学上限16K、温度.6、30步，报rollout时间降低>50%，reward曲线近似。
- 代码：Qwen3-8B、2节点16 H100，8响应、每GPU batch32、上限16K、温度.6。
- 代码执行用Ray／Xeon8468V，正文报rollout约降25%；不泛化为两任务均>30%。
- 摘要1.5–8B与§5约14B／最多6节点自述分开，不给主结果套48卡。
- Unlimited-budget消融最多再改善15%；不与相对veRL收益连乘。
- 分类L1/L3、A/B暂核；保分布取决于正确验证，不以reward近似替代证明。
- 后续保留预算推导／batch单位疑点；未复现实测，不排新算力项或向社区写入。

### 6.17 EfficientRollout T1压缩深读

- 状态：2026-10-27完成T1，非源码／性能复现验收。
- 原文：[v2 §3–5、附录A–F](https://arxiv.org/html/2606.18967v2)。
- 问题：policy变化与active batch收缩，使固定drafter／全程SD失效。
- 每步从当前target的FFN／QKVO做RTN W4，非低精度policy直接供训练。
- embedding／LM head／norm保高精度；draft权重约原33–36%，非25%。
- `speedup=τTp/(γTq+TV)`；τ含target补采token，几何接受式依赖iid假设。
- 共享KV：draft写临时状态，target全精度验证覆盖；首拒绝校正、丢弃后续KV。
- 保分布依赖上述采样／KV契约，不因共用缓存自动成立。
- roofline按B／S及上步τ预测，≥1.05开启；同一rollout内开启后不再关闭。
- 一次开启依据active batch单调缩小，不直接迁移到补新请求的continuous batching。
- Γ={5,7,9,11}、P=2、上下阈值.94／.85，仅跨训练步调γ。
- 每model–hardware剖析校准；H100有害开启0–4%，不是绝对安全边界。
- 表3 decode构成与W4／W8理想比率含模拟，不冒充GPU算子实测。
- 论文链：veRL0.7.0＋vLLM0.11.2、Ray／FSDP、W4A16 Marlin。
- 主8×A100-80GB／TP1；Qwen2.5-7/14B用SimpleRL hard，Llama3.1-8B用medium。
- GRPO batch128×8响应、温度1／上限8192；计时排首步、随后100步均值。
- Qwen14B额外token rejection[.5,2]／frequency penalty .05，勿混作SD接受规则。
- 主rollout缩短24.2／17.1／11.2%，全步15.6／10.6／7.3%；全步含量化Prep 1.3–2.6秒。
- α由实测τ按iid公式反推、分固定γ段加权，95.1–98.1%不是直接接受计数。
- 表7累计消融走移除CPU气泡前路径，终点66.3秒不与主表62.4秒混比。
- Qwen7B三配对seed全步缩短14.0±1.5百分点（SD）；验证准确率相近不等同轨迹。
- 16K／温度.6的40.1／34.0%仅匹配50步附加配置，不与主设置混为统一峰值。
- EAGLE基线是同栈移植、未做异步重叠；历史Spec-RL有lenience，非完整系统公平排名。
- 分类L1/L2、A；[论文代码入口](https://github.com/furiosa-ai/EfficientRollout)未核commit／文件，留KV／logp接口待查，不排设备或M5专项。

### 6.18 Heddle T1压缩深读

- 状态：2026-10-27完成T1，非源码／性能复现验收。
- 原文：[v1 §3–8、算法1–2、表1–2](https://arxiv.org/html/2603.28101v1)。
- makespan代理为max(queue＋tokens·base_time·interference＋tool)，不是TTFT目标。
- control plane全局编排，data plane执行；轨迹ID串联多次LLM／tool请求。
- 历史(context,remaining_length)微调Qwen0.6B，每步结合新上下文预测剩余长度。
- 预测与tool并行；步骤1→2精度提升是实验趋势，非每条轨迹单调保证。
- PPS近似LPT，长轨迹优先；高优先请求抢占低优先active并保存prefix KV。
- 不丢弃被抢占轨迹，改变执行顺序不自动标C；不把LPT称任意agent图全局最优。
- 放置代理`min max(F(group)·max(length)·T)`，F来自剖析驱动模拟。
- 排序连续分组DP的最优性限已知长度／同构worker／F只随组大小单增。
- DP复杂度O(n²m)；合并短轨迹为提速启发式，不延伸精确最优保证。
- 在线更新rank、按剩余数量比例缩原分组，非每次重跑DP。
- tool间隙GPU-Direct RDMA迁prefix KV；长轨迹优先且迁移端点互斥。
- 长轨迹配高MP、短轨迹低MP；排序映射＋redistribute／split／merge退火是启发式。
- 论文链：veRL＋SGLang＋Ray，Rust router；PyTorch2.8／CUDA12.2／Ray2.49。
- 测试床8节点64 Hopper GPU（未列具体型号），NVLink900GB/s／IB400Gb/s。
- Qwen3-8/14/32B；CodeForces代码／HotpotQA搜索／DAPO-Math数学工具任务。
- GRPO每prompt16响应，输出上限40K，温度1／top-p .9；基线worker batch100、同global batch。
- 基线8/14/32B MP=1/1/2；Heddle异构；veRL／veRL*／Slime统一SGLang后端。
- rollout tokens/s对veRL1.4–2.3×、veRL*1.1–2.4×、Slime1.2–2.5×，非全训练或质量收益。
- 分项调度1.1–1.26×／放置1.2–1.5×／资源1.1–1.3×，不可相乘。
- 14B数学tool均值.046秒＜预测.27／迁移.25秒，不能普称完全遮蔽或零关键路径成本。
- 放置约36–38毫秒，资源搜索约5秒且周期执行；不把搜索每token计或当零成本。
- 分类L3/L4、B暂核；无reward／收敛对照，跨MP KV布局／版本与工具状态一致性待源码核，不开设备队列。

### 6.19 ObsSpec T1压缩深读

- 状态：2026-10-27完成T1，非源码／性能复现验收。
- 原文：[v1 §2–3、算法1、附录B–C](https://arxiv.org/html/2610.15246v1)。
- 问题：工具返回前生成下一动作，以重叠代替LLM与tool严格串行等待。
- 真实环境执行当前动作，同时world model预测observation；真实先完成则取消预测。
- 预测先完成即条件生成下一动作；observation完全匹配才保留，否则取消／重采。
- 动作必须核验后顺序执行，不在错误预测分支产生环境副作用。
- D限制向前动作深度，K限制候选observation广度；丢的是推测工作，不是训练轨迹。
- WM用真实observation tokens含结束标记做masked CE，policy用rollout reward更新。
- 实验先训policy500步、保存轨迹并同序离线重放训WM，非已测在线并行co-training全步时间。
- policy Qwen3-8/32B各8／16 GPU，WM Qwen3-0.6B另占8 GPU；型号未列，非同总算力比较。
- veRL编排／vLLM异步serving，policy TP2温度1、WM TP1温度0，默认D=K=1。
- Endless Terminals的bash／done；每批16 prompts×16响应，32轮上限，总响应16384／单轮2048 tokens。
- agent均值用4096rollouts／16批，排sandbox setup／verification；batch与async用65536且包含二者。
- 8B agent41.98→38.06秒，batch156.26→149.42秒；缩短9.3%与4.4%不可混。
- 32B agent缩短14.2%、batch7.8%、async吞吐增加11.5%，不是端到端训练或GPU成本收益。
- 47762次attempt：match66.45%、rollback30.50%、cancel3.05%；WM .105秒对tool1.685秒。
- match不等全部空输出；cat占40.9%且常复用已见文件，date几乎不可预测，依赖工作负载。
- D=1→8丢弃token198→472、decode145.5→139.5 tok/s，agent时延基本不变。
- K=8去重后仅3.14候选，match多2.6百分点却agent慢25.3%，相对K=1。
- 大WM rollback减少但更慢／cancel更多；训练到500步时更新WM较冻结100步者match高9.1百分点。
- TIR负项：55.82→56.42秒，tool仅占21.2%；更大WM也无实质收益。
- 表中误差为SE，reward相近不等同seed轨迹；条件lossless不认证时变环境／RNG／logp契约。
- 分类L4、A/B；exact-match与顺序提交门槛下不因rollback自动标C。
- [论文代码入口](https://github.com/kylemontgomery1/verl-recipe/tree/obsspec/obsspec)未核commit／文件；不排新算力或接续M5专项。

### 6.20 FP8-RL T1压缩深读

- 状态：2026-10-27完成T1，非源码／性能复现验收。
- 原文：[v2 PDF §2、附录A–C](https://arxiv.org/pdf/2601.18150v2)；整页渲染核第4、10–12页。
- 真FP8 policy产生rollout，不是量化draft加BF16 target验证；与前项SD路径分开。
- W8A8采用E4M3、权重128×128块静态量化，activation每forward动态量化。
- QKVO／MLP／expert量化，embedding／norm／lm_head排除；MoE router排除FP8可降低失配。
- 每RL步同步新训练权重及scale，rollout批内权重固定；同版本仍可有数值失配。
- veRL推理侧每步重置KV scale标记、首forward校准，非只初始化一次。
- NeMo-RL训练侧用新权重／数据重校准再同步，报校准占全步2–3%，不算免费。
- token IS比率`πtrain(a|s)/πFP8(a|s)`，训练端重算BF16 logp；TIS阈值C=2。
- 截断／逐token校正非完整轨迹重要性乘积，不推出exact BF16 rollout或无偏轨迹更新。
- 式3只写clip(w,C)，上下界／实现语义待核；MIS／router replay是建议，非已用实证。
- W8A8主8B用8 H100、30B-A3B用16 H100；FSDP＋vLLM、DAPO／AIME24验证。
- prompt32、n16、训练／mini-batch32、上限20K；正文却列rollout32×3×16，×3与温度口径未释。
- dense BF16无TIS对FP8有／无TIS，MoE两侧均用TIS；不混作同一控制条件。
- 线性rollout报dense10–20%、MoE30–50%增益，图为ms/token及吞吐，不当全步缩短。
- 8B KV-only报38%、组合44%；组合还含attention，不能加20+38或写仅两组件独立归因。
- 长20K／KV不足导致频繁preemption；容量与并发改善有条件，不泛化固定batch算子收益。
- E2E MoE改32 H100、prompt128、vLLM0.12／Megatron-Core0.15，非前述FSDP16卡实验。
- 此设置TIS-only的BF16训练＋FP8 rollout约700步崩溃；精度一致降低失配但不归零。
- hybrid E4M3前向／E5M2反向较稳，纯E4M3约500步后崩溃；FP32 scale优于该测试UE8M0。
- E2E此处未观察显著训练侧加速，主要收益为失配降低；不以理论GEMM代替全步实测。
- NeMo附录另报约48%rollout增益、同样2–3%校准成本，不能合并为主veRL44%同组数据。
- 论文配置入口`rollout.quantization=fp8`；DeepGEMM／CUDA12.9为文中环境建议，未核源码commit。
- 分类L1/L2/L4、执行A/B，数值policy与TIS另C；质量曲线非普适稳定证明，不开设备队列。

### 6.21 B6 T2系统阅读

本轮八篇按§6.0轻模板记录机制、关键结果和适用条件；均为论文阅读，不增加代码路径或复现计数。ROLL与ROLL Flash独立计数，GLM仅系统片段；Long-RL尚未完成。

#### 6.21.1 AsyncFlow

- **来源／状态**：[v2正文§3–7](https://arxiv.org/html/2507.01663v2)，T2完成；L3/L4、B，异步版本与训练目标契约另核。
- **机制**：TransferQueue把控制元数据与分片数据分开，按全局索引／字段／微批就绪送下游，解除整批屏障；不是每个样本一到就独立更新policy。
- **权重**：旧权重继续生成，新权重异步接收到host，当前生成迭代结束后才安装，延迟一步；更细的sub-step依次更新实例仍是未来工作。
- **设置／基线**：Qwen2.5-7B／32B、GRPO、DeepScaleR，32–512 Ascend NPU；MindSpeed＋vLLM-Ascend对比适配veRL d13434f的FSDP＋同推理后端，并非全部组件相同。
- **结果**：端到端吞吐平均10–20迭代，平均1.59×、峰值7B／256 NPU的2.03×；不与CUDA框架跨平台排名。
- **消融／质量**：7B／512 NPU另以关闭优化的串行同栈为1，TransferQueue为2.01、加异步为2.74，不当对veRL倍数或相乘；质量检查仅7B／16 NPU奖励与长度近似，不作全配置收敛证明。
- **迁移／缺口**：可借鉴流式数据面和延迟安装，但须计元数据、host缓冲与旧样本语义；未核源码提交、未运行NPU，不为本文自动新增M5实验队列。

#### 6.21.2 LlamaRL

- **来源／状态**：[v2 §5–8](https://arxiv.org/html/2505.24034v2)，T2完成；L4、B；AIPO算法及FP8数值变化与编排分账。
- **机制**：单controller向Executor派发，生成／训练独立GPU和并行度，支持同置offload；DDMA经节点内NVLink、节点间InfiniBand传分片权重，不经CPU中转。
- **概率契约**：AIPO传behavior概率，逐token截断importance ratio（ρ通常2–10）；截断有偏差／方差权衡，不是精确on-policy等价，也不等于普通PPO双边clip。
- **设置／基线**：Llama3.1 8B／70B／405B、MATH，512 prompts×4 responses；分别256／256／1024 H100。基线是共享kernel／FSDP优化的DeepSpeed-Chat式同步重实现，不是未经改动官方框架。
- **结果**：最佳step time为22.45→8.90s、82.32→20.67s、635.8→59.5s；各2.52×／3.98×／10.7×，是各规模组合最优配置，不证明同一单独机制的增益。
- **关键限制**：405B最佳配置含训练512卡MP16 BF16、生成512卡MP8 FP8；并行度、精度与异步共同变化，不能将10.7×全归异步或直接外推time-to-quality。
- **迁移／缺口**：应先拆weight-sync、精度和placement，再评估陈旧性；未检查源码／重跑，不增加实现验收计数或接续M5专项。

#### 6.21.3 RLinf

- **来源／状态**：[v2 §3、§5](https://arxiv.org/html/2509.15965v2)，T2完成；L3/L4、B；M2Flow将命令式工作流展开为时分context switching或空间elastic pipeline。
- **机制／屏障**：细分数据块可重叠microbatch前后向，但全局batch才更新权重；GRPO同prompt响应仍须聚合后做组内归一化，不能按到达顺序提前独立训练。
- **资源／计划**：device_lock配合onload／offload避免共享设备竞用；依赖须等父任务入队并释放锁。profiling搜索比较时分与分离流水成本，最优性限成本模型与搜索空间。
- **设置**：推理类GRPO用1.5B／7B／32B、64／128／256 H100-80GB，batch512、最大长度28672；AReaL-boba混合数据。veRL v0.5／Slime v0.1采用同SGLang＋Megatron及TP设置。
- **计量／正项**：10个warmup后迭代，吞吐按整步prompt＋response tokens计；时分GRPO比veRL快1.10–1.58×，不是仅生成token吞吐。
- **负项**：空间拆分7B GRPO吞吐比veRL低44.3–68.6%，长序列／小batch使流水不足；不能把embodied场景最高2.43×或不同PPO长度／batch的收益移作该配置证据。
- **迁移／缺口**：优先让profile检验共享vs拆分，而非预设拆分必优；本文不是样本选择方法的等价证明，未核运行时源码和profile开销，未用设备。

#### 6.21.4 MindSpeed RL

- **来源／状态**：[v1 §3–5](https://arxiv.org/html/2507.19017v1)，T2完成；L3/L4、B，Ascend平台边界保留。
- **数据面**：distributed transfer dock以元数据controller及分片warehouse组织全局batch索引，TensorDict经Ray传输，状态就绪触发后续sample flow。
- **权重**：allgather-swap借临时buffer收集／选取生成权重，原训练权重D2H腾显存，之后H2D恢复；不是零host传输或免费重分片，H2D可与推理重叠。
- **主设置**：比较实验是16 NPU、Qwen2.5 7B／32B及Qwen3 MoE30B、GRPO／DeepScaleR、256 prompts×16 responses、输入2K／输出8K；集群总384 NPU不等主测规模。
- **指标／结果**：TPS按prompt＋response除整步时间且再除设备数，平均5迭代；1.42–3.97×对veRL／OpenRLHF含kernel、并行与两项系统设计，不是单纯dock收益或整机输出TPS。
- **限制**：32B质量曲线另用2K／2K、1000步，不与8K性能设置混为同一实验证明；带宽估算表不当实测，扩大节点每节点固定64 prompts的线性度不是固定总工作强扩展。
- **迁移／缺口**：可借鉴数据面／内存生命周期，但必须重测host与通信成本；未核源码或访问NPU，不把硬件论文阅读自动升级为M5算力事项。

#### 6.21.5 NeMo-Aligner

- **来源／状态**：[v2 §3–4](https://arxiv.org/html/2405.01481v2)，T2完成；L1/L2/L4、A/B，不与后续NeMo RL混作同一实现年代。
- **多模型**：actor／reference同组、critic／reward同组，切换时CPU offload；PyTriton异步RPC重叠critic训练等工作，不据异步通信一词断言actor rollout跨版本。
- **生成**：首次构建TensorRT-LLM engine后常驻，每步原地Refitter；内存允许时推理改TP-only，去掉PP通信，动态DP worker池均衡生成。
- **设置**：Llama3-70B actor＋70B critic、PPO／HelpSteer2、batch128、BF16，8 H100-80GB-SXM／节点；actor＋critic从8＋4节点扩到16＋8节点，即96→192 GPU。
- **结果**：整步53.7→29.8s（1.80×）、生成35.8→17.5s（2.05×），是加倍硬件扩展；均值5步并排除首次build，不能称固定GPU系统优化1.80×。
- **消融**：8＋4节点基准53.7s，分别去TRT／reshard／refit／异步请求为372.2／207.6／169／82.8s；各自移除一项，不是可相加或相乘的优化链。
- **迁移／缺口**：常驻engine＋refit必须同时计峰值显存、权重转换与冷启动；未核源码／测量完整训练质量，不增加代码路径或复现计数。

#### 6.21.6 ROLL

- **来源／状态**：[v1 §3–5](https://arxiv.org/html/2506.06122v1)，T2完成；L3/L4、B，数据筛选／算法recipe单列，不与Part II重复计数。
- **执行面**：单controller、Ray ParallelWorker／Role／PlacementGroup，训练接Megatron／DeepSpeed、生成接vLLM／SGLang；资源池与AutoDeviceMapping映射worker到设备。
- **流水／权重**：逐sample reward、环境及工具异步，使等待者不堵其他任务；ModelUpdateGroup经NCCL同步权重。存在add／abort接口不代表默认丢样本，也不直接证明rollout与训练off-policy。
- **实验**：Qwen2.5-7B Base／Qwen3-30B-A3B Base多域训练，数学40%／代码30%／通用30%；采用PPO loss与REINFORCE returns而非GAE，recipe不能作为纯编排消融。
- **结果口径**：平均准确率0.18→0.52、0.27→0.62是学习质量，不是2.89×／2.30×吞吐；Sokoban等agent成功率曲线同理。
- **限制／迁移**：本版本§5未给可比固定硬件系统计时／加速对照；千卡、200B、两周部署叙述不是受控benchmark。可迁移worker抽象／生命周期，不能据其推导跨框架性能排名。
- **未验证**：源码接口与执行语义尚未对照；论文训练效果不认证每步样本集合不变，未运行实现或新增M5交接项。

#### 6.21.7 ROLL Flash

- **来源／状态**：[v1 §4–5、附录A](https://arxiv.org/html/2510.11345v1)，T2完成；L3/L4、B，冗余／过滤另C；不把Part I的质量结果移作本篇速度证据。
- **机制**：LLMProxy、EnvManager、SampleBuffer及AsyncController串接；换权重经历suspend→model_update→resume，训练与下一批收集重叠，环境等待可让其他轨迹继续生成。
- **版本／概率**：α限制从启动版本计的每样本年龄，buffer≤(1＋α)×batch；中断后可新权重续跑，故单响应可混版本。附录temperature=1、top-p=1，另用截断IS（如cap5）处理训推引擎差异，不是无偏等价证明。
- **调度vs选样**：prompt replication将n候选展开独立任务，不自动增加最终训练量；冗余env超过目标batch、先够数即停及动态零方差组过滤则改变收集集合，须标C。
- **设置边界**：RLVR用DAPO-Math-18K，SGLang0.4.6／vLLM0.8.4＋Megatron；agent用R2E-Gym-Lite／ALFWorld等。真实环境对照的具体GPU／模型配置本轮未落实，不补猜或跨配置排名。
- **真实vs模拟**：保持Sync训练仅加env异步，SWE 10.22→8.32h、ALFWorld 13.37→8.44h；冗余env的5.45×来自高斯延迟模拟，不是真实环境倍数。
- **限制／迁移**：真实env冗余模式又改变并发组数／候选数，不能当固定训练集合纯调度；保留年龄起点、resume版本及终止集合，未核源码／独立质量复现。

#### 6.21.8 GLM 4.5 Slime系统片段

- **来源／状态**：[报告v1 §3.5](https://arxiv.org/html/2508.06471v1#S3.SS5)，仅本系统片段T2完成，**不称整份GLM-4.5报告已深读**；L1/L3/L4、A/B，数值与任务过滤另C。
- **结构**：Megatron训练、SGLang＋Router生成、Data Buffer解耦数据交换，Ray管理资源；不是只更换推理kernel的单层优化。
- **两种模式**：一般数学／代码采用同置同步；agent采用分离异步、生成与训练专用GPU、周期权重同步。应按workload选择，而非宣称全报告使用全异步。
- **数值契约**：BF16训练、FP8推理，每次更新在线block-wise量化；是真实生成policy数值路径，不是仅量化draft，不能自动认定与BF16轨迹精确等价，§3.5未充分说明校正。
- **环境／集合**：Docker隔离执行、HTTP统一endpoint与消息列表任务池减少集成成本；任务过滤／动态采样与执行提速分开，不能因基础设施支持即当固定集合B。
- **证据缺口**：§3.5未给独立系统硬件／计时消融，不拿AIME等模型质量当Slime系统加速；未对照源码提交、未运行，不增加代码或完整复现计数，也未派发M5实验。

### 6.22 Long-RL与权重同步及失配

本轮以B6余Long-RL＋B7前七篇组成八篇T2；只读论文的机制、结果与必要附录，源码路径和完整复现计数不增。权重bit-exact、概率一致、数学更新等价与统计质量分别判定。

#### 6.22.1 Long-RL

- **来源／分类**：[v4 §3–6](https://arxiv.org/html/2507.07966v4)，T2完成；L1/L2/L4、MR-SP为A/B，数据构建／筛选另C。
- **编码复用**：视频帧分到多GPU的vision tower，all-gather embeddings后供多个rollout复用；省去重复视觉编码，不等于复用完整decoder KV或跨policy版本无条件缓存。
- **prefill**：复用输入embedding，padding后沿序列分片，policy与reference均采用SP处理长输入；长视频输入显存与prefill才是主要对象。
- **设置／计量**：8 A100-80GB、SP4、LongVILA-R1-7B，batch每卡1、rollout数5；10次warmup后平均5次，§6.2明确测每步forward时间，非完整GRPO墙钟。
- **结果／限制**：512帧最高2.1×，完整两阶段可到1024帧；仅Stage1超过512帧OOM。摘要8192帧模型能力／3600帧可运行不当本次2.1×配置。
- **集合／迁移**：§3用10次预测过滤全对／全错题，属于数据选择；主GRPO组数8与性能测试rollout5不同。embedding复用须按encoder状态／输入保持有效，未核源码或进行设备复现。

#### 6.22.2 SparseRL-Sync

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2605.07330v1)，T2完成；L4、B，压缩post-cast权重而非稀疏梯度或删训练样本。
- **机制**：比较cast-and-copy前后BF16模型参数，多步changed-index取并集，传索引＋当前绝对新值，接收侧覆盖而非浮点加delta；仍构造完整NaN-masked tensor做layout转换。
- **代价**：BF16值2B＋int32索引4B时，密度ρ的原始压缩比约1/(3ρ)，不能由99%稀疏直接推出100×；索引差分／值编码后的60–101×与原始32–54×分开。
- **设置**：Helix＋Megatron／SGLang，106B实测128 H100、64训＋64推；30B正确性500步权重逐tensor相等，奖励MAE仍0.0186，不称采样轨迹逐条一致。
- **关键结果**：106B IB-off全量广播45.6→2.26s；671B延迟由106B有效带宽外推，§4.3只用原始索引／值，附加无损编码收益留待测量，不当全训练100×。
- **边界**：bit-exact声称针对同目标权重重建，不保证异构kernel概率一致；一般化还需核cast／loader与NaN、signed-zero等比较语义及临时显存，未检查实现或运行。

#### 6.22.3 PULSE

- **来源／分类**：[v2 §4–5、附录E–F](https://arxiv.org/html/2602.03839v2)，T2完成；L4。Sync为B，LoCo优化行为另列，不能统称无损同一通道。
- **Sync**：按BF16 bitwise变化发送索引／绝对新值，解码覆盖不做差值相加；未可见更新仍留FP32 master累积，精确性相对于当前BF16视图，不是FP32更新为零。
- **LoCo**：trainer间交换DiLoCo式FP32 pseudo-gradient，BF16门未通过项留error-feedback；外层动量在同步后应用。不是逐步DDP梯度，也无每步与DiLoCo bit-identical保证。
- **Sync部署**：Qwen2.5-7B／Coder7B，MATH／MBPP各3 seeds；trainer单B200、匿名推理节点、约400Mb/s。14GB→平均108MB（约130×）是上传字节比，低LR1e-6与codec基准3e-6的79×不同。
- **LoCo评估**：4 workers，7B用B300、小模型A100，Qwen H8／Llama H4；同周期raw payload比DiLoCo小12.8×，编码7B末轮>17×，相对DDP再含减少同步次数H，不能互换分母。
- **负项／边界**：LoCo早期学习可落后，最后在seed方差内接近；大H受陈旧性限制。grail另有跨窗replay偏新采样／过旧淘汰，不能把该训练集合变化归于Sync压缩或当固定集合质量对照；未核源码／复现。

#### 6.22.4 NeMo-DCR

- **来源／分类**：[v1 §3–8](https://arxiv.org/html/2610.08430v1)，T2完成；L4、B，bit-exact目标是与dense refit相同的参数／buffer，**不含KV请求状态**。
- **放置**：分片owner投影到canonical坐标，native loader决定接收布局；只对支持的affine变换直接投影，其余构造canonical residual。source baseline占host约1.03–1.04倍checkpoint，不是无基线内存。
- **版本恢复**：保bit且不重叠的路径用XOR，其余overwrite；XOR重复应用会撤销更新，必须去重，失败重试改overwrite；所有required receivers ACK后joint commit才推进policy及下轮baseline。
- **设置／基线**：延迟测试32 GB300训＋64 H100推跨AWS region、每节点跨集群流至5Gbps；3%／5%人工changed元素压测，5次warmup＋10次计时。对照是全checkpoint对象存储传输，非NCCL快路径。
- **结果／窗口**：120B refit22.6–49.7s，对传输参考750s快15.1–33.2×；包含构建／loader／commit，但排初次加载、prewarm和post-apply checks，不当全RL步收益。
- **规模／真实性**：1T通过550B模型层数翻倍做refit测量，非1T训练收敛实验；真实Qwen3-30B训练另为50 GRPO步、5次运行及周期kill，与synthetic压测分开。
- **迁移条件**：loader须逐元素确定、全部storage写可拦截；不支持路径提前拒绝。KV每次转换失效，异步先继续旧版本服务、请求排空后应用，不把bit-exact refit当mixed-policy采样证明；未核实现。

#### 6.22.5 ThunderSyncRL

- **来源／分类**：[v1 §3–4、附录D/G](https://arxiv.org/html/2610.05935v1)，T2完成；L4、B，整批固定policy、一次optimizer更新，不是边到边更新。
- **GRPO拆分**：单轨迹reward就绪即算score梯度Hᵢ，分别累计ΣrᵢHᵢ和ΣHᵢ；整组收齐再组合均值／标准差，提前的是反向，不是使用不完整组advantage。
- **OPD／屏障**：一turn被冻结teacher评分后释放反向；batch闭合后按总scored tokens归一，完成延迟梯度／reduce／clip才step，发布新权重并清policy相关状态、ACK后开下一批。
- **设置**：Qwen-3.8-27B，单节点8 B200（另H200）；4卡learner，GRPO余4卡rollout，OPD余2推＋2 teacher。每步32 prompts×8／4 trajectories，最长300 turns×1024生成tokens；SWE-smith／SETA-Env。
- **收益**：同配置Sync基线下B200 Qwen整步1.91×／1.94×；SWE pass@3到74.01%的GRPO训练成本233.5→121.3 GPU-hours。不同GPU／GLM收益较低，不以单一1.9×推广所有比例。
- **成本**：learner峰值显存GRPO117→122.8GB，另有pinned host FP32梯度／组统计，不在GPU峰值中；解析平衡曲线无实测点，不能作集群扩展结果。
- **等价边界**：固定batch重放10步，FP32梯度相对误差约1e-6量级、BF16有数值差；数学等价不称bit-exact，也不推广任意多epoch／非线性目标。未核运行实现，未做设备验证。

#### 6.22.6 PipelineRL

- **来源／分类**：[v2 §4–5](https://arxiv.org/html/2509.19128v2)，T2完成；L2/L4，编排B，途中换权重／旧KV改变behavior另C。
- **机制**：vLLM actor、reference preprocessor与DeepSpeed trainer经Redis串流；每次optimizer后暂停很短即换生成权重，再续在途序列，保留历史KV，形成序列内mixed-policy。
- **设置**：Qwen2.5-7B Base、OpenReasoner Zero17K题，1000 optimizer步、batch1024、LR1e-6；16 DGX-H100共128卡，48生成（H64）＋80训练，importance-weighted REINFORCE cap5。
- **基线**：Conventional RL由相同流水缓冲BG再执行G步来模拟；吞吐在较小卡数测后修正到128卡，并非128卡同机运行完整veRL对照；文首4节点叙述不替代§5的16节点主设置。
- **结果**：对G32约2×样本／同奖励学习速度；G64发散。另一batch4096的MATH50084.6／AIME19.8不与batch1024主速度试验混作同一配置。
- **概率边界**：ESS近G8不等零陈旧性；max lag可>50K samples。保旧KV影响另在gmax32／长度2048、checkpoint0/100/190实验中较小，不证明任意上下文或新policy精确采样。
- **迁移**：应同时记录token版本、KV来源、样本有效性和基线估算方法；不能由标题on-policy排除C或把模型质量相关当不变分布证明，未核源码或运行。

#### 6.22.7 Missing Old Logits

- **来源／分类**：[v3 §3–6、附录G/H](https://arxiv.org/html/2605.12070v3)，T2完成；L4系统成本／算法边界，token masking的有效集合变化另C。
- **语义**：当前训练概率／历史推理概率分解为同版本训推差异与训练版本变化；中间历史训练概率缺失时，插值分母虽代数抵消，clip／mask的独立语义不恢复。
- **精确路线**：历史快照切换、独立old-logit模型、换权重前中断partial并算历史前向；分别消耗host／I/O、额外模型资源、同步与资源切换，不能只比比率公式。
- **近似**：PPO-EWMA按staleness窗口设衰减并在有效token比例低时reset；是proxy不是旧logits精确恢复，β较大甚至β0.75无reset仍可晚期mask塌陷。
- **实验**：ROLL、Qwen3-4B／30B-A3B、版本gap≤3；τ²-Bench／VitaBench。30B in-store avg@2从18.28到33.41是成功率百分点变化，不是吞吐倍数；精确参考34.62仍较高，EWMA并非普遍超越。
- **系统成本／限制**：附录G快照4B／30B各77／11步窗口，stage均值95.20／178.22s、驻留快照最多40.22／76.43GB；均最终Ray actor不可用而终止，不认定快照是唯一原因，也不当最终训练结论。
- **迁移**：精确old logits仍不消除阈值对active-token集合的耦合；完整硬件／batch长度未在所读实验附录落实，不作跨系统吞吐排名，未核实现或新增M5算力事项。

#### 6.22.8 Beyond Precision

- **来源／分类**：[v1 §3–4、附录C](https://arxiv.org/html/2602.01826v1)，T2完成；L1/L4优化诊断边界，不是系统吞吐优化，数据全对／全错组拒收另C。
- **机制**：用响应长度激增时点指导衰减周期，周期性LR减半直到初始值10%；1.8×激增时点是经验启发式。算法1本身只有周期减半，未写在线激增阈值检测器，不能标为完整自动触发实现。
- **设置**：Qwen3-4B／8B Base、过滤DAPO约13K题，veRL FSDP＋vLLM、16 H100；batch64、每prompt16响应、RLOO、PPO epoch1、最大响应8192、LR1e-6。
- **结果／负项**：4B周期125／204较稳、250仍塌陷；8B激增约90步、160优于epoch对齐204；不能将周期或1.8系数定为通用常数。
- **数值边界**：FP16配置仍有塌陷、LR衰减改善稳定性，说明特定配置下仅换精度不足；不证明所有数值误差都已消除，亦不否定其他精度工作在其条件下的结果。
- **迁移／限制**：长度、梯度噪声与失配共变是诊断线索；O(T²)是带有界性假设的梯度误差上界，不是实际误差必二次增长或新系统加速定律。未核代码或进行复现。

### 6.23 精度失配与长尾调度

本批完成B7余两篇与B8前六篇，共8篇T2；仅机制、关键结果与条件核对，不增加源码／复现计数。Rethinking未选入B7三选二，仍为边界未读条目。

#### 6.23.1 FP16

- **来源／分类**：[v1 §3–5、附录](https://arxiv.org/html/2510.26788v1)，T2完成；L1/L4数值诊断，不硬归调度A/B；数据筛选另列。
- **机制**：训练和推理同时改FP16，以更多尾数位降低训推差异；指数范围更窄，须loss scaling及跨rank溢出／scaler同步，不是只改rollout dtype。
- **设置**：R1-Distill-Qwen-1.5B、筛后MATH 1460题，8 A100 80GB；batch64题×8响应、最长8000、每policy迭代4次梯度更新。veRL／Oat含作者修整，非原封不动框架对照。
- **结果**：该配置BF16 GRPO先升后塌陷，TIS仍可失败；序列MIS较稳但慢。作者最好FP16配置训练准确率99%、AIME39%，序列MIS为95%／34%；不同recipe不当单一dtype净收益。
- **成本对照**：BF16训练＋FP32推理也可稳定，但推理约慢3倍；FP16提供另一精度／速度折衷，未据此建立全训练吞吐排名。
- **负项／迁移**：FP16仍有小失配及框架特定数值尖峰，离线AIME 8K也非总优于BF16；“消除失配”不作bit-exact证明或通用精度建议。与Beyond Precision配置不同，不据一篇否定另一篇；未核实现。

#### 6.23.2 VeXact

- **来源／分类**：[v2 §3–5、附录实验设置](https://arxiv.org/html/2605.14220v2)，T2完成；L1/L4诊断参考，rejection sampling改变有效训练集合另C。
- **机制**：FSDP内复用HF模型与kernel，固定tiling／reduction实现batch-invariant运算，处理RMSNorm／BMM／MoE并禁用attention KV split；确定性与吞吐有取舍，不是只对齐权重。
- **隔离实验**：REINFORCE单批单更新减少陈旧性与PPO clipping混淆；Qwen3-1.7B用16 H100、batch64、最长响应8192，30B-A3B用64 H100、batch512、最长20480；另有8卡GRPO实验。
- **结果**：MoE vLLM对照约280步后reward／验证从.574／.293跌至.255／.067，VeXact随后到.753／.534；是文中受控TIM诊断，不推广所有引擎、数据和配方。
- **指标反例**：GRPO曲线恶化时聚合KL估计可长期平坦；bypass与recompute的梯度表现也不同，不能只凭平均KL判定训推一致。
- **修正／限制**：TIS＋序列RS较接近精确参考；失配修正比率与PPO比率分开，非线性clip／筛选下不宣称严格目标等价。主证据为数学任务，未核实现或认证可直接替换生产引擎。

#### 6.23.3 TailSieve

- **来源／分类**：[v2 §3–4、附录A/E](https://arxiv.org/html/2608.22788v2)，T2完成；L1/L3/L4，A/B及**C（跨轮批集合／顺序）**，修订此前暂A/B标签。
- **路由**：partial只识别尾部prompt组，不继承旧token／KV／logp；当前policy重新生成整组。内环调尾组数以平衡完成时间，外环调整副本容量，局部候选搜索不当全局精确优化。
- **集合边界**：一批由上一轮保留尾组加本轮最早完成bulk组组成；当前权重生成不等固定同一批轨迹。无额外长度偏差结论限稳态，warmup及尾组配额变化有过渡影响。
- **设置**：单节点8 H100，大小模型分别4×TP2／8×TP1；DeepScaleR、KodCode-Light10K，64组×8响应，最长16384、温度.7、thinking关闭；稳态测试排除controller warmup。
- **结果**：routing-only最高1.67×，叠MTP／DFlash最高2.59×。30步2B实验rollout均值119.34→82.49s，加MTP为46.61s；这部分包含controller、不含optimizer，不叫整轮RL提速。
- **负项／迁移**：默认不启用suffix draft，原因含接受率有限及ragged CUDA Graph成本；低并发尾池才是推测收益条件。未核作者实现，不能据整组／on-policy自述消去每步集合变化。

#### 6.23.4 WAR

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2607.17299v1)，T2完成；L1/L3 A/B，多轮执行的sandbox超时边界另核，不按workload-aware名称推断筛样本。
- **机制**：低负载用历史序列SuffixDecoding并由target验证；高负载按sandbox TTL紧迫性、短turn估计、cache命中和在途数排序，低队列负载回退least-request／sticky。历史draft不是旧policy轨迹回放。
- **控制**：AIMD依据cache命中调请求窗口，推测按本地batch负载开关；TTL优先调度是避免过期，不能承诺执行中绝无超时终止。
- **设置**：veRL0.8／SGLang0.5.5／Megatron0.13.1，Qwen3-32B＋YaRN 96K、SWE 500题；两节点16 H100，batch4–64题×8响应，观察5个训练步。
- **结果条件**：仅统计prefill／decode／turn工作量与基线相差5%以内的运行；这是评估筛选，不是训练过滤机制，也不证明所有运行都生成同一轨迹集。组合prefill缩短35–63%、decode缩短32–63%，非完整训练时间。
- **负项／迁移**：调度单独使用时batch4的decode约退化4%，batch64才改善约47%；短窗口不足以证明长期收敛或任意负载受益。未核源码与TTL失败消费语义。

#### 6.23.5 Seer

- **来源／分类**：[v3 §3–5、表1–3](https://arxiv.org/html/2511.14617v3)，T2完成；L1/L2/L3 A/B，同步colocated rollout，不把异步索引更新误作异步RL。
- **执行**：按response chunk调度，Mooncake共享KV池使用DRAM／SSD与RDMA；组内probe先完成一条响应，再用已完成最大长度作近似最长优先，并加防饥饿控制。不是免费迁移或精确长度oracle。
- **推测**：组内响应维护压缩suffix tree，target验证draft；异步更新CST，按batch与接受率profile调draft长度，不额外依赖GPU drafter模型。
- **设置**：Moonlight／Qwen2-VL-72B／Kimi-K2分别32／128／256 H800，group8／16／8、最大长度65536／40960／98304；节点配2TB DRAM和4TB NVMe。基线统一in-house vLLM，StreamRL-Oracle只作该栈长度oracle对照。
- **结果口径**：5次rollout迭代的输出tokens/s较veRL提升44–104%；尾部时间减少72–94%指**最后10%请求处理时段**，不是P95延迟或整训练收益。
- **消融／限制**：divided rollout→调度→推测为累计叠加，不能把各列当独立倍率相乘；同版本KV与target验证是迁移条件，未核实现／随机数契约或长期学习质量。

#### 6.23.6 Infinite Sampling

- **来源／分类**：[v1 §3–5、算法2／表1–2](https://arxiv.org/html/2506.22950v1)，T2完成；L2/L3/L4，固定slot路径B；dynamic-slot工作量变化单列，非无条件同任务对照。
- **机制**：复用prompt KV、回收响应slot，micro group连续交错；实际生成prefix供长度预测并继续使用。静态分组配合SJF refill，全G条reward齐后算组基线再分微批反向，不按micro group各自更新。
- **目标／算法疑点**：§3.4只写减组均值，与前文带std的GRPO式不同；算法2称FPTAS，但所列舍入＋first-fit流程未见放不下的处理或近似保证证明，不据命名认证可行／全局最优，亦不推断未核实现必有bug。
- **设置**：Qwen3-1.7B／8B、A100 80GB，GSM8K／MATH／KK；B=1、G=32、默认g=4、最大1024、温度.8，GPU数未落实。
- **结果口径**：1.7B显存21.55→10.64GB对应g1，而非默认g4；GSM8K固定slot decode rounds 3250→1770、均长186不变。round数不是墙钟秒或整RL步。
- **负项／迁移**：8B dynamic-slot虽只360 rounds，平均长度变21而对照186，不能全算同工作量加速；Infinite*是事后oracle下界。未核实现及长期收敛，不宣称原GRPO目标普遍不变。

#### 6.23.7 CadenceRL

- **来源／分类**：[v1 §4–6、附录](https://arxiv.org/html/2609.36899v1)，T2完成；L2/L3/L4 B，跨步消费／年龄仍另核，不因pacing替换resident就归删样本。
- **机制**：长context暂存pending、短请求先补，准入耗尽后按亲和性聚尾；pending只在原policy版本恢复，排空才释放该源worker。准入credit约束超前batch供给，**不是逐轨迹版本年龄硬界**。
- **状态迁移**：late-bound KV准备经host／relay分散传输；只在兼容集群复用，跨硬件或集群回退token re-prefill，不称任意异构KV无缝复用。
- **设置**：Steptron训练H800、vLLM rollout用A910X／BI-V150／H800；Qwen3-8／14／32B，最大64K／32K／16K，512题×16响应，PPO单次mini更新。trainer刻意保留余量，系统以rollout为瓶颈。
- **结果口径**：异构配置最高TPS改善48%、P95下降64%是文中各项上界，不擅自配为同点；P95从首次推理到完成、含pending，TPS与tokens/$不等整训练time-to-quality。
- **负项／限制**：同构14B／32B吞吐可低于Laminar／AReaL；Sync／Oneoff延迟仍更低。credit过大吞吐回落、延迟上升，concentration有小额吞吐开销；质量仅8B、30迭代窗口，未作长期或实现验收。

#### 6.23.8 Weave

- **来源／分类**：[OSDI'26论文§4–6及附录](https://www.usenix.org/system/files/osdi26-wu-tianyuan.pdf)，T2完成；L3/L4 B，既有会议身份，不新增arXiv条目。
- **机制**：跨job co-execution group交错rollout／train气泡；层间按边际成本、SLO与host驻留容量放置，组内round-robin meta-iteration各执行一次rollout与train。模型状态留host支持warm切换，非所有模型同时常驻GPU。
- **最优性边界**：打印页815定理只对unsaturated／full group及其负载假设证明组内利用率最优；全局放置是NP-hard启发式。尾迁移限同一H20池，保留模型／KV／RNG及batch组成，不跨异构池搬KV。
- **设置**：训练池328 H800、rollout池328 H20，池内400Gbps IB／跨池20Gbps以太网；价格按H800为H20的2.85倍。微基准7／14／32B，在相同时租成本下对Solo-D吞吐1.82–1.99×，不是单job延迟普遍改善。
- **测试集群**：200个job、两周trace回放，单job SLO为相对Solo的1–2倍slowdown；本次100%达标、总成本$188.8K，较Solo-D／veRL分别低1.84×／1.38×。是该回放与计价条件，不是通用账单保证。
- **负项／模拟**：共调度干扰仍使相对Solo慢1.6–9.8%；打印页820图14为testbed、图15另为300-job／580小时合成RL特征模拟，1.01–1.12×最优成本只属后者。PDF逐页视觉复核，未核实现或新增算力任务。

### 6.24 采样预算与推测执行

本批完成B8余六篇T2，B8收束12/12；采样预算、准入顺序与draft来源分别分类，阅读不计源码或性能复现。

#### 6.24.1 CERO

- **来源／分类**：[v1 §2–5、附录C](https://arxiv.org/html/2606.05606v1)，T2完成；workload／算法边界，**C（跨epoch采样分配）**，不是固定轨迹集合的执行调度。
- **机制**：Beta posterior的`E[p(1-p)]`估计prompt信息价值，不是posterior variance；凹饱和效用配合prompt／全局预算双变量，超过预算价格才给Nmax条，否则跳过，末批受剩余预算截断。
- **目标边界**：保留GRPO loss形式不等保留期望更新；prompt采样频率与组大小已改变。O(√K) regret限固定utility、紧投影域与有界梯度，不能认证policy更新及posterior刷新的非平稳实现同样成立。
- **设置**：veRL、DAPO-Math-17K，R1-Distill-1.5B／Qwen3-4B Base与Instruct／Qwen2.5-Math-7B；前两尺度4 H800、7B用8卡，CERO训练500更新，平均预算8、上限16响应。未给有效batch与长度，不据GPU-hours作配对性能排名。
- **结果／反例**：表1平均准确率较GRPO增4.84／6.04／3.28／1.53个百分点；但7B AIME26从.1250降至.1187，与正文“所有model–benchmark pair均提升”不符。KnapsackRL栏空白，不作已测基线。
- **迁移**：每次分配更新少于1秒是局部成本，不是训练提速；预算按响应条数，不保证等token／GPU时间。数学任务的信息代理与固定效用保证不外推通用reward，未核实现。

#### 6.24.2 Scheduling Mixed RL Rollouts

- **来源／分类**：[MISA-T v1 §3–6](https://arxiv.org/html/2608.11152v1)，T2完成；L3 B。HOLD保留需求账，不改任务标签；有限步消费混合仍可能变化，不作每批分布精确不变证明。
- **机制**：active＋pending共同占session cap，防快照滞后超额准入；已有continuation优先，held／candidate计需求但未准入不占cap。类预算按未完成session数×KV footprint×驻留时间分配，再交cache-aware placement选副本。
- **契约**：驻留时间含工具间隔，非纯GPU算时；完成混合趋近trainer目标依赖各类backlog相对总到达量趋零。算法近似该目标，不直接证明队列稳定或保证固定类吞吐。
- **设置**：50迭代Step3.7训练用6 H200节点＋4 rollout节点、rollout每节点TP8；固定checkpoint消融另用Step3.7两节点或Qwen3.6-35B-A3B的16 H100／4×TP4，128K context，基线扫静态并发后选优。
- **结果口径**：完整训练平均迭代时间降22.8%、完成样本率增35.6%；rollout-only增53.3%／43.6%不能挪作全步。消费混合TV偏差4.14→2.71个百分点、非零；50迭代pass@4差小于.5个百分点。
- **负项／迁移**：高prefix hit不必高吞吐，容量安全的保守基线仍可慢；effective prefill TPS包含cache命中token。依赖及时session快照和类标签，逻辑保护预算不替代物理KV管理，未核实现或长期收敛。

#### 6.24.3 SRT

- **来源／分类**：[v1 §3–4、附录A/B](https://arxiv.org/html/2601.09083v1)，T2完成；L1/L3 A/B，历史及run-ahead token只供draft，不直接作为actor学习目标，不能按历史复用自动归C。
- **机制**：每prompt树缓存续写子串及频次，匹配当前最长suffix后按路径频率扩展候选树，由当前policy并行验证；未命中回退AR。分布保证依赖正确的推测采样／残差处理，非仅按greedy首个匹配判断即可证明随机采样等价。
- **维护边界**：当前rollout可在线入树；尾部空闲worker为active或即将到来的prompt run-ahead，生成物入树后丢弃。主表计时**未启用run-ahead**，在线更新／run-ahead的拆分收益在§4.2为模拟，不混作额外实测训练收益。
- **设置**：Qwen2.5-1.5B、veRL＋vLLM、8张Hopper（具体型号未列）；300步，GRPO batch128×5响应、最长4096，DAPO batch128×16、8192，ReTool最长16384／8 turns。
- **结果**：GRPO生成31.8→15.4s、整步42.9→26.2s；最高约2.08×指rollout，不等完整训练同倍率。DAPO中SuffixDecoding生成44.1→62.5s反而变慢，SRT为31.5s，不能泛化所有历史draft均受益。
- **迁移／限制**：树频次不是当前policy概率，依赖重复prompt及target验证；GPU空闲是否足够支付run-ahead成本尚不能由接受长度模拟认证。主表为效率结果，不据此声称长期收敛或数值实现已验收。

#### 6.24.4 S3R

- **来源／分类**：[v1 §3–6、附录B/C/F](https://arxiv.org/html/2610.13202v1)，T2完成；L1/L2/L4 A/B，稀疏仅作draft；直接稀疏采样＋过滤的对照另C，不与本方法混记。
- **精确条件**：同一冻结policy共享权重，稀疏draft记录完整q，由dense p逐token接受／残差校正；全组独立随机数及基线采样规则相同，才恢复dense组联合分布。只校正每条自身似然通常不足以恢复GRPO的组内reward比较，不等所有group-level IS都不可能。
- **KV／调度**：历史dense KV完整保留，验证全层重算并覆盖临时draft KV；按KV压力及恢复backlog调并发、先恢复paused请求、不丢组。故稀疏读取不等同减掉95%的KV存储，跨步释放／recapture也有成本。
- **设置**：veRL0.7／基线vLLM0.12、单节点8 H100、4×TP2；Qwen3-1.7B／8B，DAPO-Math-17K，256题×8响应、温度1、最大16K／24K，k6、稀疏预算5%，主比较关闭样本／组过滤。
- **结果口径**：相对AR，训练吞吐1.39×／1.35×，rollout时间降38.6%／33.2%；吞吐按完整step的非padding prompt＋response tokens／GPU计，排首步warmup，任务分数取最佳checkpoint，不是固定质量达标时间。
- **负项／限制**：低KV体量的串行stage模型中，B16／S4K即使全接受也不足摊薄成本；prompt n-gram基线在1.7B训练吞吐降至.72×。主收益是完整后端组合，非单一稀疏kernel或跨步生命周期的独立消融；未核实现。

#### 6.24.5 NgramRL

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2610.32881v1)，T2完成；L1/L2 A/B，host n-gram与CPU–GPU预取协同，不因缓存旧token自动归回放训练C。
- **机制**：每prompt维护近期续写LUT，最长n-gram匹配供draft；当前target按p/q接受及正残差重采。有限历史限制host增长，但不消除policy漂移；分布相同不等逐run随机轨迹、梯度或训练结果逐位相同。
- **预取**：预测下一轮`(接受长度j, target续写token b)`，按近期因果统计分配fan-out；真实b不删、不替换，覆盖失败回退普通lookup。成立的是提交语义回退，非计入全部预取开销后墙钟永不退化的保证。
- **设置**：两A100 80GB做GRPO，Qwen3-8B／DeepSeek-7B、64题×5响应、15epoch、k16，GSM8K／Spider／CodeContests；运行时profile另为8B／batch64在RTX PRO6000 96GB及32核Threadripper上，不混用训练机与计时机。
- **结果口径**：主表TPS是**每target forward提交token数**，非tokens/s；Qwen8B Spider全程5.94、最后30%为7.77，warm-cache区间不作全程均值。§4.1.5按CUPTI时段与coverage代入式3报告预取1.38×，相对同步NgramRL，不是相对AR的完整训练提速。
- **负项／迁移**：首epoch冷启动、CodeContests早期全局cache可领先；更多旧历史可能降接受率，host内存不为零。大fan-out本身须藏进GPU时段，高熵任务收益只约1.05–1.06×；未核实现或用该结果建立跨论文排名。

#### 6.24.6 TLT

- **来源／分类**：[v3 §4–6、表3/5](https://arxiv.org/html/2511.16665v3)，T2完成；L1/L2/L3/L4 A/B，drafter训练与rollout协同，不把异步drafter checkpoint误作actor异步更新。
- **机制**：完成rollout的worker释放内存后执行可抢占Spot Trainer；host DataBuffer缓存hidden states，当前partial＋上一步长响应只用于训draft。单decoder层共享冻结embedding／head，target仍按原RL屏障更新；旧draft训练数据不等actor重放。
- **适配**：余请求低于阈值（默认32）启用SD；batch分桶的epsilon-greedy按提交token／延迟选配置，target／draft分开capture并合并相同graph。降低graph占用不等零内存：四策略30.39→10.69GB，单策略仍仅7.81GB。
- **设置**：集群最多64 H100，另有A100平台；Qwen2.5-7／32B、R1-Distill7B、Llama70B，Eurus-2-RL子集、GRPO温度.9／最大32K，TP2／4／8；SGLang rollout、veRL训练。主计时为warmup后**三步均值且使用已训draft**，非含全部warmup的长期time-to-quality。
- **结果边界**：表3的7B由8卡1.21×到64卡1.76×，32B的32／64卡为1.83×／2.12×、8／16卡OOM；不能把摘要“over1.7×”推广每种规模。消费卡表2是BS1／TP1生成微基准，不是消费卡完整RL收益。
- **成本／迁移**：额外阶段切换小于1%、激活SD需最长约3秒re-prefill，并存在spot协调竞争；“免费”是利用空闲预算而非零计算、显存或I/O。长尾／partial只影响drafter供数，分布保证仍要求正确target验证及基线规则；未核实现。

### 6.25 推测配置与在线适配

本批完成B9前六篇T2。接受率、生成计时、全步收益及理论前提分别登记；仅论文阅读，不计源码或性能复现。

#### 6.25.1 RLHFSpec

- **来源／分类**：[v2 §5–7](https://arxiv.org/html/2512.04752v2)，T2完成；L1/L3/L4 A/B。
- **机制**：以接受token估计／验证时间预测选树；迁移短prefix、低接受率请求到轻载实例，固定请求数，不按效率删样本。
- **适用条件**：离线profile＋在线更新、树搜索形状假设与冷却限制；局部TPS最大不等全局makespan最优。两阶段KV迁移不替代完整采样状态契约审计。
- **设置**：8 L40S、veRL `0759489`、Llama3.1-8B-Instruct＋EAGLE、LMSYSChat1M／GSM8K，最大生成2048。
- **结果**：相对自身veRL，平均rollout 2.10／2.17×，全步1.47／1.42×；相对OpenRLHF的更高倍率还混入其无参数offload导致的microbatch差异。
- **限制／迁移**：消融1→1.18→1.95→2.32是逐项累加配置，不是独立乘数；总管理开销低于3.87%仍非零。优先复用成本建模思路，不据论文认证迁移、数值或收敛实现。

#### 6.25.2 Bebop

- **来源／分类**：[v1 §3–6及理论限制](https://arxiv.org/html/2606.12370v1)，T2完成；L1/L4 A，MTP训练只改draft，target验证不可省。
- **机制**：拒绝采样接受率`α=1−TV(p,q)`；预RL冻结backbone，按多步接受率乘积优化TV，默认RL期间不再更新MTP。这里e2e指多步draft目标，不是全训练计时。
- **理论核验**：§3.2的`max p`随熵单调下降并非一般事实。本轮三类反例`(.6,.399,.001)→(.7,.15,.15)`，H为.680002→.818808 nats、max为.6→.7；下界`max p≥exp(−H)`不推出单调。仅限定论证，不据此否定实测。
- **设置**：预RL主模型Qwen3.5-35A3B，batch256／256K／一epoch、五步MTP；RL用Qwen3.6／3.7，veRL＋SGLang，推理64K／SWE128K。预训练batch不移作RL batch，GPU配置未列。
- **结果口径**：同Qwen3.5对照Agent接受率90.3→97.0%，是6.7个百分点；RL step报告1.5–1.8×，agent rollout最高2.4×不能互换。跨3.6／3.7表格不当纯loss消融。
- **限制／迁移**：熵不敏感仍依赖SFT覆盖范围及失配假设；在线CE可能破坏已有收益，不能推为所有drafter都无需在线训练。尚未核实现或跨任务性能。

#### 6.25.3 FastGRPO

- **来源／分类**：[v1 §4–5、附录B](https://arxiv.org/html/2509.21792v1)，T2完成；L1/L2/L4 A/B，完成序列提前退出不等丢弃长尾样本。
- **机制**：用单token forward的batch拐点估计`C_peak`，随active batch调draft／验证配置；每步复用target响应与hidden／logits更新draft。
- **边界**：`C_peak/B`是profile启发式，不是长KV下通用硬件常数或全局最优证明；在线只训练draft仍须正确target验证。
- **设置**：H800 SXM，4 prompts/GPU×8 responses、10 epochs；Qwen2.5-7B系列／Llama3.1-8B／R1-Distill-7B，数学数据。GPU总数与最大长度未列，ShareGPT预训68K×10 epochs非免费。
- **结果／归因**：报告全步2.35–2.72×；消融均值完整2.49、无在线1.74、无并发感知2.21、AR提前退出1.47。原始倍率不能全归并发感知或单一SD机制，需同提前退出对照分账。
- **限制／迁移**：在线训练称占2–3%；无预训消融在128步后才开SD，非冷启动即免费获益。独立draft评测温度0不代替RL采样设置，未核源码与训练质量复现。

#### 6.25.4 System Integrated SD

- **来源／分类**：[Iso等v1 §3–5](https://arxiv.org/html/2604.26779v1)，T2完成；L1/L4 A/B，NeMo RL＋vLLM，不混作旧NeMo-Aligner。
- **机制**：训练forward复用hidden／logp监督EAGLE3，detach隔离draft梯度；主实验固定in-domain draft、k3，target概率仍用于GRPO目标。
- **设置**：Qwen3-8B Think／Zero、DAPO-Math17K、32 GB200（8节点×4）。大规模模拟的batch4096不填入主实验未列的batch。
- **结果口径**：Think／Zero生成1.54／1.77×，全步1.35／1.41×；ngram接受长度不低却比AR慢，k3增到5／7也可负收益，不能以接受率代替速度。
- **在线／异步**：优质离线draft的Zero 1.77→在线1.78×收益小，弱draft才更受益；另测异步16节点、lag1，全步1.24×，暴露生成等待缩小不等全步同倍率。
- **限制／迁移**：最高2.5×大规模机会来自专有性能模拟，非235B／2048卡实测。SD精确验证不消除异步actor版本陈旧；先按draft初始化质量与验证成本判断，未核实现。

#### 6.25.5 SPARK

- **来源／分类**：[v2 §3–4、附录A/B](https://arxiv.org/html/2608.04962v2)，T2完成；L1/L2 A，v1名SpecRoll，不新增身份或沿用旧机制。
- **机制**：同prompt共享低秩logit修正B，每轮rollout重置；已完成验证的稀疏p/q反馈更新B，projection A跨rollout学。无额外target forward不等零计算／显存。
- **精确条件**：当前轮q须先于验证固定并正确传入，验证随机数独立；附录据此恢复组联合分布。不能反用本轮结果改已提交q，也不保证逐run轨迹逐位相同。
- **设置**：七模型1.5–14B×三数学集共21对照；组8、温度1、top-p .95、prompt／response各2048、rank8。EAGLE3／DFlash-style配对共同在线训练；硬件／总batch未列，预训排计时。
- **结果**：两draft家族平均全步2.31／2.11×AR，较自身draft基线增32.2%／8.5%；这是21项倍率均值，非汇总wall-clock。Qwen3-4B一组1.55→1.55舍入持平，不称严格全赢。
- **负项／迁移**：跨rollout保留B或全prompt共享B反而较差；只支持数学证据，不外推agent。proposal在线反馈不等actor在线改写，未核实现或优化收敛。

#### 6.25.6 SpecActor

- **来源／分类**：[v3 §4–5](https://arxiv.org/html/2511.16193v3)，T2完成；L1/L3/L4 A/B，解耦draft与验证，动态窗口限制失败浪费。
- **机制**：离线draft ladder估计接受率对应速度；初始单方法、尾部借空闲worker加draft与verifier，FoN取最先完成验证EOS的路径。冻结draft的稳定接受率是经验前提。
- **契约待核**：论文称exact matching，但FoN取胜与随机输出的耦合／独立性细节不足以由正文直接审计；不能仅凭每路target验证就认证按完成时间后选仍保组分布，也不能据此断言已发现实现偏差。
- **设置**：vLLM0.10、256 Hopper 80GB、Qwen2.5-32B，温度1／20K；GRPO／DAPO／PPO batch为8192／16384／4096（GRPO含组因子）。200步trace抽至少10%checkpoint重测，不是连续跑完整训练。
- **结果**：dense trace平均全步1.5–1.7×、rollout2.0–2.4×；235B另仅测首尾少量checkpoint，全步1.4–2.3×，不当200步MoE质量证明。RLHFuse基线为作者重实现。
- **限制／迁移**：模型预驻留隐藏部署成本但不免KV恢复；预测模型、空闲worker及整batch可驻留是适用条件。无法从抽样计时证明time-to-quality或全流程正确性，下一步只登记待核契约，不启动设备任务。

### 6.26 共训与跨阶段执行

本批完成B9去重后的余五篇与B10 SkyRL-Agent，共六篇T2。训练目标、有效梯度集合及系统执行重排分账；不增加源码或完整复现计数。

#### 6.26.1 Online Draft Co Training

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2609.07108v1)，T2完成；L1/L2/L4 A/B，target features stop-gradient，共训不回传draft梯度到policy。
- **机制**：packed zigzag-ring分别算causal主序列与rank-local branch attention，再以log-sum-exp合并；TapChannel侧路传跨PP features，mailbox stamp及CUDA事件管理buffer复用，不改policy拓扑。
- **设置**：NeMo RL／GRPO，8B主实验H100、TP/PP/CP=2/2/2，数学输入4096／响应16384、工具任务32K；更大MoE用GB200。总卡数／batch未列，不由EP度直接推卡数。
- **结果口径**：表1跨模型全步1.16–1.88×、rollout1.19–2.23×；工具任务全步1.25–1.43×。无SD且无共训对照不独立证明在线优于冻结draft，奖励／KL曲线近似不当逐位相同。
- **算子／开销**：256K为attention扩展测试，不是完整RL长度；相对USP最多2.9×还含免padding。PP表前10更新共训开销13.6–34.3%，tap等待仅其中一部分，不把总开销归为通信。
- **迁移**：保持target验证与stop-gradient，先判断draft质量及PP slack；不把高接受长度推广MoE／linear-attention高全步收益，未核具体实现。

#### 6.26.2 MTP RL

- **来源／分类**：[ACL Findings论文§3–4、附录A/E](https://aclanthology.org/2026.findings-acl.1871.pdf)，T2完成；L1/L4，解码A，**主模型辅助目标另C**，不与Bebop同篇。
- **机制**：先冻结主模型，以自蒸馏数据训练跨层共享MTP；RL阶段`I[A≥0]`只门控MTP的KL项，CE另加。不是直接从RL loss删负优势轨迹。
- **目标边界**：式5及算法1明确主模型也更新`RL+Adv-Opt`，不是仅draft适配；严格接受描述不代替随机采样精确证明，更不能推出训练更新与纯RL相同。
- **设置**：Qwen2.5-7B／14B、GRPO／GSPO、Slime＋SGLang；8题×32响应、温度.8、三次draft。预训batch16×5 epochs；硬件与最大长度未列。
- **结果**：主表rollout1.30–2.24×，单步最高3.15×；短0.4 epoch另计装备成本，全程时间降16.36%／35.58%，不是同一计时。表1／4原图核7B数学76.8→76.2、14B代码85.4→84.2，不认定严格全无损。
- **限制／迁移**：仅adv-KL的τ1.20低于CE＋adv-KL的2.49；正优势门控与共享梯度共同改变训练，需按算法＋系统联合方案评估，不能只按接受率排名。

#### 6.26.3 AReaL DTA

- **来源／分类**：[v2 §3–5、附录A/B](https://arxiv.org/html/2602.00482v2)，T2完成；L1/L3/L4训练侧A/B，不是推测decode tree，也不接续prompt_mean专项。
- **机制**：DFS只保活跃root-to-leaf状态，汇入后代loss梯度后回传共享prefix；优化顺序＋分块重算控制显存，额外NoGrad anchor forward有约7%开销。
- **调度边界**：按DFS顺序连续分区、二分最大tree-token成本；最优仅限固定顺序／成本代理，非任意树切分或实测时间全局最优。共享计算须保留各轨迹loss贡献。
- **设置**：8 H800、GRPO／τ²-bench、训练context16K；1.7B／8B全流程，4B／14B另为消融。原始树压缩9.43×，不是9.43×rollout加速；batch未列。
- **结果口径**：局部训练最高8.31×对Dense+CKPT，梯度计算消融不含optimizer；全流程1.28／2.28×同时将rollout/train卡比由2/6调到5/3，不能全归单kernel。
- **负项／迁移**：GSM8K低共享时1.7B仅.62×、8B .86×；接近无共享的AIME 14B为.95×。训练复用机会依赖工作负载，异步奖励相似不等数值／梯度逐位一致，保留训练侧范围账。

#### 6.26.4 FlashNext RL

- **来源／分类**：[v1 §2–5](https://arxiv.org/html/2610.28998v1)，T2完成；L1/L4 A/B，冻结层及零优势组处理另C，模型特定混合证据不转算纯rollout系统。
- **机制**：exact-size pinned arena、GPU expert梯度累积、稀疏attention／MoE kernels；共享host n-gram、分层权重广播与有界批buffer应对host峰值，router replay降低训推失配。
- **设置**：Qwen3.8-Flash-Next，4×8 B200，一节点训／三节点推；BF16主路径、131072 context、672在途、目标1024可训rollouts；冻结首两层、n-gram、indexer和MTP，非全部参数更新。
- **比较边界**：full-model 395→4438 nonpadding tokens/s/GPU的长度／batch变了，不作受控11×；8层同形状累加梯7.4×也非48层完整RL收益，编译首级反而−19.9%。
- **live口径**：步骤5–15均值22.4分钟、收1393／训1026；全零优势组不训练，故不是固定生成工作量。4737 trainer tokens/s/GPU含padding且只计forward/backward，不与4438直接比较；11–14节点需求仅乐观外推。
- **限制／迁移**：GPU kernel提升须先解host瓶颈；共享内存用PSS、不累加RSS。短跑熵／KL有界不证任务质量或长期收敛，live关闭expert prefetch以保host余量，不照搬开发最优配置。

#### 6.26.5 Rollplex

- **来源／分类**：[v1 §3–5及生命周期附录](https://arxiv.org/html/2608.14498v1)，T2完成；L2/L4 B，主路径保持同步snapshot更新，不因kernel并行归off-policy。
- **机制／契约**：response-independent的reference／actor prefix放到decode窗口；actor边界KV保autograd，reference detached，更新前所有旧权重读者静止，chunk update前先完成全局norm／clip。
- **内存／布局**：VMM稳定虚址、生命周期offload与optimizer流式驻留；跨TP仅整数比且布局兼容的张量共享，不兼容QKV重建。92.8%共享不是任意模型全参数零拷贝。
- **设置**：ROLL、Megatron0.13／vLLM0.17、32 H800、Qwen2.5-VL-32B、四视频集；32题×8、prompt8K／response2K，首步排warmup。所有基线已有相同prefix共享。
- **结果**：全步较串行colocate1.23–1.30×、同32卡disagg1.57–2.24×；收益受decode窗口与干扰约束，不能归为新增去重。H20／7B资源策略消融不挪作主硬件结果。
- **限制／迁移**：prefix短则机会小；叠加partial／oversampling只比各自开关，不横排改变工作量的绝对时间。MPS／IPC共享无独立故障恢复边界，不能由奖励接近推逐位轨迹相同。

#### 6.26.6 SkyRL Agent

- **来源／分类**：[v1 §3–5](https://arxiv.org/html/2511.16108v1)，T2完成；L4，dispatcher B，**工具recipe／有效梯度集合另C**，修订此前C?。
- **机制**：init／agent-run／reward各有界队列，pipeline重叠CPU与GPU工作；transition记录token／可得logp，经post-process接训练后端。执行异步不自动意味跨policy更新异步。
- **训练边界**：SWE将超32K context／50 turns终止的轨迹屏蔽梯度，仍参与reward／advantage；不改reward不等更新目标不变。AST搜索、提示及leave-one-out／去归一化另属recipe。
- **设置**：SWE Qwen3-32B、R2E-Gym4.5K、batch＝minibatch64×8响应，系统图2×8 H100。评测40K／100步且无AST工具，不能将训练与评测预算混同。
- **结果口径**：pipeline较bounded batch约1.55×针对生成调度；SA-SWE 39.4% Pass@1、4601 H100-hours相对DeepSWE 9180还混入recipe／样本效率，不能全归调度。其他scaffold报告分数不直接横比。
- **反例／迁移**：OSWorld训练reward升而验证几无改善；工具服务不足会污染轨迹，不仅拖慢吞吐。token原样传输不证明版本／数值一致，纠错提示也不等工具副作用可安全重试；论文与现代SkyRL代码地图分开。

### 6.27 Agentic执行与训练契约

本批完成B10八篇T2，累计9/12。异步发生在租户、micro-batch还是policy更新之间须分别核；以下是论文证据与阅读判断，不是源码验证或机器复现。

#### 6.27.1 EARL

- **来源／分类**：[v1 §3–5](https://arxiv.org/html/2510.05943v1)，T2完成；L4 B，原型动态并行只覆盖rollout，非训练全阶段自适应。
- **机制**：先profile长度桶／TP吞吐，按平均context在下一轮rollout前切策略；layout-aware直接all-to-all替代集中汇聚／重分发。不是活跃请求KV在线迁移。
- **设置**：ROLL、Qwen2.5-72B-Instruct、Connect Four、自定REINFORCE；16×8 H100 80GB，vLLM0.8.4。32响应短context的TP4较TP8吞吐高31%，长context优选反转，128响应32K时TP4 OOM。
- **分发结果**：reference-logp张量46／93／187MiB每worker，TCP-over-Ethernet同基线下降9.7–11.2×传输延迟；未把advantage所需全部张量纳入此去中心路径，也非RDMA实测。
- **计量边界**：开篇1024卡／200B／1TB／20分钟是工业动机，不是该128卡受控实验；局部通信与decode profile不能相乘为完整训练收益。
- **限制／迁移**：平均长度不是峰值显存保证；切换成本与长期质量缺完整对照。适合提炼长度—并行度交叉点与布局契约，不据原型推任意agent负载通用加速。

#### 6.27.2 InfoTree

- **来源／分类**：[v1 §3–5、附录B/C/M/O/P](https://arxiv.org/html/2605.05262v1)，T2完成；L3/L4，**树选样／预算C**、异步展开B，不是固定样本集prefix复用。
- **机制**：UUCB结合Q、探索、action-token熵、深度／成本选中间状态；ABA给困难组追加预算及高温救援，分层advantage也改变训练统计。旧Q快照最多差2版，回收时重评分、仅当前top-K接受，否则回滚；这不是actor policy年龄硬界或精确推测解码。
- **设置**：16 A100 80GB、veRL、Qwen2.5-7B-RA-SFT，128 prompts、3 epochs，训练16叶／评测32，tool cap5、深度6；另有1.5／14B规模扫描。
- **结果**：达到Mean@32＝28的步数248→138，报告time-to-target 1.57×；每步57.3→65.5秒反而慢14.3%。Speculative Expansion降至60.1秒，非全步2.9×。
- **预算边界**：ABA将混合结果组58.1→76.3%，同时平均叶16.29→17.08、每步65.5→66.9秒；不据此称严格等16叶预算。梯度非零量也不等梯度质量或收敛保证。
- **理论限制（阅读判断）**：独立二元结果全同概率为`p^B+(1-p)^B`，固定`0<p<1`时随B趋零；有限预算的Bδ界不能推出任何预算均无效。单调次模F／精确贪心的条件保证，也不能仅凭附录的一阶UUCB对应、经验均值−方差／visit熵与相关性就转成实际算法保证，保持待核。

#### 6.27.3 AsyncWebRL

- **来源／分类**：[v3 §3–5、附录A.5/B/C.5](https://arxiv.org/html/2606.05597v3)，T2完成；L4 B＋**C**，修订算法边界C?。
- **机制**：常驻web worker、截图驻专用内存actor只传引用，重叠rollout／update／refresh；轨迹可跨policy版本。proximal／behavior分解IS加PPO／dual clipping，不据未截断IS推最终目标严格无偏。
- **目标／选样**：常数1/10替换轨迹**步数**倒数，而非token长度；跳过全成功／全失败组，收128条混合结果轨迹（G8），KL为0。改变工作量／梯度集合，不是只换执行次序。
- **设置**：2×8 B200，Qwen3-VL-8B两变体、SGLang context16K／每步最多4096新token；训练horizon10/20/30、测试30/50/70。RAFT++ batch120与GRPO128、资源分配不同，不能当完全同配置。
- **结果口径**：约3100对1300／1050 trajectories/hour的2.4–2.9×是端到端轨迹吞吐，含recipe／长度变化；Instruct成功45.4对42.9是+2.5pp，非纯异步质量收益。RAFT++反而39.3。
- **限制／迁移**：1/k运行前后20步的11–15%更新／18–19%全步收缩是同一run轨迹变短，不是固定工作量跨方法速度比；单种子且报告peak test，不升级成稳定等质或通用系统排名。

#### 6.27.4 AcceRL

- **来源／分类**：[v3 §3–6、附录C/E/F](https://arxiv.org/html/2603.18464v3)，T2完成；VLA边界、L4 B；**GIPO／延迟norm／world model C**，非纯语言rollout证据。
- **机制**：环境／推理／训练三段解耦；推理攒够batch或最老请求超时即发，FIFO replay无明确policy年龄硬界。micro-batch forward复用value／GAE，整批累梯前冻结权重；不能把这一条件等价扩展到所有训练语义。
- **统计边界**：advantage用上一更新步全局均值／方差，附录承认平滑仅近似；GIPO软权重不等精确IS。imagined短轨迹与预测稠密reward另改动力学／目标，potential shaping仍须核γ及终止条件。
- **系统结果**：OpenVLA-OFT、H200；表1四卡2000步**估算**6.5h对RLinf15.8h／SimpleVLA16.7h，约2.4／2.6×；七trainer卡104.22 SPS含ZeRO2带来的更大micro-batch，不是同batch纯扩卡。LIBERO Goal98.8仍低于SimpleVLA99.2。
- **样本结果与冲突**：Spatial约200×指到相近return所需真实环境步，不是wall-clock倍数或免离线数据；摘要／图5写预训练1000条、§6.4却写2000条，**离线预算待核**，不能无条件认证同总成本收益。
- **限制／迁移**：DIAMOND WM配置batch1280、想象horizon2，与Cosmos batch128／horizon8及model-free设置分开；附录总594.5 H200-hours不自动给每个对照完整成本。Cosmos跑通闭环不等同质高效，未据此新增M5设备实验。

#### 6.27.5 MARLaaS

- **来源／分类**：[v1 §4–7、表2–4](https://arxiv.org/html/2605.08527v1)，T2完成；L4 B，**multi-tenant不是multi-agent**。
- **机制／版本**：冻结base共享、每租户独立LoRA／optimizer；跨租户推理批处理，FIFO trainer一次更新一租户。租户完成一批后等该租户新权重再采样，不等所有任务异步都使用旧policy。
- **设置**：最多两节点、每节点16 Ascend 64GB，Qwen3-0.6／14／32B，AReaL1.0.1＋vLLM-Ascend0.14rc1；wiki-search max生成1024、batch32，GSM8K2048／64，AMC4096／32。此为论文实现，不改M5专项。
- **结果口径**：十个wiki任务副本各100步，14B总wall-clock24.48→3.72h（对SingleDisagg）；这是1000聚合步／任务批makespan，不是单任务每步6.6×。AI Core利用率14B也仅4.45→8.67%，倍率大不等饱和。
- **并发代价**：0.6B扩1–32任务提高聚合吞吐、TTFS优于串行，但TPTS对独占单任务有额外开销；不得同时宣称总吞吐与每任务延迟均无代价改善。
- **限制／迁移**：KV准入为估算而非硬内存／SLA保证，reward-versus-step接近不证明逐位等价、强租户隔离或公平性。更适合独立LoRA任务池，不直接外推共享policy的多agent训练。

#### 6.27.6 FlexMARL

- **来源／分类**：[v1 §4–6](https://arxiv.org/html/2602.09578v1)，T2完成；L3/L4 B，multi-agent执行流水化与整批更新分开。
- **机制／契约**：experience store按agent／版本／字段ready发布，micro-batch提前算梯度并缓存，**global batch结束统一update**；数学累梯等价还需相同组优势／normalizer／固定参数，不是收到一条立即更新。
- **调度／状态**：query与分支并行、agent内最轻负载、agent间按queue gap迁资源；未完请求重排，不等工具副作用exactly-once。host checkpoint／本地性恢复未枚举所有RNG及缓存状态，不认证完整容错。
- **设置**：48节点×16商用NPU 64GB，vLLM0.9.1／DeepSpeed0.16.9 ZeRO3，GRPO global64／micro16、max response8192；MA为Qwen2.5-14B，CA含14／32B，私有业务数据。
- **结果口径**：MA平均每样本全阶段914.4→126.1s为对naive MAS-RL 7.3×，对MARTI174.1s仅1.38×；CA相应5.6×／1.43×。此E2E定义不是完整job wall-clock／time-to-quality，不与7.6×TPS混作同一数字。
- **限制／迁移**：平均利用率MA32.4%／CA19.8%仍非饱和；32B offload3.8s、完整swap最高11s，隐藏须有足够并行窗口。作者称逻辑不变而未给独立质量实验，不能据吞吐表认证同质收敛。

#### 6.27.7 Agent Lightning 2025

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2508.03680v1)，T2完成；L4，Training-Agent Disaggregation B、**credit／AIR C**，与2026为不同文献。
- **接口机制**：client拥有应用／工具逻辑，trainer提供LLM端点，以观测或proxy收每次调用的input／output／reward／metadata；解耦软件职责不自动保证跨更新on-policy。
- **算法边界**：LightningRL当前将最终return分配给每个LLM调用，再按任务组transition做GRPO；调用数影响训练统计。可选AIR从工具状态塑形reward，不是只把同一损失搬到另一进程。
- **实验**：Llama3.2-3B-Instruct，Spider SQL／MuSiQue RAG／Calc-X，分别接LangChain／Agents SDK／AutoGen；SQL选择writer／rewriter transition训练，checker未选不等共享参数完全不受更新。
- **证据范围**：展示任务学习曲线与接入能力，未给可认证的受控rollout TPS／全训练速度对照；硬件、batch、长度对齐不足，不编造加速倍数。
- **限制／迁移**：transition格式免整条轨迹强拼接，不代表实际prompt天然短或prefill去重；故障重派也不证工具可安全重试。可复用的是事件与训练数据接口，credit、组归一化、token／版本语义仍须独立核。

#### 6.27.8 Agent Lightning 2026

- **来源／分类**：[v1 §2–4、附录A](https://arxiv.org/html/2608.17528v1)，T2完成；L4 B，**优势／loss／选样另C**，系统v1.0不是2025论文的arXiv v2。
- **token契约**：保留实际prompt／action token，仅前缀逐token精确匹配才合相邻请求，否则另开训练行；不能补写prompt强拼旧action。36% rollout仅一行、均值2.41行，rollout不等sample。
- **更新契约**：先算rollout级group advantage，所有子行放同一optimizer step，按rollout内部response-token均值再跨rollout平均；区别于逐行／整batch token平均。共卡async drain当前调用再更新，episode仍可跨版本，不是整轨迹同policy证明。
- **状态／重试**：控制API幂等不等生成幂等；同prompt留最后一次的去重规则须再核request／attempt身份和真实harness消费，未据文字判定源码bug。K8s状态仅best-effort eventual consistency，不保证工具副作用exactly-once。
- **实验口径**：Llama3.2-3B搜索batch512×4，EM25.1→41.7；Qwen3.5-9B coding完整变体128步38.2对baseline35.0，**仅改rollout-adv为33.1负项**。SWE Verified41.8→56.4同时含数据／环境recipe；约2×效率宣称缺充分配套硬件／计时表，不认证通用等工作量倍数。
- **限制／迁移**：SWEsmith清理与难度筛选、联网白名单和防reward hacking另账；严格token合并可复用，不能把选样、loss修订及质量提升全部归给执行解耦。论文阅读不等已核现代Agent Lightning实现。

### 6.28 弹性执行与样本语义

本批完成B10余三与B11前五篇，共八篇T2。资源来源、更新批次与训练target分别核对；局部profile、短跑和完整训练不混作同一速度指标。

#### 6.28.1 TideRL

- **来源／分类**：[v1 §4–6、附录B/C](https://arxiv.org/html/2608.10402v1)，T2完成；L2/L3/L4 B，**弹性batch另C**，修订暂B。
- **机制**：CTB按token占用准入、暂停／亲和恢复；ready backlog与到达间隔决定Ref／Actor拆分或同置，rank在更新边界迁角色。跨policy更新清KV并重prefill，不是复用旧KV或零重算。
- **更新契约**：允许batch落在`[Bmin,Bideal]`、缩fetch保留reservoir；因此消费集合／更新频率未必等固定batch。准入限流不独自证明在途轨迹年龄硬界，恢复可混版本。
- **设置**：32 H100＋独立1024核环境；vLLM／Megatron，Qwen2.5-7／14B、Qwen3-VL-4B／3.5-9B。文本20 turns／8K，GUI50 turns及滑动图像；batch上下界具体数值未列。
- **结果口径**：文本训练goodput较veRL5.6×、AReaL1.8×，是被trainer消费token/s；文本100步pass-rate仍落后同步0.5pp、耗时为48.9%。不能把goodput等同同质收敛速度或逐位更新等价。
- **限制／迁移**：CTB缓存命中1.58×、生成1.15×为局部，RA2P降44.3%训练时段不乘到全流程；GUI40步有轻微质量损失。状态／数据ready可指导资源切换，但不凭监控loss认证算法正确。

#### 6.28.2 ROSE

- **来源／分类**：[v2 §4–6](https://arxiv.org/html/2605.06534v2)，T2完成；L3/L4 B，DAPO非零方差组选样沿用recipe另账。
- **机制**：serving优先VMM内存共享、TTFT／TPOT slack准入；压力时abort rollout请求、turn级亲和重路由。保留任务进度不证明取消生成的RNG／token完全相同。
- **权重契约**：Mooncake分片异步传输、零差COO delta并在旧权重上加回；不是稀疏policy。需相同base版本、dtype及原子安装，论文未给逐位重建审计，不只凭零差率认证。
- **设置**：H800专用16／48卡＋最多借16／64 serving卡，跨集群200Gbps；Qwen3-8／32B，32K、G16、batch256／1024；Microsoft24小时trace，P99 TTFT／TPOT双SLO。
- **结果口径**：GRPO平均1.31–1.46×、DAPO1.42／3.31×对固定池，含额外serving容量；对RLBoost+的1.20／1.26×针对rollout时间。通信12.4／9.1×只是同步，不当完整训练倍率。
- **限制／迁移**：无SLO违规限所测trace及模型配对，不是任意突发硬保证；借卡并非免费，模型预部署／激活最多5秒和HBM占用仍有成本。turn重路由、版本安装与副作用重试须分别验收。

#### 6.28.3 QwenGyre

- **来源／分类**：[v2 §4–6、附录A/B](https://arxiv.org/html/2609.33848v2)，T2完成；L3/L4 B＋**C**，明确修订路径处理C?。
- **执行契约**：core持唯一optimizer，satellite从固定权重快照加入累梯，整batch统一norm／clip后一次update；水位控制cell回收，harness／workspace不迁，模型请求可取消重试并RDMA交接KV。
- **target契约**：TITO留原context／behavior logp；按角色优先级最多5路径、共享target只训一次，execution内token均值再跨execution平均。排除无有效评分、部分成果评分均另C；固定target换路径归属不变，**采样本身会改变target集合**。
- **年龄边界**：平均调度陈旧`φ+(μ−1)/2`需固定B／μ、稳态、每组恰消费一次；度量从准入版本算，不是每次生成policy年龄、最大lag或有限48步的自动证明。
- **设置／结果**：122B用32节点八cell、2.4T用192节点四cell，GPU型号未列；NL2Repo G16、B32，Megatron／SGLang。2.4T48步75.42h对Async134.55h／Colocate91.47h为1.78／1.21×，不套用跨任务最大1.85×。
- **限制／迁移**：cap5的forward/backward降25.2%非纯调度；cap1质量差。多步burst不能全局shuffle，排序质量影响未隔离；改小cell的2.2%仅同32节点，不外推小卡预算。700K／近1M为执行生命周期token量，非单请求context。

#### 6.28.4 ARL-Tangram

- **来源／分类**：[v1 §4–6](https://arxiv.org/html/2603.13019v1)，T2完成；L3/L4 B，工具并行语义另核，不当policy内核优化。
- **机制**：action而非trajectory长期占资源；FCFS候选＋弹性分配启发式，以平均ACT（排队＋执行）为目标。单关键弹性资源假设、局部DP／近似后续ACT，不是全局最优或RL尾延迟最优证明。
- **资源契约**：CPU按exec改cgroup、保容器内存；GPU缓存静态服务、host恢复／LRU逐出，API按并发／quota限流。`pytest`改并行执行需核测试顺序／状态等义，不能默认reward完全不变。
- **设置**：veRL最多48×8 Hopper训练卡；外部15×256 CPU核与5×8 GPU，型号未细列；Qwen3-32B／MiMo-V2-Flash，coding／MOPD／Search batch1280／2048／2048，部分内置数据。
- **结果口径**：报告平均ACT最高4.3×，十步平均RL step的coding仅1.4×、Search1.5×；不同任务用不同专用基线，不能横排成通用加速或将调用延迟倍数计为整步倍率。
- **负项／迁移**：MOPD仍被最慢轨迹支配、整步收益小；Search单reward模型有restore反而更慢。流控减少API失败也改变有效反馈，不能以平均ACT提升推所有轨迹分布／质量严格相同。

#### 6.28.5 DART GUI

- **来源／分类**：[v1 §3–5、附录A.4](https://arxiv.org/html/2509.23866v1)，T2完成；L3/L4 B＋**C**，不与DARTS同篇。
- **机制**：env／rollout／manager／trainer解耦，单轨迹结束就重用环境、逐worker换权重；其他worker继续旧版服务，不是整轨迹同版本保证。
- **选样／目标**：高成功率少采、按历史成功长度截轨迹、全败组补旧成功样本、仅top80%高熵step训练。组adv在所有step上归一，长轨迹重复reward改变组统计；截断IS（C＝1）不能认证无偏。
- **设置**：UI-TARS-1.5-7B，203个OSWorld训练题，8 H100 trainer／每rollout worker2 H100（总worker数未列），180 Ubuntu环境、初始N8／最多30 turns。
- **结果口径**：22.6→43.6 actions/min为非解耦对照，非tokens/s；42.13对27.52＝**+14.61pp**，正文12.71算术不符。两模型max steps30对100，不当同预算系统质量收益。
- **限制／迁移**：45训练题消融72.28%不替全OSWorld42.13%；成功池额外采集成本另账，训练基准题不认证独立OOD泛化。表中高熵等开关可能累加，不把每列差都归单模块独立因果。

#### 6.28.6 MCP Universe RL

- **来源／分类**：[v1 §3–4、附录B–E](https://arxiv.org/html/2608.22167v1)，T2完成；L3/L4 B，流水与异步placement分账。
- **机制**：每trajectory独立环境，hold至评分完成再reset／destroy；共享配置预热、SWE按需Apptainer。Acquire／Run／Eval分别限并发，MCP统一调用不替代清理／评分逻辑。
- **设置**：gpt-oss-20b GRPO，batch16×8、max seq128K，SWE最多100 turns；吞吐实验3 H200、无高速NVLink，acquire24、run24–120、live环境固定120。
- **结果口径**：warmup后五步147→410 rollout tokens/s（2.8×）；placement同并发120，最初五步1164→572秒／step约2×。两消融不同，不相乘为全训练5.6×。
- **版本边界**：async独立GPU池允许最多一步旧轨迹；同policy／evaluator不证明跨更新统计信号严格相同。短五步计时也不证明长期等质收敛或均摊所有环境冷启成本。
- **迁移**：适合把环境生命周期与GPU队列分开调；多域奖励曲线是训练信号，不是跨域独立评测SOTA。保持live120的对照支持内存预算不变，不意味着任意加run worker无需环境容量。

#### 6.28.7 PEARL

- **来源／分类**：[v1 §4–6](https://arxiv.org/html/2609.35158v1)，T2完成；L2/L3/L4 B，PD选择与外部扩卡／训练卡借用联合。
- **机制／契约**：profile估batch makespan、枚举PD比例，hysteresis与收益门槛控切换；稳定worker/GPU绑定，借卡必须先归还训练。中断保生成prefix／环境进度再提交，新worker权重就绪才准入。
- **设置**：32 H800 80GB／800Gbps IB；训练固定16卡，rollout保留4＋最多12抢占卡，Bamboo三段各2h trace。Qwen3-8／30B-A3B、GRPO G16、rollout batch64、32K、lag≤1步、TP2。
- **结果口径**：同资源trace对RLBoost+平均+5.8–26.9%／+7.4–36.3%；对固定ROLL2.17–2.79×还含额外卡。指标为生成＋训练token/step时间，不当task-quality收益。
- **负项**：四rollout卡强拆PD降20.6／30.9%；gate估高成本时会错过借卡机会。模型以平均参数／理想KV带宽估算，不是每个长尾trace严格最优。
- **成本／迁移**：扩容准备56–67s、finalize暂停新准入11–18s；借用激活约6s、归还3.2–4.4s。恢复额外prefill<1%仅该项开销，非整个scale-in零成本；不以吞吐曲线认证精确policy等价。

#### 6.28.8 SAO

- **来源／分类**：[v1 §3–4、附录A/B](https://arxiv.org/html/2607.07508v1)，T2完成；L4算法边界 **C**，不是GRPO等价系统提速。
- **机制**：直接用rollout logp作behavior，DIS在概率比区间外将token梯度mask为0，不是通常PPO裁剪或无偏IS。critic每actor步更新两次、冻结attention，Skip-Observation GAE跳过外部反馈token。
- **设置**：Qwen3-30B-A3B-Thinking-2507、128K；SAO128题×1对GRPO16题×8，轨迹数相同而prompt多样性不同；coding OpenHands最多300 turns，硬件／完整墙钟对照未列。
- **结果口径**：SWE Verified23.0→29.8，GRPO＋DIS27.0；vanilla GRPO约160步崩，报告其崩前最后有效值，不是统一训练预算的最终质量比较。
- **额外成本／负项**：value预训练、双更新与冻结策略是联合recipe；去快critic使BeyondAIME74.8→69.8。少等组尾不等免critic成本；400步step-level GAE弱于token-level，不外推所有agent场景。
- **迁移边界**：线上写作是受控偏好切换模拟，非真实用户在线部署实证；仍依赖可靠逐token behavior概率。保存logp可免历史checkpoint重算，但不能由此认证混版本轨迹严格on-policy。

### 6.29 执行底座与数据流边界

B11余三＋补三完成T2，各6条；均读HTML正文及必要附录，不是源码或性能复现。下面的理论疑点是本轮阅读判断，不把原文主张直接当已证明结论。

#### 6.29.1 Rollout Infrastructure Tax

- **来源／分类**：[v1 §2–6](https://arxiv.org/html/2607.01415v1)，T2完成；L4环境执行成本测量、B边界，非新RL算法或端到端训练系统。
- **机制／范围**：固定命令轨迹对照单容器、hosted sandbox、K8s容器与云VM；测创建、命令／文件／测试及reset，排除模型推理和reward计算。T0极简、T1预装依赖、T2仓库补丁测试，不是实际LLM生成策略。
- **条件**：每配置100次，冷启动与10／50／150动作；逐动作数是整轨迹摊销，不等单命令服务时间。地区、硬件与缓存配置相关；不足以形成普适供应商排名。
- **结果口径**：最轻冷启动差最高110×，重T2缩至约1.9×；150动作轨迹42.5对23.4秒。百万轨迹外推11811对6495 worker-hours，约1.8×，不是GPU-hours、账单或并发job墙钟。
- **额外成本／限制**：外推按轨迹数×单轨迹延迟／3600，未覆盖推理、奖励、更新与完整生产排队；实agent命令、仓库大小和复用状态会变。不能据此声称完整RL快1.8×。
- **迁移边界**：分阶段warm pool与缓存亲和路由是设计建议，非已实现增益；后续比较应固定轨迹／隔离要求，分别报启动、交互、重置与吞吐，避免把弱隔离当免费优化。

#### 6.29.2 ExpThink

- **来源／分类**：[v2 §3–4、附录B及消融](https://arxiv.org/html/2605.07501v2)，T2完成；workload／算法边界 **C**，reward塑形与advantage重加权，不是同一生成工作量的执行提速。
- **机制**：buffer存每题历史最短正确长度ℓ*而非必然回放轨迹；目标1.1ℓ*，短正确reward1、长正确0.5、错误0，ℓ*只缩不增是统计属性，不保证当前模型始终掌握该解法。
- **优势／疑点**：以正确数k替代std，A=(r−mean r)/(k+ε)。阅读反例：二值reward、固定G时失败样本|A|=k/[G(k+ε)]随k增而非严格减；故分母设计不能独自证明所有参数梯度幅度随正确数严格下降。
- **条件**：DeepScaleR约40K、三种1.5／7／8B骨干；8 A800 80GB、FSDP、batch512、G16、最大16K、300步。评测准确率、token长度与IPT，不是系统TPS。
- **结果口径**：Qwen3-8B五数学基准平均相对token减77.18%、准确率相对增5.33%，不是百分点或汇总总token节省；MATH500为94.2%／755.7 tokens，不推导等比例墙钟收益。
- **负项／迁移**：r_pen=0消融虽IPT高，AIME24却27.9→7.92，说明比值可掩盖质量崩溃；默认优势与标准GRPO优势也非逐基准质量支配。长度塑形应与引擎优化分账，保留质量—token—训练成本三轴。

#### 6.29.3 Intelligent Router

- **来源／分类**：[v2 §5–6、附录A](https://arxiv.org/html/2408.13510v2)，T2完成；L3 serving迁移旁读，不增原生rollout覆盖；此处RL训练的是router，不是被服务LLM。
- **机制**：预测输出长度桶、按硬件profile估算混合负载，DQN向同模型副本路由或等待；混合负载启发式逐渐退场。不是按题难度选择不同质量模型，也无一般全局最优调度保证。
- **条件**：4 V100／Llama2-7B／vLLM FCFS、20请求/秒、20轮×2000请求、路由tick20ms；长度预测和decode计时依赖任务分布及硬件。
- **结果口径**：对Round-Robin，论文请求集E2E延迟指标减19.18秒／11.43%，吞吐相近；不是11.43% TPS增长。A100／Llama3.1切换需重新profile／训练，不是零样本跨平台迁移。
- **额外成本**：DistilBERT batch64约10ms GPU／800ms CPU，不能视为零开销；生产来源仅抽4000请求按80RPS重放，不等保持原始一小时到达轨迹的在线实证。
- **迁移边界**：适合借鉴负载估计而非照搬目标；RL rollout另有整组完成、策略版本／年龄与演化长度分布。router的TTFT／TBT收益不直接认证训练time-to-quality改善。

#### 6.29.4 AgentRL

- **来源／分类**：[v1 §3–4、附录C/E](https://arxiv.org/html/2510.04206v1)，T2完成；L4 B；主动cross-policy与task优势归一另 **C**。
- **执行契约**：coroutine异步rollout／trainer、动态batch区间、有界队列并每步搬走就绪数据；统一function-call与容器环境生命周期。队列有界不自动给逐轨迹年龄硬界或零off-policy保证。
- **采样／目标**：每action可随机换policy；训练用间隔多步才刷新的stale engines实现主动新旧策略混合，非仅被动通信延迟。按task内全部token优势零均值／单位方差，改变任务权重而非纯调度。
- **条件／结果**：H800，14B至少16卡，G8、temperature0.8、SGLang／FSDP；改编AgentBench-fc五任务，32B平均37.2→70.4。四次是评测重复，不是四个训练seed；质量增益不当吞吐倍率。
- **负项**：14B多任务平均67.7对专训组合67.8，DB与Webshop仍弱；BFCL部分单轮项86.0→85.8。OS是命令环境，非GUI；最大batch／长度等未由这些结果形成统一系统对照。
- **理论／迁移判断**：附录C“严格扩大支持集”非无条件成立——相同policy混合即等集；支持集变大也不保证成功概率上升或语言必然有效。可以保留探索假说，但须核行为概率、task混合与实际吞吐分母。

#### 6.29.5 RollArt

- **来源／分类**：[v2 §5–9](https://arxiv.org/html/2512.22560v2)，T2完成；L2/L4 B，过旧逐出与冗余够数终止另 **C**；不沿用暂定纯B。
- **机制**：静态domain亲和映射、逐轨迹环境解耦、serverless reward；取batch后停rollout、装权重、重算KV再续跑。旧token不会因KV刷新变成新policy生成，仍可混版本。
- **选样契约**：按轨迹初始版本限制α，默认1，过旧轨迹逐出；冗余模式够数后终止在途。α增至2已有后期time-to-score退化，不能只取最高step吞吐。
- **条件／结果**：Qwen3 8／14／32B、32K、batch512、G8；主配置96 H800＋32 H20，对照128 H800；32B五步平均step快2.05×（Sync+）／1.35×（One-off）／1.31×（重实现AReaL），非完全同硬件或整job小时对照。
- **成本／负项**：serverless的158→77秒同时使本地rollout卡4→8，外部reward资源不免费；环境消融注入高斯延迟。PD的3P1D最差、最优依模型；未测Laminar的孤立收益相乘不是可信组合“下界”（阅读判断）。
- **迁移边界**：传权重有残余成本，不叫零税；3000+GPU生产与128卡受控实验分开，生产首25步1.66×含资源比例／cache调优。domain映射尚非在线自动最优，选样、混卡价格与系统重叠收益须分别核。

#### 6.29.6 AstraFlow

- **来源／分类**：[v1 §3–4、附录A](https://arxiv.org/html/2605.15565v1)，T2完成；L4 B架构，replay／GRESO／零优势过滤等可组合策略另 **C**，接口通用不保证算法等义。
- **机制**：dataflow、RaaS与trainer自主解耦，以policy／version／时间等元数据连接生产消费；支持多policy与拉取权重。保留元数据不自动解决staleness校正与整组归属。
- **主对照条件**：Solver／Verifier各Qwen3-8B，8 H100、batch256、G8、最大4096、1200轮、关闭动态采样；对verl DrMAS同硬件重计迭代，212.64→77.65秒=2.74×，是串行全迭代对重叠trainer节拍，排除eval，非完整job／对AReaL倍率。
- **质量／负项**：主准确率基线沿用DrMAS报告，非全表本地重跑；平均54.4→56.5但AIME25为41.5→40.6。另8B同配置AReaL对照137→139.6秒反而慢，不能一概宣称更快。
- **弹性成本**：14B在H200 rollout池6–11卡伸缩，约358–359→312 GPU-hours，但墙钟23.9→24.4小时、准确率68.0→67.9；省资源不等更快。代码多policy质量实验还从8扩16卡，不能当同卡系统收益。
- **跨域／数据边界**：同H200功率帽与4Gbps／300ms链路塑造异构跨区，不是真实多地异卡实测；稀疏传输有周期全量。动态采样生成200K→700K另有成本，不能只数训练接受token；四次生成评测／择优checkpoint不等多seed长跑。

### 6.30 Serving文献triage

B12首批20条范围复核：19条新增T3，Intelligent Router沿用§6.29.3的T2，不重复计深度。本批按原站版本摘要／会议介绍判定范围，不抽取未核正文的性能倍数；“不追加T2”是当前冲刺的阅读优先级，不是认定机制无用。下列迁入RL时的条件是本专题判断，而非原serving论文已经验证的训练契约。

| 条目／来源 | 本批处置与理由 | 与rollout的关联及迁移条件 |
|---|---|---|
| [Orca OSDI22](https://www.usenix.org/conference/osdi22/presentation/yu) | T3；连续批处理的L2基础，不另做RL系统轻模板 | iteration-level scheduling与selective batching可供引擎比较；逐请求能提前返回不代表同步RL的整组／整批屏障消失。 |
| [vLLM v1](https://arxiv.org/abs/2309.06180v1) | T3；分页KV基础，具体实现留C2代码路径核对 | 动态KV分配和共享有利并发多采样；仍须同版本、相同前缀及实际驻留，不能由内存节省推出最长轨迹缩短。 |
| [SGLang v2](https://arxiv.org/abs/2312.07104v2) | T3；program执行／RadixAttention基础，现代实现另核 | 前端控制流、KV复用与结构化解码可承载agent调用；约束输出会定义不同采样空间，更新policy后还需缓存版本契约。 |
| [FlashAttention v2](https://arxiv.org/abs/2205.14135v2) | T3；L1 IO优化，非独立rollout编排方案 | tiled exact attention是后端基础；训练／prefill的IO收益不直接套给逐token decode，也不等数值路径逐位一致。 |
| [FlashAttention2 v1](https://arxiv.org/abs/2307.08691v1) | T3；L1 work partitioning，与FA1保持独立身份 | 减non-matmul工作及改block／warp分工；须结合prefill长度、batch与硬件判断瓶颈，不将kernel利用率当RL气泡消除。 |
| [FlashInfer v2](https://arxiv.org/abs/2501.01005v2) | T3；可定制attention后端，不另排训练系统T2 | KV布局、JIT与负载均衡／CUDAGraph兼容性可迁；kernel和serving端到端指标分开，实际RL需核具体backend调用。 |
| [Sarathi Serve v3](https://arxiv.org/abs/2403.02310v3) | T3；serving吞吐—尾延迟权衡，不直接按SLO排名RL | chunked prefill与stall-free decode混批可借鉴；须按RL真实输入／输出比例和整组完成重评，不能预设prompt主导。 |
| [DistServe v3](https://arxiv.org/abs/2401.09670v3) | T3；TTFT／TPOT goodput目标，不是trainer／rollout拆分 | PD各自资源与并行策略可供L2/L3对照；RL迁移需计KV传输及组尾，SLO合格请求数不等训练样本消费速率。 |
| [Mooncake v5](https://arxiv.org/abs/2407.00079v5) | T3；PD与分层KV架构，过载拒绝是serving语义 | 亲和放置、CPU／DRAM／SSD缓存可借；若把early rejection用于训练准入，须另核任务混合／样本集合，不默认纯B。 |
| [Medusa v3](https://arxiv.org/abs/2401.10774v3) | T3；多头推测基础，不把两个训练版本合成无损保证 | Medusa1冻结骨干、Medusa2联合训练；迁到RL需分别核骨干目标、接受规则和版本适配，质量保持不替代精确分布契约。 |
| [EAGLE v3](https://arxiv.org/abs/2401.15077v3) | T3；原始feature-level draft，不混入EAGLE2／3 | 特征预测＋前移token处理不确定性可供drafter背景；RL policy持续变化后的接受率／适配成本由具体RL系统另证。 |
| [SuffixDecoding v3](https://arxiv.org/abs/2411.04975v3) | T3；重复agent推理的历史文本draft，不等RL训练回放 | suffix tree缓存prompt／旧输出并调推测长度；迁入RL应由当前target验证，不把命中文本直接当当前policy样本或旧KV。 |
| [Leviathan等 v2](https://arxiv.org/abs/2211.17192v2) | T3；精确推测采样的基础来源，RL细化见现有T1/T2 | 快draft与大target并行验证／修正是概率契约依据；保分布不代表同随机种子逐token相同，仍需核实际p/q与采样实现。 |
| [Chen等 v1](https://arxiv.org/abs/2302.01318v1) | T3；独立的modified rejection sampling来源，不与前篇去重 | 原文保留hardware numerics条件；迁入RL须匹配temperature／截断后的实际分布，不能把“精确”扩张为全栈bit-exact。 |
| [Aurora v5](https://arxiv.org/abs/2602.06932v5) | T3；RL用于在线训练speculator，非post-training actor rollout系统 | live traces、拒绝反馈与热换draft可启发适配；target是否同步更新、draft学习成本和训练目标须在RL场景另核。 |
| [Intelligent Router既有T2](#6293-intelligent-router) | 仅复核范围；不重读、不增T3或T2 | 同模型副本的长度／混负载路由可借；serving延迟不等rollout组makespan，仍需policy年龄与分布漂移条件。 |
| [CachedAttention ATC24](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | T3；多轮对话缓存分层，不追加模型更新系统T2 | 逐层预载／异步保存及scheduler-aware换入换出可借；解决context溢出的缓存有效性，不自动保证更新权重后的KV有效。 |
| [CONCUR v1](https://arxiv.org/abs/2601.22705v1) | T3；agent批推理拥塞控制，非RL采样配额管理 | 用cache反馈调active agents以保连续执行；迁入RL需区分延迟准入和永久丢弃，另核整组完成、task配额与staleness。 |
| [Continuum v7](https://arxiv.org/abs/2511.02230v7) | T3；工具暂停间KV TTL与program FCFS，非终止轨迹策略 | 按reload／排队成本留KV可减少跨turn重载；TTL到期只允许evict cache，不等abort rollout，跨policy更新的KV另核。 |
| [Helium v1](https://arxiv.org/abs/2603.16104v1) | T3；workflow query plan优化，非RL训练管线实证 | proactive caching／cache-aware调度可迁到调用DAG；若复用随机输出而非仅复用KV，须核组内采样独立性、版本和reward语义。 |

**22:55完成时的阅读账（本轮前快照，当前见§0／§6.31）**：121条矩阵当时为20 T1＋64 T2＋19 T3＋18未标完成；这些是互斥身份数，不把Router双计。18条中HybridFlow仍进行中1；原§5.20单列而未T2的ASymPO／CIS／Each Prompt Matters共3；另14条仍是直接候选，不能仅因批次表B1–B11完成就自动降为T3或称全文已读完。

22:55时14条直接候选未读清单：DistFlow、Asynchronous RLHF、ReaL、RLHFuse、OpenRLHF、DeepSpeed-Chat、PUZZLE、DynaRL、PivotRL、RobustRL、RLBoost、Relax、SPEC-RL、ProRL Agent。代码广扫已有OpenRLHF不等论文T2完成，Relax局部多模态补核也不等深读。B12其余边界与§5.23五条线索仍待实质判定；当时接续为先核远端与分级，将直接候选按6–8篇T2补批、在认领时回写名单，不以重复serving条目凑T3规模。HybridFlow与Code批次保持各自验收标准。

### 6.31 L4执行计划与异步补读

B6补六篇T2，按§6.0名单微调由9扩为15；各6条。正文与必要附录阅读不替代代码／性能验收。算术复算为本轮判断，条件与原始时间保留，不据单一汇总错误否定全部系统机制。

#### 6.31.1 DistFlow

- **来源／分类**：[v4 §4–7、附录A/B](https://arxiv.org/html/2507.13833v4)，T2完成；L4 B。分散控制／数据面，不等于policy异步更新。
- **执行契约**：DAG Planner为colocated资源将同深度节点加依赖串行化，DAG Worker本地执行；“DAG”不代表任意阶段都并行。
- **数据机制**：DP不变走本地中间张量cache，变化才经每节点Databuffer重分片，TP0提交；Constrained LPT按已知序列长度平衡训练侧且限制每worker等条数，非预测生成长度或丢弃尾样本；double buffer交换指针后后台释放。
- **条件**：Qwen2.5 7／32／72B、Hopper集群、vLLM0.8.5.post1／FSDP；默认prompt≤2048／response≤4096。主LLM至128卡，VLM至512卡同时增global batch，是弱扩展，不是固定job强扩展。
- **结果口径**：正文PPO对verl为1.09–1.64×、GRPO最高2.62×（摘要2.63×不混用）；20-epoch的32B／32卡reward与entropy接近、总时间减21%，不等严格采样等价证明。
- **负项／迁移**：小batch负载均衡收益可被调度开销抵消；附B 8K／G8／32卡step192.326→91.881秒，但rollout compute25.169→29.236秒反慢，收益主要不能归decode kernel。排除异步基线也不能证明所有异步算法必然错误。

#### 6.31.2 Asynchronous RLHF

- **来源／分类**：[v3 §3–5、附录A/D](https://arxiv.org/html/2410.18252v3)，T2完成；L4 B，off-policy及best/worst选样另 **C**，鲁棒性是实证而非无偏／on-policy保证。
- **机制**：Cleanba式用θ_i生成本批、训练前一批得到θ_i+1，分离生成／训练资源；N批由同旧policy生成再多次更新的off-policy扫描，与实际一步旧流水分开。
- **条件／基线**：TLDR用Pythia0.41／1／2.8B、batch512、256步、4 A100；同步HF生成对异步vLLM还换了backend，25% headline不全是异步重叠。
- **较同栈对照**：No Robots Llama3.1-8B、8 H100、1卡vLLM＋7卡训练，两侧均vLLM；100K episodes，报告耗时约减38%、GPT4o win-rate均57.2%，是此任务结果，不普遍承诺同质量。
- **选样／成本**：训练受限时K2→4，仅取最高／最低reward，LR与步数减半，2.5×包含配方改变；多次复用旧batch或扩K都可提高KL，不能称目标不变的纯B提速。
- **单位／负项**：GSM8K Rho1B表2为218→129分钟，即速度1.69×／时间减40.8%，不把“68% faster”写成耗时减68%；pass@1为52.2→52.6。附录实际异步step15.1秒高于理想12.9秒，仍有通信／同步开销。

#### 6.31.3 ReaL

- **来源／分类**：[v2 §4–8、附录A](https://arxiv.org/html/2406.14088v2)，T2完成；L4 B；同ID的ReaLHF→ReaL不加身份。
- **机制**：函数级DAG同时含数据／参数版本依赖，为各generation／inference／train选mesh和DP／TP／PP／微批数；权重按共享层／TP分区broadcast重分配，master只派控制信息，数据留GPU。
- **计划成本**：profile层级运算＋插值、通信按大小／带宽估计，MCMC时限内选最低估计成本；同构GPU／带宽和可预测调用是假设，原文明确不保全局最优。1024卡搜索不等1024卡实际RL训练。
- **条件**：主测16–128 H100随模型／batch增长；base batch512、context2048、prompt1024、8 PPO mini-batches。附录用随机prompt且强制生成满最大长度，排除了真实EOS／长尾，不能拿它验收实际agent分布。
- **结果口径**：warmup后20步吞吐最高3.58×对多系统，搜索对自身heuristic平均+54%、长context最高+81%；不是完整收敛墙钟，也不等每模型／配置同一增益。
- **迁移限制**：7B+7B actor／critic可重叠，70B+7B反而选择串行，重叠不是普遍最优；生成长度动态变化会伤成本预测。保持图依赖是语义设计，不替代RNG、训推数值或质量验证。

#### 6.31.4 RLHFuse

- **来源／分类**：[v3 §3–7](https://arxiv.org/html/2409.13221v3)，T2完成；L3/L4 B，Stage Fusion为同篇，不新增名称数。
- **跨阶段契约**：已完成样本先评分，将未完尾样本与KV聚到少数生成实例、释放卡作Ref／RW／Critic inference；训练仍等完整生成集随机分mini-batch，非够数就提前更新或丢尾。
- **阶段内机制**：actor／critic微批双向流水填彼此气泡；受依赖、无环和activation内存约束，以退火求近优，局部达到下界不证明任意配置最优。规划实验用了768 CPU核，非免费执行开销。
- **条件**：256 Hopper、LLaMA13／33／65B四种actor／critic组合、HH-RLHF，global512／mini64、输出上限512／1024／2048；DSChat为能运行改mini256，更新次数不再完全一致。
- **结果分账**：warmup后20步samples/s，对DSChat2.5–3.7×、ReaLHF1.4–2.4×，但对同栈无融合Base仅1.2–1.4×；3.7×不能全部归融合，更不是质量达标时间。
- **负项／迁移**：迁移太早可使少数尾实例过载，约20%阈值仅早期负载观察、随分布调整；主测用RDMA直接迁KV，不能外推慢网仍零成本，保同步依赖也不等逐位随机一致。

#### 6.31.5 OpenRLHF

- **来源／分类**：[v6 §3–5、附录C](https://arxiv.org/html/2405.11143v6)，T2完成；L2/L4 B，DAPO动态筛组另 **C**；论文不替代既有固定commit代码广扫。
- **机制**：Ray编排rollout与ZeRO engines、vLLM生成；AutoTP＋ZeRO／DP＋Ring序列并行为文中“3D”，不是默认等同经典TP／DP／PP；异步RPC描述不自动给每条轨迹版本／年龄硬界。
- **主对照条件**：OpenRLHF0.8.5对verl0.4.0，8 H200、PyTorch2.7，DeepSeek蒸馏Qwen1.5／7／14B，DAPO同超参、local batch1、prompt≤1024、输出1K–8K；计平均step、去掉前10步，不是全训练time-to-quality。
- **原表核对**：[PDF第5页表1](https://arxiv.org/pdf/2405.11143v6)视觉确认时间与HTML一致；按其图注 `exp(mean(log(t_verl/t_ours)))` 复算三行约1.146／1.181／1.393×，不能复现原列1.22／1.42／1.68×。这是汇总算术疑点，所有原始配置时间仍较短，不否定这些时间记录。
- **其他结果**：PPO 1024 prompts一epoch为855.09→236.8秒、约3.6×对DSChat；不合并为主长CoT实验倍率。两侧均DAPO也不能将全部系统差异归因DAPO算法本身，代码行数更不认证可维护性或故障安全。
- **迁移边界**：附录给rollout logp与PPO流程，未逐项交代主性能对照的队列／版本／组消费契约；不能由“fully async”推出零陈旧、精确on-policy或通用算法稳定性。

#### 6.31.6 DeepSpeed Chat

- **来源／分类**：[v1 §3–5](https://arxiv.org/html/2308.01320v1)，T2完成；L1/L2/L4 A/B；可选Mixture Training的预训练loss另 **C**，不把整个框架都算目标变化。
- **机制**：Hybrid Engine按train／eval模式切换ZeRO训练与TP生成，配合推理kernel、KV／中间内存管理与分区重配置；是分阶段统一资源，不是actor生成与更新永远同时进行。
- **条件**：2023报告的OPT系列、生成256 tokens＋prompt256、最大global batch1024；以可用内存选batch／GPU数。不能把短输出最大batch性能直接套给长CoT或固定batch强扩展。
- **结果分账**：13B actor＋350M reward在8 A100-40GB上，全三阶段13.6小时、其中Step3为10.8小时；图中6–19×对Colossal-AI及1.4–10.5×对HF DDP主要是Step3，不是SFT＋RM＋PPO全程或现代框架排名。
- **额外成本／限制**：EMA checkpoint与mixture均可增加内存／训练成本；报告的小时／历史云价不是同质量ChatGPT复现或当前账单。175B端因内存限制batch、有效吞吐下降，不是规模越大总更高效。
- **迁移边界**：超线性扩展同时来自更多内存允许更大batch；后继论文的DSChat版本、mini-batch修改和backend必须单列，不能将2023公开倍率移植到其他论文的对照设置。

**23:27完成时未读账（历史快照，当前见§0／§6.32）**：121＝20 T1＋70 T2＋19 T3＋12未标完成；12＝HybridFlow进行中1＋原单列边界3＋直接候选8。余八为PUZZLE、DynaRL、PivotRL、RobustRL、RLBoost、Relax、SPEC-RL、ProRL Agent；下轮核远端后合批T2并在认领时回写名单。B12边界三项及§5.23五条线索仍待，代码6路径／4框架、完整复现0不增；不将B6补齐或T2达到70称作Phase 2整体完成。

### 6.32 资源迁移 容错与采样边界

余八直接候选完成T2，七篇补B6、一篇补B8；每篇6–7条。仅正文／必要附录及两份会议PDF图表核对，未读实现或运行原生实验。SPEC-RL反例为本轮对论文伪代码的数学检查，不是实现缺陷定位。

#### 6.32.1 PUZZLE

- **来源／分类**：[ATC24正式PDF §4–7](https://www.usenix.org/system/files/atc24-lei.pdf)，T2完成，L4 B；历史PPO系统，不当现代长链基准。
- **机制**：无依赖模型微批时分重叠填pipeline空隙；训练／生成布局联合计入转换成本，分段optimizer更新与权重广播重叠。
- **条件**：8–32 A100、Megatron对DeepSpeed-Chat，prompt／generation各256；PDF第10页表2按模型组合取batch128／256，非恒定batch强扩展。
- **结果**：两集群样本吞吐平均1.83／1.50×；组合跨软件栈，不能全归单一时分优化或推为time-to-quality。
- **消融**：固定生成最优／训练最优布局对照平均1.22／1.34×；转换需通信，不因子秒开销写成零成本。
- **迁移边界**：可借鉴跨阶段布局与权重分块，短输出／离线profile及无依赖前提需重评；未核数值等价、代码或长链表现。

#### 6.32.2 DynaRL

- **来源／分类**：[OSDI26正式PDF §4–7](https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf)，T2完成，L3/L4 B；实现于RLinf之上，不当独立底座。
- **机制**：动态hypergraph汇总负载／亲和性；安全点迁移保模型与optimizer状态，候选分配按预测吞吐选；多轮按已完成工具轮数／前缀优先。
- **条件**：64／128 H100、1.5B–32B、batch512／group16、长度28672；预热后10迭代平均。PDF第11页确认同SGLang／Megatron基线及自实现RLHFuse。
- **结果**：128卡32B对verl为1.98×、对RLinf仅1.40×；agentic含优先调度范围1.27–1.64×，不混两任务指标。
- **正确性**：迁移保持global batch并重分微批；1.5B奖励曲线近似不等逐值一致或任意异步更新等价。
- **迁移边界**：收益依赖可中断点、状态搬移与轮长特征；不由统一接口推所有组件零成本恢复，未审源码或故障实测。

#### 6.32.3 PivotRL

- **来源／分类**：[v1 §5–8、附录C](https://arxiv.org/html/2610.16866v1)，T2完成，L3/L4 B；含GenRM的三角色场景。
- **机制**：CPU分布式GPS供权重；optimizer等offload、轻context保留；rollout／reward按replica转移，trainer整体转移。
- **迁移／规划**：保留已生成token、重prefill而非迁移KV；共享队列重派，planner比较预期收益减转移代价，默认窗口60s。内部模拟不等主实验模拟。
- **条件／结果**：128 H20、四组policy／GenRM、DAPO数学＋生成式奖励；对同veRL栈共置1.81–2.05×、拆分1.50–1.95×整步token吞吐。
- **开销**：大模型shift-up可约9s，序列迁移表内总均值至1.18s但有方差；不能把并行planner视为没有成本。
- **边界**：奖励／AIME随step近似、墙钟更快非bit-exact；不外推无GenRM任务，未核源码／复现。

#### 6.32.4 RobustRL

- **来源／分类**：[v1 §4–7](https://arxiv.org/html/2512.22492v1)，T2完成，L4 B容错；核心是故障损失，不是稳态decode提速。
- **机制**：trainer故障重启trainer组并保rollout；同机房同构rollout可作备用，UCX权重relay动态重连；保存到host与持久化时间分开。
- **失效域**：CPU AgentWorker／RequestManager等不获容错；首次／重复错误仍整任务重启。rollout保存到工具轮边界，不承诺零token损失。
- **条件**：主端到端128 H20、Qwen3 8B／32B、GRPO、100步；每10步区间随机kill trainers，ByteRobust仅保留原地重启比较。
- **结果**：此故障设置下重启效率1.5–1.7×；ETTR把重做rollout也计有效计算，不能直接当有用样本吞吐或真实故障率收益。
- **边界**：训练趋势相似非确定性复现；warm standby不适用于同步共置无独立rollout情形，未审源码／部署或新增算力任务。

#### 6.32.5 RLBoost

- **来源／分类**：[v3 §4–6、附录A/C](https://arxiv.org/html/2510.19225v3)，T2完成，L3/L4 B；延续§5.21范围，不另计身份。
- **机制**：固定trainer＋弹性rollout；本地限时partial seeding、完整响应动态微批累积梯度；token流保前缀、抢占后重prefill续跑而非直接丢请求。
- **版本条件**：专用pull权重agent，仅向已装最新权重实例派单；前后端网络不争用为其传输假设，不认证随机数逐值重现。
- **评估**：8B–32B H100；三个2小时真实可用性trace在按需实例回放，非真实spot训练账单。价格采用跨地区／云厂商均值估算。
- **结果**：相对只用reserved训练卡的veRL，平均吞吐1.66／1.97／1.51×；借额外卡，非固定卡等资源增益。Disagg.BAL为作者实现而非原版StreamRL。
- **边界**：权重到齐前不派单、prefill重算仍耗时；请求迁移不保护trainer／manager故障，未核源码或真实性能复现。

#### 6.32.6 Relax

- **来源／分类**：[v2 §3–5及附录B](https://arxiv.org/html/2604.11554v2)，T2完成，L3/L4 B；异步采样语义不能仅由服务解耦认证。
- **机制**：外部TQ字段／microbatch就绪流水、producer陈旧性门控；encoder置PP0、ViT在TP组复制；DCS独立同步权重。
- **恢复边界**：Critic／Advantages可原位恢复；Actor／Rollout／ActorFwd涉及权重或通信组时全局重启，不是所有角色独立恢复。
- **结果／条件**：16 H800、4B／DAPO同栈对veRL为150.5→125.6s（1.20×）；正文设off-policy，与结论on-policy用语冲突仍保留。
- **模式不可混算**：同文模式比较对colocate为1.76×；16 H20的30B omni为2.00×。关键路径rollout为0指被隐藏，非零GPU成本；奖励曲线不等分布证明。
- **疑点／未验证**：§3.3写`s=v_t−v_r`且仅`s<max_staleness`生成，却实验称阈值0为on-policy；不据此还原实现边界。无源码／故障验收。

#### 6.32.7 SPEC-RL

- **来源／分类**：[v3 Algorithm 1、§3–4、附录A/B](https://arxiv.org/html/2509.23232v3)，T2完成，L1/L2由暂A改A/C；旧轨迹作draft不自动保证采样无偏。
- **机制**：整批teacher-force旧响应，逐token按`min(1,ℓ·q/p)`接受，首拒绝后从当前policy继续；复用从第二epoch起，GRPO／PPO／DAPO默认ℓ分别为`e^0.5／e^0.3／e^0.15`。
- **采样疑点（本轮推导）**：Algorithm 1第9行直接从q重采，未给残差分布修正。即ℓ=1、单token旧p=(0.8,0.2)、新q=(0.2,0.8)，接受质量为(0.2,0.2)，拒绝概率0.6，输出为(0.32,0.68)≠q；故该伪代码不足支撑精确on-policy声明，不据此判定未读实现有同样缺陷。
- **条件**：8 H100、batch1024、response4096；DeepMath-6K多epoch，重复prompt条件不适用于一次遍历无复用。
- **计时**：8B GRPO rollout为2.51×，全步1266.4→841.0s仅约1.51×，含74.7s验证；减少decoded tokens非等量减少全部forward。
- **负项**：高lenience更快但1.7B平均分下降；8B PPO亦57.3→57.2。不以有限基准近似维持质量替代分布保证。
- **边界**：C标记采样变化而非reward／actor公式变化；数学反例非GPU复现或源码审计，无对外报告／修复动作。

#### 6.32.8 ProRL Agent

- **来源／分类**：[v1 §3–4](https://arxiv.org/html/2603.18815v1)，T2完成，L3/L4 B/C；HTTP服务／rootless sandbox与选样分别归因。
- **机制**：INIT／RUN／EVAL独立池，任务亲和路由、token IDs及logp透传，避免重tokenize；API换backend不证明在途轨迹单版本。
- **样本语义**：DAPO过滤零方差组、错误rollout排出梯度；足数取消与未完跨轮保留同时出现，不能推每个任务exactly-once或集合不变。
- **评估**：训练用32 H100；吞吐消融另用8 H100／14B，完整0.37 instance/s，无LB 0.25、无高效Bash 0.29、无陈旧清理0.30，非整RL加速比。
- **质量边界**：SWE Verified分数提升是recipe训练结果，跨报告模型对照不单独隔离基础设施效应；节点扩展不当同资源效率提高。
- **恢复／未验证**：异常回调返回结构化结果、取消释放资源不等持久化恢复保证；计数器文字有任务／调用口径差异，未读实现或运行sandbox。

**本批后账**：121＝20 T1＋78 T2＋19 T3＋4未标完成；B6为22/22、B8为13/13，其余既有T2批次不变。4＝HybridFlow进行中1＋原单列边界3；§5.23另5线索不入121。当前矩阵直接候选无未标完成项，但原生覆盖／检索、HybridFlow、B12及Code整体验收未完成；代码6路径／4框架、完整复现0，不因论文数超目标就称Phase 2结束。

### 6.33 恢复 前缀与概率边界

B13六篇完成T2，每篇7条；阅读固定版本HTML的方法、实验及必要附录，未审源码／核PDF图像／运行原生实验。§5.25保留当时triage快照，三项T3升级后只计T2；ScaleRLT v1 HTML本轮可读，不再沿用旧cache miss作为全文阻塞。

#### 6.33.1 Belayer

- **来源／分类**：[v2 §4–8](https://arxiv.org/html/2608.14635v2)，T2完成，L4 B容错；减少故障损失，不等稳态decode提速。
- **恢复机制**：独立server持有权重／KV空间，shadow重建易失状态并隔离旧worker；仅owner存活、GPU健康时复用，清空语义KV后按256-token日志前缀重prefill。
- **故障范围**：worker局部fail-stop／stall，不含SDC／fail-slow；环境只保容器本地FS＋CRIU runtime及合法上下文前缀，不保外部副作用exactly-once。
- **环境协议**：动作结束后冻结并制作复合checkpoint，两部分成功才ready；恢复先FS后runtime。自适应频率权衡故障重做与未遮蔽开销，不是零成本快照。
- **条件／结果**：Qwen3 4B–32B、16／32 H200、GRPO batch64×8；注入worker故障后训练时间增1.16%对基线8.72%，无故障Math约1%差异。
- **证据强度**：17个目标bug仅复现8个，八项确定性文本恢复匹配；环境593个有效合成故障匹配oracle，不证明任意agent语义。频繁故障扩展另含LLM sleep模拟。
- **边界**：HDD自适应仍有21.6%额外总时间，SSD为2.1%；非所有存储均近零。论文结果非本机实测，未部署Linux容器或算力任务。

#### 6.33.2 BIRD-RL

- **来源／分类**：[v1 §3–5、附录C/D.2](https://arxiv.org/html/2610.29426v1)，T2完成，L4 B基础设施＋C课程；两类贡献分别核。
- **机制**：worker独占副本子池；trajectory ID始终绑定同一DB直到终态，超时回收。脏副本释放时不立刻reset，下一次租用前恢复，隔离轨迹而不切断轮间状态。
- **训练语义**：先两阶段SFT再单轮／多轮GRPO，最后方案的执行测试奖励；课程和reward设计不能归因于纯资源复用。
- **条件**：7B／14B Qwen2.5-Coder、8 H100、veRL＋vLLM；SQLite调试500题、五次评估。跨数据库基准与SQLite训练基础设施不混为同一验证。
- **系统指标**：B=16、R=16时在线生命周期I/O降12.8×；排除69 GiB离线预建副本成本，只有9.25%轨迹触发改写重置，不是TPS或全训练提速。
- **质量指标**：Talon 7B／14B为44.4%／48.0%成功率；是完整课程结果，不单独隔离数据库池效果。
- **边界**：动态池复杂度有均匀／稳态假设，实际用预建池抗热点；足够容量是所测负载经验，不能推任意倾斜不耗尽。未审实现或运行DB。

#### 6.33.3 ScaleRLT

- **来源／分类**：[v1 §2–7、§10、附录B/C](https://arxiv.org/html/2610.34427v1)，T2完成，L1/L2/L4 A/B；相对原生模型的混合深度读出另C，非只调度同一policy。
- **机制**：Ouro-1.4B先两loop、每四token深读出；浅token的深层工作延后批处理，几乎最终仍要计算，非直接省掉一半层。
- **概率契约**：`ℓ=dense+sg(schedule−dense)`取调度replay的值、dense的梯度；same-weight gate可bitwise，梯度仅同精度误差级，非精确梯度等价。每batch一步，额外步须重算replay。
- **条件／结果**：GH200 FP32、512–4096长度，同混合读出defer/eager两侧均graph时1.18–1.54×；双节点8卡、512上限的准入＋graph组合使update为2.71×。
- **质量负项**：100更新、500题、每设置单种子；混合读出落后原生7.6–9个百分点，原生20步已胜混合100步，不以更快update代替time-to-quality。
- **数值负项**：512-token BF16 gated／ungated质量未分离；exact gate此处买一致性，不已证提质。graph捕获、allocator残留和replay都有成本。
- **边界**：跨节点ring只是通信实测支撑的设计，未实现完整mapped engine；所测概率一致不认证任意batch／precision／训练路径，无源码或设备验收。

#### 6.33.4 PrefixFlow

- **来源／分类**：[v4 §3–6](https://arxiv.org/html/2606.01143v4)，T2完成，L4训练侧B边界；不是生成KV cache优化。
- **机制**：prefix一次forward；suffix微批读KV、累积gK/gV；最后一次prefix backward。hot KV/gKV留GPU，休眠激活可offload后用于suffix预算。
- **正确性条件**：固定参数、mask、位置及prefix前向trace，MoE须固定路由；实数梯度叠加证明不等浮点bitwise。20步BF16校验使用绝对1e-3＋相对1e-2容差。
- **分布式**：CP逐层KV预取／gKV归约、PP局部切阶段，避免全局屏障；offload可退回按需加载，不以完全遮蔽作为正确性前提。
- **闭环结果**：8 H20、Qwen3-8B、MuSiQue、N=16、prefix约95%、200步；actor更新6.1×，全步1.90×，生成和policy evaluation未提速。
- **负项／适用域**：低复用N=2 compiled可更慢；数学／SWE短prefix不套长RAG收益。采样轨迹可种子级分叉，冻结trace核trainer偏移不等奖励逐值一致。
- **边界**：全序列同栈TorchTitan基线；不作跨成熟度Megatron计时排名。内存释放、长度上限与吞吐分账，未应用缓存或运行训练。

#### 6.33.5 psRL

- **来源／分类**：[v1 §4–7](https://arxiv.org/html/2608.25683v1)，T2完成，L4训练侧B边界；不把静态样本update优化转成生成加速。
- **机制**：完整semantic group按去重后工作量分配worker；跨微批共享、长序列按token切批。收集共享prefix的全部consumer梯度后只反向一次。
- **内存／语义**：可变块贴合复用段、引用计数及已知未来访问指导eviction；attention／advantage mask保留序列依赖和loss归属。不同权重／optimizer步间缓存失效协议尚未由源码核验。
- **条件**：veRL＋Megatron、A100；标准任务8卡，工业DTN用Qwen3-235B及生产工作负载，主配1536卡。扩到12288卡同时提高DP，不视为固定工作量强扩展。
- **指标拆分**：Fig.9是update吞吐1.2–5.2×，Fig.10整RL迭代至2.1×；虽正文局部称end-to-end，不能写成全RL 5.2×。
- **基线边界**：vLLM-PS／SGLang-PS是作者移植到训练engine的共享机制，不是原生serving框架的公平性能排名；长序列扩展插入dummy token，不代表真实语义长任务。
- **未验证**：文中工业扩展与mask正确性自述不替代公开原始trace、逐梯度对照或质量收敛证据；本轮仅文献，不读取真实生产数据／设备。

#### 6.33.6 CIS

- **来源／分类**：[v1 §2–4、附录B/C/D.4](https://arxiv.org/html/2609.32444v1)，T2完成，L1/L4算法边界；不是rollout系统提速。
- **目标边界**：`p/q`为旧权重下train／infer失配，PPO另有current／old更新ratio；只校正已观测prefix下next-token分布，不修整条prefix分布或任意版本陈旧性。
- **机制**：仅截正向log-odds位移，对应`min(p/q,1+λ·max(1−p,κ))`；旧train概率定confidence，权重detach、infer须取sampling原始logp，不用加工后概率代替。
- **理论条件**：有界梯度payload及独立token／受控协方差；相关token用有效样本数。有限cap控二阶矩但引入截断偏差，不证明CIS普遍优于所有cap。
- **条件／结果**：三MoE、GSM8K训练、五数学测试、三种子；Qwen1.5平均34.78对无修正30.99，部分跨域差距仍在种子波动内。
- **数值／成本**：κ=0.005防高置信度舍入，无floor降到29.02；八A100示例每步总时间增3.2%，不含checkpoint／eval。算子本身逐元素，不等整流程无需rescore。
- **边界**：floor端放松理想位移阈值，λ=0可崩溃；有限设置稳定不等exact训练或普适超参。只读论文，未接续AReaL prompt_mean、修改代码或增加算力队列。

**本批后账**：126身份、120唯一arXiv＋6会议不增；20 T1＋84 T2＋21 T3＋1未标，后三项由T3替换为T2，不双计。B13为6/6，HybridFlow仍进行中；代码7路径／4框架、完整复现0。下一批C2 scheduler 1–2路径，再C3；不因T2超目标称Phase 2完成。

## 7. Code & System Landscape

**本地已有资产**（重要：不用重新 clone）：
- `/Users/jackiemac/workspace/verl`：veRL 源码（含 #8281 工作分支，近期深读过 `experimental/agent_loop/` 全链）；
- `/Users/jackiemac/workspace/UniRL-main`：UniRL 源码（train/stack 双栈结构已深读）；
- M5 服务器：verl-e2e 环境（4090-1，可跑 rollout 冒烟）。

veRL三条地图使用的只读检出：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/verl-7913`，HEAD固定`24f25b03aa4b54249a273655ebbcce06f484192b`，origin为`verl-project/verl`，核对时工作树干净。此快照不称最新main，不与上方历史工作区／论文发表年代混为一版；文件夹名不是接续旧PR的授权。

OpenRLHF本轮只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/openrlhf-rollout-scan`，origin为`OpenRLHF/OpenRLHF`，固定HEAD `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53`，工作树干净；未安装依赖或运行项目。两框架快照分别标版本，不把实现差异写成同年代／同配置的性能对比。

NeMo RL只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/nemo-rl-rollout-scan`，origin为`NVIDIA-NeMo/RL`，固定HEAD `a9d1a4bec45a28fd0d455519a2a979a37da2ac4a`，工作树干净。这里是SingleController路径，不与legacy async GRPO、NeMo-Aligner论文或其他快照拼为同一实现；未安装依赖或运行项目。

SkyRL只读克隆：`/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/skyrl-rollout-scan`，origin为`NovaSky-AI/SkyRL`，固定HEAD `e0893f6eb878f4d396184cb14c4f9d27bbbb0aef`，工作树干净。只读统一包的fully-async／SkyRLGym路径，不同于SkyRL-Agent论文原版、默认同步trainer或模拟trainer；未安装依赖或运行项目。

**代码地图覆盖**：累计7条／4框架：veRL原三条见§8.2–8.4，V1 TransferQueue数据面第四条见§8.8；OpenRLHF／NeMo RL／SkyRL各一条见§8.5–8.7。§8.8另核默认V1入口，不将旧RayPPOTrainer或experimental fully-async路径当所有配置的默认。AReaL prompt_mean仍由M5单一负责，不在此代码阅读中接续；7条不等整个框架读完或性能验收。

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

### 8.8 veRL V1 TransferQueue数据面

固定[veRL `24f25b03`](https://github.com/verl-project/verl/tree/24f25b03aa4b54249a273655ebbcce06f484192b)，通过已有`work/verl-8250`仓库的Git对象只读取得该版本，不切换其HEAD或改文件。范围是一条V1的数据提交→元数据选批→worker取数／回写→清理路径；TQ独立项目的存储、事务与容错实现不在本次范围。

**入口修正**：[ppo_trainer.yaml:228](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/config/ppo_trainer.yaml#L228)默认`use_v1=true`、`trainer_mode=sync`；[main_ppo.py:183](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/main_ppo.py#L183)据此前往TaskRunnerV1，否则才转legacy main_ppo_v0。故§8.2–8.4对旧RayPPOTrainer／experimental fully-async分支的分析仍属于那些分支，不能用其入口概括当前快照默认工作流。基础TQ YAML虽`enable=false`，V1 [run:133](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/main_ppo.py#L133)设true并`tq.init`，结束finally调用close；默认配置与执行时配置分开。

| 环节 | 固定入口 | 数据与控制的区别 |
|---|---|---|
| 提交prompt | [trainer_base:1403](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L1403) | 先登记prompt uid／pending／global_steps，再调用manager；非sync另保存prompt字段供checkpoint恢复。 |
| 生成完成与组完成 | [agent_loop_tq:59](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/agent_loop_tq.py#L59)、[_run_prompt:107](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/agent_loop_tq.py#L107) | worker创建后台prompt任务；manager的ray.get仅等任务派发。每prompt内n个session全部settle后才写finished／failure；单条trajectory入库不等整prompt可训练。 |
| 写trajectory | [postprocess:150](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/agent_loop_tq.py#L150) | key为uid_session_index；写input_ids／position_ids、loss_mask等字段及长度／版本标签；去掉raw多模态数据，不代表不存处理后的多模态张量。 |
| 选批与所有权 | [replay_buffer:405](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/replay_buffer.py#L405)、[materialize:377](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/replay_buffer.py#L377) | 所读sync sample先轮询终态／逐出与足数组，再返回KVBatchMeta；选中时清prompt控制key，trajectory仍留存。启用DAPO时排空在途并丢弃超额组，非TQ无损承诺。 |
| 元数据分派与合并 | [BatchData.chunk:1268](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/protocol.py#L1268)、[concat:1288](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/protocol.py#L1288) | KVBatchMeta先解析为BatchMeta再chunk，减少每rank控制面往返；合并BatchMeta后重新构建KV引用，不等张量都经过trainer聚合。 |
| worker桥接 | [register:398](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/single_controller/base/decorator.py#L398)、[tqbridge:347](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L347) | register自动包桥；无元数据直接调用。有元数据时取实际TensorDict→执行worker→按collect规则写回并返回引用，row数assert不验证行语义或policy版本。 |
| old概率与更新 | [old_log_prob:1541](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L1541)、[update_actor:1734](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L1734) | bypass把rollout_log_probs复制到old字段；否则worker重算再回写。actor读引用执行train_mini_batch，TQ不赋予概率来源新的数学保证。 |
| 回收 | [trainer_base:488](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L488) | 正常迭代末清已训练trajectory keys，与选中时清prompt key不同；此调用链没有证明训练成功与清理之间具有事务／exactly-once恢复。 |

**字段与元信息不是同一通道。** [读取:160](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L160)调用async_get_data后，把深拷贝extra_info附回TensorDict；[写回:184](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L184)把Tensor／NonTensorStack作为数据字段，NonTensorData转入extra_info，且仅有数据字段时才put并替换extra_info。不能承诺任意metadata-only输出也会持久化。非收集rank的检查发生在worker执行之后，因此不写回不等未执行计算。

KV接口的“key-value”不是attention K/V。[KV→Batch:302](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L302)按keys／partition取meta、`create=false`，可筛fields；[Batch→KV:323](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L323)要求单partition并构建空tags。直接KV进入tqbridge时会暂存／还原tags，但chunk／concat经过BatchMeta的分支不能据此声称原tags自动完整保留；不把transport metadata当逐token版本证明或样本筛选策略。

**异步能力不决定更新粒度。** [step:511与_step_once:540](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L511)按parameter_sync_step分子批，每次仍依序sample→reward→balance→概率→advantage→update。TQ支持字段读写、生成任务可并发，不表示每个完成token或trajectory立即optimizer.step；sync／separate_async／colocate_async还需分别核其调度hooks与sampler，不能用本条路径覆盖全部模式。

同步桥的[_run_async_in_temp_loop:137](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/utils/transferqueue_utils.py#L137)实际复用后台event loop并取future.result，调用者仍阻塞；async桥才await。源码配置默认SimpleStorage且Mooncake为实验性可选项，不把接口读写称无拷贝／RDMA或低延迟实测；worker计算logp／update的所读返回路径还显式`.cpu()`，不是全程GPU驻留证明。

验证限度：固定Git blob的10份Python通过Python 3.12.14 AST解析，23个函数行号复核；另读2份YAML及文档相关段落。仅提取原样的4个stdlib异步桥函数，五组CPU检查通过：返回值、100次复用、已有event loop调用、异常传播、关闭线程／loop。系统Python 3.9及旧web venv因match语法无法解析replay_buffer，改用已有3.12后通过，不是源码缺陷。未导入verl／torch／Ray／TQ、运行原生pytest、Linux FD泄漏测试、存储／故障／性能／设备实验。新计代码路径1条而非10条；累计7路径／4框架，完整复现0；无M5专项或新增算力交接。

### 8.9 vLLM token预算与抢占

固定[vLLM `eaf1627c01f553ea9a26b3d77f3cff8495dbf404`](https://github.com/vllm-project/vllm/tree/eaf1627c01f553ea9a26b3d77f3cff8495dbf404)，独立稀疏检出`work/vllm-c2-readonly`。一条路径覆盖`Scheduler.schedule`→KV准入→抢占／恢复→输出计数回滚，不是全引擎源码验收。L2 B为主；引擎内调度不是RL训练侧异步更新。

- **预算不是一个batch数。** [schedule:688](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L688)用computed追赶已知token＋spec＋placeholder。scheduled-token预算与输入预算分开，后者逐请求还预留draft slots；同时受模型长度、encoder预算与Mamba／MTP边界约束。异步placeholder不等已采样token。
- **运行中先于等待。** [running循环:783](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L783)先排已有请求；encoder等条件导致零token时可跳过而非阻塞队首，不是严格FCFS。发生抢占后[本步不再纳新等待请求:1039](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L1039)；KV持有等待队列又先于普通waiting。局部prefill cadence有容量饱和逃逸，不是按固定间隔永远拒绝新prefill。
- **KV准入与计算预算分开。** [allocate_slots:380](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/kv_cache_manager.py#L380)计local／external hit、新计算及lookahead；waiting／preempted在已有运行请求时才加watermark。默认[full ISL gate:190](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/config/scheduler.py#L190)检查当前已知输入序列能否容纳，不保证未来`max_tokens`全部预留。滑窗旧块按已处理而非乐观推进的边界回收。
- **抢占先释放再重排。** 分配失败时FCFS选running尾，priority选最大`(priority,arrival_time)`；若victim本步已排，退还计算／encoder预算。[preempt:1773](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L1773)清computed／spec、保已生成token并prepend waiting；重入按完整已知序列含输出重算，允许prefix hit，不是把物理KV原封不动保留。
- **计算进度是乐观账。** [after_schedule:1819](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L1819)先加computed与in-flight；[输出处理:2233](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L2233)再减在途和拒绝draft。抢占后stale output可能交付，也可能因reset／connector条件丢弃；它不应再回滚已重置的计数。“stale”在此是引擎流水状态，不等RL policy staleness。
- **资源释放有fence。** [free guard:3048](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L3048)在异步connector可能仍写块时延迟回收；不能由一次抢占调用推KV已立即可用。[prefix hit:266](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/kv_cache_manager.py#L266)默认最多命中`num_tokens−1`以重算logits；cache hit率不等零prefill。
- **异步与换权重边界。** [AsyncScheduler:24](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/async_scheduler.py#L24)预留输出placeholder并限制PP下一次decode时机，不决定RL整组消费／optimizer粒度。[reset_prefix_cache:3129](https://github.com/vllm-project/vllm/blob/eaf1627c01f553ea9a26b3d77f3cff8495dbf404/vllm/v1/core/sched/scheduler.py#L3129)可选择抢占运行请求，仍可能因remote KV在途失败；换权重后的缓存失效和轨迹版本需调用方协议，不能仅凭scheduler认证。

迁移到RL应记录每组最后完成时间、准入等待、重算token、抢占／abort、实际输出token及版本，而不只记录scheduled tokens/s。固定RL工作集合下重排可作为B；若调用者过滤异常、丢弃慢组或跨版本续跑，则须另审集合／概率语义，不能由引擎排队规则证明on-policy。

### 8.10 SGLang prefill准入与decode回撤

固定[SGLang `cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1`](https://github.com/sgl-project/sglang/tree/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1)，独立稀疏检出`work/sglang-c2-readonly`，不切换已有六月checkout。一条路径覆盖事件循环→等待排序→PrefillAdder预算→decode回撤／重入；PD／beam／hybrid仅核与该链相接的分支，不扩大为各后端全读。

- **CPU/GPU overlap不是RL async。** [event_loop_overlap:2061](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L2061)先规划／launch、保存batch副本，再消费前一结果；有批次会关闭overlap。它不保证逐轨迹optimizer更新、整组终态或单policy版本。
- **prefill与decode有不同准入。** [get_next_batch:3894](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L3894)先过滤终态、处理chunk，再尝试prefill，否则decode；[raw prefill:4079](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L4079)同时受request行槽、PP微批、LoRA、cache及计算预算限制，chunk continuation可绕过部分纳新门控，不代表新请求可无限进入。
- **排序目标不是RL组尾。** [SchedulePolicy:213](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_policy.py#L213)有FCFS／LPM／DFS／HRRN等；LPM／HRRN大队列可退FCFS。实验性shortest-prefill-first按未缓存输入工作排序，不预测最终输出长度；不能直接当RollPacker式组makespan策略。
- **计算预算与显存预算不同。** [PrefillAdder:608](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_policy.py#L608)扣混合decode行、估计已有请求输出余量；[_update_prefill_budget:895](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_policy.py#L895)把replay算入计算charge，KV按页取整。单池[PrefillBudget:62](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/mem_cache/prefill_budget.py#L62)动态查询可用＋可逐出容量，不等仅看free显存；hybrid池另有约束。
- **logprob影响批形状。** [mixed chunk条件:4377](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L4377)要求prefill与running均不返回logprob，且无input_embeds／beam；decode余量不够时该步让其等待，不在mixed分支直接回撤。因此serving默认计时不能不带配置迁到RL采样。
- **回撤不总是无损续跑。** [update_running_batch:4459](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L4459)在decode容量不足时回撤并重排；[retract_decode:3451](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_batch.py#L3451)对beam组、PD无备份或最后请求仍放不下可abort，不是“始终至少保一请求”。长度策略优先回撤已输出少、同输出下prompt长者；配置priority另分支。
- **重算、备份与重启分开。** [release_req:2360](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_batch.py#L2360)在decode disaggregation可先备份；[reset_for_retract:2049](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/schedule_batch.py#L2049)清prefix／KV进度，普通token请求保output重入，但input_embeds分支清output并重新生成。该差异不应隐藏在统一“retract”计数里。
- **版本记录不等缓存失效。** [record_weight_version_change:5636](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L5636)记录活请求版本事件，函数本身不flush；[flush_cache:5518](https://github.com/sgl-project/sglang/blob/cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1/python/sglang/srt/managers/scheduler.py#L5518)有idle条件。不能由版本metadata存在推单轨迹单版本或缓存自动适配新权重。

**验证与计数**：[复跑入口及结果](../experiments/rollout-scheduler-controls/readme.md)固定上述OID和九份源码，Python 3.12.14 AST通过；八组原函数体控制检查MTP chunk边界（8700组合）、抢占计数、乐观进度、释放fence、回撤排序、单池预算、未缓存prefill排序与ratio估计。请求／队列／cache／capacity为替身，不导入框架，不执行原生pytest、真实分配／kernel、并发／故障／性能／设备；不称修复验证。C2新增两路径，累计9路径／6系统＝4 RL框架＋2推理引擎，完整复现0；与既有RL框架固定版本并非自动兼容组合。下一轮C3 rLLM／slime，HybridFlow原版映射仍待；无新增M5算力任务。

### 8.11 rLLM 组消费与同步窗口

固定[rLLM `4e828c81680e5a85b638ceedeaf09d1e90109ca8`](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8)，独立稀疏只读检出`work/rllm-c3-readonly`。一条路径为unified trainer→TrajectoryGroupBuffer→SyncCoordinator→权重发布；不是所有rLLM入口或后端实现的认证。当前`trainer/agent_trainer.py`仅veRL包装，不能用旧Tinker笔记概括此统一入口。主要L4 B，过滤／删尾及跨版本语义另审C。

- **async函数不等fully-async。** [_fit_fully_async:585](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/unified_trainer.py#L585)仅在显式开启async配置后选用，默认enable为false。[AsyncTrainingConfig:75](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/algorithms/config.py#L75)还区分sync每步／每K步及partial；staleness=0且K>1仍标为stream off-policy，不能把“零阈值”普遍写成严格on-policy。partial的dataclass与YAML默认值不同，应记录实际解析配置。
- **派发单位是prompt组。** [_generation_loop:629](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/unified_trainer.py#L629)每prompt记一次dispatch，创建n个episode任务。[coordinator配额:17](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/sync_coordinator.py#L17)为`int((1+threshold)*sync_steps*mini_batch_size)`，不乘group_size；这不是n个独立请求各占一份，也不是任意token年龄的硬上限。
- **齐组之后仍可能零入队。** [add_episode:166](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/buffer.py#L166)凑满同task_id的n个episode后transform、剔除不满足最小组的项、算advantage并可过滤全零组。一个TaskBatch可含多个转换后的TrajectoryGroup；空TaskBatch记filtered而不入队。原始episode指标在过滤前记，不等实际训练样本量。
- **消费不释放同步窗口配额。** [on_group_consumed:65](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/sync_coordinator.py#L65)只减in-flight；filtered才同时退配额并恢复派发。[on_sync_complete:93](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/sync_coordinator.py#L93)版本加一并将仍在途组结转为下一窗口已用配额。它不是简单“队列取走即可补一个”的bounded queue。
- **出队早于optimizer成功。** [_training_loop:669](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/unified_trainer.py#L669)按TaskBatch凑mini-batch，get后即记consumed，再分fwd/bwd块；不足完整mini-batch的尾部直接结束而不执行该次update。buffer的[get:282](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/buffer.py#L282)消费计数不等optimizer提交计数；[generation_complete:314](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/buffer.py#L314)会清理不足n的pending组并放sentinel，不保证所有生成任务最终训练或持久化exactly-once。
- **版本标签不认证物理原子换权重。** [_perform_weight_sync:812](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/unified_trainer.py#L812)非partial先pause／drain，调用backend发布，再设rollout engine／gateway版本与coordinator版本；partial分支不在此等待整个workflow排空。[get_model_response:87](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/engine/rollout/rollout_engine.py#L87)在await前捕获版本，[组版本:414](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/buffer.py#L414)取已观察step版本最小值、缺省0；两者是metadata规则，不证明每token同policy或backend无中间态。

性能记录需分prompt dispatch、episode完成、有效TaskBatch、出队、成功update和已发布版本。过滤、转换粒度、尾批与partial配置必须随吞吐一起报告；本轮未测训练吞吐、概率一致性或物理同步。

### 8.12 Slime 默认组生成与训练交界

固定[Slime `540ea919e45b8b929198abce008a103be9cfc779`](https://github.com/THUDM/slime/tree/540ea919e45b8b929198abce008a103be9cfc779)，独立稀疏只读检出`work/slime-c3-readonly`。一条路径为默认`train.py`→RolloutManager→sglang_rollout组生成→BatchBuilder；非整套fully_async／straw后端验收。L4 B为主，动态过滤、过量取消及partial旧token mask另记C边界。

- **默认驱动仍有阶段顺序。** [rollout-function默认:377](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/arguments.py#L377)指向sglang_rollout；flush-cache-interval默认1，非1受分离GPU等约束并切换另一fully_async函数，不将该分支冒充本轮已读。[train:13](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/train.py#L13)先装恢复权重再training_ready，循环generate→等待训练结束→training_completed→pause admission／update_weights／resume；Ray方法名async_train不等逐trajectory更新。
- **目标数量按prompt组而非token。** [generate_rollout_async:492](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L492)按rollout_batch_size收组，每组n_samples；过量取任务，FIRST_COMPLETED逐批收结果，动态过滤后补组。收够目标即abort余量，不是所有派发轨迹无损进入训练，也不能凭完成优先策略推等价于固定工作集合。
- **组内gather与reward有边界。** [generate_and_rm_group:388](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L388)组内异步生成、gather保顺序；已完成且有reward的分段重试可跳过再生成，group-RM另在gather后执行。[filter判定:17](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/filter_hub/base_types.py#L17)还可配置不足目标时保留被拒组，不是所有keep=False都无条件丢弃。
- **取消区分所属者与回收资格。** [abort:442](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L442)取消本地登记任务，必要时请求server abort并等余任务；非partial丢弃余组，partial仅保留至少一个ABORTED且已有response的组。[_run_request_abortable_generate:258](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L258)只将由abort移出登记集的取消转ABORTED，其他CancelledError继续抛；server active计数finally回落。这不是通用吞异常或所有慢组必恢复。
- **续跑保token不等全token参与loss。** [generate_and_rm:289](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L289)在partial＋mask_offpolicy配置下把已有response的loss_mask置0，随后续生成；生成本身[generate:158](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L158)向server索logprob并追加返回token／logprob。mask不能独立证明概率修正精确或整轨迹同版本。wrapper将回收组交data_source，[add_samples:221](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/data_source.py#L221)校验n并缓冲，下一轮优先消费；不是持久化事务证据。
- **训练数据转换还有一层契约。** [manager._generate:335](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/ray/rollout.py#L335)经BatchBuilder plan／convert／split；[convert:256](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L256)处理reward、rollout身份、loss mask、remove_sample全零mask及跨segment分母，带rollout_log_probs；[split_by_dp:445](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L445)按长度／身份排DP计划，不能把segment当独立prompt。自定义转换hook可以改变这些行为。
- **完成通知不是联合checkpoint。** [training_completed:144](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L144)完成batch／释放队列容量，driver明确与模型、optimizer及数据的共同持久化分开。pause admission也不自动等全部在途排空；本轮未读物理权重传输实现，不给atomic发布或故障exactly-once结论。

**限定验证与计数**：[脚本、口径和结果](../experiments/rollout-group-controls/readme.md)固定两完整OID／13份Python源码AST；八组CPU控制运行原stdlib coordinator／filter模块及AST提取原取消／版本函数，核配额、sync结转、pause／resume、失败cause、cancel drain、min-version、取消所属与条件过滤。真实asyncio、假sample／state；不导入框架，不运行HTTP／Ray／存储／optimizer／设备或性能实验，不称源码修复。C3本批2/2，代码9→11路径、6→8系统（6 RL框架＋2推理引擎），文献126与20/84/21/1、完整复现0不变。下轮先核HybridFlow已有认领再补原版代码／数据与覆盖审计；不启动M5专项或新增算力队列。

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

V1 TQ路径的[gen计时块:543](https://github.com/verl-project/verl/blob/24f25b03aa4b54249a273655ebbcce06f484192b/verl/trainer/ppo/v1/trainer_base.py#L543)包住on_sample hooks、replay_buffer.sample和extra_info设置；可包含轮询／选批及对应hooks，不是所有生成任务的累计GPU时间。桥接get／put局部日志分别只围绕取数或写数，worker计算在两者之间；不能把其读写日志和异步重叠的生成／训练时间简单相加作为整步成本。默认V1与旧分支区别见§8.8，此处未测量耗时。

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

- **2026-10-28（M1 C3 rLLM／Slime两路径）**：认领先推送ef69115；§8.11–8.12固定rLLM 4e828c81／Slime 540ea919。rLLM派发配额与in-flight、TaskBatch与转换组、出队与update、partial metadata与物理换权重分账；Slime默认阶段循环与另一fully_async入口分开，收够组后取消、条件过滤／partial回收资格、旧token mask与联合checkpoint边界保留。13源码AST／八组限定CPU控制通过，原stdlib模块＋提取函数，无框架／设备／性能验收；代码11路径／8系统，论文20/84/21/1及复现0不变。下轮先核HybridFlow已有认领，补历史实现／原始数据及覆盖审计；不以数量称Phase 2完成。无设备探测、M5专项、算力队列或社区写入。

- **2026-10-28（M1 C2推理调度器两路径）**：认领先推送e5b874b；§8.9–8.10固定vLLM eaf1627c／SGLang cb55d77d，独立稀疏只读检出，未改旧六月分支。核token／输入／KV预算、纳新与运行次序、抢占／re-prefill、乐观计数／spec回滚、deferred free；SGLang mixed分支排除logprob、回撤可abort或input_embeds重启，版本事件不等flush。九源码AST与八组原函数体控制通过（8700 chunk组合），替身接口／无原生引擎／性能证明；脚本和结果入experiments。代码7→9路径、4 RL框架＋2引擎＝6系统，文献20/84/21/1与复现0不变。下轮C3 1–2路径；数量达下限不称Phase 2完成，HybridFlow仍待。无GPU／NPU／SSH／VPN、M5专项或社区写入。

- **2026-10-28（M1 B13六篇T2）**：认领先推送505a5cf，§6.33完成Belayer／BIRD-RL／ScaleRLT及PrefixFlow／psRL／CIS；三原T3替换为T2，126＝20＋84＋21＋1，B13为6/6。Belayer显存分配／语义KV及环境本地／外部副作用分开；BIRD的12.8×是在线I/O且有离线成本；ScaleRLT读出policy另C、质量负项及same-weight概率值／dense梯度分开；两项训练前缀共享的update／整步与生成分账，CIS保留条件token目标、floor、偏差及rescore成本。均固定HTML方法／实验／必要附录阅读，无PDF图像、源码或设备验收。下一轮C2后C3，HybridFlow与代码目标仍待；代码7条／4框架、完整复现0，无M5专项、算力队列或社区写入。

- **2026-10-28（M1 C1 veRL数据面与B12余项）**：认领先推送d86477d，§8.8完成固定24f25b03的V1 TQ数据路径，累计7路径／4框架；默认use_v1=true／sync修正旧分支默认入口概括，生成后台提交不等训练逐样本更新，prompt／trajectory key清理、BatchMeta／tags／extra_info与概率来源分别核。10 Python AST／23锚点及4原函数五组stdlib CPU检查通过；3.9解析match失败改用现有3.12，不运行框架／TQ或设备。§5.25附带8范围triage：旧五线索入库，ScaleRLT身份可核；PrefixFlow／psRL单列T3，Belayer／BIRD-RL／ScaleRLT待T2，非自动降级。126＝120 arXiv＋6会议＝20 T1＋78 T2＋24 T3＋4未标；B12范围28/28，B13六篇待认领，HybridFlow及代码目标仍待。完整复现0；无设备、M5专项、算力队列或社区写入。

- **2026-10-27（M1 余八直接候选T2）**：§6.32完成PUZZLE／DynaRL／PivotRL／RobustRL／RLBoost／Relax／SPEC-RL／ProRL Agent；认领900f515，B6补后22/22、B8补后13/13，T2累计78、T1仍20／T3仍19。PDF核PUZZLE表2短256-token与变batch、DynaRL依托RLinf及基线差异；GenRM三角色／真实抢占trace回放／估价分账。RobustRL CPU角色未容错、Relax部分角色全局重启及0阈值表述疑点保留；SPEC-RL补A/C及Algorithm 1二分类反例、rollout2.51×与全步1.51×分开；ProRL补选样／取消C边界。121身份不增，互斥账20＋78＋19＋4；HybridFlow1＋边界3及另5线索仍待，下一轮C1 veRL数据面1路径可附带B12余项triage。代码6条／4框架、复现0不增，仍非Phase 2完成；无设备、M5专项、算力队列或社区动作。

- **2026-10-27（M1 B6补六篇L4 T2）**：§6.31完成DistFlow、Asynchronous RLHF、ReaL、RLHFuse、OpenRLHF、DeepSpeed-Chat，各6条；B6补后15/15，T2累计70、T1仍20、T3仍19。DistFlow本地中间张量cache／弱扩展与decoder计算分账；Async RLHF换backend／选best-worst／减步数及KL另C，GSM8K速度1.69×不写耗时减68%。ReaL强制满长度与真实长尾、估算MCMC与最优、RLHFuse同栈融合1.2–1.4×与对DSChat3.7×／mini-batch差异分开。PDF第5页确认OpenRLHF表1，几何均值复算1.146／1.181／1.393×不沿原汇总列；DSChat Step3与三阶段时间、历史基线／batch扩展分账。121身份不增，互斥账20＋70＋19＋12，直接候选余8；HybridFlow／B12余项／Code未完成，代码6条／4框架、复现0不增。无设备、M5专项或社区动作。

- **2026-10-27（M1 B12首批serving triage）**：§6.30复核§5.5全20条，19条新增T3、Router引用既有T2不双计；T1／T2仍20／64。19项回原站版本摘要／会议介绍，逐条写不追加T2理由及rollout迁移条件；PD goodput／组尾、缓存eviction／轨迹终止、历史文本draft／训练回放、speculator RL／actor RL分别保留。121身份、115 arXiv＋6会议不增；互斥深度账20＋64＋19＋18，18未标完成＝HybridFlow1＋边界3＋直接候选14，不能据批次表完成漏掉14项；另5条线索分账。B12未整体完成；下轮先核远端、按6–8篇T2补直接候选并在认领时微调批次，边界另作triage，不重复serving凑数。代码6条／4框架、复现0不增，无设备、M5专项或社区动作。

- **2026-10-27（M1 B11余六篇T2收束）**：§6.29完成Infrastructure Tax／ExpThink／Intelligent Router／AgentRL／RollArt／AstraFlow，B11为11/11、T2累计64、T1仍20；121身份不增，Router保留serving旁读。固定命令worker-hours外推与完整RL、输出token与TPS、混卡／外部reward与同资源收益分别登记；RollArt过旧逐出及冗余终止改B/C，AgentRL主动跨policy／task归一、ExpThink梯度单调性反例保留。AstraFlow弹性省GPU-hours却墙钟变慢、主计时与引用质量基线分账。HybridFlow进行中1、代码6条／4框架、复现0不变；下轮B12首15–20条triage，已T2只引用不重复计数；仍非Phase 2整体完成，无设备、M5专项或社区动作。

- **2026-10-27（M1 B10收束＋B11前五篇T2）**：§6.28完成八篇，B10为12/12、B11为5/8、T2累计58、T1仍20。TideRL弹性batch、QwenGyre路径target集合／评分、DART动态采样／回放／step优势和SAO单rollout目标另C；Qwen平均调度年龄前提不当behavior硬界。ROSE额外serving卡／局部同步、PEARL同trace／扩卡／四卡PD负项与切换延迟、Tangram ACT与RL尾延迟、MCP五步与长跑分别保留。DART原数相减14.61pp不沿正文12.71。121身份、HybridFlow进行中1、代码6条／4框架、复现0不增；下轮B11余三与AgentRL／RollArt／AstraFlow合批六篇，认领时按未读清单微调。无设备、M5专项或社区动作。

- **2026-10-27（M1 B10八篇agentic T2）**：§6.27完成EARL、InfoTree、AsyncWebRL、AcceRL、MARLaaS、FlexMARL及两篇Agent Lightning，T2累计50、B10为9/12、T1仍20。区分租户间异步／整批梯度更新／跨版本轨迹；InfoTree预算／理论前提、AsyncWebRL step-norm与组选样、AcceRL GIPO／WM及1000／2000离线轨迹冲突、Lightning credit／rollout-loss另C。EARL局部传输、MARLaaS任务批makespan、FlexMARL对naive／MARTI及质量缺证分账。121身份、HybridFlow进行中1、代码6条／4框架、复现0不增；下轮B10余三与B11前五共八篇，先核远端认领。无设备、M5专项或社区动作。

- **2026-10-27（M1 B9收束＋SkyRL-Agent T2）**：§6.26完成六篇，B9大小写别名去重为11/11，B10首篇完成，T2累计42、T1仍20。MTP-RL辅助目标回传policy、FlashNext冻结层／零优势组、SkyRL-Agent超外部限制梯度屏蔽另C；DTA训练8.31×／全流程2.28×、Online Draft 256K算子、Rollplex同步屏障与共享布局边界保留。MTP原PDF表1／4核负项，不把接受率、训练吞吐、异步执行或奖励接近当等价证明；121身份、HybridFlow进行中1、代码6条／4框架、复现0不增。下轮B10八篇先核远端再认领；无设备、M5专项或社区动作。

- **2026-10-27（M1 B9前六篇T2）**：§6.25完成RLHFSpec、Bebop、FastGRPO、System-Integrated SD、SPARK、SpecActor，T2累计36、T1仍20。保留局部成本模型／全局makespan、熵单调性反例、AR提前退出对照、实测／专有模拟、因果proposal与组独立性、抽样trace／连续训练的区别。B9余项先核别名与范围再同B10合批6–8篇，不按名称数认证分母；121身份、HybridFlow进行中1、代码6条／4框架、复现0不增，无设备、M5专项或社区动作。

- **2026-10-27（M1 B8余六篇T2）**：§6.24完成CERO、MISA-T、SRT、S³R、NgramRL、TLT，B8收束12/12、T2累计30，T1仍20。CERO固定utility理论与更新posterior实现、平均提升与7B AIME26负项分开；MISA-T混合有限偏差非零，SRT run-ahead模拟不并入主计时；S³R完整dense KV与组独立性前提、NgramRL token/forward及profile公式转换、TLT三步已训draft与非零开销保留。121身份、HybridFlow进行中1、代码6条／4框架、复现0不增；下轮先核远端认领B9前六篇，无设备、M5专项或社区动作。

- **2026-10-27（M1 B7余两篇＋B8前六篇T2）**：§6.23八篇完成，B7收束9/9、B8为6/12、T2累计24，T1仍20。FP16不当bit-exact，VeXact隔离TIM与PPO修正；TailSieve当前权重生成仍跨轮组批，修订A/B＋C；WAR计时后选与小batch负项、Seer最后10%尾时段、Infinite Sampling轮数／长度改变与伪代码保证、Cadence准入credit／轨迹年龄分别登记；Weave PDF原图保留testbed回放／模拟与局部最优前提。121身份、HybridFlow进行中1、代码6条／4框架、复现0不增；下一轮B8余六篇，先核远端再认领。无设备、M5专项或社区动作。

- **2026-10-27（M1 Long-RL＋B7前七篇T2）**：§6.22八篇完成，B6收束9/9、B7为7/9、T2累计16、T1仍20。保留forward／全步、通信字节／实测延迟／外推、Sync与LoCo两通道、NeMo-DCR压测／训练及KV边界；ThunderSyncRL数学更新等价不当bit-exact，PipelineRL旧KV混版本及估算基线另列，Missing Old Logits保留故障终止窗口，Beyond Precision保留启发式与伪代码差别。121身份、HybridFlow进行中1、代码6条／4框架、复现0不增。B7余三选二与B8中4–6篇合批6–8篇，先核远端再认领；无设备、M5专项或社区操作。

- **2026-10-27（M1 B6前八篇T2）**：§6.21完成AsyncFlow、LlamaRL、RLinf、MindSpeed RL、NeMo-Aligner、ROLL、ROLL Flash、GLM-4.5 §3.5，T2累计8、B6为8/9、T1仍20。Ascend平台／基线混栈、10.7×组合配置、扩卡与固定卡、RLinf空间负项、ROLL学习质量及ROLL Flash模拟／冗余选样分别保留；GLM只读系统片段，不增源码／复现计数。121身份、HybridFlow进行中1、代码6条／4框架、复现0不变；Long-RL下轮与B7中5–7篇合批，先核远端再认领。无设备、M5专项或社区动作。

- **2026-10-27（M1 B5后半T1压缩深读）**：§6.19–6.20完成ObsSpec v1／FP8-RL v2各24行，B5收束4/4、T1累计20；尚非Phase 2整体验收。ObsSpec核exact-match与动作核验后顺序执行、WM离线重放及额外8卡，agent／batch／async不同计量边界与TIR负项保留。FP8-RL PDF核真实低精度policy、token TIS非精确BF16轨迹校正，数值变化另C；32卡E2E MoE约700步仍崩溃，训练侧未见显著加速，不把44% rollout推广全步。121身份、代码6条／4框架、复现0不变；下一轮核远端后认领B6的T2批次，无设备、M5专项或社区动作。

- **2026-10-27（M1 B5前半T1压缩深读）**：§6.17–6.18完成EfficientRollout v2／Heddle v1各24行，B5仅2/4、T1累计18。W4只作draft、target覆盖临时KV并残差恢复；一次开启假设、反推α、模拟与实测、主100步／附加50步及CPU优化前后消融分开。Heddle长轨迹抢占不丢样本，排序DP最优限代理假设，异构MP退火为启发式；数学工具窗口短于预测／迁移，不宣称一律全遮蔽。121身份、代码6条／4框架、复现0不变；下一轮先核远端再认领ObsSpec／FP8-RL，无设备、M5专项或社区动作。

- **2026-10-27（M1 B4后半T1压缩深读）**：§6.15–6.16完成BubbleSpec／DAS v1各24行，B4收束4/4、T1累计16。BubbleSpec只用inter-GPU气泡，旧policy生成物仅作draft，按当前p逐token接受／残差恢复；统一attention与decode步数、rollout墙钟分账。DAS PDF第7页确认式7与式6约束代数不一致，CPU反例与适用域写于原文旁，不推断实现bug或否定实验；数学>50%／代码约25%分开。121身份、代码6条／4框架、复现0不变；下一轮核远端后认领B5前半EfficientRollout／Heddle，无设备、M5专项或社区动作。

- **2026-10-27（M1 B4前半T1压缩深读）**：§6.13–6.14完成RhymeRL／ReSpec v1各24行，B4仅2/4、T1累计14。历史token只供draft，HistoPipe一步陈旧／跨步迁移另C，不把历史匹配率当实际接受率或千卡部署自述当主测试。ReSpec加权对象为drafter KD；全Q与1/I消费口径留疑，peak／median／mean及累计消融分开。对其方差→偏差归因按精确SD条件与概率链规则提出论证疑点，不断言实际实现错误或actor目标改变。121身份、代码6条／4框架、复现0不变；下一轮BubbleSpec／DAS，无设备、M5专项或社区动作。

- **2026-10-27（M1 B3后半T1压缩深读）**：§6.11–6.12完成APRIL v3／DARTS v1各24行，B3累计4/4、T1累计12。APRIL按整prompt组纳入，续跑混版本／五版本观测与硬界区分，吞吐和质量百分点分账，abs／表1及40%计量口径差异保留。DARTS的7短1长、预算代理模型、饱和剪枝另C；token流水只提前前向，奖励／反向等全轨迹。同栈Tail Batching不冒充完整RollPacker，BBH负项保留。121身份、代码6条／4框架、复现0不变；下一轮B4前半RhymeRL／ReSpec，无设备、M5专项或社区动作。

- **2026-10-27（M1 B3前半T1压缩深读）**：§6.9–6.10完成RollPacker／SortedRL各24行，B3仅2/4、T1累计10。前者prompt延期不等保留原响应，最快完成响应与自适应评分超时改B/C；流式算梯度不提前更新。后者prompt池分组与GRPO组区分，重跑／partial行为logp及批归一化分别登记；PDF证实AIME正文／表1归属矛盾。队列空槽不当SM闲置，固定长度rollout TPS不当time-to-quality。121身份、代码6条／4框架、复现0不变；下一轮APRIL／DARTS，无设备、M5专项或社区动作。

- **2026-10-27（M1 B2后半T1压缩深读）**：§6.7–6.8完成PACE v1／ECHO-2 v6各24行，B2累计4/4、T1累计8。PACE核原PDF公式／表1–3，分开token均值年龄、供给阈值与硬界、4.95个百分点与同预算相对增益、六卡消融与16卡主配置；ReTool非等GPU对照。ECHO容量与分发分位数有前提，dated租价成本不当账单／单步更快，两者拒收另C。121身份、代码6条／4框架、复现0不增；下一轮认领B3前半两篇，无设备、M5专项或社区动作。

- **2026-10-27（M1 B2前半T1压缩深读）**：§6.5–6.6完成AReaL v5论文／RolloutPipe v2两篇24行模板，B2仅2/4、T1累计6，后半PACE／ECHO-2待认领。AReaL准入／混版本／目标调整与RolloutPipe固定生成权重／每U组更新／全R组发布分开；2.77×同栈训练小时、veRL估算小时及主时窗缩短百分比分账，等待比定义留待实现核。121身份、代码6条／4框架、复现0不变；不推进prompt_mean专项、设备或社区事项。

- **2026-10-27（M1 B1后半T1压缩深读）**：§6.3–6.4完成Laminar v1／DORA v2两篇22／24行模板，B1累计4/4、T1累计4，HybridFlow仍进行中。Laminar观测最大陈旧性不当硬界，relay链修复不当全任务恢复；DORA的rollout-only时段、实际消费tokens/s和生产非CUDA结果分开，RBS／TBS原文单位留疑，保留轨迹不认证每步等价，抽象理论不替源码协议。121身份、代码6条／4框架及复现0不变；下一轮核远端后认领B2前半两篇，不接续M5 prompt_mean，无设备或社区动作。

- **2026-10-27（M1 B1前半T1压缩深读）**：§6.1–6.2完成StreamRL v1／StaleFlow v2两篇20／21行模板，B1仅2/4、Laminar／DORA仍待认领；按新压缩标准计完成2，HybridFlow仍进行中1。StreamRL强制长度／真实收敛及同后端／跨后端分开；StaleFlow主评估无冗余，附录冗余／Stuck过滤另C，续跑迁移不混丢弃。121身份、代码6条／4框架不增，源码与完整复现未验收。下一轮先认领B1后半；无设备、M5专项或社区写入。

- **2026-10-27（用户加急指令：Phase 2 两天截止）**：121 篇深读须 ≤2026-10-29 完成——分级深度（T1 全模板压缩 20 篇／T2 中等 ~60 篇／T3 triage 其余）+ 14 批次表（§6.0）+ 认领规则（最小未完成批次，先认领后工作）；单次定时触发规模=T1×2 或 T2×6–8 或 T3×15–20 或 Code×1–2。放弃原 3–4 周排期。

- **2026-10-27（M1 HybridFlow映射成本与原图补核）**：§6补算法1的组内sum／组间max／跨stage sum及解析模拟前提，核附录与§8.3–8.5；PDF第11–13页视觉核对区分图12／13、transition／整步与TP／KV条件。图9(c)无逐柱精确比值，不猜上界GPU点；图16联合变化不当渐近复杂度证据。文献121、代码6条／4框架不增，深读仍进行中。后续定位论文年代实现／原始数据，无设备、M5专项或社区写入。

- **2026-10-27（M1 Phase 2接续保全与HybridFlow正文补核）**：保留79e892f深读方向、顺序及日志，恢复91105d0五行矩阵、§5.22–5.24／§13与121统计，不恢复旧Phase 1方向。§6补核v2重分片及实验条件，修正摘要＋源码即完成／以现代代码替论文的口径。深读完结0、进行中1、代码6条／4框架，无作者实现／设备／M5专项／社区写入。

- **2026-10-27（M1 Phase 2 启动：深读 #1 HybridFlow/veRL）**：用户指令——广度已足，转入深读（四维 L1–L4 对焦），M1 定时任务以深读为主。深读顺序 10 篇已列（§6）；完成 #1 HybridFlow（摘要+源码合读，实验细节标 unknown 待 PDF 补全）；发现论文表述与现行 agent_loop 代码的代差（以代码为准）。

- **2026-10-27（M1混合准入与环境流水补核）**：§5.23两条接续线索MISA-T／MCP-U RL并入§5.2／§5.4，§5.24分开有限混合偏差与渐近目标、固定checkpoint与训练循环、阶段并发与异步placement。121身份＝115arXiv＋6会议，86直接候选＋15单列＋20serving；余五项仍分账。旧119行／15条清单／代码地图保留，深读0、代码6条／4框架；仅局部机制／设置，无作者实现／设备／M5专项／社区动作。下步继续余项范围或未覆盖框架，不冻结分级。

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
