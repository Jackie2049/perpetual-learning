# Serving → Rollout 迁移判据四要素备忘（G5 转稿）

> 专项：[rollout-performance-research.md](rollout-performance-research.md) §13.15 G5 的转稿成果（2026-10-29 M5）。
> 状态：本地备忘（可复用检查清单），不写 survey 正文、不冻结 taxonomy、不构成上游动作（铁律只读）。
> 用途：对任一 serving 优化问"在 rollout 下 objective 是否不变"时，用本文四要素判据逐条核对，再判定迁移类别。

## 1. 目的与结论

serving 优化迁移到 RL rollout 时，"性能收益保留"与"训练 objective 不变"是两个独立问题。本文把 §6.30 十九条 serving T3 逐条迁入条件与 D2/D8/D10 深化核（Helium/CONCUR/SuffixDecoding/FlashInfer）归纳为**四要素判据**，作为迁移前核对清单。

**核心结论（负面证据）**：现有 126 篇中**没有一条 serving 优化被证明在 rollout 下 objective 完全不变**。最接近"可直接迁移"的是三类：L1 后端替换（FA/FA2/FlashInfer，机制方向可用但指标口径与数值一致性须另核）、draft-only 路径（SuffixDecoding/Leviathan/ReSpec 同构，须当前 target 验证）、确定性缓存限定（Helium，只缓存确定性算子）；但它们各自仍差一步（版本失效契约／保分布前提核验／随机独立性自证）。其余多数 serving 优化迁移后**须补契约**（13 条），个别存在 **C 类风险**（Mooncake early rejection→训练准入）。

## 2. 四要素判据定义

对任一 serving 优化 O，迁移到 RL rollout 前按四要素核对（任一要素不满足即须补契约）：

| 要素 | 问题 | 反面例子（已核） |
|---|---|---|
| **V 版本约束** | O 依赖的缓存/中间产物是否绑定 policy 版本？policy 更新后如何失效/重算？ | CachedAttention"不自动保证更新权重后的 KV 有效"；SGLang"更新 policy 后还需缓存版本契约"；SuffixDecoding 无 policy 语义 |
| **S 陈旧性（staleness）** | O 引入的延迟/缓冲是否产生 off-policy 数据？有无上界或收敛保证？ | CONCUR 全文无 staleness 语义；Intelligent Router 需 policy 年龄与分布漂移条件 |
| **G 组完整性** | 同步 RL 的整组完成/屏障语义是否被破坏（准入=暂停 vs 丢弃、TTL 终止、尾批）？ | Orca"逐请求提前返回不代表同步 RL 的整组/整批屏障消失"；Mooncake early rejection 是 serving 语义 |
| **R 随机独立性** | 复用/缓存/重排是否保持采样独立性（确定性 vs 随机输出、缓存命中不改变分布）？ | Helium 缓存限确定性算子、随机输出复用无保证；Leviathan 保分布≠同随机种子逐 token 相同 |

## 3. 十九条 × 四要素核对表

来源：§6.30 逐条迁入条件（B12 triage）+ D2/D8/D10 深化核。判定域：**A**=objective 不变可直接迁移；**B**=须补契约后迁移；**C**=迁移改变训练分布（单独标记）；**M**=机制方向可用（仅启发，不保证 objective）。

| # | 条目 | 四要素结论 | 判定 |
|---|---|---|---|
| 1 | [Orca OSDI22](https://www.usenix.org/conference/osdi22/presentation/yu) | G：逐请求提前返回 ≠ 同步 RL 整组屏障消失 | **B**（组完整性须保持） |
| 2 | [vLLM v1](https://arxiv.org/abs/2309.06180v1) | V：KV 共享须同版本、相同前缀、实际驻留；指标：内存节省不能推出最长轨迹缩短 | **M/B**（机制方向可用，版本契约＋指标口径须另核） |
| 3 | [SGLang v2](https://arxiv.org/abs/2312.07104v2) | V：RadixAttention 缓存版本契约；R：约束输出定义不同采样空间 | **B**（版本＋采样空间语义） |
| 4 | [FlashAttention v2](https://arxiv.org/abs/2205.14135v2) | 训练/prefill 的 IO 收益不直接套逐 token decode，不等数值路径逐位一致 | **M**（L1 后端；数值/计量不可迁移） |
| 5 | [FlashAttention2 v1](https://arxiv.org/abs/2307.08691v1) | 须结合 prefill 长度/batch/硬件判断瓶颈，kernel 利用率不当 RL 气泡消除 | **M**（L1 后端；指标口径不可迁移） |
| 6 | [FlashInfer v2](https://arxiv.org/abs/2501.01005v2) | D10 核：KV 布局/JIT/负载均衡可迁；端到端收益限定 SGLang v0.3.4/MLC-Engine 接入；**训练前向-only**，vLLM 无端到端数据 | **M**（L1 后端；训练侧不可迁移） |
| 7 | [Sarathi Serve v3](https://arxiv.org/abs/2403.02310v3) | G：须按 RL 真实输入/输出比例和整组完成重评，不能预设 prompt 主导 | **B**（负载口径＋组完成） |
| 8 | [DistServe v3](https://arxiv.org/abs/2401.09670v3) | G：RL 迁移需计 KV 传输及**组尾**；指标：SLO 合格请求数≠训练样本消费速率 | **B**（组尾＋指标口径） |
| 9 | [Mooncake v5](https://arxiv.org/abs/2407.00079v5) | G：overload rejection 是 serving 语义；若用于训练准入须另核任务混合/样本集合 | **C 风险**（准入改变训练分布，须单独标记） |
| 10 | [Medusa v3](https://arxiv.org/abs/2401.10774v3) | V：Medusa1 冻结骨干 vs Medusa2 联合训练须分别核；R：接受规则与版本适配，质量保持不替代精确分布契约 | **B**（版本＋保分布前提） |
| 11 | [EAGLE v3](https://arxiv.org/abs/2401.15077v3) | V：RL policy 持续变化后接受率/适配成本须由具体 RL 系统另证 | **B**（接受率随版本变化无契约） |
| 12 | [SuffixDecoding v3](https://arxiv.org/abs/2411.04975v3) | D8 核：历史输出只作 draft 由当前 serving 模型单次前向验证；无 cross-step/worker/policy 语义 | **M**（draft-only 路径；版本契约须自建） |
| 13 | [Leviathan 等 v2](https://arxiv.org/abs/2211.17192v2) | R：保分布≠同随机种子逐 token 相同，需核实际 p/q 与采样实现 | **M**（数学基础；实现级须核） |
| 14 | [Chen 等 v1](https://arxiv.org/abs/2302.01318v1) | R：迁入 RL 须匹配 temperature/截断后实际分布，不能把"精确"扩张为全栈 bit-exact | **B**（分布匹配含温度/截断） |
| 15 | [Aurora v5](https://arxiv.org/abs/2602.06932v5) | V：target 是否同步更新、draft 学习成本和训练目标须在 RL 场景另核 | **B**（target/draft 版本关系） |
| 16 | [Intelligent Router 既有 T2](#6293-intelligent-router) | S：serving 延迟≠rollout 组 makespan，仍需 policy 年龄与分布漂移条件 | **B**（staleness＋组度量） |
| 17 | [CachedAttention ATC24](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | V：解决 context 溢出的缓存有效性，**不自动保证更新权重后的 KV 有效** | **B**（版本失效语义） |
| 18 | [CONCUR v1](https://arxiv.org/abs/2601.22705v1) | D8 核：准入=agent 粒度暂停且 "without discarding their execution state"（不改变样本集合，兼容性好）；但全文无 staleness/训练组/整组完成语义 | **B**（整组＋staleness 须自建；准入语义与 RL 兼容） |
| 19 | [Continuum v7](https://arxiv.org/abs/2511.02230v7) | V：跨 policy 更新的 KV 另核；G：TTL 到期只允许 evict cache **不等 abort rollout**（不终止轨迹，兼容性好） | **B**（版本失效；轨迹不终止语义与 RL 兼容） |
| 20 | [Helium v1](https://arxiv.org/abs/2603.16104v1) | D2 核：缓存**仅限确定性算子**（non-LLM＋greedy/零温度）；若复用随机输出须自证组内采样独立性、版本和 reward 语义 | **B**（随机独立性自证） |

（#16 Intelligent Router 为既有 T2 引用，不重复计数；#1–20 对应 §6.30 二十行。）

## 4. 判定汇总与反例登记

- **A（objective 完全不变）**：0 条。§15 Q1 的反例搜索结论——现有证据中无一 serving 优化达到该标准；这是本备忘的负面证据价值。
- **M（机制方向可用）**：#2、#4、#5、#6、#12、#13（L1 后端替换与 draft-only 路径），但每条都带"指标口径/数值一致性/版本契约另核"的尾巴，不能直接写"可迁移"。
- **B（须补契约）**：#1、#3、#7、#8、#10、#11、#14、#15、#16、#17、#18、#19、#20——补契约集中在三处：版本失效语义（V）、整组完成语义（G）、随机独立性自证（R）。
- **C 类风险**：#9 Mooncake early rejection→训练准入（若采用该机制，须按 §4 集合规则单独标记）；#3 SGLang 约束输出定义不同采样空间（若作训练数据生成亦改变分布）。
- **兼容性正面样本**：#18 CONCUR 准入=暂停非丢弃、#19 Continuum TTL 只 evict cache 不 abort rollout——这两条表明"serving 侧失败语义"存在与 RL 样本保留原则相容的设计，是迁移时值得保留的边界（与 WAR 的 TTL 到期终止轨迹形成对照，WAR 论文级已确认产生 incomplete rollout，§6.23.4）。

## 5. 适用范围与证据来源

- 本备忘覆盖 §6.30 全部 19 条 serving T3＋1 条 T2 引用；深化核来源：D2（§13.3 Helium 随机独立性）、D8（§13.9 CONCUR 准入语义/SuffixDecoding draft-only）、D10（§13.11 FlashInfer 前向-only/端到端限定）。
- 判据四要素命名（版本约束/staleness/组完整性/随机独立性）沿用 §13.15 G5 原文，未新增术语。
- 不新计深读、不扩库；126 身份 20/84/21/1、代码 12 路径/8 系统、复现 0 不变。
- 后续使用：survey outline 备料时本备忘可直接作为"serving→rollout 迁移"主题的论据骨架；G8 算力恢复后如需实证，可选取 #6 FlashInfer（4090 可跑 kernel 级）或 #2 vLLM（同栈锚点）做受控对比。
