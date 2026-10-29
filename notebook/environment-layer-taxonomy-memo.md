# Environment/Sandbox 是否独立成层：论证备忘（G6 转稿）

> 专项：[rollout-performance-research.md](rollout-performance-research.md) §13.15 G6 的转稿成果（2026-10-29 M5）。
> 状态：本地备忘（survey outline 备料的开放设计问题论证），**不冻结 taxonomy**（禁令），不写 survey 正文、不构成上游动作（铁律只读）。
> 用途：回答 taxonomy 是否要第五层（environment/sandbox）时，按本文证据清单与判定标准讨论；当前结论=候选保留、建议"L4 下显式子维"的中间路径。

## 1. 问题与结论

agentic 簇多次出现 environment bubble（观测预测、sandbox TTL、工具并行、环境生命周期），但现有 taxonomy 无独立层。问题：**environment/sandbox 是否够独立成第五层？**

**结论**：environment bubble 具备与 L1–L4 现有层不同的三个特征（非 GPU 墙钟成本、外部有状态正确性语义、独立优化杠杆），**够资格作为独立研究主题**；但"是否独立成层"是 taxonomy 设计决策，判定标准（§4）当前不满足（无论文量化 env 气泡在整步中的占比），因此**候选保留、不冻结**。建议的中间路径：不新建 L5 层，而是把"environment bubble"作为 L4 下的显式子维（气泡源④，§3）在 survey outline 中单列论证，避免 taxonomy 膨胀且不丢失论据。

## 2. 证据清单（§13.15 G6 原文 + 各深读）

| 工作 | env 相关机制（已核原文） | 出处 |
|---|---|---|
| ObsSpec 2610.15246v1 | 工具返回前用 world model 预测 observation；exact-match 保留、错误丢弃；agent 时延含 sandbox/tool 交互（tool 1.685s vs WM 0.105s；match 66.45%）而 batch 指标不含——同篇三口径不可互比 | §6.19、§13.12 |
| WAR 2607.17299v1 | sandbox 非活跃 TTL；"Once the sandbox reaches ttl_sandbox, the trajectory is terminated, resulting in an incomplete rollout"——论文级 incomplete rollout 确认（C 类失败消费） | §6.23.4、§13.7 |
| DART GUI 2509.23866v1 | GUI 环境自动化；42.13 vs 27.52 = +14.61pp（正文 12.71 算术不符，负项证据） | §6.28.5、§13.10 |
| ARL-Tangram 2603.13019v1 | `pytest` 改 `pytest -n` 并行直接启用，无测试顺序/状态等义验证（警示保留） | §6.28.4、§13.5 |
| BIRD-RL 2610.29426v1 | trajectory 绑定同一 DB 副本到终态、超时回收；脏副本释放不立刻 reset，下次租用前恢复（dirty-aware lazy reset）；在线生命周期 I/O 降 12.8× | §6.33.2 |
| Rollout Infrastructure Tax 2607.01415v1 | 固定命令轨迹对照 container/hosted sandbox/K8s/VM；最轻冷启动差最高 110×，重 T2 缩至约 1.9×；百万轨迹外推约 1.8× worker-hours（排除推理/reward/更新） | §6.29.1 |
| MCP Universe RL 2608.22167v1 | 每 trajectory 独立环境、hold 至评分完成再 reset/destroy；共享配置预热、SWE 按需 Apptainer | §6.28.6 |
| RollPacker 2509.21009v1 | 自适应 sandbox 超时 `clip(1.5T_anchor, 2s, 30s)`，超时记零分——**env 超时改变奖励边界** | §5.1 矩阵行、§6.9 |

气泡来源分类（§3）：④ environment/tool 等待是五类气泡之一；§13.1/§5.24 已把 Rollout Infrastructure Tax 放"L4 环境执行边界"、不据此冻结第五层。

## 3. environment bubble 与现有层的三个区分特征

1. **成本模型不同（非 GPU 墙钟）**：env 等待是外部进程/工具/网络墙钟，不占用 GPU FLOPs；GPU 侧指标（tokens/s、ms/token）完全看不见 env 成本——ObsSpec agent 时延含 sandbox/tool 交互而 batch 指标不含（§13.12），Rollout Infrastructure Tax 的 1.8× 外推明确排除推理/reward/更新。现有 L1（token 执行）、L2（engine）、L3（cluster）的指标体系均不以墙钟等待为本位。
2. **正确性语义不同（外部有状态）**：env 有状态与副作用——BIRD-RL 的 DB 副本绑定/脏感知 lazy reset、ARL-Tangram 的 pytest 顺序/状态等义未验证、ObsSpec 的错误预测分支需顺序提交防副作用、WAR/RollPacker 的 TTL/超时终止轨迹（incomplete rollout、记零分）直接改变训练集合/奖励边界（C 类）。L1–L3 讨论的是模型分布与执行调度，不涵盖外部状态一致性。
3. **优化杠杆独立**：sandbox 池化/复用/预热（MCP Universe 共享配置预热）、生命周期管理（BIRD-RL 终态释放/lazy reset）、并行化（ARL-Tangram pytest -n）、预测重叠（ObsSpec WM）、TTL 管理（WAR）——一组与 GPU 内核/批调度/kernel 复用不同的杠杆集合。

## 4. 独立成层的判定标准（不冻结，仅列标准）

若未来要冻结 L5 = "Environment/Sandbox Execution"（核心问题：如何让轨迹的环境交互便宜且安全），须满足：
1. **独立杠杆集合**：已有（§3.3），但多数杠杆只在 agentic 簇出现；
2. **可量化气泡占比**：**不满足**——无任何论文量化 env 气泡在整步墙钟中的占比（Rollout Infrastructure Tax 最接近但排除模型推理，ObsSpec agent 时延是口径而非全步分解）；没有占比数据，"独立成层"缺乏量纲支撑；
3. **与 L1–L4 的非平凡交互**：部分成立——env 等待与 GPU 计算的重叠可调度性（ObsSpec 预测重叠、WAR TTL 紧迫性排序）表明交互非平凡。

**反论证**：非 agentic RL 大部分无 env；taxonomy 分层应反映主导气泡结构；L4 已含气泡源④。因此当前判定 = **候选保留、不冻结**。

## 5. 建议的中间路径与用途

- **中间路径（推荐）**：不新建 L5 层，在 survey outline 中将 "environment bubble" 作为 L4 pipeline co-design 下的显式子维（气泡源④的展开：成本构成/失败消费语义/杠杆谱系）单列论证，同时保留"是否独立成层"为开放设计问题。这样既不冻结 taxonomy，也不丢失 agentic 簇的论据。
- **用途**：survey outline 备料时作为"环境层论证"主题的论据骨架；G7 中与 env 相关的源码级事项（WAR TTL 源码消费、CadenceRL in-flight 时机）随各系统源码可获取时核（铁律约束）。
- **证据来源**：§13.15 G6、§3 气泡分类、§5.24 环境流水账、§6.19/§6.23.4/§6.28.4–6.28.6/§6.29.1/§6.33.2 各深读、D4–D6/D10 核对账（§13.5/§13.7/§13.11）。
- 不新计深读、不扩库；126 身份 20/84/21/1、代码 12 路径/8 系统、复现 0 不变。
