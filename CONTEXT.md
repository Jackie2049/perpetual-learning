# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

本次会话完成 §21.12（轨迹之外的证据与容量约束，补强 §21.11），待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§21.12**（语料内**无任何系统同时采集 kernel 活动与 KV 命中计数**，该判据仍属方法学断言；"哪类通道能分"取决于阈值落在哪，高负载下四条通道默认阈值全部同答案；唯一与语义绑定的是"真空泡里计数为 0"，到缓存命中量级 recall 1.0→0 而 specificity 恒 1（错误单向）；判据需 ≥2 个采样点 w/k ≥ 2）；此前 §21.11（分辨率与镜像不可分）／§21.10（气泡可辨识性）／§21.9／§21.8／§13.41／§13.40。先读主文档 §0/§16 再选已授权补强项；**气泡可辨识性这条线（§21.10–§21.12）暂无可再补的口子**。 |
| M5 | TS-001 算力支持 | 最新记录：第四十八批候选 H（update_weights bucket 扫描，2 卡双传输，§13.41）已完成并回写专项。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-005（掘金 B 类，M1 2026-11-02 16:1x）**：verl `use_prefix_grouper` 在 FSDP 引擎上不生效但副作用照常生效；取证材料与影响量化（DP 负载不均衡 0–6.6%）均已入库，**对外动作（issue／评论／PR）全部未做，待用户逐项授权**。见 [TM-005](queue_tech-miner.md) 与 [evidence](open-source/verl/evidence/prefix-grouper-inert/readme.md)。

- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）目录仍未跟踪，由该会话自行提交；本会话保留原状。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11／§21.12）是本会话的工作，已入库；本批新增 `fusion_evidence.py`／`test_fusion_evidence.py`／`fusion.json`／`readme-fusion.md`，`run-info.json` 已同时覆盖三批的 sha256。三套脚本共用目录但各自独立归档（`summary.json`／`resolution.json`／`fusion.json`）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
