# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

本次会话完成四文档迁移，待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为迁移时已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§21.9**（会计恒等式吃的是被测分项：§13.25 的重建精确不构成归因正确的证据；C1 缺口传播 5.9%–17.6%／0.6%–4.6%／0.8%–9.0%）；此前 §21.8（C3 同步化基准不可从 trace 内取）／§13.41（候选 H）／§13.40（`use_prefix_grouper` 死代码）／§13.38／§13.39。先读主文档 §0/§16 再选已授权补强项。 |
| M5 | TS-001 算力支持 | 最新：第四十八批候选 H（update_weights bucket 扫描，2 卡双传输，§13.41）已完成并回写专项；bucket 16× 跨度无效应（−1.2%／−2.6%），负扩展由副本数决定；方法学：多卡必须先设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（否则 update_actor +40.5%）。实验目录仍为未跟踪，由该会话自行提交。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）已完成，目录仍为未跟踪，由该会话自行提交；本会话保留原状，不代为提交或删除。
- `experiments/rollout-real-example-configs/`（§13.40）已完成并入库。`experiments/rollout-c3-sync-baseline/`（§21.8，14:4x 提交）与 `experiments/rollout-identity-measurement-bias/`（§21.9，15:0x 提交）**都是本会话的工作**；此前 CONTEXT 把 c3-sync-baseline 记作"另一并行会话"有误，已更正。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- 历史记录曾提及 `diary/2026-08-14.md` 为用户未跟踪工作；本次状态输出未显示它，不据此推断被删除或已归档。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。

并行更新追溯：[合并前远端 WORK 快照](https://github.com/Jackie2049/perpetual-learning/blob/55f70f4/WORK.md)。§13.39 已有独立实验结果，不据旧待办重复启动。
