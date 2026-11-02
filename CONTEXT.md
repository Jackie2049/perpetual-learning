# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

本次会话完成四文档迁移，待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为迁移时已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 原记录最新为 §13.37（force_group_size 生产者核查）；先读主文档 §0/§16，核对并行 PP 气泡工作是否已提交，再选已授权补强项。 |
| M5 | TS-001 算力支持 | 原记录最新为候选 G NUMA 对照；A/B 复测曾因并行 candidate-H 共用 GPU4 而失败，不能计入数据。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- 迁移前工作区存在未跟踪 `experiments/rollout-g8-candidate-h/`，本会话保留原状，不代为提交或删除。
- 原记录提及并行 `experiments/rollout-real-example-configs/` 与专项编号冲突风险；是否仍运行未知，接续方先核对实际文件及主文档。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- 历史记录曾提及 `diary/2026-08-14.md` 为用户未跟踪工作；本次状态输出未显示它，不据此推断被删除或已归档。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
