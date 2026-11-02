# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

本次会话完成 §21.11（气泡归因的分辨率与镜像不可分，补强 §21.10），待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§21.11**（分辨率与镜像不可分：10s 采样下短于采样间隔的窗口**整段读不到**而非读成同长，最小可分辨窗口≈一个采样间隔，窗长比门槛 2；同一轨迹换标签统计量逐位相同，判据是归类器不是识别器；可加的是读数不是质量）；此前 §21.10（气泡可辨识性）／§21.9（会计恒等式吃的是被测分项）／§21.8（C3 同步基准不可从 trace 取）／§13.41（候选 H）／§13.40（`use_prefix_grouper` 死代码）。先读主文档 §0/§16 再选已授权补强项。 |
| M5 | TS-001 算力支持 | 最新记录：第四十八批候选 H（update_weights bucket 扫描，2 卡双传输，§13.41）已完成并回写专项。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）目录仍未跟踪，由该会话自行提交；本会话保留原状。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11）是本会话的工作，已入库；`resolution_limits.py`／`test_resolution_limits.py`／`resolution.json`／`readme-resolution.md` 为本批新增，`run-info.json` 已同时覆盖两批的 sha256。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
