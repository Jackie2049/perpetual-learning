# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

本次会话完成 §13.43（C 类机制账，落地 §18.9 的下一候选），待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§13.43**（从 §13.28／§13.29 归档 per_seed 重算，不重新训练：L1 生成量／L2 每步成本／L3 步数／L4 墙钟分层，缺口 `gap = 1−(1−r)/(1−κ)`；**七机制无一属"每步兑现型"**——`freq` 每步省 72.3% 而墙钟只省 18.7%（gap −1.93）、`cap` 每步省 0 却墙钟省 6.3%、`cancel` 每步省 1.6% 却墙钟省 34.0%、§13.28 删组每步省 44.9% 而到 0.9 墙钟慢 2.7%；归四类 I–IV；**两处订正**——§13.29 墙钟比列 `cancel` 0.660 是基线/臂应为 1.515、§13.28"只再省 3.5%"是总成本比之比）；§18.9 7.6 已改写为分栏要求＋四类收益来源。此前 §21.12／§21.11／§21.10／§21.9／§21.8／§13.41／§13.40。**气泡可辨识性线（§21.10–§21.12）与 7.6 骨架线均暂无可再补的口子**；下一候选剩 G2/G9 负项索引的进一步实证。 |
| M5 | TS-001 算力支持 | 最新：第四十九批候选 G 放置效应机制（**§13.42**）——同一空闲窗口 A/B/A/B 复现 16.3% 差（排除漂移），taskset 2×2 排除主机 CPU/内存本地性；NCCL SHM all-reduce 带宽随「每 socket 最多 rank 数」下降（4+0 3.08／3+1 4.09／1+3 4.07／2+2 5.99 GB/s），端到端 2+2 放置 4 卡整步 14.167s＝1.944×（效率 0.486）、2 卡 1+1 18.403s＝1.496×（0.748）；§13.36 的 1.631×／1.251× 引用处必须附放置条件，后续多卡锚点建议固定 `0,1,4,5`。剩余：gen 二态隔离（逐引擎计时）未做。此前第四十八批候选 H（bucket 扫描，§13.41）。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-006（掘金 B 类，M1 2026-11-02 17:3x）**：verl `tiled_mlp` 配置在 `verl/workers/engine/` 下 10 个引擎全部无消费点（正式配置项＋官方文档齐全，配置完全无效且无提示；无副作用）。工具沉淀为 [tools/pyflag](tools/pyflag/readme.md)。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类，M1 2026-11-02 16:1x）**：verl `use_prefix_grouper` 在 FSDP 引擎上不生效但副作用照常生效；取证材料与影响量化（DP 负载不均衡 0–6.6%）均已入库，**对外动作（issue／评论／PR）全部未做，待用户逐项授权**。见 [TM-005](queue_tech-miner.md) 与 [evidence](open-source/verl/evidence/prefix-grouper-inert/readme.md)。

- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）已入库（commit `cec0382`），此前"仍未跟踪"的记录已过期。本会话本批（第四十九批）的 `experiments/rollout-g8-candidate-g/`（README 新增小节＋`ab-summary.json`＋10 份训练日志＋微基准/链路日志＋md5sums）已随本批提交。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11／§21.12）是本会话的工作，已入库；本批新增 `fusion_evidence.py`／`test_fusion_evidence.py`／`fusion.json`／`readme-fusion.md`，`run-info.json` 已同时覆盖三批的 sha256。三套脚本共用目录但各自独立归档（`summary.json`／`resolution.json`／`fusion.json`）。
- `experiments/rollout-c-class-ledger/`（§13.43）为本批新增：`ledger.py`／`test_ledger.py`／`ledger.json`／`readme.md`／`run-info.json`；只读引用 `rollout-p3-dapo-group-filter/summary.json` 与 `rollout-p4-c-mechanisms/summary.json`，**未修改那两批的归档**（sha256 记入本批 run-info.json 的 `input_sha256`）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
