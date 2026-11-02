# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M1 本轮完成 §13.43（C 类机制账，落地 §18.9 的下一候选）；M5 本轮完成 §13.44（gen 二态隔离，4090-1 真机三轮，插桩已还原）。待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§13.43 第二十四批订正**：**派生量 `gap` 化简后 `≡ 1 − wall_ratio/cost_ratio`，步数被约掉**——"`gap<0` ⇒ 收敛变快／`gap=1` ⇒ 全部被吃掉"两条读法**作废**（反例 `freq` gap −1.93 而 `steps_ratio` 1.070 变慢；`C` vs `B` gap 1.0 而 `steps_ratio` 0.965 变快）；**`cap` 改为"省样本"**（`traj_ratio` 0.282）、`cancel` 的"收益在收敛"不成立（轨迹 0.984，真实机制每轮固定成本、分项未定）；四类收益来源改按 `cost_ratio`／`traj_ratio`／`wall_ratio` 直接读数归类；**G2 撤回"`gap` 是收敛代价的度量"**，改为需四字段同报、语料内齐全者为零。同日第二十三批立账记录保留在 §16，其 ②③ 的数字不变、①④⑥ 的读法已推翻。此前 **§21.13**（§21 八批实证横向总览）／§21.12／§21.11／§21.10／§21.9／§21.8／§13.41／§13.40。**气泡可辨识性这条线（§21.10–§21.13）暂无可再补的口子**，C 类机制账已做了两轮（立账＋订正）。 |
| M5 | TS-001 算力支持 | 最新：第五十一批多卡锚点按 2+2 重报（**§13.45**，4090-1 真机，10:59–11:24 UTC）——同配置重测 g1p／g2p(1+1)／g4p(2+2)＝27.522／18.031／14.173s＝**1.526／1.942×**（效率 0.763／0.485），各 15/15 步、exit 0；g4p vs pC1＝1.0004×（各阶段 ≤1.4%）、g2p vs pD1＝0.980×（差全在 gen 慢/快态配比，分层后 ≤1.2%，§13.44 机制第三次复现）。§13.36 的 1.251／1.631× 降级为 2+0／3+1 放置值（同卡数慢 22.1%／19.1%），多卡锚点自此固定 2+2（0,1,4,5）／1+1（0,4）。最佳放置下 update_actor G=2.087、update_weights 0.770 仍负扩展，加卡收益递减结论不变；**弱扩展也已按 1+1／2+2 重测**：g2wp／g4wp 31.702／42.246s（vs 旧放置 38.520／51.246s），聚合吞吐 1.749／2.629×（效率 0.868／0.651，旧 1.417／2.145×、0.709／0.536），`gen` 基本不动、收益全在集合通信。**M5 登记的剩余算力事项已清空。** 此前 §13.44（gen 二态＝2.0s 轮询量子）／§13.42（候选 G 放置效应）／§13.41（候选 H bucket）。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-006（掘金 B 类，M1 2026-11-02 17:3x）**：verl `tiled_mlp` 配置在 `verl/workers/engine/` 下 10 个引擎全部无消费点（正式配置项＋官方文档齐全，配置完全无效且无提示；无副作用）。工具沉淀为 [tools/pyflag](tools/pyflag/readme.md)。**本地补丁已就绪**：[patches/tiled-mlp-wiring](open-source/verl/patches/tiled-mlp-wiring/readme.md)（3 行接线，`patch -p1` 与修复前后参数对比均通过；未上 GPU 验证）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类，M1 2026-11-02 16:1x）**：verl `use_prefix_grouper` 在 FSDP 引擎上不生效但副作用照常生效；取证材料与影响量化（DP 负载不均衡 0–6.6%）均已入库，**对外动作（issue／评论／PR）全部未做，待用户逐项授权**。见 [TM-005](queue_tech-miner.md) 与 [evidence](open-source/verl/evidence/prefix-grouper-inert/readme.md)。

- **`experiments/rollout-g8-gen-bimodality/`（第五十批，§13.44）已入库**：3 份训练日志＋逐请求/相位 JSONL（gz）＋`summary.json`＋插桩/还原脚本＋md5；服务器 verl 树的插桩已回滚（`llm_server.py`／`ray_trainer.py`／`v1/trainer_base.py`／`v1/trainer_sync.py`／`v1/replay_buffer.py` 五文件 md5 与备份一致，全树无 `_m5_log` 残留）。
- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）已入库（commit `cec0382`），此前"仍未跟踪"的记录已过期。本会话本批（第四十九批）的 `experiments/rollout-g8-candidate-g/`（README 新增小节＋`ab-summary.json`＋10 份训练日志＋微基准/链路日志＋md5sums）已随本批提交。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11／§21.12）是本会话的工作，已入库；本批新增 `fusion_evidence.py`／`test_fusion_evidence.py`／`fusion.json`／`readme-fusion.md`，`run-info.json` 已同时覆盖三批的 sha256。三套脚本共用目录但各自独立归档（`summary.json`／`resolution.json`／`fusion.json`）。
- `experiments/rollout-c-class-ledger/`（§13.43）为本批新增并同日两次写入：`ledger.py`／`test_ledger.py`／`ledger.json`／`readme.md`／`run-info.json`；第二十四批新增 `gap_pieces`／`traj_ratio`／bootstrap，`run-info.json` 已记录两轮 revision 与三条 superseded 读法。只读引用 `rollout-p3-dapo-group-filter/summary.json` 与 `rollout-p4-c-mechanisms/summary.json`，**未修改那两批的归档**（sha256 已记入本批 run-info.json 的 `input_sha256`，且与上一版一致）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
