# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M1 本轮完成**四个耦合面的定量可证伪点成文（§11.4）＋会计面实测复核（§13.46）**：把 §11.3 的四个面各压成一条带阈值的判据并标注阈值来源，然后用 M5 既有归档把会计面那条**跑了一遍**。核心读数：① 恒等式的份额 `s_k` **必须以各阶段墙钟之和 Σt_k 为分母**——本栈整步比分项和大 0.28%，整步口径造出 0.22%（2 卡）／0.51%（4 卡）伪偏移，分项和口径下公式与逐阶段相加**逐位相同**；② 实验值"优于投影"的 0.42%／0.28% 是两段同号计时缺口叠加，**不含额外重叠收益**，§20.3.5 的"0.040–0.076s 残差"是口径量；③ A/B 八档缺口 −0.041±0.001s ＝本栈计时缺口标定值。只读两份既有归档、无 GPU、无新读数。M5 本轮无新工作（上一轮 TM-006 FSDP2 真机冒烟已完成）。待后续指令；本次维护不启动、暂停或接管其他会话的自主运行。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§11.4 四耦合面判据＋§13.46 会计面实测（第二十八批）**：算力面 w/c（需同负载关机制基线，语料无）／会计面代数判据（已实测）／语义面 r/R（R 从未被量）／状态面净账符号（读不出＝P-2）。会计面三条读数与两处连带修订（§20.3.1 锚② 行、§18.5 3.4 行加"份额分母＋残差+缺口"口径条件）见 §13.46。三条可选后续（w/c 对照／R 的回收对照／P-2 成对测量）均需新实验，未立项。此前为 **§13.43 第二十六批口径检查（⑩–⑫）**：比值与四类归属不动，只判定分辨力边界——① p4 基线 **28/30** 种子最终准确率 = 1.0、四臂最高准确率 30/30 打满，最终评测 200 条 ⇒ 网格 0.005，**比 §13.29 ⑥ 的 Δacc 区间还宽**；② p4 步数差非零对 16–24/30 且与墙钟差撞同一网格 ⇒ **不归因** `cap` 的墙钟收益；③ 四臂 `total_gen ≡ total_cost` 逐种子成立、`traj_ratio` 在 `freq`／`replay`／`cancel` 上等于 `cost_ratio` ⇒ 四类判据的**独立实例数是 1 不是 4**（只有 `cap`）；④ 注入三个缺陷验证检查对输入敏感。单测 30 个，`--verify` 通过。此前为 **§20.3.5**（P-1 完成度由 0 变约一半：同栈多配置满足、跨系统仍缺）／**§13.43 第二十四批订正**（`gap ≡ 1 − wall_ratio/cost_ratio`，两条旧读法作废）／§21.13／§21.12／§21.11／§21.10／§13.41／§13.40。**气泡可辨识性这条线（§21.10–§21.13）暂无可再补的口子**，C 类机制账已做两轮（立账＋订正＋口径检查）。 |
| M5 | TS-001 算力支持 | **上一轮（2026-11-02 12:0x–12:2x UTC）TM-006 的 FSDP2 真机冒烟已完成**（4090-1 单卡，上游 `6879e1ab07` tarball＋本地补丁）：FSDP2 下 `enabled=True` 真实 patch 到 MLP 类并跑通前反向；不打补丁与 `enabled=False` 逐位相同；fp64 CPU 对照精确等价；**FSDP1 三种变体全部硬报错**；显存仅在关掉梯度检查点时省（16.39→8.74 GB），**verl 默认检查点开启时 5.46 GB 不变**。归档 `experiments/verl-tiled-mlp-fsdp-smoke/`。**M5 登记的剩余算力事项仍为空**。本轮无新工作。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-006（掘金 B 类，M1 2026-11-02 17:3x；M5 同日 12:2x UTC 补真机冒烟）**：verl `tiled_mlp` 配置在 `verl/workers/engine/` 下 10 个引擎全部无消费点。工具沉淀为 [tools/pyflag](tools/pyflag/readme.md)。**本地补丁已就绪**：[patches/tiled-mlp-wiring](open-source/verl/patches/tiled-mlp-wiring/readme.md)（3 行接线）。**真机验证已完成**：FSDP2 下真实生效、fp64 精确等价；**FSDP1 下开启硬报错**（需在 PR 里补配置期校验）；**默认梯度检查点开启时无显存收益**。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类，M1 2026-11-02 16:1x）**：verl `use_prefix_grouper` 在 FSDP 引擎上不生效但副作用照常生效；取证材料与影响量化（DP 负载不均衡 0–6.6%）均已入库，**对外动作（issue／评论／PR）全部未做，待用户逐项授权**。见 [TM-005](queue_tech-miner.md) 与 [evidence](open-source/verl/evidence/prefix-grouper-inert/readme.md)。

- **`experiments/verl-tiled-mlp-fsdp-smoke/`（M5 2026-11-02，TM-006 冒烟）已入库**：`smoke_tiled_mlp.py`／`cpu_equiv.py`／`run_smoke.sh`／`summarize.py`＋`raw/`（32 份 JSON，含 `results.json` 汇总）＋`logs/`（29 份）＋`md5sums.txt`。服务器受测树在 `/tmp/m5-tiled-mlp/`（`verl-6879e1ab07` 原样与打补丁两份，128G 空间），未占用 `/home`。
- **`experiments/rollout-g8-gen-bimodality/`（第五十批，§13.44）已入库**：3 份训练日志＋逐请求/相位 JSONL（gz）＋`summary.json`＋插桩/还原脚本＋md5；服务器 verl 树的插桩已回滚（`llm_server.py`／`ray_trainer.py`／`v1/trainer_base.py`／`v1/trainer_sync.py`／`v1/replay_buffer.py` 五文件 md5 与备份一致，全树无 `_m5_log` 残留）。
- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）已入库（commit `cec0382`），此前"仍未跟踪"的记录已过期。本会话本批（第四十九批）的 `experiments/rollout-g8-candidate-g/`（README 新增小节＋`ab-summary.json`＋10 份训练日志＋微基准/链路日志＋md5sums）已随本批提交。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11／§21.12）是本会话的工作，已入库；本批新增 `fusion_evidence.py`／`test_fusion_evidence.py`／`fusion.json`／`readme-fusion.md`，`run-info.json` 已同时覆盖三批的 sha256。三套脚本共用目录但各自独立归档（`summary.json`／`resolution.json`／`fusion.json`）。
- `experiments/rollout-c-class-ledger/`（§13.43）同日三轮写入：`ledger.py`／`test_ledger.py`／`ledger.json`／`readme.md`／`run-info.json`。第二十四批新增 `gap_pieces`／`traj_ratio`／bootstrap 与三条 superseded 读法；**第二十六批新增 `account_checks`（天花板饱和／离散量符号／共线回退／字段退化／评测网格）与 14 条新单测（含 3 条注入缺陷），`ledger.json` 已重算，`run-info.json` 记 round_3。** 两个输入归档的 sha256 与前几版一致，未改动上游数据。只读引用 `rollout-p3-dapo-group-filter/summary.json` 与 `rollout-p4-c-mechanisms/summary.json`，**未修改那两批的归档**（sha256 已记入本批 run-info.json 的 `input_sha256`，且与上一版一致）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **第二十八批推送异常（2026-11-02 21:0x，需后续会话知晓）**：当时 `github.com:443` 无法直连（多次重试均 75s 超时），`api.github.com` 可达。改用 REST API 逐对象上传（9 blob＋5 tree＋commit）并 PATCH `refs/heads/main`，**内容已完整到达远端**：远端 main `df8df63c` 的 tree `7284f84b` 与本地提交 `350ee33b` 的 tree **逐字节相同**（已核）。但因 API 重建 commit 时提交者时间戳不同，**远端提交 SHA（`df8df63c`）≠ 本地提交 SHA（`350ee33b`）**；本地 `refs/remotes/origin/main` 已指向 `df8df63c` 但对象不在本地库，`git status` 会显示 `[gone]`。**下一次能连 GitHub 时先 `git fetch origin`**，届时会换成真实远端提交、`[gone]` 自动消失；在此之前**不要**用 `git reset --hard origin/main` 或 force push（本地对象缺失，会误伤）。
- **并行会话正在写同一 checkout（2026-11-02 21:07–21:08）**：`diary/2026-11-02.md` 与 `notebook/rollout-performance-research.md` 的工作区改动属于另一个执行者——它在复核本批 §13.46 后订正了自己昨天写的 §20.3.5（把"残差 0.040–0.076s"改判为计时缺口而非舍入）。**本会话未提交、未回退这些改动**，按 AGENTS.md 只提交自己的工作。接续时先看这两份文件的未提交状态归属再动。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
