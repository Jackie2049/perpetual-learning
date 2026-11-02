# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M1 本轮先**消解上轮的推送异常**（`github.com:443` 已恢复）：`git fetch` 成功，远端比本地多 2 个提交（M5 的 TM-008 真机实验），本地多 1 个未推提交；rebase 到 `origin/main` 后只有 `diary/2026-11-02.md` 一处冲突（两份不同时间段的日志条目各自追加），**两边全保留**并按时间重排（21:36 在前、21:55 在后），推送到 `614c8b6e6`。远端 `[gone]` 问题随之消失，**本地 refs 与远端一致，不再需要 REST 上传路径**。

然后完成 **TS-001 第二十九批：C 类机制账的收尾（§13.47）**——把 §13.43 ⑩ 留下的四处"未定"逐条判成"测不出／可定／可报符号"。四处结论：① `cancel` 的 III 类分项**结构性测不出**（p4 归档只有"生成多少 token"一个成本字段，`step_costs` 全是 112＝剩余任务×14 的整数倍，叠加 `total_gen ≡ total_cost`，发起次数从未被单独记录）；② `C vs B` 差额归属**同样读不出**（p3 逐种子 `upd_share ≡ 1.0`，生成侧被 `max(生成,更新)` 完全吸收，`Δgen ≡ 0` 是代数结果）；③ 阈值离散化**可定且比原先设想的更稳**（到点余量是网格的 p4 3.0–4.0／p3 1.9–3.6 倍，故"撞网格"改写为"分辨力上限 ±1 步、触发需 2–4 格抖动"）；④ 天花板下换曲线面积，符号可报（`freq` +1.385、`cancel` −0.476、`cap` +0.165，`replay` 跨 0）但量级只是基线墙钟的 0.0012–0.0104，`cap` 墙钟仍不归因。中途**自查订正一处**：初稿把"余量"与"网格"比反了（写"p3 不到一格"，实为网格的 1.9–3.6 倍），数值不变、判断已改。新口径纪律：**到收敛每步成本比 ≠ 全部步 `cost_ratio`**（`C/B` 1.063 vs 0.551，方向相反），已加单测钉住。

无 GPU、无新实验读数；只读两份既有归档。M5 本轮无新工作（它上一轮完成了 TM-008）。其他执行者的进程状态未核实，以下仅为已有记录。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 rollout 专项 | 最新为 **§13.47 机制账收尾（第二十九批）**：材料 `experiments/rollout-c-class-ledger/`（新增 `resolution.py`／`test_resolution.py`／`resolution.json`，`readme.md` 加第四批一节，`run-info.json` 记 round_4）。**§13.43 这条线（立账→订正→口径检查→收尾）四轮做完**，⑩ 的每处"未定"都有明确归因；再往下都需新字段或新实验（`cancel` 分项要插桩加发起次数／调度等待字段；生成/更新分解要能拆两侧的成本字段；`cap` 归因要更密评测集），**不在当前授权范围**。此前为 §11.4＋§13.46（第二十八批，四耦合面判据＋会计面实测）／§20.3.5／§13.43 第二十六批／第二十四批。 |
| M5 | TS-001 算力支持 | **上一轮完成了 TM-008 真机实验**（verl V1 采样等待的 `poll_interval` 不可配置，4090-1：2 卡 −3.0~3.2%、4 卡 −2.0%，补丁默认 2.0 不变；归档 `experiments/verl-v1-poll-quantum/`）。**M5 登记的剩余算力事项仍为空**。本轮无新工作。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-008（掘金 C 类，M5 2026-11-02 13:0x–13:5x UTC）**：verl V1 sync trainer 的 `ReplayBuffer.poll_interval` 是**死值 2.0s**——`TaskRunnerV1._build_replay_buffer` 不传、`ppo_trainer.yaml` 无该键、文档未列，用户只能写自定义 sampler 子类。同窗口 A/B：2 卡整步 21.686→21.001s（−3.2%）、gen 3.0→2.2s；4 卡 14.216→13.934s（−2.0%）。**0.5 不是通用解**（4 卡生成跨度约 1.6s，仍需 0.1–0.2）。补丁 3 文件 +8/−1，**默认 2.0 不变**。**对外动作全部未做，待用户逐项授权**；另需用户先决定"只加可达性（默认 2.0）"还是"同时改默认 0.1–0.2"。见 [TM-008](queue_tech-miner.md) 与 [experiment](experiments/verl-v1-poll-quantum/README.md)。
- **TM-006（掘金 B 类，M1 2026-11-02 17:3x；M5 同日 12:2x UTC 补真机冒烟）**：verl `tiled_mlp` 配置在 `verl/workers/engine/` 下 10 个引擎全部无消费点。工具沉淀为 [tools/pyflag](tools/pyflag/readme.md)。**本地补丁已就绪**：[patches/tiled-mlp-wiring](open-source/verl/patches/tiled-mlp-wiring/readme.md)（3 行接线）。**真机验证已完成**：FSDP2 下真实生效、fp64 精确等价；**FSDP1 下开启硬报错**（需在 PR 里补配置期校验）；**默认梯度检查点开启时无显存收益**。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类，M1 2026-11-02 16:1x）**：verl `use_prefix_grouper` 在 FSDP 引擎上不生效但副作用照常生效；取证材料与影响量化（DP 负载不均衡 0–6.6%）均已入库，**对外动作（issue／评论／PR）全部未做，待用户逐项授权**。见 [TM-005](queue_tech-miner.md) 与 [evidence](open-source/verl/evidence/prefix-grouper-inert/readme.md)。

- **`experiments/verl-v1-poll-quantum/`（M5，TM-008）已入库**：6 份训练日志（`raw/*.txt`）＋`summary.json`＋补丁＋应用/分析脚本＋`md5sums.txt`＋`run-info.json`。md5 与服务器逐文件一致。
- **`experiments/verl-tiled-mlp-fsdp-smoke/`（M5，TM-006 冒烟）已入库**：`smoke_tiled_mlp.py`／`cpu_equiv.py`／`run_smoke.sh`／`summarize.py`＋`raw/`（32 份 JSON）＋`logs/`（29 份）＋`md5sums.txt`。服务器受测树在 `/tmp/m5-tiled-mlp/`，未占用 `/home`。
- **`experiments/rollout-g8-gen-bimodality/`（第五十批，§13.44）已入库**：3 份训练日志＋逐请求/相位 JSONL（gz）＋`summary.json`＋插桩/还原脚本＋md5；服务器 verl 树的插桩已回滚（五文件 md5 与备份一致，全树无 `_m5_log` 残留）。
- `experiments/rollout-g8-candidate-h/`（第四十八批候选 H，§13.41）已入库（commit `cec0382`）。`experiments/rollout-g8-candidate-g/`（第四十九批）已入库。
- `experiments/rollout-bubble-attribution/`（§21.10／§21.11／§21.12）已入库，含 `fusion_evidence.py`／`test_fusion_evidence.py`／`fusion.json`／`readme-fusion.md`，`run-info.json` 覆盖三批 sha256。
- `experiments/rollout-c-class-ledger/`（§13.43／§13.47）**四轮写入**：`ledger.py`／`test_ledger.py`（30 tests）／`ledger.json`，本轮新增 `resolution.py`／`test_resolution.py`（17 tests）／`resolution.json`，`readme.md` 加第四批一节，`run-info.json` 记 `round_4` 与新增 artifacts 的 sha256。**两个输入归档的 sha256 与前几版一致**（`rollout-p3-dapo-group-filter/summary.json` = `662682fc…`、`rollout-p4-c-mechanisms/summary.json` = `d88356ba…`），未改动上游数据。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **推送通道现状（2026-11-02 21:5x 复核）**：`github.com:443` 已恢复，常规 `git fetch`／`git push` 可用；本地 `refs/remotes/origin/main` 已与远端一致。**上轮用 REST API 上传造成的"远端历史是本地重放副本、SHA 不同"的状态已结束**——本轮的 rebase + push 把本地提交并入了真实远端历史。若再遇直连失败，按上轮经验用 `api.github.com` 逐对象上传，但**事后必须 `git fetch` 重建本地 refs**，且不要 `reset --hard` 到尚未 fetch 的远端引用上。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
