# perpetual-learning — 执行接续

更新时间：2026-11-03。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M5 本轮（2026-11-03 00:5x–01:4x CST，对应服务器 UTC 2026-11-02 16:5x–17:4x）做了一件此前登记为「未测」的算力事项，**第五十四批 §13.50：真实 rollout 的响应长度结构实测**。现场条件：4090-1 上 liyaowei 的 4 卡 vLLM（GPU 2/3/6/7，100%）＋1 个 EngineCore（GPU 0，20.8 GB）占用，**只有 GPU 1/4/5 空闲**；`/home` 已满（8.1 G 余），`/` 余 188 G，数据全部写 `/tmp`。昇腾 910C VPN 仍断（网关 10.205.0.1 与 192.168.13.154 均 100% 丢包，第 28 天）。

**实验内容**：用与 verl 训练一致的采样口径（T=1.0／top_p=1.0／top_k=−1、`max_tokens=256`、chat template、gsm8k `train_sub480` 全 480 prompt × n=8）在单卡上实采 5 档（1.5B／0.5B × cap256／cap1024＋1.5B seed43 重复），共 19,200 条响应、生成耗时 41–65 s/档。**三个结论**：① 构造档 `realistic=(0.60,0.08)` 隐含 ICC ≈0.98，实测只有 **0.450 [0.409,0.489]（1.5B cap256）／0.364 [0.321,0.407]（0.5B）**，且真实**组内离散大于组间离散**（0.166 vs 0.161），方向与构造档相反；② `max_response=256` 截断 **45.3%（1.5B）／60.8%（0.5B）** 的响应——干净组重算出的 ICC 0.174／0.128 是选择偏差、不可引，无截断读数由 1024 臂给出（0.522／0.394）；③ 回代 §13.38 的两个主读数在 1 SE 内不动：`ldm/grouped` 1.0462→1.044–1.048、`interleaved/ldm` 1.0045→1.0027–1.0065。归档 [experiments/rollout-length-structure-real](experiments/rollout-length-structure-real/README.md)，`verify.py` exit 0。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TM-009（掘金 B 类）四件套完成，待用户决定是否对外 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。TS-001 上一轮为 §13.47 机制账收尾（第二十九批），`experiments/rollout-c-class-ledger/` 四轮完成，再往下需新字段或新实验。 |
| M5 | TS-001 算力支持 | **第五十四批 §13.50 已完成**（见上）。**M5 侧登记的剩余算力事项仍为空**；要再往前只剩需新条件或新立项的：TM-008 的 4 卡 `inflight=1`、`max_inflight_gen_batches>2`、async `wait_for_sampleable`、stale 逐出（前三项要 4 卡以上空闲窗口，本轮只有 3 卡）；P-1 跨系统（AReaL／OpenRLHF／SkyRL，需用户立项）；P-2 FP8 训练侧（待实测）。**下一轮先核对 GPU 归属与占用再启动，不据旧记录假定空闲。** |

## 未完成操作与并行工作

- **TM-009（掘金 B 类，M1 2026-11-03 完成四件套）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效 + 守卫不可达。**对外动作（issue／PR）全部未做，待用户逐项授权**；需用户决定：① 是否提独立小 issue；② 若提 PR，是否连补丁一起（补丁会让上游 `test_python_tool` 变红）。上游 `rllm-org/rllm` main `4e828c81`（2026-11-02 核对，未变）。见 [TM-009](queue_tech-miner.md) 与[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- **TM-008（掘金 C 类，M5）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。2 卡−3.2%、4 卡−2.0%、DAPO inflight=2 −4.9%／inflight=1 −3.4%、4 卡 DAPO A/B −4.0%（§13.49）。补丁 3 文件 +8/−1，默认 2.0 不变，对 `3fdf2b54`／`4b26c788` dry-run 均 3/3 通过。**对外动作全部未做，待用户逐项授权**；需用户决定「只加可达性」还是「同时改默认 0.1–0.2」。**未测**：4 卡 `inflight=1`、`max_inflight_gen_batches>2`、async `wait_for_sampleable`、stale 逐出。
- **TM-006 / TM-005（掘金 B 类）**：tiled_mlp 无消费路径（补丁＋FSDP 冒烟完成）、prefix_grouper 在 FSDP 不生效（取证＋影响量化完成）。**对外动作全部未做，待授权**。
- `experiments/rollout-length-structure-real/`（本轮新增）：`sample_lengths.py`／`analyze_lengths.py`／`reanalysis_length_regime.py`／`verify.py`＋`raw/`（5 份逐样本 jsonl.gz＋meta）＋`summary-*.json`／`reanalysis-*.json`／`run-info.json`／`README.md`。服务器侧原始文件在 `/tmp/m5-length/`（非持久）。
- **推送通道**：先试 `git push`；`github.com:443` 直连历史上多次不通，`gh-proxy.com` 只读，备选 REST API 工具 [tools/git-push-api](tools/git-push-api/readme.md)。`api.github.com`／`gh api` 正常。
- **共享 checkout 被并行会话并发使用**：动手前先看 `git log --graph` 与 `git worktree list`，冲突集中在 CONTEXT.md 与当日日志时按「保留双方条目、按时间重排」解决。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)（§13.50 本轮、§13.38 回代对象）。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 本轮成果：[长度结构实验](experiments/rollout-length-structure-real/README.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
