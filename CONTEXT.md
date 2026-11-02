# perpetual-learning — 执行接续

更新时间：2026-11-02。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M1 本轮做了两件事，都不是 TS-001 的继续。

**一、verl 候选时效性复核（因本地检出落后 `origin/main` 361 个提交）**。TM-008 前提**仍成立**：`4b26c788` 上 `ReplayBuffer.__init__` 仍 `poll_interval: float = 2.0`、`_wait_for_next_poll` 仍 `time.sleep(self.poll_interval)`、`trainer_base.py::_build_replay_buffer` 仍不传该键、`ppo_trainer.yaml` 的 `trainer.v1.sampler` 段仍无该键。**但补丁需重放**：上游 `v1/` 已重构出 `trainer_sync.py`，`experiments/verl-v1-poll-quantum/poll-interval-config.patch` 后两个 hunk 的上下文已变，授权后须先重做再应用。**（M5 23:2x 实测更正：对 main `3fdf2b54` 与 `4b26c788` 两棵树的三个目标文件分别做 `patch -p1 --dry-run`，均 3/3 通过、RC=0、零 fuzz ⇒ 补丁可直接应用，不需要重放；`trainer_sync.py`／`trainer_colocate_async.py` 只是 `PPOTrainer` 薄子类、不覆写 `_build_replay_buffer`，故 base 一处接线覆盖全部 V1 trainer。）**

**二、新掘金候选 TM-009（rLLM，B 类）**。扩池选 rLLM（本地旧快照 06-17 → 拉到活跃的 `4e828c8`），与已有 #605 主题无关。pyflag 判出 2 个"失效"，人工复核后**两个都不是表面看到的问题**：`use_tci` 是全仓零调用者的死代码（非用户可配开关）；`use_async` 是**假阳性**（默认 `False` 但文档写"默认 None 自动探测"，扫描器扫不到 `None` 路径）。顺此读源码找到真缺陷：`has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward`，两半对任何子类**恒真**，故纯同步工具在 `use_async=None`／`True` 下返回**未被 await 的协程**，同一 `if` 里的 `NotImplementedError` 守卫不可达。用真实 `Tool` 类复现（Python 3.12.13），六项断言 exit 0；负向对照（改 `type(self).async_forward is not Tool.async_forward`）三条行为转正确。影响面定级 6/10：11 个工具只实现 `forward`、仅 2 个实现 `async_forward`，但 `use_async=` 命中**全在 `__main__` 演示块**，生产路径不传 ⇒ 现有训练不受影响。**对外动作全部未做**。

TS-001：§13.43 四轮已完成，再往下需新字段或新实验；P-1 缺跨系统、P-2 需 FP8 栈，均待用户立项。本轮未推进，无新实验读数、无 GPU。**M5 本轮（2026-11-02 14:02–15:02 UTC）完成 TM-008 的边界实验（第五十二批，§13.48）**：DAPO 组过滤 + `max_inflight_gen_batches=2` 下六跑，poll 2.0→0.1 使 gen 8.474→7.289s（−14.0%）、整步 26.534→25.246s（−4.9%），但每步多生成 13% 的组（117.4→132.8，surplus +39%）——墙钟仍净降，但"调小就只是省等待"在 DAPO 配方上不成立。其他执行者进程状态未核实。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TM-009（掘金 B 类）+ TS-001 待决 | 最新为 **TM-009 取证完成、待用户决定是否对外**：[取证](open-source/rllm/tool-use-async-autodetect.md)、[复现](experiments/rllm-tool-use-async-probe/readme.md)。**对外前必须先补一步未做核查：检索上游 issue/PR 是否已有人认领**。TS-001 上一轮为 §13.47 机制账收尾（第二十九批），`experiments/rollout-c-class-ledger/` 四轮完成，再往下需新字段或新实验，不在当前授权范围。 |
| M5 | TS-001 算力支持 | **第二会话本轮（2026-11-02 14:58–15:2x UTC）**：GPU 先被上面那轮 DAPO 六跑占用，本会话不抢卡。① `git fetch` 解除推送异常（远端换回真实历史 `80dd6e1..651da19`，随后又推 `c98bf5c`）。② **更正上一段"补丁需重放"**（见 §当前现场 一）。③ 上游 verl main `3fdf2b54` 未变；PR #8313 open、`mergeable_state` MERGEABLE→**unstable**（head `71fd3562` 未变）。④ **独立复核六跑归档**（[verification.md](experiments/verl-v1-poll-quantum-dapo/verification.md)）：换解析路径重算 `summary.json` 全部字段 1e-9 内一致、md5 9/9、`dp200b`／`dp010b` 与服务器 sha256 一致；六份日志无 stale／failed 逐出 ⇒ `generated = 32+evicted+surplus` 完备；两指标计的是 **prompt 组 uid**。回归（共同斜率 46 ms/组）给出：dbase 固定偏移 +0.001s（CI 含 0，**补丁惰性独立成立**）、0.1 为 gen −1.87s／整步 −1.95s（CI 下界约 −1.4s）；原始整步差 −1.29s ≈ 固定等待 −1.95s ＋ 多生成 +0.71s（残差 0.05s），两笔账闭合；跨跑极差 2.0 臂 0.85s vs 0.1 臂 0.08s。⑤ **4 卡（2+2）DAPO 基线已测未归档**：15:03:56–15:11:23，15/15 exit 0，整步 18.518±0.78s／gen 6.520±0.91s／约 122 组每步；第二臂 0.1 与并行会话新开的 inflight=1 基线在 GPU 0,1 撞车，**已主动 kill 让路**，A/B 未完成，基线尾段有重叠（ms/组 58.7 vs 55.2）。⑥ 昇腾 910C VPN 复探仍断（.154／.174 100% 丢包）。**本轮（2026-11-02 14:02–15:02 UTC）完成 TM-008 边界实验（第五十二批，§13.48）**：把 TM-008 补丁自己标注的未测项"DAPO 过滤／refill 下采样时机可能改变可选组集合"补掉——4090-1 2 卡、DAPO `filter_groups`＋`max_inflight_gen_batches=2`、六跑（原树 2.0／patched 2.0／0.5／0.1＋2.0、0.1 各一次重复）全部 exit 0。gen 8.474→7.289s（−14.0%）、整步 26.534→25.246s（−4.9%），其余相位 ±0.1%、补丁惰性成立、逐 step sd 1.5→0.7s；**但每步多生成 13% 的组**（117.4→132.8，surplus +39%，约 2 SE），墙钟仍净降而两项同量级。并行会话独立复核把两笔账对齐（46.0 ms/组；固定等待 −1.87／−1.95s、多生成 +0.707s、残差 0.05s）。顺带记 `filter_groups.metric` 注释写 `seq_final_reward` 但 V1 只产出 `acc`。归档 `experiments/verl-v1-poll-quantum-dapo/`。**仍用 `24f25b03` 旧树**，上游 `4b26c788` 的 `v1/` 重构结论未复核。**M5 登记的剩余算力事项仍为空**（本轮为自主选题的 C 类边界实验）。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-009（掘金 B 类，M1 2026-11-02 23:0x）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效。**对外动作（issue／PR）全部未做，待用户逐项授权**；另需用户决定"只提独立小 issue"还是"连 PR 一起"（后者含签名默认改 `None` 的行为变化）。见 [TM-009](queue_tech-miner.md) 与 [取证](open-source/rllm/tool-use-async-autodetect.md)。
- **TM-008（掘金 C 类，M5 2026-11-02 13:0x–13:5x UTC；边界实验 14:02–15:02 UTC）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。2 卡整步 21.686→21.001s（−3.2%）、4 卡 14.216→13.934s（−2.0%）；**0.5 不是通用解**。补丁 3 文件 +8/−1，**默认 2.0 不变**。**2026-11-02 复核前提仍成立，但补丁需按新 `v1/` 结构（`trainer_sync.py`）重放**（M5 23:2x 更正：不需要——dry-run 在 main `3fdf2b54` 与 `4b26c788` 上都 3/3 通过，见 §当前现场 一）。**边界实验（第五十二批，§13.48）**：DAPO 过滤 + `max_inflight_gen_batches=2` 下收益更大（gen −14.0%、整步 −4.9%），但每步多生成 13% 的组（约 2 SE）——PR 风险段必须写这一条。**对外动作全部未做，待用户逐项授权**；另需用户决定"只加可达性"还是"同时改默认 0.1–0.2"。
- **TM-006（掘金 B 类）**：verl `tiled_mlp` 十引擎无消费点。[补丁就绪](open-source/verl/patches/tiled-mlp-wiring/readme.md)；真机冒烟完成（FSDP2 生效、FSDP1 硬报错、默认梯度检查点下无显存收益）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类）**：verl `use_prefix_grouper` 在 FSDP 上不生效但副作用照常。取证与影响量化（DP 负载不均衡 0–6.6%）已入库。**对外动作全部未做，待用户逐项授权**。
- `experiments/rllm-tool-use-async-probe/`（TM-009，本轮新增）已入库：`repro.py`／`readme.md`／`run-info.json`。复现依赖 uv 建的 Python 3.12 venv（系统 3.9.6 因 PEP 604 无法导入 rllm），依赖清单写在 `run-info.json`。
- `tools/pyflag/` 本轮新增 `scan-rllm-4e828c8.json`，readme 加第五类易错点"三态形参"。
- **`experiments/verl-v1-poll-quantum/`、`verl-tiled-mlp-fsdp-smoke/`、`rollout-g8-gen-bimodality/`、`rollout-c-class-ledger/` 等均已入库**，不重复登记。
- **`experiments/verl-v1-poll-quantum-dapo/`（M5，第五十二批 §13.48）已入库**：6 份训练日志（`raw/*.txt`）＋`summary.json`＋`run-poll-dapo.sh`＋`make-summary-dapo.py`＋`md5sums.txt`＋`verification.md`（并行会话独立复核）。服务器跑树 `/tmp/m5-poll-ab/verl`（patched）与原树 `/home/zxw/venvs/rollout-g8-src/verl` 均保留。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **推送通道现状（2026-11-02 23:2x）**：`github.com` 直连正常，`git fetch`／`push` 可用；远端 main 已是真实历史（不再是 API 重放副本）。上游 verl 的 GitHub main 实测为 `3fdf2b54`（11-02T11:38Z），**不是** `4b26c788`——后者是 M1 本地 checkout 的 origin 快照（10-18）。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 本轮成果：[TM-009 取证](open-source/rllm/tool-use-async-autodetect.md)、[复现](experiments/rllm-tool-use-async-probe/readme.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
