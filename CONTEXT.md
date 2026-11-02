# perpetual-learning — 执行接续

更新时间：2026-11-03。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

M1 本轮（2026-11-03 00:0x）把 TM-009 从"已取证待决定"推到"材料齐备待决定"。三件事，都不是 TS-001 的继续：

**一、生产可达性结构化核实**。此前"生产路径不触达"只是 grep 印象。新建 `experiments/rllm-tool-use-async-probe/prod_reachability.py`（四段检查、exit 0）：AST 扫描全包，`use_async=` 在 `if __name__ == "__main__":` 块之外 **0 处**；块内 4 处取值全是 `True`；生产构造点 `code_reward.py` 调 `codetool(code=..., timeout=30)` 走默认同步正常返回、改传 `None` 即刻返回 coroutine。

**二、查清 rLLM 有两套互不相干的工具栈**（此前混着说会误导影响面）：`rllm/tools/` 的 `Tool` 子类走 `__call__` 三元路由；`rllm/harnesses/tools/` 的 `BashTool` 等是**普通类**，协议 `.schema` + `.execute(sandbox, ...)`，不继承 `Tool`。agent 主流路径结构上绕开本缺陷。

**三、候选补丁就绪，并暴露一个比缺陷更有说服力的事实**。补丁只改 `has_async` 一行（`self.__class__.async_forward is not Tool.async_forward`）+ 三条常量断言测试；修复后 3 passed、未修复 2 failed（负向对照有效）。但**上游既有的 `test_python_tool` 由绿转红**——它拿纯同步的 `PythonInterpreter` 调 `use_async=True`，过去靠缺陷吞掉才"通过"。那处未擅自改（涉及上游测试意图），留给维护者。价值评分**维持 6 不因加补丁上调**。

TS-001：§13.43 四轮已完成，再往下需新字段或新实验；P-1 缺跨系统、P-2 需 FP8 栈，均待用户立项。**M5 并行会话已归档 4 卡（2+2）DAPO poll 基线**（`experiments/verl-v1-poll-quantum-dapo-g4/`，A/B 因并发推迟）。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TM-009（掘金 B 类）四件套完成，待用户决定是否对外 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。TS-001 上一轮为 §13.47 机制账收尾（第二十九批），`experiments/rollout-c-class-ledger/` 四轮完成，再往下需新字段或新实验，不在当前授权范围。 |
| M5 | TS-001 算力支持 | **第二会话本轮（2026-11-02 14:58–15:2x UTC）**：GPU 先被上面那轮 DAPO 六跑占用，本会话不抢卡。① `git fetch` 解除推送异常（远端换回真实历史 `80dd6e1..651da19`，随后又推 `c98bf5c`）。② **更正上一段"补丁需重放"**（见 §当前现场 一）。③ 上游 verl main `3fdf2b54` 未变；PR #8313 open、`mergeable_state` MERGEABLE→**unstable**（head `71fd3562` 未变）。④ **独立复核六跑归档**（[verification.md](experiments/verl-v1-poll-quantum-dapo/verification.md)）：换解析路径重算 `summary.json` 全部字段 1e-9 内一致、md5 9/9、`dp200b`／`dp010b` 与服务器 sha256 一致；六份日志无 stale／failed 逐出 ⇒ `generated = 32+evicted+surplus` 完备；两指标计的是 **prompt 组 uid**。回归（共同斜率 46 ms/组）给出：dbase 固定偏移 +0.001s（CI 含 0，**补丁惰性独立成立**）、0.1 为 gen −1.87s／整步 −1.95s（CI 下界约 −1.4s）；原始整步差 −1.29s ≈ 固定等待 −1.95s ＋ 多生成 +0.71s（残差 0.05s），两笔账闭合；跨跑极差 2.0 臂 0.85s vs 0.1 臂 0.08s。⑤ **4 卡（2+2）DAPO 基线已归档、A/B 推迟**：[4 卡基线](experiments/verl-v1-poll-quantum-dapo-g4/README.md)。15:03:56–15:11:23，15/15 exit 0，整步 18.518±0.78s／gen 6.520±0.91s／约 114.8 组每步；2→4 卡整步 1.43×（gen 1.30×、update_weights 0.98×、update_actor 1.71×），gen 占比 32.0%→35.2%。第二臂启动即与并行会话新开的 inflight=1 跑撞车，**主动 kill 让路**；此后对方又开重复跑，三次相撞（两次 vLLM KV 配额、一次 memory-profiling），均死在启动阶段。15:42 起停止抢卡。基线尾段与对方启动重叠（ms/组 58.7 vs 55.2），18.518s 记为上界。**下一轮第一优先：独占窗口重跑两臂。**⑥ 昇腾 910C VPN 复探仍断（.154／.174 100% 丢包）。**本轮（2026-11-02 14:02–15:02 UTC）完成 TM-008 边界实验（第五十二批，§13.48）**：把 TM-008 补丁自己标注的未测项"DAPO 过滤／refill 下采样时机可能改变可选组集合"补掉——4090-1 2 卡、DAPO `filter_groups`＋`max_inflight_gen_batches=2`、六跑（原树 2.0／patched 2.0／0.5／0.1＋2.0、0.1 各一次重复）全部 exit 0。gen 8.474→7.289s（−14.0%）、整步 26.534→25.246s（−4.9%），其余相位 ±0.1%、补丁惰性成立、逐 step sd 1.5→0.7s；**但每步多生成 13% 的组**（117.4→132.8，surplus +39%，约 2 SE），墙钟仍净降而两项同量级。并行会话独立复核把两笔账对齐（46.0 ms/组；固定等待 −1.87／−1.95s、多生成 +0.707s、残差 0.05s）。顺带记 `filter_groups.metric` 注释写 `seq_final_reward` 但 V1 只产出 `acc`。归档 `experiments/verl-v1-poll-quantum-dapo/`。**仍用 `24f25b03` 旧树**，上游 `4b26c788` 的 `v1/` 重构结论未复核。**M5 登记的剩余算力事项仍为空**（本轮为自主选题的 C 类边界实验）。按分工核对占用及实验归属后接续，不能仅据旧记录启动。 |

## 未完成操作与并行工作

- **TM-009（掘金 B 类，M1 2026-11-03 完成四件套）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效 + 守卫不可达。**对外动作（issue／PR）全部未做，待用户逐项授权**；需用户决定：① 是否提独立小 issue；② 若提 PR，是否连补丁一起（补丁会让上游 `test_python_tool` 变红，需在 PR 里说明或顺带修）。见 [TM-009](queue_tech-miner.md) 与[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- **TM-008（掘金 C 类，M5 2026-11-02 13:0x–13:5x UTC；边界实验 14:02–15:02 UTC）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。2 卡整步 21.686→21.001s（−3.2%）、4 卡 14.216→13.934s（−2.0%）；**0.5 不是通用解**。补丁 3 文件 +8/−1，**默认 2.0 不变**。**2026-11-02 复核前提仍成立，但补丁需按新 `v1/` 结构（`trainer_sync.py`）重放**（M5 23:2x 更正：不需要——dry-run 在 main `3fdf2b54` 与 `4b26c788` 上都 3/3 通过，见 §当前现场 一）。**边界实验（第五十二批，§13.48）**：DAPO 过滤 + `max_inflight_gen_batches=2` 下收益更大（gen −14.0%、整步 −4.9%），但每步多生成 13% 的组（约 2 SE）；补测 DAPO 默认 `inflight=1` 四跑（池化 gen −13.0%、整步 −3.4%、生成组 +5.3%）⇒ **生成量代价随 in-flight 余量缩小（13%→5%）**——PR 风险段必须写这一条。**对外动作全部未做，待用户逐项授权**；另需用户决定"只加可达性"还是"同时改默认 0.1–0.2"。
- **TM-006（掘金 B 类）**：verl `tiled_mlp` 十引擎无消费点。[补丁就绪](open-source/verl/patches/tiled-mlp-wiring/readme.md)；真机冒烟完成（FSDP2 生效、FSDP1 硬报错、默认梯度检查点下无显存收益）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类）**：verl `use_prefix_grouper` 在 FSDP 上不生效但副作用照常。取证与影响量化（DP 负载不均衡 0–6.6%）已入库。**对外动作全部未做，待用户逐项授权**。
- `experiments/rllm-tool-use-async-probe/`（TM-009）已入库：`repro.py`／`prod_reachability.py`（本轮新增）／`readme.md`／`run-info.json`；`open-source/rllm/patches/use-async-autodetect/`（本轮新增）含补丁与 readme。复现依赖 uv 建的 Python 3.12 venv（系统 3.9.6 因 PEP 604 无法导入 rllm），依赖清单写在 `run-info.json`。
- `tools/pyflag/` 本轮新增 `scan-rllm-4e828c8.json`，readme 加第五类易错点"三态形参"。
- **`experiments/verl-v1-poll-quantum/`、`verl-tiled-mlp-fsdp-smoke/`、`rollout-g8-gen-bimodality/`、`rollout-c-class-ledger/` 等均已入库**，不重复登记。
- **`experiments/verl-v1-poll-quantum-dapo/`（M5，第五十二批 §13.48）已入库**：10 份训练日志（`raw/*.txt`，inflight=2 六份＋inflight=1 四份）＋`summary.json`／`summary-inflight1.json`＋`run-poll-dapo.sh`＋`make-summary-dapo.py`＋`md5sums.txt`＋`verification.md`（并行会话独立复核）。服务器跑树 `/tmp/m5-poll-ab/verl`（patched）与原树 `/home/zxw/venvs/rollout-g8-src/verl` 均保留。**同目录的 4 卡补充**见 `experiments/verl-v1-poll-quantum-dapo-g4/`（第二会话，仅 2.0 基线，A/B 因抢卡推迟）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **推送通道现状（2026-11-03 00:3x）**：`github.com:443` 直连**不通**（两次 75s 超时），`gh-proxy.com` 只代理**只读**（push 报 `No anonymous write access`），`api.github.com` 与 `gh api` 正常。本轮改用 REST API 推送（[tools/git-push-api](tools/git-push-api/readme.md)），远端 main 已推进到 `60869da2` 并与本地对齐。**下次运行先试 `git push`**，仍不通再用该工具；工具会压平本地多层提交。上游 verl 的 GitHub main 实测为 `3fdf2b54`（11-02T11:38Z），**不是** `4b26c788`——后者是 M1 本地 checkout 的 10-18 快照。
- **共享 checkout 已被并行会话并发使用**：本轮两次出现另一会话在同一目录 commit／reset（TM-008 inflight=1 补测、`01de74f8a` 订正传播）。合并冲突集中在 CONTEXT.md 与当日日志，已按"保留双方条目、按时间重排"解决。**动手前先看 `git log --graph` 与 `git worktree list`**，不要假设 HEAD 还是自己的。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 本轮成果：[TM-009 候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[取证](open-source/rllm/tool-use-async-autodetect.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
