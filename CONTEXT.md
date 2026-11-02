# perpetual-learning — 执行接续

更新时间：2026-11-03（M1 02:0x）。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

### M1 本轮（2026-11-03 02:0x）：TS-001 补强——锚② 的可换算性复核（§20.3.6／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，按其"在已有范围内筛选仍有明确新增信息的补强项"继续，选的是 §20.3.5 里一句一直被当已核实的话。

**结论**：把 `s_k` 与 `G_k` 代回 §13.25 写下的乘积式、与**实测整步比**对照（这一步此前从未做过）——2 卡重建 **1.4485 vs 1.5264（−5.1%）**、4 卡 **1.6429 vs 1.9419（−15.4%）**；且该式在 `G_k≡1` 时给 **1** 而非 `T1/TN`，说明它度量的是**单配置内的相对敏感性**，不是两配置的整步秒数比。差额**是形式问题不是口径问题**：份额随配置移动（`update_actor` 占分项和 0.484→0.450、`update_weights` 0.066→0.166），而"取哪一侧份额当权重"只值 0.2–0.5%（§13.46）。真正闭合的账是"**逐阶段秒数相加 + 计时缺口**"，缺口在 8 个臂上 **0.23–0.31%**。

**改了什么**：锚② 收窄为"分阶段中介"、强度由「强」降「中强」；新增投稿前必补项 **P-6**（给出能重建整步比的形式）；会计面判据的可证伪形式改写进 §11.6。**没改什么**：§13.21–§13.27 任何读数、§13.45／§13.46 结论、§20.3.4 对 P-1 的判断均不动——本轮否定的是一个**读法**，不是测量值。

**核对产物**：两套多卡锚点（2+2 与 2+0／3+1）的分阶段 `G_k` 已并列入 `experiments/rollout-anchor2-identity/summary.json`，写作时从该文件取、不转述（§20.3.5 引用的五对本身无错配，风险在转述环节）。

入口：[§20.3.6](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### 上一轮（2026-11-03 01:0x–01:3x）：新登记 TM-010

技术秘书侧无新用户任务；TM-001～TM-009 的对外动作均待用户逐项授权、无并行会话占用，按规则从掘金 B 类选题。选了 rLLM 的**训练路径**（TM-009 是同一仓库的工具层）。

**结论**：Fireworks 后端自 PR #934（10-09）默认 `router_replay: R3`，但路由矩阵装不进训练帧——`transform.py:85-87` 用 `model_copy(update=...)` 塞的属性不在 `tinker.ModelInput` 声明字段里（只有 `chunks`、`from_ints()` 无该参数、pydantic frozen+forbid），赋值抛 `ValidationError`。三个 tinker 版本一致。上游 cookbook 用的 `from_ints(routing_matrices=...)` 在所有可用 tinker 版本里也不存在 ⇒ **不是 rLLM 单方面能补的**。

**排除了两条先怀疑的方向**（不写进结论）：多步矩阵布局错位（两形状实测都对齐）、verl 蒸馏 `compute_logprobs` 缺失（实为已定义）。

**未验证**：Fireworks 服务端对未声明属性是忽略还是报错；是否已影响线上训练。**不主张训练出错或性能损失。**

入口：[取证](open-source/rllm/fireworks-r3-routing-matrices.md) · [复现](experiments/rllm-fireworks-r3-replay/readme.md) · [TM-010](queue_tech-miner.md)。对外动作全部未做。

### 上一轮（2026-11-03 00:0x）：TM-009 材料齐备

M1 上一轮把 TM-009 从"已取证待决定"推到"材料齐备待决定"：生产可达性结构化核实（`prod_reachability.py`，`use_async=` 在 `__main__` 外 0 处）、查清 rLLM 两套工具栈、候选补丁就绪并暴露"上游 `test_python_tool` 靠缺陷吞异常才通过"。细节见[取证](open-source/rllm/tool-use-async-autodetect.md)。

### M5 本轮（2026-11-03 00:5x–01:4x CST）：第五十四批 §13.50

M5 本轮（2026-11-03 00:5x–01:4x CST，对应服务器 UTC 2026-11-02 16:5x–17:4x）做了一件此前登记为「未测」的算力事项，**第五十四批 §13.50：真实 rollout 的响应长度结构实测**。现场条件：4090-1 上 liyaowei 的 4 卡 vLLM（GPU 2/3/6/7，100%）＋1 个 EngineCore（GPU 0，20.8 GB）占用，**只有 GPU 1/4/5 空闲**；`/home` 已满（8.1 G 余），`/` 余 188 G，数据全部写 `/tmp`。昇腾 910C VPN 仍断（网关 10.205.0.1 与 192.168.13.154 均 100% 丢包，第 28 天）。


**实验内容**：用与 verl 训练一致的采样口径（T=1.0／top_p=1.0／top_k=−1、`max_tokens=256`、chat template、gsm8k `train_sub480` 全 480 prompt × n=8）在单卡上实采 5 档（1.5B／0.5B × cap256／cap1024＋1.5B seed43 重复），共 19,200 条响应、生成耗时 41–65 s/档。**三个结论**：① 构造档 `realistic=(0.60,0.08)` 隐含 ICC ≈0.98，实测只有 **0.450 [0.409,0.489]（1.5B cap256）／0.364 [0.321,0.407]（0.5B）**，且真实**组内离散大于组间离散**（0.166 vs 0.161），方向与构造档相反；② `max_response=256` 截断 **45.3%（1.5B）／60.8%（0.5B）** 的响应——干净组重算出的 ICC 0.174／0.128 是选择偏差、不可引，无截断读数由 1024 臂给出（0.522／0.394）；③ 回代 §13.38 的两个主读数在 1 SE 内不动：`ldm/grouped` 1.0462→1.044–1.048、`interleaved/ldm` 1.0045→1.0027–1.0065。归档 [experiments/rollout-length-structure-real](experiments/rollout-length-structure-real/README.md)，`verify.py` exit 0。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 第三十一批（§20.3.6／§11.6）完成；TM-010／TM-009 待用户决定是否对外 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。TS-001 上一轮为 §13.47 机制账收尾（第二十九批），`experiments/rollout-c-class-ledger/` 四轮完成，再往下需新字段或新实验，不在当前授权范围。 |
| M5 | TS-001 算力支持 | **第五十四批 §13.50 已完成**（见上）。**M5 侧登记的剩余算力事项仍为空**；要再往前只剩需新条件或新立项的：TM-008 的 4 卡 `inflight=1`、`max_inflight_gen_batches>2`、async `wait_for_sampleable`、stale 逐出（前三项要 4 卡以上空闲窗口，本轮只有 3 卡）；P-1 跨系统（AReaL／OpenRLHF／SkyRL，需用户立项）；P-2 FP8 训练侧（待实测）。**下一轮先核对 GPU 归属与占用再启动，不据旧记录假定空闲。** |

## 未完成操作与并行工作

- **推送状态（2026-11-03 02:4x）**：`git fetch` 与本机到 github.com 直连仍不通，本轮改用 [git-push-api](tools/git-push-api/readme.md) 经 REST API 推送成功——**远端 main 已是 `198bb82e`，内容包含本地全部提交**（tree 哈希 `f84f0253…` 与本地 HEAD 逐位相同）。副作用：远端与本地**提交链 SHA 分叉**（API 重建了 commit 对象），`git status` 仍显示 `ahead 4`，**这是显示问题不是数据问题**。下轮网络恢复后执行 `git fetch origin && git reset --hard origin/main` 对齐；**不要**手工写 commit 对象（本轮试过，父提交缺失导致 `Failed to traverse parents`，已回滚）。

- **TS-001 第三十一批（M1 2026-11-03 02:0x）**：§20.3.6／§11.6 已落笔；`experiments/rollout-anchor2-identity/` 已入库（`check_identity.py --verify` exit 0、`negative_control.py` exit 0）。**推送（2026-11-03 02:1x–02:2x）**：`git push` 直连 github.com:443 仍不通（75s 超时），用 [tools/git-push-api](tools/git-push-api/readme.md) 分两次推到远端 main `9bc7cd05`；本地为 `b1c8e4786`＋`5d0e48b86`（树相同、SHA 不同，API 以远端父提交重建）。**两次推送合起来才完整**——第一次推的是工作树，第二次补上了收尾写进 CONTEXT／diary 的两处；逐文件 blob 已核对一致。**下次运行先 `gh api` 取远端 sha，本地未对齐属已知偏差**。
  **下一次触碰锚② 前必须知道的**：不能写"恒等式已获同栈实测校验"——乘积式重建不出整步比，`G_k≡1` 时给 1。可发表写法是"逐阶段秒数相加 + 计时缺口"。
  **P-6 是新增的投稿前必补项**（给出能重建整步比的形式），属写作与分析，不需算力。
- **TM-010（掘金 B 类，M1 2026-11-03 01:3x）**：rLLM Fireworks 后端默认 `router_replay: R3`，但路由矩阵装不进 `tinker.ModelInput`（无声明字段 / `from_ints` 无该参数 / 赋值被 pydantic 拒）。**未验证项：服务端行为与线上影响**，不得主张训练出错。对外动作全部未做，待用户决定是否提 issue（以及是否连"默认改回 disabled"一起）。见 [TM-010](queue_tech-miner.md)。
- **TM-009（掘金 B 类，M1 2026-11-03 完成四件套）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效 + 守卫不可达。**对外动作（issue／PR）全部未做，待用户逐项授权**；需用户决定：① 是否提独立小 issue；② 若提 PR，是否连补丁一起（补丁会让上游 `test_python_tool` 变红，需在 PR 里说明或顺带修）。见 [TM-009](queue_tech-miner.md) 与[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- **TM-008（掘金 C 类，M5 2026-11-02 13:0x–13:5x UTC；边界实验 14:02–15:02 UTC）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。2 卡整步 21.686→21.001s（−3.2%）、4 卡 14.216→13.934s（−2.0%）；**0.5 不是通用解**。补丁 3 文件 +8/−1，**默认 2.0 不变**。**2026-11-02 复核前提仍成立，但补丁需按新 `v1/` 结构（`trainer_sync.py`）重放**（M5 23:2x 更正：不需要——dry-run 在 main `3fdf2b54` 与 `4b26c788` 上都 3/3 通过，见 §当前现场 一）。**边界实验（第五十二批，§13.48）**：DAPO 过滤 + `max_inflight_gen_batches=2` 下收益更大（gen −14.0%、整步 −4.9%），但每步多生成 13% 的组（约 2 SE）；补测 DAPO 默认 `inflight=1` 四跑（池化 gen −13.0%、整步 −3.4%、生成组 +5.3%）⇒ **生成量代价随 in-flight 余量缩小（13%→5%）**——PR 风险段必须写这一条。**对外动作全部未做，待用户逐项授权**；另需用户决定"只加可达性"还是"同时改默认 0.1–0.2"。
- **TM-006（掘金 B 类）**：verl `tiled_mlp` 十引擎无消费点。[补丁就绪](open-source/verl/patches/tiled-mlp-wiring/readme.md)；真机冒烟完成（FSDP2 生效、FSDP1 硬报错、默认梯度检查点下无显存收益）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类）**：verl `use_prefix_grouper` 在 FSDP 上不生效但副作用照常。取证与影响量化（DP 负载不均衡 0–6.6%）已入库。**对外动作全部未做，待用户逐项授权**。
- `experiments/rllm-tool-use-async-probe/`（TM-009）已入库：`repro.py`／`prod_reachability.py`（本轮新增）／`readme.md`／`run-info.json`；`open-source/rllm/patches/use-async-autodetect/`（本轮新增）含补丁与 readme。复现依赖 uv 建的 Python 3.12 venv（系统 3.9.6 因 PEP 604 无法导入 rllm），依赖清单写在 `run-info.json`。
- `experiments/rollout-length-structure-real/`（本轮新增）：`sample_lengths.py`／`analyze_lengths.py`／`reanalysis_length_regime.py`／`verify.py`＋`raw/`（5 份逐样本 jsonl.gz＋meta）＋`summary-*.json`／`reanalysis-*.json`／`run-info.json`／`README.md`。服务器侧原始文件在 `/tmp/m5-length/`（非持久）。
- `tools/pyflag/` 本轮新增 `scan-rllm-4e828c8.json`，readme 加第五类易错点"三态形参"。
- **`experiments/verl-v1-poll-quantum/`、`verl-tiled-mlp-fsdp-smoke/`、`rollout-g8-gen-bimodality/`、`rollout-c-class-ledger/` 等均已入库**，不重复登记。
- **`experiments/verl-v1-poll-quantum-dapo/`（M5，第五十二批 §13.48）已入库**：10 份训练日志（`raw/*.txt`，inflight=2 六份＋inflight=1 四份）＋`summary.json`／`summary-inflight1.json`＋`run-poll-dapo.sh`＋`make-summary-dapo.py`＋`md5sums.txt`＋`verification.md`（并行会话独立复核）。服务器跑树 `/tmp/m5-poll-ab/verl`（patched）与原树 `/home/zxw/venvs/rollout-g8-src/verl` 均保留。**同目录的 4 卡补充**见 `experiments/verl-v1-poll-quantum-dapo-g4/`（第二会话，仅 2.0 基线，A/B 因抢卡推迟）。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **推送通道现状（2026-11-03 00:3x）**：`github.com:443` 直连**不通**（两次 75s 超时），`gh-proxy.com` 只代理**只读**（push 报 `No anonymous write access`），`api.github.com` 与 `gh api` 正常。本轮改用 REST API 推送（[tools/git-push-api](tools/git-push-api/readme.md)），远端 main 已推进到 `60869da2` 并与本地对齐。**下次运行先试 `git push`**，仍不通再用该工具；工具会压平本地多层提交。上游 verl 的 GitHub main 实测为 `3fdf2b54`（11-02T11:38Z），**不是** `4b26c788`——后者是 M1 本地 checkout 的 10-18 快照。
- **共享 checkout 已被并行会话并发使用**：本轮两次出现另一会话在同一目录 commit／reset（TM-008 inflight=1 补测、`01de74f8a` 订正传播）。合并冲突集中在 CONTEXT.md 与当日日志，已按"保留双方条目、按时间重排"解决。**动手前先看 `git log --graph` 与 `git worktree list`**，不要假设 HEAD 还是自己的。

- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)（§13.50 本轮、§13.38 回代对象）。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 本轮 M5 成果：[长度结构实验](experiments/rollout-length-structure-real/README.md)。
- 本轮 M1 成果：[锚② 复核归档](experiments/rollout-anchor2-identity/README.md)、[§20.3.6](notebook/rollout-performance-research.md)、[§11.6](notebook/rollout-performance-research.md)。
- 上一轮 M1 成果：[TM-010 取证](open-source/rllm/fireworks-r3-routing-matrices.md)、[复现](experiments/rllm-fireworks-r3-replay/repro.py)、[承载检查](experiments/rllm-fireworks-r3-replay/carrier_check.py)。
- 上一轮成果：[TM-009 候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[取证](open-source/rllm/tool-use-async-autodetect.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
