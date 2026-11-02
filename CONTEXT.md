# perpetual-learning — 执行接续

更新时间：2026-11-03（M1 04:0x 第三十三批／M5 02:4x；结论推至 `f006fbf94` 之后）。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

### M1 本轮（2026-11-03 04:0x）：TS-001 补强——P-6 的适用面加 async 轨迹与同配置重复臂（§20.3.8／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，选的是上一批留下的**最明显的一处缺口**：§20.3.7 的 8 个臂**全在 sync 路径上**，"形式不挑执行模式"此前没有数据。

**补了两行已入库归档，不改形式本身**：① **async 2 卡 poll 2.0→0.1（两跑池化）**实测整步比 1.0194，完整式误差 **+0.049%**（第一项单独 +0.029%，两侧缺口差 −0.027%）；② **g4 vs g4n（同配置重复，2+2 与 0,1,3,4）**实测 0.8372，误差 **−0.060%**，是本栈两侧缺口差最大的臂（0.029%，约其他臂 3 倍）、加速只有 1.19×。

**判决**：P-6 的适用面由"8 个 sync 臂"扩为"8 sync＋1 async＋1 同配置重复，误差都在 0.1% 内"——形式的成立**不依赖 sync 的步结构**。**两行都不进主结论**：同配置重复臂的"加速"是运行间噪声（同一配置两次跑、只有放置微差），跨放置臂（`pbase→p4base`，0.163%）按既有约定只作对照。**锚② 强度仍是「中强」，本批不回调**；跨系统仍归 P-1。

**核对产物**：`check_reconstruct_form.py --verify` exit 0（主结论臂 0.1% 阈值，两行新臂只打印不判定）、`negative_control_form.py` 4/4 抓到 exit 0；`check_identity.py --verify`／`negative_control.py` 重跑仍 exit 0。

入口：[§20.3.8](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### M1 上一轮（2026-11-03 03:0x）：TS-001 补强——P-6 完成，整步比的可重建形式（§20.3.7／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，承接上一批（§20.3.6）留下的投稿前必补项 P-6。

**结论**：形式为 `整步比 = [1/(Σ_k s_k^基线/G_k)] × (δ^基线/δ^目标)`，`s_k^基线 = t_k^基线/Σt^基线`、`δ = S/T`（计时缺口）。**是代数恒等不是拟合**（由 `Σ_k s_k^基线·(T^基线/t_k^基线) = 1/δ^基线` 化简）。同栈 **8 个臂**（2 加卡＋6 单轴）误差 **≤0.04%**，加卡臂由旧式的 −5.1%／−15.4% 变为 −0.007%／+0.010%；跨放置对照臂 0.16%，只作对照。

**旧式失效的机理（本批新增诊断）**：同一算子把份额取在**目标侧**时，`Σ_k s_k^目标·(T^目标/t_k^目标) = 1/δ^目标`，式子退化为 `δ^目标·(S^目标/S^基线)`，连加速倍数都不是。问题**不是"用了份额"，而是份额取了加速之后的那一侧**——效应被提前算进权重，越大算得越少。

**两个限定已写进正文**：① 本栈两侧缺口几乎相同（|差| ≤0.08%），第一项单独就接近整步比，但这是**数据事实不是恒等**（负向对照把缺口拉开 20%，第一项偏 20%、完整式偏 0）；② 单栈单点，显著程度不外推。**锚② 强度保持「中强」不回调**——补的是形式可用，跨系统仍是 P-1。

**核对产物**：`check_reconstruct_form.py --verify` exit 0、`negative_control_form.py` 4/4 注入抓到 exit 0；上一批的 `check_identity.py --verify`／`negative_control.py` 重跑仍 exit 0，未破坏既有结论。

入口：[§20.3.7](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### 上一轮（2026-11-03 02:0x）：TS-001 补强——锚② 的可换算性复核（§20.3.6／§11.6）


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

### M5 本轮（2026-11-03 03:0x–04:3x CST）：第五十六批 §13.52 + 第五十七批（stale 逐出补测）

**§13.52 `max_inflight_gen_batches=3` 已测**：GPU 4,5（同 socket＝2+0）、同一 patched 树、DAPO filter on、2.0／0.1 交替各两跑、15 步/跑、4/4 exit 0（18:59–19:39 UTC）。池化 gen **8.589→7.357s（−14.3%）**、整步 **26.964→25.738s（−4.5%）**、生成组 **113.7→133.0（+17.0%）**；其余相位 |Δ|≤0.5%。臂内 FE 斜率 38.8／44.2 ms/组，固定等待 −1.97／−2.04s、多生成 +0.75／+0.83s，残差 ≈−0.01s。跨深度 +5.3%→+13.1%→+17.0%、净收益 −0.96→−1.29→−1.23s ⇒ **depth 2 之后饱和**；depth 2→3 增量 <1 SE，不外推。**方法订正**：跨臂共斜率被处理效应压低（21.2 vs 38.8），同族分析按臂内固定效应报。归档 [experiments/verl-v1-poll-quantum-dapo](experiments/verl-v1-poll-quantum-dapo/README.md#补测max_inflight_gen_batches3-2026-11-03m5-第五十六批)，四份日志 md5 与服务器一致。

**stale 逐出补测（第五十七批，进行中/已完成见下）**：`ReplayBufferAsync` 的 stale 逐出只在 `max_off_policy_threshold` 生效；threshold=2 冒烟 4 步仍是 span≤2、零逐出，**threshold=1 才触发**（每步丢弃整批 32 组、staleness=2、生成量翻倍）。A/B 四跑（2.0／0.1 各两跑，15 步）在 GPU 4,5 上跑。

### M5 上一轮（2026-11-03 02:0x–02:4x CST）：第五十五批 §13.51

**TM-008 的 async 分支已测**：`trainer.v1.trainer_mode=colocate_async`（warmup=1）、2 卡（GPU 4,5 同 socket＝2+0）、同一 patched 树（`replay_buffer.py` 与 pristine 逐字节相同，只加配置接线），poll 2.0／0.1 交替各两跑、15 步/跑、4/4 exit 0。现场：liyaowei 的 4 卡 vLLM（GPU 2/3/6/7）＋1 个 EngineCore（GPU 0）占其余，仍只有 3 卡空闲；`/home` 仍满（8.0 G 余），全部写 `/tmp`。

**三个结论**：① 量子在 async 同样存在——2.0 臂 gen 两态 **0.22s vs 2.23–2.33s**，8–9/12 步命中（sync 2 卡 6/12）；② **2.245s ≠ 浪费 2.0s**——0.1 臂 0/12 命中、gen 落到 **0.75–1.89s**，量子只是把真实等待向上取整，超额 0.1–1.3s/步；③ 池化 gen −16.7%（−3.4 SE）、整步 21.212→20.809s＝**−1.9%**（run 级 ≈2 SE；单跑 −3.0%／−0.7%），整步差只有 gen 可归因。**补丁适用面不是「仅 sync」**；async 也没有消掉等待（仍 1.4–1.7s/步）。归档 [experiments/verl-v1-poll-quantum-async](experiments/verl-v1-poll-quantum-async/README.md)，`verify.py` exit 0，四跑日志 md5 与服务器一致。

**昇腾状态变化（未可用）**：910C VPN 网关 `10.205.0.1` 第 28 天来首次 ping 通（0% 丢包、6ms），但 `192.168.13.154/155/156` 仍 100% 丢包、SSH 22 端口超时——链路只恢复到网关，NPU 主机仍不可达，不据此认为可以开工；无新增 NPU 任务需要用户处理。

### M5 上一轮（2026-11-03 00:5x–01:4x CST）：第五十四批 §13.50

真实 rollout 的响应长度结构实测（gsm8k `train_sub480` × n=8，1.5B／0.5B 五档，19,200 条响应）：构造档 `realistic=(0.60,0.08)` 隐含 ICC ≈0.98，实测 **0.450 [0.409,0.489]（1.5B cap256）／0.364 [0.321,0.407]（0.5B）**，且真实组内离散大于组间（0.166 vs 0.161）；`max_response=256` 截断 45.3%／60.8%，干净组重算的 0.174／0.128 是选择偏差不可引；回代 §13.38 两主读数在 1 SE 内不动（`ldm/grouped` 1.0462→1.044–1.048、`interleaved/ldm` 1.0045→1.0027–1.0065）。归档 [experiments/rollout-length-structure-real](experiments/rollout-length-structure-real/README.md)，`verify.py` exit 0。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 第三十三批（§20.3.8：P-6 加 async 与同配置重复臂）完成；TM-010／TM-009 待用户决定是否对外 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。TS-001 再往前只剩 P-1 跨系统（需用户立项），`experiments/rollout-c-class-ledger/` 四轮完成、再往下需新字段或新实验，不在当前授权范围。 |
| M5 | TS-001 算力支持 | **第五十六批 §13.52 已完成；第五十七批（stale 逐出 threshold=1 A/B）运行中**（见上）。TM-008 剩余未测：4 卡 `inflight=1`（需 ≥4 卡空闲窗口）、`separate_async`（需独立 rollout 池＋非 naive backend）。P-1 跨系统（AReaL／OpenRLHF／SkyRL，需用户立项）；P-2 FP8 训练侧（待实测）。**下一轮先核对 GPU 归属与占用再启动，不据旧记录假定空闲。** |

## 未完成操作与并行工作

- **TS-001 第三十三批（M1 2026-11-03 04:0x）**：§20.3.8 已落笔，P-6 适用面扩到 async。**下一次触碰锚② 前必须知道的（更新）**：可发表形式仍是 §20.3.7 那个式子（份额取**基线侧**、乘两侧缺口之比），现在可以说它「在同栈 sync＋async 两条轨迹上闭合」，但**仍不能说跨系统**（P-1 全归跨系统）。锚② 强度「中强」，不回调。
- **TS-001 第三十二批（M1 2026-11-03 03:0x）**：§20.3.7／§11.6 已落笔，P-6 完成。**下一次触碰锚② 前必须知道的（更新）**：可发表形式是 §20.3.7 那个式子（份额取**基线侧**、乘两侧缺口之比），不要再写"恒等式已获同栈实测校验"，也**不要**把份额取在目标侧——那是被否掉的旧式。强度「中强」、跨系统仍缺 P-1。
- **TS-001 第三十一批（M1 2026-11-03 02:0x）**：§20.3.6／§11.6 已落笔；`experiments/rollout-anchor2-identity/` 已入库（`check_identity.py --verify` exit 0、`negative_control.py` exit 0）。**推送（2026-11-03 02:1x）**：`git push` 直连 github.com:443 仍不通（75s 超时），已用 [tools/git-push-api](tools/git-push-api/readme.md) 推到远端 main `0eee4eb7`；本地 commit 为 `b1c8e4786`，**两者树相同、SHA 不同**（API 以远端父提交重建），本地尚未对齐——下次运行先 `gh api` 取远端 sha 再决定是否 reset。**推送（2026-11-03 02:1x）**：`git push` 直连 github.com:443 仍不通（75s 超时），已用 [tools/git-push-api](tools/git-push-api/readme.md) 推到远端 main `0eee4eb7`；本地 commit 为 `b1c8e4786`，**两者树相同、SHA 不同**（API 以远端父提交重建），本地尚未对齐——下次运行先 `gh api` 取远端 sha 再决定是否 reset。**推送（2026-11-03 02:1x–02:2x）**：`git push` 直连 github.com:443 仍不通（75s 超时），用 [tools/git-push-api](tools/git-push-api/readme.md) 分两次推到远端 main `9bc7cd05`；本地为 `b1c8e4786`＋`5d0e48b86`（树相同、SHA 不同，API 以远端父提交重建）。**两次推送合起来才完整**——第一次推的是工作树，第二次补上了收尾写进 CONTEXT／diary 的两处；逐文件 blob 已核对一致。**下次运行先 `gh api` 取远端 sha，本地未对齐属已知偏差**。
  **下一次触碰锚② 前必须知道的**：不能写"恒等式已获同栈实测校验"——乘积式重建不出整步比，`G_k≡1` 时给 1。可发表写法是"逐阶段秒数相加 + 计时缺口"。
  **P-6 已于第三十二批补完**（§20.3.7），可发表形式见该节。
- **推送通道（2026-11-03 03:0x 更新）**：`git fetch`／`git push` 直连**已恢复**。M1 本轮先 commit 再 push 被拒（API 分叉），`git fetch` 后与 `origin/main`（M5 第五十五批）合并、冲突按「保留双方、按时间重排」解决，已推至 **`f006fbf94`**，本地 origin/main 一致。约定不变：先试 `git push`，不通再用 [git-push-api](tools/git-push-api/readme.md)；API 推送会重建 commit 对象，与本地同树不同 SHA 属已知偏差。
- **TM-010（掘金 B 类，M1 2026-11-03 01:3x）**：rLLM Fireworks 后端默认 `router_replay: R3`，但路由矩阵装不进 `tinker.ModelInput`（无声明字段 / `from_ints` 无该参数 / 赋值被 pydantic 拒）。**未验证项：服务端行为与线上影响**，不得主张训练出错。对外动作全部未做，待用户决定是否提 issue（以及是否连"默认改回 disabled"一起）。见 [TM-010](queue_tech-miner.md)。
- **TM-009（掘金 B 类，M1 2026-11-03 完成四件套）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效 + 守卫不可达。**对外动作（issue／PR）全部未做，待用户逐项授权**；需用户决定：① 是否提独立小 issue；② 若提 PR，是否连补丁一起（补丁会让上游 `test_python_tool` 变红）。上游 `rllm-org/rllm` main `4e828c81`（2026-11-02 核对，未变）。见 [TM-009](queue_tech-miner.md) 与[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- **TM-008（掘金 C 类，M5）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。已测：2 卡 sync 固定批 −3.2%、2 卡 DAPO inflight=2 −4.9%／inflight=1 −3.4%、4 卡 DAPO −4.0%、**2 卡 colocate_async −1.9%（≈2 SE，§13.51）**。补丁 3 文件 +8/−1，默认 2.0 不变，对 `3fdf2b54`／`4b26c788` dry-run 均 3/3 通过。**对外动作全部未做，待用户逐项授权**；需用户决定「只加可达性」还是「同时改默认 0.1–0.2」。
- **TM-006（掘金 B 类）**：verl `tiled_mlp` 十引擎无消费点。[补丁就绪](open-source/verl/patches/tiled-mlp-wiring/readme.md)；真机冒烟完成（FSDP2 生效、FSDP1 硬报错、默认梯度检查点下无显存收益）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类）**：verl `use_prefix_grouper` 在 FSDP 上不生效但副作用照常。取证与影响量化（DP 负载不均衡 0–6.6%）已入库。**对外动作全部未做，待用户逐项授权**。
- `experiments/verl-v1-poll-quantum-async/`（本轮新增）：`run-async-ab.sh`／`run-chain.sh`／`make-summary-async.py`／`verify.py`＋`raw/`（4 份控制台日志＋md5）＋`summary.json`／`README.md`。服务器侧原始日志在 `/tmp/rollout-poll-async-*.log`、patched 树在 `/tmp/m5-poll-ab/verl`（均非持久）。
- `experiments/rllm-tool-use-async-probe/`（TM-009）已入库：`repro.py`／`prod_reachability.py`／`readme.md`／`run-info.json`；`open-source/rllm/patches/use-async-autodetect/` 含补丁与 readme。复现依赖 uv 建的 Python 3.12 venv（系统 3.9.6 因 PEP 604 无法导入 rllm）。
- `experiments/rollout-length-structure-real/`（第五十四批）：`sample_lengths.py`／`analyze_lengths.py`／`reanalysis_length_regime.py`／`verify.py`＋`raw/`＋`summary-*.json`／`README.md`。服务器侧原始文件在 `/tmp/m5-length/`（非持久）。
- `tools/pyflag/` 新增 `scan-rllm-4e828c8.json`，readme 加第五类易错点"三态形参"。
- **`experiments/verl-v1-poll-quantum/`、`verl-v1-poll-quantum-dapo/`、`verl-v1-poll-quantum-dapo-g4/`、`verl-tiled-mlp-fsdp-smoke/`、`rollout-g8-gen-bimodality/`、`rollout-c-class-ledger/` 等均已入库**，不重复登记。
- **上游状态（本轮只读核对，均为历史快照）**：verl main `3fdf2b54`（11-02T11:38Z，未变）；verl PR #8313 open、`mergeable=true`、`mergeable_state=blocked`、head `71fd3562` 未变；rLLM main `4e828c81` 未变。下一轮重新核对，不据此推断实时状态。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **共享 checkout 被并行会话并发使用**：动手前先看 `git log --graph` 与 `git worktree list`，冲突集中在 CONTEXT.md 与当日日志时按「保留双方条目、按时间重排」解决。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)（§13.51 本轮、§20.3.6／§11.6 M1 本轮）。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 本轮 M5 成果：[async 轮询实验](experiments/verl-v1-poll-quantum-async/README.md)。
- 本轮 M1 成果：[§20.3.8 适用面](notebook/rollout-performance-research.md)、[P-6 重建形式](experiments/rollout-anchor2-identity/check_reconstruct_form.py)、[负向对照](experiments/rollout-anchor2-identity/negative_control_form.py)、[§20.3.7](notebook/rollout-performance-research.md)、[§11.6](notebook/rollout-performance-research.md)。
- 上一轮 M5 成果：[长度结构实验](experiments/rollout-length-structure-real/README.md)。
- 上一轮 M1 成果：[锚② 复核归档](experiments/rollout-anchor2-identity/README.md)、[§20.3.6](notebook/rollout-performance-research.md)、[TM-010 取证](open-source/rllm/fireworks-r3-routing-matrices.md)、[TM-009 候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
