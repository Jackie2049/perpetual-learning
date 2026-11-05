更新时间：2026-11-05 11:2x（**M1 TM-023：FastVideo 扩池第八批——65 个开关 / 8 条候选，真缺陷 0 条**。2118 个 `.py`（快照 `36dc971d`）扫出 65 个开关；逐条绑行号后 **R3 内部默认 4**（`use_bf16`／`use_128x4_sf_layout`／`use_queue_mode`／`use_nvfp4` 转发型）、**R5 独立子工程内部 4**（全在 `fastvideo-kernel/`，自带 pyproject）⇒ **真缺陷 0 条**。方法产出三条：① **多包仓库按包分组**（混在一起会把子工程内部的包装算漏）；② **backend impl 类工厂是内部默认高发区**（接收端 `**extra_impl_args`、发送端只传公共参数）；③ **`mode=always` 与 `producers_total` 回答不同问题**（`use_128x4_sf_layout` 三处调用点全显式传 `True`）。**两处判据错**都由重跑抓出（声明形态凭印象写错；把 docstring 提及算成生产者，改用 AST）。价值 **3/10**，**不对外**。纯静态 + 全仓扫描，未装 fastvideo、未跑推理、未触上游。）此前：（**M1 TM-022：AReaL `use_cpu` 的 CPU 回退开关既打不开、也不用打**——唯一调用点 `moe/utils.py:212` 传 5 个位置实参不含它、全仓零生产者；`kernels.py:5-7` **顶层 `import triton`** ⇒ 无 Triton 的机器整模块进不来（本机实测），其测试用 `skipif(not HAS_TRITON)` 静默跳过。**但那段 CPU 实现本身是对的**：抄本（`check_snapshot.py` 双向断言）在**上游自己的缓冲口径** `round_up(总 token + E×对齐, 对齐)`（`utils.py:204-205`）下与 GPU 排布**逐位相同**（3 档 + 2550 组随机输入全对），只有把缓冲压到 token 总数时才抛 `RuntimeError`。⇒ 「开关不可达 + 实现可用」，**不是坏掉的回退**。**3 处自错**都由对照抓出，最要紧的是①用 `m_offsets[-1]` 当缓冲 ⇒ 四档全崩、差点写成"一跑就崩"。价值 **3/10**，**不对外**。纯 CPU、抄本口径、**未装 areal、未触上游**。）此前：（**M1 TM-021：MLC-LLM `mlc_llm router` 的子服务器恒开 debug 端点、用户关不掉**。`router/router.py:82` 给每个 `PopenServer` 传 `enable_debug=True`，而 `Router.__init__` 的形参表与 `cli/router.py` 的 10 个参数里**都没有**这个开关 ⇒ 用 router 部署时 `/debug/reset_engine`（重置引擎、清运行数据）、`/debug/cuda_profiler_start|stop`、`/debug/dump_event_trace`、`/debug/dump_engine_metrics` 恒可用且关不掉；同仓 `mlc_llm serve` 有 `--enable-debug`（默认关）⇒ **两条部署路径不一致**。四跳全取**真实函数体**在 CPU 上执行（`PopenServer`／`requests`／`fastapi` 用替身，清单进读数）：C1 两档都 `True`、C1 对照项 `prefix_cache_mode` 随档变（`radix`/`disable`）⇒ 读数不是死的；C2 `True` 档 argv 含 `--enable-debug`（**2 次**）；C3 `interface/serve.py:124` 真挂 debug 路由、`openai_entrypoints.py:139/140/206/314` 真读它。**历史：不是回归**——router 由 `5c9ebcb5`（#3064，2024-12-16）落地即如此，`git log -S` 此后无人动过这一行；`PopenServer` 的默认值由 `0b2cbb2c`（#2417）从 `False` 改 `True` **并在同一提交里加了第二处追加** ⇒ `--enable-debug` 被追加两次，但 `store_true` 幂等、真 argparse 三形状实测无害（5/5）。**5 处自错**，最要紧的是「把函数体截到 `subprocess.Popen(` 之前」——argv 恰只在那里拼，截掉后读数成了「两支都没有该 flag」，**看着像结论成立**，已加"抓不到 argv 就抛异常"的守卫。自检 **5 项 / 6 注入全过**。价值 **5/10**（端点标为 debug 用途、默认只绑 `127.0.0.1` ⇒ **不能写成安全漏洞**）。**不对外**，三个最小动作 (a)改注释/(b)把开关接出来/(c)删重复追加 待用户逐项授权。纯 CPU、未起服务、未发 HTTP、未触上游。）
更新时间：2026-11-05 09:2x（**M1 TM-020：verl `algorithm.filter_groups.metric` 的三份说明互相不自洽，两个名字在 V1 上零写入点**。V1 组过滤按**名字**去 `extra_fields.reward_extra_info` 取键（`replay_buffer.py:299` 的 `filter_groups_metric not in reward_extra_info`），取不到抛 `RuntimeError`。而「能填哪些名字」有三份不同说明：配置 docstring（`reward` canonical／`acc`／`score`／**`seq_reward`**）、`docs/algo/dapo.md:80`（举例 `score / seq_reward / seq_final_reward`）、随仓库脚本 `grpo_qwen35_35b_megatron_async.sh:64`（**`response_filter`**）。**后三个在 V1 零写入点**——真 `ReplayBuffer`（`stub_env.py` 只替身 `transfer_queue`／`SkipManager`，判定是上游原文）按上游 CPU 测试同一套写盘顺序造「2 条同分轨迹的组」跑：`reward`／`acc` ran，`score` 只在 `math_dapo` scorerd 下 ran（gsm8k 等走 `float(res)` 只给 `acc`），`seq_reward`／`seq_final_reward`／`response_filter` **给多少键都 raised**；缺**一条**轨迹的键也整组抛。**上游单测只测过 `acc`(13 处) 与 `reward`(1 处)** ⇒ CI 抓不到。**历史（决定对外措辞）**：docstring 自 `c936ec7`(2025-07-04) 就写着 `seq_reward`，直到 `7e9875249`(2026-10-12) 才改；**DAPO recipe 的 `dapo_ray_trainer.py:240` 真有 `if metric_name == "seq_final_reward":` 分支**（verl-recipe 仓）⇒ 这些名字在 V1 之前有主；V1 自 `c88e875`(2026-07-23) 引入起只认字面键 ⇒ **名字搬进 V1 后语义没跟上**；#7792(`7cb6501`,2026-09-08) 改注释时**留了一个 `seq_reward`**。⇒ 对外只能写「说明与实现没对齐」，**不能写「开关坏了」**。价值 **5/10**（官方 recipe 全用 `acc` 不触发；但崩在第一步且报错会误导）。**自检 8/8、负向对照 4/4**（A 夹具丢键／B uid 含下划线／C 改坏 canonical 常量／D 所有名字都可读，四种破坏各让指定检查变红）。**不对外**（上游查重 `filter_groups.metric` 0 条、`seq_reward`／`seq_final_reward` 各 8 条无关）。纯 CPU、**未装 ray/torch、未跑训练、未触上游**。）
更新时间：2026-11-05 08:4x（**M5 无卡轮：计数器补丁的死登记已删，4 卡臂加了"树必须是受测树"的硬门槛**——`validation/sampler/blocking_polls` 永不可达（全仓 `partition_id="val"` 只有 `_validate()`，且 `batch, _ = replay_buffer.sample(...)` 丢弃 metrics），补丁 +9→**+8 行**（`sha256=4c142752…`）；上一版自检只数键名，现改为**执行真实 `MetricsAggregator`**（喂 2 与 3：打补丁树 **5.0**=sum、未打补丁 **2.5**=weighted_avg），并加"注册了 validation 键 ⇔ val 调用点保留 metrics"不变量；sim 扩到**两个 buffer 类**（含 `wait_for_sampleable`）**12/12**、惰性仍 **480 行逐字段相同**。新增 [prep-tree.sh](experiments/verl-v1-poll-quantum-dapo-g4/prep-tree.sh)（复制＋两补丁＋4 标记）、`run-poll-dapo.sh` 的 `CHECK_TREE=1`（缺任一补丁 **exit 65**，`CHECK_ONLY=1` 只查树）、[preflight-g4.py](experiments/verl-v1-poll-quantum-dapo-g4/preflight-g4.py) 必填 `--run-tree`（树标记＋在受测树本体跑 sim＋守卫 0/65 双对照），自检 **11/11**；读数 [preflight-g4-instrumented.json](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4-instrumented.json) `ok=true`。**零 GPU 读数；计数器仍未在服务器上执行过；双链路仍断**（4090-1 100% 丢包＋`:22` 超时，昇腾无 VPN 路由）。）。此前各轮摘要见下一行与 `###` 小节。
更新时间：2026-11-05 08:1x（**M1 TM-019：MLC-LLM 扩池第六批——真缺陷 0 条**；两条候选都是扫描器把工厂转发当成决策者，已修进 pyflag 第十三类。完整叙述见 [实验](experiments/mlcllm-pyflag-triage/README.md) 与 [diary](diary/2026-11-05.md)。）此前各轮摘要见下方 `###` 小节与 [diary](diary/2026-11-05.md)，任务状态见 [技术秘书](queue_tech-secretary.md)／[技术掘金](queue_tech-miner.md)。

### M1 本轮（2026-11-05 11:2x）：TM-023——FastVideo 扩池第八批，**8 条候选没有一条是缺陷**

**一、为什么选 FastVideo。** 前七批：SGLang／vLLM／LMDeploy／TensorRT-LLM／llama.cpp／MLC-LLM（LLM 推理六批）＋ AReaL（RL 训练）。视频生成是生态里另一个大块，形状也不同：**一张 `FastVideoArgs` 配置面 + 一堆按 backend 实例化的 impl 类 + 一个独立的内核子工程**。

**二、读数。** `36dc971d`（2026-11-04，2118 个 `.py`）→ **65 个开关 / 8 条 `silently_inert`**。逐条绑行号后 **真缺陷 0 条**：**R3 内部默认 4**（`use_bf16` 是 SLA 两个 impl 类的 dtype 选择；`use_128x4_sf_layout` 三个调用点全显式传 `True`；`use_queue_mode` 由 `from_fastvideo_args` 吃默认；`use_nvfp4` 是 `kwargs.pop` 默认 + 原样转发的转发型）、**R5 独立子工程内部 4**（`use_high_prec_o`／`use_qat_qkv_backward`／`enable_smoothing_k`／`enable_smoothing_q`，全在 `fastvideo-kernel/`）。

**三、方法产出（真东西）。** ① **多包仓库按包分组**：`fastvideo-kernel/` 自带 `pyproject.toml` 与自己的调用约定，与主包混在一起数"零生产者"会把子工程内部的包装算漏。② **backend impl 类工厂是内部默认的高发区**：接收端 `**extra_impl_args`／`**kwargs`、发送端只传公共参数——这是第七类易错点（`**kwargs` 兜底）的**接收侧镜像**，不该按"失效"处理。③ **`mode=always` 与 `producers_total` 回答的不是同一件事**。

**四、本批两处判据错（都由重跑抓出）。** ① 拿 `use_high_prec_o: bool = False` 当声明形态，实际是位置默认参数 `use_high_prec_o=False,`（`:1243`）——**判据里的字面形态要现读源码**。② "外部生产者只应在测试里"把 `uniproc_executor.py` 一行 **docstring 提及**算成生产者 ⇒ 改用 AST，只认真实赋值与调用关键字。

**五、价值与判断。** **3/10，不对外**：8 条里没有一条造成"配了没反应"。

**核对产物**：`verify_findings.py` **24 checks / 0 failed**、`--selftest` **3/3**、`classify.py` 8 条全覆盖。纯静态 + 全仓扫描，**未装 fastvideo、未跑推理、未上 GPU/NPU、未触上游**。

入口：[实验](experiments/fastvideo-pyflag-triage/README.md) · [判据](experiments/fastvideo-pyflag-triage/verify_findings.py) · [归因](experiments/fastvideo-pyflag-triage/classify.py) · [读数](experiments/fastvideo-pyflag-triage/raw/scan-fastvideo.json) · [证据页](open-source/fastvideo/evidence/pyflag-2026-11.md) · [TM-023](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 上一轮（2026-11-05 11:1x）：TM-022——AReaL 的 CPU 回退开关**打不开，但也不用打**

**一、为什么做。** AReaL 扩池第七批（TM-019 的下一批，已同批入库）把 `use_cpu` 判成"真不可达"后留了半句：那段 CPU 实现本身是不是好的？这决定它该被删掉还是该被接线。

**二、三段事实。** ① 唯一调用点 `areal/experimental/models/archon/moe/utils.py:212` 传 5 个位置实参、不含 `use_cpu`，全仓（含 `tests/`）零生产者；② `kernels.py:5-7` **顶层 `import triton`** ⇒ 没有 Triton 的机器连模块都进不来（本机实测 `ModuleNotFoundError`），其测试 `test_moe_common.py` 用 `skipif(not HAS_TRITON)` 静默跳过，CI 不会红；③ `fill_indices_cpu` 起笔用**对齐后**的 `write_offsets[e]`、累加用**未对齐**的 `length`——两个口径确实不一致。

**三、读数（纯 CPU，抄本）。** 把 `fill_indices_cpu` 与 `generate_permute_indices` 算偏移那几行逐行抄出（`check_snapshot.py` 双向断言抄本与快照一致，3 条注入自检全过；上游源码逐字节快照在 `raw/upstream/`）。在**上游自己的缓冲口径** `round_up(总 token + E×对齐, 对齐)`（`utils.py:204-205`）下，抄本与 GPU 排布**逐位相同**——A/B/C 三档 0 点差异，另跑 2550 组随机输入（R∈{1,2,4,8}、E∈{1,4,16,64}、对齐∈{1,8,16,64}）全对、写入条数恒等于 token 总数。只有把缓冲压到 token 总数时才抛 `RuntimeError: upper bound and lower bound inconsistent with step sign`。

**四、结论口径。** 「**开关不可达 + 实现可用**」，**不是**"坏掉的回退"。上一批"CPU 实现从未执行过"仍然成立，本批补的是它是**可用但不可达**。

**五、本批三处自错。** ① 用 `m_offsets[-1]`（对齐后总量）当缓冲 ⇒ 四个档位全崩，差点写成"一跑就崩"；换成上游真实口径后 0 差异——**同一段代码两种口径给出相反结论，差别只在缓冲从哪来**。② 把"抄本与 GPU 排布必须有差异"写成期望 ⇒ 三档 0 差异、判据自己红。③ "与位置脱节"自洽判据隐含 rank-major 假设（实际是 expert-major、段内 rank 序），把正确实现打成"脱节 10 处"，已删。

**六、价值与判断。** **3/10，不对外**：无配置字段、无 CLI、默认关闭、开发期调试开关 ⇒ 无用户受影响。留下的是方法结论：**报"回退路径坏了"之前，先把调用方实际给的缓冲/参数口径对齐**。

**核对产物**：`probe.py` 6 类 15 条判据全 PASS（含两个改坏对照）、`check_snapshot.py` 4 条 PASS + 注入自检 **3/3**。纯 CPU、抄本口径；**真实模块在本机 import 不进来**，没有"AReaL 在本机跑起来"的读数。未装 areal、未上 GPU/NPU、未触上游。

入口：[实验](experiments/areal-use-cpu-fallback/README.md) · [探针](experiments/areal-use-cpu-fallback/probe.py) · [抄本一致性](experiments/areal-use-cpu-fallback/check_snapshot.py) · [读数](experiments/areal-use-cpu-fallback/raw/probe.json) · [证据页](open-source/areal/evidence/use-cpu-fallback-2026-11.md) · [TM-022](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 上一轮（2026-11-05 10:0x）：TM-021——`mlc_llm router` 的子服务器恒开 debug 端点，**四跳读数里第一次出现"看起来对"的假答案**

**一、为什么做。** TM-019 的结论是"MLC-LLM 那条 `enable_debug` 扫成 `producers_total=1`，看着正常"。本轮顺着这条生产者往下走：如果它接的是**字面量**，那"有人传"就不等于"配得动"。

**二、问题。** `router/router.py:82` 给每个子服务器传 `enable_debug=True`（写死）。`Router.__init__` 的形参表没有它，`cli/router.py` 的 10 个参数也没有。⇒ 用 `mlc_llm router` 部署时 `/debug/reset_engine`（重置引擎、清运行数据）、`/debug/cuda_profiler_start|stop`、`/debug/dump_event_trace`、`/debug/dump_engine_metrics` 恒可用且**关不掉**。同仓 `mlc_llm serve` 有 `--enable-debug`（默认关）⇒ 两条部署路径不一致。

**三、四跳读数（全部执行真实函数体，CPU）。** C1 `Router.__init__` 全程（含内部 `start_server` 闭包）：两档都构造 `PopenServer(enable_debug=True)`，**对照项 `prefix_cache_mode` 随档变**（`radix`/`disable`）⇒ 读数不是死的；C2 `PopenServer.start()` 全程（只桩 `subprocess.Popen` 与 `requests`）：`True` 档 argv 含 `--enable-debug` 且**出现 2 次**、`False` 档 0 次；C3 `interface/serve.py:124` 真挂 `debug_entrypoints.app`，`openai_entrypoints.py:139/140/206/314` 真读 `server_context.enable_debug`。`finding_holds: true`。

**四、不是回归。** `router.py` 由 `5c9ebcb5`（#3064 MicroServing，2024-12-16）新建时就是 `enable_debug=True`，`git log -S` 此后无人动过。`PopenServer` 的默认值由 `0b2cbb2c`（#2417，2024-05-25）从 `False` 改成 `True`，**同一提交里又加了第二处追加**（`popen_server.py:70-71` 与 `:114-115`）⇒ `--enable-debug` 被追加两次；用快照里**真实的** `mlc_llm.support.argparse` 实测 `once`/`twice`/`zero` 三形状都正常、`twice ≡ once`、`model` 位置参数没被吃掉 ⇒ **重复幂等无害，不构成独立缺陷**。

**五、本批 5 处自错。** ① 连着 `def start(...)` 签名拼函数体 ⇒ 内嵌 `def`、外层函数体为空；② `class _Subprocess: Popen = _Popen` 类作用域看不到外层局部名；③ **把函数体截到 `subprocess.Popen(` 之前**——argv 恰只在那里拼，截掉后读数成了"两支都没有该 flag"，**看起来像结论成立**；已加"抓不到 argv 就抛异常"的守卫；④ `Popen.poll()` 返回 `0` 被 `is not None` 判成"子进程已退出"；⑤ `router_cli_can_turn_it_off` 第一版只看 CLI 有没有 debug 开关，自检"只加开关、不转发"的注入当场暴露 ⇒ 改为"CLI 有开关 ∧ 真转发进 `Router` ∧ Router 内不是字面量"。③ 是本批最要紧的一条：它给出的是**假读数**，不是崩溃。

**六、价值与判断。** **5/10**：分支活的、下游真分叉、用户关不掉、可证伪且行号可机读；但端点标为 debug 用途、`PopenServer.host` 默认 `127.0.0.1` ⇒ **不能写成安全漏洞**。**不对外**，三个最小动作待用户逐项授权：(a) 改注释说明 router 路径恒开；(b) 把 `enable_debug` 接成形参 + CLI + 转发（默认保持 `True`）；(c) 删掉重复那次追加。

**核对产物**：`probe_router_debug.py` `--selftest` **5 项 / 6 条注入全过**；`probe_duplicate_flag.py` **5/5**；`raw/snapshot/` 存 7 个源文件逐字节副本（无克隆也能跑）。纯 CPU、**未起服务、未发 HTTP、未装 mlc_llm、未建 TVM、未触上游**。

入口：[实验](experiments/mlcllm-router-debug-hardcoded/README.md) · [四跳探针](experiments/mlcllm-router-debug-hardcoded/probe_router_debug.py) · [重复 flag 探针](experiments/mlcllm-router-debug-hardcoded/probe_duplicate_flag.py) · [读数](experiments/mlcllm-router-debug-hardcoded/raw/probe-router-debug.json) · [证据页](open-source/mlcllm/evidence/router-debug-endpoints-always-on.md) · [pyflag 第三类](tools/pyflag/readme.md) · [TM-021](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 上一轮（2026-11-05 09:2x）：TM-020——verl 组过滤指标的说明与实现对不上，**两个名字在 V1 上零写入点**

**一、为什么做。** §13.49 结尾「顺带记一个文档不一致」只写了一句话（`FilterGroupsConfig.metric` 的注释举了 `seq_final_reward`，但 V1 reward manager 只写 `acc`），当时明确不构成 issue 提案、也没查证。本轮把它当独立问题查清：**是只有这一个名字错，还是一类**。

**二、机制。** V1 的 `_dapo_filtered_keys`（`replay_buffer.py:263`）对除 `"reward"` 以外的所有名字，都按**字面键**去 `extra_fields.reward_extra_info` 取，取不到就整组进 `missing_metric_uids` ⇒ `RuntimeError: Finished groups are missing DAPO metric '<名字>'`。键的来源只有两处：reward manager 写死的五个（`acc`／`overlong`／`overlong_reward`／`timeout`／`error`），和「scorer 返回 dict 时原样塞进去」。官方 scorer 里只有 `math_dapo` 返回 dict（`score`／`acc`／`pred`），其余数据集都 `return float(res)` ⇒ **真实场景下只有 `acc`**。

**三、三份说明互相不同。** 配置 docstring：`"reward"` / `"acc"` / `"score"` / **`"seq_reward"`**；yaml 注释：`reward` / `acc`；`docs/algo/dapo.md:80`：`score / seq_reward / seq_final_reward`；随仓库脚本 `grpo_qwen35_35b_megatron_async.sh:64`：**`response_filter`**。**`seq_reward`、`seq_final_reward`、`response_filter` 三个名字在 V1 里零写入点。**

**四、真跑读数（CPU，零 GPU）。** `stub_env.py` 用 exec 加载真实 `replay_buffer.py`，只替身 `transfer_queue` 与 `SkipManager`（都不参与该判定），按**上游 CPU 测试同一套写盘顺序**（先每条轨迹、再把 prompt 标 finished）造「2 条同分轨迹的组」。`reward`／`acc` ran；`score` 在「只有 acc」的真实输入下 raised，在「能给的都给」下 ran；`seq_reward`／`seq_final_reward`／`response_filter` **两种输入都 raised** ⇒ 不是「scorer 没给」，是没有任何代码写。另测「两条轨迹一条缺键」→ 整组仍抛。

**五、CI 为什么没抓到。** `tests/trainer/ppo/v1/` 里 `filter_groups_metric=` 的字面值只有 `acc`（13 处）与 `reward`（1 处），`seq_*` 一次没出现。

**六、历史归属（决定对外措辞）。** docstring 的 `seq_reward` 从该文件拆出那天（`c936ec7`，2025-07-04）就在，一路到 `7e9875249`（2026-10-12）才改；**DAPO recipe 的 `dapo_ray_trainer.py:240` 真的有 `if metric_name == "seq_final_reward":` 分支**（verl-recipe 仓，2026-11-05 读取）⇒ 这些名字在 V1 之前有主；V1 自 `c88e875`（2026-07-23）引入起就只认字面键 ⇒ **名字搬进 V1 后语义没跟上，文档慢了两轮**；#7792（`7cb6501`，2026-09-08）在改注释那一轮**留了一个 `seq_reward`**，`docs/algo/dapo.md` 一字未改。

**七、价值与判断。** **5/10**：影响面窄（仓内所有 recipe 脚本都用 `acc`，官方路径不触发），但崩在**第一个训练步**、报错信息会把人引向「我的 scorer 少返回一个键」这一错误结论，而修复成本极低。**只能写成「说明与实现没对齐」**，不能写「这些开关坏了」——最小对外动作应是改注释/文档或加一行加载期校验，**不该给 V1 补 `seq_*` 语义**（那是新增语义，得设计者定）。

**八、本批两处自错（都由检查抓到并已改）。** ① 第一版抽 docstring 用「8 空格缩进」作参数块边界，而 `ast.get_docstring` 已剥掉公共缩进（参数行剩 4 空格）⇒ 只读到 `reward` 一个名字，差点据此判断「docstring 没问题」；② `negative_control.py` 第一版把破坏打在外层包装上，内层没被改到，B 组跑出「假通过」（期望失败项与实测不符才发现）。

**核对产物**：`probe_metric_tables.py`（`--selftest` **8/8**）、`negative_control.py`（**4/4**，四种破坏各让指定检查变红）、读数 [metric-tables.json](experiments/verl-filter-groups-metric-names/raw/metric-tables.json)。纯 CPU、零 GPU、**未装 ray/torch、未跑训练、未触上游**。

**留给下一轮**：① TM-020 **不对外**，对外动作待用户逐项授权（三个选项已列在队列条目里）；② TM-015（vLLM `enable_kv_events`）与 TM-005／TM-006／TM-007（verl／SkyRL 接线）仍待用户逐项授权；③ TS-001 仍停在「纯归档重算已到边际」，本条属 TM 侧不属于 TS-001。

入口：[实验](experiments/verl-filter-groups-metric-names/README.md) · [三表扫描](experiments/verl-filter-groups-metric-names/probe_metric_tables.py) · [替身环境](experiments/verl-filter-groups-metric-names/stub_env.py) · [负向对照](experiments/verl-filter-groups-metric-names/negative_control.py) · [证据页](open-source/verl/evidence/filter-groups-metric-names.md) · [TM-020](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 上一轮（2026-11-05 08:1x）：TM-019——MLC-LLM 扩池第六批，**两条候选都是扫描器自己的误判**

**一、为什么选 MLC-LLM。** 五批之后常规池只剩两个没扫的：MLC-LLM 与训练侧（AReaL / NeMo-RL / slime）。MLC-LLM 是推理部署引擎里唯一"配置面在 Python、C++ 只管服务"的架构，能检验前五批的排除判据在这种形状上还成不成立。

**二、读数。** `44fd06d0`（2026-10-30）Python 415 个 `.py`／**4.0 秒** → **7 个开关 / 2 条 `silently_inert`**；C++ 88 个 C/C++／**0.5 秒** → **22 个有消费者字段 / 1 条 `unwritten`**。逐条绑行号后**真缺陷 0 条**。

**三、两条候选是同一个根因：工厂函数被当成了决策者。** `interface/router.py:17` 的 `serve()` 函数体里没有 `if enable_prefix_cache:`，但这个名字出现在 `router_type(..., enable_prefix_cache=enable_prefix_cache, ...)`（`:38`）的实参位置 ⇒ 老扫描器把它记成消费者，再因为"没人按关键字传给它"报 `silently_inert`。真决策在 `router/router.py:86`。

**四、四跳在 CPU 上真跑。** `probe_router_chain.py` 取三处**真实源码**（真 `Router.__init__` 含它的 `start_server` 闭包、真 `interface.router.serve`、真 argparse），第三方模块用替身（清单进读数）：`True` → 两台 `PopenServer` 都拿到 `radix`、`False` → 都拿到 `disable`；`Router(...)` 两档都收到原值。`verdict: WIRED`，selftest **8/8**。

**五、本批最要紧的一处自错：探针第一版给了反向读数。** 第一版把 `uvicorn` 换成了空模块，而真 `serve()` 末尾是 `uvicorn.run(app, ...)` ⇒ `AttributeError`，那一轮被记成失败、读数成了 `<absent>`，**我一度准备把它写成一条真缺陷**。是 `--selftest` 的红灯抓出来的：反事实注入本该让读数翻转，它却没动——因为基线本来就是 `<absent>`。⇒ 与 TM-017 的 `sed` 事故同类，而且更狠：**判据红也不代表源码坏**。

**六、修进工具（真产出）。** ① 名字作为**调用关键字实参**出现也算消费；② 只把形参转给下游的消费者标 `forwarded`、落 `deciding_consumers`，`silently_inert` 只看决策者；③ 容器键位写入单列 `container_writes`，**只作线索、不进 `producers_total`**（字典写了键 ≠ 下游真读到）。回归 verl（8）/SkyRL（7）/OpenRLHF（0）三份归档 `silently_inert` **逐条相同**；单测 16 passed。

**七、另两条不是缺陷。** `enable_disaggregation` 是**编译期按模型接**：38 个 `create_generic` 调用点里只有 3 个（llama／llama4／nemotron）显式传、其余 35 个吃默认 `False`；`dispatch_kv_cache_creation.py` 那两行 `kwargs.pop(...)` / `kwargs[...] = ...` 的 `kwargs` 是**具名形参**（不是 `**kwargs`）。C++ 的 `PrefillInput::is_decode` 写入在 `batch_prefill_base.cc:238` 的聚合初始化第四项 `/*is_decode=*/true`（TM-018 第二类形态），三处消费者真读它。C++ 只有 22 个字段可判**不是方法失效**——MLC-LLM 的 C++ 是服务层。

**核对产物**：`verify_findings.py` **21 checks 0 failed / selftest 4 注入全红**、`probe_router_chain.py` **8/8**（含仓库内快照副本，无克隆也能跑）、`probe_container_write.py` **5/5**、`classify.py` 7/7 全覆盖、`scan_cpp_flags.py --selftest` 7/7。**纯静态 + CPU 源码执行，未装 mlc_llm、未建 TVM、未起服务、未上 GPU/NPU、未触上游。**

**留给下一轮**：① TM-019 是负结论，**不对外**；② 常规扫描池六批已扫完（SGLang／vLLM／LMDeploy／TensorRT-LLM／llama.cpp／MLC-LLM），剩下**训练侧**（AReaL／NeMo-RL／slime／DeepSpeed）未扫；③ TM-015（vLLM `enable_kv_events`）、TM-017（TRT-LLM `use_split_scale_stages`）仍待用户逐项授权；④ TS-001 仍停在「等用户立项」，M5 侧 critic 三臂与 4 卡臂仍等窗口。

入口：[实验](experiments/mlcllm-pyflag-triage/README.md) · [判定](experiments/mlcllm-pyflag-triage/verify_findings.py) · [四跳探针](experiments/mlcllm-pyflag-triage/probe_router_chain.py) · [编译期探针](experiments/mlcllm-pyflag-triage/probe_container_write.py) · [7 条归因](experiments/mlcllm-pyflag-triage/classify.py) · [读数](experiments/mlcllm-pyflag-triage/raw/) · [证据页](open-source/mlcllm/evidence/pyflag-2026-11.md) · [工具第十三类](tools/pyflag/readme.md) · [TM-019](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 上一轮（2026-11-05 04:0x）：TM-016——LMDeploy 扩池，5 条「静默失效」没有一条能对外

**一、为什么做。** TM-015 留给下一轮的第③条：SGLang 与 vLLM 已扫完 → 换池取 LMDeploy，检验前两批的排除判据在新形状上还成不成立。

**二、读数。** `3228395546`（2026-11-03，1113 个 `.py`、**16.5 秒**）：**38 个开关／5 条 `silently_inert`**，逐条绑行号后**真缺陷 0 条**。

**三、最像缺陷的那条也是通的，这是本批最要紧的。** `--disable-fastapi-docs` 从 CLI 一路传到 `serve()`，而 `launch_server` 的九个具名形参里**没有它**——只看签名会判"断在这里"。实际 `cur_server_kwargs.update(kwargs)` → `_run_server` → `serve(model_path, **kwargs)` 一路转发。`probe_kwargs_route.py` 把这三段**真实源码**取出来在 CPU 上串起来跑：`True` → FastAPI 三个 URL 全 `None`、`False` → `docs_url='/'`，**7/7**，并配两条负向对照（收集段换成空 dict／分支换成永远开都要让对应判据变红）。

**四、历史归属：不存在"值曾被吞掉"的版本。** `--disable-fastapi-docs` 由 #2540（`6f34738db0`，2024-10-08）引入；`launch_server` 由 #3414（`776aa1b686`，2025-05-19）引入、**入即 `**kwargs` 转发**。上游查重：#2540／#4263 都在**加**这个功能，无 issue 说它不生效。

**五、另外四条各属一类结构原因。** `use_fa3` 是 import 期 SM/CUDA 探测的模块级全局（真赋值只有 `:15`/`:22`）；`use_proxy` 消费者与 2 个生产者都在 `autotest/`；`use_tqdm` 仓内生产者只在 `benchmark/`，但 `Pipeline.__call__` 透传 `**kwargs` → `infer`；`with_version` 是 `setup.py` 内部 helper（docstring `default=False` vs 代码 `True`，无用户面）。

**六、方法读数（真产出）。** 新增两类易错点并写进 pyflag readme：**第八类 `**kwargs` 兜底**（签名里没有 ≠ 收不到）、**第九类别名调用**（`from X import f as g` 后按 `f` 找读到 0 处）。三批真缺陷率：SGLang 34→2、vLLM 12→1、LMDeploy 5→0。

**七、本批五个自错（已改，都由自检/复核抓出）。** ① 多行调用的关键字实参只看首行 → 读到空集；② 自证 check 要求签名解析器"看见 `disable_fastapi_docs`"，与结论自相矛盾；③ 「别名找得到」与「调用点传了值」混在一条判据 → 注入时连带变红；④ `x = ...` 三种假阳性（形参默认值／关键字实参／续行实参）让赋值点集合反复横跳；⑤ 先剥三引号再剥注释 → docstring 收尾行的 `#` 截断、三引号配不上对。

**核对产物**：`verify_findings.py` **39 checks 0 failed**、`--selftest` **8/8**（红错地方也算失败）；`probe_kwargs_route.py` **7/7**（2 条负向对照）。**纯静态 + CPU 源码执行，未装 lmdeploy、未起服务、未上 GPU/NPU、未触上游。**

**留给下一轮**：① TM-016 是负结论，**不对外**，无需用户授权动作；② 下一批换池只剩 TensorRT-LLM／llama.cpp／MLC（LMDeploy 已扫）；③ TM-015（vLLM `enable_kv_events`）仍待用户逐项授权；④ TS-001 仍停在「等用户立项」，M5 侧 critic 三臂仍等单卡窗口。

入口：[实验](experiments/lmdeploy-pyflag-triage/README.md) · [证据页](open-source/lmdeploy/evidence/inert-flags-triage-2026-11.md) · [TM-016](queue_tech-miner.md)。

### M1 本轮（2026-11-05 03:0x）：TM-015——vLLM 扩池，263 个开关里 12 条静默失效、真缺陷 1 条

**一、为什么选 vLLM。** pyflag 在 verl／SkyRL／OpenRLHF／rLLM 与 SGLang 上都做到「候选 ≤1–2 条」，readme 写明「再扫应换新框架而非重扫」。vLLM 是常规扫描池里最大且完全没扫过的对象，按扩池要求本轮只纳入它一个。

**二、读数。** `da0b6ac2`（2026-11-05，5976 个 `.py`），**2 分 42 秒**扫出 **263 个开关／12 条 `silently_inert`**（`never` 7／`always` 5）。逐条绑行号后：**真缺陷 1、模板硬编码 1、结构原因 10**（属性转发误报 2、config 字典键 1、模型实现内部默认 4、同文件小表 2、测试专用旋钮 1）。

**三、真缺陷是「设计不一致」而不是「开关坏了」，这条口径最要紧。** `enable_kv_events` 的两个 tier 形参都活着（`False`→`None`，`True`+全局开→真的缓冲 `[]`），**唯一构造路径把 tier 配置原样 `**config` 展开** ⇒ 透传字面上成立、**用户手写 `enable_kv_events: true` 就能用**。所以对外只能写「库内无生产者」，不能写 bug。真正值得说的是它和同一矩阵里 `kvcr` 的 `enable_telemetry`（显式 `extra_config.get`）**不同层**。

**四、对照组证明同一批里也有误报。** `use_page_cache` 被报成 `never`，实际 `connector:153 → :232 → worker:49 → disk_backend:104` 四跳全通。`page_cache_e2e.py` 把真实的 `extra_config` 处理段 `exec` 出来跑三档，读数 `parsed=False/True/False`，`verdict: WIRED`。⇒ 这与 SGLang 那批（34→2）是同一条方法读数的第二次复现：**大仓库上总量不能当缺陷清单**。

**五、本批三处自错（都已改，判据进 readme）。** ① 用 `text.count(name)` 判「某处没传」，把 docstring、告警文案、`if flag:` 一起算进去，当场 MISMATCH ⇒ 逐行分类（第七类易错点）；② 自检写成 `ok = False if inj else ok`，8 条注入「全过」但**一条判据都没参与** ⇒ 注入绑到具体 check；③ 行为分叉判定的切片配对写反，判出「没分叉」⇒ 显式 `pick(tier, enable, global)`。

**核对产物**：`verify_findings.py` **37 checks 0 failed**、`--selftest` **8/8**（每条注入让目标判据变红）；`page_cache_e2e.py` exit 0／`--selftest` 1/1；`kv_events_fork.py` `--selftest` 2/2。**全部静态 + CPU 源码执行，未启动 vLLM、未跑 transfer、未发真实 KV 事件、未触上游。**

**留给下一轮**：① TM-015 对外（文档澄清 or 提 issue）**待用户逐项授权**；② 若判「不算缺陷」，本批降级为方法读数；③ 下一批换池应取 TensorRT-LLM／LMDeploy／llama.cpp（SGLang 与 vLLM 已扫完，不宜重扫）；④ TS-001 仍停在「等用户立项」，M5 侧 critic 三臂仍等单卡窗口。

入口：[实验](experiments/vllm-pyflag-triage/README.md) · [证据页](open-source/vllm/evidence/kv-events-per-tier-flag.md) · [判定](experiments/vllm-pyflag-triage/verify_findings.py) · [配置链对照](experiments/vllm-pyflag-triage/page_cache_e2e.py) · [行为分叉](experiments/vllm-pyflag-triage/kv_events_fork.py) · [扫描归档](tools/pyflag/scan-vllm-da0b6ac2.json) · [TM-015](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。


### M1（2026-11-04 20:0x–2026-11-05 00:0x）：TM-014——SGLang 的 34 条静默失效全量归档，真缺陷只剩 2 条
### M1（2026-11-05 01:5x–02:0x）：TM-005／TM-006 运行时对照——「字段在、消费者没了」这次是跑出来的，不是 grep 出来的

**一、为什么做。** TM-006 前三段（配置消费点／接口面／启动配置键）与 TM-005 的原批全是**文本与 AST 静态取证**。静态结论能证明这个名字没人**直接**读，但间接路径（回调实参、反射、动态装配）扫不到，也排除不了"就算有人读、行为也不变"。这一轮换成运行时：**把开关真的传进去，看输出动没动**。

**二、`tiled_mlp` 三层读数。** ① FSDP `_build_module` 那次 `apply_monkey_patch(...)` 的实参**不含 `use_tiled_mlp`**（全仓唯一调用点，`transformer_impl.py:322`）；② **正向对照**：显式传 `use_tiled_mlp=True` 会把 `LlamaMLP.forward` 换成 `_patch_mlp_class.<locals>.tiled_forward`，改写前后 logits 最大差 **5.96e-08**（fp32 舍入，符合"只省显存不改数值"）；③ 照引擎实参调用后 `forward` **仍是原生 `LlamaMLP.forward`**。⇒ 比纯静态强的地方很具体：**上游哪天接上，读数③会直接变，探针自己会报**。

**三、`use_prefix_grouper`。** 补丁层有效（`_PREFIX_GROUPER_PATCHED` `False→True`，这条是正向对照），但 `verl/workers/` 里把 `prefix_grouper=` 喂进前向的地方 **0**、`prefix_grouper_utils` 导入方 **0**，trainer 侧（`ray_trainer.py:367/1197/1238` ＋ `experimental/` 三个 trainer）仍读 4 处 ⇒ 只付约束代价、拿不到加速。

**四、历史归属改口（本轮唯一改结论的地方）。** `d6bc6296`（2026-01-05，#4368）引入 `prefix_grouper_utils.py`；`044bbba2^` 的 `verl/workers/actor/dp_actor.py:138` **真的** import 过 `forward_micro_batch_with_prefix_grouper`、同文件 8 处读该开关；`044bbba2`（2026-04-20，#6067）删掉该文件、新引擎无等价入口 ⇒ **与 `tiled_mlp` 同形，是重构回归**，对外不能写"这个选项从没生效过"。

**五、自己否掉一条推论。** 旧 worker 的 `_build_model_optimizer(..., role="actor")` 是真的，但 `apply_monkey_patch` **在 `044bbba2^` 上就已经没有 `role` 形参**、那次调用也不传 ⇒ "迁移时丢了 role"不成立，不列入证据。**旧代码里读起来自相矛盾的一对调用，先核基线里那条是不是本来就断的。**

**六、本批一处判据设计错（已改）。** 第一版把注入 `tiled_noeffect` 也接在 `engine_flag_inert` 上，自检立刻报"未检出"。追下去是**判据本身错了**——"传进去但不起作用"是另一种 bug，算成"引擎没接线"会让结论在两种完全不同情况下都成立。改为 `engine_flag_inert` **只**由调用点决定，`machinery_alive` 单独成项。

**核对产物**：`probe.py`（exit 0 = 结论成立，2 = 上游已变）、`--selftest` **3 条注入全被接住**、`check_docs.py`（`docs/examples/config.rst:267-273` 与 `examples/` 3 处仍在教 `tiled_mlp`）。verl `31cfccb4`（2026-11-04）、transformers 5.22.1、torch 2.15.0、Python 3.12.13。**纯 CPU 导入真实模块，未跑训练、未上 GPU/NPU、未触上游。**

**留给下一轮**：① TM-005／TM-006 对外（issue／PR／文档修订）**均待用户逐项授权**；② `tiled_mlp` 显存收益不在本轮（已有 M5 4090 冒烟 −46.7%）；③ 只覆盖 FSDP（全仓唯一 `apply_monkey_patch` 调用点），Megatron／mindspeed／automodel 不改 HF MLP 类、本就不适用；④ `prefix_grouper` 第三方包本机未装，那条取自源码关键词而非真的 import 尝试；⑤ `freeze_vision_tower` 仍只有静态证据（本机无 VL 模型），要动需 M5；⑥ SGLang 池已扫完（TM-013／TM-014），再扫应换新框架（TensorRT-LLM／LMDeploy／llama.cpp 均未碰）；⑦ TS-001 侧仍停在「等用户立项」。

入口：[实验](experiments/verl-inert-flags-runtime/readme.md) · [探针](experiments/verl-inert-flags-runtime/probe.py) · [文档面](experiments/verl-inert-flags-runtime/check_docs.py) · [读数](experiments/verl-inert-flags-runtime/raw/probe.json) · [TM-005 证据更新](open-source/verl/evidence/prefix-grouper-inert/readme.md) · [TM-005](queue_tech-miner.md) · [TM-006](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。


### M1 本轮（2026-11-05 05:5x）：TM-017——TensorRT-LLM 扩池，352 个开关里 35 条静默失效、真缺陷 1 条

**一、为什么选 TensorRT-LLM。** LMDeploy 收口后，常规扫描池剩下的最大对象就是 TensorRT-LLM（6084 个 `.py`），本批只纳入它一个。

**二、读数。** `f6405a5965`（2026-11-04 15:08 -0500，6084 个 `.py`、**12 分 39 秒**）：**352 个开关／35 条 `silently_inert`**（never 23 / always 8 / mixed 4）。逐条绑行号后：**真缺陷 1、结构原因 30、死代码与内部面 4**。原始读数靠不住这件事第四次复现。

**三、真缺陷的形状是「加了没接，而且兄弟挡路」。** `use_split_scale_stages` 有声明、有分支、确实没生产者——这一半和普通误报长得一样。区别在两处 `elif` **永远轮不到**：同族四个兄弟 constexpr 在 QK 调用点全被显式置 `True`，且在 `if/elif` 链上优先。`probe_split_scale.py` 把那段真实源码文本抽出来按调用点实际常量执行九条判据：真实取值 → 两块都不取 split arm（B7）；兄弟关掉、开关打开 → split arm 真的活且偏移与 else arm 不同（B8/B9）。

**四、历史归属：合入当天就是这样。** 文件由 #19070（`4bbe1af658`，2026-09-18）一次性新增 7859 行，同一开关在该 commit 里只有 2594（声明）／2674／2681（两处 `elif`）⇒ 不存在「曾经接上过」的版本。上游查重 `"use_split_scale_stages"` 0 条 issue。

**五、本批方法读数（真产出）。** 第十类 **`**dict` 展开**（字典字面量写键、调用点不出现标识符，AST 的 `Call.keywords` 是空的——第八类 `**kwargs` 的发送端镜像）；第十一类 **兄弟开关恒先命中**（比"加错/丢线"更隐蔽：代码本身对、不崩、不出告警，只是那段优化永远走不到）；**「显式传 `None`」也是生产者**（`use_replay_state_update=None` 语义是"未指定"，结构上是实参）。

**六、本批一处工具坑（已修）。** `--json` 不带值时写 `scan-<目录名>.json`，目标写成 `.` 时目录名是空串 ⇒ 落成 `scan-.json` 并互相覆盖；本批两次扫描都写进了同一个文件。现在 `--json <path>` 写指定路径、裸 `--json` 写 `scan-<dir|repo>.json`，readme 已同步。

**七、本批一处真事故（比上面六个自错严重得多）。** 负向对照用 `sed -i ''`（macOS 写法）在 zsh 下 `''` 不是参数，备份与还原 `cp` 都没执行，**把快照里 QK 调用点的四兄弟 `True` 改成了 `False`**。更糟的是 26 checks 与 probe 9/9 **全部照绿**——判据只校验字面量形态，不校验"读的是不是那份快照"。我据此一度误判"README 写反了、主文件其实是 False"，比对 `git hash-object` 与 HEAD blob 后才定位。已 `git checkout` 还原（blob 与 HEAD 一致），加前置 `S0.snapshot_pristine`（`git status --porcelain` 非空即 `exit 2`、不产出结论），负向对照走完"故意污染 → exit 2 → 恢复 → 28 checks 0 failed"。同时补 `F1h.pv_dead_by_enclosing_guard`：PV `:4522` 传 `stage_s*f_from_smem=False`，整个 `if` 块进不去 ⇒ **PV 与 QK 是两种死法**，原记录折成"两个调用点都不传"会把修法藏掉。`shadowed_arms()` 已进扫描器并调了六轮判据（逐轮见 diary「后续补强」），最终此快照上**主文件命中 4 条**（`use_split_scale_stages` 两处 + `use_split_operand_stages` 两处）、孪生文件 4 条；非测试合计 9 条，人工复核 8 真 1 假（`has_bias`，同名不同函数的固有局限）。

**核对产物**：`verify_findings.py` **28 checks 0 failed**（含 `S0` 前置）、`--selftest` **6/6**；`probe_split_scale.py` **9/9**、`--selftest` **2/2**；`classify.py` **35 条全覆盖**（缺一条 exit 1）；pyflag 单测 **14 passed**。**纯静态 + CPU 源码文本执行，未装 TensorRT-LLM、未编译、未起服务、未上 GPU/NPU、未触上游。**

**留给下一轮**：① TM-017 对外（issue/PR）**待用户逐项授权**；② 下一批换池剩 llama.cpp／MLC-LLM（SGLang／vLLM／LMDeploy／TensorRT-LLM 均已扫）；③ TM-015（vLLM `enable_kv_events`）仍待用户逐项授权；④ TS-001 仍停在「等用户立项」，M5 侧 critic 三臂仍等单卡窗口。

入口：[实验](experiments/trtllm-pyflag-triage/README.md) · [判定](experiments/trtllm-pyflag-triage/verify_findings.py) · [行为探针](experiments/trtllm-pyflag-triage/probe_split_scale.py) · [35 条归因](experiments/trtllm-pyflag-triage/classify.py) · [读数](experiments/trtllm-pyflag-triage/findings.json) · [扫描归档](tools/pyflag/scan-trtllm-f6405a5965.json) · [TM-017](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M1 本轮（2026-11-05 07:4x）：TM-018——llama.cpp 扩池，**37 条候选没有一条是缺陷**

**一、为什么先扫 Python 会得到空结果。** `scan_unwired_flags.py` 扫 `3de52ac3` 的 250 个 `.py`：**7 个开关、0 条 `silently_inert`**——这些是 `conversion/`（115 个模型格式转换）、`gguf-py/`、`scripts/` 与 server 测试。⇒ 对 llama.cpp 用 Python 版没有信息量，「已扫过五个框架」里它要按 C++ 重新计数。

**二、C++ 的判定口径与读数。** 形态是结构体成员默认值（`struct X { bool f = false; }`），三步不变：**字段声明 → 消费者（`if`／三目／`&&`／`||`）→ 生产者（赋值／实参／聚合初始化）**。`3de52ac3`（1574 个 C/C++ 文件、约 4 分钟）：**1636 个有分支字段 → 37 条 `unwritten` → 真缺陷 0 条**。

**三、三类 C++ 特有的写入形态（本批真正的方法产出，各写成一条判据，删掉对应行必须变红）。** ① **单行 `if` 里赋值**：`common/arg.cpp:3862` 的 `/**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }` 既是条件行又是写入行，先判消费者会把整个字段记成只读（71 条里有 8 条出自这里）。② **`/* name */ value` 聚合初始化**：`server-context.cpp:1729-1736` 的 `chat_params = { /* allow_image */ mtmd_cap.vision, … }`，剥注释后整行只剩 `mtmd_cap.vision,`。③ **位置实参**：`drama3-gen.cpp:12` 的 `push_back({"tower_pos", 1, 1, true})`（第四项是 `in_place`），而同结构体在 `qwen3tts-gen.cpp:624` 只传三项 ⇒ 必须两个文件都看；这条没有通用修法，是「报没人写必须人读」的具体例证。

**四、唯一被怀疑过的那条，复核后是接通的（也是本批最严重的一处自错）。** `reasoning_control` 被扫成 `unwritten`，因为看起来唯一的写入是 `server-common.cpp:1530` 写 JSON。实际 `server-schema.cpp:351` 用 `field_bool(name, bool &)` **按引用绑定**字段，写入在 `field_bool::eval`（`:636-647`）靠引用完成；`sampling.cpp:315` 的守卫真读它。第一版探针把「写不写该字段」当成守卫唯一变量，而守卫里还有 `!reasoning_budget_end.empty()`、其形状取决于模板有没有 thinking 标记 ⇒ 读数正好反过来，报了一个不存在的缺陷。重写成 6 档网格后 7 条判据全过。⇒ **第十二类易错点：绑定 ≠ 赋值**。

**五、两条查过的负结论（都不建议对外）。** `filter_allowed_keys` 的配套白名单在 #24739（`552258c535`）被有意移除、上游 #27929 开放讨论中；`show_key_map_stats` 的唯一写入机会是构造函数而构造函数不收它，字段与 `if` 分支由 #19253（`b4d05a3d2f`）同一次提交引入、从未接命令行。

**核对产物**：`scan_cpp_flags.py --selftest` **7/7**（含 3 条负向对照）、`classify.py` **37/37 exit 0**、`verify_findings.py` **15 checks 0 failed** / `--selftest` **4 注入全红**、`probe_reasoning_control.py` **7/7** / `--selftest` **6/6**、pyflag Python 单测 **14 passed**。**纯静态 + CPU 源码执行，未编译 llama.cpp、未起 server、未跑推理、未触上游。**

**留给下一轮**：① TM-018 是负结论，**不对外**，无需授权动作；② 常规扫描池已扫五批（SGLang／vLLM／LMDeploy／TensorRT-LLM／llama.cpp），剩下 MLC-LLM 与训练侧（AReaL／NeMo-RL／slime 等）没扫；③ TM-015（vLLM `enable_kv_events`）、TM-017（TRT-LLM `use_split_scale_stages`）仍待用户逐项授权；④ TS-001 仍停在「等用户立项」，M5 侧 critic 三臂仍等单卡窗口。

入口：[实验](experiments/llamacpp-pyflag-triage/README.md) · [扫描器](tools/pyflag/scan_cpp_flags.py) · [37 条归因](experiments/llamacpp-pyflag-triage/classify.py) · [判定](experiments/llamacpp-pyflag-triage/verify_findings.py) · [接线探针](experiments/llamacpp-pyflag-triage/probe_reasoning_control.py) · [证据页](open-source/llamacpp/evidence/cpp-fields-triage-2026-11.md) · [TM-018](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

## 当前现场

> **推送已恢复（2026-11-04 16:2x）**：此前积压的 5 个本地提交（`b850df424`／`ab204152f`／`db7a7aeed`／`3860b0b6b`／`9816bf5f8`）与本轮提交已全部推上远端 main。`git push` 仍连不上 `github.com:443`（75 s 超时），改用 [REST 推送工具](tools/git-push-api/readme.md)（`api.github.com` 正常）；远端新提交 `01c8bf13` 的 SHA 与旧本地不同（消息与树相同），已 fetch 并 `reset --hard` 对齐，本地树 `618e2a82` 与推送内容一致、`--verify` 通过。⇒ **下次网络同样断开时优先走 `tools/git-push-api`，不要反复等直连。**

### M5 本轮（2026-11-05 08:0x–08:4x CST）：无卡轮——窗口里读不到计数器等于没跑：死登记删除 + 受测树硬门槛

**连通性（探一次，20 分钟后再探一次，均失败）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、本机仍无 `10.205.0.0/24`／`192.168.13.0/24` 路由。本轮**零 GPU／NPU 读数**。

**为什么做这个。** 4 卡 `inflight=1` 臂存在的意义就是读 `training/sampler/blocking_polls`（判断一个训练步里有几次派发轮次）。上一轮做出补丁后，`run-poll-dapo.sh` 不检查树、preflight 不检查补丁 ⇒ **未打补丁的树照样 exit 0**，4 张卡烧完才发现没有计数器。

**一、上一版补丁有一条不可达登记（已删）。** `validation/sampler/blocking_polls` 永不出现：全仓 `partition_id="val"` 只有 `trainer_base.py::_validate()` 一处，且是 `batch, _ = self.replay_buffer.sample(...)`（metrics 被丢弃）。补丁 **+9→+8 行**、`sha256=4c142752f78d02b3…`；对 `31cfccb4` 的 `replay_buffer.py`／`utils.py` `patch -p1 --fuzz=0` 通过，且与 `poll-interval-config.patch` **文件集不相交**（顺序无关）。

**二、旧自检为什么没抓到：它数的是字符串。** 旧检查只断言两个键名都在 `utils.py` 里，写错规则、写进死路径都能过。现改为把真实 `MetricsAggregator` 类源码 `exec` 出来喂两次迭代（2 与 3）：打补丁树 **5.0**（sum）、未打补丁树 **2.5**（默认 weighted_avg）。另加不变量"注册了 validation 键 ⇔ val 调用点保留 metrics"（现在一真一假；上游哪天记录 val metrics 就翻红）。

**三、sim 扩到两个类。** `sim-loop.py` 现在 exec `ReplayBuffer` 与 `ReplayBufferAsync`，新增 async 场景（`wait_for_sampleable` 按次清零、`sample()` 转发 metric，3 次轮询读数 3）；打补丁／未打补丁树各 **12/12**，惰性 **480 行逐字段相同**。

**四、把"树必须是受测树"变成三道硬检查。** [prep-tree.sh](experiments/verl-v1-poll-quantum-dapo-g4/prep-tree.sh)（复制＋两补丁＋4 标记，dst 已存在拒绝）；`run-poll-dapo.sh` 的 `CHECK_TREE=1`（默认，缺任一补丁 **exit 65** 并打印补救命令；`CHECK_ONLY=1` 只查树）；[preflight-g4.py](experiments/verl-v1-poll-quantum-dapo-g4/preflight-g4.py) 新增**必填** `--run-tree`（树 4 标记＋在受测树本体跑 sim＋wrapper 守卫 0/65 双对照），并接受"已打补丁的树"（`already_applied`）。自检 **7→11/11**。

**五、口径变化。** ① 基线臂此后也跑同一受测副本（不传 poll_interval，回读默认 2.0），与归档"原树基线"只差一条惰性补丁 ⇒ 窗口内 A/B 自洽，跨批次绝对值不要混用。② `separate_async` 会对同一键记两次（`prepare_step` 的 switch-wait 与 `sample()` 各一次 `wait_for_sampleable`），本臂（sync）不涉及。

**边界**：零 GPU 读数；计数器仍未在服务器上执行过；`CHECK_TREE=1` 会挡住"在未打补丁树上重跑旧配置"，需要时显式 `CHECK_TREE=0`（preflight 的 argv 捕获即如此）。

**留给下一轮**：① critic 三臂仍需**单卡窗口**；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口，开窗前按"prep-tree.sh → preflight（`--run-tree`）→ run-poll-dapo.sh"三步走，跑后读 `blocking_polls`（预期 ≥2）；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做；④ 昇腾需用户重连 VPN；⑤ 对外动作仍待用户逐项授权，本轮**未触上游**。

**产物**：[计数器补丁](experiments/verl-v1-poll-quantum-polls/poll-count.patch) · [sim-loop.py](experiments/verl-v1-poll-quantum-polls/sim-loop.py)（12/12） · [读数](experiments/verl-v1-poll-quantum-polls/raw/sim-polls.json) · [未打补丁对照](experiments/verl-v1-poll-quantum-polls/raw/sim-polls-pristine.json) · [prep-tree.sh](experiments/verl-v1-poll-quantum-dapo-g4/prep-tree.sh) · [run-poll-dapo.sh](experiments/verl-v1-poll-quantum-dapo-g4/run-poll-dapo.sh) · [preflight-g4.py](experiments/verl-v1-poll-quantum-dapo-g4/preflight-g4.py) · [开窗前读数](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4-instrumented.json) · [TM-008](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M5 上一轮（2026-11-05 07:0x–07:4x CST）：无卡轮——采样等待是**量化**，不是固定扣 2s；顺带给下一次窗口补上轮询计数器

**一、为什么做。** §13.49 把 4 卡 DAPO 的等待拆成"固定等待"后写成"每步平均等待 ≈0.59×poll_interval"，并记 2 卡同模型是 0.98，结论停在"等待量随配置变、不能当常量迁移"。这描述了现象但没解释斜率为何变，TM-008 对外写作时也只能写成方向性收益。

**二、模型。** `sample()` 的结构是"回读元数据 → 不够就 `sleep(P)`"，所以训练步被释放的时刻是**第一个越过终态**的轮询点：`gen(P) = ceil(D/P)·P + c`（D 为本步生成耗时，c 为最后一个 sleep 之后的工作）。它只说**何时被释放**，不说 GPU 快慢，硬签名是取值只能落在网格上。

**三、读数一（不配对、不拟合斜率）。** 稳态 steps 3–14、0.1 s 容差聚类：2 卡固定批 P=2.0 两簇 **2.1575（n=6）／4.1717（n=6）**，间隔 **2.0142**、簇内 0.0115／0.0294；同一批运行 P=0.5 两簇 **2.1621／2.6606**，间隔 **0.4985**；4 卡固定批 P=2.0 单簇 2.1607（散度 0.0264）。五个簇、三个独立臂给出**同一偏移 c = 0.1625 ± 0.0048 s**——`timing_s/gen` 里约 0.16 s 不属于任何 sleep，此前没被单独记过。

**四、读数二（逐步预测 + 负向对照）。** 用 P=0.1 臂的 D 预测 P=2.0 的释放时刻：4 卡固定批 **12/12**（最大偏差 0.021、sd 0.0083），2 卡 **9/12**（3 步的 D 压在 2.0 边界、跨运行抖动 0.3 s 导致整格翻转）。负向对照：错格子 1.0／0.5／3.0 全部 **0/12**；打乱 D 与 step 配对 500 次，中位 7/12（最小 5）——退化但不如错格子彻底，因为 2 卡臂 D 分布窄，所以这条配对读数的信息量弱于簇结构。

**五、负结论（预期内）。** DAPO 三个臂对单轮公式只有 0–4/12 命中：一个训练步里有多次派发轮次（过滤逐出后按 `2k` 补发、受 `max_inflight_gen_batches` 限流），每轮各自量化一次。⇒ 归档的斜率差异是**同一机制的不同相位与轮数**，不是另一个机制；固定批 4 卡 0.21／2 卡 0.51，DAPO 0.59／0.98。

**六、计数器（本轮可执行产物）。** `poll-count.patch`（2 文件 +9 行）在 `_wait_for_next_poll` 计数、`sample()`／`wait_for_sampleable()` 出口写 `{prefix}/sampler/blocking_polls`，并在 `MetricsAggregator` 的 `sum` 规则登记（`parameter_sync_step>1` 时按迭代求和；本系列 `v1.sync`／`v1.colocate_async` 配置块为空、默认 1，即每步一次）。验证：`sim-loop.py` 从树里取出 `ReplayBuffer` 类源码原样 `exec`，只桩数据层与时钟，虚拟时钟数真实 `sleep`——打补丁／未打补丁树各 **9/9**，补丁惰性 **120 场景 ×4 步 = 480 行逐字段相同**。

**七、边界。** 零 GPU 读数；逐步配对假设两次运行同 workload，2 卡那 3 个边界步已证明该假设在 D 接近边界时失效；c 的归属（元数据回读／选择／materialise／计时边界）未拆；计数器尚未在服务器上跑过。

**八、留给下一轮。** ① 4 卡 `inflight=1` 臂仍要 ≥4 卡窗口，开窗前先 `preflight-g4.py`，届时读 `blocking_polls`（预期 ≥2；若恒为 1，则归档斜率差不能归因轮数）；② 2 卡 DAPO 臂已测完，无需重跑；③ 对外动作仍待用户逐项授权。

**成果入口**：[实验](experiments/verl-v1-poll-quantum-polls/README.md) · [量化读数](experiments/verl-v1-poll-quantum-polls/raw/quantization.json) · [分析脚本](experiments/verl-v1-poll-quantum-polls/analyze-quantization.py) · [计数器补丁](experiments/verl-v1-poll-quantum-polls/poll-count.patch) · [循环自检](experiments/verl-v1-poll-quantum-polls/sim-loop.py) · [自检读数](experiments/verl-v1-poll-quantum-polls/raw/sim-polls.json) · [TM-008](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。

### M5 本轮（2026-11-05 06:0x CST）：无卡轮——4 卡 `inflight=1` 臂的开窗前检查，**主分支的 base recipe 少了一个钉**

**连通性（只探一次＋45 s 后复探一次，均失败）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时（走 default 路由，公网路径断，与 10-28 同类）；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时，且 `netstat -rn` 里**根本没有** `10.205.0.0/24`／`192.168.13.0/24` 的路由 ⇒ **昇腾 VPN 本机未连通**，不是服务端抖动。本轮**零 GPU／NPU 读数**。

**为什么换到这件事。** critic 三臂已连续五轮只能做无卡准备，再加静态打磨没有增量；而 TM-008 只剩 4 卡 `inflight=1` 一条没测，要的是 4 张空闲卡——本机最稀缺的资源（11-04 起长期只有 1/4/5 空闲）。这条臂是针对 `24f25b03` 准备的，main 已到 `31cfccb4`。

**一、失败模式是"不报错，训练另一套配置"。** `run-poll-dapo.sh` 完全靠环境变量驱动 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 拼命令行（19 个变量＋recipe 里的 47 条 override）。env 改名或 recipe 少一条 override，都不会抛错。

**二、真的少了一条。** 逐行 diff 该 recipe（`24f25b03` vs `31cfccb4`）**只有一处**差异：`actor_rollout_ref.rollout.enable_chunked_prefill=False` 被删。其默认值是 **True**（`verl/workers/config/rollout.py:291`、`_generated_ppo_trainer.yaml:287`，两版本相同）⇒ 同一份 wrapper 在 main 上把 `True` 交给引擎，而本目录归档的七跑全是 `False`。

**三、不是"vLLM V1 反正总开着"那类空转。** vLLM 0.24 的 `EngineArgs` 里该字段是 `bool | None = None`，`None` 才回落到 `is_chunked_prefill_supported`；显式 `False` 走另一条分支——`_set_default_chunked_prefill_and_prefix_caching_args` 打 `warning_once("This model does not officially support disabling chunked prefill…")`，调度器也不再切分 prefill（`vllm/engine/arg_utils.py`，tag `v0.24.0`，**源码阅读、未执行**）。verl 侧唯一读该值的分支（`_validate_configs` 把 `max_num_batched_tokens` 抬到 `max_model_len`）在本臂**空转**：默认 8192 > 2048，条件不成立 ⇒ **配置面回读两档都是 8192，差别只在引擎手里**。

**四、把判定做成检查（`preflight-g4.py`，自检 7/7）。** 关键设计：override 表**不在检查脚本里重抄**。桩 `python3` 让 wrapper→recipe **打印真实 argv**，再拿这份 argv 做 Hydra 组合与回读——env 层与 override 层同时在检。读数（[preflight-g4.json](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4.json)）：58 条 override（recipe 47＋wrapper 11）／与归档树逐键 **`drift.same=true`**／poll 补丁对 `31cfccb4` 三文件 3/3 `--fuzz=0` 通过／两臂组合 18 个预登记键全中／`trainer.v1.sampler.poll_interval` 未打补丁树**不存在**（buffer 用构造默认 2.0）、打补丁树回读 `0.1`／负向对照如期失败。对照读数（[preflight-g4-nopin.json](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4-nopin.json)）：去掉那个钉恰好少一条 override，main 上生效值 `true`。

**五、本批三处自错（都已改）。** ① 本地 bash-3.2 适配写成 `set -xe pipefail`——`set` 把 `pipefail` 当**位置参数**，`"$@"` 的 10 条尾部 override 被吃掉，捕获只剩 48 条（"看起来在检查、其实换了对象"），现已是自检项；② 自检注入打在 `rollout.n` 上而受检键是 `actor_rollout_ref.rollout.n` ⇒ "值被改"这条判据**没被覆盖却报绿**；③ `+` 前缀检查拿 `key` 比 `key=value` 列表，真实运行与自检双双击出误报。

**边界**：零 GPU 读数；"生效值"是 Hydra 解析结果，不是 vLLM 运行时行为；本机只有 bash 3.2，recipe 的空 `EXTRA=()` 在 `set -u` 下报错，本地用"只去掉 `-u`"的副本（服务器 bash ≥4 不需要），替换方式记进读数；归档那七跑实际用的是 `False`，而按源码生成式 decoder 的 `is_chunked_prefill_supported` 为真（`model.py:1790-1791`）⇒ 会命中 `arg_utils.py:2472-2480` 那条"手动关闭 chunked prefill 可能崩或算错"的 `warning_once`（七跑各 15 步 exit 0；若改用默认 `True` 则归档数值不可直接对比）。

**留给下一轮**：① critic 三臂仍需**单卡窗口**；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口，开窗前先跑 `preflight-g4.py`；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做；④ 昇腾需用户重连 VPN（本机无路由），无 NPU 事项；⑤ 对外动作仍待用户逐项授权，本轮**未触上游**。

**产物**：[实验 README](experiments/verl-v1-poll-quantum-dapo-g4/README.md#2026-11-05-无卡轮4-卡-inflight1-的准备base-recipe-少了一个钉臂会静默换配置) · [preflight-g4.py](experiments/verl-v1-poll-quantum-dapo-g4/preflight-g4.py) · [读数](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4.json) · [无钉对照](experiments/verl-v1-poll-quantum-dapo-g4/raw/preflight-g4-nopin.json) · `run-poll-dapo.sh`（已钉回） · [TM-008](queue_tech-miner.md) · [handoff](experiments/handoff-m5.md)。

### M5 上一轮（2026-11-05 05:0x CST）：无卡轮——value-head 分支在本机真跑，**探针自检在服务器上会崩**

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢。本轮**零 GPU 读数**。

**为什么做这个。** 04:0x 的结论（Qwen2 走 `AutoModelForTokenClassification` 第一分支、trl 全程不 import）建立在固定版本**源码阅读**上，当时写的边界是“本机 py3.9 装不上 transformers，real-mapping 分支第一次执行在服务器上”。这条边界把“探针本身是否能跑”留给了稀缺资源去验证——正是前几轮反复在收口的那类风险。

**一、真跑立刻抓到一个只在服务器上才会出现的崩。** `probe-valuehead-branch.py` 的 `selftest()` 里 `case()` 存 4 元组、real-mapping 三行存 3 元组，报告循环 `for name, ok, got, want in results` 一律解包 4 个 ⇒ **只要 `transformers` 能 import 就 `ValueError` exit 1**。旧版原样复现见 `raw/probe-selftest-prefix-crash.txt`，修复后同一 venv **8/8**（`raw/probe-selftest-transformers-5.16.1.txt`）。wrapper 只调 `--model`，所以不烧窗口，但说明**“自检通过”的覆盖面取决于跑自检的机器**。

**二、分支结论从源码阅读变成运行时读数。** `repro-valuehead-branch.py`：真实 `Qwen2ForCausalLM` 存 tiny checkpoint（hidden 32／2 层／vocab 128），按 `load_valuehead_model` 同组实参调 `AutoModelForTokenClassification.from_pretrained(..., config=cfg, attn_implementation="sdpa")`，`num_labels=1` 与 dropout 覆盖照抄 `transformer_impl.py:291-300`。读数 `raw/valuehead-branch-runtime-5.16.1.json` **7/7**；`score` 新头只有 33 参数（hidden 32＋bias）；前向 `(1,4,1)`→`(1,4)` ⇒ `prepare_model_outputs` 走 `output.logits.squeeze(-1)`。**未打补丁的 FA2 路径**抛 `ImportError: ... FlashAttention2 ...`，verl 的 `except BaseException` 把它翻成 `RuntimeError: ... please install trl ...` ⇒ 窗口里不打补丁时的报错**指向错误的包**。

**三、preflight 的 msprobe 检查原来是 stat。** 目录在、包被清一半时会放行，错误要到第一个训练步（模型加载之后）才出现。现用同一解释器/PYTHONPATH 执行 `from msprobe.pytorch import PrecisionDebugger` 并解析 `MSPROBE_CONFIG`，失败 exit 65；mock 夹具补可导入 stub，`run-mock.sh` 新增“目录在、包删掉 → 必须 65”的负向对照（PASS），三臂管线仍 rc=0／judge exit 0。

**边界**：本机 torch 2.15.0（服务器 2.11.0+cu129）；tiny 随机权重只证明“能载入＋头是新的＋形状对”，**不证明 0.5B 真权重数值行为**；sdpa↔FA2 数值等价仍未测；`/tmp/m5-trf-venv` 是一次性物料，重建命令见实验 README。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（依赖名册现在按证据分档、msprobe 预检真 import）；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作仍待用户逐项授权。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `repro-valuehead-branch.py`（自检 2/2；读数 7/7） · `raw/valuehead-branch-runtime-5.16.1.json` · `raw/probe-selftest-prefix-crash.txt` · `raw/probe-selftest-transformers-5.16.1.txt` · `run-critic-pdbg.sh`（msprobe 真 import） · `mock/run-mock.sh`（负向对照） · [handoff](experiments/handoff-m5.md)。

### M5 上一轮（2026-11-05 04:0x CST）：无卡轮——critic 的模型构建分支查清，**preflight 把 trl 当硬依赖是误设**

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时。本轮**零 GPU 读数**。

**为什么做这个。** 三臂的每个"会在窗口里炸"的前提都收口到只剩开跑，但**依赖名册本身**从没被问过。`run-critic-pdbg.sh` 从 01:0x 起把 `/tmp/m5-trl-0.27.0` 和 msprobe 放进同一个 `for dep` 循环，任缺即 `exit 65`——服务器 `/home` 100% 满、`/tmp` 被清一部分不是假设 ⇒ **一个本来能跑的好窗口会被自己的预检判死**。

**一、读数：Qwen2 critic 走第一分支，trl 全程不 import。** `load_valuehead_model`（`verl/utils/model.py:467`）先试 `AutoModelForTokenClassification`（`:471`），只有 `except BaseException`（`:479`）才 import trl（`:488`）。而 transformers 的 `_get_model_class`（`auto_factory.py:178`）只做 `model_mapping[type(config)]` 一次查表，`config.architectures` **只用于在候选类列表里挑一个**（`:184-187`）⇒ 拿 causal-LM 权重去建 token-classification 模型**不报错**（随机初始化 `score` 头）。Qwen2 在映射里（`modeling_auto.py:1686`），引擎又在调用前把 `hf_config.num_labels` 设成 1（`transformer_impl.py:294`）⇒ 分支 1 恒胜。

**二、顺带一条对证据页有用的口径。** 下游 `prepare_model_outputs` 按 `hasattr(module,'v_head')` 分派（`:1538`／`:1559`），Qwen2 走 `output.logits.squeeze(-1)` ⇒ critic 的 value 头是**从零训练的全新 1 维头**，不是预训练 value model；写材料时不能笼统说"加载了 value model"。

**三、依赖名册改成两类。** msprobe 仍是硬依赖（没有它 profiler 是 no-op、`au` 正臂不可能出 dump，窗口会白烧成"无效跑"）；trl 只在回退分支才需要。新探针 [probe-valuehead-branch.py](experiments/verl-profiler-oom-and-critic-gpu/probe-valuehead-branch.py) 只读 `config.json`（不加载权重、不碰 GPU），wrapper 按它的退出码分派，四档行为用桩探针**实测**过：分支 1＋trl 缺 → 继续（原误杀）；分支 2＋trl 缺 → 中止 65 并打印确切安装命令；不可判定＋trl 缺 → 中止（沿用旧严格规则）；不可判定＋trl 在 → 继续。

**四、判定不动。** `verify-critic-pdbg.py` 只把 `valuehead_branch` 打进 reported（不进任何门槛），自检 12→**14**，新增两条钉住"缺失或改成 `trl_valuehead` 都仍 exit 0"。每臂 `arm-<ARM>.json` 记 `valuehead_branch`（含 `resolved_class`）。

**边界**：对**固定版本源码**的核对（verl `31cfccb4`、transformers 5.16.1、trl 0.27.0，两个 wheel 的 sha256 与行号进 [读数](experiments/verl-profiler-oom-and-critic-gpu/raw/valuehead-branch-evidence.json)），**不是运行时读数**；本机解释器 3.9 装不上 transformers 5.16.1，探针的 real-mapping 分支**第一次执行在服务器上** ⇒ 失败方向必须保守：任何异常都归"不可判定"、沿用严格规则，最坏丢一个窗口，不会静默换错分支。trl 0.27.0 两条导入路径与 `pretrained_model`／`v_head` 都在，回退分支本身成立，只是这条路用不到。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（依赖名册现在也按证据分档了）；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作仍待用户逐项授权，本轮**未触上游**。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `probe-valuehead-branch.py`（自检 5/5） · [读数](experiments/verl-profiler-oom-and-critic-gpu/raw/valuehead-branch-evidence.json) · [证据页三·补二](open-source/verl/evidence/precision-debugger-stage-producers/readme.md) · [handoff](experiments/handoff-m5.md)。

### M5 上一轮（2026-11-05 03:0x CST）：无卡轮——critic 不是"少一行装饰器"，补上也解析不出模型

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时。本轮**零 GPU 读数**。

**为什么做这个。** 证据页第三节一直写着"补装饰器要动引擎层的 role 身份，属设计选择"——这是**推断**，而且它默认了"补上装饰器就能采到"。本轮把这条链路走完：`PrecisionDebuggerProfiler.start()` 在 `os.makedirs` 与构造 msprobe **之前**先要 `_resolve_model(self_instance, stage)`，它按 `_MODEL_ATTRS_BY_ROLE[role]` 在 worker 对象上找模型、兜底 `self.module`。

**一、读数：解析在新 worker 上必然失败。** 探针 [probe-critic-model-resolve.py](experiments/verl-profiler-oom-and-critic-gpu/probe-critic-model-resolve.py) 用 AST 从**真实文件**抽 `_STAGE_TO_ROLE`／`_MODEL_ATTRS_BY_ROLE`／`_resolve_model`／`start` 等方法源码执行（CPU、无 torch／msprobe／GPU），替身 worker 的属性名**从同一棵树的真实类体读出**：`TrainingWorker` 只有 11 个 `self.*` 赋值，含 `engine`，**不含 `module`、不含 `critic`** ⇒ `critic_update`／`compute_values` 解析均 **None**；`start(model=None)` 返回 False、打一条 `PrecisionDebugger model not resolved for stage 'critic_update'`、**不建目录**（`makedirs` 在模型检查之后）。正对照：`ActorRolloutRefWorker` 的 `actor.engine.module`／`ref.engine.module` 都命中 ⇒ 不是表全坏，是 critic 那一支没有落点。

**二、这不是"表坏了"，是 worker 布局变了。** 两棵树的 `_STAGE_TO_ROLE`／`_MODEL_ATTRS_BY_ROLE` **逐项相同**；父提交 `6eeb5711` 的 `CriticWorker` 有 `self.critic_module`（`fsdp_workers.py:1651`），正好是表里 critic 的最后一个候选 ⇒ 旧布局能解析，新布局（`Role.Critic -> TrainingWorker`）不能。⇒ **双缺口**：装饰器丢了，解析链的落点也没迁。

**三、role 身份问题还有一层具体形状。** `train_batch` 同时服务 actor 与 critic（`ActorRolloutRefWorker.update_actor → self.actor.train_mini_batch → self.train_batch`）⇒ 给它写字面量 role 无论写哪个角色都会错一边。

**四、候选修法的形状已验证（未改上游）。** 给每个 role 的候选链**末尾**追加 `"engine.module"` 后，critic 两臂解析成功，actor／ref 两个正对照读数**不变**（原候选先命中）。但装饰器与 role 身份仍要解决 ⇒ 最小可靠对外做法仍是"文档标注＋加载期一次 warning"，不是偷偷补一行。

**边界**：替身不是真 worker；未装 msprobe、未跑训练、未上卡。它证明的是"这条解析路径在 `TrainingWorker` 上拿不到模型"，不是真机上 msprobe 的行为。三臂真机跑的 P2／P3（`cu`／`cv` 零 dump）**两种原因都能满足**（没有装饰器，或装饰器在但解析失败），本轮把后者从推断变成静态可查。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（判定与计数器都已验过，本轮又堵掉一个"补丁无效"的岔路）；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作仍待用户逐项授权，本轮**未触上游**。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `probe-critic-model-resolve.py`（自检 8/8） · [main 读数](experiments/verl-profiler-oom-and-critic-gpu/raw/critic-model-resolve-main-31cfccb4.json) · [父提交读数](experiments/verl-profiler-oom-and-critic-gpu/raw/critic-model-resolve-pre6067.json) · [证据页三·补](open-source/verl/evidence/precision-debugger-stage-producers/readme.md) · [handoff](experiments/handoff-m5.md)。

### M5 上一轮（2026-11-05 02:0x CST）：无卡轮——critic 判定的**读数**拿 104 份真日志验了一遍，两处会误判的都改掉

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时。本轮**零 GPU 读数**。

**为什么做这个。** 01:0x 把判定预登记了，但**判定读的数**没拿真日志核过。预登记只保证"不按结果挑标准"，不保证"量出来的东西是它声称的那个量"。

**一、`steps_seen` 在原实现下恒为 0 ⇒ 有效跑会被判无效。** 原实现 `re.finditer(r"^step:(\d+)", text, re.M)`，而 Ray 写控制台时每行都有 `<ANSI>(TaskRunnerV1 pid=…)<ANSI> ` 前缀，**没有一行以 `step:` 开头**。读数来自全仓 **104 份真控制台日志**（[check-log-counters.py](experiments/verl-profiler-oom-and-critic-gpu/check-log-counters.py) `--verify`，逐文件表 `raw/log-counter-check.json`）：旧计数器 **104/104 读到 0**；新计数器（每个训练步一个 `timing_s/step` 标记，取不同步号）与标记数 **104/104 一致**，合计 **1432 步**；随手去掉行首锚点也有 **103/104** 读错（g8-g4w 读到 58 而真值 12、`run-pda` 读到 46 而真值 2——`timing_s/step:8.2` 与 `global_step_46` 都会被当成步号）。⇒ 判据只能是"带 `- ` 分隔的步行 ＋ `timing_s/step`"，它对裸行与带前缀两种形状都成立。V2 门槛（`≥2`）不动，只换量法。

**二、为什么上一轮的端到端自检没抓到：夹具不是真日志的形状。** 假 example 脚本打印裸 `step:1 - …`，旧计数器在夹具上给 2、在真机上给 0。新夹具（[mock/make-mock-tree.sh](experiments/verl-profiler-oom-and-critic-gpu/mock/make-mock-tree.sh)）改成真形状后，端到端跑通（三臂 rc=0、judge exit 0），**当场又抓出第二处**：V4 原来数**全文件** `attn_implementation="flash_attention_2"` 计数，夹具在 `load_valuehead_model` 块外留了一处（上游目前块外恰好 0，所以线上没暴露）⇒ 树已打好补丁也判 `flash_attention_2_left=2`。现改为读预检用的同一分类器（`patch-critic-sdpa.py analyze() → state == "patched"`），文件级计数保留作信息；负向对照 `outside_literal_not_counted` 必须 exit 0。旧夹具 `raw/mock-run/` 保留作反例，新读数在 `raw/mock-run-real-shape/README.json`。

**三、V3 前提补了静态证据。** `use_critic = need_critic(config)`，显式 `critic.enable=True` ⇒ 恒 True（`utils.py:98`）；`marked_timer("values")`（`trainer_base.py:595`）与 `("update_critic")`（`:604`）都在 `if self.use_critic` 下、**不受 `critic_warmup` 影响**；`_compute_metrics`（`:1885`）把整个 `timing_raw` 交给 `compute_timing_metrics`（`:1963`）⇒ 有 critic 时每步各一条 `timing_s/update_critic`。计数器也改成只数这个键（不再数裸词）。

**四、两个外部前提顺手关掉。** ① 组合把关原在 `3fdf2b54` 上做，而到当前 main `31cfccb4` 之间 **8 个配置 yaml 变了**（`actor/actor.yaml`、`engine/fsdp.yaml`、`engine/megatron.yaml`、`rollout/rollout.yaml` ＋ 四个 `_generated_*`），全部是删 QAT 段与改注释 ⇒ 仍重跑：**三臂组合成功、负向对照如期失败、17 个回读键与旧批次逐键相同**（hydra 1.3.8／omegaconf 2.3.1 对旧批 1.3.2／2.3.0），读数 `raw/compose-check-main-31cfccb4.json`；树用 `codeload` tarball 取，`verl/utils/model.py` sha256 仍 `a13d8595…`。② wrapper 调用的 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 在 `31cfccb4` 仍在（5368 B），其 25 个环境变量覆盖 wrapper 设置的全部键 ⇒ 没有"脚本改签名"这类静默错配。

**边界**：`104/104`、`1432 步`统计的是**仓库里已归档的日志**，不是新跑的结果；新计数器在真机 critic 日志上会给出什么，要等窗口的第一份日志才算实测；`raw/mock-run-real-shape/` 仍是管线夹具。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（开跑成本已压到分钟级、预检秒级失败、判定与计数器都已验过）；② 4 卡 `inflight=1` 仍需 ≥4 卡窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 仍未做（18:12 后不可达）；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作（OOM 路径打一行 dump 路径／示例补 `enable=True`／profiler 文档）仍待用户逐项授权，本轮**未触上游**。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `check-log-counters.py`（`--verify` 104 份日志／`--selftest` 6/6） · `raw/log-counter-check.json` · `mock/`（真形状夹具＋`run-mock.sh`） · `raw/mock-run-real-shape/` · `raw/compose-check-main-31cfccb4.json` · [handoff](experiments/handoff-m5.md)。

### M5 本轮（2026-11-05 01:0x CST）：无卡轮——critic 三臂的窗口前置做成脚本与预登记

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时；`api.github.com` 200。本轮**零 GPU 读数**。

**为什么做这个。** 上一轮把 critic 三臂写成"已备未跑"，但两个前提只活在服务器 `/tmp`：手改的 FA2→sdpa 树、`--target` 装的 msprobe／trl；而且**当时没有预登记判定**——跑完再商量"算不算通过"，等于让窗口自己给自己打分。这一轮只补这两件，不假装有卡。

**一、树内适配变成脚本（`patch-critic-sdpa.py`）。** 只在 `load_valuehead_model` 块内把两处 `attn_implementation="flash_attention_2"` 换成 `"sdpa"`，块外一个字不动；`--check` 只读、`--revert` 逆操作、**锚点数≠`--expect`(2) 即 exit 2**（上游改了形状就拒绝假装修好——否则一个"自称打好补丁"的树会烧掉整个窗口）。用**真上游文件**验证：`verl-project/verl` `3fdf2b548882` 的 `verl/utils/model.py`（sha256 `a13d8595…`，raw.githubusercontent 只读取回）——`--check` 报 `unpatched`／2 处锚点在第 **475／500** 行／函数块 467–510／块外 0 ⇒ exit 1；`apply` 精确改 **2** 行、块外 0 ⇒ exit 0；`--revert` 回到**逐字节**相同（sha256 与上游一致）；再 `--check` 又回 exit 1。自检 **18/18**（块外同字面量不被改、二次 apply sha 不变、1 锚点／3 锚点／函数不存在全 exit 2 且不写文件、树不存在 exit 3）。

**二、判定预登记（`verify-critic-pdbg.py`）。** 只读每臂的 `arm-<ARM>.json`：有效性 V1–V4＝三臂 exit 0、`steps_seen ≥ 2`、`cu`/`cv` 日志里出现 `update_critic`（v1 打的是 `timing_s/update_critic`，所以"没采到"不能是"没有 critic"）、树里 `flash_attention_2_left == 0`；不过 → **exit 1 无效跑**。门控 P1＝`au` 在 `step_*/actor_update/` 下必须有 `construct.json`／`stack.json`／`dump.json`（否则 exit 1）；P2／P3＝`cu`／`cv` dump 文件数必须为 **0**，否则 **exit 2 推翻静态结论**。`Unknown precision_debugger stage: train_batch` 条数（静态说法是与用户配置无关地每步一条，含 `au`）**只报告不门控**。自检 **10/10**。

**三、wrapper 加固 + 端到端自检。** `run-critic-pdbg.sh` 开跑前检查 7 个依赖（树、msprobe／trl 两个 `--target`、config json、模型、两份数据）+ 树补丁状态（`AUTO_PATCH=1` 自动打、`=0` 中止），失败 exit 65——秒级失败而不是几分钟后死在建模型；每臂结束写 `arm-<ARM>.json`（dump 清单带字节数、warning 计数、步数、树 sha 与 FA2 残留数、依赖在否），**失败也写**。本机用假 example 脚本把整条管线跑通：三臂 wrapper rc=0、judge exit 0（产物 `raw/mock-run/`，**管线夹具不是读数**）。

**边界**：`sdpa` 只是文本替换，critic 侧**数值等价未测**；本轮**零 GPU 读数**，`raw/mock-run/` 里的一切不得当作 critic 行为证据；真正的答案仍来自三份真日志。 **上游只读复核（同日 01:1x）**：main 已到 `31cfccb4`，但 `verl/utils/model.py` 与 `3fdf2b54` **逐字节相同**（sha256 `a13d8595…`）⇒ 换最新 main 建树也照样适用；`engine_workers.py`（903 行）的 `annotate` 仍只有 `train_batch`／`ref_compute_log_prob`／`actor_compute_log_prob`／`actor_update` 四个 role，**没有** `critic_update`／`compute_values`（只核这两个文件，非全仓重扫）。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `patch-critic-sdpa.py` · `verify-critic-pdbg.py` · `run-critic-pdbg.sh` · `raw/critic-sdpa-patch.json`／`.diff`（真上游文件） · `raw/mock-run/`（夹具） · [handoff](experiments/handoff-m5.md)。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（本轮已把开跑成本压到几分钟，预检 + 自动补丁 + 自动判定）；② 4 卡 `inflight=1` 仍需 ≥4 卡空闲窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 未做（18:12 后不可达）；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作（OOM 路径打一行 dump 路径／示例补 `enable=True`／profiler 文档）仍待用户逐项授权，本轮**未触上游**。

### M5 本轮（2026-11-05 00:0x CST）：无卡轮——接手 OOM 探针并把它做成可复跑的判定

**连通性**：`4090-1` ICMP 100% 丢包＋`:22` 超时（11-04 15:58 那个窗口已在 18:12 后关闭；上一会话 18:12 的产物就是这轮接手的存档）；昇腾网关 `10.205.0.1` 2 包全丢；`api.github.com` 200 但 `github.com:443` 超时 ⇒ 推送走 [REST 工具](tools/git-push-api/readme.md)。**未重试、未空转**。

**一、三臂 OOM 探针（上一会话采集，本轮核对）**：`attached`（rank 0、`ranks=[0]`，root DEBUG＋Capture）与 `nocap`（同配置不装 handler）**各写 1 个 6275 B pickle**（`oom_<ts>/torch_memory_oom_rank0_pid*.pickle`，key＝`allocator_settings`/`device_traces`/`external_annotations`/`segments`，1 段、trace 1×3；0 s 即完整，1/3/10/20 s 不变）；`unranked`（rank 3、`ranks=[0]`）**5 次 poll 全空**。不挂是结构性保证（`torch_memory_profile.py:80-81`）。三臂同一张 4090、分配器几乎为空（free 24.836／total 25.251 GB）。

**二、"挂上却无日志"＋一个新缺口**：attach 是 **INFO**（`torch_memory_profile.py:110`，默认 WARN 挡掉）；OOM 三条是 **ERROR**，默认经 `lastResort` 打到 stderr（`nocap` 臂实测 3 条）。而 `dump_memory_snapshot` 成功后的 `[memory_visualize] dumped: <path>`（`memory_utils.py:321`）**永远不打印**：该模块 `logger = logging.getLogger(__file__)`（`:28`，名字是文件路径，不在 `verl.*` 层级）＋ `VERL_LOGGING_LEVEL` 默认 **WARN**（`:29`）⇒ root 与 `verl` 都调 DEBUG 也不出现（`attached` 臂 4 条记录里 0 条来自 `memory_utils`）。**第三方交叉核对**：16:5x 三份真 PPO 日志里快照确实写了 23.5 MB，`[memory_visualize]`／`dumped:` **各 0 条**（`torch_memory` 的 10 处命中全是 Hydra 配置回显）⇒ 对周期性 `stop()` 一样成立。最小改面＝`_on_out_of_memory` 用掉被丢弃的返回值打一行路径、`None` 时补一行失败；不动 `getLogger(__file__)`（全仓 56 处同写法）。**未授权。**

**三、critic 三臂：已备、未跑**。`run-critic-pdbg.sh` 只换 `precision_debugger.stages`（`au` 正臂、`cu`/`cv` 被测），配方 `gae`＋`critic.enable=True`＋`critic.model.path`。本轮 `compose_check.py` 在 main `3fdf2b54`（worktree `/tmp/verl-3fdf2b5`）上组合把关：35 条 override（10 基础配方＋17 探针 EXTRA＋7 收尾参数＋1 `stages`）**三臂全过**、负向对照（`critic.fsdp.no_such_field=1`）**如期失败**；回读确认生效值 `gae`（基础配方的 `grpo` 被 EXTRA 覆盖，顺序要紧）、`critic.fsdp.model_dtype=bf16`、`critic.forward_max_token_len_per_gpu=3000`。两个窗口杀手：① `load_valuehead_model` **写死** `attn_implementation="flash_attention_2"`（`verl/utils/model.py:475`／`:500`，调用点不传，`+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 只救 actor/rollout）⇒ 服务端树上的本地补丁**重建树后必须重打**；② `critic.use_dynamic_bsz` 继承 actor 为 True ⇒ `critic.ppo_micro_batch_size_per_gpu=1` 是惰性键。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `verify-oom-archive.py`（**49 项 exit 0**、`--selftest` 7/7）· `compose_check.py`（三臂＋负向对照）· `raw/`（3 份 result JSON、`nocap.stderr`、`compose-check.json`）· [M1 证据第七节](open-source/verl/evidence/engine-refactor-torch-memory/readme.md) · [handoff](experiments/handoff-m5.md)。

**留给下一轮**：① 4 卡 `inflight=1` 仍需 ≥4 卡窗口（11-04 只有 1/4/5 空闲，且服务器现不可达）；② critic 三臂需单卡窗口＋树上的 FA2→sdpa 补丁在位；③ 本轮未重跑 GPU、**服务端原始文件未做 md5**（18:12 后不可达）；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作（OOM 路径补一行 dump 路径；示例补 `enable=True`）仍待用户逐项授权，本轮**未触上游**。

### M5（2026-11-04 16:5x–17:3x）profiler 真机两验已归档

`torch_memory` 只在 `enable=True` 时落盘（23.5 MB／39 segments）、官方示例形状一张不写、窗口跟 `global_profiler.steps`；msprobe `import` 零副作用、`actor_update` dump 14 MB、`critic_update` 零 dump 零 warning。全文与边界见 [实验](experiments/verl-torch-memory-gpu-smoke/readme.md) 与 [diary 17:29](diary/2026-11-04.md)。

### M5（2026-11-04 15:5x–16:4x）rollout-bound 负结论 + 量子端点已归档

三个负载点 GATE 全未过 ⇒ 本机 rollout-bound 区间不可达（生成余量 4.5–4.7×）；换等待来源（`max_off_policy_threshold=1`＋drop）后量子回到关键路径（gen +0.544 s、整步 +0.789 s，四条预登记全 PASS）。全文与边界见 [实验](experiments/verl-v1-poll-quantum-separate-bound/README.md) 与 [diary 16:41](diary/2026-11-04.md)。

### M1 本轮（2026-11-04 18:0x）：TM-013 新登记——SGLang 扩池，扫出第一条真失效

**一、为什么做。** TM-007 的 pyflag 已把 verl／SkyRL／OpenRLHF／rLLM 都做到「候选 ≤1 条真失效」，AGENTS 写明「**再扫应换新框架而非重扫**」。按扩池要求一次只纳入 1–2 个新对象，本轮取 **SGLang**。

**二、读数：原始 34 条里排在最前的几条都是误报。** `27aedf7f`（2026-11-04）上 386 个开关、**34 条 `silently_inert`**（18 never／13 always／3 mixed）。但前三条（`enable_cfg_parallel`／`use_decode_cuda_graph`／`use_zigzag_page_table`）全是**属性传参**（第五类误报），`mode=mixed` 本身也只是「多个消费者默认值不一致」的统计假象。⇒ **大仓库上 pyflag 的总量不能直接当缺陷清单**，必须先按「是不是 config 字段」「有没有位置参数调用」分桶。

**三、真正可核的一条：`fused_moe_router_shim(..., enable_deterministic_inference: bool = False)` 零生产者。** 这个形参只做一件事——确定性推理时让开 tensorcore 版路由内核（注释 `we keep using single kernel to avoid non-deterministic behavior`）。装函数处 `grok.py:149` 用 `functools.partial` **只钉住第一个位置参数** `moe_softcapping`，唯一调用点 `topk.py:3239` 只传 `hidden_states`／`gating_output`／`topk`／`renormalize` ⇒ 形参恒 False、tensorcore 恒被选中 ⇒ **`--enable-deterministic-inference` 对 Grok1/Grok2 的 MoE 路由没有影响**，不报错也不打日志。

**四、行为上确实分叉（不是死代码）。** 把 `router.py` 里三个函数的**真实源码**取出、桩掉 triton 与内核启动，在 CPU 上执行原文件的真实分支：照 `topk.py` 的调用走 **tensorcore**，同形状强制 `True` 走 **cudacore**。

**五、最有力的证据是「同仓四处都这么写，只有这一条走样」。** `mega_gate.py:39`、`deepseek_v2.py:581`、`gpt_oss.py:144`、`kimi_k3.py:922` 都是**直读执行配置**让路；只有 Grok 这条把判断留给没接线的形参。

**六、历史归属：加的时候就没接。** 形参由 `9b8ebb2798`（2025-10-09，标题为「move more files under srt/utils」）顺带加入，**同一提交没改 `grok.py` 也没改 `topk.py`** ⇒ 不是重构回归（与 TM-006 的 `tiled_mlp` 形状不同：那边是「曾经接过、后来丢了」）。

**七、本批两处自错（都已修）。** ① 初稿写「同一个 `router.py` 里 aten 兜底分支也是直读 `get_exec()`」——该文件 387 行里 `get_exec()` **出现 0 次**，是把 `mega_gate.py` 的写法凭印象挪过来的；改为四文件对照表，并加了一条「对照文件仍在且仍直读」的判据钉住它。② 同一段还写「`topk.py` 有 5 处直读确定性配置」——实测 **0 处**（那 3 处 `get_exec()` 全是 `moe.enable_eplb` 之类），已撤。

**边界（对外必须一起写）**：只影响 Grok1/Grok2 且 `num_local_experts > 0`；**不影响正确性**（两个内核都算 top-k）；**不能写成「确定性推理在 Grok 上坏了」**，只能写「这个开关对 Grok 的 MoE 路由没有影响」。**未验证**：真机 logits 是否逐位漂移、两个内核的数值差异（本机 Apple Silicon，triton 只有 manylinux 轮子，`uv` 已实测报错）。价值 **5/10**。

**核对产物**：`repro.py` **7/7 PASS exit 0**、`--selftest` **exit 1**；快照与 `raw.githubusercontent.com/27aedf7f` 的 `router.py`／`topk.py` **逐字节相同**；历史归属经 GitHub commits API + 逐版本 `raw` 核对。**纯静态 + CPU，未装 SGLang、未跑推理、未上 GPU、未触上游。**

**留给下一轮**：① TM-013 的两种最小修法已备好（调用点按 `get_exec()` 传值／shim 内直读，本仓库倾向后者），**对外待用户逐项授权**；② SGLang 那 34 条 `silently_inert` 里除已取证这条外**只做了抽样判型，未逐条取证**——若继续，剩下的高 reads 条目（`use_sliding_window`／`use_dsa`／`enable_linear_replayssm` 等）是下一批候选，但**判定前先按「是不是 config 字段」分桶**；③ TRL 也已扫过（**只有 6 个开关、0 条 `silently_inert`**，全仓 `use_*/enable_*` 形参默认值只有 1 处 ⇒ 这个框架不适合当 pyflag 对象，勿再扫）；④ TM-006／TM-005／TM-012 等对外候选不变，仍待用户逐项授权；⑤ TS-001 侧仍停在「等用户立项」。

入口：[证据页](open-source/sglang/evidence/grok-router-detflag.md) · [实验](experiments/sglang-grok-router-detflag/README.md) · [复现](experiments/sglang-grok-router-detflag/repro.py) · [TM-013](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1 上一轮（2026-11-04 17:0x）：TM-007——SkyRL 那 7 条「静默失效」逐条查证，真失效只剩 1 条

**一、为什么做。** TM-007 上一条记录只有「`use_liger_kernel`、`use_remove_padding` **确认为真**」这五个字，没写**在哪个 ref、用哪条事实**确认。两轮之后没人能复核。A 类扫 issue 这批已被认领完、算力链路仍不可达，正好把这条方法学补成可复跑的东西。

**二、读数：扫描稳定，结论要大改。** SkyRL `1089e18`（2026-11-04）重跑 pyflag：38 个开关、7 条 `silently_inert`，与 2026-11-02 那批**同名同量**。逐条按「可 grep 的事实」判定后：

| 判定 | 条数 | 开关 | 依据 |
|---|---|---|---|
| **DEAD** | **1** | `use_liger_kernel` | 零关键字实参；`config.py` **无该字段**；`fsdp_worker.py` 传 15 个关键字偏偏没有它 |
| **NOT_USER_FACING** | 1 | `use_remove_padding` | 分支是 `… or ulysses_sp_size > 1`，唯一调用点又在 `sp > 1` 守卫内 ⇒ 形参不产生观测差异 |
| INDIRECT_PRODUCER | 2 | `enable_sample_support_replay`／`enable_pd` | 配置字段读出后传给同名形参（`worker.py:1122→1166`／`setup.py:94→133,155`） |
| PUBLIC_HELPER_API | 1 | `disable_print` | `pprint_registry` 的返回字符串开关；skyrl-gym 独立发包 |
| FALSE_POSITIVE | 1 | `use_combine` | Megatron patch 内部形参；子类走**位置参数** `super().__init__(config, use_combine=False)` |
| DEPRECATED_BY_DESIGN | 1 | `use_signals` | docstring 标 `(Deprecated)`，默认走多进程 |

⇒ **原记「两条确认为真」要改成一条**：`use_remove_padding` 确实从没被传过，但分支有第二个或项，`mode=always` 的真身在那里，谈不上「设了没用」。

**三、影响面必须写清。** `use_liger_kernel` **配不了也没人配**（config 无字段、依赖里也没有 liger）⇒ 当前没有任何用户的训练受影响。它是「看着能配的开关」，不是 verl 那种「字段在、消费者没了」的回归。对外价值 **3/10**，不建议单独提。

**四、历史归属只说到能做到的一步。** `model_wrapper.py` **全部 18 次提交**（2026-02-16 → 2026-10-30，GitHub contents API 逐版本取文件）里计数恒为 3（docstring／默认值／分支），同版本 `fsdp_worker.py` 恒为 0。**没有**遍历 `fsdp_worker.py` 自己的历史，也不是全仓 bisect。

**五、本批两处自错（都已修）。** ① v1 用 `"use_liger_kernel=" in text` 判「有生产者」，命中的正是 `model_wrapper.py:148` 的**形参默认值声明行**——而这恰好是 pyflag readme 自己写的易错点第 4 条，读数当场反了；现改为「跳过 `def` 行 + 要求前面有 `调用名(`」。② 差点把 `skyrl-agent/examples/run_verl/*.sh` 里 **verl 的 Hydra 键** `actor_rollout_ref.model.use_remove_padding=True` 当成 SkyRL 形参的生产者。

**核对产物**：`verify_findings.py` 7 条判定 **0 不一致** exit 0、`--selftest` **8 条注入全部翻转**；扫描与判定 JSON 归档在同一目录。**纯静态，未跑训练、未上 GPU/NPU、未触上游。**

**留给下一轮**：① `use_liger_kernel` 若要对外，最小动作只有「删掉形参与分支」或「接进 `fsdp_worker.py`」，**待用户逐项授权**；② 其余 6 条不建议对外；③ 方法沉淀已写进 [pyflag readme 第五类易错点](tools/pyflag/readme.md)；三个框架都跑到「候选 ≤ 1 条真失效」，**再扫应换新框架而非重扫**；④ TS-001 侧仍停在「等用户立项」。

入口：[逐条取证](experiments/skyrl-pyflag-triage/readme.md) · [判定脚本](experiments/skyrl-pyflag-triage/verify_findings.py) · [pyflag](tools/pyflag/readme.md) · [TM-007](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1 上一轮（2026-11-04 15:5x–16:1x）：TS-001 第十八批——正文引用的数，归档里有没有托底

**一、为什么做。** TS-001 的「下一步」明写 M1 这条线已到边际：纯归档重算、判据一致性、带宽／可读性诊断都做过。剩下的不是再接一层诊断，而是问一个从没问过的问题——**每批都在跑 `--verify`，但它只证明"已归档的字段没变"，证明不了"正文写的那个数在归档里存在"**。

**二、读数：一处真缺口。** 挑第四十批订正那组 `R`（14.7／34.0／42.4／44.9／56.3／63.4）逐个对到字段：`legacy_group_arm_check.py` 的 `CVS = (0.2, 0.6, 1.2)` **止于 1.2**，而正文与 readme 写的是 `cv 0.2→1.5` ⇒ **44.9%（cv 1.5、4 批）、56.3% 与最大 63.4%（cv 1.5、16 批）三个数在任何归档文件里都不存在**，只有台账。

**三、读数本身没错，是托底没了。** 两条独立实现（`summary.json` 的 L2 扫描、审计脚本自带的 `fixed_group_arm`）重算 1.5 端点一致：4 批 **0.4486**、16 批 **0.5629**、16 批最大 **0.6336**。

**四、改动三处，读数不变。** ① `CVS` 补到 1.5；② 新增 `claims_vs_sweep`，把正文五个引用值逐值对账、不一致 **exit 1**（负向对照：0.449→0.430 报 `AUDIT FAIL`）；③ 新增 `arm_crosscheck`，两条组臂实现逐种子比对 36 组、最大差 <1e-12，把第四十批"修正后逐位相同"从一句断言变成检查项。

**五、不沿用的断言。** 脚本事后自证"正文六个数逐项对上旧路径"，但归档里 cv 1.2 旧路径是 0.5433／0.6626，与正文的 1.5 端点、"上限 61%"对不上 ⇒ 本轮**只把能复现的端点做成字段**，不宣称旧路径值可逐项复核。

**核对产物**：`legacy_group_arm_check.py`（`--verify` exit 0；归档新增 73 行、既有字段 0 变 0 删）、`test_group_split_tradeoff.py` **56 passed**、`test_real_lengths_policy.py` **29 passed**。纯 CPU 重算，未跑 GPU、未触上游。

**留给下一轮**：① **"对台账"这一类到此为止**——`band_floor`／`sign_regime` 的引用值自带 `n_*` 字段，`threshold-calibration.json`／`sampling-reach.json`／`prediction-floor.json` 各自独立成档且 `--verify` 覆盖，扫下来只这一处；② TS-001 仍停在「纯归档重算已到边际，下一轮无新输入应转 P-1 立项讨论或语义面新模型」；③ TM-006 对外候选（配置键删除／`freeze_vision_tower` 接线／torch_memory 示例缺 `enable`／容器 dict 下传）**均待用户逐项授权**。

入口：[审计脚本](experiments/rollout-group-split-tradeoff/legacy_group_arm_check.py) · [归档](experiments/rollout-group-split-tradeoff/legacy-group-arm-check.json) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-04.md)。

### M5 上一轮（2026-11-04 14:5x–15:1x CST）：无卡轮——把"只能真机定级"拆成静态定死 + 一条 import 冒烟

**连通性（本轮只探一次，未重试）**：`4090-1`（`219.223.198.62`）ICMP 2 包全丢 + `:22` TCP 超时（公网直连，非 VPN）；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154/155/156:22` 全超时。外网正常 ⇒ 本机无恙，恢复只在用户侧；这是同日第四次两条链路同时不可达。

**一、接手的是 M1 登记的 profiler 交接，但先把"必须 NPU + msprobe"这一半压到最小。** 白名单 `_STAGE_TO_ROLE`（`precision_debugger_profile.py:26`）6 个 stage 逐个找触发者：main `7454326` 上只有 actor 侧 3 个有（`engine_workers.py:751/758/768`）；`compute_values`、`critic_update`、**`compute_rm_score`** 一个都没有。对照 #6067 父提交 `6eeb5711`：前两个在 FSDP（`:1677`／`:1697`）与 Megatron（`:1249`／`:1266`）两套 worker 上**都挂过**装饰器 ⇒ **重构回归**；`compute_rm_score` 在父提交里**也一处没有** ⇒ **从来没接过**（自集成引入 2026-04-03 以来四个 ref 均 0 处；M1 上一轮只点了前两个）。

**二、触发路径唯一，旁路逐条排除。** 只有 `@DistProfiler.annotate(role=<字面量>)` 能进 `start(stage=…)`；`start_profile(stage=…)` 全仓 **0 处**，2 处 `start_profile(role="train", …)`（`ray_trainer.py:1132`、`trainer_base.py:1430`）的 `role` 只是标签——`start()` 首参是 `stage`、函数体不读 `kwargs["role"]`。`msprobe` 全仓只 2 处（`import_utils.py:75` 探测、`precision_debugger_profile.py:208` 懒 import），`PrecisionDebugger` 只在 `start()` 的 `:215` 构造，dump 目录也只在 `:211-212` 才 `makedirs` ⇒ **不调用 `start()` 磁盘上不会有任何东西**。

**三、机制比"删了个装饰器"复杂：不是补三行能修的。** 新引擎里 critic 训练走 `TrainingWorker.train_batch`（`engine_workers.py:377`），装饰器在但 role 是 `train_batch`——不在白名单，判 unknown 后丢弃，**且与用户配置无关地每步刷一条 warning**；critic 取值前向走 `infer_batch`（`:433`）无装饰器。`TrainingWorker` 同时服务 `Role.Critic`（`trainer_base.py:813`）与 SFT（`sft_trainer_ray.py:138`）⇒ 补装饰器要先解决"这个 worker 现在扮演谁"，属设计选择。

**四、文档面同时对上。** Ascend en 文档 `## Supported stages`（`:78`，bullet `:81-87`）与 6 处配置注释（`profiler/config.py:205`、`actor.yaml:316`、`critic.yaml:225`、`ppo_trainer.yaml:451`、`ref.yaml:164`、`rollout.yaml:460`）把 6 个 stage 全列为可选；文档 `:240-241` 用含 critic 两 stage 的 L0 组合报过 3–4× 开销，当前 main 上那组已不可复现。`strict=True` 也救不了——三个名字都在白名单里。**订正一处引用**：`docs/` 在 `7454326` 上 `find -name "*_zh*"` 命中 0 条，M1 引的 `precision_debugger_zh.md` 无法核对，本文只引 en 文档。

**产物**：[证据](open-source/verl/evidence/precision-debugger-stage-producers/readme.md) · [实验](experiments/verl-precision-debugger-stages/readme.md) · `scan_stage_producers.py`（AST + 逐行正则兜底、`--preset main|pre6067`、`--selftest` 4/4、两棵树判定 exit 0）· 两棵树 JSON。

**真机需求已降级**：不再需要"带 critic 的 PPO 小步训练"，只需"import msprobe + 正常跑一步 + 不配置也不启动采集，确认 `save_path` 无 `step_*/`"，且**优先级低于所有已排队算力实验**。**边界**：纯静态，未跑 verl、未上 GPU/NPU；唯一残留假设是 msprobe 的 import 期副作用。

**本轮自己抓出的四个脚本坑**：① `SkipManager.annotate(role="rollout")` 被误算成触发者（匹配面太宽，和上一轮 `\btiled_mlp\b` 同型）；② 把 `self` 当成 `start()` 首参，判定差点反读；③ 本机 Python 3.9 解析不了 9 个文件，其中含父提交的 `fsdp_workers.py`——**差点漏掉两个 critic 装饰器**，改正则兜底后补回 6 条且与 `grep` 逐行一致；④ 兜底分支 `src` 跨文件残留。

**留给下一轮**：① rollout-bound A/B、4 卡 `inflight=1`、profiler 冒烟都等窗口；② **CONTEXT 已 175KB，超过 150KB 软告警**，增量主要在 M1 批次段，建议由写入者按"已归档"精减（本轮只删了自己一条已被 diary/handoff 覆盖的 3 行小节）；③ 对外动作（文档标注或加载期校验）待用户逐项授权。

### M1（2026-11-04 13:0x–13:4x）：同一次重构的第二个信号族——接口名与文档引用

**一、为什么做。** 上一批自己写下的"#6067 另外 71 个改动文件的语义变化未扫"太大，收窄到可做的一格：**被整份删除的那 10 个模块里，除了配置字段，还有哪些"名字"没了但外面还在用**。这与配置消费点互补——配置的消费者是引擎代码，接口名的消费者是**文档、下游脚本、装饰器**。

**二、读数。** 50 个公开方法/函数名 → 32 仍有同名定义；13 条"无定义但非历史文档仍引用"（去重 **3 个名字**：`update_policy`、`update_critic`、`compute_values`）；5 个无定义无引用；`docs/blog/` 引用 0 条。

| 名字 | main 里现在是什么 | 文档在哪还教 |
|---|---|---|
| `update_policy` | `update_actor`（`engine_workers.py:656`） | `perf_tuning.rst:135`、`dpo_extension.rst:18` |
| `update_critic` | **无同名 RPC**；trainer 调 `critic_wg.train_mini_batch`（`ray_trainer.py:1359`） | `ray_trainer.rst:194`、`perf_tuning.rst:135` |
| `compute_values` | **无同名 RPC**；trainer 调 `critic_wg.infer_batch`（`ray_trainer.py:1221`） | `ray_trainer.rst:162`、`precision_debugger_zh.md:48/85` |

**三、必须先说的限定。** 这三个名字在**代码侧没有任何调用点**（`verl/**` 非测试、`tests/`、`examples/` 全 0）⇒ 没有静默失效、不影响训练结果。而且 `dpo_extension.rst` 与 `ray_trainer.rst` **#6067 根本没碰**，属存量文档没跟上重构，不是这次造成的回归。⇒ **对外不主张提 issue**，价值评级 4/10。

**四、真正的发现：装饰器 role 是字符串常量，任何符号级扫描都看不见。** 旧 `fsdp_workers.py:1677/1697` 给 critic 两个 RPC 挂了 `@DistProfiler.annotate(role="compute_values")`／`(role="critic_update")`；重构后 `verl/**` 这两串**各 0 处**（engine 侧只剩 `actor_update`／`actor_compute_log_prob`／`ref_compute_log_prob`），但前端白名单 `_STAGE_TO_ROLE`（`precision_debugger_profile.py:26-33`）**仍接受**这两个 stage，`strict: False` 下**连 warn 都没有** ⇒ 用户配了"不报错也不采"。**口径只能写到"前端接受、后端无触发点"**：`_MODEL_ATTRS_BY_ROLE` 里 critic 的解析路径还在，实际采不采得到要 NPU + msprobe。

**五、两条存量文档漂移（不属 #6067）。** ① `perf_tuning.rst:140-141` 说 values 前向的 token 上限由 `critic.forward_micro_batch_size_per_gpu` 控制，而 main 里真正的 token 上限是 `critic.forward_max_token_len_per_gpu`（`critic.yaml:49`），文档全文 0 次提及；② 同文档把归一化代码指向 `engine_workers.py`（#6067 改的链接），但新目标文件里没有归一化逻辑（`ppo_mini_batch_size` 只有 `:511` 一处赋值）。

**六、本批自己三处错（都由复核抓出）。** ① 判定脚本三条文档正则**凭记忆写**（中英文混写成"和"、路径拼成 `docs/advanced/../`、行首符号写成 `-` 而原文是 `*`），全 MISMATCH；② `critic-infer-token-cap` 一条**方向写反**（我先写"文档提 `forward_max_token_len_per_gpu` 但 main 没有"，复核 `critic.yaml:49` 发现 main 里**有的正是它**）；③ `_STAGE_TO_ROLE` 初读成"删了 critic 域"，实际是"表里留着、删掉的是装饰器"——两者对外分量完全不同。

**核对产物**：`scan_api_surface.py`（50 名，可复现）、`verify_findings.py` 7 条 **0 不一致** exit 0、`--selftest` 注入同名 def 后四条判定**全部翻转**；另两条 `doc_drift` 的手工注入也翻 False（未固化进脚本）。verl main `74543265`（2026-11-04 fetch）、父提交 `044bbba2^`、Python 3.12.13。**纯静态，未跑 verl、未上 GPU/NPU、未触上游**。

**留给下一轮**：① ~~两次体检都只覆盖"被整份删除的文件"，第三次要换信号族~~ **已于同日第四段换到启动配置键**（见下条）；仍未扫的是 71 个被改文件里**函数体被改写**那一类，若再做须先说明它不是重复；② profiler 缺口已登记 [M5 交接](experiments/handoff-m5.md)，最小对照是"`stages: [actor_update]` 有 dump vs `stages: [critic_update]` 是否同样有"；③ TM-006 对外两条（删配置项／补 `freeze_vision_tower` 接线）仍待用户逐项授权；④ TS-001 侧仍停在「等用户立项」。

入口：[接口面体检](open-source/verl/evidence/engine-refactor-api-surface/readme.md) · [实验 readme](experiments/verl-6067-api-surface-scan/readme.md) · [扫描脚本](experiments/verl-6067-api-surface-scan/scan_api_surface.py) · [判定脚本](experiments/verl-6067-api-surface-scan/verify_findings.py) · [M5 交接](experiments/handoff-m5.md) · [TM-006](queue_tech-miner.md) · [tracker](open-source/verl/tracker.md) · [diary](diary/2026-11-04.md)。
### M1（2026-11-04 14:0x–14:4x）：同一次重构的第三个信号族——被删掉的启动配置键，还有谁在传
**一、为什么做。** 上一次自己在「留给下一轮」写的是"71 个被改文件语义变化仍未扫，第三次要换信号族，否则只是重复"。挑了一格最像"改动而非删除"的：`verl/trainer/config/ppo_trainer.yaml` 在 #6067 里 **94 → 93 个叶子键**，唯一被删的是 `trainer.use_legacy_worker_impl`（旧 worker / 新引擎的开关，旧实现已整份删了）。**删了配置键，命令行里还在写的脚本不会自己消失。**

**二、读数。** 全仓（15 种后缀）4 处引用，按"会不会失败"分三档：

| 档 | 处数 | 位置 |
|---|---|---|
| `INVOKE / COMPOSE_FAIL`（**启动即报错**） | **1** | `tests/special_e2e/run_fully_async_policy_genrm.sh:177`（无 `+` 前缀，2026-04-27 新增、晚于 #6067） |
| `INVOKE / SILENT_OK`（不报错，也**不生效**） | 2 | `run_fully_async_policy_opd.sh:240`、`examples/grpo_trainer/run_qwen3_30b_veomni.sh:102`（都带 `+`） |
| `DOC_TABLE_CELL` | 1 | Ascend `parameter_and_metrics.md:368`，默认值仍写 `auto` |
| 代码侧读取点 | **0** | 两条 `+` 传进去的字段无人消费 |

**三、先说限定。** ① **"不报错"恰好是最安全的一档**——`+k` 只是让 Hydra 收下新字段，`verl/**` 对这个键 0 读取，写成"两个脚本坏了"是夸大；② **会失败的那条不在任何 CI 命令里**（`genrm` 脚本没被 `.github/workflows` 引用，`opd` 被引用但带 `+`）⇒ 是"谁手动跑谁撞墙"，不是"CI 现在红的"；③ `docs/workers/engine_workers.rst:221` 已标 `(removed)`，被扫描器按豁免档正确排除。

**四、关键证据是真实复现，不是推断。** `hydra_repro.py` 用最小树跑 hydra **1.3.8**：键在＋unprefixed＝OK、键缺＋unprefixed＝**FAIL**（`ConfigCompositionException: Could not override`）、键缺＋`+`＝OK、存在的键（负向对照）＝OK。顺带否掉一条反驳：同一条 `defaults` 链上 `cfg.trainer.<新键> = …` 抛 `ConfigAttributeError`，不是 open dict。

**五、为什么前两批扫不到（这是本批最该记住的一条）。** 配置消费点问"字段还有人读吗"——字段已从 yaml 消失就不进候选；接口面查 `def`——Hydra 键不是 `def`。⇒ **"删掉的东西"要按它的形状找消费者**：删文件看引用、删字段看读取点、**删配置键看命令行与参数表**。

**六、顺带一条未计入的候选。** `scripts/megatron_merge_lora.py` 在 #6067 里新写的两行调 `engine.bridge.save_weights` / `save_hf_weights`；这两个方法由第三方包（`mbridge`／`megatron.bridge`）提供，本机未安装 ⇒ **静态判不了**。旁证两个方向都有（引擎自身只调 `export_*`；但 `megatron_checkpoint_manager.py:993-1015` 仍在 `inspect.signature` 里探测 `save_weights`），故只记"需装包/真机才能判"，不进对外材料。

**七、本批自己三处工具错误（都已修）。** ① `git grep -l -- '*.sh' …` 把 glob 当**待匹配文本**，整轮读出"0 处引用"；② `entry_context` 把引号里的 `"${VERL_ROOT}/examples/data_preprocess/geo3k.py"` 当启动命令，三个命中全解析错；③ 自检的"假注入"——`write_text(x.replace(...))` 在 `replace` 没命中时照样写出，"注入了"是假的，现在先比替换前后字符串。

**核对产物**：`scan_shell_refs.py`／`verify_findings.py`（4 判定 + 3 自证 **0 不一致**）／`--selftest`（3 扰动全部被捕捉）／`hydra_repro.py` **4/4 PASS**。verl main `74543265`（2026-11-04 fetch）、Python 3.12.13、hydra 1.3.8（一次性 venv `/tmp/hydra-probe-venv`）。**未装 verl、未跑训练、未上 GPU/NPU、未触上游**。

**留给下一轮**：① 启动配置键这一族已查完（1 键、4 处），对外三条最小动作**均待用户逐项授权**；② `megatron_merge_lora.py` 的 bridge 两行需装 `mbridge` 或真机，不在 M1 范围；③ 71 个被改文件里**函数体被改写**的那一类仍未扫，若再做须先说明它不是对前三段的重复；④ TM-006 前两段的两条对外候选不变；⑤ TS-001 侧仍停在「等用户立项」（M1 侧已无纯归档重算可做）。

入口：[启动配置键体检](open-source/verl/evidence/engine-refactor-shell-config/readme.md) · [实验](experiments/verl-6067-shell-config-scan/readme.md) · [扫描](experiments/verl-6067-shell-config-scan/scan_shell_refs.py) · [判定](experiments/verl-6067-shell-config-scan/verify_findings.py) · [Hydra 复现](experiments/verl-6067-shell-config-scan/hydra_repro.py) · [TM-006](queue_tech-miner.md) · [tracker](open-source/verl/tracker.md) · [diary](diary/2026-11-04.md)。

### M1（2026-11-04 15:0x–15:4x）：同一次重构的第五个信号族——名字都在，但值的形状变了

**一、为什么做。** 前四段（配置消费点／接口名与文档／启动配置键）都靠"名字还在不在"判断。这一段换到最像"没改动"的一格：`torch_memory` 的显存快照。旧办法是 trainer 每步喊一声 `actor_rollout_wg.dump_memory_snapshot(...)`（#3099，2025-08-20）；#6198（2026-04-29，收 #6181）把这声喊整段删掉，PR 说明写"capabilities are fully integrated in DistProfiler"，并给出配方 `tool=torch_memory` + `save_path` + `steps` + **`actor.profiler.enable=True`**。**两边的方法名、类名、配置类名一模一样**，所以任何文本扫描都会答"没问题"。

**二、读数（三层，都由脚本判定，不靠人眼）。**

| 层 | 读数 | 结论 |
|---|---|---|
| 入口面（AST，父提交 vs main 两个 worker 类） | 父提交有 `dump_memory_snapshot`；main 的两个 worker 类 **0 个** snapshot 方法 | 旧入口整体消失 |
| 新链路（真实配置 → `DistProfiler.start/stop`） | **写 1 张**：`outputs/profile/step1/`，tag `torch_memory` | **通**，不是"删了就没了" |
| 官方示例（照抄 overrides 渲染） | `actor`／`rollout`／`critic` 的 `profiler.enable` **全为 False** | **一步都不写**（沉默失败） |
| 容器 dict 下传（只给 `config` 不给 `tool_config`） | `AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'` | **启动期硬失败**（仅 `tool=torch_memory`/`npu`） |

**三、必须先说的限定。** ① `enable=False` 这条**不是新回归**：示例最后修改 2026-08-31、文档 `docs/perf/verl_profiler_system.md` 最后修改 2026-09-08，都晚于 #6198，只是没补那一项 ⇒ 属**文档/示例欠债**，对外只能写"照示例操作拿不到快照，配方缺 `enable=True`"，不能写"示例坏了"。② 容器 dict 那条是**推断不是日志**：CPU 上证明了"配置喂进去会崩"，但没跑 `run_ppo`。③ 旧 worker 的显式调用现在会 `AttributeError`——这正是 #6198 处理过的事（#6067 之后 trainer 里那条调用已被 `hasattr` 挡成静默跳过），本段只作时间线证据。

**四、本批自己两处错（都由复核抓出）。** ① 第一版探针按 `cfg=compose(ppo_trainer)` 原样跑，没带 `global_profiler.tool=torch_memory`，于是后端是 `_NoOpProfiler`，"不写"成了同义反复——补上 overrides 后才看到真读数；② 一开始把"容器 dict 下传"的调用点写成 `main_ppo.py:157-158` 与 `engine_workers.py:174-176`，逐个 `sed` 核行号后改为 `:187`（v1 分支）与 `:645-647`，并核清 `:155-166` 的 `TrainingWorker` 是**显式取内层**、实测正常——原先那句会把一个能工作的调用点也列进问题清单。

**五、为什么前四段扫不到。** 前三段问"字段还有没有人读""`def` 还在不在""yaml 键有没有被删"。这一段的名字全在，坏的是**配置从"直接读 yaml"换成"先转 dataclass 再往回传"之后，两级 `ProfilerConfig` 只有一级被正确填充**（角色级 → 工具级 `tool_config.<name>`）。⇒ **这一类只能把真实配置喂进真实调用链，然后 spy 住唯一写盘点。**

**核对产物**：`repro_torch_memory.py` **10/10 PASS**（含两条 selftest 扰动）、`probe_official_example.py` **3/3 PASS**，读数落在同目录两份 JSON。verl main `74543265`（2026-11-04 fetch）、父提交 `6eeb5711`、Python 3.12.13 + ray 2.59.0／omegaconf 2.3.1／hydra 1.3.8／torch 2.15.0。**纯 CPU，未跑训练、未上 GPU/NPU、未触上游**。

**留给下一轮**：① 对外三条（示例补 `enable=True`／文档配方同步／修容器 dict 下传）**均待用户逐项授权**；② 快照是否真落盘、OOM 观察者是否挂上分配器，需 GPU/NPU（M5 范围，最小目标 `tool=torch_memory` + `actor.profiler.enable=True` + `steps=[1]`，看 `save_path/step1/*.pickle`）；③ 前四段的对外候选不变，仍待用户逐项授权；④ TS-001 侧仍停在「等用户立项」。

入口：[证据](open-source/verl/evidence/engine-refactor-torch-memory/readme.md) · [实验](experiments/verl-6067-torch-memory-path/readme.md) · [复现](experiments/verl-6067-torch-memory-path/repro_torch_memory.py) · [示例探针](experiments/verl-6067-torch-memory-path/probe_official_example.py) · [TM-006](queue_tech-miner.md) · [tracker](open-source/verl/tracker.md) · [diary](diary/2026-11-04.md)。

### M5（2026-11-04 13:5x–14:1x CST）：无卡轮继续，把"只能窗口验"的部分提前验掉，抓出三个坑

**连通性（本轮实测，只探一次）**：`4090-1`（`219.223.198.62:22`）`nc` TCP 超时（12:5x 轮 ICMP 也 100% 丢包）——**公网直连、不经 VPN**，不能按"VPN 掉线"解释；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154/155/156:22` 全超时。外网正常 ⇒ 本机无恙，恢复只在用户侧（10-28 有过约 28 h 电信段断裂先例）。**未重试、未空转**。

**一、真日志核对关掉了"窗口期才能验"的最大一项。** 新增 `crosscheck-real-logs.py`，拿上一轮 separate_async 的**真实控制台日志**（同 mode／同机／同脚本家族，只差负载）跑三个**互相独立**的解析器：**40 项断言全过**（4 文件 × 6 指标，三路均值差 <1e-9），每个日志 **104 处 `np.float64(...)` repr 都被正确取值**，每份 8 条 step 行且 `logged == parsed`（无 EngineCore 污染），配置抽取在真 dump 上可读。⇒ 此前记的"np.float64 形态／控制台尾巴需窗口核对"已可关闭；仍留给窗口的只有新配置值（1／1024）的打印形态与 64 条/步下的稳态缺口。

**二、三个会在窗口里炸的坑（全部已修＋负向对照）。** ① **最后一步会被缺口判据判死**：旧版把 `step − Σ相位` 的检查套到步 8（=最后一步），而归档日志里 sync／DAPO／colocate 的**最后一步都比相位和多 3.9–7.6 s**（吞了收尾工作），separate_async 只是恰好干净（+0.03 s）⇒ 现改为**最后一步不进门、只打 INFO**；负向对照：同一份"最后一步 +6 s"夹具旧版 **rc=1（4 条失败）**、新版 **rc=0**。② **缺相位键直接抛 `KeyError` 栈**而不是点名失败 ⇒ 改为先做存在性检查；同时把写死的 0.1 s 阈值改成 `max(0.1 s, 中位步长×2%)`（真日志稳态缺口 0.027–0.035 s，但与步长同变，固定阈值会把固定开销当坏账）。③ **冒烟命令默认撞 CUDA 217**：协议里冒烟是裸跑 `run-sep-bound.sh`，而脚本默认 `NCCL_SHM_DISABLE=0`（与"本模式必须 1"矛盾，A/B 链脚本自己导出了 1、冒烟路径没有）⇒ 默认改为 1。另核清一条不能混用的事实：sync／DAPO／colocate 有 `timing_s/ref`（≈2.4 s）而本模式没有，**"5 个相位套所有模式"本身就会造出 ~2.4 s 的假缺口**，判据只能按模式写。

**三、自检扩到 14 项。** 新增"最后一步 +6 s → 仍 exit 0"与"缺相位键 → exit 1 且无 traceback"两条夹具；`python3 selftest-bound.py` 14 PASS、`crosscheck-real-logs.py` 40 PASS、`py_compile` 与 `bash -n` 全过。

**边界**：本目录仍**零 GPU 读数**，不得引用为实测；翻不翻区间只有 GATE 能答。绝对步时受 `NCCL_SHM_DISABLE=1` 影响，不可与其他批次比。

入口：[实验准备与判定规则](experiments/verl-v1-poll-quantum-separate-bound/README.md) · [真日志互核](experiments/verl-v1-poll-quantum-separate-bound/crosscheck-real-logs.py) · [离线自检](experiments/verl-v1-poll-quantum-separate-bound/selftest-bound.py) · [TM-008](queue_tech-miner.md) · [真机交接](experiments/handoff-m5.md)。对外动作全部未做。

### M5（2026-11-04 12:5x–13:0x CST）：两条算力通路都不可达，回退到准备物料并入库

**做了什么**：把上一会话中断时留下的未跟踪目录 `experiments/verl-v1-poll-quantum-separate-bound/` 补完并入库——§13.54 最后一个空格（separate_async 在 rollout **成为瓶颈**、训练步真的在 `sample()` 上等的区间未测）。**三处实改**：① `total_storage_size` 512→1536（512 恰好等于本配置一步 64 条，会构造性复现 `Storage capacity exceeded`）；② `verify-bound.py` 判定拆三类（机械无效 exit 1／区间没翻 **exit 2**／全过 exit 0），效应量改 INFO；③ 新增 `selftest-bound.py`（10 项断言）。**自检抓到两个错**：夹具用双引号键（真实是单引号）；**0.1 臂原门槛「稳态 gen 均值 < 1.9 s」是错的**——rollout-bound 区间真实等待本身可超 2 s，会把好数据判死，已改为只用 2.0 臂「过半数稳态步 ≥1.9 s」作 GATE。细节见 [diary](diary/2026-11-04.md) 与本轮 README。


### M1（2026-11-04 11:3x–12:0x）：把 `tiled_mlp` 的单条回归查成一类——#6067 配置消费点体检

**一、为什么做。** 上一批自己在 TM-006 第②条写下"下一步最值项：查同期重构还丢了哪些配置"，且已尝试过一次自动化比对（`config_sweep.py`，候选几乎全是假阳性、脚本已删）。本批换做法：**先只做文本级定位，再把判据逐条写死**。

**二、读数。** `044bbba2^` 的两个 worker 文件里 `config.<role>.<key>` 读取路径去重 **102 个**；在当前 main 的 `verl/workers/engine/**` + `verl/trainer/**` 查叶子名，报 **30 条**"字段还在、引擎侧查不到读取"。逐条判定：

| 判定 | 条数 | 代表 |
|---|---|---|
| **真失效** | **3** | `tiled_mlp`、**`freeze_vision_tower`**、`use_prefix_grouper` |
| 改名／换父配置后生效 | 26 | `actor.fsdp_config → actor.engine`（dataclass 整体换位）、`log_prob_* → ppo_*`（engine_workers 改名）、`rollout.free_cache_engine` 由 rollout 层读 |
| 搬了家 | 1 | MTP 关闭时 `num_nextn_predict_layers` 清零 → `HFModelConfig.__post_init__:235-243` |

⇒ **是一"类"回归，但不是"一片"**：写"几十个配置全失效"会是错的，这也是本批最需要守住的口径。

**三、新发现：`actor.freeze_vision_tower`。** 旧实现两处（actor `:545-553`、critic `:1554-1562`）调用同文件的 `get_vl_model_vision_tower()` 冻结视觉塔并置 `use_orig_params=True`；该辅助函数随文件删除，`git grep vision_tower HEAD -- "*.py"` 全仓只剩 `config/actor.py:161` 一行声明。而 `actor.yaml:51-52` 的注释与 `docs/ascend_tutorial/.../parameter_and_metrics.md:45` 的参数表都还写着它。特性来源 PR [#3178](https://github.com/verl-project/verl/pull/3178)（2025-09，issue #2526）。

**四、一条不是失效、但语义变了（不列入对外材料）。** 旧 `fsdp_workers.py:429-432` 是 `if mtp.enable: raise NotImplementedError("MTP is not supported in FSDP")`，新代码只保留清零、丢掉报错 ⇒ FSDP 上开 MTP 不再报错。不影响默认配置，且是否有意需维护者判断，故只记录、不主张。

**五、本批两处错（都由自检抓出）。** ① `lost` 判据的读取搜索范围包含 `verl/workers/config/`，字段**声明行**被算成读取，3 条真失效全部报"对不上"；② 修掉①后写 `--selftest`，注入 `apply(use_tiled_mlp=True)` 仍报"没有读取"——`\btiled_mlp\b` 匹配不上 `use_tiled_mlp`（`_` 是单词字符）。改子串匹配后自检通过、正式判定 7 条 **0 不一致**。⇒ **自检失败时暴露的是工具，不是被测对象。**

**核对产物**：`scan_consumers.py`（102 路径／30 候选）、`verify_findings.py`（7 条判定 exit 0）、`--selftest` 通过；verl main `1a9e711e`、父提交 `044bbba2^`、Python 3.12.13。**纯静态，未跑 verl、未上 GPU、未触上游**。

**留给下一轮**：① TM-006 对外可选三条（删配置项＋文档／补 `freeze_vision_tower` 接线／加"字段有、消费者无"的加载期告警，与 TM-007 的 pyflag 同源），**均待用户逐项授权**；② `freeze_vision_tower` 的显存对照需 VL 模型真机（M5 范围）；③ #6067 另外 71 个改动文件的**语义变化**未扫——本次只覆盖配置消费点，MTP 那条是读 diff 才发现；④ TS-001 侧仍停在「等用户立项」。

入口：[类回归体检](open-source/verl/evidence/engine-refactor-config-consumers/readme.md) · [实验 readme](experiments/verl-6067-config-consumer-scan/readme.md) · [扫描脚本](experiments/verl-6067-config-consumer-scan/scan_consumers.py) · [判定脚本](experiments/verl-6067-config-consumer-scan/verify_findings.py) · [TM-006](queue_tech-miner.md) · [tracker](open-source/verl/tracker.md) · [diary](diary/2026-11-04.md)。

### M1（2026-11-04 10:5x–11:2x）：掘金渠道轮转——rLLM 三条 A 类全被认领，改做 #945／PR #946 复核

**一、为什么换轨。** TS-001 第十五到第十七批连续记录「已到边际，下一轮若无新输入应停在这里等用户立项」。按 AGENTS 的渠道要求（技术掘金不能只剩扫 issue），本轮先做 A 类的快速扫描，再看是否转向 B/C 类深度工作。

**二、A 类扫描结论：这一批没有可捡的活。** rLLM 开放 issue 里符合"无人认领 + 可本机验证 + 影响正确性"的三条，**全部已有对应 PR**：#916→#917、#945→#946、#929→#930/#948，其中两个 PR 于 2026-11-02 提交、维护者到 2026-11-04 未回应。⇒ **不重复认领**，这条线索到此为止，登记进 TM-012。

**三、改做独立复核（选 #945／#946）。** 在 main `4e828c81`（上游最后提交 2026-10-09）的真实 `compute_score` 上：

| 答案 | 题目数字 | main 得分 |
|---|---|---|
| `1.5 * 2` | 1, 5, 2 | **1.0**（小数点把 1、5 粘成一个） |
| `.5 * 8` | 5, 8 | **1.0** |
| `3 ** 2 * 5 - 7` | 3, 2, 5, 7 | **1.0** |
| `7 // 2` | 7, 2 | **1.0** |
| `(3*2)+(5-7)`（合法对照） | 3, 2, 5, 7 | 1.0 |

补丁评估：白名单去 `.` ＋ 显式禁 `**`／`//`。在 **22639** 条于 main 上得 1.0 的合法答案上逐条比较，**补丁后失去满分 0 条**（含 `10 / 4 * 2` 这类中间值带小数、结果整数的写法）。**负向对照**把补丁改成"拒绝一切"后同一搜索报 **159** 条 ⇒ 搜索不是空转。

**四、必须收窄的一处。** issue 说指数链"不返回"，实测 `44**19**2` 瞬时、`44**19**5` **2.1 秒**；`44**19**35` 约 10^45 位、**跑不完，本仓库不把"跑不完"当测量结果**。⇒ 可写"一条采样答案能拖住奖励 worker"，不可写"必然长期挂死"。

**五、本轮自己两处假通过（都已修，判据已写进文档）**：① 第一版回归搜索把 22639 条全报成"零敏感文本"——是**统计口径**漏了 `**`，判定式本身没错（用 2 万条随机字符串暴力对照确认）；② 第二版把搜索范围收窄成"补丁明面上会拒绝的文本"，于是负向对照报 0 条——**一个连"拒绝一切"都查不出来的搜索不能证明没有误伤**。改成对全部合法答案比较后才报出 159 条。⇒ **负向对照必须在"被测逻辑明显错误"的强条件下仍能被触发**；把范围收窄到"只比较预期类别"等于把对照一起关掉。

**六、第二件：TM-006 的口径订正——`tiled_mlp` 是引擎重构回归，不是"从没接上"，而且丢的是三个角色。** 核 verl 侧候选时顺手查了 `tiled_mlp` 的历史：PR [#4649](https://github.com/verl-project/verl/pull/4649)（2026-01-04 合入）**当年真的把接线做进 `verl/workers/fsdp_workers.py`**（取值 `:913-915`、转发 `:935-936`／`:503-504`／`:1481-1482`、FSDP1 拒绝 `:374`）；该文件被 PR [#6067](https://github.com/verl-project/verl/pull/6067)（2026-04-20，BREAKING "deprecate workers, migrate to engines"）**整体删除**，接线没迁到新引擎。main `74543265` 复核：全仓 **704** 个 `.py` 里提到 `tiled_mlp` 的只有 **6 个**（定义／配置／测试），FSDP 引擎唯一调用点 `transformer_impl.py:322` 传的五个关键字里没有它；其余五个引擎计数全 0。⇒ 2026-01-04～04-20 之间该选项**有效**，回归窗口从重构那天起，**对外表述必须改口**（原证据文档与补丁说明都已加更新块）。**另两点新发现**：① **FSDP1 守卫也一起丢了**，而新引擎**先建模型（:322）再按 strategy 分支（:421／:448）**、`dp_actor.yaml:26` 默认 `strategy: fsdp` ⇒ 只补接线不补守卫，默认配置下会先打 TiledMLP 再走 FSDP1；② 同一次重构可能还丢了别的配置，**本批只查了这一个**（试着写自动比对脚本，候选几乎全是改名／改层级的假阳性，已放弃自动化、留作人工逐条核）。**同批订正一处计数**：删除前 `tiled_mlp` 覆盖 **actor／ref／critic 三个角色**（取值 `:913-915`／`:972-976`／`:1454-1456`，转发 `:935-936`／`:989-990`／`:1481-1482`），守卫在 `:374`（actor）与 `:1459`（critic）；新引擎三条路径都汇到 `FSDPEngine._build_module()`，故三个角色**一起**失效——先前记的"两处转发"是按调用点数、读起来像只有 actor 与 critic。

**核对产物**：`repro.py` exit 0（`--json` 同样）、负向对照 159 条；未装 rllm（两个导入用最小桩，`compute_score` 照跑）、Python 3.12.13。**未跑真实训练、未跑 rLLM 测试套件、未触上游**。

**留给下一轮**：① **TM-012 与 TM-006 的对外动作逐项待用户授权**，未授权前不建 fork 分支、不评论、不开 PR；② 这处漏洞对训练的实际影响（模型是否真会写 `1.5*2`）**无数据**，要答需要训练侧的 rollout 统计，属新模型不是补一条实验；③ 若要继续 A 类，得换仓库（rLLM 这一批已被认领完），按队列"扩池"要求一次只纳入 1–2 个新对象并先写短评估；④ **下一步最值的一件事**：拿 `044bbba2` 那次删除的 diff 扫一遍，看同年四月重构还丢了哪些配置接线——这能把一条回归变成一类回归；⑤ TS-001 侧仍停在「等用户立项」。

入口：[复核](open-source/rllm/countdown-reward-review-945.md) · [复现](experiments/rllm-countdown-reward-review/repro.py) · [run-info](experiments/rllm-countdown-reward-review/run-info.json) · [TM-012](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1（2026-11-04 10:2x–10:5x）：收尾——批号消歧、队列去重、注入方向复核

**三件事，都不改任何读数。**

**一、批号消歧。** 同一份 `notebook/rollout-performance-research.md` 里「第十七批」被三个专项各用过一次：**2026-11-02** M1 的 §13.40（微批与气泡）、**2026-10-30** M5 的 §14 G8（自建环境 GPU smoke）、**2026-11-04** M1 的 §13.30（截断即长度预测）。已在 §13.30 第十七批节抬头加归属说明：引用必须带专项名。**这是本仓库的第三、四次批号撞号**（第十五／十六批也各消歧过一次）⇒ 批号是专项内计数，跨专项引用一律写「某专项第 N 批」。

**二、队列去重。** 两个 queue 文件各有两个「更新时间」段（新段插在标题前、旧段留在标题后）。旧段已并入新段作「此前：…」，各减 3 行，内容无丢失。

**三、注入方向复核（结论=确认，不是推翻）。** 把 `floor_predict` 的 `>=` 改 `>` 后 15 条测试全绿。追下去发现**这一行本来就不是恒等式的边界**——恰好等于 `c` 的样本两种写法都返回 `c`，`max` 照样精确。真正的边界是**记录标准**：改成 `x >= c - 1` 后 `--verify` 报 6 处差异 exit 1、测试 1 条红。与并行会话同日做的复核结论一致。

**本轮两次操作失误（都已恢复）**：① 注入后 `cp` 恢复因 `&&` 链中断而**没执行**，注入版一度留在工作区，靠下一次 `git diff` 才发现；② `git checkout -- <path>` 把并行会话刚写进同一文件的改动一起回退，重做两次。⇒ **共享 checkout 上做注入，恢复后用 `git diff` 复核、别信 `cp` 的退出码；改文件前先确认它没有并行改动。**

**核对产物**：`test_prediction_floor` **15 passed**；`test_group_split_tradeoff`＋`test_real_lengths_policy` **85 passed**（365 秒）；`prediction_floor.py --verify` **0 差异**；归档只差 `seconds` 一个运行时长字段，已回退不提交。只读归档、未跑 GPU、未触上游。

**留给下一轮**：① 语义面四项缺口（逐条路径预测误差／分桶／迁移／抢占／KV 搬运）**都需要新模型或新数据**——逐条权重需要撞顶样本的真实长度（M5 真机采集范围），分桶需先改造排程器（TM-011 前置不变）；② 本专项在 M1 侧**已无纯归档重算可做**，下一轮若无新输入应停在这里等用户立项，不要重复加诊断层；③ `diary/2026-08-14.md` 是未跟踪的历史汇报文件（2026-11-03 生成、内容与今日工作无关），**未纳入提交**，处置待用户决定。

入口：[§13.30 第十七批补充（含批号归属）](notebook/rollout-performance-research.md) · [实验 readme 复核补记](experiments/rollout-group-split-tradeoff/readme.md) · [TS-001](queue_tech-secretary.md) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1（2026-11-04 04:45）：门槛校准——`sign_regime` 假阳为 0，`band_floor` 的严松不对称

**一、为什么做。** 第十／十一批把 `sign_regime` 四档与 `band_floor` 三档写进归档时，三处文档都只写"是判读门槛不是显著性检验"，并在「留给下一轮」记了"未做假阳率模拟校准"。**那句性质说明没有信息量**：判据的"性质"和"代价位"是两件事。

**二、`sign_regime` 在已知真相上很干净**（每格 4000 列 × 20 种子）：真同向的列判出 `mixed` **0.000**、真 `thin` 的列判出 `mixed` **0.000** ⇒ 第十批判词（"不能把 `consistent_but_thin` 当符号不稳"）在门槛层面站得住。唯一假阴 **1.1%**（纯掷硬币被判 `stable`），而这**就是二项分布本身**：门槛只要求表态种子 ≥80% 同号，20 硬币双侧 ≥16 同号 = **1.18%**。⇒ **压下去只能加种子，不能挪门槛**；且门槛是断崖式的（`min_same_active` 0.8→0.55 时该假阴跳到 **82.2%**）。

**三、`band_floor` 两边严重不对称。** 真相设成"每个种子 `R` 恒 0"、扫非零种子比例 p：p=0 → `no_reading` 1.000；p=0.05（期望 1.5 个非零）→ **0.242**；p=0.5 → **0.578 判成 `usable`**。原因是 `no_reading` 要求**全部** 30 个种子的 max R 为 0。

**四、真实 16 格的报告纪律（本批唯一改口的地方）。**

| 负载 | 2 批 | 4 批 | 8 批 | 16 批 |
|---|---|---|---|---|
| 1.5B cap256 | `no_reading` 30/30 零 | `usable` **26/30 非零**（2.83pp） | `usable` 30/30 | `usable` 30/30 |
| 0.5B cap256 | `no_reading` 30/30 零 | `degenerate` 22/30 零 | `usable` 30/30（2.15pp） | `usable` 30/30 |

⇒ **引用 `band_floor` 时必须同时给 `n_r_identically_zero`**。`usable` 只说明"过半种子带宽非零"，不说明基线稳。

**五、本批自己犯的错（已改正）。** readme 初稿写"1.1% 说明门槛 0.8 几乎不起作用"，并把概率归成"20 个全同号"≈1.9e-6——**`stable` 只要求 ≥80% 同号，不是 100%**。现算二项分布才对上实测。同族于第五批补遗，但错法不同：**把区间判据误当成点判据**。

**核对产物**：新增 [calibrate_thresholds.py](experiments/rollout-c-class-ledger/calibrate_thresholds.py)（`--json`／`--verify` 逐字段可复算）与 [test_threshold_calibration.py](experiments/rollout-c-class-ledger/test_threshold_calibration.py) **13 passed**；两次注入被测红后恢复（`min_same_active` 0.8→0.55 ⇒ 3 条红；`band_floor` `== 0.0`→`<= 0.05` ⇒ 3 条红），注入前打印被改行号与原文；`test_cross_script_consistency` **35 passed**、`test_group_split_tradeoff` + `test_real_lengths_policy` **81 passed**；`calibrate_thresholds.py --verify` OK。只读归档与合成数据、未跑 GPU、未触上游。

**留给下一轮**：① 20 个种子的分辨力上限已量出（1.1%），**要压下去必须加种子**——`real-lengths` 30 种子、`bucketing-probe` 20 种子，加种子要重跑全链路（约 10 分钟档），是否值得由引用场景决定；② `no_reading` 仍只回答"这一格问不出来"，**"换一种抽样能否问得出来"未答**（需另立实验，第十一批已记）；③ 语义面剩余缺口仍是长度预测误差／分桶／迁移／抢占／KV 搬运，P-1 跨系统需用户立项。

入口：[校准脚本](experiments/rollout-c-class-ledger/calibrate_thresholds.py) · [校准归档](experiments/rollout-c-class-ledger/threshold-calibration.json) · [C 类账 readme 第十批](experiments/rollout-c-class-ledger/readme.md) · [paired-diff](tools/paired-diff/readme.md) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-04.md)。

### M5（2026-11-03 03:0x–04:3x CST）：第五十六批 §13.52 + 第五十七批（stale 逐出补测）

已归档：完整叙述见 [diary](diary/) 与对应专项文档；此处不再保留过程记录。

### M5 补记（2026-11-03 04:1x–05:3x CST，第二会话）：第五十六批独立复核 + main 树运行时冒烟 + `wait` 臂

已归档：完整叙述见 [diary](diary/) 与对应专项文档；此处不再保留过程记录。

### M5（2026-11-03 06:5x–07:4x CST）：第五十八批 §13.54——separate_async 跑通，轮询量子不进关键路径

已归档：完整叙述见 [diary](diary/) 与对应专项文档；此处不再保留过程记录。
