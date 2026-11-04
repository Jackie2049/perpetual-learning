更新时间：2026-11-05（**M5 01:0x 无卡轮：双链路仍不可达，把 critic 三臂的窗口前置做成脚本＋预登记判定——FA2→sdpa 适配幂等可回退（真上游文件 `3fdf2b54` 验证 475／500 两处、改 2 行、revert 逐字节还原、自检 18/18），判定先写死（validity 不过 exit 1、`cu`/`cv` 冒 dump exit 2、自检 10/10），wrapper 预检 7 项依赖并输出 `arm-<ARM>.json`；零 GPU 读数**。此前**M5 00:0x 无卡轮**：4090-1 自 11-04 18:12 后不可达——前一日 15:58 恢复的窗口已关闭：ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 2 包全丢；`github.com:443`（HTTPS）仍超时，**但 SSH（`git@github.com:22`）本轮直推成功**（`be22e6c..cbaf725`）⇒ 下次先试 SSH、失败再走 REST 工具。本轮接手上一会话 18:12 中止留下的 OOM 探针并核对入库：**OOM 自动快照真落盘**——rank 0 挂观察者写 6275 B pickle（4 key／1 段／trace 1×3，0 s 即完整），不挂观察者的 rank 3 **5 次 poll 全空**；**"挂上却无日志"查清**（attach 是 INFO 被默认 WARN 挡、OOM 是 ERROR 默认可见），并新发现**dump 成功后 `[memory_visualize] dumped: <path>` 永远不打印**（`memory_utils.py:28-29` 用 `getLogger(__file__)`＋默认 WARN，root/`verl` 调到 DEBUG 也救不回来；16:5x 三份真日志各 0 条交叉印证）⇒ 最小改面两行（`_on_out_of_memory` 用掉被丢弃的返回值），对外仍待授权。critic 侧三臂**已备未跑**，本轮加 `compose_check.py` 在 main `3fdf2b54` 上把关：**三臂组合全过＋负向对照如期失败**，并记下两个窗口杀手（`load_valuehead_model` 写死 FA2；`critic.use_dynamic_bsz` 继承 True 使 micro batch 键惰性）。此前 M1 TM-007 SkyRL 7 条 pyflag 取证（真失效只剩 `use_liger_kernel`）；M5 16:5x–17:3x profiler 真机验证（torch_memory 落盘／示例沉默失败／窗口跟配置步）；M5 15:5x rollout-bound 判成不可达＋量子回到关键路径。 规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。此前 M1 18:0x **M1 18:0x TM-013 新登记**：pyflag 扩池扫 SGLang `27aedf7f`（386 开关／34 条 `silently_inert`），原始读数靠不住（前三条都是属性传参误报、`mode=mixed` 只是默认值不一致），逐条取证后抓到 **`fused_moe_router_shim` 的 `enable_deterministic_inference` 形参零生产者**——注释写着「确定性时让开 tensorcore 内核」，但 `grok.py:149` 的 partial 只钉第一个位置参数、`topk.py:3239` 只传四个无关关键字 ⇒ `--enable-deterministic-inference` 对 Grok 的 MoE 路由没有影响；CPU 上用**真实源码**跑出行为分叉（tensorcore vs cudacore），同仓四处同类路径（`mega_gate`／`deepseek_v2`／`gpt_oss`／`kimi_k3`）都直读执行配置让路、只有这条走样；形参由 `9b8ebb2798`（2025-10-09）顺带加入、同提交未改调用点 ⇒ **加的时候就没接**，不是重构回归。价值 5/10，对外待授权。TRL 同批扫过：6 个开关、0 条失效，不适合当 pyflag 对象。此前 M5 15:5x–16:4x 有卡轮（TM-008 rollout-bound 判成不可达 + stale-wait 量子回到关键路径）、M1 17:0x TM-007 SkyRL 7 条降到 1 条、M1 15:5x–16:1x TS-001 第十八批（正文引用值托底）、M5 14:5x–15:1x／13:5x–14:1x 无卡轮（profiler stage 白名单／真日志互核）。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。此前 **M5 16:5x–17:3x 有卡轮**：profiler 的"只能窗口验"全部关闭——`torch_memory` 加 `enable=True` **落盘 23.5 MB / 39 segments**、官方示例形状**一张不写**、`steps=[2]` 落 `step2/`、OOM 观察者 `_oom_observer_attached=True` 但日志无痕、容器 dict 的 `AttributeError` 逐字复现但 **v1 真跑不触发**；msprobe `import` 零副作用、`actor_update` dump 14 MB 而 `critic_update` **零 dump 零 warning**。4 卡 `inflight=1` 仍无窗口。此前 **M5 15:5x–16:4x 有卡轮**：4090-1 当日恢复，7 跑全部 exit 0——**TM-008 的 rollout-bound 区间判成不可达**（三个负载点 GATE 全未过：1 rollout : 2 train 下训练 0.204–0.248 ms/token vs 生成 0.055–0.069 ms/token，余量 4.5–4.7×，加负载只拉大差距）；换等待来源（`max_off_policy_threshold=1`＋drop）后**量子回到关键路径**：gen +0.544 s、整步 +0.789 s（+7.9%），四条预登记全 PASS。M1 15:5x–16:1x 技术秘书第十八批：**正文引用的 `R` 端点在归档里不存在**——`legacy_group_arm_check.py` 的 cv 扫描止于 1.2，正文写的 44.9%／56.3%／63.4%（cv 1.5）只有台账可查；补到 1.5 并新增 `claims_vs_sweep`（正文五个引用值硬对账、不一致 exit 1，改坏 0.449→0.430 已验）与 `arm_crosscheck`（两条组臂实现 36 组逐种子比对 <1e-12）。**读数不变**，56＋29 测试通过。⇒ "对台账"这一类到此为止。此前 M1 TM-006 第五段 `torch_memory` 快照链路：链路通、但官方示例未开 `*.profiler.enable=True`（一步都不写）、不带 `dataclass_type` 的 `omega_conf_to_dataclass` 启动期 `AttributeError`，两个脚本 10/10、3/3 PASS。此前 M5 14:5x–15:1x／13:5x–14:1x 无卡轮：profiler 6 个 stage 里 3 个无触发者（2 回归＋1 从未接线）、rollout-bound 工具真日志互核 40/40。此前 M1：接口面第三段、启动配置键第四段、配置消费点（真失效 3 条）、rLLM 复核（22639 条 0 回归）。 M1 17:0x 技术掘金 **TM-007：SkyRL 那 7 条「静默失效」逐条查证**——在当前 HEAD `1089e18` 重跑 pyflag（38 个开关、7 条 `silently_inert`，与 2026-11-02 同名同量），逐条绑定源码事实判定：**真失效只剩 1 条 `use_liger_kernel`**（全仓零关键字实参、`config.py` 无该字段、docstring 却写 `Defaults to False`；`model_wrapper.py` 全部 18 次提交里计数恒为 3、同版本 `fsdp_worker.py` 恒为 0 ⇒ 这 18 个 ref 上从没接过线），**原记「`use_remove_padding` 确认为真」降级为 NOT_USER_FACING**（分支是 `use_remove_padding or ulysses_sp_size > 1`，唯一调用点又在 `sp > 1` 守卫内 ⇒ 形参不产生观测差异）；其余 5 条：属性传参盲区 2、内部形参位置参数 1、公开 helper 1、弃用开关 1。**对外价值 3/10、不建议对外**（配不了也没人配 ⇒ 无用户受影响）。本批两处自错已修：子串匹配把**形参默认值声明行**算成生产者（读数当场反了）、差点把 verl 的 Hydra 键当成 SkyRL 形参的生产者。 M1 15:5x–16:1x 技术秘书第十八批（正文引用的 cv 1.5 端点归档里没有 ⇒ 补 `CVS` 到 1.5 + `claims_vs_sweep` 硬对账）；M5 14:5x–15:1x 无卡轮（profiler 白名单 6 个 stage 里 3 个没有触发者）；M5 13:5x–14:1x 无卡轮（rollout-bound 三路互核 40/40）；M1 #6067 五个信号族体检）。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。M1 20:0x–00:0x **TM-014**：：把 TM-013 那批 SGLang 的 34 条 `silently_inert` **全量归档**——先按消费者位置分桶（内核 2／配置 dataclass 3／`__init__` 存 self 8／源码分支 21），再逐条绑到可 grep 的事实，**真缺陷只剩 2 条**（TM-013 那条 + 新的 `use_logits_buffer`）。新的一条：`logits_processor._get_logits(..., use_logits_buffer: bool = True)` 的唯一调用点传 `num_chunks == 1`，而 `num_chunks` 按 `SGLANG_LOGPROB_CHUNK_SIZE`（默认 2048）切 ⇒ **输入 logprob 一长，默认值 `True` 就不可达**，注释里那句「形状不匹配才跳过」也从未被读过；注入两条分叉在 CPU 上用**真实源码**跑出来（`True`→复用 graph buffer，`False`→`logits.float()`），**不影响正确性**（函数体内还有形状/dtype 比对），价值 3/10，对外待授权。`--selftest` 3/3。另两条方法学结论：生产者可以藏在**回调实参**里（`use_zigzag_page_table` 由 `cp/zigzag.py:388` 传 `True`），以及**形参有生产者≠语义对**（pyflag 结构上看不见）

### M1（2026-11-04 20:0x–2026-11-05 00:0x）：TM-014——SGLang 的 34 条静默失效全量归档，真缺陷只剩 2 条

**一、为什么做。** TM-013 留给下一轮的第②条写着「剩下 33 条只做了抽样判型，若继续，判定前先按『是不是 config 字段』分桶」。这一轮把它做成脚本：`build_triage.py` 按**消费者所在位置**分桶，`verify_findings.py` 把 34 条逐条绑到可 grep 的事实，再对唯一的新候选做 CPU 行为分叉。⇒ **不再靠 reads 排序用人眼扫**——SGLang 386 个开关里 reads 最高的那条（236）恰好是属性传参误报。

**二、分桶读数与判定账。** 内核启动器 2 / 配置 dataclass 3 / `__init__` 存 self 8 / 源码分支 21。判定：`KERNEL_CONSTEXPR` **10**、`INTERNAL_OR_DEFAULT` **12**、`ATTR_PASSING` **6**、`CONFIG_DEFAULT` **3**、`INTERNAL_PRODUCER` **1**（`use_dsa`，子类 `memory_pool.py:5245` 传 `use_dsa=True`）、`TM013_DUP` **1**、**`TOPIC_ESCAPE_FAILS_OPEN` 1**。合计 34，脚本核对不漏不重。

**三、新发现的这一条：`use_logits_buffer` 的默认值不可达。** `logits_processor.py:1011` 的 `_get_logits(..., use_logits_buffer: bool = True)` 只控制「算完 lm_head 要不要直接收进 graph 复用的 logits buffer」。唯一调用点 `logprob_processor.py:657` 传 `use_logits_buffer=num_chunks == 1`，`num_chunks` 按 `SGLANG_LOGPROB_CHUNK_SIZE`（**2048**）切 ⇒ 输入 logprob 超过 2048 行即恒 `False`。而注释（`:1411-1412`）声明的是「**形状**不匹配才跳过」，调用点用的是「**块数**」——语义被换掉了。两个方向不对称：单块且形状不匹配时仍传 `True`（靠函数体内比对兜住），多块且形状恰好匹配时一律 `False`（少了一次本该发生的复用）。

**四、行为上确实分叉。** 取 `_copy_logits_to_buffer` 的**真实源码**、桩掉 `torch` 四个成员，在 CPU 上执行两条分支：`True` → 返回复用的 `graph_buffer<-logits`；`False` → `logits.float` 新分配。⇒ 不是两条等价的路。

**五、口径（对外必须一起写）。** **不影响正确性**；**未上真机**，两条分支的显存峰值差异与省下的 cast **一个数都没有**；调用点写 `num_chunks == 1` 是否**故意保守**（注释里提过多块复用同一 buffer 会 alias）**本仓库无定论** ⇒ 不写补丁、只留证，对外只能问。

**六、两条方法学结论（已写进 pyflag readme）。** ① **生产者可以藏在回调实参里**：`trtllm_mha_backend` 把内部函数 `_trtllm_context_attn` 交给 `cp_strategy.run_attention(...)`，`cp/zigzag.py:388` 再 `use_zigzag_page_table=True` 回调；② **形参有生产者 ≠ 语义对**——`use_logits_buffer` 天天有人赋值，pyflag 根本不会报它，坏的是"传进去的东西不是那个意思"。

**七、本批三处自错（都已修）。** ① 分桶脚本初稿把「`__init__` 里存 self」整类说成「调用点走 `*_ENABLED` 常量」——`use_dsa` 同一形状却有内部生产者，已改成「线索不是结论」并单独取证；② 负向对照第一版只把一支改成 `return logits_buffer`，产物名不同但两支都不再 `.float()`，判据仍说「分叉」——判据改为「产物名相同**或** False 分支不再退回 `.float()`」；③ 误判注入最初选源码桶的两条（`disable_recompute`／`use_safe_gate`），改标签对账不报——换成内核桶的 `enable_bmm` 才真正点亮「分桶与判定不一致」。

**核对产物**：`build_triage.py`（分桶 + 名字读取点）、`verify_findings.py`（34 条对账 + 读取点分类 + 真实源码双分支 + `--selftest` **3/3**）；另人工注入 `use_zigzag_page_table` 判成源码桶 ⇒ 被拦，34 条全改错 ⇒ 报 5 条。快照 `27aedf7f`（40 MB tarball，经 gh-proxy），**纯静态 + CPU 桩执行**：未装 SGLang、未跑推理、未上 GPU、未触上游。

**留给下一轮**：① TM-013／TM-014 两条对外（issue／PR）**均待用户逐项授权**；② SGLang 这一池已扫完（34 条全归档），再扫应换新框架，且按 AGENTS「一次只纳入 1–2 个新对象」——候选池里 TensorRT-LLM／LMDeploy／llama.cpp 都没碰过；③ 若继续 SGLang，唯一有新增信息的方向是 `use_logits_buffer` 的**真机显存差**（需 GPU，属 M5 范围）；④ TS-001 侧仍停在「等用户立项」。

入口：[证据页](open-source/sglang/evidence/logits-buffer-escape-hatch.md) · [实验与分类账](experiments/sglang-pyflag-triage/README.md) · [分桶](experiments/sglang-pyflag-triage/build_triage.py) · [判定](experiments/sglang-pyflag-triage/verify_findings.py) · [TM-014](queue_tech-miner.md) · [diary](diary/2026-11-05.md)。


## 当前现场

> **推送已恢复（2026-11-04 16:2x）**：此前积压的 5 个本地提交（`b850df424`／`ab204152f`／`db7a7aeed`／`3860b0b6b`／`9816bf5f8`）与本轮提交已全部推上远端 main。`git push` 仍连不上 `github.com:443`（75 s 超时），改用 [REST 推送工具](tools/git-push-api/readme.md)（`api.github.com` 正常）；远端新提交 `01c8bf13` 的 SHA 与旧本地不同（消息与树相同），已 fetch 并 `reset --hard` 对齐，本地树 `618e2a82` 与推送内容一致、`--verify` 通过。⇒ **下次网络同样断开时优先走 `tools/git-push-api`，不要反复等直连。**

### M5 本轮（2026-11-05 01:0x CST）：无卡轮——critic 三臂的窗口前置做成脚本与预登记

**连通性（只探一次，未重试）**：`4090-1` ICMP 100% 丢包＋`:22` TCP 超时；昇腾网关 `10.205.0.1` 2 包全丢、`192.168.13.154:22` 超时；`api.github.com` 200。本轮**零 GPU 读数**。

**为什么做这个。** 上一轮把 critic 三臂写成"已备未跑"，但两个前提只活在服务器 `/tmp`：手改的 FA2→sdpa 树、`--target` 装的 msprobe／trl；而且**当时没有预登记判定**——跑完再商量"算不算通过"，等于让窗口自己给自己打分。这一轮只补这两件，不假装有卡。

**一、树内适配变成脚本（`patch-critic-sdpa.py`）。** 只在 `load_valuehead_model` 块内把两处 `attn_implementation="flash_attention_2"` 换成 `"sdpa"`，块外一个字不动；`--check` 只读、`--revert` 逆操作、**锚点数≠`--expect`(2) 即 exit 2**（上游改了形状就拒绝假装修好——否则一个"自称打好补丁"的树会烧掉整个窗口）。用**真上游文件**验证：`verl-project/verl` `3fdf2b548882` 的 `verl/utils/model.py`（sha256 `a13d8595…`，raw.githubusercontent 只读取回）——`--check` 报 `unpatched`／2 处锚点在第 **475／500** 行／函数块 467–510／块外 0 ⇒ exit 1；`apply` 精确改 **2** 行、块外 0 ⇒ exit 0；`--revert` 回到**逐字节**相同（sha256 与上游一致）；再 `--check` 又回 exit 1。自检 **18/18**（块外同字面量不被改、二次 apply sha 不变、1 锚点／3 锚点／函数不存在全 exit 2 且不写文件、树不存在 exit 3）。

**二、判定预登记（`verify-critic-pdbg.py`）。** 只读每臂的 `arm-<ARM>.json`：有效性 V1–V4＝三臂 exit 0、`steps_seen ≥ 2`、`cu`/`cv` 日志里出现 `update_critic`（v1 打的是 `timing_s/update_critic`，所以"没采到"不能是"没有 critic"）、树里 `flash_attention_2_left == 0`；不过 → **exit 1 无效跑**。门控 P1＝`au` 在 `step_*/actor_update/` 下必须有 `construct.json`／`stack.json`／`dump.json`（否则 exit 1）；P2／P3＝`cu`／`cv` dump 文件数必须为 **0**，否则 **exit 2 推翻静态结论**。`Unknown precision_debugger stage: train_batch` 条数（静态说法是与用户配置无关地每步一条，含 `au`）**只报告不门控**。自检 **10/10**。

**三、wrapper 加固 + 端到端自检。** `run-critic-pdbg.sh` 开跑前检查 7 个依赖（树、msprobe／trl 两个 `--target`、config json、模型、两份数据）+ 树补丁状态（`AUTO_PATCH=1` 自动打、`=0` 中止），失败 exit 65——秒级失败而不是几分钟后死在建模型；每臂结束写 `arm-<ARM>.json`（dump 清单带字节数、warning 计数、步数、树 sha 与 FA2 残留数、依赖在否），**失败也写**。本机用假 example 脚本把整条管线跑通：三臂 wrapper rc=0、judge exit 0（产物 `raw/mock-run/`，**管线夹具不是读数**）。

**边界**：`sdpa` 只是文本替换，critic 侧**数值等价未测**；本轮**零 GPU 读数**，`raw/mock-run/` 里的一切不得当作 critic 行为证据；真正的答案仍来自三份真日志。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `patch-critic-sdpa.py` · `verify-critic-pdbg.py` · `run-critic-pdbg.sh` · `raw/critic-sdpa-patch.json`／`.diff`（真上游文件） · `raw/mock-run/`（夹具） · [handoff](experiments/handoff-m5.md)。

**留给下一轮**：① critic 三臂仍需**单卡窗口**（本轮已把开跑成本压到几分钟，预检 + 自动补丁 + 自动判定）；② 4 卡 `inflight=1` 仍需 ≥4 卡空闲窗口；③ 服务端 `/tmp/m5-oom` 原始文件 md5 未做（18:12 后不可达）；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作（OOM 路径打一行 dump 路径／示例补 `enable=True`／profiler 文档）仍待用户逐项授权，本轮**未触上游**。

### M5 本轮（2026-11-05 00:0x CST）：无卡轮——接手 OOM 探针并把它做成可复跑的判定

**连通性**：`4090-1` ICMP 100% 丢包＋`:22` 超时（11-04 15:58 那个窗口已在 18:12 后关闭；上一会话 18:12 的产物就是这轮接手的存档）；昇腾网关 `10.205.0.1` 2 包全丢；`api.github.com` 200 但 `github.com:443` 超时 ⇒ 推送走 [REST 工具](tools/git-push-api/readme.md)。**未重试、未空转**。

**一、三臂 OOM 探针（上一会话采集，本轮核对）**：`attached`（rank 0、`ranks=[0]`，root DEBUG＋Capture）与 `nocap`（同配置不装 handler）**各写 1 个 6275 B pickle**（`oom_<ts>/torch_memory_oom_rank0_pid*.pickle`，key＝`allocator_settings`/`device_traces`/`external_annotations`/`segments`，1 段、trace 1×3；0 s 即完整，1/3/10/20 s 不变）；`unranked`（rank 3、`ranks=[0]`）**5 次 poll 全空**。不挂是结构性保证（`torch_memory_profile.py:80-81`）。三臂同一张 4090、分配器几乎为空（free 24.836／total 25.251 GB）。

**二、"挂上却无日志"＋一个新缺口**：attach 是 **INFO**（`torch_memory_profile.py:110`，默认 WARN 挡掉）；OOM 三条是 **ERROR**，默认经 `lastResort` 打到 stderr（`nocap` 臂实测 3 条）。而 `dump_memory_snapshot` 成功后的 `[memory_visualize] dumped: <path>`（`memory_utils.py:321`）**永远不打印**：该模块 `logger = logging.getLogger(__file__)`（`:28`，名字是文件路径，不在 `verl.*` 层级）＋ `VERL_LOGGING_LEVEL` 默认 **WARN**（`:29`）⇒ root 与 `verl` 都调 DEBUG 也不出现（`attached` 臂 4 条记录里 0 条来自 `memory_utils`）。**第三方交叉核对**：16:5x 三份真 PPO 日志里快照确实写了 23.5 MB，`[memory_visualize]`／`dumped:` **各 0 条**（`torch_memory` 的 10 处命中全是 Hydra 配置回显）⇒ 对周期性 `stop()` 一样成立。最小改面＝`_on_out_of_memory` 用掉被丢弃的返回值打一行路径、`None` 时补一行失败；不动 `getLogger(__file__)`（全仓 56 处同写法）。**未授权。**

**三、critic 三臂：已备、未跑**。`run-critic-pdbg.sh` 只换 `precision_debugger.stages`（`au` 正臂、`cu`/`cv` 被测），配方 `gae`＋`critic.enable=True`＋`critic.model.path`。本轮 `compose_check.py` 在 main `3fdf2b54`（worktree `/tmp/verl-3fdf2b5`）上组合把关：35 条 override（10 基础配方＋17 探针 EXTRA＋7 收尾参数＋1 `stages`）**三臂全过**、负向对照（`critic.fsdp.no_such_field=1`）**如期失败**；回读确认生效值 `gae`（基础配方的 `grpo` 被 EXTRA 覆盖，顺序要紧）、`critic.fsdp.model_dtype=bf16`、`critic.forward_max_token_len_per_gpu=3000`。两个窗口杀手：① `load_valuehead_model` **写死** `attn_implementation="flash_attention_2"`（`verl/utils/model.py:475`／`:500`，调用点不传，`+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 只救 actor/rollout）⇒ 服务端树上的本地补丁**重建树后必须重打**；② `critic.use_dynamic_bsz` 继承 actor 为 True ⇒ `critic.ppo_micro_batch_size_per_gpu=1` 是惰性键。

**产物**：[实验 README](experiments/verl-profiler-oom-and-critic-gpu/readme.md) · `verify-oom-archive.py`（**49 项 exit 0**、`--selftest` 7/7）· `compose_check.py`（三臂＋负向对照）· `raw/`（3 份 result JSON、`nocap.stderr`、`compose-check.json`）· [M1 证据第七节](open-source/verl/evidence/engine-refactor-torch-memory/readme.md) · [handoff](experiments/handoff-m5.md)。

**留给下一轮**：① 4 卡 `inflight=1` 仍需 ≥4 卡窗口（11-04 只有 1/4/5 空闲，且服务器现不可达）；② critic 三臂需单卡窗口＋树上的 FA2→sdpa 补丁在位；③ 本轮未重跑 GPU、**服务端原始文件未做 md5**（18:12 后不可达）；④ 昇腾仍未探、无 NPU 事项；⑤ 对外动作（OOM 路径补一行 dump 路径；示例补 `enable=True`）仍待用户逐项授权，本轮**未触上游**。

### M5 本轮（2026-11-04 16:5x–17:3x CST）：profiler 两项真机验证全关闭（torch_memory 落盘 / msprobe 静默无操作）

**连通性**：`4090-1` ping 0% 丢包、`:22` 可连（15:58 起的窗口延续）；昇腾本轮未重探（无 NPU 事项）。**GPU 现场**：0 为 liyaowei EngineCore（20.8 G）、2/3/6/7 为其 4 卡 Qwen3-VL-32B ⇒ 只有 **1/4/5** 空闲，**4 卡 `inflight=1` 仍无窗口**；本轮只用 GPU 1，`/home` 100% 满、全写 `/tmp`。

**一、torch_memory 五个训练臂**（main `3fdf2b54` 树 + 2 步 PPO，0.5B/gsm8k）：`tool=torch_memory` + `steps=[1]` + `actor.profiler.enable=True` → `save_path/step1/torch_memory_rank0_pid*.pickle` **23.5 MB、39 segments、device_traces 1×100,000**（撞 `trace_alloc_max_entries`）；**只去掉 `enable`**（官方示例形状，`steps=[1]` 保留）→ exit 0 但 **`save_path` 目录都不建**；`steps=[2]` → 落 `step2/`、**无 `step1/`**。⇒ M1 的"链路本身通"与"示例沉默失败"两条都在真机复现，且窗口跟随 `global_profiler.steps`。

**二、OOM 观察者与容器 dict（GPU 探针）**：A（内层 dataclass）→ `_oom_observer_attached=True`，本机 torch 2.11.0+cu129 有 `_cuda_attach_out_of_memory_observer`，**但驱动日志里一条 `[torch_memory]` 都没有** ⇒ 别拿日志判断；B（容器 dict）→ `AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'` 逐字复现，**但 v1 真跑不触发**（5 臂全 exit 0）⇒ 对外口径要收窄成"调用点依赖"，不能写"选了 `torch_memory` 必崩"；C（`enable=False`）→ `check_enable()=False`、`start()` no-op。

**三、msprobe**：共用 venv 没有 msprobe；`pip --target /tmp/m5-msprobe-target` 装 mindstudio-probe 26.2.0 + matplotlib 等 8 个依赖（**venv 一字节未动**）。`import msprobe` 在空 cwd **建 0 个目录** ⇒ "import 期副作用"假设关闭。`tool=precision_debugger` + `stages=[actor_update]` → dump **14 MB**（`step_1/actor_update/step0/rank0/{construct,stack,dump}.json`）；`stages=[critic_update]`（白名单接受、本跑次无产生者）→ **零 dump、零 warning**，"配了不报错也不采"真机成立。

**本轮自己三个坑**：① 首次 `en` 撞 `transfer_queue…SimpleStorage.total_storage_size` 默认 null（`math.ceil(None/n)` TypeError）⇒ 补 `=100000`，与之前批次同源、**非上游缺陷**；② `python /path/script.py` 的 `sys.path[0]` 是**脚本目录**，第一版探针因此导入了 venv 里另一棵树 —— 改 `PYTHONPATH` 钉树 + 打印 `verl.__file__`；③ `ARM=s2` 会同时关掉 `enable=True` 且脚本 `rm -rf prof-en`，已 kill 重做（s2 独立脚本、独立 `save_path`）。

**产物**：[实验 README](experiments/verl-torch-memory-gpu-smoke/readme.md) · 脚本 4 个 · `raw/` 15 份（md5 逐文件与服务端一致）· 读数补进 [M1 证据](open-source/verl/evidence/engine-refactor-torch-memory/readme.md) 与 [handoff](experiments/handoff-m5.md)。

**留给下一轮**：① 4 卡 `inflight=1` 仍需 ≥4 卡窗口（现有 1/4/5）；② "OOM 时真落 snapshot"需人为造 OOM（未做）；③ 带 critic 的真跑里 `critic_update` 行为未测；④ 昇腾未探、无新 NPU 事项；⑤ 对外动作仍待用户逐项授权，本轮**未触上游**。

### M5 本轮（2026-11-04 15:5x–16:4x CST）：4090-1 恢复，把 TM-008 最后一个空格填上——负结论 + 量子端点

**连通性**：`4090-1`（`219.223.198.62`）15:58 复核 **ICMP 0% 丢包、`:22` 可连**（12:5x／13:5x 两次都是公网直连 100% 丢包）；期间出现过一次约 5 分钟的 SSH 超时（同日的分钟级抖动类），nohup 起的远端任务未受影响。昇腾网关 `10.205.0.1` 与 `192.168.13.154/155/156:22` 仍全不可达。**GPU 现场**：0 为 liyaowei 的 Embedding EngineCore（20.8 G）、2/3/6/7 为其 4 卡 Qwen3-VL-32B vLLM（各 23.6 G）⇒ 1/4/5 空闲，正好够 separate_async 的 2 train（4,5 同 socket）＋1 rollout（1）。`/home` 100% 满，全程写 `/tmp`；受测树、TM-008 补丁（`trainer_base.py:170`）、device-map 适配、`/tmp/cupy-pkgs`、venv 与模型数据全部在位（未重建）。

**一、先按预登记阶梯压负载，三点 GATE 全未过。** 校准臂 `n=16`／`max_response=1024`／`sync_step=1`／`mini=4`（64 条/步）稳态 `timing_s/gen` **0.0119／0.0119 s**；抬 cap 到 1536 → 0.0103／0.0111；再抬 `n=32` → 0.0129／0.0112（整步 8.9→13.4 s）。**第一级结构性无效**：响应撞顶率恒 1.6%、均值 ~330，抬 cap 一个 token 也没多给。

**二、机制：两侧按同一 token 数计价。** 训练步消费的就是 rollout 刚产出的轨迹，`n`／长度一抬两侧同倍增长。同一份日志反解训练 **0.241／0.248／0.204 ms/token**（含固定开销 2.0–2.1 s 的权重同步＋adv）；引擎级探针（`gen-cost-probe.py`，vLLM 0.24.0 离线、同模型、GPU 1、同批形状；64 序列 2.37–2.66 s、128 序列 4.38–4.78 s）测生成 **14.5k／18.2k tok/s ＝ 0.069／0.055 ms/token** ⇒ **生成余量 4.5–4.7×**，加负载只把差距拉大（步时 +4.4 s 全在训练侧）。要翻转需生成慢约 4.5×或训练吞吐高约 4.5×（1:2 下≈再加 7 张训练卡，**线性外推、未实测**）。⇒ **预登记规则禁止在此读 A/B，本轮没有跑 2.0/0.1 对照**；"rollout-bound 区间在本机不可达"是负结论，写进 §13.54 的边界。

**三、换等待来源：量子确实回到关键路径。** 窗口内**先预登记、后执行**一条 stale-wait 臂（配置同上，唯一改动 `max_off_policy_threshold` 8→1、`strategy=drop` 不变）：`parameter_sync_step=1` 每步同步权重 ⇒ 上一步轨迹全陈旧，threshold=1 把在飞的 4 条判死（每步 `evicted_samples=4.0`、staleness 2.0），缓冲见底后 `sample()` 必须等下一批（~2 s）。四跑（2.0／0.1 各两跑、8 步、4/4 exit 0）：池化 **gen 2.223→2.766 s（+0.544）、整步 10.038→10.827 s（+0.789）**。**四条预登记全 PASS**（`analyze-stale.py` exit 0；`verify-bound.py raw-stale` 机械 0 失败、GATE 双跑通过、exit 0）。**量化签名在分布上**：2.0 臂 16 个步实例无一低于 1.9 s（最小 2.02／2.09）、**5/16 落在 4.03–4.11 s**（两个量子）；0.1 臂无两量子步、sb01a 4/8 步低于 1.9 s（最小 1.72）。余下 **+0.246 s 是 token 量抽样差**（两臂 +3.7%，按 token 归一后 update_actor 只差 +0.052 s）——**不能只报归一值**。

**口径**：这是**人为制造的等待**（max staleness 1），不代表生产配比，**不能写成"separate_async 平均慢 7.9%"**；单 seed、每臂两跑无 run 级 CI；该臂整步 10–11 s 由等待主导，不可与其他批次秒数比。但它把 §13.54 那句"收益是训练步是否阻塞在 `sample()` 的函数"从推断变成跨区间证据：**同一 mode 从 +0.1%（rollout 领先）到 −7.9%（等待主导）**。

**顺带关掉两项"只能窗口验"的欠账**：真 dump 读作 `poll_interval=2.0`／`max_response_length=1024`／`parameter_sync_step=1`；64／128 条每步的稳态缺口 0.035–0.053 s（阈值 0.174–0.265 s）。另记一条环境噪声：训练正常结束后 `atexit` 会打一条 `DataLoader worker ... killed by signal: Killed` 的 traceback，`exit` 仍 0，属预期。

**产物**：[实验 README](experiments/verl-v1-poll-quantum-separate-bound/README.md) · `raw/`（3 份 GATE 日志＋2 份探针 JSON）· `raw-stale/`（4 份 A/B 日志）· `analyze-gate.py`（exit 2 是正确退出码）· `analyze-stale.py` · `gen-cost-probe.py` · §13.55 · md5 逐文件与服务器一致。

**留给下一轮**：① 4 卡 `inflight=1` 仍需 ≥4 卡窗口（当前只有 1/4/5）；② profiler import 冒烟仍在 M5 交接里排队、优先级低于算力实验；③ 昇腾仍不可达，NPU 侧无新事项；④ 对外动作（TM-008 的 issue／PR）仍待用户逐项授权，本轮未触上游。

> **只读最上面几节（最近两轮 M5 与 M1）即可恢复现场。** 之后的旧批次只留标题索引
> （2026-11-04 瘦身，正文在 `diary/` 与专项文档）。
> 判断"现在该做什么"看 `## 未完成操作与并行工作` 与两个队列，不要从头读到尾。

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

**§13.52 `max_inflight_gen_batches=3` 已测**：GPU 4,5（同 socket＝2+0）、同一 patched 树、DAPO filter on、2.0／0.1 交替各两跑、15 步/跑、4/4 exit 0（18:59–19:39 UTC）。池化 gen **8.589→7.357s（−14.3%）**、整步 **26.964→25.738s（−4.5%）**、生成组 **113.7→133.0（+17.0%）**；其余相位 |Δ|≤0.5%。臂内 FE 斜率 38.8／44.2 ms/组，固定等待 −1.97／−2.04s、多生成 +0.75／+0.83s，残差 ≈−0.01s。跨深度 +5.3%→+13.1%→+17.0%、净收益 −0.96→−1.29→−1.23s ⇒ **depth 2 之后饱和**；depth 2→3 增量 <1 SE，不外推。**方法订正**：跨臂共斜率被处理效应压低（21.2 vs 38.8），同族分析按臂内固定效应报。归档 [experiments/verl-v1-poll-quantum-dapo](experiments/verl-v1-poll-quantum-dapo/README.md#补测max_inflight_gen_batches3-2026-11-03m5-第五十六批)，四份日志 md5 与服务器一致。

**昇腾状态（本轮复核，未可用）**：VPN 网关 `10.205.0.1` 本轮 2 包全丢，回到不可达；`192.168.13.154/155/156` 仍不可达。无新增 NPU 任务需要用户处理。

**§13.53 stale 逐出通道已测**：threshold=2 冒烟 4 步**零逐出**（span 恰为 2，warmup=1 的自然跨度就是 1–2）；threshold=1 才触发——每步丢弃上一步整批 32 组、staleness=2、生成组/步 64。四跑 A/B（2.0／0.1 各两跑，15 步，4/4 exit 0，19:53–20:29 UTC）：2.0 臂 gen 严格两态 2.43/2.47 与 4.43/4.39（差 2.00/1.92s），撞量子 5/12、8/12 步；0.1 臂 0/12。池化 **gen 3.506→2.786s（−20.5%）、整步 22.473→21.997s（−2.1%）**；2.0 臂两跑整步差 0.835s、0.1 臂差 0.049s ⇒ −0.475s ≈1 SE，只作方向性证据。**逐出量与 poll 无关（两臂都 32 组/步）**；threshold=1 本身每步多花约 1.1–1.5s，轮询收不回被丢弃的批次。归档 [experiments/verl-v1-poll-quantum-stale](experiments/verl-v1-poll-quantum-stale/README.md)，md5 与服务器一致。

### M5 补记（2026-11-03 04:1x–05:3x CST，第二会话）：第五十六批独立复核 + main 树运行时冒烟 + `wait` 臂

**独立复核第五十六批**：另写 `verify-inflight3.py`（不同解析路径、按跑分层 bootstrap），逐跑/池化/生成组/有效率全部与 `summary-inflight3.json` 在 1e-12 内一致；md5 12/12、服务器 raw 与本地 4/4 字节一致；四跑均 15/15 步。臂内 FE 斜率 38.79／43.69 ms/组、固定等待 −1.98／−2.07s（原值 −1.97／−2.04）。**口径提示**：`observed = fixed_wait + generation_cost` 是定义式，残差≈0 不算独立证据；独立的只有斜率 CI（36.2–40.8 ms/组）与"跨臂共斜率被压低"。入口 [verification-inflight3.md](experiments/verl-v1-poll-quantum-dapo/verification-inflight3.md)、`verify-inflight3.py` exit 0。

**TM-008 补丁在 main `3fdf2b54` 的运行时冒烟**：浅克隆 main `3fdf2b548882` ＋ `patch -p1 --fuzz=0`（3/3），GPU 4,5、DAPO filter、poll 0.1、inflight 2、3 步，**3/3 exit 0**（20:38:50–20:43:26Z）；解析配置 `'poll_interval': 0.1`，三步均带逐出。`_build_replay_buffer` 全树仅 `trainer_base.py` 一处定义、子类不覆写。**两处环境适配与补丁无关**：① TransferQueue 需显式 `total_storage_size`（main yaml 的 null 与 venv 版本错配，`math.ceil(None/n)` 抛 TypeError）；② `single_controller/base/worker.py` 本地 device-map 适配（非连续掩码 4,5 否则 `invalid device ordinal`）。**非计时测量**；`3fdf2b54` 上的效应量仍需完整 A/B。入口 [README](experiments/verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话)。

**§13.53 补记：`max_off_policy_strategy=wait`（dropless）**：同配置（colocate_async、threshold=1、GPU 4,5、15 步、2.0／0.1 各两跑、4/4 exit 0，20:53–21:28Z）。**零逐出**（`evicted_samples` 缺失、spans／staleness 恒 1），生成组/步 **32**（drop 为 64）。2.0 臂 gen 严格两态 **4.24／2.22（差 2.02s）**，0.1 臂 1.85–2.51。池化 **gen 3.151→2.177（−30.9%）、整步 22.323→21.079（−5.6%）**；2.0 臂跨跑差 0.408s、0.1 臂 0.029s ⇒ 强于 drop 的 ≈1 SE。**跨策略（相邻窗口，方向性）**：wait 在 2.0／0.1 两档都比 drop 快（22.32 vs 22.47；21.08 vs 22.00）且生成量减半 ⇒ wait 在本窗口支配 drop。归档 [stale 实验](experiments/verl-v1-poll-quantum-stale/README.md)，md5 与服务器一致。

**wait 臂独立复核（第二会话，只读）**：另写 `verify-wait.py`（逐 token 切分、按跑分层 4000 次 bootstrap），逐跑/池化 gen、step 与 `summary-wait.json` 在 **1e-9** 内一致；md5 12/12、服务器 raw 与本地 4/4 字节一致；四跑 15/15 步。bootstrap（步级、按跑分层）Δstep **[−1.786, −0.724]s**、Δgen **[−1.392, −0.566]s**，两臂只有两跑 ⇒ **没有 run 级 CI**，−1.24s 只能说窗口内稳定。**代码层两条独立证据**：① `_stale_terminal_keys` 在 `strategy != "drop"` 时直接返回空集 ⇒ wait 永不逐出，`evicted_samples` 键是**缺失而非 0**；② `_has_enough_samples` 用 `>= threshold`，threshold=1 时任何在飞批次都阻塞采样 ⇒ **完整屏障**，`spans==1` 是结构必然不是经验发现；threshold>1 的 wait 行为不同、未测。**口径**：跨策略表同时变了策略与每步生成量（32 vs 64）、且相邻窗口，只作方向性。入口 [verification-wait.md](experiments/verl-v1-poll-quantum-stale/verification-wait.md)、`verify-wait.py` exit 0。

### M5（2026-11-03 06:5x–07:4x CST）：第五十八批 §13.54——separate_async 跑通，轮询量子不进关键路径

**GPU 现场（启动前核对）**：GPU 0 为 liyaowei 的 Embedding EngineCore（20.8 G）、2/3/6/7 为其 4 卡 Qwen3-VL-32B vLLM（各 23.3 G，已跑 5.4 h）⇒ **只有 1/4/5 空闲，4 卡 `inflight=1` 的窗口仍不具备**。改用 2 train（4,5 同 socket）＋1 rollout（1）。**昇腾**：网关 `10.205.0.1` 与 `192.168.13.154` 本轮各 2 包／1 包全丢，仍不可用；只记一次、未重试，无 NPU 任务待办。

**做了什么**：TM-008 清单上最后一个未测的 trainer mode。上游自带 `tests/special_e2e/run_v1_separate_async.sh`（此前只查了 `examples/`），其校验允许 2 train + 1 rollout（训练卡偶数、rollout 卡被 TP 整除，TP=1）。2 步冒烟先撞两处本机环境限制，均非上游缺陷：① `nccl_checkpoint_engine` 加载期 `import cupy`，venv 没有 → 装到 `/tmp/cupy-pkgs`（`/home` 只剩 8 G）；② **`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 在本模式不可用**——`ray.util.collective` 的 `barrier()` 在 `_used_gpu_indices` 为空时按本进程可见卡数建通信器，NOSET 下每 actor 见 3 张卡 ⇒ 每 rank 都占 0/1/2，报 `Multiple Ranks are using the same GPU`；去掉 NOSET（Ray 单卡掩码）后训练侧 FSDP 广播撞 `Cuda failure 217 'peer access is not supported'`，`NCCL_SHM_DISABLE=1` 绕开。代价：训练集合通信退化为 socket，**绝对步时不可与其他批次比**。

**结论（4 跑、8 步、2.0／0.1 各两跑、4/4 exit 0，23:15–23:31Z）**：稳态 `timing_s/gen` **0.031–0.038 s，两档无差别**；整步 **8.211→8.201 s＝+0.010 s（t=0.12）** ⇒ **poll 不在关键路径上**。机制：训练步 8.2 s（old_log_prob 3.1＋update_actor 3.0＋update_weights 1.85），rollout 池单卡生成 16 条/步远快于此，队列始终非空，`ReplayBufferAsync.sample()` 不进入睡眠循环；step 1 的 4–6 s 等待是 warmup 且与 poll 无关（0.1 臂两跑 5.73 vs 0.03 s）。`spans`／`staleness` 恒 1.00／1.00。**TM-008 的模式覆盖至此完整**：收益是"训练步是否阻塞在 `sample()`"的函数——固定批 −3.2%、DAPO −3.4／−4.9%、4 卡 −4.0%、colocate −1.9%、stale drop −2.1%、wait t=1 −5.6%、wait t=2 ≈0、**separate_async ≈0**。归档 [separate 实验](experiments/verl-v1-poll-quantum-separate/README.md)，`verify.py` 34 项全 PASS、md5 5/5、`summary.json` 入库。

**边界**：单栈单点（0.5B／gsm8k／单 seed／8 步／每臂两跑）；本配置是 rollout-ahead 区间（1 rollout 卡 vs 2 训练卡），rollout 成为瓶颈时会空队列、量子可能回到关键路径，**该区间未测**；rewards 池化为 0（0.5B 在 256 token 内做不出 gsm8k），训练侧偏轻。


> **历史盘点（2026-11-04 瘦身）**：以下旧批次叙述已全部保存在 `diary/` 与专项文档中，
> 这里只是按标题保留的一句话索引，正文请到 diary 对应日期查看。

| 段 | 一句话 |
|---|---|
| M1 上一轮（2026-11-04 13:0x–15:4x）：#6067 三次体检（接口面／启动配置键／torch_memory 快照） | 三个信号族：50 个旧接口名里 13 条仍在文档里、`use_legacy_worker_impl` 删后 4 处引用、torch_memory 链路通但示例缺 `enable` |
| M1 上一轮（2026-11-04 09:0x–10:0x）：第十七批——截断本身就是组的长度预测，它对 `R` 没有剩余空间 | 第十五、十六两批都记「这条线已到边际，下一轮应转语义面新模型」。§11.5 语义面缺口四项（长度预测误差／分桶／迁移／抢占／KV 搬运）里，**长度预测**是唯一不需用户立项、能… |
| M1 上一轮（2026-11-04 08:0x–08:5x）：第十六批——`band_floor` 那一侧的判据也有单位数问题，而且它**不单调** | 第十五批留给下一轮的第①条：`band_floor` 的 `no_reading`（"全部 30 个种子恒零"）与 `degenerate`（"过半"）还没接到 `falsy_r… |
| M1 上一轮（2026-11-04 07:0x–08:0x）：第十五批——同一个门槛，判据被用在多少个单位上？ | 第十三批的结论是"`stable` 假阳为 0、假阴 1.1%，且 1.1% 就是二项分布本身"。拉出真实 16 格一看：**表态种子只有 3–27 个**（1.5B cap25… |
| M1 上一轮（2026-11-04 06:0x–06:5x）：第十四批——"这一格问不出来"是负载属性，还是这次抽样？ | 第十一批把带宽为 0 分成三档，`no_reading`（每个种子的 `R` 都恒为 0）的判词写成"这一格问不出来"，并在「已知未做」里留了一句"没有回答换一种抽样能否问得出来… |
| M1 上一轮（2026-11-04 05:0x–05:3x）：第十三批——"档名稳"不等于"效应超带宽" | 上一批（门槛校准）量出 `sign_regime` 假阳为 0，三处文档据此写了"`stable` 档名可用"。核了一下那个 0 的取法：场景把真值设成"每种子均值 0.30、尺… |
| M1 上一轮（2026-11-04 04:3x–05:1x）：第十二批——审 `band_floor` 三档判据，补掉它自记的边界缺口，抓出一处「同口径」名不副实 | … |
| M1 上上轮（2026-11-04 04:1x）：第十一批——带宽为 0 有两种成因，`fifo_band` 加三档准入 | … |
| M1 上一轮（2026-11-04 03:1x）：第十批——同号占比分母拆开、四档 `sign_regime`；并修好上一批遗留的崩溃与三处计数错 | … |
| M1 上上轮（2026-11-03 22:2x）：推送恢复；补齐非计数输出的逐值对照与盲区扫描 | … |
| M1 上一轮（2026-11-04 02:0x–02:5x）：先核实工作区那份未提交的拆分，再修掉判据里的过度概括（第十批前半） | … |
| M1 上一轮（2026-11-04 00:0x）：分桶探针与真实负载纳入逐值对照；两个探针脚本的容差接进共享工具 | … |
| M1 上一轮（2026-11-03 21:0x）：归档不变 ≠ 两版实现相同——补做逐值对照与跨脚本一致性 | … |
| M1 上一轮（2026-11-03 20:0x–20:3x）：判据抽成共享工具；`wall_ratio` 也建在同一个 0.9 上 | … |
| M1 上一轮（2026-11-03 16:4x–19:0x）：上一轮那个"未定位的失败"是判据错了；阈值摆幅拆开后，p4 的形状只值 0.3–3.1 步 | … |
| M1 上一轮（2026-11-03 16:0x–16:3x）：C 类账的"收敛快慢"还带着一个阈值——0.85–0.95 之间摆 1.7–18.2pp，p3 的 `A` 会翻号 | … |
| M1 上一轮（2026-11-03 15:0x–16:0x）：真实负载上的策略位移也要分档——cap256 那两格"反号"落在基线自身抖动里 | … |
| M1 上一轮（2026-11-03 14:0x–14:3x）：分桶探针的均值读数分档——只有 cap1024 ≥4 批站得住，并核出 TM-011 立项前置是改造排程器 | … |
| M1 上一轮（2026-11-03 13:1x–13:3x）：分桶探针的两处口径再修一次，并把 `grouping_tax` 组臂修复与档案一起落地 | … |
| M1 上一轮（2026-11-03 13:0x–13:2x）：§13.30 第一批的 `R` 六个读数作废并重算——组臂归还粒度曾经退化 | … |
| M1 上一轮（2026-11-03 11:3x–12:1x）：TS-001——给排程器补一条输入守卫（参差组能静默量出负税），并把"分桶／预测"缺口量了个上限 | … |
| M1 上一轮（2026-11-03 09:0x–11:1x）：TS-001 复核——上一批归档字段里挑出两处「检查在报 OK 但没在看那个东西」 | … |
| M1 上一轮（2026-11-03 08:0x–09:1x）：TS-001 补强——把 `R` 搬到真实 rollout 长度上，第三批的结论在真实负载上**反号**（§13.30 第四批／§11.4／§11.5） | … |
| M1 上一轮（2026-11-03 07:0x）：TS-001 补强——清掉语义面里一句从没被量过的推断（§13.30 第三批／§11.4／§11.5） | … |
| M1 上一轮（2026-11-03 06:0x）：TS-001 补强——按 §19.4 第 8 条把核验扩到放置效应，抓出一处分母串用 | … |
| M1 上一轮（2026-11-03 05:0x）：TS-001 补强——先修核对脚本的配对错档，再补同节点配对两档（§20.3.9／§11.6） | … |
| M1 上一轮（2026-11-03 04:0x）：TS-001 补强——P-6 的适用面加 async 轨迹与同配置重复臂（§20.3.8／§11.6） | … |
| M1 上一轮（2026-11-03 03:0x）：TS-001 补强——P-6 完成，整步比的可重建形式（§20.3.7／§11.6） | … |
| 上一轮（2026-11-03 02:0x）：TS-001 补强——锚② 的可换算性复核（§20.3.6／§11.6） | … |
| 上一轮（2026-11-03 01:0x–01:3x）：新登记 TM-010 | … |
| 上一轮（2026-11-03 00:0x）：TM-009 材料齐备 | … |
| M5 上一轮（2026-11-03 02:0x–02:4x CST）：第五十五批 §13.51 | … |
| M5 上一轮（2026-11-03 00:5x–01:4x CST）：第五十四批 §13.50 | … |
