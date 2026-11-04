# verl profiler：OOM 时快照是否真落盘（真机三臂）

更新：2026-11-04 18:0x CST 采集（上一会话中断，未入库）；2026-11-05 00:0x CST 核对、判定脚本与文档；**2026-11-05 01:0x CST 无卡轮：critic 三臂的树内适配与判定做成可复跑脚本（零 GPU 读数）**；**2026-11-05 02:0x CST 无卡轮：判定读的日志计数器拿 104 份真日志验过一遍——原实现会把有效跑判成无效，已修（零 GPU 读数）**。

回答 [M5 交接](../handoff-m5.md) 里"`torch_memory` 的自动 OOM 快照只在**观察者挂上了**这一步验过、OOM 时是否真写文件未测"这一格。上一轮（11-04 16:5x）在 2 步 PPO 里只拿到 `_oom_observer_attached=True`、驱动日志 0 条 `[torch_memory]`，没有制造 OOM。

## 环境与树

| 项 | 值 |
|---|---|
| 机器 / 卡 | `4090-1`，GPU 1（RTX 4090 24 GiB） |
| 树 | `/tmp/m5-main-smoke/verl`（main `3fdf2b54` 浅克隆，即 16:5x 轮同一棵） |
| torch | 2.11.0+cu129；`torch._C._cuda_attach_out_of_memory_observer` 存在 |
| 时间 | `oom_20261104-100119` / `-100422`（UTC 10:01:19 / 10:04:22 ＝ CST 18:01 / 18:04） |

## 三臂设计（两两对照）

同一个脚本、同一张卡、同一棵树上跑三个独立进程，只有一处不同：

| 臂 | rank / `config.ranks` | 观察者 | 日志 | 作用 |
|---|---|---|---|---|
| `attached` | 0 / `[0]` | 挂上 | root DEBUG ＋ Capture handler | 正臂 |
| `nocap` | 0 / `[0]` | 挂上 | **不装 handler**（用户默认形态） | 只改日志，看"用户能不能看见" |
| `unranked` | 3 / `[0]` | **不挂** | 不装 handler | 负向对照 |

三臂都用同一种方式逼出分配器 OOM：`torch.empty(int(1e12), dtype=torch.uint8, device="cuda")`。dump 走的是**出厂实现**（`TorchMemoryProfiler._on_out_of_memory` → `MemorySnapshotSampler.dump_memory_snapshot`），不是重写。

## 读数

| 指标 | `attached` | `nocap` | `unranked` |
|---|---|---|---|
| `should_profile_this_rank` / 观察者 | True / True | True / True | **False / False** |
| OOM 抛出 | `torch.OutOfMemoryError`（931.32 GiB 请求） | 同 | 同 |
| 落盘 | **1 个 pickle，6275 B** | **1 个，6275 B** | **0 个（5 个 poll 全空）** |
| `oom_request_seconds` | 0.145 | 0.178 | **0.005** |
| 进程内日志 | 4 条（1 INFO ＋ 3 ERROR） | **0 条**（无 handler） | 0 条 |
| stderr | — | **3 条 ERROR**（`lastResort`） | — |
| OOM 后 CUDA 可用 | True | True | True |

- 目录形状：`<save>/oom_<ts>/torch_memory_oom_rank0_pid<PID>.pickle`，**在第一个 poll（0 s）就已完整**，1/3/10/20 s 四次复查内容与大小不变 ⇒ 不是"异步慢写"造成的假阴性。
- 内容：`keys = [allocator_settings, device_traces, external_annotations, segments]`，`device_traces` 1×3 项，`segments` 1。
- 文件名里的 `rank0` 取自**环境变量 `RANK`**（`memory_utils.py:300`，缺省 `"0"`），不是 `TorchMemoryProfiler(rank=…)`。本探针直接构造 profiler、不经过 Ray，进程没设 `RANK`，所以写出来是 0；真实训练里由 Ray 设，但"文件名 rank ≠ `config.ranks`"这条形状在，排查时别拿文件名当 `config.ranks` 的证据。
- 负向对照是**结构性**的：`_attach_oom_observer()` 只在 `_should_profile_this_rank()` 为真时调用（`torch_memory_profile.py:80-81`），rank 3、`ranks=[0]`、`all_ranks=False` ⇒ 永远不挂；同一份 OOM 代码路径下它一个文件也没写。

## 为什么"挂上了"上一轮却一条日志也没有

`_attach_oom_observer` 的成功提示是 **INFO**（`torch_memory_profile.py:110`），而 root logger 默认 WARNING ⇒ 默认配置下"挂上了"永远看不见；OOM 路径的三条日志是 **ERROR**，默认就能经 `logging.lastResort` 打到 stderr（`nocap` 臂实测）。上一轮与本轮读的不是矛盾，是两条不同事件。

**还有一条：dump 成功也没有人告诉你在哪。** `dump_memory_snapshot` 成功后打 `"[memory_visualize] dumped: <path>"`（`memory_utils.py:321`），但

```python
logger = logging.getLogger(__file__)          # memory_utils.py:28  logger 名是文件路径，不在 verl.* 层级
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))   # :29  默认 WARN ⇒ INFO 被自己挡掉
```

⇒ 即使把 root 和 `verl` 都调到 DEBUG（`attached` 臂就是这么做的），这条 INFO 仍然不会出现——两份独立读数一致：静态（logger 名＋默认级别）与真机（`attached` 臂 4 条记录里 0 条来自 `memory_utils`）。OOM 现场用户只看到"dumping allocator snapshot"，看不到"dump 到哪/是否成功"。

**第三方交叉核对（不花 GPU）**：上一轮 16:5x 的 3 份真 PPO 日志里快照确实写了 23.5 MB，但 `run-en.txt`／`run-s2.txt`／`run-ex.txt` **各 0 条 `[memory_visualize]`、0 条 `dumped:`**（`torch_memory` 那 10 处命中全是 Hydra 配置回显）⇒ 同一条抑制对**周期性 `stop()` 的 dump 一样成立**，不只 OOM。

**可改的最小面（对外动作，未授权）**：`_on_out_of_memory` 丢掉了 `dump_memory_snapshot` 的**返回值**，两行即可——拿到 `Path` 后按 ERROR 打一行路径；返回 `None` 时补一行"dump 失败"。不建议改 `getLogger(__file__)`：全仓 56 处同一写法，属既有约定。

## 边界（引用时必带）

1. **这是类级探针，不是真训练 OOM。** 分配器几乎是空的（free 24.836 / total 25.251 GB，`allocated_memory_gb≈5e-7`），所以快照只有 1 段、3 条 trace、6.3 KB。真训练 OOM 时分配器状态大得多，**dump 耗时与文件大小都不可由本例外推**（0.145 s 只是"空分配器"的代价，且单跑无重复）。
2. **`nocap` 与 `attached` 只差日志**，所以"落盘与否"不受日志配置影响这一条成立；但 `unranked` 同时差了 rank 与日志配置——它的"不落盘"由 `_should_profile_this_rank` 的静态结构兜底（见上），不是仅凭一次对照。
3. **未与服务端互核**：JSON 由上一会话拷回，本轮 4090-1 自 18:12 后不可达（ICMP 100% 丢包、`:22` 超时），服务端 `/tmp/m5-oom` 的原始文件未做 md5 对照；本地唯一证据是 `raw/` 里三份 JSON 与 `nocap.stderr`（其时间戳 `10:04:22,541` 与 nocap 的目录名 `oom_20261104-100422-541802` 对得上）。
4. NPU 侧未做；`_on_out_of_memory` 在 npu 分支改走 `device.memory._snapshot()`（`memory_utils.py:313-318`），行为不同，不能套用本页。

## critic 侧三臂：**已备、未跑（2026-11-05 起适配与判定都已脚本化）**

`run-critic-pdbg.sh` 是为 `critic_update` / `compute_values`（白名单里有、全仓无装饰器）准备的真机对照：同一配方下只换 `precision_debugger.stages`，`au=[actor_update]` 作正臂、`cu`/`cv` 作被测臂；配方用 `adv_estimator=gae` ＋ `critic.enable=True` ＋ `critic.model.path`，所以"没采到"不能解释成"这次没有 critic"。

本轮无卡，只做了**开跑前的静态把关**（[compose_check.py](compose_check.py)，CPU）：

- 35 条 override（10 条基础配方＋17 条探针 EXTRA＋7 条收尾参数＋1 条 `stages`）在 main `3fdf2b54`（worktree `/tmp/verl-3fdf2b5`）上**三臂全部组合成功**，负向对照（`critic.fsdp.no_such_field=1`）**如期失败** ⇒ 不会死在 Hydra 组合。
- 键序要紧：基础配方先给 `adv_estimator=grpo`，探针的 `=gae` 在后 ⇒ 生效值是 **gae**（脚本里数组在 EXTRA 之前，Hydra 后者胜）。组合后回读确认：`critic.fsdp.model_dtype=bf16`、`critic.fsdp.optimizer_offload=True`、`critic.forward_max_token_len_per_gpu=3000`、`actor.model.override_config.attn_implementation=sdpa` 都按预期取值。
- **`critic.use_dynamic_bsz` 继承 actor 为 True** ⇒ `critic.ppo_micro_batch_size_per_gpu=1` 这条是**惰性的**，真正约束显存的是 `ppo_max_token_len_per_gpu=3000`。写结论时不要把 micro batch 当变量。
- 两个会在窗口里直接炸的点（都不是上游缺陷，但脚本必须带住）：
  1. `load_valuehead_model` 在 `verl/utils/model.py:475` 与 `:500` **写死 `attn_implementation="flash_attention_2"`**，且调用点（`workers/engine/fsdp/transformer_impl.py:304`）不传该参数 ⇒ 本机没装 FA2 时 critic 建模型即失败，而 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` **只作用于 actor/rollout**，救不了 critic。补丁原先只手改在服务端 `/tmp/m5-main-smoke/verl` 上（未入库）⇒ **重建树后必须重打**；2026-11-05 起由 [patch-critic-sdpa.py](patch-critic-sdpa.py) 幂等完成，`run-critic-pdbg.sh` 预检会自动调用（见下节）。
  2. msprobe 与 trl 都装在 `/tmp` 的 `--target` 目录（`/tmp/m5-msprobe-target`、`/tmp/m5-trl-0.27.0`），共享 venv 未动；`/home` 100% 满时这两个目录也可能被清掉。

## 窗口前置（2026-11-05 M5 无卡轮）：两件怕丢的东西变成可复跑，判定先写死

上一轮把 critic 三臂准备好，但两个前提只存在于服务器 `/tmp` 里，而且**当时没有预登记判定**——跑完再商量"算不算通过"，等于让窗口自己给自己打分。本轮（双链路仍不可达）只做这两件事，零 GPU 读数。

**一、树内适配变成脚本：`patch-critic-sdpa.py`。** 只在 `load_valuehead_model` 块内把两处 `attn_implementation="flash_attention_2"` 换成 `"sdpa"`（`--impl` 可改），块外一个字不动。`--check` 只报状态，`--revert` 逆操作，**锚点数量不等于 `--expect`（=2）就 exit 2**：上游若自己加了参数、删了一个调用或改了字面量，脚本拒绝报"已修好"——一个自称打好补丁其实没打的树会烧掉整个窗口。

用**真上游文件**验证（`verl-project/verl` `3fdf2b548882` 的 `verl/utils/model.py`，sha256 `a13d8595…`，只读取回）：

| 步骤 | 读数 |
|---|---|
| `--check`（未打） | `state=unpatched`，`anchors_in_block=2`，第 **475／500** 行，函数块 467–510，块外 0 处 ⇒ exit 1 |
| `apply` | 精确改 **2** 行，块外 0 处，新 sha256 `61c23872…` ⇒ exit 0 |
| diff | 只有两个 hunk，各一行 `flash_attention_2` → `sdpa`（[raw/critic-sdpa-patch.diff](raw/critic-sdpa-patch.diff)） |
| `--revert` | 回到**逐字节**相同，sha256 与上游一致 ⇒ exit 0；再 `--check` 又回 exit 1 |

自检 **18/18**：块外同字面量不被改、二次 apply 幂等（sha 不变）、一个锚点／三个锚点／函数不存在全部 exit 2 且**不写文件**、树不存在 exit 3。证据 [raw/critic-sdpa-patch.json](raw/critic-sdpa-patch.json)。

**二、判定预登记：`verify-critic-pdbg.py`。** 判的是每臂的 `arm-<ARM>.json`（由 `run-critic-pdbg.sh` 生成，**失败也写**），不靠手抄数字：

| 类别 | 检查 | 不过的处理 |
|---|---|---|
| 有效性 V1–V4 | 三臂 exit 0；`steps_seen ≥ 2`；`cu`/`cv` 日志里出现 `timing_s/update_critic` ⇒ 真有 critic；树的分类器状态 `fa2.state == patched` | **exit 1 无效跑**，三臂什么也不能证明 |
| 门控 P1 | `au` 在 `step_*/actor_update/` 下至少有 `construct.json`／`stack.json`／`dump.json` | exit 1（正臂不亮，plumbing 假设不成立） |
| 门控 P2／P3 | `cu`／`cv` 的 dump 文件数 **= 0** | **exit 2 推翻**：某 stage 真有产生者，静态结论要重写 |
| 只报告 | 三臂的 `Unknown precision_debugger stage: train_batch` 条数（静态说法是它与用户配置无关地每步刷一条，含 `au`）、dump 清单、窗口时间 | 不参与判定；为 0 就不要引用那条说法 |

自检 **10/10**：干净夹具 exit 0，`au` 缺 dump／dump 落在别的 stage 目录／臂缺失／某臂 exit≠0／步数不足／没有 critic／树丢补丁全部 exit 1，`cu`／`cv` 冒 dump 全部 exit 2。

**三、wrapper 也加固了。** `run-critic-pdbg.sh` 开跑前检查 8 个依赖（树、树内适配脚本、msprobe／trl 两个 `--target`、config json、模型、两份数据）并**在秒级失败**，再查树是否已打补丁（`AUTO_PATCH=1` 自动打，`=0` 则中止而不是擅自改树）；每臂结束写 `arm-<ARM>.json`。用**假 example 脚本**在本机把整条管线跑通（三个臂 wrapper rc=0、judge exit 0），产物在 [raw/mock-run/](raw/mock-run/README.json)——**那是管线夹具，不是读数**，只证明预检、自动打补丁、JSON 生成与判定能接上。

**边界**：`sdpa` 只是文本替换，critic 侧**数值等价未测**；本轮**零 GPU 读数**；`raw/mock-run/` 里的一切都不得当作 critic 行为的证据。窗口里真正的答案仍来自三份真日志。

**四、上游现状只读复核（2026-11-05 01:1x，不改变任何结论）。** 最新 main 已前移到 `31cfccb4`（2026-11-04T12:54Z，`#8463`）。两点与窗口直接相关：① `verl/utils/model.py` 在 `main` 与 `3fdf2b54` **逐字节相同**（sha256 `a13d8595…`，475／500 两处 FA2 都在）⇒ 即使窗口里改用最新 main 建树，`patch-critic-sdpa.py` 同样适用，锚点数仍是 2；② `verl/workers/engine_workers.py`（903 行）的 `@DistProfiler.annotate` 只有 **4 个 role**：`train_batch`（`:377`）、`ref_compute_log_prob`（`:751`）、`actor_compute_log_prob`（`:758`）、`actor_update`（`:768`）——**仍无 `critic_update`／`compute_values`**，静态结论在最新 main 上不变。**边界**：本轮只核了 `model.py` 与这一个 worker 文件，**不是全仓重扫**；`compute_rm_score` 在该文件里 0 处，但它本来就不在这个 worker 里产生。

## 2026-11-05 02:0x 无卡轮：判定用的日志计数器拿真日志验了一遍——**原来的会误判**

上一轮把判定预登记下来，但**判定读的数**没有拿真日志核过。本轮（两条算力链路仍不可达：4090-1 ICMP 100% 丢包＋`:22` 超时，昇腾网关 2 包全丢）只做这件事，仍是**零 GPU 读数**。

**一、`steps_seen` 在真实日志上恒为 0 ⇒ 有效跑会被判无效。** 原实现是 `re.finditer(r"^step:(\d+)", text, re.M)`，而 Ray 写控制台时每行前面都有 `<ANSI>((TaskRunnerV1 pid=…)<ANSI> ` 前缀，**没有任何一行以 `step:` 开头**。V2 的门槛是 `steps_seen ≥ 2`，所以一个完全正常的两步跑会拿到 0，判 `exit 1 无效跑`——窗口白烧，而且预登记的判定还被"看起来跑过了"掩盖。

读数不是靠推理，是把仓库里**全部 104 份真控制台日志**过一遍（[check-log-counters.py](check-log-counters.py) `--verify`，逐文件表在 [raw/log-counter-check.json](raw/log-counter-check.json)）：

| 计数器 | 读数 |
|---|---|
| 旧计数器（`^step:`）为 0 的日志 | **104 / 104** |
| 新计数器（每个训练步的 `timing_s/step` 标记）与标记数一致 | **104 / 104**（合计 **1432** 个训练步） |
| 随手去掉行首锚点（在任何位置找 `step:N`）与真值不一致 | **103 / 104**（g8-g4w：`naive_max=58` 而真值 12；`run-pda`：46 对 2） |

⇒ "只要把锚点去掉就行"也是错的：`timing_s/step:8.2` 里的 `step:8`、checkpoint 路径 `global_step_46` 都会被当成步号。**判据只能是"带 `- ` 分隔的步行 ＋ `timing_s/step` 标记"**，它对裸行与带前缀两种形状都成立。

**二、为什么上一轮的端到端自检没抓到：夹具不是真日志的形状。** 假 example 脚本打印的是裸 `step:1 - …`（[raw/mock-run/](raw/mock-run/README.json)），于是旧计数器在夹具上给出 2、在真机上给出 0。⇒ **端到端跑通只证明管线接上了，不证明读数是真日志能长出来的**。新夹具改成真形状（ANSI＋`TaskRunnerV1 pid=` 前缀、`step:0` 验证行无 `timing_s/*`、混入 `global_step_46` 路径），见 [mock/make-mock-tree.sh](mock/make-mock-tree.sh) 与 [raw/mock-run-real-shape/](raw/mock-run-real-shape/README.json)。

**三、新夹具当场又抓出第二处：V4 数错了范围。** V4 原来要求**全文件** `attn_implementation="flash_attention_2"` 计数为 0；夹具在 `load_valuehead_model` 块外故意留了一处（上游目前块外恰好 0 处，所以线上一直没暴露），于是树明明已打好补丁也判 `flash_attention_2_left=2`。现改为读**预检用的同一个分类器**（`patch-critic-sdpa.py` 的 `analyze()` → `state == "patched"`），文件级计数保留作信息字段；负向对照 `outside_literal_not_counted`（state=patched 且文件里有 1 处块外字面量）必须 exit 0。

**四、`timing_s/update_critic` 这条 V3 前提补了静态证据。** `use_critic = need_critic(config)`，配置里显式 `critic.enable=True` ⇒ 恒 True（`utils.py:98`）；步内 `marked_timer("values")`（`trainer_base.py:595`）与 `("update_critic")`（`:604`）都在 `if self.use_critic` 下，且**不受 `critic_warmup` 影响**（那只挡 `update_actor`）；`_compute_metrics`（`:1885`）把整个 `timing_raw` 交给 `compute_timing_metrics`（`:1963`），后者输出 `timing_s/{k}` ⇒ 有 critic 时**每步各出现一次**。

**五、顺手关掉两个"窗口里才发现"的外部前提。** ① 组合把关原是在 `3fdf2b54` 上做的，而 `3fdf2b54 → 31cfccb4` 之间 **8 个配置 yaml 变了**（`actor/actor.yaml`、`engine/fsdp.yaml`、`engine/megatron.yaml`、`rollout/rollout.yaml` 与四个 `_generated_*`），全部是删 QAT 段与改注释——但仍重跑了一遍：`31cfccb4` 上**三臂组合成功、负向对照如期失败、17 个回读键与旧批次逐键相同**（hydra 1.3.8／omegaconf 2.3.1，旧批 1.3.2／2.3.0），读数在 [raw/compose-check-main-31cfccb4.json](raw/compose-check-main-31cfccb4.json)；树用 `codeload` tarball 取，`verl/utils/model.py` sha256 仍是 `a13d8595…` 可对上。② wrapper 调用的 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 在 `31cfccb4` 上仍在（200，5368 B），且脚本读取的 25 个环境变量**覆盖 wrapper 设置的全部键**（`MODEL_PATH`／`TRAIN_FILE`／`TEST_FILE`／`NGPUS_PER_NODE`／批大小／长度／`ROLLOUT_*`／`SAVE_FREQ`／`TEST_FREQ`／`PROJECT_NAME`／`EXPERIMENT_NAME`／`VERL_USE_UV`）⇒ 没有"脚本改签名"这类静默错配。

**边界（本轮仍然全部适用）**：零 GPU 读数，`raw/mock-run-real-shape/` 仍是管线夹具；`104/104` 与 `1432 步`统计的是**仓库里已归档的日志**，不是新跑的结果；新计数器在真机日志上会给出什么，要等窗口给出第一份 critic 日志才算实测。另外，**旧夹具 `raw/mock-run/` 的 arm JSON 是旧 schema**（`fa2.state` 记的是预检状态 `already-patched` 而不是分类器状态），拿现在的 judge 去判它会得到 `exit 1 INVALID RUN` 与 `fa2.state=already-patched` 的明细——这是**有意 fail-loud**：混用新旧 arm JSON 时会直接说不合格，而不是静默放过。

**六、顺带把 wrapper 的两条失败路径补住**（都不是读数）：① 预检原来只查树与 `/tmp` 依赖，**不查 `patch-critic-sdpa.py` 自己**——缺了会走成"`--check` rc=2（锚点形状变了？）"，把排查引向错的地方；现把它列进依赖，报 `MISSING: …/patch-critic-sdpa.py`（exit 65，秒级）。② arm JSON 原来是 `python3 … > tmp` 之后**无条件** `mv`，生成器中途失败会把空/截断文件顶上去，judge 读它直接抛异常而不是给判定；现改为只在文件非空时才替换。两条都用同一套夹具验过：`/tmp/m5-isolated` 里只放 wrapper、不放适配脚本 → `MISSING`＋exit 65；正常夹具三臂 rc=0／judge 0 不变。

## 2026-11-05 03:0x 无卡轮：critic 的第二个拦路点——**就算把装饰器补回去，模型也解析不出来**

前几轮把 critic 三臂的窗口前置做得差不多了，但证据页第三节那句"补装饰器要动引擎层的 role 身份，属设计选择"一直是**推断**。这一轮把它换成可执行的读数：`PrecisionDebuggerProfiler.start()` 在真正构造 msprobe 之前还要过一道 `_resolve_model(self_instance, stage)`，它按 `_MODEL_ATTRS_BY_ROLE[role]` 在**worker 对象**上找模型，最后兜底 `self.module`。新引擎里 critic 是一个独立的 `TrainingWorker`（`trainer_base.py`：`Role.Critic -> TrainingWorker`），模型在 `self.engine.module`，**既没有 `self.critic` 也没有 `self.module`** ⇒ 四条 critic 候选链全部落空。

读数由 [probe-critic-model-resolve.py](probe-critic-model-resolve.py) 产出：用 AST 从**真实文件**里抽出 `_STAGE_TO_ROLE`、`_MODEL_ATTRS_BY_ROLE`、`_resolve_model`／`start` 等方法源码执行（不需要 torch／msprobe／GPU），替身 worker 的属性名**从同一棵树里的真实类体读出来**，不是手写。

| 检查 | 当前 main `31cfccb4` | 父提交 `6eeb5711` |
|---|---|---|
| `_STAGE_TO_ROLE`／`_MODEL_ATTRS_BY_ROLE` 字面量 | 与父提交**完全相同**（4 个 role／19 条候选逐项相等） | — |
| critic 两个 stage 的装饰器字面量（全仓 `verl/**/*.py`） | `critic_update` 0／`compute_values` 0 | `fsdp_workers.py`＋`megatron_workers.py` 各 2 处 |
| critic 的模型解析 | `TrainingWorker` 上 `critic_update`／`compute_values` 都 **None** | `CriticWorker` 上命中 `critic_module` ⇒ **Module** |
| actor／ref 正对照 | 命中（`ActorRolloutRefWorker.actor.engine.module`、`.ref.engine.module`） | — |
| `start(stage="critic_update", model=None)` | 返回 **False**、打一条 `PrecisionDebugger model not resolved for stage 'critic_update'`、**不建目录**（`os.makedirs` 在模型检查之后） | — |

三点需要写进口径：

1. **解析表不是这次坏的，是 worker 布局变了。** 两棵树的表逐字相同；旧 `CriticWorker` 有 `self.critic_module`，新 `TrainingWorker` 只有 `self.engine`（AST 读出 11 个 `self.*` 赋值，无 `module`、无 `critic`）。所以这是一个**双缺口**：装饰器丢了，解析链的落点也没了。
2. **同一个 `train_batch` 同时服务 actor 和 critic**（`ActorRolloutRefWorker.update_actor → self.actor.train_mini_batch → self.train_batch`）⇒ 给 `train_batch` 写一个**字面量** role 无论写哪个角色都会错一边，这也是"role 身份"问题的具体形状。
3. **候选修法可以在本机验**：给每个 role 的候选链**末尾**追加 `"engine.module"` 后，critic 两臂解析成功，actor／ref 两个正对照读数**不变**（表里原候选先命中）。这只是形状验证，**没有改动上游任何文件**。

自检 **8/8**（抽取完整、3.9 语法垫片恰好 1 处、基线为 None、三种扰动各自翻转、表恢复、unknown stage 仍被拒）；两棵树的读数在 [raw/critic-model-resolve-main-31cfccb4.json](raw/critic-model-resolve-main-31cfccb4.json) 与 [raw/critic-model-resolve-pre6067.json](raw/critic-model-resolve-pre6067.json)（父提交文件含 `match` 语句，本机 Python 3.9 走正则兜底，读数里记 `parse_mode`）。

**边界（与本轮结论一起引用）**：替身 worker 只有真实类体里的属性名，**不是真 worker 实例**；未装 msprobe、未跑训练、未上卡。它证明的是"这条解析路径在 `TrainingWorker` 上不可能拿到模型"，而不是"msprobe 在真机上会怎么表现"。另外，三臂真机跑的 P2／P3（`cu`／`cv` 零 dump）**两种原因都能满足**——没有装饰器，或者装饰器在但模型解析不出来——本轮把后者从推断变成静态可查，但三臂本身只给行为侧证据，分不出是哪一个原因。

## 2026-11-05 04:0x 无卡轮：critic 的模型构建分支查清了——**trl 不在这条路上，而 preflight 一直把它当硬依赖**

三臂的每个窗口前置都逐轮在收口，但这轮翻的是另一种性质的隐患：**不是"跑起来会不会错"，而是"依赖清单本身对不对"**。`run-critic-pdbg.sh` 从 01:0x 起把 `/tmp/m5-trl-0.27.0` 和 msprobe 一起放在同一个 `for dep in ...` 里，任缺即 `exit 65`。如果服务器 `/tmp` 被清掉一部分（`/home` 100% 满，这不是假设），**一个本来能跑的好窗口会被自己的预检判死**。

**一、读数：Qwen2 critic 走第一分支，trl 全程不 import。** `verl/utils/model.py:load_valuehead_model`（`:467`）先试 `AutoModelForTokenClassification.from_pretrained(...)`（`:471`），只有它抛异常（`:479` 的 `except BaseException`）才落到 trl 的 `AutoModelForCausalLMWithValueHead`（`:488`）。而 transformers 的 `_get_model_class`（`auto_factory.py:178`）只做 `model_mapping[type(config)]` 一次查表，`config.architectures` **只用来在候选类列表里挑一个**（`:184-187`）——所以拿一份 causal-LM 权重去建 token-classification 模型**不会报错**，只会随机初始化 `score` 头并告警。Qwen2 在映射里（`modeling_auto.py:1686` 的 `("qwen2", "Qwen2ForTokenClassification")`），引擎又在调用前把 `hf_config.num_labels` 设成 1（`transformer_impl.py:294`）⇒ 路径固定为第一分支，`trl` 一次都不会被 import。

**二、所以依赖名册要分两类。** msprobe 仍是**硬依赖**（没有它 profiler 是 no-op，`au` 正臂不可能出 dump，窗口会白烧成"无效跑"）；trl 只在**回退分支**才需要。[probe-valuehead-branch.py](probe-valuehead-branch.py) 只读 `config.json`（不加载权重、不碰 GPU）回答这个问题，wrapper 按它的退出码分派：

| 探针结论 | trl 目录 | wrapper 行为 | 含义 |
|---|---|---|---|
| 分支 1（`rc=0`） | 缺失 | **继续** | 这正是此前的误杀：trl 不在路径上 |
| 分支 2 不可用（`rc=65`） | 缺失 | 中止 65 | 真需要，并打印确切的 `pip install --target` 命令行 |
| 不可判定（`rc=2`） | 缺失 | 中止 65 | 未知一律沿用旧严格规则 |
| 不可判定（`rc=2`） | 在 | 继续 | 未知但回退件在 |

四行都由桩替换探针实测（不是推断），并且每臂的 `arm-<ARM>.json` 现在记 `valuehead_branch`（含 `resolved_class`），证据页可以写"critic 是 `Qwen2ForTokenClassification` ＋ 全新 1 维头"，而不是笼统的"一个 value model"。

**三、判定不动。** `verify-critic-pdbg.py` 只把 `valuehead_branch` 打进 reported（`_branch_label()`：记录到类名／`undecidable (...)`／`not-recorded`），**不进任何门槛**——预登记是在没有这个字段时写的，分支也不改变 P1–P3 问的事。自检 12 → **14**，新增两条钉住这一点：`valuehead_branch` 缺失 → 仍 exit 0；改成 `trl_valuehead` → 仍 exit 0。

**边界（与本节结论一起引用）**：① 这是对**固定版本源码**的核对（verl `31cfccb4`、transformers 5.16.1 wheel、trl 0.27.0 wheel，两个 wheel 的 sha256 在 [raw/valuehead-branch-evidence.json](raw/valuehead-branch-evidence.json)），**不是运行时读数**；② 本机解释器是 3.9，`transformers 5.16.1` 装不上，所以探针的 real-mapping 分支**第一次执行是在服务器上**——因此失败方向必须是保守的：任何异常都归 `undecidable`，沿用严格规则，最坏是丢一个窗口，不会静默换成错分支；③ trl 0.27.0 两条导入路径都在（`trl/experimental/ppo/__init__.py:16`、`trl/__init__.py:46` 的废弃导入），类有 `pretrained_model`／`v_head`（`modeling_value_head.py:634/678`）⇒ 回退分支本身是成立的，只是这条路用不到。

## 复现

```bash
# OOM 三臂（服务端，单卡）
SRC=/tmp/m5-main-smoke/verl G=1 OUT=/tmp/m5-oom bash run-oom-probe.sh
# 本地核对（只用 raw/ 里的 JSON，不需要 GPU）
python3 verify-oom-archive.py            # 49 项，exit 0
python3 verify-oom-archive.py --selftest # 7 项注入，全部被捕捉
# critic 三臂（服务端，单卡，逐臂跑；ARM=au|cu|cv）
#   预检会自动补 FA2→sdpa（AUTO_PATCH=0 则只检查不改树），每臂写 arm-<ARM>.json
SRC=/tmp/m5-main-smoke/verl G=4 ARM=cu bash run-critic-pdbg.sh
# 判定（本机，CPU，只读 arm-*.json）：0=符合预登记 1=无效跑 2=被推翻
python3 verify-critic-pdbg.py --out <OUT>
python3 verify-critic-pdbg.py --selftest   # 14 项，含"cu 冒 dump → 2"与"valuehead_branch 不进门槛"
# critic 用哪条分支建模型（预检内部会调，也可单独跑；只读 config.json，不需要 GPU）
python3 probe-valuehead-branch.py --model /home/zxw/models/Qwen2.5-0.5B-Instruct --json /tmp/vh.json
python3 probe-valuehead-branch.py --selftest   # 5 项（real-mapping 两条要有 transformers 才跑）
# 判定用的日志计数器：拿仓库里全部真日志验（不需要 GPU、不需要服务器）
python3 check-log-counters.py --selftest                             # 6 项，含负向对照
python3 check-log-counters.py --verify --out raw/log-counter-check.json
# 整条管线（预检→自动补丁→跑→arm JSON→判定）用真形状夹具跑一遍，不需要 GPU
bash mock/run-mock.sh /tmp/m5-critic-mock
# 树内适配（独立可用）：--check 只读，--revert 逆操作，锚点形状变了 exit 2
python3 patch-critic-sdpa.py --tree <verl 树> --check
python3 patch-critic-sdpa.py --tree <verl 树> --selftest   # 18 项
# 开跑前的组合把关（本机，CPU；worktree 用被测的那个 commit 建）
#   git -C <verl 检出> worktree add --detach /tmp/verl-3fdf2b5 3fdf2b548882
python3 compose_check.py --tree /tmp/verl-3fdf2b5 --out raw/compose-check.json
python3 compose_check.py --selftest    # 负向对照能否被打翻
# critic 的模型解析（本机，CPU；两棵树各一份读数，0=结论成立 2=上游已变）
python3 probe-critic-model-resolve.py --selftest
python3 probe-critic-model-resolve.py --tree /tmp/verl-31cfccb4 --preset main \
  --json raw/critic-model-resolve-main-31cfccb4.json
python3 probe-critic-model-resolve.py --tree /tmp/verl-pre6067 --preset pre6067 \
  --json raw/critic-model-resolve-pre6067.json
```

## 产物

- 探针与跑批：[probe-oom-dump.py](probe-oom-dump.py) · [run-oom-probe.sh](run-oom-probe.sh) · [run-critic-pdbg.sh](run-critic-pdbg.sh)（**未跑真机**）
- 树内适配：[patch-critic-sdpa.py](patch-critic-sdpa.py) · 真上游文件验证 [raw/critic-sdpa-patch.json](raw/critic-sdpa-patch.json)／[.diff](raw/critic-sdpa-patch.diff)
- 判定与把关：[verify-critic-pdbg.py](verify-critic-pdbg.py) · [check-log-counters.py](check-log-counters.py) · [verify-oom-archive.py](verify-oom-archive.py) · [compose_check.py](compose_check.py) · [probe-critic-model-resolve.py](probe-critic-model-resolve.py) · [probe-valuehead-branch.py](probe-valuehead-branch.py) · [raw/valuehead-branch-evidence.json](raw/valuehead-branch-evidence.json)（分支核对与 preflight 四档实测） · [raw/compose-check.json](raw/compose-check.json)（`3fdf2b54`）· [raw/compose-check-main-31cfccb4.json](raw/compose-check-main-31cfccb4.json)（当前 main）· [raw/critic-model-resolve-main-31cfccb4.json](raw/critic-model-resolve-main-31cfccb4.json)／[raw/critic-model-resolve-pre6067.json](raw/critic-model-resolve-pre6067.json)
- 原始读数：`raw/result-{attached,nocap,unranked}.json` · `raw/nocap.stderr` · [raw/log-counter-check.json](raw/log-counter-check.json)（104 份真日志的计数器对照）
- 管线夹具（**非读数**）：[raw/mock-run/](raw/mock-run/README.json)（旧形状，掩盖了计数器问题）· [raw/mock-run-real-shape/](raw/mock-run-real-shape/README.json)（真日志形状）· [mock/](mock/make-mock-tree.sh)
- 上游对应：[证据文档](../../open-source/verl/evidence/engine-refactor-torch-memory/readme.md) · [M5 交接](../handoff-m5.md)
