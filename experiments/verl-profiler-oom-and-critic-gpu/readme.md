# verl profiler：OOM 时快照是否真落盘（真机三臂）

更新：2026-11-04 18:0x CST 采集（上一会话中断，未入库）；**2026-11-05 00:0x CST 本轮核对、判定脚本与文档**。

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

## critic 侧三臂：**已备、未跑**

`run-critic-pdbg.sh` 是为 `critic_update` / `compute_values`（白名单里有、全仓无装饰器）准备的真机对照：同一配方下只换 `precision_debugger.stages`，`au=[actor_update]` 作正臂、`cu`/`cv` 作被测臂；配方用 `adv_estimator=gae` ＋ `critic.enable=True` ＋ `critic.model.path`，所以"没采到"不能解释成"这次没有 critic"。

本轮无卡，只做了**开跑前的静态把关**（[compose_check.py](compose_check.py)，CPU）：

- 35 条 override（10 条基础配方＋17 条探针 EXTRA＋7 条收尾参数＋1 条 `stages`）在 main `3fdf2b54`（worktree `/tmp/verl-3fdf2b5`）上**三臂全部组合成功**，负向对照（`critic.fsdp.no_such_field=1`）**如期失败** ⇒ 不会死在 Hydra 组合。
- 键序要紧：基础配方先给 `adv_estimator=grpo`，探针的 `=gae` 在后 ⇒ 生效值是 **gae**（脚本里数组在 EXTRA 之前，Hydra 后者胜）。组合后回读确认：`critic.fsdp.model_dtype=bf16`、`critic.fsdp.optimizer_offload=True`、`critic.forward_max_token_len_per_gpu=3000`、`actor.model.override_config.attn_implementation=sdpa` 都按预期取值。
- **`critic.use_dynamic_bsz` 继承 actor 为 True** ⇒ `critic.ppo_micro_batch_size_per_gpu=1` 这条是**惰性的**，真正约束显存的是 `ppo_max_token_len_per_gpu=3000`。写结论时不要把 micro batch 当变量。
- 两个会在窗口里直接炸的点（都不是上游缺陷，但脚本必须带住）：
  1. `load_valuehead_model` 在 `verl/utils/model.py:475` 与 `:500` **写死 `attn_implementation="flash_attention_2"`**，且调用点（`workers/engine/fsdp/transformer_impl.py:304`）不传该参数 ⇒ 本机没装 FA2 时 critic 建模型即失败，而 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` **只作用于 actor/rollout**，救不了 critic。脚本依赖服务端树上的本地补丁（`/tmp/m5-main-smoke/verl`，未入库）⇒ **重建树后必须重打，否则 critic 臂必然失败**。
  2. msprobe 与 trl 都装在 `/tmp` 的 `--target` 目录（`/tmp/m5-msprobe-target`、`/tmp/m5-trl-0.27.0`），共享 venv 未动；`/home` 100% 满时这两个目录也可能被清掉。

## 复现

```bash
# OOM 三臂（服务端，单卡）
SRC=/tmp/m5-main-smoke/verl G=1 OUT=/tmp/m5-oom bash run-oom-probe.sh
# 本地核对（只用 raw/ 里的 JSON，不需要 GPU）
python3 verify-oom-archive.py            # 49 项，exit 0
python3 verify-oom-archive.py --selftest # 7 项注入，全部被捕捉
# critic 三臂（服务端，逐臂跑；ARM=au|cu|cv）
ARM=cu G=4 bash run-critic-pdbg.sh
# 开跑前的组合把关（本机，CPU；worktree 用被测的那个 commit 建）
#   git -C <verl 检出> worktree add --detach /tmp/verl-3fdf2b5 3fdf2b548882
python3 compose_check.py --tree /tmp/verl-3fdf2b5 --out raw/compose-check.json
python3 compose_check.py --selftest    # 负向对照能否被打翻
```

## 产物

- 探针与跑批：[probe-oom-dump.py](probe-oom-dump.py) · [run-oom-probe.sh](run-oom-probe.sh) · [run-critic-pdbg.sh](run-critic-pdbg.sh)（未跑）
- 判定与把关：[verify-oom-archive.py](verify-oom-archive.py) · [compose_check.py](compose_check.py) · [raw/compose-check.json](raw/compose-check.json)
- 原始读数：`raw/result-{attached,nocap,unranked}.json` · `raw/nocap.stderr`
- 上游对应：[证据文档](../../open-source/verl/evidence/engine-refactor-torch-memory/readme.md) · [M5 交接](../handoff-m5.md)
