更新时间：2026-11-04（M1，TM-006 第五段）

# #6067 之后，`torch_memory` 的周期快照还灵不灵

## 一句话结论

**灵，但只灵一半，而且官方示例脚本本身跑不出快照。** 走 `start_profile/stop_profile` 这条新链路时快照会写出来（CPU 侧真实调用链验证：`outputs/profile/step1/`，tag `torch_memory`）；但三个具体缺口是实的：

1. **官方示例 `examples/profile/run_qwen2_5_vl_7b_torch_memory.sh` 在当前 main 上不产出任何快照**——它没给 `*.profiler.enable=True`，而快照只在 `DistProfiler.start()` 里推进窗口，`enable=False` 直接跳过。`docs/perf/verl_profiler_system.md` 也没把 `enable=True` 写进配方（#6198 的 PR 说明里写了，文档里没有）。
2. **旧 worker 那条显式调用在新后端上是 `AttributeError`**——`'TorchMemoryProfiler' object has no attribute 'dump_memory_snapshot'`。（这正是 #6198 当初删调用的原因，所以这条只是把事件按时间线摆正，不是新 bug。）
3. **不带 `dataclass_type` 的 `omega_conf_to_dataclass(cfg.profiler)` 会让后端在构造期崩**——`tool_config` 是**容器 dict**（键是工具名），`DistProfiler` 在没显式传 `tool_config` 时会整个容器交给后端，于是 `'dict' object has no attribute 'trace_alloc_max_entries'`。`verl/trainer/main_ppo.py:187`（`if config.trainer.use_v1` 分支，`run_ppo` 里最后一次调用的形状）与 `verl/workers/engine_workers.py:645-647`（`ActorRolloutRefWorker.init_model` 给内层 actor 建 profiler）都是这个写法。

对外价值判断：**前两条属文档/示例欠债，第三条是会在启动期抛异常的硬失败**，但它只在 `tool=torch_memory`（或 `npu`）且走到那两个调用点时才触发，默认配置不受影响。

## 怎么查的（都在 CPU 上，没跑训练、没上 GPU）

仓库：`/tmp` 的独立 clone，钉在 verl main `74543265`（2026-11-04 fetch；父提交 `044bbba2^` = `6eeb5711`）。
环境：Python 3.12.13 + ray 2.59.0 / omegaconf 2.3.1 / hydra 1.3.8 / torch 2.15.0 / transformers 5.22.1 / tensordict 0.15.4。

```
/tmp/verl-seqlb-venv/bin/python 的 ray 太旧（缺 orjson 等），本轮另建 /tmp/tm-probe-venv
```

两个脚本，各自带负向对照：

| 脚本 | 查什么 | 读数 |
|---|---|---|
| [`repro_torch_memory.py`](repro_torch_memory.py) | 入口面（父 vs HEAD 两个 worker 类的公开方法）+ 真实配置驱动 `DistProfiler` 的三个探针 | 10/10 PASS（`--selftest` 另 2 条扰动也全过） |
| [`probe_official_example.py`](probe_official_example.py) | 照官方示例脚本的 overrides 渲染真实配置 | 3/3 PASS |

复现命令：

```bash
git clone --no-checkout --quiet /path/to/verl /tmp/verl-6067-w3 && cd /tmp/verl-6067-w3
git checkout -q 74543265
python repro_torch_memory.py --repo /tmp/verl-6067-w3 --json repro-result.json --selftest
python probe_official_example.py --repo /tmp/verl-6067-w3 --json example-result.json
```

## 三条发现的证据

### 1. 入口：父提交有 `dump_memory_snapshot`，HEAD 两个 worker 类都没有

AST 读类体（不看注释）：

| | 父提交 `044bbba2^` | HEAD `74543265` |
|---|---|---|
| `ActorRolloutRefWorker` | 含 `dump_memory_snapshot` | 无（`compute_log_prob`/`compute_ref_log_prob`/`execute_checkpoint_engine`/`init_model`/`load_checkpoint`/`save_checkpoint`/`set_loss_fn`/`to`/`update_actor`/`update_weights`） |
| `CriticWorker` / `TrainingWorker` | `CriticWorker` 本就没有 dump | `TrainingWorker` 也没有 |

时间线（都对得上提交）：

| 日期 | 提交 | 做了什么 |
|---|---|---|
| 2025-08-20 | `864ba998` | #3099 引入每步后的 `actor_rollout_wg.dump_memory_snapshot(...)` |
| 2026-04-20 | `044bbba2` | #6067 删掉 fsdp/megatron worker（连带这个入口），**当时 trainer 里的调用已经拦在 `hasattr` 后面、静默跳过** |
| 2026-04-21 | `84ac897d` | #6075 把 separation trainer 的 `_fit_torch_memory` 整体删掉 |
| 2026-04-29 | `78afaa5b` | #6198 删掉 `ray_trainer.py` 里最后那条调用，PR 说明写"capabilities 已并入 DistProfiler" |

→ 所以"旧调用现在会 AttributeError"是**已经发生并被处理过**的事件，本轮不作为对外主张；它的作用是把时间线摆正，说明 `torch_memory` 现在**只剩** `start_profile/stop_profile` 一条路。

### 2. 这条路本身是通的（含两条负向对照）

真实配置（`critic.profiler.enable=True` + `global_profiler.tool=torch_memory`）→ `omega_conf_to_dataclass` 拿到 `ProfilerConfig(tool='torch_memory', enable=True, save_path='outputs/profile', tool_config.torch_memory=TorchMemoryToolConfig)` → `DistProfiler.start(role='train', profile_step=1)` / `stop()` → 间谍记录到 **1 次** `dump_memory_snapshot(out_dir='outputs/profile', tag='torch_memory', sub_dir='step1')`。

负向对照：① 不选工具时后端是 `_NoOpProfiler` 且**不写**；② `memory_snapshot_num_steps=2` 时单步**不写**（窗口判据确实在生效）。⇒ "能写"不是探针自说自话。

### 3. 官方示例脚本跑不出快照

示例脚本的 overrides 只有 `global_profiler.tool=torch_memory`、`save_path`、两个 tool_config 值。按它渲染真实配置后：

| 角色 | tool | **enable** | save_path |
|---|---|---|---|
| `actor_rollout_ref.actor` | torch_memory | **False** | ./mem_snapshots |
| `actor_rollout_ref.rollout` | torch_memory | **False** | ./mem_snapshots |
| `critic` | torch_memory | **False** | ./mem_snapshots |

`DistProfiler.check_enable()` = False ⇒ `start()` 不推进窗口、`stop()` 早退 ⇒ **一次都不写**。手工补 `actor.profiler.enable=True` 后立刻写出 1 张，路径是角色级 `save_path`（继承全局 `./mem_snapshots`，与示例的 `PROFILE_SAVE_PATH` 一致）。

该示例脚本最后修改是 2026-08-31（`b356f430`），文档 `docs/perf/verl_profiler_system.md` 最后修改是 2026-09-08（`f1c27141` #7770）——**都晚于 #6198**，等于"删掉旧入口之后，示例和文档都没补上 `enable`"。该脚本没有被 `.github/workflows` 引用，所以 CI 不会提醒。

### 4. 不带 `dataclass_type` 的 instantiate 会让后端构造期崩

```
omega_conf_to_dataclass(cfg.critic.profiler)              → ProfilerConfig(tool_config=<dict>)
DistProfiler(rank=0, config=prof)                         → tool_config 取 config.tool_config（整个容器 dict）
TorchMemoryProfiler(..., tool_config=<dict>)              → AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'
```

`tool_config` 是容器（`{'torch_memory': TorchMemoryToolConfig, 'npu': …, 'precision_debugger': …}`），内层那个才是真 dataclass。显式传内层（`tool_config=<该工具自己的 dataclass>`）就正常——trainer 侧确实这么做了，但**另外两个调用点没有**：

- `verl/trainer/main_ppo.py:187-188`（`run_ppo` 的 v1 分支，选中 `TaskRunnerV1`）与 `main_ppo_v0.py` 的旧 runner：配置进到 runner 里已经 `OmegaConf.resolve` 过，`omega_conf_to_dataclass` 只对**当次传入的 DictConfig** 做 `instantiate`，已经转成 dataclass 的属性不会被递归转换 ⇒ `tool_config` 保持容器 dict。
- `verl/workers/engine_workers.py:645-647`：`ActorRolloutRefWorker.init_model` 给内层 actor 建 profiler（注释自称"必须走 hydra path 才能拿到真 dataclass"）。
  - 补一条**已核清的限定**：同一个文件 `:155-166` 的 `TrainingWorker` 是**显式**取内层（`profiler_config.tool_config.get(tool)`），实测能正常构造 —— 所以问题只在"不显式取内层"的那些调用点。`omega_conf_to_dataclass` 对已是 dataclass 的输入是 no-op（实测同一对象），所以重复转换本身不制造问题，**问题在 `tool_config` 容器被整体下传**。

**边界**：这两处触发要求 `tool=torch_memory`（或 `npu`）；默认 `tool=None` 走 `_NoOpProfiler`，不报错。`ActorRolloutRefWorker` 那处在 `use_v1=True`（新默认）时不会被调用——它的 profiler 归 trainer 的 `start_profile` 管。**没有跑端到端**，所以"用户选 torch_memory + 走 v1 分支就一定崩"这一点是静态 + CPU 复现的推断，不是训练日志。

## 没覆盖的

- 快照是否**真落盘**：CPU 上 `MemorySnapshotSampler` 直接返回 None；
- OOM 观察者是否真的挂到分配器上：需要 CUDA/NPU；
- 端到端 `run_ppo` 冒烟：需要 GPU + 数据（M5 范围，若要写进 [handoff](../handoff-m5.md) 最小目标是 `global_profiler.tool=torch_memory` + `actor.profiler.enable=True` + `steps=[1]`，看 `save_path/step1/*.pickle` 是否出现）。

## 与前四段的关系（为什么不重复）

| 段 | 信号族 | 找的是 |
|---|---|---|
| 第一段（11:3x） | 配置消费点 | 字段还在、引擎侧没人读 |
| 第二段（13:0x） | 接口名 / 文档 | 参数名还在文档里，代码里没了 `def` |
| 第三段（14:0x） | 启动配置键 | Hydra 键被删，命令行还在传 |
| **本段（第五段）** | **运行期入口 + 配置替换来源** | 方法名两边都在（`start_profile`），**但配置从"直接读 yaml"换成了"先 dataclass 再往回传"，两级 `ProfilerConfig` 只有一级填得对** |

前四段都靠"名字还在不在"判断；这一段两边名字一模一样，**坏掉的是值的形状**，只能靠把配置真正喂进调用链才看得见。
