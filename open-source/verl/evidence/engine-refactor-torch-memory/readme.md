# verl #6067 的第五个体检信号族：`torch_memory` 的快照，官方示例自己跑不出来

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架）

**【贡献点】** 同一次重构（[#6067](https://github.com/verl-project/verl/pull/6067)，`044bbba2`）的第五次体检。前四批查"名字还在不在"；这一批查**名字一模一样、但值的形状变了**的地方：`torch_memory` 的显存快照原来由 trainer 每步直接调 `actor_rollout_wg.dump_memory_snapshot(...)`（#3099），[#6198](https://github.com/verl-project/verl/pull/6198)（2026-04-29，收 [#6181](https://github.com/verl-project/verl/issues/6181)）删掉调用、改由 `DistProfiler` 在 scheduled step 上采。这条路**确实通**（本轮在 CPU 上按真实配置驱动验证），但：**官方示例 `examples/profile/run_qwen2_5_vl_7b_torch_memory.sh` 没给 `*.profiler.enable=True`，在当前 main 上不产出任何快照**；`docs/perf/verl_profiler_system.md` 的配方同样缺这一项（#6198 的 PR 说明里有，文档里没有）；另外**不带 `dataclass_type` 的 `omega_conf_to_dataclass(cfg.profiler)` 会把容器 dict 当参数传给后端，启动期 `AttributeError`**。

**【类型】** 问题诊断（B 类：主动发掘）

**【价值评估】** 5 / 10 — 比第三、四批（3/10）高。第一条是**用户照着官方示例操作得不到结果**（沉默失败，无报错、无日志）；第三条是**会抛异常的硬失败**，但只在 `tool=torch_memory`/`npu` 且走到 `run_ppo` 的 v1 分支时才触发，默认配置不受影响。方法上的价值最高：前四批都靠"名字在不在"判断，这一段两边名字完全一样（`start_profile`/`stop_profile`/`ProfilerConfig`），坏的是**配置从"直接读 yaml"换成"先转 dataclass 再往回传"之后，两级 `ProfilerConfig` 只有一级被正确填充**——只有把配置真的喂进调用链才看得见。

**【进展】** 2026-11-04 M1 本地完成。真实 `ppo_trainer.yaml` + 真实 `omega_conf_to_dataclass` + 真实 `DistProfiler` 调用链，CPU 上复现，**两个脚本各带负向对照**（10/10、3/3 PASS）。**未跑训练、未上 GPU/NPU、未触上游**（无 issue／comment／PR）。复现与归档见 [实验目录](../../../../experiments/verl-6067-torch-memory-path/readme.md)。**同日 M5 在 4090-1 真机补记（2026-11-04）**：加 `actor.profiler.enable=True` 后快照**确实落盘**（22.4 MB pickle、39 segments）；官方示例形状（无 `enable`）**一张也不写、目录都不建**；`steps=[2]` 落在 `step2/`、无 `step1/`；OOM 观察者**确实挂在分配器上**（`_oom_observer_attached=True`），但驱动日志里没有任何 `[torch_memory]` 行；容器 dict 的 `AttributeError` 逐字复现，**默认 v1 真跑不触发**（5 个训练臂全部 exit 0）。真机材料见 [experiments/verl-torch-memory-gpu-smoke](../../../../experiments/verl-torch-memory-gpu-smoke/readme.md)。

**【大白话讲明白】**

显存快照是排查 OOM / 显存碎片最直接的工具之一：给它开个开关，它每隔几步把"每个时刻是谁分配了多少显存"拍一张快照，事后能在 PyTorch 的显存可视化页面里回放。verl 有两套采集方式：

- **老办法**：训练循环每跑完一步，主动喊一声"拍一张"。这声喊是个方法调用。
- **新办法**（#6198 改的）：把"拍快照"这件事交给统一的分析器，配置里写 `tool=torch_memory` 加"在第几步拍"，分析器自己按步数节奏拍。

问题出在两件事上：

1. **新办法要多一个开关没写进示例。** 统一分析器只在"这个角色开启了分析"时才干活（`enable=True`）。官方那份示例脚本只写了"用哪个工具、存到哪"，没有写"给哪个角色开"。于是运行起来**一切正常、没有任何报错**，就是目录里一张快照都没有。
2. **有一处转换会把配置变得"形状不对"。** 配置进到训练进程后要转成 Python 对象再传给分析器。有的调用点转得很细（把某工具自己的那份配置单独取出来传），有的调用点直接把"所有工具的配置字典"整个递过去。后者在碰到具体工具时就会撞上"这个字典没有 `trace_alloc_max_entries` 属性"，训练在启动阶段就崩。

这两条的共同点是：**名字都没变**，代码里搜 `dump_memory_snapshot`、搜 `start_profile` 都能搜到，只有把配置真的喂进调用链才看得见。

**【What–Why–How】**

- **What：** ① 官方示例脚本 `examples/profile/run_qwen2_5_vl_7b_torch_memory.sh` 的 overrides 渲染后，`actor` / `rollout` / `critic` 三个角色的 `profiler.enable` **都是 False** ⇒ 快照路径一步都不触发；② `omega_conf_to_dataclass(cfg.profiler)`（不带 `dataclass_type`）产出 `ProfilerConfig.tool_config` 是**容器 dict**，`DistProfiler` 在没显式传 `tool_config` 时整体下传 ⇒ `TorchMemoryProfiler` 构造期 `AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'`。
- **Why：** #6067 删旧 worker 时把 `dump_memory_snapshot` 一起删了（trainer 里那条调用当时已在 `hasattr` 后面静默跳过），#6198 随后把调用整段删掉并声称"能力已并入 DistProfiler"。这条新链在**代码层面确实可用**，但示例与文档都没补 `enable=True`；而配置的"两级结构"（角色级 `ProfilerConfig` → 工具级 `tool_config.<name>`）在新链路上靠调用方自己取，取错一层就崩，`torch_memory` 恰好是第一个会去读属性的后端。
- **How：** ① `repro_torch_memory.py` 用 AST 比父提交与 main 的 worker 公开方法（入口面），再用 hydra compose 真实 `ppo_trainer.yaml` + `OmegaConf.resolve` + `omega_conf_to_dataclass` 造出 trainer 视角的 profiler 配置，按 `start_profile → DistProfiler.start/stop` 的真实链驱动，间谍记录写盘调用（探针 1）；把同一后端按旧 worker 的调用形状调（探针 2）；只给 `config` 不给 `tool_config`（探针 3）。② `probe_official_example.py` 逐行照抄示例脚本的 overrides 渲染，读三个角色的 `enable`，再手工打开 `enable=True` 看写到哪。③ 负向对照：不选工具时后端必须是 `_NoOpProfiler` 且不写；`memory_snapshot_num_steps=2` 时单步不写。**边界**：CPU 上 `MemorySnapshotSampler` 直接返回 None，所以"文件是否落盘"未验证；OOM 观察者（需要 CUDA/NPU 分配器）未验证；端到端 `run_ppo` 未跑，第③条属静态 + CPU 复现的推断。

<!-- contribution-summary:end -->

## 一、结论

| 信号 | 结果 | 是否算问题 |
|---|---|---|
| 父提交 `ActorRolloutRefWorker` 公开方法里的 `dump_memory_snapshot` | 有 | — |
| 当前 main 两个 worker 类（`ActorRolloutRefWorker` / `TrainingWorker`）里的 snapshot 方法 | **0** | 入口已整体消失 |
| `start_profile → DistProfiler → TorchMemoryProfiler.stop` 是否写出快照 | **写 1 次**（`outputs/profile/step1/`，tag `torch_memory`） | 链路本身**是通的** |
| 官方示例脚本渲染后三个角色的 `profiler.enable` | **全为 False** | **示例欠债：跑不出快照** |
| `docs/perf/verl_profiler_system.md` 配方是否含 `enable=True` | 否（#6198 的 PR 说明里有） | **文档欠债** |
| 不带 `dataclass_type` 的 `omega_conf_to_dataclass(profiler)` → `DistProfiler(config=…)` | **AttributeError** | **硬失败**（仅 `tool=torch_memory`/`npu`，仅特定调用点） |
| 旧 worker 那种显式 `impl.dump_memory_snapshot(...)` 打在新后端上 | `AttributeError: 'TorchMemoryProfiler' object has no attribute 'dump_memory_snapshot'` | 已在 #6198 处理，仅作时间线证据 |

## 二、时间线（每条都对得上提交）

| 日期 | 提交 | 做了什么 |
|---|---|---|
| 2025-08-20 | `864ba998` | #3099 引入每步后的 `actor_rollout_wg.dump_memory_snapshot(...)` |
| 2026-04-20 | `044bbba2` | #6067 删除 fsdp/megatron worker（连带 `dump_memory_snapshot`）；trainer 里的调用已拦在 `hasattr` 后面，**静默跳过** |
| 2026-04-21 | `84ac897d` | #6075 删掉 separation trainer 的 `_fit_torch_memory`（连带 `dump_memory_snapshot` 调用） |
| 2026-04-29 | `78afaa5b` | #6198 删掉 `ray_trainer.py` 最后那条调用，PR 说明："capabilities are fully integrated in DistProfiler"，配方给了 `global_profiler.tool=torch_memory` + `save_path` + `steps` + **`actor.profiler.enable=True`** |
| 2026-08-31 | `b356f430` | 示例脚本 `run_qwen2_5_vl_7b_torch_memory.sh` 最后修改——**没补 `enable`** |
| 2026-09-08 | `f1c27141` | #7770 给 torch_memory 加 OOM 观察者与多步窗口；文档 `docs/perf/verl_profiler_system.md` 最后修改——**配方仍缺 `enable`** |

⇒ 示例与文档**都在 #6198 之后**改过，但都没补上那一项。

## 三、三条发现的证据

### 1. 链路本身是通的（不是"删了就没了"）

真实配置（`critic.profiler.enable=True` + `global_profiler.tool=torch_memory`）→ `omega_conf_to_dataclass` 得到 `ProfilerConfig(tool='torch_memory', enable=True, save_path='outputs/profile', tool_config.torch_memory=TorchMemoryToolConfig(memory_snapshot_num_steps=1))` → `DistProfiler.start(role='train', profile_step=1)` / `stop(run_command=False)` → **1 次** `dump_memory_snapshot(out_dir='outputs/profile', tag='torch_memory', sub_dir='step1')`。

负向对照（`--selftest`）：不选工具 ⇒ 后端 `_NoOpProfiler`、不写；`memory_snapshot_num_steps=2` ⇒ 单步不写。

### 2. 官方示例跑不出来

照示例脚本的 overrides 渲染后：

| 角色 | tool | enable | save_path |
|---|---|---|---|
| `actor_rollout_ref.actor` | torch_memory | **False** | ./mem_snapshots |
| `actor_rollout_ref.rollout` | torch_memory | **False** | ./mem_snapshots |
| `critic` | torch_memory | **False** | ./mem_snapshots |

`DistProfiler.check_enable()` = False ⇒ `start()` 不推进窗口、`stop()` 早退 ⇒ 一次都不写。手工补 `actor.profiler.enable=True` 后立即写出 1 张，路径为角色级 `save_path`（继承全局 `./mem_snapshots`）。该示例脚本不在 `.github/workflows` 里，CI 不会提醒。

### 3. 容器 dict 被当参数传 ⇒ 构造期 `AttributeError`

```
omega_conf_to_dataclass(cfg.critic.profiler)   → ProfilerConfig(tool_config=<dict>)   # 容器：{'torch_memory':…, 'npu':…, …}
DistProfiler(rank=0, config=prof)              → tool_config 取 config.tool_config（整个容器）
TorchMemoryProfiler(..., tool_config=<dict>)   → AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'
```

显式传内层（`DistProfiler(..., tool_config=prof.tool_config.get('torch_memory'))`）则正常。**已核清的限定**：`verl/workers/engine_workers.py:155-166` 的 `TrainingWorker` 就是显式取内层，实测能构造；问题只在"不显式取内层"的调用点，如 `:645-647` 的 `ActorRolloutRefWorker.init_model`（注释自称"必须走 hydra path 才能拿到真 dataclass"）与 `run_ppo` 的 v1 分支（`main_ppo.py:187`）。`omega_conf_to_dataclass` 对已是 dataclass 的输入是 no-op（实测返回同一对象），所以**重复转换本身无害，问题在容器被整体下传**。

## 四、口径与边界

- 本轮**只证明"配置喂进调用链后会出现什么"**；没有跑训练，所以"用户选 `torch_memory` + 走 v1 就一定崩"是**推断**，不是训练日志。
- 快照是否真落盘、OOM 观察者是否挂上分配器，都需要 GPU/NPU（M5 范围）。
- 不主张"示例脚本坏了"——它只是**不产出结果**（无报错），这是文档/示例欠债的典型形态，对外表述应写"照示例操作拿不到快照，配方缺 `enable=True`"。

## 五、与同系列其他体检的关系

| 批次 | 信号族 | 找的是 |
|---|---|---|
| [配置消费点](engine-refactor-config-consumers/readme.md) | 字段还在、引擎侧没人读 | 3 条真失效（`tiled_mlp`、`freeze_vision_tower`、`use_prefix_grouper`） |
| [接口面](engine-refactor-api-surface/readme.md) | 名字没了但文档仍在教 | 3 个名字（`update_policy` 等）+ profiler critic 侧 stage 缺口 |
| [启动配置键](engine-refactor-shell-config/readme.md) | 键被删、命令行还在传 | 1 处启动即报错、2 处传了没人读 |
| **本批** | **名字两边都在，值的形状变了** | 示例缺 `enable`（沉默失败）+ 容器 dict 下传（硬失败） |

前三批都能靠文本/符号比对发现；本批必须**把真实配置真正喂进调用链**才看得见。

## 六、真机补记（2026-11-04 M5，`4090-1` 单卡）

CPU 阶段列为"需 GPU"的三项已全部关闭，另加两条 M1 没提的读数：

| 读数 | 配置 | 结果 |
|---|---|---|
| 快照是否落盘 | `tool=torch_memory` + `steps=[1]` + `actor.profiler.enable=True` | `save_path/step1/torch_memory_rank0_pid*.pickle`，23,494,044 B；39 segments、`device_traces` 1×100,000（撞 `trace_alloc_max_entries` 上限） |
| 官方示例形状 | 同上，**只去掉 `enable`**（`steps=[1]` 保留） | 2 步 exit 0，`save_path` **目录都不存在** ⇒ 沉默失败真机复现 |
| 窗口是否跟配置走 | `steps=[2]` | 落在 `step2/`，**无 `step1/`** |
| OOM 观察者 | 构造 `DistProfiler(tool=torch_memory, enable=True)` | `_oom_observer_attached=True`；本机 torch 2.11.0+cu129 有 `_cuda_attach_out_of_memory_observer`。**但驱动日志里一条 `[torch_memory]` 都没有**（成功行与失败 warning 都没有）⇒ 别拿日志判断它挂没挂 |
| 容器 dict | `tool_config={torch_memory: ...}` | `AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'` 逐字复现；**但默认 v1 路径不会走到**——真跑 `tool=torch_memory` 的 5 个臂全部 exit 0 |

⇒ 对外口径要收窄一处：第三条的"启动期硬失败"是**调用点依赖**的，不是"选了 `torch_memory` 就走 v1 必崩"。真机材料（脚本、15 份原始日志、md5 对照）见 [experiments/verl-torch-memory-gpu-smoke](../../../../experiments/verl-torch-memory-gpu-smoke/readme.md)。**本轮仍未触上游**。
