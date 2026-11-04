更新时间：2026-11-04（M5 有卡轮）

# torch_memory / precision_debugger 在真机上的落盘验证

## 一句话结论

M1 第五段（[torch_memory 链路](../../experiments/verl-6067-torch-memory-path/readme.md)）在 CPU 上的三条读法，在 `4090-1` 上一一对上；其中一条需要改口径（OOM 观察者不能拿日志判断）。

1. **快照真的落盘**：`global_profiler.tool=torch_memory` + `steps=[1]` + `actor.profiler.enable=True` + 2 步 PPO，产出 `save_path/step1/torch_memory_rank0_pid*.pickle`，22.4 MB，`sha256 155b747b…`；解开是 39 个 segment、`device_traces` 1×100,000（撞 `trace_alloc_max_entries` 上限）。
2. **不给 `enable=True` 就一张也不写**：同一配置只去掉 `enable`（官方示例 `run_qwen2_5_vl_7b_torch_memory.sh` 的形状，`steps=[1]` 保留），跑完 2 步 exit 0、`save_path` **目录都不存在**。⇒ 静默失败在真机上复现，且原因被隔离到 `enable` 一项。
3. **窗口跟着配置步走**：`steps=[2]` 时快照落在 `step2/`，**`step1/` 目录不存在** ⇒ 不是"只要有 enable 就每步都写"。
3. **OOM 观察者确实挂上了**：`_oom_observer_attached=True`，本机 torch 2.11.0+cu129 有 `torch._C._cuda_attach_out_of_memory_observer`。但**驱动日志里一条 `[torch_memory]` 都没有**（既没有 attach 成功行，也没有失败 warning）——判断它挂没挂只能看进程内状态。
4. **容器 dict 的 `AttributeError` 逐字复现**（`'dict' object has no attribute 'trace_alloc_max_entries'`），但**默认 v1 路径不会走到它**：真跑 `tool=torch_memory` 的 4 个臂全部 exit 0，没有启动期崩溃。
5. **msprobe**：`import msprobe` 在 cwd 创建 **0 个目录**（"import 期副作用"这条残留假设关闭）；`tool=precision_debugger` 下，**有产生者的 stage（`actor_update`）正常 dump（14 MB），白名单接受但本跑次无产生者的 stage（`critic_update`）零 dump、零 warning**——"配了不报错也不采"在真机上成立。

## 怎么查的

| 项 | 值 |
|---|---|
| 机器 | `4090-1`（219.223.198.62），GPU **1**（单卡；0 与 2/3/6/7 是 liyaowei 的 vLLM） |
| 受测树 | `/tmp/m5-main-smoke/verl`（main `3fdf2b54` 解包 + TM-008 poll 补丁 + 本机 device-map 适配；无 `.git`，`verl/version` = 0.10.0.dev） |
| 补丁是否碰 profiler | 否——`poll-interval-config.patch` 只改 `ppo_trainer.yaml`、`trainer/ppo/v1/trainer_base.py`、`trainer/ppo/v1/trainer_separate_async.py` |
| 环境 | venv `/home/zxw/venvs/rollout-g8`，Python 3.10.12、torch 2.11.0+cu129 |
| 模型/数据 | Qwen2.5-0.5B-Instruct、gsm8k `train_sub480`/`test_sub120` |
| 配方 | 2 步、`train_batch=8`、`n=2`、`max_prompt=512`、`max_response=128`、`total_training_steps=2` |
| 输出 | 全部写 `/tmp`（`/home` 100% 满，剩 7.6 G） |

四个训练臂，同树同卡同配方，只差 profiler 配置：

| 臂 | profiler 配置 | 退出 | 落盘 |
|---|---|---|---|
| `en` | tool=torch_memory, steps=**[1]**, save_path, **actor.enable=True** | 0（09:02:37–09:05:47Z） | `prof-en/step1/torch_memory_rank0_pid2387464.pickle`，23,494,044 B |
| `ex` | 同 `en`，**只去掉 enable**（官方示例形状；steps=[1] 保留） | 0（09:06:28–09:09:04Z） | **无**（`prof-ex` 目录不存在） |
| `s2` | 同 `en`，`steps=**[2]**` | 0（09:23:19–09:26:22Z） | `prof-s2/step2/torch_memory_rank0_pid2486586.pickle`，23,651,307 B；**无 `step1` 目录** |
| `pda` | tool=precision_debugger, stages=**[actor_update]**, enable=True, strict=False | 0（09:15:04–09:18:17Z） | `prof-pda/step_1/actor_update/step0/rank0/{construct,stack,dump}.json`（14 MB） |
| `pdc` | 同 `pda`，stages=**[critic_update]** | 0（09:19:14–09:21:51Z） | **无**（`prof-pdc` 不存在），日志 0 条 warning |

## 快照内容（不是"文件存在"就算数）

```
keys = [allocator_settings, device_traces, external_annotations, segments]
segments len = 39
device_traces = [100000]   # == trace_alloc_max_entries 默认 100000，说明记录窗口是满的
torch = 2.11.0+cu129, cuda_available = True
```

pickle 本体 23 MB **不入库**（二进制、可重跑），只留 `raw/verify-snapshot.json` 与 sha256。

## 构造路径探针（`probe-profiler-path.py`，GPU 上跑，钉树）

| 探针 | 输入 | 读数 |
|---|---|---|
| A | `tool_config` = **内层** `TorchMemoryToolConfig` | 构造 OK；`_impl=TorchMemoryProfiler`；**`_oom_observer_attached=True`**；`check_enable()=True` |
| B | `tool_config` = **容器 dict** `{torch_memory: ...}` | `AttributeError: 'dict' object has no attribute 'trace_alloc_max_entries'`（与 M1 静态读法逐字一致） |
| C | `enable=False` | `check_enable()=False`；`start()` 返回 `None`；`impl.this_step=False` ⇒ 真 no-op |

**A 反过来说明 B 不是默认路径**：真跑时 worker 侧显式取内层（`engine_workers.py` 的 `TrainingWorker.__init__` 与 `ActorRolloutRefWorker.__init__`），所以 `pda/en/s2` 都能跑通；B 只在"把整个容器下传"的调用点才会炸。

## msprobe 一侧

- venv **没有** msprobe；本轮 `pip install --no-deps --target /tmp/m5-msprobe-target mindstudio-probe`（26.2.0, cp310 wheel），再补 `matplotlib 3.10.9` + `pyparsing/cycler/kiwisolver/fonttools/python-dateutil/packaging/pillow` 才能 `from msprobe.pytorch import PrecisionDebugger`（**共用 venv 一个字节没动**，靠 `PYTHONPATH`）。
- `import msprobe` 后在空 cwd 列目录：**0 项** ⇒ "msprobe 的 import 期会自行落盘"这条假设否掉。
- `actor_update` 臂有 4 处真实 dump 文件；`critic_update` 臂连目录都没建，且**没有** `Unknown precision_debugger stage` 之类的 warning —— 与 M1 的"前端接受、后端无触发点、`strict=False` 下连 warn 都没有"完全一致。

## 本轮自己踩的坑（都已修）

1. **`total_storage_size` 默认 null**：第一次 `en` 起跑即 `TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`（`transfer_queue/.../simple_storage_bootstrap.py:45`）。补 `++transfer_queue.backend.SimpleStorage.total_storage_size=100000` 后正常；与本机 venv 的 TransferQueue 版本错配同源，**不是上游缺陷**（失败日志留 `raw/run-en-fail1.txt`）。
2. **`python /path/script.py` 的 `sys.path[0]` 是脚本目录，不是 cwd**：第一版探针因此导入了 venv 里装的另一棵树（那里没有 `_oom_observer_attached`），把"构造成功"读成了 `AttributeError`。改成 `PYTHONPATH=<tree>` 并打印 `verl.__file__` 后读数才自洽。⇒ 以后凡"钉某一棵树"的探针，必须显式写 PYTHONPATH 并把 `verl.__file__` 打进结果。
3. **`ARM=s2` 忘了它同时控制 `enable=True`**，且脚本会 `rm -rf prof-$ARM`：第一次 s2 启动会覆盖 `prof-en` 证据、而且根本不会 enable。已 kill 重做——s2 用独立脚本、`PROF=prof-s2`、`ARM` 仍为 `en`。

## 边界

- 单 seed、单卡、2 步、0.5B/gsm8k/`max_response=128`；**不是性能测量**，不报耗时结论（pda 整跑 3m13s，含启动）。
- "OOM 时真的会落 snapshot" **未验**：只验证回调已挂上，没有人为制造 OOM。
- NPU 侧同理未测（本轮昇腾仍不可达）。
- `[torch_memory]` 日志行缺失的原因未定位（worker logger 级别/去向），本轮只以进程内对象状态为准。
- `pda` 的 dump 是 msprobe L1 statistics；**内容正确性**（统计值是否真对）不在本轮范围。

## 与 M1 第五段的关系

M1 的三层读数（入口面消失／新链路通／官方示例 enable=False）里，"通"与"示例不写"两条在本轮从 CPU 探针升级为真机读数；M1 标为"需 GPU"的两项（快照真落盘、OOM 观察者）本轮关闭。**新增一条 M1 没提的**：`_oom_observer_attached` 与驱动日志不一致，别拿日志当证据。

## 产物

- 脚本：[`run-torchmem.sh`](run-torchmem.sh)（en/ex）· `raw/run-torchmem-s2.sh`（s2）· [`run-precision-debugger.sh`](run-precision-debugger.sh)（pda/pdc）· [`probe-profiler-path.py`](probe-profiler-path.py) · [`verify-snapshot.py`](verify-snapshot.py)
- 原始日志：`raw/` 中 **15 份与服务端逐文件 md5 一致**（en/ex/fail1 + 两个 nohup + 探针 JSON 共 6；pda/pdc + 两个 nohup + msprobe config + `run-pdbg.sh` 共 6；s2 的日志/nohup/脚本共 3），清单见 `raw/server-md5*.txt` 与对应的 `local-md5*.txt`；本地另有 3 份派生产物（`verify-snapshot.json`、`pda-dump-tree.txt`、md5 清单本身）。归档时 `.log` 统一改名 `.txt`（仓库 `.gitignore` 吞 `*.log`），内容逐字节未变
- 读数：[`raw/probe-profiler-path.json`](raw/probe-profiler-path.json) · [`raw/verify-snapshot.json`](raw/verify-snapshot.json) · [`raw/pda-dump-tree.txt`](raw/pda-dump-tree.txt)
