# verl #6067 的第三个体检信号族：被删掉的启动配置键，还有谁在传

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架）

**【贡献点】** 同一次重构（[#6067](https://github.com/verl-project/verl/pull/6067)，`044bbba2`）的第三次体检。前两次查的是**被整份删除的文件**（配置消费点、接口名），这一批查**被改动的启动入口**：`verl/trainer/config/ppo_trainer.yaml` 里 `trainer.use_legacy_worker_impl` 被整条删掉，而命令行还在传它的脚本**一行都没改**。全仓 4 处引用：**1 处没加 `+` ⇒ Hydra 合并配置时直接报错**（`tests/special_e2e/run_fully_async_policy_genrm.sh:177`），**2 处加了 `+` ⇒ 不报错但值没人读**（`opd.sh:240`、`run_qwen3_30b_veomni.sh:102`），**1 处在 Ascend 参数表里仍列为可配**。

**【类型】** 问题诊断（B 类：主动发掘）

**【价值评估】** 3 / 10 — 比前两批都弱。**会报错的那处在当前 CI 触发条件下不会阻碍任何东西**（fully-async 的 genrm 脚本不在 `.github/workflows` 的 paths/命令里），其余三处是"文档／示例残留"，不影响训练结果。真正的价值是**方法**：这类"删了配置键、调用点没跟上"的问题，前两批的信号族都扫不到（配置消费点扫的是读取点、接口面扫的是 `def`），只有把**两次提交的 yaml 键空间做差集**才看得见。

**【进展】** 2026-11-04 M1 本地完成。静态扫描 + **真实 Hydra 1.3.8 四形状对照**（不是机制推断）。**未触上游**、未建 issue、未评论、未开 PR。复现与归档见 [实验目录](../../../../experiments/verl-6067-shell-config-scan/readme.md)。

**【大白话讲明白】**

verl 原本有一个开关，用来选"用旧代码跑还是用新引擎跑"。四月那次重构把旧代码整份删了，这个开关也就没了意义——顺手把开关本身也删掉了。但有几份启动脚本还照老习惯传它，于是分成两种命运：

- **加了加号前缀的**（`+trainer.use_legacy_worker_impl=disable`）：配置库把它当新字段收下，程序照跑。**不报错，但这个字段再没有任何代码去读**——等于传了个没用但无害的东西。
- **没加加号的**：配置库在合并那一刻就拒绝，直接报 `Could not override 'trainer.use_legacy_worker_impl'.`，脚本连启动都到不了。

这条经验适用于所有"删了配置项"的重构：**删配置项不是改代码，而是改所有会在命令行里写它的地方**，而后者通常没人负责。

**【What–Why–How】**

- **What：** #6067 从 `verl/trainer/config/ppo_trainer.yaml` 删掉了 `trainer.use_legacy_worker_impl`（94 → 93 个叶子键，唯一一个被删的）。当前 main 上全仓还有 4 处引用：1 处 unprefixed（会失败）、2 处 `+` 前缀（不失败但不生效）、1 处文档参数表。
- **Why：** 前两批的信号族都漏这一类。配置消费点问"字段还在、有没有人读"，答"字段已经不在了"就结束了；接口面查的是 `def`，而 Hydra 键不是 `def`。**删掉的键的消费者是命令行和文档**，必须按"两次提交的键空间差集"去找。
- **How：** ① `scan_shell_refs.py` 用缩进栈解析 `044bbba2^` 与 `044bbba2` 两份 yaml 的叶子键全路径，取差集，再全仓定位引用点并分档（`INVOKE` / `DOC_TABLE_CELL` / 已标 removed 的豁免）；② 对每条 `INVOKE` 解析它落在哪个 hydra 入口（`@hydra.main` 的 `config_path`/`config_name` 或调用脚本的 `--config-name`），沿 `defaults:` 展开判断这条链是 struct 模式还是 open dict，再结合有没有 `+` 判定"会失败"还是"不报错"；③ `verify_findings.py` 4 条判定写死 + 3 条结构性自证，`--selftest` 三种扰动全部要被捕捉；④ `hydra_repro.py` 用真实 Hydra 复刻 A/B/C/D 四种形状，**证明"会失败"的那一档不是推测**。**边界**：只查一个键、只覆盖 `ppo_trainer.yaml` 一个 boot 配置、未跑 verl 训练；`scripts/megatron_merge_lora.py` 里两行桥接调用因第三方包缺失**未能静态判定**，不计入读数。

<!-- contribution-summary:end -->

## 一、结论

| 信号 | 结果 | 是否算问题 |
|---|---|---|
| #6067 从 `ppo_trainer.yaml` 删除的叶子键 | 1（`trainer.use_legacy_worker_impl`） | — |
| 当前 main 全仓引用点 | 4 | — |
| 其中 **unprefixed、会报错** | **1**（`run_fully_async_policy_genrm.sh:177`） | **调用点遗留**（不在 CI 触发路径上） |
| 其中 **`+` 前缀、不报错也不生效** | 2（`opd.sh:240`、`run_qwen3_30b_veomni.sh:102`） | 示例／测试残留，无害 |
| 文档参数表仍列为可配（默认 `auto`） | 1（Ascend `parameter_and_metrics.md:368`） | **文档欠债** |
| 代码侧读取点 | **0** | 两条 `+` 传进去的值无人消费 |

**必须先说的两条限定**：

1. **"不报错"不等于"生效"**。`+key` 只是让 Hydra 接受这个新字段；`git grep` 在 `verl/**` 里对这个键的计数是 **0**，所以它今天什么都没做。对外只能说"传了个没人读的字段"。
2. **会失败的那一条不影响当前 CI**。`tests/special_e2e/run_fully_async_policy_genrm.sh` 没有被任何 `.github/workflows` 的命令引用（`opd.sh` 被引用，但它带 `+`、不会失败）⇒ 这是"**将来谁手动跑它就会撞墙**"，不是"CI 现在是红的"。

## 二、真实 Hydra 复现（不是机制推断）

最小树复刻三种形状，实测 hydra **1.3.8** / Python 3.12.13：

| 形状 | 树里有这个键 | 传法 | 实测结果 |
|---|---|---|---|
| A（`044bbba2^` 的形状） | 有 | `trainer.use_legacy_worker_impl=disable` | **OK**，值进入配置 |
| B（`044bbba2` 之后的形状） | 无 | 同上（unprefixed） | **FAIL**：`ConfigCompositionException: Could not override 'trainer.use_legacy_worker_impl'.` |
| C（同 B） | 无 | `+trainer.use_legacy_worker_impl=disable` | **OK**，作为新键被接受 |
| D（负向对照） | 无 | `trainer.device=npu`（存在的键） | **OK** ⇒ B 的失败不是"所有 override 都失败" |

C 这一行顺带否掉一条可能的反驳："fully-async 走 `ppo_trainer` 的组合链，也许 open dict 允许新增键"——
同一条链上直接赋值 `cfg.trainer.<新键> = ...` 抛 `ConfigAttributeError`，不成立。

## 三、时间线：这一条是谁留下的

| 时间 | 事件 | 留下的形态 |
|---|---|---|
| 2026-04-20 | #6067 删除配置键；同一次提交把 `examples/router_replay/*.sh` 里的 `USE_LEGACY_WORKER_IMPL` 分支连同传参一起删掉 | 其余调用点未动 ⇒ 进入形状 B |
| 2026-04-27 | fully-async genrm e2e 脚本新增（晚于 #6067，树上已经没有这个键） | 写了 unprefixed ⇒ **形状 B，会失败** |
| 2026-05-06 | fully-async OPD e2e 脚本新增（同样晚于 #6067） | 写了 `+` 前缀 ⇒ 不失败 |
| 2026-05-19 | veomni 示例更新 | 带 `+` ⇒ 不失败 |

⇒ 三条的写法在"树上有没有这个键"这个维度上**恰好与树的状态相反**：写在键还在的时候不需要 `+`，写在键已经没了之后就必须加 `+`。
`genrm` 那条是抄了旧模板；另两条是抄了新模板。**没有一条是 #6067 本身写坏的**，但三条都是"开关死了、引用还活着"。

## 四、顺带查的第二条：`scripts/megatron_merge_lora.py`（未计入读数）

同一次重构把该脚本从 `verl.workers.megatron_workers.ActorRolloutRefWorker` 迁到
`verl.workers.engine_workers.ActorRolloutRefWorker`，新写的两行调用 `engine.bridge.save_weights(...)` /
`engine.bridge.save_hf_weights(...)`。

**为什么不计入**：`MegatronEngine.bridge` 是 `AutoBridge` 实例（或 `None`），这两个方法由**第三方包**提供
（`mbridge` / `megatron.bridge`），本机未安装 ⇒ **无法静态确认方法是否存在**。可用的旁证是两种方向的：

- 引擎自身代码只调 `export_weights` / `export_hf_weights` / `export_adapter_weights`（`verl/workers/engine/megatron/transformer_impl.py:818-826`）；
- 但 `verl/utils/checkpoint/megatron_checkpoint_manager.py:993-1015` 仍用 `inspect.signature` **探测** `bridge.save_weights`，说明至少某些 bridge 版本有这个方法。

⇒ 记为"需真机或装 `mbridge` 才能判定"的候选，不进对外材料。

## 五、为什么这一类问题前两批扫不到

| 信号族 | 问的问题 | 对"删了配置键"的回答 |
|---|---|---|
| 配置消费点（第一批） | `config.x.y` 这个字段还有人读吗 | 字段都不在 yaml 里了，**扫描器根本不会把它列为候选** |
| 接口面（第二批） | 这个 `def` 还有人引用吗 | Hydra 键不是 `def`，**不在扫描器的名字表里** |
| 本批（第三批） | 两次提交的 **yaml 键空间差集**里，谁还在命令行写它 | **命中**（4 处） |

⇒ 判据：**"删除"这一类重构要按被删的东西的形状去扫**——删文件看引用、删字段看读取点、**删配置键看命令行与参数表**。

## 六、建议的最小动作（对外仍需用户逐项授权）

1. `run_fully_async_policy_genrm.sh:177` 删掉那一行（同一行前后都没有 `+`，删它等于恢复"树上没有这个键"的状态）；
2. `opd.sh:240`、`run_qwen3_30b_veomni.sh:102` 一并删（`+` 前缀让它今天无害，但源码里已无人读）；
3. Ascend 参数表那一行删掉或标"已移除"（`docs/workers/engine_workers.rst:221` 已经把同一件事标成 `(removed)`，是仓库自己的写法先例）。

三条都属文档／示例修订，不涉及架构决策。**未获用户授权前不做任何对外动作**。
