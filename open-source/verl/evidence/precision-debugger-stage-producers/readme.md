# verl 的 msprobe 集成：6 个"支持的 stage"里有 3 个没有任何触发点

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架）

**【贡献点】** 精度调试器（msprobe）的白名单 `_STAGE_TO_ROLE` 有 6 个 stage，文档与 6 份配置注释都把它们当作可选值；但全仓只有一条路能把 stage 送进采集器（`@DistProfiler.annotate(role=<字面量>)`），而 `compute_values`、`critic_update`、`compute_rm_score` **一个触发者都没有**。对比 [#6067](https://github.com/verl-project/verl/pull/6067) 的父提交：前两个当时在 FSDP 与 Megatron 两套 worker 里都挂着装饰器（**重构回归**），第三个**从来没有过**。用户照文档配 critic 采集，不报错、不告警、也不出 dump。

**【类型】** 问题诊断（B 类：主动发掘）

**【价值评估】** 6 / 10 — 用户可见后果明确（照文档配 `critic_update` 会静默无输出，且 `strict=False` 下连 warning 都没有），但只影响 Ascend 上的精度采集，不影响训练正确性；静态可证到"没有触发点"，NPU 侧只剩"import 是否自己落盘"这一条假设。

**【进展】** 2026-11-04 M5 无卡轮完成，纯静态（verl main `7454326` 与父提交 `6eeb5711` 两棵树互核，脚本 4/4 自检 + 两棵树判定全一致）。**未触上游**：未建 issue、未评论、未开 PR。复现见[实验目录](../../../../experiments/verl-precision-debugger-stages/readme.md)。

**【大白话讲明白】**

verl 有一套给昇腾 NPU 用的"精度调试器"，能让训练停在某几个阶段、把张量数值 dump 出来比对。它支持的阶段名写在一张白名单里，文档和 5 份配置模板也照着这张表告诉用户"你可以选这 6 个"。可实际采集要靠代码里挂的标签，而其中三个标签——"算 critic 的值""更新 critic""算奖励模型的分"——现在的代码里一个都没有。用户按文档配上，程序既不报错也不提示，只是一条数据也不出。

这不是训练坏了，而是**调试工具对文档承诺失约**。而且三个里只有两个是四月份那次 worker→engine 重构弄丢的；"算奖励模型的分"这个名字从加进白名单那天起就没接过。

**【What–Why–How】**

- **What：** `verl/utils/profiler/precision_debugger_profile.py` 的 `_STAGE_TO_ROLE`（`:26`）声明 6 个 stage。当前 main 上只有 actor 侧 3 个有触发者；critic 的 2 个与 reward model 的 1 个没有。
- **Why：** 整条采集链只有一条入口：`@DistProfiler.annotate(role="<stage>")` → `PrecisionDebuggerProfiler.annotate` → `start(stage=...)`。`#6067` 删掉旧 worker 后，critic 的取值／更新改到新引擎的 `TrainingWorker`，其唯一装饰器是 `role="train_batch"`（不在白名单，会被判 unknown 后丢弃），取值前向 `infer_batch` 则完全没有装饰；reward model 走 `RewardLoopManager.compute_rm_score`，历史上也没有装饰。
- **How：** `scan_stage_producers.py` 解析白名单、扫全树装饰器字面量、核对 `start()` 的签名与 msprobe 的唯一构造点，并对 #6067 父提交做同一扫描；`--selftest` 用合成树验证"补上装饰器后无触发者集合变空"等四条判据会翻转。**边界**：纯静态、未跑 verl、未上 NPU；"msprobe 是否在 import 阶段自行落盘"未实测，是唯一残留假设。

<!-- contribution-summary:end -->

## 一、结论

在 main `7454326`（2026-11-04 fetch，提交时间 10:07 CST）上逐 stage 核对：

| stage | 白名单 role | 当前 main 的触发者 | #6067 父提交 `6eeb5711` | 定性 |
|---|---|---|---|---|
| `actor_update` | actor | `engine_workers.py:768` | fsdp:1028 / megatron:777 / engine:646 | 在 |
| `actor_compute_log_prob` | actor | `engine_workers.py:758` | fsdp:1124 / megatron:899 / engine:638 | 在 |
| `ref_compute_log_prob` | ref | `engine_workers.py:751` | fsdp:1176 / megatron:873 / engine:631 | 在 |
| `compute_values` | critic | **无** | **fsdp:1677 / megatron:1249** | **重构回归** |
| `critic_update` | critic | **无** | **fsdp:1697 / megatron:1266** | **重构回归** |
| `compute_rm_score` | reward_model | **无** | **无** | **从来没接过** |

这张表把上一轮的两条读数各推进一格：① critic 两个 stage 的消失确实是 #6067 造成的（上一轮只说"engine 侧 0 处"，本批在父提交上找到了对应装饰器）；② 白名单里还有第三个死名字 `compute_rm_score`，它**不是**回归——父提交的 FSDP／Megatron worker 里都没有它，全仓只有白名单、配置注释和文档在提它。

## 二、触发路径只有一条，且已逐条排除旁路

| 可能的入口 | 实际情况 |
|---|---|
| `@DistProfiler.annotate(role="<stage>")` | **唯一**入口（`profile.py:334-380` → `impl.annotate` → `start(stage=role)`） |
| `start_profile(stage=...)` | 全仓 **0 处**；`start_profile(role="train", …)` 有 2 处（`ray_trainer.py:1132`、`trainer_base.py:1430`），但 `PrecisionDebuggerProfiler.start()` 首参是 `stage`，`role` 落进 `**kwargs` 且函数体不读它 |
| 直接构造 msprobe | `PrecisionDebugger` 只在 `precision_debugger_profile.py:215` 构造，位置在 `start()` 内部 |
| import 副作用 | `msprobe` 全仓仅 2 处：`import_utils.py:75` 的 `find_spec` 探测、`precision_debugger_profile.py:208` 的懒 import |
| 其它包装器 | `SkipManager.annotate(role="rollout"/"async_rollout")` 是另一套机制，不进入本白名单（第一版脚本把它们误算成触发者，已分离） |

而且 dump 目录本身也是在 `start()` 里创建的（`:211-212` 的 `os.makedirs(save_path/step_{n}/{stage})`）。**不调用 `start()`，磁盘上不会出现任何东西**——这让"NPU 上看一眼有没有目录"成为一个干净的判据。

## 三、为什么这不是"补回三行"

新引擎里 critic 的训练走 `TrainingWorker.train_batch`（`engine_workers.py:377-378`），**是有装饰器的**，但名字叫 `train_batch`：

- `train_batch` 不在 `_STAGE_TO_ROLE` 里 ⇒ `_should_collect` 打一条 `Unknown precision_debugger stage: train_batch` 后返回 False（`precision_debugger_profile.py:173-181`）。**这条 warning 与用户配置无关**——即使只配 `stages: [actor_update]`，只要跑到带 critic 的训练步就会打。
- critic 的取值前向走 `infer_batch`（`:433`），没有装饰器，`compute_values` 彻底没人调。
- `TrainingWorker` 同时服务 `Role.Critic`（`trainer_base.py:813`）与 SFT（`sft_trainer_ray.py:138`），所以"给它挂一个 `role="critic_update"`"要求运行期知道这个 worker 当前扮演谁——**是新机制，不是恢复旧代码**。

`compute_rm_score` 同理：奖励走 `RewardLoopManager.compute_rm_score`（`reward_loop.py:343`）与 `worker.compute_score_batch` RPC，两者在父提交上也没有装饰器，属"白名单先写、接线从未做"。

## 四、文档与配置里的同一张表

文档把 6 个 stage 全部列为可选，并给了"actor/ref/critic 常用组合"：

- `docs/ascend_tutorial/en/dev_guide/precision_analysis/precision_debugger.md`：`## Supported stages`（`:78`）的 bullet 列表 `:81-87`；yaml 示例 `:48-50`；"most common useful combinations" `:94-98`。
- 同一文档 `:240-241` 用 L0 采集那组 stage（含 `compute_values`、`critic_update`）报过 3–4× 的开销——**该组合在当前 main 上不可复现**（5 个 stage 里 2 个不采集），表里的数字本身不作废，但它对应的采集面已经不是文档描述的那一组。
- 配置注释 6 处（`verl/utils/profiler/config.py:205`、`actor.yaml:316`、`critic.yaml:225`、`ppo_trainer.yaml:451`、`ref.yaml:164`、`rollout.yaml:460`）都把 `compute_values, critic_update, compute_rm_score` 并列写出。

`strict` 救不了这件事：三个名字都在白名单里，`_normalize_stages` 不会把它们算成 unknown，`strict=True` 也不会报错。

## 五、边界与残留

- **纯静态**：未跑 verl、未上 NPU/GPU、未装 msprobe。结论止于"没有触发点"，不写"一定采不到"——唯一未覆盖的假设是 msprobe 在 import 阶段自己落盘。若要做真机确认，最小动作是"import msprobe、不调用 start、跑一步、看 dump 目录为空"，不需要为它跑一次带 critic 的 PPO。
- **不影响训练**：这三个 stage 只影响精度采集，不影响 loss／梯度／吞吐。
- **未引中文文档**：`docs/` 下 `find -name "*_zh*"` 在 `7454326` 上 0 命中，上一轮引用的 `precision_debugger_zh.md:48/85` 在本 ref 上无法核对，本文只引 en 文档。
- **未判 torch profiler**：`docs/perf/torch_profiling.md:162` 提到 `critic_update` 行，而 torch 工具的 role 来自同一批 annotate，同样值得复核；本轮不判定，留作下一轮。

## 六、复现

```bash
python3 scan_stage_producers.py --selftest                                   # 4/4 PASS
python3 scan_stage_producers.py --root /tmp/verl-main-7454326 --preset main  # exit 0
python3 scan_stage_producers.py --root /tmp/verl-pre6067 --preset pre6067    # exit 0
```

两棵树的原始输出在 `scan-main-7454326.json` / `scan-pre6067-6eeb5711.json`。本轮三次自我订正（SkipManager 误算、`self` 当首参、Python 3.9 解析失败导致漏掉 fsdp 的两个装饰器）写在[实验 readme](../../../../experiments/verl-precision-debugger-stages/readme.md#脚本自身的四个坑都由本轮发现并修掉)。

## 七、若要对外（待用户逐项授权）

1. **最省的做法是文档 + 加载期校验**：标注三个无触发者的 stage，或在前端对"配了但没有触发点"的 stage 打唯一一次 warning（不会像 `train_batch` 那样每步刷）。
2. 补装饰器要动引擎层的 role 身份，属设计选择，不宜作为一条"文档修复"捎带。
3. **任何上游可见动作都需用户对这一具体动作明确同意**；本轮全部未做。
