# precision_debugger 的 stage 白名单 vs 实际触发者（实验，纯静态）

**目的**：上一轮的[接口面体检](../../open-source/verl/evidence/engine-refactor-api-surface/readme.md)发现 `compute_values`／`critic_update` 的 `@DistProfiler.annotate(role=...)` 随 #6067 消失。本批把口径从"两个名字"扩到**整张白名单**：`_STAGE_TO_ROLE` 的 6 个 stage 逐个找触发者，并区分"重构丢的"与"从来没接过"。

**结论（详见[证据文档](../../open-source/verl/evidence/precision-debugger-stage-producers/readme.md)）**：

| 树 | 有触发者的 stage | 无触发者 |
|---|---|---|
| main `7454326`（2026-11-04） | 3 个（actor 侧） | **`compute_values`、`critic_update`、`compute_rm_score`** |
| #6067 父提交 `6eeb5711`（2026-04-20） | 5 个 | `compute_rm_score` |

⇒ **两条是重构回归**（critic 的取值与更新，当时 FSDP 与 Megatron 两套 worker 都挂了装饰器），**一条是从来没接过**（reward model）。触发路径只有一条——`@DistProfiler.annotate(role=<字面量>)`：`start_profile(stage=...)` 全仓 0 处，`start_profile(role="train")` 有 2 处但那个 `role` 只是标签（`PrecisionDebuggerProfiler.start()` 首参是 `stage`，且不读 `kwargs["role"]`）。而 6 份配置注释与 Ascend 文档把 6 个 stage 都当作可选值列给用户。

## 文件

| 文件 | 作用 |
|---|---|
| `scan_stage_producers.py` | 解析 `_STAGE_TO_ROLE`，扫全树 `DistProfiler.annotate(role=...)`，按 stage 汇总触发者；`--preset main` / `--preset pre6067` 做判定（exit 0/1），`--selftest` 用合成树验证判据会翻转 |
| `scan-main-7454326.json` | main 的扫描结果（可复现） |
| `scan-pre6067-6eeb5711.json` | #6067 父提交的扫描结果（可复现） |

## 复现

```bash
# 取两棵只读树（不改任何仓库）
git -C <verl 仓库> archive origin/main | tar -x -C /tmp/verl-main-7454326
curl -sL https://codeload.github.com/verl-project/verl/tar.gz/6eeb5711babe178ca3cc3caba7c9f158efa9fcd6 \
  | tar -xz -C /tmp/verl-pre6067 --strip-components=1

python3 scan_stage_producers.py --selftest                                   # 4/4 PASS
python3 scan_stage_producers.py --root /tmp/verl-main-7454326 --preset main  # exit 0
python3 scan_stage_producers.py --root /tmp/verl-pre6067 --preset pre6067    # exit 0
```

本轮环境：macOS arm64，本机 Python 3.9.6（**低于 verl 源码的语法版本**，见下"脚本自身的坑"第③条）。**未跑 verl、未上 GPU/NPU、未触上游**。

## 判据（三条，缺一条结论就不成立）

1. **stage 只能以字面量 `role=` 进入**：`DistProfiler.annotate` 的 `role` 必须走 `**kwargs_outer`，而 `PrecisionDebuggerProfiler.annotate` 只读 `kwargs_outer.get("role")`；`role` 传成位置参数或写成 `message=` 都不会触发。
2. **训练侧的 `start_profile(**kwargs)` 到不了 stage 采集**：`PrecisionDebuggerProfiler.start()` 首参是 `stage`，`role` 落进 `**kwargs` 且函数体不读它；全仓 `start_profile(stage=...)` 0 处。
3. **没有绕开包装器的旁路**：`msprobe` 全仓只出现 2 次（`import_utils.py:75` 的 `find_spec` 探测、`precision_debugger_profile.py:208` 的懒 import），`PrecisionDebugger` 实例只在 `start()` 的 `:215` 创建，dump 目录 `step_{n}/{stage}` 也只在 `start()` 的 `:211-212` 才 `makedirs` ⇒ **不调用 `start()` 就不会有任何目录或文件落地**。

## 机制（为什么不是"补回三行"）

新引擎里 critic 的训练走 `TrainingWorker.train_batch`（`engine_workers.py:377-378`），它**有**装饰器，但 role 是 `train_batch`——不在白名单里，`_should_collect` 会打一条 `Unknown precision_debugger stage: train_batch` 然后返回 False。critic 的取值前向走 `infer_batch`（`:433`），一条装饰器都没有。而 `TrainingWorker` 同时服务 `Role.Critic`（`trainer_base.py:813`）与 SFT（`sft_trainer_ray.py:138`），所以"给 critic 补一个 `role="critic_update"`"需要运行期知道这个 worker 当前是谁——这不是机械恢复，是一次设计选择。

## 判据与边界

- **静态结论只到"没有触发点"**：`start()` 一旦被调用就会建目录、起 msprobe，所以"目录不存在"几乎是必然结果；唯一没覆盖的假设是 **msprobe 在 import 阶段自己落盘**（不合理，但没有实测）。真要证伪，最小的 NPU 动作不是跑 PPO，而是"import msprobe 后跑一步什么都不 start，看 dump 目录是否为空"。
- **不是训练正确性问题**：`compute_values`／`critic_update` 只影响精度采集，不影响 loss、梯度或吞吐。文档里 L0 那组 stage 的耗时数字（`precision_debugger.md:240-241`）在当前 main 上已不可复现（5 个 stage 里 2 个不采集）。
- **没引 zh 文档**：`docs/` 下 `find -name "*_zh*"` 在 `7454326` 上命中 0 条，所以 M1 上一轮引的 `precision_debugger_zh.md:48/85` 在本 ref 上无法核对，本批只引 en 文档。这是引用口径问题，不改任何上一轮的读数。
- **`docs/perf/torch_profiling.md:162` 提到 `critic_update` 行**属另一个 profiler（torch），本轮未判定其好坏；只记录"同一批 annotate 是它唯一的 role 来源"，所以那条文档同样值得复核，留作下一轮。

## 脚本自身的四个坑（都由本轮发现并修掉）

1. **`SkipManager.annotate(role="rollout")` 被算成 profiler 触发者**：第一版匹配所有 `*.annotate(role=...)`，白名单外的 role 报成 4 个（含 `rollout`／`async_rollout`）；收紧成只认 `DistProfiler` 后剩 2 个（`train_batch`／`forward_backward`）。与上一轮 `\btiled_mlp\b` 那次同型：**匹配面一宽，假阳性先来**。
2. **把 `self` 当成 `start()` 的首参**：判定因此报"首参实测 self vs 预期 stage"，差点被读成"触发链成立"。剥掉 `self`/`cls` 后才对。
3. **本机 Python 3.9 解析不了 9 个文件**（源码用了 3.10+ 语法），其中恰好包括 #6067 父提交的 `verl/workers/fsdp_workers.py`——第一版因此**漏掉 fsdp 上的两个 critic 装饰器**，会把"两条回归"错判成"只有 megatron 有"。改成"AST 失败则逐行正则兜底、结果标 `via=regex`"后补回 6 条，且与直接 `grep -n 'annotate(.*role='` 的 6 行逐行一致。
4. **兜底分支的 `src` 跨文件残留**：第一版写 `src if "src" in dir() else ""`，某个文件读失败时会把上一个文件的内容当成它的内容。改成显式初始化 `src = None` 后再兜底。

## 下一步

- 若要对外，最小动作是**文档与配置校验**（把无触发者的三个 stage 标出来，或加载期对"配了但没人触发"的 stage 告警），不是补装饰器。**对外动作全部待用户逐项授权**，本轮未建 issue／未评论／未开 PR。
- 真机侧已登记 [M5 交接](../../experiments/handoff-m5.md)：NPU 对照从"跑 PPO 看 dump"降级为"确认 import 不落盘"的冒烟。
