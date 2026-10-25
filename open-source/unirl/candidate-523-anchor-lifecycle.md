# UniRL：UnifiedModelTrainStack 跳过 eval-mode anchor freeze 与 prepare_part（#523）

<!-- contribution-summary:start -->
**【仓库】** [UniRL](https://github.com/Tencent-Hunyuan/UniRL)（腾讯统一多模态 RL）

**【贡献点】** [Issue #523](https://github.com/Tencent-Hunyuan/UniRL/issues/523)：`UnifiedModelTrainStack.train_track` 未复刻 `TrainStack` 的 anchor-freeze 生命周期（缺 `eval()` / `prepare_part` / `train()`），是 cross-stack drift。

**【类型】** 漏洞修复（跨训练栈一致性）

**【价值评估】** 7 / 10 — 明确的源码契约缺口 + 当前可做窗口（前置 #406/#495 已 merged）+ 纯 CPU 可验证结构修复；但 shipped 配方当前不受影响、完整验收需 GPU 对照 replay-sourced anchor。

**【进展】** 2026-10-22 贡献扫描：确认 main `6fd82cba` 上 `UnifiedModelTrainStack.train_track` 只调 `prepare_segment_anchors`、缺 eval/freeze/prepare_part/train，而 `TrainStack._prepare_for_training`（single.py:272-277）四步齐全；基类 `prepare_part` 默认 no-op，PPO/DiffusionOPD override。issue 无 assignee、无对应 PR；前置 #406/#495 已 merged（排序前提已满足）。**2026-10-22 修复已写 + CPU 验证通过 + 已推 fork**：在 `train_track` 把两处 `prepare_segment_anchors` 包进 `eval → freeze(ar,image)+prepare_part → train` 生命周期（复刻 single.py，finally 恢复 train）。CPU 逻辑单测 3 项全过（eval/freeze/prepare/train 顺序、prepare 抛异常仍恢复 train、no-op prepare_part 兼容 shipped 配方）+ 真实文件 AST 断言通过。patch 存 [523-lifecycle-fix.patch](523-lifecycle-fix.patch)。**已推送 fork 分支 [fix-523-anchor-lifecycle @ 04101ad5](https://github.com/Jackie2049/UniRL/tree/fix-523-anchor-lifecycle)**（base 上游 main `6fd82cba`，仅 unified.py +8/-2；因 git 协议不通改经 GitHub REST API 构造 blob→树→提交→分支，内容与本地验证版逐字节一致）。**PR 已建在用户 fork 上供审阅：[Jackie2049/UniRL#1](https://github.com/Jackie2049/UniRL/pull/1)**（base=fork main `6fd82cba`，head=fix-523-anchor-lifecycle `04101ad5`，1 文件 +8/-2；PR 描述含问题/修复/行为边界/验证与 replay 验收缺口，可直接复用为上游 PR 描述）。**2026-10-25 用户授权后上游 PR 已开：[Tencent-Hunyuan/UniRL#629](https://github.com/Tencent-Hunyuan/UniRL/pull/629)**（正文按上游模板重写：Fixes #523、L0 四断言 + pristine 对照入 Test Plan、e2e 因载体配方硬件门槛如实记 Not run 并说明 replay 路径未被 shipped 算法使用；三项检查 SUCCESS、MERGEABLE；Test Plan 按用户反馈精简后**同日关闭**，fork 分支/PR 保留可重开）。主仓库（Tencent-Hunyuan/UniRL）仍未开 issue/PR——向上游提交需用户明确授权。**2026-10-23 用户审阅并认可方案设计与代码实现**；测试验证与 e2e 训练冒烟已作为**正式任务指派 M5 闭环**（[handoff 专设小节](../../experiments/handoff-m5.md)）：L0 真实模块生命周期冒烟（CPU 可做，含补丁三重字节校验）+ L1 e2e 可行性判定、正向回归与 replay 数值验收设计；结构修复本身不需 GPU。**✅ M5 已完成（2026-10-23）**：L0 真实模块冒烟四项断言全过（真实模块 + 真实 anchor.py + pristine 对照），L1 判定「待资源/待权重」（缺 ≥8 卡整段占用 + 未授权权重；**2026-10-25 修正：两载体配方在本机物理不可行，恢复条件实为 8×96GB 级/32 设备多机**，见 [handoff](../../experiments/handoff-m5.md)）；补丁文件已修正为 fork 一致版（原补丁多 3 行注释、与用户已认可的 fork `04101ad5` 不符，修正后指纹核验通过），详见 [实验材料](../../experiments/unirl-523-l0/readme.md)。

**【大白话讲明白】**

UniRL 有两条训练栈：单模型栈每次训练前会把模型切到"推理模式"、冻结旧策略、再切回训练；但统一的"共享主干多算法"栈漏了这套切换，也没调用算法的准备钩子。现在用的配方恰好不触发问题，但一旦搭配 PPO/DiffusionOPD 或 replay-sourced anchor，就会静默产生错误。结构修复可纯 CPU 验证。

**【What–Why–How】**

- **What：** `UnifiedModelTrainStack.train_track` 与 `TrainStack._prepare_for_training` 的 anchor-freeze 生命周期不一致。
- **Why：** unified 栈缺 `eval()`/`prepare_part`/`train()`，replay-sourced anchor 与未来 PPO/DiffusionOPD 会静默出错。
- **How：** 对照 single.py:272-277 与 unified.py train_track；基类 prepare_part no-op、PPO/DiffusionOPD override；前置 #406/#495 已 merged，可 CPU 做结构修复。
<!-- contribution-summary:end -->

返回 [UniRL 贡献入口](readme.md) · [tracker](tracker.md) · [框架学习](../../notebook/frameworks/unirl-survey.md)

## 问题（What）

`TrainStack._prepare_for_training`（[single.py:272-277](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/single.py#L272)）在每次训练前走完整生命周期：

1. `fsdp_backend.model.eval()`
2. `prepare_segment_anchors(...)` — 冻结 π_old 锚
3. `part = algorithm.prepare_part(part)`
4. `fsdp_backend.model.train()`

而 `UnifiedModelTrainStack.train_track`（[unified.py](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/unified.py)）**只做了第 2 步**（对 ar/image 两个算法各调一次 `prepare_segment_anchors`），**从不**：
- 先把共享 backbone 切到 `eval()` 再冻结锚；
- 调用 `algorithm.prepare_part`；
- 结束后切回 `train()`。

## 为什么值得做（Why）

- **当前 shipped 配方不受影响**：所有 `examples/unified_model/*.yaml` 用 GRPO/FlowGRPO/BagelFlowUniGRPO，`old_logp_source: rollout` → `recomputes_anchor` 为 False，`prepare_segment` 不跑 model forward，且这三者都不 override `prepare_part`。
- **但存在真实契约缺口**：
  - `old_logp_source: replay` 时 π_old 在 `train()` 模式计算，正确性依赖各 stage 自己管模式（Bagel/HunyuanImage3 各自处理，无 stack 级保证）——两栈在不同模式下冻结 π_old；
  - 未来 unified stack 搭配 **PPO**（ppo.py:165 override `prepare_part`，`recomputes_anchor=True`）或 **DiffusionOPD**（diffusionopd.py:116 override）会**静默跳过 `prepare_part` 钩子**，无任何报错。

这是 issue #439 同类 cross-stack drift；#495 只共享 per-micro stitch（`prepare_segment_anchors`），未共享 lifecycle。

## 源码证据链（固定 main `6fd82cba`）

| 位置 | 行为 |
|---|---|
| single.py:272-277 `_prepare_for_training` | eval → prepare_segment_anchors → prepare_part → train（四步齐全） |
| unified.py `train_track` | 仅两次 `prepare_segment_anchors`，无 eval/prepare_part/train |
| anchor.py `prepare_segment_anchors` | 只冻结声明的锚，不切模式、不调 prepare_part |
| base.py:369 `prepare_part` 默认 | no-op；`recomputes_anchor` 默认 False |
| ppo.py:87,165 | `recomputes_anchor=True`，override `prepare_part` |
| diffusionopd.py:116 | override `prepare_part` |
| grpo.py / flowgrpo.py / bagel_flow_unigrpo.py | 不 override `prepare_part` |

## 建议修复方向（issue 已给出）

给 anchor-freeze 生命周期一个单一 owner，两栈共用。例如在 `prepare_segment_anchors` 旁加 helper，运行 eval → freeze → `prepare_part` → train；unified stack 在包住两个 part 的锚冻结时切共享模型模式一次，而不是每个算法一次。

## 验收标准

- 两栈在相同 model mode 下冻结锚，且都调用 `algorithm.prepare_part`。
- 所有 shipped unified recipes（rollout-sourced anchors）行为完全不变。
- `old_logp_source: replay` 的 unified recipes 需在至少一个 HunyuanImage3/Bagel 配方上做 live 对照（first-update ratio、clip fraction、loss curve）——这部分需 GPU。

## 验证范围（M1 当前可做 / 需 GPU）

- **纯 CPU 可验证**：结构修复本身——写 CPU 小模型 harness，对照 unified stack 修复前后是否调用 `prepare_part`、`model.eval()/train()`、以及在 PPO/DiffusionOPD 下不再跳过钩子。
- **需 GPU（M5/后续）**：replay-sourced anchor 的 live comparison、真实训练 loss 对照——**2026-10-23 已作为正式任务指派 M5**（[handoff 专设小节](../../experiments/handoff-m5.md)：L1 e2e 可行性判定（shipped 统一配方需 ≥8 设备/32 设备 + 大模型权重，先判资源）→ 正向回归（打补丁 vs 不打补丁 rollout-sourced 逐指标一致）→ replay 数值验收（视统一算法对 recomputes_anchor 的支持而定）。

## 前置与风险

- issue 要求排在 #406、#495 之后——**两者均已在 2026-09-27 MERGED**，前提已满足。
- 无 assignee、无对应 PR；作贡献前应在 issue 留言（该项目外部 PR 合入率高、接受社区贡献）。
- 完整 live 验收需 GPU；结构修复可先行并单独提交。
