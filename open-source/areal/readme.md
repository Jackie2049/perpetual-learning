# AReaL 贡献轨道

> 上游：[areal-project/AReaL](https://github.com/areal-project/AReaL)（蚂蚁系，2026-10 核验仓库名；2026-10-20 快照 5.8k stars、135 open issues）
> 定位：用户方向偏好的第二条 RL 训练框架轨道（verl 已为主贡献轨道）。2026-10-20 活跃度调研中 AReaL 合并效率（91%）为国内六家最高。
> 状态与候选追踪见 [tracker.md](tracker.md)。源码/原理学习见 [notebook/frameworks/](../../notebook/frameworks/readme.md)。

**负责人（2026-10-20 用户明确）**：prompt_mean 候选/PR 的思路、代码、GPU 实验、定性与发布全部由 macbook-M5 Agent 统一负责。M1 已阶段收尾，后续不并行推进此项；统一入口见 [交接说明](handoff-m5.md)。

## 贡献规范（community/CONTRIBUTING.md）

- 需 **DCO 签名**：`git commit -s`（Developer Certificate of Origin）。
- 提交前安装 pre-commit（format/lint/commit message 检查），PR 标题 `type(scope): description`（如 `feat(engine)`、`fix(trainer)`）。
- 有 `good first issue`、`help wanted` 标签可作为入口信号（截至 2026-10-20 无 open 的 good-first-issue；help-wanted 仅 1 个大型 feature #1110）。
- 对外发布与提交按 [AGENTS.md](../../AGENTS.md#执行边界) 执行（需用户授权）。

## 当前结论（2026-10-20 扫描）

近期可行动 bug 均被 **reporter 本人同日 PR** 覆盖（#1786→#1787、#1730→#1732），与 verl 的「reporter 深挖≈reporter 已开 PR」规律一致。**当前无未认领、可行动的贡献空间**；`#1730` 与 verl `#7913` 同属 microbatch 不变性 bug 类，且同 reporter，我方 verl geo_mean 经验可复用作评论/review 依据。详见 [tracker.md](tracker.md)。
