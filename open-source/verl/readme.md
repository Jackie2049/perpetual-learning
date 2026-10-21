# verl 开源贡献

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** verl 开源贡献。[项目](https://github.com/verl-project/verl)

**【类型】** 项目总览

**【价值评估】** 7 / 10 — 集中同一项目的候选、证据和进展，减少误读与重复探索；目录本身不计作上游成果。

**【进展】** [HDO修复PR #8286](https://github.com/verl-project/verl/pull/8286)已合并；2026-10-21复核纠正#8250/GSPO/GMPO及#7913旧验证声明，clip旧RFC不再推进，#7913已有同核心方案不重复提交；其余按证据与用户决定推进。

**【大白话讲明白】**

这里整理我们在 verl 上围绕优化器卸载、训练损失、提示词处理与权重同步做过的调查、方案和跟进。先看每份文档的进展和大白话说明，再决定是否深入代码与实验。

**【What–Why–How】**

- **What：** verl 的贡献材料入口与当前关注范围。
- **Why：** 同一问题会经历草稿、验证、撤回与发布，文件名本身不能说明当前可用性。
- **How：** 通过索引进入规范化文档，沿证据与关联讨论回看；技术原理学习另见 notebook/frameworks。
<!-- contribution-summary:end -->

保存贡献候选、方案、验证及跟进材料。以下是历史积累，草稿不代表已提交，原文状态不代表当前上游状态；继续推进前先核对关联讨论和验证证据。

返回 [贡献总览](../README.md) · [框架学习](../../notebook/frameworks/readme.md#verl)。

## 材料索引

- [#8250诊断边界与待补观测](candidate-8250-async-entropy-collapse.md)
- [#7913真实调用链与旧split验证更正](candidate-7913-gmpo-geo-mean.md)
- [clip旧RFC撤回与目标核验](verl-clip-semantics-rfc-draft.md)
- [#8281截断契约与#8313最小长度反例](candidate-8281-agent-loop-shared-cap-vs-truncation.md) — PR已存在；middle/cap=1原方法违例已记录，未改PR/补丁，修正待负责人决定
- [Candidate: del_local_ckpt_after_load 不删本地 checkpoint](candidate-del-local-ckpt.md)
- [Candidate: dynamic-cp ceil 切分致高 DP rank 空 (#6786)](candidate-dynamic-cp-split.md)
- [Candidate: filter_overlong_prompts 多进程 SIGTERM 挂死 (#7163)](candidate-filter-sigterm.md)
- [Candidate: tool_parser.py 静默吞异常](candidate-tool-parser-silent-except.md)
- [Candidate: process_validation_metrics 崩在 sparse reward_extra_info 的 None](candidate-validation-metrics-none.md)
- [Candidate: 训练前 weight-update 自检 RFC（#8252）](candidate-8252-weight-checker-rfc.md)
- [Candidate: veomni transformer `del_local_after_load` 类型标注错误](candidate-veomni-type-annotation.md)
- [verl 扫描报告 — 2026-08-03 18:14](scan-0803-1814.md)
- [verl + 跨仓库扫描报告 — 2026-08-04 10:16](scan-0804-1016.md)
- [verl 贡献追踪](tracker.md)
- [verl #6401 Technical Comment Draft](verl-6401-comment.md)
- [verl #6401 Prefix-Tree Shared Attention 贡献分析](verl-prefix-tree-contribution.md)
