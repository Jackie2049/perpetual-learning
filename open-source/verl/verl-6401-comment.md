# verl 共享前缀评论草稿已撤回

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 撤回本项目面向 [Issue #6401](https://github.com/verl-project/verl/issues/6401) 的旧评论草稿；[PR #4368](https://github.com/verl-project/verl/pull/4368) 的实际输入合并与原文判断不符。[更正证据](../../notebook/frameworks/verl-prefix-grouper-gap-analysis.md)

**【类型】** 草稿撤回与论证纠错

**【价值评估】** 1 / 10 — 原提案不再独立推进，保留错误分析经验，不作为贡献收益。

**【进展】** 2026-10-21 withdrawn/not-pursued，不再 READY；未核实本草稿历史上是否发布，本轮没有对上游发布、撤回或编辑任何内容。

**【大白话讲明白】**

原草稿认为旧方案只在注意力里共享前缀，因而遗漏大部分模型计算。实际输入在进入模型前已经合并，相同提示词不会按原样本数量重复进入后面的逐 token 运算。草稿还把两种不同语义的自写微基准当作同一框架的性能差距，收益判断不成立，因此停止推进。

**【What–Why–How】**

- **What：** 撤回本项目旧评论和 full-model sharing RFC，不代表否定或撤回上游讨论。
- **Why：** 原生输入先合并；0.99x不是PG测量，2.46x模拟缺少前缀K/V依赖，不能推出2.5倍缺口。
- **How：** 只读固定源码并做5组原生CPU输入检查，更新关联笔记和状态；不执行模型/GPU或对外操作。
<!-- contribution-summary:end -->

## 不能再发布的论点

旧稿中的“PG不节省MLP”“2.5倍可行动缺口”“训练收益固定乘0.76”均撤回。局部余弦相似不能认证生产训练安全；`is_causal=True` 对不等长Q/K的含义还依赖具体后端和对齐方式，不能泛化为必然错误。

旧英文可发布正文已由本记录替换，历史版本可在 Git 中恢复；不得再复制为新评论。现有源码分析与验证范围以[架构纠错](../../notebook/frameworks/verl-prefix-grouper-gap-analysis.md)和[撤回的RFC](../../notebook/frameworks/verl-6401-rfc-full-model-ps.md)为准。已有作者的 [PR #8300](https://github.com/verl-project/verl/pull/8300) 只作学习参考，不是我方成果。
