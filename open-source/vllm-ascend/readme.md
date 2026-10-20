# vLLM-Ascend 开源贡献

<!-- contribution-summary:start -->
**【仓库】** [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)（昇腾推理适配项目）

**【贡献点】** vLLM-Ascend 开源贡献。[项目](https://github.com/vllm-project/vllm-ascend)

**【类型】** 项目总览

**【价值评估】** 7 / 10 — 集中同一项目的候选、证据和进展，减少误读与重复探索；目录本身不计作上游成果。

**【进展】** 保留 2026 年 6 月草稿，不作为当前 NPU 实测或已发布贡献。

**【大白话讲明白】**

这里整理我们在 vLLM-Ascend 上围绕MoE 数值、设备通信和休眠唤醒做过的调查、方案和跟进。先看每份文档的进展和大白话说明，再决定是否深入代码与实验。

**【What–Why–How】**

- **What：** vLLM-Ascend 的贡献材料入口与当前关注范围。
- **Why：** 同一问题会经历草稿、验证、撤回与发布，文件名本身不能说明当前可用性。
- **How：** 通过索引进入规范化文档，沿证据与关联讨论回看；技术原理学习另见 notebook/frameworks。
<!-- contribution-summary:end -->

保存贡献候选、方案、验证及跟进材料。以下是历史积累，草稿不代表已提交，原文状态不代表当前上游状态；继续推进前先核对关联讨论和验证证据。

返回 [贡献总览](../README.md) · [框架学习](../../notebook/frameworks/readme.md#vllm-ascend)。

## 材料索引

- [vLLM-Ascend #10579: MoE NaN — Review Comment Draft (UPDATED Session 3)](vllm-ascend-10579-moe-nan-comment-draft.md)
- [vLLM-Ascend #10592: NPUIPC Security Bugs — Comment Draft](vllm-ascend-10592-npuipc-security-comment-draft.md)
- [vLLM-Ascend #10684: DSA Hadamard Sleep/Wake — Comment Draft](vllm-ascend-10684-dsa-hadamard-comment-draft.md)
