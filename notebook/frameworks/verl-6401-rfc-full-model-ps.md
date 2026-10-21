# verl Full Model Prefix Sharing 旧 RFC 已撤回

2026-10-21 撤回本项目这份草稿，不代表撤回或评价上游 #6401。不要据此发布 issue、评论、PR，或继续执行旧三阶段方案。它基于错误的 PrefixGrouper 架构判断和不等价的微基准比较；完整证据见[架构与性能论证更正](verl-prefix-grouper-gap-analysis.md)。

## 撤回原因

旧稿称 PrefixGrouper 只在 attention 内省计算，MLP/QKV/norm 仍重复处理每样本的 prompt；实际旧集成在进入模型前已合并输入。attention 是专用可见域的接入位置，不是唯一受益的算子。

0.99 倍来自自写 attention 微基准，不是原生 PG。2.46 倍来自没有注入 prefix KV 的 suffix-only MiniTransformer 模拟，不能证明等价训练的性能。两者不能相除得到 2.5 倍框架缺口，旧 1.87 至 4.56 倍训练收益也不再成立。

## 不再沿用的结论

- 不再提出“给当前 PG 增加 MLP skip”作为已确认缺口或 quick win。
- 不再以局部梯度余弦相似或某个 toy 配置推出任意 GRPO/PPO、LoRA、checkpointing、FSDP 或 hybrid 模型安全等价。
- 不再将简化 hybrid 投影模拟称为原生 DeltaNet recurrent state、卷积上下文和梯度验证。
- 不再把 serving rollout 引擎的位置混同为 FSDP 训练前向所在位置。
- 不再用固定 `forward_speedup×0.76` 预测训练收益；应分解实际 forward、backward、optimizer 和新增开销。

## 可继续使用的学习入口

共享计算的基本思想仍有价值，但布局、位置、注意力可见域、首 response 读出、梯度归集与损失分母必须一同保持。旧库通过 grouped input 和 prefix-last output 复用，新 FSDP [PR #8300 的路径](verl-shared-prefix-packing.md)则重复最后一个 prompt token 并使用显式分段 attention；不能把后一方案当成本项目旧稿的实现或成果。

本轮原生 CPU 检查仅确认旧输入拼接的 token、mask、position 与形状。过去草稿中的 GPU/模型数字未在本机重新执行或认证；历史实验分别保留在[attention 微基准](../fundamentals/prefix-sharing-packed-thd-rtx4090.md)、[full-model 模拟](../fundamentals/full-model-ps-rtx4090.md)与[训练模拟](../fundamentals/grpo-training-ps-rtx4090.md)，适用范围以各文件最新更正为准。
