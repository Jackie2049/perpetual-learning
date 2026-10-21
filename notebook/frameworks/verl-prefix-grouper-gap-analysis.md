# verl PrefixGrouper 计算范围与旧性能论证更正

PrefixGrouper 不只是省去 attention 内部的重复计算。旧 verl 集成先把同组 prompt 和多个 response 拼成一条模型输入，再在 attention 内分开处理可见域。共享前缀因此也不会以原先每样本一份的形式重复进入 QKV 投影、MLP 和逐 token norm。**原文“只节省 attention、漏掉 91.2% 的计算，存在 2.5 倍优化缺口”的判断撤回。**

2026-10-21 更正。原文形成于 2026-06-07，其性能数字保留为历史记录，不能作为原生 verl 基线或待实施优化的收益。旧贡献推进计划停用；[关联 RFC](verl-6401-rfc-full-model-ps.md)同步撤回，不发布、不竞争已有作者的 [FSDP packing PR #8300](https://github.com/verl-project/verl/pull/8300)。

## 接入位置不等于节省范围

固定 [verl #4368 merge d6bc6296](https://github.com/verl-project/verl/commit/d6bc62964d06588b0728122a1d2414b098165295) 的 `prefix_grouper_utils.py`：

```text
按相邻 uid 分组 → 每组选一个 prompt
→ PrefixGrouper.concat_input
→ model(input_ids=concat_input_ids, prefix_grouper=...)
→ attention 内解组计算并重组输出
→ split_output 恢复每个 response 的 logits
```

官方库固定 [1accedc0 的 concat_input](https://github.com/CASIA-IVA-Lab/PrefixGrouper/blob/1accedc043c9ea79165888f48ab163fb8d6d5bb0/src/prefix_grouper/__init__.py) 通过 `GroupFunction` 将每组 prefix 写入一次，再顺序写入各 suffix。attention 输出重新回到同一 grouped 布局，随后 MLP 等逐 token 算子继续处理这个布局；“monkey patch 位于 attention”不能推出模型其余部分仍处理 G 份有效 prompt。

例如 P=3、G=3、每个 R=2，独立输入有 15 行，grouped 输入为 `[prompt, response1, response2, response3]`，只有 9 行。它与 [#8300](verl-shared-prefix-packing.md) 的 `prompt[:-1]` trunk、每 branch 重复末 prompt token 不同：旧库通过 `split_output(include_prefix_last=1)` 为各 response 带回末 prompt 位置的输出。两条路径不要混用布局或启用限制。

旧集成的安装说明是未锁版本的 `pip install prefix_grouper`；本轮所选库版本不代表历史服务器实际安装版本。verl 的 helper 在 merge 与本轮基线 `cd58f550` 间无 diff；在后者中旧 forward 只找到定义，不能把历史实现称为当前 engine 默认路径。

## 有效 token 与矩形 padding

旧库每组逻辑长度是 `P + ΣR_i`，但批次张量要 pad 到最长组，逐 token dense 算子会处理这些 pad 行。共享减少的是重复有效前缀，实际算子行数还受组大小、长度分布与 padding 影响。

P=2、R=4、组大小为 3 和 1 时：独立布局为 `4×6=24` 行；grouped 有效长度为 14 和 6，总共 20；矩形布局却是 `2×14=28` 行。这是“共享有效前缀不保证总行数下降”的具体反例，不是对所有 PG 配置的负性能结论。

## 原生 CPU 输入检查

[脚本](../../experiments/scripts/verify_prefix_grouper_inputs.py)从固定 Git blob 原样加载官方包，不安装依赖、不检出源码；再 AST 提取原始 verl 两个输入构造函数，没有模型或索引替身。Python 3.12.13、torch 2.9.1、全部 tensor 在 CPU。5 组人工输入的 token 顺序、mask、position 和张量形状通过：

| 输入条件 | 原独立 padded 行数 | grouped padded 行数 | grouped 有效行数 |
|---|---:|---:|---:|
| 单组 3 个相同 prompt | 15 | 9 | 9 |
| 不均匀组大小 3 和 1 | 24 | 28 | 20 |
| 相同 token 但 uid 各不相同 | 9 | 9 | 9 |
| 相同 uid 被其他组隔开 | 9 | 9 | 9 |
| prompt 和 response 长度混合 | 18 | 14 | 11 |

```bash
python experiments/scripts/verify_prefix_grouper_inputs.py /path/to/verl /path/to/PrefixGrouper
```

两个仓库须已有固定对象及所需 blob；脚本禁用 Git lazy fetch，打印每个加载源文件的 SHA256。没有执行 attention、模型、梯度、FSDP、checkpointing、设备或性能测试。所选库源码支持上述布局，不代表其与任意 transformers/verl 组合兼容。

## 旧数字为什么不能支撑性能缺口

| 原说法 | 核对结果 | 处理 |
|---|---|---|
| 原生 PG 是 0.99 倍 | 数值来自本项目自写的 attention 微基准，没有导入 PG 或 verl | 不再作为 PG 基线 |
| 完整语义的 full-model PS 是 2.46 倍 | 自写 MiniTransformer 的 suffix-only 分支没有注入前缀 K/V，且未以同一输入比较输出 | 不作为等价实现的加速证明 |
| 两者形成 2.5 倍可行动缺口 | 是不同计算任务、计时边界的比值，旧架构判断也不成立 | 撤回缺口与优先级结论 |
| 训练收益可乘 0.76 得到 1.87 倍 | 单配置比例不能成为通用换算因子，且前向基数来自上述非等价模拟 | 撤回训练预估 |
| MLP 固定占 82%，attention 必然 memory-bound | 占比取决于模型、长度、batch、后端与硬件；这里没有相应瓶颈 profiling | 不外推至 PG 或 #8300 |

具体源代码问题：

- [attention 微基准](../../tools/prefix_sharing_packed_thd_4090.py)的 Exp2 在计时中创建随机 Q/K/V；基线处理 n 份 suffix query，所谓 PS 路径只处理 n−1 份，provider attention 没有计入；`gqa_sdpa` 默认非 causal。没有原生 PG，也未验证输出一致或测量真实 HBM 流量。
- [full-model 微基准](../../tools/full_model_ps_benchmark_4090.py)的 Exp3 对 provider 和 reusers 分别调用 `model`；后者只有 suffix，未传任何 prefix KV，attention 不可能看到 prompt。输入还分别随机生成。其 hybrid 层用简化投影模拟，不是原生 DeltaNet 状态注入验证。
- 加速应按时间组成估算。若原时间份额为 f、b、o，对应加速 S_f、S_b、1，则在这些份额和开销假设下 `S_step = 1/(f/S_f + b/S_b + o)`；不能普遍使用 `S_f×0.76`。

原始数字和脚本保留作历史材料，不更改计时逻辑冒充已修复 benchmark。先前对 step-2+ loss 不一致的猜测也不作为根因；输入、位置、mask、随机性、梯度和归约需分别建立配对证据。

## 后续边界

本项架构误判已收束。#8300 的 no-padding、分支末 prompt、FA2/FA3 与调度集成仍有独立学习价值，但不能包装成我方提出的新 full-model sharing。真实性能比较须同时保证相同样本与损失语义、完整 provider/branch 成本、匹配后端和稳态计时；GPU 实验由 M5 按用户分工负责，本机不启动。

原生 helper SHA256 `22f66f26c8c918f7ecf5e8522ed434f40e94660733b1f3dadfd31ff8b7699db0`；库 `__init__.py` SHA256 `7df14f9661946fdd6836dbf3050dd0c21dd8f2c1785bf50d0e03a4744733f6c4`。历史 H800 表的条件可追溯到[原集成说明](https://github.com/verl-project/verl/blob/d6bc62964d06588b0728122a1d2414b098165295/examples/prefix_grouper/README.md)，不能与自写 4090 微基准直接相除。
