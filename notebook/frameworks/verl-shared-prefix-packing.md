# verl FSDP shared-prefix packing

同一 prompt 的多个 rollout 可以共享训练前向中的前缀，但不能简单拼成一条普通因果序列。verl [PR #8300](https://github.com/verl-project/verl/pull/8300)把它表达为一条 trunk 和多个互相隔离的 branch，并在输出恢复到原样本布局之后交给损失函数。本轮验证的是布局，不是完整模型或加速效果。

2026-10-21 阅读。PR 作者 DeeproChoudhury；读取时 open。固定 head `d6dfcb284d5acbe8a2ecb860843edee7b9d078be`、base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`，下文不自动适用于后续提交。这是已有作者方案的学习，不是我方贡献候选或审阅结论。

## 与旧 PrefixGrouper 的区别

旧[阅读笔记](verl-prefix-grouper.md)描述的是 `prefix_grouper` 外部库和旧 actor 集成。在上述 base 对 `forward_micro_batch_with_prefix_grouper` 做 `git grep`，verl 内只找到函数定义；不能把那篇笔记当作当前 FSDP engine 的启用说明。

| 方面 | 旧笔记所描述的集成 | #8300 固定 head |
|---|---|---|
| 路径 | `trainer/ppo/prefix_grouper_utils.py` | `workers/engine/fsdp/transformer_impl.py` |
| 分组 | 旧集成按 uid | packing helper 比较相邻样本的 prompt token 和长度，不按 uid 判断相同内容 |
| 无 padding | 旧笔记列为不兼容 | 新路径要求 `use_remove_padding=True` |
| 注意力 | 外部 PrefixGrouper | 分段 Q、索引聚集 K/V、显式不同的 Q/K 累积长度 |
| 批处理 | 旧笔记列 dynamic batch 不兼容 | 新方案按压缩长度预算，超预算或 DP 所需微批数会拆组、重建 trunk |

新 helper 不做全局搜索、排序或最长公共子串压缩。相同 prompt 被其他样本隔开不会在此合并；prompt 长度小于 2 也不共享。调度按 uid 尽量保组，不能代替进入 helper 时的 token 一致性检查，也不保证一个组永远完整留在同 rank/微批。

## 为什么重复最后一个 prompt token

设 prompt 为 `[a,b,c]`，两个 response 为 `[x,y]`、`[z,w]`：

```text
独立输入      [a b c x y]  [a b c z w]
packed 行     [a b] [c x y] [c z w]
position_ids  [0 1] [2 3 4] [2 3 4]
可见 key      trunk  trunk+本分支  trunk+本分支
```

trunk 是 `prompt[:-1]`，每个 branch 从 `prompt[-1]` 开始。第一枚 response 的 logprob 来自最后一个 prompt 位置的 logits；若只保留一个 `c`，全局 `roll(-1)` 标签只能指向其中一个分支。重复 `c` 让两个读出位置分别预测 `x` 和 `z`，各自也能带上对应样本的 temperature。

`pack_index` 同步选择 token、原 position 和 temperature，`unpack_index` 恢复逐 token 输出。各段末尾的 rolled label 可能指向下一段，但响应读出窗口只读每个样本 `P−1 .. L−2`，不读取该样本最后一行；不能把“有效读出对齐”误说成“所有 packed 行的 label 都有原序列含义”。

## 位置与注意力是两件事

分支 position 从 `P−1` 开始，保持原 RoPE 坐标；这本身不隔离分支。真正的可见域来自 Q/K 分段和 `kv_index`：trunk 只看 trunk，branch 看 trunk 加自身，不能看到兄弟 branch。

在 [attention wrapper](https://github.com/verl-project/verl/blob/d6dfcb284d5acbe8a2ecb860843edee7b9d078be/verl/models/transformers/monkey_patch.py) 中，K/V 按 `kv_index` 聚集后传给 `_flash_attention_forward`；显式传 `cu_seq_lens_q/k` 和最大段长，位置推断边界的参数置空。分支 Q 比 K 短，需要 **bottom-right causal alignment**：branch 的第 i 个 query 可看完整 trunk 和本分支至 i 的 key。若用 top-left 三角，最前面的分支 query 看不全前缀。

尾部 padding 是独立的一段，pad 行可以看本段更早 pad 行，不接触真实 token；不是每枚 pad 都各自独立。逐层计数器检查每层是否进入 wrapper；FSDP 复制 dataclass 参数时共享计数器。计数正确仅是路由检查，不足以证明所有模型的语义正确。

## 梯度与损失等价需要什么

下面是依据源码的条件分析，不是本机完整训练证明：

- 对相同位置的确定性前缀运算，共享节点的链式法则梯度应累加各分支贡献。`index_select` backward 对重复索引求和，不需要人工乘或除 rollout 数。
- 这不保证开启 dropout 或其他随机操作时与独立计算逐样本、逐 RNG 位相等；不同微批分割还可能改变数值舍入、随机流及其他状态。作者 CPU 模型 oracle 使用 `eval()`，不能据此宣称任意训练配置严格等价。
- engine 在去 padding 后按 `unpack_index` 恢复 logprob、entropy、sum_pi_squared 或 value，再恢复原样本形状。全局归一化 token 数在 packing 前由原 `loss_mask.sum()` 跨 DP 汇总。压缩行数不是新的 loss 分母；仍须保留原响应 mask、每样本权重和 loss 的归约契约。
- 本轮没有执行完整 loss、backward 或 optimizer；源码位置一致不证明任意自定义 loss 或分布式训练等价。

## 节省了什么

G 个等长 prompt（P）和 response（R）的原始 token 行数是 `G(P+R)`，packed 后是 `(P−1)+G(R+1)`，节省 `(G−1)(P−1)` 行。这减少前缀的投影、MLP、激活等重复处理；K/V 在 attention 入口仍按分支聚集，不能理解成所有层都只存一份前缀 K/V 张量。

忽略常数的注意力交互量由 `O(G(P²+PR+R²))` 变为 `O(P²+G(PR+R²))`。response 仍必须看 prompt，`GPR` 不消失。实现还存在索引、K/V 聚集、段组织与调度开销；`prefix_packing_token_ratio` 只是每微批 `unpacked/packed` 的均值指标，不是全局加权压缩比，也不是加速比、性能下界或收益保证。PR 作者报告的 GPU benchmark 未在本机复现。

## 固定 head 的支持范围

默认关闭。配置 guard 要求 remove-padding、Ulysses SP=1、CUDA、FA2/FA3，拒绝旧 top-left mask 的 flash-attn、vision model 及直接接收 `cu_seqlens` 的模型（例如部分线性注意力 hybrid）。微批 guard 还拒绝 top-k distillation、score centering、router replay、多模态输入、缺少 prompts。

`split_groups_to_budget` 会拆超预算组，每个分组重算 trunk；单个过长样本仍可能独占且超预算，因此该 helper 本身不是硬上限保证。上层 `rearrange_micro_batches` 另外检查样本/组预算，并协调 DP 微批数；本轮未执行这些集成与通信路径。

## 本机限定执行

[复现脚本](../../experiments/scripts/verify_prefix_packing_layout.py)从固定 Git 对象 AST 提取原始布局定义及原始测试，保留函数体；移除两个函数的参数化装饰器，显式展开同样的参数。无需导入完整 verl，无框架配置替身、未修改源码 checkout，也未安装新依赖。

```bash
python experiments/scripts/verify_prefix_packing_layout.py /path/to/verl
```

源码仓库须已含上述 head；脚本不会联网 fetch。2026-10-21 M1 既有独立环境 Python 3.12.13、torch 2.9.1，全部 tensor 在 CPU。38 组通过：分组 7、布局/还原/尾 pad 5、预算 helper 5、rolled label 4、dense mask 对独立逐行可见域 oracle 16、torch 原生 `_apply_to_tensors` 复制计数器 1。运行出现 torch FX 的 `is_fx_tracing` 提示，没有断言失败。

源文件 SHA256：

```text
prefix_packing.py                 14f7366e0161ee82b7e6971ec5296a21876ea205f92abc4c7d80ca3e85b284e1
test_prefix_packing_on_cpu.py     403e3913635acf1cbb650498efd425591840034d1f6a93e2c3b774aa0682bf48
```

不是执行完整 pytest/conftest；未运行 TensorDict prompt-length、tiny Qwen 的 logprob/entropy/梯度/value/temperature、engine、FA2/FA3、DP/SP 或 GPU 测试。既有环境缺 transformers/TensorDict 等集成依赖，不为本项学习安装。布局检查不能升级为作者 GPU 测试、训练等价或性能证明。

源码入口：[plan 与 guard](https://github.com/verl-project/verl/blob/d6dfcb284d5acbe8a2ecb860843edee7b9d078be/verl/workers/utils/prefix_packing.py)、[CPU 测试](https://github.com/verl-project/verl/blob/d6dfcb284d5acbe8a2ecb860843edee7b9d078be/tests/workers/test_prefix_packing_on_cpu.py)、[engine](https://github.com/verl-project/verl/blob/d6dfcb284d5acbe8a2ecb860843edee7b9d078be/verl/workers/engine/fsdp/transformer_impl.py)、[批预算](https://github.com/verl-project/verl/blob/d6dfcb284d5acbe8a2ecb860843edee7b9d078be/verl/utils/seqlen_balancing.py)。
