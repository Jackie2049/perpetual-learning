# verl：共享提示词前缀的训练计算复用分析

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 共享提示词前缀的训练计算复用分析。[Issue #6401](https://github.com/verl-project/verl/issues/6401) · [PR #4368](https://github.com/verl-project/verl/pull/4368)

**【类型】** 贡献分析

**【价值评估】** 6 / 10 — 能够解释现有分组与真正共享注意力之间的差距；具体实现及收益未验证。

**【进展】** 历史分析/草稿：已阅读 PrefixGrouper 并整理差距，未记录本文对应评论发布或共享注意力实现完成。

**【大白话讲明白】**

同一个问题生成多个回答时，开头的提示词相同，重复计算这部分会浪费资源。把这些样本分到一起只完成了第一步，还需要底层计算真正复用前缀，才能获得预期收益。

**【What–Why–How】**

- **What：** 现有 PrefixGrouper 与前缀树共享注意力 RFC 的差距。
- **Why：** 分组、数据布局和注意力内核的复用是不同层次，不能把分组直接视为性能优化完成。
- **How：** 对照现有分组、掩码和内核接口提出分阶段方案；性能收益需要后续实测。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 深度源码分析: verl PrefixGrouper 现状 + RFC #6401 差距分析

## 1. 现有 PrefixGrouper 架构

### 核心文件
| 文件 | 作用 |
|------|------|
| `verl/trainer/ppo/prefix_grouper_utils.py` | verl 集成层 |
| `verl/models/transformers/monkey_patch.py` | 注意力函数补丁 |
| `verl/utils/seqlen_balancing.py` | 组感知负载均衡 |
| 外部: `pip install prefix_grouper` | 核心算法库 |

### 工作原理
1. 按 uid 分组 (相同 uid = 相同 prompt)
2. 拼接为 `[prefix | suffix1 | suffix2 | ... | suffixN]`
3. Monkey-patch 拦截注意力调用 → 委托给 `prefix_grouper.forward()`
4. 分解为 prefix attention + suffix attention
5. 输出通过 `split_output()` 恢复每个 suffix 的 logits

### 性能 (Qwen3-4B, 4×H800)
- 4K context: 1.26-1.30x 加速
- 8K context: 1.56-1.70x 加速
- 更长 prompt → 更大加速比

## 2. GRPO 前缀共享三层架构

### Layer 1: Rollout 推理前缀缓存
- vLLM `enable_prefix_caching=True` (默认开启)
- 相同 prompt 复制 rollout.n 次 → vLLM 自动复用 KV 块
- 每次 weight update 后 `reset_prefix_cache()`

### Layer 2: 批处理均衡
- `get_group_balanced_partitions()` 按 uid 分组
- Karmarkar-Karp 分区算法保持 uid 组完整

### Layer 3: 训练前缀共享 (PrefixGrouper)
- `use_prefix_grouper=True` 配置标志
- ppo_mini_batch_size 按 rollout.n 缩放
- Monkey-patch 全局包装 ALL_ATTENTION_FUNCTIONS

## 3. RFC #6401 vs 现有架构差距

| 方面 | 现有 PrefixGrouper | RFC #6401 (Magi-based) |
|------|-------------------|----------------------|
| 分组模型 | 扁平 (相同 uid) | 树形 (任意深度) |
| 注意力后端 | FA2/FA3/SDPA | Magi Attention (稀疏掩码) |
| 前缀检测 | 基于 uid (隐式) | 基于 hash 的 prefix_segments |
| 支持 | FSDP only | Megatron + FSDP |
| KV 缓存 | 每层注入 | Flat layout + block-sparse mask |
| 跨 batch | 不支持 | 未来: cache-based |

## 4. 贡献机会

### 4.1 最小可行贡献 (Quick Win)
1. **文档/测试**: 为 PrefixGrouper 添加单元测试
2. **Bug 修复**: RFC 中提到的 loss 差异问题 (step1 匹配, step2+ 分歧)
3. **FSDP 引擎集成修复**: 当前 FSDPEngineWithLMHead.forward_step() 没有传递 prefix_grouper kwarg

### 4.2 核心贡献 (需要更多时间)
1. **prefix_tree_utils.py**: 新的树形前缀检测模块
2. **Monkey-patch 扩展**: 支持每样本 prefix 长度
3. **Batch 均衡扩展**: 基于 trie 结构的负载均衡
4. **配置扩展**: `prefix_sharing: PrefixSharingConfig`

### 4.3 建议的贡献路径

```
Step 1 (本周):
  - 在 #6401 issue 下评论技术分析
  - 提交 prefix_grouper FSDP 集成修复 PR

Step 2 (下周):
  - 实现 prefix_tree_utils.py (树形检测)
  - 添加 monkey-patch 扩展支持

Step 3 (后续):
  - Magi Attention 后端集成
  - Megatron 后端支持
  - 跨 micro-batch cache 支持
```

## 5. 与 prefix-sharing 项目的关系

Rollout-infra 的 prefix-sharing 项目 (`Jackie2049/prefix-sharing`) 已实现:
- `TriePrefixDetector`: 基于 trie 的前缀检测
- `PrefixSharingConfig`: 完整配置数据类
- KV injection + 恢复逻辑

这些可以直接贡献给 verl 的 RFC #6401 实现!
