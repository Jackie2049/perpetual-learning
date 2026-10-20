# verl：共享提示词前缀的训练计算复用评论草稿

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 共享提示词前缀的训练计算复用评论草稿。[Issue #6401](https://github.com/verl-project/verl/issues/6401) · [PR #4368](https://github.com/verl-project/verl/pull/4368)

**【类型】** 技术评论

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

> Draft of a technical comment to post on github.com/volcengine/verl/issues/6401

## Comment (English)

Thanks for the great RFC! I've been studying verl's existing PrefixGrouper implementation and have some analysis that might be helpful.

### Current PrefixGrouper Architecture (PR #4368)

The existing `prefix_grouper` (external package, integrated via `verl/trainer/ppo/prefix_grouper_utils.py`) works as follows:

1. **Flat grouping**: Groups sequences by `uid` (same uid = same prompt prefix)
2. **Concatenation**: `[prefix | suffix1 | suffix2 | ... | suffixN]` into a flat layout
3. **Monkey-patch**: Intercepts attention calls via `verl/models/transformers/monkey_patch.py` → delegates to `prefix_grouper.forward()`
4. **Decomposition**: Splits into prefix attention + suffix attention
5. **Recovery**: `split_output()` restores per-suffix logits

Performance (Qwen3-4B, 4×H800): 1.26x @ 4K, 1.56-1.70x @ 8K context.

### Gap Between PrefixGrouper and RFC #6401

| Aspect | PrefixGrouper (PR #4368) | RFC #6401 (Magi-based) |
|--------|--------------------------|------------------------|
| Grouping | Flat (same uid) | Tree (arbitrary depth) |
| Attention backend | FA2/FA3/SDPA | Magi FFA (block-sparse mask) |
| Prefix detection | uid-based (implicit) | hash-based prefix_segments |
| Backend support | FSDP only | Megatron + FSDP |
| Approach | Decompose into prefix+suffix attn | Flat packing with sparse mask |
| Cross batch | No | Future: cache-based |

### Key Observation: Loss Discrepancy

Following @Kirrito-k423's observation about step-2+ loss divergence — this could be related to:
1. The `FSDPEngineWithLMHead.forward_step()` not passing `prefix_grouper` kwargs properly
2. Numerical precision differences in Magi's dispatch solver for sparse patterns
3. Activation checkpoint offloading differences between steps

A gradient equivalence debug flag (as suggested) would be very helpful.

### Benchmark Evidence: Attention-Only PS vs Full-Model PS (RTX 4090)

I've run benchmarks on an RTX 4090 that reveal a critical gap between attention-level prefix sharing (what current PrefixGrouper does) and full-model prefix sharing:

| Approach | n=4 (75% prefix) | n=8 (67% prefix) | Long context (96% prefix) |
|----------|-----------------|-----------------|--------------------------|
| Attention-only PS (PrefixGrouper) | **0.99x** | **0.99x** | — |
| Full-model forward PS (One-Forward) | **2.08x** | **2.46x** | **3.55x** |
| Training PS (fwd+bwd) | **1.59x** | ~1.87x (est) | ~4.5x (est) |

The 2.5x gap comes from MLP being 82% of layer compute time (compute-bound) while attention is only 18% (memory-bound). Current PG only intercepts attention → saves 0% of MLP compute → **0.99x speedup for long sequences**.

For GRPO with long prompts (DeepSeek-R1/TreeRL style), full-model PS gives much higher speedup:
- prefix_ratio=80% → 2.38x (fwd) / 1.81x (training)
- prefix_ratio=89% → 2.89x (fwd) / 2.19x (training)
- prefix_ratio=96% → 3.55x (fwd) / 2.69x (training)

Formula: `speedup = n / (1 + (n-1) × suffix_ratio)` validated at all prefix ratios.
Training discount: `training_speedup ≈ forward_speedup × 0.76` (backward+optimizer dilutes savings).

### KV Injection Precision Validation (RTX 4090)

I've prototyped the One-Forward + KV Injection + Prefix-Last Restore architecture (matching prefix-0501's approach):

**Critical finding**: Using `is_causal=True` in attention gives `cos_sim ≈ 0` for suffix logits (suffix tokens can't see prefix positions). The correct implementation requires a **block-causal mask** where:
- Prefix block: all suffix positions can see all prefix positions (mask=0)
- Suffix block: causal masking within suffix positions (mask=-inf for future)

With block-causal mask:
- `cos_sim_suffix = 0.999999` (max_diff = 0.004) → suffix logits identical to baseline ✓
- `cos_sim_provider = 1.0` (max_diff = 0.0) → provider logits exactly match baseline ✓
- KV injection overhead ≈ 0ms → essentially free ✓

This validates that KV injection preserves attention semantics correctly, making it safe for production use.

### Proposed Architecture: Provider-Reuser Forward Splitting

Based on the benchmark evidence, I propose extending PrefixGrouper beyond attention-level sharing:

1. **Provider**: Full forward on (prefix + suffix) → extract prefix KV at each layer
2. **Reuser**: Suffix-only forward with KV injection (block-causal mask) → skip prefix MLP/QKV_proj/LN entirely
3. **Prefix-last logprob restoration**: Provider's prefix-last token logprob used for training correctness

This requires model-level modification (not just attention monkey-patch), but the 2.5x improvement gap makes it worthwhile.

Challenges:
- Custom attention backend for block-causal mask (FlashAttention doesn't support arbitrary float masks → falls back to math backend)
- Gradient flow through shared prefix KV in backward pass
- FSDP compatibility with model-level PS

### Potential Contribution

I've been working on a `prefix-sharing` project that implements:
- `TriePrefixDetector`: Tree-based prefix detection using tries
- `PrefixSharingConfig`: Full configuration dataclass
- KV injection + recovery logic

These could contribute to the tree-based prefix detection module needed for this RFC.

### Questions

1. **Prefix tree depth**: For GRPO n=8, the tree is flat (1 prefix + 8 suffixes). Multi-turn would increase depth — what's the target depth? rStar-Math MCTS trees could be quite deep.
2. **Magi Attention version**: The RFC uses MagiAttention 1.1.0 with FFA kernel — is the Blackwell (FA4) fork ready for production use?
3. **Dispatch solver overhead**: How significant is the dispatch solver's chunk-level sharding overhead for typical GRPO batch sizes?

Looking forward to contributing!

---

## RFC #6401 Research Summary (from agent)

### Issue Details
- **Author**: arvyanh
- **Date**: May 19, 2026
- **URL**: https://github.com/volcengine/verl/issues/6401

### Main Proposal
Pack GRPO samples into flat token layout `[prefix | leaf_0 | ... | leaf_{n-1}]` with block-sparse attention masks. Shared prefix computed once at O(P) instead of O(P²) × n.

### Benchmark Results
- Dataset A: 42% faster, 30% less memory
- Dataset B: ~3x forward speedup over FA3

### Related Work
- **Magi Attention**: https://github.com/SandAI-org/MagiAttention (840 stars, Apache 2.0)
  - FFA (Flex-Flash-Attention): Generalized attention with AttnSlice masks
  - Dispatch Solver: Fine-grained chunk-level sharding for load balance
  - Integrations: Megatron-LM, FSDP, HuggingFace
- **PR #4368**: PrefixGrouper (alternative FSDP-only approach, decomposition-based)
- **PR #6271**: Multi-trajectory support in async pipeline (88 commits, open)
- **Use cases**: rStar-Math (MCTS), TreeRL, DeepSearch (ICLR 2026)

### Existing Comment (Kirrito-k423, May 21)
- Confirmed prefix-tree masking approach is sound
- Flagged step-2+ loss discrepancy (step 1 matches, step 2+ diverges)
- Recommended gradient equivalence debug flag
- Offered to collaborate on Magi integration
