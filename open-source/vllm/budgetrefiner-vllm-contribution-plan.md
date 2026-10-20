# vLLM：按服务延迟目标动态安排计算预算（方案）

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** 按服务延迟目标动态安排计算预算（方案）。[目标调度器](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core) · [来源项目 vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)

**【类型】** 新特性规划

**【价值评估】** 5 / 10 — 服务延迟目标有明确应用价值，但迁移方案、性能数据和当前上游空间尚未验证。

**【进展】** 历史方案：BudgetRefiner 迁移与 PR 文案已规划，仍缺源码适配和 RTX 4090 profiling；没有可引用的已提交 PR。

**【大白话讲明白】**

推理服务不只是追求总吞吐，也希望每个用户及时收到回答。这个方案想根据剩余时间和负载调整每轮处理多少内容，让调度更关注响应期限；目前仍是需要验证的设计。

**【What–Why–How】**

- **What：** 将 BudgetRefiner 的服务目标感知调度思路用于 vLLM。
- **Why：** 固定预算可能无法同时满足首 token 延迟、后续输出间隔与吞吐目标。
- **How：** 先核对来源实现和 vLLM 调度接口，再采集延迟/吞吐基线和压力场景；不沿用旧文中未经实测的迁移比例或收益承诺。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 2026-06-16 | vllm-project/vllm | vllm-project/vllm-ascend | BudgetRefiner | SLO-aware scheduling
> ★★★★★★★ BudgetRefiner = vLLM-Ascend最有价值upstream贡献 → 95%+ GPU-generic → RTX 4090 profile data unique!
> ★★★★★★★ vLLM V1 has ZERO SLO-aware scheduling → BudgetRefiner fills genuine gap → 无竞争PR

## 1. ★★★★★★★ 为什么BudgetRefiner SLO是最有价值贡献

```
★★★★★★★★★ 对比分析:

| Contribution | Impact | Novelty | Scope | Our Advantage | Priority |
|-------------|--------|---------|-------|---------------|----------|
| BudgetRefiner SLO | ★★★★★★★ | ★★★★★★★ | Medium (300 lines) | ★★★★★ RTX 4090 profile data | ★★★★★★★★ #1 |
| Inductor SM<90 Fusion Guard | ★★★★★★★ | ★★★★★ | Large | ★★★★★ Root cause understanding | ★★★★★★ #2 |
| QuantKey refactor #32268 | ★★★ | ★★★★ | Small (refactor) | ★★★★ INT4/SM89 knowledge | ★★★ #3 |

★★★★★★★★★ BudgetRefiner #1原因:
  → ★★★★★★★★★★ 填补真实空白 → vLLM V1无SLO-aware scheduling → 无人竞争 → → ★★★★★★★★★★★★★ RTX 4090 profile data → 独家贡献 → 无其他贡献者有此数据 → → ★★★★★★★★★★★★★★★ 小scope → 105行核心逻辑 → 易review → 易test → → ★★★★★★★★★★★★★★★★★★ 生产级影响 → Cloud serving需要SLO → 商业价值 → → ★★★★★★★★★★★★★★★★★★★★★ 学习价值 → scheduler = AI infra最核心skill!
```

## 2. ★★★★★★★ Ascend vs GPU Generic Breakdown

```
★★★★★★★★★ 源码级分析 — 95%+ GPU-generic:

| Component | Ascend-Specific | GPU-Generic | Porting Effort |
|-----------|----------------|-------------|---------------|
| BudgetRefiner class | 0% | 100% | Direct copy + rename |
| refine_budget() logic | 0% | 100% | Direct copy |
| _read_lookup_table() | 0% | 100% | Direct copy |
| _align_key() | 0% | 100% | Direct copy |
| _get_max_budget() | 0% | 100% | Direct copy |
| profile_table.csv data | 100% (910B3) | Needs GPU data | New profiling required ★★★★★★ |
| Decode-first reordering | 0% | 100% | Direct copy (d_lst + p_lst) |
| SLO_limits config | 0% | 100% | Add to SchedulerConfig |
| Pandas CSV loading | 0% | 100% | Direct copy |
| ProfilingChunkPredictor | 5% (min_chunk=4096) | 95% | Adjust min_chunk for GPU |

★★★★★★★★★ 结论: 只有profile_table.csv是Ascend-specific → 其他95%+直接可移植!
```

## 3. ★★★★★★★★ Contribution Plan — 4 Phases

### Phase 0: Pre-Work (1-2 weeks)

```
★★★★★★★★★ Goal: 建立community credibility

Actions:
  → 1. Complete Tier 1-2 contributions (#32268 QuantKey, #43204 cleanup) → build merge history
  → 2. Comment on closed SJF RFC (#29406) → reference BudgetRefiner as concrete alternative
  → 3. Open new vLLM issue: "[Feature] SLO-aware dynamic token budget for V1 scheduler"
  → 4. Engage vLLM scheduler maintainers on Slack → informal concept discussion
  → 5. Read vLLM SchedulerConfig source → understand config extension patterns
```

### Phase 1: RFC + Minimal Implementation (3-4 weeks)

```
★★★★★★★★★ Goal: RFC + proof-of-concept

Step 1.1: Write RFC
  → Motivation: vLLM V1 fixed token budget → no SLO guarantees → decode blocked by long prefill
  → Proposed: BudgetRefiner → SLO_limits_for_dynamic_batch → SchedulerConfig → default=-1 → disabled
  → Design: decode-first running queue + GPU profile tables + BudgetRefiner.refine_budget() before RUNNING

Step 1.2: Proof-of-Concept Code
  → vllm/v1/core/sched/scheduler_slo.py → new file → inherits Scheduler
  → → BudgetRefiner class (~105 lines) → ported from vllm-ascend
  → → SchedulerSLO class (~200 lines) → decode-first + budget refinement
  → vllm/config/scheduler.py → extend → SLO_limits: int = Field(default=-1)
  → vllm/v1/core/profile_table_gpu/ → new directory
  → → profile_table_a100.csv → A100 profiling data
  → → profile_table_rtx4090.csv → ★★★★★★★★ RTX 4090 profiling data (OUR unique contribution!)

Step 1.3: GPU Profile Data Collection
  → ★★★★★★★★★★★★★★★ RTX 4090 profiling = our unique contribution → no other contributor has this!
  → → CSV format: ctx_len, d_num, cost, chunk_size
  → → Profile script: tools/profile_slo_budget.py → synthetic requests → measure latency → generate CSV
```

### Phase 2: Full Implementation + Testing (4-6 weeks)

```
★★★★★★★★★ Goal: Complete feature with tests + benchmarks

Step 2.1: Design Decisions
  → Profile table loading → Option B + C (user-provided + auto-detect GPU type)
  → Pandas dependency → Keep (common ML dependency)
  → Decode-first ordering → Add within RUNNING queue (d_lst + p_lst)
  → Block size → vLLM GPU block_size=16 → profile_table需要recalibrate

Step 2.2: Test Coverage
  → Unit tests → BudgetRefiner (lookup table loading, key alignment, budget refinement)
  → Unit tests → decode-first ordering
  → Integration tests → SchedulerSLO (full schedule() cycle)
  → Benchmark → SLO-aware vs fixed-budget on A100 + RTX 4090

Step 2.3: Expected Results
  → SLO_limits=50 → ITL p99 under target → effective throughput increase
  → SLO_limits=35 → stricter SLO → ITL lower but throughput reduced
  → Without BudgetRefiner → ITL spikes when long prefill blocks decode
```

### Phase 3: Community Engagement + Merge (4-8 weeks)

```
★★★★★★★★★ Goal: RFC approval + merge

Key objections to anticipate:
  → "Why not adjust max_num_batched_tokens manually?" → BudgetRefiner dynamic per-iteration → not static
  → "What about Pandas dependency?" → Minimal → commonly available
  → "Profile tables hardware-specific?" → Yes → but framework portable → tables are just data
  → "How interact with watermark?" → Complementary! BudgetRefiner + watermark = comprehensive
  → "Decode-first breaks FCFS?" → Only within RUNNING → WAITING still FCFS

Merge strategy:
  → 1. Get RFC approved first (separate issue)
  → 2. Implement as single PR
  → 3. Include profile tables as package data
  → 4. Include profiling tool for users
  → 5. DCO sign-off on all commits
```

### Phase 4: Extensions (Ongoing, after merge)

```
★★★★★★★★★ Goal: Expand feature

Step 4.1: Online Profiling (Quadratic Model)
  → Port ProfilingChunkPredictor → f(l) = al^2 + bl + c → hardware-independent
  → Online fitting during warmup → no need for pre-generated CSV

Step 4.2: SLO Metrics Integration
  → Prometheus metrics → vllm_slo_budget_refined_total, vllm_slo_budget_actual, vllm_slo_violation_total

Step 4.3: TTFT SLO Support
  → Extend to control TTFT → balance TTFT + ITL simultaneously

Step 4.4: RTX 4090 Specific Profile Tables
  → INT4 weights + INT8 KV → highest throughput path
  → BF16 weights + INT8 KV → standard path
  → Qwen2.5-7B, Llama-3.1-8B → most common small models
```

## 4. ★★★★★★★★ BudgetRefiner核心源码分析

```
★★★★★★★★★ BudgetRefiner核心 (~105 lines):

class BudgetRefiner:
    def __init__(self, slo_limits_for_dynamic_batch, profile_table_path):
        self.slo_limits = slo_limits_for_dynamic_batch  # SLO target (ms)
        self.lookup_table = self._read_lookup_table(profile_table_path)
        self.max_budget = self._get_max_budget()

    def refine_budget(self, num_decode_tokens, num_decode_seqs):
        """Dynamic token budget based on SLO target and decode workload."""
        # Step 1: Look up max chunk_size that keeps decode latency under SLO
        key = self._align_key(num_decode_tokens, num_decode_seqs)
        max_chunk_size = self.lookup_table.get(key, self.max_budget)

        # Step 2: Budget = max_chunk_size (prefill budget per iteration)
        return max_chunk_size

★★★★★★★★★ 核心思想:
  → decode tokens已知 → look up profile_table → 找到max chunk_size → 保证decode latency ≤ SLO
  → → ★★★★★★★★★★★★★★★ decode-first → prefill budget由decode workload决定 → 不是固定值!
  → → → ★★★★★★★★★★★★★★★★★★ 这解决了vLLM的根本问题: 长prefill阻塞decode → latency spike!
```

## 参考
- vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py — BudgetRefiner + SchedulerDynamicBatch
- vllm-ascend/vllm_ascend/core/profiling_chunk_predictor.py — ChunkSizePredictor + ProfilingChunkManager
- vllm-ascend/vllm_ascend/ascend_config.py — SLO_limits_for_dynamic_batch config
- vllm SJF RFC #29406 — closed/stale → community interest confirmed
- vllm/v1/core/sched/scheduler.py — standard vLLM V1 scheduler (no SLO-aware scheduling)
- Profile table: vllm-ascend.obs.cn-north-4.myhuaweicloud.com/dynamic_batch_scheduler/A2-B3-BLK128.csv
- Related notes: mindie-production-deployment-reading.md, vllm-v1-scheduler-deep-reading.md, vllm-v1-scheduler-watermark-reading.md
