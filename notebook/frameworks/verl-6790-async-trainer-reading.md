# verl PR #6790: Separate Async Trainer with Switch+Offload Strategies — Source-Level Reading

2026-10-22 补充：[局部更新的指标聚合](#局部更新的指标聚合)固定于 verl `1876b06d`，区分日志归约与实际计时接线。下方 #6790 的架构、PR 状态及性能判断是 2026-06-18 历史阅读，不属于本次 CPU 验证范围。

2026-10-25接续：[异步空闲指标与调度等待](#异步空闲指标与调度等待)固定主线`3c6d9b43`：混合引擎已有借给rollout再收回的实现，不能沿用六月“永不切换”的结论。采样等待、切换等待、路由请求数与GPU利用率分开，本次仅源码核对。

2026-10-23 补充：[会话水位与未消费组上限](#会话水位与未消费组上限)分析未合并draft #8203的准入单位与资源释放；旧warmup与进程内ReplayBuffer说明不自动适用于这条新路径。

2026-10-30 接续：[流式数据加载的缓存重放约束](#流式数据加载的缓存重放约束)固定 #8203 新head `6dc7bd9e`，V1已从拒绝skip改为采样边界直返缓存；异步前缀、恢复队列与缓存文件检查各有不同保证。此候选仍未合并。

> 2026-06-18 | MERGED | Author: Begunner | +17/-16 | 2 files changed
> ★★★★★★★★ CRITICAL: This PR removes the NotImplementedError barrier — separate_async trainer is NOW RUNNABLE
> ★★★★★★★★ Hybrid engine can now DYNAMICALLY switch between trainer/rollout modes (switch_to_trainer / switch_to_rollout)
> ★★★★★★★★ Two independent LLMServerManagers: hybrid (co-located with training) + standalone (disaggregated)
> ★★★★★★★★ `start_rank` parameter avoids Ray named-actor collisions when hybrid and standalone replicas coexist
> ★★★★★★★★ `should_switch_to_rollout()` currently returns False — switch strategies are TODO, coming in future PRs
> ★★★★★★★★ bypass_mode FORCED True — Decoupled PPO not yet supported in separate async
> ★★★★★★★★ Offload strategy TODOs in both switch_to_rollout and switch_to_trainer — "disable auto offload in config and offload according to the switch strategy"
> ★★★★★★★★ RTX 4090 IMPACT: separate_async requires nnodes>0 + n_gpus>0 + non-naive checkpoint backend → NOT viable on single GPU! This is multi-GPU/multi-node only

---

## 1. PR Overview

### 1.1 PR Metadata

| Field | Value |
|-------|-------|
| Title | `[trainer] feat: A runnable separate async trainer` |
| Author | Begunner |
| State | MERGED (closed) |
| Merged at | 2026-06-17T14:25:45Z |
| Base branch | main |
| Head branch | separate-async |
| Additions/Deletions | +17/-16 |
| Files changed | 2 |
| Commits | 2 |

### 1.2 PR Body (verbatim)

> "See above. Currently the colocated group never switches to rollouter, similar to the fully async implementation. Switch and offload strategies will be introduced later."

Key claim: **"currently the colocated group never switches to rollouter"** — the hybrid replicas stay in trainer mode, never switch to rollout mode. Only the standalone replicas handle generation.

### 1.3 What Changed

**File 1: `verl/trainer/ppo/v1/trainer_separate_async.py` (+8/-10)**

| Change | Before | After | Significance |
|--------|--------|-------|-------------|
| NotImplementedError removed | `raise NotImplementedError(...)` | Removed | **NOW RUNNABLE** — was blocked before |
| `super().__init__()` moved | After `bypass_mode` assignment | Before `bypass_mode` assignment | Parent init must complete before config mutation |
| `bypass_mode=True` kept | `self.config.algorithm.rollout_correction.bypass_mode = True` | Same position, moved after super init | bypass_mode STILL forced |
| `LLMServerManager.create()` now passes `start_rank` | No start_rank | `start_rank=hybrid_num_replicas` | Avoid Ray actor name collision |
| `on_validate_end()` removed | Had switch_to_trainer on validate end | Removed | Switch happens in on_sample_end instead |
| `switch_to_rollout()` added TODO | No comment | `# TODO: disable auto offload in config and offload according to the switch strategy` | Offload strategy placeholder |
| `switch_to_trainer()` added TODO | No comment | `# TODO: disable auto offload in config and offload according to the switch strategy` | Offload strategy placeholder |
| `should_switch_to_rollout()` | Not implemented | `return False` | **Always False** — trainer never voluntarily switches to rollout |

**File 2: `verl/workers/rollout/llm_server.py` (+9/-6)**

| Change | Before | After | Significance |
|--------|--------|-------|-------------|
| `LLMServerManager.__init__` adds `start_rank` param | No start_rank | `start_rank: int = 0` | Instance-level default for replica rank offset |
| `_initialize_llm_servers` defaults | `start_rank: int = 0` (parameter default) | `start_rank: int = None` → falls back to `self.start_rank` | Parameter default moves to instance attribute |

---

## 2. Architecture Deep Dive: PPOTrainerSeparateAsync

### 2.1 Class Structure (149 lines)

**File**: `verl/trainer/ppo/v1/trainer_separate_async.py`

```
@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are separate, trainer may switch to rollout if idle.
    2. Partial rollout is enabled.
    """

    __init__(43-61):         Config validation + super().__init__() + bypass_mode=True
    _setup(63-84):           ★★★ Two LLMServerManagers + CheckpointEngineManagers
    get_llm_client(85-87):   Returns FullyAsyncLLMServerClient from standalone manager
    on_init_end(89-92):      Update weights on both checkpoint managers
    on_train_begin(94-98):   Warmup batches from config
    on_validate_begin(100-103): Switch hybrid→rollout if in TRAINER mode
    on_sample_begin(105-108): Switch hybrid→rollout if needed (currently never triggers)
    on_sample_end(110-113):  Switch hybrid→trainer (always, if in ROLLOUT mode)
    on_step_end(115-119):    Periodic weight sync to standalone replicas
    switch_to_rollout(121-126): Update weights → resume → add replicas to balancer + TODO offload
    switch_to_trainer(128-133): Remove from balancer → abort → sleep + TODO offload
    add_replicas_to_balancer(135-140): Add hybrid replicas to standalone's load balancer
    remove_replicas_from_balancer(142-144): Remove hybrid replicas from standalone's load balancer
    should_switch_to_rollout(146-148): return False — ALWAYS False currently
```

### 2.2 Dual LLMServerManager Architecture

★★★★★★★★★ This is the CORE architectural innovation of #6790:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 PPOTrainerSeparateAsync                                  │
│                                                                         │
│  ┌─────────────────── HYBRID ───────────────────────┐                   │
│  │  LLMServerManager (rank 0..N-1)                   │                   │
│  │  → worker_group = actor_rollout_wg (same process) │                   │
│  │  → In-process: training engine fused with rollout │                   │
│  │  → Mode: TRAINER or ROLLOUT (switchable!)         │                   │
│  │  → CheckpointEngineManager (naive backend)        │                   │
│  │     → sync via in-process update_weights()         │                   │
│  └────────────────────────────────────────────────────┘                   │
│                                                                         │
│  ┌─────────────────── STANDALONE ───────────────────┐                   │
│  │  LLMServerManager (start_rank=N, separate GPUs)   │                   │
│  │  → worker_group = None (disaggregated)            │                   │
│  │  → Separate process: dedicated rollout GPUs        │                   │
│  │  → Always in ROLLOUT mode (never switches!)        │                   │
│  │  → CheckpointEngineManager (nccl/nixl/mooncake)   │                   │
│  │     → sync via NCCL/NIXL/Mooncake weight transfer  │                   │
│  └────────────────────────────────────────────────────┘                   │
│                                                                         │
│  ★★★ Global Load Balancer: routes requests to BOTH sets of replicas     │
│     → Hybrid replicas can be added/removed dynamically                   │
│     → Standalone replicas always serve generation requests               │
└─────────────────────────────────────────────────────────────────────────┘
```

**How it works**:

1. `self.llm_server_manager` — HYBRID: initialized in `super()._setup()` with `worker_group=self.actor_rollout_wg`
   - Replicas start at rank 0
   - Uses naive checkpoint engine (in-process weight sync)
   - Can switch between TRAINER and ROLLOUT modes

2. `self.standalone_server_manager` — STANDALONE: initialized in `PPOTrainerSeparateAsync._setup()` with `start_rank=hybrid_num_replicas`
   - Replicas start at rank `N` (hybrid_num_replicas) — avoids Ray named-actor collisions!
   - Uses nccl/nixl/mooncake checkpoint engine (cross-process weight transfer)
   - Always in ROLLOUT mode (always generating)

3. `self.standalone_checkpoint_manager` — connects actor_rollout_wg to standalone replicas via NCCL/NIXL/Mooncake
   - Periodic weight sync every `parameter_sync_step` steps

### 2.3 Switch Strategy: Current State and Future

★★★★★★★★★ The switch mechanism is a DUAL-MODE dynamic resource allocation system:

**Current behavior** (should_switch_to_rollout returns False):

```
Training step timeline:

  on_sample_begin:
    → should_switch_to_rollout() = False
    → Hybrid stays in TRAINER mode
    → Only standalone replicas generate

  on_sample_end:
    → If hybrid was in ROLLOUT mode → switch_to_trainer()
    → But currently hybrid is ALWAYS in TRAINER mode → switch never triggers

  Result: Hybrid NEVER generates. Only standalone replicas do rollout.
```

**Future behavior** (when switch strategy is implemented):

```
Training step timeline (future, when should_switch_to_rollout = True sometimes):

  on_sample_begin:
    → should_switch_to_rollout() checks replay buffer fill level + switch overhead
    → If True: switch_to_rollout() → hybrid becomes rollout → adds to load balancer
    → If False: hybrid stays trainer → only standalone generates

  on_sample_end:
    → If hybrid was in ROLLOUT mode → switch_to_trainer()
    → Remove from balancer → abort → sleep

  Result: Hybrid DYNAMICALLY joins/leaves generation pool based on need.
```

★★★★★★★★★ **Switch strategy TODO** (from PR body and code comments):

The switch strategy should consider:
1. **Replay buffer fill level**: If the buffer has enough samples, trainer should stay in TRAINER mode. If buffer is empty/starving, trainer should switch to ROLLOUT to help generate more data.
2. **Switch overhead**: Sleep/wake cycle costs time. If overhead > benefit of additional generation, don't switch.
3. **Offload strategy**: When switching to rollout mode, should the trainer offload optimizer states/params to CPU? When switching back to trainer mode, should it reload? Currently this is a TODO in both `switch_to_rollout()` and `switch_to_trainer()`.

### 2.4 start_rank: Avoiding Ray Named-Actor Collisions

★★★★★★★★★ This is a subtle but important detail:

When both hybrid and standalone replicas coexist, they each create Ray named actors for their server processes. If both start at rank 0, the second set would collide with the first.

```python
# trainer_separate_async.py:68-71
hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
    config=self.config, start_rank=hybrid_num_replicas
)
```

Example: If hybrid has 2 replicas (ranks 0, 1), standalone starts at rank 2 (ranks 2, 3, ...).

```python
# llm_server.py:414-416 (in _initialize_llm_servers)
self.rollout_replicas = [
    self.rollout_replica_class(
        replica_rank=start_rank + replica_rank,  # ← offset by start_rank!
        ...
    )
    for replica_rank in range(num_replicas)
]
```

The `start_rank` flows from `LLMServerManager.__init__` → `_initialize_llm_servers` → `RolloutReplica(replica_rank=...)`.

This is the same pattern used by `FullyAsyncLLMServerManager` (a subclass referenced in the old `_initialize_llm_servers` docstring).

---

## 3. Three Trainer Variants Comparison

### 3.1 V1 Trainer Family

| Trainer | Registry Name | Key Feature | Weight Sync | Generation Mode | Rollout Interruption |
|---------|--------------|-------------|-------------|----------------|---------------------|
| **PPOTrainerSync** | `"sync"` | Synchronous, colocated, no partial rollout | naive (in-process) | sleep → update → wake | NO (full generation before training) |
| **PPOTrainerColocateAsync** | `"colocate_async"` | Async, colocated, partial rollout | naive (in-process) | abort → sleep → update → resume | YES (abort ongoing requests, resume later) |
| **PPOTrainerSeparateAsync** | `"separate_async"` | Async, separate rollout, trainer may switch | naive + nccl/nixl/mooncake | Hybrid: switchable; Standalone: always generating | YES (abort + resume on hybrid; standalone always running) |

### 3.2 Lifecycle Hooks Comparison

| Hook | Sync | ColocateAsync | SeparateAsync |
|------|------|---------------|---------------|
| `on_init_end` | update_weights | update_weights | update_weights (both managers) |
| `on_train_begin` | — | warmup batches | warmup batches |
| `on_validate_begin` | — | — | switch hybrid→rollout if in TRAINER |
| `on_sample_begin` | — | — | switch hybrid→rollout if needed |
| `on_sample_end` | sleep_replicas | abort + sleep | switch hybrid→trainer |
| `on_step_end` | update_weights | update_weights + resume | periodic standalone weight sync |
| `on_validate_end` | — | — | removed (switch in on_sample_end) |

### 3.3 Config Requirements

```yaml
# separate_async config (from ppo_trainer.yaml:218-225)
trainer:
  v1:
    trainer_mode: separate_async
    separate_async:
      num_warmup_batches: 4        # More warmup than colocate_async (1)
      parameter_sync_step: 4       # Sync standalone weights every 4 steps

# Mandatory requirements (from trainer_separate_async.py:43-56):
# 1. train_batch_size == ppo_mini_batch_size (no mini-batch splitting in separate mode)
# 2. rollout.nnodes > 0 (standalone replicas need their own nodes!)
# 3. rollout.n_gpus_per_node > 0 (standalone needs dedicated GPUs)
# 4. checkpoint_engine.backend != "naive" (must use nccl/nixl/mooncake for standalone sync)
```

★★★★★★★★★ **Requirement #4 is CRITICAL**: The standalone checkpoint engine MUST use nccl/nixl/mooncake (NOT naive). Naive only works for in-process sync. The hybrid manager still uses naive internally (in-process), but the standalone manager requires cross-process weight transfer.

---

## 4. Global Load Balancer Integration

### 4.1 How Hybrid Replicas Join/Leave the Load Balancer

★★★★★★★★★ The separate_async trainer dynamically manages which replicas serve generation requests:

```python
# switch_to_rollout (lines 121-126):
def switch_to_rollout(self):
    # TODO: disable auto offload in config and offload according to the switch strategy
    self.checkpoint_manager.update_weights(self.global_steps)       # Sync weights
    self.checkpoint_manager.resume_generation_replicas()             # Resume generation
    self.add_replicas_to_balancer()                                  # ★ Add to standalone's LB
    self.current_mode = HybridEngineMode.ROLLOUT

# switch_to_trainer (lines 128-133):
def switch_to_trainer(self):
    # TODO: disable auto offload in config and offload according to the switch strategy
    self.remove_replicas_from_balancer()                             # ★ Remove from standalone's LB
    self.checkpoint_manager.abort_replicas()                         # Abort in-flight requests
    self.checkpoint_manager.sleep_replicas()                         # Sleep (free GPU memory)
    self.current_mode = HybridEngineMode.TRAINER
```

### 4.2 add_replicas_to_balancer / remove_replicas_from_balancer

```python
# lines 135-144:
def add_replicas_to_balancer(self):
    global_load_balancer = self.standalone_server_manager.global_load_balancer
    servers = dict(
        zip(self.llm_server_manager.server_addresses,
            self.llm_server_manager.server_handles, strict=True)
    )
    ray.get(global_load_balancer.add_servers.remote(servers))

def remove_replicas_from_balancer(self):
    global_load_balancer = self.standalone_server_manager.global_load_balancer
    ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))
```

★★★★★★★★★ **Key insight**: The hybrid replicas JOIN the standalone load balancer — not their own! This means when hybrid is in ROLLOUT mode, its replicas are served alongside standalone replicas, and `FullyAsyncLLMServerClient` can route requests to either set.

### 4.3 GlobalRequestLoadBalancer Features

**File**: `verl/workers/rollout/llm_server.py:46-146`

The load balancer supports:
- **Sticky session**: Maps `request_id → server_id` via LRU cache (multi-turn conversations → same server)
- **Least-loaded selection**: When no sticky session, selects server with fewest in-flight requests
- **Dynamic add/remove**: `add_servers()` and `remove_servers()` for elastic scaling
- **Atomic acquire/release**: Single Ray RPC for acquire (server_id + handle), fire-and-forget for release

This is the SAME load balancer used by `FullyAsyncLLMServerClient` for abort-resume generation.

---

## 5. Weight Sync: Dual CheckpointEngineManager Architecture

### 5.1 Two CheckpointEngineManagers

★★★★★★★★★ The separate_async trainer has TWO checkpoint managers:

```
Manager 1: self.checkpoint_manager (HYBRID)
  → backend = "naive" (always, set in trainer_base.py:269)
  → actor_wg = self.actor_rollout_wg
  → replicas = self.llm_server_manager.get_replicas()
  → Method: in-process update_weights() (ColocatedCheckpointEngine)

Manager 2: self.standalone_checkpoint_manager (STANDALONE)
  → backend = nccl/nixl/mooncake (from config)
  → actor_wg = self.actor_rollout_wg
  → replicas = self.standalone_server_manager.get_replicas()
  → Method: NCCL/NIXL/Mooncake CUDA IPC weight transfer
```

**Why two managers?**: The hybrid replicas are in the same process as the training engine — they sync via naive (in-process) mechanism. The standalone replicas are in separate processes on separate GPUs — they need cross-process CUDA IPC (NCCL/NIXL/Mooncake).

### 5.2 Weight Sync Flow in on_step_end

```python
# trainer_separate_async.py:115-119:
def on_step_end(self):
    if self.global_steps % self.config.trainer.v1.separate_async.parameter_sync_step == 0:
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
```

★★★★★★★★★ **parameter_sync_step=4 default**: Standalone replicas are updated every 4 steps (not every step). This means standalone replicas run on stale weights for up to 4 training steps. The ReplayBuffer handles staleness via `max_off_policy_threshold=8` and `max_off_policy_strategy=drop`.

### 5.3 CheckpointEngineManager.update_weights() Full Flow

**File**: `verl/checkpoint_engine/base.py:469-515`

When backend is NOT "naive":
```
1. abort_replicas()            — abort all in-flight requests (partial rollout)
2. create RayWorkerGroup        — temporary worker group for all replica workers
3. release_kv_cache_replicas()  — free KV cache, keep weights in place
4. build_process_group()        — NCCL/NIXL/Mooncake communication setup
5. update_weights()             — CUDA IPC weight transfer
6. finalize()                   — cleanup communication
7. resume_kv_cache_replicas()   — restore KV cache
8. resume_generation_replicas() — resume aborted requests
```

★★★★★★★★★ **KV cache preservation**: The `release_kv_cache` + `resume_kv_cache` pattern means weights are written directly into existing weight buffers without full sleep/wake. This is similar to the in-process update path but via CUDA IPC.

---

## 6. June 18 Four-PR Architectural Shift

### 6.1 The Four MERGED PRs

| PR | Title | Author | +/− | Files | Merged |
|-----|-------|--------|------|-------|--------|
| **#6790** | [trainer] feat: A runnable separate async trainer | Begunner | +17/-16 | 2 | June 17 14:25 |
| **#6791** | [megatron,doc] feat: support DSV4/GLM5/KimiK2.5 via Megatron Lite | ISEEKYAN | +864/-9 | 9 | June 18 00:12 |
| **#6765** | [worker] feat: add per-step optimizer param overrides | Luosuu | +110/-8 | 2 | June 18 00:25 |
| **#6789** | [data] fix: offload process_vision_info to thread executor | huaiyizhao | +15/-1 | 1 | June 17 14:30 |

### 6.2 How They Form a Coordinated Architectural Shift

★★★★★★★★★ These four PRs form a COMPLEMENTARY set that enables the **full async training architecture**:

```
#6790 (separate_async trainer):
  → Core infrastructure: separate trainer + rollout workers
  → Enables multi-GPU async training with dynamic resource allocation
  → Switch strategies (TODO) will enable hybrid trainer to join rollout when idle

#6791 (Megatron Lite DSV4/GLM5/KimiK2.5):
  → Model support: enables training DSV4, GLM5, KimiK2.5 on separate infrastructure
  → Megatron Lite = experimental, agent-friendly path → plugs into verl via external mlite
  → 256-GPU configs with PP8/EP8/CP8 → multi-node training

#6765 (per-step optimizer overrides):
  → Training control: enables per-step LR/betas/eps/weight_decay overrides
  → Critical for Muon LR scheduling (#6765 enables #6731 CPPO's Muon)
  → OptimStepParams scoped to Tinker worker → explicit optimizer stepping

#6789 (vision thread executor):
  → Concurrency: fixes event loop blocking in agent-loop multimodal processing
  → CPU-bound vision processing (PNG decode + smart_resize) offloaded to thread
  → Unblocks asyncio.gather for batch agent loops → enables agentic rollout (#6779)
```

**The coordinated shift**: Together, these enable:
1. **Separate training infrastructure** (#6790) — trainer and rollout on different GPUs
2. **New model support** (#6791) — DSV4/GLM5/KimiK2.5 training paths
3. **Per-step optimizer control** (#6765) — fine-grained Muon/Adam scheduling
4. **Async agent-loop concurrency** (#6789) — non-blocking multimodal processing

This is the architecture for **fully disaggregated RLHF training** at scale.

### 6.3 Connection to RolloutMode Architecture

The separate_async trainer maps to the existing RolloutMode architecture:

| RolloutMode | Trainer Variant | Process Layout | Weight Sync |
|-------------|----------------|---------------|-------------|
| **HYBRID** | sync, colocate_async | Same process (training + rollout fused) | naive (in-process) |
| **COLOCATED** | (not yet a v1 trainer) | Same placement group, separate process | NCCL/NIXL/Mooncake |
| **STANDALONE** | separate_async (standalone_server_manager) | Separate GPUs, separate process | NCCL/NIXL/Mooncake |

★★★★★★★★★ **The separate_async trainer uses BOTH HYBRID and STANDALONE modes simultaneously**: The `llm_server_manager` creates HYBRID replicas (in-process with training), and the `standalone_server_manager` creates STANDALONE replicas (disaggregated).

This is a novel combination not seen in the legacy (v0) trainer architecture.

---

## 7. Comparison: verl Separate Async vs rLLM Tinker

### 7.1 Architecture Comparison

| Feature | verl SeparateAsync (#6790) | rLLM Tinker |
|---------|---------------------------|-------------|
| **Trainer-rollout communication** | Ray actors + NCCL/NIXL/Mooncake | Client-server SDK |
| **Weight sync** | CUDA IPC (NCCL/NIXL/Mooncake) | Checkpoint → new SamplingClient (zero-copy) |
| **Generation coordination** | GlobalLoadBalancer (sticky + least-loaded) | SyncCoordinator (asyncio, 172 lines) |
| **Off-policy staleness handling** | ReplayBuffer + max_off_policy_threshold | TrajectoryGroupBuffer + NVMe offload |
| **Dynamic resource allocation** | add/remove replicas from balancer | No dynamic allocation (fixed backend) |
| **Switch strategy** | should_switch_to_rollout (TODO, currently False) | No switch (trainer never becomes rollout) |
| **Serving backend** | vLLM/SGLang/TRT-LLM (RolloutReplica registry) | Tinker (in-process) + vLLM/SGLang (external) |
| **bypass_mode** | Forced True (Decoupled PPO TODO) | Required (bypass_mode=True in RTX 4090 config) |
| **parameter_sync_step** | Every 4 steps (configurable) | Single-step (no staleness) |
| **Partial rollout** | Yes (abort + resume) | No (full generation per step) |
| **Agent loop** | Yes (AgentLoopManagerTQ) | No (simple generate → compute reward) |

### 7.2 Key Architectural Differences

★★★★★★★★★ **verl has dynamic load balancing**: The GlobalRequestLoadBalancer can add/remove hybrid replicas at runtime. rLLM Tinker has fixed backend allocation — no dynamic switching.

★★★★★★★★★ **rLLM has simpler weight sync**: Tinker uses `save_weights → new SamplingClient → set_sampling_client` — zero-copy in-process. verl's separate_async uses NCCL/NIXL/Mooncake — cross-process CUDA IPC.

★★★★★★★★★ **verl has partial rollout**: The abort + resume mechanism allows ongoing generation requests to be interrupted mid-stream and resumed after weight sync. rLLM completes all generation before training.

★★★★★★★★★ **verl is multi-node**: separate_async requires nnodes>0 and n_gpus>0 for standalone replicas. rLLM Tinker is single-GPU by default.

---

## 8. RTX 4090 Impact Analysis

### 8.1 Why Separate Async is NOT Viable on RTX 4090

★★★★★★★★★ **CRITICAL**: The separate_async trainer requires:
- `rollout.nnodes > 0` — standalone replicas need SEPARATE nodes/GPUs
- `rollout.n_gpus_per_node > 0` — standalone needs dedicated GPU allocation
- `checkpoint_engine.backend != "naive"` — must use cross-process weight transfer

On a single RTX 4090 (24 GiB), you CANNOT have both:
1. A hybrid engine (training + rollout in same GPU)
2. A standalone engine (separate GPUs for dedicated rollout)

There are not enough GPUs. This mode is designed for multi-GPU/multi-node setups.

### 8.2 RTX 4090 GRPO Training Mode Ranking (Updated)

| Rank | Mode | Trainer | Viability on RTX 4090 |
|------|------|---------|----------------------|
| **#1** | HYBRID colocate_async + bypass | PPOTrainerColocateAsync | VIABLE — best throughput |
| **#1.5** | Tinker in-process | rLLM TinkerBackend | VIABLE — simpler architecture |
| **#2** | HYBRID sync + bypass | PPOTrainerSync | VIABLE — simpler but no partial rollout |
| **NOT VIABLE** | separate_async | PPOTrainerSeparateAsync | NOT VIABLE — requires separate GPUs |
| **NOT VIABLE** | STANDALONE only | (no v1 trainer variant yet) | NOT VIABLE — no training on single GPU |

### 8.3 When Separate Async Becomes Relevant

★★★★★★★★★ **Future RTX 4090 relevance**: If the user has access to a second GPU (e.g., university GPU + matpool GPU), the separate_async architecture could enable:
1. Training on GPU 1 (RTX 4090) — hybrid mode
2. Rollout on GPU 2 — standalone mode
3. Dynamic switching — when training is idle, GPU 1 joins generation pool

But currently both external GPUs are offline (per MEMORY), so this remains theoretical.

### 8.4 Switch Strategy RTX 4090 Implications

If a switch strategy is implemented in the future:

```
★★★★★★★★★ Potential switch strategy for multi-GPU setups:

  GPU 1 (RTX 4090) — HYBRID mode:
    → Default: TRAINER mode (training + weight updates)
    → When replay buffer starved: switch to ROLLOUT (add to balancer, generate)
    → When buffer full: switch back to TRAINER (remove from balancer, train)

  GPU 2 (any GPU) — STANDALONE mode:
    → Always ROLLOUT mode (always generating)
    → Weight sync every parameter_sync_step=4 steps

  Offload strategy:
    → When GPU 1 switches to ROLLOUT: offload optimizer states to CPU (free GPU for KV cache)
    → When GPU 1 switches to TRAINER: reload optimizer states from CPU
    → This is similar to the existing HYBRID sleep/wake but across mode switches
```

---

## 9. MUST DO / MUST NOT

### 9.1 MUST DO (for separate_async, multi-GPU setups)

- Use `checkpoint_engine.backend = "nccl"` or `"nixl"` or `"mooncake"` for standalone — naive WILL NOT WORK
- Set `parameter_sync_step >= 1` — controls how often standalone replicas get weight updates
- Set `bypass_mode = True` — Decoupled PPO not yet supported in separate_async
- Set `num_warmup_batches >= 4` — replay buffer needs initial data before training starts
- Ensure standalone replicas have sufficient GPU resources (`nnodes > 0`, `n_gpus_per_node > 0`)
- Monitor trajectory staleness metrics: `training/off_policy/trajectory_staleness/mean`

### 9.2 MUST NOT

- Do NOT use `separate_async` on single-GPU RTX 4090 — requires separate GPUs for standalone replicas
- Do NOT use `checkpoint_engine.backend = "naive"` for standalone — requires cross-process sync
- Do NOT set `train_batch_size != ppo_mini_batch_size` — assertion will fail
- Do NOT assume hybrid replicas will switch to rollout — `should_switch_to_rollout()` currently returns False
- Do NOT use `Decoupled PPO` with separate_async — bypass_mode forced True, IS correction not implemented
- Do NOT use `vLLM-Ascend` — sleep_level=1 NOT supported, would break the sleep/wake cycle for hybrid replicas

---

## 10. Future Directions: What #6790 Enables

### 10.1 Switch Strategies (explicitly stated as TODO)

The PR body says: **"Switch and offload strategies will be introduced later."**

Potential switch strategies:
1. **Buffer-based**: Switch to rollout when replay buffer fill level < threshold
2. **Overhead-aware**: Only switch if expected generation throughput > switch overhead
3. **Hybrid adaptive**: Mix of buffer level + overhead calculation
4. **Priority-based**: Switch based on generation request queue depth

### 10.2 Offload Strategies (explicitly stated as TODO)

Both `switch_to_rollout` and `switch_to_trainer` have the same TODO comment:

```python
# TODO: disable auto offload in config and offload according to the switch strategy
```

This suggests:
- Currently auto-offload is handled by the training config (param_offload, grad_offload)
- Future: offload should be CONTROLLED BY THE SWITCH STRATEGY — offload optimizer states when switching to rollout, reload when switching back to trainer
- This connects to the existing HYBRID sleep/wake architecture (sleep_level=1/2)

### 10.3 Connection to verl #6794 (Delta Weight Sync)

★★★★★★★★★ **#6794 delta weight sync + #6790 separate_async = COMPLEMENTARY**:

- #6794 reduces standalone weight sync payload by ~100x (bytewise diff)
- #6790 enables standalone weight sync via NCCL/NIXL/Mooncake
- Combined: standalone replicas get weight updates via ~100x smaller payload every `parameter_sync_step` steps
- BUT: #6794 is currently SGLang-only and deferred for LoRA path

### 10.4 Connection to verl #6512 (Per-Unit LoRA Summon)

★★★★★★★★★ **#6512 per-unit summon + #6790 separate_async = COMPLEMENTARY**:

- #6512 enables per-unit LoRA summon (10x peak memory reduction, 60→6-8 GiB)
- #6790 enables separate async trainer (hybrid + standalone replicas)
- Combined: hybrid replicas can use per-unit summon during training, standalone replicas use LoRA adapter for generation
- This would be the optimal path for multi-GPU LoRA GRPO training

### 10.5 Connection to RolloutMode.COLOCATED

★★★★★★★★★ **There is currently NO v1 trainer variant for COLOCATED mode**:

The existing RolloutMode enum has 3 modes (HYBRID, COLOCATED, STANDALONE), but the v1 trainer registry only has:
- `sync` → HYBRID
- `colocate_async` → HYBRID (with async features)
- `separate_async` → HYBRID + STANDALONE (dual)

A COLOCATED-only trainer variant (separate process, same placement group, no weight sync needed) is NOT yet implemented. This would be useful for GRM (LLM as a judge) scenarios where the rollout engine stays in a separate process but shares the same GPU.

---

## 11. Source Files Read

| File | Key Lines | Purpose |
|------|-----------|---------|
| `verl/trainer/ppo/v1/trainer_separate_async.py` | 1-149 (full) | ★★★ Separate async trainer: dual LLMServerManager, switch strategy, HybridEngineMode |
| `verl/trainer/ppo/v1/trainer_base.py` | 1-1507 (full) | PPOTrainer ABC, step() 10-step data flow, _setup() initialization |
| `verl/trainer/ppo/v1/trainer_sync.py` | 1-43 (full) | Sync trainer variant |
| `verl/trainer/ppo/v1/trainer_colocate_async.py` | 1-58 (full) | Colocate async variant |
| `verl/workers/rollout/llm_server.py` | 1-484 (full) | LLMServerManager, GlobalRequestLoadBalancer, LLMServerClient, FullyAsyncLLMServerClient |
| `verl/checkpoint_engine/base.py` | 1-586 (full) | CheckpointEngine ABC, CheckpointEngineManager, naive backend, weight sync flow |
| `verl/trainer/ppo/v1/replay_buffer.py` | 1-237 (full) | ReplayBuffer, TransferQueue KV storage, off-policy staleness management |
| `verl/trainer/config/ppo_trainer.yaml` | 1-471 (full) | separate_async config: num_warmup_batches=4, parameter_sync_step=4 |

## 12. Related PRs and Notes

| Item | Connection |
|------|-----------|
| **verl #6791** (MERGED June 18) | DSV4/GLM5/KimiK2.5 via Megatron Lite — multi-node model support for separate_async |
| **verl #6765** (MERGED June 18) | Per-step optimizer overrides — enables fine-grained LR/Muon scheduling in separate_async |
| **verl #6789** (MERGED June 18) | Vision thread executor — fixes event loop blocking, enables agentic rollout |
| **verl #6794** (OPEN) | Delta weight sync — ~100x payload reduction for standalone weight updates |
| **verl #6512** (OPEN) | Per-unit LoRA summon — 10x memory reduction for hybrid training |
| **verl #6699** (MERGED) | Detach memory fix — 4x reduction (FSDP only, other backends unfixed) |
| **verl #6782** (OPEN) | LoRA rank=64 breaks EOS — MUST use rank=32/alpha=64 |
| **verl #6731** (OPEN) | CPPO — bypass_mode mandatory, connects to #6765 optimizer overrides |
| **rLLM SyncCoordinator** | 172 lines pure asyncio — simpler than verl GlobalLoadBalancer |
| **rLLM TinkerBackend** | Client-server SDK — simpler weight sync than NCCL/NIXL/Mooncake |
| **verl HYBRID sleep/wake** | sleep_level=1/2 — the offload strategy for hybrid replicas |

---

## 局部更新的指标聚合

一个 `global_step` 内的多次局部更新不意味着所有日志值都求平均。固定版本的 [`MetricsAggregator`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/utils.py) 先查精确规则，再按指标名选择归约；多数指标按样本数加权，并非按 token 数加权。这只改变日志汇总，不证明优化器采用了同样的损失归约。

| 指标 | 当前汇总规则 |
|---|---|
| off-policy、filter_groups、rollout_failure 等明确列出的淘汰计数 | 求和；training 和 validation 名称均有精确规则 |
| training/global_step、training/rollout_probs_diff_valid，以及 lr 或以 /lr、_lr 结尾的名称 | 取最后一次记录的标量 |
| timing_s、timing_per_token_ms | 求和 |
| 其他含 max 或 min 的名称 | 分别取最大或最小；max 检查先于 min、sum、mean |
| 其他含 sum、total 的名称 | 求和 |
| mean、avg 或无特殊名称 | 按该 key 出现时的 sample_count 加权；总权重为 0 时退回算术平均 |
| off_policy/evicted_samples_staleness/mean | 均值权重优先采用同次报告的淘汰数，而非保留的 batch 大小 |
| 特定 DAPO 奖励分布字典 | 按奖励值累计 count，保留字典，不作为普通标量丢弃 |

缺席的 key 不补零，也不把该次样本数加入其分母。`global_seqlen/minmax_diff` 在 min/max 都存在时重新用全局 max 减 min，而不是沿用每次差值的最大值。Python `bool` 被忽略，单元素 `torch.bool` Tensor 却会经 `.item()` 转成浮点数；“忽略布尔值”不能泛化到所有载体。多元素 Tensor 和普通非特例字典不进入标量汇总。

名称就是这层接口的一部分。例如合成 key `custom/max_entropy/mean` 仍命中 max，不会因为尾部有 mean 就加权；这不是对现有生产 key 的缺陷报告。新增日志应先核对归约规则与分母，不能只凭指标标签认定统计量含义。

### 比率合并与真实计时接线

合成两次生成：第一次 1 个样本、100 个 response token、100 ms；第二次 3 个样本、3 个 response token、9 ms。原 `compute_timing_metrics` 得到局部每 token 耗时 1 和 3 ms/token。把这两个值交给聚合器，会得到 `timing_per_token_ms/gen=4`；改用默认均值 key 则按样本得到 2.5。两者都不等于合并的 `109/103≈1.058252` ms/token。比率合并需要保留分子、分母，而不是随意选一种平均。[原计时公式](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/metric_utils.py)

但这不能直接定性 V1 训练器的正式计时日志有误。[`trainer_base.py`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py) 的实际顺序是：

1. `step` 创建聚合器，`prepare_metrics` 以默认 0 权重加入；每次 `_step_once` 返回后，用非 padding tag 数作为 `sample_count`，再汇总局部 `iter_metrics`。
2. 多次更新共用 `timing_raw`；计时器的 [`_timer`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/profiler/performance.py) 对同名阶段累加秒数，`step` 合并各次 batch 的 keys/tags 后返回。
3. 训练循环随后调用 `_compute_metrics`，重取合并 batch 并执行 `compute_timing_metrics`。gen 的分母是 response token，其他受支持阶段是 prompt 加 response token；该次调用不经过前面的局部指标聚合器。

这里的计时函数使用完整 `batch`，另一些数据指标使用过滤后的 `metrics_batch`，不要把 `sample_count` 与 token 分母统一解释成“有效样本数”。上述接线只是固定版本源码核对，不是 TransferQueue、padding、计时设备同步或实际吞吐的运行验收。helper 接受任意同名输入的行为，与训练器正式生成该指标的路径必须分别取证。

### CPU 复查范围

[原 19 项测试](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/trainer/ppo/v1/test_metrics_aggregator_on_cpu.py)经 pytest 全部通过，包括样本加权、淘汰数权重、计数求和、末值、缺 key、Tensor/布尔过滤、reset、全局极差重算以及 DAPO 分布字典累计。另 5 组观察通过：全零权重回退、混合 0/正权重、上述比率合并、布尔载体区别、自定义名称优先级。零权重例 `[1,3]` 分别得到算术均值 2，以及权重 `[0,3]` 时的 3。

复查入口：[实验脚本](../../experiments/scripts/verify_verl_metrics_aggregator.py)。Python 3.12.13、torch 2.9.1、NumPy 2.0.2、pytest 8.4.2；运行命令：

```bash
python experiments/scripts/verify_verl_metrics_aggregator.py /path/to/verl
```

脚本从固定 Git 对象选取完整原聚合类、DAPO key 常量和两个原计时函数，注解延后；完整加载原测试模块，包壳只提供这些真实定义，不运行 ReplayBuffer。pytest 隔离 conftest/插件/cache 并限定收集目录；比率例以结构化小对象提供 `batch` 中的真实 CPU 长度 Tensor，并非真实 DataProto。没有运行 trainer/logger、Ray、进程组、模型或 GPU/NPU，也不构成训练/性能或补丁验收。

源码 SHA256：聚合模块 `a0168e86b2ee0872eb5f896d58cfa342e7c9687d7e098c8b4624fe691748412c`，原测试 `5dbeac03bda63b86b7498a4e1504660c4e6c3141ec021aa326e66aca3bcbcdd2`，计时模块 `d857834a6593301ad932ce2fd2e41e781ee140d0f1dafcfe157b97a0625e847f`。官方 API 与本地聚合模块 Git blob 同为 `e9dec3134a3c70b945e6abd03cd5b6e5924fce2e`。

## 会话水位与未消费组上限

保持rollout忙碌与限制积压是两个不同的控制目标。[verl #8203](https://github.com/verl-project/verl/pull/8203)将V1的数据加载、TQ控制平面与AgentLoopManager迁到独立Ray actor，用会话水位补新prompt，再以未消费组数限制积压。2026-10-23读取open/draft、作者Begunner；固定head `b6880198eadd813272290fb1c98ba15e42317071`、目标base及实际祖先 `18cfb74852a053a27dcf301d3318a093be81b628`（ahead22/behind0），39文件3310增2152删。这是已有作者的破坏性重构，不是可直接套用的性能补丁。

### 两个上限的计数单位

[StreamingDataloaderAsync](https://github.com/verl-project/verl/blob/b6880198eadd813272290fb1c98ba15e42317071/verl/trainer/ppo/v1/streaming_dataloader.py)按固定训练fanout `n`计算：目标会话数为`round(watermark_batches × train_batch_size × n)`；pending组占n槽，running组占`n − 已结束会话数`，finished/failure组不占运行槽。会话finished与failure都算释放一槽，不是只统计成功轨迹，也不是必须等整个GRPO组完成才补新prompt。

另外，`separate_async.max_outstanding_batches`换算成prompt组数；pending、running以及尚未采样/淘汰的finished/failure组都计入。补发prompt数是`floor(max(0, 目标会话数 − 未结束会话数) / n)`与剩余组额度的较小值。async拒绝训练时改变单个prompt的n，避免按默认fanout补发却用另一种数量占槽；补发仍是一整组prompt，不发送半组来填最后几个槽。

下面只按原公式手算，不是actor实测：n=2、目标8会话，快照含1个pending、1个running及3个finished组；running的两个会话中已完成1个。因此未结束会话为3，未消费组为5，空闲槽为5。

| 未消费组上限 | 会话水位允许补发 | 组额度允许补发 | 最终新prompt数 |
|---|---|---|---|
| 5组 | 2组 | 0组 | 0 |
| 6组 | 2组 | 1组 | 1 |
| 禁用 | 2组 | 不限 | 2 |

即使补2组也只新增4会话，剩1槽不能独立接半个prompt。相反，组额度已满时，即使没有满水位也会停补。配置要求组上限的batch数不小于watermark_batches；`colocate_async`不设置该组上限，不能把separate模式的约束推广到所有模式。

### 组数不等于陈旧度或内存

运行中长尾组仍只占自己的剩余会话槽，其他组可继续被消费和补发。因此有限未消费组数不严格限定每一组跨过的模型版本数；文档中“接近若干step”是稳态解释，不是逐样本定理。原`drop`策略只淘汰过期finished组；`wait`在pending/running组的`global_steps − prompt_global_steps + 1 >= threshold`时阻止采样。持续补水位与threshold=1的wait组合会一直阻塞，当前初始化明确拒绝，不能以降低积压上限替代该校验。prompt派发跨度与轨迹实际min/max行为版本也不是同一指标。

采样时`_materialize_batch`先移除prompt和session marker，并同步本轮内存快照，从而立即归还组额度；trajectory keys由trainer完成训练并读取后续指标数据后另行清理。**已采样、尚在训练的轨迹不再计入未消费组上限。** 该上限能约束某一类积压，但不是TQ全部key、token字节、GPU KV或系统峰值内存的统一上限。

### 单协调循环与会话结束标记

actor在一轮中复用一次TQ元数据快照，处理终态淘汰与请求，再补水位；原生`_clear_groups`/`_materialize_batch`同时修正本地集合，避免同轮补发继续按旧组数计算。COUNT/WAIT虽不消费训练组，也会进入维护路径并触发淘汰/补发，所以`count_sampleable`不是对系统无副作用的纯查询。validation暂停train维护，load完成前也有门控；“独立actor”不等于所有模式/phase都永不停歇地产生rollout。

[AgentLoopWorkerTQ](https://github.com/verl-project/verl/blob/b6880198eadd813272290fb1c98ba15e42317071/verl/trainer/ppo/v1/agent_loop_tq.py)在每个会话的轨迹写入settle后才发布finished标记；prompt终态又等待全部siblings settle，避免失败组被清理后还有兄弟继续写。manager轮转小批派发的起始worker，避免每次都从worker0开始。这些是源码顺序，不认证真实跨进程故障、取消、传输或负载均衡收益。新的key/tag协议区分prompt、session、trajectory，UID不得含下划线；标准数据入口生成UUID，不把协议前提误写成任意外部字符串已被校验。

### 本机证据与接续范围

本机只对23个新增/修改Python模块执行AST/compile，并从固定Git对象完整加载原`tq_schema.py`及原48行schema测试模块，跳过verl包初始化后直接调用4个参数例，全部通过；未运行pytest收集、流式actor、TQ、Ray或任何模型。schema结果只覆盖key归属/解析及普通字符串tag，不能认证上面的协调和容量设计。可在匹配完整环境用`python -m pytest -q tests/trainer/ppo/v1/test_tq_schema_on_cpu.py`复查原测试；本机隔离调用不等同于这条命令或actor/sampling大套件通过。本轮不为源码手算新建调度器替身或一次性工具。

在上述10-23快照中，作者正文仅写CPU unit tests，未给通过数量；GPU e2e的colocate_async+swe-rebench仍为TODO，未据此推断吞吐、陈旧度或训练收益。真实计数/内存/行为版本条件记入[原M5可选参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动升级原报告环境、套用draft或把它归因为#8250。10-30候选更新见下一节，不将这里的状态沿用到新head。

原SHA256：streaming_dataloader `9a50ea881cd0de5ccb74887224d780ce749f6cdf13016358aa4bbf720f09d90f`、agent_loop_tq `323686bc5a480c8e37439366cb87fe18cc1658d313c97a25c5df999946f4a5ff`、tq_schema `0940a334e4754090f5b114ecf231c70c2b43a457ae0dc507fc5d028ea0ca50eb`、原schema测试 `40bc68e8c5cc7106e3f9936d47460c2b608ccd6e76b12a4e78035ffd565a6c09`。

## 流式数据加载的缓存重放约束

**先把缓存写进共享队列，再让普通采样器挑选，并不能保证取回的正是那批缓存。** [#8203 的新差异](https://github.com/verl-project/verl/compare/8691564c57cb52e00ea68f7cc4efb3c33e8ac4fc..6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96)把skip接到actor的`sample`边界：命中时推进actor自己的dataloader、生成新UID，装入一份缓存inner batch，直接返回对应`KVBatchMeta`，不调用正常采样请求，也不添加prompt finished marker。未命中则照常采样后保存。validation分区、显式传入prompts或非配置step仍走原调用。[原装饰器](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/utils/skip/skip_manager.py#L129)

2026-10-30读取候选为OPEN、非draft、未合并；head `6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96`，目标base `a4b9354968705a441fb7da7367924b15b902e771`。新源码删除了V1全模式拒绝`skip.rollout_tq.enable`的guard，仓内skip／async文档也已改写；PR正文Breaking changes仍写“全模式拒绝”，不能据正文概括当前代码。正文新增的64 GPU、80步及score 55→64属于作者报告，不是本机结果或缓存等价性证据。

### 连续前缀与恢复队列

sync可按配置step独立捕获或重放；async则从**当前训练起始step**筛出连续前缀，整段只选capture或replay，不能交错。`parameter_sync_step`决定每step所需inner目录数；直接采样每次只加载一个inner，超额调用抛错。[原缓存实现](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/utils/skip/rollout_skip.py#L270)

async恢复先读`data.pt`和可用的TQ快照，再配置缓存模式；若选择replay且train分区含任何key，就拒绝继续，而不是把checkpoint中未消费组强行对应到缓存step。空train队列的replay不重发inflight prompt，在置`_train_state_ready=True`及启动coordinator之前完成该选择；前缀内停止watermark refill，前缀后恢复。**该门控不是完整恢复事务**：拒绝发生在读取状态之后，也不保证模型、数据集或缓存内容身份相同。[原恢复顺序](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/trainer/ppo/v1/streaming_dataloader.py#L914)

### 文件存在与可重放不是同一保证

系统Python 3.9.6，从固定Git对象AST提取原`RolloutTqSkip`的10个缓存判断方法，去掉继承／注册器；目录入口由临时路径提供，实际用原Path检查、step解析和模式判断。每step要求两个inner，每个存在性夹具仅创建空`tq_batch.pt`与`meta.json`，**不反序列化、不声称它们是有效缓存**。七个自编控制结果：

| 配置steps与起点 | 文件布局 | action | 原方法结果 |
|---|---|---|---|
| [1,3]，起点1 | 无 | cache | ValueError，不连续 |
| [1,2]，起点1 | 无 | cache | capture |
| [1,2]，起点1 | 两step均完整 | cache | replay |
| [1,2]，起点1 | 仅step1完整 | cache | ValueError，混合命中 |
| [1]，起点1 | 仅inner0 | cache | ValueError，inner不齐 |
| [1,2]，起点2 | 仅step2完整 | cache | replay，忽略起点前配置 |
| [1,2]，起点1 | 仅step1完整 | repeat | replay，step2可解析到step1 |

这里“完整”只表示原检查要求的两个文件均存在。`repeat`的“全命中”由`has_v1_cache`经过最近可用step解析决定，不要求每个目标step都有独立缓存；因此不能将文档的fully cached误读成逐step精确匹配或payload校验。直接重放还会按缓存组顺序映射新UID，组数不足或组内session不足时循环使用，推进游标本身不认证新prompt内容与缓存语义一致。

版本标签也不再一律刷新为当前版本：`global_steps`改为当前step，`min/max_global_steps`加上`当前step − 解析后的缓存step`。例如缓存step2、版本[0,1]用于step5，得到[3,4]，相对年龄与跨度保留；这是源码算式，不代表生成真的使用了step3／4权重。内容等价、随机状态与训练轨迹仍须另证，不能把skip当作一次等价的真实rollout。

五份完整Python输入AST／内存compile通过，含新343行原测试文件；只执行上述七个存在性控制及replay前缀外退出断言，**没有执行原pytest／TQ测试、actor、TensorDict、torch.load、模型或checkpoint恢复**。旧四schema参数例不重跑、不迁作新验证。固定源码SHA256：rollout_skip `fc145c932a129f633c17d52f3349065d48c24a89b5b532bce67481b5e417d278`；skip_manager `1f48cf05db7630cce84534da514f7443060e9704dac0300ed79d8713c2eaa785`；streaming_dataloader `1d63726f1834ef9a1eee84cc6ac2a92de382bbd063d75543a13628e862d9a45c`；trainer_base `a46330c2a529a743ee42e3966f71d308a7311b22d5309cd03340b7c2cab88ff0`；原测试 `7c3f714e3a2ec5772767939dd5844ba7cc1397301c0cd6850bae0264d9321e7e`。这是本机缓存契约学习，不新增硬件任务或改M5性能专项。

## 异步空闲指标与调度等待

**等待训练数据不等于整组GPU没有工作，路由中有请求也不等于推理kernel满载。** [#7601](https://github.com/verl-project/verl/issues/7601)请求给V1 `separate_async`补trainer／rollouter idle ratio。2026-10-25读完整正文和三条评论：hsusul已提出路由副本时间积分方案；报告者[mikequan0425](https://github.com/verl-project/verl/issues/7601#issuecomment-5505514047)指出混合引擎在ROLLOUT模式时不应计为trainer资源空闲，并倾向rollouter只统计standalone副本；协作者[tardis-key](https://github.com/verl-project/verl/issues/7601#issuecomment-5505659130)说明当时RL-Insight提供时间线而未计算这两项数值。后者是2026-09-02的说明，不是本轮对所有RL-Insight版本的审计。编号PR搜索为空不代表无人推进，不另写计量实现或替讨论参与者定案。

### 当前计时与状态的含义

固定源码为`3c6d9b43b67c74ec343d77874cd8fa6e841523da`。在[V1训练循环](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/trainer_base.py#L474)中，每个global step清空`timing_raw`，多个局部更新共享该字典；标准计时实现对同名阶段累加，NVTX／MSTX入口同样委托该实现，平台插件的自定义计时不在本次范围。正式`timing_s/*`由`compute_timing_metrics`在step完成后生成，不是直接把旧fully_async的比率函数装到V1。

| 观测 | 固定源码的含义 | 不能替代 |
| --- | --- | --- |
| `timing_s/gen` | `_step_once`中的`on_sample_begin → replay_buffer.sample → metadata更新 → on_sample_end`整段累计 | 纯sample阻塞秒数，或混合GPU空闲比例 |
| `timing_s/switch_wait` | 混合引擎仍在ROLLOUT模式时等待足够sampleable组，随后才切回trainer | trainer资源完全闲置时间 |
| `separate_async/switch/idle` | `_step_sample_wait_seconds > poll_interval`得到的布尔决策，写为float | 按时间积分的idle ratio或硬件采样率 |
| 路由器`get_status` | 当前每副本inflight请求数、总数与注册副本数 | 历史空闲副本时间、后端排队／token调度槽或GPU利用率 |
| 旧`fully_async/trainer/idle_ratio` | 独立旧聚合器用汇总gen秒数除以step秒数 | 已发布的V1 idle指标，或跨模式等价的资源利用率 |
| 旧`fully_async/rollouter/idle_ratio` | `reset_staleness`按版本窗口与最后idle起点计算尾段等待；另有active-task/capacity积分指标 | V1 standalone副本空闲比例，或全部空闲区间已累加的保证 |

V1的`step`计时包括step钩子、训练、到期checkpoint保存及step末同步，validation、指标计算与日志写出在外。因此即使自行算gen/step，保存／同步开销也会改变分母，不能仅据比率升降定位生成速度。当前V1目录与默认router没有`idle_ratio`或`resource_utilization`字段命中；这是该固定源码范围的检索结论，不涵盖插件或外部监控。

### 混合资源与路由请求的边界

当前[混合切换](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/trainer_separate_async.py#L292)会在prepare阶段等待组数，再执行remove／abort／sleep回收混合副本；step末按等待量与实测切换成本决定是否重新借出。等待数据、切换成本和训练计算已经属于不同阶段，不应把`switch_wait`简单加到gen后统称“GPU闲置”。

默认[GlobalRequestLoadBalancer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/rollout/router.py#L133)只存当前计数：acquire递增、release递减、add置零、remove删除。standalone与借出的hybrid共用它；直接对所有注册副本取零值比例会混入不同资源角色，且单点快照没有时长。讨论中的时间积分仍需确定窗口、副本范围与容量变化规则，当前类没有积累这些历史。

[LLMServerClient](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/rollout/llm_server.py#L124)在发生成RPC前acquire，finally中以不等待的RPC提交release；其inflight范围覆盖服务请求生命周期而非纯kernel时间。一个长请求足以令副本“非零”，多轮工具间隙又不等于该会话已消失；partial-resume还会重复经过该生成调用。这里只按调用关系解释计数语义，未验证Ray消息延迟、异常恢复或后端实际忙闲。

### 验证与接续范围

本机对11份完整相关模块做AST／内存compile，未导入verl、解析硬件平台、执行原切换测试、调度器、Ray、TQ、推理服务或训练。旧fully_async的active-task/capacity积分也未运行，不迁移为V1的验证结果。原CPU切换测试入口为`tests/trainer/ppo/v1/test_separate_async_step_switch_on_cpu.py`，本轮只读入口，不报告其通过。

后续调资源应先固定trainer模式、switch配置、实际standalone／hybrid副本集合、同一统计窗口和生成／采样／同步时间线，再区分等待、容量与后端忙碌；这不是新监控方案或新的训练队列。条件只补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，等已有获准匹配观测时独立选用，不抢先启基准或改既定专项。

## Key Findings Summary

★★★★★★★★★ #6790 removes NotImplementedError — separate_async trainer is NOW RUNNABLE (but requires multi-GPU)
★★★★★★★★★ Dual LLMServerManager architecture: hybrid (in-process) + standalone (disaggregated) on separate GPUs
★★★★★★★★★ `start_rank` avoids Ray named-actor collisions between hybrid and standalone replicas
★★★★★★★★★ `should_switch_to_rollout()` returns False — hybrid trainer NEVER switches to rollout currently
★★★★★★★★★ Switch + offload strategies explicitly TODO — future PRs will implement dynamic mode switching
★★★★★★★★★ bypass_mode FORCED True — Decoupled PPO not supported in separate async yet
★★★★★★★★★ parameter_sync_step=4 default — standalone replicas updated every 4 steps (configurable staleness)
★★★★★★★★★ Four June-18 PRs (#6790/#6791/#6765/#6789) form coordinated disaggregated RLHF architecture
★★★★★★★★★ RTX 4090: separate_async NOT viable on single GPU — requires nnodes>0 + n_gpus>0
★★★★★★★★★ Future: switch strategy could enable hybrid trainer to join rollout pool when replay buffer starved
★★★★★★★★★ Offload strategy: optimizer state offload when switching modes — similar to HYBRID sleep/wake
