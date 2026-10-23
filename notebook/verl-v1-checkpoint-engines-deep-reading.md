# verl V1 Checkpoint Engines Deep Reading

**Date**: 2026-07-15
**Source**: volcengine/verl main branch
**Focus**: 6 checkpoint engine architecture, naive engine for RTX 4090 dp=1, weight sync interaction with GRPO training

2026-10-23 修订：原六后端描述是 7 月历史快照，不是当前后端清单或硬件支持矩阵。下文撤回“4090/dp=1 只能 naive”“所有分布式引擎固定增加 4 GiB”“同进程就共享模型权重”的推断；没有本项目显存/性能测量支持这些结论。新读的 [#8355](https://github.com/verl-project/verl/pull/8355) 是**尚未合并的文件系统传输候选**，不是推荐立即采用的稳定后端。

## 文件系统权重同步的发布与清理

[#8355](https://github.com/verl-project/verl/pull/8355) 读取时 OPEN/non-draft，作者 wdl339 已实现，不另开竞争方案。固定 head `7d981cde954fce079d8c973379738d6e8c258ac2`、目标 base `a4b9354968705a441fb7da7367924b15b902e771`，共同祖先 `2bc6ac46043e31855cea194236ff7548c91817c8`，head ahead 5 / behind 43；三点 diff 为 6 文件 311 增 42 删，不能把目标 base 当直接祖先。

### 文件是传输介质而非恢复状态

`CheckpointEngine` 按用途是 trainer→rollout 的权重同步，不是“绝不落盘”的承诺。[FileSystemCheckpointEngine](https://github.com/verl-project/verl/blob/7d981cde954fce079d8c973379738d6e8c258ac2/verl/checkpoint_engine/filesystem_checkpoint_engine.py) 将当前权重导出到共享挂载的 `global_step_<n>`，加载完成后删除；不保存 optimizer、scheduler、RNG、数据游标或完整恢复元数据。HF safetensors 格式本身不把临时传输变成可恢复训练 checkpoint。配置要求所有节点可访问的**挂载路径**，构造器拒绝 `hdfs://...` 等 URI，不是内置 HDFS 客户端。

| 顺序 | 原实现动作 | 所依赖的条件 |
|---|---|---|
| 1 | 每个 actor rank 消费相同权重流，按相同规则切 shard | 名称、次序、shape、dtype 及过滤规则必须一致；不减少原参数流的 collective |
| 2 | shard_index % num_writers 的拥有者写本地临时文件，再顺序复制到共享路径 | 非拥有者仍遍历全部输入和边界；distributed_write=False 也不让其他 rank 提前退出流 |
| 3 | actor barrier 后由 master 复制 index | barrier 前各 writer 的复制必须已完成；跨客户端可见性仍取决于实际挂载 |
| 4 | manager 等 actor export 返回后才 abort rollout、释放 KV，再调用 rollout 加载 | 文件准备阶段可继续生成，不等于加载阶段无暂停或无 KV 生命周期 |
| 5 | manager 等加载 RPC 返回，再 finalize，然后恢复 KV/生成 | 接收生成器读回不等于 server 已正确安装模型；完整确认链另需实测 |

[`export_hf_safetensors`](https://github.com/verl-project/verl/blob/7d981cde954fce079d8c973379738d6e8c258ac2/verl/utils/checkpoint/hf_safetensors.py) 对拥有的 tensor 做 CPU copy，避免参数迭代器复用缓冲区后覆盖已收集的值；可选 FP32→目标 dtype 并保留指定名字的 dtype，排除 tied alias 的名字由调用方提供。FSDP manager 复用该 helper，但仍仅 rank0 写、其他 rank drain；本轮没有执行真实 FSDP 导出或认证模型 tied-key 推导。

每个 tensor 是不可拆的 shard 成员，`max_shard_bytes` 约束已有 shard 的拼入判断，不是序列化文件大小、单个超大 tensor 或总 host/GPU 内存的硬上限。每个 writer 的本地 index 都描述**完整权重映射**，其中许多 shard 不在该 writer 的临时目录；只在所有 shard 汇合之后才能当完整模型使用。

### 索引最后发布不是原子事务

`_copy_file` 用 `open(dst, "wb")` 和顺序 copy，index 也如此，并非 rename/replace 提交点；源码没有 fsync、内容 checksum、run UUID、自动重试或失败目录回收协议。正常 manager 路径等发送 RPC 完成才要求接收，因此不能仅凭“index 非原子写”就定性健康流程存在竞态；也不能把这一路径当成对任意外部观察者、断电、FUSE 缓存或同 step 重入的事务保证。版本目录名只含 global_steps，多作业隔离与失败重试若要支持，需先界定契约，不能自动提出/发布修复。

receiver 直接读 index 后按唯一文件名逐 shard 打开并 yield tensor，没有轮询缺文件或校验内容 hash；`finalize` 只在 master 且 exported_dir 已记录时删该目录，清理 OSError 记 warning 后清空记录。这是传输文件生命周期，不是保留历史版本的恢复策略。

### 本地文件验证范围

[复查脚本](../experiments/scripts/verify_verl_filesystem_export.py)复用已有 Python 3.12.13 / torch 2.14.1 / safetensors 0.8.0 环境，两次一致通过真实 CPU safetensors 写入/打开/复制/读回：

- 复用可变缓冲区的 8 条输入，过滤 alias 后 7 tensor；BF16 转换及 p4/int64 保留正确，输出 payload 共 104 字节，四 shard 为 24/24/40/16 字节。40 字节 tensor 未按 24 字节预算拆分，实际文件另含格式头；原缓冲最终改成 999 不影响已保存数据。
- 三次**独立、顺序的本地 writer 调用**输出 2/1/1 个互不重叠 shard，各 index 相同、全部输入被消费；用原复制函数汇合后，原 receiver CPU 读回 7 tensor 的值/dtype 完全相同。它们不是三个并行 rank，没有 collective 或共享挂载验证。
- 缺 index 与 index 已有但 shard 全缺的目录均抛 FileNotFoundError；空流只写空 index。直接控制 exported_dir 字段时，非 master finalize 不删、master 只删指定临时版本、其他目录保留；URI 被拒。这不是 manager 完成链或故障恢复验收。

完整原 exporter 和 filesystem 模块执行；仅原 base 的 Registry/ABC 两类用 AST 隔离 Ray/worker 导入，get_device_name 显式绑定 cpu。未调用 send_weights、barrier、manager、FSDP、server adapter、GPU/NPU 或 HDFS/FUSE；8 个变更模块版本另做 compile（已有文件为祖先/head，新文件仅 head）。默认环境缺 safetensors 后直接复用已有隔离环境，未安装新包、未改变其内容。

```bash
python3.12 -B experiments/scripts/verify_verl_filesystem_export.py /path/to/verl
```

脚本固定对象、禁 lazy fetch 并打印 SHA256；helper 为 `ac8e8060aacb19d5cfa3465cb382d4f33b6823da8be8fb2be1490624881bebbe`，filesystem 模块为 `f903ccdb598cd5208be2e0113584f2555c76363814d24af3bb1d671a88b9ca6e`。作者报告 A100/FSDP/vLLM/HDFS-FUSE、4+4 与 16+16 卡、两种 writer 模式的训练/验证及清理，属于作者证据，不是本机结果；未从图表提取或外推性能。真实挂载可见性、rank 流一致性和 server 加载确认仅补[原 M5 可选参考](../experiments/handoff-m5.md#verl-异步策略一致性补证)，不部署新挂载/环境或启动训练。

---

## 1. Architecture Overview

### 1.1 Dual "Checkpoint" Concepts in verl

verl uses "checkpoint" in **two distinct senses** that must not be confused:

1. **Checkpoint Engine** (`verl/checkpoint_engine/`): The mechanism for **weight synchronization** between trainer (actor) and rollout workers. This is NOT about saving model state to disk for fault tolerance -- it is about transferring updated model weights from training processes to inference processes during live GRPO training.

2. **Checkpoint Manager** (`verl/utils/checkpoint/`): The mechanism for **saving/loading model state to disk** for fault tolerance and training resumption. This is the traditional checkpoint concept (model weights, optimizer state, lr_scheduler, RNG state saved to filesystem).

The **Checkpoint Engine** is the focus of this deep reading. It is the critical mechanism for GRPO training weight sync, and the choice of backend directly impacts RTX 4090 memory overhead and throughput.

### 1.2 Registry Pattern

All 6 checkpoint engines use a **registry pattern** via `CheckpointEngineRegistry`:

```python
class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):  # decorator
        def wrapper(cls):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls
        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        return cls._registry[backend](*args, **kwargs)
```

Registration calls in each engine file:
- `@CheckpointEngineRegistry.register("naive")` -> `ColocatedCheckpointEngine`
- `@CheckpointEngineRegistry.register("nccl")` -> `NCCLCheckpointEngine`
- `@CheckpointEngineRegistry.register("nccl")` -> `HCCLCheckpointEngine` (note: same registry key as NCCL! HCCL overrides NCCL on Ascend)
- `@CheckpointEngineRegistry.register("nixl")` -> `NIXLCheckpointEngine`
- `@CheckpointEngineRegistry.register("kimi_ckpt_engine")` -> `KIMICheckpointEngine`
- `@CheckpointEngineRegistry.register("mooncake")` -> `MooncakeCheckpointEngine`

All engines except naive are imported conditionally (`try/except ImportError`) so they only register when their dependencies are available.

### 1.3 CheckpointEngineConfig (YAML Config)

```python
@dataclass
class CheckpointEngineConfig(BaseConfig):
    _mutable_fields = {"backend"}
    backend: Optional[str] = "naive"          # naive, nccl, nixl, hccl, kimi_ckpt_engine, mooncake
    update_weights_bucket_megabytes: int = 2048  # bucket size in MB
    engine_kwargs: dict = field(default_factory=dict)  # per-backend extra kwargs
    custom_backend_module: Optional[str] = None  # plugin backend registration
```

Default backend is `"naive"` -- this is the **colocated (in-process) weight pass-through** used for sync training.

---

## 2. The 6 Checkpoint Engines: Detailed Analysis

### 2.1 naive (ColocatedCheckpointEngine) for a colocated process

**Class**: `ColocatedCheckpointEngine` registered as `"naive"`

**How it works**: This engine passes a weight generator within the colocated process. Its pass-through method does not itself create a transport group or send/receive bucket. That does not remove collectives in the parameter generator, device copies or memory required by the rollout loader:

```python
class ColocatedCheckpointEngine(CheckpointEngine):
    def send_weights(self, weights, global_steps=None):
        self.weights = weights  # just store the generator reference

    def receive_weights(self, global_steps=None):
        yield from self.weights  # yield from stored generator
        self.weights = None      # cleanup
```

**Key properties**:
- `prepare()`, `init_process_group()`, `finalize()`, `build_topology()` all raise `NotImplementedError` -- no distributed setup needed
- No dedicated send_buf/recv_buf allocation in this pass-through object; not a zero-memory guarantee for the update
- No transport operation in the pass-through itself; the surrounding generator/loader must be examined separately
- Asynchronous but trivially so (just a generator reference)
- Only works when actor and rollout are in the **same process** (colocated)

**Placement and capacity**: A colocated trainer can use this reference-sharing interface. GPU model name or dp=1 alone does not establish placement, memory fit or fastest performance. For a transport that actually allocates two device buckets of B bytes in one process, their payload is 2B; B=2 GiB gives 4 GiB. This conditional calculation is neither every rank's allocation nor a measurement of the entire update. A filesystem candidate has a different host/staging cost model.

### 2.2 nccl (NCCLCheckpointEngine) -- NVIDIA GPU Off-Policy

**Class**: `NCCLCheckpointEngine` registered as `"nccl"`

**How it works**: Uses NCCL collective broadcast to transfer weights from actor rank 0 to all rollout ranks. Communication uses:
- **NCCL broadcast** for tensor data (bucket-based)
- **ZeroMQ PUB/SUB** for bucket metadata (tensor names, shapes, offsets)

**Architecture**:
```
Actor side:
  ME0 (rank 0) -> CE0 -> NCCL broadcast -> rollout ranks 1..N
  ME1..N (rank -1) -> CE1..N (consume weights, no send)
```

**Topology**: Star topology -- actor rank 0 is the master, all rollout ranks receive via NCCL broadcast. Actor ranks >0 are assigned rank=-1 (spectator mode, they just consume the generator to free memory).

**Bucket protocol**:
1. Actor fills `send_buf` (cupy array) with weight chunks
2. After bucket is full, sync CUDA, start `BroadcastOperation` in separate thread
3. Metadata sent via ZMQ PUB/SUB (bucket_meta dict with TensorMeta per tensor)
4. NCCL broadcast of the bucket buffer
5. Double-buffering: swap send_buf/recv_buf for pipelined transfer
6. Rollout receives in recv_buf, yields tensors from send_buf, pipelines next bucket

**Key properties**:
- `bucket_size`: Historical default 2048 MiB. Two allocated buffers of that size contribute 2B of payload; inspect sender/receiver roles rather than assuming both buffers exist in every process. This is not the full update peak.
- Uses **cupy** on master rank to avoid PyTorch expandable_segments memory registration conflict
- `rebuild_group`: Optional rebuild of NCCL group per update (for elastic scaling)
- `rollout_dtype`: Cast weights to target dtype (default bf16)
- ZMQ PUB/SUB for metadata: Master publishes, subscribers receive
- `BroadcastOperation`: Async broadcast in executor thread (non-blocking for the asyncio event loop)

**Elastic scaling**: `rebuild_group=True` destroys and recreates NCCL group each update -- allows adding/removing rollout replicas. Default `False` for fixed clusters.

**Benchmark**: ~7s for Qwen3-30B-A3B, 8.25 GB/s bandwidth on 4x8 H100 with ConnectX-7 400 Gbps InfiniBand.

### 2.3 hccl (HCCLCheckpointEngine) -- Ascend NPU Off-Policy

**Class**: `HCCLCheckpointEngine` registered as `"nccl"` (same registry key! Overrides NCCL on Ascend)

**How it works**: HCCL equivalent of NCCLCheckpointEngine. Uses:
- **HCCL broadcast** (via `StatelessProcessGroup`) for tensor data
- **ZeroMQ PUB/SUB** for bucket metadata
- **torch.npu** device operations

**Key difference from NCCL**: Uses `StatelessProcessGroup` (from vLLM) instead of `ray.util.collective` for HCCL communication. Barrier uses HCCL all_reduce of a signal tensor.

**Note**: This engine registers under the `"nccl"` key, meaning on Ascend NPUs, `backend="nccl"` in YAML actually uses HCCL. This is a design choice for transparent backend selection.

**Benchmark**: ~11s for Qwen3-30B-A3B, 5.3 GB/s on 2x16 Ascend 910C.

### 2.4 nixl (NIXLCheckpointEngine) -- P2P Ring, Elastic, Multi-Backend

**Class**: `NIXLCheckpointEngine` registered as `"nixl"`

**How it works**: Uses NVIDIA's NIXL (NVIDIA Inference Xfer Library) for P2P transfers in a **ring topology**:
- Ring topology: rank 0 -> rank 1 -> rank 2 -> ... -> rank N-1 -> rank N (each rank sends to next, receives from previous)
- Supports multiple transport backends: UCX, UCCL, Mooncake, etc.
- Uses NixlAgent wrapper for registration, metadata exchange, and async transfer operations

**Architecture**:
```
Actor (rank 0) -> NIXL P2P write -> Rollout rank 1 -> NIXL P2P write -> Rollout rank 2 -> ... -> Rollout rank N
                    <- NIXL P2P notification <-         <- NIXL P2P notification <-        <- NIXL P2P notification <-
```

**Key properties**:
- **Ring topology** instead of star/broadcast: Better for elastic scaling (adding/removing nodes only affects local ring segment)
- **High elasticity**: Dynamic ring topology adjustment without rebuilding full process group
- Supports **multiple transport backends** (UCX, UCCL, Mooncake) via NIXL abstraction
- Uses ZMQ async (zmq.asyncio) for metadata exchange between ring neighbors
- `ReadableOperation` and `ReadOperation` for async NIXL RDMA transfers
- Each rank only needs to know prev_agent and next_agent -- minimal topology knowledge
- Double-buffering with send_buf/recv_buf swap for pipelined transfers

**Benchmark**: ~7s for Qwen3-30B-A3B, 8.25 GB/s on 4x8 H100 (same as NCCL on high-bandwidth InfiniBand).

### 2.5 kimi_ckpt_engine (KIMICheckpointEngine) -- Mooncake+Broadcast Hybrid

**Class**: `KIMICheckpointEngine` registered as `"kimi_ckpt_engine"`

**How it works**: Two-phase transfer:
1. **Phase 1 (CPU offload)**: Actor offloads all weights to CPU using ThreadPoolExecutor (32 threads) with non_blocking=True
2. **Phase 2 (P2P + Broadcast)**: Uses Mooncake Transfer Engine's ParameterServer for P2P transfer to a specific rollout worker, then NCCL/HCCL broadcast within rollout subgroup

**Architecture**:
```
Actor ranks: weights -> CPU offload -> register_checkpoint -> gather_metas
Rollout:     parameter_server.receive_tensor -> P2P from actor rank -> broadcast within rollout group
```

**Key properties**:
- **All actor ranks participate** (rank 0..actor_wg_world_size-1), not just rank 0
- Offloads to CPU first: Reduces GPU memory pressure during transfer
- Uses `checkpoint_engine` external package (pip install `checkpoint-engine[p2p]`, version >= 0.4.0)
- Requires Mooncake Transfer Engine (compiled from source for Ascend)
- `ParameterServer` with RDMA registration for P2P transfers
- Bucket-based sharding: weights distributed across actor ranks by tensor_idx % world_size
- Broadcast within rollout subgroup using NCCL/HCCL
- `rollout_ranks` and `rollout_group` for intra-rollout broadcast

**Benchmark**: offload: 7s, update: 3.5s, 16.5 GB/s effective on 2x16 Ascend 910C (highest bandwidth on Ascend due to CPU offload reducing GPU contention).

**Ascend Note**: CANN >= 8.5.0 requires `HCCL_INTRA_ROCE_ENABLE=1` for intra-node RoCE. Not supported on A5 NPUs.

### 2.6 mooncake (MooncakeCheckpointEngine) -- RDMA Ring P2P

**Class**: `MooncakeCheckpointEngine` registered as `"mooncake"`

**How it works**: Uses Mooncake TransferEngine for RDMA P2P transfers in ring topology:
- Ring topology similar to NIXL
- RDMA write for data transfer, RDMA read for receiving
- Uses `StatelessProcessGroup` (from vLLM or SGLang) for topology coordination
- Magic bytes protocol for completion signaling (0xAB, 0xDC, 0xEF, 0x88)

**Architecture**:
```
Rank 0 (actor): fill bucket -> RDMA write to rank 1 -> wait magic completion
Rank 1 (rollout): RDMA read from rank 0 -> forward metadata to rank 2 -> RDMA write magic completion signal to rank 0
...
Rank N-1 (last rollout): RDMA read from rank N-2 -> yield tensors -> RDMA write magic completion
```

**Key properties**:
- Ring P2P topology: rank 0 sends, each subsequent rank reads from previous and forwards metadata to next
- RDMA transfer_sync_read and transfer_sync_write for zero-copy GPU-to-GPU transfers
- Magic completion signal via dedicated `magic_recv` buffer (8 bytes) to prevent data corruption from local GPU side-effects
- Double-buffering: 2 * bucket_size contiguous allocation (`self.buf = torch.empty(2 * bucket_size)`)
- `device_name` filter for Mooncake RDMA device selection
- Supports both NVIDIA (`rdma`) and Ascend (`ascend_direct`) transport
- `StatelessProcessGroup.create` for TCPStore-based coordination

**Benchmark**: 5.93s, 9.44 GB/s on 2x8 H100 ConnectX-7 InfiniBand.

---

## 3. CheckpointEngineManager: The Coordinator

### 3.1 Role in V1 Trainer

`CheckpointEngineManager` is the top-level coordinator that manages the **full weight sync lifecycle** between actor worker group and rollout replicas. It is created in `PPOTrainer._setup()`:

```python
# In trainer_base.py _setup():
checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
checkpoint_engine_config.backend = "naive"  # HARDCODED to naive in base class!
self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
    config=checkpoint_engine_config,
    actor_wg=self.actor_rollout_wg,
    replicas=self.llm_server_manager.get_replicas(),
)
```

**CRITICAL NOTE**: The base `PPOTrainer._setup()` hardcodes `backend = "naive"`. This is overridden in `PPOTrainerSeparateAsync._setup()`, which does NOT hardcode -- it uses the config value. The sync and colocate_async trainers always use naive because actor and rollout are colocated.

### 3.2 update_weights() Flow

The `update_weights()` method in `CheckpointEngineManager` has a **split path** based on backend:

```python
async def update_weights(self, global_steps=None):
    # 0. NAIVE PATH: direct in-process update
    if self.backend == "naive":
        ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
        return

    # 1-8. DISTRIBUTED PATH for nccl/nixl/kimi/mooncake:
    # 1. abort all unfinished requests (partial rollout)
    await self.abort_replicas()
    # 2. create temporary worker group for all replicas
    rollout = RayWorkerGroup(worker_handles=..., ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
    # 3. release kv_cache before weight sync (weights stay in place)
    await self.release_kv_cache_replicas()
    # 4. build process group (prepare + build_topology + init_process_group)
    self.build_process_group(rollout)
    # 5. update weights of all workers
    ray.get(actor_wg.update_weights(...) + rollout.update_weights(...))
    # 6. finalize all workers
    ray.get(actor_wg.execute_checkpoint_engine(["finalize"]*...) + rollout.execute_checkpoint_engine(["finalize"]*...))
    # 7. restore kv_cache after weight sync
    await self.resume_kv_cache_replicas()
    # 8. resume all unfinished requests (partial rollout)
    await self.resume_generation_replicas()
```

**Naive path**: Just calls `actor_wg.update_weights(mode="naive")` -- a single Ray call. No abort, no process group, no kv_cache release/resume.

**Distributed path**: 8-step pipeline requiring KV cache release before weight sync (to free GPU memory for NCCL buffers), process group construction, weight transfer, finalization, and KV cache restoration.

### 3.3 Sleep/Wake Lifecycle

The `CheckpointEngineManager` also manages sleep/wake lifecycle for memory optimization:

```python
async def sleep_replicas(self):
    """Free weight and kv_cache device memory on all rollout replicas."""
    await asyncio.gather(*[r.sleep() for r in self.replicas])

async def wake_up_replicas(self):
    """Recover kv_cache and weights device memory on all rollout replicas."""
    await asyncio.gather(*[r.wake_up() for r in self.replicas])
```

Each trainer mode uses sleep/wake differently:

| Trainer Mode | Sleep/Wake Pattern | When Sleep | When Wake |
|---|---|---|---|
| **sync** | sleep after sampling, wake+update after step | `on_sample_end()` | `on_step_end()` via `update_weights()` |
| **colocate_async** | abort+sleep after sampling, wake+update+resume after step | `on_sample_end()` | `on_step_end()` via `update_weights()` + `resume_generation_replicas()` |
| **separate_async** | hybrid engine switches modes: sleep when switching to trainer, wake when switching to rollout | `switch_to_trainer()` | `switch_to_rollout()` |

### 3.4 KV Cache Release/Resume (Distributed Engines Only)

For distributed checkpoint engines (nccl/nixl/mooncake), the weight sync needs GPU memory for NCCL/RDMA buffers. The solution is `release_kv_cache_replicas()` / `resume_kv_cache_replicas()`:

```python
async def release_kv_cache_replicas(self):
    """Release kv_cache before NCCL weight sync. Weights stay untouched."""
    await asyncio.gather(*[r.release_kv_cache() for r in self.replicas])

async def resume_kv_cache_replicas(self):
    """Restore kv_cache after NCCL weight sync."""
    await asyncio.gather(*[r.resume_kv_cache() for r in self.replicas])
```

This is a **partial sleep**: only KV cache is freed, model weights remain on GPU. This is different from `sleep_replicas()` which frees BOTH weights and KV cache. The partial release is needed because NCCL/RDMA writes directly into the existing weight buffers, so weights must stay resident.

---

## 4. Weight Sync in GRPO Training: Trainer-Specific Flows

### 4.1 PPOTrainerSync (Sync Trainer)

```python
@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)  # initial weight sync

    def on_step_end(self):
        with marked_timer("update_weights", ...):
            self.checkpoint_manager.update_weights(self.global_steps)  # per-step weight sync

    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas()  # free weights+KV cache for training
```

**Flow per GRPO step**:
1. `on_sample_begin()`: Rollout generates prompts (weights+KV cache are resident from wake)
2. `on_sample_end()`: Sleep replicas (free weights+KV cache) -- GPU memory for training
3. Training: Actor update, critic update, reference compute -- uses freed GPU memory
4. `on_step_end()`: `update_weights()` -> naive path -> `actor_wg.update_weights(mode="naive")`

In the **naive path** within `ActorRolloutRefWorker.update_weights()`:
1. Resume rollout weights (if `free_cache_engine=True`: `rollout.resume(tags=["weights"])`)
2. `actor.engine.get_per_tensor_param()` -- get weights as generator from FSDP/Megatron
3. `rollout.update_weights(per_tensor_param, peft_config, ...)` -- direct CUDA weight copy
4. Offload actor to CPU if param_offload enabled
5. Resume KV cache (`rollout.resume(tags=["kv_cache"])`)
6. `base_sync_done = True` for LoRA two-phase sync

**LoRA two-phase sync** (naive path): If `lora.merge=False` and `peft_config is not None`:
- First call: base weight sync only (`base_sync_done=False`)
- Subsequent calls: LoRA adapter sync only (`base_sync_done=True`)
- `sleep_level=1` for LoRA (only adapter changes, base weights stay resident)

### 4.2 PPOTrainerColocateAsync (Colocate Async Trainer)

```python
@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", ...):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()  # resume partial rollout

    def on_sample_end(self):
        self.checkpoint_manager.abort_replicas()    # abort in-flight requests
        self.checkpoint_manager.sleep_replicas()    # free weights+KV cache
```

Same naive backend, but adds **abort/resume** for partial rollout (in-flight requests are paused during weight sync, then resumed).

### 4.3 PPOTrainerSeparateAsync (Separate Async Trainer)

```python
@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    def __init__(self, config):
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive"
        # MUST use nccl/nixl/mooncake for separate async

    def _setup(self):
        super()._setup()
        # Create standalone rollout replicas (NOT colocated with actor)
        self.standalone_server_manager = LLMServerManager.create(...)
        # Create SEPARATE checkpoint engine manager for standalone rollout
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,  # NOT hardcoded to naive!
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )
```

This trainer has **TWO CheckpointEngineManagers**:
1. `self.checkpoint_manager` (naive, base class) -- for **colocated hybrid** replicas
2. `self.standalone_checkpoint_manager` (nccl/nixl/mooncake) -- for **disaggregated standalone** replicas

The hybrid engine switches between trainer and rollout modes:
- `switch_to_rollout()`: wake colocated replicas, update weights, add to load balancer
- `switch_to_trainer()`: remove from load balancer, abort, sleep colocated replicas

---

## 5. Bucket Transfer Protocol Details

### 5.1 Weight Chunking

All engines that transfer over network/IPC use a **bucket-based chunking protocol**. Large tensors that exceed bucket_size are split into chunks:

```python
async def split_weight_chunks(weights, bucket_size):
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(name=name, shape=weight.shape, dtype=weight.dtype,
                                     chunk_offset=chunk_offset, chunk_size=chunk_size, offset=None)
            yield (tensor_meta, buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size
```

And reassembled on the receiving end:

```python
async def merge_weight_chunks(chunks, bucket_size):
    # Small weights: yield directly
    # Large weights: accumulate chunks into pre-allocated buffer, yield when complete
```

### 5.2 TensorMeta

Each weight tensor is described by `TensorMeta`:
```python
@dataclass
class TensorMeta:
    name: str           # weight name (HF key)
    shape: torch.Size   # tensor shape
    dtype: torch.dtype  # tensor dtype (e.g., bf16)
    chunk_offset: int   # byte offset within original tensor
    chunk_size: int     # byte size of this chunk
    offset: int         # byte offset within the bucket buffer
```

### 5.3 Double-Buffering

All network engines use **double-buffering** (send_buf + recv_buf swap):
1. Fill send_buf with current bucket
2. Start transfer of send_buf (broadcast/RDMA/P2P)
3. While transfer is in-flight, swap buffers: recv_buf becomes next send_buf
4. When transfer completes, previous send_buf (now recv_buf) is ready for next bucket
5. Pipeline: transfer bucket N while filling bucket N+1

Double buffering permits overlap in the design; no measured total-time reduction is established here. When both B-sized device buffers are allocated, their payload is 2B, separate from other live memory.

---

## 6. Engine Comparison Summary

Historical transport/bucket sketches below are not a current support or peak-memory matrix. The 2B entries describe two B-sized buffers where actually allocated, not a universal per-rank charge; roles, versions and surrounding gather/loader allocations must be checked separately.

| Engine | Backend | Topology | Hardware | Elastic | Memory Overhead | Best For |
|--------|---------|----------|----------|---------|-----------------|----------|
| **naive** | Generator pass-through | Same-process reference | Loader-dependent | NA | No dedicated transport bucket | Colocated actor/rollout |
| **nccl** | NCCL broadcast + ZMQ PUB/SUB | Star (rank 0 -> all) | NVIDIA GPU | Low (rebuild_group) | 2 * bucket_size | Multi-GPU off-policy, fixed cluster |
| **hccl** | HCCL broadcast + ZMQ PUB/SUB | Star (rank 0 -> all) | Ascend NPU | Low (rebuild_group) | 2 * bucket_size | Ascend NPU off-policy |
| **nixl** | NIXL P2P ring | Ring (rank 0 -> 1 -> 2 -> ... -> N) | Multi-backend | **High** (dynamic ring) | 2 * bucket_size | Elastic rollout, fault tolerance, heterogeneous hardware |
| **kimi_ckpt_engine** | Mooncake P2P + NCCL/HCCL broadcast | P2P to one + broadcast to all | NVIDIA/Ascend | Low (rebuild_group) | CPU offload + 2 * bucket_size | Ascend high-bandwidth, save checkpoint each time |
| **mooncake** | Mooncake RDMA ring P2P | Ring (rank 0 -> 1 -> ... -> N) | NVIDIA/Ascend | **High** (dynamic ring) | 2 * bucket_size | RDMA clusters, elastic, multi-node |

**Benchmark (Qwen3-30B-A3B)**:
| Hardware | Backend | Time (s) | Bandwidth (GB/s) |
|----------|---------|----------|-------------------|
| 4x8 H100, IB 400Gbps | NCCL | ~7 | 8.25 |
| 4x8 H100, IB 400Gbps | NIXL | ~7 | 8.25 |
| 2x8 H100, IB 400Gbps | mooncake | 5.93 | 9.44 |
| 2x16 Ascend 910C | HCCL | ~11 | 5.3 |
| 2x16 Ascend 910C | kimi_ckpt_engine | offload:7 + update:3.5 | 16.5 |

---

## 7. RTX 4090 dp=1 Recommendations

### 7.1 Match backend to placement before hardware

The earlier “ONLY viable” claim is withdrawn. `backend != "naive"` and `rollout.nnodes > 0` are not proofs that training DP must be at least two or that a GPU model cannot use a separated transport. A single physical GPU, actor DP=1 and one rollout worker are different resource statements. Naive needs the colocated pass-through path; a separate layout needs a compatible transport and enough independently accounted resources. No RTX 4090 fit or performance test is established here.

### 7.2 Config Recommendation

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: naive            # historical colocated example, not a dp=1 hardware rule
      update_weights_bucket_megabytes: 2048  # irrelevant for naive, but keep default

trainer:
  v1:
    trainer_mode: sync          # historical example, not a measured optimum
```

### 7.3 Memory accounting requires phase and allocation evidence

The old ~24/~28 GiB fit/OOM table is withdrawn: it combined unmeasured estimates and treated same-process placement as model storage sharing. Account actor parameters/master/optimizer, rollout weights, KV, activations, transient gather/cast and transport allocations at their actual overlapping phases. A generator reference alone neither aliases model storage nor eliminates loader copies. Bucket payload arithmetic is not the full allocator peak; no fixed 7B fit claim follows.

### 7.4 Weight Sync Flow on RTX 4090 (naive)

```
Step N:
  1. on_sample_begin(): Generate prompts (weights + KV cache resident on GPU)
  2. on_sample_end(): sleep_replicas() -> free weights + KV cache
  3. Training phase: actor.update_mini_batch() (uses freed GPU memory)
  4. on_step_end(): checkpoint_manager.update_weights() -> actor_wg.update_weights(mode="naive")
     -> ActorRolloutRefWorker.update_weights():
        a. rollout.resume(tags=["weights"])  # re-allocate weight memory
        b. actor.engine.get_per_tensor_param()  # FSDP summon, yield generator
        c. rollout.update_weights(per_tensor_param)  # direct CUDA copy (same GPU!)
        d. actor offload to CPU (if param_offload)
        e. rollout.resume(tags=["kv_cache"])  # re-allocate KV cache
  5. Step N+1: repeat
```

The key insight: on RTX 4090 with naive backend, the weight sync is a **local GPU memory operation** (allocate -> copy -> free), not a network/IPC transfer. The `rollout.resume(tags=["weights"])` allocates GPU memory for weights, `rollout.update_weights()` copies from FSDP-summoned params into rollout's weight buffers, and then actor can offload its params to CPU.

---

## 8. Checkpoint Recovery (Disk Persistence)

### 8.1 Checkpoint Manager (NOT Checkpoint Engine)

The disk-based checkpoint system (`verl/utils/checkpoint/`) is separate from the checkpoint engine. It handles:

- **Model state**: FSDP sharded state or Megatron dist_checkpointing shards
- **Optimizer state**: Sharded optimizer state
- **Extra state**: LR scheduler, RNG states
- **HF model**: Full HuggingFace format model (via mbridge for Megatron)

`BaseCheckpointManager` provides:
- `save_checkpoint()`, `load_checkpoint()`
- `ensure_checkpoint_capacity()`, `register_checkpoint()` -- manage max_ckpt_to_keep
- `get_rng_state()`, `load_rng_state()` -- RNG state persistence

### 8.2 Resume Modes

```python
# In PPOTrainer._load_checkpoint():
if self.config.trainer.resume_mode == "disable":
    return  # train from scratch
elif self.config.trainer.resume_mode == "auto":
    # find latest checkpoint in default_local_dir
    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
elif self.config.trainer.resume_mode == "resume_path":
    # explicit path to global_step_N directory
    global_step_folder = self.config.trainer.resume_from_path
```

### 8.3 TransferQueue Checkpoint (Async Modes)

For async trainers, PR #7037 (MERGED July 14) adds **TransferQueue checkpointing**:

```python
if self.trainer_mode != "sync" and _tq_supports_checkpoint():
    tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
    if os.path.exists(tq_ckpt_path):
        tq.load_checkpoint(tq_ckpt_path)
```

This allows restarting pending/running prompt groups after resume, preserving finished samples. The `_tq_supports_checkpoint()` function checks for TransferQueue version >= 0.1.9 with `save_checkpoint` and `load_checkpoint` methods.

### 8.4 Reissue Inflight Prompts

After checkpoint recovery, async trainers can **re-issue inflight prompts**:

```python
def _reissue_inflight_prompts(self, partition_id="train"):
    # Find pending/running prompts in TransferQueue
    inflight_uids = [key for key, tag in items.items()
                     if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")]
    # Clear old trajectories for these prompts
    # Re-dispatch as new generation requests
    self.agent_loop_manager.generate_sequences(batch)
```

### 8.5 FSDP Checkpoint Directory Structure

```
checkpoints/${project_name}/${experiment_name}/
  global_steps_${i}/
    actor/
      huggingface/       # config + tokenizer (optional: hf_model weights)
      fsdp_config.json
      model_world_size_N_rank_R.pt
      optim_world_size_N_rank_R.pt
      extra_state_world_size_N_rank_R.pt
    critic/
      ... (same structure)
  latest_checkpointed_iteration.txt
```

---

## 9. Recent Developments

### 9.1 PR #7037 MERGED (July 14): Streaming Dataloader + Async Checkpoint Recovery

- Supports adding arbitrary number of prompts from dataloader without `train_batch_size` limitation
- Improves async trainers with `drop` strategy: dropping stale prompt groups and refilling new prompts
- Allows restarting pending/running prompts after resume via TransferQueue checkpointing
- TransferQueue version >= 0.1.9 required

### 9.2 PR #7041 MERGED (July 15): Qwen2.5-0.5b Fully Async OOM Fix

- Fixes OOM when allocating `recv_buf` in fully async weight sync
- Root cause: recv_buf requests too much memory OR inference engine occupies too much GPU memory
- Solution: Reduce recv_buf size and inference engine memory proportion
- Relevant for RTX 4090: confirms that buffer allocation is a critical memory concern for async modes

### 9.3 Custom Backend Plugin System

The `custom_backend_module` field in `CheckpointEngineConfig` allows registering custom backends:

```python
custom_backend_module: Optional[str] = None  # Python module path
```

Before instantiation, `import_external_libs()` imports this module, which can register additional backends in `CheckpointEngineRegistry`. This enables organization-specific checkpoint engines (e.g., custom RDMA backends, proprietary transfer engines).

---

## 10. Key Findings and RTX 4090 Implications

### 10.1 Backend compatibility follows placement and implementation

Naive is a colocated generator interface, not a hardware exclusivity rule. Separate actor/rollout paths require their own supported transport and resource checks. The open filesystem proposal illustrates a different transport design; it is not proof that any tested training configuration fits a single 24 GiB card.

### 10.2 Weight Sync Memory Budget is Critical

Transport bucket size is one input, not a measured process peak. The earlier fixed weight/KV/activation sum and “naive determines viable vs OOM” conclusion are withdrawn; see section 7.3 for the required accounting boundary.

### 10.3 Sleep/Wake Architecture Matters

The sleep/wake lifecycle for naive backend:
- Sleep: Free BOTH weights and KV cache (full GPU memory available for training)
- Wake: Resume weights first, update, then resume KV cache
- This is the same as the HYBRID sleep/wake pattern documented in verl architecture notes

### 10.4 LoRA Two-Phase Sync

For LoRA GRPO on RTX 4090:
- First step: base weight sync (full model weights transferred)
- Subsequent steps: LoRA adapter sync only (sleep_level=1, base weights stay resident)
- This reduces weight sync payload by ~80x (only adapter deltas, not full model)
- The naive engine handles this seamlessly via `base_sync_done` flag in `update_weights()`

### 10.5 PR #7041 OOM Fix Confirms Memory Pressure

The OOM fix for recv_buf allocation in async mode confirms that buffer sizing is critical. On RTX 4090, even the naive backend's `rollout.resume(tags=["weights"])` allocates weight memory, and the timing of allocation/deallocation must be carefully managed.

---

## 11. Source Files Reference

| File | Purpose |
|------|---------|
| `verl/checkpoint_engine/__init__.py` | Registry exports, conditional imports |
| `verl/checkpoint_engine/base.py` | CheckpointEngine ABC, ColocatedCheckpointEngine (naive), CheckpointEngineRegistry, CheckpointEngineManager, CheckpointEngineWorker, TensorMeta, bucket utilities |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | NCCL broadcast + ZMQ PUB/SUB engine |
| `verl/checkpoint_engine/hccl_checkpoint_engine.py` | HCCL broadcast + ZMQ PUB/SUB engine (Ascend) |
| `verl/checkpoint_engine/nixl_checkpoint_engine.py` | NIXL P2P ring engine |
| `verl/checkpoint_engine/kimi_checkpoint_engine.py` | Mooncake P2P + broadcast hybrid engine |
| `verl/checkpoint_engine/mooncake_checkpoint_engine.py` | Mooncake RDMA ring P2P engine |
| `verl/checkpoint_engine/README.md` | Official documentation, benchmark results |
| `verl/workers/config/rollout.py` | CheckpointEngineConfig dataclass |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker.update_weights() -- naive and distributed paths |
| `verl/trainer/ppo/v1/trainer_base.py` | PPOTrainer._setup() (naive hardcoded), _load_checkpoint(), _save_checkpoint() |
| `verl/trainer/ppo/v1/trainer_sync.py` | PPOTrainerSync (sleep after sample, update after step) |
| `verl/trainer/ppo/v1/trainer_colocate_async.py` | PPOTrainerColocateAsync (abort+sleep, update+resume) |
| `verl/trainer/ppo/v1/trainer_separate_async.py` | PPOTrainerSeparateAsync (assert backend!=naive, two CheckpointEngineManagers) |
| `verl/utils/checkpoint/checkpoint_manager.py` | BaseCheckpointManager (disk persistence) |
| `docs/advance/checkpoint.rst` | Official checkpoint documentation |
