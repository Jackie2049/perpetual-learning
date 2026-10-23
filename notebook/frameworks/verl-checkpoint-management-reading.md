# verl Checkpoint Management 源码级深度阅读

> 2026-06-15 | 源码: verl/utils/checkpoint/ + verl/checkpoint_engine/ + verl/model_merger/ + verl/trainer/ppo/ray_trainer.py
> 核心: 2层架构(CheckpointManager→磁盘持久化+CheckpointEngineManager→实时权重同步); FSDP每rank独立shard+HF export(rank0 gather); Megatron dist_ckpt+async save; CheckpointEngineManager 5后端(naive/NCCL/NIXL/Kimi/Mooncake/HCCL); FSDPModelMerger离线合并→HF; GRPO只存actor+dataloader→省critic

2026-10-22 补充：[保留目录与完整恢复的边界](#保留目录与完整恢复的边界)固定于 verl `1876b06d`。`max_ckpt_to_keep=1` 保存前保留旧目录，成功登记后才删旧；目录保留、保存完成通知与可恢复性是不同保证。下方早期架构段落的版本和行号不自动更新为本次验证范围。

## 1. Checkpoint 2层架构

```
★ ★ verl checkpoint = 2层系统!

Layer 1: CheckpointManager → 磁盘持久化
  → BaseCheckpointManager → abstract
  → FSDPCheckpointManager → FSDP1/FSDP2 per-rank shard + HF export
  → MegatronCheckpointManager → dist_ckpt + mbridge HF export

Layer 2: CheckpointEngineManager → 实时权重同步
  → 训练→rollout GPU 权重传输
  → 5后端: naive/CUDA IPC/NCCL/NIXL/Mooncake/HCCL
  → 不是磁盘checkpoint → 是live weight sync!

★ 关键: CheckpointManager = 保存到磁盘 → 恢复训练 → 持久化
  CheckpointEngineManager = 训练→rollout权重同步 → 不写磁盘 → 实时!
```

## 2. FSDPCheckpointManager — Per-Rank Sharded Save/Load

```
fsdp_checkpoint_manager.py (57-395):

★ ★ Save (220-395): 每rank独立保存

每rank保存3个文件:
  model_world_size_{W}_rank_{R}.pt → sharded model state dict
  optim_world_size_{W}_rank_{R}.pt → sharded optimizer state dict
  extra_state_world_size_{W}_rank_{R}.pt → lr_scheduler + RNG(cpu/numpy/random/cuda)

FSDP1 context:
  with ShardedStateDictConfig(offload_to_cpu=True):
    → FlatParameter shards → torch.save per rank

FSDP2 context:
  with nullcontext(): → DTensor local shards → torch.save per rank
  → FSDP2 state_dict()直接返回DTensor local shard → 不需要ShardedStateDictConfig!

★ ★ HF Export (287-395): rank 0 only + optional

if "hf_model" in checkpoint_save_contents:
  → rank 0: get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)
  → init_empty_weights() → AutoModelForCausalLM.from_config() → 空骨架
  → save_model.save_pretrained(hf_local_path, state_dict=state_dict) → safetensors!

rank 0 always saves:
  huggingface/ → model_config + tokenizer/processor + generation_config

★ LoRA: lora_train_meta.json → {r, lora_alpha, task_type} → 下游恢复PEFT config

★ Load (138-218): 每rank独立加载
  torch.load(model shard) → model.load_state_dict()
  torch.load(optim shard) → optimizer.load_state_dict()
  torch.load(extra shard) → lr_scheduler + RNG恢复
  ShardedStateDictConfig(offload_to_cpu=True) → CPU加载→避免GPU OOM!
```

## 3. MegatronCheckpointManager — Dist Checkpoint + Async Save

```
megatron_checkpoint_manager.py (115-1298):

★ ★ V2 Split Layout (3个sibling目录):

local_path/
  ckpt_contents.json → manifest(最后写入→signals completion)
  transformer_config.json → Megatron TransformerConfig
  model/
    huggingface/ → HF weights via mbridge(可选)
    dist_ckpt/ → Megatron sharded model weights
  optimizer/dist_ckpt/ → optimizer + lr_scheduler shards
  extra/dist_ckpt/ → rng_state shards

★ ★ Async Save:
  → Megatron.AsyncCallsQueue → async_save_request返回
  → finalize callback(manifest写入+HDFS上传+tracker更新)→attached to last request
  → 不阻塞训练! → 大模型checkpoint保存极慢 → async → 继续训练!

★ ★ Save (1169-1298):
  → FullyParallelSaveStrategyWrapper → 多rank并行写
  → _build_model_sharded_state_dict(metadata):
    → model.sharded_state_dict(metadata=model_metadata) → per VP rank
  → Megatron-FSDP: model.state_dict_for_save_checkpoint() → DTensor-aware
  → mbridge HF export: bridge.save_weights() / bridge.save_hf_weights()

★ Load (856-985):
  → dist_checkpointing.load() → FullyParallelLoadStrategyWrapper
  → Megatron-FSDP: load_fsdp_dtensor_checkpoint
  → Legacy v1 layout detection: _raise_for_old_layout() → reject!
```

## 4. CheckpointEngineManager — 实时权重同步

```
checkpoint_engine/base.py (345-515):

★ ★ 架构: Trainer → [NCCL/NIXL/RDMA] → Rollout → ServerAdapter

update_weights() (470-514) 核心流程:
  1. if backend=="naive"(colocated): trainer.update_weights() → 零拷贝generator!
  2. else:
     a. abort_replicas() → 取消所有in-flight rollout请求
     b. 创建临时RayWorkerGroup → 所有replica workers
     c. release_kv_cache_replicas() → 释放KV cache内存(weights留在原位)
     d. build_process_group() → prepare+build_topology+init_process_group
     e. trainer.update_weights() + rollout.update_weights() → NCCL/NIXL send/receive
     f. finalize() → 所有workers
     g. resume_kv_cache_replicas()
     h. resume_generation_replicas()

★ ★ 5+后端实现:
  → ColocatedCheckpointEngine("naive") → 同进程generator零拷贝
  → NCCLCheckpointEngine → NCCL collective send/receive+bucketing
  → NIXLCheckpointEngine → RDMA(SM90/NVLink必需)
  → KIMICheckpointEngine → Kimi专用
  → MooncakeCheckpointEngine → Mooncake
  → HCCLCheckpointEngine → Ascend HCCL

★ ★ Weight提供: engine.get_per_tensor_param() → yields (name, tensor)

FSDP engine实现(transformer_impl.py:794+):
  → LoRA: collect_lora_params() or collect_merged_lora_params() → merge/unmerge
  → FSDP2: model.named_parameters() → param.full_tensor().detach().cpu()
  → Bucketing: split_weight_chunks() / merge_weight_chunks() → 大tensor分bucket

★ ★ vs rLLM Tinker: rLLM用save_checkpoint→new SamplingClient → 磁盘中转!
  verl用CheckpointEngineManager → 实时通信 → 不写磁盘 → 更快!
  但: naive backend = 同进程generator → 等价于rLLM in-process → 极快!
```

## 5. Training Loop Checkpoint Integration

```
ray_trainer.py (974-1108):

★ ★ _save_checkpoint() (974-1041):
  1. actor_rollout_wg.save_checkpoint(actor_local_path, ...)
  2. critic_wg.save_checkpoint(critic_local_path, ...) → only if use_critic!
  3. torch.save(self.train_dataloader.state_dict(), data.pt)
  4. latest_checkpointed_iteration.txt → 同步路径直接open("w")/write；async路径交给worker finalize

★ ★ GRPO只保存:
  → actor checkpoint + dataloader state → 无critic
  → PPO额外保存critic；实际存储比例取决于模型、优化器及save_contents，不能固定算50%或3倍

★ ★ _load_checkpoint() (1043-1108):
  1. 读取latest_checkpointed_iteration.txt → step number
  2. global_steps = int(folder.split("global_step_")[-1])
  3. actor_rollout_wg.load_checkpoint(actor_path)
  4. critic_wg.load_checkpoint(critic_path) → only if use_critic
  5. torch.load(data.pt) → dataloader.load_state_dict()
  6. checkpoint_manager.update_weights(global_steps) → sync到rollout replicas

★ resume_mode 3种:
  → "disable" → 不恢复
  → "auto" → 找最新checkpoint
  → "resume_path" → 指定路径
```

### 保留目录与完整恢复的边界

`max_ckpt_to_keep` 控制已登记目录的淘汰，不是保存事务或磁盘容量上限。`BaseCheckpointManager` 将“保存前腾位置”和“成功后登记”拆开，避免上限为 1 时提前删掉唯一旧目录。[固定版本源码](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/checkpoint_manager.py)

| 上限 | 保存前 `ensure_checkpoint_capacity` | 成功后 `register_checkpoint` |
|---|---|---|
| 1 | 不删除旧目录 | 加入新路径后删除超出的旧目录，只跟踪新目录 |
| 2 及以上 | 已登记数达到上限时，先裁到上限减 1 | 加入新路径，再裁掉超限部分 |
| 0 | 不删除 | 继续记录所有路径，不按数量删除 |

因此，上限为 1 的保存过程中可以同时存在旧目录与正在写入的新目录，需要相应磁盘余量。路径列表从当前 manager 的空列表开始，并非磁盘目录扫描；删除使用 `shutil.rmtree(..., ignore_errors=True)`，列表裁短也不认证磁盘删除已经成功。

`register_checkpoint` 的前提是调用方已完成保存；它不检查模型分片、优化器、manifest 或恢复结果。不能绕过前提，手动登记一个空路径，再据此判定标准保存流程丢失数据。实际接线须与保留 helper 分开看：

- [FSDP manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/fsdp_checkpoint_manager.py)：rank 0 在保存前调用容量清理，在保存流程尾部登记；这些 helper 不是 trainer 的 actor、critic、dataloader 整轮联合提交。
- [Megatron manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/megatron_checkpoint_manager.py)：同步路径先清理，async 路径跳过保存前清理。`_dispatch_finalize` 将回调挂到最后一个 async request，并将所有请求入队；`_finalize_save` 按 manifest、可选 HDFS copy、async rank 0 tracker、登记保留策略的顺序执行。这是源码中的完成通知安排，不是本机对 Megatron 队列、跨 rank 故障或存储持久化的验收。
- [Trainer](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py)：写完 `data.pt` 后，同步路径以 `open(..., "w")` 和 `write` 更新 tracker；actor async 时跳过该更新，由 worker finalize 写入。尽管注释有 `for atomic usage`，这两处写 tracker 都不是临时文件加原子替换；不能据此称整个 checkpoint 保存具备崩溃原子性，也未由此复现恢复故障。

**CPU 复查**：[原 5 项清理测试](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py)全部通过，覆盖上限 1 的保存前保留、保存后删除、上限 2 预清理、0 不限量、连续三轮上限 1。测试创建的内容只有 `checkpoint.txt`，不含模型分片。原 fixture 自带空 model/optimizer 及 rank/world-size monkeypatch；没有初始化进程组。

复现使用[实验脚本](../../experiments/scripts/verify_verl_checkpoint_retention.py)，Python 3.12.13、torch 2.9.1、NumPy 2.0.2、pytest 8.4.2：

```bash
python experiments/scripts/verify_verl_checkpoint_retention.py /path/to/verl
```

脚本从固定 Git 对象选择完整原 `BaseCheckpointManager` 类、延后注解求值并提供真实标准库/torch/NumPy，完整加载原测试模块，由 pytest 执行 5 项原测试及 fixture。包根、框架导入、conftest/外部插件和缓存被隔离；收集限定到脚本目录。临时目录与删除审计限制在本次新建 scratch，退出时清理。未调用 RNG/设备方法或 FSDP/Megatron save/load，不属于完整包、真实 checkpoint 恢复或断电/进程崩溃测试。

源文件 SHA256：manager `be4fa702f0fab63dfb1d482716d31bc13a3b8f5129adbbd61872a169b07cbbc6`；原测试 `30f2df35a31c6eeb21db737557336c9178bb4b69a5427a44621121eaa4a18a18`。

## 6. FSDP2 / DTensor Checkpoint

```
fsdp_utils.py (422-471):

★ ★ Version Detection:
  fsdp_version(model):
    isinstance(model, FSDP) → 1 (FSDP1)
    isinstance(model, FSDPModule) → 2 (FSDP2)
    else → 0

★ ★ FSDP1: ShardedStateDictConfig → FlatParameter shards
★ ★ FSDP2: nullcontext() → DTensor local shards → state_dict()直接返回!

get_fsdp_full_state_dict() (438-471):
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)

★ ★ FSDP2 Sharded Save/Load to CPU (1044-1142):
  fsdp2_sharded_save_to_cpu():
    → iterate all parameters → _local_tensor.detach().cpu() + DTensorSpec
  fsdp2_sharded_load_from_cpu():
    → verify device_mesh consistency → copy _local_tensor back from CPU
```

## 7. FSDPModelMerger — 离线合并→HF格式

```
model_merger/fsdp_model_merger.py (35-):

★ ★ 合并FSDP shards → 标准HF格式:
  1. 读取fsdp_config.json → world_size
  2. 加载rank-0 state dict → 提取DTensor/device_mesh信息
  3. ThreadPoolExecutor并行加载所有rank shards
  4. 合并DTensor shards: Shard(N) → torch.cat(dim=N)
  5. save_pretrained() → HF format(safetensors+config.json+tokenizer)

★ CLI:
  python -m verl.model_merger merge --backend fsdp --local_dir <path> --target_dir <path>
  python -m verl.model_merger merge --backend megatron ...

★ LoRA: lora_train_meta.json → 重建PEFT config → 分离adapter和base weights
```

### 复制分片的存储隔离

离线合并中的`Replicate()`表示取一个副本的值，不自动表示返回值拥有独立存储。[verl #7503](https://github.com/verl-project/verl/pull/7503)将该分支的`return tensors[0]`改为`.clone()`；2026-10-23读取open，作者amanyagami，固定head `e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`、实际祖先 `8df88be746801b6d87c42e15f9a5a0ec1d5eeeae`（ahead2/behind89），两文件104增1删。当前版本只改变复制分支，已撤去此前广泛拒绝共享storage的guard；不能把旧版guard或目标分支的其他变更算进本候选。

[原函数](https://github.com/verl-project/verl/blob/e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b/verl/model_merger/fsdp_model_merger.py)处理的是已从DTensor取出的普通Tensor。`Shard(dim)`仍cat，`Partial()`仍不支持；复制分支仍只选首副本，clone不比较各rank值是否一致。它隔离的是存储所有权，不验证原checkpoint、模型权重绑定或HF文件键完整性。HF保存前另有显式`drop_tied_target_keys`路径，不能仅凭相同storage地址就把两个key判为错误重复。公开评审提出的plain-tensor分支是另一条路径，未给出实际共享storage的checkpoint复现；不把它自动扩张为已证实残余缺陷或另开修复。

#### 共享存储不等于重叠元素

一个缓冲的前四个元素和随后四个元素是两个不重叠view：`data_ptr()`不同，`untyped_storage().data_ptr()`相同。旧分支让输出继续引用源缓冲；更改首view会改源对应元素，但不会改第二view。新分支clone后，两输出与源缓冲各自独立。实际loader的`.bfloat16()`也有条件：输入已是BF16时可原样返回，FP32转BF16会新分配，不能假定所有输入都以相同方式保留别名。

[CPU复查](../../experiments/scripts/verify_verl_merger_storage.py)使用torch2.9.1的真实Tensor和Replicate/Shard/Partial，从固定对象提取整个原merger类，以object替代HF依赖基类，仅调用不读self的合并方法；作者三个测试函数体保持不变。基线2项通过、独立storage断言预期失败；head3项通过。另四组控制覆盖真实torch.save/load的BF16共享view往返、dtype转换、存储字节与源持有者、首副本/Partial规则，两次执行一致。没有原生verl导入、pytest收集、DTensor构造/进程组、完整merger CLI或HF保存；作者19项全套报告不是本机结果。

#### 输出占用与合并峰值

1024元素的BF16源缓冲共2048字节，两段各4元素的输出合计只有16字节有效值。对唯一storage去重计数得到：

| 仍可达的Tensor集合 | 基线共享view | head独立clone |
|---|---|---|
| 只计两个输出 | 2048字节 | 16字节 |
| 源view也仍被持有 | 2048字节 | 2064字节 |

这是实际storage.nbytes的CPU对照，不是RSS、allocator回收或峰值测量。clone可以避免**输出自身**长期保留整个源缓冲，却需要复制时的新分配；完整CLI还要结合源对象何时释放、并行加载、dtype转换和保存阶段判断峰值，不能由第一行直接断言整个合并峰值更低。

[原报告#6259](https://github.com/verl-project/verl/issues/6259)是64卡训练后离线转换出现跨safetensors文件的重复key；作者及公开评审已收窄根因声明。当前CPU控制只证明存储隔离，不复现原文件症状，也不把不同key共享storage等同于文件key重复。已有作者修复覆盖，不新增竞争PR；完整转换首先需要匹配checkpoint及Transformers/HF依赖，离线核验本身不要求重跑64卡训练，不因此新增M5算力任务。

复现：`python -B experiments/scripts/verify_verl_merger_storage.py /path/to/verl-git`，需Python3.12+/torch及本地固定Git对象，无新增依赖。脚本另断言目标base的合并方法AST与祖先相同，完整模块并不相同。原merger SHA256：base `2ae5f3f7fc95871b08a2e185def4f1a5cec09164579cd15eeb2314cb73ef1f0f`、head `29be8e51f7e6bc36a062ff0dbb7a9feb27186a1c3b6cba432c6d8661e0f4d1e1`；原测试 `890253f9060f545948bb9e87cdb8e7bf6d4d8665965bc29e0348db2ac0703c1f`。

## 8. 关键设计洞察

```
1. 2层架构 → 磁盘持久化+实时同步 → 分离关注点!
   → CheckpointManager → 磁盘 → 恢复训练 → 离线合并 → 下游使用
   → CheckpointEngineManager → 实时 → trainer→rollout → 不写磁盘
   → 分离 = 不同需求不同路径 → 磁盘慢但持久 → 实时快但临时 → 设计正确!

2. Per-rank sharded → 不gather完整模型 → 省内存!
   → 每rank保存自己的shard → torch.save → 不需要AllGather
   → Load: ShardedStateDictConfig(offload_to_cpu=True) → CPU加载 → GPU OOM安全!
   → vs consolidated: rank0 gather → 大模型=70GB+ → OOM风险 → sharded更安全!

3. HF export → rank 0 only → 生产部署桥梁!
   → get_fsdp_full_state_dict → rank0_only=True → 只1个rank gather → 省通信
   → init_empty_weights → 空骨架 → 不占GPU内存 → 填入state_dict → save_pretrained
   → safetensors格式 → vLLM/SGLang直接加载 → 生产部署无缝!

4. Megatron async save → 不阻塞训练 → 大模型关键!
   → AsyncCallsQueue → async_save_request → finalize callback → 继续训练
   → ckpt_contents.json最后写入 → completion signal → 检测是否完整
   → vs FSDP: torch.save阻塞 → 大模型=分钟级 → 训练暂停 → 性能损失!

5. CheckpointEngineManager → vs rLLM Tinker → 两种weight sync范式!
   → rLLM: save_checkpoint → disk → new SamplingClient → 磁盘中转 → 简单但慢
   → verl naive: generator零拷贝 → 同进程 → 等价rLLM in-process → 极快
   → verl NCCL: collective send/receive → 跨GPU → 不写磁盘 → 比磁盘快!
   → verl NIXL: RDMA → 跨节点 → 极快 → 需SM90/NVLink → RTX 4090不能用!

6. GRPO checkpoint → 省critic → 50%存储节省!
   → PPO: actor + critic + dataloader → 3×
   → GRPO: actor + dataloader → 2× → 无critic → 省1整套模型存储!
   → 这与GRPO省50%内存/compute一致 → 全系统优化!

7. FSDPModelMerger → 离线工具 → 训练→推理桥梁!
   → 训练产出FSDP shards → 不直接可用于推理 → 需合并
   → merger: ThreadPoolExecutor + torch.cat → 离线合并 → HF format → vLLM直接用
   → 这是训练→推理的完整pipeline → 训练checkpoint → merge → vLLM serve → 闭环!

8. LoRA checkpoint → lora_train_meta.json → adapter独立管理!
   → 训练: collect_lora_params() → 只保存adapter → 极小(~2.6GB for 7B r=16)
   → 合并: lora_train_meta.json → PEFT config → merge adapter into base → or keep separate
   → 部署: LoRA merge → vLLM serve → or vLLM dynamic LoRA loading → 两种路径!
```

---

Sources:
- verl/utils/checkpoint/checkpoint_manager.py (BaseCheckpointManager)
- verl/utils/checkpoint/fsdp_checkpoint_manager.py (FSDPCheckpointManager)
- verl/utils/checkpoint/megatron_checkpoint_manager.py (MegatronCheckpointManager)
- verl/utils/checkpoint/checkpoint_handler.py (CheckpointHandler for SFT)
- verl/checkpoint_engine/base.py (CheckpointEngineManager)
- verl/utils/fsdp_utils.py (FSDP1/FSDP2 state dict helpers)
- verl/trainer/ppo/ray_trainer.py (training loop checkpoint)
- verl/workers/engine/fsdp/transformer_impl.py (FSDP engine save/load)
- verl/model_merger/fsdp_model_merger.py (FSDP shard consolidation)
- Background agent research (verl checkpoint system)
