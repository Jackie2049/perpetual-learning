# verl (HybridFlow) 架构深度分析

2026-10-24 接续：[FSDP 后端配置的权威来源](#fsdp-后端配置的权威来源)区分请求配置、实例化后的engine选择和训练数值差异；原校验函数接真实OmegaConf的局部控制，不是模型或训练验收。

> 基于 verl 源码的架构阅读笔记，重点关注 RL 训练框架设计、vLLM 集成和前缀缓存机制

2026-10-23 接续：[DeepEP 环境边界](#deepep-的包解析与动态库加载)核对固定候选版本的依赖、镜像和验证层级；其余历史架构概括不作为当前后端支持清单。

2026-10-24 接续：[只读模型缓存与测试临时目录](#只读模型缓存与测试临时目录)区分模型输入、生成 checkpoint、配置锁和临时文件系统选择；真实 pytest 目录控制不等于引擎或设备验收。

## 1. 整体架构

verl 是一个面向 RL 训练（PPO/GRPO 等）的分布式框架，核心设计理念：

- **Single Controller 模式**：由一个主控进程编排所有工作流
- **ActorRolloutRef 合体设计**：Actor、Rollout、Ref 模型在同一组 GPU 上复用
- **可插拔 Rollout 引擎**：支持 vLLM、SGLang、TensorRT-LLM
- **DataProto 统一数据协议**：所有组件间通过 DataProto 交换数据

### 核心组件关系

```
                    RayWorkerGroup
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ActorRolloutRef   CriticWorker   RewardWorker
   (训练+推理+参考)    (价值估计)     (奖励计算)
          │
    ┌─────┼─────┐
    │     │     │
  Actor  Rollout Ref
  (训练) (推理) (参考)
         │
    vLLM/SGLang/TRT-LLM
    (推理引擎)
```

## 2. DataProto — 统一数据协议

**文件**: `verl/protocol.py`

```python
class DataProto:
    batch: dict[str, torch.Tensor]       # 张量数据
    non_tensor_batch: dict[str, Any]     # 非张量数据（token IDs, masks 等）
    meta_info: dict[str, Any]            # 元数据
```

关键方法：
- `chunk(n)` — 分割为 n 个子 DataProto
- `concat(list)` — 拼接多个 DataProto
- `to(device)` — 设备迁移
- `fold_batch_dim()` / `unfold_batch_dim()` — 批次维度折叠/展开
- `split(batch_size)` — 按批次大小切分

DataProto 是 verl 中所有数据流动的载体，从 rollout 生成的 trajectories 到训练时的 mini-batch 都用它。

## 3. ActorRolloutRefWorker — 核心 Worker

**文件**: `verl/workers/engine_workers.py`

这是 verl 最关键的设计：将 Actor（训练）、Rollout（推理）、Ref（参考模型）三合一。

### 3.1 角色组合

```python
# 支持的角色组合
roles = ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
```

- **actor_rollout_ref**: 最常用，Actor/Rollout/Ref 共享 GPU
- 好处：权重同步零拷贝（同一进程内的参数更新立即可见）
- Rollout 时用推理引擎（vLLM），训练时用训练框架（FSDP/Megatron）

### 3.2 关键方法

| 方法 | 功能 |
|------|------|
| `compute_log_prob` | Actor 计算当前策略的 log probability |
| `compute_ref_log_prob` | Ref 模型计算参考策略的 log probability |
| `update_actor` | PPO Actor 更新 |
| `update_weights` | 从训练引擎同步权重到推理引擎 |

### 3.3 权重同步机制

```
训练引擎 (FSDP/Megatron) 权重更新
        │
        ▼
update_weights()  ← 权重同步
        │
        ▼
推理引擎 (vLLM) 加载新权重 → 生成新的 rollout
```

## 4. Rollout 引擎集成

**文件**: `verl/workers/rollout/`

### 4.1 架构层次

```
BaseRollout (抽象基类)
    │
    ├── vllm_rollout/ServerAdapter    ← 异步 HTTP 模式
    ├── sglang_rollout/               ← SGLang 后端
    └── trtllm_rollout/               ← TensorRT-LLM 后端
```

### 4.2 vLLM ServerAdapter

**文件**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

verl 使用 vLLM 作为推理引擎的方式有两种：
1. **嵌入式**：直接调用 vLLM Python API（旧模式）
2. **ServerAdapter**：启动 vLLM HTTP Server，通过 API 交互（新模式，推荐）

ServerAdapter 核心流程：

```python
class ServerAdapter:
    def __init__(self):
        # 启动 vLLM HTTP Server
        self.server = vLLMHttpServer(model, ...)
        # 权重传输器
        self.weight_sender = BucketedWeightSender()

    def generate_sequences(self, prompts):
        # 通过 HTTP 发送生成请求
        return self.server.generate(prompts)

    def update_weights(self, model_state_dict):
        # 分桶传输权重到 vLLM
        self.weight_sender.send(model_state_dict)
```

### 4.3 权重传输：BucketedWeightSender

```
训练引擎参数 → 按层分桶 → HTTP/RPC 传输 → vLLM 热更新
```

避免一次性传输整个模型权重，按层分桶传输减少内存峰值。

## 5. 三级前缀缓存机制

**这是 verl 最独特的设计，也是与 prefix-sharing 项目最相关的特性。**

### 5.1 第一级：vLLM 推理层前缀缓存

```yaml
# vLLM 配置
enable_prefix_caching: True
```

- vLLM 的 PagedAttention 支持 prefix caching（Merkle Hash Chain）
- 相同前缀的 KV Cache 块在请求间复用
- 对 RL 训练中相同 prompt 前缀的 rollout 生成直接加速

### 5.2 第二级：负载均衡器 Sticky Sessions

```
请求路由:
  相同 prompt hash → 路由到同一个 vLLM 实例
                     → 复用该实例上的 KV Cache
```

- 当部署多个 vLLM 实例时，负载均衡器使用 sticky session
- 相同前缀的请求总是路由到同一个 GPU
- 确保 KV Cache 前缀复用的命中率

### 5.3 第三级：训练侧 PrefixGrouper

**文件**: `verl/trainer/ppo/prefix_grouper_utils.py`

```python
# 核心功能
class PrefixGrouper:
    """将共享前缀的序列分组，在训练时复用前缀的计算结果"""

    def build_position_ids_for_prefix_grouper():
        """为分组后的序列构建 position IDs"""
        # 共享前缀只计算一次
        # 各序列的不同后缀分别计算

    def build_pg_from_micro_batch():
        """从 micro-batch 构建 prefix group"""
```

训练侧的优化：
- 将共享前缀的 samples 在一个 batch 中分组
- 前向计算时前缀部分只做一次，各样本的 unique suffix 分别计算
- 减少训练时的重复计算

### 5.4 三级缓存协同

```
Prompt: [系统提示 | 用户输入]

推理时 (vLLM):
  ┌─────────────────────────┐
  │ 系统提示 KV Cache 缓存   │ ← 第1级：vLLM prefix caching
  │ 只计算一次，多请求复用    │
  └─────────────────────────┘
  路由到同一 GPU            ← 第2级：sticky session

训练时 (PrefixGrouper):
  ┌───────────┬───────────┐
  │ 共享前缀   │ 后缀 A    │
  │ (计算1次)  │ 后缀 B    │ ← 第3级：训练侧分组
  │           │ 后缀 C    │
  └───────────┴───────────┘
```

## 6. PPO 训练主循环

**文件**: `verl/trainer/main_ppo_sync.py`, `verl/trainer/ppo/ray_trainer.py`

### 6.1 Ray Trainer 流程

```python
# 简化的 PPO 主循环
for epoch in range(num_epochs):
    # 1. 生成 rollout
    trajectories = actor_rollout.generate(prompts)

    # 2. 计算奖励
    rewards = reward_model.compute(trajectories)

    # 3. 计算参考 log prob
    ref_log_probs = actor_rollout.compute_ref_log_prob(trajectories)

    # 4. 计算优势
    values = critic.compute_values(trajectories)
    advantages = compute_gae(rewards, values)

    # 5. PPO 更新
    for ppo_epoch in range(num_ppo_epochs):
        # Actor 更新
        actor_rollout.update_actor(trajectories, advantages, ref_log_probs)
        # Critic 更新
        critic.update_critic(trajectories, values, rewards)

    # 6. 同步权重到推理引擎
    actor_rollout.update_weights()
```

### 6.2 同步模式 vs 异步模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| Sync | Actor 和 Rollout 共置，零拷贝权重同步 | 单节点，GPU 资源充足 |
| Async | Rollout 独立部署，通过 RPC 通信 | 多节点，大规模 |

## 7. 分布式执行模型

### 7.1 Ray 单控制器

**文件**: `verl/single_controller/ray/base.py`

```python
class RayWorkerGroup:
    """管理一组 Ray Actor Worker"""
    def __init__(self):
        self.workers = []  # Ray Actor 列表

    def dispatch(self, data_proto):
        """将 DataProto 分发到各 worker"""
        chunks = data_proto.chunk(len(self.workers))
        futures = [w.process(chunk) for w, chunk in zip(self.workers, chunks)]
        return futures

    def collect(self, futures):
        """收集各 worker 的结果"""
        results = ray.get(futures)
        return DataProto.concat(results)
```

### 7.2 资源管理

```python
class ResourcePoolManager:
    """管理 GPU 资源池，分配 worker 到 GPU"""
    def create_colocated_worker_cls():
        """创建共置 worker（多个角色共享 GPU）"""
```

## 8. 关键文件索引

| 文件 | 内容 |
|------|------|
| `verl/protocol.py` | DataProto 定义 |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker, TrainingWorker |
| `verl/workers/rollout/base.py` | Rollout 基类与工厂函数 |
| `verl/workers/rollout/vllm_rollout/` | vLLM 推理引擎集成 |
| `verl/trainer/ppo/ray_trainer.py` | PPO Ray 训练器 |
| `verl/trainer/ppo/prefix_grouper_utils.py` | 训练侧前缀分组优化 |
| `verl/trainer/main_ppo_sync.py` | 同步 PPO 训练入口 |
| `verl/single_controller/ray/base.py` | Ray WorkerGroup 与资源管理 |
| `verl/single_controller/base/worker.py` | Worker 基类 |

## 9. 学习要点

1. **ActorRolloutRef 三合一** — 避免权重跨进程传输，训练和推理共享 GPU
2. **DataProto 统一协议** — 类似 gRPC protobuf 的设计，但面向 PyTorch 优化
3. **可插拔 Rollout 后端** — 通过工厂模式支持 vLLM/SGLang/TRT-LLM 切换
4. **三级前缀缓存** — 推理层（vLLM）+ 路由层（sticky session）+ 训练层（PrefixGrouper）
5. **Ray 单控制器** — 简化分布式编排，与 vLLM 的多进程模型互补
6. **权重同步** — BucketedWeightSender 分桶传输，减少内存峰值

## 10. 与 prefix-sharing 项目的关系

verl 的三级前缀缓存机制直接对应 prefix-sharing 的核心需求：
- **第1级**：推理时 KV Cache 复用（vLLM 原生支持）
- **第3级**：训练时前缀计算复用（PrefixGrouper）
- 中间的路由优化（第2级）是系统层面的粘合

要深入理解 prefix-sharing，可以：
1. 研究 `prefix_grouper_utils.py` 的具体实现
2. 分析 vLLM 的 prefix caching 与 verl 的集成方式
3. 理解 Prompt 结构如何影响三级缓存的命中率

## DeepEP 的包解析与动态库加载

**能解析出 Python 包、能加载原生库、能完成通信是三种不同结果。** [#8354](https://github.com/verl-project/verl/pull/8354) 为 native Megatron DeepEP 补包和镜像依赖，2026-10-23 读取 OPEN、API draft=false；正文仍称需 GPU 验证和人工审阅，不能只凭非 draft 状态判定已经可用。作者 ETOgaosion 已实现，本项目不另写竞争补丁。固定 head `137438a62f5ab994487e62b1417c8e2f9b8ed307`、base 及祖先 `a4b9354968705a441fb7da7367924b15b902e771`，ahead4/behind0；22 文件98增39删。

### Python 环境和系统库分开提供

[pyproject.toml](https://github.com/verl-project/verl/blob/137438a62f5ab994487e62b1417c8e2f9b8ed307/pyproject.toml)将 `deep-ep==1.2.1` 加入 `megatron` extra，经 `tool.uv.sources` 定向到 Torch2.13/CUDA13 的显式 wheelhouse；其他包不因此全改走该索引。包发行名是 `deep-ep`，导入名是 `deep_ep`。单装 NVSHMEM 不会创建这个 Python 模块。

该包的 dependency-metadata 显式设置 `requires-dist=[]`，由核心/Megatron 栈承担 torch、numpy、einops 与 nvidia-ml-py 等依赖，避免另一 pynvml 发行包争用模块。这是重定义解析元数据，不是 wheel 本身没有依赖或扩展 ABI 已验证；环境组合是否真的提供可用依赖仍属运行期问题。

[uv.lock](https://github.com/verl-project/verl/blob/137438a62f5ab994487e62b1417c8e2f9b8ed307/uv.lock)列出 x86_64/aarch64 × cp311/cp312 四个 wheel URL，但项目的 uv resolution environments 限定 Linux、Python≥3.12、两种 CPU 架构。列有 cp311 文件不等于本项目承诺 Python3.11 支持；aarch64 也是 CPU 平台标签，不是 GPU 型号或跨节点拓扑认证。这四项 lock 记录没有 hash 字段，本轮只核配置文本和路径，未下载核制品字节或 wheel ELF。

[Dockerfile](https://github.com/verl-project/verl/blob/137438a62f5ab994487e62b1417c8e2f9b8ed307/docker/Dockerfile.uv.cu130)另在系统 Python 3.12 路径安装 NVSHMEM3.4.5，设置 `NVSHMEM_DIR`、host库无版本 symlink、ld.so.conf、ldconfig 与 LD_LIBRARY_PATH，使动态加载不依赖每个 uv venv 恰好带同一路径。作者报告检查过 cp312 x86_64 wheel 的 `libnvshmem_host.so.3` 依赖和 RUNPATH；这是作者制品检查，不是本机实测。

系统 NCCL Debian pin 改为 `2.28.9-1+cuda13.0`，锁中的 runtime wheel 仍是2.29.7。源码有意将 base 开发库与训练环境的 runtime wheels 分开；不能据 Debian 后缀不同就判安装失败，也不能据注释分层就认定进程最终加载的是哪一个版本。实际链接/加载路径、ABI及多进程库选择要在目标镜像中查证。

### 配置选择不是通信验收

[安装文档](https://github.com/verl-project/verl/blob/137438a62f5ab994487e62b1417c8e2f9b8ed307/docs/start/install.rst)示例经 `moe_token_dispatcher_type=flex`、`moe_enable_deepep=True`、`moe_flex_dispatcher_backend=deepep` 和 EP group 选择训练端路线。独立 reference model 需要自己的配置；镜像中装有包不代表 trainer 实际使用该 dispatcher，更不代表推理引擎的 DeepEP 或内部 FlashComm adapter 被启用。

| 层级 | 本轮能确认的内容 | 尚不能据此确认 |
| --- | --- | --- |
| 原文件解析 | 两版 pyproject/lock 共4份 TOML 可解析，新增 DeepEP 及 Megatron 路由一致 | uv solver 结果、包下载或安装成功 |
| 镜像与 CI 声明 | NVSHMEM 同为3.4.5；17份变更工作流 YAML 可解析，正文仅 dev4→dev5 替换 | dev5 镜像已发布、CI 已运行通过或容器可启动 |
| 原生加载 | 源码配置了系统库路径；作者另有 wheel 检查报告 | 实际 import、ELF 解析、加载到的库版本 |
| 训练通信 | 文档声明 native Megatron 路线及独立 ref 配置 | dispatch/combine 数值、拓扑支持、吞吐或完整训练 |

本机复用现有 Python3.12.13/PyYAML6.0.3 读取固定 Git 对象；lock 的 package records 除新增 deep-ep 与更新 verl 自身记录外，其余完整记录不变。17份 `.yml` 的新旧文本逐份比较仅上述标签替换；不把整个 `.github` 都宣称已执行或验证。未运行 uv lock/sync、构建/拉取镜像、下载 wheel、导入 native 包或发现设备，复现代码留[当日日志](../../diary/2026-10-23.md)。真实加载/通信缺口沿[原 M5 参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不为此升级环境或新排实验。

## 只读模型缓存与测试临时目录

**生成 checkpoint 不应与预置模型共用输出目录；用例隔离和文件系统容量又是两层问题。** [verl #8343](https://github.com/verl-project/verl/pull/8343) 同时调整测试输出位置和 Ascend CI 环境。2026-10-24 读取为 OPEN、非 draft，作者 lxb007981；固定 head `37f7d0c6b5389bcfd06a455188359c092194ae80`、base/祖先 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，ahead1/behind0，2文件13增2删。已有作者实现，不另写竞争补丁。

### 输入模型与生成模型分开

[model_ascend.yml](https://github.com/verl-project/verl/blob/37f7d0c6b5389bcfd06a455188359c092194ae80/.github/workflows/model_ascend.yml) 的 Prepare weights 将 `~/models` 链到 `/root/.cache/modelscope/hub/models`。旧 `test_critic_engine` 把生成模型写到 `~/models/test_model`，因此输出会跟随这个 symlink 进入模型缓存；`fsdp`、`fsdp2` 两个参数化用例还共用同一目的路径。普通可写缓存中未必报错，但固定输出名仍不提供用例隔离。

[test_engine.py](https://github.com/verl-project/verl/blob/37f7d0c6b5389bcfd06a455188359c092194ae80/tests/models/test_engine.py#L259) 改成 `str(tmp_path / "test_model")`。`create_value_model` 的输入仍由 `get_test_language_model` 选择；模型、tokenizer、config 的三个 `save_pretrained` 调用使用新输出目录。改变的是生成产物，不是移动或复制预置模型缓存。该 Python 测试改动不限定 Ascend，其他运行该测试的平台也会获得 fixture 隔离。

### 三种路径设置各有范围

| 设置 | 控制的内容 | 不提供的保证 |
| --- | --- | --- |
| `tmp_path / "test_model"` | 每个参数化测试各自的输出路径 | 临时文件系统有足够容量，或训练引擎能运行 |
| 步骤级 `PYTEST_DEBUG_TEMPROOT=/root/.cache/temp` | 该 model_engine 步骤的 pytest 临时目录根，避免默认走 `/tmp` | 扩大配额、目录在目标容器可写；也不覆盖前一个直接 `torchrun` 的 checkpoint 步骤 |
| 工作流级 `MEGATRON_CONFIG_LOCK_DIR` | 声明锁目录为 `/root/.cache/temp/megatron-config-locks` | 所用外部模块确实按此创建/获取锁，或锁实现正确 |

固定 head 全仓搜索只在5个工作流找到锁变量，verl Python 源码中未见其消费者；FSDP2 job 另声明 `VERL_USE_EXTERNAL_MODULES=megatron_adaptor`。因此这里能确认 CI 配置意图，不能据此认证镜像内外部适配层的运行行为。新增 CANN 文件与 `npu-smi` 步骤提供诊断信息，也不替代测试结果。

### 本地目录控制

本机 Python3.12.13、pytest9.1.1、PyYAML6.0.3，复用现有环境。静态解析 YAML 并核对上述变量作用位置；新旧 `test_engine.py` 均通过 `compile`。从两版原测试 AST 仅取 `value_model_path` 赋值，放入真实 pytest 的两个参数化用例执行，并在新临时目录写入/读回小文本文件：2 passed。两项旧输出都解析到同一个模拟缓存路径，新输出分别处于指定临时根下的两个独立测试目录，模拟缓存保持空目录。

这个 CPU 控制验证路径计算、symlink 解析和 pytest fixture 行为，不是原 verl 测试、模型序列化、真实只读挂载或 `/tmp` 配额故障复现；未导入 verl、启动 Ray 或执行设备命令。[复现代码与命令](../../diary/2026-10-24.md)保存在当日日志，不新增永久工具或硬件排队事项。这里可复用的排错顺序是先明确每种文件的所有者与生命周期，再分别检查隔离路径、实际挂载/权限/容量及锁消费者，不能用一个“换目录”结论覆盖全部层级。

## FSDP 后端配置的权威来源

**脚本写了FSDP2，不代表实际engine就是FSDP2；修正这一点也不能独自解释loss变化。** [#7540](https://github.com/verl-project/verl/issues/7540)比较v0.9.0与`1dda039b`的多教师OPD首步损失。其唯一后续评论指出嵌套`actor.fsdp_config.strategy=fsdp2`会被顶层默认`actor.strategy=fsdp`覆盖，并将配置问题与数值观察分开。yuchenwang3已有非draft [#7597](https://github.com/verl-project/verl/pull/7597)，2026-10-24读取OPEN/unmerged；固定head`96ad10eb7d52b3775edf925a4fe057fdfd4259bf`，API目标base`3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际共同祖先`9ac85640765b29d3079292fbfbb5c0c130f5c18e`，ahead1/behind24。不能将目标base说成候选已包含的主线，不另认领。

### 校验顺序决定能看见什么

固定候选的[`main_ppo.main`](https://github.com/verl-project/verl/blob/96ad10eb7d52b3775edf925a4fe057fdfd4259bf/verl/trainer/main_ppo.py)先调用`validate_config`，之后才选V1或旧TaskRunner。[新校验](https://github.com/verl-project/verl/blob/96ad10eb7d52b3775edf925a4fe057fdfd4259bf/verl/utils/config.py)是validate_config首个动作，发生在actor配置转dataclass之前：只有`trainer.use_v1=True`且有fsdp_config时，才拒绝既非默认`fsdp`、又不等于顶层strategy的嵌套值。它没有重写配置或根据嵌套键选择engine。

随后[`FSDPActorConfig.__post_init__`](https://github.com/verl-project/verl/blob/96ad10eb7d52b3775edf925a4fe057fdfd4259bf/verl/workers/config/actor.py)仍以顶层值覆盖engine.strategy；候选还在该类构造时独立检查顶层只允许fsdp/fsdp2/fsdp_turbo。这项类级校验不是V1专用冲突guard。嵌套默认fsdp被视为兼容占位，因此顶层fsdp2加嵌套fsdp允许通过；这不等于两个显式冲突选择一概被拒绝。固定main`3c6d9b43`也保留顶层覆盖engine的语句，本轮没有在完整主线配置上复现有效值。

### 原模块的有限 CPU 控制

Python3.12.13、真实OmegaConf2.3.1，完整加载候选`verl/utils/config.py`为独立叶模块；其顶层只导入标准库与OmegaConf，不构造假verl包或替换函数体。七组真实DictConfig直接调用原helper，结果如下；每组执行或抛错后原配置均不变。

| 输入 | helper结果 | 含义 |
| --- | --- | --- |
| V1，顶层fsdp，嵌套fsdp2 | 拒绝 | 不再静默接受嵌套单独选后端 |
| V1，顶层fsdp2，嵌套fsdp | 接受 | 保留嵌套默认占位 |
| V1，两处均fsdp2 | 接受 | 冗余一致合法 |
| V1，顶层fsdp2，嵌套fsdp_turbo | 拒绝 | 非默认冲突 |
| 非V1，顶层fsdp，嵌套fsdp2 | 接受 | 弃用路径不新增此冲突检查，不表示它会选择FSDP2 |
| V1，无fsdp_config | 接受 | 此helper不管其他engine |
| V1，顶层typo，嵌套fsdp | 接受 | 顶层合法性由后续类构造检查，不是helper认证成功 |

本机没有执行完整validate_config、Hydra compose、dataclass构造、TaskRunner或engine；有效engine值是上述赋值源码及作者组合测试的证据，不是本机七组控制的输出。另对actor模块与原CPU测试做compile，V1 launcher通过`bash -n`；无设备命令、训练或模型执行。[脚本、hash与命令](../../diary/2026-10-24.md)留日志，不新增永久工具或配置修复。

### 配置问题与数值问题分开收束

完整7条PR评论及2条review记录显示维护者要求转V1，当前五文件只改共享配置、测试、一个活跃V1 launcher与文档；不能沿用issue早期“改21个launcher”的范围。作者正文15项/12 subtests及后续19项/12 subtests为不同suite报告，本机均未复跑；作者还明确完整配置suite有与main相同的本地模型fixture缺失，不能称整套通过。

#7540评论称两个报告版本的蒸馏实现无diff，并建议固定实际响应IDs及teacher/student logprobs再比较。本轮未独立重做该跨版本diff或训练；相同seed不等于相同采样输入，这个配置修复也不证明原loss差异消失。若M5以后在已有获准工作中遇到相同问题，先记录请求/有效后端与固定响应、路由、logprobs和mask身份，再决定是否需要数值A/B，见[原一致性参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)。缺匹配资源或需新部署/模型下载即记未测，不自动迁移V1、切后端或新增训练队列，不改专项。

## Critic 批处理配置如何进入引擎

**配置必须传到消费者读取的位置，声明字段和默认值本身不保证生效。** [#7462](https://github.com/verl-project/verl/issues/7462)报告 critic 转为裸 `TrainingWorker` 后漏传动态批处理开关和 micro-batch 大小，V1 还把推理 token 预算写到训练预算。已有同报告者 [#7465](https://github.com/verl-project/verl/pull/7465)，2026-10-25 核对 OPEN、非 draft、未合并，head `2609808db4847d0ff0a45b1866e46749afd2f002`；实际父提交／共同祖先 `23a341e361e18063b4ec53e1f7cc4eae68f102cf`，不是 API 目标 base `3f9d4c0d`。十文件160增4删，不另实现。

### 外层配置与消费者默认值

[`TrainingWorker`](https://github.com/verl-project/verl/blob/2609808db4847d0ff0a45b1866e46749afd2f002/verl/workers/engine_workers.py)读取 `config.engine_config`；train／infer 分别用训练／推理预算与 micro-batch 大小，将它们作为 batch 元数据的默认值注入。**batch 已带同名键时不覆盖**，所以 engine 配置正确仍不证明每次调用都采用这些值。

以静态批处理、训练 micro=4、训练预算4096、显式推理预算2048为输入，执行祖先两条原赋值语句，真实配置类所得值如下。这里没有运行 trainer 方法或 worker。

| 接线 | 动态开关 | 训练／推理 micro | 训练／推理 token 预算 |
| --- | --- | --- | --- |
| 原 v0 两条赋值 | True | None／None | 4096／2048 |
| 原 V1 两条赋值 | True | None／None | 2048／2048 |
| 候选 helper，显式推理 micro=2 | False | 4／2 | 4096／2048 |

[`apply_engine_batching`](https://github.com/verl-project/verl/blob/2609808db4847d0ff0a45b1866e46749afd2f002/verl/workers/config/critic.py)统一复制五项，推理 micro 优先显式 infer、其次 forward、最后训练 micro；推理 token 优先非 None 的 infer，再取 forward。动态路径拒绝缺失 token 预算，静态路径拒绝缺失每 GPU micro；弃用的全局 micro 不能直接代替。此处检查 None，不是全部取值合法性、显存容量或后端支持认证。当前原测试名虽含 V1，测试体只调用 helper；v0／V1 两个调用点由源码核对，已弃用 separation 路径不在当前补丁范围。

### YAML 插值与直接构造的区别

候选在公共 [`critic.yaml`](https://github.com/verl-project/verl/blob/2609808db4847d0ff0a45b1866e46749afd2f002/verl/trainer/config/critic/critic.yaml)加入 `ppo_infer_max_token_len_per_gpu: ${.forward_max_token_len_per_gpu}`，DP 配置另让 infer micro 跟随 forward micro。真实 OmegaConf 读取四份完整 generated YAML，仅覆盖并取出批处理字段，再送入原基础 CriticConfig／EngineConfig，得到：训练 micro=4、训练预算4096、forward预算2048时，四份的推理预算均为2048；DP 显式 forward micro=2 时推理 micro=2，Megatron／TorchTitan／VeOmni 三份均为4。显式 infer预算1024则优先于 forward2048。这是字段插值和基础类接线，不是 Hydra 完整组合或四种后端实例化。

直接 `CriticConfig(forward_max_token_len_per_gpu=2048, …)` 不会继承 YAML 插值；未指定 infer 时，它的 dataclass 默认仍是32768，helper保留32768；显式 `ppo_infer_max_token_len_per_gpu=None` 才回退到2048。两种入口的默认语义不同，不能把“forward 参数改了”直接等同于“推理预算改了”，也不据此另认领修复。

### CPU 验证范围

Python3.12.13／OmegaConf2.3.1／pytest9.1.1，原 BaseConfig、CheckpointConfig、ProfilerConfig、OptimizerConfig、EngineRouterReplayConfig、EngineConfig、CriticConfig 七个完整类 AST 原样载入，仅延迟类型注解以避开未用 HFModelConfig 导入，不实例化模型或伪造配置类。原测试模块 imports／断言不改，逐个直接执行六个原测试体，全通过；覆盖静态／动态复制、训练与推理预算分离、micro 回退及两种缺失字段拒绝。不是原生 pytest 收集或完整框架导入。[复现方式与版本](../../diary/2026-10-25.md)留日志；worker／切分模块与两个 trainer 只核源码和语法，无训练、设备或模型下载。

本题收束为配置接线知识，不应用候选、不新增校验器或 M5 算力任务。后续排查同类问题，依次核 YAML 解析值、dataclass 默认、engine 有效字段与 batch 是否显式覆盖；不要从外层配置推导实际执行，更不能从本轮 CPU 结果推导训练收益。

## Meta 加载后的共享参数与梯度

**权重值与前向 loss 相同，不代表模型仍共享同一个可训练参数。** [#7834](https://github.com/verl-project/verl/issues/7834)报告大规模 tied-embedding 模型在各 rank 重复加载造成主机内存压力；已有作者 inin-zou 的 [#7835](https://github.com/verl-project/verl/pull/7835)。2026-10-25核对 OPEN、非 draft、未合并，head `93f5328f0fad3756925337505e7318b18f2bcb62`、父提交／共同祖先 `a9f2985159536a607211dcac730d3f5d55028950`，三文件123增3删；API目标 base `3f9d4c0d` 不能充当候选已包含的主线。旧 [#5746](https://github.com/verl-project/verl/pull/5746)已关闭未合并，不另实现。

### 加载数值与恢复共享是两件事

当前固定 main `3c6d9b43` 的 FSDP engine 仍以 `not tie_word_embeddings` 选择 meta 初始化。候选对 fsdp2 放开 meta 路径，并在 [`fsdp2_load_full_state_dict`](https://github.com/verl-project/verl/blob/93f5328f0fad3756925337505e7318b18f2bcb62/verl/utils/fsdp_utils.py)的 `set_model_state_dict` 之后、单独广播 buffers 之前调用 `tie_weights()`；条件是 config 声明 tied 且模块提供该方法。FSDP1 选择不变。初始化 helper 使用 mesh 最后一维坐标判断物化者，mesh=None 才使用全局 rank，因此“rank0-only”要结合实际 mesh 解释。

本机完整执行原 CPU 测试模块的三个测试函数／四个参数化用例，均通过；不是原生 pytest 收集。模型来自内存中的 tiny LlamaConfig，未下载 checkpoint：两层、hidden128、vocab64，真实 Transformers 模型、Accelerate meta 构造与 PyTorch 前反向。原测试以普通 `load_state_dict` 和逐个 buffer 拷贝代替分布式加载，没有调用完整 verl loader 或 FSDP。

补充同一真实模型、seed0、ids形状2×8的 CPU 控制中，复制相同权重及非持久 buffer 后，未重新绑定的模型与参考模型 logits／loss 逐bit相同，loss为4.05669641494751，但输入 embedding 梯度不同。记共享梯度为 `g_shared`、独立输入／输出梯度为 `g_input`／`g_output`，实测 `g_shared ≈ g_input + g_output`；这不表示两个分量相同或固定“减半”。

| 模型状态 | 输入与输出为同一 Parameter | 输入 embedding 梯度范数 |
| --- | --- | --- |
| 普通 CPU 参考模型 | 是 | 4.465865 |
| meta 物化并复制权重，未重绑 | 否 | 3.190585 |
| 执行候选原重绑条件 | 是 | 与参考梯度 assert_close |

未重绑时输出头梯度范数3.150110；范数不能直接相加，只有向量和与共享梯度相符。本控制没有 optimizer 更新或跨 rank 通信，不能推导完整训练轨迹一致。

### Meta 地址相等不是共享证据

原测试 [`_is_tied`](https://github.com/verl-project/verl/blob/93f5328f0fad3756925337505e7318b18f2bcb62/tests/utils/test_fsdp2_tied_weights_on_cpu.py)只比较 `data_ptr()`。在本机 torch2.14.1／Transformers5.20.2／Accelerate1.15.0，原 `init_empty_weights()` 构造后两 Parameter 已不是同一对象，但地址均为0，因此原“仍在 meta 时共享”的断言通过。将真实 LlamaConfig 明确设为 untied 后，同一原判据仍返回 True。这是 **meta 前置断言的假阳性**，不是四个 CPU 用例完全无效，也不否定物化后恢复梯度的结果。

[PyTorch meta 文档](https://docs.pytorch.org/docs/2.14/meta.html)说明该设备只记录元数据，`to_empty()`留下未初始化参数。本机安装源码中 Accelerate 默认 `include_buffers=False` 的参数注册包装会重新创建 Parameter，不能把这条路径的身份变化全归因于 `to_empty`。另用原生 `with torch.device('meta')` 构造同一真实 Llama：对象身份依次为共享 → `to_empty(cpu)` 后不共享 → `tie_weights()` 后共享，独立验证物化危险。meta 上应先核对象身份；物化后再结合存储关系与梯度，不以空地址代替结构证明。

本地可审阅反馈：明确 meta 前置断言的对象身份及依赖版本，区分构造时已断开与 `to_empty` 导致断开；保留物化后梯度正负对照。仅留项目内，不替作者改测试、评论或推送已有 PR head。原作者使用 torch2.11／Transformers5.9 的 GPU 与内存数据不移作本机结果；其30B修复后训练停滞仍未归因，本轮不复跑、不下结论或新增 M5 队列。[版本、命令与 hash](../../diary/2026-10-25.md)留日志，五份源码语法通过不等于完整 FSDP／offload／内存验收。

## TRL 包装器的打包边界探测

**能转交参数，不等于外层签名显式声明了底层能力。** [#7931](https://github.com/verl-project/verl/issues/7931)针对 TRL value-head critic：FSDP只检查外层 `forward` 是否含 `cu_seqlens`，而该包装器通过 `**kwargs` 转交给底层，导致已支持边界的 patched Qwen3.5仍得不到两个边界参数。报告者 liruiluo 已有 [#7932](https://github.com/verl-project/verl/pull/7932)，2026-10-25核对 OPEN／非draft／未合并，head `918c80b0ca9a3d659dc58e21f3c3ea1ffb312aec`、父提交／实际共同祖先 `3efe38c759c14622fd1b2c9e3679f2d02f86bdac`，两文件288新增；API目标base `3f9d4c0d`不是候选已包含的主线，不另实现。

### 只检查已知透明包装器

固定候选的 [`load_valuehead_model`](https://github.com/verl-project/verl/blob/918c80b0ca9a3d659dc58e21f3c3ea1ffb312aec/verl/utils/model.py)先尝试原生token分类模型，再走可选TRL fallback；不能把这个问题泛化为所有Qwen3.5 critic。受影响条件为 remove-padding、多个独立样本打包、外层为TRL value head且底层patched模型支持边界；不要求SP大于1。原input builder只有在 `pass_packed_cu_seqlens` 为True时传两边界，补齐槽位还追加最终offset。Qwen3.5线性分支从kwargs取显式边界，不能用position_ids重置直接替代这条接口。

候选保留显式签名优先；仅在value_model、类名及TRL命名空间匹配时尝试可选导入，再要求**导入类对象与实例精确同类，绑定forward就是该类forward**，才检查 `pretrained_model.forward` 的显式能力。名称／命名空间只是避免无关导入的前置过滤，不是最终可信依据；未知包装器、子类、同名伪装或实例替换forward不因此获准。继续调用原value-head包装器，不绕过value计算，也不把任意 `**kwargs` 视为支持所有参数。此处为源码范围，非安全沙箱保证。

### 升级版本不等于修复接口

本机从官方annotated tags解析到commit，读取完整源文件，定义原forward方法但不调用模型体；真实 `inspect.signature` 与AST核对结果如下。前三份方法AST完全相同，均把kwargs交给 `self.pretrained_model`；这不是已安装TRL的实参转发测试。

| TRL版本 | 固定commit前缀 | wrapper显式含cu_seqlens | 原PPO/value-head源码 |
| --- | --- | --- | --- |
| 0.27.0 | 17acd61f | 否 | 存在 |
| 0.28.0 | 49ef3342 | 否 | 存在 |
| 1.12.0 | 59c4a8e1 | 否 | 存在 |
| 1.13.0 | 3d9261f1 | 不适用 | 完整715项Git tree无原路径 |

前三版的包装器签名不会仅因升级而让旧探测变True；底层Qwen原normal forward的AST则明确声明cu_seqlens。官方 [TRL #7020](https://github.com/huggingface/trl/pull/7020)已合并，移除PPO及value-head模块；1.13.0不属于保留原fallback接口的直接升级修复。固定verl候选FSDP extra仍钉TRL0.27.0，本轮没有改pin、安装TRL或提出替代实现。

### 源码证据与模型隔离分开

Python3.12.13、仅标准库与只读GitHub API，三份原方法定义／签名及四份verl文件语法核对完成。[复现、完整commit与hash](../../diary/2026-10-25.md)留日志。本机缺TRL／PEFT，未导入其完整类、执行候选可选导入分支、原35／53例或另造recording模型。作者原测试stub了分布式初始化与FSDP wrapping；其已安装TRL forward配tiny recorder的边界对象／梯度测试支持接口路由，不等于真实Qwen线性注意力的样本隔离、SP／kernel或训练收益。

本题收束为包装器能力与依赖版本知识。真实checkpoint数值隔离仍是另一层证据，不以源码或包装器测试补齐，也不为已有作者修复自动新增M5队列；不改模型kernel／训练专项或对上游发布。

## 参考

- [verl GitHub](https://github.com/volcengine/verl)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
