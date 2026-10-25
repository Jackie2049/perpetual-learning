# Ray 分布式框架深度解析

> verl 的基础设施 — 理解 Actor/Task/Placement Group 模型

## 版本固定与控制面安全

2026-10-25接续：**依赖声明、锁文件、已安装运行时、认证启用和网络边界不是同一项验收。** verl [#7599](https://github.com/verl-project/verl/issues/7599)已有报告者jameswong-hai的OPEN/non-draft [#7600](https://github.com/verl-project/verl/pull/7600)，head `821cb4f71bca4ccdb615bc150dd59a949b2a0f28`；本节不另写升级补丁，也不将候选视作已合并或已部署。

### 同一仓库的不同安装入口

固定当轮main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，只读Git blob，使用Python3.12的tomllib、setup.py赋值AST及packaging Requirement解析。没有执行setup.py、resolver或安装命令。下表只比较选定文件的两个快照，不声称全镜像／全依赖验证：

| 入口 | 主线快照 | 候选快照 |
|---|---|---|
| pyproject 的verl-core extra／setup.py的install_requires | ray[default]>=2.41.0 | ray[default]==2.58.0 |
| requirements.txt／requirements-npu.txt | ray[default]，无版本限制 | ray[default]==2.58.0 |
| uv.lock中的ray package条目 | 2.55.1 | 2.58.0 |
| TRT-LLM选定Dockerfile直接安装行 | 2.54.1 | 2.58.0 |
| Ascend A2选定Dockerfile直接安装行 | 2.46.0 | 2.58.0 |

主线声明的范围允许2.47.1，**不表示主线uv.lock选择了它**，更不表示某台机器正在运行它。候选把选定入口对齐2.58.0，是声明与锁文件事实；旧镜像／既有集群不因文件改动自动升级。锁文件中一个package条目可解析，也不等于 `uv lock --check`、wheel元数据或所有backend约束已通过。代码及结果hash见[当日日志](../../diary/2026-10-25.md)。

### 认证默认值必须绑定版本

[Ray 2.58.0固定版本文档](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-core/internals/token-authentication.rst)说明该版本默认 `RAY_AUTH_MODE=disabled`，启用token模式还需各节点配置一致；仅设置凭据来源不等于切换认证模式。本节未读取、生成或输出任何token，也未检查真实集群配置。

当日[latest认证文档](https://docs.ray.io/en/latest/ray-core/internals/token-authentication.html)已描述Ray2.59本地集群的默认变化。因此既不能用latest的本地默认替代2.58语义，也不能把旧“2.52及以后都默认关闭”的说法无限推广。先固定实际版本、启动方式与节点角色，再判断默认值；PR选择2.58.0本身不是对所有部署的升级建议。

[Ray安全说明](https://docs.ray.io/en/latest/ray-security/index.html)将Dashboard／Jobs／Client视作可控制集群的开发者入口，要求可信代码和受控网络；token认证属于附加防护，不能替代网络隔离，同集群多个job也不构成相互隔离。不要将模型推理HTTP接口的访问规则直接当作Ray控制面的边界，更不能把依赖pin解释为服务器已经安全。

### 核对范围与接续

#7600目标base为 `3f9d4c0d`，共同祖先 `ea53291385ce764019a2b40733605f21d8317583`，ahead1／behind87；祖先diff共34文件，uv.lock有大段重锁变化。读取目录时已处理API默认30项分页，34文件目录不等于全patch审计。本轮读了元数据／requirements／helper／Docker说明六处diff、选定七文件的两版完整解析，以及PR唯一CLA评论；reviews空。作者报告的lock检查、Ray版本导入、wheel构建及pre-commit未在本机复跑，专用镜像构建作者也未跑，不能将这些报告标成本项目测试。

本轮没有安装／导入Ray、构建镜像、启动服务、读取凭据或探测端口／GPU／服务器；离线依赖核对不是认证、网络隔离或事件响应验收，也没有证据将某版本与现有服务器事件建立因果。只有未来明确授权且环境安全时才由相应负责人安排运行验证；当前不新增M5任务、升级现有环境或写社区／已有PR head。下一轮转独立需求／实质反馈，不重写已有pin方案、扩安装矩阵或把静态声明当部署结果。

## 1. Ray 概述

### 1.1 定位

Ray 是一个通用的分布式计算框架，提供：
- **分布式 Python**：将单机 Python 代码变为分布式
- **Actor 模型**：有状态的远程服务
- **Task 模型**：无状态的远程函数
- **资源管理**：GPU/CPU 自动调度

```
verl 使用 Ray 的方式:
  Ray Actor = GPU Worker (ActorRolloutRef, Critic, Reward)
  Ray Task  = 数据处理、分发、收集
  Placement Group = GPU 资源拓扑管理
```

### 1.2 Ray vs 其他方案

| 框架 | 编程模型 | 适用场景 |
|------|---------|---------|
| Ray | Actor + Task | ML 训练、推理、RL |
| MPI | SPMD (单程序多数据) | HPC、传统分布式 |
| Kubernetes | 容器编排 | 微服务、批处理 |
| Spark | RDD/DataFrame | 数据处理 |
| Dask | Task Graph | 数据分析 |

Ray 的优势：对 Python 和 GPU 友好，适合 ML 工作流。

## 2. 核心概念

### 2.1 Remote Functions (Tasks)

```python
import ray

ray.init()

@ray.remote(num_cpus=2)
def process_data(data):
    # 无状态函数，远程执行
    return data * 2

# 提交 task，立即返回 future
future = process_data.remote([1, 2, 3])
result = ray.get(future)  # 阻塞等待结果
```

**特点**：
- 无状态 — 每次调用独立
- 自动调度 — Ray 决定在哪个节点执行
- 返回 future — 支持异步

### 2.2 Remote Actors

```python
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, model_name):
        self.model = load_model(model_name)

    def predict(self, input_data):
        return self.model(input_data)

    def update_weights(self, new_weights):
        self.model.load_state_dict(new_weights)

# 创建 actor（占用 1 GPU）
worker = ModelWorker.remote("llama-7b")

# 调用 actor 方法
future = worker.predict.remote(input_data)
result = ray.get(future)
```

**特点**：
- 有状态 — 实例变量跨调用保持
- 绑定资源 — 创建时分配 GPU/CPU
- 单线程 — 默认串行处理请求（可通过 `max_concurrency` 调整）

### 2.3 Object Store

```
Ray 的共享内存对象存储:

Task/Actor → 写入 Object Store → 其他 Task/Actor 读取

特点:
  - 零拷贝：同一节点上的 actor 共享数据无需序列化
  - 引用计数：自动回收不再使用的对象
  - 跨节点：通过 Plasma store 共享
```

## 3. 资源管理

### 3.1 资源声明

```python
# Task 级别
@ray.remote(num_cpus=4, num_gpus=1, memory=8*1024*1024*1024)
def train_step(batch):
    ...

# Actor 级别
@ray.remote(num_gpus=2)
class Trainer:
    ...
```

### 3.2 Placement Groups

```python
# 控制资源分配的拓扑
from ray.util.placement_group import placement_group

# 创建 placement group — 确保 actors 在同一节点或特定拓扑上
pg = placement_group([
    {"GPU": 2},  # bundle 0: 2 GPU
    {"GPU": 2},  # bundle 1: 2 GPU
    {"CPU": 4},  # bundle 2: 4 CPU
], strategy="STRICT_SPREAD")

ray.get(pg.ready())

# 在特定 bundle 上创建 actor
worker_0 = ModelWorker.options(
    placement_group=pg,
    placement_group_bundle_index=0
).remote("model")
```

**策略**：
- `STRICT_SPREAD`：每个 bundle 必须在不同节点
- `STRICT_PACK`：所有 bundle 在同一节点
- `PACK`：尽量打包，必要时分散

### 3.3 verl 中的资源管理

```python
# verl 的 ResourcePoolManager
class ResourcePoolManager:
    def create_colocated_worker_cls(self, resource_pool):
        """在同一个 placement group 中创建共置 worker"""
        # Actor + Rollout + Ref 共享 GPU
        # 避免权重跨节点传输
```

## 4. Ray 在 verl 中的使用

### 4.1 架构映射

```
verl 组件           →  Ray 概念
─────────────────────────────────
ActorRolloutRefWorker  →  Ray Actor (num_gpus=N)
CriticWorker           →  Ray Actor (num_gpus=N)
RewardWorker           →  Ray Actor (num_gpus=N)
RayWorkerGroup         →  Ray ActorPool
ResourcePoolManager    →  Placement Group
DataProto              →  Ray Object Store
```

### 4.2 数据流动

```python
# verl 中的 dispatch/collect 模式
class RayWorkerGroup:
    def dispatch(self, data_proto):
        """分发数据到各 worker"""
        chunks = data_proto.chunk(len(self.workers))
        futures = []
        for worker, chunk in zip(self.workers, chunks):
            future = worker.process.remote(chunk)
            futures.append(future)
        return futures

    def collect(self, futures):
        """收集各 worker 的结果"""
        results = ray.get(futures)
        return DataProto.concat(results)
```

### 4.3 通信优化

```
Ray Object Store 的零拷贝:
  同一节点上的 worker 共享 Object Store
  → 数据传输不需要序列化/反序列化
  → 类似共享内存的效果

DataProto 设计:
  batch (Tensor) → 直接通过 Object Store 共享
  non_tensor_batch → 序列化传输
```

## 5. Ray vs MPI：RL 训练的选择

### 5.1 为什么 RL 训练选 Ray 而非 MPI

```
MPI (SPMD):
  - 所有进程运行相同代码
  - 适合: 数据并行训练 (DDP)
  - 不适合: Actor/Rollout/Critic 不同角色

Ray (Actor 模型):
  - 不同 actor 运行不同代码
  - 适合: 多角色 RL 训练
  - Actor 需要 vLLM 推理，Critic 需要训练，Reward 需要推理
  - 各角色资源需求不同
```

### 5.2 组合使用

```
verl = Ray (角色编排) + Megatron/DeepSpeed (GPU 内并行)

Ray: 管理 actor 生命周期、资源分配、数据路由
Megatron/DeepSpeed: 在每个 actor 内部做 TP/PP/DP

两层并行:
  外层: Ray 管理的 Actor 级并行
  内层: Megatron 管理的 GPU 级并行
```

## 6. 实用技巧

### 6.1 调试

```python
# 查看 Ray 集群状态
ray.status()

# 查看 actors
ray.util.list_named_actors()

# 获取 actor 信息
ray.get_actor("worker_0")

# Ray Dashboard (Web UI)
ray.init(dashboard_port=8265)
# 浏览器打开 http://localhost:8265
```

### 6.2 常见问题

```
1. OOM:
   - Ray Object Store 满了
   - 解决: ray.init(object_store_memory=10*1024*1024*1024)

2. Actor 死锁:
   - actor A 等 actor B 的结果，B 也在等 A
   - 解决: 使用 max_concurrency 或异步调用

3. 序列化错误:
   - 数据包含不可序列化的对象
   - 解决: 用 DataProto 分离 tensor 和非 tensor

4. 资源不足:
   - 请求的 GPU 多于可用
   - 解决: 检查 placement group 策略
```

## 7. 学习要点

1. **Ray 的核心是 Actor + Task + Object Store** — 三大原语构建分布式应用
2. **Actor = 有状态远程服务** — verl 中每个 GPU worker 是一个 Actor
3. **Placement Group 控制资源拓扑** — 确保 co-located workers 在同一节点
4. **Object Store 零拷贝** — 同节点数据共享无需序列化
5. **Ray 适合异构工作流** — 不同角色的 RL 训练组件
6. **Ray + Megatron = 两层并行** — 外层 Ray 编排，内层 Megatron GPU 并行

## 参考

- [Ray Documentation](https://docs.ray.io/en/latest/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [verl: Beyond Human Data](https://github.com/volcengine/verl)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)
