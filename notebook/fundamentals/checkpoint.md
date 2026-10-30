# 分布式 Checkpoint 机制

> 大规模训练中的状态保存与恢复 — 从单机到分布式

## 1. 为什么需要 Checkpoint

### 1.1 训练故障恢复

```
大规模训练的现实:
  - 1000+ GPU 训练，平均每小时有一次硬件故障
  - 没有 checkpoint = 故障后从头开始
  - 有了 checkpoint = 从最近状态恢复

Checkpoint 间隔权衡:
  太频繁 → 大量 I/O 时间 (保存一次可能要几分钟)
  太稀疏 → 故障后浪费更多计算
  典型: 每 1000-5000 步保存一次
```

### 1.2 Checkpoint 内容

```
完整 checkpoint 包含:
  1. 模型参数 (FP16/BF16 weights)
  2. 优化器状态 (FP32 master weights, Adam m, v)
  3. RNG 状态 (Python, NumPy, CUDA RNG)
  4. 训练状态 (step, epoch, learning rate scheduler)
  5. 数据加载器状态 (iterator position)
```

## 2. 单机 Checkpoint

### 2.1 PyTorch 基础

```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': global_step,
    'rng_state': torch.cuda.get_rng_state(),
}, 'checkpoint.pt')

# 恢复
ckpt = torch.load('checkpoint.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
torch.cuda.set_rng_state(ckpt['rng_state'])
```

### 状态字典与独立快照

`model.state_dict()` 不是冻结的一刻：默认返回的 Tensor 已脱离 autograd，但仍与参数、持久 buffer 共享存储；重新包一层 `dict` 不会复制这些存储。这是 [PyTorch 2.9 的接口契约](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.state_dict)。官方[保存教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)也提醒，保留最佳模型状态时应复制或及时序列化，而非只暂存字典引用。

2026-10-22 本机用 Python 3.12.13 / torch 2.9.1 做了[真实 CPU 对照](../../experiments/scripts/verify_state_dict_snapshot.py)：单个 FP64 参数初值1，持久 int64 buffer `seen=0`，SGD 的 lr=0.1、momentum=0.9，手动给梯度1后取状态，再给梯度2更新。不是模型 forward/backward 或训练效果测试。

| 第一轮后保留的对象 | 第二轮后的 weight / seen | 是否保留第一轮值 |
|---|---|---|
| 原 `state_dict()` | 0.61 / 2 | 否，共享存储 |
| `dict(state)`、逐值 `detach()` | 0.61 / 2 | 否，只换容器或 Tensor 包装 |
| 逐值 `.cpu()`（输入已在 CPU） | 0.61 / 2 | 否，没有跨设备复制 |
| `copy.deepcopy(state)` | 0.9 / 1 | 是，本例独立存储 |
| 逐 Tensor `clone()` | 0.9 / 1 | 是，本例独立存储 |
| 第二轮前已完成 `torch.save` 的 BytesIO，之后再加载 | 0.9 / 1 | 是，序列化先于后续修改 |

检查不仅比较数值，也核对存储指针：原参数与状态值不是同一个 Python 对象，但指针相同；深拷贝/clone/加载后的值与活参数指针不同。优化器同样要看内部 Tensor：`optimizer.state_dict()` 的 SGD 动量与活状态共享存储，第二轮由1变2.9；深拷贝和已完成的序列化仍是1。这是本例 SGD 行为，不推定所有自定义优化器都相同。

容器操作与原地改写也不同：给浅拷贝字典的 `weight` 键重新赋一个99，不会替换模型参数；对原状态的 `seen` 原地加10，却会把活 buffer 从2改成12。`requires_grad=False` 不代表只读或拥有独立存储。

因此，上方立即同步 `torch.save({...}, path)` 的例子不因这里的别名证据自动失效。需要警惕的是把活字典留到后面、或交给延迟任务后继续更新；完整 checkpoint 还要处理优化器等嵌套状态。逐值 clone 只适用于这里全为 Tensor 的平坦例子，不能当通用复制器：模块 metadata、custom extra state、嵌套结构和共享存储拓扑需另核。

复查（项目根目录，需 Python 3.10+ 和 torch；本轮无安装）：

```bash
python experiments/scripts/verify_state_dict_snapshot.py
```

脚本打印版本及原方法文本 SHA256，本轮 torch git 为 `5811a8d7da873dd699ff6687092c225caffcf1bb`：

```text
Module.state_dict          8cac9a6b14cc6a45fab450de5c5caf516a6d904ea6606ded408f00386018f1f9
Module._save_to_state_dict 8ed57d3b17ba0959d7144c77de0aefda92e895329059c27b06b102716f73f756
Optimizer.state_dict      32626b5dd54af43bf30882883bf278c720f3a7cc99159475b920ef09eec236c8
```

全部观察通过；这是自己的受控实验，不是上游原测试。只保存/加载自己生成的内存 payload，`weights_only=True`、`map_location="cpu"`；无磁盘 checkpoint、线程、设备探测、分布式保存或完整恢复验证。完成序列化后继续更新的证据，不认证并发 staging 一致性、CUDA 流同步、崩溃原子性或 fsync；这些没有在本轮执行，也不据此调整框架实现。

### assign 与优化器的参数引用

**模型和optimizer状态都加载成功，不代表optimizer仍引用当前模型参数。** [PyTorch2.9接口](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)明确要求：`assign=True` 时应在模型加载后创建optimizer，除非启用了swap参数机制。默认 `assign=False` 将值copy到现有参数；在普通Parameter且swap关闭的路径，`assign=True` 可用新对象替换模型属性。原optimizer保存的是参数对象引用，不会因模型属性被替换而自动重绑。

下面用真实CPU SGD、FP64单参数验证该区别。源参数1经梯度2、lr0.1更新为0.8，动量为2；每条恢复分支都从独立deepcopy加载相同模型／optimizer状态，再对`weight²/2`反向并step。`strict=True`下空missing／unexpected列表不是optimizer参数身份检查。

```python
import copy
import hashlib
import inspect
import json
import torch

torch.set_num_threads(1)

class Scalar(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))

def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=False)

source = Scalar(1.0)
source_opt = optimizer(source)
source.weight.grad = torch.tensor(2.0, dtype=torch.float64)
source_opt.step()
model_state = copy.deepcopy(source.state_dict())
optim_state = copy.deepcopy(source_opt.state_dict())

def run(assign, create_before, swap):
    torch.__future__.set_swap_module_params_on_conversion(swap)
    model = Scalar(10.0)
    original = model.weight
    opt = optimizer(model) if create_before else None
    keys = model.load_state_dict(copy.deepcopy(model_state), strict=True, assign=assign)
    assert keys.missing_keys == keys.unexpected_keys == []
    opt = opt if opt is not None else optimizer(model)
    opt.load_state_dict(copy.deepcopy(optim_state))
    torch.testing.assert_close(model.weight.detach(), torch.tensor(0.8, dtype=torch.float64))
    bound = opt.param_groups[0]["params"][0]
    assert opt.state[bound]["momentum_buffer"].item() == 2.0
    opt.zero_grad(set_to_none=True)
    (model.weight.square() / 2).backward()
    tracks_model = bound is model.weight
    assert tracks_model == (not assign or not create_before or swap)
    gradient = model.weight.grad.item()
    bound_has_no_grad = bound.grad is None
    assert gradient == 0.8 and bound_has_no_grad == (not tracks_model)
    opt.step()
    expected = 0.54 if tracks_model else 0.8
    torch.testing.assert_close(model.weight.detach(), torch.tensor(expected, dtype=torch.float64))
    assert opt.state[bound]["momentum_buffer"].item() == (2.6 if tracks_model else 2.0)
    return {
        "original_parameter_preserved": original is model.weight,
        "optimizer_tracks_model": tracks_model,
        "model_gradient": gradient,
        "optimizer_parameter_grad_is_none": bound_has_no_grad,
        "weight_after_step": model.weight.item(),
        "momentum_after_step": opt.state[bound]["momentum_buffer"].item(),
    }

original_swap = torch.__future__.get_swap_module_params_on_conversion()
try:
    rows = {
        "copy_optimizer_before": run(False, True, False),
        "assign_optimizer_before": run(True, True, False),
        "assign_optimizer_after": run(True, False, False),
        "assign_swap_optimizer_before": run(True, True, True),
    }
finally:
    torch.__future__.set_swap_module_params_on_conversion(original_swap)
assert torch.__future__.get_swap_module_params_on_conversion() == original_swap
print(json.dumps({"torch": torch.__version__, "git": torch.version.git_version,
                  "initial_swap": original_swap, "rows": rows}, indent=2))
for method in (torch.nn.Module.load_state_dict, torch.nn.Module._load_from_state_dict):
    print(method.__name__, hashlib.sha256(inspect.getsource(method).encode()).hexdigest())
```

2026-10-30，Python3.12.13／Torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）的结果如下；四条分支的模型键检查与optimizer状态加载均成功，当前模型梯度均为0.8：

| 模型加载方式与optimizer构造时机 | 保留加载前Parameter对象 | optimizer引用当前模型 | 下一步weight／动量 |
|---|---|---|---|
| assign=False，提前创建 | 是 | 是 | 0.54／2.6 |
| assign=True，提前创建，swap=False | 否 | 否，旧参数grad为None | 0.8／2.0，未更新当前模型 |
| assign=True，加载后创建 | 否 | 是 | 0.54／2.6 |
| assign=True，提前创建，swap=True | 是 | 是 | 0.54／2.6 |

这与[按参数序列错配状态](adamw-optimizer-deep-dive.md#恢复时的参数身份)是两种检查：本例没有参数换序，单参数数量也相同；即使optimizer状态已加载，其参数对象仍可能已不在模型中。先完成改变对象身份的模型加载，再以当前模型参数创建optimizer并加载其状态，是上述默认assign路径的对应顺序；不靠`strict=True`推断这种引用关系。

完整小例只保存在本节，复查时用所列Torch环境抽取唯一Python代码块执行。两个最终新进程输出逐字一致；本机方法文本SHA256分别为 `load_state_dict: 3a7358fc601586ba7de25759d7c07f648a5b976fc4c7b46b4737674fd2c66b84`、`_load_from_state_dict: 3f9df94e8b6c11933c677da1bdcc5ad3e954cf8e349742c88feaee4b508c6e0d`。源码的copy_／setattr／swap_tensors三分支解释此对象身份差异，不从接口返回成功替代一步更新检查。

本例只做内存状态加载和一步CPU更新，不读写checkpoint文件、不运行框架／分片optimizer、AMP或设备操作；swap选项仅在该进程的对照内设置，并在finally恢复。不能由这组普通Parameter实验推荐为所有框架统一启用全局swap开关，也不能把它认定为verl／UniRL恢复缺陷。

### 2.2 显存占用

```
7B 模型 checkpoint 大小:
  模型参数 (FP16):           14 GB
  优化器状态 (Adam FP32):     84 GB
  RNG + 训练状态:            ~1 MB
  总计:                      ~98 GB

保存到磁盘时间 (SSD, 7 GB/s): ~14s
保存到磁盘时间 (HDD, 200 MB/s): ~490s ≈ 8 分钟！
```

## 3. 分布式 Checkpoint

### 3.1 朴素方法：每个 rank 独立保存

```
问题: 每个 rank 保存完整模型副本

N 个 GPU × 98 GB = N × 98 GB 的磁盘写入
N=64: 6.3 TB 的 checkpoint！
大量冗余（DDP 中每个 rank 的参数完全相同）
```

### 3.2 Rank 0 保存（DDP）

```python
# 只有 rank 0 保存
if dist.get_rank() == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')

# 恢复: rank 0 加载后 broadcast
if dist.get_rank() == 0:
    state_dict = torch.load('checkpoint.pt')
else:
    state_dict = None
# Broadcast state_dict to all ranks
```

**问题**：恢复时需要 broadcast，N 个 rank 等待 rank 0 加载完成。

### 3.3 分片保存（ZeRO/Megatron）

```
每个 rank 只保存自己的分片:

DDP/ZeRO-1/2:
  rank i 保存: 优化器状态分片 i + (可选) 完整参数
  恢复: 每个 rank 读取自己的分片

ZeRO-3:
  rank i 保存: 参数分片 i + 梯度分片 i + 优化器状态分片 i
  每个分片大小: 98/N GB
  恢复: 每个 rank 并行读取自己的分片
```

**关键优势**：N 个 rank 并行写入/读取，总 I/O 时间几乎不随 N 增长。

### 3.4 Megatron-LM 的分布式 Checkpoint

```python
# Megatron 使用 torch.distributed.checkpoint
from torch.distributed.checkpoint import save, load

# 保存 (每个 rank 只写自己的分片)
save(
    state_dict={'model': model.state_dict()},
    checkpoint_id='checkpoint_dir/',
    storage_writer=...,
)

# 恢复 (每个 rank 只读自己的分片)
load(
    state_dict={'model': model.state_dict()},
    checkpoint_id='checkpoint_dir/',
)

# 目录结构:
# checkpoint_dir/
#   ├── metadata
#   ├── __0_0.pt  (rank 0 的分片)
#   ├── __0_1.pt  (rank 1 的分片)
#   └── ...
```

## 4. 异步 Checkpoint

### 4.1 问题：保存阻塞训练

```
同步保存:
  训练 → 暂停 → 保存 (14s) → 继续训练
  每次保存浪费 14s 的 GPU 时间

异步保存:
  训练 → 继续训练
         ↘ 后台线程保存 (14s)
```

### 4.2 实现方式

```python
# 方法 1: CPU 线程异步保存
import threading

def async_save(state_dict, path):
    # 1. 先将 GPU tensor 拷贝到 CPU
    cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}

    # 2. 后台线程写入磁盘
    def write_to_disk():
        torch.save(cpu_state, path)

    thread = threading.Thread(target=write_to_disk)
    thread.start()
    return thread

# 方法 2: 使用 torch.distributed.checkpoint 的异步写入
# (Megatron-LM 支持)
```

这里的手写例子只处理平坦 Tensor 字典，且复制在启动写盘线程之前完成；不能直接用于嵌套优化器状态。已在 CPU 的 Tensor 仅调用 `.cpu()` 不会隔离存储，所以例子中的 `clone()` 不能因此省略，见[CPU 对照](#状态字典与独立快照)。本轮没有运行该线程示例或验证 GPU 多流 staging、一致时刻快照与后台异常传播。

### 4.3 内存考量

```
异步保存需要额外 CPU 内存:
  同时存在: 当前训练的参数 + 正在保存的副本

7B 模型:
  训练占用: ~98 GB (GPU)
  保存副本: ~98 GB (CPU RAM)
  需要: 足够的 CPU RAM 来容纳副本
```

## 5. Memory-Mapped 加载

### 5.1 问题：加载时间长

```
加载 98 GB checkpoint:
  磁盘读取: ~14s (SSD)
  反序列化: ~10s
  CPU→GPU 传输: ~1s (PCIe)
  总计: ~25s
```

### 5.2 Memory Mapping

```python
import mmap

# 使用 memory mapping 加载
state_dict = torch.load('checkpoint.pt', mmap=True)

# 效果:
#   - 不立即读取整个文件到内存
#   - 按需读取 (page fault 时才读取)
#   - 大幅减少加载时间 (只读 metadata)
#   - 实际读取延迟到首次访问时

# Megatron-LM 支持 mmap 加载
torch.distributed.checkpoint.load(
    state_dict=...,
    checkpoint_id=...,
    enable_mmap=True,  # 启用 memory mapping
)
```

## 6. Checkpoint 格式对比

| 格式 | 特点 | 适用场景 |
|------|------|---------|
| `torch.save` (pickle) | 简单，但慢 | 单机小模型 |
| `safetensors` | 快速，安全，支持 mmap | 推理部署 |
| `torch.distributed.checkpoint` | 分布式分片 | 大规模训练 |
| DeepSpeed ZeRO checkpoint | ZeRO 专用格式 | ZeRO 训练 |
| `msgpack` / `numpy` | 轻量 | 特定场景 |

### 6.1 Safetensors 格式

```
优势:
  - 零拷贝加载 (memory mapping)
  - 惰性加载 (按需读取张量)
  - 安全 (不使用 pickle，避免代码执行)
  - 跨框架兼容 (PyTorch, TensorFlow, JAX)

HuggingFace 模型默认使用 safetensors
```

## 7. 实践建议

### 7.1 Checkpoint 策略

```
小模型 (<1B):
  → torch.save, 单文件，简单直接

中等模型 (1B-13B):
  → 分布式保存, 每个 rank 保存自己的分片
  → 异步保存避免阻塞

大模型 (13B+):
  → torch.distributed.checkpoint
  → 异步保存 + memory mapping
  → 考虑 SSD 或分布式文件系统

超大模型 (70B+):
  → 所有上述 + 只保存优化器状态分片
  → 定期保存完整 checkpoint，频繁保存轻量 checkpoint
```

### 7.2 Checkpoint 管理

```
保存策略:
  - 保留最近 N 个 checkpoint (滚动删除)
  - 保留某些关键 checkpoint (最佳模型、特定 epoch)
  - 异步清理旧 checkpoint

命名约定:
  step_001000/  ← 基于 step 编号
  step_002000/
  best_model/   ← 最佳 checkpoint
  latest → 软链接到最新 checkpoint
```

## 8. 学习要点

1. **分布式 checkpoint 的核心**：每个 rank 只保存自己的分片
2. **异步保存**：CPU 拷贝 + 后台线程写入，避免阻塞训练
3. **Memory mapping**：按需加载，减少启动时间
4. **Safetensors**：推理部署的首选格式（安全、快速、mmap）
5. **Checkpoint 大小**主要由优化器状态决定（Adam = 12Ψ bytes）
6. **故障恢复频率**决定 checkpoint 策略 — 更频繁 = 更安全但更慢

## 参考

- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [Safetensors Documentation](https://huggingface.co/docs/safetensors)
- [Megatron-LM Checkpoint](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/checkpoint.html)
- [DeepSpeed Checkpoint](https://www.deepspeed.ai/tutorials/checkpoint/)
