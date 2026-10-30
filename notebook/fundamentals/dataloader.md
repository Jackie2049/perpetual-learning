# DataLoader 优化技术

> 训练中的数据加载瓶颈 — 为什么 GPU 在等数据

2026-10-22 补充：[分布式采样的补齐与丢弃](#分布式采样的补齐与丢弃)，以 torch 2.9.1 的真实 CPU sampler/loader 核对样本覆盖和评测分母；本轮不测 I/O、预取或训练吞吐，下方原有性能概述不是本次基准结果。

2026-10-30 补充：[持久worker的epoch与随机状态](#持久-worker-的-epoch-与随机状态)，用两个真实CPU spawn worker区分主进程sampler、worker数据副本与随机游标；完整运行可以复现，不等于每个epoch自动重置增强状态。

## 1. 问题：数据加载成为瓶颈

```
理想训练循环:
  GPU 计算 → GPU 计算 → GPU 计算 → ...

实际训练循环:
  GPU 计算 → 等数据 → GPU 计算 → 等数据 → ...
             ↑ I/O 瓶颈！

数据加载时间占比:
  小模型 (1.3B): ~10-30% 时间在等数据
  大模型 (70B):  ~5-10% (计算时间更长，掩盖了 I/O)
  多模态 (图像+文本): ~20-40% (图像解码慢)
```

## 2. Prefetch 预取

### 2.1 基本原理

```
无 Prefetch:
  时间: |--计算step1--|---加载数据2---|--计算step2--|---加载数据3---|
  GPU 在加载期间空闲！

有 Prefetch:
  时间: |--计算step1--|--计算step2--|--计算step3--|
        | 加载数据2 | 加载数据3 | 加载数据4 | ← 后台线程
  GPU 几乎不等待！
```

### 2.2 PyTorch DataLoader Prefetch

```python
# PyTorch DataLoader 内置 prefetch
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,       # 多进程并行加载
    prefetch_factor=2,   # 每个 worker 预取 2 个 batch
    pin_memory=True,     # 使用 page-locked 内存，加速 CPU→GPU 传输
)

# 效果:
#   num_workers=0: 单线程加载，GPU 等待
#   num_workers=4: 4 个进程并行预取
#   prefetch_factor=2: 总共缓存 4×2=8 个 batch
```

### 2.3 CUDA Prefetch

```python
# 将数据预取到 GPU
class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            # 异步拷贝到 GPU
            self.next_batch = {
                k: v.to(device="cuda", non_blocking=True)
                for k, v in self.next_batch.items()
            }

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            self.preload()  # 立即开始预取下一个
        return batch
```

## 3. Memory Mapping (mmap)

### 3.1 原理

```
普通文件读取:
  磁盘 → 内核缓冲区 → 用户空间 → Python 对象 → Tensor
  多次拷贝，慢

mmap 读取:
  磁盘 → GPU/CPU 直接访问映射区域
  零拷贝（或极少拷贝），快
```

### 3.2 NumPy mmap

```python
# 大文件 mmap 加载
data = np.memmap("large_dataset.npy", dtype=np.float32, mode="r", shape=(1_000_000, 1024))

# 只加载需要的部分 (按需 page fault)
batch = data[0:32]  # 只读取 32 行，不加载整个文件
```

### 3.3 HuggingFace Datasets 的 mmap

```python
from datasets import load_dataset

# HuggingFace datasets 默认使用 memory-mapped Apache Arrow 格式
dataset = load_dataset("allenai/c4", "en", streaming=False)
# 文件在磁盘上，访问时按需读取 → 内存占用极小
# 可以处理比内存大得多的数据集
```

## 4. Streaming Dataset

### 4.1 为什么需要 Streaming

```
问题: 某些数据集 > 1 TB (如 C4, RedPajama)
不可能全部下载到本地磁盘

解决: Streaming 模式
  - 边下载边训练
  - 不需要完整数据集在本地
  - 数据顺序随机化需要特殊处理
```

### 4.2 HuggingFace Streaming

```python
dataset = load_dataset("allenai/c4", "en", streaming=True)
# 不下载，直接从 HuggingFace Hub 流式读取

for batch in DataLoader(dataset, batch_size=32):
    train_step(batch)
```

### 4.3 Megatron-LM 的数据加载

```
Megatron-LM 使用预处理的数据格式:
  1. 预处理: 原始文本 → tokenize → 二进制格式 (.bin/.idx)
  2. 索引文件 (.idx): 记录每个 document 的 offset 和长度
  3. 数据文件 (.bin): 连续存储 token IDs
  4. 加载时: 通过索引文件定位，mmap 读取

优势:
  - 极快的随机访问 (O(1) seek)
  - 低内存占用 (mmap)
  - 支持高效 shuffle (只 shuffle 索引)
```

## 5. 分布式文件系统上的 DataLoader

### 5.1 瓶颈

```
Lustre/GPFS 上的数据加载:
  - 高聚合带宽 (100+ GB/s)
  - 高延迟 (~ms 级 vs 本地 SSD ~μs 级)
  - 元数据操作 (ls, stat) 很慢
  - 小文件性能差

典型问题:
  - 1000 个 worker 同时读取 → 元数据服务器过载
  - 小文件 (每个样本一个文件) → 效率极低
  - 随机读取 → 缓存失效
```

### 5.2 最佳实践

```
1. 预处理为少量大文件 (如 WebDataset tar 格式)
   好处: 减少元数据操作，顺序读取

2. 缓存到本地 SSD
   首次从分布式 FS 读取 → 缓存到本地
   后续从本地读取 → 快

3. 使用 blended datasets
   多个小数据集混合为一个大流

4. 避免全局 shuffle
   用 local shuffle + epoch shuffle 替代
```

## 6. WebDataset 格式

```
传统: 每个样本一个文件
  data/000001.json  → 100 万个小文件 → 元数据灾难

WebDataset: 样本打包到 tar 文件
  shard-000000.tar  (含 10000 个样本)
  shard-000001.tar
  ...

每个 tar 内:
  000001.jpg  000001.json  000001.txt
  000002.jpg  000002.json  000002.txt
  ...

优势:
  - 顺序读取 (高带宽利用)
  - 少量大文件 (低元数据开销)
  - 天然支持 streaming
```

## 分布式采样的补齐与丢弃

`DistributedSampler.drop_last` 处理“数据集长度能否被 rank 数整除”，DataLoader 的 `drop_last` 则由 BatchSampler 处理“本 rank 的样本能否组成完整 batch”。前者为 False 时会复制索引补齐，而不是创建带无效标记的空样本；后者不知道哪些索引是复制来的。[PyTorch 2.9 官方契约](https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.distributed.DistributedSampler)

### 两个开关不能互相替代

真实 CPU 对照使用 10 个样本（ID 0–9）、3 个 rank、每 rank batch size=2、`shuffle=False`。明确指定每个 sampler 的 `num_replicas/rank`，在同一进程依次枚举；DataLoader 为 `num_workers=0 / pin_memory=False`。下表是实际读取结果，不是手写分片模拟：

| sampler drop_last | loader drop_last | 总读取次数 | 不同 ID 数 | 结果 |
|---|---|---|---|---|
| False | False | 12 | 10 | 0 和 1 各读两次 |
| False | True | 12 | 10 | 每 rank 已有完整两批，重复仍保留 |
| True | False | 9 | 9 | 丢掉 ID 9，各 rank 保留最后一个单样本 batch |
| True | True | 6 | 6 | 每 rank 再丢一个尾样本，最终缺 ID 6–9 |

补齐后的 rank 索引分别是 `[0,3,6,9]`、`[1,4,7,0]`、`[2,5,8,1]`。开 sampler 丢弃后变成 `[0,3,6]`、`[1,4,7]`、`[2,5,8]`；再开 loader 丢弃，各自只读前两个。对这个配置，loader 开关无法去重；两个开关同时打开也不保证“只丢一个样本”。

一般地，每 rank 的 sampler 长度在不丢弃时为 `ceil(N/R)`，丢弃时为 `floor(N/R)`。若 loader 还丢尾，则每 rank 实际读取 `B*floor(sampler长度/B)` 条。这是两个独立的取整层次；shuffle=True 时补齐或截掉的是当次排列的头/尾，不必仍是上述 ID。

小数据也有明显边界：N=2、R=4 时，保留 sampler 尾部得到 `[0]`、`[1]`、`[0]`、`[1]`；sampler 丢尾则所有 rank 都为空。即使 sampler 补齐，B=2 且 loader 丢尾也会让所有 loader 为空。本机均通过真实类读取核对。

### 评测分母要保留原始样本身份

合成二元评分只让 ID 0 得 1，其余得 0。原始 10 个样本的均值是 1/10；默认补齐后，ID 0 被读两次，即使正确按所有输出的 sum/count 汇总，也得到 2/12=1/6。这里的问题不是各 rank 均值怎么平均，而是原始样本被重新加权。

若目标是原数据集逐样本均值，需要追踪原始 ID、补齐项和有效性约定，不能仅凭各 rank 等长、loader 设置或最终标量推断无重复/无遗漏。训练有意重复与评测完整覆盖也不是同一目标；具体去重、掩码或采样方案还要满足执行端的 collective/步数契约，不在此给出通用替换。本轮没有诊断或修改 verl、DeepInfer 的评测管道。

### epoch 控制与复现范围

另外用 N=12、R=3、共同 seed=17 核对 shuffle：同一 epoch 重复枚举得到相同顺序；`set_epoch(1)` 后排列改变；改回 epoch 0 能复现最初顺序。两轮都恰好覆盖全部 12 个 ID。应在创建每轮 loader iterator 前设置各 rank 的同一 epoch；只增加外层循环计数不会自动改变 sampler 的 epoch。这个对照不认证数据增强随机性或断点续训。[官方 epoch 约定](https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.distributed.DistributedSampler)

复查入口：[实验脚本](../../experiments/scripts/verify_distributed_sampler_tail.py)，Python 3.12.13 / torch 2.9.1：

```bash
python experiments/scripts/verify_distributed_sampler_tail.py
```

四种主配置、小数据三种边界、合成均值和 epoch 对照全部通过，使用真实 `DistributedSampler`、`BatchSampler` 与 `DataLoader`；不是上游原测试或多 rank 进程实测。未初始化进程组、启动 worker/collective/DDP、运行模型、设备或 I/O 性能测试，也不新增 M5 硬件任务。

运行时 torch git 为 `5811a8d7da873dd699ff6687092c225caffcf1bb`。按 `inspect.getsource` 取得的完整类文本 SHA256：DistributedSampler `725ae8b532bd36ebd6035eaca1f0a3f380d432fa3fdd6c3cf8cb9ad66c5ae24a`，BatchSampler `6bad7b88e69b07178ade438773b0030a23d86ca036ad8f11a8c044cf4876c593`；指纹是类文本而非整个源文件。

## 持久 worker 的 epoch 与随机状态

**sampler改变索引顺序，不会自动把主进程dataset的新属性广播给持久worker，也不等于重置worker的随机游标。** [PyTorch2.9文档](https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.DataLoader)说明，persistent_workers保留进程及其中的dataset实例；worker初始化回调在进程启动的seed设置后执行。主进程里的sampler与worker内执行的取样／增强逻辑必须分别管理状态。

### 两轮真实 CPU worker 对照

[脚本](../../experiments/dataloader-worker-state/verify.py)使用8条自建map-style记录、batch2、两个spawn worker、默认Tensor合批、prefetch_factor2、pin_memory=False，完整消费两个epoch。每轮在创建iterator前更新主dataset.epoch并调用真实DistributedSampler.set_epoch；sampler显式指定num_replicas=1／rank=0、seed7，不初始化进程组。DataLoader的独立generator固定seed2026，worker只额外执行标量torch.rand并回传epoch／初始化计数／seed。

两种persistent设置的索引序列完全相同：epoch0为`[7,0,3,4,1,5,2,6]`，epoch1为`[3,6,5,7,2,1,4,0]`。因此，本例第二轮的索引重排确实生效，但worker状态不同：

| 第二轮状态 | persistent=False | persistent=True |
|---|---|---|
| worker进程 | 本机观察为新PID | 同一PID复用 |
| worker内dataset.epoch | 1，新副本带入主进程更新 | 仍为0 |
| 初始化回调计数 | 每个新副本为1 | 原副本仍为1，没有第二次调用 |
| worker0的initial_seed | 7618721793331721144 | 仍为4046212693407858054 |
| worker0本轮首个随机值 | 0.6343041658 | 0.3416987658 |
| 随机流对应位置 | 新seed的第1–4个数 | 原seed的第5–8个数 |

worker1有同样的状态关系，seed为worker0+1。核对没有只停留在“seed相同／随机值不同”：主进程按每个实际seed创建独立Generator，逐次标量抽样，精确对齐每个worker的四个值；持久模式第二轮须跳过前四次。本例完整消费且每worker每轮恰读四样本，不能照搬这个offset到提前break、异常或不同prefetch调度的恢复场景。

### seed 值不等于随机游标

持久模式的initial_seed不变，但RNG已向前推进；set_epoch(1)只改变sampler生成的索引。worker里的普通Python属性也仍是创建时的副本，所以“主进程改了epoch”与“worker内按新epoch生成增强”不是同一证据。若dataset逻辑依赖epoch，必须明确设计传播或worker内更新方式，不能仅将epoch写到主对象；这里不把关闭persistent_workers当作通用修复或性能建议。

固定安装的DataLoader.__iter__在持久模式复用iterator并调用_reset；_BaseDataLoaderIter._reset重建sampler iterator，worker收到_ResumeIteration后重建fetcher。本例普通map-style路径没有再次调用init_fn或torch.manual_seed；IterDataPipe另有shared_seed分支，不能外推为所有dataset／loader类型都不重设随机状态。

Python3.12.13／Torch2.9.1，git `5811a8d7da873dd699ff6687092c225caffcf1bb`。[完整结果](../../experiments/dataloader-worker-state/result.json)保存每轮索引、dataset／初始化状态、实际seed、32个随机值以及三个原方法文本hash；64位seed以十进制字符串保存，避免JSON消费端将其舍入，PID仅保存是否复用。两次最终新进程完整结果一致，说明这组完整运行仍可复现，**不是persistent_workers必然破坏可复现性的结论**。

```bash
python -B experiments/dataloader-worker-state/verify.py
```

默认沙箱首次拒绝torch_shm_manager，随即触发20秒取批超时；该次不计通过。经本机纯CPU权限批准后保持默认合批路径完成验证，没有用Mock或单进程替代。脚本在finally仅对自己创建的iterator调用固定版本私有_shutdown_workers并确认子进程停止；这是实验清理，不作为应用的公开close API。真实执行了数据worker／CPU共享存储，不涉及DDP／collective、模型或checkpoint恢复；只验证Torch随机抽样，不验证其他库随机流、吞吐或内存峰值，不新增M5算力任务。

## 7. 学习要点

1. **Prefetch 是最基本的优化** — 多 worker + pin_memory + prefetch_factor
2. **mmap 实现零拷贝加载** — 处理大数据集的关键
3. **Streaming 用于超大数据集** — 边下载边训练
4. **Megatron 的二进制格式** — 预处理 + mmap + 索引，最高效
5. **分布式 FS 上要避免小文件** — 用 tar/WebDataset 打包
6. **重叠效果需要测量** — 预取能否隐藏 I/O 等待取决于具体管道，不能仅靠配置保证消除数据加载瓶颈

## 参考

- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [WebDataset](https://github.com/webdataset/webdataset)
- [Megatron-LM Data Preprocessing](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
