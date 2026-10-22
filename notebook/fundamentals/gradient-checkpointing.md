# Gradient Checkpointing / Activation Recomputation

> 用重计算减少保存的激活；均匀链式模型的分段策略可达到 O(sqrt(n))，不代表任意逐层 wrapper 的复杂度。

## 1. 核心问题：激活值的显存占用

训练时，前向传播的中间激活需要保存在显存中，用于反向传播计算梯度。

### 1.1 激活显存估算

对于一个 Transformer 层：

```
每层的激活大小 ≈ 11 × b × s × h  bytes (FP16)

b = batch size
s = sequence length
h = hidden dimension

例如: b=8, s=4096, h=4096, L=32 层
每层激活 ≈ 11 × 8 × 4096 × 4096 × 2 bytes = ~2.75 GB
32 层总计 ≈ 88 GB  ← 仅仅激活值！
```

### 1.2 激活 vs 参数的显存占比

| 模型 | 参数显存 | 激活显存 (b=8, s=2K) | 激活占比 |
|------|---------|---------------------|---------|
| 1.3B | 2.6 GB | ~16 GB | 86% |
| 7B | 14 GB | ~60 GB | 81% |
| 13B | 26 GB | ~90 GB | 78% |

**结论**：激活值占训练显存的大部分，尤其是长序列和大 batch 时。

## 2. Gradient Checkpointing 原理

### 2.1 基本思想

不保存所有中间激活，只保存部分**检查点**。反向传播时，从最近的检查点重新计算需要的激活。

```
无 Checkpointing:
  前向: [A1] → [A2] → [A3] → [A4] → [A5] → [A6] → Loss
  保存: A1, A2, A3, A4, A5, A6  ← 全部保存
  显存: O(n) 激活
  计算: 1x 前向

标准 Checkpointing (sqrt(n) 个检查点):
  前向: [A1] → A2 → [A3] → A4 → [A5] → A6 → Loss
  保存: A1, A3, A5              ← 只保存检查点
  反向时:
    需要A6: 从A5重算末段 → A6 ← 用完释放
    需要A4: 从A3重算本段 → A4 ← 用完释放
    需要A2: 从A1重算本段 → A2 ← 用完释放
  显存: O(sqrt(n)) 激活
  计算: ~2x 前向 (多一次重计算)
```

### 2.2 数学分析

以下是等成本链式网络的示意，不是 PyTorch wrapper 的通用显存公式。将 n 层均匀分成 k 段，保存段边界并在反向重建一段，空间由边界约 k 与段内临时值约 n/k 两部分组成：

| 策略 | 主要激活空间示意 | 额外前向工作 |
|------|-----------|-----------|
| 无 checkpointing | O(n) | 无 |
| 均匀 k 段 | O(k+n/k) | 约一次全网前向 |
| 每层各自 checkpoint | 仍保存各层输入边界，不能写成0 | 单次反向、非嵌套时各包层约额外一次前向，不是n次全网前向 |

在这一简化分段模型中，k约sqrt(n)平衡两项；[Chen等论文](https://arxiv.org/abs/1604.06174)给出O(sqrt(n))空间与额外一次前向的算法，也讨论更深的时间空间权衡，因此不是所有策略的绝对最优。这里“2x”若使用，仅指原前向加重算前向，不是包含反向、通信的训练总耗时。

### 2.3 显存节省公式

```
无 checkpointing: Memory_act = n × (每层激活大小)
均匀k段的示意: Memory_act ≈ (k + n/k) × (每层等效激活大小)

节省因子: 从 n 降到 ~2×sqrt(n)
对于 32 层: 从 32x 降到 ~11.3x → 节省约 65%
```

## 3. 选择性激活重计算 (Selective Recomputation)

### 3.1 核心观察

并非所有激活都一样大！Transformer 层中：

```
Attention 计算:
  Q, K, V 矩阵: 3 × b × s × h → 相对小
  Attention scores: b × heads × s × s → 巨大！(与 s² 成正比)
  Attention output: b × s × h → 相对小

MLP:
  中间激活: b × s × 4h → 较大 (4x hidden)

Dropout mask:
  b × s × h → 小但需要保存
```

### 3.2 选择性策略

```python
# Megatron-LM 的选择性重计算
只重计算 Attention scores (与 s² 成正比的部分)
保存 MLP 中间激活、LayerNorm 输出等

收益:
  - 显存节省: 只丢弃选中算子的中间值，不保证比完全重算节省更多
  - 计算开销: 比全 checkpointing 更少 (只重计算 attention)
```

### 3.3 三种策略对比

| 策略 | 激活显存 | 额外计算 | 实现复杂度 |
|------|---------|---------|-----------|
| 无 checkpointing | O(n) | 0 | 低 |
| 标准 (sqrt(n) 检查点) | O(sqrt(n)) | ~1x | 中 |
| 选择性 (只重算 attention) | O(n) 但常数小 | ~0.2x | 高 |

选择性策略取决于实际算子、attention实现与剩余显存，不能仅用s>4K决定。[Megatron论文](https://arxiv.org/abs/2205.05198)报告特定大模型配置的收益，不是所有硬件/配方的比例保证。

## 4. Megatron-LM 中的实现

### 4.1 配置选项

```bash
# 完全不 checkpoint
--recompute-granularity null

# 按层 checkpoint (每层一个检查点)
--recompute-granularity full
--recompute-method block  # 或 uniform

# 选择性 checkpoint (只重算 attention)
--recompute-granularity selective
```

### 4.2 实现机制

```python
# PyTorch 的 gradient checkpointing 实现
from torch.utils.checkpoint import checkpoint

def custom_forward(layer, x):
    # 这段代码在前向时不保存中间激活
    # 反向时从 x 重新计算
    return layer(x)

# 使用 checkpoint
output = checkpoint(custom_forward, layer, input_tensor, use_reentrant=False)
```

### 4.3 Megatron 的 Pipeline Parallel 交互

```
在 PP 中，checkpointing 的行为:
  - 每个 pipeline stage 独立决定是否 checkpoint
  - warmup 阶段: 需要保存更多激活 (等待反向传播)
  - steady state 阶段: 可以更激进地释放

Megatron 优化:
  num_microbatches_with_partial_activation_checkpoints
  → 在 warmup microbatch 中对部分层做 checkpoint
  → 在 steady state 中保存完整激活
  → 平衡显存和计算
```

## 5. 与其他技术的交互

### 5.1 与 TP 的交互

```
TP 中每个 GPU 只计算一部分 head 的 attention:
  - 激活值已经减少 (只存部分 head)
  - checkpointing 的收益相对减少
  - 但 MLP 部分的激活仍然完整 → 仍有收益
```

### 5.2 与 Sequence Parallel 的交互

```
SP 中激活沿序列维度切分:
  - 每个 GPU 只存 s/TP 的激活
  - checkpointing 仍有效，但收益减少
  - Attention scores: (s/TP)² 而非 s² → 更小
```

### 5.3 与 ZeRO 的交互

```
ZeRO 分区参数和优化器状态:
  - 激活值不被 ZeRO 分区
  - checkpointing 是减少激活显存的主要手段
  - 两者互补: ZeRO 减少参数/梯度/优化器，checkpointing 减少激活
```

## 6. 性能影响

以下为原笔记未附模型、设备、版本和复现命令的示意数字，不作为本项目实测或选型依据。实际收益需要在目标配置测量；本轮CPU正确性对照不验证这些比例。

### 6.1 吞吐量影响

```
标准 checkpointing (sqrt(n)):
  吞吐量下降: ~20-35%
  显存节省: ~60-70%

选择性 checkpointing:
  吞吐量下降: ~5-15%
  显存节省: ~40-50%

不做 checkpointing:
  吞吐量: 最高
  显存: 需要最多
```

### 6.2 历史示意数据

| 配置 | 显存 (GB) | 吞吐量 (tokens/s) |
|------|----------|-------------------|
| 无 checkpoint | 75 | 10000 |
| 选择性 | 50 | 9200 |
| 完全 | 35 | 7500 |

## 重计算的随机数与可变状态

**默认重算检查不是梯度等价检查，重算也不是事务回滚。** [PyTorch 2.9文档](https://docs.pytorch.org/docs/2.9/checkpoint.html)说明，非重入版本的default检查比较保存/重算张量的shape、dtype和device，不比较值；默认保留Torch CPU与一种推断设备类型的RNG状态，不涵盖任意外部状态。若反向期间函数行为改变，可能静默得到不同梯度。

2026-10-22在Python3.12.13 / torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）上，直接调用真实`checkpoint(use_reentrant=False)`与CPU autograd。以下均为自编eager控制例，不是UniRL原测试：

| 控制 | 实际观察 | 解释 |
| --- | --- | --- |
| CPU dropout，保留Torch RNG | checkpoint与直算前向、输入梯度、反向后CPU RNG状态逐位一致 | 覆盖了本例的随机源与计算，不是任意模型等价证明 |
| 同例关闭preserve_rng_state | 原前向仍一致，梯度与反向后CPU RNG状态不同 | 只对最初loss或输出不足以核重算 |
| 闭包中的Python `random.Random`，保留Torch RNG | 原前向一致，梯度和该generator终态不同 | Torch RNG保护没有管理这个外部随机源 |
| 闭包系数在前向后由2改3 | default检查未报错；直算/不变重算梯度[8,16]，改变后[12,24] | 张量元数据相同不代表计算所用的值相同 |
| 重算改变保存张量形状2×2→4 | 抛CheckpointError（metadata不匹配） | 此guard能发现这类结构变化，并非完全无检查 |
| 函数入口/末尾计数 | early_stop=True为2/1，False为2/2；两者梯度一致 | 非重入重算可在最后一个所需张量生成后提前退出 |

系数例为`y=(x*s)^2`、x=[1,2]，前向s=2。普通前向图保存2x，梯度是8x；重算在s=3时重建成3x，但原图乘法的局部常数仍为2，于是得到2×3x×2=12x。它甚至不是简单地对整段“换成s=3”求导。默认检查通过，只说明本例没有触发保存张量的结构不匹配。

因此，重算区域应避免依赖会变化的闭包/全局值；缓存映射、模块buffer、随机generator或计数器若会改变，需要按实际调用语义处理，不能假设checkpoint自动复制/恢复所有对象。提前停止也使函数末尾副作用不可靠；关闭early_stop只能让本例完整执行，不能修复外部随机数或可变状态问题。

UniRL固定`6fd82cba13f52f394747832344039408699ea378`的[FSDP包装](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/backend/fsdp/wrap.py)使用非重入checkpoint，并为具有key_cache/value_cache的特定对象复制映射：调用时留一份，前向/重算每次再复制。源码注释依赖BAGEL替换各层条目而非原地修改已有K/V张量；这不是tensor存储深拷贝或任意kwargs快照承诺。本轮仅读此接线，没有执行该wrapper、BAGEL、FSDP、torch.compile或设备路径，不据CPU例诊断框架bug，也不新增真机任务。

复现资产：[verify_activation_recompute_state.py](../../experiments/scripts/verify_activation_recompute_state.py)。只创建显式CPU tensor，仅给CPU default_generator设种子，无设备发现、模型/环境替身或新依赖。打印实现版本及checkpoint源码SHA256；本轮为 `431b229da53972d79bed7e5dbfb6ace2922fff30192ea103208f484e59490f23`。未测显存/耗时，不能据此沿用或更新上文历史性能数字。

```bash
python -B experiments/scripts/verify_activation_recompute_state.py
```

## 7. 学习要点

1. **核心权衡**：用计算换显存 — 重计算 vs 存储
2. **sqrt(n) 分段**是简化均匀链的一个平衡点，不是所有wrapper的复杂度或训练耗时保证
3. **选择性重计算**只丢弃部分中间值，收益取决于算子实现，不保证比完全重算更省显存
4. **Megatron 的分层策略** — warmup 时部分 checkpoint，steady state 完整保存
5. **与 ZeRO 互补** — ZeRO 处理参数/优化器，checkpointing 处理激活
6. **长序列会放大激活压力** — 是否重算与具体收益取决于batch、attention实现、精度和分片配置
7. **重算语义需稳定** — 默认元数据检查不认证梯度相等；Torch RNG、外部随机源和可变状态分别核对

## 参考

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (Megatron-LM, Korthikanti et al., 2022)
- [PyTorch Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
