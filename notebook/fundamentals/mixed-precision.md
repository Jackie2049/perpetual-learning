# 混合精度训练深度解析

> FP32 → FP16 → BF16 → FP8：为什么降低精度反而更快

2026-10-23 补充：[AMP 跳步与学习率计数](#amp-跳步与学习率计数)。torch 2.9.1 的真实 CPU GradScaler 对照区分梯度检查、优化器调用与 scheduler 推进；没有测量低精度算子吞吐或 GPU 溢出。

## 1. 浮点数表示

### IEEE 754 格式

```
浮点数 = (-1)^sign × 2^exponent × 1.mantissa

sign: 符号位
exponent: 指数位（决定动态范围）
mantissa: 尾数位（决定精度）
```

### 格式对比

| 格式 | Sign | Exponent | Mantissa | 总位数 | 动态范围 | 最小正规数 |
|------|------|----------|----------|--------|----------|-----------|
| FP32 | 1 | 8 | 23 | 32 | ±3.4×10³⁸ | 1.2×10⁻³⁸ |
| FP16 | 1 | 5 | 10 | 16 | ±65504 | 6.1×10⁻⁵ |
| BF16 | 1 | 8 | 7 | 16 | ±3.4×10³⁸ (同FP32!) | 1.2×10⁻³⁸ |
| FP8 E4M3 | 1 | 4 | 3 | 8 | ±448 | 2⁻⁶ ≈ 0.0156 |
| FP8 E5M2 | 1 | 5 | 2 | 8 | ±57344 | 2⁻¹⁴ ≈ 6.1×10⁻⁵ |
| INT8 | - | - | - | 8 | [-128, 127] | 1 |

### 关键区别

```
FP16 vs BF16:
  FP16: 更多 mantissa (10位) → 更精确，但范围小（最大65504）
  BF16: 更多 exponent (8位) → 范围=FP32，但精度低

  FP16 的最大有限值为65504；小于最小正规数仍可能是次正规数，并非全部为0
  BF16 范围较宽，但仍可能 overflow/underflow，也不能防止 NaN；表示精度较低

FP8 两种模式:
  E4M3: 更多精度，范围小 → 前向传播（权重、激活）
  E5M2: 更多范围，精度低 → 反向传播（梯度）
```

### 数值精度对比（π 的表示）

```
FP32: 3.1415927...  (7 位有效数字)
BF16: 3.140625      (3 位有效数字)
FP16: 3.140625      (3-4 位有效数字)
FP8 E4M3: 3.125     (2 位有效数字)
```

## 2. 为什么混合精度能加速

### Tensor Core 加速比

GPU 有两种计算单元：
- **CUDA Core**: 通用标量计算
- **Tensor Core**: 专用矩阵乘法单元

| GPU | FP32 | FP16 Tensor Core | BF16 Tensor Core | FP8 Tensor Core |
|-----|------|-----------------|-----------------|----------------|
| A100 | 19.5 TFLOPS | 312 TFLOPS (**16x**) | 312 TFLOPS | N/A |
| H100 | 67 TFLOPS | 990 TFLOPS | 990 TFLOPS | 1979 TFLOPS (**30x**) |
| H200 | 67 TFLOPS | 990 TFLOPS | 990 TFLOPS | 3958 TFLOPS |

**核心原因**：Tensor Core 一次处理矩阵块而非单个元素，低精度让寄存器/SMEM 放更多数据。

### 显存节省

```
普通 PyTorch autocast 不会把模型参数永久转为 FP16/BF16。
若叶子参数保持 FP32，其累积梯度通常也为 FP32；不能按低精度
算子的输出 dtype 推算参数、梯度或通信量自动减半。

低精度参数 + FP32 master copy、原生 BF16 参数与 FP32 参数的 autocast
是不同存储方案。优化器状态 dtype、是否有额外主副本、激活保存与
通信策略都要按实现分别计量；不能统一套用一个总显存倍数。
```

## 3. FP16 混合精度训练 (AMP)

### 3.1 核心机制

```
普通 autocast：FP32 叶子参数 → 按算子规则选择 dtype 的前向与 loss
scaled loss → 反向计算 → FP32 叶子上的已缩放累积梯度
unscale + 检查 inf/NaN → 梯度有限才调用普通非融合优化器
```

### 3.2 Loss Scaling 原理

**问题**：FP16 的最小正规数约为 6.1×10⁻⁵，但更小的值并非全部立即变为 0；还存在次正规数，具体算子也可能采用 flush-to-zero。小梯度可能丢失，loss scaling 用来降低这类下溢风险，不是精度或有限性保证。

**解决**：在前向计算完成后，将 loss 乘以一个缩放因子 S（如 S=65536）：

理想算术下，`∂(S·loss)/∂θ = S·∂loss/∂θ`，因此更新前除回 S 可恢复原梯度尺度。实际低精度计算仍有舍入和溢出风险。PyTorch 的 `loss.backward()` 将结果累积到叶子 `.grad`，返回值不是梯度张量；不能用 `grad = backward(loss)` 接收梯度再相除。

### 3.3 动态 Loss Scaling

静态缩放的问题：S 太大可能导致梯度溢出为非有限值；S 太小则仍可能下溢。

启用的 GradScaler 在 `unscale_` 时记录对应优化器梯度中的 inf/NaN；普通非融合优化器由 `step` 决定调用还是跳过。`update` 再根据本轮检查结果调整 scale：发现非有限梯度则回退，否则累计连续无异常轮数，达到 `growth_interval` 才增长。不是按外层尝试次数取模，也不保证 scale 始终大于等于 1。多优化器每个单独 `step`，最后只 `update` 一次。[PyTorch 2.9 AMP 示例](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers)

### 3.4 PyTorch AMP 使用

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast("cuda", dtype=torch.float16):
        output = model(data)      # 部分算子在 FP16，部分在 FP32
        loss = criterion(output, target)

    scaler.scale(loss).backward() # 缩放 loss → 反向传播
    scaler.step(optimizer)        # 检查梯度，有限才调用普通优化器
    scaler.update()               # 更新缩放因子
```

### 3.5 自动精度选择规则

autocast 按设备和具体算子选择 dtype，不是给整个前向统一换类型。以 CUDA 的官方列表为例，符合条件的 matmul/linear 可选低精度，softmax/layer_norm 等可转 FP32；未列出的算子通常遵循输入类型或自身类型提升规则，不能把所有 element-wise 操作归为低精度。原地操作、显式 `out=` 或指定 `dtype=` 也不能直接套用列表。

损失函数同样要看具体 API：CUDA autocast 下 `binary_cross_entropy`/BCELoss 会拒绝执行，`binary_cross_entropy_with_logits`/BCEWithLogitsLoss 才是支持的组合。CPU 的算子列表另列，不照搬 CUDA 表。[官方算子规则](https://docs.pytorch.org/docs/2.9/amp.html#autocast-op-reference)

## 4. BF16 训练

### 4.1 优势

```
1. 通常无需 Loss Scaling
   - BF16 的指数范围接近 FP32，降低 FP16 那类范围不足风险
   - 不表示梯度绝不会 underflow/overflow 或产生 NaN
   - 简化训练代码，消除 scaler 调参

2. 减少因 FP16 范围不足导致的问题
   - 不等于所有模型和任务更稳定，仍需检查 loss 与梯度

3. 同样享受 Tensor Core 加速 (312 TFLOPS on A100)
```

### 4.2 PyTorch BF16 训练

```python
# 方法 1: 自动混合精度
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()  # BF16 autocast 通常不配 GradScaler

# 方法 2: 模型完全 BF16
model = model.to(torch.bfloat16)

# 方法 3: DeepSpeed BF16
ds_config = {
    "bf16": {"enabled": True},
}
```

### 4.3 BF16 的代价

```
精度降低:
  FP32: 7 位有效数字
  BF16: 3 位有效数字

影响:
  - 微小的梯度更新可能被舍入掉
  - 累加操作精度下降（需要更高精度的规约）
  - 最终质量差异需要按模型、任务与训练配置验证

BF16 autocast 与直接把全部模型参数转为 BF16 不等价。
```

## 5. FP8 训练

### 5.1 两种 FP8 格式

```
E4M3 (前向传播):
  - 4 位指数，3 位尾数
  - 范围: [-448, 448]
  - 更精确 → 适合权重和激活

E5M2 (反向传播):
  - 5 位指数，2 位尾数
  - 范围: [-57344, 57344]
  - 更大范围 → 适合梯度（值分布更宽）
```

### 5.2 FP8 训练流程

```
┌──────────────────────────────────────────┐
│ FP32/BF16 主参数                          │
│         ↓ quantize to FP8 E4M3           │
│ FP8 前向 (Tensor Core, 2x BF16速度)      │
│         ↓                                │
│ FP8 Loss                                 │
│         ↓                                │
│ FP8 反向 (E5M2 梯度)                      │
│         ↓ dequantize to FP32/BF16        │
│ FP32/BF16 参数更新 (Adam)                 │
└──────────────────────────────────────────┘
```

### 5.3 延迟缩放 (Delayed Scaling)

FP8 量化需要确定缩放因子。常用策略：

```python
# 延迟缩放: 用前一步的统计量确定当前步的缩放因子
scale_factor = max_abs_value(history_gradients) / FP8_MAX
quantized = float_to_fp8(gradient, scale_factor)
```

### 5.4 当前状态 (2024-2025)

- **H100/H200**: 硬件支持 FP8 Tensor Core
- **Transformer Engine**: NVIDIA 库，自动管理 FP8 量化/反量化
- **Megatron-LM**: 支持 FP8 训练
- **挑战**:
  - 训练稳定性（某些模型/任务精度下降）
  - 量化噪声累积
  - 需要精细的缩放策略
  - 收敛可能需要调参

```python
# Transformer Engine FP8 使用
import transformer_engine as te

model = te.Linear(768, 768, bias=False)

with te.fp8_autocast():
    output = model(input_bf16)  # 自动 FP8 计算
```

## 6. 实践建议

### 选择决策树

```
训练 LLM?
  ├── 训练精度要求高 → BF16 (推荐)
  ├── 推理速度优先 → FP8 量化推理
  ├── 硬件不支持 BF16 (旧 GPU) → FP16 + Loss Scaling
  └── 最新硬件 (H100+) → FP8 训练 (尝鲜)

微调?
  ├── LoRA/QLoRA → BF16 或 FP16
  └── Full fine-tune → BF16

推理?
  ├── FP16/BF16 → 通用部署
  ├── INT8 (GPTQ/AWQ) → 量化推理
  └── FP8 → H100 部署
```

### 精度 vs 速度 vs 显存

| 格式 | 速度 (相对FP32) | 显存 (相对FP32) | 训练稳定性 |
|------|----------------|----------------|-----------|
| FP32 | 1x | 1x | 最高 |
| FP16 + AMP | ~3-8x | ~50-70% | 需调参 |
| BF16 | ~3-8x | ~50-70% | 高 |
| FP8 | ~6-16x | ~30-50% | 中等 (发展中) |

## 7. 学习要点

1. **混合精度的核心**：计算用低精度（快），参数更新用高精度（准）
2. **FP16 需要 Loss Scaling** — 防止梯度 underflow
3. **BF16 通常不需要 Loss Scaling** — 范围较宽不等于不会产生非有限值
4. **FP8 是前沿方向** — 需要 H100+ 硬件支持，稳定性仍在改进
5. **Tensor Core 是加速的根本原因** — FP16/BF16 在 Tensor Core 上快 16x
6. **分别计量参数、梯度、状态和激活** — autocast 不自动将参数与梯度存储减半

## AMP 跳步与学习率计数

GradScaler 的一次 `step` 尝试，不等于优化器完成了一次参数更新。若学习率计划以有效优化器调用计数，必须明确跳步时是否推进；若以外层轮次、样本数或 epoch 为时间轴，则应按那个契约处理，不能笼统要求所有 scheduler 都采用同一种门控。

### 单优化器的三种计数

真实 CPU 对照：FP32 标量参数初值10，普通非融合 SGD、lr=0.1，StepLR 每调用一次乘0.5；GradScaler 初始 scale=8、连续两轮无异常才增长。四次 loss 的梯度依次为2、inf、2、2，均通过真实 autograd 产生。这里只验证缩放和控制流程，不使用 autocast，也不把人为的 inf 当成实际 FP16 溢出。

| 尝试 | 普通 SGD 是否调用 | update 后 scale | 每次尝试都推进时的候选 LR | 仅 SGD 调用后推进时的候选 LR |
|---|---|---|---|---|
| 1 | 是 | 8 | 0.1 | 0.1 |
| 2 | 否 | 4 | 0.05，未用于更新 | 0.05，未用于更新 |
| 3 | 是 | 4 | 0.025 | 0.05 |
| 4 | 是 | 8 | 0.0125 | 0.025 |

两条轨迹均只有3次 SGD 调用，scheduler 却分别推进4次和3次，最终参数约9.725和9.65。第二次跳步并不自动撤销外层 `scheduler.step()`；两条轨迹都不触发“首次 scheduler 早于 optimizer”的警告，因为第一轮已经调用过 optimizer。仅靠无警告不能认证计数一致。

每次 `scaler.step` 返回值都为 `None`：成功的普通 SGD 默认也返回 `None`，不能据此判断跳步。成功轮的 scale 也可能保持不变，不能把“scale 没增长”解释为失败。[当前版本 GradScaler 实现](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/amp/grad_scaler.py)

### 跳过哪些状态

另用普通非融合 AdamW（lr=0.1、weight decay=0.2）先做一次有限梯度更新，再分别输入 inf 和 NaN 梯度。两组中，参数、`exp_avg`、`exp_avg_sq` 和 optimizer 的 `step=1` 均逐张量保持不变；连 AdamW 衰减也没有单独执行。GradScaler 的 scale 从8退至4，但非有限 `.grad` 仍保留。调用方清梯度并做有限反向后恢复更新，optimizer step 才变为2。

检查对象是梯度而不是 loss 标量：合成 `loss=p+inf常量` 的前向 loss 为 inf，但对 p 的梯度有限，普通 SGD 仍调用，参数10变9.9。反过来，有限 loss 反向后显式将其梯度改为 inf，会跳过更新。这不是建议忽略异常 loss，而是说明 GradScaler 不是通用 loss/模型健康检查。

### 多优化器不是原子更新

两个不共享参数的 SGD 共用一个 scaler：p 的梯度为2，q 的梯度为inf。分别调用两个 `scaler.step` 后统一 `update`，无论调用顺序如何，p 均从10变9.8，q 保持10，全局 scale 则降至4。后一个失败不会回滚已成功的优化器；先调用失败者也不会阻止另一个成功。

因此，“这一轮 scale 下降”不能告诉每个优化器是否更新，更不能直接作为所有 scheduler 的统一成功标志。需要原子协同更新的算法，必须另行定义联合有限性检查、同步与状态策略；这里没有实现替换方案。[官方多优化器契约](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers)

### 复查范围

[CPU 脚本](../../experiments/scripts/verify_amp_step_clock.py)正常导入已安装 torch，无包壳或替身：

```bash
python experiments/scripts/verify_amp_step_clock.py
```

Python3.12.13 / torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）：两条 scheduler 轨迹、两组 AdamW 跳步/恢复、loss与梯度的两向控制、两个优化器顺序均通过。脚本打印 GradScaler、SGD.step、AdamW 继承的 Adam.step 及 StepLR 源文本 SHA256；GradScaler 指纹为 `807c51d445e644458cc02024a83a33aaffbfa2c21fcb06a79fd390b1c0e55c7e`。

本例用公开 optimizer post-hook 观察普通非融合优化器调用，配合参数/状态对照，不是通用更新认证方案。融合或自定义 AMP 优化器可能在 `optimizer.step` 内处理跳步；即使普通优化器被调用，lr=0等条件也可让参数不变。没有运行模型、DDP/FSDP、融合优化器、FP16算子或设备；不据此诊断某个框架已有故障，也不新增 M5 硬件任务。

## 参考

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR 2018)
- [A Study of BF16 for Deep Learning](https://arxiv.org/abs/1904.06950)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
