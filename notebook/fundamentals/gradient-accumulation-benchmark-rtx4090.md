# Gradient Accumulation Benchmark — OPT-125M LoRA on RTX 4090

## Overview

这份历史 OPT-125M LoRA 表格比较真实大批量与梯度累积的时间、显存。2026-10-30 的脚本复核纠正两点：**表中性能比例不是通用定律；有效 batch 相同，也不自动保证训练目标相同。** 下方 CPU 例子说明不等长序列的归一化问题，不重测历史 GPU 数字。

同日补[DDP的全局token分母](#ddp-的全局-token-分母)：两个真实CPU／Gloo进程确认局部mean再平均可反转梯度，单除全局分母也会遗漏DDP平均系数；不是OPT／verl／UniRL或GPU性能复现。

历史脚本把 `max_length=64` 同时用作吞吐分子，但 `padding=True` 只补到输入池实际最长序列，并不保证宽度为 64；labels 直接复制 input_ids，没有将 padding 标成 ignore_index。因此表中 `tok/s` 统一理解为**按设定长度 64 折算的名义速率**，不是已核实的有效训练 token/s。原值保留，不用猜测的 token 数重算。

## Results

### True Batch Size vs Gradient Accumulation

| Config | Step Time | Throughput | Memory |
|---------|-----------|------------|--------|
| **True B=4**   | 0.042s | 6,082 tok/s | 346 MB |
| **True B=8**   | 0.042s | 12,188 tok/s | 421 MB |
| **True B=16**  | 0.043s | 24,074 tok/s | 572 MB |
| **True B=32**  | 0.043s | 47,591 tok/s | 872 MB |
| **True B=64**  | 0.043s | 95,244 tok/s | 1,471 MB |
| Accum=2 (B=4×2) | 0.083s | 6,154 tok/s | 352 MB |
| Accum=4 (B=4×4) | 0.164s | 6,239 tok/s | 353 MB |
| Accum=8 (B=4×8) | 0.327s | 6,262 tok/s | 353 MB |
| Accum=16 (B=4×16) | 0.654s | 6,263 tok/s | 352 MB |

### Comparison: Real vs Accum-equivalent

| Effective B | Real Throughput | Accum Throughput | Ratio | Real Mem | Accum Mem | Mem Ratio |
|-------------|----------------|------------------|-------|----------|-----------|-----------|
| 8   | 12,188 tok/s | 6,154 tok/s | **0.50x** | 421 MB | 352 MB | 0.84x |
| 16  | 24,074 tok/s | 6,239 tok/s | **0.26x** | 572 MB | 353 MB | 0.62x |
| 32  | 47,591 tok/s | 6,262 tok/s | **0.13x** | 872 MB | 353 MB | 0.40x |
| 64  | 95,244 tok/s | 6,263 tok/s | **0.07x** | 1,471 MB | 352 MB | 0.24x |

### Full FT vs LoRA Memory at Different Effective Batch

| Eff B | Full FT | LoRA Real | LoRA Accum | LoRA vs Full | Accum vs LoRA |
|-------|---------|-----------|------------|--------------|---------------|
| 8   | 1,249 MB | 422 MB | 352 MB | **66.2% saving** | 16.6% saving |
| 16  | 1,261 MB | 573 MB | 352 MB | **54.6% saving** | 38.6% saving |
| 32  | 1,258 MB | 871 MB | 352 MB | **30.8% saving** | **59.6% saving** |

## Key Findings

### 时间和吞吐的比较基线

表中累积次数 2→16 时，一个 optimizer step 的时间约随次数增长，但名义吞吐约保持 6,154–6,263；它没有相对 B=4 的 6,082 按次数衰减。约 15 倍差距来自 **B=4×16 对比真实 B=64**，后者在这组历史数据中扩大 batch 几乎未增加 step 时间。

一般地，忽略其他开销、令一次微批前反向耗时为 `t_fb(b)`，则累积 A 次的更新时间约为 `A*t_fb(b)+t_opt`，真实大批为 `t_fb(A*b)+t_opt`；二者处理同量数据，吞吐比取决于这两个时间，而不是恒为 `1/A`。小模型扩批收益不能外推成任意模型、序列长度或设备的线性定律。

### 显存和更新等价的范围

历史累积表的峰值约 352–353 MB；逐微批 backward 释放计算图，可避免同时保留整批激活。但峰值还受微批形状、最长序列、优化器状态、精度及图保留方式影响，不是任何累积配置都恒定。OPT-125M 表格也不能推出 7B LoRA 固定约 6GB、B=1 累积 32 一定适配 24GB，或某个精度／配方必然最快、最准；撤回这些旧外推，不据此改训练配置。

历史脚本 Exp2 的 warmup、计时循环及 Exp4 累积部分直接对每个 mean loss 调用 backward，再 step，未除累积次数或统一有效 token 分母。即使各微批分母相同，梯度也是 mean 梯度的 A 倍；不能将此脚本当作“仅换 batch 切分而更新保持一致”的验证。AdamW、随机 dropout 等进一步使训练轨迹不能只凭 batch 数相同来认定等价。

## 不等长序列的累积归一化

先明确目标是 token 等权还是序列等权。若目标是**未加类别权重、硬标签、无 label smoothing 的有效 token 平均交叉熵**，第 i 个微批有 `n_i` 个有效 token、loss 总和 `S_i`，整窗 `N=Σn_i>0`，则应累积 `S_i/N` 的梯度。对非空微批等价于对局部 mean `μ_i=S_i/n_i` 乘 `n_i/N`。直接累积 `μ_i/A` 只在各 `n_i` 相等（或恰好抵消）时得到同一梯度；每微批样本数相等，不代表有效 token 数相等。

[PyTorch 2.9 CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) 的硬标签 mean 分母排除 ignore_index；启用类别权重时分母变为有效标签权重之和，不能仍机械使用 token 个数。序列等权目标本来就不同，不应为了匹配 token mean 而擅自改掉。

### CPU 梯度和单步更新对照

两条序列各一个微批，第一条仅 1 个有效 token，第二条 3 个；共享标量 θ 生成两类 logits `[x*θ,0]`。在 θ=0 时两微批 mean loss 都是 log(2)，**相同的 loss 标量仍可掩盖梯度方向差异**。

```python
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
x = torch.tensor([[2., 9., 9.], [1., 1., 1.]], dtype=torch.float64)
y = torch.tensor([[0, -100, -100], [1, 1, 1]])
valid = int(y.ne(-100).sum())
expected = {
    "full": (0.125, -0.0125),
    "token_weighted": (0.125, -0.0125),
    "micro_mean": (-0.25, 0.025),
    "micro_sum": (-0.5, 0.05),
}
for mode in expected:
    theta = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    opt = torch.optim.SGD([theta], lr=0.1)

    def loss(rows, reduction):
        z = x[rows] * theta
        logits = torch.stack((z, torch.zeros_like(z)), dim=-1)
        return F.cross_entropy(
            logits.reshape(-1, 2), y[rows].reshape(-1),
            ignore_index=-100, reduction=reduction,
        )

    scalar = 0.
    rows = [slice(None)] if mode == "full" else [slice(0, 1), slice(1, 2)]
    for row in rows:
        value = loss(row, "sum" if mode == "token_weighted" else "mean")
        divisor = valid if mode == "token_weighted" else (2 if mode == "micro_mean" else 1)
        value = value / divisor
        scalar += value.item()
        value.backward()
    grad = theta.grad.item()
    opt.step()
    torch.testing.assert_close(
        torch.tensor([grad, theta.item()], dtype=torch.float64),
        torch.tensor(expected[mode], dtype=torch.float64), rtol=0, atol=1e-12,
    )
    print(mode, "loss=", scalar, "grad=", grad, "theta=", theta.item())
```

2026-10-30，Python 3.12.13／Torch 2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`），CPU FP64 单线程执行：

| 归约 | 累积 loss | 梯度 | SGD 后 θ |
|---|---:|---:|---:|
| 整批 token mean | 0.69314718056 | 0.125 | −0.0125 |
| 各微批 sum／整窗 4 token | 0.69314718056 | 0.125 | −0.0125 |
| 各微批 mean／2 | 0.69314718056 | −0.25 | 0.025 |
| 各微批 mean 直接相加 | 1.38629436112 | −0.5 | 0.05 |

这是原生交叉熵＋SGD 的局部对照，不是历史 OPT／verl／UniRL 训练复现，也不说明实际故障频率或性能。对可按微批分解的同一目标，累积期间固定参数，梯度才按线性相加；BatchNorm、跨样本目标、随机操作或浮点加法顺序仍可能破坏与大批逐位相同的预期。

### 实现时的检查点

在窗口开始前或按既有训练器契约取得整窗有效分母，逐微批 backward 后统一 step；不要在循环中 zero_grad 或 step。空微批不要先算 `0/0` 的 mean 再乘零；整窗 N=0 时需明确更新策略，零梯度也不保证优化器不更新参数，见 [None 与零梯度](adamw-optimizer-deep-dive.md#缺失梯度与零梯度)。

[PyTorch 2.9 AMP 累积示例](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#gradient-accumulation) 要求整窗保持同一缩放，累积完成才 unscale／step／update；其示例的除累积次数不能代替对实际 loss 分母的核对。分布式还须计入全局分母和梯度归约约定，本例不是 DDP/FSDP/AMP 的实现模板。

### DDP 的全局 token 分母

**梯度归约对rank求平均，目标函数却可能要求对token求平均。** [PyTorch 2.9 DDP文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)明确普通DDP会平均各进程梯度。设该归约组有W个rank，rank r的有效token数为n_r、逐token损失和为S_r，全局N=Σn_r且N>0，目标梯度为`Σ∇S_r/N`。因此每rank应反向`L_r=W*S_r/N`，让DDP的`1/W`抵消；直接用`S_r/N`会再缩小W倍，用局部`S_r/n_r`则通常得到不同目标。这里的W是实际梯度平均组大小，不是任意集群GPU数。

[本机脚本](../../experiments/ddp-token-denominator/verify.py)用原生DDP、交叉熵与SGD构造四条独立更新：CPU FP64线性参数θ从0开始，logits为`[θ*x, 0]`，lr=0.1；rank0一条x=2／label0，rank1三条x=1／label1。所有标签都有效，每行作为一个token；生产统计仍须排除padding、ignore_index或loss mask无效位，不能直接拿张量长度当通用分母。

| 布局与每rank反向loss | 归约后梯度（两rank相同） | SGD后θ | 单进程全局token均值参考 |
|---|---:|---:|---:|
| n=1／3，各自local mean | −0.25 | +0.025 | +0.125 |
| n=1／3，S_r/N | +0.0625 | −0.00625 | +0.125 |
| n=1／3，W*S_r/N | +0.125 | −0.0125 | +0.125 |
| n=0／3，W*S_r/N | +0.5 | −0.05 | +0.5 |

最后一行让rank0的空输入仍经过模型前向，对交叉熵`reduction="sum"`所得有图的0执行backward；它依然参与归约，因而也收到rank1贡献的全局梯度。不能因本rank无有效token就独自跳过DDP同步；但本例没有N=0的全局空批，未决定那种情况下是否step，也不靠零梯度推断优化器一定不动。

2026-10-30，Python3.12.13／Torch2.9.1（git5811a8d7），两个spawn工作进程、单线程、Gloo强制本机loopback，分母用真实SUM collective，梯度由DDP默认归约、结果经all_gather_object核两rank一致。两次独立启动的完整[结果JSON](../../experiments/ddp-token-denominator/result.json)相同，包含每rank计数／梯度／更新值与各自单进程autograd参考；不是用peer常数或后端替身计算。

这是普通DDP／SGD的归一化实测，不涵盖FSDP、SP／TP、DDP.join、自定义通信hook、AMP、no_sync累积或实际训练器。后端若执行SUM、另有预缩放或改变参与组，需先核其归约契约，不能再盲乘W。与[UniRL原loss resolver的既有接口控制](../frameworks/unirl-survey.md#原缩放函数的五个条件对照)只是概念关联，不能将本轮DDP结果迁移为该框架的分布式验收。沙箱首次拒绝loopback uv_bind，获准本机执行后两次通过；没有GPU／服务器访问或新增算力队列。

## Tool

历史脚本：[gradient_accumulation_benchmark_4090.py](../../tools/gradient_accumulation_benchmark_4090.py)。2026-10-30 复核固定于个人仓库 [490546c](https://github.com/Jackie2049/perpetual-learning/blob/490546c406abdab1914f5342432622591f4b85bb/tools/gradient_accumulation_benchmark_4090.py#L130)，SHA256 `7d17fbe92a5c6f15c777a20d77b9b47ee19d79fd3c3d3ed247d94737e5877bd0`。脚本原样保留，不运行或修改 GPU 工作；上面小例可直接在 CPU 执行。

## Related

- LoRA Fine-tuning: `notebook/fundamentals/lora-finetuning-benchmark-rtx4090.md`
- Mixed Precision: `notebook/fundamentals/mixed-precision-training-benchmark-rtx4090.md`
- FSDP Scaling: `notebook/fundamentals/fsdp-scaling-125m-benchmark-rtx4090.md`
