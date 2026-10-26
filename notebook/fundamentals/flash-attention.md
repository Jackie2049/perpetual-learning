# FlashAttention 原理深度解析

> IO-Aware Tiling：为什么 FlashAttention 比标准 Attention 快 2-4x

2026-10-26补充：[MHA 与 SDPA 的调用契约](#mha-与-sdpa-的调用契约)区分布尔mask含义、全屏蔽行与eval时dropout。下文历史GPU性能表不是本轮重新验证结果，CPU MATH控制也不代表FlashAttention kernel性能。

## 实验验证 (A16, PyTorch 2.5.1+cu118)

**SDPA (Flash Attention) vs Naive Attention:**

| 序列长度 | SDPA (ms) | Naive (ms) | 加速比 | 显存 (SDPA/Naive) |
|---------|-----------|-----------|--------|-------------------|
| 512     | 0.165     | 0.729     | 4.41x  | 2.2/35.7 MB       |
| 1024    | 0.638     | 2.848     | 4.46x  | 4.3/138.4 MB      |
| 2048    | 2.481     | 22.736    | **9.16x** | 8.7/545.3 MB   |
| 4096    | 9.585     | 66.591    | 6.95x  | 17.3/2164.3 MB    |
| 8192    | —         | OOM       | —      | 34.6/8623.5 MB    |

Naive 显存随 S² 增长 (O(N²))，Flash Attention 随 S 线性增长 (O(N))。
S=8192 时 Naive 仅 attention 矩阵就需要 8.6 GB。工具: `tools/flash_attention_bench.py`

## 1. 核心问题：标准 Attention 的瓶颈

标准 attention 计算：
```
S = Q @ K^T     # [N, d] @ [d, N] → [N, N]
P = softmax(S)  # [N, N]
O = P @ V       # [N, N] @ [N, d] → [N, d]
```

**问题**：中间矩阵 S 和 P 都是 N×N，必须写入 HBM 再读回来。
- N=8192, d=128 时，单个 head 的 S 矩阵 = 8192×8192×2B = 256MB
- 标准流程需要 3 次读取 + 3 次写入 N² 规模数据

## 2. 关键洞察：Memory-Bandwidth Bound ≠ Compute Bound

### GPU 内存层次 (A100)

| 层级 | 容量 | 带宽 |
|------|------|------|
| HBM (显存) | 40-80 GB | ~2 TB/s |
| SRAM (每SM) | ~192 KB | ~19 TB/s |
| L2 Cache | ~40 MB | ~数 TB/s |

**A100 FP16 算力：312 TFLOPS**
- 计算 QK^T (2N²d FLOPs, N=8192, d=128): ~0.055 ms
- 从 HBM 读 256MB attention 矩阵 (2 TB/s): ~0.128 ms
- **结论：等待数据传输的时间 > 计算时间**

**FlashAttention 的核心洞察：减少 HBM 访问次数比减少 FLOPs 更重要！**

## 3. FlashAttention-1：IO-Aware Tiling 算法

### 3.1 核心思想

把 attention 切分为小块（tile），每块在 SRAM 中完成计算，**永远不在 HBM 中 materialize N×N 矩阵**。

```
For each query block Q_i (size B_q × d):
    初始化: o_i = 0, l_i = 0, m_i = -inf

    For each KV block (K_j, V_j) (size B_kv × d):
        s_ij = Q_i @ K_j^T              # [B_q × B_kv] — 在 SRAM 中
        m_ij = rowmax(s_ij)              # [B_q]
        p_ij = exp(s_ij - m_ij)          # [B_q × B_kv]
        l_ij = rowsum(p_ij)              # [B_q]

        # 更新运行统计量
        m_new = max(m_i, m_ij)
        l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij

        # 更新运行输出
        o_i = (l_i * exp(m_i - m_new) * o_i + p_ij @ V_j) / l_new

        m_i = m_new
        l_i = l_new

    Write o_i to HBM  # 只写一次最终结果
```

### 3.2 内存占用对比

| 方法 | HBM 读取 | HBM 写入 | 总访问量 |
|------|----------|----------|----------|
| 标准 | O(Nd + N²) | O(N² + Nd) | Θ(N² + Nd) |
| FlashAttention | Q+K+V 各读一次 + O 写一次 | O only | Θ(N²d²/M + Nd) |

加速因子 ≈ M/d² (M = SRAM 大小)，约 12x 更少的 HBM 访问。

### 3.3 SRAM 足迹

```
SRAM ≈ B_q × B_kv + B_q × d + B_kv × d + 3 × B_q
典型值: B_q = B_kv = 64, d = 128 → ~200KB (适合 SRAM)
```

## 4. Online Softmax：数学基础

### 4.1 标准 softmax 需要 3 遍扫描

```
Pass 1: m = max(x)           # 数值稳定
Pass 2: d = sum(exp(x - m))  # 归一化常数
Pass 3: o = exp(x - m) / d   # 输出
```

### 4.2 Online Softmax：单遍扫描

逐元素更新运行统计量：

```python
m_0 = -inf; d_0 = 0; o_0 = 0

for i in range(N):
    m_i = max(m_{i-1}, x_i)
    d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
    o_i = o_{i-1} * (d_{i-1}/d_i) * exp(m_{i-1} - m_i) + (exp(x_i - m_i)/d_i) * V[i]

# 最终 o_N = softmax(x) @ V (精确值！)
```

**关键**：当 m 增大时，所有之前累积的指数需要乘以 `exp(m_old - m_new)` 进行 rescale。

### 4.3 块级扩展 (FlashAttention)

将 online softmax 从逐元素扩展到逐块：

```python
s_ij = Q_i @ K_j^T                    # 块级分数
m_ij = rowmax(s_ij)                    # 本块最大值
p_ij = exp(s_ij - m_ij)               # 本块 softmax 分子
l_ij = rowsum(p_ij)                   # 本块归一化常数

# 合并运行统计量
m_new = max(m_i, m_ij)
l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
o_new = [l_i * exp(m_i - m_new) / l_new] * o_i
      + [exp(m_ij - m_new) / l_new] * (p_ij @ V_j)
```

## 5. FlashAttention-2：更好的工作分配

FA2 的三大改进：

1. **减少非矩阵乘法 FLOPs**
   - Tensor Core GEMM 比非 GEMM 快 ~16x
   - 重排操作顺序，最大化 GEMM 占比

2. **沿序列长度并行**
   - FA1 只在 batch 和 head 上并行
   - FA2 额外在序列长度上并行 → 更好的 GPU 占用率

3. **更好的 warp 内工作分配**
   - Q block 保持在寄存器中（最快）
   - K/V block 流过 shared memory
   - 内循环迭代 K/V block 而非 Q block

**性能**：比 FA1 快 2x，达到 A100 理论峰值 50-73%。

## 6. FlashAttention-3：Hopper 专用优化

FA3 针对 H100 的三大特性：

1. **Warp 特化 (异步)**
   - 一个 warpgroup 做 GEMM
   - 另一个做 softmax/rescale
   - 并发运行！

2. **Ping-pong 调度**
   - 两个 warpgroup 交替：
   - Warpgroup A 算 GEMM 时，Warpgroup B 做上一 tile 的 softmax

3. **Warpgroup 内交错**
   - 当前 tile 的 GEMM epilogue 与下一 tile 的 GEMM prologue 重叠

**性能**：比 FA2 快 1.5-2x，H100 FP16 达到 740 TFLOPS/s (75%)。

## 7. 对比总结

| 特性 | FA1 (2022) | FA2 (2024) | FA3 (2024) |
|------|-----------|-----------|-----------|
| 核心 | IO-aware tiling | + 工作分配优化 | + 异步重叠 |
| GPU | Ampere (A100) | Ampere | Hopper (H100) |
| 利用率 | ~25-40% | 50-73% | 75% |
| 内存 | O(N) | O(N) | O(N) |
| 精度 | FP16/BF16 | FP16/BF16 | FP16/BF16 + FP8 |

## 8. 与 AI Infra 的关联

- **vLLM PagedAttention**：在 FA 基础上增加分页 KV Cache 管理
- **Megatron-LM TP**：每个 GPU 独立计算不同 head 的 attention，自然适配 FA
- **Sequence Parallelism**：FA 的 O(N) 内存使超长序列训练成为可能
- **推理优化**：FA 是 continuous batching 和 prefix caching 的基础

## 模拟验证

- `tools/flash_attention_sim.py` — FlashAttention 算法模拟器（4 个实验，CPU 可运行）
  - 实验 1: 标准 vs Flash 数值等价（Max Abs Err < 3.3e-7, Cos Sim > 0.999999）
  - 实验 2: HBM IO 分析（标准 N²×d vs Flash N²×d²/M, 16K 序列节省 2.2x）
  - 实验 3: Online Softmax 逐步演示（correction factor exp(m_old - m_new), 误差 < 1.4e-7）
  - 实验 4: Block Size 影响（bs=8→4096 pairs vs bs=512→1 pair, 精度不受 bs 影响）

关键发现:
- **Online softmax 修正因子**: `exp(m_old - m_new)` 调整之前累积值，保证等价
- **IO 复杂度**: 标准 O(N²d), Flash O(N²d²/M), 序列越长节省越多
- **中间存储**: 标准 N=16K 时需要 ~32GB, Flash = 0（零中间矩阵!）
- **Block size 不影响精度**: online softmax 数值等价与 block size 无关

## MHA 与 SDPA 的调用契约

**切换注意力接口时，不能只保留相同mask张量和dropout数值。** [PyTorch 2.14 SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)与[MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)对布尔mask采用相反含义；SDPA也没有模块training状态参数。接口语义正确与选择哪个kernel是两件事。

| 参数 | bool为True | 浮点mask |
|---|---|---|
| SDPA attn_mask | 允许该query关注该key | 加到attention score的bias |
| MHA attn_mask | 禁止该位置的注意力 | 加性bias |
| MHA key_padding_mask | 忽略对应key | 加性bias；本轮只核文档，未执行此参数 |

迁移相同布尔可见域时要反转极性，并核对batch／head／query／key的广播维度。浮点bias不能机械取反或把bool直接cast为0/1：允许项常用0、禁用项用负无穷，其他有限值则改变分数，不等于删除位置。

### 用可见值的均值检查极性

[CPU脚本](../../experiments/sdpa-interface-contract/verify.py)使用真实torch2.14.1、CPU FP64、单线程，并显式限定SDPA MATH。Q/K均0，V为 `[0,2,8]`，一行query、一个head、head_dim=1；MHA投影和输出权重手工置1、bias关闭，使结果直接对应可见值均值。这是算子控制，不是训练模型或重写注意力kernel。

SDPA传 `[True,False,True]` 得4；MHA原样传此bool mask得2，反转后才得4。相同加性bias `[0,−inf,0]` 在两接口均得4。[原始结果](../../experiments/sdpa-interface-contract/result.json)还保留函数源码及脚本指纹；本例只执行MHA的attn_mask，没有把key_padding_mask定义当成另一个实测。

### 全屏蔽行不是正常概率分布

在同一输入中，SDPA的全False mask输出0；MHA全True、need_weights=True时输出NaN，而need_weights=False输出0。安装版[source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py)中，前者显式对全负无穷score做softmax，后者调用SDPA。测试的Q/K/V是不同输入，单head，不能将这个观察移作MHA原生self-attention inference fastpath的结果。

这里没有合法key时，通常的softmax条件分布本就未定义；不同支路对这个退化输入的处理不应被当作统一API保证。也不应只为掩盖NaN就切need_weights。调用方需明确无可见key的行应拒绝、跳过还是按既定语义返回零；零attention经过带bias的输出投影也未必仍为零，本例专门关闭bias。没有检查反向梯度、混合精度或其他backend。

### eval 不会自动传递给函数式 dropout

SDPA按传入dropout_p执行，外围Module.eval()或no_grad()不自动把它改为0；调用方应使用 `dropout_p=(self.p if self.training else 0.0)`。本机MHA函数入口则在training=False时把p置0。

用p=1这个确定性端点控制：固定p=1的SDPA wrapper即使eval仍输出0；按training显式门控的wrapper在eval输出4；原MHA dropout=1时eval输出4、train输出0。p=1只用于无随机干扰地展示调用契约，不是普通dropout率的分布／seed复现测试，更不是训练效果结论。

复现：`python experiments/sdpa-interface-contract/verify.py --output experiments/sdpa-interface-contract/result.json`。环境Python3.12.13／torch2.14.1（git `5c4886908584029761b579af026dcfb627c84070`）；未安装依赖、创建模型服务或访问GPU／NPU／MPS。此处不改框架或新增M5任务，也不复跑[旧因果位置控制](#缓存后缀的因果对齐)；后续只在实际接口迁移需求时接续，不扩shape／mask／dtype矩阵。

## 缓存后缀的因果对齐

**K/V均合法可见时，单token后缀可不加mask；这不能外推多token后缀，调用mask错位也不是kernel数学错误。** 此处补充2026-10-22的CPU算子验证，与上文历史GPU性能表分开。

[PyTorch 2.9 SDPA文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)将非方阵`is_causal=True`定义为左上对齐，布尔attn_mask的True表示允许参与注意力。假设已有2个prefix token，本次query对应绝对位置2、3、4，K/V含位置0..4：正确可见域要按key_position≤query_position建立，而不是把这三个query重新当成位置0、1、2。

令Q/K全0、V的每个值等于key位置，注意力就是可见值的均值。真实CPU SDPA得到：

| 3行query对应位置 | 左上is_causal=True | 不加mask | 按绝对位置对齐 |
| --- | --- | --- | --- |
| 2 | 0 | 2 | 1 |
| 3 | 0.5 | 2 | 1.5 |
| 4 | 1 | 2 | 2 |

左上三角少看了prefix；全部放行则前两行看到了本次suffix里尚不可见的未来key。若只保留最后一个query（位置4），且K/V恰好都是它允许访问的历史及当前token，没有padding或其他可见性限制，不加mask才与该行的因果注意力等价。这是旧Q=1 benchmark的适用条件，不能升级成“decode必须关闭causal”或“SDPA/FA2不能正确decode”。

[causal_lower_right文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.bias.causal_lower_right.html)提供对角偏移S−L的右下对齐bias。对本例3×5，它与绝对位置mask相同；前提是query正好对齐有效K/V的末尾。**物理容量不是有效序列长度**：若K/V张量有7槽但只有前5个有效，按3×7右下对齐再屏蔽padding，首行仍会错误地放行全部5个有效key。本机得到[2,2,2]，正确结果仍是[1,1.5,2]；仅补一个key-valid mask并不能修复已算错的query偏移。

调用方应使用真实位置与有效长度，例如以下是单样本的布尔mask构造；实际batch的每条序列还需自己的长度/位置元数据：

```python
query_pos = torch.arange(2, 5, device='cpu')
key_pos = torch.arange(7, device='cpu')
valid_key = key_pos < 5
allowed = (key_pos[None, :] <= query_pos[:, None]) & valid_key[None, :]
output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                       is_causal=False, dropout_p=0.0)
```

这里只解决连续、同一序列的suffix位置关系；窗口、packed多序列、树状验证、分页槽与逻辑位置的映射仍须各自满足可见性契约，不能只从L/S两个shape推出通用正确mask。

复现：[verify_sdpa_suffix_alignment.py](../../experiments/scripts/verify_sdpa_suffix_alignment.py)，Python3.12.13 / torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）、CPU float64、MATH后端、dropout_p=0。两组可见均值控制及(prefix,suffix)=(4,1)/(2,3)/(0,5)/(3,2)四布局均通过；显式mask的输出与完整因果注意力相应后缀相同，Q/K/V输入局部梯度也在rtol=1e−11、atol=1e−12内匹配，四例输出最大误差实际为0。这不是跨backend逐位保证或真实KV引擎/模型训练等价证明。

脚本只把官方CausalBias的私有`_materialize(cpu)`用于对照已读源码定义，未调用其自动后端分派；应用示例不依赖该私有API。所有tensor与generator显式CPU，无设备发现、GPU/NPU/MPS、模型/缓存分配器替身或显存/吞吐测量；不复跑或认证历史GPU速度数据，也不新增M5真机任务。

```bash
python -B experiments/scripts/verify_sdpa_suffix_alignment.py
```

## 参考

- [FlashAttention-1](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (ICLR 2024)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024)
- [Tri Dao's Blog](https://tridao.me/blog/2024/flash3/)
- [Online Softmax to FlashAttention 讲义](https://courses.cs.washington.edu/cse599m/)
