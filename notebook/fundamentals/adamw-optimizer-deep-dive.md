# AdamW Optimizer 深度解析 — 从数学到实战

> 2026-06-07 | 为什么AdamW是LLM训练标配: decoupled weight decay的理论与实践

2026-10-22补充：见[缺失梯度与零梯度](#缺失梯度与零梯度)的真实CPU AdamW/autograd对照。下文历史小模型结果不是通用超参数推荐；本次没有复验旧RTX 4090性能数据。公式与实现以[PyTorch 2.9 AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)及本机torch2.9.1为核对范围。

## 一、Adam到AdamW的进化

### 1.1 Adam: 自适应学习率 + momentum

```
Adam (Kingma & Ba, 2015):
  m_t = β₁·m_{t-1} + (1-β₁)·g_t          #一阶momentum (均值)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          #未中心化二阶矩，不是方差
  m̂_t = m_t / (1-β₁^t)                     #bias correction
  v̂_t = v_t / (1-β₂^t)
  θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)     #参数更新

直觉:
  m̂_t ≈ 梯度的历史平均值 → 方向稳定(不随单个样本剧烈变化)
  v̂_t ≈ 梯度平方的历史均值 → 用历史尺度调节步长
  √v̂_t ≈ 每个参数的"个性化学习率"

→ 对LLM训练关键: 不同参数梯度方差差异极大
  embedding层: 梯度方差小 → Adam给大步长 → 快速学习
  attention层: 梯度方差大 → Adam给小步长 → 稳定学习
  → Adam自动适配, 不需要手动调每层lr!
```

### 1.2 L2 正则与解耦衰减不同

```
Adam + L2正则:
  loss = L(θ) + λ/2 · ||θ||²
  g_t = ∇L(θ) + λ·θ              #梯度加正则项
  → 正则项被Adam的v_t缩放!

  → m_t和v_t都受正则项影响，v还包含梯度与λθ的交叉项
  → 不能把分母简写成只有variance + λ²θ²

区别: 把λθ加入梯度，不等价于单独乘(1-α·λ)
  → 这是两种更新规则，不是L2一定发散或AdamW一定收敛的定理
  → 历史小模型对比保留在第七节，不据它认定任意模型/超参的优劣
```

### 1.3 AdamW: Decoupled Weight Decay

```
AdamW (Loshchilov & Hutter, 2019):
  m_t = β₁·m_{t-1} + (1-β₁)·g_t          #momentum (不变)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          #二阶矩 (不变)
  θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)     #Adam更新 (不变)
  θ_t = θ_t - α·λ·θ_{t-1}                  #weight decay (独立!)

关键区别: wd不进入梯度 → 不被v_t缩放!
  → wd强度一致 → 大参数大惩罚 → 正确行为!
  → wd不经过moment缩放，但实际衰减因子仍含learning rate: 1-α·λ

理论推导:
  Adam+L2: θ_{t+1} = θ_t - α·(∇L + λθ)/(√v + ε)
           → λθ被√v缩放 → 正则化强度∝λ/(√v) → 不一致!

  AdamW:   θ_{t+1} = θ_t - α·∇L/(√v + ε) - α·λ·θ
           → wd项独立 → 强度∝α·λ → 一致!

→ AdamW = SGD wd的"自适应版本"
  SGD wd: θ = θ - α·∇L - α·λ·θ → wd∝α·λ (一致)
  AdamW wd: θ = θ - α·∇L/(√v+ε) - α·λ·θ → wd∝α·λ (一致,不受√v影响)
```

## 二、超参数解析

### 2.1 β₁=0.9, β₂=0.999, ε=1e-8

```
β₁=0.9 (momentum衰减):
  m_t = 0.9·m_{t-1} + 0.1·g_t
  → 有效窗口 ≈ 1/(1-β₁) = 10步
  → 梯度的10步滑动平均 → 短期趋势
  → 为什么0.9? 经验最优值(SGD momentum也用0.9)
  → 不建议调整(几乎所有论文都用0.9)

β₂=0.999 (方差衰减):
  v_t = 0.999·v_{t-1} + 0.001·g_t²
  → 有效窗口 ≈ 1/(1-β₂) = 1000步
  → 梯度方差的1000步滑动平均 → 长期统计
  → β₂过小(如0.9) → v_t不稳定 → 步长波动 → 训练不稳定

ε=1e-8 (防除零):
  θ = θ - α·m̂/(√v̂ + ε)
  → 当v̂≈0 → √v̂≈0 → ε防止除零
  → ε太大 → 步长被截断 → 学习慢
  → 1e-8是经验最优(大模型训练标配)

Bias Correction:
  m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t)
  → 初始几步: m_t偏向0(v初始化为0) → correction放大
  → t→∞时: β^t→0 → correction→1 → 不影响
  → PyTorch Adam/AdamW执行bias correction；不是默认关闭
  → t按参与更新的每个参数分别维护，缺失梯度时不推进，见后文CPU对照
```

### 2.2 Learning Rate 与 Warmup

```
LLM训练典型lr:
  Pretrain: α=3e-4 (GPT-3/LLaMA)
  SFT: α=2e-5 to 5e-5
  RL(GRPO/PPO): α=1e-5 to 5e-5
  我们实测: α=1e-3最优(小模型mini GRPO)

为什么RL lr比SFT更小?
  RL训练已经基于SFT模型 → θ接近最优
  → 大lr推离最优 → 破坏SFT知识 → 不稳定
  → 小lr → 只微调 → 保持SFT基础 → 强化

Warmup (0→α in first N steps):
  → 冷启动: Adam的m和v≈0 → 步长不稳定
  → warmup让m和v逐渐积累 → 步长稳定
  → LLM训练: warmup 2000步(GPT-3)/2%总步数(LLaMA)
  → 实测: warmup+constant ↓11.9% > cosine ↓8.7% (短训练)
```

### 2.3 Weight Decay λ

```
典型wd值:
  LLaMA: λ=0.1
  GPT-3: λ=0.1
  实测: λ=0.1最优(AdamW)

为什么wd=0.1而不是0.01?
  → wd从moment估计解耦，不从lr解耦；不能由此推导应设较大值
  → wd=0.1 → 每步衰减参数0.1×lr → 参数不无限增长
  → wd太小 → 正则不足 → 过拟合
  → wd太大 → 正则过强 → 学不到知识

数学: wd的效果
  θ_{t+1} = (1-α·λ)·θ_t - α·∇L/(√v+ε)
  → 每步参数被(1-α·λ)衰减 → 防止参数爆炸
  → α=1e-3, λ=0.1 → 衰减率=0.0001 → 温和衰减
```

## 三、AdamW vs 其他优化器

```
AdamW vs SGD+Momentum:
  → SGD: 所有参数用同一lr → 需手动调
  → AdamW: 每个参数自适应lr → 不需要
  → 实测: AdamW↓6.5% > SGD↓3.3% (小模型)
  → 大模型: AdamW收敛更快 → LLM标配

AdamW vs AdaFactor:
  → AdaFactor: 分解v_t为行×列 → 内存O(d) vs O(d²)
  → 用于大模型训练(T5)
  → 精度略低 → 但内存节省显著
  → ZeRO-3分片Adam → 内存已解决 → AdamW继续主流
```

## 四、AdamW 与策略优化的不同约束

优化器、优势归一化、梯度范数裁剪与策略ratio clipping是不同层次。GRPO的组内标准化不等于梯度裁剪，也没有因此取消ratio clip；[DeepSeekMath原论文式3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1)明确保留min/clip surrogate。PPO是否归一化优势由具体实现决定，不能单从GAE名称判断。

见[优势标准化与两种裁剪的CPU反例](backpropagation-theory.md#优势标准化与两种裁剪)：相同标准化优势与ratio=1，改变参数到logits的尺度仍使梯度范数增100倍；范数裁剪返回的是裁剪前值，不能把该返回值当裁剪后约束失效。

学习率、weight decay及clip阈值要随模型、奖励尺度与更新配置核对。SFT初始化不意味着已接近RL目标最优点，不能从该前提推出所有任务必须用同一个学习率区间或norm=1.0；裁剪也不保证消除reward hacking。

## 五、Adam 的预条件不等于自然梯度

```
自然梯度 (Amari, 1998):
  ∇_nat J = F^{-1}·∇J     #F=Fisher信息矩阵
  → 参数空间的"最短路径" → 考虑参数分布的几何

Adam使用历史g²的EMA，更新近似按1/sqrt(v)缩放，并使用一阶动量
  → 即使额外假设v能对应Fisher对角，1/sqrt(v)也不是1/v
  → 一般训练梯度的平方不自动等于Fisher对角
  → 可以称为对角自适应预条件，不能据此认证为自然梯度实现
```

## 六、PyTorch实现要点

```text
普通实数、非AMSGrad路径的示意，不是可直接运行的optimizer:
for p in params:
    if p.grad is None: continue
    若尚无state，初始化p自己的m、v、t为0
    t += 1
    p *= (1 - lr * wd)            # 先衰减旧参数
    m = β₁*m + (1-β₁)*grad
    v = β₂*v + (1-β₂)*grad²
    m_hat = m/(1-β₁^t)
    v_hat = v/(1-β₂^t)
    p -= lr*m_hat/(sqrt(v_hat)+eps)
```

先做Adam更新再衰减更新后的参数，会把Adam项也多乘衰减因子，不是这里的PyTorch顺序。参数是否参加更新由.grad是否为None筛选；requires_grad控制autograd，不在该筛选条件中。

内存按实际state dtype算：7×10⁹个FP32元素每份为28 GB（约26.08 GiB），m和v两份合56 GB。若某混合精度实现另有一份FP32 master weights，再加28 GB；并非所有PyTorch AdamW路径都强制创建master copy。本轮CPU FP64参数的m/v同为FP64，不是显存预算实测，也不能据此给出总训练内存占比或ZeRO固定节省倍数。

## 缺失梯度与零梯度

**None表示本轮跳过该参数，零张量表示以零梯度参加更新。** [zero_grad官方契约](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)明确两者行为不同；本机torch2.9.1的Adam._init_group只收集.grad非None的参数，再由单tensor实现递增该参数step、衰减旧参数和推进moment。zero_grad(set_to_none=False)只清零已有grad，不会给原本为None的参数补零。

真实CPU对照使用lr=0.1、weight_decay=0.2、betas=(0.9,0.999)、eps=1e-8、float64，禁用foreach/fused/capturable。先用autograd产生梯度1并step，参数从1变为约0.880000001，再清梯度并step：

| 第二步的grad | 参数值 | 本参数step | 一阶moment |
| --- | --- | --- | --- |
| None | 0.880000001 | 1 | 0.1 |
| 零张量 | 0.795394176514 | 2 | 0.09 |

None路径的参数及全部已存state与第一步后逐项相同；零梯度路径除了weight decay，旧动量也继续产生更新。把weight_decay设0的控制仍从约0.900000001变为0.832994176534。尚无历史state时则不同：首次None保留参数1、无state；首次零梯度把它衰减到0.98并初始化step=1。

双参数autograd控制先用p+q、下一轮只用p²。q第一步后约1.860000001；set_to_none=True时它保持该值，两参数step为(2,1)；False时已有q.grad被清成零，q继续变到1.755794176514、step为(2,2)。这是简单条件计算图的真实行为，不是MoE路由或分布式unused-parameter策略的验收。

[requires_grad_(False)](https://docs.pytorch.org/docs/2.9/notes/autograd.html#setting-requires-grad)阻止后续autograd累积该叶子的梯度，但不会替调用者清掉旧grad。显式保留第一步的grad=1后冻结p、仅对q反向，AdamW仍把p从0.880000001更新为0.76240000198；冻结后清为None的控制不变。动态冻结时要同时核对已有grad、optimizer参数集合和state保留意图，不能只改标志便断言旧梯度已失效。这不否定训练开始前冻结且从未有grad的正常用法。

[复查脚本](../../experiments/scripts/verify_adamw_missing_grad.py)直接执行真实Adam/AdamW与autograd。另以三步梯度1/0/−2的标量公式逐步比对参数、m/v和step：AdamW以及weight_decay=0的Adam均通过，核对bias correction与衰减顺序，不是用手写优化器替代实测。

```bash
python -B experiments/scripts/verify_adamw_missing_grad.py
```

环境为Python3.12.13/torch2.9.1，torch git `5811a8d7da873dd699ff6687092c225caffcf1bb`；脚本输出版本与zero_grad、_init_group、_single_tensor_adam源码hash。无随机输入、设备发现、模型权重/数据、AMP、DDP/FSDP、上游原测试或吞吐/显存比较。本题CPU语义已收束，不另开GPU/NPU任务。

## 恢复时的参数身份

**模型权重按名称恢复成功，不代表优化器动量也按名称关联正确。** 本节固定PyTorch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）的普通AdamW，不描述分布式checkpoint系统。[Optimizer.load_state_dict实现与其文档](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/optim/optimizer.py)先检查参数组数量和各组参数数目，再将保存的ID序列与当前参数序列zip配对。`param_names`不参与自动匹配；保存状态中有该字段时，还会覆盖当前组的名称字段。

### 同形状换序可以无报错地错配

真实CPU控制用两个FP64标量`left=1`、`right=2`，同一参数组，lr0.1、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，关闭foreach／fused／capturable。先给梯度1／−2更新一次，再deepcopy模型和优化器state_dict：参数约为0.900000001／2.0999999995，一阶动量0.1／−0.2，二阶状态0.001／0.004。

每条恢复轨迹都先用真实`Module.load_state_dict(strict=True)`按名恢复模型，并逐项检查权重相等。随后加载优化器状态，再以两个零梯度做一次原生AdamW更新；下表参数值取六位小数，完整状态见结果。

| 恢复方式 | left接到的一阶／二阶状态 | 下一步left／right |
|---|---|---|
| 当前参数仍按left、right排列 | 0.1／0.001 | 0.832994／2.167006 |
| 当前参数换成right、left，原状态不改 | −0.2／0.004 | 0.967006／2.032994 |
| 已换序，只把保存的param_names改为right、left | −0.2／0.004 | 0.967006／2.032994 |
| 已换序，同时显式对齐保存ID序列与名称 | 0.1／0.001 | 0.832994／2.167006 |

四次加载都不报错；第二行加载后组内名称甚至仍显示旧left、right，第三行名称看似正确但状态仍错配。shape／数量相同和名称字段存在都不能认证逻辑身份。另一个只有一个参数的加载控制会被原生数量检查拒绝，说明“能检测组大小变化”与“能检测同大小换序”并非一回事。

最后一行只是这个单组、唯一名称例子的显式对齐参考：按旧名称→旧ID关系重排保存的params列表，同时更新名称；没有更改state内容。它不是通用迁移器、自动验证器或生产pre-hook，多组变化、共享参数、模型重命名与分片布局都需要独立身份规则。不能把这一普通optimizer机制直接认定为verl、UniRL或name-aware FSDP／DCP／ZeRO的恢复缺陷。

### 复现与范围

[脚本](../../experiments/optimizer-state-identity/verify.py)与[完整结果](../../experiments/optimizer-state-identity/result.json)记录四条真实load／step轨迹、一项数量guard，以及原load_state_dict方法文本SHA256 `6da63f21961dae4c8e051bb768c6552372eed7f606a835b31a3e56f224243bd0`。两次新进程JSON相同，源模型和优化器快照未被其他轨迹改变。

```bash
python3.12 -B experiments/optimizer-state-identity/verify.py
```

环境为Python3.12.13／Torch2.9.1，无随机输入或新依赖；手动有限梯度、内存deepcopy与真实原生优化器，不是Mock。没有序列化／磁盘／scheduler、框架checkpoint manager、分片optimizer、AMP或设备／训练性能验收。本题在参数身份的CPU语义范围收束，不开社区缺陷、不改框架候选、不新排M5算力或调整其环境。

## 七、实测数据汇总

以下为2026-06-07历史小模型记录，本轮未复验其原始配置/日志；保留为历史线索，不据它推荐通用学习率、wd或判定Adam+L2必然发散。

```
实验 | 配置 | 结果
-----|------|------
AdamW(lr=0.001,wd=0.1) | 最优配置 | loss↓6.5% ✓✓✓
AdamW(lr=0.001,wd=0) | 无wd | 略差, 过拟合风险
Adam+L2(wd=0.1) | 灾难配置 | loss飙升47x! ✗✗✗
SGD(lr=0.1) | 非自适应 | loss↓3.3% (差)
Warmup+constant | 短训练最优 | ↓11.9%
Cosine decay | 长训练最优 | ↓8.7% (短训练更差)
Gradient accumulation | 等效大batch | 无额外内存(43.9MB不变)
BF16训练 | B≥128才有1.07x | B=16反而慢0.54x

→ 使用前需恢复同一模型、数据、超参、步数和测量条件。
  不从这张表推导普遍最优配置或ZeRO总内存固定倍数。
```

工具: `tools/mini_grpo_training.py` (优化理论实验)
