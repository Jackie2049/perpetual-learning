# GMPO：完整裁剪目标、符号分支与旧结论更正

来源：[论文 v1 Eq.4 / Algorithm 1](https://arxiv.org/html/2507.20673v1)；[作者代码固定提交](https://github.com/callsys/GMPO/blob/ae9cbebd8cc6b58a6db6c61cb90e4b13ff131d94/train_zero_math_gmpo.py#L675)；[verl 报告版本](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py#L1928)。2026-10-21 重审。关联：[损失综述](../rl-policy-loss-functions-mathematical-derivation.md)、[GSPO 更正](gspo.md)。

**结论**：论文 Algorithm 1 本身在 log 空间裁剪，默认 epsilon=0.4 对应 exp(±0.4)，不是 1±0.4。“论文与 verl 配置空间不一致”的旧判断撤回。原脚本只比较双侧 clamp、缺优势符号与最终目标，不能支持完整等价；当前只在明确条件下建立代数对应和有限数值验证。[旧 RFC](../../open-source/verl/verl-clip-semantics-rfc-draft.md)不再独立推进，没有新上游缺陷或训练根因结论。

## 从完整目标推导

设有效 token 对数比率 $z=\log\pi_{\mathrm{new}}-\log\pi_{\mathrm{old}}$、同一响应恒定优势 A、$s=\operatorname{sign}(A)$，有效 token 数 N>0。以下均是要最小化的负收益，不含辅助损失。

论文 Eq.4 先对 **r^s** 做 min/clip，再取 s 次幂与几何均值。写成 log 形式：

$$
\begin{aligned}
\phi_{\mathrm{paper}}(z,s)&=s\min\bigl(sz,\operatorname{clamp}(sz,\log L,\log U)\bigr),\\
\ell&=-A\exp\bigl(\operatorname{mean}_{\mathrm{valid}}\phi_{\mathrm{paper}}\bigr).
\end{aligned}
$$

这里 L/U 是 signed ratio 的阈值。Algorithm 1 取 log L=−ε、log U=ε；作者代码用相反 sign 与 max，标量优势、对称窗口下是同一代数形式。只读过该分支，未运行作者训练程序，也未验证其 batch 约束。

verl 指定版本的次序不同：先 clamp **原始 z**，再乘优势 sign、取 min、乘回 sign：

$$
\phi_{\mathrm{verl}}(z,s)=s\min\bigl(sz,s\operatorname{clamp}(z,-a,b)\bigr).
$$

它化简为：

| 优势 | phi_verl | 哪一侧被截平 |
|---|---|---|
| A>0 | min(z,b) | z>b；相反的低比率尾部仍有梯度 |
| A<0 | max(z,−a) | z<−a；相反的高比率尾部仍有梯度 |
| A=0 | 孤立 PG loss 为 0 | 不推论辅助项也为零 |

因此不能把双侧 clamp 本身当完整 PG 目标，也不能说所有窗口外 token 均被裁平。各有效 token 的内部导数在未裁平侧为 loss/N，在裁平侧为 0（这里对 z 求导，非模型参数/优化器或 clip 拐点）。

## 什么条件下对应

- **对称窗口 a=b=ε**：Eq.4 取 L=exp(−ε)、U=exp(ε)，Algorithm 1 与 verl 的理想化非空响应目标相同。未裁剪时，序列几何均值给响应内 token 共享权重；不是未归一全序列乘积。
- **一般原始 ratio 窗口**：常规 PPO `min(A*r, A*clip(r, exp(-a), exp(b)))` 取绝对值后的几何均值再恢复 A 的符号，与上述 verl 理想模型对应。正负优势均须包含；旧笔记漏写了负优势分支。
- **Eq.4 非对称 signed-ratio 窗口不可直接沿用同一 L/U**：其正优势有效上界为 U，负优势原始 ratio 下界为 1/U；verl 对应 exp(b)/exp(−a)。若 a≠b，直接套同一 signed window 会不匹配，需要依 sign 映射。例 a=.1,b=.4,A=−1,z=−.3：Eq.4 loss=.740818221，verl=.904837418。论文/作者默认是对称窗口；这只是参数映射反例，未证明实现违背其非对称配置契约，不据此报 bug。

log-ratio z 是采样 token 的对数概率差，**不是分布 KL**。上述阈值不应写成“KL ±ε nats”的硬界；与 GSPO 序列几何均值之后裁剪也不是普遍等价。

## 验证与边界

```bash
python3 experiments/scripts/verify_gmpo_clip_semantics.py /path/to/verl-8250
```

[脚本](../../experiments/scripts/verify_gmpo_clip_semantics.py)仅 NumPy 模型与有限差分，对输入源码做 AST 形状检查并打印 SHA256，**未执行原始 torch 函数/autograd**。Python 3.9.6 与 3.12.13、NumPy 2.0.2 均通过：

- 36 组长度1/2/7/16、epsilon .0003/.2/.4、优势−2/0/1.5，含部分有限 padding，比较完整 Eq.4/Algorithm 1/verl 理想目标值、有限差分及独立分支解析导数；避开裁剪拐点。
- 原始 ratio 非对称窗口的正负优势值/梯度对照；signed-window 不匹配及旧 clamp-only 漏项反例。
- 有限 padding 不影响有效值；混合 token 优势不能用其均值替代论文恒定序列优势；空响应和外层平均分母控制。

verl 原代码两处 N+1e−8 稳定项使其与理想 N 分母**并非逐位严格相等**；上述样本最大值差约1.3650e−7，不能把这一观测当普适误差上界。源码哈希 `a54b6d812db4441e8b0989ae69d2c46b0a0a7b584296f73d8316f264462c37cb`。未覆盖非有限输入、任意 token 优势、rollout IS 权重、真实训练/优化器、分布式归一或模型参数梯度。

外层 `torch.mean(pg_losses)` 是响应行均值，空响应产生0但仍计入分母。局部均值直接累加不自动等于全局均值；脚本仅演示该算术事实。**不能由此认证 #7913 的完整修复**：还需核对真实 actor/engine 的 microbatch 缩放、DP 归约、全局分母和空响应政策。旧“任意切分修复后即论文目标”的无条件表述撤回，本轮未改 #7913 patch、未运行旧 split 脚本，也未改变上游竞争状态。

## 方法教训与接续

原脚本多数判断只打印 allclose，未 assert；即使输出通过，也仅覆盖了 clamp，不是完整符号目标。新的回归检查必须包含最终标量损失、正负优势、有效掩码和外层分母，并明确数值稳定项与真实框架边界。#7913调用链已在下一轮完成限定复核：旧DP段误把loss当梯度，不能作真实DP修复证明；#7916同样使用agg_loss，停止重复候选。详见[更正记录](../../open-source/verl/candidate-7913-gmpo-geo-mean.md)，仍未运行native训练，不承接GPU或AReaL工作。
