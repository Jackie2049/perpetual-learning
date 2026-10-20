# GSPO：长度归一、token变体与梯度比较更正

来源：[GSPO论文v2](https://arxiv.org/html/2507.18071v2)。2026-10-21复核Eq.7/14后更正；旧稿可从Git历史回看，但“论文求和、verl求均值”“detach证明实现偏离”的判断撤回。

## 论文实际定义

```text
s_i = exp(mean_valid_tokens(log π_new - log π_old))         # Eq.7
s_i,t = stopgrad(s_i) * π_new,t / stopgrad(π_new,t)         # Eq.14
```

前者是按长度归一的几何均值，不是未归一的全序列概率乘积；后者正是论文提出的GSPO-token。不能因源码采用detach就判定与论文不符。token中间表达式的Jacobian不同，也不能跳过最终token均值归约而断言参数目标梯度不同。

## 限定数值对照

[修正脚本](../../experiments/scripts/verify_gspo_detach_gradient.py)以显式固定基点表示stop-gradient，使用NumPy/有限差分，无torch或autograd。报告版本verl `1876b06d0a3e4e71e06230be10af14492ca8a75b` 的源码形状/hash另核对，不执行原始函数。

```bash
python3 experiments/scripts/verify_gspo_detach_gradient.py /path/to/verl-8250
```

长度1/2/7/16、序列ratio0.5/1/2、优势-1/0/1，共36组目标值和梯度对照通过。适用条件：同一响应内优势恒定，相同有效token均值归约，避开clip拐点与源码max10数值闸门；**不证明任意token优势/掩码/额外权重/聚合方式/优化器或完整训练等价**。原脚本错误的“paper=sum”实现已替换，不能再用旧输出支持差异结论。

## clip需要分清对象与方向

报告源码对序列几何均值的ratio使用下界1−0.0003、上界1+0.0004。它不是每token KL硬界；token对数变化可在均值中抵消。窗外是否截断取决于优势符号和更新方向，不能说所有窗外样本都无梯度。

不能把该参数与PPO的0.2直接按数字比例解释为500倍失效，也没有证据证明它造成[#8250](https://github.com/verl-project/verl/issues/8250)坍塌；该报告另有相似配置稳定run。论文实验epsilon具体值本轮不作断言，数值边界也不是诊断阈值。

## 接续

训练根因仍需版本、总奖励分项、有效优势/mask与ratio观测。[诊断候选](../../open-source/verl/candidate-8250-async-entropy-collapse.md)已同步纠偏；[clip草稿](../../open-source/verl/verl-clip-semantics-rfc-draft.md)退回重审，无对外发布。
