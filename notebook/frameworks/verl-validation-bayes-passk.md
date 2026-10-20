# Bayes@N与pass@k验证指标

**先固定要回答的问题，再选择指标。** 平均单次成功概率、k次至少成功一次，以及固定题集平均成功概率的后验不确定性，是不同对象。Bayes@N不是pass@k的新名字；相同N与先验下，其后验均值也不是一种新的平均准确率排序。

阅读范围为[Bayes@N论文v4的二元特例与Algorithm 1](https://arxiv.org/html/2510.04265v4#S2)、[verl PR #8303](https://github.com/verl-project/verl/pull/8303)固定head `8971a9ff7666dd2584484d3c8f042d3da4162457`及[HumanEval估计器](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py#L13)。2026-10-21 06:32心跳核对#8303仍open，由方法作者mohsenhariri负责；这里做学习与既有笔记纠错，不实现竞争方案，也不将作者报告的108测试通过当成本机结果。

## 三种量分别估计什么

设固定题集有M题，每题N次二元观测，题i的未知成功率为p_i、成功数为s_i。

| 量 | 对象与条件 |
|---|---|
| 平均准确率 | 观测均值，估计固定题集的平均单次成功率 |
| pass@k | 独立同分布尝试模型下的平均至少一次成功概率，逐题为1−(1−p_i)^k |
| 本PR的Bayes@N | 每题独立Beta(1,1)先验下，固定题集平均p_i的后验均值与SD |

pass@k适合描述多次尝试的成功机会，不保证系统能够从k份回答中识别出正确答案；也不等于普通GRPO的训练目标。rollout_n只说明同题采样数量，不能单凭它推导best-of-N优化或pass@n等价。

## 二元后验与题集平均

每题后验为Beta(s_i+1,N−s_i+1)。由Beta矩得到

$$
\mu_i=\frac{s_i+1}{N+2},\qquad
v_i=\frac{(s_i+1)(N-s_i+1)}{(N+2)^2(N+3)}.
$$

在该独立题目后验模型下，题集平均的均值和标准差为

$$
\mu=\frac1M\sum_i\mu_i,\qquad
\sigma=\frac{\sqrt{\sum_i v_i}}{M}.
$$

不能平均各题SD，也不能把所有观测合成一个共享p的Beta后验。两题各两次时，`[[0,0],[1,1]]`与`[[0,1],[0,1]]`均值都为0.5，但平均p的后验方差分别为3/160与1/40；若错误地合并为Beta(3,3)，方差是1/28，已经换了模型。

同N时令观测平均准确率为a，可得μ=(Na+1)/(N+2)，是正斜率仿射变换，排序与a完全相同。N跨模型不同时不能套用：3/5的经验准确率高于29/50，后验均值4/7却低于30/52。全错仍可能报告正后验均值，来源是先验收缩，不是观测到新成功。

这里的SD是给定先验、题集与观测模型的后验不确定性，不是completion分数离散度、bootstrap标准差、跨新题泛化误差或现成置信区间。PR只加均值与SD，不自动提供可信区间或胜负判定；采样参数和不同session ID也不能证明真实观测独立。

## PR怎样守住输入边界

[_bayes_at_n_metrics](https://github.com/verl-project/verl/blob/8971a9ff7666dd2584484d3c8f042d3da4162457/verl/trainer/ppo/metric_utils.py#L898)按(source,uid)区分题目，要求预先声明题目和试次数，每个trial ID恰好出现一次且acc为有限0/1。它在原有缺失session补零逻辑之前调用，所以开启时不能把缺失当成真实错误观测。缺题、重复／越界trial和非二元acc都应报错，而不是给出看似完整的后验。

PR的配置范围还限于V1 sync、fresh随机single-turn、注册naive rule grader；没有把一般连续训练reward改为二元，也没有覆盖论文的分类权重、先验历史或区间功能。这些入口限制在源码与PR说明中可见，本轮只执行下面的数值helper，不认证完整构造/调度/grader路径或参数检查已经证明IID。

## pass@k的组合估计与旧笔记纠正

HumanEval对n份回答、c份正确、1≤k≤n使用

$$
\widehat{\operatorname{pass@k}}=1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

n−c<k时失败组合数为0；k=1还原c/n。比如n8/c2/k2为13/28。它是从n份样本中取k份的成功比例；在同题IID Bernoulli模型下，对1−(1−p)^k无偏。无偏不等于低方差、所有采样策略都独立，或与平均准确率拥有相同排名。例如两模型p分别为(1,0)和(0.4,0.4)，平均准确率0.5>0.4，而pass@2为0.5<0.64；它们回答的是不同效用问题。

旧[rLLM阅读](rllm-v0.3-terminal-rl-reading.md#4-passk-evaluation--pr-649)的组合公式、Kolmogorov归属，以及“普通GRPO rollout_n等价pass@n／训练评估完美对齐”撤回；只修数学和论证，不将本轮当作rLLM当前CLI或#649代码复核。旧实践指南的正确组合公式保留并补适用条件。

## 本机执行与适用范围

```bash
python3.12 experiments/scripts/verify_validation_estimands.py /path/to/verl-source
```

[脚本](../../experiments/scripts/verify_validation_estimands.py)从固定Git对象AST提取未改写的原始helper，不导入完整verl。Python3.12.13/NumPy2.0.2执行682个M,N各1至3的全部二元矩阵，与Fraction精确后验矩比较，绝对容差1e−14；另做682次保留trial身份的逆序到达控制和10组非法输入拒绝。独立Bernoulli模型以有理数验证180组组合估计期望；不是执行HumanEval、rLLM、原始trainer测试或真实模型采样。

同总成功数但不同逐题分布、不同N的排序反例均通过。没有运行作者的108测试、完整CPU CI、TQ/Ray、训练或GPU；也未复现论文的经验收敛／排名收益。源码SHA256：metric_utils.py `f3aabf26b0de5ecb98df4a0821c8938a247980e955aa42dc7701029568da1a7a`。下一步如继续，只读追预声明题目/trial身份如何到达helper，不能由数值正确推断端到端cohort正确。
