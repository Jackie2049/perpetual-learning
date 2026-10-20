# verl：异步训练坍塌的诊断边界与待补观测

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 区分异步训练异常的观测与原因，纠正旧诊断草稿中的过强推断。[Issue #8250](https://github.com/verl-project/verl/issues/8250)

**【类型】** 问题诊断

**【价值评估】** 6 / 10 — 排障路径有复用价值，但缺原始数据与受控对照，旧因果结论已撤回。

**【进展】** 2026-10-21待补证，旧英文稿不可直接发布；V1策略/logprob来源与最小观测契约已补，7组原始路由替身控制及7类数学反例通过，没有确证根因、修复或训练实测。上游状态见00:01核验，发言仍需用户授权。

**【大白话讲明白】**

日志显示模型生成与训练指标先后恶化，但并不能单靠曲线确定是谁先出了错。结果奖励为零，也不代表总奖励相同或所有梯度为零。我们撤回旧稿把假设写成定论的部分，改为先对齐模型版本、样本、奖励分项和有效掩码，再用对照定位问题。

**【What–Why–How】**

- **What：** 诊断旧策略采样、训练更新、权重同步和奖励变化可能造成的相似症状。
- **Why：** 汇总日志缺少组内奖励/优势、ratio、版本与掩码信息，不能唯一确定故障链。
- **How：** 固定报告版本核对指标和目标，用可复现数值反例排除错误推断，再列最小补证清单；不是完整 verl 或 GPU 实测。
<!-- contribution-summary:end -->

## 当前证据与状态

2026-10-21 00:01 心跳读取 issue 全文、timeline 与编号 PR search：open、0评论、无assignee；timeline无交叉引用，编号查询 total_count=0、incomplete_results=false。**这不是全仓库无人修复的证明**。没有发布评论、PR或修改上游代码。旧稿可从 Git 历史回看，但不能复用 `/tmp/comment8250_draft.md` 的旧结论直接发言。

报告的9B基线为 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，还有私有修改；旧27B精确版本未知。4组run不构成受控消融，C为相近GSPO配置稳定的反例；D的结果/完成率虽为0，总奖励仍约0.28，因另有format/efficiency分项。不能写成“所有奖励全0”。vLLM安装版本尚未确认，仓库pin不是运行环境证明。

## 撤回与更正

| 旧表述（撤回） | 当前允许的判断 |
|---|---|
| KL runaway→优势离群→更新爆炸已确证；grad爆炸必然伴随优势离群 | 只是若干假设；标准有限、有效组n>1、std归一GRPO有组内幅度界，ratio/掩码/聚合/模型Jacobian也影响梯度 |
| result_mean=0→所有组A=0→全梯度0；不是mask问题 | 先查**总**奖励组内差异、实际优势及有效mask。A=0可使孤立PG项为零，但不能排除辅助项/其他分支；A非零也可因clip侧、零mask等无PG梯度 |
| teacher-forced不一致等于weight-sync/cache故障，足以判根因 | 先固定完整token/掩码、权重版本、采样变换和数值容差；三路对照可缩小范围，不能唯一定位 |
| rollout_corr/kl是reference KL，可直接决定KL系数 | 源码是采样token上 rollout_log_prob-old_log_prob 的masked mean；不是对reference的KL，有限样本也非精确非负分布KL |
| GSPO窗外token都无梯度，1±0.0004是token KL硬界，导致无法恢复 | clip取决于A的符号和更新方向；几何均值ratio不是每token KL界。无ratio/clip与符号分布不能断言原因或恢复性 |
| GSPO论文用全序列乘积，verl均值/detach因此偏离论文 | [论文Eq.7与14](https://arxiv.org/html/2507.18071v2)明确长度归一与GSPO-token；原比较漏项，已更正[论文笔记](../../notebook/papers/gspo.md) |

## 源码与限定验证

核对报告版本的 [core_algos.py](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py)（GRPO/GSPO）与 [rollout_corr_helper.py](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/rollout_corr_helper.py#L904)。GRPO使用[torch.std默认sample correction=1](https://docs.pytorch.org/docs/2.14/generated/torch.std.html)；对有效组n=8、有限reward、std归一分支，理论界为7/sqrt(8)，epsilon只缩小幅度；不包括关闭归一、后续额外权重、组大小变化或非有限输入。

```bash
python3 experiments/scripts/verify_gspo_detach_gradient.py /path/to/verl-8250
```

这是带源码形状检查/hash的NumPy模型，**未执行原始torch函数或autograd**。36组限定目标值/有限差分梯度对照通过：同一响应A恒定、相同token均值归约、避开clip和max10边界时，GSPO与GSPO-token目标梯度一致。窗外四象限导数显示“全部归零”不成立；1,000组有限GRPO奖励满足界2.474873734；result全0但其他奖励变化仍可有非零A；非零A可落在零梯度clip侧。独立toy辅助KL也可在A=0时有梯度，不代表D实际启用了KL。

环境Python3.9.6/numpy2.0.2。源码SHA256：core_algos.py `a54b6d812db4441e8b0989ae69d2c46b0a0a7b584296f73d8316f264462c37cb`；rollout_corr_helper.py `763158342551bc4077805a8a009d959a7a2043c4d43b772fe6045f4a40932f3e`。无模型、优化器、真实mask、私有数据、GPU训练或因果复现。

## 建议补证顺序（待授权交流，不给固定“治愈阈值”）

1. 对齐每条样本的uid/group、总奖励与所有分项、组内标准差、优势分布、有效token/mask、是否拒绝或过滤，以及clip按A正负的统计。
2. 对齐rollout生成版本、old-logprob重算版本和训练update版本；先记录bypass/解耦路径及CPU快照trigger，再比较固定完整token上下文。版本标签、采样token概率与安装后tensor内容是不同证据；同backend干净重载与live对照只能缩小范围，不唯一指向同步/cache。详见[来源映射和对照条件](../../notebook/frameworks/verl-async-logprob-provenance.md)。
3. 获取首次异常更新前后的有限性、ratio分位数、梯度测量位置（裁剪前后）、有效样本量和同步事件；再分别控制reward、staleness、融合路径等变量。

未得到数据前，不把0.05或某KL系数当通用阈值，不宣布mask、数值、同步或reward分支已排除。[clip旧RFC](verl-clip-semantics-rfc-draft.md)经GMPO复核后已撤回且不再推进；本候选仍待补证，发布边界不变。
