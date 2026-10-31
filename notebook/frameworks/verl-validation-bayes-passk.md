# Bayes@N与pass@k验证指标

2026-11-01补充：[失败补零与条件均值](#失败补零与条件均值)对照#8468的原四指标测试体；补零是计分政策，不是补出真实二元观测，也不保证请求已取消。

**先固定要回答的问题，再选择指标。** 平均单次成功概率、k次至少成功一次，以及固定题集平均成功概率的后验不确定性，是不同对象。Bayes@N不是pass@k的新名字；相同N与先验下，其后验均值也不是一种新的平均准确率排序。

阅读范围为[Bayes@N论文v4的二元特例与Algorithm 1](https://arxiv.org/html/2510.04265v4#S2)、[verl PR #8303](https://github.com/verl-project/verl/pull/8303)固定head `8971a9ff7666dd2584484d3c8f042d3da4162457`及[HumanEval估计器](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py#L13)。2026-10-21 06:32心跳核对#8303仍open，由方法作者mohsenhariri负责；这里做学习与既有笔记纠错，不实现竞争方案，也不将作者报告的108测试通过当成本机结果。

## 失败补零与条件均值

**“没返回分数”与“观测到错误答案”不是同一种事件；将前者计零，是明确选择失败代价。** [verl #8468](https://github.com/verl-project/verl/pull/8468)在2026-11-01读取为OPEN／draft，作者zzzzzzzxh，head `9b6ebeb4ca3828cc17ff110943f873f5f8739b14`、API目标base `8e48d72de4766228c36623c90f9f5c49ee5b45f3`。该草稿把预声明session计数用于reward／acc／score，保留有返回值的条件统计；正文已说明uni-agent #294独立集成、不再依赖本草稿，不应自行安排两仓合入顺序或另造竞争修复。

固定[指标函数](https://github.com/verl-project/verl/blob/9b6ebeb4ca3828cc17ff110943f873f5f8739b14/verl/trainer/ppo/metric_utils.py)先按(source,uid)收集非None值，再把缺少的数量补0。reward只要在infos_dict中声明，就可为清单中的全失败来源产生零指标；acc／score仅对实际存在至少一个非None观测的来源启用，不能因为另一来源有score就替它捏造score。`*_success_only`的success是流程有返回值，不是答案正确；得分0也保留，空列表则不产生条件均值。

候选的[原四测试体](https://github.com/verl-project/verl/blob/9b6ebeb4ca3828cc17ff110943f873f5f8739b14/tests/trainer/ppo/test_metric_utils_on_cpu.py)分别检查：

| 输入 | 补零结果 | 条件统计或来源限制 |
|---|---|---|
| 预声明5题各1次，仅2题返回0、1 | reward／acc／score均mean@1=0.2 | 三项success_only均mean@1=0.5，包含已观测的0 |
| reward-only来源3题，仅1题返回2 | reward mean@1=2/3 | 不添加acc／score |
| scored来源有score，另一来源的score只有None | scored score mean@1=0.5 | scored条件值1，另一来源不产生score |
| 清单5题、无返回行，但声明reward空列 | reward mean@1=0 | 用键存在性断言，避免defaultdict取值凭空制造0；不添加acc／score |

本机Python3.12.13／NumPy2.0.2／pytest8.4.2，从固定Git blob保留原`process_validation_metrics`、`bootstrap_metric`、`calc_maj_val`完整函数及四测试体，使用真实pytest.approx、不导入verl。基线四项均断言失败，候选四项均通过；原bootstrap在需要时仍执行1000次重采样，未替换为公式。命令：`python -B experiments/verl-validation-missing/verify.py /path/to/verl-git`；[脚本](../../experiments/verl-validation-missing/verify.py)及[结果与五输入hash](../../experiments/verl-validation-missing/result.json)。这是四个测试体的版本对照，不是原生pytest收集或作者83项套件。

### 分母完整不等于统一估计对象

若每题尝试次数不同，原聚合仍按指标键N分桶，并在拥有同一键的uid之间等权平均。自编两题控制：预声明次数2／4，分别只返回reward1／4，得到`mean@2=0.5`、`mean@4=1`，条件`mean@1=2.5`；不会自动返回所有6次尝试的加权均值5/6。报表若选某个最大N键，也不能把它称为全部题目的均值。补齐每题分母与选择跨题权重仍是两个决策。

补零也不保证数值必然下降：另一自编控制中，一次返回reward−1、预声明2次，条件均值−1、补零后−0.5。因而它实现的是“失败效用为0”的政策，不是任意reward上的惩罚或对未观察答案分数的无偏估计。下文旧#8303的Bayes路径则要求完整真实0／1试次并拒绝缺失；不应把不同版本的补零输出直接接成它已观测的Bernoulli数据，也未验证两候选组合。

### 统计结算与取消确认

[V1入口](https://github.com/verl-project/verl/blob/9b6ebeb4ca3828cc17ff110943f873f5f8739b14/verl/trainer/ppo/v1/trainer_base.py)在生成前建立(source,uid)清单并声明reward空列；收到空TQ batch时continue，计算指标后才调用可选publish钩子。生成、收集、日志或dump的前置异常仍可能阻止publish，代码不是覆盖所有失败的finally。局部指标测试不能证明完整验证一定结算。

[client取消分支](https://github.com/verl-project/verl/blob/9b6ebeb4ca3828cc17ff110943f873f5f8739b14/verl/workers/rollout/llm_server.py)保留已发generation_ref，在本地CancelledError后对该ref调用ray.cancel，再在线程中ray.get(timeout=5)。内部只吞TaskCancelledError；其他确认异常可从这里传播，5秒参数本身不证明远端终止或总能保留最初取消异常。finally调用release也不是等待路由器确认。以上仅源码契约，未用Ray替身宣称远端行为已通过。

本轮只执行CPU统计；五完整源输入AST／compile，未运行trainer、真实采样／评分、TQ、Ray、uni-agent、取消或训练。草稿的83项CPU／真实Ray actor及早期GPU观察均属作者报告。无需为这一局部统计检查新增算力交接；后续仅按实质反馈接续，不代建取消方案或改M5任务。

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

## 预声明题集与试次身份链

**完整观测清单必须先于生成结果，而不是从幸存结果反推分母。** 固定同一#8303 head，2026-10-21 07:03心跳接续。以下为源码链与选定片段控制，不是原生验证管道通过。

| 阶段 | 身份与检查 |
|---|---|
| [构造与验证入口](https://github.com/verl-project/verl/blob/8971a9ff7666dd2584484d3c8f042d3da4162457/verl/trainer/ppo/v1/trainer_base.py#L1034) | 配置guard在callback/replay构造前调用，_validate再次调用；只接受所支持的同步、采样、agent、grader与正整数N |
| 生成前建清单 | 每batch按prompt行生成新UUID，记录(source,uid)→N，跨batch累计；拒绝重复身份和不支持的逐题agent/N覆盖，然后才提交TQ与generate |
| [原生session与写入](https://github.com/verl-project/verl/blob/8971a9ff7666dd2584484d3c8f042d3da4162457/verl/trainer/ppo/v1/agent_loop_tq.py#L111) | _run_prompt按range(n)分配session_id，postprocess按uid_session_output_index写key；所读常规结果/Exception路径等待兄弟任务settle再报组终态，不作任意取消/进程退出保证 |
| 收集结果 | 原有逻辑先按session选最高output index；Bayes路径进一步要求最终key后缀0、UID匹配、session为十进制整数，再保留trial ID |
| 指标入口 | 原始helper对预声明题目检查每个trial恰好一次、范围0到N−1且acc为观测0/1；在旧补零逻辑前执行 |

UUID在每次_validate中重新生成，不是永久dataset ID或prompt内容哈希；“固定题集”是数据内容与实验设计条件，不能由UUID本身保证。key用rsplit从右拆两个分隔符，UID含下划线也不必被错误切碎。字符串不同的session如01与1解析成同一个整数，最终重复trial检查仍会拒绝。

[naive grader](https://github.com/verl-project/verl/blob/8971a9ff7666dd2584484d3c8f042d3da4162457/verl/experimental/reward_loop/reward_manager/naive.py)返回dict时保留score与acc等extra字段；标量结果则把score用作acc。_compute_score将reward_extra_info写入最终输出，TQ序列化后_validate收集；缺失的extra项在列表中补None，不等于已经观测到错误0。Bayes开启时由helper拒绝不完整／非二元acc，不能用连续score代替二元acc；关闭时原有缺失session补零策略仍是另一条语义路径。

### 失败发生在哪个阶段

前置配置／清单检查可在提交生成前拒绝不支持的实验；结果key检查发生在生成和读回之后。最终helper还位于每batch的kv_clear以及验证日志／可选dump之后，因此“检查失败”不等于全程没有副作用或有事务回滚。这里描述调用位置，没有注入真实队列故障或认证失败后恢复。

完整清单和一致ID是必要的观测结构，不是答案真实性、prompt归属、grader正确性或IID的证明。若key与存储UID一同被错误改写，单靠二者互相比较也不验证其内容。不能把这个原则边界直接称为本PR缺陷；它没有承诺对任意自定义组件做内容溯源认证。

### 选定原始片段控制

```bash
python3.12 experiments/scripts/verify_validation_estimands.py /path/to/verl-source --check-identity
```

新增21组控制保留原始_check_bayes_at_n_config，以及_validate中“UID已生成之后的manifest块”和最终key解析块，再接原始helper。覆盖支持／关闭选项、7类不支持配置、合法清单与4类非法batch、带下划线UID的逆序结果、4类非法key、重复数字session和整题缺失。使用属性字典配置替身；不运行构造函数、UUID生成、完整_validate、最终output去重、agent、TQ、grader、cleanup或日志。原682矩阵／重排、10非法输入、180期望检查均回归通过。

本机Python3.12.13/NumPy2.0.2，无新增依赖。既有独立torch CPU环境有torch，但缺hydra、omegaconf、tensordict、transfer_queue、ray与transformers；未安装或尝试宣称作者完整原生测试已通过。阶段收束为“数值矩与身份清单是不同证据层”，不继续堆同类替身替代原生管道，后续转读其他框架课题。

新增SHA256：trainer_base.py `6ab352b220cd2eec7f71315f5850a44f8eb3e4d39750c497cb3b6ef049a0e597`；agent_loop_tq.py `255ca2dd6b6149afca8a25df6f1dcd92874cd7a2e3ac503dd88309377ea6eea5`；naive.py `85a7a6872e2c9684c1252393755c8fe39d8fdf9393608c98cacd025821ba1d51`。
