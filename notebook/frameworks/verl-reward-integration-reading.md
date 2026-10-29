# verl Reward Integration 源码级深度阅读

2026-10-30接续：[Minerva装饰与答案身份](#minerva-装饰与答案身份)核已有#8450：原21参数测试体候选全过／基线14断言失败，但末尾实质星号被删可造成符号串误接收；反引号跨行和冒号前标签仍未覆盖。已有bot反馈，本机只复核、不补丁或发布。

2026-10-26状态纠正：[数学奖励候选关闭与方程后续](#候选关闭与方程后续)确认#8347关闭未合并、主线三个评分文件仍同旧基线；方程提取已有新贡献者认领，不把旧64项通过当成已部署修复。

2026-10-26接续：[colocated 组装与 V1 生成 mask](#colocated-组装与-v1-生成-mask)核对新#8379补上#8368明确排除的路径；原13参数项候选通过，实际TQ服务／RM推理与完整PPO更新仍分开验收。

2026-10-26补充：[DisRM返回契约与部分写入](#disrm-返回契约与部分写入)核对已有#7608：原三回归候选过；旧版缺字段时已写分数，同一输出再次调用会跳过评分，不能据“没再报错”认证恢复。

2026-10-25补充：[异步奖励的验证标志与元数据范围](#异步奖励的验证标志与元数据范围)核对已有#7217：原两参数用例通过，修复显式验证标志，不透传全部批级元数据；原假远程句柄不是实际奖励服务。

2026-10-25补充：[数学表达式的语法与资源边界](#数学表达式的语法与资源边界)比较两项已有eval替换方案：限制可执行语法、保持答案兼容和限制资源是不同要求。只做原方法及良性输入CPU对照，不运行攻击或耗尽载荷。

2026-10-24补充：[代码判分的格式与逐例接线](#代码判分的格式与逐例接线)比较#8360/#8175原测试：两候选在部分分数和字典参数上各有不同覆盖，不能以“支持APPS”互相替代。

2026-10-24补充：[数学评分池的恢复与错误信号](#数学评分池的恢复与错误信号)用真实math-verify及spawn进程核对#8015：旧版进程死亡/超时留池反例成立，候选原七测试两次通过；超时回默认分仍不等于上层可识别超时。

2026-10-24 补充：[异步奖励的补齐边界](#异步奖励的补齐边界)用 #8101 原张量构造与消费者片段核对末条 response 的读取长度；仅 CPU 几何对照，不是完整 worker/TQ 或训练验收。

2026-10-23 补充：[沙箱运行器注册与执行边界](#沙箱运行器注册与执行边界)固定于未合并 #8345，14 个原 runner 用例仅验证本地契约；[数学奖励的分层判分与精度边界](#数学奖励的分层判分与精度边界)固定于未合并 #8347，仅覆盖 CPU 字符串判分；[Prime 进程池的结果协议与失败范围](#prime-进程池的结果协议与失败范围)用 #8217 原测试核对真实 CPU 进程池。其余 2026-06 架构与硬件建议不属于这些复查范围。

> 2026-06-15 | 源码: verl/trainer/ppo/reward.py + ray_trainer.py + workers/reward_manager/ + experimental/reward_loop/ + core_algos.py + utils/reward_score/
> 核心: 两层架构(legacy RewardManager + experimental RewardLoop) → 3 reward路径(rule/DisRM/GenRM) → outcome reward=最后token非零 → GRPO group-relative → colocate RM sleep/wake → RTX 4090=rule-based唯一可行

## 整条奖励的落点与回报传播

**奖励总和没变，不代表优势估计器消费到了它。** 2026-10-26核对新[#8367](https://github.com/verl-project/verl/issues/8367)及已有OPEN/non-draft [#8368](https://github.com/verl-project/verl/pull/8368)。报告者的SWE轨迹末尾为模板换行、response_mask=0；原agent loop把整条reward_score放在最后响应位置，GAE／REINFORCE++会跳过这个位置的奖励。作者同时指出其SWE默认配方用GRPO，不属于该失败条件；不能把它扩写成默认SWE训练已坏。

固定head `9cf35a1d3692e2680c96e5c594e204638fcf08bc`，base／共同祖先与当轮main均为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，ahead2／behind0、两文件71增5删。候选只改[AgentLoopOutput.as_dict与AgentLoopWorker._postprocess](https://github.com/verl-project/verl/blob/9cf35a1d3692e2680c96e5c594e204638fcf08bc/verl/experimental/agent_loop/agent_loop.py)：把分数放在response_mask最后一个非零位置，而非更后的模板后缀；未改变评分器看到的轨迹或优势估计器。没有生成位时，未补齐输出仍回退最后响应位置，补齐batch回退最后有效响应位置。这是兼容回退，不保证全零mask行能产生学习信号；也不应通过把模板mask改成1来绕过问题。

### 两个原回归与一个数学对照

[复现脚本](../../experiments/verl-outcome-placement/verify.py)从固定Git对象装载原Pydantic输出类、DataProto／DataProtoItem和完整_postprocess方法，类壳只绕过worker初始化。两个新增原测试体及其_pad_1d／_to_internal助手不改：as_dict使用mask=[1,0,1,0]，postprocess同时检查模板尾部、补齐行和无生成位回退。base两项断言预期失败，head两项通过；真实CPU torch／TensorDict及Pydantic，不是完整原pytest收集或Ray worker。

另按issue的六token几何构造一个本地输出：mask=[1,1,0,1,1,0]、reward_score=1。经各版原as_dict生成rm_scores，再交给原GAE／REINFORCE++、原masked_whiten等函数，gamma=lam=1、V=0；仅跳过注册装饰器，RF配置是只含gamma的本地对象。

| 版本 | 非零奖励落点 从0计数 | 奖励总和 | 首token GAE return | 首token RF++ return |
| --- | --- | --- | --- | --- |
| base 3c6d9b43 | 5 即mask为0的末位 | 1 | 0 | 0 |
| head 9cf35a1d | 4 即最后生成位 | 1 | 1 | 1 |

head完整GAE returns为[1,1,1,1,1,0]，RF++ returns为[1,1,0,1,1,0]；中间观察位是否显示carry与最终优化位不是同一层。两个估计器AST及完整core_algos文件在两版相同。GRPO按整行sum获取分数的源码路径与这两者不同，本轮未执行GRPO／其余估计器，不将报告者14种估计器检查计为本项目结果。

**return恢复不等于本例已有非零actor优势。** 此单轨迹在四个有效位置的return全为1，masked_whiten后有效位优势仍全部为0。GAE在被mask的末位还产生−10000，这是本例零方差下对非有效位置的标准化数值，不是有效token上的异常、模型不稳定或新候选缺陷证明。这里只保留原输出，不另改标准化／掩码策略，也未执行loss或梯度。

### 不同奖励路径仍需分开

作者明确排除后续token-level rejection改mask的情况，以及colocated RewardManagerBase.assemble_rm_scores路径。前者涉及return mask与loss mask及已混入KL项的奖励语义，不能机械在所有估计器里把分数搬回去；后者也不因agent loop两处通过就视为修好。这些是已有提案的范围边界，不另开通用mask修复或修改M5专项。

[JSON](../../experiments/verl-outcome-placement/result.json)保存两版原测试结果、奖励／return／优势及九份源码hash。Python3.12.13、torch2.14.1、Pydantic2.14.1、NumPy2.5.4、TensorDict0.15.2，两次语义一致；九份完整模块AST／compile通过。作者21／42项、mutation与真实SWE重建／tokenizer结果均非本机验收；本轮没有模型下载、配方重建、完整训练或GPU操作，不新排M5队列。复现运行 `python -B experiments/verl-outcome-placement/verify.py <已有固定对象的只读verl仓库>`。现有候选已有作者，不另实现或发布；下轮转独立需求／实质反馈，不扩mask／估计器矩阵。

### colocated 组装与 V1 生成 mask

2026-10-26新增OPEN／非draft [#8379](https://github.com/verl-project/verl/pull/8379)，作者Levius-Fubuki，head `0447a46d59b842da56e38e7d25b0d05e47f6b42e`，父版／base／main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，五文件209增9删。这不是#8368重复提交：新候选只改RewardManagerBase的组装和V1 colocated输入，不改agent-loop或估计器；两个候选不能因主题相同就互相替代，也未在本机拼接。

组装器用 `response_mask.bool() & response_attention_mask.bool()` 找最大的生成且非padding索引，避免把奖励放到后面的模板后缀或带错误生成标记的padding。没有mask／行内没有生成位时仍回退 `attention.sum()-1`，延续原右补齐有效前缀假设；它不是任意稀疏attention的通用最后非零算法，也不保证全零生成mask行能产生学习信号。新空batch用例为0行、响应宽4，不认证零响应宽度。

V1另从TQ明确请求保存的 `response_mask`，转成padded后放进RM输入；attention仍由prompt／response各自行长度重建，不用生成mask裁剪评分上下文。主线agent-loop已有输出该字段的接线，原V1却没在此入口取它。**组装器可接受缺mask，不等于新V1入口也可缺字段**：后者无条件请求并索引TQ中的mask；原缺字段兼容用例只测试组装器，不是所有历史TQ记录兼容保证。

[本机脚本](../../experiments/verl-colocated-outcome/verify.py)用完整原DataProto类／typed AlgoConfig、三原估计器、masked helpers、原assemble方法和V1两方法类壳，保留原get_tensordict。候选新增assembly测试文件的全部原函数，加V1新增测试与原StubTrainer，由pytest在内存中按原参数／断言收集；跳过包／服务初始化及估计器注册，不改测试数值。Python3.12.13／torch2.14.1 CPU／TensorDict0.15.2／NumPy2.5.4／pytest9.1.1，11个组装／估计器项加train／val两个V1项：候选13过，父版4过9失败。父版7个组装／return项断言失败，V1两项均因RM输入缺response_mask失败。[父版](../../experiments/verl-colocated-outcome/base-result.json)／[候选结果](../../experiments/verl-colocated-outcome/head-result.json)保留每项结果和源码hash。

原夹具含内部观察位、模板尾、补齐和2／−1两种outcome；bool／int／float mask项只用二值mask，不能扩为任意浮点权重语义。V1原测试保留TQ读写与RM推理替身，但实际执行原方法、真实jagged张量与DataProto：评分输入attention仍含模板尾，写回长度仍为6／3，奖励分别位于从0计的4／1；train与val是分区参数检查，不是启动两套队列服务。三估计器用例执行真实GAE／REINFORCE++／ReMax，确认有效位置returns保留2或−1；反向用 `-(log_probs * returns.detach() * mask).sum()`，log_probs是独立叶子。它既未消费估计器返回的advantages，也不是实际PPO clipped loss、actor参数或AdamW更新。

作者另报63项套件在两版torch通过及CUDA合成PPO／AdamW smoke；均为外部报告，不计入本机13项，也不是完整rollout／RM推理训练。本轮未执行真实TQ服务／RM模型、完整trainer、CUDA、多GPU或训练，不重跑#8368旧控制。缺口补[M5原条件入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用候选、新排训练或改其专项；有实质反馈或真实匹配需求才接续，不扩mask矩阵。任何社区写入和已有PR-head推送仍逐项待用户明确同意，PR正文的外部提交授权不改变本项目权限。

复用：已有固定Git对象的只读verl checkout，执行 `python experiments/verl-colocated-outcome/verify.py /absolute/path/to/verl base` 或末参 `head`。base返回脚本exit0表示记录到了9个预期失败，不表示原测试全过；无需网络／模型下载。

## Minerva 装饰与答案身份

**格式容错不能自动把边界符号当装饰。** 2026-10-30读取RerankerGuo的[#8449](https://github.com/verl-project/verl/issues/8449)与已有OPEN／非draft [#8450](https://github.com/verl-project/verl/pull/8450)，不另认领修复。固定head `e8e82c2fc349c64e199392eaf6f7aaf79d185d42`，API目标base为main `a08ea02a784d7d4982b1d81df36d188a476192bb`，实际共同祖先却是 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`（ahead1／behind22）。祖先与该main的math_dapo逐字节相同，下面仅据这一评分文件对应主线，不称整仓库相同。

原Minerva优先匹配最后一个 `Answer:`，再走指定文本normalizer。候选增加装饰剥离：成对包裹先处理，残留单侧星号／反引号再移除；下划线仅剥成对包裹，以保留 `x_1`／`_1`／`x_`。默认正则另允许冒号后的强调标记跨空白跳到答案。**匹配标签、取答案跨度、清理装饰、归一化与选择后备**是不同步骤；只看最终21个绿色断言不能判断所有步骤都保持答案身份。

### 原测试覆盖与反向边界

Python3.12.14／仅标准库，完整执行两版[原评分器](https://github.com/verl-project/verl/blob/e8e82c2fc349c64e199392eaf6f7aaf79d185d42/verl/utils/reward_score/math_dapo.py)，直接调用[原测试](https://github.com/verl-project/verl/blob/e8e82c2fc349c64e199392eaf6f7aaf79d185d42/tests/utils/reward_score/test_math_dapo_minerva_markdown_on_cpu.py)九函数展开的21参数体：基线7通过／14断言失败，候选21通过。测试体、参数值和断言保留，只去pytest／框架imports及parametrize装饰器；不是原生pytest收集，也不是作者39项奖励manager套件。所测多行字符串经解析确有真实换行，不是字面反斜杠n。

两条已有[bot行内反馈](https://github.com/verl-project/verl/pull/8450#discussion_r4415579531)、[格式边界反馈](https://github.com/verl-project/verl/pull/8450#discussion_r4415579540)涉及下列问题，本机固定head复核，不将建议当执行指令或新发现。读取时唯一review为CodeRabbit的COMMENTED（针对旧d7d931de），无人工批准；当前两条行内记录标head e8e82c2f，结论以实际代码为准，不照搬“低风险”摘要。

| 小输入与目标 | 基线输出 | 候选输出 | 所说明的边界 |
|---|---|---|---|
| `Answer: x*`，目标`x*` | +1，pred=`x*` | −1，pred=`x` | 清理删掉答案的末尾符号 |
| 同一输入，目标`x` | −1，pred=`x*` | +1，pred=`x` | 在本符号串契约下从拒绝变误接收 |
| 反引号包裹Answer标签，值在下一行 | −1，pred为反引号 | −1，pred=`[INVALID]` | 冒号后跨行规则未含反引号，并非本候选新引入的拒绝 |
| `**Answer**: 42`，目标42 | −1，`[INVALID]` | −1，`[INVALID]` | 标签结束标记在冒号前，根本未匹配 |

这些是有限输入的字符串判分控制，不是符号数学等价器、常见数据频率或训练质量结论；末尾星号也不能仅因“通常是Markdown”就对全部目标删去。测试保住内部 `3*4` 不代表保住边界 `x*`，保住下划线也不等所有运算／标记语义无损。没有在候选上回改正则或剥离策略。

### 后备代码未改但可达性会变

六个既有函数（boxed提取／去壳、normalizer、strict box、verify、compute_score）两版AST相同。候选把空装饰串改回 `[INVALID]`，因此 `\boxed{42}` 后接空 `**Answer:**` 的本地控制由−1变+1：不是boxed函数变了，而是现在能走到后备。相反，明确提取出错的 `Answer: 41` 即使旁边有正确box，仍−1；strict模式绕过Minerva则给+1。原[后备选择与窗口](#boxed-答案的归一化与选择优先级)条件不变，不能写成“Minerva判错就再选box”。

本地 `\boxed{1,000}` 对目标1000两版仍−1；它属于另一项boxed归一化需求，而非本次Minerva修复。未把#8124拼进#8450或将其旧9项移作当前候选验收。

[复现脚本](../../experiments/verl-math-dapo-markdown/verify.py)和[逐项结果／四源码hash](../../experiments/verl-math-dapo-markdown/result.json)另存八个本地边界控制，区别于21原测试；两次结果与JSON相同。运行 `python3.12 -B experiments/verl-math-dapo-markdown/verify.py /path/to/verl-git`，需已有固定Git blob、禁止隐式fetch。无第三方依赖、verl父包／模型／worker／设备导入、完整manager或训练；不据bot工具警告声称已验证ReDoS。CPU证据收束，无需新GPU队列，候选有实质修订再接续；不修改M5专项或对社区写入，所有上游动作／PR-head推送仍须用户逐项明确授权。

## boxed 答案的归一化与选择优先级

答案的字符串身份、提取优先级和数学等价性是三个要求。[#8123](https://github.com/verl-project/verl/issues/8123)已有报告者KardeniaPoyu的[#8124](https://github.com/verl-project/verl/pull/8124)，2026-10-25读取为open、非draft、未合并，不另实现。固定head `c5e26eec5ab5941e443d1b9f6853a9e210972487`，父提交／API base／共同祖先均为 `8718ca30a3f002f93b7c4fd99b9b2506718681bc`，两文件52增2删；基线math_dapo与主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 同字节。

[候选完整评分器](https://github.com/verl-project/verl/blob/c5e26eec5ab5941e443d1b9f6853a9e210972487/verl/utils/reward_score/math_dapo.py)只在非strict的boxed后备分支调用已有normalizer：比较两侧归一化文本，也返回归一化的pred。Python3.12.13本机加载两版完整原模块，直接调用[原测试体](https://github.com/verl-project/verl/blob/c5e26eec5ab5941e443d1b9f6853a9e210972487/tests/utils/reward_score/test_math_dapo_boxed_fallback_on_cpu.py)的七组表示、错答案与strict控制，共9项：基线2过7失败，候选9过，两次相同。strict控制仍对`1,000`与`1000`给−1、保留原pred；非strict错`1,001`也仍拒绝，不能把统一表示理解为放宽所有答案。

返回pred与只改分数不同。对gt `1000`，三条响应`\boxed{1,000}`、`\boxed{1000}`、`Answer: 1000`的pred计数，基线为`{"1,000":1,"1000":2}`，候选为`{"1000":3}`。[固定主线的calc_maj_val](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/metric_utils.py#L862)确实按pred键计票，maj@N再经过bootstrap和聚合；本机只数这三个实际评分输出的键，没有执行完整指标链或证明多数投票质量提升。

后备的触发条件是“未提取到Answer”，而不是“Answer判错”。`Answer: 4\nFinal result: \boxed{5}`对gt `5`，两版完整compute_score均返回`score=-1, acc=False, pred="4"`，不因后面boxed正确而改选它。原有外层末300字符、boxed提取末100字符的窗口也未改；这不是[#7440](https://github.com/verl-project/verl/issues/7440)的窗口问题修复，不扩展格式矩阵。

normalizer是指定文本规则，不是通用符号计算。`\boxed{\frac{1}{2}}`对gt `0.5`，两版均−1、pred仍为`\frac{1}{2}`；这是已有能力边界，不指称候选引入回归，也不为此另写数学grader。

包壳仅绕过verl父包初始化，原生产／测试源码与断言不改；无Torch、模型、Ray或设备导入。9项是直接调用测试体，不是pytest原生收集、完整奖励套件或训练验收；原作者9项／套件声明、外部评论11项均独立标注为他人报告，不计入本机结果。三份完整源码compile通过，命令与hash见[当日日志](../../diary/2026-10-25.md)。本项无需GPU交接，不应用候选、改M5专项或新增任务；上游写入及已有PR的fork head推送仍逐项待用户明确同意。

## 代码奖励的已知失败与部分分数

“不再把已知错解评为满分”与“按全部用例计算正确率”是两个目标。[#8021](https://github.com/verl-project/verl/issues/8021)已有报告者no-hup的[#8022](https://github.com/verl-project/verl/pull/8022)，2026-10-25仍open、非draft、未合并，唯一普通评论为CLA机器人、0行内评论，不另实现。固定head `a5aef7cd78978554e709ccd93e1dadfd3fcddf7c`、父提交／API base／共同祖先均为 `6093e007cc341973c9d9a6fb3867a85976c7c458`，四文件32行新增。

原`default_compute_score`把codecontests／apps／codeforces／taco路由到continuous评分。本地[prime_code](https://github.com/verl-project/verl/blob/a5aef7cd78978554e709ccd93e1dadfd3fcddf7c/verl/utils/reward_score/prime_code/__init__.py)先以timeout=5调用完整列表，但runner遇首个失败就返回；未全过时，以timeout=10逐例重测最多前10个。这两个值是调用参数，不是含进程创建的端到端硬截止。旧实现丢掉首轮第11例及以后的已知错误，可能返回1。候选保存full_res，若其末项严格为Python False且超出重测范围，向部分分数分母再计一次失败。它没有继续跑完后面的全部用例。

Python3.12.13、macOS本机fork进程，用完整原prime_code、utils与testing_util及原default路由函数，执行固定的无害程序`n=int(input()); print(2*n if n<=10 else 0)`。11个输入期望各自两倍，原版错解／正确程序均实得1；候选[原真实执行测试体](https://github.com/verl-project/verl/blob/a5aef7cd78978554e709ccd93e1dadfd3fcddf7c/tests/utils/reward_score/test_sandbox_on_cpu.py)通过，错解10/11、正确程序1。原测试体在base失败，候选通过；不是把模拟返回值接给本地runner。

单一21用例控制区分了评分范围：同一程序按构造前10个正确、后11个错误，候选本地分数仍为10/11≈0.9091，并非全量比例10/21≈0.4762。后者是根据程序逻辑计算的参考，不是runner已执行21例后的统计。第一轮已在第11例停止，返回的metadata只来自重测前10例，长度10；分数计入的第11例失败没有新增成一条返回metadata。

[sandbox_fusion](https://github.com/verl-project/verl/blob/a5aef7cd78978554e709ccd93e1dadfd3fcddf7c/verl/utils/reward_score/sandbox_fusion/__init__.py)沿其已有结果列表评分：前N（最多10）项全为True且后面存在严格False时，才把分母加1。作者[原mock测试体](https://github.com/verl-project/verl/blob/a5aef7cd78978554e709ccd93e1dadfd3fcddf7c/tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu.py)在base失败／候选通过；21项结果包`10×True+11×False`也给10/11，但保留全部21条mock metadata。这是聚合函数的结果包测试，未连接服务，不能据同分认证实际执行顺序、重试或metadata一致。

错误类别仍影响策略。仅给sandbox聚合器`10×True+[-1]`的合成结果包，本机候选返回1；此次没有运行超时、崩溃或攻击代码。修复只追记严格False，不能泛化为“任何已知失败都无法满分”。这是候选刻意保留的错误码路径，不作为另开评分补丁的理由；真实数据分布、奖励对训练的影响和外部服务行为仍需另外证据。

本机只运行上述已审阅程序。原`reliability_guard`自己声明不是安全沙箱，不能据这些用例认证任意模型代码可在主机上安全执行。默认执行沙箱首次阻止Manager本地Unix socket，随后经工具权限审查允许本地IPC重跑；最终harness无遗留子进程。包壳仅连接原模块、default函数与测试导入，远端调用被禁止／mock替代；未原生收集作者28过13跳套件、跑完整reward manager或训练，fork设置仅限独立harness进程。两版四个原模块及head两个测试模块共10份完整源码compile通过，命令与校验值见[当日日志](../../diary/2026-10-25.md)。不应用候选、拼接上节的调用格式修复或新排M5算力；任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## 数学表达式的语法与资源边界

奖励判分器应把模型答案当数据；替换通用 `eval` 后，也必须明确哪些表示仍应接受、结果是什么类型，以及资源预算在哪里生效。[#7603](https://github.com/verl-project/verl/issues/7603)的两个执行位置已有[#7607](https://github.com/verl-project/verl/pull/7607)和[#7701](https://github.com/verl-project/verl/pull/7701)覆盖，2026-10-25读取均open、非draft、未合并，不另开竞争修复。它们独立于[#8347](https://github.com/verl-project/verl/pull/8347)的区间/数值正确性保护，不能拿那一项的64个参数例认证解析安全，也不能自动叠加补丁。

固定版本：#7607 head `b7d469f63b8d369d289918fcc482e59e687d2622`、实际祖先 `ea53291385ce764019a2b40733605f21d8317583`，两文件327增4删；#7701 head `752c6ebe10ee4d6b4c370dab33c20d91e638f28e`、实际祖先 `4eff9ae30a23b59554758fa0ca93940605ae5cdf`，两文件99增4删。两PR的API目标base为3f9d4c0，并非上述实际祖先；两祖先的grader与本轮对照主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 同字节。

### 语法白名单与结果类型

[#7607的递归求值器](https://github.com/verl-project/verl/blob/b7d469f63b8d369d289918fcc482e59e687d2622/verl/utils/reward_score/prime_math/grader.py)明确选择数值常量、部分运算符和list/tuple，并拒绝bool；pi替换与矩阵解析共用它。[#7701的节点检查](https://github.com/verl-project/verl/blob/752c6ebe10ee4d6b4c370dab33c20d91e638f28e/verl/utils/reward_score/prime_math/grader.py)检查AST节点类别，再用空builtins执行编译结果；它没有检查Constant的值类型，运算符集合还来自整个operator/unaryop子类集合。因而“只允许某些节点”不等于“结果只可能是数值”。小规模原helper对照如下：

| 良性输入 | #7607 | #7701 |
|---|---|---|
| `1+2*3` | int 7 | int 7 |
| `[1/2, 3]` | list `[0.5, 3]` | ValueError |
| `True` | ValueError | bool True |
| `'ab'*3` | ValueError | str ababab |
| `1<<2` | ValueError | int 4 |

这张表只说明helper接收的语言和结果类型不同，不表示上述非数值输入能拿到完整评分器的分数。两项方案对普通 `2\pi` 都得到约6.283185307；不含pi的 `1+2*3` 在 `handle_pi` 中仍原样返回，不是所有答案都会进入算术求值。

### 矩阵字面量与矩阵内算式

#7701矩阵分支使用 `ast.literal_eval`，不是它的算术helper。该标准库函数接受字面量及容器，而不承诺执行一般算式。[Python 3.12 文档](https://docs.python.org/3.12/library/ast.html#ast.literal_eval)还明确提示，它不执行通用Python代码并不等于没有内存、栈或CPU耗尽风险，不能当不可信输入的资源沙箱。

完整原 `grader.math_equal` 的三个单元素矩阵对照中，主线与#7607都接受 `[1]` 对pmatrix中的1、`[1/2]` 对pmatrix中的0.5，并拒绝 `[1]` 对9；#7701分别为True、False、False。分数表达式的差异到达了后备判分返回值，不只是内部parser的输出差异。这里只覆盖单元素形式，不声称任意多行矩阵兼容或完整 `default_compute_score` 奖励路由等价。

### 资源限制与完整 worker 隔离

#7607新增字符数、递归节点/深度、指数和中间整数位数限制；#7701所读helper没有对应显式限额。本机以很小的 `2**1025` 控制确认前者按指数策略拒绝，后者得到1026位整数；这不是性能或耗尽实验。两者对 `1e309` 均可得到float inf，因此“数值类型”还不等于“有限数值”。

这些局部限额也不能直接称为整个worker的资源预算。#7607允许list经过算术运算，本机 `[1]*3` 得到三项列表；其位数检查只针对int，源码没有给此处容器结果的元素数量设同样上限。这里仅用三项输入核对行为，没有放大分配或构造拒绝服务载荷。两方案也仍保留后续SymPy解析、比较与原timeout路径；修改两个eval位置不等于已经建立独立进程、逐样本状态隔离或整个判分器的安全边界。

### 本机验证范围

Python3.12.13、SymPy1.14.0、pytest8.4.2，复用已有pylatexenc2.10/antlr4-runtime4.9.3 target，无新增安装。三个独立进程从固定Git对象加载完整原grader、prime_math入口、normalizer、路由及原timeout依赖，仅跳过verl/utils父包初始化。每个候选各直接调用四个原良性测试方法，共8方法通过；另做上述八种helper输入、三版各三项矩阵和两项pi控制，两次结果一致。没有pytest原生收集或复跑作者完整安全/攻击套件，也没有执行命令注入、文件载荷、状态投毒或耗尽测试；原timeout使用进程内显式fork，不认证macOS默认spawn。

[日志](../../diary/2026-10-25.md)保留原方法名、输入和hash；一次性harness留工作目录，不新增永久工具。结论是三个独立审查问题：语法是否受限、答案表示是否兼容、资源与状态是否隔离。不能以某一项通过替代另两项；本项不需GPU，不向M5新增任务，也不应用任一候选或做未经用户同意的上游动作。

## 代码判分的格式与逐例接线

**能读懂测试数据，不代表逐例评分仍在调用同一个函数。** [#8174](https://github.com/verl-project/verl/issues/8174)已有报告者prabhu-gopal的[#8175](https://github.com/verl-project/verl/pull/8175)；新[#8360](https://github.com/verl-project/verl/pull/8360)由liangqianxing实现。2026-10-24两PR均open/non-draft、未合并，属于已有覆盖工作，不另认领。#8360固定head`af4f9a3b2aca145c7944a116701dcde480f966be`、base/祖先`3c6d9b43b67c74ec343d77874cd8fa6e841523da`，2文件114增23删；#8175固定head`569c875e10d6e2bc55b96ef78cda5ae32e627e9f`、base/祖先`fc72e2f384212470868a9928b62baedb9bcc3ff0`，3文件62增3删。两base的prime_code目录同字节，比较不混入其他主线变化。

### 数据格式与函数身份是两条链

无sandbox URL时，默认apps/taco/codecontests/codeforces入口调用prime_code并启用continuous。底层run_test按fn_name是否存在区分函数调用与stdin程序；完整批次全通过可直接返回1，只有出现失败才进入逐例重测。因此“正确解满分”不会检验重测时的函数身份。

[#8175评分入口](https://github.com/verl-project/verl/blob/569c875e10d6e2bc55b96ef78cda5ae32e627e9f/verl/utils/reward_score/prime_code/__init__.py)在重建单例时保留fn_name。#8360只改[底层格式处理](https://github.com/verl-project/verl/blob/af4f9a3b2aca145c7944a116701dcde480f966be/verl/utils/reward_score/prime_code/testing_util.py)，入口与main同字节，仍丢fn_name；部分正确的add会被按stdin重测而不是函数调用，两个布局的2/3样例均得0。

反过来，#8175保留了旧的整数字典键恢复逻辑：第一个参数是字典时，可把整个参数列表改成仅含该字典的一项。#8360把这段转换限定于JSON-line路径，native列表中的`[{"1":5}, "1"]`可保留键类型和第二参数；原lookup测试在#8175得[-1]、在#8360通过。不能只用两个分支的总通过数决定全面优劣，或据此直接叠加补丁。

### 三版原测试的真实CPU对照

M1 macOS、Python3.12.13、NumPy2.5.4、pytest9.1.1。每版完整复制五个原模块（默认评分分派、import_utils、prime_code入口/utils/testing_util），保留原包名；仅verl及verl.utils两级祖先初始化替换为空壳以避免框架初始化。两份原测试的imports、fixture、参数与断言均不改，真实multiprocessing Manager/Process执行已检查的add/greet/lookup代码，不使用假评分器或AST重写方法。

| 固定代码 | #8175原6项 | #8360原2项 | 失败点 |
| --- | --- | --- | --- |
| main 3c6d9b4 | 3过3失败 | 0过2失败 | native列表与逐例接线 |
| #8175 569c875e | 6过 | 1过1失败 | 新测试最后的字典参数断言 |
| #8360 af4f9a3b | 4过2失败 | 2过 | 两布局的部分分数断言 |

首次sandbox运行因Manager Unix socket bind被拒而受阻，结果不计为代码回归；获得本地执行权限后，才得到表中可比较结果。三次session收尾均0活动子进程。五模块×三版及两测试均与固定Git bytes精确相等；命令、hash及环境边界见[当日日志](../../diary/2026-10-24.md)。这不是完整verl初始化、Ray/奖励manager、数据集抽样或训练验收；报告者30题与机器人review不计作本机结果。

### JSON可解析不代表输出类型明确

固定#8360对JSON-line输入的字符串expected尝试json.loads，解析失败才保原文。真实echo控制中，原生`hello world`无法解析JSON而保留，评分通过；原生数字形文本`3`可解析成整数，返回字符串的echo反而判错；将expected写为JSON字符串文本`"3"`才通过。三个控制只说明当前格式契约，不宣称JSON编码的数值输出本来就应按文本处理。

同样的字节`3`既能表示原生文本，也能表示JSON整数；只看输入是字符串、或expected是否可被json.loads，不足以无歧义区分二者。#8360已有机器人评审提及字符串问题，但旧评审的普通文本异常已被当前catch改变，不能照抄为当前失败。应先明确数据schema/编码约定再决定实现，不用更多类型猜测改判分。本轮不改候选、发布回复或新增GPU/M5队列；现有reliability_guard本身也明示不是安全沙箱，不能把这些受控算术用例外推为可安全执行任意模型代码。

## 数学评分池的恢复与错误信号

**错答案、验证器失败和超时不是同一种训练信号；返回普通0分会掩盖这种区别。** [#8011](https://github.com/verl-project/verl/issues/8011)报告数学评分的共享进程池在worker死亡后永久不可用，正确答案也持续得0分；ChdDongyang已有[#8015](https://github.com/verl-project/verl/pull/8015)，不另写竞争修复。2026-10-24读取issue两条评论及PR正文，PR为open/non-draft，固定head `0c4c3d40d46d5f47d1918c905641eea969b7fa09`，base/共同祖先`6093e007cc341973c9d9a6fb3867a85976c7c458`。生产模块在固定main`3c6d9b4`与该base逐字节相同；真正改动两文件238增9删。

### 共享池与两种超时

[旧模块](https://github.com/verl-project/verl/blob/6093e007cc341973c9d9a6fb3867a85976c7c458/verl/utils/reward_score/math_verify.py)缓存一个四worker、spawn上下文的ProcessPoolExecutor。`future.result(timeout=...)`只限制调用者等待，不终止正在执行的任务；旧实现既不替换坏池，又把普通异常打印后降为0分。全局共享是每个导入进程内的单例，不是所有Ray worker共用一个跨进程池。该模块是显式选用的评分入口；固定[默认分派](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/reward_score/__init__.py)仍把math_verify调用放在注释示例，不能把问题归给所有数学任务或默认训练。

[候选](https://github.com/verl-project/verl/blob/0c4c3d40d46d5f47d1918c905641eea969b7fa09/verl/utils/reward_score/math_verify.py)最多两次尝试。BrokenProcessPool第一次重置并重试，第二次向外抛出；一般异常不再被统一吞掉。外层Future超时会cancel、重置并kill旧池worker，然后返回`timeout_score`；math-verify自身的TimeoutException则只返回默认分，不清池。两种超时的结束条件不同，不能仅按同一个“timeout”名字合并处理。

重置在锁内核对池对象身份，只有仍是当前池才置空，随后快照旧池进程并kill，避免迟到调用者清掉已替换的新池。清一个共享池也会打断其他在飞任务，另一调用者可能依靠那一次重试恢复；这不是每个任务独立取消，也不是已经证明所有并发交错无竞态。每次尝试自己的等待上限加启动/重试开销，不是整个compute_score严格只有一个timeout总预算。

### 原测试接真实进程

M1 macOS、Python3.12.13、pytest9.1.1。math-verify0.9.0等五个包装入独立临时target，不改现有venv；依赖为latex2sympy2-extended1.11.0、antlr4-python3-runtime4.13.2、SymPy1.14.0、mpmath1.3.0。两版完整生产模块原bytes分别保存为同名可导入叶模块，避开verl父包及GPU发现；[原测试](https://github.com/verl-project/verl/blob/0c4c3d40d46d5f47d1918c905641eea969b7fa09/tests/utils/reward_score/reward_score/test_math_verify_on_cpu.py)只改两条import指向它，所有fixture、函数体与断言原样，由pytest收集运行。没有假ProcessPoolExecutor或假验证器。

| 原测试范围 | base结果 | head结果 |
| --- | --- | --- |
| 正确、错误、等价答案三个控制 | 3通过 | 3通过 |
| SIGKILL一个worker后正确答案恢复 | 预期失败：返回0而非1 | 通过 |
| 重试后的池再次死亡，异常传播 | 未运行 | 通过 |
| 四worker被阻塞，超时后旧池须替换 | 预期失败：仍是旧池 | 通过 |
| 一调用超时，另一健康调用重试恢复 | 未运行 | 通过 |

head七项分别跑两次，均7通过。死亡由测试向自己的真实子进程发SIGKILL，不是复现实际OOM；阻塞由原测试的1000秒sleep任务注入，不是发现了能卡死真实verify的数学输入。外部session收尾只清理该pytest进程自己的子进程；一次旧版失败后的首次快照曾显示1个活动子进程，改为有界重扫后复跑确认0，随后只读进程检查也未见spawn/resource_tracker残留。此清理不是候选生产代码的一部分，也不改变测试断言。完整版本、hash、命令与失败记录见[当日日志](../../diary/2026-10-24.md)。

### 恢复服务不等于完整故障标注

固定[RateLimitedRewardManager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/experimental/reward_loop/reward_manager/limited.py)能对传播来的异常写`reward_extra_info.error`，对它自己捕获的异步超时写timeout。但候选的两种评分超时仍返回普通float；若默认0，该上层只会按分数消费，不自动获得超时标志。故障恢复、行级可观测性和数学等价判分应分别验收，不能把“池恢复了”改述成“所有失败均可区分”。本轮没有执行完整manager、Ray/RewardLoop、训练或大规模并发，也不推断#8250根因/收敛收益。纯CPU问题无需GPU资源，不新增M5任务、不自动采用候选或对外回复。

## DisRM 返回契约与部分写入

独立DisRM的模型请求成功，不代表AgentLoop已经完整接收奖励结果。[#7368](https://github.com/verl-project/verl/issues/7368)报告生产者只返回 `reward_score`，消费者却同时要求 `reward_extra_info`；已有[候选 #7608](https://github.com/verl-project/verl/pull/7608)补上空字典，并同步文档。2026-10-26核验OPEN、非draft、未合并，两个普通评论仅作者请求启动CI，reviews／行内评论为空；旧[#7389](https://github.com/verl-project/verl/pull/7389)为CLOSED且mergedAt=null，不能当已合入修复。不另写同一补丁或替作者催审。

固定head `e1e753776781129ae39de77a91fd2ee06ac74508`，父提交／与当前main共同祖先 `ea53291385ce764019a2b40733605f21d8317583`，相对main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` ahead1／behind203；三文件112增2删。API目标base `3f9d4c0d` 不替代实际祖先。编号与关键词检索已见现有覆盖，未将空assignee或历史候选关闭解释成可重新认领。

### 返回键与消费路径

[原生产者](https://github.com/verl-project/verl/blob/e1e753776781129ae39de77a91fd2ee06ac74508/verl/experimental/reward_loop/reward_loop.py)的vLLM路径取 `data[-1].probs[-1]` 并请求classify，SGLang路径取 `data[-1].embedding[-1]` 并请求v1/embeddings；候选不改分数计算或HTTP格式，只让两路都返回 `{"reward_score": score, "reward_extra_info": {}}`。原两个参数用例分别使用固定响应0.875／0.75，不是奖励模型数值或质量验证。

[AgentLoop消费者](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/experimental/agent_loop/agent_loop.py)先给末条output的reward_score赋值，再严格索引附加字段；三版整个_compute_score方法AST相同。候选修的是生产者契约，没有把消费者改成容忍所有缺字段。相邻的共置[批处理路径](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/experimental/reward_loop/reward_loop.py)使用 `output.get("reward_extra_info", {})`，因此不能拿它对缺键的容忍证明独立路径也安全；这也不认证不同样本内部附加key的对齐。

### 原回归与同一输出再次调用

[复查脚本](../../experiments/verl-disrm-contract/verify.py)执行候选[两种生产者参数例及一个消费者回归](https://github.com/verl-project/verl/blob/e1e753776781129ae39de77a91fd2ee06ac74508/tests/experimental/reward_loop/test_disrm_reward_contract_on_cpu.py)。原测试体、参数表、装饰器及断言不变，按固定blob AST装配完整生产／消费方法、原DataProto与Pydantic类型，类壳绕过框架初始化。候选三项通过；祖先／main各为两项返回字典断言失败、一项精确 `KeyError("reward_extra_info")`。预处理、HTTP和远程句柄沿用原AsyncMock／本地代理，因此原测试docstring的“end-to-end”只对应这条隔离方法链，不是服务或训练贯通。

另一个固定输入控制保留输出对象：祖先／main首次抛缺键异常后，reward_score已是0.875，extra_fields仍为空。再次对**同一个Python输出对象**调用原方法，因为reward_score不再为None而跳过评分，HTTP替身累计仍只调用一次，缺失附加字段也没补齐。候选首次完成两字段写入，第二次正常使用已评分状态。这仅说明该局部对象的失败不是原子操作；不证明框架实际会重用失败对象、自动重试或静默接纳坏训练批，不能把局部再次返回无异常当成恢复成功。

[结果](../../experiments/verl-disrm-contract/result.json)保存两次调用前后状态、原测试结果及源码SHA256。Python3.12.13、torch2.14.1、NumPy2.5.4、pytest9.1.1、macOS arm64；两次执行结果一致。三版四个完整模块及原测试AST／compile通过，实际运行只有所述方法／容器与替身；没有导入原verl包、运行真实tokenizer／模型／HTTP／Ray／TransferQueue或完整训练。

本项不应用候选、不降低消费者的缺键检查、不扩故障／后端矩阵或改恢复策略。CPU契约无新增GPU缺口，不另排M5队列或改其专项；任何上游动作仍待用户逐项明确同意。

```bash
/path/to/python experiments/verl-disrm-contract/verify.py /path/to/verl
```

## 异步奖励的验证标志与元数据范围

异步奖励的新建输入不会自动继承 rollout 批的元数据。[#7121](https://github.com/verl-project/verl/issues/7121)报告自定义奖励无法从 `data.meta_info` 区分训练与验证；[已有 #7217](https://github.com/verl-project/verl/pull/7217)把现成的 `validate` 显式传到评分方法，再写入 `meta_info={"validate": validate}`。这恢复的是一个标志，不是整批元数据透传。2026-10-25 核验候选 OPEN、非 draft；四条普通评论是机器人声明／CLA与作者催审，唯一review为COMMENTED而非APPROVED，不能当维护者验收或合并。

固定 head `e8508932b0a7ae99b97ec8c3512fddbadd35d9d0` 是合并提交：首父 `7fcdd5e1` 已含修复，第二父及与当前 main 的共同祖先为 `2781c1e405fa80871a838bfdf9495b412df19c35`。相对 main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` ahead2／behind298；真正候选差异为三文件67增3删，不能用首父差异中的主线合并量描述修复。编号及关键词查重已有覆盖，不另实现或发反馈。

### 批级标志与逐行字段分开传递

[普通 AgentLoopWorker](https://github.com/verl-project/verl/blob/e8508932b0a7ae99b97ec8c3512fddbadd35d9d0/verl/experimental/agent_loop/agent_loop.py)在 postprocess 中将一个 output 包成列表评分；[TQ postprocess](https://github.com/verl-project/verl/blob/e8508932b0a7ae99b97ec8c3512fddbadd35d9d0/verl/trainer/ppo/v1/agent_loop_tq.py)可传整个 outputs 列表。候选两处都以关键字传 `validate`，评分方法把它设为必需参数，而不是默认 False。缺接线会显式报错，不会静默把验证算成训练；本机对两文件AST确认这两处调用，未执行完整 postprocess／TQ。

评分方法只有在末条 output 没有预计算 reward、且存在 reward worker 句柄时才新建 DataProto 并调用奖励；结果仍写回末条 output。它将 `kwargs` 复制到 `non_tensor_batch` 的逐行数组，另构造仅含 validate 的 `meta_info`，两者并不互为备份。[奖励 worker](https://github.com/verl-project/verl/blob/e8508932b0a7ae99b97ec8c3512fddbadd35d9d0/verl/experimental/reward_loop/reward_loop.py)再把收到的 DataProto 交给 `run_single`；自定义消费者必须知道字段在哪一层。候选不会恢复任意父批字段或自动补上 `reward_extra_keys`，也不修复[评分张量的补齐几何](#异步奖励的补齐边界)。

### 原回归测试与本机边界

[复查脚本](../../experiments/verl-reward-validation/verify.py)按固定 blob 提取完整原 DataProto／DataProtoItem、Pydantic 输出类型、四个原 worker 方法、位置ID函数与计时 helper；worker仅用本地类壳装配这四方法以绕过 Ray 初始化。原测试的两种 validate 参数、装饰器、断言和假远程奖励句柄均保留，直接 await 原测试体，候选两项通过；不是原模块的完整 pytest 或作者25+1项套件。

同一原测试体接祖先／main 时，两参数例都在调用处因 `validate` 非法关键字而失败，不能把这四项说成已到达远程后的断言失败。另一个独立边界对照按各版真实签名调用，祖先／main 的接收端 meta_info 为空，候选为 `{"validate": true}`；三个版本都把同一 `batch_marker` 留在 non_tensor_batch，并得到原假句柄的1.0分。这个1.0是恒定测试返回值，不是实际奖励函数算出的分数。

[JSON结果](../../experiments/verl-reward-validation/result.json)保存签名失败／通过、真实容器内容、调用关键字与源码SHA256。Python3.12.13、torch2.14.1、NumPy2.5.4、TensorDict0.15.2、macOS arm64；使用真实CPU张量／容器，不导入原 verl 包或初始化 Ray／TQ／模型。测试只走 processor=None 的纯文本支路，完整方法中的其他分支不计为已覆盖；奖励服务、自定义manager、模型及训练均未运行。这个元数据问题无需新增 M5 算力队列，现有专项与上游只读边界不变。

```bash
/path/to/python experiments/verl-reward-validation/verify.py /path/to/verl
```

## 异步奖励的补齐边界

**mask 的有效 token 总数正确，不代表消费者从尾部截取时能找到正确的 response。** [#8101](https://github.com/verl-project/verl/pull/8101) 修复多输出 trajectory 的异步评分输入，2026-10-24 读取 OPEN、API draft=false，作者 zupengwang 已实现；正文仍保留待人工亲自复跑的说明，状态字段不认证测试或就绪程度。本机固定 head `a58b1d2b9bbc465d89910de583f8a30aa7dc917e`，base/祖先 `8718ca30a3f002f93b7c4fd99b9b2506718681bc`，ahead1/behind0；3文件229增4删，不另开竞争修复。

### 生产与消费必须共用同一条边界

旧 [_compute_score](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/experimental/agent_loop/agent_loop.py)先把每条 `prompt + response` 紧凑拼接，再分别对 prompts、responses、input_ids 和 attention_mask 做右侧 pad_sequence。这些张量各自取不同的最大宽度：`max(P+R)` 不一定等于 `max(P)+max(R)`，即使偶然相等，也不保证 response 起点一致。

[DAPO run_single](https://github.com/verl-project/verl/blob/a58b1d2b9bbc465d89910de583f8a30aa7dc917e/verl/experimental/reward_loop/reward_manager/dapo.py)先取最后一条输出，以 responses 的列宽 R 截取 attention_mask 最后 R 列，求和后再从 response_ids 开头取这么多 token。mask 错位会把 prompt 计入 response，或把答案挤到 tail 窗口之前；padding token 是否被 tokenizer 当特殊符号忽略，不能修正错误的长度惩罚。

[候选构造](https://github.com/verl-project/verl/blob/a58b1d2b9bbc465d89910de583f8a30aa7dc917e/verl/experimental/agent_loop/agent_loop.py)先用原 `_pad_token_ids` 将每条 prompt 左补到 max(P)、response 右补到 max(R)，再拼 IDs 与对应的两个 attention mask。所有输出共享 prompt/response 分界，之后 pad_sequence 不再改变该边界。`prompt_len`/`response_len` 元数据仍来自原列表长度；不能靠保存了正确长度字段，就认为忽略它们的消费者也会自动正确。

### CPU 几何对照

[复查脚本](../../experiments/scripts/verify_verl_reward_padding.py)使用真实本地 WordLevel/PreTrainedTokenizerFast，无模型或词表下载；token0是普通 `LEAK`，pad=3。它提取两版原 `_pad_token_ids`、张量构造和四个 pad_sequence 表达式，另执行 DAPO 原四个长度/切片赋值及长度惩罚分支。字段容器只供 self/output/data_item 属性，没有完整 callback、DataProto、TQ、Ray 或 reward manager 实例。

| trajectory 的 (P,R) | base 末条推得长度 | head 长度 | input_ids 宽度 base→head | DAPO 长度惩罚 base→head |
| --- | --- | --- | --- | --- |
| (3,5), (9,2) | 5，解码含3个 LEAK | 2 | 11→14 | -0.5→0 |
| (10,2), (1,6) | 1，只留下首个答案 token | 6 | 12→16 | 0→-0.75 |
| (3,2), (3,2) | 2 | 2 | 5→5 | 0→0 |

惩罚配置固定 max_resp_len=7、buffer_len=4、factor=1：预期免罚长度为3。第二行的 -0.75 是长度恢复后按配置得到的惩罚，不是修复使所有 reward 更高。表中只改变几何与读取，不证明任务答案质量、RL 收敛或真实性能。

Python3.12.13、torch2.14.1、Transformers5.20.2、tokenizers0.23.3，三组两版本控制运行两次一致，head 另断言 `input_ids == cat(prompts,responses)`。本机没有执行位置ID、多模态处理、Naive/DisRM完整路径、precomputed分支、评分器异常或原作者12项pytest；两个完整 agent_loop.py 与 head dapo.py 共3个模块版本仅 compile。原作者7失败/5通过→12通过、相邻53通过及13项fixture错误均不是本机结果。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B experiments/scripts/verify_verl_reward_padding.py /path/to/verl-git
```

### attention mask 不等于训练 response mask

候选对 response_ids 的 attention mask 仍把观测内容视为有效 token；不会按训练用 `response_mask` 删除工具观察。真实 token0也不能当 padding 丢弃。这里解决的是评分输入边界，不改变作者约定的“评分看完整 response、默认 manager 取末条输出”语义；本机控制只覆盖表中纯文本数据。

代价是拼接宽度可从 max(P+R) 增到 max(P)+max(R)，额外临时补齐不能被称作性能收益或内存硬上限。缺少完整异步框架与真实后端验收仍留为未测；本题没有新增 GPU/NPU 问题，不另排 M5 硬件队列、切环境或改 AReaL/#8313/#523/DeepInfer 专项。

## 1. 系统架构概览: 两层奖励系统 ★★

verl的奖励系统经历了重大架构演进,目前存在**两套并行系统**:

```
┌──────────────────────────────────────────────────────────┐
│                  Ray PPO Trainer (ray_trainer.py)         │
│                                                          │
│  1. 生成rollout → gen_batch_output                        │
│  2. 计算reward → _compute_reward_colocate OR extract_reward│
│  3. KL-in-reward → apply_kl_penalty (可选)                │
│  4. 计算advantage → compute_advantage (GRPO/GRPO_VEC/...) │
│  5. 更新actor → _update_actor                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Legacy Layer (verl/workers/reward_manager/)              │
│  ├── AbstractRewardManager (__call__ interface)           │
│  ├── NaiveRewardManager   (loop per-item)                │
│  ├── BatchRewardManager   (batch scoring)                │
│  ├── DAPORewardManager    (overlong penalty)              │
│  ├── PrimeRewardManager   (async ProcessPoolExecutor)    │
│  └ Registry: @register decorator                         │
│                                                          │
│  ★ Experimental Layer (verl/experimental/reward_loop/)    │
│  ├── RewardLoopManager → distributed worker orchestration │
│  ├── RewardLoopWorker   → async compute_score (3 paths)  │
│  ├── RewardManagerBase  → async run_single interface     │
│  ├── RewardModelManager → RM server lifecycle(sleep/wake) │
│  ├── NaiveRewardManager  (async, event loop executor)    │
│  ├── RemoteRewardManager (Ray actor, separate process)   │
│  ├── RateLimitedRewardManager (3-layer rate limiting)    │
│  ├── GDPORewardManager   (per-dimension reward)          │
│  └ Registry: @register decorator (separate from legacy)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**★ 关键区别**:
- Legacy: `__call__(data) → reward_tensor` (同步, 直接返回tensor)
- Experimental: `async run_single(data) → {"reward_score", "reward_extra_info"}` (异步, 返回dict)
- Experimental是**默认路径**: ray_trainer.py 通过 `load_reward_manager()` 加载experimental RewardManager
- 两套registry互不干扰: legacy用`REWARD_MANAGER_REGISTRY`, experimental用`REWARD_MANAGER`
- ★ RateLimitedRewardManager **同时注册到两套registry** (`@register_manager("rate_limited")` + `@register_manager_legacy("rate_limited")`)

## 2. 奖励计算3条路径 ★★★

### Path 1: Rule-Based Reward (custom_reward_function) ★★★

**配置**: `reward.custom_reward_function.path` + `reward.custom_reward_function.name`

**加载流程** (reward.py:50-86):
```python
def get_custom_reward_fn(config):
    module_path = config.reward.custom_reward_function.path  # e.g., "my_reward.py"
    fn_name = config.reward.custom_reward_function.name      # e.g., "compute_score"

    # ★ 动态导入: load_extern_object → importlib动态加载外部文件
    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    # ★ 合入reward_kwargs: partial(_call_with_kwargs, raw_fn, reward_kwargs)
    # → 调用时自动把reward_kwargs合并进函数参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)  # sync → wrapped
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)  # async → wrapped async
```

**函数签名** (文档: docs/preparation/reward_function.rst):
```python
# Sync版本
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs):
    # data_source: 数据集来源标识 → 决定用什么评分逻辑
    # solution_str: 模型生成的文本 (decoded from token IDs)
    # ground_truth: 正确答案 (从parquet文件的reward_model.ground_truth字段)
    # extra_info: 额外信息 (num_turns, rollout_reward_scores等)
    # reward_kwargs: 从config合并的额外参数

    return 1.0  # 返回float → reward = score
    # 或返回dict → {"score": 1.0, "format_reward": 0.5, ...} → GDPO需要!

# Async版本 ★ (推荐用于需要API调用的场景)
async def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                        reward_router_address=None, reward_model_tokenizer=None):
    # ★ reward_router_address: RM HTTP路由地址 → GenRM时使用
    # ★ reward_model_tokenizer: RM的tokenizer → GenRM时使用

    async with aiohttp.ClientSession() as session:
        result = await session.post(reward_router_address + "/v1/chat/completions", ...)
    return {"score": parsed_score}
```

**★★★ 自定义reward函数=RTX 4090唯一可行路径**: 纯CPU执行 → 无GPU开销 → 与rollout/actor共存

### Path 2: Discriminative Reward Model (DisRM)

**配置**: `reward.reward_model.enable=True` + `reward.reward_model.model_path` + 无custom_reward_function

**流程** (reward_loop.py:231-270):
```python
async def compute_score_disrm(self, data):
    # Step 1: 预处理 → tokenizer.apply_chat_template → 构造RM输入
    disrm_prompt = await self._preprocess_reward_inputs(data)
    # → 把prompt+response组成chat格式 → RM判断质量

    # Step 2: HTTP请求 → RM router → RM server推理
    engine_name = config.reward.reward_model.rollout.name
    if engine_name == "vllm":
        payloads = {"model": model_name, "input": disrm_prompt, "use_activation": False}
        output = await self._post_request(payloads, "classify")
        rm_score = output["data"][-1]["probs"][-1]  # ★ 取最后token的prob → 分类概率
    elif engine_name == "sglang":
        payloads = {"model": model_name, "input": disrm_prompt}
        output = await self._post_request(payloads, "v1/embeddings")
        rm_score = output["data"][-1]["embedding"][-1]  # ★ 取embedding最后一维

    return {"reward_score": rm_score}  # 单scalar
```

**DisRM本质**: classifier → 给prompt+response打分 → 输出概率/embedding → 取最后一个维度作为score
**★ 需要额外GPU**: RM模型需要独立GPU运行(vLLM/SGLang server) → RTX 4090不可行

### Path 3: Generative Reward Model (GenRM)

**配置**: `reward.reward_model.enable=True` + `reward.custom_reward_function.path` (必须!)

**为什么GenRM必须指定custom_reward_function** (reward_loop.py:146-155):
```python
async def compute_score(self, data):
    if config.reward.custom_reward_function.path is not None:
        # ★ 直接用user-customized reward function
        return await self.reward_manager.run_single(data)
    else:
        if config.reward.reward_model.enable:
            # ★ we assume the rm is disrm
            # ★ genrm must set custom_reward_function → 否则走DisRM路径!
            return await self.compute_score_disrm(data[-1:])
        else:
            return await self.reward_manager.run_single(data)  # default rule-based
```

**GenRM流程**: custom_reward_function → 构造GenRM prompt → HTTP请求到router → 生成文本 → parse score
**★ GenRM更灵活**: 可以做LLM-as-judge → 但需要自定义prompt模板和parser → 用户完全掌控

### 3条路径决策树 ★★

```
custom_reward_function.path != None?
  └─ Yes → Path 1: reward_manager.run_single(data) → user自定义逻辑
  │        (可能内部调用RM router → GenRM场景)
  └─ No → reward.reward_model.enable?
           └─ Yes → Path 2: compute_score_disrm(data[-1:]) → DisRM classifier
           │        (只取最后一个response序列 → multi-turn时只评最终)
           └─ No → Path 3: reward_manager.run_single(data) → default_compute_score
                    (rule-based → 按data_source分派 → gsm8k/math/code等)
```

## 3. Outcome Reward: 只有最后token非零 ★★★

**★★★ 这是verl GRPO最核心的设计**: 所有RewardManager都把reward放在最后一个valid token位置!

### Legacy RewardManager (naive.py:54-100):

```python
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
# ★ 所有位置初始化为0!

for i in range(len(data)):
    score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., ...)
    # ★★★ 只在最后一个valid token赋值!
    reward_tensor[i, valid_response_length - 1] = reward  # 不是所有token, 只有最后一个!
```

### Experimental RewardManager (base.py:62-82):

```python
@classmethod
def assemble_rm_scores(cls, data, scores):
    prompt_length = data.batch["prompts"].size(1)
    valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
    rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    # ★★★ 同样只在最后位置赋值!
    rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = scores
    return rm_scores
```

### 所有RewardManager一致的outcome pattern:

| Manager | 代码位置 | 赋值行 |
|---------|---------|--------|
| NaiveRewardManager (legacy) | naive.py:100 | `reward_tensor[i, valid_response_length - 1] = reward` |
| BatchRewardManager | batch.py:110 | `reward_tensor[i, length - 1] = reward` |
| DAPORewardManager | dapo.py:132 | `reward_tensor[i, valid_response_length - 1] = reward` |
| PrimeRewardManager | prime.py:177 | `reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]` |
| assemble_rm_scores (base) | base.py:79-80 | `rm_scores[..., valid_response_length - 1] = scores` |
| RateLimitedRewardManager | limited.py:527 | `reward_tensor[i, valid_response_length - 1] = reward` |

**★★★ 统一结论**: reward_tensor形状=[bs, response_length], 只有`reward_tensor[i, last_valid-1]`非零!

### 如何流入GRPO advantage ★★★

```python
# core_algos.py:304 → GRPO advantage第一步
scores = token_level_rewards.sum(dim=-1)
# ★ 因为只有最后token非零 → sum(dim=-1) = 最后token的值 = scalar reward!

# 然后group-relative:
id2score[index[i]].append(scores[i])  # 按prompt分组
a_i = (r_i - μ_g) / (σ_g + ε)        # group内标准化
advantages = scalars.unsqueeze(-1) * response_mask  # broadcast到所有token
```

**Outcome reward与GRPO的完美对齐**:
- Outcome reward = 单scalar → sum(dim=-1) = scalar → GRPO直接用
- GRPO不需要per-token reward → 只需要group内的相对排名
- ★★ 如果用process reward (per-token nonzero), GRPO仍然会sum → 语义变了!
- ★★ 但verl的GRPO advantage是outcome-only → 只适合outcome reward!

## 4. GRPO Advantage: 奖励塑造与归一化 ★★★

### 4.1 GRPO (loop-based) — core_algos.py:268-331

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # outcome → scalar per response

    # ★ 按index(prompt ID)分组 → rollout_n=8时, 同prompt8个response形成1个group
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # ★★★ Group normalization: μ_g和σ_g
    for idx in id2score:
        if len(id2score[idx]) == 1:  # ★ 单样本group → μ=0, σ=1
            id2mean[idx] = 0.0
            id2std[idx] = 1.0
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    # ★★★ 核心公式:
    # norm_adv_by_std_in_grpo=True (原始GRPO):
    #   a_i = (r_i - μ_g) / (σ_g + ε)  → 标准化 → 约零均值, 方差约1
    # norm_adv_by_std_in_grpo=False (★ Dr.GRPO, arxiv 2503.20783):
    #   a_i = r_i - μ_g  → ★ 不除std → 防止小groupσ=0时梯度消失!

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    # ★ broadcast: 每个token获得相同advantage → 乘response_mask
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores  # advantage = returns (outcome-only, no value function)
```

**★★★ 关键设计决策**:
- 单样本group(rollout_n=1): μ=0, σ=1 → advantage=score本身 → **无group normalization** → 等效REINFORCE
- Dr.GRPO(False): 不除std → 当group内σ很小(所有response分数接近)时 → 除σ→advantage爆炸 → Dr.GRPO避免此问题
- advantage=returns: outcome-only → 不需要value function → GRPO跳过critic

### 4.2 GRPO_VECTORIZED — core_algos.py:335-358

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
    # ★ 同公式但vectorized → 不需要Python loop → 快10-100x!
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)

    if norm_adv_by_std_in_grpo:
        scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
    else:
        scalars = scores - mean_g[g]

    advantages = scalars.unsqueeze(-1) * response_mask
    return advantages, advantages
```

**★★ 推荐使用GRPO_VECTORIZED**: 大batch时快10-100x, 数学结果完全一致!

### 4.3 GDPO: Per-Dimension Reward Normalization ★★

```python
@register_adv_est(AdvantageEstimator.GDPO)
def compute_gdpo_outcome_advantage(...):
    # ★★★ GDPO核心思想: 不是先sum再normalize, 而是per-dimension normalize再weighted sum!

    # Step 1: 每个reward dimension独立做GRPO normalization
    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],  # 第i个reward维度
            ...
        )
        new_advantage += weights[i] * normalized_score  # weighted aggregation

    # Step 2: batch-level whiten → 全局标准化
    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask
```

**★★★ GDPO vs GRPO**: GRPO = sum → normalize → GDPO = normalize → weighted_sum → whiten
- GDPO防止dominant reward信号淹没弱信号 → 每个维度独立标准化
- 需要custom_reward_function返回dict: `{"score": 1.0, "format_reward": 0.5, "accuracy_reward": 0.8}`
- config.algorithm.gdpo_reward_keys = ["format_reward", "accuracy_reward"]
- config.algorithm.gdpo_reward_weights = [0.3, 0.7] → 可选权重
- ★ RTX 4090可行: rule-based返回dict → 纯CPU → 不需要RM → GDPO+rule=RTX4090最优?

### 4.4 GRPO_PASSK: 只有最佳response非零 ★

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(...):
    # ★★★ 只有最好的response获得非零advantage!
    # a_best = (r_max - r_second_max) / (σ + ε)
    # 其他response → advantage = 0 → 不参与训练!
    for idx in id2scores:
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
```

**适用**: pass@k evaluation → 只训练最优解 → 与rLLM pass@k eval对齐

### 4.5 KL-in-Reward vs KL-in-Loss ★★

```
GRPO默认: use_kl_in_reward=False + use_kl_loss=True → KL只在loss端

KL-in-reward (apply_kl_penalty, ray_trainer.py:76-115):
  token_level_rewards = token_level_scores - beta * kld
  → reward端减KL → 改变送给advantage估计器的奖励

KL-in-loss (默认):
  total_loss = pg_loss + kl_loss_coef * KL(pi || pi_ref) - entropy_coef * H
  → loss端加KL → 单独的可微正则项，不直接改写reward
  → kl_loss_coef=0.001为这里的历史配置示例，不是实际梯度大小保证
```

2026-10-22 更正：仅凭reward/loss的位置不能判定约束更强或更温和；还取决于系数、KL模式、采样/归约与调用图。KL-in-loss也不自动省去ref开销：固定版本的PPO loss显式读取ref_log_prob，它必须由具体路径提供，不能把bypass名称当作零forward成本保证。[KL前向与反向核对](verl-grpo-core-algos-reading.md#kl-前向值与反向规则)中k3+显示值虽受clamp，局部梯度仍走k2；日志值或小系数均不能单独认证实际梯度强弱。本轮未运行完整ref/model路径或比较两种配置的训练收益，旧“无额外ref forward/零开销”推荐撤回。

## 5. RewardManager 详细分析 ★★

### 5.1 NaiveRewardManager (legacy) — naive.py:27-122

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __call__(self, data, return_dict=False):
        # ★ 先检查rm_scores → 如果已存在直接返回 → 不重复计算!
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):  # ★ per-item loop → 慢但灵活
            data_item = data[i]
            # decode: tokenizer.decode(valid_response_ids)
            # 提取: ground_truth, data_source, extra_info, reward_scores
            score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
            # dict → score["score"] + reward_extra_info
            # float → score = result
            reward_tensor[i, valid_response_length - 1] = reward  # ★ outcome!
```

### 5.2 BatchRewardManager — batch.py:26-128

```python
@register("batch")
class BatchRewardManager(AbstractRewardManager):
    def verify(self, data):
        # ★ 批量decode → 批量scoring → 不是per-item!
        responses_str = [tokenizer.decode(response_ids[i][:valid_len]) for i in range(len(data))]
        scores = self.compute_score(
            data_sources=data_sources,     # ★ 列表!
            solution_strs=responses_str,    # ★ 列表! (不是solution_str)
            ground_truths=ground_truths,    # ★ 列表! (不是ground_truth)
            extra_infos=extras,             # ★ 列表!
            **self.reward_kwargs,
        )
        # ★ compute_score接收batch输入 → 可以做batch级处理 → 更高效
```

**★★★ Batch vs Naive关键差异**:
- Naive: `compute_score(data_source, solution_str, ground_truth, extra_info)` → 单item
- Batch: `compute_score(data_sources, solution_strs, ground_truths, extra_infos)` → 批量
- Batch的compute_score可以内部做batch优化 → 适合外部API调用场景

### 5.3 DAPORewardManager — dapo.py:26-154

```python
@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    # ★★★ overlong penalty: 超长response惩罚!
    def __init__(self, ..., overlong_buffer_cfg=None, max_resp_len=None):
        # overlong_buffer_cfg.enable → 开启惩罚
        # overlong_buffer_cfg.len → buffer长度
        # overlong_buffer_cfg.penalty_factor → 惩罚系数

    def __call__(self, data, ...):
        reward = score
        if self.overlong_buffer_cfg.enable:
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)
            # ★★★ reward += overlong_reward → 只减不增!
            reward += overlong_reward
```

**★★ DAPO overlong penalty**: 防止模型生成过长response → 惩罚线性增长 → reward=score+overlong_penalty

### 5.4 PrimeRewardManager

#### Prime 进程池的结果协议与失败范围

固定 [#8217](https://github.com/verl-project/verl/pull/8217) head `fe9ff514507e6f7a68cf927b2213ad8d8d2501ff`、目标 base 与共同祖先 `2035f58dcb06d8cad4caee107b30f4045d724367`（ahead 1 / behind 0，2 文件新增 67 行）。[#8216](https://github.com/verl-project/verl/issues/8216) 的字典返回问题由报告者自己修复；同题 [#8262](https://github.com/verl-project/verl/pull/8262) 已关闭且未合并，不另立竞争候选。这不是 `prime_math` 字符串判分问题，而是评分函数返回后如何转成数值的问题。

调用链为 `verify → run_reward_scoring → parallel_compute_score_async → single_compute_score`。最内层将同步 scorer 送入真实 `ProcessPoolExecutor`，以 `asyncio.wait_for` 等待；普通评分异常与等待超时在这里捕获并返回 `None`。`gather` 保持输入顺序，回收后把 `None` 转成该行的 `0.0`，所以一条 scorer 抛错不必抹掉其他行。默认 helper 和 `verify` 使用 `max_workers=64`，是进程数上限，不是每批必定启动 64 个进程，更不能由此判断比其他 manager 更快。

结果解析发生在上述 worker 异常处理之外。旧版接受数值或取 `result[0]`，字典因此抛 `KeyError: 0`；#8217 增加 `isinstance(result, dict)` 后取 `float(result["score"])`，保留原数值和 tuple 路径。该修复只提取标量，`acc`、`pred` 等元数据不在这里传播；`verify` 写入的 `data.batch["acc"]` 是这些标量分数，不是字典的 `acc` 字段。

两层失败范围必须区分：scorer 抛错按行归零；结果解析抛错则逃出 helper，被 `verify` 的广泛异常回退变成整批零。候选中缺少 `score` 的字典或不可转 float 的值仍会逃出解析，`NaN` 可原样成为分数；这是本次控制输入的范围观察，不把违反结果契约的输入直接宣称为新的上游缺陷。整批回退在 [固定版本 manager 源码](https://github.com/verl-project/verl/blob/fe9ff514507e6f7a68cf927b2213ad8d8d2501ff/verl/workers/reward_manager/prime.py) 中核对，本机未构造 DataProto/tokenizer 执行 `verify`。

300 秒是 `single_compute_score` 对提交 future 的等待预算，含在池中排队的等待，不是整个批次含清理的硬截止。清理位于 `gather` 退出后的 `finally`：遍历池的私有 `_processes`，尝试 `terminate`、逐个等待最多 5 秒，再尝试 `kill`，清理异常会忽略；随后仍退出 executor 上下文。不能把它描述成“某行超时立即杀掉对应进程”或“所有情形都在 300 秒内返回”。本轮只观察正常短任务的真实子进程清理，没有运行 300 秒超时、死循环或破损池测试。

[CPU 复查脚本](../../experiments/scripts/verify_verl_prime_pool.py)在 Python 3.12.13、psutil 7.2.2、pytest 8.4.2 的独立临时环境，用 macOS 原生 `spawn` 和真实进程池，两次运行结果一致：原 4 个测试体在 base 为 3 过、字典例预期 `KeyError: 0`，head 全过；base 的混合成功/失败行控制通过，head 的 5 组控制分别核对成功行保留、字典混合顺序、缺键、坏值和非有限值。三个原评分函数及原 `get_event_loop` 完整 AST 原样执行，原测试模块完整载入；绕过包初始化、Ray/torch 和 manager 类，不替换 executor、psutil、时钟或 scorer 超时实现。直接调用原测试体，不是原生 pytest 收集或作者声称的全目录 22 项。无需 GPU 或新增 M5 实验。

```bash
uv venv --python /path/to/python3.12 /tmp/prime-pool-check
uv pip install --python /tmp/prime-pool-check/bin/python psutil==7.2.2 pytest==8.4.2
/tmp/prime-pool-check/bin/python -B experiments/scripts/verify_verl_prime_pool.py /path/to/verl
```

两版 `prime.py` SHA-256 分别为 `e30e34ca239ddb629b009eb30eee017c55ca4fdf37f9f797da1d4611239371a4`、`bf33317ac9f65636fd90ddd9d1dd4e32618dc3c3c686d4d1d5592a976a020399`；`ray_utils.py` 相同为 `1f70a9ac05b4a6f92a3ad2486bee17ac9721a9568e7e72b17e76172a60a7751b`。完整两版模块与原测试另做语法编译；不由局部成功推出训练奖励、进程故障恢复或性能结论。

#### 新编号与已有修复

**新PR编号不代表出现新的独立修复。** 2026-10-24核对[#8358](https://github.com/verl-project/verl/pull/8358)，liangqianxing、head`052e2fe87beef900a15fa5979142a09627bcece4`、base`3c6d9b43b67c74ec343d77874cd8fa6e841523da`，2文件114增0删。它和报告者Virtuoso461的[#8217](https://github.com/verl-project/verl/pull/8217)均OPEN、未合并，都明确修同一个[#8216](https://github.com/verl-project/verl/issues/8216)；不能把前者视为无人认领的新机会，也不能把两个开放候选写成主线已修复。

直接比较完整`prime.py`，#8358相对#8217仅新增helper的docstring、删除字典分支旁两行注释。两模块AST去除docstring后完全相同；旧base2035f58与新base3c6d9b4的该文件也逐字节相同。这是静态计算语句对照，不是跨依赖版本的运行验收或整个PR等价证明，docstring和测试本身确实不同。

#8358的两个新测试函数分别检查dict/tuple/float的helper结果，以及manager将字典score写到最后有效response token、其他token为0并写入`data.batch['acc']`。manager例使用原测试的固定字符串tokenizer替身和恒定score=1，不执行真实math_dapo/rlla判分；源码中acc仍来自标量scores，不因返回字典含acc就单独传播该字段。这个测试也没有覆盖按行异常、解析整批回退或超时清理，不能用“新增manager测试”替代这些具体边界。

本机只读取两个完整生产模块和新测试、完成AST/字节对照及三模块compile；没有运行新测试、构造DataProto/manager/tokenizer、重跑旧进程池控制或训练。上节旧#8217的真实spawn结果保留其原版本和范围，不升格为#8358的manager验收。已有作者覆盖，本轮不另写补丁、发布重复提醒或将机器人合并建议当查重结论；[来源、hash与复核代码](../../diary/2026-10-24.md)留日志，无GPU需求或新M5任务。

### 5.5 Experimental NaiveRewardManager — reward_loop/naive.py:24-99

```python
@register("naive")  # experimental registry
class NaiveRewardManager(RewardManagerBase):  # ★ 不同的基类!
    async def run_single(self, data):  # ★ async interface!
        data = data[-1:]  # ★★★ multi-sequence时只取最后一个sequence!
        # → multi-turn agent loop → 只有最终response参与reward

        # ★★ 异步解码: run_in_executor → 不阻塞event loop
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # ★★ 异步评分: sync → run_in_executor; async → await directly
        if self.is_async_reward_score:
            result = await self.compute_score(...)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(...))

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
```

### 5.6 RemoteRewardManager — reward_loop/remote.py:40-131

```python
@register("remote")
class RemoteRewardManager(RewardManagerBase):
    def __init__(self, ...):
        # ★★★ Ray actor → 独立进程 → 避免thread pool问题!
        # https://github.com/verl-project/verl/issues/3407
        # Math-Verify等CPU密集型任务 → thread pool有fork安全问题
        self.reward_worker = [
            RewardComputeWorker.options(
                NodeAffinitySchedulingStrategy(node_id=..., soft=True)
            ).remote(self.compute_score)
            for _ in range(num_reward_workers)
        ]
        self.reward_worker_pool = itertools.cycle(self.reward_worker)  # ★ round-robin

    async def run_single(self, data):
        reward_worker = self.choose_reward_worker()
        result = await reward_worker.compute_score.remote(...)  # ★ Ray remote call
```

**★★★ Remote vs Naive**: Remote用Ray actor → 独立进程 → 解决thread pool的fork安全问题 → Math-Verify推荐

### 5.7 RateLimitedRewardManager — limited.py:172-540 ★★

```python
@register_manager("rate_limited")
@register_manager_legacy("rate_limited")
class RateLimitedRewardManager(RewardManagerBase):
    # ★★★ 三层限速: concurrency + RPM + TPM
    # class-level只在同一进程共享，不自动跨Ray actor/节点共享配额

    # Layer 1: Concurrency → asyncio.Semaphore(max_concurrent)
    # Layer 2: RPM → AsyncTokenBucket(max_rpm / 60 tokens/sec)
    # Layer 3: TPM → AsyncTokenBucket(max_tpm / 60 tokens/sec)

    async def run_single(self, data):
        # 1. acquire RPM token (1.0 per request)
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire(1.0)

        # 2. acquire TPM tokens (estimated_tokens_per_request per request)
        if self._tpm_limiter is not None:
            await self._tpm_limiter.acquire(estimated_tokens)

        # 3. acquire concurrency semaphore
        async with self._semaphore:
            result = await asyncio.wait_for(
                self._compute_reward(...), timeout=self.timeout  # ★ 300s timeout
            )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
```

**适用场景**：外部API奖励的进程内准入控制；全局API key配额仍须结合worker数量与服务方计量方式判断，不能由三层本地限流推断不会超限。

#### 进程配额与取消边界

2026-10-22 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，核对[原 limited.py](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/experimental/reward_loop/reward_manager/limited.py)。原文“所有worker共享、跨节点统一限速”的笔记表述已纠正：`_rpm_limiter/_tpm_limiter/_semaphore` 是普通Python类属性，只有同一进程内的该类实例共享。[RewardLoopManager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/experimental/reward_loop/reward_loop.py)创建多个Ray actor，各actor在自身初始化路径加载manager；没有在这些类属性上建立跨进程共享服务。若N个独立worker各用同一数值R，不能把它当总计R的共享配额，内建桶的合计长期准入能力可能接近N倍。这里是源码作用域推断，未运行Ray集群或请求外部API。

`asyncio.Lock` 保护同一事件循环的协程竞争，不是操作系统线程锁，也不是跨进程锁。[Python官方说明](https://docs.python.org/3.12/library/asyncio-sync.html)明确这类原语不具备线程安全保证。原桶文档的thread-safe措辞，以及名为 `test_thread_safety_with_lock` 的测试，都不能扩大该保证；后者实际只用 asyncio.gather 启动协程。

manager将RPM/TPM除60作为每秒补充率，同时把桶容量设为这个**一秒额度**，不是整分钟额度；初始桶满。它允许短时突发，按时间连续补充，不是服务方任意滚动60秒窗口、账单token或实际请求完成数的精确计量。TPM每次扣的是配置估值 estimated_tokens_per_request，不会在评分返回后按真实usage结算。首次class初始化还会固定本进程的限流器，后续改变RPM/TPM仅警告并忽略，不能当动态重配置。

两个 acquire 分支对取消有不同含义：

| 请求 | 不足令牌时如何等待 | 本机取消后的状态 |
|---|---|---|
| 不超过桶容量 | 释放锁，睡眠后重新判断，成功时才扣款 | 未扣本次请求费用；保留当时已计算的非负余额 |
| 超过桶容量 | 锁内先扣整笔成负余额，释放锁后等待欠额补足 | 没有refund分支，预扣负余额保留 |

本机真实事件循环对照使用 rate=1、capacity=1：普通路径先耗尽1，再等待另1并取消，余额为微小非负补充量；超容量请求5在第一次sleep前把余额设为−4，取消后仍为−4。后续acquire会按经过的时间补充，这不是永久卡死；但“请求已取消且尚未调用评分API”不等于“本地预算已退还”。未据此改变扣款策略或定性为新的上游缺陷，是否应退款须先明确业务准入契约。

run_single的顺序是 `decode → RPM → TPM → semaphore → wait_for(score)`。timeout只限制最后评分等待，不包含前面的decode、令牌桶或并发队列等待；失败评分也没有统一退还已消费RPM/TPM。async with会释放已进入的并发槽，不代表撤销令牌消费或远端计算。该manager顺序与配置行为是源码核对，没有用替身manager冒充完整请求验证。

[复查脚本](../../experiments/scripts/verify_verl_token_bucket.py)AST选择完整原AsyncTokenBucket和原get_event_loop，完整加载[原测试模块](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/experimental/reward_loop/test_async_token_bucket_on_cpu.py)，逐项直接await原16个测试方法，全部通过；原装饰器、断言、时间窗和0.5/1/4秒等睡眠未改。另2组取消观察通过。没有fake clock或缩短sleep；“直接await测试体”不是完整pytest、跨线程安全或manager集成验收。

```bash
python experiments/scripts/verify_verl_token_bucket.py /path/to/verl
```

Python3.12.13、pytest8.4.2，无torch/Ray/模型/API/设备依赖。原manager集成测试需要DataProto、tokenizer及其他依赖，本轮未运行。官方API/本地limited.py Git blob同 `3707ed20e5ab2b8ee2db854d8a344803a26b39cb`；SHA256：limited模块 `62ea50e918f2ea1bd2df250ef3cd02564bc02fe8d567f53727f4c538d3751d65`，原测试 `6ed7dadeafdccbc64976b46d67843f2877444802df85afe9dcf650e6eb80445f`，原helper模块 `1f70a9ac05b4a6f92a3ad2486bee17ac9721a9568e7e72b17e76172a60a7751b`。本项修正知识范围，不改上游实现、创建候选或新增M5硬件任务。

### 5.8 全Registry清单 ★★

| 名称 | 层 | 基类 | 接口 | 特点 |
|------|---|------|------|------|
| naive (legacy) | workers/ | AbstractRewardManager | `__call__` | per-item loop, sync |
| batch (legacy) | workers/ | AbstractRewardManager | `__call__` | batch scoring, dict return |
| dapo (legacy) | workers/ | AbstractRewardManager | `__call__` | overlong penalty |
| prime (legacy) | workers/ | AbstractRewardManager | `__call__` | ProcessPoolExecutor(64) |
| naive (exp) | experimental/ | RewardManagerBase | `async run_single` | async, event loop executor |
| remote (exp) | experimental/ | RewardManagerBase | `async run_single` | Ray actor, separate process |
| rate_limited (both) | both registries | RewardManagerBase | `async run_single` | 3-layer rate limiting |
| gdpo (exp) | experimental/ | RewardManagerBase | `async run_single` | per-dimension reward dict |
| dapo (exp) | experimental/ | RewardManagerBase | `async run_single` | same as legacy dapo |

## 6. Colocated RM: Sleep/Wake模式 ★★★

### 6.1 架构

```
┌─────────────────────────────────────────────────────────────┐
│ Colocate Mode (default): RM共享actor/rollout的GPU            │
│                                                             │
│  训练step:                                                  │
│  1. Rollout model醒来 → 生成rollout                          │
│  2. Rollout model睡觉 → 释放GPU                              │
│  3. ★ RM model醒来 → 加载到同一GPU → 推理                    │
│  4. ★ RM model睡觉 → 释放GPU                                 │
│  5. Actor model醒来 → 继续训练                                │
│                                                             │
│  Standalone Mode: RM有独立resource_pool                      │
│  → RM始终醒着 → 与rollout并行 → streaming reward             │
│  → enable_resource_pool=True + nnodes + n_gpus_per_node      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 RewardModelManager — reward_model.py:27-124

```python
class RewardModelManager:
    def __init__(self, config, resource_pool=None):
        # ★ RM replica = vLLM/SGLang/TRT-LLM server
        # resource_pool → colocate模式 → 共享actor GPU
        # None → standalone模式 → 独立GPU pool

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=i, config=rollout_config,
                model_config=HFModelConfig(path=config.model_path),
                is_reward_model=True,  # ★ 标记为RM → 不同初始化逻辑
            )
            for i in range(num_replicas)
        ]

        # ★ Colocate: split_resource_pool → 每replica一部分GPU
        if self.resource_pool:
            split_resource_pools = split_resource_pool(resource_pool, split_size=rollout_world_size)
            self._run_all([server.init_colocated(pool) for server, pool in zip(...)])

        # ★ 初始化路由 → load balancing
        self._initialize_router()  # naive_router → HTTP proxy

    def wake_up(self):
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        self._run_all([replica.sleep() for replica in self.rollout_replicas])
```

### 6.3 compute_rm_score 流程 — reward_loop.py:323-352 ★★★

```python
class RewardLoopManager:
    def compute_rm_score(self, data):
        # ★★★ Step 1: Wake up RM (colocate模式)
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        # ★ Step 2: 分chunk到worker → parallel计算
        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get([worker.compute_score_batch.remote(chunk) ...])

        # ★ Step 3: assemble rm_scores → outcome tensor (只有最后token非零!)
        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, scores)

        # ★ Step 4: Sleep RM (释放GPU → actor可以继续训练)
        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(data)),
            non_tensor_batch={key: np.array([...]) for key in reward_extra_keys},
            meta_info={"reward_extra_keys": reward_extra_keys}
        )
```

### 6.4 训练loop中的reward集成 — ray_trainer.py:1518-1603 ★★★

```python
# ★ 训练step的reward部分:
with marked_timer("reward", timing_raw):
    # ★ Step 1: Colocate RM → wake → compute → sleep
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)  # → wake RM → compute → sleep RM
        batch = batch.union(batch_reward)  # → rm_scores加入batch

    # ★ Step 2: extract reward → 统一接口
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    # → extract_reward: batch.batch["rm_scores"] → reward_tensor
    # → 如果没有rm_scores → 用reward_manager计算 → 同样是outcome tensor

# ★ Step 3: reward → token_level_scores
batch.batch["token_level_scores"] = reward_tensor

# ★ Step 4: KL-in-reward (可选)
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
    # → token_level_rewards = token_level_scores - beta * kld
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    # ★★★ 直接赋值! reward = scores → 无KL修改

# ★ Step 5: compute advantage
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)
# → GRPO: scores = token_level_rewards.sum(dim=-1) → group-relative → broadcast
```

### 6.5 Streaming Reward ★★

```python
# ray_trainer.py:940-956
# ★★★ streaming条件: rule-based OR RM with extra resource pool
enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

# ★ rule-based (no RM): reward_loop_workers传给agent loop → 每个sample立即reward
# ★ RM standalone (enable_resource_pool=True): RM有独立GPU → 可以并行
# ★ RM colocate (default): 不能stream → 必须等所有rollout完成 → 然后wake RM
```

**★★★ Colocate vs Standalone对比**:

| 特性 | Colocate | Standalone |
|------|----------|------------|
| GPU | 共享actor GPU | 独立GPU pool |
| 执行 | 先rollout→wake RM→compute→sleep RM | RM始终醒着→并行 |
| Streaming | ✗ 必须等全部rollout | ✓ 每个sample立即reward |
| GPU内存 | 需要RM能fit进actor sleep后的空间 | 需要额外GPU |
| 配置 | 默认 | enable_resource_pool=True |
| RM延迟 | 高(serialize) | 低(parallel) |

## 7. Built-in Reward Score函数 ★★

### verl/utils/reward_score/__init__.py: default_compute_score

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None,
                          sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None, **kwargs):
    # ★ 按data_source分派到具体评分函数:
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)  # 字符串匹配

    elif data_source in ["lighteval/MATH", ...]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)  # LaTeX对比

    elif data_source in ["math_dapo", "math", ...]:
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in ["codecontests", "apps", ...]:
        if sandbox_fusion_url:  # ★ sandbox执行 → 需要URL!
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(sandbox_fusion_url, ..., solution_str, ground_truth)
        else:
            from . import prime_code
            res = prime_code.compute_score(...)

    elif data_source in ["searchR1_nq", ...]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(...)

    else:
        raise NotImplementedError(f"Reward function not implemented for {data_source}")
```

**★★ 已支持的数据集**: GSM8k, MATH, MATH-500, math_dapo, AIME, numina系列, codecontests/apps/codeforces/taco, geo3k, searchR1系列

**★★ Sandbox Fusion**: 代码任务 → 需要执行代码 → sandbox_fusion_url → 云端代码执行 → concurrent_semaphore控制并发

这条旧概括描述 HTTP 服务路径；下面的 #8345 候选还允许包注册的本地 runner，并非所有代码奖励都必须是云端服务。

## 沙箱运行器注册与执行边界

2026-10-23 阅读已有 [#8345](https://github.com/verl-project/verl/pull/8345)，OPEN/non-draft、作者 danbugs，固定 head `72ccdb6d242302157251a8d2e59943d9bd9ea113`、目标 base 与共同祖先 `d430157d15516d89e0cc853fd10fcd21d9234c44`，ahead 1 / behind 0，9 文件 476 增 12 删。它给既有 sandbox_fusion 评分流程增加 runner 派发点，不是新评分算法或我方贡献候选；也不是默认已安装、已合并的沙箱后端。

### 注册与 URL 路由

[`sandbox_runner_scheme`](https://github.com/verl-project/verl/blob/72ccdb6d242302157251a8d2e59943d9bd9ea113/verl/utils/reward_score/sandbox_fusion/utils.py)按自身规则解释 URL：自定义 scheme 转小写，支持 my_box 等非标准 URI scheme；http/https 及 http+unix 等扩展、无 scheme 的地址继续走服务路径。自定义 runner 收到 strip 后的原 URL，以及按关键字传入的 code/stdin、compile/run timeout、memory limit 和 language。自定义语言由 runner 决定，不受 HTTP 服务语言表限制。

注册表是**每进程的 Python 状态**。`verl.__init__` 的 entry-point 加载受 `VERL_USE_EXTERNAL_PLUGINS` 控制，或可由显式外部模块触发注册；驱动侧装好包不等于每个实际 reward 进程都已注册。注册拒绝非 callable 和 coroutine function；相同对象可重复注册，同模块对象可按重载语义替换，不同模块争用同 scheme 则记录冲突、warning，并在 lookup 时拒绝，而非静默选最后一个。模块名判据不是包来源、代码身份或安全认证。

### 预检、逐例错误与执行责任

| 边界 | 原实现行为 | 不能据此声称 |
|---|---|---|
| manager 启动预检 | 在无 custom reward function、配置 sandbox URL 的分支查本进程 runner，再创建 semaphore | 所有自定义 reward/远端 worker 都提前验证成功 |
| 找不到或冲突 | lookup 抛 ValueError；逐例 API 包装将它变成错误，整条详细信息每 scheme 只记首次 | 每次错误都能靠日志看到完整环境原因 |
| runner 抛异常 | 按 scheme/异常类型只首次记 traceback，各次仍返回错误 | 后续没有 traceback 就代表成功，或失败已被自动恢复 |
| runner 返回值 | 检查 `(dict-or-None, str-or-None)` 二元组及至少一个真值；空 error 归 None，后续复用状态/输出判分 | 已深度验证响应 schema、真正运行过程序或建立了安全隔离 |

`_process_single_case` 继续把包装层错误映射为 `api_error` / -1，成功回复再按既有 stdout/状态协议判分。注册不是执行沙箱的实现：runner 自己负责代码隔离、线程安全、语言支持与资源限制。`call_sandbox_api` 直接同步调用 runner，没有替它加超时中断；`check_correctness` 的线程池也不会因传入 run_timeout 自动杀掉阻塞 runner。HTTP 的请求 timeout/504 重试逻辑不能直接套到这个分支。

[`load_reward_manager`](https://github.com/verl-project/verl/blob/72ccdb6d242302157251a8d2e59943d9bd9ea113/verl/trainer/ppo/reward.py)的启动检查只覆盖它所在进程与特定默认评分分支；测试通过不能升级成真实 plugin entry-point 已加载、每个 Ray worker 环境一致或全部自定义 reward 接线正确。这里没有把进程内注册表/锁或所传 semaphore 当成集群全局配额。

### 原测试复跑范围

Python 3.12.13、pytest 9.1.1、requests 2.34.2 的已有隔离环境，两次运行均为 **14 passed / 16 deselected**。选中原新增 runner 测试：关键字派发与大小写/空白、echo 结果通过原 check_correctness 得到 True/False、缺失与冲突、异常/畸形返回、空 error、runner 自定语言、注册校验，以及 HTTP scheme 的 7 个参数例。echo/报错 runner 均是作者夹具，未执行 code 字符串。HTTP scheme 测试只核路由分类，不是实际 HTTP 请求验收。

完整原 utils、包入口和测试文件均从固定 Git 对象载入，未改函数/测试体；只用空根包壳隔离 verl 根初始化及其 Ray/设备/entry-point 副作用。原测试自己的 fixture/mock 保留，另阻断 requests.post，实际未发 HTTP。16 个未选项包括 11 个真实服务用例、4 个旧 unit 用例和 1 个 manager 启动预检，**不是把它们算作通过或 skip**；未用假 manager 代替启动链。两版 5 个变更 Python 文件共 10 模块另做 compile。基础 CPU 环境缺 requests 后复用已有环境，没有安装新包。

复现 harness 与 git archive 命令留[当日日志](../../diary/2026-10-23.md)，不新增永久工具。原 utils SHA256 `173c605ff7e51310f44592b754844cbe3ba3cc72b6055ea742a54a873c5c257a`，原测试 `272cc64036092074bf8b4f15e6064a83f31fe3a9f7f5238cb433e85029d06584`。PR 作者报告的 19 passed/11 skipped、真实 Hyperlight 0.1.0 micro-VM 4/4 不是本机结果；本轮没有安装插件、加载 entry-point、部署服务或验证安全性/真实超时/训练。这里没有新增 GPU/NPU 验证需求，不为未测服务或 micro-VM 自动排 M5 任务。

## 8. 配置体系 ★★

### reward.yaml (verl/trainer/config/reward/reward.yaml):

```yaml
num_workers: 8  # ★ RewardLoopWorker数量 → 控制并行度

custom_reward_function:
  path: null       # ★ 外部文件路径 → null=用default_compute_score
  name: compute_score  # ★ 函数名 → 默认compute_score
  reward_kwargs: {}    # ★ 合入的额外参数

reward_manager:
  source: register  # register(内置) or importlib(外部)
  name: naive       # ★ RewardManager类型 → naive/dapo/batch/prime/remote/rate_limited/gdpo
  module:
    path: null      # importlib时需要
    name: custom_reward_manager

reward_model:
  enable: False     # ★★★ 是否启用RM → False=rule-based only!
  enable_resource_pool: False  # True=standalone RM
  n_gpus_per_node: 8
  nnodes: 0
  model_path: null  # RM模型路径
  rollout:
    name: ???       # vllm/sglang/trtllm
    tensor_model_parallel_size: 2  # ★ RM默认TP=2 → 2GPU minimum!
    gpu_memory_utilization: 0.5
    free_cache_engine: true  # ★ sleep时释放KV cache

sandbox_fusion:
  url: null            # sandbox URL
  max_concurrent: 64   # 最大并发sandbox请求
  memory_limit_mb: 1024
```

### ★★★ GRPO典型配置 (RTX 4090):

```yaml
reward:
  num_workers: 4              # RTX 4090: 4 worker足够
  custom_reward_function:
    path: my_reward.py        # ★ 自定义reward函数
    name: compute_score
  reward_manager:
    name: naive               # 简单per-item → CPU执行
  reward_model:
    enable: False             # ★★★ 关闭RM → rule-based only!
```

## 9. 完整数据流: Rollout → Reward → Advantage ★★★

```
训练step的数据流 (ray_trainer.py):

1. Rollout
   gen_batch → async_rollout_manager.generate_sequences(batch)
   → gen_batch_output: {input_ids, responses, attention_mask, log_probs, ...}

2. ★ Reward Computation
   if use_rm:
     → _compute_reward_colocate(batch)
       → reward_loop_manager.compute_rm_score(batch)
         → RM.wake_up()
         → RewardLoopWorker.compute_score_batch.remote(chunk)
           → if custom_reward_function: reward_manager.run_single(data)
           → elif DisRM: compute_score_disrm(data)
           → else: reward_manager.run_single(data) → default_compute_score
         → assemble_rm_scores: zeros→最后token赋值→outcome tensor
         → RM.sleep()
       → batch.union(batch_reward) → batch.batch["rm_scores"] = outcome tensor
   else:
     → extract_reward(batch)
       → reward_tensor = batch.batch["rm_scores"] (if exists)
       → or reward_manager(batch) → outcome tensor (只有最后token非零)

3. ★ Reward → Scores
   token_level_scores = reward_tensor  # (bs, response_length), 只有最后token非零

4. ★ KL-in-Reward (可选)
   if use_kl_in_reward:
     token_level_rewards = token_level_scores - beta * KL(old_log_probs, ref_log_prob)
   else:
     token_level_rewards = token_level_scores  # ★★★ GRPO默认: reward=scores

5. ★ Advantage Computation
   compute_advantage(batch, adv_estimator="grpo_vectorized")
   → scores = token_level_rewards.sum(dim=-1)  # ★ sum=scalar (因为只有最后token非零!)
   → group_mean_std → μ_g, σ_g per prompt
   → scalars = (scores - μ_g) / (σ_g + ε) or (scores - μ_g)  # Dr.GRPO
   → advantages = scalars.unsqueeze(-1) * response_mask  # broadcast到所有token
   → batch.batch["advantages"] = advantages

6. Actor Update
   → _update_actor(batch) → PPO/GRPO loss → backward → optimizer step
```

## 10. ★★★ RTX 4090专项分析

### 10.1 Rule-Based Reward = 唯一可行路径 ★★★

| Reward路径 | GPU需求 | RTX 4090可行性 | 原因 |
|-----------|---------|---------------|------|
| Rule-based (custom_reward_function) | 0 GPU | ✓✓✓ 完美可行 | 纯CPU → tokenizer.decode + score函数 |
| DisRM (7B classifier) | 2+ GPU | ✗ 不可行 | TP=2 → 需2GPU → 单GPU24GB → 即使colocate也fit不进 |
| DisRM (INT4 7B) | 1 GPU | ✗ 不推荐 | ~3.5GB权重 → 但vLLM/SGLang server overhead → KV cache → 总>5GB → marginal |
| GenRM (7B) | 1+ GPU | ✗ 不可行 | 同DisRM → 需要LLM server |
| Standalone RM | 2+ GPU total | ✗ 不可行 | 需要额外GPU → RTX 4090只有1个 |

**★★★ 结论**: RTX 4090只能用rule-based reward → 无GPU空间给RM → custom_reward_function是唯一选择

### 10.2 Outcome Reward与GRPO的完美对齐 ★★★

```
Outcome reward: reward_tensor[i, last_valid-1] = reward → 其他位置=0
→ token_level_rewards.sum(dim=-1) = scalar_reward
→ GRPO: a_i = (r_i - μ_g) / (σ_g + ε) → group-relative标准化
→ ★★★ 天然对齐! GRPO本就是outcome-only算法!

Process reward (per-token nonzero):
→ token_level_rewards.sum(dim=-1) ≠ scalar → 是累计奖励
→ GRPO仍然sum → 语义变了 → 变成"累计奖励的group-relative"
→ ★★★ verl不支持process reward → 所有Manager都是outcome!
→ 如需process reward → 需要custom compute_score返回per-token scores → 但目前没有Manager支持per-token!
```

**★★★ GRPO+outcome=RTX 4090最优**: 不需要process reward → 不需要RM → rule-based outcome → 纯CPU → 单GPU训练+推理

### 10.3 Sleep/Wake RM: 理论可能但实际不行 ★★

```
Colocate RM sleep/wake on RTX 4090:
  Step 1: actor model ~14GB (7B BF16) + LoRA ~0.5GB + KV cache ~2GB → ~17GB active
  Step 2: actor sleep → 释放KV+部分权重 → ~14GB remain
  Step 3: RM wake → 7B BF16 ~14GB → ★ 只剩~10GB → RM fit但KV cache不够!
  Step 4: RM compute → 低吞吐 → blocking → training loop卡住
  Step 5: RM sleep → actor wake → 继续训练

  ★★★ 实际问题:
  1. RM推理吞吐极低(vLLM 7B on 10GB → ~50tok/s → 阻塞training)
  2. sleep/wake开销: 模型加载+卸载 → 每step ~10-30s → 严重拖慢训练
  3. 内存碎片: actor sleep → RM wake → actor wake → 内存管理复杂
  4. INT4 RM: ~3.5GB → 但vLLM INT4不支持所有RM → 不通用

  ★★★ 结论: sleep/wake RM on RTX 4090 = 不现实 → rule-based是最优解
```

### 10.4 GDPO + Rule-Based: RTX 4090新可能性 ★★

```
GDPO需要: custom_reward_function返回dict → {"score": 1.0, "format": 0.5, "accuracy": 0.8}
→ 纯CPU → 不需要RM → RTX 4090可行!

示例: math reward函数
  def compute_score(data_source, solution_str, ground_truth, extra_info):
      format_score = check_format(solution_str)      # 0 or 0.1
      accuracy_score = check_answer(solution_str, ground_truth)  # 0 or 1
      return {"score": accuracy_score, "format_reward": format_score, "accuracy_reward": accuracy_score}

→ GDPO: 每个维度独立GRPO normalization → weighted sum → whiten
→ ★★★ 防止accuracy(0/1)淹没format(0/0.1) → 多维度独立标准化!
→ RTX 4090最优: GDPO + rule-based + GRPO_VECTORIZED + LoRA + bypass_mode
```

### 10.5 推荐配置 ★★★

```yaml
# ★★★ RTX 4090最优GRPO配置 (reward部分)
reward:
  num_workers: 4  # 4 CPU workers
  custom_reward_function:
    path: my_math_reward.py  # 自定义 → 纯CPU → 返回dict(支持GDPO)
    name: compute_score
  reward_manager:
    name: naive  # 简单per-item → CPU → 无GPU开销
  reward_model:
    enable: False  # ★★★ 关闭RM!

algorithm:
  adv_estimator: grpo_vectorized  # ★ 推荐(快10-100x)
  norm_adv_by_std_in_grpo: False  # ★ Dr.GRPO → 防止梯度消失
  use_kl_in_reward: False  # ★ KL只在loss端 → bypass_mode友好
  use_kl_loss: True  # ★ loss端KL → kl_coef=0.001
```

## 11. 关键发现总结 ★★★

1. **★★★ 两层Reward系统**: legacy(AbstractRewardManager.__call__) + experimental(RewardManagerBase.async run_single) → experimental是默认路径 → 通过load_reward_manager()加载

2. **★★★ Outcome reward = 最后token非零**: 所有Manager一致 → reward_tensor[i, last_valid-1] = score → sum(dim=-1) = scalar → GRPO天然对齐

3. **★★★ 3条reward路径**: custom_reward_function → DisRM(HTTP classify/embeddings) → GenRM(custom+HTTP) → 决策树清晰

4. **★★★ Colocate RM sleep/wake**: wake_up→compute→sleep → 共享GPU → 但RTX 4090不可行 → 内存太紧

5. **★★★ Streaming reward**: rule-based only → reward_loop_workers传给agent loop → 每个sample立即reward → RTX 4090可受益

6. **★★★ GDPO**: per-dimension normalization → 防止dominant信号淹没 → rule-based返回dict → RTX 4090可行!

7. **★★★ Dr.GRPO**: norm_adv_by_std_in_grpo=False → 不除σ_g → 防止小group梯度消失 → RTX 4090推荐

8. **★★★ RTX 4090唯一可行路径**: rule-based + outcome reward + GRPO_VECTORIZED + Dr.GRPO → 纯CPU reward → 单GPU训练+推理

## 数学奖励的分层判分与精度边界

奖励函数能返回分数，不代表它保留了题目要求的数学语义。[Issue #8346](https://github.com/verl-project/verl/issues/8346)报告prime_math会接受错误区间括号、100倍/百分之一数值与大整数差1，又拒绝部分原样回填的方程。报告者jimmy-holdout的[#8347](https://github.com/verl-project/verl/pull/8347)已于2026-10-26关闭未合并；以下是2026-10-23的独立CPU复查，不是主线已修复的记录。当时候选open、非draft，head `7f34e0188697b89ec0def7ce243633ba4076b707`、目标base `bfff4d90b911f412a0856abecffc15eeafb78977`、实际祖先 `d430157d15516d89e0cc853fd10fcd21d9234c44`（ahead1/behind1）。四文件185增3删，其中三份生产源码50增3删；脚本断言祖先与目标base的这三份源码完全一致。

### 候选关闭与方程后续

2026-10-26核验：#8347由wuxibin89在02:27:59 UTC关闭，merged=false、merged_at=null，head仍为上述7f34e018。公开关闭事件和普通评论未说明关闭理由；[作者此前的CLA签署障碍](https://github.com/verl-project/verl/pull/8347#issuecomment-6389576199)不能直接当作关闭原因，更不能据此判断修复在技术上被否定。

最新主线`3dd97a0e1b4b2cdf84c302c53431efaad441ae9b`的[评分入口](https://github.com/verl-project/verl/blob/3dd97a0e1b4b2cdf84c302c53431efaad441ae9b/verl/utils/reward_score/prime_math/__init__.py)、[grader](https://github.com/verl-project/verl/blob/3dd97a0e1b4b2cdf84c302c53431efaad441ae9b/verl/utils/reward_score/prime_math/grader.py)与[normalizer](https://github.com/verl-project/verl/blob/3dd97a0e1b4b2cdf84c302c53431efaad441ae9b/verl/utils/reward_score/prime_math/math_normalize.py)三个Git blob均与原基线d430157d相同。区间／数值修复没有通过这份PR进入这些文件；旧64项候选通过的结论仍只属于候选，不认证当前完整运行环境。#8346仍OPEN，报告者也明确表示区间括号及缩放／整数问题尚未解决。本轮只核源码身份，不重复执行旧用例。

方程部分已有新的责任人：[JianhaoYu-AI认领](https://github.com/verl-project/verl/issues/8346#issuecomment-6410234797)仅在gold含等号、最后一个boxed／fbox内容与gold除外层空白外完全相同时增加接受分支；非相同答案仍走原提取和判分。这不是一般方程等价判断，也不删除旧路径的误接受。作者同时拟纠正当前selector优先任意boxed、仅无boxed时才找fbox的行为，使两类框按实际位置选择；该提取顺序与数值／区间比较是不同改动。

新贡献者报告本地CPU测试27项通过，其中七项后备路由测试mock了grader；这些例子证明分流，不证明真实grader会拒绝所有改单位／变量／函数的答案。其基线还出现自述的Windows多进程错误，未声称完成Linux或完整套件验证。该评论未附源码，本项目未复验。报告者[同意其继续并愿在PR出现后复核Linux的1,024行数据](https://github.com/verl-project/verl/issues/8346#issuecomment-6410710152)，属于未来协作，不是已完成数据集验收、维护者批准或本项目的上游写入授权。本项目不接管该认领，也不以关键词检索未命中为据另开区间／数值修复。

### 前段拒绝不是最终否决

`default_compute_score`将六类numina数据源路由到prime_math。它先提取答案、调用`grade_answer`；只有该函数接受才直接返回，否则继续`math_equal`。因此，前段的严格整数或区间检查失败，不会阻止更宽松的后段重新给满分。

原CPU路径对`22501`/gold `22500`得到：祖先`grade_answer=False`、`math_equal=True`、最终1；候选为False/False/0。只测试前段或只读其中“整数精确匹配”的注释，无法发现最终奖励的误接受。后备判分不是“保留前段所有拒绝条件”的单调增强。

### 历史候选处理的表示

[候选grader](https://github.com/verl-project/verl/blob/7f34e0188697b89ec0def7ce243633ba4076b707/verl/utils/reward_score/prime_math/grader.py)在去除括号比较内容前，先对区间形状检查端点括号序列；无穷端点的方/圆括号则归一为开口，Unicode∞也被统一。它是现有字符串判分路径的限定保护，不是任意区间/集合语法的完整解析器。

百分比转换先保留哪一侧带`%`，再决定候选值方向；无百分号的普通数不再默认接受100倍与百分之一。兼容策略仍允许gold `83\%`匹配`83`或`0.83`，而不是把所有无单位数字强制改成唯一数学表示。

整数字面量（可带正负号或零小数尾）在三处归一/比较中使用Python int，避免超过2^53时先经float丢掉相邻整数的差异。**这不保护所有等值写法**：科学计数法仍走旧float路径。本机候选中`9.007199254740993e15`对gold `9007199254740992`仍得1；这是作者已明确列出的表示范围，不是本项目新发现的回归。非整数decimal保留1e-4相对容差，`348.69`对`348.68`仍得1；舍入金标与精确整数不能混为一种匹配政策。

### 保留下来的信息损失

本机两版中，原样boxed的`x - \sqrt{3}y + 2 = 0`仍得0：`match_answer`先取boxed内容，再按最后一个等号裁成`0`，丢掉方程。短左侧另有旧归一规则，`q = 2`对gold `m = 2`仍得1。不能简单改为“看到等号就比较两份归一化字符串”：变量名、单位与函数调用也需要保留，报告者已用反例说明该方向另需规格和测试。

路径之间也可能抵消局部保护。直接`math_equal(A \leftarrow B, A \rightarrow B)`为False，但完整路由仍得1，因为更早的MATH normalizer用普通substring replacement去掉`\left`/`\right`，将两个箭头命令都损为`arrow`。新helper避免误删箭头的5项测试，只证明这个helper所在后备路径，不证明整个组合判分器已保持箭头语义。

### 原测试与本机范围

[复查脚本](../../experiments/scripts/verify_verl_prime_math.py)从固定Git对象加载完整原评分、路由、归一化、timeout及依赖模块，仅跳过verl/utils包初始化；原测试体和参数不改写，逐项直接调用，不是pytest原生收集。

| 原参数组 | 祖先 | 候选 |
|---|---|---|
| 24项目标修复 | 0过/24预期失败 | 24过 |
| 35项保持行为 | 35过 | 35过 |
| 5项后备箭头检查 | 4过/1预期失败 | 5过 |

合计祖先39过25预期失败、候选64过，另7组范围控制通过，两次执行一致。环境为Python3.12.13、SymPy1.14.0、pytest8.4.2、torch2.9.1，pylatexenc2.10与antlr4-runtime4.9.3仅装在独立work target，未修改复用环境。子进程显式用fork运行原multiprocessing timeout，与Linux式CPU上下文一致，不是macOS默认spawn验收；timeout/parser未替换，也未叠加另一个#7437修复。脚本仅喂固定良性字面量，不把原eval/SymPy parser当作不可信输入的安全沙箱。

复现：`python -B experiments/scripts/verify_verl_prime_math.py /path/to/verl-git --deps /path/to/parser-target`。没有下载Eurus数据、复跑作者1024行/变体统计、运行完整reward manager/模型或RL训练；CPU判分结果不等于奖励分布、优化收益或训练安全验收。本题不需要GPU，不新排M5任务，也不自动采用/发布上游补丁。

原测试SHA256 `559f06fc8beb1e4914e1b13817a3cd6c6a0f56f325fd9214f4d9197524805564`；head评分入口`aea08d3e8c7e4fb06b8be07503257c1df54ba95b94d189ab86df7c630cb02740`、grader `461964b772e133095aedcfc84671abd9d184018b6734c97a50063133330d4720`、normalizer `b8acad2a1382307e08c233b3e3cb7747a38fc09460818414c3c07aded3c9f9d8`。脚本输出祖先/head hash并禁止隐式Git fetch。

## 源码文件索引

| 文件 | 关键内容 |
|------|---------|
| verl/trainer/ppo/reward.py | get_custom_reward_fn, load_reward_manager, extract_reward |
| verl/trainer/ppo/ray_trainer.py | _compute_reward_colocate, apply_kl_penalty, training loop reward integration |
| verl/trainer/ppo/core_algos.py | GRPO/GRPO_VEC/GDPO/GRPO_PASSK advantage, kl_penalty |
| verl/workers/reward_manager/abstract.py | AbstractRewardManager ABC, _extract_reward_from_rm_scores |
| verl/workers/reward_manager/naive.py | NaiveRewardManager (legacy) |
| verl/workers/reward_manager/batch.py | BatchRewardManager (batch interface) |
| verl/workers/reward_manager/dapo.py | DAPORewardManager (overlong penalty) |
| verl/workers/reward_manager/prime.py | PrimeRewardManager (ProcessPoolExecutor) |
| verl/workers/reward_manager/registry.py | Legacy REWARD_MANAGER_REGISTRY |
| verl/workers/config/reward.py | RewardConfig, RewardModelConfig, RewardManagerConfig |
| verl/experimental/reward_loop/reward_loop.py | RewardLoopManager, RewardLoopWorker, compute_rm_score |
| verl/experimental/reward_loop/reward_model.py | RewardModelManager (sleep/wake, router) |
| verl/experimental/reward_loop/reward_manager/base.py | RewardManagerBase ABC, assemble_rm_scores |
| verl/experimental/reward_loop/reward_manager/naive.py | NaiveRewardManager (async) |
| verl/experimental/reward_loop/reward_manager/remote.py | RemoteRewardManager (Ray actor) |
| verl/experimental/reward_loop/reward_manager/limited.py | RateLimitedRewardManager (3-layer rate limiting) |
| verl/experimental/reward_loop/reward_manager/gdpo.py | GDPORewardManager |
| verl/utils/reward_score/__init__.py | default_compute_score (gsm8k/MATH/math_dapo/code/geo3k/searchR1) |
| verl/trainer/config/reward/reward.yaml | Default reward config |
