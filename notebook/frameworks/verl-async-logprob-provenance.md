# verl 异步训练策略版本与 logprob 对照条件

固定 token 的 log-prob 对照能缩小故障范围，但必须先对齐策略版本、上下文、有效掩码和概率处理。**相同版本标签不证明权重相同，采样 token 概率相同不证明完整分布相同，后者也不证明参数内容相同。** 这些检查应组合使用，不能用一个曲线阈值替代来源追踪。

适用源码为 verl `1876b06d0a3e4e71e06230be10af14492ca8a75b` 的 V1 separate-async。关联 [#8250](https://github.com/verl-project/verl/issues/8250) 报告明确使用该入口、私有修改及 temperature=1；安装的vLLM版本仍未知。这里不将旧experimental fully_async_policy路径或依赖pin当作该运行环境。阅读日期2026-10-21。

## 四种策略与概率字段

| 对象 | 字段与来源 | 比较时须保留的身份 |
|---|---|---|
| 生成策略 π_rollout | server输出采样token的log_probs，经agent loop变成rollout_log_probs | replica、request/turn、分段版本、实际sampling参数、logprobs_mode |
| proximal策略 π_old | 解耦模式由actor重算old_log_probs；separate-async可交换CPU快照以固定它 | CPU快照/导出checkpoint身份、local_trigger_step、重算时间 |
| 正在更新的 π_θ | ppo_loss中本次forward的log_prob，参与对old的ratio | optimizer/mini-batch位置、实际参数身份、精度与kernel |
| reference策略 π_ref | 启用相应项时的ref_log_prob；不是π_old的别名 | reference checkpoint与是否启用KL/奖励惩罚 |

[base `_compute_old_log_prob`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L1649)有两个分支：bypass将rollout_log_probs直接写成old_log_probs；解耦模式调用actor compute_log_prob，并按response_mask从nested输出还原。bypass中的“old”因此不是一次独立重算，也不能拿old与rollout相等证明同步正确。

[separate-async覆盖方法](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_separate_async.py#L187)在local_trigger_step=0时保存CPU slot 0；后续trigger保存当前actor、恢复slot 0计算old，再恢复当前actor并清理临时slot。它是在同一actor上交换快照，不是另起独立冻结模型。[step循环](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L547)允许一个parameter_sync_step周期内多次本地更新，因此发布版本单位也不能直接当作optimizer step数。

## Bypass 字段替换与 worker 的 loss 配置

`old_log_probs = rollout_log_probs` 只选择概率锚点，不自动决定远端 actor 使用哪一个 loss。[verl #8348](https://github.com/verl-project/verl/pull/8348)将 loss 配置从训练循环移到 worker 创建之前，修复的是配置传递时机，不是权重同步。2026-10-23读取 open、非draft、作者 aoshen02；固定 head `4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349`、base及共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`，ahead2/behind0，9文件375增87删。上节旧版概率来源分析仍限定于原固定版本；本节不将新PR当成已合并行为或 #8250 根因。

### 当前 head 的范围收窄

**2026-10-24接续：以下旧版19项配置测试不适用于当前head。** [#8348](https://github.com/verl-project/verl/pull/8348)仍OPEN/non-draft、作者aoshen02，当前 `2b4940dfc4802a9a26ef333a147a6b36d2afffde` 是上节旧head `4b3b9f84` 后的一个提交，base/共同祖先仍 `a4b9354968705a441fb7da7367924b15b902e771`（ahead3/behind0），现7文件54增48删。相对旧head，删除199行原CPU测试文件，删除`_parsed_setting`及`configure_bypass_mode`中的冲突、镜像、PP/动态CP保护；原FSDP entropy改动也撤回，当前完整transformer_impl.py与PR基线字节相同。这不是仅增加测试的新版本，也不是相关组合已经得到支持。

当前[原配置函数](https://github.com/verl-project/verl/blob/2b4940dfc4802a9a26ef333a147a6b36d2afffde/verl/trainer/ppo/rollout_corr_helper.py)仍在四个trainer的worker创建前调用。bypass关闭时不动；启用时仅对vanilla／bypass_mode复制algorithm侧rollout_correction并设为bypass_mode，显式其他loss直接返回。真实OmegaConf2.3.1、struct=True的五组输入，两head各执行原函数，得到：

| 输入 | 旧head 4b3b9f84 | 当前head 2b4940df |
| --- | --- | --- |
| 默认vanilla，开启bypass | 切换并复制配置 | 同旧版 |
| 显式gspo，同时设置token IS | ValueError | 保留gspo，不新增actor镜像 |
| vanilla的actor镜像loss_type与algorithm冲突 | ValueError | 用algorithm配置覆盖镜像 |
| token IS batch normalize，Megatron PP=2 | ValueError | 接受并设bypass_mode |
| token IS batch normalize，dynamic CP=true | ValueError | 接受并设bypass_mode |

这10次直接配置调用使用本地输入夹具，不运行旧19项pytest，也不构造worker。**被接受不等于配置生效、归约参与者正确或能够训练。** 新函数不再检查显式其他loss下IS/RS是否有意义，也不再认证原镜像与algorithm等价；不能把旧版的拒绝表、阈值同义比较和并行组合保护继续作为新版保证。

当前helper仍在分布式IS batch normalize分支调用`distributed_masked_mean`，因此本轮确认的是“启动guard已删除”，不是PP死锁已复现或动态CP计数已测。自动评论提出的风险也须保持这一证据层级。新版还改变了sequence IS统计对padding行的处理；本轮没有执行张量指标、loss、entropy或collective，不继承旧版相关结论。作者正文的4GPU A/B与113项既有测试是外部报告，未计入这10次配置调用。

本机Python3.12.13，复用已有OmegaConf目标目录，无安装；旧版四个完整原函数、新版一个完整原函数AST隔离，阻止torch/Ray/verl等框架导入；五模块两版共10次compile，四处configure调用仍存在。旧helper SHA256 `bfe5de2a25311db1236e323fcd1753fa763326622a3e41307d81613b14e56c08`，新helper `cb38e82609f9e1dd8d4a46a46610c7808bae473b1071f442c117a18f83036b15`。[本轮复现](../../diary/2026-10-24.md)不替换下方锁定旧head的历史脚本；[M5原条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)已补新版本，不自动应用、替作者补回guard或排训练。

### 旧 head 的创建前接线

旧 [apply_bypass_mode](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/trainer/ppo/rollout_corr_helper.py)一面替换 batch 字段，一面修改 trainer 持有的 policy_loss。worker 已经用配置创建后，修改 driver 的对象不等于更新远端副本。候选把这两件事分开：`configure_bypass_mode(config)` 在构造阶段处理 loss，`apply_bypass_mode(batch)` 只检查 rollout_log_probs 存在并替换 old_log_probs。

[legacy trainer](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/trainer/ppo/ray_trainer.py)、[V1 base](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/trainer/ppo/v1/trainer_base.py)、[experimental fully async](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/experimental/fully_async_policy/fully_async_trainer.py)及[one step off policy](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/experimental/one_step_off_policy/ray_trainer.py)各新增一次构造器调用。V1在后续init→_setup中才把 actor_rollout_ref 交给 RayClassWithInitArgs、spawn worker并init_model；这条源码顺序解释了为何应早配，而不构成真实RPC序列化/worker启动验收。

### 旧 head 的选择与拒绝规则

[原配置函数](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/trainer/ppo/rollout_corr_helper.py)只在 algorithm 的 bypass_mode 为真时行动：

| 输入 | 处理 |
|---|---|
| 默认vanilla或已显式bypass_mode | 设为bypass_mode，并将algorithm.rollout_correction复制到actor policy_loss |
| 显式其他loss，如gspo | 保留该loss；old仍采用rollout概率，但不因此获得bypass loss的IS/RS语义 |
| 其他loss同时要求IS、RS、score_centering或reinforce | 抛ValueError，避免选项被静默忽略 |
| 已有actor侧校正镜像 | 按镜像中出现的键检查一致；阈值按启用模式解析后比较，不只比较字符串 |
| IS batch normalize搭配Megatron PP大于1或dynamic CP | 提前拒绝；不是实现了这些并行组合 |

因此 `2.0` 与字符串 `"2.0"` 可等价；k1 RS标量2与范围 `0.5_2.0` 也可等价。未启用对应IS/RS时，阈值比较返回None，不代表所有原始配置字节相同。源码对PP/动态CP的理由分别是只有末stage计算loss、样本副本数可变；本轮只验证配置拒绝，没有制造collective挂起或验证设备计数。

### 旧 head 原配置测试的 CPU 范围

[复查脚本](../../experiments/scripts/verify_verl_bypass_config.py)使用真实OmegaConf2.3.1、pytest8.4.2、Python3.12.13。四个完整原函数经AST隔离执行，未导入原helper的torch/Ray依赖；[原测试模块](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/tests/trainer/ppo/test_bypass_mode_config_on_cpu.py)字节未改，pytest选择其19个配置参数例，明确deselect另5个padding-only张量/loss例。两次结果均19过、5未运行：5个loss选择、8个拒绝、6个镜像接受用例。测试以struct=True的DictConfig模拟Hydra的结构限制，但没有实际Hydra合成。base没有新configure函数，不虚造“base19例失败”的统计。

```bash
python3.12 -B experiments/scripts/verify_verl_bypass_config.py /path/to/verl-git --deps /path/to/omegaconf-target
```

本机复用既有CPU解释器与 `/private/tmp/perpetual-tq-config-ybC7aP/deps`，没有安装新依赖。另对变更Python源码做base7份/head8份完整AST/内存compile。head helper SHA256 `bfe5de2a25311db1236e323fcd1753fa763326622a3e41307d81613b14e56c08`，base helper `763158342551bc4077805a8a009d959a7a2043c4d43b772fe6045f4a40932f3e`，原测试 `aec35fb58bda99c2ff7c33c210cbefbe8f9d710190b4af236042fceb7c1faa6d`；完整patch `856b5a4c6067b682dfcf4ab7daaaf59340994b78ac9764aa3ce54f213c9b5ff5`。

PR还修改padding-only统计与[FSDP entropy反向开关](https://github.com/verl-project/verl/blob/4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349/verl/workers/engine/fsdp/transformer_impl.py)。19例没有执行loss、空rank归约、entropy梯度或引擎；作者正文的24项、其他102项以及4GPU短训练报告均为外部结果，不能合并进本机通过数或推出性能/收敛结论。真实worker收到的配置及实际选中loss，才是下一层证据；条件已补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用候选或增加训练队列。

## 现成 logprob 对照与权重自检

**verl 已有功能对照脚本，但通过平均 log-prob 门槛不等于认证权重内容。** [#8252](https://github.com/verl-project/verl/issues/8252)在2026-10-20收到[维护者回复](https://github.com/verl-project/verl/issues/8252#issuecomment-6286585440)，指向`scripts/rollout_logprob_diff.py`和QAT文档；10-21报告者继续解释其期望的slime式自检。旧评估里的“0评论、维护者未回应、今天无验证机制”不再成立。以下固定当前主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`，不改变前文旧版separate-async的适用范围。

### 真实更新之后比较所生成的 token

[脚本](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/scripts/rollout_logprob_diff.py)创建Megatron actor与vLLM/SGLang rollout、调用`init_model`，再经`CheckpointEngineManager.update_weights()`执行初次同步。默认8 prompt各生成4条，actor对这些真实response重算`old_log_probs`，与rollout记录的概率比较；不是bypass复制。它会配置量化/QAT/R3，但具体是否启用取决于参数。`--refit-rounds`默认1，额外轮次先再次同步、再重新生成和重算；循环本身不做optimizer更新，也不固定各轮生成token相同。

这能暴露所选配置和轨迹上的训练／推理数值差异，却没有RFC要求的独立已知正确runtime基线、主动破坏及恢复比较。尤其[QAT文档](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/docs/low_precision/qat.md)描述rollout从dummy权重启动后refit，不能未经核对就把首次push前的引擎张量视为同checkpoint参考。文档中Hopper/Blackwell的测量是上游报告，不是本项目复测；相同checkpoint的不同计算后端也允许非零数值差异。

### 指标报告与退出条件不同

原[compute_offpolicy_metrics](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/rollout_corr_helper.py#L904)计算`d=old_log_prob-rollout_log_prob`，`k3_kl`是有效response token上`exp(d)-d-1`的均值。它不是逐词表精确KL或逐张量比较。脚本还报告最大绝对残差，但最终退出只检查`kl/k3_kl`有限及`k3_kl < --max-k3-kl`（默认0.01）；等于阈值也失败。多轮时保留`refit_runs`数组供检查，退出判定使用最后一轮，不自动要求每轮过门槛。

Python3.12.13/torch2.14.1 CPU FP64，三个完整原函数（metric、masked_sum、masked_mean）及原汇总／判定语句的局部控制：

| 显式输入控制 | k3_kl | 最大绝对残差 | 原默认判定 |
| --- | --- | --- | --- |
| 1000 token逐项相同 | 0 | 0 | 通过 |
| 全1000 token的d=0.2 | 0.02140275816 | 0.2 | 失败 |
| 仅1/1000 token的d=1，其余0 | 0.0007182818285 | 1 | 通过 |

再把第二行作为早轮、第一行作为末轮交给原汇总／判定语句，数组仍保留早轮超阈值，退出却通过。这里的log-prob和轮次字典都是显式夹具，没有执行模型或refit；用来解释平均与最后一轮门槛，不据此宣称上游真实运行有异常或提出竞争修复。

两类检查应分别命名：功能对照回答“在这些输入、token和概率处理下有多大差异”；破坏再恢复的内容自检回答“覆盖的runtime张量能否沿真实更新路径恢复参考内容”。后者也受参考来源、覆盖张量/replica及量化容差限制，不能认证未来每次发布或有在飞请求时的生命周期。完整脚本、CLI/Hydra、Ray、DataProto、actor、引擎与硬件均未在本机执行；三模块compile及[完整复现/hash](../../diary/2026-10-24.md)留日志，不新建永久工具。若M5既有匹配观测需要，见[原交接参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动启跑或排新矩阵。

## 版本标签说明什么

[vLLM server](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L689)在生成返回处附加self.global_steps；[客户端](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/llm_server.py#L281)拼接中断后续写的token/log_probs，记录首末返回版本。版本正常单调时，它们形成生成区间；**不是逐token权重清单或tensor内容校验**。多turn和agent重写后的token/mask边界还须一同保留。

源码staleness使用 `(training_step-1)-max_version` 与 `(training_step-1)-min_version`，span使用 `max-min+1`。例如首末版本7/9、training_step=10，对应新鲜度下界0、最坏2、区间宽度3；这不证明版本8实际参与过生成，也不表示经历3次optimizer更新。解释这些量时须知道指标采集相对发布的时序，不能仅照注释中的“不同版本数量”理解。

[named-tensor安装路径](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L210)在有server且提供版本时，先等待返回的update future（若有），再clear_kv_cache、set_global_steps。[同步管理器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/checkpoint_engine/base.py#L510)还围绕传输做abort、release、finalize和resume。这是控制顺序，不是已经证明每个shard/LoRA/量化权重正确应用；版本赋值、线缆校验和功能对照是不同证据。

## 同权重也可能产生不同 logprob

rollout配置[默认logprobs_mode](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/config/rollout.py#L197)是processed_logprobs，并被传入vLLM启动参数。actor重算路径也有temperature缩放和fused/非fused分支。实际返回概率经过哪些processor，要按安装的vLLM版本及运行参数核对，不能仅从字段名认定是同一种概率。

即使权重相同，temperature、惩罚项、top-k/top-p/约束处理、tokenizer/template、完整上下文、position/attention mask、模态预处理或数值后端不同，也可能改变可比较的量。#8250报告temperature=1，不能据此将温度假设认作其故障原因；它只是对照时必须锁定的一项。

还要保留逐token残差。两个token的log-prob差为+.2/−.2时均值为0，非逐项一致；mean≈0不是“完全匹配”，也不是精确分布KL。只观测采样token的概率时，更不能排除其余词表区域的差异。

## 采样截断后的支持域与归一化

**同权重不保证 rollout 与 actor 的概率同口径；先分清采样分布，再解释 IS 权重、拒绝率和 K3。** [#8020](https://github.com/verl-project/verl/issues/8020)报告 top-p／top-k／min-p 截断与 processed_logprobs 配合时的筛选问题，报告者已有支持域大小回放原型。本节2026-10-25固定主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，只核原数值函数与显式小分布，不将作者的设备曲线算作本机结果，也不认定这是 #8250 的原因。

### 相同模型与不同采样分布

设未截断分布为 p，采样保留集合 S，其概率质量 Z=Σₛp；对 S 内 token，截断并归一化后的 q=p/Z。因此即使权重不变，比较全词表 log p 与采样 log q，差仍为 log Z；单 token 的 p/q=Z，长度 L 且每步保留质量相同的序列比值为 Zᴸ。它是两个不同分布在已采样位置的比值，不是指数或减法算错，也不能直接当同步失败。

原 [IS／RS helper](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/rollout_corr_helper.py)在 Python3.12.13／torch2.14.1、CPU FP64 下执行：六个完整原函数及三个常量经 AST 隔离，masked_sum／masked_mean 也取完整原定义，无指标替身。输入 p=[0.6,0.3,0.1]、S={0,1}、Z≈0.9，十个有效 token 为 [0,1] 重复五次，response_mask 全1；TIS上限2、batch normalize关闭。

| 原函数观测 | 全词表 p 对截断 q | 将 actor 也归一化到同一已知 S 的控制 |
| --- | --- | --- |
| token IS 权重 | 0.9 | 1 |
| sequence IS 权重 | 0.3486784401 | 约1 |
| seq_mean_k1，范围0.999_1.001 | 保留0/10 token | 保留10/10 |
| seq_sum_k1，范围0.5_2 | 保留0/10 token | 保留10/10 |
| seq_mean_k3，上限0.01 | 保留10/10；统计量约0.00536051565 | 保留10/10；接近0 |

k1源码统计量是 −log_ratio，而非 log_ratio；此处两种双侧界限仍均拒绝。K3 的 exp(d)−1−d 在 d=log0.9 时约0.00536，而本例精确 KL(q∥p)=−log0.9≈0.10536。q 在 S 外为0，E_q[p/q]=Z而非1，不能将该 K3 样本均值当完整精确 KL。这个有限例子说明小 K3 与严格筛选全拒绝可以同时发生，不表示任何截断或任意长度都会全拒绝。

同支持域控制改变了 actor 的归一化口径，用于验证差异来源；**不是对原目标 p 的无偏修复**。q 在 S 外没有覆盖，不能单凭删掉 log Z 或改为 raw log-probs 就证明完整目标的支持条件、温度／惩罚处理和估计器均正确。

### 配置保护不等于支持域回放

已有 [#8207](https://github.com/verl-project/verl/pull/8207)是 OPEN、非draft、未合并；固定 head `0ed74d918f8b874ec2d742bd8078ced240719f28`，base及共同祖先 `2035f58dcb06d8cad4caee107b30f4045d724367`，3文件256增4删。读取两份生产改动：配置保护检查 algorithm 或 actor policy_loss 镜像是否启用 IS／RS，再拒绝 processed_logprobs 与 top_p<1、top_k>0 或 min_p>0 的组合。它阻止一种配置，不实现报告者的支持域回放，也不认证其余 logprobs_mode 可用于所有后端。另补全零 IS 的 ESS 分母保护及每进程一次的全拒绝／全零告警；告警不代替逐步有效计数或真实梯度检查。

已合并的 [#8196](https://github.com/verl-project/verl/pull/8196)是另一个范围：merge `8ced5d2f4b148c2991217554e0c6f03028fa1470` 为上述主线祖先。[score_centering_auto_correction](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/score_centering.py#L262)按概率提供者的逐 token 完整支持域凭据选择支持域归一化；没有凭据时保留 trainer 全词表尾部近似，sampled-score shift为0。记录预算、配置中的 k 大小或 head 质量接近1都不能单独证明完整覆盖。不能据此宣布普通 IS／RS 路径已解决 #8020。

支持域大小本身也有条件。另一个显式控制令 actor 分布变为 [0.46,0.10,0.44]：旧采样集合 {0,1} 的质量是0.56，但在 actor 上重取 top2 得到 {0,2}、质量0.9。因此只有 k=2 不能在排名跨越后恢复原集合；这不是对报告者完整原型的实现判错，而是回放协议必须核对的身份条件。

两次 CPU 数值运行一致；五份完整 Python 源码另过 AST／内存 compile。没有运行 #8207 原测试、配置启动、完整 helper 模块导入、score centering 梯度、概率提供者或真实 vLLM／SGLang／训练。复现输入、函数清单及 hash 留[当日日志](../../diary/2026-10-25.md)；不新增永久合成工具或竞争补丁。真实后端的概率身份只补[既有 M5 条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动启跑、改 sampler 或排新训练。

## 最小观测契约

每条用于对照的记录应能回答以下四组问题，缺失项用unknown而不是推定值补齐：

1. **同一输入吗**：sample/group/request/turn/token索引，完整prompt和response token身份，工具上下文、有效mask、position与模态处理；明确已生成token是否被重新tokenize或重写。
2. **同一策略吗**：rollout replica与分段版本、old快照与trigger、θ更新位置、reference身份；导出manifest应说明参数名映射、shape/dtype与shard，必要时校验安装后的内容。不是只记录一个global_step。
3. **同一概率吗**：采样/teacher-forcing方式、temperature、penalty和约束参数、logprobs_mode、库/kernel/精度及量化设置；明确当前比较raw还是processed概率。
4. **如何判差异**：逐token绝对/相对误差、分位数/最大误差、有效计数、非有限值和按版本/长度/turn分组。容差来自同设置重复基线，不先规定一个通用“治愈阈值”。

## 三路对照如何缩小范围

| 对照 | 条件成立后能支持什么 | 不能单独断言什么 |
|---|---|---|
| 训练actor与同checkpoint重载actor | 所选token上的导出/重载与该推理路径是否一致 | 全部参数或所有输入等价；vLLM已同步 |
| 同backend干净重载与live rollout | 固定版本、处理及cache政策后，将差异定位到live状态相关路径 | 唯一原因就是weight-sync或cache；仍可能是配置、shard、路由/状态差异 |
| rollout存储值与重算old/current | 配对版本与分段后，观察合法staleness或数据/概率路径差异 | 混合旧版本的response必须等于全当前版本重算；不等就一定是bug |

单个采样token概率相同可对应不同的其余词表分布；给所有logits加同一常数则整个softmax不变。功能对照可以发现不一致，但“对照通过”不能反向认证tensor内容完全相同。要验证参数内容，应使用规范化后的安装manifest/内容校验，并区分传输前、传输后与应用后检查位置。

## 本机验证范围

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250
```

[脚本](../../experiments/scripts/verify_verl_logprob_provenance.py)默认模式在Python3.9.6/3.12.13、NumPy2.0.2通过7组成功路径控制：原始base bypass分支、原始separate-async路由方法，queue/base重算/actor均替身，CPU slot只存标识符。另有7类概率/元数据反例，覆盖处理差异、采样token覆盖不足、softmax平移、均值抵消、版本区间宽度、声明标签与混合版本重算。它不运行模型、vLLM、真实快照/传输或GPU；可选异常检查见下节。

脚本打印输入hash。核心路由文件SHA256为trainer_base `f6c18e2056f6ae2f3fa5436ab24a97ff2c90f8d4bc59bf4f82cb20f80a77baf9`、trainer_separate_async `fb169bfe966816f18cdaaa361761b1e4dc58b577e69e7103211717ab25f70cfb`。源码与这些替身检查说明上述观测为何需要分层，不足以确定私有训练故障根因或提出修复。

## 同步完成点与异常传播

同一固定版本的named-tensor适配器与非naive管理器有两层成功顺序：

```text
适配器  发起receiver RPC → 发送bucket → 等待receiver future
        → 清cache → 标记版本
管理器  abort生成 → release cache → 建通信组 → 等待actor及rollout更新
        → 等待finalize → resume cache → resume生成
```

`_execute_method(non_block=True)`在有server时返回future，调用方随后等待它；这里的non_block不等于“已经安装完成”。管理器在同一次ray.get前已提交actor与rollout更新，某个ref报错也不代表其他远端工作未开始或已取消。以上是调用关系，不是测得的真实并行时序。

| 最后到达的阶段 | 失败后不能推定的状态 |
|---|---|
| bucket发送或receiver确认 | 不能推定没有部分应用，也不能推定所有shard已一致 |
| clear cache | 后续版本调用未发生不等于参数仍是旧版；cache清理可能也只部分完成 |
| set version | 调用报错时远端是否已赋值仍需核对，不能只看客户端异常判断版本状态 |
| manager finalize | 已有worker可能完成更新，方法没有统一撤销先前效果 |
| resume cache或resume生成 | 资源恢复与对外可服务是不同完成点，方法返回前的单个成功阶段不能替代全体就绪 |

这些方法体中没有捕获所测异常并自动rollback/resume的补偿块；这不等于整个进程没有外层清理，也不等于缺陷。失败后继续暂停可能是在避免暴露混合权重。**不能直接把“在finally无条件resume”当修复，也不能未确认各worker状态就宣称重试幂等。** 对恢复的判断需要worker集合、目标版本、最后一次提交/确认、安装后manifest及cache代次共同支持。

有两条路径不能套用上图：naive后端直接交给actor更新；delta-flush空流有仅推进版本、不发送payload和清cache的专用分支，协议意图是值未改变。后者不能只因没有传新payload就判漏同步；delta流内容与协议正确性不在此次执行范围。

### 阶段化故障注入

```bash
python3 experiments/scripts/verify_verl_weight_sync_control.py /path/to/verl-8250
```

[脚本](../../experiments/scripts/verify_verl_weight_sync_control.py)提取未修改的ServerAdapter.update_weights和CheckpointEngineManager.update_weights协程体，移除框架装饰器后以标准库asyncio运行。Ray/worker、sender、接收future、cache和版本设置均为本机替身，不读取或写入设备权重。Python3.9.6与3.12.13均通过21组：10组adapter控制（3种成功条件、非法format、delta委托以及5处异常）和11组manager控制（普通/naive成功及9处异常）。

注入点记录的是**调用尝试**而非成功提交。所测异常均传播到本机调用者，后续阶段未执行；fake ray.get按给定次序解析ref，仅检查管理器程序控制流。未测试真实Ray取消/并发、超时、worker死亡、通信错误语义、CUDA流完成、跨rank内容一致、外层auto_await运行模式或故障恢复。它是边界检查，不是native同步故障复现或#8250根因证据。

源文件SHA256：ServerAdapter `e82e1c6782573e03c871b982eb0653801f54cb1647367a604aa8b1c5454773cb`；CheckpointEngineManager `97b51131a5770892007b7fb07264a2657fe83255b085927acd08c952a5682f64`。

## CPU快照交换异常与训练退出

**方法未恢复当前权重，不等于默认训练会带着旧权重继续运行。** 同一固定版本的separate-async覆盖方法没有围绕重算设置恢复用的try/finally；当trigger≥1、恢复slot 0已经成功而base重算抛出异常时，后面的恢复当前快照与清理临时slot均不会由该方法执行。这是源码控制流边界，不是#8250故障复现。

下表使用标量actor替身：每次故障在该操作自身修改状态前抛出，先前已完成的操作保留效果。真实RPC/设备操作可能先部分修改再失败，不能套用这些标量状态。

| 注入位置 | 替身中保留的状态与未执行步骤 |
|---|---|
| trigger=0，保存后重算失败 | actor仍为本周期π_old；没有后续交换步骤 |
| trigger≥1，恢复slot 0后重算失败 | actor为π_old，当前快照仍在临时slot；未恢复当前、未清slot |
| trigger≥1，恢复当前失败 | 本替身仍为π_old、临时slot尚在；真实恢复是否部分完成未知 |
| trigger≥1，清临时slot失败 | 本替身已恢复当前权重，但临时slot仍在；不能由清理异常反推权重未恢复 |

V1 separate-async实际选用的[DetachActorWorker快照方法](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/experimental/separation/engine_workers.py#L123)还要求核对slot前置条件：**当cpu_saved_models字典已经存在但指定key缺失时，restore直接返回，不调用restore_handler。** 因而“方法正常返回”本身不是快照已安装的独立校验；正常调用序列是否会丢失key尚未证明，不能据此报新bug。fsdp2/veomni还将保存的state与global_spec分别传给handler，其他策略直接传保存值；本轮没有执行这些原生tensor helper。

外层语义不同：[PPOTrainer.fit](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L426)没有在该step周围吞掉异常并重试；重算失败使本次调用无法继续到后续actor update。[TaskRunnerV1.run](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/main_ppo.py#L133)在finally中结束tracking（失败exit_code=1），再以嵌套finally关闭TransferQueue，即使tracking.finish自身失败也会尝试close。**这些清理不是模型/slot回滚。** 默认这次fit/run调用退出；不据此推断Ray actor必定销毁、外部调度不重启，或私有调用者不捕获异常复用actor。后几项未检查，若要支持继续服务/重试，需另定义状态核验与恢复契约。

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250 --check-snapshot-failures
```

Python3.9.6/3.12.13、NumPy2.0.2均通过新增21组：15组原始separate方法异常传播（解耦trigger 0/1/3各阶段及bypass重算异常）、2组原始worker方法分支（fsdp/fsdp2，标量保存/恢复/清理及缺key不操作）、4组原始TaskRunner方法成功/初始化失败/fit失败/fit与tracking.finish均失败控制。方法体由AST提取并去除装饰器，base重算、actor、handler、trainer、tracker与queue均替身；TaskRunner检查与快照检查分别执行，未贯通真实fit。它不证明原生快照深拷贝、跨rank一致、collective完成、真实队列关闭、Ray失败聚合或故障后的恢复/继续训练安全。

新增源文件SHA256：DetachActorWorker `e6799095d8e83890806582e625a6a334cb3f6f7ae17b408673f096bb5c6d96d1`；main_ppo `270ffe0d0f5ddbc4e9af8d0f6eece5d11b1a6d6d76166603b5ebac81345dd5cc`。这些21组去装饰器检查没有执行RPC层，包装层另见下节。

## 快照RPC等待与失败传播

**默认阻塞调用不是跨rank事务，异常返回也不是全体worker已停止的回执。** 同一固定版本的三个快照方法只指定ONE_TO_ALL；[register](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/base/decorator.py#L398)默认execute_mode=ALL、blocking=True。ONE_TO_ALL把同一个slot参数复制给全部worker；[方法绑定器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/base/worker_group.py#L185)读取这些元数据，生成的[调用包装器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/ray/base.py#L49)先execute，再ray.get整个结果列表，最后collect。execute_all虽是异步提交方法的别名，不表示这个已绑定的快照调用会直接返回refs。

[execute_all_async](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/ray/base.py#L862)逐个提交worker RPC，收集refs；ONE_TO_ALL的collect原样返回列表，没有额外校验slot、内容或跨rank状态。三个worker方法正常完成均可返回None，因此一组正常返回值不是模型内容校验。

| 调用结果 | 可以据此判断与仍需核对的边界 |
|---|---|
| ray.get正常返回整个列表 | 这些远程调用已返回；不证明保存/恢复的tensor内容相同，缺key不操作仍可能正常返回 |
| 提交阶段本身抛异常 | 源码循环可能只提交了前缀worker；包装器没有等待已发refs、取消或回滚的补偿块 |
| ray.get因某任务失败而抛异常 | 不再进入collect；不能据此认定其他任务都已结束、都失败或已撤销 |

[Ray官方get文档](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html)说明，列表输入保持结果顺序，但任一关联任务报错时可立即抛异常，不等待其余任务完成；**结果顺序不是失败等待顺序**。2026-10-21所读文档标为Ray2.59.0，报告环境安装版本未知，verl只声明ray≥2.41.0；这项文档契约不能冒充reporter环境的原生复现。包装器本身没有补偿、全量异常汇总或本地get超时参数，也不排除外层系统另有超时/取消政策。

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250 --check-snapshot-rpc
```

可选模式要求Python≥3.10，因原始register检查含运行时类型union；实跑环境为Python3.12.13/NumPy2.0.2。6组控制执行原始注册元数据生成、参数复制/collect、方法绑定器及RayWorkerGroup提交/包装方法体：三个默认阻塞快照调用、显式非阻塞对照、get抛错、提交阶段抛错。Ray actor/ref/get、枚举、DataProto类型及TQ bridge为替身；get失败用“一个完成、一个失败、一个仍pending”的假设日程检查包装器不做补偿，不模拟Ray调度或测量真实任务状态。没有执行远端worker及其decorator inner、tensor快照、TQ转换、colocated/fused进程建立、真实异常收敛或GPU。

输入SHA256：decorator `718876c4c59e9f3cdb0339185be156aeb7b6d8eb42a2a6f919a3163961634ace`；ray/base `a61952c697b6c7f8220f7cb91043e514beadc9061007c1c140acb260e6aed8c5`；worker_group `3723ad4ba93afc8b715287a1e9f8754e4915af6fd93a44e57646beb28a8591ab`。本轮将V1来源与失败边界学习阶段收束；#8250仍需要真实配置、配对观测和受控对照，不以增加替身数量替代根因证据，也不据此提出自动恢复修复。

## 新版阶段 tracing 与证据关联

2026-10-21 读取已有 [verl #8315](https://github.com/verl-project/verl/pull/8315)，open、作者mengchengTang，head `4d6b3eb8606e24c1d70d648e92a1114129ec6c62`、目标base `619c397dc9917fed278b6e47c52a6620aef05466`、共同祖先 `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。这是新版观测入口，不是前文1876b06d报告环境已有的能力，也不是#8250修复。**阶段被记录，不能单独证明阶段成功、设备完成或参数内容一致。**

**2026-10-22版本更新**：PR仍open，head已到 `0cea139623cc8913d2241d21223df5682474041c`，目标base `f2923cbf154731918b1e36db8c567433f349fede`，共同祖先不变。相对前head仅新增65行原CPU测试、将update_weight实际lane改为字符串；tracking/两个server实现未变。当前真正PR为5文件138增4删，下方首轮“无新增测试”只描述旧head。新增实跑范围见[新版原测试与实际接线](#新版原测试与实际接线)。

### 区间究竟包住什么

[trace_state_decorator](https://github.com/verl-project/verl/blob/4d6b3eb8606e24c1d70d648e92a1114129ec6c62/verl/utils/tracking.py)只接受async方法；调用时先检查 `VERL_RL_INSIGHT_ENABLE == "1"`，关闭时直接await原方法，开启时计算lane并在trace_state上下文内await整个方法。它没有向context传递method参数、global_steps、request身份、权重hash或返回值。异常状态如何呈现还取决于实际rl-insight后端，不能只凭这个wrapper断言成功或失败标签完整。

| 新状态 | lane与覆盖范围 | 不能单独证明 |
|---|---|---|
| update_weight | 旧head为self.rank整数，当前为rank_加rank的字符串；位于register内层，包住worker方法 | controller提交/排队总时延、所有receiver安装完成、全rank一致 |
| vLLM/SGLang的wake_up、sleep | Ray actor name，与生成阶段使用相同actor lane | 真正发生了内存操作；方法内仍可提前返回或跳过某种mode |
| clear/release/resume_kv_cache | 同一server lane，包住原异步方法 | cache代次一致、实际释放字节数、模型参数已应用 |

例如非node_rank0、free_cache_engine关闭、部分colocated/standalone路径会在原方法内返回或跳过；装饰器在这些判断之前进入，所以短状态也可能只是no-op调用。vLLM release路径仍是sleep后只wake weights，SGLang则按tags调用memory occupation接口；同名阶段不意味着两个backend执行相同操作。

[update_weights](https://github.com/verl-project/verl/blob/4d6b3eb8606e24c1d70d648e92a1114129ec6c62/verl/workers/engine_workers.py)的非naive分支等待本rank的send_weights并可offload后返回；接收端应用/finalize/cache恢复仍须按实际checkpoint engine链追踪。naive分支的同步/barrier也来自原方法，不能把它推广到所有模式。该PR没有新增显存采样、参数manifest、逐token对照或为tracing专设的设备完成事件；“memory stages”是阶段名，不是显存字节测量。

### 标量 API 与开启成本

新log先检查实际模块的metric_log是否callable：可用时保留 `reward/mean` 这样的名字并传int(step)；否则只走旧metric_gauge，名字中的斜杠改下划线，且不双写。[rl-insight #206](https://github.com/verl-project/rl-insight/pull/206)在本轮API读取时已合并，但不能由此推断某台机器安装的包包含该API。版本号和API能力须分别记录。

这个能力判断不是容错协议：metric_log存在但调用出错时，代码没有自动改投gauge。开启trace后的lane求值、初始化和context进入也在原方法之前，wrapper没有用异常隔离来保证监控失败时照常执行。因此“11个函数体未改”不等于“开启监控完全不影响控制流”；本轮没有运行后端或验证开销/故障语义。

### 本轮核验与真机接续

首轮4d6b3eb8相对共同祖先的PR diff只有4文件、73增4删，没有新增配套测试。M1完整读取新增wrapper/标量分派、11处接线及对应原方法；Python3.9.6下4份完整源码AST解析和内存compile通过。另将两版方法的装饰器列表清空后比较AST，11个async方法的参数/函数体/返回注解与共同祖先相同，并核对每项新增一个装饰器。首轮没有exec这些源码或复跑CPU测试；这段仅保留旧版本静态核验范围，当前测试结果如下。

复查对象为tracking.py、engine_workers.py和两份async server。可用 `git diff cd58f550...4d6b3eb8 -- <四文件>` 定位变更；以 `git show <head>:<path>` 获取字节，ast.parse后compile但不exec。比较11方法时只清decorator_list，再对ast.dump(include_attributes=False)核相等，不移除或改写函数体。SHA256依次为：

- tracking.py：`c22afac06ddac8d6ddba4c3aa5a5ae4ad0d20170fddb4f6ac6756e2884f075f9`
- engine_workers.py：`2f6dd946a4989460ca64ae2f8de576fb278a32b529ac19b0bb80be3a2acdd79b`
- vllm_async_server.py：`b9694ff07d776395349159b68934e4e991a0e6027691af8f97feb690a21f236d`
- async_sglang_server.py：`57729bfebce2507fb271e28cc5c7cd4073afb8a5ecc31ec87b8815a4c8b9c354`

涉及真实权重/显存/生成的后续检查已补入[原M5补证条目](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另开任务或默认采用PR。保留分支/配置、rank/actor与step关联、异常/no-op标记及独立显存/安装证据；若缺获准监控环境或要新增数据外发，先记录条件，不启动服务或改变原报告版本。M1没有访问服务器，也没有新的连接失败实测。

### 新版原测试与实际接线

当前[原日志测试文件](https://github.com/verl-project/verl/blob/0cea139623cc8913d2241d21223df5682474041c/tests/utils/test_rl_insight_logger_on_cpu.py)共有12项，新增5项覆盖带step的metric_log，以及async decorator的动态lane、固定lane、关闭时不求lane、拒绝sync函数。Mock客户端把metric_log显式设None，再按测试启用MagicMock，避免“属性不存在但Mock自动生成callable”的假能力判断。

[三版本复查](../../experiments/verl-rl-insight/readme.md)在Python3.12.13/pytest8.4.2下，完整原RLInsightLogger/RolloutTraceConfig类经AST隔离执行，完整原测试模块保留fixture/断言；跳过的是包根、tracking其余后端与无关依赖导入。客户端为原测试MagicMock，没有运行rl-insight、Ray、服务或设备。

| 实现版本配当前原测试 | 结果 | 实际worker的lane（rank=2） |
|---|---|---|
| 祖先cd58f550 | 7过、5预期失败 | 尚无本decorator |
| 前head4d6b3eb8 | 12过 | 整数2 |
| 当前head0cea1396 | 12过 | 字符串rank_2 |

原测试中的Worker自行定义字符串lambda，因此即使生产接线仍传整数，logger suite也能全过。另直接执行两份engine_workers中**原lane表达式**得到上表差异，没有构造worker或执行update_weights。作者说明时间线查询需要非空字符串，当前接线与此约定相符；本机没有执行真实后端查询，不能把这项源码检查说成时间线显示已修好。

这区分了三层：helper正确处理传入lane、生产调用点传入正确身份、后端接收/关联/显示该身份。12项测试只验证前一层和Mock API调用。已有16协程lazy-init例也仅在同一事件循环内计数，不能外推跨线程/进程安全。旧“区间可含no-op、sender返回非receiver完成、开启监控可能改变异常路径”的边界保持不变；新增测试不使它变成真实同步或GPU验收。

当前tracking SHA256仍为上列`c22afac0…`；engine_workers为 `7166ff29e71eb101031ce88aa22feb32f0d960a5e795af0e8116f37aff757ec1`，原测试为 `874a695c536517254d666dc5b42a7309993a7f86d33e654ec7fbf66c6de58cf3`。详细命令、原类隔离方式、输入hash及首次网络审计误伤asyncio本地socketpair后的修正见[实验记录](../../experiments/verl-rl-insight/readme.md)。原M5条目只同步新版/验证范围，不另开任务或默认部署监控。

### Gauge 元数据与训练步存储

**日志标签带了step，不等于后端已按训练步存储。** 2026-10-24新出现的draft[#8357](https://github.com/verl-project/verl/pull/8357)由ETOgaosion实现，head `cd12877931cdd441adb731978a8b3785858dba27`，base/祖先 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`（ahead1/behind0），两文件9增2删。它保留旧flattened gauge名和值，增加 `rl_insight_step=int(step)`、`rl_insight_metric_name=str(key)` 两个标签；不新增exporter或存储。与已有#8315的step-aware API目标重叠，但不是相同实现，不应盲目叠加或另写竞争修复。

官方RL-Insight源码固定在 `64618dea13ae1977f04b355d94db96bb573eeb13`。[metric_gauge](https://github.com/verl-project/rl-insight/blob/64618dea13ae1977f04b355d94db96bb573eeb13/rl_insight/api.py)将参数写进gauge event的labels；[PushMonitorClient](https://github.com/verl-project/rl-insight/blob/64618dea13ae1977f04b355d94db96bb573eeb13/rl_insight/client/push/client.py)默认将gauge交给 `emit_store`，标签转字符串，不解释这两个新字段，也不据原名改写指标名。`metric_log`则走trace事件，携带原 `metric.name`、数值 `metric.value` 与整数 `global_step`。这正是API路径而非单纯字段拼写的差别。

本机Python3.12.13/packaging26.3，隔离执行三个版本的完整原RLInsightLogger类、SDK原四个emit/API函数、完整原PushMonitorClient/MonitorClient类和两个原dataclass；跳过包入口及构造初始化。两步7/8，各发 `actor/loss=0` 与 `actor_loss=1`，另一个object值被跳过：

| 生产端分支 | 四次sink边界调用 | 名称与步数 |
| --- | --- | --- |
| 固定base | store | 均为trainer.actor_loss，无原名/step标签 |
| #8357 gauge | store | 名称仍相同；原名两种、step字符串7/8，合计四个不同标签组 |
| #8315旧API分支 | store | 与base同样未携带原名/step |
| #8315 metric_log分支 | span | 名称trace.metric_log；attributes保留两个原名和整数global_step |

四组有限断言通过。#8315旧分支用显式 `metric_log=None` 控制能力判断，仍接同一固定新版SDK的gauge函数，不是实际旧版SDK兼容测试。SDK状态及client字段由夹具注入；唯一sink是捕获参数的边界替身，不是可运行exporter。没有执行logger/SDK init、原pytest suite、Prometheus registry、Ray、服务投递、存储查询或dashboard；未导入torch或设备模块。完整复现和输入hash留[当日日志](../../diary/2026-10-24.md)。

[Ray collector](https://github.com/verl-project/rl-insight/blob/64618dea13ae1977f04b355d94db96bb573eeb13/rl_insight/collector/ray_monitor_hub.py)把gauge labels原样交给[MetricRegistry.value](https://github.com/verl-project/rl-insight/blob/64618dea13ae1977f04b355d94db96bb573eeb13/rl_insight/utils/prometheus_utils.py)，后者用所有合并标签调用 `gauge.labels(...).set(value)`。因此若不消费/移除step，每个不同step是新标签组合，而非同一时序点的独立横轴；本机四组标签不是Prometheus容量/存储实测。固定SDK的[原生dashboard查询](https://github.com/verl-project/rl-insight/blob/64618dea13ae1977f04b355d94db96bb573eeb13/rl_insight/config/services/grafana/dashboards/verl/verl_tainer_v1_with_vllm_engine.json)查找metric_log trace、global_step和metric.value，新gauge标签不会自行变成这些字段。

评估采用时要先确定实际后端和消费协议：需要新API且可用时，核trace接收/保留与查询；需要旧gauge兼容路径时，核exporter如何恢复原名、解析step并避免将它留作时序维度。当前草案已明确该取舍；不把传参通过称为监控修复完成，不因本项部署监控、外发训练数据、追加GPU任务或改#8250/#8313。已有#8315的原12测试与同步边界结论保持原版本，不重复执行。

## 单 rollout worker 的 NIXL 拓扑

2026-10-21核对已有[#8316](https://github.com/verl-project/verl/pull/8316)，读取时open、rafiw，head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`、目标base `f2923cbf154731918b1e36db8c567433f349fede`、共同祖先 `619c397dc9917fed278b6e47c52a6620aef05466`。真正PR只改一行切片，未提交测试文件；正文勾选“新增单测”不能替代固定diff证据。这里验证的是**拓扑参数契约，不是权重同步完成**，也不说明#8250实际使用了此backend或故障由它造成。

**后续版本更新**：同日核对head已变为 `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`，仅比上述首轮head新增28行[原单测](https://github.com/verl-project/verl/blob/c2d800e71cdc4c7295b1e12e60bad82dd2fb637b/tests/checkpoint_engine/test_nixl_checkpoint_engine_on_cpu.py)，两份实现文件逐字未变，共同祖先及目标base不变。当前PR已含测试；上段“未提交测试”只描述旧固定head，不再作为当前覆盖判断。

### 单元素切片与链尾

[build_topology](https://github.com/verl-project/verl/blob/3a60b8710f8e1001110820f7a14cd0e71d9d5e07/verl/checkpoint_engine/nixl_checkpoint_engine.py#L289)将A个actor、R个rollout的metadata组织成链。有效通信rank为actor[0]的0及rollout的1到R，world_size=R+1；其余actor rank=-1、不连前后邻居。链关系为 `actor[0] → rollout[0] → … → rollout[R-1] → None`，不是所有A+R个worker都加入通信链。

rollout的next原式为 `metadata[-R + 1:] + [None]`。R=1时起点为0而不是“列表末尾之后”，因此返回整个metadata加None，长A+2；候选改为 `metadata[len(metadata)-R+1:] + [None]`，此时从len(metadata)切起，只得[None]。对满足metadata长度A+R且A/R均正的R≥2，两种起点表示同一位置。

### CPU证据与真实调用阶段

Python3.12.13，保留原classmethod、原manager两组长度断言及init_process_group的纯端点检查，A=1/2/4/8、R=1/2/3/4/8每版20组。基线16组过、4个R=1触发原 `rollout_kwargs[next_agent_metadata] must have length of 1`；候选20组全过，16个多rollout场景的完整输出与旧版相同。另按链关系检查rank/prev/next，没有新造NIXL agent或网络替身。方法、输入hash和[复查脚本](../../experiments/verl-nixl-topology/readme.md)保留；这是自建合成metadata控制，不是作者原始测试或完整manager执行。

[manager](https://github.com/verl-project/verl/blob/3a60b8710f8e1001110820f7a14cd0e71d9d5e07/verl/checkpoint_engine/base.py#L422)在真实调用中先prepare并收集actor/rollout metadata，再构造拓扑、检查每字段长度，最后才分派init_process_group。NIXL prepare还涉及buffer分配/注册和agent身份；这些本机均未执行。update_weights把各replica.workers展平成临时rollout worker group，所以R是该组world_size，不能直接等同replica数。

固定版本既有NIXL correctness用例为2 trainer/6 rollout，带temporary skip，含Ray/模型/原生backend；本机未运行，也不把device=cpu参数当成纯Python测试。列表长度及端点通过后，buffer、连接、传输、权重应用/finalize与恢复生成仍是独立完成点。若M5已有诊断确需单worker NIXL，可参考[原交接条目](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另排队、自动采用补丁或变动既有环境。

### 新增原单测的执行范围

新增测试只构造字符串metadata `["trainer", "rollout"]`，A=R=1，断言prev为[trainer]、next为[None]。复用原脚本的 `--upstream-case`，AST原样执行这一测试函数及其两个断言，类壳只承载原build_topology。Python3.9.6及3.12.13两环境均为基线1项预期AssertionError、新head 1项通过；不是扩写新模拟或把旧20组控制改称原测试。原测试SHA256 `699eb471d14bb946d47cd794416a23c04ee60a44b3ac206d2dafb8146e91541c`。

该文件顶层直接导入nixl_checkpoint_engine，而后者仍急切导入cupy、nixl、Ray、torch、ZMQ。本机跳过了整模块导入与pytest收集，没有安装/探测这些运行时。因此这里只确认原测试体对纯拓扑方法的回归能力，不能由文件名on_cpu或本次隔离通过推断无原生依赖的CPU CI可收集。新增测试仍不覆盖多worker、真实agent元数据或权重传输；原有M5交接边界不扩大。

## 组初始化的等待超时与任务退出

[verl #7045](https://github.com/verl-project/verl/pull/7045)是控制端等待的保护栏，不是 #6967 原生挂起的根因修复。2026-10-22读取open、作者aryanyadav0402，固定head `6f458f6d49076c647a42c00d240b5eaa7449fdb7`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际merge-base `220e9039902c6db56860e2afd659803dc34ec005`。真正diff三文件246增2删：新增97行helper、144行测试，以及manager接线。**当前head没有新增worker超时线程**；NCCL实现相对祖先逐字不变，不能混用旧worker侧版本的验证。

### 哪一段等待受到约束

[build_process_group](https://github.com/verl-project/verl/blob/6f458f6d49076c647a42c00d240b5eaa7449fdb7/verl/checkpoint_engine/base.py#L388)先prepare并取metadata，再构造/检查拓扑，最后分派actor和rollout两组init，拼接refs后交给新helper。其覆盖并不是整个weight sync的截止时间：

| 阶段 | 固定head的等待方式与边界 |
|---|---|
| abort请求、release KV | 在update_weights中先于建组；不受新helper计时 |
| prepare与metadata收集 | 仍是无显式timeout的ray.get |
| 构造拓扑、分派init RPC | 在进入helper之前；不受其ray.get超时约束 |
| 等actor与rollout的init结果 | 同一refs列表，一次ray.get，默认600秒 |
| 传权重、finalize、恢复KV/生成 | 仍为各自调用；不受这个组初始化timeout覆盖 |

`update_weights`每次非naive同步都调用建组；即便NCCL默认复用已存在group，init方法仍可能执行连接/barrier。因而新等待边界不只存在于第一次同步，也不由manager限定为NCCL backend；naive路径提前返回。这里是调用范围，不认证所有backend都做相同初始化或必定能在预算内结束。

### 停止等待不等于取消或回滚

[wait_for_group_init](https://github.com/verl-project/verl/blob/6f458f6d49076c647a42c00d240b5eaa7449fdb7/verl/checkpoint_engine/_group_init.py)只解析预算并调用 `ray.get(refs, timeout=...)`，将GetTimeoutError转成带原异常cause的CheckpointEngineInitError；其他异常不在该except中处理。没有cancel、kill、重试或补偿。当前 [Ray get文档](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html)将timeout定义为等待上限；[cancel是独立API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html)，行为还依任务/Actor执行模型而异。读取的文档为Ray2.59.0，不代表原报告或本项目已安装此版本。

超时异常沿manager当前方法向外传播，方法内没有finally继续finalize或恢复KV/生成；此前的abort/release动作不能据异常自动视为撤销。这说明必须区分“调用者获知失败”与“远程操作停止、资源释放、可重新开始”，不证明本次已观察到资源泄漏，也不提出在原group上重试的方案。PR选择由外层launcher结束作业；其正文报告的2×A100清理/新进程恢复针对旧head `da00403`，并明确当前控制端head复验待做，不能挪用为本版本验收。

### 参数合法性与检查时机

`_resolve_timeout_s`优先显式timeout_s，否则每次读`VERL_CKPT_ENGINE_INIT_TIMEOUT_S`，缺省600；转换成float后要求有限且大于0。原函数的12组纯CPU控制如下，Python3.12.13，仅AST隔离该函数和两常量，math/os为真实标准库：

| 输入 | 结果 |
|---|---|
| 不传参数、环境未设 | 600.0 |
| 不传参数、环境为12.5 | 12.5 |
| 显式5.0、环境为ten | 5.0，显式值优先 |
| 环境0 / −1 / inf / nan / ten | 5组均抛ValueError，消息指向环境变量 |
| 显式0 / −1 / inf / nan | 4组均抛ValueError，消息指向timeout_s |

拒绝发生在helper的ray.get之前，但manager先求值参数、分派两组init RPC才进入helper。因此“非法值不开始等待”不等于“非法配置在任何建组副作用前失败”。本轮只核源码求值顺序，没有分派RPC来制造该场景。

原15项测试中13项使用mock ray.get，另2项在本地Ray集群上以stand-in worker groups分别阻塞actor/rollout。作者的测试入口能覆盖接线，不是真NCCL挂起复现。本机独立CPU环境没有Ray，未补装、未运行这15项、完整helper或base模块导入；三原文件仅AST/内存compile通过。没有启动本地cluster、定时线程、通信、模型或设备，也不用更多Ray替身认证取消/恢复。

下面从固定Git对象复查相同12组参数控制（Python≥3.10；在含该对象的verl仓库中，不导入Ray）：

```bash
GIT_NO_LAZY_FETCH=1 python - <<'PY'
import ast, math, os, subprocess
from unittest.mock import patch
rev = "6f458f6d49076c647a42c00d240b5eaa7449fdb7"
tree = ast.parse(subprocess.check_output(["git", "show", rev + ":verl/checkpoint_engine/_group_init.py"]))
nodes = [n for n in tree.body if
         isinstance(n, ast.FunctionDef) and n.name == "_resolve_timeout_s" or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
             ("INIT_TIMEOUT_ENV", "DEFAULT_INIT_TIMEOUT_S") for t in n.targets)]
ns = {"math": math, "os": os}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<original-resolver>", "exec"), ns)
resolve, key = ns["_resolve_timeout_s"], ns["INIT_TIMEOUT_ENV"]
with patch.dict(os.environ):
    os.environ.pop(key, None)
    assert resolve(None) == 600.0
    os.environ[key] = "12.5"
    assert resolve(None) == 12.5
    os.environ[key] = "ten"
    assert resolve(5.0) == 5.0
    for is_env, values in ((True, ("0", "-1", "inf", "nan", "ten")),
                           (False, (0.0, -1.0, float("inf"), float("nan")))):
        for value in values:
            if is_env: os.environ[key] = value
            try: resolve(None if is_env else value)
            except ValueError as error: assert (key if is_env else "timeout_s") in str(error)
            else: raise AssertionError(value)
print("12 original pure-resolver controls passed; no Ray integration")
PY
```

SHA256：helper `ce401095e0d8d9c9711144dbe429aa62298faab431c57235a8f293edbb1724cb`；base.py `a86b084bdd68fe3dd6a91587147bd2e80c51b455863f6fe344a1623addf3c4a9`；原测试 `a5afc6997fd7473a25a4c1864d5444b0dab1a8b7ea0855abb75376628ed8eb78`；NCCL文件 `cc36e7a53579d60a76cb6ee811ca464dcddffdbc233a94c1af777f6f124a5eef`。当前版本的真机退出/释放/恢复条件仅补[原M5事项](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另排队或默认应用，不将此保护栏当#8250根因或训练恢复证明。

## Resume 返回值与扩展字段合并

FullyAsyncLLMServerClient会重建一个TokenOutput来累计分段生成结果；因此，即使只调用后端一次，也不能只复制预先认识的字段。[verl #8341](https://github.com/verl-project/verl/pull/8341)补上backend自定义extra_fields的传递。2026-10-23读取open、非draft，作者linyueqian；固定head `0cede26bf2c0d7433af0cba5c2c38ce278e6b07e`，base及实际祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，ahead1/behind0。两文件117增0删，其中客户端17行、CPU测试100行；不是权重同步或#8250训练修复。

### 推测解码计数器的跨尝试累加

2026-10-24接续：[#8341](https://github.com/verl-project/verl/pull/8341)当前head为`277b01de3709590c5c359ddd418ab8ab3c0957ca`，仍OPEN/non-draft，base仍`dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`。相对上面的旧head增加一个提交、两文件35增1删；只给三个推测解码计数器增加求和规则及两项原测试，不把旧字段保留结果当新增工作。

当前[原generate](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/workers/rollout/llm_server.py)将`spec_num_draft_tokens`、`spec_num_accepted_tokens`、`spec_num_verify_steps`列入`_SUMMED_EXTRA_FIELDS`：首次出现直接保留，后续出现转int后相加。其他通用键仍按最后一次出现覆盖，原top-k、cache与版本规则不变。一个generate内部的resume累加，与[tool agent](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/experimental/agent_loop/tool_agent_loop.py)在多个assistant turn间对已有三个键求和，是不同层的统计，不能混作同一个“回合”。

复用原隔离加载器，仅执行新增的两个未改异步测试体：单次返回`(draft,accepted,verify)=(10,6,4)`，旧／新head均通过；abort后第二次返回`(5,3,2)`，旧head的求和断言失败，新head得到`(15,9,6)`。两客户端完整原类只去掉tracing decorator，TokenOutput使用真实Pydantic2.14.1；后端generate和sleep仍是作者fixture，非真实引擎resume或RPC验收。没有重跑旧两个字段测试或其三项补充控制。

沿[生产者](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/workers/rollout/vllm_rollout/vllm_async_server.py)源码，vLLM从最终返回的`request_spec_decode_stats`取三计数，缺失时警告并省略；[SGLang](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/workers/rollout/sglang_rollout/async_sglang_server.py)映射meta_info中的三键，缺draft键还会回退到配置中的草稿数。合并器本身不认证后端计数是否同单位、仅覆盖本次请求而非累计历史；实际安装版本的返回语义仍须核对，不能从统一名字反推等价。

[V1消费端](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/trainer/ppo/v1/trainer_base.py)从TQ的extra_fields取计数，再交给[原compute_spec_decode_metrics](https://github.com/verl-project/verl/blob/277b01de3709590c5c359ddd418ab8ab3c0957ca/verl/trainer/ppo/ray_trainer.py)：先逐request算`accepted/draft`和`1+accepted/verify`，再等权平均，不是整批先求总数再取比。直接调用该完整原函数，作者这组旧末段`(5,3,2)`与新总数`(15,9,6)`均返回接受率0.6、接受长度2.5；**率指标没变，不能证明工作量计数未丢失。** 这不是速度提升或真实接受率测量。

本机Python3.12.13、NumPy2.5.4、pytest9.1.1，复用现成环境，无安装。新客户端SHA256 `145c56ffaa2ea738abdfc303d2356f94361d829939bb29b82800e5fa59ab6f72`，新测试`90ea97cded9f54f195eebcaaa199323a4eb473a6af001c8d0cf03b3681e59331`；[完整本轮复现](../../diary/2026-10-24.md)保留一次性脚本，不改下方锁定旧head的历史脚本。未原生导入verl、pytest收集、运行后端/AgentLoop/TQ、模型或设备，未复跑作者整套49项。真机条件更新[M5原参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用或排新实验。

### 旧 head 的字段保留与序列对齐

[原generate实现](https://github.com/verl-project/verl/blob/0cede26bf2c0d7433af0cba5c2c38ce278e6b07e/verl/workers/rollout/llm_server.py)新增逐键赋值，但排除已有专门合并规则的七个extra_fields键。几类值不能用同一种“取最新”概括：

| 字段 | 分段返回时的规则 |
|---|---|
| 其他extra_fields，如text、multi_modal_output | 最后一次**出现该键**的值覆盖；后续缺键保留旧值，显式None也会覆盖 |
| response_topk_ids、response_topk_log_probs、response_support_complete | 专门按生成段拼接；有top-k而缺support证明的段补False，不用通用覆盖 |
| num_cached_tokens | 保留首次非None报告；不是各次cache hit求和，首段缺失时可能来自后段 |
| global_steps、min_global_steps、max_global_steps | 专门维护末段值、首次非None值、末段值；名字不意味着对任意乱序版本做数值min/max |
| TokenOutput的token_ids、log_probs、routed_experts、num_preempted | 原有字段各自累计；路由后续只追加新token对应后缀，抢占数求和，不走extra_fields循环 |

通用循环直接赋值，不递归合并或深拷贝。例如首段`nested={a:1}`、次段`nested={b:2}`得到后者，不会自动合成两个键。首段text被保留也不说明它描述整段累计token；若某backend返回增量音频、逐token标记或嵌套payload，producer与consumer仍须约定其作用范围。保留字段解决“丢失”，不替任意字段实现序列拼接、完整性证明或RPC序列化协议。

### 旧 head 原测试体的 CPU 对照

[复查脚本](../../experiments/scripts/verify_verl_extra_fields.py)从固定Git对象提取完整原TokenOutput、normalize_token_ids和两客户端类，仅去掉generate的tracing decorator，包壳绕过原生verl导入。使用真实Pydantic、NumPy和pytest.MonkeyPatch，执行[作者新增的两个未改测试体](https://github.com/verl-project/verl/blob/0cede26bf2c0d7433af0cba5c2c38ce278e6b07e/tests/workers/rollout/test_llm_server_extra_fields_on_cpu.py)；base.generate和sleep替身沿用作者fixture，不连接后端。

- 单次返回：基线在读取text时KeyError；head保留text及2×3 NumPy多模态值，global_steps为3。
- aborted后resume：基线同样丢text；head得到token `[101,102,103]`，text为second，only_first保留，top-k三行拼接，cache为2，版本3→4。
- 三组补充控制：缺键与显式None不同；nested整项替换且本地payload保留对象引用；同一客户端的独立下一次请求不继承自定义字段。两次运行结果一致，后一次将下一请求控制强化为复用同一客户端。

环境为Python3.12.13、NumPy2.0.2、pytest8.4.2、Pydantic2.14.1。Pydantic及依赖装在独立work target，不改原venv。没有torch/Ray导入、原生pytest收集或AgentLoop/引擎运行；作者正文报告的四文件49项通过不是本机结果。这个对照检验字段合并，不认证实际多模态生成、远程payload传输或训练效果。

```bash
# 已有Python环境需含numpy/pytest；deps为独立Pydantic2安装目录。
python -B experiments/scripts/verify_verl_extra_fields.py /path/to/verl-git --deps /path/to/pydantic-target
```

SHA256：head客户端 `1ef9a84c10e11f7f15d89e79450f5d578b9585acf8b974ad5b96b1459da3febe`，base客户端 `a31a51faf3bc7a877628eff52917cb442b1d3e17ae5249a7f3cdbc9282a45e96`，原测试 `a2cc81ea48d96176c0d10f66c686935ceadd83796ba27ac4d676ba6246495aab`；脚本逐次打印schema/normalizer哈希供复核。真实backend→客户端→AgentLoop消费的条件补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另排队、自动采用候选或重开#8250因果判断。
