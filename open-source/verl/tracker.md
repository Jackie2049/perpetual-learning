# verl 贡献追踪

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** verl 贡献追踪。[PR #8286](https://github.com/verl-project/verl/pull/8286) · [候选目录](readme.md)

**【类型】** 候选追踪

**【价值评估】** 8 / 10 — 记录成功、重复覆盖和放弃原因，避免反复投入已失效问题。

**【进展】** [我方 #8313](https://github.com/verl-project/verl/pull/8313)由M5单一负责：①10-24 22:32、②10-25 12:00 回复均按用户逐字文本发出，head 71fd3562（用户裁决：middle 对齐 postprocess_data、cap=1 保留首 token、测试改名回退；CPU 18 passed + 10-25 01:08 GPU e2e 2/2 PASS + 负例 fail-loudly PASS），仍 OPEN/MERGEABLE 等维护者 approve（requested reviewers wuxibin89+dajiaohuang，10-22 后无新审查）；③trainer 级问题讲解中、④草稿 v3 待用户逐字审定，见[原草稿](draft-8313-review-reply.md)。另核[异步恢复库存](#异步恢复库存与预热容量)：#7579非原历史回放，原9 warmup测试体仅Mock接线，不另实现。另核[选定token概率](#选定-token-概率提案的范围收窄)：#7712已撤回组合器，恒等式CPU数学对照非kernel验收，不另实现。另核[V1 idle统计](#v1-异步空闲指标的统计范围)：等待／布尔决策／路由请求数非GPU利用率，不另实现。另核[VeOmni后端RFC](#veomni-端到端低精度路线)：作者已开#1360，配置迁移与职责待反馈，不另实现。另核[S3层级](#s3-下载与完整-checkpoint-生命周期)：下载测试不等持久化验收，不另写后端。另核[激活卸载入口](#激活卸载的辅助输出入口)：已有#8277，原五CPU测试体head通过，recording图顺序非真机验收，不另实现。另核[多轮旧配置](#多轮旧配置的接线或弃用)：活跃AgentLoop未消费两键，旧strict差异只告警，等待维护者选择接线或弃用，不另认领。2026-10-25核[system-only输入契约](#仅-system-消息的契约裁定)：当前VL CT仍有wrapper调用点，但维护者要求初始prompt包含user，不另推进通用支持。[原生MTP](#megatron-迁移的已有原型与验收边界)、[FSDP精度](#fsdp-精度默认的两份候选)等历史验证各限原版本与范围，不另竞争实现或对外发布。 另核[OPD／GKD名称映射](#opd-与-gkd-损失的已有说明)：#7975已有draft #7983，不另认领；当前tail-bucket模式也非recipe JSD等价。 另核[padding序列统计](#padding-行与序列统计的已有修复)：#7785原一CPU回归候选通过，统计修复非IS／loss或质量验收，不另实现。 另核[逐轮奖励路线](#逐轮奖励的基线与已有过程接口)：#7244已有作者，draft #8305不claim同一LOO估计器，不另实现。 另核[Ray版本与认证](#ray-版本固定的已有候选)：#7599已有#7600，离线声明／lock核对不作安装或安全验收，不另升级。 另核[GSPO裁剪指标](#gspo-裁剪指标的讨论与边界)：局部token比例／MEAN归约／最终梯度分开，#7356已有社区解释，不重开旧RFC或调参。 另核[自适应组大小](#自适应组大小的接入范围)：#8077原作者待接口方向，预设n非分轮停止，源码接线非recipe或训练验收，不另实现。 另核[teacher温度](#teacher-温度的已有修复与精度边界)：#7769已有#8253，CPU钩子精度控制非vLLM／训练验收，不另实现。 另核[SFT重启保留](#sft-重启保留的已有修复与路径条件)：#7396原22项有路径条件，别名失败为保守保留，反馈不发布、不另修复。 另核[批均衡标量](#批均衡的标量转换已有候选)：#7281原两CPU测试体通过，FP32分区语义与局部性能分开，不另实现。 另核[奖励验证标志](#异步奖励验证标志的已有修复)：#7217原两参数CPU测试体通过，单项meta_info非全批透传，不另实现。 另核[LoRA导出](#lora-导出与可选融合的已有候选)：#7341原8配置／早返回用例通过，融合调用Mock非数值验收，不另实现。 另核[工具清理](#工具重试需求与清理失败的边界)：#6978维护者指向uni-agent，原8用例和1双故障仅CPU替身，不另实现通用重试。 另核[DisRM契约](#disrm-返回契约的已有修复)：#7608原三回归过，局部部分写入／再次跳过非真实恢复，不另实现。 另核[异步并发预算](#异步并发预算的已有配置)：#6946旧需求在主线已有倍率，组任务非逐副本序列配额，不另实现；源码核对非吞吐验收。 另核[checkpoint覆盖](#checkpoint-配置覆盖的两个已有候选)：#7388／#7426范围不同，真实Hydra缺键与对象默认值分开，非后端保存验收，不另实现。 另核[Rollout调度guard](#rollout-调度诊断的已有候选)：#7382原四状态替身测试体通过，诊断非资源预留／就绪，全启动硬期限未验收，不另实现。 另核[整条奖励落点](#整条奖励落点的已有候选)：#8368原两回归base失败／head通过，sum同而return不同，非梯度／训练验收，不另实现。 另核[VeOmni optimizer](#veomni-optimizer-覆盖的已有候选)：draft #8370原五接线用例通过，独立CPU单步非卸载性能验收；M5仅条件参考，不另实现。 另核[估计器附加字段](#估计器附加字段的已有候选)：#7473原三CPU测试体候选通过，引用透传非有效性过滤，V1七字段未改；不另实现或新排训练。 另核[NPU卸载](#npu-参数卸载的重叠候选)：新#8372／3／4重叠，原7CPU参数例候选过非NPU内存验收；显式pin策略分开，条件交M5，不另实现。 另核[平方概率分块](#平方概率分块的已有候选)：#8365原16CPU参数项过不覆盖空行反向，已有两反馈复现；非GPU峰值／训练验收，不另实现。

**【大白话讲明白】**

这张表区分哪些问题修好了、哪些只是草稿、哪些已有人在做。它也保留查重失误，提醒后续不能把“没查到”当成“没人修”。

**【What–Why–How】**

- **What：** verl 候选从发现、验证到发布或放弃的状态。
- **Why：** 社区响应快，过期结果会误导投入；本地验证成功不代表仍有贡献空间。
- **How：** 按证据更新状态，将已合并、他人覆盖、未验证和待用户决定分别记录。
<!-- contribution-summary:end -->

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/verl-project/verl （2026-10 核验：仓库已从 volcengine/verl 迁移至 verl-project/verl，原链接 301 跳转）

## 平方概率分块的已有候选

2026-10-26读OPEN/non-draft [#8365](https://github.com/verl-project/verl/pull/8365)全文、两文件diff、原三测试定义及一普通bot评论／一个COMMENTED bot review／两行内；已有作者，不另认领。head0ed8b370、父版／目标base／祖先及main3c6d9b43，ahead1／behind0，两文件31增2删；查重已命中，不把bot评语当人工验收。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#平方概率分块与空输入梯度)：原16CPU参数项候选过、父版既有7过，空行反向／负chunk两条已有反馈仍有界复现；未读取未初始化值，不改候选或另发反馈。2048行非整体内存上限，作者4090峰值／配合#8364三步训练非本机结果；真机仅补原M5条件参考，不拼接接线／新增训练或写社区。

## NPU 参数卸载的重叠候选

2026-10-26读新OPEN/non-draft [#8372](https://github.com/verl-project/verl/pull/8372)／[#8373](https://github.com/verl-project/verl/pull/8373)／[#8374](https://github.com/verl-project/verl/pull/8374)全文、diff、测试与全部评论。同一CodeRabbit app，普通评论仅bot／CLA未签，reviews及行内空，作者自述未运行，不作人工验收。head648423c5／340680d3／8b7aa26c，共同父版与目标basee009360c，相对main3c6d9b43均ahead4／behind0、祖先main；三个中间LoRA sleep提交未审核，不当单提交补丁整体应用。

见[原FSDP专题](../../notebook/frameworks/verl-fsdp2-source-deep-reading.md#npu-卸载的复制策略与固定内存)：两份生产模块AST同，第三仅模型；#8374原7CPU参数项head过／base4过3预期失败，仅原Mock接线。固定torch-npu pin_out源码非NPU运行；FSDP2显式CPUOffloadPolicy另有[#8057](https://github.com/verl-project/verl/pull/8057)，不混同或拼接。真机缺口条件交M5，不探测、另实现／排训练或写社区。

## 估计器附加字段的已有候选

2026-10-26读OPEN/non-draft [#7473](https://github.com/verl-project/verl/pull/7473)完整正文、两文件diff及三个原测试；关键词检索已命中，不另实现。headc9e1e915、父版／祖先b256ebf8，相对main3c6d9b43 ahead1／behind239，两文件100增5删；API目标base3f9d4c0d另计。唯一普通评论仅CLA机器人，reviews／行内空，不当批准。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#估计器附加字段与样本有效性)：候选三原CPU测试体过，父版／main字段读取断言失败、固定签名调用过；签名helper仅候选存在，不伪造基线通过。原registry、容器和调用函数隔离执行，额外引用控制非有效性过滤；V1仍七字段读取，仅源码非TQ／训练。不替作者实现V1选择或mask算法、不新排M5或写社区。

## VeOmni optimizer 覆盖的已有候选

2026-10-26读新OPEN/draft [#8370](https://github.com/verl-project/verl/pull/8370)完整正文、七文件diff及原测试；关键词查重只命中本候选，不另实现。head44c97bcd，base／父版／祖先及main3c6d9b43，ahead1／behind0、七文件120增1删；唯一普通评论是bot跳过draft审查，reviews／行内空，不当批准。#8362／#8310已有研究，本轮筛选后未重复验证。

见[原VeOmni专题](../../notebook/frameworks/verl-veomni-opd-source-reading.md#optimizer-覆盖与-cpu-offload-默认)：原五接线用例base1过4预期失败、head5过，原builder记录器非VeOmni；特定offload flag下fused默认True、显式False保留。锁定VeOmni源码的AdamW分支令foreach=not fused，仅阅读。另真实CPU257元素单步参数／状态差0，不测内存／时间，不挪用AMD堆叠表的首步或running项。新增checkpoint接线只源码，真机条件补M5原参考，不新排训练／部署或写社区。

## 整条奖励落点的已有候选

2026-10-26读新[#8367](https://github.com/verl-project/verl/issues/8367)全文／0评论及OPEN/non-draft [#8368](https://github.com/verl-project/verl/pull/8368)正文、全部两文件diff；编号与rm_scores关键词查重已命中，不另实现。固定head9cf35a1d、base／祖先／当轮main3c6d9b43，ahead2／behind0，两文件71增5删；两普通评论为CLA／bot，reviews及行内为空，不当批准。

见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#整条奖励的落点与回报传播)：原两个新增CPU测试体base两断言失败／head两通过，分别核as_dict与补齐batch落点及原回退。六token单一数学对照原GAE／RF++回报首位0→1，奖励总和均1；有效位标准化优势仍0，非梯度／学习收益。原类／容器和方法隔离执行，不含作者全21／42项、真实SWE／tokenizer或训练；RS后改mask及colocated RM仍排除，不另写方案或新排M5。

## Rollout 调度诊断的已有候选

2026-10-26读[#7381](https://github.com/verl-project/verl/issues/7381)／[#7383](https://github.com/verl-project/verl/issues/7383)全文，各0评论；已有OPEN/non-draft [#7382](https://github.com/verl-project/verl/pull/7382)，head8ebc3558、父版／祖先dc709afb，相对main3c6d9b43 ahead1／behind5，五文件131增3删。三普通评论为作者两次跟进／bot总结，唯一review COMMENTED、一个行内评论含两项技术担忧，不当批准。不另造调度guard、PG预留或拓扑限制。

见[原生命周期专题](../../notebook/frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md#rollout-调度等待与引擎就绪)：原四协程测试体候选通过，get_actor仍原替身，三后端接线只AST；缺状态／非pending／引擎就绪分开，300秒非全启动硬期限。固定Ray2.41官方同步REST／30秒核对不是集群或所有关闭dashboard配置的失败验收；原报告缺required_resources，不借测试GPU=1证明一卡拓扑根因。无新算力、专项或上游动作。

## Checkpoint 配置覆盖的两个已有候选

2026-10-26读[#7387](https://github.com/verl-project/verl/issues/7387)全文／0评论及两个OPEN/non-draft候选完整正文／评论：[#7388](https://github.com/verl-project/verl/pull/7388) head0e540bc3仅actor一文件3增；[#7426](https://github.com/verl-project/verl/pull/7426) head51f8f311含actor／critic／SFT、四生成YAML和原反射测试，八文件112增。相对main3c6d9b43分别ahead1／behind267与ahead1／behind203，共同祖先d6f8a400／ea532913，不把API目标base3f9d4c0d当父版。#7388有作者协调重叠与CLA两评论，#7426仅CLA一评论，reviews／行内均空；不当批准或无人负责。

见[原检查点专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#配置字段可覆盖与对象默认值)：真实Hydra四版三入口／五显式节点，自写控制显示基础／main三代表覆盖拒绝、仅actor候选一项接受、完整候选三项接受；原配置对象默认值仍可补齐。#7426与父版五处asdict值同，不是字节／训练或文件存储等价。旧Mcore子类与主线基础类型分开，原反射pytest及生成检查未跑，不另造修复、排M5、推候选或写社区。

## 异步并发预算的已有配置

2026-10-26读[#6946](https://github.com/verl-project/verl/issues/6946)全文／0评论及[#6306](https://github.com/verl-project/verl/issues/6306)全文／两评论。前者仍OPEN，但固定main3c6d9b43已在两份fully_async配置声明async_training.concurrent_samples_per_replica=16，构造器确实读取，不再按旧提案认领配置功能。编号all-state PR检索空不能推出无人处理；关键词OPEN [#7010](https://github.com/verl-project/verl/pull/7010)是锁作用域修复、非配置补丁，其作者测试未在本轮运行。

见[原异步专题](../../notebook/frameworks/verl-fully-async-policy-reading.md#样本任务并发与推理序列数)：并发限制全局整组任务，每组按n展开；受max_required_samples截断，不等逐副本严格序列配额或KV峰值。仅四完整模块AST／compile、两YAML默认值核对，非Hydra／框架或训练验收。旧2.29倍为作者特定NPU配置报告；#6306回复API关联NONE且语气不确定，不归因为维护者背书。不另实现、调参或新增M5队列，不写社区。

## DisRM 返回契约的已有修复

2026-10-26读[#7368](https://github.com/verl-project/verl/issues/7368)全文／0评论，查重已有amanyagami的OPEN/non-draft [#7608](https://github.com/verl-project/verl/pull/7608)，head e1e75377、父版／祖先ea532913，相对当前main3c6d9b43 ahead1／behind203，三文件112增2删；两个普通评论仅作者催CI，reviews／行内空。旧[#7389](https://github.com/verl-project/verl/pull/7389) CLOSED／未合并，不当修复已落地或据此另认领。

见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#disrm-返回契约与部分写入)：原两生产者参数＋一消费者回归候选通过，祖先／main两断言失败／精确缺键异常；三版消费者AST同。单一同对象再次调用控制说明旧版已写reward_score后失败，再调用因已评分跳过，非框架实际重试／恢复证据。原HTTP／预处理／Ray替身、真实容器，非模型／服务／完整训练；不改严格索引、应用候选或新排M5／改专项。

## 工具重试需求与清理失败的边界

2026-10-26读[#6978](https://github.com/verl-project/verl/issues/6978)全文／唯一维护者评论：复杂sandbox应参考[uni-agent](https://github.com/verl-project/uni-agent)，不等于接受ToolAgentLoop通用重试。编号all-state PR为空，关键词开放PR只见无关SFT，不据此认领；本轮不写重试实现／配置或发布反馈。

见[原多轮专题](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#工具异常与资源释放的边界)：固定main3c6d9b43原8个unittest两次通过，另单一双故障中create／execute／release各一次，释放OSError阻止原错误文本返回；不是所有异常都被吞成response。真实Pydantic、原FakeTool／本地记录器，非sandbox、FunctionTool分支／并发取消或恢复验收。uni-agent仅README定位，不借其架构声明安全重试／恰好一次；无新M5队列／专项或上游动作。

## LoRA 导出与可选融合的已有候选

2026-10-25读[#7285](https://github.com/verl-project/verl/issues/7285)全文／0评论，报告者明确当时没有做hash比较，标题byte-identical不当实测。已有OPEN/non-draft [#7341](https://github.com/verl-project/verl/pull/7341)，headfb304fc1、父版／祖先b75f0d06，相对main3c6d9b43 ahead1／behind280；两文件366增1删，唯一评论CLA、reviews／行内评论空。不另实现或发布。

见[原检查点专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#分片重组与-lora-融合不是同一步)：默认仍分离adapter，可选融合成功路径使用PEFT，但固定adapter目录删除无finally，不当异常事务。原8项配置／无adapter测试体通过、5项未运行；分离方法三版AST相同。融合原测试Mock并不认证真实增量，PEFT／accelerate未装，无文件导出／模型数值或GPU验收，不新排M5或改专项。

## 异步奖励验证标志的已有修复

2026-10-25读[#7121](https://github.com/verl-project/verl/issues/7121)全文／唯一协调评论；编号及关键词检索已有YAO-001的OPEN/non-draft [#7217](https://github.com/verl-project/verl/pull/7217)，四普通评论、唯一review为COMMENTED，行内评论空，不当批准或合并。head e8508932是merge，首父7fcdd5e1已有修复，第二父／实际祖先2781c1e4；相对当前main3c6d9b43 ahead2／behind298，真正三文件67增3删，不另实现／催审。

见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#异步奖励的验证标志与元数据范围)：原两参数CPU测试体候选通过，祖先／main因新增必需参数预期失败；另按各版签名调用才核到空meta_info。validate单项元数据与kwargs逐行字段分开，非父批全透传。真实DataProto／TensorDict、原假远程句柄，仅纯文本原方法隔离执行；两postprocess接线仅AST，不认证作者25+1、真实Ray／TQ／服务或训练，不新排M5或写上游。

## 批均衡的标量转换已有候选

2026-10-25读[#7279](https://github.com/verl-project/verl/issues/7279)全文／唯一评论，编号与关键词查重命中OPEN/non-draft [#7281](https://github.com/verl-project/verl/pull/7281)，head05ef5dcc、父版／与main共同祖先87c68d21，相对当前main3c6d9b43 ahead1／behind297；五文件diff、正文完整读取，普通／行内评论及reviews均空。不另提转换补丁，相关[#8137缓存spread](https://github.com/verl-project/verl/pull/8137)／[#7097动态批复用](https://github.com/verl-project/verl/pull/7097)也不当无人负责。

见[原GRPO均衡专题](../../notebook/frameworks/verl-grpo-source-reading.md#python-分区的标量类型与性能范围)：原两CPU测试体两次通过，复用原FP32输入三版Tensor分区一致，裸tolist却改变分区；NumPy首项累加须保源dtype。作者四卡SFT报告局部约61ms节省、整步无提速，不能照搬微基准倍率。仅原函数提取／五文件语法，不认证完整8／9测试、NumPy1.x、trainer／DP或性能；无候选应用、M5新队列／专项改动或社区写入。

## SFT 重启保留的已有修复与路径条件

2026-10-25读[#7395](https://github.com/verl-project/verl/issues/7395)全文／0评论、原作者ai-yang的OPEN/non-draft [#7396](https://github.com/verl-project/verl/pull/7396)完整正文／七文件目录、全部生产diff及两原测试文件，comments/reviews空。head8a1a031a、base3f9d4c0d、父版本／比较祖先09ac3725，ahead1/behind149，七文件375增1删；不另实现或把作者4×3090／额外27项当本机结果。

见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#重启后的保留状态与路径身份)：已加载旧路径延后到新保存发布后才考虑删除，不扫描其他历史目录。原22项在统一scratch路径下全过，父版与main的同一重启回归各预期失败；本机系统别名拼写则head21过1失败，因abspath未统一/var与/private/var，属于保守保留而非误删／Linux训练证据。反馈只存本地：明确路径身份契约与原测试主机条件，不机械改realpath扩大删除边界。脚本／JSON可复现，无真实FSDP／RPC／HDFS／恢复验收；仅原M5条件参考，无新增算力、清理或社区动作。

## Teacher 温度的已有修复与精度边界

2026-10-25读[#7769](https://github.com/verl-project/verl/issues/7769)全文／0评论；编号及关键词查到yph22的OPEN/non-draft [#8253](https://github.com/verl-project/verl/pull/8253)，head2920c014、base/祖先5b550b0c，八文件226增24删。已读完整PR正文、选定接线diff与两个原测试文件；长机器人评论输出截断，另核review元数据只有COMMENTED，不视为批准或完整review。已有作者，不另实现、回帖或接管测试发布。

见[原OPD专题](../../notebook/frameworks/verl-veomni-opd-source-reading.md#teacher-采样温度与打分温度)：请求temperature仍1，vLLM teacher另在归一化前缩放logits，其他后端不一并获得该能力。原默认3参数例在main／旧候选／head均过，作者固定logits的局部CPU控制显示旧FP16溢出、BF16误差784，head本例均有限且误差0；不是vLLM Sampler、配置／角色隔离、原30项或训练验收。脚本及JSON随个人仓库，仅补原M5条件参考，不启部署／应用候选、新排算力或改专项。

## 自适应组大小的接入范围

2026-10-25读[C-GRPO #8077](https://github.com/verl-project/verl/issues/8077)全文／0评论，仍OPEN；作者希望作为recipe，明确等待位置与多轮生成接口反馈后再写实现。编号8077 in:body及精确名称的开放PR搜索空，不等无负责人；宽关键词命中的DAPO、运行时等方案不据名称认作同题。本轮未审论文／参考实现，不转述其收益为本项目结论，也不提前实现、回复或另开RFC。

见[原数据流专题](../../notebook/frameworks/verl-grpo-data-flow-reading.md#自适应组大小与逐轮生成)：main3c6d9b43的loop GRPO按实际UID分组，V1支持逐prompt预设__rollout_n__，但一次启动该n个session并在全部settle后结束组；不是答案驱动的分轮追加。ReplayBuffer取样单位是prompt，训练倍数／actor mini-batch仍以配置n计，DAPO是终态同reward过滤，不是提前停止判据。六完整模块仅AST／compile，无原测试、校准器／TQ／模型或训练；不据此判主线bug、C-GRPO有收益或已受支持。等待作者／维护者方向，无新增M5队列或专项动作。

## GSPO 裁剪指标的讨论与边界

2026-10-25只读[#7356](https://github.com/verl-project/verl/issues/7356)全文及[唯一社区评论](https://github.com/verl-project/verl/issues/7356#issuecomment-5755317393)，评论者ot4562-glitch不当维护者裁定。作者在Qwen3.5 fully async GRPO／GSPO上报告clipfrac约0.4、reward改善慢；未给精确源码SHA，不将当前主线等同其安装。开放PR按7356 in:body为空，但已有解释不等于无人处理或需要另开修复；未使用／推荐评论中的第三方诊断工具。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#gspo-的-clipfrac-分母与梯度范围)：固定main3c6d9b43中单次GSPO clipfrac按有效token，包装器标为MEAN，最终日志须另核微批／DP归约；原Metric类标量1／0得.5，假定token数3／1池化才得.75。此控制不执行GSPO、worker、真实DP或训练。clip标志不含后乘IS权重及独立KL／entropy梯度，lower固定0非无下界裁剪；配方0.2／0.28是真GSPO但不是普适建议。五模块AST／compile、shell仅bash -n，不据0.4归因staleness／调参或判训练健康，不重开withdrawn/not-pursued的旧clip RFC；仅补原M5条件参考，不排新实验、不发布社区回复。

## Ray 版本固定的已有候选

2026-10-25读[#7599](https://github.com/verl-project/verl/issues/7599)全文／0评论，编号与关键词查重均有原作者jameswong-hai的OPEN/non-draft [#7600](https://github.com/verl-project/verl/pull/7600)，head821cb4f7、目标base3f9d4c0d／祖先ea532913，ahead1/behind87。完整正文、34文件目录（非默认30项）、选定六diff及唯一CLA评论已读，reviews空；不代作者签CLA／催CI，也不另写版本固定方案。

见[原Ray专题](../../notebook/fundamentals/ray-framework.md#版本固定与控制面安全)：固定主线3c6d9b43与候选选定七文件仅TOML／AST／Requirement解析，主线lock条目2.55.1，不能把宽范围或某旧镜像版本当实际运行版本。候选2.58pin不启认证；固定Ray2.58文档与latest2.59本地默认有版本边界，认证不代替隔离。没有resolver／Ray导入、wheel／镜像构建、凭据／服务／网络或GPU验证；作者报告与运行安全分开，无新M5队列、环境升级、专项／社区写入。

## 逐轮奖励的基线与已有过程接口

2026-10-25只读[#7244](https://github.com/verl-project/verl/issues/7244)全文／唯一评论。作者lijuliana已有fork实现，GBX-Max1220另询问反馈缺失／延迟扩展。开放PR按7244 in:body为空，但per-turn关键词发现luzy99的OPEN/draft [#8305](https://github.com/verl-project/verl/pull/8305)，head07d3a6d6、目标basecd58f550。已读正文、19文件目录、完整process_reward模块及三处接线diff、唯一机器人评论，reviews空；不说全patch已审、无人实现或机器人认可，不替作者触发评审。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#逐轮奖励与停止动作的基线)：主线GRPO行求和与V1最终输出广播、RFC的LOO return-to-go、draft的后置actor优势塑形各有不同契约。#8305保留outcome rewards／critic returns，不claim解决该RFC；作者CPU报告与GPU待验非本项目证据。三模块仅AST／compile及一个精确二元代数例，没有运行原算法／候选、测试、MDP／反馈研究或训练；不据此判draft有错或已无偏，不新排M5任务／另写实现或社区动作。

## Padding 行与序列统计的已有修复

2026-10-25读[#7771](https://github.com/verl-project/verl/issues/7771)全文／0评论、已有rrrxxx0510的OPEN/non-draft [#7785](https://github.com/verl-project/verl/pull/7785)正文／全部两文件diff和三评论。head7ffe2fa4、目标base3f9d4c0d、共同祖先d040717b，ahead1/behind49，祖先diff仅46增3删；评论里的31项／报告者52项不当本项目范围，也不代作者催CI或回复。

接续[此前仅查重条目](#近期候选覆盖核对)，见[原概率专题](../../notebook/frameworks/verl-async-logprob-provenance.md#padding-行与序列指标的分母)：原新增一回归体接三版完整metric与masked函数CPU两次一致，祖先／main失败、候选通过。三空行稀释序列指标，token三项在本例不变；另一个聚合解释例不算原测试或新矩阵。调用链仅源码，无padding构造／TQ／IS／loss或模型，不能据此称质量或全trainer已修；仅补M5原观测参考，无新队列／专项或社区写入。

## OPD 与 GKD 损失的已有说明

2026-10-25读[#7975](https://github.com/verl-project/verl/issues/7975)全文／0评论与关联[recipe #151](https://github.com/verl-project/verl-recipe/issues/151)全文／0评论；编号查重命中Liu-KM的OPEN/draft [#7983](https://github.com/verl-project/verl/pull/7983)，head28879f83、base8e03c039。完整正文、单文档diff、一条CLA评论已读，reviews空；不是无人实现，也不将机器人／作者构建报告当人工审定或已发布接口。

见[原OPD专题](../../notebook/frameworks/verl-veomni-opd-source-reading.md#主线与-gkd-的损失名称和支持域)：主线kl/k1为采样token反向估计，recipe kl为未重归一化截断前向和，rkl另在teacher top-k内归一化。固定main3c6d9b43还已有tail-bucket KL，不等于JSD或尾部全分布；beta端点不切纯KL。六原模块仅AST／compile，无数值／TP／custom backward、Sphinx或训练；就地纠正本项目旧k2／k3公式，无新修复／M5队列或社区写入。

## 异步恢复库存与预热容量

2026-10-25读[#7579](https://github.com/verl-project/verl/issues/7579)完整RFC／两评论，作者明确“旧模型＋新journal”非一致时间点恢复，不保留已消费轨迹，未测更晚行为策略库存对loss的影响。已有117文件原型，不下载／运行或另写后端；编号查重只命中已合并[#7762](https://github.com/verl-project/verl/pull/7762)，其warmup修复不是durable方案落地。

见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#异步恢复的存活库存与训练历史)：main3c6d9b43原9个warmup测试体隔离通过，仅完整方法／OmegaConf／原Mock接线，无TQ／重发／checkpoint往返或训练。恢复池所有prompt扣减预取目标，不能只扣pending／running重发数；新版fractional目标仅源码核对。默认flush进程保证、journal尾时延、harness生成工作秒与GPU证据分开。脚本／JSON随仓库，M5原入口仅补条件，不新排故障／GPU队列、不改专项或写上游。

## S3 下载与完整 checkpoint 生命周期

2026-10-25核[S3 RFC #7641](https://github.com/verl-project/verl/issues/7641)全文及一评论：维护者询问FUSE路线，编号PR搜索无命中，不据此认领或宣布路线已采纳。关键词另有旧[#2334](https://github.com/verl-project/verl/pull/2334)（head23bd321f、17文件774增54删）与[#6850](https://github.com/verl-project/verl/pull/6850)（headc87b60ce、实际祖先2390a3f5、ahead2／behind344、三文件120增4删），均OPEN／非draft／未合并；本轮仅读#2334正文／元数据，不认证其全实现或mock结果。

见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#对象存储下载与-checkpoint-提交)：#6850原五下载测试体候选全过、main／祖先各1过4项检查失败；完整原fs／hdfs模块接真实fsspec memory缓存，无S3/GCS/HDFS、上传或checkpoint执行。AWS单key原子性非跨key事务，RFC的_COMMITTED与状态身份／校验／恢复须另验；只补M5原条件参考，不写后端、部署云服务或扩大权限，任何上游动作仍待用户明确同意。

## 激活卸载的辅助输出入口

2026-10-25核[#7472](https://github.com/verl-project/verl/issues/7472)全文／一条评论，编号与关键词查重命中已有[#8277](https://github.com/verl-project/verl/pull/8277)，OPEN／非draft／未合并，head8c227d6a、base／祖先4b26c788，两文件225增16删，不另写竞争修复。原报告的tuple推断须区分卸载器内部(key,shape)描述与模型结构输出；已有评论已指出commit入口问题，不重复发布同一观察。

见[原架构笔记](../../notebook/frameworks/verl-architecture.md#激活卸载的反向入口与重载顺序)：原五CPU测试体两次head全过、初稿fea49f19为4过1失败（detached边界），base只跑单一辅助输出回归、以helper旧参数顺序适配后报重载前pop，不计其余测试或API错误。原RecordingOffloadHandler验证图顺序，非CUDA传输／FSDP／checkpoint或原27B验收；候选只限顶层tuple，正文“两测试”不代表当前文件只有两项。条件留[M5参考](../../experiments/handoff-m5.md#verl-激活卸载重载顺序参考)，不自动排训练／应用或改其专项；任何上游动作仍逐项待用户明确同意。

## 多轮旧配置的接线或弃用

2026-10-25读[#7662](https://github.com/verl-project/verl/issues/7662)完整正文及全部两评论；编号与inference_chat_template的all-state PR搜索均无命中，但两位参与者已在等维护者定方向，不据空搜索宣布可独立认领。选择接入use_inference_chat_template会改后续prompt语义，选择弃用会改文档契约；tokenization_sanity_check_mode需另定检查契约。

见[原CT专题](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#15-多轮配置的消费路径与告警契约)：固定main3c6d9b43的360份生产Python import AST核对中无旧schema导入／get_generation_prompt_ids调用，两键仅见配置类与旧schema；旧strict差异仅warning，活跃CT另维护真实token／mask／logprob。五相关模块compile，非原请求类／模板／模型测试；不证明仓外插件不可达或训练影响。只留证据，不另写接线／弃用补丁或排M5任务，任何社区写入仍逐项待用户明确同意。

## 仅 system 消息的契约裁定

2026-10-25读[#7798](https://github.com/verl-project/verl/issues/7798)全文及全部三评论：维护者明确CT／SFT／RL输入至少有一条user；报告者关于Uni-Agent配方归属的后续追问尚无进一步决定，不据issue仍OPEN认领通用支持。旧#7725虽CLOSED/completed，关联[#7726](https://github.com/verl-project/verl/pull/7726)为CLOSED/unmerged、heade4582b87，不能标成修复已合入；完整正文、两普通／两行内评论已核，作者25测试与processor结果不计本机。

见[原CT专题](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#14-仅-system-消息的输入契约)：固定main3c6d9b43的VL builder仍调用共享wrapper，旧“CT不再调用”的概括不适用于所有路径；这不推翻输入契约。四份原源码AST／compile及调用点核对，无真实processor／旧测试、模型或GPU。不造替身、补用户消息／合并system、不改原配方／wrapper或重复开修复；本题无新增GPU缺口／M5队列，所有社区写入仍逐项待用户明确同意。

## Megatron 迁移的已有原型与验收边界

2026-10-25读取[Modern Megatron #8324](https://github.com/verl-project/verl/issues/8324)完整正文／0评论；编号检索只命中#8338，不能据此宣布无人做迁移，正文已链接作者个人fork原型。[Core #263](https://github.com/ISEEKYAN/Megatron-LM/pull/263)当前OPEN/draft、未合并、0评论／行内评论，headb1a29d72／base728ef141，十文件339增27删。读取完整正文／十文件diff，聚焦MTP，不另接管整体迁移、补依赖或开同题PR。

见[原输出头专题](../../notebook/frameworks/verl-fused-output-softcap.md#原生-mtp-的梯度与归一化控制)：完整原辅助损失／autograd／roll与原八参数测试体，两次CPU通过；同一输出层参数的显式传入不等于真实embedding绑定，等计数mask不覆盖不均匀全局分母。仅base/head新增关键字签名绑定，无base梯度测试、完整Core／CP或模型／GPU验收；不与作者80／15／8报告合计。源码契约仅补M5原条件参考，无新队列／专项改动，任何社区写入及既有PR-head推送仍逐项待用户明确授权。

## FSDP 精度默认的两份候选

2026-10-25核#8076完整正文／0评论，编号及FSDP dtype关键词查重，旧[#8083](https://github.com/verl-project/verl/pull/8083) headba4b3f2d和新[#8359](https://github.com/verl-project/verl/pull/8359) head74eac195均OPEN/non-draft/unmerged。完整构造方法去文档字符串后AST相同，新PR另有QAT导出回退，旧PR另有文档与原生CUDA测试，非整个PR完全相同，不另实现／拼接或对外评论。

见[原架构笔记](../../notebook/frameworks/verl-architecture.md#fsdp-初始化与计算精度的优先级)：初始化／计算／归约／导出分开，显式param_dtype仍优先。旧原测试体20参数例两次在新旧候选均20过、main3c6d9b43为8过12断言失败；真实torch策略对象配原Mock，不是FSDP/scaler／训练验收。新CPU文件是FSDP六例加FSDP2一例，本轮未运行；QAT与delta-shard仅源码核对，不沿用作者设备数字。仅补M5原条件入口，无新算力队列或专项改动，任何社区写入与已有PR-head推送仍逐项待用户明确授权。

## SGLang 概率模式的已有候选

2026-10-25读取 #8362 与作者 #8363（OPEN/non-draft，head107c8b08、base3c6d9b43，三文件44增0删），不再另实现或重复反馈。[概率口径笔记](../../notebook/frameworks/verl-async-logprob-provenance.md#sglang-概率模式与请求参数)区分raw温度前概率、processed常规路径与截断归一化。原四配置测试体AST隔离执行两次：基线2过2项预期检查失败，候选4过；仅val_kwargs.top_p=0.8构造零告警，已有机器人行内意见指出同一请求覆盖问题。

仅配置与源码验证，非完整pytest收集、真实请求或引擎；两版共12份完整源码compile，未导入Torch/Ray/SGLang/vLLM/Transformers。锁定SGLang0.5.20源码开关在模块导入读取，真实启动环境仍须核对；作者两步4090报告不计本机结果、不认证逐token一致。M5条件参考补原入口，不新增队列，不改#8313/AReaL/#523/DeepInfer；所有社区写入和既有PR-head推送仍须用户逐项明确同意。

## 响应截断指标的已有修复

2026-10-25读[#7817](https://github.com/verl-project/verl/issues/7817)、[#7823](https://github.com/verl-project/verl/issues/7823)全文／均0评论，编号及短关键词查重分别已有[#7821](https://github.com/verl-project/verl/pull/7821)、[#7906](https://github.com/verl-project/verl/pull/7906)，两项open、非draft、未合并；PR正文全读，普通评论分别0／2（作者请求review）、行内均0。#7821 head7255b49c／祖先a9f29851，两提交三文件231增4删；#7906 head2b553602／父祖先5426b926，一提交16文件412增17删，API目标base3f9d4c0d另计。不认领同题或自动拼接候选。

见[原数据流专题](../../notebook/frameworks/verl-grpo-data-flow-reading.md#响应长度触顶与真实停止原因)：真实CPU TensorDict、原DataProto／指标函数AST，#7821原5测试体候选通过，旧版因kwargs缺失5次TypeError，另核旧签名0.5假阳性→显式cap0；#7906原13项候选通过／基线7过6失败。False与None不同、unknown不从分母剔除，零长度aborted指标非全部后端抢占。未执行原生pytest／完整V1、Ray／TQ、后端或作者41／101套件，真机条件只补原M5参考，不应用候选、新排实验或改专项，任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## boxed 判分的已有修复

2026-10-25读[#8123](https://github.com/verl-project/verl/issues/8123)全文／0评论、报告者[#8124](https://github.com/verl-project/verl/pull/8124)全文／两文件diff及两条普通评论（CLA、外部11项验证报告），0行内评论。编号及关键词查重已有覆盖，PR为open、非draft、未合并；head c5e26eec，父提交／API base／祖先8718ca30，两文件52增2删，不另实现。

见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#boxed-答案的归一化与选择优先级)：原七组表示加错误／strict两例，完整原模块直接调用测试体，基线2过7失败、候选9过，两次一致；不是pytest原生收集或完整reward套件。三个pred从两键变一键，但未跑完整maj@N；冲突Answer仍优先，分数与小数仍不等价。此纯CPU任务不需M5新队列，不应用候选／修改专项；任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## Triton 输出形状的已有修复

2026-10-25读[#8089](https://github.com/verl-project/verl/issues/8089)全文／0评论，编号及Triton output shape查重已有报告者[#8090](https://github.com/verl-project/verl/pull/8090)，open、非draft、未合并，唯一普通评论仅CLA机器人、0行内评论，不另实现。head90c8667a、共同祖先fbb4b3a8，两提交六文件43新增；父提交6ac75615已含修复，最后提交仅加测试，API目标base8718ca30另计。

见[原输出头专题](../../notebook/frameworks/verl-fused-output-softcap.md#triton-输出的批维与序列维)：五个Triton函数AST只增三条形状恢复，Torch与标签选择未改。原新测试声明20个模型／kernel替身参数例，**本机未运行**；旧假kernel二维结果不覆盖真实平铺外形。14完整源码快照AST／compile，无原生包／kernel／模型、SP或训练验收；真实条件补M5原参考，不应用候选／新排任务、改专项或上游写入。任何外部动作及既有PR-head推送仍逐项待用户明确同意。

## 代码部分分数的已有修复

2026-10-25读[#8021](https://github.com/verl-project/verl/issues/8021)全文／0评论，编号及code reward关键词查重已有报告者no-hup的[#8022](https://github.com/verl-project/verl/pull/8022)，open、非draft、未合并、唯一普通评论仅CLA机器人、0行内评论。head a5aef7cd、父提交／API base／祖先6093e007，四文件32新增，不另实现；#8175／#8360是已有调用格式修复，未拼接。

见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#代码奖励的已知失败与部分分数)：完整原本地判题器执行固定无害程序，错第11例的旧版得1、候选10/11；原本地真实测试与远端mock测试均base失败／head通过。21例中10对11错仍得10/11，不是10/21全量比例；本地metadata10条与远端mock21条不一致，合成错误码−1路径仍可得1。仅CPU fork／结果包聚合，不认证外部服务、安全隔离／训练或作者28过13跳套件。无候选应用、M5新队列／专项或上游写入，任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## V1 到期保存的已有修复

2026-10-25读[#7757](https://github.com/verl-project/verl/issues/7757)全文／0评论，编号及esi_redundant_time查重已有报告者[#7760](https://github.com/verl-project/verl/pull/7760)，open、非draft、未合并、0普通／行内评论，不另实现。head16cd240d、共同祖先a9f29851，两提交两文件211增7删；父提交6043685b已含第一版，API目标base3f9d4c0d另计，未当完整修复基线。

见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#v1-到期前保存的触发与计时范围)：完整原helper／两方法接作者stub，12原测试体通过；MLP要求正剩余时间与正历史耗时，AWS没有同样门槛。继承helper的非法AWS时间戳在禁用写盘时仍先解析抛错，非本候选新增回归；fit保存使用前面历史，计时含保存／不含验证。源码已有调用，但原测试说明中的fit守卫并不存在，初始化也未执行；反馈仅留个人项目待审。无真实trainer／云服务／写盘恢复或GPU，不改共享helper／专项、新排M5任务；任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## RLOO 两项契约的已有修复

2026-10-25读[#7793](https://github.com/verl-project/verl/issues/7793)及唯一评论，编号／RLOO singleton查重已有[#7794](https://github.com/verl-project/verl/pull/7794)、[#7581](https://github.com/verl-project/verl/pull/7581)及较宽的[#7150](https://github.com/verl-project/verl/pull/7150)，另有独立dtype [#7964](https://github.com/verl-project/verl/pull/7964)，四项均open、非draft、未合并，不另实现。读#7794／#7964正文、全部两文件diff及全部1／2普通评论，两项均0行内评论；head58b37737／ea236890，实际祖先f1c27141／f542ae22，两文件各26增1删／32增1删。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#rloo-的单样本回退与输出精度)：完整原函数AST和注册接线，原singleton一例及dtype三例分别候选通过，#7964仍不处理singleton。真实CPU bincount对FP32／低精度输入并非一律FP64；BF16 [256,1]候选与标量均为[256,−254]，原vectorized／FP64参考为[255,−255]，类型一致不证明数值不变。现有dtype测试只断言类型，供用户审视的描述／数值覆盖反馈留项目内，未跑原生24／26套件、trainer或GPU。不拼候选、改专项／新排M5任务；任何上游动作及既有PR-head推送仍逐项待用户明确同意。

## 确定性路由的已有修复

2026-10-25读取[#7666](https://github.com/verl-project/verl/issues/7666)全文／0评论，编号及deterministic routing查重见报告者hsusul已有[#7667](https://github.com/verl-project/verl/pull/7667)，open、非draft、未合并；一条普通评论仅CLA机器人、0行内评论。head b2b5f4f6、父提交／祖先4eff9ae3，两文件85增5删，API目标base3f9d4c0d不冒充实际祖先，不另实现。

见[原确定性专题](../../notebook/frameworks/verl-6572-full-determinism-source-reading.md#确定性路由的哈希与副本集合)：完整原类AST接真实cachetools，原两测试体候选通过；两版各4个真实子进程核hash seed，候选结果相同，基线不同。3→4副本时sticky保留旧映射，清缓存后本例21/30改变；CRC32与排序只对固定ID集合消除hash／插入顺序依赖，不认证物理worker或全训练。未跑原整包跨进程测试／原生25项套件、Ray或GPU，不应用候选、改专项／新增M5队列，上游任何动作及已有PR-head推送仍逐项待用户明确同意。

## 多语言分词的官方修复与验证范围

2026-10-25读取[#8030](https://github.com/verl-project/verl/issues/8030)及唯一评论，编号查重无verl PR；问题在Transformers，已有官方[#49116](https://github.com/huggingface/transformers/pull/49116)合并（a4fe1d5b），不是关闭未合并的[#49077](https://github.com/huggingface/transformers/pull/49077)regex kwarg方案。读官方#49066及全部三评论，另核#49114／#49341仍OPEN未合并；不据未发布候选重写verl tokenizer。

见[原数据流专题](../../notebook/frameworks/verl-dataproto-data-flow-reading.md#多语言预分词与类路由)：TF5.20.2／tokenizers0.23.3真实本地toy BPE，三句组合标记文本round-trip均正常但IDs有别；已知qwen3_5 model_type时，显式Qwen2声明仍经Auto纠正并与原backend一致。不把issue的5.20.1具体衍生checkpoint反馈泛化到全部显式声明，也不以本例否认其具体结果；无公开checkpoint／原Hub测试／actor-rollout或训练质量验收。未改全局pin、安装依赖或新增M5队列，上游任何动作及PR-head推送仍逐项待用户明确同意。

## TRL 边界探测的已有修复

2026-10-25读取[#7931](https://github.com/verl-project/verl/issues/7931)全文／全部两评论，编号与cu_seqlens查重见报告者已有[#7932](https://github.com/verl-project/verl/pull/7932)，OPEN／非draft／未合并，0普通／行内评论；head918c80b0、父提交／祖先3efe38c7，API目标base3f9d4c0d，两文件288新增。其他delta-kernel／Ulysses候选不替代此wrapper探测修复，不另开发。

见[原架构专题](../../notebook/frameworks/verl-architecture.md#trl-包装器的打包边界探测)：精确类身份及类绑定forward后才查底层能力，不把任意kwargs当支持。独立核TRL0.27／0.28／1.12原forward AST相同且显式签名均不含cu_seqlens，1.13完整tree及官方移除PR确认原模块不存在；升级不是直接补救。本机仅源码／签名／四文件语法，不执行TRL、作者35／53测试、model或kernel；无候选应用／M5新队列或专项改动，上游任何动作及PR-head推送仍逐项待用户明确同意。

## KL-Cov 选词范围的已有修复

2026-10-25读取[#7829](https://github.com/verl-project/verl/issues/7829)全文／0评论，编号与kl_cov查重见报告者已有[#7830](https://github.com/verl-project/verl/pull/7830)，OPEN／非draft／未合并、0普通／行内评论；head79a7af1c、父提交／祖先6fc912f0，API目标base3f9d4c0d，八文件531增26删。旧[#7828](https://github.com/verl-project/verl/pull/7828)关闭未合并且同head；[#6538](https://github.com/verl-project/verl/pull/6538)是ratio稳定性方向，不另开发。

见[原算法专题](../../notebook/frameworks/verl-grpo-core-algos-reading.md#kl-cov-的选词范围与微批切分)：原五纯CPU例通过、真实原配置与函数AST，配额1对8及相同配额仍选不同位置分开；固定整批mask切片后输入logp梯度与整批一致。仅单进程helper，不复跑作者worker壳／Gloo／模型或GPU，额外前向与DP数据交换只有源码证据，不冒充性能收益。无候选应用／M5新队列或专项改动，上游任何动作及PR-head推送仍逐项待用户明确同意。

## 共享参数加载的已有修复

2026-10-25读取[#7834](https://github.com/verl-project/verl/issues/7834)全文／0评论，编号与tied embeddings rank0查重见报告者inin-zou已有[#7835](https://github.com/verl-project/verl/pull/7835)，OPEN／非draft／未合并；head93f5328f、父提交／祖先a9f29851，API目标base3f9d4c0d，三文件123增3删。读完两条普通评论及两条行内讨论；旧[#5746](https://github.com/verl-project/verl/pull/5746)关闭未合并，不另实现。

见[原架构专题](../../notebook/frameworks/verl-architecture.md#meta-加载后的共享参数与梯度)：真实tiny Llama原四用例CPU通过，但原meta `data_ptr`断言对显式untied模型也返回True。本机Accelerate路径构造后对象已不同；原生torch meta另证共享→物化断开→重绑恢复。前向loss逐bit相同不能替代梯度结构，候选原重绑条件恢复共享梯度。反馈只留项目内待用户审视，不改作者测试或发表review；无完整FSDP／分布式loader／内存验证、M5新队列或专项改动，上游任何动作及PR-head推送仍逐项待用户明确同意。

## Critic 批处理配置的已有修复

2026-10-25读取[#7462](https://github.com/verl-project/verl/issues/7462)全文／0评论，编号与critic batching查重见同报告者已有[#7465](https://github.com/verl-project/verl/pull/7465)，OPEN／非draft／未合并，无普通／行内评论；head2609808d、实际父提交／祖先23a341e3，API目标base3f9d4c0d，十文件160增4删。当前范围仅v0／V1，不含已弃用separation，不另认领。

见[原架构专题](../../notebook/frameworks/verl-architecture.md#critic-批处理配置如何进入引擎)：真实原配置类AST与六原测试体CPU通过，四份generated YAML仅取批处理字段核插值；直接构造仍保留infer默认32768，显式None才回退forward。worker仅为batch缺失键注入engine默认值，外层字段正确不证明实际调用值。没有完整Hydra／backend／训练，不应用候选或新增M5任务／专项改动；任何上游评论／issue／PR及既有PR的fork head推送仍逐项待用户明确同意。

## SFT 验证尾批的已有修复

2026-10-25读取[#7464](https://github.com/verl-project/verl/issues/7464)全文／0评论，编号及SFT drop_last validation查重均见同报告者已有[#7467](https://github.com/verl-project/verl/pull/7467)，OPEN／非draft／未合并。固定head87dc0d00、实际祖先a9f29851，API目标base3f9d4c0d，五文件625增38删；当前第三提交已从样本权重改token权重，不沿用旧LGTM／正文旧概述或认领重复修复。

见[原数据流专题](../../notebook/frameworks/verl-dataproto-data-flow-reading.md#sft-验证的尾批与-token-分母)：完整原helper／collator／TensorDict及原16测试体CPU通过，真实sampler重复索引本例多计3token，零mask补位有效数为55；显式NLL token加权约1.0385、样本加权1.8，非模型测量。原静态micro切分函数AST隔离、逻辑rank依次执行，无完整trainer／动态打包／设备通信，七份语法检查不等于后端验收。无新M5任务／专项改动、候选应用或上游写入，任何外部可见评论／PR及PR-head推送仍逐项待用户明确同意。

## 工具 schema 约束的已有修复

2026-10-25读取[#7505](https://github.com/verl-project/verl/issues/7505)全文／0评论，编号与tool schema短关键词查重确认YZJF已有非draft [#7523](https://github.com/verl-project/verl/pull/7523)，OPEN／未合并、headd7f5e73b；直接父提交1dda039b下两文件20增4删，API目标base3f9d4c0d不能冒充本地未求出的共同祖先。另一[#7766](https://github.com/verl-project/verl/pull/7766)关闭但未合并，不作主线已有修复或无人负责。

见[原工具专题](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#13-工具-schema-保真与执行校验)：完整原schema／function_tool接真实Pydantic及无副作用调用，原1回归体主线AssertionError、候选通过；约束保真、元schema合法性、JSON解码与Python参数调用分层，extra=allow不承诺任意schema或运行时校验。五份源码快照语法通过，未运行原生pytest／AgentLoop／工具服务或安全攻击，不搬关闭PR的设备测试报告。本题不需M5新队列、不应用候选／另写校验器或改专项，任何上游评论／PR及PR-head推送仍逐项待用户明确同意。

## 数据子集与验证入口的已有修复

2026-10-25读取[#7816](https://github.com/verl-project/verl/issues/7816)／[#7824](https://github.com/verl-project/verl/issues/7824)全文、均0评论；编号及短关键词查重已有[#7819](https://github.com/verl-project/verl/pull/7819)与[#8141](https://github.com/verl-project/verl/pull/8141)，两项open、非draft、未合并，不认领第三修复。固定head0554e35f／9a7bd53b，实际祖先a9f29851／75879f7f；前者API目标base为3f9d4c0d，按实际祖先核四文件，不把主线进展算入修复。

见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#rl-数据子集与恢复身份)：原类／工厂／采样器接真实Parquet／JSONL／StatefulDataLoader，两候选各5原测试体通过，非原生pytest／完整框架。固定seed不校验内容；#8141四个原验证调用表达式中V1正确传False，但旧同步、one-step与fully-async省略参数，仍走默认True。原工厂测试显式传参，不能覆盖此接线缺口；#7819当前正文的调用点说明也不能替代真实diff。未构造trainer／恢复模型或碰设备。

**供用户审视的反馈要点**：请#8141现有作者核对三处验证调用的is_train=False接线与调用层测试，同时保留“头部选择会改变评测来源组成”的语义讨论。当前行审0评论、普通评论仅CLA机器人；本项目只保存源码／CPU证据，不评论、编辑PR／推其head或另发重复修复，任何上游可见动作仍逐项待用户明确同意。本题无需新M5队列，不改其专项。

## 采样概率口径的已有方案

2026-10-25读取[#8020](https://github.com/verl-project/verl/issues/8020)全文及唯一作者补充，报告者已有支持域大小回放原型，不认领新修复。已有[#8207](https://github.com/verl-project/verl/pull/8207)仍open、非draft、未合并，head0ed74d91、base/祖先2035f58d；两份生产改动是启动保护、零ESS保护及告警，非支持域回放。[#8196](https://github.com/verl-project/verl/pull/8196)已合并且为固定主线祖先，但完整支持域凭据选择的是score centering路径，不代表一般IS／RS已经解决该问题。

见[原概率专题](../../notebook/frameworks/verl-async-logprob-provenance.md#采样截断后的支持域与归一化)：原六helper／两个mask函数接有限CPU分布，0.9保留质量导致0.9单token和约0.349十token序列权重，严格k1拒绝而K3仅约0.00536；这不是完整精确KL、同步根因或训练回归。已知同支持域控制恢复1不证明原目标无偏；actor排名变化时相同k不保证相同集合。五份完整源文件语法检查，未跑候选原测试／配置启动／SC或后端；[M5条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)仅供已有获准匹配观测，无新队列／候选或专项修改，任何上游写入及PR head推送仍逐项待用户明确同意。

## 长文本预处理的已有修复

2026-10-25读取[#8261](https://github.com/verl-project/verl/issues/8261)全文/0评论；编号开放及历史PR查询为空，但报告已明确官方[Transformers #47580](https://github.com/huggingface/transformers/pull/47580)合并，不把空结果当修复空缺。见[原数据流专题](../../notebook/frameworks/verl-dataproto-data-flow-reading.md#多模态特殊-token-检查的-cpu-开销)：精确5.12.1/5.15.0原方法接list/CPU torch/NumPy，18组成对契约一致；131,072-token CPU方法中位耗时约319.422→1.379ms，非完整processor/分词或训练吞吐，未下载模型资产。

固定verl3c6d9b4锁仍为Transformers5.12.1/SGLang0.5.20/Torch2.13组合；本机torch2.14.1仅提取方法控制，不能认证升级组合。None绕法保留list计数且后续仍可能转Tensor，无模态dataset过滤又走tokenizer，不外推所有入口。已有官方路线及依赖协调，不写竞争shim、不改pin/环境、不新排M5任务；任何上游写入及PR head更新仍逐项待用户明确同意。

## Megatron 调度恢复的已有修复

2026-10-25读取[#7953](https://github.com/verl-project/verl/issues/7953)全文/0评论、已有同报告者[#7954](https://github.com/verl-project/verl/pull/7954)全文及相关加载/构造接线。两条开放PR查重查询均仅该覆盖；head53e63285、base/祖先fbb4b3a8，open、非draft，不另认领。见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#megatron-调度进度与超参数恢复)：两manager入口恢复仍受should_load_optimizer控制，flag选择受管理的超参数来源，不应据默认False丢进度。

官方MCore0.19.2三个完整原模块以轻量导入接线执行，真实CPU SGD及torch状态文件六控制通过；进度恢复、不同超参数、重复load增量与参数组覆盖各有范围，SGD组反例不定性Megatron bug。五相关候选模块仅compile，未复跑作者43/57 manager mock测试或Megatron-FSDP恢复；条件补[M5原checkpoint入口](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不应用、新排训练或改专项，任何上游动作仍逐项待用户明确同意。

## Megatron 融合消费的已有修复

2026-10-25读取[#8324](https://github.com/verl-project/verl/issues/8324)全文/0评论，已有fork原型及分阶段迁移，不展开竞争迁移。沿其前置[#8028](https://github.com/verl-project/verl/issues/8028)全文/0评论核已有同报告者[#8029](https://github.com/verl-project/verl/pull/8029)：open、非draft、未合并，head99b6f022、祖先4b26c788（API base263653e7），编号检索仅该覆盖，不把关键词空结果当无人负责。见[原输出头专题](../../notebook/frameworks/verl-fused-output-softcap.md#megatron-融合权重的就绪与调度)：直接读取权重会绕过模块前钩子，handle排空、下一桶派发权、hidden gather及graph阶段各有边界。

原CPU文件未改、pytest原生收集23项通过；作者fake DDP/Work/graph与CPU替代融合数学不认证真实MCore通信或Triton。官方MCore0.19.2预钩子/bucket/API源码另核，六候选加三官方模块九份语法检查；作者2–4H20报告非本机结果，VPP未覆盖。条件补[M5原输出头入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不应用、拼接FSDP方案或新排队，不改专项；任何上游评论、PR及已建PR的fork head推送仍逐项待用户明确同意。

## 数学表达式解析的已有方案

2026-10-25读取[#7603](https://github.com/verl-project/verl/issues/7603)正文/唯一评论、[#7607](https://github.com/verl-project/verl/pull/7607)和[#7701](https://github.com/verl-project/verl/pull/7701)完整正文/两份diff，PR review comments均0；两项open、非draft、未合并，不认领第三种实现。固定head分别b7d469f6、752c6ebe，祖先分别ea532913、4eff9ae3；旧grader均与主线3c6d9b4同字节。见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#数学表达式的语法与资源边界)：递归求值与节点检查/编译的语言不同，list内算式与literal_eval的兼容不同，局部整数限额也不是完整worker预算。

两个候选各4个原良性方法共8方法通过，另有限helper/原math_equal CPU控制两次一致；不运行攻击、状态投毒或耗尽载荷，不搬作者完整安全套件报告。#7701 helper接受bool/string/位移并不证明这些输入得分；单元素矩阵的分数算式差异才明确到达math_equal返回值。本轮不合并/采用或对外提醒，不改专项、不新排M5任务；如需上游评论、PR或其fork head更新，仍须用户对具体动作明确同意。

## QAT mask 默认与后端契约

2026-10-25只读[#8361](https://github.com/verl-project/verl/pull/8361)正文/完整三文件diff及唯一机器人评论，open、非draft、未合并，head `cf6eb5181e17b4db67afff110eabc61daf61fb4d`、base/实际祖先 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`。已有作者，不另认领。见[原量化专题](../../notebook/frameworks/verl-quant-layer-audit.md#qat-的-mask-默认与-packed-接口)：按固定锁文件Bridge0.6.2/MCore0.19.2/TE2.16.1核清arbitrary、causal、padding_causal选择链；setdefault保留显式True/None，mask类别满足不等于完整THD输入或部署backend可用。

七份完整原文件AST/compile与helper静态检查通过，没有执行原两项替身测试、真实Bridge/TE/量化或设备。作者等价override的GH200/5090观测不是候选分支验收，文档中限定MXFP4/FP8 activation方案的Hopper部署报告也不能外推全部低位宽模型。[M5原量化入口](../../experiments/handoff-m5.md#verl-量化层审计新版复验)只补已有获准匹配工作中的条件参考，无新任务/竞争实现或专项修改；上游评论、PR动作及已建PR的fork head推送均待用户逐项明确授权。

## 代码判分的格式与逐例接线

2026-10-24读取[#8174](https://github.com/verl-project/verl/issues/8174)全文/0评论与新[#8360](https://github.com/verl-project/verl/pull/8360)（af4f9a3b、base/祖先3c6d9b4）及已有[#8175](https://github.com/verl-project/verl/pull/8175)（569c875e、base/祖先fc72e2f），两者open/non-draft、未合并。编号检索已见两个覆盖，不另开实现。见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#代码判分的格式与逐例接线)：两原测试共8项接三版完整评分模块、真实Manager/Process，原imports/断言不改；main3过5失败、#8175七过一失败、#8360六过两失败。#8360仍缺逐例fn_name保留，#8175仍改写native字典参数，不能以同标题或满分正确解判等价。

当前#8360另三真实echo控制确认普通文本catch与数字形文本JSON解码不同；需明确编码schema，不新增猜测式修复。初次sandbox socket拒绝不计回归，放行后才比较；祖先包空壳隔离不是完整verl初始化/数据集或训练验收。只保存本地学习，无候选补丁、M5硬件队列或上游写入，任何外部可见动作仍待用户逐项授权。

## 异步 checkpoint 的能力与发布

2026-10-24读[#7952](https://github.com/verl-project/verl/issues/7952)及唯一评论；同报告者已有[#7955](https://github.com/verl-project/verl/pull/7955)（headb8c1aa6a、祖先23a341e3、ahead1/behind6），不另实现。见[原checkpoint专题](../../notebook/frameworks/verl-checkpoint-management-reading.md#异步保存能力与恢复指针)：配置白名单是拒绝尚不支持的组合，不是非Megatron异步保存；[#4311](https://github.com/verl-project/verl/pull/4311)旧head f0411c95与目标base相差1148提交，其DCP future/完成日志不提供当前trainer所需tracker发布，不能直接改白名单宣称集成安全。

本机只运行原查找helper的五个真实文件系统控制与原一项测试方法，确认空目录可被找到但不认证内容完整；未执行ActorConfig构造/原14项、FSDP/DCP保存/恢复或训练。条件留[原项目M5参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新增训练队列、合并两方案或发布新补丁；上游任何可见动作须用户逐项授权。

## 数学评分池的故障恢复

2026-10-24读取[#8011](https://github.com/verl-project/verl/issues/8011)两条评论与已有[#8015](https://github.com/verl-project/verl/pull/8015)（ChdDongyang，open/non-draft，head0c4c3d40、base/祖先6093e007）；不是空缺，不另实现。见[原奖励专题](../../notebook/frameworks/verl-reward-integration-reading.md#数学评分池的恢复与错误信号)：原七测试仅调整导入，生产模块逐字节不改，真实math-verify0.9.0/CPU spawn两次全过；旧版三个评分控制过，worker死亡及超时不换池各原例预期失败。SIGKILL/sleep为故障注入，非真实OOM/恶意公式复现，不等于完整verl/CI或RewardLoop验收。

候选只对BrokenProcessPool重试一次，Future超时清共享池会影响其他在飞任务；超时仍返回float默认分，不自动变成manager的timeout标志。服务恢复与错误可观测性分开，不归因#8250或据通过宣布任意并发安全；没有GPU缺口，不新增M5队列。本地学习收束，采用/上游回复/PR及PR-head推送仍须用户逐项明确同意。

## R3 路由传输的阶段选择

2026-10-24读[#8160](https://github.com/verl-project/verl/issues/8160)全文（open/0评论）；报告者weiziyoung已有内部patch、计划提交PR，不另认领。见[原TQ专题](../../notebook/frameworks/verl-transferqueue-worker-lifecycle-deep-reading.md#路由记录的层选择与全局坐标)：固定main3c6d9b4中bridge未按PP切字段内部层维，Megatron主路径按真实模型router的全局层号消费；局部紧凑张量不自动等价，VPP/非均匀布局、R2/R3及CP/SP需分开。四份AST/compile不是原测试/传输或作者128GPU报告复跑；有限编号/关键词检索未命中开放关联PR，不表示无人负责。

目前只保存架构学习与[原M5条件参考](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)，不开发新split/concat方案或自动迁移数据schema，不新增PP训练队列。若有实质公开实现或明确协作需求再核范围；上游回复/issue/PR及fork PR-head推送均须用户逐项明确同意。

## DSA 依赖与有效输入范围

2026-10-24读取[#8238](https://github.com/verl-project/verl/issues/8238)全文及两条评论；[原专题补充](../../notebook/frameworks/verl-6791-dsv4-megatron-lite-reading.md#dsa-奇数输出与依赖版本)固定main3c6d9b4的Frontend1.26.0锁定、Megatron43e45e13调用及官方1.26/1.27存储代码。官方[#407](https://github.com/NVIDIA/cudnn-frontend/pull/407)已合并，1.27源码含奇数标量回退；不是新kernel贡献空缺。有效top-k来自压缩KV维度，且旧assert只在选中向量宽度2的奇数上触发，不是任意奇数token长度。三模块语法检查、锁文件解析非GPU/全栈复现，外部H100/GB200报告不搬本机。

编号及cudnn/cudnn-frontend开放PR检索只作有限归属检查，噪声命中不算覆盖，空的精确关联也不算无人负责。升级路线仍需维护者兼容决策；本项目不自动改锁、另开issue、评论或PR。硬件缺口见[条件参考](../../experiments/handoff-m5.md#verl-dsa-依赖兼容参考)，不新增Pro训练或环境升级队列；任何上游可见动作（含fork上的PR head推送）须用户逐项明确同意。

## 当前活跃

**（2026-10-19 全候选复查重后）** 扫描报告见 [scan-1019-1140.md](scan-1019-1140.md)。注意：全新 issue 认领窗口 ≈ 0-1 天，且 **reporter 深挖过的 issue 大概率 reporter 本人已开 PR**（本轮 4/4 验证）。

**（2026-10-20 更新）** verl 新 issue 池恢复（10-16/10-19 共 3 个：#8261 perf、#8265 问卷非 bug、#8281 bug）。#8281 深挖完成（analysis-done + CPU 验证），登记为第 3 个活跃候选。

**（2026-10-23 扫描补充，10-24接续）** 原扫描仅登记 #8324 Modern Megatron integration、#8334 teacher-generated rollouts 和 #8342 VeOmni 端到端低精度训练。后续已确认 [#8334](https://github.com/verl-project/verl/issues/8334)有作者的 [draft #8335](https://github.com/verl-project/verl/pull/8335)，不再列为“尚未查明归属”；#8324仅读RFC确认已有fork原型，不展开大型迁移。[#8342后续尽调](#veomni-端到端低精度路线)确认作者已有原型、维护者要求先在VeOmni讨论，不作未认领候选。

| # | 候选 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|
| 0 | megatron `optimizer_offload_fraction<1` 首轮 update 崩溃 (#8237) | P0 | 42 | [#8286](https://github.com/verl-project/verl/pull/8286) **MERGED** `472dd21` | **已合并**（2026-10-20）：根因 100% 实锤（无差别 `to("cpu")` 污染 GPU 半 + fp32 参数悬空 view；P1/P2a/DeepSeek 模型源码三链）；方案 2（HDO 状态在 offload 完全不动）PR，issue #8237 已回复，4090 全验证 + 外部 GB200 验证（linmuchuiyang）→ wuxibin89 approve 后 08:06 合入。详见 [candidate](candidate-8237-offload-fraction.md) | 2026-10-19 |
| 1 | async熵异常与坍塌诊断 (#8250) | 调查/评论 | 重评见candidate | — | **needs-evidence**（2026-10-21）：仍open/0评论；旧“结果0必然全梯度0、优势离群必然性、同步唯一根因、紧窗导致无法恢复”撤回。报告源码+NumPy反例已补，不是训练实测；需有效奖励/优势/mask/ratio/版本数据，旧英文稿不可直接发 | 2026-10-19 |
| 3 | agent loop shared cap 忽略data.truncation（#8281） | P2 bug | ~25 | [#8313](https://github.com/verl-project/verl/pull/8313) | **open，等维护者 approve；新增社区审查待处理**（2026-10-23 23:07 轮）：head `2079ab0e`（fork `Jackie2049:fix-8281-truncation-policy`，4 文件），维护者 dajiaohuang review 意见已闭环（`d0ca3be1` RolloutConfig 校验 prompt_length>0 + left cap=0 正向切片；`2079ab0e` 注释精简）；middle/cap=1 反例已修复；GPU e2e 冒烟已完成。**新进展：社区成员 Jackmeson1（author_association NONE）10-23 14:30Z 发 PR 评论，四条意见经 M5 核验全部属实**——①测试文件命名 `test_cap_text_prompt_length.py` 无 `_on_cpu` 后缀且位于 `tests/experimental/`，而 verl CPU 工作流只收 `test_*_on_cpu.py`（`python_files = *_on_cpu.py`）、GPU 工作流显式 `--ignore-glob="tests/experimental"` → **当前该测试两工作流都不收集，零 CI 覆盖**，建议改名 `test_cap_text_prompt_length_on_cpu.py`；②middle 分法与 `torch_functional.postprocess_data` 不一致（postprocess 用 `left_half = max_length // 2` 奇数余量给尾，我们 `(prompt_length+1)//2` 给头；cap=1/3/5 结果不同，如 cap=5 `[0,1,2,8,9]` vs `[0,1,7,8,9]`，cap=1 `[0]` vs `[9]`，偶数 cap 相同——M5 服务器实测对照）；③测试 middle 断言自引用（期望前缀取自 `out` 自身，`[0,7,7,8,9]` 这类错误输出也能过），建议 caps 1-6 显式期望；④error 在 trainer 级是否真正中止训练需澄清（V1 可能按 prompt group 捕获异常）。②是否对齐 postprocess_data 属语义决策、④需设计澄清，改 PR/回评论均需用户授权；M5 已备改进补丁草案与回复要点，见 [candidate](candidate-8281-agent-loop-shared-cap-vs-truncation.md#当前候选的最小正上限反例)。**10-24/25 状态**：①②回复已按用户逐字文本发出（10-24 22:32、10-25 12:00），head 经用户裁决为 71fd3562（cap=1 保留首 token、测试改名回退、middle 对齐 postprocess_data，CPU 18 passed + GPU e2e 2/2 PASS + 负例 fail-loudly PASS）；仍 OPEN/MERGEABLE 等维护者 approve（requested reviewers wuxibin89+dajiaohuang），③讲解中、④草稿 v3 待用户逐字审定 | 2026-10-20 |
| 4 | 训练前 weight-update 自检 RFC (#8252) | RFC/评论 | 旧评约13，非当前接受度预测 | — | **assessed，旧结论已纠正**（2026-10-24）：维护者10-20已指向现成rollout_logprob_diff，报告者10-21仍解释slime式自检；功能对照经真实更新后测采样token概率，不等于独立参考下的破坏/恢复内容自检。原metric/退出片段CPU控制仅说明平均/末轮门槛，无模型/同步实测。#7108旧stale/冲突不作当前断言，仍不建议我方实现；详见[candidate](candidate-8252-weight-checker-rfc.md#现有验证能力与当前判断)，参与/发布待授权 | 2026-10-20 |

## 近期候选覆盖核对

2026-10-23 本轮读取前两项 issue 正文与全部评论（均为 0），按编号和短关键词查开放 PR，再读真正相关 PR 正文并核 API 状态/head；#8334 沿用当日已核正文，另重核关联 PR 状态。这里判断的是**是否已有覆盖工作**，不是验证补丁完整正确、CI 通过或可合并。

| 问题 | 覆盖证据 | 本项目处理 |
|---|---|---|
| [#8174 call-based 代码奖励](https://github.com/verl-project/verl/issues/8174) | 报告者 prabhu-gopal 的 [#8175](https://github.com/verl-project/verl/pull/8175) OPEN/non-draft，head `569c875e10d6e2bc55b96ef78cda5ae32e627e9f`；正文同时声明处理 APPS/TACO 参数列表与 JSON-line 字符串，以及逐例重测保留 fn_name | 两个问题均已有声明覆盖，不只按 issue 标题把 fn_name 漏失另立新修复；未执行代码评分器、下载数据或复跑作者 6 项 CPU/30 题报告 |
| [#7771 padding 稀释序列诊断](https://github.com/verl-project/verl/issues/7771) | 另一贡献者 rrrxxx0510 的 [#7785](https://github.com/verl-project/verl/pull/7785) OPEN/non-draft，head `7ffe2fa4a4f5581c27e09b8f2e438e3b61d8e7ee`；正文明确 Fixes #7771，仅调整全 mask 行的 per-sequence 指标 | 不是新空缺，不把指标修复解读为 IS/RS、loss 或模型质量修复；未复跑 reporter 的 52 项或 PR 作者的 31 项测试 |
| [#8334 teacher-generated rollout](https://github.com/verl-project/verl/issues/8334) | 同一作者 lbaolin 的 [#8335](https://github.com/verl-project/verl/pull/8335) OPEN/draft，head `8029bfb5fc5b8b6034704f9c062610613f1bf05c` | 延续已核归属，不重复实现；draft 仍待 placement/callback 反馈，不把作者 GPU 报告当本项目验收 |

本轮未验证主线问题是否仍可复现、两份候选代码的全部边界或测试充分性；不新增 fork 补丁、Issue/PR/评论、依赖或 M5 算力事项。关键词搜索还命中其他 prime_code 工作，不能据同文件名就当同一问题；#8175 正文将 #8022 的“第十例后失败”与自身两项修复分开，本轮未独立复核 #8022 的当前状态或正确性。

### 判分与恢复问题的续核

2026-10-23 续读以下三个 issue 全文及全部评论（均为0），按编号检索 PR，再读取对应 PR 完整正文和 API。三个 PR 均 OPEN/non-draft；这里只判定已有作者工作覆盖，不将正文中的测试或硬件报告当作本项目结果。

| 问题 | 对应实现与声明范围 | 本项目处理 |
| --- | --- | --- |
| [#8123 boxed 判分归一化](https://github.com/verl-project/verl/issues/8123) | 同报告者 KardeniaPoyu 的 [#8124](https://github.com/verl-project/verl/pull/8124)，head `c5e26eec5ab5941e443d1b9f6853a9e210972487`；fallback两侧及pred归一化，strict保持原比较 | 2026-10-25本机原9项测试体候选全过／基线2过7失败，非完整套件；[详见专题](../../notebook/frameworks/verl-reward-integration-reading.md#boxed-答案的归一化与选择优先级)。不另修同一问题；Answer优先级与数学等价边界不变，作者完整套件／外部11项报告未复跑 |
| [#8076 FSDP dtype 未作默认](https://github.com/verl-project/verl/issues/8076) | 报告者 aryanyadav0402，另一贡献者 zupengwang 的 [#8083](https://github.com/verl-project/verl/pull/8083)，head `ba4b3f2d833dafe195ad9dc049d92a6a47b6f9df`；缺失或部分 mixed_precision 配置时回退 engine dtype，显式 param_dtype 优先，reduce/buffer 默认 FP32，model_dtype 仍管初始化 | 不是只剩文档/告警尚无人修；已有功能修复及说明，不另改默认。作者31 CPU/12 CUDA、world_size1报告不认证多卡、GRPO收敛或本项目精度结果 |
| [#7953 Megatron 调度器恢复](https://github.com/verl-project/verl/issues/7953) | 同报告者 YeonwooSung 的 [#7954](https://github.com/verl-project/verl/pull/7954)，head `53e632859898a248f89c8dd7e9abbf56e2248944`；正文声明两处加载恢复已有 scheduler state，flag 仍选择超参数来源并保留缺状态的严格断言；文档区分 Megatron optimizer 与 FSDP extra 的存放位置 | 不把报告者已说明的“extra与optimizer差异”另立相同修复。作者manager测试 mock底层加载，明确未验真实LR续接及Megatron-FSDP第二路径；缺验收不等于我方可重复提交实现 |

#8076 报告者只主动提出告警/文档选项，不代表另一作者的功能方案没有覆盖它。编号检索也命中不相关的已关闭 #7596，不能把所有搜索命中都当修复证据；上表以对应正文的明确关联为准。没有执行评分器、FSDP/manager、原测试或设备，没有本轮主线复现，也不为作者已列明的真实环境缺口另排 M5 队列；仅在出现具体协作需求、范围或状态实质变化后重新判断。

### 部分覆盖与当前状态

2026-10-24 读取两条 issue 全文、各自全部1条评论，再查编号引用 PR 并读完整正文/API。**旧评论不覆盖当前状态，存在关联 PR 也不等于整个问题已解决。**

| 线索 | 已核实的公开信息 | 本项目处理 |
| --- | --- | --- |
| [#8122 教师 jagged tensor 崩溃](https://github.com/verl-project/verl/issues/8122) | 同报告者 NGDBZ 的 [#8125](https://github.com/verl-project/verl/pull/8125) OPEN/non-draft，head `abc5cb9e5fa8cdf07ff5c1c86eb3190b18d4c644`，仅添加 is_nested guard；作者[更新](https://github.com/verl-project/verl/issues/8122#issuecomment-6029901088)和 PR 正文均明确：加 guard 后仍复现相同崩溃，深层原因未定 | guard 已有人做，不另提；不能把同一异常字符串、短 split sizes 或关闭 balance_batch 后越过旧失败步当作合成 padding 的因果证明。不建议据此普遍关闭均衡，也不直接认领尚未定位的根因 |
| [#8000 Megatron SFT 异步保存](https://github.com/verl-project/verl/issues/8000) | 唯一旧评论称准备关闭，但当前 API 是 `state=open/state_reason=reopened/closed_at=null`。同报告者 ewan0x79 的 [#8003](https://github.com/verl-project/verl/pull/8003) OPEN/non-draft，head `7bd75a88a5d2acfa723f15ac299fd6ee074faba9`，声明覆盖 SFT queue progress、完成后发布和退出 drain；同作者 [#8056](https://github.com/verl-project/verl/pull/8056) OPEN/non-draft，head `ce5aec5948a6f8981cb6086987aa13e0d8d518fb`，声明叠在 #8003 上处理 RL actor/critic、tracker 与 retention | 不按旧评论标成已关闭/撤回，也不把两个 PR 当相互竞争或完全独立差异。已有作者推进，不重写 queue/drain；声明完成范围与真实 save/restore 验收分开 |

#8122 的初帖把 guard 称为修复，后续已明确收窄，不能只引用首段“根因”。#8003 正文另将 epoch-boundary 数据游标问题归于 #7974；本轮未核该 PR，不扩大成“恢复训练全部正确”。#8056 作者报告的 GPU 运行是回移到 verl0.8 的改动，不是当前 head 的原生训练验收；其25项CPU报告也未在本项目复跑。这里没有执行 jagged tensor/分布式保存、评分器、原测试或设备，也未逐行复审候选代码，不新增 GPU/NPU 问题或 M5 队列。后续只有出现可核实的独立缺口或协作需求才重新判断，不用更多替身堆出“修复已证实”。

### 资源默认与更新调度

2026-10-24 读取两条 issue 的完整正文、全部评论（分别2条、1条），按编号与配置关键词检索 PR，再读取相关 PR 正文/API。**默认资源策略、固定更新次数、UID 分组和损失分母是不同契约；已有工作也不代表问题已完全解决。**

| 线索 | 已核实的公开信息 | 本项目处理 |
| --- | --- | --- |
| [#7398 TransferQueue 默认单元数](https://github.com/verl-project/verl/issues/7398) | 同报告者 nataliekung 已有 OPEN/non-draft [#8340](https://github.com/verl-project/verl/pull/8340)，head `881fb96937f5ddfae5d894bb9a48538d1b049e1e`。正文声明 null 时在初始化前解析为 `2 * trainer.nnodes`，保留显式值，新增 Hydra 3项测试及生成配置；另一作者 liuhao-labs 的旧 [#7525](https://github.com/verl-project/verl/pull/7525)是 CLOSED/unmerged，不是已合入修复 | 由现有作者推进，不重写同一默认。报告者[更正](https://github.com/verl-project/verl/issues/7398#issuecomment-6347916344)：最初1节点/16CPU是算例，后来另做 Ray2.59.0/TransferQueue5fe4e49 CPU-only复现；不能把初帖算例说成完整训练卡住实测，也不能把按节点数取默认等同于CPU资源准入或有界初始化 |
| [#8099 多输出固定更新次数](https://github.com/verl-project/verl/issues/8099) | 报告者 luyuzhe111 已发布下游实现，限定 V1 actor-only、无distillation、`seq-mean-token-sum`；eteluna 已[公开说明有原型](https://github.com/verl-project/verl/issues/8099#issuecomment-6254265298)，涉及静态micro-batch倍数、padding分布、拒绝shuffle，损失分母仍待定。[#8201](https://github.com/verl-project/verl/pull/8201) OPEN/non-draft，FZYsheep、head `bb057a8c0cedabacadeafc3b1d1b9318d1b59517`，正文声明V1按UID分组与分单元padding/均衡；[#6025](https://github.com/verl-project/verl/pull/6025) OPEN/non-draft，guillemgt、head `e827294d97d5020e17c9b0cd8681e75e5bdbe2aa`，面向旧同步trainer分组 | 不据 `fixed_count` PR关键词搜索空结果认定无人实现，也不标成 #8201 已完整解决 RFC。固定配置步数与按实际materialized UID分组不同；#8201只有每个prompt产生轨迹或启用失败组refill等条件下才维持原定步数。保持同UID不跨步、每步分母选真实行或配置rollout数，也不能互相替代 |

#8340 作者报告3项配置测试和上述CPU复现；#8201作者报告26 passed/10 skipped，#6025作者称做过多输出训练，本项目均未复跑或逐行复审。旧#7525正文还提醒恢复checkpoint须保留原storage-unit数，这属于旧方案的兼容说明，不当作新PR已完成恢复验收。这里不判断三种batching方案谁的训练质量更好，也不搬用RFC中的MigrationBench收益数字；本轮不执行Ray/TQ/训练、创建补丁、对外评论或新增M5任务。只有出现独立缺口、具体协作或契约决策才重新投入。

### 选定 token 概率提案的范围收窄

2026-10-25读[#7712](https://github.com/verl-project/verl/issues/7712)全文／5评论及[旧fork draft #1](https://github.com/eteluna/verl/pull/1)完整正文／文件清单。作者10月3日已撤回组合器／插件／归约schema，剩默认off引擎输出，提议用标签概率与logit差相消而不改kernel；WenzheWang早先请求kernel切片不等已获确认。旧fork headf894f21e仍draft、拒绝启用selected与fused共存，不是新方案已发布。编号／字段上游PR查重空不当无人认领，不另实现。

见[原输出头专题](../../notebook/frameworks/verl-fused-output-softcap.md#选定-token-概率与共享归一化项)：一个固定FP64真实CPU数学对照核全词表值／hidden与weight梯度，detach锚点保持值却丢梯度；集合条件概率另有分母。不是作者原测试／kernel实现，脚本与JSON随仓库。main3c6d9b43四模块仅源码／语法，H200单算子报告与旧训练证据分开，不认证舍入、padding／SP、原地CE或FSDP；真机条件只补原M5参考，不新排队列／改专项或写上游。

### V1 异步空闲指标的统计范围

2026-10-25读[#7601](https://github.com/verl-project/verl/issues/7601)全文及三评论：hsusul已提出路由副本积分，报告者要求区分混合ROLLOUT与standalone，协作者说明当时RL-Insight时间线非两项数值。all-state编号搜索无PR；关键词命中已合并[#7373](https://github.com/verl-project/verl/pull/7373)混合借卡及旧fully_async工作，不当作#7601实现完成或空白可认领。

见[原异步专题](../../notebook/frameworks/verl-6790-async-trainer-reading.md#异步空闲指标与调度等待)：main3c6d9b43的gen包含采样钩子、switch/idle是布尔等待决策，router仅当前请求计数且共用混合／独立副本；不能直接推出GPU利用率或完整窗口空闲比。旧gen/step与版本尾段idle不能直接移植V1。11模块AST／compile，无原pytest、调度／服务或训练执行；不另写计量实现／合成积分器，真机观测条件只补原M5参考，不新增队列或上游动作。

### VeOmni 端到端低精度路线

2026-10-25接续：读[#8342](https://github.com/verl-project/verl/issues/8342)完整正文及两条评论；amenomori已按wuxibin89建议创建[VeOmni RFC #1360](https://github.com/ByteDance-Seed/VeOmni/issues/1360)并回链。新RFC完整正文／评论列表核实open、0评论，后端／verl分工及验收矩阵仍为待反馈提案；保留现有`qat_implementation`语义、迁移另议，不能把#8342的拟议替换块视作已批准API。不认领或另开实现。2026-10-24已读[#8186](https://github.com/verl-project/verl/issues/8186)全部8评论；当时编号PR搜索空、torchao搜索仅无关issue不代表无人推进；旧组织名查询422已用正确仓库成功重查，均非本次重新认证。

见[原量化笔记](../../notebook/frameworks/verl-quant-layer-audit.md#veomni-低精度计算与通信边界)：[#8330](https://github.com/verl-project/verl/pull/8330)已合并，仅接线节点内低精度梯度RS，不是参数AllGather或全部E2E计算；作者H100通信收益与H200 actor无加速报告分开，最终树未重跑CUDA。torchao v0.18.0原源码确有限制标准rowwise与float8 AllGather的guard，不能靠新配置名消除它。不另写统一配置、通信hook或QAT方案；只有明确协作需求/接口变化才重新评估。无本机量化、通信或训练结果，无上游写入，真机条件只补原M5量化参考而不另排队。

### 图片payload的既有修复

2026-10-24读[#7391](https://github.com/verl-project/verl/issues/7391)完整正文和唯一评论，确认同报告者leooop-al已开非draft [#7400](https://github.com/verl-project/verl/pull/7400)，读取时OPEN、head`dc721cfd3cbd4e18f49763f029d498519c74eb91`。编号/图片bytes语义查重同时命中其他传输方案，不能将所有命中当同一修复；#7400完整正文及两文件diff明确限定dataset侧，不改AgentLoop/DataProto/actor派发。

见[原数据流笔记](../../notebook/frameworks/verl-dataproto-data-flow-reading.md#图片字节与解码对象的生命周期)：真实Pillow配原方法/原五例，候选通过、祖先和当前main仅清理例失败；这不是完整pytest或RSS证据。已有作者覆盖字典浅拷贝/pop bytes/解码copy，不另写或发布同一补丁，也不将未处理的完整raw_prompt传输自动当我方新候选。#7391的path-only外部内存对照不认证#7400仍保留PIL的方案；旧本地“零拷贝/开销可忽略”及230KB算例已在原笔记纠正。无Ray/模型/设备执行或新M5任务，后续按具体协作/实质反馈再判断。

### Prime字典评分的重复覆盖

2026-10-24核[#8216](https://github.com/verl-project/verl/issues/8216)正文/0评论及编号查重，已有[#8217](https://github.com/verl-project/verl/pull/8217)、新[#8358](https://github.com/verl-project/verl/pull/8358)均OPEN/unmerged；旧#8262检索仍closed，不替作者提交、关闭或评论任何PR。新作者liangqianxing的head052e2fe8与报告者Virtuoso461旧headfe9ff514的完整生产文件仅注释/docstring差异，去docstring后AST相同；各自base的prime.py逐字节相同。见[原奖励笔记](../../notebook/frameworks/verl-reward-integration-reading.md#新编号与已有修复)。

新PR两函数测试增加manager写回位置的范围，使用固定tokenizer/恒定scorer，不能当真实判分器或全训练验收。本机没有运行新测试或复跑旧进程池；旧4例spawn证据不升格为新manager结果。新PR的两条评论为CLA/评审机器人、reviews API空，机器人“可合并”摘要不替代查重/维护者决定。本轮保留重复覆盖结论，不新增候选、补丁或M5硬件队列。

### 多轮 SFT 的已有修复

2026-10-24读取[#7807](https://github.com/verl-project/verl/issues/7807)全文及唯一认领评论，编号查重命中akushonkamen的非draft [#8197](https://github.com/verl-project/verl/pull/8197)，OPEN/unmerged、head`65195c8557d47a2c43774f31ee2f35f6325c7745`。完整正文、两文件diff和两条评论已读；机器人“Code Review Failed”没有具体代码意见，不能当补丁缺陷证据，也不根据CLA提示代作者操作。

见[原多轮分词专题](../../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#12-多轮-sft-的上下文与监督边界)：text-only reasoning会话的assistant改用上下文差分，前缀不等仅告警，原整段sanity_check及忽略开关不变；新增Qwen3例检查IDs与监督内容，Qwen2.5例比较旧路径，但只有一段五消息fixture。本机四次compile及源码核对，不搬用作者2+4测试通过报告、不将推理异常建立为训练因果或拿SFT补丁验收runtime CT。不另开发/发布修复，不新增M5任务；具体数据或实质反馈出现后再判断。

### FSDP head 生命周期的替代方案

2026-10-24读取[#8091](https://github.com/verl-project/verl/issues/8091)全文/全部3评论，已有非draft [#8096](https://github.com/verl-project/verl/pull/8096)与draft [#8097](https://github.com/verl-project/verl/pull/8097)均OPEN/unmerged。见[原输出头专题](../../notebook/frameworks/verl-fused-output-softcap.md#独立分片-head-的执行与梯度归属)：前者注册FSDP forward方法、保留独立head分片；后者在autograd apply前显式聚合并单独处理head每microbatch同步，是替代设计而非可叠加补丁。包裹FSDP pre/post不等于恢复所有普通Module hook，默认full_tensor也不声明Partial梯度。

两候选完整正文/生产diff、#8096原CPU和分布式测试与安装版PyTorch相关实现已核。13次compile非原测试或通信；作者144 case不乘rank，root-owned同kernel与unfused参考分开，不搬作本机验收。不重写同题修复、不向上游评论，真机缺口补原M5输出头参考，非新训练队列；#7991 softcap是另一语义层，旧数值验证不认证本题。

### 后端选择与蒸馏数值观察

2026-10-24读[#7540](https://github.com/verl-project/verl/issues/7540)全文及唯一评论，编号检索已有非draft [#7597](https://github.com/verl-project/verl/pull/7597)，OPEN/unmerged、head96ad10eb。读完整正文、7评论、2条review记录与当前五文件diff；维护者要求聚焦V1，当前已撤回弃用runner改动，旧21-launcher评论不当现范围。

见[原架构笔记](../../notebook/frameworks/verl-architecture.md#fsdp-后端配置的权威来源)：原完整config模块接真实OmegaConf七组，只验证无改写的冲突guard；类级顶层合法性、构造后engine赋值与数值问题分开。候选ahead1/behind24，不把API目标base当已含主线；作者19项/12 subtests不是本机结果。已有配置修复不证明OPD loss差异解决，不另写补丁、切后端或发上游评论，真机条件仅补原M5一致性参考。

## 本地论证撤回，不再独立推进

- **geo_mean/GSPO clip旧RFC（2026-10-21）**：withdrawn/not-pursued。论文GMPO Algorithm 1已有log空间裁剪；对称窗口下完整序列PG的36组限定值/梯度对照通过。旧“配置空间不一致”、GSPO token KL界及训练因果依据不成立；非对称映射反例不等于上游bug。保留[更正记录](verl-clip-semantics-rfc-draft.md)，不再列作待授权发布稿。分布式/engine等价仍未确证。

## 已被上游他人 PR 覆盖（2026-10-19 全候选复查重修正）

**第 5 轮初查重因限流空响应误报 0 PR，10-19 复查重（pulls API 验证创建时间）纠正：以下 4 个候选的覆盖 PR 均为 issue 发布当天/次日由 reporter 本人或他人开出。本地验证材料保留作参考（fork 补丁已还原），不再提交。**

| # | 候选 | 覆盖 PR（open，未合并） | 覆盖时间 |
|---|---|---|---|
| 3 | GMPO geo_mean 损失随 minibatch 切分变化 (#7913) | [#7916](https://github.com/verl-project/verl/pull/7916)与[#5614](https://github.com/verl-project/verl/pull/5614)：2026-10-21重核均open未合并；#7916同样agg_loss接入并加正分母guard/测试，旧“配置层 vs 我方agg_loss”判断撤回。旧DP/verified-works证据不足，见[更正](candidate-7913-gmpo-geo-mean.md)，不重复提交 | 同日 9-18 |
| 4 | 等长 batch 3D mRoPE position_ids 不一致 (#7899) | [#7901](https://github.com/verl-project/verl/pull/7901)（iceflysnow，rebuild jagged layout） | 同日 9-17 |
| 1 | Triton fused kernels `use_remove_padding=False` 崩溃 (#8089) | [#8090](https://github.com/verl-project/verl/pull/8090)（reporter kylemontgomery1 本人，triton fused output shape） | 次日 10-02 |
| 2 | Fused Megatron LM-head 遗留 AllGather pending (#8028) | [#8029](https://github.com/verl-project/verl/pull/8029)（reporter ewan0x79 本人，publish output weights before fused kernels） | 同日 9-27 |

> 注（2026-10-21纠正）：#7916与我方旧思路的核心同为agg_loss，不因BREAKING标题就认定实质不同；缺全局分母时的本地fallback并非可靠兼容修复。#7913不再默认存在差异评论空间。#8090/#8029的下述比较是历史记录，本轮未重核；任何发言仍须查重及用户决定。

以下候选本地已验证（patch + 单测），但上游已存在他人 PR 修同一问题（均未合并）。本地 patch 与竞争实现等价或更完整，但按 verl 贡献政策「PR 描述须说明为何不重复现有 PR」，不再提交。本地材料保留在 candidate-*.md，patch 保存在 verl-pr/verl_main/（未推送任何远端）。

| # | 候选 | 竞争 PR（open，未合并） | 对比结论 |
|---|---|---|---|
| 0 | process_validation_metrics 崩在 sparse None (#6830) | [#6845](https://github.com/verl-project/verl/pull/6845)（abinggo，6-25，metric_utils +17-1、2 单测）；[#7151](https://github.com/verl-project/verl/pull/7151)（keepkeen，7-26）；[#7588](https://github.com/verl-project/verl/pull/7588)（jmaicher，8-27） | #6845 与我们的 patch 思路相同（过滤 None + pred 对齐 + 单测）。其实现仅过滤投票路径（vote_pred_vals），我们额外同步 var2vals 全量一致；核心修复等价，评论价值低 |
| 1 | dynamic-cp ceil 切分致高 DP rank 空 (#6786) | [#6869](https://github.com/verl-project/verl/pull/6869)（ajinkyajawale14499，6-27，megatron_utils +5-3 + 新 seqlen_balancing.py + 66 行测试） | 与我们的 even 切分方案一致，还抽出了独立模块。无评论空间 |
| 2 | filter_overlong_prompts 多进程 SIGTERM 挂死 (#7163) | [#7164](https://github.com/verl-project/verl/pull/7164)（mikequan0425=reporter 本人，7-27，rl_dataset +34-5） | reporter 自己提交了修复（signal 处理思路）。issue 内另有 workaround 评论（datasets≥3.3.2） |
| 5 | resume 崩在 offload pin_memory cudaError (#6690) | [#7342](https://github.com/verl-project/verl/pull/7342)（shotsan，8-10，`_try_pin_memory` 降级 + 97 行单测，即 reporter 提出的方案）；[#8235](https://github.com/verl-project/verl/pull/8235)（linmuchuiyang，10-14，onload 优化器再 load，治内存耗尽之源） | 问题的两个半场（pin 失败降级 + 耗尽原因）各有一个 open PR，均未合并。无提交空间 |

## 已过期（verl 上游已修 / 已关闭）

| # | 候选 | 上游处理 | 备注 |
|---|---|---|---|
| 3 | del_local_ckpt_after_load 不删 checkpoint (#7213) | issue 已 close（2026-08-07） | 认领者 yuyz-cyber 处理后关闭；BLOCKED 状态解除后直接作废 |
| 7 | colocate reward model 路径缺失 (#6700) | 目标文件 `verl/trainer/main_ppo_sync.py` 已从 main 删除 | 2026-10-19 核验：trainer 目录重构为 main_ppo.py/main_ppo_v0.py；colocate 功能经 #4197/#7859 等在其他路径演进。issue 本身是问句（计划支持还是删除），无代码可修，作废 |
| 6 | [FSDP] Nemotron-H 无法开 gradient checkpointing (#7146) | 根因在 transformers 上游已修 | 2026-10-19 核验：transformers main `NemotronHPreTrainedModel` 已加 `supports_gradient_checkpointing = True`（modeling_nemotron_h.py:994），发布后 verl 公开 API 调用即可用，verl 侧兼容 patch 不再必要（reporter 声称已准备但 3 个月未提交）。作废 |
| — | 蒸馏损失 micro-batch 归一化 (#7200) | 594c51b (#7225) | 扫描当天就被修了 |

## 已拒 / 已弃

| # | 候选 | 状态 | 原因 |
|---|---|---|---|
| — | tool_parser.py 静默吞异常 | REJECTED | Phase ④ 复核：仅 3 处真静默（L404/469/527），14 处有 logger。3 行 debug 日志 + experimental/ 变动快 → 性价比 <10 不达标 |
| 4 | veomni transformer del_local_after_load 类型标注错误 | HOLD | 单行类型标注，攒着等同文件实质改动顺带修，不单独提 |
| 5 | [perf] pinned Transformers 慢处理长 prompt (#8261) | REJECTED | 依赖协调型：需升级 Transformers 5.12.1→5.15+，但与 pinned SGLang 0.5.20 冲突（0.5.22 需 TF 5.19+Torch 2.14）；reporter 明确"adopting processor fix without duplicating #8240 工作"、仍在等社区决策；无干净自修点，排除 |
| 6 | prime_math reward 误判：区间括号/100x/1/100x/差一/大整数给满分，正确等式给 0 (#8346) | REJECTED | reporter 自修（2026-10-23 核实）：[PR #8347](https://github.com/verl-project/verl/pull/8347) OPEN、REVIEW_REQUIRED，新增 135 行测试（ground truth 取自 PRIME-RL/Eurus-2-RL-Data 真实数据行），覆盖区间括号/100x·1/100x·差一/>2^53 整数精度/百分号四类；已有作者实现与测试，不另开竞争修复；测试数不能认证全部判分边界。修复点 `prime_math/__init__.py`（_str_is_int/_normalize）+ `grader.py`（is_digit），纯 CPU |

## 经验教训

- **2026-10-19**：**查重输出 "None" ≠ 0 PR，必须区分限流空响应与真实零**。第 5 轮扫描时 4 个候选的 search API 返回空（限流），脚本打印 `None` 被误读为"0 个 PR，无人认领"，导致 4 个候选深挖 + 验证 + 打补丁后复查重才发现全部被 reporter 同日/次日 PR 覆盖。教训：search 返回空必须重试 + 用 pulls API 交叉验证；"无人认领"结论必须基于可解析的 JSON。
- **2026-10-19**：**reporter 把 issue 写得很深（根因+机制链+修复方向）≈ reporter 已开或即将开 PR**。本轮 4/4：gss10282025（#7913→#7916）、iceflysnow（#7899→#7901）、ewan0x79（#8028→#8029）、kylemontgomery1（#8089→#8090）全部同日/次日出 PR。深挖前先按 reporter 名字查其 PR 列表。
- **2026-10-19**：**verl 的覆盖 PR 多为 reporter 本人自修**（4/5），与 6-8 月的"他人抢修"模式不同；同期还出现 #8237 这类无人认领 5 天的例外。判断竞争优先级时应先看 reporter 身份与 issue 深度。
- **2026-10-19**：**旧候选核验要把「文件是否还在 main」也查了**。#6700 指向的 main_ppo_sync.py 已从 main 删除，问题形态随重构消失——只查 issue 状态会误判为"仍开放待修"。核验顺序：issue 状态 → 引用该 issue 的 PR → **目标文件/代码路径在 main 是否存在**。
- **2026-10-19**：**issue 里 reporter 说"打算提 PR"不等于会提**。#6690/#7146 的 reporter 都声明了意图（分别 4 个月/3 个月），最终都是别人先修（#7342）或上游根因修复（transformers）。认领声明只能降低竞争优先级，不能当作长期占位。
- **2026-10-19**：**verl bug 候选的失效速度 ≈ 2-3 个月**。8-04 验证的候选 10 月全部失效；6-11/7-25 的 issue 到 10-19 也全部有主。verl + transformers 双活跃轨道上，"旧候选核对"的价值已耗尽，应转向全新扫描或换框架。
- **2026-10-19**：**VERIFIED-WORKS 后必须尽快请求授权并提交，不能等**。4 个候选 8-04 验证完成后停在「PR 待你同意」两个月，期间上游出现 5 个覆盖 PR（6-25 至 8-27 陆续被抢），1 个 issue 被关闭。本轮核验时全部已不可提交。verl 社区对 bug 的实际响应窗口 ≈ 1-2 个月。
- **2026-10-19**：**提交前查重要用仓库新名字**。verl 迁移至 verl-project/verl 后，旧 org 名的 search 直接返回 invalid（resources do not exist）；issues API 会 301 跳转但 search API 不跳。核验上游先 `repositories/<id>` 确认当前 full_name。
- **2026-08-03**：**`grep except | wc -l` ≠ 静默吞异常**。首轮扫到 tool_parser.py "79 个 bare except"就报了"17 处静默吞异常"；Phase ④ 严格复核（看每个 except 后第一非空语句）发现 **14 处都有 `logger.warning`，只有 3 处真静默**。这类信号必须逐个看 except 后有没有日志，否则会高估价值、浪费 Phase ④。
- **2026-08-03**：**必须读完 issue 正文再评 接受度**。首次扫 #7213 只看到"无 assignee/0 评论"就给了 接受度=5；读正文才发现作者 `yuyz-cyber` 明确写了 "I can submit a focused fix"（已认领意图）。接受度应从 5 降到 2，且需先协调归属再动手。
- **2026-08-03**：**单行类型标注 / 3 行 debug 日志不值得单独 PR**。这类 P1 太琐碎，maintainer 大概率觉得刷 commit。策略：攒着，等同文件有实质改动时顺带修。
- **2026-08-03**：**checkpoint manager 的删除语义要细看 issue 期望**。初稿以为删单文件（`os.remove`），issue 正文明确要删整个角色目录（`shutil.rmtree`）+ barrier 后 + 保留 `data.pt`。Megatron 侧 `os.remove(目录)` 本来就会失败——容易漏。
- **2026-08-03**：verl 不使用 `good first issue` / `call for contribution` 标签（均为 0 个）。issue 信号靠 reaction/comment 排序 + 人工读标题。
- **2026-08-03**：`experimental/` 目录（45 个 .py）是技术债集中区（121 个 TODO 中的大量在此）。但 experimental/ 模块变动快，改动容易被冲掉，A 分要扣。
