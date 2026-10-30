# UniRL — 腾讯统一多模态 RL 框架调研

2026-10-30接续：[Qwen3.5 MoE发布文件与接线](#qwen35-moe-的发布文件与接线)：#639新44f文档断言经官方wheel补核，0.1.11两生成模块都带坏mask关键字、0.1.12都移除；四文件各同对应tag。共用bundle按真实model_type条件导入，非三配方／模型或EP验收。

2026-10-30接续：[候选关闭与证据范围](#候选关闭与证据范围)：#632／#633／#636／#637由作者主动关闭、均未合并；#634已合并，#639仍开放。关闭原因属于作者优先级判断，旧CPU结果仍只绑定原快照，不把关闭、非空merge_commit_sha或新版文档当验收。

2026-10-30接续：[策略专属窗口检查与请求快照](#策略专属窗口检查与请求快照)核#633新795732cc：random不再因未使用init过大被拒，扫描／超长窗口保护保留；PE原replace会重验解析后索引。四组原模块／赋值控制，非所有引擎静默或训练结论。

2026-10-30接续：[VeOmni发布制品与入口](#发布制品与初始化入口)核新#639：0.1.12官方wheel的GPU／NPU mask产物及初始化入口与tag相符；0.1.13有tag但PyPI版本404且公共入口已改名。仅制品／AST绑定审计，不等Linux导入、mask数值或EP训练验收。

2026-10-30接续：[采样构造校验与修改](#采样参数的构造校验与后续修改)核#636：新增下界不等于新增类型／有限值保护，直接赋值与浅拷贝不校验、replace重跑继承字段；纯标准库原模块控制，非Hydra／引擎验收。#634最终已合并且改用elif，旧空列表兼容性结论限定历史head。

2026-10-30接续：[SFT空ID与评估随机键](#sft-空-id-与评估随机键)核#637：原Dataset五行由3个ID恢复5个，原评估抽样不再共用null键；去重守卫实际在Sample而非Part，不能据PR正文断言普通SFT训练必报错。仅CPU原定义隔离控制，非模型损失／质量验收。

2026-10-30接续：[AR 概率锚点的响应校验](#ar-概率锚点的响应校验)核已合并#631：按实际采样token取值、拒绝缺失／长度错误／非有限FP32，而非补零或整批丢锚点；20个本地wire输入与原容器操作只验证CPU数据契约，不等GSPO／HI3训练验收。

2026-10-30接续：[AR逐样本调用](#合并后的逐样本调用)核#630已合并：显式seed按AR身份拆call，未设seed保留原批处理；七组原helper标准库控制不是模型输出／副本独立性或吞吐验收。旧Omni未接seed的结论保留为合并前历史，不用于当前代码。

2026-10-30 接续：[奖励服务的测试收集与空结果](#奖励服务的测试收集与空结果)核已有#632：原文件快照的八组原生pytest collection区分6个误收集CLI节点、空收集退出5和缺路径退出4；排除误收集不是新增服务测试或训练验收。

2026-10-30 接续：[SDE 索引的两层边界校验](#sde-索引的两层边界校验)核已有 #633／#634：窗口上界与采样日程长度检查互补，但负起点和运行时返回集合仍须分开判断；14组原模块CPU控制不当引擎或训练验收。

2026-10-26 接续：[AR种子协作分工](#ar-种子配置与逐样本请求)中，报告者已让JianhaoYu-AI接手逐请求seed，engine副本偏移另议；八相关文件同旧分析，不将认领评论当实现完成或重复原CPU验证。

2026-10-26 接续：[奖励能力校验与模型加载](#奖励能力校验与模型加载)按 #621 最终合并版修正：managed 子进程也延后启动，实验 VideoAlign 增配置检查；加载幂等不等于整图回滚，旧版42项替身报告不迁为新版验收。

2026-10-26 接续：[Scheduler 子进程环境与依赖版本](#scheduler-子进程环境与依赖版本)核已有#468；环境恢复不等于真实动态库选择／启动，旧memory-saver假设须与实际依赖版本分开。

2026-10-25 接续：[SDE 窗口的停留步数与坐标](#sde-窗口的停留步数与坐标)核已合并#489：输入计rollout_id，阈值比较去噪窗口起点；原调度模块CPU序列不等于MixGRPO参考等价或训练提速。

2026-10-25 接续：[PPO 的轨迹边界与价值目标](#ppo-的轨迹边界与价值目标)核#86三个已合并分项；原GAE/ValueHead及value-loss函数CPU控制不认证完整trainer，loss mask也不是GAE轨迹有效位。

2026-10-25 接续：[前缀缓存的参数与失效窗口](#前缀缓存的参数与失效窗口)核#262与已合并#392：合法字段、缓存实际启用、真实命中和吞吐是不同证据；原过滤函数四组本地检查不是引擎启动或性能验收。

2026-10-24 接续：[视频时间步精度与比率诊断盲区](#视频时间步精度与比率诊断盲区)核已合并#438：9/10个FP32差异与2/10个双重舍入差异分开，小平均log-ratio和ODE尾段不认证完整轨迹一致，更不等于#412奖励问题已解决。

2026-10-24 接续：[样本观察拆分与音频通道](#样本观察拆分与音频通道)核#628基础层和#597接线层的新版本；原WAV函数及W&B输入表达式的CPU控制，不将旧GPU报告搬到新head。

2026-10-24 接续：[VeOmni 生成代码的 mask 接口边界](#veomni-生成代码的-mask-接口边界)核#627固定依赖的不兼容及上游已有调整；参数绑定成功不等于因果mask或EP训练验收。

2026-10-24 接续：[教师指纹与控制器失败传播](#教师指纹与控制器失败传播)区分已返回异常、权重内容不一致和读取卡住；固定主线6fd82cba，仅原指纹函数CPU控制，不是Ray/Gloo或训练验收。

2026-10-24 接续：[离线 embedding 的身份与索引发布](#离线-embedding-的身份与索引发布)以原store模块和真实CPU safetensors区分键覆盖、读取、resume元数据与训练身份；单次rename失败控制不是断电或共享存储验收。

2026-10-24 接续：[Judge 回复解析与失败状态](#judge-回复解析与失败状态)核原评分方法的七组本地响应控制；关键词匹配非语义判断，异常零分也不自动形成失败标志，非真实服务或训练验收。

> 目标：调研腾讯 `Tencent-Hunyuan/UniRL`——是否开源、支持哪些硬件加速器、社区活跃度、**外部 PR 合入可能性**（用户核心关切：社区提的 PR 他们是否接受，还是只自己开发）。
>
> 调研时间：2026-10-22（GitHub API 实时核验）。
> 状态：**加入技术掘金高优关注项**（2026-10-22 用户定）。

## 候选关闭与证据范围

2026-10-30 13:19Z只读复核PR API及完整timeline：以下四项均为作者294962silent主动关闭，`merged_at=null`。它们虽有非空`merge_commit_sha`，不能据此写成已合并；当前main仍为`b8e9740e981a68a5556fbeb8225558040a4d47a4`，对应真正已合并的[#634](https://github.com/Tencent-Hunyuan/UniRL/pull/634)（09:13:19Z）。下方早先的OPEN与CPU验证记录保留为对应快照的历史，不延伸到关闭时尚未核过的新源码。

| 候选与作者关闭说明 | PR关闭时间 UTC | 关闭时head | 作者给出的优先级判断 |
|---|---|---|---|
| [#632 奖励服务收集](https://github.com/Tencent-Hunyuan/UniRL/pull/632#issuecomment-6523631898) | 13:10:38 | bdeeb6af | 文档／测试入口整理，不是已在CI中失败的测试 |
| [#633 SDE窗口](https://github.com/Tencent-Hunyuan/UniRL/pull/633#issuecomment-6523633235) | 13:10:43 | 795732cc | 作者称所查六份配方未命中guard，属于预防 |
| [#636 采样参数](https://github.com/Tencent-Hunyuan/UniRL/pull/636#issuecomment-6523634446) | 13:10:46 | bfcfda53 | 作者称333个配方字段值均未触发新增下界，属于提前拒绝 |
| [#637 SFT空ID](https://github.com/Tencent-Hunyuan/UniRL/pull/637#issuecomment-6523635574) | 13:10:50 | 62514b42 | 作者称六份随附清单没有null ID，优先级较低 |

这些是作者缩小审阅队列的说明，不是维护者否定、合并或本机重新扫描配方的结果。#633关闭评论仍称越界窗口会在“每个引擎”静默退化为ODE；[PE原请求快照赋值的既有控制](#策略专属窗口检查与请求快照)却会因负索引抛错，所以不能照搬这一全路径概括。#637的重复ID异常也不能直接外推普通SFT必经Sample构造，见[原数据调用边界](#sft-空-id-与评估随机键)。关闭不撤销这些固定版本的技术观察，也不自动产生重开／替代PR或新增算力任务。

## 一句话结论

> UniRL 是腾讯混元团队出品的 **Apache-2.0 开源、几乎日更的「统一多模态 RL」框架**（统一 AR 自回归 + diffusion 扩散模型的 RL），**目前只支持 NVIDIA GPU（Linux x86_64，CUDA 13），不支持昇腾 NPU**。**外部 PR 合入可能性高**：最近 100 个已关闭 PR 合并率 94%，且明确接受了来自 CMU/NTU/个人开发者的社区 PR——不是"只有腾讯自己在开发"。

## 项目概况

- **仓库**：`Tencent-Hunyuan/UniRL`（Tencent-Hunyuan 组织，腾讯混元团队）
- **描述**：UniRL is a Framework for **Unified Multimodal Model Reinforcement Learning**
- **官网**：https://unirl-project.github.io/unirl/
- **规模**：1023 stars / 102 forks / 30 contributors / 107 open issues
- **创建**：2026-06-08（较新）
- **语言**：Python
- **注意同名项目**：GitHub 另有 `showlab/UniRL`（VLM RL）、`G-U-N/UniRL`（ICML 工具箱），非腾讯出品，勿混淆。

## 是否开源 —— 是（Apache-2.0）

- LICENSE 明确："UniRL is licensed under the **Apache-2.0**"（Copyright 2026 Tencent）。
- GitHub 自动 license 检测显示 `NOASSERTION`，但那是文件头非标准模板所致，实际许可是 Apache-2.0，开源无疑。

## 硬件加速器支持 —— 只支持 NVIDIA GPU（Linux x86_64）

- **INSTALL.md 原文**："Python **>=3.12,<3.14**, Linux x86_64, **NVIDIA GPUs**"。
- README 全文**无任何 NPU / Ascend / 昇腾 / 910 提及**（精确搜索确认）。
- 依赖的推理/rollout 引擎：**SGLang、vLLM-Omni、fastvideo、VeOmni EP**——均为 NVIDIA CUDA 栈。
- **CUDA 版本**：CUDA 13。安装文档面向 NVIDIA，不等于任意普通卡、显存或卡数都能运行每个配方；例如下述 HI3 双引擎路径另有每节点至少8设备的源码前置条件。具体型号兼容性与容量仍须实测。
- **与硬件关注不重叠**：不涉及昇腾/910C（区别于 DeepInfer 的 Ascend 支持）。

## 活跃度 —— 非常活跃

- **几乎每天提交**：最近 30 天持续高频 commit（10-19/18/17/16/15... 各类 perf/feat/fix/refactor）。
- **正式 release**：v0.1.0（10-07）、v0.2.0（10-08）。
- **Issue 活跃**：最近 30 天新增 33 个 open issue。
- **内容广度**：支持统一 AR + diffusion，覆盖 Qwen3、MiniMax-H3、Hunyuan Video 等多模型，引入 DAPO、GSPO、DRPO、FlowGRPO 等算法。

## 外部 PR 合入可能性 —— 高（核心关切，已用数据核验）

用户关切：社区提的 PR 他们是否接受？数据结论：**接受，外部 PR 合入可能性高。**

1. **合并率高**：最近 100 个已关闭 PR 中 **94 个被合并（94%）**，仅 6 个关闭未合并。
2. **大量外部作者被采纳**（非腾讯核心 `CjhHa1`）：
   - `ruiling-smartbear` = **CMU（卡内基梅隆大学）** 的 Ruilin Gao
   - `chi030303` = **NTU（南洋理工大学）** 的 Tianqi He
   - `Ideny42`、`MinhaoLi0318`、`LingYi-Liang`、`Dmaner`、`inaniloquentee` 等个人开发者
3. **不是"只有自己开发"**：核心维护者 `CjhHa1` 贡献最多（90），但 contributors 前 15 混着大量不同账号，社区 PR 高频被合。
4. **贡献门槛友好**：被合 PR 多为 `fix(...)`/`docs(...)`/`chore(...)`/`perf(...)`——**修复具体 bug、文档、小优化易被接受**，符合"修复具体 bug 比大幅优化更可能被合入"的规律。

## 定位判断（给后续使用）

- **硬件**：仅 NVIDIA GPU（Linux/CUDA 13），不涉及昇腾——和 910C/昇腾关注不重叠。
- **价值**：作为**统一 AR + diffusion 多模态 RL** 框架，范围比 verl 广；外部 PR 合入友好度高。
- **建议定位**：**技术掘金高优关注项**，偏「学习 + 潜在贡献」；因其硬件范围（NVIDIA only），不与 DeepInfer（Ascend 支持）在硬件上冲突，但与 verl（同为 RL 训练框架）可能有研究视角重叠。

## Agentic 奖励组的统计边界

**组内没有终局奖励差异，是诊断对象，不是已经证明的训练无效或可直接丢弃条件。** GBX-Max1220（Baixin Guo）新开[研究RFC #635](https://github.com/Tencent-Hunyuan/UniRL/issues/635)，2026-10-30核为OPEN、无评论。作者已有#621贡献，愿实现opt-in只读统计，先看有限奖励数、失败与组内方差，再研究其与held-out结果的关系；没有提出修改采样、奖励或优势。本项目只核源码前提，不另做同范围统计器或将RFC当已有实现。

固定主线 `5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8`，[AgenticTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/trainer/agentic.py)的三个集合并不完全相同：

| 轨迹／服务状态 | 评分与组基线 | 训练行 |
| --- | --- | --- |
| 无生成部分或terminal harness_status=failed | 不送评分，reward保留NaN占位，归一化忽略它 | 不进入 |
| 有限reward、普通轨迹 | 参与本组mean／std | 每个generated part都可进入，即使终局优势为0 |
| overflow且`mask_overflow_loss=True` | 仍评分且保留在本组基线 | 仅这里排除其梯度行；默认False不作此排除 |
| 评分服务缺rewards、数量不匹配或返回非有限值 | **直接抛错**，不是当作一条“缺奖励轨迹”继续 | 本轮不走正常训练接续 |

因此`failed`、有限reward计数、参与基线数和可训练轨迹数要分开；已有`agent/train_rows`还含多轮generated parts与DP补齐行，不是有效轨迹数。全组失败时归一化返回零，训练行为空的路径返回optimizer_updates=0；不能把这与“有有限reward但优势为零、仍送入stack”混为同一跳过机制。以上为原调用路径分析，未构造真实Sample、服务或stack。

[原`finite_mean_std`](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/types/advantages.py)在N=0时返回mean0／std1，N=1返回该reward／std1；多样本才用总体std。因此**normalizer的std1回退不是观测到的方差1**。诊断若直接复用该std判断“零方差组”，会遗漏singleton的退化含义；无有限样本的方差也不应补成一次实测。

[CPU函数控制](../../experiments/unirl-agentic-group-semantics/verify.py)使用完整原helper／`_group_advantages`与[clip surrogate](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/algorithms/base.py)，只延后注解并移除装饰器；组成员只是供len使用的占位，不冒充真实Sample。Python3.12.13／torch2.9.1／单线程CPU，[结果与四源文件hash](../../experiments/unirl-agentic-group-semantics/result.json)记录：

| 一组reward（NaN为失败占位） | normalizer mean／std | 输出优势 |
| --- | --- | --- |
| [NaN,NaN] | 0／1 | [0,0] |
| [2,NaN] | 2／1 | [0,0] |
| [2,2] | 2／0 | [0,0] |
| [0,2,NaN] | 1／1 | [−1,1,0] |

这些采用可精确表示的奖励，不把四个小输入推广成任意FP32同分值的逐位全零保证。另以两个真实CPU logprob叶子接原`grpo_clip_loss`：优势均0时，sum loss与对new_logp梯度均0，但`approx_kl`指标约0.0125。所核[AR GRPO](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/algorithms/grpo.py)将该surrogate归约后backward，k3等放在metrics，不是另一个KL正则损失；源码没有按“优势全零”提前跳过replay／backward。不能用非零KL日志证明奖励信号，也不能用零surrogate梯度推断完整optimizer是否更新——本机未运行optimizer或trainer。

分析上可先按N=0、N=1、N≥2同分、N≥2异分四互斥桶说明现象，再从中组合RFC所需比率，明确零方差的定义及分母。跨step聚合应保留组数／有限轨迹数，避免把不同规模step的比率无权平均；overflow屏蔽策略与generated-part／padding计数另列。频率与held-out结果的相关性仍可能受题目难度、失败机制和训练时序共同影响，不能由toy输入或同step相关性解释成干预收益。这些是基于调用路径的分析建议，未落地计数器、采集真实数据或改作者范围。

```bash
python3.12 -B experiments/unirl-agentic-group-semantics/verify.py /path/to/unirl-git
```

固定Git blob须已缓存，运行禁止隐式fetch；无GPU／NPU、服务、模型、原suite或训练频率／质量验收。本项到此收束，等待维护者反馈或作者实质实现再核；不扩控制矩阵、新排M5训练或对社区发布，任何上游／既有PR-head写入仍逐项待用户明确授权。

## AR 与 diffusion 的源码统一边界

2026-10-22 首次源码阅读固定官方 main `6fd82cba13f52f394747832344039408699ea378`。结论分两层：**普通 AR 与 diffusion 共享类型和机制，但保留不同 trainer；HI3 联合模型路径则进一步共享训练骨干、backend 和联合更新。** “统一多模态 RL”不能简化成所有模型、采样器和损失都走同一个循环。

| 层次 | 源码入口 | 统一与保留的差异 |
|---|---|---|
| 独立 AR | `train_ar.py` → `ARTrainer` | rollout 后取最后的生成 Part 交给 stack；可按算法是否需要优势决定 reward/advantage |
| 独立 diffusion | `train_diffusion.py` → `DiffusionTrainer` | 可积累多个 rollout 的末端 Part 后交给一次 train_track；有独立 residency/采样窗口逻辑 |
| HI3 联合模型 | `train_unified_model.py` → `UnifiedModelTrainer` | 传完整 Sample 给 `UnifiedModelTrainStack`，保留 AR 与图像的共同 prompt 树 |
| 共用机制 | `Sample` / `Part`、`StageAlgorithm`、Hydra `_target_`、placement/backend | 统一数据与接线接口，不代表相同 logprob、loss、设备生命周期或数值行为 |

入口分别见固定版 [AR](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_ar.py)、[diffusion](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_diffusion.py) 与 [unified](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_unified_model.py)。这些是不同入口构造不同 trainer，不是一个 mode 开关自动抹平领域差异。

### HI3 的样本树与信用分配

[`UnifiedModelTrainer._build_request_sample`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/unified_model.py#L205)从 P 条输入依次 fork N 个 AR 分支、每个再 fork M 个图像分支。driver 要求 `[input, ar_shell, image_shell]`，预期行数分别 P、P×N、P×N×M；两引擎路径先生成 AR recaption，再把每条 recaption 重复 M 次送入 DiT，最后填回原 shell。driver 保留原 shell 的 lineage，但接线与行数检查本身不认证引擎返回顺序和身份正确。

`train_step` 在 reward 后调用 `Sample.propagate_rewards(op="mean")`：它仅向尚无 reward 的父 Part 回传子节点均值，不覆盖已评分的父节点。随后 AR 按原 prompt 的兄弟分组算优势；M>1 时图像按 AR 父分支算自己的优势。**M=1 走显式共享优势路径**，检查两路 batch size 相等后将 AR advantages 赋给图像 Part，而不是让图像每个单成员组独立居中为0。[类型实现](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py)保留 parent/child ID、条件、segment、rewards 与 advantages；分组要求等分支、同父兄弟连续。

这里存在一处需要分清的证据：同版 [`hi3_vllmomni.yaml`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/examples/unified_model/hi3_vllmomni.yaml#L205)的旧注释仍称 N×M 是平坦 AR/图像对、真正两层需后续实现；但当前 driver 已显式构造两层及复用 recaption。**配方注释与 driver 接线描述不一致**；本轮只确认 driver 的构造、传参和计数检查，不据此宣布真实 engine 已正确保留全部身份或生成语义，也不自动修文档/提 PR。

### 两路损失怎样进入同一次更新

trainer 将同一 bundle 注入 pipeline 和 backend，再把同一 pipeline 注入两路算法。HI3 的 [`pipeline._assemble`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/hunyuan_image3/pipeline.py#L101)把同一 bundle 给 AR 与 diffusion stage，两者的 `trainable_module()` 都返回 `model.transformer.model`。配方选择 AR 的 GRPO 与图像的 FlowGRPO，配置单份 LoRA；不是用两个同配置但独立的模型分别训练后再合并。

[`UnifiedModelTrainStack._train_one_step`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/unified.py#L145)顺序为 `zero_grad → AR backward → image backward → optimizer_step`。两路仍分别计算 loss scale/微批损失；只在至少一路报告 backward 时尝试一次 backend step。`train_track` 按 `num_updates_per_batch` 重复这个联合更新（默认1），所以“一次更新共同累积”不等于“任意配置下每 rollout 永远只有一步”，也不保证两路梯度同权或训练收益。

### 本轮检查与真机边界

只读 clone 后读取了仓库 AGENTS、入口、trainer 接线、联合 stack、Sample/Part 与 HI3 stage 共享对象；未安装 UniRL、下载权重或导入训练模块。Python 3.12.13 运行原静态依赖方向检查 `python -B lint/check_core_dependencies.py` 得到 `ok`；四个核心文件 AST/内存 compile 通过。该 guard 检查指定层的静态 imports、跳过 TYPE_CHECKING，不认证动态导入、服务或训练执行。

四文件 SHA256 可用标准库复查，不需要任何设备：

```bash
python -B - <<'PY'
import ast, hashlib
from pathlib import Path
for name in ('unirl/train_unified_model.py', 'unirl/trainer/unified_model.py',
             'unirl/train/stack/unified.py', 'unirl/types/sample.py'):
    data = Path(name).read_bytes()
    compile(ast.parse(data, filename=name), name, 'exec')
    print(name, hashlib.sha256(data).hexdigest())
PY
```

```text
train_unified_model.py   4b0c65de767bc8f55dbe55ae13f4004d9753d1111df4d07e08f6a005bf7cc19a
trainer/unified_model.py f9b47b971380deb854491b17ba3fa2cb6e4c80dcbf75e0d56541c2fa3cb3ab1c
train/stack/unified.py   d2da53dea5143e98f0228465cbde5cc90f014a3c6bf3b123de658354dfe95028
types/sample.py         04c7a8caa840953e16fea401676ad30f4f1080fdf494fcf1daf095a15d78a460
```

HI3 双引擎构造器明确要求每节点至少8设备，batch_size 为 num_devices 的正倍数；这不是已完成的资源测量，也不能外推为所有 UniRL 路径都至少8卡。此配方的真实两层 rollout、共享参数更新与同步均未执行，资源条件已记入 [M5 接续](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不据安装说明给单4090或昇腾承诺支持。

## 稀疏 SDE 与轨迹状态位置

**选3个 SDE 转移，不是只做3步去噪，也不是只保存3份 latent。** 2026-10-22 沿用固定 `6fd82cba13f52f394747832344039408699ea378`，直接正常导入原 [`types/sampling.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sampling.py)、[`sde/index_schedule.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/sde/index_schedule.py)及 require 模块。无 AST 摘取、包壳或替身；这些 leaf 模块只需标准库和 NumPy，本轮没有导入 torch/Ray/Hydra/OmegaConf。

与 HI3 配方相同，设总步数16、`timestep_fraction=[0,0.5]`、`num_sde_steps=3`。AllSDEScheduler 的候选池是半开区间 `[0,8)`，每个 rollout id 经独立 NumPy generator 无放回选3个。64个 id 的原函数输出均满足范围/数量/去重，逆序重查完全一致，NumPy 全局 RNG 状态未变；不要求不同 id 必选不同集合。当前 NumPy 2.0.2 下 rollout0 选 `[4,5,7]`。

`DiffusionSamplingParams.get_sde_indices` 的优先级是**显式列表（包括空列表）→ scheduler → 全部 num_inference_steps**。因此省略列表和 scheduler 时得到0至15；显式 `[]` 即使配了 scheduler 也返回空，`[1,2]` 则返回该列表。AllSDEScheduler 本身不设 num_sde_steps 时返回过滤池的全部8步，设0返回空；原构造器拒绝负数或超过池容量的请求。这是 helper 的解析契约，不是说直接传原始 None 给 stage 也会全选。

标准 [`DiffusionTrainer._build_request_sample`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/diffusion.py#L687)先解析列表，再以 `dataclasses.replace(..., sde_indices=..., scheduler=None)` 固定到本 rollout；unified trainer 同样如此。HI3 与 SD3 的 `diffuse` 则读这个已解析列表，在 `for i in range(T)` 中逐步设 `step_eta = params.eta if i in sde_set else 0.0`。所以源码仍有全部 T 次去噪迭代，非选中步走 eta0；这不等于声称恰好 T 次模型前向或量化了运行时间。

原 `compute_trajectory_positions` 给每个选中转移 i 保留相邻位置 i 和 i+1，再排序去重。HI3/SD3 采样循环还显式 `needed.add(T)`，保证最终状态可用于后续解码。下表端点列为原 helper 实测，最后一列仅按这条源码规则推导，未生成任何 latent：

| SDE 索引（T=16） | helper 端点 | 加终态后的预期位置数 |
|---|---|---|
| `[]` | `[]` | 1，仅位置16 |
| `[0,1,2]` | `[0,1,2,3]` | 5，包括16 |
| `[0,2,4]` | `[0,1,2,3,4,5]` | 7，包括16 |
| `[14,15]` | `[14,15,16]` | 3，终态不重复 |
| 全部0至15 | 全部0至16 | 17 |

穷举前半段8步的全部56种三索引组合，原 helper 端点均与独立集合表达式 `S ∪ {i+1}` 一致；加终态后5/6/7个位置分别有6/30/20组。原因是相邻转移共享中间状态，不能把存储位置数固定写成 SDE 数或2倍 SDE 数。[LatentSegment](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/segments/latent.py)也分开保存完整 sigma 表、状态位置 indices 与 sde_indices，三者不是同一轴。

空选择不是“自动继续原 FlowGRPO 训练”：[`FlowGRPO.compute_loss_and_backward`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/algorithms/flowgrpo.py#L116)对空 sde_indices 直接报告 has_backward=False；有选择时仍要求实际 transition logprob／replay 契约匹配。这里只核源码，没有执行 algorithm 或以索引个数保证 logprob 张量可用。

核心复查命令（在固定 UniRL 源码根目录，Python3.12.13 / NumPy2.0.2，无安装）：

```bash
python -B - <<'PY'
from collections import Counter
from itertools import combinations
import sys
import numpy as np
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.sde.index_schedule import AllSDEScheduler
sched = AllSDEScheduler(16, [0,0.5], 3)
params = DiffusionSamplingParams(num_inference_steps=16, scheduler=sched)
before = np.random.get_state()
chosen = {i: params.get_sde_indices(i) for i in range(64)}
assert all(len(s)==3 and s==sorted(set(s)) and set(s)<=set(range(8)) for s in chosen.values())
assert all(params.get_sde_indices(i)==chosen[i] for i in reversed(range(64)))
after = np.random.get_state()
assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
assert DiffusionSamplingParams(num_inference_steps=16).get_sde_indices(0)==list(range(16))
assert DiffusionSamplingParams(sde_indices=[], scheduler=sched).get_sde_indices(0)==[]
assert DiffusionSamplingParams(sde_indices=[1,2], scheduler=sched).get_sde_indices(0)==[1,2]
hist = Counter()
for group in combinations(range(8),3):
    ends = compute_trajectory_positions(set(group),16)
    assert ends==sorted(set(group)|{i+1 for i in group})
    hist[len(set(ends)|{16})] += 1  # consumer 的 +T 规则，非执行采样
assert hist=={5:6,6:30,7:20}
assert not any(n in sys.modules for n in ('torch','ray','hydra','omegaconf'))
print(chosen[0], dict(sorted(hist.items())))
PY
```

三原文件 SHA256 分别为 require `d30f89e77ff72e9dddaaf8b8f0ab403790b3e2984fbd6afea28f6bcc60177238`、sampling `95325713e5f9d426ef56a2b84c0d7a2f701f0cd7c3a0f7f76b728d420a4f9ddc`、index_schedule `4a313b44df9275b8e470a1dd1226d3b835e5170eb4a707ab162c269fe307f3c4`。上面的核心复查及额外首尾/空/全步端点、0/全池和非法计数控制均通过；这是自编原模块检查，不是上游原测试。未执行 stage、engine、latent 构造、replay/backward 或性能/显存实验；位置数量不能外推节省比例、跨引擎数值一致或质量。需要真机时沿用 [M5 原条目](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不新增必跑矩阵。

## SFT 预取与数据消费游标

**预取下一批不推进消费游标，但消费当前批也不等于 optimizer 已成功更新。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，用真实 JSONL 读取原 [`SupervisedDataSource`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/data/sft.py#L216)验证了前半句；后半句来自 trainer 调用顺序，不是模型故障实测。

`get_samples` 立即推进 epoch/position；`peek_next_batch` 暂时取出下一批与取完后的状态，再恢复公开游标，将结果留在 pending；`commit_peeked_batch` 才安装 pending 状态。state_dict 不保存 pending 内容，因此在预取后、commit 前记录状态，恢复时会重新读到该批，而不是跳过去。

5行训练数据 r0～r4、batch_size3、shuffle=False 的实测：

| 操作 | 取得的样本 | 公开 epoch / position |
|---|---|---|
| 直读第一批 | r0,r1,r2 | 0 / 3 |
| peek 下一批 | r3,r4,r0 | 仍0 / 3 |
| commit pending | 同一批 r3,r4,r0 | 1 / 1 |
| 从 peek 后保存的状态另建 DataSource 恢复再读 | r3,r4,r0 | 1 / 1 |

原实现对 pending 时再次 get、再次 peek，以及无 pending 的 commit 都抛 RuntimeError。shuffle=False/True 各10批与不预取的直读序列、游标逐批一致，跨越多个epoch，全局 Python random 状态不变；这是原数据源本地随机顺序的检查，不是分布式采样验收。

恢复还检查训练 manifest 的文件字节 SHA256 与 seed/shuffle。相同5行数量但内容变更、seed变更或shuffle变更，都被原 load_state_dict 拒绝；不是只核数据集长度。缺 fingerprint 的 legacy 状态则发警告后允许恢复，不能把兼容接受当身份已验证。另用3行验证集得到2+1两批，未改变训练游标或既有 pending。

[`SFTTrainer.train`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/sft.py#L201)先 get／commit 当前批，再 peek 下一批，然后才执行 train_step；循环没有按 result.optimizer_updates 回退数据游标。prefetch 的目的是准备下一批并保持保存游标指向已消费数据，不是 optimizer 提交协议。异常中断、非有限梯度跳步或“模型和游标共同恢复到同一步”仍需完整路径证据；本轮不将单个数据源控制说成 exactly-once 训练。

[复查源码与只读夹具](../../experiments/unirl-sft-cursor/readme.md)保存于本项目，不修改 UniRL：

```bash
python -B experiments/unirl-sft-cursor/verify.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1，真实原 dataset/normalizer/Batch/DataSource。仅跳过 tensor 包初始化器中与本题无关的 Ray transport 汇总导入，未替换这些类或摘取方法。全部观察通过，不是上游原测试；无 trainer/worker预取、模型、optimizer、磁盘 checkpoint 或设备执行。本题不新增M5真机事项。

完整源文件 SHA256：data/sft.py `45dc13244d15a0168919e48e33f126b6c9f259f02587007f4adcbc2c24ed782c`；trainer/sft.py `de66b86676f81422a832ce560942fd22e239b2052a00059b685631661ea3d2c9`。后者仅读取，没有执行；脚本也核对这两个hash，以免不同源码混用结果。

## SFT 空 ID 与评估随机键

**显式 `null` 不等于键缺失，样本身份也不只是日志标签。** [#637](https://github.com/Tencent-Hunyuan/UniRL/pull/637)在2026-10-30只读核为open、无评论／review，固定head `62514b42afbcee2672028c205c82554f4d160bf0`、base `b8e9740e981a68a5556fbeb8225558040a4d47a4`（#634合并版），只改一个文件。原 `str(item.get("sample_id", item.get("prompt_id", default_sample_id)))` 只对缺键回退；显式None会变成字符串 `"None"`。候选改为依次 `is None` 检查 sample_id、prompt_id，再取Dataset传入的文件名与行序号。修复不是 `value or fallback`：空字符串、0和False仍各自保留为 `""`、`"0"`、`"False"`，不负责通用ID清洗或全局唯一性；显式重复ID仍可冲突。

本机Python3.12.13／Torch2.9.1，真实临时JSONL读取原 `SupervisedDataset` 与两版原normalizer。11个直接输入中，仅sample_id为None、prompt_id为None、两者None及sample_id None但有prompt_id四项变化；最后一项正确回退到 `p7` 而非行号。五行清单的前三行sample_id为null、第四行显式ID、第五行缺键，结果如下：

| 原定义控制 | base | head |
|---|---|---|
| 五行ID | None、None、None、explicit、manifest.jsonl:4（均为字符串） | manifest.jsonl:0、:1、:2、explicit、manifest.jsonl:4（前三项同文件名前缀） |
| 不同ID数 | 3 | 5 |
| 隔离Part初始化守卫接空payload夹具 | 接受 | 接受 |
| 隔离Sample初始化守卫接单Part夹具 | duplicate sample_ids错误 | 接受 |
| 前两行评估sigma及整块noise | 完全相同 | 不同 |
| 整批与2＋3拆批的逐行抽样 | 精确一致 | 精确一致 |

**去重位置须与实际调用链对应。** PR正文将拒绝重复ID归到 `Part.__post_init__`，但固定版本的[原sample.py](https://github.com/Tencent-Hunyuan/UniRL/blob/62514b42afbcee2672028c205c82554f4d160bf0/unirl/types/sample.py)在 `Sample.__post_init__` 才检查非空／重复ID；Part的检查针对primitive形状与元数据。普通[SFTTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/62514b42afbcee2672028c205c82554f4d160bf0/unirl/trainer/sft.py)的 `train_step` 直接把builder返回的Part传给stack.train_track，`_build_part` 只比行数；[三个builder](https://github.com/Tencent-Hunyuan/UniRL/blob/62514b42afbcee2672028c205c82554f4d160bf0/unirl/train/sft/track_builder.py)也直接返回Part。不能用Sample的守卫证明普通SFT路径一定在构建时因重复ID失败；本轮没有构造真实模型或运行完整训练来判断其他消费者。

[FlowMatchSFT._eval_draws](https://github.com/Tencent-Hunyuan/UniRL/blob/62514b42afbcee2672028c205c82554f4d160bf0/unirl/algorithms/sft.py)为每行用 `sample_eval_seed(eval_seed, key)` 初始化独立Torch Generator，然后先抽sigma、再抽与该行x0同形状的noise。本机固定eval_seed42、五个 `[2,3]` CPU latent占位，旧前三行共享seed `7275843180089811391`，sigma均0.7539501190，noise首元素均0.4883066416，且整块noise相等；新前三行seed不同，sigma分别0.6647263765／0.5436111093／0.6421657801。64位seed保存为十进制字符串。相同ID与形状的抽样仍与microbatch拆法无关，错误在行身份被合并，而非拆批算法失去确定性。共享抽样会改变固定评估的随机配对并造成行间相关；**不能仅凭同噪声推出评估均值的期望有偏、偏差方向或训练质量损害**。

[脚本](../../experiments/unirl-null-sample-id/verify.py)通过AST选取完整原定义，跳过包导入；媒体类型、normalizer和Dataset保留原实现，但guard接SimpleNamespace的空payload／单根Part夹具，FlowMatchSFT配置也是夹具，不是原生Part／Sample实例或算法构造器。两个新进程的完整[结果](../../experiments/unirl-null-sample-id/result.json)一致，含六份完整输入hash；没有执行原生pytest、模型前向／损失、distributed stack或GPU。此题在数据身份与随机键层收束，不新建竞争修复或算力任务。

复现：`/path/to/python-with-torch -B experiments/unirl-null-sample-id/verify.py /path/to/UniRL`，需上述固定Git对象已缓存；脚本禁止lazy fetch，不自动安装依赖。

## Packed 文本的样本轴与 token 轴

**packed 文本按 token 存储，重排却按样本选择；零 token 的样本仍是一行。** 固定 `6fd82cba13f52f394747832344039408699ea378`，2026-10-22 在真实 CPU 上运行原 [`TextSegment`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/segments/text.py)和 [`Batch`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/tensor/batch.py)，没有重写方法或用容器替身替代。

`TextSegment` 的 tokens/log_probs/loss_mask 是 PACKED 字段，平坦轴长度为各样本 token 数之和；status 是 CONCAT 字段，按样本行排列。`pack` 将每个 packed 字段的逐样本 tensor 列表拼成一维 tensor，同时建立 `[N+1]` 的 cu_seqlens，lengths 由相邻 offset 差得到。正确构造不只需要一个平坦 tokens 数组，还要保存样本边界。

三行长度 `[2,0,3]` 的实际结果：

| 操作 | 样本数 | tokens | cu_seqlens |
|---|---|---|---|
| pack | 3 | 100,101,300,301,302 | 0,2,2,5 |
| select `[2,0,2]` | 3 | 300,301,302,100,101,300,301,302 | 0,3,5,8 |
| slice `[1:3]` | 2 | 300,301,302 | 0,0,3 |

`select` 可重排、重复或选择空集；它按原 offsets 取每行 token 段，再重建 offsets，而非对平坦 token 直接应用样本索引。`slice` 同时截 token 范围并将 offset 起点归零。concat 时后续 offsets 要加上之前的 token 总数，不是样本数；相邻切片拼回后，tokens、log_probs、mask、status 和边界均应恢复。重复选择也真实重复 payload，不是共享前缀压缩。

[复查脚本](../../experiments/scripts/verify_unirl_packed_text.py)用4种三行布局 `[2,0,3]`、`[0,0,0]`、`[1,2,1]`、`[0,1,0]`，逐字段核数值、dtype、长度和 offsets：36次选择（含permutation/重复/空集/CPU索引tensor）、40次合法连续切片、16个concat切点与4次chunk拼回全部通过。另在不提供 status 时，3个零长度 tensor 仍得到 batch_size3、cu `[0,0,0,0]`；零 token 行不能据 token 总数丢掉。

入口保护也有范围：pack 拒绝直接传已打包 Tensor，并要求不同 PACKED 字段的逐行长度一致；3行chunk(2)拒绝不能均分；故意绕过pack、又没有offset的普通构造对象不能执行packed slice。4项拒绝控制符合预期；不是说pack校验了所有字段语义，也不将违反构造前提的输入称作正常训练缺陷。

[`ARSupervisedTrackBuilder.build`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/sft/track_builder.py#L316)确实用 TextSegment.pack 构建 tokens/mask，token-budget planner 最终调用 part.select(perm)。这两处仅源码接线核对；本轮没有执行 Part、planner、tokenizer、attention mask、TensorRef transport、模型前后向或训练，因此不从容器一致性外推注意力隔离、prefix共享、梯度等价或性能。

```bash
python -B experiments/scripts/verify_unirl_packed_text.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1；只用一个真实__path__包壳跳过 tensor 包初始化器的 Ray transport 汇总导入，其余原类按源码加载。无Ray/Hydra/OmegaConf/worker-local transport导入、设备发现或环境安装。这是自编原容器检查，不是上游原测试；本题不新增M5硬件事项。

脚本核对完整文件SHA256：batch.py `a33122410e68c2295631bbc3e70ce4b69ba41081da7ed49abd06774640676e3c`；segments/base.py `f7b9ce0a6ae77bcdbde81a93b43096bdfe975368234753ef74ab6191d794ff32`；segments/text.py `9f58114b40f3949eea70fba0692fb96b746e3e47af0c94c7db99e314cf237b34`。

## AR 概率锚点的响应校验

**有 token 而没有对应 logprob，应在响应构造处拒绝，不能把缺失值伪装成概率。** [jasonchen505 的 #631](https://github.com/Tencent-Hunyuan/UniRL/pull/631)已于2026-10-29 06:15:04Z合并，最终head `e6651631cad409bbea4408da338b699d8b3783a3`、合并提交 `5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8`、父版 `ffad0b832707f5f1a87609e96f351426ede3f043`。10-30 03:21Z前后的只读API仍返回main为此合并提交，不能把它与PR head混称。[原 build_ar_segment](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/rollout/engine/vllm_omni/utils/tracks.py)只取每请求首个stage0的首个completion，构造`TextSegment`的tokens、log_probs、rollout_log_probs；不是遍历所有候选completion的概率验证器。

### 采样身份与缺失值分开处理

旧版用`next(iter(step.values()))`取字典首项。它可能属于别的候选token；在本地输入`{99: -0.125, 11: -0.75}`、实际token11时，旧版保存−0.125，新版按`choices.get(11)`保存−0.75。数值有限、长度一致仍不能证明概率对应正确身份。

旧版还会截长、补短／坏step为0；非空token行缺整个logprob列表时，`have_logp`可将整批两个概率字段置None。新版要求逐token的非空dict、采样token必须存在；在构造float32 Tensor后用`torch.isfinite`检查，因此有限Python float `−1e100`转为FP32 `−inf`也会被拒绝。logprob为0本身可以合法表示概率1，但无来源的0不是一个可用锚点。

| 本地输入 | 父版原函数 | 合并版原函数 |
|---|---|---|
| 两token只有一步概率 | 补成`[-0.5, 0.0]` | RuntimeError |
| 一token有两步概率 | 截成`[-0.5]` | RuntimeError |
| 非空token行缺None／空列表 | 两概率字段为None | RuntimeError |
| 非dict／空dict step | 补0 | RuntimeError |
| 仅有别的token | 取别的token概率 | RuntimeError |
| NaN／−inf／转FP32溢出 | 保存非有限Tensor | RuntimeError |

有效批中可以有空行。六行`[无输出, 非stage0, stage0无completion, 空tokens+None, 空tokens+[], 一有效token]`两版均保留batch_size6、cu_seqlens`[0,0,0,0,0,0,1]`；全部无token仍返回None。但“空token带非空logprob”新版报长度错误，不把所有空行无条件跳过。正有限值0.25仍被接受；缺logprob属性和无法float转换分别仍抛AttributeError／ValueError，故本项不是完整概率域校验或统一异常类型封装。

### 原容器对照与训练边界

[脚本](../../experiments/unirl-ar-logprob-contract/verify.py)在Python3.12.13／Torch2.9.1、单CPU线程执行20个自编wire输入，两版原`build_ar_segment`函数仅AST隔离，响应对象为明确的SimpleNamespace夹具。五份完整原模块（Batch、两Condition、Segment、TextSegment）通过只跳过包初始化的module壳加载，类与pack／concat／slice／select均未替换；这五文件在父版／合并版逐字相同，tracks在最终head／合并版相同。两个新进程[结果](../../experiments/unirl-ar-logprob-contract/result.json)一致：13类输入新抛RuntimeError，采样非首项取值纠正，其余正常／特定拒绝边界如上。

另用原容器核混合空行切片后concat完整还原，重复select`[5,0,5]`得到tokens`[11,11]`与cu`[0,1,1,2]`。独立控制中，旧版两行一行缺None导致整批log_probs为None；再与有效分片concat，原Batch报`TextSegment.log_probs: per-sample value is missing in items [1] but set in others`。这是实际CPU容器拒绝，不是仅检查传给pack的参数。

[GSPO](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/algorithms/gspo.py)调用的[prepare_ar_logp_anchor](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/algorithms/base.py)在log_probs=None时直接返回，GSPO loss入口也会早返回；此处仅源码，未运行replay／loss。响应校验不是重新计算概率、保证锚点来自正确权重版本或证明联合HI3训练成功。本机不是作者所报16项harness／原生pytest，没有真实Omni请求、adapter、Sample／Part、transport或设备训练；已有作者已合并，不另认领实现或新排M5算力。

```bash
python3.12 -B experiments/unirl-ar-logprob-contract/verify.py /path/to/UniRL-git
```

原Git对象须预缓存，脚本禁隐式fetch、不安装依赖；输出记录13份版本／路径源码SHA256与Git blob，供限定范围复查。本题CPU契约收束，后续仅随实质反馈重开，不扩充同义异常矩阵。

## 按域奖励的零分与失败标志

**0 分、评分失败与不适用的分量是三种不同状态。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，在真实 CPU 上直接调用原 [`PerDomainRewardScorer`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/local/per_domain.py)，配两个原 MCExactMatch 配置：graded_format_reward=True 与 require_answer_tag=True。它们用正则计算，不加载模型权重；successes 表示评分是否成功，不是答案是否正确。

四行有效请求得到以下原始响应；graded/strict 是本次配置的域名称，不是训练阶段：

| 行 / 域 | 生成文本（标准答案） | 主奖励 | successes | graded 分量 | strict 分量 |
|---|---|---|---|---|---|
| 0 / graded | A（A） | 0.5 | True | 0.5 | NaN |
| 1 / strict | A（A） | 0 | True | NaN | 0 |
| 2 / strict | `<answer>B</answer>`（B） | 1 | True | NaN | 1 |
| 3 / graded | `<answer>D</answer>`（D） | 1 | True | 1 | NaN |

严格配置下没有 answer 标签，得 0 分仍是成功评分。另加 unknown 域或缺 domain 的行，主奖励保留初始化的 0，但 successes=False，errors 说明没有配置对应 scorer；所有域分量均为 NaN。不能只看主奖励把失败解释成正常低分。路由器按域选取原请求行，再把响应散回原位置；排列 `[3,0,2,1]`、重复选择 `[3,1,3,0]`、空批和整体缺 metadata 的控制均通过。

分量的 NaN 则表示该域未为这一行评分，不应随意补 0 后平均。直接运行原 [`_tensor_stats`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/utils/wandb_metrics.py)：它过滤非有限值，四行有效请求的 graded 均值为 0.75，strict 为 0.5；故意先补 0 的对照分别变为 0.375、0.25。全 NaN 输入返回空统计。源码中的 rollout component 日志调用此 helper，但本轮仅执行 helper，没有运行 WandB、Part 汇总或训练日志链路。

**这不是可直接接入文本训练的配方。** 当前 router 的 preferred_input_kind 固定为 image，内层 MC scorer 是 text。本轮直接构造合法的文本 RewardRequest，验证的是 backend 路由契约；标准 [`RewardService.score_and_attach`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/service.py)在调用 backend 前先要求 frontier 含所需生成模态，因此不能由本轮推出它接受 text-only frontier。服务源码还在收到响应后对 False successes 抛错，不是静默把未映射域的占位 0 用于训练；这不等于宣称它拒绝任意 NaN。以上服务接线仅读源码，未导入或执行。

[复查脚本](../../experiments/scripts/verify_unirl_domain_rewards.py)保存在本项目，不修改 UniRL：

```bash
python -B experiments/scripts/verify_unirl_domain_rewards.py /path/to/UniRL /path/to/pillow-deps
```

Python3.12.13 / torch2.9.1 / Pillow12.0.0。本机原 CPU 环境缺 Pillow，使用 uv --target 仅安装到临时目录 `/tmp/perpetual-unirl-reward.D8w73Z/deps`；未修改原 venv。复现时需自行提供 torch/Pillow，已有 Pillow 可省略第二参数，脚本不会自动安装。Pillow 仅为原类型导入依赖，未构造图像或加载奖励模型。

两个真实 __path__ 包壳分别跳过 tensor 包的 Ray transport 汇总初始化和 local reward 包的重型模型 scorer 汇总初始化；其余 Batch/Texts/Request/Response/router/MC 类及统计 helper 均从原模块加载，无摘取方法或假 scorer。MC 的空 _load_model 方法会由原构造器调用，model 保持 None；base_device 参数也不代表执行过设备加载。无 Ray/Hydra/OmegaConf/Transformers、服务/HTTP、设备发现、trainer、反向或 GPU/NPU 执行。这是自编原模块控制，不是上游原测试；不据此定性训练缺陷或新增 M5 必跑事项。

脚本锁定七个完整源文件 SHA256（包括仅读的 service.py）。关键 hash：per_domain.py `6aca28149c4dcc48b3f8e29e74ef7195daa997aa833daac778618d4f3b2b73f8`；mc_exact_match.py `452413e2c7db3cf722fd3a887f660a61e5a13513f015a539f103b0a32d1cf526`；wandb_metrics.py `e92b9c40ebab4b78759ad15e36db4a833f853f6b122db4e395577966cc150648`。本题收束，不堆 NaN/路由变体，也不将组件统计等同于服务验收。

## 多参考图像的样本行与附件轴

**一行样本可以有多张参考图；增加参考图，不等于增加样本或生成候选。** [#478](https://github.com/Tencent-Hunyuan/UniRL/pull/478)的DyRef路径新增有序图像集合，2026-10-24读取open/non-draft，固定head `33756c23ec20da7c338602c07a0dce7c1738f187`、base/实际祖先 `272e0e6ea0d72e7a0c3e1be26434abd140980d42`。已有作者实现与训练报告，本节只研究原容器，不是完整PR审阅、替代实现或整条多模型路径验收。

[ImageSet](https://github.com/Tencent-Hunyuan/UniRL/blob/33756c23ec20da7c338602c07a0dce7c1738f187/unirl/types/primitives.py#L183)是一行的有序附件，内部使用原Images的packed像素与形状；metadata按图逐项对齐。[ImageSets](https://github.com/Tencent-Hunyuan/UniRL/blob/33756c23ec20da7c338602c07a0dce7c1738f187/unirl/types/primitives.py#L229)才是外层batch，`len`数行、`counts`数各行附件。[原契约](https://github.com/Tencent-Hunyuan/UniRL/blob/33756c23ec20da7c338602c07a0dce7c1738f187/unirl/types/README.md#ordered-image-sets)另将生成候选定义为fork出的独立行，而不是集合成员；本机没有执行fork。

Python3.12.13 / torch2.14.1 / Pillow13.0.0，完整原primitives、Batch及所需类型模块在CPU加载，只跳过tensor包的Ray transport聚合初始化，无方法摘取或假容器。用三行参考集合`[2,0,3]`、五张不同宽度的3通道小张量及逐图tag metadata，得到：

| 操作 | 原容器结果 | 单位与边界 |
| --- | --- | --- |
| flatten | 五图，owners=`[0,0,2,2,2]`，像素与原形状保持 | 附件轴映回样本行；空行仍在外层batch |
| select行`[2,0,2]` | counts=`[3,2,3]`，附件/metadata顺序保持 | 重复的是整行，且同一原ImageSet对象仍被引用，不是深拷贝 |
| slice后concat | 恢复`[2,0,3]`及原图顺序 | 拼的是外层行，不把其附件拍成新样本 |
| 两行各两图转to_slots | 第0槽取各行第0图，第1槽取各行第1图 | 从行内附件转成两个对齐的Images batch；混合counts会拒绝 |
| primary_images | 非空行各取第一图；有空行则拒绝 | 明确丢弃后续附件，不等同完整参考集合 |

legacy `Images`里两张图，经`as_image_sets`变为**两行各一图**；显式`ImageSets.from_rows([images])`才是**一行两图**。single-image消费者的`require_exactly_one`拒绝多图行，不能悄悄改用primary取图。原metadata数量不齐、rows类型错误也被拒绝；tag对齐不认证subject/style等角色在数据中的语义正确。

容器可表达变长，并不保证每个模型消费者都支持变长。[PR类型说明](https://github.com/Tencent-Hunyuan/UniRL/blob/33756c23ec20da7c338602c07a0dce7c1738f187/unirl/types/README.md#ordered-image-sets)将Qwen逐行处理与FLUX.2稠密槽区分，后者需要按参考图数分桶；本机只验证上表纯容器限制，没有运行编码器、这些模型分支、Part/fork、manifest加载、TensorRef搬运、奖励、Hydra或训练，也不借作者Qwen曲线认证未跑的FLUX配方。

复现代码见[当日日志](../../diary/2026-10-24.md)，不新增永久工具、竞争PR或硬件队列。固定primitives.py SHA256为`cbc7db341a9d7ed7608f862f40fa3d50926f1ae82e1c51a85ebb097aab4842c1`，Batch为`574cb025e85e8e183474e601d5e27fd50ca5a5967ed6c85db1dbca713dbbe871`。

## 可选提示与模式必需条件

**字段可以缺省，不代表每种评分都能缺它；某种评分需要文本，也不代表共享入口应强制所有模式提供文本。** [#497](https://github.com/Tencent-Hunyuan/UniRL/pull/497)将`RewardBackend.prompts()`从缺所选prompt返回`[]`改为抛ValueError，并删LLMJudge的一处空串回退。2026-10-24读取仍open/non-draft，head `104aad6e1f3cae2bd4461db6d59e99015e394415`、base/实际祖先 `c8af5d699c81fb02a0086e34b342b1efacc7d431`，三文件9增2删。[维护者10-19评论](https://github.com/Tencent-Hunyuan/UniRL/pull/497#issuecomment-6259340578)明确不倾向按现状合入：现有数据源已有文本祖先、全局加固会影响ImageBind非文本模式，且未完整处理变成死分支的下游回退。open状态不能当作方案已获认可。

固定主线`6fd82cba13f52f394747832344039408699ea378`的[RewardRequest](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/reward.py)允许original/generation prompt分别为None，但所有**已提供**字段须同行数。[RewardService](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/service.py)遇文本turn才填这两种语义；[ImageBind原方法](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/local/imagebind.py)则无条件先取prompts，再计算need_text。其默认audio_video模式不使用文本嵌入。因此“取prompt”不能自动解释为“该模式必须使用prompt”；这里是原调用路径的推论，不是已运行ImageBind模型的回归报告。

Python3.12.13 / torch2.14.1 / Pillow13.0.0，在原生CPU Audios、Videos和RewardRequest上，AST提取base、候选head、当前main的完整`prompts`方法，各给同一组六输入；self只提供prompt_source，不构造后端或模型：

| 请求与选择 | base与当前main | 候选head |
| --- | --- | --- |
| 合法一行音视频、无任何prompt；选original或generation（两项） | `[]` | ValueError |
| 两种prompt都在；分别选择original/generation（两项） | 返回对应文本 | 保持一致 |
| 只有original，却选generation | `[]`，不跨语义回退 | ValueError |
| generation存在，文本为`""` | `[""]` | `[""]` |

另给一行媒体、两行generation prompt，原RewardRequest在构造时就拒绝。候选检查的是None，不是空字符串、批大小或评分质量；已有长度校验也不保证模式所需模态齐全。输入契约控制与源码路径合起来支持维护者指出的兼容性风险，但不证明候选在所有配方可达，也未穷尽核验所有内置数据源。

复现代码见[当日日志](../../diary/2026-10-24.md)。原类型正常导入，仅跳过tensor包的Ray transport聚合初始化；没有执行ImageBind构造/预处理/模型、RewardService、Hydra或远端评分，也未复跑作者所述全路径检查。已有作者与明确评审意见，不另起竞争的prompt校验/回退修复或新硬件任务。head base.py SHA256为`89bcb5ba33d094a6ab5258175f4d74db8b846e608867391c6ba4318d1eccb833`；主线ImageBind为`ff1bdc783e85c505bff25e857834aae256d8d929c4ff8680d28c68edb55b8c0c`。

## 奖励能力校验与模型加载

**最终合并版已将 managed 子进程纳入延迟加载；下文初读时的“只延后本地模型”仅适用于旧 head。** 2026-10-26只读核 [#621](https://github.com/Tencent-Hunyuan/UniRL/pull/621)：GBX-Max1220 的实现于09:31:11 UTC合并，最终head `ceacde5ed0bb55a49d41d7d03b59656ebd87a594`，main为 `8cfbac17765004afb68fef08a4e011a4c8daf555`，相对base `6fd82cba` 共10文件、93增52删。两个提交的tree均为 `912365244d0c33d7b9992f26b374e0b4fc46fa80`，但提交祖先不同；不能拿GitHub三点compare的10文件当两棵树仍有差异。这是他人的已合并修复，不是我方贡献或待认领任务。

### 最终版新增的物化范围

[`ManagedScorerProcessBackend`](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/reward/managed_process.py)构造器只保存配置、准备HTTP客户端和锁，不再启动孩子。`load()`在RLock内调用 `_start_child()`，该方法创建监听socket／日志、Popen并等待匹配的health后返回地址；启动异常会调用 `_stop_child()` 再抛出。已有进程且 `poll() is None` 时直接返回，**并不会每次load重新验证health**。因此它与本地叶子 `_loaded` 成功标志是不同的幂等机制，不能把“进程还活着”写成“当前评分服务必然健康”。纯Remote仍继承no-op，由外部服务管理自己的模型。

[`RewardService`](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/reward/service.py)在prompt-video声明、输入种类与截断配置检查之后才调用backend.load；T2AV／PerDomain／Video继续递归转发。这个顺序依赖叶子与自定义wrapper遵守生命周期约定，不是对任意 `_target_` 构造器／模块导入副作用的隔离。直接构造scorer的调用方须先load，compute不会自动补上首次加载；不应把它描述成“第一条请求到来才加载”。[最终README](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/reward/README.md)现已明确这份调用契约。

另一个新增范围是[实验ReFL VideoAlign scorer](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/experimental/refl/reward/videoalign/scorer.py)：能力判断改读spec.w_ta；dataclass的 `__post_init__` 拒绝空reward_model_path、非正resize尺寸／micro_batch_size及非有限三项权重；offload/onload在model为None时返回。[wrapper](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/experimental/refl/reward/videoalign/wrapper.py)也不再把micro_batch_size静默夹到至少1。这里是实验模块的 `reward_model_path/w_ta` 接口，不要与内置 `unirl/reward/local/videoalign.py` 的 `checkpoint_path/ta_coef` 混用；字段通过不证明权重目录内容可加载。

**局部清理没有升级为全图事务。** managed新增启动异常清理，不等于T2AV／PerDomain会撤销先前成功加载的兄弟；它们仍按顺序调用孩子load而无整体回滚。Local的成功标志也仍不证明失败中途分配已回收。这里仅据控制流说明契约，未注入真实加载失败或判定新回归。

本轮Python3.12.13对最终10个变更文件中的9个完整Python模块做AST／内存compile，均通过；没有导入奖励包、执行构造器／loader、查询设备、启动HTTP／子进程或下载模型。Git partial-clone缺两个blob，默认代理失联、直连Git取对象失败后改用只读GitHub contents API固定ref获取；不是GPU连接尝试。最终关键SHA256：managed_process.py `eb5677bacfcae709d4569524fe6df9a613a737fcebab8ce462b784b904ed6000`，实验scorer.py `738640e8ee1f8723153dd9e6a075fec9677b1a00b3e85b3589ea0631389eb721`。此前42项fake-loader报告和本机12次compile均绑定旧head，不移作最终版模型、梯度、健康恢复或训练验收。原M5条目仅更新版本与条件，不新增实验队列。

### 初读版本与历史证据

**配置能声明一种评分能力，不等于模型已能加载；把加载延后，也不等于整个构造过程没有副作用。** 2026-10-22只读[#621](https://github.com/Tencent-Hunyuan/UniRL/pull/621)，读取时open、未合并，固定head `27cb49307fc89dd60331f86d23cadb2e8cc4b0cb`、base及官方compare祖先`6fd82cba13f52f394747832344039408699ea378`（ahead1/behind0），7文件40行新增。它将本地reward的模型物化从叶子构造器移到服务层校验之后，不修改奖励公式或配置项。

#### 递归构造为何让校验太晚

[Worker._resolve_init_kwargs](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/distributed/group/worker.py)先递归构造字典的children，最后调用该层_target_。T2AV是文本到音视频；其组合scorer需要至少一个正权重分量连接prompt与video。旧版ImageBind(mode=audio_video)+CLAP会先在两个叶子的构造器调用_load_model，再由T2AVCompositeScorer拒绝缺少prompt-video能力的组合。缺依赖时，甚至会先遇到模型导入错误，而到不了组合配置错误。这是错误发现顺序，不是原校验条件不存在。

| 阶段 | base中的顺序 | 候选中的顺序 |
| --- | --- | --- |
| 构造本地叶子 | 保存配置并调用_load_model | 保存配置，model/processor为None、_loaded=False |
| 构造组合与服务 | 此时叶子模型已尝试加载，再校验 | 先校验组合能力，再校验服务输入种类及截断配置 |
| 物化模型 | 已在叶子构造器发生 | RewardService末尾调用backend.load，组合逐个转发给叶子 |

变化见[LocalRewardBackend](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/base.py)与[RewardService](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/service.py)。这不是首次compute_rewards时自动懒加载：直接构造local scorer的脚本须显式load；服务构造成功返回前仍执行load。T2AV、PerDomain、Video三个wrapper新增转发，基类load本身是no-op，自定义组合不能靠继承自动遍历孩子。

#### 能力声明的适用范围

[ImageBind.covers_prompt_video](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/imagebind.py)按mode及text_video权重判断，[VideoAlign](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/videoalign.py)按ta_coef>0判断，均不需要先有模型对象。但RewardService只对实现PromptVideoReward协议的backend检查该声明；它不是任意奖励模型语义、checkpoint、依赖或评分质量的通用验证器。preferred_input_kind属于image/video/text也只验证种类名，实际frontier是否包含该模态仍在score_and_attach检查。

服务层require_prompt_video=False只跳过服务自己的检查，不撤销[T2AV构造器](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/t2av_composite.py)的正权重prompt-video要求。组合load遍历所有配置的孩子，既不按当前domain按需加载，也不跳过零权重项；延迟到统一时点不等于按实际请求节省模型内存。

#### 成功标志不是资源事务

LocalRewardBackend在_load_model正常返回后才置_loaded=True；后续顺序load可直接返回。这个标志不能由model is not None替代：无权重的MCExactMatch也有空_load_model。onload只把已有model移回device，不承担首次加载；offload也不清除_loaded。源码没有并发load锁，也没有失败自动回滚：某孩子已加载、后一个抛错时，组合循环不撤销前一个；失败叶子的False只说明未成功返回，不证明中途分配已释放。它提供重试入口，不是整图加载的原子性保证。

本PR不改变[ManagedScorerProcessBackend](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/managed_process.py)：其构造器仍_start_child，创建监听socket/日志并Popen，等待子进程health达到预期状态，然后才返回；load继承no-op。纯RemoteRewardBackend构造HTTP session、模型在外部服务；不能从“本进程没有model”推出“配置拒绝前没有启动子进程或远端工作”。本地叶子构造的resolve_device也仍可能查询CUDA可用性；这里延后的是_load_model，不是所有设备相关操作。这些文件与base的Git blob相同，是范围边界而非本PR新增行为。

#### 验证范围与复查

本机Python3.9.6只对6个修改的完整Python模块在base/head各做内存compile，共12次通过；另核worker、managed_process、remote、ImageBind、VideoAlign、device六文件Git blob前后相同。未导入奖励包、执行resolver/构造器/load、启动子进程/HTTP或探测设备，也未制造新模型替身。最小语法复查（对象须已在本地）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
for rev in ['6fd82cba13f52f394747832344039408699ea378',
            '27cb49307fc89dd60331f86d23cadb2e8cc4b0cb']:
    for path in ['base.py', 'local/base.py', 'local/per_domain.py',
                 'local/t2av_composite.py', 'local/video.py', 'service.py']:
        src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{rev}:unirl/reward/{path}'])
        compile(src, path, 'exec')
PY
```

head SHA256：local/base.py `b977c2075fa417c17a2f09e1dfa3c159ad1e2382977c112adf8f36955706d221`；service.py `4284399e73f5e01c6e8be65770b10acdfe6bbde3f2d80651b0b5a63762469cd2`。作者报告两种resolver的CPU harness基线24/42、候选42/42，所有_load_model都替换为fake，harness未提交；这是构造顺序/转发的作者证据，不是实际checkpoint加载或本机验证。有效ltx2配方在两版都通过，作者也明确它只是回归检查，不是修复区分项。

本题以初始化生命周期学习收束；真实依赖/权重及失败后的资源状态留[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，只在其未来独立选择匹配reward任务时考虑，不为本题新增加载或训练任务。

## Token 预算与 update 成员

**token_budget 约束拼批代价，不是显存保证，也不是逐样本截断上限。** 固定 `6fd82cba13f52f394747832344039408699ea378`，2026-10-22 用原 [`TokenBudgetPlanner`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/planner/packed.py)、Part、TextSegment 与 TextTokenCondition 做单进程 CPU 检查。规划器先按 num_updates 等分原行，再在每份内按总长度降序 first-fit 拼批，最后用 part.select 重排。原行不能被搬到另一份 update 来改善装箱；总行数不能整除 update 数时直接拒绝。

两种代价模型不能混用。sum 是各样本 `(prompt + response)` 之和；dense 是 `(max(prompt) + max(response)) × 样本数`。两行长度分别为 `(8,1)`、`(1,8)` 时，sum 为18，dense 为32，不是“每行总长最大值9再乘2”。预算18下，前者可同批，后者不可。这里是规划器的代价估计，不是 attention kernel 的实际分配量。

8行的 `(prompt,response)` 为 `[(8,1),(1,8),(2,2),(20,1),(1,8),(8,1),(2,2),(1,20)]`，num_updates=2、预算18。原条件 attention_mask 给出 prompt 长度，原 packed segment 给出 response 长度；实际输出如下：

| 代价模型 | 第一次 update 的原行分组 | 第二次 update 的原行分组 | 各微批代价 |
|---|---|---|---|
| sum | `[3] [0,1] [2]` | `[7] [4,5] [6]` | 21,18,4 / 21,18,4 |
| dense | `[3] [0] [1] [2]` | `[7] [4] [5] [6]` | 21,9,9,4 / 21,9,9,4 |

21-token 单样本仍被放进独立微批。first_fit_decreasing 只在尝试加入已有微批时检查预算，找不到位置就新建单样本批，不切割或拒绝超预算样本。因此“配置预算18”不能推出每批代价≤18，更不能推出不会 OOM。两种模型下原 sample_ids、文本、metadata、prompt 条件与 response tokens 均按相同排列对齐，每行只出现一次，原 Part 不被重排修改。

有有效逐样本长度时，将 micro_batch_size 从1改999不改变本例 token 规划。若 segment 不提供长度，planner 则警告并回退原 CountPlanner：本例8行、两次update、micro_batch_size3得到 `[0:3,3:4]` 与 `[4:7,7:8]`，此时不再按 token 预算拼批。读取配置名还不够，须确认实际输入提供了哪种长度。

[`_sync_micro_count`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/planner/packed.py#L93)在多进程组内用 MAX 对齐微批数量，必要时调用 balance_into_k；本次真实调用的是未初始化 process group 的原 no-op 分支，没有替换同步函数或启动通信。此处数量对齐不证明跨 rank 预算或梯度等价。[`TrainStack`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/single.py#L322)将外层 plan 分别交给 _run_update；实际 optimizer_step 还取决于 backward 等条件，计划两份不等于实测完成两次更新。这段训练接线仅核源码。

[复查脚本](../../experiments/scripts/verify_unirl_token_budget.py)：

```bash
python -B experiments/scripts/verify_unirl_token_budget.py /path/to/UniRL /path/to/pillow-deps
```

Python3.12.13 / torch2.9.1 / Pillow12.0.0，复用上一节隔离 Pillow 依赖，无新安装。脚本使用真实路径包壳，仅跳过 tensor 包的 Ray transport 汇总初始化和 stack 包的训练模块汇总初始化；原 planner 包、Part 与条件/segment 均正常加载，无假类、方法摘取或规划器改写。两种分批、成员/字段对齐、数量回退及三项配置拒绝控制通过；这是自编原模块检查，不是上游原测试。没有模型、训练、通信 collective、设备发现或吞吐/显存测量，不新增 M5 必跑事项。

脚本锁定七个完整源文件 hash，包括仅读的训练接线。packed.py SHA256 `b5f2789f768cbdb7d766bebb5c1d9feb316dbe32db52637e3683e8e6dc8fe70f`，types.py `d9278a269104724a63abae6eceedde694ad92b1016c415f1b88ed087dcb3e8be`；其他见脚本。超预算独占批符合原实现说明，不据此提出缺陷或自动改规划策略。

## EMA 衰减方向与更新时机

**EMA 的 decay 乘旧影子，更新频率与传入的计数同样影响结果。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，原 [`EMA`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/ema.py)执行 `shadow = decay × shadow + (1-decay) × live`，decay≤0 时直接复制 live。不能把小 decay 解释成“只混入少量新参数”；例如 EmaLoraConfig 默认0.001，对应新参数权重0.999，但何时混入还取决于 timing。

原 make_decay_fn 对 full EMA 使用 `min((1+t)/(10+t), target_decay)`，不是从第一步就固定用 target。target=0.5 时，t=0/1/10/100 分别为0.1、2/11、0.5、0.5。LoRA 配置另有 constant、linear 和 warmup：本次 uprate=0.1、uphold=0.5、flat_steps=2 的 warmup 在 t=0/1/2 都为0，t=3为0.1，t=7为0.5；零衰减阶段会复制当前参数，不是暂停 EMA。

CPU 对照使用原 inject_mirror 与一个仅含两元素可训练参数的 nn.Module，手动赋值模拟参数变化，没有执行 optimizer。初始 live 从 `[2,4]` 改为 `[3,5]` 后才执行原 apply_deferred_ops，影子复制到 `[3,5]`，说明初始复制是延后回调，不是在注册 shadow 时立即完成。原 mirror 只为 requires_grad 参数创建影子；本例冻结参数与 buffer 不在配对中，不能将它理解成完整 module 状态副本。

两份 mirror 从相同影子 `[3,5]` 开始，live 先后手动改为 `[13,25]`、`[23,45]`；均使用 full target=0.5：

| timing | 实际调用 | 最终 shadow |
|---|---|---|
| optimizer_step | step(0)，step(1) | `[21,41]` |
| rollout_end | 两次 step 无效，之后 on_rollout_end(2) | `[18,35]` |

第一路 decay 依次0.1、2/11，第二路只混合一次且用t=2的0.25；差异既来自混合次数，也来自计数。反向时机的调用不会更新，另用原 constant decay=0 验证 shadow 直接变为 live `[23,45]`。这些是 EMA 本体的实测，不是两次真实 optimizer 更新或完整 rollout。

[`BaseTrainBackend`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/backend/base_backend.py)源码中，grad_norm 有限时先 optimizer.step / scheduler.step，再以尚未递增的 _optimizer_step_count 调 EMA.step，随后计数加1；on_rollout_end 则传当前已提交计数。grad_norm 非有限时提前返回，不执行该次 optimizer/EMA.step 或计数递增。因此训练步时机与 rollout-end 时机不是同一个 t 定义；这里仅核对后端调用顺序，没有导入后端或制造梯度故障。

原 full mirror 的 use_shadow 用数据存储交换而非重新绑定 Parameter 对象，finally 再交换恢复。单层 CPU 上下文内主动抛异常后，live/影子数值、data_ptr 和 Parameter 身份均恢复，requires_grad 未改变、梯度仍为空。这不认证嵌套/并发使用、PEFT 双适配器切换或远端导出。

[复查脚本](../../experiments/scripts/verify_unirl_ema.py)：

```bash
python -B experiments/scripts/verify_unirl_ema.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1，全部原模块正常导入，无包壳、方法摘取或假 Shadow；真实 inject_mirror 返回原 Shadow，原 local_view 直接处理普通 CPU Tensor。四类衰减、延后复制、两种时机、零衰减及异常退出恢复控制通过。这是自编原模块检查，不是上游原测试；无新依赖安装、预训练权重、forward/backward、optimizer、LoRA 注入、DTensor/FSDP、设备发现或 GPU/NPU 运行，不新增 M5 必跑事项。

脚本核对五个完整源文件 SHA256，包括仅读的 backend。ema.py `b4b057cd565780b467c56d0ca8032a48c0e5b5366702fbd11cf2f06c9654701b`，configs.py `18dcc241700eeaf83f6e1aa9cc3a73519350c60c83d1c1631f7a8c574e29e03e`；其余见脚本。此处解释实现契约，不推荐衰减值，也不由局部 CPU 结果判断训练质量。

## 奖励服务的测试收集与空结果

**pytest不再误收集CLI，不等于奖励服务有了通过的测试。** 294962silent的[#632](https://github.com/Tencent-Hunyuan/UniRL/pull/632)删除过期`testpaths=["tests"]`，新增递归排除目录，并把`test_videoalign.py`重命名为`videoalign_smoke.py`、同步文档与启动提示。2026-10-30核为OPEN／非draft，head `ff13197a2ba6208a2287c32d72daf202fd4f68e0`，compare确认共同祖先 `8cfbac17765004afb68fef08a4e011a4c8daf555`、ahead2／behind0；reviews／行内comments为空，不推定人工批准或另认领。

两固定Git树的子项目各90个跟踪路径，均无`tests/`或`conftest.py`；旧树唯一匹配pytest默认文件名的文件就是smoke CLI，新树匹配文件为零。这只描述所核版本，不把作者“历史上从未提供测试”的说法扩成我方全历史审计。脚本重命名前后除顶层usage文档串外，完整可执行AST一致；其六个`test_*`函数接受`url`／`video`等CLI实参，并未新增pytest夹具。

同日后续01:08Z更新为head `3d02f036096c54410bc892ccca56f040d0d7c227`，仍OPEN／未合并；它同步main5d1da78d，所核五个奖励服务文件blob与旧head逐一相同。新旧compare还显示主线seed／logprob变更，不能将其算作此候选新增功能；以下结果仍固定旧输入，不重跑或升级为新服务验收。

[复现](../../experiments/unirl-pytest-collection/verify.py)在独立Python3.12.13／pytest8.4.2／requests2.32.5环境，使用各版**原pyproject与原完整CLI两文件快照**，以原树清单检查未漏匹配测试／conftest及scripts包init。真实pytest子进程全部带`--collect-only`，不替换生产代码、不生成假测试；关闭插件自动加载和cacheprovider，未安装完整项目／server／Pillow或模型栈。[结果](../../experiments/unirl-pytest-collection/result.json)含四原文件hash、两树清单hash、环境与逐命令节点／退出码：

| 原配置或显式覆盖 | 收集节点数 | 退出码 | 含义 |
| --- | --- | --- | --- |
| 旧版默认／`-m "not gpu and not slow and not integration"`／仅`-o testpaths=`，三组 | 各6 | 各0 | 标记过滤不排除这些无标记CLI；只去过期路径告警仍会发现CLI |
| 旧版仅`-o norecursedirs=scripts` | 0 | 5 | 递归排除可以阻止默认发现；旧testpaths告警仍在 |
| 候选默认／`-o norecursedirs=`，两组 | 各0 | 各5 | 当前树上仅文件重命名已足以避开默认模式，不否认目录排除对未来文件的作用 |
| 候选显式传`scripts/videoalign_smoke.py` | 6 | 0 | 显式文件仍会收集原函数；默认递归排除不是禁止手工指定 |
| 候选显式传不存在的`tests/integration/ -m integration` | 0 | 4 | 改说明没有生成该目录；旧命令仍是路径使用错误 |

这里的退出0只意味着**本次collection成功**，六个函数体均未执行，不是“6 tests passed”。[pytest官方退出码](https://docs.pytest.org/en/stable/reference/exit-codes.html)把5定义为空收集，不能在CI中一概当成功或用`|| true`吞掉其他失败：预期有测试的任务拿到5应检查测试是否意外消失，明确允许空树的任务则须另列实际smoke／集成验收。[递归排除配置](https://docs.pytest.org/en/stable/reference/reference.html#confval-norecursedirs)只负责发现范围，不提供模型、输入文件、服务可用性或HTTP行为的证据。

本机未运行作者的普通pytest命令，因此不将Windows上“旧版exit2／候选exit5”外部报告改记为本机base/head结果。`asyncio_mode`告警来自未加载pytest-asyncio；该包已列在原dev extra，不能把受控的最小环境告警另立成新缺陷。复现不发HTTP请求，不启动reward service、Ray、Torch／GPU或训练；已有MPS／奖励加载真机事项不重复排队。

```bash
python3.12 -B experiments/unirl-pytest-collection/verify.py /path/to/unirl-git
```

固定Git blob须预先缓存；脚本禁止隐式fetch，临时独立index只导出原文件，不改源仓库index／工作树。本题到此收束，实质修订或真实测试需求再核，不扩脚本矩阵、补作者方案或写社区。上游和既有PR-head写入仍须用户对具体动作明确授权。

## Smoke 结果格式与训练验收

UniRL [已有 PR #576](https://github.com/Tencent-Hunyuan/UniRL/pull/576)把 benchmark 结果统一为 schema v1，但它的 `smoke-synthetic` **只测 Python 空循环，不运行训练**。固定 head `a6ed0ec8899f353e33ed1d9ad364bdb560a0f179`、base `6fd82cba13f52f394747832344039408699ea378`，四文件新增 240 行；生产训练、奖励、评测代码没有改动。候选中的 `BenchmarkResult` 实际调用方只有这个 smoke CLI，README 对 rollout/reward/train producers 的描述是接入方式，不是三路已接通的证据。

### 样本计数与可选字段

[`benchmark_result.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/a6ed0ec8899f353e33ed1d9ad364bdb560a0f179/unirl/utils/benchmark_result.py)接收调用方声明的环境、负载和计时，再由 `samples / wall_clock_s` 派生吞吐；它不探测或认证这些声明。`started_at` 要求带时区，输出转 UTC；JSON 顶层键顺序固定，phases 名称排序。

| 字段或操作 | 固定候选的行为 | 解释 |
|---|---|---|
| `workload.samples` | 总测量样本数，不是每迭代数量 | CLI 的 samples=3、measured=4 得 12；warmup=2 不计入 |
| `wall_clock_s` | 必须正且有限；吞吐不能单独填写 | 合成 12 样本 / 3 秒得到 4 samples/s，只验证公式 |
| `torch=None` | 保留为 JSON null | 不把版本未知补成空字符串或某个版本 |
| memory/phases 未提供 | 对应键省略 | 不等于实测为 0；显式 `peak_memory_bytes=0` 保留 |
| `phases_s` | 复制后用只读 mapping 保存 | 后续修改传入字典不改变记录；不自动测量阶段 |
| 同一路径再次写入 | `open("x")` 报 FileExistsError | 旧文件保持原字节，不证明断电原子性/持久性 |

schema 允许 samples=0，此时吞吐为 0；CLI 则额外要求每迭代样本数为正。结果没有 success/status 字段，数值校验也不检查模型质量、收敛、checkpoint 或 GPU 正确性，不能把“JSON 可写”汇总成“训练通过”。`outputs/benchmark/` 与评测的 `benchmarks_results/` 分离；schema_version 是格式版本，不是模型或代码身份哈希。

### CPU 核对与最小复查

2026-10-22 使用 Python 3.12.13，将六个所需原文件从固定 Git 对象原样归档到独立临时目录，正常导入，无 AST 摘取、包壳或假类。`-S -B` 隐去 site-packages 并禁写 bytecode；本机另一个环境装有 torch，这里 `torch=null` 只代表该进程查不到其发行元数据，**不是整台机器没有 torch**。没有安装依赖、导入 torch/Ray 或发现设备。

原 CLI 以 samples=3、warmup=2、measured=4 执行，输出 `smoke-synthetic`、total samples=12、正墙钟时间，吞吐精确等于 12/该时间；这里只报告合同检查，不引用空循环速度作为性能。再次运行同 ID 返回 FileExistsError、文件原字节不变；measured=0 返回 argparse 退出码 2，旧文件仍不变。

直接调用原 schema 另核对 UTC、键序、可选缺省/显式 0、phase 输入快照、0 样本记录，以及六项拒绝：bool 样本数、0 measured、0/NaN 墙钟、Inf phase、负显存。均符合上述规则。这是自编检查；作者在 PR 正文报告的 19 项临时 harness 未提交，也没有被当作本轮运行过的原测试。

最小 CLI 复查（使用支持该候选的 Python 3.12；需要已有 Git 克隆）：

```bash
git -C /path/to/UniRL fetch origin pull/576/head
schema_scratch=$(mktemp -d /tmp/unirl-schema.XXXXXX)
git -C /path/to/UniRL archive a6ed0ec8899f353e33ed1d9ad364bdb560a0f179 \
  unirl/__init__.py unirl/utils/__init__.py unirl/utils/run_id.py \
  unirl/utils/benchmark_result.py benchmarks/__init__.py \
  benchmarks/speed_benchmarks/benchmark_smoke.py | tar -x -C "$schema_scratch"
cd "$schema_scratch"
UNIRL_RUN_ID=smoke-contract python3.12 -S -B -m benchmarks.speed_benchmarks.benchmark_smoke \
  --samples 3 --warmup-iterations 2 --measured-iterations 4 --out results
python3.12 -S -B - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('results/smoke-contract.json').read_bytes())
assert data['benchmark'] == 'smoke-synthetic'
assert data['workload'] == {'samples': 12, 'warmup_iterations': 2, 'measured_iterations': 4}
assert data['environment']['torch'] is None
metrics = data['metrics']
assert metrics['wall_clock_s'] > 0
assert metrics['samples_per_second'] == 12 / metrics['wall_clock_s']
assert set(metrics) == {'wall_clock_s', 'samples_per_second'}
print('schema/CPU-loop check only, not training or UniRL performance')
PY
```

SHA256：schema `760bcc3bb262907eebc1600045b5f83b770c717027d97ea43f4f88eff608a17c`；CLI `cc42e43ca953c871042daf829ca3ce5c500067e40def2c388a71fc7d2a048aac`；复用 run_id helper `e8038f48a11f2ac48e1958acc4b07fb8a4be98a2a5eb523a7aa7ffd9de148a82`。没有向 UniRL 提交 harness、采用候选或启动训练，此项不新增 M5 硬件任务。真实性能比较仍需对应模型/算法、负载、硬件、奖励与测量窗口；格式统一不能替代这些条件。

## 原生配置比较与训练计时

**比较完整训练时间，必须在两边都计入旧策略重算；接近的步耗时也不代表精度或学习等价。** [#588原报告](https://github.com/Tencent-Hunyuan/UniRL/issues/588)记录2026-10-12的MiniMax-H3 T2VA Flow-GRPO：UniRL固定`0fe4f99dc2384a022fe0c8472aec889c216d3c07`，veRL-Omni固定`756ebcefe837b13086c78687e987cf32352df471`，分别在不同物理8×H20节点运行，排除一步warmup后各取五步。它不是当前main的性能结论，也不是本项目设备实验。

2026-10-24用Python标准库复算报告公开的step2–6原浮点行，得到：

| 计时口径 | UniRL 秒 | veRL-Omni 秒 | UniRL相对耗时 |
| --- | ---: | ---: | ---: |
| step中位数 | 1522.119 | 1527.570 | −0.357% |
| step均值 | 1525.729 | 1526.838 | −0.073% |
| 含old-log-prob重算的训练中位数 | 1098.744 | 1090.215 | +0.782% |
| 错配：完整训练对actor update中位数 | 1098.744 | 821.971 | +33.672%，不能作性能结论 |

[固定veRL计时源码](https://github.com/verl-project/verl-omni/blob/756ebcefe837b13086c78687e987cf32352df471/verl_omni/trainer/diffusion/v1/trainer_base.py#L468)将`old_log_prob`与`update_actor`分开；[UniRL phase包装](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/utils/wandb_logger.py#L140)计整个`stack.train_track`，其内含[anchor准备与更新](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/train/stack/base.py#L309)。因此先逐步求veRL两阶段之和，再取中位数；1090.215269不等于两个阶段中位数之和1090.493823。该归并补齐主要重算项，不认证两个区间的全部子阶段相同。step timer也不是进程启动到退出的总墙钟，veRL driver的gen含等待生成/奖励，不能当纯GPU采样时间，或再叠加各worker耗时。

报告的“native configuration”指框架各自执行/精度路径，**不是不改默认配方，更不是单因素比较**。两边都声明8 prompt×16样本、10个CPS转移，但UniRL为FP32 master/BF16 compute、GPU reward、trainside，veRL为BF16 LoRA、CPU reward及参数/optimizer offload、vLLM-Omni；attention、初始化/RNG及轨迹/logp路径也不同。作者尝试的veRL FP32-LoRA路径因dtype错误未完成，已排除，不能把失败配置混入完成跑分。五个连续测量步不是五次独立重复，更不能因小百分比便判定统计等价。

按固定[PickScore文本](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/datasets/pickscore/train.txt)的strip、去空、保序去重、取首64条，再取首48条并保留末尾换行，重建SHA256 `6f1323912724fed9e4a99cced25d9bc8713eb71ece26d78ad0125b9587511362`，与报告一致。这确认公开选取规则可复现，不认证私有Parquet、checkpoint内容或实际训练输入；未取得原始日志/解析配置，作者20项分析helper测试未复跑。报告还明确veRL末步来自console补证、UniRL外层launcher非零退出，不能一律抹成完整launcher成功。

本机只做公开数值算术、固定文本哈希和两边计时源码核对，复现见[当日日志](../../diary/2026-10-24.md)；没有性能实测、训练/显著性检验或设备探测。现有证据不支持据此提出“UniRL慢33%”的优化，也不支持精度匹配、质量、收敛或未来main等价；不另排重复硬件任务。

## 损失单位与微批次权重

**平均 loss 必须同时说明单位与窗口。** [已有 PR #622](https://github.com/Tencent-Hunyuan/UniRL/pull/622)将单位声明移到 `StageAlgorithm.loss_units(part)`，训练栈不再从 algorithm 的属性猜测单位；固定 head `9dded4e2fd92732dd9de40252a4ae5b342b164e0`、base `6fd82cba13f52f394747832344039408699ea378`，四文件 53 增/34 删。这是 #580 第一阶段接口迁移：微批缩放仍为 `w × G / D`，不是引入新的 token 目标，也没有将 `LossNorm` 传入每个算法的 backward 接口。

### None 与零计数

基类在 token-mean 时要求 TextSegment，按 packed cu_seqlens 对 loss_mask 做累计和差分，得到每行有效 token 数；没有 mask 时回退 lengths。其他模式返回 None。训练栈随后分两路：

| `loss_units` 结果 | 微批权重与归约 | 约束 |
|---|---|---|
| None | 微批行数 / 本 update 行数；本 helper 不做单位 all-reduce | 表示 sample share，不是无监督或跳过训练 |
| 计数 tensor | w 是该微批有效单位数，D 是全局单位总数，G 是梯度平均 world；权重 w×G/D | 要求 SP size=1；不能把 G 随意替成总卡数 |
| tensor 全为0 | 仍进入计数路径；全局 D≤0 时返回零权重与该 D | 不能将“本 rank 无 token”改成 None，否则 collective 分支可能不一致 |

`LossNorm(denominator, grad_world)` 在 helper 内保存 D/G，不执行归约或认证单位。backend 的 `all_reduce_loss_sums` 与 `gradient_average_world_size` 使用同一 FSDP mesh；G 用于抵消梯度平均，因此分布式下单个 rank 的 scale 不必加总为1。自定义 `loss_units` 必须在所有 DP rank 一致返回 None 或都返回计数，这是文档前提，不是本轮证明的自动一致性检查。

原 single/unified stack 没有在该 PR 中改动：TrainStack 对全局零 token update 抛错；unified 单路为零时保留零 scale 及警告，由另一轨道继续。这是源码接线，不是本机执行过的 optimizer 行为。[算法基类](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/algorithms/base.py)、[缩放 helper](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/train/stack/loss_scale.py)、[训练约定](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/train/readme.md)。

### 原缩放函数的五个条件对照

2026-10-30补齐原 `loss_units → resolve_loss_scales` 的局部执行，版本仍为上述9dded4e2／6fd82cba；[#622](https://github.com/Tencent-Hunyuan/UniRL/pull/622)仍OPEN／非draft、四文件53增34删。[作者10-27催审](https://github.com/Tencent-Hunyuan/UniRL/pull/622#issuecomment-6436928213)不是维护者批准，读取时无review。本轮不是新代码反馈或新缺陷，亦不复跑此前原reducer四组梯度对照。

原候选方法按mask产生行单位，再分别交给两版完整原resolver。同一算法替身携带原候选方法；旧resolver仍只读其loss_agg_mode，不调用该方法。peer总量与梯度平均world由记录后端显式给定，下面的“归约”仅指该后端方法被调用：

| 条件 | 两版原helper结果 | 后端记录 |
|---|---|---|
| seq-mean、SP2、微批行数2／1 | scales为2/3、1/3，分母None | 不调用单位归约或gradient world |
| token-mean，行单位1／3／0，重排为3／1／0，peer=6、G=2 | 两微批scale为0.6／0.2，分母10 | 以本地4调用归约，再取G |
| 本地两行均0，peer=6、G=2 | 两scale均0，分母6 | 以本地0调用归约，仍取G |
| 本地与peer均0 | 两scale均0，分母0 | 调用归约，零总量早返回，不取G |
| token-mean、SP2 | ValueError | 在单位归约之前拒绝 |

本地scale之和0.8不是必须补成1的遗漏：式子仍为 `w×G/D`，分母还含peer单位；真实梯度平均未在这里执行。**局部全零、全局全零和None走不同协议路径，不能互换。** helper的全局零早返回也不是TrainStack是否backward／抛错的验证。

[复现脚本](../../experiments/unirl-loss-unit-resolution/verify.py)与[完整结果](../../experiments/unirl-loss-unit-resolution/result.json)保留四完整输入SHA256。Python3.12.13／Torch2.9.1，`python3.12 -B experiments/unirl-loss-unit-resolution/verify.py /path/to/UniRL-git`；预缓存固定commit后禁止隐式fetch。五项本地控制候选／基线一致，两新进程JSON相同；只执行原loss-mode模块、LossNorm、loss_units及两版resolver，TextSegment／Part／RankInfo为SimpleNamespace接口替身，未构造真实Segment或进程组。作者的Gloo/FSDP大矩阵仍属其报告；这里无collective、反向／optimizer或训练，不建立通用bit-identity与rank一致性。本题在条件控制范围收束，不另实现作者接口或排M5任务。

### 本机聚合函数对照

2026-10-22 用原 `aggregate_token_losses` 和真实 CPU autograd 核对两种局部归约。合成 packed 行长 `[2,3]`，loss 为 `[2,999 | 4,6,8]`，999 所在槽始终被掩码；horizon=1。两个版本的原聚合函数 AST 相同，本轮没有执行新 `loss_units` 或 `resolve_loss_scales`。

| 每行有效 token 数 | 归约目标 | 原函数整批 loss | 两个单行微批的手动权重 |
|---|---|---|---|
| 1、3 | token-mean | 5 | 1/4、3/4 |
| 1、3 | seq-mean-token-mean | 4 | 1/2、1/2 |
| 0、3 | token-mean | 6 | 0、1 |
| 0、3 | seq-mean-token-mean | 3 | 1/2、1/2 |

四组重组的前向值及对 loss 输入的梯度均与整批相同；第一组 token 梯度为 `[1/4,0,1/4,1/4,1/4]`，seq 梯度为 `[1/2,0,1/6,1/6,1/6]`。完全掩码的序列在 seq-mean 下贡献0但仍占一行，不能从分母删除。把 token-mean 的微批误乘等行数份额，会得到另一个目标，不能用“都是 mean”解释为等价。

窗口也有独立语义：single stack 的 accumulation window 逐 Part 求缩放，再乘1/M，因此平均各 Part 的 token mean，而不是把全部 Part 拼成一个 token 分母。例如有效计数1、3的两 Part 均值为2、6时，Part 平均为4，全 token 平均为5；这是现有文档规定的窗口目标，不据差值定性缺陷。本轮没有运行 accumulation stack。

复查（Python3.12.13 / torch2.9.1，显式 CPU）：

```bash
git -C /path/to/UniRL fetch origin pull/622/head
python -B experiments/scripts/verify_unirl_loss_reduction.py /path/to/UniRL
```

[脚本](../../experiments/scripts/verify_unirl_loss_reduction.py)完整执行原 loss_agg 词汇模块，AST 仅选择完整原聚合函数、延后注解，无包壳/假 backend。微批权重是显式单进程算术，不是假装运行生产缩放器。打印完整源码 SHA256：head base.py `f586474992dee187755ae47f431a90dc4857180fa1b1aa01dd8b27e2f59a4f1a`，loss_scale.py `d627ea261f0867d46335a1264622911913ff1d684450b7333aaf0dc67209f13c`，loss_agg.py `8049957fa4ecbf2086afab6e066fa1beeb9d35bdabebfd3d8890f7ee078ecd3f`。

本机环境缺 Ray/Hydra/OmegaConf/Pillow，未补装或导入训练栈，未复跑作者报告的双 rank Gloo/FSDP2 gate、80,640 组差分或 AROPD。这里确认局部目标差异和固定源码路径，不认证候选所有输入 bit-identical、分布式归约、模型梯度或训练收益；不自动采用或新增 M5 必跑事项。

## 离线 DPO 的成对布局与零步控制

**零步loss等于log 2，是一致性控制，不是配对语义、参考冻结或生成质量的充分证明。** [#476](https://github.com/Tencent-Hunyuan/UniRL/pull/476)新增Qwen3-Omni Thinker离线DPO，2026-10-24读取open/non-draft，head `f824e6954e702ce4fe2e4bb9ec0394152c7306ce`、base/实际祖先 `e6934259da95c5f212a87d572b9ba55284a66ae8`。DPO是偏好目标，不是仓库中名称相近的Divergence-PPO。已有作者实现，本节限定输入几何及原数学片段，不是完整候选审阅或新增修复。

### 记录数与样本行数

[ARPreferenceTrackBuilder](https://github.com/Tencent-Hunyuan/UniRL/blob/f824e6954e702ce4fe2e4bb9ec0394152c7306ce/unirl/train/sft/track_builder.py#L571)声明每条记录两行，把chosen/rejected按相邻顺序展开，共用重复的prompt条件。令全局记录数B、DP数D、每数据批更新数U，则每rank有`2B/D`行，每次更新需分得完整偶数行；micro大小也须为2的倍数。[SFTTrainer guard](https://github.com/Tencent-Hunyuan/UniRL/blob/f824e6954e702ce4fe2e4bb9ec0394152c7306ce/unirl/trainer/sft.py#L89)检查这些整除关系，而[DPO._split_adjacent](https://github.com/Tencent-Hunyuan/UniRL/blob/f824e6954e702ce4fe2e4bb9ec0394152c7306ce/unirl/algorithms/dpo.py)只按偶/奇行切分并拒绝奇数行，不读取prompt ID来证明语义匹配。本机没有运行builder、tokenizer、DP分片或trainer guard。

### 归约单位与 sigmoid 目标

原`_reduce_to_sequences`按长度把packed log-prob放到`[S,Lmax]`再沿固定轴求和，先乘loss_mask；开启average_log_prob才除以监督mask之和，分母最小1。它不是默认token均值，且该中间量随`S×Lmax`增长，不能把packed输入等同于没有稠密临时量。`_real_pair_mask`以两行是否有监督token的OR判定，只有两支都完全mask的pad pair会被剔除。

默认sigmoid、label_smoothing=0时，令每行`d=logπ−logπ_ref`，pair margin为`d_chosen−d_rejected`，loss为`softplus(−β×margin)`；奖励准确率用严格`βd_chosen > βd_rejected`。在策略与参考逐行相等时，margin=0、loss=log 2、准确率0；若同样置换策略和参考的行，这些值仍不变。因而这个控制可帮助定位初始数值不一致，但无法独自识别跨prompt错配，也不证明后续更新中的参考不漂移。作者另报adapter-disabled与无adapter模型的独立对照，本机未复验。

### 本机仅执行原数学片段

Python3.12.13 / torch2.14.1 / CPU FP64，从固定源码AST提取完整归约、相邻分对和pad-mask方法；`_pair_loss`只保留从分对开始的原后半段，显式输入合成policy/ref sequence分数，跳过replay与adapter上下文，绝不构造替身stage。segment仅为lengths/mask属性夹具：

- 长度`[3,2,4,1]`及声明的0/1 mask，原sum为`[-3,-9,-23,-10]`，average为`[-1.5,-4.5,-23/3,-10]`；与独立逐段求和的值及局部梯度逐位相同。奇数三行被原split拒绝。
- 两pair、β0.1，正确顺序与人为跨prompt重排`[0,3,2,1]`均得到loss `0.6931471805599453`、accuracy0、margin0；两者policy序列分数梯度均为`[-.025,.025,-.025,.025]`。准确率0不等于没有训练信号。
- 只把chosen相对参考提高2，loss `0.5981388693815918`、accuracy1、奖励margin0.2。三pair的监督模式`[1,1,1,0,0,0]`得到keep `[True,True,False]`，不是要求每支都有监督。

这些检查不认证CUDA原子归约的确定性、BF16长序列、完整DPO/IPO、LoRA参考身份、模型、训练恢复或偏好数据质量。作者的4×H20训练、长度偏差分析及学习指标属于其报告；正文也明确GPU结果早于rebase，不搬到当前head当验收。复现见[当日日志](../../diary/2026-10-24.md)，不新排硬件任务或演化竞争配对补丁。原dpo.py SHA256 `453d179928786688154a2ea2e3402c07599d8b059d92870121d2692a08aaa7a2`。

## AROPD 的采样梯度与 KL 目标

[已有 PR #534](https://github.com/Tencent-Hunyuan/UniRL/pull/534)新增 AR 端 on-policy distillation：teacher 评价 student 自己生成的 token，再将 teacher/student 的 log-prob 差作为冻结权重。固定 head `1962740c4aabe104c2528d0a79ed853c52d05046`、base `0fe4f99dc2384a022fe0c8472aec889c216d3c07`；本节区分源码机制和固定前缀下的数学，不把采样 surrogate 的值直接当作 KL。

### Teacher 与 student 的接线

[`AROPD`](https://github.com/Tencent-Hunyuan/UniRL/blob/1962740c4aabe104c2528d0a79ed853c52d05046/unirl/algorithms/aropd.py)要求注入 backend，并在同一模型上找到命名的 teacher adapter。配方是同 backbone/tokenizer 的可训练 default LoRA 加 frozen teacher LoRA，不是任意远程大模型 teacher 服务。`prepare_segment` 在 no_grad 与 adapter_active 上下文内 replay，将结果 detach 到 CPU 的 packed `teacher_log_probs`；anchor contract 按训练微批几何准备并恢复行序。teacher forward、adapter 冻结及实际输出身份这里只读源码，未运行 PEFT/模型。

student 的旧 log-prob 来自 `segment.log_probs`，新 log-prob 来自当前 replay；raw gap 是 `sg(teacher_logp-old_logp)`，只有 new_logp 接收梯度。代码声明无需奖励优势、单 teacher、同步路径与不支持 multi-update；配方 num_updates_per_batch=1，teacher replay 温度取 sampling.temperature。一次 update 的声明本身不证明实际 rollout/replay 概率逐位相等，代码另记录 drift 指标。

默认 `normalize_gap=true` 还对当前微批 gap 做中心化/除标准差（元素数>1才做），指标保存 raw gap。本节的恒等式仅讨论 **raw gap**，不能据此认证默认 z-score 的无偏性；普通 baseline 的零期望论证要求其相对于当前动作独立，不能不加条件套到动作相关权重或样本自身统计量。

### 固定前缀的梯度恒等式

令 student 分布 p=softmax(z)，teacher 分布 q 冻结，所有动作概率为正、旧新 student 相同，不改变温度或采样支持；这里把前缀当固定输入。reverse KL 为 `K=Σ p·(log p−log q)`，其梯度为 `E_p[(log p−log q)∇log p]`。对应最小化 surrogate 是 `L=−sg(log q−log p_old)·log p_new`；两者**期望梯度**相同，不要求标量值相同。

与它不同，冻结抽样权重后直接微分 `log p−log q` 只剩 score，其期望 `Σp∇log p=0`；有限 batch 仍可能有非零采样噪声。这不表示 KD 普遍无效：完整词汇分布 KL 的梯度正是这里的比较基准。另一个不同目标是 teacher cross-entropy `C=−E_p log q`；`K=C−H(p)`，少了 student entropy 项便不是同一个梯度。固定候选 README 将最大化 E_p log q 与 raw gap 式连写、把 student log-prob 称为 control variate，这个等同关系不能直接沿用。

这也要区分原论文的范围：[GKD §3 式2、4](https://arxiv.org/html/2306.13649v3#S3)在 student 生成的前缀上比较 teacher/student 的 token 分布，并明确不反传外层序列抽样过程；它允许不同 divergence。on-policy 描述数据来自谁，不自动证明某个 sampled-token、归一化或完整轨迹目标都彼此等价。

### 本机精确枚举

2026-10-22 用自编 CPU torch FP64 参考，完整枚举三个动作，不实际抽样。p=[0.2,0.3,0.5]、q=[0.7,0.2,0.1]，对 student logits 的梯度如下：

| 计算图 | 梯度 |
|---|---|
| 完整 reverse KL | `[−0.385714,−0.081102,0.466816]` |
| raw score surrogate 的精确期望 | 同上 |
| 直接 gap，冻结抽样权重 | `[0,0,0]`（浮点误差内） |
| teacher cross-entropy，保留 p 的梯度 | `[−0.269757,−0.028806,0.298563]` |

KL=0.675806，而 surrogate 值约−0.300991：负 distill surrogate 不是负 KL 的证据。再令 q=p，reverse KL/raw surrogate 的梯度均为0，但 teacher cross-entropy 梯度为 `[0.115957,0.052296,−0.168253]`，仍会改变已匹配的分布。两例均核对 `∇C+∇(Σp logp)=∇K`，且原始 KL 的中心有限差分与 autograd 一致。

[参考脚本](../../experiments/scripts/verify_aropd_score_gradient.py)：

```bash
python -B experiments/scripts/verify_aropd_score_gradient.py
```

Python3.12.13 / torch2.9.1，显式 CPU，无包壳、模型或依赖安装。**这是自编参考数学，不是执行原 AROPD，也不是作者测试**；没有证明多步轨迹梯度、默认 gap normalization、off-policy correction、PEFT/FSDP 或训练收益。源码/配方 SHA256 分别为 `c96283715000b9979f84b0d1712c542fb7140575e4997a493ac3a3110e221795` / `4664747d7123e874bfb13d61ee5526880344becc54f5427dbcbe2e25f27305d6`，推导 README 为 `6214d59d75759899dfea09bd2df500c24d10d18223a24e9daaef70d7f7db35e9`。

作者报告的6×H20训练早于该 head 的 anchor/单位接线 rebase，正文明确未在 rebase 后复跑 GPU；其 CPU harness 也有模型 forward 替身，不当作本机结果。此处只保存理论限定与学习证据，不改上游、发布审查结论/候选、启动 teacher 实验或新增 M5 必跑事项。

## 关键帧条件的数据身份与功能就绪

**rollout和replay读到同一份条件数据，是概率比较的前提之一；消费接口存在却不等于关键帧生成功能已能启用。** 2026-10-22只读已有[#584](https://github.com/Tencent-Hunyuan/UniRL/pull/584)，读取时open，固定head `55a3cad3e934cb0e637d8f3cf1c37d2a91886891`、目标base `accba01082115534e8b60d849e7276a636eba269`、merge-base `c7c6b8c0e0c60db7205d9b81fdafca778c747ece`。6文件125增22删，全部在MiniMax-H3模型包，无提交测试文件。FL2VA指首帧、末帧或首末帧条件的视频/音频生成；本候选仅实现路线图的消费端step1a。

### 条件行不等于原始图像或目标轨迹

[`MiniMaxH3KeyframeCondition`](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/conditions.py)按样本保存3个对齐列表：已经加噪的fp32 rows、anchors和未padding的text_token_tags。rows不是待读取的图片路径，也不由replay重新编码/抽噪。模型latent通道24、patch为(1,2,2)，故每行宽96；rows_per_frame由目标画布latent高宽除patch尺寸后相乘。

| 样本/批形态 | keyframes表示 | 必须保留的边界 |
|---|---|---|
| 纯文本生成批 | 全批keyframes=None | 使用原t2va布局 |
| 带关键帧的混合批中的文本样本 | anchors=()、fp32 rows形状[0,96]、全text tags | 是一个空条目，不是省略该样本 |
| 首帧或末帧条件 | anchors=(first,)或(last,)，rows为[R,96] | 标签长度对应去padding后的文本 |
| 首末帧条件 | anchors=(first,last)，rows为[2R,96] | 顺序和每样本对应关系不能只靠形状推定 |

R表示rows_per_frame。[build_layout](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/packing.py)核对batch-1、合法anchor元组、long型0/1 tags及长度、fp32 rows形状；空anchors必须全text tag。条件容器另核3列表等长，以及keyframes样本数等于text样本数。它们是结构检查，不认证某行确属对应rollout样本、图像内容或随机噪声。

通用[Batch拼接](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/distributed/tensor/batch.py)对字典key取并集后递归处理：其他项已有值时，非零行项缺少key会报错；零行项可以缺省。这项保护来自已合入的通用规则，不是本PR重新实现。否则将3份关键帧条目与4份文本行对齐，后续样本可能整体错位。混合批要为每个真实样本提供条目，不能用“这个样本没有关键帧”作为删掉一项的理由。

`trim_text_padding`改用dataclasses.replace，仅替换text字段、保留keyframes等槽，避免重建只含text的容器时丢条件。tags本来就应未padding，不能跟着已padding文本长度生成。保留字段并不让可变tensor/list成为不可变快照；运输/切片/训练对齐还需保证实际payload没有被换样本或转低精度，结构guard不是字节身份校验。

### 条件参与前向但不变成待评分目标

三个入口generate、replay和predict_noise_at_step都经相同的[`predict_noise`](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/diffusion.py)：将携带rows stack后搬到目标device，放在video_sample前，再调用transformer；返回时裁掉前面的condition video velocity，只将target video/audio送回下游。NFT的clean latents则由segment末态video/audio打包，不把条件行附到目标上。本轮未执行NFT loss，不能由forward形状正确推断训练正确。

联合音视频replay还要求segment.aux_latents，不能只凭视频轨迹和关键帧恢复整个前向。在[batched replay](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/types/batched_replay.py)中，多step tile通过 `type(conditions).concat([conditions] * S)`复制条件，再送predict_av_noise；因此条件数据和目标状态的sample/step排列必须一起核对，不能只对最终log-prob均值。

直观地，比较同一转移的条件概率时，分子与分母需要对应同一已实现的条件c；若训练时重抽条件噪声得到c'，它就不再只是“在相同输入上换一套策略参数”。这是条件身份的解释，不是本轮对某个SDE核或NFT目标的数学/数值验收。README所述diffusers/vLLM-Omni噪声与第二帧resize差异，是作者的实现对照；本机未审这两条外部生成路径。

### 仍有两个启用前提

固定head的[pipeline.generate](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/pipeline.py)仍仅构造 `MiniMaxH3Conditions(text=...)`，与目标base逐字节相同；它没有本PR所需的keyframe编码、conditioner prompt和数据producer。类型导出/下游接线不能算实际用户请求已到达关键帧路径。

`row_timestep_plan`的base/head AST也相同：标量分支将condition_video_timestep设为target video_t=1−sigma；向量分支先将所有行填video_t，再只改audio target列。两条分支都未给条件帧独立的近clean timestep。[候选README](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/README.md)明确尚需step2的max(1−sigma,0.999)条件时间，之前禁止recipe启用FL2VA。即使合成条件在rollout/replay之间保持一致，也只说明这条实现的一致性，不说明已等价发布模型的条件语义。

作者报告外部一次性harness的合成条件parity、tiny随机transformer及CPU/MPS往返；harness未提交，本轮无法复跑。作者也明确没有实际producer、NFT loss或GPU/FSDP验证。这些报告不当本项目结果，也不以“205通过”或t2va兼容声称真实关键帧训练已就绪。

M1只做源码学习和静态检查：Python3.12.13对5个改动Python文件及pipeline/config/Batch/batched_replay共9个完整模块内存compile；核row_timestep_plan AST未变、pipeline字节未变且仅一个text条件构造点，均通过。没有导入torch/UniRL、实例化条件/变换器、复跑parity或loss、使用MPS/GPU/NPU，未增加容器/模型替身或永久一次性工具。

复查静态入口（只compile不执行，须本地已有Git对象）：

```bash
python3 -B - /path/to/UniRL-git <<'PY'
import ast, subprocess, sys
head, base = '55a3cad3e934cb0e637d8f3cf1c37d2a91886891', 'accba01082115534e8b60d849e7276a636eba269'
def src(rev, name):
    return subprocess.check_output(['git', '-C', sys.argv[1], 'show',
                                   f'{rev}:unirl/models/minimax_h3/{name}.py'])
for name in ['conditions', 'packing', 'diffusion', 'pipeline']:
    compile(src(head, name), name, 'exec')
def plan(rev):
    return next(n for n in ast.parse(src(rev, 'packing')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'row_timestep_plan')
assert ast.dump(plan(head)) == ast.dump(plan(base))
assert src(head, 'pipeline') == src(base, 'pipeline')
PY
```

SHA256：conditions.py `59d217094b12f23bd09dd434ea3009ad4f2db16004447d886c766624f1eb1206`；packing.py `ae1693dc89e6523ac5a19cfb458058fab3a58bada75228722e7feefe05f8cc19`；diffusion.py `c9bc12117fa8babc041ee451576cdc47baffb795bb40ec3fd9465969891a26c9`；pipeline.py `7164e4d552338ed344e8a0244426758cb905b5c9c3f229239ea08323b2fd0efa`。真机缺口仅补[原M5 UniRL入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)：先等producer/条件时间语义就绪并独立选型，当前不启用、不另开硬件任务，也不修改已有作者的方案。

## 设备资源与通信后端的分层边界

**资源预约、进程可见性、实际设备与通信组不是同一个承诺。** 2026-10-22只读[#623](https://github.com/Tencent-Hunyuan/UniRL/pull/623)：当轮API状态为closed、merged=false，关闭于09:39:42Z；不是已合入的支持声明。固定head `c1477a030cda826b6befd2e6e787ef64f67e41dd`，实际父提交/官方compare merge-base `c6d4b11b24743fe1e933b61192f9d49558f7171e`，仅3文件46增7删、无新增测试。API目标base为 `6fd82cba13f52f394747832344039408699ea378`，两侧ahead1/behind59；不能把目标分支至head的两点diff当作本PR改动。本地浅克隆起初找不到merge-base，已用官方compare/commit parent核定并获取父对象，不据失败猜版本。

### 四层分别核对

| 层次 | 固定head中的行为 | 不能由此推出 |
| --- | --- | --- |
| Ray资源预约 | placement group仍按每设备`GPU:1`与CPU bundle、STRICT_PACK预约 | actor的`num_gpus=0`不等于无GPU资源需求或CPU集群可跑 |
| Actor可见性 | `_spawn_worker`无平台分支地将num_gpus改0；从driver的CUDA_VISIBLE_DEVICES列表按全局device_id选一个值注入 | bundle_index不是物理卡身份；全局索引超列表长度不注入，不能认证多节点映射 |
| Torch设备选择 | cuda可用时，单个纯数字ID且小于device_count则pin该ID，否则pin0；设置`cuda:<pin>` | 回退0不是正确分卡的验证；仍用cuda命名空间，不代表原生NPU支持 |
| Tensor传输组 | 先建显式ProcessGroupNCCL/eager connect；任意Exception转`init_process_group(backend="eccl")`并取WORLD | 不等于自动选择任意硬件后端，也不保证默认组生命周期无冲突 |

具体见[device_pool.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/group/device_pool.py)、[worker.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/group/worker.py)与[store.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/tensor/backend/colocate_store/store.py)。[Ray资源文档](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)区分逻辑资源与物理使用，[placement group文档](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)说明bundle预约，[加速器文档](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)说明资源请求及环境变量分配。它们解释API职责，不是对作者GCU栈或本PR的认证。

设备编号的源码推演例：若单值CVD="2"被运行时重映射为仅1张可见卡，判断得到pin0；若运行时忽略该mask且报告8张，则得到pin2。空/多值/非数字（如UUID）或越界也走0。这只是条件分支推导，M1没有调用is_available/device_count/set_device，没有证明具体插件遵守哪种语义。原注释称NVIDIA等价，但改动没有NVIDIA豁免分支，回归仍需独立证据。

### 回退还改变了组的所有权

`Worker.setup_global_pg → transport.setup_transfer → TensorStore.setup_global_pg`为这条传输组接线。显式NCCL对象失败后改初始化进程默认组，不只是换一个局部变量的后端。[PyTorch初始化文档](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group)明确该API初始化默认进程组；[第三方后端文档](https://docs.pytorch.org/docs/2.14/distributed.html#third-party-backends)要求插件注册并由实现决定能力。文档版本用于解释语义，非作者环境版本。需要实际栈确认eccl已注册、默认组是否已有持有者、rank/world与销毁职责；宽泛异常也可能来自非厂商原因，源码没有在except中显式清理已部分创建的NCCL对象。这些是未核条件，不是已复现的故障或修复建议。

send/recv按`hasattr(group, "send"/"recv")`选择成员方法，否则调用`dist.send/recv(..., group=group)`；不是按后端名选择。取到WORLD也可能仍走成员方法，方法存在不认证P2P可用。原put仍按tensor.is_cuda决定本地设备存储或ray.put，不能单凭新增eccl外推原生NPU高速传输。

作者正文报告8张燧原GCU L600上GRPO 10/10 rollout、曲线形态及NVIDIA等价检查，未附可复跑命令/配置/原始日志；本机没有复验，不能转称4090、昇腾或跨节点结果。PR关闭原因未知，不据merge_commit_sha字段推断合并，也不自动采用、修改或对外评论。

### 本机范围与复查

Python3.12.13仅标准库，对父/head各3个完整模块内存compile，共6次通过。没有导入UniRL/Ray/torch、执行构造器、模拟设备/通信成功或运行原测试；这是语法与源码证据，非设备、训练或NVIDIA回归验收。最小静态复查（Git对象须已在本地，禁止隐式拉取）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
for rev in ['c6d4b11b24743fe1e933b61192f9d49558f7171e',
            'c1477a030cda826b6befd2e6e787ef64f67e41dd']:
    for path in ['unirl/distributed/group/device_pool.py',
                 'unirl/distributed/group/worker.py',
                 'unirl/distributed/tensor/backend/colocate_store/store.py']:
        src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{rev}:{path}'])
        compile(src, f'{rev}:{path}', 'exec')
PY
```

head SHA256：device_pool.py `eeb37485a54a1c8b4d9aaa07f3fad7db20c72916433fb2189a61d331651dc543`；worker.py `2f2c2577b8649ee9836006503d748826c42505067432995b470d0b5c94eeb189`；store.py `c19fef9e947de74a2d8ff61508d25221cbad6995e8f806f82cf090f60b1ebcc2`。真机缺口记入[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不另开硬件队列。本题以分层契约学习收束，不继续造设备/通信替身。

## TempFlow 分支共享与逐行时间索引

**共享确定性前缀的输入，不等于复用前缀计算；合并存储槽，也不等于合并实际时间。** 2026-10-22只读[#552](https://github.com/Tencent-Hunyuan/UniRL/pull/552)，当轮open、未合并，固定head `ef9cb362ee8f92f1dcae6ec474075ed922ef994d`、base/官方compare共同祖先 `6fd82cba13f52f394747832344039408699ea378`，ahead17/behind0，23文件1367增62删。差异包含#546/#589的Flash-GRPO及replay前置工作，不全算作TempFlow独有实现。以下解释此候选的接线，不认证与论文/官方实现等价。

### 共用初始噪声但仍重复完整生成

> 版本提示：本节后续的算法／噪声控制以ef9cb36为固定历史对象；2026-10-26接续核对的候选已到0e6a620，范围与计数类型变化见下方。旧GPU报告不自动迁为新head验收。

[trainer._build_branch_request_samples](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/trainer/diffusion.py)为每个选中的步j建立一次包含全部prompt的请求，固定`sde_indices=[j]`。生成sample_id带步组序号和分支序号以保持唯一；init_noise_group_id则只带prompt父ID和seed序号，使同seed的6分支、不同步组使用同一初始噪声键。它是当前输入批次内的共享，不据此承诺跨rollout、批形状或引擎逐位轨迹一致。

这些显式键通过[NoiseRecipe.from_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/types/noise_recipe.py)优先于init_same_noise的默认分组；[SD3Pipeline](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/sd3/pipeline.py)再生CPU fp32 x_T并送diffuse。因此不能因继承配方init_same_noise=false就推断此分支路径没有共享x_T；缺driver latent shape时trainer反而拒绝branched布局。

[SD3 diffuse](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/sd3/diffusion.py)仍遍历全部T步，仅j的eta非零，其余为ODE步。分支前相同输入沿相同确定性路径是设计目标，但每组重新调用generate；源码没有在这里将前缀算一次后缓存给所有步组。减少训练的随机转移数量，不等于只做一次模型前向或缩短为一步图像生成。

固定[配方](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/examples/diffusion/sd3/sd3_tempflowgrpo.yaml)继承10步SD3.5-M生成，24样本/prompt=4种子×6分支，候选分支步为0..8，每rollout轮转选3步。按源码计数，步组依次为[0,1,2]、[3,4,5]、[6,7,8]，每prompt产生72行；全9步组则216行。batch_size=16时，每个步组384行、三组共1152行；这是逻辑行数，不是并发峰值、显存、调用耗时或训练收益。池排除最后去噪步，候选还检查其存储形状与早期步组不兼容。

### 先组内算优势再保存每行的真实步号

trainer逐步组完成reward/advantage后才合并生成frontier，不复制共享的root前缀。配方advantage_group=prompt、adv_use_global_std=false，所以每个(step,prompt)的24行一起归一化，不跨步组比较奖励；可选seed分组则以连续6行为一组。共享初始seed和优势分组是不同选择，不能把默认配方称为“每seed单独归一化”。这里只核原Part.compute_advantages及trainer接线，未运行组内奖励计算。

合并前的`_stamp_sde_index_per_sample`把每行潜变量固定为以下存储槽，同时单独保存真实j：

| 合并后的字段 | 含义 | 消费方式 |
| --- | --- | --- |
| latents[:,0/1/2] | 本行分支前、本行分支后、终态 | 固定槽位置，不是所有行共有的全局去噪时间 |
| sde_index_per_sample | 每行原来的j | 按j/j+1取完整sigma表中的两项 |
| indices=0,1,2与sde_indices=[0] | 合并布局的槽/单转移元数据 | 不能由这个0推出所有行都在去噪第0步 |

[PerSampleStepReplayMixin](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/types/per_sample_replay.py)据每行j取sigma/sigma_next，用槽0/1重算该转移，返回log_probs[N,1]；不会用共享latents_at(j)解释合并后的槽。它还在replay端init_schedule，不能假设该rank已跑过生成。step_index=0这个调用参数也不代替逐行sigma。

[PerSampleStepGRPO](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/algorithms/per_sample_grpo.py)取每行旧logp首槽，clip优势后算clipped ratio loss；[TempFlowGRPO.row_weights](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/algorithms/tempflowgrpo.py)再乘weight_scale×FlowSDEStrategy.transition_std（配方weight_scale=2.25），后者含std_dev_t×sqrt(−dt)，不是只用扩散系数或一个统一步权重。构造要求beta=0、eta>0及支持逐行索引的FlowSDE stage；async trainer明确拒绝此布局。配方的old_logp_source继承sd3_trainside的replay，并非类默认rollout；本轮没有执行anchor准备、loss或backward。

### 本机仅验证原噪声模块

Python3.12.13 / torch2.9.1，将固定Git blob的完整`unirl/sde/noise.py`原样载入独立命名空间，无方法摘取、包壳或模型替身。该文件与base的blob相同，是复用的基础模块，不是本PR修复验证。向regen_initial_noise显式输入2个prompt×4个seed×6个分支的48个测试键、latent_shape=(2,3)、CPU fp32/base_seed42：三次再生相同，同seed六行相同，测试中的不同seed/prompt不同；逆序输入对应逆序输出，改base_seed会改变输出，CPU全局RNG状态未变。这些键是控制输入，不是实际执行trainer生成的Sample，也不证明去噪前缀/branch logp或loss parity。

最小复查（对象须已在本地，模块第三方依赖仅torch；本例只调用CPU噪声函数）：

```bash
GIT_NO_LAZY_FETCH=1 python -B - /path/to/UniRL-git <<'PY'
import subprocess, sys, torch
head = 'ef9cb362ee8f92f1dcae6ec474075ed922ef994d'
src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{head}:unirl/sde/noise.py'])
ns = {'__name__': 'tempflow_original_noise'}
exec(compile(src, 'unirl/sde/noise.py', 'exec'), ns)
keys = ['p::seed0'] * 6 + ['p::seed1'] * 6
x = ns['regen_initial_noise'](keys, 42, (2, 3), torch.device('cpu'))
assert torch.equal(x[:6], x[0].expand(6, 2, 3))
assert not torch.equal(x[0], x[6])
PY
```

noise.py、noise_recipe、trainer.diffusion、TempFlow/PerSample GRPO、PerSampleReplay、SD3 diffusion/pipeline及Sample共9完整模块内存compile通过，其余模块不执行。noise.py SHA256 `694a81dcf17cf2ff1c5de13399dcc5d721bc02c100394620acc845a9b7b1830c`；trainer.diffusion.py `55dc251e5f707af2501c16e76d892efd65794f60efb0207855c397a12e2d5458`；per_sample_replay.py `bbf5b6358c65c3aa047312ce8a5c9fbab9945d90240286a52286ad1f50a4c6f3`。

PR未提交所述外部harness；作者40项CPU/早期8×L40S smoke与5a1eff0的fp32 parity报告均非本机结果，且正文明确最新API/replay修改未重跑GPU。其原生批形状latent阈值未过、统一microbatch=4后才逐位匹配的观测，也不能外推默认bf16或训练等价。真机缺口补[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动选用、部署配方或新增训练队列；本题以分支与时间轴契约学习收束。

### 后续范围收窄与计数类型

2026-10-26核[#552](https://github.com/Tencent-Hunyuan/UniRL/pull/552)仍OPEN／非draft，作者ruiling-smartbear；新head `0e6a620c1058234e021c56230083c235abc5828a`比旧ef9cb36多一个[收窄提交](https://github.com/Tencent-Hunyuan/UniRL/commit/0e6a620c1058234e021c56230083c235abc5828a)，增量七文件19增418删。相对base／共同祖先及当前main `6fd82cba13f52f394747832344039408699ea378`，真正PR现为18文件957增51删、ahead18／behind0，不再沿用上文旧23文件的范围。唯一普通评论是作者要求先审#589／#546前置，reviews／行内评论空，不是维护者批准或我方接单。

该提交从候选树移出两份WAN Flash配方，并将SGLang adapter的flash映射、patch_dance中的flash白名单／公式分支移回主线状态；当前整个sglang_diffusion子树相对固定main无diff。不能再把旧候选中的SGLang Flash适配算作新head功能，也不能据此宣称UniRL全局废弃Flash。作者称这些工作保留在独立分支，本轮未核其分支内容；TempFlow仍依赖replay／Flash训练侧前置。

算法[构造器](https://github.com/Tencent-Hunyuan/UniRL/blob/0e6a620c1058234e021c56230083c235abc5828a/unirl/algorithms/tempflowgrpo.py)与driver的[_configure_branches](https://github.com/Tencent-Hunyuan/UniRL/blob/0e6a620c1058234e021c56230083c235abc5828a/unirl/trainer/diffusion.py)现要求type(count) is int，不只检查“能转成整数”。branches_per_seed必须正整数；branch_steps_per_rollout可为None，否则算法要求正整数、driver还要求不超过候选pool长度。driver的类型检查在本条件的模运算之前短路；算法构造器仍先调用super().__init__，不据此承诺所有模型／资源初始化之前全局fail-fast。

本机Python3.12.13仅抽取这四个**完整原if节点**，给每个guard分别输入6、True、6.0，driver上下文仅固定samples_per_prompt=24、pool=0..8：

| 输入 | 旧版四个guard | 新版四个guard |
| --- | --- | --- |
| Python int 6 | 均放行 | 均放行 |
| bool True | 均放行 | 均ValueError |
| float 6.0 | 均放行 | 均ValueError |

共24次if求值，不是24项原单元测试或完整算法／driver执行；没有执行super、真实采样参数、构造请求、modulo到range的整链或张量计算。四份完整Python源码内存compile，row_weights与_build_branch_request_samples的完整AST在两head相同。复核代码和四源文件SHA256见[当日日志](../../diary/2026-10-26.md)。未扩充非法数值矩阵，也不把旧int(NaN)本就报错的情况说成“此前全部非法值都能通过”。

作者新报告的38计数／40 replay与实际loss-backward检查使用未提交的外部harness，本机未复跑；正文明确此次范围／校验变更未重跑GPU。原fp32批形状及早期8×L40S证据仍只归其原版本，不认证新head、默认bf16或学习收益。仅同步[原M5条件入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不应用候选、补回移出配方／适配或新排训练；后续仅在实质接口／模型证据变化时接续。

### 逐行步号与潜变量引用保留

2026-10-24，[前置PR #589](https://github.com/Tencent-Hunyuan/UniRL/pull/589)更新到`ede4429a2af244fbc932820686b551647fdc1960`，base `6fd82cba13f52f394747832344039408699ea378`，8文件345增32删。它在合并main的`a86f5ca8d27ea4e4d4adaf521a5b5c6892178539`之后，新增[保留潜变量句柄](https://github.com/Tencent-Hunyuan/UniRL/commit/87e68f407ece2750e16e6d44a678c2ab0c178a95)及[先hydrate共享索引](https://github.com/Tencent-Hunyuan/UniRL/commit/ede4429a2af244fbc932820686b551647fdc1960)两项修复。此处是#589的新代码，不把它自动套到上面固定的#552版本，也不把前置工作当作完整FlashGRPO loss/kernel。

新的`_stamp_sde_index_per_sample`要求生成端**已经**按`[j, j+1, T]`存好三槽潜变量：检查`latents.shape[1]==3`及`hydrate(indices).tolist()`后，只把共享indices改为`[0,1,2]`、sde_indices改为`[0]`，保留原latents对象。`sde_index_per_sample`继续逐行记录真实j，sampling params的共享sde_indices清空以支持后续拼接。driver不能将TensorRef当作完整torch.Tensor使用；形状元数据可读，不代表它支持二维索引、比较、`tolist`或`torch.stack`。

Python3.12.13 / torch2.14.1，完整原TensorRef/Batch/LatentSegment/DiffusionSamplingParams及本地hydrate模块加载；AST只提取三提交的完整stamping方法。frontier/sample为属性夹具，handle为显式计数的CPU本地协议夹具，不是远端handle。三种布局对照如下：

| 潜变量与共享indices | 合并main后 | 保留引用修复后 | 最终head |
| --- | --- | --- | --- |
| 两者均为CPU tensor | 通过，但重建latents | 通过，保留原对象 | 通过，保留原对象 |
| latents为TensorRef，indices为tensor | tuple索引NotImplementedError | 通过，无潜变量读取 | 保持通过 |
| 两者均为TensorRef | 比较得到bool后调用nonzero报错 | TensorRef无tolist报错 | 通过，仅indices handle读取一次（三个整数） |

两个step组j=2/5、各两行的原`LatentSegment.concat`得到四行，真实步号为`[2,2,5,5]`，原latent handles保留且读取计数仍0；随后显式`hydrate`才各读一次，结果与两份原CPU潜变量拼接逐值完全一致。错误步终点、错误总步数及两槽布局均被最终方法拒绝，无潜变量读取。该方法先写逐行步号再校验，不宣称拒绝具有事务回滚保证。

复现代码见[当日日志](../../diary/2026-10-24.md)；没有Sample/Part concat、完整trainer、实际Ray/远端传输、模型replay或GPU运行。作者另报真实TensorRef stamp/concat及20项CPU replay检查，本机没有复跑其harness；PR明确本次未重跑GPU，旧#546 Wan结果不覆盖当前head。已有作者修复，不另开重复补丁或新硬件队列。三版trainer SHA256依次为`39bcc77c4d66f39da510eb938aa8385c856f557fe04af338c5555e375a1d49e3`、`58e38d6c15968aa826d3ab8334a5cef211110307e7faf36c47d6fdb21a2e34f8`、`40eceb30fd4e0221d18c4e7db93db41d70ad71d6e6764e5c7cc1f0c02beaee17`。

## 异步快照与指标消费

**先分清状态、事件和采样时刻，再解释异步曲线。** 2026-10-22只读[#620](https://github.com/Tencent-Hunyuan/UniRL/pull/620)，读取时open、未合并，固定head `f404c9024c00bbf8a5d05315153444cee894865a`。API目标base为`6fd82cba13f52f394747832344039408699ea378`，官方compare共同祖先为`0397ddb6a6fc0c9ccdc3508c0bee0e687000bca7`，ahead4/behind19；按共同祖先核对才是7文件228增9删，不把落后主线的历史差异算作本PR修改。新增接口不代表此版已部署，也不是训练通过证据。

### 队列深度不是一种单位

[RolloutPool.snapshot](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/dispatch.py)在condition锁内复制本地值，不probe handle、不resolve或发RPC；[manager.snapshot](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/rollout.py)随后读取driver持有的ready/pending状态。应在driver操作之间读取；只有pool部分加锁，不是集群全局原子快照。closed/failed会拒绝读取，后台错误可能因此较早暴露，不是“监控永不影响报错时机”。

| 字段 | 本地计数点 | 不能当成 |
| --- | --- | --- |
| queued_tasks | 尚在派发队列 | 已在设备运行的工作 |
| launch_reserved_tasks | 已出队、尚未登记返回handle；调用可能已经发出 | 尚未发出任何远端调用 |
| running_tasks | 已登记、尚未被后台观察为ready的handle | 精确的GPU执行中数量 |
| completed_unrouted_tasks | 后台已观察ready，driver尚未取走 | 已resolve、组装、过滤且可训练的样本 |
| ready_groups / pending_roots | 已组装的逻辑组 / 尚未齐全的terminal sibling根 | 同一种任务数、chunk数或token数 |

native AR/DiT每个任务提交一个prompt树，所以trainer日志用prompts命名；agentic任务则是轨迹，不能把通用manager所有计数一律解释成唯一prompt数。slot_occupancy=(running+reserved)/sum(capacities)，是本地逻辑slot占用比，不是GPU利用率；capacity也不是trainer的max_inflight准入预算。

2026-10-23对[#529](https://github.com/Tencent-Hunyuan/UniRL/issues/529)的[候选尽调](../../open-source/unirl/tracker.md#补充尽调与派发边界)补核固定主线6fd82cba：异步AR先fork再按root树split，Agentic则显式展开K份独立任务，pool一次slot对应整个task。“most-free必把任意GRPO siblings拆散”不由前者推出；同worker也不自动保证KV就绪或100%命中。本轮未执行pool、Sample、engine或缓存实验，且已有贡献者公开表达实现意向，不新增并行方案。

### 路由策略与基准入口

**基准必须实际经过被改动的调度器，缓存命中也不能替代吞吐与尾延迟。** [#529在2026-10-24的新反馈](https://github.com/Tencent-Hunyuan/UniRL/issues/529#issuecomment-6381413964)建议将首轮范围收窄到Agentic siblings，并指出原RFC指定的同步AR配方不会经过RolloutPool。这是贡献者MciG-ggg提出的范围讨论，尚非维护者批准的策略或性能结果。固定main仍为`6fd82cba13f52f394747832344039408699ea378`。

| 路径 | 当前派发单位与调用链 | 修改RolloutPool能验证什么 |
| --- | --- | --- |
| 同步AR | ARTrainer直接调用rollout.generate；SGLang声明DP_SCATTER，经pytree_chunk→Batch.chunk→Sample.slice按完整root树分片 | 此路径不经过pool，原同步基准不能观测pool改动效果 |
| 异步AR | ARTrainer先fork K个siblings，共享async循环再按root树split，一个完整树提交为一个task | pool已不把该task内K个siblings分给不同launcher；不能按K条独立task解释slot负载 |
| Agentic siblings | AgenticTrainer把每个root重复K次并分别submit | most-free确可将这些独立轨迹分到不同launcher；收益仍取决于前缀复用与负载 |
| 单条Agentic普通回合 | 一个generate中运行本地harness，policy绑定inner.generate，工具后仍调用该绑定 | 工具回合本身不重新进入manager，不能据原RFC断言每回合频繁跨worker迁移 |

[同步入口](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_ar.py)明确构造ARTrainer；原`qwen3_grpo_4b_base_dapo_sglang.yaml`运行说明使用这个入口。因此即使测出该配方tok/s变化，也不能直接归因于只改pool的group_packing。[async提交](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/async_rollout.py)、[Agentic提交](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/agentic.py)以及[pool选择launcher](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/manager/dispatch.py)分别决定任务粒度和负载单位，不是算法名GRPO决定走哪个调度器。

[AgenticRolloutEngine](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/agentic/engine.py)构造本地inner，HarnessContext保存其generate绑定；[ToolAgentHarness](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/harness/tool_agent.py)在同一次run内循环generate/env.step/observe。ruiling-smartbear也已[收回此前普通回合需要session路由的判断](https://github.com/Tencent-Hunyuan/UniRL/issues/529#issuecomment-6376664385)。这不提供全生命周期黏着保证：manager的quiesce/finish及async权重发布会处理carried任务，不能从普通turn链推定重新提交、故障重试或跨版本一定仍用同一worker；本轮也没有证明这些场景必须增加新的映射方案。

同worker只是位置条件，不保证前缀KV已完成、仍驻留或命中100%；集中siblings也可能降低调度弹性，使较慢工具调用拖长尾部。后续若明确选此题，应先证实真实Agentic请求进入目标pool并记录分派，再分别观察cache/prefill、完成吞吐和尾延迟，而非沿用不经过pool的同步配方认证效果。严格packing或软亲和仍待社区定范围，不在本地演化竞争调度器。

本轮只核正文/全部4评论和上述源码，不运行pool/Handle/harness、模拟worker/cache或做GPU/NPU测量。[M5原条件参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)只补基准前提，不新增硬件队列；[完整接续证据](../../diary/2026-10-24.md)留日志。旧原生Sample/指标控制不作为本轮路由或缓存验收。

### lag 与横轴各有自己的时刻

[CompleteGroups.oldest_output_version](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/buffers.py)只取ready存储中generated Part的最小版本；不覆盖queued/running/completed-unrouted。trainer以`train_version - oldest`生成max_ready_lag_updates，单位为optimizer update，不是秒、rollout轮数或所有在途工作最大lag。没有generated Part则为None；[同版logger](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/utils/wandb_logger.py)跳过None，缺点不能补成0或沿用上个值。

[共享async循环](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/trainer/async_rollout.py)在本批训练、增加optimizer版本及reset transport后取pre_sync，再按需quiesce/finish/publish/resubmit，随后取step_end；即使本步不需sync，两个名字仍分别是这两个采样点。pre_sync才记录reserved/running/unrouted/occupancy，step_end不记录这些字段，不是把它们写成零。两份状态一起通过log_rollout(step=rollout_id+1)发送到rollout/async/*，绑定rollout/step；不是两次设备采样或optimizer-step横轴。记录发生在eval/save之前，因此不能由该点认证后续eval/checkpoint成功。

collect/refill的三个headroom分别来自容量、freshness、hard boundary，扣除outstanding后换算为prompt，实际新准入取三者最小值、整批提交。日志保存的是**这次提交前**计算的余量，而非提交后的剩余容量；任一项为0足以禁止新准入，不能只凭slot空闲判断调度停滞。新表达式与原min预算再扣outstanding在正batch_size和batch对齐前提下代数等价，源码变化不是放宽准入策略的证据。

### 增量只有一个消费者

snapshot保留累计事件；drain_metrics计算“当前累计−共享baseline”后推进baseline，初值为0。额外观察者应读snapshot：两个drain调用者不各有独立窗口。连续无事件时delta为0，不能据此判断生命周期总数为0；计数随新manager重置，checkpoint不恢复。

launch计handle登记，complete计观察ready，resolve仍可能失败；dropped_groups只计成功complete-buffer过滤移除的逻辑组，既不专指staleness，也不包括所有incomplete-root丢弃。carried计成功quiesce返回任务，finished计成功finish的输入，均是事件而非去重样本。baseline在成功读取时已更新，先于logger调用；logger还会吞下记录异常并warning，所以不是W&B持久化确认后才消费的可靠事件队列。此为源码契约，不是本机注入日志失败的复现，也未据此提出修复。

### 证据范围

本机Python3.9.6仅对manager __init__/buffers/dispatch/rollout、trainer.async_rollout及utils.wandb_logger共6完整模块内存compile通过；没有导入或执行它们，没有构造假handle/训练包壳、启动后台pool、Ray、模型、设备或W&B。最小复查（Git对象须已在本地）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
head = 'f404c9024c00bbf8a5d05315153444cee894865a'
paths = ['rollout/manager/__init__.py', 'rollout/manager/buffers.py',
         'rollout/manager/dispatch.py', 'rollout/manager/rollout.py',
         'trainer/async_rollout.py', 'utils/wandb_logger.py']
for path in paths:
    src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{head}:unirl/{path}'])
    compile(src, path, 'exec')
PY
```

head SHA256：dispatch.py `2515cab5013f1645f7e441b28833627a766ef0b61f83a6fcc6bb8cbc9112e707`；rollout.py `25e9701e0525ed4c2cdb7b1cd0d9e6ce0c074a7a8730abbc7f1ca9ce94739db2`；async_rollout.py `b1769c6a4fb3333d472e55672dc60c5a12e12c56cbf1456e2996f799e78e5a04`。作者正文的CPU mock检查及5次100-rollout/400-update、500条W&B记录均为作者报告，PR未提交该harness，本机未复跑；报告中没有staleness drop，不用平滑曲线认证全部过滤路径。

本题以监控契约学习收束，不新建CPU替身或GPU实验队列。若M5日后独立选择匹配的async真机任务，只在[原UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)保留观测入口；缺环境或不属原任务则维持未测，不为指标说明部署训练。

## 原生 IPC 的共识门与物化边界

**失败共识要放在实际通信边界之前，原始参数名不一定等于通信单位。** [#494](https://github.com/Tencent-Hunyuan/UniRL/pull/494)尝试加固vLLM native packed-IPC：2026-10-24读取OPEN/non-draft，作者CjhHa1，head `0832ea6b382bbd1ee024c171343794bf1cf1bdef`、base及祖先 `a77575acaaf3ed54386b866747dda2a22d5b6b0c`（ahead1/behind0），5文件334增22删。当前main `6fd82cba13f52f394747832344039408699ea378`的native_engine.py与该PR基线字节相同，没有新物化边界字段；本节不把候选当已合并行为，也不作完整PR验收。

### 两套边界对应不同的下一步

[IPC计划构造器](https://github.com/Tencent-Hunyuan/UniRL/blob/0832ea6b382bbd1ee024c171343794bf1cf1bdef/unirl/distributed/weight_sync/full/ipc.py)为一个raw state-dict项展开canonical输出，在最后一个输出标记`_materialization_boundary`。发送器另按字节预算分chunk，两种分组可以交叉。新增标记只进入发送器的边界索引，ParamMeta仍只携带name/dtype/shape；相对PR基线，完整publication协议模块字节未变，结构manifest也只归一化原来的五项schema字段，不包含payload内容。

| 共识位置 | 意图保护的下一步 | 条件与限制 |
| --- | --- | --- |
| buffer分配／reduce_tensor之后 | 开始读取惰性source | 捕获本地分配/导出异常，其他rank不先进入物化 |
| source边界 | 下一项的物化通信 | 依赖边界计划真正覆盖producer的collective |
| chunk边界 | 继承自vLLM的句柄汇总 | 即使chunk在同一raw项中间结束，也先报告本地异常 |
| 每次发送之后 | 下一块发送／最终验证 | rank0的发送异常须让其余rank知道 |

[发送器](https://github.com/Tencent-Hunyuan/UniRL/blob/0832ea6b382bbd1ee024c171343794bf1cf1bdef/unirl/distributed/weight_sync/transfer/vllm_native_engine.py)遇到本地错误时，选择不早于当前tensor索引的最近source边界与chunk末端中的较小者作为共识门；正常rank继续到同一门，期间先同步当前CUDA stream。这个策略只有在两门之间没有必须由所有rank共同进入的其他通信时，才能支持“错误rank先等、同伴继续到门”。`_consensus`通过Gloo组收集error，phase用于错误报告，并未比较所有rank的phase是否相等；一致的schema、遍历顺序和门序列仍是前提。已卡在collective里或进程已退出的rank不能靠Python异常汇报自动恢复。

### 原始项边界仍须与真实导出器核对

候选[原导出器](https://github.com/Tencent-Hunyuan/UniRL/blob/0832ea6b382bbd1ee024c171343794bf1cf1bdef/unirl/utils/peft_merge.py)的`raw_state_dict → _iter_rollout_tensors`并不总是“一次物化、随后纯本地拆片”。对于packed gate/up或down，先调用`_prepare_qwen_moe_dtensor`，再逐expert调用`_to_full_tensor`。后者若收到的仍是DTensor，会执行`redistribute(placements=[Replicate()] * mesh.ndim).to_local()`；前者自身也可能重分片。**所以“一个raw项的最后输出”不是静态证明下一次collective位置的充分条件。** 这三个producer helper相对PR基线AST未变，新增门没有在这里逐次接线。

这是条件性的源码疑点，不是已复现死锁：实际slice类型、placements、后端是否发起通信及错误位置尚未观测。若要判断该head的保证，须把真实DTensor导出调用序列与控制门放在一起核对，不能只证明索引满足最后一项。也不能用当前main后来变化的producer替代该PR固定head。

发送器每次copy后删除自身`flat/tensor`引用，避免额外持有上一个输出；这不保证生成器frame、DTensor存储或CUDA allocator已释放，不能从`del`推算峰值显存。构造器要求Gloo control timeout严格大于`update_native_weights`命令预算，TP4示例从900改960秒；长短排序为错误汇报留机会，不是整个publication的统一deadline、RPC取消或失败回滚。外层失败后由rank0尝试poison rollout，而不是继续承诺原地重试安全。

### 静态检查与作者测试范围

本机Python3.9.6，仅三个版本各四模块的AST/compile及两份原测试源码compile，共14次模块语法检查；chunk planner、schema validator和三个producer helper相对base AST一致，protocol.py相对base字节一致。未导入torch/vLLM或执行原pytest、Gloo/FSDP、DTensor、CUDA IPC、stream、模型及硬件。

作者报告8项CPU测试通过、GPU smoke未跑。原测试用自制Tensor/Buffer/Source、空vLLM基类和本地回调替换分配、stream同步、共识、句柄汇总及发送；能够检查单进程门标签与分支，不能据此证明多rank真实collective时序、内存峰值或超时退出。这里不复跑这些替身或另造通信模拟；[复现/hash](../../diary/2026-10-24.md)留日志，实机前提仅补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不应用候选、新开实验矩阵或竞争作者方案。

## Omni 原生 IPC 的副本与导出边界

**训练rank的集体参与、向哪个rollout副本发权重、权重内容是否正确，是三个不同问题。** 2026-10-22只读[#617](https://github.com/Tencent-Hunyuan/UniRL/pull/617)，读取时open、未合并，固定head `19f263968deecdecd26cebe8966f8ba012a5575d`；API目标base为`3ecbbc32b2fc373ba30b9842032552e041e84b12`，官方compare祖先`6624b28aab3299d116ae1e3d90aaa14926676911`（ahead4/behind36）。按祖先核16文件454增92删，研究范围是Qwen-Image全参数训练与vLLM-Omni原生IPC接线，不把落后主线的差异当本PR改动。

### 每卡副本不是共享 TP 引擎

[IPCWeightSync](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/full/ipc.py)以rollout的rank_local_weight_transfer选择发布者；[Omni engine](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/rollout/engine/vllm_omni/engine.py)声明该标志。两种拓扑不能混读：

| 接收布局 | 训练端参与 | 发布者与接收者 |
| --- | --- | --- |
| 本候选Omni：每训练卡一个独立TP1副本 | 所有训练rank物化参数、交换IPC handle并做错误共识 | 每rank向自己卡上的副本发布；本地fanout须为1 |
| 原共享vLLM TP引擎 | 仍需所有训练rank参与上述工作 | rank0负责发布，接收fanout与训练world匹配 |

[VLLMNativeIPCTrainerEngine](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/transfer/vllm_native_engine.py)的is_sender控制init/start/finish客户端动作，不把其他rank排除在_send_planned_packed之外。因此“只让rank0调用整个sync”会违反这条物化/collective接线；不能由只有一个发布者推导只有一个参与者。

原生路径先核同节点、每actor一个CUDA可见设备、物理UUID与接收worker按TP rank排序的UUID对应。[Omni backend](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/rollout/engine/vllm_omni/backends/native.py)还要求恰好一个启用WTE的diffusion stage。supports_native_weight_transfer只报告发现WTE配置，不认证资源、UUID或reload实现就绪。这里是同节点CUDA IPC，不是跨节点或NPU传输支持。

### 元数据规则与真实数据物化分开

[peft_merge.py](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/utils/peft_merge.py)抽出_base_state_dict与_iter_rollout_tensor_views，让raw_state_dict_metadata和实际raw/merged导出使用相同的embedding过滤与packed-expert命名/切片规则。metadata用全局shape创建meta tensor；wire dtype只改浮点字段，整数buffer保留原dtype。tie_word_embeddings=True按配置跳过精确名称lm_head.weight，不在此验证真实模型绑定关系。

共享模块也覆盖Qwen MoE packed专家布局：gate_up为[E,2I,H]，每expert前I行作gate、后I行作up，down为[E,H,I]。这是共享导出模块的规则，不表示Qwen-Image是MoE。metadata和CPU视图不需CUDA；实际_iter_rollout_tensors仍调用_to_full_tensor，其中有.cuda及DTensor redistribute，不能用meta检查冒充数据导出parity。

本机以完整原模块执行两个窄接口，见[固定Git对象复查脚本](../../experiments/scripts/verify_unirl_export_metadata.py)。手写真实nn.Module状态布局含embedding/head及浮点/整数buffer，不是Qwen模型：tied开关×wire dtype(None/fp32/bf16)共6组，名称顺序、shape、meta设备与dtype全部通过，整数count始终int64。另对实际CPU arange张量[E=2,I=3,H=4]核gate/up的4片与down的2片，值/名称/形状及共享storage通过；奇数gate宽、与I不符的gate宽/down宽三种输入均拒绝。没有替换原函数或模拟通信，也没有运行raw_state_dict/merged_state_dict/CUDA导出。

```bash
python -B experiments/scripts/verify_unirl_export_metadata.py /path/to/UniRL-git
```

Python3.12.13/torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）；原模块SHA256 `e03bf04d38966e548bf0a0f742260dc805cf29ec96270e5ffe3eb8526abc8798`，脚本校验此hash并禁止Git隐式拉取。8个修改的完整Python模块另经Python3.9.6内存compile通过；只有上述元数据/CPU视图接口实际执行，没有DTensor进程组、PEFT、Ray、vLLM、模型forward/reload或设备发现。

### 发布完成不是内容一致或事务提交

[structural_manifest](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/transfer/vllm_native_protocol.py)哈希有序的name/shape/dtype/numel/nbytes，不读张量payload；相同fingerprint不是参数值逐字节相等。发送端按schema验证实际流，各训练rank比对结构manifest；Omni检查RPC报告的失败，再做rank共识。上游layerwise reload和CUDA同步由配套Omni实现负责，本轮没有复审或运行该外部实现，不能把本仓库返回检查当内容回读。

IPC内部_next_model_version从1起、成功发布后递增，是发布序号；[DiffusionTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/trainer/diffusion.py)另外读取optimizer step count，sync/push正常返回后才调用rollout.set_version用于生成标记。每轮多次optimizer update时，两者不必相等，不能把header序号直接当optimizer版本。

失败可能已经部分覆写接收权重；失败共识后的poison/shutdown是停止使用并重建运行时，不是回滚到旧权重。Omni保留level-1 sleep，以免仅更新transformer却丢掉未发布的冻结text encoder/VAE；这也要求主机RAM留备份余量。原生preflight检查最大单tensor不超buffer，并用3×buffer+按名字估计的最大layer+headroom核空闲显存；不是全模型训练峰值或任意显卡容量保证。boot还默认设置VLLM_ALLOW_INSECURE_SERIALIZATION=1以传递pickled IPC handle，这条控制链应处于可信环境，不能当成不可信HTTP输入的安全认证。

### 配方与作者实测范围

新增YAML的1x1覆盖LoRA为null、cpu_offload=false、256MiB IPC buffer；1x2再指定两FSDP rank、colocate及train_resident。只读defaults/override，不宣称本机已完成Hydra组合或单卡运行。原生Omni路径限一个diffusion WTE stage和每卡TP1副本，不能外推AR、多WTE stage、量化或rollout TP>1。

作者报告UniRL此head搭配Omni `8607b97ea5`（[配套PR #7686](https://github.com/vllm-project/vllm-omni/pull/7686)）、vLLM0.31/torch2.13，在两张约140GiB的L20X上跑tiny/full×RPC/HTTP四个短实验：三optimizer step、每副本两次publication、版本[0,1,2]；tiny审77个source tensor、full只审3个选定tensor。HTTP只覆盖生产WTE控制请求，生成仍native，不是通用HTTP生成backend或服务CLI验收。上述报告不是本机结果，也不认证收敛；1x1配方未独立GPU验收，GPU发布失败注入未跑。作者CPU/Gloo/PEFT harness未入仓库，不能记作本机或上游已提交单测。

真实接线/权重及失败关闭缺口记[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署配套分支、下载模型或新排训练。本题以副本/元数据边界学习收束，不继续造IPC替身。

## 原始 block 权重训练与冻结外围

**冻结参数不等于切断该模块的输入梯度，也不等于整个模型已经能训练。** [#593](https://github.com/Tencent-Hunyuan/UniRL/pull/593)（2026-10-23读取open draft）为MiniMax-H3增加原始`transformer_blocks`权重训练。固定head `da8d285ac529c2f06819bf3a098e3cc657ae6618`、base及实际祖先 `2d9e57fdee65ea69254fdb958d3fca9ce4872661`，ahead3/behind0；四文件仅新增42行，无已提交测试。它不是全transformer微调，也不是LoRA方案的新名字。

### 参数选择与原分片布局

[bundle](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/models/minimax_h3/bundle.py)在meta/eager两条transformer构造分支之后、backend包装之前执行：

```python
if config.full_finetune_blocks:
    transformer.requires_grad_(False)
    transformer.transformer_blocks.requires_grad_(True)
```

因此text refiner、输入/输出投影和timestep模块保持冻结，只有block子树的原参数可训练；text encoder及两个VAE沿用原冻结路径。默认false只是不执行新增选择，不代表自动构造LoRA。[新配方](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/examples/diffusion/minimax_h3/minimax_h3_t2va_trainside_blocks.yaml)同时设true和`backend.lora_cfg: null`：后者单独取消backend的adapter注入，不能只把bundle开关当成对所有LoRA/EMA配置的互斥保护。

继承配方只wrap `MiniMaxH3TransformerBlock`，保留`root_wrap:false`、`cast_forward_inputs:false`、BF16计算与FP32 trainable masters。外围投影/timestep原有混合精度不由root统一转换。[原fsdp_wrap](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/train/backend/fsdp/wrap.py)在默认进程组已初始化且world_size>1时，拒绝仍可训练却不属于DTensor分片组的参数；冻结外围解决这项布局前提，但不证明collective、权重装载或完整反向正确。`trainable_module()`仍返回整个transformer；原优化器另以`requires_grad`过滤参数，不按返回模块的名字推定所有权重可训练。

### 冻结输出头仍要传递梯度

固定[vendor前向](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/models/minimax_h3/vendor/transformer_minimax_h3.py)在block之后仍执行`norm_out`、视频/音频输出投影和行选择，没有为冻结头插入`no_grad`。冻结叶子参数只是不为它积累参数梯度；只要输入需要梯度，这些运算仍能把链式法则传回block。直接把整个冻结输出头放进`no_grad`会切断这条链，不能作为等价省显存办法。`train()/eval()`控制的模块行为又是另一件事，新增两条选择语句没有切换模式。[PyTorch 2.9 autograd说明](https://docs.pytorch.org/docs/2.9/notes/autograd.html#locally-disabling-gradient-computation)

这还关联首个block的checkpoint：冻结输入投影且原输入不求导时，输入activation可以没有`requires_grad`，但block参数仍需训练。原FSDP checkpoint helper用`use_reentrant=False`，不要求输入中先有需梯度tensor；重入版本有这项要求。[PyTorch checkpoint说明](https://docs.pytorch.org/docs/2.9/checkpoint.html) 这是原helper已有行为，不是本PR新增修复或本项目提出的替代方案。

### 本机 CPU 控制与作者真机边界

[复查脚本](../../experiments/scripts/verify_unirl_block_freeze.py)固定Git对象，无隐式fetch；Python3.12.13 / torch2.9.1，显式CPU FP64。只将原选择条件/两语句应用到声明的三Linear数学夹具，输入2、外围权重3/7、block权重5，不是tiny H3。完整原`OptimizerConfig`与optimizer模块正常执行，预加载叶子模块绕过UniRL父包；checkpoint仅取原两个helper的AST，未执行FSDP wrapper。

| 控制 | 实际结果 |
|---|---|
| 开关关闭与开启 | 关闭保留三份可训练参数；开启仅保留block，Parameter身份与train模式不变 |
| 冻结外围的正常前向/反向 | 输出210，block梯度42，与全可训练参考一致；外围grad为None |
| 原optimizer factory及一次AdamW | lr=.01、weight_decay=.1，只有block进入参数组/状态，5更新为4.985000000002381，外围仍3/7 |
| 输出头改用no_grad的负控制 | 输出无反向图，backward按预期拒绝，block没有梯度 |
| 原非重入helper与重入负控制 | 前者保留梯度42；后者在本例无需梯度输入时告警并失去反向图 |

```bash
python -B experiments/scripts/verify_unirl_block_freeze.py /path/to/UniRL
```

两次执行结果一致。修改的两个完整Python模块另仅内存compile；optim、FSDP wrap和vendor transformer的base/head Git blob逐项相同。冻结不保证外围激活/反向计算免费，也不证明实际H3参数规模、混合精度、meta物化、LoRA默认输出、FSDP/TrainStack、replay、恢复或学习质量。这里没有运行原H3或作者外部harness、安装新依赖、设备探测/模型加载。

作者报告tiny H3参数选择、CPU/Gloo FSDP/TrainStack更新与恢复成功，但harness和原始日志在仓库外，本项目未复跑。[PR正文及新增README](https://github.com/Tencent-Hunyuan/UniRL/pull/593)明确：原checkpoint `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`在8×H100 80GB、256×384/124帧、10 CPSSDE步、FP32 masters及resident VAE条件下能load/generate，首次backward仍OOM；开启expandable segments的后一次也未到Adam，报告75.94GiB allocated与1.43GiB reserved-but-unallocated。16卡更新验证待完成，不能把LoRA配方历史收益或加载成功挪为原始block权重训练成功，更不能外推4090可跑。真机前提补[原M5入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动启配方、扩大卡数或演化offload方案。

源SHA256：bundle `9d7681dda134e5a38c1bd0b20c07330193ace5d93104250a9d542e92c847e0cc`；config `f84ce6b98089e73f49016b22a977f5b002192e1220f32df6c9d644c7a3b80864`；optimizer `a0f0fcf664da5049c1f3652d2e5f3fe1e469e721ea91c0b613294374f4a5da05`；wrap `0a0a6ec9f587b2ce7fe26f8bc759ca3a701ff1b9b67e769e19b476120ca5a915`。

## 全词表条件熵与有效 token 汇总

2026-10-26接续：[再次同步主线后的代码身份](#再次同步主线后的代码身份)核当前候选cc0df170／base8cfbac177；13个专题文件与前次d2f0a489逐blob相同，本轮不重跑旧20项或将历史GPU报告迁作新组合验收。以下00e5597f记录保留首次学习与实验的版本范围。

**一次采样的负log-prob不是该位置的全词表熵，微批均值的均值也未必是全局token均值。** [#563](https://github.com/Tencent-Hunyuan/UniRL/pull/563)（2026-10-23读取open）新增可选AR熵监控。固定head `00e5597f6ff6de639d7997a2ea21266d688e9f72`、base/实际祖先 `94e8f26b8768a3695b5985c2fa072e07d9dbfa0d`，ahead3/behind0；13文件270增43删，无提交测试。本节是已有方案的源码学习，不是本项目新增修复或对熵坍塌故障的定性。

### 监控哪个分布

[Qwen3 replay](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/models/qwen3/ar.py)在原LM-head分块内对输出先`.float()`再除以温度，求`p = softmax(logits/T)`，以`sum(entr(p))`计算完整词表条件熵，单位nats。`entr(0)=0`可处理softmax下溢的零概率，不需用`0*log(0)`；这不修复非有限logits。[PyTorch entr定义](https://docs.pytorch.org/docs/2.9/special.html#torch.special.entr)

正温度沿用sampling_temperature，否则回退T=1；没有应用top-k/top-p截断。因此greedy rollout时该指标可以非零，它不是确定性采样器的熵。对二元`p=(.9,.1)`，条件熵约0.325083，而采到第一项的surprisal仅0.105361；只有在同一分布下取适当期望时，二者才有熵的关系，不能把单次选择或另一rollout分布的样本均值直接当完整词表求和。

`_categorical_entropy`使用`no_grad`，结果detach后作为`ReplayResult.entropy`返回；原log-prob仍走训练图，`ReplayResult.logits`不因此被填充。GRPO/GSPO/DPPO/DRPO/CPPO/PPO的`monitor_entropy`默认false，开启才请求额外字段，不添加entropy loss。仅输出每token标量不等于没有词表临时量：每块仍多做全词表softmax/entr，开销与词表、块形状及后端有关，不能外推作者短序列时延为普遍零成本。外层replay及其chunk checkpoint循环本机未执行。

### 分母是有效位置的观察次数

[原metric helper](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/algorithms/base.py)接收packed response entropy，对`loss_mask`转bool筛位置；None表示全部位置，包含实际存在的EOS。mask不是软权重：非零浮点值也转True。prompt/padding由replay布局排除，tool等位置由原mask控制，不能把helper视为独立核验数据身份的入口。

每个微批保留`policy_entropy_sum`及`policy_entropy_count`，mean为sum/count；空响应或全False mask的sum/count均0，mean用0占位，不表示观察到确定性策略。masked_fill去掉非活跃NaN/Inf，活跃NaN仍传播，不是健康认证或异常修复。[原聚合器](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/utils/metrics.py)仅对这两个键求和并重算mean，其他普通数值键仍按原均值规则处理。

本机合成微批共用4词表：一枚位置在前两项均匀，熵ln2；另三枚位置全词表均匀，熵ln4。对同样的两份字典，旧通用聚合器等权mean为1.039720774；新聚合器sum=4.852030277、count=4，token mean为1.213007569，分层聚合一致。这是给旧/新聚合函数同样新增指标字典的对照，不代表base原本已有此熵监控入口或完整训练bug已复现。

`TrainStack`在微批结束后以两次stack/tolist搬回sum/count，再交原backend的loss-reduction mesh求和；当rank_info存在时，仅TP rank0、SP rank0且pipeline最后阶段贡献，其他副本置零。CPU字典检查不认证真实mesh/副本去重。跨optimizer updates汇总时，同一数据位置在不同策略版本下会再次计数：结果是各次条件熵观察的加权平均，不是唯一语料token计数，也不是平均策略分布的熵。

### 本机局部复查与真机范围

[复查脚本](../../experiments/scripts/verify_unirl_policy_entropy.py)固定本地Git对象、禁止隐式fetch；Python3.12.13 / torch2.9.1，全部显式CPU。AST原样选择两个LM-head chunk函数、原温度赋值、entropy和metric helper；完整原ReplayResult与base/head metrics模块执行。LM-head采用声明的Linear数学夹具，不伪装Qwen backbone，不运行含CUDA探测的外层forward或补模型替身。

- FP32/BF16 × T=-2/0/.5/1/2 × flat/padded两种chunk形状，共20组：监控开关前后log-prob、输入及Linear参数局部梯度逐位相同，entropy为detach FP32；另对实际head输出的FP64全词表参考，以atol/rtol=3e-7核对。
- mask转bool、detach、空/全masked归零、活跃NaN传播、类型/缺字段/两类形状错误的4项拒绝均符合原helper；未更改断言或soft-fail。
- 1/3长度微批的加权与分层聚合结果如上；11个改动Python完整模块仅内存compile，不把这项检查当成六种完整算法或原作者测试通过。

```bash
python -B experiments/scripts/verify_unirl_policy_entropy.py /path/to/UniRL
```

两次执行一致。没有Qwen模型、packed/padded外层replay、checkpoint循环、loss/optimizer、FSDP/TrainStack/Ray、TP/SP/DP collective或GPU/NPU运行，也未安装依赖。作者另报tiny L4、Qwen3-0.6B单/双A100与Ray/FSDP一次更新证据，原harness在仓库外；这些不是本项目复跑，也不覆盖ARTrainer的完整rollout/reward/sync循环或VeOmni。后续真机核对仅补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署候选或把它当#8250根因诊断。

SHA256：Qwen3 ar `49557b0cf37b0823940307531f99efa2505242411492b2fe506f5f14387e5972`；metric helper所在base `da4a741a0315c0ad8f530c7952593c8bade2bdaffffcb9e68730af527d306b30`；新metrics `2897870ce27577033b2d7963df9dbc71178fef3a523b53924d786172ae585626`，旧metrics `f19ac2211065ea4927927648a3ec108ecb5b1b7bd7e8f1d3b73dd3ef198beab9`。

### 单阶段拆分后的熵统计接线

2026-10-24，[#563](https://github.com/Tencent-Hunyuan/UniRL/pull/563)仍open/non-draft，head更新为`d2f0a489211200f5655e755b3337f875ab7144c4`，base `6fd82cba13f52f394747832344039408699ea378`。`4f6ebd80`合并旧head与main，末提交仅整理三个算法的导入顺序；对当前base仍为13文件270增43删。旧`unirl/train/stack/base.py`中的熵实体化、聚合与副本过滤/reduce段迁至`unirl/train/stack/single.py`。

AST比较确认该完整统计段、`policy_entropy_metrics`、`_categorical_entropy`和完整`ReplayResult`与原固定head相同；这是限定源码结构比较，不认证整个replay/算法/TrainStack等价。旧20组数值控制与脚本仍锁原版本，本轮未重跑。作者本次另报六算法loss/梯度、拆分后单栈实体化及两rank Gloo检查，且明确未重跑GPU；历史A100/L4证据不迁移为新head验收。

### 再次同步主线后的代码身份

2026-10-26核[#563](https://github.com/Tencent-Hunyuan/UniRL/pull/563)仍OPEN／非draft／未合并，head `cc0df170a507220383dd0f1a1df5aec67e8d1c90`，API目标base `8cfbac17765004afb68fef08a4e011a4c8daf555`，相对base ahead6／behind0、13文件270增43删；ahead数包含历史与合并提交，不表示新增六项熵修复。关联[#564](https://github.com/Tencent-Hunyuan/UniRL/issues/564)明确要求监控而非正则化，已有作者实现；普通评论与reviews为空，不推导已获维护者批准。

[新旧head比较](https://github.com/Tencent-Hunyuan/UniRL/compare/d2f0a489211200f5655e755b3337f875ab7144c4...cc0df170a507220383dd0f1a1df5aec67e8d1c90)显示从d2f0a489前进两提交，仅合入[#621奖励加载](https://github.com/Tencent-Hunyuan/UniRL/pull/621)的8cfbac177及merge提交；十个实际变更文件都属于reward／ReFL VideoAlign。与候选[相对新base的13文件差异](https://github.com/Tencent-Hunyuan/UniRL/compare/8cfbac17765004afb68fef08a4e011a4c8daf555...cc0df170a507220383dd0f1a1df5aec67e8d1c90)是两种比较，不能把前者的奖励改动算成熵功能扩大。

从未截断Git tree逐blob核对，两head的全部13个熵专题文件相同：六算法及base、算法／Qwen3说明、Qwen3 ar、ReplayResult、`train/stack/single.py`和metrics。关键blob为ar `8316db745f00d62634b30f8180210ec51338156c`、single `765028742e803466ce21f81df17790686c936e52`、metrics `d6dde64944fe7c4af631f4b845e7758dbbb542a4`。单阶段熵实体化／副本过滤没有因此新增改动；本轮只做Git对象身份核验，没有执行原20个CPU控制、完整算法或TrainStack。

PR正文保留前次sync的CPU/Gloo报告，并明确该sync未重跑GPU；早期L4／A100及固定响应FSDP-Ray单步仍是作者外部结果，不能改写为cc0df170上完整ARTrainer rollout／reward／weight-sync或VeOmni验收。新奖励生命周期与熵文件保持各自范围，也不因合并主线就认为二者组合已训练通过。更新[原M5熵参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)的版本与观察条件，不新增队列／环境或候选组合；后续仅跟真实代码／反馈变化。

## 有界动态采样与数据游标

**补样预算限制生成批次数，不保证找到足够有效组；恢复数据位置也不等于恢复随机生成。** [#562](https://github.com/Tencent-Hunyuan/UniRL/pull/562)（2026-10-23读取open）为同步AR GRPO增加可选动态采样。固定head `e8cd33daa6067a0da69466f24cc6f1bbd3a082bc`、base及实际祖先 `772fd7c50ec8221a032179aeec8320df935b3abf`，ahead2/behind0，7文件207增15删，无提交测试。它只是筛组/补样这一片，不是完整DAPO recipe或学习收益证明。

### 筛选整个组而非单条正确答案

[ARTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/e8cd33daa6067a0da69466f24cc6f1bbd3a082bc/unirl/trainer/ar.py)先generate并由原reward服务评分，再读取最终`part.rewards`。一组必须包含预期全部siblings，reward全部有限且至少一个值与首项精确不等才保留。它不检查“正确/错误”标签，也没有epsilon或最低方差阈值：全对同分与全错同分都可丢弃；连续reward只要存在可表示差异便可通过。仅核源码，没有把手写奖励数组控制当作原Sample/Part或trainer测试。

保留前`batch_size`个有效整组。即使中途已够数，仍统计该生成批的其余组；多余有效组丢弃，不缓存在下一次update。优势在最终拼好完整训练批之后才计算。`unfiltered_reward_mean`包含所有生成组中的有限reward，连被整体拒绝的非有限组内的有限成员也会进入这个指标；通常训练reward则只看保留组。过滤提高训练reward均值不能直接当作策略改进，需与独立评测和真实生成成本分开。

### 批数上限和失败边界

配置`dynamic_sampling_max_batches=K`的0表示关闭，正整数K**包含初始批**，不是首批之外再补K次。每次不足都重新取完整`batch_size=B`输入组，而不是只取缺少的数量；正常完成的K批至多生成`K×B`组、`K×B×siblings`条completion。这是源码计数推导，不是吞吐或显存实测；预算不限制单次调用耗时，远程切片可能保留整批backing storage，不能理解成内存字节硬上限。

| 数量 | 在本路径中的含义 |
|---|---|
| generated_batches | 本次训练批实际生成并评分的整批数，包含首批 |
| generated_groups / valid_group_fraction | 所有已评分组及其有效比例，包含最终丢弃的surplus |
| batch_size | 每次生成的输入组数，也是训练前必须凑齐的保留组数 |
| data_batches_consumed | 跨训练轮次累计的数据源取批次数，不是optimizer step数 |

耗尽K批会抛错，调用栈尚未进入优势计算及`stack.train_track`，不会拿残缺批继续该次训练；但已消费的数据、生成/评分及sleep/wake成本不回滚。主训练循环finally做日志/runtime清理，正常checkpoint保存位于成功train_step之后，预算失败没有新增“保存当前消费游标”的路径。因此恢复依据已有checkpoint的游标，而不是失败现场最新内存计数。

### 游标恢复与策略驻留是不同状态

新增driver metadata保存`data_batches`及`batch_size`，恢复时对新建数据源重复调用get_samples跳过这些批次。带此字段的checkpoint要求动态采样仍启用且batch_size相同；这里未校验K相同、数据集revision或随机生成状态。只有重建相同seed/数据读取流等前提成立，批次计数才对应同样输入请求；不是序列化数据源全部内部状态或保证同样completion。refill另以`d<已消费批数>:<sid>`命名空间区分再次访问的数据行，不是内容哈希或永久dataset ID。[原保存/恢复路径](https://github.com/Tencent-Hunyuan/UniRL/blob/e8cd33daa6067a0da69466f24cc6f1bbd3a082bc/unirl/trainer/base.py)

同一采样批的refill不抽取更新后的训练权重；仅首次attempt按原sync条件同步，后续如有已staged的LoRA adapter便在wake后重推，以应对sleep释放adapter。SGLang memory saver启用时，配置guard要求weight CPU backup；明确字段优先于engine_kwargs。该检查位于需要rollout sleep的构造分支、支持识别的SGLang配置内，不认证引擎实际保存了同一权重。两类生命周期都需真实后端证据，不能用游标正确替代。[原配置guard](https://github.com/Tencent-Hunyuan/UniRL/blob/e8cd33daa6067a0da69466f24cc6f1bbd3a082bc/unirl/config/validation.py)

### 当前验证范围

本机仅源码及base/head各5个完整Python模块的AST/内存compile通过；没有导入trainer/Hydra/Ray、运行Sample筛选/concat、TensorRef transport、备份/重推、训练或checkpoint恢复，也没有设备/服务器探测、新依赖或trainer替身。单文件语法核对可复现为：

```bash
git show e8cd33daa6067a0da69466f24cc6f1bbd3a082bc:unirl/trainer/ar.py | python3 -c 'import sys; compile(sys.stdin.read(), "ar.py", "exec")'
```

当前API正文明确区分版本：作者在e8cd33d报告39项CPU、TensorRef/adapter和配置组合检查，harness在仓库外；GPU full/LoRA/恢复/预算耗尽报告属于旧10e57c6，不是新版刷新后的GPU复跑。历史两L40S/Qwen3-0.6B的小型算术任务含受控reward路径，不能作当前head的端到端验收、数值GPU parity、收敛或吞吐证明。网页抓取仍停在旧正文，不据缓存页覆盖API的版本限定。真机缺口补[原M5入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署该配方、重开vLLM版本问题或改M5专项。

Head SHA256：ARTrainer `068fbc203478e7dff538cc24f3d23e2a2cdce05e77eaeb578935f44208df4c0c`；trainer/base.py `a2de136a3b2ad01a6d36e3164b7b691b6401285e0e8a5010a29a4a6a2337e07c`；validation `ba9be8d0bcd2fd3b4c1bf354e0b05fe0b7cefe7b6f33790e148db422be345fd6`。

## Replay 锚点与 rollout 采样校正

**PPO裁剪的策略变化与rollout采样分布的差异是两种比率，先裁剪再加权不能随意合成一个比率。** [#557](https://github.com/Tencent-Hunyuan/UniRL/pull/557)（2026-10-23读取open）为AR GRPO增加默认关闭的token校正。固定head `300170e99209f26908d637971b3146d2ac7f98ff`、base/实际祖先 `852c0454a6cb63592b2582358f88623870848df1`，ahead2/behind0；仅GRPO与README两文件46行新增，无提交测试。这里不修改anchor生命周期或判断M5的#523专项。

### 两个比率的位置

[原GRPO路径](https://github.com/Tencent-Hunyuan/UniRL/blob/300170e99209f26908d637971b3146d2ac7f98ff/unirl/algorithms/grpo.py)以冻结replay anchor `old`计算`r = exp(new_logp - old_logp)`，先形成原PPO clipped token loss。新增分支再乘以冻结的`w = exp(min(clamp(old_logp - rollout_logp, -20, 20), log(cap)))`。启用要求`old_logp_source='replay'`、有限正cap及已保存rollout log-prob；默认None不进入新增分支。它不重新采样，也没有把PPO裁剪范围改成cap。

若未裁剪/限幅，两比率相乘可形式上约去old，但实际目标有PPO clipping和权重cap，顺序不交换。本机单token正优势控制令当前策略等于anchor、anchor/behavior比率3、cap2、PPO上界1.4：原顺序loss=-2、对当前log-prob梯度=-2；改成直接对behavior做PPO clipping则loss=-1.4、梯度0。后者不是同一目标，也不是本项目提出的替代实现。

新增权重计算显式detach，因此不会通过这个乘数向old或behavior回传梯度；完整PPO仍依赖old为冻结输入，这不能反过来证明prepare_segment或跨update锚点生命周期正确。权重截断引入偏差，且此式不修复tokenization、温度、top-k/p支持集不匹配；不是序列IS、拒绝采样或所有状态分布差异的无偏修复。

### 分母和诊断保留什么含义

[原reducer](https://github.com/Tencent-Hunyuan/UniRL/blob/300170e99209f26908d637971b3146d2ac7f98ff/unirl/algorithms/base.py)的三种模式未变。权重乘token损失后仍按原mask计数、逐序列均值或horizon归约，不除以权重之和。本机两枚单位损失乘`[1,2]`后token-mean=1.5，不是自归一化IS的1。

权重采用FP32 anchor操作数，先做[-20,20]安全限幅再做cap。极端差值的权重因此有安全地板/天花板；`rollout_is_clipped_fraction`比较的是安全限幅后的log_weights和log(cap)，不是所有原始差值饱和的计数。构造器允许任何有限正cap，并未要求cap≥1：相同anchor/behavior在cap2时权重1，在cap.5时权重.5，不能无条件称同策略总是单位权重。这些是实现范围核对，不推荐极端cap或推断精度收益。

新增mean/max、clip fraction及ESS只描述当前micro的active权重；有限输入的全masked micro给这四项0。归约仍沿用mask数值，不认证任意非二值/非有限mask。归一化ESS公式为`(sum w)^2/(count*sum w²)`；两个micro分别有`[1,1]`和`[2,2]`时，各自ESS fraction均1，拼接tensor的值却为0.9。平均两个micro的1不能冒充全局ESS，本机未运行分布式聚合。

### 本机局部 CPU 对照

[复查脚本](../../experiments/scripts/verify_unirl_rollout_correction.py)固定本地Git对象，Python3.12.13 / torch2.9.1、显式CPU。执行完整原loss-mode模块、AST原样选取的clip/active-token/reducer三个函数，以及GRPO新增完整校正if分支。仅两个属性容器提供cap和保存的behavior/mask张量，不冒充TextSegment、GRPO或stage；new/old等是声明的数学输入，不执行prepare_segment或完整compute_loss_and_backward。

36组值/当前log-prob局部梯度对照通过：三种reducer×无/部分/全零mask×None/.5/1/2四种cap，含非对称PPO裁剪，参考权重用独立Python标量计算，atol/rtol=2e-7。另核新增乘数对两anchor操作数detach、两比率顺序反例、原分母、同策略两cap、±100差值安全限幅及micro ESS反例。最终脚本重复通过；None控制只验证局部数学，不称完整base/head disabled路径逐位等价。

```bash
python -B experiments/scripts/verify_unirl_rollout_correction.py /path/to/UniRL
```

三个输入模块均做编译检查，完整GRPO/base模块不执行；base.py与loss_agg.py的base/head Git blob相同。没有完整GRPO构造/输入guard测试、模型/replay、optimizer、FSDP/TrainStack、transport、anchor冻结/恢复或设备执行，没有安装依赖/探测服务器。

作者外部harness另报83项CPU检查、单独Gloo、Qwen3-0.6B单/双L40S及100轮在线小算术实验，本项目未复跑。[PR最新正文](https://github.com/Tencent-Hunyuan/UniRL/pull/557)的受控单seed报告中，recipe LR=1e-6时启闭均未坍塌，最终held-out avg@4为关闭0.9375、cap2为0.796875，且cap2未实际截断token；不证明校正提升效果，也不能证明普遍有害。较高LR=1e-5的坍塌pilot发生在关闭校正时，不得归因为这个可选分支或挪为#8250诊断。真机核对仅补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不启训练或改anchor专项。

SHA256：GRPO `31531e0edc1301ce5f80012499ae89d6266715dfac871aaea266b5b2ef3d87c7`；base.py `326291439342130064abb865a5ef69fb5a49475edb65f040d38e0db39edfae52`；loss_agg.py `8049957fa4ecbf2086afab6e066fa1beeb9d35bdabebfd3d8890f7ee078ecd3f`。

## 回合边界续跑与行为版本

**多轮 agent 可以在不同回合使用不同策略，但暂停点必须保留工具观察，训练也必须保留每个回合自己的行为 log-prob。** [#558](https://github.com/Tencent-Hunyuan/UniRL/pull/558)（2026-10-23读取open）增加有界 agentic partial rollout。固定head `76a3c755210728bf03dcaaaeefe41187472d1dbb`、base及实际祖先 `852c0454a6cb63592b2582358f88623870848df1`，ahead3/behind0；12文件266增57删，无提交测试。它是回合边界续跑，不是中断一次生成的token流。

### 暂停点与环境状态

[ToolAgentHarness](https://github.com/Tencent-Hunyuan/UniRL/blob/76a3c755210728bf03dcaaaeefe41187472d1dbb/unirl/rollout/harness/tool_agent.py)在每轮开始检查暂停，包括首轮生成之前；已有回合须执行完`generate → env.step → observe`才到下一个暂停点。恢复时跳过`env.reset`，从Sample保存的观察继续，已生成回合计入`max_turns`。这条正常续跑路径不重做先前工具调用，但不是故障重放或外部副作用的exactly-once保证。

[ToolEnvironment](https://github.com/Tencent-Hunyuan/UniRL/blob/76a3c755210728bf03dcaaaeefe41187472d1dbb/unirl/rollout/env/tool_environment.py)仅在没有`StatefulTool`时声明支持续跑；worker本地session不能随意迁移。自定义环境同样须保证Sample包含所需状态，且每次harness退出的`close`不使续跑失效。[RolloutManager](https://github.com/Tencent-Hunyuan/UniRL/blob/76a3c755210728bf03dcaaaeefe41187472d1dbb/unirl/rollout/manager/rollout.py)的quiesce先停发任务、设置暂停标志，再drain当前工作；权重同步要求无queued/in-flight工作。因此慢生成或工具调用会推迟下一次发布，window不是墙钟截止时间。

### 窗口与执行重叠

[AgenticTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/76a3c755210728bf03dcaaaeefe41187472d1dbb/unirl/trainer/agentic.py)默认`rollout_window_size=1`，维持barrier。W大于1时，窗口开始一次准入最多`B×W`个root prompt（B为batch_size），每个再展开siblings；每次收集B个完整组用于训练，窗口内不持续补新root。窗口还会被save/final边界缩短，不是无限流式调度。

| 布局 | 收齐当前训练批后 | 下次发布前 |
| --- | --- | --- |
| 同卡 colocated | quiesce未完成轨迹，rollout sleep后训练 | wake、同步、重新提交carry |
| 分卡 separate | 剩余rollout可与训练重叠 | quiesce后同步，再续跑 |

分卡配置按`train_fraction`切训练/rollout资源，使用NCCLWeightSync；同卡要求TensorWeightSync。[新配方](https://github.com/Tencent-Hunyuan/UniRL/blob/76a3c755210728bf03dcaaaeefe41187472d1dbb/examples/deep_research/deep_research_async.yaml)取0.5和W=2，并关闭rollout memory saver及CPU backup。配方存在不证明当前机器的资源、引擎或通信已匹配。

### 每个回合的行为身份

W大于1的构造guard要求精确GRPO类、`old_logp_source=rollout`和cache flush。训练前逐个generated Part检查行为版本存在且不晚于当前训练版本，并要求保存的segment log-prob；原版本复制到每行`metadata.behavior_version`，再清除待拼接Part的单一`output_version`。这样混合策略轨迹不会被伪装成一个统一发布版本，原行为log-prob仍作GRPO分母。版本标签只记录来源，不证明真实参数安装内容。此模式与上一节#557的replay-anchor前提不同，不能直接叠加两候选。

`_train_version`从backend optimizer step count恢复，并按`result.optimizer_updates`递增。`agent/max_behavior_lag`在`stack.train_track`返回后、driver递增版本之前，用原轨迹版本计算；它不是本批更新完成后的lag。W数训练批，行为版本数optimizer update，不能把W直接当版本差上限；分卡时`_carried`已重新提交，`carried_trajectories`也不等于manager全部未完成轨迹数。

### 排空与checkpoint

窗口最后一批在训练前要求`_carried`及manager均空，最后一次训练消费完远程视图后才reset transport buffers，然后进入保存路径。**生产者排空与消费者读完是两个条件**，不能只见无in-flight就提前释放训练输入。save硬边界不携带未完成trajectory；恢复时重建数据源并跳过已训练批，依赖同样的读取流，不是保存进程中任意暂停点或恢复工具副作用。

### 验证范围

本机仅阅读完整差异和相关原方法，并以Python3.9的`compile`检查两版各7个改动Python模块；不导入模块，无manager/harness/训练或设备执行，未新增替身脚本。head的trainer SHA256为`294f87d21a6e48c1a69db5d6c8feeb71e705cd9cf18172ddfbe88da3b897b130`，harness为`df6b23ebc40d317ae694f4eb7ee84136f1701ed680518825347832e7dd4c1f5e`。

作者另报外部CPU harness及L40S/A100小模型验证，不是本项目复跑。A100恢复/重叠测试对应`9b68dc9f1b5158eb0c528a57aac7203476312cb1`，不是当前合并头；延迟工具的计时对照生成token数量不同，不能当普遍加速比或学习收益。TransferQueue集成、DCP恢复和长训练仍未验证。真机条件补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署候选、不改其anchor或AReaL专项。

## 空响应的反向通信与零梯度更新

**没有监督贡献，不等于可以独自跳过通信；梯度为零，也不等于参数不动。** [#598](https://github.com/Tencent-Hunyuan/UniRL/pull/598)（2026-10-23读取open draft）针对部分DP rank收到空AR响应的情况，固定head `fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73`、base/实际祖先 `6c828f5ed81649153315d996f0f597e8808ff144`，ahead1/behind0；6文件42增7删，无提交测试。本节关注反向图与更新语义，不重复[packed样本轴](#packed-文本的样本轴与-token-轴)的布局检查。

### 为什么局部空行不能直接跳过

[#581](https://github.com/Tencent-Hunyuan/UniRL/issues/581)报告原算法遇零token会在replay之前返回，GSPO/PPO也有replay后全mask提前返回。若只有某个rank这样做，其余rank仍进入FSDP参数all-gather/梯度reduce-scatter，调用序列可能失配；各rank都有相同micro数量不代表各自实际执行相同collective。最后一个micro尤其重要：早先反向若延迟同步，最后一次反向正是待累积梯度的同步点。

[训练输入对齐](https://github.com/Tencent-Hunyuan/UniRL/blob/fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73/unirl/train/stack/base.py)在设备搬运、planner和anchor准备前，为零长度TextSegment行插入token 0，同时对已有的七个packed字段补零并重建offset；缺loss_mask时先给原token填1，占位位置保持0。这是原地修改训练segment，不是生成了一个真实completion或增加sample数。placeholder只为进入replay，不能把其长度当有效监督token，也不能借此补齐缺segment/缺logp等不完整输入。eval不传`training=True`，不走该补齐路径。

### 零损失仍须连到对应输出

[GSPO](https://github.com/Tencent-Hunyuan/UniRL/blob/fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73/unirl/algorithms/gspo.py)在无valid sequence时，对`new_logp.sum()*0`执行backward；[PPO](https://github.com/Tencent-Hunyuan/UniRL/blob/fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73/unirl/algorithms/ppo.py)则把`new_logp`和`new_values`都连入零损失。两者返回loss0、有效数量0，但`has_backward=True`。独立创建一个常量0，或只触及policy输出，都不能代替对实际所需输出图的反向。这里也不把乘0视为NaN/Inf清理器；本机控制仅用有限输出。

[单栈](https://github.com/Tencent-Hunyuan/UniRL/blob/fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73/unirl/train/stack/single.py)逐micro设置grad-sync，并检查启用deferred sync时最后micro是否真的backward；只是关掉报错不能补出缺失同步。[全局有效数量](https://github.com/Tencent-Hunyuan/UniRL/blob/fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73/unirl/train/stack/loss_scale.py)仍独立于这次图连接：单栈token-mean全局为0时在算法前拒绝；unified允许AR一轨有效token为0，另一轨继续。GSPO走sample-share，现全空也会零梯度并step，不能泛化为所有全空批都step或都skip。

### 零梯度和没有梯度

PyTorch明确区分`.grad=None`与零张量：前者对应参数不参加这次optimizer处理，后者仍执行该参数的更新规则。[zero_grad文档](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)

本机FP64普通优化器控制均从参数2开始，lr0.1。warm表示先用梯度1更新一次，再比较下一次None与零梯度：

| 优化器与已有状态 | 下一次grad=None | 下一次grad=0 |
| --- | --- | --- |
| AdamW，weight_decay0.1，无历史 | 参数2，无参数step状态 | 参数1.98，参数step1 |
| 同一AdamW，已有一次更新 | 保持约1.880000001，step1 | 约1.794194177，step2，历史moment继续演化 |
| SGD，momentum0.9、无decay，已有一次更新 | 保持1.9 | 1.81，来自历史momentum |

因此全mask回合的“数据损失贡献为0”，并不承诺无参数变化、无优化器状态变化或等价于删掉该次更新。对已累积有效梯度的最后micro，零贡献反向还可承担同步工作，不能用上表的孤立参数实验替代累积窗口验收。

### 本机与作者验证分开

[复查脚本](../../experiments/scripts/verify_unirl_empty_backward.py)从固定Git对象AST提取原AlgorithmStepResult及GSPO/PPO各自全mask的完整if分支，包装后直接给有限CPU张量图，不构造假stage或trainer。base/head×两算法×1/4输出共8组：base没有梯度，head给所触及叶子分配零梯度，PPO含两输出；policy-only反例中独立critic仍为None。另6组真实torch optimizer控制得到上表，Python3.12.13/torch2.9.1，显式CPU。复现命令：

```bash
python -B experiments/scripts/verify_unirl_empty_backward.py /path/to/UniRL-git
```

两版各5个改动Python模块仅compile通过。没有执行padding helper、完整算法/replay、归一化、TrainStack、accumulation、FSDP/Gloo/NCCL或设备任务。head GSPO SHA256 `d53ab94ef99af6390873d171f5387c6ee373adf873f8dd44a5a1dc1c11c1a055`；PPO `9403798c110a961b0d6f2cfa8adc786a4d24e145648658734fb3f4256d5aa003`。

作者外部harness报告两rank CPU复现及2×H100的14项小型FP64线性模型通信/梯度检查，并非本项目复跑、预训练模型或真实recipe验收；完整HI3/BAGEL和多rank CUDA PPO仍未测。原issue已有该draft覆盖，不另开同题修复；实际collective/最后micro与optimizer状态缺口补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不应用候选或干预其专项。

### 纯补齐反向与 ratio 诊断

2026-10-24，[#598](https://github.com/Tencent-Hunyuan/UniRL/pull/598)仍为open draft，新head为`471690694c51f3070e012276429585fcae877a4e`，base为`6fd82cba13f52f394747832344039408699ea378`，6文件48增8删。它不是单纯rebase：父提交`784893d10c01ab21fdf2e33fbd7cec18123bf80f`合并旧head `fa0c306`与main，随后[新增guard修复](https://github.com/Tencent-Hunyuan/UniRL/commit/471690694c51f3070e012276429585fcae877a4e)。上面的旧版图与optimizer控制仍只对应原固定版本，旧复查脚本不改指向。

空GSPO/PPO micro返回`has_backward=True`、有效数量0、`metrics={}`。旧`_guard_first_update_ratio`仅按`has_backward`筛选，便会读取不存在的`ratio_min`而抛`KeyError`；通信需要的反向与有效样本的统计资格不能共用一个布尔值。[新guard](https://github.com/Tencent-Hunyuan/UniRL/blob/471690694c51f3070e012276429585fcae877a4e/unirl/train/stack/single.py)额外要求`num_steps_or_tokens > 0`，并给`max`设置`default=0.0`：混合窗口保留有效micro的最坏偏差，全空窗口的偏差记0。这个0表示没有纳入ratio诊断的有效micro，**不是测得策略与anchor完全一致**。它也没有撤销零贡献backward或更改optimizer step。

Python3.12.13上，AST提取父/新提交的完整原guard方法，给本地dataclass结果及micro属性fixture，分别执行六种输入：

| 输入与开关 | 父提交 | 新提交 |
| --- | --- | --- |
| 全空，guard开启 | `KeyError: ratio_min` | 偏差0，无warning |
| 空与有效混合，有效ratio范围0.8–1.1，tol0.1 | `KeyError: ratio_min` | 偏差约0.2，warning一次 |
| 仅上述有效micro | 偏差约0.2，warning一次 | 保持一致 |
| guard关闭、整体无反向、忽略无反向micro（三项） | 均通过 | 均通过 |

新提交六项均通过，原metrics保持且输入结果未被原地修改；父提交两项预期异常、四项控制通过。仅隔离执行该原方法，logger记录调用，不构造TrainStack或完整GSPO/PPO/FSDP训练；复现代码见[当日日志](../../diary/2026-10-24.md)。父/新完整`single.py` SHA256分别为`aee263c3ee0e8d02ef1f039e655c0632779e2205d9dc0b1a0df5f8a2b5de01f0`、`e8c8578114b958b2cffb6514a9621416bc7cca28abdf2a44ff96fc684c8420c4`。

作者PR正文为此次同步另报packed padding、七算法零贡献、两rank CPU FSDP deferred-window及GSPO/PPO ratio回归通过，且明确未重跑GPU；此前H100结果仍绑定旧提交。已有作者修复，不另起同题补丁或增加硬件交接任务。

## 冻结教师路由与调用方状态恢复

**切回default不等于恢复调用方，冻结参数也不等于关闭梯度记录。** [#519](https://github.com/Tencent-Hunyuan/UniRL/pull/519)（2026-10-23读取open）收窄为冻结教师adapter的状态保护，不声称修复main已有的普通DiffusionOPD路由。固定head `fd1ca404804dc2cc0b395bd0aa5765c012104d9a`、base及实际祖先 `ff1d3a6b2a8070ba31d5e74ac02c0e965762569c`，ahead5/behind0；2文件35增12删，无提交测试。这里不是行为策略anchor生命周期、教师checkpoint或完整D2集成的实现。

### 路由和梯度开关是两类状态

[PEFT 0.20的set_adapter](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py)遍历adapter banks，除选择活动adapter外，还按活动名称及inference_mode设置trainability，默认会使活动adapter可训练。因此“切到已冻结的教师”这一步本身可能改梯度开关，调用上下文必须管理副作用。

[UniRL adapter_active](https://github.com/Tencent-Hunyuan/UniRL/blob/fd1ca404804dc2cc0b395bd0aa5765c012104d9a/unirl/train/lora.py)现在先拒绝default、不存在的教师及名称匹配参数中仍可训练的教师，而不是替调用方静默冻结。然后保存每个LoraLayer的完整`active_adapters`列表，以及全部模型参数原来的`requires_grad`。切换成功后立即恢复原梯度开关，在保持教师路由的同时不改变调用方的可训练集合；退出时先恢复逐层选择，再恢复梯度开关，避免恢复路由再次覆盖开关。旧实现退出后一律激活default并强制其可训练，无法保留逐层不同选择或调用方的额外冻结。

| 状态 | 负责的行为 | 此上下文的范围 |
| --- | --- | --- |
| active_adapters | 本次forward使用哪个adapter | 保存并恢复逐层列表 |
| requires_grad | 参数是否参与求导 | 保存并恢复所有原参数的开关 |
| no_grad | 当前执行是否记录反向图 | 调用方负责，不由切换自动提供 |
| model.training | dropout等模块的训练模式 | 不在此上下文管理范围内 |

### 切换失败与恢复失败要分开

切换动作移入try内部，所以某层切换抛错、前面若干层已改变时，也会进入finally尝试恢复；正文replay抛错同样如此。恢复路由另包一层finally，即使某次恢复`set_adapter`抛错，仍会尝试还原参数梯度开关。**这不是无条件的原子回滚**：恢复某层本身失败时，逐层循环可能中断，不能承诺其后各层选择均已恢复。这是按源码控制流界定范围，不是本机故障注入结果或新增修复候选。

快照保存的是已有参数对象和布尔开关，不保存参数值、optimizer状态、RNG或外部资源，也没有为并发调用加锁。移除旧`trainable=`关键字是调用接口变化；不能将本次收窄解释成支持任意交错activation-checkpoint重算。重算发生时使用的路由状态，不能仅靠早先退出时恢复正确来认证。

### 调用方与adapter banks

[DiffusionOPD.prepare_part](https://github.com/Tencent-Hunyuan/UniRL/blob/fd1ca404804dc2cc0b395bd0aa5765c012104d9a/unirl/algorithms/diffusionopd.py)先按单一domain选择teacher，在`torch.no_grad()`与`adapter_active`内replay；退出上下文后再把teacher transition means detach到CPU。no_grad和路由保护分别承担不同职责，普通路由正确不代表所有异常退出都恢复原状态。

adapter枚举与梯度设置改为遍历`adapter_layer_names`，不再只看lora_A/B；对bank元素调用`requires_grad_`，可覆盖Module与Parameter两种存储。[PEFT LoraLayer](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/layer.py)的embedding banks是ParameterDict，不能假定元素都有`.weight`。这不等于embedding-only教师checkpoint加载已打通：同版frozen loader的covered/权重映射仍以lora_A/B为范围，DoRA等也另有拒绝规则。

### 源码身份与验证范围

本机现有CPU环境缺PEFT、Transformers和Diffusers，未安装依赖或构造adapter替身。仅完整差异/方法阅读、base/head各一份lora.py语法检查，以及作者publication `5390b5afd3041b59f691f9a07b367f37eb24b0b0`与当前head的lora.py/diffusionopd.py语法和AST比较，均无模块导入或运行。

publication与head的六个切换/枚举/名称解析helper及`DiffusionOPD.prepare_part` AST相同；但两份完整文件的blob不同，后续提交还调整了构造器与frozen injection等主线代码。作者报告的真实PEFT CPU异常路径与质量检查绑定到publication源码身份，不作当前整个head的重新执行，更不作GPU/训练/同步或恢复验收。网页缓存旧正文不覆盖本轮API的收窄描述。

SHA256：head lora.py `5233713e1ec325f3218e592a3f6bca6eb89669d3ac126289194b4067ea1801d9`，base lora.py `ebfa9c52c54a29dcb3cda2fd88cba6427d80fe1c94ffdf8a189a8d8d7f9b16fc`；head diffusionopd.py `12ea1270d375cc3180eacb0a4d8a4585ed1009f5bb37810bd87a509640d68469`。真实PEFT/模型缺口补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署候选或改其anchor/AReaL专项。

## 离线回放的到达契约与结果隔离

**记录下来的提交时间，与“前序请求完成后才可提交”是两种不同输入；格式合法也不等于捕获完整或性能预测准确。** [#487](https://github.com/Tencent-Hunyuan/UniRL/pull/487)（2026-10-23读取open）仅新增experimental离线schema/converter，固定head `e8779559d2ceb441c062f603f90e7fd31370e69b`，目标base `b5079be170635afd87acc83873a18d2fb2114732`，实际祖先 `f8d95462837df00188dab1376e19347b43996307`，ahead1/behind15；4文件567行新增，无提交测试或runner。这里的replay是工作负载回放，不是训练前向重算log-prob。

### 到达时间与前序依赖

[原转换器](https://github.com/Tencent-Hunyuan/UniRL/blob/e8779559d2ceb441c062f603f90e7fd31370e69b/experimental/aisim/convert.py)的open-loop保留记录的arrival，统一减去最小offset并在sidecar记录origin；不添加session、wait_for或推算工具等待。本机输入到达`[100,100,130]ms`得到`[0,0,30]ms`，相同到达按原输入顺序，不按request_id字典序重新排序。

dependency路径则输出前序ID加明确的external delay，首轮没有前序时用窗口release offset。外部等待不能由相邻submit差反推：那个差还可能含前序服务与排队时间；如果换了后端，前序完成时刻本就应重新变化。未知delay必须报错，真正0等待必须显式写0。本机首轮release7ms、次轮wait_for前序且delay0，分别输出arrival7与tool_wait0，次轮不同时带observed arrival。

group/trajectory ID不自动代表collector barrier或trajectory slot。dependency sidecar明确将两者标为`not_modelled`；仅有依赖边不能推导训练采样窗口的准入、排空或组尾完成行为。

### 来源与测量结果分开

[原schema](https://github.com/Tencent-Hunyuan/UniRL/blob/e8779559d2ceb441c062f603f90e7fd31370e69b/experimental/aisim/schema.py)分manifest、workload与MeasuredResult，converter不接收MeasuredResult参数。workload包含给定input/output token长度：这是固定负载条件，不是在预测生成文本或输出长度。测量的完成/首token/服务耗时只供报告；用它们作预测输入会把待验证答案提前送给预测器。

`assert_no_outcome_leakage`检查转换后每行顶层已知结果键，`result_sidecar`标记report_only；这不是任意嵌套数据的通用信息流证明。测量校验要求每个request恰好一条结果，拒绝缺失、重复和未知ID；ID覆盖不证明时钟边界、真实捕获或参数内容可信。manifest中版本/硬件/profile等来源字段也仍需由捕获过程核实，非填非空字符串即可认证。

### 上游head与作者后续guard

通用`Manifest.validate`允许不同replay_contract和complete/partial/incomplete状态，必须由具体转换入口再检查用途。上游to_open_loop已有用途与incomplete保护，to_dependency缺这两项。作者已在[现有fork draft #1](https://github.com/0z5a/UniRL/pull/1)修正，固定 `4ad73720989c8b7c9229da4ecabd9ee654e67dc4`直接接在上游head后，仅9行新增；不重复实现或另开同题贡献。

| 输入 | 上游e8779559本机结果 | 作者fork 4ad73720本机结果 |
| --- | --- | --- |
| observed manifest配dependency请求 | 错误接受 | 拒绝 |
| incomplete dependency capture | 错误接受 | 拒绝 |
| complete dependency capture | 正常转换 | 正常转换，输出JSON与上游相同 |

不能把fork拒绝行为写成当前上游已有，也不能把partial与incomplete合并成一个状态。schema及guard只拒绝不支持的窗口，不负责重试执行或故障恢复；不应删除失败请求后把剩余部分改称完整捕获。

### 原模块CPU验证与性能边界

[复查脚本](../../experiments/scripts/verify_unirl_trace_contract.py)在Python3.9.6运行两版完整原schema/convert模块，只为experimental父包提供命名空间，无函数摘取、假schema或模拟器。用明确标注synthetic的请求验证上述三组对照、open-loop原点/同刻顺序、混用拒绝、显式零等待与缺失等待、未知前序/环、JSONL往返、测量ID覆盖及已知结果键泄漏拒绝。两次结果一致，未导入torch/Ray/vLLM/SGLang/AISimulate或安装依赖。

```bash
python3 -B experiments/scripts/verify_unirl_trace_contract.py /path/to/UniRL-git
```

两版schema SHA256均为`11fb135496a1ad4b046a691d8c9811bd145101bbf89b83bac089d7fba449406c`；上游convert为`1e2a2f8513eb7337f13f76d20588b727667d026248bdeb10372584d3b15b3704`，fork为`32889b6d94a452e903e91c0dd0f1d9b8fd6e5b3799efad62b3689b3f81c5da26`。

这是原数据契约执行，不是原作者全部外部测试、捕获hook、runner或模拟时序验收。最新API另指向作者独立fork runner与CPU profile smoke，不能移作本机或上游head结果；真实匹配GPU执行与模拟的保真度仍未测。本题不产生加速比、H100/H20/4090准确性或训练结论。真实capture/profile/版本匹配条件补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不安装模拟器、租算力、启训练或改其专项。

## 常驻与流式 block 的规划账本

**可行的卸载计划不是已实现的执行器，内存账本恒等也不是预测精度。** [#486](https://github.com/Tencent-Hunyuan/UniRL/pull/486)（2026-10-23读取open）新增纯stdlib placement planner和测量契约，固定head `57cec429b5978d7da4bfd5f751764c0fd5da4c06`、目标base `b5079be170635afd87acc83873a18d2fb2114732`，实际祖先 `f8d95462837df00188dab1376e19347b43996307`，ahead1/behind15；2文件586行新增，无提交测试。它不安装offload策略、不移动参数，也不读取checkpoint。

### 排他账本与执行器能力

[原规划器](https://github.com/Tencent-Hunyuan/UniRL/blob/57cec429b5978d7da4bfd5f751764c0fd5da4c06/unirl/train/backend/fsdp/offload_plan.py)对每个phase/rank计算：排除所有block参数后的phase高水位，加root/fixed、该phase的常驻block、streamed staging与headroom。每个block的staging为shard+full+cast三种存储角色之和。[测量契约](https://github.com/Tencent-Hunyuan/UniRL/blob/57cec429b5978d7da4bfd5f751764c0fd5da4c06/unirl/train/backend/fsdp/README.md)要求compute时间排除transfer等待；否则先将参数或等待计入profile、再由planner加一次，会重复计费。数值类型通过不证明采集者真的做到了排除。

`live_streamed_bound=None`时，所有streamed block都计为同时live。只有执行器已验证同时物化上限，才能按最大的若干block算staging；不能因希望双缓冲，就把原生FSDP的生命周期替换成“两块在场”。本机合成block staging为`[30,60,90,120]B`，root7、phase基线13、headroom5，五种resident前缀k得到：

| k | 保守None上限 | 声明live上限2 |
| --- | ---: | ---: |
| 0 | 325 | 235 |
| 1 | 325 | 265 |
| 2 | 325 | 325 |
| 3 | 325 | 325 |
| 4 | 325 | 325 |

保守模型里resident与streamed只是同一笔总存储改标签，单靠GPU预算不能挑出节省显存的部分后缀。右列是声明能力条件下的模型结果，不是真实两槽执行器或显存测量；容量、stall限制和host预算可以互相冲突。

### 共同计划与目标函数

枚举的是所有rank共用的调用顺序前缀resident、后缀streamed，不是任意子集的全局最优放置。每rank预算分别约束，同节点host则将各rank基线和offloaded shard需求相加；double_buffer再使host shard项翻倍。fixed保持常驻，alias跨切分边界会拒绝。没有候选满足时返回INFEASIBLE及原因，不自动回退全常驻。

选择目标是`max_offloaded_suffix`：按k从小到大取第一个可行项，**不是选最快可行项**。本机每phase四块各compute10ms、transfer3ms的合成例中，k0有3ms暴露等待，k1可隐藏等待；250B预算下上表右列只有k0可行，仍选k0。`latency_optimal_k=1`来自包括不可行项的另一次最小值搜索，不能绕过预算直接采用。预算270B再要求stall0，才选k1部分后缀。

### 按计算式解读预测字段

串行copy/compute双时间线在transfer完成后才允许该块计算，不把transfer开始当ready。profile应提供互不重计的compute、transfer、startup和collective成本；当前纯模型不认证真实copy lane、通信竞争或生命周期。

源码的`predicted_total_ms`是rank×phase预测值的最大值，不是四phase相加的完整训练步时间。`transfer_busy_ms`当前取`compute_free-startup_ms`，也不是复制时长之和：上述合成k0中它为43ms，而四次copy时长之和仅12ms，`no_compute_transfer_ms`为3ms。字段名不能直接换算DMA利用率、PCIe带宽或端到端加速比。

rollout、replay_backward、optimizer、transition中完全没有测量的phase会使所有候选拒绝；不能从另一个phase外推transition。本版missing检查以各rank phase的并集为依据，采集者仍需保证逐rank、逐block成本完整及排他口径，接受的profile不自动等于完整实测。

### 原模块CPU控制与作者证据

[复查脚本](../../experiments/scripts/verify_unirl_offload_plan.py)执行固定版完整原模块，在Python3.9.6用36组明确标注synthetic的profile/capability配置，对180个候选的GPU/host账本、stall、max-phase时间、可行性与选择做独立算术对照，全部通过。另核node-sum/double-buffer、紧rank预算、缺transition全拒、fixed/alias及确定性JSON；两次结果一致。没有torch/Ray/vLLM/SGLang导入、模型或设备张量分配、设备执行及新依赖。

```bash
python3 -B experiments/scripts/verify_unirl_offload_plan.py /path/to/UniRL-git
```

原模块SHA256 `d695e765f614d22a4e294e7df99d96b66a0a85f45692bc3abe2fcd7765370902`。这是planner算术和拒绝路径验证，不是profile校准、FSDP数值、真实驻留上限、显存回收或性能验收。

作者另报2×H100小型残差MLP的FSDP能力对照和随机权重SD3组件测量；前者非完整RL recipe，后者transition未测、profile被拒。all-resident账本delta0仅是构造恒等，不能作预测准确性；planner对执行器误差、稳态性能仍未验证。真机资源/profile/执行器前提补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署planner、不启offload/训练或改其专项。

## Prompt 组边界与前向分块

**完整组上计算优势，不要求所有siblings挤在同一个forward；共享初始噪声，也不保证改chunk大小后轨迹完全相同。** [#485](https://github.com/Tencent-Hunyuan/UniRL/pull/485)（2026-10-23读取open）将原whole-group shape bucketing收窄为默认关闭的`validate_group_boundaries`，固定head `9e36cf99eea03e9e2f6035976ac97c1a70759f9e`、base/实际祖先 `772fd7c50ec8221a032179aeec8320df935b3abf`，ahead5/behind0；8文件130增10删，无提交测试。它只检查既有固定分块，不改变FBS、行序或生成调度。

### 允许整组或对齐的等大子组

[原helper](https://github.com/Tencent-Hunyuan/UniRL/blob/9e36cf99eea03e9e2f6035976ac97c1a70759f9e/unirl/rollout/engine/trainside/plan.py)先要求同一个group ID只出现一个连续run。组不大于FBS时须完全落在一个chunk内；更大的组须从chunk边界开始，组长为FBS的整数倍。校验接受后返回None，不重新分桶或放大内存容量。

| 布局 | FBS | 原helper结果 |
| --- | ---: | --- |
| 单组16行 | 4、8或1 | 允许跨多个完整chunk |
| 单组16行 | 6 | 拒绝不等大尾子组 |
| A2行后B3行 | 4 | B跨边界，拒绝 |
| A、B、A | 64 | group重现，helper拒绝 |
| A2、B2、C4 | 4 | 接受，不要求不同组等长 |

这不是预期fan-out数、组内容身份、异构geometry或优势公式的校验。FBS按行限制每次调用，不能当显存字节上限；为满足“一组一个forward”而强制放大FBS会改变原本用小批控制峰值的调度。

### 实际引擎何时调用校验

[TrainsideRolloutEngine.generate](https://github.com/Tencent-Hunyuan/UniRL/blob/9e36cf99eea03e9e2f6035976ac97c1a70759f9e/unirl/rollout/engine/trainside/engine.py)只在`bs > FBS`且开关为真时调用helper。FBS未设或一批已放得下时直接单次generate，因此表中A/B/A、FBS64的helper拒绝不能写成该引擎路径实际拒绝。开关不是全路径的组身份保护。

实际分块仍按原始行连续slice，每次与同一批input Parts组成Sample，最后按顺序concat frontier再供下游reward/组优势使用。校验位于锁内、eval/no_grad的try范围内，在首个pipeline.generate之前；普通校验异常会经过finally恢复原训练模式，但此前sigma等准备不等于全部输入状态回滚。本机没有运行这条引擎异常路径。

只要设置FBS，引擎就另行拒绝含多个gen Part的Sample，避免每个chunk重跑内部stage却只保存frontier；这个保护独立于本次是否实际切成多个chunk。故新开关不是对组合多stage采样或异构调度的扩展。

### 噪声身份与轨迹复现

[NoiseRecipe.from_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/9e36cf99eea03e9e2f6035976ac97c1a70759f9e/unirl/types/noise_recipe.py)在driver recipe启用时优先使用显式noise keys，否则在init_same_noise下选group IDs，普通路径选sample IDs；已有initial_latents可直接返回，需要再生且有shape/key时才按seed与身份构造x_T。[共享噪声原语](https://github.com/Tencent-Hunyuan/UniRL/blob/9e36cf99eea03e9e2f6035976ac97c1a70759f9e/unirl/sde/noise.py)使用每组派生seed，不把chunk号作为种子。这里是x_T身份契约，不能扩展为每一步随机数分配也相同。

同版[引擎README](https://github.com/Tencent-Hunyuan/UniRL/blob/9e36cf99eea03e9e2f6035976ac97c1a70759f9e/unirl/rollout/engine/trainside/README.md)区分SD3全局SDE RNG与采用逐sample/step generator的管线。改变FBS可能重排行与去噪步消费全局随机序列的先后；低精度批形状也可能改变末位数值。禁用driver x_T且不共享噪声时，初始噪声自身也可能依赖分块。因此“固定FBS下开关校验前后parity”与“更换FBS后样本完全一致”不是同一个命题。

### 有界原函数对照

[复查脚本](../../experiments/scripts/verify_unirl_group_chunks.py)执行完整原plan.py，用独立的“列出实际chunk再计各组占用”oracle，对0～7行、3种ID、FBS1～5的16,400种布局逐一比较：953接受，其余拒绝，全部一致。另核6个大组几何及小组straddle/交错/mixed-group控制，接受/拒绝均不修改输入。两次执行一致，Python3.9.6，未导入torch/Ray/vLLM/SGLang；两版engine及head NoiseRecipe仅compile，无Sample/Part、RNG、模型、引擎parity或设备执行。

```bash
python3 -B experiments/scripts/verify_unirl_group_chunks.py /path/to/UniRL-git
```

head helper SHA256 `2bc079e42da5e4949754f68079c2645fa4a0ab95054ac5b3656aca831ca4fee0`；engine `0a8b9d7df2ad8378c1dd1bfd13f28da256c7123dcff1bb213f9b83a0361a250c`。此有限枚举不证明任意输入/引擎或模型行为。作者当前报告的77项CPU引擎parity对应merge fc50ac2，后续改错误消息另报随机布局对照；H100测试属旧683c320，未在当前head复跑。本项目不将这些结果移作实测，也不推训练/显存/吞吐收益。匹配配方与真实模型缺口补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不改变FBS、启用开关或部署候选。

## 多教师周期与恢复身份

多教师OPD不是把一个batch交给所有教师后平均logits。新草稿[#624](https://github.com/Tencent-Hunyuan/UniRL/pull/624)让每次rollout只含一个域、路由到该域的冻结LoRA教师，再把完整域周期交给现有TrainStack。2026-10-23读取open/draft、head `375c2273ed5b0ef89d3075350ba96ba2bcdb31eb`，base/祖先 `6fd82cba13f52f394747832344039408699ea378`。GitHub总diff18文件693增32删，包含#534前置；作者指定的A3独立边界是 `0142d612fbd824c9a47d3f6446a4b914a04eb1c3`，其后仅15文件316增31删。不能把前置AROPD目标重算成多教师新增实现。

### 一个周期才是更新边界

[ARTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/375c2273ed5b0ef89d3075350ba96ba2bcdb31eb/unirl/trainer/ar.py)要求`accumulate_rollouts`等于域数；仅同步trainside、每batch一次update、不支持TransferQueue。数据源按声明顺序round-robin，每批只带一个域。trainer先完成周期中所有rollout，再传tuple给TrainStack；每个Part在micro replay前按domain选择教师，混域、漏域和重复域不能凑成合法周期。

现有窗口训练先zero_grad，中间Part不step、最后Part才走正常有梯度时的optimizer step，各域梯度权重为`1/域数`。所以目标是**等域加权的域内token均值**，不是所有域有效token拼成一个全局均值；也不是每个教师训练一次学生。此处只核接线，不复跑归约、anchor生命周期或M5的#523方案。

两域时，rollout 0/1组成第一个周期，2/3组成第二个；`num_rollouts=100`是100个rollout，即50个周期，而非100次更新。日志和checkpoint使用周期末的rollout号；`checkpoint-20`对应完成20个rollout。总rollout数、正save/eval间隔须能被周期长度整除，恢复起点也不能落在半周期。`train_step`拒绝不完整窗口，不提供从中间一个域接着累加未保存梯度的恢复模式。

### 路由记录与教师内容校验分层

`ar_teacher_cycle.json`记录域到adapter名称的映射、有序域列表、data seed、rollout seed、batch size和设备数。恢复先读取并逐项比较，缺记录直接失败，不匹配在backend.load前拒绝；随后再核恢复rollout位置是否对齐周期。

这份记录不是教师权重文件的内容哈希，也不是完整配方签名。[FrozenAdapters](https://github.com/Tencent-Hunyuan/UniRL/blob/375c2273ed5b0ef89d3075350ba96ba2bcdb31eb/unirl/train/lora.py)和backend承担另一层内容校验：正常checkpoint中的frozen adapter哈希与当前内容不同时拒绝；旧元数据缺失或空记录在该helper中不加约束。不能将周期JSON里的同名映射直接当成教师内容相同，也不能由这份JSON推断数据文件内容已被hash锁定。

周期记录在模型checkpoint调用之后以临时文件加replace发布。这是单文件替换，不把模型、优化器、trainer_state和周期JSON合成一个崩溃原子事务；缺周期记录时的恢复拒绝与整套checkpoint的持久化验收要分开。

### 快进数据与重建生成随机数

恢复重新按data seed构建各域DataLoader，然后按已完成rollout数调用get_samples快进；不重跑旧模型生成。新的trainside seed为`base_seed + rollout_id × dp_size + dp_rank`，在fork_rng范围内设置Torch CPU/当前CUDA生成器，并在退出时恢复调用方Torch RNG。这个键依赖rollout位置和DP布局，不是按sample_id跨任意切分保持不变的逐样本seed，不能与#507的Omni请求seed契约混用。

Qwen3禁用cuDNN SDPA的时机从第一次replay前移到stage构造，避免首次生成与恢复生成使用不同设置。这里是源码改动，不是本机对CUDA确定性或NaN的验证。数据游标相同、生成RNG相同、参数/Adam状态和完整训练轨迹相同是不同保证；作者也要求匹配数据、确定性kernel和零dropout。

### 作者报告与本机范围

作者报告固定于同一head：2×A100 80GB、Qwen3-4B-Base及两个合成SFT教师；4个连续update与2+restore+2在token/logp、学生参数及Adam状态一致，raw/normalized gap各跑32次update，但normalized-gap恢复未另验。报告是训练内合成对齐检查，不是held-out质量或生产benchmark；人类完整diff审阅checkbox仍未勾选，不在此判定候选ready。

本机仅对A3边界前10个、head11个相关完整Python模块做AST/compile，无import或执行；没有复跑作者36项CPU行为、12项startup或GPU harness（均未提交）。不增加任务周期/RNG/训练栈替身去填补这些缺口。head trainer SHA256 `08627af30b621de72c7a148a730fbb8af4b33fe292b6ba018f3547130dbf5cb9`、engine `393d316c7c68340f94e780813de526d12b35fac4b6a00ff92bfd60be4d948f68`、多教师算法 `e707b39215ad6b5cd487911b745378939c3aaed0dc9406d4db915ddb58850e6a`。

真实周期/恢复条件仅记入[原M5可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动应用draft、下载教师/模型或排训练。此为源码契约学习，已有作者覆盖，不新增竞争PR或改动M5专项。

## 样本观测的行上限与数据搬运

样本观测需要同时限制输出、保留谱系、避免干扰训练，三个要求不能由一个`max_items`替代。UniRL draft [#597](https://github.com/Tencent-Hunyuan/UniRL/pull/597)以共享SampleRecord接通W&B和独立本地输出，已有作者ruiling-smartbear负责；这里核验其数据契约，不另做竞争实现。2026-10-23 API读取open/draft，固定head `d507c4dc1e10db427cadff4d6e29482801a12065`、目标main快照及实际祖先 `6c828f5ed81649153315d996f0f597e8808ff144`（ahead3/behind0），16文件400增235删；不是当前已合入能力。

### 先选生成行 再找祖先

[build_sample_records](https://github.com/Tencent-Hunyuan/UniRL/blob/d507c4dc1e10db427cadff4d6e29482801a12065/unirl/types/sample_record.py)先取当前生成Part的前K个sample_id，再用原祖先Parts与这个小frontier构造Sample，通过conditioning_at按parent ID回溯。它不是用当前行号整除fanout来猜prompt或recaption。CPU对照将root设为A/B，recaption行排成B1/A0/B0/A1，输出排成A1/B0/A0/B1；前两条仍得到`prompt A + recap A1 + reward1`、`prompt B + recap B0 + reward10`。

BaseTrainer对**每个生成Part**应用同一个上限，两个Part各限2条会形成4条记录，而非整轮共2条。固定前缀也不是随机抽样或全体质量估计；eval只观察每套评测首个scored chunk，奖励指标仍另按整套汇总。root_id、group_id和sample_id保留各自层级，不能把多个生成阶段的记录当同一批独立样本。

原Sample构造器会先拒绝缺失parent的链；本机非法输入也在此失败。记录构建器另有malformed-lineage警告回退，但该保护不代表正常构造器允许坏链，不绕过构造器制造一个“框架接受”的反例。

### 条数限制不等于字节限制

[媒体预览](https://github.com/Tencent-Hunyuan/UniRL/blob/d507c4dc1e10db427cadff4d6e29482801a12065/unirl/types/media_preview.py)从选中行扩展到TensorRef块边界后才hydrate；一个块覆盖多行时会额外实体化未展示行，块端点不能映射到样本累计长度时保守回退整个Part。这是源码路径，不是本机远程流量测量。resident CPU音频只验证前2条预览及原训练primitive身份/值不变，不外推driver RSS或通信峰值。

上限按行而非字符、时长或字节计算，本地文本与metadata原样进入JSON；小K不能自动保证短文本、小视频或敏感内容已脱敏。rollout观测完成或跳过后，finally清空生成Part的decoded primitives、primitive_metadata和preview；这里不清理训练segment/reward，也没有调用transport释放接口。评测phase不走该清理分支。

### 双输出与尽力而为的持久化

`logging.sample_dump_dir`可在W&B关闭时独立工作；训练从step1开始按media_log_interval取样，eval由eval_interval节奏控制。正常路径中逐Part记录构建、W&B记录输出、本地dump各有异常捕获；这不是任意trainer错误或进程故障的恢复协议。旧非空顶层dump_dir及ROLLOUT_DUMP_DIR/ROLLOUT_DUMP_N会在创建worker前报迁移提示，不是悄悄忽略。

[本地sink](https://github.com/Tencent-Hunyuan/UniRL/blob/d507c4dc1e10db427cadff4d6e29482801a12065/unirl/utils/sample_dump.py)先逐个保存媒体，最后写`<phase>/step_<n:06d>/samples.jsonl`。真实CPU对照确认：WAV为8kHz/单声道/16bit PCM，3与4个采样点可逐字节读回；同step重观测会替换索引，却不清理上次多出的WAV；metadata不能JSON序列化时函数警告返回，但已写WAV可留下而没有索引。这符合best-effort定位，不作为训练checkpoint或原子归档保证，也不据此另提事务化修复。

### CPU复查与真实配方边界

[复查脚本](../../experiments/scripts/verify_unirl_sample_records.py)从固定Git对象加载完整原模块，仅替换tensor包汇总初始化、重新导出原Batch/ref符号以跳过Ray transport；没有方法摘取或假Sample/media类。Python3.12.13、torch2.9.1、Pillow12.0.0（复用旧临时target，无新安装）下，6组控制两次通过：乱序谱系/每Part上限、四种cap、原构造器拒坏链、resident音频、JSONL/WAV读回与重写、路径/JSON失败返回。11个改动Python模块compile通过。复现：`python -B experiments/scripts/verify_unirl_sample_records.py /path/to/UniRL-git /path/to/Pillow-target`。

这些是原CPU容器与本地文件控制，不运行BaseTrainer、TensorRef搬运、W&B/PyAV或模型训练，也没有作者未提交的11+5组临时harness。本机首次坏链夹具在Sample构造时就被拒绝，改为验证原拒绝行为后通过；没有关闭校验或修改候选来“跑通”。

作者另报告H100上的SD3.5/Qwen3/PE三次短训与媒体契约；训练属于`b784cb8`，最终head仅在pyproject/requirements加入soundfile依赖及README说明（Git核为3文件4增1删），最终媒体契约单独报告。原Omni/HI3配方及多节点driver RSS仍未验证；小模型或合成CUDA媒体不等同于这些结果。需要的真实资源/最小观测已补[原M5可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不排新必跑任务或自动部署draft。

原模块SHA256：sample_record `a52d819e057fca021025e9b3ed140ad6cd166026d701c02d6aacb538e0ba21b0`、media_preview `2bd1aba6fbcac008094ca15e51c9782b71adef3e152d8e0fb5be55e3c5240725`、sample_dump `52e9bfce9c976dfd02603a2fafa545213331f930fa2ef799c5ff9f908bfe705a`；脚本打印hash并禁止隐式Git fetch。

### 样本观察拆分与音频通道

2026-10-24作者将[#591](https://github.com/Tencent-Hunyuan/UniRL/issues/591)的已有实现拆成draft [#628](https://github.com/Tencent-Hunyuan/UniRL/pull/628)基础层与draft [#597](https://github.com/Tencent-Hunyuan/UniRL/pull/597)接线层，均OPEN，仍由ruiling-smartbear负责。固定基础head`94304dfabd9c2f1419e3d672aca73fc842b62126`，接线head`e5961d7edd375b4ec3acda1766cf5134b49625da`，两者API目标base及与其共同祖先均为`6fd82cba13f52f394747832344039408699ea378`。#628只改依赖、types文档/preview/record和两个sink文件，没有trainer调用点改动；#597相对该基础的文件diff为11文件141增193删，不是第二套竞争实现，也不表示两项已合并。

三个基础文件在两head字节相同，`log_sample_records`方法AST也相同。相对原`d507c4d`，media_preview和sample_dump仍逐字节相同，sample_record只有docstring指向变化；但logger及trainer接线有新变更，所以旧六组CPU控制与作者旧GPU结果不能认证新版整体。当前#597正文还将新CPU适配、原Mac合成边界、旧H100短训分别列出；原Omni/HI3模型与多节点RSS缺口仍在。

**独立音频传入SDK的数组布局，与原生音频容器、WAV交错PCM和视频复用，是不同边界。** 新[W&B音频入口](https://github.com/Tencent-Hunyuan/UniRL/blob/94304dfabd9c2f1419e3d672aca73fc842b62126/unirl/utils/wandb_logger.py)在2维且首轴为1或2时先转置，再`.numpy()`；旧代码直接`.numpy()`。[本地WAV sink](https://github.com/Tencent-Hunyuan/UniRL/blob/94304dfabd9c2f1419e3d672aca73fc842b62126/unirl/utils/sample_dump.py)原本就把channel-first输入转为`[L,C]`，写16bit PCM。不能将SDK接受一个数组当成其时长、通道或声音内容正确。

本机Python3.12.13/torch2.14.1，以6个采样点、8kHz和左右不同的数值核对三种布局。只摘取原MediaItem/SampleRecord dataclass定义、原sink函数完整体，以及两版`wandb.Audio`的第一个参数表达式；不导入训练框架，也没有假W&B对象。原函数实际写WAV，再用标准库wave逐值读回：

| 输入张量 | 旧W&B输入数组 | 新W&B输入数组 | 本地WAV读回 |
| --- | --- | --- | --- |
| `[6]` | `[6]` | `[6]` | 6帧×1声道 |
| `[1,6]` | `[1,6]` | `[6,1]` | 6帧×1声道 |
| `[2,6]` | `[2,6]` | `[6,2]` | 6帧×2声道 |

三项的新数组逐值符合预期，PCM通道顺序与转换后数值一致，输入张量未变。这里只观察传给SDK之前的数组，未运行`wandb.Audio`、上传、音频听感或PyAV复用；本机W&B/soundfile/PyAV均未安装。视频路径另有mono扩为stereo的normalize helper，不能将本次独立音频结果当该路径验收，也不扩大到任意通道数或短轴歧义。

[脚本、hash与命令](../../diary/2026-10-24.md)留日志，不重跑旧谱系/cap/失败矩阵，不新增永久harness或音频修复。原[M5样本观察参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)仅更新两head与证据范围，作者新CPU报告及旧GPU短训仍各绑原版本；不因新编号重复排Omni/HI3训练或改专项。

## Tokenizer 实例修补与批量生成边界

接口返回足够的行，与这些行能正确进入真实生成，是不同保证。UniRL [#520](https://github.com/Tencent-Hunyuan/UniRL/issues/520)报告HI3 Instruct checkpoint的batchify只编码首个样本；该项已指派ruiling-smartbear，已有非draft修复[#544](https://github.com/Tencent-Hunyuan/UniRL/pull/544)，不作为我方竞争候选。2026-10-23固定open head `d41122e97e1ab631328c86baad00250c406d2b17`、目标base及实际祖先 `4fb0b1f7b291f3cd5010597b85d41ed44659ef40`（ahead6/behind0），3文件48增8删。

### 在批量入口修补而非重写拼批

2026-10-30复核#544新head `23ea7a811159d3cbf0c844e143ef63aa00943d07`、base5d1da78d；两个未截断Git树核compat.py／text_embed.py分别仍为blob `ccd82bc2c563decb5e4026e9f639a2d4d347b32a`／`9150e06825f9d4fc5e99481263aa3129e9362acc`，同上述旧d411输入。只更新生产代码身份，不重新执行旧AST控制、作者20例或迁移早期GPU报告；README状态说明与真实生成仍须分开理解。

报告指出，checkpoint将一个`prompt_list`元素与多份逐样本kwargs交给`batch_gen_infer`，后者zip后只处理首份。已有#543先加入`len(prompts) × cfg_factor`的行数保护，避免悄悄继续；#544则为每份message list提供一个prompt占位，仍委托原`batch_gen_infer`负责批量padding、CFG与输出组合，不另写逐行编码后拼接的替代器。

[repair_hi3_tokenizer_batchify](https://github.com/Tencent-Hunyuan/UniRL/blob/d41122e97e1ab631328c86baad00250c406d2b17/unirl/models/hunyuan_image3/compat.py)捕获实例上原bound method，通过`signature.bind`及defaults解释调用参数；非batchify分支原样调用原方法，batchify分支剥离外层message_list/batchify/cfg_factor/uncond_p后构造逐样本kwargs，要求messages非空且各元素为list。传给批量helper的`infer_fn`是捕获的原方法，不是新wrapper，避免递归进入同一修补分支。

wrapper赋给`tokenizer.apply_general_template`，没有写回tokenizer类；marker避免同实例重复包装。[调用点](https://github.com/Tencent-Hunyuan/UniRL/blob/d41122e97e1ab631328c86baad00250c406d2b17/unirl/models/hunyuan_image3/text_embed.py)位于HI3 stage实际编码之前、已有backend修复之后，不是全框架替换所有tokenizer。**但结构筛选不是版本认证**：helper只检查两个方法可调用、是否已有marker及四个参数名，不检查类名、源码hash或checkpoint revision。README记录的`2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`是已知缺陷版本，不是函数运行时的硬白名单；升级checkpoint时仍需重核并决定移除。

### 行数保护与真实生成

保留的guard只核`output.tokens.shape[0] == len(batch_prompt) × cfg_factor`，并不逐字段认证token内容、顺序、`real_pos`、sections、mask或KV写入位置。标准gen-image stage的cfg布尔值只产生1/2因子，作者底层tokenizer测试另覆盖因子3；两者不能混作相同入口的支持矩阵。

作者报告使用指定checkpoint的真实tokenizer做20个CPU例，对照上游一行修正的全部输出字段，覆盖文本、图像条件、图像生成、B1/3和CFG1/2/3；临时harness未入库。这是作者证据，不是本机复跑。真实模型GPU生成、token/logp parity仍未运行，配方继续固定`forward_batch_size: 1`。不能由主机tokenizer通过就直接放宽批量，尤其不能用行数正确代替ragged attention/RoPE/KV以及CFG条件对应关系的验证。

本机仅对base/head的compat.py与text_embed.py共4个完整模块AST/compile；两份已有compat函数体AST不变，stage5个方法中只有_apply_chat_template增加一次repair调用，去掉该新增调用后方法体AST与base相同。这些同一性检查只限定代码改动位置，不保证调用链行为等价；没有导入transformers、加载真实tokenizer、执行作者20例或模型/设备，也未用假tokenizer替代。真实资源及最小观测记[原M5可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不新排必跑任务/下载大模型、自动应用候选或拼接重叠#300方案。

固定源码可用`git diff 4fb0b1f7b291f3cd5010597b85d41ed44659ef40...d41122e97e1ab631328c86baad00250c406d2b17 -- unirl/models/hunyuan_image3/compat.py unirl/models/hunyuan_image3/text_embed.py`复查。head SHA256：compat `1ddc093e2b606acbb1a698b95bc85cf9c0da3f603b7d80f70e8cbb7c3107e0fb`、text_embed `446930c6ba85af42b93c7f97737f5325bef644c9c2ba9cd281d8403dd0f2608c`。

## 行级奖励派发与分组身份

2026-10-23 阅读已有 [#517](https://github.com/Tencent-Hunyuan/UniRL/pull/517)，open/non-draft，head `fca5a55a0e5b30d108289f7394a7c50850593bf2`、目标 base 与共同祖先 `7a5303647d0c1ec27807124f70939cbb8edde6e9`（ahead 1 / behind 0，2 文件 76 增 10 删）。它为 DiffusionTrainer 增加显式 `reward_dispatch: row`，默认仍为 `prompt_tree`；已有维护者实现，不另立贡献候选。

原 `DP_SCATTER` 按完整 root 树拆 Sample，奖励 DP 数须整除 prompt 数。若 8 个 prompt 各生成 8 行，按行评分有 64 个单位，可分给 32 个奖励 DP rank，而按整棵树不能这样分。`_reward_row_view` 没有直接丢掉树结构：它按每个 frontier leaf 的路径为每一层选择对应祖先，把叶路径的 `/` 换成 `:` 作为新 root，再给后续层追加 `/0`。每个生成行成为独立 root 的完整单分支链，原两 root/六 leaf 例子的每一层都变为六行。祖先行的重复不代表深复制、零拷贝或物理搬运成本已消失。

内容与身份是两套契约。原 `_build_reward_request` 仍通过 `Sample.turns()` 对齐初始提示、最近的 generation prompt 和非文本 conditioning，通过 `root_metadata()` 带入原始记录；重建后这些内容可相同。但 scorer 接收的 `sample_ids` 与 `group_ids` 变为单行谱系，原本同 prompt 的 siblings 不再共用原组身份。依赖组内比较、原 group ID 缓存或跨行状态的 scorer 应保留默认模式，不能因文本/媒体相同就视为等价。

评分后 `_attach_row_rewards` 要求返回 frontier ID 列表与预期的单行 ID **顺序和身份都相同**，再把 `rewards` 和 `component_rewards` 贴到原 frontier，保留原祖先 Parts 和生成内容。它不将评分服务返回的整份 Sample 替换训练输入，也不凭 ID 检查认证分数计算正确。train/eval 入口都改走 `_score`；默认分支仍直接 `score_and_attach(sample)`，本轮只核源码，没有执行 trainer。

放宽的仅是奖励几何。原 helper 在 row 模式检查 `(batch_size × samples_per_prompt) % reward_dp_size == 0`，rollout 仍须完整 prompt-tree 整除（调用方明确不要求时例外）。更外层训练校验还检查正值、训练 DP 和每批更新数；只通过这个局部 helper 不代表整份配方可运行。#378 的 32-GPU H3 配方是作者首个预期使用者，不是此主机数据转换对所有模型的硬件最低要求。

[复查脚本](../../experiments/scripts/verify_unirl_reward_rows.py)以 Python 3.12.13、torch 2.9.1、Pillow 12.0.0 运行两次，5 组 CPU 控制通过：原 Sample/Part 的 2→6 root 重建；乱序三层链的原 RewardRequest 提示/生成文本/条件图像/metadata 一致且 ID 改变；原 slice/concat 后合成奖励标签回贴、原祖先与输入不变；反向拼接分片被拒；8×8、reward DP32/48 与 rollout DP32 的几何边界。没有原上游测试集，标签由夹具生成，不是模型 scorer 输出。完整原类型模块载入，只在 tensor 包汇总初始化隔离 Ray transport；四个完整原 helper 以 AST 执行，未运行 RewardService、远程 dispatch、trainer、packed segment 搬运、TensorRef hydration、Hydra 组合、GPU 或吞吐实验。两版两个变更 Python 文件另做语法编译。

```bash
python3.12 -B experiments/scripts/verify_unirl_reward_rows.py /path/to/UniRL /path/to/Pillow-target
```

首次因部分克隆缺少 trainer blob 停止，补取固定源码后离线重跑通过；没有更换函数实现。head `diffusion.py` SHA256 `cdb2d5ba64f034b51ef940f7032bb34d073c61fbb1cbd71bec4ff4c4eb937949`，`service.py` 为 `679be52255e2cbfdd455bacc63ca85df610953130a872db416053cea91e10f7b`。真实奖励后端与派发条件见[原 M5 可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动采用候选、凑 32 卡或新增训练任务。

## 异步计时窗口与性能比较

2026-10-23 阅读已有 [#539](https://github.com/Tencent-Hunyuan/UniRL/pull/539)（#455 的计时部分），读取时 open/non-draft，head `89fdd083a4cefd090e0d46fa37da37abe185d850`、目标 base 与共同祖先 `9e14d007b48ad6ecabe06df1ec4691d69492ebb1`，ahead 4 / behind 0，5 文件 90 增 53 删。它补足异步 AR/diffusion 绕过同步 `train_step` wrapper 后缺失的计时，不是调度优化或新贡献候选。

旧版在 `_advantage_and_train` 内生成 `step_time_s`，发生在本轮必要的权重发布之前。新版将整轮关键工作置于 `async_step_timing`：收集、评分、训练及其指标/进度记录之后，按边界要求 quiesce、完成 carry、发布权重并按需 resubmit，检查最终队列为空；随后读取 total 并 `log_perf`，最后才 eval/save。因此同名 `perf/step_time_s` 的区间扩大，按该字段计算 samples/s 时分母也改变，不能把曲线变慢直接归因于性能退化。它仍不包含 eval/save、初始同步与初始评估、结束时 manager.close；不是整次运行的摊销墙钟。

| phase 名 | 被包装的调用 | 解释边界 |
| --- | --- | --- |
| rollout_collect_wait | manager.collect | 等待/收集本批可用组，不是每行完整生成时间 |
| reward | reward.score_and_attach | 调用方观测的评分耗时 |
| train | stack.train_track | 训练调用区间，不是单个 optimizer update 的设备时间 |
| quiesce | manager.quiesce | 发布边界的收敛调用 |
| carry_finish | manager.finish | 对已有 carry 的补完调用 |
| weight_sync | manager.sync_weights | 权重发布调用区间 |

异步没有 `generate` phase，因为生成与训练重叠；不能用 `step_time - train_time` 或 phase 总和倒推出隐藏生成耗时、GPU利用率或重叠率。未包装的准入、组装、检查和日志等也在 total 区间，phase 不是完整排他的设备时间分解。包装器仅在调用返回时用 `time.perf_counter` 计主机墙钟，没有 CUDA event 或显式设备同步；是否已经等到远端设备完成，须另核真实 RPC/后端语义。

`PhaseTimer` 两版 AST 相同：同名 phase 累加，异常退出也在 `finally` 记录；`async_step_timing` 的 `finally` 只清除 trainer 的活动 timer，不冻结返回对象。`total()` 每次都是当前主机时钟减构造时间，新循环在窗口后紧接着读它，所以总时长截至**读取点**，不能把 context 退出当成缓存的精确终点。窗口内异常会清除活动 timer，但控制流跳过后面的 `log_perf`，本改动不是失败轮次完整遥测。窗口外已包装的方法不计 phase；缺少某 phase 也不能自动补成零。

计时经 `log_perf(step=rollout_id+1)` 写到 `rollout/step` 横轴，不是新增 optimizer 步。作者报告 Qwen3-4B/SGLang 的异步 100 rollouts/400 updates 和同步 32/128 的 W&B 核验；本机只做两版各四个完整 Python 模块编译、`PhaseTimer` AST 同一性及 3 组真实标准库控制（phase 累加/异常、正常窗口清理与 total 继续取时、异常窗口清理），没有假训练循环/时钟，也未执行 wrapper 安装、trainer、Ray、W&B 或设备。复现代码与 hash 见[当日日志](../../diary/2026-10-23.md)；不新增一次性工具。真实跑次比较的前提补[原 M5 参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不启动新训练或自动合并 #620 等指标分支。

## SDE 窗口的停留步数与坐标

2026-10-25 核验。[#488](https://github.com/Tencent-Hunyuan/UniRL/issues/488)中原本列出却无法派发的 `decay` / `exp_decay`，已由同一作者在[#489](https://github.com/Tencent-Hunyuan/UniRL/pull/489)补齐，2026-09-23合并（head `dfc35aaa`，merge `951a0734`）。RFC仍开放不表示这两种策略尚待实现；TempFlow／Flash是另两条算法路线，不能由窗口调度完成推断全部完成，也不另认领已有参与者的工作。

**先分两个计数。** 固定main `6fd82cba13f52f394747832344039408699ea378`，[原调度器](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/sde/index_schedule.py)的输入 `step` 经当前 diffusion trainer传入的是 **rollout_id**；采样参数选好索引后，trainer把固定列表放入请求并移除scheduler。它不是每次optimizer update、每次采样调用或去噪循环步数的自动计数器。停留25意味着连续ID下25个rollout编号对应同一窗口，不保证25次参数更新。另一方面，公式里的 `t` 是**该窗口在去噪日程中的起点**，`T` 是 `num_inference_steps`：

- 线性停留次数：`max(min_iters, int(max_iters*(1-t/T) + min_iters*t/T))`。
- 指数停留次数：`ceil(iters_per_window * exp(-k*max(0, t-threshold)))`，默认threshold=13、k=0.1。

窗口起点以 `window_size-overlap_size` 为stride，只保留完整窗口，不追加裁短窗口。编号达到全部停留次数之和后，rollback=True取模回首窗，False停留末窗。覆盖full-T坐标不等于每个去噪位置最终都被选作SDE；未选位置仍走确定性求解，不能把窗口数量直接当模型调用节省。

### 原模块 CPU 序列

[复现脚本](../../experiments/unirl-window-dwell/verify.py)读取固定Git blob，完整执行原 `index_schedule.py`，不导入UniRL包、torch或设备运行时。Python3.12.13／NumPy2.5.4／PyYAML6.0.3，两次运行结果一致，[原始JSON](../../experiments/unirl-window-dwell/results.json)含源码与配方SHA256。T=10、W=4、overlap=1、init=0时，三窗依次为0–3／3–6／6–9：

| 策略与控制参数 | 各窗口停留编号数 | 回卷周期 |
|---|---|---|
| progressive，iters=25 | 25／25／25 | 75 |
| decay，max=10、min=1 | 10／7／4 | 21 |
| exp_decay，iters=25、threshold=3 | 25／25／19 | 69 |
| exp_decay，默认threshold=13 | 25／25／25 | 75 |

复现时从本仓库根目录运行 `python -B experiments/unirl-window-dwell/verify.py <只读UniRL仓库>`；该仓库须已有固定提交的源码／配方Git blob，缺失即失败，不懒取。依赖numpy、pyyaml；本机解释器为 `/private/tmp/verl312/bin/python`。

同一编号乱序查询所得窗口相同，另核周期边界回卷及不回卷时驻留末窗；这只是无状态索引选择，不认证随机轨迹一致。未运行原作者大矩阵、旧base/head对照、MixGRPO参考harness或训练。sampling／trainer两个完整模块仅AST／compile与调用点阅读，不是请求／引擎执行。

### 默认阈值不能跨配方概括

固定main四份 `*_mixgrpo.yaml` **实际全部仍为progressive**，W=4、overlap=1、iters=25。脚本只解析显式YAML标量并构造原WindowConfig，没有Hydra组合；下面最后一列是**仅切换strategy到exp_decay、保留默认threshold=13的反事实**，不是已启用配置：

| 配方 | T | 实际末窗／progressive周期 | 反事实指数停留 |
|---|---:|---|---|
| SD3 | 10 | 6–9／75 | 三窗均25 |
| Wan2.1 T2V | 14 | 9–12／100 | 四窗均25 |
| Wan2.2 T2V 14B | 20 | 15–18／150 | 前五窗25，末窗21；周期146 |
| Qwen Image | 12 | 6–9／75 | 三窗均25 |

因此README中“短日程默认阈值下不衰减”的提醒适用于这些短配方，但不能概括当前Wan2.2的T=20；它的末窗起点15已超过13。这里没有改任何上游配方，也不把未覆盖尾位置说成当前bug。

### 参考等价与加速证据另列

作者已在[#488后续更正](https://github.com/Tencent-Hunyuan/UniRL/issues/488#issuecomment-5768148430)限定：沿用MixGRPO的停留公式，但参考实现的T−2样本字典坐标与UniRL完整去噪坐标不同，不能沿用此前216/216的全序列等价说法。本机未执行参考实现，不独立认证作者矩阵。PR所报H100／SD3.5-medium／BF16 LoRA五rollout、十optimizer step smoke固定在旧head `fa01335d`，不是本机或合并最终版GPU验收；104／81秒含初始化，非稳态吞吐或质量结论。

滑动窗口决定哪些位置随机采样、各窗口停留多久；[窗口后的压缩积分](#sde-窗口后的压缩积分尾段)另决定尾段求解网格，二者不能用同一“少步数”替代实际模型调用数与完成吞吐。只在原M5入口补未来已有获准匹配观测的计数口径，不新排GPU／基准任务、不改专项或对外发布；下一轮转独立真实需求或实质反馈，不扩窗口矩阵、修改阈值或重复实现已有策略。

## 采样参数的构造校验与后续修改

**构造时有效，不代表对象在整个生命周期都有效。** [#636](https://github.com/Tencent-Hunyuan/UniRL/pull/636)在2026-10-30核为open、无评论／review；head `13847881dd8bd93a58b8966b4cf60f4463812f76`，base `b8e9740e981a68a5556fbeb8225558040a4d47a4`。单文件新增四处下界：Base的samples_per_prompt≥1（AR和diffusion继承），AR的max_new_tokens≥1、temperature≥0，以及diffusion的eta≥0。它们在原类型归一化之后执行；temperature和eta的0仍可构造，不据此认证greedy rollout的训练概率或所有后端的ODE行为。

本机Python3.12.14，只用标准库加载两版完整[原sampling模块](https://github.com/Tencent-Hunyuan/UniRL/blob/13847881dd8bd93a58b8966b4cf60f4463812f76/unirl/types/sampling.py)及完整require，包壳只避开框架初始化，没有摘取方法或替换校验函数。14个自编构造输入逐一断言：

| 输入组 | base | head |
|---|---|---|
| Base／AR／diffusion的samples_per_prompt=0（三项） | 接受 | ValueError |
| AR temperature=−0.1、max_new_tokens=0、diffusion eta=−0.5（三项） | 接受 | ValueError |
| temperature=0、eta=0（两项） | 接受并转float | 相同 |
| temperature字符串0.7、max_new_tokens字符串1（两项） | 接受并归一化 | 相同 |
| samples_per_prompt=True、max_new_tokens=1.5（两项） | TypeError | 相同 |
| temperature=NaN、eta字符串inf（两项） | ValueError | 相同 |

有限值和严格类型拒绝属于原`_coerce_type`，不能把本来就拒绝的NaN／Inf或bool归功于新下界。head上的后续操作另有四组控制：对有效AR对象replace负temperature会拒绝且原对象仍0.7；直接赋NaN成功、浅拷贝也保留NaN；此后即使不传更新，replace仍因继承的NaN拒绝；显式覆盖为字符串0.6则成功产生float0.6的新对象，原对象依然NaN。普通可变dataclass没有赋值验证器，浅拷贝也不调用构造器，这不是新增回归或已发现的训练损坏。

实际[diffusionopd.prepare_part](https://github.com/Tencent-Hunyuan/UniRL/blob/13847881dd8bd93a58b8966b4cf60f4463812f76/unirl/algorithms/diffusionopd.py)通过replace只改teacher guidance_scale，[Boogu pipeline](https://github.com/Tencent-Hunyuan/UniRL/blob/13847881dd8bd93a58b8966b4cf60f4463812f76/unirl/models/boogu_image/pipeline.py)通过replace补noise_group_ids；两处都将重跑其他继承字段的校验。对固定版本`unirl/**`检索，没有找到给这四个规范字段直接赋非法值的现有生产路径；模型自身的同名temperature属性不是修改采样对象。合适的消费约定是通过构造／replace产生有效配置，不把本地赋值反例扩大成框架已发生绕过。replace是浅重建，**并不自动深拷贝全部嵌套对象**。

复现代码如下，在全新进程以`python3.12 -B - /path/to/UniRL`从标准输入运行；固定对象须先缓存，禁止lazy fetch。两次最终新进程的输出逐字一致。构造输出base前十项accepted、后两项TypeError再两项ValueError；head前六项ValueError、中间四项accepted、后四项与base相同。

```python
import copy, dataclasses, hashlib, json, os, platform, subprocess, sys, types
repo = sys.argv[1]
head = "13847881dd8bd93a58b8966b4cf60f4463812f76"
base = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
def read(rev, path):
    return subprocess.check_output(["git", "show", rev + ":" + path], cwd=repo,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
def module(name, raw):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(raw, name, "exec"), mod.__dict__)
    return mod
for name in ["unirl", "unirl.config", "unirl.types"]:
    sys.modules[name] = types.ModuleType(name)
    sys.modules[name].__path__ = []
require_raw = read(head, "unirl/config/require.py")
assert require_raw == read(base, "unirl/config/require.py")
assert hashlib.sha256(require_raw).hexdigest() == "d30f89e77ff72e9dddaaf8b8f0ab403790b3e2984fbd6afea28f6bcc60177238"
module("unirl.config.require", require_raw)
results = {}
for label, rev, expected_hash in [
    ("base", base, "ab5f16dbdaf1db1c3eb2f6666b0cccb449732fab2763433aa7156c3b4b3626ff"),
    ("head", head, "b19ece97fda68ea556fcd4a87d25f2adf77bb7d8d73ab1a97b8bbab166c79539")]:
    raw = read(rev, "unirl/types/sampling.py")
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    m = module("unirl.types.sampling", raw)
    cases = [("BaseSamplingParams", "samples_per_prompt", 0, "ValueError"),
        ("ARSamplingParams", "samples_per_prompt", 0, "ValueError"),
        ("DiffusionSamplingParams", "samples_per_prompt", 0, "ValueError"),
        ("ARSamplingParams", "temperature", -0.1, "ValueError"),
        ("ARSamplingParams", "max_new_tokens", 0, "ValueError"),
        ("DiffusionSamplingParams", "eta", -0.5, "ValueError"),
        ("ARSamplingParams", "temperature", 0, "accepted"),
        ("DiffusionSamplingParams", "eta", 0, "accepted"),
        ("ARSamplingParams", "temperature", "0.7", "accepted"),
        ("ARSamplingParams", "max_new_tokens", "1", "accepted"),
        ("ARSamplingParams", "samples_per_prompt", True, "TypeError"),
        ("ARSamplingParams", "max_new_tokens", 1.5, "TypeError"),
        ("ARSamplingParams", "temperature", float("nan"), "ValueError"),
        ("DiffusionSamplingParams", "eta", "inf", "ValueError")]
    outcomes = []
    for i, (cls, field, value, expected) in enumerate(cases):
        try:
            getattr(m, cls)(**{field: value})
            outcome = "accepted"
        except (TypeError, ValueError) as exc:
            outcome = type(exc).__name__
        assert outcome == ("accepted" if label == "base" and i < 6 else expected)
        outcomes.append(outcome)
    results[label] = outcomes
obj = m.ARSamplingParams()
def rejects_replace(obj, **updates):
    try:
        dataclasses.replace(obj, **updates)
    except ValueError:
        return
    raise AssertionError("replace must revalidate")
rejects_replace(obj, temperature=-0.1)
assert obj.temperature == 0.7
obj.temperature = float("nan")
assert str(obj.temperature) == str(copy.copy(obj).temperature) == "nan"
rejects_replace(obj)
fixed = dataclasses.replace(obj, temperature="0.6")
assert fixed.temperature == 0.6 and str(obj.temperature) == "nan"
results["lifecycle"] = "negative replace rejected; assignment/copy retain nan; inherited nan rejected; override repaired"
print(json.dumps({"python": platform.python_version(), "results": results}, sort_keys=True))
```

本轮没有导入Torch、Hydra、引擎或trainer，没有运行作者的132个配方块／301项扫描或上游原生测试，也不把它们写成本机验收。此题在参数对象语义内收束，不新增验证器、社区动作、GPU实验或M5队列。

## SDE 索引的两层边界校验

### 策略专属窗口检查与请求快照

**窗口是否有效取决于策略真正使用的字段，解析后的索引还可能在请求构造时再次校验。** [#633](https://github.com/Tencent-Hunyuan/UniRL/pull/633)在本节验证时OPEN，后于13:10:43Z关闭未合并，见[关闭记录](#候选关闭与证据范围)；head `795732cc378a782489d113ab8f6a4892adf0fa73`、base `b8e9740e981a68a5556fbeb8225558040a4d47a4`。与旧714b71bb分支已分叉，不是原候选简单追加：GitHub compare为ahead2／behind1，其中还包含#634合并；以下只核新旧窗口模块与指定主线接线，不把整支差异当单一guard。

新[WindowScheduler](https://github.com/Tencent-Hunyuan/UniRL/blob/795732cc378a782489d113ab8f6a4892adf0fa73/unirl/sde/index_schedule.py)对random只要求window_size≤num_timesteps；random每次从[0,M−W]抽起点，根本不读init_timestep。progressive／decay／exp_decay仍要求M−W≥init，all不构造窗口。完整原模块的三个小对照（Python3.12.13／NumPy2.0.2）如下，不是作者21,760／115,920配置穷举：

| 策略与配置 | 旧714b71bb | 新795732cc |
|---|---|---|
| random，M10／W3／init8，rollout0 | ValueError | [6,7,8]，合法 |
| progressive，M10／W4／init9 | ValueError | ValueError |
| random，M4／W8／init0 | ValueError | ValueError |

负起点仍需分层判断：新窗口M10／W4／init−1可被构造，采样参数的M≤T检查也通过，直接get_sde_indices(0)得到[-1,0,1,2]；它没有在选择方法返回时自动变成合法集合。但[主线PETrainer._build_request_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/b8e9740e981a68a5556fbeb8225558040a4d47a4/unirl/trainer/pe.py)随后用原赋值`diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)`生成请求参数快照。只执行该原赋值AST、接前述真实原dataclass对象，会抛`DiffusionSamplingParams.sde_indices [-1] outside the 10-step schedule [0, 9].`，原对象未改。因此“选择器可返回越界”与“这条PE路径会将其静默送进引擎”不是同一结论。本轮没有执行整个PETrainer、request.fork或任意引擎。

#634合并到b8时确实删去[fastvideo engine入口](https://github.com/Tencent-Hunyuan/UniRL/blob/b8e9740e981a68a5556fbeb8225558040a4d47a4/unirl/rollout/engine/fastvideo/engine.py)原来的坏索引列表检查，改由参数构造校验承担；所以本节下方“fastvideo会在驱动前显式拒绝”的旧源码结论只适用8cfbac版本。不过删去某处重复检查，不证明所有producer与engine路径都静默；PE快照正是一个重验证边界，其他路径须分别追踪，不能照搬当前PR正文的“every engine”概括。部分越界中还可能有有效SDE步，也不能一概称整段退化为ODE。

复现仅需固定对象已缓存、Python与NumPy，脚本禁lazy fetch；包壳只跳过根初始化，require／sampling／窗口模块原样执行，PE只取一条原赋值。两个最终新进程输出逐字一致。以下代码以`python3.12 -B - /path/to/UniRL`从标准输入执行：

```python
import ast, dataclasses, hashlib, json, os, platform, subprocess, sys, types
import numpy as np
repo = sys.argv[1]
old = "714b71bb5dce6145292fb6389ffe96ae64a2abe4"
new = "795732cc378a782489d113ab8f6a4892adf0fa73"
main = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
hashes = {}
def raw(rev, path):
    value = subprocess.check_output(["git", "show", rev + ":" + path], cwd=repo,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[rev + ":" + path] = hashlib.sha256(value).hexdigest()
    return value
def load(name, value):
    m = types.ModuleType(name)
    sys.modules[name] = m
    exec(compile(value, name, "exec"), m.__dict__)
    return m
for name in ["unirl", "unirl.config"]:
    sys.modules[name] = types.ModuleType(name)
    sys.modules[name].__path__ = []
load("unirl.config.require", raw(main, "unirl/config/require.py"))
previous = load("previous_window", raw(old, "unirl/sde/index_schedule.py"))
current = load("current_window", raw(new, "unirl/sde/index_schedule.py"))
sampling = load("current_sampling", raw(main, "unirl/types/sampling.py"))
rows = []
for strategy, total, size, start in [("random", 10, 3, 8),
        ("progressive", 10, 4, 9), ("random", 4, 8, 0)]:
    row = []
    for mod in [previous, current]:
        try:
            s = mod.WindowScheduler(total, mod.WindowConfig(strategy=strategy,
                window_size=size, init_timestep=start))
            row.append(sorted(s.get_sde_indices(0)))
        except ValueError:
            row.append("ValueError")
    rows.append(row)
assert rows == [["ValueError", [6, 7, 8]], ["ValueError", "ValueError"],
    ["ValueError", "ValueError"]]
scheduler = current.WindowScheduler(10, current.WindowConfig(
    strategy="progressive", window_size=4, init_timestep=-1))
diff_params = sampling.DiffusionSamplingParams(num_inference_steps=10, scheduler=scheduler)
sde_indices = diff_params.get_sde_indices(0)
assert sde_indices == [-1, 0, 1, 2]
# Execute only the original PE assignment, not the trainer or request fork.
tree = ast.parse(raw(main, "unirl/trainer/pe.py"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PETrainer")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_request_sample")
stmt = next(n for n in fn.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "diffusion" for t in n.targets))
ns = dict(dataclasses=dataclasses, diff_params=diff_params, sde_indices=sde_indices)
try:
    exec(compile(ast.Module(body=[stmt], type_ignores=[]), "original-pe-assignment", "exec"), ns)
except ValueError as exc:
    error = str(exc)
    assert "outside the 10-step schedule" in error
else:
    raise AssertionError("the original PE replacement must reject invalid resolved indices")
assert diff_params.scheduler is scheduler and diff_params.sde_indices is None
assert not any(x.split(".")[0] in {"torch", "ray", "vllm", "sglang"} for x in sys.modules)
print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
    "three_case_results_old_new": rows, "resolved_before_snapshot": sde_indices,
    "original_pe_assignment_error": error, "source_sha256": hashes}, sort_keys=True))
```

输入完整hash随输出保留：旧／新窗口分别`1354e84c432bfe6aef5d556c583dcd3e828e3f3c895b7570fb2c0e9bfdcff6d6`／`f6c33f460f1ade4c0682ca5ddc654c5ac5b66e814ee84e0593566c990eb55f51`；PE为`367f67263b8dbea56c7bee5fbdf4576fc8ef8503c503f4042c71aa58ab02472a`。无Torch／Ray／Hydra／engine导入、真实SDE／轨迹或训练，没有重跑下方旧14组。该CPU语义更新不新增修复、外部评论或M5硬件任务。

### 历史候选与既有验证

**后续版本更新（2026-10-30）：** #634已于09:13:19Z合并，最终head `01e3955476258fab4810dd895f1f6d7b9f2e4746`、合并提交 `b8e9740e981a68a5556fbeb8225558040a4d47a4`；#633仍OPEN，head714b71bb。与下文旧be8baa25快照相比，最终sampling新增num_inference_steps≥1，并将独立scheduler长度检查改为显式列表分支后的`elif`。因此“显式空列表仍被未使用scheduler长度拒绝”是旧候选的兼容性结果，不用于最终合并版。这里仅比对完整源码diff，未重跑下面旧14组SDE／scheduler矩阵；下文保留当时的版本和结果。

2026-10-30核验：[窗口候选 #633](https://github.com/Tencent-Hunyuan/UniRL/pull/633)（head `d31017aea1c91f881043d896637b7c3a6db4bc6b`）和[采样参数候选 #634](https://github.com/Tencent-Hunyuan/UniRL/pull/634)（head `523d245e61dec961b714ba96ef9663ab877d36d4`）均为294962silent的OPEN／非draft／未合并候选。共同祖先与API base均为 `8cfbac17765004afb68fef08a4e011a4c8daf555`，分别仅改一个生产文件（10行／14行新增）；当前main `5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8` 的两文件与祖先相同。已有作者，不另认领或叠加实现。两份正文和完整diff已读；普通评论只有作者本地lint通过与CI待批准的自述，reviews／行内反馈为空，不视为维护者验收。

### 日程长度与返回集合不是同一契约

同日后续01:08Z两候选同步main5d1da78d，#633为 `714b71bb5dce6145292fb6389ffe96ae64a2abe4`，#634为 `be8baa25ece143539a3780ce35687e4566a07c2f`，仍OPEN／未合并。未截断Git树逐文件核index_schedule.py的blob仍为`5e2f7b91c9545531b02326fbdaa423a3a55551bd`、sampling.py仍为`f9c358f9fa8753c179f8836610aae561c8aedd2e`，均与上方旧输入相同；只更新身份、不将主线变更或旧14组控制计为新增验证。

窗口调度器先在自己的长度M内选位置；采样参数再对应推理的T步日程。合法SDE索引应在 `[0,T)`，而非包含T的sigma点位置。#633在非all策略构造时要求 `M−window_size >= init_timestep`，拦截窗口超出右端；all不使用窗口几何。#634在原int归一化后检查显式列表，并拒绝 `scheduler.num_timesteps > T`，允许较短scheduler覆盖日程前缀。

两者不互相替代：#634并未在 `get_sde_indices` 后检查返回集合；若单独配原WindowScheduler(M=10,W=4,init=9)，T=10仍返回 `[9,10,11,12]`。#633虽拦截此右溢出，却仍接受progressive的负起点init=−1并返回 `[-1,0,1,2]`。因此不能把两份候选概括为“所有生成索引已保证合法”。这是原构造器／选择器的可复现边界，不是已观测到训练损坏。

[vLLM-Omni原消费片段](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/rollout/engine/vllm_omni/pipelines/_shared/flow_match_sde_scheduler.py#L143)按当前sigma_idx做集合成员测试；无有效匹配的索引不会打开该步SDE。部分越界不等整段均为ODE；[fastvideo入口](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/rollout/engine/fastvideo/engine.py#L328)另显式拒绝坏列表，早于驱动调用。此处仅核源码片段，不推广到其他后端或负索引的所有消费者。

### 构造校验还收紧未使用的配置

`get_sde_indices`仍优先返回显式列表（含空列表），但#634的scheduler长度检查没有相同的条件分支。因此T=10、显式 `[]`、scheduler=AllSDE(20)在祖先返回空列表，候选已在构造时拒绝。AllSDE(20,fraction=.5)实际只返回0…9，也同样被拒绝：检查的是声明长度，不是返回池。它们说明兼容范围比“仅拒绝实际越界输出”更窄；不据此自动更改候选的fail-closed设计。

`dataclasses.replace`重新运行构造校验：保留 `[9]` 只改height仍通过，T从10缩到8则拒绝；同时清空索引与scheduler、eta设0的eval-like替换仍通过。[真实eval helper](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/trainer/diffusion.py#L246)在eta<=0时先清两字段，eta>0且改步数则先报错。本机未调用完整trainer，不能把字段替换控制称作eval管线实测。

### 原模块 CPU 对照

[脚本](../../experiments/unirl-window-dwell/verify_bounds.py)和[结果](../../experiments/unirl-window-dwell/bounds-results.json)固定上述Git对象，Python3.12.14／NumPy2.3.5执行14组base/head对照；每组断言其预期行为，包括仍接受的反例，并非“候选全部正确”的14项验收。完整执行原require、两版sampling、两版index_schedule；空包壳仅避开根初始化，未改原函数、伪造scheduler或合并两份候选。

```bash
python3.12 -B experiments/unirl-window-dwell/verify_bounds.py /path/to/UniRL
```

输入为已有固定blob的只读Git仓库，脚本禁止隐式fetch，输出源码SHA256；本轮重复结果与JSON逐项一致。范围为构造、选择和replace，未运行作者穷举矩阵、原生测试、Hydra配方组合、引擎、模型或GPU。此项CPU证据已收束，不新排M5训练；后续仅在作者候选实质修订或有真实需求时复核，不自行补丁、评论或推既有PR-head，所有社区写入仍需用户逐项明确授权。

## SDE 窗口后的压缩积分尾段

2026-10-23 阅读已有 [#545 MixGRPO-Flash](https://github.com/Tencent-Hunyuan/UniRL/pull/545)，open/non-draft，作者 ruiling-smartbear 已实现 [#516](https://github.com/Tencent-Hunyuan/UniRL/issues/516)，不另开竞争方案。固定 head `7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8`、目标 base 与共同祖先 `c6d4b11b24743fe1e933b61192f9d49558f7171e`，ahead 1 / behind 0，8 文件 222 增 10 删。**滑动 SDE 窗口与窗口后的 ODE 压缩是两个开关**：只改变窗口停留/衰减策略，并没有减少尾段积分的模型求值。

### 保留的是哪个边界

令原积分步数为 T，最后一个 SDE 步的零基索引为 w，压缩比例为 r。候选的步数为 `K = w + max(floor((T-w)r), 2)`；若 `w=T-1`，直接保留原 T 步。这里的最少 2 指**尾段 σ 点数**，不是两次尾段模型求值：窗口内 0…w 共 w+1 步，尾段还有 K-w-1 步，总共 K 步。只留一个尾点不能同时保留窗口出口与到达 σ=0。

[`compute_post_window_sigma`](https://github.com/Tencent-Hunyuan/UniRL/blob/7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8/unirl/sde/runtime.py)先取 `full[:w+1]`，再拼从 `full[w+1]` 对应的未移位时间到 0 重新采样、施加 shift 的尾段。静态分支因此保留到 **σ[w+1]**，包括最后一次 SDE 转移的终点；不是从 σ[w] 直接跳到新网格。压缩后张量长度 K+1。r=1 时网格数学上回到原日程，但重新 linspace 的舍入不保证逐比特相同。

| T | 末个 SDE 索引 w | r | 推得总步数 K | 边界解释 |
|---|---|---|---|---|
| 25 | 3 | 0.4 / 0.2 | 11 / 7 | SDE 窗口 [0,4)，其后压缩 |
| 10 | 3 / 6 / 9 | 0.4 | 5 / 8 / 10 | 对应作者 0…3 / 3…6 / 6…9 的末索引，不把 6 当排他上界 |

表中是原函数返回的积分步数，结合 SD3 的逐步调用路径理解为 NFE 预算，**本机没有观测模型调用或测时间**。r 并不是整次 rollout 的加速倍数；窗口越晚，保留的前缀越长，末窗口不再节省步数。

[`PostWindowDPMStrategy`](https://github.com/Tencent-Hunyuan/UniRL/blob/7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8/unirl/sde/kernels.py)在 `index<=w` 时调用原策略并记录每一步 `x0 = sample - σ×noise_pred`；其后改用确定性 DPM-Solver++。记录包含窗口之前的前缀，二阶历史不只来自第一个尾段点。`init_schedule` 重置包装器历史并传递给 base；尾段返回的 log_prob/mean 为 None。保留的窗口 σ 与原策略调用，不能单独证明真实模型 replay log-prob 相同，仍需对应模型、条件、精度与轨迹验证。

### 开关接线与比较口径

[`PostWindowODE`](https://github.com/Tencent-Hunyuan/UniRL/blob/7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8/unirl/types/sampling.py)只接受 order 1/2 与有限的 `0<r<=1`，默认 sampling 字段为 None。SD3 stage 显式支持；有非空 SDE 集合时才构造压缩步数和包装器，空集合沿全日程。trainside 按 diffusion stages 的支持标志放行；`ensure_sample_sigmas` 的能力拒绝在“已有显式 sigmas 就返回”之前，不能靠预填张量绕过其他 stage/server engine 的检查。这些接线仅源码核对和完整模块语法编译，没有构造替身 pipeline 来冒充执行。

新 [`sd3_mixgrpo_flash.yaml`](https://github.com/Tencent-Hunyuan/UniRL/blob/7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8/examples/diffusion/sd3/sd3_mixgrpo_flash.yaml)还把 `iters_per_window` 从继承配方的 25 改为 20，并更换日志标签。直接比较两份默认配方不是仅改变尾段求解器的单因素实验；观察 reward-vs-wallclock 时须另固定窗口调度、更新量、数据、模型及其余配置。不能将这个 SD3 opt-in 外推为 Qwen/FLUX、全部动态 shift 模型或 server engine 已支持。

### 本机数值检查与仍缺的证据

[复查脚本](../../experiments/scripts/verify_unirl_post_window.py)在 Python 3.12.13 / torch 2.9.1 的 CPU 环境两次运行一致：

- 720 组静态日程：T 为 2/3/10/25/50、所有合法 w、r 为 .02/.2/.4/1、shift 为 1/3。与祖先原完整 runtime 对照未启用日程；检查步数、前缀到 σ[w+1] 逐比特不变、严格递减、有限及末点 0。r=1 的最大 σ 差为 `1.1920928955078125e-7`。
- 64 组小张量求解控制，每组初始化并完整运行两次：原 DPM2 作为前缀 base，候选包装器作尾段，向量场为夹具 `0.2*x + 0.01*(index+1)`。检查前缀与独立原 DPM2 相同、保存的 x0/历史计数、所有结果有限和重置后结果相同；一阶尾段每个局部转移对照 Euler，最大差 `5.960464477539063e-8`。**前缀不是 Flow SDE，既没有随机 SDE/log-prob，也不是 SD3 模型或原作者测试。** 该控制不认证完整 Euler 轨迹、二阶精度/收敛阶或 MixGRPO 参考实现 parity。
- 两版 6 个变更 Python 文件共 12 个完整模块 compile；七个既有 DPM 定义 AST 相同。完整原 sampling/runtime/kernels/require 模块载入，空包壳只隔离无关根初始化，没有替换数学函数、安装 diffusers/模型或导入 trainer/engine。

```bash
python3.12 -B experiments/scripts/verify_unirl_post_window.py /path/to/UniRL
```

脚本固定 Git 对象、禁 lazy fetch，并打印载入源码 SHA256；关键 head runtime 为 `1f26e564758b40112c7feecda926fec3cc8bfcc3898a1e9b47c629ef85b927cb`，kernels 为 `784a0a223d8d82c4414d474ed0382ba8f9eedb766dfc9474a91826682bf90b1c`。本轮未运行动态 shift/diffusers 分支、原 156 例参考 harness、SD3 stage/replay、Hydra 或训练。PR 正文的参考误差 ≤4e-7、8×L40S 六次 rollout 的时间及平均 replay-logp 误差 1.4e-5 均是**作者报告**，不是本项目结果；作者也明确未做 reward-vs-wallclock 训练对比。真实模型证据缺口只补[原 M5 可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动采用候选、加载权重或启动训练。

## Replay 分块与反向更新边界

2026-10-23 核对 [#615 replay 架构 RFC](https://github.com/Tencent-Hunyuan/UniRL/issues/615)及实际主线。其状态表仍把 [#465](https://github.com/Tencent-Hunyuan/UniRL/pull/465)列作开放实现，但 API 已确认它于 2026-10-18 合并，merge commit 为 `3ecbbc32b2fc373ba30b9842032552e041e84b12`。当前 main `6fd82cba13f52f394747832344039408699ea378` 经 API compare 在该提交之后 5 commit；本地浅克隆的祖先查询不作为反证。以下以当前主线源码为准，不将已合并设计重新当贡献空缺。

### Forward 批形状与反向时机

[`BatchedStepReplayMixin.replay_steps`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/types/batched_replay.py)按 `batch_replay_steps`、目标步数和 SDE strategy 决定组装方式：合并时按 step-major 拼为 `[C×B,...]`，否则按目标步逐个 forward，再把 logp/means 拼回 `[B,C,...]`。它不在内部 backward；串行 forward 返回的多份结果仍共同连到调用方的损失图。因此“小 forward”与“及时释放每块反向保存的中间量”不是同一个优化。

当前 [FlowGRPO](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/algorithms/flowgrpo.py)和 [FlowDPPO](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/algorithms/flowdppo.py)用 `timestep_chunk_size` 切目标 SDE 列表，每块 replay 后立即 backward，再请求下一块。设目标步数 S=8，单流 mixin 的行为可分为：

| batch_replay_steps | timestep_chunk_size | 算法一次 replay 的范围 | backward 边界 |
|---|---|---|---|
| False | None | 8 个逐步 forward，再返回整组结果 | 整组一次 |
| True | None | 符合 SDE 条件时合成 8B 行 | 整组一次 |
| False | 2 | 每块两个逐步 forward | 每块一次 |
| True | 2 | 符合条件时每块合成 2B 行 | 每块一次 |

表中是源码请求/拼接形状，不是实测模型调用、吞吐或峰值；不外推为 CFG、joint A/V、Navit packed 或 per-sample replay 的完整执行矩阵。分块使每块 backward 可以释放该块保存的反向中间量，但 rollout latents、参数/梯度、anchor、输出数据和通信缓冲并不随之全部消失；不是以 k 定义的总显存硬上限。

### 尾块权重和指标汇总

每块的原 loss 为 `[B,C]` 元素均值，backward 乘 `loss_scale × C/S`，报告的 raw loss 则累加 `chunk_loss.detach() × C/S`。在每个 chunk 使用相同 B、每步输出和可分离目标相同的前提下，这是整张 `[B,S]` 均值的分解；不能推广成任意算法、随机/跨样本层或任意不同批形状下的逐比特等价。尤其 S=5、k=2 时权重应为 2/5、2/5、1/5，不是每块 1/3。当前算法不会在块间 optimizer.step，优势也由输入提供，不在每块重新组内归一化。

FlowGRPO 多块时拼接 detached 新旧 logp 后重新计算完整 ratio 指标；FlowDPPO 对可平均的指标按步数加权，std 重算全量 ratio，min/max 取各块极值。std 不是各块 std 的加权平均。保存 detached `[B,S]` logp 用于指标不保留模型反向图，但仍占数据存储，不能把“图释放”读成“全量指标数据也已删除”。

本机只做两个损失聚合控制：从原 base.py 完整提取 `grpo_clip_loss`、`select_active_tokens`、`slice_step_chunks` 和 validator，torch 2.9.1 / CPU / FP64，固定 2×5 合成 logp 与正负优势，两次结果一致。完整均值与正确加权块的 loss 均为 `-0.16236533560257804`，输入 logp 梯度最大差 0；错误等块权重的 loss 为 `-0.1651367206401591`、梯度最大差 `0.06666666666666665`。全量 ratio std 为 `0.1564756634067121`，按步数平均的块 std 为 `0.15847379215833968`。这是原元素损失的直接 autograd，不是完整算法、模型 replay、BF16/AdamW、FSDP 通信或学习收益验收；没有替身 stage/trainer。复现完整代码和校验值留在[当日日志](../../diary/2026-10-23.md)，不新增永久实验工具。

### Anchor 几何和通信条件

两算法的 replay anchor 与训练使用同一 `slice_step_chunks`。FlowGRPO 的 no_grad anchor 在拼接后转 CPU；FlowDPPO 逐块将 logp/means 转 CPU再拼接，且 rollout-sourced logp 模式仍需 replay means 供 KL-ADV。匹配 chunk 形状消除一项 anchor/训练几何差异，但 no_grad、同权重或同形状均不能单独证明真实模型 ratio=1。当前 guard 要求开启 batched replay 时使用 replay anchor；这不是所有 dtype、随机状态和内核的数值认证。

[`TrainStack._run_update`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/single.py)在整次更新前 zero_grad，遍历 micro 后才尝试 optimizer_step。它按 `do_optimizer_step and i==last_micro` 设置 grad-sync；[backend setter](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/backend/base_backend.py)只在延期同步启用等条件满足时切 FSDP2 flags。不能把作者“额外 backward 增加 reduce-scatter”的特定跑次描述泛化为每个 chunk 必有通信；必须连同 backend、defer_grad_sync 与所在 micro 一起观测。

本轮 5 个完整主线模块 compile，仅执行四个纯函数 AST 和小张量聚合；没有运行 replay mixin、FlowGRPO/FlowDPPO、reference-KL、TrainStack 或设备。#465 原报告的一张4090、S4/micro1、FP32 梯度误差/峰值与 BF16 偏差属于当时作者版本，不能搬为当前 main 的实测结果。真实几何、ratio/梯度及通信缺口记[原 M5 可选参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，缺现成获准匹配配方就保留未测，不新增训练或重新实现已合并功能。

## 教师指纹与控制器失败传播

**检测到异常、检测到内容不同，以及让卡住的读取结束，是三个不同保证。** [#515](https://github.com/Tencent-Hunyuan/UniRL/issues/515) 的原帖担心一个 rank 读教师失败、其他 rank 进入 collective 后控制器等到超时；但维护者[后续说明](https://github.com/Tencent-Hunyuan/UniRL/issues/515#issuecomment-6138540697)指出，当前 actor 构造链已有 #258 的 fail-fast collector。该 issue 已派给 inaniloquentee，其后续 [#575](https://github.com/Tencent-Hunyuan/UniRL/pull/575) 收窄为权重指纹一致性，于2026-10-13 05:37:27 UTC 合并，merge `6c828f5ed81649153315d996f0f597e8808ff144`；不是尚无人处理的原始挂起修复。

### 控制器传播的是已经返回的失败

固定主线 `6fd82cba13f52f394747832344039408699ea378`：[remote_hydra](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/hydra.py)进入 placement.remote；[GroupHandle 构造](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/group/handle.py)把各 worker 的 add_remote refs 交给 get_actor_results。backend 的[_inject_structural](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/backend/base_backend.py)在 LoRA 结构注入时调用 FrozenAdapters.inject，而不是控制器外单独执行的文件读取。

[get_actor_results](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/group/ray_utils.py)每次 ray.wait 一个完成结果，用 ref→rank 映射保持最终返回顺序。某个 ready ref 的 ray.get 抛异常时，立即尝试 mark_poisoned、对 pending refs 调用 `ray.cancel(force=False)`，然后重新抛原异常；不先等待所有 rank 成功。poison 失败会记录日志，取消也是 best-effort，运行中的 actor 可能继续执行。这不是 collective 自己已恢复、其他 rank 已退出或旧资源已释放的证明。

构造处没有显式传 timeout，collector 默认 None。一个读取一直不返回、也不抛错的 ref 不会凭空变成失败结果；若所有 pending 都无结果，collector 没有“快速异常”可传播。显式 timeout 的分支抛 GetTimeoutError，但该分支本身不调用上述 poison/cancel helper，不能由超时返回推导远端任务已被终止。完整驱动退出和资源回收另有生命周期，不从这两个函数单独认证。

### 指纹比较发生在读取之后

当前 [FrozenAdapters.inject](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/lora.py)逐教师完成本地读取、结构校验和 deferred-load 注册后返回 hash；有教师时才 all_gather_object 比较整份 name→hash 字典，不一致则渲染各 rank 字典并拒绝继续。它不是为读取错误新增共识层，也不为文件系统/Hub 下载加超时；卡在 gather 之前的读取仍未被这层解决。没有教师时跳过该 gather，是源码分支，不是本机分布式运行结果。

`_weights_sha256` 对排序后的 key、dtype、shape 与 tensor bytes 求 hash，先 detach、CPU、contiguous，因此文件编码与原 tensor stride 不决定指纹。它描述读入的权重，不包括任意 PEFT 配置、基座模型、Hub 提交身份、RNG或可用性。后续 `_load_frozen_adapter` 用 `broadcast_from_rank0=False` 让各 rank 从自己的权重副本加载 shard，所以检查内容一致有意义；hash 一致仍不是已经完成 materialize、正确安装或 forward 相同的证明。

### 本机只验证纯指纹函数

Python3.12.13/torch2.9.1，执行当前原 `_weights_sha256` 的完整函数 AST，6组真实CPU小张量控制通过：字典插入顺序和相同值的非连续布局不改变 hash；改一个值、dtype、shape 或 key 都改变 hash。样本 digest `aa15b9a280dfbf32b820988104f268fe6120c7d448bade77e549a8cbbce2f5bc`。这是函数的有限输入控制，不是密码学碰撞证明或跨 rank 实验。

5个完整当前模块仅 compile（lora、ray_utils、handle、hydra、base_backend）；未导入 UniRL/PEFT/Ray、启动进程组或设备、执行注入/gather/取消/驱动退出。作者 #575 报告的双进程Gloo、真实Ray异常传播及更早版本H100 smoke不计为本项目结果。复现代码与源码hash留[当日日志](../../diary/2026-10-24.md)，不另造分布式替身、不复跑旧硬件报告或新增M5任务；读取卡住的契约若将来要改，须先确认负责人、允许的超时/终止语义及实际环境，而不是直接套一层collective。

## 离线 embedding 的身份与索引发布

**缓存中存在prompt key、能读到tensor，以及tensor来自正确编码器，是三层不同保证。** 固定当前主线 `6fd82cba13f52f394747832344039408699ea378`，GitHub API与本地一致；这里阅读的是已合入 [#513](https://github.com/Tencent-Hunyuan/UniRL/pull/513) 后的当前代码，不重复上一轮RFC合并状态尽调，也不据此宣布新bug。

### 哪些身份被检查

[offline_text_embed.py](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/minimax_h3/offline_text_embed.py)的 `compute_prompt_key` 对原始UTF-8字符串做SHA-256并截16位十六进制，**不strip**；`prompt`和` prompt `是不同key，不能将RFC中的示例归一化方式当成当前实现。

| 入口 | 检查内容 | 不能单独证明 |
| --- | --- | --- |
| reader `from_dir` | schema/extractor/target_layer/feature_dim四个结构字段 | 与当前训练encoder checkpoint/revision及dtype一致；读者不接收这些expected参数 |
| `verify_coverage_or_raise` | 每个prompt的key在index.entries中 | 分片存在、可解析、含目标tensor或tensor内容正确 |
| reader `get` | 分片存在、safetensors取key、tensor为二维 | 第二维及dtype符合当前运行目标、数值与实时编码一致 |
| writer `resume` | 四结构字段加model_checkpoint字符串和dtype字符串 | 同一路径下权重/processor/tokenizer内容未变；index没有权重内容hash |

writer `add`另检查 `[L,5120]` 和目标dtype，不能与reader入口的保证混用。[Bundle.from_config](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/minimax_h3/bundle.py)传给store的只有cache路径；随后跳过文本编码器加载。[text_embed.embed](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/minimax_h3/text_embed.py)将读到的tensor转到运行device/dtype，这种cast不是来源身份认证。

[CLI](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/tools/precompute_minimax_h3.py)的 `--check-coverage` 走上述键检查；`--check-parity` 用本次extractor重算前若干prompt并allclose，帮助检查写读往返，不是自动对照trainside编码器配置。CLI帮助已明确此边界，不能以这些限定检查推导线上训练必然一致。

### 分片先写与索引发布

writer先写 `embeddings_N.safetensors`，再写 `index.json.tmp`，最后 `os.replace` 发布index；`resume`以已发布index的shards数量选择下一编号。文件未在索引登记不等于可直接被reader发现。CLI正常异常展开时的finally会close，不能把它当SIGKILL、掉电或跨节点持久性保证；代码没有将分片与索引组成带fsync的多文件事务。

本机Python3.12.13、torch2.14.1、safetensors0.8.0，复用现有环境，完整执行原store/require/dtypes模块，包入口用空壳隔离，vendor只绑定原layer=50常量；未载入模型。真实小张量文件控制两次结果一致：3个不同原始prompt、两片、变长BF16逐值读回；resume对另一model字符串和dtype各拒绝一次；仅改index的model/dtype后reader仍读回原BF16；复制index到缺分片目录，coverage通过而get拒绝。

在原writer中仅将 `os.replace` 注入一次OSError，真实分片和tmp索引已写、旧index字节不变。新建resume writer只看到原3项，重写未登记的下一片后，原3项和新第4项均能读回。这个控制说明本地单writer在该发布点失败可恢复，不是所有中断点、并发writer、NFS/CephFS或断电验证。没有运行CLI、模型编码、原作者harness、GPU/MPS或训练parity；[完整复现与源码hash](../../diary/2026-10-24.md)留日志，不新增永久工具或M5任务。使用缓存时应分别确定模型身份、键覆盖、制品可读性和数值比较范围，不将它们压成一个“验证通过”。

## Judge 回复解析与失败状态

**判错、协议外回复与请求失败可得到相同的0分，但不是相同事件。** 固定当前主线 `6fd82cba13f52f394747832344039408699ea378` 的 [LLMJudgeRewardScorer](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/local/llm_judge.py)，默认prompt要求只回 `correct` 或 `incorrect`。实现却对整段lowercase文本先搜负向regex、再搜正向regex；它没有抽取“最终裁决”字段，也不是自然语言否定解析器。

### 原方法的本地控制

Python3.12.13、requests2.34.2，执行原 `_judge_one` 完整方法及原prompt/regex常量。属性袋只提供默认配置值，不执行scorer构造或真实配置组合；以真实 `requests.Response` 执行json和raise_for_status，唯一transport替身是 `requests.post`，不发送HTTP或读取API key。七项断言通过：

| 本地响应 | 分数 | transport替身调用数 |
| --- | --- | --- |
| `correct` | 1 | 1 |
| `incorrect` | 0 | 1 |
| `This is not the correct answer.` | 1 | 1 |
| `I cannot judge.` | 0 | 1 |
| `At first I thought incorrect. Final verdict: correct.` | 0 | 1 |
| HTTP200、choices空列表 | 0 | 3 |
| HTTP503 | 0 | 3 |

负向regex识别相邻 `not correct` / `not equivalent` 等形式，但不识别 `not the correct`；正向 `correct` 因而命中。任何负向命中又优先于整个字符串中的正向词。无法识别的正常JSON回复直接返回0，不触发重试；循环内HTTP/JSON/字段访问异常才重试，`max_retries=3` 表示共三次尝试。表中次数不是真实服务请求，也未测试网络超时；每次post有独立timeout，不是对整批设统一deadline。

长回复偏离默认单词要求，因此这些反例证明的是当前解析边界，不直接证明上游应接受任意自然语言或应该换成哪个regex。修改之前需要先确定严格verdict、显式字段或宽松文本哪一种协议，以及协议外回复与服务失败是否保留独立状态；继续追加否定词并不能完整解决自然语言范围。

### 零分如何进入本地响应

源码中缺少metadata.answer时直接记0而不请求judge；上述重试耗尽也在 `_judge_one` 内吞掉并返回0。[LocalRewardBackend.compute_rewards](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/local/base.py)随后对返回列表统一生成 `successes=True`、`errors=None`。这里是调用链阅读，没有运行RewardRequest/RewardResponse、服务网关或RewardStack，不能把方法内0分当作框架已经标记失败并过滤。

也不能泛化成“所有错误都不会中止训练”：prompt_template.format发生在try循环前，构造时缺endpoint另会抛错。当前代码在循环内的异常转零是显式策略；改变它会影响已有奖励语义，应与解析协议一起评估，不能仅凭HTTP503替身就宣称训练修复完成。

本轮没有调用任何judge endpoint、真实LLM、重试计时、模型或GPU/NPU，也没有复跑社区harness。公开关键词尽调未核实到专门的解析修复，但空检索不证明没有重叠工作；[#416](https://github.com/Tencent-Hunyuan/UniRL/issues/416)的可选provider文档议题不是解析规则许可。不新增provider/正则补丁、硬件队列或对外发布；[原方法复现、输出和hash](../../diary/2026-10-24.md)留日志，不升格永久工具。后续只有协议或具体协作范围明确后才评估贡献，不继续堆近义句测试。

## AR 种子配置与逐样本请求

**种子字段存在、请求携带种子、真实生成可复现，是三层证据。** 2026-10-30核[JianhaoYu-AI的 #630](https://github.com/Tencent-Hunyuan/UniRL/pull/630)已于10-29 03:30:24Z合并，最终head `e259ae3dd49dda0b720dc7255dd9e533f0dfbec1`、合并提交 `ffad0b832707f5f1a87609e96f351426ede3f043`。固定main `5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8`的六个变更文件逐字同合并版；旧笔记“Omni AR未读取seed”的结论不再适用于这条新路径。[#507](https://github.com/Tencent-Hunyuan/UniRL/issues/507)仍OPEN，逐请求实现不等副本engine seed偏移或生成输出独立性也已闭环。

### 合并后的逐样本调用

新[seed_ar_calls](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/rollout/engine/vllm_omni/utils/sampling.py)在seed=None时直接返回原calls。显式seed时先按call与prompt的既有顺序展开行，以`zip(..., strict=True)`对应sample_ids；每行复用原`derive_sampling_seed(base_seed, sample_id)`，只为AR阶段生成带seed的新kwargs和StageSampling，并产生一个prompt的GenerateCall。映射基于身份而非当前位置，但调用方仍须提供语义对齐的ID；长度检查不会证明身份正确或保证哈希无碰撞。

[HI3原方法](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/rollout/engine/vllm_omni/adapters/hi3.py)无DiT或有DiT／x_T recipe两分支最终均传`ar_part.sample_ids`，不是diffusion child ID；[Qwen3-Omni](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/rollout/engine/vllm_omni/adapters/qwen3_omni.py)传frontier的sample_ids。[UnifiedModelTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/trainer/unified_model.py)也改为从chunk AR shell取ID、替换斜杠后分片，避免原来的局部idx构造；这是源码接线观察，非本机Sample lineage或跨chunk rollout验收。

[原生后端generate](https://github.com/Tencent-Hunyuan/UniRL/blob/5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8/unirl/rollout/engine/vllm_omni/backends/native.py)逐call构造阶段参数，并消费完`list(omni.generate(...))`再处理下一call。因此新文档明确：显式seed可能因逐样本顺序调用降低吞吐，seed=None保留原批处理；不能只看到映射更稳定就称性能无代价。本机没有调用此后端或测量代价。

### 原 helper 的有限条件控制

[复现脚本](../../experiments/unirl-ar-call-seeds/verify.py)用Python3.12.14标准库完整执行原backends/base.py，取得真实GenerateCall／StageSampling；AST隔离完整原helper及两个派生函数、原最大seed常量，避开Sample和注册入口。九个原输入保存hash，八Python文件compile通过；adapter、trainer与native.generate仅源码核对，不执行。七组本地控制为：

| 条件 | 已确认范围 |
| --- | --- |
| seed=None，故意提供空ID列表 | 返回同一calls对象；此早返回不校验ID数量 |
| 两个call含2＋1个prompt，显式17 | 三个单prompt调用保序，AR seed等于原派生函数；输入不变 |
| prompt和ID一起重排 | 同一ID映射不变 |
| prompt和ID一起分成1＋2行 | 合并后的ID映射不变 |
| ID过少／过多 | strict zip均抛ValueError |
| seed边界与非法输入 | 0／最大非负int64可派生；负值、越界、bool、float、空／空白ID拒绝 |
| 两行复用同一个ID | seed相同，不替调用方分配独立身份 |

新AR kwargs仅为浅拷贝：本地控制中嵌套stop_token_ids列表仍共享，diffusion StageSampling直接复用原对象，其seed／recipe未改。这里的“不变”是helper调用期间不改输入，不是返回后可任意修改共享嵌套内容的深隔离保证；未为此另造生产补丁。[完整结果](../../experiments/unirl-ar-call-seeds/result.json)两个新进程逐字一致、未导入Torch。没有旧失败／新通过的adapter对照，不把作者78项／27项外部替身测试计为本机通过，也没有原生采样参数构造、RNG采样、模型输出、GPU或吞吐验收。已有作者且已合并，不竞争实现、不追加M5训练队列或写上游。

```bash
python3.12 -B experiments/unirl-ar-call-seeds/verify.py /path/to/UniRL-git
```

固定Git对象预缓存后运行不联网；不检出或改源仓库index／工作树。所有社区及既有PR-head写入仍逐项待用户明确授权。

### 合并前的解析入口

以下10-24代码分析与六组CPU入口控制固定 `6fd82cba13f52f394747832344039408699ea378`；10-26主线8cfbac177的八相关文件与其blob相同，是当时的历史对照。10-26[报告者同意JianhaoYu-AI接手](https://github.com/Tencent-Hunyuan/UniRL/issues/507#issuecomment-6416315321)之后，#630已完成逐请求切片，不能继续用当时507 in:body查询未命中或协作评论代替当前代码状态。

前置[#474](https://github.com/Tencent-Hunyuan/UniRL/pull/474)于2026-09-24合并为`67d33392254822556734fd3df76aaa668c52ee47`，只删除用os.urandom补每请求seed的patch，不是后来#630的逐请求接线。旧Omni两adapter blob为f5caa6a5／6eecc3b0、trainer为0904478a，当前完整源身份另见上述结果；历史六组控制不迁为新adapter验收。

[ARSamplingParams](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sampling.py)默认seed=None。[SGLang原解析器](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/sglang/utils/sampling.py)只有`enable_deterministic_inference is True`或`rl_on_policy_target is not None`才走确定性请求规划；显式seed却没启用这一路径会抛ValueError。启用路径但未设seed，则以42作为派生基准。不能把旧报告的“SGLang honor seed”理解为任何配置下设置seed就可运行。

Python3.12.13/torch2.14.1 CPU，使用完整原Sample/Part与ARSamplingParams构造2个prompt、各fork3个兄弟，再直接调用两个完整原参数模块，得到以下六组入口行为；config属性为本地夹具：

| 解析器与输入 | 每payload的n | 派生基准seed | 组fanout |
| --- | --- | --- | --- |
| SGLang，seed=None，未启用确定性路径 | 3 | None | 3 |
| SGLang，seed=17，未启用确定性路径 | 拒绝 | — | — |
| SGLang，seed=17，启用确定性路径 | 1 | 17 | 3 |
| SGLang，seed=None，启用确定性路径 | 1 | 42 | 3 |
| direct vLLM，seed=None | 3 | None | 3 |
| direct vLLM，seed=17 | 1 | 17 | 3 |

因此`n=1`不是把GRPO组减为一个样本。按[共享TextLMAdapter](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/sglang/adapters/text.py)源码，有基准seed时按fanout展开请求，使每个叶子sample_id对应一个payload；无基准seed时保留每父prompt的n>1。`derive_sampling_seed`由base_seed与稳定sample_id派生，不使用当前排序或分片号；这个性质属于种子输入，不自动认证采样输出对任意batch、版本或调度不变。[direct-vLLM driver](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/vllm/engine.py)复用该adapter，再把sampling_seed改名为seed，runtime交给原生SamplingParams；它没有沿用SGLang的确定性开关前提。

### 合并前的 Omni 接线与引擎副本

旧6fd82cba的[HI3 AR StageSampling](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/vllm_omni/adapters/hi3.py)只传temperature/top_p/top_k/max_tokens/logprobs；[Qwen3-Omni](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/vllm_omni/adapters/qwen3_omni.py)另带可选stop_token_ids，但当时两处AR参数构造都不读取ar.seed。这正是后来#630新增接线的范围；不能因旧HI3全文含diffusion seed就认为原AR已接通。

另一问题是引擎初始化：[UnifiedModelTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/unified_model.py)确有多引擎创建循环，r用于设备位置与AR/DiT分配；该循环未按r注入AR引擎seed。这不证明所有外部配置都用同一个seed，也不能将请求seed支持当成副本seed策略已经落实。

#474作者报告H20单卡、vLLM/Omni0.28 sampler探针64行得到55种token，重设Torch/CUDA seed可复现整批，**这是作者的sampler级结果，不是本机或多副本模型验收**。本机没有构造adapter/tokenizer/引擎、build wire或提交请求，未执行RNG采样、旧探针、模型、Ray/DP、GPU/NPU或输出确定性测试。原Sample依赖加载只绕过tensor聚合入口的Ray导出；两个参数模块完整代码以独立模块名加载，不执行SGLang包的引擎注册入口。六个相关调用方模块只compile。[完整复现/hash](../../diary/2026-10-24.md)留日志；真机前提补[原M5条件参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不补种子方案、开多节点训练或竞争报告者工作。

## 内部 Pipe 的版本字段与请求身份

**去掉一个恒定版本号，不等于去掉消息校验，也不等于其他协议不再需要版本。** [#568](https://github.com/Tencent-Hunyuan/UniRL/pull/568)于2026-10-12合并，merge `6624b28aab3299d116ae1e3d90aaa14926676911`；它删除vLLM内部driver/runtime重复的`protocol_version=1`，另清理两个模型的过期版本guard。PR明确把Cosmos3的H20问题留给[#567](https://github.com/Tencent-Hunyuan/UniRL/issues/567)，不是其硬件修复。2026-10-24核对的源码固定主线`6fd82cba13f52f394747832344039408699ea378`，当前接收/发送函数已名为`recv_request`与`send_response`，driver集中在`request`方法，不直接沿用历史PR中的私有函数名。

[driver](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/vllm/engine.py)用spawn创建同checkout的子进程及双向Pipe；这是内部进程树的控制通道，不是可独立部署、协商schema的HTTP服务。版本常量删除后，仍有以下不同层的检查：

| 层次 | 当前检查 | 本机证据 |
| --- | --- | --- |
| runtime信封 | dict、精确int且等于last+1的request_id、非空字符串command | 原函数与真实spawn/Pipe执行 |
| runtime启动与命令分派 | 首命令必须startup；未知非空命令在分派中报错 | 源码路径，不是本机LLM运行 |
| driver回复与连接状态 | reply id/command匹配、ok为bool、有result及命令对应结果形状；异常时按连接状态处理 | 原request源码，未执行状态机/销毁 |

[接收器](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/vllm/runtime.py)拒绝bool作为request_id，即使Python中bool是int子类；它只要求command非空，不负责命令白名单。额外的`protocol_version=999`当前会随请求被接收，但回复不带此字段：这是该内部信封不再解释版本，而不是跨版本兼容性已获认证。[奖励HTTP客户端](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/remote.py)仍发送并校验字符串版本`"1"`；[原生权重IPC header](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/weight_sync/transfer/vllm_native_protocol.py)仍检查自身的`VLLM_NATIVE_PROTOCOL_VERSION`及transport，两者不能随内部Pipe字段一起删除。

Python3.12.13仅标准库，完整原runtime模块加载但不调用`engine_process_main`：8组独立spawn/Pipe、共9条请求，连续id1/2及额外版本字段按原函数往返；非dict、重复id、跳号、bool id、空command、非字符串command六项均由原接收器抛ProtocolError；未知非空command通过信封层。错误分类回复与成功echo内容由夹具生成，不是runtime主循环的实际错误处理或命令执行。没有导入torch/vLLM/Ray，未执行driver构造、request状态机、模型、权重发布或设备。复现代码见[当日日志](../../diary/2026-10-24.md)；不据此提出重复清理或放宽H20 gate。

完整runtime SHA256为`66e7aa74a3f442d2b44b094c614124e4b0c91e5e137a09f901ab5216f7dc59fb`，driver为`48218188b8009e9c80b22f48bbe5bf40a247dbd4f7caf63baa55c3ecf2ca9b3c`。

## 监控奖励的目标隔离与失败契约

**小权重不是只监控，零权重也不是跨聚合方式的监控契约。** [#419](https://github.com/Tencent-Hunyuan/UniRL/issues/419)已于2026-10-20明确转派GoDiao，继续由社区作者负责。本节只解释当前主线`6fd82cba13f52f394747832344039408699ea378`的行为，不另写实现或认领需求。

### 聚合系数与组内标准化

当前[远程奖励解析器](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/remote.py#L366)把所有`required_rewards`放入分量明细与聚合列表。`weighted_sum`实际是除以总权重的加权平均；`mean/min/max`完全不看权重。issue里的历史`_aggregate_scores`片段已内联到`_parse_score_response`，不能按旧函数名判断当前实现。

在原解析方法的两样本控制中，两项主奖励均为0.5，探针分别0和1，权重为`1,1,1e-6`：

| 聚合方式 | 两条聚合奖励 | 探针权重改成0后 |
| --- | --- | --- |
| weighted_sum | 0.499999750000125 / 0.500000249999875 | 两条均0.5 |
| mean | 1/3 / 2/3 | 不变 |
| min | 0 / 0.5 | 不变 |
| max | 0.5 / 1 | 不变 |

因此，当前`weighted_sum`确实能用零权重消除该分量的数值贡献，同时保留明细；问题不宜表述成“任何方式都不能零权重”。但它既不防止切换聚合器后探针进入目标，也没有声明哪些分量是监控项。

此外，原生[Part.compute_advantages](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py#L245)按组计算`(reward-mean)/(std+1e-8)`，并转成FP32。上述`1e-6`控制经过真实Sample fork的两兄弟Part，中心化值为`±2.3841858e-7`，标准化优势却约为`±0.95974535`；两项主奖励相同，组内差异完全来自探针。零权重时优势为0；在同一0.5偏置控制中改用`1e-9`，差异因FP32舍入消失，优势也为0。结论是**优势尺度取决于组内主奖励方差、epsilon及数值精度，不能只用探针系数保证可忽略**，不是所有小权重都会被放大。当前[diffusion调用方](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/diffusion.py#L870)在需要优势时调用该归一化；这里没有执行训练损失或测量参数更新。

### 不参与目标仍可参与失败判定

零权重探针若缺失或返回NaN，原解析器仍给该样本`successes=False`；即使另外两项使聚合数值保持0.5，也不代表样本有效。[RewardService](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/service.py#L125)在附加奖励之前检查失败标志并抛错。`raise_on_failure=False`是后端异常时返回失败响应，不是让这个service忽略失败的开关。

这与维护者在[#419的设计回复](https://github.com/Tencent-Hunyuan/UniRL/issues/419#issuecomment-5573395257)一致：拟议`monitor_rewards`仍属于`required_rewards`、仍须评分且有限，只从目标聚合及总权重排除；全部分量均为监控项应在配置构造时拒绝。10月后续接手方案还选择拒绝监控名与显式权重冲突。**这些是已确定的需求，不是当前RemoteRewardSpec已提供的字段。** “与目标隔离”“探针可用”“自动停止训练”是三项不同能力，不能从第一项推导另外两项。

### 本机验证范围

Python3.12.13/torch2.14.1 CPU，执行固定版三个完整原解析方法和原Sample/Part、Batch/ref、RewardResponse模块；只跳过tensor包聚合入口里的Ray导出。输入JSON、解析器配置属性为本地夹具，未构造HTTP后端、调用服务、运行评分模型、完整trainer/loss/optimizer或设备。上层失败处理为源码追踪，不是服务实测；五模块通过compile。初次harness漏传原Sample.fork必需的sampling_params，补为原DiffusionSamplingParams后通过，不归因为上游错误。[完整复现与hash](../../diary/2026-10-24.md)留日志，无新增永久工具或M5硬件任务。

## 音频奖励配方的评估协议与后续反证

**早期曲线上升、训练无NaN，不等于同一配方能稳定走到终点。** [#475](https://github.com/Tencent-Hunyuan/UniRL/pull/475)在2026-10-24读取仍open/non-draft，head `555ebb29472b4767dcd20874d771cdbb5318ed91`，base/祖先 `1a0ddbfd414c580beabff5f8eef5226aba891d67`（ahead21/behind0），13文件835增46删。该项已有作者及社区复跑，本节核对证据和源码边界，不是本项目训练结果或新稳定化方案。

### 当前配方与不同报告

固定[CLAP-only YAML](https://github.com/Tencent-Hunyuan/UniRL/blob/555ebb29472b4767dcd20874d771cdbb5318ed91/examples/diffusion/ltx2/ltx2_3_t2av_clap_trainside.yaml)为8设备、8 prompt×8 samples、100轮、训练10去噪步/eta0.7/256×384；评估每20轮取8 prompt×1 sample、30步/eta0/512×768。LoRA rank32/alpha256，每批2次更新，常数LR1e-4、无warmup、clip0.005；YAML未写beta，原FlowGRPO默认0。正文若干较早结果使用不同生成/评估几何，不能混为同一条可比曲线。

| 来源 | 报告的评估协议 | 报告结果 |
| --- | --- | --- |
| PR正文的ODE sweep | 32固定prompt×4 samples，4步，256×384 | retained recipe的composite从−0.1784到−0.0433 |
| [9月27日社区复跑](https://github.com/Tencent-Hunyuan/UniRL/pull/475#issuecomment-5853845270)，明确555ebb29 | 8 prompt×1 sample，30步，512×768，eta0；报告按prompt固定噪声 | seed42从0.182升至@60的0.484，@100为−0.183；seed43也晚期退化 |
| [同协议beta0.04跟进](https://github.com/Tencent-Hunyuan/UniRL/pull/475#issuecomment-5854847366) | 仅加beta，其他沿seed42复跑 | @100为−0.276；该系数没有阻止报告中的退化 |

社区还报告训练rollout奖励同步下降、梯度/ratio spread先升高，而无NaN/OOM/retry。这里保留的是反对“当前CLAP-only配方已稳定验收”的证据，不独立认证实际曲线、统计显著性、听感质量或根因。早期成功评论不能覆盖后续配方变更；8样本评估也不能当完整AudioCaps质量benchmark。没有读取W&B原始history、下载媒体或复跑任何训练。

### 三种不同的音频目标

[CLAP源码](https://github.com/Tencent-Hunyuan/UniRL/blob/555ebb29472b4767dcd20874d771cdbb5318ed91/unirl/reward/local/clap.py)的base奖励是matched cosine加retrieval margin，unit weights时为 `2×matched−hardest_negative`，不是百分比或单个cosine。event-grounded overlay另加事件coverage并启用RMS归一化，AST overlay再加独立分类器项；目标和预处理都变了，不能把该变体较好的终点仅归因为同一奖励下的稳定器。

[LTX2的log-prob混合](https://github.com/Tencent-Hunyuan/UniRL/blob/555ebb29472b4767dcd20874d771cdbb5318ed91/unirl/models/ltx2/diffusion.py)在权重None时按latent元素数混合，显式w时为 `(1−w)logp_video+w logp_audio`。本配方w=1仅说明policy ratio采用音频项，不是视频分支不生成、视频参数/质量不受影响或联合模型解耦。

同文件replay把video/audio的prev_sample_means拼接；[reference KL](https://github.com/Tencent-Hunyuan/UniRL/blob/555ebb29472b4767dcd20874d771cdbb5318ed91/unirl/algorithms/base.py)按拼接张量的所有latent维求mean，没有读取上述audio权重。启用beta不自动变成音频等权KL，音频贡献仍随latent几何变化。社区将更新幅度/缺乏约束视为可能原因并建议进一步试验，但“alpha与更新次数相乘约8倍”不是对真实Adam更新的精确推导，KL未奏效也不证明KL无用；本轮不据此设计或应用另一套修复。

### 本机仅做静态配方核对

Python3.12.13/PyYAML6.0.3直接safe_load三份原YAML，核上述显式标量、defaults链与overlay字段，并以AST核FlowGRPO默认beta0；这不是Hydra组合/插值解析或实例化。8个改动Python模块仅内存compile通过，无torch/Transformers/Hydra/UniRL导入、数据转换器、CLAP/AST或模型执行。固定clap.py SHA256为 `cd0d04c18fe0af251339eaeceff25bb94cbefe76f7ce95617191434cfe2eb324`，与社区声称复跑的前缀一致，但hash相同不认证远端部署或运行记录。

[复现与完整指纹](../../diary/2026-10-24.md)留日志，真机条件与后续反证补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)。暂不把现成100轮配方作为已稳定方案推荐或自动启用；没有新增GPU矩阵、采用社区调参、改#523/AReaL或对外发布。若未来独立选择此方向，先明确同版配方、数据、checkpoint和评估协议，再决定质量与稳定性判据，不由一条无错误退出记录替代它们。

## 评估尾批的补齐与计分成员

**让调度批次整齐，与决定哪些真实样本计入均值，是两件事。** [#464](https://github.com/Tencent-Hunyuan/UniRL/pull/464)在2026-10-24读取仍open/non-draft，head `91e96bd5e3b0794c76822e67c31a9053cc757e0d`，API base `c21940d802ab9993ab87d7112615899653e7b989`，实际共同祖先 `79d1e8ab14c027d99cffe253121e965bac7d1587`（ahead3/behind2），4文件59增46删。已有作者处理，不另写竞争修复。

祖先的PE/Unified `_eval_pass` 用 `usable=n-n%chunk or n` 决定循环范围；13个prompt、chunk8只会评前8个，改chunk4则前12个。候选保留真实prompt数，按需复制最后一棵输入树补齐，再在累计奖励时剔除pad对应的尾部叶子。补齐样本依然进入generate和score，不是免费占位或由scorer跳过。

| 调用入口 | 补齐条件与单位 | 累计范围 |
| --- | --- | --- |
| PE/Unified `_eval_pass` | n>chunk才调用helper，multiple为max(1,训练batch_size)；n≤chunk原样发送 | 每chunk只累积n_real×fanout个前缀奖励 |
| AR evaluate | 每个eval batch按lcm(rollout_dp,reward_dp)补齐，两者先max(1,dp) | 只累积真实prompt数×fanout个前缀奖励 |
| 共享pad_eval_inputs | 正整数multiple下，已整除含0时原对象返回；否则复制末棵树、映射全部层ID并concat | 不执行生成、评分或过滤 |

[helper](https://github.com/Tencent-Hunyuan/UniRL/blob/91e96bd5e3b0794c76822e67c31a9053cc757e0d/unirl/trainer/base.py)不是简单复制文本：`Sample.slice`按完整root tree切片，新根命名为 `<last>:eval-pad:i`，每层保留原ID后缀，文本和metadata随原类型concat。这个名称模式不等于对任意数据ID都保证无碰撞，后续fork仍要求唯一ID。helper支持多层输入不扩大具体trainer的输入能力；[PE请求准备](https://github.com/Tencent-Hunyuan/UniRL/blob/91e96bd5e3b0794c76822e67c31a9053cc757e0d/unirl/trainer/pe.py)仍传 `require_single_input_part=True`。

### 原类型与原聚合片段的 CPU 控制

Python3.12.13/torch2.14.1，固定head完整Sample/Part、Batch/ref、primitives/sampling等叶模块执行；tensor包聚合入口仅跳过Ray transport初始化并重导出原Batch/ref。原样执行pad helper、prepare helper、PE dispatch赋值与非None奖励聚合块；没有执行完整trainer或构造替身engine/scorer。

- 原生13棵两层输入树补到16棵，三个新根及其 `/0` 子层ID、末棵文本与metadata对应，原输入仍13棵；已整除对象保持identity，空Sample返回0棵。另以原prepare helper确认PE单Part条件会拒绝两层输入。
- 聚合控制改用单输入Part；原生2×3 fork产生每root六个叶子。两个8-root块合计96个dispatch叶子，奖励是按root编号0..12和pad=999构造的身份夹具；原聚合片段仅累计78个真实叶子，均值6，pad不污染均值。原PE赋值对3-root集合保持未补齐。
- 尾块应有48奖励，删一个变47由原count guard拒绝。另用合法的 `Sample.select` 故意把三个pad root移到前面，仍48叶子，原片段接受数量但前30项均值变602.8而非真实尾部均值10；这是明确违反保序前提的对照，不是观测到真实引擎重排。

因此 `roots×fanout` 检查保证数量，不认证root身份或顺序。prefix trim还要求生成/评分保持root次序及每root相邻的固定fanout；`total_samples_per_prompt`取各模态乘积，并不执行引擎来证明实际返回满足它。奖励为None时不进入count guard，最终分母max(1,count)也不证明存在有效观测。这些边界不能据本轮夹具认定框架已坏或需要另一种修复。

作者报告的原Sample/完整评估harness、4090真实dispatch以及旧AR helper等价矩阵均未在本项目复跑。[复现与源码hash](../../diary/2026-10-24.md)留日志，非真实奖励/模型、DP gather、TensorRef传输、完整eval/请求builder或checkpoint选择验收。真机最小观测条件补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不新增必跑矩阵、启动评估或修改专项。采用候选时应先固定真实评估集合、计分成员及实际DP顺序，不将“补齐整批”当作“评估语义完整验证”。

## 激活卸载的保存对象与包装顺序

**激活卸载改变反向所需张量的保存方式，不是权重/优化器卸载，也不保证更快。** 2026-10-24读取[#626](https://github.com/Tencent-Hunyuan/UniRL/pull/626)仍open draft，作者ruiling-smartbear，head `fb2b4bacda31fd395a93bc96820d31b723390985`，base/祖先 `6fd82cba13f52f394747832344039408699ea378`（ahead1/behind0），6文件92增19删。[#625](https://github.com/Tencent-Hunyuan/UniRL/issues/625)已由此实现覆盖，不另起竞争方案。

[默认配置](https://github.com/Tencent-Hunyuan/UniRL/blob/fb2b4bacda31fd395a93bc96820d31b723390985/unirl/train/configs.py)的 `activation_offload=false` 与 `cpu_offload` 分开。[native FSDP](https://github.com/Tencent-Hunyuan/UniRL/blob/fb2b4bacda31fd395a93bc96820d31b723390985/unirl/train/backend/fsdp/wrap.py)先按需checkpoint_wrapper，再在外层offload_wrapper；没有匹配block时开启offload会拒绝。只有offload时fully_shard选其wrapper；同时开启checkpoint时，`ac_wrap_order=outside`选外层activation wrapper，inside选原block，决定重算是否重新经过FSDP hooks。

[VeOmni路径](https://github.com/Tencent-Hunyuan/UniRL/blob/fb2b4bacda31fd395a93bc96820d31b723390985/unirl/train/backend/veomni/wrap.py)先parallelize，再依次改写原block的forward为checkpoint、save_on_cpu、可选compile；不新增包裹模块，`ac_wrap_order`不用于该backend。这里的 `enable_fsdp_offload=False` 未改。保持层级为已有EP参数分组与名称提供前提，不是仅凭名称相同就证明EP通信或恢复正确。

### 原回调的有限 CPU 控制

Python3.12.13/torch2.14.1，torch git `5c4886908584029761b579af026dcfb627c84070`。从固定head原样提取native `wrap_block`及其两个checkpoint helper、VeOmni两个forward factory，真实PyTorch wrapper/非重入checkpoint/autograd执行；没有运行外层fsdp_wrap/veomni_parallelize、模块选择、分布式初始化或FSDP替身。

自编CPU FP64算子模块 `y=(x*scale)^2`，x与参数均4×3，x为stride(1,4)的非连续叶子。用记录hook包住原save_on_cpu的pack/unpack，只增加观测、不替代其复制。两条接线各四种checkpoint/offload开关，共八组：

| checkpoint | offload | 记录到offload pack的张量 | 数值及身份 |
| --- | --- | --- | --- |
| off | off | 未进入offload hook | 作为直算参考 |
| off | on | 参数、输入、中间值各12元素 | 输出/输入梯度/参数梯度与参考逐值一致 |
| on | off | 未进入offload hook | 三项数值与参考一致，不等于图不保存输入 |
| on | on | 仅输入12元素 | 三项数值一致，backward未新增offload pack |

八组均保留原参数对象及 `state_dict` 的scale key；VeOmni factory另保持模块对象身份。开启offload时，本例记录的CPU保存副本均保持dtype且不与源tensor共存储，输入stride从(1,4)变为(3,1)。保存参数副本不等于重设模型Parameter的设备/所有权；36→12是本例pack元素计数，不是去重内存、RSS或GPU节省比例。checkpoint内的保存机制替代外层hook，不能把两项技术各自的节省简单相加。

本机安装版save_on_cpu在pin_memory=True分支通过empty+copy保存；CPU路径也可复制，不应把其文档概述“CPU无复制”套给所有分支。它是否实际使用可页锁定内存取决于运行设备能力，本轮没有验证pin、D2H/H2D、异步重叠或性能。dtype相同而layout变化，也说明本例逐值一致不能推广成任意模型逐bit保证。完整复现与源码指纹留[当日日志](../../diary/2026-10-24.md)，不是官方测试或完整候选验收；没有执行optimizer/恢复、EP、compile或真实模型。

### 容量收益与作者报告

[作者报告](https://github.com/Tencent-Hunyuan/UniRL/pull/626)区分H100上synthetic模型的真实FSDP2/NCCL/EP组合与随机初始化Qwen3的固定几何比较。以单H100、190.9M参数、BF16 compute/FP32 master、checkpoint开启、8×8192为例，native FSDP峰值8.94→8.14GiB，step550.6→903.9ms，进程RSS4.44→5.94GiB；这是以主机容量和时间换设备容量，不是本机测量或真实RL收敛报告。VeOmni完整Qwen3-MoE在offload关闭基线就遇到Transformers签名不兼容，不能归因于新flag，也不能据synthetic EP通过宣布完整MoE可用。

真实配方的最小接续条件留[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)：先固定版本/配置与获准资源，再比较loss、梯度/更新、checkpoint及GPU/host峰值、步时；没有匹配条件就记未测。不自动启flag、修改现有训练、部署新环境、扩充算力矩阵或改#523/AReaL专项。

## 视频 VAE 分片的状态协商与复本

**拆分解码计算，不等于拆分最终视频的驻留，也不自动构成通用故障恢复。** [#469](https://github.com/Tencent-Hunyuan/UniRL/pull/469)在2026-10-24读取仍为open draft，head `363d5bacf71113d147dd08ec28f733b1b8ab76fb`；API base `f989c03825f187607bcbe6eaf6178909b798b43f`，实际共同祖先 `c21940d802ab9993ab87d7112615899653e7b989`，ahead 6 / behind 1。已有作者在带原生H3 Ulysses支持的fork上开发，本节限定源码契约，不是新修复或候选验收。

### 生效依赖与两次汇总

开关 `video_decode_shard_across_sp` 默认false；关闭或未取得SP组时，[helper](https://github.com/Tencent-Hunyuan/UniRL/blob/363d5bacf71113d147dd08ec28f733b1b8ab76fb/unirl/models/minimax_h3/vae.py)直接走原 `vae.decode`。开启时检查distributed/VeOmni状态、组大小和组内rank；状态未初始化只对指定ValueError文本回退，不吞任意异常。固定主线 `6fd82cba13f52f394747832344039408699ea378`及候选的[SP注册入口](https://github.com/Tencent-Hunyuan/UniRL/blob/363d5bacf71113d147dd08ec28f733b1b8ab76fb/unirl/train/backend/veomni/sp/diffusion/models/__init__.py)均仅导入flux2/qwen_image/sd3/wan，没有H3边界适配器。标准backend请求不支持的SP>1会报缺边界spec，不能说配置自动降到1；常规H3 SP1路径才沿用原解码。作者的fork性能不是当前主线现成能力。

所需SP前提满足时，各rank已有相同的完整latent序列；提案按clip编号round-robin分配 `_decode_clip`，不是每rank各处理不同sample。

| 阶段 | 所有rank的动作 | 保证与前提 |
| --- | --- | --- |
| 本地decode与检查 | 捕获异常，清空失败rank的local结果；检查5维、同rank等形状和dtype | 错误先记下，不在此处单独退出 |
| 状态汇总 | all-gather固定7个int64，包含ok、是否有clip及5维shape | 无clip的rank仍参与；共同发现失败或跨rank不同shape，在payload汇总前抛错 |
| payload汇总与复原 | 填充ceil(chunks/world)个clip的等形slab，all-gather后按编号复原 | 每rank得到全部clip并执行blend/concat；要求共同chunk数、dtype/device及可用通信组 |

7个clip/SP8时，第8个rank没有本地decoder工作，却仍交换状态及填充slab，最终也持有全部clip。计算冗余减少不意味着最终视频或通信缓冲按SP度缩小。probe不核对latent内容或chunk数；一致拒绝依赖所有rank都能到达并完成相同的状态collective。组发现失败、状态分配失败、后续slab分配/通信故障或进程消失不在本地decode的try内，不能推广成任意故障都不会挂住。

### 私有算法与公开包装

固定head的[vendor实现](https://github.com/Tencent-Hunyuan/UniRL/blob/363d5bacf71113d147dd08ec28f733b1b8ab76fb/unirl/models/minimax_h3/vendor/autoencoder_kl_minimax_h3.py)与API base、固定main的文件字节相同。本机Python3.9.6仅静态检查：将self改名为vae后，原 `_decode` 的几何/补齐8个语句、每clip内部blend循环，以及循环后拼接/裁尾AST均与候选对应段相同。三项检查、三个候选Python文件加vendor共四模块compile通过；不是执行解码后的数值parity。

分片helper复用的是私有 `_decode` 算法，而非公开 `decode` 包装。后者带forward hook，且 `use_slicing && batch>1` 时逐sample解码；分片路径直接调用 `_decode_clip`，未复制这两项包装行为。空间tiling仍在原 `_decode_clip` 内。内层AST相同不能认证所有batch布局、offload hook或未来revendor版本；作者报告主要配方的forward batch为1。

### 作者证据的比较对象

[PR报告](https://github.com/Tencent-Hunyuan/UniRL/pull/469)给出8×H20、768²×124、固定SP8的开关对照：video decode/sample 51.47→8.22秒，整步3536.2→2160.7秒，峰值显存两臂相同。作者另报真实VAE三seed输出逐值相同，以及32个完整视频hash与各rank一致。相同reward mean不足以替代全视频比较，因为该报告的VideoPickScore只取中间帧。

[后续150轮报告](https://github.com/Tencent-Hunyuan/UniRL/pull/469#issuecomment-5742687138)比较SP1/DP8与SP8/DP1，并降至192²×124；两臂均完成、无hang，是该fork配置的存活/稳定性证据，不是纯decoder因果对照，因为DP拓扑也变了。作者CPU Gloo异常注入和真实GPU结果均未在本项目复跑；本机没有导入torch/模型、启动进程组或复制假VAE验证。[当日日志](../../diary/2026-10-24.md)保留复现和源码hash，真机条件补[原M5参考入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)。尚缺主线H3 SP前提，不据草案新增必跑矩阵、部署环境或竞争实现。

## VeOmni 生成代码的 mask 接口边界

**包能解析或导入，不代表按另一版Transformers生成的模型能前向。** [#627](https://github.com/Tencent-Hunyuan/UniRL/issues/627)由ruiling-smartbear在[#626](https://github.com/Tencent-Hunyuan/UniRL/pull/626)验证中独立登记，2026-10-24读取OPEN、0评论：作者报告4×H100、EP2×FSDP2、activation offload关闭时，首次前向因`create_causal_mask`不接受`cache_position`失败，不能归因于卸载flag。本机未访问其Modal作业或执行训练。

固定UniRL main `6fd82cba13f52f394747832344039408699ea378`的[依赖](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/pyproject.toml)为Transformers `>=5.12,<5.13`、Linux VeOmni extra `==0.1.11`，vLLM extra另精确钉5.12.1。[Qwen3-MoE bundle](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/qwen3_moe/bundle.py)显式调用`ensure_qwen3_moe_installed`，shim随后导入该VeOmni模型。选择性导入避免全模型zoo执行，但不消除被选择模型的接口依赖。

### 生成配置与生成产物

VeOmni官方v0.1.11 `f90b3dc6fbb0ce693745223cc7a94064123dbf4d`的[生成配置](https://github.com/ByteDance-Seed/VeOmni/blob/f90b3dc6fbb0ce693745223cc7a94064123dbf4d/veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py#L238)及[GPU产物](https://github.com/ByteDance-Seed/VeOmni/blob/f90b3dc6fbb0ce693745223cc7a94064123dbf4d/veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py#L606)均传六个关键字给`mask_function`，含`cache_position`；按sliding_window选择全量或滑窗函数。Transformers v5.12.1对应`ddb849abe009d1089e6c691bfc897f27211c663c`的[两个实际签名](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/masking_utils.py#L893)都没有该参数或`**kwargs`；残留docstring仍列deprecated参数，不能代替签名。

VeOmni [#774](https://github.com/ByteDance-Seed/VeOmni/pull/774)已于2026-05-29合并为`2d36580b80a6579f95a5318c83939844aee93fb2`，升级至Transformers5.9并调整生成配置，Qwen3-MoE的mask调用不再传该关键字。官方v0.1.12 `fd99abfda9ef4d9d485f0dae14841de88d30963d`的同一配置和GPU产物也都移除了它。已有接口调整不等于UniRL整体升级安全；并行化、模型输出、选择性导入及实际安装制品仍须联合核验。#774作者的A100测试不是UniRL5.12.1验收，不另写竞争吞参shim或自动升级依赖。

### 发布制品与初始化入口

2026-10-30 13:19Z接续：[#639](https://github.com/Tencent-Hunyuan/UniRL/pull/639)仍OPEN／未合并，head更新为`d37dd75014cce902a90678dfbfca74157cfcd1e6`、base仍b8e9740e；完整两文件diff为pyproject的0.1.12 pin与新增[backend README](https://github.com/Tencent-Hunyuan/UniRL/blob/d37dd75014cce902a90678dfbfca74157cfcd1e6/unirl/train/backend/veomni/README.md)。新增文档将Qwen3-MoE与两个Qwen3.5-MoE配方统称经过同一坏mask路径；下列既有wheel核对仅覆盖Qwen3-MoE生成文件，不能认证这个三配方概括。文档所称0.1.13“从未发布”也强于本机当前PyPI版本API的404证据。这里不迁移旧head的完整TOML比较／签名绑定为新head验收，也不据作者文档宣布#627闭环。

**源码标签、可安装制品和消费端接口，是三项不同约束。** 新[#639](https://github.com/Tencent-Hunyuan/UniRL/pull/639)在2026-10-30读取open、无评论／review；head `97ed729b2ca2952420cb109714b890f1732a7a01`、base `b8e9740e981a68a5556fbeb8225558040a4d47a4`。对两版完整pyproject做TOML解析，移除veomni extra后整个配置dict相同：语义只把Linux `veomni==0.1.11`改为`==0.1.12`，其他变化为注释，不是新的兼容shim。作者对14个导入符号的审计是作者报告，本机本轮限定下列入口与发布制品。

从[PyPI官方0.1.11元数据](https://pypi.org/pypi/veomni/0.1.11/json)和[0.1.12元数据](https://pypi.org/pypi/veomni/0.1.12/json)读取非yanked wheel，在内存取回ZIP、校验整包字节数和官方SHA256，再只读METADATA与三个模块，未解压安装或导入。两个wheel的Python约束均为`>=3.11,<3.13`，不含extra marker的Requires-Dist声明集合相同；这不等于所有可选extra相同或完整依赖求解通过。

| 核对项 | 0.1.11 wheel | 0.1.12 wheel | v0.1.13 Git标签 |
|---|---|---|---|
| 官方PyPI制品 | 1,288,446字节，5月26日上传 | 1,885,489字节，9月9日上传 | 本轮[对应版本API](https://pypi.org/pypi/veomni/0.1.13/json)返回404，不推断其他索引或历史发布行为 |
| Qwen3-MoE生成代码的mask调用 | GPU／NPU均含cache_position | 两份均去掉该关键字、仍传position_ids | 本轮未审其模型代码 |
| parallel_state公共初始化入口 | 有init_parallel_state定义 | 有init_parallel_state定义 | 无该公共定义或顶层赋值／导入别名，改为_init_parallel_state与init_parallel_state_from_config |
| 元数据版本 | METADATA为0.1.11 | METADATA为0.1.12 | 源码_version.py仍写0.1.12，不能将tag名当包版本 |

0.1.12 wheel整包SHA256为`682c0ae21b2a7835c928fb378ca740746778f5bcf5d7a64ca15718c58bf2316f`；0.1.11为`fe399ead11350fa3d53cc53293cd60d0c1042f2fa15d599aeb89db9aa417b129`。0.1.12的parallel_state、Qwen3-MoE GPU／NPU三个文件逐字等于[固定tag fd99abfd](https://github.com/ByteDance-Seed/VeOmni/tree/fd99abfda9ef4d9d485f0dae14841de88d30963d)对应文件；这只认证三个文件，不是整个wheel与Git树相同。GPU产物hash为`54246091c6720cd02a10efb64f7653dc16a43e2d7b76dfef47d960d4dfc7eb26`，NPU为`83ab52e13363bed8815af3ea54fab3196ef008774387a16a447135526b855109`。

当前[UniRL backend](https://github.com/Tencent-Hunyuan/UniRL/blob/97ed729b2ca2952420cb109714b890f1732a7a01/unirl/train/backend/veomni/backend.py)仍按公共名导入init_parallel_state，传dp_size、ulysses_size、dp_mode、device_type，EP>1再传extra_parallel_sizes／names。从0.1.12原函数AST构造inspect.Signature，两组关键字都能bind；只做参数名／必需项匹配，不执行默认值表达式或函数体。[v0.1.13固定源码86c5ec1d](https://github.com/ByteDance-Seed/VeOmni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/veomni/distributed/parallel_state.py)已无该公共入口，不能用“最新版tag存在”替代消费端接口检查。

本机Python3.12.14标准库完成官方元数据／制品读取、ZIP哈希、AST／compile及签名绑定，没有pip／uv安装、Linux实际导入、函数调用、mask／attention数值、GPU／NPU或EP更新。本节补的是旧阅读尚缺的实际wheel身份与命名入口；不重跑上节旧HF签名控制，也不宣布#627训练闭环。实际mask和offload关闭的EP最小更新条件回写[原M5条目](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不新增算力队列、不自动采用pin。

### Qwen3.5 MoE 的发布文件与接线

**架构名称不同，必须检查实际模块，不能借用另一模型的验证结论。** 2026-10-30核[#639](https://github.com/Tencent-Hunyuan/UniRL/pull/639)仍OPEN，head`44f2a96db2db13cbf4eb9dc3181bd786eeaa29e4`、base仍b8e9740e。对旧d37和新44f做直接Git diff，只有backend README变化，pyproject未再修改；GitHub三点compare显示ahead1／behind1并包含共同祖先的pin差异，不应把它当成两个head的直接改动。

新文档明确将Qwen3.5-MoE纳入旧mask接口影响。本机补核[官方0.1.11 wheel](https://pypi.org/pypi/veomni/0.1.11/json)与[0.1.12 wheel](https://pypi.org/pypi/veomni/0.1.12/json)：再次核整包字节数／官方SHA256后，在内存读取`veomni/models/transformers/qwen3_5_moe/generated/`两份完整Python，AST／compile并取精确create_causal_mask调用，而不是搜索任意带mask字样的表达式。

| 官方wheel | GPU生成文件的mask调用 | NPU生成文件的mask调用 | 对应固定tag |
|---|---|---|---|
| 0.1.11 | 第2038行，含cache_position | 第1980行，含cache_position | 四文件中的这两份各逐字等于f90b3dc6 |
| 0.1.12 | 第1996行，无该关键字，仍传position_ids | 第1981行，无该关键字，仍传position_ids | 另两份各逐字等于fd99abfd |

这项核对支持两个模型家族都有相同的**调用接口差异**；并不证明0.1.12是所有组合下“首个可工作版本”。两版GPU文件均从transformers.masking_utils导入create_causal_mask；旧[HF5.12.1签名证据](#生成配置与生成产物)仍适用。0.1.12其他attention／model调用仍传cache_position，不能全局删除同名参数。NPU文件仅作发布制品核对，不表示UniRL支持NPU。

两份[文本DAPO配方](https://github.com/Tencent-Hunyuan/UniRL/blob/44f2a96db2db13cbf4eb9dc3181bd786eeaa29e4/examples/ar/qwen3_5_moe_grpo_35b_a3b_base_dapo_sglang.yaml)与[视觉Geo3K配方](https://github.com/Tencent-Hunyuan/UniRL/blob/44f2a96db2db13cbf4eb9dc3181bd786eeaa29e4/examples/ar/qwen3_5_moe_grpo_35b_a3b_geo3k_mc_sglang.yaml)都显式meta_init_transformer=true、选VeOmniBackend，并指向共用[Qwen3_5Bundle](https://github.com/Tencent-Hunyuan/UniRL/blob/44f2a96db2db13cbf4eb9dc3181bd786eeaa29e4/unirl/models/qwen3_5/bundle.py)。bundle依据加载到的HF config.model_type分支：仅在qwen3_5_moe且meta-init时调用ensure_qwen3_5_moe_installed，再build_foundation_model；[shim](https://github.com/Tencent-Hunyuan/UniRL/blob/44f2a96db2db13cbf4eb9dc3181bd786eeaa29e4/unirl/train/backend/veomni/_compat.py)选择导入VeOmni的qwen3_5_moe包。不是按配方文件名直接决定模型类型，也不是所有Qwen3.5加载都进入此路径。

复查只需Python标准库，以下命令在内存读wheel和对应tag，不安装／解压／导入模型；输出四个完整文件hash与mask关键字：

```python
import ast, hashlib, io, json, urllib.request, zipfile
refs = {"0.1.11": "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
        "0.1.12": "fd99abfda9ef4d9d485f0dae14841de88d30963d"}
for version, ref in refs.items():
    metadata = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/veomni/{version}/json", timeout=30))
    wheel = next(f for f in metadata["urls"] if f["packagetype"] == "bdist_wheel" and not f["yanked"])
    raw = urllib.request.urlopen(wheel["url"], timeout=30).read()
    assert len(raw) == wheel["size"] and hashlib.sha256(raw).hexdigest() == wheel["digests"]["sha256"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for device in ("gpu", "npu"):
            path = f"veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_{device}.py"
            source = archive.read(path)
            tree = ast.parse(source)
            compile(tree, path, "exec")
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "create_causal_mask"]
            assert len(calls) == 1
            keywords = [kw.arg for kw in calls[0].keywords]
            assert ("cache_position" in keywords) == (version == "0.1.11")
            tag = urllib.request.urlopen(f"https://raw.githubusercontent.com/ByteDance-Seed/VeOmni/{ref}/{path}", timeout=30).read()
            assert source == tag
            print(version, device, hashlib.sha256(source).hexdigest(), calls[0].lineno, keywords)
```

本机系统Python3.9.6仅标准库完成；UniRL两个完整Python模块另compile，两原始YAML只读显式字段、没有Hydra组合或模型配置加载。四生成文件hash见[当日日志](../../diary/2026-10-30.md)，不把静态接口核对升为实际mask／attention数值、三配方首forward或EP训练通过。原[M5条件参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)补模型分支身份，不部署升级、不新增必跑队列或对外反馈。

### 本机确认的调用边界

本机Python3标准库读取固定官方源码并compile六份文件；从HF AST构造`inspect.Signature`，用VeOmni原调用AST的关键字名做bind，不执行函数体。0.1.11配置/产物各对两函数均因`cache_position`拒绝；#774合并配置、0.1.12配置/产物各对两函数均接受。首次直接搜索`create_causal_mask(`未命中，是因为实际经`mask_function`调用，不是没有调用；补齐0.1.12产物后结果一致。[命令与hash](../../diary/2026-10-24.md)留日志。

HF预处理由cache的`get_seq_length/get_mask_sizes`推导偏移/长度，无cache则从输入或attention_mask推导，并有position_ids的packed-sequence检测。签名接受不认证这些数值行为、VeOmni自定义attention或EP更新；这里未运行mask、导入依赖、加载模型、升级环境或执行设备，也不删除模型其他位置仍使用的cache_position。真机缺口补[原M5激活卸载参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，先恢复offload关闭的匹配基线，再谈卸载比较，不改#523/AReaL或新增必跑矩阵。

## 视频时间步精度与比率诊断盲区

**小平均log-ratio不能排除模型条件不同，修正条件也不等于已解释奖励曲线。** [#412](https://github.com/Tencent-Hunyuan/UniRL/issues/412)仍open、已派nussejzz；维护者后续要求报告者在含[#438](https://github.com/Tencent-Hunyuan/UniRL/pull/438)的main重跑，并未提供最终确认。#438于2026-09-12合并，head`293ea337c27a0b3014a42d4dae7377340b8f6c9d`、base`27be70b3a2c37b86e34b1e24119a2de77da9043c`，合并提交`f0905f5d9db3ea7bd54af3987cc2ecf007aa2357`；一文件2增1删，不另立相同修复。当前固定main`6fd82cba13f52f394747832344039408699ea378`的实际表达式也保留FP32。

### 两种差异计数对应不同比较对象

[predict_noise](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/hunyuan_video10/diffusion.py)把sigma乘1000作为transformer timestep。旧代码先将sigma转模型dtype再乘；修复先保FP32并缩放，传给transformer的timestep仍FP32，而不是乘后再转BF16。模型latent、prompt或guidance的BF16不应反推时间条件也应BF16。

M1以完整原runtime模块生成当前[原始配方](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/examples/diffusion/hunyuan_video10/hunyuan_video10_t2v_sglang.yaml)的10步、shift=5静态sigma表；torch默认FP32。只提取旧/head/main三版原timestep赋值表达式接真实CPU Tensor，不调用模型或伪造transformer。旧结果与FP32参考有9/10不同，head/main均0/10；若额外把参考转BF16再与旧值比，仅2/10不同。后者只是双重舍入比较，不能作为修复后完整FP32行为的替代测试。

| 零起始step | sigma | 旧BF16后乘 | 修复FP32 | 乘后再转BF16的反例 |
| --- | --- | --- | --- | --- |
| 1 | 0.97826087 | 976 | 978.260864 | 980 |
| 7 | 0.68181819 | 684 | 681.818176 | 680 |

原PR正文既讨论9/10完整差异，也突出2/10大偏差，其Test Plan伪代码还出现乘后再转BF16；阅读时须以真实源码和比较对象为准，不能把“只有两步不同”或“改后仍BF16”当最终实现。

### 平均log-ratio与单点误差不同

对同方差的一维高斯，设旧均值mu、新均值mu+delta、标准差s，同一个观测x下，直接相减log密度得到：

`log(p_new/p_old) = delta*(x-mu)/s² - delta²/(2*s²)`。

单点有一次项；只有在旧分布下取期望且残差均值为0时，一次项消失，得到负的二次项。不能把作者“quadratic”解释无条件套到每个样本、每个维度或不同方差的比较。源码[FlowSDEStrategy](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/sde/kernels.py)的log密度后续对非batch维取均值，平均值也不是最大逐坐标偏差。

完整原kernels模块的compute_log_prob接float64 CPU数学控制：mu=0、s=1、delta=.001、x=-1/+1，得到约-.0010005与+.0009995，均值约-5e-7；与上述含一次项公式逐点一致。这只是解析公式的有限控制，不是实际DiT误差分布或训练ratio复现，不能从它倒算真实模型偏差。

### ODE尾段与训练证据

当前配方把SDE候选限制在[0,5)，每次选4步；原AllSDEScheduler在rollout0选1/2/3/4，其余rollout的具体四步可不同。5到9始终不在该候选范围。实际stage在非SDE步设置eta=0；原SDEStrategy._finalize_logp对此返回None，只收非None log_prob。因此第7步的条件修正不会作为本步SDE密度进入ratio，但仍可能改变终态视频；这不证明它对最终reward的影响方向或大小。

#438作者明确披露：补丁训练和基线的batch_size/GPU数量不同，按rollout画出的改善不能证明按同一消耗prompt预算也改善。其H20运行不是本机结果，维护者的“likely cause”也不是已闭环因果。Python3.12.13/torch2.14.1本轮仅三个完整数值模块、原赋值表达式、eta=0分支与raw YAML核对，无完整predict_noise、embedding、SGLang/Diffusers引擎、PickScore或训练；[命令与hash](../../diary/2026-10-24.md)留日志，缺口记[现有M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)。不重复认领、自动部署或将配方8设备写成所有路径的硬需求。

## 前缀缓存的参数与失效窗口

**配置接受某个缓存字段，不代表请求复用了缓存；字段缺省也不代表缓存默认关闭。** [#262](https://github.com/Tencent-Hunyuan/UniRL/issues/262)仍open、无assignee，但正文已有yhl48计划实现配置与文档v1，维护者要求先解决[#261](https://github.com/Tencent-Hunyuan/UniRL/issues/261)的静默丢参。该前置已由[#392](https://github.com/Tencent-Hunyuan/UniRL/pull/392)于2026-09-19合并，head`1b1c44ef686aa2fe85de119dfc78915aa0e2820c`、merge`07ac948a5d70a1a08777920fd191390fc0556ac2`。不能据旧追踪帖再认领同一丢参修复，也不能把#392视为整个缓存RFC完成。

### 字段接受与运行生效

固定UniRL main`6fd82cba13f52f394747832344039408699ea378`的[SGLangEngineConfig](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/sglang/config.py)先收engine_kwargs，再覆盖显式typed字段、extra和runtime_overrides，最后覆盖保留端口。HTTP/native boot都从实际安装的ServerArgs dataclass读取允许字段，交给[同一过滤函数](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/sglang/backends/base.py)：普通未知字段默认告警并删除，设置UNIRL_SGLANG_STRICT_SERVER_ARGS=1才拒绝；所声明必需的关键字段缺失则直接拒绝。缓存字段不在该关键字段集合，不能把“现在有告警”改述为“一切拼错都启动失败”。

UniRL Linux extra钉sglang[diffusion]==0.5.19，不等于本机已安装。官方[v0.5.19 ServerArgs](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py)声明disable_radix_cache=False，没有enable_prefix_caching字段。这只是字段默认值，未核实际模型、其他参数及运行时调整；不能由此宣布某次运行已启用或获得命中。

M1从该固定源码AST取488个直接声明字段名，未构造ServerArgs。完整原UniRL backend/base.py正常定义后，四组本地合同检查通过：disable_radix_cache=False保留；enable_prefix_caching=True被告警删除；同一未知项在严格模式抛错；concurrency作为UniRL自用字段被排除且不告警。没有导入SGLang、Hydra compose、构造engine、执行后端boot或发现设备；这不替代live dataclass、字段值合法性或服务器启动验收。

### 缓存复用有权重与生命周期边界

[engine.sleep](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/rollout/engine/sglang/engine.py)在全量释放或tags含kv_cache时先flush，再释放内存；只释放weights的分支不由这里触发相同flush。wake_up恢复资源不等于恢复先前的前缀内容。因此不笼统说每次sleep都清缓存，也不把CPU权重backup等同KV缓存backup。

[FullWeightSync](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/weight_sync/full/base.py)的flush_cache默认True；[TensorWeightSync](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/weight_sync/full/tensor.py)只在最后bucket的最后dtype组传最终flush标志，再经SGLang weight-sync seam传给后端。这是调用方的发布时序，不是M1已验服务器完成清理、并发无竞态或同一权重内容。不能为了提高命中率直接关闭刷新，否则比较可能混入旧权重下的KV。

| 要确认的事实 | 需要的证据 | 不足以证明它的观察 |
| --- | --- | --- |
| 字段可用 | 实际版本、settled ServerArgs及启动结果 | YAML存在/本地字典保留 |
| 前缀可复用 | 同一权重代、worker、实际token前缀与未失效窗口 | 同一prompt字符串/同worker |
| 缓存命中 | 后端实际计数或可追溯prefill观测 | 开关打开/路由亲和 |
| 性能改善 | 相同负载、同步节奏和测量窗口的完成吞吐/尾延迟 | 字段通过/单个命中比例 |

这里沿用[原路由入口分析](#路由策略与基准入口)，不重做同步AR绕过pool或siblings派发控制。缓存RPC、模型KV、同步与sleep实际效果均未运行；其余六个相关原模块只compile，版本/hash见[当日日志](../../diary/2026-10-25.md)。硬件观测前提补[现有M5路由参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动设置缓存/严格模式、启动配方、修改刷新或开发并行路由方案。

## PPO 的轨迹边界与价值目标

**开放追踪项不等于功能尚未实现，局部算法正确也不等于训练曲线已验收。** [#86](https://github.com/Tencent-Hunyuan/UniRL/issues/86)仍open、派haonan3；正文提议者为yhl48，维护者先限定AR并在后续说明可由社区集群补GPU曲线。其三分项[#254](https://github.com/Tencent-Hunyuan/UniRL/pull/254)、[#256](https://github.com/Tencent-Hunyuan/UniRL/pull/256)、[#259](https://github.com/Tencent-Hunyuan/UniRL/pull/259)已于2026-07-26／08-01合并，依次提供纯张量GAE、Qwen3 value与数据接线、PPO损失和配方，不另开发同题实现。

### 有效位和优化位不能互换

固定main`6fd82cba13f52f394747832344039408699ea378`的[GAE helper](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/advantages.py)逐时间倒推TD残差，每段尾部隐含next value=0。可选mask会把无效位置的递推清零，并去掉下一无效位置的value；它没有独立bootstrap值或terminated/truncated参数，不能自动代表带时限截断的通用环境GAE。

[Part.compute_gae_advantages](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py)把每条样本奖励放在其最后一个response token，并按cu_seqlens逐轨迹调用helper，**不把loss_mask传作GAE mask**。loss_mask随后用于样本摘要的取均值，以及[PPO](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/algorithms/ppo.py)中actor和critic优化位选择。被排除优化的token仍可承接后续奖励传播，不应凭同一个“mask”名字切断它。

完整原advantages模块接真实CPU张量，gamma=lambda=1，给出两个手算对照：

| 输入与处理 | 原函数结果 | 解释 |
| --- | --- | --- |
| 单轨迹reward=[0,0,1]、value全0，不传mask | advantage=[1,1,1] | 末奖励传回全部三个状态 |
| 误传优化位[1,0,1]作为有效位 | [0,0,1] | 中间位被当递推边界，前端信号消失 |
| 两轨迹reward=[1,3]、cu=[0,2,4]，分别GAE | [1,1,3,3] | terminal scatter为[0,1,0,3] |
| 错把同一packed数组直接当单轨迹 | [4,4,3,3] | 第二轨迹奖励漏进第一条 |

后两条的逐段循环按调用方切分规则执行，没有构造或运行Part；错误用法是负控制，不是声称当前代码正在这样做。

### 价值裁剪和零初始化的梯度

原value loss先detach old values和returns，用新预测与old附近的裁剪值分别计算平方误差，再取较大值乘0.5。以old全0、returns全1、clip=.2、新预测[1,2,-1]核原完整函数体，loss约[.32,.5,2]、新预测梯度[0,1,-2]，old/returns均无梯度。超出裁剪范围不意味着梯度总为0：由哪个误差分支较大决定，不能把value clipping解释成硬截所有更新。

完整原[ValueHead](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/types/value_head.py)是FP32线性头，weight/bias零初始化。真实CPU Linear输入[2,3]时输出0；对目标1做0.5平方误差，head weight梯度[-2,-3]、bias=-1，而hidden梯度为0。这仅说明零线性权重下的这一次critic贡献，不是actor无梯度、整模型停训或后续更新也相同；没有构造Qwen3或执行meta/sharded加载。

### 准备时机与训练证据

源码中PPO声明按planned micro重算values：stack先eval、按计划取值并复原顺序，再prepare_part算GAE，然后回train并更新；这不同于任意整批forward算一次old values。当前[配方](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/examples/ar/qwen3_ppo_4b_base_dapo_sglang.yaml)声明advantage_mode=gae、use_value_head=true、同步排除value_head.*。本轮只核调用源码与raw YAML，不认证micro/replay/优化或真实同步。

#259作者将8×H20、BF16、SGLang0.5.12.post1的三次更新定位为正确性smoke，使用临时交替奖励而非实际DAPO学习曲线；它不等于长程收敛，也不是本机结果。当前raw配方为32设备、batch64、model_precision=fp32，不能直接沿用该smoke作为当前默认配置的验收。M1 Python3.12.13/torch2.14.1仅四组有限CPU控制：两个完整原模块、一个AST摘取的value-loss原函数；其他六模块只compile。没有完整Part、PPO实例、stage、trainer、forward replay、actor loss、FSDP或设备运行；[命令与hash](../../diary/2026-10-25.md)留日志，真实条件补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不改专项或新增训练队列。

## 奖励共置的调度份额与 MPS 边界

Ray允许两个奖励actor共用一张卡，不等于给它们各划出半张卡。[#463](https://github.com/Tencent-Hunyuan/UniRL/issues/463)仍open，已派jyizheng和Zcchill；[#482](https://github.com/Tencent-Hunyuan/UniRL/pull/482)于关闭时注明由[#492](https://github.com/Tencent-Hunyuan/UniRL/pull/492)接替，前者未合并，后者于2026-10-14合并。#492固定head `38a13a4ec847370b38c3804af0e2a0eaee080f57`、merge `0ac8bc3e0e9f59fc7c7019aab61645f545e1650b`。以下执行链以当前固定main `6fd82cba13f52f394747832344039408699ea378`为准，不将PR正文的全部设计声明搬成主线行为，也不另开同范围实现。

### 配置份额和实际设备资源不同

[Ray官方说明](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators)把fractional GPU用于调度共置，显存用量仍由应用负责。num_gpus=0.5不是显存上限，也不代表固定吞吐份额。Ray通常在运行actor前设置CUDA_VISIBLE_DEVICES；UniRL的[_pin_cuda_to_ray_gpus](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/workers/actor.py)还会重新赋值，不能照抄该函数注释，将其解释为Ray一般不设置可见设备。

主线[_actor_options](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/workers/group.py)分别传num_gpus和MPS环境：份额不自动换算成线程百分比。官方示例是两个0.5份额actor、各100% active threads、各20GiB上限，不是各50%算力。runtime_env在actor启动前注入pipe、thread与memory变量；显存字符串中的`0=`是客户端可见设备编号，不能直接当物理卡0的观测证据。

[NVIDIA环境变量说明](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)区分客户端线程可用上限、设备内存分配上限和daemon层约束；旧MPS v2客户端设置只能进一步收紧daemon限制，已建context不随任意后改环境自动变化。当前这套变量接线不是物理显存、带宽或故障域分区，也不去重模型权重；100%只是不额外缩小该项线程上限，并非保证每个actor独占全卡。

### 主线解析成功不认证运行正确

Python3.12.13／PyYAML6.0.3仅加载完整原[config.py](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/config.py)，四组本地检查如下；没有Ray、Torch或MPS模块导入：

| 配置 | 原解析器结果 | 说明 |
|---|---|---|
| 原双奖励示例，各0.5 GPU／100%／20GiB | 接受，各20480MiB | 未加载任何模型或检查路径中的权重 |
| 单CLIP，0.5 GPU／50%／12GiB，dtype=float16 | 接受，12288MiB | 当前解析器不据半精度拒绝低线程比例 |
| active_thread_percentage=101 | ValueError | 检查整数范围1至100 |
| MPS加cluster.ray_address=auto | ValueError | 当前实现仅允许本地单节点配置 |

#492正文描述的低线程半精度guard、scorer白名单、external daemon模式及30秒协调关停，不能直接视为这份主线的已执行保护。该源码接受上述50%配置；MPS采用本服务管理的UID/PID隔离路径，而[WorkerGroup.shutdown](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/workers/group.py)直接ray.kill。这里未核保护逻辑演变的提交历史，不将描述差异定性为新引入回归或另认领补丁。

### 客户端接入和关停都需要真机证据

[create_app](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/server.py)在WorkerPool之前进入[mps_daemon](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl-reward-service/reward_service/mps.py)，后者检查控制命令、拒绝已有路径、启动daemon与watchdog，退出时请求quit并清理目录。源码顺序和目录清理并不证明actor已连到目标server、在飞CUDA工作已排空或所有进程已退出。[NVIDIA架构说明](https://docs.nvidia.com/deploy/mps/latest/architecture.html)还明确：连接daemon失败时客户端可以不使用MPS继续运行；共享卡上的致命故障也可能影响其他客户端。因此服务能评分、Python进程分离或健康ping成功，均不足以认证MPS接入或独立GPU故障域。

作者#492的H20／driver535报告也不能简化成“MPS统一加速”：其空闲确认后的CLIP+PickScore对照，model batch1为49.51→48.01 items/s（−3.0%），batch4为99.82→124.96（+25.2%）。这是作者同卡特定负载结果，不是本机测量；service batch与模型实际forward batch还需逐scorer核对。其旧未清空干扰负载的吞吐已被作者撤回；#482另一贡献者的B300报告又说明负载瓶颈与低线程数值行为可随软件／设备栈改变，不能跨卡照搬收益或放宽保护。

本机四组只验证YAML解析，七份完整源码compile不认证进程调度、kernel并发、数值或安全关停；命令与hash见[当日日志](../../diary/2026-10-25.md)。真实接入、配对分数、资源／尾延迟与清理条件补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不启动daemon、改compute mode、借用他人负载或新增基准队列；任何社区动作仍逐项待用户明确同意。

## Scheduler 子进程环境与依赖版本

**GPU可见性、库路径、preload及真实driver兼容是不同层。** [#468](https://github.com/Tencent-Hunyuan/UniRL/pull/468)把SGLang scheduler启动环境从[#225](https://github.com/Tencent-Hunyuan/UniRL/pull/225)拆出，不是TP权重同步实现。2026-10-26核OPEN/non-draft、维护者CjhHa1已有实现，head`3dac11fbdf691dde142c74b0db0a6556a0d52966`、父版／目标base及实际祖先`f989c03825f187607bcbe6eaf6178909b798b43f`，相对main`6fd82cba13f52f394747832344039408699ea378` ahead1／behind105；四文件214增5删，普通评论／reviews／行内空。已读完整正文／diff，不另开兼容补丁；[旧TP限制与后续实现](../../open-source/unirl/tracker.md#旧-tp-限制与后续实现)已有研究，不重复认领。

### 父进程恢复与子进程生命周期

[候选base helper](https://github.com/Tencent-Hunyuan/UniRL/blob/3dac11fbdf691dde142c74b0db0a6556a0d52966/unirl/rollout/engine/sglang/backends/base.py)从显式LD_PRELOAD中识别实际存在、文件名以libcuda.so开头的driver路径，再将driver目录和当前Python环境的CUDA wheel／torch库目录放在继承LD_LIBRARY_PATH之前。HTTP将环境上下文放在process.start边界；native包住Engine构造并转入显式设备列表。两者在退出finally中恢复父进程原LD_LIBRARY_PATH与CUDA_VISIBLE_DEVICES，**恢复也包含原先不存在的键**。

另一个上下文临时替换torch_memory_saver.configure_subprocess，将显式driver放在其hook preload之前，再恢复该函数；scheduler target另替换HookUtilModePreload.get_path_binary，只选恰好一个memory-saver hook。这个后者替换按设计保留在子进程生命周期，而不是退出启动上下文就撤回；两种恢复范围不能混写。真正的库映射顺序、兼容driver能否使用及SGLang私有回调接线仍须真实进程证据，不能从字符串顺序认证。

即使没有driver preload，_scheduler_spawn_environment仍会前置当前Python库目录并可能修改设备列表。PR所说“无libcuda时近似no-op”不能理解成所有环境键完全不动；这是其主动选择运行库的范围，不据此指称新故障。环境变量和函数替换是进程级状态，finally证明单个上下文恢复，不等于线程间隔离或并发启动安全。

### 依赖已经变化的部分

候选pyproject声明SGLang0.5.12.post1，而当前main声明0.5.19；[SGLang v0.5.19依赖](https://github.com/sgl-project/sglang/blob/v0.5.19/python/pyproject.toml)要求torch_memory_saver>=0.0.9.post1，不是该包的精确版本锁。不能仅由main配置推出M5实际安装版，更不能机械沿用旧“替换整个preload／整个字符串当ctypes路径”的假设。

固定[torch_memory_saver v0.0.10 `9026084a`](https://github.com/fzyzcjy/torch_memory_saver/blob/9026084a4ea65b75fd6126a6a0665a88a810d172/torch_memory_saver/hooks/mode_preload.py)：configure_subprocess已经把hook与现有LD_PRELOAD拼接，get_path_binary已经按冒号分割后要求恰好一个相关路径。其[master `118b09b3`](https://github.com/fzyzcjy/torch_memory_saver/blob/118b09b3bc7b0cf386745db1eac2601d4aa7e681/torch_memory_saver/hooks/mode_preload.py)还将已映射libcudart目录置入子进程库路径。这里只读两个固定版本，不推断0.0.9.post1所有行为，也不以一个依赖的改善判整个#468冗余；driver前后顺序、当前wheel路径、HTTP／native接线仍是不同问题。

### 本机只验证环境字典契约

[复查脚本](../../experiments/unirl-scheduler-environment/verify.py)执行候选完整原_split_preloads与_scheduler_spawn_environment，os.environ替换为**独立普通dict**，目录提供者给定夹具，没有改真实进程环境、加载.so或启动scheduler。混合空白／冒号解析返回两个预期路径；无driver时库路径从`/legacy:/active/lib`变为`/active/lib:/legacy`、可见性暂为`3,5`，主动抛异常后原两键逐值恢复。原两键缺失的正常退出控制也恢复到缺失；原helper在父版／main不存在，没有伪造基线运行。

[JSON](../../experiments/unirl-scheduler-environment/result.json)保存两次一致的结果、三版四完整模块的SHA256和Python3.12.13／Darwin环境；12模块AST／compile通过。目录发现、driver文件检查、memory-saver wrapper／hook、真实os.environ、multiprocessing、Ray、HTTP/native Engine、CUDA/GPU均未执行。这不是作者原环境smoke、完整PR审核或新head GPU验收；作者旧#225集成8×H20报告不迁为本机／独立head结果。

```bash
python -B experiments/unirl-scheduler-environment/verify.py /path/to/UniRL
```

真机缺口补[原M5入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)：先核实际依赖和映射库，再在已有安全、获准部署中决定是否需要该补丁，不自动安装compat、改preload或重启服务。不扩模拟spawn／hook矩阵，不改专项、不新增训练或上游写入。

## 下一步

继续以 UniRL 为高优源码/潜在贡献轨道，优先围绕真实配置与类型契约选题；不是已找到 bug 或获得对外发布许可。真机验证由 M5 独立选择获准且资源匹配的配方，条件不符时保留未测，不为复现默认 HI3 直接启动大模型或升级环境。
