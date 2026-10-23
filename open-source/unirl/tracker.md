# UniRL 贡献追踪

<!-- contribution-summary:start -->
**【仓库】** [UniRL](https://github.com/Tencent-Hunyuan/UniRL)（腾讯统一多模态 RL）

**【贡献点】** UniRL 贡献追踪。[候选目录](readme.md) · [候选 #523](candidate-523-anchor-lifecycle.md)

**【类型】** 候选追踪

**【价值评估】** 7 / 10 — 甄别"未认领、无 PR、可落地"的开放空间，避免重复投入已有人认领的 RFC。

**【进展】** #523 的我方 fork PR 已获用户认可，测试与 e2e 正式指派 M5；M5 已闭环 L0（四项断言 + pristine 对照），L1 仍待 ≥8 卡及获准权重/存储，见[M5交接](../../experiments/handoff-m5.md)。#567 待匹配 H20；#470/#481/#484已有作者或验证条件。2026-10-24 续核 #515 的[维护者澄清](#补充尽调与派发边界)及 #450/#537 的[已合并分项](#开放-rfc-的已合并分项)，不把开放 RFC 当无人实现，也不将分项合并当全部完成。

**【大白话讲明白】**

这张表区分尚有贡献空间、已有负责人、只覆盖部分需求和缺验证条件的问题。训练栈一致性修复（#523）已经交给 M5 单一推进，不是 M1 的待认领任务；其他线索也要读正文、评论及对应补丁，不能只看 assignee 是否为空。

**【What–Why–How】**

- **What：** UniRL 候选从扫描、验证到认领/放弃的状态。
- **Why：** 社区响应快、RFC 多被认领，本地验证成功不代表仍有贡献空间。
- **How：** 按证据更新状态，将已认领、需 GPU、已上路线和待用户决定分别记录。
<!-- contribution-summary:end -->

> 扫描时间 2026-10-22（GitHub 实时 API）。main 固定 `6fd82cba`。状态以当日核验为准，继续前重新核对关联 issue/PR。

## 本次扫描（2026-10-22）核心结论

UniRL 仓库当前 **73+ open issues**，其中大量是带 `help wanted` 标签的 RFC/贡献机会，且外部 PR 合入率高（最近 100 关闭 PR 合并率 94%）。多数热门 RFC 已被社区认领（维护者会 assign 给 contributors）。**真正"未认领、无对应 PR、可落地"的开放空间**是本次要甄别的。

| 候选 | 类型 | 状态（2026-10-22） | 可做性 | 备注 |
|---|---|---|---|---|
| [#523](https://github.com/Tencent-Hunyuan/UniRL/issues/523) unified stack 跳过 anchor freeze/prepare_part | 跨栈契约 bug | **OPEN · 无 assignee · 无 PR** | **用户已认可方案与实现**（2026-10-23；fork 内 [Jackie2049/UniRL#1](https://github.com/Jackie2049/UniRL/pull/1)：fix-523-anchor-lifecycle @ 04101ad5，base fork main 6fd82cba，unified.py +8/-2；CPU 单测 3 项 + AST 断言通过）；**M5 已闭环 L0 + 判定 L1 待资源/待权重**（[handoff 专设小节](../../experiments/handoff-m5.md)；L0 四项断言全过，L1 需 ≥8 卡 + 获准权重）；上游 PR 待用户授权后再开（铁律）。见 [candidate-523](candidate-523-anchor-lifecycle.md) |
| [#567](https://github.com/Tencent-Hunyuan/UniRL/issues/567) cosmos3 H20 cuBLAS workaround 被错版本 gate | 配置/验证型 | **OPEN · 无 assignee** | **blocked-on-H20**（2026-10-23） | reporter 明确两个分支（删死代码 vs 放宽 gate）都要在 H20+pinned torch 2.13/cu130 实机对照（loss + cublasGemmEx 探针）才能定；我们与 M5（4090）均无 H20，写 PR 即投机，整项挂起不推进 |
| [#581](https://github.com/Tencent-Hunyuan/UniRL/issues/581) FSDP2 0-token micro collective mismatch | bug | **OPEN · 无 assignee** | 可做但已上 #580 路线 | reporter 给了完整 CPU repro；issue 说明 #580 phase 3 会修，宜与 #580 owner 沟通或作为其一部分 |
| [#419](https://github.com/Tencent-Hunyuan/UniRL/issues/419) monitor-only rewards | enhancement/help wanted | **已 assign GoDiao** | 已认领 | 非我方空间 |
| [#423](https://github.com/Tencent-Hunyuan/UniRL/issues/423) 复合奖励能力预检 | enhancement/help wanted | **已 assign GBX-Max1220，有 PR #621** | 已认领 | 非我方空间 |
| [#453](https://github.com/Tencent-Hunyuan/UniRL/issues/453) 主仓库文档社区刷新 | documentation/help wanted | **已 assign 6 人，WIP** | 已认领 | 非我方空间 |
| [#520](https://github.com/Tencent-Hunyuan/UniRL/issues/520) batched chat-template tokenization bug | bug | **已 assign ruiling-smartbear，关联 PR #544** | 已认领 | 非我方空间 |

## 补充尽调与派发边界

2026-10-23只读核对#458/#529正文、全部评论及开放PR清单；关键词affinity/sibling/control/overlay未发现标题明确覆盖的开放PR，**不等于确认没有重叠工作**。编号全文检索还会命中不相关PR，须回到具体需求、评论与源码。固定源码为`6fd82cba13f52f394747832344039408699ea378`，未修改上游。

| 线索 | 可核实的现状 | 本项目处理 |
|---|---|---|
| [#529 sibling affinity](https://github.com/Tencent-Hunyuan/UniRL/issues/529) | 无assignee；MciG-ggg已[公开表示准备实现](https://github.com/Tencent-Hunyuan/UniRL/issues/529#issuecomment-6340056553)初版RolloutManager group_packing，尚非维护者正式指派 | 暂不竞争；接续前先核贡献者进展和实际路径，不据空assignee认领 |
| [#458 Part.control](https://github.com/Tencent-Hunyuan/UniRL/issues/458) | 无assignee、0评论；问题正文明确是设计债而非#263用户回归 | 设计方向待确认，不创建通用deep-merge补丁或声称可发布；须先确定每hop所有权、显式继承/替换/清除及兼容范围 |
| [#473 vLLM/Omni 补丁清理](https://github.com/Tencent-Hunyuan/UniRL/issues/473) | 2026-10-23重核仍OPEN/无assignee，但维护者已[明确分工](https://github.com/Tencent-Hunyuan/UniRL/issues/473#issuecomment-5945617113)：幂等与DiT委派、同步Omni透传归MinhaoLi0318；HI3版本gate与MoE workspace key留维护者 | 不把空assignee当空缺，不升级环境/删shim或另开竞争PR；贡献者[10-03回报](https://github.com/Tencent-Hunyuan/UniRL/issues/473#issuecomment-5966804614)的stub身份检查不是当前真实0.28组合验收 |
| [#515 冻结教师读取失败](https://github.com/Tencent-Hunyuan/UniRL/issues/515) | 2026-10-24核正文/全部3评论，已assign inaniloquentee；维护者[澄清](https://github.com/Tencent-Hunyuan/UniRL/issues/515#issuecomment-6138540697)当前启动链已有fail-fast。作者[#575](https://github.com/Tencent-Hunyuan/UniRL/pull/575)于10-13合并，实际收窄为各rank教师权重hash一致性，不是为原报告新增错误共识 | 不据原帖另开修复；异常返回、内容分歧和读取不返回是不同问题，后者不因#575合并而解决。见[当前源码范围](../../notebook/frameworks/unirl-survey.md#教师指纹与控制器失败传播)，本机仅纯hash函数CPU控制，不是Ray/Gloo或教师训练验收，不新增M5队列 |

#529的派发粒度不能一概而论：[AsyncARTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/async_ar.py)沿ARTrainer构造fork后的完整请求，再由async循环按`Sample.split()`拆成root树提交；[Sample.split](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py)保留该root的所有children。相反，[AgenticTrainer._collect_groups](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/agentic.py)将每个prompt重复K次成为独立任务。pool按任务计一个slot，并不在内部拆K个row。因此“普通异步AR的K个siblings必被pool散到K个worker”不能直接由most-free策略推出；同worker也不保证KV已就绪、命中100%或吞吐提高。同步DP_SCATTER、engine内部执行及session续跑须另核，见[原单位说明](../../notebook/frameworks/unirl-survey.md#队列深度不是一种单位)。

#458同样不能只凭浅合并就认定bug：[prepare_input_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/base.py)顶层覆盖，而PE显式传空ar/chat袋以擦除父输入控制；无条件deep merge会保留本来要清掉的字段。单个字典算法无法替代跨trainer/engine的所有权设计。本轮只做源码和社区状态尽调，无执行、性能/缓存验证、认领或对外操作，不为这两个线索排M5实验。

#473本轮读取正文与全部4条评论；上表只判贡献归属，没有重跑报告者19-check真实包harness或SD3回归。贡献者评论提到#413/#474已覆盖部分清理及上游vllm-omni#8196待复审，属于当时报告；本轮未重核这些PR的当前状态，不据此自动退休passthrough或移除仍有独立用途的stage-0 gate，也不新增M5必跑矩阵。另读已有[#545](https://github.com/Tencent-Hunyuan/UniRL/pull/545)做[压缩积分学习](../../notebook/frameworks/unirl-survey.md#sde-窗口后的压缩积分尾段)，作者已有实现，CPU控制不转为我方贡献认领。

## RFC 归属与验证条件

2026-10-23 续核下面三条开放 RFC 的完整正文、全部评论、编号引用的开放 PR 检索结果及相关 PR 正文。**部分实现、负责人归属和验证许可是不同结论。** 本表仅用于选题，不认证补丁正确性或作者的实验数字。

| 线索 | 已核实的社区状态 | 本项目处理 |
| --- | --- | --- |
| [#470 多参考图像与 DyRef](https://github.com/Tencent-Hunyuan/UniRL/issues/470) | 无 assignee、0 评论，但报告者 Ideny42 明确计划实现，并已有同作者开放非 draft [#478](https://github.com/Tencent-Hunyuan/UniRL/pull/478)，head `33756c23ec20da7c338602c07a0dce7c1738f187`；正文只覆盖 DyRef 数据、Qwen-Image-Edit-2511 和 FLUX.2 Klein 训练路径，不关闭 HI3/BAGEL/Qwen3-Omni/Janus-Pro 项。依赖 [#417](https://github.com/Tencent-Hunyuan/UniRL/pull/417) 仍 open、未合并 | 不把空 assignee 当无人做，也不把 #478 写成整个 RFC 已完成；其余模型是原作者声明的后续范围，不能直接认作我方空缺 |
| [#481 agentic 内存与上下文预算](https://github.com/Tencent-Hunyuan/UniRL/issues/481) | assignees 为 celve、Dayuxiaoshui；CjhHa1 [评论](https://github.com/Tencent-Hunyuan/UniRL/issues/481#issuecomment-5713801099)明确支持拆分 recipe、缺 flag 保护、context budget 和文档，并要求 recipe 修好后重测工作集/KV 池比值 R，再决定是否做 HiCache/Mooncake。编号引用的开放 PR 检索无结果，不推导尚无实现 | 已派发，不接手缺 flag 或确定性 4xx 的小补丁；单 H20 的报告不能变成本机测量或通用 0.88 内存配置，也不启动 host-tier 集成 |
| [#484 离线模拟接入](https://github.com/Tencent-Hunyuan/UniRL/issues/484) | 0z5a 已有开放非 draft [#487](https://github.com/Tencent-Hunyuan/UniRL/pull/487)，head 仍 `e8779559d2ceb441c062f603f90e7fd31370e69b`。维护者早期在 [RFC](https://github.com/Tencent-Hunyuan/UniRL/issues/484#issuecomment-5757256238)及 [PR](https://github.com/Tencent-Hunyuan/UniRL/pull/487#issuecomment-5757256603)要求匹配 H20 校准数据；PR 后续正文由作者称已获准用 H100 验证，本轮读到的两条 PR 评论未包含该许可原文 | 保留两者的时间和来源，既不宣称永久只准 H20，也不认证 H100 许可。匹配的真实 capture/后端版本/profile 与硬件对比仍未完成，作者的 CPU profile smoke 不等于保真度验收；不抢做 runner 或另排硬件任务 |

#487 仍是 schema/converter，不含作者 fork draft 的 guard 或 runner；已有[本机契约验证](../../notebook/frameworks/unirl-survey.md#离线回放的到达契约与结果隔离)不重复运行。本轮没有执行模型、评分器、模拟器或设备测试，没有修改上游或发布评论；三个题目都不进入 M1 实现队列。只有负责人或范围发生实质变化，或收到明确协作需求，才重新判断贡献空间。

## 开放 RFC 的已合并分项

2026-10-24 读取 [#450](https://github.com/Tencent-Hunyuan/UniRL/issues/450) 全文/全部3评论与 [#537](https://github.com/Tencent-Hunyuan/UniRL/issues/537) 全文（0评论），再按编号检索并读下表PR完整正文/API。两个RFC仍OPEN、无assignee，但这些分项已经 `merged=true`，不是仅有关闭状态。

| RFC 分项 | 合并事实与作者声明范围 | 本项目处理 |
| --- | --- | --- |
| #450 H3离线文本embedding | 同报告者 BeatSeat 的 [#513](https://github.com/Tencent-Hunyuan/UniRL/pull/513)，2026-09-26合并 `c01c4e92`；H3专用safetensors分片、独立CLI、`text_embed_cache_path` 跳过32B加载、carried condition优先及未padding查表 | 不另写pipeline bypass、skip-TE或同一缓存工具。正文限定H3和当前单样本forward，不覆盖其他encoder/VAE/reward；合并不证明32B跳过加载、数值parity及训练在本项目已验收 |
| #537 module3 布局与首更新告警 | 同报告者 nussejzz 的 [#548](https://github.com/Tencent-Hunyuan/UniRL/pull/548)，2026-10-14合并 `0fbf3ad0`；vLLM-Omni BAGEL重建路径比较布局，`first_update_ratio_tol` 为可选告警 | 布局检查不是数值等价证明；正文说明ratio指标仅DP rank0、各rank各自告警，不称全局聚合；带carried contexts的路径不作该布局比较 |
| #537 module1 每组画布 | 同作者 [#549](https://github.com/Tencent-Hunyuan/UniRL/pull/549)，2026-10-14合并 `2c19e14e`；vLLM-Omni BAGEL图像/FlowGRPO，按行metadata画布分连续run调用，存储padding后replay裁回 | 不是所有模型/算法、视频或ragged segment已支持；正文明确不含跨rank调度，不因能混画布就声称长尾已解决 |
| #537 module4a it2i sibling打包 | 同作者 [#550](https://github.com/Tencent-Hunyuan/UniRL/pull/550)，2026-10-14合并 `f6044285`；两个CFG scale均≤1时按同组连续run打包，micro可切group，x_T按row寻址 | 不重写同一打包；CFG>1、trainside BAGEL dense batching和Qwen-Image prompt去重不在该PR范围。剩余路线也不因此认定无人做或自动接手 |

#537原文把单位表、owner/ring/steal调度列为module2；以上合并事实不能证明该模块完成，也不能证明如今尚无其他实现。#513与#548/#549/#550中的CPU harness、H20训练和性能数字均为作者报告，本轮未复跑、未逐行审查或核当前main运行行为；不把合并状态搬成本机训练/内存收益。这里仅作贡献范围尽调，无候选源码执行、依赖或上游写入，不新增M5硬件任务。只有独立缺口、明确协作或范围实质变化出现时才重判投入。

## 已认领/已覆盖，不重复推进

- #368 RFC on-policy distillation——维护者列出 Open 项（D3/D4/D5、A3…）等认领，D2 已 assign，contribution 门槛高（P0/P1 多为大 feature）。
- #520/#423/#419/#453 见上表，均有 assignee 或对应 PR。
- 大量 open PR 由社区作者（ruiling-smartbear、MinhaoLi0318、nussejzz、leviking98z-rgb 等）挂多个 RFC 功能实现，合入排队积压较重；避免与在途 feature PR 撞车。

## 决策待定

- #523 已有获认可的 fork 实现，由 M5 接续验证与发布判断；M1 不再留言认领、重写方案或并行测试。外部主仓库行动仍须用户授权。
- #567/#581 需 GPU 或需协调 #580，标为"后续/需沟通"，不作为优先。
