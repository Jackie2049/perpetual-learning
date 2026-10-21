# M5 真机工作交接

GPU/NPU 的访问、环境与实验由 macbook-M5 Agent 负责。M1 遇到需要真机的事项在此记录，M5 机器可用时按原有任务优先级接续，不中断已有实验。本页是待接续入口，不代表 M5 已接受、服务器已在线或任务已启动。

更新日期：2026-10-21。M1 本轮按分工未尝试连接服务器，硬件与可用性未核验；没有新的连接失败实测。AReaL prompt_mean 整项保持 M5 单一负责，继续以[原交接](../open-source/areal/handoff-m5.md)为准，本页不重新打开或演化该方案。

## 接续与回写

- 接手前同步项目 main，核对该项是否已有负责人/结果；同一事项只维护一条。M5 接手后将状态改为“已接手”，记录选定版本和资源，不由 M1 推定接单。
- 先核实可用硬件、显存、安装依赖和数据条件；保留已有环境与工作区，实验环境隔离。资源不足或条件不匹配时写明“待资源/待资料”及恢复条件，不直接启动长训练或全模型加载。
- 实验结论回写对应主笔记与 diary，并在本页链接结果：保留版本、配置、命令、原始日志/失败、真实执行范围。没有采到的字段写 unknown，不能由依赖 pin 或历史机器记录补齐。
- 对上游发言/提交与公开发布仍按用户原授权边界执行；交接不增加对外操作权限。不将所有 CPU 学习的未测项自动变成必跑任务。

## DeepInfer 硬件适配核对与最小冒烟

状态：M5 已记录第一段 4090 真机验证，见[原专题结果](../notebook/frameworks/deepinfer-civilian-port.md#4090-真机验证2026-10-21m5-服务器-4090-1-实测)及 WORK：基线774ce244下 `_C` 编译/导入成功，完整引擎/离线推理未完成；昇腾侧仍待恢复条件。此处依据M5回写同步，不是M1新连接或重复实测；后续以M5专项接续为准。

目的：接续项目已有[平民版专题](../notebook/frameworks/deepinfer-civilian-port.md)的硬件适配与编译验证，不是在另一台机器另起移植实现。原专题基线 `8b16149741a4935e1dec8cea27853f273b374c5b` 只代表静态研究；其可行性分层不能直接当实际支持矩阵。

2026-10-21 本次有界窗口新增的核对入口（读取时均 open，共同 base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`）：

- [#148 Hopper 支持](https://github.com/deepseek-ai/DeepInfer/pull/148)，head `731508b4282cee87f526a492c6bad8cabcc2f35c`。作者报告 8×H20 的 kernel、KV、EP/CP、eager EP8 和投机解码检查，Engram GIN 尚在验证。这里只核正文与版本，未审完整实现或复跑；不能外推为 4090/A100/任意 Hopper 配置已支持。
- [#152 Ascend 910B RFC](https://github.com/deepseek-ai/DeepInfer/pull/152)，head `7da92dd58fb7c952b7b781d09506b97f52baa881`。这是设计文档，明确不声称 runtime 支持，且关联 #143 的 910C 阶段；不能把内存估算或转换器复用当作全模型可运行证明。910B 也不是 910C 的同一目标。

M1 已有的[发现层 CPU 检查](../notebook/frameworks/deepinfer-initial-survey.md#vendor-registry-与模型运行支持)只覆盖合成目录、身份/分派；不证明目标卡 kernel、量化、通信、graph 或内存可用。这些上游方案已有作者，本项目不竞争实现，本窗口也不是完整查重。

构建/安装身份参考（不另起移植任务）：已有 [#159](https://github.com/deepseek-ai/DeepInfer/pull/159)，2026-10-21读取open，base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`、head `80c73344ce880ec39ddda475decd3c9f488df2d3`，隔离本次wheel并按精确路径强制重装。M1只做[源码/语法与版本排序核对](../notebook/frameworks/deepinfer-initial-survey.md#vendor-wheel-的本次产物与安装身份)，未复跑作者8项安装测试或原生编译；不自动应用补丁。M5在既有获准操作需要安装时，可独立核对子模块提交、工具链、wheel文件/hash、解释器及import路径；产物身份不解决硬件/ABI堵点，构建阶段失败保护也不代表pip失败可整体回滚。

参数参考（同一条目，不新增实验或前置条件）：已有[#184](https://github.com/deepseek-ai/DeepInfer/pull/184)，2026-10-21读取open，head `0e80d7c835ab1f022b1e3949414aebfa5ee1dc14`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。M1仅做[20组原表达式/环境读取观察](../notebook/frameworks/deepinfer-initial-survey.md#lookahead-与-draft-长度的不同入口)：旧lookahead=1不表示Worker/DFlash nextn也为1；改为环境值还改变prefill最低工作量/准入门槛。若M5日后独立选用该候选且已有获准的匹配引擎环境，可固定nextn，分开记录有效CacheConfig.lookahead、Worker/DFlash nextn、最终prefill q及实际draft/接受数量，关注短prompt/剩余chunk/checkpoint尾部低于lookahead的实际处理。不自动应用、重启或下载权重，不强加测试矩阵；条件不符则注明未测，不代表M5已接单或候选已验收。

快照路径参考（不新增任务/前置条件）：已有[#183](https://github.com/deepseek-ai/DeepInfer/pull/183)，2026-10-21读取open，head `1ff0fee2e2445fbf17ecf37701598404bd670117`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。[M1仅12组路径/目录控制](../notebook/frameworks/deepinfer-initial-survey.md#内存快照的输出路径与目录创建)，未调用CUDA/NPU快照。若M5已有获准诊断确需该产物且匹配runtime，独立决定是否选用，先固定提交、cwd、有效根目录/开关和rank，使用独立可写目录核实际文件及异常；没有runtime/空间/权限就记未测，不为路径验证加载模型、打断任务或强制开启采集，不向仓库上传原始快照。默认路径迁到/tmp/tracing，空串为相对tracing，固定文件名并非任务隔离；此项不代表已接单/采集成功。

抢占状态参考（同一条目，不另排队/改移植方案）：[#188](https://github.com/deepseek-ai/DeepInfer/pull/188)，2026-10-21读取open，head `c0ce645b1a5166e835c1e59fb389c334d1baa4db`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。M1仅[源码与测试语法核对](../notebook/frameworks/deepinfer-initial-survey.md#抢占重算与-prompt-logprob-的身份)，作者两项原生plain/chunked测试也未报告运行；Gloo并不去除prepare_sampler_inputs中的pinned/CUDA或PrivateUse1拷贝。若M5已有匹配此head的_C和获准设备runtime，可独立决定在隔离进程跑tests/engine/test_prompt_logprobs_recompute.py，记录构建/import身份、首次prompt计数与重算selected indices/flag/空输出、EOS及块回收。原fixture注入logprob/采样值，不需权重；通过也不是模型概率/性能或ZOS/CP验收。若需新编译/切环境、runtime不匹配或影响现有任务，先记待条件，不为本参考强制重编、加载模型或在真实服务耗尽KV。未启动或接单，既有M5优先级不变。

M5 最小下一步：先按实际硬件查已有方案和精确 head，读清依赖、权重格式、显存及其原始测试入口；核对通过后只选一个可隔离、可停止的小范围兼容检查，再决定是否具备 offline inference 条件。保留初始失败点与配置；不要直接按旧专题的“换三个包/放宽检查”当成已验证操作，也不要先覆盖已有环境。

停止条件：没有匹配硬件、内存/数据预算不成立、测试入口不明确或需扩大范围时，记录原因和下一项缺口，不自动下载巨量权重、改写移植方案或进行长时间编译。完成标准是目标配置下可复现的实测记录或明确阻塞，不是必须产出性能数字。

## DeepInfer 显存遥测的轻量核对

状态：待 M5 按可用资源安排；独立小范围检查，不是上面移植/模型实验的前置条件，也不代表已接单。

来源：[遥测笔记](../notebook/frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划)、已有 [#151](https://github.com/deepseek-ai/DeepInfer/pull/151)，2026-10-21读取时open，base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`、head `d72f3c4a7d1cfc1bd337adb03dd1319973b084d4`。只改单次查询/warning，作者也未做GPU runtime验证。M1仅核源码/AST，未执行设备API或替身测试；不预设旧查询在稳定total时必错，或此项能修复KV规划/OOM。

最小下一步：M5先核对目标设备、可见编号与实际torch/backend版本，在独立进程/既有可用环境中核对正常mem_get_info及helper返回和日志。若隔离导入helper，须记录方式并保持原函数体；不为验证日志加载权重或编译整套DeepInfer。保留free/total对应的采样身份与时序，不用后一次查询的结果强求前一次值相等；NPU重映射语义另行标明，不能从CUDA结果外推。

停止条件：无合适runtime或需打断现有任务就记待资源，不诱发驱动故障、不强制错误device或修改训练来触发告警，不要求必须补真实异常分支。回写实际命令/版本/日志与未测路径，不把单次正常读取称为原子全局快照、完整日志集成或性能验证。

## verl 量化层审计新版复验

状态：待 M5 核对匹配硬件/软件栈，本项目尚无下述新版路线的真机复验；#8306作者另有Blackwell观测，不是M5结果。该项不代表已接单，也不作为其他任务的前置条件。

来源：[源码与24项CPU单测](../notebook/frameworks/verl-quant-layer-audit.md)、已有 [#8132](https://github.com/verl-project/verl/pull/8132)，2026-10-21读取时open，head `0defd9a231cb876ce56812611b0afb3b2b25831a`，base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。第7部分依赖前序量化堆叠且按#8287重做；不能盲目只cherry-pick审计。作者旧B200/ModelOpt观察不是新版验收；M1只运行作者自带替身的CPU单元测试，没有真实TE/compressor/engine。

旧#8132审计路线的最小下一步：先核实际设备是否支持目标TE FP8路径、依赖/模型/格式是否匹配及是否已有作者新结果；固定完整提交与安装版本，不覆盖已有环境或打断工作。条件齐备后，使用获准的最小模型/配置，真实FP8 forward后做一次权重导出/同步，保留训练workspace/marker、导出格式与引擎方法的证据、实际route、checked/skipped/unresolved及no-verdict状态。非colocated路径不能声称完成同样的live-vLLM复查，SGLang guard另列执行范围。

SGLang 旧#8132路径接续（同一任务，不另排队）：见[分桶与失败边界](../notebook/frameworks/verl-quant-layer-audit.md#sglang-分桶加载与失败清理)。若实际选择该路径，先核 native MXFP8 路由/版本，再记录每 bucket 与 finalize 完成点、实际 dense/MoE backend、暂存/回绑及 graph 使用范围；保留 `VERL_MXFP8_REFIT_CHECK`/阈值、probe 层/expert、误差或明确 skipped 原因。恢复 scale 指针不是恢复旧参数，正常 finalize 也不代表所有 probe 执行；不诱发生产故障或在失败后未经状态核验直接续训。旧路线源码已有 dense fallback 存储重绑 TODO，按实际路径注明未测项，不把 MoE 检查外推全部 backend。

新重叠路线核对：[#8306](https://github.com/verl-project/verl/pull/8306)，2026-10-21读取open、wuxibin89，固定head `1beb35554a763d18ed76fcbefaac1b224eaa1eeb`，目标base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`（merge-base `5a8dd519ab32bfd36d2509a137f830ba79e89c54`）。见[覆盖对照](../notebook/frameworks/verl-quant-layer-audit.md#新旧-sglang-refit-路线的覆盖关系)：增加FP4 routed-expert与dense存储copy-back，拒绝custom refit delta_flush，但未接入旧审计/guard/数值probe，不能搬用旧测试结果。M1仅源码/语法核对，没有loader或真机执行；作者报告Blackwell/Qwen3.5-35B-A3B EP8/TP2默认graph观测，无新增配套CPU测试。M5已有4090不等于匹配该验证资源。接手时先重核当前head/作者状态、安装SGLang版本，独立选一条路线并固定config/exporter/loader/marker/backend；不自动拼接、应用补丁或同时开两套实验，不把“恢复live storage”当旧参数回滚。

停止条件：无匹配设备/TE/引擎或模型授权就写待资源/资料，不大规模下载或长训练，不用fake workspace替代真机验收。完成标准是明确版本与路由的可追溯结果或阻塞；选层相符不等于数值相同、回滚安全或#8250根因，不能凭此发布修复/PR。

## verl 异步策略一致性补证

状态：待数据与运行环境核对；不是 GPU 空闲后即可复现原报告的任务。

入口：[当前诊断边界](../open-source/verl/candidate-8250-async-entropy-collapse.md)、[策略来源与最小观测契约](../notebook/frameworks/verl-async-logprob-provenance.md)、[Issue #8250](https://github.com/verl-project/verl/issues/8250)。2026-10-21 API 读取仍 open、0 评论；没有复查全仓库修复覆盖。报告版本 `1876b06d0a3e4e71e06230be10af14492ca8a75b` 外加私有修改，私有数据/模型不可由项目公开材料还原，实际 vLLM 安装版本仍未知。

M1 已撤回确定因果诊断。已有数值、路由与异常控制未运行模型、原生权重同步或真实训练；没有证明 sync/cache、mask、reward 或数值路径是哪一个根因。

可选观测入口（不另排队、不默认采用）：已有[verl #8315](https://github.com/verl-project/verl/pull/8315)，2026-10-21读取open，head `4d6b3eb8606e24c1d70d648e92a1114129ec6c62`、目标base619c397d、merge-basecd58f550，见[阶段tracing边界](../notebook/frameworks/verl-async-logprob-provenance.md#新版阶段-tracing-与证据关联)。M1仅4模块语法/11方法去decorator后AST一致检查，未运行rl-insight或引擎。它记录rank update_weight与actor memory阶段，但no-op也可有区间，sender返回非全体应用证明，未新增显存字节/参数身份校验。真实同步/设备观测由M5在既有任务条件满足时独立决定，非接单记录。

若已有获准的匹配观测环境，可先固定verl/rl-insight实际提交、包路径、版本及metric_log callable分派，核对rank/actor lane与同步step关联、是否实际执行/异常，并保留独立显存与安装证据。缺后端、需要新增数据外发或会改变原报告环境时先记待条件；不为这项参考自动升级环境、部署服务、重跑长训练或上传私有payload，阶段图不能替代下述配对观测。

可选NIXL拓扑参考（同一事项，不另排队）：[#8316](https://github.com/verl-project/verl/pull/8316)，2026-10-21读取open，head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`、目标basef2923cbf/祖先619c397d。[M1原拓扑及纯断言检查](../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑)每版20组：旧R=1长度失败、候选全过，未prepare/注册/连接/传输。只有已有获准观测实际选择NIXL且恰为单rollout checkpoint worker时，M5独立决定是否核对此候选，先固定安装版本、backend、actor/rollout worker数和metadata顺序，再分开记录建组、传输、应用及恢复点；并非一个replica就必为R=1。不为此新建网络/下载模型、强制改backend或重跑训练，缺匹配资源就记未测；没有证据表明#8250使用该backend或由此导致。

版本补充：#8316后续head `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`只新增1项原CPU测试，实现未变。M1隔离运行原测试体：基线失败、新版通过；没有原生导入/pytest收集或NIXL传输验收。无需另开资源任务，后续若选用仍以当时实际head和上述条件为准。

M5 最小下一步：先判断是否有获准使用的对应配置、完整 token/mask、checkpoint/版本身份和实际环境；条件齐备后按既有观测契约采集训练 actor、干净重载、live rollout 的配对观测，明确 raw/processed 概率与采样处理，保留逐 token 残差及重复基线。需要同步证据时区分发送、接收、应用、cache、版本标记及恢复生成；不由单个版本号或均值接近零认证一致。

停止条件：缺私有材料/配对身份就记录“待资料”，不启动原报告的长训练，也不以公共 toy 或另一模型称已复现 #8250。对照有差异只能缩小范围；没有原始观测前，不设计新的修复、不发布旧诊断稿或承诺治愈阈值。完成标准是可追溯的配对证据，或明确缺失材料清单。
