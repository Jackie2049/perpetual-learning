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

M5 最小下一步：先按实际硬件查已有方案和精确 head，读清依赖、权重格式、显存及其原始测试入口；核对通过后只选一个可隔离、可停止的小范围兼容检查，再决定是否具备 offline inference 条件。保留初始失败点与配置；不要直接按旧专题的“换三个包/放宽检查”当成已验证操作，也不要先覆盖已有环境。

停止条件：没有匹配硬件、内存/数据预算不成立、测试入口不明确或需扩大范围时，记录原因和下一项缺口，不自动下载巨量权重、改写移植方案或进行长时间编译。完成标准是目标配置下可复现的实测记录或明确阻塞，不是必须产出性能数字。

## DeepInfer 显存遥测的轻量核对

状态：待 M5 按可用资源安排；独立小范围检查，不是上面移植/模型实验的前置条件，也不代表已接单。

来源：[遥测笔记](../notebook/frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划)、已有 [#151](https://github.com/deepseek-ai/DeepInfer/pull/151)，2026-10-21读取时open，base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`、head `d72f3c4a7d1cfc1bd337adb03dd1319973b084d4`。只改单次查询/warning，作者也未做GPU runtime验证。M1仅核源码/AST，未执行设备API或替身测试；不预设旧查询在稳定total时必错，或此项能修复KV规划/OOM。

最小下一步：M5先核对目标设备、可见编号与实际torch/backend版本，在独立进程/既有可用环境中核对正常mem_get_info及helper返回和日志。若隔离导入helper，须记录方式并保持原函数体；不为验证日志加载权重或编译整套DeepInfer。保留free/total对应的采样身份与时序，不用后一次查询的结果强求前一次值相等；NPU重映射语义另行标明，不能从CUDA结果外推。

停止条件：无合适runtime或需打断现有任务就记待资源，不诱发驱动故障、不强制错误device或修改训练来触发告警，不要求必须补真实异常分支。回写实际命令/版本/日志与未测路径，不把单次正常读取称为原子全局快照、完整日志集成或性能验证。

## verl 量化层审计新版复验

状态：待 M5 核对匹配硬件/软件栈，尚无当前新版真机结果，不代表已接单；不作为其他任务的前置条件。

来源：[源码与24项CPU单测](../notebook/frameworks/verl-quant-layer-audit.md)、已有 [#8132](https://github.com/verl-project/verl/pull/8132)，2026-10-21读取时open，head `0defd9a231cb876ce56812611b0afb3b2b25831a`，base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。第7部分依赖前序量化堆叠且按#8287重做；不能盲目只cherry-pick审计。作者旧B200/ModelOpt观察不是新版验收；M1只运行作者自带替身的CPU单元测试，没有真实TE/compressor/engine。

最小下一步：先核实际设备是否支持目标TE FP8路径、依赖/模型/格式是否匹配及是否已有作者新结果；固定完整提交与安装版本，不覆盖已有环境或打断工作。条件齐备后，使用获准的最小模型/配置，真实FP8 forward后做一次权重导出/同步，保留训练workspace/marker、导出格式与引擎方法的证据、实际route、checked/skipped/unresolved及no-verdict状态。非colocated路径不能声称完成同样的live-vLLM复查，SGLang guard另列执行范围。

SGLang 接续补充（同一任务，不另排队）：见[分桶与失败边界](../notebook/frameworks/verl-quant-layer-audit.md#sglang-分桶加载与失败清理)。若实际选择该后端，先核 native MXFP8 路由/版本，再记录每 bucket 与 finalize 完成点、实际 dense/MoE backend、暂存/回绑及 graph 使用范围；保留 `VERL_MXFP8_REFIT_CHECK`/阈值、probe 层/expert、误差或明确 skipped 原因。恢复 scale 指针不是恢复旧参数，正常 finalize 也不代表所有 probe 执行；不诱发生产故障或在失败后未经状态核验直接续训。源码已有 dense fallback 存储重绑 TODO，按实际路径注明未测项，不把 MoE 检查外推全部 backend。

停止条件：无匹配设备/TE/引擎或模型授权就写待资源/资料，不大规模下载或长训练，不用fake workspace替代真机验收。完成标准是明确版本与路由的可追溯结果或阻塞；选层相符不等于数值相同、回滚安全或#8250根因，不能凭此发布修复/PR。

## verl 异步策略一致性补证

状态：待数据与运行环境核对；不是 GPU 空闲后即可复现原报告的任务。

入口：[当前诊断边界](../open-source/verl/candidate-8250-async-entropy-collapse.md)、[策略来源与最小观测契约](../notebook/frameworks/verl-async-logprob-provenance.md)、[Issue #8250](https://github.com/verl-project/verl/issues/8250)。2026-10-21 API 读取仍 open、0 评论；没有复查全仓库修复覆盖。报告版本 `1876b06d0a3e4e71e06230be10af14492ca8a75b` 外加私有修改，私有数据/模型不可由项目公开材料还原，实际 vLLM 安装版本仍未知。

M1 已撤回确定因果诊断。已有数值、路由与异常控制未运行模型、原生权重同步或真实训练；没有证明 sync/cache、mask、reward 或数值路径是哪一个根因。

M5 最小下一步：先判断是否有获准使用的对应配置、完整 token/mask、checkpoint/版本身份和实际环境；条件齐备后按既有观测契约采集训练 actor、干净重载、live rollout 的配对观测，明确 raw/processed 概率与采样处理，保留逐 token 残差及重复基线。需要同步证据时区分发送、接收、应用、cache、版本标记及恢复生成；不由单个版本号或均值接近零认证一致。

停止条件：缺私有材料/配对身份就记录“待资料”，不启动原报告的长训练，也不以公共 toy 或另一模型称已复现 #8250。对照有差异只能缩小范围；没有原始观测前，不设计新的修复、不发布旧诊断稿或承诺治愈阈值。完成标准是可追溯的配对证据，或明确缺失材料清单。
