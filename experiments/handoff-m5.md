# M5 真机工作交接

GPU/NPU 的访问、环境与实验由 macbook-M5 Agent 负责。M1 遇到需要真机的事项在此记录，M5 机器可用时按原有任务优先级接续，不中断已有实验。本页是待接续入口，不代表 M5 已接受、服务器已在线或任务已启动。

更新日期：2026-10-22。M5 完成 DeepInfer 4090 移植冒烟第二段并回写本页（llm_engine import 链全通、发现 tilelang PDL 硬堵点）；PDL 探针续：5 个 kernel 去 PDL 改写编译 + 数值对照全过（含最复杂 fused_sample）；**四续：load/store_kv_cache 全矩阵数值验证（上游测试 60/60 PASS 于 nosync 变体），发现 sync_grid→cooperative-launch grid 上限、pdl_sync 依赖型应删除而非替换**；**五续：全量 nosync 改写验证——55 处 pdl_sync 全部依赖型分类、43/43 SM89 编译全过、7 个 kernel 数值保持，改写规则收敛为「全部删除，sync_grid 无适用用例」**。昇腾 910C VPN 掉线（100% 丢包）待用户恢复，SSH 密钥仍未配置。AReaL prompt_mean 整项保持 M5 单一负责，继续以[原交接](../open-source/areal/handoff-m5.md)为准，本页不重新打开或演化该方案。

## 接续与回写

- 接手前同步项目 main，核对该项是否已有负责人/结果；同一事项只维护一条。M5 接手后将状态改为“已接手”，记录选定版本和资源，不由 M1 推定接单。
- 先核实可用硬件、显存、安装依赖和数据条件；保留已有环境与工作区，实验环境隔离。资源不足或条件不匹配时写明“待资源/待资料”及恢复条件，不直接启动长训练或全模型加载。
- 实验结论回写对应主笔记与 diary，并在本页链接结果：保留版本、配置、命令、原始日志/失败、真实执行范围。没有采到的字段写 unknown，不能由依赖 pin 或历史机器记录补齐。
- 对上游发言/提交与公开发布仍按用户原授权边界执行；交接不增加对外操作权限。不将所有 CPU 学习的未测项自动变成必跑任务。

## UniRL 联合路径的资源与身份边界

状态：待资源匹配与M5选题，非已接手或即时执行任务；不打断既有GPU/AReaL工作。目的：后续若获准推进UniRL真机，区分driver样本树接线与实际engine行为，避免把模板注释或静态源码当实测。

固定官方main `6fd82cba13f52f394747832344039408699ea378`，[M1源码入口与证据](../notebook/frameworks/unirl-survey.md#ar-与-diffusion-的源码统一边界)。M1仅原静态依赖guard与4文件AST/compile通过，无训练包导入/设备探测。HI3 driver构造P→P×N→P×N×M，但同版配方旧注释仍称平坦对；没有用CPU替身替代真实engine验收。

条件：该HI3双引擎构造器要求每节点至少8设备，batch_size为num_devices的正倍数；[固定版安装文档](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/INSTALL.md)为Linux x86_64/Python3.12–3.13/NVIDIA CUDA13，vllm与sglang extra不应混装。权重/显存/engine兼容与当前资源均未核验。不能因已有单4090工作或安装文档未限架构，就默认此HI3配方可跑；也不外推所有UniRL至少8卡。

最小接续：仅当M5后续选定该路径且已有获准匹配资源/权重时，先确认实际recipe与版本，不直接照默认大模型加载；一次有界rollout记录三层ID/行数、每个AR文本实际对应的M份图像条件及同步版本，再决定是否值得训练步验证。数量正确不代替身份或数值正确。资源/许可/数据不匹配则记录未测并停止，不装环境、下载权重或启动长训练；无需为本条新租算力。本条不授权上游修改/评论/PR，也不代表M5已接单。

稀疏SDE参考（同版，不新增排队/前置条件）：[M1原配置/调度模块检查](../notebook/frameworks/unirl-survey.md#稀疏-sde-与轨迹状态位置)覆盖64个rollout id及56种前半段三索引组合；T16/S3源码仍遍历16去噪迭代，端点加终态预期5～7位置不是实测显存。若M5后续本来就要采实际rollout，可记录有效sde_indices、完整sigma长度、LatentSegment.indices与sde_logp形状，确认engine/训练replay字段对应；不以配置数量代替实际logprob可用性或节省比例，不要求为本参考额外跑模型/矩阵。无匹配资源或不在原选题范围则保留未测。

## DeepInfer 硬件适配核对与最小冒烟

状态：M5 已完成第二段 4090 真机验证（2026-10-22），见[专题结果](../notebook/frameworks/deepinfer-civilian-port.md#移植冒烟第二段2026-10-22m5-服务器-4090-1-实测)与[实验材料](deepinfer-civilian-port-stage2/readme.md)：基线774ce244下四 kernel 包 stub + 架构检查放宽后 **llm_engine 等 20 个核心模块 import 全通**；tilelang 0.1.15 支持 SM89（saxpy/sync_grid 编译运行 OK）但 **43 个 cuda tilelang kernel 全部显式使用 PDL（Hopper+ 专属），SM89 编译抛 PDL is not supported——需逐个去 PDL 改写**；LLMEngine 实例化停在数据前提（需 V4 架构 config/权重）。**2026-10-22 续（M5）**：PDL 堵点源码级定位——根因在 tilelang 编译管线 `MarkCudaSyncCalls(have_pdl(target))`（`contrib/nvcc.py` 仅 SM90+ 返回 True，`lower_pdl.cc:44` 在 SM89 遇 `pdl_sync`/`pdl_trigger` 即 `ICHECK` 抛错，无配置可关）；语义映射确认 `pdl_trigger` 可安全删除、`pdl_sync` 用 `T.sync_grid()` 替代（顺序 launch 语义）；**5 个真实 kernel 文本级去 PDL 后 SM89 编译全过，其中 norm 与 torch RMSNorm 数值对照 max err=0.0（PASS）**——「去 PDL 改写」是机械工程（28/30 源文件、56 处 pdl_sync+8 处 pdl_trigger），代价是失去 kernel 间 PDL 重叠（性能未测）。**2026-10-22 再续（M5）**：`probe_pdlfree_numeric.py` 给其余 4 个 kernel 补数值对照（[实验材料](deepinfer-civilian-port-stage2/readme.md#数值对照扩展2026-10-22-再续m5)），**5/5 变体数值全过**——kv_index_cat 拷贝+偏移加 eq、load_kv_index 索引展开 eq、dflash_input_store hc_mult=1/2 max err=0.0、fused_sample（最复杂：两级 launch + lse_workspace 跨 launch 共享）logprobs vs log_softmax max err=1e-6、lse0/lse1 err=0.0、candidate 合法。改写语义保持从 norm 扩展到含最复杂 kernel 的 5 个。**2026-10-22 三续（M5）**：`scan_pdlfree_all.py` 全量改写编译扫查——28/28 个 PDL 文件、**43/43 个 `_tl` 工厂全部文本级去 PDL 变体在 SM89 编译通过**（含 fused_transform 全 kwonly 参数、load/store_kv_cache 的 KVCacheFormat 常量、state_cache 条件启用分支 enable 参数；替换计数核对 51 处 sync_grid 与改写前 pdl_sync 调用逐一相等、8 处 pdl_trigger→pass 均为独立语句、变体无 PDL 调用残留）。「去 PDL 改写」工程量与可机械性在真实硬件全量实证，PDL 非 4090 移植编译层隐藏障碍。**2026-10-22 四续（M5）**：load/store_kv_cache 两个每请求必经 kernel 用上游自带 GPU 测试矩阵数值对照——**pdlfree（pdl_sync→sync_grid）小用例 20/20 PASS、67890 大用例 10/10 FAIL（CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE）**：tilelang `sync_grid()` 生成 `cooperative_groups::this_grid().sync()`，**要求 cooperative launch（grid 全部 block 同时驻留），4090（128 SM）grid 上限约 1024-4096，大输入启动失败；上游 PDL 无此驻留限制**——sync_grid 替代引入运行期新硬边界（编译扫查小 grid 未暴露）。**nosync 变体（pdl_sync 直接删除）全矩阵 60/60 PASS（含全部大用例）**：两个 kernel 的 pdl_sync 是「等待 dependency launch 前序数据」，顺序 launch 下由 stream 顺序天然保证，**正确改写是删除而非替换 sync_grid**（详见 [stage2 readme](deepinfer-civilian-port-stage2/readme.md#loadstore_kv_cache-全矩阵数值验证--sync_grid-grid-上限发现2026-10-22-四续m5)）。**2026-10-22 五续（M5）**：把「删除而非替换」推广到全部 43 个 kernel——静态语义分类（28 个 PDL 源文件逐处核对）确认**全部 55 处 pdl_sync 均为依赖 launch 前序数据型（位于 kernel 开头），无一处是同一 launch 内跨 block 通信（那才需要 sync_grid）**，8 处 pdl_trigger 均为提前释放下游；`gen_nosync_variants.py` 生成全量 nosync 变体（删除 pdl_sync、pdl_trigger→pass），`scan_nosync_all.py` **43/43 SM89 编译全过**（重跑确认可复现），`probe_nosync_numeric.py`/`probe_norm_nosync.py` **7 个 kernel nosync 数值保持**（norm max err=0.0、kv_index_cat/load_kv_index eq、dflash_input_store max err=0.0、fused_sample logprobs err=1e-6、load/store 全矩阵 60/60）。**改写规则收敛：全部 pdl_sync 删除（nosync）、pdl_trigger 删除；sync_grid 在本项目 43 个 kernel 中无适用用例（且受 cooperative-launch grid 上限约束）——三续的 51 处 sync_grid 替换为次优改写**（详见 [stage2 readme](deepinfer-civilian-port-stage2/readme.md#全量-nosync-改写验证4343-编译--7-kernel-数值保持2026-10-22-五续m5)）。昇腾侧 VPN 2026-10-22 掉线（100% 丢包）待用户恢复，SSH 密钥亦未配置。此处依据M5回写同步，不是M1新连接或重复实测；后续以M5专项接续为准。

目的：接续项目已有[平民版专题](../notebook/frameworks/deepinfer-civilian-port.md)的硬件适配与编译验证，不是在另一台机器另起移植实现。原专题基线 `8b16149741a4935e1dec8cea27853f273b374c5b` 只代表静态研究；其可行性分层不能直接当实际支持矩阵。

2026-10-21 本次有界窗口新增的核对入口（读取时均 open，共同 base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`）：

- [#148 Hopper 支持](https://github.com/deepseek-ai/DeepInfer/pull/148)，head `731508b4282cee87f526a492c6bad8cabcc2f35c`。作者报告 8×H20 的 kernel、KV、EP/CP、eager EP8 和投机解码检查，Engram GIN 尚在验证。这里只核正文与版本，未审完整实现或复跑；不能外推为 4090/A100/任意 Hopper 配置已支持。
- [#152 Ascend 910B RFC](https://github.com/deepseek-ai/DeepInfer/pull/152)，head `7da92dd58fb7c952b7b781d09506b97f52baa881`。这是设计文档，明确不声称 runtime 支持，且关联 #143 的 910C 阶段；不能把内存估算或转换器复用当作全模型可运行证明。910B 也不是 910C 的同一目标。

M1 已有的[发现层 CPU 检查](../notebook/frameworks/deepinfer-initial-survey.md#vendor-registry-与模型运行支持)只覆盖合成目录、身份/分派；不证明目标卡 kernel、量化、通信、graph 或内存可用。这些上游方案已有作者，本项目不竞争实现，本窗口也不是完整查重。

2026-10-22 Sage新模型线索（只补参考，不另排队/改V4.1方案）：[#210](https://github.com/deepseek-ai/DeepInfer/pull/210)，读取open，head `2a421f129809852b7b001c1e3b3f1b4f04f06156`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，见[M1装载阅读](../notebook/frameworks/deepinfer-initial-survey.md#sage-fp8-分层装载与独立引擎)。SageLLM绕过V4 Worker/paged KV/MTP，以主机FP8专家池+GPU LRU执行单序列文本；作者5090 D 32GB/SM120观测不是4090/SM89支持。M1仅五模块compile/标准库config与预算（默认专家池30.0586GiB主机、14GiB理论4769槽），无torch/设备执行。最小下一步是先确认研究目标是否包含Sage；若获准选择，先核本地权重revision/授权、主机RAM/磁盘、目标卡runtime及真实compute模式，再由M5独立决定小型probe/固定短prompt与FP8/BF16-dequant诊断。缺条件或仍只做V4.1就停止本线，不自动下载约39GB权重、切环境/编译/打断任务；保留实际cap、冷/热缓存差值和精度基准口径，不把BF16 fallback、注册名或单次生成视为V4移植/质量验收。此记录不代表接单或启动。

2026-10-22 新方案入口（仅补参考，不另排队）：[#192 A2 eager text runner](https://github.com/deepseek-ai/DeepInfer/pull/192)，读取 open、作者 Liuchenbing-2026，head `19ba026deca68936ef18b9e370ad6fd812ed0088`，base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。作者报告在910B4-1上用真实W4A8权重完成40层prefill/decode及一个算术样例/EOS；正文明确是单设备实验路径，不接生产scheduler/graph/distributed，关闭vision/MTP，INT4解包为INT8 GEMM，非模型质量/长上下文/吞吐证明。M1仅核API正文及版本，未审16文件实现、复跑CPU/设备测试或认证镜像可用性；也不能外推910C兼容。由M5在原适配任务中独立决定是否值得核对，不自动采用、下载权重、切环境或打断既有任务；已有最小下一步/停止条件不变。

构建/安装身份参考（不另起移植任务）：已有 [#159](https://github.com/deepseek-ai/DeepInfer/pull/159)，2026-10-21读取open，base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`、head `80c73344ce880ec39ddda475decd3c9f488df2d3`，隔离本次wheel并按精确路径强制重装。M1只做[源码/语法与版本排序核对](../notebook/frameworks/deepinfer-initial-survey.md#vendor-wheel-的本次产物与安装身份)，未复跑作者8项安装测试或原生编译；不自动应用补丁。M5在既有获准操作需要安装时，可独立核对子模块提交、工具链、wheel文件/hash、解释器及import路径；产物身份不解决硬件/ABI堵点，构建阶段失败保护也不代表pip失败可整体回滚。

参数参考（同一条目，不新增实验或前置条件）：已有[#184](https://github.com/deepseek-ai/DeepInfer/pull/184)，2026-10-21读取open，head `0e80d7c835ab1f022b1e3949414aebfa5ee1dc14`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。M1仅做[20组原表达式/环境读取观察](../notebook/frameworks/deepinfer-initial-survey.md#lookahead-与-draft-长度的不同入口)：旧lookahead=1不表示Worker/DFlash nextn也为1；改为环境值还改变prefill最低工作量/准入门槛。若M5日后独立选用该候选且已有获准的匹配引擎环境，可固定nextn，分开记录有效CacheConfig.lookahead、Worker/DFlash nextn、最终prefill q及实际draft/接受数量，关注短prompt/剩余chunk/checkpoint尾部低于lookahead的实际处理。不自动应用、重启或下载权重，不强加测试矩阵；条件不符则注明未测，不代表M5已接单或候选已验收。

快照路径参考（不新增任务/前置条件）：已有[#183](https://github.com/deepseek-ai/DeepInfer/pull/183)，2026-10-21读取open，head `1ff0fee2e2445fbf17ecf37701598404bd670117`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。[M1仅12组路径/目录控制](../notebook/frameworks/deepinfer-initial-survey.md#内存快照的输出路径与目录创建)，未调用CUDA/NPU快照。若M5已有获准诊断确需该产物且匹配runtime，独立决定是否选用，先固定提交、cwd、有效根目录/开关和rank，使用独立可写目录核实际文件及异常；没有runtime/空间/权限就记未测，不为路径验证加载模型、打断任务或强制开启采集，不向仓库上传原始快照。默认路径迁到/tmp/tracing，空串为相对tracing，固定文件名并非任务隔离；此项不代表已接单/采集成功。

抢占状态参考（同一条目，不另排队/改移植方案）：[#188](https://github.com/deepseek-ai/DeepInfer/pull/188)，2026-10-21读取open，head `c0ce645b1a5166e835c1e59fb389c334d1baa4db`、base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。M1仅[源码与测试语法核对](../notebook/frameworks/deepinfer-initial-survey.md#抢占重算与-prompt-logprob-的身份)，作者两项原生plain/chunked测试也未报告运行；Gloo并不去除prepare_sampler_inputs中的pinned/CUDA或PrivateUse1拷贝。若M5已有匹配此head的_C和获准设备runtime，可独立决定在隔离进程跑tests/engine/test_prompt_logprobs_recompute.py，记录构建/import身份、首次prompt计数与重算selected indices/flag/空输出、EOS及块回收。原fixture注入logprob/采样值，不需权重；通过也不是模型概率/性能或ZOS/CP验收。若需新编译/切环境、runtime不匹配或影响现有任务，先记待条件，不为本参考强制重编、加载模型或在真实服务耗尽KV。未启动或接单，既有M5优先级不变。

PD 原生接口参考（2026-10-22，不另排队/采用方案）：新增 [#195](https://github.com/deepseek-ai/DeepInfer/pull/195)，head `229c58c326f3f3ba98959240f84c1c88029baa96`、base/祖先 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，仅3文件21行新增。目的为今后评估PD入口时核对队列与资源所有权；[M1源码结论](../notebook/frameworks/deepinfer-initial-survey.md#pd-队列交接与资源释放)：移出finished只归还MTP槽，KV/state留待显式free；入队三项元数据检查不认证传输完成，非forked项不克隆。M1仅完整stub AST/内存compile与声明检查，无_C/原生测试，本PR无新增测试文件；作者GB300/B200报告不是本机证据。若M5今后确需采用且已有匹配head的_C/runtime，可先核实际binding/import身份，再以最小原生序列记录队列、MTP槽、KV/state占用及导出完成后显式释放的顺序，不把计数合法当数据已到达。缺口是匹配build、设备及实际拷贝完成证据；若需重编/切环境、传输同步协议不明确、资源不足或影响原任务，先记录待条件，不自动编译、下载权重、启动#197或并行改方案。原优先级与停止条件不变，记录不代表M5已接单。

Host cache 参考（2026-10-22，沿原条目、不另排队）：已有[#144](https://github.com/deepseek-ai/DeepInfer/pull/144)，head `e1f71ecdf8f1e53224bf44eee6989dee9de5f28a`、base/祖先 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。[M1完整源码核对](../notebook/frameworks/deepinfer-initial-survey.md#host-cache-的可复用边界)：KV命中还要与ready state/replay上限相交，size_bytes只管KV页，另计pinned state池及默认16份GPU staging。目的为今后评估缓存时核对预算/恢复证据，不改变移植方案；M1仅4文件AST/内存compile，未跑原3项GPU测试。若M5独立决定采用且已有兼容CUDA/编译链，可先核固定池和staging预算，再运行原入口 `python -m pytest -q tests/hicache/test_host_cache.py`；它用真实copy kernel但FakeSeq/替换state kernel，不验证新block_table binding或真实SWA/模型恢复。原生接入仍缺匹配head的_C、compact/额外state配置与实际state/replay证据，不能外推4090/NPU兼容。需重编、额外权重/内存预算、新平台移植或影响原任务时，先记待条件，不自动JIT、下载、切环境或启动服务；作者性能报告非本项目结果，记录不代表接单。

聊天封装参考（2026-10-22，非新部署任务）：已有[#147](https://github.com/deepseek-ai/DeepInfer/pull/147)，head `aaf1e62c9eab1ce5bd260a594dfaea613760e89f`、base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，见[M1源码阅读](../notebook/frameworks/deepinfer-initial-survey.md#聊天服务的亲和路由与完成口径)。目的为将来评估服务时区分亲和路由、真实cache命中、前端完成和客户端交付；M1仅8项原纯affinity控制与语法检查，未启动HTTP/engine/设备。缺口是匹配_C/模型/DP+EP运行环境、获准端口及host-cache预算；作者4×GB300结果不能外推4090/NPU。只有M5独立决定研究服务、已有可用获准引擎时，才考虑最小文本请求，保留固定提交、rank、编码fallback、engine reason/响应reason、usage和客户端完成证据；原A/B是先bypass后cache、TTFT按首个非空content，不当首次冷缓存或完整协议验收。需新编译/权重/服务部署、条件不匹配或影响原任务就停止并记待条件，不自动运行server.sh、启端口、改变移植方案或另排队；原优先级不变，记录不代表接单。

M5 最小下一步：先按实际硬件查已有方案和精确 head，读清依赖、权重格式、显存及其原始测试入口；核对通过后只选一个可隔离、可停止的小范围兼容检查，再决定是否具备 offline inference 条件。保留初始失败点与配置；不要直接按旧专题的“换三个包/放宽检查”当成已验证操作，也不要先覆盖已有环境。

停止条件：没有匹配硬件、内存/数据预算不成立、测试入口不明确或需扩大范围时，记录原因和下一项缺口，不自动下载巨量权重、改写移植方案或进行长时间编译。完成标准是目标配置下可复现的实测记录或明确阻塞，不是必须产出性能数字。

2026-10-22 另一个参考入口（不另排队）：[#193 V4-Flash-0731 EP8 runner](https://github.com/deepseek-ai/DeepInfer/pull/193)，读取open、wangzhao-11a，head `a87157392233938be508170f44790b76c2a0510c`，base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。作者报告8×910B4-1、真实W8A8权重、43层的短生成，并列出输出逆RoPE和逐head无权重RMSNorm的修正经验；这是0731参考路径，不是#192的4.1单卡配置或生产engine支持。每token重算全序列、无paged KV，稀疏indexer未接入，不能外推吞吐/910C兼容。M1只核正文/版本，未审实现或复跑；供M5在原任务独立判断，不自动应用、下载权重/切环境、采用作者命令或打断现有实验，原停止条件不变。

## DeepInfer 显存遥测的轻量核对

状态：**M5 已完成（2026-10-22，4090-1）**，见[实验材料](deepinfer-mem-telemetry/readme.md)：真机核对 `get_occupied_gpu_memory` base/head（#151）与 `show_gpu_memory_stats`——两次调用稳定时 base 即第二次采样对应 occupied（base==head==412286976）；Allocated/Reserved/Occupied 三口径语义不同（del 后 0.000/1.000/1.445 GiB）；异常路径 base 静默返 0、head 返 0 并发 warning；base/head utils.py SHA256 与 M1 记录逐字节一致；真实 import 独立进程验证通过。原条目前提不变：M1 只核源码，作者未做 GPU runtime 验证——本项补真机行为证据，不构成对 #151 的验收/评论（已有作者，不竞争实现）。

来源：[遥测笔记](../notebook/frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划)、已有 [#151](https://github.com/deepseek-ai/DeepInfer/pull/151)，2026-10-21 读取时open，base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`、head `d72f3c4a7d1cfc1bd337adb03dd1319973b084d4`。只改单次查询/warning，作者也未做GPU runtime验证。M1仅核源码/AST，未执行设备API或替身测试；不预设旧查询在稳定total时必错，或此项能修复KV规划/OOM。

最小下一步（已完成）：M5先核对目标设备、可见编号与实际torch/backend版本，在独立进程/既有可用环境中核对正常mem_get_info及helper返回和日志。若隔离导入helper，须记录方式并保持原函数体；不为验证日志加载权重或编译整套DeepInfer。保留free/total对应的采样身份与时序，不用后一次查询的结果强求前一次值相等；NPU重映射语义另行标明，不能从CUDA结果外推。

停止条件：无合适runtime或需打断现有任务就记待资源，不诱发驱动故障、不强制错误device或修改训练来触发告警，不要求必须补真实异常分支。回写实际命令/版本/日志与未测路径，不把单次正常读取称为原子全局快照、完整日志集成或性能验证。

## verl 量化层审计新版复验

状态：待 M5 核对匹配硬件/软件栈，本项目尚无下述新版路线的真机复验；#8306作者另有Blackwell观测，不是M5结果。该项不代表已接单，也不作为其他任务的前置条件。

来源：[源码与24项CPU单测](../notebook/frameworks/verl-quant-layer-audit.md)、已有 [#8132](https://github.com/verl-project/verl/pull/8132)，2026-10-21读取时open，head `0defd9a231cb876ce56812611b0afb3b2b25831a`，base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。第7部分依赖前序量化堆叠且按#8287重做；不能盲目只cherry-pick审计。作者旧B200/ModelOpt观察不是新版验收；M1只运行作者自带替身的CPU单元测试，没有真实TE/compressor/engine。

旧#8132审计路线的最小下一步：先核实际设备是否支持目标TE FP8路径、依赖/模型/格式是否匹配及是否已有作者新结果；固定完整提交与安装版本，不覆盖已有环境或打断工作。条件齐备后，使用获准的最小模型/配置，真实FP8 forward后做一次权重导出/同步，保留训练workspace/marker、导出格式与引擎方法的证据、实际route、checked/skipped/unresolved及no-verdict状态。非colocated路径不能声称完成同样的live-vLLM复查，SGLang guard另列执行范围。

SGLang 旧#8132路径接续（同一任务，不另排队）：见[分桶与失败边界](../notebook/frameworks/verl-quant-layer-audit.md#sglang-分桶加载与失败清理)。若实际选择该路径，先核 native MXFP8 路由/版本，再记录每 bucket 与 finalize 完成点、实际 dense/MoE backend、暂存/回绑及 graph 使用范围；保留 `VERL_MXFP8_REFIT_CHECK`/阈值、probe 层/expert、误差或明确 skipped 原因。恢复 scale 指针不是恢复旧参数，正常 finalize 也不代表所有 probe 执行；不诱发生产故障或在失败后未经状态核验直接续训。旧路线源码已有 dense fallback 存储重绑 TODO，按实际路径注明未测项，不把 MoE 检查外推全部 backend。

新重叠路线核对：[#8306](https://github.com/verl-project/verl/pull/8306)，2026-10-21读取open、wuxibin89，固定head `1beb35554a763d18ed76fcbefaac1b224eaa1eeb`，目标base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`（merge-base `5a8dd519ab32bfd36d2509a137f830ba79e89c54`）。见[覆盖对照](../notebook/frameworks/verl-quant-layer-audit.md#新旧-sglang-refit-路线的覆盖关系)：增加FP4 routed-expert与dense存储copy-back，拒绝custom refit delta_flush，但未接入旧审计/guard/数值probe，不能搬用旧测试结果。M1仅源码/语法核对，没有loader或真机执行；作者报告Blackwell/Qwen3.5-35B-A3B EP8/TP2默认graph观测，无新增配套CPU测试。M5已有4090不等于匹配该验证资源。接手时先重核当前head/作者状态、安装SGLang版本，独立选一条路线并固定config/exporter/loader/marker/backend；不自动拼接、应用补丁或同时开两套实验，不把“恢复live storage”当旧参数回滚。

ModelOpt忽略规则参考（2026-10-22，同一量化入口不另排队）：已有[#8329](https://github.com/verl-project/verl/pull/8329)，head `5c7e2d2ed49bd9f448043883d96b6e191e187e95`、base `16ec093cc4a235c603c29aed1f2c4e9f551fab48`，见[M1阅读](../notebook/frameworks/verl-quant-layer-audit.md#modelopt-模块忽略与子量化器)。目的为将来核对actor实际fake-quant选层，不仅比较导出/加载字节；M1原helper两版各11组标准库名称控制、4文件compile，缺ModelOpt/CT，未跑原34测试或量化/梯度/设备。缺口是匹配依赖与已选QAT路线，作者H100单seed10步不是本项目或4090结果。若M5独立选择且已有获准依赖环境，先核head和ModelOpt/CT版本，再考虑原入口 `python -m pytest -q tests/utils/test_modelopt_mxfp8_on_gpu.py`，分开记录真实is_enabled/兄弟隔离/BF16前反向；文件gpu后缀不保证每例都在CUDA，不为名字分配GPU。额外训练需模型/硬件/预算另获准，若缺条件、需切环境或影响当前任务就记待条件；不自动应用、拼接#8132/#8306、安装或开展H100训练，不将作者34通过当本机结果/CI成功，不代表接单。

停止条件：无匹配设备/TE/引擎或模型授权就写待资源/资料，不大规模下载或长训练，不用fake workspace替代真机验收。完成标准是明确版本与路由的可追溯结果或阻塞；选层相符不等于数值相同、回滚安全或#8250根因，不能凭此发布修复/PR。

## verl 异步策略一致性补证

状态：待数据与运行环境核对；不是 GPU 空闲后即可复现原报告的任务。

入口：[当前诊断边界](../open-source/verl/candidate-8250-async-entropy-collapse.md)、[策略来源与最小观测契约](../notebook/frameworks/verl-async-logprob-provenance.md)、[Issue #8250](https://github.com/verl-project/verl/issues/8250)。2026-10-21 API 读取仍 open、0 评论；没有复查全仓库修复覆盖。报告版本 `1876b06d0a3e4e71e06230be10af14492ca8a75b` 外加私有修改，私有数据/模型不可由项目公开材料还原，实际 vLLM 安装版本仍未知。

M1 已撤回确定因果诊断。已有数值、路由与异常控制未运行模型、原生权重同步或真实训练；没有证明 sync/cache、mask、reward 或数值路径是哪一个根因。

可选观测入口（不另排队、不默认采用）：已有[verl #8315](https://github.com/verl-project/verl/pull/8315)，2026-10-21读取open，head `4d6b3eb8606e24c1d70d648e92a1114129ec6c62`、目标base619c397d、merge-basecd58f550，见[阶段tracing边界](../notebook/frameworks/verl-async-logprob-provenance.md#新版阶段-tracing-与证据关联)。M1仅4模块语法/11方法去decorator后AST一致检查，未运行rl-insight或引擎。它记录rank update_weight与actor memory阶段，但no-op也可有区间，sender返回非全体应用证明，未新增显存字节/参数身份校验。真实同步/设备观测由M5在既有任务条件满足时独立决定，非接单记录。

若已有获准的匹配观测环境，可先固定verl/rl-insight实际提交、包路径、版本及metric_log callable分派，核对rank/actor lane与同步step关联、是否实际执行/异常，并保留独立显存与安装证据。缺后端、需要新增数据外发或会改变原报告环境时先记待条件；不为这项参考自动升级环境、部署服务、重跑长训练或上传私有payload，阶段图不能替代下述配对观测。

2026-10-22版本补充（同一参考不另排队）：#8315当前head `0cea139623cc8913d2241d21223df5682474041c`、目标basef2923cbf/祖先cd58f550，仅比前head新增65行CPU测试及worker lane整数→rank_字符串。M1隔离原类/原12项日志测试：前后head均12过，祖先7过5预期失败；另核原生产lambda为int2→str rank_2。客户端为原Mock，未跑真实后端/接收/时间线查询；已有原测试不能认证生产接线或GPU同步。若M5原条件满足且独立选用，按实际head核lane类型、step及后端接收/显示，仍不新增监控部署或实验任务，也不把通过Mock测试当#8250根因证据。

可选NIXL拓扑参考（同一事项，不另排队）：[#8316](https://github.com/verl-project/verl/pull/8316)，2026-10-21读取open，head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`、目标basef2923cbf/祖先619c397d。[M1原拓扑及纯断言检查](../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑)每版20组：旧R=1长度失败、候选全过，未prepare/注册/连接/传输。只有已有获准观测实际选择NIXL且恰为单rollout checkpoint worker时，M5独立决定是否核对此候选，先固定安装版本、backend、actor/rollout worker数和metadata顺序，再分开记录建组、传输、应用及恢复点；并非一个replica就必为R=1。不为此新建网络/下载模型、强制改backend或重跑训练，缺匹配资源就记未测；没有证据表明#8250使用该backend或由此导致。

版本补充：#8316后续head `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`只新增1项原CPU测试，实现未变。M1隔离运行原测试体：基线失败、新版通过；没有原生导入/pytest收集或NIXL传输验收。无需另开资源任务，后续若选用仍以当时实际head和上述条件为准。

可选超时保护栏参考（2026-10-22，同一事项不另排队）：[#7045](https://github.com/verl-project/verl/pull/7045)，head `6f458f6d49076c647a42c00d240b5eaa7449fdb7`、目标base3f9d4c0d/祖先220e9039，见[M1源码与12组纯参数检查](../notebook/frameworks/verl-async-logprob-provenance.md#组初始化的等待超时与任务退出)。本机无Ray，未复跑原15项；当前head只给控制端init refs等待加上限，prepare/传输/finalize等不在此预算内，不取消RPC/回滚。作者2×A100退出清理结果针对旧da00403，当前head待复验。仅当M5已有获准且匹配的同步诊断环境、独立选择该方案时，先固定完整head、Ray/NCCL/launcher版本及有效timeout，分别记录控制端异常、远端/driver退出、设备资源释放与独立新进程恢复；缺资源或需改动/打断已有任务则记未测，不造生产挂死、不原地重试group、不切backend/下载模型/自动采用补丁，也不认定#8250由此导致。此参考不代表接单或M1已触发真机任务。

M5 最小下一步：先判断是否有获准使用的对应配置、完整 token/mask、checkpoint/版本身份和实际环境；条件齐备后按既有观测契约采集训练 actor、干净重载、live rollout 的配对观测，明确 raw/processed 概率与采样处理，保留逐 token 残差及重复基线。需要同步证据时区分发送、接收、应用、cache、版本标记及恢复生成；不由单个版本号或均值接近零认证一致。

停止条件：缺私有材料/配对身份就记录“待资料”，不启动原报告的长训练，也不以公共 toy 或另一模型称已复现 #8250。对照有差异只能缩小范围；没有原始观测前，不设计新的修复、不发布旧诊断稿或承诺治愈阈值。完成标准是可追溯的配对证据，或明确缺失材料清单。

## verl V1 可变行布局参考

状态：待选型/匹配依赖，只保存本轮新线索，不加入自动训练队列或作为原任务前置条件，不代表M5接单。已有[#8201](https://github.com/verl-project/verl/pull/8201)，head `e891fd814e24c0b2d0c70a47e234f75a63d10ad6`、base `18cfb74852a053a27dcf301d3318a093be81b628`；[M1源码阅读](../notebook/frameworks/verl-grpo-source-reading.md#v1-源-prompt-单位与可变轨迹行)仅12项原纯配置/layout控制、8模块compile，未跑原测试/TQ/设备。目的为将来采用V1多输出或部分成功保留时，核对源UID计步与实际行边界，而不是诊断#8250或修改#8313。

缺口：匹配固定head的TQ/TensorDict/OmegaConf依赖及实际DP运行环境；作者32卡旧均衡benchmark不验证新版可变输出链。仅在M5独立选择此方向且已有获准环境时，先跑原 `tests/trainer/ppo/v1/test_minibatching_on_cpu.py` 与 `tests/workers/test_engine_workers_mini_batch_trace_on_cpu.py`，区分测试内StubTrainer/padding/engine替身；真实路径再核UID键格式、单元padding、各rank对应global_mini_batch_sizes、每epoch调用数和实际optimizer/LR步数，不把helper或替身通过当端到端。缺环境、需新模型/服务/编译、资源或扩大训练预算就停并记录，不自动安装、切环境、部署TQ、应用PR或拉起32卡benchmark；M5原任务优先级不变。

## verl #8281 超长 prompt 截断修复 —— GPU e2e 训练冒烟

状态：**M5 已完成 GPU e2e 冒烟（2026-10-22，4090-1），正向跑通 + 负向 fail-loudly 均验证**，本条目回写完成。详见下方 M5 实测记录。

任务背景：`_cap_text_prompt_length` 无条件左截断、忽略 `data.truncation` 四策略。修复让它遵循 `error/left/right/middle`（`error` 时 fail-loudly）。

- **PR**：[verl #8313](https://github.com/verl-project/verl/pull/8313)，head 分支 `Jackie2049:fix-8281-truncation-policy`（fork `Jackie2049/verl`），base 上游 verl main。
- **改动**（仅 2 文件，129 增 / 3 删）：
  - `verl/experimental/agent_loop/agent_loop.py`：`_cap_text_prompt_length` 遵循 `data.truncation`。
  - `tests/experimental/agent_loop/test_cap_text_prompt_length.py`：CPU-only 单测，11 项（M1 本地 Python 3.12.13 venv 全过；修复前对照验证失败，证明测试有效）。

**M5 实测（2026-10-22，4090-1，用户指派的 GPU 冒烟）**：
- 环境：服务器 4090-1（GPU 0）+ conda `verl-e2e`（ray 2.59.0 + torch 2.11.0+cu128 + transformers 5.8.1 + vllm 0.22.1），editable 源码 `/home/zxw/verl-pt-impr`。
- 部署：GitHub 从服务器直连超时（TLS 截断），无法拉 fork 分支；改为**从 M5 本机 `gh pr diff` 取 PR #8313 完整 diff（154 行）**，与本地 patch [8281-truncation-policy.patch](../open-source/verl/patches/8281-truncation-policy.patch) 核对实际改动行**完全一致**，`patch -p1` 干净应用到服务器源码（hunk 偏移 -60 行仅因 base 版本不同）。应用后 `_cap_text_prompt_length` 即 PR 修复版本。
- 单测：`test_cap_text_prompt_length.py` 11 项在服务器全过（5.08s）。
- **正向冒烟**：grpo + ToolAgentLoop + Qwen2.5-0.5B（HF 缓存）+ gsm8k tool_agent 数据（16 train/4 test，预处理脚本生成），单 GPU、`trainer.total_training_steps=2`、`max_prompt_length=512`（prompt 实际 111-120 token 远低于 cap）→ **2 个 global step 全部跑通**（step1 35.5s / step2 9.6s，num_turns=2，无截断无 raise），日志末尾 `SMOKE COMPLETED OK`。正常训练路径未因改动退化。
- **负向验证**（关键验收）：同样配置但 `max_prompt_length=32`（prompt 118/120 token 必超长）→ 默认 `error` 策略 **fail-loudly**：`ValueError: Prompt of 118 tokens exceeds rollout.prompt_length=32; configured data.truncation='error' which does not truncate. Increase rollout.prompt_length or set data.truncation to 'left'/'right'/'middle'`。修复前该路径是静默左截断；训练终止为预期行为。`left/right/middle` 三策略的真实训练行为未逐一跑（单测已覆盖策略分支，冒烟确认 error 契约即可）。
- 脚本：`/tmp/smoke_8281.py`、`/tmp/smoke_8281_neg.py`、`/tmp/make_gsm8k_agent.py`（服务器）；日志 `/tmp/smoke_8281.log`、`/tmp/smoke_8281_neg.log`。
- 边界：未改上游/PR/用户分支，未在 GitHub 上做任何操作（取 diff 为只读）；验证用的源码是 PR 内容的等价应用，非严格 git 分支 checkout（服务器无法访问 GitHub）；`left/right/middle` 仅单测覆盖。

给 M5 的验证目标（原）：
1. 正常训练路径不因该改动报错/退化 → **已验证通过**（正向冒烟 2 step）。
2. 若可配置，验证 `data.truncation` 不同策略在真实训练中行为符合预期 → **error 策略真实训练 fail-loudly 已验证**；left/right/middle 由单测覆盖。
3. 保留版本、命令、配置、原始日志/失败、真实执行范围 → 见上方实测记录。

停止条件：完成标准是有可追溯的 e2e 冒烟结果（正常跑通 + 记录版本/命令/日志），或明确缺失条件清单（缺 GPU/环境/模型授权等）。——**已完成**。

**最新分支复验（2026-10-22，用户要求推送并确保测试通过）**：
- fork 分支远端 head 已确认 `6db4b38d`（`872bc281` 用户风格 + `6db4b38d` middle/cap=1 修复）；PR #8313 仍 CLOSED（head 停在 `80ee82bd`），未重开、未评论上游。
- **服务器源码树已修一致**：`/home/zxw/verl-pt-impr` 原为 10-14 包快照，只打补丁替换了 10-21 的 `agent_loop.py`，导致 `build_initial_tokens(..., mm_inputs_out=...)` 与旧 `continuous_token.py` 不兼容（`TypeError`，与截断修复无关）。已把完整 `verl/` 包同步为本地 `HEAD`（`954bd28f` + 两 commit），旧包备份 `verl.bak-20261022`。**后续在该树工作前先核对该备份，避免覆盖其他会话的本地改动。**
- 复验结果：CPU 单测 **12 passed**；GPU 正向 2 step（30.8s / 4.5s，num_turns=2）`SMOKE COMPLETED OK`；负向 `max_prompt_length=32` → `ValueError` fail-loudly。日志 `/tmp/smoke_8281_latest2.log`、`/tmp/smoke_8281_neg_latest.log`。详见 [实验记录](verl-8281-gpu-smoke/readme.md#2026-10-22-最新分支复验872bc281--6db4b38d含-middlecap1-修复)。
