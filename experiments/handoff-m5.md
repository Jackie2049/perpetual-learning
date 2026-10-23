# M5 真机工作交接

GPU/NPU 的访问、环境与实验由 macbook-M5 Agent 负责。M1 遇到需要真机的事项在此记录，M5 机器可用时按原有任务优先级接续，不中断已有实验。本页是待接续入口，不代表 M5 已接受、服务器已在线或任务已启动。

更新日期：2026-10-22。M5 完成 DeepInfer 4090 移植冒烟第二段并回写本页（llm_engine import 链全通、发现 tilelang PDL 硬堵点）；PDL 探针续：5 个 kernel 去 PDL 改写编译 + 数值对照全过（含最复杂 fused_sample）；**四续：load/store_kv_cache 全矩阵数值验证（上游测试 60/60 PASS 于 nosync 变体），发现 sync_grid→cooperative-launch grid 上限、pdl_sync 依赖型应删除而非替换**；**五续：全量 nosync 改写验证——55 处 pdl_sync 全部依赖型分类、43/43 SM89 编译全过、7 个 kernel 数值保持，改写规则收敛为「全部删除，sync_grid 无适用用例」**。昇腾 910C VPN 掉线（100% 丢包）待用户恢复，SSH 密钥仍未配置。AReaL prompt_mean 整项保持 M5 单一负责，继续以[原交接](../open-source/areal/handoff-m5.md)为准，本页不重新打开或演化该方案。

## 接续与回写

- 接手前同步项目 main，核对该项是否已有负责人/结果；同一事项只维护一条。M5 接手后将状态改为“已接手”，记录选定版本和资源，不由 M1 推定接单。
- 先核实可用硬件、显存、安装依赖和数据条件；保留已有环境与工作区，实验环境隔离。资源不足或条件不匹配时写明“待资源/待资料”及恢复条件，不直接启动长训练或全模型加载。
- 实验结论回写对应主笔记与 diary，并在本页链接结果：保留版本、配置、命令、原始日志/失败、真实执行范围。没有采到的字段写 unknown，不能由依赖 pin 或历史机器记录补齐。
- 对上游发言/提交与公开发布仍按用户原授权边界执行；交接不增加对外操作权限。不将所有 CPU 学习的未测项自动变成必跑任务。

## UniRL 联合路径的资源与身份边界

Replay分块参考（2026-10-23，非新增训练队列）：[#465](https://github.com/Tencent-Hunyuan/UniRL/pull/465)已于10-18合并，当前main固定 `6fd82cba13f52f394747832344039408699ea378`；[M1阅读](../notebook/frameworks/unirl-survey.md#replay-分块与反向更新边界)仅5模块compile/两个原元素损失CPU聚合控制，无stage/算法/TrainStack/FSDP执行。若M5以后独立选择且已有获准匹配diffusion观测，先固定模型/精度/条件/随机状态与anchor权重、实际chunk列表/末块及batch形状；分开核ratio/梯度/更新、反向保存量与保留trajectory/anchor、defer_grad_sync和各micro的通信。#465旧4090报告不是当前main验收；缺现成配方/权重/资源或需新部署训练、影响专项即保留未测，不修改#523/AReaL或堆替身验证，参考非接单。

压缩积分参考（2026-10-23，非新增训练队列）：已有[#545](https://github.com/Tencent-Hunyuan/UniRL/pull/545) head `7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8`、base/祖先 `c6d4b11b24743fe1e933b61192f9d49558f7171e`；[M1原纯模块](../notebook/frameworks/unirl-survey.md#sde-窗口后的压缩积分尾段)仅720静态日程/64合成向量场小张量控制、12模块compile，无Flow SDE/logp、SD3或训练验收。若M5日后独立选择且已有获准匹配SD3任务，需先固定模型/条件/精度/seed与完整配置、记录实际SDE索引及σ出口、模型调用和窗口轨迹/logp；示例另把iters_per_window从25改20，NFE减少不等于reward-vs-wallclock提高，作者L40S报告非本项目。无现成匹配环境/权重/资源、需新下载部署或影响专项即保留未测，不启动新训练、不拼其他PR或改#523/AReaL；参考非接单。

异步计时参考（2026-10-23，非新增训练队列）：已有 [#539](https://github.com/Tencent-Hunyuan/UniRL/pull/539) head `89fdd083a4cefd090e0d46fa37da37abe185d850`、base/祖先 `9e14d007b48ad6ecabe06df1ec4691d69492ebb1`；[M1窗口核对](../notebook/frameworks/unirl-survey.md#异步计时窗口与性能比较)仅8模块编译/3标准库原timer控制，无trainer/Ray/W&B/设备。若M5日后已有获准匹配异步跑次且独立选用，先固定计时版本、rollout/update横轴与sync/eval/save边界，核实际publication完成点及phase覆盖；主机调用耗时、设备时间与全运行摊销分别记录。缺匹配模型/后端或现成观测、需部署/启动新训练或影响专项就停止记未测，不由旧新同名曲线判退化、不用collect等待反推GPU重叠、不自动拼#620或改#523，本参考非接单。

行级奖励派发参考（2026-10-23，非新增训练队列）：已有 [#517](https://github.com/Tencent-Hunyuan/UniRL/pull/517) head `fca5a55a0e5b30d108289f7394a7c50850593bf2`、base/祖先 `7a5303647d0c1ec27807124f70939cbb8edde6e9`；[M1 原类型与4 helper](../notebook/frameworks/unirl-survey.md#行级奖励派发与分组身份)5组CPU控制通过，分数为夹具标签，无真实服务/dispatch/模型或设备。若M5日后独立选择且已有获准匹配DiffusionTrainer/奖励后端，先确认scorer是否依赖原group身份，再核真实DP分片/聚合顺序、prompt/conditioning/metadata与packed或引用数据的对应及回贴；分开记录奖励计算与搬运成本。无匹配权重/依赖/资源、需下载部署/扩卡或影响专项就停止记未测，不自动切row、采用#378/32卡配方、拼接其他PR或改#523专项，此参考非接单。

HI3 tokenizer参考（2026-10-23，非新增算力队列）：已有非draft[#544](https://github.com/Tencent-Hunyuan/UniRL/pull/544)head`d41122e97e1ab631328c86baad00250c406d2b17`、base/祖先`4fb0b1f7b291f3cd5010597b85d41ed44659ef40`；[M1阅读](../notebook/frameworks/unirl-survey.md#tokenizer-实例修补与批量生成边界)仅两版共4模块语法/AST身份检查，无真实tokenizer/模型执行。作者20CPU例绑定checkpoint`2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`，GPU生成/token-logp parity尚未验，配方仍FBS1。若M5日后已有获准匹配HI3资源且独立选择此修复，先重核checkpoint源码/签名与shim是否仍需，最小记录各行ID/token/real_pos/sections与CFG对应关系，再核短生成、ragged mask/RoPE/KV写入及token/logp；行数不替代身份。缺授权权重/匹配资源、需新下载/部署或影响专项即保留未测停止，不直接提高FBS、不拼接#300或改变#523任务；参考非接单。

样本观测参考（2026-10-23，非新增算力队列）：draft[#597](https://github.com/Tencent-Hunyuan/UniRL/pull/597)head`d507c4dc1e10db427cadff4d6e29482801a12065`、base/祖先`6c828f5ed81649153315d996f0f597e8808ff144`；[M1学习](../notebook/frameworks/unirl-survey.md#样本观测的行上限与数据搬运)仅原CPU Sample/音频/本地sink六组控制，无trainer/远程hydrate/W&B/模型执行。作者H100小模型短训非Omni/HI3原配方或多节点driver RSS验收。若M5日后已有获准匹配UniRL任务并独立选择该draft，最小记录选中ID/祖先文本/reward、实际TensorRef块与hydrate字节、driver RSS和本地输出失败后的训练更新；先固定模型/配方及transport版本，关闭不必要的数据外发。缺授权权重、匹配多卡/transport/存储或需新增下载部署、影响专项即保留未测停止，不为本参考启动70/169GB模型或新增必跑矩阵；参考非接单，不改#523/AReaL。

多教师周期参考（2026-10-23，非新增训练队列）：新draft[#624](https://github.com/Tencent-Hunyuan/UniRL/pull/624)head`375c2273ed5b0ef89d3075350ba96ba2bcdb31eb`，A3独立边界`0142d612fbd824c9a47d3f6446a4b914a04eb1c3`、目标base/祖先`6fd82cba13f52f394747832344039408699ea378`；[M1阅读](../notebook/frameworks/unirl-survey.md#多教师周期与恢复身份)仅10/11完整模块AST/compile，无作者CPU/GPU harness或真实模型执行。作者2×A100/Qwen3-4B-Base合成教师结果非本项目，normalized-gap恢复仍未另验。若日后M5已有获准匹配多教师选题并独立选用，需固定教师内容hash、数据内容/顺序、周期/DP布局、dropout及kernel设置，最小先核完整周期连续/恢复的样本与domain、token/logp、教师冻结、学生和Adam状态；JSON路由相同不替代内容/持久化核验。缺现成匹配环境/授权权重、需新部署/下载或影响专项就停留未测，不启draft配方、不拼接改#523/AReaL或新增必跑矩阵；参考非接单。

Trainside组边界参考（2026-10-23，非新增算力队列）：[#485](https://github.com/Tencent-Hunyuan/UniRL/pull/485)head`9e36cf99eea03e9e2f6035976ac97c1a70759f9e`、base/祖先`772fd7c50ec8221a032179aeec8320df935b3abf`；[M1原helper控制](../notebook/frameworks/unirl-survey.md#prompt-组边界与前向分块)仅16,400有界布局与大组几何CPU对照，两版engine/NoiseRecipe仅compile，无Sample/Part、RNG、模型或设备执行。作者H100属旧683c320非当前head复跑。若M5日后已有获准匹配diffusion选题，可保持FBS/seed/recipe不变，最小核开关前后调用行序/组ID、初始噪声、轨迹/logp及组优势；更换FBS另列为RNG/批形状实验，不混为flag parity。缺匹配模型/环境或需新部署/下载、影响专项即保留未测停止，不放大FBS、不启开关/训练或新排矩阵，本参考非接单。

常驻/流式规划参考（2026-10-23，非新增算力队列）：[#486](https://github.com/Tencent-Hunyuan/UniRL/pull/486)head`57cec429b5978d7da4bfd5f751764c0fd5da4c06`、祖先`f8d95462837df00188dab1376e19347b43996307`；[M1原planner对照](../notebook/frameworks/unirl-survey.md#常驻与流式-block-的规划账本)仅36组synthetic配置/180候选算术和拒绝路径，非测量profile/FSDP/执行器或设备验证。作者H100数值能力与组件profile不等于新执行器收益，transition仍缺。若M5以后已有获准匹配选题，最小先核排除block参数的逐rank四phase高水位、互不重计的compute/transfer、shard/full/cast角色与实证live bound，再对既有执行器核预测峰值/等待误差；无完整profile/现成执行器、需部署/租算力/新模型或影响专项就停留未测。不因计划FEASIBLE启offload，不将max-phase/transfer_busy字段当整步时间或DMA利用率，不自动排训练/改其专项，本参考非接单。

离线回放来源参考（2026-10-23，非新增算力队列）：[#487](https://github.com/Tencent-Hunyuan/UniRL/pull/487)head`e8779559d2ceb441c062f603f90e7fd31370e69b`、祖先`f8d95462837df00188dab1376e19347b43996307`；作者fork guard`4ad73720989c8b7c9229da4ecabd9ee654e67dc4`独立。见[M1原模块CPU对照](../notebook/frameworks/unirl-survey.md#离线回放的到达契约与结果隔离)：仅synthetic stdlib契约，无捕获/runner/simulator/设备验证。上游两个已知错误接受不能当已修；profile-backed CPU模拟也非真机。若M5以后已有获准匹配回放选题，缺真实capture、固定runner/simulator与model/backend/dtype/TP/topology/profile一致证据；最小先核请求ID、arrival语义和边界，才比较同负载逐请求完成。作者后续目标H100，不自动外推4090/H20；无现有匹配环境/资源、需新部署/租用/下载或影响专项即停止保留未测。不装模拟器、不启训练、不排新必跑矩阵，本参考非接单。

冻结教师路由参考（2026-10-23，非新增训练队列）：[#519](https://github.com/Tencent-Hunyuan/UniRL/pull/519)head`fd1ca404804dc2cc0b395bd0aa5765c012104d9a`、base/祖先`ff1d3a6b2a8070ba31d5e74ac02c0e965762569c`；[M1原调研](../notebook/frameworks/unirl-survey.md#冻结教师路由与调用方状态恢复)仅源码/语法及六helper+prepare_part的publication/head AST同一性，缺PEFT/Transformers/Diffusers，无运行或新替身。作者报告属于5390b5af的PEFT0.20/torch2.13环境，当前完整模块已变化，非本项目或整个新head验收。若M5今后已有获准匹配教师LoRA选题，可先在现有真实PEFT环境核逐层选择/全部requires_grad、异常退出与普通输出，再按既定模型确认no_grad/重算边界；无现成环境/授权权重、需新部署/下载或影响专项则停止保留未测。不是embedding-only加载/同步/恢复或#523行为anchor任务，不自动应用/增加必跑矩阵，本参考非接单。

空响应反向参考（2026-10-23，非新增训练队列）：[#598](https://github.com/Tencent-Hunyuan/UniRL/pull/598)open draft，head`fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73`、base/祖先`6c828f5ed81649153315d996f0f597e8808ff144`；[M1原调研](../notebook/frameworks/unirl-survey.md#空响应的反向通信与零梯度更新)8组原GSPO/PPO分支的有限CPU图对照及6组普通optimizer控制通过，无完整算法/replay、padding、FSDP或设备验证。作者2×H100报告仅tiny FP64线性模型/SGD，不是HI3/BAGEL或多rank CUDA PPO。若M5日后已有获准匹配FSDP选题，最小记录非对称空响应与最后micro的实际replay/backward/collective次序、PPO两输出及更新前后grad/optimizer状态；缺匹配模型/引擎/mesh或需新部署/下载、影响专项就停止保留未测。零数据贡献不当参数不变，不自动应用draft、重开#581修复或改#523/AReaL，本参考不表示接单。

Agentic回合续跑参考（2026-10-23，非新增训练队列）：[#558](https://github.com/Tencent-Hunyuan/UniRL/pull/558)head`76a3c755210728bf03dcaaaeefe41187472d1dbb`、base/祖先`852c0454a6cb63592b2582358f88623870848df1`；[M1原调研](../notebook/frameworks/unirl-survey.md#回合边界续跑与行为版本)仅源码及两版各7模块compile，无agent/manager/训练或设备执行。作者A100报告属旧9b68dc9f，模型Qwen3-0.6B revision`c1899de289a04d12100db370d81485cdf75e47ca`，外部harness非本机复跑。若M5今后已有获准匹配agentic选题，缺口是实际SGLang/NCCL、授权模型/数据、可恢复的无worker状态工具和transport生命周期；最小核一条跨发布轨迹的工具观察/调用次序、逐turn行为logp与实际权重内容，再核窗口最终训练消费完才释放视图及drained checkpoint恢复。未具备环境、需新部署/下载或影响原专项即保留未测停止；不启配方、不拼接#557、不改#523/AReaL，不表示M5已接单。

Rollout token校正参考（2026-10-23，非新增训练任务）：[#557](https://github.com/Tencent-Hunyuan/UniRL/pull/557)读取open，head`300170e99209f26908d637971b3146d2ac7f98ff`、base/祖先`852c0454a6cb63592b2582358f88623870848df1`；[M1原调研](../notebook/frameworks/unirl-survey.md#replay-锚点与-rollout-采样校正)原分支/clip/reducer36组CPU局部值/梯度控制通过，无完整GRPO/anchor生命周期/模型或设备执行。作者Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`的L40S/在线100轮报告仅小型单seed，不证明校正收益，关闭校正的高LR坍塌不作本项目故障归因。若M5已有获准匹配AR选题，最小保留同token/温度/支持集的new、冻结old/behavior字段及版本，核多update不改两anchor、原分母/局部梯度和同步后的真实权重内容；ESS按实际统计范围标注，不平均micro冒充global。缺环境或需新部署/下载/干预专项就停止，不自动启用cap、跑学习对比或修改#523/AReaL，记录非接单。

有界补样参考（2026-10-23，不新增训练队列）：已有[#562](https://github.com/Tencent-Hunyuan/UniRL/pull/562)读取open，head`e8cd33daa6067a0da69466f24cc6f1bbd3a082bc`、base/祖先`772fd7c50ec8221a032179aeec8320df935b3abf`；[M1原调研](../notebook/frameworks/unirl-survey.md#有界动态采样与数据游标)仅源码/两版各5模块compile，无trainer/Sample/TensorRef/恢复或设备执行。当前head作者CPU外部harness报告与旧10e57c6的2×L40S/Qwen3-0.6B（revision`c1899de289a04d12100db370d81485cdf75e47ca`）GPU报告须分开；旧输出不作新head验收。若M5今后已有获准匹配同步AR选题，最小核同版完整组/ID与远程视图、K含首批的耗尽前无update，以及refill前后full权重/LoRA payload、sleep备份与checkpoint消费批次；区分同数据请求与随机completion重现。缺匹配引擎/数据/权重或需新部署/下载/打断专项就保留未测停止，不自动应用配方、设计DAPO组合或重开vLLM版本修复。本参考非接单，不增加原任务优先级。

AR熵参考（2026-10-23，不新增训练队列）：已有[#563](https://github.com/Tencent-Hunyuan/UniRL/pull/563)读取open，head`00e5597f6ff6de639d7997a2ea21266d688e9f72`、base/祖先`94e8f26b8768a3695b5985c2fa072e07d9dbfa0d`；[M1原调研](../notebook/frameworks/unirl-survey.md#全词表条件熵与有效-token-汇总)原LM-head/Linear夹具20组CPU局部梯度及纯指标聚合控制通过，11模块仅compile，无完整Qwen/replay/TrainStack/设备执行。作者另报Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`的A100/真实FSDP-Ray一次更新，外部harness非本机复跑。若M5今后已有获准匹配AR选题，可最小核同一响应/模型版本/T/mask下启闭监控的logp和梯度，再核真实DP分片/TP-SP副本的sum/count及rank返回；重复update计观察次数，greedy时T=1熵不当采样器熵。缺匹配权重/后端/mesh或需新部署/下载、干预专项即保留未测并停止，不自动应用、启训练或把此项认定#8250根因；记录非接单。

原始block权重参考（2026-10-23，非新增训练任务）：已有[#593](https://github.com/Tencent-Hunyuan/UniRL/pull/593)为open draft，head`da8d285ac529c2f06819bf3a098e3cc657ae6618`、base/祖先`2d9e57fdee65ea69254fdb958d3fca9ce4872661`；[M1源码/CPU数学控制](../notebook/frameworks/unirl-survey.md#原始-block-权重训练与冻结外围)区分冻结外围与no_grad、原非重入helper、优化器参数过滤，未执行H3/FSDP/TrainStack/replay/恢复。作者原checkpoint `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`的8×H100 80GB/256×384/124帧/10步/FP32 masters/resident VAE首次反向仍OOM，尚无成功原模型更新。若M5今后独立选择此方向，先核授权权重、精确配方/后端与训练峰值预算，再决定是否有条件做一次更新及固定状态恢复；需新权重/编译/扩卡/租用/offload设计或影响原专项则记录待条件并停止，不把CPU夹具、加载/生成成功或LoRA历史收益当本路线验收。不自动应用候选或改资源方案，本参考不表示接单。

状态：待资源匹配与M5选题，非已接手或即时执行任务；不打断既有GPU/AReaL工作。目的：后续若获准推进UniRL真机，区分driver样本树接线与实际engine行为，避免把模板注释或静态源码当实测。

固定官方main `6fd82cba13f52f394747832344039408699ea378`，[M1源码入口与证据](../notebook/frameworks/unirl-survey.md#ar-与-diffusion-的源码统一边界)。M1仅原静态依赖guard与4文件AST/compile通过，无训练包导入/设备探测。HI3 driver构造P→P×N→P×N×M，但同版配方旧注释仍称平坦对；没有用CPU替身替代真实engine验收。

条件：该HI3双引擎构造器要求每节点至少8设备，batch_size为num_devices的正倍数；[固定版安装文档](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/INSTALL.md)为Linux x86_64/Python3.12–3.13/NVIDIA CUDA13，vllm与sglang extra不应混装。权重/显存/engine兼容与当前资源均未核验。不能因已有单4090工作或安装文档未限架构，就默认此HI3配方可跑；也不外推所有UniRL至少8卡。

最小接续：仅当M5后续选定该路径且已有获准匹配资源/权重时，先确认实际recipe与版本，不直接照默认大模型加载；一次有界rollout记录三层ID/行数、每个AR文本实际对应的M份图像条件及同步版本，再决定是否值得训练步验证。数量正确不代替身份或数值正确。资源/许可/数据不匹配则记录未测并停止，不装环境、下载权重或启动长训练；无需为本条新租算力。本条不授权上游修改/评论/PR，也不代表M5已接单。

稀疏SDE参考（同版，不新增排队/前置条件）：[M1原配置/调度模块检查](../notebook/frameworks/unirl-survey.md#稀疏-sde-与轨迹状态位置)覆盖64个rollout id及56种前半段三索引组合；T16/S3源码仍遍历16去噪迭代，端点加终态预期5～7位置不是实测显存。若M5后续本来就要采实际rollout，可记录有效sde_indices、完整sigma长度、LatentSegment.indices与sde_logp形状，确认engine/训练replay字段对应；不以配置数量代替实际logprob可用性或节省比例，不要求为本参考额外跑模型/矩阵。无匹配资源或不在原选题范围则保留未测。

关键帧消费端参考（2026-10-22，非新训练任务）：已有UniRL[#584](https://github.com/Tencent-Hunyuan/UniRL/pull/584)，head `55a3cad3e934cb0e637d8f3cf1c37d2a91886891`、base `accba01082115534e8b60d849e7276a636eba269`；[M1源码学习](../notebook/frameworks/unirl-survey.md#关键帧条件的数据身份与功能就绪)仅9模块compile及静态未变核对，无parity/NFT loss/模型或设备执行。目的为日后选MiniMax-H3关键帧路线时核条件数据身份，不继承HI3“至少8设备”约束，也不代表4090可跑。**当前先停止于功能前提**：pipeline未设置keyframes，标量/向量条件timestep仍是target时间，README禁止recipe启用FL2VA；先等producer及step2语义完成，不为消费端参考准备算力。

若后续M5独立选型且前提已就绪，缺口还包括授权权重/数据、匹配H3模型和TQ/transport/精度环境；最小检查应从真实producer取一份条件，按sample/step保留rows/anchors/tags与dtype，分别核generate→replay和NFT forward，再决定是否具备loss或设备验收条件。作者合成tiny模型外部harness未提交，不能当现成本地原测试或真机结果；缺条件、需新模型/环境/预算或影响原任务就记录未测并停止，不自动下载、cast到fp16、部署、启recipe或演化并行方案。本参考不另排队、不代表接单。

设备与后端参考（2026-10-22，不新增算力任务）：[#623](https://github.com/Tencent-Hunyuan/UniRL/pull/623)本轮已关闭未合并，head `c1477a030cda826b6befd2e6e787ef64f67e41dd`、实际父 `c6d4b11b24743fe1e933b61192f9d49558f7171e`；[M1分层阅读](../notebook/frameworks/unirl-survey.md#设备资源与通信后端的分层边界)仅两版各3模块compile，无设备/通信执行。actor num_gpus=0仍保留PG的GPU:1预约，driver CVD全局索引不认证每节点卡身份；异常回退硬编码eccl且占用默认WORLD，不是原生昇腾支持。作者8卡GCU报告无可复跑日志，非NVIDIA或本项目验收。仅当M5日后独立选择该路线且已有匹配获准环境时，核实际Ray资源/插件、逐节点actor可见性与物理/逻辑卡对应、eccl注册及默认组所有权，再决定有界小张量P2P是否适用；无GCU/匹配后端、需新环境/预算、默认组冲突或影响既有任务则记录未测并停，不自动应用关闭候选、改环境、开训练或将本参考加入必跑队列。

TempFlow分支参考（2026-10-22，非新增训练任务）：[#552](https://github.com/Tencent-Hunyuan/UniRL/pull/552)固定head `ef9cb362ee8f92f1dcae6ec474075ed922ef994d`、base/祖先 `6fd82cba13f52f394747832344039408699ea378`，含Flash/replay前置；[M1阅读](../notebook/frameworks/unirl-survey.md#tempflow-分支共享与逐行时间索引)只执行完整原noise模块的CPU小张量控制及9模块compile，无trainer/replay/loss/模型运行。共享x_T不是前缀计算缓存；默认4种子×6分支×3步组=72行/prompt，仍每组完整10步，实际峰值未知。若M5今后独立选择且已有获准SD3.5-M/匹配FlowSDE、reward、FSDP和资源，可先固定提交、sigma表、精度/批形状及实际old_logp_source，最小核step组/seed/sample ID、x_T与分支前后/终态三槽、每行j和单转移logp；先区分数据接线/精度差异，再决定是否值得loss/训练验证。作者早期8×L40S/特定fp32批形状报告非当前head或4090验收，不继承HI3必需至少8设备的限制，也不因配方num_devices=8就宣称普遍硬需求。无匹配权重/环境/预算、需新编译部署或影响原任务就记录未测并停，不自动下载、启配方/EMA/async、改M5环境或另排训练矩阵；此参考不代表接单。

异步指标参考（非新增实验任务）：[#620](https://github.com/Tencent-Hunyuan/UniRL/pull/620)固定head `f404c9024c00bbf8a5d05315153444cee894865a`；[M1监控契约阅读](../notebook/frameworks/unirl-survey.md#异步快照与指标消费)仅6模块compile，无pool/Ray/W&B/设备执行，作者5次训练及500记录未复验。若M5日后本来就在获准匹配环境开展async，可保留同rollout/step的pre_sync/step_end、optimizer版本和单一drain消费者，区分准入前headroom与提交后状态；未知lag不补0，逻辑slot占用不当GPU利用率，日志发出不当持久保存/后续eval成功。缺环境或不属现有选题则保留未测，不部署候选、启训练或增加必跑矩阵，不打断原任务。

奖励加载参考（非新增算力任务）：[#621](https://github.com/Tencent-Hunyuan/UniRL/pull/621)head `27cb49307fc89dd60331f86d23cadb2e8cc4b0cb`，[M1生命周期阅读](../notebook/frameworks/unirl-survey.md#奖励能力校验与模型加载)仅两版各6模块compile及未改文件blob核对，无构造器/模型/设备执行；作者42项fake-loader检查非真实权重验收。若M5日后独立选择已有获准匹配的reward路径，可先固定local/remote/managed类型与依赖/权重revision，分开观察校验错误、实际加载及异常后的资源/子进程状态；local的_loaded=False不保证已回收，managed仍在构造器启动孩子。缺授权权重/环境/预算或影响原任务则保留未测，不下载、部署候选、启动服务/训练或增加必跑队列。

Omni原生IPC参考（非新增训练任务）：[#617](https://github.com/Tencent-Hunyuan/UniRL/pull/617)head `19f263968deecdecd26cebe8966f8ba012a5575d`，[M1阅读/原helper CPU范围](../notebook/frameworks/unirl-survey.md#omni-原生-ipc-的副本与导出边界)只覆盖meta/视图、8模块compile，无CUDA导出/IPC/模型。若M5日后独立选择该路线，先核配套Omni WTE（作者测试8607b97ea5）、授权模型/依赖、同节点每卡TP1副本及UUID、CPU offload关闭下的显存与sleep备份RAM；不继承HI3至少8卡，也不因1x1 YAML存在就认定4090可跑。最小观察分开记schema、实际权重内容、publication序号/optimizer版本与失败后的运行时关闭；结构hash不认证内容，失败不回滚，控制链含pickle需可信。缺现有匹配环境/资源或需安装配套分支、下载权重、打断原任务就停留未测，不自动部署/启训练或增加必跑矩阵。

#523 锚生命周期修复（2026-10-23）：用户已审阅并认可 fork PR [#1](https://github.com/Jackie2049/UniRL/pull/1) 的方案设计与代码实现，测试验证与 e2e 冒烟已升级为**正式指派任务**，见下方专设小节「UniRL #523 unified 栈锚生命周期修复 —— 测试设计与冒烟」；本节其余 UniRL 参考条目不变。**✅ M5 已闭环 L0 并判定 L1（2026-10-23，本会话）**：L0 真实模块冒烟四项断言全过（含 pristine 对照），L1 判定「待资源/待权重」——详见专设小节下方完成记录与 [实验材料](../experiments/unirl-523-l0/readme.md)。

## UniRL #523 unified 栈锚生命周期修复 —— 测试设计与冒烟（2026-10-23 用户指派，交 M5 闭环）

**✅ M5 完成记录（2026-10-23 本会话，4090-1）**：**L0 真实模块生命周期冒烟完成**——真实 `unirl.train.stack.unified`（上游 main `6fd82cba` + 修正补丁，指纹已核）取 `train_track.__wrapped__` 真实方法体绑定替身 self，`prepare_segment_anchors` 用真实 anchor.py 函数，替身逐项记录（fsdp_backend.model 为真实 torch Linear 包装、算法三种变体、sample/Part/planner/`_train_one_step` 最小替身）；**四项断言全过**：①调用序 eval→freeze(ar)→prepare_part(ar)→freeze(image)→prepare_part(image)→train、②(c) 抛异常后 train() 由 finally 恢复且异常上抛、③(a) 两个 part 原样返回（同一对象流入 _train_one_step）、④(b) 两个 prepare_part 各调恰一次；**pristine 对照**（同脚本指向未打补丁树）确认旧代码无 eval/train/prepare_part、四项 patched 断言全部失败、失败替身永不触发——断言有区分度。环境 Python 3.12.15 / torch 2.11.0+cu128（verl-e2e）；日志 `l0_patched.log`/`l0_pristine.log`，脚本与细节见 [experiments/unirl-523-l0](../experiments/unirl-523-l0/readme.md)。**L1 判定：待资源/待权重**——GPU 仅 0/1/7 空闲（3 卡，2–6 他人 VLLM 占用），无法整段占用 ≥8 卡（hi3 需 8 / bagel 需 32 多机），Bagel/HI3 权重未授权未下载（HI3 MoE 远超单卡 24 GiB，磁盘剩 149G）；恢复条件 = ≥8 卡同时空闲 + 获准权重与存储，满足后按下方四步执行。已按停止条件停（未下载未授权权重、未为 L1 新租算力、未打断其他会话）。

**任务来源与边界**：用户 2026-10-23 审阅 fork PR [Jackie2049/UniRL#1](https://github.com/Jackie2049/UniRL/pull/1)（base=fork main，未涉上游）后明确：方案设计和代码实现都没问题，设计好测试验证方式与 e2e 训练冒烟，交 M5 测试闭环。本条为指派任务，不再是参考条目；对上游 Tencent-Hunyuan/UniRL 仍无任何操作权限，结果回写本页 + [candidate-523](../open-source/unirl/candidate-523-anchor-lifecycle.md) + diary。

**执行顺序澄清（2026-10-23 傍晚补，回应"无空闲卡跑不了"的判定）**：L0 **零 GPU 依赖**——真实模块 + 替身协作者在 CPU 即可执行，与本轮或任何时点的空闲卡数无关，**不得因 L1 资源判定未过而搁置 L0**；本机 Python/torch 可用即可开始。L1 才受"≥8 卡同时空闲 + 获准权重"约束（17:07 轮判定：GPU 0/1/7 空闲但仅 3 卡、2-6 被他人 VLLM 占用，不满足）——该判定对 L0 不适用。

**修复内容**（一句话）：`UnifiedModelTrainStack.train_track` 补齐 single 栈四步生命周期 `eval() → prepare_segment_anchors(ar) → prepare_part(ar) → prepare_segment_anchors(image) → prepare_part(image) → train()`（try/finally 兜底恢复 train），仅 `unirl/train/stack/unified.py` +8/-2。M1 已做提取逻辑级验证（3 项单测 + 真实文件 AST 断言）；本任务升级为真实模块与真实环境验证。

**物料与字节校验（第一步，必须先做；M5 已于 2026-10-23 真机完成）**：
- 补丁唯一来源：项目仓库 [523-lifecycle-fix.patch](../open-source/unirl/523-lifecycle-fix.patch)（sha256 `a2d2437bd0ef429fcd62bf3d381862048cf388fcd461f6d297aedaa17b03ec0c`；**2026-10-23 M5 已修正为与 fork 一致版**——原补丁含 3 行注释、应用到 pristine 得 blob `1814dd43` ≠ 验收指纹，fork `04101ad5`（用户已认可）提交无注释、blob `2cced465` 与验收指纹一致，权威版本是 fork；已去注释修正并重新核验）；服务器 GitHub 直连不稳时同步项目仓库取用，勿从其他分支/来源拼凑。
- 校验方法：对 pristine 上游 main `6fd82cba` 的 `unirl/` 应用补丁（`patch -p1`），然后 `unirl/train/stack/unified.py` 应满足：sha256 = `9a55deabdf1821acbfb57d36bb1377418a4880a652c8a28dd7e89163e541a4e3`、git blob = `2cced46552f98e8d44cf57df9f91d46945902f33`（后者与 fork head `04101ad5` 的 blob 一致，M1 已预核，**M5 已在服务器双向复核通过**）。任一不符即停，勿手改凑数。

**L0 真实模块生命周期冒烟（CPU/单卡即可，核心交付，不依赖大模型权重）**：
把 M1 的"提取逻辑 exec"升级为**真实模块验证**——从打补丁后的真实 `unirl.train.stack.unified` 取 `train_track` 方法体（绑定到替身 self 或最小构造的真实类实例），协作者用替身：
- `fsdp_backend.model`：小 nn.Module（如单层 Linear），包装记录 `eval()`/`train()` 调用序列；
- ar/image 两个算法三种替身：(a) 不 override `prepare_part` 且 `recomputes_anchor=False`（复刻 shipped GRPO/FlowGRPO/BagelFlowUniGRPO 语义）；(b) 记录型 `prepare_part`（复刻 PPO/DiffusionOPD override 语义）；(c) `prepare_part` 抛异常的失败替身；
- `prepare_segment_anchors` 用真实函数（anchor.py）；Part/Plan 等构造依赖过重时允许最小替身，逐项记录哪些是替身。
- 断言：① 调用序 = eval → freeze(ar) → prepare_part(ar) → freeze(image) → prepare_part(image) → train；② (c) 抛异常后 `train()` 仍被恢复且异常上抛；③ (a) 两个 part 原样返回（shipped 配方零行为变化的模块级证据）；④ (b) 两个算法的 `prepare_part` 各被调恰一次。
- 记录：Python/torch 版本、命令、完整输出。不需要下载任何模型权重。

**L1 e2e 统一训练冒烟（先做可行性判定，大概率待资源）**：
shipped 统一配方的真实门槛（源码核实 2026-10-23）：`bagel_trainside_unigrpo.yaml` 要 num_devices=32 多机 + BAGEL-7B-MoT 权重 + fp32 master；`hi3_it2i.yaml`/`hi3_vllmomni*.yaml` 要 num_devices=8 + HunyuanImage-3.0-Instruct（MoE，权重体量远超单卡 24 GiB）。**注意 `hi3_trainside_t2i.yaml` 自带头部注释明确 "No AR rollout, no AR algorithm, no unified two-track stack"——不是统一栈载体**；`examples/pe/*replay*` 全走单模型栈（TrainStack 本来就正确），对本修复无验证意义。
L1 前置判定：服务器能否整段占用 ≥8 卡 + 是否有获准的 Bagel/HI3 权重与存储。任一不满足 → 记录"待资源/待权重"及恢复条件即停；不下载未授权权重、不为 L1 新租算力、不打断其他会话任务（14:37 定时轮记录 GPU 0/1/7 仍被占满）。
**为什么不能用"小模型 + 少数几张卡"替代 L1**（2026-10-23 补）：统一栈算法与"AR+扩散双部件"模型结构强耦合（共享骨干、图像 VAE latent、mRoPE 等），上游没有 tiny 统一模型、也没有 tests/ harness；自建随机权重小模型等于自造验证台，rollout 引擎链路还要为它单独打通，工程量不小且生命周期断言价值与 L0 重叠（L0 的替身控制更精确）。shipped 配方回归与 replay 数值验收只有真实权重才有意义；且即便只看显存，14B 级全参训练的 AdamW 状态（fp32 master + m + v ≈ 12 字节/参数 ≈ 168 GiB）在 3×24 GiB 上放不下——这正是 bagel 配方要求 32 设备的原因。故 L1 维持"待 ≥8 卡 + 获准权重"，不降级为小模型替代。
若条件满足，按序执行：
1. **正向回归（shipped rollout-sourced）**：任一统一配方按其注释的最小化改法（batch/steps 调小）跑 2 个 global step，打补丁 vs 不打补丁各一遍，逐指标对比 loss / first-update ratio / clip fraction——rollout-sourced 理论零差异，容差内 PASS；出现差异先排查 nondeterminism/版本漂移再上报。
2. **生命周期观测**：e2e 里确认 eval→train 切换真实发生（临时调试日志或 profiler 区间核对，结束时还原，不改语义）。
3. **replay 数值验收（核心验收，视算法支持而定）**：翻 `old_logp_source: replay` 前先核对该统一算法是否支持（`recomputes_anchor` 语义与配置管道；shipped 统一算法均为 rollout-sourced，replay 需自建最小配置）。支持则对照打补丁前后 first-update ratio / clip fraction / loss curve（修复前锚在 `train()` 模式算、修复后 `eval()` 冻结——#523 的核心数值差异点）；不支持则记录"该路径需算法改动、超出本修复范围"，不自行改算法凑验收。
4. 负向路径由 L0 替身 (c) 覆盖，e2e 不必人为注入故障。

**停止条件**：L0 完成且 L1 判定为"待资源/待权重"即构成可回写的闭环；不得为凑 L1 而放宽物料校验、混装 vllm/sglang extra、或用单栈配方冒充统一栈验证。

## DeepInfer 硬件适配核对与最小冒烟

2026-10-23 EPLB精度参考（不另排队/改移植方案）：[#212](https://github.com/deepseek-ai/DeepInfer/pull/212)读取open draft，head`3b950ebc0b66fd507ecdff02ef3193136b2fa5b9`、base/祖先`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`；[M1原笔记](../notebook/frameworks/deepinfer-initial-survey.md#长历史的公共基线与峰值下降量)原32项显式CPU pytest对照main26过6断言失败/#191 30过2失败/#212全过，长历史公共基线会吞掉0.5收益，非任意范围证明。CPU算法学习已收束，不把draft与#191自动拼接或应用。若M5原任务今后确需选择该policy且已有匹配获准环境，最小下一步是独立核目标后端/dtype的原policy测试与实际重排映射/权重一致性，再讨论性能；缺匹配build、设备或权重/通信证据时记录待条件，需新编译/部署/下载或干预原任务就停止。M1未跑设备/真实shuffle；这不是M5已接单或新增训练任务，原优先级和停止条件不变。

状态：M5 已完成第二段 4090 真机验证（2026-10-22），见[专题结果](../notebook/frameworks/deepinfer-civilian-port.md#移植冒烟第二段2026-10-22m5-服务器-4090-1-实测)与[实验材料](deepinfer-civilian-port-stage2/readme.md)：基线774ce244下四 kernel 包 stub + 架构检查放宽后 **llm_engine 等 20 个核心模块 import 全通**；tilelang 0.1.15 支持 SM89（saxpy/sync_grid 编译运行 OK）但 **43 个 cuda tilelang kernel 全部显式使用 PDL（Hopper+ 专属），SM89 编译抛 PDL is not supported——需逐个去 PDL 改写**；LLMEngine 实例化停在数据前提（需 V4 架构 config/权重）。**2026-10-22 续（M5）**：PDL 堵点源码级定位——根因在 tilelang 编译管线 `MarkCudaSyncCalls(have_pdl(target))`（`contrib/nvcc.py` 仅 SM90+ 返回 True，`lower_pdl.cc:44` 在 SM89 遇 `pdl_sync`/`pdl_trigger` 即 `ICHECK` 抛错，无配置可关）；语义映射确认 `pdl_trigger` 可安全删除、`pdl_sync` 用 `T.sync_grid()` 替代（顺序 launch 语义）；**5 个真实 kernel 文本级去 PDL 后 SM89 编译全过，其中 norm 与 torch RMSNorm 数值对照 max err=0.0（PASS）**——「去 PDL 改写」是机械工程（28/30 源文件、56 处 pdl_sync+8 处 pdl_trigger），代价是失去 kernel 间 PDL 重叠（性能未测）。**2026-10-22 再续（M5）**：`probe_pdlfree_numeric.py` 给其余 4 个 kernel 补数值对照（[实验材料](deepinfer-civilian-port-stage2/readme.md#数值对照扩展2026-10-22-再续m5)），**5/5 变体数值全过**——kv_index_cat 拷贝+偏移加 eq、load_kv_index 索引展开 eq、dflash_input_store hc_mult=1/2 max err=0.0、fused_sample（最复杂：两级 launch + lse_workspace 跨 launch 共享）logprobs vs log_softmax max err=1e-6、lse0/lse1 err=0.0、candidate 合法。改写语义保持从 norm 扩展到含最复杂 kernel 的 5 个。**2026-10-22 三续（M5）**：`scan_pdlfree_all.py` 全量改写编译扫查——28/28 个 PDL 文件、**43/43 个 `_tl` 工厂全部文本级去 PDL 变体在 SM89 编译通过**（含 fused_transform 全 kwonly 参数、load/store_kv_cache 的 KVCacheFormat 常量、state_cache 条件启用分支 enable 参数；替换计数核对 51 处 sync_grid 与改写前 pdl_sync 调用逐一相等、8 处 pdl_trigger→pass 均为独立语句、变体无 PDL 调用残留）。「去 PDL 改写」工程量与可机械性在真实硬件全量实证，PDL 非 4090 移植编译层隐藏障碍。**2026-10-22 四续（M5）**：load/store_kv_cache 两个每请求必经 kernel 用上游自带 GPU 测试矩阵数值对照——**pdlfree（pdl_sync→sync_grid）小用例 20/20 PASS、67890 大用例 10/10 FAIL（CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE）**：tilelang `sync_grid()` 生成 `cooperative_groups::this_grid().sync()`，**要求 cooperative launch（grid 全部 block 同时驻留），4090（128 SM）grid 上限约 1024-4096，大输入启动失败；上游 PDL 无此驻留限制**——sync_grid 替代引入运行期新硬边界（编译扫查小 grid 未暴露）。**nosync 变体（pdl_sync 直接删除）全矩阵 60/60 PASS（含全部大用例）**：两个 kernel 的 pdl_sync 是「等待 dependency launch 前序数据」，顺序 launch 下由 stream 顺序天然保证，**正确改写是删除而非替换 sync_grid**（详见 [stage2 readme](deepinfer-civilian-port-stage2/readme.md#loadstore_kv_cache-全矩阵数值验证--sync_grid-grid-上限发现2026-10-22-四续m5)）。**2026-10-22 五续（M5）**：把「删除而非替换」推广到全部 43 个 kernel——静态语义分类（28 个 PDL 源文件逐处核对）确认**全部 55 处 pdl_sync 均为依赖 launch 前序数据型（位于 kernel 开头），无一处是同一 launch 内跨 block 通信（那才需要 sync_grid）**，8 处 pdl_trigger 均为提前释放下游；`gen_nosync_variants.py` 生成全量 nosync 变体（删除 pdl_sync、pdl_trigger→pass），`scan_nosync_all.py` **43/43 SM89 编译全过**（重跑确认可复现），`probe_nosync_numeric.py`/`probe_norm_nosync.py` **7 个 kernel nosync 数值保持**（norm max err=0.0、kv_index_cat/load_kv_index eq、dflash_input_store max err=0.0、fused_sample logprobs err=1e-6、load/store 全矩阵 60/60）。**改写规则收敛：全部 pdl_sync 删除（nosync）、pdl_trigger 删除；sync_grid 在本项目 43 个 kernel 中无适用用例（且受 cooperative-launch grid 上限约束）——三续的 51 处 sync_grid 替换为次优改写**（详见 [stage2 readme](deepinfer-civilian-port-stage2/readme.md#全量-nosync-改写验证4343-编译--7-kernel-数值保持2026-10-22-五续m5)）。**2026-10-22 六续（M5）**：把「其余 36 个 kernel 仅编译 + 分类」的缺口推进为**全量官方测试套件数值验收**——`run_nosync_upstream_tests.py` 把 `ops` 的 43 个 `*_tl` 工厂全部 monkeypatch 为 nosync 变体后跑 24 个 wrapper 文件内嵌官方 GPU 测试（排除 benchmark）：**1134 passed / 67 failed / 4 skipped（39 分 26 秒）**，67 个失败逐类核因、**无一归因于改写的数据竞争/逻辑错误**——23 OOM（gather 18 / random_sampling 3 / candidate_blocks 2，均为上游测试大输入在 24 GiB 单卡显存上限）+ 6 动态共享内存超限（sampler `remap_draft_topk`，kernel 需 ≈103 KB > 4090 的 99 KB 上限）+ 27 norm 间歇 OSError（tilelang `@jit` 取源码失败，机制为 hide/restore 变体文件使 linecache 失效；**单独重跑 norm 全部官方测试 392 passed / 37 skipped，数值完好**）+ 11 fused_transform `torch.equal` 1-ULP 级微差（norm-bf16 用例 116/1.53 亿元素、e4m3 用例 3/419 万元素，rel-err ~5e-3 = bf16 一个量化步长，散布而非聚集 → 非数据竞争；**pdlfree sync_grid 变体同用例直接 COOPERATIVE_LAUNCH_TOO_LARGE 无法运行，nosync 是唯一可运行改写形式**；1-ULP 差异是否改写引入需 SM100 原版对照定性，4090 无此可能、记录待测）。**改写规则经受全量官方数值对照，未发现依赖型分类错误导致的数据错乱**（详见 [stage2 readme](deepinfer-civilian-port-stage2/readme.md#全量上游测试套件验证1134-passed--67-failed-全分类2026-10-22-六续m5)）。**六续交叉复验（GPU 7，2026-10-22）**：同一脚本在 GPU 7 独立重跑 **1163 passed / 38 failed / 4 skipped（505.67s）**，失败类别与 1134/67 版完全同构（21 OOM + 6 动态 smem + 11 fused_transform 微差），27 个 norm OSError 未复现——佐证其间歇环境性；fused_transform 追加 norm-rope-bf16 n=32769 两遍同 seed 复验 n_diff=1803 两次完全一致（确定性，排除数据竞争）；结论不变（详见 [stage2 readme 交叉复验](deepinfer-civilian-port-stage2/readme.md#六续交叉复验独立重跑gpu-72026-10-22)）。昇腾侧 VPN 2026-10-22 掉线（100% 丢包）待用户恢复，SSH 密钥亦未配置。此处依据M5回写同步，不是M1新连接或重复实测；后续以M5专项接续为准。

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

抢占结束交接参考（2026-10-22，不另排队/改移植）：新[#211](https://github.com/deepseek-ai/DeepInfer/pull/211)，head `4432ebe41efd6c6cb263ddfc619ccb3e87883ff7`、base/祖先 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。[M1阅读](../notebook/frameworks/deepinfer-initial-survey.md#抢占在途结果的执行代次与结束交接)仅4文件compile/原测试8函数11组合静态计数，C++未编译、无原生执行。目的为后续独立研究ZOS时核旧执行结果归属和pending finalize；不与#188 prompt计分混同。原测试为CPU/Gloo+真实_C，不需模型，当前缺口首先是匹配原生build，不是GPU在线本身；只有M5自行选择且已有获准匹配构建时，才考虑隔离进程跑`tests/engine/test_preemption_recovery.py`，核epoch/重复处理拒绝、abort优先、终止前缀、下一schedule交接及资源计数。真实设备侧还缺sample事件/host缓冲归属和实际模型收益证据，CPU测试不能替代；仅在原获准ZOS工作需要时再决定观察。需重编/新环境/模型/预算或打断任务就停，不自动启PCR、耗尽真实KV、应用候选或另排矩阵；作者合成收益不当本项目吞吐，记录不代表接单。

PD 原生接口参考（2026-10-22，不另排队/采用方案）：新增 [#195](https://github.com/deepseek-ai/DeepInfer/pull/195)，head `229c58c326f3f3ba98959240f84c1c88029baa96`、base/祖先 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，仅3文件21行新增。目的为今后评估PD入口时核对队列与资源所有权；[M1源码结论](../notebook/frameworks/deepinfer-initial-survey.md#pd-队列交接与资源释放)：移出finished只归还MTP槽，KV/state留待显式free；入队三项元数据检查不认证传输完成，非forked项不克隆。M1仅完整stub AST/内存compile与声明检查，无_C/原生测试，本PR无新增测试文件；作者GB300/B200报告不是本机证据。若M5今后确需采用且已有匹配head的_C/runtime，可先核实际binding/import身份，再以最小原生序列记录队列、MTP槽、KV/state占用及导出完成后显式释放的顺序，不把计数合法当数据已到达。缺口是匹配build、设备及实际拷贝完成证据；若需重编/切环境、传输同步协议不明确、资源不足或影响原任务，先记录待条件，不自动编译、下载权重、启动#197或并行改方案。原优先级与停止条件不变，记录不代表M5已接单。

Host cache 参考（2026-10-22，沿原条目、不另排队）：已有[#144](https://github.com/deepseek-ai/DeepInfer/pull/144)，head `e1f71ecdf8f1e53224bf44eee6989dee9de5f28a`、base/祖先 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。[M1完整源码核对](../notebook/frameworks/deepinfer-initial-survey.md#host-cache-的可复用边界)：KV命中还要与ready state/replay上限相交，size_bytes只管KV页，另计pinned state池及默认16份GPU staging。目的为今后评估缓存时核对预算/恢复证据，不改变移植方案；M1仅4文件AST/内存compile，未跑原3项GPU测试。若M5独立决定采用且已有兼容CUDA/编译链，可先核固定池和staging预算，再运行原入口 `python -m pytest -q tests/hicache/test_host_cache.py`；它用真实copy kernel但FakeSeq/替换state kernel，不验证新block_table binding或真实SWA/模型恢复。原生接入仍缺匹配head的_C、compact/额外state配置与实际state/replay证据，不能外推4090/NPU兼容。需重编、额外权重/内存预算、新平台移植或影响原任务时，先记待条件，不自动JIT、下载、切环境或启动服务；作者性能报告非本项目结果，记录不代表接单。

聊天封装参考（2026-10-22，非新部署任务）：已有[#147](https://github.com/deepseek-ai/DeepInfer/pull/147)，head `aaf1e62c9eab1ce5bd260a594dfaea613760e89f`、base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，见[M1源码阅读](../notebook/frameworks/deepinfer-initial-survey.md#聊天服务的亲和路由与完成口径)。目的为将来评估服务时区分亲和路由、真实cache命中、前端完成和客户端交付；M1仅8项原纯affinity控制与语法检查，未启动HTTP/engine/设备。缺口是匹配_C/模型/DP+EP运行环境、获准端口及host-cache预算；作者4×GB300结果不能外推4090/NPU。只有M5独立决定研究服务、已有可用获准引擎时，才考虑最小文本请求，保留固定提交、rank、编码fallback、engine reason/响应reason、usage和客户端完成证据；原A/B是先bypass后cache、TTFT按首个非空content，不当首次冷缓存或完整协议验收。需新编译/权重/服务部署、条件不匹配或影响原任务就停止并记待条件，不自动运行server.sh、启端口、改变移植方案或另排队；原优先级不变，记录不代表接单。

M5 最小下一步：先按实际硬件查已有方案和精确 head，读清依赖、权重格式、显存及其原始测试入口；核对通过后只选一个可隔离、可停止的小范围兼容检查，再决定是否具备 offline inference 条件。保留初始失败点与配置；不要直接按旧专题的“换三个包/放宽检查”当成已验证操作，也不要先覆盖已有环境。

停止条件：没有匹配硬件、内存/数据预算不成立、测试入口不明确或需扩大范围时，记录原因和下一项缺口，不自动下载巨量权重、改写移植方案或进行长时间编译。完成标准是目标配置下可复现的实测记录或明确阻塞，不是必须产出性能数字。

2026-10-22 另一个参考入口（不另排队）：[#193 V4-Flash-0731 EP8 runner](https://github.com/deepseek-ai/DeepInfer/pull/193)，读取open、wangzhao-11a，head `a87157392233938be508170f44790b76c2a0510c`，base `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。作者报告8×910B4-1、真实W8A8权重、43层的短生成，并列出输出逆RoPE和逐head无权重RMSNorm的修正经验；这是0731参考路径，不是#192的4.1单卡配置或生产engine支持。每token重算全序列、无paged KV，稀疏indexer未接入，不能外推吞吐/910C兼容。M1只核正文/版本，未审实现或复跑；供M5在原任务独立判断，不自动应用、下载权重/切环境、采用作者命令或打断现有实验，原停止条件不变。

## DeepInfer mock 权重链路跑通（2026-10-23 用户授权推进，除真实权重环节）

用户目标（2026-10-23 明确）：**demo 级（能正常推理即可）**，先造 mock 权重跑通链路，调通后再上真实权重调精度。除「真实权重下载」环节外，其他环节授权 M5 定时任务推进闭环；本条目为交接。

**背景与架构结论（2026-10-23 M1 已核实源码）**：DeepInfer 的 L1 调度（LLMEngine/Scheduler/Worker）是框架侧不耦合模型，但 L2 缓存（MLA 布局）、L3 模型执行（v4_modules/decoder_layer/HC）、L4 算子（deep_gemm/deep_ep/flash_mla）全部深度绑定 DeepSeek-V4.1——**不是通用模型框架**。`ModelRegistry._MODELS` 只有 `DeepseekV41ForCausalLM` 一项；`load_hf_model_config` 强制 `model_type=deepseek_v41` + fp8/fp4 量化 + text_config/vl_aligner。**Qwen 等小模型无法直接适配**（需整套 dense 后端，工程量等于移植半个 vLLM）。

**mock 方案可行性（源码级确认）**：
- `DeepseekForCausalLM.load_model` 的 `skip_load=True` 走 `HFCoverage()`（不读权重文件），`build_model_on_meta` 随机初始化——**支持纯 mock**
- `load_processor_config` 只读 config.json（要求 `model_type=deepseek_v41` + `vision_config` 几何字段），不需要真实权重
- 因此只需**一个伪造的 config.json**（V4.1 完整结构 + 极小尺寸），指向本地任意目录即可

**M5 执行步骤**：
1. **拉 config 模板**：从 hf-mirror 拉真实 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 的 config.json（KB 级，即使慢也只需一次）作结构模板；同时拉 processor_config.json / tokenizer.json（小文件）
2. **构造 mock 模型目录**：`/home/zxw/deepinfer-mock/` 放伪造 config.json——保留 V4.1 全部字段结构（model_type/architectures/dtype/quantization_config/text_config 全部字段/vision_config/vl_aligner 相关），**尺寸调极小**（num_hidden_layers=2、hidden_size=128、vocab_size=1000、num_attention_heads 按比例、moe 专家数=1、num_nextn_predict_layers=0 等）；`_assert_supported` 要求 hidden_act=silu / attention_bias=false / attention_dropout=0 / num_key_value_heads=1 / dtype=bfloat16
3. **跑通 LLMEngine 初始化 + prefill/decode**：`skip_load=True`，验证 L1 调度 + L2 缓存分配 + L3 前向 + 采样器全链路；预期输出随机 token（无意义但链路通）
4. **记录**：版本、命令、config 全文、日志、失败点；与 stage2 的 43/43 编译 + 1134 测试衔接

**边界与停止条件**：mock 只验证「链路能跑」，输出无意义；「真实推理出文本」仍需真实权重（约 300GB，磁盘剩 109G + hf-mirror 慢，待用户决策磁盘清理/下载路径——**真实权重环节不归本条目**）。mock config 若有字段缺失/校验失败，逐项补齐 `load_hf_model_config` 的取值字段即可；不下载巨量权重、不编译整套 DeepInfer、不打断现有任务。

**✅ M5 完成（2026-10-23 本会话，4090-1 GPU 0）**：mock 权重链路**完整跑通**——LLMEngine 初始化（含 CUDA graph 捕获 5.8s）+ prefill/decode 全链路真实执行，`generate` 返回随机文本（`'的事儿 referral渗透 Sloan制造的\x1b rigurede'`，mock 权重无意义为预期），显存 0.79 GiB allocated / 1.32 GiB reserved，8 token 生成约 0.21s。最终复现命令与证据见 [experiments/deepinfer-mock-chain/readme.md#最终复现](deepinfer-mock-chain/readme.md)（`DEEPINFER_USE_NEXTN_DECODE=0` + `DEEPINFER_USE_MULTIMODAL_POOL=0` + token id 输入 < 1000 + 自定义 MASTER_PORT）；服务器存档 `/home/zxw/deepinfer-mock/README-mock.md`（含环境改动清单：28 个 nosync kernel 覆盖、deepseek.py dummy sf dtype 修复 + FP4 packing 跳过、runtime_config PDL 门控、stub torch 回退）。**真实权重环节**（约 300GB、磁盘剩 109G、hf-mirror 慢）仍待用户决策磁盘清理/下载路径，不归本条目。结构堵点与 stub 设计（含本会话早段迭代记录）见 [experiments/deepinfer-mock-chain/readme.md](deepinfer-mock-chain/readme.md)。

**独立复验（13:07 轮，4090-1 GPU 7，文本提示路径）**：另一条等价 PASS——mock config `vocab_size=130000`（覆盖真实 tokenizer 词表 54994 + image_token_id 129264）后可直接用真实 tokenizer 的**文本提示**跑通（无需手造 token id）：`LLMEngine constructed OK` → `GENERATED OUTPUT 'unch旅游景区 Ballet sama imaginationcheid谧-random'` → `MOCK CHAIN COMPLETED OK`（EXIT=0，GPU 7 空闲运行）。stub 差异点：`_dq`/`_dequant_v41` 加 `nan_to_num`（随机 e4m3 权重 NaN 码型进 logits 触发 fused_sample「Invalid logits」设备断言——即上文 CUDA_ERROR_ASSERT 的根因之一）、`mega_moe_compute` einsum 向量化（兼容 [hidden,2*inter] 转置布局）、`flash_mla_with_kvcache` 按 decode 调用方契约返回、`paged_mqa_logits` 先 gather 后解包。脚本 [run_mock_chain.py](experiments/deepinfer-mock-chain/run_mock_chain.py) + 日志 [mock_chain_run23.log](experiments/deepinfer-mock-chain/mock_chain_run23.log)。

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

新MXFP8检查参考（2026-10-22 M5登记、10-23 M1完成[关系/指标哨兵核对](../notebook/frameworks/verl-quant-layer-audit.md#新版运行期检查与指标哨兵)，不接单/另排队）：[#8327](https://github.com/verl-project/verl/pull/8327)读取open，head `a2012b598e2e5eabc9a79847ed0a245dc73bc478`，祖先`d1390eea`，24文件4442增7删、六个CPU测试文件（修正原登记的5个）。作者明确替代#8132等旧拆分、基于#8306/#8308；M1只执行原纯Pythonsentinel 20参数例/5控制，无框架/设备导入，不是作者150例或量化验收。三检查默认warn；实际probe可能skip、audit可能no verdict、sentinel配置失败可缓存disabled，须保留mode/enabled/覆盖日志。作者已追加cf32f6ae的1×B200正常/坏scale/变化权重短跑（head只多文档），不是M5结果。若M5后续独立选择且已有匹配获准依赖/模型/硬件，先重核完整版本与backend，再区分硬拒绝、数值probe、层审计和指标；实际graph replay、权重内容与默认backend/ignore分歧仍需自己的证据。缺资源、需新环境或影响原任务就保留未测，不拼接旧路线、自动应用、注入生产故障或启动训练。

ModelOpt忽略规则参考（2026-10-22，同一量化入口不另排队）：已有[#8329](https://github.com/verl-project/verl/pull/8329)，head `5c7e2d2ed49bd9f448043883d96b6e191e187e95`、base `16ec093cc4a235c603c29aed1f2c4e9f551fab48`，**已于 2026-10-22 合入 verl main（merge commit `5d09d065`，本条 M5 定时轮 gh 核验）**，见[M1阅读](../notebook/frameworks/verl-quant-layer-audit.md#modelopt-模块忽略与子量化器)。目的为将来核对actor实际fake-quant选层，不仅比较导出/加载字节；M1原helper两版各11组标准库名称控制、4文件compile，缺ModelOpt/CT，未跑原34测试或量化/梯度/设备。缺口是匹配依赖与已选QAT路线，作者H100单seed10步不是本项目或4090结果。若M5独立选择且已有获准依赖环境，先核head和ModelOpt/CT版本，再考虑原入口 `python -m pytest -q tests/utils/test_modelopt_mxfp8_on_gpu.py`，分开记录真实is_enabled/兄弟隔离/BF16前反向；文件gpu后缀不保证每例都在CUDA，不为名字分配GPU。额外训练需模型/硬件/预算另获准，若缺条件、需切环境或影响当前任务就记待条件；不自动应用、拼接#8132/#8306、安装或开展H100训练，不将作者34通过当本机结果/CI成功，不代表接单。

停止条件：无匹配设备/TE/引擎或模型授权就写待资源/资料，不大规模下载或长训练，不用fake workspace替代真机验收。完成标准是明确版本与路由的可追溯结果或阻塞；选层相符不等于数值相同、回滚安全或#8250根因，不能凭此发布修复/PR。

## verl 异步策略一致性补证

LoRA 接收存储参考（2026-10-23，非新增训练队列）：[#8352](https://github.com/verl-project/verl/pull/8352) 已于 22:35 +08 合并为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`；测试 head `7b9b25a551eb8cded7d0dda0c6b8437bb720b796`、base/祖先 `a4b9354968705a441fb7da7367924b15b902e771`，合并处两个相关源文件与 head 相同。[M1 原暂存语句 CPU 控制](../notebook/verl-v1-checkpoint-engines-deep-reading.md#lora-分桶接收的主机存储所有权)只核真实小 tensor 的布局/cast/借用源覆盖与字典所有权，未跑完整 callback、IPC、锁页、loader 或 GPU。若日后 M5 独立选择且已有获准匹配 LoRA 同步任务，最小固定安装 vLLM/torch、resolved lora_dtype 和 pin policy，区分逐桶自有存储、整 adapter 累计、真实 D2H/cast、loader 是否复用及完整确认链；接收侧微基准与整轮延迟/峰值分别记录。缺环境/授权模型或需升级部署、影响专项则保留未测停止，不自动升级/新增必跑矩阵或将作者 GB300 收益外推 4090；参考非接单，不改 #8313/#523/AReaL/DeepInfer。

文件系统同步参考（2026-10-23，非新增训练队列）：已有[#8355](https://github.com/verl-project/verl/pull/8355) head `7d981cde954fce079d8c973379738d6e8c258ac2`、共同祖先 `2bc6ac46043e31855cea194236ff7548c91817c8`（目标base另为a4b93549）；[M1本地IO](../notebook/verl-v1-checkpoint-engines-deep-reading.md#文件系统权重同步的发布与清理)仅原export/复制/CPU receiver及直接finalize方法，writer顺序独立调用、未跑send/collective/manager/FUSE。若日后M5独立选择且已有获准匹配的共享挂载与FSDP/vLLM任务，最小核各rank参数流/分片边界一致、复制/barrier/index跨客户端可见性、server安装确认后清理和失败目录隔离；分开记录传输与恢复协议、主机暂存与设备峰值。无现成匹配挂载/授权模型/资源，需新部署下载或影响专项即保留未测，不部署FUSE/切后端/开训练或改#8313/#8250；参考非接单，作者A100报告非本项目。

二维logprob分块参考（2026-10-23，非新增训练队列）：已有[#8351](https://github.com/verl-project/verl/pull/8351) head `285e2dd83d131a324b440ed5464c0dc61e3cf0b4`、base/祖先 `a4b9354968705a441fb7da7367924b15b902e771`；[M1 CPU移植范围](../notebook/frameworks/verl-6572-full-determinism-source-reading.md#二维低精度-logprob-的分块预算)原17测试体绑定cpu通过、补充storage/图节点与形状控制，无原生pytest/dispatcher/引擎或GPU/NPU执行。32 MiB为单块目标非整步峰值，单行超预算仍取一行，反向可保存全部块；作者GB300计时非本项目结果。若M5日后已有获准匹配fallback观测且独立采用，先确认实际选中v2（禁flash不覆盖NPU CE），固定dtype/二维布局/词表及torch/backend，再区分no_grad与反向的峰值/调用数和相同输入值/梯度；缺匹配环境或需新安装/模型/训练、影响专项即记未测停止。不自动启环境变量/应用候选或新增必跑矩阵，不认定#8250根因，本参考非接单。

LoRA-only恢复参考（2026-10-23，非新增训练任务）：已有[#8350](https://github.com/verl-project/verl/pull/8350) head `19b89e7100c06d7731247545274b117887ad8d38`、base/祖先 `a4b9354968705a441fb7da7367924b15b902e771`；[M1原分支八组CPU控制](../notebook/frameworks/verl-checkpoint-management-reading.md#lora-only-的键完整性与加载副作用)用真实小nn.Module确认缺base允许/缺adapter候选拒绝，以及抛错前已有参数写入，无完整manager/分片I/O/FSDP/PEFT/优化器或训练。若M5日后已有获准匹配LoRA恢复选题且独立采用，最小固定base checkpoint身份、adapter配置/键和实际分片，核各rank恢复的adapter值及失败后是否退出；不要将缺键拒绝当回滚或base身份验证。缺授权权重/匹配后端、需下载部署或干预专项即记未测停止，不自动应用/新排训练或认定#8250根因，AReaL/#8313/#523/DeepInfer分工不变；本记录非接单。

Bypass配置参考（2026-10-23，非新增训练队列）：已有[#8348](https://github.com/verl-project/verl/pull/8348) head `4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349`、base/祖先 `a4b9354968705a441fb7da7367924b15b902e771`；[M1原19配置例](../notebook/frameworks/verl-async-logprob-provenance.md#bypass-字段替换与-worker-的-loss-配置)两次通过，真实OmegaConf但仅4函数AST隔离，5个padding张量例未跑，无worker/Ray/FSDP/训练。若M5日后已有获准匹配bypass选题且独立采用，最小先核创建前配置与worker实际loss/校正镜像，再区分batch的old/rollout字段相等和真正的IS/RS执行；padding-only统计、跨rank归约及entropy反向需另核实际后端。作者4GPU短跑非本项目验收；缺匹配环境/授权权重、需下载部署或影响专项即记未测停止，不自动应用或开训练，不认作#8250根因。本记录不是接单，不改AReaL/#8313/#523/DeepInfer任务。

初始化 buffer 精度参考（2026-10-23，非新增训练队列）：已有 [#8189](https://github.com/verl-project/verl/pull/8189) head `4e830c72bd0eb2b0d6ecfbd4c2c4144040637324`、base/祖先 `1d367a4f459a43768f76312c0d3430a99ec46b8c`；[M1 原6测试体与4组CPU控制](../notebook/frameworks/verl-fsdp2-source-deep-reading.md#参数转换与缓冲区精度)确认只转换参数可避免破坏新鲜FP32 buffer，不重建已舍入值或处理完整keep-list/训练状态。无HF RoPE/FSDP/model执行，#8154作者CUDA固定token replay不是本项目结果。若M5日后独立选择此边界且已有获准权重/匹配后端，可先固定HF/torch/engine版本及model_dtype、FSDP1/2策略，记录构造后/转换后/包装后buffer实际值，再在同权重/token/position下比较logp与恢复控制；不先启长训练。缺授权模型/匹配环境、需下载部署或干预专项就停止记未测，不自动应用、拼接#7165/量化方案或认定#8250根因，本参考非接单。

流式准入参考（2026-10-23，非新增训练队列）：已有draft[#8203](https://github.com/verl-project/verl/pull/8203)head`b6880198eadd813272290fb1c98ba15e42317071`、base/祖先`18cfb74852a053a27dcf301d3318a093be81b628`；[M1源码阅读](../notebook/frameworks/verl-6790-async-trainer-reading.md#会话水位与未消费组上限)仅23模块AST/compile及4原schema参数例，无actor/Ray/TQ/设备验收，作者GPU e2e仍TODO。若M5将来已有获准匹配V1/TransferQueue选题并独立选用此破坏性draft，最小分别记录会话槽、pending/running/terminal组、已采样在训轨迹字节与实际行为版本；比较同负载准入与训练节奏，不用水位/组数代替TQ/GPU峰值或因果证据。缺原私有数据/模型、匹配后端/资源或需升级部署、干预专项即保留未测停止；不自动套补丁、改#8250环境或新增必跑矩阵，记录不代表接单。

状态：待数据与运行环境核对；不是 GPU 空闲后即可复现原报告的任务。

入口：[当前诊断边界](../open-source/verl/candidate-8250-async-entropy-collapse.md)、[策略来源与最小观测契约](../notebook/frameworks/verl-async-logprob-provenance.md)、[Issue #8250](https://github.com/verl-project/verl/issues/8250)。2026-10-21 API 读取仍 open、0 评论；没有复查全仓库修复覆盖。报告版本 `1876b06d0a3e4e71e06230be10af14492ca8a75b` 外加私有修改，私有数据/模型不可由项目公开材料还原，实际 vLLM 安装版本仍未知。

M1 已撤回确定因果诊断。已有数值、路由与异常控制未运行模型、原生权重同步或真实训练；没有证明 sync/cache、mask、reward 或数值路径是哪一个根因。

Actor公式对齐参考（2026-10-23，非新增训练队列）：已有[#8026](https://github.com/verl-project/verl/pull/8026)head`6a7110b42ae45068b8c7d1c20c56db841c5283af`、目标dc709afb/祖先831c5d86；[M1原4数学测试体](../notebook/frameworks/verl-6572-full-determinism-source-reading.md#actor-前向与-logprob-公式的一致性)及3组CPU控制通过，仅同logits公式/局部VJP/保存引用，13模块仅compile，无vLLM override/engine/模型。作者2×H20四层Qwen3-Omni额外算子对齐后零mismatch不认证4090/完整反向，gradient norm仍有差。若M5已有获准匹配对齐选题并独立选用，先固定vLLM/模型算子/权重、温度及支持、非融合路径与入口环境生效时机，再分别核同logits公式、actor/rollout前向及更新后权重同步；缺现成匹配依赖/模型、需新部署/下载或影响专项即记未测停止。不自动装vLLM/启开关/应用候选、重开P9或#8250诊断，也不新增必跑矩阵；此参考非接单。

返回字段参考（2026-10-23，非新增算力队列）：已有[#8341](https://github.com/verl-project/verl/pull/8341)，head `0cede26bf2c0d7433af0cba5c2c38ce278e6b07e`、base/祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`；[M1原2测试体CPU对照](../notebook/frameworks/verl-async-logprob-provenance.md#resume-返回值与扩展字段合并)确认基线丢text/head保留，另3组控制区分缺键/None、浅替换和调用隔离。真实Pydantic但包壳/原作者分段替身，无原生导入、Ray、backend或AgentLoop。仅当M5已有获准匹配工作涉及自定义生成字段且独立选用时，先固定安装head及producer/consumer字段语义，再观察单段与中断续跑的实际payload、累计token及消费结果；缺匹配模型/后端、需新部署/下载或影响专项就记未测停止，不为此开训练/重排队或自动应用。保留字段非任意多模态片段自动拼接，无证据连接#8250根因；此参考非接单。

教师生成参考（2026-10-23，不新增训练队列）：draft [#8335](https://github.com/verl-project/verl/pull/8335)，head `8029bfb5fc5b8b6034704f9c062610613f1bf05c`、base/祖先 `fb55b9288560f0c7bf380c1c1c3b45ea170538b0`；[M1源码核对](../notebook/frameworks/verl-veomni-opd-source-reading.md#教师生成与蒸馏目标)仅16模块compile/3文件字节及2方法AST未变，无原测试/模型运行。按submission选择student或teacher，教师生成后仍另行软top-k评分，非纯学生on-policy；hard guard不认证文档所有不支持组合和tokenizer匹配。作者8×H200/17更新/48validation报告非本项目或4090验收。仅当M5日后独立选择这条同步V1/vLLM/单教师/纯文本监督蒸馏且已有获准依赖、权重与匹配词表/资源，才考虑固定source/step、生成和评分两请求、真实padding/版本标签、恢复后的选择及validation来自学生的有界观测；缺依赖或资源、需新模型/部署/预算或影响原任务则停止并记未测，不自动装环境、下载、应用draft或凑8卡。此参考不代表接单，与#8250无已证因果；既有GPU/AReaL/#8313/#523分工不变。

输出头语义参考（2026-10-23，非新增训练任务）：已有[#7991](https://github.com/verl-project/verl/pull/7991)，head `b145ec9b405ca93092c45b5f9e7995fb4604dbd2`、base/祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，[M1阅读](../notebook/frameworks/verl-fused-output-softcap.md)核通用fused分支拒绝真值softcap，非实现softcap或完整初始化回滚。完整原torch输出头36组CPU语义对照/FP64误差已记录；原24HF配置测试因缺Transformers未跑，首先是CPU导入依赖缺口，不是GPU不在线。仅当M5日后原有获准模型工作恰好涉及此语义时，独立核模型/text config、原forward、backend与temperature，对齐同hidden/label后再决定是否需要设备输出/梯度检查；缺匹配环境/权重或需影响原任务就停，不自动应用、下载或另排训练。没有证据表明#8250使用受影响模型/路径，不能据此重开因果诊断；此参考不代表接单。

可选观测入口（不另排队、不默认采用）：已有[verl #8315](https://github.com/verl-project/verl/pull/8315)，2026-10-21读取open，head `4d6b3eb8606e24c1d70d648e92a1114129ec6c62`、目标base619c397d、merge-basecd58f550，见[阶段tracing边界](../notebook/frameworks/verl-async-logprob-provenance.md#新版阶段-tracing-与证据关联)。M1仅4模块语法/11方法去decorator后AST一致检查，未运行rl-insight或引擎。它记录rank update_weight与actor memory阶段，但no-op也可有区间，sender返回非全体应用证明，未新增显存字节/参数身份校验。真实同步/设备观测由M5在既有任务条件满足时独立决定，非接单记录。

若已有获准的匹配观测环境，可先固定verl/rl-insight实际提交、包路径、版本及metric_log callable分派，核对rank/actor lane与同步step关联、是否实际执行/异常，并保留独立显存与安装证据。缺后端、需要新增数据外发或会改变原报告环境时先记待条件；不为这项参考自动升级环境、部署服务、重跑长训练或上传私有payload，阶段图不能替代下述配对观测。

2026-10-22版本补充（同一参考不另排队）：#8315当前head `0cea139623cc8913d2241d21223df5682474041c`、目标basef2923cbf/祖先cd58f550，仅比前head新增65行CPU测试及worker lane整数→rank_字符串。M1隔离原类/原12项日志测试：前后head均12过，祖先7过5预期失败；另核原生产lambda为int2→str rank_2。客户端为原Mock，未跑真实后端/接收/时间线查询；已有原测试不能认证生产接线或GPU同步。若M5原条件满足且独立选用，按实际head核lane类型、step及后端接收/显示，仍不新增监控部署或实验任务，也不把通过Mock测试当#8250根因证据。

可选NIXL拓扑参考（同一事项，不另排队）：[#8316](https://github.com/verl-project/verl/pull/8316)，2026-10-21读取open，head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`、目标basef2923cbf/祖先619c397d。[M1原拓扑及纯断言检查](../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑)每版20组：旧R=1长度失败、候选全过，未prepare/注册/连接/传输。只有已有获准观测实际选择NIXL且恰为单rollout checkpoint worker时，M5独立决定是否核对此候选，先固定安装版本、backend、actor/rollout worker数和metadata顺序，再分开记录建组、传输、应用及恢复点；并非一个replica就必为R=1。不为此新建网络/下载模型、强制改backend或重跑训练，缺匹配资源就记未测；没有证据表明#8250使用该backend或由此导致。

版本补充：#8316后续head `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`只新增1项原CPU测试，实现未变。M1隔离运行原测试体：基线失败、新版通过；没有原生导入/pytest收集或NIXL传输验收。无需另开资源任务，后续若选用仍以当时实际head和上述条件为准。

可选超时保护栏参考（2026-10-22，同一事项不另排队）：[#7045](https://github.com/verl-project/verl/pull/7045)，head `6f458f6d49076c647a42c00d240b5eaa7449fdb7`、目标base3f9d4c0d/祖先220e9039，见[M1源码与12组纯参数检查](../notebook/frameworks/verl-async-logprob-provenance.md#组初始化的等待超时与任务退出)。本机无Ray，未复跑原15项；当前head只给控制端init refs等待加上限，prepare/传输/finalize等不在此预算内，不取消RPC/回滚。作者2×A100退出清理结果针对旧da00403，当前head待复验。仅当M5已有获准且匹配的同步诊断环境、独立选择该方案时，先固定完整head、Ray/NCCL/launcher版本及有效timeout，分别记录控制端异常、远端/driver退出、设备资源释放与独立新进程恢复；缺资源或需改动/打断已有任务则记未测，不造生产挂死、不原地重试group、不切backend/下载模型/自动采用补丁，也不认定#8250由此导致。此参考不代表接单或M1已触发真机任务。

分桶失败传播参考（2026-10-23，同一事项不另排队）：已有[#7366](https://github.com/verl-project/verl/pull/7366)，head `48d91130abff6b6704aadac3f382e0c3d344de93`、base/祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，见[M1原8项CPU复查](../notebook/frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md#分桶接收失败与退出)。真实ZMQ/SHM进程测试两次通过，但设备入口为CPU facade；无CUDA IPC、NPU或vLLM实测，成功例只核名称不核tensor值。目的为日后同步诊断区分错误ACK、逐次超时、外层RPC退出与部分权重状态，非#8250根因证据。若M5独立选题且已有匹配获准环境，先固定torch/pyzmq/backend及named_tensors或delta路径、有效ms值，再决定运行原`tests/utils/test_bucketed_weight_transfer.py`的适用用例，分别保留pass/skip、值校验、异常/资源退出证据；真实callback恢复需独立进程，不在原训练注入故障或超时后盲目续用。缺软件栈/模型/设备、需新部署编译或影响原任务就停止并记未测；不自动应用、修改其方案、拉起训练或认定接单。

外部KV版本盐参考（2026-10-23，同一事项不另排队）：[#8323](https://github.com/verl-project/verl/pull/8323)，head `4e93e4761ddde2ea5bf9d3032b86b6c3844126fd`、目标base16ec093c/祖先f2923cbf，[M1源码阅读](../notebook/frameworks/prefix-cache-failure-modes.md#权重版本与外部缓存)仅11完整模块compile，原CPU/设备套件未跑。默认reset不变；salt要求namespace/step真正区分权重且store保留带盐key，旧条目依eviction回收，非安装认证或回滚。作者单卡三后端GPU用例实际改标签/测命中，没有传入另一组权重；HiCache file旧KV复用报告为另行one-off，不挪为本项目。若M5日后独立选择且已有匹配获准环境，先固定backend/connector实现及软件版本、run namespace/真实权重身份、step、门和外部层，再决定是否跑适用原测试；须把标签/冷热命中与真实安装后配对输出分开，保留pass/skip和store eviction/容量条件。缺依赖/外层资源/模型授权、需新部署或影响原任务即停记未测，不拼接其他缓存候选、不自动切salt、全局清共享store、照搬CI的全局ray stop、下载模型或启动训练，也不认定#8250由此导致；此参考不代表接单。

M5 最小下一步：先判断是否有获准使用的对应配置、完整 token/mask、checkpoint/版本身份和实际环境；条件齐备后按既有观测契约采集训练 actor、干净重载、live rollout 的配对观测，明确 raw/processed 概率与采样处理，保留逐 token 残差及重复基线。需要同步证据时区分发送、接收、应用、cache、版本标记及恢复生成；不由单个版本号或均值接近零认证一致。

停止条件：缺私有材料/配对身份就记录“待资料”，不启动原报告的长训练，也不以公共 toy 或另一模型称已复现 #8250。对照有差异只能缩小范围；没有原始观测前，不设计新的修复、不发布旧诊断稿或承诺治愈阈值。完成标准是可追溯的配对证据，或明确缺失材料清单。

## verl TransferQueue 启动资源参考

状态：只留日后独立选题参考，不加入自动训练队列或代表接单。已有[#8340](https://github.com/verl-project/verl/pull/8340)，head `881fb96937f5ddfae5d894bb9a48538d1b049e1e`、base/祖先dc709afb；作者CPU/Ray复现使用TransferQueue `5fe4e49fe74aef544fe54da1fbeabf0b36d497eb`及Ray2.59.0。[M1原3配置测试/四类generated配对](../notebook/frameworks/verl-transferqueue-worker-lifecycle-deep-reading.md)通过，未运行Ray/tq.init。目的为区分U+1逻辑CPU基础预留、实际硬亲和节点和其他actor配额；不是设备离线或GPU性能结论。

若M5已有获准TQ环境且恰需启动诊断，先固定verl/TQ/Ray版本与实际backend，核trainer.nnodes、最终U、候选alive节点/required_node_resource、逐节点可调度CPU及其他会话占用，再独立决定是否需要无模型的隔离CPU复现；保留actor状态、初始化完成/退出、命名控制器是否复用和清理证据。缺匹配环境/授权、需新集群部署/资源/模型或影响原任务即停记未测，不造生产PENDING、不全局ray stop、不自动应用候选或拉起训练。GPU/NPU和AReaL/#8313/#523原分工不变。

## verl Megatron 重算回收参考

状态：待M5独立选题与匹配环境，非新增必跑队列，不打断现有任务。[#8338](https://github.com/verl-project/verl/pull/8338)固定head `c099b0179cfa70987b7b75e5b1c9b685d39c4d5c`、目标base dc709afb/共同祖先 `d1390eea0c94467d4ae1387eeda9e2f78f8d73ea`；[M1源码/CPU入口](../notebook/fundamentals/gradient-checkpointing.md#重算叶子与共享存储)仅原两条清理语句的4组CPU所有权/梯度/延后反向控制和3文件compile，无完整patch、原5项GPU测试或显存/stream验收。

目的：若后续已有获准Megatron全重算工作，区分叶子引用留存与共享storage被截断，不能据MTP名字特判认证普通共享输入安全。缺口是匹配MCore/PyTorch/CUDA版本、原测试真实执行/skip、显存floor/peak与释放时序；作者0.18.0/torch2.11/H800和8卡MTP报告不外推不同MCore/后端或4090环境。只有M5独立选择且已有兼容获准环境时，才考虑隔离进程运行 `tests/models/test_mcore_recompute_reclaim.py`，保留5项具体结果/skip、梯度与stock对照；实际模型回收及跨stream/分布式条件另行判断，不按作者矩阵自动追加训练。缺依赖、需新编译/切环境/模型/预算或影响其他任务即停并记未测，不自动改环境、应用patch或在原训练制造非法访存；AReaL/#8313/#523及移植专项边界不变，本条不代表接单。

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
