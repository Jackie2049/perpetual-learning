# verl VeOmni OPD (On-Policy Distillation) Engine — Source-Level Analysis

> 2026-06-16 | PR #6072 (MERGED) | Author: hjshi84 | Merged: 2026-04-21
> Repository: volcengine/verl | PR title: "[veomni] feat: enable VeOmni engine for on-policy distillation"
> 190 additions, 8 deletions, 3 files changed
> Key reference paper: "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016)
> Related: DeepSpeed OPD #8027 (DRAFT OPEN), OPSD reference (HJSang/OPSD_OnPolicyDistillation)

2026-10-23 补充：[教师生成与蒸馏目标](#教师生成与蒸馏目标)以新draft #8335为准，区分生成方、软标签打分方和学生更新。下方June阅读、星级及硬件建议不是该实验路线的就绪或性能结论。

2026-10-25 接续：[主线与 GKD 的损失名称和支持域](#主线与-gkd-的损失名称和支持域)核#7975／已有draft #7983；纠正旧2.2节的k2／k3公式，区分采样估计、top-k归一化、tail bucket与JSD端点。配置名称相同不代表梯度或更新相同。

## Rollout 量化导出与字节布局

**导出流经过量化器，不等于已验证推理引擎能加载该布局；pass-through也不总是对象完全不变。** 新draft [#8369](https://github.com/verl-project/verl/pull/8369)给VeOmni补trainer-side rollout quantizer接口。2026-10-26读取OPEN／draft、作者xiaohong42，head `ca0aea265f4bc4dab927a106cc7f9d43fc201337`，父版／base／共同祖先与当前main均为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`；两文件60增2删。唯一bot评论表示draft未自动review，reviews／行内评论空，不当人工验收或无人负责的新机会。

### 接口与两条导出路径

主线[init_model](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine_workers.py)在actor路径构造checkpoint engine，再由dispatcher选量化器；返回非None才调用engine.set_rollout_weight_quantizer。未实现该接口的引擎继承[BaseEngine](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/base.py)的NotImplementedError。它不要求开启QAT：原checkpoint的native quantization_config也可能选出FP8量化器。本轮没有运行init_model或复现作者的整机启动错误。

[候选VeOmni实现](https://github.com/verl-project/verl/blob/ca0aea265f4bc4dab927a106cc7f9d43fc201337/verl/workers/engine/veomni/transformer_impl.py)在构造器记None，setter保存callable；无量化器时helper原样返回输入，否则调用它。get_per_tensor_param的converter.export_weights与state_dict生成器两条返回流均经helper，第二个返回值仍None。AST对照移除恰好两处wrapper后，完整导出方法与父版相同；构造器移除新增赋None后也相同。原offload、DTensor收集及EP广播逻辑未改，不代表这些分支已由本机执行；shard/delta接口与LoRA支持也不能由这两处接线推出。

[原CPU测试](https://github.com/verl-project/verl/blob/ca0aea265f4bc4dab927a106cc7f9d43fc201337/tests/workers/test_veomni_rollout_quantizer_on_cpu.py)只有一项，手动用object.__new__避开engine初始化：先检查无量化器时迭代器身份不变，再装把名字加.q的lambda，值仍是整数1／2。原测试体与断言经AST隔离，候选通过，父版因缺_quantize_for_rollout而AttributeError；这不是上段NotImplementedError的真实启动复现。原文件顶层的VeOmni MagicMock注册／整包导入未执行，也没调用完整get_per_tensor_param。

### Pass through 仍可重解释 expert 字节

固定主线[Fp8ConfigWeightQuantizer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/qat/compressed_tensors.py)按model_type包含deepseek_v4／deepseek_v41选passthrough。该分支对名字含.experts.且dtype为int8或float8_e8m0fnu的张量执行view(uint8)，其余直接yield；没有把这些expert重新作通用BF16→FP8压缩，但“直接转交”仍可能改变张量视图dtype。其他模型的标准native FP8路径另有dynamic、128×128及store_dtype限制，不是任意FP8配置通吃。

[复查脚本](../../experiments/verl-veomni-rollout-quantizer/verify.py)加载完整原量化器类，在真实torch2.14.1／CPU用一个四字节int8输入作控制：expert名下的[-128,-1,0,127]导出为uint8的[128,255,0,127]，data_ptr相同；同值非expert名保持原int8对象。字节未重新量化，负数显示改变来自同一bit pattern的解释方式。这个固定小例不验证E8M0分支、真实converter混合FP8／FP4布局或vLLM接收端。

Python3.12.13；六完整源文件版本AST／内存compile，源码SHA256、原测试及CPU结果见[JSON](../../experiments/verl-veomni-rollout-quantizer/result.json)，复跑语义一致。命令：`python -B experiments/verl-veomni-rollout-quantizer/verify.py /path/to/verl-git`，只读已有固定对象，不自动fetch或安装；未导入verl／VeOmni／Ray、运行模型或分布式导出。

PR正文当前报告组合#8370／#8371在2×8 MI355X完成三步，第四步和MI308X第一步仍running；这是作者外部报告，不是本机或单独此补丁的验收。真实格式／scale／加载及更新后数值缺口仅补[原M5条件参考](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不下载模型、部署AMD、应用候选或新增训练队列。

## Optimizer 覆盖与 CPU offload 默认

**配置被传到构建器，与优化器在真实卸载模型中节省了多少内存，是两种证据。** 新draft [verl #8370](https://github.com/verl-project/verl/pull/8370)将`optim.override_optimizer_config`交给VeOmni build_optimizer，并仅在`engine_config.enable_fsdp_offload`为真时补默认`fused=True`；显式`fused=False`保留。检查的是这个特定配置开关，不是运行时检查参数所在设备，也不能把它等同于所有名为optimizer_offload的路径。

2026-10-26固定head `44c97bcd4bd167b93462e3bc380bee6947a90e5f`，base／父版／共同祖先与当轮main均为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，ahead1／behind0、七文件120增1删。唯一普通评论是机器人跳过draft审查的通知，reviews／行内为空；不是已批准或已合并功能，不另写竞争补丁。

### 原五用例的接线对照

[复现脚本](../../experiments/verl-veomni-optimizer/verify.py)原样装载BaseConfig、engine／optimizer原dataclass、完整_build_optimizer与原测试体／fixture；仅engine类壳绕过分布式构造。builder保持原fixture的记录器，没有安装VeOmni或拿真实AdamW冒充VeOmni构建器。

| 原用例 | base 传给 builder | head 传给 builder | 原断言结果 base／head |
| --- | --- | --- | --- |
| 未offload、无覆盖 | 不含fused | 不含fused | 过／过 |
| 未offload、显式True | 不含fused | fused=True | 失败／过 |
| offload、无覆盖 | 不含fused | fused=True | 失败／过 |
| offload、显式False | 不含fused | fused=False | 失败／过 |
| offload、eps=1e−6 | 不含eps／fused | eps=1e−6、fused=True | 缺键失败／过 |

候选先复制覆盖字典，再setdefault；不是无条件覆盖用户设置。这里实跑四个原参数实例和一个原额外键测试，不是原生pytest收集、hook／distributed mesh或完整engine初始化。

锁文件固定VeOmni `10a26aca258f91a7af3e698cc0db8c6c51705989`。其[原构建器](https://github.com/ByteDance-Seed/VeOmni/blob/10a26aca258f91a7af3e698cc0db8c6c51705989/veomni/optim/optimizer.py#L459)的AdamW分支用`foreach = not fused`，再调用torch AdamW；这解释为何新fused键会改变实现选择。覆盖字典的接收方是**VeOmni构建器的签名**，不是任意torch AdamW关键字透传。本轮只读该依赖源码，未执行其参数分组、ExtraParallel或Muon分支。

### 小型数值对照不是性能复验

另直接在本机torch2.14.1运行一个FP32参数张量、257元素的一次AdamW更新，固定相同输入／梯度、lr=1e−3、betas=(0.9,0.999)、eps=1e−8、weight_decay=0.01，分别使用foreach和fused。参数逐位相等、最大差0；step／exp_avg／exp_avg_sq最大差也为0。这只认证这个CPU输入的一次更新，未用VeOmni、DTensor或FSDP2，**没有测耗时、host RSS或峰值内存**，不能推出所有形状／dtype／步数逐位等价，也不复现作者4GB参数的内存与速度表。

原PR的AMD整机表在读取时只有2×8 MI355X第一步有数值，第二步及MI308X第一步仍标running，且与[#8369](https://github.com/verl-project/verl/pull/8369)量化导出、[#8371](https://github.com/verl-project/verl/pull/8371)ROCm默认和外部依赖组合运行。它不是本机结果、两种架构均完成验收或单独#8370的收益归因，不据此推荐用户更换硬件／安装整套堆叠。

同一PR还新增`broadcast_model_weights_from_rank0=True`与`ep_sharded_stream_load=False`两字段并接入build_parallelize_model、更新生成YAML；这些路径本轮仅源码阅读，没有真实checkpoint读盘／分片／网络传输验证，不能由optimizer五项过认定其正确或提速。

[JSON](../../experiments/verl-veomni-optimizer/result.json)保存两版五用例的实际kwargs、九份原文件hash、小型AdamW差值和环境（Python3.12.13、pytest9.1.1）；两次输出语义一致，九完整Python模块AST／compile通过。复现运行 `python -B experiments/verl-veomni-optimizer/verify.py <已有固定对象的只读verl仓库>`，缺对象不自动联网。真机条件补[原M5 VeOmni参考](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不是新增训练队列；无匹配获准任务即保留未测，不探测GPU、部署AMD、下载模型、应用候选或改专项。后续转独立需求／实质反馈，不扩optimizer替身或微基准矩阵。

## Teacher 采样温度与打分温度

**对已有token打分与生成下一个token，可能使用不同的温度入口。** [#7769](https://github.com/verl-project/verl/issues/7769)询问为何teacher温度固定为1；2026-10-25读取为OPEN、0评论，已有yph22的OPEN／non-draft [#8253](https://github.com/verl-project/verl/pull/8253)，head `2920c0144f2cb8e0d416f7ad35ce28034795fdc9`、base及比较祖先 `5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517`。本节区分固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 与未合并候选，不另做竞争实现。

| 入口 | 当前主线 | 候选的改变 |
|---|---|---|
| teacher请求 | [manager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/experimental/teacher_loop/teacher_manager.py)把已有序列作为prompt，要求prompt_logprobs，另生成1个未使用token；请求temperature为1 | 请求仍为1，不靠改此参数缩放prompt分布 |
| teacher目标概率 | 主线该入口不将配置温度用于teacher logits缩放，即使[配置](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/config/distillation/distillation.yaml)默认插值学生rollout温度 | [vLLM teacher专属启动参数](https://github.com/verl-project/verl/blob/2920c0144f2cb8e0d416f7ad35ce28034795fdc9/verl/workers/rollout/vllm_rollout/vllm_async_server.py)送到compute_logits钩子，在prompt归一化前计算logits／T；普通actor／reward传1 |
| student概率 | [选定FSDP非fused路径](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/fsdp/transformer_impl.py)先缩放logits，再交给蒸馏logits_processor或采样token logprob计算 | 默认配置继承的同一数值开始实际作用于vLLM teacher；不代表所有后端／kernel完成一致性验收 |

候选的teacher目标是 `log P_T(v)=z_v/T−logsumexp(z/T)`，不是把已返回的top-k log-prob简单除以T。后一种做法缺少变化后的全词表归一化项；有限top-k不能直接恢复所有尾部token在新温度下的质量。观察token和top-k目标都应来自相同的完整分布。teacher／student温度是否刻意不同属于目标选择，不能把“必须相同”当数学要求；该候选的默认行为是让已有配置继承真正生效。若要保留旧teacher分布，候选文档给出的覆盖为 `distillation.teacher_models.teacher_model.inference.temperature=1.0`，本轮未应用任何配置。

### 除法之前的精度边界

[完整原logits钩子](https://github.com/verl-project/verl/blob/2920c0144f2cb8e0d416f7ad35ce28034795fdc9/verl/workers/rollout/vllm_rollout/utils.py)先屏蔽词表外尾部及禁用token；T≠1时先转FP32再除温度，T=1不额外转换。前一候选提交 `d44e8847c4e9312290ac0271f888c83e12cc61a7` 仍在模型dtype中原地相除，后续FP32 log-softmax无法补救已发生的溢出或舍入。

[CPU脚本](../../experiments/verl-teacher-temperature/verify.py)原样摘取三版完整钩子、作者原FakeModel及默认温度测试体，去掉参数化装饰器后直接调用：FP16／BF16／FP32原3参数例在main、旧候选和head均通过，只保证该夹具下T=1的dtype／值与屏蔽不变。另沿用作者固定logits和T=0.0001，对旧候选／head各两个dtype做局部控制：旧FP16有效位置log-prob非有限，旧BF16相对FP32参考最大绝对误差784；head两者有效位置有限且本例误差0，屏蔽位仍为负无穷。[原始结果](../../experiments/verl-teacher-temperature/result.json)两次一致。

非单位温度控制使用真实torch CPU log_softmax，**没有调用vLLM Sampler**，不算作者原低精度测试或30项套件通过；也未验证server启动／角色隔离、Hydra继承／配置拒绝、真实模型、GPU或训练。作者报告基于vLLM0.23，明确尚未做其所称主线pin0.29的运行时及GPU端到端验证；机器人COMMENTED不当维护者批准。其他teacher后端在本候选中仍保留未缩放的prompt分数和告警，不据vLLM局部控制切换SGLang或推广所有engine。设备缺口只补[原M5观测参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不新排部署／训练或应用补丁；下一轮转独立需求，不扩温度矩阵或另改教师实现。

## 主线与 GKD 的损失名称和支持域

**从GKD recipe迁到主线OPD，不能原样搬 `kl`、beta或rkl_ratio。** [#7975](https://github.com/verl-project/verl/issues/7975)已有Liu-KM的文档[draft #7983](https://github.com/verl-project/verl/pull/7983)，2026-10-25核为OPEN/draft、head `28879f83095a7f1a38cd4ae70d6eb331cdcf0dcc`；只读到一条CLA机器人评论、无reviews，不当已发布说明或另认领。同日固定主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，recipe固定该PR引用的 `5451026758c3f1dc3ebc2a46a0e52b14406f0f72`。

### 同名模式的输入不同

以下P表示teacher，Q表示student，K是teacher选出的top-k token集合。主线配置入口为 `distillation.distillation_loss.loss_mode`；recipe为 `actor_rollout_ref.actor.distill_loss.name`，beta／rkl_ratio在同一distill_loss命名空间。[主线registry与包装器](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/distillation/losses.py)与[recipe四模式](https://github.com/verl-project/verl-recipe/blob/5451026758c3f1dc3ebc2a46a0e52b14406f0f72/gkd/megatron/megatron_distill_losses.py)对应如下：

| 模式 | 支持域与前向目标 | 迁移时的区别 |
|---|---|---|
| 主线 `kl`／`k1` | 同一已采样token的log Q−log P；在Q采样的期望下估计反向KL | 不是recipe前向KL；PG路径把负值detach后作优势 |
| 主线 `forward_kl_topk` | Σ(K) P(log P−log Q)，保留全词表概率，不在K内重新归一化 | 默认先对两边log-prob下限裁剪，外层再将负的逐token结果裁为0 |
| recipe `kl` | 相同形式的截断前向和，不重归一化 | 没有主线包装器的相同裁剪；同一前向形式也不认证自定义backward等价 |
| recipe `rkl` | Q、P都在**teacher的K**内归一化，再算KL(Q̂∥P̂) | 不是主线采样反向KL，也不是全词表反向KL |
| recipe `kl_rkl` | (1−r)×上述recipe kl＋r×上述recipe rkl；r裁到[0,1] | 两项本来使用不同归一化，不能只搬一个混合比例 |
| recipe `jsd` | P仅保留K且外部为0，Q仍是全词表概率；M=βP(K)+(1−β)Q，含K外Q∥M解析项 | 无top-k重归一化，不能简化成两个recipe纯KL的线性混合 |

当前主线还注册 `forward_kl_topk_tail`，不能把旧比较表当作完整模式目录。它将K外teacher／student概率质量各折成一个tail bucket，形成K+1类的粗粒度前向KL；[FSDP原实现](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/distillation/fsdp/losses.py)与[tail helper](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/distillation/tail_kl.py)保留这一个尾部总质量，不恢复尾部内部token分布，也没有JSD混合分布M。学生尾质量有epsilon保护；[配置](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/config/distillation.py)要求该模式 `log_prob_min_clamp=None`。因此它不是recipe rkl／kl_rkl／jsd的直接等价入口。

### 标量损失与更新路径分开

主线 `use_policy_gradient` 是更新方式，不是第三种支持域：True时将负的逐token蒸馏损失detach为优势，走采样token的policy-loss；False时直接聚合蒸馏损失反向。相同log-ratio值不保证两条路径梯度相同。`loss_max_clamp`、response mask、聚合分母、task-reward开关及系数也须一起核，不能把前向公式相同写成训练迁移验收。

recipe JSD把beta裁到 `[1e-6,1−1e-6]`，配置0／1并不切换到独立kl／rkl模式。[recipe #151](https://github.com/verl-project/verl-recipe/issues/151)的NumPy端点分析和27B观察属于作者报告，其中生产观察来自另一实现；“原始损失或梯度很小”不能升级成“无学习／优化器更新按比例缩小”。Adam状态、epsilon及其他损失都可能改变最终更新；本轮不复跑该NumPy例、不改变公式或阈值。

### 本轮核对范围

六个完整Python模块仅AST／内存compile：主线loss registry、FSDP loss、tail helper、worker config、core_algos，以及固定recipe四模式模块。未导入框架、执行loss／自定义backward／TP collective或跑原作者测试、Sphinx、模型与GPU；源码hash和检查方式见[当日日志](../../diary/2026-10-25.md)。#7983所报323个Sphinx警告与基线相同是作者的文档构建报告，不是本项目验证。旧文中的k3=绝对值、mse无1/2及k2另为概率比公式已在下方原位置纠正。下轮转独立需求／实质反馈，不扩端点矩阵、自动搬配方、修改kernel或新排M5训练；社区发布和已有PR head推送仍须用户逐项明确同意。

---

## 1. Overview: VeOmni OPD in verl

### What OPD (On-Policy Distillation) Is

On-policy distillation trains a small **student** model to imitate a large frozen **teacher** model, but on **data generated by the student's own current policy**. This differs fundamentally from:

1. **Off-policy distillation**: training prefixes do not come solely from the student's current generation distribution; teacher-generated prefixes are one example. This does not mean a mixed training run never includes the student's own generations.
2. **GRPO/RL**: student learns from reward signals (high variance, requires reward design)

OPD provides dense, per-token targets from the teacher's distribution on the student's own rollouts. This reduces distribution mismatch and avoids high-variance reward optimization.

### VeOmni Engine

VeOmni is ByteDance's proprietary training engine (external pip package `veomni`) that provides:

- **FSDP2-based data parallelism** (VeOmniEngine inherits from FSDPEngine)
- **Ulysses Sequence Parallelism** (SP sharding along sequence dimension)
- **Expert Parallelism** for MoE models
- **CPU parameter/optimizer offloading** (param_offload, optimizer_offload)
- **Fused kernels**: `chunk_logprobs` (fused log-prob + entropy without logits materialization), `chunk_topk_distill_function` (top-K forward-KL without [B,T,V] logits tensor)
- **Router Replay** (R2/R3) for MoE routing determinism across forward/backward passes
- **Activation offloading** with `activation_gpu_limit` threshold
- **VL (Vision-Language) model support**: Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3.5-MoE
- **Ascend NPU support** alongside NVIDIA CUDA

### What PR #6072 Adds

The PR is an **integration-level feature** (3 files, 190 additions) that enables VeOmni as a training backend for the existing OPD pipeline:

1. **`verl/trainer/distillation/losses.py`**: Match `'veomni'` alongside `'fsdp'` in `compute_topk_loss()` so VeOmni reuses FSDP's topk forward KL loss implementation. VeOmni inherits from FSDP2 internally, so loss computation is identical.

2. **`verl/experimental/separation/engine_workers.py`**: Add `'veomni'` to `DetachActorWorker`'s save/restore strategy handlers. VeOmni's DTensors are compatible with FSDP2's `fsdp2_sharded_save_to_cpu`/`fsdp2_sharded_load_from_cpu`. Pass `**kwargs` through `__init__` for `distillation_config` compatibility.

3. **`examples/on_policy_distillation_trainer/run_qwen_gsm8k_veomni.sh`**: Example script for OPD with VeOmni backend, following `model_engine=veomni` configuration pattern with `actor.veomni.*` settings.

The PR does NOT introduce a new distillation architecture. It plugs VeOmni into the **existing** OPD infrastructure that was already built for FSDP/Megatron backends.

---

## 2. OPD Architecture in verl (Full Pipeline Trace)

### 2.1 Training Loop: Student Rollout -> Teacher Scoring -> Student Update

The OPD pipeline in verl follows this flow (from `docs/advance/async-on-policy-distill.md`):

```
Step Loop:
  1. Actor parameters synchronized to rollout worker group (NCCL broadcast)
  2. Rollout workers (vLLM/SGLang-backed) generate sequences asynchronously
     Student model generates responses on-policy (its own current distribution)
  3. Teacher client (ZMQ REQ/REP) returns top-k log-probabilities + token indices
     For each student-generated sequence, teacher provides:
     - teacher_topk_log_probs: (bsz, seqlen, topk) — sparse probability distribution
     - teacher_topk_ids: (bsz, seqlen, topk) — token indices for the top-k entries
  4. Student forward pass with distillation loss:
     - VeOmni/FSDP: forward KL on top-k teacher distribution
     - Or single-sample estimator loss (k1, k3, etc.)
  5. Student backward + optimizer step
  6. (Optional) Schedulers overlap phases for throughput
```

### 2.2 Loss Families and Update Paths

The two input families below are teacher-top-k objectives and sampled-token estimators. Policy-gradient versus direct backpropagation is a separate update choice; the current tail-aware top-k variant is described in the 2026-10-25 comparison above.

#### Mode 1: forward_kl_topk (Top-K Sparse Forward KL)

```
Teacher provides top-k distribution:
  teacher_topk_log_probs: (bsz, seqlen, topk)  — log P(topk tokens)
  teacher_topk_ids: (bsz, seqlen, topk)         — token indices

Student computes:
  student_log_probs = F.log_softmax(student_logits, dim=-1)  — full [B,T,V] needed
  student_topk_log_probs = gather(student_log_probs, teacher_topk_ids)  — at teacher's support

KL divergence on teacher's top-k support:
  KL(P||Q) = sum_k [P_k * (log P_k - log Q_k)]  where Q_k = student prob at teacher's token index

With VeOmni fused kernel:
  chunk_topk_distill_function avoids materializing [B,T,V] logits
  Streams lm_head projection chunk-by-chunk
  Saves significant GPU memory for large-vocabulary models (V=152K)
```

#### Mode 2: Single-Sample KL Estimators (k1, k3, kl, abs, mse, k2, low_var_kl)

The corrected expressions follow the fixed [core_algos implementation](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/core_algos.py), not an inferred expansion from estimator names.

```
Student generates tokens on-policy (single sample a_t ~ pi_student)
Teacher provides log_prob of that SAME token: teacher_log_probs(a_t)

Per-token quantities (corrected 2026-10-25 from main 3c6d9b43):
  d = log pi_student(a_t) - log pi_teacher(a_t)
  kl / k1: d                          — signed, can be negative
  abs: abs(d)
  mse / k2: 0.5 * d^2
  low_var_kl / k3: exp(-d) + d - 1
    implementation clamps -d to [-20,20], then the result to [-10,10]

Recommended pairings:
  use_policy_gradient=True + loss_mode=k1  — treat -distillation_loss as reward, PG update
  use_policy_gradient=False + loss_mode=k3  — direct supervised backprop
```

#### Update Choice: Policy Gradient Distillation (use_policy_gradient=True)

```python
# From losses.py distillation_loss() function:
if loss_config.use_policy_gradient:
    # Use negative distillation loss as reward
    # Following ThinkingMachines' on-policy distillation approach
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    old_log_prob = data["old_log_probs"]  # from rollout
    distillation_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=-distillation_losses.detach(),  # NEGATIVE distillation loss = reward!
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=loss_config,
        rollout_is_weights=rollout_is_weights,
    )
else:
    # Direct backprop (supervised distillation)
    distillation_loss = agg_loss(
        loss_mat=distillation_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
```

Key insight: `use_policy_gradient=True` treats `-distillation_losses.detach()` as **advantages** and applies PPO-clip/vanilla policy gradient. This means:
- Student's own rollout provides `old_log_probs` (from rollout phase)
- Distillation loss per token = teacher divergence = "how bad is the student at this position"
- Negative of this = "reward" = how much the student should increase probability at this position
- Standard PPO-clip/vanilla PG update applied with these rewards

**Warning from code**: `forward_kl_topk` + `use_policy_gradient=True` is inefficient because PG only uses the sampled token's gradient (single token logprob), wasting the top-k distributional signal. Recommended: `k1` + `use_policy_gradient=True` or `k3`/`forward_kl_topk` + `use_policy_gradient=False`.

### 2.3 Two-Path Loss Injection: Logits Processor + Final Loss

The `distillation_ppo_loss` function is called in **two modes**:

1. **Logits processor mode** (`student_logits is not None`): Called during model forward pass at the logits layer. Computes top-k KL loss on student logits before they're reduced to log_probs. This is the **memory-efficient path** — compute KL before vocab-size reduction.

2. **Final policy loss mode** (`student_logits is None`): Called after model forward, combines distillation loss with optional PPO policy loss:
   ```python
   policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
   if not distillation_loss_config.use_task_rewards:
       policy_loss = 0.0  # NO task reward, only distillation
   policy_loss += distill_loss * distillation_loss_coef
   ```

When `use_task_rewards=False` (common for OPD), policy_loss is zeroed out and only distillation loss drives training. When `use_task_rewards=True`, both GRPO reward and distillation are combined.

### 2.4 Overlap Diagnostics (arXiv:2604.13016)

The `compute_forward_kl_topk` function computes **teacher/student top-k overlap** diagnostics:

```python
# From verl/trainer/distillation/fsdp/losses.py:
overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
overlap_count = overlap_mask.sum(dim=-1)  # how many teacher top-k tokens are also in student top-k

# Token-level KL contribution on overlapped positions:
token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)

# Metrics:
distillation/overlap_ratio = (overlap_count / topk) — fraction of overlap
distillation/overlap_token_advantage — average KL reduction on overlapped positions
```

This directly implements diagnostics from "Rethinking On-Policy Distillation" (arXiv:2604.13016), which found that the **overlap ratio** between teacher and student top-k sets is a key indicator of distillation effectiveness. Higher overlap = more of the teacher's signal is directly actionable by the student.

---

## 3. VeOmni Engine Architecture (Source-Level)

### 3.1 VeOmniEngine: FSDP2 Inheritance + Enhanced Features

```python
class VeOmniEngine(FSDPEngine):
    # VeOmni inherits from FSDPEngine — same base architecture
    # Key additions:
    data_parallel_mode = "fsdp2"  # ONLY fsdp2 (no fsdp1 support)

    # Distributed state initialization:
    parallel_state.init_parallel_state(
        dp_size=dp_size,                             # = world_size / ulysses_parallel_size
        dp_replicate_size=data_parallel_replicate_size,  # HSDP replicate group
        dp_shard_size=data_parallel_shard_size,          # HSDP shard group (= fsdp_size)
        extra_parallel_sizes=(expert_parallel_size,),    # EP for MoE
        ulysses_size=ulysses_parallel_size,              # Ulysses SP
        dp_mode="fsdp2",                                 # FSDP2-style sharding
    )
```

**DTensor compatibility**: VeOmni's parameters are DTensors (FSDP2-sharded), which makes them compatible with `fsdp2_sharded_save_to_cpu`/`fsdp2_sharded_load_from_cpu`. This is why the `DetachActorWorker` can reuse the same handlers for both FSDP2 and VeOmni.

### 3.2 Offloading: param_offload + optimizer_offload

```python
# VeOmni engine config (veomni.yaml):
param_offload: False       # Offload model parameters to CPU when not computing
optimizer_offload: False   # Offload optimizer state to CPU
activation_gpu_limit: 0.0  # Threshold for activation offloading

# Implementation (utils.py):
@torch.no_grad()
def offload_veomni_model_to_cpu(model, empty_cache=True):
    # 1. Set all FSDP2 modules to IDLE state
    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None: continue
        fsdp_param_group._training_state = TrainingState.IDLE
    # 2. Reshard (release all-gathered full parameters)
    model.reshard()
    # 3. Move to CPU
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()

@torch.no_grad()
def load_veomni_model_to_gpu(model):
    device = get_device_id()
    model.to(device)  # Load back to GPU for forward/backward

# MultiOptimizer support for EP+FSDP2:
def offload_veomni_optimizer(optimizer):
    # Check if MultiOptimizer (separate for EP and non-EP params)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    # Per-param state offload:
    for param_group in opt.param_groups:
        for param in param_group["params"]:
            state = opt.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
```

**CAVEAT from PR**: When `param_offload=True`, parameters may reside on CPU at time of `save_model_to_cpu`. The `fsdp2_sharded_save_to_cpu` assumes parameters are on GPU (as DTensors). Callers must ensure model is loaded back to GPU before calling save/restore in offload scenarios.

### 3.3 Fused Kernel: chunk_logprobs + chunk_topk_distill

The VeOmni fused kernel system is the key advantage over vanilla FSDP for OPD:

```python
# VeOmniEngineWithLMHead.prepare_model_inputs():
use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)

if use_fused_kernels and use_remove_padding:
    # ForCausalLMLoss short-circuits to per-token log_probs/entropy
    # No [B,T,V] logits materialization!
    model_inputs["labels"] = input_ids_rmpad
    model_inputs["shift_labels"] = shift_labels
    model_inputs["return_log_probs"] = True

    # OPD: pass teacher top-K tensors for fused distillation
    distillation_use_topk = tu.get_non_tensor_data(micro_batch, "distillation_use_topk", default=False)
    if distillation_use_topk and "teacher_ids" in micro_batch.keys():
        teacher_topk_ids = micro_batch["teacher_ids"].values().unsqueeze(0)
        teacher_topk_log_probs = micro_batch["teacher_logprobs"].values().unsqueeze(0)
        # SP-slice if Ulysses SP enabled
        if self.use_ulysses_sp:
            teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1, padding=True)
            teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1, padding=True)
        model_inputs["teacher_topk_ids"] = teacher_topk_ids
        model_inputs["teacher_topk_log_probs"] = teacher_topk_log_probs
```

**Memory savings**: For a Qwen model with V=152K vocab, the full `[B,T,V]` logits tensor at bf16 is enormous. For `B=2, T=1536, V=152K`: logits = 2 * 1536 * 152576 * 2 bytes = ~936MB. The fused `chunk_topk_distill` path avoids this by streaming the `lm_head` projection chunk-by-chunk, computing top-K forward KL without ever materializing the full logits buffer.

For long context (>=64K tokens), there's also a `_chunked_topk_log_probs` function in the FSDP losses module that streams `logsumexp + gather` in chunks of 4096 tokens:

```python
def _chunked_topk_log_probs(logits, topk_ids, chunk_size=4096):
    # Identity: log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    # Stream chunks to avoid [B,T,V] log_softmax buffer
    for s in range(0, N, chunk_size):
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
```

### 3.4 Router Replay (R2/R3) for MoE Determinism

VeOmni introduces Router Replay to fix MoE routing nondeterminism across forward/backward passes:

```python
# VeOmniEngineConfig.router_replay:
mode: disabled  # or "R2" or "R3"

# R2 (Record-then-Replay, actor-side):
# - RECORD: compute_log_prob (forward_only=True) records top-k expert indices per token
# - REPLAY: actor update substitutes recorded indices, guaranteeing bit-equal forward

# R3 (Rollout-then-Replay, rollout-side):
# - RECORD: rollout backend records response-token routing during generation
# - REPLAY: both compute_log_prob and actor update use recorded indices
# - Requires rollout.enable_rollout_routing_replay=True
# - Response-only mask (prompt tokens go through native routing)
```

This is critical for MoE OPD because atomic-add nondeterminism in fused MoE experts means fresh router calls may pick different experts than the forward pass, breaking gradient correctness.

### 3.5 Activation Offloading Context

```python
# VeOmniEngine._build_model_optimizer():
self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
    self.model_config.enable_activation_offload,
    self.model_config.enable_gradient_checkpointing,
    self.engine_config.activation_gpu_limit,
)
```

When `activation_gpu_limit > 0`, activations above the threshold are offloaded to CPU during forward pass and reloaded during backward. This is VeOmni's equivalent of DeepSpeed's NVMe offload for activations.

---

## 4. Async Scheduling: Zero-Step / One-Step / Two-Step Off-Policy

The OPD documentation describes three scheduling modes for overlapping phases:

### Zero-Step-Off (Synchronous, Strict On-Policy)

```
Step 1: [rollout] -> [teacher scoring] -> [actor update]
Step 2: [rollout] -> [teacher scoring] -> [actor update]
```

Pipeline bubbles: GPU idle while waiting for each phase. Strictest on-policy guarantee.

### One-Step-Off-Policy

```
Warm-up: 2 steps
Step 3: [rollout while actor update] + [weight sync while teacher retrieving]
```

Overlap pattern:
- Rollout generation happens simultaneously with actor update
- Weight sync to rollout workers overlaps with teacher knowledge retrieval
- Timing keys: `sync_rollout_weights`, `wait_prev_gen`, `wait_prev_teacher`

### Two-Step-Off-Policy

```
Warm-up: 3 steps
Step 4: [rollout], [actor update while teacher retrieving]; interleave weight sync
```

Overlap pattern:
- Three phases concurrent: rollout, actor update, teacher retrieval
- Weight sync interleaved between phases
- Timing: `sync_rollout_weights`, `max(wait_prev_gen, wait_prev_prev_teacher)`

**Practical tip**: Use `two_step_off` when teacher takes much more time than sync; `one_step_off` for simpler overlapping. These schedulers trade some strict on-policy guarantees for throughput gains.

---

## 5. Multi-Teacher OPD (MOPD)

verl supports **multi-teacher distillation** where different datasets route to different teacher models:

```python
# DistillationConfig:
teacher_models: dict[str, DistillationTeacherModelConfig]
teacher_key: str = "data_source"  # Route examples by data_source field

# Example: GSM8K (text) + Geo3K (VL) with different teachers:
+distillation.teacher_models.gsm8k.key="openai/gsm8k"
+distillation.teacher_models.gsm8k.model_path=Qwen/Qwen3-32B
+distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
+distillation.teacher_models.geo3k.model_path=Qwen/Qwen3-VL-32B-Instruct
distillation.teacher_key=data_source  # routing field in dataset
```

Each teacher model runs as a separate Ray cluster with configurable replicas and TP size. The `DistillationTeacherModelConfig` validates that:
- `num_replicas * per_replica_world_size` equals the teacher resource pool size
- Teacher's `max_model_len >= student_prompt_length + student_response_length + 1`
- vLLM teacher: `max_logprobs >= topk` (or SGLang: per-request top_logprobs_num)

---

## 6. Separation Mode (Sleep/Wake) + VeOmni Integration

The `DetachActorWorker` in `verl/experimental/separation/engine_workers.py` supports **separation mode** (GPU time-multiplexing) where the actor model is saved to CPU and restored from CPU between rollout and training phases:

```python
class DetachActorWorker(ActorRolloutRefWorker):
    def __init__(self, config, role, distillation_config=None, **kwargs):
        # Pass distillation_config through for OPD support in separation mode
        ActorRolloutRefWorker.__init__(self, config, role,
                                       distillation_config=distillation_config, **kwargs)

    def save_model_to_cpu(self, n):
        # FSDP/FSDP2/VeOmni: fsdp2_sharded_save_to_cpu (DTensor sharded save)
        # Megatron: copy_megatron_model_to_cpu

    def restore_model_from_cpu(self, n):
        # FSDP/FSDP2/VeOmni: fsdp2_sharded_load_from_cpu (DTensor sharded load)
        # Megatron: restore_megatron_model_from_cpu
```

**VeOmni-specific caveat**: When `param_offload=True`, parameters may be on CPU at the time of save/restore. The `fsdp2_sharded_save_to_cpu` assumes parameters are on GPU. Must reload to GPU before saving.

---

## 7. verl VeOmni OPD vs DeepSpeed OPD #8027: Detailed Comparison

### 7.1 Architecture Comparison

| Feature | verl VeOmni OPD | DeepSpeed OPD #8027 |
|---------|-----------------|---------------------|
| **Training engine** | VeOmni (FSDP2-based, external package) | DeepSpeed ZeRO-0/2 |
| **Rollout backend** | vLLM/SGLang (Ray-based async) | vLLM or HuggingFace generate |
| **Teacher serving** | Ray cluster (ZMQ REQ/REP for async) + native vLLM teacher worker | TeacherWrapper (CPU-resident logit cache) |
| **Student rollout** | Student parameters -> NCCL broadcast -> vLLM rollout -> async generate | RolloutEngine.generate() (vLLM or HF) |
| **Loss modes** | k1, k3, forward_kl_topk, kl, abs, mse, k2, low_var_kl (7+1 registered) | Forward KL (top-k or full vocab) |
| **PG mode** | use_policy_gradient=True: -distillation_loss as reward, PPO/vanilla PG | Not explicitly mentioned (supervised-only?) |
| **Task rewards** | use_task_rewards=True/False (combine GRPO+distill or pure distill) | Not mentioned (pure distillation) |
| **Multi-teacher** | MOPD: multiple teachers with data_source routing | Single teacher only |
| **Async scheduling** | zero_step_off / one_step_off / two_step_off | Synchronous three-phase loop |
| **MoE routing** | Router Replay (R2/R3) for determinism | Not addressed |
| **VL models** | Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3.5-MoE VL | Not mentioned (text-only) |
| **Memory optimization** | chunk_topk_distill (no [B,T,V] logits), chunked_topk_log_probs, activation_gpu_limit | CPU logit cache, streamed divergence, chunked fetch |
| **Parameter offload** | VeOmni param_offload + optimizer_offload (FSDP2 DTensor compatible) | ZeRO offload, CPU_Adam |
| **Separation/Sleep/Wake** | DetachActorWorker with fsdp2_sharded save/load | Not implemented (ZeRO-0 minimal) |
| **Status** | MERGED (April 2026) | DRAFT OPEN (not merged) |

### 7.2 Key Architectural Differences

**1. Teacher serving architecture**:

verl: Teacher runs as a **separate Ray resource pool** with dedicated GPUs. Each teacher is a vLLM/SGLang inference server. Student sends batched micro-requests via ZMQ/Ray. The teacher resource pool size is explicitly configured:
```yaml
distillation.n_gpus_per_node=4  # teacher GPU pool
distillation.nnodes=1
```

DeepSpeed: Teacher runs on **CPU** with TeacherWrapper. TeacherLogitCache is CPU-resident. This saves GPU memory but is slower for large teachers.

**2. Loss computation path**:

verl VeOmni: **Two-path injection** (logits processor during forward + final loss after forward). The logits processor computes top-K KL at the logits layer, BEFORE vocab-size reduction to log_probs. VeOmni's fused `chunk_topk_distill` avoids materializing [B,T,V] entirely.

DeepSpeed: **Streamed divergence** — student forward produces full logits [B,T,V], then streamed_distillation_loss fetches teacher chunks from CPU cache. Still materializes [B,T,V] on GPU.

**3. Scheduling flexibility**:

verl: Supports zero/one/two-step off-policy scheduling. Can overlap rollout, teacher, and update phases for throughput.

DeepSpeed: Synchronous three-phase loop. No overlap options yet.

**4. Multi-teacher capability**:

verl: Full MOPD support with dataset routing. Can distill from multiple domain-specific teachers simultaneously.

DeepSpeed: Single teacher only.

### 7.3 Memory Comparison for RTX 4090

| Scenario | verl VeOmni OPD | DeepSpeed OPD |
|----------|-----------------|---------------|
| Student: Qwen2.5-0.5B | ~1GB (bf16, FSDP2 sharded) | ~1GB (bf16, ZeRO-0) |
| Student: Qwen3-0.6B | ~1.2GB | ~1.2GB |
| Teacher logits (topk=32) | GPU: nested tensor ~3MB/batch | CPU: full vocab chunks streamed |
| Teacher logits (topk=64) | GPU: ~6MB/batch | CPU: larger chunks |
| Student logits [B,T,V] | AVOIDED by chunk_topk_distill! | Materialized on GPU (~936MB for V=152K) |
| Optimizer state | Can offload to CPU (optimizer_offload=True) | CPU_Adam on CPU (5-7x faster) |
| Parameters | Can offload to CPU (param_offload=True) | ZeRO-2 sharded or ZeRO-0 unsharded |
| Peak GPU memory | ~2-4GB (student + small topk buffer) | ~1-2GB (student) + CPU teacher logits |

**Key advantage of verl VeOmni**: The fused `chunk_topk_distill` kernel avoids the [B,T,V] logits materialization entirely. For V=152K (Qwen models), this saves ~936MB of peak GPU memory per forward step. This is transformative for RTX 4090 with only 24GB VRAM.

**Key advantage of DeepSpeed**: CPU_Adam (C++ SIMD AVX512) is 5-7x faster than torch.optim.Adam on CPU. For LoRA training with only ~0.6GB trainable parameters, optimizer offload choice matters less, but for full-parameter training CPU_Adam is superior.

---

## 8. RTX 4090 Practical Implications

### 8.1 Viable Model Configurations

**Single-GPU RTX 4090 OPD** (most practical):

```
Student: Qwen2.5-0.5B (~1GB bf16)
Teacher: Qwen2.5-1.5B or Qwen2.5-3B-Instruct (served on separate GPUs or remote)

Configuration:
  model_engine=veomni
  actor.veomni.param_offload=True     # Offload params to CPU when not computing
  actor.veomni.optimizer_offload=True  # Offload optimizer to CPU
  actor.veomni.enable_full_shard=True  # FSDP2 full sharding
  distillation.enabled=True
  distillation.distillation_loss.loss_mode=k1
  distillation.distillation_loss.use_policy_gradient=True
  distillation.distillation_loss.use_task_rewards=False
  actor_rollout_ref.model.use_fused_kernels=True  # CRITICAL: avoids [B,T,V] logits
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

**Problem**: The teacher needs separate GPUs. On RTX 4090, student + rollout + teacher all compete for 24GB. The example script `run_qwen_gsm8k_veomni.sh` uses `STUDENT_WORLD_SIZE=2, TEACHER_WORLD_SIZE=4` (6 GPUs total). This doesn't fit on a single RTX 4090.

**RTX 4090 workaround options**:
1. **Teacher on CPU/cloud**: Run teacher inference on a remote server or cloud GPU
2. **Teacher precomputed**: Cache teacher logits offline for static datasets
3. **Smaller teacher**: Use Qwen2.5-0.5B -> Qwen2.5-1.5B-Instruct (both small enough)
4. **HYBRID/FusedWorker**: In single-GPU mode, student and rollout share one process (verl HYBRID mode), teacher is separate

### 8.2 Memory Budget Analysis (Single RTX 4090)

```
Student model (Qwen2.5-0.5B, bf16): ~1GB
Rollout (vLLM, gpu_memory_utilization=0.3): ~7.2GB
Optimizer (offloaded to CPU): 0GB GPU
Parameters (offloaded to CPU): 0GB GPU during rollout phase
Activations + gradients during training: ~2-3GB
Fused kernel path (no [B,T,V] logits): saves ~936MB vs vanilla FSDP

Total GPU during training: ~3-4GB
Total GPU during rollout: ~8-9GB (vLLM needs 7.2GB for student model inference)

This leaves ~15GB headroom for teacher inference on the SAME GPU!
Qwen2.5-1.5B at gpu_memory_utilization=0.5: ~3GB -> viable on same GPU
Qwen2.5-3B at gpu_memory_utilization=0.5: ~6GB -> tight but possible with param_offload

BUT: student rollout and teacher scoring happen at different times
      Separation mode (DetachActorWorker) enables time-multiplexing!
```

### 8.3 Recommended RTX 4090 Configuration

```bash
# Single-GPU RTX 4090 OPD with VeOmni
# Student: Qwen2.5-0.5B, Teacher: Qwen2.5-1.5B-Instruct
# Using separation mode (GPU time-multiplexing)

STUDENT_MODEL=Qwen/Qwen2.5-0.5B
TEACHER_MODEL=Qwen/Qwen2.5-1.5B-Instruct
STUDENT_WORLD_SIZE=1
TEACHER_WORLD_SIZE=1

model_engine=veomni
actor_rollout_ref.model.use_fused_kernels=True
actor_rollout_ref.actor.veomni.param_offload=True
actor_rollout_ref.actor.veomni.optimizer_offload=True
actor_rollout_ref.actor.veomni.enable_full_shard=True

distillation.enabled=True
distillation.n_gpus_per_node=1
distillation.nnodes=1
distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.topk=32

actor_rollout_ref.rollout.gpu_memory_utilization=0.3
actor_rollout_ref.actor.optim.lr=1e-5
```

### 8.4 Loss Mode Selection for RTX 4090

| Mode | PG? | Memory | Recommendation | RTX 4090 Notes |
|------|-----|--------|----------------|----------------|
| k1 + PG=True | Yes | Low (single-sample) | Best for RTX 4090 | Most efficient, dense token targets |
| forward_kl_topk + PG=False | No | High unless fused | Second choice | VeOmni fused kernel reduces memory |
| k3 + PG=False | No | Low (single-sample) | Direct supervised | Simple, stable, but no exploration |
| forward_kl_topk + PG=True | Yes | Wasteful | Avoid | PG wastes top-k distributional signal |

**RTX 4090 recommendation**: `k1 + use_policy_gradient=True` is the best combination:
- Single-sample estimator = minimal memory
- PG treats -distillation_loss as reward = exploration
- Works with LoRA for further memory savings
- Matches the example script defaults (`DISTILLATION_LOSS_MODE="k1"`, `USE_POLICY_GRADIENT=True`)

---

## 9. Comparison Summary: verl VeOmni vs DeepSpeed for RTX 4090 OPD

### 9.1 verl VeOmni Advantages

1. **Fused kernel memory savings**: `chunk_topk_distill` avoids [B,T,V] logits materialization. Saves ~936MB per forward for V=152K. Transformative for 24GB VRAM.

2. **Multi-teacher OPD (MOPD)**: Can distill from multiple domain experts simultaneously. Not available in DeepSpeed.

3. **Async scheduling**: zero/one/two-step overlap for throughput. DeepSpeed is synchronous only.

4. **VL model support**: Qwen2.5-VL, Qwen3-VL distillation. DeepSpeed doesn't handle VL models in OPD.

5. **Router Replay (R2/R3)**: MoE routing determinism for bit-equal forward/backward. Critical for MoE models like Qwen3-MoE.

6. **Production maturity**: PR #6072 MERGED. DeepSpeed OPD #8027 is DRAFT OPEN.

7. **Ray-native orchestration**: Mature worker lifecycle, weight sync, resource pool management.

### 9.2 DeepSpeed OPD Advantages

1. **CPU_Adam**: 5-7x faster than torch.optim.Adam on CPU via C++ SIMD AVX512. Superior optimizer offload performance.

2. **CPU teacher logit cache**: Teacher entirely on CPU. No teacher GPU needed. Perfect for single-GPU setups.

3. **Streamed divergence**: Fetches teacher chunks from CPU during student backward. No GPU memory for teacher logits at all.

4. **ZeRO simplicity**: ZeRO-0 for OPD = minimal overhead. No FSDP2 complexity.

5. **LoRA integration**: DeepSpeed has LoRAOptimizedLinear with offload_ratio. verl relies on VeOmni's external LoRA support.

6. **Qwen2.5-0.5B fits RTX 4090**: Both frameworks can fit ~1GB student. DeepSpeed's CPU-only teacher makes it simpler for single GPU.

### 9.3 RTX 4090 Ranking for OPD

```
DeepSpeed OPD #8027: #1 for RTX 4090 OPD
  - CPU teacher = no GPU contention
  - CPU_Adam = fast optimizer on CPU
  - ZeRO-0 = minimal overhead
  - BUT: DRAFT OPEN, not merged

verl VeOmni OPD #6072: #2 for RTX 4090 OPD
  - Fused kernel saves ~936MB GPU memory
  - MOPD + async scheduling = more features
  - MERGED and production-ready
  - BUT: teacher needs separate GPU (or separation mode workaround)
  - External veomni dependency adds complexity

verl FSDP OPD: #3 for RTX 4090 OPD
  - Same loss architecture as VeOmni
  - No fused kernel = must materialize [B,T,V] logits
  - Simpler setup (no external dependency)
```

**Practical recommendation**: For a single RTX 4090, DeepSpeed OPD is theoretically simpler (CPU teacher). But it's not merged. verl VeOmni OPD is production-ready and the fused kernel is transformative for memory. With separation mode (time-multiplexing), both student and teacher can share one GPU.

---

## 10. Key Source Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `verl/trainer/distillation/losses.py` | Core loss functions: distillation_ppo_loss, compute_topk_loss, distillation_loss, registered loss modes (k1/k3/forward_kl_topk etc.) | ~250 |
| `verl/trainer/distillation/fsdp/losses.py` | FSDP/VeOmni loss implementation: compute_forward_kl_topk, _chunked_topk_log_probs, kl_divergence, overlap diagnostics | ~120 |
| `verl/workers/config/distillation.py` | DistillationConfig, DistillationLossConfig, DistillationTeacherModelConfig dataclasses | ~150 |
| `verl/workers/engine/veomni/transformer_impl.py` | VeOmniEngine (FSDP2-based), VeOmniEngineWithLMHead (fused chunk_logprobs + chunk_topk_distill injection) | ~1070 |
| `verl/workers/engine/veomni/utils.py` | offload/load model/optimizer, MoE param handlers | ~100 |
| `verl/experimental/separation/engine_workers.py` | DetachActorWorker: save/restore model to CPU, VeOmni+FSDP2 strategy handler support | ~140 |
| `verl/trainer/config/distillation/distillation.yaml` | Default distillation config (loss_mode, topk, teacher_models) | ~80 |
| `verl/trainer/config/engine/veomni.yaml` | VeOmni engine config (param_offload, optimizer_offload, ulysses_parallel_size, router_replay) | ~80 |
| `examples/on_policy_distillation_trainer/run_qwen_gsm8k_veomni.sh` | PR #6072 example: Qwen2.5-0.5B + Qwen2.5-3B-Instruct, k1 + PG=True | 161 |
| `examples/on_policy_distillation_trainer/run_qwen3_0.6b_opd_veomni.sh` | Qwen3-0.6B + Qwen3-1.7B, fused kernels, k1 + PG=True, 7+1 GPUs | ~120 |
| `examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_veomni.sh` | MOPD: Qwen3-VL-8B + GSM8K+Geo3K teachers, fused kernels | ~120 |
| `docs/advance/async-on-policy-distill.md` | Architecture documentation: OPD pipeline, schedulers, teacher serving, metrics | ~300 |

---

## 11. Open Questions and Future Directions

1. **Single-GPU teacher serving**: Can the teacher vLLM server share the RTX 4090 with the student via separation mode? Need GPU validation.

2. **VeOmni external dependency**: `veomni` is an external pip package. Installation and compatibility with consumer GPUs needs verification.

3. **Fused kernel on SM89**: Does `chunk_topk_distill_function` work on RTX 4090 (SM89)? VeOmni's fused kernels may have SM architecture requirements.

4. **DeepSpeed OPD merge timeline**: When #8027 merges, the CPU-teacher approach becomes viable, potentially surpassing verl VeOmni for single-GPU setups.

5. **LoRA + VeOmni OPD**: Can LoRA (VeOmni's built-in support) combine with OPD for even lower GPU footprint on RTX 4090?

6. **Async trainer + OPD**: Can verl's async trainer (rLLM-style concurrent loops) overlap with OPD phases for further throughput gains?

7. **OPD + GRPO hybrid**: verl supports `use_task_rewards=True` to combine distillation loss with GRPO reward. What's the optimal mixing coefficient for RTX 4090?

---

## 12. Star Rating Summary

★★★★★★★★ VeOmni OPD architecture: production-ready, merged PR, comprehensive loss mode catalog (8 registered), fused kernel memory savings, MOPD support, async scheduling

★★★★★★★ Fused chunk_topk_distill: avoids [B,T,V] logits materialization, saves ~936MB for V=152K models, transformative for 24GB VRAM

★★★★★★ Overlap diagnostics (arXiv:2604.13016): teacher/student top-k overlap tracking, overlap_ratio + overlap_token_advantage metrics

★★★★★★ Router Replay (R2/R3): MoE routing determinism, critical for Qwen3-MoE OPD, bit-equal forward guarantee

★★★★★ Separation mode integration: VeOmni save/restore via fsdp2_sharded, param_offload caveat documented

★★★★ RTX 4090 viability: student fits (~1GB), but teacher needs separate GPU; separation mode workaround possible but needs validation; VeOmni external dependency adds complexity

★★★ vs DeepSpeed OPD: verl VeOmni = more features + merged + fused kernel; DeepSpeed = simpler single-GPU (CPU teacher) + CPU_Adam but DRAFT OPEN

★★★★★★★ Overall: verl VeOmni OPD is the most comprehensive OPD implementation in the OSS ecosystem. Fused kernel + MOPD + async scheduling = unique capability set. DeepSpeed OPD remains simpler for single-GPU but unmerged.

## 教师生成与蒸馏目标

[#8335](https://github.com/verl-project/verl/pull/8335)增加的是可选训练数据来源，不是把教师变成被优化的actor。2026-10-23 API读取为open/draft；固定head `8029bfb5fc5b8b6034704f9c062610613f1bf05c`、base及共同祖先 `fb55b9288560f0c7bf380c1c1c3b45ea170538b0`，23文件977增5删。实验配置面向同步V1，不能沿用旧VeOmni/async阅读的支持面或4090建议。

### 生成 打分与更新

[`teacher_rollout.py`](https://github.com/verl-project/verl/blob/8029bfb5fc5b8b6034704f9c062610613f1bf05c/verl/experimental/teacher_rollout.py)加载一次同步selector。训练端在 `_submit_batch_to_rollout` 的TransferQueue写入和worker分派之前调用它，输入是尚未重复response的prompt batch、恢复后的global_step和rollout seed。返回值必须是一个 `student` 或 `teacher` 字符串，决定整次submission，而不是逐prompt、逐worker或逐token混合；所有重复响应继承该选择。返回列表、非法字符串或抛异常会停止本次提交，不静默回退。

| 阶段 | 学生生成批 | 教师生成批 | validation |
|---|---|---|---|
| 生成响应 | student client | 已有唯一teacher client | student client，且不调用selector |
| 取得蒸馏软标签 | teacher对完整prompt+response另行打分 | 同样另行请求teacher打分 | 跳过teacher蒸馏打分 |
| 训练目标 | 学生在这些prefix上做监督top-k蒸馏 | 学生在教师prefix上做同一种蒸馏 | 不做训练更新 |

[`AgentLoopWorker._run_agent_loop`](https://github.com/verl-project/verl/blob/8029bfb5fc5b8b6034704f9c062610613f1bf05c/verl/experimental/agent_loop/agent_loop.py#L694)切换的是client。教师生成沿用学生采样参数，并仅为该请求覆盖完整response预算和EOS策略，因为同一teacher服务平时为打分配置response_length=1。生成结束后，TQ后处理仍调用 `_compute_teacher_logprobs`，对完成序列取软top-k targets；教师采到的那个token不是唯一硬标签，也不能把生成和评分算成一次请求。

validation不走训练提交入口；worker还会将残留teacher来源强制覆为student。`SingleTurnAgentLoop`在selector启用时拒绝解析到的非空图像/视频/音频，无论该批最终选择哪一方。

### 不是纯学生 on-policy

teacher-only或混合生成改变了训练prefix的分布，不能继续称为纯学生on-policy蒸馏。当前接口要求 `forward_kl_topk`、`use_policy_gradient=False`、`use_task_rewards=False`：学生拟合这些prefix上的软分布，而不是把教师采样概率塞进PPO概率比。

两份原蒸馏loss模块在base/head字节相同。既有 `distillation_ppo_loss` 在上述两个布尔开关都关闭时不调用PPO loss；原学生old-logprob方法也未改。正常非bypass路径仍由actor计算old_log_probs/entropy，并非改成teacher行为概率。`calculate_log_probs=False`只关闭rollout端该字段，不能理解为消除了学生所有概率计算成本；文档也未支持与rollout correction混用。

### 配置保护与使用前提

新检查按**是否配置selector**生效，而非按它这次是否选择teacher生效。即使固定概率为0、实际总选student，也必须满足这条实验路径的限制；移除selector才恢复原配置自由度。

- **配置中明确拒绝**：非同步V1、非vLLM rollout/teacher、非single-turn默认agent、multi-turn、非forward_kl_topk、PG或task reward开启、rollout logprobs开启、任一temperature不等于1、topk缺失或≤0、自定义manager/agent配置/回放sampler、failed-group refill，以及解析后不是恰好一个teacher或无model_path。
- **运行时继续检查**：selector必须可调用且返回合法单字符串；每样本agent和解析后的模态仍受限制。用户回调应视prompt batch只读，但这是调用契约，不是代码沙箱或自动冻结。
- **文档要求但新检查没有逐项认证**：学生/教师tokenizer和词表匹配；不与critic、额外KL/entropy目标、rollout correction、分数居中、router replay、投机解码或group filtering组合。通过该函数不等于这些条件已验证，也不等于权重、内存或后端可用。

内置概率selector用局部 `random.Random` 和seed/step生成决定，不消耗全局Python RNG。同seed/step的多次submission会得到同一选择，不是每次调用独立抽签；这也不保证生成文本逐位相同。自定义回调的任意内部状态没有自动checkpoint，恢复责任仍在策略作者。

### 指标口径

source写入输出field/tag。`distillation/teacher_rollout_fraction`按**非padding轨迹条数**计教师比例，不是教师token占比、batch选择概率或已持久交付的认证。缺source的旧tag按student处理，故指标依赖元数据契约。

staleness只读非padding的student tags；teacher-only批没有student版本样本，相关off-policy指标被省略，不是lag=0或teacher已与actor同步。源码标签/过滤逻辑与真实服务内容身份是不同证据。

### 验证与接续

本机完整读取146行新模块，核训练提交/worker路由/TQ输出/old-logprob/validation/loss关键路径和5份相关原测试；不是完整审查23文件。Python3.12.13仅对16个修改Python文件AST/内存compile，另核两份loss及teacher_manager共3文件base/head字节相同，`_compute_old_log_prob`/`_validate`两方法AST相同。未导入框架或执行selector/原pytest，本机缺OmegaConf/TensorDict，不加配置、client或Ray替身。

作者报告同一head：183项CPU通过、10项既有critic配置测试跳过，8×H200上的6种执行场景、17次优化器更新和48条学生validation输出检查通过。这是外部报告，不是本项目结果，也不是质量/收敛或消费级卡支持证明；作者还明确全padding actor rank仍可能触发既有指标失败。

真机参考写入[M5原一致性入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不新增训练任务。只有M5后续独立选择且已有获准匹配依赖、模型/tokenizer及teacher资源，才决定是否做有界source/评分/掩码/恢复观测；先缺导入依赖时保留未测，不下载权重、部署draft或把作者8卡报告外推为4090可跑。与#8250没有已证因果关系。
