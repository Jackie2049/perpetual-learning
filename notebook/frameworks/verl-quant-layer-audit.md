# verl：训练与 rollout 的量化层审计

2026-10-21，M1 只读学习已有作者的 [PR #8132](https://github.com/verl-project/verl/pull/8132)。核心收获：**配置一致、导出格式一致、引擎实际选层一致是不同证据；选层一致仍不等于权重数值或 logprob 一致。** 本文不是新缺陷报告或竞争实现。

2026-10-24接续：[写入前检查与失败握手](#写入前检查与失败握手)说明#8327新版将引擎格式检查前移，以及失败后排空IPC握手的边界；本轮只做源码与语法核对。此前[新版运行期检查](#新版运行期检查与指标哨兵)的纯Python哨兵结果属于旧head，不迁移为新版本验收。旧#8132/#8306分析也保留各自固定版本范围。

2026-10-25接续：[VeOmni计算与通信边界](#veomni-低精度计算与通信边界)补入作者新开的VeOmni #1360：后端分工与配置迁移仍待确认；端到端RFC、已合并的ReduceScatter接线及torchao版本限制分别看待，通信微基准收益不能直接外推actor更新速度。

2026-10-25补充：[QAT 的 mask 默认与 packed 接口](#qat-的-mask-默认与-packed-接口)核清默认策略如何经过 Bridge、MCore 到达 TE。mask 类别通过不等于完整 THD 输入有效，也不能解决量化 MoE 的部署 kernel 限制。

2026-10-26接续：[原生混合精度模型的后端选择](#原生混合精度模型的后端选择)追踪新议题#8381。顶层FP8配置不代表experts也为FP8；显式DeepGEMM会阻止自动候选回退，切换Marlin还改变activation精度，不能仅按“成功启动”认证训练与rollout一致。

## LoRA 权重名称与加载器契约

**同一个 `.base_layer` 后缀，可能需要保留、补回或删除；是否能找到真实目标与谁负责后续映射必须一起判断。** 2026-10-30只读学习HollowMan6的[#7483](https://github.com/verl-project/verl/pull/7483)：当前head `0ed94ba020747101bb22db6005849c9a6ca06aa6`、目标base `a08ea02a784d7d4982b1d81df36d188a476192bb`，OPEN／非draft，45提交52文件5210增331删。本节只覆盖[utils.py的三个名称helper](https://github.com/verl-project/verl/blob/0ed94ba020747101bb22db6005849c9a6ca06aa6/verl/utils/vllm/utils.py)，不是对整个模型适配栈的审查或新修复候选。

### 严格查表与递归加载

严格加载器最终按名称查参数字典，LoRA包装后的真实键可能是 `q_proj.base_layer.weight`；递归的扁平加载器则可能自行进入包装层，需要接收 `q_proj.weight`。候选不是按模型名设白名单，而是沿 `.model`／`.language_model` 检查下一层 `load_weights` 的代码对象：`co_names` 不含 `AutoWeightsLoader` 时视为严格，结果按外层类缓存。这是结构启发式，不是执行加载契约；同类实例复用缓存还依赖其包装结构稳定。原测试注释仍提到 `inspect.getsource`／`params_dict[`，**固定版本的实际判断并不搜索源码文本或字典下标**。

候选再结合真实参数名、`hf_to_vllm_mapper` 和 `packed_modules_mapping` 决定一个后缀的去留。映射只辅助查找：返回值仍是发送端的未映射名称，留给vLLM完成映射与分片融合。否则“本地找到真实目标”可能变成接收端重复改名。

| 原测试条件 | 解析结果 |
|---|---|
| 新加载接口夹具、严格内层、真实键含base_layer | 已有后缀保留；普通名称补回后缀 |
| 相同真实键、扁平内层 | 删除输入的base_layer后缀 |
| 严格内层，但视觉merger真实键无包装 | 删除merger后缀，语言投影后缀仍保留 |
| compressor.wkv／wgate分片，融合目标无包装 | 删除后缀，保留分片名与原前缀；输入无model前缀也不补 |
| gate_proj／up_proj分片，融合目标含包装 | 保留后缀；原旧接口夹具在严格／扁平两种内层均保留 |

例如输入 `layers.0.attn.compressor.wkv.base_layer.weight`，真实目标为 `model.layers.0.attn.compressor.fused_wkv_wgate.weight`，原测试结果仍是 `layers.0.attn.compressor.wkv.weight`，不是目标全名。这里证明的是解析决策，未调用真实vLLM的stacking或参数写入。

### 八个原测试体的隔离结果

本机Python3.12.13／Torch2.9.1／pytest8.4.2，抽取原三个完整生产函数、原缓存／哨兵赋值及[原测试文件](https://github.com/verl-project/verl/blob/0ed94ba020747101bb22db6005849c9a6ca06aa6/tests/utils/test_vllm_weight_name_normalization_on_cpu.py)的九个假模型／mapper／worker辅助定义，直接运行八个未改测试体：**8通过、17次名称解析，两次当前版本新进程结果相同**。保留原断言与真实`pytest.MonkeyPatch`，但不是原生pytest收集；worker类型为空分配夹具、特性标志显式设定，只创建空CPU张量，原严格／扁平loader函数体均未执行。

见[脚本](../../experiments/verl-lora-loader-names/verify.py)与[完整结果](../../experiments/verl-lora-loader-names/result.json)。复现：`python3.12 -B experiments/verl-lora-loader-names/verify.py /path/to/verl-git`；需预先缓存脚本中的两个commit，运行禁止隐式联网。初读快照 `2671d862e3fb5e937b0da70da348b86e37df724f` 与当前head的两份完整输入逐字节相同，脚本强校验并保存SHA256，不能由此推断其余50文件相同。

这组结果不覆盖真实能力探测、混合VLM多层前缀、全部MoE分支、量化重载或模型训练。PR作者的双节点GPU报告也不计为本机验收。已有作者在推进，本题在名称契约范围收束，不另写修复、部署候选或新增M5算力队列；上游可见动作仍逐项待用户明确授权。

## 原生混合精度模型的后端选择

**后端应按实际层的格式与能力判断，不能只看模型顶层量化名称。** [议题#8381](https://github.com/verl-project/verl/issues/8381)于2026-10-26报告：原生DeepSeek-V4-Pro在H100上因MXFP4 expert被指定DeepGEMM而加载失败；显式Marlin后完成三步。报告者已有方案草稿并询问是否并入[#8371](https://github.com/verl-project/verl/pull/8371)，读取时0评论；本文不另实现方案。此前QAT预设中的GH200限制与此原生checkpoint问题相关，但不能混为同一模型格式或同一次实验。

### 默认配置如何成为显式约束

verl固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，vLLM固定v0.29.0的commit `98dff2a81d747d1dba01a47f939f48c3526d4206`：

| 位置 | 实际行为 | 推论边界 |
|---|---|---|
| [verl启动接线](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/rollout/vllm_rollout/vllm_async_server.py) | 取override或HF的quantization_config，仅把该配置和engine_kwargs传给默认函数 | 不传独立hf_config，函数无法看到expert_dtype |
| [verl默认函数](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/vllm/vllm_quant_utils.py) | Mapping且非NVFP4时，分别setdefault linear_backend和moe_backend为deep_gemm | 保留用户显式值，但缺省值到vLLM后已不再是auto；E8M0 scale开关不改变MoE候选 |
| [vLLM原生V4量化配置](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/deepseek_v4/quant_config.py) | 未忽略的RoutedExperts在expert_dtype=fp4且无NVFP4 override时返回Mxfp4MoEMethod；fp8则交父类FP8路径 | 同一顶层FP8配置可对应不同expert格式；该版本缺expert_dtype时默认fp4，不宜直接推广到所有checkpoint |

[Mxfp4MoEMethod](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/quantization/mxfp4.py)调用专用V4 selector。[selector与映射](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)将显式deep_gemm映为单候选DEEPGEMM_MXFP4，检查失败后抛出last_error，不进入auto列表。只有auto分支才依次检查TRTLLM MXFP8、DeepGEMM、Marlin、Batched Marlin；每项仍有设备、量化和布局等支持检查，列表存在不保证总有可用后端。

[DeepGemmFP4Experts](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py)要求is_deep_gemm_supported且设备属于100或120能力家族，排除SM90。源码注释的“SM100+”不能理解成所有数值更大的SM都必然通过。FP8 dense层可用DeepGEMM，不会使另一种FP4 expert kernel也在Hopper上可用。

### 能加载与可持续重载分开

同一vLLM oracle把DeepGEMM映为动态FP8 group-128 activation，Marlin的量化配置则走W4A16/BF16 activation；相同MXFP4权重不保证两者logprob或训练轨迹相同。verl的[refit名单](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/vllm/vllm_fp4_utils.py)包含Marlin与Batched Marlin，说明已有存储重装路线，**不是完整refit或CUDA graph数值验收**；不能据此把所有auto可加载后端都当成可更新权重。

固定main在verl、docs、tests中未命中`VERL_ROLLOUT_USE_DEEP_GEMM`。它出现在报告者提案中，不能把PR描述里的opt-out当作当前可执行开关。报告者H100的Marlin三步、GB200 DeepGEMM非法地址及Marlin速度观察均属其环境，GB200批大小阈值尚待隔离；源码支持Hopper的后端约束解释，不据此判定GB200根因或保证单改一个参数即可部署所有模型。

本轮七份完整Python源文件AST/compile通过，未导入vLLM、构造替身dispatch、运行加载或新增GPU实验。硬件差额仅补[原M5量化入口](../../experiments/handoff-m5.md#verl-量化层审计新版复验)的条件参考：已有获准、安全、匹配任务确需采用时，再记录实际安装commit、expert_dtype、最终backend及activation格式、一次加载与更新后同token数值；缺模型／匹配硬件或需新部署则停止，不另排队。方案归属与接口等待社区实质反馈，任何上游动作仍逐项待用户明确同意。

## QAT 的 mask 默认与 packed 接口

### 合并版不再替换任何 spec

[#8361](https://github.com/verl-project/verl/pull/8361)于2026-10-26 12:00:40 UTC合并，最终head `781c0dcc01bd17f46125afbabd8500f4b6b6b80c`，主线提交[4f8f090a](https://github.com/verl-project/verl/commit/4f8f090af34bb5962e78dccb0f061a5dd52df13d)，累计三文件5增7删。它比此前记录的8a697618再前进一提交：**删除整个spec替换helper及调用，默认和专用Bridge spec均保留，也不再额外setdefault mask。** 对用户显式provider override的正常应用仍在，不是禁止用户调整spec或mask。后续ModelOpt原地量化接线未因此被删除。

最终变化还包括[compressed-tensors FP4适配器](https://github.com/verl-project/verl/blob/781c0dcc01bd17f46125afbabd8500f4b6b6b80c/verl/utils/vllm/vllm_fp4_utils.py)：其子类直接调用FusedMoEMethodBase初始化而非Mxfp4MoEMethod初始化，现补 `activation_key=kFp8Dynamic128Sym` 和两个K3路径标志False。字段补齐不等价于新kernel能力或vLLM0.32部署验收。文档保留限定MXFP4权重×FP8 activation的Hopper说明，移除了“默认spec必须TP=1或EP=1”的ModelOpt SequentialMLP段落；不能再把该旧段当合并版的通用限制。

固定主线 `f78fd053e13c3d021e51de5e81afc54dddd13993` 中，QAT文档与FP4文件blob分别为 `42ae7a2cfa8de532ea0d0b75a24dcb87a8d40a02`、`66b1ac06c7e5bc2916a5857011c03a45340c928d`，与最终head和QAT合并提交相同。engine整文件blob不同，只因主线另有optimizer checkpoint恢复前onload；`_build_tf_config` AST完全相同。最终head、合并提交和该主线均不存在旧三项测试文件；本轮三份完整Python源文件仅AST／compile，未重复原三例或构造新替身。

维护者的全保留建议已落实为代码，随后有空正文APPROVED；但读取时PR正文仍写旧helper、80项CPU与ModelOpt默认spec表格，机器人摘要也停在8a697618。**合并、批准和旧正文都不能补出最终781c版的Qwen3-MoE各mode量化覆盖、导出／refit运行记录。** [原实验](../../experiments/qat-spec-selection/result.json)只属于cf6→8a历史对照，不能移作删除helper后的验收。此前OLMoE报告也保持其原版本范围，不因合并升级为全模型证明。

### 新版先保留模型结构

本小节保留合并前8a版本的源码与实验记录，不描述最终781c版。2026-10-26前次核[#8361](https://github.com/verl-project/verl/pull/8361)时仍OPEN／非draft／未合并，head `8a697618897d473a43b6d9e83ba6c3341a23067b` 比旧 `cf6eb5181e17b4db67afff110eabc61daf61fb4d` 前进一提交；base仍 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`，累计三文件100增5删。8a不再对所有QAT provider替换spec或补mask默认，下面初版“适用于进入QAT分支的调用”只描述cf6旧head。

[新helper](https://github.com/verl-project/verl/blob/8a697618897d473a43b6d9e83ba6c3341a23067b/verl/workers/engine/megatron/transformer_impl.py)按对象身份判断：仅 `provider.transformer_layer_spec is default_layer_spec` 时替换成ModelOpt spec并setdefault False；其余对象直接返回，spec与override字典均不改。显式True／None的保留规则只属于替换分支，不是全模型mask策略。它发生在provider应用overrides并finalize之前，不能单凭此身份检查认证最终模型结构。

这不是仅靠模型名猜测。[Bridge 0.6.2 OLMoE provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/olmoe/olmoe_provider.py)用自己的olmoe_layer_spec，将attention换成OLMoESelfAttention，对Q/K做全宽归一化；[OLMoE bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/olmoe/olmoe_bridge.py)对应HF q_norm／k_norm权重映射。通用spec的替换可能丢掉模型专用模块，不能把mask默认修正当成拓扑兼容。候选随后仍逐chunk调用[apply_qat_to_modules → apply_qat → mtq.quantize](https://github.com/verl-project/verl/blob/8a697618897d473a43b6d9e83ba6c3341a23067b/verl/utils/qat/modelopt.py)；这一调用接线不证明具体模块量化覆盖。

本机运行新head[原三项测试体](https://github.com/verl-project/verl/blob/8a697618897d473a43b6d9e83ba6c3341a23067b/tests/workers/test_megatron_qat_layer_spec_on_cpu.py)及其原假Bridge fixture，接AST提取的两版原helper：旧版2过／专用spec断言失败，新版3过。见[复现脚本](../../experiments/qat-spec-selection/verify.py)与[原始结果／源文件哈希](../../experiments/qat-spec-selection/result.json)。Python3.12.13／pytest9.1.1；命令为 `/private/tmp/verl312/bin/python experiments/qat-spec-selection/verify.py ../qat-spec-source`，输入分别是上述两head的完整transformer_impl.py和新head原测试，重现时按result.json固定路径／commit获取。只省略生产模块重导入，保留原测试定义与fixture；没有原生pytest套件、真实Bridge／TE／ModelOpt／模型或设备执行。七份完整源文件AST／compile通过；初次按旧checkout路径utils/modelopt/qat_utils.py请求新head返回404，核新接线改取utils/qat/modelopt.py，不将缺路径判断成缺量化调用。

作者更新报告须与本机结果分开：其80项CPU与Bridge0.6.2配置级provider检查不是本机运行；其1张GH200的OLMoE FP8_BLOCK linear+MoE／qat-mode fp8在旧版加载norm shape失败、新版跑通，报告128个enabled TE quantizer及k3 KL约6.3e-3（关QAT约9.1e-3）。这只支持其OLMoE环境，不覆盖全部自定义spec；dummyweights rollout收到导出权重也不是完整refit数值验收。Qwen3-Next／文本Qwen3.5仍只是作者源码观察，不能把VL provider和文本结构混称一类。

当时的[人工反馈](https://github.com/verl-project/verl/pull/8361#issuecomment-6414899795)来自wuxibin89：建议连默认spec也保留，前提是验证Qwen3-MoE的原TE spec在各支持QAT mode的量化器覆盖、前反向、导出expert scales与refit。**该8a head仍替换默认spec，后续781c才移除**；本项目不代作者实现。8a文档另称默认spec的ModelOpt SequentialMLP不支持TP+EP同时启用，不能外推为全部TE／ModelOpt后端限制，且该段未进入最终版。仅更新[原M5条件参考](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不新排实验或改专项；社区写入仍待用户具体授权。

### 初版 mask 调用链（历史版本）

QAT 的量化层选择、attention mask 与 rollout 部署格式是三个独立条件。[verl #8361](https://github.com/verl-project/verl/pull/8361)读取时 open、非 draft、未合并，固定 head `cf6eb5181e17b4db67afff110eabc61daf61fb4d`，base 和实际共同祖先均为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`；三文件 69 增 5 删。它为 QAT 选 ModelOpt layer spec 时补一个默认值，不是新量化方案，也没有验证所有 packing 组合。

该 base 的 [pyproject.toml](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/pyproject.toml)与 [uv.lock](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/uv.lock)声明 Bridge 0.6.2、MCore 0.19.2、TE 2.16.1。这些是依赖声明，不是某台机器的安装证明；下面按对应官方 tag 的固定提交说明接口，而非使用本地旧 checkout 的依赖文本。

| 组件 | 选择或转换 | 对 packed 路径的影响 |
|---|---|---|
| Bridge 0.6.2 | `use_arbitrary_attention_mask=None` 时，以实际 CP world size 是否为 1 选择 arbitrary | CP1 默认走 arbitrary，不只是查看配置字典里的 CP 数字 |
| MCore 0.19.2 ModelOpt spec | True 选 arbitrary，False 选 causal | mask 选择独立于是否传入 packed QKV |
| MCore TE adapter | THD 且相应 TE 条件成立时，causal 改为 padding_causal，no_mask 改为 padding；arbitrary 不改 | 不能把 arbitrary 自动解释成 packed causal |
| TE 2.16.1 | THD 分支要求 mask 类型名包含 padding，另检查 QKV 维度与 cu_seqlens | 通过 mask 类别检查只是完整输入契约的一部分 |

固定源码：[Bridge provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/gpt_provider.py)、[MCore ModelOpt spec](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/post_training/modelopt/gpt/model_specs.py)、[MCore TE adapter](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/extensions/transformer_engine.py)、[TE attention](https://github.com/NVIDIA/TransformerEngine/blob/c9877beb87ad7e711e1869dd0b5062167ede447a/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py)。由这条调用链可解释作者遇到的 CP1、remove_padding 路径：未覆盖时默认 arbitrary，adapter 不转换，随后不满足 TE 的 THD mask 类别要求；显式 False 则经 causal 转为 padding_causal。后者名字有 padding，并不意味着必须保留实际 padding token；仍须满足累计长度、布局等其余条件。

候选 [transformer_impl.py](https://github.com/verl-project/verl/blob/cf6eb5181e17b4db67afff110eabc61daf61fb4d/verl/workers/engine/megatron/transformer_impl.py)用 `provider_overrides.setdefault("use_arbitrary_attention_mask", False)` 补默认。显式 `override_transformer_config` 已先写入字典，因此 True 仍是 True，显式 None 仍可触发 Bridge 的自动选择；不是强制关闭或合法性校验。helper 自身也没有 CP1 条件，适用于进入该 QAT 分支的调用。覆盖随后交给 provider finalize，并非仅改一个未被使用的局部值。

原 [两项 CPU 测试](https://github.com/verl-project/verl/blob/cf6eb5181e17b4db67afff110eabc61daf61fb4d/tests/workers/test_megatron_qat_layer_spec_on_cpu.py)使用假 Bridge 模块与 SimpleNamespace，检查缺省 False 和保留显式 True；它们不构造真实 ModelOpt spec 或 TE attention。本轮只对 head/base 生产文件、该测试文件及上述四份官方源文件共七份做 AST/compile，并静态确认 setdefault 参数；没有执行这两项测试或真实 Bridge/TE。作者的 19 项 CPU 报告不记为本机结果。

作者在 GH200 与 RTX 5090 的报告采用等价显式 False override，明确不是该候选分支的 GPU 验收；小模型的 k3 KL 观察也不是长程训练结论。同一 PR 的 [QAT 文档](https://github.com/verl-project/verl/blob/cf6eb5181e17b4db67afff110eabc61daf61fb4d/docs/low_precision/qat.md)另记录 vLLM 0.29 下 GH200 的 MXFP4 权重、FP8 activation experts 预设启动受限，涉及其 FP8 1x128 activation scheme 与可用 MoE backend，不可泛化成全部四位模型或全部 Hopper 不支持。这里沿用作者限定的部署报告，未独立核验 vLLM dispatch；mask 修正不能补出缺失的 MoE kernel。

真机条件仅补入[已有 M5 量化入口](../../experiments/handoff-m5.md#verl-量化层审计新版复验)：M5 在已有获准、匹配的 QAT 工作中独立选用时，固定实际安装版本、最终 mask、QKV layout 与累计长度，再判断 forward；部署 backend 另行判断。不新增训练队列、应用候选或修改专项，任何上游可见动作仍逐项待用户明确同意。

## 版本与问题

读取时 PR open，作者 wengeezhang；固定 head `0defd9a231cb876ce56812611b0afb3b2b25831a`，base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。这是量化支持堆叠的第 7 部分，依赖前序 PR，并按 #8287 的导出接口重做；整个 PR diff 包含前序提交，不能全算作本次审计新增代码。

作者旧版 B200/ModelOpt 观察发现：训练与 rollout 的忽略规则看似相同，dense `gate_up_proj` 却可能受 `mlp.gate` 前缀匹配影响而走不同精度。作者据此将新版改为观察真实导出权重流，而非重建引擎忽略规则。**这些旧版真机结果不是当前 head 的验收**；作者明确新版尚未复跑硬件。本机也没有做真机验证。

完整阅读固定 head 的 [审计模块](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/quant_layer_audit.py)、[原始 CPU 测试](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/tests/utils/test_quant_layer_audit_on_cpu.py)、[QAT 导出模块](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/qat/compressed_tensors.py)；worker、Megatron、vLLM 只读相应接线片段，不代表审过整个训练/推理栈。

## 三类证据各自证明什么

| 来源 | 实际观察 | 不能直接推出 |
|---|---|---|
| 训练模块 | Transformer layer 子模块存在非空 TE `_fp8_workspaces`，或 offload 前保留的 `_verl_fp8_workspace_seen` 标记 | 每个权重、当前 step 都实际运行同一种 FP8 kernel |
| 导出权重流 | 原生 FP8 dtype，或 `.weight_packed` 规范化为量化权重；普通 weight 记未量化 | packed 数值、scale、布局、量化 recipe 或内容相等 |
| live vLLM 模块 | 解析实际 `quant_method`/scheme，处理名称映射、packed QKV/gate-up、MoE holder、LoRA base layer | 所有 worker 一致，或该方法已执行正确 kernel |

训练信号是 **layer 粒度**：靠类名与 `layer_number` 识别层，将全局 1-based 层号转为 0-based；不存在于本地 PP stage 的层返回未知。标记可跨 workspace 清理保留，不是逐次 GEMM trace。层外参数还受 `outside_layers_quantized` 判断影响。

导出记录只看格式，不比较 tensor 内容；引擎侧把方法描述中不以 `unquantized` 开头的项当量化。两阶段比较的是“量化/未量化”布尔值，不能认证相同位宽、recipe 或误差。

`kept_high_precision_by_design` 复用 QAT 模块的真实 `DEFAULT_IGNORE` 与匹配函数，覆盖 lm_head、embedding、visual、mtp、router/shared-expert gate 等。训练/导出比较遇到这些权重会**先跳过**；`by_design` 计数不是“已经观测它们全为 BF16”。非本地 PP 层的未知项也不判一致或不一致。

## 完成、沉默与失败的边界

审计只为量化 rollout + TE FP8 recipe 条件创建；默认 `warn`，配置也可关闭。不能从“没有告警”反推审计已创建、运行且匹配。

1. `record` 包装导出 generator，逐个记录并原样 yield；**完全消费流之后**才进行训练/导出比较。提前停止消费不等于完成比较。
2. 首次没有 FP8 workspace 信号时等待。两次不同同步仍无信号，或禁用 parameter transpose cache 时，结束并解释 **no verdict**；未知状态不是一致性通过，`raise` 模式也不把这种说明变成 mismatch 异常。
3. 有信号后完成一次判定，设置 `done`/verdict，再告警或抛错；后续同步不重复审计。它不是每步持续验证，也不保证后续切换精度仍被发现。
4. `raise` 位于流被消费之后，不能代表“权重从未应用”或自动回滚；异常完成点与接收端事务是两件事。

引擎复查需要已有 verdict，且只复查一次。无法解析的名称/PPMissingLayer 返回未知并跳过；应保留 checked、unresolved 的覆盖范围。空/None 返回也会消耗这次复查机会，不可写作“全量匹配”。合并 worker 返回时取首个有效非 None 方法，不做跨 worker 冲突检测或多数共识。

这些是观察边界，不在缺少运行证据时把它们定性成新 bug。

## 接线路径并非同一覆盖范围

固定 head 的 Megatron 导出在 quantizer 之后记录，adapter-only 路径不做同样审计；workspace 清理前留下标记。`engine_workers.update_weights` 的 colocated/naive 分支在 rollout 更新返回后、恢复 cache 前尝试引擎复查；非 naive checkpoint 路径发送后提前返回，不经过该复查。获取引擎方法的 RPC 异常只警告退出，不能视为匹配证明。

SGLang MXFP8 loader 使用独立 engine guard，逐请求写入前检查格式；接续源码分析见下节。未穷尽 delta/replica 等全部路线，不能将 colocated vLLM 的覆盖外推为所有 rollout 后端。

## SGLang 分桶加载与失败清理

2026-10-21 接续阅读同一 head 的完整 [MXFP8 loader](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/workers/rollout/sglang_rollout/mxfp8_refit_loader.py)、[数值探针 helper](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/mxfp8_refit_check.py)，以及 [rollout 调用片段](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L364)。PR/API 再核仍 open，head/base 未变。以下是源码结论，没有运行 loader 或其单测。

同一权重同步会分为多个请求。普通 bucket 只加载，最后的 `FINALIZE_MARKER` 请求才统一派生 kernel 布局；逐 bucket 派生会碰到后续尚未写入的专家 scale，也可能重复处理已变换的数据。调用方先依次 await bucket，再 await finalize，正常返回后才 flush cache、更新 global step。这个先后关系不证明网络中断后的远端状态或全 rank 原子提交。

| 阶段 | 本地动作 | 证据边界 |
|---|---|---|
| 请求预检 | 对本请求可解析的 weight 比较输入/引擎 dtype 类别，再调用 `load_weights` | 拒绝的是本请求，不撤销前面成功写入的 bucket |
| 首 bucket 暂存 | 所需 MoE canonical scale buffer 全部分配成功后，才重绑参数；随后加载 | 分配失败不遗留部分重绑，不代表加载过程事务化 |
| finalize | 取 MoE 参考快照，处理 dense/MoE 布局，回写 live storage，再做抽样自检 | 处理/拷贝依次发生，中途异常可能已修改前面的层 |
| try 内异常 | `_abandon_sync` 移除同步标记，令暂存 scale 参数重新指向原 live storage，然后重抛 | 恢复指针，不恢复全部旧权重/scale 数值，也不是安全继续生成许可 |

格式 guard 只比较 `{BF16, FP16, FP32}` 与“其他 dtype”，并非验证精确 FP8 recipe。scale 等非 weight 叶子不走这项比较；未知名称跳过，解析 dtype（包括未知结果）按名字缓存在 model 上。因而首条一致日志或返回的 checked 数量不是全同步、全参数覆盖证明。

### Canonical 数据与图捕获存储

MoE scale 只有在 live shape/dtype 不再是 canonical `[... , K/32]` uint8 时才暂存；未改变 shape/dtype 的 runner 可能直接覆写 live buffer。暂存 buffer 填 `0xFF`，检查开启时，finalize 在处理对应 MoE 层前检查残留哨兵；它发现未写条目，但不认证写入值来自正确版本，也不覆盖所有未暂存 scale。

重新派生后，结果必须恢复为原 live shape/dtype；必要时 `live.copy_(new)`，再将参数指回 live storage，并检查所有暂存项均已回绑。这里分开维护两个条件：**内容已更新**与**CUDA graph 捕获的存储仍可读到内容**。CPU `data_ptr` 断言不能替代真实 graph replay。源码已有 dense DeepGEMM→FlashInfer fallback 可能重绑 swizzled scale 的 TODO，不能只凭 MoE 的回绑逻辑宣称所有 backend 安全。

### 自检通过与跳过必须分开

自检选最小 dense 层、最小 MoE 层的 local expert 0，默认 seed 0 的 8 行输入。比较的是 kernel 输出与 canonical 权重/scale 的反量化参考，不是训练 actor 与 rollout 的参数相等；默认相对误差阈值 0.25 也不是通用健康界或完整模型认证。

MoE EP>1、TP>1 且无 reduce_results、无法用 logits 固定单专家的路由或非 canonical scale 会跳过。helper 对非有限 reference 和超阈值/非有限输出报错，但对探针内部的一般 Exception 警告并返回 NaN；loader 不将这类返回重新判失败。因此 finalize 正常返回仍可能伴随 skipped probe。`VERL_MXFP8_REFIT_CHECK=0` 同时绕过格式 guard、暂存 scale 哨兵检查和数值自检，不仅关闭日志。

对应[原始测试](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/tests/utils/test_sglang_mxfp8_refit_loader_on_cpu.py#L421)的分桶、失败清理、分配失败和预检案例，只读核对；测试中的 SGLang backend、TopK、quant method 与 swizzle 本身为替身（如 clone+1）。本轮不新增替身、不声称复跑作者 loader 测试。现有 CPU 环境缺 SGLang、hydra、omegaconf、tensordict、compressed_tensors，未补装或启动服务。

源码 SHA256：loader `073b917564abd783e724aed3759912ded9a74eb65444ce6a633989d2e4f35b30`；loader 测试 `65945e46ef916fe4e059d03a7df2dd7089d6b7ee934bf050fdcf60b6d6132247`。与下方 24 项 audit 单测是不同执行范围。

## 新旧 SGLang refit 路线的覆盖关系

2026-10-21 接续核对新 [#8306](https://github.com/verl-project/verl/pull/8306)，作者 wuxibin89，open；本轮固定 head `1beb35554a763d18ed76fcbefaac1b224eaa1eeb`，不是上一轮列表看到的 `457aa56d…`。旧 #8132 仍 open/head `0defd9a2…`。**两条是有重叠的并行方案，不能按编号较新就视为替代，更不能把测试和配置直接拼接。** #8306 作者也明确标注与 #8130 的 MXFP8 export/refit 重叠，以及与 #7971 NVFP4 方向的概念重叠；没有据此新增我方实现。

API 的目标 base 为 `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`，但与新 head 的 merge-base 是 `5a8dd519ab32bfd36d2509a137f830ba79e89c54`。以 `base...head` 核对8个PR改动文件，与API列表一致；没有新增/修改测试。直接 `base head` 两点比较还会显示目标分支独有的 optimizer-offload 测试差异，不能误说是本PR撤销了该修复。

### 扩展格式不等于继承全部审计

| 范围 | 旧笔记固定 #8132 路径 | 新 #8306 固定 head |
|---|---|---|
| 专用 refit 格式 | 已读的 whole-model MXFP8 loader | MXFP8，以及 BF16 dense + MXFP4/NVFP4 routed experts；拒绝 MXFP8 dense + MXFP4 MoE |
| 关闭同步 | bucket加载后专门marker触发派生 | 同样要求完整bucket序列；显式拒绝custom refit的delta_flush |
| 存储处理 | MoE scale回绑；dense fallback留有TODO | 加入dense swizzled/deepgemm scale捕获与copy-back，以及FP4参数/派生量回绑 |
| 保护与证据 | 选层审计、dtype guard、未写scale哨兵、抽样数值自检；本机24项audit单测 | 已读三个新模块未接入这些审计/guard/数值probe；作者报告Blackwell观测，本PR diff没有提交配套CPU测试 |

这是不同分支的覆盖对照，不是说新PR从旧实现“删除”了保护。旧24项测试和作者旧loader测试也不能算作新路线回归通过。

完整阅读固定 head 的 [sglang_quant_utils.py](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_quant_utils.py)、[FP8 helper](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_fp8_utils.py)、[FP4 helper](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_fp4_utils.py)，以及export/worker接线差异和文档相关片段；没有审完整SGLang或全部训练路径。

### 配置和消息边界必须成套核对

新配置入口优先解析rollout override，再看模型native配置；refit kind由配置结构判定，不只凭“MXFP8”名字。all-MXFP8 compressed-tensors配置转native SGLang schema，已知regex ignore映射为引擎名称，未知regex拒绝；部分ignore还经 `SGLANG_FP8_IGNORED_LAYERS` 传递。导出复用CT量化器，将 `.weight_scale` 名字改为 `.weight_scale_inv`，该包装器没有把tensor数值取倒数。

新loader FQN是 `verl.utils.sglang.sglang_quant_utils.load_and_reprocess`，marker也与旧 `mxfp8_refit_loader` 不同。非NVFP4配置的接线要求DeepGEMM，NVFP4留给SGLang选择；FP4 helper另检查支持的runner条件。启动配置、exporter、每bucket的load_format、finalize marker及实际backend必须属于同一版本路线，不能混用旧请求或根据另一分支的默认值推定兼容。

新 `_finish_sync` 按 FP8 dense、FP8 MoE、MXFP4、NVFP4 顺序重建，完成后移除同步标记。FP4 staging可直接复用live tensor或同一存储的dtype view，并不总是额外副本；`restore_*_live` 恢复参数/派生量引用与alias，未备份全体旧内容。因此失败清理仍不等于参数回滚或跨rank事务。dense捕获/copy-back分支解决的是源码中的存储身份处理，不能仅凭该分支宣称真实graph replay已经验证。

### 本轮只确认源码与语法

作者报告Qwen3.5-35B-A3B、Megatron EP8/SGLang TP2、Blackwell默认CUDA Graph条件下的三类量化mismatch观测；这是作者证据，不是本项目复跑，也不是全部配置的通用阈值或长期训练证明。作者说明CPU套件受torchcodec/libc10_cuda.so导入问题阻挡，最终diff未带配套测试。本机的torchcodec本身未安装，没有复现作者同一故障。

M1只在Python3.12.13对三个固定Git blob做AST解析和内存compile（272/211/260行），均通过，不导入或执行模块、不生成pyc；现有环境也缺SGLang、compressed_tensors、hydra/omegaconf。没有新造fake workspace或用旧单测认证新loader。

SHA256：quant helper `9557c1c4ae3b65c41d36b0a462e14c66e737691cf99e3aad1c4f6431050c08a6`；FP8 helper `cd167e604bf66231673d7571664592a6c21ad3f5c4edb9aab80f2a93f3679b57`；FP4 helper `266f9c4fa4a2501fab01d04fe67d87b91c0a766d9b71832c27ece05bb5db28b0`。真机接续补入[同一M5条目](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不要求M5同时验证两套或自动merge/cherry-pick；先按实际资源与已选路线决定。M5已有4090记录不等于具备本PR所述Blackwell验证条件。

## 本机复现：24 项原始 CPU 单测

[复现入口](../../experiments/scripts/verify_verl_quant_layer_audit.py) 从指定只读 Git 仓库取旧 #8132 固定 head `0defd9a2…`，不覆盖 #8306，不检出、不安装、不下载模型：

```bash
python experiments/scripts/verify_verl_quant_layer_audit.py /path/to/verl-git
```

2026-10-21，Python 3.12.13、torch 2.9.1、pytest 8.4.2：**24 passed**。完整原样执行审计、QAT 模块与测试体/fixture，只用 namespace shell 隔离 `verl`/`utils`/`qat` 的包初始化，并用 collector 指向原始测试模块；禁用外部 pytest 插件、conftest 和缓存。复用已有 Git 加载 helper，打印 SHA256：

- `quant_layer_audit.py`：`d4c02f4b317e37b7f0d12c84c87f3d8bfb44faa39fb6287cdc6f982dc823a27d`
- `test_quant_layer_audit_on_cpu.py`：`fbd62da8919235d9b90993dc21f66fabd77b4f539e02cde67342ffdb2062dfce`
- `qat/compressed_tensors.py`：`bd466f3abca89b642cc5af5637236b1920466c432be9e0d45b9fd320f4bc6c83`

真实 CPU torch tensor/Module/dtype 参与，但原测试自己构造 Megatron/TE workspace、vLLM quant method 等替身。它验证记录、跳过、告警/抛错、等待、映射等**单元逻辑**，不是 TE forward/offload、实际 compressor、引擎/RPC、SGLang loader 或真实训练。外部 compressed_tensors、Transformer Engine、vLLM/SGLang 未导入；现有环境也缺这些包。本轮未运行作者声称的 63/137 项组合，不补装依赖来伪装完整栈验收。

## 如何用到后续诊断

量化层审计补充的是“选层/格式”的可观测性；[策略来源笔记](verl-async-logprob-provenance.md)讨论的权重身份、raw/processed 概率与配对 token 仍需独立证据。这里的 Megatron/TE 量化路径不能直接解释 #8250 私有 FSDP2 异步训练退化，也没有推出新修复。

两条固定路线的真机接续统一登记 [M5 真机交接](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，先选版本路线，再记录分桶/finalize、实际 backend、graph，以及该路线实际具备的 guard/probe 覆盖。状态是待匹配环境，不代表已接单。M1 的本项源码学习收束，不探测服务器、不竞争已有 PR、不改 AReaL，不能用更多替身替代缺失的真机证据。

## ModelOpt 模块忽略与子量化器

**权重导出并正确加载，不保证 actor forward 已按要求跳过 fake quantization。** 2026-10-22 只读已有 [#8329](https://github.com/verl-project/verl/pull/8329)，作者LZC-BELIEVER，读取时open。固定head `5c7e2d2ed49bd9f448043883d96b6e191e187e95`、目标base `16ec093cc4a235c603c29aed1f2c4e9f551fab48`、merge-base `f2923cbf154731918b1e36db8c567433f349fede`；真正PR diff为2文件66增5删：生产helper新增3行，另一项是测试扩充及cpu→gpu文件改名。不是#8132的审计实现或#8306的refit路线更新，不能自动叠加或借用其验证。

### 模块名和量化器名属于不同层次

[`_ignore_patterns_to_quant_cfg`](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/verl/utils/qat/modelopt.py#L281)旧版只追加 `{"quantizer_name": key, "enable": False}`。给模块模式 `*self_attention.linear_proj`，实际叶子名却是 `...linear_proj.input_quantizer` 和 `...linear_proj.weight_quantizer`，后缀使旧模式不能匹配。候选保留原规则，再追加 `key + ".*"`；点号使精确结尾in_proj_a不会顺带选择in_proj_ab，而不是粗略追加星号。

[ModelOpt 0.44.0 官方消费代码](https://github.com/NVIDIA/Model-Optimizer/blob/c897fbeaaff66d53d61033f107885b7c5432f235/modelopt/torch/quantization/conversion.py)按列表顺序应用规则，后匹配项可覆盖前项；只对TensorQuantizer/SequentialQuantizer匹配名称，并使用fnmatch，另有专家名称归一化。只有enable=False而没有cfg的规则切换启用状态，不重写其他数值属性。verl将ignore_cfg放在preset/default规则之后，因而它能覆盖前面的enable；不是修改一份未被消费的配置字符串。

本机两版原helper配标准库fnmatch的名称选择对照如下，输入为合成名称，不构造量化器：

| 用户模式 | base选择 | head选择 | 范围 |
|---|---|---|---|
| `projection` / `projectio?` | 无 | projection的input、weight | 模块模式扩到后代 |
| `projection.*` | 两个叶子 | 两个叶子 | 已有通配用法不变 |
| `projection.weight_quantizer` | 仅weight | 仅weight | 保留直接叶子规则，input不被关闭 |
| `*linear_attn.in_proj_a` | 无 | 仅in_proj_a的两个叶子 | in_proj_ab不被选中 |
| `*self_attention.linear_proj` | 无 | 对应两个叶子 | 实际Megatron模块名 |

每版共11组精确选择集合控制及空列表检查通过，另覆盖 `*input_quantizer`、3个旧alias和一个未映射名称。alias字典只对完整pattern相等时替换：`lm_head`→`*output_layer*`、`*mlp.gate`→`*router*`、`*self_attn*`→`*self_attention*`。它不是任意HF→Megatron名字翻译器；例如 `*self_attn.o_proj`不会自动变成 `*self_attention.linear_proj`。actor使用glob，rollout示例使用显式re:正则，必须对应到同一层，不能照抄同一字符串就认为两边选层相同。`.*`可匹配更深后代，也不是“只允许一层”的路径解析器。

最小原helper复查（本地须已有固定Git对象；这里只展示模块边界与直接叶子两组）：

```bash
python3 -B - /path/to/verl-git <<'PY'
import ast, fnmatch, subprocess, sys
for rev in ['16ec093cc4a235c603c29aed1f2c4e9f551fab48',
            '5c7e2d2ed49bd9f448043883d96b6e191e187e95']:
    src = subprocess.check_output(['git', '-C', sys.argv[1], 'show',
                                   rev + ':verl/utils/qat/modelopt.py'])
    f = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
             and n.name == '_ignore_patterns_to_quant_cfg')
    ns = {}
    exec(compile(ast.Module(body=[f], type_ignores=[]), 'original-helper', 'exec'), ns)
    for pattern in ['projection', 'projection.weight_quantizer']:
        rules = ns[f.name]([pattern])
        print(rev[:8], pattern, [n for n in ['projection.input_quantizer',
              'projection.weight_quantizer', 'projection_extra.weight_quantizer']
              if any(fnmatch.fnmatch(n, r['quantizer_name']) for r in rules)])
PY
```

### 配置检查与真实行为证据

[原测试](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/tests/utils/test_modelopt_mxfp8_on_gpu.py)不再只断言配置末尾出现ignore字符串，而是调用真实mtq.quantize后检查两个叶子的is_enabled、兄弟隔离、直接叶子模式、alias和原preset未变；BF16例要求forward、weight.grad和input.grad与未量化参考精确相等。关闭量化器不等于冻结该层参数。这些是原测试的行为要求，本机未执行它们。

作者PR正文报告ModelOpt0.44.0、compressed-tensors0.17/0.18两环境各34项通过；H100/Qwen3-0.6B、单seed、两臂各10步GRPO中，被忽略28层的56个启用量化器变成0，首层QDQ误差由0.0859375到0，参数仍逐步更新。与此同时，选定首层导出BF16与vLLM实际加载权重在**修复前后均11/11相等**。这说明字节refit检查与actor计算精度可以给出不同层面的答案，不能只凭前者认证forward一致。作者未把不同轨迹的reward/K3均值当改进证明，也未跑多GPU、MXFP8端到端或Qwen3.5/GDN；这些都是作者报告，不是本项目真机结果或当前head全栈验收。

固定head的[GPU单测workflow](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/.github/workflows/gpu_unit_tests.yml)排除`*on_cpu.py`；改名让文件进入该收集范围，并不保证任务实际触发、依赖存在或测试通过。新增BF16例未显式将tensor送到CUDA，文件名本身也不是每个断言都在GPU执行的证据。

M1 Python3.12.13只执行两版完整原helper AST及标准库名称控制，对两版modelopt.py和原测试文件共4模块内存compile通过；未导入torch/ModelOpt/compressed-tensors、运行原pytest、量化器、梯度或设备。已有环境find_spec确认缺modelopt/compressed_tensors，不补装、不以stub替代真实mtq；临时控制不新增永久一次性工具。SHA256：生产base `8ef392c4592728c77cd693d0762564ee74b5c2f7867ac83fc2aad151f3801e9e`、head `c926b4987e48c5e487865c62046de0547bff75cb551bbd50691a3d16129b1828`；测试base `27987bd731b460fb69d13e67bcabb58b041caba1ee80255fed6b7591f1fd8c1f`、head `76ccbb40aa951c1695b1912ccb520d4be5f42fa6bd1608afa3d1b56137382b9b`。真实行为缺口补[原M5量化条目](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不另排队/应用或开展H100训练；本题收束，不据此诊断#8250或扩写量化器替身。

## 新版运行期检查与指标哨兵

**告警、选层判定、抽样kernel自检和拒绝不支持配置不是同一层保证。** 2026-10-23接续M5登记的[#8327](https://github.com/verl-project/verl/pull/8327)，读取open、未合并，固定head `a2012b598e2e5eabc9a79847ed0a245dc73bc478`、目标base `d707198afbe5d699173cd27e60ff3f9221ce5a09`，官方compare祖先`d1390eea0c94467d4ae1387eeda9e2f78f8d73ea`（ahead7/behind1），24文件4442增7删。作者明确它基于#8287/#8306/#8308，替代旧拆分#8128/#8130/#8131/#8132；不是将旧审计和另一作者#8306手工拼接的许可。旧版测试不能直接算作新版回归。

### 保护范围与默认行为

| 机制 | 新候选的观察或处理 | 不能据此认证 |
| --- | --- | --- |
| vLLM BF16 MoE重载修复 | 量化模型中未量化MoE也重新做kernel prep，包括同shape的gate/up换序 | 仅shape相等不证明布局语义相同；本机未跑真实runner |
| SGLang硬拒绝 | delta+任意量化，以及已改布局的BF16 MoE，在对应写入前拒绝 | 不因关闭告警就支持这些配置，也不回滚已完成的旧bucket |
| post-sync self-check | 每次sync后，对选中的小dense层/一个expert调用kernel，与canonical反量化参考比较 | 不是全层/全expert检查，更不是CUDA graph replay验证 |
| quant-layer audit | TE层级信号→导出格式；vLLM另复查接收方法，SGLang逐bucket比dtype类别 | 一次选层判定不认证每步数值；unknown/no verdict不算匹配 |
| quant sentinel | 从现有步级指标发现非有限差值、异常KL、近乎全截断 | 不能区分合法难题/随机模型与量化故障，也不定位根因 |

接线见[vLLM refit](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/vllm/vllm_quant_utils.py)、[unquantized MoE重派生](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/vllm/vllm_unquant_utils.py)、[SGLang loader](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/sglang/sglang_quant_utils.py)与[候选Checks文档](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/docs/low_precision/mxfp8.md)。这是关键函数/调用点的源码对照，不是完整审查24文件或运行量化后端。

三项可观测检查各有独立环境开关：VERL_MXFP8_REFIT_CHECK、VERL_QUANT_LAYER_AUDIT、VERL_QUANT_SENTINEL，默认均warn，发现问题后日志记录并继续；raise才对各自发现的问题抛错，0关闭该检查。硬拒绝配置是另外的代码路径。SGLang check_sync_matches_engine在model.load_weights之前比较高精度/量化类别，但默认warn仍会继续写入；不能把“写入前检查”改述为“默认拒绝不匹配”。

[refit helper](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/mxfp8_refit_check.py)对一般probe执行异常仍warning并返回NaN，故raise也不代表每种backend都完成了probe。[auditor](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/quant_layer_audit.py)仍等完整消费导出流、等TE信号后做一次判定；无信号可终止为no verdict，引擎复查也只请求一次。异常发生前可能已更新部分权重，以上机制均不提供同步事务或参数回滚。

### 指标哨兵的输入与缓存

[quant_sentinel.py](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/utils/quant_sentinel.py)仅用标准库。它按structured rollout.quantization的linear/moe/config_file或本地config.json的native量化声明启用；空的结构化配置对象不代表量化，远程模型路径不会在这里下载配置。没有识别到量化就不检查，不能从沉默反推模型健康。

三条规则是：有效比较存在时diff_mean非有限；KL非有限或严格大于阈值（默认1.0）；clip_ratio≥0.999达到连续观测次数（默认2）。ignore_eos或clip_steps=0只关截断规则，另外两条仍在。阈值是告警策略，不是健康模型的普适分界。缺少或不可转为float的指标不会产生该项判定；空metrics不等于通过健康检查。

连续计数按收到的clip指标推进，step只用于消息，不去重或检查编号相邻。CPU控制中，step1的clip=1、step2缺clip、step3的clip=1仍触发；重复step7两次也触发。正常trainer一次一调用的路径不因此构成故障，但额外监控消费者不能假定标签会自动防重复。源码在收到可转float的clip后，以clip≥0.999递增、否则归零；缺项不进入该分支。

trainer hook首次构造后缓存sentinel；解析环境阈值/配置抛错时，会警告并缓存disabled实例，即使请求raise。CPU控制先给非法KL阈值、再修正环境，缓存实例仍关闭。应保留实际enabled/mode与配置失败日志，不只保存环境变量字符串。[V0调用点](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/trainer/ppo/ray_trainer.py)位于步级指标汇总，前面已经可能执行actor update、save和权重同步；此处raise不是生成前拦截或当前步撤销。[V1](https://github.com/verl-project/verl/blob/a2012b598e2e5eabc9a79847ed0a245dc73bc478/verl/trainer/ppo/v1/trainer_base.py)也接入指标组装，本轮未执行任一trainer。

### 本机测试与作者报告分开

[复查脚本](../../experiments/scripts/verify_verl_quant_sentinel.py)完整加载原sentinel和原测试，不摘取方法、不改断言，通过pytest collector保留原fixture/参数化：16函数展开20例，全部通过；另5组边界控制通过。预加载精确模块名并设置阻断导入器，拒绝verl父包、torch、Ray/vLLM/SGLang/CT/TE；没有package shell或引擎stub。原tmp_path测试只在自己的临时目录写config.json。执行环境Python3.12.13/pytest8.4.2，不运行上游conftest或外部pytest插件。

```bash
python -B experiments/scripts/verify_verl_quant_sentinel.py /path/to/verl-git
```

原模块SHA256 `7e2ac00ca4ae8671c6ec2361c6ec957ddfcb9785e70f98d7a5e62a04aa63255f`，原测试`747dd52b7c274ed6677ff394c07670c55e56d78c6ebee5c8e90c357a747333b1`。全部23个修改Python生产/测试文件另只做AST与内存compile；没有执行另外五套CPU测试、审计类、加载器、量化器、kernel、训练或设备发现。作者六文件合计150通过包含CT与所需引擎替身，不等于本机20例覆盖整个PR。

当前API正文已追加cf32f6ae在1×B200上的vLLM/SGLang正常与scale故意破坏对照，以及带entropy bonus使权重实际变化的短跑；head a2012b59相对cf32f6ae只有两行文档图变动。旧网页正文仅述#8314硬件的状态不能覆盖这次补报。结果仍是作者报告，不是本项目验收或任意卡兼容证明；作者也另列主线vLLM BF16 expert默认backend拒绝、SGLang ignore前缀/精确匹配分歧，均未在本PR修复。

真机接续更新[原M5量化条目](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不新增队列、拼接路线或改变其环境。本题完成新版关系与sentinel范围核对；其余数值/graph/实际backend仍按匹配资源和独立选型保留未测。

## 写入前检查与失败握手

**格式检查前移，缩小了错误写入范围；它没有把权重同步变成可回滚事务。** [#8327](https://github.com/verl-project/verl/pull/8327)在2026-10-24读取仍open，head `ba2852ec76e3a11cc90fc0883ffd70416c831661`、API目标base仍`d707198afbe5d699173cd27e60ff3f9221ce5a09`。相对上节旧head `a2012b598e2e5eabc9a79847ed0a245dc73bc478`，20文件1091增603删，已非只有文档更新。以下是新增公共检查、两引擎接线及IPC接收顺序的局部学习，不是完整PR review。

### 检查位置与开关变化

旧vLLM路径在训练侧判定后，向引擎询问模块的quant method；新版[SyncFormatCheck](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/verl/utils/quant_sync_format_check.py)在引擎进程中读取当前桶将写入的参数dtype。训练与导出仍由`TrainExportAudit`独立比较；引擎侧现在不必等待TE训练信号。两部分共用`VERL_QUANT_LAYER_AUDIT`。旧SGLang格式guard挂在`VERL_MXFP8_REFIT_CHECK`下的描述不适用于新版；后者继续控制数值自检，不能只保留一个旧开关字符串判断覆盖。

公共检查将BF16、FP16、FP32归为高精度，其余dtype归另一类，只判断两类是否相同，不核对具体FP8格式、scale、位宽、布局或内容。它只收`weight`、`weight_packed`及两种融合expert叶子；scale不参与，未解析名称累计为`unmapped`而非通过。默认warn记录后继续写，raise在当前桶写入前抛错，0关闭。

[vLLM名称解析](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/verl/utils/vllm/vllm_sync_format_check.py)处理HF mapper、融合projection、MoE holder与LoRA wrapper；PP缺失层不判断。`is_checkpoint_fp8_serialized=False`的在线量化路径直接跳过，因为它本来就接收高精度输入。[SGLang解析与接线](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/verl/utils/sglang/sglang_sync_format_check.py)在MXFP8/MXFP4/NVFP4专用refit loader内生效，不能外推未走该loader的FP8 blockwise路径。vLLM这里检查main model，虽另有drafter加载调用，也不能把检查覆盖扩大为drafter。

### 完成一次检查与完成一次更新

检查器驻留于model：`begin_sync`清除未完成轮的计数，每桶检查，正常后处理/finalize后`end_sync`标记finished；只有结束过且至少见到一桶才停止后续检查。这依赖后续同步格式保持不变的设计前提，不是每次更新内容认证。warn模式即使发现不一致，完整走到end也会finished；finished意为首轮检查流程结束，不等于一致或安全。失败后重新计数，也不恢复旧权重。

新版[vLLM IPC接线](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/verl/workers/rollout/vllm_rollout/utils.py)把逐桶回调的普通`Exception`暂存，后续回调直接返回；接收器完成后再抛原异常，跳过正常后处理与`end_sync`。这避免回调异常直接截断ACK链。不是所有故障都走该路径：准备阶段、接收器自身和设备同步异常不在这个回调catch内，`BaseException`中断也不捕获。

[真实接收器](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py)仍逐桶接收metadata、重建tensor/view并同步设备；非末桶随后ACK，末桶在清理buffer后ACK。因此“后续桶不加载”不是“不接收、不物化、无传输成本”。此前成功桶已经可能改写参数，失败桶的prepare阶段也已发生；排空握手不提供参数回滚或可直接续训保证。

### 新版本的证据范围

作者正文把B200结果固定在`cf32f6ae`，明确其后的引擎检查迁移、probe skip告警及review修正尚未重跑硬件。当前七文件161项CPU通过是作者报告；[新增格式测试](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/tests/utils/test_quant_sync_format_check_on_cpu.py)使用形似引擎的torch模块，[IPC测试](https://github.com/verl-project/verl/blob/ba2852ec76e3a11cc90fc0883ffd70416c831661/tests/workers/rollout/test_vllm_weight_update_utils_on_cpu.py)用ScriptedReceiver/FakeFormatCheck计ACK，不能据此宣布真实ZMQ/CUDA链路验收。

本机仅对七份相关生产文件和两份上述测试做AST/内存compile，共9份通过；未执行这些原测试或运行引擎、通信、设备。源码还显示sentinel新增`response_length/max`与配置cap条件，避免V1等长短回答因clip_ratio=1误报；缺cap或max时仍回退旧条件，空字符串override也改为继续查native量化声明。上节旧20例/5控制的结果不覆盖这些变化，本轮不重复旧矩阵或构造假引擎。复现范围与hash见[当日日志](../../diary/2026-10-24.md)，后续条件只更新[原M5量化入口](../../experiments/handoff-m5.md#verl-量化层审计新版复验)。

## VeOmni 低精度计算与通信边界

**低精度计算、参数通信、梯度通信和rollout量化不能用一个开关或一组性能数字互相认证。** [verl #8342](https://github.com/verl-project/verl/issues/8342)提出FSDP2加torchao、模型中保留分片FP32主权重的端到端路线；作者amenomori已有本地原型，完整实现仍待推进。维护者wuxibin89建议先向VeOmni提出RFC；作者于2026-10-25T06:34:34Z创建[VeOmni #1360](https://github.com/ByteDance-Seed/VeOmni/issues/1360)，并于06:41:16Z[回链确认](https://github.com/verl-project/verl/issues/8342#issuecomment-6394489917)。当日核对#8342完整正文及两条评论、#1360完整正文及评论列表，后者open、评论为0；这项协作转移不等于后端已批准接口或完成实现，不另认领。

### 后端 RFC 的分工与迁移条件

#1360拟由VeOmni负责配方、转换、分布式集成、量化状态与checkpoint/export契约及后端测试；verl消费接口，负责RL编排、rollout/refit集成及端到端验收。作者提出的阶段仍是float8计算、MXFP8、参数AllGather、torchao QAT，原生QAT另与已有负责人协调，不是只完成一个通信选项就关闭整个路线。

**配置迁移尚未定案。** #8342正文拟在已有配方与kernel选择被表示后用QAT块替换`qat_implementation`；#1360则明确保留现有默认值与该字段语义，迁移另行议定。两份提案不能拼成已发布统一配置，也不应据旧示例抢先改名或弃用。后端RFC还提出与[FP8 MoE #1356](https://github.com/ByteDance-Seed/VeOmni/pull/1356)协调；这是作者列出的依赖关系，本次未复查该PR实现或测试。

其拟议验收逐配方覆盖真实kernel与通信、数值容差、梯度累积、FP32更新、checkpoint/resume与export；下游另核rollout格式、train/rollout数值偏差，以及固定基线下的吞吐和峰值显存。初始模型、硬件与依赖矩阵仍请求维护者确认，不把验收清单当成通过记录。后续只在维护者给出实质接口／协作反馈时重开技术判断，无需为RFC先排新算力矩阵。

### 各工作流的完成边界

| 工作流 | 当前实现或RFC要求 | 不能由此推出 |
| --- | --- | --- |
| 训练计算 | RFC拟按引擎选择BF16、float8 tensorwise/rowwise或MXFP8；转换放在FSDP包装前 | 存在fake-quant或低位权重格式，就实际执行了对应低精度GEMM |
| 参数AllGather | RFC要求在聚集前量化FP32分片，并处理payload与scale；MXFP8需专门集成 | 更换linear计算wrapper就自动缩小AllGather通信 |
| 梯度ReduceScatter | 已合并的[#8330](https://github.com/verl-project/verl/pull/8330)暴露默认关闭的节点内BF16传输，FP32累加与输出；VeOmni判定可用分组 | 参数AllGather或HSDP副本AllReduce也变成低精度，或所有拓扑都走该路径 |
| QAT与rollout格式 | RFC分别包含原生VeOmni QAT、torchao NVFP4/MXFP4 fake-quant，并要求checkpoint/resume及export/refit闭环 | QAT的部署格式、scale和模块覆盖已与真实rollout一致，或整体训练更快 |

[#8186父路线图](https://github.com/verl-project/verl/issues/8186)的8条完整评论显示，共同量化配置已有贡献者提案及撤回记录，低精度通信也已有amenomori推进；不把规划表中的未勾选项当作没人做。#8342要求保留现有ReduceScatter选项和默认FSDP策略，其示例`low_precision`块是拟议配置，不是可以直接复制到当前主线的已发布API。

### 已合并的梯度通信接线

#8330于2026-10-22T07:03:51Z合并为`fb55b9288560f0c7bf380c1c1c3b45ea170538b0`；固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`在它之后8个提交。[当前配置](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/config/engine/veomni.yaml)的`low_precision_reduce_scatter_comm`默认false；[engine](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/veomni/transformer_impl.py)将其直接传给`build_parallelize_model`。同一构造路径在mixed_precision启用时先按float32加载模型，再调用VeOmni并行化和优化器builder；这是源码调用顺序，不是本机观察到的分片dtype或更新结果。当前extra依赖固定`veomni==0.1.12`，不能沿用旧checkout的0.1.11描述现行环境。

底层[VeOmni #1167](https://github.com/ByteDance-Seed/VeOmni/pull/1167)已合并为`019b1c0276f0fee6175413367910c0a649593131`。其正文说明节点内低精度all-to-all后本地FP32求和，跨节点RS回退原生，副本AllReduce和参数AllGather保持原路径；额外临时缓冲和实际拓扑会影响收益。这里核对了正文与verl接线，未逐行审查底层通信实现。

性能证据也要按层级分开：#1167作者在两节点、每节点8张H100上报告64–1024MiB区间的本地RS延迟下降11.9–14.8%；#8330作者在2张H200 NVL、428M参数Qwen3、长度512的actor更新中却没有观察到加速，低精度两次中位数130.853/119.163ms，原生两次117.023/118.891ms，且保留显存更高。后者只报告波动，不给稳定减速比例；不同模型、硬件和计时范围不支持据此定位原因。#8330的GPU结果还属于删除兼容/forward-only guard前的旧实现，最终树只报告CPU回归与hooks，未重跑CUDA。以上均为作者报告，不是本项目实测。

### torchao 配置限制与接续

RFC引用的[torchao v0.18.0原配置](https://github.com/pytorch/ao/blob/v0.18.0/torchao/float8/config.py)中，`Float8LinearConfig.__post_init__`在weight的scaling_granularity不是TENSORWISE时，断言不得启用`enable_fsdp_float8_all_gather`；标准ROWWISE配方将weight设为AXISWISE。因此“rowwise计算+float8 AllGather”在这个具体接口/版本不是任意可组合选项。这不证明其他版本或独立MXFP8接口同样受限，也不能以关闭guard替代后端集成。

本轮为RFC、API状态和源码学习：没有安装/import torchao或VeOmni、运行配置构造/原测试、通信、量化kernel或模型。后续只有明确的协作范围或实质接口变动才重新评估贡献；[原M5参考](../../experiments/handoff-m5.md#verl-量化层审计新版复验)仅补证据层级，不新排训练/通信矩阵，不把CPU配置检查当端到端验收。[检索与来源](../../diary/2026-10-24.md)保留日志。
