# verl 融合输出头与 logit softcap

2026-10-25接续：[Megatron 融合权重的就绪与调度](#megatron-融合权重的就绪与调度)说明直接读取参数会绕过DDP前向钩子。#8029原23项CPU生命周期测试通过，不是实通信或融合kernel验收；FSDP注册方法、Megatron bucket就绪和softcap数学仍是不同问题。

2026-10-24 接续：[独立分片 head 的执行与梯度归属](#独立分片-head-的执行与梯度归属)讨论FSDP生命周期；它与下方softcap数学语义是两个独立问题，旧36组CPU结果不验证分布式修复。

通用 fused output head 直接消费hidden states和输出权重，绕过模型原始forward中的输出处理，因此必须保留原模型对logits的变换。省去 `c*tanh(z/c)` 不只是更换计算精度，会改变 log-prob、熵和梯度。verl [#7991](https://github.com/verl-project/verl/pull/7991)选择在不支持时拒绝通用 fused 路径，而不是补上 softcap kernel。

## Megatron 融合权重的就绪与调度

融合输出头读取 `weight`，不一定调用该参数所属模块的 `forward`。计算公式即使正确，也可能绕过分片参数的准备时机。[#8028](https://github.com/verl-project/verl/issues/8028)报告的链条是：decoder预取输出权重的AllGather，融合LM-head跳过output_layer前钩子，遗留handle；下一批再次派发时触发断言。原故障来自作者下游MCore0.16.2/H20配置，不能写成本机或当前主线GPU复现。

报告者已有[#8029](https://github.com/verl-project/verl/pull/8029)，2026-10-25读取open、非draft、未合并；固定head `99b6f0223892a13f699bcbadcf95a25b6e942891`、实际祖先 `4b26c788d41ce932ce7706c5ee8e249ab52f4c51`（API目标base另为263653e7），7文件769增1删。head锁文件声明MCore0.19.2、Bridge0.6.2、TE2.16.1，不是本机安装证明。[Modern Megatron RFC #8324](https://github.com/verl-project/verl/issues/8324)也把参数就绪列作原生契约，并要求复用这项已有工作；本轮不展开其大型迁移或fork原型。

### 完成参数通信与派发下一桶

官方MCore0.19.2固定提交 `4b4acac9a1d28ea6829c8d4f566d75698a21249d` 的[DDP前钩子](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/distributed/distributed_data_parallel.py)调用所属bucket的 `finish_param_sync`，并由align或optimizer overlap设置是否禁止预取下一桶。[bucket实现](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/distributed/param_and_grad_buffer.py)在非force派发前要求handle为空；完成方法必要时启动本桶，等待后清handle，还可能派发下一桶。`finish_grad_sync`重置dispatch标志，不代替参数handle的完成边界。因此不能把“等待一个handle”与“可以自己启动下一次collective”视为同一职责。

候选[兼容回调](https://github.com/verl-project/verl/blob/99b6f0223892a13f699bcbadcf95a25b6e942891/verl/utils/megatron/param_sync.py)按实际DDP拥有者的策略处理，而不是无条件强制同步所有桶：

| 状态 | 兼容回调行为 | 所保留的职责边界 |
|---|---|---|
| 已派发且handle为空 | 直接返回 | 同一参数桶本轮不用重复完成 |
| DDP拥有前钩子调度 | 优先调用其finish helper；旧接口沿align／optimizer overlap决定下一桶预取 | 复用拥有者策略，不另造调度规则 |
| 外部调度且已有handle | 完成现有通信，禁止下一桶派发 | 不替外部schedule启动或预取 |
| 外部调度且无handle | 不派发 | 不能凭回调返回就认定参数已准备好 |
| CUDA graph capture或旧owner已释放 | 不由此发起新通信 | 跳过不等于已经验证图内就绪或参数新鲜度 |

注册发生在普通／layer-wise DDP包装完成之后，参数对应真实bucket；同桶共用回调，以weakref指向DDP和bucket，避免参数长期保留旧通信缓冲。重新包装会先移除旧兼容标记，关闭overlap时不再注册；不在DDP参数buffer中的冻结参数不标记。若原生回调能力存在则优先使用，兼容层不能据此宣称当前旧版也有原生API：上述官方0.19.2的 `core/utils.py` 中没有该两项名称。

候选的[两个融合消费入口](https://github.com/verl-project/verl/blob/99b6f0223892a13f699bcbadcf95a25b6e942891/verl/models/mcore/model_forward_fused.py)在调用 `linear_cross_entropy` 前对实际权重调用就绪helper。原生processor优先使用传入的tied `output_weight`，否则选output_layer.weight；legacy分支也处理共享embedding。hidden states的sequence-parallel gather不能替代权重bucket的完成，它们服务不同tensor。这里不是FSDP的 `register_fsdp_forward_method`，也不是修改softcap或LM-head梯度公式。

### 原 CPU 测试证明的范围

固定head的[原CPU文件](https://github.com/verl-project/verl/blob/99b6f0223892a13f699bcbadcf95a25b6e942891/tests/utils/megatron/test_param_sync_on_cpu.py)未经修改，pytest原生收集23项全部通过。它加载完整候选helper，使用作者自己的假DDP／bucket／Work和graph状态替身；Work在wait时复制小CPU权重，consumer通过AST读取原函数体，但融合数学以普通torch算式代替。覆盖调度策略、native能力优先、weakref回收、重新包装，以及native/legacy×tied/untied三步输出和梯度；无就绪负控制也仅在这套生命周期模型里显示旧权重和下一次派发失败。不是主线真实DDP故障复现，更不是23项真实AllGather测试。

本机Python3.12.13／torch2.14.1 CPU／pytest9.1.1，无依赖安装；六份候选Python及三份官方源码共九份AST/compile通过。模型测试里的事件顺序断言仅读，未执行该完整模型测试或分布式脚本。作者另报基线4b26c788上的tiny Qwen3 BF16、2–4 H20、四次optimizer更新及若干TP/PP/packed组合数值完全一致；VPP未覆盖，legacy和可选native API只有CPU覆盖。这些是作者报告，不是本项目硬件结果或全部平台验收。

[M5原输出头参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)只补已有获准匹配Megatron工作的版本、实际消费参数、bucket拥有者和调度条件，不自动采用补丁或新增训练队列；上游评论、PR及其fork head推送仍逐项待用户明确同意。复现命令与hash见[当日日志](../../diary/2026-10-25.md)。

## 固定版本

2026-10-23读取API时PR为open，head `b145ec9b405ca93092c45b5f9e7995fb4604dbd2`，base及共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`（ahead2/behind0）。实际差异为4文件、61行新增，其中生产逻辑仅 `monkey_patch.py` 的7行检查；其余是44行CPU测试、性能文档提示和2026年贡献者license头支持。

[`dense_common.py`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/models/transformers/dense_common.py)和[`torch_functional.py`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/utils/experimental/torch_functional.py)在base/head逐字节相同。因此，下面对原torch输出头的观察不是“修复后kernel验收”。

## 拒绝范围

[`patch_forward_with_backends`](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/verl/models/transformers/monkey_patch.py#L240)先判断是否启用、backend是否在torch/triton/liger列表，再选择模型专用或通用adapter。新检查只在通用分支读取 `model.config.get_text_config()`，发现 `final_logit_softcapping` 为真值便抛ValueError，提示 `use_fused_kernels=False`。

| 入口条件 | 这段函数的行为 | 不能推出的结论 |
|---|---|---|
| fused关闭或backend不在列表 | 开头直接返回 | 不认证配置合法，也不撤销先前patch |
| 模型专用Qwen/GLM分支 | 沿原专用adapter选择 | 新guard没有审计这些adapter的softcap语义 |
| 通用分支，softcap为真值 | 在import通用adapter、设置backend属性和替换forward之前抛错 | 不是融合softcap实现或完整模型可用性证明 |
| 通用分支，softcap缺失或为假值 | 继续原backend选择 | 不认证模型不存在其他特殊输出变换 |

torch与liger都选 `forward_with_torch_backend`，后者再按实例backend属性分派；triton选另一入口。三者都会经过同一个通用guard，改backend名称不能绕过限制。检查不是一般的softcap数值范围校验，条件使用Python真值。

该函数成功后修改的是 `model.__class__.forward`，backend标记则在实例上。关闭fused只跳过这次操作，不自动恢复已修改的类。外层 `apply_monkey_patch` 还可能先安装TiledMLP、PrefixGrouper或attention patch，最后才调用这个检查；故“抛错前未替换此处forward”不能外推为整个初始化没有副作用或具备回滚。

## 为什么不只是舍入差异

通用torch路径直接用hidden states和 `lm_head.weight` 计算 `z=hWᵀ`，再除temperature后计算选中token的log-prob和熵，没有调用原模型的输出处理，也没有softcap参数。[HF v4.57.3 Gemma2源码](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/gemma2/modeling_gemma2.py#L565)则在LM-head投影后先除c、tanh、再乘c，之后才交给loss或作为logits返回。

本对照先执行这一模型输出变换，再施加temperature，比较 `softmax(c*tanh(z/c)/T)` 与 `softmax(z/T)`。softcap的导数还带有 `1-tanh²(z/c)`，不能只在最终log-prob上修一个常数。

本机合成CPU例子，`z=[0,60,120]`、选token2、T=1、c=30：

| 计算 | 选中token log-prob | 熵 |
|---|---|---|
| 原torch输出头，FP32 | 0 | 0 |
| raw logits参考，FP64 | 0（舍入） | 约5.25e-25 |
| softcap参考，FP64 | −0.297720 | 0.570416 |

这里没有模型、采样或训练收益测量；只是相同hidden/weight/label在两种分布定义下的直接对照，不将合成logits说成Gemma实测。

## CPU 对照与数值限制

[复查脚本](../../experiments/scripts/verify_verl_fused_softcap.py)从本地固定Git对象执行完整原 `torch_functional.py`，不删函数或替换反向实现。Liger/FlashAttention均须未安装，原模块正常进入torch fallback；不创建verl包壳、假模型或配置，也不导入Transformers/设备后端。

对照矩阵为2D/3D输入、chunk=1/2、T=0.7/1/2及log-prob单独、熵单独、二者组合三种反向目标，共36组。每组都用真实autograd核hidden/weight的梯度，并与独立FP64 raw/softcap参考比较。36组中，softcap语义差异均大于各自raw数值误差的100倍；每组分别取前向和梯度的最大绝对差，跨组最小值为29.8434和1.10977。这是本组输入的观察，不是全输入或每个token的误差下界。

最初对FP32参考设 `atol=rtol=2e-5` 时，熵相关梯度有一例超限（绝对差约1.57e-4）。改用FP64参考量化后，没有把容差调宽称为全过：36组中只有28组的四类最大绝对误差都在2e-5内。各类跨组最大值如下：

| 原FP32实现相对raw FP64参考 | 最大绝对差 |
|---|---|
| 选中token log-prob | 2.18e-6 |
| 熵 | 2.73e-6 |
| hidden梯度 | 1.61e-4 |
| weight梯度 | 1.34e-6 |

原熵公式含 `logsumexp(z) - sum(p*z)` 的大项相减；这里保留实际误差，不认证原手写反向与autograd数值精确等价，也不据此另立训练缺陷或修改误差容许标准。它与缺少softcap导致的上述分布变化是两个问题。

Python3.12.13 / torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`），只运行CPU FP32原输出头和FP64参考：

```bash
python experiments/scripts/verify_verl_fused_softcap.py /path/to/verl
```

脚本还对base/head四个完整Python文件及head原测试模块做9次AST/内存compile，并核两个输出头文件未变。完整原实现SHA256为 `812d6db39676454f6bdb4881eedc9eceeef28c8e08a1b505bb54346c7b6a7485`；其他指纹随命令打印。

## 原测试与接续

作者[原测试](https://github.com/verl-project/verl/blob/b145ec9b405ca93092c45b5f9e7995fb4604dbd2/tests/models/test_fused_softcap_on_cpu.py)以真实HF Gemma2/Gemma3 config及无权重的PreTrainedModel覆盖3 backend×2嵌套布局×2softcap值×2启用状态，共24例。它检查拒绝时这段forward/backend选择未修改，不运行模型或三种kernel；作者报告24过，不是本机结果。本机缺Transformers，未执行这些测试，不用假配置替代。

原生guard回归首先需要匹配HF及verl导入依赖，不需要先申请GPU。若日后在已有获准模型任务中核对真实融合路径，还需匹配模型config/权重、原始forward及backend环境，比较相同hidden/label/temperature下的输出和梯度。真机参考补入[M5原一致性入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动部署候选、下载模型或增加训练队列；与#8250没有已证因果关系，AReaL/#8313/#523分工不变。

## 独立分片 head 的执行与梯度归属

**拿到完整权重、进入FSDP执行入口、按正确布局归约梯度，并非同一件事。** [#8091](https://github.com/verl-project/verl/issues/8091)报告untied LM-head被单独FSDP2包装后，父模型的融合路径直接读`lm_head.weight`，绕开其参数准备及反向记账，表现可随backend/offload/延后同步变成不同错误。固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`的dense torch分支仍直接传该weight。2026-10-24读取已有两种替代候选均OPEN/unmerged：cyanseek的非draft [#8096](https://github.com/verl-project/verl/pull/8096)，head`6217269365034565d0c032a329dd8eda2cebc4a5`、base/祖先`ddf700725c34bb8966685c8ecf5ca36a2cc54a35`（ahead3）；kolehma8的draft [#8097](https://github.com/verl-project/verl/pull/8097)，head`2d4ad53c02b5a10d238cbe57a4431d18144c4f19`、base/祖先`8718ca30a3f002f93b7c4fd99b9b2506718681bc`（ahead2）。两者不应叠加成重复修复，也不另认领。

### 注册执行入口与显式聚合的区别

[#8096原helper](https://github.com/verl-project/verl/blob/6217269365034565d0c032a329dd8eda2cebc4a5/verl/models/transformers/fused_lm_head.py)在FSDPModule head上懒绑定一次`_verl_fused_forward`，通过PyTorch公开`register_fsdp_forward_method`注册，再在该方法内部读取weight并调用原torch/liger或triton输出头。普通head直接执行helper、不加此属性；原`head.forward`仍返回logits。六个生产文件把dense/GLM4V/Qwen2VL/Qwen3VL/Qwen3.5接至共享入口。Qwen3-VL/Qwen3.5的部分gather/cast旗标保留既有非FSDP DTensor处理，不是通用并行梯度认证。原`torch_functional.py`在该候选base/head字节相同，修的是执行边界，不是换一套融合数学。

本机torch2.14.1（git`5c4886908584029761b579af026dcfb627c84070`）的[公开注册实现](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L927)包装指定方法，依次调用FSDP state的`_pre_forward`、原方法、`_post_forward`；普通Module则no-op。它没有通过`head(...)`调用普通`nn.Module.__call__`，不能把“恢复FSDP生命周期”泛化成所有第三方forward hooks都恢复。这是安装版源码核对，不把2.14实现当作者2.11/2.13实测环境。

[#8097原helper](https://github.com/verl-project/verl/blob/2d4ad53c02b5a10d238cbe57a4431d18144c4f19/verl/utils/experimental/torch_functional.py)改走显式物化：在custom autograd `apply`之前搬动DTensor shards至hidden设备/精度，再`full_tensor(grad_placements=[Partial("avg")] * mesh.ndim)`，将聚合/搬运接入外层autograd图。其[同步helper](https://github.com/verl-project/verl/blob/2d4ad53c02b5a10d238cbe57a4431d18144c4f19/verl/utils/fsdp_utils.py)在其他模块延后同步时仍让独立FSDP head保持同步，engine上下文改经此入口。作者明确代价是完整head留给反向、每个microbatch都做head梯度通信，不声称吞吐或显存更优。

安装版[DTensor.full_tensor](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/tensor/_api.py#L830)先redistribute至Replicate，再转普通Tensor；`grad_placements`描述返回Tensor未来梯度布局，省略时假设Replicate。由各DP rank不同数据产生的梯度，不能只因前向权重相同就当成副本；这里的Partial(avg)还须与实际mesh/损失归一化匹配，不是任意TP/HSDP的万能修复。本机未执行DTensor通信或验证其梯度。

### 测试参考分别隔离什么

#8096新增CPU测试只有普通Linear的cast开/关两例：核log-prob、entropy、hidden/weight梯度、原forward未改及未添加FSDP入口，不会走FSDP分支。其分布式原测试以六种小随机真实模型、三backend、tied/offload与两种reference构成144个case；rank数不能再乘进独立case数。同kernel/root-owned head参考保持独立两步轨迹，主要隔离生命周期；unfused HF参考每次更新先对齐参数，容差还包含BF16投影/temperature舍入差异，不能把两种reference合称逐bit数学等价。Issue后续第三方称三种候选都过原五复现，也不是吞吐/峰值或完整RL验收。

以上GPU结果均为作者/社区报告；#8097另明确无完整FSDPEngine e2e、Ascend或组合TP验收。本机只核两候选完整生产diff、原CPU/分布式测试源码及PyTorch叶实现，对两候选各六个生产Python和#8096原CPU测试共13次compile，无原pytest、FSDP进程组、模型或设备执行；旧softcap数值实验不复跑、不挪作证据。[命令与hash](../../diary/2026-10-24.md)留日志，缺口只补[原M5输出头参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用、拼接方案或新增必跑矩阵。
