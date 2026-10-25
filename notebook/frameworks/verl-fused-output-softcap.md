# verl 融合输出头与 logit softcap

2026-10-25补充：[选定 token 概率与共享归一化项](#选定-token-概率与共享归一化项)区分#7712已收窄的引擎输出提案、旧fork组合器和新的恒等式；FP64 CPU数学／梯度对照不是fused、布局或训练验收。

2026-10-25接续：[Megatron 融合权重的就绪与调度](#megatron-融合权重的就绪与调度)说明直接读取参数会绕过DDP前向钩子。#8029原23项CPU生命周期测试通过，不是实通信或融合kernel验收；FSDP注册方法、Megatron bucket就绪和softcap数学仍是不同问题。

2026-10-24 接续：[独立分片 head 的执行与梯度归属](#独立分片-head-的执行与梯度归属)讨论FSDP生命周期；它与下方softcap数学语义是两个独立问题，旧36组CPU结果不验证分布式修复。

通用 fused output head 直接消费hidden states和输出权重，绕过模型原始forward中的输出处理，因此必须保留原模型对logits的变换。省去 `c*tanh(z/c)` 不只是更换计算精度，会改变 log-prob、熵和梯度。verl [#7991](https://github.com/verl-project/verl/pull/7991)选择在不支持时拒绝通用 fused 路径，而不是补上 softcap kernel。

## 选定 token 概率与共享归一化项

**返回少数列不代表计算天然便宜，但已有的标签概率可以携带全词表归一化项。** [RFC #7712](https://github.com/verl-project/verl/issues/7712)作者于2026-10-03[撤回目标组合器、插件加载及归约schema](https://github.com/verl-project/verl/issues/7712#issuecomment-5966319867)，称GenRank已通过`TrainingWorker.set_loss_fn`组合目标；剩余请求是默认关闭的`model_output["selected_token_logprobs"]`，包括fused路径，不再以新增kernel为前提。2026-10-25读完整正文及全部5评论；WenzheWang之前已请求认领kernel切片，但没有在这段讨论中获得确认，不另建组合器、kernel或竞争方案。

[旧fork draft #1](https://github.com/eteluna/verl/pull/1)仍open／draft／未合并，head `f894f21e7cec29f04ddef30cc9593ddfab7a1503`、base `e42d6af6e85dd37c907af5ea99326355c376bd97`，更新时间为9月9日；正文仍是组合器与eager共享log-softmax、启用selected时拒绝fused。它不是10月3日恒等式方案的已发布实现。编号与selected_token_logprobs的上游all-state PR搜索均为空，不据此认定无人推进；旧66项CPU测试和作者GPU报告均非本机结果。

### 恒等式与梯度来源

设`z`为**同一位置、同一温度、同一logit变换与精度语义**下的最终logits，`A=logsumexp(z)`，标签为`y`，选定ID为`i`：

```text
log p(i) = z_i - A
         = log p(y) + z_i - z_y
```

线性无偏置输出头可用`h @ W[ids].T / temperature`取得少量选定logits，并取得每行标签logit。额外投影不必另建完整`[T,V]`输出，但原标签路径仍要计算全词表归一化与其梯度；输出权重／权重梯度也仍有词表维度。这不是将整个输出头计算量降至只依赖选定列数。

若两种输出收到上游权重`a_t`、`b_ti`，复用表达式令标签概率支路收到`a_t + sum_i b_ti`，另外两项投影补回选定与标签logit的梯度。**不能detach标签概率锚点**：那会丢失全词表归一化项的导数，虽然forward值不变。精确算术下，最终梯度与完整log-softmax相同；这不自动保证任意后端的舍入、自定义backward或原地修改相容。

选定列仍是全词表概率，通常和不为1。若消费者确实要条件于选定集合的分布，可再减去这些列的logsumexp，此时共同归一化项消去；这是不同的目标。均匀7词表选2列时，每列全词表概率为1/7，集合内条件概率为1/2，不能把二者都叫“校准概率”而不标分母。

### 当前引擎边界与性能证据

固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`的[通用dense适配器](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/models/transformers/dense_common.py)只返回log_probs与entropy，优先采用engine提供的shift_labels；[FSDP引擎](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/fsdp/transformer_impl.py)在remove-padding／SP路径先全局roll再切标签，随后gather／去padding／jagged重包。生产树检索没有selected_token_logprobs字段；新输出必须沿同一token身份与重包路径，不能把局部hidden与全局标签直接相减。

当前fused入口还拒绝逐样本温度；[torch输出头](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/experimental/torch_functional.py)先投影／除温度，再转FP32求log-softmax；[Triton包装器](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/kernel/linear_cross_entropy.py)则调用专用kernel。本轮未逐行审查Triton内部算术，不能由相同dtype名称宣布两条支路舍入相同。带bias／softcap等变换、ignored标签、不同head及FSDP参数可读生命周期也各须匹配实际契约；纯恒等式没有解决这些集成条件。

作者10月3日报告的是H200、BF16、H=1024、V=151936、M=5的单算子forward+backward：6.8k token时，Liger label-only到identity方案的算子额外峰值为0.59→0.90 GB、耗时20.1→21.1 ms；不是整步训练只加1 ms。相对旧eager参考的节省还包含更换backend的差异，torch+identity在同表中反而慢于eager参考，不能概括为“恒等式对所有后端均提速”。此前16×H200的eager组合器训练报告属于旧方案，不迁移为新接口的端到端证据。

### FP64 CPU 数学对照

[复现脚本](../../experiments/verl-selected-logprob-identity/verify.py)与[原始结果](../../experiments/verl-selected-logprob-identity/results.json)使用真实torch2.14.1 CPU autograd，固定T=4、H=3、V=7、M=2、temperature=0.7及seed7712。标签概率来自dense参考，不是原fused kernel，脚本本身物化`[T,V]`作为oracle，不用于显存／速度测量。

恒等式与dense全log-softmax的输出、hidden梯度与weight梯度最大误差分别为8.88e-16、4.44e-16、3.33e-16；detach锚点负控制保持相同值，梯度却分别偏离2.6754与1.2801。集合内重归一化的值与梯度也在1e-12内一致；该控制只有一个固定FP64形状，没有BF16、ignored标签、padding／SP或kernel覆盖。两次运行结果一致。

```bash
/private/tmp/verl312/bin/python -B experiments/verl-selected-logprob-identity/verify.py
```

四份完整相关主线模块另过AST／内存compile；没有执行作者测试、engine、模型、原地CE、FSDP或GPU。后续只有接口实质变化或明确协作范围才再评估；[原M5条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)不新增训练队列，不自动接管作者工作。

## 原生 MTP 的梯度与归一化控制

**升级依赖、提供原生接口、移除verl补丁是三个不同决定。** [Modern Megatron RFC #8324](https://github.com/verl-project/verl/issues/8324)由ISEEKYAN提出，2026-10-25读取OPEN、0评论，已有个人fork原型，不另认领迁移。其Core配套 [ISEEKYAN/Megatron-LM #263](https://github.com/ISEEKYAN/Megatron-LM/pull/263)仍是OPEN/draft、未合并的个人fork草稿；固定head `b1a29d72d161ed3cff6128889b11379da1127e3b`、base `728ef1414008ce4b51ce1c0db86993c3eb871601`，十文件339增27删。RFC提出Core0.20／Bridge0.8开发快照，不是仅按版本号就能获得这些新增契约。

### 计算开关与梯度开关不能互换

[Core草稿MTP源码](https://github.com/ISEEKYAN/Megatron-LM/blob/b1a29d72d161ed3cff6128889b11379da1127e3b/megatron/core/transformer/multi_token_prediction.py)将输出权重分离从旧的整条MTP输入分离中拆出来；[GPT入口](https://github.com/ISEEKYAN/Megatron-LM/blob/b1a29d72d161ed3cff6128889b11379da1127e3b/megatron/core/models/gpt/gpt_model.py)另外传递每次forward的计算控制。

| 控制项 | 对应职责 | 不应混作 |
| --- | --- | --- |
| compute_mtp_loss | 传给postprocess，控制其中MTP执行／辅助损失；fine-grained schedule同时选择是否建MTP层计划 | 把输出权重detach就等于关闭MTP |
| mtp_detach_output_weight | MTP投影使用detach后的显式output_weight，或output_layer.weight | 同时冻结主干、输入embedding或主任务投影 |
| mtp_detach_heads | 原有MTP主干hidden输入与embedding输入分离；也使MTP输出权重detach | 仅让策略head不收辅助梯度 |
| mtp_loss_normalization_factor | 乘到辅助损失缩放因子上，再沿现有token归一化分支处理 | 自动修正所有DP／CP全局分母 |

MTPLossAutoScaler的forward保留主返回hidden，backward另给保存的辅助损失梯度。因此把主返回值乘0，仍可有MTP辅助梯度；这不是“整个目标都为零却凭空产生梯度”。主损失反向缩放还由main_loss_backward_scale提供，不能把本例设1的结果推广到真实AMP或pipeline调度。

### 原八个梯度用例的范围

本机Python3.12.13／torch2.14.1／pytest9.1.1，执行[原测试体](https://github.com/ISEEKYAN/Megatron-LM/blob/b1a29d72d161ed3cff6128889b11379da1127e3b/tests/unit_tests/transformer/test_mtp_native_rl.py)的八个参数例，两次全过。完整原process_mtp_loss、MTPLossAutoScaler与roll_tensor经AST隔离；原tensor_observation模块保留默认关闭状态，没有用自写损失／roll或观察器替身。未导入完整Core、TE／Apex／Transformers／Ray或初始化process group。

八例是显式传入共享权重／不传、detach开关、per-token开关的2×2×2组合；F.linear／交叉熵计算真实CPU梯度。原断言确认MTP hidden部分梯度非零、主hidden部分为零，输出权重是否有梯度随detach变化，并匹配独立移位参考的归一化系数。**它的“tied”只是传入同一输出层参数，没有把该参数同时用于真实embedding查表；不能据此认证完整模型的权重绑定或主干参数梯度。**

原mask为[0,1,1]，移位后[1,1,0]，有效数都为2；没有覆盖滚动前后token数不等、空段或多rank汇总。两种归一化分支在这个夹具中分别使用0.1×2／2与0.1×2×(2／2)的系数，但完整分母仍依赖上层调度。源码自身也注明per-token分支使用本地计数，不能直接认证不均匀packed／DP／CP输入的精确全局等价。

本轮对base／head函数仅另做签名绑定：base拒绝loss_normalization_factor，head接受；没有运行base梯度对照，不把接口缺失写成旧版八次数值失败。同文件第九个runtime-CP测试、Hybrid callback及router replay测试均未执行；CP覆盖、GPT开关和schedule传递只核源码diff。六份完整复制快照compile通过，复现与hash见[当日日志](../../diary/2026-10-25.md)。

### 迁移条件仍由整体契约决定

RFC作者报告的Core15项、verl80项及Bridge8项CPU结果是外部证据；其环境不同，不能与本机八例相加。草稿明确仍待人工逐行审阅和分布式GPU验收。本轮只说明辅助梯度的局部契约可检验，不支持据此删除旧MTP／router补丁、自动升级Core／Bridge／TE，或认定Hybrid／多模态／Ascend支持成立。若M5已有获准匹配迁移工作并独立选用，沿[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)记录版本、真实mask／token分母、调度和下一次更新；没有新增算力队列或对外行动。

## Triton 输出的批维与序列维

元素个数相同，不代表满足下游张量契约。[#8089](https://github.com/verl-project/verl/issues/8089)报告Triton fused路径返回平铺logprob／entropy，关闭remove-padding后，引擎在`output.log_probs.shape[1]`处失败。已有报告者的[#8090](https://github.com/verl-project/verl/pull/8090)，2026-10-25核open、非draft、未合并，唯一普通评论为CLA机器人、0行内评论；编号及Triton output shape查重已覆盖，不另实现。

固定head `90c8667a149c457a335b42bd362311f989d9315c`、实际共同祖先 `fbb4b3a8bf636f290c9c59fc346f756849e9c241`，两提交六文件43新增。直接父提交`6ac75615`已含生产修复，最后提交仅加28行测试；API目标base `8718ca30a3f002f93b7c4fd99b9b2506718681bc`也不等于本次比较祖先。只看父提交diff会漏掉五个适配器的改动。

### 形状恢复不负责恢复标签身份

[原kernel包装器](https://github.com/verl-project/verl/blob/90c8667a149c457a335b42bd362311f989d9315c/verl/utils/kernel/linear_cross_entropy.py)先把hidden从`[B,S,H]`变成`[B×S,H]`、labels变成一维，再调用融合实现；forward直接返回kernel结果，原形状保存给反向，不等于forward自动恢复两维。[FSDP消费者](https://github.com/verl-project/verl/blob/90c8667a149c457a335b42bd362311f989d9315c/verl/workers/engine/fsdp/transformer_impl.py)在不使用rmpad／SP且fused、DatasetPadMode.NO_PADDING的分支，以第二维构造每行长度mask，再把有效token重新包成jagged。这里需要的是`[B,S]`与行身份，不只是B×S个元素。

源码AST比较确认dense_common、glm4v、qwen2_vl、qwen3_vl、qwen3_5五个Triton函数都**仅增加同三条语句**：从hidden_states.shape取B、S，对log_probs和entropy各reshape(B,S)。移除这三条后完整函数AST与祖先相同；五个Torch函数未改，固定main `3c6d9b43`的五个Triton函数也仍与祖先相同。reshape保留平铺顺序，但不会修正错误的shift_labels、token排列或有效mask，也不补FSDP输出头参数准备。此项与SP标签、softcap数学和untied参数生命周期分别验收，不互相替代。

### 测试替身必须保留真实后端的外形

作者[原测试模块](https://github.com/verl-project/verl/blob/90c8667a149c457a335b42bd362311f989d9315c/tests/models/test_fused_kernels_ulysses_sp_on_cpu.py)已有标签路由测试，但旧fake_linear_ce直接按input_ids形状造二维零结果，本身不会暴露真实平铺输出的缺口。新测试对10个adapter／backend组合与两种形状声明20个参数例，显式令Triton假kernel返回flatten结果、Torch假kernel返回二维结果，再检查logprob／entropy恢复为预设顺序。

这20例的模型与kernel均为作者替身；`packed`只是`(1,6)`测试ID，没有真实cu_seqlens、Ulysses通信或多模态body。它们针对外形契约，不计算真实log-softmax、entropy或反向。**本机未执行这20例或构造模型替身**，只做原测试／生产接线的源码核对，不能写成20例通过或真实kernel等价。

Python3.12.13仅标准库AST／内存compile：两版五个适配器和测试模块，加head kernel包装器／FSDP消费者，共14个完整快照语法通过，没有导入torch／Transformers／Triton、发现设备或运行模型。作者“首训练步不再崩溃”的报告不是本项目GPU实测或长程质量证明。匹配资源下的真实kernel、非等长有效token、标签与反向条件补[M5原输出头参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用候选、新排训练或改其专项；任何上游动作及既有PR-head推送仍逐项待用户明确同意。命令与校验值见[当日日志](../../diary/2026-10-25.md)。

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
