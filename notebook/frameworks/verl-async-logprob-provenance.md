# verl 异步训练策略版本与 logprob 对照条件

固定 token 的 log-prob 对照能缩小故障范围，但必须先对齐策略版本、上下文、有效掩码和概率处理。**相同版本标签不证明权重相同，采样 token 概率相同不证明完整分布相同，后者也不证明参数内容相同。** 这些检查应组合使用，不能用一个曲线阈值替代来源追踪。

适用源码为 verl `1876b06d0a3e4e71e06230be10af14492ca8a75b` 的 V1 separate-async。关联 [#8250](https://github.com/verl-project/verl/issues/8250) 报告明确使用该入口、私有修改及 temperature=1；安装的vLLM版本仍未知。这里不将旧experimental fully_async_policy路径或依赖pin当作该运行环境。阅读日期2026-10-21。

## 四种策略与概率字段

| 对象 | 字段与来源 | 比较时须保留的身份 |
|---|---|---|
| 生成策略 π_rollout | server输出采样token的log_probs，经agent loop变成rollout_log_probs | replica、request/turn、分段版本、实际sampling参数、logprobs_mode |
| proximal策略 π_old | 解耦模式由actor重算old_log_probs；separate-async可交换CPU快照以固定它 | CPU快照/导出checkpoint身份、local_trigger_step、重算时间 |
| 正在更新的 π_θ | ppo_loss中本次forward的log_prob，参与对old的ratio | optimizer/mini-batch位置、实际参数身份、精度与kernel |
| reference策略 π_ref | 启用相应项时的ref_log_prob；不是π_old的别名 | reference checkpoint与是否启用KL/奖励惩罚 |

[base `_compute_old_log_prob`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L1649)有两个分支：bypass将rollout_log_probs直接写成old_log_probs；解耦模式调用actor compute_log_prob，并按response_mask从nested输出还原。bypass中的“old”因此不是一次独立重算，也不能拿old与rollout相等证明同步正确。

[separate-async覆盖方法](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_separate_async.py#L187)在local_trigger_step=0时保存CPU slot 0；后续trigger保存当前actor、恢复slot 0计算old，再恢复当前actor并清理临时slot。它是在同一actor上交换快照，不是另起独立冻结模型。[step循环](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L547)允许一个parameter_sync_step周期内多次本地更新，因此发布版本单位也不能直接当作optimizer step数。

## 版本标签说明什么

[vLLM server](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L689)在生成返回处附加self.global_steps；[客户端](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/llm_server.py#L281)拼接中断后续写的token/log_probs，记录首末返回版本。版本正常单调时，它们形成生成区间；**不是逐token权重清单或tensor内容校验**。多turn和agent重写后的token/mask边界还须一同保留。

源码staleness使用 `(training_step-1)-max_version` 与 `(training_step-1)-min_version`，span使用 `max-min+1`。例如首末版本7/9、training_step=10，对应新鲜度下界0、最坏2、区间宽度3；这不证明版本8实际参与过生成，也不表示经历3次optimizer更新。解释这些量时须知道指标采集相对发布的时序，不能仅照注释中的“不同版本数量”理解。

[named-tensor安装路径](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L210)在有server且提供版本时，先等待返回的update future（若有），再clear_kv_cache、set_global_steps。[同步管理器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/checkpoint_engine/base.py#L510)还围绕传输做abort、release、finalize和resume。这是控制顺序，不是已经证明每个shard/LoRA/量化权重正确应用；版本赋值、线缆校验和功能对照是不同证据。

## 同权重也可能产生不同 logprob

rollout配置[默认logprobs_mode](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/config/rollout.py#L197)是processed_logprobs，并被传入vLLM启动参数。actor重算路径也有temperature缩放和fused/非fused分支。实际返回概率经过哪些processor，要按安装的vLLM版本及运行参数核对，不能仅从字段名认定是同一种概率。

即使权重相同，temperature、惩罚项、top-k/top-p/约束处理、tokenizer/template、完整上下文、position/attention mask、模态预处理或数值后端不同，也可能改变可比较的量。#8250报告temperature=1，不能据此将温度假设认作其故障原因；它只是对照时必须锁定的一项。

还要保留逐token残差。两个token的log-prob差为+.2/−.2时均值为0，非逐项一致；mean≈0不是“完全匹配”，也不是精确分布KL。只观测采样token的概率时，更不能排除其余词表区域的差异。

## 最小观测契约

每条用于对照的记录应能回答以下四组问题，缺失项用unknown而不是推定值补齐：

1. **同一输入吗**：sample/group/request/turn/token索引，完整prompt和response token身份，工具上下文、有效mask、position与模态处理；明确已生成token是否被重新tokenize或重写。
2. **同一策略吗**：rollout replica与分段版本、old快照与trigger、θ更新位置、reference身份；导出manifest应说明参数名映射、shape/dtype与shard，必要时校验安装后的内容。不是只记录一个global_step。
3. **同一概率吗**：采样/teacher-forcing方式、temperature、penalty和约束参数、logprobs_mode、库/kernel/精度及量化设置；明确当前比较raw还是processed概率。
4. **如何判差异**：逐token绝对/相对误差、分位数/最大误差、有效计数、非有限值和按版本/长度/turn分组。容差来自同设置重复基线，不先规定一个通用“治愈阈值”。

## 三路对照如何缩小范围

| 对照 | 条件成立后能支持什么 | 不能单独断言什么 |
|---|---|---|
| 训练actor与同checkpoint重载actor | 所选token上的导出/重载与该推理路径是否一致 | 全部参数或所有输入等价；vLLM已同步 |
| 同backend干净重载与live rollout | 固定版本、处理及cache政策后，将差异定位到live状态相关路径 | 唯一原因就是weight-sync或cache；仍可能是配置、shard、路由/状态差异 |
| rollout存储值与重算old/current | 配对版本与分段后，观察合法staleness或数据/概率路径差异 | 混合旧版本的response必须等于全当前版本重算；不等就一定是bug |

单个采样token概率相同可对应不同的其余词表分布；给所有logits加同一常数则整个softmax不变。功能对照可以发现不一致，但“对照通过”不能反向认证tensor内容完全相同。要验证参数内容，应使用规范化后的安装manifest/内容校验，并区分传输前、传输后与应用后检查位置。

## 本机验证范围

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250
```

[脚本](../../experiments/scripts/verify_verl_logprob_provenance.py)默认模式在Python3.9.6/3.12.13、NumPy2.0.2通过7组成功路径控制：原始base bypass分支、原始separate-async路由方法，queue/base重算/actor均替身，CPU slot只存标识符。另有7类概率/元数据反例，覆盖处理差异、采样token覆盖不足、softmax平移、均值抵消、版本区间宽度、声明标签与混合版本重算。它不运行模型、vLLM、真实快照/传输或GPU；可选异常检查见下节。

脚本打印输入hash。核心路由文件SHA256为trainer_base `f6c18e2056f6ae2f3fa5436ab24a97ff2c90f8d4bc59bf4f82cb20f80a77baf9`、trainer_separate_async `fb169bfe966816f18cdaaa361761b1e4dc58b577e69e7103211717ab25f70cfb`。源码与这些替身检查说明上述观测为何需要分层，不足以确定私有训练故障根因或提出修复。

## 同步完成点与异常传播

同一固定版本的named-tensor适配器与非naive管理器有两层成功顺序：

```text
适配器  发起receiver RPC → 发送bucket → 等待receiver future
        → 清cache → 标记版本
管理器  abort生成 → release cache → 建通信组 → 等待actor及rollout更新
        → 等待finalize → resume cache → resume生成
```

`_execute_method(non_block=True)`在有server时返回future，调用方随后等待它；这里的non_block不等于“已经安装完成”。管理器在同一次ray.get前已提交actor与rollout更新，某个ref报错也不代表其他远端工作未开始或已取消。以上是调用关系，不是测得的真实并行时序。

| 最后到达的阶段 | 失败后不能推定的状态 |
|---|---|
| bucket发送或receiver确认 | 不能推定没有部分应用，也不能推定所有shard已一致 |
| clear cache | 后续版本调用未发生不等于参数仍是旧版；cache清理可能也只部分完成 |
| set version | 调用报错时远端是否已赋值仍需核对，不能只看客户端异常判断版本状态 |
| manager finalize | 已有worker可能完成更新，方法没有统一撤销先前效果 |
| resume cache或resume生成 | 资源恢复与对外可服务是不同完成点，方法返回前的单个成功阶段不能替代全体就绪 |

这些方法体中没有捕获所测异常并自动rollback/resume的补偿块；这不等于整个进程没有外层清理，也不等于缺陷。失败后继续暂停可能是在避免暴露混合权重。**不能直接把“在finally无条件resume”当修复，也不能未确认各worker状态就宣称重试幂等。** 对恢复的判断需要worker集合、目标版本、最后一次提交/确认、安装后manifest及cache代次共同支持。

有两条路径不能套用上图：naive后端直接交给actor更新；delta-flush空流有仅推进版本、不发送payload和清cache的专用分支，协议意图是值未改变。后者不能只因没有传新payload就判漏同步；delta流内容与协议正确性不在此次执行范围。

### 阶段化故障注入

```bash
python3 experiments/scripts/verify_verl_weight_sync_control.py /path/to/verl-8250
```

[脚本](../../experiments/scripts/verify_verl_weight_sync_control.py)提取未修改的ServerAdapter.update_weights和CheckpointEngineManager.update_weights协程体，移除框架装饰器后以标准库asyncio运行。Ray/worker、sender、接收future、cache和版本设置均为本机替身，不读取或写入设备权重。Python3.9.6与3.12.13均通过21组：10组adapter控制（3种成功条件、非法format、delta委托以及5处异常）和11组manager控制（普通/naive成功及9处异常）。

注入点记录的是**调用尝试**而非成功提交。所测异常均传播到本机调用者，后续阶段未执行；fake ray.get按给定次序解析ref，仅检查管理器程序控制流。未测试真实Ray取消/并发、超时、worker死亡、通信错误语义、CUDA流完成、跨rank内容一致、外层auto_await运行模式或故障恢复。它是边界检查，不是native同步故障复现或#8250根因证据。

源文件SHA256：ServerAdapter `e82e1c6782573e03c871b982eb0653801f54cb1647367a604aa8b1c5454773cb`；CheckpointEngineManager `97b51131a5770892007b7fb07264a2657fe83255b085927acd08c952a5682f64`。

## CPU快照交换异常与训练退出

**方法未恢复当前权重，不等于默认训练会带着旧权重继续运行。** 同一固定版本的separate-async覆盖方法没有围绕重算设置恢复用的try/finally；当trigger≥1、恢复slot 0已经成功而base重算抛出异常时，后面的恢复当前快照与清理临时slot均不会由该方法执行。这是源码控制流边界，不是#8250故障复现。

下表使用标量actor替身：每次故障在该操作自身修改状态前抛出，先前已完成的操作保留效果。真实RPC/设备操作可能先部分修改再失败，不能套用这些标量状态。

| 注入位置 | 替身中保留的状态与未执行步骤 |
|---|---|
| trigger=0，保存后重算失败 | actor仍为本周期π_old；没有后续交换步骤 |
| trigger≥1，恢复slot 0后重算失败 | actor为π_old，当前快照仍在临时slot；未恢复当前、未清slot |
| trigger≥1，恢复当前失败 | 本替身仍为π_old、临时slot尚在；真实恢复是否部分完成未知 |
| trigger≥1，清临时slot失败 | 本替身已恢复当前权重，但临时slot仍在；不能由清理异常反推权重未恢复 |

V1 separate-async实际选用的[DetachActorWorker快照方法](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/experimental/separation/engine_workers.py#L123)还要求核对slot前置条件：**当cpu_saved_models字典已经存在但指定key缺失时，restore直接返回，不调用restore_handler。** 因而“方法正常返回”本身不是快照已安装的独立校验；正常调用序列是否会丢失key尚未证明，不能据此报新bug。fsdp2/veomni还将保存的state与global_spec分别传给handler，其他策略直接传保存值；本轮没有执行这些原生tensor helper。

外层语义不同：[PPOTrainer.fit](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L426)没有在该step周围吞掉异常并重试；重算失败使本次调用无法继续到后续actor update。[TaskRunnerV1.run](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/main_ppo.py#L133)在finally中结束tracking（失败exit_code=1），再以嵌套finally关闭TransferQueue，即使tracking.finish自身失败也会尝试close。**这些清理不是模型/slot回滚。** 默认这次fit/run调用退出；不据此推断Ray actor必定销毁、外部调度不重启，或私有调用者不捕获异常复用actor。后几项未检查，若要支持继续服务/重试，需另定义状态核验与恢复契约。

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250 --check-snapshot-failures
```

Python3.9.6/3.12.13、NumPy2.0.2均通过新增21组：15组原始separate方法异常传播（解耦trigger 0/1/3各阶段及bypass重算异常）、2组原始worker方法分支（fsdp/fsdp2，标量保存/恢复/清理及缺key不操作）、4组原始TaskRunner方法成功/初始化失败/fit失败/fit与tracking.finish均失败控制。方法体由AST提取并去除装饰器，base重算、actor、handler、trainer、tracker与queue均替身；TaskRunner检查与快照检查分别执行，未贯通真实fit。它不证明原生快照深拷贝、跨rank一致、collective完成、真实队列关闭、Ray失败聚合或故障后的恢复/继续训练安全。

新增源文件SHA256：DetachActorWorker `e6799095d8e83890806582e625a6a334cb3f6f7ae17b408673f096bb5c6d96d1`；main_ppo `270ffe0d0f5ddbc4e9af8d0f6eece5d11b1a6d6d76166603b5ebac81345dd5cc`。这些21组去装饰器检查没有执行RPC层，包装层另见下节。

## 快照RPC等待与失败传播

**默认阻塞调用不是跨rank事务，异常返回也不是全体worker已停止的回执。** 同一固定版本的三个快照方法只指定ONE_TO_ALL；[register](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/base/decorator.py#L398)默认execute_mode=ALL、blocking=True。ONE_TO_ALL把同一个slot参数复制给全部worker；[方法绑定器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/base/worker_group.py#L185)读取这些元数据，生成的[调用包装器](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/ray/base.py#L49)先execute，再ray.get整个结果列表，最后collect。execute_all虽是异步提交方法的别名，不表示这个已绑定的快照调用会直接返回refs。

[execute_all_async](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/single_controller/ray/base.py#L862)逐个提交worker RPC，收集refs；ONE_TO_ALL的collect原样返回列表，没有额外校验slot、内容或跨rank状态。三个worker方法正常完成均可返回None，因此一组正常返回值不是模型内容校验。

| 调用结果 | 可以据此判断与仍需核对的边界 |
|---|---|
| ray.get正常返回整个列表 | 这些远程调用已返回；不证明保存/恢复的tensor内容相同，缺key不操作仍可能正常返回 |
| 提交阶段本身抛异常 | 源码循环可能只提交了前缀worker；包装器没有等待已发refs、取消或回滚的补偿块 |
| ray.get因某任务失败而抛异常 | 不再进入collect；不能据此认定其他任务都已结束、都失败或已撤销 |

[Ray官方get文档](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html)说明，列表输入保持结果顺序，但任一关联任务报错时可立即抛异常，不等待其余任务完成；**结果顺序不是失败等待顺序**。2026-10-21所读文档标为Ray2.59.0，报告环境安装版本未知，verl只声明ray≥2.41.0；这项文档契约不能冒充reporter环境的原生复现。包装器本身没有补偿、全量异常汇总或本地get超时参数，也不排除外层系统另有超时/取消政策。

```bash
python3 experiments/scripts/verify_verl_logprob_provenance.py /path/to/verl-8250 --check-snapshot-rpc
```

可选模式要求Python≥3.10，因原始register检查含运行时类型union；实跑环境为Python3.12.13/NumPy2.0.2。6组控制执行原始注册元数据生成、参数复制/collect、方法绑定器及RayWorkerGroup提交/包装方法体：三个默认阻塞快照调用、显式非阻塞对照、get抛错、提交阶段抛错。Ray actor/ref/get、枚举、DataProto类型及TQ bridge为替身；get失败用“一个完成、一个失败、一个仍pending”的假设日程检查包装器不做补偿，不模拟Ray调度或测量真实任务状态。没有执行远端worker及其decorator inner、tensor快照、TQ转换、colocated/fused进程建立、真实异常收敛或GPU。

输入SHA256：decorator `718876c4c59e9f3cdb0339185be156aeb7b6d8eb42a2a6f919a3163961634ace`；ray/base `a61952c697b6c7f8220f7cb91043e514beadc9061007c1c140acb260e6aed8c5`；worker_group `3723ad4ba93afc8b715287a1e9f8754e4915af6fd93a44e57646beb28a8591ab`。本轮将V1来源与失败边界学习阶段收束；#8250仍需要真实配置、配对观测和受控对照，不以增加替身数量替代根因证据，也不据此提出自动恢复修复。

## 新版阶段 tracing 与证据关联

2026-10-21 读取已有 [verl #8315](https://github.com/verl-project/verl/pull/8315)，open、作者mengchengTang，head `4d6b3eb8606e24c1d70d648e92a1114129ec6c62`、目标base `619c397dc9917fed278b6e47c52a6620aef05466`、共同祖先 `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。这是新版观测入口，不是前文1876b06d报告环境已有的能力，也不是#8250修复。**阶段被记录，不能单独证明阶段成功、设备完成或参数内容一致。**

**2026-10-22版本更新**：PR仍open，head已到 `0cea139623cc8913d2241d21223df5682474041c`，目标base `f2923cbf154731918b1e36db8c567433f349fede`，共同祖先不变。相对前head仅新增65行原CPU测试、将update_weight实际lane改为字符串；tracking/两个server实现未变。当前真正PR为5文件138增4删，下方首轮“无新增测试”只描述旧head。新增实跑范围见[新版原测试与实际接线](#新版原测试与实际接线)。

### 区间究竟包住什么

[trace_state_decorator](https://github.com/verl-project/verl/blob/4d6b3eb8606e24c1d70d648e92a1114129ec6c62/verl/utils/tracking.py)只接受async方法；调用时先检查 `VERL_RL_INSIGHT_ENABLE == "1"`，关闭时直接await原方法，开启时计算lane并在trace_state上下文内await整个方法。它没有向context传递method参数、global_steps、request身份、权重hash或返回值。异常状态如何呈现还取决于实际rl-insight后端，不能只凭这个wrapper断言成功或失败标签完整。

| 新状态 | lane与覆盖范围 | 不能单独证明 |
|---|---|---|
| update_weight | 旧head为self.rank整数，当前为rank_加rank的字符串；位于register内层，包住worker方法 | controller提交/排队总时延、所有receiver安装完成、全rank一致 |
| vLLM/SGLang的wake_up、sleep | Ray actor name，与生成阶段使用相同actor lane | 真正发生了内存操作；方法内仍可提前返回或跳过某种mode |
| clear/release/resume_kv_cache | 同一server lane，包住原异步方法 | cache代次一致、实际释放字节数、模型参数已应用 |

例如非node_rank0、free_cache_engine关闭、部分colocated/standalone路径会在原方法内返回或跳过；装饰器在这些判断之前进入，所以短状态也可能只是no-op调用。vLLM release路径仍是sleep后只wake weights，SGLang则按tags调用memory occupation接口；同名阶段不意味着两个backend执行相同操作。

[update_weights](https://github.com/verl-project/verl/blob/4d6b3eb8606e24c1d70d648e92a1114129ec6c62/verl/workers/engine_workers.py)的非naive分支等待本rank的send_weights并可offload后返回；接收端应用/finalize/cache恢复仍须按实际checkpoint engine链追踪。naive分支的同步/barrier也来自原方法，不能把它推广到所有模式。该PR没有新增显存采样、参数manifest、逐token对照或为tracing专设的设备完成事件；“memory stages”是阶段名，不是显存字节测量。

### 标量 API 与开启成本

新log先检查实际模块的metric_log是否callable：可用时保留 `reward/mean` 这样的名字并传int(step)；否则只走旧metric_gauge，名字中的斜杠改下划线，且不双写。[rl-insight #206](https://github.com/verl-project/rl-insight/pull/206)在本轮API读取时已合并，但不能由此推断某台机器安装的包包含该API。版本号和API能力须分别记录。

这个能力判断不是容错协议：metric_log存在但调用出错时，代码没有自动改投gauge。开启trace后的lane求值、初始化和context进入也在原方法之前，wrapper没有用异常隔离来保证监控失败时照常执行。因此“11个函数体未改”不等于“开启监控完全不影响控制流”；本轮没有运行后端或验证开销/故障语义。

### 本轮核验与真机接续

首轮4d6b3eb8相对共同祖先的PR diff只有4文件、73增4删，没有新增配套测试。M1完整读取新增wrapper/标量分派、11处接线及对应原方法；Python3.9.6下4份完整源码AST解析和内存compile通过。另将两版方法的装饰器列表清空后比较AST，11个async方法的参数/函数体/返回注解与共同祖先相同，并核对每项新增一个装饰器。首轮没有exec这些源码或复跑CPU测试；这段仅保留旧版本静态核验范围，当前测试结果如下。

复查对象为tracking.py、engine_workers.py和两份async server。可用 `git diff cd58f550...4d6b3eb8 -- <四文件>` 定位变更；以 `git show <head>:<path>` 获取字节，ast.parse后compile但不exec。比较11方法时只清decorator_list，再对ast.dump(include_attributes=False)核相等，不移除或改写函数体。SHA256依次为：

- tracking.py：`c22afac06ddac8d6ddba4c3aa5a5ae4ad0d20170fddb4f6ac6756e2884f075f9`
- engine_workers.py：`2f6dd946a4989460ca64ae2f8de576fb278a32b529ac19b0bb80be3a2acdd79b`
- vllm_async_server.py：`b9694ff07d776395349159b68934e4e991a0e6027691af8f97feb690a21f236d`
- async_sglang_server.py：`57729bfebce2507fb271e28cc5c7cd4073afb8a5ecc31ec87b8815a4c8b9c354`

涉及真实权重/显存/生成的后续检查已补入[原M5补证条目](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另开任务或默认采用PR。保留分支/配置、rank/actor与step关联、异常/no-op标记及独立显存/安装证据；若缺获准监控环境或要新增数据外发，先记录条件，不启动服务或改变原报告版本。M1没有访问服务器，也没有新的连接失败实测。

### 新版原测试与实际接线

当前[原日志测试文件](https://github.com/verl-project/verl/blob/0cea139623cc8913d2241d21223df5682474041c/tests/utils/test_rl_insight_logger_on_cpu.py)共有12项，新增5项覆盖带step的metric_log，以及async decorator的动态lane、固定lane、关闭时不求lane、拒绝sync函数。Mock客户端把metric_log显式设None，再按测试启用MagicMock，避免“属性不存在但Mock自动生成callable”的假能力判断。

[三版本复查](../../experiments/verl-rl-insight/readme.md)在Python3.12.13/pytest8.4.2下，完整原RLInsightLogger/RolloutTraceConfig类经AST隔离执行，完整原测试模块保留fixture/断言；跳过的是包根、tracking其余后端与无关依赖导入。客户端为原测试MagicMock，没有运行rl-insight、Ray、服务或设备。

| 实现版本配当前原测试 | 结果 | 实际worker的lane（rank=2） |
|---|---|---|
| 祖先cd58f550 | 7过、5预期失败 | 尚无本decorator |
| 前head4d6b3eb8 | 12过 | 整数2 |
| 当前head0cea1396 | 12过 | 字符串rank_2 |

原测试中的Worker自行定义字符串lambda，因此即使生产接线仍传整数，logger suite也能全过。另直接执行两份engine_workers中**原lane表达式**得到上表差异，没有构造worker或执行update_weights。作者说明时间线查询需要非空字符串，当前接线与此约定相符；本机没有执行真实后端查询，不能把这项源码检查说成时间线显示已修好。

这区分了三层：helper正确处理传入lane、生产调用点传入正确身份、后端接收/关联/显示该身份。12项测试只验证前一层和Mock API调用。已有16协程lazy-init例也仅在同一事件循环内计数，不能外推跨线程/进程安全。旧“区间可含no-op、sender返回非receiver完成、开启监控可能改变异常路径”的边界保持不变；新增测试不使它变成真实同步或GPU验收。

当前tracking SHA256仍为上列`c22afac0…`；engine_workers为 `7166ff29e71eb101031ce88aa22feb32f0d960a5e795af0e8116f37aff757ec1`，原测试为 `874a695c536517254d666dc5b42a7309993a7f86d33e654ec7fbf66c6de58cf3`。详细命令、原类隔离方式、输入hash及首次网络审计误伤asyncio本地socketpair后的修正见[实验记录](../../experiments/verl-rl-insight/readme.md)。原M5条目只同步新版/验证范围，不另开任务或默认部署监控。

## 单 rollout worker 的 NIXL 拓扑

2026-10-21核对已有[#8316](https://github.com/verl-project/verl/pull/8316)，读取时open、rafiw，head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`、目标base `f2923cbf154731918b1e36db8c567433f349fede`、共同祖先 `619c397dc9917fed278b6e47c52a6620aef05466`。真正PR只改一行切片，未提交测试文件；正文勾选“新增单测”不能替代固定diff证据。这里验证的是**拓扑参数契约，不是权重同步完成**，也不说明#8250实际使用了此backend或故障由它造成。

**后续版本更新**：同日核对head已变为 `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`，仅比上述首轮head新增28行[原单测](https://github.com/verl-project/verl/blob/c2d800e71cdc4c7295b1e12e60bad82dd2fb637b/tests/checkpoint_engine/test_nixl_checkpoint_engine_on_cpu.py)，两份实现文件逐字未变，共同祖先及目标base不变。当前PR已含测试；上段“未提交测试”只描述旧固定head，不再作为当前覆盖判断。

### 单元素切片与链尾

[build_topology](https://github.com/verl-project/verl/blob/3a60b8710f8e1001110820f7a14cd0e71d9d5e07/verl/checkpoint_engine/nixl_checkpoint_engine.py#L289)将A个actor、R个rollout的metadata组织成链。有效通信rank为actor[0]的0及rollout的1到R，world_size=R+1；其余actor rank=-1、不连前后邻居。链关系为 `actor[0] → rollout[0] → … → rollout[R-1] → None`，不是所有A+R个worker都加入通信链。

rollout的next原式为 `metadata[-R + 1:] + [None]`。R=1时起点为0而不是“列表末尾之后”，因此返回整个metadata加None，长A+2；候选改为 `metadata[len(metadata)-R+1:] + [None]`，此时从len(metadata)切起，只得[None]。对满足metadata长度A+R且A/R均正的R≥2，两种起点表示同一位置。

### CPU证据与真实调用阶段

Python3.12.13，保留原classmethod、原manager两组长度断言及init_process_group的纯端点检查，A=1/2/4/8、R=1/2/3/4/8每版20组。基线16组过、4个R=1触发原 `rollout_kwargs[next_agent_metadata] must have length of 1`；候选20组全过，16个多rollout场景的完整输出与旧版相同。另按链关系检查rank/prev/next，没有新造NIXL agent或网络替身。方法、输入hash和[复查脚本](../../experiments/verl-nixl-topology/readme.md)保留；这是自建合成metadata控制，不是作者原始测试或完整manager执行。

[manager](https://github.com/verl-project/verl/blob/3a60b8710f8e1001110820f7a14cd0e71d9d5e07/verl/checkpoint_engine/base.py#L422)在真实调用中先prepare并收集actor/rollout metadata，再构造拓扑、检查每字段长度，最后才分派init_process_group。NIXL prepare还涉及buffer分配/注册和agent身份；这些本机均未执行。update_weights把各replica.workers展平成临时rollout worker group，所以R是该组world_size，不能直接等同replica数。

固定版本既有NIXL correctness用例为2 trainer/6 rollout，带temporary skip，含Ray/模型/原生backend；本机未运行，也不把device=cpu参数当成纯Python测试。列表长度及端点通过后，buffer、连接、传输、权重应用/finalize与恢复生成仍是独立完成点。若M5已有诊断确需单worker NIXL，可参考[原交接条目](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另排队、自动采用补丁或变动既有环境。

### 新增原单测的执行范围

新增测试只构造字符串metadata `["trainer", "rollout"]`，A=R=1，断言prev为[trainer]、next为[None]。复用原脚本的 `--upstream-case`，AST原样执行这一测试函数及其两个断言，类壳只承载原build_topology。Python3.9.6及3.12.13两环境均为基线1项预期AssertionError、新head 1项通过；不是扩写新模拟或把旧20组控制改称原测试。原测试SHA256 `699eb471d14bb946d47cd794416a23c04ee60a44b3ac206d2dafb8146e91541c`。

该文件顶层直接导入nixl_checkpoint_engine，而后者仍急切导入cupy、nixl、Ray、torch、ZMQ。本机跳过了整模块导入与pytest收集，没有安装/探测这些运行时。因此这里只确认原测试体对纯拓扑方法的回归能力，不能由文件名on_cpu或本次隔离通过推断无原生依赖的CPU CI可收集。新增测试仍不覆盖多worker、真实agent元数据或权重传输；原有M5交接边界不扩大。

## 组初始化的等待超时与任务退出

[verl #7045](https://github.com/verl-project/verl/pull/7045)是控制端等待的保护栏，不是 #6967 原生挂起的根因修复。2026-10-22读取open、作者aryanyadav0402，固定head `6f458f6d49076c647a42c00d240b5eaa7449fdb7`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际merge-base `220e9039902c6db56860e2afd659803dc34ec005`。真正diff三文件246增2删：新增97行helper、144行测试，以及manager接线。**当前head没有新增worker超时线程**；NCCL实现相对祖先逐字不变，不能混用旧worker侧版本的验证。

### 哪一段等待受到约束

[build_process_group](https://github.com/verl-project/verl/blob/6f458f6d49076c647a42c00d240b5eaa7449fdb7/verl/checkpoint_engine/base.py#L388)先prepare并取metadata，再构造/检查拓扑，最后分派actor和rollout两组init，拼接refs后交给新helper。其覆盖并不是整个weight sync的截止时间：

| 阶段 | 固定head的等待方式与边界 |
|---|---|
| abort请求、release KV | 在update_weights中先于建组；不受新helper计时 |
| prepare与metadata收集 | 仍是无显式timeout的ray.get |
| 构造拓扑、分派init RPC | 在进入helper之前；不受其ray.get超时约束 |
| 等actor与rollout的init结果 | 同一refs列表，一次ray.get，默认600秒 |
| 传权重、finalize、恢复KV/生成 | 仍为各自调用；不受这个组初始化timeout覆盖 |

`update_weights`每次非naive同步都调用建组；即便NCCL默认复用已存在group，init方法仍可能执行连接/barrier。因而新等待边界不只存在于第一次同步，也不由manager限定为NCCL backend；naive路径提前返回。这里是调用范围，不认证所有backend都做相同初始化或必定能在预算内结束。

### 停止等待不等于取消或回滚

[wait_for_group_init](https://github.com/verl-project/verl/blob/6f458f6d49076c647a42c00d240b5eaa7449fdb7/verl/checkpoint_engine/_group_init.py)只解析预算并调用 `ray.get(refs, timeout=...)`，将GetTimeoutError转成带原异常cause的CheckpointEngineInitError；其他异常不在该except中处理。没有cancel、kill、重试或补偿。当前 [Ray get文档](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html)将timeout定义为等待上限；[cancel是独立API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html)，行为还依任务/Actor执行模型而异。读取的文档为Ray2.59.0，不代表原报告或本项目已安装此版本。

超时异常沿manager当前方法向外传播，方法内没有finally继续finalize或恢复KV/生成；此前的abort/release动作不能据异常自动视为撤销。这说明必须区分“调用者获知失败”与“远程操作停止、资源释放、可重新开始”，不证明本次已观察到资源泄漏，也不提出在原group上重试的方案。PR选择由外层launcher结束作业；其正文报告的2×A100清理/新进程恢复针对旧head `da00403`，并明确当前控制端head复验待做，不能挪用为本版本验收。

### 参数合法性与检查时机

`_resolve_timeout_s`优先显式timeout_s，否则每次读`VERL_CKPT_ENGINE_INIT_TIMEOUT_S`，缺省600；转换成float后要求有限且大于0。原函数的12组纯CPU控制如下，Python3.12.13，仅AST隔离该函数和两常量，math/os为真实标准库：

| 输入 | 结果 |
|---|---|
| 不传参数、环境未设 | 600.0 |
| 不传参数、环境为12.5 | 12.5 |
| 显式5.0、环境为ten | 5.0，显式值优先 |
| 环境0 / −1 / inf / nan / ten | 5组均抛ValueError，消息指向环境变量 |
| 显式0 / −1 / inf / nan | 4组均抛ValueError，消息指向timeout_s |

拒绝发生在helper的ray.get之前，但manager先求值参数、分派两组init RPC才进入helper。因此“非法值不开始等待”不等于“非法配置在任何建组副作用前失败”。本轮只核源码求值顺序，没有分派RPC来制造该场景。

原15项测试中13项使用mock ray.get，另2项在本地Ray集群上以stand-in worker groups分别阻塞actor/rollout。作者的测试入口能覆盖接线，不是真NCCL挂起复现。本机独立CPU环境没有Ray，未补装、未运行这15项、完整helper或base模块导入；三原文件仅AST/内存compile通过。没有启动本地cluster、定时线程、通信、模型或设备，也不用更多Ray替身认证取消/恢复。

下面从固定Git对象复查相同12组参数控制（Python≥3.10；在含该对象的verl仓库中，不导入Ray）：

```bash
GIT_NO_LAZY_FETCH=1 python - <<'PY'
import ast, math, os, subprocess
from unittest.mock import patch
rev = "6f458f6d49076c647a42c00d240b5eaa7449fdb7"
tree = ast.parse(subprocess.check_output(["git", "show", rev + ":verl/checkpoint_engine/_group_init.py"]))
nodes = [n for n in tree.body if
         isinstance(n, ast.FunctionDef) and n.name == "_resolve_timeout_s" or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
             ("INIT_TIMEOUT_ENV", "DEFAULT_INIT_TIMEOUT_S") for t in n.targets)]
ns = {"math": math, "os": os}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<original-resolver>", "exec"), ns)
resolve, key = ns["_resolve_timeout_s"], ns["INIT_TIMEOUT_ENV"]
with patch.dict(os.environ):
    os.environ.pop(key, None)
    assert resolve(None) == 600.0
    os.environ[key] = "12.5"
    assert resolve(None) == 12.5
    os.environ[key] = "ten"
    assert resolve(5.0) == 5.0
    for is_env, values in ((True, ("0", "-1", "inf", "nan", "ten")),
                           (False, (0.0, -1.0, float("inf"), float("nan")))):
        for value in values:
            if is_env: os.environ[key] = value
            try: resolve(None if is_env else value)
            except ValueError as error: assert (key if is_env else "timeout_s") in str(error)
            else: raise AssertionError(value)
print("12 original pure-resolver controls passed; no Ray integration")
PY
```

SHA256：helper `ce401095e0d8d9c9711144dbe429aa62298faab431c57235a8f293edbb1724cb`；base.py `a86b084bdd68fe3dd6a91587147bd2e80c51b455863f6fe344a1623addf3c4a9`；原测试 `a5afc6997fd7473a25a4c1864d5444b0dab1a8b7ea0855abb75376628ed8eb78`；NCCL文件 `cc36e7a53579d60a76cb6ee811ca464dcddffdbc233a94c1af777f6f124a5eef`。当前版本的真机退出/释放/恢复条件仅补[原M5事项](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不另排队或默认应用，不将此保护栏当#8250根因或训练恢复证明。
