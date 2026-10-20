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
