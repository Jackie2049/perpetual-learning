# verl TransferQueue 数据交换与启动资源

TransferQueue把训练轨迹的元数据传递与实际字段数据交换分开；它不是verl的权重同步backend。默认SimpleStorage使用Ray actor管理存储单元，并通过ZMQ交换数据。存储单元数量影响启动时的逻辑CPU预留与数据分片，不等于显存容量或吞吐保证。

2026-11-01补[流式采样的双重限流](#流式采样的双重限流)：#8203按未完成session补发，但另用未消费prompt组库存阻止积压；原19个CPU参数项与自编元数据控制区分两种槽位，非真实Ray／TQ消费或吞吐验收。

2026-11-01补[R2 padding 的专家负载与总分配](#r2-padding-的专家负载与总分配)：#7524原三个CPU参数体与自编计数区分峰值rank和总token–expert对；EP32峰值240→60同时总数240→1920，不等显存／吞吐收益。另见[路由值与记录有效性](#路由值与记录有效性)：#8471原22个CPU控制器测试通过，真实expert0与未知占位需要独立mask；debug与数值回退仍有边界，不能代替生产者／EP验收。

2026-10-30补[critic可选字段的取数边界](#critic-可选字段的取数边界)：已有#8457针对release/v0.9.1，在优势和指标取数前按use_critic选择values；下游指标已有开关不能保护上游取数。源码／语法核对不是TransferQueue版本兼容或训练验收。

2026-10-23更新：固定阅读[verl #8340](https://github.com/verl-project/verl/pull/8340)，head `881fb96937f5ddfae5d894bb9a48538d1b049e1e`，base及共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，7文件58增6删，读取时open/非draft。TransferQueue资源及数据路径以作者复现所用提交 `5fe4e49fe74aef544fe54da1fbeabf0b36d497eb`为准，不代表verl强制依赖此revision或本机安装了它。

## 流式采样的双重限流

**session结束可以释放生成槽位，却不等于prompt组已被训练消费。** Begunner的[已有 #8203](https://github.com/verl-project/verl/pull/8203)把V1的dispatch／replay控制面移入StreamingDataloader actor；读取时OPEN／非draft，固定head `6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96`、API base及实际共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`。这是候选接口，不能直接替换main `8df5baee`或M5现有环境；本轮人工reviews及inline comments为空，机器人评论不当作维护者批准。

[原异步实现](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/trainer/ppo/v1/streaming_dataloader.py)用两种单位控制补发。令训练batch含B个prompt、每个prompt固定n个session：

- 目标session水位 `W = round(watermark_batches * B * n)`。pending组占满n槽；running组占`n − 已终态session数`槽，完成及失败marker取并集。记总占用为F，未达到n个空槽不补一个prompt，基本补发数为`max(0, W − F) // n`。
- 未消费组数D包含pending、running、finished和failure四类；只有`separate_async`再设`C = round(max_outstanding_batches * B)`，将补发数限制为`min(基本补发数, max(0, C − D))`。终态组尚在库存中，因此只限制F不足以限制积压。
- colocate模式不设这个库存cap；`None`也表示不设。原初始化要求正水位且舍入后非零、cap batches不小于watermark batches；async的wait＋off-policy阈值1被拒绝。这些是局部配置门，不是稳定性或内存字节上限的证明。

session marker是[共享schema](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/trainer/ppo/v1/tq_schema.py)的`{uid}_{session_id}`，trajectory是`{uid}_{session_id}_{index}`；uid不得含下划线。计数读的是session终态而不是trajectory行数，同一session多次输出不能当成多次释放。超出n个终态marker会拒绝；该数量guard不验证每个session_id的语义合法性，依赖生产者遵守schema。

### CPU 控制与背压单位

[复现脚本](../../experiments/verl-session-watermark/verify.py)在Python3.12.13、pytest8.4.2、OmegaConf2.3.1下提取原七个方法、请求类型与指标累加器，加载完整原schema；选取[原actor测试](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/tests/trainer/ppo/v1/test_streaming_dataloader_actor_on_cpu.py)14个函数，保留参数化装饰器、fixture helper及断言AST，原生pytest执行**19项通过**。测试用原SimpleNamespace／方法记录器；基类为空、SkipManager注册表为空，没有导入完整verl、Torch、Ray或TQ。三个完整输入仅compile／SHA256记录在[结果](../../experiments/verl-session-watermark/result.json)。

额外自编控制直接调用原计数与补发方法，手动修改集合、以`_refill_exact`记录请求，不实际dispatch或消费。n=4、W=8、C=4时：

| 手动元数据状态 | 未完成session F | 未消费组 D | 补发prompt |
|---|---|---|---|
| 两组全在运行 | 8 | 2 | 0 |
| 三个session结束，尚不足一个新组 | 5 | 2 | 0 |
| 再一个session失败，凑齐四空槽 | 4 | 2 | 1 |
| 一个长尾session仍在、三个终态组待消费 | 1 | 4 | 0 |
| 手动移除一个已消费组 | 1 | 3 | 1 |

十个状态控制另核：pending始终占完整n槽；同一marker同时出现在finished／failure集合只计一次；orphan marker不减其他running组占用。n=0／−1和超额progress三次拒绝，另核validation idle gate。它们核的是原方法对给定元数据的响应，不是分布式元数据快照一致性或真实消费后的恢复速度。

### 完成顺序与适用边界

原19项也覆盖训练请求单次poll、未就绪时继续维护、validation请求不补train、count不消费、wait未达目标仍补发，以及materialize异常交付future。`_coordinator_step`先serve请求（内部poll，sample就绪时materialize），随后才refill；不能把count返回视为消费释放库存。validation idle gate停止训练侧维护，不代表取消已提交的生成。

[AgentLoop生产者](https://github.com/verl-project/verl/blob/6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96/verl/trainer/ppo/v1/agent_loop_tq.py)的session终态位于`_run_agent_loop`返回／异常之后，prompt终态还等待所有session任务settle。本轮仅阅读这段接线；轨迹写入完成、marker可见性、异常清理、真实TQ消费／重新补发与checkpoint恢复均未执行。若生产者提前发终态，控制面算式本身无法证明实际设备并发仍受限。

作者报告sync／separate-async保存恢复到step2、真实cap阻塞及消费后恢复；另报告64GPU、Qwen3.5-35B＋uni-agent＋swe-rebench的80步score55→64。以上均为外部自测，不是本机复现、同栈吞吐对照或调度策略的因果收益证明。已有实现不另开竞争方案；真机缺口只补[原条件交接](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)，不排新算力任务、不改M5专项；任何上游及已建PR的fork head写入仍须用户对具体动作明确同意。

## 启动默认值与 CPU 预留

[候选helper及调用点](https://github.com/verl-project/verl/blob/881fb96937f5ddfae5d894bb9a48538d1b049e1e/verl/trainer/main_ppo.py)将SimpleStorage默认单元数从固定8改为null，在TaskRunnerV1中填`2 * trainer.nnodes`，之后才打印/resolve配置并调用`tq.init(config.transfer_queue)`。显式值保持原样；helper没有新增类型/正数/资源容量校验，也没有取issue建议中的`max(2, ...)`。V1仍强制`transfer_queue.enable=True`，不是由该字段关闭整条数据通路。

[SimpleStorageUnit](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/storage/simple_storage.py#L872)和[TransferQueueController](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/controller.py#L985)均显式声明`@ray.remote(num_cpus=1)`。仅这两类actor合计需要`U + 1`个逻辑CPU，未包括训练worker、agent、奖励服务和其他会话：

| 训练节点配置 | 单元数 U | 单元加控制器的逻辑 CPU |
|---|---|---|
| 旧固定默认值，任意节点数 | 8 | 9 |
| 新默认，nnodes=1 | 2 | 3 |
| 新默认，nnodes=4 | 8 | 9 |
| nnodes=4，显式单元数3 | 3 | 4 |

这里不是物理核心占用率或线程上限。[Ray资源文档](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#physical-resources-and-logical-resources)将逻辑资源用于调度准入；显式CPU请求在actor运行期也占用配额，但不提供CPU隔离，不限制进程只使用一个物理核。因此CPU利用率低、内存有余量仍可能与actor等待同时发生。

[关联issue #7398](https://github.com/verl-project/verl/issues/7398)描述固定8单元挤占小节点CPU；PR正文另报告Ray2.59.0、单节点8逻辑CPU、TransferQueue5fe4e49下：8单元有actor停在PENDING_CREATION，120秒仍未完成初始化；改2单元能返回并余5CPU。**这是作者的CPU/Ray复现，不是本项目实验**，也不是GPU吞吐或完整PPO启动结果。

## 节点数量不等于可放置节点集合

[SimpleStorage bootstrap](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/storage/bootstrap/simple_storage_bootstrap.py)先创建U个actor，再取各自ZMQ信息。[节点选择helper](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/utils/common.py#L48)对alive节点ID排序，可用`required_node_resource`筛选，按模轮转并设置`NodeAffinitySchedulingStrategy(soft=False)`：

- 训练配置只决定U；实际参与轮转的是Ray当时符合筛选条件的节点，不自动等同trainer节点集合。
- 分配给每个候选节点的actor数接近均分，不代表该节点有空闲CPU、内存或SSD路径。`required_node_resource`在这里用于节点筛选，没有作为每actor的自定义资源数量请求。
- 因而“2×训练节点数”不能无条件表述成“每个训练节点实际运行两个单元”。总CPU够，也可能有硬亲和目标局部资源不足。

[process_zmq_server_info](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/utils/zmq_utils.py#L667)逐个`ray.get(actor.get_zmq_server_info.remote())`，没有传timeout。候选只改默认数量，未修改该等待、增加CPU admission check或启动失败回滚。[tq.init](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/interface.py)若发现已有命名控制器，会连接并使用已有配置；本次传入的单元数并不自动重建已存在的集群。

verl的`tq.init`位于其训练try/finally之前；若初始化本身未完成，不能用随后`tq.close()`的存在证明本次启动已有清理保障。降低默认开销有明确作用，但不能称为任意集群上的fail-fast修复或可重试启动事务。

## 存储单元与数据容量

该版SimpleStorage按`field_name → global_index → value`保存字段；容量检查统计活跃global_index key，不按字段数或字节数计数。配置了`total_storage_size`时，每单元上限为`ceil(total_storage_size / U)`；null表示这项key数量上限不设，并不代表主机内存无限。

因此增加单元数会增加actor/连接等开销，并改变每单元分片容量；不会凭空增加RAM、显存或SSD容量。某字段单样本很大时，即使key数量未达到上限仍可能占用大量内存。启用SSD offload还有路径和编码条件，不能从U直接估算可支持的模型或batch大小。

## 元数据与实际数据通路

[verl transferqueue_utils](https://github.com/verl-project/verl/blob/881fb96937f5ddfae5d894bb9a48538d1b049e1e/verl/utils/transferqueue_utils.py)中，BatchMeta标识partition/global index/字段；KVBatchMeta通过key和partition取元数据。它们不是跨进程GPU指针。`tqbridge`发现元数据后取实际TensorDict、调用worker方法，再按输出类型和collect条件决定是否写回字段；无元数据输入可以直接调用原方法。

例如rollout响应、reward/logprob和训练字段可以通过元数据关联；这不意味着`train_mini_batch`的参数更新自动成为TQ中的新模型权重，也不意味着所有方法都无条件写回数据。元数据轻量化减少控制端重复搬运，与底层数据传输是否复制是两件事。

[SimpleStorage manager](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/storage/managers/simple_storage_manager.py)用ZMQ multipart发送/接收序列化字段。[该版encoder](https://github.com/Ascend/TransferQueue/blob/5fe4e49fe74aef544fe54da1fbeabf0b36d497eb/transfer_queue/utils/serial_utils.py#L178)对不连续tensor先contiguous，对非CPU tensor调用cpu，再提取memoryview；decoder可从接收buffer构建view。**某个CPU buffer提取或frombuffer步骤免复制，不等于GPU到GPU全链路零复制**。其他backend如Mooncake不能与默认SimpleStorage混用结论，也不能据此断言TQ必须同GPU/NVLink或不支持跨节点。

模型权重走另一条控制链：[CheckpointEngine](https://github.com/verl-project/verl/blob/881fb96937f5ddfae5d894bb9a48538d1b049e1e/verl/checkpoint_engine/base.py)。naive的ColocatedCheckpointEngine保存/消费参数generator，其他backend可使用独立通信组；vLLM adapter还有分桶IPC通路。发送、接收、应用、cache恢复与版本标签仍是不同完成点，见[worker同步笔记](verl-worker-lifecycle-ray-weight-sync-reading.md#分桶接收失败与退出)和[一致性边界](verl-async-logprob-provenance.md)。

## Critic 可选字段的取数边界

**不消费某字段，还必须在取数前决定不请求它。** renjunying2的[已有 #8457](https://github.com/verl-project/verl/pull/8457)于2026-10-30新增，读取时OPEN／非draft，head `7c8d16de6ee74f9047876353a350a705366a3734`，目标是`release/v0.9.1`的base `77d4b4b93811cadd23a76cf55e7e696e2b8f1c0f`，不是默认main。作者报告新TransferQueue拒绝缺字段时，未启用critic仍请求values会阻断GRPO；正文没有给具体TQ revision、失败栈或已执行测试结果，不能将此报告外推所有TQ版本。

[V1 trainer](https://github.com/verl-project/verl/blob/7c8d16de6ee74f9047876353a350a705366a3734/verl/trainer/ppo/v1/trainer_base.py)初始化用[need_critic](https://github.com/verl-project/verl/blob/7c8d16de6ee74f9047876353a350a705366a3734/verl/trainer/ppo/utils.py)派生use_critic：显式critic.enable优先，未设时才由GAE决定，因此不能把“GRPO”直接等同use_critic=False。创建critic、调用`_compute_values`及更新critic均有此开关；`_compute_values`经critic infer后转换并写回values。

父版却在`_compute_advantage`与`_compute_metrics`的初始fields列表中无条件包含values。候选只在use_critic=True时append该字段；关闭时不请求，开启时仍要求真实生产，未制造零值或把GAE输入变成可缺。两原方法去除初始fields赋值及紧邻新增guard后，剩余AST与base分别相同；并未改写优势公式或指标统计。开启时values移动到列表末尾，不把字段集合相同误称请求顺序也相同。

[指标包装器](https://github.com/verl-project/verl/blob/7c8d16de6ee74f9047876353a350a705366a3734/verl/trainer/ppo/metric_utils.py)会先把fields传给kv_batch_get，之后才执行`compute_data_metrics(..., use_critic=...)`；后者的values统计已有if use_critic。包装器只对含routed_experts的ValueError尝试去掉该附加字段重取，不会从原fields删除values。这正是“下游有可选保护，但上游请求仍过早”的两个不同边界，不能靠在结果TensorDict上再pop字段修复此前的取数错误。

同次只读API核main为[321e4e450fcfc47a3491c946cd01dacd9ea7281a](https://github.com/verl-project/verl/blob/321e4e450fcfc47a3491c946cd01dacd9ea7281a/verl/trainer/ppo/v1/trainer_base.py)，其两个方法仍无条件请求values；故本release候选不能写成“默认主线已经修复”。这只是代码身份观察，不是主线／旧TQ／M5现有环境的失败复现，也不据此另开竞争PR。

### 示例配置是另一条兼容边界

PR另改[Qwen3.5 FSDPTurbo示例](https://github.com/verl-project/verl/blob/7c8d16de6ee74f9047876353a350a705366a3734/examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh)：旧GPU eager／NPU fused dispatcher改为两者all2all，再以GPU False／NPU True的`ep_use_fused_ops`分别传入actor和reference的ep_plan。dispatcher名称与fused实现开关被拆开；这不是只修trainer字段，也不能从bash语法通过推出Hydra组合、Turbo实现或任一设备可运行。原脚本有自动设备探测，本机未执行它。

本轮仅Python3.12.14的六份完整版本／路径输入AST／compile、两个原脚本`bash -n`，原函数、严格字段后端、Ray／TQ、Hydra及训练均未运行；不是作者pytest或旧失败／新通过验收。源码SHA256、复查入口和Git对象读取受阻记录见[当日日志](../../diary/2026-10-30.md#1146--critic-可选字段与取数时机)。已有作者，不改候选或M5脚本／环境，不新增算力队列；后续待实质代码／测试反馈，不扩成“全部可选字段均已修复”的结论。

## 路由记录的层选择与全局坐标

**减少拉取的层数，需要同时保留消费者理解的层号；不能只把全局张量切小。** [#8160](https://github.com/verl-project/verl/issues/8160)提出将R3的`routed_experts`按层存成独立TQ字段，只获取当前流水线阶段需要的层。2026-10-24读取issue为open、0评论，报告者weiziyoung明确已有内部运行时patch并计划提交PR；这是已有贡献方向，不据未搜到关联开放PR另认领。本节固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`，不是对未公开patch的审查。

### 字段选择发生在取数之前

[register装饰器](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/single_controller/base/decorator.py)接入`tqbridge`；后者在调用原worker前将元数据变成实际TensorDict。[KV元数据转换](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/transferqueue_utils.py)能先按`fields`调用`select_fields`，但`_async_meta_to_realdata`本身只是把传入元数据交给`async_get_data`，没有PP层区间选择。若元数据仍包含一个覆盖全部层的`routed_experts`字段，这个桥接层不会自动把字段内部的层维切窄；在已取出的TensorDict上切片也不能追回此次网络字节。

这说明“字段级过滤”与“张量维度切片”是不同接口。提案先把层拆成字段，才有机会在取数前表达所需层；固定宽度层名是其保证排序的约定，不是当前bridge自动解析层号的功能。字段选中还须保持相同样本键、token顺序、dtype和层名到全局层号的映射，不能以拼回后的shape正确代替身份正确。本文没有运行TQ存储或核定实际wire字节。

### 当前消费者不只依赖阶段偏移

[Megatron forward](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/megatron/transformer_impl.py)在R3模式先对每条序列补齐可能缺失的末token路由，再构造replay mask，并向`set_router_replay_data`传入实际`unwrapped_model`。普通末token占位是token轴对齐，不是为缺失层提供映射。

[replay消费者](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/megatron/router_replay_utils.py)先按THD/BSHD、量化对齐与CP布局预处理，随后将整个层维的路由tensor搬到设备并做sequence-parallel scatter，再转成层优先。层数等于`tf_config.num_layers`时按全局层下标读，否则按MoE层序号读。有model参数的主路径遍历该模型真实router的`layer_number-1`，不是仅按全局router实例列表或固定`rank×L/PP`截取。直接交一个紧凑的局部层tensor，可能进入另一种索引解释，出现漏覆盖错误或错配；必须明确保留全局坐标或显式映射，不能把“更小”当作语义等价。

无model的兼容路径才通过本地offset/end及router列表分配目标。层数helper还处理自定义pipeline layout、不均匀首尾阶段、embedding/loss占位与VPP；因此“每阶段拥有均匀连续的L/PP层”只是受限布局。提案已选择VPP、非均匀等情况回退，当前源码不能代替这些fallback的实现与验证。

提案把恢复全层宽度放在SP scatter之后，意图避免把不消费的层提前搬运/处理；当前主线没有这项优化。层轴裁剪、token轴ragged/CP/SP重排、专家ID位宽是三个不同维度：改一维不能认证其他维未错。也不能把R3路径结论直接迁移到仅含MoE层的R2或其他后端。

### 性能报告与本项目范围

作者报告48层、TP4×PP4×CP8、16节点/128GPU配置中，单元fan-in由157GB降至39GB，`timing_s/ref`从94.6秒降至65.7秒；这是内部patch与特定栈的结果。12/48层说明路由payload约四分之一的来源，不代表整批所有字段、全部网络/H2D字节或整步时间都降四分之三。均值logprob差和grad norm接近也不是逐token路由、loss与梯度完全相等的证明。

M1只阅读issue、有限编号/关键词PR检索及上述调用点，对bridge、router工具、Megatron engine与register四个原模块做AST/内存compile；没有执行原测试、worker、TQ、Megatron、collective、路由或性能实验，不造PP/TQ替身来验证尚未公开的patch。版本/hash留[当日日志](../../diary/2026-10-24.md)；需要匹配资源的后续条件补[原TQ交接](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)，不新增训练队列或竞争实现。对外协作仍须用户逐项授权。

## R2 padding 的专家负载与总分配

**把 padding 分散到专家，可以降低最忙 rank 的分配数，同时增加总分配数。** EazyReal的[#7524](https://github.com/verl-project/verl/pull/7524)在2026-11-01读取为OPEN／非draft，固定head `a79099997aeb95bbdf1771fd7631bd9e67101c4c`，API目标base与实际共同祖先均为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`。作者已明确披露此取舍，不能另当新发现的缺陷或GPU优化成果。

[原helper与入口](https://github.com/verl-project/verl/blob/a79099997aeb95bbdf1771fd7631bd9e67101c4c/verl/utils/megatron/router_replay_utils.py)只对`valid_rows=False`的行原地重填top-k。连续分配先轮转EP rank，再轮转rank内专家，每层另加偏移；真实记录行不变。入口仅在`replay_mask is None`且`moe_expert_capacity_factor is None`时启用，显式0也会跳过；不能把“无容量限制”随意解释成数值0，或只凭R2／R3名称判断门是否打开。跳过是为了维持已有capacity selection／token dropping行为，不是已移除padding对capacity的影响。

THD路径为真实route values建全True jagged有效行标记，与routes使用相同quant_align_size、min_local_rows、local_cp_size、cp_layout预处理；BSHD路径从attention_mask.bool()走同一packing。分散发生在CP packing之后、SP scatter之前。此临时有效行标记区分真实token与对齐padding，不是下节R3的rollout记录来源mask；全True输入也不能替代packing后的行对齐验证。

### CPU 分配计数

[脚本](../../experiments/verl-r2-padding-balance/verify.py)AST提取完整原helper及[原参数测试](https://github.com/verl-project/verl/blob/a79099997aeb95bbdf1771fd7631bd9e67101c4c/tests/utils/megatron/test_router_replay_model_walk_on_cpu.py)，保留decorator，手动执行三个原参数体：256专家、K8、两层，EP64／padding8及EP32／padding5、32均通过。断言覆盖真实行保留、padding专家范围／每行互异，以及每层rank和专家分配差至多1；不是原生pytest收集、原模块fixture或整套Megatron测试。

另以240个padding行、256专家、K8、一层做自编布尔`row × expert`计数，每行重复ID折叠成唯一token–expert对，再按连续专家块汇总rank。旧输入均为8个expert0，新输入由原helper产生：

| EP 大小 | 旧最忙 rank 对数 | 新最忙 rank 对数 | 旧总对数 | 新总对数 |
|---|---|---|---|---|
| 32 | 240 | 60 | 240 | 1920 |
| 8 | 240 | 240 | 240 | 1920 |
| 4 | 240 | 480 | 240 | 1920 |

三组原始top-k索引条目始终都是240×8＝1920；增加的是去重后的token–expert对，不是索引tensor多了8倍元素。这种计数不是实际MCore dispatcher的分配／通信／存储测量，峰值rank均衡也不能单独推导显存峰值、gate梯度或整步性能。自编无padding控制确认原对象／内容不变；原入口guard表达式的八个独立求值确认mask有无×capacity None／0／0.5／1的开关，仅表达式执行，不是完整入口或容量选择实跑。

复查：`python experiments/verl-r2-padding-balance/verify.py /path/to/verl-git`，需预缓存固定Git blob；执行禁lazy fetch。Python3.12.13／Torch2.9.1／pytest8.4.2，[JSON](../../experiments/verl-r2-padding-balance/result.json)保留输入hash及逐组结果。作者59项原mock测试、另73项补充harness和14项hook是作者报告，未计入本机通过数；真实THD／BSHD、CP／SP、Megatron dispatch、capacity／bias更新、训练与GPU显存／时间均未执行。缺口沿[原M5参考](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)保留条件，不自动应用候选或新排训练。

## 路由值与记录有效性

**路由到expert0是有效数据，缺失行填0是存储约定；二者不能只靠同一个值区分。** horcham的[#8471](https://github.com/verl-project/verl/pull/8471)在2026-11-01读取为OPEN／非draft，head `7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e`、API目标base `8e48d72de4766228c36623c90f9f5c49ee5b45f3`，五文件。候选为VeOmni R3添加`routed_experts_mask`，表示该token行是否确实记录了rollout路由；不是前述Megatron的层裁剪方案，也不另造竞争实现。

### 控制器如何消费掩码

[原控制器](https://github.com/verl-project/verl/blob/7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e/verl/utils/veomni/router_replay.py)只替换top-k索引，不重算模型专用的gate权重。显式mask存在时，True使用记录target，False保留当前native；缺mask时才检查重复top-k与整行全零，两者任一命中就退回native。topk1不可能有两个重复slot，新增全零检查避免未知行无条件变成expert0，但无法保留没有mask的真实expert0。

候选另有`VERL_ROUTER_REPLAY_DEBUG=1`检查：显式True且target整行全零会报错。自编CPU控制使用topk1 targets`[[0],[3]]`、native`[[7],[7]]`：

| 模式 | 原控制器结果 | 解释 |
|---|---|---|
| 无mask，debug关 | `[[7],[3]]` | 全零回退无法区分真实0与未知占位 |
| mask=`[True,False]`，debug关 | `[[0],[7]]` | 有效0保留，第二行即使值3也不重放 |
| 同mask，debug开 | AssertionError | 该guard仍拒绝真实topk1 expert0，不能把普通路径的支持外推debug模式 |
| 单行target=`[[5,5]]`、mask=True，debug开 | `[[5,5]]` | debug只核全零，不认证所有重复ID或下游permute安全 |

这些是原控制器的局部行为，不是发现真实EP崩溃或另立修复。**显式有效性掩码与专家ID合法性是两项检查**；True不能自动证明记录内容正确。begin_microbatch会拒绝mask行数与首层target行数不等，但同shape不能证明token／层身份相同。模块还按router对象身份处理重复回调，首轮fire顺序决定层slot；mask没有解决层映射问题。

### 生产者与布局必须一起传递

[agent输出](https://github.com/verl-project/verl/blob/7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e/verl/experimental/agent_loop/agent_loop.py)在as_dict右侧对齐路径将实际拷入的前num_rows行置True；左padding路径则将start_pos:end_pos区间置True。它标的是拷入区间，不逐行溯源；若上游数组内部已经夹有未知占位，仅凭区间长度无法识别，需生产者保证。不能只引用字段注释便认定所有多轮工具响应空洞已正确标记。

[padding重包](https://github.com/verl-project/verl/blob/7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e/verl/workers/utils/padding.py)让routes和mask使用同一unpad indices／cu_seqlens；[VeOmni engine](https://github.com/verl-project/verl/blob/7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e/verl/workers/engine/veomni/transformer_impl.py)再对二者补相同pad后缀（mask补False）、按相同Ulysses规则切token轴，最后把routes按层unbind，mask仍为每token一条。R2完整record返回时mask全True。上述是源码链，未执行生产者／重包或SP；同一schema字段不等于已证明全链路持久化和对齐。

### 原 CPU 测试与实际范围

[脚本](../../experiments/verl-router-row-mask/verify.py)在Python3.12.13／Torch2.9.1／pytest8.4.2下，将完整原控制器与[未修改的原测试模块](https://github.com/verl-project/verl/blob/7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e/tests/utils/veomni/test_router_replay_on_cpu.py)加载到内存，由自定义pytest.Module交给原生pytest收集、fixture与assert执行，22项全过；仅绕过verl包初始化，没有改测试体。fixture是原nn.Module空router，install／uninstall使用原VeOmni hook替身；反向重算测试是按逆序手动触发回调，不是模型activation checkpoint实跑。完整[JSON](../../experiments/verl-router-row-mask/result.json)保留22项逐名结果、四自编控制、五输入hash。复查：`python -B experiments/verl-router-row-mask/verify.py /path/to/verl-git`，需预缓存固定blob，执行禁lazy fetch。

五完整Python输入AST／compile通过，生产者／padding／engine三模块只读与编译；未跑作者另13个engine helper测试、真实SparseMoeBlock、VeOmni后端、TQ、SP、EP、训练或设备。作者瞬态分配下降及48层外推不是本机峰值显存／吞吐测量。M5若日后在已有获准匹配R3任务中独立选用，沿[原TQ交接](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)先核真实有效性来源、token／层坐标与debug／topk1边界；不自动套候选、部署或新增训练矩阵。

## 配置验证范围与复现

[复查脚本](../../experiments/scripts/verify_verl_tq_config.py)固定base/head Git对象，用真实Hydra合成完整配置树；只AST执行原`set_default_tq_storage_units`，将该helper暴露给完整原测试模块，跳过main_ppo其余导入和Ray装饰器。没有改原测试体或用dict替代DictConfig，也没有导入torch、Ray、TransferQueue或构造调度器替身。

M1 Python3.12.13、Hydra1.3.2、OmegaConf2.3.1、pytest8.4.2：

- 原3个参数化测试通过：nnodes1→2、4→8、显式3保持3。
- 四个generated配置（ppo、megatron、torchtitan、veomni）各做base/head配对合成；nnodes1时base保持8，head填值前null、调用helper后2。共8次额外合成，不是8次训练启动。
- Hydra/OmegaConf等仅装入临时target，未改现有venv；2个变更Python模块另经AST/compile。未运行TaskRunner、tq.init、资源预留、数据传输、性能或GPU/NPU验证。

```bash
# 使用已有Python3.12+/pytest环境，依赖可装到独立target。
tq_deps="$(mktemp -d)"
python -m pip install --target "$tq_deps" hydra-core==1.3.2 omegaconf==2.3.1
python -B experiments/scripts/verify_verl_tq_config.py /path/to/verl-git --deps "$tq_deps"
```

SHA256：head main_ppo `04cb697f16d3242092e592a70fad8ae26739dfa1b25c860208ec1534ec7e634f`；原helper AST `a0d2f8fe5490d249f79602890d54f5656218adfb5e391836e01b471464d186f3`；原CPU测试 `7c463c1ddf512809afbe47558cfacc5f64dfcd6e268687d9843a8b4fbfedc1be`。

真实启动首先缺CPU/Ray复现，不是“GPU离线”。若M5日后已有获准TQ环境恰需诊断，参考[启动资源交接](../../experiments/handoff-m5.md#verl-transferqueue-启动资源参考)，不为本候选自动部署、切环境或启动训练。

## 旧生命周期笔记的纠正与相关入口

本页原2026-07-15记录把TQ数据交换、checkpoint权重同步、ZeRO优化器分片与KV传输混在同一比较中。撤回其中“GPU共享内存即TQ”“14GiB权重约1ms”“100倍或280倍加速”“RTX4090必选某配置/永不使用sleep level2”等推导及固定19.2GiB预算；它们没有本项目对应配置和实测证据，不能作为部署依据。LoRA更新量较小也不保证无陈旧权重或部分失败风险。原始措辞可在Git历史查阅，不继续复制到当前结论。

原worker生命周期的可复用分层保留：生成数据、计算训练字段、执行参数更新、同步推理参数、恢复生成分别有自己的状态与资源约束。TrainingWorker封装engine的train/infer调用，并非固定CPU_Adam/ZeRO2；分组和sleep行为需按当时实现/配置判断。相关主笔记：

- [worker、Ray与权重同步](verl-worker-lifecycle-ray-weight-sync-reading.md)：区分部署模式及同步完成点。
- [GRPO单样本与同分组](verl-grpo-core-algos-reading.md#单样本与同分组的不同回退)：按UID归约与回退，不用未经核验的group_by_prompt开关代替实现。
- [GRPO数据流](verl-grpo-data-flow-reading.md)：rollout、会话完整性与batch字段。
- [梯度检查点](../fundamentals/gradient-checkpointing.md)：重算与存储，不由静态显存加法认证模型适配。

该head的[vLLM版本选择](https://github.com/verl-project/verl/blob/881fb96937f5ddfae5d894bb9a48538d1b049e1e/verl/third_party/vllm/__init__.py)在NPU分支保留level1、支持的vLLM版本分支可设level2；adapter又可受layered_summon影响。这里仅读源码，不据消费卡型号独立断言sleep策略安全或性能，也不改M5既有方案。

