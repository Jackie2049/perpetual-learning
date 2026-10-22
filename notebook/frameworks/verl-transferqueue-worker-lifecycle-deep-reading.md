# verl TransferQueue 数据交换与启动资源

TransferQueue把训练轨迹的元数据传递与实际字段数据交换分开；它不是verl的权重同步backend。默认SimpleStorage使用Ray actor管理存储单元，并通过ZMQ交换数据。存储单元数量影响启动时的逻辑CPU预留与数据分片，不等于显存容量或吞吐保证。

2026-10-23更新：固定阅读[verl #8340](https://github.com/verl-project/verl/pull/8340)，head `881fb96937f5ddfae5d894bb9a48538d1b049e1e`，base及共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，7文件58增6删，读取时open/非draft。TransferQueue资源及数据路径以作者复现所用提交 `5fe4e49fe74aef544fe54da1fbeabf0b36d497eb`为准，不代表verl强制依赖此revision或本机安装了它。

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

