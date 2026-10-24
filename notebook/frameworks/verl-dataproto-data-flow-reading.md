# verl DataProto 数据流架构源码阅读

> 2026-06-15 | 源码: verl/utils/dp_rpc.py + verl/utils/tensor_dict.py + verl/trainer/ppo/ray_trainer.py
> 历史概览：旧版DataProto的split/merge设计。下方6月结构保留当时的学习视角，不代表2026-10当前实现的完整复核，也不提供跨Worker零拷贝保证。

2026-10-24补充：[图片字节与解码对象的生命周期](#图片字节与解码对象的生命周期)。原方法与真实Pillow对照确认dataset侧清理边界；原“数据开销完全可忽略”概括撤回，张量字节算例也已纠正，不能覆盖raw_prompt中的图片或序列化峰值。

2026-10-25补充：[多模态特殊 token 检查的 CPU 开销](#多模态特殊-token-检查的-cpu-开销)区分原检查方法、完整processor与训练吞吐；纯文本走多模态processor也可能付出标记计数成本，但不是所有纯文本入口都走此路径。

## 多模态特殊 token 检查的 CPU 开销

[#8261](https://github.com/verl-project/verl/issues/8261)的性能问题已有官方[Transformers #47580](https://github.com/huggingface/transformers/pull/47580)修复，后者已合并；本项目不另写重复补丁。旧 `_check_special_mm_tokens` 对每种已声明模态，把input_ids的每行转成list后计数。对torch Tensor，list中的元素仍是零维Tensor，逐项比较和真假判断带来大量Python／Tensor边界操作。即使文本没有图片，检查仍须扫描IDs来确认数量为0。

新方法先对整个input_ids调用一次 `tolist()`（若支持），再用原list计数逻辑。它不是在Tensor上做向量化sum，也没有消除每种模态的线性扫描；减少的是逐元素Tensor比较，以及反复跨表示边界。固定官方tag：5.12.1为 `ddb849abe009d1089e6c691bfc897f27211c663c`，5.15.0为 `5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9`。从这两个版本[旧](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/processing_utils.py)／[新](https://github.com/huggingface/transformers/blob/5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9/src/transformers/processing_utils.py)源码AST提取完整原方法，绑定仅含特殊token属性的输入夹具，不导入整个Transformers或构造processor。

### 本机函数级对照

macbook-M1、Python3.12.13／torch2.14.1／NumPy2.5.4，torch intra-op线程设1。输入为一行131,072个int64 token，文本不含图像标记，检查一种image模态；数据构造在计时外，每种条件预热1次后取5次中位数：

| input_ids表示 | 5.12.1原方法 | 5.15.0原方法 |
|---|---|---|
| CPU torch Tensor | 319.422 ms | 1.379 ms |
| Python list | 0.851 ms | 0.844 ms |

这是本机原方法的局部耗时，不是完整processor、长文本分词、内存峰值、GPU或训练吞吐。它也不是原issue中Xeon／torch2.13的耗时复跑；不能把两台CPU或两种测量范围的时间互相替代。

另以纯文本、正确image计数、缺少token、额外token、image/video两模态、总数相同但逐行分配错误六种输入，各接list／CPU torch／NumPy三表示，共18组成对比较、36次方法调用；两版接受／拒绝结果及完整ValueError文本一致，调用后原input_ids对象和数值未变。逐行 `[2,0]` 对文本计数 `[1,1]` 仍拒绝，不能用batch总数替代逐例检查。这些仅是原计数契约，不认证实际图像内容、模板或完整tokenizer等价。

### verl 的调用范围与依赖约束

固定verl主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的[processor helper](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/tokenizer/tokenizer.py)默认setdefault(return_tensors, pt)，显式 `mm_processor_kwargs.return_tensors=None` 不被覆盖。因此issue临时绕法改变的是processor返回表示，不是关闭计数检查或修好全部预处理。

[RL dataset过滤](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/dataset/rl_dataset.py)在无image/video/audio时直接调用tokenizer，不应把该计数开销外推到所有纯文本过滤；[AgentLoop后处理](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/experimental/agent_loop/agent_loop.py)只要processor存在便可能调用helper，取出多模态字段后还显式convert_to_tensors(pt)。None中间表示不等于全流程不再构造Tensor。这里仅读接线，没有运行这两条完整路径或认证临时绕法对全部模型／模态安全。

该固定lock声明SGLang0.5.20、Transformers5.12.1及Torch2.13的CPU／CUDA分支；本机微基准并未安装这组完整环境。依赖升级应协调推理后端及整体锁文件，不能仅因局部函数更快就单改pin或全局monkeypatch。五份完整源码AST/compile通过（两官方processing_utils、三verl接线）；本轮无依赖安装、processor资产／模型下载、GPU或新M5任务，原输入和耗时样本见[日志](../../diary/2026-10-25.md)。

## 1. DataProto类层次

```
DataProtoBase
  ├── meta_info: dict (非tensor元数据)
  ├── batch: BatchTensorDict (tensor数据容器)
  └── pad_batch: BatchTensorDict (padding后的数据,用于对齐batch_size)

DataProto (extends DataProtoBase)
  ├── split(split_size): 按batch维度分割 → list[DataProto]
  ├── merge(data_proto_list): 按batch维度合并 → DataProto
  ├── reconstructor: callable → 知道如何合并split后的结果
  ├── select(batch_ids): 选择特定batch索引
  ├── slice(length): 截取前N个batch
  ├── union(other): 合并两个DataProto的meta
  └── reorder_by_indicies(indices): 重排batch顺序
```

## 2. BatchTensorDict设计

```python
# verl/utils/tensor_dict.py
class BatchTensorDict:
    """所有tensor共享batch_dim的字典容器"""
    # 核心属性:
    # - 所有tensor必须有相同的batch_size(第0维)
    # - split/merge沿batch_dim操作
    # - 支持torch.compile(静态shape)

    def split(self, split_size):
        # 每个tensor沿dim=0切分 → 每个chunk是新的BatchTensorDict
        chunks = {}
        for key, tensor in self.items():
            chunks[key] = tensor.split(split_size, dim=0)
        return [BatchTensorDict(dict(zip(self.keys(), v))) for v in zip(*chunks.values())]

    def merge(self, tensor_dict_list):
        # 每个key沿dim=0 cat → 重建完整BatchTensorDict
        merged = {}
        for key in self.keys():
            merged[key] = torch.cat([td[key] for td in tensor_dict_list], dim=0)
        return BatchTensorDict(merged)
```

**关键**: BatchTensorDict强制所有tensor共享batch维度 → split/merge只需沿dim=0 → 极简!

## 3. DataProto核心API

```python
# verl/utils/dp_rpc.py
class DataProto(DataProtoBase):

    def split(self, split_size):
        """按batch维度分割DataProto → list[DataProto]

        用途: 将一个batch分成多个chunk → 发给多个Worker
        流程: batch.split(split_size) → 每个chunk成为新DataProto
        meta_info: 深拷贝到每个chunk(非tensor数据不可split → 全量复制)
        """
        data_proto_list = []
        batch_list = self.batch.split(split_size)
        for batch in batch_list:
            data_proto = DataProto(batch=batch, meta_info=deepcopy(self.meta_info))
            data_proto_list.append(data_proto)
        return data_proto_list

    def merge(self, data_proto_list):
        """合并多个DataProto → 1个DataProto

        用途: 从多个Worker收集结果 → 合成完整batch
        流程: BatchTensorDict.merge → meta_info取第一个(假设一致)
        """
        batch = BatchTensorDict.merge([dp.batch for dp in data_proto_list])
        meta_info = data_proto_list[0].meta_info  # 取第一个的meta
        return DataProto(batch=batch, meta_info=meta_info)

    def reconstructor(self):
        """返回merge函数 → 用于Ray RPC回调

        用途: split后发送给远程Worker → Worker返回结果 → reconstructor知道如何合并
        流程: return lambda data_proto_list: DataProto.merge(data_proto_list)
        关键: reconstructor随split数据一起发送 → coordinator不需要知道合并逻辑!
        """
        return lambda data_proto_list: self.merge(data_proto_list)
```

## 4. Split-Merge-Reconstruct模式详解

```
完整数据流 (PPO训练loop):

1. ActorRolloutRefWorker.generate_sequences:
   → 输出: DataProto(batch={tokens, log_probs, ...}, meta_info={uid, ...})

2. Coordinator拿到DataProto → split(split_size=N_workers):
   → [DataProto_chunk_0, DataProto_chunk_1, ..., DataProto_chunk_N]
   → 每个chunk包含1/N的batch + 完整meta_info(深拷贝)

3. 每个chunk发送到对应Worker(通过Ray RPC):
   → CriticWorker.infer(chunk_i) → chunk_i新增values tensor
   → RewardWorker.compute(chunk_i) → chunk_i新增rewards tensor

4. Coordinator收集所有Worker返回的chunk → reconstructor合并:
   → DataProto.merge([chunk_0_result, chunk_1_result, ...])
   → 重建完整DataProto, 包含所有Worker的输出

5. Coordinator发送完整DataProto到下一步Worker:
   → ActorWorker.update_policy(full_DataProto)

关键: split→distribute→process→merge→next step!
```

### 为什么meta_info不split?

```
meta_info是非tensor数据(uid/token_ids/config flags等):
  → 不可沿batch维度切分!
  → 解决: 深拷贝到每个chunk → 每个Worker都有完整meta_info
  → 代价: meta_info内存=N×原始(每个Worker都持有完整副本)
  → 优化: meta_info通常很小(几KB) → 深拷贝成本可忽略

  但如果meta_info很大(如长文本)?
  → 需要手动优化: 只传必要的meta字段 → 减少深拷贝开销
```

## 5. DataProto在PPO训练loop中的流动

```
ray_trainer.py 中的数据流:

Step 1: ActorRolloutRefWorker
  Input:  gen_batch = DataProto(batch={input_ids, attention_mask})
  Output: gen_batch_output = DataProto(batch={
    input_ids,           # prompt + response tokens
    responses,           # response tokens only
    log_probs,           # old log probabilities
    response_mask,       # mask for response tokens
  }, meta_info={
    uid,                 # unique identifier for each prompt
    temperature,         # sampling temperature
    rollout_n,           # number of completions per prompt (for GRPO)
  })

Step 2: Reward Computation
  Input:  gen_batch_output (split by coordinator)
  Output: gen_batch_output + batch.token_level_scores / token_level_rewards

Step 3: [PPO ONLY] CriticWorker.infer
  Input:  gen_batch_output (split by coordinator)
  Output: gen_batch_output + batch.values [bsz, resp_len]

Step 4: Advantage Computation
  PPO:  GAE(rewards, values, gamma, lam) → advantages [bsz, resp_len]
  GRPO: group_mean_std(rewards, uid) → advantages [bsz] (broadcast via mask)

Step 5: ActorWorker.update_policy
  Input:  DataProto(batch={input_ids, log_probs, advantages, response_mask})
  Output: train_stats (loss, reward, entropy, etc.)
```

## 6. Reconstructor机制深度分析

```
Reconstructor的核心价值: 解耦数据格式和处理逻辑

问题: 不同Worker需要不同数据格式
  Actor: 需要input_ids + log_probs + advantages
  Critic: 需要input_ids + attention_mask (只看上下文,不看response logits)
  Reward: 需要response tokens (只计算reward,不需要log_probs)

解决: DataProto作为统一容器 → Worker自己从DataProto提取所需字段
  → 不需要为每个Worker定制不同的数据格式!
  → 每个Worker只需: DataProto.batch[需要的key]

Reconstructor: 知道如何合并split后的结果
  → Coordinator不需要知道每个Worker内部的数据格式!
  → 只需: split → 发给Worker → 收回 → reconstructor.merge → 下一步

→ 整个数据流变成: DataProto → split → [Worker_i] → merge → DataProto → split → [Worker_j] → merge → ...
→ 每个步骤只关心: input DataProto → output DataProto (新增字段)
→ Worker间数据格式完全解耦!
```

## 7. 与rLLM数据流对比

| 维度 | verl DataProto | rLLM Tinker |
|------|---------------|-------------|
| **数据容器** | DataProto(batch+meta_info) | Tinker.Datum(prompt+response+reward+mask) |
| **跨Worker传输** | split→Ray RPC→merge | 无(in-process→直接Python调用) |
| **数据格式** | 统一BatchTensorDict | 自定义Datum结构 |
| **Reconstructor** | 需要(Ray远程合并) | 不需要(本地直接传递) |
| **meta_info深拷贝** | 是(每个Worker完整副本) | 否(本地共享) |
| **序列化开销** | Ray serialization(ZMQ/pickle) | 无(in-process→Python引用) |
| **适用规模** | 多GPU分布式 | 单GPU |

```
verl: DataProto → Ray → 远程Worker → serialization overhead
  → 多GPU必要: 通信开销 unavoidable
  → 灵活: 不同Worker可运行不同模型/不同GPU

rLLM Tinker: Datum → in-process → 直接Python调用
  → 单GPU最优: 无通信开销
  → 简单: 但无法跨GPU/跨节点
```

## 8. DataProto vs rLLM prefix-merge数据增强

```
verl单步RL:
  DataProto: input_ids=[prompt+response], log_probs, advantages
  → 1个response per prompt (或N个response for GRPO)

rLLM多步RL(prefix-merge):
  Datum: input_ids=[A0,obs1,A1,obs2,A2...], mask=[1,0,1,0,1...]
  → 多个observation+action交织 → 只action tokens(A)参与loss
  → mask使observation tokens的advantage=0 → loss只惩罚action选择
  → prefix-merge: 所有[A0,obs1,A1,obs2,A2]合并为1个序列 → 1次forward搞定!

  verl如果要支持多步: 需要修改DataProto → 新增response_mask/episode_ids
  → rLLM已在DataProto字段扩展: response_mask, episode_ids, trajectory_ids, group_roles, routed_experts
```

## 9. 列举张量的字节数算例

以下仅计算明确列出的五个张量，不估计整个DataProto或进程内存。旧稿把int32写成2字节且总和有误，原230KB及“开销完全可忽略”结论不再使用。Python3.12.13/torch2.14.1 CPU按下面shape与dtype实际构造五个empty张量，逐项`numel() * element_size()`核为491520字节，即480KiB；不包含对象头、allocator、元数据、图片或序列化副本。

```
限定算例：rollout_n=4, batch_size=4, seq_len=2048，shape均[16, 2048]

  input_ids:          int64，8字节/元素 = 262144字节 = 256KiB
  log_probs:          BF16，2字节/元素  =  65536字节 =  64KiB
  response_mask:      bool，1字节/元素  =  32768字节 =  32KiB
  token_level_rewards: BF16            =  65536字节 =  64KiB
  advantages:         BF16             =  65536字节 =  64KiB

列出张量合计：491520字节 = 480KiB
这不是总payload、RSS、共享内存或训练峰值；实际字段和dtype须按运行路径测量。
```

## 10. 关键设计洞察

```
1. 统一数据抽象 → 一行代码切换模型
   DataProto不关心模型类型 → 任何模型只需输出DataProto格式
   → Qwen/Llama/ChatGLM → 统一接口 → verl自动处理

2. split-merge-reconstruct → Ray分布式核心
   split: 1个batch→N个chunk → N个Worker并行处理
   merge: N个结果→1个batch → coordinator合并
   reconstructor: 随数据发送 → Worker不需要知道合并逻辑

3. 非tensor对象也要计入生命周期
   小元数据算例不能覆盖raw_prompt中的图片bytes/PIL对象或序列化副本

4. BatchTensorDict → batch维度约束
   所有tensor共享batch_dim → split/merge只需dim=0 → 极简

5. DataProto扩展 → rLLM兼容
   rLLM在DataProto字段上扩展(response_mask/episode_ids等)
   → 理论上rLLM可以无缝使用verl的数据格式
   → 但rLLM Tinker选择Datum → 更简单、in-process无split/merge需求

6. 限制: meta_info不支持split → 大meta_data场景需优化
   → 如长文本prompt → 可能需要分chunk传递meta
```

## 图片字节与解码对象的生命周期

**从返回字典删掉压缩bytes、断开解码流、释放整个多模态payload，是三件不同的事。** [#7391](https://github.com/verl-project/verl/issues/7391)的同报告者leooop-al已有非draft修复[#7400](https://github.com/verl-project/verl/pull/7400)，2026-10-24读取仍OPEN。固定head `dc721cfd3cbd4e18f49763f029d498519c74eb91`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际祖先`09ac37258ea66b0cb69b2738eec3074ea4e7261c`，ahead1/behind149，2文件25增1删。不能把目标base当成候选已包含的主线，也不另提相同修复。

### dataset 清理了什么

[原方法](https://github.com/verl-project/verl/blob/dc721cfd3cbd4e18f49763f029d498519c74eb91/verl/utils/dataset/rl_dataset.py)先浅拷贝单张image字典，再在副本中pop bytes；通过`with Image.open(BytesIO(...))`及`decoded_image.copy()`返回已解码的新图像。[Pillow生命周期文档](https://pillow.readthedocs.io/en/stable/reference/open_files.html)说明，open先读元数据，图像派生操作会load原像素；新图像不再关联原文件。因此只pop字段，不能保证旧PIL对象不再通过延迟加载持有流。

本机Python3.12.13、Pillow13.0.0，只执行完整原`_build_messages`方法，绑定到不含dataset构造过程的AST类载体；使用head原factory和原测试体/断言。路径1例、视频参数化3例、图片清理1例，共五例：

| 版本 | 路径与视频4例 | 图片清理原断言 | 新建PNG在首次读像素前的观察 |
| --- | --- | --- | --- |
| 祖先09ac3725 | 均通过 | bytes仍在输出，失败 | 源字典被加image，返回PIL仍有打开的fp |
| 候选dc721cfd | 均通过 | 通过 | 源字典不加image，返回PIL无打开的fp |
| 固定主线3c6d9b43 | 均通过 | 与祖先同样失败 | 与祖先同样保留字段/流 |

另一个同样2×2红色PNG的直接控制在bytes旁放path和min_pixels：三版都保留path/metadata且像素可读，候选也仍然选择解码bytes，并未优先读取path。源image字典仍保留bytes是有意保持输入不变；整个example并非不可变，原方法依然把prompt message的content原地改为列表。这里的“源不变”只指单张image字典，不等于深拷贝所有输入。

### 不把局部清理当内存验收

候选返回payload仍含解码后的PIL图像，并未改变AgentLoop、DataProto、reward/logging或actor dispatch。当前主线`__getitem__`把方法结果放到raw_prompt再pop顶层images列；pop顶层列并不删除嵌在raw_prompt内的对象。候选copy需要物化像素，可能改变局部峰值和解码时点，这是生命周期推断，不是已测RSS下降；不能从输出少一个字段推导完整训练内存一定下降。

#7391报告者的2节点×8GPU生产内存记录和path-only对照属于外部证据；path-only同时避免了这里的PIL物化，不能当作#7400 bytes清理后的同方案结果。对象引用重复、序列化副本、Ray共享内存和进程RSS也不是同一种统计。#7400作者报告5 passed/5 deselected，本机五例通过只针对源码提取的候选方法，不是其完整pytest、dataset/HF processor或训练验收。

本机未运行dataset构造/`__getitem__`、完整pytest/conftest、DataProto、Ray/actor dispatch、RSS或共享内存测量，也未执行模型、GPU/NPU；仅三版生产模块compile和上述真实Pillow控制。[完整脚本、hash与命令](../../diary/2026-10-24.md)留日志，不增加永久工具、硬件队列或竞争补丁。接续以独立需求/实质反馈为条件，不堆图片格式和序列化替身扩大结论。

---

Sources:
- [verl GitHub - DataProto implementation](https://github.com/volcengine/verl/blob/main/verl/utils/dp_rpc.py)
- [verl Documentation - DataProto Architecture](https://verl.readthedocs.io/en/latest/architecture/dataproto.html)
- [verl arXiv paper (2025)](https://arxiv.org/abs/2502.12025)
- notebook/frameworks/verl-ppo-vs-grpo-training-loop-comparison.md
- notebook/frameworks/rllm-gateway-backend-trainer-source-reading.md
