# verl DataProto 数据流架构源码阅读

2026-10-30接续：[TensorDict 升级的三个独立边界](#tensordict-升级的三个独立边界)按维护者新反馈、原始修复与官方分发元数据，分清轴信息序列化、非张量拼接和jagged批操作；升级依赖不等于自动修正所有位置张量的构造契约。

> 2026-06-15 | 源码: verl/utils/dp_rpc.py + verl/utils/tensor_dict.py + verl/trainer/ppo/ray_trainer.py
> 历史概览：旧版DataProto的split/merge设计。下方6月结构保留当时的学习视角，不代表2026-10当前实现的完整复核，也不提供跨Worker零拷贝保证。

2026-10-24补充：[图片字节与解码对象的生命周期](#图片字节与解码对象的生命周期)。原方法与真实Pillow对照确认dataset侧清理边界；原“数据开销完全可忽略”概括撤回，张量字节算例也已纠正，不能覆盖raw_prompt中的图片或序列化峰值。

2026-10-25补充：[多模态特殊 token 检查的 CPU 开销](#多模态特殊-token-检查的-cpu-开销)区分原检查方法、完整processor与训练吞吐；纯文本走多模态processor也可能付出标记计数成本，但不是所有纯文本入口都走此路径。2026-10-26接续[回移候选的绑定条件](#回移候选的绑定条件与评审前提)，不重跑旧微基准。

## TensorDict 升级的三个独立边界

TensorDict不同版本解决的不是同一个问题。2026-10-30，peterdsharpe在[verl #6082的新评论](https://github.com/verl-project/verl/pull/6082#issuecomment-6512340425)自述维护TensorDict，并报告在verl `2ab3529`上对0.10.0／0.14.3／0.15.3运行1,689项CPU测试：后两版仅三项失败。该数量、性能比较及整栈兼容性属于其外部报告；本机此次只核原始修复和发布记录，没有执行该套件。

| 官方发布边界 | 原始变更 | 对 verl 契约的含义 |
| --- | --- | --- |
| [0.12.2](https://github.com/pytorch/tensordict/releases/tag/v0.12.2)，2026-04-20 | [#1675](https://github.com/pytorch/tensordict/pull/1675)，合并 `9792dff3e089c6c9ee8c06655e63e41feb81438a` | consolidate记录已有`_ragged_idx`，反序列化时传回NestedTensor；保留已有轴，不推断轴的业务含义 |
| [0.14.2](https://github.com/pytorch/tensordict/releases/tag/v0.14.2)，2026-09-09 | [#1786](https://github.com/pytorch/tensordict/pull/1786)，合并 `906274646e07e55b05c4f51fac0cc91f0beb8cc1` | 非张量cat改为保留各输入值，不再默默重复首值；依赖首份批元数据的调用方须自己表达该策略 |
| [0.15.3](https://github.com/pytorch/tensordict/releases/tag/v0.15.3)，2026-10-24 | [#3113](https://github.com/pytorch/tensordict/pull/3113)，合并 `392ea01b0061b93f406398e48d6e429d878f2052` | 增加首batch维上的jagged索引、切片、split／chunk／cat等；不是所有nested操作或全部布局都获得支持 |

前两项已直接读修复diff。#1675在叶元数据末尾追加ragged轴，读取时仍兼容旧五字段格式；旧数据若未保存轴，读取器并不能凭空恢复它。#1786增加`_cat_non_tensor`，先按输入batch维unbind收集值，再stack，避免旧空内部TensorDict使后续值丢失。这解释了[本机0.11.0仍保留首份](#本机版本没有触发报告中的失败)为何不否定0.15.1作者报告，也将“从0.11开始”的宽泛阈值收窄为有原始变更和发布记录支持的0.14.2边界；不是本机跨版本二分结果。

轴保存与轴构造尤其不能混用。#1675能保存传入的轴信息，却不把等长位置样本自动识别成mRoPE序列轴；[队列原packing后被强制retag的四组控制](#队列打包与位置张量的轴语义)属于另一条条件路径。维护者把#7899／#6851列为升级收益，不足以据此宣布所有TQ写入布局闭环，亦不能自动删除当前消费者的兼容代码。

### 小版本的依赖也须分别核对

2026-10-30读取PyPI官方JSON：[0.15.1](https://pypi.org/pypi/tensordict/0.15.1/json)声明`torch>=2.13.0a0`，[0.15.3](https://pypi.org/pypi/tensordict/0.15.3/json)改为`torch>=2.11.0a0`，且后者基础依赖不再含orjson。两者均有`py3-none-any`轮，但都不满足本机Torch2.9.1；不能把某个0.15小版本的Torch下限推广为整个0.15系列，也不能因纯Python轮就忽略Torch与设备包约束。verl自己使用orjson时应独立声明依赖，不能把它偶然由其他库安装当成自身依赖契约。

所核[verl #6082](https://github.com/verl-project/verl/pull/6082)仍OPEN，head `13a7c067d62ca7a0faf995cc9d401b1d87885aea`、base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，三提交16文件。候选把多处依赖改为无上界`>=0.12.2`；其`maybe_fix_3d_position_ids`实际上保留条件及版本assert，仅注释原赋值，并非整个helper已删除。维护者提出的首份元数据处理、直接依赖orjson、旧测试预期调整和NPU Torch约束，属于升级时需协调的不同事项；不是该旧候选已经完成的兼容修复或本机安装／NPU验收。不改环境、不套候选，也不为本次只读证据更新新增M5任务。

## 批元数据与逐样本字段的拼接

**保留首份批元数据与拼接逐样本字段是两种不同契约；保留首个rank的内存字典也不等于求全局峰值。** 2026-10-30只读学习jialei777的OPEN／非draft [#8453](https://github.com/verl-project/verl/pull/8453)，head `1ffe98bf14949f0ac1cc000facf056ec0b7808d9`、base `a08ea02a784d7d4982b1d81df36d188a476192bb`，一提交两文件37增2删。作者报告TensorDict0.15.1将不同的`metrics`拼为逐行列表，SFT调用方仍按`metrics["loss"]`读取而失败；其TPU CI是外部报告，不计本机验证。

[候选helper](https://github.com/verl-project/verl/blob/1ffe98bf14949f0ac1cc000facf056ec0b7808d9/verl/utils/tensordict_utils.py)在普通／nested两次`TensorDict.cat`之后检查首输入的顶层键：首值为NonTensorData、输出却不再是该类型时，用首值的数据重新包装。它不聚合rank指标，也不覆盖原本就是NonTensorStack的逐样本列表；空batch维仍直接返回首输入。这里的修复目标是返回类型与既有首份语义，不是监控量的全局归约。

### 本机版本没有触发报告中的失败

本机Python3.12.13／Torch2.9.1，独立安装[TensorDict0.11.0分发](https://pypi.org/project/tensordict/0.11.0/)，未改原Torch环境。分别执行两版**完整原utils模块**，接候选[两个原测试体](https://github.com/verl-project/verl/blob/1ffe98bf14949f0ac1cc000facf056ec0b7808d9/tests/test_protocol_v2_on_cpu.py)：既有`test_concat_tensordict`和新增`test_concat_tensordict_keeps_the_first_metadata`，均未改函数体。两版各2通过，直接调用而非原生pytest收集；旧测试还检查逐样本labels拼接及输入不变。

| 实际CPU容器条件 | 基线 | 候选 |
|---|---|---|
| 普通Tensor，两rank的peak为17.9／18.3 | batch4、NonTensorData、首份17.9 | 相同 |
| jagged输入，每rank样本长3／5 | batch4、NonTensorData、首份17.9 | 相同 |

所以本机没有“旧失败、新通过”证据。PR正文与注释将变化起点写为“从0.11开始”，**不能仅凭该版本号认定所有组合都会失败**；这里0.11.0＋Torch2.9.1已经保留首份。初步直接`TensorDict.cat`探针及capture开关True／False也都返回首份，不据此断言0.15.1报告错误，或推断精确变更版本。

### 目标新版本仍有依赖差额

[0.15.1分发](https://pypi.org/project/tensordict/0.15.1/)的METADATA声明`torch>=2.13.0a0`，不满足本机Torch2.9.1。最初无依赖安装后导入止于缺pyvers，未形成有效容器测试；之后仅对0.11目录补齐轻依赖，没有为0.15升级Torch或伪造导入。缓存0.15.1的`tensorclass.py`可静态读到专门的`_cat_non_tensor`：非一致值走unbind再stack，与0.11没有该方法的源码不同；这只是源码解释，不是0.15运行验收。

[脚本](../../experiments/verl-first-metadata/verify.py)与[结果](../../experiments/verl-first-metadata/result.json)保存两版生产文件、原测试和两分发源码／元数据哈希、实际轻依赖版本。命令：`PYTHONPATH=/path/to/td011 python3.12 -B experiments/verl-first-metadata/verify.py /path/to/verl-git /path/to/td0151`；第一目录需有0.11.0及结果中固定轻依赖，第二目录只供读取0.15.1缓存，不得放进PYTHONPATH。固定Git对象预缓存后运行不联网。两次新进程JSON一致，真实TensorDict与CPU张量，无批数据／worker／Ray、TPU或训练验收。

本机复现仍以该版本条件下的否定结果收束；后来维护者反馈与原始变更见[三个升级边界](#tensordict-升级的三个独立边界)。已有作者修复，不代改其阈值或发布反馈；匹配0.15依赖的CPU复验仍未做，不属于GPU访问阻塞，也不新排M5算力队列或调整其既定环境。社区及既有PR-head写入仍逐项待用户授权。

## 队列打包与位置张量的轴语义

**改ragged轴标签不等于重排values和offsets；不报错也不等于样本边界正确。** [#7899](https://github.com/verl-project/verl/issues/7899)已有iceflysnow的OPEN／非draft [#7901](https://github.com/verl-project/verl/pull/7901)，2026-10-30核head2310b690。10-29的[新社区反馈](https://github.com/verl-project/verl/issues/7899#issuecomment-6504217618)补充TransferQueue自身也会由逐样本张量构造jagged batch，因此不能把只修verl padding写入点的#8413当作所有写入来源已闭环。此边界已有作者及社区方案，不另认领修复。

固定[TransferQueue 5c15a2c8的`_pack_field_values`](https://github.com/Ascend/TransferQueue/blob/5c15a2c8d65f05a52514830da6de4c5ec4336c1c/transfer_queue/storage/managers/simple_storage_manager.py)，纯Tensor列表分支直接调用`as_nested_tensor(..., layout=torch.jagged)`；[另一merge方法](https://github.com/Ascend/TransferQueue/blob/5c15a2c8d65f05a52514830da6de4c5ec4336c1c/transfer_queue/storage/managers/base.py)也有同类构造，但本机只读它、不执行。固定[verl main a08ea02a](https://github.com/verl-project/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/utils/tensordict_utils.py)仍对3D nested position_ids直接设`_ragged_idx=2`；同版本padding已改显式token轴构造，未消除队列这条独立生产路径。

[CPU复现](../../experiments/verl-tq-position-layout/verify.py)AST提取完整原packing及retag方法，后者接plain dict数据适配，非真实TensorDict；不构造队列或替换方法体。Python3.12.13／torch2.9.1、单线程、C=4，样本含不同的整数内容而非全零；[结果与两原文件hash](../../experiments/verl-tq-position-layout/result.json)如下：

| 样本长度 | 原packing布局 values／offsets／ragged轴 | 原retag后unbind | 显式序列轴构造参考 |
| --- | --- | --- | --- |
| [7,7,7] | (12,7)／[0,4,8,12]／1 | RuntimeError，本机为Expected cond错误 | (4,21)／[0,7,14,21]／2，逐样本精确相等 |
| [5,7,9] | (4,21)／[0,5,12,21]／2 | 正确 | 同布局，精确相等 |
| [7] | (4,7)／[0,4]／1 | split_with_sizes RuntimeError | (4,7)／[0,7]／2，精确相等 |
| [8,8] | (8,8)／[0,4,8]／1 | **无异常，但返回两个(8,4)，并非原(4,8)样本** | (4,16)／[0,8,16]／2，精确相等 |

所有原packing输出在retag**之前**都能unbind回原样本；它们是通用jagged表示的自洽结果，不应把“不是mRoPE要求的序列轴”直接叫作Torch内存损坏。原retag既没有移动数据也没有更新offsets，公开shape还保留原来的ragged位置。方形例中`offsets[-1]==values.shape[1]`恰好成立，仍切错语义轴；对这里的紧凑布局，只看长度和轴尺寸一致仍不足以认证样本身份。必须将序列长度／组件数等语义契约与表示一起核对。

参考用原samples沿末轴cat、按已知长度构造offsets，再调用原生`nested_tensor_from_jagged(..., jagged_dim=2)`；四组均经同一原retag后精确往返。它只说明**显式构造契约**，不是我方修复补丁、#7901验收或生产输入的自动推断器。[10-27社区替代方案](https://github.com/verl-project/verl/issues/7899#issuecomment-6435660247)以input_ids offsets约束布局，#7901则另有形状签名／方形判断；两者的完整正确性不由本轮四小输入裁决。社区63项／Qwen3.5 e2e、原报告35B训练与#7901测试均不计本机通过。

```bash
python3.12 -B experiments/verl-tq-position-layout/verify.py /path/to/verl-git /path/to/simple_storage_manager.py
```

第二输入为上述固定TQ提交的完整原文件，脚本强校验Git blob `d14e67c3c2f38bd4e829b78bac354d20924a454a`；verl对象须已缓存，运行不联网。只有纯Tensor分支，未为NonTensorStack fallback造替身；异常文字随torch版本不同，不能要求所有版本都逐字报split_with_sizes。未导入TQ／verl全包、构造服务、pickle、microbatch、trainer、模型或设备；无GPU／性能／训练验收。本项收束，后续只跟实质版本／反馈，不替作者改方案或新增M5任务；所有上游及既有PR-head写入仍逐项待用户授权。

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

### 回移候选的绑定条件与评审前提

2026-10-26已有同报告者WinstonLiyt的开放非draft [verl #8375](https://github.com/verl-project/verl/pull/8375)，head `240f7f0ae09286429fbd81d21a2a403e323665fb`，base／main仍为 `3c6d9b43b67c74ec343d77874cd8fa6e841523da`；两文件78增1删。候选不升级依赖、不把processor输出改成list，而是在检查时临时转换IDs、继续调用上游计数方法。安装位置在 `hf_processor`：版本低于5.15.0且类继承原基类方法时才绑定到实例；自定义类方法不覆盖，5.15.0路径不安装。这里的5.15.0是分支控制，不代表verl当前依赖范围已允许升级。

[行内机器人反馈](https://github.com/verl-project/verl/pull/8375#discussion_r4372894021)称受支持最低版本5.5.3缺少该基类方法，会令工厂捕获AttributeError并返回None。官方[v5.5.3 tag](https://github.com/huggingface/transformers/tree/v5.5.3)解引用到 `c6c8503869367af938666810e01a71866ca4fe93`，其[原文件](https://github.com/huggingface/transformers/blob/c6c8503869367af938666810e01a71866ca4fe93/src/transformers/processing_utils.py#L1995)已直接定义 `_check_special_mm_tokens`；5.12.1与5.15.0也有。因此“5.5.3缺方法”这一故障前提不成立，不能根据该评审再造缺失API或修复补丁。这不证明所有第三方processor均兼容。

[本机控制脚本](../../experiments/verl-processor-binding/verify.py)从两版verl提取完整原 `hf_processor`，用原版本比较函数、模拟包版本、fake AutoProcessor／AutoConfig加载器及按官方方法构造的类壳执行四情形：5.5.3继承、5.12.1继承、5.12.1覆盖、5.15.0继承。两版共8次工厂调用均返回原夹具对象、无告警；候选只在前两情形绑定wrapper，父版均不绑定，后两情形保持原方法。[结果](../../experiments/verl-processor-binding/result.json)保留固定源码SHA256。这不是安装三组Transformers或执行真实Mllama／Qwen processor，也不是候选原测试套件；候选三项原测试直接调用helper，未覆盖工厂安装点，机器人关于该覆盖缺口的提示与错误的缺方法前提应分开判断。

作者另报Qwen3.5-2B完整processor的131,073-token耗时0.800→0.292秒及22项测试通过，均为作者环境结果，不替换上节M1旧版原方法计时。当前只补新绑定边界与评审核验，不改pin、安装依赖、下载模型、写竞争shim或新增M5队列；后续仅在候选实质更新或实际兼容需求出现时接续。

复用：固定公开对象已取回的verl checkout，使用Python3.12、packaging及已认证gh执行 `python experiments/verl-processor-binding/verify.py /absolute/path/to/verl`；脚本只读三个固定官方源码文件及本地Git对象，结果输出到stdout。

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

## SFT 验证的尾批与 token 分母

**保留验证尾批、消除采样重复、使用正确的指标分母，是三个条件。** [#7464](https://github.com/verl-project/verl/issues/7464)报告两条SFT入口将验证集按训练大batch且drop_last=True读取，小集合可能零batch，随后空均值成为NaN。已有同报告者的[#7467](https://github.com/verl-project/verl/pull/7467)，本项目只核现有方案，不另写修复。2026-10-25读取OPEN、非draft、未合并，head `87dc0d0071d3868ae86bcbf245ba9fbd88e2927a`；API目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际共同祖先 `a9f2985159536a607211dcac730d3f5d55028950`，5文件625增38删。

### 不丢尾批不等于没有重复

旧主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的SFT验证sampler和loader均drop_last=True。原CPU测试的200行、batch256、单副本显式rank得到零batch；独立torch空张量mean为NaN。候选的batch来自正数val_batch_size，否则全验证集；micro_batch_size_per_gpu仍用于引擎切分，不成为默认loader batch。null、0、负数都按全量处理。

单改为 DistributedSampler(drop_last=False) 会补重复索引，而非自动产生“无权重的空位”。真实CPU对照以10行、4个显式rank、shuffle=False，每行i的loss-mask token数为i+1，得到：

| 构造 | 四个rank的行索引／补位 | 有效token总数 |
| --- | --- | --- |
| 普通DistributedSampler | [0,4,8]、[1,5,9]、[2,6,0]、[3,7,1] | 58，头两行被重复计数 |
| 原候选LossMaskedPaddedDataset及loader | 先将数据集补到12槽位，补位行loss mask为0 | 55，与10个真实行一致 |

这保证的是有效loss计数，不是补位行免于前向或没有计算／显存开销。SPMD的loader已按真实DP分片，后续只需满足每rank静态micro-batch倍数；Ray driver持整批，送worker前还要满足真实DP分派倍数。候选将这两个补位位置分开；静态模式的行倍数为DP×micro_batch_size_per_gpu，动态模式只要求DP。这里不据一个helper通过认证真实worker分派、动态token打包或任意并行布局。

### token 均值不能按样本数重新平均

原 [sft_loss](https://github.com/verl-project/verl/blob/87dc0d0071d3868ae86bcbf245ba9fbd88e2927a/verl/workers/utils/losses.py#L28)使用loss-mask token总数作分母；FSDP路径会跨DP求和batch_num_tokens。因此，在引擎返回该批全局每token NLL的前提下，跨batch应算Σ(lossᵦ×tokensᵦ)/Σtokensᵦ，而不是样本均值或batch均值。

原测试的显式NLL夹具：三个batch分别20／20／1000个有效token，批内每token NLL为2／2／1，样本数2／2／1。原reduce函数得token加权1080/1040≈1.0384615；误用样本数得1.8，等权平均batch则约1.6666667。这是已给定NLL的汇总对照，不是模型输出或验证质量测量。原SPMD接线用本地token数与已全局归约的每批loss构造分子／分母，再作SUM；Ray端使用整批token数。局部数学不代替实际collective验收。

### 本机执行的范围

Python3.12.13／torch2.14.1、torchdata0.11.0、TensorDict0.15.2、pytest9.1.1，复用现有CPU环境、无安装。完整原TensorDict工具／collator／sft_val_utils和原测试模块保持字节不改，仅空父包隔离框架初始化；完整原prepare_micro_batches函数经AST隔离，原测试只走静态切分，避免导入设备发现模块。原16测试体直接调用两次全过，覆盖200行尾批、1319行／8个逻辑rank、零mask补位、静态整除、分母与汇总；这些rank在单进程依次枚举，没有进程组或真实设备通信。

原测试名中的“end-to-end”仅串联本地加载器与数学；其_per_token_nll是作者明确的stand-in，没有执行模型、实际sft_loss的因果mask移位、完整trainer或engine。因此不记为原生pytest／训练／收敛验收，也不凭所有测试通过宣布任意padding模式与后端正确。七份完整源码快照另过AST／内存compile；候选正文开头仍有旧“sample-weighted”概述，当前第三提交的生产代码／原测试已改为token权重，不能沿用早轮LGTM或旧摘要。

输入、命令及hash见[当日日志](../../diary/2026-10-25.md)。这是CPU数据流学习，不新增M5硬件队列、不升级／应用候选或补写竞争方案，不改已有专项；任何社区写操作仍待用户逐项明确同意。

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

## 多语言预分词与类路由

**文本能往返解码、训练与rollout的IDs相同，都不能独自证明与checkpoint分词约定一致。** [verl #8030](https://github.com/verl-project/verl/issues/8030)关联Qwen新版本的组合标记问题。固定verl main `3c6d9b43`仍钉Transformers5.12.1、SGLang0.5.20；原 `hf_tokenizer`调用AutoTokenizer后主要校正pad，没有自建多语言regex修复。本轮不改pin或运行引擎。

### 已合并修复选择类而非修改 Qwen2 默认规则

2026-10-25核对官方 [Transformers #49066](https://github.com/huggingface/transformers/issues/49066)已关闭；最初读取kwargs里regex的 [#49077](https://github.com/huggingface/transformers/pull/49077)关闭**未合并**，不能当作当前修复。实际已合并的是 [#49116](https://github.com/huggingface/transformers/pull/49116)，merge `a4fe1d5bde6fd245899d9b0486d2dc25b76f90c7`：为部分Qwen model_type补注册映射，并加入“Hub声明类不正确”的例外集合，使AutoTokenizer可选择Qwen3_5Tokenizer。它没有把所有Qwen2Tokenizer的默认regex改成新规则，避免影响旧checkpoint的约定。

固定TF5.12.1 `ddb849abe009d1089e6c691bfc897f27211c663c`的Qwen2默认regex不含Unicode `\p{M}`；本机TF5.20.2的同一默认值仍相同。refined规则把组合标记纳入字母连续段并从符号段排除。类别路由、实例内预分词规则、BPE词表／merge与最终IDs是不同层，不能只比JSON是否相同或token数量。

### 真实 CPU tokenizer 的局部对照

Python3.12.13／Transformers5.20.2／tokenizers0.23.3，离线使用真实Rust预分词／BPE与AutoTokenizer。本地toy BPE只在以下四句上训练、vocab_size400、ByteLevel初始字母表、NFC；不含Qwen checkpoint文件或权重，也不是预训练词表复现。先按refined规则保存tokenizer.json，配置声明 `tokenizer_class=Qwen2Tokenizer`、`model_type=qwen3_5`。

| 输入 | 旧／refined预分词片段数 | 直接Qwen2与原toy backend IDs相同 |
| --- | --- | --- |
| สวัสดีครับ | 4／1 | 否 |
| नमस्ते दुनिया | 7／2 | 否 |
| مُحَمَّد | 5／1 | 否 |
| hello world | 2／2 | 是 |

表中是**预分词片段数，不是模型token数**。四句均能在两路径decode回原文；组合标记三句的完整ID列表却不同。直接Qwen2即使给refined kwarg仍保留旧默认，save_pretrained会把其旧pattern序列化；原toy文件继续保留refined。这只证明本地对象与保存路径，不冒充报告中的公开衍生checkpoint污染范围。

同一fixture的AutoTokenizer真实加载结果为Qwen3_5Tokenizer，四句IDs均与原toy backend一致，保留refined规则。说明在**本机版本及已知qwen3_5 model_type、无auto_map的条件下，显式Qwen2声明本身并不会阻止类纠正**。官方issue末条评论称5.20.1某衍生checkpoint仍受影响，本轮未读取该checkpoint或运行5.20.1，不能反驳其具体结果；也不能据该评论泛化“显式类名总是优先”。应核实际model_type、auto_map、最终类及ID行为，而不是只看类名字符串。

官方 [#49114](https://github.com/huggingface/transformers/pull/49114)的regex透传与 [#49341](https://github.com/huggingface/transformers/pull/49341)的行为告警仍OPEN／未合并，前者不当现行实现，后者不当全部Unicode输入的完备检测。没有重复开发或发表反馈。

[脚本与hash](../../diary/2026-10-25.md)保留版本／fixture条件。本轮没有重装5.12.1、复跑上游Hub集成用例、读取公共checkpoint tokenizer资产、运行actor／rollout或测训练质量；toy BPE结果不能量化多语言能力下降或推导统一升级安全。若未来排查真实工作，应同时核checkpoint原IDs与实际两端IDs；只修训练进程中的对象不自动修复rollout独立进程。本题知识收束，不新增M5算力任务或改专项。

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
