# verl Checkpoint Management 源码级深度阅读

2026-10-30补[多角色异步保存的发布与清理](#多角色异步保存的发布与清理)：已有#8056将角色确认、tracker替换和保留清理分开；八组原协调器标准库控制确认清理失败时新指针可能已发布、pending仍留待重试。不是原OmegaConf测试、真实角色写入／恢复或作者H20验收。

2026-10-30接续：[随机状态与恢复顺序](#随机状态与恢复顺序)用四个原生CPU重启对照区分重设seed、独立生成器和构造顺序；模型与优化器状态相同仍不足以保证下一步一致。这是普通PyTorch小模型控制，不是verl恢复缺陷或完整manager验收。

> 2026-06-15 | 源码: verl/utils/checkpoint/ + verl/checkpoint_engine/ + verl/model_merger/ + verl/trainer/ppo/ray_trainer.py
> 核心: 2层架构(CheckpointManager→磁盘持久化+CheckpointEngineManager→实时权重同步); FSDP每rank独立shard+HF export(rank0 gather); Megatron dist_ckpt+async save; CheckpointEngineManager 5后端(naive/NCCL/NIXL/Kimi/Mooncake/HCCL); FSDPModelMerger离线合并→HF; GRPO只存actor+dataloader→省critic

2026-10-22 补充：[保留目录与完整恢复的边界](#保留目录与完整恢复的边界)固定于 verl `1876b06d`。`max_ckpt_to_keep=1` 保存前保留旧目录，成功登记后才删旧；目录保留、保存完成通知与可恢复性是不同保证。下方早期架构段落的版本和行号不自动更新为本次验证范围。

2026-10-24补充：[异步保存能力与恢复指针](#异步保存能力与恢复指针)核对#7955及旧DCP候选#4311；原查找函数的一例测试与五组真实文件系统控制说明“找到目录”并不验收恢复内容，异步写入能力也不自动提供tracker发布。

2026-10-25补充：[Megatron 调度进度与超参数恢复](#megatron-调度进度与超参数恢复)用官方调度器和真实CPU optimizer区分进度、超参数来源与参数组覆盖；六组控制不等于完整分布式checkpoint恢复。

## 随机状态与恢复顺序

**恢复权重与优化器之后，还要恢复随机流的位置；初始seed不表示中断位置。** 以下是独立的普通PyTorch CPU小模型实验，不是对verl／UniRL保存器的故障复现。[优化器参数身份](../fundamentals/adamw-optimizer-deep-dive.md#恢复时的参数身份)解决状态接到哪个参数，这里固定参数顺序，只改变随机状态与恢复时机。

Python3.12.13／Torch2.9.1（commit `5811a8d7da873dd699ff6687092c225caffcf1bb`）／NumPy2.0.2，单CPU线程、FP64。模型为Linear(3,4)→Tanh→Dropout(0.5)→Linear(4,1)，AdamW用lr=.01、weight_decay=.01，foreach／fused／capturable关闭。先真实更新两步，深拷贝模型、优化器和四份随机状态到内存；连续运行第三至第五步作为参考。每条重启轨迹都先断言模型参数与优化器全部状态精确恢复，再执行同样三步。

四个随机源各有用途：Python模块默认Random产生输入增益，NumPy模块全局RandomState产生目标噪声，Torch CPU默认生成器供初始化／Dropout使用，独立`torch.Generator(device="cpu")`决定固定8行数据池的四行子集。每步重新randperm，不是DataLoader、持久游标或多进程预取模拟。

| 重启方式 | 第三步前随机状态与checkpoint的关系 | 第三步loss | 后续三步与连续运行 |
| --- | --- | --- | --- |
| 只重设四个初始seed，再加载模型／optimizer | 四份都不在中断位置 | 0.8555002523 | 不同 |
| 构造、加载后恢复三份全局状态，漏独立采样生成器 | Python／NumPy／Torch CPU相同，采样不同 | 1.5131241491 | 不同 |
| 先恢复全部随机状态，再构造和加载模型 | Python／NumPy／采样相同，Torch CPU已被初始化推进 | 0.5111180357 | 不同 |
| 先构造并加载模型／optimizer，再恢复全部四份状态 | 四份相同 | 2.9594540384 | 本例逐步一致 |

参考第三步取行`[5,1,3,7]`；漏独立生成器时取`[3,6,0,4]`，而该步输入增益与噪声仍精确匹配参考。构造前恢复时行、增益和噪声都匹配，但随机初始化已推进默认Torch状态；加载checkpoint覆盖了初始化权重，却没有倒回这个副作用。最后一组的三步输入、输出、loss及梯度／参数摘要一致，最终模型参数、AdamW状态逐Tensor精确相等，四随机流的最终摘要也一致。前三组最终模型和optimizer均不同；结果不是仅凭一次loss相近判断恢复成功。

### 各生成器的所有权要单独列清

安装版本的[torch/random.py](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/random.py)明确：`get_rng_state()`／`set_rng_state()`仅操作CPU默认生成器；独立Generator有自己的get_state／set_state。Python与NumPy模块的状态也不自动包含用户另建的Random、default_rng或其他生成器。这里实际覆盖的是上述四个实例，不能把四份状态当作任意训练程序的完整清单。[Linear初始化源码](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/nn/modules/linear.py)中的kaiming_uniform／uniform解释构造阶段为何消耗随机数。

本例可复用的检查方式是：列清随机源，记录下一步前的状态，完成可能消耗它们的重建／加载后再恢复，并比较下一批样本和下一次更新。它不是“全局最后一行总能恢复”的通用方案：有初始化后仍会消耗随机数的后台任务、worker预取、设备独立流、不同调用顺序或算子非确定性时，还需对应协议。这里未测试磁盘序列化、调度器、AMP、DataLoader、FSDP／DCP／ZeRO、分布式rank或任何设备，也不承诺跨Torch版本／平台bitwise一致；不据此新增框架修复或M5训练队列。

[完整脚本](../../experiments/checkpoint-rng-order/verify.py)与[结果](../../experiments/checkpoint-rng-order/result.json)保存每步数据索引、增益、噪声、输出、loss和梯度／参数摘要，以及四份随机状态摘要、四个已安装Torch方法的源码SHA256。两个最终新进程JSON完全相同，checkpoint原快照也逐项保持不变。运行不联网、不装依赖，不调用设备发现或全设备seed接口：

```bash
python3.12 -B experiments/checkpoint-rng-order/verify.py
```

## 自定义数据源的保存与恢复边界

**模型checkpoint可保存、loader有恢复方法、数据状态文件有效，是三个不同条件。** 2026-10-30只读Mecoli1219的 [#8454](https://github.com/verl-project/verl/pull/8454)，固定head `c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a`、base `a08ea02a784d7d4982b1d81df36d188a476192bb`，一提交13文件940增50删。初核为draft；06:12Z后复核已OPEN／非draft，新增[bot COMMENTED review](https://github.com/verl-project/verl/pull/8454#pullrequestreview-5708696150)，不是维护者批准。本节限于保存、恢复与数据耗尽的checkpoint边界，不评审全部流式训练接线。

[桥接接口](https://github.com/verl-project/verl/blob/c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a/verl/utils/dataset/dataloader.py)允许训练loader只实现迭代；若提供任一checkpoint方法，构造检查要求`state_dict`与`load_state_dict`都可调用。按[保存源码](https://github.com/verl-project/verl/blob/c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a/verl/trainer/ppo/v1/trainer_base.py)，无状态loader仍可走actor／critic保存，跳过数据状态并清除同目录旧data.pt；这不是完整训练可重启承诺。该保存路径、Hydra构造和清理只做源码核对，未在本轮执行。

### 失败位置取决于缺失条件

本机抽取完整原`_load_checkpoint`，用六个自设计条件核对返回、错误、step与调用边界。临时目录内的data.pt仅由本脚本写入简单字典／列表，使用真实torch.save/load；actor／critic及loader状态接口、latest查找和配置是明确替身。

| loader与checkpoint条件 | 原方法结果 | actor／critic替身是否已收到加载调用 |
|---|---|---|
| 无恢复接口，auto找到checkpoint | ValueError，step0 | 否 |
| 无恢复接口，显式resume_path | ValueError，step0 | 否 |
| 无恢复接口，auto找不到checkpoint | 正常新训练入口，step0 | 否 |
| 有恢复接口，缺data.pt | FileNotFoundError，step7 | 是 |
| 有恢复接口，data.pt存的是列表 | TypeError，step7 | 是 |
| 有恢复接口，data.pt为字典cursor=3 | loader接口收到该字典，step7 | 是 |

接口缺失的拒绝在设置恢复step／加载权重前；内容缺失或类型错误则在这两步之后。**明确报错不等于所有条件都已预检，也不证明恢复具有回滚能力。** 后三行的“收到调用”不是实际模型状态已改变；最后一行只验字典透传，不证明下一批样本身份、远端流offset或优化器／RNG一致。

[原CPU测试](https://github.com/verl-project/verl/blob/c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a/tests/utils/dataset/test_custom_dataloader_on_cpu.py)已有无状态保存／恢复前拒绝、stateful游标回放及缺文件等用例。本轮读取但未运行这些原测试或其Hydra／trainer fixture，不把作者测试和本机控制合并计数。完整trainer、bridge与原测试三文件AST／compile通过；只执行了一个原生产方法。

[脚本](../../experiments/verl-custom-loader-resume/verify.py)与[结果](../../experiments/verl-custom-loader-resume/result.json)保存固定输入SHA256。Python3.12.13／Torch2.9.1，`python3.12 -B experiments/verl-custom-loader-resume/verify.py /path/to/verl-git`，Git对象预缓存后运行不联网。六条件通过、两新进程JSON相同；没有模型权重／optimizer加载、真实loader恢复、Ray／TQ／设备或训练验收。用户数据或外部pickle不应交给这个临时夹具。

### 提前耗尽与尾部保存

同head的[新bot反馈](https://github.com/verl-project/verl/pull/8454#discussion_r4420932965)指出step-based reader提前结束会漏尾部保存。固定[trainer_base](https://github.com/verl-project/verl/blob/c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a/verl/trainer/ppo/v1/trainer_base.py#L494)支持以下**静态控制流结论**：显式total_training_steps会令steps_per_epoch=None，即使loader有len也如此；`_fetch_one_gen_batch`此时将StopIteration转为DataLoaderExhausted，不重绕数据。`step`先在prepare_step取整批，再进入局部更新循环；prepare阶段的该异常能到`_fit`的break，跳过本轮保存／验证。循环尾部只有on_train_end与dump关闭，基础on_train_end无操作；[同步子类](https://github.com/verl-project/verl/blob/c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a/verl/trainer/ppo/v1/trainer_sync.py)也未补存。

例如**假设**总步数5、save_freq=2，已经完整完成1–3步，准备第4步时耗尽，源码只触发周期保存2，不触发补存3；若第3步不在验证周期，也没有最终补验。完成步数为零、最后完整步已在周期保存、保存或验证被显式关闭，均须分别看待，不能写成“所有提前结束都丢最后一步”。这是路径推演，不是真实权重丢失记录。

更新阶段不同：`_step_once`／refill中出现DataLoaderExhausted会被`step`转为RuntimeError，而非走正常break；`fit`的finally仍尝试关闭reader，但关闭资源不等于保存或回滚。原测试`test_step_mode_ignores_epochs_and_stops_cleanly_without_rewinding`替换了step，检验消费数量／关闭；默认save_freq=-1，没有验收正保存周期下的尾部保存。另有partial-update转RuntimeError测试；这些测试本轮只阅读，未重跑或改写。

bot的补存草图仅暂时把global_steps改为最后完整步。**编号回退不等于数据状态回退**：`_next_train_batch`逐个取gen_batch_size块，凑齐后才提交整批；若取到部分块后耗尽，reader可能已消费未训练数据，而`_save_checkpoint`会调用当下state_dict。这是补存方案须处理的源码风险推断，不能仅靠保存目录叫global_step_3就认证样本连续性，不自动采用该草图。

另核[文档反馈](https://github.com/verl-project/verl/pull/8454#discussion_r4420932952)：同head的docs/index.rst确有`advance/custom_dataloader`，Git文档树无对应文件；这是悬空入口的静态确认，不是本机Sphinx构建失败。trainer_base、同步子类与原CPU测试三完整Python输入编译通过，AST断言确认异常分支和继承边界；未导入trainer、执行生产方法或跑训练。输入SHA256：

```text
trainer_base.py   cf084ad149e953fc7da73a5d5073e1abd2c986f77bbc14e9b70df662b544781c
trainer_sync.py   bc805f2990910f29e11b45c43a0687eae645cee375ab6b9a7211650022f00186
原CPU测试        be6fb6e2155e4c2424a2e49fa36f6f3cb3d6c2b700015c874a7e672ee778db64
docs/index.rst   902010d23c496315c4fbe750141679e1be44ef5f6a79fdc1e8121b925757bea9
```

已有作者推进候选；原恢复条件验证不扩矩阵，新增反馈只作上述静态核对，不改其接口、重复认领、发布评论或新排M5训练任务；社区与既有PR-head写入仍逐项待用户明确授权。

## 配置字段可覆盖与对象默认值

**dataclass里有字段，不代表Hydra普通命令行覆盖能找到这个键。** 2026-10-26核对[#7387](https://github.com/verl-project/verl/issues/7387)及两个OPEN/non-draft候选：[#7388](https://github.com/verl-project/verl/pull/7388)只补actor的`save_lora_only`；[#7426](https://github.com/verl-project/verl/pull/7426)同时补actor、critic、SFT，并更新四份生成YAML、增加字段一致性测试。后者作者已在#7388评论中说明重叠、提出按维护者选择拆分；不另认领或提交第三份修复。

固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的[基础配置](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/config/config.py#L24)含`save_lora_only=False`、`strict=True`及`async_save=False`。但原YAML缺actor的第一项、critic的前两项、SFT的三项。`_target_`只是后续实例化的目标标识，不会在Hydra组合阶段自动把类的默认字段补进普通YAML节点；缺键发生在构造对象之前。

### 真实 Hydra 组合对照

[复现脚本](../../experiments/verl-checkpoint-config/verify.py)从固定Git对象抽取各版完整配置树，在临时目录用真实Hydra组合三个根配置；对其中五个明确checkpoint节点检查原dataclass字段，再调用各版原`omega_conf_to_dataclass`及真实Hydra实例化。仅用包壳绕过verl初始化，BaseConfig完整执行、CheckpointConfig与旧McoreCheckpointConfig类定义未改；没有伪造这些配置类型，也没有加载模型后端。

以下是三个代表性**无加号**覆盖，不是完整main_ppo启动或原作者pytest：

| 版本 | PPO actor save_lora_only=True | Megatron critic strict=False | SFT async_save=True |
| --- | --- | --- | --- |
| #7426父版 ea532913 | 缺键拒绝 | 缺键拒绝 | 缺键拒绝 |
| #7388 head 0e540bc3 | 接受 | 缺键拒绝 | 缺键拒绝 |
| #7426 head 51f8f311 | 接受 | 接受 | 接受 |
| 当轮main 3c6d9b43 | 缺键拒绝 | 缺键拒绝 | 缺键拒绝 |

拒绝均为ConfigCompositionException、原因为ConfigAttributeError。#7426的五节点全部暴露各自dataclass字段；仅actor候选仍缺critic的strict／save_lora_only、SFT的async_save／strict／save_lora_only。缺字段的旧节点却仍可由原类默认值构造成对象；#7426与其父版在五处默认构造值的`asdict`结果相等。这是**配置对象值相等**，不是序列化字节相同或训练行为／存储格式相同。

历史候选的Megatron节点实际指向McoreCheckpointConfig、包含mbridge_config；当轮main指向基础CheckpointConfig，不能强行把不同版本的对象类型视为相同。脚本分别加载各版原类型，默认值等价断言只比较#7426与其真实父版，不把旧候选当成已能直接应用当前main的补丁。

### 验证范围与接续

Python3.12.13、Hydra1.3.7、OmegaConf2.3.1，两次运行JSON语义一致；[原始结果](../../experiments/verl-checkpoint-config/result.json)保留四版配置树manifest hash、类／转换器来源hash、五节点字段与构造值及覆盖结果。原#7426测试会递归调用get_class发现所有目标类及子类；本轮仅五处显式路径，**没有运行这套原反射测试、作者20项sanity、完整resolve或生成文件再生成检查**。也未执行checkpoint文件写读、LoRA过滤、后端异步保存或训练。新增可覆盖字段不认证这些底层功能。

复现从本项目根目录运行 `python -B experiments/verl-checkpoint-config/verify.py <已有固定对象的只读verl仓库>`；缺对象直接失败、不自动联网。问题已有两份重叠候选，保留供维护者选择，不另修复、应用候选或排新M5算力；后续等待实质反馈／独立需求，不继续扩类型或覆盖排列矩阵。任何社区写入及已有PR head推送仍需用户对具体动作明确授权。

## 异步恢复的存活库存与训练历史

**保住崩溃前的未消费轨迹，不等于重建检查点之后做过的训练。** [RFC #7579](https://github.com/verl-project/verl/issues/7579)提出TransferQueue持久后端、秒级metadata journal及离线reconciliation。2026-10-25读完整RFC与两条评论，参与者[Begunner](https://github.com/verl-project/verl/issues/7579#issuecomment-5448893542)提出恢复状态跨时间点的问题；作者[savaresejeremy明确确认](https://github.com/verl-project/verl/issues/7579#issuecomment-5466730166)：模型／optimizer从原trainer checkpoint恢复，queue／可选dataloader位置来自更新的journal；已消费对象被删除，不保留原训练顺序或批次，也没有一致时间点恢复。其[已有原型分支](https://github.com/savaresejeremy/verl/tree/feat/durable-replay-buffer)不是主线功能，本轮没有下载或执行这套117文件实现。

### 新库存恢复与训练回放的区别

假设模型checkpoint在step 5，journal在step 7。以下是这套提案的恢复语义，不是本项目执行过的崩溃实验：

| 状态 | 恢复来源或缺口 | 含义 |
| --- | --- | --- |
| 模型与optimizer | step 5 trainer checkpoint | step 6／7的参数更新回滚 |
| 存活轨迹与prompt状态 | step 7 journal对应的库存，经DROP／FENCE协调 | 较新未训练库存可少做生成，但不包含全部历史 |
| dataloader位置 | 若启用journal记录则为较新位置 | 不再重派部分已消费prompt，也不会因此重建丢失的更新 |
| step 6／7已消费轨迹 | 原型已删除，不留待回放 | 恢复后推进相同步号，不等于重做原批次或顺序 |

作者确认，剩余库存可能由比恢复模型更新的策略产生；kill drill验证库存／重发数量，没有测量这段版本差对loss轨迹的影响。消费日志与延后删除只是作者讨论的后续方向，不是已具备的“精确回放”。对这个提案的判断应先明确目标：接受换用幸存库存继续训练，还是要求重放原消费历史／恢复可比轨迹；减少重新生成不能单独回答后一个问题。

### 主线重发与预热容量

固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`的[恢复入口](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/trainer_base.py)依次加载actor、可选critic、data.pt与异步TransferQueue，再统计所有`is_prompt`组。fit将global step推进一格后，先重发pending／running prompt，再进入on_train_begin；重发会清除这些prompt原有trajectory并重新标记当前step，不是续接半生成token。finished／failure不重发，但都计入恢复prompt池。

已合并[#7762](https://github.com/verl-project/verl/pull/7762)修复的是预取窗口：两个异步trainer共用`_add_async_warmup_batches`，只补`max(0, target_prompts − restored_prompts)`，满额或超额都不再追加。当前main另支持按gen_batch_size对fractional warmup目标向下取整；下述原9例只有整数输入，不据此认证所有小数配置。**恢复prompt数不等于重发数，也不等于sampleable组数**；用重发数作为唯一扣减量，会漏掉已经恢复的终态组。

本轮编号查重仅命中#7762对#7579的引用，它不是持久后端实现；不据PR已合并宣布该RFC已落地。PR报告的两步e2e也不是本机结果，主线已经修复warmup不意味着新journal可精确恢复训练。

### 持久化保证与证据单位

RFC把写身份FENCE与索引复用区分：旧metadata若指向被回收并复用的索引，可能读到形状／dtype相同的另一条样本；其原型以日志顺序和写身份协调，而不是只看索引存在。这里是作者原型及报告，不是本机复现或新缺陷判断。

RFC默认日志与对象是flush而非fsync，将保证限定为进程级故障；节点掉电／丢失的更强保证需要另选屏障条件。journal配置间隔也不是独立最坏损失上限：报告中共享文件系统有9–13秒tick尾部，作者按间隔与tick耗时较大者限定窗口。其“27B-like”确定性harness统计的是累计trajectory-generation-seconds，非27B实跑节约的墙钟时间；8×B200 kill drill与内部27B观测是另外的外部证据。原型声明的253项包测试、153／155 V1测试均未在本机执行，更不能用“0个重发”证明完整恢复或收敛一致。

### 原 warmup 测试体的本机范围

[复查脚本](../../experiments/verl-async-resume-warmup/verify.py)从固定main抽取完整`_load_checkpoint`、`_add_async_warmup_batches`与两种on_train_begin方法，移除原测试的verl导入，保留其9个完整测试体、OmegaConf与MagicMock断言。九例两次通过：fresh 12、满池／超额不补、恢复6时补6、零目标／skip不补、colocate补8、separate补8，以及Mock加载后的计数接线。[原始结果](../../experiments/verl-async-resume-warmup/results.json)含完整源码hash与版本。

最后一例只有真实临时目录及exists检查，queue／actor loader为原Mock，恢复计数按原测试固定为6；没有data.pt内容加载、实际TQ／Ray、重发测试套件、checkpoint往返、journal、reconciler或训练。三份生产模块与原warmup测试另过语法检查，不算额外功能例。复现：

```bash
/private/tmp/verl312/bin/python -B experiments/verl-async-resume-warmup/verify.py /path/to/verl-git
```

后续只在明确恢复目标、实质接口反馈或已有获准匹配工作时再评估。[原M5 checkpoint参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)仅补状态／时间点核对条件，不新排崩溃实验、不在共享或安全未恢复环境注入故障，不自动安装原型或改现有恢复路径。

## V1 到期前保存的触发与计时范围

[verl #7757](https://github.com/verl-project/verl/issues/7757)指出默认V1 trainer没有接入ESI到期判定，已有报告者YeonwooSung的[#7760](https://github.com/verl-project/verl/pull/7760)。2026-10-25核open、非draft、未合并，普通／行内评论均0；编号及esi_redundant_time查重已有该覆盖，不另实现。固定head `16cd240d14cf0928961492fbb017b8a4356905bd`、实际共同祖先 `a9f2985159536a607211dcac730d3f5d55028950`，两提交、两文件211增7删；直接父提交`6043685b`已经有首版修复，API目标base`3f9d4c0d`也不能替代本次比较祖先。

候选将原频率／末步保存条件收进`_maybe_save_checkpoint`，再加入共享`should_save_ckpt_esi`。`save_freq > 0`仍是写盘前提；默认配置为−1，设了冗余时间也不自动开启保存。`__init__`和每次`fit()`起点都把max_steps_duration设为0。fit中的保存判断位于step计时块内，结束计时后才记录历史最大值，再进入validation，因此当前判断使用此前步骤的历史，历史值含保存耗时但不含验证。共享helper的保存耗时估计仍默认60秒，不等于实测写盘时间，也不是必能赶在回收前完成的保证。

### 两个提供方不是同一到期条件

候选没有修改[共享helper](https://github.com/verl-project/verl/blob/16cd240d14cf0928961492fbb017b8a4356905bd/verl/utils/checkpoint/checkpoint_manager.py)，其完整函数AST在共同祖先、head及固定main `3c6d9b43`相同。本机Python3.12.13用原函数和真实系统时钟，临时环境变量退出后恢复，得到以下有限控制：

| 到期输入 | max_steps_duration | 原helper结果 |
| --- | --- | --- |
| Volcengine MLP距现在30秒 | 0 | False |
| SageMaker AWS距现在30秒 | 0 | True |
| AWS已过期30秒 | 0 | True |
| AWS非法整数字符串，且候选save_freq=−1 | 0 | 解析ValueError，未调用保存 |

MLP分支要求剩余时间>0、历史最大耗时>0且剩余时间≤阈值；非法float字符串返回False。AWS分支只比较时间差小于阈值，没有前两项门槛，且整数解析异常没有同样捕获。这是已存在helper的不同契约，不是#7760新加的回归；这里不擅自统一云厂商语义。候选在检查save_freq前调用helper，所以“禁用写盘”不等于“跳过到期参数解析”。PR所说的“首次未记录耗时就不受ESI保护”应限于MLP分支；常规末步／周期保存仍可触发。

### 原测试通过与实际接线的区别

[原12个测试体](https://github.com/verl-project/verl/blob/16cd240d14cf0928961492fbb017b8a4356905bd/tests/trainer/ppo/v1/test_v1_esi_save_on_cpu.py)隔离执行全部通过。原测试自己用SimpleNamespace、真实OmegaConf及MagicMock保存方法；本机绑定未改的两个原方法，接原helper及原CPU fallback计时器／真实codetiming，没有完整PPOTrainer或写盘。原fixture隔离MLP／AWS环境，但设置到期变量的用例均走MLP；这12项不认证AWS行为、文件完成或恢复。

源码核对确认head的fit确有保存调用及计时后的记录调用，祖先与固定main则没有这两个调用。**这是调用位置的静态证据，不是执行fit。** 当前测试文件的说明提到`test_fit_records_the_step_duration`，实际12个定义中没有它；stub也自行提供max_steps_duration，从未执行真实`__init__`。因此这12项不能替代fit调用／初始化守卫，不能据PR正文的mutation报告认定当前文件已具备这些回归保护。

**供用户审视的评审要点**：请已有作者收窄首步／计时描述的提供方范围，补回声明的fit和初始化接线守卫，并将AWS继承行为与本次V1接入分开说明。是否改共享helper需单独决策，不把历史差异塞进竞争修复。反馈只留个人项目，任何上游评论、PR修改或既有PR-head推送仍逐项待用户明确同意。

本机未原生导入trainer／TransferQueue、运行全目录测试、真实云实例或模型，也没有checkpoint文件和恢复验收。两版trainer／helper及head原测试／CPU timer共六份完整快照compile通过；命令、版本和校验值见[当日日志](../../diary/2026-10-25.md)。本轮只解决触发契约与证据范围，不新排M5算力任务；保存完成和恢复指针仍见[原专题](#异步保存能力与恢复指针)。

## Megatron 调度进度与超参数恢复

恢复训练时，继续到哪一步与使用哪套调度超参数是两个问题。[#7953](https://github.com/verl-project/verl/issues/7953)已有报告者的[#7954](https://github.com/verl-project/verl/pull/7954)，2026-10-25读取open、非draft；固定head `53e632859898a248f89c8dd7e9abbf56e2248944`，base及实际祖先 `fbb4b3a8bf636f290c9c59fc346f756849e9c241`，5文件111增16删。不另开竞争修复。

候选在Megatron普通dist checkpoint与Megatron-FSDP两条加载路径中，只要payload包含scheduler且对象存在，就调用 `load_state_dict`；开启 `use_checkpoint_opt_param_scheduler` 时仍要求payload必须有该字段。两处仍嵌在 `should_load_optimizer` 内，不是改成从extra恢复；FSDP的存储位置不能直接套给Megatron。[构造接线](https://github.com/verl-project/verl/blob/53e632859898a248f89c8dd7e9abbf56e2248944/verl/utils/megatron/optimizer.py)把该flag与其逻辑反值分别传给官方scheduler的use-checkpoint和override选项，正常推进方法则调用 `step(1)`。

### 选择超参数不应丢掉计数

官方MCore0.19.2固定提交 `4b4acac9a1d28ea6829c8d4f566d75698a21249d` 的[完整调度器](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/optimizer_param_scheduler.py)由 `_check_and_set` 选择其管理的超参数，而 `num_steps` 在外部经 `step(increment=...)` 恢复。这里不能把flag理解为“False就不加载”。

本机采用warmup 2、总长10的线性调度，min/init LR为0、constant WD为0.1；数值1和2只为易核算的控制值，不是训练超参数建议。旧scheduler的max LR=1，在计数4保存；新对象max LR=2，从计数0开始。真实CPU SGD参数组、原scheduler和 `torch.save/load(weights_only=True)` 得到：

| 恢复方式 | 恢复后计数 | 恢复后全局max LR | 参数组当前LR | 再推进1后的LR |
|---|---|---|---|---|
| use-checkpoint=True | 4 | 1 | 0.75 | 0.625 |
| use-checkpoint=False、override=True | 4 | 2 | 1.5 | 1.25 |
| 不调用load | 0 | 2 | 0 | 1.0，此时计数仅为1 |

前两行的下一步分别与对应超参数下的连续调度一致。另将两flag均设False、当前与保存max LR不同，会在超参数检查时抛AssertionError；这不是verl默认配对，因为verl会同时传入互补的override标志。

### 新对象与参数组也是恢复条件

该版本的加载通过增加计数，而不是无条件赋值。本例先恢复到4并推进到5，再对同对象加载保存计数4，结果为9、LR为0.25；不能把它当幂等的“重新设回第4步”接口。冷恢复应核新对象初值，不能只检查checkpoint里存在num_steps。

全局超参数也不是实际LR的唯一来源。原 `get_lr` 优先读取optimizer参数组的 `max_lr/min_lr`，再回退scheduler的全局字段。有限反例中，旧SGD组保存max_lr=1，新对象配置2，按候选顺序先恢复optimizer组、再恢复scheduler并保留当前全局超参数后，全局max_lr仍为2，但参数组max_lr被恢复为1，计数4时实际LR为0.75。**这是普通SGD附加字段的真实状态控制，不是已证明Megatron分布式optimizer存在同一分组回归**；它说明完整验收需同时观察参数组与全局值。[PyTorch加载文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.load_state_dict.html)也提示初始化和状态加载次序会影响恢复后的学习率，不能跨scheduler实现照搬次序。

### 验证范围与接续

Python3.12.13／torch2.14.1，六组有限控制通过。官方scheduler、rank日志和Slurm环境helper三个原模块完整执行；仅用包壳和utils重导出连接原日志函数，不是原生导入整个MCore包。没有假scheduler或optimizer；真实torch临时状态文件已自动清理，但没有执行SGD更新、模型、Megatron optimizer、manager load或分布式保存恢复。五份候选相关Python另做AST/compile，连同三个官方模块共八份语法检查；一次性harness留工作目录，复现输入与hash见[日志](../../diary/2026-10-25.md)。

作者原manager测试mock底层checkpoint加载和scheduler对象，仅检查调用payload；本轮未复跑其43/57测试报告，也未执行Megatron-FSDP第二路径。真实完整后端的进度、参数组、下一次更新、optimizer/RNG与数据位置一致性仅补入[M5原checkpoint参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新增训练队列或自动应用候选；上游写入及已建PR的fork head推送仍逐项待用户明确同意。

## 1. Checkpoint 2层架构

2026-10-23 补充：[文件系统权重同步候选 #8355](../verl-v1-checkpoint-engines-deep-reading.md#文件系统权重同步的发布与清理)说明两层应按用途而非“是否写文件”区分。临时 safetensors 在 rollout 加载后删除，不含完整恢复状态；原 helper/receiver 的本地 I/O 控制不认证共享挂载或 server 安装。下方后端数量和早期代码仍按原日期理解。

```
★ ★ verl checkpoint = 2层系统!

Layer 1: CheckpointManager → 磁盘持久化
  → BaseCheckpointManager → abstract
  → FSDPCheckpointManager → FSDP1/FSDP2 per-rank shard + HF export
  → MegatronCheckpointManager → dist_ckpt + mbridge HF export

Layer 2: CheckpointEngineManager → 实时权重同步
  → 训练→rollout GPU 权重传输
  → 5后端: naive/CUDA IPC/NCCL/NIXL/Mooncake/HCCL
  → 不是磁盘checkpoint → 是live weight sync!

★ 关键: CheckpointManager = 保存到磁盘 → 恢复训练 → 持久化
  CheckpointEngineManager = 训练→rollout权重同步 → 非训练恢复协议；传输可经内存/网络或临时文件
```

## 2. FSDPCheckpointManager — Per-Rank Sharded Save/Load

```
fsdp_checkpoint_manager.py (57-395):

★ ★ Save (220-395): 每rank独立保存

每rank保存3个文件:
  model_world_size_{W}_rank_{R}.pt → sharded model state dict
  optim_world_size_{W}_rank_{R}.pt → sharded optimizer state dict
  extra_state_world_size_{W}_rank_{R}.pt → lr_scheduler + RNG(cpu/numpy/random/cuda)

FSDP1 context:
  with ShardedStateDictConfig(offload_to_cpu=True):
    → FlatParameter shards → torch.save per rank

FSDP2 context:
  with nullcontext(): → DTensor local shards → torch.save per rank
  → FSDP2 state_dict()直接返回DTensor local shard → 不需要ShardedStateDictConfig!

★ ★ HF Export (287-395): rank 0 only + optional

if "hf_model" in checkpoint_save_contents:
  → rank 0: get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)
  → init_empty_weights() → AutoModelForCausalLM.from_config() → 空骨架
  → save_model.save_pretrained(hf_local_path, state_dict=state_dict) → safetensors!

rank 0 always saves:
  huggingface/ → model_config + tokenizer/processor + generation_config

★ LoRA: lora_train_meta.json → {r, lora_alpha, task_type} → 下游恢复PEFT config

★ Load (138-218): 每rank独立加载
  torch.load(model shard) → model.load_state_dict()
  torch.load(optim shard) → optimizer.load_state_dict()
  torch.load(extra shard) → lr_scheduler + RNG恢复
  ShardedStateDictConfig(offload_to_cpu=True) → CPU加载→避免GPU OOM!
```

### LoRA only 的键完整性与加载副作用

LoRA-only恢复可以缺少冻结基座权重，不能因此忽略缺失的adapter张量。[verl #8350](https://github.com/verl-project/verl/pull/8350)补的是这层区别，而不是完整恢复事务。2026-10-23读取open、非draft、作者aoshen02；固定head `19b89e7100c06d7731247545274b117887ad8d38`，base及共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`，ahead1/behind0，两文件29增1删。已有作者实现，不另开竞争修复，也不把候选当主干行为。

原[键判定](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/utils/checkpoint/checkpoint_manager.py)要求字典非空、所有键均包含 `lora_` 或 `.adapter_`。进入该分支后，[FSDP manager](https://github.com/verl-project/verl/blob/19b89e7100c06d7731247545274b117887ad8d38/verl/utils/checkpoint/fsdp_checkpoint_manager.py)用strict=False加载模型：原来只拒绝unexpected_keys，候选再从missing_keys中过滤adapter键并拒绝。检查依赖load_state_dict返回的诊断；返回None时按空missing列表处理，不独立重算键清单。遗漏的base键仍允许，键名规则也不是任意PEFT格式或基座身份认证。

模型的load_state_dict先执行，结果检查在后，optimizer和extra/RNG的加载又在模型之后。因此拒绝缺键能阻止本次调用继续恢复后续状态，却不撤销已经写进模型的张量。这是恢复失败后的对象状态边界，不是另一个待补rollback缺陷；不能捕获异常后就假设对象仍为干净初值。

#### 真实 CPU 小模块的对照

从两版原load_checkpoint中取出完整的LoRA判定if/else分支，保留原分支体；原静态键判定和日志函数也原样执行。仅用属性容器提供model/rank/判定函数，model为真实torch.nn.Module，包含base、lora_A、lora_B三个1×1 Linear，无FakeFSDP/PEFT。每次初值base=99、A=B=7，输入adapter值A=0.5、B=0.3：

| 输入 | base版本 | PR head | 执行后的模型 |
|---|---|---|---|
| 完整adapter键，不含base | 接受 | 接受 | base仍99，A=0.5、B=0.3 |
| 缺lora_B | 接受 | missing adapter keys的ValueError | 两版均已把A写为0.5，B仍7 |
| 多出lora_C键 | unexpected keys的ValueError | 同左 | 两版均已写A=0.5、B=0.3 |
| 完整state，包含base=42 | 接受，走默认严格加载 | 同左 | base=42，A=0.5、B=0.3 |

两版各四组、合计八组CPU控制两次一致。第一行也说明：adapter键齐全不证明加载到正确基座；LoRA-only文件不提供这份base权重。lora_train_meta.json中的r/alpha/task_type并非基座内容hash，也不代替完整adapter键检查。

Python3.12.13/torch2.9.1，复用已有CPU环境，无新增依赖。没有运行完整manager、torch.load文件读取、FSDP/DTensor、优化器/RNG、跨rank通信或训练。作者[原17项测试](https://github.com/verl-project/verl/blob/19b89e7100c06d7731247545274b117887ad8d38/tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py)使用FakeFSDP/PEFT并修改missing_keys返回，本机只读源码并做完整模块compile，没有复跑该套件，不能把八组控制叫作原17项通过。另两版各两个完整变更Python文件AST/内存compile，共四份。

base/head manager SHA256分别为 `ff5aa26224081af3937155f19561629e8e1da69bf290ec2fc67953dd4704ff22` / `5c494485f1d9a91d7da1282a9c33a21e6b7cc8aedf39f8d0bbf9821bd3614714`，原head测试 `1966b702ed7658970a2a93cf8ace161c1c0e5ce724885549e0cc3935a444c40d`；完整patch `a7e68e89145ffb78128f42702caeb75c18c85210f2450231de1f0f5011780a2f`。实际分片/PEFT恢复条件仅补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用候选、下载模型或启动恢复实验。

以下复查仅需Python3.12+/torch及含固定对象的verl Git仓库；不新建永久实验工具：

<!-- lora-key-control:start -->
```bash
python3.12 -B - /path/to/verl-git <<'PY'
import ast
import hashlib
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import torch

BASE = "a4b9354968705a441fb7da7367924b15b902e771"
HEAD = "19b89e7100c06d7731247545274b117887ad8d38"
ROOT = sys.argv[1]
PATH = "verl/utils/checkpoint/fsdp_checkpoint_manager.py"


def source(rev, path):
    data = subprocess.check_output(["git", "-C", ROOT, "show", rev + ":" + path],
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(rev[:8], path, hashlib.sha256(data).hexdigest())
    return ast.parse(data)


predicate_tree = source(BASE, "verl/utils/checkpoint/checkpoint_manager.py")
base_class = next(n for n in predicate_tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseCheckpointManager")
predicate = next(n for n in base_class.body if isinstance(n, ast.FunctionDef) and n.name == "is_lora_only_state_dict")
shell = ast.parse("class Predicate: pass")
shell.body[0].body = [predicate]
ns = {}
exec(compile(ast.fix_missing_locations(shell), "<original-predicate>", "exec"), ns)
log_tree = source(BASE, "verl/utils/logger/aggregate_logger.py")
log = next(n for n in log_tree.body if isinstance(n, ast.FunctionDef) and n.name == "log_with_rank")
ns["logging"] = logging
exec(compile(ast.Module(body=[log], type_ignores=[]), "<original-logger>", "exec"), ns)
logger = logging.getLogger("lora-key-control")
assert torch.get_default_device().type == "cpu"
print("Python", sys.version.split()[0], "torch", torch.__version__)

for rev in (BASE, HEAD):
    tree = source(rev, PATH)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPCheckpointManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
    branches = [n for n in ast.walk(method) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == "is_lora_only_state_dict"]
    assert len(branches) == 1
    wrapper = ast.parse("def apply_state(self, model_state_dict, remote_model_path, logger): pass")
    wrapper.body[0].body = branches
    exec(compile(ast.fix_missing_locations(wrapper), "<original-load-branch>", "exec"), ns)
    for case in ("complete_adapter", "missing_adapter", "unexpected_adapter", "full_state"):
        model = torch.nn.Module()
        for name, value in (("base", 99.0), ("lora_A", 7.0), ("lora_B", 7.0)):
            layer = torch.nn.Linear(1, 1, bias=False, device="cpu")
            layer.weight.data.fill_(value)
            model.add_module(name, layer)
        state = {"lora_A.weight": torch.tensor([[0.5]]), "lora_B.weight": torch.tensor([[0.3]])}
        if case == "missing_adapter": state.pop("lora_B.weight")
        if case == "unexpected_adapter": state["lora_C.weight"] = torch.tensor([[0.9]])
        if case == "full_state": state["base.weight"] = torch.tensor([[42.0]])
        owner = SimpleNamespace(model=model, rank=0, is_lora_only_state_dict=ns["Predicate"].is_lora_only_state_dict)
        error = None
        try:
            ns["apply_state"](owner, state, "<in-memory-state>", logger)
        except ValueError as caught:
            error = str(caught)
        expected_error = case == "unexpected_adapter" or (case == "missing_adapter" and rev == HEAD)
        assert (error is not None) == expected_error
        if error: assert ("unexpected keys" if case == "unexpected_adapter" else "missing adapter keys") in error
        for name, expected in (("base", 42.0 if case == "full_state" else 99.0),
                               ("lora_A", 0.5), ("lora_B", 7.0 if case == "missing_adapter" else 0.3)):
            torch.testing.assert_close(getattr(model, name).weight, torch.tensor([[expected]]))
        print(rev[:8], case, "ValueError" if error else "accepted", "A=0.5; B=" + ("7" if case == "missing_adapter" else "0.3"))
print("8 original-branch CPU controls passed; no FSDP, checkpoint I/O, optimizer or training.")
PY
```
<!-- lora-key-control:end -->

## 3. MegatronCheckpointManager — Dist Checkpoint + Async Save

```
megatron_checkpoint_manager.py (115-1298):

★ ★ V2 Split Layout (3个sibling目录):

local_path/
  ckpt_contents.json → manifest(最后写入→signals completion)
  transformer_config.json → Megatron TransformerConfig
  model/
    huggingface/ → HF weights via mbridge(可选)
    dist_ckpt/ → Megatron sharded model weights
  optimizer/dist_ckpt/ → optimizer + lr_scheduler shards
  extra/dist_ckpt/ → rng_state shards

★ ★ Async Save:
  → Megatron.AsyncCallsQueue → async_save_request返回
  → finalize callback(manifest写入+HDFS上传+tracker更新)→attached to last request
  → 不阻塞训练! → 大模型checkpoint保存极慢 → async → 继续训练!

★ ★ Save (1169-1298):
  → FullyParallelSaveStrategyWrapper → 多rank并行写
  → _build_model_sharded_state_dict(metadata):
    → model.sharded_state_dict(metadata=model_metadata) → per VP rank
  → Megatron-FSDP: model.state_dict_for_save_checkpoint() → DTensor-aware
  → mbridge HF export: bridge.save_weights() / bridge.save_hf_weights()

★ Load (856-985):
  → dist_checkpointing.load() → FullyParallelLoadStrategyWrapper
  → Megatron-FSDP: load_fsdp_dtensor_checkpoint
  → Legacy v1 layout detection: _raise_for_old_layout() → reject!
```

## 4. CheckpointEngineManager — 实时权重同步

```
checkpoint_engine/base.py (345-515):

★ ★ 架构: Trainer → [NCCL/NIXL/RDMA] → Rollout → ServerAdapter

update_weights() (470-514) 核心流程:
  1. if backend=="naive"(colocated): trainer.update_weights() → 零拷贝generator!
  2. else:
     a. abort_replicas() → 取消所有in-flight rollout请求
     b. 创建临时RayWorkerGroup → 所有replica workers
     c. release_kv_cache_replicas() → 释放KV cache内存(weights留在原位)
     d. build_process_group() → prepare+build_topology+init_process_group
     e. trainer.update_weights() + rollout.update_weights() → NCCL/NIXL send/receive
     f. finalize() → 所有workers
     g. resume_kv_cache_replicas()
     h. resume_generation_replicas()

★ ★ 5+后端实现:
  → ColocatedCheckpointEngine("naive") → 同进程generator零拷贝
  → NCCLCheckpointEngine → NCCL collective send/receive+bucketing
  → NIXLCheckpointEngine → RDMA(SM90/NVLink必需)
  → KIMICheckpointEngine → Kimi专用
  → MooncakeCheckpointEngine → Mooncake
  → HCCLCheckpointEngine → Ascend HCCL

★ ★ Weight提供: engine.get_per_tensor_param() → yields (name, tensor)

FSDP engine实现(transformer_impl.py:794+):
  → LoRA: collect_lora_params() or collect_merged_lora_params() → merge/unmerge
  → FSDP2: model.named_parameters() → param.full_tensor().detach().cpu()
  → Bucketing: split_weight_chunks() / merge_weight_chunks() → 大tensor分bucket

★ ★ vs rLLM Tinker: rLLM用save_checkpoint→new SamplingClient → 磁盘中转!
  verl用CheckpointEngineManager → 实时通信 → 不写磁盘 → 更快!
  但: naive backend = 同进程generator → 等价于rLLM in-process → 极快!
```

## 5. Training Loop Checkpoint Integration

```
ray_trainer.py (974-1108):

★ ★ _save_checkpoint() (974-1041):
  1. actor_rollout_wg.save_checkpoint(actor_local_path, ...)
  2. critic_wg.save_checkpoint(critic_local_path, ...) → only if use_critic!
  3. torch.save(self.train_dataloader.state_dict(), data.pt)
  4. latest_checkpointed_iteration.txt → 同步路径直接open("w")/write；async只跳过trainer写入，需有实际backend finalize承接（见下节，不是所有backend都已支持）

★ ★ GRPO只保存:
  → actor checkpoint + dataloader state → 无critic
  → PPO额外保存critic；实际存储比例取决于模型、优化器及save_contents，不能固定算50%或3倍

★ ★ _load_checkpoint() (1043-1108):
  1. 读取latest_checkpointed_iteration.txt → step number
  2. global_steps = int(folder.split("global_step_")[-1])
  3. actor_rollout_wg.load_checkpoint(actor_path)
  4. critic_wg.load_checkpoint(critic_path) → only if use_critic
  5. torch.load(data.pt) → dataloader.load_state_dict()
  6. checkpoint_manager.update_weights(global_steps) → sync到rollout replicas

★ resume_mode 3种:
  → "disable" → 不恢复
  → "auto" → 找最新checkpoint
  → "resume_path" → 指定路径
```

### 异步保存能力与恢复指针

**保存出文件、异步请求完成、发布恢复入口和恢复出完整训练状态，是不同保证。** [#7952](https://github.com/verl-project/verl/issues/7952)报告非Megatron后端开启actor async_save后，trainer跳过恢复指针写入而后端没有承接；已有同报告者YeonwooSung的[#7955](https://github.com/verl-project/verl/pull/7955)，不另开修复。2026-10-24读取issue全文/唯一评论及两份候选正文，#7955与下述#4311均open/non-draft、未合并。

#### 当前保护针对配置能力

#7955固定head `b8c1aa6a076ba38e1710ce0e8e1a2adb7f554e9b`，API目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际祖先`23a341e361e18063b4ec53e1f7cc4eae68f102cf`（ahead1/behind6）。真正PR范围三文件84增1删，不把目标base当已包含的主线。[ActorConfig保护](https://github.com/verl-project/verl/blob/b8c1aa6a076ba38e1710ce0e8e1a2adb7f554e9b/verl/workers/config/actor.py)在post_init中读checkpoint.get(async_save)，只允许白名单megatron；它没有实现非Megatron异步保存，也不修改trainer发布协议。critic flag不在本项保护中，不能说所有保存配置均已校验。

固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`的[v0 trainer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/ray_trainer.py)和[V1 trainer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/trainer_base.py)仍按actor async_save跳过自己写tracker；同步路径直接open写文本，不是临时文件+rename式原子发布。[Megatron manager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/megatron_checkpoint_manager.py)确有async队列finalize和rank0 tracker写入，但这段actor侧逻辑的存在，不能单独认证critic、dataloader、远端副本及退出阶段都形成了同一个完整快照。本轮没有执行这些保存流程。

#### 旧DCP候选不能直接替代发布协议

[#4311](https://github.com/verl-project/verl/pull/4311)head `f0411c956ee5ae91070cc378115b434a357dcc9c`，API目标base同为3f9d4c0，官方compare祖先却是`f9d6198684f86112451d25c3227f185b1249973b`（ahead1/behind1148）。其八文件512增94删是旧分支改动，不是当前主线已完成的集成。不能按PR编号/正文“支持FSDP async”就从#7955白名单增加一个名字。

该head的[FSDP manager](https://github.com/verl-project/verl/blob/f0411c956ee5ae91070cc378115b434a357dcc9c/verl/utils/checkpoint/fsdp_checkpoint_manager.py)保存DCP future，开始下一次保存时可等待先前future；完成日志线程等待结果并记录成功/异常。这里未写latest_checkpointed_iteration，八文件改动也没有PPO trainer发布接线。**日志线程等待完成不是恢复指针发布者。** 这不是对整个旧PR的完整review或当前训练故障复现；只说明若将来整合到当前trainer，仍须明确谁在所有必要状态就绪后发布、如何传播失败以及退出如何drain，不能以DCP有async API代替这些契约。

#### 查找入口只检查指针和路径

主线[find_latest_ckpt_path](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/checkpoint_manager.py)读取tracker中的整数并检查拼出的路径是否存在，不扫描global_step目录选择最大值，也不验证模型/optimizer/extra/data.pt、manifest或checksum。M1用原两helper AST接真实os/torch，在真实临时目录做五组控制：path=None→None；目录在但指针缺失→None；指针3+空global_step_3→返回该路径；指针4+对应目录缺失→None；非整数指针→ValueError。最后一种不是静默“从头开始”，不能与缺指针情形混写。

另从#7955原测试类摘取`test_missing_tracker_makes_auto_resume_start_from_scratch`完整原方法，绑定原helper及真实unittest/临时目录，一项通过。它只证明发现入口，不是实际恢复模型或完整ActorConfig测试。Python3.12.13/torch2.14.1，无distributed初始化、trainer、Ray、FSDP/DCP或checkpoint数据保存/加载；四个相关原模块另仅AST/compile，作者14测试/5 subtests不是本机结果。版本与hash留[当日日志](../../diary/2026-10-24.md)，实际整合条件留[M5参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新增实验队列、应用候选或对外发布。

### 多角色异步保存的发布与清理

**写入完成、发布恢复指针、清理旧快照不是同一完成点。** ewan0x79的[已有 #8056](https://github.com/verl-project/verl/pull/8056)构建在SFT候选[#8003](https://github.com/verl-project/verl/pull/8003)之上，补RL多角色发布与退出drain。2026-10-30读取OPEN／非draft，head `c76d46ea6474f9fc9837a469c2ab74b58d921bbc`、API目标base `2ab35292039a51175a4b3d3b3b2d76e951a0d3e4`，28文件；本节只核driver协调器、有限保存／退出调用点，不是整个PR验收。旧#7955的后端能力白名单与本项发布协议不能互相替代。

[完整原协调器](https://github.com/verl-project/verl/blob/c76d46ea6474f9fc9837a469c2ab74b58d921bbc/verl/trainer/ppo/async_checkpoint.py)每次最多留一个pending_step。非阻塞progress会调用各异步worker group，即使较早角色正常返回未完成，也不短路后续角色。每组返回列表必须非空、每项必须`is True`；blocking仍不完整就报错。这是对**返回的确认**的检查，没有在协调器里另核预期rank数量或文件内容。一个RPC抛异常会直接传播，后续角色不再调用；“不短路未完成返回”不是“异常后仍保证所有角色继续执行”。

全部确认后，driver先写tracker.tmp并`os.replace`到正式tracker，再按retention列表清理，最后才置pending_step=None。retention可含同步角色，但只有异步角色收到finalize RPC。[V1保存入口](https://github.com/verl-project/verl/blob/c76d46ea6474f9fc9837a469c2ab74b58d921bbc/verl/trainer/ppo/v1/trainer_base.py)先给角色传`update_tracker=False, defer_retention=True`，再保存data.pt及异步模式下TQ状态，之后才设置pending_step；[legacy入口](https://github.com/verl-project/verl/blob/c76d46ea6474f9fc9837a469c2ab74b58d921bbc/verl/trainer/ppo/ray_trainer.py)同样在数据状态保存后设置pending。辅助状态就绪是调用方协议，不是协调器读取／验证manifest的结果。

| 失败位置 | 本地tracker | pending_step | 此轮清理 |
|---|---|---|---|
| 角色返回未完成或finalize抛错 | 旧值1 | 3 | 未调用 |
| tracker替换注入失败 | 旧值1，tmp为3 | 3 | 未调用 |
| tracker已替换、critic清理注入失败 | 新值3 | 3 | actor清理已调用，critic报错 |
| 重试清理全部返回 | 新值3 | None | 两角色清理再次调用 |

因此pending既可能表示尚未发布，也可能表示发布后清理未结束；异常类型和已发布指针须一起判断。重试会再次progress worker并重放清理调用，底层登记／删除的幂等性不能仅从driver推断。替换前失败保留旧指针、替换后失败不回滚新指针，也不等于整个保存具有磁盘断电／远端存储事务保证。

#### 原模块的有限文件系统控制

[脚本](../../experiments/async-checkpoint-publication/verify.py)在Python3.12.14标准库中直接执行完整原模块，无AST改写。显式worker记录器提供确认／异常，prune仅记调用；真实临时目录只写tracker并测试替换，没有角色权重或删除旧目录。八组[结果](../../experiments/async-checkpoint-publication/result.json)核空pending无操作、未完成仍progress下一角色、truthy整数1不当True、首RPC异常中断、替换失败／重试、发布后清理失败／重试及完成后无操作。两新进程相同；清理回调观察到的tracker均已为3，不从“收到prune调用”声称实际目录已删除。

协调器与前一提交d91f870e逐字相同，Git blob为2129be7ecb5dd321acb01bd6a7774cb6a069f07c。作者正文的六阶段／253项H20报告固定d91f870e，更早22阶段／1690项固定ec880ec1；单模块相同不能将这些整栈外部报告升级为当前head或本机通过。本机缺OmegaConf，没有运行原协调器配置测试、原retention／trainer集成套件、Megatron持久FIFO、Ray collective、恢复／模型或GPU。没有为此安装依赖或增加硬件队列。

#### 正常退出与边界反馈

RL legacy／V1在步边界非阻塞progress，最后步返回前和正常循环耗尽后blocking finalize；这不是任意异常／进程终止都必经的finally。两SFT fit已有`global_step >= min(total_training_steps, steps_per_epoch * total_epochs)`判最后步，并将`blocking=is_last_step`传给save，因此不能仅因循环后无drain就断定普通训练必漏尾部保存。反过来，该判断也不保证恢复耗尽游标后仍能到达最后一步。

#### 恢复边界与异步 drain 的不同责任

2026-10-30 07:13:58Z，[作者新回复](https://github.com/verl-project/verl/pull/8056#issuecomment-6517157160)明确报告：已在SPMD／Ray两SFT循环复现epoch-boundary恢复问题，并以真实StatefulDataLoader、训练／I/O替身的隔离CPU harness验证[#7974](https://github.com/verl-project/verl/pull/7974)；但它尚未合并或集成进#8056，所以当前head仍可复现。这是**作者的人工复现报告**，比此前bot线索更明确，但不是本轮独立运行或维护者批准。

固定#8056 c76d46ea的[handler](https://github.com/verl-project/verl/blob/c76d46ea6474f9fc9837a469c2ab74b58d921bbc/verl/utils/checkpoint/checkpoint_handler.py)仍直接load_state_dict；两SFT构造调用也未传steps_per_epoch。静态核对支持“未集成#7974的边界策略”。问题条件是保存时位于epoch末、恢复了耗尽状态，外层却已按global_step进入下一epoch；在既有SFT采样器条件下，会跳空该epoch，可能未执行足够更新而到不了is_last_step。尾部drain即使等待完已排队快照，也不能补回跳过的训练更新；**恢复样本进度**与**等待异步写入完成**不能相互替代。

#7974本次head `93f1763a219b5efa819c60e20bbf5bdb41f36d10`、API目标base `6f3f449e113b7eb84eef8c0912aad33ffdd322cd`，仍OPEN／非draft、未合并。对旧cfad9696与新93f1763a的共享handler构造／_at_epoch_boundary／_load_dataloader_state，以及两SFT _build_ckpt_handler，去除docstring后的AST全部相同；仍按当前epoch长度取模，不新增旧长度元数据。两head的Ray fit还含token求和／val_loss构造设备变化，故不能说整份文件或整个PR“只有文档变化”。本次PR文件列表仍无V1生产修改，旧正文声称覆盖V1及对#7402的比较仍须看下方[版本限定](#sft-与-v1-的恢复顺序)。

本轮只编译旧cfad、新93f及#8056三个版本的上述三模块，共九Python输入，并做指定方法AST／调用关键字检查；没有运行生产方法、旧19项恢复测试、八组driver控制或作者新harness，也不把这些历史结果迁移成93f整栈通过。新93f三输入SHA256依次为handler `f304df3b83556648faeb91bb77f6815970bd27b27f4c32f4e1cd4b6ff4ed838d`、SPMD `dbef9ee5b4e35eb099c822492b853c848a74713e89636741779228109f02e58e`、Ray `d017a6a15167a7fe113ce32c8b5849d606390a365e2a12733e2a225fe4c62600`。本轮不合并两个候选、补drain、改恢复策略或新增算力任务。

```bash
python3.12 -B experiments/async-checkpoint-publication/verify.py /path/to/verl-git
```

Git对象须预缓存，脚本禁隐式fetch。此driver边界学习收束，不扩模拟角色矩阵、不替作者改代码或M5环境；所有社区与既有PR-head写入仍逐项待用户明确授权。

### 对象存储下载与 checkpoint 提交

**下载入口能读对象存储，不等于训练状态已经完整写入并可恢复。** [S3 RFC #7641](https://github.com/verl-project/verl/issues/7641)提出原生FSDP2对象存储生命周期；2026-10-25读取全文及唯一评论，维护者[询问S3 FUSE挂载路线](https://github.com/verl-project/verl/issues/7641#issuecomment-5487704372)，尚不是对某个实现的采纳或验收。查重另有旧[#2334](https://github.com/verl-project/verl/pull/2334)（head23bd321f，17文件774增54删）与下载侧[#6850](https://github.com/verl-project/verl/pull/6850)，两PR均OPEN／非draft／未合并；本轮不另写后端或自动选型。

RFC的 `trainer.default_remote_dir` 与 `_COMMITTED` 是提案接口；#2334正文给出的是 `trainer.s3_checkpoint_dir`。不能把不同候选中的配置名拼成主线已可使用的统一API。

#### 当前路径与下载候选

固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的[fs.py](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/fs.py)只将hdfs://归为non_local；默认use_shm=False时，`copy_to_local("s3://...")`原样返回URI，并没有发起S3下载。[FSDP manager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/fsdp_checkpoint_manager.py)的hdfs_path明确未使用；legacy PPO的_load_checkpoint遇default_hdfs_dir非空直接拒绝，SFT共享handler的发现入口也以本地路径为准。不能给hdfs参数换一个URI就宣布获得对象存储恢复；挂载路径则应另按实际文件系统语义判断。

#6850固定head `c87b60ceace43f33eee027c95cbab32d05f8bc74`，API目标base3f9d4c0d；官方compare实际祖先为 `2390a3f5cff9d80d4844f8b4616ef8c2616276de`，ahead2／behind344，三文件120增4删。head本身是merge，直接第一父提交已含功能，不能将空parent diff当作没有实现。候选保持is_non_local仅识别HDFS，另加is_fsspec_path，将下载交给fsspec.fs.get，复用MD5路径缓存／filelock／目录记录；没有上传器、分布式checkpoint marker或remote resume发现协议。

本机Python3.12.13、fsspec2026.7.0、filelock4.2.4、pytest9.1.1，执行完整原fs.py／hdfs_io.py模块，原测试仅替换框架import，选择该候选新增的五测试体，结果重复一致：

| 原五测试体 | 固定main／实际祖先 | 候选head |
| --- | --- | --- |
| URI分类 | 缺is_fsspec_path，AttributeError | 通过 |
| 单文件下载、glob拒绝、目录尾斜线 | 三项预期检查失败 | 三项通过 |
| 本地路径透传 | 通过 | 通过 |

使用真实fsspec MemoryFileSystem和临时本地缓存，不是S3服务；未跑同文件另外三项HDFS mock测试或完整原生pytest套件，也未安装s3fs／gcsfs、读取云凭据、访问桶／HDFS或执行checkpoint manager。URI分类通过不证明相应云后端已安装，memory下载通过不证明认证、重试、远端校验或持久性。主线另一个S3 URI透传控制只核函数返回值，没有云端请求。

#### 多对象完成条件需要应用协议

[AWS一致性说明](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel)区分单key原子更新与跨key事务：成功PUT后的读取／列举可见，不代表多个rank的不同对象同时完成。因而“能LIST到global_step前缀”不是完整checkpoint条件；[条件写入](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)可约束单对象覆盖，但不会自动建立整步事务。RFC提出先上传分片及common.pt，再发布actor/.metadata与最后的_COMMITTED，这是待实现／验收的协议，不是主线已有保证。

审阅时应追问marker引用的对象集合是否固定、如何绑定run／step／world_size及对象内容、是否容许随后覆写，以及多writer重试怎样避免同前缀混入另一快照。只有marker名或对象存在不能回答这些问题。multipart完成只组装**一个对象**，不是一次提交所有rank；[AWS文档](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)也说明最终ETag不必是内容MD5，校验不能无条件以ETag代替约定的checksum。这些是存储协议审阅条件，不是本机发现某个已实现S3后端的新故障。

RFC的初期范围为actor-only、相同FSDP world size；包含critic的训练仍须另说明critic与数据／scheduler／RNG的同一步恢复契约。作者1.37TiB／64rank案例中，save_state返回约5.1秒与durable save约39.4秒分别描述暂停和持久化延迟，不能把前者当全部远端完成。本轮没有重跑该案例，也没有核验#2334完整17文件或其mock上传测试，不将其正文中的auto-resume主张挪作RFC实现或本机结果。

FUSE可以提供另一种路径接口，但仍须指定实现／版本并验证写入、seek、rename／关闭／同步、并发可见和冷恢复语义；不因能挂载就视为POSIX等价，也不未经实测否决这条路线。后续选型应先取得维护者方向与用户明确的云资源授权，而不是拼接#2334／#6850／RFC或从本机更多缓存测试推导durability。[复查脚本](../../experiments/verl-object-store-read/verify.py)、[结果](../../experiments/verl-object-store-read/results.json)随仓库保存，命令／hash见[当日日志](../../diary/2026-10-25.md)；GPU／云端恢复条件只补[原M5参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新建存储服务或实验队列。

### 保留目录与完整恢复的边界

`max_ckpt_to_keep` 控制已登记目录的淘汰，不是保存事务或磁盘容量上限。`BaseCheckpointManager` 将“保存前腾位置”和“成功后登记”拆开，避免上限为 1 时提前删掉唯一旧目录。[固定版本源码](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/checkpoint_manager.py)

| 上限 | 保存前 `ensure_checkpoint_capacity` | 成功后 `register_checkpoint` |
|---|---|---|
| 1 | 不删除旧目录 | 加入新路径后删除超出的旧目录，只跟踪新目录 |
| 2 及以上 | 已登记数达到上限时，先裁到上限减 1 | 加入新路径，再裁掉超限部分 |
| 0 | 不删除 | 继续记录所有路径，不按数量删除 |

因此，上限为 1 的保存过程中可以同时存在旧目录与正在写入的新目录，需要相应磁盘余量。路径列表从当前 manager 的空列表开始，并非磁盘目录扫描；删除使用 `shutil.rmtree(..., ignore_errors=True)`，列表裁短也不认证磁盘删除已经成功。

`register_checkpoint` 的前提是调用方已完成保存；它不检查模型分片、优化器、manifest 或恢复结果。不能绕过前提，手动登记一个空路径，再据此判定标准保存流程丢失数据。实际接线须与保留 helper 分开看：

- [FSDP manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/fsdp_checkpoint_manager.py)：rank 0 在保存前调用容量清理，在保存流程尾部登记；这些 helper 不是 trainer 的 actor、critic、dataloader 整轮联合提交。
- [Megatron manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/megatron_checkpoint_manager.py)：同步路径先清理，async 路径跳过保存前清理。`_dispatch_finalize` 将回调挂到最后一个 async request，并将所有请求入队；`_finalize_save` 按 manifest、可选 HDFS copy、async rank 0 tracker、登记保留策略的顺序执行。这是源码中的完成通知安排，不是本机对 Megatron 队列、跨 rank 故障或存储持久化的验收。
- [Trainer](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py)：写完 `data.pt` 后，同步路径以 `open(..., "w")` 和 `write` 更新 tracker；actor async 时跳过该更新，由 worker finalize 写入。尽管注释有 `for atomic usage`，这两处写 tracker 都不是临时文件加原子替换；不能据此称整个 checkpoint 保存具备崩溃原子性，也未由此复现恢复故障。

**CPU 复查**：[原 5 项清理测试](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py)全部通过，覆盖上限 1 的保存前保留、保存后删除、上限 2 预清理、0 不限量、连续三轮上限 1。测试创建的内容只有 `checkpoint.txt`，不含模型分片。原 fixture 自带空 model/optimizer 及 rank/world-size monkeypatch；没有初始化进程组。

复现使用[实验脚本](../../experiments/scripts/verify_verl_checkpoint_retention.py)，Python 3.12.13、torch 2.9.1、NumPy 2.0.2、pytest 8.4.2：

```bash
python experiments/scripts/verify_verl_checkpoint_retention.py /path/to/verl
```

脚本从固定 Git 对象选择完整原 `BaseCheckpointManager` 类、延后注解求值并提供真实标准库/torch/NumPy，完整加载原测试模块，由 pytest 执行 5 项原测试及 fixture。包根、框架导入、conftest/外部插件和缓存被隔离；收集限定到脚本目录。临时目录与删除审计限制在本次新建 scratch，退出时清理。未调用 RNG/设备方法或 FSDP/Megatron save/load，不属于完整包、真实 checkpoint 恢复或断电/进程崩溃测试。

源文件 SHA256：manager `be4fa702f0fab63dfb1d482716d31bc13a3b8f5129adbbd61872a169b07cbbc6`；原测试 `30f2df35a31c6eeb21db737557336c9178bb4b69a5427a44621121eaa4a18a18`。

### 重启后的保留状态与路径身份

**同一进程中的轮换历史，不会自动随模型checkpoint恢复。** [#7395](https://github.com/verl-project/verl/issues/7395)报告SFT进程重启后，manager的 `previous_saved_paths` 重新为空；加载step 1再保存step 2时，上限1仍留下两个目录。2026-10-25读全文／0评论，已有ai-yang的OPEN／non-draft [#7396](https://github.com/verl-project/verl/pull/7396)，head `8a1a031a9c03105839dcc6659e15cd975bfd5246`。API目标base为3f9d4c0d，实际父提交／比较祖先为 `09ac37258ea66b0cb69b2738eec3074ea4e7261c`，ahead1／behind149，七文件375增1删；不把目标分支的其他更新算进修复。主线另固定为3c6d9b43，未合入此候选。

[候选handler](https://github.com/verl-project/verl/blob/8a1a031a9c03105839dcc6659e15cd975bfd5246/verl/utils/checkpoint/checkpoint_handler.py)在模型加载及dataloader恢复入口完成后记录已加载路径；dataloader文件不存在时原入口只告警，不把它升级为强制完整状态校验。保存时先由engine登记新模型目录，再写dataloader／替换tracker、处理可选HDFS copy及既有SPMD barrier，最后才调用保留finalize。[manager](https://github.com/verl-project/verl/blob/8a1a031a9c03105839dcc6659e15cd975bfd5246/verl/utils/checkpoint/checkpoint_manager.py)只考虑**这次记住的已加载目录**：上限恰为1、新目录已登记且存在、两路径属于相同父目录的global_step数字序列、且新step更大，才删旧路径。没有扫描或清理所有历史目录；上限1仍不是全盘目录数／容量保证。

| 原测试情形 | 候选中的保留行为 | 不能据此推导 |
|---|---|---|
| 加载1后新2正常保存 | 旧1删除，新2与tracker 2保留 | 模型分片及跨rank恢复内容已验收 |
| dataloader保存或tracker替换失败 | 旧1仍在，tracker仍1 | 所有写入被回滚、全局保存具备事务性 |
| HDFS copy返回False | 旧1仍在，但本地tracker已2 | tracker已推进就代表远端完整完成 |
| 另有更早但未记住的同级目录 | 只处理被记录的旧路径，不扫其他目录 | 自动修复历次重启积累的全部历史 |

[CPU复查脚本](../../experiments/verl-resume-retention/verify.py)复用旧保留测试的隔离加载器，原样加载完整manager类、handler定义及原文件系统／logger helper，配合作者原Mock engine、dataloader、barrier和HDFS失败返回；测试体、fixture及断言不改。临时根统一为解析后的路径时，[原两文件22参数项](https://github.com/verl-project/verl/blob/8a1a031a9c03105839dcc6659e15cd975bfd5246/tests/utils/ckpt/test_checkpoint_handler_on_cpu.py)全部通过；父版本及main只运行同一原重启回归，各以“旧目录仍在”的AssertionError失败，未将缺方法或收集错误当复现。SPMD顺序例仅Mock事件序列，不是实际多rank同步。[结果及hash](../../experiments/verl-resume-retention/results.json)随仓库保存。

首次保留本机系统临时路径拼写时，head原22项为21过1失败，原相对登记用例保留了旧目录。本机 `/var/...` 经chdir后成为 `/private/var/...`；`abspath` 不解析符号链接，同一物理目录的父路径字符串可以不同，因而被“不同series”保护挡住。仅将新建scratch入口统一为其resolve路径，不改候选或原测试，22项即通过；切回system拼写又复现21过1失败。**这是macOS路径别名条件下的保守不删除，不是Linux训练复现、误删或数据丢失证据。**

本地反馈要点是请作者说明series身份按路径字符串还是物理目录定义，并让原相对路径测试覆盖／规避这一主机条件；不能为追求全绿直接把生产删除边界统一改成realpath。是否允许沿符号链接扩大“同一series”的范围，须与外部恢复路径保护一起审视。本轮不改候选、不发布review或另开同题修复。

Python3.12.13／torch2.14.1／NumPy2.5.4／pytest9.1.1，仅真实临时文件、torch.save/load小字典与隔离定义；原Mock不包含真实FSDP分片、optimizer／RNG恢复、Ray RPC或HDFS。作者22＋27项及4×3090结果不合并计为本机验证，未跑模型／GPU、真实进程重启或故障／断电恢复。脚本对已声明的baseline／别名失败可返回0，JSON仍保存实际pytest_exit=1，不能据脚本成功称原套件全绿。目录删除审计仅允许本次新建scratch并在退出时移除，没有扫描／删除用户检查点。真机条件只补[原M5发布参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新排训练、清理或自动应用候选；下轮转独立问题，不扩路径／故障注入矩阵。

### SFT epoch 边界与加载器状态

取到最后一批数据，不等于迭代器已经观察到`StopIteration`。[verl #7402](https://github.com/verl-project/verl/pull/7402)处理的正是这个差别：SFT从下一epoch开始训练，却恢复了尚停在上一epoch末尾的加载器状态，第一次遍历可能没有任何batch。已有作者ai-yang负责该修复，本项目只核验数据流，不另做竞争实现。

2026-10-23 API核对为open、未合并；固定head `b7d1f36e446da4caeb85a7f9d770daa416a9e76f`，目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际祖先 `09ac37258ea66b0cb69b2738eec3074ea4e7261c`（ahead4/behind149）。相对祖先5文件322增5删，不把目标分支的149个提交算进修复范围。

2026-10-30人工反馈复核：UgaTheDev于[10-29的普通评论](https://github.com/verl-project/verl/pull/7402#issuecomment-6511331959)确认sidecar移除、长度变化告警及完成保护测试断言的修订，并把legacy边界残留列为非阻塞提醒。head仍为b7d1f36e，没有新实现。发言者明确自述缺torchdata、未跑CPU测试；这不是APPROVED review，也不能升级作者24项或本机历史控制的验证等级。

评论建议说明legacy限制，或考虑按当前长度取模回退；当前[PR兼容说明](https://github.com/verl-project/verl/pull/7402)已明确legacy耗尽迭代器保留旧行为，[原issue的10-23更新](https://github.com/verl-project/verl/issues/7401)也已披露修复需要新保存的长度元数据。因而不能把这条提醒记为新发现、缺少说明或已接受的新恢复策略。仅凭当前长度猜边界仍无法识别旧长度／数据变化，与现有“不丢弃未知旧位置”的兼容选择不同。

本轮只核反馈与既有契约，不重跑下方历史迭代器／八组控制，不新增fallback、作者回复草稿或M5实验。后续若代码或兼容策略有实质改变再续；普通评论不能授权本项目修改其PR。

#### 最后一批与结束确认

本机Python3.12.13、torch2.9.1、torchdata0.11.0的真实CPU对照：8条数据、batch2、单副本显式rank的DistributedSampler、shuffle/seed0、两层drop_last、0worker且不pin。取完4批后立即保存，与再请求一次、确认结束后保存，恢复到新加载器并设置epoch1，结果不同：

| 保存时机 | 恢复后第一次遍历 |
|---|---|
| 最后一批已返回，尚未观察到StopIteration | 0批 |
| 已观察到StopIteration | 4批，顺序`[5,4] [2,6] [7,3] [1,0]` |

这两个快照在该版torchdata中的私有字段`_iterator_finished`分别是False/True。它解释本次现象，不应作为跨版本的公共API约定；不能笼统说“任何耗尽快照都会丢掉下一epoch”。

#### 边界修复需要旧长度

[原CheckpointHandler](https://github.com/verl-project/verl/blob/b7d1f36e446da4caeb85a7f9d770daa416a9e76f/verl/utils/checkpoint/checkpoint_handler.py)将保存时的`len(train_dataloader)`写入原`data_<dp_rank>.pt`字典的`_verl_steps_per_epoch`，加载时先取出，才把剩余状态交给torchdata。只有保存/当前长度相等且为正，恢复step也为正且整除该长度时，才跳过加载器状态、使用新迭代器。中途位置正常恢复；长度变化或缺元数据时警告，并按原生语义恢复，不猜测新边界。

[复查脚本](../../experiments/scripts/verify_verl_sft_loader_resume.py)提取祖先/head未改写的原加载方法及原日志helper，使用真实加载器、采样器与torch.save/load临时文件；只用属性容器提供self的loader/rank字段，不构造引擎。以下8组自编控制两次执行一致；前六组保存时均尚未观察到StopIteration，保存epoch长度为4。

| 控制 | step / 当前epoch批数 | 祖先→head返回批数 | 边界 |
|---|---|---|---|
| 新元数据、相同长度的边界 | 4 / 4 | 0→4 | 修复目标 |
| 新元数据、相同长度的中途 | 2 / 4 | 2→2 | 保留游标 |
| 新元数据、epoch缩短（batch4） | 2 / 2 | 1→1 | 不用新长度猜边界 |
| 新元数据、epoch变长（batch1） | 4 / 8 | 0→0 | 回退不保证补全下一轮 |
| legacy无元数据、中途 | 2 / 4 | 2→2 | 兼容旧位置 |
| legacy无元数据、边界 | 4 / 4 | 0→0 | 旧空迭代行为未修复 |
| 新元数据、已观察到结束 | 4 / 4 | 4→4 | 原生行为本已开始下一轮 |
| 新元数据、长度相同但数据ID全部加100 | 4 / 4 | 0→4 | 不检测数据身份变化 |

基线对照也读取带新增键的字典：该版torchdata容忍额外键；另有不带该键的legacy对照。这不是基线save格式已包含元数据。改变batch大小的结果只是该固定原生加载器的回退现象，不认证更换数据或拓扑后的精确恢复。相同batch数更不代表同一份数据。

#### 完成保护与验证范围

两份SFT `fit()`源码在`resume_global_step >= total_training_steps`时立即返回，早于tracker、sampler设置和batch遍历；但构造trainer及初次加载引擎checkpoint在fit之前，不能据此宣称完成任务不会加载模型。当前修复没有新增sidecar、配置校验异常或checkpoint发布顺序保证。

本机执行的是2个原生迭代状态控制及8组原方法数据流控制，并非原生verl导入、完整handler构造、Ray/SPMD调度或fit/模型/optimizer恢复。作者在同head报告的24项测试（12个checkpoint例、12个fit例，torch2.11.0+cpu/Ray2.56.1，含引擎/同步替身）与本机结果分开；本机未收集执行那24项。作者也明确没有重跑分布式/GPU、HDFS和异步finalize，旧实现的GPU报告不移作当前证明。本题是CPU加载器学习，不为未执行的完整训练另排M5任务，也不改上游实现。

复现：`python -B experiments/scripts/verify_verl_sft_loader_resume.py /path/to/verl-git --deps /path/to/torchdata-target`，需要上述固定Git对象、torch及torchdata0.11.0。本次仅将torchdata用`uv pip install --target ... --no-deps torchdata==0.11.0`装入独立work目录，未修改复用的torch环境。原handler SHA256：祖先`3b547c2b97e01b7587866053b4aa624d81b8cdd9d75fff20a97420690206a896`、head`676cf43c504922e610e88d236f0afcc5b4a449b63de1893616dee96a384960e8`；原日志helper模块`64d8556b0d6acc667a5be2dd43908111c9e58aae58d779ee9b02f0f28e2b42f7`。脚本打印hash并禁止隐式Git fetch。

### SFT 与 V1 的恢复顺序

**恢复后能继续取batch，不等于数据顺序与未中断运行一致；相同的边界判断也不能跨采样器照搬。** [#7974](https://github.com/verl-project/verl/pull/7974)是ewan0x79对[#7401](https://github.com/verl-project/verl/issues/7401)的另一现存候选，2026-10-26核为OPEN/non-draft。该轮固定head `cfad96962d5aa9640cce052774a34b810382b20f` 相对实际祖先 `339a5636367772b4ae3131588f718b5e8b28aa77` ahead2／behind0、五文件311增0删；API目标base为3f9d4c0d，不能替代实际祖先。2026-10-30新93f1763a的有限源码比较及与#8056的未集成关系，见[恢复与drain责任](#恢复边界与异步-drain-的不同责任)；本节原19参数项仍只属于cfad固定版本。

修订前提交 `4105a9914658ae2dc411d80cb530681ce759546e` 同时给SFT共享handler和V1加了边界skip。[社区反馈](https://github.com/verl-project/verl/pull/7974#issuecomment-6393919528)指出V1会重复最初排列；[作者新提交](https://github.com/verl-project/verl/commit/cfad96962d5aa9640cce052774a34b810382b20f)已完整撤回V1生产改动，新增数据顺序测试。当前V1文件与上述祖先逐字节相同；SFT三份生产文件则与修订前相同。旧正文仍称覆盖V1，不能以正文判断当前实现。

#### 两种采样器的状态来源

SFT的DistributedSampler由 `set_epoch(epoch)` 重新确定本轮排列；外层for-loop遇到恢复的耗尽状态就结束该epoch。因此在固定epoch长度、每step消费一batch的条件下，丢弃上一轮结束游标后重新设置epoch，可避免跳空一整轮。

V1的[原采样器工厂](https://github.com/verl-project/verl/blob/cfad96962d5aa9640cce052774a34b810382b20f/verl/trainer/ppo/utils.py)使用torchdata RandomSampler及专属torch.Generator；其RNG进度随loader state保存。[原_fetch_one_gen_batch](https://github.com/verl-project/verl/blob/cfad96962d5aa9640cce052774a34b810382b20f/verl/trainer/ppo/v1/trainer_base.py)捕获StopIteration后会再建迭代器、再取一次，所以恢复耗尽状态本来不会丢掉下一轮。跳过恢复却丢失RNG进度，使新loader重复初始排列。这里seed=None也仍新建显式Generator、不是自动采用全局随机种子，所测默认状态同样可重复。

#### 原测试与修订前对照

[隔离脚本](../../experiments/verl-epoch-resume-order/verify.py)原样加载新head两份测试模块，由pytest保留原fixture／参数／断言收集。完整原handler定义、原采样器工厂／TensorDict转换helper及原V1两个方法接真实torchdata／torch CPU／小状态文件；V1类外壳跳过模型初始化，actor仍是作者MagicMock。不是完整verl导入或训练器构造。

Python3.12.13／torch2.14.1／torchdata0.11.0+cpu／pytest9.1.1、单线程、原参数num_workers=0：

| 固定生产版本与原参数项 | 结果 | 实际覆盖 |
|---|---|---|
| 新head SFT 15项 | 全过 | 5个边界判定、4个加载策略、6个实际数据顺序项 |
| 新head V1 4项 | 全过 | seed42／None × 中途step3／边界step8，连续3轮共24批 |
| 修订前V1，同一新head原4项 | 2过2失败 | 中途两例过；边界两例均在完整顺序相等断言失败 |

旧V1在两个边界例的首个不一致均为全程第9批（零基index8）；seed42时实际 `[38,24,48,63,10,29,36,51]` 等于最初第1批，而未中断参考为 `[8,55,19,61,44,39,57,5]`。这些是本机原测试局部变量，不是社区GPU表格。[新结果](../../experiments/verl-epoch-resume-order/head-result.json)和[旧V1结果](../../experiments/verl-epoch-resume-order/old-v1-result.json)保留源码／脚本指纹与失败详情；旧模式脚本在确认预期两失败后返回0，JSON中的pytest_exit仍是1。

SFT六顺序项使用num_replicas／rank显式枚举 `(1,0),(2,0),(2,1)`，各自产生真实DistributedSampler数据流；**没有启动两个rank进程或做collective**。原测试只覆盖相同数据、batch和长度，不能升级为拓扑变化、自定义stateful sampler、worker进程、异步TQ、模型／optimizer恢复或完整训练验收。作者提交所报Linux35项及社区GPU观察另属外部证据。

#### 与元数据候选的真实区别

#7974按**当前**steps_per_epoch取模，不增加旧长度元数据，因而可对legacy文件触发skip，但也无法判断保存时长度是否相同。另一个[#7402当前head b7d1f36e](https://github.com/verl-project/verl/blob/b7d1f36e446da4caeb85a7f9d770daa416a9e76f/verl/utils/checkpoint/checkpoint_handler.py)仍为上节已读版本：把旧长度存入现有data字典，长度变化或legacy缺键时告警并原生恢复。它已不是“新增版本化sidecar、长度变化硬报错”的方案；#7974正文对它的描述过期。两者的legacy修复范围／保守判断不同，不能只按改动行数认定谁全面更好，也不叠加两个未合并候选。

本轮不重跑旧#7402八组控制、不另写修复或发表review；仅记录新修订的边界与本机原测试结果。没有GPU工作或新M5队列，用户未明确授权前不改社区及已有PR-head。复现命令为 `python verify.py /absolute/verl --deps /absolute/torchdata-target --revision head` 或末参 `old-v1`，可选 `--output result.json`；仅使用固定本地Git对象，禁止隐式fetch。隔离加载最初缺__file__、pytest未限制祖先收集、失败栈未附内存源码时均曾中止；只修验证器上下文和诊断，不改原测试或生产逻辑。

### RL 数据子集与恢复身份

**加载器恢复了相同索引，不保证这些索引仍指向相同行。** [#7816](https://github.com/verl-project/verl/issues/7816)讨论重启时重新随机抽取 max_samples 子集；[#7824](https://github.com/verl-project/verl/issues/7824)讨论验证子集使用哪个 shuffle 标志。它们与上节 SFT 的迭代结束状态不同：即使恢复位置正确，重建数据的身份仍可能改变。2026-10-25核对已有两位作者的方案，不另开竞争实现。

#### 固定种子与验证集选择分开

[#7819](https://github.com/verl-project/verl/pull/7819)仍 OPEN、非draft、未合并，固定 head `0554e35fcf642a16b53877d330a7e7d21286752e`。API目标base为 `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际共同祖先为 `a9f2985159536a607211dcac730d3f5d55028950`；相对祖先4文件182增6删。生产变化为未配置 data.seed 时用42抽子集，显式seed及shuffle=False的前N行分支不变。固定主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的完整 rl_dataset.py 与该祖先字节相同，尚无这个fallback。

另一个 [#8141](https://github.com/verl-project/verl/pull/8141)仍 OPEN、非draft、未合并，head `9a7bd53b4c56d68f23ed0b647a899cf98e7880ef`、base及祖先 `75879f7f475fd6b64c779f7d9212e45503f58b8f`，2文件125增1删。它只在 `create_rl_dataset(..., is_train=False)` 时复制配置，将 validation_shuffle 映射到 dataset.shuffle；未配置该键则退回 data.shuffle。这个head仍使用旧的无seed抽样逻辑，因此“validation_shuffle=True且seed未设置”不因该PR自动稳定。

两个修复不能互相替代。固定种子解决固定输入下的子集重建；选择 shuffle 标志决定随机抽取还是按文件拼接顺序取前N行。后者改变评测数据组成，不能只因配置名更一致就称统计更具代表性；头部截取也不保证每个数据来源都有样本。

#### 原测试体与真实数据读取

复用 Python3.12.13／torch2.14.1、torchdata0.11.0+cpu、datasets5.1.0、PyArrow26.0.0、pandas3.0.6、NumPy2.5.4、OmegaConf2.3.1。完整原 RLHFDataset 类、get_dataset_class、工厂及采样器经 AST 隔离，fs／hdfs_io载入完整原模块；本地 Parquet／JSONL、HF Dataset 和 StatefulDataLoader 均真实运行。关闭长度过滤、tokenizer=None，原测试自带 row-ID view 避开tokenization；没有伪造采样器、文件加载或状态恢复。

直接调用两份原测试的函数体与其原fixture／helper，只移除 verl 导入声明并绑定上述原定义，不改断言；不是原生 pytest 收集或完整包导入。两次执行得到相同通过／失败集合：

| 原测试体 | 祖先 | 对应候选 |
| --- | --- | --- |
| #7819显式seed、前N行、未设seed重建、恢复后续行 | 2过、后两项AssertionError | 4过 |
| #7819默认常量固定为42 | 无该符号，不运行 | 通过 |
| #8141五项工厂测试 | 4过、validation_shuffle=False项AssertionError | 5过 |

#7819的64行取8行，fallback子集为 `[62,63,53,38,60,26,5,44]`；显式seed1234为 `[6,16,10,57,55,22,58,56]`。恢复例先消费两批共4行，state_dict之后的剩余行与重建／load_state_dict后的行一致。再保持文件路径和行数、将全部 row_id 加1000：加载器接受状态并读到原剩余行对应的加1000值。后者仅说明固定种子不是文件内容校验，不是该PR承诺支持数据变更却失败。[NumPy随机流兼容政策](https://numpy.org/doc/stable/reference/random/compatibility.html)不提供任意跨版本／构建／环境的Generator一致性保证；文件顺序／内容、过滤条件或tokenizer变化也须分别固定，不能以相同长度推出身份一致。

#### 工厂测试通过不代表所有入口接线完整

在 #8141 固定head上，原五项测试都显式传入 is_train；它们未覆盖遗漏该参数的调用者。将四个生产文件中**原验证集 create_rl_dataset 调用表达式**单独接到该head的原工厂和真实JSONL读取，配置 shuffle=True、validation_shuffle=False、seed1234、64行取8行：

| 原调用入口 | is_train 参数 | 返回 row_id |
| --- | --- | --- |
| [V1 trainer_base](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/trainer/ppo/v1/trainer_base.py#L718) | False | 0至7 |
| [旧同步 ray_trainer](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/trainer/ppo/ray_trainer.py#L387) | 省略，默认True | 6,16,10,57,55,22,58,56 |
| [one_step_off_policy](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/experimental/one_step_off_policy/main_ppo.py#L84) | 省略，默认True | 同上 |
| [fully_async_rollouter](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/experimental/fully_async_policy/fully_async_rollouter.py#L393) | 省略，默认True | 同上 |

这里 self／config 仅提供原表达式需要的字段，没有构造或执行trainer、Ray或rollout。因此证据是：**该head的三处调用表达式未进入新增验证分支**，不是三条真实训练失败记录。#7819正文仍提及调用点已显式传参，但其当前四文件diff没有这类修改；不能依赖这段描述假定叠加两PR就会接好。是否补齐入口和是否采用头部选择语义，应由现有作者／维护者与用户审视，不在本项目另写第三个修复。

供用户审阅的反馈要点：#8141的直接工厂测试通过，但三个旧入口漏传 is_train=False；建议现有作者核对这些调用者，并增加调用层覆盖。2026-10-25读取PR行审评论为0、普通评论仅CLA机器人；本项目未发评论、改PR或推其head分支。

七份固定源码快照另通过完整 AST／内存 compile；输入、环境、hash与一次性复现命令见[当日日志](../../diary/2026-10-25.md)。未运行完整trainer checkpoint／模型恢复、prompt过滤／tokenization、原生pytest、并行worker或设备，不需要为这项CPU学习另排M5任务，不把已有作者成果计为我方修复。

## 6. FSDP2 / DTensor Checkpoint

```
fsdp_utils.py (422-471):

★ ★ Version Detection:
  fsdp_version(model):
    isinstance(model, FSDP) → 1 (FSDP1)
    isinstance(model, FSDPModule) → 2 (FSDP2)
    else → 0

★ ★ FSDP1: ShardedStateDictConfig → FlatParameter shards
★ ★ FSDP2: nullcontext() → DTensor local shards → state_dict()直接返回!

get_fsdp_full_state_dict() (438-471):
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)

★ ★ FSDP2 Sharded Save/Load to CPU (1044-1142):
  fsdp2_sharded_save_to_cpu():
    → iterate all parameters → _local_tensor.detach().cpu() + DTensorSpec
  fsdp2_sharded_load_from_cpu():
    → verify device_mesh consistency → copy _local_tensor back from CPU
```

## 7. FSDPModelMerger — 离线合并→HF格式

```
model_merger/fsdp_model_merger.py (35-):

★ ★ 合并FSDP shards → 标准HF格式:
  1. 读取fsdp_config.json → world_size
  2. 加载rank-0 state dict → 提取DTensor/device_mesh信息
  3. ThreadPoolExecutor并行加载所有rank shards
  4. 合并DTensor shards: Shard(N) → torch.cat(dim=N)
  5. save_pretrained() → HF format(safetensors+config.json+tokenizer)

★ CLI:
  python -m verl.model_merger merge --backend fsdp --local_dir <path> --target_dir <path>
  python -m verl.model_merger merge --backend megatron ...

★ LoRA: lora_train_meta.json → 重建PEFT config → 分离adapter和base weights
```

### 分片重组与 LoRA 融合不是同一步

`model_merger merge` 的 merge 首先指重组训练分片，不自动表示把 LoRA 增量融入基座。[#7285](https://github.com/verl-project/verl/issues/7285)报告者描述三次评估疑似用了基座，并明确承认**当时没有做文件哈希比较**；其标题“byte-identical”不能转述为已实测的哈希结论。源码能确认未主动融合，但实际剩余权重、dtype和序列化布局仍须分别检查，不能据此推断所有导出目录都与最初基座逐字节相同。

2026-10-25 核验已有[候选 #7341](https://github.com/verl-project/verl/pull/7341)，OPEN、非 draft、未合并；唯一普通评论为当时的CLA提示，reviews／行内评论为空。固定head `fb304fc116fbacc2ccfef6c01eefc8acae76bf65`，父提交／与当前main共同祖先 `b75f0d06a72b74e471cd4d3402bcbb0ceaa2e0c8`；相对main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` ahead1／behind280，两文件366增1删。编号与关键词检索已有该覆盖，不另写融合补丁；候选CLI不当作主线已有功能。

#### 默认仍分离 adapter

[原导出方法](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/model_merger/base_model_merger.py)先从state_dict弹出含 `lora_` 的键，保存到 `target_dir/lora_adapter`，再将余下基座键改名后交给HF保存。候选、父版与main的整个 `save_lora_adapter` 方法AST相同；新增的是另一条可选路径，未改变这个分离契约。若使用默认导出而推理只加载根目录、没有加载独立adapter，不能把测得结果当成“基座加LoRA”的效果。

候选 `--fuse-lora` 仅在merge子命令提供，配置默认False；test子命令通过getattr回退仍为False。[融合方法](https://github.com/verl-project/verl/blob/fb304fc116fbacc2ccfef6c01eefc8acae76bf65/verl/model_merger/base_model_merger.py)先调用同一分离函数，再 `model.load_state_dict(..., strict=False)`、`PeftModel.from_pretrained`、`merge_and_unload()`，最后用融合模型的state_dict替换原字典。无adapter键时直接返回None，不尝试导入PEFT。返回None既可以表示完成融合，也可以表示没有adapter，不能单凭返回值认证已应用增量。

代码注释称其adapter目录为临时目录，实际路径仍固定为 `target_dir/lora_adapter`，并非新建唯一临时路径；成功后才调用rmtree，没有try/finally包住加载／融合。在匹配真实数据前要核清输出目录归属与失败残留，不应把它描述为自动回滚或异常安全清理。原分离方法对缺失／零alpha元数据发出警告并写0；本机未执行PEFT，不对其数值后果作验收结论。

#### 配置测试不等于数值融合

[本机脚本](../../experiments/verl-lora-export/verify.py)保留候选原函数、配置类、BaseModelMerger类与原测试体，按AST装载以绕过包初始化；原6个配置／CLI用例和2个无adapter早返回用例通过。[结果](../../experiments/verl-lora-export/result.json)列出8项通过、5项未运行及四份原文件SHA256。配置测试中的os.makedirs仍由原patch拦截，类方法测试仍用原MagicMock；没有导出模型或删除任何adapter目录。Python3.12.13、torch2.14.1、pytest9.1.1、macOS arm64。

未运行的5项涉及真实分离保存或HF分支／PEFT替身；其中两项融合测试把 `PeftModel.from_pretrained` 与 `merge_and_unload` 返回对象设为Mock，fused_state也是独立随机张量，断言没有比较它与真实基座＋adapter的数值关系。即使这些调用测试通过，也只说明接线与成功路径清理，不能认证增量正确。候选完整13项也不是本机8项的别称。

本机没有安装PEFT／accelerate，本轮不扩环境或下载模型。进一步验收应在独立CPU小模型与有效adapter元数据下，对比“基座＋adapter”和融合模型的同输入输出，并检查载入键／dtype／容差；单个文件hash变了仍不足以证明数值正确。这一离线缺口不要求访问GPU，不新增M5算力队列或改其专项；上游只读，不应用候选或发布评审。

```bash
/path/to/python experiments/verl-lora-export/verify.py /path/to/verl
```

### 复制分片的存储隔离

离线合并中的`Replicate()`表示取一个副本的值，不自动表示返回值拥有独立存储。[verl #7503](https://github.com/verl-project/verl/pull/7503)将该分支的`return tensors[0]`改为`.clone()`；2026-10-23读取open，作者amanyagami，固定head `e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`、实际祖先 `8df88be746801b6d87c42e15f9a5a0ec1d5eeeae`（ahead2/behind89），两文件104增1删。当前版本只改变复制分支，已撤去此前广泛拒绝共享storage的guard；不能把旧版guard或目标分支的其他变更算进本候选。

[原函数](https://github.com/verl-project/verl/blob/e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b/verl/model_merger/fsdp_model_merger.py)处理的是已从DTensor取出的普通Tensor。`Shard(dim)`仍cat，`Partial()`仍不支持；复制分支仍只选首副本，clone不比较各rank值是否一致。它隔离的是存储所有权，不验证原checkpoint、模型权重绑定或HF文件键完整性。HF保存前另有显式`drop_tied_target_keys`路径，不能仅凭相同storage地址就把两个key判为错误重复。公开评审提出的plain-tensor分支是另一条路径，未给出实际共享storage的checkpoint复现；不把它自动扩张为已证实残余缺陷或另开修复。

#### 共享存储不等于重叠元素

一个缓冲的前四个元素和随后四个元素是两个不重叠view：`data_ptr()`不同，`untyped_storage().data_ptr()`相同。旧分支让输出继续引用源缓冲；更改首view会改源对应元素，但不会改第二view。新分支clone后，两输出与源缓冲各自独立。实际loader的`.bfloat16()`也有条件：输入已是BF16时可原样返回，FP32转BF16会新分配，不能假定所有输入都以相同方式保留别名。

[CPU复查](../../experiments/scripts/verify_verl_merger_storage.py)使用torch2.9.1的真实Tensor和Replicate/Shard/Partial，从固定对象提取整个原merger类，以object替代HF依赖基类，仅调用不读self的合并方法；作者三个测试函数体保持不变。基线2项通过、独立storage断言预期失败；head3项通过。另四组控制覆盖真实torch.save/load的BF16共享view往返、dtype转换、存储字节与源持有者、首副本/Partial规则，两次执行一致。没有原生verl导入、pytest收集、DTensor构造/进程组、完整merger CLI或HF保存；作者19项全套报告不是本机结果。

#### 输出占用与合并峰值

1024元素的BF16源缓冲共2048字节，两段各4元素的输出合计只有16字节有效值。对唯一storage去重计数得到：

| 仍可达的Tensor集合 | 基线共享view | head独立clone |
|---|---|---|
| 只计两个输出 | 2048字节 | 16字节 |
| 源view也仍被持有 | 2048字节 | 2064字节 |

这是实际storage.nbytes的CPU对照，不是RSS、allocator回收或峰值测量。clone可以避免**输出自身**长期保留整个源缓冲，却需要复制时的新分配；完整CLI还要结合源对象何时释放、并行加载、dtype转换和保存阶段判断峰值，不能由第一行直接断言整个合并峰值更低。

[原报告#6259](https://github.com/verl-project/verl/issues/6259)是64卡训练后离线转换出现跨safetensors文件的重复key；作者及公开评审已收窄根因声明。当前CPU控制只证明存储隔离，不复现原文件症状，也不把不同key共享storage等同于文件key重复。已有作者修复覆盖，不新增竞争PR；完整转换首先需要匹配checkpoint及Transformers/HF依赖，离线核验本身不要求重跑64卡训练，不因此新增M5算力任务。

复现：`python -B experiments/scripts/verify_verl_merger_storage.py /path/to/verl-git`，需Python3.12+/torch及本地固定Git对象，无新增依赖。脚本另断言目标base的合并方法AST与祖先相同，完整模块并不相同。原merger SHA256：base `2ae5f3f7fc95871b08a2e185def4f1a5cec09164579cd15eeb2314cb73ef1f0f`、head `29be8e51f7e6bc36a062ff0dbb7a9feb27186a1c3b6cba432c6d8661e0f4d1e1`；原测试 `890253f9060f545948bb9e87cdb8e7bf6d4d8665965bc29e0348db2ac0703c1f`。

## 8. 关键设计洞察

```
1. 2层架构 → 磁盘持久化+实时同步 → 分离关注点!
   → CheckpointManager → 磁盘 → 恢复训练 → 离线合并 → 下游使用
   → CheckpointEngineManager → 实时 → trainer→rollout → 不写磁盘
   → 分离 = 不同需求不同路径 → 磁盘慢但持久 → 实时快但临时 → 设计正确!

2. Per-rank sharded → 不gather完整模型 → 省内存!
   → 每rank保存自己的shard → torch.save → 不需要AllGather
   → Load: ShardedStateDictConfig(offload_to_cpu=True) → CPU加载 → GPU OOM安全!
   → vs consolidated: rank0 gather → 大模型=70GB+ → OOM风险 → sharded更安全!

3. HF export → rank 0 only → 生产部署桥梁!
   → get_fsdp_full_state_dict → rank0_only=True → 只1个rank gather → 省通信
   → init_empty_weights → 空骨架 → 不占GPU内存 → 填入state_dict → save_pretrained
   → safetensors格式 → vLLM/SGLang直接加载 → 生产部署无缝!

4. Megatron async save → 不阻塞训练 → 大模型关键!
   → AsyncCallsQueue → async_save_request → finalize callback → 继续训练
   → ckpt_contents.json最后写入 → completion signal → 检测是否完整
   → vs FSDP: torch.save阻塞 → 大模型=分钟级 → 训练暂停 → 性能损失!

5. CheckpointEngineManager → vs rLLM Tinker → 两种weight sync范式!
   → rLLM: save_checkpoint → disk → new SamplingClient → 磁盘中转 → 简单但慢
   → verl naive: generator零拷贝 → 同进程 → 等价rLLM in-process → 极快
   → verl NCCL: collective send/receive → 跨GPU → 不写磁盘 → 比磁盘快!
   → verl NIXL: RDMA → 跨节点 → 极快 → 需SM90/NVLink → RTX 4090不能用!

6. GRPO checkpoint → 省critic → 50%存储节省!
   → PPO: actor + critic + dataloader → 3×
   → GRPO: actor + dataloader → 2× → 无critic → 省1整套模型存储!
   → 这与GRPO省50%内存/compute一致 → 全系统优化!

7. FSDPModelMerger → 离线工具 → 训练→推理桥梁!
   → 训练产出FSDP shards → 不直接可用于推理 → 需合并
   → merger: ThreadPoolExecutor + torch.cat → 离线合并 → HF format → vLLM直接用
   → 这是训练→推理的完整pipeline → 训练checkpoint → merge → vLLM serve → 闭环!

8. LoRA-only checkpoint需要区分adapter和基座身份
   → 固定a4b93549：save_lora_only且模型有LoRA时，按adapter键规则过滤保存
   → r/alpha/task_type元数据不是base权重hash或恢复完整性证明
   → #8350拒绝缺adapter键，但检查发生在模型写入后，不保证回滚
   → 实际大小取决于目标层/结构/dtype，不沿用旧段未测的固定GB数字
```

---

Sources:
- verl/utils/checkpoint/checkpoint_manager.py (BaseCheckpointManager)
- verl/utils/checkpoint/fsdp_checkpoint_manager.py (FSDPCheckpointManager)
- verl/utils/checkpoint/megatron_checkpoint_manager.py (MegatronCheckpointManager)
- verl/utils/checkpoint/checkpoint_handler.py (CheckpointHandler for SFT)
- verl/checkpoint_engine/base.py (CheckpointEngineManager)
- verl/utils/fsdp_utils.py (FSDP1/FSDP2 state dict helpers)
- verl/trainer/ppo/ray_trainer.py (training loop checkpoint)
- verl/workers/engine/fsdp/transformer_impl.py (FSDP engine save/load)
- verl/model_merger/fsdp_model_merger.py (FSDP shard consolidation)
- Background agent research (verl checkpoint system)
