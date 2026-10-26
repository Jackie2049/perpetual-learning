# verl Worker 生命周期与 Ray 分布式架构源码阅读

2026-10-26接续：[MLflow 初始化的文件锁与异常范围](#mlflow-初始化的文件锁与异常范围)核#8393新f1dd14b：with管理句柄不缩小异常捕获；原三控制仍出现回调ImportError后无锁重试。trainer初始化另无同锁，不把局部串行化当整个启动流程互斥。

2026-10-26接续：[外部 KV 缓存的权重命名空间](#外部-kv-缓存的权重命名空间)核#8323：缓存失效不必等于删除存储，但版本与真实权重必须对应；两个原helper的CPU检查不认证引擎命中、并发门或淘汰效果。

2026-10-26接续：[SGLang 停用后的请求准入](#sglang-停用后的请求准入)区分移除路由、处理已取得的handle、临时暂停与拒绝重试；新#8387已有作者，本轮只做源码／原测试范围核对。

2026-10-26接续：[Rollout 解释器与导入边界](#rollout-解释器与导入边界)对照新#8388与Slime已合并实现：资源分离、Python包隔离和跨栈权重通信是三个不同验收目标；不另写launcher或resolver。

> 2026-06-15 | 源码: verl/single_controller/ + verl/workers/engine_workers.py + verl/checkpoint_engine/ + verl/workers/rollout/
> 核心: WorkerGroup=Ray actor管理+Dispatch路由 → RolloutMode(HYBRID/COLOCATED/STANDALONE) → Sleep/Wake GPU内存切换 → 3种Weight Sync后端(naive/NCCL/NIXL)

## MLflow 初始化的文件锁与异常范围

**初始化互斥、数据库修复和完整trace写入安全是不同问题。** analytistic的[#8393](https://github.com/verl-project/verl/pull/8393)为已有候选：2026-10-26再次核验仍OPEN／非draft／未合并，当前head `f1dd14b0a843570b2542568678ef286d81aaed7f`，目标base `174888fcf4b65dea032566a02fd16c5f76fb5b5b`，两提交、一个生产文件22增1删，无提交测试。旧head `8313d7b6961f399609e3aae34ef739c0cac55cb5` 是首提交；新增提交仅将文件句柄改为上下文管理，三斜杠PR正文与四斜杠源码的差异仍在。当前两条COMMENTED review及三条行内均来自机器人，不作维护者认可。作者两进程及八worker报告仍是外部结果，不认证新版完整启动。

### 句柄关闭不等于异常捕获或全体初始化已修复

[新版原类](https://github.com/verl-project/verl/blob/f1dd14b0a843570b2542568678ef286d81aaed7f/verl/utils/rollout_trace.py)把 `open`／显式close换为 `with open`，仍在outer `try/except ImportError` 内执行锁获取、业务回调和解锁。因此回调抛ImportError仍进入平台fallback，在锁已释放后再次调用。复用下方原三条件、不新增用例，[新版结果](../../experiments/mlflow-init-lock/result-revised.json)的cases与旧JSON逐项相同；新增 `--revision revised` 只选择完整源码hash，原脚本默认模式和旧结果保留。不能因“改成context manager”撤回这条异常边界，也不能说锁完全没有效果：正常回调期间子进程仍被阻挡，返回后可拿锁。

从结构看，with也会在flock获取／释放本身抛错时关闭文件；旧版可能在执行显式close前退出。原三条件没有注入这两类错误，因此结果相同不代表所有失败路径语义相同，不把这次重构说成完全没有作用。

另一个覆盖条件是所有相关初始化者都使用同一锁。[固定新版 Tracking.__init__](https://github.com/verl-project/verl/blob/f1dd14b0a843570b2542568678ef286d81aaed7f/verl/utils/tracking.py#L116)在trainer启用mlflow且未设置MLFLOW_RUN_ID时，直接调用set_experiment再start_run；没有使用旁置文件锁。[V1 fit](https://github.com/verl-project/verl/blob/f1dd14b0a843570b2542568678ef286d81aaed7f/verl/trainer/ppo/v1/trainer_base.py#L424)会构造Tracking。若它与worker初始化重叠并指向同库，worker持有的协作锁不约束trainer。这是源码覆盖范围，不是本机已复现并发迁移；trainer自己的异常重试最终可降级为无MLflow日志，也不能机械推成每次都启动崩溃。

对应[机器人反馈](https://github.com/verl-project/verl/pull/8393#discussion_r4377446960)已经提出trainer侧覆盖问题，不另认领／复制修复。另两条反馈涉及[URI解析](https://github.com/verl-project/verl/pull/8393#discussion_r4377345999)及[锁文件打开](https://github.com/verl-project/verl/pull/8393#discussion_r4377346014)，本轮只确认字面split和open模式仍未改，不执行其建议或新增安全／故障实验。三份完整源码AST／compile及Git blob与GitHub固定tree相符，分别为rollout_trace `5da7ac1e…`、tracking `3107f50b…`、trainer `117c1a98…`；未执行Tracking或完整trainer。

```bash
python experiments/mlflow-init-lock/verify.py /path/to/f1dd14b/rollout_trace.py --revision revised
```

以下原三条件的锁探针方法在新版沿用；旧结果仅代表8313d7b，当前结论对应上面的新版JSON。

首版[AgentLoopWorker调用点](https://github.com/verl-project/verl/blob/8313d7b6961f399609e3aae34ef739c0cac55cb5/verl/experimental/agent_loop/agent_loop.py#L580)让各worker调用[RolloutTraceConfig.init](https://github.com/verl-project/verl/blob/8313d7b6961f399609e3aae34ef739c0cac55cb5/verl/utils/rollout_trace.py)。`_instance`／`_initialized`只属于各进程自己的Python类，不能充当跨worker屏障。两版都对字面以sqlite:开头的URI打开旁置 `.lock` 文件，用阻塞LOCK_EX仅包住set_experiment，最后解锁并关闭；enable_async_logging和set_tracking_uri在锁外，后续start_span／set_trace_tag也不在这段锁内。代码没有超时、数据库回滚、损坏schema检测或修复步骤。

默认URI实际为 `sqlite:////tmp/mlruns.db`，四斜杠对应绝对路径 `/tmp/mlruns.db`，锁为 `/tmp/mlruns.db.lock`；不能照PR正文的三斜杠当成同一路径。候选用 `split("///", 1)[1]` 和字符串追加来生成锁名，不是SQLAlchemy URL解析。[SQLAlchemy官方连接规则](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#connect-strings)区分相对／绝对路径、内存数据库及查询参数。因此要按真实数据库身份评估锁是否一致：例如query被原样拼入锁名，SQLAlchemy驱动前缀sqlite+pysqlite:则不进入此startswith分支；sqlite://连分割的第二项都没有。这是源码／官方格式对照，不代表已验证这些URI在特定MLflow安装中可用，也不扩写通用解析补丁。

### 原类与真实子进程锁探针

[有界脚本](../../experiments/mlflow-init-lock/verify.py)提取完整原RolloutTraceConfig类、保留原方法体，MLflow仅用显式记录／异常注入回调替代；fcntl是真实系统调用，锁争用由独立Python子进程的LOCK_EX|LOCK_NB及EACCES／EAGAIN判断，不靠sleep或时间差猜测。Python3.12.13／Darwin，使用新临时目录，未访问默认数据库；原完整模块AST／compile通过。

| 注入条件 | 回调期间另一进程的拿锁结果 | 返回状态 |
| --- | --- | --- |
| 成功，再调用同进程init | 仅一次回调，blocked | initialized=True；重复init直接返回 |
| 首次回调抛RuntimeError | 一次回调，blocked | 原错误传播，initialized=False |
| 首次回调抛ImportError，第二次成功 | 第一次blocked，第二次acquired | 同一次init发生两次回调，最终initialized=True |

第三项源于outer `except ImportError`覆盖整个加锁和回调段，不仅是import fcntl。首次回调的finally已释放锁，catch随后无锁重试；条件是**回调自身抛ImportError**，并非已证明实际MLflow正常启动必然如此。三组结束后子进程均可拿锁，而.lock文件仍在，说明文件存在不等于锁仍被持有；也没有必要据此删除共享锁文件。RuntimeError路径的config字段已在回调前赋值，initialized=False不等于所有状态回滚。

[原始结果](../../experiments/mlflow-init-lock/result.json)固定完整源码SHA256及commit；重现命令为 `python experiments/mlflow-init-lock/verify.py /path/to/pinned/rollout_trace.py`。没有导入真实MLflow／Alembic／SQLAlchemy、建立SQLite库、启动Ray／AgentLoopWorker或GPU，不把该控制写成迁移race复现／修复验收。LOCK_EX与非阻塞错误解释见[Python官方fcntl文档](https://docs.python.org/3.12/library/fcntl.html#fcntl.flock)。

候选适用性仍取决于所有初始化者使用同一锁文件及实际文件系统锁语义；作者已限定单机SQLite，多节点应接其MLflow服务路线，不从本机Darwin探针推广共享文件系统。后续仅等待作者／维护者反馈或真实用户场景，不实施另一套锁、数据库清理或错误重试方案，不安装依赖／启动服务，也不新增M5算力任务。任何上游评论、PR或既有PR-head推送仍须用户逐项明确授权。

## 外部 KV 缓存的权重命名空间

### 新版的显式清理与 sleep 路径

2026-10-26再核[#8323](https://github.com/verl-project/verl/pull/8323)：仍OPEN／非draft／未合并，新head `1787d6a29250365da07c13e2ab1331a8a3a9cdcc`，base／实际祖先 `174888fcf4b65dea032566a02fd16c5f76fb5b5b`，ahead2／behind0，20文件872增21删。与旧4e93 head已经分叉（ahead23／behind2），整仓比较混入多项main合并，不能全部算本题新增；旧CPU helper结果不迁作新20文件验收。

作者新增[sleep说明](https://github.com/verl-project/verl/pull/8323#issuecomment-6416461545)，维护者另问是否将部分修复放在引擎侧。**salt隔离查找身份，不保证外部缓存经完整生命周期始终保留。** 固定[候选server](https://github.com/verl-project/verl/blob/1787d6a29250365da07c13e2ab1331a8a3a9cdcc/verl/workers/rollout/vllm_rollout/vllm_async_server.py)与vLLM0.32.0 commit `5df63a7203cdb9328d824a27b682e77c7b9928be` 的[AsyncLLM](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/async_llm.py)、[EngineCore](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py)显示三条不同路径：

| 入口 | salt模式的实际接线 | 能说明什么 |
| --- | --- | --- |
| 显式clear／wake／resume中的 `_reset_prefix_cache` | 传reset_connector=False | 这一次调用不请求重置外层，不能约束其他入口 |
| abort并要求清cache | vLLM≥0.32用mode=abort，salt先clear_cache=False，再经clear_kv_cache仅清本地 | 对新版pause API有适配，不是原生引擎已实测 |
| sleep／release_kv_cache借用sleep | level≥1经pause(clear_cache=True) → `_finish_pause` → `_reset_caches`默认reset_connector=True | 绕过上面的False；free_cache_engine及rollout_mode决定是否走到该入口 |

[checkpoint manager](https://github.com/verl-project/verl/blob/1787d6a29250365da07c13e2ab1331a8a3a9cdcc/verl/checkpoint_engine/base.py)非naive路径在abort后、传权重前调用release_kv_cache_replicas；server对非COLOCATED路径借sleep后仅唤醒weights，正常HYBRID sleep也走level1／2。vLLM0.32另已有release_kv_cache_memory，但该方法自身也调用默认 `_reset_caches()`，不能只凭名字就替换成“保留connector”的修复。

[scheduler消费点](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L2840)还区分**请求reset**与**外层实际清空**：只有reset_connector=True才调用connector.reset_cache，具体效果取决于连接器实现；没有connector是no-op成功。因此sleep可能减少外层复用，但不能称所有store都被清空。若正确绑定新权重版本并按带盐key寻址，即使未清外层，旧版本仍不应命中；反之“命中率下降／无命中”不证明版本绑定正确。共享store的性能及其他副本影响仍需对应环境观察，不把作者“只会over-clear”的说明扩大成所有布局验收。

本轮四份完整Python源码仅AST／compile及上述scheduler方法阅读；没有执行sleep、pause、原CPU／GPU套件、connector、模型或权重更新。作者vLLM0.29／SGLang0.5.20／TRT1.3rc15报告不作为0.32新组合结果，不造FakeEngine清理矩阵。缺口仅更新[原M5条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)：未来已有获准匹配观测时，连同sleep调用点、connector reset实现与命中计数分别记录；不自动切salt、关闭free_cache_engine、改引擎或新排部署。

### 旧版身份与请求接线记录

**旧权重生成的KV不能用于新权重；让旧缓存不再被查到，不等于把它从存储中删除。** 已有daegyu94的[#8323](https://github.com/verl-project/verl/pull/8323)将外部缓存失效拆成默认 `reset` 与可选 `salt`。2026-10-26核为OPEN／非draft，固定head `4e93e4761ddde2ea5bf9d3032b86b6c3844126fd`、直接父版19835b21，PR元数据base16ec093c，20文件869增20删。本轮聚焦六个运行模块和一个原CPU测试，不是整个PR或三种推理引擎的完整验收，也不另开竞争实现。

### 缓存身份与存储回收

[`prefix_cache_salt`](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/workers/rollout/replica.py)在salt模式返回 `namespace:global_steps`，reset模式返回None；RolloutReplica在namespace尚为None时填入创建进程共享的模块UUID。默认新driver进程得到新namespace，恢复到相同步数也不复用上一进程的键；已显式填写的namespace则保留。salt不是权重内容hash，也没有证明两个副本的参数已逐字节一致。

按候选[配置检查](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/workers/config/rollout.py)，vLLM仅接受列出的按block hash寻址的connector：MooncakeStore、Offloading、LMCacheMP和HF3FS；MultiConnector递归检查孩子，按token查找的LMCacheConnectorV1被拒绝。检查同时接受 `kv_transfer_config` 与 `kv-transfer-config` 两种拼写，支持JSON字符串。它检查配置名称，不加载插件或证明具体部署确实按该hash查找。salt还拒绝PD disaggregation，不能把prefill／decode跨进程拆分当已覆盖。

候选[vLLM重置方法](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L953)在salt模式传 `reset_connector=False`，这条显式路径不请求清外部存储，仍请求本地cache清理；它不覆盖sleep的其他清理路径。旧版本条目占用仍需后端淘汰策略与容量管理；换键不自动回收空间，也不保证吞吐提升。TRT-LLM这里只传cache_salt，不由verl配置外部存储；SGLang HiCache文件存储的失效缺口与命中结果仍是作者报告，不推广到所有tier。

### 版本必须在正确的请求时点绑定

vLLM先过 `_park_until_admitted`，再给prompt加salt；TRT-LLM先过 `_park_until_allowed`，再传salt关键字。SGLang在salt模式增设verl侧Event，等待它恢复后才给GenerateReqInput赋salt，避免请求先带旧salt进入tokenizer manager等待、恢复后却运行新权重。源码明确依赖SGLang0.5.20在该调用间隔的调度语义；原CPU测试使用FakeTokenizerManager，本轮不据它认证真实并发时序。

vLLM的准入屏障若超时，salt模式抛错阻止继续更新，reset模式仍可警告后继续；暂停时salt先避免引擎顺带重置connector，再单独清本地cache。这是控制流策略，不是已验证故障后的权重回滚协议。[CheckpointEngineManager](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/checkpoint_engine/base.py#L510)在非naive更新路径只检查global_steps是否为None，不校验新编号、单调性或权重内容；naive分支在该检查之前返回，作者解释其无步数调用用于同权重的validation唤醒。因此正确性仍要求调用方让权重变化与新版本对应，不能把“传了一个非空步数”当身份一致性证明。

### 两个原 helper 的有界检查

[复查脚本](../../experiments/kv-cache-salt-contract/verify.py)强校验七个源码副本hash并做完整AST／内存compile，只执行原salt格式化函数与connector检查函数；SimpleNamespace仅提供两个必需字段，没有构造RolloutConfig、引擎或假缓存。Python3.12.13标准库下，[原始结果](../../experiments/kv-cache-salt-contract/result.json)确认：同run的3／4步为 `run-a:3`／`run-a:4`，另一run的3步为 `run-b:3`；重复3步仍是同键，helper直接收到None会格式化成 `run-a:None`，不替调用层检查版本。namespace为空则AssertionError。后两点说明职责边界，不单独证明正常trainer会错误复用缓存。

另按原测试的四个connector输入、两种拼写做8次原helper调用：两个允许项接受、LMCacheConnectorV1及包含它的MultiConnector拒绝。没有运行原配置类／原CPU测试套件、暂停协程、Ray、cache RPC、模型或设备；格式化结果不是缓存hit／miss证据。输入须为固定head的源码副本：

```bash
python -B experiments/kv-cache-salt-contract/verify.py /path/to/pinned-source
```

作者报告RTX PRO 4000上的vLLM0.29／SGLang0.5.20／TRT-LLM1.3.0rc15小模型对照，以及HiCache file在reset下仍命中384／400 tokens、salt下0命中；这是外部报告，不是本轮测量，也不等于20文件均完成集成回归。另一个[#8390依赖升级](https://github.com/verl-project/verl/pull/8390)面向vLLM0.32并改pause API，不能未经接线核对就把两份候选或其报告叠成已验证组合。缺少的真实版本／请求绑定／跨副本与淘汰证据只补[原M5条件参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不新排部署或训练任务。

## Rollout 调度等待与引擎就绪

**暴露调度失败，不等于预留资源或证明引擎已就绪。** 2026-10-26核对[#7381](https://github.com/verl-project/verl/issues/7381)与[#7383](https://github.com/verl-project/verl/issues/7383)：报告者在两节点、每节点一卡的7B colocate运行中观察到一个vLLM server actor长期PENDING_CREATION，0.5B同布局却成功；明确没有保存其required_resources。报告将挂起定位到首次actor方法结果等待处，不足以证明确定的GPU短缺、所有一卡拓扑不受支持，或NCCL就是根因。#7383的资源预留／分离部署／限制拓扑仍是待维护者裁定的方向。

已有OPEN/non-draft [#7382](https://github.com/verl-project/verl/pull/7382)，固定head `8ebc3558c39d5192a6bf312a806c889a2197f293`，父版／共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，相对当轮main `3c6d9b43` ahead1／behind5，五文件131增3删。不另实现超时helper或PG方案。主线[vLLM创建路径](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1534)仍硬节点亲和、随后等待get_master_address；这个调用点没有显式num_gpus请求，不能把原CPU测试字典中的GPU=1当真实actor资源需求。

### Guard 实际覆盖什么

候选在vLLM、SGLang和TRTLLM创建server actors后、首次启动相关RPC前插入[wait_rollout_servers_scheduled](https://github.com/verl-project/verl/blob/8ebc3558c39d5192a6bf312a806c889a2197f293/verl/workers/rollout/replica.py#L45)。缺省300秒、轮询间隔2秒；只有PENDING_CREATION／DEPENDENCIES_UNREADY留在pending，查不到actor的None也按pending处理。它不是“每个actor都为ALIVE”的断言，更不检查模型加载、HTTP服务或生成请求成功。查不到状态也不单独证明节点没空位。

循环先同步逐个调用get_actor，再决定是否超时；不传每次查询的剩余预算，没有把整个循环包在可取消的超时器内。若没有pending则直接返回，不再检查deadline。因此300秒是pending轮询的检查阈值，不是整个LLMServerManager.create／引擎启动的硬上限。helper也没有reserve、取消／kill actor或部分启动回滚；报错文本提供建议，不等于已实施这些操作。

CodeRabbit在同一行内评论提出State API依赖和阻塞两项担忧，review状态为COMMENTED，不是批准。独立核对[Ray 2.41.0官方get_actor](https://github.com/ray-project/ray/blob/ray-2.41.0/python/ray/util/state/api.py#L554)确为普通同步函数，构造REST StateApiClient并有独立timeout；[默认值为30秒](https://github.com/ray-project/ray/blob/ray-2.41.0/python/ray/util/state/common.py#L47)。候选未捕获查询异常，它们不会变成None分支的调度诊断。这个固定版本源码核对**不认证报告／部署安装的Ray版本，也不证明任意include_dashboard=False配置必然失败**；后者需区分UI开关、State服务可达性及具体安装，不能直接照抄机器人判断。

### 四个原测试体的验证范围

[复现脚本](../../experiments/verl-server-schedule-guard/verify.py)原样AST装载helper／两常量和原四协程测试体，保留原get_actor替身：ALIVE立即返回、两次pending后ALIVE、持续pending的诊断异常、None持续缺失的异常；四项两次通过且[JSON](../../experiments/verl-server-schedule-guard/result.json)一致。真实asyncio／monotonic／sleep与pytest MonkeyPatch／raises，Python3.12.13、pytest9.1.1；不是原生pytest收集，也没有Ray State HTTP、真实actor或调度器。

五份完整模块AST／compile通过，三调用端各一个await接线仅AST核对，未运行其启动路径。测试里人工填的node0／GPU=1不能补足原报告的资源证据；未验证API异常、所有状态、清理、调度修复或GPU引擎，不继续扩替身矩阵。本题是现有候选的范围审计，不新排M5算力、不改专项或发布反馈；若后续确有获准运行，应先捕获同版真实actor资源／节点快照再判断根因，不能据这四项控制调整GPU布局。另见[组初始化等待与退出](verl-async-logprob-provenance.md#组初始化的等待超时与任务退出)，两者都不能把等待失败自动视为资源已回收，但属于不同启动阶段。

## SGLang 停用后的请求准入

**移除路由不等于撤销已经发出的actor handle。** donghyq的draft [#8387](https://github.com/verl-project/verl/pull/8387)处理这样的窗口：客户端刚取得hybrid副本的handle，控制器随后移除该副本并归还GPU；请求晚一步进入已暂停的SGLang，旧版会在native manager内等待源副本恢复，而不是退出后重新路由。这与已进入引擎的请求如何保留KV、进程重建或版本盐缓存隔离是不同阶段。

2026-10-26读取OPEN／draft，head `df92e6a205b8d8f6f111fd6b4c007d4a4e38f17a`、实际父版／共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`；API目标base `0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b`并非候选已包含的祖先，相对它ahead1／behind4。相对祖先四文件340增14删，已有实现及原测试，不另写竞争gate。

### 临时暂停和停用采用不同准入行为

[候选server](https://github.com/verl-project/verl/blob/df92e6a205b8d8f6f111fd6b4c007d4a4e38f17a/verl/workers/rollout/sglang_rollout/async_sglang_server.py)在generate的第一条语句等待准入判断，位于PD派发和GenerateReqInput构造之前。状态由submission_paused、rejecting与resume_event表达：

| 状态 | 新到请求 | 事件作用 |
|---|---|---|
| 正常 | 继续进入原生成路径 | event设为可通过 |
| 临时pause，reject_request=False | 在verl外层等待，不先进入native manager | clear；成功resume后唤醒 |
| 停用，reject_request=True | 返回空token、stop_reason=aborted及global_steps | set；唤醒此前停在临时pause的请求，让它们也重新判断 |

因此event被set并不总表示“可生成”；被唤醒的协程还会检查暂停／拒绝状态。后续新的临时pause也会重新clear，不能把reject_request当永久锁死开关。这里的global_steps是返回标签，不是“没有观察过混合权重”或内容身份的认证。

pause／resume用同一个asyncio.Lock串行控制，但请求并不持有这个锁完成整个生成。原pause调用前先关闭外层准入；其BaseException路径将状态保持为暂停并拒绝。已经关闭准入时，native resume失败不会执行后面的重新开放，成功后才清除两状态。它不是原子回滚native状态、GPU释放或全副本分布式屏障，也不替代native层处理早已进入引擎的请求。仅从catch BaseException可推断代码意图覆盖取消，不代表原测试已经验证所有取消／竞态。

### 先停止路由再处理迟到请求

[V1 switch_to_trainer](https://github.com/verl-project/verl/blob/df92e6a205b8d8f6f111fd6b4c007d4a4e38f17a/verl/trainer/ppo/v1/trainer_separate_async.py)原来就按remove_servers → abort_replicas(reject_request=True) → sleep_replicas → 设置TRAINER模式执行；[manager](https://github.com/verl-project/verl/blob/df92e6a205b8d8f6f111fd6b4c007d4a4e38f17a/verl/checkpoint_engine/base.py)也已经传递该flag，缺口在SGLang replica旧实现只告警并忽略。候选改为向node-rank 0的primary server透传，避免对多节点每个server广播native控制。legacy dynamic controller则新增True参数；只改controller不能让旧SGLang server获得拒绝能力。

若pause抛错，V1顺序调用不会走到sleep与模式修改；这表示该切换没有顺利完成，不等于路由已自动恢复或所有副本仍处于同一状态。[现有FullyAsync客户端](https://github.com/verl-project/verl/blob/df92e6a205b8d8f6f111fd6b4c007d4a4e38f17a/verl/workers/rollout/llm_server.py)可在aborted后延迟1秒重新acquire，保留已返回token前缀；实际partial_rollout配置仍决定是否重试。普通base客户端没有这层循环。only_hybrid且LB为空时会继续等，其他配置可抛无可用server，所以**返回aborted只是给出重试机会，不保证存在peer、固定故障恢复时限、KV迁移或相同随机轨迹**。

### 原测试和目标环境的验收范围

新增文件含12个测试函数，其中replica flag的False／True参数组使执行项为13：七项server准入／控制、两项flag转发、一项legacy顺序、三项V1链路。原TokenizerManager、Ray responses与load balancer均用本地记录器；通过object.__new__绕过真实server／trainer初始化。V1三项能检查原方法的调用顺序、pause失败不sleep与disabled-hybrid早退，不构成真实多actor、GPU释放或peer重试验收。源码中的两个sleep(0)只安排该夹具的协程，不证明所有交错都安全。本轮未执行这13项，不新增替身测试矩阵。

作者报告Linux／Python3.12.11正常imports下13过，以及同机双A100、TP1 synthetic Qwen2、真实Mooncake TCP的固定权重probe：combined修改经peer完成、controller-only仍等待。该报告明确不是完整hybrid Trainer或吞吐benchmark，且CPU资格绑定a4b93549加四文件，目标main／default-lock CI仍待完成，不能计为本机实测。

祖先、候选、目标的V1 trainer／manager／客户端三个完整文件逐字节相同，两个生产修改文件在祖先与目标也相同；但现有GPU workflow的镜像从祖先dev4变为目标dev5。候选新增显式CPU pytest step并对该step设CUDA_VISIBLE_DEVICES为空，仍位于原GPU job及其SGLang／Megatron环境，并没有建立独立轻量CPU runner或证明Actions已执行。不能把源码可接入、作者旧基线通过和目标锁定依赖通过合成一项结论。

本轮六完整Python模块AST／compile、generate首条gate接线与YAML BaseLoader结构检查通过，无SGLang／Ray imports、原测试／引擎或故障注入。源码hash见[当日日志](../../diary/2026-10-26.md)；仅向[M5原一致性入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)补条件参考，不新排pause／resume实验、不改专项或采用候选。所有上游写入与既有PR-head推送仍逐项待用户明确同意。

## Rollout 解释器与导入边界

**另一个进程不自动意味着另一个包环境，另一个解释器也不自动意味着可跨版本同步权重。** ZihaoW123的[RFC #8388](https://github.com/verl-project/verl/issues/8388)于2026-10-26提出，读取OPEN／0评论，基于main`0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b`。它先限定standalone vLLM、FSDP2、separate_async且关闭hybrid切换；字段拼写、resolver、preflight和兼容矩阵均是提案，不是已实现能力。作者明确未保留各实现阶段的所有权；已有设计不代表实现被覆盖，0评论也不表示无需协调。

### 解释器要覆盖整个执行子树

固定源码展示的不是单个server启动点：

| 边界 | 当前路径 | 为什么只改server不够 |
|---|---|---|
| 控制侧导入 | [replica注册器的_load_vllm](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/workers/rollout/replica.py)导入vllm_async_server里的vLLMReplica | 同模块顶层导入vLLM多个子模块，尚未启动远端actor也会碰后端依赖 |
| rollout权重接收端 | standalone创建RayWorkerGroup，使用[CheckpointEngineWorker](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/checkpoint_engine/base.py) | worker构造器会创建rollout adapter及通信组，也需要相应后端依赖 |
| HTTP server actor | [vLLMReplica.launch_servers](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/workers/rollout/vllm_rollout/vllm_async_server.py)另设node affinity、runtime_env.env_vars与并发度 | 不走同一个worker-group创建入口，不能只在WorkerGroup覆盖解释器 |
| 引擎与TP／DP子进程 | server启动AsyncLLM／headless等后端入口 | actor的sys.executable只是第一层证据，子进程实际继承／重启行为还须按后端版本验证 |

同文件[RolloutReplica.init_hybrid](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/workers/rollout/replica.py)直接复用已初始化训练worker；同一进程内的两个角色不能各用一个Python解释器。init_colocated虽然另开进程，当前也没有因此自动选择另一套包；GPU placement、内存共享与Python依赖是不同维度。

[全局ppo配置](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/trainer/config/ppo_trainer.yaml)已有ray.init.runtime_env.py_executable并注释其被TaskRunner与worker继承，但它不能替换已经运行的driver。新提案要让训练侧不装vLLM，还须把controller-facing定义与模块级后端导入分开；仅把类名写成字符串，若launch前仍解析并导入原模块，也没有跨过导入边界。

### 不混淆本地选项覆盖与 Ray 继承

[RayClassWithInitArgs.update_options](https://github.com/verl-project/verl/blob/0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b/verl/single_controller/ray/base.py)直接做_options.update；后续_create_worker重新写入runtime_env，包含内部rank／master变量，CUDA profiling时另含nsight。若较早把完整runtime_env放在同一class options中，后写字典会替换它，不是按字段深合并。**这不等于Ray从父job继承的runtime_env必定全部丢失**：两者属于不同层，不能用普通dict实验代替Ray继承验收。

RFC因此提出在两个actor创建边界统一解析角色覆盖，保留code distribution／profiling／placement和内部设备变量，拒绝冲突的pip／conda／uv规格。当前_create_worker已经拒绝用户覆盖其保护变量；“用户env overlay”不能被解读成可改RANK／WORLD_SIZE。[Ray环境文档](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management)说明py_executable可选择worker解释器及命令参数，父uv环境会被子任务继承，Ray／Python版本还需匹配cluster。该页面读取为2.59.0，而固定verl锁为Ray2.55.1；它支持概念边界，不能算本项目锁定栈已验证这些组合。

### Slime 已有实现隔离的是服务子进程

[Slime #2484](https://github.com/THUDM/slime/pull/2484)已于2026-10-19合并，固定head`b52055443810ba37e4f66d89e3316b4ae1ca1365`、merge`a4202e0595467d9374d9a68d45a34f45c3f8d230`。其[runtime原模块](https://github.com/THUDM/slime/blob/b52055443810ba37e4f66d89e3316b4ae1ca1365/slime/backends/sglang_utils/runtime.py)由普通Popen启动`[executable, '-m', 'slime.backends.sglang_utils.runtime']`，通过stdin传JSON，在子进程重建ServerArgs，而不是把父进程的Torch对象交给multiprocessing序列化。

这里executable是一个argv元素，文档用绝对Python路径或保留参数的wrapper；它不是Ray py_executable可带参数的命令前缀，不能机械搬用同一种字符串。JSON只承载启动参数，不是权重数据面的替代协议。环境从父os.environ复制，针对allocator和Mooncake端口做有限处理，不是隔离全部PATH／PYTHONPATH／动态库搜索路径的安全沙箱。

[Slime文档](https://github.com/THUDM/slime/blob/b52055443810ba37e4f66d89e3316b4ae1ca1365/docs/en/advanced/sglang-config.md#independent-sglang-python-environment)明确Ray控制actor／router仍用训练环境，训练侧仍需SGLang Python helper与router包，并要求对齐SGLang版本；服务侧可不装Megatron。这比“训练环境完全没有推理后端”窄。[engine_group改动](https://github.com/THUDM/slime/pull/2484/files)还要求共享GPU退役时确认服务退出，失败／超时停止交接，不是父actor退出就认证子进程已释放GPU。模块里的终止函数只作源码阅读，本轮未执行任何kill或进程树清理。

### 包隔离和跨栈同步分两道验收

verl的TensorMeta仍含torch.Size与torch.dtype，接收端按dtype.itemsize和chunk.view重建权重；切换解释器不自动提供跨PyTorch序列化、CUDA IPC／NCCL／NIXL或动态库ABI兼容。RFC先要求相同Python／Ray／verl revision，并以相同核心栈的两个环境证明选择与导入隔离，再对明确的transport／package tuple做generate→train→sync→generate及checkpoint／清理验收。不同CUDA用户态库也不等于隔离宿主GPU driver。Slime已有磁盘替代建议不应被悄悄移植成verl的新默认权重路径。

本轮四份verl完整Python模块及Slime runtime完整原模块AST／compile通过，另读配置／lock、Slime相关patch／文档及Ray一手说明；没有安装新环境、调用resolver／Popen／Ray、运行Slime原测试或设备通信。hash见[当日日志](../../diary/2026-10-26.md)。硬件与端到端缺口只补[M5原条件入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)的参考，不接单部署／实现新launcher、不影响专项；上游写入及既有PR-head推送仍逐项需用户明确同意。

## 1. WorkerGroup 类层次

```
WorkerGroup (base/worker_group.py:123-255)
  ├── _workers: list[worker_handle] → Ray actor引用
  ├── _worker_names: list[str] → worker标识
  ├── _dispatch_info / _collect_info: dict → lazy dispatch注册
  ├── world_size = len(self._workers)
  ├── _bind_worker_method(): 扫描@register装饰的方法 → 动态绑定到WorkerGroup

RayWorkerGroup (ray/base.py:416-909)
  ├── _init_with_resource_pool(): 创建Ray PG → 每bundle 1 Ray actor
  ├── execute_all() / execute_all_async() / execute_all_sync()
  ├── execute_rank_zero() / execute_rank_zero_async()
  ├── spawn(prefix_set): 从现有workers创建detached子WorkerGroup
  ├── spawn_fused(prefix_set): 为FusedWorker创建WorkerGroup(colocated)
  ├── fuse(prefix_set): 合并多个WorkerGroup → ColocateWorkerGroup
```

### Worker创建流程

```python
# RayWorkerGroup._init_with_resource_pool()
for bundle in placement_group.bundles:
    worker = RayClassWithInitArgs.__call__()  # 创建Ray actor
    # num_gpus = 1 / resource_pool.max_colocate_count  → 分数GPU for colocation!
    # max_colocate_count=3 → 每actor得1/3 GPU → 3个WorkerGroup共享同一GPU
```

## 2. Dispatch 路由机制

### Dispatch 模式枚举

```python
# decorator.py:26-47
Dispatch.register("RANK_ZERO")        # 只有rank 0执行
Dispatch.register("ONE_TO_ALL")       # 相同参数广播到所有workers
Dispatch.register("ALL_TO_ALL")       # 参数直接传递(已per-worker)
Dispatch.register("DP_COMPUTE")       # 参数已按DP rank分好; 输出per-rank
Dispatch.register("DP_COMPUTE_PROTO") # 拆分DataProto by DP size, 自动padding, concat结果
Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC") # 拆分DataProto, 传function到每个worker
Dispatch.register("DP_COMPUTE_METRIC") # 拆分DataProto, 收集per-rank(不concat)
Dispatch.register("DIRECT_ROLLOUT_METHOD") # 禁止! 必须通过rollout replica调用
```

### @register 装饰器工作流

```python
# decorator.py:398-444
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=True)
def generate_sequences(self, data: DataProto):
    ...

# _bind_worker_method()处理:
# 1. 读取MAGIC_ATTR → {dispatch_mode, execute_mode, blocking}
# 2. 从DISPATCH_MODE_FN_REGISTRY获取 dispatch_fn + collect_fn
# 3. 选择execute_fn (execute_all / execute_rank_zero)
# 4. 创建Functor: dispatch_fn(args) → execute_fn(remote_call) → collect_fn(output)
```

### 关键Dispatch实现

```
ONE_TO_ALL (行120-123):
  args = tuple([arg] * world_size for arg in args) → 所有worker相同参数
  → 用于: init_model, reset, save_checkpoint (所有DP worker做相同事)

DP_COMPUTE (行148-156):
  args必须已是list[world_size] → 每worker拿对应slice
  → 用于: 已经按DP rank组织的数据

DP_COMPUTE_PROTO (行167-199):
  DataProto.split(world_size) → 自动padding确保均匀 → 每worker1块
  collect: _concat_data_proto_or_future → 合并回完整DataProto
  → 用于: generate_sequences, update_policy, compute_log_prob (最常用!)

ND Compute (行266-304):
  make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")
  → 查询worker_group._dispatch_info[mesh_name]
  → 根据dp_rank_mapping映射DataProto到正确worker
  → 支持非均匀DP-TP-PP拓扑!
```

### engine_workers.py中的Dispatch使用

| 方法 | Dispatch模式 | 原因 |
|------|-------------|------|
| `init_model` | ONE_TO_ALL | 所有worker初始化相同模型 |
| `compute_log_prob` | ND_compute(mesh="actor") | 按actor DP拓扑分数据 |
| `update_actor` | ND_compute(mesh="actor") | 按actor DP拓扑 |
| `compute_ref_log_prob` | ND_compute(mesh="ref") | 按ref DP拓扑 |
| `train_mini_batch` | ND_compute(mesh="train") | 按训练引擎DP拓扑 |
| `save/load_checkpoint` | ONE_TO_ALL | 所有worker保存/加载相同 |

## 3. RolloutMode: 3种部署模式

```python
# replica.py:54-68
class RolloutMode(Enum):
    HYBRID = "hybrid"       # Rollout+Training 同进程
    COLOCATED = "colocated"  # Rollout+Training 同PG但不同进程
    STANDALONE = "standalone" # 分离GPU资源, disaggregated
```

### HYBRID模式 (init_hybrid, 行131)

```
特征: Rollout引擎和FSDP训练引擎共享同一Ray actor/进程
  → FusedWorker包装actor+rollout进1个进程per GPU

GPU内存管理:
  → Sleep/Wake必需: rollout必须sleep(释放GPU) → training使用GPU → wake_up恢复
  → sleep_level=2(full weights) 或 sleep_level=1(LoRA adapters)

Weight Sync:
  → in-process → naive/ColocatedCheckpointEngine → Python generator!
  → self.actor.engine.get_per_tensor_param() → generator of (name, tensor)
  → yield from self.weights → 同进程零拷贝!

适用: on-policy PPO训练 (最高效, 最简单)

RTX 4090:
  → HYBRID最优! 单进程→无IPC/Ray overhead
  → LoRA→sleep_level=1→只释放KV cache→更快wake_up
  → GRPO→无critic→actor+rollout共用GPU→HYBRID完美!
```

### COLOCATED模式 (init_colocated, 行160)

```
特征: Rollout和训练引擎共享同一Ray Placement Group(同GPU)但不同进程
  → CheckpointEngineWorker作为独立Ray actor per GPU

GPU内存管理:
  → sleep_level=1 only (只释放KV cache → 为NCCL weight buffer腾空间)
  → 不需要sleep_level=2! (rollout进程保留weights → NCCL直接写入)

Weight Sync:
  → CUDA IPC → BucketedWeightSender/Receiver → ZMQ socket + torch.multiprocessing IPC
  → 流程: CheckpointEngine进程 → ZMQ IPC → vLLM进程 → GPU tensor直传
  → bucket_size=2048MB → 多个weight打包成1个bucket → 减少传输次数

适用: GRM(LLM as judge), rollout需要独立进程但同GPU

RTX 4090:
  → 可行但不如HYBRID→IPC开销(虽然同一GPU但跨进程)
  → 适合需要vLLM独立进程的场景(如MoE模型)
```

### STANDALONE模式 (init_standalone, 行189)

```
特征: Rollout有自己的ResourcePool → 独立GPU集
  → 新ResourcePoolManager → 独立Placement Groups

GPU内存管理:
  → 无Sleep/Wake! (rollout有自己的GPU → 不需要释放)

Weight Sync:
  → NCCL/NIXL checkpoint engine → 跨GPU集传输
  → NCCL: 建立ProcessGroup → AllReduce/Broadcast weight buckets
  → NIXL: RDMA P2P ring → 双buffer(send_buf+recv_buf) → swap per bucket

适用: off-policy训练, disaggregated架构(推理和训练分离GPU)

RTX 4090:
  → 不适用! 需要至少2×GPU集 → PCIe带宽不够NCCL weight sync
  → NVLink集群→可行但不如HYBRID高效
```

## 4. Sleep/Wake机制详解

### VLLM_SLEEP_LEVEL

```python
# third_party/vllm/__init__.py
VLLM_SLEEP_LEVEL = 1  # 默认
# vLLM >= 0.8.5: VLLM_SLEEP_LEVEL = 2
# Ascend NPU: 强制=1 (sleep_level=2不支持)
```

```
Level 1: 释放KV cache和部分模型权重 → LoRA-safe → 只释放adapter?
  → 保留模型config和部分metadata → wake_up更快

Level 2: 释放所有(KV cache + 全部模型权重 + 更多)
  → 完全清空GPU内存 → training可以全量使用 → wake_up需要重新加载weights
  → HYBRID full weight sync必须level=2 → LoRA mode只需level=1
```

### HYBRID sleep/wake流程

```python
# vllm_async_server.py:937-955
async def _sleep_hybrid(self):
    if self.lora_as_adapter or is_torch_npu_available():
        sleep_level = 1  # LoRA只需level 1 → 不释放base weights
    else:
        sleep_level = 2  # Full weight sync需要level 2 → 完全清空
    await self.engine.sleep(level=sleep_level)

# vllm_async_server.py:604-623
async def wake_up(self):
    # HYBRID: wake_up(tags=["kv_cache", "weights"]) → 恢复全部GPU内存
    # COLOCATED: wake_up(tags=["kv_cache", "weights"]) → weights已通过IPC更新
    # STANDALONE: skip → 不需要wake_up(有自己的GPU)
```

### 完整Rollout-Training切换时序

```
HYBRID模式 (一个step的GPU内存生命周期):

  1. Rollout阶段: vLLM awake → 占据全部GPU(weights+KV cache)
     → generate_sequences → rollout完成 → 收集responses

  2. sleep(level=1 or 2):
     → vLLM释放GPU内存 → FSDP训练引擎可以使用!
     → Level 1: 释放KV cache → FSDP可以做forward+backward(但weights仍在!)
     → Level 2: 释放KV cache+weights → FSDP独占全部GPU!

  3. Training阶段: FSDP引擎活跃
     → compute_log_prob → advantage → update_policy → backward+optimizer

  4. weight_sync: actor→rollout
     → Level 1(LoRA): 只sync adapter → rollout.resume(tags=["kv_cache"])
     → Level 2(full): sync全部weights → rollout.resume(tags=["kv_cache", "weights"])

  5. wake_up: vLLM恢复GPU内存 → 重新分配weights+KV cache → 可以rollout了!

  关键: sleep→training→weight_sync→wake_up = 1个step的完整cycle!
  → GPU时间分时复用: rollout占→sleep→training占→wake→rollout占
  → 这就是为什么HYBRID需要sleep/wake → GPU不能同时被rollout和training使用!
```

## 5. Weight Sync 3种后端

### 5.1 naive (ColocatedCheckpointEngine)

```python
# checkpoint_engine/base.py:220-275
# HYBRID模式 → in-process → Python generator零拷贝!

class ColocatedCheckpointEngine:
    def send_weights(self):
        self.weights = actor.engine.get_per_tensor_param()  # generator
        # 不做任何网络传输! 直接Python引用传递!

    def receive_weights(self):
        yield from self.weights  # 同进程 → Python generator → 零拷贝!

# 优点: 最快 → 无IPC/NCCL/RDMA overhead → Python引用传递
# 缺点: 只能HYBRID模式 → rollout和actor必须同进程
# RTX 4090最优: HYBRID+naive → 零weight sync开销!
```

### 5.2 NCCL

```
# COLOCATED或STANDALONE → 跨进程/跨GPU → NCCL ProcessGroup

流程:
  1. Abort所有in-flight rollout requests → 确保vLLM不在生成
  2. 释放KV cache (保留weights → NCCL写入weight buffer)
  3. 建NCCL ProcessGroup(trainer workers + rollout workers)
  4. Trainer: NCCL AllReduce/Broadcast weight buckets
  5. Rollout: 接收weights → server_adapter.update_weights()
  6. 关闭ProcessGroup → 恢复KV cache → resume generation

bucket_size=2048MB → 多个weight tensor打包成1个bucket → 减少NCCL launch次数

优点: 跨GPU weight sync → 支持STANDALONE模式
缺点: NCCL overhead → PCIe瓶颈 → RTX 4090上很慢!
```

### 5.3 NIXL (RDMA P2P Ring)

```python
# checkpoint_engine/nixl_checkpoint_engine.py
# RDMA → NVIDIA NIXL库 → UCX/UCCL/Mooncake backends

class NIXLCheckpointEngine:
    def send_weights(self):
        # Ring topology: trainer_rank0 → rollout_rank1 → ... → rollout_rankN → back
        # 双buffer: send_buf + recv_buf → swap per iteration
        for weight_chunk in bucketed_weights:
            readable_op = nixl_agent.create_read_operation(send_buf)
            nixl_agent.push(readable_op, next_agent)  # RDMA push到下一个
            swap(send_buf, recv_buf)  # 双buffer交替

    def receive_weights(self):
        # 从prev_agent接收 → yield到vLLM → 同时接收下一个bucket
        yield from send_buf  # 当前bucket → vLLM处理
        recv_buf = nixl_agent.pull(prev_agent)  # RDMA接收下一个

# 优点: RDMA → GPU直传 → 无CPU拷贝 → 最快跨节点weight sync
# 缺点: 需要NIXL库+RDMA网络 → RTX 4090无RDMA → 不适用!
# 适用: NVLink+RDMA集群 → H100/H800跨节点
```

### CUDA IPC (BucketedWeightSender/Receiver)

```python
# rollout/vllm_rollout/bucketed_weight_transfer.py
# COLOCATED模式 → 同GPU但不同进程 → CUDA IPC

class BucketedWeightSender:
    # ZMQ IPC socket: ipc:///tmp/rl-colocate-zmq-...sock
    # 主路径: CUDA IPC handles (torch.multiprocessing.reductions.reduce_tensor)
    #   → handle共享源GPU存储；装入目标模型是否复制由callback决定，不是全链路零拷贝保证
    # 备选: Shared memory (Ascend NPU → IPC不支持)

    # Bucketed: 多个weight打包成bucket_size buffer → 减少ZMQ消息次数
    # 流程: send metadata → send CUDA IPC handle → receiver重建tensor

class BucketedWeightReceiver:
    # 接收ZMQ消息 → 从CUDA IPC handle重建tensor → 写入vLLM模型
```

新版失败传播与ACK边界见[分桶接收失败与退出](#分桶接收失败与退出)，不以发送函数返回代替模型安装/恢复验收。

### Weight Sync对比

| 后端 | 传输方式 | 适用模式 | 开销 | RTX 4090 |
|------|---------|---------|------|---------|
| **naive** | Python generator(零拷贝) | HYBRID | 0 | ✅ 最优 |
| **CUDA IPC** | ZMQ+torch IPC(GPU直传) | COLOCATED | 很低 | ✅ 可行 |
| **NCCL** | AllReduce/Broadcast | STANDALONE | 中 | ❌ PCIe慢 |
| **NIXL** | RDMA P2P ring | STANDALONE+跨节点 | 最低(跨节点) | ❌ 无RDMA |

## 6. ResourcePoolManager: GPU资源分配

```python
# ray/base.py:182-239
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]  # {"pool_0": [8], "pool_1": [8]}
    mapping: dict[int, str]                    # {0: "pool_0", 1: "pool_1"}
    max_colocate_count: int = 3                # FSDP: actor+rollout+ref共享

# max_colocate_count=3 → 每Ray actor得1/3 GPU → 3 WorkerGroups共享同一GPU!
#   → actor WorkerGroup + rollout WorkerGroup + ref WorkerGroup
#   → 或 actor + critic + ref (PPO模式)
```

### Ray PlacementGroup创建

```python
# ray/base.py:112-161
class RayResourcePool:
    def get_placement_groups(self):
        # 每bundle: {CPU: max_colocate_count, GPU: 1}
        # STRICT_PACK策略 → 所有bundles强制同一节点 → 确保colocation!
        # PG按node IP排序 → rank稳定性
```

### split_resource_pool: 分割资源池

```python
# ray/base.py:269-315
# 从一个pool创建SubRayResourcePool → 分配不同角色
# → rollout可以用不同的TP/PP配置 → 与训练不同的并行拓扑!
# → 关键: rollout(vLLM TP=4) 和 training(FSDP DP=4) 可以在同一组GPU上!
```

## 7. ActorRolloutRefWorker 初始化顺序

```
init_model (engine_workers.py:500-632) → @register(ONE_TO_ALL)

初始化顺序(有意的!):

1. Ref模型 (行504-539):
   → ref_config: ActorConfig → TrainingWorker → FSDP引擎
   → ref.reset() → 构建模型+FSDP wrap → 只forward(no backward)
   → CPU offload选项 → ref不参与训练 → 可以offload到CPU省GPU内存!

2. Actor模型 (行542-588):
   → actor_config: ActorConfig → TrainingWorker → FSDP引擎
   → actor.reset() → 构建模型+FSDP wrap+optimizer+LR scheduler
   → loss_fn = ppo_loss(config=actor_config) → PPO loss绑定
   → 完整训练引擎 → forward+backward+optimizer

3. Rollout引擎 (行591-616):
   → 3D DeviceMesh(dp, infer_tp, infer_pp) → 推理并行拓扑
   → rollout_cls = vllm.ServerAdapter / sglang.ServerAdapter / trtllm.ServerAdapter
   → LoRA flags: base_sync_done, layered_summon, peft_merge

4. Checkpoint引擎 (行619-629):
   → CheckpointEngineRegistry.new() → weight sync pipeline

5. GPU缓存清空 (行632):
   → aggressive_empty_cache(force_sync=True) → 为vLLM释放GPU内存!

关键: Ref→Actor→Rollout→Checkpoint顺序:
  → Ref先初始化(CPU offload) → Actor占GPU → Rollout接管GPU
  → 清空GPU缓存后 → Rollout引擎可以分配KV cache
```

## 8. FSDP引擎初始化详解

### FSDP2 wrapping流程

```python
# fsdp/transformer_impl.py + fsdp_utils.py

FSDPEngine._build_fsdp_module (行340-428):

FSDP1 (行340-404):
  module = FSDP(module, auto_wrap_policy=..., sharding_strategy=..., ...)

FSDP2 (行405-428):
  # 先保存full state_dict (FSDP2初始化后参数变为DTensor → 需先保存!)
  full_state = module.state_dict()
  apply_fsdp2(module, fsdp_kwargs, engine_config)
  fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)

# apply_fsdp2 (fsdp_utils.py:559-600):
  # 先wrap每个transformer layer → 然后wrap根module
  modules = _select_fsdp2_wrap_targets(model, transformer_layer_cls)
  for module in modules:
      fully_shard(module, **fsdp_kwargs)  # per-layer DTensor sharding
  fully_shard(model, **fsdp_kwargs)       # root module sharding
```

### Sharding策略选择

```python
# fsdp/utils.py:61-92
def get_sharding_strategy(device_mesh, zero3_enable=True):
    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD       # ZeRO-3等价
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD      # HSDP (ZeRO-3+DDP)
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP     # ZeRO-2等价
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        return fsdp_strategy                               # 1D → 全分片
    elif device_mesh.ndim == 2:                             # 2D → HSDP
        return hsdp_strategy                               # (ddp, fsdp) mesh
```

### DeviceMesh创建

```python
# fsdp/utils.py:40-58
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 1D mesh → 全FSDP → ZeRO-3等价
        device_mesh = init_device_mesh("cuda", (world_size,), ["fsdp"])
    else:
        # 2D mesh → HSDP → ZeRO-3+DDP
        device_mesh = init_device_mesh("cuda",
            (world_size // fsdp_size, fsdp_size),
            ["ddp", "fsdp"])
```

## 9. Rollout-Training Sharding转换 (无正式ShardingManager!)

```
verl没有独立的ShardingManager类!
→ 分片转换嵌入在ActorRolloutRefWorker的sleep/wake/update_weights流程中

转换本质: 内存所有权交换 + weight同步

训练阶段:
  FSDP引擎拥有GPU → DTensor shard在FSDP mesh上
  → AllGather→完整参数→forward→backward→ReduceScatter→1/N shard

Rollout阶段:
  vLLM拥有GPU → TP shard在vLLM内部
  → PagedAttention + KV cache + continuous batching

桥接: actor.engine.get_per_tensor_param()
  → FSDP: AllGather所有参数 → yield (name, full_tensor)
  → Rollout: update_weights → vLLM内部做TP分片 → 不需要显式resharding!
  → 关键: FSDP gather完整参数 → vLLM自己做TP分片 → 中间是完整tensor!

RTX 4090影响:
  → 单GPU: TP=1 → FSDP DP=1 → 无任何sharding转换 → 最简单!
  → HYBRID+naive: 同进程 → Python generator → 最快weight sync!
  → 多GPU PCIe: NCCL weight sync → 灾难 → 只能单GPU+LoRA
```

## 10. RTX 4090最佳配置

```
RTX 4090 24GB, 7B模型:

最优: HYBRID + naive + GRPO + LoRA
  → RolloutMode=HYBRID → actor+rollout同进程
  → Weight Sync=naive → Python generator零拷贝 → 0开销
  → Sleep/Wake: sleep_level=1(LoRA) → 只释放KV cache → 快速wake_up
  → GRPO: 无critic → actor+rollout共用GPU → 17GB peak → ✓fit 24GB
  → LoRA: 只sync adapter weights → ~2.6GB → 极快weight sync

时序:
  1. Rollout: vLLM generate(占据GPU, weights+KV cache)
  2. Sleep(level=1): 释放KV cache → FSDP做forward+backward
  3. Weight sync: naive → generator → 只sync LoRA adapters(~2.6GB)
  4. Wake_up: vLLM恢复KV cache → 可以rollout了!

不适合RTX 4090:
  → STANDALONE: 需要多GPU集 → PCIe NCCL weight sync灾难
  → COLOCATED NCCL: 跨进程IPC → 可行但不如HYBRID
  → NIXL: 无RDMA → 不适用
  → PPO: 2×模型内存 → 28GB ✗超出24GB

8×NVLink A100/H100 80GB:
  → HYBRID或COLOCATED均可 → NCCL weight sync很快
  → FSDP2+compile → 2-3x训练加速
  → PPO可用(critic独立GPU池) → 2×模型内存→80GB fit
  → NIXL RDMA → 跨节点weight sync → 最快
```

## 11. 关键设计洞察

```
1. FusedWorker → 同进程colocation → 最极致的GPU共享
   → HYBRID模式的核心 → actor+rollout=1个Ray actor → 0 IPC开销

2. Sleep/Wake → GPU分时复用 → 解决rollout+training争GPU问题
   → vLLM API: engine.sleep(level) / engine.wake_up(tags)
   → Level 1: LoRA-safe → 只释放KV cache → 快
   → Level 2: 完全清空 → 全量weight sync → 慢但必须

3. 3种Weight Sync → 适配3种部署模式
   → naive=HYBRID(in-process) → NCCL=STANDALONE(跨GPU) → NIXL=RDMA(跨节点)
   → CUDA IPC=COLOCATED(同GPU跨进程) → BucketedWeightSender

4. Dispatch路由 → WorkerGroup自动管理DP并行
   → @register(ONE_TO_ALL) → 广播
   → @register(DP_COMPUTE_PROTO) → DataProto自动split/merge
   → ND_compute → 支持非均匀DP-TP-PP拓扑

5. ResourcePoolManager → Ray PG + 分数GPU → 灵活colocation
   → max_colocate_count=3 → 3 WorkerGroups共享同一GPU
   → split_resource_pool → rollout用不同TP/PP → 灵活并行配置

6. 初始化顺序 → Ref→Actor→Rollout→Checkpoint → 有意的
   → Ref可CPU offload → Actor是核心 → Rollout接管GPU → Checkpoint桥接

7. 无ShardingManager → FSDP gather完整参数 → vLLM内部TP分片
   → 中间是完整tensor → 不需要显式resharding → 简化!
```

---

## 分桶接收失败与退出

2026-10-23核对已有[verl #7366](https://github.com/verl-project/verl/pull/7366)，读取时open、非draft，head `48d91130abff6b6704aadac3f382e0c3d344de93`，目标base及共同祖先 `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`，ahead2/behind0，11文件424增33删。它给接收异常增加错误ACK、给发送端逐次ACK等待加上限，不是权重事务或自动恢复实现；上文六月架构概述不作为当前性能测量。

### ACK 等待的范围

[完整协议实现](https://github.com/verl-project/verl/blob/48d91130abff6b6704aadac3f382e0c3d344de93/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py)把以下四处原裸recv改为同一个同步`_receive_ack`，先poll再recv：

| ACK阶段 | 发起条件 | 收到成功ACK代表什么 |
|---|---|---|
| buffer initialization | 发送共享buffer信息或设备IPC handle | 接收端重建buffer，尚未装模型 |
| intermediate bucket | 下一个tensor放不下，先冲刷当前bucket | 当前callback返回且设备synchronize返回 |
| oversized tensor | 非SHM路径单tensor大于bucket | 该直接IPC tensor的callback和同步返回；SHM仍拒绝超大tensor |
| final bucket | 迭代结束，包含空权重流 | 最后callback和接收端清理走到末尾ACK；非全rank/模型状态认证 |

接收端在初始化、metadata解码/视图构造、callback及其后同步抛`Exception`时，尝试NOBLOCK发送`error:异常类型: 消息`，except分支再raise；外层finally清理仍可能另失败。发送端将错误ACK内容转成RuntimeError。错误发送本身遇ZMQError只告警，硬退出/对端丢失也可能没有错误ACK，所以**超时是错误文本未送达时的退路，不是所有异常均能原样送达的保证**。成功响应只检查是否有error前缀，没有新增权重manifest、内容hash或版本验证。

最终成功ACK仍延后到receiver清理末尾，以免sender抢先回收通信buffer。两端新设LINGER=0；[libzmq文档](https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html#zmq-linger-set-linger-period-for-socket-shutdown)规定这会在close时丢弃未送达消息，不能把close等同交付。receiver的SHM close若因活跃view抛BufferError则只warning，清理函数名也不是所有引用已消失的证明。

[配置及调用点](https://github.com/verl-project/verl/blob/48d91130abff6b6704aadac3f382e0c3d344de93/verl/workers/rollout/vllm_rollout/vllm_rollout.py)需要区分配置值、实际调用与等待范围：

- `rollout.checkpoint_engine.update_weights_ack_timeout_seconds`默认30，dataclass要求正且有限，普通named_tensors路径取`int(seconds*1000)`传给sender；sender仅检查ms大于0。因此小于1ms的正值仍可能在截断为0后被拒绝，并非所有正有限配置都可用。
- 该head的delta_flush路径另建sender但没传新配置，仍取构造器默认30000ms，不能称整个vLLM adapter都遵循用户设定值。
- 每次ACK重新用完整预算；buffer分配/同步、权重生成、send、receiver等待新请求、外层Ray future及cache恢复均未获得整轮截止时间。超时不取消远端callback，不恢复已经写入的权重，也不提供原地重试契约。

### 原 CPU 测试与事件循环观察

[复查脚本](../../experiments/scripts/verify_verl_bucketed_transfer.py)从固定Git对象解包完整生产模块和完整原`test_bucketed_weight_transfer_on_cpu.py`，不改函数/断言。为避免导入verl根包、Ray和自动设备发现，启动时仅给包根及四个device入口提供显式CPU facade；原测试自身还替换同步/empty_cache、IPC能力和sender的iterator helper。因此这是**真实本机ZMQ与共享内存协议测试，不是原生verl包导入、设备同步或vLLM集成**。

M1 macOS，Python3.12.13、torch2.9.1、pytest8.4.2、pyzmq27.1.0/libzmq4.3.5：原8例在沙箱外连续两次全过（9.99s、9.94s，仅执行耗时）。其中3例是recv-only/timeout/解码故障socket替身控制；5例是真实spawn进程、IPC socket及SHM，覆盖成功、初始化失败、空流最终callback失败、首个中间bucket callback失败和最终ACK前peer退出，均执行原socket路径与共享段不存在断言。成功例只断言收到名称，没有核tensor值；它们不认证权重数值正确或模型安装完整。

首跑沙箱内3过5失败：torch_shm_manager被Operation not permitted拒绝，其余进程等待未完成；获准本机IPC后保持源码/断言不变复跑，不将权限失败定性为协议缺陷。pyzmq只安装在临时target，未改现有venv。另对该PR全部6个变更Python模块仅做AST/compile；配置原测试及旧accelerator测试文件未执行。

脚本另用真实inproc REQ/REP请求但不回复，调用原`_receive_ack(150ms)`：10ms asyncio tick实际延后约156.8ms，20ms `asyncio.wait_for`未能提前打断，同步poll结束后先得到原RuntimeError。**async_send_weights声明为async，不代表ACK等待让出事件循环**；这里只证明本机控制流，不外推生产调度延迟或设备性能。复现需本机IPC/共享内存权限，且本地Git中已有固定对象：

```bash
# 使用已有Python3.12+/torch/pytest环境；pyzmq也可用环境现有版本并记录版本。
bwt_deps="$(mktemp -d)"
python -m pip install --target "$bwt_deps" --only-binary=:all: pyzmq==27.1.0
python -B experiments/scripts/verify_verl_bucketed_transfer.py /path/to/verl-git --deps "$bwt_deps"
```

输入SHA256：协议模块`58d8198dd6ce3c5743d560d7064820c96b11b9f947475670afd4bf7b518ee954`；原CPU测试`0aaf5ba03e4c75a21556d482d2bce71db0b3cdd4b806cda4b80334e2505c0130`；base…head完整patch `dc67cbc8251489035df1d101d3b910ac25df17bd0da8fb6385c9924d3f316268`。

CUDA IPC、NPU的host→device拷贝、真实vLLM callback/多rank及故障后资源释放仍未测；[作者说明](https://github.com/verl-project/verl/pull/7366)也未声称CUDA或完整vLLM验证。条件补[原M5同步交接](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，仅供日后独立选题，不应用候选、故障注入生产或据此诊断#8250。

Sources:

- verl/single_controller/base/worker_group.py — WorkerGroup基类
- verl/single_controller/ray/base.py — RayWorkerGroup + ResourcePoolManager + PlacementGroup
- verl/single_controller/base/decorator.py — Dispatch + @register装饰器
- verl/workers/engine_workers.py — ActorRolloutRefWorker + TrainingWorker
- verl/workers/rollout/replica.py — RolloutMode + RolloutReplica sleep/wake
- verl/workers/rollout/vllm_rollout/vllm_async_server.py — vLLM sleep/wake API
- verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py — CUDA IPC weight transfer
- verl/checkpoint_engine/base.py — CheckpointEngine ABC + ColocatedCheckpointEngine
- verl/checkpoint_engine/nixl_checkpoint_engine.py — NIXL RDMA weight sync
- verl/workers/engine/fsdp/transformer_impl.py — FSDPEngine
- verl/utils/fsdp_utils.py — apply_fsdp2 + create_device_mesh
- notebook/frameworks/verl-rl-infra-reading.md + verl-dataproto-data-flow-reading.md
