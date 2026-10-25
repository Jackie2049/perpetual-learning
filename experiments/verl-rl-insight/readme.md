# verl RL Insight 日志测试与 worker 接线

复查已有[#8315](https://github.com/verl-project/verl/pull/8315)的新CPU测试及实际lane表达式。主要结论见[原同步笔记](../../notebook/frameworks/verl-async-logprob-provenance.md#新版原测试与实际接线)：logger层测试通过不代表生产worker接线或时间线后端验收。

## 固定输入与命令

[verify.py](verify.py)支持共同祖先`cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`、前head`4d6b3eb8606e24c1d70d648e92a1114129ec6c62`和当前head`0cea139623cc8913d2241d21223df5682474041c`。当前目标base为`f2923cbf154731918b1e36db8c567433f349fede`；真正PR从祖先为5文件138增4删，新旧head之间只增65行原测试并把worker的lane由整数改为rank字符串。

Python3.12.13 / pytest8.4.2 / packaging26.3，无新增安装。项目根执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-rl-insight/verify.py /path/to/verl --revision current
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-rl-insight/verify.py /path/to/verl --revision previous
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-rl-insight/verify.py /path/to/verl --revision ancestor
```

GIT_NO_LAZY_FETCH=1读固定对象，不切换checkout或写上游源码。AST原样隔离完整RLInsightLogger与RolloutTraceConfig类（保留装饰器/注解/方法），提供真实标准库和Version；包壳跳过verl根及tracking其余后端、orjson/Ray导入。完整当前原测试模块执行，原autouse reset、monkeypatch、MagicMock客户端及全部断言不改。内存collector以本脚本作锚点，禁用conftest、外部插件、pytest缓存与字节码。不是完整原包导入/官方CI环境。

## 2026年10月22日结果

| 被测类版本 | 当前原12项测试 | 实际worker lane表达式（rank=2） |
|---|---|---|
| 共同祖先 | 7过、5预期失败 | 未加入此装饰器，不执行表达式 |
| 前head | 12过 | int `2` |
| 当前head | 12过 | str `rank_2` |

祖先5项失败为metric_log断言及4项新增decorator测试；原pytest退出1，包装脚本核对计数/失败集合后退出0，不改xfail。原测试自定义Worker使用自己的字符串lambda，没有调用生产ActorRolloutRefWorker，因此前head同样全过。另从实际engine_workers的装饰器中只选lane原lambda、以SimpleNamespace(rank=2)求值，未构造或执行worker。

原测试的客户端仍为MagicMock：metric_log默认显式设None，避免Mock自动生成可调用属性；单项测试再显式启用它，核step=100且不双写gauge。asyncio的16协程例仅检查同进程事件循环内的lazy init调用次数，不证明跨线程/进程初始化安全、真实投递/存储或绘图。没有启动rl-insight、Prometheus、Ray、模型或设备。

首次审计钩子拒绝全部socket事件，误伤asyncio自唤醒用的本地匿名AF_UNIX socketpair，出现4失败/8通过及未等待协程警告；这是本复查脚本的限制，不是上游失败。修正为仅允许AF_UNIX socket创建，继续拒绝INET创建及bind/connect/DNS等事件，三版本重跑得到上表结果且无该警告。没有放开服务连接或修改原async测试。

脚本打印完整输入SHA256，核torch/ray/orjson/rl_insight未导入。当前tracking与前head逐字相同；当前engine_workers SHA256 `7166ff29e71eb101031ce88aa22feb32f0d960a5e795af0e8116f37aff757ec1`，原测试 `874a695c536517254d666dc5b42a7309993a7f86d33e654ec7fbf66c6de58cf3`。本实验不认证监控后端接收、lane筛选/显示、真实性能或权重同步；真实观测仍依M5原事项条件，不新增实验排队或默认采用候选。

## Gauge 元数据原测试补证

2026-10-26只补[#8357](https://github.com/verl-project/verl/pull/8357)的完整原七项logger测试，不重开[已有事件路由结论](../../notebook/frameworks/verl-async-logprob-provenance.md#gauge-元数据与训练步存储)。两个新revision选项分别固定父版／main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 和head `cd12877931cdd441adb731978a8b3785858dba27`；均配head原测试文件。类、fixtures、测试体和断言不改，隔离机制沿用上节，真实客户端仍是原MagicMock。

Python3.12.13／pytest9.1.1／Darwin：候选7项通过；父版6项通过，仅test_log_lazy_inits_and_writes_gauge因缺少两个新增标签而预期AssertionError。原pytest对父版返回1，包装器核对精确失败集合后才退出0，没有改xfail或吞掉任意失败。环境关闭、初始化／清理、16协程及旧特性降级的Mock契约都包含在七项中，不是七项新功能。源码hash、每项结果与环境见[gauge-result.json](gauge-result.json)，两次执行语义一致。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-rl-insight/verify.py /path/to/verl --revision gauge-base
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/verl-rl-insight/verify.py /path/to/verl --revision gauge-head
```

两版各两完整生产模块及同一原测试文件均内存compile；不执行整包导入。修改脚本后的默认current模式12项也回归通过，只保证旧复查入口未破坏，不新增tracing或后端结论。本轮初筛漏检原笔记，短暂重复的事件控制已停止、未保留成新实验或新发现；未安装缺失的Prometheus客户端，没有registry／服务／存储验证或新M5任务。
