# DeepInfer 共享池句柄身份检查

验证已有 [#187](https://github.com/deepseek-ai/DeepInfer/pull/187) 的CPU分配身份检查，不修改上游源码。结论见[主笔记](../../notebook/frameworks/deepinfer-initial-survey.md#共享池句柄的分配身份与视图寿命)。

## 固定输入

- base/共同祖先：`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。
- 候选及测试：`5c2bb4625506e5cec761eb5a7f08f4336ab45bd7`。
- 原实现：deepinfer/shared_memory.py；原测试：tests/test_shared_pool_generations.py（9项）。

[verify.py](verify.py)从已有Git对象读取固定输入，GIT_NO_LAZY_FETCH=1，不自行fetch。AST原样保留BlockHandle、_shared_array、SharedTorchMemoryPool全部节点及其普通依赖导入；不加载独立的posix_ipc SharedMemory类、懒加载池工厂或DeepInfer包初始化。为pickle身份注册同名隔离module，logger用标准logging，PLATFORM只提供两个会主动报错的占位入口，由原fixture替换为pin/unpin no-op。没有改原方法、fixture、参数化、测试体或断言。

Pytest自定义collector仅将原样内存测试模块挂到当前脚本路径，不写临时源码，不读取上游conftest/外部plugins。真实torch CPU张量、multiprocessing.shared_memory、共享元数据、锁和fork仍执行；torch线程数设1。默认沙箱会阻止shm_open，需允许本机IPC；不需要GPU/NPU或服务器。

## 复查命令

使用已有Python3.12+环境，需torch、numpy、pytest；不会自动安装。上游克隆须已包含两固定对象，在本项目根执行（替换解释器和克隆位置）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-pool-generations/verify.py /path/to/DeepInfer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-pool-generations/verify.py /path/to/DeepInfer 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
```

2026-10-21，macOS15.4.1 arm64、Python3.12.13、torch2.9.1、numpy2.0.2、pytest8.4.2：

| 原测试分组 | base | 候选 |
|---|---|---|
| 旧句柄三入口 free/buffer/tensor | 3项未抛ValueError | 3过 |
| 其他池句柄三入口 | 3项未抛ValueError | 3过 |
| 本机可信pickle往返 | 1过 | 1过 |
| 重复合并后的generation唯一性 | 缺generation属性 | 1过 |
| 真fork分配后父进程旧句柄 | 未抛ValueError | 1过 |
| 合计 | **1过、8失败，退出1** | **9过，退出0** |

基线失败原样保留，不改xfail。默认沙箱首次9项全部因shm_open权限在fixture初始化报错，不计为实现失败；允许IPC后才得到上表。候选复跑仍9过。

候选成功后另执行 **1项自建视图寿命控制**，不是第10项上游测试：复用原fixture创建512字节池，保留原get_buffer_slice视图，free旧句柄再alloc同尺寸，新视图填17，旧视图也读17；通过旧视图写73，新视图读73。随后释放当前句柄、删除视图并退出fixture关闭池。这个主动保留引用的控制说明代际检查不会撤销已返回视图，不是线上引用逃逸或设备拷贝故障复现。

## 范围与校验

没有设备注册/拷贝、完整包、模型、调度器或多模态流水线运行。fork仅覆盖原测试的单子进程分配/Queue回传，不证明spawn、进程崩溃、任意并发读写或H2D寿命。可信本机pickle往返不验证不可信反序列化安全。原close负责本轮池清理，输出无resource_tracker泄漏警告，但没有另做全部系统句柄枚举或异常退出回收验证。

输入SHA256：实现base `1aff9ee8f589c59e9b982707fe518bc4a8722814f971f741631d07d8a1319ce0`、候选 `0e531dca4d7da898ac8b687ab7e8c5eab62c3c46656e8214d2c48452a5860ce1`；原测试 `ef25c126c2cd4f231a8f0237b8f4726f976dde3f929c4f59462eeec01100f324`。脚本运行时打印完整输入hash。本题不竞争作者修复、不自动应用补丁或新增M5设备任务。
