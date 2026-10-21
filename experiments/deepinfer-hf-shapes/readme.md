# DeepInfer checkpoint 预检与 gate up 顺序

复查已有[#153](https://github.com/deepseek-ai/DeepInfer/pull/153)的55项原CPU测试，并用真实文件/原加载器验证名称前缀是否改变两个半块的语义顺序。理解与边界见[原DeepInfer笔记](../../notebook/frameworks/deepinfer-initial-survey.md#checkpoint-布局预检与-gate-up-顺序)。

## 输入与运行

固定base/merge-base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，候选 `c2bbb61623c5120364faa879829feabb6d709434`，读取时open/draft，作者liuyinchu。真正PR三文件456增19删，不应用补丁或切换上游checkout。

Python3.12.13、torch2.9.1、pytest8.4.2、safetensors0.8.0；torch须支持float8_e8m0fnu。本轮经获准仅向新建`/tmp/perpetual-hf-order.MqSr0T/deps`安装safetensors，未改变既有CPU解释器包集或M5环境。临时依赖目录不保证长期存在；复查时用已有这些依赖的独立环境，或把独立deps目录放入PYTHONPATH，再在项目根运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-hf-shapes/verify.py /path/to/DeepInfer --revision fix
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python experiments/deepinfer-hf-shapes/verify.py /path/to/DeepInfer --revision base
```

[脚本](verify.py)从固定Git对象完整加载原logger、names、parameters、weights、loading与候选366行原测试，不提取函数/替换文件读取器。包壳仅跳过DeepInfer和hugging_face根初始化；原fixture/参数化/断言不变，内存collector禁用conftest、外部pytest插件、cache/字节码。GIT_NO_LAZY_FETCH=1，网络审计拒绝socket操作。真实CPU张量、真实safetensors写入/头部/数据读取；没有数据下载或完整模型构造。

## 2026年10月22日结果

候选原 **55项全过**；base **22过、33项预期失败**，其中32项为未拒绝非法布局，1项为覆盖顺序断言。脚本核失败函数集合及原因；原pytest退出1，包装脚本确认预期后退出0，不改xfail。首次包装脚本把1个已有alias检查误列为失败、漏列跨shard FP8 scale失败而拒绝结果；按真实原断言核清并修正预期集合后重跑，不是修改上游测试或掩盖额外失败。

原顺序测试在base的coverage断言处就停，不能据该测试声称已观察到反向装载。因此另做每版4个原加载控制：w1/w3各32×64 BF16，值分别2和7，分属两个真实safetensors文件；构造同路径CPU参数，调用原build_coverage→validate_hf_shapes→load_covered_params，逐元素检查前后32行：

| w1前缀 / w3前缀 | base加载的两半值 | 候选 |
|---|---|---|
| 无 / 无 | 2 / 7 | 2 / 7 |
| model. / model. | 2 / 7 | 2 / 7 |
| model. / 无 | **7 / 2** | 2 / 7 |
| 无 / model. | 2 / 7 | 2 / 7 |

这些控制复用原测试的造文件/参数helper，但不是原55项，单独计数；未使用假reader或只手写torch.cat证明结果。TemporaryDirectory仅清理本次自建的合成小文件，不涉及用户checkpoint。完整suite含普通FP8 scale值1/2/4/8恢复、Engram本地padding以及header-only guard；packed expert用例主要校验布局，未跑实际专家分配/权重重排。

原log_load_phase读取CUDA是否已初始化的本地标志，未初始化/查询设备或连接服务器；结束核tile_kernels/deepinfer._C/deep_ep未导入。未运行model_loader的完整调度、原生quant kernel、collective或全模型推理，不认证任意checkpoint、事务回滚或性能。预检接受Engram跨scale shard不等于加载支持，原表加载仍要求weight/scale同shard。

脚本逐项打印SHA256。候选loading `1a273325339961f111c7679b94a8ca3752a83814148752ca1fef5bc2effba8ac`、weights `503a2ceaeb72d0b746b442edb73dc0c34cda4c1366e872ae578a2d9a7335268d`、原测试 `714ba4b30ba2b5d43e9437e8adc43221428cc12a9b6ac59951c0a534e4859063`。本课题CPU闭环，不新增M5真机任务或竞争上游实现。
