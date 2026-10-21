# verl 单 rollout worker 的 NIXL 拓扑检查

核对已有[#8316](https://github.com/verl-project/verl/pull/8316)的一行切片修正，只执行原拓扑函数和纯参数断言，不初始化NIXL/Ray或传输权重。理解见[原同步笔记](../../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑)。

## 固定输入和运行

head `3a60b8710f8e1001110820f7a14cd0e71d9d5e07`；目标base `f2923cbf154731918b1e36db8c567433f349fede`；共同祖先/本机对照基线 `619c397dc9917fed278b6e47c52a6620aef05466`。两份实现文件在祖先与目标base无diff；真正PR相对祖先只有nixl_checkpoint_engine.py 1增1删。目标base另有两份NPU CI文件变化，不能用双点diff把它们算作本PR删除。

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/verl-nixl-topology/verify.py /path/to/verl
```

2026-10-21以Python3.12.13执行，标准库足够，无安装。[verify.py](verify.py)用GIT_NO_LAZY_FETCH=1读取两固定对象，AST原样选NIXLCheckpointEngine.build_topology（保留classmethod，外包无继承/初始化的类壳）；另执行原CheckpointEngineManager.build_process_group的两个纯For/Assert，以及原init_process_group第一个参数校验If。未执行这些方法的prepare、Ray调用、属性写入或agent连接部分。长度所需world_size用SimpleNamespace提供，metadata是只含actor/rollout标签的字典，不是真实agent地址或网络元数据。

## 有界矩阵结果

合法输入约束：A≥1、R≥1，metadata按A个actor后接R个rollout排序且长度恰好A+R。A取1/2/4/8，R取1/2/3/4/8，每版本20组，是本轮自建控制，不是作者新增20单测。

| 实现 | 原长度/端点检查 | 独立链关系核对 |
|---|---|---|
| 共同祖先 | 16组过，4组R=1触发原长度断言 | 其余16组相邻关系符合预期 |
| 候选 | 20组全过 | 20组rank/prev/next关系符合预期 |

R=1的旧next列表长A+2，错误为 `rollout_kwargs[next_agent_metadata] must have length of 1`；候选只含[None]。R≥2的16组两版本完整输出相等。脚本将基线四项失败作为预期负对照检查，整体退出0不表示旧实现通过；没有执行完整build_process_group或任一设备单测。

PR正文勾选新增单测，但固定head实际diff没有测试文件。既有test_correctness_on_gpu.py中的NIXL用例为2 trainer/6 rollout、并有temporary skip标记，且device=cpu仍需NIXL/Ray环境；本机未解除skip或运行它。拓扑长度/端点正确不认证网络、buffer注册、数据传输、应用或同步后恢复生成。

## 后续新增原测试

同日读取的新head `c2d800e71cdc4c7295b1e12e60bad82dd2fb637b`仅比首轮head新增28行tests/checkpoint_engine/test_nixl_checkpoint_engine_on_cpu.py；实现逐字未变。当前PR已含这一测试，上段“没有测试文件”仅适用首轮3a60b871。脚本默认模式继续固定旧版矩阵，以保留原复查结果；新增模式执行新head的原测试函数：

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/verl-nixl-topology/verify.py /path/to/verl --upstream-case
```

原函数及两个断言不改，AST选择函数并提供只承载原classmethod的类壳，跳过顶层后端import。A=R=1、原字符串metadata：基线测试体1项预期失败，新head1项通过；Python3.9.6/3.12.13均相同。模式检查预期负对照后退出0，不代表基线通过，也不等于完整pytest或模块导入验收。测试SHA256 `699eb471d14bb946d47cd794416a23c04ee60a44b3ac206d2dafb8146e91541c`。

原文件直接导入NIXL实现，仍需其cupy/nixl/Ray/torch/ZMQ依赖可导入；本机没有安装或执行这些导入。on_cpu表示测试逻辑没有设备操作，不自动意味着测试收集无原生依赖。新测试只检查两个邻居字段，不覆盖原矩阵所有rank/world_size或真实通信；不新增M5任务。

## 输入校验

SHA256：nixl实现基线 `5ba0aa3ecb41ef71849f9919d2e291defa2b78be7e29a3a640ddc25602493a71`、head `eefe6a3fc0f69e0b091edca67736bda288265b8a3411b9feb04de064d52a87da`；manager base.py `97b51131a5770892007b7fb07264a2657fe83255b085927acd08c952a5682f64`。没有导入torch/ray/nixl/cupy/zmq/verl，未应用上游补丁；真实传输仅在M5已有任务确需该backend且资源匹配时独立决定，不据此诊断#8250。
