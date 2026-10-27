# 早期veRL源码溯源与静态核查

2026-10-28，M1。[统一笔记§8.13](../../notebook/rollout-performance-research.md#813-早期verl重分片与生成路径)对照公开`v0.1rc`生成／重分片调用链；[§13.1](../../notebook/rollout-performance-research.md#131-phase-2覆盖与证据缺口)记录覆盖账和剩余证据。不接管HybridFlow的原并行认领。

## 复跑

[verify.py](verify.py)仅依赖Python标准库，固定`53bb5d21372a667954afb7109a2ef1df485e3137`；来源checkout需含四个早期提交对象及脚本列出的12份源码。代码路径：`git clone --filter=blob:none --branch v0.1rc https://github.com/verl-project/verl.git /absolute/path/to/source`；也可独立浅克隆后deepen四个提交。不改已有用户工作区。

在个人项目根目录执行：

```sh
python3.12 experiments/verl-v01rc-provenance/verify.py --source /absolute/path/to/source
```

本轮来源为`work/verl-v01rc-c4-readonly`，复用`work/pl-c2-read-controls/.venv/bin/python`（3.12.14）。[result.json](result.json)保存stdout。检查HEAD／tag／root、四提交历史、工作树干净、源码等于blob及12份AST；另核sharding enter调用顺序、actor重算位于context之后，以及126身份／120唯一arXiv／20 T1＋84 T2＋21 T3＋1未标的互斥账。

## 证据范围

公开链根提交为2024-10-31，`v0.1rc`提交为11-01，晚于论文v2的10-02。这里只认证该公开历史链，不证明作者不存在其他内部／已改写历史，也不认证该tag就是论文实验commit。README安装例、YAML默认配置和论文实验条件分别保存，不拼装为复现配方。

脚本仅静态检查，不执行原源码函数，不导入torch／Ray／Megatron／vLLM，不运行tensor、collective、KV／内存分配、设备或性能实验。未下载模型／数据集，没有GPU交接任务。矩阵账变化后脚本的固定断言会要求显式更新，不能自动认证未来覆盖。
