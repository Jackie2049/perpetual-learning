# Rollout组消费的限定CPU控制

2026-10-28，M1。本材料支持[统一笔记§8.11–8.12](../../notebook/rollout-performance-research.md#811-rllm-组消费与同步窗口)，不是框架、训练或性能复现。

## 固定来源与复跑

[verify.py](verify.py)仅用Python标准库，要求Python 3.12；检查两个只读checkout的完整HEAD和13份源码与Git blob一致，再解析AST。rLLM固定`4e828c81680e5a85b638ceedeaf09d1e90109ca8`，Slime固定`540ea919e45b8b929198abce008a103be9cfc779`，不自动拉取或修改来源。

```sh
python3.12 experiments/rollout-group-controls/verify.py --rllm /absolute/path/to/rllm-checkout --slime /absolute/path/to/slime-checkout
```

本机复用`work/pl-c2-read-controls/.venv/bin/python`（3.12.14），来源为`work/rllm-c3-readonly`及`work/slime-c3-readonly`。[result.json](result.json)保存首轮通过的stdout，可与复跑JSON逐字段比较。

## 检查内容与限制

八组检查：派发配额消费／过滤差异、同步结转及step阈值、asyncio暂停恢复、失败cause和等待释放、取消drain、组最小版本／缺省0、Slime取消所属与server计数finally、过滤不足目标的条件分支。

运行完整原stdlib SyncCoordinator与filter模块；按AST原样提取最小版本及两取消helper。asyncio是真实event loop，sample／state为替身。没有导入RL框架、HTTP／Ray请求、存储／checkpoint、optimizer、GPU／NPU或吞吐测量；没有运行原仓库pytest，也不将控制组数当独立代码路径或修复证明。源码阅读另核默认驱动、buffer转换与训练交界，测试没有覆盖整个调用链。
