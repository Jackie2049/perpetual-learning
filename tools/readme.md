# 可复用工具

这里沉淀能反复解决实际问题的工具。项目总览见 [README.md](../README.md)，技术知识见 [notebook/](../notebook/readme.md)。

## 工具价值标准

新增或保留一个工具，应能说明：

1. **解决什么问题**：有具体使用场景、输入和输出，能够在后续任务中复用。
2. **为什么值得维护**：先检索现有工具和成熟方案；优先复用、合并或扩展，避免重复造低价值轮子。
3. **怎样使用与验证**：提供用法、依赖、适用环境、边界和验证记录；模拟估算明确假设，不能把假设当成实测结果。

一次性修复、纯实验脚本、概念演示和已被替代的重复实现放在 [legacy/](legacy/readme.md)，保留研究材料。进入 legacy 不意味着内容没有学习价值，但不作为默认推荐工具。

## 使用入口

以下是具有复用场景的既有入口，状态仅表示本次实际完成的检查，不代表全面验收。

| 工具 | 用途 | 依赖与当前验证范围 |
|---|---|---|
| [gpu_monitor.py](gpu_monitor.py) | GPU 状态监控、CSV 日志及分析 | Python、NVIDIA nvidia-smi；本次仅检查 CLI 帮助，未连接 GPU |
| [training_recipe.py](training_recipe.py) | 根据输入生成训练配置和启动命令 | Python；本次仅检查 CLI 帮助，生成命令仍需对应框架验证 |
| [ai_infra_cheatsheet.py](ai_infra_cheatsheet.py) | 按主题查阅常用命令 | Python；历史命令需结合当前环境核对 |
| [inference_estimator.py](inference_estimator.py) | LLM 推理性能估算（显存/吞吐/延迟） | Python；数值为估算，基于 50% 计算利用率与 60% 带宽利用率假设，非实测 |

示例（在项目根目录执行）：

```bash
python3 tools/gpu_monitor.py --help
python3 tools/training_recipe.py --help
```

## 历史资产整理状态

- 本次将 7 个特定 issue/环境的一次性补丁脚本、2 个基于预设参数的概念模拟脚本移入 legacy，保留原内容。
- [历史工具导航](legacy/catalog.md) 保留旧 README 的索引，方便找回已有材料。
- 根目录其余历史脚本仍需按用途、依赖和证据逐项检查；留在原位不代表已通过上述价值标准。不要仅凭 `simulator`、`benchmark` 等文件名批量判定质量。
- [inference_estimator.py](inference_estimator.py) 已修复单请求带宽硬编码（`30 GB/s` 改为按 GPU HBM 规格）和吞吐展示口径（区分最大 batch 的每步耗时、bs=1 单流吞吐、理论带宽极限），新增 `--batch` 与 h20/rtx4090 规格；已按 A100/H20/RTX 4090 多配置验证输出合理性。

实验结论与实战经验记录在 [notebook/experiments/](../notebook/experiments/)，与代码互相链接。后续若将 legacy 内容升级为正式工具，需要补齐复用场景、接口和验证证据。
