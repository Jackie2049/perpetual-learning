# 实验结果

保存实验原始结果；分析与结论继续放在 `notebook/experiments/` 等对应笔记中。

| 目录 | 内容 | 本次归档文件数 |
|---|---|---:|
| `training/` | GRPO、DAPO、RLOO、DPO、蒸馏与奖励模型实验 | 14 |
| `distributed/` | DDP、FSDP、专家并行 | 4 |
| `kernels/` | Triton 算子性能 | 1 |
| `inference/` | 采样、RAG、推理实验 | 4 |
| `architecture/` | DeepSeek-V3、MTP、位置编码 | 3 |

2026-10-18：原根目录的 26 个 JSON 按主题归档，文件名与内容保持不变。既有 `nandetectmode_validation/` 和其他本地输出保留原位。

已调整对应本地脚本的默认输出位置，以脚本所在项目为基准创建结果目录。支持 `--output` 的脚本仍允许显式指定路径。未找到直接生成脚本的历史结果，仅归档数据，不推测运行来源或环境。
