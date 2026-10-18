# 历史实验环境

| 文件 | 保存的环境 |
|---|---|
| [environment.yml](environment.yml) | 本地 ai-infra Conda 环境，包含 macOS 包构建及 Python 3.11 依赖 |
| [environment-gpu.yml](environment-gpu.yml) | GPU 机器的 Linux Conda 环境，包含 Python 3.9 与训练、推理依赖 |

两份文件是历史机器的导出快照，保留原始依赖和 prefix，供复现实验时参考；并非当前项目统一的可移植安装清单。迁移没有修改依赖或验证重新安装。返回 [实验总览](../readme.md)。
