# 历史脚本与实验性代码

保存一次性、纯实验性、概念演示或待整合的旧实现，供复盘与提炼使用。正式工具入口及价值标准见 [tools/readme.md](../readme.md)。

## 本次归档

| 文件 | 归档原因 |
|---|---|
| [apply_6274_fix.py](apply_6274_fix.py) | 针对 TRL 特定问题、安装路径和源码形态的补丁应用脚本 |
| [apply_6361_fix.py](apply_6361_fix.py) | 针对 TRL 特定模板问题的补丁应用脚本 |
| [apply_p7_2_fix.py](apply_p7_2_fix.py) | 为一次 TRL reward clipping 贡献准备的补丁脚本 |
| [apply_p8_1_fix.py](apply_p8_1_fix.py) | 为一次 TRL group size 贡献准备的补丁脚本 |
| [apply_p9_1_fix.py](apply_p9_1_fix.py) | 为一次 TRL bypass mode 贡献准备的补丁脚本 |
| [apply_rllm_605_fix.py](apply_rllm_605_fix.py) | 针对 rLLM 特定源码版本的分组逻辑补丁 |
| [fix_p91_config.py](fix_p91_config.py) | TRL 配置的一次性补充修复，依赖固定安装路径 |
| [ai_ethics_governance_simulator.py](ai_ethics_governance_simulator.py) | 基于合成数据和预设规则的学习模拟，不能充当真实系统评估 |
| [ai_infrastructure_security_simulator.py](ai_infrastructure_security_simulator.py) | 基于预设成功率与防御比例的概念模拟，没有真实系统测量 |

这些文件从 `tools/<文件名>` 移到 `tools/legacy/<文件名>`，源码保持不变。本次检查了 Python 语法、内容一致性和仓库内引用，未执行补丁或连接其目标环境。补丁脚本会改写目标安装文件，使用前应确认版本、路径、备份与授权。

## 查找与再利用

- [历史工具导航](catalog.md) 保存旧首页的工具索引，其中未归档的文件仍链接回原位置。
- 归档脚本中的结论和配置只适用于其原有假设与环境。使用时读取实现与对应实验记录，不将输出当作当前事实。
- 有明确复用价值的实现可以提炼回 tools；补齐用法、适用范围和验证记录后再加入正式入口。
