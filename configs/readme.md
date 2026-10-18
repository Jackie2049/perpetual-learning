# 历史训练配置

这里目前保存 4 份面向 RTX 4090 的 DeepSpeed 训练配置，属于实验执行材料，不承担 Agent 引擎的调度、模型选择或运行预算配置。

| 配置 | 对应训练场景 |
|---|---|
| [lora-grpo_rtx4090.json](lora-grpo_rtx4090.json) | LoRA + GRPO |
| [moe-autoep_rtx4090.json](moe-autoep_rtx4090.json) | MoE / AutoEP |
| [opd-distill_rtx4090.json](opd-distill_rtx4090.json) | OPD 蒸馏 |
| [muon_lora_zero2_rtx4090.json](muon_lora_zero2_rtx4090.json) | Muon + LoRA + ZeRO-2 |

现有 [配置生成器](../tools/deepspeed_config_generator.py)、[实验编排脚本](../tools/rtx4090_gpu_experiment_runner.py) 和 [训练到部署笔记](../notebook/fundamentals/rtx4090-training-to-deployment-pipeline.md) 使用这些路径。文件保留原样，实际使用前按目标模型、软件版本和硬件复核。

后续建议随对应实验一起组织，无需作为独立一级板块长期保留；待实验执行材料的结构确定后，再同步迁移文件与调用路径。返回 [项目首页](../README.md)。
