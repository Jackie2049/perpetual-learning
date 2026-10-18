# 历史训练配置

这里目前保存 4 份面向 RTX 4090 的 DeepSpeed 训练配置，属于实验执行材料，不承担 Agent 引擎的调度、模型选择或运行预算配置。

| 配置 | 对应训练场景 |
|---|---|
| [lora-grpo_rtx4090.json](lora-grpo_rtx4090.json) | LoRA + GRPO |
| [moe-autoep_rtx4090.json](moe-autoep_rtx4090.json) | MoE / AutoEP |
| [opd-distill_rtx4090.json](opd-distill_rtx4090.json) | OPD 蒸馏 |
| [muon_lora_zero2_rtx4090.json](muon_lora_zero2_rtx4090.json) | Muon + LoRA + ZeRO-2 |

现有 [配置生成器](../../../tools/deepspeed_config_generator.py)、[实验编排脚本](../../../tools/rtx4090_gpu_experiment_runner.py) 和 [训练到部署笔记](../../../notebook/fundamentals/rtx4090-training-to-deployment-pipeline.md) 使用这些路径。文件保留原样，实际使用前按目标模型、软件版本和硬件复核。

配置随 RTX 4090 实验收纳在 `experiments/rtx4090/configs/`。生成器与检查器默认使用该目录，也可通过 `--output` 或 `--config` 指定其他路径。返回 [项目首页](../../../README.md)。

相关记录：[RTX 4090 训练技巧](../../../notebook/experiments/rtx4090-training-techniques.md) · [LoRA 微调](../../../notebook/experiments/rtx4090-lora-finetune.md) · [RLHF/GRPO](../../../notebook/experiments/rtx4090-rlhf-grpo.md)。

Muon 训练脚本示例还引用单独的 `lora_rtx4090.json` adapter 配置；该文件原本未收录，本次迁移没有生成它。
