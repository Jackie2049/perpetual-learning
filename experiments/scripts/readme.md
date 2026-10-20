# 实验与验证脚本

这里的脚本用于具体实验、故障复现和开源贡献验证；项目通用工具的入口在 [tools/](../../tools/readme.md)。返回 [实验总览](../readme.md)。

| 入口 | 用途 |
|---|---|
| [run_all_gpu_tests.py](run_all_gpu_tests.py) | 按类别运行已有 GPU 验证，默认结果写入 experiments/results/ |
| [gpu_pr_validation_experiments.sh](gpu_pr_validation_experiments.sh) | DeepSpeed、GRPO、vLLM、SGLang、verl 等验证实验的编排 |
| [gpu_experiment_helpers.py](gpu_experiment_helpers.py) | 这些验证脚本共用的设备、计时和结果保存辅助函数 |
| [bench_single-gpu-lora.py](bench_single-gpu-lora.py) | 单 GPU LoRA 基准 |
| [grpo_smoke_test_h20.py](grpo_smoke_test_h20.py) · [grpo_training_runtime_test_h20.py](grpo_training_runtime_test_h20.py) · [grpo_training_validation_test.py](grpo_training_validation_test.py) | GRPO 训练与烟雾验证 |
| [test_up_grpo_h20.py](test_up_grpo_h20.py) | UP-GRPO 损失验证 |
| [test_vllm_lora_nan_h20.py](test_vllm_lora_nan_h20.py) · [test_vllm_lora_nan_targeted_h20.py](test_vllm_lora_nan_targeted_h20.py) | vLLM LoRA NaN 复现与验证 |
| [cuda_multi_stream_race_demo_h20.py](cuda_multi_stream_race_demo_h20.py) · [cuda_multi_stream_race_demo_v2_h20.py](cuda_multi_stream_race_demo_v2_h20.py) | CUDA 多流竞争复现 |
| [verify_areal_prompt_mean_split.py](verify_areal_prompt_mean_split.py) | AReaL prompt_mean 归约切分敏感点验证（numpy 复刻，无 torch）；含 M 线性放大与 dummy-slot 膨胀用例 |

部分脚本包含历史服务器路径或会修改所安装的框架源码；这是实验代码，运行前应查看对应脚本和验证对象，不能将其视为通用巡检入口。迁移本身不表示重新完成 GPU 验证。
