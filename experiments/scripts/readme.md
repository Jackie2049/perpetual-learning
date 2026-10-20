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
| [verify_areal_prompt_mean_split.py](verify_areal_prompt_mean_split.py) | AReaL 归约契约 CPU 参考验证（标准库，无 torch）；区分外层优化步数与内层累积，含 token_mean 对照、合法 DP dummy 调度、Megatron 默认路径 P 抵消及标量 Adam 反例；非框架集成实测，见源码笔记 §10.7 |
| [areal_prompt_mean_gpu_probe.py](areal_prompt_mean_gpu_probe.py) | 由 M5 单一维护的 AReaL GPU 探针原稿，GPU 集成待验证；M1 的离线修整仅为未应用参考 patch，见 [交接说明](../../open-source/areal/handoff-m5.md)，不要据草稿预检结果判断原稿可运行 |
| [verify_deepinfer_seed_contract.py](verify_deepinfer_seed_contract.py) | 从指定 DeepInfer 源码提取 seed/EngineType 定义，在无 torch 环境检查 uint64 种子行为和调度默认值；打印输入 SHA256，非 sampler kernel 或模型一致性实测 |
| [verify_deepinfer_zos_host_flow.py](verify_deepinfer_zos_host_flow.py) | 原始 Worker 主机替身；--check-event-pool 查 #54 计时引用错配；--check-so-counter 查 SO 上界与漏更新故障；--check-output-copy 编译小型 CPU 扩展核验 pybind 数组构造及原始收集/释放分支，需显式提供头文件。非完整 C++ 调度、真实输出池/语法/CUDA/通信/性能验证 |
| [verify_deepinfer_mtp_budget.py](verify_deepinfer_mtp_budget.py) | 编译原始 MTPVarlen/估时头文件，仅替换 Sequence/TORCH_CHECK 依赖；用穷举前缀 oracle 检查 token 预算、两轮置信度与反馈边界，非 GPU 分配或性能实测；需 C++17 编译器 |
| [verify_deepinfer_yoco_visibility.py](verify_deepinfer_yoco_visibility.py) | 默认编译原始 YoCo 可见域 C++ helper；--cp-reference-only 组合上游 Python CP 索引参考，检查跨 rank 身份、padding 与绝对 key 坐标。均非完整 metadata、GPU/通信或上游 L1/L2；默认模式需 C++17 编译器 |

部分脚本包含历史服务器路径或会修改所安装的框架源码；这是实验代码，运行前应查看对应脚本和验证对象，不能将其视为通用巡检入口。迁移本身不表示重新完成 GPU 验证。
