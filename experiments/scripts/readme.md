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
| [verify_gspo_detach_gradient.py](verify_gspo_detach_gradient.py) | 修正后的GSPO/GSPO-token目标值与有限差分梯度对照；GRPO幅度界/clip方向诊断反例。需指定verl源码，打印hash/形状检查；NumPy模型不执行原torch/autograd，非完整训练或根因证明 |
| [verify_gmpo_clip_semantics.py](verify_gmpo_clip_semantics.py) | GMPO完整序列PG的正负优势、符号clip与mask对照，含非对称窗口映射、稳定分母和外层均值边界。需指定verl源码，NumPy/有限差分模型不执行原始函数，不证明engine/DP/训练等价 |
| [verify_gmpo_geo_mean_split.py](verify_gmpo_geo_mean_split.py) | 指定#7913报告源码，执行原始train/累积主机方法的替身控制检查；独立NumPy共享参数模型显式模拟DP平均/IS权重/固定全局分母。非真实forward/backward、optimizer、DP或上游patch验证；旧手写torch DP段错误已撤回 |
| [verify_verl_logprob_provenance.py](verify_verl_logprob_provenance.py) | 指定V1报告源码，7组成功路由+7类反例；`--check-snapshot-failures`查21组快照/退出边界；`--check-snapshot-rpc`（Python≥3.10）查6组原始注册/绑定/提交包装逻辑。actor/ref/get/TQ等均替身，不验证原生模型、快照、Ray调度、vLLM或恢复安全 |
| [verify_verl_weight_sync_control.py](verify_verl_weight_sync_control.py) | 原始ServerAdapter/CheckpointEngineManager协程体的21组本机控制与故障注入，检查发送/确认/cache/版本/finalize/resume先后；移除框架装饰器，Ray/权重/通信均替身，不证明回滚、取消、恢复或重试安全 |
| [verify_validation_estimands.py](verify_validation_estimands.py) | 固定#8303源码AST执行原始Bayes@N helper，对682二元矩阵查精确后验矩及保留trial身份的逆序控制；10组非法输入、180组有理数pass@k期望检查。`--check-identity`另查21组原始配置guard/manifest/key片段。Python3.10+/NumPy，配置替身，非完整trainer/会话/grader管道、真实IID或模型/设备验证 |
| [verify_prefix_packing_layout.py](verify_prefix_packing_layout.py) | 固定#8300 Git对象，AST执行原始布局helper及38组原始CPU测试；分组、pack/unpack、标签、dense可见域、预算与FSDP元数据复制计数。Python3.10+/支持jagged nested的torch；非完整pytest、TensorDict/模型/梯度/engine/FA2/3/DP/SP/GPU验证 |
| [verify_prefix_grouper_inputs.py](verify_prefix_grouper_inputs.py) | 从固定Git blob原样导入官方PrefixGrouper包，接原始verl输入helper，5组CPU检查token/mask/position/矩形布局，含有效token减少而pad行数增加反例。Python3.10+/torch，不安装或检出，禁用lazy fetch；非attention/模型/梯度/性能验证 |
| [test_vllm_lora_nan_h20.py](test_vllm_lora_nan_h20.py) · [test_vllm_lora_nan_targeted_h20.py](test_vllm_lora_nan_targeted_h20.py) | vLLM LoRA NaN 复现与验证 |
| [cuda_multi_stream_race_demo_h20.py](cuda_multi_stream_race_demo_h20.py) · [cuda_multi_stream_race_demo_v2_h20.py](cuda_multi_stream_race_demo_v2_h20.py) | CUDA 多流竞争复现 |
| [verify_areal_prompt_mean_split.py](verify_areal_prompt_mean_split.py) | AReaL 归约契约 CPU 参考验证（标准库，无 torch）；区分外层优化步数与内层累积，含 token_mean 对照、合法 DP dummy 调度、Megatron 默认路径 P 抵消及标量 Adam 反例；非框架集成实测，见源码笔记 §10.7 |
| [areal_prompt_mean_gpu_probe.py](areal_prompt_mean_gpu_probe.py) | AReaL prompt_mean GPU 受控实测（§10.7 判据：固定外层 M 只变内层 P、M=2 vs 4 训练行为对照 + token_mean 对照），真实 PPOActor+FSDPEngine 单卡 + 固定 batch，输出参数更新 delta_L2；2026-10-20 已在服务器 4090-1 实测完成（Qwen2.5-0.5B）：prompt_mean 跨 M 1.301 ≈ token_mean 1.273、P-spread 1.216 ≈ 1.192，**不构成 defect**，见源码笔记 §10.10 |
| [verify_deepinfer_seed_contract.py](verify_deepinfer_seed_contract.py) | 从指定 DeepInfer 源码提取 seed/EngineType 定义，在无 torch 环境检查 uint64 种子行为和调度默认值；打印输入 SHA256，非 sampler kernel 或模型一致性实测 |
| [verify_deepinfer_sampling_params.py](verify_deepinfer_sampling_params.py) | 原样加载#121的SamplingParams测试及base/#119实现，pytest collector收集53项：基线51过/2预期失败、#119全过；各加8组构造边界观察。Python3.10+/torch/pytest，包根为空命名空间，不执行conftest/config/绑定/engine或GPU；接受输入不代表引擎合法 |
| [verify_deepinfer_eplb_window.py](verify_deepinfer_eplb_window.py) | 固定main与#107 Git对象，真实CPU torch运行原始窗口3例、4组日程、策略输入反例及policy26例；`--check-step-lifecycle`另查8组原始manager/view/step/local统计条件。Python3.12+/torch/pytest；进程组/重排回调替身，不跑producer kernel、完整包、collective、实际shuffle或性能，不修改checkout |
| [verify_deepinfer_chunk_sizing.py](verify_deepinfer_chunk_sizing.py) | 固定基线/#108 Git对象，编译原始C++ chunk sizing与图片边界方法，10组成对控制；核对#95/#98该片段仍同基线。`--check-caller`另跑8组原始Python prefix helper/片段轨迹控制。Sequence/step/缓存推进是夹具、TORCH_CHECK仅本地异常；非完整Scheduler/fork/Gloo/上游pytest/KV分配或DP挂起验证，需C++20编译器 |
| [verify_deepinfer_zos_host_flow.py](verify_deepinfer_zos_host_flow.py) | 原始 Worker 主机替身；--check-event-pool 查 #54 计时引用错配；--check-so-counter 查 SO 上界与漏更新故障；--check-output-copy 编译小型 CPU 扩展核验 pybind 数组构造及原始收集/释放分支，需显式提供头文件。非完整 C++ 调度、真实输出池/语法/CUDA/通信/性能验证 |
| 同上 `--check-worker-lifecycle` | 原始 CP 命令/关闭/队列方法的单槽内存与假时钟检查；区分读标记、方法完成、父死注册和 ring timeout。非 Linux prctl、实际共享内存、多进程、socket 或 collective 验证 |
| 同上 `--check-shm-cleanup` | 原始 ShmRingBuffer/PG cleanup 与现有清理断言；真实本机映射、所有者close/unlink和派生view故障控制，仅适配torch清零。记录macOS页粒度，非完整torch/Gloo、多进程广播或设备验证 |
| [verify_deepinfer_mtp_budget.py](verify_deepinfer_mtp_budget.py) | 编译原始 MTPVarlen/估时头文件，仅替换 Sequence/TORCH_CHECK 依赖；用穷举前缀 oracle 检查 token 预算、两轮置信度与反馈边界，非 GPU 分配或性能实测；需 C++17 编译器 |
| [verify_deepinfer_mtp_commit.py](verify_deepinfer_mtp_commit.py) | 固定774ce244，原始process_outputs逐请求循环、_check_stop与缓存/在途状态定义的12组C++控制；`--check-output-identity`另跑8组原始formatter/记账、4组Python SO mask和2组离线收集控制。注入采样payload、heap pool和硬件替身，非真实sampler、原池/pybind/API或设备验证；需C++20编译器 |
| [verify_deepinfer_yoco_visibility.py](verify_deepinfer_yoco_visibility.py) | 默认编译原始 YoCo 可见域 C++ helper；--cp-reference-only 组合上游 Python CP 索引参考，检查跨 rank 身份、padding 与绝对 key 坐标。均非完整 metadata、GPU/通信或上游 L1/L2；默认模式需 C++17 编译器 |

部分脚本包含历史服务器路径或会修改所安装的框架源码；这是实验代码，运行前应查看对应脚本和验证对象，不能将其视为通用巡检入口。迁移本身不表示重新完成 GPU 验证。
