<!-- 从原根目录 README 迁入，保留历史发现入口。 -->

# 历史工具导航

本页保存旧 README 的工具索引，不是当前推荐清单。“已验证”等状态仅为历史标注，本次未重新验收。尚在 tools/ 根目录的条目也可能是待整理的实验或演示脚本；根据 [工具价值标准](../readme.md) 逐项评估，不能仅凭文件名判断。

返回 [工具总览](../readme.md)。

| 工具 | 说明 | 原状态记录 |
|------|------|------|
| [training_recipe.py](../training_recipe.py) | 输入模型规格和 GPU 配置，生成 Megatron/DeepSpeed/DDP 训练启动命令 | 已验证 |
| [gpu_benchmark.py](../gpu_benchmark.py) | GPU 环境验证 + GEMM TFLOPS / HBM 带宽 / 混合精度 / Attention 性能测试 | 已验证 |
| [training_config_guide.py](../training_config_guide.py) | 根据模型大小和硬件推荐并行策略、batch size、梯度累积 | 已验证 |
| [ddp_train_demo.py](../ddp_train_demo.py) | PyTorch DDP 完整训练示例（多进程、梯度同步、checkpoint） | 已验证 |
| [comm_benchmark.py](../comm_benchmark.py) | 集合通信性能测试（AllReduce/AllGather/ReduceScatter 延迟和带宽） | 已验证 |
| [ai_infra_cheatsheet.py](../ai_infra_cheatsheet.py) | AI Infra 常用命令速查表（nvidia-smi/nccl/pytorch/triton） | 已验证 |
| [inference_estimator.py](../inference_estimator.py) | LLM 推理性能估算（显存/KV Cache/吞吐/延迟） | 已验证 |
| [memory_estimator.py](../memory_estimator.py) | 训练显存估算（参数/梯度/优化器/激活值逐项计算） | |
| [scaling_simulator.py](../scaling_simulator.py) | 多卡扩展效率模拟（通信占比、线性度、最优卡数） | |
| [gpu_monitor.py](../gpu_monitor.py) | GPU 实时监控（利用率/显存/温度/功耗） | |
| [triton_examples.py](../triton_examples.py) | Triton kernel 示例集（vector_add/GEMM/softmax/flash_attn） | |
| [ddp_cpu_demo.py](../ddp_cpu_demo.py) | PyTorch DDP CPU 模式 demo（无 GPU 也可跑） | |
| [collective_ops_viz.py](../collective_ops_viz.py) | 集合通信原语可视化（AllReduce/AllGather/Broadcast 数据流） | |
| [vllm_experiment.py](../vllm_experiment.py) | vLLM 推理 benchmark（不同 batch/seq 配置、Prefix Caching 测试） | 待 GPU |
| [moe_router_demo.py](../moe_router_demo.py) | MoE Router 模拟器（Top-K/Capacity 路由、负载均衡分析、Dense vs MoE 对比） | 已验证 |
| [expert_parallelism_sim.py](../expert_parallelism_sim.py) | Expert Parallelism 模拟（All-to-All 通信、EP vs TP vs DP 对比、DeepSeek-V3 EP） | 已验证 |
| [gpu_profile_experiment.py](../gpu_profile_experiment.py) | GPU profiling（GEMM/带宽/Attention/Kernel Launch/Reduction 五类操作） | 已验证 |
| [cuda_graph_demo.py](../cuda_graph_demo.py) | CUDA Graph 实验（eager vs graph 对比、batch/kernel 数量 scaling） | 已验证 |
| [flash_attention_bench.py](../flash_attention_bench.py) | Flash Attention 微基准（SDPA vs Naive、显存对比、序列长度/batch scaling） | 已验证 |
| [inference_latency_breakdown.py](../inference_latency_breakdown.py) | 推理延迟分解（逐操作耗时、GPT-2/LLaMA decode step profiling） | 已验证 |
| [quantization_bench.py](../quantization_bench.py) | 量化推理 benchmark（FP32/FP16/BF16/INT8 性能+精度+内存对比） | 已验证 |
| [pytorch_inference_bench.py](../pytorch_inference_bench.py) | PyTorch 原生推理 benchmark（KV Cache 加速比、Prefill/Decode 延迟、Batch 扩展效率） | 已验证 |
| [fp8_simulation.py](../fp8_simulation.py) | FP8 量化模拟（E4M3/E5M2 格式、缩放策略、Outlier 影响、误差传播） | 已验证 |
| [fsdp_memory_sim.py](../fsdp_memory_sim.py) | FSDP 内存模拟（DDP vs FSDP 对比、Sharding 策略、Wrapping 分析、Prefetching） | 已验证 |
| [sampling_simulator.py](../sampling_simulator.py) | LLM 采样模拟（Temperature/Top-K/Top-P/Min-P/Beam Search/重复惩罚） | 已验证 |
| [memory_allocator_sim.py](../memory_allocator_sim.py) | GPU 内存分配器模拟（Naive vs Caching vs Paged, 碎片化分析） | 已验证 |
| [prefix_caching_sim.py](../prefix_caching_sim.py) | Prefix Caching 模拟（Hash/Trie/Group 策略对比, RL 场景, 收益量化） | 已验证 |
| [continuous_batching_sim.py](../continuous_batching_sim.py) | Continuous Batching 模拟（Static vs Continuous, 调度策略, KV Cache 压力, Chunked Prefill） | 已验证 |
| [speculative_decoding_sim.py](../speculative_decoding_sim.py) | Speculative Decoding 模拟（Draft-Verify、质量影响、K 最优化、批量扩展） | 已验证 |
| [disaggregated_serving_sim.py](../disaggregated_serving_sim.py) | Disaggregated Serving 模拟（Monolithic vs P/D 分离、KV Transfer、实例比例优化） | 已验证 |
| [flash_attention_sim.py](../flash_attention_sim.py) | FlashAttention 算法模拟（tiling、online softmax、IO 分析、block size 影响） | 已验证 |
| [sequence_parallel_sim.py](../sequence_parallel_sim.py) | 序列并行模拟（Ulysses vs Ring Attention vs USP、通信量、扩展效率） | 已验证 |
| [gemm_roofline.py](../gemm_roofline.py) | GEMM Roofline 分析（Prefill vs Decode 瓶颈、算术强度、GPU 对比） | 已验证 |
| [tensor_parallel_sim.py](../tensor_parallel_sim.py) | 张量并行通信分析（TP 效率、通信占比、模型规模、推荐策略） | 已验证 |
| [profiling_guide.py](../profiling_guide.py) | LLM 推理性能诊断（Prefill/Decode 分解、KV 压力、吞吐估算、瓶颈诊断） | 已验证 |
| [pipeline_parallel_sim.py](../pipeline_parallel_sim.py) | 流水线并行分析（GPipe vs 1F1B 气泡、micro-batch 影响、3D 并行策略推荐） | 已验证 |
| [training_optimizer_sim.py](../training_optimizer_sim.py) | 训练优化模拟（梯度累积、混合精度、梯度检查点、综合显存估算） | 已验证 |
| [collective_comm_sim.py](../collective_comm_sim.py) | 集合通信深度分析（Ring/Tree AllReduce、AllGather/ReduceScatter、通信重叠） | 已验证 |
| [nccl_tuning_sim.py](../nccl_tuning_sim.py) | NCCL 调优参数分析（算法/协议选择、通道调优、跨节点拓扑、速查表） | 已验证 |
| [checkpoint_perf_sim.py](../checkpoint_perf_sim.py) | Checkpoint 性能分析（大小估算、保存策略、加载恢复、存储推荐） | 已验证 |
| [dataloader_perf_sim.py](../dataloader_perf_sim.py) | DataLoader 优化分析（加载瓶颈、Prefetch 预取、存储对比、调优推荐） | 已验证 |
| [ray_schedule_sim.py](../ray_schedule_sim.py) | Ray 分布式调度模拟（Actor/Task 调度、Placement Group、RLHF PPO） | 已验证 |
| [long_context_serving_sim.py](../long_context_serving_sim.py) | 长上下文推理模拟（KV Cache 压力、Chunked Prefill、Sliding Window、Prefix Caching、服务策略） | 已验证 |
| [inference_planner.py](../inference_planner.py) | LLM 推理端到端规划器（单卡部署、多卡并行策略、Batch 优化、成本效率） | 已验证 |
| [kv_cache_offload_sim.py](../kv_cache_offload_sim.py) | KV Cache 分层存储模拟（GPU/CPU/SSD 三层容量、Offload 性能、服务策略、成本优化） | 已验证 |
| [chunked_prefill_sim.py](../chunked_prefill_sim.py) | Chunked Prefill 模拟（无 chunking vs chunked、chunk size 影响、并发调度、Prefix Caching 组合） | 已验证 |
| [kv_cache_quant_sim.py](../kv_cache_quant_sim.py) | KV Cache 量化模拟（FP16/FP8/INT4/2-bit 对比、Batch 影响、组合优化、成本效率） | 已验证 |
| [serving_framework_compare.py](../serving_framework_compare.py) | LLM Serving 框架对比（vLLM/SGLang/TRT-LLM 吞吐、KV 共享、结构化生成、扩展效率） | 已验证 |
| [inference_cost_calculator.py](../inference_cost_calculator.py) | LLM 推理部署成本计算器（GPU 选型、量化策略、场景成本、能效分析） | 已验证 |
| [cuda_kernel_simulator.py](../cuda_kernel_simulator.py) | CUDA Kernel 执行模拟器（Grid/Block、内存层次、Occupancy、Roofline 分析） | 已验证 |
| [cuda_stream_simulator.py](../cuda_stream_simulator.py) | CUDA Stream 并发模型模拟器（通信计算重叠、TP overlap、vLLM 多 Stream） | 已验证 |
| [nccl_tuning_cheatsheet.py](../nccl_tuning_cheatsheet.py) | NCCL 调优速查表（算法选择、通道调优、多节点配置、问题排查） | 已验证 |
| [llm_latency_estimator.py](../llm_latency_estimator.py) | LLM 推理延迟估算器（Prefill/Decode 分解、TP 影响、场景估算、GPU 对比） | 已验证 |
| [model_compression_analyzer.py](../model_compression_analyzer.py) | LLM 模型压缩策略分析器（蒸馏/剪枝/量化/综合策略对比、生产部署选型） | 已验证 |
| [dp_serving_simulator.py](../dp_serving_simulator.py) | Data Parallel Serving 模拟器（DP vs TP 扩展、MoE EP、Wave 协调、DBO、生产选型） | 已验证 |
| [engine_core_simulator.py](../engine_core_simulator.py) | Engine Core 模拟器（step 循环、Batch Queue、ZMQ 通信、KV auto-fit、GPU 利用率） | 已验证 |
| [triton_practice.py](../triton_practice.py) | Triton Kernel 实战练习（Vector Add、Fused OP、Softmax、Tiled GEMM、Flash Attention） | 已验证 |
| [rlhf_training_simulator.py](../rlhf_training_simulator.py) | RLHF/GRPO 训练模拟器（PPO vs GRPO、KL 惩罚、Prefix Caching、分布式扩展） | 已验证 |
| [distributed_training_comm_simulator.py](../distributed_training_comm_simulator.py) | 分布式训练通信模拟器（AllReduce、ZeRO、MoE A2A、1F1B overlap、生产配置） | 已验证 |
