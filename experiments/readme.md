# 上机实验与执行材料

这里保存实验的代码、配置、环境快照和原始结果。[notebook/experiments/](../notebook/experiments/) 保存实验记录、分析和可复用经验，两者通过链接关联。返回 [项目首页](../README.md#上机实战)。

## 内容导航

| 位置 | 内容 | 关联记录 |
|---|---|---|
| [handoff-m5.md](handoff-m5.md) | 需要 GPU/NPU 的事项交接、资源/数据前提及停止条件；M5 可用时接续 | 对应主笔记和原始实测结果就近回写 |
| [unirl-sft-cursor/](unirl-sft-cursor/readme.md) | 原SFT DataSource/真实JSONL的预取、消费游标、身份与eval隔离；未运行trainer/optimizer | [消费不等于参数更新](../notebook/frameworks/unirl-survey.md#sft-预取与数据消费游标) |
| [deepinfer-close/](deepinfer-close/readme.md) | #175原关闭方法与13项原测试的隔离控制流对照，资源为Mock | [关闭失败与一次性标志](../notebook/frameworks/deepinfer-initial-survey.md#关闭失败后的独立清理与一次性标志) |
| [deepinfer-pool-generations/](deepinfer-pool-generations/readme.md) | #187原9测试的真实CPU共享存储/fork对照，另1项保留视图控制；pin/unpin为no-op | [分配身份与视图寿命](../notebook/frameworks/deepinfer-initial-survey.md#共享池句柄的分配身份与视图寿命) |
| [deepinfer-hf-shapes/](deepinfer-hf-shapes/readme.md) | #153原55项真实CPU safetensors对照，另各4组原加载器前缀/值顺序控制 | [布局预检与gate up顺序](../notebook/frameworks/deepinfer-initial-survey.md#checkpoint-布局预检与-gate-up-顺序) |
| [verl-ct-fallback/](verl-ct-fallback/readme.md) | #8311原172项CPU测试的隔离对照；原合成tokenizer/processor，不下载模型 | [新版选择与回退](../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#11-新版自动选择与-tokenizer-能力回退) |
| [verl-rl-insight/](verl-rl-insight/readme.md) | #8315原12项日志测试三版本隔离对照，原Mock客户端；另核实际worker lane表达式 | [日志测试与真实接线](../notebook/frameworks/verl-async-logprob-provenance.md#新版原测试与实际接线) |
| [verl-nixl-topology/](verl-nixl-topology/readme.md) | #8316两版各20组CPU控制及后续1原测试隔离对照；无原生导入/传输 | [单worker拓扑](../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑) |
| [deepinfer-civilian-port-stage2/](deepinfer-civilian-port-stage2/readme.md) | DeepInfer 平民版移植冒烟第二段：四 kernel 包 stub、架构检查放宽、llm_engine import 链、tilelang SM89 编译探针 | [平民版专题真机验证](../notebook/frameworks/deepinfer-civilian-port.md#移植冒烟第二段2026-10-22m5-服务器-4090-1-实测) |

| [kernels/fused_rms_norm/](kernels/fused_rms_norm/) | Fused RMSNorm CUDA/C++ 实现、Python 接口、测试及基准 | [算子实验记录](../notebook/experiments/rtx4090-fused-rmsnorm-cuda.md) |
| [scripts/](scripts/readme.md) | GPU 实验、故障复现及贡献验证脚本 | [GRPO 排错](../notebook/grpo-training-debugging-guide.md) |
| [rtx4090/configs/](rtx4090/configs/readme.md) | RTX 4090 DeepSpeed 训练配置 | [训练技巧](../notebook/experiments/rtx4090-training-techniques.md) |
| [environments/](environments/readme.md) | 历史本地与 GPU 机器的 Conda 环境快照 | 保留复现条件，使用时按目标机器调整 |
| [results/](results/readme.md) | 训练、分布式、算子、推理与模型架构实验的原始结果 | 结论与局限保存在对应实验笔记 |
| [run_grpo_exp1_qwen3_8b.sh](run_grpo_exp1_qwen3_8b.sh) | GRPO 基线训练启动脚本 | 历史启动参数 |
| [run_grpo_exp4_cppo_qwen3_8b.sh](run_grpo_exp4_cppo_qwen3_8b.sh) | CPPO 训练启动脚本 | 历史启动参数 |
| [run_grpo_exp5_v1_sync_cppo.sh](run_grpo_exp5_v1_sync_cppo.sh) | V1 同步 CPPO 启动脚本 | 历史启动参数 |

## 组织约定

- 新实验优先按具体课题组织，把其源码、配置和复现入口放在一起；一个课题内部可以使用 scripts、configs、results 等子目录。
- 现有跨实验脚本与历史结果集中保存在本板块的 scripts/、results/，后续明确所属课题时再就近收纳。它们不作为项目一级板块。
- 实验笔记链接实际使用的脚本、配置、环境及结果；原始数据不替代分析，脚本存在也不等于实验已经运行或结论已验证。
- 已有脚本依赖的硬件和上游版本各不相同。命令未特别说明时，从仓库根目录执行；运行前核对相关记录中的条件。
