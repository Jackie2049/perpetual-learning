# 上机实验与执行材料

这里保存实验的代码、配置、环境快照和原始结果。[notebook/experiments/](../notebook/experiments/) 保存实验记录、分析和可复用经验，两者通过链接关联。返回 [项目首页](../README.md#上机实战)。

## 内容导航

| 位置 | 内容 | 关联记录 |
|---|---|---|
| [verl-checkpoint-config/verify.py](verl-checkpoint-config/verify.py) | 四版完整配置树的真实Hydra三入口／五显式节点对照，[结果](verl-checkpoint-config/result.json)；原类／转换器，非#7426反射pytest、trainer或文件保存 | [配置覆盖与默认值](../notebook/frameworks/verl-checkpoint-management-reading.md#配置字段可覆盖与对象默认值) |
| [verl-disrm-contract/verify.py](verl-disrm-contract/verify.py) | #7608原三回归接三版原方法及1同输出再次调用控制，[结果](verl-disrm-contract/result.json)；HTTP／预处理／Ray为原替身，非模型／恢复验收 | [DisRM契约与部分写入](../notebook/frameworks/verl-reward-integration-reading.md#disrm-返回契约与部分写入) |
| [verl-tool-cleanup/verify.py](verl-tool-cleanup/verify.py) | main原8工具调用CPU测试及1双故障控制，[结果](verl-tool-cleanup/result.json)；原FakeTool／本地记录器，无sandbox／模型输出代码 | [工具异常与释放](../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#工具异常与资源释放的边界) |
| [verl-lora-export/verify.py](verl-lora-export/verify.py) | #7341原6配置／2无adapter CPU测试体，[结果](verl-lora-export/result.json)；5项未运行，无PEFT／融合或文件导出 | [分片重组与LoRA融合](../notebook/frameworks/verl-checkpoint-management-reading.md#分片重组与-lora-融合不是同一步) |
| [verl-reward-validation/verify.py](verl-reward-validation/verify.py) | #7217原两参数CPU测试体／三版原方法，[结果](verl-reward-validation/result.json)；真实容器、原假远程句柄，非Ray／TQ／奖励服务 | [验证标志与元数据](../notebook/frameworks/verl-reward-integration-reading.md#异步奖励的验证标志与元数据范围) |
| [verl-balance-scalars/verify.py](verl-balance-scalars/verify.py) | #7281原两项CPU测试体及同一原FP32反例的三版分区对照，[结果](verl-balance-scalars/result.json)；非trainer／DP或性能测试 | [标量类型与局部收益](../notebook/frameworks/verl-grpo-source-reading.md#python-分区的标量类型与性能范围) |
| [verl-resume-retention/verify.py](verl-resume-retention/verify.py) | #7396两原文件22项隔离CPU测试及两版原重启回归，[结果](verl-resume-retention/results.json)保留macOS别名条件失败；只删新建scratch，非真实checkpoint恢复 | [重启保留与路径身份](../notebook/frameworks/verl-checkpoint-management-reading.md#重启后的保留状态与路径身份) |
| [verl-teacher-temperature/verify.py](verl-teacher-temperature/verify.py) | #8253原默认3参数例三版CPU钩子隔离执行及旧／新候选低精度控制，[结果](verl-teacher-temperature/result.json)；非Sampler／服务／训练 | [采样与打分温度](../notebook/frameworks/verl-veomni-opd-source-reading.md#teacher-采样温度与打分温度) |
| [verl-padding-metrics/verify.py](verl-padding-metrics/verify.py) | #7785原一回归体接三版原CPU函数，[结果](verl-padding-metrics/results.json)；另一个聚合解释例，非TQ／IS／loss执行 | [Padding与序列分母](../notebook/frameworks/verl-async-logprob-provenance.md#padding-行与序列指标的分母) |
| [unirl-window-dwell/verify.py](unirl-window-dwell/verify.py) | 固定main原窗口模块CPU序列／四份配方显式标量，[结果](unirl-window-dwell/results.json)；反事实策略切换非Hydra／模型或参考parity | [停留计数与阈值坐标](../notebook/frameworks/unirl-survey.md#sde-窗口的停留步数与坐标) |
| [handoff-m5.md](handoff-m5.md) | 需要 GPU/NPU 的事项交接、资源/数据前提及停止条件；M5 可用时接续 | 对应主笔记和原始实测结果就近回写 |
| [verl-async-resume-warmup/verify.py](verl-async-resume-warmup/verify.py) | 固定main原9 warmup测试体／完整方法隔离执行，[结果](verl-async-resume-warmup/results.json)；原Mock加载，非TQ／checkpoint往返 | [库存恢复与训练历史](../notebook/frameworks/verl-checkpoint-management-reading.md#异步恢复的存活库存与训练历史) |
| [verl-selected-logprob-identity/verify.py](verl-selected-logprob-identity/verify.py) | #7712恒等式的单一FP64 CPU autograd控制，[结果](verl-selected-logprob-identity/results.json)；dense标签oracle，非作者实现／fused或性能验证 | [选定概率与梯度锚点](../notebook/frameworks/verl-fused-output-softcap.md#选定-token-概率与共享归一化项) |
| [verl-object-store-read/verify.py](verl-object-store-read/verify.py) | #6850原五下载测试体的真实memory后端／临时缓存对照，[结果](verl-object-store-read/results.json)；无云请求或checkpoint执行 | [下载与checkpoint提交](../notebook/frameworks/verl-checkpoint-management-reading.md#对象存储下载与-checkpoint-提交) |
| [verl-offload-commit/verify.py](verl-offload-commit/verify.py) | #8277原五CPU测试体与前版／单一基线控制，[结果](verl-offload-commit/results.json)；原recording handler，无异步传输 | [反向入口与重载顺序](../notebook/frameworks/verl-architecture.md#激活卸载的反向入口与重载顺序) |
| [unirl-sft-cursor/](unirl-sft-cursor/readme.md) | 原SFT DataSource/真实JSONL的预取、消费游标、身份与eval隔离；未运行trainer/optimizer | [消费不等于参数更新](../notebook/frameworks/unirl-survey.md#sft-预取与数据消费游标) |
| [deepinfer-close/](deepinfer-close/readme.md) | #175原关闭方法与13项原测试的隔离控制流对照，资源为Mock | [关闭失败与一次性标志](../notebook/frameworks/deepinfer-initial-survey.md#关闭失败后的独立清理与一次性标志) |
| [deepinfer-pool-generations/](deepinfer-pool-generations/readme.md) | #187原9测试的真实CPU共享存储/fork对照，另1项保留视图控制；pin/unpin为no-op | [分配身份与视图寿命](../notebook/frameworks/deepinfer-initial-survey.md#共享池句柄的分配身份与视图寿命) |
| [deepinfer-hf-shapes/](deepinfer-hf-shapes/readme.md) | #153原55项真实CPU safetensors对照，另各4组原加载器前缀/值顺序控制 | [布局预检与gate up顺序](../notebook/frameworks/deepinfer-initial-survey.md#checkpoint-布局预检与-gate-up-顺序) |
| [verl-ct-fallback/](verl-ct-fallback/readme.md) | #8311原172项CPU测试的隔离对照；原合成tokenizer/processor，不下载模型 | [新版选择与回退](../notebook/frameworks/verl-6779-continuous-token-agentic-rollout-reading.md#11-新版自动选择与-tokenizer-能力回退) |
| [verl-rl-insight/](verl-rl-insight/readme.md) | #8315原12项日志测试三版本隔离对照，原Mock客户端；另核实际worker lane表达式 | [日志测试与真实接线](../notebook/frameworks/verl-async-logprob-provenance.md#新版原测试与实际接线) |
| [verl-nixl-topology/](verl-nixl-topology/readme.md) | #8316两版各20组CPU控制及后续1原测试隔离对照；无原生导入/传输 | [单worker拓扑](../notebook/frameworks/verl-async-logprob-provenance.md#单-rollout-worker-的-nixl-拓扑) |
| [deepinfer-civilian-port-stage2/](deepinfer-civilian-port-stage2/readme.md) | DeepInfer 平民版移植冒烟第二段：四 kernel 包 stub、架构检查放宽、llm_engine import 链、tilelang SM89 编译探针；去 PDL 改写（5 数值 + 43 编译 + load/store 全矩阵 60/60 nosync + sync_grid 上限发现） | [平民版专题真机验证](../notebook/frameworks/deepinfer-civilian-port.md#移植冒烟第二段2026-10-22m5-服务器-4090-1-实测) |
| [deepinfer-mem-telemetry/](deepinfer-mem-telemetry/readme.md) | #151 显存遥测 helper 真机核对：base/head 单次查询语义、Allocated/Reserved/Occupied 三口径、异常回退 | [显存遥测采样与容量规划](../notebook/frameworks/deepinfer-initial-survey.md#显存遥测采样与容量规划) |
| [deepinfer-mock-chain/](deepinfer-mock-chain/readme.md) | DeepInfer mock 权重链路真机跑通（M5 定时，✅ 完成）：dummy 权重跑通 LLMEngine 初始化 + CUDA graph 捕获 + prefill/decode 全链路，generate 返回随机文本；四 stub torch fallback + 结构堵点打通，最终复现命令见 readme | [handoff-m5 mock 条目](handoff-m5.md#deepinfer-mock-权重链路跑通2026-10-23-用户授权推进除真实权重环节) |
| [verl-8281-gpu-smoke/](verl-8281-gpu-smoke/readme.md) | verl #8281 修复（PR #8313）GPU e2e 训练冒烟：正向 2 step 跑通 + 超长 prompt error 策略 fail-loudly | [handoff-m5 #8281 条目](handoff-m5.md#verl-8281-超长-prompt-截断修复--gpu-e2e-训练冒烟) |

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
