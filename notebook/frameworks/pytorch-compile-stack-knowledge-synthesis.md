# PyTorch torch.compile 知识综合图谱

> 2026-06-15 | 综合10篇源码阅读: compile e2e + Dynamo internals + FX IR + AOTAutograd + Inductor(lowering+scheduler+codegen) + custom_op + FSDP2 internals + DTensor autograd + compile tracer + aotautograd visualizer
> ★ ★ ★ 这是PyTorch compile栈的完整知识体系 → 从C-level hook到Triton codegen → 每一步都有源码级理解

2026-10-26校正：[数据选择与结构化分支](#数据选择与结构化分支)不再把torch.where与torch.cond当成可互换的if替代。新增结果仅限torch2.14.1 CPU／Dynamo eager backend与公开export；其他历史版本、兼容性和性能条目未在本轮全面复核。

## 1. 编译栈全景 (5层→10篇→完整覆盖)

```
★ ★ ★ torch.compile完整编译栈 — 5层10篇全覆盖

Layer 1: 入口 + 框架
  └───────────────────────────────────────
  torch.compile(mode, fullgraph, dynamic)  → pytorch-compile-e2e-reading.md
    → _TorchCompileInductorWrapper → torch._dynamo.optimize → OptimizedModule
    → Backend Registry → inductor backend (lazy import)

Layer 2: Dynamo (C-level → Python → FX)
  └───────────────────────────────────────
  _PyInterpreterState_SetEvalFrameFunc     → pytorch-dynamo-internals-reading.md
    → 替换CPython帧评估函数! → 最底层控制
    → 100+ VariableTracker hierarchy → TensorVariable/NNModuleVariable/FSDPManagedNNModuleVariable
    → InstructionTranslator → bytecode→FX Node → O(1) dispatch table
    → Guard system: CacheEntry+GuardedCode+8种guard → 最多64次recompile → fallback
    → Graph break: Unsupported/SkipFrame/RestartAnalysis → 多段编译

  torch._dynamo.explain(model)             → torch_compile_diagnostic.py
    → 6模式诊断 → decision/diagnose/fusion/config/guards/full

Layer 3: FX IR + AOTAutograd (中间表示 → 分离fwd/bwd)
  ┌─────────────────────┐ ┌──────────────────────────────┐
  │ FX IR               │ │ AOTAutograd                  │
  │ Node 6op+SymInt     │ │ make_fx → joint fwd+bwd      │
  │ Graph双向链表       │ │ min-cut → partition           │
  │ Interpreter+boxed   │ │ functionalization(mutation→fn)│
  │ GraphModule         │ │ CompiledFunction+Backward     │
  │ OutputGraph+Guards  │ │ invoke_subgraph(嵌套编译)     │
  │ ★ PyTorch的LLVM IR  │ │ ★ 自动gradient checkpointing │
  └─────────────────────┘ └──────────────────────────────┘
  → pytorch-fx-ir-source-reading.md
  → pytorch-aotautograd-internals-reading.md
  → aotautograd_partition_visualizer.py (可视化)

Layer 4: Inductor (FX→IR→Scheduler→Codegen)
  ┌──────────────┐ ┌─────────────────┐ ┌─────────────────┐
  │ Lowering     │ │ Scheduler       │ │ Codegen         │
  │ aten→lowerings│ │ 10轮fusion      │ │ Triton/C++      │
  │ Pointwise/   │ │ can_fuse 7条件  │ │ CachingAutotuner│
  │ Reduction/   │ │ CSE消除中间     │ │ 3 heuristic     │
  │ Extern/      │ │ Epilogue fusion │ │ Persistent/     │
  │ Fallback     │ │ GEMM不融合      │ │ Cooperative/    │
  │ ★ lazy design │ │ ★ removed_buf  │ │ Looped          │
  └──────────────┘ └─────────────────┘ └─────────────────┘
  → pytorch-inductor-triton-codegen-reading.md
  → pytorch-inductor-scheduler/lowering/codegen-source-reading

Layer 5: 运行时 + 系统集成
  ┌──────────────────────────┐ ┌──────────────────────────────┐
  │ FSDP2 + compile          │ │ custom_op / torch.library    │
  │ intentional graph breaks │ │ CustomOpDef 5 slot           │
  │ per-param DTensor        │ │ pt2_compliant_tag → compile兼容│
  │ 3-stream AllGather       │ │ make_autograd_impl → auto    │
  │ fullgraph=False(2.12)    │ │ Dynamo MOD_INLINELIST/SKIPLIST│
  │ ★ compile +15-16%        │ │ ★ 无需手写backward!          │
  │ ★ Float8 +48-50%         │ │                              │
  └──────────────────────────┘ └──────────────────────────────┘
  → pytorch-fsdp2-internals-reading.md
  → pytorch-custom-op-library-system-reading.md
  → pytorch-dtensor-autograd-reading.md
  → torch_compile_tracer.py (实测)
```

## 2. 关键设计决策 → 为什么这样设计?

```
★ ★ ★ 10个关键设计决策及其原因:

1. C-level eval_frame hook → 为什么不Python层面?
   → Python层面太慢(eval每帧1μs→C级<1μs) → C级直接拦截
   → ★ 这是Dynamo能在production使用的前提!

2. 100+ VariableTracker → 为什么这么多类?
   → Python有100+数据类型 → 每种都需要精确追踪
   → ★ TensorVariable追踪tensor → NNTensorVariable追踪参数
   → ★ FSDPManagedNNModuleVariable → FSDP2专用 → intentional breaks

3. Guard system → 为什么最多64次recompile?
   → 防止recompilation风暴 → 64次后fallback eager
   → ★ Symbolic Shapes(2.7+) → 1次编译覆盖所有seq_len → 根本解决!

4. Joint fwd+bwd tracing → 为什么不分开trace?
   → 分开trace → backward无法知道哪些中间值可以重计算
   → joint → min-cut → 最优保存/重计算 → 自动gradient checkpointing!
   → ★ ★ 这是AOTAutograd的核心价值!

5. min-cut partition → 为什么用max-flow算法?
   → min-cut theorem → 数学保证最优 → 不可能更好
   → 网络流 → 节点容量=tensor大小 → 最小化保存量
   → ★ ★ 结果: 自动选择最省内存的checkpoint策略!

6. Functionalization → 为什么需要消除mutation?
   → FX graph需要pure functional → 确定性 → 编译友好
   → in-place → out-of-place → 无side effects
   → ★ compile兼容的前提 → 所有op必须是functional!

7. Inductor IR 3层 → 为什么不是1层?
   → TensorBox(deferred shape)→StorageBox(memory planning)→Buffer(realized)
   → unrealized→realized → lazy evaluation → 最优memory planning
   → ★ pointer swings → 无需重新创建Buffer → 内存省!

8. Scheduler 10轮fusion → 为什么这么多轮?
   → 每轮可能发现新的fusion机会 → 前轮的fusion创造新机会
   → ★ can_fuse 7条件 → 逐轮检查 → 直到无新fusion

9. CachingAutotuner → 为什么不用Triton内置autotune?
   → Triton autotune → 运行时试所有config → 每次都试
   → Inductor CachingAutotuner → precompile all configs → cache to disk → process pool
   → ★ ★ 首次编译试所有 → 后续直接读cache → 无autotune开销!

10. FSDP2+compile → 为什么intentional graph breaks?
    → FSDP hooks(eager) → 参数gather/释放 → 不在编译图内
    → compute(compiled) → 编译优化 → +15-16%
    → ★ ★ hooks不编译 → 保证正确 → compute编译 → 保证速度
```

## 3. RTX 4090 Compile 实战配置

```
★ ★ ★ RTX 4090最优compile配置:

训练 (7B LoRA GRPO):
  torch.compile(mode='reduce-overhead', fullgraph=False)
  → ★ LoRA → treat_parameters_as_free_to_save → min-cut几乎不需要重计算
  → ★ ★ GRPO无critic → 省50% compute → compile只编译actor
  → ★ ★ rLLM TinkerBackend + bypass_mode → 省1个forward pass!

推理 (7B INT4):
  torch.compile(mode='max-autotune', fullgraph=True)
  → ★ 推理只有forward → 无AOTAutograd partition → 简单!
  → ★ INT4 kernel已优化 → compile Triton收益~5-10%
  → ★ vLLM内部用CUDA Graph → 不需要额外compile

调试/诊断:
  torch._dynamo.explain(model) → graph breaks数量
  TORCH_LOGS="+dynamo+aot" → 编译流程追踪
  tools/torch_compile_diagnostic.py → 6模式诊断
  tools/torch_compile_tracer.py → 实际compile追踪

FSDP2多GPU (H100集群):
  fully_shard + compile + Float8 → +48-50% MFU
  → intentional breaks → fullgraph=False
  → reshard_after_forward=True → 省内存(2×AG per module)
  → ★ HSDP → DataParallelMeshDims → 零配置!
```

## 4. 不兼容问题完整清单

```
★ ★ ★ compile不兼容的7种场景:

1. ZeRO-3 + compile ✗✗✗
   → dynamic AllGather → 每层graph break → 无限段 → partition无意义
   → ★ 根源: AllGather破坏joint graph完整性
   → 解决: 用FSDP2代替

2. FSDP1 + compile ✗✗
   → FlatParameter → 所有参数合并 → 难partition → 不兼容
   → ★ 根源: FlatParameter vs per-param DTensor
   → 解决: 用FSDP2(per-param DTensor)

3. FSDP2 fullgraph=True (PyTorch 2.12+) ✗
   → hooks不在graph内 → fullgraph=True不支持
   → ★ 2.12 breaking change → 必须fullgraph=False
   → 解决: compile before FSDP wrapping 或 fullgraph=False

4. PPO + compile (RTX 4090) ✗
   → PPO需要critic → 2×模型内存 → 48GB → 超出24GB
   → ★ 不是compile问题 → 是内存问题
   → 解决: 用GRPO代替(无critic → 17GB)

5. dynamic shapes + compile → recompilation风暴 ✗
   → 每个新shape → 重新编译 → 64次后fallback
   → ★ Symbolic Shapes(2.7+) → 根本解决
   → 解决: 固定shape或Symbolic Shapes

6. Tensor值驱动的Python分支可能产生graph break
   → torch.where是逐元素选择，输入表达式已先计算，不是惰性if
   → torch.cond是单值谓词的受限结构化分支，需要两分支都可捕获
   → 本轮只验证CPU eager backend／export，不据此认定Inductor或CUDA Graph支持

7. MoE dynamic routing + compile ✗
   → expert选择 → 数据依赖 → graph break
   → ★ verl freeze_moe_router → 固定路由 → compile兼容
   → 解决: freeze router 或 router_replay
```

## 数据选择与结构化分支

**避免graph break与避免无效分支的数值影响，不是同一个目标。** [torch.where文档](https://docs.pytorch.org/docs/2.14/generated/torch.where.html)定义逐元素数据选择；常规Python调用先求输入参数，`torch.where(pred, sqrt(x), x²)`并不会阻止sqrt先执行。[torch.cond文档](https://docs.pytorch.org/docs/2.14/generated/torch.cond.html)则将其定义为结构化控制流，以单元素谓词选择分支函数；当前仍是prototype，签名、输出元数据和副作用受限制，不是任意Python if／while的通用替代。

### 同前向不等于同梯度

[CPU控制](../../experiments/torch-cond-selection/verify.py)对同一个分段标量函数比较三种写法：正数取sqrt(x)，否则取x²。第三种仍用where，但在sqrt之前把非正位置的输入换成1，最后再按原谓词选择输出；它不改变本例正数支路的值，也不计算非法sqrt。

Python3.12.13／torch2.14.1（git `5c4886908584029761b579af026dcfb627c84070`）、CPU FP64、单线程，全部显式 `torch.compile(backend="eager", fullgraph=True)`：

| 写法 | x=−4的输出／梯度 | x=4的输出／梯度 |
|---|---|---|
| 先算sqrt与square，再where选择 | 16／NaN | 2／0.25 |
| cond在两个纯函数间选择 | 16／−8 | 2／0.25 |
| 在sqrt前准备安全输入，再where选择 | 16／−8 | 2／0.25 |

NaN来自已进入图的无效sqrt反向，不是本轮发现了where前向选错值，也不是cond自动修复了任意上游NaN。[原结果](../../experiments/torch-cond-selection/result.json)记录实际反向与非有限标志。已有[掩码统计专题](verl-grpo-core-algos-reading.md#掩码统计与梯度的数值边界)已说明遮掉最终结果不能删除更早非法运算，本轮不重跑其20组控制。

### 两分支可捕获不等于两分支都在主图计算

同一脚本用公开 `torch.export.export(..., strict=True)` 导出前向图。where主图按顺序含sqrt、square、where；cond主图含higher-order cond，另有true_graph_0中的sqrt与false_graph_0中的square。safe_where主图则在sqrt之前多一次选择安全输入。**两个cond分支都要进入捕获／检查范围**，不能把运行时择支理解为未选分支可含任意不可追踪代码、异常或全局副作用。本轮没有用计数器／print给分支加副作用来“证明惰性”。

cond的单值谓词选择整个分支，不能把逐元素mask随意改成any／all后宣称等价；向量混合正负、广播、alias／mutation及分布式控制还需其自身契约。安全输入的where也只是本例对已知数学定义域的处理，不能用常数替换掩盖实际有效数据中的错误。

eager backend证明这几段Dynamo完整捕获及本地autograd可执行，export保存图结构；没有Inductor代码生成、AOT导出反向、CUDA conditional graph、性能、模型或分布式验收。复现为 `python experiments/torch-cond-selection/verify.py --output experiments/torch-cond-selection/result.json`，无需依赖安装或GPU。此项不改框架、不新排M5任务；后续只在实际控制流迁移需求中接续，不扩谓词／dtype／backend矩阵。

## 5. 编译栈工具矩阵

```
★ ★ ★ 5个编译栈知识合成工具:

| 工具 | 类型 | 覆盖栈层 | 关键功能 |
|------|------|---------|---------|
| torch_compile_diagnostic.py | 诊断 | 1-5 | 6模式诊断 → decision/diagnose/fusion/config/guards/full |
| torch_compile_tracer.py | 追踪 | 1-5 | 4模型×6阶段×backward → 实际编译流程追踪 |
| aotautograd_partition_visualizer.py | 可视化 | 3 | min-cut partition可视化 → fwd/bwd/checkpoint/recompute |
| dtensor_sharding_playground.py | 可视化 | 5 | DTensor分片 → 5模式 → FSDP2/TP/compare/hsdp/gradient |
| ai-infra-quick-reference-card.md | 速查 | 1-5 | 3秒决策树 → 训练/推理/compile → 7框架参数表 |

★ ★ 每个工具都有可运行测试 → CPU兼容 → GPU就绪
```

## 6. 从编译栈到实战 (RTX 4090完整路径)

```
★ ★ ★ RTX 4090 从compile到部署完整路径:

Step 1: 训练 (GRPO + LoRA + rLLM Tinker)
  → rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode
  → ★ Tinker in-process → 0分布式通信 → 单GPU最快
  → ★ ★ bypass_mode → 省forward pass → 极快!

Step 2: 编译加速 (可选)
  → torch.compile(mode='reduce-overhead') → actor only
  → ★ 首次编译慢(30s) → 但缓存后<1s
  → ★ ★ LoRA → min-cut几乎无重计算 → compile收益更大

Step 3: Checkpoint → 推理
  → save_pretrained → merge LoRA → HF format
  → ★ rLLM直接HF → 最简路径! → 0步转换
  → vs ZeRO-3 → 4步转换(ZeRO→FP32→HF→INT4)

Step 4: INT4推理部署
  → HF → vLLM INT4 + INT8KV + GQA-8 + prefix caching
  → → 4,791 tok/s (7B)
  → ★ ★ EAGLE + INT4 → 9,088 tok/s → 8.3x加速!

★ ★ ★ 完整路径:
  rLLM GRPO+LoRA → merge → HF → INT4 vLLM → 4,791 tok/s
  → 最简 → 最快 → 最省 → RTX 4090最优路径!
```

## 7. 下一步学习方向

```
★ 编译栈已全面覆盖 → 下一步方向:

1. GPU实测 → 等GPU上线 → 运行tracer/benchmark/diagnostic
   → compile实测 → 不同mode对比 → FSDP2 vs ZeRO-3 benchmark

2. vLLM MRv2深入研究 → v0.23.0新默认model runner
   → 两阶段执行 → FlashInfer sampler → PP bubble elimination

3. CUDA programming → 深入理解CUDA graph + stream + event
   → 已有CUDA Graph vLLM reading → 补充PyTorch内部机制

4. 推理scaling + GRPO训练实战 → AI infra最有价值方向
   → inference-time compute → GRPO rollout_n → vLLM受益最大

5. 7框架开源贡献 → vLLM PR #45494 comment → Megatron docs
   → 真正参与开源项目 → 专家身份认证!
```
