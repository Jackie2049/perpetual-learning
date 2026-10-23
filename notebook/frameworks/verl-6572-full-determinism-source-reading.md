# verl #6572 Full Determinism vLLM Rollout — Source-Level Reading

> 2026-06-18 历史快照：PR #6572 当时OPEN，993 additions、23 files；Authors: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long。下方旧配置/文件位置不代表当前主线。
>
> 2026-10-23 更正：撤回“P9必要且足够”“SM90+天然完整确定性”“P9加batch-invariant即可保证4090完整RL训练”的推论。局部融合/批形状检查不能替代模型、采样、同步及反向的端到端证据。当前限定验证见下一节。

## Actor 前向与 logprob 公式的一致性

[verl #8026](https://github.com/verl-project/verl/pull/8026)研究训练端与vLLM rollout对齐，不能与旧#6572的运行间复现保证混为一谈。2026-10-23 API读取open、head `6a7110b42ae45068b8c7d1c20c56db841c5283af`，目标base `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb`、实际祖先 `831c5d86e58972ef2e26f00345fde1be700611cf`（ahead6/behind4），16文件406增7删。已有作者cr-gao负责，不新增竞争实现。

### 三个不同的比较对象

| 比较 | 所需条件 | 本轮证据 |
|---|---|---|
| 同一logits算出相同logprob | dtype、公式、温度/词表支持及计算路径一致 | 原4测试体与补充CPU控制 |
| 改变批形状仍有相同actor前向 | 覆盖实际模型算子、硬件与kernel；再对齐rollout | 源码/作者报告，本机未测 |
| 更新和恢复后完整轨迹相同 | 还需采样、数据顺序、同步、梯度归约和optimizer等匹配 | 不由前两项推出 |

[新helper](https://github.com/verl-project/verl/blob/6a7110b42ae45068b8c7d1c20c56db841c5283af/verl/utils/torch_functional.py)把logits转FP32，在**完整词表**做log_softmax再gather，返回FP32。它不应用temperature或top-k/top-p；actor接线在调用前缩放logits。相同数学恒等式`选中logit−logsumexp`不保证浮点逐元素精确相等，直接BF16 log_softmax也不是同一数值路径。

函数按首轴split（默认2048），逐块用非重入checkpoint重算；不保留每块完整FP32 log-softmax输出，但需要保留重算输入，forward/backward仍有块级FP32临时张量。三维输入的首轴chunk不等于把全部前导维flatten成固定token行数；这不是零额外内存或完整actor峰值的保证。`inplace_backward`参数为调用兼容而保留，实际未使用。

### 真实 CPU 数学检查

[复查脚本](../../experiments/scripts/verify_verl_sampler_logprobs.py)AST隔离三个原函数，并调用作者四个未改测试体，真实torch2.9.1/Python3.12.13，无包壳、引擎或vLLM导入。四项通过：FP32公式/shape、区别BF16原路径、chunk大小1/7/50/2048与整块相同、BF16输入的局部反向与参考精确相同且未保存FP32词表输出。

补充三组控制也通过：固定seed123的50×4096 FP32 logits有31/50个结果不同于原gather−logsumexp，最大绝对差`3.814697265625e-6`；两个inplace_backward取值都不改输入、局部VJP与参考相同；温度输入变化及截断支持分别改变结果。三项都只是同logits上的CPU数学控制，不复现作者大词表GPU计数或实际sampler。

保存张量检查须区分输入和中间结果：本例保存的词表宽FP32切片只别名原输入，checkpoint仍需这些输入来重算。不能把“没有保存FP32输出”说成“没有保存FP32张量”。复跑结果一致。

### 入口环境与进程级覆盖

固定head在main_ppo的ray.init之前导出cuBLAS/vLLM相关环境，再由worker安装vLLM aten overrides；它不只是把一个布尔值传给单次forward。worker将初始化器对允许名单外的环境改动恢复，保留训练自己的NCCL配置；full_determinism随后执行，相关设置由它最终决定。aten替换仍是**进程全局**的，actor/ref共进程时不保证只影响actor，恢复环境变量也不会卸载已注册的算子。

FSDP默认不开batch_invariant，打开时拒绝use_fused_kernels，因为后者自己算logprob；需要训练进程可导入匹配vLLM。候选将两处非融合logprob调用切换到helper，熵与sum_pi_squared仍走原路径，剩余attention、norm、RoPE和MoE还需逐模型对齐。不能由helper四项通过宣称引擎开关、默认/拒绝路径或全部CPU测试已通过；本机未运行这些原生导入/环境替身测试。

作者报告的2×H20、截为4层的Qwen3-Omni thinker在额外模型算子对齐后，两步含更新/权重同步共8192 token零mismatch；同时明确不同布局的gradient norm仍不同。它不是本项目、4090或任意模型的完整训练确定性证明。新增raw logprob计数按转FP32后的数值不等统计，不是底层bit pattern检查；空有效域/未更新前等条件也须结合真实调用范围解释，不以一个零计数认证全链路。

复现：`python -B experiments/scripts/verify_verl_sampler_logprobs.py /path/to/verl-git`。原torch_functional SHA256 `00f2e686418dd3b304f8ed81ed6e15d3a8773f2c0a25a5a037045023004f95bd`，原测试 `e810d1bbef1d585e1597e62b104ae17aff2db846f28d33f06ed909ada7dcc9e2`；13个head完整Python模块仅AST/compile，无GPU发现或kernel/模型执行。真机条件补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动启开关/装vLLM/改模型或重开#8250因果诊断。

---

## 1. 5-Layer Determinism Architecture

```
旧稿归纳的五层接线（历史快照，并非端到端确定性的充分条件）：

Layer 1: PyTorch-level
  → enable_full_determinism(seed) → sets:
    - PYTHONHASHSEED → stabilizes Python hash values, not insertion-ordered dict semantics
    - CUBLAS_WORKSPACE_CONFIG → deterministic cuBLAS
    - FLASH_ATTENTION_DETERMINISTIC → deterministic flash attention
    - seeds all RNGs (torch, numpy, random)
    - torch.use_deterministic_algorithms(True, warn_only=True)
    - disables cuDNN benchmarking

Layer 2: Environment propagation
  → main_ppo.run_ppo() sets 3 env vars before ray.init():
    - PYTHONHASHSEED → from rollout.seed → all Ray actors
    - VERL_FULL_DETERMINISM → signals subprocesses to apply determinism
    - VLLM_BATCH_INVARIANT → 请求启用实际版本/模型/硬件覆盖的批形状不变路径
  → Forwarded via PPO_RAY_RUNTIME_ENV to all Ray actors

Layer 3: vLLM batch invariance + per-request seed
  → VLLM_BATCH_INVARIANT=1 → activates vLLM batch_invariant.py (984 lines!)
  → Each generate() call → SamplingParams.seed = replica_rank + config.seed
  → Resets RNG per request → deterministic sampling within batch

Layer 4: Priority scheduling + deterministic routing
  → Each sample gets globally unique priority (global_priority_offset + sample_index)
  → SingleTurnAgentLoop → request_id=f"det-{priority}" → not random UUID!
  → GlobalRequestLoadBalancer → hash(request_id) % len(candidates) → deterministic tie-breaking
  → PYTHONHASHSEED frozen → hash() deterministic → routing deterministic!

Layer 5: Reward model serialization
  → vLLM /classify endpoint → no priority/batch invariance support
  → full_determinism=true → RewardModelManager forces max_num_seqs=1
  → Serializes RM inference one request at a time → deterministic but SLOW!
```

---

## 2. 历史算子覆盖推测

旧稿根据当时vLLM实现，推测RMSNorm或reduction融合可能绕过batch-invariant算子。该假设需要固定vLLM/Inductor版本、硬件、实际graph与调用路径核对；既没有证明所有SM89模型都需要P9，也没有证明SM90+天然具备完整确定性。TMA/WGMMA或设备代次本身不是端到端数值保证。

P9的局部fusion检查不能证实训练、rollout和采样全过程一致。本项不重新部署P9、不据旧推测宣告新缺陷或恢复已撤回的贡献方案。

---

## 3. Configuration Reference

```
★★★★★★★★★ verl #6572 determinism config parameters:

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| rollout.full_determinism | false | Rollout | Enables deterministic rollout generation |
| rollout.seed | 42 | Rollout | Base seed; each replica uses replica_rank + seed |
| actor.fsdp_config.full_determinism | false | Actor | Enables deterministic PyTorch ops |
| ref.fsdp_config.full_determinism | false | Ref model | Enables deterministic PyTorch ops |
| reward_model.rollout.full_determinism | false | RM | Enables deterministic RM (max_num_seqs=1) |
| reward_model.rollout.seed | 42 | RM | Base seed for RM vLLM server |

★★★★★★★★★ Quick start config:
  actor_rollout_ref.rollout.full_determinism=true
  actor_rollout_ref.rollout.seed=42
  actor_rollout_ref.actor.fsdp_config.full_determinism=true
  actor_rollout_ref.ref.fsdp_config.full_determinism=true
  reward.reward_model.rollout.full_determinism=true
  reward.reward_model.rollout.seed=42
```

---

## 4. 性能与适用条件

旧稿的固定10–30%吞吐损失及“4090可绕过full_determinism用于生产”没有本项目端到端测量支持，撤回。序列化奖励模型可能牺牲并发，但具体代价依模型、调度和资源而异，不按批大小直接推算整个训练吞吐。

是否启用相关开关，应以实际版本、模型算子覆盖、复现目标和性能对照为依据；前向对齐不能代替反向和恢复检查，也不自动成为生产配置建议。

---

## 5. Backend Support 历史快照

```
★★★★★★★★★ Backend determinism support from #6572:

| Backend | Rollout | Reward model |
|---------|---------|--------------|
| vLLM | ✅ (batch_invariant + seed) | ✅ (serialized max_num_seqs=1) |
| SGLang | ❌ | ❌ |
| TensorRT-LLM | ❌ | ❌ |

★★★★★★★★★ SGLang NOT supported yet:
  → SGLang当时的kernel覆盖与调度接线需分别核对，不能据覆盖数量比较整体确定性
  → But: no priority scheduling support → no request_id deterministic routing
  → No per-request seed injection → SamplingParams.seed not implemented for SGLang backend
  → Future: SGLang deterministic rollout possible → needs seed + priority support

★★★★★★★★★ vLLM is the ONLY backend with determinism support:
  → VLLM_BATCH_INVARIANT → batch-independent outputs
  → SamplingParams.seed → per-request RNG reset
  → Priority scheduling → deterministic request ordering
  → 覆盖受版本、模型与硬件限制，不能据SM代次认定complete
```

---

## 6. Multi-turn Limitations

```
★★★★★★★★★ Multi-turn agent determinism limitations:

  → tool_agent_loop → does NOT pass priority → request ordering NOT deterministic
  → Only single_turn_agent_loop has full determinism support
  → Multi-turn determinism → future work → needs priority propagation across turns

★★★★★★★★★ RTX 4090 GRPO implication:
  → single-turn缩小了请求协议范围，但不自动认证全部模型和训练确定性
  → Multi-turn agent training → NOT deterministic yet → only for debugging
  → #6572与P9的局部接线不构成完整SM89 GRPO确定性证明
```

---

## 7. Changed Files (23 files, 993 additions)

```
★★★★★★★★★ Key files changed in #6572:

Documentation:
  → docs/advance/determinism.md (NEW, 143 lines) → comprehensive 5-layer determinism doc
  → docs/index.rst (+1 line) → add determinism.rst to docs index

Config additions:
  → 5 _generated_*.yaml files → full_determinism + seed config fields
  → verl/trainer/config/algorithm.py → RolloutConfig + full_determinism + seed fields

Core implementation:
  → verl/experimental/agent_loop/agent_loop.py → priority injection + request_id deterministic
  → verl/experimental/agent_loop/single_turn_agent_loop.py → priority + deterministic request_id
  → verl/experimental/reward_loop/reward_model.py → max_num_seqs=1 for RM determinism
  → verl/trainer/main_ppo.py → PYTHONHASHSEED + VERL_FULL_DETERMINISM + VLLM_BATCH_INVARIANT propagation

Tests:
  → tests/workers/rollout/rollout_vllm/test_vllm_generation_determinism.py (NEW, 308 lines)
  → tests/experimental/reward_loop/run_determinism_e2e_with_rm.py (NEW, 381 lines)
  → .github/workflows/vllm.yml → determinism test in CI
```

---

## 8. P9 与确定性保证的边界

P9限制某些fusion，最多影响其中一段算子的执行路径，不能替代seed、数据顺序、请求路由、权重同步、模型其他算子或训练反向的检查。“所有SM通用确定性”“填补全部缺口”“完整SM89 GRPO确定性”均不再成立为本项目结论。

若将来重新研究，必须先明确要比较的是同输入单算子、同权重模型前向、采样输出，还是含更新/恢复的训练轨迹；分别给出版本与实际对照。当前#8026的CPU公式检查仅对应第一层，真机条件沿用原M5入口，不自动接续旧P9集成计划。

---

## Key Findings Summary

- batch-invariant前向、相同logprob公式和完整训练确定性是不同保证。
- #8026原四项数学测试体及补充CPU控制通过，非引擎或GPU验收。
- 既有P9局部证据不足以推出完整SM89/SM90+训练确定性；固定吞吐损失和无full_determinism的生产推荐撤回。
- 旧#6572配置与后端支持表只作历史快照，使用前须核对实际版本和模型路径。

---

## References

- PR #6572: https://github.com/verl-project/verl/pull/6572
- vLLM batch_invariant.py: notebook/frameworks/vllm-batch-invariant-source-reading.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Fusion Guard PR: open-source/pytorch/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
