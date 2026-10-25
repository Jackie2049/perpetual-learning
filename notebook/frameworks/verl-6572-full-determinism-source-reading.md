# verl #6572 Full Determinism vLLM Rollout — Source-Level Reading

> 2026-06-18 历史快照：PR #6572 当时OPEN，993 additions、23 files；Authors: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long。下方旧配置/文件位置不代表当前主线。
>
> 2026-10-23 更正：撤回“P9必要且足够”“SM90+天然完整确定性”“P9加batch-invariant即可保证4090完整RL训练”的推论。局部融合/批形状检查不能替代模型、采样、同步及反向的端到端证据。当前限定验证见下一节。

> 2026-10-25 更正：固定 Python hash seed 也不足以保证路由；还需相同副本身份、排列及缓存条件。下文区分已有路由修复与完整训练确定性。

## 确定性路由的哈希与副本集合

固定请求 ID 不保证跨进程选中相同副本。[verl #7666](https://github.com/verl-project/verl/issues/7666)已有报告者 hsusul 的[#7667](https://github.com/verl-project/verl/pull/7667)，2026-10-25仍open、非draft、未合并，不另实现。固定head `b2b5f4f6ce42c3689f86449037c33ba747a07106`，实际父提交／共同祖先 `4eff9ae30a23b59554758fa0ca93940605ae5cdf`，两文件85增5删；API目标base `3f9d4c0d`不是此比较祖先。固定主线 `3c6d9b43` 的完整负载均衡类AST仍与父版本相同。

[原路由](https://github.com/verl-project/verl/blob/4eff9ae30a23b59554758fa0ca93940605ae5cdf/verl/workers/rollout/router.py)在缓存未命中且full_determinism开启时，对**全部副本**做`list(self._servers)[hash(request_id) % len(self._servers)]`，不是最小负载候选间的tie-breaking。不同Python进程的字符串hash及副本注册顺序都可能改变结果；固定启动seed只处理前一项。候选改为`sorted(server_ids)[zlib.crc32(request_id.encode("utf-8")) % len(server_ids)]`，对相同字符串ID集合消除这两项依赖，仍不保证ID对应相同物理worker、模型或执行状态。

Python3.12.13／真实cachetools7.2.4的CPU对照保留两版**完整原类**，仅AST隔离导入并延迟Ray类型注解，没有Ray替身或服务器进程。作者[原测试类](https://github.com/verl-project/verl/blob/b2b5f4f6ce42c3689f86449037c33ba747a07106/tests/workers/rollout/test_router_on_cpu.py)的插入顺序与sticky／clear两个测试体未改：候选2项通过，基线顺序项失败、sticky项通过。另以相同30个`det-i`和四个启动seed `0/42/12345/random`运行两版各4个真实子进程；基线相对seed0分别有0／19／23／18个路由不同，候选均0。random一列只记录这次运行，不是固定oracle。

跨进程控制使用上述AST加载器，未执行原跨进程测试中的verl整包导入，也未运行作者报告的原生25项套件。四份完整源码快照均compile通过；这些结果只验证该Python类的选址，不认证Ray启动环境、平台间结果或模型／GPU。一次性脚本与源文件校验值见[当日日志](../../diary/2026-10-25.md)。

副本集合和缓存是另一层条件：候选中先把30个请求映射到3个副本，再加入第4个，同一进程命中sticky时0个改变；清掉30条缓存后，21个请求改变目标，与新建4副本对象一致。固定哈希不意味着扩缩容时不重映射，也不是最小迁移的consistent hashing。clear只删sticky记录，不清inflight计数；full_determinism分支继续哈希，不能沿用原方法注释把清缓存后的行为一概称作least-loaded。

[Python启动环境文档](https://docs.python.org/3.12/using/cmdline.html#envvar-PYTHONHASHSEED)说明hash seed用于解释器初始化。本机同进程先后赋`os.environ["PYTHONHASHSEED"]="0"/"42"`，同一字符串hash不变；运行时设置变量不会追溯重置当前解释器。复现实验应记录实际子进程启动环境、稳定副本身份／集合和缓存状态，而不是仅看配置里出现seed。此路由条件满足也不推出采样、权重同步、前反向或完整RL轨迹确定性。

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

## 二维低精度 logprob 的分块预算

[verl #8351](https://github.com/verl-project/verl/pull/8351)优化的是原logprobs_from_logits_v2的二维低精度分支，不是#8026的FP32 sampler公式或完整训练确定性。2026-10-23读取open、非draft，作者aoshen02；固定head `285e2dd83d131a324b440ed5464c0dc61e3cf0b4`，base及共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`，ahead2/behind0，两文件78增6删。已有作者实现，不竞争或自动应用。

### 哪一条路径被分块

[dispatcher](https://github.com/verl-project/verl/blob/285e2dd83d131a324b440ed5464c0dc61e3cf0b4/verl/utils/torch_functional.py)只有在未选择flash-attn CE、且NPU CE也不可用时才调v2；`VERL_DISABLE_FLASH_ATTN_CE=1`排除的是flash分支，不会强制覆盖NPU分支。本轮没有执行dispatcher或设备发现。

原v2对非FP32/64输入沿首轴循环。remove-padding的logits若为`[tokens,vocab]`，原来每token一次log_softmax；候选在二维分支用split按行分块，原FP32/64公式和其他维数分支的AST保持不变。新labels形状检查也只放在这个二维分支，不是全函数通用校验。

默认目标预算为32 MiB，块行数为 `max(1, budget // (vocab * element_size))`。它约束单块log_softmax输出的目标大小，不是整体峰值内存上限；当一行就超过预算时仍处理一行。词表151936、BF16时默认约110行/块。`.split()`可以只产生一个输入拆分节点，但整张反向图仍有每块的log-softmax/gather节点，不能读成整个backward只有一个节点。

### CPU 移植测试与反向保存量

[复用脚本](../../experiments/scripts/verify_verl_sampler_logprobs.py)新增`--row-chunk-pr`，完整原v2函数和[三个原测试定义](https://github.com/verl-project/verl/blob/285e2dd83d131a324b440ed5464c0dc61e3cf0b4/tests/utils/test_torch_functional.py)经AST隔离，方法体/断言不改；测试的get_device_name显式绑定cpu，不调用原设备helper。16个参数例覆盖BF16/FP16、连续/转置布局，以及单块、不整除、每块一行、大词表：本机输出和梯度的int16原始位检查全部通过。另一个原计数测试在head通过（64×1000只调用一次）；base在最后的调用次数断言预期失败，确认为64个逐token调用。base没有新预算常量，未把其16参数例宣称为全部对照失败。

补充控制固定CPU BF16、37×1000、head预算改为8192字节（4行/块），两版输出/局部梯度逐位相同。saved_tensors_hooks按storage去重记录词表宽张量，并检查实际反向图：

| 版本 | log-softmax调用及反向节点数 | 最大保存词表storage | 去重保存词表storage合计 |
|---|---|---|---|
| base逐行 | 37 | 2000字节 | 74000字节 |
| head分块 | 10 | 8000字节 | 74000字节 |

head还含一个SplitBackward0和一个CatBackward0，并非只有一个反向节点。此处字节数是选定保存张量的逻辑storage量，不是CPU RSS、CUDA allocated/reserved峰值或全部图元数据。分块减少调用/节点，不自动消除所有块为反向保存的输出；不能把32 MiB预算解释为整个actor更新只多32 MiB。另一个CPU控制确认二维BF16的错误labels形状抛明确ValueError。

Python3.12.13/torch2.9.1/pytest8.4.2，无新依赖；新模式重复一致，原#8026默认模式的四原测试体及三组控制回归通过。不是原生pytest收集、完整torch_functional suite、引擎或设备验收；本机逐位一致也不外推其他后端/构建。作者GB300、torch2.15 nightly的17项及性能/显存表均是外部报告，本项目没有GPU计时或峰值测量。真机条件仅补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不启开关、装kernel或新排训练。

```bash
python -B experiments/scripts/verify_verl_sampler_logprobs.py /path/to/verl-with-8351-objects --row-chunk-pr
# 旧#8026模式与原固定对象保持独立
python -B experiments/scripts/verify_verl_sampler_logprobs.py /path/to/verl-with-8026-objects
```

base/head torch_functional SHA256分别为 `bc7ff798e2f5c06e0121668490739814efc519ff66d4d7dd7c8fc40b4840eb03` / `7df3880cb74ff2577e92208708e310f6bfaae17fc453747604980b0237e8f3ed`，原head测试 `bd4599baddd1d7082def66cc84068bbbb24edf86cfabd3bac63b7f561541f2e2`；完整patch `41f33d8e3b59ed8c0d79025c0728efeeed398b02ce11f24f3ea9476578ab15df`。两版两个完整Python模块另AST/内存compile，共四份。

## 1. 5-Layer Determinism Architecture

```
旧稿归纳的五层接线（历史快照，并非端到端确定性的充分条件）：

Layer 1: PyTorch-level
  → enable_full_determinism(seed) → sets:
    - PYTHONHASHSEED → 须在解释器启动前设置；运行时赋值不重置当前hash，也不固定字典插入顺序
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
  → 旧GlobalRequestLoadBalancer → list(全部副本)[hash(request_id) % 副本数]，绕过least-loaded
  → 固定启动hash seed不足以固定副本顺序；已有#7667改sorted IDs + CRC32
  → 相同副本身份/集合及缓存条件另核；路由稳定不等于完整训练确定性

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
