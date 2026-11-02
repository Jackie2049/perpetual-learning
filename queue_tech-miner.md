# 技术掘金任务队列

更新时间：2026-11-02。规则见 [AGENTS.md](AGENTS.md)，现场见 [CONTEXT.md](CONTEXT.md)。技术秘书优先；以下外部状态均来自迁移前记录，不代表本次重新查询。

## 当前方向与选题原则（2026-11-02 用户指令，覆盖此前的框架与选题范围）

**一、范围不限于已熟悉的框架。** 此前聚焦 UniRL／verl／DeepInfer 是因为熟悉，不是因为这些就是全部值得投入的对象。技术掘金的范围是整个主流开源 AI Infra 生态，重点是后训练／RL 训练、推理引擎、分布式训练、数据与评测等与既有积累可迁移的相邻领域。选择对象看"问题价值 × 可验证性 × 采纳概率"，不看"是否听说过"。

- **常规扫描池**（非穷尽，按需扩展并登记到本队列）：推理与部署——vLLM、SGLang、TensorRT-LLM、LMDeploy、llama.cpp、MLC；训练与后训练——verl、AReaL、OpenRLHF、NeMo-RL、SkyRL、slime、rLLM、TRL、DeepSpeed、Megatron-LM、FSDP/TorchTitan、axolotl、unsloth；Agent／评测——Agent Lightning、OpenHands、SWE-bench 系、tau-bench 系；基础设施——Ray、NCCL/DeepEP、FlashInfer、FlashAttention、Triton、Mooncake、LMCache、KTransformers、SGLang Router／Gateway；以及新出现的活跃项目。
- **扩池不是无差别扫**：每次只把 1–2 个新对象纳入一篇短评估（活跃度、贡献门槛、与既有知识的迁移价值、潜在问题类别），写进 `open-source/<repo>/` 后决定是否深入。不做全生态平铺清单。

**二、贡献形式不限于修已有 issue 的 bug。** 修别人已提的 bug 采纳率最高，是基线渠道而不是唯一渠道，更不能成为每日扫描未认领 issue 的全部工作。四条渠道按"可发现性×价值"并列：

| 渠道 | 说明 | 典型证据要求 |
|---|---|---|
| **A. 修已知 bug** | 社区已提 issue、无人认领且确有方案（性价比最高，基线） | 复现 + 根因 + 回归用例 |
| **B. 主动发掘问题提 issue** | 自己通过读源码、跑实验、看文档不一致发现的缺陷或设计问题，先提 issue（可附最小复现） | 最小复现脚本 + 影响面判断 + 不会误报的边界说明 |
| **C. 做优化 PR** | 不等 bug：性能优化、内存/通信优化、可观测性、易用性、文档与测试基建等改进型 PR | **同环境基线对照 + 可复现 benchmark + 明确不改变正确性** |
| **D. 参与设计讨论／RFC** | 对架构提案给出有证据的评审意见或替代方案 | 指向具体代码路径与实测/理论依据 |

- **优先级**：有算力窗口时 C（优化 PR）价值最高、也最能区别于"bug 闭环 agent"；B 次之（先发现在先）；A 是随时可做的稳态工作；D 依赖已有深入阅读。
- **B/C 类的前提**：必须建立在**先有理解**的基础上——读源码、跑同栈基线、找出可量化的差距，再动手。不允许为凑贡献而制造"看起来像问题"的改动。
- **C 类的额外要求**：必须给出"改前/改后"同环境对照数据（延迟、吞吐、显存、通信量至少一项），并说明是否改变数值结果；没有基线的优化 PR 不提交。

**三、每日工作不应只是扫描 issue。** 按四渠道轮转：每次触发先看有无 A 类可做（快速），再投入 B/C 类的深度工作（读源码、跑实验、找可量化差距）。连续多轮只有 A 类扫描而无 B/C 产出，说明方向选择有问题，应换对象而不是继续扫。

**已有的具体方向**：2026-10-23 用户曾指定两周聚焦 UniRL 与 verl，该聚焦已于 2026-11-02 由本节的扩池要求覆盖——verl／UniRL 仍是候选池成员，但不再享有排他优先级；DeepInfer 为常规关注。

## TM-001 — verl 学习与贡献

- **问题与价值**：围绕 RL 训练生命周期、权重同步与量化路径寻找可验证的实际问题，形成可复用理解或贡献。
- **优先级 / 负责人 / 状态**：掘金重点；M1 做本地学习、M5 按真机分工补证；等待（已有候选待决，新问题须另有证据）。
- **下一步与等待条件**：#8313 回复草稿 v2 待用户逐字审定；#8250 补证待授权。#7382 Actor 创建/取消边界、#7971 共享重载研究待实质反馈或独立缺口，不因状态无变化重复扫描。
- **停止条件**：无新证据、缺少验证条件或候选已失效时停止重复投入；不得自行回复上游或推送已关联 PR 的 head 分支。
- **成果入口**：[tracker](open-source/verl/tracker.md)、[回复草稿](open-source/verl/draft-8313-review-reply.md)、[生命周期笔记](notebook/frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md)、[量化审计](notebook/frameworks/verl-quant-layer-audit.md)。

## TM-002 — UniRL 学习与贡献

- **问题与价值**：理解训练架构并验证贡献候选，当前跟踪 #523 anchor lifecycle 修复及 PR #629。
- **优先级 / 负责人 / 状态**：掘金重点；沿用原候选负责人，真机验证归 M5，不重新分派；等待。
- **下一步与等待条件**：2026-11-02 已按用户指示重开 #629，等待维护者 review；L1 权重验证需要 ≥8 卡与存储。新反馈出现后准备本地可审阅方案，上游动作另需逐项授权。
- **停止条件**：无新反馈或无算力条件时不重复验证；历史重开授权不延伸为代码更新或评论授权。
- **成果入口**：[候选](open-source/unirl/candidate-523-anchor-lifecycle.md)、[调研](notebook/frameworks/unirl-survey.md)。

## TM-003 — DeepInfer 生态与贡献跟踪

- **问题与价值**：理解生产推理框架的设计与贡献可行性，避免在高门槛方向重复投入。
- **优先级 / 负责人 / 状态**：常规关注；沿用原贡献负责人；等待。
- **下一步与等待条件**：用户 fork 的 DeepInfer-Go PR #2 待用户审视；出现新源码、反馈或独立问题后再选验证动作。硬件适配项目见 [TS-002](queue_tech-secretary.md#ts-002--deepinfer-平民硬件适配)，不在两队列重复调度。
- **停止条件**：没有新证据时停止例行重复检查；上游行动仍需用户明确授权。
- **成果入口**：[贡献入口](open-source/README.md)、[生态判断](notebook/frameworks/deepseek-ecosystem-pr-adoption.md)。

## TM-005 — verl `use_prefix_grouper` 失效取证（B 类：主动发掘）

- **问题与价值**：`actor.use_prefix_grouper=True` 在 FSDP 引擎上**不启用前缀共享前向**（引擎侧无生产者），但 `ray_trainer` 仍按该开关做组级均衡并跳过批内重排——用户只付约束代价、拿不到加速，且无任何提示。官方示例还同时开了 README 声明互斥的 `use_dynamic_bsz=True`。
- **优先级 / 负责人 / 状态**：B 类候选，M1 本地取证完成，**等待用户决定是否对外**。
- **已有证据**：对 main `6879e1ab07`（2026-11-02）的 AST 静态取证，脚本自证 exit=0、干净环境复现通过；上游 #7202 曾修此接线但**关闭未合并**，维护者表态转向 MAGI attention（#6689 仍 open）。**影响面已量化**（直接跑上游 `seqlen_balancing.py` 源码）：组级均衡 vs 逐条均衡的 DP 负载不均衡在批 1024 时 **+1.27%**、批 256 时 **+6.61%**、批 16384 时 **+0.02%**；跳过批内重排按 §13.39 不构成损失。
- **下一步与等待条件**：① 是否以「文档／示例修订」为最小对外动作（成本最低、最可能被接受）；② 是否接受"维护者已选 MAGI 路线、补接线大概率被再关"的判断而不提修复类 issue。**两者都需用户逐项授权，不得自行发布**。
- **停止条件**：用户否决对外动作，或上游补上接线／删除该选项（脚本会返回 1 提示结论失效）。
- **成果入口**：[取证材料](open-source/verl/evidence/prefix-grouper-inert/readme.md)、[影响量化](experiments/verl-prefix-grouper-side-effects/readme.md)、[tracker](open-source/verl/tracker.md)。

## TM-006 — verl `tiled_mlp` 配置无消费路径（B 类：主动发掘）

- **问题与价值**：`actor_rollout_ref.model.tiled_mlp` 有正式配置项与官方文档，但 `verl/workers/engine/` 下 **10 个引擎目录全部 grep 为 0**，生产调用点从不传 `use_tiled_mlp` ⇒ 配置完全无效且无提示。比 TM-005 更干净（无副作用）。
- **优先级 / 负责人 / 状态**：B 类候选，M1 取证 + M5 真机冒烟均完成，**等待用户决定是否对外**。
- **已有证据**：[取证材料](open-source/verl/evidence/tiled-mlp-unwired/readme.md)；[pyflag 扫描](tools/pyflag/scan-verl-6879e1ab07.json)（`mode=never`）；**M5 4090 真机冒烟（2026-11-02）**：补齐丁后 FSDP2 下 `enabled=True` 真实 patch 到 MLP 类、前反向跑通、fp64 数值精确等价；**不打补丁时与 `enabled=False` 逐位相同**；**FSDP1 下开启硬报错**（storage 越界）；**verl 默认梯度检查点开启时无显存收益**（5.46 GB 不变），关掉才 16.39 → 8.74 GB（−46.7%，4 片饱和）。见 [experiments/verl-tiled-mlp-fsdp-smoke](experiments/verl-tiled-mlp-fsdp-smoke/readme.md)。
- **下一步与等待条件**：这条不涉及路线分歧（不像 prefix_grouper 那样有 MAGI 替代），"补 3 行接线"或"配置校验期拒绝"都不需要架构决策。**本地补丁已就绪且真机验证完成**：[patches/tiled-mlp-wiring](open-source/verl/patches/tiled-mlp-wiring/readme.md)。**若提 PR，建议同时加一条配置期校验**（`tiled_mlp.enabled=True` 且 `strategy != "fsdp2"` 时抛明确错误），否则把 FSDP1 的静默无效换成难懂的运行期崩溃；PR 描述也不能笼统写「省显存」。**对外仍需用户逐项授权**。
- **停止条件**：用户否决，或上游接上／删除该配置。
- **成果入口**：[取证](open-source/verl/evidence/tiled-mlp-unwired/readme.md)、[工具](tools/pyflag/readme.md)、[真机冒烟](experiments/verl-tiled-mlp-fsdp-smoke/readme.md)。

## TM-007 — 跨框架布尔开关静默失效扫描（B 类：方法沉淀）

- **问题与价值**：TM-005/006 是同一类模式的两个实例。已把方法沉淀为 [tools/pyflag](tools/pyflag/readme.md)，在三个框架上跑过：verl 37 个开关／8 个失效（2 个已人工确认为真）、SkyRL 39／7（`use_liger_kernel`、`use_remove_padding` 确认为真）、OpenRLHF 5／0。
- **优先级 / 负责人 / 状态**：方法已留在 `tools/`（复用价值明确）；**SkyRL 的两条尚未做完整取证**，作为后续候选。
- **下一步与等待条件**：SkyRL 两条若要对外，需补齐取证材料与影响面；`use_remove_padding` 是 `mode=always`（默认 True 且无人传），属"关不掉"而非"开不了"，结论表述要区分。
- **停止条件**：无新框架待扫时暂停；不为凑数反复扫同一批仓库。
- **成果入口**：[工具说明](tools/pyflag/readme.md)、[扫描报告](tools/pyflag/)。

## TM-008 — verl V1 采样等待的 `poll_interval` 不可配置（C 类：优化 PR）

- **问题与价值**：`ReplayBuffer.__init__(poll_interval=2.0)` 是采样等待的轮询间隔（`_wait_for_next_poll` → `time.sleep`），但生产构造路径 `_build_replay_buffer` 不传该值、`trainer.v1.sampler` 没有这个键、文档参数表也未列，内置 buffer 只能取默认 2.0s；用户只能写自定义 sampler 子类才能改。默认值把采样等待按 2.0s 量子化：§13.44 已证明这是 2 卡 `timing_s/gen` 二态的唯一来源。与 TM-005／006 同族（配置不接线），区别是默认值本身有可测代价，且补丁可做成零行为变化（默认不变、只加可达性）。
- **优先级 / 负责人 / 状态**：C 类候选，**M5 本地实测与补丁已完成**；对外（issue／PR）待用户逐项授权。
- **已有证据**（2026-11-02，4090-1，verl `24f25b03`＋Qwen2.5-1.5B＋gsm8k，15 步，steps 3–14）：2 卡默认整步 21.785s／gen 3.165s（6/12 步 >3s）；补丁＋默认值 2.0 的对照为 21.686s（补丁惰性）；`poll_interval=0.5` → 21.033s（−3.0%）、gen 2.370s、0/12 慢步；`0.1` → 21.001s（−3.2%）、gen 2.199s。4 卡（2+2）默认 14.216s／gen 2.161s（恒定多等约 0.4s）→ 0.1 后 13.934s（−2.0%）／gen 1.763s。**0.5 不是通用解**：4 卡生成跨度约 1.6s，0.5 仍在 2.0 边界取整，必须 0.1–0.2。轮询开销没有可测惩罚（p050 vs p010 每步差约 16 次 `kv_list`，整步差 −0.03±0.26s）。数值差异不超过同行为两次运行的噪声（pbase vs p200 已差 reward 0.1161 vs 0.1146）。见 [experiments/verl-v1-poll-quantum](experiments/verl-v1-poll-quantum/README.md)。
- **本地补丁**：[poll-interval-config.patch](experiments/verl-v1-poll-quantum/poll-interval-config.patch)（3 文件 +8/−1：配置键、传给 buffer、`_step_had_idle` 阈值与轮询间隔解耦）。默认 2.0 不变 ⇒ 本补丁不改变现有行为；调小为显式 opt-in。
- **2026-11-02 时效性复核（M1，针对 `origin/main` `4b26c788`，本地检出落后 361 个提交）**：前提**仍然成立**。`verl/trainer/ppo/v1/replay_buffer.py` 的 `__init__` 仍有 `poll_interval: float = 2.0`，`_wait_for_next_poll` 仍直接 `time.sleep(self.poll_interval)`；`trainer_base.py::_build_replay_buffer` 的 `replay_buffer_kwargs` 依旧不含该键；`ppo_trainer.yaml` 的 `trainer.v1.sampler` 段（约 292–312 行）仍无 `poll_interval`。**注意补丁需重放（M5 23:2x 已证伪，见下）**：上游 `v1/` 已重构出 `trainer_sync.py`，patch 的其余两个 hunk 上下文需按新文件重做后再应用。
- **边界实验已补（2026-11-02 第五十二批，§13.48）**：DAPO 组过滤下十跑。**inflight=2**：原树 2.0／patched 2.0／0.5／0.1 ＋ 2.0、0.1 各一次重复——gen 8.474→7.289s（−14.0%）、整步 26.534→25.246s（−4.9%），其余相位 ±0.1%；**但不是纯省等待**：每步多生成 13%（117.4→132.8，surplus +39%）。**inflight=1（DAPO 默认）**：每臂两跑——池化 gen −13.0%、整步 −3.4%、生成组 +5.3%（干净窗口那对为 −10.6%／−3.0%／+7.0%，另一对与并行会话 4 卡作业重叠只作一致性检查）。**生成量代价随 in-flight 余量缩小（13%→5%）**，且默认值本身更省（有效率 31% vs 27%）。补丁形态不用改，PR 风险段要写这一条。见 [experiments/verl-v1-poll-quantum-dapo](experiments/verl-v1-poll-quantum-dapo/README.md)。
- **4 卡 A/B 已闭环（2026-11-02 第五十三批，§13.49）**：独占窗口内三臂各两跑（6/6 exit 0）——整步 18.588→17.846s（0.1，−4.0%）／17.786s（0.5，−4.3%），gen −12.0%／−14.3%，其余相位 ±0–2.4%。回归拆分：**每步平均等待 ≈ 0.59×poll_interval**（0.1 臂 −1.14s、0.5 臂 −0.86s；同配置负对照 −0.155s，CI 跨 0），2 卡同模型斜率 0.98 ⇒ **等待量不能当常量迁移**，"4 卡应得 −10%"与"固定批 −2.0%"两条先验都不对。0.5 与 0.1 在 4 卡 DAPO 上墙钟等价（差 0.06s）。4 卡多生成代价 +6.4%（1.4 SE）定不下来，PR 风险段仍用 2 卡的 +13.1%。订正首轮 18.518s 的"并发上界"标注（干净窗口 18.588±0.14s，重叠成本 ≤0.15s）。见 [4 卡 A/B](experiments/verl-v1-poll-quantum-dapo-g4/README.md)。
- **下一步与等待条件**：对外前需用户决定两件事——① 只做配置可达性＋文档（默认 2.0），还是同时把默认改到 0.1–0.2；② 算力侧已测完 2 卡 inflight=1／2、4 卡 inflight=2 的 2.0／0.5／0.1，**仍未测** async trainer 的 `wait_for_sampleable` 分支、stale 逐出触发的路径、`max_inflight_gen_batches>2`、4 卡 `inflight=1`；4 卡生成量代价要判定需每臂 ≥4 跑。**补丁不需要重放**——对 main `3fdf2b54` 与 `4b26c788` 两棵树的三目标文件 `patch -p1 --dry-run --fuzz=0` 均 3/3 通过；`trainer_sync.py`／`trainer_colocate_async.py` 只是 `PPOTrainer` 薄子类、不覆写 `_build_replay_buffer`。（M1 的"上下文已变"来自其旧检出，已订正。）对外动作全部未做。
- **停止条件**：上游已支持该键或改了默认（脚本/代码核对会发现），或用户否决对外动作。

## TM-009 — rLLM 工具调用的 `use_async=None` 自动探测失效（B 类：主动发掘）

- **问题与价值**：`rllm/tools/tool_base.py::Tool.__call__` 的文档与 docstring 都声称"`use_async` 为 `None` 时自动选择 async/sync 实现"，但①签名默认值是 `False` 而非 `None`，自动探测分支默认不可达；②`has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward` 对**任何子类恒为 True**（`Tool.async_forward` 本身是普通 `async def`，`X is X` 也恒真），于是纯同步工具在 `use_async=None`／`True` 时返回**未被 await 的协程**而非结果，同一个 `if` 里的 `NotImplementedError` 守卫成为不可达代码。仓库内 11 个工具只实现 `forward`、仅 2 个实现 `async_forward`，受影响面正是大多数。
- **优先级 / 负责人 / 状态**：B 类候选（新框架扩池，2026-11-02 按「范围不限于已熟悉框架」选取）；M1 取证、查重、可达性复核、候选补丁四项完成，**等待用户决定是否对外**。
- **已有证据**：main `4e828c8` 静态取证 + 用**真实 `Tool` 类**的最小复现（非手写替身），Python 3.12.13，`exit 0`＝缺陷存在；负向对照（改用 `type(self).async_forward is not Tool.async_forward`）三项行为均转为正确。[取证材料](open-source/rllm/tool-use-async-autodetect.md)、[复现脚本](experiments/rllm-tool-use-async-probe/readme.md)。
- **影响面边界（决定价值评估为 6 而非更高）**：`grep -rn "use_async=" rllm/` 的命中**全部在各文件 `__main__` 演示块**，生产路径不传该参数，故现有训练/评测不受影响；`tests/tools/test_tools.py:202` 用了 `use_async=True` 但只 `print` 不断言，缺陷因此长期无人发现。**不得描述成"训练结果有问题"**。
- **可达性复核（2026-11-02，[prod_reachability.py](experiments/rllm-tool-use-async-probe/prod_reachability.py)，exit 0）**：把上面的 grep 印象升级为结构化核对——AST 扫描全包，`use_async=` 出现在 `__main__` 块之外 **0 处**；块内 4 处取值全是 `True`；生产构造点 `code_reward.py` 调 `codetool(code=..., timeout=30)` 走默认同步、正常返回。**并查清 rLLM 有两套互不相干的工具栈**：`rllm/tools/` 的 `Tool` 子类走 `__call__`，`rllm/harnesses/tools/` 的 `BashTool` 等是普通类、协议 `.schema` + `.execute(sandbox, ...)`，agent 主流路径结构上绕开本缺陷。
- **候选补丁已就绪（2026-11-02）**：[patches/use-async-autodetect](open-source/rllm/patches/use-async-autodetect/readme.md)。`has_async` 改一行 + 三条常量断言测试；修复后 3 passed、未修复 2 failed（负向对照）。**新事实：上游 `test_python_tool` 对纯同步的 `PythonInterpreter` 传 `use_async=True`，过去靠缺陷吞掉才通过，修好后按设计抛 `NotImplementedError` 而变红**——这正是"守卫被测试掩盖"的证据，但怎么改那处测试涉及上游测试意图，留给维护者。
- **下一步与等待条件**：对外前需用户决定两件事——① 是否提一条**独立小 issue**（附复现脚本，成本低、指向明确）；② 若提 PR，是否同时改签名默认为 `None`（对下游是行为变化，需在读 PR 人的角度说明）。~~另需先补一步未做的核查：检索上游 issue/PR 是否已有人认领该缺陷~~ → **已完成（2026-11-02，独立复核）**：`repo:rllm-org/rllm async_forward` **0 命中**、`use_async` 仅 1 条无关 PR（#460 Daytona），**上游无人认领**。复核同时补了第三点危害：基类 `async_forward` 体内是 `return self.forward(...)`，故**只实现同步的工具被误判为异步时不报错，而是静默返回 coroutine**（未 await 仅 RuntimeWarning）——比原记录的"分支不可达"更实际。见 [独立复核](open-source/rllm/tm-009-independent-review.md)。对外动作全部未做。
- **停止条件**：上游修复 `has_async` 判定（复现脚本会以非零退出提示结论失效）、配置删除，或用户否决对外动作。
- **成果入口**：[取证](open-source/rllm/tool-use-async-autodetect.md)、[复现](experiments/rllm-tool-use-async-probe/readme.md)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)、[独立复核](open-source/rllm/tm-009-independent-review.md)、[pyflag 扫描](tools/pyflag/scan-rllm-4e828c8.json)。

## 已闭环与保留方向

- **TM-004 — AReaL prompt_mean：终止修复候选。** M5 单一负责人；2026-10-20 实测跨 M 与 token_mean 一致，不构成 defect，修复撤销，不提 issue/PR。M1 不继续修改脚本或演化并行方案，用户明确的只读 review 除外。证据及分工：[handoff](open-source/areal/handoff-m5.md)、[tracker](open-source/areal/tracker.md)。AReaL 的长期学习偏好仍保留，新的问题应独立登记。

## 新选题要求

写清问题、已有证据、探索价值、下一次验证与停止条件再投入。没有新信息增量时保持等待或终止；不得以文件数、提交数或自评分数替代价值。

每条新选题需标明：**渠道类型（A/B/C/D）**、**目标仓库**、**问题来源**（issue 编号／源码路径／实测现象）、**可量化目标**（仅 C 类）、**停止条件**。缺任一项不进入队列。
