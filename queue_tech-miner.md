更新时间：2026-11-04（M1 **rLLM A 类扫描 + 倒计时奖励复核**：按 A 类渠道扫 rLLM 开放 issue，筛出的三条（#916／#945／#929）**全部已有对应 PR**，无可捡活、不重复认领；转而独立复核 #945／PR #946——四类漏洞在真实源码上复现（四例全 1.0），补丁在 **22639** 条合法答案上 **0 回归**、负向对照报 **159** 条；"挂死"收窄为"`**5` 已 2.1 秒、`**35` 跑不完未复现"。**对外动作未做，待用户决定是否参与该 PR 讨论**。此前 M1 **语义面·长度预测上限批（第十七批）**：截断上界对组最大精确，`R` 用截断上界与全真值逐位相同，§11.5「长度预测误差」在组粒度上关掉。）

# 技术掘金任务队列

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
- **2026-11-04 追加（M1）——这不是"从没接上"，是引擎重构丢了接线，而且丢的不止一条**：PR [#4649](https://github.com/verl-project/verl/pull/4649)（2026-01-04 合入）当年**真的**把 `tiled_mlp` 接到了 `verl/workers/fsdp_workers.py`（**覆盖 actor／ref／critic 三个角色**：取值 `:913-915`／`:972-976`／`:1454-1456`，转发 `:935-936`／`:989-990`／`:1481-1482`，FSDP1 拒绝在 `:374`（actor）与 `:1459`（critic）；新引擎三条路径都汇到 `FSDPEngine._build_module()`，故三个角色一起失效）。该文件被 PR [#6067](https://github.com/verl-project/verl/pull/6067)（2026-04-20，BREAKING "deprecate workers, migrate to engines"）整体删除，接线没迁到新引擎。main `74543265` 复核：全仓 704 个 `.py` 提到 `tiled_mlp` 的只有 **6 个**（定义／配置／测试），生产调用点 `verl/workers/engine/fsdp/transformer_impl.py:322` 仍不传该参数，其余五个引擎计数全 0。⇒ **对外表述必须写"重构回归"**，不能写"配置从未生效"。**另有两点本批新发现**：① **FSDP1 守卫也一起丢了**，而新引擎**先建模型（:322）再按 strategy 分支（:421／:448）**，`dp_actor.yaml:26` 默认 `strategy: fsdp` ⇒ 只补接线不补守卫，默认配置下会先打 TiledMLP 再走 FSDP1；② 同一次重构可能还丢了别的配置项，**只查了 `tiled_mlp` 一个**；写过一个自动比对脚本，候选几乎全是改名／改层级的假阳性，**已放弃自动化、留作人工逐条核**。见 [复核](experiments/verl-tiled-mlp-regression/readme.md) 与 [脚本](experiments/verl-tiled-mlp-regression/check_wiring.py)。
- **下一步与等待条件（2026-11-04 覆盖）**：① 若要对外，最小动作是"补接线 + 补 FSDP1 拒绝"（补丁已就绪，见成果入口），描述按"重构回归"写；② **更值得先做的一步**是查同期重构还丢了哪些配置（同一删除提交 `044bbba2` 的 diff 里 grep `config.` 取值即得），这能把一条回归变成一类回归；③ 对外仍需用户逐项授权。
- **停止条件**：用户否决，或上游接上／删除该配置。
- **成果入口**：[取证](open-source/verl/evidence/tiled-mlp-unwired/readme.md)、[工具](tools/pyflag/readme.md)、[真机冒烟](experiments/verl-tiled-mlp-fsdp-smoke/readme.md)、[重构回归复核](experiments/verl-tiled-mlp-regression/readme.md)、[补丁](open-source/verl/patches/tiled-mlp-wiring/readme.md)。

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
- **async 分支已测（2026-11-03 第五十五批）**：`colocate_async`、2 卡（GPU 4,5 同 socket＝2+0）、四跑交替（2.0／0.1 各两跑，15 步/跑，4/4 exit 0）。poll 2.0 两跑 gen **1.741／1.580s**，12 个稳态步里 **9／8 步撞量子**（二态 0.22s vs 2.23–2.33s）；poll 0.1 两跑 gen **1.379／1.386s**，0/12 撞量子（0.75–1.89s）。池化整步 21.212→20.809s＝**−1.9%**（run 级 ≈2 SE；单跑 −3.0%／−0.7%），gen −16.7%（−3.4 SE）。**关键读法：2.245s 不是「浪费 2.0s」**——0.1 臂测出真实等待 0.75–1.9s，量子只是向上取整，超额约 0.1–1.3s/步；整步差里只有 gen（−0.28s）可归因，update_actor −0.21s 两臂重复间不一致、不算机制项。patched 树 `replay_buffer.py` 与 pristine **逐字节相同**（只加配置接线）。结论：补丁适用面**不是仅 sync**，对外可写「2 卡 sync −3.2%、colocate_async −1.9%（≈2 SE）」；async 没有消掉等待（仍 1.4–1.7s/步）。见 [async 实验](experiments/verl-v1-poll-quantum-async/README.md)。
- **`max_inflight_gen_batches=3` 已补（2026-11-03 第五十六批，§13.52）**：四跑干净窗口（GPU 4,5，2+0），池化 **gen 8.589→7.357s（−14.3%）、整步 26.964→25.738s（−4.5%）、生成组 113.7→133.0（+17.0%）**。分解（臂内固定效应斜率 38.8／44.2 ms/组）：固定等待 **−1.97／−2.04s**、多生成 **+0.75／+0.83s**，两笔账残差 ≈−0.01s。跨深度：生成代价 +5.3%→+13.1%→+17.0%，净收益 −0.96→−1.29→−1.23s ⇒ **depth 2 之后饱和**；depth 2→3 增量 +3.9 组 <1 SE，不外推。**方法订正**：跨臂"共斜率"在本设计下被处理效应压低（21.2 vs 臂内 38.8–44.2），以后同族分析按固定效应报。见 [DAPO 实验补测](experiments/verl-v1-poll-quantum-dapo/README.md#补测max_inflight_gen_batches3-2026-11-03m5-第五十六批)。
- **stale 逐出通道已测（2026-11-03 第五十七批，§13.53）**：threshold=2 冒烟 4 步**零逐出**（span 恰为 2，warmup=1 的自然跨度就是 1–2）；threshold=1 才触发——每步丢弃上一步整批 32 组、staleness=2、**生成组/步 64（默认 async 的两倍）**。四跑 A/B（2.0／0.1 各两跑，15 步，4/4 exit 0）：2.0 臂 gen 严格两态 **2.43/2.47 与 4.43/4.39（差 2.00/1.92s）**，撞量子 5/12、8/12 步；0.1 臂 0/12。池化 **gen 3.506→2.786s（−20.5%）、整步 22.473→21.997s（−2.1%）**；但 2.0 臂两跑整步差 0.835s、0.1 臂差 0.049s ⇒ **−0.475s 小于臂内跨跑极差，≈1 SE，只作方向性证据**。**逐出量与 poll 无关（两臂都 32/步）**；threshold=1 本身每步多花约 1.1–1.5s 墙钟，轮询只能收回约 0.5s，收不回被丢弃的批次。见 [stale 实验](experiments/verl-v1-poll-quantum-stale/README.md)。
- **补丁在 main `3fdf2b54` 上实跑冒烟（2026-11-03，M5 第二会话）**：新 main（`3fdf2b548882`）打补丁后 **3/3 步 exit 0**，解析配置 `'poll_interval': 0.1` 生效；`_build_replay_buffer` 全树仅 `trainer_base.py` 一处定义、子类不覆写。两处环境适配（TransferQueue 显式容量、本地 device-map）与补丁无关；**非计时测量**。见 [README](experiments/verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话)。
- **`max_off_policy_strategy=wait`（dropless）已补（2026-11-03，M5 第二会话，§13.53 补记）**：同配置（colocate_async、threshold=1、GPU 4,5、15 步、2.0／0.1 各两跑、4/4 exit 0），**零逐出**（`evicted_samples` 缺失、spans／staleness 恒 1），生成组/步 32（drop 为 64）。2.0 臂 gen 严格两态 **4.24／2.22（差 2.02s）**，0.1 臂 1.85–2.51。池化 **gen 3.151→2.177（−30.9%）、整步 22.323→21.079（−5.6%）**（2.0 臂跨跑差 0.408s、0.1 臂 0.029s ⇒ 强于 drop 的 ≈1 SE）。**跨策略同窗口（方向性）**：wait 在 2.0／0.1 两档都比 drop 快（22.32 vs 22.47；21.08 vs 22.00）且生成量减半 ⇒ 若关心 async 的 on-policy 严格性，wait 比"靠 poll 补偿 drop"更有效。见 [stale 实验](experiments/verl-v1-poll-quantum-stale/README.md)。**独立复核（第二会话，只读）**：[`verification-wait.md`](experiments/verl-v1-poll-quantum-stale/verification-wait.md)／`verify-wait.py` exit 0，与 `summary-wait.json` 在 1e-9 内一致；代码层确认 wait 永不逐出（`_stale_terminal_keys` 对非 drop 返回空集）且 threshold=1 是完整屏障（`>= threshold`），故 `spans==1` 属结构必然；步级 bootstrap Δstep [−1.786, −0.724]s，**两跑/臂没有 run 级 CI**。
- **`wait` + threshold=2 已补（2026-11-03，M5 第三会话）**：这是唯一有区分度的 dropless 档——t=1 是完整屏障（连本步新派发的批一起等），t≥3 永不阻塞（结构等价默认 async，仅代码推断、未实测）。同配置四跑（21:32–22:05Z，4/4 exit 0）：**零逐出**、span 均值 1.62–1.93（允许一步重叠）、staleness ≤1。池化 **gen 1.739→1.359（−21.9%）、整步 20.893→20.667（−1.1%）**，bootstrap CI **Δstep [−0.943, +0.512]、Δgen [−0.881, +0.126]，均跨 0** ⇒ **t=2 下 poll 收益与噪声不可分**（对照 t=1 的 −5.6%）。绝对步时 20.9／20.7s，比 t=1 低 1.43／0.41s、与默认 async 相当 ⇒ **若要 dropless，t=2 比 t=1 屏障实用**。归档 [stale 实验](experiments/verl-v1-poll-quantum-stale/README.md)，`summary-wait-t2.json` 由 `verify-wait.py --group t2` 生成、exit 0，md5 4/4 与服务器一致。
  **t=2 第二份独立复核（第四会话，只读）**：[`verification-wait-t2.md`](experiments/verl-v1-poll-quantum-stale/verification-wait-t2.md)／`verify-wait-t2.py` exit 0，与 `summary-wait-t2.json` 的 12 个值在 1e-9 内一致、`md5sum -c` 23/23、四跑 15/15 步。**两条补强**：① **留一检验**——去掉 step 3 后 Δgen −0.380→−0.164s、Δstep −0.226→**+0.019s**，steady state（4–14）两臂整步差≈0；同法对 t1 只把 −1.243 削到 −1.120 ⇒ t=2 的轮询**基本无效**（不只是"与噪声不可分"）。② **span 分布**——`trajectory_spans` 数的是轨迹生成时跨越的权重版本数：span=2 的窗口步 2.0 臂 **18/24**、0.1 臂 **24/24**，2.0 臂 span 均值 1.62／staleness 0.38 对 0.1 臂 1.92／0.08（最坏 lag 都恒 1）⇒ poll 越小跨更新越多，两档各有不同的 off-policy 面，不能简单说哪档更 on-policy。
- **下一步与等待条件**：对外前需用户决定两件事——① 只做配置可达性＋文档（默认 2.0），还是同时把默认改到 0.1–0.2；② 算力侧已测完 2 卡 sync inflight=1／2／3、4 卡 sync inflight=2、**2 卡 colocate_async**、**2 卡 colocate_async + threshold=1（stale，drop／wait 两臂）**、**separate_async**，**仍未测**只剩 4 卡 `inflight=1`（需 ≥4 卡空闲窗口）与 `wait` 的 t≥3 档（结构上等同默认 async，未单独实测）；4 卡生成量代价要判定需每臂 ≥4 跑。**补丁不需要重放**——对 main `3fdf2b54` 与 `4b26c788` 两棵树的三目标文件 `patch -p1 --dry-run --fuzz=0` 均 3/3 通过，并已在 `3fdf2b54` 上实跑 3 步冒烟（上条）；`trainer_sync.py`／`trainer_colocate_async.py` 只是 `PPOTrainer` 薄子类、不覆写 `_build_replay_buffer`。（M1 的"上下文已变"来自其旧检出，已订正。）对外动作全部未做。
  **`separate_async` 已测（2026-11-03 第五十八批，§13.54）**：上游自带 `tests/special_e2e/run_v1_separate_async.sh`（此前"examples 下无脚本"只看了 examples），其校验允许 **2 train + 1 rollout**（训练卡偶数、rollout 卡被 TP 整除，取 TP=1）。四跑（GPU 4,5 + 1、8 步、2.0／0.1 各两跑、**4/4 exit 0**）：**稳态 `timing_s/gen` 0.031–0.038s，两档无差别**；整步 8.211→8.201s＝**+0.010s（t=0.12）**⇒ **poll 在本模式下不在关键路径上**。机制：训练步 8.2s（old_log_prob 3.1＋update_actor 3.0＋update_weights 1.85），rollout 池单卡生成 16 条/步远快于此，队列始终非空，`ReplayBufferAsync.sample()` 不进入睡眠循环；step 1 的 4–6s 等待是 warmup，与 poll 无关（0.1 臂两跑 5.73 vs 0.03s）。`spans`／`staleness` 恒 1.00／1.00。**本机两条前置（非上游缺陷）**：① `nccl_checkpoint_engine` 需 cupy，装到 `/tmp/cupy-pkgs`（`/home` 只剩 8G）；② **不能设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`**——`ray.util.collective` 的 barrier 按本进程可见卡数建通信器，NOSET 下每 actor 见 3 卡 ⇒ 每 rank 都占 0/1/2，报 `Multiple Ranks are using the same GPU`；去掉 NOSET 后训练侧 NCCL 撞 CUDA 217（`ncclShmImportShareableBuffer`），用 `NCCL_SHM_DISABLE=1` 绕开（训练集合通信退化为 socket，**绝对步时不可与其他批次比**）。**边界**：本配置是 rollout-ahead 区间（1 rollout 卡 vs 2 训练卡 + 0.5B），rollout 成为瓶颈时会空队列、量子可能回到关键路径——该区间未测；rewards 池化为 0（0.5B 做不出 gsm8k）、训练侧偏轻。归档 [separate 实验](experiments/verl-v1-poll-quantum-separate/README.md)，`verify.py` 34 项全 PASS、md5 5/5。**至此 TM-008 的模式覆盖完整**：收益是"训练步是否阻塞在 `sample()`"的函数（固定批 −3.2%、DAPO −3.4／−4.9%、4 卡 −4.0%、colocate −1.9%、stale drop −2.1%、wait t=1 −5.6%、wait t=2 ≈0、separate_async ≈0）。
- **`separate_async` 前置（2026-11-03 第四会话只读评估，第五十八批实测通过）**：`PPOTrainerSeparateAsync.__init__` 硬性要求 ① `actor_rollout_ref.rollout.nnodes > 0` 且 `n_gpus_per_node > 0`（独立 rollout 池）；② `rollout.checkpoint_engine.backend != "naive"`（nccl/nixl/mooncake）；③ `train_batch_size == parameter_sync_step × ppo_mini_batch_size`（默认 `parameter_sync_step=4`，沿用 batch 32 需改 1 或调 mini_batch）；④ 开 reward model 时须 `enable_resource_pool=True`（本实验不用 RM）。**2026-11-03 已用 2 train + 1 rollout 跑通**，具体环境适配与结果见上条。
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

## TM-010 — rLLM Fireworks 后端默认开着的 R3 路由回放拿不到训练帧（B 类：主动发掘）

- **问题与价值**：`rllm/trainer/config/rllm/backend/fireworks.yaml:78` 自 PR #934（2026-10-09 合入）起默认 `router_replay: R3`，但逐 token 路由矩阵在 Tinker/Fireworks 训练路径上**装不进训练帧**——`rllm/trainer/tinker/transform.py:85-87` 用 `model_copy(update={"routing_matrices": ...})` 注入，而 `tinker.ModelInput` 的声明字段只有 `['chunks']`、`from_ints()` 不接受该参数、模型 `frozen=True`+`extra='forbid'`，赋值被 pydantic 拒绝。默认配置下开关在记录数据、训练侧结构上拿不到。同路径另有两处不一致：Python 侧 `AlgorithmConfig.router_replay` 默认 `"disabled"` 与 yaml 的 `R3` 不同源；verl 0.8.0 里 `actor.router_replay.mode='R3'` 语义是"记录到 record_file"，rLLM 又另开 `rollout.enable_rollout_routing_replay=True`，两条通道并存。
- **优先级 / 负责人 / 状态**：B 类候选（扩池：rLLM 的托管训练后端）；M1 本地取证完成，**等待用户决定是否对外**。
- **已有证据**（2026-11-03，main `4e828c8`）：[取证](open-source/rllm/fireworks-r3-routing-matrices.md)、[复现](experiments/rllm-fireworks-r3-replay/repro.py)（两阶段，真实 rLLM 类 + 真实 FireworksEngine 打桩 sampler，exit 0＝缺陷在）、[承载检查](experiments/rllm-fireworks-r3-replay/carrier_check.py)（tinker 0.22.2／0.23.0／0.36.0 三版本一致：无字段、`from_ints` 无该参数、赋值抛 `ValidationError`）。**已排除两条旁支**：①"多步累积时矩阵布局错位"实测两个形状都对齐；② verl 蒸馏路径的 `OpenAIEngine.compute_logprobs` 实为已定义。上游 cookbook（`bba676c`）走 `ModelInput.from_ints(..., routing_matrices=...)` 且先做 `build_r3_routing_matrices` 对齐——但它依赖的 tinker 在所有可用版本里也没有该参数，**这不是 rLLM 单方面能改的事**。
- **下一步与等待条件**：对外前需用户决定 ① 是否提 issue（先问"Fireworks 服务端收到未声明属性是忽略还是报错"）；② 若提，是否连"默认改回 disabled"一起。**未验证项必须先说清**：服务端行为（需真实 Fireworks 账号）、是否已影响线上训练——**不得主张训练出错或性能损失**。
- **停止条件**：上游改默认值、补上承载字段或改用 cookbook 路线（复现脚本会以非零退出提示结论失效），或用户否决对外动作。
- **成果入口**：[取证](open-source/rllm/fireworks-r3-routing-matrices.md)、[复现](experiments/rllm-fireworks-r3-replay/readme.md)、[run-info](experiments/rllm-fireworks-r3-replay/run-info.json)。

## TM-011 — 长度预测／分桶对"组粒度代价"的作用：需要建模型才能回答（B 类：主动发掘，内部研究项）

- **问题与价值**：§11.5 把语义面剩余缺口写成"真实 L3 调度机制（长度预测／分桶／迁移）之上的 `R` 读数"。第三批量了**排队顺序**这一维，第四批换到真实负载，第三十九批试了"逐条补位"的上限——但**分桶本身没被建模**。若分桶能把组成本 `max(g)` 压下去，它对组粒度代价的作用可能比排队顺序大得多。
- **优先级 / 负责人 / 状态**：候选（TS-001 语义面的后续）；**未立项**，M1 第三十九批只做了备料。
- **已有证据**（2026-11-03）：真实 1.5B cap256 上 FIFO `R` 4 批 2.4%／16 批 10.8%，"两臂各自按长度排"（`lof_flat`，**不可实现**）12.9%／13.8%；**真实组最大 cv 仅 0.07**（组内极差 85 token），四个组级策略在同一负载上读数可完全相同 ⇒ **组级排序在组成本齐平时无事可排**。见 [CONTEXT 第三十九批](CONTEXT.md)。
- **2026-11-03 追加（第四十批，M1）——"只排序"这一维已经有否定读数，停止条件要改写**：把「长度优先」的 `R` 效应按配对口径重读（FIFO 吃到达序、LOF 吃按组最大降序），真实 rollout 长度上**多数配置让 `R` 变差**：1.5B cap1024 四档 −3.34／−6.33／−2.93／−1.44pp（积压 2/4/8/16），0.5B cap1024 4 批 −8.14pp，1.5B cap256 4 批 −1.97pp；只有低离散＋深积压（cap256 16 批 +1.23pp、0.5B cap256 16 批 +0.48pp）转正，且不到 1.3pp。机制是**哪一臂被排得更整齐**（`R1−R0=(T0/G0)−(T1/G1)` 的代数恒等），同一张表里组臂增益与轨迹臂增益的相对大小直接决定符号。详见 [bucketing-probe.md](experiments/rollout-group-split-tradeoff/bucketing-probe.md)。⇒ **原停止条件"作用 <1pp 就关闭"对本维已触发但方向为负，不适用"记否定并关闭"这一种读法**——它不是"分桶没用"，而是"**只排序不改批组成**没用，甚至有害"。**⚠ 本条里的 cap256 各档数值（−1.97／+0.54／+1.23／+0.48pp）已由下一条（第四十一批）订正为"符号不稳、不支持方向性结论"**，引用时以下一条为准。
- **下一步与等待条件**：要立项需先定义两件事——① 分桶粒度（按预测长度分 K 桶，K 与桶宽都是参数）；② 预测误差如何映射到组序（真实长度预测误差分布需要有来源或显式假设）。**不得用 `lof_flat` 当分桶代理**：它是"两臂各自最优"，不对应任何可实现策略。**新停止条件**：若把"改批组成"也建模后，`R` 的相对改善在所有可及负载上仍 <1pp，记否定结论并关闭；若只做到"排序"这一维（当前状态），结论固定为"单独排序有害"，不再重复测。
- **2026-11-03 追加（第四十一批，M1）——均值判读订正，且立项前必须先改排程器**：给探针补了两条判读量，16 格读数因此分档。① **FIFO 基线自身依赖"到达顺序"**：同批组只换传入顺序，`R` 就动 **1.4–5.5pp**（4 批档最大 5.52pp），比上一版报的多数"效应"还宽；② **符号稳定性**——只有 **cap1024 且积压 ≥4 批**的六格是 19–20/20 同号（方向为"长度优先让 `R` 变大"），**cap256（RL 实际配置）各档符号在种子间翻转**（3/13、13/7、10/8、5/15、10/10），2 批档 sd 是均值的两倍。⇒ 上一版正文的 "−1.97／+0.54／+1.23pp" 是把均值当效应，**本批订正为"这些格子不支持方向性结论"**；主导结论"只排序有害"**只在 cap1024 ≥4 批成立**。
  **立项可行性（本批核过，三条硬约束）**：① `cap = n_groups * n_per` 恒等于单批轨迹总数、② 组到达时刻 `arrival = b * batch_span`（同步到达）、③ 单批内组臂一次放下全部组（乱序对 `group_makespan` 零影响，实测 10 次全同）。⇒ **"哪些组进同一批"在本排程器里不是可改的变量**，分桶改不动批组成；且"到达顺序"没有独立于抽样样本的物理含义，是探针残留噪声的根源。**要立项必须先改排程器**（引入独立到达流＋可分离的批窗口），不是加一层分桶逻辑。详见 [bucketing-probe.md](experiments/rollout-group-split-tradeoff/bucketing-probe.md) 第三版。
- **2026-11-03 追加（第四十二批，M1）——同一判据补到真实负载上，cap256 的"反号"作废**：分桶探针补的两条判据（同号比例＋基线顺序带宽）此前只用在构造档；真实 rollout 负载这一侧（§13.30 第四批）仍只报均值。补上之后：① 真实档 FIFO **组集合固定、只换顺序**，`R` 就动 **0.7–11.7pp**（4 批档中位 2.8–4.0pp）；② 第四批报的 +1.99／−0.95／+0.44／−0.39pp **全部落在各自带宽内或带边**，0.5B cap256 低积压有 **19/30 个种子两臂读数完全相等**（无分辨率）；③ 把 FIFO 的到达序换成随机序，配对差几乎不动（逐种子同号 23–30/30）⇒ **该钉的不是"到达序选错"，而是这类负载上配对差本身就有这么厚一层随机性**。**可引用的只剩 cap1024（cv 0.30–0.36）六格：28–30/30 同号，方向为 `R` 变大。** ⇒ TM-011 的"只排序有害"**只在组内不齐平（cv≳0.30）的负载上成立**；RL 实际配置（cap256、cv≈0.15）上**方向未定，不是已知相反**。详见 §13.30 第五批与 [readme 第五批](experiments/rollout-group-split-tradeoff/readme.md)。
- **2026-11-04 追加（第十批，M1）——同号占比的分母拆开，cap256 的判词要改**：`same_sign_fraction` 把"两臂读数完全相等"的种子留在分母里，读数恒 ≤ 只看表态种子的口径，于是"方向一致、分辨率薄"会被读成"方向混杂"。四档 `sign_regime`（`stable`／`mixed`／`consistent_but_thin`／`no_resolution`）已写进两份探针归档：真实档 cap256 那六格是 **2 格 `consistent_but_thin`（方向一致、分辨率薄）、2 格 `mixed`（真混杂）、2 格 `stable`（方向稳但效应量在带宽内）**，另 1 格 `no_resolution`；**没有一格能主张方向**（结论不变），但**不能把 `consistent_but_thin`／`stable` 那四格当"符号不稳"的证据**。另记：`qwen15b_cap256` 积压 2 的 `fifo_band` 全为 0（该格 `R` 恒为 0），**噪声为零的带宽不构成门槛**。见 [paired-diff readme](tools/paired-diff/readme.md) 与 §13.30 第五批订正框。
- **2026-11-04 追加（门槛校准批，M1）——「效应超带宽」这句话的准入门槛，现在有误判率了**：此前只写「带宽是判读门槛不是显著性检验」。量出来两边完全不对称：**`sign_regime` 假阳 0**（真同向／真 `thin` 从不判 `mixed`）、假阴 1.1% **就是二项分布本身**（`stable` 要表态种子 ≥80% 同号，20 硬币双侧 ≥16 同号 = 1.18%）⇒ 想让 `stable`/`mixed` 更可信**只能加种子**，挪门槛无效（0.8→0.55 时假阴跳到 82.2%）；**`band_floor` 的 `no_reading` 要求 30 个种子全零**，非零种子比例只要 0.05 就只剩 24% 判得出来，而 `usable` 松到过半非零即算。⇒ **对 TM-011 的直接含义**：cap256 截断档上取「方向是否稳定」这类结论时，`stable` 档名可用（假阳 0），但**报告带宽必须带 `n_r_identically_zero`**——`qwen15b_cap256` 积压 4 判 `usable` 却有 26/30 个种子非零、中位带宽 2.83pp，与 0.5B 积压 8 的 `usable`（30/30 非零）不是同一件事。不改任何已发布读数；详见 [threshold-calibration.json](experiments/rollout-c-class-ledger/threshold-calibration.json)。
- **2026-11-04 追加（第十一批，M1）——带宽为 0 不等于「基线很稳」，三档准入已入档**：第五批的「噪声为零的带宽不构成门槛」此前只是一句提醒、没变成判据，于是 `fifo_band = 0` 的格子在归档里与「基线换序完全不动」长得一样。新增 `band_floor` 三档（`no_reading`／`degenerate`／`usable`），16 格实测 **13／1／2**：`no_reading` 是两个模型 cap256 积压 2，`degenerate` 是 0.5B cap256 积压 4（22/30 种子带宽为 0）。**成因是截断贴顶**——cap256 组最大 86.0%／94.2% 恰等于 256 ⇒ 组间齐平、任何顺序无事可排，`R ≡ 0`；这与第三十九批「组最大 cv 仅 0.07、组级排队无事可排」是同一几何事实的另一后果：**截断既压扁了可排的顺序维，也压掉了基线带宽本身**。⇒ 对 TM-011 的直接含义：**在 cap256 这类截断负载上，任何「组级排序」实验连噪声都测不到**，不只是「效应小」；若将来立项，负载档必须选不截断或放宽 `n_groups`。**不改任何已发布读数**，cap256 六格仍无一格可主张方向。见 [实验 readme 第十一批](experiments/rollout-group-split-tradeoff/readme.md)。
- **2026-11-04 追加（判据独立性批，M1）——「档名」与「效应超带宽」是两件事，不能互相顶替**：此前几批把两者当同一条链上的两步（`sign_regime` 判方向 → `exceeds_band` 判量级），默认档名稳就意味着方向有内容。逐种子重算 12 格后这个默认不成立：
  - **`stable` 不代表效应超带宽**：3 格判 `stable` 而 \|效应\| < 中位带宽（0.5B cap256 积压 16 的 **0.30**、1.5B cap256 积压 4 的 **0.70**、积压 16 的 **0.76**）。
  - **`mixed` 不代表"方向混杂且效应不小"**：**0.5B cap256 积压 8** 判 `mixed`，逐种子 **0/30 个种的 |d| 超过自己的带宽**、归档 \|效应\| 只有 0.01 倍中位带宽 ⇒ **纯噪声格**，不得当负项证据。
  - **校准的适用条件**：第十批「假阳为 0」是在"效应 ≈6 倍噪声"下取的；扫 `ratio` 后拐点在 **0.25–0.40 倍带宽**（0.20／0.25／0.30／0.40／0.50 处 `stable` 为 0.09／0.15／0.37／0.82／0.99），解析预测 ≈0.30；真实格中位比值 **0.76（归档汇总量）／1.17（逐种子）正落在拐点附近** ⇒ 效应不到带宽 1/3 时，`mixed` 的多数出现是**分辨力不足**而非方向混杂。
  - **⇒ 报告纪律（替代此前"先看 `n_identical` → 同号比例 → 是否超带宽"的三步链）**：**方向看 `sign_regime`、量级看 `effect_exceeds_band`，两句分开写**；任一不过只能写"方向已定、量级未定"或"方向未定"。若只有"符号在种子间一致"就断言"按方向做能省多少"，在 0.5B cap256 积压 8 这种格上会凭空造出一个 2pp 级的负项。
- **2026-11-04 追加（判据单位数批／第十五批，M1）——报 `stable` 必须带「表态种子数」，误判率随它移动两个数量级**：门槛校准的六种场景全是 **20 个种子**，而真实 16 格里表态种子只有 **3–27** 个。同一个 `min_same_active=0.8`，掷硬币输入被判 `stable` 的概率：**3 个 0.250／5 个 0.375／19 个 0.0044／20 个 0.0118／27 个 0.0015／30 个 0.0014**，且带锯齿（19 要 16/19=0.842、18 只要 15/18=0.833）。真实 9 个 `stable` 格里最脆是 **0.5B cap256 积压 16（19 表态／18 同号，余量 2 个种子）**。⇒ 引用 `stable` 时**同时写表态种子数**（与"同时报同号比例、基线带宽、是否为零"并列成第五条）。判据已进共享工具 `paired_diff.falsy_rate(n_truth, passes=…, two_sided=…)`。**不改任何档名、任何效应量**。
- **2026-11-04 追加（`band_floor` 单位数批／第十六批，M1）——`degenerate` 的门槛也被用在多少个种子上，且**不单调**，对 TM-011 的含义是"零带宽计数不能省"**：`degenerate` 判的是"过半种子带宽为 0"，分母是**全部种子**（真实档 30），而零带宽种子是 **0–30** 个。掷硬币输入下判 `degenerate` 的概率：3 个种子 **100%**、4 个 62.5%、5 个 100%、10 个 75.4%、20 个 82.4%、30 个 **85.6%** ⇒ 这一档**偏向 `degenerate` 一侧**，单位数越小越挑不出。真实 16 格里贴门槛的只有 **0.5B cap256 积压 4（22/30 `degenerate`，再少 6 个零就掉出）**；**1.5B cap256 积压 4 是 `usable` 带 4/30 个零**（离门槛 12 个）；3 个种子那档**在本归档里没有踩中**。⇒ 与第十五批同一条纪律：**引用 `band_floor` 必须同报 `n_band_zero`**；cap256 各档"零带宽占比很小"这类读法在 4/30 与 0/30 上强度差一个量级，`usable` 两个字不足以承担。**本批不改任何档名、任何效应量**。
- **2026-11-04 追加（第十七批，M1）——「长度预测」这一项在组粒度上已经关掉，且它给出的是一条恒等式而不是读数**：截断本身就是一条零成本预测器，它对**组最大**（组臂的物理耗时）是**精确**的：`max(撞顶记 c、其余记真值) == max(g)`，四条负载 × 480 组 **480/480**，**与撞顶比例无关**（cap1024 只 0.03%–0.65% 撞顶，同样精确）。⇒ `R` 在「截断上界」与「全真值」之间**逐位相同**（30/30 种子）。**对 TM-011 的直接含义有三条**：① **"加个长度预测器就能改善组粒度代价"这条路在 `R` 口径上已经走到头**——不是因为效果小，而是因为 `R` 里唯一依赖预测的量（组臂的排队序）已经被免费预测器精确给出；② **但"预测没空间"必须与"负载有区分度"分开**：打坏预测后 `R` 在 1.5B cap256 积压 16 动 21/30 个种子、两条 cap1024 档动 30/30，而 **0.5B cap256 只动 5/30**（该档 61% 轨迹撞顶、组序几乎只由「撞没撞顶」决定）⇒ **若将来要立项做排序/分桶，负载档必须选组最大有区分度的**，cap256 这种齐平负载上任何组级策略都测不出区别（与第十一批 `band_floor` 的 `no_reading` 是同一个几何事实的第三面）；③ **逐条侧的预测误差仍是未测项**——撞顶样本的真实生成长度语料里没有记录（原始数据把 `response_tokens` 就记成 `c`），本批只能明写"测不出来"，不得用「误差为 0」代替。**同日复核补记**：`>=` 改 `>` 时 12 条测试假绿 ⇒ 恒等式的失效边界在**记录标准**、不在判据写法，已由 `identity_self_check()` 两侧钉住（12 → 15 passed）。
- **新停止条件（第四十二批汇总，覆盖前几条）**：① **不要再用 cap256 各档的均值下方向性结论**，也不要引用"−1.97／+0.54／+1.23pp"或"−0.95／−0.39pp"当效应；② **引用任何配对差前必须同时报同号比例（含表态占比与表态种子内的同号占比）与基线顺序带宽**（真实档带宽 0.7–11.7pp，比多数效应还宽），并**先看带宽是否为 0**；③ 若要立项，前置是**排程器改造**（独立到达流＋批窗口可分离＋容量与批解绑），未改造前不接"分桶模型"分支；④ 改造后若 `R` 的相对改善仍在所有可及负载 <1pp，记否定结论并关闭；⑤ 若需要真实系统的长度预测器数据而拿不到，登记等待条件而非编造。
- **停止条件（旧版，保留留痕）**：~~若建模后发现分桶对 `R` 的作用在所有可及负载上都小于 1pp，记否定结论并关闭。~~ **已被上面第四十二批汇总版覆盖。**
- **成果入口**：[实验](experiments/rollout-group-split-tradeoff/readme.md)、[排程器](experiments/rollout-group-split-tradeoff/group_split_tradeoff.py)、[run-info](experiments/rollout-group-split-tradeoff/run-info.json)。

## TM-012 — rLLM 开放 issue 这一批已被全部认领；倒计时奖励复核完成（第三方复核，等待用户决定是否参与）

- **问题与价值**：按 A 类渠道（修已知 bug）扫 rLLM 的开放 issue，按"无人认领 + 可本机验证 + 影响奖励正确性"筛出 #916（`reinforce_plus_plus_baseline` 在 `rollout.n=1` 时优势恒为零）、#945（倒计时奖励被 `1.5*2` 一类答案白拿满分，指数链可拖住判分）、#929（空 `\boxed{}` 拿满分）。**三条全部已有对应 PR**（#917 / #946 / #930＋#948），其中 #916／#945 的 PR 在 2026-11-02 提交，维护者至 2026-11-04 未回应。⇒ **这一批没有可捡的 A 类活**，不重复认领。
- **优先级 / 负责人 / 状态**：第三方复核；M1 本地完成，**等待用户决定是否参与 PR #946 的讨论**。
- **已有证据（2026-11-04，main `4e828c81`，上游最后提交 2026-10-09）**：issue #945 的技术描述**属实**——四类答案在真实 `compute_score` 上都得 1.0（`1.5 * 2`／`.5 * 8`／`3 ** 2 * 5 - 7`／`7 // 2`），合法对照也 1.0。**"挂死"要收窄**：`44**19**2` 瞬时、`44**19**5` **2.1 秒**，issue 举的 `44**19**35` 值约 10^45 位、**跑不完故未复现**，不能说成"必然长期挂死"。**PR #946 的补丁（白名单去 `.`、显式禁 `**`／`//`）未见误伤**：在 main 上得 1.0 的 **22639** 条合法答案上逐条比较，**补丁后失去满分的 0 条**（含 `10 / 4 * 2` 这种中间值带小数、结果整数的写法）；**负向对照**把补丁改成"拒绝一切"后同一搜索报出 **159** 条 ⇒ 搜索有分辨力。见 [复核](open-source/rllm/countdown-reward-review-945.md) 与 [实验](experiments/rllm-countdown-reward-review/readme.md)。
- **本轮自己两处假通过（已修，判据已写进复核文档）**：① 第一版把 22639 条全报成"零敏感文本"，是统计口径漏了 `**`（判定式本身没错，已用 2 万条随机字符串暴力对照确认）；② 第二版把搜索范围收窄成"补丁明面上会拒绝的文本"，于是负向对照报 0 条回归——**一个连"拒绝一切"都查不出来的搜索不能证明没有误伤**，改成对全部合法答案比较后才报出 159 条。
- **下一步与等待条件**：若用户要参与，可发的只有补充证据（PR #946 的验证段只给了 5 个合法例子，本地跑了 22639 条），以及一条可选建议（子串 `"**" in` 判"运算符"在白名单放宽后会与语义脱节）。**对外动作全部未做，逐项待用户授权**；不建 fork 分支、不评论、不开 PR。
- **停止条件**：PR #946 被合并或关闭（`gh api repos/rllm-org/rllm/pulls/946` 的 `state`／`merged` 变化即结论失效），或用户决定不参与。
- **成果入口**：[复核](open-source/rllm/countdown-reward-review-945.md)、[复现脚本](experiments/rllm-countdown-reward-review/repro.py)、[run-info](experiments/rllm-countdown-reward-review/run-info.json)。

## 已闭环与保留方向

- **TM-004 — AReaL prompt_mean：终止修复候选。** M5 单一负责人；2026-10-20 实测跨 M 与 token_mean 一致，不构成 defect，修复撤销，不提 issue/PR。M1 不继续修改脚本或演化并行方案，用户明确的只读 review 除外。证据及分工：[handoff](open-source/areal/handoff-m5.md)、[tracker](open-source/areal/tracker.md)。AReaL 的长期学习偏好仍保留，新的问题应独立登记。

## 新选题要求

写清问题、已有证据、探索价值、下一次验证与停止条件再投入。没有新信息增量时保持等待或终止；不得以文件数、提交数或自评分数替代价值。

每条新选题需标明：**渠道类型（A/B/C/D）**、**目标仓库**、**问题来源**（issue 编号／源码路径／实测现象）、**可量化目标**（仅 C 类）、**停止条件**。缺任一项不进入队列。
