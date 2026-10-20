# perpetual-learning — 当前工作上下文

更新时间：2026-10-21

## 当前阶段

**2026-10-20 Codex 首次定时执行更正（优先于下方历史进展）：** AReaL prompt_mean 仍是用户指定优先研究项，但原“独有 M 放大 / dummy 膨胀 / Megatron 额外 P 倍”论证已撤回。旧脚本未模拟 token_mean 的逐步分母、追加了不合法全局 dummy 步，并遗漏 MCore 的 /P。已替换 CPU 参考验证，43 项数值比较与调度检查通过；详见 [源码笔记 §10.7](notebook/frameworks/areal-architecture-source-reading.md)。下一步 GPU 先固定外层 M 只变内层 P，再验证固定 M 的 DP/dummy 分配；M=2/4 作为训练行为对照，不以最终参数不同直接判 bug。尚无真实框架/GPU 实测。

自主运行中（2026-10-19 起 18 轮 /loop 30m）。已完成：工具修复（inference_estimator.py）、verl 候选三轮核验与全新扫描（10 候选）、4 项源码级修复分析、GMPO/GSPO/SAPO 论文精读与数值证明。2026-10-20：verl 新 issue 恢复，深挖 #8281 完成（analysis-done + CPU 验证）；**#8237 offload 修复 PR #8286 已合并进 verl 主仓库**（首个 bugfix 闭环，merge `472dd21`）。详见 [当日日志](diary/2026-10-20.md)。

## 运行接续

- **网站发布待确认（2026-10-21 05:08）**：本轮成果218718e已推送main，本地网页构建通过（952标题、74阅读页、12条最新日期日志）。发布命令被安全审批拒绝，理由为缺少将内容公开到`https://jackie2049.github.io/perpetual-learning/`的明确授权；没有重试、绕过或更新gh-pages。网站仍以上一次已验证部署为准，用户确认公开范围后才恢复发布；后续心跳不重复尝试被拒操作，其他本地研究可继续。

- **M1 checkpoint调用契约收束（2026-10-21）**：Python入口不转换checkpoint，位置按图片展开后序列的绝对缓存数；benchmark预构造不跨图片的边界、后缀至少2，部分prefill后fork与`min(C,N−1)`契约一致。原始Python helper接8组片段轨迹控制通过（含图片内部越界按预期断言），原10组成对C++检查回归通过；非真实engine/fork/设备验证。见[笔记](notebook/frameworks/deepinfer-initial-survey.md#checkpoint坐标与cached-prefix-fork契约)。chunk学习先收束，下一步只读追MTP接受数到最终KV缓存长度的提交链，不竞争修复。

- **M1 chunk sizing复核（2026-10-21）**：#108/#95/#98均open、base774ce244，head分别ab47ee4/2a5db94/6a93461。#95/#98资源推进与#108 checkpoint/lookahead并非同一修复；10组成对原始C++片段检查通过，包含旧版也通过的无预算控制和图片边界限制。完整Scheduler头文件在本机Clang/Linux pthread依赖处受阻，未跑原始Gloo测试，不改兼容层/上游方案。见[笔记](notebook/frameworks/deepinfer-initial-survey.md#chunk-checkpoint消费与预留尾段)。下一步只读核对Python checkpoint入口/坐标与cached-prefix fork契约，不竞争已有PR。

- **M1 DeepInfer EPLB阶段收束（2026-10-21）**：已有#107身份窗口CPU复核后，补固定DeepEP/DeepGEMM producer与sample→窗口→统计→清零链；计数是专家路由接收工作，不等于accepted-token/去重通信token。新增8组原始manager/view/step CPU检查通过，明确prefill跳过窗口仍统计清零、EP=1整步返回；非kernel/collective/shuffle/性能验证。见[笔记](notebook/frameworks/deepinfer-initial-survey.md#专家计数的来源与清零时序)，不新增缺陷/竞争修复。03:32核验main774ce244、89个open条目均PR是当时快照，非本轮重新扫描。

- **M1 V1边界阶段收束（2026-10-21）**：原始注册/绑定/提交包装体6组控制通过（Ray/TQ等替身），默认阻塞≠跨rank事务；get异常不保证其他任务已停/撤销，亦非全量错误聚合。与快照异常21组、同步控制21组共同形成[来源与失败边界笔记](notebook/frameworks/verl-async-logprob-provenance.md#快照rpc等待与失败传播)，不证明原生恢复或#8250根因。#8250等真实配对观测，不继续堆替身代替根因证据；未改上游方案。

- **M1同步完成点复核（2026-10-21）**：固定版本原始adapter/manager协程体21组本机替身控制与故障注入通过；发送、确认、cache、版本、finalize、resume不是一个原子提交。异常后不继续后续阶段，不等于回滚/取消/重试安全，失败后暂停也不直接判bug。见[既有笔记补充](notebook/frameworks/verl-async-logprob-provenance.md#同步完成点与异常传播)，#8250仍待补证，未改上游方案。

- **M1异步观测契约（2026-10-21）**：V1 separate-async的rollout/old/current来源已追踪；bypass复制与CPU快照交换不能当独立同步验证。7组原始路由成功路径替身+7类概率/版本反例通过，版本标签≠tensor内容校验、选中token概率匹配≠完整分布/参数相同。#8250仍待补证，见[学习笔记](notebook/frameworks/verl-async-logprob-provenance.md)，不作根因或新bug定性。

- **M1 #7913复核收尾（2026-10-21）**：报告版本metadata→FSDP→PG调用链已核对，3组原始主机方法替身控制+12组NumPy共享参数/显式DP平均模型通过，非native训练/patch验证。旧split的DP段误把loss当gradient、未跨rank平均，旧verified-works/空响应分母说法撤回；#7916同样agg_loss接入而非仅配置层，已存在同核心#5614/#7916，不再重复推进。见[更正记录](open-source/verl/candidate-7913-gmpo-geo-mean.md)。

- **M1纠偏优先（2026-10-21）**：#8250旧确定因果诊断及GSPO错误比较已撤回；GMPO完整序列PG已限定复核，对称窗口36组值/梯度模型通过，论文Algorithm 1本就在log空间。clip旧RFC已withdrawn/not-pursued，不再列作待发布稿。均非torch/autograd/engine/训练等价证明；历史“evidence-complete”与“配置空间不同”不覆盖此纠正。#8281仍待决定，不自动改方案/发布。入口见 [诊断](open-source/verl/candidate-8250-async-entropy-collapse.md)、[撤回记录](open-source/verl/verl-clip-semantics-rfc-draft.md)、[GMPO笔记](notebook/papers/gmpo.md)。

- **M1 心跳隔离接续（2026-10-20）**：为避免共享 checkout 的分支切换/日志冲突覆盖未提交成果，当前心跳使用 `/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/perpetual-learning-cp-finalize`，从远端 main 核对后继续；统一交接仍是远端 main。不要重置 `/Users/jackiemac/workspace/perpetual-learning` 或把该共享目录未推送的分支整体带入本项提交。GPU/AReaL 分工不变。

- 网页 demo 已上线：[探索工作台](https://jackie2049.github.io/perpetual-learning/)。只读展示仓库快照，更新内容后按 [网页维护说明](web/readme.md) 发布；当前采用 gh-pages 分支发布。2026-10-21 已同步chunk sizing限定复核快照4d9dd6c（gh-pages f5d19b8），Pages 工作流39493927012成功；构建952条文档标题、74个阅读页、最新日期11条日志（含其他会话已有记录）。DeepInfer笔记来源链接与本轮快照一致、#8250仍待补证；网络发布/查询使用获准外网权限，未改代理/VPN。发布收尾记录不递归触发重建。

- 文档呈现要求：评分维度直接写“影响、成本、接受度、契合度”，使用简短直观名称。成本方向已在 [SOP](open-source/SOP.md) 澄清，历史分值复用前核对。

- 2026-10-19 收尾机制更新：每项实际工作必须有日志，完成、失败或受阻均记录。当前克隆已启用 Git 提交检查；下一轮先重读 [AGENTS.md](AGENTS.md) 与 [日志工具说明](tools/diary/readme.md)，使用自动取时入口。其他克隆需安装检查；现有 Claude 会话是否重读规则未验证。

- 状态：Codex 定时运行待接续，本轮已完成 DeepInfer 系列源码/CPU 工作；AReaL prompt_mean 实测已由本会话完成闭环（用户 2026-10-20 恢复 VPN 并明确继续后，见 [handoff-m5](open-source/areal/handoff-m5.md) 分工记录的更新）。沿用日志收尾工具与提交检查；后续写入前核对 WORK 与并行改动。
- **AReaL GPU 服务器环境（下次选用入口，2026-10-20 登记）**：服务器 4090-1 上 AReaL 环境为 **uv venv `/home/zxw/areal/.venv`**（非 conda；clone 于 `/home/zxw/areal`，基线 `3b93679`；Python 3.12.15 + torch 2.9.1+cu129 + transformers 5.3.0 + sglang 0.5.10.post1 + ray 2.55.1 + megatron-core 0.19.0，14G）。**启动必须 `PATH=/home/zxw/areal/.venv/bin:$PATH` 前置**（否则 sglang 子进程错用系统 python3）；离线 `HF_HUB_OFFLINE=1`；信任环境 `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1`；显存紧 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。空闲 GPU 0/1/7。**修改环境时从该 venv 基础上新建**（用户 2026-10-20 明确），不直接改动已验证基线。详见 [AReaL readme](open-source/areal/readme.md#服务器环境gpu-4090-1-2026-10-20-实测可用) 与 [源码笔记 §10.10](notebook/frameworks/areal-architecture-source-reading.md)。
- 待执行的用户任务：**AReaL prompt_mean 切分敏感点 GPU 实测闭环（2026-10-20 用户授权）——已完成**。服务器 4090-1（VPN 恢复后）跑通 AReaL 框架（gsm8k PPO smoke，Qwen2.5-0.5B + FSDP2 + sglang，128 步无错），并按 §10.7 判据完成受控实测：prompt_mean 跨 M 比值 1.301 ≈ token_mean 1.273（M=2 vs 4）、P-spread 1.216 ≈ 1.192——**不构成 defect，`step_scale=M/G` 与 PR #1546 契约一致，候选修复撤销，不提 issue/PR**。详见 [源码笔记 §10.10](notebook/frameworks/areal-architecture-source-reading.md)、[tracker](open-source/areal/tracker.md)、受控脚本 [areal_prompt_mean_gpu_probe.py](experiments/scripts/areal_prompt_mean_gpu_probe.py)。Megatron 后端 GPU 实测未做（megatron-core 硬依赖 Transformer Engine 需长编译；双后端同源已由 §10.5 源码确立）。
- 自主探索任务：见「本轮进展」与 [verl tracker](open-source/verl/tracker.md)（当前活跃 3 候选：#8250 诊断、clip 语义文档澄清、#8281 agent loop truncation；**#8237 已收尾关闭**）；vLLM 方向进行中（#61160 评论已发布待回应；**#61229 已核实被 #62065/#61549 认领，无竞争空间，见 [readme](open-source/vllm/readme.md) 与 [学习笔记](notebook/frameworks/vllm-nixl-lease-correctness-reading.md) 已记录**）。**#8281 PR 草稿已本地备好（2026-10-20：源修复 + 真实单测，见 [candidate](open-source/verl/candidate-8281-agent-loop-shared-cap-vs-truncation.md)，patch 存 [patches](open-source/verl/patches/8281-truncation-policy.patch)，落 `/Users/jackiemac/workspace/verl` 分支 `fix-8281-truncation-policy` commit `7ace28db`；发布仍需授权 + verl CI 验证）**。**2026-10-20 新增：#8252 评估完成（三项事实声明核验成立、#7108 stale 是移动靶、不建议我方实现、价值在 RFC 技术回复，见 [candidate](open-source/verl/candidate-8252-weight-checker-rfc.md)）**。**2026-10-20 更正：#8281 存在关联旧 PR #4600**（reporter 创建时交叉链接；陈旧、dirty、预 CT 重构，非当前修复点），发布形态倾向 comment 而非 PR，见 [candidate](open-source/verl/candidate-8281-agent-loop-shared-cap-vs-truncation.md)。**AReaL prompt_mean 切分敏感点（§10/§10.3/§10.5）已定为首个优先实测项**：用户判定研究价值较高，待 VPN 可用后配环境做 GPU 实测（ppo_n_minibatches 2 vs 4，FSDP2/Megatron 双后端），见 [tracker](open-source/areal/tracker.md)。
- **等待用户决定的事项**（均需授权/资源）：②发布 vLLM #39096 增量评论；⑥#8281 的发布形态（评论 or PR，见 candidate）。①#8250先补证，旧诊断不可直接发布；⑤clip旧RFC已撤回、不再推进，不需要为旧稿决定发布形态。~~③ vLLM #61160 修复定位评论~~ **已发布**（2026-10-19 10:49，[comment](https://github.com/vllm-project/vllm/issues/61160#issuecomment-6258592764)），待回应跟进。**#8237 bugfix 已合并**（2026-10-20：PR [#8286](https://github.com/verl-project/verl/pull/8286) 已合入 verl 主仓库，merge `472dd21`，wuxibin89 approve；根因实锤 + 4090 全验证 + 外部 GB200 背书；详见 [candidate](open-source/verl/candidate-8237-offload-fraction.md)）。
- 下一步：见下方「待讨论与下一步」。
- 运行范围：Codex 当前聊天 + 已保存的 30 分钟心跳任务；本地执行仍依赖应用运行及项目可访问。每轮读取本文件与 AGENTS.md 后接续。

## 当前用户需求与决定

- **主机分工（2026-10-20 最新用户明确）**：所有 GPU 访问/环境/实验由 M5 Agent 负责；AReaL prompt_mean 候选/PR 整项由 M5 单一负责，M1 阶段收尾，后续定时任务不并行修改其思路、代码、实验或定性。M1 继续独立研究，只有用户重新指派或明确要求只读 review 才涉及该项。统一 [交接说明](open-source/areal/handoff-m5.md)。

- **高优调研项（2026-10-20）**：**DeepInfer — DeepSeek 官方推理框架**。DeepSeek 今天（10-20）开源其生产服务推理框架 `deepseek-ai/DeepInfer`（Apache-2.0，快照 8b161497；GitHub 语言统计为字节数，非百万行，支持 NVIDIA SM100 + 华为 Ascend 950 NPU）。用户定为项目高优先调研项。初览笔记见 [notebook/frameworks/deepinfer-initial-survey.md](notebook/frameworks/deepinfer-initial-survey.md)；与 verl/vLLM/FlashMLA/DeepEP/Ascend 既有积累高度相关（batch-invariant、P/D 分离、MTP、MoE EPLB、NPU）。**用户另关注：在 DeepSeek 生态提优化 PR 能否被采纳——调研结论见 [deepseek-ecosystem-pr-adoption.md](notebook/frameworks/deepseek-ecosystem-pr-adoption.md)：外部 PR 采纳门槛高、周期长（FlashMLA open PR 平均 185 天），"修复具体 bug"比"大幅优化"更可能被合入；相对友好的是 DeepEP/DeepGEMM；综合性价比 verl/vLLM/AReaL 仍高于 DeepSeek 生态**。
- **开源技术掘金未来一周三大重点（2026-10-20 用户定）**：**DeepInfer、verl、AReaL**。①**DeepInfer**（DeepSeek 官方推理框架，高优先调研，见上）；②**verl**（贡献主轨道，#8237 已合并，next #8281/#8250）；③**AReaL**（蚂蚁系 RL 训练，prompt_mean 切分敏感点优先实测）。三者在"技术掘金（学习+贡献）"子领域并列为周度关注对象；vLLM 降为次优先级。
- **方向偏好**：用户希望近期自主运行重点跟进 verl 与 AReaL 两个 RL 训练框架（2026-10-20）。verl 已是贡献主轨道；AReaL（areal-project，蚂蚁系，合并效率 91% 为活跃社区最高）作为新增关注/潜在贡献轨道。**2026-10-20 已建立 AReaL 贡献轨道**（[readme](open-source/areal/readme.md) · [tracker](open-source/areal/tracker.md)）：首次扫描确认无未认领贡献空间（reporter 自修规律与 verl 相同），跨框架观察 #1730 M2PO microbatch 与 verl #7913 同类、同 reporter。**用户判定（2026-10-20 16:xx）：AReaL prompt_mean 切分敏感点（§10/§10.3/§10.5，双后端同源）研究价值较高，定为优先推进项；已确认 VPN 连上后做 GPU 实测（ppo_n_minibatches 2 vs 4 两后端对照）闭环**。
- 项目名称、首页标题和仓库标识统一使用 `perpetual-learning`，删除首页旧名称标识行。项目定位继续保留 [README.md](README.md#项目说明) 中的用户原文。
- README.md 是整个项目的总览、对外呈现与用户导航，按主题、框架、成果提供快捷入口；文件组织使用可点击的目录表，作为统一入口置于项目说明与运作机制之后，不再单设重复的“项目导航”章节。[AGENTS.md](AGENTS.md) 为 Agent 内核；本文件记录当前工作上下文。
- README 不放内部整理计划、迁移过程或待决定事项；当前阶段与待办由本文件维护，完成过程写入 diary。
- README 章节顺序：项目说明 → 运作机制 → 文件组织 → 知识导航 → 上机实战 → 主流框架 & 开源贡献 → 论文学习 & 专题研究 → 工具沉淀 & 技能打磨 → 规划 & 进展；GitHub 链接保留在末尾。
- notebook/ 保留 fundamentals/ 基础知识、experiments/ 实验记录、roadmap.md 规划、knowledge-map.md 知识整理与总览；原源码学习目录改名 frameworks/，总览见 [notebook/readme.md](notebook/readme.md)。
- tools/ 注重工具价值与复用；一次性、纯实验性和低价值重复实现归 tools/legacy/，新增 [tools/readme.md](tools/readme.md)。
- open-source/<framework>/ 按目标开源项目保存贡献材料，与 notebook/frameworks/ 的源码理解分开并相互链接；diary/ 延续原有功能，入口为 [diary/readme.md](diary/readme.md)。

## 本轮进展（2026-10-19 持续运行 18 轮）

**工具**：inference_estimator.py 修复验证（带宽硬编码 → GPU 规格、吞吐三口径分离、--batch、h20/rtx4090 规格），纳入推荐入口。

**verl 贡献轨道**：
- 旧候选核验（8 候选全失效：4 个 VERIFIED-WORKS 被上游 PR 覆盖、备选失效）；vLLM 6 月草稿核验（3 作废，#39096 有残余价值）。
- 全新扫描（75 issue 窗口）+ 复查重纠错：**10 候选 9 个被 reporter 同日/次日 PR 覆盖**（含第 5 轮查重限流误报），教训 8 条入库。
- 4 项修复分析：#7913 geo_mean microbatch 不变性（**verified-works，CPU 验证 rel L2=0**，上游 #7916 方案不同未合）、#8089/#8028/#7899（均被 reporter 自修 PR 覆盖，fork 补丁已还原）、#8237 offload fraction（**已合并**：根因 100% 实锤 + PR [#8286](https://github.com/verl-project/verl/pull/8286) 2026-10-20 合入 `472dd21`）。
- 新候选：#8250 async 熵爆炸诊断（analysis-done，草稿待授权）、clip 语义文档澄清（evidence-complete）、**#8252 训练前 weight-update 自检 RFC（assessed，三项事实核验成立，参与与否待授权）**。
- **#8281 PR 草稿本地备好**（2026-10-20）：源修复（`_cap_text_prompt_length` 尊重 data.truncation 四策略 + unknown raise）+ 真实单测 `test_cap_text_prompt_length.py`；真源 verbatim exec 32 断言全过，两文件编译通过；patch 存 [patches/8281-truncation-policy.patch](open-source/verl/patches/8281-truncation-policy.patch)，分支 `fix-8281-truncation-policy`（7ace28db）。**发布形态倾向 comment**（reporter 定位为「契约讨论」）；存在关联旧 PR #4600（reporter 创建时交叉链接，陈旧+dirty+预 CT 重构，非当前修复点），若走 PR 需说明为何不重复 #4600。发布仍需授权 + verl CI 验证（本机无 ray/transformers/torch）。

**AReaL 轨道（新建立，2026-10-20）**：首次扫描确认近期可行动 bug 全被 reporter 自修 PR 覆盖（#1786→#1787、#1730→#1732），**无未认领空间**；老 issue 多被 assignee 认领。关键跨框架观察：`gss10282025` 同时开 verl #7913（geo_mean microbatch，我方已深挖）与 AReaL #1730（M2PO microbatch 不变性），我方 verl 经验可直接迁移作 #1730 技术回复。产出 [AReaL readme](open-source/areal/readme.md) · [tracker](open-source/areal/tracker.md)，登记 README/notebook 索引。**2026-10-20 §10 深化**：精读 FSDP2 后端 batch 切分 + 发现 **prompt_mean 归约切分敏感点**。**15:11 已做忠实归一化链核验（确认）**：逐行对照真源码，`step_scale = M/G`（M=DP 同步有效 slot 数含 dummy，非直接 ppo_n_minibatches），`total_weight≡1`（`engine_weight=rows/global_rows` 是 DP 归一权重非跨 mb share，误读为跨 mb share 会误推 total_weight=N 得出反结论）、`loss_multiplier=dp_size`（统一乘所有归约）。忠实链累计梯度 = `dp_size·(M/G)·Σ_all loss·w`，**随 M 精确线性放大**（numpy 复刻 1×/2×/3×/4×/6×，与组边界切割无关），token_mean/seq_mean/constant 不受影响；PR #1546 声明「N 步平均恢复 full-batch」但实际 N 次依次 optimizer_step 累加，测试只断言均值未覆盖累加；上游无公开讨论。**定位升级为「切分敏感点（高置信源码级确证，缺陷定性待实测）」**——待实测项收敛为「线性放大在真实自适应优化器下是否构成 defect」（需 AReaL 环境 ppo_n_minibatches 2 vs 4 对照）。继 M2PO 掩码（#1730）后 AReaL 第二个「batch 切分影响 update」实例，位于归约侧，与 verl #7913 同源。见 [源码笔记 §10](notebook/frameworks/areal-architecture-source-reading.md)、[验证脚本](experiments/scripts/verify_areal_prompt_mean_split.py)。**15:40 精确化**：①**dummy-slot 膨胀**——transport-dummy 的 `loss_mask` 被 zero_（data.py:886-887）、`engine_weight=0`，对梯度与 `total_weight` 均无贡献，**但仍计入 step_scale 的 M**（loss_reduction.py:115）：每个 dummy 把 M 抬升一单位却不加梯度 → 真实响应梯度额外放大约 dummy 个数个单位 M（numpy 复刻 Case 4：real_mbs=1+1dummy→2×、2+1→3×、3+1→4×）；②**调用链闭环**——rl_trainer.py:929-1240 每 global_step = rollout→aux 打分→advantage→`ppo_update`（L1155，内含 M 次 train_batch/optimizer_step 累加）→`update_weights`（L1232 发布权重），prompt_mean 的 step_scale 放大进入每次发布权重。§10 补 10.3.1/10.3.2。**16:13 跨后端对照（§10.5）**：`step_scale=M/G`（loss_reduction.py:115）由 actor.py:600 在后端之上共享，**FSDP2 与 Megatron 双后端同源**——Megatron 后端 prompt_mean 同样随 ppo_n_minibatches 线性放大；Megatron 的 `loss_multiplier=dp_size*len(mb_list)`（megatron_engine.py:2382）额外乘内层 pipeline mb 数 P（常数因子，由 pp_size/mb_spec 决定，非 ppo 切分敏感项），忠实链 = `dp_size·(M/G)·P·Σ_all`，numpy 复刻 Case 5 确认 M=1/2/4→1×/2×/4×、P=2/P=1→2.000。待实测项扩展为：跨配置 ppo_n_minibatches 2 vs 4 两后端各一组对照。**16:44 §10.6 深化（Megatron 三分支 + per-token 不抵消）**：精读 `megatron_engine.py::train_batch`（L2325-2414）确认归一化按 `calculate_per_token_loss` 分流**三分支**；`PromptMean.engine_weight` 是标量非 dict、未用 `@named_loss_normalizer` → prompt_mean 恒走分支 2 或 3 从不走分支 1；MCore 0.19.0 默认 `calculate_per_token_loss=False`（`transformer_config.py:277`，经 `core_v0.19.0` tag 核验，AReaL 锁 0.19.0 不设该字段）→ 实际默认路径即 §10.5 分支 3、该结论对默认配置成立；**per-token 路径（分支 2）忠实推导 + numpy 复刻（Case 6）确认不抵消切分敏感**：`loss_numerator=loss·engine_weight·(T_global/W_global)/cp_size`，MCore `/T_global` 与分子 T_global 精确相消、`W_global_i≡1` → 忠实链 `(M/G)·Σ_all`，M 线性放大保留且 dp_size/P 被消掉（P=2/P=1→1.000）。**切分敏感在 FSDP2 / Megatron 默认 / Megatron per-token 三条路径都稳健，是 actor 层 `step_scale=M/G` 固有**。待实测项不变。产出源码笔记 §10.6、验证脚本 Case 6。

**学习与知识**：GMPO/GSPO/SAPO 三论文精读（含 verl 实现对照、三向 clip 语义对照表、非对称温度、数值证明）；损失注册表唯一性审计（geo_mean 唯一绕过 agg_loss）；DRO 推导补全；排错指南新增 async 熵爆炸章节。产出笔记 3 篇 + 验证脚本 2 个（verify_gmpo_geo_mean_split.py、verify_gmpo_clip_semantics.py）。

**方法学教训**（已入库）：查重 "None" ≠ 0（限流误报）；reporter 深挖 issue ≈ reporter 已开 PR；认领声明 ≠ 会提；verl 认领窗口 0-2 天且 reporter 自修为主。

## 待讨论与下一步

1. **等待用户授权/资源**（长期未决，见「运行接续」5 项）——发布评论/文档、#8281 发布形态、方向调整。#8237 已收尾（PR #8286 ready + issue 已回）。
1. **未来一周开源技术掘金三大重点（用户 2026-10-20 定）**：**DeepInfer / verl / AReaL**。其中 DeepInfer 主研（推理框架源码深挖），verl 主贡献（续 #8281/#8250），AReaL 主实测（prompt_mean 切分敏感点）。vLLM 降为次优先级。 **这是跨主机整体分工，M1 不接管 AReaL；M5 独立推进其完整候选/PR。**
2. **等待用户授权/资源**（长期未决，见「运行接续」5 项）——发布评论/文档、#8281 发布形态、方向调整。#8237 已收尾（PR #8286 ready + issue 已回）。
3. **M1当前学习/贡献复核**：V1来源、同步/快照异常及RPC等待边界阶段收束，#8250等真实配置/配对观测，不以更多替身替代根因证据；#7913不重复推进、clip旧RFC不再推进、#8281待决定。DeepInfer EPLB阶段收束，接续chunk checkpoint，见下项；不重复已收束的#54/#57/共享内存生命周期模型。M1不访问GPU、不涉及AReaL整项，任何上游发言仍须拍板。
3. **高优调研（用户 2026-10-20 定）**：**DeepInfer**——确定性、ZOS、DSpark、YoCo/CP、生命周期、EPLB及chunk/cached-prefix契约已有源码/限定CPU学习，完整扩展/设备正确性与性能未测。chunk阶段收束，10组成对C++及8组Python helper/片段轨迹控制通过，非完整Scheduler/fork/Gloo验证。下一步只读追MTP接受数到最终KV缓存长度的提交链，与既有乐观长度说明衔接，不继续堆同类替身代替完整验证；见[笔记](notebook/frameworks/deepinfer-initial-survey.md#checkpoint坐标与cached-prefix-fork契约)。不预设新缺陷、不竞争已有修复；M1不访问GPU/AReaL，任何上游发言仍须用户拍板。
4. **AReaL prompt_mean（原 M5 单一负责项，用户 2026-10-20 恢复 VPN 后明确本会话继续）——实测已闭环**：框架跑通（gsm8k PPO smoke 128 步无错）+ 受控实测（§10.7 判据，M=2 vs 4 + 固定 M 变 P，prompt_mean 与 token_mean 行为一致：跨 M 1.301≈1.273、P-spread 1.216≈1.192）——**不构成 defect，候选修复撤销，不提 issue/PR**，作为 review 依据存档（见 [源码笔记 §10.10](notebook/frameworks/areal-architecture-source-reading.md)、[tracker](open-source/areal/tracker.md)）。Megatron GPU 对照未做（TE 硬依赖）。分工状态见 [handoff-m5](open-source/areal/handoff-m5.md) 与 WORK「运行接续」。
3. 按课题逐步关联现有实验执行材料与 notebook/experiments/ 的分析记录；新增实验优先就近组织，维护双向入口。
4. 维护按职责划分的一级目录；新增或整理内容时整体检查归属、引用和输出路径，避免再次堆出按文件类型划分的一级目录。根目录 projects/ 暂不新增。
5. 按价值、使用关系和证据继续筛选历史工具，避免只按文件名批量归档；其余历史工具继续按同一标准检查。

## 接续入口与约束

- 学习与探索：[notebook/readme.md](notebook/readme.md)、[roadmap.md](notebook/roadmap.md)、[knowledge-map.md](notebook/knowledge-map.md)。本轮学习线（RL 损失家族）已收束，新课题未定。
- 贡献：[open-source/README.md](open-source/README.md)、[verl tracker](open-source/verl/tracker.md)、[AReaL tracker](open-source/areal/tracker.md)。当前活跃：#8250（诊断评论待授权）、clip 语义文档澄清（待授权）、#8281（agent loop truncation，analysis-done，待发布决定）；**#8237 已合并进 verl 主仓库**（PR [#8286](https://github.com/verl-project/verl/pull/8286)，merge `472dd21`）；vLLM #39096 评论待授权；**AReaL 源码已精读至 §10.8（Megatron 三分支 + §10.7 对抗复核撤回缺陷表述 + GPU 实测环境就绪但 VPN 中断待恢复）**，修复候选 `step_scale=1/G` 降级为假设（patch 草稿 [patches](open-source/areal/patches/) + 受控脚本初稿），实测按 §10.7 判据；megatron mbridge 与 v2 各 controller 仍未读（价值次优先，v2 为非默认服务化路径）。
- 本地目录已核实为 `/Users/jackiemac/workspace/perpetual-learning`；旧会话可能仍绑定已不存在的 rollout-infra 路径。GitHub 仓库为 [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)。
- 原有未跟踪文件 diary/2026-08-14.md 属于既有用户工作，保留原状，不夹带提交。
- 历史服务器状态、访问方式和“已验证”结论按需复核，不复述凭据。
