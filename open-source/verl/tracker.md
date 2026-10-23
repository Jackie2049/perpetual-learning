# verl 贡献追踪

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** verl 贡献追踪。[PR #8286](https://github.com/verl-project/verl/pull/8286) · [候选目录](readme.md)

**【类型】** 候选追踪

**【价值评估】** 8 / 10 — 记录成功、重复覆盖和放弃原因，避免反复投入已失效问题。

**【进展】** 2026-10-23 23:07 轮只读确认[我方 #8313](https://github.com/verl-project/verl/pull/8313)仍 OPEN，head `2079ab0e` 等维护者 approve，**新增社区成员 Jackmeson1 四条实质审查意见已全部核验属实**（测试命名零 CI 覆盖、middle 分法差异、断言自引用、trainer 级 error 语义），改进补丁与回复要点已备好待用户授权，见 [draft-8313-review-reply](draft-8313-review-reply.md)。新核两条 CPU 线索均已有覆盖 PR（#8174→#8175、#7771→#7785），#8334 也已有作者 draft #8335，见[覆盖核对](#近期候选覆盖核对)；不新增竞争实现。#8250 仍待证据，#7913/clip 旧 RFC 不重开。

**【大白话讲明白】**

这张表区分哪些问题修好了、哪些只是草稿、哪些已有人在做。它也保留查重失误，提醒后续不能把“没查到”当成“没人修”。

**【What–Why–How】**

- **What：** verl 候选从发现、验证到发布或放弃的状态。
- **Why：** 社区响应快，过期结果会误导投入；本地验证成功不代表仍有贡献空间。
- **How：** 按证据更新状态，将已合并、他人覆盖、未验证和待用户决定分别记录。
<!-- contribution-summary:end -->

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/verl-project/verl （2026-10 核验：仓库已从 volcengine/verl 迁移至 verl-project/verl，原链接 301 跳转）

## 当前活跃

**（2026-10-19 全候选复查重后）** 扫描报告见 [scan-1019-1140.md](scan-1019-1140.md)。注意：全新 issue 认领窗口 ≈ 0-1 天，且 **reporter 深挖过的 issue 大概率 reporter 本人已开 PR**（本轮 4/4 验证）。

**（2026-10-20 更新）** verl 新 issue 池恢复（10-16/10-19 共 3 个：#8261 perf、#8265 问卷非 bug、#8281 bug）。#8281 深挖完成（analysis-done + CPU 验证），登记为第 3 个活跃候选。

**（2026-10-23 扫描补充）** 原扫描仅登记 #8324 Modern Megatron integration、#8334 teacher-generated rollouts 和 #8342 VeOmni 端到端低精度训练。后续已确认 [#8334](https://github.com/verl-project/verl/issues/8334)有作者的 [draft #8335](https://github.com/verl-project/verl/pull/8335)，不再列为“尚未查明归属”；#8324/#8342 本轮未全面评估，不由 0 评论推断实现空缺。

| # | 候选 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|
| 0 | megatron `optimizer_offload_fraction<1` 首轮 update 崩溃 (#8237) | P0 | 42 | [#8286](https://github.com/verl-project/verl/pull/8286) **MERGED** `472dd21` | **已合并**（2026-10-20）：根因 100% 实锤（无差别 `to("cpu")` 污染 GPU 半 + fp32 参数悬空 view；P1/P2a/DeepSeek 模型源码三链）；方案 2（HDO 状态在 offload 完全不动）PR，issue #8237 已回复，4090 全验证 + 外部 GB200 验证（linmuchuiyang）→ wuxibin89 approve 后 08:06 合入。详见 [candidate](candidate-8237-offload-fraction.md) | 2026-10-19 |
| 1 | async熵异常与坍塌诊断 (#8250) | 调查/评论 | 重评见candidate | — | **needs-evidence**（2026-10-21）：仍open/0评论；旧“结果0必然全梯度0、优势离群必然性、同步唯一根因、紧窗导致无法恢复”撤回。报告源码+NumPy反例已补，不是训练实测；需有效奖励/优势/mask/ratio/版本数据，旧英文稿不可直接发 | 2026-10-19 |
| 3 | agent loop shared cap 忽略data.truncation（#8281） | P2 bug | ~25 | [#8313](https://github.com/verl-project/verl/pull/8313) | **open，等维护者 approve；新增社区审查待处理**（2026-10-23 23:07 轮）：head `2079ab0e`（fork `Jackie2049:fix-8281-truncation-policy`，4 文件），维护者 dajiaohuang review 意见已闭环（`d0ca3be1` RolloutConfig 校验 prompt_length>0 + left cap=0 正向切片；`2079ab0e` 注释精简）；middle/cap=1 反例已修复；GPU e2e 冒烟已完成。**新进展：社区成员 Jackmeson1（author_association NONE）10-23 14:30Z 发 PR 评论，四条意见经 M5 核验全部属实**——①测试文件命名 `test_cap_text_prompt_length.py` 无 `_on_cpu` 后缀且位于 `tests/experimental/`，而 verl CPU 工作流只收 `test_*_on_cpu.py`（`python_files = *_on_cpu.py`）、GPU 工作流显式 `--ignore-glob="tests/experimental"` → **当前该测试两工作流都不收集，零 CI 覆盖**，建议改名 `test_cap_text_prompt_length_on_cpu.py`；②middle 分法与 `torch_functional.postprocess_data` 不一致（postprocess 用 `left_half = max_length // 2` 奇数余量给尾，我们 `(prompt_length+1)//2` 给头；cap=1/3/5 结果不同，如 cap=5 `[0,1,2,8,9]` vs `[0,1,7,8,9]`，cap=1 `[0]` vs `[9]`，偶数 cap 相同——M5 服务器实测对照）；③测试 middle 断言自引用（期望前缀取自 `out` 自身，`[0,7,7,8,9]` 这类错误输出也能过），建议 caps 1-6 显式期望；④error 在 trainer 级是否真正中止训练需澄清（V1 可能按 prompt group 捕获异常）。②是否对齐 postprocess_data 属语义决策、④需设计澄清，改 PR/回评论均需用户授权；M5 已备改进补丁草案与回复要点，见 [candidate](candidate-8281-agent-loop-shared-cap-vs-truncation.md#当前候选的最小正上限反例) | 2026-10-20 |
| 4 | 训练前 weight-update 自检 RFC (#8252) | RFC/评论 | ~13 | — | **assessed**（2026-10-20）：三项核心事实声明全核验成立（README 无 post-sync 验证列、tests 无 weight-checker 测试、`check_weight_update_equal` main 不存在且 delta checksum 仅线缆完整性）；reporter shiyuan680 在 verl 无 PR 记录；关键张力=唯一实现先例 #7108 stale+不可合并（RFC 却要 extend 它）；不建议我方实现（成本 3/接受度未验证/无 ray-GPU 验证），价值在 RFC 技术回复（解耦为后端无关最小钩子 + 补 scope 声明）。详见 [candidate](candidate-8252-weight-checker-rfc.md)；参与与否待授权 | 2026-10-20 |

## 近期候选覆盖核对

2026-10-23 本轮读取前两项 issue 正文与全部评论（均为 0），按编号和短关键词查开放 PR，再读真正相关 PR 正文并核 API 状态/head；#8334 沿用当日已核正文，另重核关联 PR 状态。这里判断的是**是否已有覆盖工作**，不是验证补丁完整正确、CI 通过或可合并。

| 问题 | 覆盖证据 | 本项目处理 |
|---|---|---|
| [#8174 call-based 代码奖励](https://github.com/verl-project/verl/issues/8174) | 报告者 prabhu-gopal 的 [#8175](https://github.com/verl-project/verl/pull/8175) OPEN/non-draft，head `569c875e10d6e2bc55b96ef78cda5ae32e627e9f`；正文同时声明处理 APPS/TACO 参数列表与 JSON-line 字符串，以及逐例重测保留 fn_name | 两个问题均已有声明覆盖，不只按 issue 标题把 fn_name 漏失另立新修复；未执行代码评分器、下载数据或复跑作者 6 项 CPU/30 题报告 |
| [#7771 padding 稀释序列诊断](https://github.com/verl-project/verl/issues/7771) | 另一贡献者 rrrxxx0510 的 [#7785](https://github.com/verl-project/verl/pull/7785) OPEN/non-draft，head `7ffe2fa4a4f5581c27e09b8f2e438e3b61d8e7ee`；正文明确 Fixes #7771，仅调整全 mask 行的 per-sequence 指标 | 不是新空缺，不把指标修复解读为 IS/RS、loss 或模型质量修复；未复跑 reporter 的 52 项或 PR 作者的 31 项测试 |
| [#8334 teacher-generated rollout](https://github.com/verl-project/verl/issues/8334) | 同一作者 lbaolin 的 [#8335](https://github.com/verl-project/verl/pull/8335) OPEN/draft，head `8029bfb5fc5b8b6034704f9c062610613f1bf05c` | 延续已核归属，不重复实现；draft 仍待 placement/callback 反馈，不把作者 GPU 报告当本项目验收 |

本轮未验证主线问题是否仍可复现、两份候选代码的全部边界或测试充分性；不新增 fork 补丁、Issue/PR/评论、依赖或 M5 算力事项。关键词搜索还命中其他 prime_code 工作，不能据同文件名就当同一问题；#8175 正文将 #8022 的“第十例后失败”与自身两项修复分开，本轮未独立复核 #8022 的当前状态或正确性。

## 本地论证撤回，不再独立推进

- **geo_mean/GSPO clip旧RFC（2026-10-21）**：withdrawn/not-pursued。论文GMPO Algorithm 1已有log空间裁剪；对称窗口下完整序列PG的36组限定值/梯度对照通过。旧“配置空间不一致”、GSPO token KL界及训练因果依据不成立；非对称映射反例不等于上游bug。保留[更正记录](verl-clip-semantics-rfc-draft.md)，不再列作待授权发布稿。分布式/engine等价仍未确证。

## 已被上游他人 PR 覆盖（2026-10-19 全候选复查重修正）

**第 5 轮初查重因限流空响应误报 0 PR，10-19 复查重（pulls API 验证创建时间）纠正：以下 4 个候选的覆盖 PR 均为 issue 发布当天/次日由 reporter 本人或他人开出。本地验证材料保留作参考（fork 补丁已还原），不再提交。**

| # | 候选 | 覆盖 PR（open，未合并） | 覆盖时间 |
|---|---|---|---|
| 3 | GMPO geo_mean 损失随 minibatch 切分变化 (#7913) | [#7916](https://github.com/verl-project/verl/pull/7916)与[#5614](https://github.com/verl-project/verl/pull/5614)：2026-10-21重核均open未合并；#7916同样agg_loss接入并加正分母guard/测试，旧“配置层 vs 我方agg_loss”判断撤回。旧DP/verified-works证据不足，见[更正](candidate-7913-gmpo-geo-mean.md)，不重复提交 | 同日 9-18 |
| 4 | 等长 batch 3D mRoPE position_ids 不一致 (#7899) | [#7901](https://github.com/verl-project/verl/pull/7901)（iceflysnow，rebuild jagged layout） | 同日 9-17 |
| 1 | Triton fused kernels `use_remove_padding=False` 崩溃 (#8089) | [#8090](https://github.com/verl-project/verl/pull/8090)（reporter kylemontgomery1 本人，triton fused output shape） | 次日 10-02 |
| 2 | Fused Megatron LM-head 遗留 AllGather pending (#8028) | [#8029](https://github.com/verl-project/verl/pull/8029)（reporter ewan0x79 本人，publish output weights before fused kernels） | 同日 9-27 |

> 注（2026-10-21纠正）：#7916与我方旧思路的核心同为agg_loss，不因BREAKING标题就认定实质不同；缺全局分母时的本地fallback并非可靠兼容修复。#7913不再默认存在差异评论空间。#8090/#8029的下述比较是历史记录，本轮未重核；任何发言仍须查重及用户决定。

以下候选本地已验证（patch + 单测），但上游已存在他人 PR 修同一问题（均未合并）。本地 patch 与竞争实现等价或更完整，但按 verl 贡献政策「PR 描述须说明为何不重复现有 PR」，不再提交。本地材料保留在 candidate-*.md，patch 保存在 verl-pr/verl_main/（未推送任何远端）。

| # | 候选 | 竞争 PR（open，未合并） | 对比结论 |
|---|---|---|---|
| 0 | process_validation_metrics 崩在 sparse None (#6830) | [#6845](https://github.com/verl-project/verl/pull/6845)（abinggo，6-25，metric_utils +17-1、2 单测）；[#7151](https://github.com/verl-project/verl/pull/7151)（keepkeen，7-26）；[#7588](https://github.com/verl-project/verl/pull/7588)（jmaicher，8-27） | #6845 与我们的 patch 思路相同（过滤 None + pred 对齐 + 单测）。其实现仅过滤投票路径（vote_pred_vals），我们额外同步 var2vals 全量一致；核心修复等价，评论价值低 |
| 1 | dynamic-cp ceil 切分致高 DP rank 空 (#6786) | [#6869](https://github.com/verl-project/verl/pull/6869)（ajinkyajawale14499，6-27，megatron_utils +5-3 + 新 seqlen_balancing.py + 66 行测试） | 与我们的 even 切分方案一致，还抽出了独立模块。无评论空间 |
| 2 | filter_overlong_prompts 多进程 SIGTERM 挂死 (#7163) | [#7164](https://github.com/verl-project/verl/pull/7164)（mikequan0425=reporter 本人，7-27，rl_dataset +34-5） | reporter 自己提交了修复（signal 处理思路）。issue 内另有 workaround 评论（datasets≥3.3.2） |
| 5 | resume 崩在 offload pin_memory cudaError (#6690) | [#7342](https://github.com/verl-project/verl/pull/7342)（shotsan，8-10，`_try_pin_memory` 降级 + 97 行单测，即 reporter 提出的方案）；[#8235](https://github.com/verl-project/verl/pull/8235)（linmuchuiyang，10-14，onload 优化器再 load，治内存耗尽之源） | 问题的两个半场（pin 失败降级 + 耗尽原因）各有一个 open PR，均未合并。无提交空间 |

## 已过期（verl 上游已修 / 已关闭）

| # | 候选 | 上游处理 | 备注 |
|---|---|---|---|
| 3 | del_local_ckpt_after_load 不删 checkpoint (#7213) | issue 已 close（2026-08-07） | 认领者 yuyz-cyber 处理后关闭；BLOCKED 状态解除后直接作废 |
| 7 | colocate reward model 路径缺失 (#6700) | 目标文件 `verl/trainer/main_ppo_sync.py` 已从 main 删除 | 2026-10-19 核验：trainer 目录重构为 main_ppo.py/main_ppo_v0.py；colocate 功能经 #4197/#7859 等在其他路径演进。issue 本身是问句（计划支持还是删除），无代码可修，作废 |
| 6 | [FSDP] Nemotron-H 无法开 gradient checkpointing (#7146) | 根因在 transformers 上游已修 | 2026-10-19 核验：transformers main `NemotronHPreTrainedModel` 已加 `supports_gradient_checkpointing = True`（modeling_nemotron_h.py:994），发布后 verl 公开 API 调用即可用，verl 侧兼容 patch 不再必要（reporter 声称已准备但 3 个月未提交）。作废 |
| — | 蒸馏损失 micro-batch 归一化 (#7200) | 594c51b (#7225) | 扫描当天就被修了 |

## 已拒 / 已弃

| # | 候选 | 状态 | 原因 |
|---|---|---|---|
| — | tool_parser.py 静默吞异常 | REJECTED | Phase ④ 复核：仅 3 处真静默（L404/469/527），14 处有 logger。3 行 debug 日志 + experimental/ 变动快 → 性价比 <10 不达标 |
| 4 | veomni transformer del_local_after_load 类型标注错误 | HOLD | 单行类型标注，攒着等同文件实质改动顺带修，不单独提 |
| 5 | [perf] pinned Transformers 慢处理长 prompt (#8261) | REJECTED | 依赖协调型：需升级 Transformers 5.12.1→5.15+，但与 pinned SGLang 0.5.20 冲突（0.5.22 需 TF 5.19+Torch 2.14）；reporter 明确"adopting processor fix without duplicating #8240 工作"、仍在等社区决策；无干净自修点，排除 |
| 6 | prime_math reward 误判：区间括号/100x/1/100x/差一/大整数给满分，正确等式给 0 (#8346) | REJECTED | reporter 自修（2026-10-23 核实）：[PR #8347](https://github.com/verl-project/verl/pull/8347) OPEN、REVIEW_REQUIRED，新增 135 行测试（ground truth 取自 PRIME-RL/Eurus-2-RL-Data 真实数据行），覆盖区间括号/100x·1/100x·差一/>2^53 整数精度/百分号四类；已有作者实现与测试，不另开竞争修复；测试数不能认证全部判分边界。修复点 `prime_math/__init__.py`（_str_is_int/_normalize）+ `grader.py`（is_digit），纯 CPU |

## 经验教训

- **2026-10-19**：**查重输出 "None" ≠ 0 PR，必须区分限流空响应与真实零**。第 5 轮扫描时 4 个候选的 search API 返回空（限流），脚本打印 `None` 被误读为"0 个 PR，无人认领"，导致 4 个候选深挖 + 验证 + 打补丁后复查重才发现全部被 reporter 同日/次日 PR 覆盖。教训：search 返回空必须重试 + 用 pulls API 交叉验证；"无人认领"结论必须基于可解析的 JSON。
- **2026-10-19**：**reporter 把 issue 写得很深（根因+机制链+修复方向）≈ reporter 已开或即将开 PR**。本轮 4/4：gss10282025（#7913→#7916）、iceflysnow（#7899→#7901）、ewan0x79（#8028→#8029）、kylemontgomery1（#8089→#8090）全部同日/次日出 PR。深挖前先按 reporter 名字查其 PR 列表。
- **2026-10-19**：**verl 的覆盖 PR 多为 reporter 本人自修**（4/5），与 6-8 月的"他人抢修"模式不同；同期还出现 #8237 这类无人认领 5 天的例外。判断竞争优先级时应先看 reporter 身份与 issue 深度。
- **2026-10-19**：**旧候选核验要把「文件是否还在 main」也查了**。#6700 指向的 main_ppo_sync.py 已从 main 删除，问题形态随重构消失——只查 issue 状态会误判为"仍开放待修"。核验顺序：issue 状态 → 引用该 issue 的 PR → **目标文件/代码路径在 main 是否存在**。
- **2026-10-19**：**issue 里 reporter 说"打算提 PR"不等于会提**。#6690/#7146 的 reporter 都声明了意图（分别 4 个月/3 个月），最终都是别人先修（#7342）或上游根因修复（transformers）。认领声明只能降低竞争优先级，不能当作长期占位。
- **2026-10-19**：**verl bug 候选的失效速度 ≈ 2-3 个月**。8-04 验证的候选 10 月全部失效；6-11/7-25 的 issue 到 10-19 也全部有主。verl + transformers 双活跃轨道上，"旧候选核对"的价值已耗尽，应转向全新扫描或换框架。
- **2026-10-19**：**VERIFIED-WORKS 后必须尽快请求授权并提交，不能等**。4 个候选 8-04 验证完成后停在「PR 待你同意」两个月，期间上游出现 5 个覆盖 PR（6-25 至 8-27 陆续被抢），1 个 issue 被关闭。本轮核验时全部已不可提交。verl 社区对 bug 的实际响应窗口 ≈ 1-2 个月。
- **2026-10-19**：**提交前查重要用仓库新名字**。verl 迁移至 verl-project/verl 后，旧 org 名的 search 直接返回 invalid（resources do not exist）；issues API 会 301 跳转但 search API 不跳。核验上游先 `repositories/<id>` 确认当前 full_name。
- **2026-08-03**：**`grep except | wc -l` ≠ 静默吞异常**。首轮扫到 tool_parser.py "79 个 bare except"就报了"17 处静默吞异常"；Phase ④ 严格复核（看每个 except 后第一非空语句）发现 **14 处都有 `logger.warning`，只有 3 处真静默**。这类信号必须逐个看 except 后有没有日志，否则会高估价值、浪费 Phase ④。
- **2026-08-03**：**必须读完 issue 正文再评 接受度**。首次扫 #7213 只看到"无 assignee/0 评论"就给了 接受度=5；读正文才发现作者 `yuyz-cyber` 明确写了 "I can submit a focused fix"（已认领意图）。接受度应从 5 降到 2，且需先协调归属再动手。
- **2026-08-03**：**单行类型标注 / 3 行 debug 日志不值得单独 PR**。这类 P1 太琐碎，maintainer 大概率觉得刷 commit。策略：攒着，等同文件有实质改动时顺带修。
- **2026-08-03**：**checkpoint manager 的删除语义要细看 issue 期望**。初稿以为删单文件（`os.remove`），issue 正文明确要删整个角色目录（`shutil.rmtree`）+ barrier 后 + 保留 `data.pt`。Megatron 侧 `os.remove(目录)` 本来就会失败——容易漏。
- **2026-08-03**：verl 不使用 `good first issue` / `call for contribution` 标签（均为 0 个）。issue 信号靠 reaction/comment 排序 + 人工读标题。
- **2026-08-03**：`experimental/` 目录（45 个 .py）是技术债集中区（121 个 TODO 中的大量在此）。但 experimental/ 模块变动快，改动容易被冲掉，A 分要扣。
