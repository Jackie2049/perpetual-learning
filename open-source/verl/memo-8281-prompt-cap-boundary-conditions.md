# verl 提示词 cap/长度边界条件系统化备忘（G4 缺口转稿）

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 从 [Issue #8281](https://github.com/verl-project/verl/issues/8281)／[PR #8313](https://github.com/verl-project/verl/pull/8313) 实战提炼 prompt cap/长度边界条件矩阵（-0 切片陷阱、偶奇 cap 分配、cap=1 token 保留选择、策略空间不一致、prompt/response 校验不对称），作为 rollout 调研专项缺口 G4 的本地技术备忘录。[专项证据](https://github.com/Jackie2049/perpetual-learning/blob/main/notebook/rollout-performance-research.md)

**【类型】** 技术备忘录（本地/fork 轨道，铁律约束不上主仓库）

**【价值评估】** 6 / 10 — 边界条件矩阵与最终修复已由 fork head 71fd3562 验证，但"系统化处理未见讨论"的领域缺口结论只到文献核对层，未做跨路径统一提案。

**【进展】** 2026-10-29：基于 fork head `71fd3562`（用户侧落地：middle cap=1 保留首 token＋测试改名还原）与本地 veRL 源码快照 `c0e7551` 完成边界矩阵整理；未触碰上游（铁律）。PR #8313 仍 OPEN/MERGEABLE/REVIEW_REQUIRED 待用户逐字审定 ④ 回复。

**【大白话讲明白】**

训练里常有一个"提示词最长多少 token"的开关。早期实现无论配置都只留提示词末尾，开头指令可能被丢掉；修复按配置分流后，又冒出两个小陷阱：只允许 1 个 token 时，切片写法会让输入 10 个 token 变成 11 个（超出上限）；上限为 0 时同样会"取整个列表"。这类"边界值"问题（cap=0/1、奇偶 cap 分配、保留头还是尾）在 126 篇 rollout 论文里没有任何一篇讨论，是真实的微观空白。本备忘录把这套边界条件整理成一张可复用的检查矩阵，供以后写代码或读论文时逐项核对。

**【What–Why–How】**

- **What：** Agent Loop 共享提示词上限接口在 cap 极小（0/1）、奇偶分配、四策略（error/left/right/middle）与配置校验上的全部边界行为。
- **Why：** Python 负索引 `-0` 语义使"取末尾 N 个"在 N=0 时取整表；偶奇 cap 的余数归属改变存留 token；cap=1 时头/尾路径保留不同 token，而现有论文与跨路径实现均未系统处理。
- **How：** 逐项固定 fork head `71fd3562` 实现与本地源码对照，产出边界矩阵＋最终修复语义＋跨路径遗留分歧；作为本地备忘，不构成上游提案。
<!-- contribution-summary:end -->

## 背景与定位

本备忘录是 rollout 调研专项收尾阶段缺口 **G4（cap/长度类边界条件未见讨论）** 的转稿（专项 §13.15 G4；来源为 verl #8281/#8313 实战）。用途：

1. 为 verl 贡献轨道保留可复用的边界条件检查清单（本地/fork 准备，发布受铁律约束）；
2. 记录最终修复（fork head `71fd3562`）的精确语义，避免后续重实现时引入同类回归；
3. 记录一处未调和的跨路径分歧（cap=1 首/尾 token），供用户或后续贡献决策。

**核验版本**：fork head `71fd3562`（2026-10-24 用户侧 push，"keep first token at cap=1 for middle; revert test rename"）；本地 veRL 快照 `c0e7551`（fix/8237 分支，仍为修复前无条件左截断版本，用作对照）；`postprocess_data` 语义取自本地 `verl/utils/torch_functional.py`。

## 一、最终修复语义（fork head 71fd3562）

`AgentLoopBase._cap_text_prompt_length`（`verl/experimental/agent_loop/agent_loop.py`）：

```python
truncation_policy = self.data_config.get("truncation", "error")
if truncation_policy == "left":
    return prompt_ids[len(prompt_ids) - prompt_length:]   # 正向切片，cap=0 时空尾
if truncation_policy == "right":
    return prompt_ids[:prompt_length]
if truncation_policy == "middle":
    head = max(1, prompt_length // 2)                      # floor；cap=1 时保首 token
    tail_length = prompt_length - head
    return prompt_ids[:head] + prompt_ids[len(prompt_ids) - tail_length:]
raise ValueError(...)                                      # 默认 error / 未知策略 fail loudly
```

`RolloutConfig.__post_init__`（`verl/workers/config/rollout.py:316` 起）：`prompt_length <= 0 → ValueError`（`d0ca3be1`，采纳维护者 dajiaohuang review 建议）。

middle 显式期望（测试 `tests/experimental/agent_loop/test_cap_text_prompt_length.py`，输入 `list(range(10))`）：

| cap | 输出 | 说明 |
|---|---|---|
| 1 | `[0]` | 保**首** token（head=max(1,0)=1，tail 预算 0 → 空） |
| 2 | `[0,9]` | head 1 + tail 1 |
| 3 | `[0,8,9]` | head 1 + tail 2（余数给尾） |
| 4 | `[0,1,8,9]` | head 2 + tail 2 |
| 5 | `[0,1,7,8,9]` | head 2 + tail 3 |
| 6 | `[0,1,2,7,8,9]` | head 3 + tail 3 |

## 二、边界条件矩阵（可复用的检查清单）

### 2.1 Python -0 切片陷阱（负索引 × 零预算）

`lst[-0:] == lst[0:] == 整个列表`。凡"取末尾 N 个"用负索引 `lst[-N:]`，N=0 时上限静默失效。**实锤两处**：

- **cap=0 + left**：修复前 `prompt_ids[-0:]` 取整表（上游 review 指出；cap=0 经配置校验拦截，函数层正向切片双保险）。
- **cap=1 + middle**（原候选 `head=(cap+1)//2=1`、尾段预算 0）：`prompt_ids[-0:]` 取整表 → 10 token 输入变 11 token 输出（超出上限的**长度爆炸**，非仅策略差异）。

**防再犯规则**：末尾截断一律正向切片 `lst[len(lst)-n:]`（n=0 天然空）；负索引只用于 n>0 且经证明的情形。

### 2.2 偶奇 cap 余数归属（head/tail 分配）

- **ceil 型** `head=(cap+1)//2`：余数给头。cap=3 → `[0,1,8,9]`（保头多 1）。
- **floor 型** `head=cap//2`：余数给尾。cap=3 → `[0,8,9]`（保尾多 1）。
- 最终修复选 **floor**（与 `torch_functional.postprocess_data` 的 `left_half=max_length//2` 对齐），**唯一例外**是 cap=1（见 2.3）。

### 2.3 cap=1 首/尾 token 保留选择——跨路径未调和分歧

- `postprocess_data`（`torch_functional.py:514`，数据路径）在 `max_length=1`：`left_half=0`、`right_half=1` → `input_ids[:, :0] + input_ids[:, -1:]` 保**末** token。
- agent_loop 最终修复在 cap=1：`max(1, 1//2)=1` → 保**首** token（docstring："preserve the prompt's leading identity"）。
- **两条截断路径在 cap=1 分歧（首 vs 末）**。对指令在开头的任务 prompt，保首更保真；但分歧本身未被任何文档或测试声明，PR #8313 未调和。这是遗留设计决策，不在本备忘录裁决。

### 2.4 策略空间与默认行为一致性

| 路径 | 支持的策略 | error 行为 | middle |
|---|---|---|---|
| `agent_loop._cap_text_prompt_length`（修复后） | error/left/right/middle | raise ValueError | 支持 |
| `torch_functional.postprocess_data` | error/left/right/middle | raise NotImplementedError | 支持（floor） |
| `multiturn_sft_dataset.py:117` | **error/left/right**（assert） | raise ValueError | **不支持** |
| `rl_dataset.py:118` | 只读取 `self.truncation`，全文无应用 | 无 | 无（空契约） |

- **RLHFDataset 的 truncation 是空契约**：读配置但不执行；RL/agent_loop 路径唯一防线是 `filter_overlong_prompts=True`（默认过滤超长）。`data.truncation` 在 RL 路径实际不被过滤逻辑引用、也不被截断逻辑引用（修复前）。
- 同是 "error"，agent_loop 抛 `ValueError`、postprocess_data 抛 `NotImplementedError`——异常类型不统一（轻微）。
- middle 在 `multiturn_sft_dataset` 不支持但文档宣称四策略——路径间能力不一致。

### 2.5 prompt_length vs response_length 校验不对称

- `prompt_length <= 0` 校验为非法；**`response_length` 不校验**——0 是 skip/rollout_skip.py 的合法"不限"语义，加校验会破坏现有配置（设计决策已记录）。
- 教训：**"必须为正"的校验不能机械套用到语义含"0=不限"的字段**；按字段语义单独判定。

### 2.6 cap 极小值下的下游语义（rollout 特有）

- cap=1 时 left/right/middle 分别保留 末/首/首 token——保留的 token 是否携带任务关键信息（开头指令 vs 近 EOS token）直接影响 agent 首轮生成质量；配置者通常意识不到该选择。
- 论文语料（126 身份）无任何一篇讨论此微观类问题（专项 G4 结论）；与系统级边界留白（ReSpec 消费量、RolloutPipe 双分母、Relax off-by-one，专项 G7）构成"边界条件类"缺口的 token 级与系统级两端。

## 三、证据与验证范围

- fork head `71fd3562` 文件逐行核对：`_cap_text_prompt_length` 实现、`RolloutConfig.__post_init__` 校验、`test_cap_text_prompt_length.py` 全部用例（含 caps 1–6 显式期望、两个负向控制：left cap=0 空尾、middle cap=1 不超长）。
- 本地 veRL 快照 `c0e7551` 对照：修复前无条件左截断、`postprocess_data` floor 语义、`multiturn_sft_dataset.py:117` assert、`rl_dataset.py:118` 只读不执行。
- 历史验证（已入 candidate-8281 与记忆）：CPU 单测 18 passed（含负向控制）＋ 4090-1 GPU e2e 正向 2 step 全过＋error fail-loudly（2026-10-25 复验，fork head 与服务器 4 文件 SHA256 对齐）。
- **局限**：本备忘不做跨路径统一提案（如 postprocess_data cap=1 对齐、rl_dataset 空契约修复），不发布评论/PR（铁律）；PR #8313 ④ 回复草稿 v2 仍待用户逐字审定。

## 四、后续可选动作（均待用户决策）

1. 以本矩阵为模板，把"边界条件检查清单"沉淀为 verl 本地工具/测试夹具（可选）。
2. 若用户决定推进跨路径统一：候选点=postprocess_data cap=1 语义、rl_dataset truncation 空契约、multiturn_sft middle 支持（均属上游写操作，铁律逐项授权）。
3. 专项侧：G4 状态由"缺口"升为"已有本地备忘"，matrix 无新增文献。
