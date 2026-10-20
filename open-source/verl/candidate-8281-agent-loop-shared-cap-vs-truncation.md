# candidate-8281-agent-loop-shared-cap-vs-truncation.md

> 上游 issue：[#8281 [rollout] Shared prompt cap ignores documented data.truncation policies](https://github.com/verl-project/verl/issues/8281)
> 发现时间：2026-10-20（issue 创建于 2026-10-19）
> 核验代码版本：verl main `ddf70072`（v0.8.0-546-gddf70072，2026-10-20 最新）

## 状态

**analysis-done + CPU 验证完成**（本地提取逻辑验证，非 GPU 实测）。reporter 未自修 PR（0 引用 PR），但 reporter 深挖后把「契约决策」抛给社区。发布评论/PR 需用户授权。

**（2026-10-20 补充）PR 草稿已本地备好**：源修复 + 真实单测已落在 `/Users/jackiemac/workspace/verl` 分支 `fix-8281-truncation-policy`（commit `7ace28db`，基于上游 main `ddf70072`），补丁存 [patches/8281-truncation-policy.patch](patches/8281-truncation-policy.patch)。验证：真源方法 verbatim exec 32 断言全过（default/error/left/right/middle/unknown/short/middle 长度不变量）；两文件 py_compile 通过。**局限**：本机无 verl 全量依赖（ray/transformers/torch），真单测未在本机跑，需在 verl CI 验证；补丁已可应用用户 fork。

## 问题

`AgentLoopBase._cap_text_prompt_length`（`verl/experimental/agent_loop/agent_loop.py:467-476`）对超长文本 prompt **无条件左截断**（保留后缀 `prompt_ids[-prompt_length:]`），忽略 `data.truncation` 策略（`error/left/right/middle`）。默认 `error` 时本应报错，却被静默截断，导致 prompt 开头指令被丢弃。

验证复现（本地提取，与 issue repro 一致）：prompt 10 token、cap 4，四种 policy 全部输出同一后缀 `[6,7,8,9]`：

```text
error [6,7,8,9]   left [6,7,8,9]   right [6,7,8,9]   middle [6,7,8,9]
```

## 根因（比 reporter 描述更广）

契约矛盾有两层：

1. **agent_loop 硬编码左截断**（`_cap_text_prompt_length`，agent_loop.py:467-476）——无视 `self.data_config.truncation`。
2. **RLHFDataset 从未执行 truncation 策略**（`verl/utils/dataset/rl_dataset.py:118` 只读取 `self.truncation`，全文无应用）——RL 路径（含 agent_loop）的唯一防线是 `filter_overlong_prompts=True`（默认，`rl_dataset.py:199` 过滤超长 prompt）。因此 `data.truncation` 在 RL/agent_loop 路径实际是**空契约**：既不被过滤逻辑引用，也不被截断逻辑引用。

文档契约（`docs/examples/config.rst:76-81`）宣称 `error/left/right/middle` 四种行为，与实际不符。

触发路径：`filter_overlong_prompts=False` 时，或渲染后 token 数（agent_loop 内 `ct_build_initial_tokens` 经 chat template 渲染）超过数据集估算时，超长 prompt 到达 `_cap_text_prompt_length` 被静默左截断。

相关背景：
- `#4759` 移除了数据集 `postprocess_data(..., truncation=...)` 调用和四向 `raw_prompt_ids` 处理，策略执行未带入 agent loop。
- 无条件 cap 来自 commit `caf9fee`。
- 多模态路径（agent_loop.py:383-391）已用 `ValueError` "fail loudly"（占位符与特征需 1:1 对齐不能切），文本路径却是静默截断——**行为不对称**。

## 修复方案（option 1：尊重文档策略）

在 `_cap_text_prompt_length` 内读取 `self.data_config.get("truncation", "error")` 并分流：

```python
def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
    prompt_length = self.rollout_config.prompt_length
    if len(prompt_ids) <= prompt_length:
        return prompt_ids
    truncation = self.data_config.get("truncation", "error")  # matches rl_dataset.py:118
    if truncation == "left":
        return prompt_ids[-prompt_length:]          # keep suffix (current behavior)
    if truncation == "right":
        return prompt_ids[:prompt_length]           # keep prefix
    if truncation == "middle":
        head = (prompt_length + 1) // 2             # keep head+tail
        return prompt_ids[:head] + prompt_ids[-(prompt_length - head):]
    # 'error' (default): fail loudly, mirroring the multimodal ValueError path
    raise ValueError(
        f"Prompt of {len(prompt_ids)} tokens exceeds rollout.prompt_length={prompt_length}; "
        f"configured data.truncation='error'. Increase rollout.prompt_length or set a "
        f"non-'error' data.truncation."
    )
```

语义参考：`multiturn_sft_dataset.py:380-393` 的 error/left/right 实现；`middle` 按文档「keeps head and tail」实现（`multiturn_sft_dataset` 实际不支持 middle，agent_loop 若补需同时考虑文档一致性）。

## 验证

`experiments/scripts/verify_agentloop_truncation_policy.py`（CPU，本地独立提取逻辑，无需 verl 依赖）：

- **复现**：四种 policy 全输出同一后缀 `[6,7,8,9]` ✓（BUG reproduced）
- **修复后**：left→`[6,7,8,9]`、right→`[0,1,2,3]`、middle→`[0,1,8,9]`、error→raise ValueError ✓
- 短 prompt（未超 cap）在全部 policy 下原样返回 ✓
- middle 长度不变量（cap 5/6 保持输出长度==cap）✓

**局限**：本地独立逻辑验证，未在真实 verl 环境跑 issue 的 `AgentLoopBase` 构造复现（需 verl 依赖 + omegaconf）；未覆盖训练器错误处理路径（见下）。

## 决策点（影响 PR 设计）

- **`error` 抛异常的训练器行为**：reporter 指出 `error` raise 后 V1 按 prompt group 捕获失败（`agent_loop_tq.py:136-153`），async 会 refill 失败组、sync refill 需 opt-in，sync 批次全空会报错。需决定：raise 是否接受此行为，还是仅警告 + 跳过该组。
- **契约方向**：倾向 option 1（尊重文档策略），与多模态路径的 fail-loudly 对齐；option 2（声明左截断为契约、改文档）风险低但延续静默丢指令的问题。
- **`data.truncation` 空契约**：RLHFDataset 未执行 truncation 是更大的一致性问题，可作 follow-up，本候选聚焦 agent_loop。

## 竞争与窗口

- issue 创建 2026-10-19，0 评论（2026-10-20 核验）。
- **reporter（Jackmeson1）深挖充分，明确把 option 1/2 决策抛给社区，未自修**——与「reporter 深挖 ≈ reporter 已开 PR」的 4/4 模式不同，有贡献空间。
- **（2026-10-20 新增核验）存在关联旧 PR #4600**：reporter 在创建 #8281 后 6 秒内即把 #8281 交叉链接到 PR [#4600](https://github.com/verl-project/verl/pull/4600)（fabianlim，`[SingleTurnAgentLoop] fix: Take care of truncation`），并在 #4600 评论里将 #8281 表述为「current-main 契约讨论」。
  - #4600 是同方向（四策略 left/right/middle/error）的**旧实现**：创建于 2025-12-18，早于 Continuous Token 重构（#6779/#6804 引入 `_cap_text_prompt_length`），`mergeable_state=dirty`（与 main 冲突）、引用旧 issue #2069、维护者 2026-09-21 请求 review 后仍未合并——**实际已陈旧、非当前修复点**。
  - 当前 `SingleTurnAgentLoop.run` 已走 `ct_build_initial_tokens → _cap_text_prompt_length` 共享路径，因此 #4600 在 `run` 里再加截断会与基类 CT 路径冗余/冲突；**正确的修复点是共享的 `_cap_text_prompt_length`**（本候选的目标），覆盖 single_turn / tool_agent / 所有 CT 构建 loop，比 #4600 更通用且贴合当前架构。
  - 含义：贡献空间仍存在，但不是「0 竞争」；reporter 把 #8281 定位为「契约讨论」而非「#4600 重复」，评论形态比 PR 更契合。#4600 是陈旧参考，不是真正的竞争者。
- verl 认领窗口 0-2 天，已过约 1 天，若要提交需尽快授权。

## 建议发布形态

可选：

1. **comment（推荐，契合 reporter 的「契约讨论」定位）**：给 reporter 的技术回复——确认复现、指出 RLHFDataset 层 truncation 空契约、说明正确的修复点是共享的 `_cap_text_prompt_length`（而非 #4600 的旧 `run` 内逻辑，因当前架构已走 CT 共享路径）、给出 option 1 的具体 patch + error 路径训练器行为分析。低风险、对上游有价值，且不与陈旧 #4600 冲突。
2. **PR**：option 1 完整实现 + 单测——**已本地备好**（源修复 + `test_cap_text_prompt_length.py`，patch 见上）。发布前需：①确认 error 行为与 maintainer 方向（见决策点）；②在 PR 描述中说明为何不重复现有 #4600（按 verl 贡献政策「PR 描述须说明为何不重复现有 PR」）——要点：#4600 预 CT 重构、dirty、非当前共享修复点，本 PR 覆盖 current-main 的 `_cap_text_prompt_length`。

不发布评论则本地沉淀作 review/跟进依据。
