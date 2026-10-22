# verl：超长提示词截断未遵循配置策略

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 超长提示词截断未遵循配置策略及候选的最小长度边界。[Issue #8281](https://github.com/verl-project/verl/issues/8281) · [PR #8313](https://github.com/verl-project/verl/pull/8313) · [旧PR #4600](https://github.com/verl-project/verl/pull/4600)

**【类型】** 漏洞修复

**【价值评估】** 7 / 10 — 有明确行为差异和本地逻辑验证，但接口契约与完整环境验证仍待确认。

**【进展】** 2026-10-22：[PR #8313](https://github.com/verl-project/verl/pull/8313) 已由用户关闭（待 review）；middle/cap=1 的 `-0` 反例已在用户风格基础上修复并推送到 fork 分支 `fix-8281-truncation-policy`（head `6db4b38d`，远端已确认）。最新分支 CPU 单测 **12 passed**（含 cap=1 回归），GPU e2e 冒烟在 4090-1 复验通过（正向 2 step + `error` 策略 fail-loudly）。重开 PR 与 issue 回复仍需用户决定。

**【大白话讲明白】**

原实现不管配置都删除提示词开头，可能丢掉关键指令；现在已有按配置处理的PR。独立核对发现一个遗漏：要求只保留1个token时，头尾截断反而把10个输入变成11个。普通长度测试通过不代表所有正长度都满足上限；这次只记录复现证据，没有替负责人修改或发布。

**【What–Why–How】**

- **What：** Agent Loop 的共享长度限制接口是否遵循 error/left/right/middle 四种截断配置。
- **Why：** 共享函数固定取最后若干 token，使不同配置得到同样的左截断结果。
- **How：** 固定PR与基线原方法，各24组CPU输入观察；明确cap=1反例与原测试覆盖缺口，不把配置容器占位/方法隔离算作完整AgentLoop或训练验收。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 上游 issue：[#8281 [rollout] Shared prompt cap ignores documented data.truncation policies](https://github.com/verl-project/verl/issues/8281)
> 发现时间：2026-10-20（issue 创建于 2026-10-19）
> 核验代码版本：verl main `ddf70072`（v0.8.0-546-gddf70072，2026-10-20 最新）

## 状态

**2026-10-22当前状态优先**：已存在用户账号Jackie2049发布的[PR #8313](https://github.com/verl-project/verl/pull/8313)，固定head `80ee82bd3b6c9409e5355ec951e9d95c6afaae07`，目标base与merge-base均 `954bd28f50fe8782b09cfc1a5480e709c3ffcae5`，两文件129增3删。本轮只读核验，不推定此前发布授权来源，也不是M1本轮创建/更新的PR。下方“仅本地草稿/待发布”的描述是旧日期记录；当前新增证据见下一节，不能继续将旧middle长度检查外推为所有正上限正确。

**2026-10-21复核优先说明**：API确认issue open/0评论，#4600 open/未合并；timeline明确有#4600，编号search虽返回0且incomplete_results=false，也不能说“0关联PR”。范围关键词truncation返回37个open PR，仅作线索池，未逐一做实现等价性判断；不声明全仓库无人覆盖。本轮不改本地补丁/用户分支，不自动推进发布。

最新main `4a9dc911d57062d76988e0fd1f2468b5e7178ce3` 的共享cap仍为超长时warning并取后缀，未按data.truncation分流；这是只读源码核验，不是新完整环境测试。下方竞争窗口和草稿验证按原记录日期理解。

**analysis-done + CPU 验证完成**（本地提取逻辑验证，非 GPU 实测）。reporter 未自修 PR（0 引用 PR），但 reporter 深挖后把「契约决策」抛给社区。发布评论/PR 需用户授权。

**（2026-10-20 补充）PR 草稿已本地备好**：源修复 + 真实单测已落在 `/Users/jackiemac/workspace/verl` 分支 `fix-8281-truncation-policy`（commit `7ace28db`，基于上游 main `ddf70072`），补丁存 [patches/8281-truncation-policy.patch](patches/8281-truncation-policy.patch)。验证：真源方法 verbatim exec 32 断言全过（default/error/left/right/middle/unknown/short/middle 长度不变量）；两文件 py_compile 通过。**局限**：本机无 verl 全量依赖（ray/transformers/torch），真单测未在本机跑，需在 verl CI 验证；补丁已可应用用户 fork。

## 当前候选的最小正上限反例

在固定head的[原_cap_text_prompt_length](https://github.com/verl-project/verl/blob/80ee82bd3b6c9409e5355ec951e9d95c6afaae07/verl/experimental/agent_loop/agent_loop.py#L467)中，middle分支为 `head=(prompt_length+1)//2`，再拼接 `prompt_ids[:head] + prompt_ids[-(prompt_length-head):]`。当上限1时，head=1、尾段预算0；Python的 `-0 == 0`，`prompt_ids[-0:]`是整个列表，不是空尾段。因此输入`list(range(10))`会输出`[0,0,1,2,3,4,5,6,7,8,9]`，长度11。基线同一输入只返回`[9]`；这是候选在方法边界上新增的上限违例，不只是原策略差异。

| 输入10 token的上限 | 候选left | 候选right | 候选middle |
|---|---|---|---|
| 1 | `[9]` | `[0]` | `[0,0,1,2,3,4,5,6,7,8,9]`，违例 |
| 2 | `[8,9]` | `[0,1]` | `[0,9]` |
| 5 | `[5,6,7,8,9]` | `[0,1,2,3,4]` | `[0,1,2,8,9]` |

Python3.12.13执行固定Git对象内原方法全体，未改函数体/切片/异常；仅以真实标准库logger、SimpleNamespace提供prompt_length、普通dict提供data_config.get，不运行AgentLoop构造。基线与候选各24组观察：正上限1至6×left/right/middle共18组，以及默认/error/未知banana各1个超长和1个等长输入，共6组。候选超长三种拒绝路径抛ValueError，等长路径原样返回同一列表（未知策略也不在等长时校验）；输入列表没有被原地修改。这里“观察符合已记录结果”包括上述违例，不是宣称24组上限断言全过。

原[97行测试](https://github.com/verl-project/verl/blob/80ee82bd3b6c9409e5355ec951e9d95c6afaae07/tests/experimental/agent_loop/test_cap_text_prompt_length.py)的11项参数例中，middle只测4/5/6，未覆盖1；本机缺OmegaConf/Ray，原测试仅AST/内存compile，没有执行pytest，也没有用假OmegaConf冒充原fixture。RolloutConfig的prompt_length为int、默认512，所读__post_init__没有至少2的限制；本轮未构造完整配置/运行tokenizer或证明训练中此极小上限的实际可达性。原ct_build_initial_tokens确实调用该cap，多模态超长仍先抛错，不把文本反例外推到多模态特征或实际模型故障。

最小复查（Python3.9+；在含固定对象的verl仓库中）：

```bash
GIT_NO_LAZY_FETCH=1 python - <<'PY'
import ast, logging, subprocess
from types import SimpleNamespace
rev = "80ee82bd3b6c9409e5355ec951e9d95c6afaae07"
source = subprocess.check_output(["git", "show", rev + ":verl/experimental/agent_loop/agent_loop.py"])
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentLoopBase")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_cap_text_prompt_length")
logger = logging.getLogger("source-cap-review")
logger.addHandler(logging.NullHandler())
logger.propagate = False
ns = {"logger": logger}
exec(compile(ast.Module(body=[method], type_ignores=[]), "<original-cap>", "exec"), ns)
obj = SimpleNamespace(rollout_config=SimpleNamespace(prompt_length=1), data_config={"truncation": "middle"})
out = ns[method.name](obj, list(range(10)))
assert out == [0] + list(range(10)) and len(out) == 11
print(out, "length", len(out), "exceeds cap=1")
PY
```

SHA256：agent_loop.py基线 `e931a80ca4740c18ad8b6b2f9c01e8cd78d1c42c8fe2a491fe21845ecc69cb57`，候选 `598472035bff9bad1a3f270bea91321037d3494f97288fba630e1686bc3b4264`；原测试 `1f6789cc9aff0446553f468b0d167e88a712c5bfef9e60f8e8169b2f9afb4fcd`。本轮只保留诊断证据，不修改下方历史方案、已有patch/验证脚本、用户分支或上游PR；无需GPU/NPU补证，不新增M5任务。后续修正及对外review由负责人/用户决定。

## 历史问题

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

> 历史方案原样保留，middle在cap=1同样有上面的违例，不能作为覆盖所有正上限的完整修复使用。旧验证只测指定长度，不据“ALL CHECKS PASSED”外推。

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

## Follow-up：middle/cap=1 的 -0 反例修复（2026-10-22，已完成并验证）

在 PR #8313（head `80ee82bd`）当前源码上核验发现并修复了 middle/cap=1 的 `-0` 反例。**用户 2026-10-22 先 review 并改写了代码风格**（commit `872bc281`，变量 `truncation`→`truncation_policy`、docstring 契约说明、日志措辞 left/right/middle-truncating），随后本会话**在用户风格基础上补上漏洞**（commit `6db4b38d`，DCO 签名，落 fork 分支 `fix-8281-truncation-policy`）。

- **反例**：`middle` 分支 `prompt_ids[:head] + prompt_ids[-(prompt_length-head):]` 在 `prompt_length=1` 时 `head=1`、尾段预算 0，Python `-0==0` 使 `prompt_ids[-0:]` 取整个列表——10 token 输入变成 11 token 输出（复现：`[0,0,1,2,3,4,5,6,7,8,9]`），违反上限。原测试 middle 只覆盖 cap 4/5/6。
- **修复**（遵循用户风格，`tail_length` 命名与 `truncation_policy` 一致）：尾段改用正向切片 `prompt_ids[len(prompt_ids) - tail_length:]`（尾段预算 0 时 `list[len:]` 为空列表，天然正确）。已验证 cap=1..7 全部遵守上限、head+tail 语义与原始 cap≥2 完全一致；left/right/error 不受影响。
- **CPU 测试（12 项全过）**：原 11 项 + 新增 `test_middle_cap1_does_not_extend_prompt`，在服务器 4090-1 `verl-e2e` 环境重跑 **12 passed**（5.05s）；**对照验证**——还原旧 middle 行后 cap=1 测试失败（证明测试有效），恢复修复后通过。
- **状态**：PR #8313 已由用户主动关闭（「Closing: not yet reviewed/approved. Will reopen when the fix has been reviewed and approved」，10-22 07:07Z）；修复已 commit 并**推送到 fork 分支** `fix-8281-truncation-policy`（`872bc281` 用户风格 + `6db4b38d` 漏洞修复，远端 head 已确认 `6db4b38d`）。**最新分支复验（2026-10-22）**：CPU 单测 12 passed；4090-1 上 GPU e2e 正向 2 step 全过 + `error` 策略 fail-loudly，见 [GPU 冒烟记录](../../experiments/verl-8281-gpu-smoke/readme.md#2026-10-22-最新分支复验872bc281--6db4b38d含-middlecap1-修复)。待用户决定重开 PR 时机。
- **issue #8281 回复草稿**：[draft-8281-issue-reply.md](draft-8281-issue-reply.md)（技术确认 + 修复进展 + middle/cap=1 边界说明 + error 训练器行为契约问题），发布需用户授权。
