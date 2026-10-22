# verl #8281 GPU e2e 训练冒烟（4090-1，2026-10-22，M5）

## 目的

用户 2026-10-22 指派的真机验证任务：verl [PR #8313](https://github.com/verl-project/verl/pull/8313)（修复 [#8281](https://github.com/verl-project/verl/issues/8281) 超长 prompt 截断）在 GPU 上跑一次 e2e 训练冒烟，确认改动不破坏正常训练路径、`data.truncation='error'` 在真实训练中 fail-loudly。

## 环境（4090-1）

- GPU 0；conda `verl-e2e`：ray 2.59.0 + torch 2.11.0+cu128 + transformers 5.8.1 + vllm 0.22.1
- verl editable 源码 `/home/zxw/verl-pt-impr`（无 .git；PR diff 以 patch 形式应用，见 handoff-m5 部署说明）
- 模型：Qwen2.5-0.5B（HF 缓存）；数据：openai/gsm8k parquet → agent-loop 格式（`make_gsm8k_agent.py`，16 train / 4 test）

## 结果

1. **正向冒烟**（`smoke_8281.py`，`max_prompt_length=512`，prompt 实际 111-120 token 远低于 cap）：
   grpo + ToolAgentLoop + vllm 异步 rollout，`total_training_steps=2` **全部跑通**（step1 35.5s / step2 9.6s，num_turns=2，无截断无 raise），日志末尾 `SMOKE COMPLETED OK`（见 `pos_result.txt`）。正常训练路径未因改动退化。
2. **负向验证**（`max_prompt_length=32`，prompt 118/120 token 必超长）：默认 `error` 策略 **fail-loudly**——
   `ValueError: Prompt of 118 tokens exceeds rollout.prompt_length=32; configured data.truncation='error' which does not truncate. Increase rollout.prompt_length or set data.truncation to 'left'/'right'/'middle'.`（见 `neg_valueerror.txt`）。修复前该路径是静默左截断；训练终止为预期行为。
3. 单测：`tests/experimental/agent_loop/test_cap_text_prompt_length.py` 11 项服务器全过（5.08s）。

## 复现

- `make_gsm8k_agent.py`：从 openai/gsm8k parquet 生成 agent-loop 格式小数据集
- `smoke_8281.py`：正向冒烟配置（grpo + tool_agent + Qwen2.5-0.5B + 2 step）
- 负向：复制 smoke_8281.py 改 `max_prompt_length=32` 与输出目录（服务器 `/tmp/smoke_8281_neg.py`）

## 边界

- 服务器无法访问 GitHub（TLS 截断），PR diff 经 M5 本机 `gh pr diff` 只读获取并与本地 patch 核对一致后 patch 应用；非严格 git 分支 checkout，但实际改动行与 PR #8313 完全相同。
- `left/right/middle` 三策略的真实训练行为未逐一跑（单测已覆盖策略分支）；冒烟确认 error 契约即可。
- 未在 GitHub 上做任何写操作（取 diff 为只读），未改上游/PR/用户分支。
