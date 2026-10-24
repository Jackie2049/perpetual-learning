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

## 2026-10-22 最新分支复验（872bc281 + 6db4b38d，含 middle/cap=1 修复）

用户要求在最新分支上确保测试通过。fork 分支 `fix-8281-truncation-policy` 的远端 head 已确认是 `6db4b38d`（`git ls-remote` 与 GitHub API 一致；该 commit 已于 07:36Z 推送到 `Jackie2049/verl`）。PR #8313 本身仍为 CLOSED，head 停在 `80ee82bd`。

**发现并修复服务器源码树版本不一致**：`/home/zxw/verl-pt-impr` 原是 2026-10-14 的 `verl` 包快照，之前只把 10-21（`954bd28f`）版本的 `agent_loop.py` 打补丁进去；最新同步整文件后，`agent_loop.py` 会传 `build_initial_tokens(..., mm_inputs_out=...)`，而旧 `continuous_token.py` 没有该参数，冒烟在 agent loop 内报 `TypeError`（与截断修复无关）。已将完整 `verl/` 包从本地 `HEAD`（`954bd28f` + 两个 PR commit）同步到服务器，旧包备份为 `verl.bak-20261022`；同步后 `continuous_token.py` 含 `mm_inputs_out`，两文件 SHA256 与本地 HEAD 一致。

**最新版本结果（4090-1，`verl-e2e`，GPU 0，源码与 `6db4b38d` 一致）**：

1. CPU 单测：`test_cap_text_prompt_length.py` **12 passed**（5.58s，含 cap=1 回归）。
2. 正向 e2e 冒烟：grpo + ToolAgentLoop + Qwen2.5-0.5B + gsm8k agent 数据，`total_training_steps=2` **两 step 全过**（step1 30.8s / step2 4.5s，`num_turns=2`，`prompt_length 104-120` 未超 cap），日志末尾 `SMOKE COMPLETED OK`（`/tmp/smoke_8281_latest2.log`）。
3. 负向验证：`max_prompt_length=32` 时默认 `error` 策略仍 **fail-loudly**，`ValueError: Prompt of 118 tokens exceeds rollout.prompt_length=32; configured data.truncation='error' which does not truncate...`（`/tmp/smoke_8281_neg_latest.log`，进程以 1 退出为预期）。

**二次复跑（用户要求确保）**：核对三文件 SHA256 与上次一致后重跑——CPU `12 passed`（5.16s）；GPU 正向 2/2 step（37s，`SMOKE COMPLETED OK`，`/tmp/smoke_8281_verify_pos.log`）；GPU 负向 `ValueError: Prompt of 120 tokens exceeds rollout.prompt_length=32...`（exit 1 预期，`/tmp/smoke_8281_verify_neg.log`）。三项与首次结果一致。

**范围**：验证的是 fork 分支最新 commit 的内容；未重开 PR、未评论上游。`left/right/middle` 真实训练行为仍未逐一跑（单测覆盖）；cap=1 修复仅单测覆盖（cap=1 在实际训练中默认 512 下不可达）。服务器 `/home/zxw/verl-pt-impr` 的包替换为整包同步，若后续其他会话在该树做过本地改动需从 `verl.bak-20261022` 核对。

## 2026-10-25 最终 head（71fd3562）GPU 复验

PR 最终 head 为用户方 22:51 推送的 `71fd3562`（cap=1 保首 token `head=max(1, prompt_length//2)` + 测试改名回退）。服务器树此前停在中间态（`head=prompt_length//2`、测试仍为 `*_on_cpu.py`），本轮将 4 个受影响文件与 head 逐字节对齐后复验（SHA256 全匹配：agent_loop.py `3f03e9d2…`、test_cap_text_prompt_length.py `8995b5fc…`、rollout.py `d90cc2a6…`、test_rollout_config_on_cpu.py `479becf8…`；后两者本已一致，未改动）。

1. **CPU 单测：18 passed**（5.22s）——`test_cap_text_prompt_length.py` 16 项（caps 1-6 显式期望表）+ `test_rollout_config_on_cpu.py` 2 项（prompt_length 正数校验），与本地权威补丁验证结果一致。
2. **GPU 正向 e2e 冒烟：PASS**（GPU 0，grpo + ToolAgentLoop + Qwen2.5-0.5B + gsm8k agent 16/4，`total_training_steps=2`）——两 step 全过（step1 27.3s / step2 4.5s，`num_turns=2`，prompt_length 104-120 未超 cap 512），日志末尾 `SMOKE COMPLETED OK`，exit 0（`smoke_pos_final_head.log`）。日志尾部的 `DataLoader worker killed` 报错是 atexit 阶段 `dump_compile_times` 导入 tabulate 失败（`tabulate` 未安装）引发的关停噪音，发生在训练完成后，不影响结果。
3. **GPU 负向验证：PASS**——`max_prompt_length=32` 时默认 `error` 策略 fail-loudly，`_cap_text_prompt_length`（agent_loop.py:506）抛出精确预期的 `ValueError: Prompt of 120 tokens exceeds rollout.prompt_length=32; configured data.truncation='error' which does not truncate...`，0 步完成、无 `SMOKE COMPLETED OK`（`smoke_neg_final_head.log`）。本轮 driver 退出码 0（10-22 轮为 1）：v1 trainer 在全部 rollout 组失败后另报 `Sync replay buffer…no materializable trajectories` RuntimeError 收尾，错误传播路径差异不影响 fail-loudly 契约本身。

**范围**：验证的是 PR 最终 head `71fd3562` 的内容（服务器树 4 文件与 head 字节一致）；未在 GitHub 做任何写操作（取 diff/文件为只读），未改 PR、未发评论或推送 fork head。`left/right/middle` 真实训练行为仍未逐一跑（单测覆盖 16 项）；cap=1 在实际训练中默认 512 下不可达。
